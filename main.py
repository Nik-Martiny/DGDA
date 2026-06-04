"""Build and visualize a 200-device dynamic communication network simulation.

The topology models ten routers split into two five-router rings. Router A in the
first ring bridges to router F in the second ring. Each router has one directly
attached switch. Internal servers attach only to Router F's switch, web/edge
servers attach only to Router G's switch, and client/IoT devices attach to the
remaining switches.

The simulation exposes a 500-window discrete-time dynamic graph. Each window is a
NetworkX snapshot with stable infrastructure, normal endpoint churn, and transient
normal communication edges. Windows 251-350 are explicitly marked as the attack
injection window so future work can mutate those snapshots with attack traffic
while all other phases remain pure normal traffic.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


TOTAL_DEVICES = 200
DEVICE_COUNTS = {
    "client_workstation": 80,
    "internal_server": 40,
    "web_edge_server": 30,
    "router": 10,
    "switch": 10,
    "iot_peripheral": 30,
}

ROUTER_NAMES = tuple("ABCDEFGHIJ")
SWITCH_NAMES = tuple(f"switch_{router}" for router in ROUTER_NAMES)
INTERNAL_SERVER_ROUTER = "F"
WEB_EDGE_SERVER_ROUTER = "G"
CLIENT_IOT_ROUTERS = ROUTER_NAMES[:5] + ROUTER_NAMES[7:]
CATEGORY_ROUTER_ASSIGNMENTS = {
    "client_workstation": CLIENT_IOT_ROUTERS,
    "internal_server": (INTERNAL_SERVER_ROUTER,),
    "web_edge_server": (WEB_EDGE_SERVER_ROUTER,),
    "iot_peripheral": CLIENT_IOT_ROUTERS,
}
RNG_SEED = 42
TOTAL_TIME_WINDOWS = 500


@dataclass(frozen=True)
class TimingPhase:
    """A named inclusive range of discrete graph time windows."""

    name: str
    start_window: int
    end_window: int
    description: str
    normal_traffic_only: bool
    attack_injection_allowed: bool = False

    def contains(self, window: int) -> bool:
        """Return whether ``window`` is inside this phase's inclusive range."""
        return self.start_window <= window <= self.end_window


TIMING_PHASES = (
    TimingPhase(
        name="baseline",
        start_window=1,
        end_window=150,
        description="Training baseline: pure normal traffic only.",
        normal_traffic_only=True,
    ),
    TimingPhase(
        name="pre_attack",
        start_window=151,
        end_window=250,
        description=(
            "Pre-attack validation/calibration: normal traffic only for false-alarm "
            "checks and CUSUM/Page-Hinkley threshold calibration."
        ),
        normal_traffic_only=True,
    ),
    TimingPhase(
        name="attack",
        start_window=251,
        end_window=350,
        description=(
            "Attack interval: normal traffic plus a reserved injection hook for "
            "future attack graph mutations."
        ),
        normal_traffic_only=False,
        attack_injection_allowed=True,
    ),
    TimingPhase(
        name="recovery",
        start_window=351,
        end_window=500,
        description=(
            "Recovery: normal traffic returns so detectors can settle back to "
            "baseline and false-positive rates can be measured."
        ),
        normal_traffic_only=True,
    ),
)


ROUTER_RING_EDGES = (
    ("router_A", "router_B"),
    ("router_B", "router_C"),
    ("router_C", "router_D"),
    ("router_D", "router_E"),
    ("router_E", "router_A"),  # Complete the loop

    ("router_A", "router_F"),  # Group A & B link

    ("router_F", "router_G"),
    ("router_G", "router_H"),
    ("router_H", "router_I"),
    ("router_I", "router_J"),
    ("router_J", "router_F"),  # Complete the loop
)


CATEGORY_DISPLAY_NAMES = {
    "client_workstation": "Client Workstations",
    "internal_server": "Internal Servers",
    "web_edge_server": "Web/Edge Servers",
    "router": "Routers",
    "switch": "Switches",
    "iot_peripheral": "IoT/Peripheral Devices",
}

CATEGORY_COLORS = {
    "client_workstation": "#4E79A7",
    "internal_server": "#59A14F",
    "web_edge_server": "#F28E2B",
    "router": "#E15759",
    "switch": "#A1B213",
    "iot_peripheral": "#B07AA1",
}


ENDPOINT_CATEGORIES = (
    "client_workstation",
    "internal_server",
    "web_edge_server",
    "iot_peripheral",
)


ENDPOINT_PREFIXES = {
    "client_workstation": "client",
    "internal_server": "internal-server",
    "web_edge_server": "web-edge-server",
    "iot_peripheral": "iot-peripheral",
}

INFRASTRUCTURE_CATEGORIES = ("router", "switch")

# Probability that each endpoint category is active in a normal time window. This
# creates realistic node add/remove churn while keeping routers and switches stable.
ENDPOINT_UP_PROBABILITIES = {
    "client_workstation": 0.92,
    "internal_server": 0.99,
    "web_edge_server": 0.99,
    "iot_peripheral": 0.84,
}

# Normal edge churn rules. Each tuple is (source category, target category,
# minimum edges per window, maximum edges per window, traffic label).
NORMAL_TRAFFIC_RULES = (
    ("client_workstation", "internal_server", 70, 115, "client_to_internal"),
    ("client_workstation", "web_edge_server", 55, 95, "client_to_web_edge"),
    ("iot_peripheral", "internal_server", 20, 45, "iot_to_internal"),
    ("internal_server", "web_edge_server", 15, 35, "server_to_edge"),
    ("client_workstation", "client_workstation", 8, 18, "client_peer"),
)

AttackInjector = Callable[[nx.Graph, np.random.Generator, TimingPhase], None]


def create_network(seed: int = RNG_SEED) -> nx.Graph:
    """Create the deterministic 200-device physical network topology."""
    graph = nx.Graph(name="Dynamic Graph Device Network")
    rng = np.random.default_rng(seed)

    # Add the router/switch backbone
    _add_router_and_switch_layer(graph)

    # Add the client/server/IoT device nodes
    _add_endpoint_devices(graph, rng)

    # Make sure everything is where it is meant to be
    _validate_network(graph)

    return graph


def create_dynamic_graph_windows(
    seed: int = RNG_SEED,
    total_windows: int = TOTAL_TIME_WINDOWS,
    attack_injector: AttackInjector | None = None,
) -> list[nx.Graph]:
    """Create the full discrete-time dynamic graph as 500 graph snapshots.

    Every snapshot contains the stable physical infrastructure, a deterministic
    sample of currently active endpoints, and transient normal communication
    edges. The optional ``attack_injector`` is called only during the attack phase
    so future experiments can inject attack-specific nodes or edges without
    leaking anomalies into baseline, pre-attack, or recovery windows.
    """
    _validate_timing_phases(total_windows)
    base_graph = create_network(seed)
    rng = np.random.default_rng(seed)
    windows = []

    for window in range(1, total_windows + 1):
        phase = phase_for_window(window)
        snapshot = _create_normal_window_snapshot(base_graph, window, phase, rng)

        if phase.attack_injection_allowed and attack_injector is not None:
            attack_injector(snapshot, rng, phase)
            snapshot.graph["attack_injected"] = True
        else:
            snapshot.graph["attack_injected"] = False

        _validate_window_snapshot(snapshot, phase)
        windows.append(snapshot)

    return windows


def phase_for_window(window: int) -> TimingPhase:
    """Return the configured timing phase for a one-indexed window number."""
    for phase in TIMING_PHASES:
        if phase.contains(window):
            return phase
    raise ValueError(f"Window {window} is outside the configured timing phases.")


def _validate_timing_phases(total_windows: int) -> None:
    """Validate that timing phases exactly cover the dynamic graph timeline."""
    expected_window = 1
    for phase in TIMING_PHASES:
        if phase.start_window != expected_window:
            raise ValueError(
                f"Phase {phase.name} starts at {phase.start_window}; "
                f"expected {expected_window}."
            )
        if phase.end_window < phase.start_window:
            raise ValueError(f"Phase {phase.name} has an invalid window range.")
        expected_window = phase.end_window + 1

    if expected_window - 1 != total_windows:
        raise ValueError(
            f"Timing phases cover {expected_window - 1} windows, expected {total_windows}."
        )


def _create_normal_window_snapshot(
    base_graph: nx.Graph,
    window: int,
    phase: TimingPhase,
    rng: np.random.Generator,
) -> nx.Graph:
    """Create one normal-traffic snapshot with node and edge churn."""
    active_nodes = _active_nodes_for_window(base_graph, rng)
    snapshot = base_graph.subgraph(active_nodes).copy()
    snapshot.graph.update(
        name=f"Dynamic Graph Device Network - window {window:03d}",
        window=window,
        phase=phase.name,
        phase_description=phase.description,
        normal_traffic_only=phase.normal_traffic_only,
        attack_injection_allowed=phase.attack_injection_allowed,
        ground_truth_attack_phase=phase.attack_injection_allowed,
        ground_truth_label="attack" if phase.attack_injection_allowed else "normal",
        traffic_mode="normal_with_attack_slot"
        if phase.attack_injection_allowed
        else "normal_only",
    )

    _add_normal_traffic_edges(snapshot, rng, window)
    return snapshot


def _active_nodes_for_window(base_graph: nx.Graph, rng: np.random.Generator) -> set[str]:
    """Select active nodes for a normal time window."""
    active_nodes = set()
    for node, attributes in base_graph.nodes(data=True):
        category = attributes["category"]
        if category in INFRASTRUCTURE_CATEGORIES:
            active_nodes.add(node)
            continue

        if rng.random() <= ENDPOINT_UP_PROBABILITIES[category]:
            active_nodes.add(node)

    return active_nodes


def _add_normal_traffic_edges(
    graph: nx.Graph,
    rng: np.random.Generator,
    window: int,
) -> None:
    """Add transient normal communication edges to the current snapshot."""
    nodes_by_category = _nodes_by_category(graph)

    for source_category, target_category, min_edges, max_edges, traffic_label in NORMAL_TRAFFIC_RULES:
        candidate_sources = nodes_by_category[source_category]
        candidate_targets = nodes_by_category[target_category]
        if not candidate_sources or not candidate_targets:
            continue

        target_edges = int(rng.integers(min_edges, max_edges + 1))
        attempts = 0
        added = 0
        max_attempts = target_edges * 10

        while added < target_edges and attempts < max_attempts:
            attempts += 1
            source = str(rng.choice(candidate_sources))
            target = str(rng.choice(candidate_targets))
            if source == target or graph.has_edge(source, target):
                continue

            graph.add_edge(
                source,
                target,
                link_type="normal_traffic",
                traffic_profile=traffic_label,
                transient=True,
                window=window,
            )
            added += 1


def _nodes_by_category(graph: nx.Graph) -> dict[str, list[str]]:
    """Group graph nodes by device category."""
    nodes_by_category: dict[str, list[str]] = {category: [] for category in DEVICE_COUNTS}
    for node, attributes in graph.nodes(data=True):
        nodes_by_category[attributes["category"]].append(node)
    return nodes_by_category


def _add_router_and_switch_layer(graph: nx.Graph) -> None:
    """Add ten routers, ten switches, two router rings, and router-switch links."""
    for router_name in ROUTER_NAMES:
        router_id = f"router_{router_name}"
        switch_id = f"switch_{router_name}"
        network_name = _network_name_for_router(router_name)
        graph.add_node(
            router_id,
            category="router",
            device_type="router",
            network=network_name,
            label=f"Router {router_name}",
        )
        graph.add_node(
            switch_id,
            category="switch",
            device_type="switch",
            network=network_name,
            label=f"Switch {router_name}",
        )
        graph.add_edge(router_id, switch_id, link_type="router_to_switch")

    graph.add_edges_from(
        (source, target, {"link_type": "router_backbone"})
        for source, target in ROUTER_RING_EDGES
    )


def _add_endpoint_devices(graph: nx.Graph, rng: np.random.Generator) -> None:
    """Attach endpoint devices to switches in their appropriate access network."""
    for category in ENDPOINT_CATEGORIES:
        count = DEVICE_COUNTS[category]
        prefix = ENDPOINT_PREFIXES[category]
        switch_ids = np.array(_switch_ids_for_category(category))
        network_name = _network_name_for_category(category)

        # Iterate through the selected node count number of times
        for index in range(1, count + 1):
            # Create a prefix for the current node to use as the node identifier
            device_id = f"{prefix}_{index:03d}"

            # Select a random switch from the correct switch IDs to assign the node
            attached_switch = str(rng.choice(switch_ids))

            # Add the node to the graph with the device_id, and other attributes
            graph.add_node(
                device_id,
                category=category,
                device_type=category,
                network=network_name,
                label=f"{CATEGORY_DISPLAY_NAMES[category]} {index}",
            )

            # Add an edge for the current node and its chosen switch
            graph.add_edge(device_id, attached_switch, link_type="access")


def _switch_ids_for_category(category: str) -> tuple[str, ...]:
    """Return the exact switch pool used by each endpoint category."""
    return tuple(
        f"switch_{router_name}"
        for router_name in CATEGORY_ROUTER_ASSIGNMENTS[category]
    )


def _network_name_for_category(category: str) -> str:
    """Label endpoint categories by their logical second-ring role."""
    if category == "internal_server":
        return "internal_server_network"
    if category == "web_edge_server":
        return "web_edge_server_network"
    return "client_iot_network"


def _network_name_for_router(router_name: str) -> str:
    """Map routers to their logical access role in the simulated topology."""
    if router_name == INTERNAL_SERVER_ROUTER:
        return "internal_server_network"
    if router_name == WEB_EDGE_SERVER_ROUTER:
        return "web_edge_server_network"
    if router_name in CLIENT_IOT_ROUTERS:
        return "client_iot_network"
    return "backbone_network"


def _validate_network(graph: nx.Graph) -> None:
    """Fail fast if the constructed graph drifts from the requested structure."""

    # Check if there are the correct number of nodes in the graph
    if graph.number_of_nodes() != TOTAL_DEVICES:
        raise ValueError(
            f"Expected {TOTAL_DEVICES} devices, found {graph.number_of_nodes()}."
        )
    # Check that the number of individual nodes is the same as the expected number of nodes for each device category
    actual_counts = Counter(nx.get_node_attributes(graph, "category").values())
    if actual_counts != DEVICE_COUNTS:
        raise ValueError(
            f"Device category counts do not match: {dict(actual_counts)} != {DEVICE_COUNTS}."
        )

    # Check for any edges that are missing between core routers that make up the ring topology
    missing_backbone_edges = [
        (source, target)
        for source, target in ROUTER_RING_EDGES
        if not graph.has_edge(source, target)
    ]
    if missing_backbone_edges:
        raise ValueError(f"Missing router ring/bridge edges: {missing_backbone_edges}")

    # Check that each router has its own switch edge in the graph
    for router_name in ROUTER_NAMES:
        router_id = f"router_{router_name}"
        switch_id = f"switch_{router_name}"
        if not graph.has_edge(router_id, switch_id):
            raise ValueError(f"Missing router-switch edge: {(router_id, switch_id)}")

    # Check that all endpoint nodes (clients, servers, IoT devices) are where they are meant to be and not connected to some other network
    misplaced_endpoints = []
    for node, attributes in graph.nodes(data=True):
        category = attributes["category"]
        if category == "router" or category == "switch":
            continue

        allowed_switches = set(_switch_ids_for_category(category))
        access_switches = [
            neighbor
            for neighbor in graph.neighbors(node)
            if graph.nodes[neighbor]["device_type"] == "switch"
        ]
        if len(access_switches) != 1 or access_switches[0] not in allowed_switches:
            misplaced_endpoints.append((node, tuple(access_switches), category))

    if misplaced_endpoints:
        raise ValueError(
            "Endpoint nodes must attach only to their assigned router switches: "
            f"{misplaced_endpoints}"
        )


def _validate_window_snapshot(graph: nx.Graph, phase: TimingPhase) -> None:
    """Validate dynamic-window metadata and allowed traffic labels."""
    if graph.graph["phase"] != phase.name:
        raise ValueError(f"Window phase metadata drifted from {phase.name}.")

    if phase.normal_traffic_only and graph.graph["attack_injected"]:
        raise ValueError(f"Attack traffic leaked into the {phase.name} phase.")

    expected_link_types = {"router_to_switch", "router_backbone", "access", "normal_traffic"}
    unexpected_edges = [
        (source, target, attributes.get("link_type"))
        for source, target, attributes in graph.edges(data=True)
        if attributes.get("link_type") not in expected_link_types
    ]
    if unexpected_edges and not phase.attack_injection_allowed:
        raise ValueError(
            f"Unexpected edge types in normal-only {phase.name} phase: {unexpected_edges}"
        )

    if not nx.is_connected(graph):
        raise ValueError(f"Window {graph.graph['window']} is not connected.")


def draw_network(graph: nx.Graph, output_path: str | Path = "network_topology.png") -> None:
    """Render a network snapshot to a PNG image."""
    output_path = Path(output_path)
    positions = nx.spring_layout(graph, seed=RNG_SEED)
    node_colors = [CATEGORY_COLORS[graph.nodes[node]["category"]] for node in graph.nodes]
    node_sizes = []
    for node in graph.nodes:
        device_type = graph.nodes[node]["device_type"]
        if device_type == "router":
            node_sizes.append(620)
        elif device_type == "switch":
            node_sizes.append(360)
        else:
            node_sizes.append(70)

    plt.figure(figsize=(18, 14))
    nx.draw_networkx_edges(graph, positions, alpha=0.28, width=0.8)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.5,
        edgecolors="white",
    )

    infrastructure_labels = {
        node: graph.nodes[node]["label"].replace("Router ", "R").replace("Switch ", "S")
        for node in graph.nodes
        if graph.nodes[node]["category"] == "router" or graph.nodes[node]["category"] == "switch"
    }

    nx.draw_networkx_labels(graph, positions, labels=infrastructure_labels, font_size=8)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=CATEGORY_DISPLAY_NAMES[category],
            markerfacecolor=color,
            markersize=10,
        )
        for category, color in CATEGORY_COLORS.items()
    ]

    title = "200-Device Communication Network Simulation"
    if "window" in graph.graph:
        title = f"{title} - Window {graph.graph['window']} ({graph.graph['phase']})"

    plt.legend(handles=legend_handles, loc="upper right")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_summary(graph: nx.Graph) -> None:
    """Print a concise summary for command-line use."""
    category_counts = Counter(nx.get_node_attributes(graph, "category").values())
    print(f"Created {graph.number_of_nodes()} devices and {graph.number_of_edges()} links.")
    for category, expected_count in DEVICE_COUNTS.items():
        print(
            f"{CATEGORY_DISPLAY_NAMES[category]}: "
            f"{category_counts[category]} / {expected_count}"
        )
    print("Router backbone edges:")
    for source, target in ROUTER_RING_EDGES:
        print(f"  {source} -- {target}")
    print("Endpoint switch assignments:")
    for category in ENDPOINT_CATEGORIES:
        switch_list = ", ".join(_switch_ids_for_category(category))
        print(f"  {CATEGORY_DISPLAY_NAMES[category]}: {switch_list}")


def print_dynamic_summary(windows: Iterable[nx.Graph]) -> None:
    """Print a compact summary of the dynamic timing-window simulation."""
    windows = list(windows)
    phase_counts = Counter(graph.graph["phase"] for graph in windows)
    print(f"Generated {len(windows)} dynamic graph windows.")
    for phase in TIMING_PHASES:
        print(
            f"{phase.name}: windows {phase.start_window}-{phase.end_window} "
            f"({phase_counts[phase.name]} snapshots)"
        )

    node_counts = [graph.number_of_nodes() for graph in windows]
    edge_counts = [graph.number_of_edges() for graph in windows]
    print(
        "Dynamic snapshot range: "
        f"{min(node_counts)}-{max(node_counts)} active nodes, "
        f"{min(edge_counts)}-{max(edge_counts)} active links."
    )
    print("Attack injection hook enabled only for windows 251-350.")


if __name__ == "__main__":
    network = create_network()
    print_summary(network)

    dynamic_windows = create_dynamic_graph_windows()
    print_dynamic_summary(dynamic_windows)

    draw_network(dynamic_windows[0])
    print("Saved dynamic baseline visualization to network_topology.png")
