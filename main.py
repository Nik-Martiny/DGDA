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
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
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
ANIMATION_INTERVAL_MS = 90
ANIMATION_FPS = 12

PHASE_COLORS = {
    "baseline": "#E8F4FF",
    "pre_attack": "#FFF5D6",
    "attack": "#FFE3E0",
    "recovery": "#E6F6EA",
}

LINK_TYPE_COLORS = {
    "router_backbone": "#1F2937",
    "router_to_switch": "#6B7280",
    "access": "#D1D5DB",
    "normal_traffic": "#38BDF8",
}

LINK_TYPE_WIDTHS = {
    "router_backbone": 2.8,
    "router_to_switch": 1.8,
    "access": 0.55,
    "normal_traffic": 0.45,
}

LINK_TYPE_ALPHAS = {
    "router_backbone": 0.9,
    "router_to_switch": 0.65,
    "access": 0.24,
    "normal_traffic": 0.55,
}


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


def create_stable_layout(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    """Return stable, visually grouped positions for every device in the network."""
    positions: dict[str, tuple[float, float]] = {}
    router_angles = np.linspace(0, 2 * np.pi, len(ROUTER_NAMES), endpoint=False)
    router_radius = 3.0
    switch_radius = 4.05

    for router_name, angle in zip(ROUTER_NAMES, router_angles):
        router_id = f"router_{router_name}"
        switch_id = f"switch_{router_name}"
        positions[router_id] = (
            float(router_radius * np.cos(angle)),
            float(router_radius * np.sin(angle)),
        )
        positions[switch_id] = (
            float(switch_radius * np.cos(angle)),
            float(switch_radius * np.sin(angle)),
        )

    nodes_by_switch: dict[str, list[str]] = {switch_id: [] for switch_id in SWITCH_NAMES}
    for node, attributes in graph.nodes(data=True):
        if attributes["category"] in INFRASTRUCTURE_CATEGORIES:
            continue
        attached_switch = next(
            neighbor
            for neighbor in graph.neighbors(node)
            if graph.nodes[neighbor]["category"] == "switch"
        )
        nodes_by_switch[attached_switch].append(node)

    for switch_id, endpoint_nodes in nodes_by_switch.items():
        if not endpoint_nodes:
            continue
        switch_x, switch_y = positions[switch_id]
        outward_angle = np.arctan2(switch_y, switch_x)
        tangent_angle = outward_angle + np.pi / 2
        endpoint_nodes.sort(key=lambda node: (graph.nodes[node]["category"], node))
        ring_count = len(endpoint_nodes)
        endpoint_radius = 0.72 + min(0.72, ring_count / 90)
        for index, node in enumerate(endpoint_nodes):
            offset = (index - (ring_count - 1) / 2) * 0.12
            wobble = 0.28 * np.sin(index * 1.618)
            positions[node] = (
                float(
                    (switch_x + endpoint_radius * np.cos(outward_angle))
                    + offset * np.cos(tangent_angle)
                ),
                float(
                    (switch_y + endpoint_radius * np.sin(outward_angle))
                    + (offset + wobble) * np.sin(tangent_angle)
                ),
            )

    return positions


def animate_dynamic_graph_windows(
    windows: Iterable[nx.Graph],
    output_path: str | Path = "dynamic_graph_windows.gif",
    interval_ms: int = ANIMATION_INTERVAL_MS,
    fps: int = ANIMATION_FPS,
    dpi: int = 120,
) -> Path:
    """Render every dynamic graph window sequentially as a FuncAnimation GIF."""
    windows = list(windows)
    if not windows:
        raise ValueError("At least one graph window is required to build an animation.")

    output_path = Path(output_path)
    reference_graph = create_network()
    positions = create_stable_layout(reference_graph)
    all_nodes = list(reference_graph.nodes)
    node_categories = nx.get_node_attributes(reference_graph, "category")
    edge_layers = tuple(LINK_TYPE_COLORS)

    fig, ax = plt.subplots(figsize=(16, 12), facecolor="#F8FAFC")
    plt.subplots_adjust(bottom=0.13)

    def update(frame_index: int):
        ax.clear()
        graph = windows[frame_index]
        phase = graph.graph["phase"]
        active_nodes = set(graph.nodes)
        inactive_nodes = [node for node in all_nodes if node not in active_nodes]
        present_nodes = [node for node in all_nodes if node in active_nodes]
        active_node_colors = [
            CATEGORY_COLORS[node_categories[node]] for node in present_nodes
        ]
        active_node_sizes = [
            _node_size_for_category(node_categories[node]) for node in present_nodes
        ]

        ax.set_facecolor(PHASE_COLORS[phase])
        nx.draw_networkx_nodes(
            reference_graph,
            positions,
            nodelist=inactive_nodes,
            node_color="#CBD5E1",
            node_size=22,
            alpha=0.22,
            linewidths=0,
            ax=ax,
        )

        for link_type in edge_layers:
            edges = [
                (source, target)
                for source, target, attributes in graph.edges(data=True)
                if attributes.get("link_type") == link_type
            ]
            if not edges:
                continue
            nx.draw_networkx_edges(
                graph,
                positions,
                edgelist=edges,
                edge_color=LINK_TYPE_COLORS[link_type],
                width=LINK_TYPE_WIDTHS[link_type],
                alpha=LINK_TYPE_ALPHAS[link_type],
                ax=ax,
            )

        nx.draw_networkx_nodes(
            graph,
            positions,
            nodelist=present_nodes,
            node_color=active_node_colors,
            node_size=active_node_sizes,
            linewidths=0.65,
            edgecolors="white",
            ax=ax,
        )
        _draw_infrastructure_labels(graph, positions, ax)
        _draw_timeline_bar(ax, graph.graph["window"])
        ax.set_title(
            "Dynamic Communication Network — "
            f"Window {graph.graph['window']:03d}/500 ({phase.replace('_', ' ').title()})\n"
            f"{graph.number_of_nodes()} active nodes • {graph.number_of_edges()} active links",
            fontsize=18,
            fontweight="bold",
            color="#0F172A",
            pad=14,
        )
        ax.legend(
            handles=_animation_legend_handles(),
            loc="upper right",
            frameon=True,
            facecolor="white",
            framealpha=0.92,
            fontsize=9,
        )
        ax.axis("off")
        ax.set_aspect("equal")
        return ax.collections + ax.lines

    animation = FuncAnimation(
        fig,
        update,
        frames=len(windows),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return output_path


def draw_connection_activity_heatmap(
    windows: Iterable[nx.Graph],
    output_path: str | Path = "connection_activity_heatmap.png",
) -> Path:
    """Draw a window-by-window heatmap of every edge that appears in the simulation."""
    windows = list(windows)
    if not windows:
        raise ValueError("At least one graph window is required to draw edge activity.")

    output_path = Path(output_path)
    edge_order = _ordered_union_edges(windows)
    edge_index = {edge: index for index, edge in enumerate(edge_order)}
    activity = np.zeros((len(edge_order), len(windows)), dtype=np.uint8)
    link_type_codes = {
        "access": 1,
        "router_to_switch": 2,
        "router_backbone": 3,
        "normal_traffic": 4,
    }

    for column, graph in enumerate(windows):
        for source, target, attributes in graph.edges(data=True):
            edge = tuple(sorted((source, target)))
            activity[edge_index[edge], column] = link_type_codes.get(
                attributes.get("link_type"), 5
            )

    fig_height = max(8, min(28, len(edge_order) / 180))
    fig, ax = plt.subplots(figsize=(18, fig_height), facecolor="white")
    cmap = ListedColormap(
        ["#FFFFFF", "#D1D5DB", "#6B7280", "#111827", "#0EA5E9", "#DC2626"]
    )
    image = ax.imshow(
        activity, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=5
    )

    _shade_phase_spans(ax, len(edge_order))
    ax.set_title("Connection Activity Across All 500 Dynamic Windows", fontsize=16, fontweight="bold")
    ax.set_xlabel("Time window")
    ax.set_ylabel(f"Unique edges observed ({len(edge_order):,})")
    ax.set_xticks([0, 149, 249, 349, 499])
    ax.set_xticklabels(["1", "150", "250", "350", "500"])
    _label_edge_axis(ax, edge_order)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.01, ticks=range(6))
    colorbar.ax.set_yticklabels(
        ["absent", "access", "router→switch", "backbone", "normal traffic", "attack"]
    )
    ax.legend(handles=_phase_legend_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def draw_window_connection_matrix(
    graph: nx.Graph,
    output_path: str | Path = "window_connection_matrix.png",
) -> Path:
    """Render one snapshot as a 200-device adjacency matrix to make dense edge changes clear."""
    output_path = Path(output_path)
    reference_graph = create_network()
    node_order = _ordered_nodes_for_matrix(reference_graph)
    node_index = {node: index for index, node in enumerate(node_order)}
    matrix = np.zeros((len(node_order), len(node_order)), dtype=np.uint8)
    link_type_codes = {
        "access": 1,
        "router_to_switch": 2,
        "router_backbone": 3,
        "normal_traffic": 4,
    }

    for source, target, attributes in graph.edges(data=True):
        if source not in node_index or target not in node_index:
            continue
        source_index = node_index[source]
        target_index = node_index[target]
        code = link_type_codes.get(attributes.get("link_type"), 5)
        matrix[source_index, target_index] = code
        matrix[target_index, source_index] = code

    fig, ax = plt.subplots(figsize=(14, 12), facecolor="white")
    cmap = ListedColormap(["#FFFFFF", "#D1D5DB", "#6B7280", "#111827", "#0EA5E9", "#DC2626"])
    image = ax.imshow(matrix, interpolation="nearest", cmap=cmap, vmin=0, vmax=5)
    _draw_category_grid(ax, reference_graph, node_order)
    ax.set_title(
        "All Node-to-Node Connections — "
        f"Window {graph.graph.get('window', 0):03d} ({graph.graph.get('phase', 'static')})",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_xlabel("Target node ordered by device category")
    ax.set_ylabel("Source node ordered by device category")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, ticks=range(6))
    colorbar.ax.set_yticklabels(
        ["absent", "access", "router→switch", "backbone", "normal traffic", "attack"]
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def _node_size_for_category(category: str) -> int:
    """Map device category to a readable node size for dense visualizations."""
    if category == "router":
        return 600
    if category == "switch":
        return 340
    return 72


def _draw_infrastructure_labels(
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    ax: plt.Axes,
) -> None:
    """Label routers and switches without overwhelming endpoint animation frames."""
    labels = {
        node: graph.nodes[node]["label"].replace("Router ", "R").replace("Switch ", "S")
        for node in graph.nodes
        if graph.nodes[node]["category"] in INFRASTRUCTURE_CATEGORIES
    }
    nx.draw_networkx_labels(
        graph, positions, labels=labels, font_size=8, font_weight="bold", ax=ax
    )


def _draw_timeline_bar(ax: plt.Axes, current_window: int) -> None:
    """Add a phase-colored progress bar under the animated network."""
    inset = ax.inset_axes([0.05, -0.08, 0.9, 0.045])
    for phase in TIMING_PHASES:
        start = (phase.start_window - 1) / TOTAL_TIME_WINDOWS
        width = (phase.end_window - phase.start_window + 1) / TOTAL_TIME_WINDOWS
        inset.barh(
            0,
            width,
            left=start,
            height=1,
            color=PHASE_COLORS[phase.name],
            edgecolor="white",
        )
        inset.text(
            start + width / 2,
            0,
            phase.name.replace("_", "\n"),
            ha="center",
            va="center",
            fontsize=8,
        )
    inset.axvline(
        (current_window - 1) / TOTAL_TIME_WINDOWS, color="#0F172A", linewidth=2.2
    )
    inset.set_xlim(0, 1)
    inset.set_ylim(-0.5, 0.5)
    inset.axis("off")


def _animation_legend_handles() -> list[plt.Line2D | Patch]:
    """Build a compact legend for node categories, edge types, and phases."""
    node_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=CATEGORY_DISPLAY_NAMES[category],
            markerfacecolor=color,
            markersize=9,
        )
        for category, color in CATEGORY_COLORS.items()
    ]
    edge_handles = [
        plt.Line2D(
            [0],
            [0],
            color=color,
            lw=max(1.5, LINK_TYPE_WIDTHS[link_type]),
            label=link_type.replace("_", " "),
        )
        for link_type, color in LINK_TYPE_COLORS.items()
    ]
    return node_handles + edge_handles


def _phase_legend_handles() -> list[Patch]:
    """Return legend patches for the four timing phases."""
    return [
        Patch(facecolor=color, edgecolor="white", label=phase.replace("_", " ").title())
        for phase, color in PHASE_COLORS.items()
    ]


def _ordered_union_edges(windows: list[nx.Graph]) -> list[tuple[str, str]]:
    """Collect all observed edges sorted by endpoint category and name."""
    reference_graph = create_network()
    categories = nx.get_node_attributes(reference_graph, "category")
    edges = {tuple(sorted((source, target))) for graph in windows for source, target in graph.edges}
    return sorted(
        edges,
        key=lambda edge: (
            categories.get(edge[0], "zzz"),
            categories.get(edge[1], "zzz"),
            edge[0],
            edge[1],
        ),
    )


def _label_edge_axis(ax: plt.Axes, edge_order: list[tuple[str, str]]) -> None:
    """Label a sampled set of heatmap rows so the plot remains readable."""
    if len(edge_order) <= 30:
        ticks = list(range(len(edge_order)))
    else:
        ticks = np.linspace(0, len(edge_order) - 1, 12, dtype=int).tolist()
    ax.set_yticks(ticks)
    ax.set_yticklabels(
        [f"{edge_order[index][0]} — {edge_order[index][1]}" for index in ticks],
        fontsize=7,
    )


def _shade_phase_spans(ax: plt.Axes, edge_count: int) -> None:
    """Overlay translucent phase bands on the edge activity heatmap."""
    for phase in TIMING_PHASES:
        ax.axvspan(
            phase.start_window - 1.5,
            phase.end_window - 0.5,
            color=PHASE_COLORS[phase.name],
            alpha=0.16,
            linewidth=0,
        )
        ax.text(
            (phase.start_window + phase.end_window) / 2 - 1,
            -max(8, edge_count * 0.012),
            phase.name.replace("_", " ").title(),
            ha="center",
            va="bottom",
            fontsize=9,
            color="#334155",
        )


def _ordered_nodes_for_matrix(reference_graph: nx.Graph) -> list[str]:
    """Order nodes by category and numeric name to reveal block connection patterns."""
    category_rank = {category: rank for rank, category in enumerate(DEVICE_COUNTS)}
    return sorted(
        reference_graph.nodes,
        key=lambda node: (category_rank[reference_graph.nodes[node]["category"]], node),
    )


def _draw_category_grid(ax: plt.Axes, reference_graph: nx.Graph, node_order: list[str]) -> None:
    """Draw category block boundaries and labels on an adjacency matrix."""
    categories = [reference_graph.nodes[node]["category"] for node in node_order]
    boundaries = [0]
    labels = []
    for category in DEVICE_COUNTS:
        count = categories.count(category)
        if count == 0:
            continue
        start = boundaries[-1]
        boundaries.append(start + count)
        labels.append((start + count / 2 - 0.5, CATEGORY_DISPLAY_NAMES[category]))

    for boundary in boundaries:
        ax.axhline(boundary - 0.5, color="#475569", linewidth=0.55, alpha=0.7)
        ax.axvline(boundary - 0.5, color="#475569", linewidth=0.55, alpha=0.7)

    tick_positions = [position for position, _ in labels]
    tick_labels = [label for _, label in labels]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=8)


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

    draw_connection_activity_heatmap(dynamic_windows)
    print("Saved all-window connection activity heatmap to connection_activity_heatmap.png")

    draw_window_connection_matrix(dynamic_windows[250])
    print("Saved attack-phase connection matrix to window_connection_matrix.png")

    animate_dynamic_graph_windows(dynamic_windows)
    print("Saved FuncAnimation render to dynamic_graph_windows.gif")
