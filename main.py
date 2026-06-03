"""Build and visualize a 200-device communication network simulation.

The topology models ten routers split into two five-router rings. Router A in the
first ring bridges to router F in the second ring. Each router has one directly
attached switch. Internal servers attach only to Router F's switch, web/edge
servers attach only to Router G's switch, and client/IoT devices attach to the
remaining switches in the second router ring.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


TOTAL_DEVICES = 200
DEVICE_COUNTS = {
    "client_workstation": 80,
    "internal_server": 40,
    "web_edge_server": 30,
    "router_switch": 20,
    "iot_peripheral": 30,
}

ROUTER_NAMES = tuple("ABCDEFGHIJ")
SWITCH_NAMES = tuple(f"switch_{router}" for router in ROUTER_NAMES)
INTERNAL_SERVER_ROUTER = "F"
WEB_EDGE_SERVER_ROUTER = "G"
CLIENT_IOT_ROUTERS = ROUTER_NAMES[7:]
CATEGORY_ROUTER_ASSIGNMENTS = {
    "client_workstation": CLIENT_IOT_ROUTERS,
    "internal_server": (INTERNAL_SERVER_ROUTER,),
    "web_edge_server": (WEB_EDGE_SERVER_ROUTER,),
    "iot_peripheral": CLIENT_IOT_ROUTERS,
}
RNG_SEED = 42


ROUTER_RING_EDGES = (
    ("router_A", "router_B"),
    ("router_B", "router_C"),
    ("router_C", "router_D"),
    ("router_D", "router_E"),
    ("router_E", "router_A"),
    ("router_A", "router_F"),
    ("router_F", "router_G"),
    ("router_G", "router_H"),
    ("router_H", "router_I"),
    ("router_I", "router_J"),
    ("router_J", "router_F"),
)


CATEGORY_DISPLAY_NAMES = {
    "client_workstation": "Client Workstations",
    "internal_server": "Internal Servers",
    "web_edge_server": "Web/Edge Servers",
    "router_switch": "Routers/Switches",
    "iot_peripheral": "IoT/Peripheral Devices",
}

CATEGORY_COLORS = {
    "client_workstation": "#4E79A7",
    "internal_server": "#59A14F",
    "web_edge_server": "#F28E2B",
    "router_switch": "#E15759",
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


def create_network(seed: int = RNG_SEED) -> nx.Graph:
    """Create the 200-device network with the requested device categories."""
    graph = nx.Graph(name="Dynamic Graph Device Network")
    rng = np.random.default_rng(seed)

    _add_router_and_switch_layer(graph)
    _add_endpoint_devices(graph, rng)
    _validate_network(graph)

    return graph


def _add_router_and_switch_layer(graph: nx.Graph) -> None:
    """Add ten routers, ten switches, two router rings, and router-switch links."""
    for router_name in ROUTER_NAMES:
        router_id = f"router_{router_name}"
        switch_id = f"switch_{router_name}"
        network_name = _network_name_for_router(router_name)
        graph.add_node(
            router_id,
            category="router_switch",
            device_type="router",
            network=network_name,
            label=f"Router {router_name}",
        )
        graph.add_node(
            switch_id,
            category="router_switch",
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

        for index in range(1, count + 1):
            device_id = f"{prefix}_{index:03d}"
            attached_switch = str(rng.choice(switch_ids))
            graph.add_node(
                device_id,
                category=category,
                device_type=category,
                network=network_name,
                label=f"{CATEGORY_DISPLAY_NAMES[category]} {index}",
            )
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
    if graph.number_of_nodes() != TOTAL_DEVICES:
        raise ValueError(
            f"Expected {TOTAL_DEVICES} devices, found {graph.number_of_nodes()}."
        )

    actual_counts = Counter(nx.get_node_attributes(graph, "category").values())
    if actual_counts != DEVICE_COUNTS:
        raise ValueError(
            f"Device category counts do not match: {dict(actual_counts)} != {DEVICE_COUNTS}."
        )

    missing_backbone_edges = [
        (source, target)
        for source, target in ROUTER_RING_EDGES
        if not graph.has_edge(source, target)
    ]
    if missing_backbone_edges:
        raise ValueError(f"Missing router ring/bridge edges: {missing_backbone_edges}")

    for router_name in ROUTER_NAMES:
        router_id = f"router_{router_name}"
        switch_id = f"switch_{router_name}"
        if not graph.has_edge(router_id, switch_id):
            raise ValueError(f"Missing router-switch edge: {(router_id, switch_id)}")

    misplaced_endpoints = []
    for node, attributes in graph.nodes(data=True):
        category = attributes["category"]
        if category == "router_switch":
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


def draw_network(graph: nx.Graph, output_path: str | Path = "network_topology.png") -> None:
    """Render the simulated network to a PNG image."""
    output_path = Path(output_path)
    positions = _layout_positions(graph)
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
        if graph.nodes[node]["category"] == "router_switch"
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
    plt.legend(handles=legend_handles, loc="upper right")
    plt.title("200-Device Communication Network Simulation")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _layout_positions(graph: nx.Graph) -> dict[str, np.ndarray]:
    """Create a deterministic layout that keeps each router's access devices nearby."""
    positions: dict[str, np.ndarray] = {}
    first_ring_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 2
    second_ring_angles = np.linspace(0, 2 * np.pi, 5, endpoint=False) + np.pi / 2
    ring_centers = {"left": np.array([-4.0, 0.0]), "right": np.array([4.0, 0.0])}

    for router_name, angle in zip(ROUTER_NAMES[:5], first_ring_angles):
        positions[f"router_{router_name}"] = ring_centers["left"] + 1.7 * np.array(
            [np.cos(angle), np.sin(angle)]
        )
    for router_name, angle in zip(ROUTER_NAMES[5:], second_ring_angles):
        positions[f"router_{router_name}"] = ring_centers["right"] + 1.7 * np.array(
            [np.cos(angle), np.sin(angle)]
        )

    for router_name in ROUTER_NAMES:
        router_position = positions[f"router_{router_name}"]
        ring_center = (
            ring_centers["left"]
            if router_name in ROUTER_NAMES[:5]
            else ring_centers["right"]
        )
        direction = router_position - ring_center
        direction = direction / np.linalg.norm(direction)
        positions[f"switch_{router_name}"] = router_position + 0.9 * direction

    for switch_id in SWITCH_NAMES:
        switch_position = positions[switch_id]
        attached_devices = sorted(
            neighbor
            for neighbor in graph.neighbors(switch_id)
            if graph.nodes[neighbor]["category"] != "router_switch"
        )
        if not attached_devices:
            continue

        angles = np.linspace(0, 2 * np.pi, len(attached_devices), endpoint=False)
        radius = 0.75 + min(len(attached_devices), 24) * 0.025
        for device, angle in zip(attached_devices, angles):
            positions[device] = switch_position + radius * np.array(
                [np.cos(angle), np.sin(angle)]
            )

    return positions


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


if __name__ == "__main__":
    network = create_network()
    print_summary(network)
    draw_network(network)
    print("Saved topology visualization to network_topology.png")
