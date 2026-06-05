"""Build and validate the static physical network topology."""

from collections import Counter

import networkx as nx
import numpy as np

from dgda.config import (
    CATEGORY_DISPLAY_NAMES,
    CATEGORY_ROUTER_ASSIGNMENTS,
    CLIENT_IOT_ROUTERS,
    DEVICE_COUNTS,
    ENDPOINT_CATEGORIES,
    ENDPOINT_PREFIXES,
    INTERNAL_SERVER_ROUTER,
    RNG_SEED,
    ROUTER_NAMES,
    ROUTER_RING_EDGES,
    TOTAL_DEVICES,
    WEB_EDGE_SERVER_ROUTER,
)


def create_network(seed: int = RNG_SEED) -> nx.Graph:
    """Create the deterministic 200-device physical network.

    This is the base topology used by every dynamic time window.  Routers,
    switches, and access links are stable here; later modules copy this graph and
    temporarily hide endpoints to simulate devices going online and offline.
    """
    graph = nx.Graph(name="Dynamic Graph Device Network")
    rng = np.random.default_rng(seed)

    _add_router_and_switch_layer(graph)
    _add_endpoint_devices(graph, rng)
    validate_network(graph)

    return graph


def switch_ids_for_category(category: str) -> tuple[str, ...]:
    """Return the switches that a device category may use.

    The topology rules are encoded in ``CATEGORY_ROUTER_ASSIGNMENTS``.  This
    helper turns router names such as ``F`` into node IDs such as ``switch_F`` so
    callers do not repeat string-building code.
    """
    switch_ids = []

    for router_name in CATEGORY_ROUTER_ASSIGNMENTS[category]:
        switch_ids.append(f"switch_{router_name}")

    return tuple(switch_ids)


def network_name_for_category(category: str) -> str:
    """Return a logical network label for an endpoint category.

    The label is stored on each node.  It makes summaries and later analysis
    easier because a detector can quickly separate client/IoT, internal server,
    and web/edge server areas.
    """
    if category == "internal_server":
        return "internal_server_network"

    if category == "web_edge_server":
        return "web_edge_server_network"

    return "client_iot_network"


def network_name_for_router(router_name: str) -> str:
    """Return the logical network role for a router and its switch."""
    if router_name == INTERNAL_SERVER_ROUTER:
        return "internal_server_network"

    if router_name == WEB_EDGE_SERVER_ROUTER:
        return "web_edge_server_network"

    if router_name in CLIENT_IOT_ROUTERS:
        return "client_iot_network"

    return "backbone_network"


def validate_network(graph: nx.Graph) -> None:
    """Raise ``ValueError`` if the base topology violates the simulation design.

    Validation is important because the dynamic simulation assumes the static
    topology is correct.  A bad base graph would make every generated time window
    misleading, so this function fails fast before visualization or detection.
    """
    if graph.number_of_nodes() != TOTAL_DEVICES:
        raise ValueError(
            f"Expected {TOTAL_DEVICES} devices, found {graph.number_of_nodes()}."
        )

    actual_counts = Counter(nx.get_node_attributes(graph, "category").values())
    if actual_counts != DEVICE_COUNTS:
        raise ValueError(
            f"Device category counts do not match: {dict(actual_counts)} != {DEVICE_COUNTS}."
        )

    missing_edges = []
    for source, target in ROUTER_RING_EDGES:
        if not graph.has_edge(source, target):
            missing_edges.append((source, target))

    if missing_edges:
        raise ValueError(f"Missing router ring/bridge edges: {missing_edges}")

    for router_name in ROUTER_NAMES:
        router_id = f"router_{router_name}"
        switch_id = f"switch_{router_name}"
        if not graph.has_edge(router_id, switch_id):
            raise ValueError(f"Missing router-switch edge: {(router_id, switch_id)}")

    misplaced_endpoints = []
    for node, attributes in graph.nodes(data=True):
        category = attributes["category"]
        if category == "router" or category == "switch":
            continue

        allowed_switches = set(switch_ids_for_category(category))
        access_switches = []

        for neighbor in graph.neighbors(node):
            if graph.nodes[neighbor]["device_type"] == "switch":
                access_switches.append(neighbor)

        has_one_switch = len(access_switches) == 1
        uses_allowed_switch = has_one_switch and access_switches[0] in allowed_switches

        if not uses_allowed_switch:
            misplaced_endpoints.append((node, tuple(access_switches), category))

    if misplaced_endpoints:
        raise ValueError(
            "Endpoint nodes must attach only to their assigned router switches: "
            f"{misplaced_endpoints}"
        )


def _add_router_and_switch_layer(graph: nx.Graph) -> None:
    """Add routers, switches, backbone links, and router-to-switch links."""
    for router_name in ROUTER_NAMES:
        router_id = f"router_{router_name}"
        switch_id = f"switch_{router_name}"
        network_name = network_name_for_router(router_name)

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

        _add_physical_edge(graph, router_id, switch_id, "router_to_switch")

    for source, target in ROUTER_RING_EDGES:
        _add_physical_edge(graph, source, target, "router_backbone")


def _add_physical_edge(
    graph: nx.Graph, source: str, target: str, link_type: str
) -> None:
    """Add a stable physical link with a simple default edge weight."""
    graph.add_edge(source, target, link_type=link_type, weight=1)


def _add_endpoint_devices(graph: nx.Graph, rng: np.random.Generator) -> None:
    """Create clients, servers, and IoT devices and attach them to switches."""
    for category in ENDPOINT_CATEGORIES:
        count = DEVICE_COUNTS[category]
        prefix = ENDPOINT_PREFIXES[category]
        switch_ids = switch_ids_for_category(category)
        network_name = network_name_for_category(category)

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

            _add_physical_edge(graph, device_id, attached_switch, "access")
