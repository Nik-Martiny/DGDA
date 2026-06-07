"""Create dynamic graph windows from the static physical topology."""

from collections.abc import Callable

import networkx as nx
import numpy as np

from dgda.config import (
    DEVICE_COUNTS,
    EDGE_WEIGHT_UNIT,
    ENDPOINT_UP_PROBABILITIES,
    INFRASTRUCTURE_CATEGORIES,
    NORMAL_TRAFFIC_RULES,
    RNG_SEED,
    TOTAL_TIME_WINDOWS,
    TRAFFIC_WEIGHT_RANGES,
)
from dgda.phases import TIMING_PHASES, TimingPhase
from dgda.topology import create_network

AttackInjector = Callable[[nx.Graph, np.random.Generator, TimingPhase], None]


def create_dynamic_graph_windows(
    seed: int = RNG_SEED,
    total_windows: int = TOTAL_TIME_WINDOWS,
    attack_injector: AttackInjector | None = None,
) -> list[nx.Graph]:
    """Create all time-window snapshots for the simulation.

    Each returned graph represents one discrete moment in time.  The router and
    switch layer remains stable, endpoint devices randomly appear or disappear,
    and normal communication edges are regenerated for that single window.  The
    optional attack callback only runs during windows 251-350, which protects the
    normal baseline and recovery periods from accidental attack traffic.
    """
    validate_timing_phases(total_windows)

    base_graph = create_network(seed)
    rng = np.random.default_rng(seed)
    windows = []

    for window in range(1, total_windows + 1):
        phase = phase_for_window(window)
        snapshot = create_normal_window_snapshot(base_graph, window, phase, rng)

        if phase.attack_injection_allowed and attack_injector is not None:
            attack_injector(snapshot, rng, phase)
            snapshot.graph["attack_injected"] = True
        else:
            snapshot.graph["attack_injected"] = False

        validate_window_snapshot(snapshot, phase)
        windows.append(snapshot)

    return windows


def phase_for_window(window: int) -> TimingPhase:
    """Return the timing phase that owns a one-indexed window number."""
    for phase in TIMING_PHASES:
        if phase.contains(window):
            return phase

    raise ValueError(f"Window {window} is outside the configured timing phases.")


def validate_timing_phases(total_windows: int) -> None:
    """Ensure timing phases cover the complete simulation without gaps."""
    expected_start = 1

    for phase in TIMING_PHASES:
        if phase.start_window != expected_start:
            raise ValueError(
                f"Phase {phase.name} starts at {phase.start_window}; "
                f"expected {expected_start}."
            )

        if phase.end_window < phase.start_window:
            raise ValueError(f"Phase {phase.name} has an invalid window range.")

        expected_start = phase.end_window + 1

    covered_windows = expected_start - 1
    if covered_windows != total_windows:
        raise ValueError(
            f"Timing phases cover {covered_windows} windows, expected {total_windows}."
        )


def create_normal_window_snapshot(
    base_graph: nx.Graph,
    window: int,
    phase: TimingPhase,
    rng: np.random.Generator,
) -> nx.Graph:
    """Create one snapshot before any optional attack mutation is applied."""
    active_nodes = active_nodes_for_window(base_graph, rng)
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
        traffic_mode=(
            "normal_with_attack_slot"
            if phase.attack_injection_allowed
            else "normal_only"
        ),
        edge_weight_unit=EDGE_WEIGHT_UNIT,
    )

    reset_physical_edge_weights(snapshot)
    add_normal_traffic_edges(snapshot, rng, window)
    return snapshot


def active_nodes_for_window(base_graph: nx.Graph, rng: np.random.Generator) -> set[str]:
    """Select which devices are online during a normal time window.

    Routers and switches are always active because they are the network
    infrastructure.  Endpoints use category-specific probabilities so clients and
    IoT devices churn more often than servers.
    """
    active_nodes = set()

    for node, attributes in base_graph.nodes(data=True):
        category = attributes["category"]

        if category in INFRASTRUCTURE_CATEGORIES:
            active_nodes.add(node)
            continue

        probability = ENDPOINT_UP_PROBABILITIES[category]
        random_value = rng.random()

        if random_value <= probability:
            active_nodes.add(node)

    return active_nodes


def add_normal_traffic_edges(
    graph: nx.Graph,
    rng: np.random.Generator,
    window: int,
) -> None:
    """Route short-lived normal endpoint conversations over physical links.

    Physical links describe how devices are cabled through switches and routers.
    A normal conversation is now recorded as graph-level flow metadata instead of
    a direct endpoint-to-endpoint edge.  The sampled packet count is accumulated
    on every physical hop along the best switch/router path, so endpoint devices
    only communicate through their access switch, router, and backbone links.
    """
    nodes_by_category = group_nodes_by_category(graph)
    communication_flows = graph.graph.setdefault("communication_flows", [])
    used_flow_pairs: set[tuple[str, str, str]] = set()

    for rule in NORMAL_TRAFFIC_RULES:
        source_category, target_category, min_edges, max_edges, traffic_label = rule
        candidate_sources = nodes_by_category[source_category]
        candidate_targets = nodes_by_category[target_category]

        if not candidate_sources or not candidate_targets:
            continue

        target_edges = int(rng.integers(min_edges, max_edges + 1))
        max_attempts = target_edges * 10
        attempts = 0
        added_edges = 0

        while added_edges < target_edges and attempts < max_attempts:
            attempts += 1
            source = str(rng.choice(candidate_sources))
            target = str(rng.choice(candidate_targets))

            if source == target:
                continue

            flow_key = ordered_flow_key(source, target, traffic_label)
            if flow_key in used_flow_pairs:
                continue

            packet_count = sample_traffic_weight(traffic_label, rng)
            physical_path = shortest_physical_path(graph, source, target)
            apply_traffic_weight_to_path(graph, physical_path, packet_count)
            communication_flows.append(
                {
                    "source": source,
                    "target": target,
                    "traffic_profile": traffic_label,
                    "packet_count": packet_count,
                    "path": tuple(physical_path),
                    "window": window,
                }
            )
            used_flow_pairs.add(flow_key)
            added_edges += 1

    graph.graph["communication_flow_count"] = len(communication_flows)
    graph.graph["communication_packet_count"] = sum(
        flow["packet_count"] for flow in communication_flows
    )


def reset_physical_edge_weights(graph: nx.Graph) -> None:
    """Reset stable physical links before routing one window's traffic through them."""
    for _source, _target, attributes in graph.edges(data=True):
        if attributes.get("link_type") == "normal_traffic":
            continue

        attributes["weight"] = 0
        attributes["weight_unit"] = EDGE_WEIGHT_UNIT


def shortest_physical_path(graph: nx.Graph, source: str, target: str) -> list[str]:
    """Return the best physical path between endpoints through infrastructure."""
    physical_view = nx.subgraph_view(
        graph,
        filter_edge=lambda left, right: graph.edges[left, right].get("link_type")
        != "normal_traffic",
    )

    return nx.shortest_path(physical_view, source, target)


def ordered_flow_key(source: str, target: str, traffic_label: str) -> tuple[str, str, str]:
    """Return a stable key for one undirected endpoint conversation."""
    if source <= target:
        return (source, target, traffic_label)

    return (target, source, traffic_label)


def apply_traffic_weight_to_path(
    graph: nx.Graph, physical_path: list[str], weight: int
) -> None:
    """Add one endpoint conversation's weight to every physical hop it uses."""
    for source, target in zip(physical_path[:-1], physical_path[1:], strict=True):
        attributes = graph.edges[source, target]
        attributes["weight"] += weight
        attributes["weight_unit"] = EDGE_WEIGHT_UNIT


def sample_traffic_weight(traffic_label: str, rng: np.random.Generator) -> int:
    """Sample how often two nodes communicate in one time window."""
    min_weight, max_weight = TRAFFIC_WEIGHT_RANGES[traffic_label]

    return int(rng.integers(min_weight, max_weight + 1))


def group_nodes_by_category(graph: nx.Graph) -> dict[str, list[str]]:
    """Group node IDs by their device category."""
    nodes_by_category = {}

    for category in DEVICE_COUNTS:
        nodes_by_category[category] = []

    for node, attributes in graph.nodes(data=True):
        category = attributes["category"]
        nodes_by_category[category].append(node)

    return nodes_by_category


def validate_window_snapshot(graph: nx.Graph, phase: TimingPhase) -> None:
    """Validate one generated time window before callers use it."""
    if graph.graph["phase"] != phase.name:
        raise ValueError(f"Window phase metadata drifted from {phase.name}.")

    if phase.normal_traffic_only and graph.graph["attack_injected"]:
        raise ValueError(f"Attack traffic leaked into the {phase.name} phase.")

    allowed_link_types = {
        "router_to_switch",
        "router_backbone",
        "access",
    }

    unexpected_edges = []
    invalid_weighted_edges = []
    direct_endpoint_edges = []

    for source, target, attributes in graph.edges(data=True):
        link_type = attributes.get("link_type")
        if link_type not in allowed_link_types:
            unexpected_edges.append((source, target, link_type))
            continue

        has_numeric_weight = isinstance(attributes.get("weight"), int | float)
        if not has_numeric_weight or attributes["weight"] < 0:
            invalid_weighted_edges.append(
                (source, target, link_type, attributes.get("weight"))
            )
            continue

        if "weight_unit" not in attributes:
            invalid_weighted_edges.append((source, target, link_type, "weight_unit"))

        source_category = graph.nodes[source]["category"]
        target_category = graph.nodes[target]["category"]
        both_are_endpoints = (
            source_category not in INFRASTRUCTURE_CATEGORIES
            and target_category not in INFRASTRUCTURE_CATEGORIES
        )
        if both_are_endpoints:
            direct_endpoint_edges.append((source, target, link_type))

    validate_communication_flows(graph)

    if unexpected_edges and not phase.attack_injection_allowed:
        raise ValueError(
            f"Unexpected edge types in normal-only {phase.name} phase: {unexpected_edges}"
        )

    if direct_endpoint_edges and not phase.attack_injection_allowed:
        raise ValueError(
            f"Direct endpoint edges are not allowed in window {graph.graph['window']}: "
            f"{direct_endpoint_edges}"
        )

    if invalid_weighted_edges:
        raise ValueError(
            f"Invalid edge-weight attributes in window {graph.graph['window']}: "
            f"{invalid_weighted_edges}"
        )

    if not nx.is_connected(graph):
        raise ValueError(f"Window {graph.graph['window']} is not connected.")


def validate_communication_flows(graph: nx.Graph) -> None:
    """Validate graph-level endpoint flow metadata and routed packet counts."""
    flows = graph.graph.get("communication_flows", [])
    packet_total = 0

    if graph.graph.get("communication_flow_count", len(flows)) != len(flows):
        raise ValueError(
            f"Flow-count metadata does not match in window {graph.graph['window']}."
        )

    for flow in flows:
        source = flow.get("source")
        target = flow.get("target")
        path = list(flow.get("path", ()))
        packet_count = flow.get("packet_count")

        if source not in graph or target not in graph:
            raise ValueError(f"Flow references inactive nodes: {flow}")

        if graph.has_edge(source, target):
            raise ValueError(f"Flow was also modeled as a direct graph edge: {flow}")

        if not isinstance(packet_count, int) or packet_count <= 0:
            raise ValueError(f"Flow has invalid packet count: {flow}")

        if path[:1] != [source] or path[-1:] != [target]:
            raise ValueError(f"Flow path endpoints do not match source/target: {flow}")

        for left, right in zip(path[:-1], path[1:], strict=True):
            if not graph.has_edge(left, right):
                raise ValueError(f"Flow path uses a missing physical edge: {flow}")

        packet_total += packet_count

    if graph.graph.get("communication_packet_count", packet_total) != packet_total:
        raise ValueError(
            f"Packet-count metadata does not match in window {graph.graph['window']}."
        )
