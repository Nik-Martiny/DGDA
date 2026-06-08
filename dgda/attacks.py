"""Scheduled attack traffic for the dynamic graph simulation.

Attacks use the same routing model as normal conversations: each logical flow is
stored as graph metadata and its packet count is accumulated on every existing
physical hop along the selected route.  Attack injection never adds direct edges
between an attacker and a victim.
"""

from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from dgda.phases import TimingPhase
from dgda.routing import (
    apply_traffic_weight_to_path,
    physical_path_via,
    shortest_physical_path,
)


@dataclass(frozen=True)
class AttackStage:
    """Describe one ordered attack interval within the attack phase."""

    name: str
    start_window: int
    end_window: int
    injector: Callable[[nx.Graph, np.random.Generator], None]

    def contains(self, window: int) -> bool:
        """Return whether this stage owns ``window``."""
        return self.start_window <= window <= self.end_window


def inject_ddos(graph: nx.Graph, rng: np.random.Generator) -> None:
    """Route floods from thirty clients to one web/edge server."""
    attackers = _category_nodes(graph, "client_workstation")[:30]
    target = _category_nodes(graph, "web_edge_server")[0]

    for attacker in attackers:
        packet_count = int(rng.integers(500, 751))
        _route_attack_flow(graph, attacker, target, packet_count, "ddos")


def inject_botnet_c2(graph: nx.Graph, rng: np.random.Generator) -> None:
    """Route a dense logical ten-IoT botnet core through the physical network."""
    bots = _category_nodes(graph, "iot_peripheral")[:10]
    c2_server = _category_nodes(graph, "internal_server")[0]

    for bot in bots:
        packet_count = int(rng.integers(300, 501))
        _route_attack_flow(graph, bot, c2_server, packet_count, "botnet_c2")

    # Preserve the logical peer coordination needed by flow-level k-core
    # detectors, while routing all of those conversations over physical links.
    for index, source in enumerate(bots):
        for target in bots[index + 1 :]:
            packet_count = int(rng.integers(80, 161))
            _route_attack_flow(graph, source, target, packet_count, "botnet_peer")


def inject_mitm(graph: nx.Graph, rng: np.random.Generator) -> None:
    """Force heavy client-to-server traffic through rogue physical router D."""
    rogue_router = "router_D"
    clients = _category_nodes(graph, "client_workstation")[:30]
    servers = _category_nodes(graph, "internal_server")[:20]

    for index, source in enumerate(clients):
        target = servers[index % len(servers)]
        packet_count = int(rng.integers(250, 451))
        path = physical_path_via(graph, source, rogue_router, target)
        _route_attack_flow(graph, source, target, packet_count, "mitm", path)


def inject_port_scan(graph: nx.Graph, rng: np.random.Generator) -> None:
    """Route sixty five-packet probes from one client to rotating random nodes."""
    attacker = _category_nodes(graph, "client_workstation")[-1]
    candidates = sorted(node for node in graph if node != attacker)
    targets = rng.choice(candidates, size=60, replace=False)

    for target in targets:
        _route_attack_flow(graph, attacker, str(target), 5, "port_scan")


def inject_scheduled_attacks(
    graph: nx.Graph, rng: np.random.Generator, phase: TimingPhase
) -> None:
    """Inject the attack assigned to this window of the attack phase.

    The four attacks divide the 100-window attack phase into equal, ordered
    intervals. Actor selection is stable for DDoS, botnet, and MITM so their
    routed traffic patterns persist long enough for detectors to observe them.
    Port-scan targets rotate each window to produce short-lived probe flows.
    """
    if not phase.attack_injection_allowed:
        raise ValueError(f"Cannot inject attacks during the {phase.name} phase.")

    window = graph.graph["window"]
    for stage in ATTACK_STAGES:
        if stage.contains(window):
            graph.graph["attack_name"] = stage.name
            graph.graph["attack_stage_start"] = stage.start_window
            graph.graph["attack_stage_end"] = stage.end_window
            stage.injector(graph, rng)
            _update_attack_totals(graph)
            return

    raise ValueError(f"No scheduled attack is configured for window {window}.")


def _category_nodes(graph: nx.Graph, category: str) -> list[str]:
    """Return stable node IDs for one device category."""
    return sorted(
        node
        for node, attributes in graph.nodes(data=True)
        if attributes["category"] == category
    )


def _route_attack_flow(
    graph: nx.Graph,
    source: str,
    target: str,
    packet_count: int,
    attack_type: str,
    path: list[str] | None = None,
) -> None:
    """Record one logical attack flow and add its weight to physical hops."""
    if path is None:
        path = shortest_physical_path(graph, source, target)

    apply_traffic_weight_to_path(graph, path, packet_count)
    graph.graph.setdefault("attack_flows", []).append(
        {
            "source": source,
            "target": target,
            "attack_type": attack_type,
            "packet_count": packet_count,
            "path": tuple(path),
            "window": graph.graph["window"],
        }
    )


def _update_attack_totals(graph: nx.Graph) -> None:
    """Store compact totals used by summaries, tests, and downstream detectors."""
    flows = graph.graph.get("attack_flows", [])
    graph.graph["attack_flow_count"] = len(flows)
    graph.graph["attack_packet_count"] = sum(flow["packet_count"] for flow in flows)
    graph.graph["traffic_mode"] = f"normal_with_{graph.graph['attack_name']}"


ATTACK_STAGES = (
    AttackStage("ddos", 251, 275, inject_ddos),
    AttackStage("botnet_c2", 276, 300, inject_botnet_c2),
    AttackStage("mitm", 301, 325, inject_mitm),
    AttackStage("port_scan", 326, 350, inject_port_scan),
)
