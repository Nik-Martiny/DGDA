"""Scheduled attack traffic for the dynamic graph simulation.

Attack edges are intentionally represented as short-lived virtual communication
links.  Normal traffic continues to use the physical routed topology, while the
virtual links make the topology changes caused by each attack directly visible
to graph and spectral detectors.
"""

from collections.abc import Callable
from dataclasses import dataclass

import networkx as nx
import numpy as np

from dgda.config import EDGE_WEIGHT_UNIT
from dgda.phases import TimingPhase


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
    """Create a 30-client, one-web-server heavy flooding star."""
    attackers = _category_nodes(graph, "client_workstation")[:30]
    target = _category_nodes(graph, "web_edge_server")[0]

    for attacker in attackers:
        packet_count = int(rng.integers(500, 751))
        _add_attack_flow(graph, attacker, target, packet_count, "ddos")


def inject_botnet_c2(graph: nx.Graph, rng: np.random.Generator) -> None:
    """Create a dense ten-IoT botnet core around one internal C2 server."""
    bots = _category_nodes(graph, "iot_peripheral")[:10]
    c2_server = _category_nodes(graph, "internal_server")[0]

    for bot in bots:
        packet_count = int(rng.integers(300, 501))
        _add_attack_flow(graph, bot, c2_server, packet_count, "botnet_c2")

    # The peer coordination links turn the otherwise star-shaped C2 traffic into
    # a high-k core that can be localized by a k-core detector.
    for index, source in enumerate(bots):
        for target in bots[index + 1 :]:
            packet_count = int(rng.integers(80, 161))
            _add_attack_flow(graph, source, target, packet_count, "botnet_peer")


def inject_mitm(graph: nx.Graph, rng: np.random.Generator) -> None:
    """Route heavy Group-A-to-Group-B virtual traffic through router D."""
    rogue_router = "router_D"
    clients = _category_nodes(graph, "client_workstation")[:30]
    servers = _category_nodes(graph, "internal_server")[:20]

    for index, source in enumerate(clients):
        target = servers[index % len(servers)]
        packet_count = int(rng.integers(250, 451))
        _add_attack_path(graph, source, target, rogue_router, packet_count)


def inject_port_scan(graph: nx.Graph, rng: np.random.Generator) -> None:
    """Add sixty five-packet probes from one client to rotating random nodes."""
    attacker = _category_nodes(graph, "client_workstation")[-1]
    candidates = sorted(
        node
        for node in graph
        if node != attacker and not graph.has_edge(attacker, node)
    )
    targets = rng.choice(candidates, size=60, replace=False)

    for target in targets:
        _add_attack_flow(graph, attacker, str(target), 5, "port_scan")


def inject_scheduled_attacks(
    graph: nx.Graph, rng: np.random.Generator, phase: TimingPhase
) -> None:
    """Inject the attack assigned to this window of the attack phase.

    The four attacks divide the 100-window attack phase into equal, ordered
    intervals.  Actor selection is stable for DDoS, botnet, and MITM so their
    structures persist long enough for detectors to observe them.  Port-scan
    targets rotate each window to produce many short-lived probe edges.
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


def _add_attack_flow(
    graph: nx.Graph,
    source: str,
    target: str,
    packet_count: int,
    attack_type: str,
) -> None:
    """Add one virtual attack edge and its graph-level flow record."""
    if graph.has_edge(source, target):
        attributes = graph.edges[source, target]
        if attributes.get("link_type") != "attack_virtual":
            raise ValueError(f"Attack edge would overwrite physical link {(source, target)}.")
        attributes["weight"] += packet_count
    else:
        graph.add_edge(
            source,
            target,
            link_type="attack_virtual",
            attack_type=attack_type,
            weight=packet_count,
            weight_unit=EDGE_WEIGHT_UNIT,
        )

    graph.graph.setdefault("attack_flows", []).append(
        {
            "source": source,
            "target": target,
            "attack_type": attack_type,
            "packet_count": packet_count,
            "path": (source, target),
            "window": graph.graph["window"],
        }
    )


def _add_attack_path(
    graph: nx.Graph,
    source: str,
    target: str,
    rogue_router: str,
    packet_count: int,
) -> None:
    """Add one source-router-target MITM path and one logical flow record."""
    _add_attack_edge(graph, source, rogue_router, packet_count, "mitm")
    _add_attack_edge(graph, rogue_router, target, packet_count, "mitm")
    graph.graph.setdefault("attack_flows", []).append(
        {
            "source": source,
            "target": target,
            "attack_type": "mitm",
            "packet_count": packet_count,
            "path": (source, rogue_router, target),
            "interceptor": rogue_router,
            "window": graph.graph["window"],
        }
    )


def _add_attack_edge(
    graph: nx.Graph,
    source: str,
    target: str,
    packet_count: int,
    attack_type: str,
) -> None:
    """Add or accumulate one virtual edge without creating a flow record."""
    if graph.has_edge(source, target):
        attributes = graph.edges[source, target]
        if attributes.get("link_type") != "attack_virtual":
            raise ValueError(f"Attack edge would overwrite physical link {(source, target)}.")
        attributes["weight"] += packet_count
        return

    graph.add_edge(
        source,
        target,
        link_type="attack_virtual",
        attack_type=attack_type,
        weight=packet_count,
        weight_unit=EDGE_WEIGHT_UNIT,
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
