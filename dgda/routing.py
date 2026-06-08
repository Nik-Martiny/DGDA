"""Shared helpers for routing logical traffic over the physical network."""

import networkx as nx

from dgda.config import EDGE_WEIGHT_UNIT


def shortest_physical_path(graph: nx.Graph, source: str, target: str) -> list[str]:
    """Return the shortest path that uses only physical network links."""
    physical_view = nx.subgraph_view(
        graph,
        filter_edge=lambda left, right: graph.edges[left, right].get("link_type")
        in {"access", "router_to_switch", "router_backbone"},
    )
    return nx.shortest_path(physical_view, source, target)


def physical_path_via(
    graph: nx.Graph, source: str, intermediary: str, target: str
) -> list[str]:
    """Return a physical source-to-target path forced through ``intermediary``."""
    first_leg = shortest_physical_path(graph, source, intermediary)
    second_leg = shortest_physical_path(graph, intermediary, target)
    return first_leg + second_leg[1:]


def apply_traffic_weight_to_path(
    graph: nx.Graph, physical_path: list[str], weight: int
) -> None:
    """Add one logical conversation's weight to every physical hop it uses."""
    for source, target in zip(physical_path[:-1], physical_path[1:], strict=True):
        attributes = graph.edges[source, target]
        attributes["weight"] += weight
        attributes["weight_unit"] = EDGE_WEIGHT_UNIT
