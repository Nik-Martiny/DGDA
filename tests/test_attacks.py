"""Tests for the ordered attack injection schedule."""

import unittest

import networkx as nx
import numpy as np

from dgda.attacks import inject_scheduled_attacks
from dgda.phases import TIMING_PHASES
from dgda.topology import create_network


ATTACK_PHASE = next(phase for phase in TIMING_PHASES if phase.name == "attack")


class ScheduledAttackTests(unittest.TestCase):
    def test_ddos_creates_thirty_heavy_edges_to_one_web_server(self) -> None:
        graph = _attack_graph(251)

        inject_scheduled_attacks(graph, np.random.default_rng(1), ATTACK_PHASE)

        edges = _attack_edges(graph)
        self.assertEqual(graph.graph["attack_name"], "ddos")
        self.assertEqual(len(edges), 30)
        self.assertEqual(len({target for _, target, _ in edges}), 1)
        self.assertTrue(all(attributes["weight"] >= 500 for *_, attributes in edges))

    def test_botnet_builds_an_eleven_node_ten_core(self) -> None:
        graph = _attack_graph(276)

        inject_scheduled_attacks(graph, np.random.default_rng(2), ATTACK_PHASE)

        attack_graph = nx.edge_subgraph(
            graph, [(source, target) for source, target, _ in _attack_edges(graph)]
        )
        ten_core = nx.k_core(attack_graph, k=10)
        categories = nx.get_node_attributes(ten_core, "category")
        self.assertEqual(graph.graph["attack_name"], "botnet_c2")
        self.assertEqual(ten_core.number_of_nodes(), 11)
        self.assertEqual(list(categories.values()).count("iot_peripheral"), 10)
        self.assertEqual(list(categories.values()).count("internal_server"), 1)

    def test_mitm_places_router_d_on_every_attack_flow_path(self) -> None:
        graph = _attack_graph(301)

        inject_scheduled_attacks(graph, np.random.default_rng(3), ATTACK_PHASE)

        flows = graph.graph["attack_flows"]
        self.assertEqual(graph.graph["attack_name"], "mitm")
        self.assertEqual(len(flows), 30)
        self.assertTrue(all(flow["path"][1] == "router_D" for flow in flows))
        self.assertEqual(len(set(graph.neighbors("router_D"))) - 3, 50)
        centrality = nx.betweenness_centrality(graph)
        self.assertEqual(max(centrality, key=centrality.get), "router_D")

    def test_port_scan_creates_sixty_five_packet_probe_edges(self) -> None:
        graph = _attack_graph(326)

        inject_scheduled_attacks(graph, np.random.default_rng(4), ATTACK_PHASE)

        edges = _attack_edges(graph)
        sources = {flow["source"] for flow in graph.graph["attack_flows"]}
        self.assertEqual(graph.graph["attack_name"], "port_scan")
        self.assertEqual(len(edges), 60)
        self.assertEqual(len(sources), 1)
        self.assertTrue(all(attributes["weight"] == 5 for *_, attributes in edges))
        self.assertEqual(graph.graph["attack_packet_count"], 300)

    def test_schedule_rejects_windows_outside_the_attack_interval(self) -> None:
        graph = _attack_graph(351)

        with self.assertRaisesRegex(ValueError, "No scheduled attack"):
            inject_scheduled_attacks(graph, np.random.default_rng(5), ATTACK_PHASE)


def _attack_graph(window: int) -> nx.Graph:
    graph = create_network(seed=10)
    graph.graph["window"] = window
    return graph


def _attack_edges(graph: nx.Graph) -> list[tuple[str, str, dict]]:
    return [
        (source, target, attributes)
        for source, target, attributes in graph.edges(data=True)
        if attributes.get("link_type") == "attack_virtual"
    ]


if __name__ == "__main__":
    unittest.main()
