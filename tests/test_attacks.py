"""Tests for the ordered, physically routed attack injection schedule."""

import unittest

import networkx as nx
import numpy as np

from dgda.attacks import inject_scheduled_attacks
from dgda.phases import TIMING_PHASES
from dgda.topology import create_network


ATTACK_PHASE = next(phase for phase in TIMING_PHASES if phase.name == "attack")
PHYSICAL_LINK_TYPES = {"access", "router_to_switch", "router_backbone"}


class ScheduledAttackTests(unittest.TestCase):
    def test_ddos_routes_thirty_heavy_flows_to_one_web_server(self) -> None:
        graph = _attack_graph(251)
        original_edges = set(graph.edges)

        inject_scheduled_attacks(graph, np.random.default_rng(1), ATTACK_PHASE)

        flows = graph.graph["attack_flows"]
        targets = {flow["target"] for flow in flows}
        target = next(iter(targets))
        target_access_edge = next(iter(graph.neighbors(target)))
        self.assertEqual(graph.graph["attack_name"], "ddos")
        self.assertEqual(len(flows), 30)
        self.assertEqual(len(targets), 1)
        self.assertTrue(all(flow["packet_count"] >= 500 for flow in flows))
        self.assertEqual(
            graph.edges[target, target_access_edge]["weight"],
            graph.graph["attack_packet_count"],
        )
        self.assertEqual(set(graph.edges), original_edges)

    def test_botnet_keeps_dense_logical_core_but_routes_over_physical_links(self) -> None:
        graph = _attack_graph(276)
        original_edges = set(graph.edges)

        inject_scheduled_attacks(graph, np.random.default_rng(2), ATTACK_PHASE)

        logical_graph = nx.Graph(
            (flow["source"], flow["target"]) for flow in graph.graph["attack_flows"]
        )
        ten_core = nx.k_core(logical_graph, k=10)
        categories = nx.get_node_attributes(graph, "category")
        self.assertEqual(graph.graph["attack_name"], "botnet_c2")
        self.assertEqual(ten_core.number_of_nodes(), 11)
        self.assertEqual(
            sum(categories[node] == "iot_peripheral" for node in ten_core), 10
        )
        self.assertEqual(
            sum(categories[node] == "internal_server" for node in ten_core), 1
        )
        self.assertEqual(set(graph.edges), original_edges)
        _assert_all_flows_use_physical_links(self, graph)

    def test_mitm_routes_every_flow_through_physical_router_d(self) -> None:
        graph = _attack_graph(301)
        original_edges = set(graph.edges)
        original_neighbors = set(graph.neighbors("router_D"))

        inject_scheduled_attacks(graph, np.random.default_rng(3), ATTACK_PHASE)

        flows = graph.graph["attack_flows"]
        router_d_load = sum(
            graph.edges["router_D", node]["weight"]
            for node in graph.neighbors("router_D")
        )
        self.assertEqual(graph.graph["attack_name"], "mitm")
        self.assertEqual(len(flows), 30)
        self.assertTrue(all("router_D" in flow["path"][1:-1] for flow in flows))
        self.assertGreaterEqual(router_d_load, graph.graph["attack_packet_count"] * 2)
        self.assertEqual(set(graph.neighbors("router_D")), original_neighbors)
        self.assertEqual(set(graph.edges), original_edges)
        _assert_all_flows_use_physical_links(self, graph)

    def test_port_scan_routes_sixty_five_packet_probes_without_new_edges(self) -> None:
        graph = _attack_graph(326)
        original_edges = set(graph.edges)

        inject_scheduled_attacks(graph, np.random.default_rng(4), ATTACK_PHASE)

        flows = graph.graph["attack_flows"]
        sources = {flow["source"] for flow in flows}
        self.assertEqual(graph.graph["attack_name"], "port_scan")
        self.assertEqual(len(flows), 60)
        self.assertEqual(len(sources), 1)
        self.assertTrue(all(flow["packet_count"] == 5 for flow in flows))
        self.assertEqual(graph.graph["attack_packet_count"], 300)
        self.assertEqual(set(graph.edges), original_edges)
        _assert_all_flows_use_physical_links(self, graph)

    def test_schedule_rejects_windows_outside_the_attack_interval(self) -> None:
        graph = _attack_graph(351)

        with self.assertRaisesRegex(ValueError, "No scheduled attack"):
            inject_scheduled_attacks(graph, np.random.default_rng(5), ATTACK_PHASE)


def _attack_graph(window: int) -> nx.Graph:
    graph = create_network(seed=10)
    graph.graph["window"] = window
    return graph


def _assert_all_flows_use_physical_links(
    test_case: unittest.TestCase, graph: nx.Graph
) -> None:
    for flow in graph.graph["attack_flows"]:
        for source, target in zip(flow["path"][:-1], flow["path"][1:], strict=True):
            test_case.assertTrue(graph.has_edge(source, target))
            test_case.assertIn(graph.edges[source, target]["link_type"], PHYSICAL_LINK_TYPES)


if __name__ == "__main__":
    unittest.main()
