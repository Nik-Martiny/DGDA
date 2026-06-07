"""Tests for Stage 3 Laplacian construction and Stage 4 feature extraction."""

import unittest

import networkx as nx
import numpy as np

from dgda.spectral import (
    analyze_spectral_features,
    build_baseline_laplacian,
    build_laplacian_matrices,
    extract_spectral_features,
)


class LaplacianConstructionTests(unittest.TestCase):
    def test_undirected_routed_weights_build_expected_matrices(self) -> None:
        graph = nx.Graph()
        graph.add_edge("router", "switch", weight=250)
        graph.add_edge("switch", "client", weight=50)
        matrices = build_laplacian_matrices(graph, ("client", "router", "switch"))

        expected_adjacency = np.array(
            [[0, 0, 50], [0, 0, 250], [50, 250, 0]], dtype=float
        )
        np.testing.assert_array_equal(matrices.adjacency, expected_adjacency)
        np.testing.assert_array_equal(matrices.degree, np.diag([50, 250, 300]))
        np.testing.assert_array_equal(
            matrices.laplacian, matrices.degree - matrices.adjacency
        )
        np.testing.assert_allclose(matrices.laplacian.sum(axis=1), 0)

    def test_shared_order_keeps_missing_nodes_as_zero_rows(self) -> None:
        graph = nx.Graph()
        graph.add_edge("a", "b", weight=3)

        matrices = build_laplacian_matrices(graph, ("a", "b", "offline"))

        np.testing.assert_array_equal(matrices.adjacency[-1], np.zeros(3))
        np.testing.assert_array_equal(matrices.laplacian[-1], np.zeros(3))

    def test_directed_flows_are_symmetrized(self) -> None:
        graph = nx.DiGraph()
        graph.add_edge("a", "b", weight=200)
        graph.add_edge("b", "a", weight=50)

        matrices = build_laplacian_matrices(graph, ("a", "b"))

        np.testing.assert_array_equal(
            matrices.adjacency, np.array([[0, 250], [250, 0]], dtype=float)
        )

    def test_baseline_is_cellwise_average_of_baseline_windows(self) -> None:
        first = _window(1, "baseline", [("a", "b", 2), ("b", "c", 2)])
        second = _window(2, "baseline", [("a", "b", 4), ("b", "c", 4)])
        ignored = _window(3, "attack", [("a", "b", 100), ("b", "c", 100)])

        baseline = build_baseline_laplacian([first, second, ignored], ("a", "b", "c"))
        expected = build_laplacian_matrices(
            _window(0, "baseline", [("a", "b", 3), ("b", "c", 3)]),
            ("a", "b", "c"),
        ).laplacian
        np.testing.assert_array_equal(baseline, expected)


class SpectralFeatureTests(unittest.TestCase):
    def test_baseline_window_has_expected_five_features(self) -> None:
        graph = _window(1, "baseline", [("a", "b", 2), ("b", "c", 2)])
        laplacian = build_laplacian_matrices(graph, ("a", "b", "c")).laplacian
        _, eigenvectors = np.linalg.eigh(laplacian)

        features = extract_spectral_features(
            laplacian, laplacian, eigenvectors[:, 1], window=1, phase="baseline"
        )

        self.assertAlmostEqual(features.fiedler_value, 2.0)
        self.assertAlmostEqual(features.eigengap, 4.0)
        self.assertAlmostEqual(features.fiedler_vector_norm, 1.0)
        self.assertAlmostEqual(features.laplacian_change_norm, 0.0)
        self.assertEqual(features.cluster_changes, 0)

    def test_fiedler_orientation_does_not_create_false_sign_flips(self) -> None:
        graph = _window(1, "baseline", [("a", "b", 2), ("b", "c", 2)])
        laplacian = build_laplacian_matrices(graph, ("a", "b", "c")).laplacian
        _, eigenvectors = np.linalg.eigh(laplacian)

        features = extract_spectral_features(
            laplacian, laplacian, -eigenvectors[:, 1]
        )

        self.assertEqual(features.cluster_changes, 0)

    def test_analysis_uses_one_order_and_returns_one_record_per_window(self) -> None:
        windows = [
            _window(1, "baseline", [("b", "c", 2), ("a", "b", 2)]),
            _window(2, "baseline", [("a", "b", 4), ("b", "c", 4)]),
            _window(3, "attack", [("a", "c", 20), ("b", "c", 1)]),
        ]

        analysis = analyze_spectral_features(windows)

        self.assertEqual(analysis.node_order, ("a", "b", "c"))
        self.assertEqual(len(analysis.features), 3)
        self.assertEqual(len(analysis.feature_records()), 3)
        self.assertEqual(analysis.features[-1].window, 3)
        self.assertGreater(analysis.features[-1].laplacian_change_norm, 0)


def _window(window: int, phase: str, edges: list[tuple[str, str, int]]) -> nx.Graph:
    graph = nx.Graph(window=window, phase=phase)
    graph.add_weighted_edges_from(edges)
    return graph


if __name__ == "__main__":
    unittest.main()
