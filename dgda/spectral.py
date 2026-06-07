"""Build graph Laplacians and extract the Stage 3/4 spectral features.

The dynamic simulation stores routed packet totals on physical links.  This
module turns those weighted links into one symmetric Laplacian per window,
builds the normal baseline, and compresses each window into five spectral
features used by later detection stages.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

FloatMatrix = NDArray[np.float64]
FloatVector = NDArray[np.float64]


@dataclass(frozen=True)
class LaplacianMatrices:
    """The Stage 3 adjacency, degree, and Laplacian matrices for one window."""

    node_order: tuple[str, ...]
    adjacency: FloatMatrix
    degree: FloatMatrix
    laplacian: FloatMatrix


@dataclass(frozen=True)
class SpectralFeatures:
    """The five-number Stage 4 structural fingerprint for one window."""

    window: int
    phase: str
    fiedler_value: float
    eigengap: float
    fiedler_vector_norm: float
    laplacian_change_norm: float
    cluster_changes: int

    def as_dict(self) -> dict[str, int | float | str]:
        """Return a serialization-friendly feature record."""
        return {
            "window": self.window,
            "phase": self.phase,
            "lambda_2": self.fiedler_value,
            "eigengap": self.eigengap,
            "fiedler_vector_norm": self.fiedler_vector_norm,
            "laplacian_change_norm": self.laplacian_change_norm,
            "cluster_changes": self.cluster_changes,
        }


@dataclass(frozen=True)
class SpectralAnalysis:
    """Baseline structures and Stage 4 fingerprints for a window sequence."""

    node_order: tuple[str, ...]
    baseline_laplacian: FloatMatrix
    baseline_fiedler_vector: FloatVector
    features: tuple[SpectralFeatures, ...]

    def feature_records(self) -> list[dict[str, int | float | str]]:
        """Return all fingerprints as rows suitable for CSV/data-frame output."""
        return [feature.as_dict() for feature in self.features]


def canonical_node_order(graphs: Iterable[nx.Graph]) -> tuple[str, ...]:
    """Return one deterministic node order shared by all matrix operations."""
    nodes = {str(node) for graph in graphs for node in graph.nodes}
    if not nodes:
        raise ValueError("At least one graph with nodes is required.")
    return tuple(sorted(nodes))


def build_laplacian_matrices(
    graph: nx.Graph,
    node_order: Sequence[str] | None = None,
    weight: str = "weight",
) -> LaplacianMatrices:
    """Construct symmetric ``A``, diagonal ``D``, and ``L = D - A``.

    Undirected simulation windows already represent a symmetrized relationship:
    every physical-link weight is the total packets routed over that link in
    either direction.  Directed inputs are explicitly symmetrized by adding the
    two directional weights.  Missing nodes in a shared order become zero rows
    and columns, which keeps every window matrix the same shape.
    """
    ordered_nodes = (
        tuple(node_order) if node_order is not None else tuple(sorted(graph.nodes))
    )
    if not ordered_nodes:
        raise ValueError("Cannot build a Laplacian for an empty node order.")
    if len(set(ordered_nodes)) != len(ordered_nodes):
        raise ValueError("The node order must not contain duplicates.")

    matrix_graph = graph
    missing_nodes = set(ordered_nodes).difference(graph.nodes)
    if missing_nodes:
        matrix_graph = graph.copy()
        matrix_graph.add_nodes_from(missing_nodes)

    adjacency = nx.to_numpy_array(
        matrix_graph,
        nodelist=list(ordered_nodes),
        weight=weight,
        nonedge=0.0,
        dtype=float,
    )
    if graph.is_directed():
        adjacency = adjacency + adjacency.T
    else:
        # Average removes insignificant floating-point asymmetry without
        # doubling the already-symmetrized weights of an undirected graph.
        adjacency = (adjacency + adjacency.T) / 2.0

    np.fill_diagonal(adjacency, 0.0)
    degrees = adjacency.sum(axis=1)
    degree = np.diag(degrees)
    laplacian = degree - adjacency

    return LaplacianMatrices(
        node_order=ordered_nodes,
        adjacency=adjacency,
        degree=degree,
        laplacian=laplacian,
    )


def build_baseline_laplacian(
    graphs: Iterable[nx.Graph],
    node_order: Sequence[str] | None = None,
    baseline_phase: str = "baseline",
) -> FloatMatrix:
    """Average the Laplacians from pure-normal baseline windows cell by cell."""
    graph_list = list(graphs)
    order = (
        tuple(node_order)
        if node_order is not None
        else canonical_node_order(graph_list)
    )
    baseline_laplacians = [
        build_laplacian_matrices(graph, order).laplacian
        for graph in graph_list
        if graph.graph.get("phase") == baseline_phase
    ]
    if not baseline_laplacians:
        raise ValueError(f"No windows found for baseline phase {baseline_phase!r}.")
    return np.mean(baseline_laplacians, axis=0)


def extract_spectral_features(
    laplacian: FloatMatrix,
    baseline_laplacian: FloatMatrix,
    baseline_fiedler_vector: FloatVector,
    *,
    window: int = 0,
    phase: str = "unknown",
    sign_tolerance: float = 1e-10,
) -> SpectralFeatures:
    """Compress one Laplacian into the five Stage 4 spectral features."""
    _validate_square_matrix(laplacian, "laplacian")
    _validate_square_matrix(baseline_laplacian, "baseline_laplacian")
    if laplacian.shape != baseline_laplacian.shape:
        raise ValueError("Current and baseline Laplacians must have the same shape.")
    if laplacian.shape[0] < 3:
        raise ValueError("At least three nodes are required for lambda_2 and lambda_3.")
    if baseline_fiedler_vector.shape != (laplacian.shape[0],):
        raise ValueError("The baseline Fiedler vector length must match the Laplacian.")

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    fiedler_vector = _align_fiedler_vector(eigenvectors[:, 1], baseline_fiedler_vector)
    cluster_changes = _count_sign_changes(
        fiedler_vector, baseline_fiedler_vector, sign_tolerance
    )

    fiedler_value = float(eigenvalues[1])
    third_eigenvalue = float(eigenvalues[2])

    return SpectralFeatures(
        window=window,
        phase=phase,
        fiedler_value=fiedler_value,
        eigengap=third_eigenvalue - fiedler_value,
        fiedler_vector_norm=float(np.linalg.norm(fiedler_vector)),
        laplacian_change_norm=float(np.linalg.norm(laplacian - baseline_laplacian, ord="fro")),
        cluster_changes=cluster_changes,
    )


def analyze_spectral_features(
    graphs: Iterable[nx.Graph], baseline_phase: str = "baseline"
) -> SpectralAnalysis:
    """Run Stages 3 and 4 for every supplied dynamic graph window."""
    graph_list = list(graphs)
    if not graph_list:
        raise ValueError("At least one graph window is required.")

    node_order = canonical_node_order(graph_list)
    laplacians = [
        build_laplacian_matrices(graph, node_order).laplacian for graph in graph_list
    ]
    baseline_indices = [
        index
        for index, graph in enumerate(graph_list)
        if graph.graph.get("phase") == baseline_phase
    ]
    if not baseline_indices:
        raise ValueError(f"No windows found for baseline phase {baseline_phase!r}.")

    baseline_laplacian = np.mean(
        [laplacians[index] for index in baseline_indices], axis=0
    )
    _, baseline_eigenvectors = np.linalg.eigh(baseline_laplacian)
    baseline_fiedler_vector = baseline_eigenvectors[:, 1]

    features = tuple(
        extract_spectral_features(
            laplacian,
            baseline_laplacian,
            baseline_fiedler_vector,
            window=int(graph.graph.get("window", index + 1)),
            phase=str(graph.graph.get("phase", "unknown")),
        )
        for index, (graph, laplacian) in enumerate(zip(graph_list, laplacians))
    )
    return SpectralAnalysis(
        node_order=node_order,
        baseline_laplacian=baseline_laplacian,
        baseline_fiedler_vector=baseline_fiedler_vector,
        features=features,
    )


def _align_fiedler_vector(
    fiedler_vector: FloatVector, baseline_fiedler_vector: FloatVector
) -> FloatVector:
    """Resolve the arbitrary eigenvector sign before baseline comparison."""
    if np.dot(fiedler_vector, baseline_fiedler_vector) < 0:
        return -fiedler_vector
    return fiedler_vector


def _count_sign_changes(
    current: FloatVector, baseline: FloatVector, tolerance: float
) -> int:
    """Count meaningful Fiedler sign flips while ignoring near-zero values."""
    stable = (np.abs(current) > tolerance) & (np.abs(baseline) > tolerance)
    return int(np.count_nonzero(stable & (np.signbit(current) != np.signbit(baseline))))


def _clean_eigenvalue(value: float, tolerance: float = 1e-10) -> float:
    """Convert numerical noise around the guaranteed zero eigenvalue to zero."""
    return 0.0 if abs(value) <= tolerance else float(value)


def _validate_square_matrix(matrix: Any, name: str) -> None:
    """Raise a readable error when a matrix is not finite, square, and symmetric."""
    if (
        not isinstance(matrix, np.ndarray)
        or matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
    ):
        raise ValueError(f"{name} must be a square NumPy matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    if not np.allclose(matrix, matrix.T):
        raise ValueError(f"{name} must be symmetric for eigendecomposition.")
