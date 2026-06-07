"""Visualization helpers for static and dynamic network graphs."""

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import networkx as nx
import numpy as np

from dgda.config import (
    ANIMATION_FPS,
    ANIMATION_INTERVAL_MS,
    CATEGORY_COLORS,
    CATEGORY_DISPLAY_NAMES,
    INFRASTRUCTURE_CATEGORIES,
    LINK_TYPE_ALPHAS,
    LINK_TYPE_COLORS,
    LINK_TYPE_WIDTHS,
    PHASE_COLORS,
    ROUTER_NAMES,
    SWITCH_NAMES,
)
from dgda.phases import TIMING_PHASES
from dgda.topology import create_network


def draw_network(
    graph: nx.Graph, output_path: str | Path = "network_topology.png"
) -> Path:
    """Render one graph snapshot as a PNG image.

    This view is useful for a quick human check of the simulated topology.  It
    shows routers and switches with larger nodes, endpoint devices with smaller
    nodes, and labels only the infrastructure so the image remains readable.
    """
    output_path = Path(output_path)
    reference_graph = create_network() if "window" in graph.graph else graph
    positions = create_stable_layout(reference_graph)
    node_colors = []
    node_sizes = []

    for node in graph.nodes:
        category = graph.nodes[node]["category"]
        node_colors.append(CATEGORY_COLORS[category])
        node_sizes.append(node_size_for_category(category))

    figure, axis = plt.subplots(figsize=(20, 16), facecolor="#F8FAFC")
    axis.set_facecolor(PHASE_COLORS.get(graph.graph.get("phase"), "#FFFFFF"))

    for link_type in LINK_TYPE_COLORS:
        edges = edges_for_link_type(graph, link_type)
        if not edges:
            continue

        width = [
            edge_width_for_weight(graph, source, target, link_type)
            for source, target in edges
        ]

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edges,
            edge_color=LINK_TYPE_COLORS[link_type],
            width=width,
            alpha=LINK_TYPE_ALPHAS[link_type],
            ax=axis,
        )

    nx.draw_networkx_nodes(
        graph,
        positions,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=0.5,
        edgecolors="white",
        ax=axis,
    )

    draw_infrastructure_labels(graph, positions, axis)
    draw_edge_weight_labels(graph, positions, axis)
    axis.legend(handles=animation_legend_handles(), loc="upper right")
    axis.set_title(network_title(graph))
    axis.axis("off")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)

    return output_path


def select_window_range(
    windows: Iterable[nx.Graph],
    start_window: int | None = None,
    end_window: int | None = None,
) -> list[nx.Graph]:
    """Return snapshots whose window numbers fall inside an inclusive range."""
    all_windows = list(windows)

    if not all_windows:
        raise ValueError("At least one graph window is required to select a range.")

    available_numbers = []
    for graph in all_windows:
        available_numbers.append(graph.graph["window"])

    first_available = min(available_numbers)
    last_available = max(available_numbers)

    if start_window is None:
        start_window = first_available

    if end_window is None:
        end_window = last_available

    if start_window < first_available or end_window > last_available:
        raise ValueError(
            f"Requested windows {start_window}-{end_window}, but available windows "
            f"are {first_available}-{last_available}."
        )

    if start_window > end_window:
        raise ValueError(
            f"Start window {start_window} must be less than or equal to end window {end_window}."
        )

    selected_windows = []
    for graph in all_windows:
        window = graph.graph["window"]
        if start_window <= window <= end_window:
            selected_windows.append(graph)

    if not selected_windows:
        raise ValueError(
            f"No windows found in requested range {start_window}-{end_window}."
        )

    return selected_windows


def create_stable_layout(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    """Return positions that stay fixed across animation frames.

    Stable positions let viewers notice real graph changes instead of layout
    movement.  Routers and switches are placed on two rings.  Endpoint devices
    are then placed just outside the switch they attach to.
    """
    positions = {}
    router_radius = 3.0
    switch_radius = 4.05
    router_count = len(ROUTER_NAMES)

    for index, router_name in enumerate(ROUTER_NAMES):
        angle = 2 * np.pi * index / router_count
        router_id = f"router_{router_name}"
        switch_id = f"switch_{router_name}"

        positions[router_id] = (
            float(router_radius * np.cos(angle)),
            float(router_radius * np.sin(angle)),
        )
        positions[switch_id] = (
            float(switch_radius * np.cos(angle)),
            float(switch_radius * np.sin(angle)),
        )

    nodes_by_switch = {}
    for switch_id in SWITCH_NAMES:
        nodes_by_switch[switch_id] = []

    for node, attributes in graph.nodes(data=True):
        if attributes["category"] in INFRASTRUCTURE_CATEGORIES:
            continue

        attached_switch = attached_switch_for_node(graph, node)
        nodes_by_switch[attached_switch].append(node)

    for switch_id, endpoint_nodes in nodes_by_switch.items():
        place_endpoint_group(graph, positions, switch_id, endpoint_nodes)

    return positions


def animate_dynamic_graph_windows(
    windows: Iterable[nx.Graph],
    output_path: str | Path = "dynamic_graph_windows.gif",
    interval_ms: int = ANIMATION_INTERVAL_MS,
    fps: int = ANIMATION_FPS,
    dpi: int = 120,
    start_window: int | None = None,
    end_window: int | None = None,
) -> Path:
    """Render a selected range of dynamic windows as a GIF animation."""
    selected_windows = select_window_range(windows, start_window, end_window)
    output_path = Path(output_path)
    reference_graph = create_network()
    positions = create_stable_layout(reference_graph)
    all_nodes = list(reference_graph.nodes)
    node_categories = nx.get_node_attributes(reference_graph, "category")
    first_window = selected_windows[0].graph["window"]
    last_window = selected_windows[-1].graph["window"]

    figure, axis = plt.subplots(figsize=(16, 12), facecolor="#F8FAFC")
    plt.subplots_adjust(bottom=0.13)

    def update(frame_index: int):
        graph = selected_windows[frame_index]
        draw_animation_frame(
            axis,
            graph,
            reference_graph,
            positions,
            all_nodes,
            node_categories,
            first_window,
            last_window,
        )
        return axis.collections + axis.lines

    animation = FuncAnimation(
        figure,
        update,
        frames=len(selected_windows),
        interval=interval_ms,
        blit=False,
        repeat=True,
    )

    plt.show()

    plt.close(figure)

    return output_path

def draw_window_connection_matrix(
    graph: nx.Graph,
    output_path: str | Path = "window_connection_matrix.png",
) -> Path:
    """Draw an adjacency matrix for one graph window.

    The same graph can look crowded as a node-link plot.  The matrix view makes
    dense connectivity patterns easier to inspect because every row and column is
    one device and every filled square is one edge.
    """
    output_path = Path(output_path)
    nodes = sorted(graph.nodes, key=lambda node: (graph.nodes[node]["category"], node))
    matrix = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    color_map = ListedColormap(["#F8FAFC", "#0EA5E9"])

    figure, axis = plt.subplots(figsize=(12, 12))
    axis.imshow(matrix, cmap=color_map, interpolation="nearest")
    axis.set_title(f"Window {graph.graph.get('window', '?')} Connection Matrix")
    axis.set_xlabel("Nodes grouped by category")
    axis.set_ylabel("Nodes grouped by category")
    axis.set_xticks([])
    axis.set_yticks([])
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    return output_path


def draw_animation_frame(
    axis: plt.Axes,
    graph: nx.Graph,
    reference_graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    all_nodes: list[str],
    node_categories: dict[str, str],
    first_window: int,
    last_window: int,
) -> None:
    """Draw one frame for the dynamic graph animation."""
    axis.clear()
    phase = graph.graph["phase"]
    active_nodes = set(graph.nodes)
    inactive_nodes = []
    present_nodes = []
    active_colors = []
    active_sizes = []

    for node in all_nodes:
        if node in active_nodes:
            present_nodes.append(node)
            category = node_categories[node]
            active_colors.append(CATEGORY_COLORS[category])
            active_sizes.append(node_size_for_category(category))
        else:
            inactive_nodes.append(node)

    axis.set_facecolor(PHASE_COLORS[phase])
    nx.draw_networkx_nodes(
        reference_graph,
        positions,
        nodelist=inactive_nodes,
        node_color="#CBD5E1",
        node_size=22,
        alpha=0.22,
        linewidths=0,
        ax=axis,
    )

    for link_type in LINK_TYPE_COLORS:
        edges = edges_for_link_type(graph, link_type)
        if not edges:
            continue

        width = [
            edge_width_for_weight(graph, source, target, link_type)
            for source, target in edges
        ]

        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=edges,
            edge_color=LINK_TYPE_COLORS[link_type],
            width=width,
            alpha=LINK_TYPE_ALPHAS[link_type],
            ax=axis,
        )

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=present_nodes,
        node_color=active_colors,
        node_size=active_sizes,
        linewidths=0.65,
        edgecolors="white",
        ax=axis,
    )
    draw_infrastructure_labels(graph, positions, axis)
    draw_timeline_bar(axis, graph.graph["window"], first_window, last_window)
    axis.legend(
        handles=animation_legend_handles(),
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        fontsize=9,
    )
    axis.set_title(
        animation_title(graph, first_window, last_window),
        fontsize=18,
        fontweight="bold",
    )
    axis.axis("off")
    axis.set_aspect("equal")


def edge_width_for_weight(
    graph: nx.Graph, source: str, target: str, link_type: str
) -> float:
    """Scale snapshot and animation edges by their current communication weight."""
    base_width = LINK_TYPE_WIDTHS[link_type]
    weight = graph.edges[source, target].get("weight", 0)

    if weight <= 0:
        return base_width

    return min(base_width + 4.0, base_width + np.log1p(weight) / 2.5)


def draw_edge_weight_labels(
    graph: nx.Graph, positions: dict[str, tuple[float, float]], axis: plt.Axes
) -> None:
    """Draw labels for weighted communication edges in a snapshot."""
    labels = edge_weight_labels(graph)
    if not labels:
        return

    nx.draw_networkx_edge_labels(
        graph,
        positions,
        edge_labels=labels,
        font_size=6,
        font_color="#0F172A",
        rotate=False,
        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.76},
        ax=axis,
    )


def edge_weight_labels(graph: nx.Graph) -> dict[tuple[str, str], str]:
    """Return labels for every edge carrying non-zero routed traffic weight."""
    labels = {}

    for source, target, attributes in graph.edges(data=True):
        weight = attributes.get("weight")
        if weight is None or weight <= 0:
            continue

        labels[(source, target)] = str(int(weight))

    return labels


def attached_switch_for_node(graph: nx.Graph, node: str) -> str:
    """Return the switch connected to an endpoint node."""
    for neighbor in graph.neighbors(node):
        if graph.nodes[neighbor]["category"] == "switch":
            return neighbor

    raise ValueError(f"Endpoint {node} is not connected to a switch.")


def place_endpoint_group(
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    switch_id: str,
    endpoint_nodes: list[str],
) -> None:
    """Place endpoint nodes near their switch in a readable arc."""
    if not endpoint_nodes:
        return

    switch_x, switch_y = positions[switch_id]
    outward_angle = np.arctan2(switch_y, switch_x)
    tangent_angle = outward_angle + np.pi / 2
    endpoint_nodes.sort(key=lambda node: (graph.nodes[node]["category"], node))
    endpoint_count = len(endpoint_nodes)
    endpoint_radius = 0.72 + min(0.72, endpoint_count / 90)

    for index, node in enumerate(endpoint_nodes):
        centered_index = index - (endpoint_count - 1) / 2
        tangent_offset = centered_index * 0.12
        simple_curve = 0.28 * np.sin(index * 1.6)
        x = switch_x + endpoint_radius * np.cos(outward_angle)
        y = switch_y + endpoint_radius * np.sin(outward_angle)
        x = x + tangent_offset * np.cos(tangent_angle)
        y = y + (tangent_offset + simple_curve) * np.sin(tangent_angle)
        positions[node] = (float(x), float(y))


def node_size_for_category(category: str) -> int:
    """Return a readable plot size for a device category."""
    if category == "router":
        return 620

    if category == "switch":
        return 360

    return 70


def draw_infrastructure_labels(
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    axis: plt.Axes,
) -> None:
    """Label only routers and switches so dense endpoint plots stay readable."""
    labels = {}

    for node in graph.nodes:
        category = graph.nodes[node]["category"]
        if category not in INFRASTRUCTURE_CATEGORIES:
            continue

        label = graph.nodes[node]["label"]
        label = label.replace("Router ", "R")
        label = label.replace("Switch ", "S")
        labels[node] = label

    nx.draw_networkx_labels(
        graph, positions, labels=labels, font_size=8, font_weight="bold", ax=axis
    )


def draw_timeline_bar(
    axis: plt.Axes,
    current_window: int,
    selected_start_window: int,
    selected_end_window: int,
) -> None:
    """Draw a phase-colored progress bar below an animation frame."""
    inset = axis.inset_axes([0.05, -0.08, 0.9, 0.045])
    selected_span = selected_end_window - selected_start_window + 1

    for phase in TIMING_PHASES:
        overlap_start = max(phase.start_window, selected_start_window)
        overlap_end = min(phase.end_window, selected_end_window)

        if overlap_start > overlap_end:
            continue

        start = (overlap_start - selected_start_window) / selected_span
        width = (overlap_end - overlap_start + 1) / selected_span
        inset.barh(
            0,
            width,
            left=start,
            height=1,
            color=PHASE_COLORS[phase.name],
            edgecolor="white",
        )
        inset.text(
            start + width / 2,
            0,
            phase.name.replace("_", "\n"),
            ha="center",
            va="center",
            fontsize=8,
        )

    marker_position = (current_window - selected_start_window) / selected_span
    inset.axvline(marker_position, color="#0F172A", linewidth=2.2)
    inset.set_xlim(0, 1)
    inset.set_ylim(-0.5, 0.5)
    inset.axis("off")


def category_legend_handles() -> list[plt.Line2D]:
    """Build legend handles for device categories."""
    handles = []

    for category, color in CATEGORY_COLORS.items():
        handle = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=CATEGORY_DISPLAY_NAMES[category],
            markerfacecolor=color,
            markersize=10,
        )
        handles.append(handle)

    return handles


def animation_legend_handles() -> list[plt.Line2D | Patch]:
    """Build legend handles for animation categories and edge types."""
    handles = list(category_legend_handles())

    for link_type, color in LINK_TYPE_COLORS.items():
        handle = plt.Line2D(
            [0],
            [0],
            color=color,
            lw=max(1.5, LINK_TYPE_WIDTHS[link_type]),
            label=link_type.replace("_", " "),
        )
        handles.append(handle)

    return handles


def edges_for_link_type(graph: nx.Graph, link_type: str) -> list[tuple[str, str]]:
    """Return all edges with a requested link type."""
    edges = []

    for source, target, attributes in graph.edges(data=True):
        if attributes.get("link_type") == link_type:
            edges.append((source, target))

    return edges


def unique_edges(windows: list[nx.Graph]) -> list[tuple[str, str]]:
    """Return every physical link that carried routed packets in any window."""
    edges = set()

    for graph in windows:
        for source, target, attributes in graph.edges(data=True):
            if attributes.get("weight", 0) <= 0:
                continue

            edges.add(ordered_edge(source, target))

    return sorted(edges)


def ordered_edge(source: str, target: str) -> tuple[str, str]:
    """Return an undirected edge in a stable order."""
    if source <= target:
        return (source, target)

    return (target, source)


def network_title(graph: nx.Graph) -> str:
    """Return a descriptive title for a static snapshot plot."""
    title = "200-Device Communication Network Simulation"

    if "window" in graph.graph:
        title = f"{title} - Window {graph.graph['window']} ({graph.graph['phase']})"

    return title


def animation_title(graph: nx.Graph, first_window: int, last_window: int) -> str:
    """Return the title text for one animation frame."""
    phase = graph.graph["phase"].replace("_", " ").title()
    window = graph.graph["window"]
    active_nodes = graph.number_of_nodes()
    active_edges = graph.number_of_edges()
    flow_count = graph.graph.get("communication_flow_count", 0)
    packet_count = graph.graph.get("communication_packet_count", 0)

    return (
        f"Dynamic Communication Network — Window {window:03d}/500 ({phase})\n"
        f"Showing windows {first_window}-{last_window} • "
        f"{active_nodes} active nodes • {active_edges} physical links • "
        f"{flow_count} routed flows / {packet_count} packets"
    )
