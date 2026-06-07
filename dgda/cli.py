"""Command-line interface for running the DGDA simulation."""

import argparse
from collections import Counter
from collections.abc import Iterable

import networkx as nx

from dgda.config import (
    CATEGORY_DISPLAY_NAMES,
    DEVICE_COUNTS,
    ENDPOINT_CATEGORIES,
    ROUTER_RING_EDGES,
    TOTAL_TIME_WINDOWS,
)
from dgda.dynamics import create_dynamic_graph_windows
from dgda.phases import TIMING_PHASES
from dgda.topology import create_network, switch_ids_for_category
from dgda.visualization import (
    animate_dynamic_graph_windows,
    draw_network,
    draw_window_connection_matrix,
    select_window_range,
)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line options for summaries and visualization files."""
    parser = argparse.ArgumentParser(
        description="Build the dynamic graph and render selected visualization windows."
    )
    parser.add_argument(
        "--start-window",
        type=int,
        default=1,
        help="First one-indexed time window to visualize, inclusive (default: 1).",
    )
    parser.add_argument(
        "--end-window",
        type=int,
        default=TOTAL_TIME_WINDOWS,
        help="Last one-indexed time window to visualize, inclusive (default: 500).",
    )
    parser.add_argument(
        "--matrix-window",
        type=int,
        default=None,
        help="One-indexed time window for the adjacency matrix (default: start window).",
    )
    parser.add_argument(
        "--snapshot-output",
        default="network_topology.png",
        help="PNG output path for the first selected window snapshot.",
    )
    parser.add_argument(
        "--matrix-output",
        default="window_connection_matrix.png",
        help="PNG output path for the selected adjacency matrix window.",
    )
    parser.add_argument(
        "--animation-output",
        default="dynamic_graph_windows.gif",
        help="GIF output path for the selected-range animation.",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip GIF rendering when you only need summaries and PNG files.",
    )

    return parser.parse_args()


def print_summary(graph: nx.Graph) -> None:
    """Print a compact summary of the static 200-device topology."""
    category_counts = Counter(nx.get_node_attributes(graph, "category").values())

    print(
        f"Created {graph.number_of_nodes()} devices and {graph.number_of_edges()} links."
    )

    for category, expected_count in DEVICE_COUNTS.items():
        print(
            f"{CATEGORY_DISPLAY_NAMES[category]}: "
            f"{category_counts[category]} / {expected_count}"
        )

    print("Router backbone edges:")
    for source, target in ROUTER_RING_EDGES:
        print(f"  {source} -- {target}")

    print("Endpoint switch assignments:")
    for category in ENDPOINT_CATEGORIES:
        switch_list = ", ".join(switch_ids_for_category(category))
        print(f"  {CATEGORY_DISPLAY_NAMES[category]}: {switch_list}")


def print_dynamic_summary(windows: Iterable[nx.Graph]) -> None:
    """Print a compact summary of the generated dynamic graph windows."""
    all_windows = list(windows)
    phase_counts = Counter()
    node_counts = []
    edge_counts = []
    flow_counts = []
    packet_counts = []

    for graph in all_windows:
        phase_counts[graph.graph["phase"]] += 1
        node_counts.append(graph.number_of_nodes())
        edge_counts.append(graph.number_of_edges())
        flow_counts.append(graph.graph.get("communication_flow_count", 0))
        packet_counts.append(graph.graph.get("communication_packet_count", 0))

    print(f"Generated {len(all_windows)} dynamic graph windows.")

    for phase in TIMING_PHASES:
        print(
            f"{phase.name}: windows {phase.start_window}-{phase.end_window} "
            f"({phase_counts[phase.name]} snapshots)"
        )

    print(
        "Dynamic snapshot range: "
        f"{min(node_counts)}-{max(node_counts)} active nodes, "
        f"{min(edge_counts)}-{max(edge_counts)} physical links, "
        f"{min(flow_counts)}-{max(flow_counts)} routed flows, "
        f"{min(packet_counts)}-{max(packet_counts)} packets."
    )
    print("Attack injection hook enabled only for windows 251-350.")


def main() -> None:
    """Run the full simulation from the command line."""
    args = parse_arguments()
    network = create_network()
    print_summary(network)

    dynamic_windows = create_dynamic_graph_windows()
    print_dynamic_summary(dynamic_windows)

    selected_windows = select_window_range(
        dynamic_windows, args.start_window, args.end_window
    )
    matrix_window = args.matrix_window

    if matrix_window is None:
        matrix_window = args.start_window

    matrix_windows = select_window_range(dynamic_windows, matrix_window, matrix_window)

    print(f"Visualizing windows {args.start_window}-{args.end_window}.")
    draw_network(selected_windows[0], args.snapshot_output)
    print(f"Saved selected-range snapshot visualization to {args.snapshot_output}")

    draw_window_connection_matrix(matrix_windows[0], args.matrix_output)
    print(f"Saved window {matrix_window} connection matrix to {args.matrix_output}")

    if args.skip_animation:
        print("Skipped GIF animation rendering because --skip-animation was provided.")
    else:
        animate_dynamic_graph_windows(
            dynamic_windows,
            args.animation_output,
            start_window=args.start_window,
            end_window=args.end_window,
        )
        print(f"Saved dynamic graph animation to {args.animation_output}")
