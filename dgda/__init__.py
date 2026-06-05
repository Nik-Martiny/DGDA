"""Public entry points for the Dynamic Graph Device Analytics simulation.

The package is split by responsibility so readers can open one small file at a
time: configuration constants, timing phases, topology building, dynamic window
creation, visualization, and command-line behavior.
"""

from dgda.config import (
    ANIMATION_FPS,
    ANIMATION_INTERVAL_MS,
    CATEGORY_COLORS,
    CATEGORY_DISPLAY_NAMES,
    DEVICE_COUNTS,
    ENDPOINT_CATEGORIES,
    INFRASTRUCTURE_CATEGORIES,
    LINK_TYPE_ALPHAS,
    LINK_TYPE_COLORS,
    LINK_TYPE_WIDTHS,
    NORMAL_TRAFFIC_RULES,
    PHASE_COLORS,
    RNG_SEED,
    ROUTER_NAMES,
    ROUTER_RING_EDGES,
    SWITCH_NAMES,
    TOTAL_DEVICES,
    TOTAL_TIME_WINDOWS,
)
from dgda.dynamics import create_dynamic_graph_windows, phase_for_window
from dgda.phases import TIMING_PHASES, TimingPhase
from dgda.topology import create_network
from dgda.visualization import (
    animate_dynamic_graph_windows,
    create_stable_layout,
    draw_connection_activity_heatmap,
    draw_network,
    draw_window_connection_matrix,
    select_window_range,
)

__all__ = [
    "ANIMATION_FPS",
    "ANIMATION_INTERVAL_MS",
    "CATEGORY_COLORS",
    "CATEGORY_DISPLAY_NAMES",
    "DEVICE_COUNTS",
    "ENDPOINT_CATEGORIES",
    "INFRASTRUCTURE_CATEGORIES",
    "LINK_TYPE_ALPHAS",
    "LINK_TYPE_COLORS",
    "LINK_TYPE_WIDTHS",
    "NORMAL_TRAFFIC_RULES",
    "PHASE_COLORS",
    "RNG_SEED",
    "ROUTER_NAMES",
    "ROUTER_RING_EDGES",
    "SWITCH_NAMES",
    "TIMING_PHASES",
    "TOTAL_DEVICES",
    "TOTAL_TIME_WINDOWS",
    "TimingPhase",
    "animate_dynamic_graph_windows",
    "create_dynamic_graph_windows",
    "create_network",
    "create_stable_layout",
    "draw_connection_activity_heatmap",
    "draw_network",
    "draw_window_connection_matrix",
    "phase_for_window",
    "select_window_range",
]
