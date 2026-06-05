"""Compatibility entry point for the DGDA network simulation.

Most implementation code now lives in the ``dgda`` package.  This small module
keeps older examples such as ``python main.py`` and ``from main import
create_network`` working while the codebase stays split by responsibility.
"""

from dgda import (  # noqa: F401
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
    TIMING_PHASES,
    TOTAL_DEVICES,
    TOTAL_TIME_WINDOWS,
    TimingPhase,
    animate_dynamic_graph_windows,
    create_dynamic_graph_windows,
    create_network,
    create_stable_layout,
    draw_connection_activity_heatmap,
    draw_network,
    draw_window_connection_matrix,
    phase_for_window,
    select_window_range,
)
from dgda.cli import main


if __name__ == "__main__":
    main()
