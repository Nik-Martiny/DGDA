"""Configuration values used throughout the network simulation.

Keeping constants in one file makes the simulation easier to tune.  A reader can
change device counts, traffic rules, colors, or random seeds here without needing
to search through graph-building and plotting code.
"""

TOTAL_DEVICES = 200

DEVICE_COUNTS = {
    "client_workstation": 80,
    "internal_server": 40,
    "web_edge_server": 30,
    "router": 10,
    "switch": 10,
    "iot_peripheral": 30,
}

ROUTER_NAMES = tuple("ABCDEFGHIJ")
SWITCH_NAMES = tuple(f"switch_{router_name}" for router_name in ROUTER_NAMES)

INTERNAL_SERVER_ROUTER = "F"
WEB_EDGE_SERVER_ROUTER = "G"
CLIENT_IOT_ROUTERS = ROUTER_NAMES[:5] + ROUTER_NAMES[7:]

CATEGORY_ROUTER_ASSIGNMENTS = {
    "client_workstation": CLIENT_IOT_ROUTERS,
    "internal_server": (INTERNAL_SERVER_ROUTER,),
    "web_edge_server": (WEB_EDGE_SERVER_ROUTER,),
    "iot_peripheral": CLIENT_IOT_ROUTERS,
}

RNG_SEED = 42
TOTAL_TIME_WINDOWS = 500
ANIMATION_INTERVAL_MS = 90
ANIMATION_FPS = 12

CATEGORY_DISPLAY_NAMES = {
    "client_workstation": "Client Workstations",
    "internal_server": "Internal Servers",
    "web_edge_server": "Web/Edge Servers",
    "router": "Routers",
    "switch": "Switches",
    "iot_peripheral": "IoT/Peripheral Devices",
}

CATEGORY_COLORS = {
    "client_workstation": "#4E79A7",
    "internal_server": "#59A14F",
    "web_edge_server": "#F28E2B",
    "router": "#E15759",
    "switch": "#A1B213",
    "iot_peripheral": "#B07AA1",
}

PHASE_COLORS = {
    "baseline": "#E8F4FF",
    "pre_attack": "#FFF5D6",
    "attack": "#FFE3E0",
    "recovery": "#E6F6EA",
}

LINK_TYPE_COLORS = {
    "router_backbone": "#1F2937",
    "router_to_switch": "#6B7280",
    "access": "#D1D5DB",
    "normal_traffic": "#38BDF8",
}

LINK_TYPE_WIDTHS = {
    "router_backbone": 2.8,
    "router_to_switch": 1.8,
    "access": 0.55,
    "normal_traffic": 0.45,
}

LINK_TYPE_ALPHAS = {
    "router_backbone": 0.9,
    "router_to_switch": 0.65,
    "access": 0.24,
    "normal_traffic": 0.55,
}

ENDPOINT_CATEGORIES = (
    "client_workstation",
    "internal_server",
    "web_edge_server",
    "iot_peripheral",
)

INFRASTRUCTURE_CATEGORIES = ("router", "switch")

ENDPOINT_PREFIXES = {
    "client_workstation": "client",
    "internal_server": "internal-server",
    "web_edge_server": "web-edge-server",
    "iot_peripheral": "iot-peripheral",
}

ENDPOINT_UP_PROBABILITIES = {
    "client_workstation": 0.92,
    "internal_server": 0.99,
    "web_edge_server": 0.99,
    "iot_peripheral": 0.84,
}

NORMAL_TRAFFIC_RULES = (
    ("client_workstation", "internal_server", 70, 115, "client_to_internal"),
    ("client_workstation", "web_edge_server", 55, 95, "client_to_web_edge"),
    ("iot_peripheral", "internal_server", 20, 45, "iot_to_internal"),
    ("internal_server", "web_edge_server", 15, 35, "server_to_edge"),
    ("client_workstation", "client_workstation", 8, 18, "client_peer"),
)

# Edge weights are packet-load measurements for a single dynamic time window.
# Physical edges start each window at zero and accrue packet/byte load for every
# routed conversation that traverses them.  Transient normal-traffic edges keep
# the same packet count as their ``weight`` so downstream packet models can use
# one numeric attribute consistently across all edge types.
EDGE_WEIGHT_UNIT = "packets_per_window"
TRAFFIC_WINDOW_SECONDS = 60

# Real networks carry sharply different volumes depending on which device types
# communicate.  IoT telemetry is intentionally light, client-to-server requests
# are moderate, public web/edge services can be burstier, and server-to-edge
# synchronization is the heaviest normal profile represented here.
TRAFFIC_PACKET_PROFILES = {
    "iot_to_internal": {
        "packet_count_range": (4, 80),
        "avg_packet_size_bytes_range": (96, 384),
        "description": "Low-rate IoT telemetry and status checks",
    },
    "client_peer": {
        "packet_count_range": (20, 260),
        "avg_packet_size_bytes_range": (256, 900),
        "description": "Light workstation-to-workstation collaboration traffic",
    },
    "client_to_internal": {
        "packet_count_range": (80, 950),
        "avg_packet_size_bytes_range": (384, 1200),
        "description": "Business application requests to internal services",
    },
    "client_to_web_edge": {
        "packet_count_range": (120, 1400),
        "avg_packet_size_bytes_range": (512, 1400),
        "description": "Browser/API sessions to public web or edge servers",
    },
    "server_to_edge": {
        "packet_count_range": (750, 6500),
        "avg_packet_size_bytes_range": (900, 1500),
        "description": "Heavy server synchronization, replication, or content updates",
    },
}

PHYSICAL_LINK_CAPACITY_MBPS = {
    "access": 100,
    "router_to_switch": 1000,
    "router_backbone": 10000,
}

ROUTER_RING_EDGES = (
    ("router_A", "router_B"),
    ("router_B", "router_C"),
    ("router_C", "router_D"),
    ("router_D", "router_E"),
    ("router_E", "router_F"),
    ("router_F", "router_G"),
    ("router_G", "router_H"),
    ("router_H", "router_I"),
    ("router_I", "router_J"),
    ("router_J", "router_A"),
)
