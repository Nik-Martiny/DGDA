
This project is a continuation of the Spectral Anomaly Detection Algorithm (SADA) paper
written spring 2026.

In this paper, we will cover several novel concepts regarding anomaly detection
through use of Change-Point detection, Community Localization and Node ranking combined
with help from a LLM for anomaly reporting.

The main idea stems from the modeling of a communication network as a graph.

Rather than looking at individual packets, we will treat the entire network as a social network map.

* Every device (PC, server, router) is a **node**
* Every physical cable or routed network hop is an **edge**; endpoint conversations are recorded as routed packet flows over those edges
* The map will **evolve** over time (time-evolving graph)


There are three main components that outline what "detected" means.

* Step 1: Detect "When" something went wrong (Change-Point Detection) 
    * Here we use two alarm systems (CUSUM and Page-Hinkley)
    * Both watch signals derived from the graph and then fire an alert when those signals drift beyond normal bounds
    * Answers the question *"At what moment did the network behavior change?"*
* Step 2: Detect "Where" the attack is happening (Community Localization)
    * **When** an anomalous time window is detected, the system will use the **Fiedler vector** and the **Louvian algorithm** to shape the network into communities.
    * This will group devices that are communicating a lot with each other
    * This answers the question *"Which group of devices is involved?"*
* Step 3: Detect "Who" the key threat actors are (Node Ranking)
    * There are three ranking methods that will identify the most angerous nodes
    * K-core decomposition finds the dense core of the malicious cluster
    * Betweenness centrality identifies any bridge nodes routing the attack traffic across the network
    * Page Rank identifies the most influential/well-connected threat nodes
    * This answers the question *"Which specific devices are driving the attack?"*

Once all of these components are completed the LLM component will be utilized.
It will take all the findings and will write a human-readable report:
* When the anomaly happened
* Which community is affected
* Which nodes are at the highest risk

This will allow analysts who aren't experts in graph theory the ability to 
interpret the data collected and cast better judgment on how to handle the network.

## Initial network simulation

The implementation is split into small files under the `dgda/` package. The
`main.py` file remains as a compatibility entry point for `python main.py` and
older imports. The simulation builds a deterministic NetworkX network with
exactly 200 devices split across the requested categories:

* 80 client workstations
* 40 internal servers
* 30 web/edge servers
* 20 routers/switches
* 30 IoT/peripheral devices

The infrastructure layer contains two five-router rings: `router_A` through
`router_E` and `router_F` through `router_J`. `router_A` also connects directly to
`router_F` as the bridge between the two rings. Each router has one attached switch.
Internal server nodes attach only to `router_F` through `switch_F`, web/edge
server nodes attach only to `router_G` through `switch_G`, and client/IoT devices
attach to client/IoT access switches on routers `A`-`E` and `H`-`J`.
Running the script prints a summary and saves a visualization to
`network_topology.png`.

```bash
python main.py
```


## Code organization

The codebase is intentionally separated by implementation responsibility:

* `dgda/config.py` stores counts, topology rules, traffic rules, random seed values,
  and visualization colors.
* `dgda/phases.py` defines the named timing phases used by the simulation.
* `dgda/topology.py` builds and validates the static 200-device physical network.
* `dgda/dynamics.py` creates time-window snapshots, endpoint churn, normal traffic,
  and the attack injection hook.
* `dgda/visualization.py` contains PNG, heatmap, matrix, layout, and GIF helpers.
* `dgda/cli.py` contains command-line parsing, printed summaries, and the main run
  workflow.
* `main.py` re-exports the public functions so existing examples still work.

The functions include docstrings that explain why each step matters to the
simulation. Comments are kept close to logic that benefits from explanation, while
straightforward Python statements are left readable instead of being over-commented.

## Dynamic timing-window simulation

The dynamic simulation builds 500 one-indexed discrete time windows.
Each window is a NetworkX graph snapshot with stable router/switch infrastructure,
normal endpoint churn, and routed normal communication flows stored as graph
metadata. Endpoint conversations no longer create direct endpoint-to-endpoint
edges; instead, their packet counts are accumulated on the physical access,
switch, router, and backbone links they traverse. The deterministic seed keeps
the dynamic graph reproducible while still allowing active endpoints and routed
traffic loads to vary across time.

The timing layout is:

* **Windows 1-150: baseline phase** — pure normal traffic only so downstream
  algorithms can learn normal graph behavior.
* **Windows 151-250: pre-attack phase** — normal traffic only for validating false
  alarms and calibrating CUSUM/Page-Hinkley thresholds.
* **Windows 251-350: attack phase** — normal traffic plus a reserved attack
  injection hook. No concrete attacks are injected yet, but each snapshot is
  marked with attack-phase ground-truth metadata so future attack mutators can add
  malicious nodes/edges and detectors can be evaluated against the expected alarm
  interval.
* **Windows 351-500: recovery phase** — normal traffic returns, enabling detector
  signal recovery checks and false-positive measurement.

Programmatic use:

```python
from main import create_dynamic_graph_windows

windows = create_dynamic_graph_windows()
attack_window = windows[250]  # Window 251, because the Python list is zero-indexed.
print(attack_window.graph["phase"])
print(attack_window.graph["ground_truth_label"])
```

Future attack implementations can pass an `attack_injector` callback to
`create_dynamic_graph_windows()`. The callback is invoked only for windows 251-350,
which prevents attack traffic from leaking into the baseline, pre-attack, or
recovery phases.

## Routed packet flows

Dynamic windows use `weight` as a simple count of how many packets traverse a
physical link during one time window. The simulation does not add transient
`normal_traffic` edges between endpoint devices. Instead, each endpoint
conversation is recorded in `graph.graph["communication_flows"]` with a source,
destination, traffic profile, packet count, and selected physical path.

Traffic profiles are intentionally different by communicating device type:
IoT telemetry is low volume, IoT-to-client chatter is also light, routine
client-to-internal traffic is moderate, web/edge sessions are burstier, and
server-to-edge synchronization is the heaviest normal communication pattern.
IoT/peripheral devices can communicate with both internal servers and client
workstations, making the client/IoT network more realistic than a server-only
IoT model.

Those endpoint packet counts are routed through the physical network. For each
conversation, the simulation finds the best available physical path from the
source endpoint to its access switch, through routers/switches and backbone
links, and finally to the destination endpoint's access switch. It then adds the
packet count to every physical access, router-to-switch, and backbone edge on
that path. This means a switch-to-router link naturally receives the aggregate
weight of many endpoint conversations, while backbone router links show the
traffic that crosses network segments.

The topology snapshot visualization uses a stable router/switch layout, colors
edges by link type, scales all weighted physical edges by their current packet
load, and labels edges carrying non-zero per-window weight. This makes the saved
`network_topology.png` view a true snapshot of where traffic flows through the
network rather than a set of unrealistic direct device-to-device communication
edges.

## Dynamic graph visualizations

The simulation now includes three complementary visualization helpers for seeing
how the graph changes across the discrete timing windows:

* `animate_dynamic_graph_windows()` renders a Matplotlib `FuncAnimation` GIF that
  advances sequentially through the generated windows. It uses a stable layout so
  devices stay in the same place across frames, phase-colored backgrounds, a
  timeline progress bar, category colors, and separate edge styling for backbone,
  router/switch, and access links.
* `draw_connection_activity_heatmap()` creates a whole-simulation routed-packet
  activity heatmap. Columns are windows 1-500 and rows are physical links that
  carried traffic in at least one selected window, making it easy to see when
  links become busy, idle, or persist across phases.
* `draw_window_connection_matrix()` creates an adjacency-matrix view for a single
  window. Nodes are ordered by device category so dense all-to-all connection
  patterns are easier to inspect than in a crowded node-link drawing.

Running the script writes these visual artifacts for all 500 windows by default.
Rendering a 500-frame GIF can take time, so use `--skip-animation` when you only
need summaries and PNG files:

```bash
python main.py --skip-animation
```

Run without `--skip-animation` when you want the GIF as well:

```bash
python main.py
```

You can choose an inclusive window range from the command line when you only want
to inspect part of the timeline. For example, this renders only the attack-phase
windows 251-350 and uses window 300 for the adjacency matrix:

```bash
python main.py --start-window 251 --end-window 350 --matrix-window 300
```

Outputs:

* `network_topology.png` — node-link snapshot for the first selected window.
* `connection_activity_heatmap.png` — observed edges across the selected window range.
* `window_connection_matrix.png` — all node-to-node links in `--matrix-window`
  (or `--start-window` when omitted).
* `dynamic_graph_windows.gif` — the selected-range sequential FuncAnimation render.

Programmatic use:

```python
from main import (
    animate_dynamic_graph_windows,
    create_dynamic_graph_windows,
    draw_connection_activity_heatmap,
    draw_window_connection_matrix,
    select_window_range,
)

windows = create_dynamic_graph_windows()
attack_windows = select_window_range(windows, 251, 350)

animate_dynamic_graph_windows(
    windows,
    "attack_windows.gif",
    start_window=251,
    end_window=350,
)
draw_connection_activity_heatmap(
    windows,
    "attack_connection_activity.png",
    start_window=251,
    end_window=350,
)
draw_window_connection_matrix(attack_windows[49], "window_300_matrix.png")
```
