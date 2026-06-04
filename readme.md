
This project is a continuation of the Spectral Anomaly Detection Algorithm (SADA) paper
written spring 2026.

In this paper, we will cover several novel concepts regarding anomaly detection
through use of Change-Point detection, Community Localization and Node ranking combined
with help from a LLM for anomaly reporting.

The main idea stems from the modeling of a communication network as a graph.

Rather than looking at individual packets, we will treat the entire network as a social network map.

* Every device (PC, server, router) is a **node**
* Every communication between two nodes is an **edge**
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

The first implementation step is available in `main.py`. It builds a deterministic
NetworkX simulation with exactly 200 devices split across the requested categories:

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
attach to the remaining second-ring access switches on routers `H`, `I`, and `J`.
Running the script prints a summary and saves a visualization to
`network_topology.png`.

```bash
python main.py
```

## Dynamic timing-window simulation

`main.py` now builds the dynamic graph as 500 one-indexed discrete time windows.
Each window is a NetworkX graph snapshot with stable router/switch infrastructure,
normal endpoint churn, and transient normal communication edges that appear and
disappear between windows. The deterministic seed keeps the dynamic graph
reproducible while still allowing active endpoints and communication links to vary
across time.

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

## Dynamic graph visualizations

The simulation now includes three complementary visualization helpers for seeing
how the graph changes across the discrete timing windows:

* `animate_dynamic_graph_windows()` renders a Matplotlib `FuncAnimation` GIF that
  advances sequentially through the generated windows. It uses a stable layout so
  devices stay in the same place across frames, phase-colored backgrounds, a
  timeline progress bar, category colors, and separate edge styling for backbone,
  router/switch, access, and transient normal-traffic links.
* `draw_connection_activity_heatmap()` creates a whole-simulation edge activity
  heatmap. Columns are windows 1-500 and rows are every unique edge observed in
  the run, making it easy to see when individual connections appear, disappear,
  or persist across phases.
* `draw_window_connection_matrix()` creates an adjacency-matrix view for a single
  window. Nodes are ordered by device category so dense all-to-all connection
  patterns are easier to inspect than in a crowded node-link drawing.

Running the script writes these visual artifacts for all 500 windows by default:

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
