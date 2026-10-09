# Multiplex Graph Visualization

This project analyzes and visualizes multilayer graph edge-list datasets such as:

- `CS-Aarhus_multiplex-Copy1.edges`
- `drosophila_multiplex-Copy1.edges`
- `homo_multiplex_2-Copy1.edges`
- `Kapferer-Tailor-Shop_multiplex-Copy1.edges`
- `pierreauger_multiplex-Copy1.edges`
- `sacchpomb_multiplex_2-Copy1.edges`
- `Vickers-Chan-7thGraders_multiplex-Copy1.edges`

## Input Format

Each `.edges` row is expected to have four whitespace-separated columns:

```text
layer_id source_node target_node weight
```

Example:

```text
1 10 14 1
```

This means node `10` is connected to node `14` in layer `1` with weight `1`.

## What the Program Produces

For each dataset, `multiplex_visualizer.py` creates:

- An interactive light-theme HTML report with layer colors, mode controls, bridge/isolated-node filters, layer profile bars, and summary tables.
- A CSV ranking of influential nodes.
- A CSV summary of each layer.
- A combined `index.html` when processing a folder of datasets.

The influence score combines:

- Total node strength: 35%
- PageRank: 25%
- Betweenness centrality: 20%
- Number of active layers: 10%
- Layer participation: 10%

The metrics are computed on the full graph. The HTML network drawing is intentionally limited to the highest-ranked nodes and their strongest local neighborhood so large graphs remain readable.

## Visualization Modes

The report includes four main modes:

- **All layers merged**: shows the aggregate network using each node's dominant layer color.
- **One layer - intra edges**: shows only the selected layer and only edges inside that layer.
- **Multiple layers - intra and inter edges**: shows selected layers as separate 2D panels. Intra-layer edges stay inside each panel, while dashed inter-layer edges connect the same node across layer copies.
- **Influence mode**: colors nodes by influence intensity. The layer color identifies the layer; darker color means more influential. Use the influence scope control to switch between one selected layer and all selected layers.

Bridge and isolated-node controls are available in the report:

- **Highlight bridges**: darkens bridge edges.
- **Only bridges**: hides non-bridge edges.
- **Show isolated nodes**: keeps nodes visible even when they have no visible edge in the current view.

By default, reports are generated as undirected graphs. Use `--directed` only when you intentionally want to preserve `source -> target` direction and draw arrows.

## Run It

From this folder:

```powershell
python .\multiplex_visualizer.py --data-dir "C:\Users\RISHU_SINGH\Documents\Coding Dell\B.tech Projects\Implementation\datasets" --output .\outputs
```

Open:

```text
outputs\index.html
```

To analyze one dataset:

```powershell
python .\multiplex_visualizer.py --input "C:\Users\RISHU_SINGH\Documents\Coding Dell\B.tech Projects\Implementation\datasets\CS-Aarhus_multiplex-Copy1.edges" --output .\outputs
```

By default, the program treats edges as undirected, which is usually the better interpretation for these multiplex social, collaboration, and biological interaction datasets:

```powershell
python .\multiplex_visualizer.py --data-dir "C:\Users\RISHU_SINGH\Documents\Coding Dell\B.tech Projects\Implementation\datasets" --output .\outputs
```

For directed analysis:

```powershell
python .\multiplex_visualizer.py --data-dir "C:\Users\RISHU_SINGH\Documents\Coding Dell\B.tech Projects\Implementation\datasets" --output .\outputs --directed
```

## Useful Options

```text
--top-k 20                 Number of highest-ranked nodes to highlight
--max-nodes 160            Maximum nodes drawn in the report network
--max-edges-per-layer 1400 Maximum edges drawn per selected layer
--betweenness-samples 128  Approximate betweenness pivots for large graphs
--directed                 Preserve source-to-target direction and draw arrows
--undirected               Explicitly request undirected mode, which is already the default
```

Use `--betweenness-samples 0` for exact betweenness centrality. This can be slow for the larger biological graphs.

## Notes

The script uses only Python's standard library. No `networkx`, `matplotlib`, or `plotly` installation is required.
