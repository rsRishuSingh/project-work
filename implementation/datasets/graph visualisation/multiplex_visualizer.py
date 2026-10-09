#!/usr/bin/env python3
"""
Visualize and rank influential nodes in multiplex / multilayer edge-list graphs.

Input edge format:
    layer_id source_node target_node weight

The program writes:
    - <dataset>_influence.csv
    - <dataset>_layer_summary.csv
    - <dataset>_report.html
    - index.html when processing a directory

No third-party Python packages are required.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Sequence, Set, Tuple


Adjacency = Dict[str, Dict[str, float]]


@dataclass(frozen=True)
class Edge:
    layer: str
    source: str
    target: str
    weight: float


@dataclass
class GraphData:
    name: str
    path: Path
    directed: bool
    nodes: Set[str]
    edges: List[Edge]
    layer_edges: Dict[str, List[Edge]]
    layer_adj: Dict[str, Adjacency]
    aggregate_adj: Adjacency


def natural_key(value: str) -> Tuple[int, object]:
    """Sort numeric-looking ids numerically and all other ids lexically."""
    try:
        return (0, int(value))
    except ValueError:
        try:
            return (0, float(value))
        except ValueError:
            return (1, value.lower())


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value or "graph"


def format_number(value: float, decimals: int = 4) -> str:
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 10:
        return f"{value:,.2f}"
    return f"{value:,.{decimals}f}"


def parse_edges(path: Path) -> List[Edge]:
    edges: List[Edge] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 4:
                raise ValueError(
                    f"{path.name}:{line_number} has {len(parts)} column(s); "
                    "expected: layer source target weight"
                )
            layer, source, target, weight_text = parts[:4]
            try:
                weight = float(weight_text)
            except ValueError as exc:
                raise ValueError(
                    f"{path.name}:{line_number} has invalid weight {weight_text!r}"
                ) from exc
            edges.append(Edge(layer=layer, source=source, target=target, weight=weight))
    if not edges:
        raise ValueError(f"{path} does not contain any edges")
    return edges


def add_weight(adj: Adjacency, source: str, target: str, weight: float) -> None:
    adj.setdefault(source, {})
    adj.setdefault(target, {})
    adj[source][target] = adj[source].get(target, 0.0) + weight


def build_graph(path: Path, directed: bool) -> GraphData:
    edges = parse_edges(path)
    nodes: Set[str] = set()
    layer_edges: Dict[str, List[Edge]] = defaultdict(list)
    layer_adj: Dict[str, Adjacency] = defaultdict(dict)
    aggregate_adj: Adjacency = {}

    for edge in edges:
        nodes.add(edge.source)
        nodes.add(edge.target)
        layer_edges[edge.layer].append(edge)
        add_weight(layer_adj[edge.layer], edge.source, edge.target, edge.weight)
        add_weight(aggregate_adj, edge.source, edge.target, edge.weight)
        if not directed and edge.source != edge.target:
            add_weight(layer_adj[edge.layer], edge.target, edge.source, edge.weight)
            add_weight(aggregate_adj, edge.target, edge.source, edge.weight)

    for node in nodes:
        aggregate_adj.setdefault(node, {})
        for layer in layer_edges:
            layer_adj[layer].setdefault(node, {})

    return GraphData(
        name=path.stem,
        path=path,
        directed=directed,
        nodes=nodes,
        edges=edges,
        layer_edges=dict(layer_edges),
        layer_adj=dict(layer_adj),
        aggregate_adj=aggregate_adj,
    )


def reverse_adjacency(adj: Adjacency, nodes: Iterable[str]) -> Adjacency:
    reverse: Adjacency = {node: {} for node in nodes}
    for source, neighbors in adj.items():
        reverse.setdefault(source, {})
        for target, weight in neighbors.items():
            reverse.setdefault(target, {})
            reverse[target][source] = reverse[target].get(source, 0.0) + weight
    return reverse


def node_strengths(
    adj: Adjacency, nodes: Iterable[str], directed: bool
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, int]]:
    out_strength = {node: 0.0 for node in nodes}
    in_strength = {node: 0.0 for node in nodes}
    total_strength = {node: 0.0 for node in nodes}
    degree = {node: 0 for node in nodes}
    reverse = reverse_adjacency(adj, nodes)

    for node in nodes:
        out_neighbors = {target: weight for target, weight in adj.get(node, {}).items() if target != node}
        in_neighbors = {source: weight for source, weight in reverse.get(node, {}).items() if source != node}
        out_strength[node] = sum(max(weight, 0.0) for weight in out_neighbors.values())
        in_strength[node] = sum(max(weight, 0.0) for weight in in_neighbors.values())
        if directed:
            total_strength[node] = out_strength[node] + in_strength[node]
            degree[node] = len(set(out_neighbors) | set(in_neighbors))
        else:
            total_strength[node] = out_strength[node]
            degree[node] = len(out_neighbors)

    return out_strength, in_strength, total_strength, degree


def weighted_pagerank(
    adj: Adjacency,
    nodes: Sequence[str],
    damping: float = 0.85,
    max_iter: int = 120,
    tolerance: float = 1.0e-9,
) -> Dict[str, float]:
    if not nodes:
        return {}

    node_count = len(nodes)
    rank = {node: 1.0 / node_count for node in nodes}
    base = (1.0 - damping) / node_count

    out_weight = {
        node: sum(max(weight, 0.0) for target, weight in adj.get(node, {}).items() if target != node)
        for node in nodes
    }

    for _ in range(max_iter):
        next_rank = {node: base for node in nodes}
        dangling_total = sum(rank[node] for node in nodes if out_weight[node] == 0.0)
        dangling_share = damping * dangling_total / node_count

        for node in nodes:
            next_rank[node] += dangling_share

        for source in nodes:
            if out_weight[source] == 0.0:
                continue
            share = damping * rank[source] / out_weight[source]
            for target, weight in adj.get(source, {}).items():
                if target == source or weight <= 0.0:
                    continue
                next_rank[target] += share * weight

        error = sum(abs(next_rank[node] - rank[node]) for node in nodes)
        rank = next_rank
        if error < tolerance:
            break
    return rank


def betweenness_centrality(
    adj: Adjacency,
    nodes: Sequence[str],
    directed: bool,
    sample_size: int,
    seed: int,
    exact_threshold: int = 500,
) -> Dict[str, float]:
    """Brandes betweenness centrality on an unweighted aggregate graph."""
    if not nodes:
        return {}

    neighbors = {
        node: [target for target in adj.get(node, {}) if target != node]
        for node in nodes
    }

    if len(nodes) <= exact_threshold or sample_size <= 0 or sample_size >= len(nodes):
        sources = list(nodes)
        sample_scale = 1.0
    else:
        rng = random.Random(seed)
        sources = rng.sample(list(nodes), sample_size)
        sample_scale = len(nodes) / len(sources)

    centrality = {node: 0.0 for node in nodes}
    for source in sources:
        stack: List[str] = []
        predecessors: Dict[str, List[str]] = {node: [] for node in nodes}
        sigma = {node: 0.0 for node in nodes}
        sigma[source] = 1.0
        distance = {node: -1 for node in nodes}
        distance[source] = 0

        queue: deque[str] = deque([source])
        while queue:
            vertex = queue.popleft()
            stack.append(vertex)
            for neighbor in neighbors.get(vertex, []):
                if distance[neighbor] < 0:
                    queue.append(neighbor)
                    distance[neighbor] = distance[vertex] + 1
                if distance[neighbor] == distance[vertex] + 1:
                    sigma[neighbor] += sigma[vertex]
                    predecessors[neighbor].append(vertex)

        delta = {node: 0.0 for node in nodes}
        while stack:
            vertex = stack.pop()
            if sigma[vertex] > 0.0:
                factor = (1.0 + delta[vertex]) / sigma[vertex]
                for predecessor in predecessors[vertex]:
                    delta[predecessor] += sigma[predecessor] * factor
            if vertex != source:
                centrality[vertex] += delta[vertex]

    for node in centrality:
        centrality[node] *= sample_scale
        if not directed:
            centrality[node] /= 2.0
    return centrality


def max_normalize(values: Dict[str, float]) -> Dict[str, float]:
    maximum = max(values.values(), default=0.0)
    if maximum <= 0.0:
        return {key: 0.0 for key in values}
    return {key: value / maximum for key, value in values.items()}


def compute_layer_strengths(
    graph: GraphData, nodes: Sequence[str]
) -> Tuple[Dict[str, Counter], Dict[str, int], Dict[str, float]]:
    by_node: Dict[str, Counter] = {node: Counter() for node in nodes}
    layer_count = {node: 0 for node in nodes}
    participation = {node: 0.0 for node in nodes}

    for layer, adj in graph.layer_adj.items():
        _, _, total_strength, _ = node_strengths(adj, nodes, graph.directed)
        for node, strength in total_strength.items():
            if strength > 0.0:
                by_node[node][layer] = strength

    total_layers = max(1, len(graph.layer_adj))
    for node, strengths in by_node.items():
        layer_count[node] = len(strengths)
        total = sum(strengths.values())
        if total > 0.0 and total_layers > 1:
            raw_participation = 1.0 - sum((strength / total) ** 2 for strength in strengths.values())
            participation[node] = (total_layers / (total_layers - 1.0)) * raw_participation
    return by_node, layer_count, participation


def compute_influence_metrics(
    graph: GraphData, betweenness_samples: int, seed: int
) -> List[Dict[str, object]]:
    nodes = sorted(graph.nodes, key=natural_key)
    out_strength, in_strength, total_strength, degree = node_strengths(
        graph.aggregate_adj, nodes, graph.directed
    )
    pagerank = weighted_pagerank(graph.aggregate_adj, nodes)
    betweenness = betweenness_centrality(
        graph.aggregate_adj,
        nodes,
        directed=graph.directed,
        sample_size=betweenness_samples,
        seed=seed,
    )
    layer_strengths, layer_count, participation = compute_layer_strengths(graph, nodes)

    strength_norm = max_normalize(total_strength)
    pagerank_norm = max_normalize(pagerank)
    betweenness_norm = max_normalize(betweenness)
    layer_count_norm = {
        node: layer_count[node] / max(1, len(graph.layer_adj)) for node in nodes
    }
    participation_norm = max_normalize(participation)

    rows: List[Dict[str, object]] = []
    for node in nodes:
        influence_score = (
            0.35 * strength_norm[node]
            + 0.25 * pagerank_norm[node]
            + 0.20 * betweenness_norm[node]
            + 0.10 * layer_count_norm[node]
            + 0.10 * participation_norm[node]
        )
        rows.append(
            {
                "node": node,
                "influence_score": influence_score,
                "total_strength": total_strength[node],
                "out_strength": out_strength[node],
                "in_strength": in_strength[node],
                "degree": degree[node],
                "pagerank": pagerank[node],
                "betweenness": betweenness[node],
                "layer_count": layer_count[node],
                "participation": participation[node],
                "layer_strengths": dict(layer_strengths[node]),
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["influence_score"]),
            -float(row["total_strength"]),
            natural_key(str(row["node"])),
        )
    )
    for index, row in enumerate(rows, 1):
        row["rank"] = index
    return rows


def summarize_layers(graph: GraphData) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for layer in sorted(graph.layer_edges, key=natural_key):
        edges = graph.layer_edges[layer]
        layer_nodes = {edge.source for edge in edges} | {edge.target for edge in edges}
        self_loops = sum(1 for edge in edges if edge.source == edge.target)
        total_weight = sum(edge.weight for edge in edges)
        rows.append(
            {
                "layer": layer,
                "nodes": len(layer_nodes),
                "edges": len(edges),
                "self_loops": self_loops,
                "total_weight": total_weight,
            }
        )
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ordered_pair(first: str, second: str) -> Tuple[str, str]:
    return tuple(sorted((first, second), key=natural_key))  # type: ignore[return-value]


def select_visual_nodes(
    graph: GraphData,
    ranking: Sequence[Dict[str, object]],
    top_k: int,
    max_nodes: int,
) -> Set[str]:
    selected: Set[str] = set()
    aggregate_top_nodes = [str(row["node"]) for row in ranking[: min(top_k, max_nodes)]]

    layer_top_nodes: List[List[str]] = []
    all_nodes = sorted(graph.nodes, key=natural_key)
    for layer in sorted(graph.layer_adj, key=natural_key):
        _, _, total_strength, degree = node_strengths(
            graph.layer_adj[layer], all_nodes, graph.directed
        )
        candidates = sorted(
            all_nodes,
            key=lambda node: (-total_strength[node], -degree[node], natural_key(node)),
        )
        layer_top_nodes.append(
            [node for node in candidates if total_strength[node] > 0.0][:top_k]
        )

    candidate_lists = [aggregate_top_nodes] + layer_top_nodes
    for index in range(top_k):
        for candidates in candidate_lists:
            if index < len(candidates) and len(selected) < max_nodes:
                selected.add(candidates[index])

    top_nodes = list(selected)

    neighbor_score: Counter = Counter()
    for node in aggregate_top_nodes:
        for target, weight in graph.aggregate_adj.get(node, {}).items():
            if target != node:
                neighbor_score[target] += weight
        if graph.directed:
            for source, neighbors in graph.aggregate_adj.items():
                if source != node and node in neighbors:
                    neighbor_score[source] += neighbors[node]

    rank_lookup = {str(row["node"]): float(row["influence_score"]) for row in ranking}
    for node, _ in sorted(
        neighbor_score.items(),
        key=lambda item: (-item[1], -rank_lookup.get(item[0], 0.0), natural_key(item[0])),
    ):
        if len(selected) >= max_nodes:
            break
        selected.add(node)

    for row in ranking:
        if len(selected) >= max_nodes:
            break
        selected.add(str(row["node"]))

    return selected


def layout_edges_for_selected(graph: GraphData, selected: Set[str]) -> List[Tuple[str, str, float]]:
    weights: Counter = Counter()
    for source, neighbors in graph.aggregate_adj.items():
        if source not in selected:
            continue
        for target, weight in neighbors.items():
            if target not in selected or source == target:
                continue
            key = ordered_pair(source, target)
            weights[key] += weight
    return [(source, target, weight) for (source, target), weight in weights.items()]


def force_layout(
    nodes: Sequence[str],
    edges: Sequence[Tuple[str, str, float]],
    width: int = 1060,
    height: int = 620,
    iterations: int = 240,
    seed: int = 7,
) -> Dict[str, Tuple[float, float]]:
    if not nodes:
        return {}
    if len(nodes) == 1:
        return {nodes[0]: (width / 2.0, height / 2.0)}

    rng = random.Random(seed)
    center_x = width / 2.0
    center_y = height / 2.0
    radius = min(width, height) * 0.42
    positions: Dict[str, List[float]] = {}
    for index, node in enumerate(nodes):
        angle = 2.0 * math.pi * index / len(nodes)
        jitter = rng.uniform(-12.0, 12.0)
        positions[node] = [
            center_x + math.cos(angle) * (radius + jitter),
            center_y + math.sin(angle) * (radius + jitter),
        ]

    area = width * height
    optimal_distance = math.sqrt(area / max(1, len(nodes)))
    temperature = width / 8.0
    edge_strengths = [
        (source, target, 1.0 + math.log1p(max(weight, 0.0)))
        for source, target, weight in edges
        if source in positions and target in positions
    ]

    for step in range(iterations):
        displacement = {node: [0.0, 0.0] for node in nodes}

        for i, first in enumerate(nodes):
            first_x, first_y = positions[first]
            for second in nodes[i + 1 :]:
                second_x, second_y = positions[second]
                delta_x = first_x - second_x
                delta_y = first_y - second_y
                distance = max(0.01, math.hypot(delta_x, delta_y))
                force = (optimal_distance * optimal_distance) / distance
                offset_x = (delta_x / distance) * force
                offset_y = (delta_y / distance) * force
                displacement[first][0] += offset_x
                displacement[first][1] += offset_y
                displacement[second][0] -= offset_x
                displacement[second][1] -= offset_y

        for source, target, strength in edge_strengths:
            delta_x = positions[source][0] - positions[target][0]
            delta_y = positions[source][1] - positions[target][1]
            distance = max(0.01, math.hypot(delta_x, delta_y))
            force = (distance * distance / optimal_distance) * min(strength, 4.5)
            offset_x = (delta_x / distance) * force
            offset_y = (delta_y / distance) * force
            displacement[source][0] -= offset_x
            displacement[source][1] -= offset_y
            displacement[target][0] += offset_x
            displacement[target][1] += offset_y

        cooling = temperature * (1.0 - step / iterations)
        for node in nodes:
            delta_x, delta_y = displacement[node]
            distance = max(0.01, math.hypot(delta_x, delta_y))
            positions[node][0] += (delta_x / distance) * min(distance, cooling)
            positions[node][1] += (delta_y / distance) * min(distance, cooling)
            positions[node][0] = min(width - 30.0, max(30.0, positions[node][0]))
            positions[node][1] = min(height - 30.0, max(30.0, positions[node][1]))

    return {node: (xy[0], xy[1]) for node, xy in positions.items()}


def bridge_pairs_from_edges(edges: Iterable[Tuple[str, str]], nodes: Iterable[str]) -> Set[Tuple[str, str]]:
    """Find weak bridges using an undirected Tarjan pass."""
    adjacency: Dict[str, Set[str]] = {node: set() for node in nodes}
    for source, target in edges:
        if source == target:
            continue
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), len(adjacency) * 2 + 1000))

    visited: Set[str] = set()
    discovery: Dict[str, int] = {}
    low: Dict[str, int] = {}
    bridges: Set[Tuple[str, str]] = set()
    time = 0

    def dfs(node: str, parent: str | None) -> None:
        nonlocal time
        visited.add(node)
        discovery[node] = time
        low[node] = time
        time += 1

        for neighbor in adjacency.get(node, set()):
            if neighbor == parent:
                continue
            if neighbor not in visited:
                dfs(neighbor, node)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > discovery[node]:
                    bridges.add(ordered_pair(node, neighbor))
            else:
                low[node] = min(low[node], discovery[neighbor])

    for node in sorted(adjacency, key=natural_key):
        if node not in visited:
            dfs(node, None)
    return bridges


def build_bridge_sets(graph: GraphData) -> Dict[str, Set[Tuple[str, str]]]:
    bridge_sets: Dict[str, Set[Tuple[str, str]]] = {}
    aggregate_edges = [(edge.source, edge.target) for edge in graph.edges]
    bridge_sets["Aggregate"] = bridge_pairs_from_edges(aggregate_edges, graph.nodes)
    for layer, edges in graph.layer_edges.items():
        layer_nodes = {edge.source for edge in edges} | {edge.target for edge in edges}
        bridge_sets[layer] = bridge_pairs_from_edges(
            [(edge.source, edge.target) for edge in edges],
            layer_nodes,
        )
    return bridge_sets


def layer_palette(layers: Sequence[str]) -> Dict[str, str]:
    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
        "#be123c",
        "#4f46e5",
        "#65a30d",
        "#c026d3",
        "#0d9488",
        "#ca8a04",
        "#7c3aed",
        "#0284c7",
        "#db2777",
        "#059669",
        "#b91c1c",
        "#7c2d12",
        "#1d4ed8",
        "#15803d",
    ]
    return {layer: colors[index % len(colors)] for index, layer in enumerate(layers)}


def build_layer_node_stats(graph: GraphData, selected: Set[str]) -> Dict[str, Dict[str, Dict[str, object]]]:
    stats: Dict[str, Dict[str, Dict[str, object]]] = {}
    selected_nodes = sorted(selected, key=natural_key)
    all_nodes = sorted(graph.nodes, key=natural_key)
    for layer in sorted(graph.layer_adj, key=natural_key):
        out_strength, in_strength, total_strength, degree = node_strengths(
            graph.layer_adj[layer], all_nodes, graph.directed
        )
        strength_norm = max_normalize(total_strength)
        degree_norm = max_normalize({node: float(degree[node]) for node in all_nodes})
        layer_nodes = {
            edge.source for edge in graph.layer_edges[layer]
        } | {
            edge.target for edge in graph.layer_edges[layer]
        }
        stats[layer] = {}
        for node in selected_nodes:
            influence = 0.65 * strength_norm.get(node, 0.0) + 0.35 * degree_norm.get(node, 0.0)
            stats[layer][node] = {
                "influence": influence,
                "strength": total_strength.get(node, 0.0),
                "outStrength": out_strength.get(node, 0.0),
                "inStrength": in_strength.get(node, 0.0),
                "degree": degree.get(node, 0),
                "active": node in layer_nodes,
                "isolated": node not in layer_nodes or degree.get(node, 0) == 0,
            }
    return stats


def build_layer_positions(
    graph: GraphData,
    selected: Set[str],
    seed: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    positions: Dict[str, Dict[str, Dict[str, float]]] = {}
    selected_nodes = sorted(selected, key=natural_key)
    for index, layer in enumerate(sorted(graph.layer_edges, key=natural_key)):
        layer_nodes = {
            edge.source for edge in graph.layer_edges[layer]
        } | {
            edge.target for edge in graph.layer_edges[layer]
        }
        nodes = [node for node in selected_nodes if node in layer_nodes]
        if not nodes:
            positions[layer] = {}
            continue

        weights: Counter = Counter()
        for edge in graph.layer_edges[layer]:
            if edge.source not in selected or edge.target not in selected or edge.source == edge.target:
                continue
            visual_key = (edge.source, edge.target) if graph.directed else ordered_pair(edge.source, edge.target)
            weights[visual_key] += edge.weight

        layout = force_layout(
            nodes,
            [(source, target, weight) for (source, target), weight in weights.items()],
            width=1000,
            height=620,
            iterations=220,
            seed=seed + index + 13,
        )
        positions[layer] = {
            node: {
                "x": round(layout[node][0] / 1000, 5),
                "y": round(layout[node][1] / 620, 5),
            }
            for node in nodes
            if node in layout
        }
    return positions


def edge_rows_for_visual(
    graph: GraphData,
    selected: Set[str],
    ranking_lookup: Dict[str, Dict[str, object]],
    max_edges_per_layer: int,
    bridge_sets: Dict[str, Set[Tuple[str, str]]],
) -> Dict[str, List[Dict[str, object]]]:
    edge_map: Dict[str, List[Dict[str, object]]] = {"Aggregate": []}

    aggregate_weights: Counter = Counter()
    aggregate_layers: DefaultDict[Tuple[str, str], Set[str]] = defaultdict(set)
    for edge in graph.edges:
        if edge.source not in selected or edge.target not in selected or edge.source == edge.target:
            continue
        visual_key = (edge.source, edge.target) if graph.directed else ordered_pair(edge.source, edge.target)
        aggregate_weights[visual_key] += edge.weight
        aggregate_layers[visual_key].add(edge.layer)

    for (source, target), weight in aggregate_weights.items():
        edge_map["Aggregate"].append(
            {
                "source": source,
                "target": target,
                "weight": weight,
                "layers": len(aggregate_layers[(source, target)]),
                "bridge": ordered_pair(source, target) in bridge_sets.get("Aggregate", set()),
            }
        )

    for layer in sorted(graph.layer_edges, key=natural_key):
        layer_rows: Dict[Tuple[str, str], Dict[str, object]] = {}
        for edge in graph.layer_edges[layer]:
            if edge.source not in selected or edge.target not in selected or edge.source == edge.target:
                continue
            visual_key = (edge.source, edge.target) if graph.directed else ordered_pair(edge.source, edge.target)
            row = layer_rows.setdefault(
                visual_key,
                {
                    "source": visual_key[0],
                    "target": visual_key[1],
                    "weight": 0.0,
                    "layers": 1,
                    "bridge": ordered_pair(visual_key[0], visual_key[1])
                    in bridge_sets.get(layer, set()),
                },
            )
            row["weight"] = float(row["weight"]) + edge.weight
        edge_map[layer] = list(layer_rows.values())

    score_lookup = {
        node: float(row["influence_score"]) for node, row in ranking_lookup.items()
    }

    for layer, rows in edge_map.items():
        rows.sort(
            key=lambda row: (
                -float(row["weight"]),
                -score_lookup.get(str(row["source"]), 0.0)
                - score_lookup.get(str(row["target"]), 0.0),
            )
        )
        edge_map[layer] = rows[:max_edges_per_layer]
    return edge_map


def build_layer_profile(
    ranking: Sequence[Dict[str, object]], layers: Sequence[str], top_k: int
) -> List[Dict[str, object]]:
    profile: List[Dict[str, object]] = []
    for row in ranking[:top_k]:
        strengths = row.get("layer_strengths", {})
        assert isinstance(strengths, dict)
        total = sum(float(strengths.get(layer, 0.0)) for layer in layers)
        profile.append(
            {
                "node": row["node"],
                "rank": row["rank"],
                "score": row["influence_score"],
                "total": total,
                "layers": {
                    layer: float(strengths.get(layer, 0.0)) for layer in layers
                },
            }
        )
    return profile


def html_table(rows: Sequence[Dict[str, object]], columns: Sequence[Tuple[str, str]], limit: int | None = None) -> str:
    visible_rows = rows if limit is None else rows[:limit]
    header = "".join(f"<th>{html.escape(label)}</th>" for key, label in columns)
    body_rows = []
    for row in visible_rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = format_number(value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def build_advanced_report_html(
    graph: GraphData,
    selected: Set[str],
    data: Dict[str, object],
    top_table: str,
    layer_table: str,
) -> str:
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__ multiplex graph report</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fb;
      --fg: #17202a;
      --muted: #5e6b78;
      --panel: #ffffff;
      --soft: #eef3f7;
      --line: #d8e0e7;
      --bridge: #111827;
      --inter: #64748b;
      --top: #f59e0b;
      --shadow: rgba(31, 41, 51, 0.12);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }
    header {
      padding: 28px clamp(18px, 4vw, 48px) 18px;
      border-bottom: 1px solid var(--line);
    }
    main {
      width: min(1220px, calc(100% - 32px));
      margin: 0 auto;
      padding: 22px 0 44px;
    }
    h1, h2, h3 { margin: 0; font-weight: 650; letter-spacing: 0; }
    h1 { font-size: clamp(1.65rem, 3vw, 2.4rem); }
    h2 { font-size: 1.18rem; margin-bottom: 12px; }
    h3 { font-size: 1rem; }
    p { margin: 8px 0 0; color: var(--muted); }
    .meta, .controls, .checks, .legend, .metrics-strip {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .meta { margin-top: 16px; }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: color-mix(in srgb, var(--panel) 84%, transparent);
      color: var(--fg);
      font-size: 0.9rem;
    }
    section {
      margin-top: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 10px 28px var(--shadow);
      padding: clamp(14px, 2vw, 22px);
    }
    .controls {
      align-items: end;
      margin-bottom: 12px;
    }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    select, input[type="checkbox"] { accent-color: #0f766e; }
    select {
      min-width: 178px;
      color: var(--fg);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
    }
    .checks {
      align-items: center;
      margin: 6px 0 14px;
    }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 32px;
      padding: 4px 8px 4px 0;
    }
    .layer-checks {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(116px, 1fr));
      gap: 6px 12px;
      margin: 8px 0 14px;
    }
    .layer-option {
      display: flex;
      align-items: center;
      gap: 7px;
      color: var(--fg);
      min-width: 0;
    }
    .swatch {
      width: 12px;
      height: 12px;
      border-radius: 50%;
      flex: 0 0 auto;
    }
    .network-frame {
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background:
        linear-gradient(to bottom, transparent, color-mix(in srgb, var(--soft) 72%, transparent)),
        var(--panel);
    }
    svg {
      width: 100%;
      height: auto;
      display: block;
    }
    .layer-guide {
      stroke: var(--line);
      stroke-width: 1;
      stroke-dasharray: none;
    }
    .layer-panel {
      fill: color-mix(in srgb, var(--soft) 68%, transparent);
      stroke: var(--line);
      stroke-width: 1;
      rx: 8;
    }
    .layer-title {
      font-size: 12px;
      fill: var(--muted);
      paint-order: stroke;
      stroke: var(--panel);
      stroke-width: 3px;
    }
    .edge {
      stroke-linecap: round;
      opacity: 0.52;
    }
    .edge.inter {
      stroke: var(--inter);
      stroke-dasharray: 6 6;
      opacity: 0.45;
    }
    .edge.bridge {
      stroke: var(--bridge);
      opacity: 0.92;
      stroke-dasharray: none;
    }
    .node {
      stroke: var(--panel);
      stroke-width: 1.6;
      cursor: pointer;
    }
    .node.top {
      stroke: var(--top);
      stroke-width: 2.5;
    }
    .node.isolated {
      fill: var(--panel);
      stroke-width: 2.2;
      stroke-dasharray: 3 3;
    }
    .node-label {
      font-size: 11px;
      fill: var(--fg);
      paint-order: stroke;
      stroke: var(--panel);
      stroke-width: 3px;
      pointer-events: none;
    }
    .legend {
      align-items: center;
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    .metric {
      min-width: 150px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: color-mix(in srgb, var(--panel) 86%, var(--soft));
    }
    .metric strong {
      display: block;
      color: var(--fg);
      font-weight: 650;
    }
    .tooltip {
      position: fixed;
      max-width: 300px;
      z-index: 3;
      display: none;
      padding: 9px 10px;
      border-radius: 8px;
      background: var(--fg);
      color: var(--bg);
      box-shadow: 0 10px 24px var(--shadow);
      font-size: 0.88rem;
      pointer-events: none;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 18px;
    }
    .table-scroll { overflow-x: auto; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }
    th, td {
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }
    th {
      color: var(--muted);
      font-weight: 650;
      background: color-mix(in srgb, var(--panel) 90%, var(--bg));
    }
    td:first-child, th:first-child { padding-left: 0; }
    .profile {
      display: grid;
      gap: 8px;
    }
    .profile-row {
      display: grid;
      grid-template-columns: minmax(70px, 120px) 1fr;
      gap: 10px;
      align-items: center;
    }
    .profile-label {
      font-size: 0.88rem;
      color: var(--fg);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .bar {
      height: 20px;
      display: flex;
      gap: 2px;
      min-width: 160px;
    }
    .segment {
      min-width: 2px;
      opacity: 0.9;
    }
    footer {
      width: min(1220px, calc(100% - 32px));
      margin: 0 auto 28px;
      color: var(--muted);
      font-size: 0.9rem;
    }
    @media (max-width: 720px) {
      th, td { padding: 7px 8px; }
      .profile-row { grid-template-columns: 1fr; }
      .bar { min-width: 0; }
      select { width: 100%; }
    }
  </style>
</head>
<body>
  <header>
    <h1>__TITLE__</h1>
    <p>Multiplex graph influence report with layer colors, intra/inter-layer views, bridge highlighting, isolated-node display, and influence intensity mode.</p>
    <div class="meta">
      <span class="pill">__NODE_COUNT__ nodes</span>
      <span class="pill">__EDGE_COUNT__ raw edges</span>
      <span class="pill">__LAYER_COUNT__ layers</span>
      <span class="pill">__DIRECTED__ analysis</span>
      <span class="pill">__VISUAL_NODE_COUNT__ candidate nodes</span>
    </div>
  </header>

  <main>
    <section>
      <h2>Interactive network</h2>
      <div class="controls">
        <label>Mode
          <select id="modeSelect">
            <option value="aggregate">All layers merged</option>
            <option value="single">One layer - intra edges</option>
            <option value="multi">Multiple layers - intra and inter edges</option>
            <option value="influence">Influence mode</option>
          </select>
        </label>
        <label>Layer
          <select id="layerSelect"></select>
        </label>
        <label>Influence scope
          <select id="influenceScope">
            <option value="all">All selected layers</option>
            <option value="layer">Selected layer</option>
          </select>
        </label>
      </div>
      <div id="layerChecks" class="layer-checks" aria-label="Layer filters"></div>
      <div class="checks">
        <label class="check"><input type="checkbox" id="intraToggle" checked> Intra-layer edges</label>
        <label class="check"><input type="checkbox" id="interToggle" checked> Inter-layer edges</label>
        <label class="check"><input type="checkbox" id="bridgeToggle" checked> Highlight bridges</label>
        <label class="check"><input type="checkbox" id="bridgeOnlyToggle"> Only bridges</label>
        <label class="check"><input type="checkbox" id="isolatedToggle"> Show isolated nodes</label>
        <label class="check"><input type="checkbox" id="labelToggle" checked> Show labels</label>
      </div>
      <div class="network-frame">
        <svg id="networkSvg" viewBox="0 0 1060 650" role="img" aria-label="Multiplex graph visualization">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"></path>
            </marker>
          </defs>
        </svg>
      </div>
      <div id="legend" class="legend"></div>
      <p id="viewStatus">Ready.</p>
    </section>

    <section>
      <h2>Influence focus</h2>
      <div id="focusMetrics" class="metrics-strip" aria-live="polite"></div>
      <p>In influence mode, each node uses its layer color. Darker nodes are more influential in the selected layer or across the selected layers.</p>
    </section>

    <section>
      <h2>Layer profile of top nodes</h2>
      <div id="profile" class="profile" aria-label="Layer contribution bars"></div>
    </section>

    <section class="grid">
      <div>
        <h2>Top influential nodes</h2>
        <div class="table-scroll">__TOP_TABLE__</div>
      </div>
      <div>
        <h2>Layer summary</h2>
        <div class="table-scroll">__LAYER_TABLE__</div>
      </div>
    </section>
  </main>

  <footer>
    Influence score = 35% strength, 25% PageRank, 20% betweenness, 10% layer coverage, and 10% layer participation. Layer influence uses local layer strength and layer degree.
  </footer>

  <div id="tooltip" class="tooltip"></div>
  <script>
    const DATA = __DATA_JSON__;
    const WIDTH = 1060;
    const HEIGHT = 650;
    const svg = document.getElementById("networkSvg");
    const modeSelect = document.getElementById("modeSelect");
    const layerSelect = document.getElementById("layerSelect");
    const influenceScope = document.getElementById("influenceScope");
    const layerChecks = document.getElementById("layerChecks");
    const intraToggle = document.getElementById("intraToggle");
    const interToggle = document.getElementById("interToggle");
    const bridgeToggle = document.getElementById("bridgeToggle");
    const bridgeOnlyToggle = document.getElementById("bridgeOnlyToggle");
    const isolatedToggle = document.getElementById("isolatedToggle");
    const labelToggle = document.getElementById("labelToggle");
    const tooltip = document.getElementById("tooltip");
    const legend = document.getElementById("legend");
    const viewStatus = document.getElementById("viewStatus");
    const focusMetrics = document.getElementById("focusMetrics");
    const nodeById = new Map(DATA.nodes.map(node => [node.id, node]));
    let previousControlKey = "";

    function escapeText(value) {
      return String(value).replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[char]));
    }

    function clamp(value, low, high) {
      return Math.max(low, Math.min(high, value));
    }

    function edgeWidth(edge) {
      return Math.max(0.8, Math.min(5, 0.7 + Math.log1p(edge.weight || 1)));
    }

    function layerColor(layer) {
      return DATA.layerColors[layer] || "#0f766e";
    }

    function influenceFill(layer, influence) {
      const percent = Math.round(28 + clamp(influence || 0, 0, 1) * 72);
      return `color-mix(in srgb, ${layerColor(layer)} ${percent}%, var(--panel))`;
    }

    function dominantLayer(node) {
      let bestLayer = DATA.layers[0] || "Aggregate";
      let bestValue = -1;
      for (const layer of DATA.layers) {
        const value = node.layerStrengths?.[layer] || 0;
        if (value > bestValue) {
          bestLayer = layer;
          bestValue = value;
        }
      }
      return bestLayer;
    }

    function selectedLayers() {
      const checked = [...layerChecks.querySelectorAll("input[type='checkbox']:checked")].map(input => input.value);
      return checked.length ? checked : DATA.layers.slice(0, Math.min(3, DATA.layers.length));
    }

    function panelLayout(layers) {
      const count = Math.max(1, layers.length);
      const cols = count === 1 ? 1 : count <= 4 ? 2 : 3;
      const rows = Math.ceil(count / cols);
      const gap = 18;
      const panelWidth = (WIDTH - gap * (cols + 1)) / cols;
      const panelHeight = (HEIGHT - gap * (rows + 1)) / rows;
      const panels = {};
      layers.forEach((layer, index) => {
        const col = index % cols;
        const row = Math.floor(index / cols);
        panels[layer] = {
          x: gap + col * (panelWidth + gap),
          y: gap + row * (panelHeight + gap),
          width: panelWidth,
          height: panelHeight
        };
      });
      return panels;
    }

    function mapLayerPosition(base, layer, panels) {
      const panel = panels[layer] || { x: 30, y: 30, width: WIDTH - 60, height: HEIGHT - 60 };
      const local = DATA.layerPositions[layer]?.[base.id];
      const padding = Math.min(34, Math.max(18, Math.min(panel.width, panel.height) * 0.12));
      const xRatio = local ? local.x : base.x / WIDTH;
      const yRatio = local ? local.y : base.y / HEIGHT;
      return {
        x: panel.x + padding + clamp(xRatio, 0, 1) * Math.max(1, panel.width - padding * 2),
        y: panel.y + padding + clamp(yRatio, 0, 1) * Math.max(1, panel.height - padding * 2)
      };
    }

    function layerCopy(base, layer, panels, influenceMode) {
      const stats = DATA.layerNodeStats[layer]?.[base.id] || {};
      const influence = influenceMode ? (stats.influence || 0) : Math.max(0.15, stats.influence || 0);
      const position = mapLayerPosition(base, layer, panels);
      return {
        ...base,
        copyId: `${base.id}@@${layer}`,
        baseId: base.id,
        layer,
        x: position.x,
        y: position.y,
        layerInfluence: stats.influence || 0,
        layerStrength: stats.strength || 0,
        layerDegree: stats.degree || 0,
        active: Boolean(stats.active),
        isolated: Boolean(stats.isolated),
        radius: Math.max(4.5, Math.min(14, 4.5 + 9.5 * Math.sqrt(influence)))
      };
    }

    function singleLayerNode(base, layer, influenceMode) {
      const stats = DATA.layerNodeStats[layer]?.[base.id] || {};
      const position = mapLayerPosition(base, layer, panelLayout([layer]));
      return {
        ...base,
        copyId: base.id,
        baseId: base.id,
        layer,
        x: position.x,
        y: position.y,
        layerInfluence: stats.influence || 0,
        layerStrength: stats.strength || 0,
        layerDegree: stats.degree || 0,
        active: Boolean(stats.active),
        isolated: Boolean(stats.isolated),
        radius: influenceMode
          ? Math.max(5, Math.min(20, 5 + 15 * Math.sqrt(stats.influence || 0)))
          : base.radius
      };
    }

    function aggregateNode(base, influenceMode) {
      const layer = dominantLayer(base);
      return {
        ...base,
        copyId: base.id,
        baseId: base.id,
        layer,
        layerInfluence: base.score,
        layerStrength: base.strength,
        layerDegree: base.degree,
        active: true,
        isolated: base.degree === 0,
        radius: influenceMode ? Math.max(5, Math.min(22, 5 + 17 * Math.sqrt(base.score || 0))) : base.radius
      };
    }

    function edgeKey(source, target) {
      return [source, target].sort((a, b) => String(a).localeCompare(String(b), undefined, { numeric: true })).join("--");
    }

    function buildView() {
      const mode = modeSelect.value;
      const selectedLayer = layerSelect.value || DATA.layers[0];
      const influenceMode = mode === "influence";
      const influenceAsLayer = influenceMode && influenceScope.value === "layer";
      const multilayerMode = mode === "multi" || (influenceMode && !influenceAsLayer);
      const aggregateMode = mode === "aggregate";
      const singleMode = mode === "single" || influenceAsLayer;
      const layers = multilayerMode ? selectedLayers() : [selectedLayer];
      const panels = panelLayout(layers);
      const nodeMap = new Map();
      const edges = [];

      if (aggregateMode) {
        const rawEdges = intraToggle.checked ? (DATA.edgesByLayer.Aggregate || []) : [];
        for (const edge of rawEdges) {
          if (bridgeOnlyToggle.checked && !edge.bridge) continue;
          edges.push({ ...edge, copySource: edge.source, copyTarget: edge.target, layer: "Aggregate", type: "intra" });
        }
        const connected = new Set();
        edges.forEach(edge => {
          connected.add(edge.source);
          connected.add(edge.target);
        });
        DATA.nodes.forEach(base => {
          const node = aggregateNode(base, influenceMode);
          if (isolatedToggle.checked || connected.has(base.id)) nodeMap.set(node.copyId, node);
        });
      } else if (singleMode) {
        const rawEdges = intraToggle.checked ? (DATA.edgesByLayer[selectedLayer] || []) : [];
        for (const edge of rawEdges) {
          if (bridgeOnlyToggle.checked && !edge.bridge) continue;
          edges.push({ ...edge, copySource: edge.source, copyTarget: edge.target, layer: selectedLayer, type: "intra" });
        }
        const connected = new Set();
        edges.forEach(edge => {
          connected.add(edge.source);
          connected.add(edge.target);
        });
        DATA.nodes.forEach(base => {
          const node = singleLayerNode(base, selectedLayer, influenceMode);
          if (isolatedToggle.checked || node.active || connected.has(base.id)) {
            if (!bridgeOnlyToggle.checked || connected.has(base.id) || isolatedToggle.checked) {
              nodeMap.set(node.copyId, node);
            }
          }
        });
      } else {
        for (const layer of layers) {
          const rawEdges = intraToggle.checked ? (DATA.edgesByLayer[layer] || []) : [];
          for (const edge of rawEdges) {
            if (bridgeOnlyToggle.checked && !edge.bridge) continue;
            edges.push({
              ...edge,
              copySource: `${edge.source}@@${layer}`,
              copyTarget: `${edge.target}@@${layer}`,
              layer,
              type: "intra"
            });
          }
        }
        const connectedCopies = new Set();
        edges.forEach(edge => {
          connectedCopies.add(edge.copySource);
          connectedCopies.add(edge.copyTarget);
        });
        DATA.nodes.forEach(base => {
          for (const layer of layers) {
            const node = layerCopy(base, layer, panels, influenceMode);
            if (isolatedToggle.checked || node.active || connectedCopies.has(node.copyId)) {
              if (!bridgeOnlyToggle.checked || connectedCopies.has(node.copyId) || isolatedToggle.checked) {
                nodeMap.set(node.copyId, node);
              }
            }
          }
        });
        if (interToggle.checked && !bridgeOnlyToggle.checked) {
          DATA.nodes.forEach(base => {
            const activeLayers = layers.filter(layer => {
              const node = nodeMap.get(`${base.id}@@${layer}`);
              return node && node.active;
            });
            for (let i = 0; i < activeLayers.length - 1; i += 1) {
              edges.push({
                source: base.id,
                target: base.id,
                copySource: `${base.id}@@${activeLayers[i]}`,
                copyTarget: `${base.id}@@${activeLayers[i + 1]}`,
                weight: 1,
                layer: activeLayers[i],
                type: "inter",
                bridge: false
              });
            }
          });
        }
      }

      const visibleEdges = edges.filter(edge => nodeMap.has(edge.copySource) && nodeMap.has(edge.copyTarget));
      return { nodes: [...nodeMap.values()], edges: visibleEdges, layers, panels, multilayerMode, aggregateMode, influenceMode };
    }

    function clearSvg() {
      while (svg.lastChild && svg.lastChild.nodeName !== "defs") {
        svg.removeChild(svg.lastChild);
      }
    }

    function lineTrim(source, target) {
      const dx = target.x - source.x;
      const dy = target.y - source.y;
      const distance = Math.max(1, Math.hypot(dx, dy));
      const trim = DATA.directed ? Math.max(6, target.radius + 4) : 0;
      return [target.x - (dx / distance) * trim, target.y - (dy / distance) * trim];
    }

    function nodeFill(node, view) {
      if (node.isolated && !node.active && !view.aggregateMode) return "var(--panel)";
      if (view.influenceMode) return influenceFill(node.layer, node.layerInfluence || node.score || 0);
      if (view.aggregateMode) return influenceFill(node.layer, node.score || 0.2);
      return layerColor(node.layer);
    }

    function nodeStroke(node) {
      if (node.isolated && !node.active) return layerColor(node.layer);
      return "var(--panel)";
    }

    function tooltipPosition(event, node) {
      if (Number.isFinite(event.clientX) && Number.isFinite(event.clientY) && event.clientX > 0 && event.clientY > 0) {
        return [event.clientX + 12, event.clientY + 12];
      }
      const rect = svg.getBoundingClientRect();
      return [
        rect.left + (node.x / WIDTH) * rect.width + 12,
        rect.top + (node.y / HEIGHT) * rect.height + 12
      ];
    }

    function showTooltip(event, node) {
      const [left, top] = tooltipPosition(event, node);
      const layerLine = node.layer ? `<br>Layer: ${escapeText(node.layer)}` : "";
      tooltip.innerHTML = `
        <strong>Node ${escapeText(node.baseId || node.id)}</strong>${layerLine}<br>
        Aggregate rank: ${node.rank}<br>
        Aggregate influence: ${node.scoreLabel}<br>
        Aggregate degree: ${node.degree.toLocaleString()}<br>
        PageRank: ${node.pagerank.toExponential(3)}<br>
        Betweenness: ${Number(node.betweenness).toLocaleString()}<br>
        Layer influence: ${(node.layerInfluence || 0).toFixed(4)}<br>
        Layer degree: ${(node.layerDegree || 0).toLocaleString()}<br>
        Layer strength: ${(node.layerStrength || 0).toLocaleString()}<br>
        Status: ${node.active ? "active" : "isolated/inactive"}
      `;
      tooltip.style.left = `${left}px`;
      tooltip.style.top = `${top}px`;
      tooltip.style.display = "block";
    }

    function hideTooltip() {
      tooltip.style.display = "none";
    }

    function renderGuides(view) {
      if (!view.multilayerMode) return;
      const guideGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      view.layers.forEach(layer => {
        const panel = view.panels[layer];
        if (!panel) return;
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", panel.x);
        rect.setAttribute("y", panel.y);
        rect.setAttribute("width", panel.width);
        rect.setAttribute("height", panel.height);
        rect.setAttribute("class", "layer-panel");
        guideGroup.appendChild(rect);

        const title = document.createElementNS("http://www.w3.org/2000/svg", "text");
        title.setAttribute("x", panel.x + 12);
        title.setAttribute("y", panel.y + 20);
        title.setAttribute("class", "layer-title");
        title.textContent = `Layer ${layer}`;
        guideGroup.appendChild(title);
      });
      svg.appendChild(guideGroup);
    }

    function render() {
      const view = buildView();
      const nodeMap = new Map(view.nodes.map(node => [node.copyId, node]));
      clearSvg();
      renderGuides(view);

      const edgeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      edgeGroup.setAttribute("aria-hidden", "true");
      view.edges.forEach(edge => {
        const source = nodeMap.get(edge.copySource);
        const target = nodeMap.get(edge.copyTarget);
        if (!source || !target) return;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        const [x2, y2] = lineTrim(source, target);
        line.setAttribute("x1", source.x);
        line.setAttribute("y1", source.y);
        line.setAttribute("x2", x2);
        line.setAttribute("y2", y2);
        line.setAttribute("class", `edge ${edge.type === "inter" ? "inter" : "intra"} ${edge.bridge && bridgeToggle.checked ? "bridge" : ""}`);
        line.setAttribute("stroke", edge.type === "intra" ? layerColor(edge.layer) : "var(--inter)");
        line.setAttribute("stroke-width", edge.type === "inter" ? 1.2 : edgeWidth(edge));
        if (DATA.directed && edge.type === "intra") line.setAttribute("marker-end", "url(#arrow)");
        edgeGroup.appendChild(line);
      });
      svg.appendChild(edgeGroup);

      const nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      view.nodes.forEach(node => {
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", node.x);
        circle.setAttribute("cy", node.y);
        circle.setAttribute("r", node.radius);
        circle.setAttribute("fill", nodeFill(node, view));
        circle.setAttribute("stroke", nodeStroke(node));
        circle.setAttribute("class", `node${node.top ? " top" : ""}${node.isolated && !node.active ? " isolated" : ""}`);
        circle.setAttribute("tabindex", "0");
        circle.setAttribute("aria-label", `Node ${node.baseId || node.id}, layer ${node.layer}, influence ${node.scoreLabel}`);
        circle.addEventListener("mousemove", event => showTooltip(event, node));
        circle.addEventListener("mouseleave", hideTooltip);
        circle.addEventListener("focus", event => showTooltip(event, node));
        circle.addEventListener("blur", hideTooltip);
        nodeGroup.appendChild(circle);

        if (labelToggle.checked && (node.top || view.influenceMode || view.nodes.length <= 45)) {
          const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
          label.setAttribute("x", node.x + node.radius + 4);
          label.setAttribute("y", node.y + 4);
          label.setAttribute("class", "node-label");
          label.textContent = node.baseId || node.id;
          nodeGroup.appendChild(label);
        }
      });
      svg.appendChild(nodeGroup);
      renderLegend(view);
      renderFocus(view);
      viewStatus.textContent = `${view.nodes.length.toLocaleString()} visible nodes and ${view.edges.length.toLocaleString()} visible edges.`;
    }

    function renderLegend(view) {
      legend.innerHTML = "";
      const layers = view.aggregateMode ? DATA.layers : view.layers;
      layers.forEach(layer => {
        const item = document.createElement("span");
        item.className = "pill";
        item.innerHTML = `<span class="swatch" style="background:${layerColor(layer)}"></span> Layer ${escapeText(layer)}`;
        legend.appendChild(item);
      });
      if (bridgeToggle.checked || bridgeOnlyToggle.checked) {
        const bridge = document.createElement("span");
        bridge.className = "pill";
        bridge.textContent = bridgeOnlyToggle.checked ? "Showing only bridge edges" : "Bridge edges are dark";
        legend.appendChild(bridge);
      }
      if (view.multilayerMode && interToggle.checked) {
        const inter = document.createElement("span");
        inter.className = "pill";
        inter.textContent = "Dashed edges connect the same node across layers";
        legend.appendChild(inter);
      }
    }

    function currentInfluenceRows(view) {
      if (view.aggregateMode) {
        return DATA.nodes
          .map(node => ({ node: node.id, layer: "All", score: node.score, degree: node.degree, strength: node.strength }))
          .sort((a, b) => b.score - a.score);
      }
      const layers = view.multilayerMode ? view.layers : [layerSelect.value || DATA.layers[0]];
      const byNode = new Map();
      DATA.nodes.forEach(node => {
        const entries = layers.map(layer => DATA.layerNodeStats[layer]?.[node.id]).filter(Boolean);
        if (!entries.length) return;
        const score = entries.reduce((total, row) => total + (row.influence || 0), 0) / entries.length;
        const degree = entries.reduce((total, row) => total + (row.degree || 0), 0);
        const strength = entries.reduce((total, row) => total + (row.strength || 0), 0);
        byNode.set(node.id, { node: node.id, layer: layers.length === 1 ? layers[0] : `${layers.length} layers`, score, degree, strength });
      });
      return [...byNode.values()].sort((a, b) => b.score - a.score || b.degree - a.degree);
    }

    function renderFocus(view) {
      focusMetrics.innerHTML = "";
      currentInfluenceRows(view).slice(0, 8).forEach(row => {
        const item = document.createElement("div");
        item.className = "metric";
        item.innerHTML = `
          <strong>${escapeText(row.node)}</strong>
          Score ${Number(row.score || 0).toFixed(4)}<br>
          Degree ${Number(row.degree || 0).toLocaleString()}<br>
          Strength ${Number(row.strength || 0).toLocaleString()}<br>
          ${escapeText(row.layer)}
        `;
        focusMetrics.appendChild(item);
      });
    }

    function renderProfile() {
      const root = document.getElementById("profile");
      root.innerHTML = "";
      DATA.layerProfile.forEach(row => {
        const item = document.createElement("div");
        item.className = "profile-row";
        const label = document.createElement("div");
        label.className = "profile-label";
        label.textContent = `#${row.rank} node ${row.node}`;
        const bar = document.createElement("div");
        bar.className = "bar";
        DATA.layers.forEach(layer => {
          const value = row.layers[layer] || 0;
          const segment = document.createElement("div");
          segment.className = "segment";
          segment.style.width = `${value > 0 && row.total > 0 ? Math.max(2, (value / row.total) * 100) : 0}%`;
          segment.style.background = layerColor(layer);
          segment.title = `Layer ${layer}: ${value.toLocaleString()}`;
          bar.appendChild(segment);
        });
        item.appendChild(label);
        item.appendChild(bar);
        root.appendChild(item);
      });
    }

    function setDisabled(control, disabled) {
      control.disabled = disabled;
      const wrapper = control.closest("label");
      if (wrapper) wrapper.style.opacity = disabled ? "0.55" : "1";
    }

    function updateControls() {
      const mode = modeSelect.value;
      const influenceLayerScope = mode === "influence" && influenceScope.value === "layer";
      const aggregateMode = mode === "aggregate";
      const singleMode = mode === "single" || influenceLayerScope;
      const multiMode = mode === "multi" || (mode === "influence" && !influenceLayerScope);
      const controlKey = `${mode}:${influenceScope.value}`;
      const wasMultiMode = previousControlKey.startsWith("multi:") || previousControlKey === "influence:all";

      setDisabled(layerSelect, !(singleMode || influenceLayerScope));
      setDisabled(influenceScope, mode !== "influence");
      setDisabled(interToggle, !multiMode);
      setDisabled(intraToggle, false);
      [...layerChecks.querySelectorAll("input[type='checkbox']")].forEach(input => {
        input.disabled = !multiMode;
        input.closest("label").style.opacity = input.disabled ? "0.55" : "1";
      });

      if (!multiMode) interToggle.checked = false;
      if (multiMode && !wasMultiMode) interToggle.checked = true;
      if (bridgeOnlyToggle.checked) bridgeToggle.checked = true;
      if (!intraToggle.checked && !interToggle.checked) intraToggle.checked = true;
      previousControlKey = controlKey;
    }

    function buildControls() {
      DATA.layers.forEach((layer, index) => {
        const option = document.createElement("option");
        option.value = layer;
        option.textContent = `Layer ${layer}`;
        layerSelect.appendChild(option);

        const label = document.createElement("label");
        label.className = "layer-option";
        const checked = index < Math.min(4, DATA.layers.length) ? "checked" : "";
        label.innerHTML = `
          <input type="checkbox" value="${escapeText(layer)}" ${checked}>
          <span class="swatch" style="background:${layerColor(layer)}"></span>
          <span>Layer ${escapeText(layer)}</span>
        `;
        layerChecks.appendChild(label);
      });
      [...document.querySelectorAll("select, input")].forEach(control => {
        control.addEventListener("change", () => {
          updateControls();
          render();
        });
      });
      updateControls();
    }

    buildControls();
    renderProfile();
    render();
  </script>
</body>
</html>
"""
    replacements = {
        "__TITLE__": html.escape(graph.name),
        "__NODE_COUNT__": f"{len(graph.nodes):,}",
        "__EDGE_COUNT__": f"{len(graph.edges):,}",
        "__LAYER_COUNT__": f"{len(graph.layer_edges):,}",
        "__DIRECTED__": "Directed" if graph.directed else "Undirected",
        "__VISUAL_NODE_COUNT__": f"{len(selected):,}",
        "__TOP_TABLE__": top_table,
        "__LAYER_TABLE__": layer_table,
        "__DATA_JSON__": json.dumps(data, separators=(",", ":")),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def make_report(
    graph: GraphData,
    ranking: Sequence[Dict[str, object]],
    layer_summary: Sequence[Dict[str, object]],
    output_path: Path,
    top_k: int,
    max_nodes: int,
    max_edges_per_layer: int,
    seed: int,
) -> None:
    selected = select_visual_nodes(graph, ranking, top_k=top_k, max_nodes=max_nodes)
    ranking_lookup = {str(row["node"]): row for row in ranking}
    layout_nodes = sorted(selected, key=lambda node: (ranking_lookup.get(node, {}).get("rank", 10**9), natural_key(node)))
    positions = force_layout(
        layout_nodes,
        layout_edges_for_selected(graph, selected),
        seed=seed,
    )
    bridge_sets = build_bridge_sets(graph)
    edge_layers = edge_rows_for_visual(
        graph,
        selected,
        ranking_lookup,
        max_edges_per_layer=max_edges_per_layer,
        bridge_sets=bridge_sets,
    )
    layers = sorted(graph.layer_edges, key=natural_key)
    colors_by_layer = layer_palette(layers)
    layer_node_stats = build_layer_node_stats(graph, selected)
    layer_positions = build_layer_positions(graph, selected, seed)
    max_score = max((float(row["influence_score"]) for row in ranking), default=1.0)
    node_rows = []
    for node in layout_nodes:
        row = ranking_lookup[node]
        x, y = positions[node]
        score = float(row["influence_score"])
        node_rows.append(
            {
                "id": node,
                "rank": row["rank"],
                "x": round(x, 2),
                "y": round(y, 2),
                "score": score,
                "scoreLabel": format_number(score, 4),
                "strength": float(row["total_strength"]),
                "degree": int(row["degree"]),
                "pagerank": float(row["pagerank"]),
                "betweenness": float(row["betweenness"]),
                "layerCount": int(row["layer_count"]),
                "participation": float(row["participation"]),
                "layerStrengths": {
                    layer: float(row.get("layer_strengths", {}).get(layer, 0.0))
                    if isinstance(row.get("layer_strengths"), dict)
                    else 0.0
                    for layer in layers
                },
                "top": int(row["rank"]) <= top_k,
                "radius": round(5.0 + 17.0 * math.sqrt(score / max_score), 2) if max_score > 0 else 7.0,
            }
        )

    data = {
        "name": graph.name,
        "directed": graph.directed,
        "nodes": node_rows,
        "edgesByLayer": edge_layers,
        "layers": layers,
        "layerColors": colors_by_layer,
        "layerNodeStats": layer_node_stats,
        "layerPositions": layer_positions,
        "layerProfile": build_layer_profile(ranking, layers, min(top_k, 20)),
    }

    top_table = html_table(
        ranking,
        [
            ("rank", "Rank"),
            ("node", "Node"),
            ("influence_score", "Influence"),
            ("total_strength", "Strength"),
            ("degree", "Degree"),
            ("pagerank", "PageRank"),
            ("betweenness", "Betweenness"),
            ("layer_count", "Layers"),
            ("participation", "Participation"),
        ],
        limit=min(top_k, 30),
    )
    layer_table = html_table(
        layer_summary,
        [
            ("layer", "Layer"),
            ("nodes", "Nodes"),
            ("edges", "Edges"),
            ("self_loops", "Self loops"),
            ("total_weight", "Weight"),
        ],
    )

    report_html = build_advanced_report_html(
        graph=graph,
        selected=selected,
        data=data,
        top_table=top_table,
        layer_table=layer_table,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html, encoding="utf-8")
    return

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(graph.name)} multiplex graph report</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f7f4ed;
      --fg: #1f2933;
      --muted: #617080;
      --panel: #ffffff;
      --line: #d5d8dc;
      --accent: #0f766e;
      --accent-2: #b45309;
      --accent-3: #4338ca;
      --top: #f59e0b;
      --shadow: rgba(31, 41, 51, 0.12);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #151a1e;
        --fg: #edf2f7;
        --muted: #a7b1bd;
        --panel: #20272e;
        --line: #3a4651;
        --accent: #2dd4bf;
        --accent-2: #fbbf24;
        --accent-3: #a5b4fc;
        --top: #fbbf24;
        --shadow: rgba(0, 0, 0, 0.28);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--fg);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.45;
    }}
    header {{
      padding: 28px clamp(18px, 4vw, 48px) 18px;
      border-bottom: 1px solid var(--line);
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 22px 0 44px;
    }}
    h1, h2, h3 {{ margin: 0; font-weight: 650; letter-spacing: 0; }}
    h1 {{ font-size: clamp(1.65rem, 3vw, 2.4rem); }}
    h2 {{ font-size: 1.18rem; margin-bottom: 12px; }}
    h3 {{ font-size: 1rem; }}
    p {{ margin: 8px 0 0; color: var(--muted); }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    .pill {{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      background: color-mix(in srgb, var(--panel) 84%, transparent);
      color: var(--fg);
      font-size: 0.9rem;
    }}
    section {{
      margin-top: 22px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 10px 28px var(--shadow);
      padding: clamp(14px, 2vw, 22px);
    }}
    .controls {{
      display: flex;
      align-items: end;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    label {{
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    select, input[type="checkbox"] {{
      accent-color: var(--accent);
    }}
    select {{
      min-width: 190px;
      color: var(--fg);
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
    }}
    .check {{
      display: flex;
      align-items: center;
      gap: 8px;
      padding-bottom: 8px;
    }}
    .network-frame {{
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      background:
        radial-gradient(circle at top left, color-mix(in srgb, var(--accent) 12%, transparent), transparent 34%),
        color-mix(in srgb, var(--panel) 94%, var(--bg));
    }}
    svg {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .edge {{
      stroke: color-mix(in srgb, var(--muted) 48%, transparent);
      stroke-linecap: round;
    }}
    .node {{
      stroke: var(--panel);
      stroke-width: 1.5;
      cursor: pointer;
    }}
    .node.top {{
      stroke: var(--top);
      stroke-width: 2.5;
    }}
    .node-label {{
      font-size: 11px;
      fill: var(--fg);
      paint-order: stroke;
      stroke: var(--panel);
      stroke-width: 3px;
      pointer-events: none;
    }}
    .tooltip {{
      position: fixed;
      max-width: 280px;
      z-index: 3;
      display: none;
      padding: 9px 10px;
      border-radius: 8px;
      background: var(--fg);
      color: var(--bg);
      box-shadow: 0 10px 24px var(--shadow);
      font-size: 0.88rem;
      pointer-events: none;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 18px;
    }}
    .table-scroll {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.92rem;
    }}
    th, td {{
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-weight: 650;
      background: color-mix(in srgb, var(--panel) 90%, var(--bg));
    }}
    td:first-child, th:first-child {{ padding-left: 0; }}
    .profile {{
      display: grid;
      gap: 8px;
    }}
    .profile-row {{
      display: grid;
      grid-template-columns: minmax(70px, 110px) 1fr;
      gap: 10px;
      align-items: center;
    }}
    .profile-label {{
      font-size: 0.88rem;
      color: var(--fg);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .bar {{
      height: 20px;
      display: flex;
      gap: 2px;
      min-width: 160px;
    }}
    .segment {{
      min-width: 2px;
      background: var(--accent);
      opacity: 0.85;
    }}
    .segment:nth-child(3n + 1) {{ background: var(--accent); }}
    .segment:nth-child(3n + 2) {{ background: var(--accent-2); }}
    .segment:nth-child(3n + 3) {{ background: var(--accent-3); }}
    footer {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto 28px;
      color: var(--muted);
      font-size: 0.9rem;
    }}
    @media (max-width: 680px) {{
      th, td {{ padding: 7px 8px; }}
      .profile-row {{ grid-template-columns: 1fr; }}
      .bar {{ min-width: 0; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(graph.name)}</h1>
    <p>Multiplex graph influence report generated from a four-column layer/source/target/weight edge list.</p>
    <div class="meta">
      <span class="pill">{len(graph.nodes):,} nodes</span>
      <span class="pill">{len(graph.edges):,} raw edges</span>
      <span class="pill">{len(graph.layer_edges):,} layers</span>
      <span class="pill">{'Directed' if graph.directed else 'Undirected'} analysis</span>
      <span class="pill">{len(selected):,} visualized nodes</span>
    </div>
  </header>

  <main>
    <section>
      <h2>Interactive network</h2>
      <div class="controls">
        <label>Layer
          <select id="layerSelect"></select>
        </label>
        <label class="check">
          <input type="checkbox" id="labelToggle" checked>
          Show labels for top nodes
        </label>
      </div>
      <div class="network-frame">
        <svg id="networkSvg" viewBox="0 0 1060 620" role="img" aria-label="Multiplex graph visualization">
          <defs>
            <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor"></path>
            </marker>
          </defs>
        </svg>
      </div>
      <p>Node size is based on the composite influence score. The visualization keeps the highest-ranked nodes and their strongest local neighborhood readable.</p>
    </section>

    <section>
      <h2>Layer profile of top nodes</h2>
      <div id="profile" class="profile" aria-label="Layer contribution bars"></div>
    </section>

    <section class="grid">
      <div>
        <h2>Top influential nodes</h2>
        <div class="table-scroll">{top_table}</div>
      </div>
      <div>
        <h2>Layer summary</h2>
        <div class="table-scroll">{layer_table}</div>
      </div>
    </section>
  </main>

  <footer>
    Influence score = 35% strength, 25% PageRank, 20% betweenness, 10% layer coverage, and 10% layer participation.
  </footer>

  <div id="tooltip" class="tooltip"></div>
  <script>
    const DATA = {json.dumps(data, separators=(",", ":"))};
    const svg = document.getElementById("networkSvg");
    const select = document.getElementById("layerSelect");
    const labelToggle = document.getElementById("labelToggle");
    const tooltip = document.getElementById("tooltip");
    const nodeById = new Map(DATA.nodes.map(node => [node.id, node]));

    function escapeText(value) {{
      return String(value).replace(/[&<>"']/g, char => ({{
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }}[char]));
    }}

    function colorForNode(node) {{
      if (node.top) return "var(--top)";
      const alpha = Math.max(0.35, Math.min(0.95, node.score * 2.4));
      return `color-mix(in srgb, var(--accent) ${{Math.round(alpha * 100)}}%, var(--accent-3))`;
    }}

    function edgeWidth(edge) {{
      return Math.max(0.7, Math.min(4.5, 0.6 + Math.log1p(edge.weight)));
    }}

    function tooltipPosition(event, node) {{
      if (Number.isFinite(event.clientX) && Number.isFinite(event.clientY) && event.clientX > 0 && event.clientY > 0) {{
        return [event.clientX + 12, event.clientY + 12];
      }}
      const rect = svg.getBoundingClientRect();
      return [
        rect.left + (node.x / 1060) * rect.width + 12,
        rect.top + (node.y / 620) * rect.height + 12
      ];
    }}

    function showTooltip(event, node) {{
      const [left, top] = tooltipPosition(event, node);
      tooltip.innerHTML = `
        <strong>Node ${{escapeText(node.id)}}</strong><br>
        Rank: ${{node.rank}}<br>
        Influence: ${{node.scoreLabel}}<br>
        Strength: ${{node.strength.toLocaleString()}}<br>
        Degree: ${{node.degree.toLocaleString()}}<br>
        PageRank: ${{node.pagerank.toExponential(3)}}<br>
        Layers: ${{node.layerCount}}
      `;
      tooltip.style.left = `${{left}}px`;
      tooltip.style.top = `${{top}}px`;
      tooltip.style.display = "block";
    }}

    function hideTooltip() {{
      tooltip.style.display = "none";
    }}

    function render() {{
      const layer = select.value || "Aggregate";
      const edges = DATA.edgesByLayer[layer] || [];
      while (svg.lastChild && svg.lastChild.nodeName !== "defs") {{
        svg.removeChild(svg.lastChild);
      }}

      const edgeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      edgeGroup.setAttribute("aria-hidden", "true");
      edges.forEach(edge => {{
        const source = nodeById.get(edge.source);
        const target = nodeById.get(edge.target);
        if (!source || !target) return;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const distance = Math.max(1, Math.hypot(dx, dy));
        const trim = DATA.directed ? Math.max(8, target.radius + 5) : 0;
        line.setAttribute("x1", source.x);
        line.setAttribute("y1", source.y);
        line.setAttribute("x2", target.x - (dx / distance) * trim);
        line.setAttribute("y2", target.y - (dy / distance) * trim);
        line.setAttribute("class", "edge");
        line.setAttribute("stroke-width", edgeWidth(edge));
        line.setAttribute("opacity", layer === "Aggregate" ? "0.42" : "0.58");
        if (DATA.directed) line.setAttribute("marker-end", "url(#arrow)");
        edgeGroup.appendChild(line);
      }});
      svg.appendChild(edgeGroup);

      const nodeGroup = document.createElementNS("http://www.w3.org/2000/svg", "g");
      DATA.nodes.forEach(node => {{
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", node.x);
        circle.setAttribute("cy", node.y);
        circle.setAttribute("r", node.radius);
        circle.setAttribute("fill", colorForNode(node));
        circle.setAttribute("class", `node${{node.top ? " top" : ""}}`);
        circle.setAttribute("tabindex", "0");
        circle.setAttribute("aria-label", `Node ${{node.id}}, rank ${{node.rank}}, influence ${{node.scoreLabel}}`);
        circle.addEventListener("mousemove", event => showTooltip(event, node));
        circle.addEventListener("mouseleave", hideTooltip);
        circle.addEventListener("focus", event => showTooltip(event, node));
        circle.addEventListener("blur", hideTooltip);
        nodeGroup.appendChild(circle);

        if (labelToggle.checked && node.top) {{
          const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
          label.setAttribute("x", node.x + node.radius + 4);
          label.setAttribute("y", node.y + 4);
          label.setAttribute("class", "node-label");
          label.textContent = node.id;
          nodeGroup.appendChild(label);
        }}
      }});
      svg.appendChild(nodeGroup);
    }}

    function renderProfile() {{
      const root = document.getElementById("profile");
      const layers = DATA.layers.filter(layer => layer !== "Aggregate");
      root.innerHTML = "";
      DATA.layerProfile.forEach(row => {{
        const item = document.createElement("div");
        item.className = "profile-row";
        const label = document.createElement("div");
        label.className = "profile-label";
        label.textContent = `#${{row.rank}} node ${{row.node}}`;
        const bar = document.createElement("div");
        bar.className = "bar";
        layers.forEach(layer => {{
          const value = row.layers[layer] || 0;
          const segment = document.createElement("div");
          segment.className = "segment";
          segment.style.width = `${{value > 0 && row.total > 0 ? Math.max(2, (value / row.total) * 100) : 0}}%`;
          segment.title = `Layer ${{layer}}: ${{value.toLocaleString()}}`;
          bar.appendChild(segment);
        }});
        item.appendChild(label);
        item.appendChild(bar);
        root.appendChild(item);
      }});
    }}

    DATA.layers.forEach(layer => {{
      const option = document.createElement("option");
      option.value = layer;
      option.textContent = layer;
      select.appendChild(option);
    }});
    select.addEventListener("change", render);
    labelToggle.addEventListener("change", render);
    renderProfile();
    render();
  </script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_html, encoding="utf-8")


def process_dataset(
    path: Path,
    output_dir: Path,
    directed: bool,
    top_k: int,
    max_nodes: int,
    max_edges_per_layer: int,
    betweenness_samples: int,
    seed: int,
) -> Dict[str, object]:
    graph = build_graph(path, directed=directed)
    ranking = compute_influence_metrics(
        graph,
        betweenness_samples=betweenness_samples,
        seed=seed,
    )
    layer_summary = summarize_layers(graph)
    slug = slugify(graph.name)

    influence_csv = output_dir / f"{slug}_influence.csv"
    layer_csv = output_dir / f"{slug}_layer_summary.csv"
    report_html = output_dir / f"{slug}_report.html"

    write_csv(
        influence_csv,
        ranking,
        [
            "rank",
            "node",
            "influence_score",
            "total_strength",
            "out_strength",
            "in_strength",
            "degree",
            "pagerank",
            "betweenness",
            "layer_count",
            "participation",
        ],
    )
    write_csv(
        layer_csv,
        layer_summary,
        ["layer", "nodes", "edges", "self_loops", "total_weight"],
    )
    make_report(
        graph,
        ranking,
        layer_summary,
        report_html,
        top_k=top_k,
        max_nodes=max_nodes,
        max_edges_per_layer=max_edges_per_layer,
        seed=seed,
    )

    top_node = ranking[0] if ranking else {}
    return {
        "name": graph.name,
        "nodes": len(graph.nodes),
        "edges": len(graph.edges),
        "layers": len(graph.layer_edges),
        "top_node": top_node.get("node", ""),
        "top_score": top_node.get("influence_score", 0.0),
        "report": report_html.name,
        "influence_csv": influence_csv.name,
        "layer_csv": layer_csv.name,
    }


def write_index(output_dir: Path, summaries: Sequence[Dict[str, object]]) -> Path:
    rows = []
    for summary in summaries:
        rows.append(
            "<tr>"
            f"<td><a href=\"{html.escape(str(summary['report']))}\">{html.escape(str(summary['name']))}</a></td>"
            f"<td>{int(summary['nodes']):,}</td>"
            f"<td>{int(summary['edges']):,}</td>"
            f"<td>{int(summary['layers']):,}</td>"
            f"<td>{html.escape(str(summary['top_node']))}</td>"
            f"<td>{format_number(float(summary['top_score']), 4)}</td>"
            f"<td><a href=\"{html.escape(str(summary['influence_csv']))}\">CSV</a></td>"
            "</tr>"
        )

    index_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multiplex graph reports</title>
  <style>
    :root {{ color-scheme: light; --bg: #f7f8fb; --fg: #17202a; --panel: #fff; --line: #d8e0e7; --muted: #5e6b78; }}
    body {{ margin: 0; background: var(--bg); color: var(--fg); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    main {{ width: min(1080px, calc(100% - 32px)); margin: 0 auto; padding: 32px 0; }}
    h1 {{ margin: 0 0 8px; }}
    p {{ margin: 0 0 22px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--panel); border: 1px solid var(--line); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
    th {{ color: var(--muted); }}
    a {{ color: inherit; font-weight: 650; }}
    .scroll {{ overflow-x: auto; }}
  </style>
</head>
<body>
  <main>
    <h1>Multiplex graph reports</h1>
    <p>Open a report to explore layer-specific visualization and node influence rankings.</p>
    <div class="scroll">
      <table>
        <thead>
          <tr>
            <th>Dataset</th>
            <th>Nodes</th>
            <th>Edges</th>
            <th>Layers</th>
            <th>Top node</th>
            <th>Top score</th>
            <th>Ranking</th>
          </tr>
        </thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
  </main>
</body>
</html>
"""
    output_path = output_dir / "index.html"
    output_path.write_text(index_html, encoding="utf-8")
    return output_path


def collect_input_files(input_path: Path | None, data_dir: Path | None) -> List[Path]:
    if input_path:
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        return [input_path]

    if data_dir:
        if not data_dir.exists():
            raise FileNotFoundError(data_dir)
        files = sorted(data_dir.glob("*.edges"), key=lambda path: natural_key(path.name))
        if not files:
            raise FileNotFoundError(f"No .edges files found in {data_dir}")
        return files

    files = sorted(Path.cwd().glob("*.edges"), key=lambda path: natural_key(path.name))
    if not files:
        raise FileNotFoundError(
            "No input was provided and no .edges files were found in the current directory"
        )
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create influence rankings and interactive HTML visualizations for multiplex graphs."
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--input", type=Path, help="Path to one .edges file")
    input_group.add_argument("--data-dir", type=Path, help="Directory containing .edges files")
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="Output directory")
    parser.add_argument("--top-k", type=int, default=20, help="Number of top nodes to highlight")
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=160,
        help="Maximum number of nodes included in each HTML network visualization",
    )
    parser.add_argument(
        "--max-edges-per-layer",
        type=int,
        default=1400,
        help="Maximum visualized edges per layer in the HTML report",
    )
    parser.add_argument(
        "--betweenness-samples",
        type=int,
        default=128,
        help="Pivot nodes for approximate betweenness when a graph has more than 500 nodes; use 0 for exact",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for layout and sampling")
    direction_group = parser.add_mutually_exclusive_group()
    direction_group.add_argument(
        "--directed",
        action="store_true",
        help="Preserve source-to-target edge direction and draw arrows.",
    )
    direction_group.add_argument(
        "--undirected",
        action="store_true",
        help="Treat edges as undirected. This is the default; the flag is kept for clarity.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.max_nodes < args.top_k:
        parser.error("--max-nodes must be greater than or equal to --top-k")

    input_files = collect_input_files(args.input, args.data_dir)
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for path in input_files:
        summary = process_dataset(
            path=path,
            output_dir=output_dir,
            directed=args.directed,
            top_k=args.top_k,
            max_nodes=args.max_nodes,
            max_edges_per_layer=args.max_edges_per_layer,
            betweenness_samples=args.betweenness_samples,
            seed=args.seed,
        )
        summaries.append(summary)
        print(
            f"Wrote {summary['report']} | top node {summary['top_node']} "
            f"(score {format_number(float(summary['top_score']), 4)})"
        )

    if len(summaries) > 1:
        index_path = write_index(output_dir, summaries)
        print(f"Wrote {index_path}")


if __name__ == "__main__":
    main()
