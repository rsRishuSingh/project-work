"""
Baseline Comparison Methods for Multilayer Network Influential Node Identification.

Implements four classical baseline methods from the paper:
  1. sumDC — Sum of Degree Centrality across layers
  2. sumBC — Sum of Betweenness Centrality across layers
  3. sumPR — Sum of PageRank across layers
  4. aggKS — K-shell on Aggregated (flattened) network
"""

import networkx as nx
import numpy as np


def compute_sumDC(network):
    """
    sumDC: Sum of degree centrality across all layers.

    For each node, calculates degree centrality in each layer
    and sums them to get the overall importance score.

    Args:
        network: MultiplexNetwork object.

    Returns:
        dict: {node_id: sumDC_score}
    """
    scores = {n: 0.0 for n in network.nodes}

    for lid in network.layer_ids:
        G = network.get_layer(lid)
        dc = nx.degree_centrality(G)
        for n in network.nodes:
            scores[n] += dc.get(n, 0.0)

    return scores


def compute_sumBC(network):
    """
    sumBC: Sum of betweenness centrality across all layers.

    For each node, calculates betweenness centrality in each layer
    and sums them.

    Args:
        network: MultiplexNetwork object.

    Returns:
        dict: {node_id: sumBC_score}
    """
    scores = {n: 0.0 for n in network.nodes}

    for lid in network.layer_ids:
        G = network.get_layer(lid)
        bc = nx.betweenness_centrality(G)
        for n in network.nodes:
            scores[n] += bc.get(n, 0.0)

    return scores


def compute_sumPR(network):
    """
    sumPR: Sum of PageRank across all layers.

    For each node, calculates PageRank in each layer and sums them.

    Args:
        network: MultiplexNetwork object.

    Returns:
        dict: {node_id: sumPR_score}
    """
    scores = {n: 0.0 for n in network.nodes}

    for lid in network.layer_ids:
        G = network.get_layer(lid)
        # Only compute PageRank if the graph has edges
        if G.number_of_edges() > 0:
            pr = nx.pagerank(G)
        else:
            pr = {n: 1.0 / network.num_nodes for n in network.nodes}
        for n in network.nodes:
            scores[n] += pr.get(n, 0.0)

    return scores


def compute_aggKS(network):
    """
    aggKS: K-shell on aggregated (flattened) network.

    Merges all layers into a single graph (union of all edges),
    then computes the K-shell (core number) on the aggregated graph.

    Args:
        network: MultiplexNetwork object.

    Returns:
        dict: {node_id: aggKS_score}
    """
    # Build the aggregated graph
    agg_G = nx.Graph()
    agg_G.add_nodes_from(network.nodes)

    for lid in network.layer_ids:
        G = network.get_layer(lid)
        agg_G.add_edges_from(G.edges())

    # Compute core numbers on aggregated graph
    core_numbers = nx.core_number(agg_G)

    scores = {}
    for n in network.nodes:
        scores[n] = core_numbers.get(n, 0)

    return scores


def run_all_baselines(network, verbose=False):
    """
    Run all four baseline methods on a network.

    Args:
        network: MultiplexNetwork object.
        verbose (bool): If True, print progress.

    Returns:
        dict: {method_name: {node_id: score}}
    """
    results = {}

    if verbose:
        print(f"\n  Running baseline methods on '{network.name}'...")

    methods = [
        ("sumDC", compute_sumDC),
        ("sumBC", compute_sumBC),
        ("sumPR", compute_sumPR),
        ("aggKS", compute_aggKS),
    ]

    for name, func in methods:
        if verbose:
            print(f"    Computing {name}...")
        results[name] = func(network)

    return results


def get_ranking_from_scores(scores):
    """
    Convert scores to a ranking dictionary.

    Args:
        scores (dict): {node_id: score}

    Returns:
        list: Sorted list of (node_id, score) descending.
        dict: {node_id: rank} (1 = highest score).
    """
    sorted_nodes = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranking = {node: rank for rank, (node, _) in enumerate(sorted_nodes, 1)}
    return sorted_nodes, ranking
