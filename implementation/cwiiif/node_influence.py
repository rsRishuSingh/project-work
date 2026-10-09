"""
Phase 2: Intra-layer Influence Vector (Node Influence Computation)

For each node i in each layer α, combines:
  - KS[α]_i: Normalized K-shell value (structural coreness)
  - BC[α]_i: Normalized betweenness centrality (bridge importance)

KB[α]_i = KS[α]_i + BC[α]_i     (Equation 10)
"""

import networkx as nx
import numpy as np


def compute_kshell_values(G, nodes):
    """
    Compute K-shell (core number) for all nodes in a graph.

    K-shell decomposition iteratively removes nodes with the lowest degree,
    assigning each node a coreness value. Uses NetworkX's core_number().

    Args:
        G (nx.Graph): The layer graph.
        nodes (list): Full list of node IDs in the network.

    Returns:
        dict: {node_id: kshell_value}
    """
    core_numbers = nx.core_number(G)

    # Ensure all nodes are represented (isolated nodes get core number 0)
    kshell = {}
    for n in nodes:
        kshell[n] = core_numbers.get(n, 0)

    return kshell


def compute_betweenness_values(G, nodes):
    """
    Compute betweenness centrality for all nodes in a graph.

    Uses Brandes' algorithm (O(N*E) for unweighted graphs) via NetworkX.
    The normalized=False option gives the raw sum of fractions σ_st(v)/σ_st.

    Args:
        G (nx.Graph): The layer graph.
        nodes (list): Full list of node IDs in the network.

    Returns:
        dict: {node_id: betweenness_value}
    """
    # Use normalized=False to get raw betweenness values
    bc = nx.betweenness_centrality(G, normalized=False)

    # Ensure all nodes are represented
    betweenness = {}
    for n in nodes:
        betweenness[n] = bc.get(n, 0.0)

    return betweenness


def compute_normalized_kshell(G, nodes):
    """
    Compute normalized K-shell values for a layer.

    KS[α]_i = kshell(i) / Σ_j kshell(j)

    Args:
        G (nx.Graph): The layer graph.
        nodes (list): Full list of node IDs.

    Returns:
        dict: {node_id: normalized_kshell_value}
    """
    kshell = compute_kshell_values(G, nodes)
    total_ks = sum(kshell.values())

    normalized = {}
    for n in nodes:
        if total_ks > 0:
            normalized[n] = kshell[n] / total_ks
        else:
            normalized[n] = 0.0

    return normalized


def compute_normalized_betweenness(G, nodes):
    """
    Compute normalized betweenness centrality for a layer.

    BC[α]_i = betweenness(i) / Σ_j betweenness(j)

    Args:
        G (nx.Graph): The layer graph.
        nodes (list): Full list of node IDs.

    Returns:
        dict: {node_id: normalized_betweenness_value}
    """
    betweenness = compute_betweenness_values(G, nodes)
    total_bc = sum(betweenness.values())

    normalized = {}
    for n in nodes:
        if total_bc > 0:
            normalized[n] = betweenness[n] / total_bc
        else:
            normalized[n] = 0.0

    return normalized


def compute_intra_layer_influence(network, verbose=False):
    """
    Compute the intra-layer influence vector KB for all nodes across all layers.

    KB[α]_i = KS[α]_i + BC[α]_i     (Equation 10)

    The full influence vector for node i is:
    KB_i = [KB[1]_i, KB[2]_i, ..., KB[L]_i]     (Equation 11)

    Args:
        network: MultiplexNetwork object.
        verbose (bool): If True, print progress.

    Returns:
        dict: {layer_id: {node_id: KB_value}}
              KB[layer_id][node_id] = intra-layer influence score
    """
    KB = {}

    for lid in network.layer_ids:
        G = network.get_layer(lid)

        if verbose:
            print(f"    Computing K-shell + Betweenness for layer {lid} "
                  f"({G.number_of_edges()} edges)...")

        # Compute normalized metrics
        norm_ks = compute_normalized_kshell(G, network.nodes)
        norm_bc = compute_normalized_betweenness(G, network.nodes)

        # Combine: KB[α]_i = KS[α]_i + BC[α]_i
        KB[lid] = {}
        for n in network.nodes:
            KB[lid][n] = norm_ks[n] + norm_bc[n]

    return KB


def get_influence_vector(KB, node, layer_ids):
    """
    Get the influence vector for a specific node across all layers.

    Args:
        KB (dict): The full KB dictionary from compute_intra_layer_influence.
        node: The node ID.
        layer_ids (list): Ordered list of layer IDs.

    Returns:
        list: [KB[1]_node, KB[2]_node, ..., KB[L]_node]
    """
    return [KB[lid][node] for lid in layer_ids]
