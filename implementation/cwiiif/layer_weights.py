"""
Phase 1: Layer Weighted Coefficients (LWC)

Computes three normalized factors for each layer:
  - NAN (Normalized Active Nodes): Fraction of nodes with above-average degree
  - NAP (Normalized Active Path): Average shortest-path length among non-isolated nodes
  - NICI (Normalized Inter-layer Communication Intersection): Jaccard edge overlap

Layer weight: W(α) = NAN[α] + NAP[α] + NICI[α]
"""

import networkx as nx
import numpy as np
from collections import defaultdict


def compute_active_nodes(network):
    """
    Compute the number of active nodes per layer.

    A node i is 'active' in layer α if its degree K[α]_i exceeds
    the average degree <K[α]> of that layer (Definition 1 in paper).

    Args:
        network: MultiplexNetwork object.

    Returns:
        dict: {layer_id: active_node_count}
    """
    active_counts = {}

    for lid in network.layer_ids:
        G = network.get_layer(lid)
        degrees = [G.degree(n) for n in network.nodes]
        avg_degree = np.mean(degrees)

        # Count nodes with degree strictly greater than the average
        active_count = sum(1 for d in degrees if d > avg_degree)
        active_counts[lid] = active_count

    return active_counts


def compute_NAN(network):
    """
    Compute Normalized Active Nodes (NAN) for each layer.

    NAN[α] = b[α] / Σ_β b[β]     (Equation 4)

    Args:
        network: MultiplexNetwork object.

    Returns:
        dict: {layer_id: NAN_value}
    """
    active_counts = compute_active_nodes(network)
    total_active = sum(active_counts.values())

    NAN = {}
    for lid in network.layer_ids:
        if total_active > 0:
            NAN[lid] = active_counts[lid] / total_active
        else:
            NAN[lid] = 1.0 / network.num_layers  # Equal weight if no active nodes

    return NAN


def compute_average_active_path_length(network):
    """
    Compute the average active path length for each layer.

    An 'active path' exists between nodes i and j only if both are
    non-isolated (Definition 2). The average active path length is:

    d[α] = Σ_{i,j; j≠i} d[α]_ij / (2N)     (Equation 5)

    where N is the total number of nodes (including isolated ones),
    and only paths between non-isolated nodes are summed.

    We use BFS (O(N*E)) since networks are unweighted.

    Args:
        network: MultiplexNetwork object.

    Returns:
        dict: {layer_id: average_active_path_length}
    """
    avg_path_lengths = {}
    N = network.num_nodes

    for lid in network.layer_ids:
        G = network.get_layer(lid)

        # Identify non-isolated nodes
        non_isolated = [n for n in network.nodes if G.degree(n) > 0]

        if len(non_isolated) < 2:
            avg_path_lengths[lid] = 0.0
            continue

        # Compute shortest paths between all pairs of non-isolated nodes
        # Only within connected components
        total_path_length = 0.0

        for source in non_isolated:
            lengths = nx.single_source_shortest_path_length(G, source)
            for target, dist in lengths.items():
                if target != source and target in non_isolated:
                    total_path_length += dist

        # Equation 5: divide by 2N (N = total nodes including isolated)
        avg_path_lengths[lid] = total_path_length / (2 * N)

    return avg_path_lengths


def compute_NAP(network):
    """
    Compute Normalized Active Path (NAP) for each layer.

    NAP[α] = d[α] / Σ_β d[β]     (Equation 6)

    Args:
        network: MultiplexNetwork object.

    Returns:
        dict: {layer_id: NAP_value}
    """
    avg_path_lengths = compute_average_active_path_length(network)
    total_path = sum(avg_path_lengths.values())

    NAP = {}
    for lid in network.layer_ids:
        if total_path > 0:
            NAP[lid] = avg_path_lengths[lid] / total_path
        else:
            NAP[lid] = 1.0 / network.num_layers

    return NAP


def compute_jaccard_coefficients(network):
    """
    Compute the Jaccard coefficient for each layer against all other layers.

    Jaccard[α] = Σ_{β≠α} |E[α] ∩ E[β]| / |E[α] ∪ E[β]|     (Equation 7)

    Edges are compared as undirected (frozensets).

    Args:
        network: MultiplexNetwork object.

    Returns:
        dict: {layer_id: jaccard_sum}
    """
    # Pre-compute edge sets for all layers
    edge_sets = {}
    for lid in network.layer_ids:
        edge_sets[lid] = network.get_edge_set(lid)

    jaccard_values = {}

    for alpha in network.layer_ids:
        jaccard_sum = 0.0
        for beta in network.layer_ids:
            if beta == alpha:
                continue

            E_alpha = edge_sets[alpha]
            E_beta = edge_sets[beta]

            intersection = len(E_alpha & E_beta)
            union = len(E_alpha | E_beta)

            if union > 0:
                jaccard_sum += intersection / union

        jaccard_values[alpha] = jaccard_sum

    return jaccard_values


def compute_NICI(network):
    """
    Compute Normalized Inter-layer Communication Intersection (NICI).

    NICI[α] = Jaccard[α] / Σ_β Jaccard[β]     (Equation 8)

    Args:
        network: MultiplexNetwork object.

    Returns:
        dict: {layer_id: NICI_value}
    """
    jaccard_values = compute_jaccard_coefficients(network)
    total_jaccard = sum(jaccard_values.values())

    NICI = {}
    for lid in network.layer_ids:
        if total_jaccard > 0:
            NICI[lid] = jaccard_values[lid] / total_jaccard
        else:
            # No overlap at all: assign equal weights
            NICI[lid] = 1.0 / network.num_layers

    return NICI


def compute_layer_weights(network, verbose=False):
    """
    Compute the full Layer Weighted Coefficients (LWC) for all layers.

    W(α) = NAN[α] + NAP[α] + NICI[α]     (Equation 9)

    This is Algorithm 1 from the paper.

    Args:
        network: MultiplexNetwork object.
        verbose (bool): If True, print intermediate values.

    Returns:
        dict: {layer_id: W_value}
        dict: Detailed breakdown {layer_id: {'NAN': ..., 'NAP': ..., 'NICI': ..., 'W': ...}}
    """
    NAN = compute_NAN(network)
    NAP = compute_NAP(network)
    NICI = compute_NICI(network)

    weights = {}
    details = {}

    for lid in network.layer_ids:
        W = NAN[lid] + NAP[lid] + NICI[lid]
        weights[lid] = W
        details[lid] = {
            'NAN': NAN[lid],
            'NAP': NAP[lid],
            'NICI': NICI[lid],
            'W': W
        }

    if verbose:
        print(f"\n  Layer Weight Coefficients for '{network.name}':")
        print(f"  {'Layer':>6} | {'NAN':>8} | {'NAP':>8} | {'NICI':>8} | {'W(α)':>8}")
        print(f"  {'─'*6} | {'─'*8} | {'─'*8} | {'─'*8} | {'─'*8}")
        for lid in network.layer_ids:
            d = details[lid]
            print(f"  {lid:>6} | {d['NAN']:>8.4f} | {d['NAP']:>8.4f} | {d['NICI']:>8.4f} | {d['W']:>8.4f}")

    return weights, details
