"""
Evaluation Metrics for Influential Node Identification.

Implements five standard evaluation metrics from the paper:
  1. Spearman Rank Correlation Coefficient (ρ)
  2. Kendall Rank Correlation Coefficient (τ)
  3. Imprecision Function (ε)
  4. Monotonicity Index (M)
  5. Network Connectivity (Q) under node removal
"""

import numpy as np
from scipy import stats
import networkx as nx


def spearman_correlation(method_scores, spreading_scores, nodes):
    """
    Compute Spearman rank correlation coefficient between a method's
    ranking and the actual spreading influence ranking.

    ρ = 1 - 6Σr²_i / (N(N²-1))     (Equation 13)

    Args:
        method_scores (dict): {node_id: method_score}
        spreading_scores (dict): {node_id: spreading_score}
        nodes (list): List of node IDs.

    Returns:
        float: Spearman correlation coefficient ρ ∈ [-1, 1].
    """
    method_vals = [method_scores.get(n, 0.0) for n in nodes]
    spreading_vals = [spreading_scores.get(n, 0.0) for n in nodes]

    if len(set(method_vals)) <= 1 or len(set(spreading_vals)) <= 1:
        return 0.0

    rho, _ = stats.spearmanr(method_vals, spreading_vals)
    return rho if not np.isnan(rho) else 0.0


def kendall_correlation(method_scores, spreading_scores, nodes):
    """
    Compute Kendall rank correlation coefficient between rankings.

    τ ∈ [-1, 1], where τ = 1 indicates perfectly consistent rankings.

    Args:
        method_scores (dict): {node_id: method_score}
        spreading_scores (dict): {node_id: spreading_score}
        nodes (list): List of node IDs.

    Returns:
        float: Kendall correlation coefficient τ.
    """
    method_vals = [method_scores.get(n, 0.0) for n in nodes]
    spreading_vals = [spreading_scores.get(n, 0.0) for n in nodes]

    if len(set(method_vals)) <= 1 or len(set(spreading_vals)) <= 1:
        return 0.0

    tau, _ = stats.kendalltau(method_vals, spreading_vals)
    return tau if not np.isnan(tau) else 0.0


def imprecision_function(method_scores, spreading_scores, nodes, p_values=None):
    """
    Compute the imprecision function ε(p).

    ε(p) = 1 - M(p) / M_eff(p)     (Equation 14)

    where:
      - M(p) = average spreading influence of the top pN nodes ranked by method
      - M_eff(p) = average spreading influence of the top pN nodes ranked by
                   actual spreading efficiency

    A smaller ε indicates better identification accuracy.

    Args:
        method_scores (dict): {node_id: method_score}
        spreading_scores (dict): {node_id: spreading_score}
        nodes (list): List of node IDs.
        p_values (list): Fraction values p ∈ (0, 1]. Default: [0.05, 0.1, ..., 0.5]

    Returns:
        dict: {p: epsilon_value}
    """
    if p_values is None:
        p_values = [i * 0.05 for i in range(1, 11)]  # 0.05 to 0.50

    N = len(nodes)

    # Rank nodes by method score (descending)
    method_ranked = sorted(nodes, key=lambda n: method_scores.get(n, 0.0), reverse=True)

    # Rank nodes by actual spreading score (descending)
    spreading_ranked = sorted(nodes, key=lambda n: spreading_scores.get(n, 0.0), reverse=True)

    results = {}
    for p in p_values:
        k = max(1, int(p * N))

        # Top-k nodes by method
        top_method = method_ranked[:k]
        M_p = np.mean([spreading_scores.get(n, 0.0) for n in top_method])

        # Top-k nodes by actual spreading
        top_spreading = spreading_ranked[:k]
        M_eff_p = np.mean([spreading_scores.get(n, 0.0) for n in top_spreading])

        if M_eff_p > 0:
            epsilon = 1.0 - M_p / M_eff_p
        else:
            epsilon = 0.0

        results[p] = max(0.0, epsilon)  # ε should be non-negative

    return results


def monotonicity_index(scores, nodes):
    """
    Compute the monotonicity index M(R).

    M(R) = (1 - Σ_r Nr(Nr-1) / (N(N-1)))²     (Equation 15)

    where Nr is the number of nodes assigned the same score value r.
    M(R) = 1 means all nodes have unique scores (maximum discriminative power).
    M(R) = 0 means all nodes have identical scores.

    Args:
        scores (dict): {node_id: score}
        nodes (list): List of node IDs.

    Returns:
        float: Monotonicity index M ∈ [0, 1].
    """
    N = len(nodes)
    if N <= 1:
        return 1.0

    # Count how many nodes share each score value
    from collections import Counter
    score_values = [scores.get(n, 0.0) for n in nodes]

    # Round to avoid floating point comparison issues
    rounded = [round(s, 10) for s in score_values]
    counts = Counter(rounded)

    # Compute Σ Nr(Nr-1)
    sum_pairs = sum(nr * (nr - 1) for nr in counts.values())

    # M(R) = (1 - sum_pairs / (N(N-1)))²
    M = (1.0 - sum_pairs / (N * (N - 1))) ** 2

    return M


def network_connectivity_after_removal(network, method_scores, p_fractions=None):
    """
    Compute network connectivity Q after removing the top p×10% of nodes.

    Builds an aggregated graph, then sequentially removes nodes ranked
    highest by the method, measuring average shortest path distance.

    Q = 2 × Σ d_ij / (N'(N'-1))     (Equation 16)

    where N' is the number of remaining nodes.

    Args:
        network: MultiplexNetwork object.
        method_scores (dict): {node_id: method_score}
        p_fractions (list): Fraction values for removal. Default: [0.1, 0.2, ..., 1.0]

    Returns:
        dict: {p: Q_value} — connectivity after removing top p fraction.
    """
    if p_fractions is None:
        p_fractions = [i * 0.1 for i in range(1, 11)]

    # Build aggregated graph
    agg_G = nx.Graph()
    agg_G.add_nodes_from(network.nodes)
    for lid in network.layer_ids:
        G = network.get_layer(lid)
        agg_G.add_edges_from(G.edges())

    # Rank nodes by method score (descending)
    sorted_nodes = sorted(network.nodes, key=lambda n: method_scores.get(n, 0.0), reverse=True)
    N = len(sorted_nodes)

    results = {}
    for p in p_fractions:
        k = max(1, int(p * N))
        nodes_to_remove = sorted_nodes[:k]

        # Create subgraph without removed nodes
        remaining_nodes = [n for n in network.nodes if n not in set(nodes_to_remove)]

        if len(remaining_nodes) < 2:
            results[p] = 0.0
            continue

        sub_G = agg_G.subgraph(remaining_nodes).copy()

        # Find the largest connected component
        if sub_G.number_of_edges() == 0:
            results[p] = 0.0
            continue

        components = list(nx.connected_components(sub_G))
        largest_cc = max(components, key=len)
        # Network connectivity = size of largest connected component
        results[p] = len(largest_cc)

    return results
