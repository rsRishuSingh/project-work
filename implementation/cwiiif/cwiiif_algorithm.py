"""
CWIIIF Algorithm — Full Pipeline

Combines all three phases:
  Phase 1: Layer Weight Coefficients (LWC)
  Phase 2: Intra-layer Influence Vectors (KB)
  Phase 3: Global Integration (TI = Σ KB[α]_i × W(α))

Produces the final Total Influence (TI) ranking for all nodes.
"""

import numpy as np
from .layer_weights import compute_layer_weights
from .node_influence import compute_intra_layer_influence


def run_cwiiif(network, verbose=False):
    """
    Execute the full CWIIIF algorithm on a multiplex network.

    This is Algorithm 2 from the paper.

    Args:
        network: MultiplexNetwork object.
        verbose (bool): If True, print detailed progress.

    Returns:
        dict: {node_id: TI_score} — Total Influence scores.
        list: Ranked list of (node_id, TI_score) sorted by descending TI.
        dict: Layer weight details {layer_id: {'NAN': ..., 'NAP': ..., 'NICI': ..., 'W': ...}}.
    """
    if verbose:
        print(f"\n{'═'*60}")
        print(f"  Running CWIIIF on: {network.name}")
        print(f"  N={network.num_nodes}, L={network.num_layers}, |E|={network.total_edges()}")
        print(f"{'═'*60}")

    # ─── Phase 1: Layer Weight Coefficients ───
    if verbose:
        print(f"\n  ▶ Phase 1: Computing Layer Weight Coefficients (LWC)...")
    weights, weight_details = compute_layer_weights(network, verbose=verbose)

    # ─── Phase 2: Intra-layer Influence Vectors ───
    if verbose:
        print(f"\n  ▶ Phase 2: Computing Intra-layer Influence Vectors (KB)...")
    KB = compute_intra_layer_influence(network, verbose=verbose)

    # ─── Phase 3: Global Integration (TI) ───
    if verbose:
        print(f"\n  ▶ Phase 3: Computing Total Influence (TI) scores...")

    TI = {}
    for node in network.nodes:
        ti_score = 0.0
        for lid in network.layer_ids:
            ti_score += KB[lid][node] * weights[lid]
        TI[node] = ti_score

    # Sort nodes by TI score (descending)
    ranking = sorted(TI.items(), key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"\n  ✓ CWIIIF complete. Top-10 influential nodes:")
        print(f"  {'Rank':>6} | {'Node':>6} | {'TI Score':>12}")
        print(f"  {'─'*6} | {'─'*6} | {'─'*12}")
        for rank, (node, score) in enumerate(ranking[:10], 1):
            print(f"  {rank:>6} | {node:>6} | {score:>12.6f}")

    return TI, ranking, weight_details


def get_node_ranking(TI):
    """
    Convert TI scores to a ranking dictionary.

    Args:
        TI (dict): {node_id: TI_score}

    Returns:
        dict: {node_id: rank} (1 = most influential)
    """
    sorted_nodes = sorted(TI.items(), key=lambda x: x[1], reverse=True)
    ranking = {}
    for rank, (node, score) in enumerate(sorted_nodes, 1):
        ranking[node] = rank
    return ranking
