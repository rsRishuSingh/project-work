"""
CWIIIF Experiment Runner
========================

Main entry point that orchestrates all experiments:
  1. Loads all 7 multiplex network datasets
  2. Runs CWIIIF algorithm + 4 baseline methods on each
  3. Runs SI spreading model simulations
  4. Computes evaluation metrics (Spearman, Imprecision, Monotonicity, Connectivity)
  5. Generates comparison tables and plots
  6. Saves results to the results/ directory

Usage:
    python run_experiments.py
    python run_experiments.py --datasets CS-Aarhus Vickers-Chan
    python run_experiments.py --skip-si       (skip slow SI simulations)
    python run_experiments.py --quick         (fewer SI runs for quick testing)
"""

import os
import sys

# Force UTF-8 encoding for standard output to avoid Windows console errors with special characters
if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import time
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from tabulate import tabulate

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cwiiif.loader import load_all_datasets, load_edges_file
from cwiiif.cwiiif_algorithm import run_cwiiif, get_node_ranking
from cwiiif.baselines import run_all_baselines, get_ranking_from_scores
from cwiiif.spreading_model import compute_spreading_influence
from cwiiif.evaluation import (
    spearman_correlation,
    kendall_correlation,
    imprecision_function,
    monotonicity_index,
    network_connectivity_after_removal,
)
from cwiiif.visualization import (
    plot_spearman_vs_lambda,
    plot_imprecision,
    plot_monotonicity,
    plot_network_connectivity,
)


# ─── Configuration ───
DATASETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
CHECKPOINTS_DIR = os.path.join(RESULTS_DIR, "checkpoints")

# SI model parameters (from the paper)
LAMBDA_VALUES = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
SI_TIME_STEPS = 10
SI_NUM_RUNS = 100        # Paper uses 100 independent runs
SI_NUM_RUNS_QUICK = 10   # Quick mode for testing

METHOD_NAMES = ['CWIIIF', 'sumDC', 'sumBC', 'sumPR', 'aggKS']


def create_output_dirs():
    """Create output directories if they don't exist."""
    for d in [RESULTS_DIR, TABLES_DIR, PLOTS_DIR, CHECKPOINTS_DIR]:
        os.makedirs(d, exist_ok=True)


def run_experiment_on_dataset(network, num_runs=100, skip_si=False, lambda_values=None):
    """
    Run the full experiment pipeline on a single dataset.

    Args:
        network: MultiplexNetwork object.
        num_runs: Number of SI simulation runs.
        skip_si: If True, skip SI simulations and related metrics.
        lambda_values: List of infection probabilities to test.

    Returns:
        dict: Complete results for this dataset.
    """
    if lambda_values is None:
        lambda_values = LAMBDA_VALUES

    results = {
        'dataset': network.name,
        'N': network.num_nodes,
        'L': network.num_layers,
        'E': network.total_edges(),
    }

    # ════════════════════════════════════════
    # Step 1: Run CWIIIF algorithm
    # ════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"  Dataset: {network.name} (N={network.num_nodes}, L={network.num_layers}, |E|={network.total_edges()})")
    print(f"{'═'*70}")

    t0 = time.time()
    TI_scores, ranking, weight_details = run_cwiiif(network, verbose=True)
    cwiiif_time = time.time() - t0
    print(f"\n  ⏱  CWIIIF completed in {cwiiif_time:.2f}s")

    results['cwiiif_time'] = cwiiif_time
    results['layer_weights'] = weight_details
    results['cwiiif_ranking'] = ranking[:20]  # Top-20

    # ════════════════════════════════════════
    # Step 2: Run baseline methods
    # ════════════════════════════════════════
    print(f"\n  ▶ Running baseline methods...")
    t0 = time.time()
    baseline_scores = run_all_baselines(network, verbose=True)
    baseline_time = time.time() - t0
    print(f"  ⏱  Baselines completed in {baseline_time:.2f}s")

    # Combine all method scores
    all_scores = {'CWIIIF': TI_scores}
    all_scores.update(baseline_scores)

    # ════════════════════════════════════════
    # Step 3: Compute Monotonicity Index
    # ════════════════════════════════════════
    print(f"\n  ▶ Computing Monotonicity Index...")
    mono_results = {}
    for method_name, scores in all_scores.items():
        mono_results[method_name] = monotonicity_index(scores, network.nodes)

    results['monotonicity'] = mono_results

    # Print monotonicity table
    print(f"\n  Monotonicity Index:")
    mono_table = [[method, f"{mono_results[method]:.6f}"] for method in METHOD_NAMES]
    print(tabulate(mono_table, headers=['Method', 'M(R)'], tablefmt='simple', stralign='right'))

    # ════════════════════════════════════════
    # Step 4: Network Connectivity Experiment
    # ════════════════════════════════════════
    print(f"\n  ▶ Computing Network Connectivity under attack...")
    connectivity_results = {}
    for method_name, scores in all_scores.items():
        connectivity_results[method_name] = network_connectivity_after_removal(
            network, scores
        )
    results['connectivity'] = connectivity_results

    # ════════════════════════════════════════
    # Step 5: SI Spreading Model & Correlation
    # ════════════════════════════════════════
    if not skip_si:
        print(f"\n  ▶ Running SI spreading simulations...")
        print(f"    (λ values: {lambda_values}, T={SI_TIME_STEPS}, runs={num_runs})")

        spearman_results = {m: {} for m in METHOD_NAMES}
        kendall_results = {m: {} for m in METHOD_NAMES}
        imprecision_results = {}

        for lam_idx, lam in enumerate(lambda_values):
            print(f"\n    λ = {lam:.2f} ({lam_idx+1}/{len(lambda_values)}):")
            t0 = time.time()

            # Run SI for all nodes
            spreading_scores = {}
            pbar = tqdm(network.nodes, desc=f"      SI (λ={lam:.2f})",
                       leave=False, ncols=80)
            for node in pbar:
                avg_counts = np.zeros(SI_TIME_STEPS + 1)
                for _ in range(num_runs):
                    from cwiiif.spreading_model import run_si_single
                    counts = run_si_single(network, node, lam, SI_TIME_STEPS)
                    avg_counts += np.array(counts)
                avg_counts /= num_runs
                spreading_scores[node] = avg_counts[-1]

            si_time = time.time() - t0
            print(f"      SI completed in {si_time:.1f}s")

            # Compute Spearman & Kendall for each method
            for method_name, scores in all_scores.items():
                rho = spearman_correlation(scores, spreading_scores, network.nodes)
                tau = kendall_correlation(scores, spreading_scores, network.nodes)
                spearman_results[method_name][lam] = rho
                kendall_results[method_name][lam] = tau

            # Print Spearman results for this λ
            rho_row = [f"  ρ(λ={lam:.2f})"] + [
                f"{spearman_results[m][lam]:.4f}" for m in METHOD_NAMES
            ]
            print(f"      " + " | ".join([f"{m}: ρ={spearman_results[m][lam]:.4f}" for m in METHOD_NAMES]))

        results['spearman'] = spearman_results
        results['kendall'] = kendall_results

        # Compute imprecision at a representative λ (use the median)
        representative_lambda = lambda_values[len(lambda_values) // 2]
        print(f"\n  ▶ Computing Imprecision at λ = {representative_lambda}...")

        # Recompute spreading for imprecision
        spreading_for_imp = {}
        pbar = tqdm(network.nodes, desc=f"      SI (imprecision)",
                   leave=False, ncols=80)
        for node in pbar:
            avg_counts = np.zeros(SI_TIME_STEPS + 1)
            for _ in range(num_runs):
                from cwiiif.spreading_model import run_si_single
                counts = run_si_single(network, node, representative_lambda, SI_TIME_STEPS)
                avg_counts += np.array(counts)
            avg_counts /= num_runs
            spreading_for_imp[node] = avg_counts[-1]

        for method_name, scores in all_scores.items():
            imprecision_results[method_name] = imprecision_function(
                scores, spreading_for_imp, network.nodes
            )

        results['imprecision'] = imprecision_results
    else:
        print(f"\n  ⚠ Skipping SI simulations (--skip-si flag)")
        results['spearman'] = None
        results['kendall'] = None
        results['imprecision'] = None

    return results


def _make_json_serializable(obj):
    """Recursively convert numpy types, tuples, and non-string dict keys for JSON."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    return obj


def save_single_dataset(res, tables_dir, plots_dir, checkpoints_dir):
    """
    Save tables, plots, and checkpoints (.pkl and .json) for a single dataset immediately.
    Ensures that no progress is lost if execution is interrupted later.
    """
    ds_name = res['dataset']

    # 1. Save binary checkpoint (.pkl) for lossless restoration
    pkl_path = os.path.join(checkpoints_dir, f"checkpoint_{ds_name}.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(res, f)

    # 2. Save readable JSON checkpoint (.json)
    json_path = os.path.join(checkpoints_dir, f"checkpoint_{ds_name}.json")
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(_make_json_serializable(res), f, indent=2)
    except Exception as e:
        print(f"    [Warning] Could not save JSON checkpoint for {ds_name}: {e}")

    # 3. Save Ranking Table
    ranking = res['cwiiif_ranking']
    rows = []
    for rank, (node, score) in enumerate(ranking, 1):
        rows.append([rank, node, f"{score:.6f}"])

    table_str = tabulate(rows, headers=['Rank', 'Node', 'TI Score'],
                        tablefmt='grid', stralign='right')
    rank_filepath = os.path.join(tables_dir, f'ranking_{ds_name}.txt')
    with open(rank_filepath, 'w', encoding='utf-8') as f:
        f.write(f"CWIIIF Node Ranking — {ds_name}\n")
        f.write(f"N={res['N']}, L={res['L']}, |E|={res['E']}\n\n")
        f.write(table_str)

    # 4. Save Layer Weights Table
    weights = res['layer_weights']
    rows = []
    for lid, details in sorted(weights.items()):
        rows.append([lid, f"{details['NAN']:.4f}", f"{details['NAP']:.4f}",
                    f"{details['NICI']:.4f}", f"{details['W']:.4f}"])
    table_str = tabulate(rows, headers=['Layer', 'NAN', 'NAP', 'NICI', 'W(α)'],
                        tablefmt='grid', stralign='right')
    lw_filepath = os.path.join(tables_dir, f'layer_weights_{ds_name}.txt')
    with open(lw_filepath, 'w', encoding='utf-8') as f:
        f.write(f"Layer Weight Coefficients — {ds_name}\n\n")
        f.write(table_str)

    # 5. Save Plots for this dataset immediately
    if res.get('spearman'):
        plot_spearman_vs_lambda(res['spearman'], ds_name, plots_dir)
    if res.get('imprecision'):
        plot_imprecision(res['imprecision'], ds_name, plots_dir)
    if res.get('connectivity'):
        plot_network_connectivity(res['connectivity'], ds_name, plots_dir)

    print(f"\n  💾 [Immediate Save] Tables, plots, and checkpoints safely saved for '{ds_name}'")


def update_aggregate_results(all_results, tables_dir, plots_dir):
    """Update multi-dataset tables and plots (e.g. monotonicity comparison) incrementally."""
    # Combine in-memory results with any existing checkpoints on disk
    combined_results = {r['dataset']: r for r in all_results}
    if os.path.exists(CHECKPOINTS_DIR):
        for fname in os.listdir(CHECKPOINTS_DIR):
            if fname.startswith("checkpoint_") and fname.endswith(".pkl"):
                ds_name = fname[len("checkpoint_"):-len(".pkl")]
                if ds_name not in combined_results:
                    ckpt = load_dataset_checkpoint(ds_name, CHECKPOINTS_DIR)
                    if ckpt:
                        combined_results[ds_name] = ckpt

    merged = list(combined_results.values())
    if not merged:
        return

    # Monotonicity Comparison Table
    mono_rows = []
    for res in merged:
        row = [res['dataset']]
        for method in METHOD_NAMES:
            row.append(f"{res['monotonicity'].get(method, 0.0):.4f}")
        mono_rows.append(row)

    mono_table = tabulate(mono_rows, headers=['Dataset'] + METHOD_NAMES,
                         tablefmt='grid', stralign='right')
    filepath = os.path.join(tables_dir, 'monotonicity_all.txt')
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("Monotonicity Index Comparison\n\n")
        f.write(mono_table)

    # Monotonicity Comparison Plot
    dataset_names = [r['dataset'] for r in merged]
    mono_data = {r['dataset']: r['monotonicity'] for r in merged}
    plot_monotonicity(mono_data, dataset_names, plots_dir)


def load_dataset_checkpoint(dataset_name, checkpoints_dir):
    """
    Load saved checkpoint for a dataset if it exists.
    Returns result dict or None.
    """
    pkl_path = os.path.join(checkpoints_dir, f"checkpoint_{dataset_name}.pkl")
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                res = pickle.load(f)
            return res
        except Exception as e:
            print(f"  ⚠ Failed to load checkpoint {pkl_path}: {e}")
            return None
    return None


def save_results_table(all_results, output_dir):
    """Save results as formatted text and CSV tables."""
    for res in all_results:
        save_single_dataset(res, TABLES_DIR, PLOTS_DIR, CHECKPOINTS_DIR)
    update_aggregate_results(all_results, TABLES_DIR, PLOTS_DIR)


def generate_all_plots(all_results, output_dir):
    """Generate all visualization plots."""
    for res in all_results:
        ds_name = res['dataset']
        if res.get('spearman'):
            plot_spearman_vs_lambda(res['spearman'], ds_name, output_dir)
        if res.get('imprecision'):
            plot_imprecision(res['imprecision'], ds_name, output_dir)
        if res.get('connectivity'):
            plot_network_connectivity(res['connectivity'], ds_name, output_dir)
    update_aggregate_results(all_results, TABLES_DIR, output_dir)


def print_summary(all_results):
    """Print a comprehensive summary of all experiments."""
    print(f"\n{'═'*70}")
    print(f"  CWIIIF EXPERIMENT RESULTS SUMMARY")
    print(f"{'═'*70}")

    # ─── Monotonicity Summary ───
    print(f"\n  ▸ Monotonicity Index (higher = better discriminative power):")
    mono_rows = []
    for res in all_results:
        row = [res['dataset']]
        monos = res['monotonicity']
        for method in METHOD_NAMES:
            val = monos.get(method, 0.0)
            # Mark the best with an asterisk
            row.append(f"{val:.4f}")
        mono_rows.append(row)
    print(tabulate(mono_rows, headers=['Dataset'] + METHOD_NAMES,
                   tablefmt='simple', stralign='right'))

    # ─── Spearman Summary (average over all λ) ───
    has_spearman = any(r.get('spearman') for r in all_results)
    if has_spearman:
        print(f"\n  ▸ Average Spearman ρ (higher = better correlation with SI):")
        spear_rows = []
        for res in all_results:
            if not res.get('spearman'):
                continue
            row = [res['dataset']]
            for method in METHOD_NAMES:
                vals = list(res['spearman'][method].values())
                avg_rho = np.mean(vals) if vals else 0.0
                row.append(f"{avg_rho:.4f}")
            spear_rows.append(row)
        print(tabulate(spear_rows, headers=['Dataset'] + METHOD_NAMES,
                       tablefmt='simple', stralign='right'))

    # ─── Top-5 Nodes per Dataset ───
    print(f"\n  ▸ Top-5 Most Influential Nodes (CWIIIF):")
    for res in all_results:
        top5 = res['cwiiif_ranking'][:5]
        nodes_str = ", ".join([f"Node {n} ({s:.4f})" for n, s in top5])
        print(f"    {res['dataset']}: {nodes_str}")

    # ─── Timing ───
    print(f"\n  ▸ Execution Times:")
    for res in all_results:
        print(f"    {res['dataset']}: CWIIIF = {res['cwiiif_time']:.2f}s")

    print(f"\n{'═'*70}")
    print(f"  All results saved to: {RESULTS_DIR}")
    print(f"{'═'*70}")


def main():
    parser = argparse.ArgumentParser(description='CWIIIF Algorithm Experiment Runner')
    parser.add_argument('--datasets', nargs='+', default=None,
                       help='Specific dataset names to run (partial match)')
    parser.add_argument('--skip-si', action='store_true',
                       help='Skip SI spreading simulations')
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: fewer SI runs (10 instead of 100)')
    parser.add_argument('--lambdas', nargs='+', type=float, default=None,
                       help='Specific lambda values to test')
    parser.add_argument('--resume', action='store_true',
                       help='Resume experiment by loading existing checkpoints and skipping completed datasets')
    args = parser.parse_args()

    create_output_dirs()

    # Load datasets
    print(f"\n  ▶ Loading datasets from: {DATASETS_DIR}")
    all_networks = load_all_datasets(DATASETS_DIR)

    # Filter datasets if specified
    if args.datasets:
        filtered = []
        for net in all_networks:
            for pattern in args.datasets:
                if pattern.lower() in net.name.lower():
                    filtered.append(net)
                    break
        all_networks = filtered

    if not all_networks:
        print("  ✗ No datasets found! Check your datasets directory.")
        return

    # Print dataset summaries
    print(f"\n  Found {len(all_networks)} datasets:")
    for net in all_networks:
        print(f"    • {net}")

    # Determine SI parameters
    num_runs = SI_NUM_RUNS_QUICK if args.quick else SI_NUM_RUNS
    lambda_values = args.lambdas if args.lambdas else LAMBDA_VALUES

    # ════════════════════════════════════════
    # Run experiments on each dataset
    # ════════════════════════════════════════
    all_results = []
    total_start = time.time()

    for net in all_networks:
        # Check if we should resume from existing checkpoint
        if args.resume:
            cached_res = load_dataset_checkpoint(net.name, CHECKPOINTS_DIR)
            if cached_res is not None:
                print(f"\n{'═'*70}")
                print(f"  Dataset: {net.name} (N={net.num_nodes}, L={net.num_layers}, |E|={net.total_edges()})")
                print(f"  ⏭ Resuming: Loaded existing checkpoint from disk (computation skipped)")
                print(f"{'═'*70}")
                all_results.append(cached_res)
                update_aggregate_results(all_results, TABLES_DIR, PLOTS_DIR)
                continue

        net.summary()
        result = run_experiment_on_dataset(
            net,
            num_runs=num_runs,
            skip_si=args.skip_si,
            lambda_values=lambda_values,
        )
        all_results.append(result)

        # ─── Immediate Save: Save this dataset's results immediately ───
        save_single_dataset(result, TABLES_DIR, PLOTS_DIR, CHECKPOINTS_DIR)
        update_aggregate_results(all_results, TABLES_DIR, PLOTS_DIR)

    total_time = time.time() - total_start

    # ════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════
    print(f"\n  ✓ All tables saved to: {TABLES_DIR}")
    print(f"  ✓ All plots saved to: {PLOTS_DIR}")
    print(f"  ✓ All checkpoints saved to: {CHECKPOINTS_DIR}")

    print_summary(all_results)
    print(f"\n  Total experiment time: {total_time:.1f}s ({total_time/60:.1f}min)")


if __name__ == '__main__':
    main()
