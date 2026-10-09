"""
Visualization Utilities for CWIIIF Experiment Results.

Generates publication-quality plots for:
  1. Spearman ρ vs infection probability λ
  2. Imprecision ε(p) vs fraction p
  3. Monotonicity index comparison bar chart
  4. Network connectivity under sequential node removal
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


# ─── Style Configuration ───
COLORS = {
    'CWIIIF': '#E63946',     # Bold red
    'sumDC': '#457B9D',      # Steel blue
    'sumBC': '#2A9D8F',      # Teal
    'sumPR': '#E9C46A',      # Golden
    'aggKS': '#264653',      # Dark slate
}

MARKERS = {
    'CWIIIF': 'o',
    'sumDC': 's',
    'sumBC': '^',
    'sumPR': 'D',
    'aggKS': 'v',
}

METHOD_ORDER = ['CWIIIF', 'sumDC', 'sumBC', 'sumPR', 'aggKS']


def setup_plot_style():
    """Configure a clean, publication-style plot appearance."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_spearman_vs_lambda(spearman_results, dataset_name, save_dir):
    """
    Plot Spearman ρ vs infection probability λ for all methods.

    Args:
        spearman_results (dict): {method_name: {lambda: rho}}
        dataset_name (str): Name of the dataset.
        save_dir (str): Directory to save the plot.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHOD_ORDER:
        if method not in spearman_results:
            continue
        data = spearman_results[method]
        lambdas = sorted(data.keys())
        rhos = [data[l] for l in lambdas]

        ax.plot(lambdas, rhos,
                marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, 'gray'),
                label=method,
                linewidth=2,
                markersize=6,
                alpha=0.9)

    ax.set_xlabel('Infection Probability (λ)')
    ax.set_ylabel('Spearman Correlation Coefficient (ρ)')
    ax.set_title(f'Spearman ρ vs λ — {dataset_name}')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(-0.1, 1.05)

    plt.tight_layout()
    filepath = os.path.join(save_dir, f'spearman_{dataset_name}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    return filepath


def plot_imprecision(imprecision_results, dataset_name, save_dir):
    """
    Plot imprecision ε(p) vs fraction p for all methods.

    Args:
        imprecision_results (dict): {method_name: {p: epsilon}}
        dataset_name (str): Name of the dataset.
        save_dir (str): Directory to save the plot.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHOD_ORDER:
        if method not in imprecision_results:
            continue
        data = imprecision_results[method]
        ps = sorted(data.keys())
        epsilons = [data[p] for p in ps]

        ax.plot(ps, epsilons,
                marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, 'gray'),
                label=method,
                linewidth=2,
                markersize=6,
                alpha=0.9)

    ax.set_xlabel('Fraction of Nodes (p)')
    ax.set_ylabel('Imprecision ε(p)')
    ax.set_title(f'Imprecision Function — {dataset_name}')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(-0.02, 1.0)

    plt.tight_layout()
    filepath = os.path.join(save_dir, f'imprecision_{dataset_name}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    return filepath


def plot_monotonicity(mono_results, dataset_names, save_dir):
    """
    Plot monotonicity index comparison as a grouped bar chart.

    Args:
        mono_results (dict): {dataset_name: {method_name: M_value}}
        dataset_names (list): Ordered list of dataset names.
        save_dir (str): Directory to save the plot.
    """
    setup_plot_style()
    n_datasets = len(dataset_names)
    n_methods = len(METHOD_ORDER)
    bar_width = 0.15
    x = np.arange(n_datasets)

    fig, ax = plt.subplots(figsize=(max(10, n_datasets * 2), 5))

    for i, method in enumerate(METHOD_ORDER):
        values = []
        for ds in dataset_names:
            values.append(mono_results.get(ds, {}).get(method, 0.0))
        offset = (i - n_methods / 2 + 0.5) * bar_width
        ax.bar(x + offset, values, bar_width,
               label=method,
               color=COLORS.get(method, 'gray'),
               alpha=0.85,
               edgecolor='white',
               linewidth=0.5)

    ax.set_xlabel('Dataset')
    ax.set_ylabel('Monotonicity Index M(R)')
    ax.set_title('Monotonicity Index Comparison Across Datasets')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=30, ha='right', fontsize=9)
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    filepath = os.path.join(save_dir, 'monotonicity_comparison.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    return filepath


def plot_network_connectivity(connectivity_results, dataset_name, save_dir):
    """
    Plot network connectivity (largest component size) after node removal.

    Args:
        connectivity_results (dict): {method_name: {p: Q}}
        dataset_name (str): Name of the dataset.
        save_dir (str): Directory to save the plot.
    """
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for method in METHOD_ORDER:
        if method not in connectivity_results:
            continue
        data = connectivity_results[method]
        ps = sorted(data.keys())
        qs = [data[p] for p in ps]

        ax.plot(ps, qs,
                marker=MARKERS.get(method, 'o'),
                color=COLORS.get(method, 'gray'),
                label=method,
                linewidth=2,
                markersize=6,
                alpha=0.9)

    ax.set_xlabel('Fraction of Nodes Removed (p)')
    ax.set_ylabel('Largest Connected Component Size')
    ax.set_title(f'Network Connectivity Under Attack — {dataset_name}')
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()
    filepath = os.path.join(save_dir, f'connectivity_{dataset_name}.png')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    return filepath
