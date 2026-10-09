"""
SI (Susceptible-Infected) Spreading Model for Multilayer Networks.

Simulates information/disease spread across a multiplex network.
Each infected node attempts to infect its susceptible neighbors in
every layer with probability λ at each time step. Once infected in
any layer, a node is globally infected across all layers.
"""

import random
import numpy as np
from collections import defaultdict


def run_si_single(network, seed_node, infection_prob, time_steps):
    """
    Run a single SI simulation starting from a seed node.

    Propagation operates independently in all layers, but infection
    state is shared globally (once infected in any layer, the node
    is infected everywhere).

    Args:
        network: MultiplexNetwork object.
        seed_node: The initial infected node.
        infection_prob (float): Probability λ of infection per edge per step.
        time_steps (int): Number of simulation steps T.

    Returns:
        list: Number of infected nodes at each time step [t=0, t=1, ..., t=T].
    """
    infected = {seed_node}
    infection_counts = [1]  # At t=0, only the seed is infected

    for t in range(time_steps):
        newly_infected = set()

        for lid in network.layer_ids:
            G = network.get_layer(lid)

            # Each currently infected node tries to infect its neighbors
            for node in list(infected):
                if node not in G:
                    continue
                for neighbor in G.neighbors(node):
                    if neighbor not in infected and neighbor not in newly_infected:
                        if random.random() < infection_prob:
                            newly_infected.add(neighbor)

        infected.update(newly_infected)
        infection_counts.append(len(infected))

    return infection_counts


def run_si_simulation(network, seed_node, infection_prob, time_steps=10, num_runs=100):
    """
    Run multiple SI simulations and average the results.

    Args:
        network: MultiplexNetwork object.
        seed_node: The initial infected node.
        infection_prob (float): Probability λ.
        time_steps (int): Number of steps (default 10).
        num_runs (int): Number of independent runs to average (default 100).

    Returns:
        np.ndarray: Average infection count at each time step, shape (T+1,).
    """
    all_counts = []

    for _ in range(num_runs):
        counts = run_si_single(network, seed_node, infection_prob, time_steps)
        all_counts.append(counts)

    return np.mean(all_counts, axis=0)


def compute_spreading_influence(network, infection_prob, time_steps=10, num_runs=100,
                                 progress_callback=None):
    """
    Compute the spreading influence of every node in the network.

    For each node, runs the SI model num_runs times and records
    the average number of infected nodes at the final time step.

    Args:
        network: MultiplexNetwork object.
        infection_prob (float): Probability λ.
        time_steps (int): Number of steps (default 10).
        num_runs (int): Number of independent runs (default 100).
        progress_callback: Optional callable for progress updates.

    Returns:
        dict: {node_id: average_infected_count_at_final_step}
    """
    influence = {}

    for idx, node in enumerate(network.nodes):
        avg_counts = run_si_simulation(network, node, infection_prob,
                                       time_steps, num_runs)
        influence[node] = avg_counts[-1]  # Final step infection count

        if progress_callback:
            progress_callback(idx + 1, len(network.nodes))

    return influence


def rank_by_spreading(influence):
    """
    Rank nodes by their spreading influence (descending).

    Args:
        influence (dict): {node_id: spreading_score}

    Returns:
        list: Sorted list of (node_id, score).
        dict: {node_id: rank}
    """
    sorted_nodes = sorted(influence.items(), key=lambda x: x[1], reverse=True)
    ranking = {node: rank for rank, (node, _) in enumerate(sorted_nodes, 1)}
    return sorted_nodes, ranking
