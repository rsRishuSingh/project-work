"""
Dataset Loader for Multiplex Network Edge Files.

Parses .edges files in the format: layerID sourceNode targetNode weight
Builds one NetworkX undirected graph per layer with a unified node set.
"""

import os
import networkx as nx
import numpy as np
from collections import defaultdict


class MultiplexNetwork:
    """
    Represents a multiplex (multilayer) network where the same set of nodes
    exists across all layers, but edges differ per layer.

    Attributes:
        name (str): Name of the dataset.
        layers (dict): {layer_id: nx.Graph} mapping.
        nodes (set): Unified set of all node IDs across all layers.
        num_layers (int): Number of layers L.
        num_nodes (int): Total number of unique nodes N.
    """

    def __init__(self, name, layers, nodes):
        self.name = name
        self.layers = layers  # {layer_id: nx.Graph}
        self.nodes = sorted(nodes)
        self.num_layers = len(layers)
        self.num_nodes = len(nodes)
        self.layer_ids = sorted(layers.keys())

        # Ensure all layers have the same node set
        for lid in self.layer_ids:
            for node in self.nodes:
                if node not in self.layers[lid]:
                    self.layers[lid].add_node(node)

    def get_layer(self, layer_id):
        """Return the graph for a specific layer."""
        return self.layers[layer_id]

    def get_edge_set(self, layer_id):
        """Return the set of edges (as frozensets for undirected comparison) for a layer."""
        return set(frozenset((u, v)) for u, v in self.layers[layer_id].edges())

    def total_edges(self):
        """Return the total number of edges across all layers."""
        return sum(G.number_of_edges() for G in self.layers.values())

    def summary(self):
        """Print a summary of the network."""
        print(f"{'='*60}")
        print(f"  Dataset: {self.name}")
        print(f"  Nodes (N): {self.num_nodes}")
        print(f"  Layers (L): {self.num_layers}")
        print(f"  Total Edges: {self.total_edges()}")
        print(f"  {'─'*56}")
        for lid in self.layer_ids:
            G = self.layers[lid]
            n_edges = G.number_of_edges()
            n_isolated = sum(1 for n in G.nodes() if G.degree(n) == 0)
            avg_deg = np.mean([G.degree(n) for n in G.nodes()]) if self.num_nodes > 0 else 0
            print(f"  Layer {lid}: {n_edges} edges, {n_isolated} isolated nodes, avg degree = {avg_deg:.2f}")
        print(f"{'='*60}")

    def __repr__(self):
        return f"MultiplexNetwork(name='{self.name}', N={self.num_nodes}, L={self.num_layers}, |E|={self.total_edges()})"


def load_edges_file(filepath):
    """
    Load a multiplex .edges file.

    File format: each line has 4 space-separated values:
        layerID sourceNode targetNode weight

    Args:
        filepath (str): Path to the .edges file.

    Returns:
        MultiplexNetwork: The parsed multiplex network.
    """
    name = os.path.splitext(os.path.basename(filepath))[0]
    # Remove the "-Copy1" suffix for cleaner naming
    name = name.replace("-Copy1", "").replace("_multiplex", "").replace("_2", "")

    layer_edges = defaultdict(list)
    all_nodes = set()

    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            layer_id = int(parts[0])
            source = int(parts[1])
            target = int(parts[2])
            # weight is in parts[3] if present, but we treat all as unweighted

            # Skip self-loops
            if source == target:
                continue

            layer_edges[layer_id].append((source, target))
            all_nodes.add(source)
            all_nodes.add(target)

    # Build NetworkX graphs for each layer
    layers = {}
    for layer_id in sorted(layer_edges.keys()):
        G = nx.Graph()
        G.add_edges_from(layer_edges[layer_id])
        layers[layer_id] = G

    return MultiplexNetwork(name, layers, all_nodes)


def load_all_datasets(datasets_dir):
    """
    Load all .edges files from a directory.

    Args:
        datasets_dir (str): Path to the datasets directory.

    Returns:
        list[MultiplexNetwork]: List of loaded networks.
    """
    networks = []
    for filename in sorted(os.listdir(datasets_dir)):
        if filename.endswith('.edges'):
            filepath = os.path.join(datasets_dir, filename)
            net = load_edges_file(filepath)
            networks.append(net)

    return networks
