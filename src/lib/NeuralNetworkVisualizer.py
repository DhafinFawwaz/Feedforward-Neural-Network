import pandas as pd
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from scipy import stats

class NeuralNetworkVisualizer:
    def __init__(self, layers, weights=None, gradients=None, biases=None):
        self.layers = layers

        if weights is not None:
            self.weights = [w[0] if isinstance(w, list) and isinstance(w[0], np.ndarray) else w for w in weights]
        else:
            self.weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers) - 1)]

        if gradients is not None:
            self.gradients = [g[0] if isinstance(g, list) and isinstance(g[0], np.ndarray) else g for g in gradients]
        else:
            self.gradients = [np.random.randn(*w.shape) * 0.1 for w in self.weights]
            
        if biases is not None:
            self.biases = [b[0] if isinstance(b, list) and isinstance(b[0], np.ndarray) else b for b in biases]
        else:
            self.biases = [np.random.randn(layers[i+1]) for i in range(len(layers) - 1)]

    def plot_network(self, include_bias=True):
        G = nx.DiGraph()
        positions = {}
        node_count = 0
        layer_spacing = 3
        node_spacing = 1
        colors = []
        node_labels = {}
        bias_nodes = []

        for layer_idx, num_nodes in enumerate(self.layers):
            for node_idx in range(num_nodes):
                node_id = node_count
                positions[node_id] = (layer_idx * layer_spacing, -node_idx * node_spacing)
                G.add_node(node_id)
                node_labels[node_id] = f"{layer_idx},{node_idx}"
                if layer_idx == 0:
                    colors.append("red")  # Input layer
                elif layer_idx == len(self.layers) - 1:
                    colors.append("green")  # Output layer
                else:
                    colors.append("blue")  # Hidden layer
                node_count += 1
        
        if include_bias:
            for layer_idx in range(len(self.layers) - 1):
                num_nodes_in_layer = self.layers[layer_idx]
                bias_node_id = node_count
                positions[bias_node_id] = (layer_idx * layer_spacing, -(num_nodes_in_layer) * node_spacing - 1)
                G.add_node(bias_node_id)
                node_labels[bias_node_id] = f"B{layer_idx}"
                colors.append("yellow")  # Bias
                bias_nodes.append((layer_idx, bias_node_id))
                node_count += 1

        edge_weights = {}
        node_count = 0
        
        for layer_idx in range(len(self.layers) - 1):
            layer_size = self.layers[layer_idx]
            next_layer_size = self.layers[layer_idx + 1]
            next_layer_start = node_count + layer_size
            
            for src in range(node_count, node_count + layer_size):
                for dst_idx, dst in enumerate(range(next_layer_start, next_layer_start + next_layer_size)):
                    weight = self.weights[layer_idx][src - node_count, dst_idx]
                    G.add_edge(src, dst, weight=weight)
                    edge_weights[(src, dst)] = f'{weight:.2f}'
            
            node_count += layer_size

        if include_bias:
            for layer_idx, bias_node_id in bias_nodes:
                next_layer_idx = layer_idx + 1
                if next_layer_idx < len(self.layers):
                    next_layer_start = sum(self.layers[:next_layer_idx])
                    next_layer_end = next_layer_start + self.layers[next_layer_idx]

                    for node_idx, node_id in enumerate(range(next_layer_start, next_layer_end)):
                        bias_weight = self.biases[layer_idx][node_idx]
                        G.add_edge(bias_node_id, node_id, weight=bias_weight)
                        print(bias_weight)
                        edge_weights[(bias_node_id, node_id)] = f'{bias_weight:.2f}'

        plt.figure(figsize=(12, 10))
        
        nx.draw(G, pos=positions, with_labels=False, node_size=500, node_color=colors, edge_color="gray")

        bias_node_labels = {node_id: label for node_id, label in node_labels.items() if node_id in [node_id for _, node_id in bias_nodes]}
        nx.draw_networkx_labels(G, positions, labels=bias_node_labels, font_size=10, font_color='black')

        edge_midpoints = {}
        edges_list = list(G.edges())

        for u, v in edges_list:
            u_x, u_y = positions[u]
            v_x, v_y = positions[v]

            mid_x = (u_x + v_x) / 2
            mid_y = (u_y + v_y) / 2
            
            edge_midpoints[(u, v)] = (mid_x, mid_y)
        
        midpoint_groups = {}
        for edge, midpoint in edge_midpoints.items():
            grid_x = round(midpoint[0] * 10) / 10
            grid_y = round(midpoint[1] * 10) / 10
            grid_key = (grid_x, grid_y)
            
            if grid_key not in midpoint_groups:
                midpoint_groups[grid_key] = []
            midpoint_groups[grid_key].append(edge)

        edge_labels_pos = {}
        for grid_key, colliding_edges in midpoint_groups.items():
            if len(colliding_edges) == 1:
                edge = colliding_edges[0]
                edge_labels_pos[edge] = edge_midpoints[edge]
            else:
                for idx, edge in enumerate(colliding_edges):
                    u, v = edge
                    u_x, u_y = positions[u]
                    v_x, v_y = positions[v]
                    
                    mid_x, mid_y = edge_midpoints[edge]

                    angle = np.arctan2(v_y - u_y, v_x - u_x)
                    perp_angle = angle + np.pi/2
                    
                    offset_scale = 0.15 * (1 + len(colliding_edges)/10)
                    
                    position = idx - (len(colliding_edges) - 1) / 2
    
                    offset_x = offset_scale * position * np.cos(perp_angle)
                    offset_y = offset_scale * position * np.sin(perp_angle)
                    
                    edge_labels_pos[edge] = (mid_x + offset_x, mid_y + offset_y)
        
        for (u, v), label in edge_weights.items():
            x, y = edge_labels_pos[(u, v)]
            plt.text(x, y, label, fontsize=8, ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, boxstyle="round,pad=0.2"))
        
        for layer_idx in range(len(self.layers)):
            layer_name = "Input Layer" if layer_idx == 0 else "Output Layer" if layer_idx == len(self.layers) - 1 else f"Hidden Layer {layer_idx}"
            plt.text(layer_idx * layer_spacing, 0.5, layer_name, fontsize=12, ha='center', bbox=dict(facecolor='white', alpha=0.5))
            plt.text(layer_idx * layer_spacing, 0.25, f"Layer {layer_idx}", fontsize=10, ha='center', bbox=dict(facecolor='white', alpha=0.5))
        
        print("STRUKTUR JARINGAN")
        plt.axis('off')
        plt.show()

    def plot_weight_distribution(self, layers_to_plot, plot_type='histogram'):
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'histogram':
            for layer_idx in layers_to_plot:
                plt.hist(self.weights[layer_idx].flatten(), bins=20, alpha=0.5, label=f'Layer {layer_idx}')
                
        elif plot_type == 'line':
            for layer_idx in layers_to_plot:
                weights_flat = self.weights[layer_idx].flatten()
                
                density = stats.gaussian_kde(weights_flat)
                
                x_min, x_max = min(weights_flat), max(weights_flat)
                x = np.linspace(x_min, x_max, 200)

                plt.plot(x, density(x), label=f'Layer {layer_idx}')

                plt.plot(weights_flat, np.zeros_like(weights_flat), '|', 
                        color=plt.gca().lines[-1].get_color(), alpha=0.3, markersize=5)
        
        plt.title("Weight Distribution")
        plt.xlabel("Weight Value")
        plt.ylabel("Density" if plot_type == 'line' else "Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_gradient_distribution(self, layers_to_plot, plot_type='histogram'):
        plt.figure(figsize=(10, 6))
        
        if plot_type == 'histogram':
            for layer_idx in layers_to_plot:
                plt.hist(self.gradients[layer_idx].flatten(), bins=20, alpha=0.5, label=f'Layer {layer_idx}')
                
        elif plot_type == 'line':
            for layer_idx in layers_to_plot:
                gradients_flat = self.gradients[layer_idx].flatten()

                density = stats.gaussian_kde(gradients_flat)

                x_min, x_max = min(gradients_flat), max(gradients_flat)
                x = np.linspace(x_min, x_max, 200)

                plt.plot(x, density(x), label=f'Layer {layer_idx}')

                plt.plot(gradients_flat, np.zeros_like(gradients_flat), '|', 
                            color=plt.gca().lines[-1].get_color(), alpha=0.3, markersize=5)
        
        plt.title("Gradient Distribution")
        plt.xlabel("Gradient Value")
        plt.ylabel("Density" if plot_type == 'line' else "Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    layers = [3, 4, 2]

    custom_weights = [
        np.random.normal(0, 0.5, (3, 4)),
        np.random.normal(0, 0.5, (4, 2))
    ]

    custom_gradients = [
        np.random.normal(0, 0.1, (3, 4)),
        np.random.normal(0, 0.1, (4, 2))
    ]

    custom_biases = [
        np.random.normal(0, 0.3),  # Biases for the first hidden layer
        np.random.normal(0, 0.3)   # Biases for the output layer
    ]

    visualizer = NeuralNetworkVisualizer(layers, 
                                    weights=custom_weights,
                                    gradients=custom_gradients)

    visualizer.plot_network()

    visualizer.plot_weight_distribution([0, 1], plot_type='line')

    visualizer.plot_gradient_distribution([0, 1], plot_type='line')

    visualizer.plot_weight_distribution([0, 1], plot_type='histogram')

    visualizer.plot_weight_distribution([0])  # Plot only first layer weights
    visualizer.plot_gradient_distribution([1])  # Plot only output layer gradients