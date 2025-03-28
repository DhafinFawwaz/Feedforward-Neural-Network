import plotly.graph_objects as go
import numpy as np
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
import plotly.graph_objects as go

import sys

prev_line_length = 0
def print_progress(s):
    global prev_line_length
    padded_s = s + ' ' * max(0, prev_line_length - len(s))
    print(padded_s, end='\r')
    sys.stdout.flush()
    prev_line_length = len(s)

class NeuralNetworkVisualizerPlotly:
    def __init__(self, layers, weights, gradients, biases, loss_history):
        self.layers = layers
        self.weights = weights
        self.gradients = gradients
        self.biases = [bias.ravel() for bias in biases]
        self.loss_history = loss_history
        self.colors = sns.color_palette("husl", len(layers))
        self.layer_names = self._generate_layer_names()
    
    def _generate_layer_names(self):
        layer_names = ['Input Layer']
        for i in range(1, len(self.layers) - 1):
            layer_names.append(f'Hidden Layer {i}')
        layer_names.append('Output Layer')
        return layer_names
    
    def plot_network(self):
        fig = go.Figure()
        
        x_positions = np.linspace(0, 1, len(self.layers))
        y_positions = [np.linspace(-1, 1, num_nodes) for num_nodes in self.layers]
        
        node_positions = {}

        for layer_idx, (x, y_vals) in enumerate(zip(x_positions, y_positions)):
            for node_idx, y in enumerate(y_vals):
                node_positions[(layer_idx, node_idx)] = (x, y)

        for layer_idx in range(len(self.layers) - 1):
            for i in range(self.layers[layer_idx]):
                print_progress(f"Layer {layer_idx+1} Node {i+1}/{self.layers[layer_idx]}")
                for j in range(self.layers[layer_idx + 1]):
                    x0, y0 = node_positions[(layer_idx, i)]
                    x1, y1 = node_positions[(layer_idx + 1, j)]
                    
                    weight = self.weights[layer_idx][i, j]
                    gradient = self.gradients[layer_idx][i, j]

                    num_points = 20
                    edge_x = np.linspace(x0, x1, num_points)
                    edge_y = np.linspace(y0, y1, num_points)
                    
                    hover_text = (f"Layer {layer_idx} Node {i} → Layer {layer_idx+1} Node {j}<br>"
                                  f"Weight: {weight:.4f}<br>"
                                  f"Gradient: {gradient:.4f}")
                    
                    fig.add_trace(go.Scatter(
                        x=edge_x, 
                        y=edge_y,
                        mode='lines',
                        line=dict(color='black', width=2),
                        hoverinfo='text',
                        hovertemplate=hover_text,
                        opacity=0.7,
                        showlegend=False
                    ))

            for j in range(self.layers[layer_idx + 1]):
                x0 = x_positions[layer_idx]
                y0 = -1.5 
                x1, y1 = node_positions[(layer_idx + 1, j)]
                
                bias = self.biases[layer_idx][j]

                num_points = 20
                edge_x = np.linspace(x0, x1, num_points)
                edge_y = np.linspace(y0, y1, num_points)
                
                hover_text = (f"Bias → Layer {layer_idx+1} Node {j}<br>"
                              f"Bias Value: {bias:.4f}")
                
                fig.add_trace(go.Scatter(
                    x=edge_x, 
                    y=edge_y,
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    hoverinfo='text',
                    hovertemplate=hover_text,
                    opacity=0.7,
                    showlegend=False
                ))
        print()
        for layer_idx, (x, y_vals, layer_name) in enumerate(zip(x_positions, y_positions, self.layer_names)):
            print_progress(f"Finishing Layer {layer_idx+1}: {layer_name}")
            for node_idx, y in enumerate(y_vals):
                fig.add_trace(go.Scatter(
                    x=[x], 
                    y=[y],
                    mode='markers',
                    marker=dict(
                        size=20, 
                        color=f'rgba({int(self.colors[layer_idx][0]*255)}, {int(self.colors[layer_idx][1]*255)}, {int(self.colors[layer_idx][2]*255)}, 1)',
                        opacity=1
                    ),
                    text=[f'Node {node_idx}'],
                    textposition='bottom center',
                    hoverinfo='text',
                    showlegend=False
                ))

            if layer_idx < len(self.layers) - 1:
                fig.add_trace(go.Scatter(
                    x=[x], 
                    y=[-1.5],
                    mode='markers',
                    marker=dict(
                        size=15, 
                        color='gray',
                        opacity=1
                    ),
                    text=[f'Bias {layer_idx}'],
                    textposition='bottom center',
                    hoverinfo='text',
                    showlegend=False
                ))
            
            fig.add_annotation(x=x, y=1.2, text=layer_name, showarrow=False, font=dict(size=12, color='black'))
        fig.update_layout(
            title='Neural Network Structure with Bias Nodes',
            xaxis=dict(visible=False, range=[-0.1, 1.1]),
            yaxis=dict(visible=False, range=[-2, 1.2]),
            showlegend=False,
            height=800,
            width=1000,
            hovermode='closest',
            hoverdistance=100  
        )
        print()
        print("Plot should be displayed in your browser now!")
        fig.show()
    

    def plot_network_parallel(self):
        fig = go.Figure()
        x_positions = np.linspace(0, 1, len(self.layers))
        y_positions = [np.linspace(-1, 1, num_nodes) for num_nodes in self.layers]
        node_positions = {
            (layer_idx, node_idx): (x, y)
            for layer_idx, (x, y_vals) in enumerate(zip(x_positions, y_positions))
            for node_idx, y in enumerate(y_vals)
        }

        edge_traces = []
        bias_traces = []

        def generate_edge_trace(layer_idx, i, j):
            x0, y0 = node_positions[(layer_idx, i)]
            x1, y1 = node_positions[(layer_idx + 1, j)]
            weight = self.weights[layer_idx][i, j]
            gradient = self.gradients[layer_idx][i, j]
            edge_x = np.linspace(x0, x1, 20)
            edge_y = np.linspace(y0, y1, 20)
            hover_text = (f"Layer {layer_idx} Node {i} → Layer {layer_idx+1} Node {j}<br>"
                        f"Weight: {weight:.4f}<br>"
                        f"Gradient: {gradient:.4f}")
            return go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(color='black', width=2),
                hoverinfo='text',
                hovertemplate=hover_text,
                opacity=0.7,
                showlegend=False
            )

        def generate_bias_trace(layer_idx, j):
            x0 = x_positions[layer_idx]
            y0 = -1.5
            x1, y1 = node_positions[(layer_idx + 1, j)]
            bias = self.biases[layer_idx][j]
            edge_x = np.linspace(x0, x1, 20)
            edge_y = np.linspace(y0, y1, 20)
            hover_text = (f"Bias → Layer {layer_idx+1} Node {j}<br>"
                        f"Bias Value: {bias:.4f}")
            return go.Scatter(
                x=edge_x, y=edge_y,
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                hoverinfo='text',
                hovertemplate=hover_text,
                opacity=0.7,
                showlegend=False
            )

        with ThreadPoolExecutor() as executor:
            futures = []
            for layer_idx in range(len(self.layers) - 1):
                for i in range(self.layers[layer_idx]):
                    for j in range(self.layers[layer_idx + 1]):
                        futures.append(executor.submit(generate_edge_trace, layer_idx, i, j))
                for j in range(self.layers[layer_idx + 1]):
                    futures.append(executor.submit(generate_bias_trace, layer_idx, j))

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating connections"):
                trace = future.result()
                fig.add_trace(trace)

        for layer_idx, (x, y_vals, layer_name) in enumerate(zip(x_positions, y_positions, self.layer_names)):
            for node_idx, y in enumerate(y_vals):
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode='markers',
                    marker=dict(
                        size=20, 
                        color=f'rgba({int(self.colors[layer_idx][0]*255)}, {int(self.colors[layer_idx][1]*255)}, {int(self.colors[layer_idx][2]*255)}, 1)',
                        opacity=1
                    ),
                    text=[f'Node {node_idx}'],
                    textposition='bottom center',
                    hoverinfo='text',
                    showlegend=False
                ))
            if layer_idx < len(self.layers) - 1:
                fig.add_trace(go.Scatter(
                    x=[x], y=[-1.5],
                    mode='markers',
                    marker=dict(size=15, color='gray', opacity=1),
                    text=[f'Bias {layer_idx}'],
                    textposition='bottom center',
                    hoverinfo='text',
                    showlegend=False
                ))
            fig.add_annotation(x=x, y=1.2, text=layer_name, showarrow=False, font=dict(size=12, color='black'))

        fig.update_layout(
            title='Neural Network Structure with Bias Nodes',
            xaxis=dict(visible=False, range=[-0.1, 1.1]),
            yaxis=dict(visible=False, range=[-2, 1.2]),
            showlegend=False,
            height=800,
            width=1000,
            hovermode='closest',
            hoverdistance=100
        )
        fig.show()


    def plot_weight_distribution(self, layers_to_plot, plot_size=0.5):
        fig = go.Figure()
        
        for layer_idx in layers_to_plot:
            layer_data = self.weights[layer_idx].flatten()
            layer_name = self.layer_names[layer_idx]
            
            fig.add_trace(go.Histogram(
                x=layer_data,
                name=layer_name,
                opacity=0.7,
                xbins=dict(start=min(layer_data), end=max(layer_data), size=plot_size),
            ))
        
        fig.update_layout(
            title='Weight Distribution', 
            barmode='overlay',
            xaxis_title='Value',
            yaxis_title='Frequency'
        )
        fig.show()

    def plot_gradient_distribution(self, layers_to_plot, plot_size=0.5):
        fig = go.Figure()
        
        for layer_idx in layers_to_plot:
            layer_data = self.gradients[layer_idx].flatten()
            layer_name = self.layer_names[layer_idx]
            
            fig.add_trace(go.Histogram(
                x=layer_data,
                name=layer_name,
                opacity=0.7,
                xbins=dict(start=min(layer_data), end=max(layer_data), size=plot_size)
            ))
        
        fig.update_layout(
            title='Gradient Distribution', 
            barmode='overlay',
            xaxis_title='Value',
            yaxis_title='Frequency'
        )
        fig.show()
    
    def plot_loss(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=self.loss_history, mode='lines', name='Loss'))
        
        fig.update_layout(
            title='Loss Over Time',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            template='plotly_white'
        )
        
        fig.show()


# # # test
# layer_sizes = [35, 20, 5, 2]
# weights = [np.random.randn(35, 20), np.random.randn(20, 5), np.random.randn(5, 2)]
# gradients = [np.random.randn(35, 20), np.random.randn(20, 5), np.random.randn(5, 2)]
# biases = [np.random.randn(20), np.random.randn(5), np.random.randn(2)]
# loss_history = np.exp(-0.1 * np.arange(100)) + np.random.normal(0, 0.02, 100) 

# visualizer = NeuralNetworkVisualizerPlotly(layer_sizes, weights, gradients, biases, loss_history)
# visualizer.plot_network()
# visualizer.plot_weight_distribution([1, 2])
# visualizer.plot_gradient_distribution([2])
# visualizer.plot_loss()