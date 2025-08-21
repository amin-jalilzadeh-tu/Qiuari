# utils/visualization.py
"""
Visualization utilities for graph structures and analysis results
Includes interactive plots, graph layouts, and result dashboards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
from torch_geometric.data import Data
import folium
from folium import plugins
import logging

logger = logging.getLogger(__name__)

class GraphVisualizer:
    """Visualize graph structures and embeddings"""
    
    def __init__(self, style: str = 'dark'):
        """
        Initialize graph visualizer
        
        Args:
            style: Visual style ('dark' or 'light')
        """
        self.style = style
        self._setup_style()
        
        # Color schemes
        self.cluster_colors = px.colors.qualitative.Set3
        self.task_colors = {
            'clustering': '#1f77b4',
            'solar': '#ff7f0e',
            'retrofit': '#2ca02c',
            'electrification': '#d62728',
            'battery': '#9467bd',
            'p2p': '#8c564b',
            'congestion': '#e377c2',
            'thermal': '#7f7f7f'
        }
        
        logger.info(f"Initialized GraphVisualizer with {style} style")
    
    def _setup_style(self):
        """Setup visualization style"""
        if self.style == 'dark':
            plt.style.use('dark_background')
            self.bg_color = '#1e1e1e'
            self.text_color = '#ffffff'
            self.grid_color = '#404040'
        else:
            plt.style.use('seaborn-v0_8-whitegrid')
            self.bg_color = '#ffffff'
            self.text_color = '#000000'
            self.grid_color = '#e0e0e0'
    
    def visualize_graph_structure(self, 
                                  graph_data: Union[Data, Dict],
                                  node_colors: Optional[np.ndarray] = None,
                                  node_sizes: Optional[np.ndarray] = None,
                                  layout: str = 'spring',
                                  title: str = 'Energy Network Graph',
                                  interactive: bool = True) -> Union[go.Figure, plt.Figure]:
        """
        Visualize graph structure
        
        Args:
            graph_data: Graph data
            node_colors: Node coloring values
            node_sizes: Node sizing values
            layout: Layout algorithm
            title: Plot title
            interactive: Use interactive plot
            
        Returns:
            Figure object
        """
        # Convert to NetworkX
        if isinstance(graph_data, Data):
            G = self._pyg_to_networkx(graph_data)
        else:
            G = self._dict_to_networkx(graph_data)
        
        # Calculate layout
        pos = self._calculate_layout(G, layout)
        
        if interactive:
            return self._create_interactive_graph(G, pos, node_colors, node_sizes, title)
        else:
            return self._create_static_graph(G, pos, node_colors, node_sizes, title)
    
    def _pyg_to_networkx(self, data: Data) -> nx.Graph:
        """Convert PyTorch Geometric data to NetworkX"""
        G = nx.Graph()
        
        # Add nodes
        for i in range(data.num_nodes):
            G.add_node(i)
        
        # Add edges
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])
        
        return G
    
    def _dict_to_networkx(self, data: Dict) -> nx.Graph:
        """Convert dictionary data to NetworkX"""
        G = nx.Graph()
        
        # Add nodes
        if 'nodes' in data:
            for node_type, nodes in data['nodes'].items():
                for idx, node in nodes.iterrows():
                    G.add_node(f"{node_type}_{idx}", type=node_type, **node.to_dict())
        
        # Add edges
        if 'edges' in data:
            for edge_type, edges in data['edges'].items():
                for _, edge in edges.iterrows():
                    G.add_edge(edge['source'], edge['target'], type=edge_type)
        
        return G
    
    def _calculate_layout(self, G: nx.Graph, layout: str) -> Dict:
        """Calculate graph layout"""
        if layout == 'spring':
            pos = nx.spring_layout(G, k=1/np.sqrt(len(G)), iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'spectral':
            pos = nx.spectral_layout(G)
        else:
            pos = nx.random_layout(G)
        
        return pos
    
    def _create_interactive_graph(self, G: nx.Graph, pos: Dict,
                                 node_colors: Optional[np.ndarray],
                                 node_sizes: Optional[np.ndarray],
                                 title: str) -> go.Figure:
        """Create interactive Plotly graph"""
        # Edge trace
        edge_trace = go.Scatter(
            x=[], y=[],
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += (x0, x1, None)
            edge_trace['y'] += (y0, y1, None)
        
        # Node trace
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                size=10 if node_sizes is None else node_sizes,
                color=list(range(len(G))) if node_colors is None else node_colors,
                colorbar=dict(
                    thickness=15,
                    title='Node Value',
                    xanchor='left',
                    titleside='right'
                )
            )
        )
        
        # Node text
        node_text = []
        for node in G.nodes():
            info = f"Node: {node}"
            if G.nodes[node]:
                for key, value in G.nodes[node].items():
                    if isinstance(value, (int, float)):
                        info += f"<br>{key}: {value:.2f}"
                    else:
                        info += f"<br>{key}: {value}"
            node_text.append(info)
        
        node_trace.text = node_text
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title=title,
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           paper_bgcolor=self.bg_color,
                           plot_bgcolor=self.bg_color,
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                       ))
        
        return fig
    
    def _create_static_graph(self, G: nx.Graph, pos: Dict,
                           node_colors: Optional[np.ndarray],
                           node_sizes: Optional[np.ndarray],
                           title: str) -> plt.Figure:
        """Create static matplotlib graph"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Draw graph
        nx.draw_networkx_nodes(G, pos, 
                              node_color=node_colors if node_colors is not None else 'lightblue',
                              node_size=node_sizes if node_sizes is not None else 300,
                              alpha=0.9, ax=ax)
        
        nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
        
        ax.set_title(title, fontsize=16)
        ax.axis('off')
        
        return fig
    
    def visualize_clusters(self, 
                          clusters: Dict[int, List[int]],
                          embeddings: Optional[np.ndarray] = None,
                          building_data: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Visualize clustering results
        
        Args:
            clusters: Cluster assignments
            embeddings: Node embeddings (for 2D projection)
            building_data: Building metadata
            
        Returns:
            Interactive figure
        """
        # Prepare data
        data_points = []
        
        for cluster_id, building_indices in clusters.items():
            for building_idx in building_indices:
                point = {
                    'building_id': building_idx,
                    'cluster_id': cluster_id,
                    'cluster_color': self.cluster_colors[cluster_id % len(self.cluster_colors)]
                }
                
                # Add embeddings if available
                if embeddings is not None:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=2)
                    coords_2d = pca.fit_transform(embeddings)
                    point['x'] = coords_2d[building_idx, 0]
                    point['y'] = coords_2d[building_idx, 1]
                else:
                    # Random positioning
                    point['x'] = np.random.randn()
                    point['y'] = np.random.randn()
                
                # Add building data if available
                if building_data is not None and building_idx < len(building_data):
                    building = building_data.iloc[building_idx]
                    point['area'] = building.get('area', 0)
                    point['peak_demand'] = building.get('peak_demand', 0)
                    point['has_solar'] = building.get('has_solar', False)
                
                data_points.append(point)
        
        df = pd.DataFrame(data_points)
        
        # Create scatter plot
        fig = px.scatter(df, x='x', y='y', 
                        color='cluster_id',
                        size='peak_demand' if 'peak_demand' in df else None,
                        hover_data=['building_id', 'area', 'peak_demand'] if 'area' in df else ['building_id'],
                        title='Energy Community Clusters',
                        color_discrete_sequence=self.cluster_colors)
        
        fig.update_layout(
            paper_bgcolor=self.bg_color,
            plot_bgcolor=self.bg_color,
            font=dict(color=self.text_color)
        )
        
        return fig
    
    def visualize_solar_potential(self,
                                 solar_results: Dict,
                                 building_data: pd.DataFrame,
                                 map_center: Optional[Tuple[float, float]] = None) -> folium.Map:
        """
        Visualize solar potential on map
        
        Args:
            solar_results: Solar optimization results
            building_data: Building data with coordinates
            map_center: Map center coordinates
            
        Returns:
            Folium map
        """
        # Create base map
        if map_center is None:
            map_center = [52.0116, 4.3571]  # Default to Delft
        
        m = folium.Map(location=map_center, zoom_start=14)
        
        # Add buildings with solar potential
        if 'ranking' in solar_results:
            for rank, building_idx in enumerate(solar_results['ranking'][:20]):  # Top 20
                if building_idx < len(building_data):
                    building = building_data.iloc[building_idx]
                    
                    # Get capacity and score
                    capacity = solar_results.get('capacities', [0] * len(solar_results['ranking']))[rank]
                    score = rank + 1
                    
                    # Determine color based on ranking
                    if score <= 5:
                        color = 'green'
                    elif score <= 10:
                        color = 'orange'
                    else:
                        color = 'red'
                    
                    # Add marker
                    folium.Marker(
                        location=[building.get('latitude', map_center[0]), 
                                 building.get('longitude', map_center[1])],
                        popup=f"""
                        <b>Building {building_idx}</b><br>
                        Rank: {score}<br>
                        Capacity: {capacity:.1f} kWp<br>
                        Area: {building.get('area', 0):.0f} m²
                        """,
                        icon=folium.Icon(color=color, icon='sun', prefix='fa')
                    ).add_to(m)
        
        # Add heat map layer
        if 'capacities' in solar_results and len(building_data) > 0:
            heat_data = []
            for idx, capacity in enumerate(solar_results['capacities']):
                if idx < len(building_data) and capacity > 0:
                    building = building_data.iloc[idx]
                    heat_data.append([
                        building.get('latitude', map_center[0]),
                        building.get('longitude', map_center[1]),
                        capacity
                    ])
            
            if heat_data:
                plugins.HeatMap(heat_data).add_to(m)
        
        return m
    
    def create_dashboard(self, results: Dict, task: str) -> go.Figure:
        """
        Create comprehensive dashboard for task results
        
        Args:
            results: Task results
            task: Task type
            
        Returns:
            Dashboard figure
        """
        # Create subplots based on task
        if task == 'clustering':
            fig = self._create_clustering_dashboard(results)
        elif task == 'solar':
            fig = self._create_solar_dashboard(results)
        elif task == 'retrofit':
            fig = self._create_retrofit_dashboard(results)
        elif task == 'congestion':
            fig = self._create_congestion_dashboard(results)
        else:
            fig = self._create_generic_dashboard(results, task)
        
        return fig
    
    def _create_clustering_dashboard(self, results: Dict) -> go.Figure:
        """Create clustering dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cluster Sizes', 'Peak Reduction', 
                          'Complementarity Matrix', 'Self-Sufficiency'),
            specs=[[{'type': 'bar'}, {'type': 'indicator'}],
                   [{'type': 'heatmap'}, {'type': 'scatter'}]]
        )
        
        # Cluster sizes
        if 'clusters' in results:
            sizes = [len(buildings) for buildings in results['clusters'].values()]
            fig.add_trace(
                go.Bar(x=list(range(len(sizes))), y=sizes, name='Cluster Size'),
                row=1, col=1
            )
        
        # Peak reduction indicator
        if 'metrics' in results and 'peak_reduction' in results['metrics']:
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=results['metrics']['peak_reduction'] * 100,
                    title={'text': "Peak Reduction (%)"},
                    gauge={'axis': {'range': [0, 50]},
                          'bar': {'color': "darkgreen"},
                          'steps': [
                              {'range': [0, 10], 'color': "lightgray"},
                              {'range': [10, 30], 'color': "gray"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 40}}
                ),
                row=1, col=2
            )
        
        # Complementarity matrix (mock)
        if 'complementarity' in results:
            fig.add_trace(
                go.Heatmap(z=results['complementarity'], colorscale='RdBu'),
                row=2, col=1
            )
        
        # Self-sufficiency scatter
        if 'self_sufficiency' in results:
            fig.add_trace(
                go.Scatter(x=list(range(len(results['self_sufficiency']))),
                          y=results['self_sufficiency'],
                          mode='markers+lines',
                          name='Self-Sufficiency'),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="Clustering Dashboard")
        
        return fig
    
    def _create_solar_dashboard(self, results: Dict) -> go.Figure:
        """Create solar optimization dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Capacity Distribution', 'ROI Analysis',
                          'Top Buildings', 'Economic Summary'),
            specs=[[{'type': 'histogram'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'indicator'}]]
        )
        
        # Capacity distribution
        if 'capacities' in results:
            fig.add_trace(
                go.Histogram(x=results['capacities'], nbinsx=20, name='Capacity'),
                row=1, col=1
            )
        
        # ROI scatter
        if 'roi_years' in results and 'capacities' in results:
            fig.add_trace(
                go.Scatter(x=results['capacities'], y=results['roi_years'],
                          mode='markers', name='ROI',
                          marker=dict(color=results['roi_years'], colorscale='Viridis')),
                row=1, col=2
            )
        
        # Top buildings bar chart
        if 'ranking' in results and 'capacities' in results:
            top_10 = results['ranking'][:10]
            top_capacities = [results['capacities'][i] for i in top_10]
            fig.add_trace(
                go.Bar(x=[f"Building {i}" for i in top_10], y=top_capacities),
                row=2, col=1
            )
        
        # Economic indicator
        if 'total_capacity' in results:
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=results['total_capacity'],
                    title={'text': "Total Capacity (kWp)"},
                    delta={'reference': 500, 'relative': True}
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="Solar Optimization Dashboard")
        
        return fig
    
    def _create_retrofit_dashboard(self, results: Dict) -> go.Figure:
        """Create retrofit dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Priority Distribution', 'Savings Potential',
                          'Cost vs Savings', 'Investment Timeline'),
            specs=[[{'type': 'box'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Priority distribution
        if 'priority_scores' in results:
            fig.add_trace(
                go.Box(y=results['priority_scores'], name='Priority'),
                row=1, col=1
            )
        
        # Savings potential
        if 'energy_savings' in results:
            fig.add_trace(
                go.Bar(y=sorted(results['energy_savings'], reverse=True)[:20],
                      name='Savings'),
                row=1, col=2
            )
        
        # Cost vs Savings scatter
        if 'costs' in results and 'savings' in results:
            fig.add_trace(
                go.Scatter(x=results['costs'], y=results['savings'],
                          mode='markers', name='Cost-Benefit',
                          marker=dict(size=8, color=results['savings'])),
                row=2, col=1
            )
        
        # Investment timeline (mock)
        years = list(range(1, 11))
        cumulative_investment = np.cumsum(np.random.rand(10) * 100000)
        fig.add_trace(
            go.Scatter(x=years, y=cumulative_investment,
                      mode='lines+markers', name='Investment'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Retrofit Dashboard")
        
        return fig
    
    def _create_congestion_dashboard(self, results: Dict) -> go.Figure:
        """Create congestion prediction dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Congestion Probability', 'Alert Distribution',
                          'Temporal Forecast', 'Mitigation Impact'),
            specs=[[{'type': 'heatmap'}, {'type': 'pie'}],
                   [{'type': 'scatter'}, {'type': 'bar'}]]
        )
        
        # Congestion heatmap
        if 'congestion_matrix' in results:
            fig.add_trace(
                go.Heatmap(z=results['congestion_matrix'], colorscale='Reds'),
                row=1, col=1
            )
        
        # Alert distribution pie
        if 'alert_counts' in results:
            fig.add_trace(
                go.Pie(labels=['Green', 'Yellow', 'Orange', 'Red'],
                      values=results['alert_counts']),
                row=1, col=2
            )
        
        # Temporal forecast
        if 'temporal_forecast' in results:
            hours = list(range(len(results['temporal_forecast'])))
            fig.add_trace(
                go.Scatter(x=hours, y=results['temporal_forecast'],
                          mode='lines', name='Forecast',
                          fill='tozeroy'),
                row=2, col=1
            )
        
        # Mitigation impact
        mitigation_types = ['Load Shifting', 'Storage', 'Demand Response', 'Grid Upgrade']
        impact = [30, 25, 20, 45]  # Mock percentages
        fig.add_trace(
            go.Bar(x=mitigation_types, y=impact, name='Impact'),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Congestion Dashboard")
        
        return fig
    
    def _create_generic_dashboard(self, results: Dict, task: str) -> go.Figure:
        """Create generic dashboard for any task"""
        # Count available metrics
        metrics = {k: v for k, v in results.items() 
                  if isinstance(v, (int, float, list)) and not k.startswith('_')}
        
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig = make_subplots(rows=n_rows, cols=n_cols,
                           subplot_titles=list(metrics.keys()))
        
        for idx, (key, value) in enumerate(metrics.items()):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            if isinstance(value, list):
                # Create histogram or line plot
                fig.add_trace(
                    go.Histogram(x=value) if len(value) > 50 
                    else go.Scatter(y=value, mode='lines+markers'),
                    row=row, col=col
                )
            else:
                # Create indicator
                fig.add_trace(
                    go.Indicator(
                        mode="number",
                        value=value,
                        title={'text': key}
                    ),
                    row=row, col=col
                )
        
        fig.update_layout(height=300*n_rows, showlegend=False, 
                         title_text=f"{task.title()} Dashboard")
        
        return fig
    
    def export_visualization(self, fig: Union[go.Figure, plt.Figure], 
                           filename: str, format: str = 'html'):
        """
        Export visualization to file
        
        Args:
            fig: Figure to export
            filename: Output filename
            format: Export format ('html', 'png', 'pdf', 'svg')
        """
        if isinstance(fig, go.Figure):
            if format == 'html':
                fig.write_html(filename)
            elif format == 'png':
                fig.write_image(filename)
            elif format == 'pdf':
                fig.write_image(filename, format='pdf')
            elif format == 'svg':
                fig.write_image(filename, format='svg')
        elif isinstance(fig, plt.Figure):
            fig.savefig(filename, format=format, dpi=300, bbox_inches='tight')
        
        logger.info(f"Exported visualization to {filename}")

# Usage example
if __name__ == "__main__":
    # Create visualizer
    viz = GraphVisualizer(style='light')
    
    # Mock data
    clusters = {
        0: [0, 1, 2, 3, 4],
        1: [5, 6, 7, 8],
        2: [9, 10, 11, 12, 13, 14]
    }
    
    solar_results = {
        'ranking': list(range(20)),
        'capacities': np.random.rand(20) * 100,
        'roi_years': np.random.rand(20) * 15,
        'total_capacity': 500
    }
    
    # Create visualizations
    cluster_fig = viz.visualize_clusters(clusters)
    solar_dashboard = viz.create_dashboard(solar_results, 'solar')
    
    # Export
    viz.export_visualization(cluster_fig, 'clusters.html')
    viz.export_visualization(solar_dashboard, 'solar_dashboard.html')
    
    print("Visualizations created successfully")