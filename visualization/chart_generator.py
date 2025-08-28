"""
Chart Generation Module
Creates various visualizations for energy community analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import networkx as nx
from datetime import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class ChartGenerator:
    """Generates various charts and visualizations"""
    
    def __init__(self, output_dir: str = "results/charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'success': '#73AB84',
            'warning': '#F18F01',
            'danger': '#C73E1D',
            'info': '#6C91C2'
        }
        
        self.label_colors = {
            'A++++': '#006837',
            'A+++': '#1A9850',
            'A++': '#66BD63',
            'A+': '#A6D96A',
            'A': '#D9EF8B',
            'B': '#FEE08B',
            'C': '#FDAE61',
            'D': '#F46D43',
            'E': '#D73027',
            'F': '#A50026',
            'G': '#67001F'
        }
    
    def create_cluster_quality_heatmap(self, cluster_metrics: Dict, 
                                      save_path: Optional[str] = None) -> go.Figure:
        """Create heatmap of cluster quality metrics"""
        
        # Prepare data
        metrics_data = []
        cluster_ids = []
        
        metric_names = ['self_sufficiency', 'complementarity', 'peak_reduction', 
                       'temporal_stability', 'compactness']
        
        for cluster_id, metrics in cluster_metrics.items():
            cluster_ids.append(f"Cluster {cluster_id}")
            row = []
            for metric in metric_names:
                if hasattr(metrics, f"{metric}_score"):
                    value = getattr(metrics, f"{metric}_score", 0)
                elif hasattr(metrics, f"{metric}_ratio"):
                    value = getattr(metrics, f"{metric}_ratio", 0)
                else:
                    value = 0
                row.append(value)
            metrics_data.append(row)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=metrics_data,
            x=[m.replace('_', ' ').title() for m in metric_names],
            y=cluster_ids,
            colorscale='RdYlGn',
            colorbar=dict(title="Score"),
            text=np.round(metrics_data, 2),
            texttemplate="%{text}",
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="Cluster Quality Metrics Heatmap",
            xaxis_title="Metrics",
            yaxis_title="Clusters",
            height=400 + len(cluster_ids) * 30,
            width=800
        )
        
        if save_path:
            fig.write_html(self.output_dir / f"{save_path}.html")
            try:
                fig.write_image(self.output_dir / f"{save_path}.png")
            except (ValueError, ImportError):
                # Kaleido not installed, skip image export
                pass
        
        return fig
    
    def create_energy_flow_sankey(self, energy_flows: Dict,
                                 save_path: Optional[str] = None) -> go.Figure:
        """Create Sankey diagram of energy flows"""
        
        # Prepare nodes and links
        nodes = ["Solar Generation", "Wind Generation", "Grid Import", 
                "Building Demand", "EV Charging", "Storage", 
                "Grid Export", "P2P Sharing"]
        
        # Example flow data - replace with actual
        source = [0, 1, 2, 3, 3, 5, 7]  # From nodes
        target = [3, 3, 3, 4, 5, 6, 3]  # To nodes
        value = [100, 50, 200, 30, 20, 15, 40]  # Flow amounts
        
        if 'flows' in energy_flows:
            # Parse actual flow data
            pass
        
        # Create Sankey
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=["green", "green", "gray", "blue", "orange", 
                      "purple", "gray", "teal"]
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color="rgba(0,0,0,0.2)"
            )
        )])
        
        fig.update_layout(
            title="Energy Flow Diagram",
            font_size=10,
            height=600,
            width=1000
        )
        
        if save_path:
            fig.write_html(self.output_dir / f"{save_path}.html")
            try:
                fig.write_image(self.output_dir / f"{save_path}.png")
            except (ValueError, ImportError):
                # Kaleido not installed, skip image export
                pass
        
        return fig
    
    def create_temporal_patterns(self, temporal_data: pd.DataFrame,
                                save_path: Optional[str] = None) -> go.Figure:
        """Create temporal pattern visualizations"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Daily Load Profile", "Weekly Patterns",
                          "Seasonal Variations", "Peak vs Off-Peak"),
            specs=[[{"type": "scatter"}, {"type": "box"}],
                  [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Daily profile
        if 'hour' in temporal_data.columns and 'demand' in temporal_data.columns:
            hourly = temporal_data.groupby('hour')['demand'].mean()
            fig.add_trace(
                go.Scatter(x=hourly.index, y=hourly.values, 
                         mode='lines+markers', name='Avg Demand'),
                row=1, col=1
            )
        
        # Weekly patterns
        if 'day_of_week' in temporal_data.columns:
            fig.add_trace(
                go.Box(x=temporal_data['day_of_week'], 
                      y=temporal_data['demand'], name='Demand'),
                row=1, col=2
            )
        
        # Seasonal (monthly)
        if 'month' in temporal_data.columns:
            monthly = temporal_data.groupby('month')['demand'].mean()
            fig.add_trace(
                go.Scatter(x=monthly.index, y=monthly.values,
                         mode='lines+markers', name='Monthly Avg'),
                row=2, col=1
            )
        
        # Peak vs Off-peak
        if 'peak_hour' in temporal_data.columns:
            peak_data = temporal_data.groupby('peak_hour')['demand'].sum()
            fig.add_trace(
                go.Bar(x=['Off-Peak', 'Peak'], 
                      y=[peak_data.get(False, 0), peak_data.get(True, 0)],
                      marker_color=['green', 'red']),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Temporal Energy Patterns Analysis",
            height=800,
            width=1200,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(self.output_dir / f"{save_path}.html")
            try:
                fig.write_image(self.output_dir / f"{save_path}.png")
            except (ValueError, ImportError):
                # Kaleido not installed, skip image export
                pass
        
        return fig
    
    def create_network_graph(self, edge_index: np.ndarray, 
                           node_features: Dict,
                           cluster_assignments: Optional[List] = None,
                           save_path: Optional[str] = None) -> go.Figure:
        """Create interactive network graph visualization"""
        
        # Create networkx graph
        G = nx.Graph()
        
        # Add edges
        for i in range(edge_index.shape[1]):
            G.add_edge(edge_index[0, i], edge_index[1, i])
        
        # Calculate layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Prepare node trace
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            # Node info
            text = f"Building {node}"
            if 'energy_label' in node_features and node < len(node_features['energy_label']):
                text += f"<br>Label: {node_features['energy_label'][node]}"
            if cluster_assignments and node < len(cluster_assignments):
                text += f"<br>Cluster: {cluster_assignments[node]}"
                node_color.append(cluster_assignments[node])
            else:
                node_color.append(0)
            
            node_text.append(text)
        
        # Prepare edge trace
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create traces
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers',
            hoverinfo='text',
            text=node_text,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                color=node_color,
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Cluster',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            )
        )
        
        # Create figure
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(
                           title='Energy Community Network Graph',
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=20, l=5, r=5, t=40),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=800,
                           width=1000
                       ))
        
        if save_path:
            fig.write_html(self.output_dir / f"{save_path}.html")
            try:
                fig.write_image(self.output_dir / f"{save_path}.png")
            except (ValueError, ImportError):
                # Kaleido not installed, skip image export
                pass
        
        return fig
    
    def create_solar_roi_analysis(self, solar_results: Dict,
                                 save_path: Optional[str] = None) -> go.Figure:
        """Create solar ROI analysis charts"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("ROI Distribution", "Capacity by Energy Label",
                          "Payback Period vs Generation", "Priority Scores"),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                  [{"type": "scatter"}, {"type": "box"}]]
        )
        
        # ROI Distribution
        if 'roi_years' in solar_results:
            fig.add_trace(
                go.Histogram(x=solar_results['roi_years'], 
                           nbinsx=20, name='ROI Years'),
                row=1, col=1
            )
        
        # Capacity by label
        if 'installations_by_label' in solar_results:
            labels = list(solar_results['installations_by_label'].keys())
            capacities = list(solar_results['installations_by_label'].values())
            colors = [self.label_colors.get(l, 'gray') for l in labels]
            
            fig.add_trace(
                go.Bar(x=labels, y=capacities, 
                      marker_color=colors, name='Capacity'),
                row=1, col=2
            )
        
        # Payback vs Generation
        if 'payback_periods' in solar_results and 'annual_generation' in solar_results:
            fig.add_trace(
                go.Scatter(x=solar_results['annual_generation'],
                         y=solar_results['payback_periods'],
                         mode='markers', name='Buildings',
                         marker=dict(size=8, color='orange')),
                row=2, col=1
            )
        
        # Priority scores
        if 'priority_scores' in solar_results:
            fig.add_trace(
                go.Box(y=solar_results['priority_scores'], 
                      name='Priority', marker_color='green'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Solar Installation ROI Analysis",
            height=800,
            width=1200,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(self.output_dir / f"{save_path}.html")
            try:
                fig.write_image(self.output_dir / f"{save_path}.png")
            except (ValueError, ImportError):
                # Kaleido not installed, skip image export
                pass
        
        return fig
    
    def create_comparison_charts(self, before_data: Dict, after_data: Dict,
                               save_path: Optional[str] = None) -> go.Figure:
        """Create before/after comparison visualizations"""
        
        metrics = ['Self-Sufficiency', 'Peak Reduction', 'Cost Savings', 'Carbon Reduction']
        before_values = [
            before_data.get('self_sufficiency', 0) * 100,
            before_data.get('peak_reduction', 0) * 100,
            before_data.get('cost_savings', 0),
            before_data.get('carbon_reduction', 0)
        ]
        after_values = [
            after_data.get('self_sufficiency', 0) * 100,
            after_data.get('peak_reduction', 0) * 100,
            after_data.get('cost_savings', 0),
            after_data.get('carbon_reduction', 0)
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Before',
            x=metrics,
            y=before_values,
            marker_color='lightgray'
        ))
        
        fig.add_trace(go.Bar(
            name='After',
            x=metrics,
            y=after_values,
            marker_color='green'
        ))
        
        # Add improvement percentages
        for i, metric in enumerate(metrics):
            if before_values[i] > 0:
                improvement = ((after_values[i] - before_values[i]) / before_values[i]) * 100
            else:
                improvement = 100 if after_values[i] > 0 else 0
            
            fig.add_annotation(
                x=metric,
                y=max(before_values[i], after_values[i]) * 1.1,
                text=f"+{improvement:.1f}%",
                showarrow=False,
                font=dict(color='green', size=12)
            )
        
        fig.update_layout(
            title="Before/After Energy Community Implementation",
            xaxis_title="Metrics",
            yaxis_title="Value",
            barmode='group',
            height=600,
            width=1000
        )
        
        if save_path:
            fig.write_html(self.output_dir / f"{save_path}.html")
            try:
                fig.write_image(self.output_dir / f"{save_path}.png")
            except (ValueError, ImportError):
                # Kaleido not installed, skip image export
                pass
        
        return fig
    
    def create_lv_group_summary(self, lv_statistics: Dict,
                              save_path: Optional[str] = None) -> go.Figure:
        """Create LV group summary visualization"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Buildings per LV Group", "Cluster Coverage",
                          "Solar Potential", "Intervention Priority"),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                  [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Buildings per LV
        if 'buildings_per_lv' in lv_statistics:
            lv_ids = list(lv_statistics['buildings_per_lv'].keys())
            counts = list(lv_statistics['buildings_per_lv'].values())
            colors = ['green' if c >= 20 else 'orange' if c >= 10 else 'red' for c in counts]
            
            fig.add_trace(
                go.Bar(x=lv_ids, y=counts, marker_color=colors),
                row=1, col=1
            )
        
        # Cluster coverage pie
        if 'cluster_coverage' in lv_statistics:
            fig.add_trace(
                go.Pie(labels=['With Clusters', 'Without Clusters'],
                      values=[lv_statistics['cluster_coverage'],
                             100 - lv_statistics['cluster_coverage']],
                      marker_colors=['green', 'lightgray']),
                row=1, col=2
            )
        
        # Solar potential scatter
        if 'solar_potential' in lv_statistics and 'building_count' in lv_statistics:
            fig.add_trace(
                go.Scatter(x=lv_statistics['building_count'],
                         y=lv_statistics['solar_potential'],
                         mode='markers',
                         marker=dict(size=10, color='orange')),
                row=2, col=1
            )
        
        # Priority scores
        if 'priority_scores' in lv_statistics:
            sorted_items = sorted(lv_statistics['priority_scores'].items(), 
                                key=lambda x: x[1], reverse=True)[:10]
            lv_ids = [item[0] for item in sorted_items]
            scores = [item[1] for item in sorted_items]
            
            fig.add_trace(
                go.Bar(x=lv_ids, y=scores, marker_color='purple'),
                row=2, col=2
            )
        
        fig.update_layout(
            title="LV Group Analysis Summary",
            height=800,
            width=1200,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(self.output_dir / f"{save_path}.html")
            try:
                fig.write_image(self.output_dir / f"{save_path}.png")
            except (ValueError, ImportError):
                # Kaleido not installed, skip image export
                pass
        
        return fig
    
    def create_economic_dashboard(self, economic_data: Dict,
                                save_path: Optional[str] = None) -> go.Figure:
        """Create economic metrics dashboard"""
        
        # Create gauge charts for key metrics
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                  [{"type": "bar"}, {"type": "pie"}, {"type": "scatter"}]],
            subplot_titles=("", "", "", "Monthly Savings", "Cost Distribution", "ROI Timeline")
        )
        
        # Cost reduction gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=economic_data.get('cost_reduction_percent', 0),
                title={'text': "Cost Reduction (%)"},
                delta={'reference': 15},  # Target
                gauge={'axis': {'range': [None, 50]},
                      'bar': {'color': "green"},
                      'steps': [
                          {'range': [0, 10], 'color': "lightgray"},
                          {'range': [10, 25], 'color': "yellow"},
                          {'range': [25, 50], 'color': "lightgreen"}],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75, 'value': 15}}),
            row=1, col=1
        )
        
        # Carbon reduction gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=economic_data.get('carbon_tons_saved', 0),
                title={'text': "CO2 Saved (tons)"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': "darkgreen"}}),
            row=1, col=2
        )
        
        # ROI gauge
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=economic_data.get('avg_roi_years', 0),
                title={'text': "Avg ROI (years)"},
                gauge={'axis': {'range': [None, 15]},
                      'bar': {'color': "orange"},
                      'steps': [
                          {'range': [0, 5], 'color': "lightgreen"},
                          {'range': [5, 10], 'color': "yellow"},
                          {'range': [10, 15], 'color': "lightsalmon"}]}),
            row=1, col=3
        )
        
        # Monthly savings bar
        if 'monthly_savings' in economic_data:
            months = list(economic_data['monthly_savings'].keys())
            savings = list(economic_data['monthly_savings'].values())
            fig.add_trace(
                go.Bar(x=months, y=savings, marker_color='green'),
                row=2, col=1
            )
        
        # Cost distribution pie
        if 'cost_breakdown' in economic_data:
            fig.add_trace(
                go.Pie(labels=list(economic_data['cost_breakdown'].keys()),
                      values=list(economic_data['cost_breakdown'].values())),
                row=2, col=2
            )
        
        # ROI timeline
        if 'roi_timeline' in economic_data:
            years = list(range(len(economic_data['roi_timeline'])))
            cumulative = list(economic_data['roi_timeline'])  # Convert to list
            fig.add_trace(
                go.Scatter(x=years, y=cumulative, mode='lines+markers',
                         line=dict(color='purple', width=2)),
                row=2, col=3
            )
        
        fig.update_layout(
            title="Economic Performance Dashboard",
            height=800,
            width=1400,
            showlegend=False
        )
        
        if save_path:
            fig.write_html(self.output_dir / f"{save_path}.html")
            try:
                fig.write_image(self.output_dir / f"{save_path}.png")
            except (ValueError, ImportError):
                # Kaleido not installed, skip image export
                pass
        
        return fig