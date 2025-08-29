"""
Solar Roadmap Visualizer
Creates timeline visualizations and progress tracking for solar deployment roadmaps
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class RoadmapVisualizer:
    """
    Visualizes solar deployment roadmaps and tracks progress
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize roadmap visualizer
        
        Args:
            config: Visualization configuration
        """
        self.config = config or {}
        
        # Color schemes
        self.colors = {
            'planned': '#3498db',
            'completed': '#2ecc71',
            'delayed': '#e74c3c',
            'on_track': '#f39c12',
            'clusters': px.colors.qualitative.Set3,
            'roi': {
                'excellent': '#27ae60',
                'good': '#3498db',
                'fair': '#f39c12',
                'poor': '#e74c3c'
            }
        }
        
        logger.info("Initialized RoadmapVisualizer")
    
    def create_timeline_gantt(
        self,
        roadmap: Any,  # SolarRoadmap type
        progress: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create Gantt chart showing deployment timeline
        
        Args:
            roadmap: Solar deployment roadmap
            progress: Optional progress tracking data
            
        Returns:
            Plotly figure with Gantt chart
        """
        # Prepare data for Gantt
        tasks = []
        start_date = datetime.now()
        
        for plan in roadmap.yearly_plans:
            end_date = start_date + timedelta(days=365)
            
            # Main deployment task
            tasks.append({
                'Task': f'Year {plan.year}',
                'Start': start_date,
                'Finish': end_date,
                'Resource': 'Deployment',
                'Buildings': len(plan.target_installations),
                'Capacity_MW': plan.total_capacity_mw,
                'Investment': plan.budget_required,
                'Complete': 0
            })
            
            # Add progress if available
            if progress and f'year_{plan.year}' in progress:
                year_progress = progress[f'year_{plan.year}']
                complete_date = start_date + timedelta(
                    days=365 * year_progress['percentage_complete'] / 100
                )
                tasks.append({
                    'Task': f'Year {plan.year} Progress',
                    'Start': start_date,
                    'Finish': complete_date,
                    'Resource': 'Completed',
                    'Buildings': year_progress.get('completed_buildings', 0),
                    'Capacity_MW': year_progress.get('completed_capacity', 0),
                    'Investment': year_progress.get('spent_budget', 0),
                    'Complete': year_progress['percentage_complete']
                })
            
            start_date = end_date
        
        # Create DataFrame
        df = pd.DataFrame(tasks)
        
        # Create Gantt chart
        fig = px.timeline(
            df,
            x_start='Start',
            x_end='Finish',
            y='Task',
            color='Resource',
            hover_data=['Buildings', 'Capacity_MW', 'Investment', 'Complete'],
            title=f'Solar Deployment Timeline - {roadmap.timeframe_years} Year Plan',
            color_discrete_map={
                'Deployment': self.colors['planned'],
                'Completed': self.colors['completed']
            }
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='Timeline',
            yaxis_title='Deployment Phase',
            height=400 + len(tasks) * 30,
            showlegend=True,
            hovermode='closest'
        )
        
        # Add target line
        target_date = datetime.now() + timedelta(days=365 * roadmap.timeframe_years)
        fig.add_vline(
            x=target_date,
            line_dash='dash',
            line_color='red',
            annotation_text=f'Target: {roadmap.target_penetration:.0%}'
        )
        
        return fig
    
    def create_penetration_progress(
        self,
        roadmap: Any,
        current_progress: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create penetration rate progress chart
        
        Args:
            roadmap: Solar deployment roadmap
            current_progress: Current progress data
            
        Returns:
            Plotly figure with progress visualization
        """
        # Prepare data
        years = list(range(roadmap.timeframe_years + 1))
        planned_penetration = [roadmap.current_penetration]
        
        for plan in roadmap.yearly_plans:
            planned_penetration.append(plan.cumulative_penetration)
        
        # Create figure
        fig = go.Figure()
        
        # Add planned trajectory
        fig.add_trace(go.Scatter(
            x=years,
            y=[p * 100 for p in planned_penetration],
            mode='lines+markers',
            name='Planned',
            line=dict(color=self.colors['planned'], width=3),
            marker=dict(size=8)
        ))
        
        # Add actual progress if available
        if current_progress:
            actual_years = []
            actual_penetration = []
            
            for year, data in current_progress.items():
                if year.startswith('year_'):
                    year_num = int(year.split('_')[1])
                    actual_years.append(year_num)
                    actual_penetration.append(data['penetration'] * 100)
            
            if actual_years:
                fig.add_trace(go.Scatter(
                    x=actual_years,
                    y=actual_penetration,
                    mode='lines+markers',
                    name='Actual',
                    line=dict(color=self.colors['completed'], width=3, dash='dash'),
                    marker=dict(size=10, symbol='diamond')
                ))
        
        # Add target line
        fig.add_hline(
            y=roadmap.target_penetration * 100,
            line_dash='dash',
            line_color='red',
            annotation_text=f'Target: {roadmap.target_penetration:.0%}'
        )
        
        # Update layout
        fig.update_layout(
            title='Solar Penetration Progress',
            xaxis_title='Year',
            yaxis_title='Penetration Rate (%)',
            yaxis_range=[0, max(roadmap.target_penetration * 120, 30)],
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_capacity_buildup(
        self,
        roadmap: Any,
        by_cluster: bool = False
    ) -> go.Figure:
        """
        Create capacity buildup visualization
        
        Args:
            roadmap: Solar deployment roadmap
            by_cluster: Show breakdown by cluster
            
        Returns:
            Plotly figure with capacity buildup
        """
        # Prepare data
        years = []
        capacities = []
        clusters = {}
        
        cumulative_capacity = 0
        for plan in roadmap.yearly_plans:
            years.append(f'Year {plan.year}')
            cumulative_capacity += plan.total_capacity_mw
            capacities.append(cumulative_capacity)
            
            if by_cluster and plan.cluster_assignments:
                for building_id, cluster_id in plan.cluster_assignments.items():
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    while len(clusters[cluster_id]) < plan.year:
                        clusters[cluster_id].append(0)
                    clusters[cluster_id][-1] += plan.capacities[
                        plan.target_installations.index(building_id)
                    ] / 1000 if building_id in plan.target_installations else 0
        
        # Create figure
        fig = go.Figure()
        
        if by_cluster and clusters:
            # Stacked area chart by cluster
            for cluster_id, cluster_capacities in clusters.items():
                fig.add_trace(go.Scatter(
                    x=years[:len(cluster_capacities)],
                    y=cluster_capacities,
                    mode='lines',
                    stackgroup='one',
                    name=f'Cluster {cluster_id}',
                    fill='tonexty'
                ))
        else:
            # Single capacity buildup
            fig.add_trace(go.Bar(
                x=years,
                y=capacities,
                name='Cumulative Capacity',
                marker_color=self.colors['planned'],
                text=[f'{c:.2f} MW' for c in capacities],
                textposition='outside'
            ))
        
        # Update layout
        fig.update_layout(
            title='Solar Capacity Buildup',
            xaxis_title='Deployment Phase',
            yaxis_title='Capacity (MW)',
            hovermode='x unified',
            showlegend=by_cluster
        )
        
        return fig
    
    def create_investment_timeline(
        self,
        roadmap: Any,
        show_roi: bool = True
    ) -> go.Figure:
        """
        Create investment and ROI timeline
        
        Args:
            roadmap: Solar deployment roadmap
            show_roi: Include ROI projections
            
        Returns:
            Plotly figure with investment timeline
        """
        # Create subplots
        fig = make_subplots(
            rows=2 if show_roi else 1,
            cols=1,
            subplot_titles=['Annual Investment', 'Cumulative ROI'] if show_roi else ['Annual Investment'],
            vertical_spacing=0.15
        )
        
        # Prepare data
        years = []
        investments = []
        cumulative_inv = []
        roi_years = []
        
        total_inv = 0
        for plan in roadmap.yearly_plans:
            years.append(f'Year {plan.year}')
            investments.append(plan.budget_required)
            total_inv += plan.budget_required
            cumulative_inv.append(total_inv)
            
            # Simple ROI calculation
            if show_roi and total_inv > 0:
                annual_savings = plan.expected_self_sufficiency * 100000  # Rough estimate
                payback = total_inv / annual_savings if annual_savings > 0 else 25
                roi_years.append(payback)
        
        # Add investment bars
        fig.add_trace(
            go.Bar(
                x=years,
                y=investments,
                name='Annual Investment',
                marker_color=self.colors['planned'],
                text=[f'€{inv/1000:.0f}k' for inv in investments],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Add cumulative line
        fig.add_trace(
            go.Scatter(
                x=years,
                y=cumulative_inv,
                mode='lines+markers',
                name='Cumulative Investment',
                line=dict(color=self.colors['on_track'], width=2),
                yaxis='y2'
            ),
            row=1, col=1
        )
        
        # Add ROI if requested
        if show_roi and roi_years:
            fig.add_trace(
                go.Scatter(
                    x=years,
                    y=roi_years,
                    mode='lines+markers',
                    name='Payback Period',
                    line=dict(color=self.colors['completed'], width=2),
                    marker=dict(size=8)
                ),
                row=2, col=1
            )
            
            # Add acceptable ROI threshold
            fig.add_hline(
                y=7,
                line_dash='dash',
                line_color='green',
                annotation_text='Good ROI (< 7 years)',
                row=2, col=1
            )
        
        # Update layout
        fig.update_xaxes(title_text='Deployment Phase')
        fig.update_yaxes(title_text='Investment (€)', row=1, col=1)
        if show_roi:
            fig.update_yaxes(title_text='Payback (Years)', row=2, col=1)
        
        fig.update_layout(
            title='Investment Timeline and Returns',
            height=600 if show_roi else 400,
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_cluster_evolution(
        self,
        roadmap: Any,
        evolution_data: Optional[List[Dict]] = None
    ) -> go.Figure:
        """
        Create cluster evolution visualization
        
        Args:
            roadmap: Solar deployment roadmap
            evolution_data: Cluster evolution trajectory
            
        Returns:
            Plotly figure showing cluster changes
        """
        if not evolution_data and hasattr(roadmap, 'cluster_evolution'):
            evolution_data = roadmap.cluster_evolution
        
        if not evolution_data:
            return go.Figure().add_annotation(
                text="No cluster evolution data available",
                showarrow=False
            )
        
        # Prepare data
        years = []
        num_clusters = []
        stability_scores = []
        self_sufficiency = []
        
        for point in evolution_data:
            years.append(f"Year {point['year']}")
            num_clusters.append(point.get('num_clusters', 0))
            stability_scores.append(point.get('stability_score', 0) * 100)
            self_sufficiency.append(point.get('self_sufficiency', 0) * 100)
        
        # Create figure with secondary y-axis
        fig = make_subplots(
            specs=[[{"secondary_y": True}]]
        )
        
        # Add number of clusters
        fig.add_trace(
            go.Bar(
                x=years,
                y=num_clusters,
                name='Number of Clusters',
                marker_color=self.colors['clusters'][0],
                opacity=0.7
            ),
            secondary_y=False
        )
        
        # Add stability score
        fig.add_trace(
            go.Scatter(
                x=years,
                y=stability_scores,
                mode='lines+markers',
                name='Cluster Stability',
                line=dict(color=self.colors['planned'], width=2),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        # Add self-sufficiency
        fig.add_trace(
            go.Scatter(
                x=years,
                y=self_sufficiency,
                mode='lines+markers',
                name='Self-Sufficiency',
                line=dict(color=self.colors['completed'], width=2, dash='dash'),
                marker=dict(size=8, symbol='diamond')
            ),
            secondary_y=True
        )
        
        # Update axes
        fig.update_xaxes(title_text='Deployment Phase')
        fig.update_yaxes(title_text='Number of Clusters', secondary_y=False)
        fig.update_yaxes(title_text='Percentage (%)', secondary_y=True)
        
        # Update layout
        fig.update_layout(
            title='Cluster Evolution Over Time',
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        return fig
    
    def create_geographic_deployment(
        self,
        roadmap: Any,
        building_locations: pd.DataFrame
    ) -> go.Figure:
        """
        Create geographic deployment map
        
        Args:
            roadmap: Solar deployment roadmap
            building_locations: DataFrame with building coordinates
            
        Returns:
            Plotly figure with deployment map
        """
        # Prepare data
        deployment_data = []
        
        for plan in roadmap.yearly_plans:
            for building_id in plan.target_installations:
                if building_id in building_locations.index:
                    loc = building_locations.loc[building_id]
                    deployment_data.append({
                        'building_id': building_id,
                        'year': plan.year,
                        'x': loc.get('x', 0),
                        'y': loc.get('y', 0),
                        'capacity': plan.capacities[
                            plan.target_installations.index(building_id)
                        ]
                    })
        
        if not deployment_data:
            return go.Figure().add_annotation(
                text="No geographic data available",
                showarrow=False
            )
        
        df = pd.DataFrame(deployment_data)
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='x',
            y='y',
            color='year',
            size='capacity',
            hover_data=['building_id', 'capacity'],
            title='Geographic Deployment Plan',
            color_continuous_scale='Viridis',
            labels={'year': 'Deployment Year'}
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            hovermode='closest'
        )
        
        return fig
    
    def create_benefits_summary(
        self,
        roadmap: Any
    ) -> go.Figure:
        """
        Create benefits summary dashboard
        
        Args:
            roadmap: Solar deployment roadmap
            
        Returns:
            Plotly figure with benefits summary
        """
        benefits = roadmap.expected_benefits
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Environmental Impact',
                'Economic Benefits',
                'Grid Benefits',
                'Community Impact'
            ],
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]]
        )
        
        # CO2 reduction
        fig.add_trace(
            go.Indicator(
                mode='number+delta',
                value=benefits.get('annual_co2_reduction_tons', 0),
                title={'text': 'CO₂ Reduction<br>(tons/year)'},
                delta={'reference': 0, 'relative': False},
                number={'suffix': ' tons'},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=1
        )
        
        # Energy generation
        fig.add_trace(
            go.Indicator(
                mode='number+gauge',
                value=benefits.get('annual_energy_generated_gwh', 0),
                title={'text': 'Annual Generation<br>(GWh)'},
                number={'suffix': ' GWh'},
                gauge={'axis': {'range': [0, benefits.get('annual_energy_generated_gwh', 0) * 1.5]}},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=1, col=2
        )
        
        # Grid investment avoided
        fig.add_trace(
            go.Indicator(
                mode='number+delta',
                value=benefits.get('grid_investment_avoided', 0),
                title={'text': 'Grid Investment<br>Avoided (€)'},
                number={'prefix': '€', 'valueformat': ',.0f'},
                delta={'reference': 0},
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=1
        )
        
        # Peak reduction
        fig.add_trace(
            go.Indicator(
                mode='number+gauge',
                value=benefits.get('total_peak_reduction_mw', 0),
                title={'text': 'Peak Reduction<br>(MW)'},
                number={'suffix': ' MW', 'valueformat': '.2f'},
                gauge={
                    'axis': {'range': [0, benefits.get('total_peak_reduction_mw', 0) * 1.5]},
                    'bar': {'color': self.colors['completed']}
                },
                domain={'x': [0, 1], 'y': [0, 1]}
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title='Expected Benefits Summary',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def export_roadmap_report(
        self,
        roadmap: Any,
        filepath: str,
        include_charts: bool = True
    ):
        """
        Export comprehensive roadmap report
        
        Args:
            roadmap: Solar deployment roadmap
            filepath: Output file path
            include_charts: Include visualization charts
        """
        import json
        
        # Prepare report data
        report = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'target_penetration': roadmap.target_penetration,
                'timeframe_years': roadmap.timeframe_years,
                'total_investment': roadmap.total_investment,
                'strategy': roadmap.optimization_strategy
            },
            'yearly_plans': [],
            'benefits': roadmap.expected_benefits,
            'cluster_evolution': roadmap.cluster_evolution
        }
        
        # Add yearly details
        for plan in roadmap.yearly_plans:
            report['yearly_plans'].append({
                'year': plan.year,
                'installations': len(plan.target_installations),
                'capacity_mw': plan.total_capacity_mw,
                'investment': plan.budget_required,
                'penetration': plan.cumulative_penetration,
                'self_sufficiency': plan.expected_self_sufficiency
            })
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Roadmap report exported to {filepath}")
        
        # Create HTML report with charts if requested
        if include_charts:
            html_path = filepath.replace('.json', '.html')
            
            # Create all visualizations
            timeline = self.create_timeline_gantt(roadmap)
            penetration = self.create_penetration_progress(roadmap)
            capacity = self.create_capacity_buildup(roadmap)
            investment = self.create_investment_timeline(roadmap)
            evolution = self.create_cluster_evolution(roadmap)
            benefits = self.create_benefits_summary(roadmap)
            
            # Combine into single HTML
            from plotly.subplots import make_subplots
            import plotly.offline as pyo
            
            html_content = f"""
            <html>
            <head>
                <title>Solar Roadmap Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #2c3e50; }}
                    .summary {{ background: #ecf0f1; padding: 15px; border-radius: 5px; margin: 20px 0; }}
                </style>
            </head>
            <body>
                <h1>Solar Deployment Roadmap Report</h1>
                <div class="summary">
                    <h2>Summary</h2>
                    <ul>
                        <li>Target Penetration: {roadmap.target_penetration:.1%}</li>
                        <li>Timeframe: {roadmap.timeframe_years} years</li>
                        <li>Total Investment: €{roadmap.total_investment:,.0f}</li>
                        <li>Total Capacity: {roadmap.expected_benefits.get('total_capacity_mw', 0):.2f} MW</li>
                        <li>Strategy: {roadmap.optimization_strategy}</li>
                    </ul>
                </div>
                {timeline.to_html(include_plotlyjs='cdn')}
                {penetration.to_html(include_plotlyjs=False)}
                {capacity.to_html(include_plotlyjs=False)}
                {investment.to_html(include_plotlyjs=False)}
                {evolution.to_html(include_plotlyjs=False)}
                {benefits.to_html(include_plotlyjs=False)}
            </body>
            </html>
            """
            
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"HTML report with charts exported to {html_path}")