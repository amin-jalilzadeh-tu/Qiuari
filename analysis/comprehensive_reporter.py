"""
Comprehensive Cluster Analysis Reporter
Generates detailed reports with clusters, buildings, time steps, and interventions
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict

@dataclass
class BuildingProfile:
    """Building energy profile and characteristics"""
    building_id: str
    cluster_id: int
    lv_group: str
    area: float
    energy_label: str
    peak_demand: float
    base_load: float
    solar_potential: float
    consumption_profile: np.ndarray  # 96 timesteps
    generation_profile: np.ndarray  # 96 timesteps
    
@dataclass
class ClusterSummary:
    """Cluster-level summary statistics"""
    cluster_id: int
    lv_group: str
    num_buildings: int
    total_area: float
    avg_energy_label: float
    total_peak_demand: float
    total_base_load: float
    self_sufficiency: float
    complementarity_score: float
    peak_reduction: float
    peak_hours: List[int]
    valley_hours: List[int]
    
@dataclass
class InterventionRecommendation:
    """Intervention recommendation for building/cluster"""
    target_id: str  # building_id or cluster_id
    target_type: str  # 'building' or 'cluster'
    intervention_type: str  # 'solar', 'battery', 'retrofit', 'heat_pump'
    size: float
    cost: float
    annual_savings: float
    co2_reduction: float
    payback_years: float
    priority: float
    network_benefit: float
    cascade_potential: float

class ComprehensiveReporter:
    """Generate comprehensive analysis reports"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.timesteps_per_hour = 4  # 15-minute intervals
        self.hours_per_day = 24
        
        # Intervention costs (€/kW or €/kWh)
        self.costs = {
            'solar': 1000,  # €/kW
            'battery': 500,  # €/kWh
            'retrofit': 50,  # €/m² 
            'heat_pump': 2000  # €/kW
        }
        
        # Emission factors
        self.grid_emission_factor = 0.4  # kg CO2/kWh
        self.solar_emission_factor = 0.0  # kg CO2/kWh
        
    def generate_full_report(
        self,
        clusters: np.ndarray,
        building_data: Dict,
        gnn_outputs: Dict,
        intervention_plan: Dict,
        save_dir: str = "reports"
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive report with all analyses
        
        Args:
            clusters: Cluster assignments for buildings
            building_data: Building features and profiles
            gnn_outputs: GNN model outputs
            intervention_plan: Recommended interventions
            save_dir: Directory to save reports
            
        Returns:
            Dictionary of DataFrames with different report sections
        """
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        reports = {}
        
        # 1. Building-level report
        print("\n[1/6] Generating building-level report...")
        reports['buildings'] = self._generate_building_report(
            clusters, building_data, gnn_outputs
        )
        
        # 2. Cluster-level report
        print("[2/6] Generating cluster-level report...")
        reports['clusters'] = self._generate_cluster_report(
            clusters, building_data, gnn_outputs
        )
        
        # 3. Temporal analysis report
        print("[3/6] Generating temporal analysis report...")
        reports['temporal'] = self._generate_temporal_report(
            clusters, building_data
        )
        
        # 4. Intervention recommendations
        print("[4/6] Generating intervention recommendations...")
        reports['interventions'] = self._generate_intervention_report(
            clusters, building_data, intervention_plan, gnn_outputs
        )
        
        # 5. Network analysis report
        print("[5/6] Generating network analysis report...")
        reports['network'] = self._generate_network_report(
            clusters, building_data, gnn_outputs
        )
        
        # 6. Economic analysis report
        print("[6/6] Generating economic analysis report...")
        reports['economics'] = self._generate_economic_report(
            reports['interventions'], reports['clusters']
        )
        
        # Save all reports
        for name, df in reports.items():
            file_path = save_path / f"{name}_report_{timestamp}.csv"
            df.to_csv(file_path, index=False)
            print(f"   Saved: {file_path}")
            
        # Generate summary visualization
        self._generate_summary_visualization(reports, save_path / f"summary_{timestamp}.png")
        
        # Generate detailed Excel workbook
        excel_path = save_path / f"comprehensive_report_{timestamp}.xlsx"
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            for name, df in reports.items():
                df.to_excel(writer, sheet_name=name.capitalize(), index=False)
        print(f"\nComplete report saved to: {excel_path}")
        
        return reports
    
    def _generate_building_report(
        self,
        clusters: np.ndarray,
        building_data: Dict,
        gnn_outputs: Dict
    ) -> pd.DataFrame:
        """Generate detailed building-level report"""
        
        buildings = []
        
        for i, building_id in enumerate(building_data.get('building_ids', range(len(clusters)))):
            building = {
                'building_id': f'BLDG_{building_id:04d}',
                'cluster_id': int(clusters[i]),
                'lv_group': building_data.get('lv_group', 'LV_GROUP_0001'),
                
                # Physical characteristics
                'area_m2': building_data.get('areas', [100])[i] if i < len(building_data.get('areas', [])) else 100,
                'height_m': building_data.get('heights', [10])[i] if i < len(building_data.get('heights', [])) else 10,
                'age_years': building_data.get('ages', [20])[i] if i < len(building_data.get('ages', [])) else 20,
                'energy_label': self._decode_energy_label(
                    building_data.get('energy_labels', [4])[i] if i < len(building_data.get('energy_labels', [])) else 4
                ),
                
                # Energy characteristics
                'peak_demand_kw': building_data.get('peak_demands', [5])[i] if i < len(building_data.get('peak_demands', [])) else 5,
                'base_load_kw': building_data.get('base_loads', [1])[i] if i < len(building_data.get('base_loads', [])) else 1,
                'annual_consumption_kwh': building_data.get('annual_consumption', [5000])[i] if i < len(building_data.get('annual_consumption', [])) else 5000,
                
                # Solar potential
                'roof_area_m2': building_data.get('roof_areas', [50])[i] if i < len(building_data.get('roof_areas', [])) else 50,
                'solar_potential_kw': building_data.get('solar_potentials', [10])[i] if i < len(building_data.get('solar_potentials', [])) else 10,
                'solar_suitability': building_data.get('solar_suitability', ['high'])[i] if i < len(building_data.get('solar_suitability', [])) else 'high',
                
                # Existing systems
                'has_solar': building_data.get('has_solar', [False])[i] if i < len(building_data.get('has_solar', [])) else False,
                'has_battery': building_data.get('has_battery', [False])[i] if i < len(building_data.get('has_battery', [])) else False,
                'has_heat_pump': building_data.get('has_heat_pump', [False])[i] if i < len(building_data.get('has_heat_pump', [])) else False,
                
                # GNN outputs
                'cluster_probability': gnn_outputs['cluster_probs'][i][clusters[i]] if 'cluster_probs' in gnn_outputs and i < len(gnn_outputs['cluster_probs']) and clusters[i] < len(gnn_outputs['cluster_probs'][i]) else 1.0,
                'network_centrality': gnn_outputs.get('network_centrality_score', [0.5])[i] if i < len(gnn_outputs.get('network_centrality_score', [])) else 0.5,
                'intervention_priority': gnn_outputs.get('intervention_priority', [0.5])[i] if i < len(gnn_outputs.get('intervention_priority', [])) else 0.5,
            }
            
            # Add temporal metrics
            if 'consumption_profiles' in building_data and i < len(building_data['consumption_profiles']):
                profile = building_data['consumption_profiles'][i]
                building['morning_peak_kw'] = np.max(profile[6*4:9*4])  # 6am-9am
                building['evening_peak_kw'] = np.max(profile[17*4:21*4])  # 5pm-9pm
                building['night_minimum_kw'] = np.min(profile[0:6*4])  # 12am-6am
                building['load_factor'] = np.mean(profile) / np.max(profile) if np.max(profile) > 0 else 0
            
            buildings.append(building)
            
        return pd.DataFrame(buildings)
    
    def _generate_cluster_report(
        self,
        clusters: np.ndarray,
        building_data: Dict,
        gnn_outputs: Dict
    ) -> pd.DataFrame:
        """Generate cluster-level summary report"""
        
        cluster_summaries = []
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
                
            # Aggregate building data
            summary = {
                'cluster_id': int(cluster_id),
                'lv_group': building_data.get('lv_group', 'LV_GROUP_0001'),
                'num_buildings': len(cluster_indices),
                
                # Physical aggregates
                'total_area_m2': sum(building_data.get('areas', [100]*len(cluster_indices))[i] 
                                    for i in cluster_indices if i < len(building_data.get('areas', []))),
                'avg_age_years': np.mean([building_data.get('ages', [20]*len(cluster_indices))[i] 
                                         for i in cluster_indices if i < len(building_data.get('ages', []))]),
                
                # Energy aggregates
                'total_peak_demand_kw': sum(building_data.get('peak_demands', [5]*len(cluster_indices))[i] 
                                           for i in cluster_indices if i < len(building_data.get('peak_demands', []))),
                'total_base_load_kw': sum(building_data.get('base_loads', [1]*len(cluster_indices))[i] 
                                         for i in cluster_indices if i < len(building_data.get('base_loads', []))),
                'total_annual_consumption_mwh': sum(building_data.get('annual_consumption', [5000]*len(cluster_indices))[i] 
                                                   for i in cluster_indices if i < len(building_data.get('annual_consumption', []))) / 1000,
                
                # Performance metrics from GNN
                'self_sufficiency': gnn_outputs.get('cluster_self_sufficiency', {}).get(cluster_id, 0.0),
                'complementarity_score': gnn_outputs.get('cluster_complementarity', {}).get(cluster_id, 0.0),
                'peak_reduction': gnn_outputs.get('cluster_peak_reduction', {}).get(cluster_id, 0.0),
                'network_stress': gnn_outputs.get('cluster_network_stress', {}).get(cluster_id, 0.0),
            }
            
            # Identify peak and valley hours
            if 'consumption_profiles' in building_data:
                cluster_profile = np.zeros(96)  # 24 hours * 4 intervals
                for i in cluster_indices:
                    if i < len(building_data['consumption_profiles']):
                        cluster_profile += building_data['consumption_profiles'][i]
                
                # Find peak hours (top 20%)
                hourly_avg = [np.mean(cluster_profile[h*4:(h+1)*4]) for h in range(24)]
                threshold_high = np.percentile(hourly_avg, 80)
                threshold_low = np.percentile(hourly_avg, 20)
                
                summary['peak_hours'] = [h for h in range(24) if hourly_avg[h] >= threshold_high]
                summary['valley_hours'] = [h for h in range(24) if hourly_avg[h] <= threshold_low]
                summary['peak_valley_ratio'] = max(hourly_avg) / min(hourly_avg) if min(hourly_avg) > 0 else 0
                
            cluster_summaries.append(summary)
            
        return pd.DataFrame(cluster_summaries)
    
    def _generate_temporal_report(
        self,
        clusters: np.ndarray,
        building_data: Dict
    ) -> pd.DataFrame:
        """Generate temporal analysis report for each cluster"""
        
        temporal_data = []
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0 or 'consumption_profiles' not in building_data:
                continue
            
            # Aggregate profiles
            consumption = np.zeros(96)
            generation = np.zeros(96)
            
            for i in cluster_indices:
                if i < len(building_data['consumption_profiles']):
                    consumption += building_data['consumption_profiles'][i]
                if 'generation_profiles' in building_data and i < len(building_data['generation_profiles']):
                    generation += building_data['generation_profiles'][i]
            
            # Analyze each timestep
            for timestep in range(96):
                hour = timestep // 4
                minute = (timestep % 4) * 15
                
                temporal_data.append({
                    'cluster_id': int(cluster_id),
                    'timestep': timestep,
                    'hour': hour,
                    'minute': minute,
                    'time_label': f"{hour:02d}:{minute:02d}",
                    'consumption_kw': consumption[timestep],
                    'generation_kw': generation[timestep],
                    'net_demand_kw': consumption[timestep] - generation[timestep],
                    'self_sufficiency_instant': generation[timestep] / consumption[timestep] if consumption[timestep] > 0 else 0,
                    'is_peak': consumption[timestep] > np.percentile(consumption, 90),
                    'is_valley': consumption[timestep] < np.percentile(consumption, 10),
                    'is_surplus': generation[timestep] > consumption[timestep]
                })
        
        return pd.DataFrame(temporal_data)
    
    def _generate_intervention_report(
        self,
        clusters: np.ndarray,
        building_data: Dict,
        intervention_plan,  # Can be InterventionPlan object or dict
        gnn_outputs: Dict
    ) -> pd.DataFrame:
        """Generate detailed intervention recommendations"""
        
        interventions = []
        
        # Handle InterventionPlan dataclass
        if hasattr(intervention_plan, 'interventions'):
            # It's an InterventionPlan object
            for intervention in intervention_plan.interventions:
                # Determine target type based on intervention's target_ids
                if hasattr(intervention, 'target_ids') and intervention.target_ids:
                    for target_id in intervention.target_ids:
                        target_type = 'building' if 'building' in str(target_id).lower() else 'cluster'
                        interventions.append(self._format_intervention(
                            intervention, target_type, target_id, building_data, gnn_outputs
                        ))
                else:
                    # Default to cluster level if no specific targets
                    interventions.append(self._format_intervention(
                        intervention, 'cluster', 0, building_data, gnn_outputs
                    ))
        # Handle legacy dictionary format
        elif isinstance(intervention_plan, dict):
            # Building-level interventions
            if 'building_interventions' in intervention_plan:
                for building_id, building_interventions in intervention_plan['building_interventions'].items():
                    for intervention in building_interventions:
                        interventions.append(self._format_intervention(
                            intervention, 'building', building_id, building_data, gnn_outputs
                        ))
            
            # Cluster-level interventions
            if 'cluster_interventions' in intervention_plan:
                for cluster_id, cluster_interventions in intervention_plan['cluster_interventions'].items():
                    for intervention in cluster_interventions:
                        interventions.append(self._format_intervention(
                            intervention, 'cluster', cluster_id, building_data, gnn_outputs
                        ))
        
        # Add default recommendations if no plan provided
        if not interventions:
            for cluster_id in np.unique(clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                
                # Recommend solar for clusters with high potential
                solar_potential = sum(building_data.get('solar_potentials', [10]*len(cluster_indices))[i] 
                                    for i in cluster_indices if i < len(building_data.get('solar_potentials', [])))
                
                if solar_potential > 20:  # kW threshold
                    interventions.append({
                        'target_type': 'cluster',
                        'target_id': f'CLUSTER_{cluster_id:02d}',
                        'intervention_type': 'solar',
                        'size_kw': solar_potential * 0.8,  # 80% of potential
                        'cost_eur': solar_potential * 0.8 * self.costs['solar'],
                        'annual_generation_kwh': solar_potential * 0.8 * 1200,  # 1200 hours/year
                        'annual_savings_eur': solar_potential * 0.8 * 1200 * 0.15,  # €0.15/kWh
                        'co2_reduction_tons': solar_potential * 0.8 * 1200 * self.grid_emission_factor / 1000,
                        'payback_years': (solar_potential * 0.8 * self.costs['solar']) / (solar_potential * 0.8 * 1200 * 0.15),
                        'priority': gnn_outputs.get('cluster_intervention_priority', {}).get(cluster_id, 0.5),
                        'network_benefit': gnn_outputs.get('cluster_network_benefit', {}).get(cluster_id, 0.5),
                        'cascade_potential': gnn_outputs.get('cluster_cascade_potential', {}).get(cluster_id, 0.5),
                        'recommendation': 'High priority - significant solar potential'
                    })
                
                # Recommend battery for clusters with high peak
                peak_demand = sum(building_data.get('peak_demands', [5]*len(cluster_indices))[i] 
                                 for i in cluster_indices if i < len(building_data.get('peak_demands', [])))
                
                if peak_demand > 30:  # kW threshold
                    battery_size = peak_demand * 0.3 * 2  # 30% of peak for 2 hours
                    interventions.append({
                        'target_type': 'cluster',
                        'target_id': f'CLUSTER_{cluster_id:02d}',
                        'intervention_type': 'battery',
                        'size_kwh': battery_size,
                        'cost_eur': battery_size * self.costs['battery'],
                        'peak_reduction_kw': peak_demand * 0.3,
                        'annual_savings_eur': peak_demand * 0.3 * 200,  # €200/kW/year demand charge
                        'co2_reduction_tons': 0,  # No direct CO2 reduction
                        'payback_years': (battery_size * self.costs['battery']) / (peak_demand * 0.3 * 200),
                        'priority': gnn_outputs.get('cluster_intervention_priority', {}).get(cluster_id, 0.5),
                        'network_benefit': gnn_outputs.get('cluster_network_benefit', {}).get(cluster_id, 0.7),
                        'cascade_potential': gnn_outputs.get('cluster_cascade_potential', {}).get(cluster_id, 0.3),
                        'recommendation': 'Recommended for peak shaving'
                    })
        
        return pd.DataFrame(interventions)
    
    def _generate_network_report(
        self,
        clusters: np.ndarray,
        building_data: Dict,
        gnn_outputs: Dict
    ) -> pd.DataFrame:
        """Generate network impact analysis report"""
        
        network_impacts = []
        
        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            
            impact = {
                'cluster_id': int(cluster_id),
                'num_buildings': len(cluster_indices),
                
                # Network metrics - handle empty lists and out-of-bounds indices
                'avg_centrality': (lambda x: np.mean(x) if len(x) > 0 else 0.5)(
                    [gnn_outputs.get('network_centrality_score', [0.5]*len(clusters))[i] 
                     for i in cluster_indices if i < len(gnn_outputs.get('network_centrality_score', [0.5]*len(clusters)))]
                ),
                'max_centrality': (lambda x: np.max(x) if len(x) > 0 else 0.5)(
                    [gnn_outputs.get('network_centrality_score', [0.5]*len(clusters))[i] 
                     for i in cluster_indices if i < len(gnn_outputs.get('network_centrality_score', [0.5]*len(clusters)))]
                ),
                
                # Cascade potential
                'cascade_1_hop': gnn_outputs.get('cascade_metrics', {}).get(f'cluster_{cluster_id}_1hop', 0),
                'cascade_2_hop': gnn_outputs.get('cascade_metrics', {}).get(f'cluster_{cluster_id}_2hop', 0),
                'cascade_3_hop': gnn_outputs.get('cascade_metrics', {}).get(f'cluster_{cluster_id}_3hop', 0),
                
                # Network stress
                'transformer_loading': gnn_outputs.get('transformer_loading', {}).get(cluster_id, 0.5),
                'line_utilization': gnn_outputs.get('line_utilization', {}).get(cluster_id, 0.5),
                'voltage_deviation': gnn_outputs.get('voltage_deviation', {}).get(cluster_id, 0.0),
                
                # Multi-hop effects
                'correlation_1_hop': gnn_outputs.get('multi_hop_correlation', {}).get(f'{cluster_id}_1', 0),
                'correlation_2_hop': gnn_outputs.get('multi_hop_correlation', {}).get(f'{cluster_id}_2', 0),
                'correlation_3_hop': gnn_outputs.get('multi_hop_correlation', {}).get(f'{cluster_id}_3', 0),
            }
            
            # Classify network role
            if impact['avg_centrality'] > 0.7:
                impact['network_role'] = 'Hub'
            elif impact['avg_centrality'] > 0.4:
                impact['network_role'] = 'Connector'
            else:
                impact['network_role'] = 'Peripheral'
                
            # Intervention impact score
            impact['intervention_impact'] = (
                impact['avg_centrality'] * 0.3 +
                impact['cascade_3_hop'] * 0.3 +
                (1 - impact['transformer_loading']) * 0.2 +
                (1 - impact['line_utilization']) * 0.2
            )
            
            network_impacts.append(impact)
            
        return pd.DataFrame(network_impacts)
    
    def _generate_economic_report(
        self,
        interventions_df: pd.DataFrame,
        clusters_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate economic analysis report"""
        
        economic_summary = []
        
        if not interventions_df.empty:
            # Group by cluster
            for cluster_id in clusters_df['cluster_id'].unique():
                cluster_interventions = interventions_df[
                    interventions_df['target_id'].str.contains(f'CLUSTER_{cluster_id:02d}')
                ]
                
                cluster_info = clusters_df[clusters_df['cluster_id'] == cluster_id].iloc[0]
                
                summary = {
                    'cluster_id': cluster_id,
                    'num_buildings': cluster_info['num_buildings'],
                    
                    # Investment summary
                    'total_investment_eur': cluster_interventions['cost_eur'].sum() if 'cost_eur' in cluster_interventions else 0,
                    'solar_investment_eur': cluster_interventions[cluster_interventions['intervention_type'] == 'solar']['cost_eur'].sum() if not cluster_interventions.empty else 0,
                    'battery_investment_eur': cluster_interventions[cluster_interventions['intervention_type'] == 'battery']['cost_eur'].sum() if not cluster_interventions.empty else 0,
                    
                    # Returns summary
                    'annual_savings_eur': cluster_interventions['annual_savings_eur'].sum() if 'annual_savings_eur' in cluster_interventions else 0,
                    'annual_co2_reduction_tons': cluster_interventions['co2_reduction_tons'].sum() if 'co2_reduction_tons' in cluster_interventions else 0,
                    
                    # Financial metrics
                    'avg_payback_years': cluster_interventions['payback_years'].mean() if 'payback_years' in cluster_interventions and not cluster_interventions.empty else 0,
                    'roi_10_years': 0,  # Will calculate below
                    'npv_20_years': 0,  # Will calculate below
                    
                    # Per building metrics
                    'investment_per_building_eur': 0,
                    'savings_per_building_eur': 0,
                }
                
                # Calculate per-building metrics
                if summary['num_buildings'] > 0:
                    summary['investment_per_building_eur'] = summary['total_investment_eur'] / summary['num_buildings']
                    summary['savings_per_building_eur'] = summary['annual_savings_eur'] / summary['num_buildings']
                
                # Calculate ROI and NPV
                if summary['total_investment_eur'] > 0:
                    # 10-year ROI
                    total_returns_10y = summary['annual_savings_eur'] * 10
                    summary['roi_10_years'] = (total_returns_10y - summary['total_investment_eur']) / summary['total_investment_eur']
                    
                    # 20-year NPV with 5% discount rate
                    discount_rate = 0.05
                    npv = -summary['total_investment_eur']
                    for year in range(1, 21):
                        npv += summary['annual_savings_eur'] / ((1 + discount_rate) ** year)
                    summary['npv_20_years'] = npv
                
                # Categorize economic viability
                if summary['avg_payback_years'] > 0:
                    if summary['avg_payback_years'] < 5:
                        summary['economic_viability'] = 'Excellent'
                    elif summary['avg_payback_years'] < 8:
                        summary['economic_viability'] = 'Good'
                    elif summary['avg_payback_years'] < 12:
                        summary['economic_viability'] = 'Moderate'
                    else:
                        summary['economic_viability'] = 'Marginal'
                else:
                    summary['economic_viability'] = 'Unknown'
                    
                economic_summary.append(summary)
                
        return pd.DataFrame(economic_summary)
    
    def _format_intervention(
        self,
        intervention,  # Can be Intervention dataclass or dict
        target_type: str,
        target_id: str,
        building_data: Dict,
        gnn_outputs: Dict
    ) -> Dict:
        """Format intervention data for report"""
        
        # Check if it's a dataclass (Intervention object)
        if hasattr(intervention, '__dataclass_fields__'):
            # Extract intervention type
            if hasattr(intervention.type, 'value'):
                intervention_type = intervention.type.value
            elif hasattr(intervention.type, 'name'):
                intervention_type = intervention.type.name.lower()
            else:
                intervention_type = str(intervention.type).lower()
            
            # Extract from dataclass attributes
            formatted = {
                'target_type': target_type,
                'target_id': f'{target_type.upper()}_{target_id:04d}' if isinstance(target_id, int) else str(target_id),
                'intervention_type': intervention_type,
                'size_kw': getattr(intervention, 'capacity_kw', 0),
                'cost_eur': getattr(intervention, 'estimated_cost', 0),
                'annual_savings_eur': intervention.expected_impact.get('annual_savings', 0) if hasattr(intervention, 'expected_impact') else 0,
                'co2_reduction_tons': intervention.expected_impact.get('carbon_reduction', 0) if hasattr(intervention, 'expected_impact') else 0,
                'payback_years': getattr(intervention, 'payback_period', 0),
                'priority': getattr(intervention, 'priority_score', 0.5),
                'network_benefit': getattr(intervention, 'network_benefit', 0.5),
                'cascade_potential': getattr(intervention, 'cascade_potential', 0.5),
                'recommendation': intervention.co_benefits[0] if hasattr(intervention, 'co_benefits') and intervention.co_benefits else 'Consider implementation'
            }
        else:
            # Handle as dictionary (legacy format)
            formatted = {
                'target_type': target_type,
                'target_id': f'{target_type.upper()}_{target_id:04d}' if isinstance(target_id, int) else target_id,
                'intervention_type': intervention.get('type', 'unknown'),
                'size_kw': intervention.get('size', 0),
                'cost_eur': intervention.get('cost', 0),
                'annual_savings_eur': intervention.get('annual_savings', 0),
                'co2_reduction_tons': intervention.get('co2_reduction', 0),
                'payback_years': intervention.get('payback', 0),
                'priority': intervention.get('priority', 0.5),
                'network_benefit': intervention.get('network_benefit', 0.5),
                'cascade_potential': intervention.get('cascade_potential', 0.5),
                'recommendation': intervention.get('recommendation', 'Consider implementation')
            }
        
        # Add additional metrics based on intervention type
        if formatted['intervention_type'] == 'solar':
            formatted['annual_generation_kwh'] = formatted['size_kw'] * 1200  # Typical hours
            formatted['self_sufficiency_increase'] = formatted['annual_generation_kwh'] / 10000  # Estimate
            
        elif formatted['intervention_type'] == 'battery':
            formatted['size_kwh'] = formatted['size_kw']  # For battery, size is in kWh
            formatted['peak_reduction_kw'] = formatted['size_kwh'] / 2  # 2-hour discharge
            formatted['cycles_per_year'] = 250  # Daily cycling
            
        elif formatted['intervention_type'] == 'retrofit':
            if hasattr(intervention, '__dataclass_fields__'):
                formatted['energy_reduction_pct'] = intervention.expected_impact.get('energy_reduction', 20) if hasattr(intervention, 'expected_impact') else 20
            else:
                formatted['energy_reduction_pct'] = intervention.get('reduction', 20)
            formatted['comfort_improvement'] = 'High'
            
        elif formatted['intervention_type'] == 'heat_pump':
            formatted['cop'] = 3.5  # Coefficient of performance
            formatted['heating_capacity_kw'] = formatted['size_kw']
            
        return formatted
    
    def _decode_energy_label(self, label_code: int) -> str:
        """Convert energy label code to letter"""
        labels = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F', 7: 'G'}
        return labels.get(int(label_code), 'D')
    
    def _generate_summary_visualization(
        self,
        reports: Dict[str, pd.DataFrame],
        save_path: Path
    ):
        """Generate summary visualization of all reports"""
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        
        # 1. Cluster sizes
        if 'clusters' in reports and not reports['clusters'].empty:
            ax = axes[0, 0]
            reports['clusters']['num_buildings'].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Buildings per Cluster')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Number of Buildings')
        
        # 2. Energy performance
        if 'clusters' in reports and not reports['clusters'].empty:
            ax = axes[0, 1]
            metrics = ['self_sufficiency', 'complementarity_score', 'peak_reduction']
            if all(m in reports['clusters'].columns for m in metrics):
                reports['clusters'][metrics].plot(kind='box', ax=ax)
                ax.set_title('Cluster Performance Metrics')
                ax.set_ylabel('Score')
        
        # 3. Temporal patterns
        if 'temporal' in reports and not reports['temporal'].empty:
            ax = axes[0, 2]
            hourly = reports['temporal'].groupby('hour')['net_demand_kw'].mean()
            hourly.plot(ax=ax, color='orange', linewidth=2)
            ax.set_title('Average Daily Demand Profile')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Net Demand (kW)')
            ax.grid(True, alpha=0.3)
        
        # 4. Intervention distribution
        if 'interventions' in reports and not reports['interventions'].empty:
            ax = axes[1, 0]
            intervention_counts = reports['interventions']['intervention_type'].value_counts()
            intervention_counts.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title('Intervention Type Distribution')
        
        # 5. Investment requirements
        if 'economics' in reports and not reports['economics'].empty:
            ax = axes[1, 1]
            reports['economics']['total_investment_eur'].plot(kind='bar', ax=ax, color='green')
            ax.set_title('Investment per Cluster')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Investment (€)')
        
        # 6. Payback periods
        if 'economics' in reports and not reports['economics'].empty:
            ax = axes[1, 2]
            reports['economics']['avg_payback_years'].plot(kind='bar', ax=ax, color='coral')
            ax.set_title('Average Payback Period')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Years')
            ax.axhline(y=8, color='r', linestyle='--', label='Target')
        
        # 7. Network centrality
        if 'network' in reports and not reports['network'].empty:
            ax = axes[2, 0]
            reports['network'][['avg_centrality', 'max_centrality']].plot(kind='bar', ax=ax)
            ax.set_title('Network Centrality by Cluster')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('Centrality Score')
        
        # 8. CO2 reduction
        if 'economics' in reports and not reports['economics'].empty:
            ax = axes[2, 1]
            reports['economics']['annual_co2_reduction_tons'].plot(kind='bar', ax=ax, color='darkgreen')
            ax.set_title('Annual CO2 Reduction')
            ax.set_xlabel('Cluster ID')
            ax.set_ylabel('CO2 Reduction (tons/year)')
        
        # 9. ROI comparison
        if 'economics' in reports and not reports['economics'].empty:
            ax = axes[2, 2]
            if 'roi_10_years' in reports['economics'].columns:
                reports['economics']['roi_10_years'].plot(kind='bar', ax=ax, color='purple')
                ax.set_title('10-Year Return on Investment')
                ax.set_xlabel('Cluster ID')
                ax.set_ylabel('ROI (%)')
                ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        plt.suptitle('Energy Community Analysis Summary', fontsize=16, y=1.02)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Summary visualization saved to: {save_path}")
        
    def generate_markdown_summary(
        self,
        reports: Dict[str, pd.DataFrame],
        save_path: Optional[Path] = None
    ) -> str:
        """Generate markdown summary of the analysis"""
        
        summary = []
        summary.append("# Energy Community Analysis Report")
        summary.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Executive Summary
        summary.append("## Executive Summary\n")
        
        if 'clusters' in reports and not reports['clusters'].empty:
            n_clusters = len(reports['clusters'])
            n_buildings = reports['clusters']['num_buildings'].sum()
            avg_self_suff = reports['clusters']['self_sufficiency'].mean()
            avg_peak_red = reports['clusters']['peak_reduction'].mean()
            
            summary.append(f"- **Total Clusters**: {n_clusters}")
            summary.append(f"- **Total Buildings**: {n_buildings}")
            summary.append(f"- **Average Self-Sufficiency**: {avg_self_suff:.1%}")
            summary.append(f"- **Average Peak Reduction**: {avg_peak_red:.1%}\n")
        
        # Investment Summary
        summary.append("## Investment Summary\n")
        
        if 'economics' in reports and not reports['economics'].empty:
            total_investment = reports['economics']['total_investment_eur'].sum()
            total_savings = reports['economics']['annual_savings_eur'].sum()
            total_co2 = reports['economics']['annual_co2_reduction_tons'].sum()
            avg_payback = reports['economics']['avg_payback_years'].mean()
            
            summary.append(f"- **Total Investment Required**: €{total_investment:,.0f}")
            summary.append(f"- **Annual Savings**: €{total_savings:,.0f}")
            summary.append(f"- **Annual CO2 Reduction**: {total_co2:.1f} tons")
            summary.append(f"- **Average Payback Period**: {avg_payback:.1f} years\n")
        
        # Top Performing Clusters
        summary.append("## Top Performing Clusters\n")
        
        if 'clusters' in reports and not reports['clusters'].empty:
            top_clusters = reports['clusters'].nlargest(3, 'self_sufficiency')
            
            summary.append("| Cluster | Buildings | Self-Sufficiency | Peak Reduction | Complementarity |")
            summary.append("|---------|-----------|------------------|----------------|-----------------|")
            
            for _, cluster in top_clusters.iterrows():
                summary.append(f"| {cluster['cluster_id']} | {cluster['num_buildings']} | "
                             f"{cluster['self_sufficiency']:.1%} | {cluster['peak_reduction']:.1%} | "
                             f"{cluster['complementarity_score']:.2f} |")
            summary.append("")
        
        # Priority Interventions
        summary.append("## Priority Interventions\n")
        
        if 'interventions' in reports and not reports['interventions'].empty:
            top_interventions = reports['interventions'].nlargest(5, 'priority')
            
            summary.append("| Target | Type | Size | Cost | Payback | Priority |")
            summary.append("|--------|------|------|------|---------|----------|")
            
            for _, intervention in top_interventions.iterrows():
                size_str = f"{intervention.get('size_kw', 0):.0f} kW" if 'size_kw' in intervention else "N/A"
                summary.append(f"| {intervention['target_id']} | {intervention['intervention_type']} | "
                             f"{size_str} | €{intervention.get('cost_eur', 0):,.0f} | "
                             f"{intervention.get('payback_years', 0):.1f} years | {intervention['priority']:.2f} |")
            summary.append("")
        
        # Network Analysis
        summary.append("## Network Impact Analysis\n")
        
        if 'network' in reports and not reports['network'].empty:
            hub_clusters = reports['network'][reports['network']['network_role'] == 'Hub']
            
            if not hub_clusters.empty:
                summary.append(f"- **Hub Clusters**: {len(hub_clusters)}")
                summary.append(f"- **Average Cascade Potential (3-hop)**: {reports['network']['cascade_3_hop'].mean():.2f}")
                summary.append(f"- **Average Transformer Loading**: {reports['network']['transformer_loading'].mean():.1%}\n")
        
        # Recommendations
        summary.append("## Key Recommendations\n")
        
        summary.append("1. **Immediate Actions**:")
        summary.append("   - Implement solar PV in high-priority clusters")
        summary.append("   - Deploy battery storage for peak shaving")
        summary.append("   - Form energy communities in high-complementarity clusters\n")
        
        summary.append("2. **Medium-term Actions**:")
        summary.append("   - Retrofit buildings with poor energy labels")
        summary.append("   - Install smart meters for real-time monitoring")
        summary.append("   - Develop local energy trading mechanisms\n")
        
        summary.append("3. **Long-term Strategy**:")
        summary.append("   - Expand renewable generation capacity")
        summary.append("   - Implement district-level energy management")
        summary.append("   - Plan for EV charging infrastructure\n")
        
        markdown_text = '\n'.join(summary)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(markdown_text)
            print(f"Markdown summary saved to: {save_path}")
            
        return markdown_text


def generate_sample_report():
    """Generate a sample report with mock data for testing"""
    
    # Create sample data
    n_buildings = 50
    n_clusters = 5
    
    # Generate cluster assignments
    clusters = np.random.randint(0, n_clusters, n_buildings)
    
    # Generate building data
    building_data = {
        'building_ids': list(range(n_buildings)),
        'lv_group': 'LV_GROUP_0004',
        'areas': np.random.uniform(80, 200, n_buildings),
        'heights': np.random.uniform(8, 15, n_buildings),
        'ages': np.random.uniform(5, 50, n_buildings),
        'energy_labels': np.random.randint(1, 8, n_buildings),
        'peak_demands': np.random.uniform(3, 15, n_buildings),
        'base_loads': np.random.uniform(0.5, 3, n_buildings),
        'annual_consumption': np.random.uniform(3000, 15000, n_buildings),
        'roof_areas': np.random.uniform(40, 150, n_buildings),
        'solar_potentials': np.random.uniform(5, 25, n_buildings),
        'solar_suitability': np.random.choice(['high', 'medium', 'low'], n_buildings),
        'has_solar': np.random.choice([True, False], n_buildings, p=[0.1, 0.9]),
        'has_battery': np.random.choice([True, False], n_buildings, p=[0.05, 0.95]),
        'has_heat_pump': np.random.choice([True, False], n_buildings, p=[0.15, 0.85]),
        'consumption_profiles': np.random.uniform(1, 10, (n_buildings, 96)),
        'generation_profiles': np.random.uniform(0, 5, (n_buildings, 96))
    }
    
    # Generate GNN outputs
    gnn_outputs = {
        'cluster_probs': np.random.dirichlet(np.ones(n_clusters), n_buildings),
        'network_centrality_score': np.random.uniform(0.2, 0.8, n_buildings),
        'intervention_priority': np.random.uniform(0.3, 0.9, n_buildings),
        'cluster_self_sufficiency': {i: np.random.uniform(0, 0.3) for i in range(n_clusters)},
        'cluster_complementarity': {i: np.random.uniform(-0.2, 0.1) for i in range(n_clusters)},
        'cluster_peak_reduction': {i: np.random.uniform(0.1, 0.4) for i in range(n_clusters)},
        'cluster_network_stress': {i: np.random.uniform(0.4, 0.8) for i in range(n_clusters)}
    }
    
    # Generate intervention plan
    intervention_plan = {
        'cluster_interventions': {
            i: [
                {
                    'type': 'solar',
                    'size': np.random.uniform(50, 200),
                    'cost': np.random.uniform(50000, 200000),
                    'annual_savings': np.random.uniform(5000, 20000),
                    'co2_reduction': np.random.uniform(10, 50),
                    'payback': np.random.uniform(5, 15),
                    'priority': np.random.uniform(0.5, 0.9),
                    'network_benefit': np.random.uniform(0.4, 0.8),
                    'cascade_potential': np.random.uniform(0.3, 0.7)
                }
            ] for i in range(n_clusters)
        }
    }
    
    # Create reporter and generate report
    config = {
        'reporting': {
            'save_dir': 'reports',
            'formats': ['csv', 'excel', 'markdown']
        }
    }
    
    reporter = ComprehensiveReporter(config)
    reports = reporter.generate_full_report(
        clusters,
        building_data,
        gnn_outputs,
        intervention_plan,
        save_dir="reports"
    )
    
    # Generate markdown summary
    reporter.generate_markdown_summary(
        reports,
        Path("reports") / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    )
    
    return reports


if __name__ == "__main__":
    print("Generating sample comprehensive report...")
    reports = generate_sample_report()
    print("\nReport generation complete!")
    print(f"Generated {len(reports)} report sections:")
    for name, df in reports.items():
        print(f"  - {name}: {len(df)} rows")