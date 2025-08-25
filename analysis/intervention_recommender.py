"""
Intervention Recommender for Energy GNN
Converts discovered patterns and gaps into actionable intervention recommendations
Uses GNN intelligence for network-aware planning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
from scipy.optimize import linprog
import networkx as nx


class InterventionType(Enum):
    """Enumeration of intervention types"""
    SOLAR_PV = "solar_pv"
    BATTERY_STORAGE = "battery_storage"
    BUILDING_RETROFIT = "building_retrofit"
    SMART_METER = "smart_meter"
    HEAT_PUMP = "heat_pump"
    EV_CHARGER = "ev_charger"
    DEMAND_RESPONSE = "demand_response"
    GRID_UPGRADE = "grid_upgrade"


@dataclass
class Intervention:
    """Data class for intervention recommendations"""
    intervention_id: str
    type: InterventionType
    location: Any  # Building ID, cluster ID, or transformer ID
    size: float  # kW for power, kWh for energy, units vary by type
    specifications: Dict[str, Any]
    estimated_cost: float
    expected_impact: Dict[str, float]
    network_effects: Dict[str, float]
    priority_score: float
    implementation_timeline: str
    dependencies: List[str] = field(default_factory=list)
    co_benefits: List[str] = field(default_factory=list)


@dataclass
class InterventionPlan:
    """Complete intervention plan for a district"""
    plan_id: str
    interventions: List[Intervention]
    total_cost: float
    expected_benefits: Dict[str, float]
    implementation_phases: List[Dict[str, Any]]
    risk_assessment: Dict[str, Any]
    monitoring_metrics: List[str]


class InterventionRecommender:
    """
    Rule-based intervention recommender enhanced with GNN intelligence
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize recommender
        
        Args:
            config: Configuration with costs, constraints, etc.
        """
        self.config = config
        
        # Cost parameters ($/kW or $/kWh)
        self.costs = {
            InterventionType.SOLAR_PV: config.get('solar_cost', 1000),  # $/kW
            InterventionType.BATTERY_STORAGE: config.get('battery_cost', 500),  # $/kWh
            InterventionType.BUILDING_RETROFIT: config.get('retrofit_cost', 50000),  # $/building
            InterventionType.SMART_METER: config.get('smart_meter_cost', 200),  # $/unit
            InterventionType.HEAT_PUMP: config.get('heat_pump_cost', 5000),  # $/unit
            InterventionType.EV_CHARGER: config.get('ev_charger_cost', 2000),  # $/unit
            InterventionType.DEMAND_RESPONSE: config.get('dr_cost', 100),  # $/kW enrolled
            InterventionType.GRID_UPGRADE: config.get('grid_upgrade_cost', 100000)  # $/transformer
        }
        
        # Technical parameters
        self.solar_capacity_factor = config.get('solar_cf', 0.2)
        self.battery_efficiency = config.get('battery_eff', 0.9)
        self.retrofit_reduction = config.get('retrofit_reduction', 0.3)
        self.dr_participation = config.get('dr_participation', 0.15)
        
        # Constraints
        self.max_roof_utilization = config.get('max_roof_util', 0.8)
        self.min_battery_size = config.get('min_battery', 10)  # kWh
        self.max_battery_size = config.get('max_battery', 500)  # kWh
        
    def recommend_interventions(
        self,
        analysis_results: Dict[str, Any],
        gnn_outputs: Dict[str, Any],
        building_data: pd.DataFrame,
        network_topology: Dict[str, Any],
        budget_constraint: Optional[float] = None
    ) -> InterventionPlan:
        """
        Generate comprehensive intervention recommendations
        
        Args:
            analysis_results: Results from PatternAnalyzer
            gnn_outputs: GNN model outputs (importance scores, etc.)
            building_data: Building characteristics
            network_topology: Grid network structure
            budget_constraint: Optional budget limit
            
        Returns:
            Complete intervention plan
        """
        interventions = []
        
        # Extract GNN intelligence
        network_importance = self._extract_network_importance(gnn_outputs)
        complementarity_matrix = gnn_outputs.get('clustering_complementarity_matrix')
        cluster_assignments = gnn_outputs.get('clustering_cluster_assignments')
        
        # 1. Solar PV recommendations
        solar_interventions = self._recommend_solar(
            analysis_results['energy_gaps'],
            building_data,
            network_importance,
            cluster_assignments
        )
        interventions.extend(solar_interventions)
        
        # 2. Battery storage recommendations
        storage_interventions = self._recommend_storage(
            analysis_results['energy_gaps'],
            analysis_results['temporal_patterns'],
            network_importance,
            cluster_assignments,
            complementarity_matrix
        )
        interventions.extend(storage_interventions)
        
        # 3. Building retrofit recommendations
        retrofit_interventions = self._recommend_retrofits(
            building_data,
            analysis_results['cluster_metrics'],
            network_importance
        )
        interventions.extend(retrofit_interventions)
        
        # 4. Demand response recommendations
        dr_interventions = self._recommend_demand_response(
            analysis_results['temporal_patterns'],
            cluster_assignments,
            network_importance
        )
        interventions.extend(dr_interventions)
        
        # 5. Grid upgrade recommendations
        grid_interventions = self._recommend_grid_upgrades(
            analysis_results['network_bottlenecks'],
            network_topology
        )
        interventions.extend(grid_interventions)
        
        # 6. Smart technology recommendations
        smart_interventions = self._recommend_smart_tech(
            building_data,
            cluster_assignments,
            complementarity_matrix
        )
        interventions.extend(smart_interventions)
        
        # Calculate network effects using GNN intelligence
        interventions = self._calculate_network_effects(
            interventions,
            network_importance,
            network_topology
        )
        
        # Optimize intervention mix if budget constrained
        if budget_constraint:
            interventions = self._optimize_intervention_mix(
                interventions,
                budget_constraint
            )
        
        # Rank interventions by impact
        interventions = self._rank_interventions(interventions)
        
        # Create implementation phases
        phases = self._create_implementation_phases(interventions)
        
        # Generate complete plan
        plan = self._create_intervention_plan(
            interventions,
            phases,
            analysis_results
        )
        
        return plan
    
    def _extract_network_importance(self, gnn_outputs: Dict) -> Dict[str, float]:
        """Extract network importance scores from GNN outputs"""
        importance = {}
        
        if 'network_centrality_score' in gnn_outputs:
            centrality = gnn_outputs['network_centrality_score']
            # Handle both tensor and numpy array cases
            if hasattr(centrality, 'cpu'):
                centrality = centrality.cpu().numpy()
            elif not isinstance(centrality, np.ndarray):
                centrality = np.array(centrality)
            
            for i, score in enumerate(centrality):
                importance[f'building_{i}'] = {
                    'centrality': score,
                    'cascade_potential': gnn_outputs.get('network_cascade_potential', [0]*len(centrality))[i],
                    'intervention_priority': gnn_outputs.get('network_intervention_priority', [0]*len(centrality))[i]
                }
        
        return importance
    
    def _recommend_solar(
        self,
        energy_gaps: List[Any],
        building_data: pd.DataFrame,
        network_importance: Dict,
        cluster_assignments: np.ndarray
    ) -> List[Intervention]:
        """
        Recommend solar PV installations
        
        Uses GNN intelligence to prioritize network-critical locations
        """
        solar_interventions = []
        
        # Find generation gaps during solar hours
        solar_gaps = [g for g in energy_gaps if g.gap_type == 'generation']
        
        for gap in solar_gaps:
            cluster_id = gap.cluster_id
            required_generation = gap.magnitude  # kW
            
            # Find buildings in cluster with solar potential
            if cluster_assignments is not None and hasattr(cluster_assignments, '__len__'):
                cluster_mask = cluster_assignments == cluster_id
                # Create boolean index aligned with DataFrame
                if hasattr(cluster_mask, '__len__') and len(cluster_mask) == len(building_data):
                    cluster_buildings = building_data[cluster_mask]
                elif hasattr(cluster_mask, '__len__'):
                    # If sizes don't match, use positional indexing
                    cluster_indices = np.where(cluster_mask)[0]
                    if len(cluster_indices) > 0 and max(cluster_indices) < len(building_data):
                        cluster_buildings = building_data.iloc[cluster_indices]
                    else:
                        cluster_buildings = pd.DataFrame()  # Empty DataFrame if no valid indices
                else:
                    # cluster_mask is a scalar, use all buildings
                    cluster_buildings = building_data
            else:
                # No cluster assignments, use all buildings
                cluster_buildings = building_data
            
            # Score buildings by solar potential AND network importance
            building_scores = []
            for idx, building in cluster_buildings.iterrows():
                # Physical suitability
                roof_area = building.get('roof_area', 100)
                orientation_factor = 1.0 if building.get('orientation', 0) < 45 else 0.8
                shading_factor = building.get('shading_factor', 1.0)
                
                # Maximum installable capacity (10 m²/kW)
                max_capacity = roof_area * self.max_roof_utilization / 10
                
                # Network importance from GNN
                network_score = network_importance.get(
                    f'building_{idx}', {}
                ).get('centrality', 0.5)
                
                # Combined score: physical suitability * network importance
                combined_score = (
                    max_capacity * orientation_factor * shading_factor * 
                    (1 + network_score)  # Boost by network importance
                )
                
                building_scores.append({
                    'building_id': idx,
                    'max_capacity': max_capacity,
                    'score': combined_score,
                    'network_importance': network_score
                })
            
            # Sort by score and select top buildings
            building_scores.sort(key=lambda x: x['score'], reverse=True)
            
            # Allocate solar to best buildings
            remaining_capacity = required_generation / self.solar_capacity_factor
            
            for building in building_scores:
                if remaining_capacity <= 0:
                    break
                
                # Size installation
                install_capacity = min(
                    building['max_capacity'],
                    remaining_capacity
                )
                
                if install_capacity >= 5:  # Minimum 5 kW installation
                    intervention = Intervention(
                        intervention_id=f"solar_{building['building_id']}_{cluster_id}",
                        type=InterventionType.SOLAR_PV,
                        location=building['building_id'],
                        size=install_capacity,
                        specifications={
                            'panel_type': 'monocrystalline',
                            'inverter_size': install_capacity,
                            'tilt_angle': 35,
                            'azimuth': building_data.loc[building['building_id']].get('orientation', 180),
                            'estimated_generation': install_capacity * self.solar_capacity_factor * 8760  # kWh/year
                        },
                        estimated_cost=install_capacity * self.costs[InterventionType.SOLAR_PV],
                        expected_impact={
                            'generation_increase': install_capacity * self.solar_capacity_factor,
                            'peak_reduction': install_capacity * 0.3,  # Conservative estimate
                            'carbon_reduction': install_capacity * self.solar_capacity_factor * 8760 * 0.5 / 1000  # tons/year
                        },
                        network_effects={
                            'network_importance': building['network_importance'],
                            'cascade_factor': 1.5 if building['network_importance'] > 0.7 else 1.0
                        },
                        priority_score=0,  # Will be calculated later
                        implementation_timeline='3-6 months',
                        co_benefits=['reduced_grid_stress', 'property_value_increase']
                    )
                    
                    solar_interventions.append(intervention)
                    remaining_capacity -= install_capacity
        
        return solar_interventions
    
    def _recommend_storage(
        self,
        energy_gaps: List[Any],
        temporal_patterns: Dict,
        network_importance: Dict,
        cluster_assignments: np.ndarray,
        complementarity_matrix: Optional[np.ndarray]
    ) -> List[Intervention]:
        """
        Recommend battery storage systems
        
        Uses GNN's complementarity patterns to optimize placement
        """
        storage_interventions = []
        
        # Find storage gaps (evening peaks, time-shifting needs)
        storage_gaps = [g for g in energy_gaps if g.gap_type == 'storage']
        
        for gap in storage_gaps:
            cluster_id = gap.cluster_id
            peak_power = gap.magnitude  # kW
            duration = gap.duration  # hours
            
            # Calculate required storage capacity
            energy_capacity = peak_power * duration  # kWh
            energy_capacity = np.clip(
                energy_capacity,
                self.min_battery_size,
                self.max_battery_size
            )
            
            # Find optimal location using GNN intelligence
            if complementarity_matrix is not None and complementarity_matrix.ndim == 2:
                # Find building with highest complementarity to others
                if cluster_assignments is not None and hasattr(cluster_assignments, '__len__'):
                    cluster_mask = cluster_assignments == cluster_id
                    if hasattr(cluster_mask, '__len__'):
                        cluster_indices = np.where(cluster_mask)[0]
                    else:
                        cluster_indices = np.array([0])  # Default to first building
                else:
                    cluster_indices = np.array([0])  # Default to first building
                
                if len(cluster_indices) > 0:
                    # Extract submatrix for cluster
                    cluster_comp = complementarity_matrix[np.ix_(cluster_indices, cluster_indices)]
                    
                    # Building with lowest average correlation (highest complementarity)
                    if cluster_comp.size > 0:
                        avg_correlation = cluster_comp.mean(axis=1)
                        best_building_idx = np.argmin(avg_correlation)
                        building_id = cluster_indices[best_building_idx]
                    else:
                        building_id = cluster_indices[0]
                else:
                    if cluster_assignments is not None and hasattr(cluster_assignments, '__len__'):
                        cluster_buildings = np.where(cluster_assignments == cluster_id)[0]
                        building_id = cluster_buildings[0] if len(cluster_buildings) > 0 else 0
                    else:
                        building_id = 0
            else:
                # Fallback to network centrality or complementarity score
                if cluster_assignments is not None and hasattr(cluster_assignments, '__len__'):
                    cluster_buildings = np.where(cluster_assignments == cluster_id)[0]
                else:
                    cluster_buildings = np.array([0])
                if len(cluster_buildings) > 0:
                    # If we have a 1D complementarity array, use it as building scores
                    if complementarity_matrix is not None and complementarity_matrix.ndim == 1:
                        cluster_comps = complementarity_matrix[cluster_buildings]
                        best_idx = np.argmax(np.abs(cluster_comps))  # Highest absolute complementarity
                        building_id = cluster_buildings[best_idx]
                    else:
                        building_id = cluster_buildings[0]  # Default to first building
                else:
                    building_id = 0
            
            # Get network importance
            network_score = network_importance.get(
                f'building_{building_id}', {}
            ).get('cascade_potential', 0.5)
            
            # Determine if shared or individual battery
            if cluster_assignments is not None and hasattr(cluster_assignments, '__len__'):
                cluster_size = (cluster_assignments == cluster_id).sum() if hasattr(cluster_assignments == cluster_id, 'sum') else 1
            else:
                cluster_size = 1
            is_shared = cluster_size > 5 and network_score > 0.6
            
            intervention = Intervention(
                intervention_id=f"battery_{building_id}_{cluster_id}",
                type=InterventionType.BATTERY_STORAGE,
                location=building_id if not is_shared else f"cluster_{cluster_id}",
                size=energy_capacity,
                specifications={
                    'power_rating': peak_power,
                    'energy_capacity': energy_capacity,
                    'chemistry': 'lithium_ion',
                    'cycles': 5000,
                    'efficiency': self.battery_efficiency,
                    'shared': is_shared,
                    'control_strategy': 'peak_shaving' if gap.timestamp in [17, 18, 19] else 'self_consumption'
                },
                estimated_cost=energy_capacity * self.costs[InterventionType.BATTERY_STORAGE],
                expected_impact={
                    'peak_reduction': peak_power,
                    'self_sufficiency_increase': 0.15,  # 15% increase
                    'grid_independence_hours': duration,
                    'annual_savings': peak_power * 100 + energy_capacity * 0.1 * 365  # Simplified
                },
                network_effects={
                    'cascade_potential': network_score,
                    'cluster_benefit_multiplier': 1.5 if is_shared else 1.0,
                    'resilience_improvement': 0.3 if is_shared else 0.1
                },
                priority_score=0,
                implementation_timeline='2-4 months',
                dependencies=[f"solar_{building_id}_{cluster_id}"] if building_id in [s.location for s in storage_interventions] else [],
                co_benefits=['grid_stability', 'backup_power', 'frequency_regulation']
            )
            
            storage_interventions.append(intervention)
        
        return storage_interventions
    
    def _recommend_retrofits(
        self,
        building_data: pd.DataFrame,
        cluster_metrics: List[Any],
        network_importance: Dict
    ) -> List[Intervention]:
        """
        Recommend building retrofits
        
        Prioritizes buildings whose reduction most improves cluster balance
        """
        retrofit_interventions = []
        
        # Identify inefficient buildings
        if 'energy_intensity' in building_data.columns:
            # Buildings with high energy intensity
            threshold = building_data['energy_intensity'].quantile(0.75)
            inefficient_buildings = building_data[
                building_data['energy_intensity'] > threshold
            ]
            
            for idx, building in inefficient_buildings.iterrows():
                # Check building age and potential for improvement
                building_age = building.get('building_age', 20)
                energy_label = building.get('energy_label', 'D')
                
                if building_age > 15 and energy_label in ['D', 'E', 'F', 'G']:
                    # Network importance for prioritization
                    network_score = network_importance.get(
                        f'building_{idx}', {}
                    ).get('intervention_priority', 0.5)
                    
                    # Estimate reduction potential
                    current_consumption = building.get('annual_consumption', 50000)  # kWh
                    reduction_potential = self.retrofit_reduction * current_consumption
                    
                    # Determine retrofit measures based on building characteristics
                    measures = []
                    if building.get('insulation_quality', 'poor') == 'poor':
                        measures.append('wall_insulation')
                    if building.get('glazing_type', 'single') == 'single':
                        measures.append('window_replacement')
                    if building.get('heating_system_age', 20) > 15:
                        measures.append('heating_system_upgrade')
                    measures.append('air_sealing')
                    
                    intervention = Intervention(
                        intervention_id=f"retrofit_{idx}",
                        type=InterventionType.BUILDING_RETROFIT,
                        location=idx,
                        size=reduction_potential / 1000,  # Convert to MWh
                        specifications={
                            'current_label': energy_label,
                            'target_label': 'B',
                            'measures': measures,
                            'estimated_reduction': self.retrofit_reduction,
                            'floor_area': building.get('floor_area', 200)
                        },
                        estimated_cost=self.costs[InterventionType.BUILDING_RETROFIT],
                        expected_impact={
                            'consumption_reduction': reduction_potential,
                            'peak_reduction': reduction_potential * 0.15 / 8760,  # kW
                            'carbon_reduction': reduction_potential * 0.5 / 1000,  # tons/year
                            'comfort_improvement': 0.3
                        },
                        network_effects={
                            'network_importance': network_score,
                            'cluster_balance_improvement': network_score * self.retrofit_reduction
                        },
                        priority_score=0,
                        implementation_timeline='6-12 months',
                        co_benefits=['comfort', 'property_value', 'health']
                    )
                    
                    retrofit_interventions.append(intervention)
        
        return retrofit_interventions
    
    def _recommend_demand_response(
        self,
        temporal_patterns: Dict,
        cluster_assignments: np.ndarray,
        network_importance: Dict
    ) -> List[Intervention]:
        """
        Recommend demand response programs
        """
        dr_interventions = []
        
        for cluster_pattern in temporal_patterns.items():
            cluster_name, patterns = cluster_pattern
            cluster_id = int(cluster_name.split('_')[1])
            
            # Check if cluster has significant peak hours
            peak_hours = patterns['peak_hours']
            if len(peak_hours) >= 2:
                # Find buildings in cluster
                cluster_buildings = np.where(cluster_assignments == cluster_id)[0]
                
                # Estimate DR potential (15% of peak)
                # This would need actual peak data in practice
                estimated_peak = len(cluster_buildings) * 10  # kW (simplified)
                dr_capacity = estimated_peak * self.dr_participation
                
                intervention = Intervention(
                    intervention_id=f"dr_cluster_{cluster_id}",
                    type=InterventionType.DEMAND_RESPONSE,
                    location=f"cluster_{cluster_id}",
                    size=dr_capacity,
                    specifications={
                        'program_type': 'time_of_use',
                        'peak_hours': peak_hours,
                        'enrolled_buildings': len(cluster_buildings),
                        'response_time': '15_minutes',
                        'events_per_year': 50
                    },
                    estimated_cost=dr_capacity * self.costs[InterventionType.DEMAND_RESPONSE],
                    expected_impact={
                        'peak_reduction': dr_capacity,
                        'annual_savings': dr_capacity * 100 * 50,  # $/year
                        'grid_flexibility': dr_capacity
                    },
                    network_effects={
                        'cluster_coordination': 0.8,
                        'grid_stability_contribution': 0.6
                    },
                    priority_score=0,
                    implementation_timeline='1-2 months',
                    dependencies=[f"smart_meter_cluster_{cluster_id}"],
                    co_benefits=['grid_reliability', 'cost_savings', 'environmental']
                )
                
                dr_interventions.append(intervention)
        
        return dr_interventions
    
    def _recommend_grid_upgrades(
        self,
        network_bottlenecks: List[Any],
        network_topology: Dict
    ) -> List[Intervention]:
        """
        Recommend grid infrastructure upgrades
        """
        grid_interventions = []
        
        # Focus on critical bottlenecks
        critical_bottlenecks = [b for b in network_bottlenecks if b.utilization > 0.9]
        
        for bottleneck in critical_bottlenecks:
            if bottleneck.type == 'transformer':
                # Transformer upgrade
                current_capacity = bottleneck.capacity
                required_capacity = bottleneck.peak_load * 1.25  # 25% safety margin
                
                intervention = Intervention(
                    intervention_id=f"upgrade_{bottleneck.location}",
                    type=InterventionType.GRID_UPGRADE,
                    location=bottleneck.location,
                    size=required_capacity,
                    specifications={
                        'current_capacity': current_capacity,
                        'new_capacity': required_capacity,
                        'type': 'transformer_upgrade',
                        'voltage_level': 'LV/MV'
                    },
                    estimated_cost=self.costs[InterventionType.GRID_UPGRADE],
                    expected_impact={
                        'capacity_increase': required_capacity - current_capacity,
                        'reliability_improvement': 0.95,
                        'loss_reduction': 0.02 * bottleneck.peak_load
                    },
                    network_effects={
                        'affected_clusters': len(bottleneck.affected_clusters),
                        'criticality': bottleneck.criticality
                    },
                    priority_score=bottleneck.criticality,  # Use criticality as initial priority
                    implementation_timeline='6-9 months',
                    co_benefits=['reliability', 'loss_reduction', 'future_proofing']
                )
                
                grid_interventions.append(intervention)
        
        return grid_interventions
    
    def _recommend_smart_tech(
        self,
        building_data: pd.DataFrame,
        cluster_assignments: np.ndarray,
        complementarity_matrix: Optional[np.ndarray]
    ) -> List[Intervention]:
        """
        Recommend smart technologies (meters, controllers, etc.)
        """
        smart_interventions = []
        
        # Identify clusters that would benefit from coordination
        unique_clusters = np.unique(cluster_assignments)
        
        for cluster_id in unique_clusters:
            cluster_buildings = np.where(cluster_assignments == cluster_id)[0]
            
            # Check if cluster has high complementarity
            if complementarity_matrix is not None:
                cluster_comp = complementarity_matrix[cluster_buildings][:, cluster_buildings]
                avg_complementarity = -cluster_comp.mean()  # Negative correlation is good
                
                if avg_complementarity > 0.3 and len(cluster_buildings) > 3:
                    # Recommend smart meters for coordination
                    intervention = Intervention(
                        intervention_id=f"smart_meter_cluster_{cluster_id}",
                        type=InterventionType.SMART_METER,
                        location=f"cluster_{cluster_id}",
                        size=len(cluster_buildings),
                        specifications={
                            'meter_type': 'advanced_metering_infrastructure',
                            'communication': 'wireless',
                            'data_frequency': '15_minutes',
                            'buildings': cluster_buildings.tolist()
                        },
                        estimated_cost=len(cluster_buildings) * self.costs[InterventionType.SMART_METER],
                        expected_impact={
                            'data_visibility': 1.0,
                            'coordination_capability': avg_complementarity,
                            'billing_accuracy': 1.0
                        },
                        network_effects={
                            'cluster_coordination': avg_complementarity,
                            'p2p_trading_enabled': avg_complementarity > 0.5
                        },
                        priority_score=0,
                        implementation_timeline='1 month',
                        co_benefits=['data_insights', 'billing_accuracy', 'theft_detection']
                    )
                    
                    smart_interventions.append(intervention)
        
        return smart_interventions
    
    def _calculate_network_effects(
        self,
        interventions: List[Intervention],
        network_importance: Dict,
        network_topology: Dict
    ) -> List[Intervention]:
        """
        Calculate cascade and network effects using GNN intelligence
        """
        # Build intervention graph
        G = nx.Graph()
        
        # Add nodes for interventions
        for intervention in interventions:
            G.add_node(intervention.intervention_id, type=intervention.type.value)
        
        # Add edges based on dependencies and network topology
        for i, int1 in enumerate(interventions):
            for j, int2 in enumerate(interventions[i+1:], i+1):
                # Check if interventions are in same cluster or connected
                if self._are_connected(int1, int2, network_topology):
                    G.add_edge(int1.intervention_id, int2.intervention_id)
        
        # Calculate cascade effects
        for intervention in interventions:
            # Get connected interventions
            if intervention.intervention_id in G:
                neighbors = list(G.neighbors(intervention.intervention_id))
                cascade_multiplier = 1 + 0.1 * len(neighbors)
                
                # Update expected impact with cascade effects
                for impact_key in intervention.expected_impact:
                    intervention.expected_impact[impact_key] *= cascade_multiplier
                
                # Update network effects
                intervention.network_effects['cascade_multiplier'] = cascade_multiplier
                intervention.network_effects['affected_interventions'] = len(neighbors)
        
        return interventions
    
    def _are_connected(
        self,
        int1: Intervention,
        int2: Intervention,
        network_topology: Dict
    ) -> bool:
        """Check if two interventions are connected in the network"""
        # Simplified: check if in same cluster or adjacent
        if hasattr(int1.location, 'startswith') and hasattr(int2.location, 'startswith'):
            if int1.location.startswith('cluster_') and int2.location.startswith('cluster_'):
                return int1.location == int2.location
        
        # Check dependencies
        return (int1.intervention_id in int2.dependencies or 
                int2.intervention_id in int1.dependencies)
    
    def _optimize_intervention_mix(
        self,
        interventions: List[Intervention],
        budget: float
    ) -> List[Intervention]:
        """
        Optimize intervention selection under budget constraint
        
        Uses linear programming to maximize impact within budget
        """
        if not interventions:
            return []
        
        n = len(interventions)
        
        # Objective: maximize total impact (simplified as sum of impacts)
        c = np.zeros(n)
        for i, intervention in enumerate(interventions):
            # Combine different impacts into single score
            impact_score = (
                intervention.expected_impact.get('peak_reduction', 0) * 100 +
                intervention.expected_impact.get('carbon_reduction', 0) * 1000 +
                intervention.expected_impact.get('self_sufficiency_increase', 0) * 10000
            )
            c[i] = -impact_score  # Negative because linprog minimizes
        
        # Constraint: total cost <= budget
        A_ub = np.zeros((1, n))
        for i, intervention in enumerate(interventions):
            A_ub[0, i] = intervention.estimated_cost
        b_ub = np.array([budget])
        
        # Bounds: binary decision variables (0 or 1)
        bounds = [(0, 1) for _ in range(n)]
        
        # Solve
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        # Select interventions based on solution
        if result.success:
            selected_indices = np.where(result.x > 0.5)[0]
            selected_interventions = [interventions[i] for i in selected_indices]
            
            # Ensure dependencies are met
            selected_interventions = self._ensure_dependencies(
                selected_interventions,
                interventions
            )
            
            return selected_interventions
        else:
            # Fallback: greedy selection by priority
            interventions.sort(key=lambda x: x.priority_score, reverse=True)
            selected = []
            total_cost = 0
            
            for intervention in interventions:
                if total_cost + intervention.estimated_cost <= budget:
                    selected.append(intervention)
                    total_cost += intervention.estimated_cost
            
            return selected
    
    def _ensure_dependencies(
        self,
        selected: List[Intervention],
        all_interventions: List[Intervention]
    ) -> List[Intervention]:
        """Ensure all dependencies are included"""
        selected_ids = {i.intervention_id for i in selected}
        intervention_map = {i.intervention_id: i for i in all_interventions}
        
        added = True
        while added:
            added = False
            for intervention in list(selected):
                for dep_id in intervention.dependencies:
                    if dep_id not in selected_ids and dep_id in intervention_map:
                        selected.append(intervention_map[dep_id])
                        selected_ids.add(dep_id)
                        added = True
        
        return selected
    
    def _rank_interventions(self, interventions: List[Intervention]) -> List[Intervention]:
        """
        Rank interventions by combined priority score
        """
        for intervention in interventions:
            # Calculate priority score combining multiple factors
            impact_score = sum(intervention.expected_impact.values())
            network_score = intervention.network_effects.get('cascade_multiplier', 1.0)
            cost_effectiveness = impact_score / (intervention.estimated_cost + 1)
            
            # Timeline factor (prefer quicker implementations)
            timeline_scores = {
                '1 month': 1.2,
                '1-2 months': 1.1,
                '2-4 months': 1.0,
                '3-6 months': 0.9,
                '6-9 months': 0.8,
                '6-12 months': 0.7
            }
            timeline_factor = timeline_scores.get(intervention.implementation_timeline, 0.5)
            
            # Combined priority score
            intervention.priority_score = (
                impact_score * 0.4 +
                network_score * 1000 * 0.3 +
                cost_effectiveness * 100 * 0.2 +
                timeline_factor * 100 * 0.1
            )
        
        # Sort by priority
        interventions.sort(key=lambda x: x.priority_score, reverse=True)
        
        return interventions
    
    def _create_implementation_phases(
        self,
        interventions: List[Intervention]
    ) -> List[Dict[str, Any]]:
        """
        Create phased implementation plan
        """
        phases = []
        
        # Phase 1: Quick wins and enablers (1-2 months)
        phase1 = {
            'phase': 1,
            'name': 'Quick Wins & Enablers',
            'duration': '1-2 months',
            'interventions': [
                i.intervention_id for i in interventions
                if i.implementation_timeline in ['1 month', '1-2 months']
            ],
            'focus': 'Smart meters, demand response enrollment',
            'budget_percentage': 0.15
        }
        phases.append(phase1)
        
        # Phase 2: Core infrastructure (2-6 months)
        phase2 = {
            'phase': 2,
            'name': 'Core Infrastructure',
            'duration': '2-6 months',
            'interventions': [
                i.intervention_id for i in interventions
                if i.implementation_timeline in ['2-4 months', '3-6 months']
            ],
            'focus': 'Solar PV, battery storage',
            'budget_percentage': 0.45
        }
        phases.append(phase2)
        
        # Phase 3: Deep retrofits and upgrades (6-12 months)
        phase3 = {
            'phase': 3,
            'name': 'Deep Retrofits & Grid Upgrades',
            'duration': '6-12 months',
            'interventions': [
                i.intervention_id for i in interventions
                if '6' in i.implementation_timeline or '12' in i.implementation_timeline
            ],
            'focus': 'Building retrofits, transformer upgrades',
            'budget_percentage': 0.40
        }
        phases.append(phase3)
        
        return phases
    
    def _create_intervention_plan(
        self,
        interventions: List[Intervention],
        phases: List[Dict],
        analysis_results: Dict
    ) -> InterventionPlan:
        """
        Create complete intervention plan
        """
        # Calculate total cost and benefits
        total_cost = sum(i.estimated_cost for i in interventions)
        
        # Aggregate expected benefits
        total_benefits = {
            'peak_reduction': sum(i.expected_impact.get('peak_reduction', 0) for i in interventions),
            'carbon_reduction': sum(i.expected_impact.get('carbon_reduction', 0) for i in interventions),
            'self_sufficiency_increase': np.mean([
                i.expected_impact.get('self_sufficiency_increase', 0) 
                for i in interventions if i.expected_impact.get('self_sufficiency_increase', 0) > 0
            ]) if interventions else 0,
            'annual_savings': sum(i.expected_impact.get('annual_savings', 0) for i in interventions),
            'grid_flexibility': sum(i.expected_impact.get('grid_flexibility', 0) for i in interventions)
        }
        
        # Risk assessment
        risk_assessment = {
            'technical_risk': 'low' if len([i for i in interventions if i.type == InterventionType.GRID_UPGRADE]) < 2 else 'medium',
            'financial_risk': 'low' if total_cost < 1000000 else 'medium',
            'implementation_complexity': 'low' if len(interventions) < 10 else 'medium',
            'dependency_risk': 'low' if sum(len(i.dependencies) for i in interventions) < 5 else 'medium'
        }
        
        # Monitoring metrics
        monitoring_metrics = [
            'peak_demand_reduction',
            'self_sufficiency_ratio',
            'carbon_emissions',
            'grid_violations',
            'cost_savings',
            'system_reliability'
        ]
        
        plan = InterventionPlan(
            plan_id=f"plan_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
            interventions=interventions,
            total_cost=total_cost,
            expected_benefits=total_benefits,
            implementation_phases=phases,
            risk_assessment=risk_assessment,
            monitoring_metrics=monitoring_metrics
        )
        
        return plan
    
    def simulate_intervention_impact(
        self,
        intervention: Intervention,
        current_state: Dict[str, Any],
        gnn_embeddings: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """
        Simulate the impact of an intervention using GNN embeddings
        
        This is where GNN intelligence enhances simple rules
        """
        base_impact = intervention.expected_impact.copy()
        
        # If we have GNN embeddings, use them to refine impact estimates
        if gnn_embeddings is not None:
            # Extract relevant embedding for intervention location
            if isinstance(intervention.location, int):
                location_embedding = gnn_embeddings[intervention.location]
                
                # Use embedding to adjust impact (simplified)
                # In practice, you'd have a trained model for this
                embedding_factor = 1 + (location_embedding.mean() - 0.5) * 0.2
                
                for key in base_impact:
                    base_impact[key] *= embedding_factor
        
        # Apply network effects
        cascade_multiplier = intervention.network_effects.get('cascade_multiplier', 1.0)
        for key in base_impact:
            base_impact[key] *= cascade_multiplier
        
        return base_impact
    
    def _convert_to_native(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native(v) for v in obj]
        else:
            return obj
    
    def export_plan(
        self,
        plan: InterventionPlan,
        format: str = 'json',
        output_path: Optional[str] = None
    ) -> str:
        """
        Export intervention plan in various formats
        
        Args:
            plan: Intervention plan to export
            format: Export format ('json', 'csv', 'report')
            output_path: Optional path to save file
            
        Returns:
            Exported content as string
        """
        if format == 'json':
            # Convert to JSON-serializable format
            plan_dict = {
                'plan_id': plan.plan_id,
                'total_cost': self._convert_to_native(plan.total_cost),
                'expected_benefits': self._convert_to_native(plan.expected_benefits),
                'implementation_phases': self._convert_to_native(plan.implementation_phases),
                'risk_assessment': self._convert_to_native(plan.risk_assessment),
                'monitoring_metrics': self._convert_to_native(plan.monitoring_metrics),
                'interventions': []
            }
            
            for intervention in plan.interventions:
                int_dict = {
                    'id': intervention.intervention_id,
                    'type': intervention.type.value,
                    'location': str(intervention.location),
                    'size': self._convert_to_native(intervention.size),
                    'cost': self._convert_to_native(intervention.estimated_cost),
                    'priority': self._convert_to_native(intervention.priority_score),
                    'timeline': intervention.implementation_timeline,
                    'specifications': self._convert_to_native(intervention.specifications),
                    'expected_impact': self._convert_to_native(intervention.expected_impact),
                    'network_effects': self._convert_to_native(intervention.network_effects),
                    'dependencies': intervention.dependencies,
                    'co_benefits': intervention.co_benefits
                }
                plan_dict['interventions'].append(int_dict)
            
            content = json.dumps(plan_dict, indent=2)
            
        elif format == 'csv':
            # Convert to DataFrame for CSV export
            rows = []
            for intervention in plan.interventions:
                row = {
                    'ID': intervention.intervention_id,
                    'Type': intervention.type.value,
                    'Location': str(intervention.location),
                    'Size': intervention.size,
                    'Cost': intervention.estimated_cost,
                    'Priority': intervention.priority_score,
                    'Timeline': intervention.implementation_timeline,
                    'Peak_Reduction': intervention.expected_impact.get('peak_reduction', 0),
                    'Carbon_Reduction': intervention.expected_impact.get('carbon_reduction', 0),
                    'Network_Effect': intervention.network_effects.get('cascade_multiplier', 1.0)
                }
                rows.append(row)
            
            df = pd.DataFrame(rows)
            content = df.to_csv(index=False)
            
        else:  # report format
            content = self._generate_text_report(plan)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(content)
            print(f"Plan exported to {output_path}")
        
        return content
    
    def _generate_text_report(self, plan: InterventionPlan) -> str:
        """Generate human-readable text report"""
        report = f"""
ENERGY INTERVENTION PLAN
========================
Plan ID: {plan.plan_id}
Total Investment: ${plan.total_cost:,.0f}

EXPECTED BENEFITS
-----------------
Peak Reduction: {plan.expected_benefits.get('peak_reduction', 0):.1f} kW
Carbon Reduction: {plan.expected_benefits.get('carbon_reduction', 0):.1f} tons/year
Self-Sufficiency: {plan.expected_benefits.get('self_sufficiency_increase', 0)*100:.1f}% increase
Annual Savings: ${plan.expected_benefits.get('annual_savings', 0):,.0f}

IMPLEMENTATION PHASES
---------------------
"""
        for phase in plan.implementation_phases:
            report += f"\nPhase {phase['phase']}: {phase['name']}"
            report += f"\nDuration: {phase['duration']}"
            report += f"\nInterventions: {len(phase['interventions'])}"
            report += f"\nBudget: {phase['budget_percentage']*100:.0f}% of total\n"
        
        report += "\nTOP PRIORITY INTERVENTIONS\n"
        report += "--------------------------\n"
        
        for intervention in plan.interventions[:10]:
            report += f"\n{intervention.intervention_id}"
            report += f"\n  Type: {intervention.type.value}"
            report += f"\n  Location: {intervention.location}"
            report += f"\n  Size: {intervention.size:.1f}"
            report += f"\n  Cost: ${intervention.estimated_cost:,.0f}"
            report += f"\n  Priority: {intervention.priority_score:.1f}"
            report += f"\n  Network Effect: {intervention.network_effects.get('cascade_multiplier', 1.0):.2f}x\n"
        
        return report