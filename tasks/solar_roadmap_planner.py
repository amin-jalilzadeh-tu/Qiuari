"""
Solar Roadmap Planner
Generates multi-year deployment plans for achieving target solar penetration rates
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class YearlyPlan:
    """Plan for a single year of solar deployment"""
    year: int
    target_installations: List[int]  # Building IDs
    capacities: List[float]  # kWp per building
    total_capacity_mw: float
    cumulative_penetration: float
    expected_self_sufficiency: float
    expected_peak_reduction_mw: float
    budget_required: float
    cluster_assignments: Dict[int, int] = field(default_factory=dict)
    cascade_impacts: Dict[int, float] = field(default_factory=dict)
    

@dataclass
class SolarRoadmap:
    """Complete multi-year solar deployment roadmap"""
    target_penetration: float
    timeframe_years: int
    total_available_roof_area: float
    current_penetration: float
    yearly_plans: List[YearlyPlan]
    optimization_strategy: str
    total_investment: float
    expected_benefits: Dict[str, float]
    cluster_evolution: List[Dict]  # Cluster changes per year
    created_at: datetime = field(default_factory=datetime.now)


class SolarRoadmapPlanner:
    """
    Plans multi-year solar deployment to achieve penetration targets
    """
    
    def __init__(self, config: Dict):
        """
        Initialize roadmap planner
        
        Args:
            config: Configuration with planning parameters
        """
        self.config = config
        
        # Planning parameters
        self.default_timeframe = config.get('default_timeframe_years', 5)
        self.annual_budget = config.get('annual_budget', 1000000)  # euros
        self.cost_per_kwp = config.get('cost_per_kwp', 1200)
        
        # Optimization parameters
        self.strategies = {
            'linear': self._linear_progression,
            'accelerated': self._accelerated_progression,
            'cascade_optimized': self._cascade_optimized_progression,
            'cluster_balanced': self._cluster_balanced_progression
        }
        
        # Constraints
        self.max_annual_capacity_mw = config.get('max_annual_capacity_mw', 5.0)
        self.min_installation_size_kwp = config.get('min_installation_size_kwp', 3.0)
        self.max_installation_size_kwp = config.get('max_installation_size_kwp', 10.0)
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.05)  # Cost reduction per year
        self.efficiency_improvement = config.get('efficiency_improvement', 0.02)  # Annual improvement
        
        logger.info("Initialized SolarRoadmapPlanner")
    
    def generate_roadmap(
        self,
        building_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_solar: torch.Tensor,
        target_penetration: float,
        timeframe_years: Optional[int] = None,
        strategy: str = 'cascade_optimized',
        cluster_assignments: Optional[torch.Tensor] = None,
        additional_constraints: Optional[Dict] = None
    ) -> SolarRoadmap:
        """
        Generate complete solar deployment roadmap
        
        Args:
            building_features: Building feature matrix [N, F]
            edge_index: Building connections
            current_solar: Current solar installations (binary or capacity)
            target_penetration: Target penetration rate (0-1)
            timeframe_years: Planning horizon in years
            strategy: Optimization strategy to use
            cluster_assignments: Current cluster assignments
            additional_constraints: Extra constraints (budget, grid, etc.)
            
        Returns:
            Complete solar deployment roadmap
        """
        timeframe = timeframe_years or self.default_timeframe
        
        # Calculate available capacity
        roof_areas = building_features[:, 5].cpu().numpy()  # Roof area feature
        total_available_area = roof_areas.sum()
        
        # Current state
        current_installations = current_solar.cpu().numpy()
        current_area_used = (current_installations * roof_areas * 0.1).sum()  # Rough estimate
        current_penetration = current_area_used / total_available_area
        
        # Target state
        target_area = total_available_area * target_penetration
        area_to_install = target_area - current_area_used
        
        if area_to_install <= 0:
            logger.warning(f"Target penetration {target_penetration:.1%} already achieved!")
            return self._create_empty_roadmap(target_penetration, timeframe)
        
        # Select optimization strategy
        if strategy not in self.strategies:
            logger.warning(f"Unknown strategy {strategy}, using cascade_optimized")
            strategy = 'cascade_optimized'
        
        progression_func = self.strategies[strategy]
        
        # Generate yearly allocations
        yearly_allocations = progression_func(
            area_to_install,
            timeframe,
            building_features,
            edge_index,
            current_installations,
            cluster_assignments
        )
        
        # Create yearly plans
        yearly_plans = []
        cumulative_penetration = current_penetration
        cumulative_installations = current_installations.copy()
        evolving_clusters = cluster_assignments.clone() if cluster_assignments is not None else None
        
        for year, allocation in enumerate(yearly_allocations, 1):
            # Select buildings for this year
            selected_buildings, capacities = self._select_buildings_for_year(
                allocation,
                building_features,
                edge_index,
                cumulative_installations,
                evolving_clusters,
                year
            )
            
            # Calculate impacts
            cascade_impacts = self._calculate_cascade_impacts(
                selected_buildings,
                capacities,
                edge_index,
                building_features
            )
            
            # Update cumulative state
            for building_id, capacity in zip(selected_buildings, capacities):
                cumulative_installations[building_id] += capacity / 10.0  # Normalized
            
            cumulative_penetration += allocation['area'] / total_available_area
            
            # Estimate cluster evolution
            if evolving_clusters is not None:
                evolving_clusters = self._predict_cluster_evolution(
                    evolving_clusters,
                    selected_buildings,
                    capacities,
                    building_features
                )
            
            # Create yearly plan
            plan = YearlyPlan(
                year=year,
                target_installations=selected_buildings,
                capacities=capacities,
                total_capacity_mw=sum(capacities) / 1000,
                cumulative_penetration=cumulative_penetration,
                expected_self_sufficiency=self._estimate_self_sufficiency(
                    cumulative_installations,
                    building_features
                ),
                expected_peak_reduction_mw=self._estimate_peak_reduction(
                    cumulative_installations,
                    building_features
                ),
                budget_required=self._calculate_budget(capacities, year),
                cluster_assignments=evolving_clusters.cpu().numpy().tolist() if evolving_clusters is not None else {},
                cascade_impacts=cascade_impacts
            )
            
            yearly_plans.append(plan)
        
        # Calculate total benefits
        expected_benefits = self._calculate_total_benefits(yearly_plans, building_features)
        
        # Create roadmap
        roadmap = SolarRoadmap(
            target_penetration=target_penetration,
            timeframe_years=timeframe,
            total_available_roof_area=total_available_area,
            current_penetration=current_penetration,
            yearly_plans=yearly_plans,
            optimization_strategy=strategy,
            total_investment=sum(p.budget_required for p in yearly_plans),
            expected_benefits=expected_benefits,
            cluster_evolution=self._track_cluster_evolution(yearly_plans)
        )
        
        logger.info(f"Generated {timeframe}-year roadmap to achieve {target_penetration:.1%} penetration")
        logger.info(f"Total investment: €{roadmap.total_investment:,.0f}")
        
        return roadmap
    
    def _linear_progression(
        self,
        total_area: float,
        years: int,
        building_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_installations: np.ndarray,
        cluster_assignments: Optional[torch.Tensor]
    ) -> List[Dict]:
        """Linear progression - equal allocation per year"""
        annual_area = total_area / years
        
        allocations = []
        for year in range(years):
            allocations.append({
                'area': annual_area,
                'capacity_mw': annual_area * 0.15 / 1000,  # 150W/m²
                'priority': 'uniform'
            })
        
        return allocations
    
    def _accelerated_progression(
        self,
        total_area: float,
        years: int,
        building_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_installations: np.ndarray,
        cluster_assignments: Optional[torch.Tensor]
    ) -> List[Dict]:
        """Accelerated progression - more installations in later years"""
        # Exponential growth pattern
        growth_rate = 1.5
        base = total_area / sum(growth_rate ** i for i in range(years))
        
        allocations = []
        for year in range(years):
            area = base * (growth_rate ** year)
            allocations.append({
                'area': area,
                'capacity_mw': area * 0.15 / 1000,
                'priority': 'accelerating'
            })
        
        return allocations
    
    def _cascade_optimized_progression(
        self,
        total_area: float,
        years: int,
        building_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_installations: np.ndarray,
        cluster_assignments: Optional[torch.Tensor]
    ) -> List[Dict]:
        """Cascade-optimized - prioritize high network impact"""
        # Start with high-impact nodes, gradually expand
        
        # Calculate cascade potential for all buildings
        cascade_scores = self._calculate_cascade_potential(
            building_features,
            edge_index,
            current_installations
        )
        
        # Allocate more to early years for cascade benefits
        # Front-loaded distribution
        weights = np.array([2.0, 1.8, 1.5, 1.2, 1.0][:years])
        weights = weights / weights.sum()
        
        allocations = []
        for i, weight in enumerate(weights):
            area = total_area * weight
            allocations.append({
                'area': area,
                'capacity_mw': area * 0.15 / 1000,
                'priority': 'cascade',
                'cascade_scores': cascade_scores
            })
        
        return allocations
    
    def _cluster_balanced_progression(
        self,
        total_area: float,
        years: int,
        building_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_installations: np.ndarray,
        cluster_assignments: Optional[torch.Tensor]
    ) -> List[Dict]:
        """Cluster-balanced - ensure even distribution across clusters"""
        if cluster_assignments is None:
            # Fall back to linear if no clusters
            return self._linear_progression(
                total_area, years, building_features, 
                edge_index, current_installations, cluster_assignments
            )
        
        # Calculate cluster imbalances
        num_clusters = cluster_assignments.max().item() + 1
        cluster_penetrations = []
        
        for c in range(num_clusters):
            mask = cluster_assignments == c
            cluster_area = building_features[mask, 5].sum().item()
            cluster_solar = current_installations[mask.cpu().numpy()].sum()
            penetration = cluster_solar / (cluster_area + 1e-10)
            cluster_penetrations.append(penetration)
        
        # Allocate to balance clusters
        annual_area = total_area / years
        allocations = []
        
        for year in range(years):
            # Prioritize low-penetration clusters
            priorities = 1.0 / (np.array(cluster_penetrations) + 0.1)
            priorities = priorities / priorities.sum()
            
            allocations.append({
                'area': annual_area,
                'capacity_mw': annual_area * 0.15 / 1000,
                'priority': 'cluster_balanced',
                'cluster_priorities': priorities.tolist()
            })
        
        return allocations
    
    def _select_buildings_for_year(
        self,
        allocation: Dict,
        building_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_installations: np.ndarray,
        cluster_assignments: Optional[torch.Tensor],
        year: int
    ) -> Tuple[List[int], List[float]]:
        """Select specific buildings for installation in a given year"""
        target_area = allocation['area']
        priority = allocation.get('priority', 'uniform')
        
        # Get available buildings (not yet installed)
        available_mask = current_installations < 0.1  # Not installed
        available_indices = np.where(available_mask)[0]
        
        if len(available_indices) == 0:
            return [], []
        
        # Score buildings based on priority
        if priority == 'cascade' and 'cascade_scores' in allocation:
            scores = allocation['cascade_scores'][available_indices]
        elif priority == 'cluster_balanced' and cluster_assignments is not None:
            # Score based on cluster needs
            scores = self._score_by_cluster_balance(
                available_indices,
                cluster_assignments,
                allocation.get('cluster_priorities', [])
            )
        else:
            # Default scoring based on roof area and orientation
            scores = building_features[available_indices, 5].cpu().numpy()  # Roof area
            orientation_scores = building_features[available_indices, 7].cpu().numpy()  # Orientation
            scores = scores * (1 + orientation_scores * 0.2)
        
        # Sort by score
        sorted_indices = available_indices[np.argsort(scores)[::-1]]
        
        # Select buildings until target area is reached
        selected_buildings = []
        capacities = []
        cumulative_area = 0
        
        for idx in sorted_indices:
            roof_area = building_features[idx, 5].item()
            capacity = min(roof_area * 0.15, self.max_installation_size_kwp)
            capacity = max(capacity, self.min_installation_size_kwp)
            
            if cumulative_area + roof_area <= target_area * 1.1:  # Allow 10% overage
                selected_buildings.append(int(idx))
                capacities.append(float(capacity))
                cumulative_area += roof_area
                
                if cumulative_area >= target_area * 0.9:  # Within 10% of target
                    break
        
        # Apply learning curve to costs (installations get cheaper/better over time)
        capacities = [c * (1 + self.efficiency_improvement * year) for c in capacities]
        
        return selected_buildings, capacities
    
    def _calculate_cascade_potential(
        self,
        building_features: torch.Tensor,
        edge_index: torch.Tensor,
        current_installations: np.ndarray
    ) -> np.ndarray:
        """Calculate cascade potential for each building"""
        num_buildings = building_features.shape[0]
        scores = np.zeros(num_buildings)
        
        # Simple degree-based scoring for now
        if edge_index.shape[1] > 0:
            degrees = torch.zeros(num_buildings)
            degrees.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1]))
            degrees.scatter_add_(0, edge_index[1], torch.ones(edge_index.shape[1]))
            scores = degrees.cpu().numpy()
        
        # Penalize areas with existing solar
        for i in range(num_buildings):
            if current_installations[i] > 0:
                # Reduce scores of neighbors
                neighbors = edge_index[1][edge_index[0] == i].cpu().numpy()
                scores[neighbors] *= 0.7
        
        return scores
    
    def _calculate_cascade_impacts(
        self,
        buildings: List[int],
        capacities: List[float],
        edge_index: torch.Tensor,
        building_features: torch.Tensor
    ) -> Dict[int, float]:
        """Calculate cascade impacts for selected buildings"""
        impacts = {}
        
        for building_id, capacity in zip(buildings, capacities):
            # Simple impact based on capacity and degree
            if edge_index.shape[1] > 0:
                neighbors = edge_index[1][edge_index[0] == building_id].cpu().numpy()
                impact = capacity * len(neighbors) * 0.1  # Simple formula
            else:
                impact = capacity * 0.5
            
            impacts[building_id] = impact
        
        return impacts
    
    def _predict_cluster_evolution(
        self,
        current_clusters: torch.Tensor,
        new_installations: List[int],
        capacities: List[float],
        building_features: torch.Tensor
    ) -> torch.Tensor:
        """Predict how clusters might evolve with new installations"""
        # Simplified: clusters don't change for now
        # In full implementation, this would re-run clustering
        return current_clusters
    
    def _estimate_self_sufficiency(
        self,
        installations: np.ndarray,
        building_features: torch.Tensor
    ) -> float:
        """Estimate community self-sufficiency with current installations"""
        # Simple estimation based on solar coverage
        total_consumption = building_features[:, 3].sum().item()  # Annual consumption
        solar_generation = installations.sum() * 10 * 1200  # kWp * kWh/kWp/year
        
        return min(1.0, solar_generation / (total_consumption + 1e-10))
    
    def _estimate_peak_reduction(
        self,
        installations: np.ndarray,
        building_features: torch.Tensor
    ) -> float:
        """Estimate peak demand reduction in MW"""
        # Simple estimation
        total_capacity_kw = installations.sum() * 10  # Assuming 10 kWp average
        peak_reduction = total_capacity_kw * 0.6 / 1000  # 60% coincidence, convert to MW
        
        return peak_reduction
    
    def _calculate_budget(
        self,
        capacities: List[float],
        year: int
    ) -> float:
        """Calculate required budget for installations"""
        # Apply learning curve
        cost_reduction = (1 - self.learning_rate) ** year
        cost_per_kwp = self.cost_per_kwp * cost_reduction
        
        return sum(capacities) * cost_per_kwp
    
    def _calculate_total_benefits(
        self,
        yearly_plans: List[YearlyPlan],
        building_features: torch.Tensor
    ) -> Dict[str, float]:
        """Calculate total expected benefits from roadmap"""
        benefits = {
            'total_capacity_mw': sum(p.total_capacity_mw for p in yearly_plans),
            'final_self_sufficiency': yearly_plans[-1].expected_self_sufficiency if yearly_plans else 0,
            'total_peak_reduction_mw': yearly_plans[-1].expected_peak_reduction_mw if yearly_plans else 0,
            'annual_co2_reduction_tons': 0,
            'annual_energy_generated_gwh': 0,
            'grid_investment_avoided': 0
        }
        
        # Calculate CO2 reduction (0.4 kg/kWh * annual generation)
        total_capacity_mw = benefits['total_capacity_mw']
        annual_generation_gwh = total_capacity_mw * 1.2  # 1200 kWh/kWp
        benefits['annual_energy_generated_gwh'] = annual_generation_gwh
        benefits['annual_co2_reduction_tons'] = annual_generation_gwh * 1000 * 0.4
        
        # Grid investment avoided (rough estimate)
        benefits['grid_investment_avoided'] = total_capacity_mw * 500000  # €500k/MW avoided
        
        return benefits
    
    def _track_cluster_evolution(
        self,
        yearly_plans: List[YearlyPlan]
    ) -> List[Dict]:
        """Track how clusters evolve over time"""
        evolution = []
        
        for plan in yearly_plans:
            if plan.cluster_assignments:
                # Analyze cluster changes
                cluster_stats = defaultdict(int)
                # Handle both dict and list types
                if isinstance(plan.cluster_assignments, dict):
                    for building_id, cluster_id in plan.cluster_assignments.items():
                        cluster_stats[cluster_id] += 1
                elif isinstance(plan.cluster_assignments, list):
                    for cluster_id in plan.cluster_assignments:
                        cluster_stats[cluster_id] += 1
                
                evolution.append({
                    'year': plan.year,
                    'num_clusters': len(cluster_stats),
                    'cluster_sizes': list(cluster_stats.values()),
                    'installations_this_year': len(plan.target_installations)
                })
        
        return evolution
    
    def _create_empty_roadmap(
        self,
        target_penetration: float,
        timeframe: int
    ) -> SolarRoadmap:
        """Create empty roadmap when target is already met"""
        return SolarRoadmap(
            target_penetration=target_penetration,
            timeframe_years=timeframe,
            total_available_roof_area=0,
            current_penetration=target_penetration,
            yearly_plans=[],
            optimization_strategy='none',
            total_investment=0,
            expected_benefits={},
            cluster_evolution=[]
        )
    
    def _score_by_cluster_balance(
        self,
        available_indices: np.ndarray,
        cluster_assignments: torch.Tensor,
        cluster_priorities: List[float]
    ) -> np.ndarray:
        """Score buildings based on cluster balancing needs"""
        scores = np.zeros(len(available_indices))
        
        for i, idx in enumerate(available_indices):
            cluster_id = cluster_assignments[idx].item()
            if cluster_id < len(cluster_priorities):
                scores[i] = cluster_priorities[cluster_id]
            else:
                scores[i] = 0.5  # Default score
        
        return scores
    
    def export_roadmap_to_excel(
        self,
        roadmap: SolarRoadmap,
        filepath: str
    ):
        """Export roadmap to Excel file for reporting"""
        import pandas as pd
        
        # Create summary dataframe
        summary_data = []
        for plan in roadmap.yearly_plans:
            summary_data.append({
                'Year': plan.year,
                'Buildings': len(plan.target_installations),
                'Capacity (MW)': plan.total_capacity_mw,
                'Investment (€)': plan.budget_required,
                'Cumulative Penetration (%)': plan.cumulative_penetration * 100,
                'Self-Sufficiency (%)': plan.expected_self_sufficiency * 100,
                'Peak Reduction (MW)': plan.expected_peak_reduction_mw
            })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Create detailed installations dataframe
        details_data = []
        for plan in roadmap.yearly_plans:
            for building_id, capacity in zip(plan.target_installations, plan.capacities):
                details_data.append({
                    'Year': plan.year,
                    'Building ID': building_id,
                    'Capacity (kWp)': capacity,
                    'Cascade Impact': plan.cascade_impacts.get(building_id, 0)
                })
        
        df_details = pd.DataFrame(details_data)
        
        # Write to Excel
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            df_details.to_excel(writer, sheet_name='Installation Details', index=False)
            
            # Add metadata sheet
            metadata = pd.DataFrame([
                {'Parameter': 'Target Penetration', 'Value': f"{roadmap.target_penetration:.1%}"},
                {'Parameter': 'Timeframe', 'Value': f"{roadmap.timeframe_years} years"},
                {'Parameter': 'Total Investment', 'Value': f"€{roadmap.total_investment:,.0f}"},
                {'Parameter': 'Strategy', 'Value': roadmap.optimization_strategy},
                {'Parameter': 'Created', 'Value': roadmap.created_at.strftime('%Y-%m-%d %H:%M')}
            ])
            metadata.to_excel(writer, sheet_name='Metadata', index=False)
        
        logger.info(f"Roadmap exported to {filepath}")