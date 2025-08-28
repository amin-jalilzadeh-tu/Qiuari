# tasks/solar_optimization.py
"""
Solar panel placement optimization task
Identifies optimal locations and sizes for PV installations
Considers clustering results to maximize energy sharing benefits
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class SolarConfig:
    """Configuration for solar optimization task"""
    # Technical parameters
    panel_efficiency: float = 0.20  # 20% efficient panels
    system_losses: float = 0.14  # Inverter, wiring, soiling losses
    degradation_rate: float = 0.005  # 0.5% per year
    panels_per_kw: float = 3.0  # Number of panels per kW
    panel_area_m2: float = 2.0  # Area per panel in m²
    
    # Constraints
    min_roof_area: float = 20.0  # Minimum roof area for installation
    max_capacity_per_building: float = 100.0  # Maximum kW per building
    min_solar_score: float = 0.6  # Minimum viability score
    shading_threshold: float = 0.7  # Maximum acceptable shading
    
    # Economic parameters
    cost_per_kwp: float = 1000.0  # €/kWp installed
    electricity_price: float = 0.25  # €/kWh
    feed_in_tariff: float = 0.08  # €/kWh for export
    maintenance_cost_annual: float = 20.0  # €/kWp/year
    discount_rate: float = 0.05  # For NPV calculation
    lifetime_years: int = 25
    
    # Installation priorities
    prioritize_complementarity: bool = True
    cluster_coordination: bool = True
    maximize_self_consumption: bool = True


@dataclass
class SolarCandidate:
    """Represents a building candidate for solar installation"""
    building_id: int
    cluster_id: int
    roof_area: float
    suitable_area: float
    orientation: str
    existing_solar: bool
    solar_capacity_kw: float
    
    # Scores
    technical_score: float
    economic_score: float
    complementarity_score: float
    total_score: float
    
    # Recommendations
    recommended_capacity_kw: float
    expected_generation_annual_kwh: float
    expected_self_consumption_ratio: float
    expected_roi_years: float
    installation_cost: float
    annual_savings: float
    co2_reduction_tons: float


class SolarOptimization:
    """
    Solar placement optimization task that works with clustering results
    """
    
    def __init__(self, model, config: Union[Dict, SolarConfig]):
        """
        Initialize solar optimization task
        
        Args:
            model: Trained GNN model (can be None for rule-based)
            config: Task configuration
        """
        self.model = model
        
        # Parse configuration
        if isinstance(config, dict):
            self.config = SolarConfig(**config.get('solar_optimization', {}))
        else:
            self.config = config
        
        # Storage for results
        self.candidates = []
        self.cluster_solar_plans = {}
        self.system_metrics = {}
        
        # Building ID mappings (will be set from clustering)
        self.building_id_to_index = {}
        
        # Solar irradiance data (simplified - in practice load from weather data)
        self.annual_irradiance = {
            'north': 900,  # kWh/m²/year
            'south': 1200,
            'east': 1050,
            'west': 1050,
            'flat': 1100,
            'unknown': 1000
        }
        
        logger.info(f"Initialized SolarOptimization with config: {self.config}")
    
    def run(self,
            graph_data: Dict,
            clustering_results: Dict,
            embeddings: Optional[torch.Tensor] = None,
            temporal_data: Optional[pd.DataFrame] = None,
            cluster_quality_labels: Optional[Dict] = None) -> Dict:
        """
        Run solar optimization based on clustering results
        
        Args:
            graph_data: Building and grid information
            clustering_results: Results from clustering task
            embeddings: Building embeddings from GNN
            temporal_data: Energy consumption profiles
            
        Returns:
            Solar optimization results with rankings and recommendations
        """
        logger.info("Running solar optimization...")
        
        # Extract data
        buildings = graph_data.get('buildings', {})
        clusters = clustering_results.get('clusters', [])
        complementarity_matrix = clustering_results.get('complementarity_matrix', None)
        
        # Get building ID mapping from clustering if available
        if hasattr(clustering_results, 'building_id_to_index'):
            self.building_id_to_index = clustering_results.building_id_to_index
        else:
            # Create our own mapping
            building_ids = list(buildings.keys())
            self.building_id_to_index = {bid: idx for idx, bid in enumerate(building_ids)}
        
        # Analyze each cluster
        all_candidates = []
        cluster_plans = {}
        
        for cluster_id, cluster_buildings in enumerate(clusters):
            # Get cluster quality label if available
            cluster_quality = 'unknown'
            if cluster_quality_labels and cluster_id in cluster_quality_labels:
                cluster_quality = cluster_quality_labels[cluster_id]
            
            logger.info(f"Analyzing cluster {cluster_id} ({cluster_quality} quality) with {len(cluster_buildings)} buildings")
            
            # Analyze solar potential for cluster
            cluster_candidates = self._analyze_cluster_solar_potential(
                cluster_buildings,
                buildings,
                cluster_id,
                complementarity_matrix,
                temporal_data,
                cluster_quality  # Pass quality label
            )
            
            # Optimize solar placement within cluster
            optimized_plan = self._optimize_cluster_solar(
                cluster_candidates,
                cluster_buildings,
                buildings,
                temporal_data
            )
            
            all_candidates.extend(cluster_candidates)
            cluster_plans[cluster_id] = optimized_plan
        
        # Rank all candidates globally
        ranked_candidates = self._rank_candidates_globally(all_candidates)
        
        # Calculate system-wide metrics
        system_metrics = self._calculate_system_metrics(
            ranked_candidates,
            buildings,
            clusters
        )
        
        # Store results
        self.candidates = ranked_candidates
        self.cluster_solar_plans = cluster_plans
        self.system_metrics = system_metrics
        
        # Prepare output
        return {
            'candidates': ranked_candidates,
            'cluster_plans': cluster_plans,
            'system_metrics': system_metrics,
            'recommendations': self._generate_recommendations(ranked_candidates),
            'summary': self._generate_summary(ranked_candidates, system_metrics)
        }
    
    def _analyze_cluster_solar_potential(self,
                                        cluster_buildings: List[int],
                                        buildings: Dict,
                                        cluster_id: int,
                                        complementarity_matrix: Optional[torch.Tensor],
                                        temporal_data: Optional[pd.DataFrame],
                                        cluster_quality: str = 'unknown') -> List[SolarCandidate]:
        """
        Analyze solar potential for buildings in a cluster
        """
        candidates = []
        
        for building_id in cluster_buildings:
            if building_id not in buildings:
                continue
            
            building = buildings[building_id]
            
            # Skip if already has solar (unless we're considering expansion)
            if building.get('has_solar', False) and building.get('solar_capacity_kw', 0) > 10:
                continue
            
            # Calculate technical score
            technical_score = self._calculate_technical_score(building)
            
            if technical_score < self.config.min_solar_score:
                continue  # Not viable for solar
            
            # Calculate economic score
            economic_score, roi_years, annual_savings = self._calculate_economic_score(
                building,
                temporal_data,
                building_id
            )
            
            # Calculate complementarity score within cluster
            complementarity_score = self._calculate_complementarity_score(
                building_id,
                cluster_buildings,
                buildings,
                complementarity_matrix,
                temporal_data
            )
            
            # Calculate total score with cluster quality boost
            total_score = self._calculate_total_score(
                technical_score,
                economic_score,
                complementarity_score,
                cluster_quality  # Pass cluster quality for prioritization
            )
            
            # Determine recommended capacity
            recommended_capacity = self._calculate_recommended_capacity(building)
            
            # Calculate expected generation
            expected_generation = self._calculate_expected_generation(
                recommended_capacity,
                building
            )
            
            # Calculate self-consumption ratio
            self_consumption_ratio = self._estimate_self_consumption(
                building_id,
                recommended_capacity,
                temporal_data,
                cluster_buildings
            )
            
            # Calculate CO2 reduction
            co2_reduction = self._calculate_co2_reduction(expected_generation)
            
            # Create candidate
            candidate = SolarCandidate(
                building_id=building_id,
                cluster_id=cluster_id,
                roof_area=building.get('area', 0) * 0.6,  # Assume 60% is roof
                suitable_area=building.get('suitable_roof_area', building.get('area', 0) * 0.4),
                orientation=building.get('building_orientation_cardinal', 'unknown'),
                existing_solar=building.get('has_solar', False),
                solar_capacity_kw=building.get('solar_capacity_kw', 0),
                technical_score=technical_score,
                economic_score=economic_score,
                complementarity_score=complementarity_score,
                total_score=total_score,
                recommended_capacity_kw=recommended_capacity,
                expected_generation_annual_kwh=expected_generation,
                expected_self_consumption_ratio=self_consumption_ratio,
                expected_roi_years=roi_years,
                installation_cost=recommended_capacity * self.config.cost_per_kwp,
                annual_savings=annual_savings,
                co2_reduction_tons=co2_reduction
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _calculate_technical_score(self, building: Dict) -> float:
        """
        Calculate technical viability score for solar installation
        """
        score = 0.0
        
        # Roof area score (0-0.3)
        roof_area = building.get('suitable_roof_area', building.get('area', 0) * 0.4)
        if roof_area >= 100:
            score += 0.3
        elif roof_area >= 50:
            score += 0.2
        elif roof_area >= self.config.min_roof_area:
            score += 0.1
        else:
            return 0.0  # Too small
        
        # Orientation score (0-0.3)
        orientation = building.get('building_orientation_cardinal', 'unknown')
        orientation_scores = {
            'south': 0.3,
            'south-east': 0.25,
            'south-west': 0.25,
            'east': 0.2,
            'west': 0.2,
            'flat': 0.25,
            'north': 0.1,
            'unknown': 0.15
        }
        score += orientation_scores.get(orientation.lower(), 0.15)
        
        # Building height/shading score (0-0.2)
        height = building.get('height', 10)
        if height > 15:  # Taller buildings less likely to be shaded
            score += 0.2
        elif height > 10:
            score += 0.15
        else:
            score += 0.1
        
        # Building type score (0-0.2)
        building_type = building.get('building_function', 'residential')
        if 'commercial' in building_type.lower() or 'office' in building_type.lower():
            score += 0.2  # Better daytime match with solar
        elif 'industrial' in building_type.lower():
            score += 0.15
        else:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_economic_score(self,
                                 building: Dict,
                                 temporal_data: Optional[pd.DataFrame],
                                 building_id: int) -> Tuple[float, float, float]:
        """
        Calculate economic viability score
        Returns: (score, roi_years, annual_savings)
        """
        # Get consumption data
        avg_consumption = building.get('avg_demand_kw', 10) * 24 * 365  # kWh/year
        peak_consumption = building.get('peak_demand_kw', 15)
        
        # Estimate system size
        recommended_capacity = self._calculate_recommended_capacity(building)
        installation_cost = recommended_capacity * self.config.cost_per_kwp
        
        # Estimate generation
        annual_generation = self._calculate_expected_generation(
            recommended_capacity,
            building
        )
        
        # Estimate self-consumption (simplified)
        self_consumption_ratio = 0.3  # Default 30%
        if temporal_data is not None:
            self_consumption_ratio = self._estimate_self_consumption(
                building_id,
                recommended_capacity,
                temporal_data,
                []
            )
        
        # Calculate savings
        self_consumed_kwh = annual_generation * self_consumption_ratio
        exported_kwh = annual_generation * (1 - self_consumption_ratio)
        
        annual_savings = (
            self_consumed_kwh * self.config.electricity_price +
            exported_kwh * self.config.feed_in_tariff -
            recommended_capacity * self.config.maintenance_cost_annual
        )
        
        # Calculate ROI
        roi_years = installation_cost / annual_savings if annual_savings > 0 else 999
        
        # Calculate score
        if roi_years <= 5:
            score = 1.0
        elif roi_years <= 8:
            score = 0.8
        elif roi_years <= 12:
            score = 0.6
        elif roi_years <= 15:
            score = 0.4
        else:
            score = 0.2
        
        return score, roi_years, annual_savings
    
    def _calculate_complementarity_score(self,
                                        building_id: int,
                                        cluster_buildings: List[int],
                                        buildings: Dict,
                                        complementarity_matrix: Optional[torch.Tensor],
                                        temporal_data: Optional[pd.DataFrame]) -> float:
        """
        Calculate how well this solar installation complements the cluster
        """
        if not self.config.prioritize_complementarity:
            return 0.5  # Neutral score
        
        score = 0.0
        
        # Check if building has high daytime consumption (good for self-consumption)
        building = buildings[building_id]
        building_type = building.get('building_function', 'residential')
        
        if 'office' in building_type.lower() or 'commercial' in building_type.lower():
            score += 0.3  # Good daytime match
        
        # Check complementarity with other buildings in cluster
        if complementarity_matrix is not None and building_id in self.building_id_to_index:
            building_idx = self.building_id_to_index[building_id]
            
            # Find buildings without solar that have evening peaks
            for other_id in cluster_buildings:
                if other_id == building_id:
                    continue
                
                other_building = buildings.get(other_id, {})
                
                # If other building has no solar and is residential (evening peak)
                if not other_building.get('has_solar', False):
                    if 'residential' in other_building.get('building_function', '').lower():
                        if other_id in self.building_id_to_index:
                            other_idx = self.building_id_to_index[other_id]
                            
                            # Get complementarity from matrix
                            if building_idx < len(complementarity_matrix) and other_idx < len(complementarity_matrix):
                                comp_value = complementarity_matrix[building_idx, other_idx].item()
                                if comp_value > 0:  # Negative correlation (complementary)
                                    score += 0.1 * comp_value
        
        # Check if cluster needs more generation
        cluster_solar_count = sum(
            1 for b_id in cluster_buildings
            if buildings.get(b_id, {}).get('has_solar', False)
        )
        
        if cluster_solar_count < len(cluster_buildings) * 0.3:  # Less than 30% have solar
            score += 0.2
        
        return min(1.0, score)
    
    def _calculate_total_score(self,
                              technical: float,
                              economic: float,
                              complementarity: float,
                              cluster_quality: str = 'unknown') -> float:
        """
        Combine scores with weights and prioritize poor clusters
        """
        weights = {
            'technical': 0.3,
            'economic': 0.4,
            'complementarity': 0.3
        }
        
        if self.config.prioritize_complementarity:
            weights['complementarity'] = 0.4
            weights['economic'] = 0.3
        
        base_score = (
            weights['technical'] * technical +
            weights['economic'] * economic +
            weights['complementarity'] * complementarity
        )
        
        # BOOST PRIORITY FOR POOR PERFORMING CLUSTERS
        # This ensures solar goes to clusters that need improvement most
        quality_multipliers = {
            'poor': 1.5,      # 50% boost for poor clusters
            'fair': 1.2,      # 20% boost for fair clusters  
            'good': 1.0,      # No change for good clusters
            'excellent': 0.9, # Slight reduction for excellent (already performing well)
            'unknown': 1.0    # No change if quality unknown
        }
        
        multiplier = quality_multipliers.get(cluster_quality.lower(), 1.0)
        final_score = base_score * multiplier
        
        if cluster_quality == 'poor' and final_score > 0.5:
            logger.debug(f"Boosted score from {base_score:.2f} to {final_score:.2f} for poor cluster")
        
        return final_score
    
    def _calculate_recommended_capacity(self, building: Dict) -> float:
        """
        Calculate recommended solar capacity for a building
        """
        # Get available roof area
        roof_area = building.get('suitable_roof_area', building.get('area', 0) * 0.4)
        
        # Calculate maximum capacity based on roof area
        # Typically need 7-10 m² per kWp
        max_capacity_from_roof = roof_area / 7.0
        
        # Consider consumption
        avg_consumption = building.get('avg_demand_kw', 10)
        peak_consumption = building.get('peak_demand_kw', 15)
        
        # Size to cover ~70% of average consumption
        consumption_based_capacity = avg_consumption * 0.7
        
        # Take minimum of constraints
        recommended = min(
            max_capacity_from_roof,
            consumption_based_capacity,
            self.config.max_capacity_per_building
        )
        
        # Round to practical size (multiples of 0.5 kW)
        recommended = round(recommended * 2) / 2
        
        return max(3.0, recommended)  # Minimum 3 kW for viability
    
    def _calculate_expected_generation(self,
                                      capacity_kw: float,
                                      building: Dict) -> float:
        """
        Calculate expected annual generation in kWh
        """
        # Get orientation
        orientation = building.get('building_orientation_cardinal', 'unknown').lower()
        
        # Get irradiance for orientation
        irradiance = self.annual_irradiance.get(orientation, 1000)
        
        # Calculate generation
        # Generation = Capacity × Irradiance × (1 - System Losses)
        annual_generation = (
            capacity_kw *
            irradiance *
            (1 - self.config.system_losses)
        )
        
        return annual_generation
    
    def _estimate_self_consumption(self,
                                  building_id: int,
                                  capacity_kw: float,
                                  temporal_data: Optional[pd.DataFrame],
                                  cluster_buildings: List[int]) -> float:
        """
        Estimate self-consumption ratio (how much generated power is used locally)
        """
        if temporal_data is None:
            # Use heuristic based on building type
            return 0.3  # Default 30%
        
        # This is simplified - in practice would use detailed simulation
        # For now, estimate based on daytime consumption pattern
        
        try:
            if building_id in temporal_data.columns:
                building_profile = temporal_data[building_id]
            else:
                # Use index if available
                if building_id in self.building_id_to_index:
                    idx = self.building_id_to_index[building_id]
                    if idx < temporal_data.shape[1]:
                        building_profile = temporal_data.iloc[:, idx]
                    else:
                        return 0.3
                else:
                    return 0.3
            
            # Get daytime consumption (9 AM to 5 PM)
            daytime_mask = (temporal_data.index.hour >= 9) & (temporal_data.index.hour <= 17)
            daytime_consumption = building_profile[daytime_mask].mean()
            total_consumption = building_profile.mean()
            
            if total_consumption > 0:
                daytime_ratio = daytime_consumption / total_consumption
                
                # Higher daytime ratio = higher self-consumption
                return min(0.7, daytime_ratio * 1.2)
            
        except Exception as e:
            logger.warning(f"Error estimating self-consumption: {e}")
        
        return 0.3
    
    def _calculate_co2_reduction(self, annual_generation_kwh: float) -> float:
        """
        Calculate CO2 reduction in tons per year
        """
        # Average CO2 emissions per kWh (European grid mix)
        co2_per_kwh = 0.3  # kg CO2/kWh
        
        annual_co2_reduction_kg = annual_generation_kwh * co2_per_kwh
        annual_co2_reduction_tons = annual_co2_reduction_kg / 1000
        
        return annual_co2_reduction_tons
    
    def _optimize_cluster_solar(self,
                               candidates: List[SolarCandidate],
                               cluster_buildings: List[int],
                               buildings: Dict,
                               temporal_data: Optional[pd.DataFrame]) -> Dict:
        """
        Optimize solar placement within a cluster considering interactions
        """
        if not candidates:
            return {
                'recommended_installations': [],
                'total_capacity_kw': 0,
                'total_investment': 0,
                'expected_self_sufficiency': 0
            }
        
        # Sort by total score
        sorted_candidates = sorted(candidates, key=lambda x: x.total_score, reverse=True)
        
        # Calculate cluster consumption
        cluster_consumption_annual = sum(
            buildings.get(b_id, {}).get('avg_demand_kw', 10) * 24 * 365
            for b_id in cluster_buildings
        )
        
        # Greedy selection with diminishing returns
        selected = []
        total_capacity = 0
        total_generation = 0
        total_investment = 0
        
        for candidate in sorted_candidates:
            # Check if adding this installation improves cluster metrics
            marginal_benefit = self._calculate_marginal_benefit(
                candidate,
                selected,
                cluster_consumption_annual
            )
            
            if marginal_benefit > 0.3:  # Threshold for acceptance
                selected.append(candidate)
                total_capacity += candidate.recommended_capacity_kw
                total_generation += candidate.expected_generation_annual_kwh
                total_investment += candidate.installation_cost
                
                # Stop if we've covered enough of cluster consumption
                if total_generation > cluster_consumption_annual * 0.8:
                    break
        
        # Calculate expected self-sufficiency
        expected_self_sufficiency = min(1.0, total_generation / cluster_consumption_annual)
        
        return {
            'recommended_installations': selected,
            'total_capacity_kw': total_capacity,
            'total_investment': total_investment,
            'expected_self_sufficiency': expected_self_sufficiency,
            'expected_annual_generation_kwh': total_generation,
            'cluster_consumption_annual_kwh': cluster_consumption_annual
        }
    
    def _calculate_marginal_benefit(self,
                                   candidate: SolarCandidate,
                                   already_selected: List[SolarCandidate],
                                   cluster_consumption: float) -> float:
        """
        Calculate marginal benefit of adding this solar installation
        """
        # Current generation from selected
        current_generation = sum(c.expected_generation_annual_kwh for c in already_selected)
        
        # New generation with this candidate
        new_generation = current_generation + candidate.expected_generation_annual_kwh
        
        # Calculate improvement in self-sufficiency
        current_self_sufficiency = min(1.0, current_generation / cluster_consumption)
        new_self_sufficiency = min(1.0, new_generation / cluster_consumption)
        
        improvement = new_self_sufficiency - current_self_sufficiency
        
        # Apply diminishing returns
        if len(already_selected) > 0:
            improvement *= (1.0 - len(already_selected) * 0.1)
        
        return improvement
    
    def _rank_candidates_globally(self,
                                 candidates: List[SolarCandidate]) -> List[SolarCandidate]:
        """
        Rank all candidates across all clusters
        """
        # Sort by total score
        ranked = sorted(candidates, key=lambda x: x.total_score, reverse=True)
        
        # Add ranking
        for i, candidate in enumerate(ranked):
            candidate.global_rank = i + 1
        
        return ranked
    
    def _calculate_system_metrics(self,
                                 candidates: List[SolarCandidate],
                                 buildings: Dict,
                                 clusters: List[List[int]]) -> Dict:
        """
        Calculate system-wide metrics
        """
        total_buildings = len(buildings)
        buildings_with_solar = sum(1 for b in buildings.values() if b.get('has_solar', False))
        recommended_new = len(candidates)
        
        # Calculate totals
        total_new_capacity = sum(c.recommended_capacity_kw for c in candidates)
        total_investment = sum(c.installation_cost for c in candidates)
        total_generation = sum(c.expected_generation_annual_kwh for c in candidates)
        total_co2_reduction = sum(c.co2_reduction_tons for c in candidates)
        avg_roi = np.mean([c.expected_roi_years for c in candidates]) if candidates else 0
        
        # Calculate system consumption
        system_consumption = sum(
            b.get('avg_demand_kw', 10) * 24 * 365
            for b in buildings.values()
        )
        
        # Current and projected solar coverage
        current_coverage = buildings_with_solar / total_buildings if total_buildings > 0 else 0
        projected_coverage = (buildings_with_solar + recommended_new) / total_buildings if total_buildings > 0 else 0
        
        return {
            'total_buildings': total_buildings,
            'current_solar_buildings': buildings_with_solar,
            'recommended_new_installations': recommended_new,
            'total_new_capacity_kw': total_new_capacity,
            'total_investment_required': total_investment,
            'expected_annual_generation_kwh': total_generation,
            'expected_co2_reduction_tons': total_co2_reduction,
            'average_roi_years': avg_roi,
            'current_solar_coverage': current_coverage,
            'projected_solar_coverage': projected_coverage,
            'system_consumption_annual_kwh': system_consumption,
            'projected_renewable_fraction': (total_generation / system_consumption) if system_consumption > 0 else 0
        }
    
    def _generate_recommendations(self, candidates: List[SolarCandidate]) -> List[Dict]:
        """
        Generate actionable recommendations
        """
        recommendations = []
        
        # Top priority installations
        top_candidates = candidates[:10] if len(candidates) >= 10 else candidates
        
        for candidate in top_candidates:
            rec = {
                'building_id': candidate.building_id,
                'cluster_id': candidate.cluster_id,
                'action': 'Install Solar PV',
                'capacity_kw': candidate.recommended_capacity_kw,
                'investment': candidate.installation_cost,
                'annual_savings': candidate.annual_savings,
                'roi_years': candidate.expected_roi_years,
                'co2_reduction_tons': candidate.co2_reduction_tons,
                'priority': 'HIGH' if candidate.total_score > 0.8 else 'MEDIUM',
                'notes': []
            }
            
            # Add specific notes
            if candidate.complementarity_score > 0.7:
                rec['notes'].append('Excellent complementarity with cluster')
            
            if candidate.expected_roi_years < 6:
                rec['notes'].append('Fast payback period')
            
            if candidate.expected_self_consumption_ratio > 0.5:
                rec['notes'].append('High self-consumption expected')
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_summary(self,
                         candidates: List[SolarCandidate],
                         system_metrics: Dict) -> Dict:
        """
        Generate summary statistics
        """
        if not candidates:
            return {
                'status': 'No viable solar candidates found',
                'total_candidates': 0
            }
        
        return {
            'status': 'Solar optimization complete',
            'total_candidates': len(candidates),
            'high_priority_candidates': sum(1 for c in candidates if c.total_score > 0.8),
            'total_capacity_recommended_kw': system_metrics['total_new_capacity_kw'],
            'total_investment_required': system_metrics['total_investment_required'],
            'average_roi_years': system_metrics['average_roi_years'],
            'expected_co2_reduction_tons_annual': system_metrics['expected_co2_reduction_tons'],
            'projected_renewable_fraction': system_metrics['projected_renewable_fraction']
        }
    
    def evaluate(self, ground_truth: Optional[Dict] = None) -> Dict:
        """
        Evaluate solar optimization performance
        """
        if not self.candidates:
            return {}
        
        metrics = {
            'coverage': {
                'buildings_analyzed': len(self.candidates),
                'clusters_covered': len(self.cluster_solar_plans),
                'avg_candidates_per_cluster': np.mean([
                    len(plan['recommended_installations'])
                    for plan in self.cluster_solar_plans.values()
                ]) if self.cluster_solar_plans else 0
            },
            'technical': {
                'avg_technical_score': np.mean([c.technical_score for c in self.candidates]),
                'avg_capacity_kw': np.mean([c.recommended_capacity_kw for c in self.candidates]),
                'total_capacity_kw': sum(c.recommended_capacity_kw for c in self.candidates)
            },
            'economic': {
                'avg_roi_years': np.mean([c.expected_roi_years for c in self.candidates]),
                'best_roi_years': min([c.expected_roi_years for c in self.candidates]),
                'total_investment': sum(c.installation_cost for c in self.candidates),
                'total_annual_savings': sum(c.annual_savings for c in self.candidates)
            },
            'environmental': {
                'total_co2_reduction_tons': sum(c.co2_reduction_tons for c in self.candidates),
                'avg_self_consumption': np.mean([c.expected_self_consumption_ratio for c in self.candidates])
            }
        }
        
        # Compare with ground truth if available
        if ground_truth:
            # Implement comparison logic
            pass
        
        return metrics
    
    def save_results(self, filepath: str):
        """Save optimization results to file"""
        import pickle
        
        results = {
            'candidates': self.candidates,
            'cluster_plans': self.cluster_solar_plans,
            'system_metrics': self.system_metrics,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Saved solar optimization results to {filepath}")
    
    def export_to_csv(self, filepath: str):
        """Export recommendations to CSV for reporting"""
        if not self.candidates:
            logger.warning("No candidates to export")
            return
        
        # Convert candidates to DataFrame
        data = []
        for c in self.candidates:
            data.append({
                'Building ID': c.building_id,
                'Cluster': c.cluster_id,
                'Total Score': round(c.total_score, 2),
                'Technical Score': round(c.technical_score, 2),
                'Economic Score': round(c.economic_score, 2),
                'Complementarity Score': round(c.complementarity_score, 2),
                'Recommended Capacity (kW)': round(c.recommended_capacity_kw, 1),
                'Investment (€)': round(c.installation_cost, 0),
                'Annual Savings (€)': round(c.annual_savings, 0),
                'ROI (years)': round(c.expected_roi_years, 1),
                'Annual Generation (kWh)': round(c.expected_generation_annual_kwh, 0),
                'Self-Consumption': f"{c.expected_self_consumption_ratio:.1%}",
                'CO2 Reduction (tons/year)': round(c.co2_reduction_tons, 2)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(self.candidates)} candidates to {filepath}")


