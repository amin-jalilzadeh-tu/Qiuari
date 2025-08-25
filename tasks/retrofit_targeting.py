# tasks/retrofit_targeting.py
"""
Building retrofit prioritization task
Identifies buildings needing energy efficiency improvements
Considers cluster-level impacts and complementarity with renewable installations
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class RetrofitConfig:
    """Configuration for retrofit targeting task"""
    # Technical thresholds
    min_energy_label_for_retrofit: str = 'D'  # Buildings D or worse need retrofit
    max_acceptable_energy_intensity: float = 150.0  # kWh/m²/year
    target_energy_label: str = 'B'  # Target after retrofit
    min_improvement_potential: float = 0.25  # Minimum 25% reduction required
    
    # Building age categories
    age_categories: Dict = None  # Will be set in __post_init__
    
    # Economic parameters
    retrofit_cost_per_m2: Dict = None  # Will be set in __post_init__
    energy_price_kwh: float = 0.25  # €/kWh
    gas_price_kwh: float = 0.08  # €/kWh for heating
    subsidy_rate: float = 0.30  # 30% government subsidy
    discount_rate: float = 0.05  # For NPV calculation
    payback_period_max: int = 15  # Maximum acceptable payback
    
    # Intervention types and impacts
    intervention_impacts: Dict = None  # Will be set in __post_init__
    
    # Cluster optimization
    cluster_coordination: bool = True
    prioritize_worst_performers: bool = True
    consider_thermal_bridges: bool = True  # Adjacent buildings affect each other
    
    def __post_init__(self):
        """Initialize default dictionaries if not provided"""
        if self.age_categories is None:
            self.age_categories = {
                'pre_1945': {'priority': 1.5, 'typical_label': 'F'},
                '1945_1975': {'priority': 1.3, 'typical_label': 'E'},
                '1975_1995': {'priority': 1.1, 'typical_label': 'D'},
                '1995_2010': {'priority': 0.9, 'typical_label': 'C'},
                'post_2010': {'priority': 0.5, 'typical_label': 'B'}
            }
        
        if self.retrofit_cost_per_m2 is None:
            self.retrofit_cost_per_m2 = {
                'light': 50,    # €/m² - Basic insulation, sealing
                'medium': 150,  # €/m² - Full insulation, windows
                'deep': 300,    # €/m² - Complete envelope upgrade
                'nzeb': 500     # €/m² - Nearly Zero Energy Building
            }
        
        if self.intervention_impacts is None:
            self.intervention_impacts = {
                'wall_insulation': {'reduction': 0.25, 'cost_m2': 80},
                'roof_insulation': {'reduction': 0.20, 'cost_m2': 60},
                'floor_insulation': {'reduction': 0.10, 'cost_m2': 40},
                'window_upgrade': {'reduction': 0.15, 'cost_m2': 200},
                'air_sealing': {'reduction': 0.10, 'cost_m2': 20},
                'hvac_upgrade': {'reduction': 0.20, 'cost_m2': 100}
            }


@dataclass
class RetrofitCandidate:
    """Represents a building candidate for retrofit"""
    building_id: int
    cluster_id: int
    current_label: str
    target_label: str
    building_age: str
    area_m2: float
    
    # Energy metrics
    current_energy_intensity: float  # kWh/m²/year
    target_energy_intensity: float
    potential_reduction_percent: float
    annual_energy_savings_kwh: float
    
    # Scores
    urgency_score: float  # How badly it needs retrofit
    impact_score: float   # Cluster-level impact
    feasibility_score: float  # Technical/economic feasibility
    total_score: float
    
    # Retrofit details
    retrofit_level: str  # light/medium/deep/nzeb
    recommended_interventions: List[str]
    total_cost: float
    subsidy_amount: float
    net_cost: float
    
    # Economic metrics
    annual_cost_savings: float
    simple_payback_years: float
    npv_20_years: float
    co2_reduction_tons_annual: float
    
    # Cluster effects
    thermal_bridge_neighbors: List[int]
    cluster_benefit_factor: float


class RetrofitTargeting:
    """
    Retrofit targeting and prioritization task
    Works with clustering results to identify energy efficiency opportunities
    """
    
    def __init__(self, model, config: Union[Dict, RetrofitConfig]):
        """
        Initialize retrofit targeting task
        
        Args:
            model: Trained GNN model (can be None for rule-based)
            config: Task configuration
        """
        self.model = model
        
        # Parse configuration
        if isinstance(config, dict):
            self.config = RetrofitConfig(**config.get('retrofit_targeting', {}))
        else:
            self.config = config
        
        # Storage for results
        self.candidates = []
        self.cluster_retrofit_plans = {}
        self.intervention_packages = {}
        self.system_metrics = {}
        
        # Energy label to intensity mapping (kWh/m²/year)
        self.label_to_intensity = {
            'A': 50, 'B': 100, 'C': 150,
            'D': 200, 'E': 250, 'F': 300, 'G': 400
        }
        
        # Heating system efficiency factors
        self.heating_efficiency = {
            'gas_boiler': 0.85,
            'oil_boiler': 0.75,
            'electric': 1.0,
            'heat_pump': 3.0,  # COP of 3
            'district_heating': 0.90,
            'unknown': 0.80
        }
        
        logger.info(f"Initialized RetrofitTargeting with config")
    
    def run(self,
            graph_data: Dict,
            clustering_results: Dict,
            solar_results: Optional[Dict] = None,
            embeddings: Optional[torch.Tensor] = None,
            temporal_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run retrofit targeting analysis
        
        Args:
            graph_data: Building and grid information
            clustering_results: Results from clustering task
            solar_results: Results from solar optimization (optional)
            embeddings: Building embeddings from GNN
            temporal_data: Energy consumption profiles
            
        Returns:
            Retrofit targeting results with prioritized candidates
        """
        logger.info("Running retrofit targeting analysis...")
        
        # Extract data
        buildings = graph_data.get('buildings', {})
        clusters = clustering_results.get('clusters', [])
        
        # Analyze each cluster
        all_candidates = []
        cluster_plans = {}
        
        for cluster_id, cluster_buildings in enumerate(clusters):
            logger.info(f"Analyzing cluster {cluster_id} for retrofit opportunities")
            
            # Identify retrofit candidates in cluster
            cluster_candidates = self._identify_retrofit_candidates(
                cluster_buildings,
                buildings,
                cluster_id,
                temporal_data
            )
            
            # Analyze thermal bridges (adjacent buildings)
            cluster_candidates = self._analyze_thermal_bridges(
                cluster_candidates,
                buildings,
                clustering_results
            )
            
            # Create optimized retrofit plan for cluster
            cluster_plan = self._optimize_cluster_retrofits(
                cluster_candidates,
                cluster_buildings,
                buildings,
                solar_results
            )
            
            all_candidates.extend(cluster_candidates)
            cluster_plans[cluster_id] = cluster_plan
        
        # Rank all candidates globally
        ranked_candidates = self._rank_candidates_globally(all_candidates)
        
        # Create intervention packages
        intervention_packages = self._create_intervention_packages(
            ranked_candidates,
            buildings
        )
        
        # Calculate system-wide metrics
        system_metrics = self._calculate_system_metrics(
            ranked_candidates,
            buildings,
            clusters
        )
        
        # Store results
        self.candidates = ranked_candidates
        self.cluster_retrofit_plans = cluster_plans
        self.intervention_packages = intervention_packages
        self.system_metrics = system_metrics
        
        # Prepare output
        return {
            'candidates': ranked_candidates,
            'cluster_plans': cluster_plans,
            'intervention_packages': intervention_packages,
            'system_metrics': system_metrics,
            'recommendations': self._generate_recommendations(ranked_candidates),
            'summary': self._generate_summary(ranked_candidates, system_metrics)
        }
    
    def _identify_retrofit_candidates(self,
                                     cluster_buildings: List[int],
                                     buildings: Dict,
                                     cluster_id: int,
                                     temporal_data: Optional[pd.DataFrame]) -> List[RetrofitCandidate]:
        """
        Identify buildings needing retrofit in a cluster
        """
        candidates = []
        
        for building_id in cluster_buildings:
            if building_id not in buildings:
                continue
            
            building = buildings[building_id]
            
            # Get current energy label
            current_label = building.get('energy_label_simple', 
                                        building.get('energy_label', 'E'))
            if len(current_label) > 1:
                current_label = current_label[0]  # Take first character
            
            # Check if needs retrofit
            if not self._needs_retrofit(building, current_label):
                continue
            
            # Calculate current energy intensity
            current_intensity = self._calculate_energy_intensity(
                building, temporal_data, building_id
            )
            
            # Determine building age category
            building_age = self._determine_age_category(building)
            
            # Calculate urgency score
            urgency_score = self._calculate_urgency_score(
                current_label, current_intensity, building_age
            )
            
            # Determine retrofit level needed
            retrofit_level = self._determine_retrofit_level(
                current_label, self.config.target_energy_label
            )
            
            # Calculate target intensity
            target_intensity = self.label_to_intensity.get(
                self.config.target_energy_label, 100
            )
            
            # Calculate potential reduction
            reduction_percent = (current_intensity - target_intensity) / current_intensity
            reduction_percent = max(0, min(1, reduction_percent))
            
            # Calculate energy savings
            area = building.get('area', 100)
            annual_consumption = current_intensity * area
            annual_savings_kwh = annual_consumption * reduction_percent
            
            # Identify recommended interventions
            interventions = self._identify_interventions(
                building, current_label, retrofit_level
            )
            
            # Calculate costs
            base_cost = self._calculate_retrofit_cost(
                area, retrofit_level, interventions
            )
            subsidy = base_cost * self.config.subsidy_rate
            net_cost = base_cost - subsidy
            
            # Calculate economic metrics
            annual_cost_savings = self._calculate_annual_savings(
                annual_savings_kwh, building
            )
            
            payback_years = net_cost / annual_cost_savings if annual_cost_savings > 0 else 999
            
            npv = self._calculate_npv(
                net_cost, annual_cost_savings, 20, self.config.discount_rate
            )
            
            # Calculate CO2 reduction
            co2_reduction = self._calculate_co2_reduction(annual_savings_kwh, building)
            
            # Calculate impact score (cluster-level benefit)
            impact_score = self._calculate_impact_score(
                building_id, cluster_buildings, buildings, reduction_percent
            )
            
            # Calculate feasibility score
            feasibility_score = self._calculate_feasibility_score(
                payback_years, retrofit_level, building
            )
            
            # Calculate total score
            total_score = self._calculate_total_score(
                urgency_score, impact_score, feasibility_score
            )
            
            # Create candidate
            candidate = RetrofitCandidate(
                building_id=building_id,
                cluster_id=cluster_id,
                current_label=current_label,
                target_label=self.config.target_energy_label,
                building_age=building_age,
                area_m2=area,
                current_energy_intensity=current_intensity,
                target_energy_intensity=target_intensity,
                potential_reduction_percent=reduction_percent,
                annual_energy_savings_kwh=annual_savings_kwh,
                urgency_score=urgency_score,
                impact_score=impact_score,
                feasibility_score=feasibility_score,
                total_score=total_score,
                retrofit_level=retrofit_level,
                recommended_interventions=interventions,
                total_cost=base_cost,
                subsidy_amount=subsidy,
                net_cost=net_cost,
                annual_cost_savings=annual_cost_savings,
                simple_payback_years=payback_years,
                npv_20_years=npv,
                co2_reduction_tons_annual=co2_reduction,
                thermal_bridge_neighbors=[],
                cluster_benefit_factor=1.0
            )
            
            candidates.append(candidate)
        
        return candidates
    
    def _needs_retrofit(self, building: Dict, current_label: str) -> bool:
        """
        Check if building needs retrofit
        """
        # Check energy label
        if current_label >= self.config.min_energy_label_for_retrofit:
            return True
        
        # Check energy intensity if available
        intensity = building.get('energy_intensity_kwh_m2', 0)
        if intensity > self.config.max_acceptable_energy_intensity:
            return True
        
        # Check building age
        year = building.get('building_year', 2000)
        if year < 1990:  # Older buildings likely need retrofit
            return True
        
        return False
    
    def _calculate_energy_intensity(self,
                                   building: Dict,
                                   temporal_data: Optional[pd.DataFrame],
                                   building_id: int) -> float:
        """
        Calculate current energy intensity (kWh/m²/year)
        """
        # Check if already calculated
        if 'energy_intensity_kwh_m2' in building:
            return building['energy_intensity_kwh_m2']
        
        area = building.get('area', 100)
        
        # Try to get from temporal data
        if temporal_data is not None:
            try:
                if building_id in temporal_data.columns:
                    # Average consumption in kW
                    avg_kw = temporal_data[building_id].mean()
                    # Convert to annual kWh
                    annual_kwh = avg_kw * 8760
                elif hasattr(self, 'building_id_to_index') and self.building_id_to_index:
                    idx = self.building_id_to_index.get(building_id)
                    if idx is not None and idx < temporal_data.shape[1]:
                        avg_kw = temporal_data.iloc[:, idx].mean()
                        annual_kwh = avg_kw * 8760
                else:
                    # Fallback
                    annual_kwh = building.get('annual_electricity_kwh', 0) + \
                                building.get('annual_heating_kwh', 0)
                    if annual_kwh == 0:
                        # Estimate from demand
                        avg_demand = building.get('avg_electricity_demand_kw', 10)
                        avg_heating = building.get('avg_heating_demand_kw', 5)
                        annual_kwh = (avg_demand + avg_heating) * 8760
                
                return annual_kwh / area if area > 0 else 200
                
            except Exception as e:
                logger.warning(f"Error calculating intensity from temporal data: {e}")
        
        # Estimate based on energy label
        label = building.get('energy_label_simple', 'E')
        if len(label) > 1:
            label = label[0]
        
        return self.label_to_intensity.get(label, 250)
    
    def _determine_age_category(self, building: Dict) -> str:
        """
        Determine building age category
        """
        year = building.get('building_year')
        
        if not year:
            # Try to infer from age_range
            age_range = building.get('age_range', '')
            if 'pre' in age_range.lower() or '1945' in age_range:
                return 'pre_1945'
            elif '1975' in age_range:
                return '1945_1975'
            elif '1995' in age_range:
                return '1975_1995'
            elif '2010' in age_range:
                return '1995_2010'
            else:
                return '1975_1995'  # Default assumption
        
        if year < 1945:
            return 'pre_1945'
        elif year < 1975:
            return '1945_1975'
        elif year < 1995:
            return '1975_1995'
        elif year < 2010:
            return '1995_2010'
        else:
            return 'post_2010'
    
    def _calculate_urgency_score(self,
                                current_label: str,
                                intensity: float,
                                age_category: str) -> float:
        """
        Calculate how urgently building needs retrofit
        """
        score = 0.0
        
        # Energy label component (0-0.4)
        label_scores = {'G': 0.4, 'F': 0.35, 'E': 0.3, 'D': 0.25, 
                       'C': 0.15, 'B': 0.05, 'A': 0}
        score += label_scores.get(current_label, 0.2)
        
        # Energy intensity component (0-0.3)
        if intensity > 300:
            score += 0.3
        elif intensity > 250:
            score += 0.25
        elif intensity > 200:
            score += 0.2
        elif intensity > 150:
            score += 0.1
        
        # Age component (0-0.3)
        age_priority = self.config.age_categories.get(
            age_category, {}
        ).get('priority', 1.0)
        score += min(0.3, age_priority * 0.2)
        
        return min(1.0, score)
    
    def _determine_retrofit_level(self,
                                 current_label: str,
                                 target_label: str) -> str:
        """
        Determine required retrofit level
        """
        label_values = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 
                       'E': 3, 'F': 2, 'G': 1}
        
        current_val = label_values.get(current_label, 3)
        target_val = label_values.get(target_label, 6)
        
        improvement_needed = target_val - current_val
        
        if improvement_needed >= 4:
            return 'nzeb'  # Nearly Zero Energy Building
        elif improvement_needed >= 3:
            return 'deep'
        elif improvement_needed >= 2:
            return 'medium'
        else:
            return 'light'
    
    def _identify_interventions(self,
                               building: Dict,
                               current_label: str,
                               retrofit_level: str) -> List[str]:
        """
        Identify specific interventions needed
        """
        interventions = []
        
        # Check insulation quality
        insulation = building.get('insulation_quality', 'poor')
        
        if retrofit_level in ['deep', 'nzeb']:
            # Complete envelope upgrade
            interventions.extend([
                'wall_insulation',
                'roof_insulation',
                'floor_insulation',
                'window_upgrade',
                'air_sealing'
            ])
            
            # Check heating system
            heating = building.get('heating_system', 'gas_boiler')
            if 'heat_pump' not in heating.lower():
                interventions.append('hvac_upgrade')
                
        elif retrofit_level == 'medium':
            # Partial upgrade
            interventions.extend(['wall_insulation', 'roof_insulation'])
            
            if current_label in ['F', 'G']:
                interventions.append('window_upgrade')
            
            interventions.append('air_sealing')
            
        else:  # light
            # Basic improvements
            interventions.extend(['roof_insulation', 'air_sealing'])
            
            if insulation == 'poor':
                interventions.append('wall_insulation')
        
        return interventions
    
    def _calculate_retrofit_cost(self,
                                area: float,
                                level: str,
                                interventions: List[str]) -> float:
        """
        Calculate total retrofit cost
        """
        # Base cost by level
        base_cost_m2 = self.config.retrofit_cost_per_m2.get(level, 150)
        base_cost = area * base_cost_m2
        
        # Adjust based on specific interventions
        if interventions:
            intervention_cost = 0
            for intervention in interventions:
                cost_m2 = self.config.intervention_impacts.get(
                    intervention, {}
                ).get('cost_m2', 50)
                intervention_cost += area * cost_m2
            
            # Average of base and intervention-based estimates
            return (base_cost + intervention_cost) / 2
        
        return base_cost
    
    def _calculate_annual_savings(self,
                                 energy_savings_kwh: float,
                                 building: Dict) -> float:
        """
        Calculate annual cost savings from energy reduction
        """
        # Split between electricity and heating
        heating_fraction = 0.7  # Assume 70% is heating
        electricity_fraction = 0.3
        
        heating_savings = energy_savings_kwh * heating_fraction
        electricity_savings = energy_savings_kwh * electricity_fraction
        
        # Get energy prices
        electricity_price = self.config.energy_price_kwh
        
        # Heating price depends on system
        heating_system = building.get('heating_system', 'gas_boiler')
        if 'gas' in heating_system.lower():
            heating_price = self.config.gas_price_kwh
        elif 'heat_pump' in heating_system.lower():
            heating_price = electricity_price / 3  # COP of 3
        else:
            heating_price = electricity_price
        
        annual_savings = (electricity_savings * electricity_price +
                         heating_savings * heating_price)
        
        return annual_savings
    
    def _calculate_npv(self,
                      initial_cost: float,
                      annual_savings: float,
                      years: int,
                      discount_rate: float) -> float:
        """
        Calculate Net Present Value
        """
        npv = -initial_cost
        
        for year in range(1, years + 1):
            # Assume 2% annual energy price increase
            adjusted_savings = annual_savings * (1.02 ** year)
            discounted_savings = adjusted_savings / ((1 + discount_rate) ** year)
            npv += discounted_savings
        
        return npv
    
    def _calculate_co2_reduction(self,
                                energy_savings_kwh: float,
                                building: Dict) -> float:
        """
        Calculate CO2 reduction in tons per year
        """
        # CO2 factors (kg CO2/kWh)
        electricity_co2 = 0.3  # Grid average
        gas_co2 = 0.2  # Natural gas
        
        # Split savings
        heating_fraction = 0.7
        electricity_fraction = 0.3
        
        heating_system = building.get('heating_system', 'gas_boiler')
        
        if 'gas' in heating_system.lower():
            heating_co2_factor = gas_co2
        else:
            heating_co2_factor = electricity_co2
        
        co2_reduction_kg = (
            energy_savings_kwh * electricity_fraction * electricity_co2 +
            energy_savings_kwh * heating_fraction * heating_co2_factor
        )
        
        return co2_reduction_kg / 1000  # Convert to tons
    
    def _calculate_impact_score(self,
                               building_id: int,
                               cluster_buildings: List[int],
                               buildings: Dict,
                               reduction_percent: float) -> float:
        """
        Calculate cluster-level impact of retrofitting this building
        """
        score = 0.0
        
        # Size of building relative to cluster
        building_area = buildings.get(building_id, {}).get('area', 100)
        cluster_area = sum(
            buildings.get(b, {}).get('area', 100)
            for b in cluster_buildings
        )
        
        size_factor = building_area / cluster_area if cluster_area > 0 else 0.1
        score += size_factor * 0.3
        
        # Energy consumption relative to cluster
        building_demand = buildings.get(building_id, {}).get('avg_demand_kw', 10)
        cluster_demand = sum(
            buildings.get(b, {}).get('avg_demand_kw', 10)
            for b in cluster_buildings
        )
        
        demand_factor = building_demand / cluster_demand if cluster_demand > 0 else 0.1
        score += demand_factor * 0.3
        
        # Potential reduction impact
        score += reduction_percent * 0.4
        
        return min(1.0, score)
    
    def _calculate_feasibility_score(self,
                                    payback_years: float,
                                    retrofit_level: str,
                                    building: Dict) -> float:
        """
        Calculate technical and economic feasibility
        """
        score = 0.0
        
        # Economic feasibility (0-0.5)
        if payback_years <= 5:
            score += 0.5
        elif payback_years <= 8:
            score += 0.4
        elif payback_years <= 12:
            score += 0.3
        elif payback_years <= 15:
            score += 0.2
        else:
            score += 0.1
        
        # Technical feasibility (0-0.3)
        # Easier for newer buildings
        year = building.get('building_year', 1980)
        if year > 2000:
            score += 0.3
        elif year > 1990:
            score += 0.25
        elif year > 1970:
            score += 0.2
        else:
            score += 0.1
        
        # Retrofit level feasibility (0-0.2)
        if retrofit_level == 'light':
            score += 0.2
        elif retrofit_level == 'medium':
            score += 0.15
        elif retrofit_level == 'deep':
            score += 0.1
        else:  # nzeb
            score += 0.05
        
        return min(1.0, score)
    
    def _calculate_total_score(self,
                              urgency: float,
                              impact: float,
                              feasibility: float) -> float:
        """
        Combine scores with weights
        """
        if self.config.prioritize_worst_performers:
            weights = {
                'urgency': 0.5,
                'impact': 0.3,
                'feasibility': 0.2
            }
        else:
            weights = {
                'urgency': 0.3,
                'impact': 0.3,
                'feasibility': 0.4
            }
        
        return (weights['urgency'] * urgency +
                weights['impact'] * impact +
                weights['feasibility'] * feasibility)
    
    def _analyze_thermal_bridges(self,
                                candidates: List[RetrofitCandidate],
                                buildings: Dict,
                                clustering_results: Dict) -> List[RetrofitCandidate]:
        """
        Analyze thermal bridge effects between adjacent buildings
        """
        if not self.config.consider_thermal_bridges:
            return candidates
        
        # Map candidates by building ID
        candidate_map = {c.building_id: c for c in candidates}
        
        for candidate in candidates:
            building_id = candidate.building_id
            building = buildings.get(building_id, {})
            
            # Get adjacent buildings (shared walls)
            adjacent = building.get('shared_walls', [])
            if not adjacent:
                # Try to get from adjacency relationships
                adjacent = []
                # This would need actual adjacency data from clustering
            
            thermal_bridge_neighbors = []
            benefit_factor = 1.0
            
            for neighbor_id in adjacent:
                if neighbor_id in candidate_map:
                    # Neighbor also needs retrofit
                    thermal_bridge_neighbors.append(neighbor_id)
                    benefit_factor += 0.1  # 10% additional benefit
                else:
                    # Neighbor is efficient - less benefit
                    neighbor = buildings.get(neighbor_id, {})
                    neighbor_label = neighbor.get('energy_label_simple', 'E')
                    if neighbor_label <= 'C':
                        benefit_factor -= 0.05  # 5% less benefit
            
            candidate.thermal_bridge_neighbors = thermal_bridge_neighbors
            candidate.cluster_benefit_factor = min(1.5, max(0.5, benefit_factor))
            
            # Adjust scores
            candidate.impact_score *= candidate.cluster_benefit_factor
            candidate.total_score = self._calculate_total_score(
                candidate.urgency_score,
                candidate.impact_score,
                candidate.feasibility_score
            )
        
        return candidates
    
    def _optimize_cluster_retrofits(self,
                                   candidates: List[RetrofitCandidate],
                                   cluster_buildings: List[int],
                                   buildings: Dict,
                                   solar_results: Optional[Dict]) -> Dict:
        """
        Optimize retrofit plan for a cluster
        """
        if not candidates:
            return {
                'recommended_retrofits': [],
                'total_investment': 0,
                'expected_energy_reduction': 0,
                'coordination_benefits': {}
            }
        
        # Sort by total score
        sorted_candidates = sorted(candidates, key=lambda x: x.total_score, reverse=True)
        
        # Check coordination with solar plans
        solar_buildings = set()
        if solar_results and 'candidates' in solar_results:
            solar_buildings = {
                c.building_id for c in solar_results['candidates']
                if c.cluster_id == candidates[0].cluster_id
            }
        
        # Select retrofits considering synergies
        selected = []
        total_investment = 0
        total_energy_reduction = 0
        budget_remaining = float('inf')  # Could add budget constraint
        
        for candidate in sorted_candidates:
            # Check synergies
            synergy_bonus = 1.0
            
            # Retrofit before solar is better
            if candidate.building_id in solar_buildings:
                synergy_bonus *= 1.2
            
            # Adjacent retrofits are better (shared contractors, etc.)
            if any(c.building_id in candidate.thermal_bridge_neighbors for c in selected):
                synergy_bonus *= 1.15
            
            # Adjust score for synergies
            adjusted_score = candidate.total_score * synergy_bonus
            
            if adjusted_score > 0.5 and candidate.net_cost <= budget_remaining:
                selected.append(candidate)
                total_investment += candidate.net_cost
                total_energy_reduction += candidate.annual_energy_savings_kwh
                budget_remaining -= candidate.net_cost
        
        # Calculate coordination benefits
        coordination_benefits = {
            'bulk_discount': len(selected) * 0.02,  # 2% per building
            'shared_contractors': min(0.1, len(selected) * 0.02),
            'thermal_bridge_improvement': sum(
                len(c.thermal_bridge_neighbors) * 0.05 for c in selected
            ) / len(selected) if selected else 0
        }
        
        return {
            'recommended_retrofits': selected,
            'total_investment': total_investment,
            'expected_energy_reduction_kwh': total_energy_reduction,
            'coordination_benefits': coordination_benefits,
            'average_payback_years': np.mean([c.simple_payback_years for c in selected]) if selected else 0,
            'total_co2_reduction_tons': sum(c.co2_reduction_tons_annual for c in selected)
        }
    
    def _rank_candidates_globally(self,
                                 candidates: List[RetrofitCandidate]) -> List[RetrofitCandidate]:
        """
        Rank all candidates across all clusters
        """
        # Sort by total score
        ranked = sorted(candidates, key=lambda x: x.total_score, reverse=True)
        
        # Add ranking
        for i, candidate in enumerate(ranked):
            candidate.global_rank = i + 1
        
        return ranked
    
    def _create_intervention_packages(self,
                                     candidates: List[RetrofitCandidate],
                                     buildings: Dict) -> Dict:
        """
        Create standardized intervention packages
        """
        packages = {
            'emergency': [],  # G and F labels, urgent
            'priority': [],   # E labels, high impact
            'standard': [],   # D labels, good ROI
            'optional': []    # C labels, nice to have
        }
        
        for candidate in candidates:
            if candidate.current_label in ['G', 'F']:
                packages['emergency'].append(candidate)
            elif candidate.current_label == 'E':
                packages['priority'].append(candidate)
            elif candidate.current_label == 'D':
                packages['standard'].append(candidate)
            else:
                packages['optional'].append(candidate)
        
        # Create package summaries
        package_summaries = {}
        for package_name, package_candidates in packages.items():
            if package_candidates:
                package_summaries[package_name] = {
                    'count': len(package_candidates),
                    'total_investment': sum(c.net_cost for c in package_candidates),
                    'total_energy_savings_kwh': sum(c.annual_energy_savings_kwh for c in package_candidates),
                    'average_payback': np.mean([c.simple_payback_years for c in package_candidates]),
                    'total_co2_reduction': sum(c.co2_reduction_tons_annual for c in package_candidates)
                }
        
        return package_summaries
    
    def _calculate_system_metrics(self,
                                 candidates: List[RetrofitCandidate],
                                 buildings: Dict,
                                 clusters: List[List[int]]) -> Dict:
        """
        Calculate system-wide retrofit metrics
        """
        total_buildings = len(buildings)
        
        # Current state
        current_labels = defaultdict(int)
        for building in buildings.values():
            label = building.get('energy_label_simple', 'E')
            if len(label) > 1:
                label = label[0]
            current_labels[label] += 1
        
        # Calculate current inefficient buildings
        inefficient_current = sum(
            count for label, count in current_labels.items()
            if label >= 'D'
        )
        
        # After retrofits
        retrofitted_count = len(candidates)
        inefficient_after = inefficient_current - retrofitted_count
        
        # Energy metrics
        total_energy_reduction = sum(c.annual_energy_savings_kwh for c in candidates)
        total_investment = sum(c.net_cost for c in candidates)
        total_co2_reduction = sum(c.co2_reduction_tons_annual for c in candidates)
        
        # Economic metrics
        avg_payback = np.mean([c.simple_payback_years for c in candidates]) if candidates else 0
        total_npv = sum(c.npv_20_years for c in candidates)
        
        # System consumption (rough estimate)
        system_consumption = sum(
            b.get('avg_demand_kw', 10) * 8760  # kWh/year
            for b in buildings.values()
        )
        
        return {
            'total_buildings': total_buildings,
            'current_inefficient_buildings': inefficient_current,
            'candidates_identified': retrofitted_count,
            'inefficient_after_retrofit': inefficient_after,
            'efficiency_improvement_percent': (inefficient_current - inefficient_after) / inefficient_current * 100 if inefficient_current > 0 else 0,
            'total_investment_required': total_investment,
            'total_energy_reduction_kwh': total_energy_reduction,
            'energy_reduction_percent': total_energy_reduction / system_consumption * 100 if system_consumption > 0 else 0,
            'total_co2_reduction_tons': total_co2_reduction,
            'average_payback_years': avg_payback,
            'total_npv_20_years': total_npv,
            'current_label_distribution': dict(current_labels)
        }
    
    def _generate_recommendations(self, candidates: List[RetrofitCandidate]) -> List[Dict]:
        """
        Generate actionable retrofit recommendations
        """
        recommendations = []
        
        # Top priority retrofits
        top_candidates = candidates[:10] if len(candidates) >= 10 else candidates
        
        for candidate in top_candidates:
            rec = {
                'building_id': candidate.building_id,
                'cluster_id': candidate.cluster_id,
                'action': f'Retrofit from {candidate.current_label} to {candidate.target_label}',
                'retrofit_level': candidate.retrofit_level,
                'interventions': candidate.recommended_interventions,
                'investment': candidate.net_cost,
                'annual_savings': candidate.annual_cost_savings,
                'payback_years': candidate.simple_payback_years,
                'energy_reduction_kwh': candidate.annual_energy_savings_kwh,
                'co2_reduction_tons': candidate.co2_reduction_tons_annual,
                'priority': 'URGENT' if candidate.current_label in ['G', 'F'] else 'HIGH' if candidate.current_label == 'E' else 'MEDIUM',
                'notes': []
            }
            
            # Add specific notes
            if candidate.thermal_bridge_neighbors:
                rec['notes'].append(f'Coordinate with {len(candidate.thermal_bridge_neighbors)} adjacent buildings')
            
            if candidate.simple_payback_years < 7:
                rec['notes'].append('Excellent ROI')
            
            if candidate.potential_reduction_percent > 0.4:
                rec['notes'].append('High energy reduction potential')
            
            recommendations.append(rec)
        
        return recommendations
    
    def _generate_summary(self,
                         candidates: List[RetrofitCandidate],
                         system_metrics: Dict) -> Dict:
        """
        Generate summary statistics
        """
        if not candidates:
            return {
                'status': 'No retrofit candidates identified',
                'total_candidates': 0
            }
        
        # Count by urgency
        urgent_count = sum(1 for c in candidates if c.current_label in ['G', 'F'])
        high_priority = sum(1 for c in candidates if c.current_label == 'E')
        
        return {
            'status': 'Retrofit analysis complete',
            'total_candidates': len(candidates),
            'urgent_retrofits': urgent_count,
            'high_priority_retrofits': high_priority,
            'total_investment_required': system_metrics['total_investment_required'],
            'total_energy_reduction_kwh': system_metrics['total_energy_reduction_kwh'],
            'average_payback_years': system_metrics['average_payback_years'],
            'total_co2_reduction_tons': system_metrics['total_co2_reduction_tons'],
            'efficiency_improvement_percent': system_metrics['efficiency_improvement_percent']
        }
    
    def evaluate(self, ground_truth: Optional[Dict] = None) -> Dict:
        """
        Evaluate retrofit targeting performance
        """
        if not self.candidates:
            return {}
        
        metrics = {
            'coverage': {
                'buildings_analyzed': len(self.candidates),
                'clusters_covered': len(self.cluster_retrofit_plans),
                'urgent_identified': sum(1 for c in self.candidates if c.urgency_score > 0.7)
            },
            'technical': {
                'avg_energy_reduction': np.mean([c.potential_reduction_percent for c in self.candidates]),
                'total_energy_savings_kwh': sum(c.annual_energy_savings_kwh for c in self.candidates),
                'avg_current_intensity': np.mean([c.current_energy_intensity for c in self.candidates]),
                'avg_target_intensity': np.mean([c.target_energy_intensity for c in self.candidates])
            },
            'economic': {
                'total_investment': sum(c.net_cost for c in self.candidates),
                'total_annual_savings': sum(c.annual_cost_savings for c in self.candidates),
                'avg_payback_years': np.mean([c.simple_payback_years for c in self.candidates]),
                'best_payback_years': min([c.simple_payback_years for c in self.candidates]),
                'total_npv': sum(c.npv_20_years for c in self.candidates)
            },
            'environmental': {
                'total_co2_reduction_tons': sum(c.co2_reduction_tons_annual for c in self.candidates),
                'avg_co2_per_building': np.mean([c.co2_reduction_tons_annual for c in self.candidates])
            },
            'intervention_distribution': defaultdict(int)
        }
        
        # Count intervention types
        for candidate in self.candidates:
            for intervention in candidate.recommended_interventions:
                metrics['intervention_distribution'][intervention] += 1
        
        return metrics
    
    def save_results(self, filepath: str):
        """Save retrofit results to file"""
        import pickle
        
        results = {
            'candidates': self.candidates,
            'cluster_plans': self.cluster_retrofit_plans,
            'intervention_packages': self.intervention_packages,
            'system_metrics': self.system_metrics,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Saved retrofit targeting results to {filepath}")
    
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
                'Current Label': c.current_label,
                'Target Label': c.target_label,
                'Building Age': c.building_age,
                'Area (m²)': round(c.area_m2, 0),
                'Current Intensity (kWh/m²/yr)': round(c.current_energy_intensity, 0),
                'Target Intensity (kWh/m²/yr)': round(c.target_energy_intensity, 0),
                'Energy Reduction': f"{c.potential_reduction_percent:.1%}",
                'Retrofit Level': c.retrofit_level,
                'Total Cost (€)': round(c.total_cost, 0),
                'Net Cost (€)': round(c.net_cost, 0),
                'Annual Savings (€)': round(c.annual_cost_savings, 0),
                'Payback (years)': round(c.simple_payback_years, 1),
                'NPV 20yr (€)': round(c.npv_20_years, 0),
                'CO2 Reduction (tons/yr)': round(c.co2_reduction_tons_annual, 2),
                'Total Score': round(c.total_score, 2)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(self.candidates)} retrofit candidates to {filepath}")


# Utility function for testing
def test_retrofit_targeting():
    """Test retrofit targeting with dummy data"""
    
    # Create dummy clustering results
    clustering_results = {
        'clusters': [
            [1, 2, 3, 4, 5],  # Cluster 0
            [6, 7, 8, 9],      # Cluster 1
            [10, 11, 12]       # Cluster 2
        ]
    }
    
    # Create dummy building data
    graph_data = {
        'buildings': {}
    }
    
    energy_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    building_types = ['residential', 'office', 'commercial']
    
    for i in range(1, 13):
        # Make some buildings inefficient
        if i <= 6:
            label = np.random.choice(['E', 'F', 'G'])
        else:
            label = np.random.choice(['C', 'D', 'E'])
        
        graph_data['buildings'][i] = {
            'area': 100 + np.random.rand() * 300,
            'building_year': 1950 + int(np.random.rand() * 70),
            'energy_label_simple': label,
            'building_function': np.random.choice(building_types),
            'avg_electricity_demand_kw': 5 + np.random.rand() * 15,
            'avg_heating_demand_kw': 10 + np.random.rand() * 20,
            'insulation_quality': np.random.choice(['poor', 'average', 'good']),
            'heating_system': np.random.choice(['gas_boiler', 'electric', 'heat_pump']),
            'shared_walls': [i-1, i+1] if 2 <= i <= 11 else []
        }
    
    # Configuration
    config = RetrofitConfig()
    
    # Initialize task
    retrofit_task = RetrofitTargeting(model=None, config=config)
    
    # Run analysis
    results = retrofit_task.run(
        graph_data=graph_data,
        clustering_results=clustering_results
    )
    
    print("Retrofit Targeting Results:")
    print(f"Total candidates: {len(results['candidates'])}")
    print(f"Summary: {results['summary']}")
    
    # Show top recommendations
    if results['recommendations']:
        print("\nTop 3 Recommendations:")
        for i, rec in enumerate(results['recommendations'][:3]):
            print(f"{i+1}. Building {rec['building_id']}: "
                  f"{rec['action']}, "
                  f"Investment: €{rec['investment']:,.0f}, "
                  f"Payback: {rec['payback_years']:.1f} years")
    
    return results


if __name__ == "__main__":
    # Test the implementation
    test_results = test_retrofit_targeting()
    print("\n✅ Retrofit targeting task implementation complete!")