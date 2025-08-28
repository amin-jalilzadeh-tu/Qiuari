# training/solar_learning_loop.py
"""
Solar Installation Learning Loop with Correct Prioritization
Prioritizes poor energy labels (E/F/G) as they benefit most from solar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SolarCandidate:
    """Building candidate for solar installation"""
    building_id: int
    priority_score: float
    roof_area_m2: float
    orientation_factor: float
    energy_label: str
    current_demand_kwh: float
    has_solar: bool
    expected_generation_kwp: float
    expected_roi_years: float
    cascade_impact_score: float
    lv_group_id: int
    cluster_id: int


class SolarLearningLoop:
    """
    Iterative solar deployment with learning feedback
    Prioritizes poor energy labels (E/F/G) correctly
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Energy label scores - POOR LABELS GET HIGHER SCORES
        self.label_scores = {
            'G': 1.0,   # Worst efficiency - highest priority
            'F': 0.9,   # Very poor - high priority
            'E': 0.8,   # Poor - high priority
            'D': 0.6,   # Below average - medium priority
            'C': 0.4,   # Average - lower priority (default for missing)
            'B': 0.2,   # Good - low priority (likely has solar)
            'A': 0.1,   # Excellent - lowest priority (likely has solar)
            'A+': 0.05,
            'A++': 0.02,
            'A+++': 0.01,
            'A++++': 0.0
        }
        
        # Solar parameters
        self.kwp_per_m2 = config.get('kwp_per_m2', 0.15)  # 150W per m²
        self.annual_generation_per_kwp = config.get('annual_generation_per_kwp', 1200)  # kWh/kWp/year
        self.cost_per_kwp = config.get('cost_per_kwp', 1200)  # euros
        self.electricity_price = config.get('electricity_price', 0.25)  # euros/kWh
        self.feed_in_tariff = config.get('feed_in_tariff', 0.08)  # euros/kWh
        
        # Learning parameters
        self.learning_rate = config.get('learning_rate', 0.1)
        self.n_rounds = config.get('n_rounds', 5)
        self.buildings_per_round = config.get('buildings_per_round', 10)
        
        # Tracking
        self.installation_history = []
        self.prediction_errors = []
        self.cumulative_benefits = {
            'energy_generated': 0,
            'co2_reduced': 0,
            'peak_reduction': 0,
            'self_sufficiency_improvement': 0
        }
        
        # Learnable weights
        self.priority_weights = {
            'energy_label': 0.3,  # High weight for poor labels
            'roof_potential': 0.2,
            'cascade_impact': 0.2,
            'demand_match': 0.2,
            'cluster_benefit': 0.1
        }
    
    def calculate_priority_score(self,
                                building_features: Dict,
                                building_id: int,
                                cascade_impact: float,
                                cluster_metrics: Optional[Dict] = None) -> float:
        """
        Calculate solar installation priority
        Poor energy labels (E/F/G) get HIGHER priority
        """
        # Get building features
        energy_label = building_features.get('energy_label', 'C')  # Default C for missing
        roof_area = building_features.get('roof_area', 50)
        orientation = building_features.get('orientation', 'south')
        has_solar = building_features.get('has_solar', False)
        annual_demand = building_features.get('annual_consumption', 5000)
        
        # Already has solar - zero priority
        if has_solar:
            return 0.0
        
        # Energy label score - POOR LABELS GET HIGH SCORES
        label_score = self.label_scores.get(energy_label, 0.4)  # Default 0.4 for unknown
        
        # Roof potential score
        orientation_factors = {
            'south': 1.0, 'south-east': 0.95, 'south-west': 0.95,
            'east': 0.85, 'west': 0.85, 'flat': 0.9,
            'north-east': 0.65, 'north-west': 0.65, 'north': 0.5
        }
        orientation_factor = orientation_factors.get(orientation, 0.8)
        roof_potential = (roof_area / 100) * orientation_factor  # Normalize by 100m²
        roof_potential = min(roof_potential, 1.0)
        
        # Demand match score - buildings with high demand benefit more
        demand_score = min(annual_demand / 10000, 1.0)  # Normalize by 10MWh
        
        # Cluster benefit score
        cluster_score = 0.5  # Default
        if cluster_metrics:
            # Clusters with low self-sufficiency benefit more from solar
            self_sufficiency = cluster_metrics.get('self_sufficiency', 0.5)
            cluster_score = 1.0 - self_sufficiency
        
        # Calculate weighted priority
        priority = (
            self.priority_weights['energy_label'] * label_score +
            self.priority_weights['roof_potential'] * roof_potential +
            self.priority_weights['cascade_impact'] * min(cascade_impact, 1.0) +
            self.priority_weights['demand_match'] * demand_score +
            self.priority_weights['cluster_benefit'] * cluster_score
        )
        
        return priority
    
    def select_candidates(self,
                         building_features: Dict,
                         cascade_impacts: Dict[int, float],
                         cluster_assignments: torch.Tensor,
                         cluster_metrics: Dict,
                         n_candidates: int = 10) -> List[SolarCandidate]:
        """
        Select top candidates for solar installation
        """
        candidates = []
        
        for building_id in range(len(building_features['energy_label'])):
            # Skip if already has solar
            if building_features['has_solar'][building_id]:
                continue
            
            # Get cluster metrics
            cluster_id = cluster_assignments[building_id].item()
            cluster_metric = cluster_metrics.get(cluster_id, {})
            
            # Calculate priority
            features = {
                'energy_label': building_features['energy_label'][building_id],
                'roof_area': building_features['roof_area'][building_id],
                'orientation': building_features.get('orientation', ['south'] * len(building_features['energy_label']))[building_id],
                'has_solar': building_features['has_solar'][building_id],
                'annual_consumption': building_features.get('annual_consumption', [5000] * len(building_features['energy_label']))[building_id]
            }
            
            priority = self.calculate_priority_score(
                features,
                building_id,
                cascade_impacts.get(building_id, 0.5),
                cluster_metric
            )
            
            # Calculate expected generation
            roof_area = features['roof_area']
            orientation_factor = {'south': 1.0, 'east': 0.85, 'west': 0.85}.get(features['orientation'], 0.8)
            expected_kwp = roof_area * self.kwp_per_m2 * orientation_factor
            expected_annual_gen = expected_kwp * self.annual_generation_per_kwp
            
            # Simple ROI calculation
            annual_savings = min(expected_annual_gen, features['annual_consumption']) * self.electricity_price
            annual_export = max(0, expected_annual_gen - features['annual_consumption']) * self.feed_in_tariff
            annual_revenue = annual_savings + annual_export
            installation_cost = expected_kwp * self.cost_per_kwp
            roi_years = installation_cost / (annual_revenue + 1e-6)
            
            candidate = SolarCandidate(
                building_id=building_id,
                priority_score=priority,
                roof_area_m2=roof_area,
                orientation_factor=orientation_factor,
                energy_label=features['energy_label'],
                current_demand_kwh=features['annual_consumption'],
                has_solar=False,
                expected_generation_kwp=expected_kwp,
                expected_roi_years=roi_years,
                cascade_impact_score=cascade_impacts.get(building_id, 0.5),
                lv_group_id=building_features.get('lv_group_id', [0] * len(building_features['energy_label']))[building_id],
                cluster_id=cluster_id
            )
            
            candidates.append(candidate)
        
        # Sort by priority and select top N
        candidates.sort(key=lambda x: x.priority_score, reverse=True)
        
        return candidates[:n_candidates]
    
    def simulate_installation(self,
                            candidate: SolarCandidate,
                            network_state: Dict) -> Dict:
        """
        Simulate the impact of installing solar on a building
        """
        # Calculate direct impact
        direct_generation = candidate.expected_generation_kwp * self.annual_generation_per_kwp
        self_consumption = min(direct_generation, candidate.current_demand_kwh)
        export_to_grid = direct_generation - self_consumption
        
        # Calculate network impact
        cluster_buildings = network_state.get(f'cluster_{candidate.cluster_id}_buildings', [])
        cluster_demand = sum(network_state.get(f'building_{b}_demand', 0) for b in cluster_buildings)
        
        # Energy can be shared within cluster
        cluster_consumption = min(export_to_grid, cluster_demand - self_consumption)
        final_export = export_to_grid - cluster_consumption
        
        # Peak reduction estimate
        peak_reduction = 0.1 * (self_consumption / candidate.current_demand_kwh)
        
        # CO2 reduction (0.5 kg/kWh)
        co2_reduction = direct_generation * 0.5
        
        return {
            'building_id': candidate.building_id,
            'direct_generation': direct_generation,
            'self_consumption': self_consumption,
            'cluster_sharing': cluster_consumption,
            'grid_export': final_export,
            'peak_reduction': peak_reduction,
            'co2_reduction': co2_reduction,
            'roi_years': candidate.expected_roi_years,
            'self_sufficiency_improvement': self_consumption / (candidate.current_demand_kwh + 1e-6)
        }
    
    def update_weights(self, predicted: Dict, actual: Dict):
        """
        Update priority weights based on prediction error
        """
        # Calculate error
        generation_error = abs(predicted['direct_generation'] - actual['direct_generation']) / (actual['direct_generation'] + 1e-6)
        roi_error = abs(predicted['roi_years'] - actual['roi_years']) / (actual['roi_years'] + 1e-6)
        
        avg_error = (generation_error + roi_error) / 2
        self.prediction_errors.append(avg_error)
        
        # Update weights if error is high
        if avg_error > 0.2:  # 20% error threshold
            # Adjust weights based on what performed better than expected
            if actual['self_sufficiency_improvement'] > predicted['self_sufficiency_improvement']:
                # Demand matching was more important
                self.priority_weights['demand_match'] *= (1 + self.learning_rate)
                self.priority_weights['energy_label'] *= (1 - self.learning_rate * 0.5)
            
            if actual['roi_years'] < predicted['roi_years']:
                # Roof potential was more important
                self.priority_weights['roof_potential'] *= (1 + self.learning_rate)
                self.priority_weights['cascade_impact'] *= (1 - self.learning_rate * 0.5)
            
            # Normalize weights
            total = sum(self.priority_weights.values())
            for key in self.priority_weights:
                self.priority_weights[key] /= total
    
    def run_learning_loop(self,
                         building_features: Dict,
                         cascade_analyzer,
                         cluster_assignments: torch.Tensor,
                         cluster_metrics: Dict,
                         network_state: Dict) -> Dict:
        """
        Run complete learning loop with multiple rounds
        """
        logger.info("Starting solar learning loop...")
        
        all_results = []
        installed_buildings = set()
        
        for round_num in range(self.n_rounds):
            logger.info(f"Round {round_num + 1}/{self.n_rounds}")
            
            # Calculate cascade impacts
            cascade_impacts = {}
            for building_id in range(len(building_features['energy_label'])):
                if building_id not in installed_buildings:
                    impact = cascade_analyzer.calculate_impact(
                        building_id, 
                        network_state.get('edge_index'),
                        max_hops=3
                    )
                    cascade_impacts[building_id] = impact
            
            # Select candidates
            candidates = self.select_candidates(
                building_features,
                cascade_impacts,
                cluster_assignments,
                cluster_metrics,
                n_candidates=self.buildings_per_round
            )
            
            round_results = []
            
            for candidate in candidates:
                # Simulate installation
                predicted = self.simulate_installation(candidate, network_state)
                
                # In real deployment, would measure actual performance
                # For simulation, add some noise
                actual = predicted.copy()
                for key in ['direct_generation', 'self_consumption', 'roi_years']:
                    actual[key] *= (1 + np.random.normal(0, 0.1))  # 10% noise
                
                # Update weights based on error
                self.update_weights(predicted, actual)
                
                # Record installation
                self.installation_history.append(candidate)
                installed_buildings.add(candidate.building_id)
                
                # Update cumulative benefits
                self.cumulative_benefits['energy_generated'] += actual['direct_generation']
                self.cumulative_benefits['co2_reduced'] += actual['co2_reduction']
                self.cumulative_benefits['peak_reduction'] += actual['peak_reduction']
                self.cumulative_benefits['self_sufficiency_improvement'] += actual['self_sufficiency_improvement']
                
                # Mark building as having solar
                building_features['has_solar'][candidate.building_id] = True
                
                round_results.append(actual)
            
            all_results.append(round_results)
            
            logger.info(f"Round {round_num + 1} complete: "
                       f"{len(round_results)} installations, "
                       f"Total generation: {sum(r['direct_generation'] for r in round_results):.0f} kWh/year")
        
        return {
            'rounds': all_results,
            'total_installations': len(installed_buildings),
            'cumulative_benefits': self.cumulative_benefits,
            'final_weights': self.priority_weights,
            'avg_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0
        }
    
    def generate_report(self) -> str:
        """Generate learning loop report"""
        report = []
        report.append("=" * 60)
        report.append("SOLAR LEARNING LOOP REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append(f"Total Installations: {len(self.installation_history)}")
        report.append(f"Total Energy Generated: {self.cumulative_benefits['energy_generated']:.0f} kWh/year")
        report.append(f"CO2 Reduced: {self.cumulative_benefits['co2_reduced']:.0f} kg/year")
        report.append(f"Peak Reduction: {self.cumulative_benefits['peak_reduction']:.1%}")
        report.append("")
        
        # Priority distribution by energy label
        label_dist = {}
        for install in self.installation_history:
            label = install.energy_label
            label_dist[label] = label_dist.get(label, 0) + 1
        
        report.append("Installations by Energy Label:")
        for label in ['G', 'F', 'E', 'D', 'C', 'B', 'A']:
            if label in label_dist:
                report.append(f"  {label}: {label_dist[label]} buildings")
        report.append("")
        
        report.append("Final Priority Weights:")
        for key, weight in self.priority_weights.items():
            report.append(f"  {key}: {weight:.2f}")
        
        if self.prediction_errors:
            report.append(f"\nAverage Prediction Error: {np.mean(self.prediction_errors):.1%}")
        
        return "\n".join(report)