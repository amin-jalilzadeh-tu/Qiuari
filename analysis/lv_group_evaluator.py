"""
LV Group Evaluation Module
Evaluates and scores LV groups based on their suitability for:
1. Energy community formation (complementarity potential)
2. Intervention targeting (retrofit priorities)
3. Grid optimization opportunities
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class DiversityMetrics:
    """Metrics for assessing diversity within an LV group"""
    building_type_diversity: float  # 0-1, variety of building types
    temporal_diversity: float  # 0-1, variation in peak times
    generation_diversity: float  # 0-1, mix of prosumers/consumers
    size_diversity: float  # 0-1, variation in building sizes
    energy_label_diversity: float  # 0-1, variation in efficiency levels
    occupancy_diversity: float  # 0-1, different occupancy patterns
    
    @property
    def overall_diversity_index(self) -> float:
        """Calculate overall diversity index (DI)"""
        weights = {
            'building_type': 0.25,
            'temporal': 0.20,
            'generation': 0.20,
            'size': 0.10,
            'label': 0.15,
            'occupancy': 0.10
        }
        
        di = (
            weights['building_type'] * self.building_type_diversity +
            weights['temporal'] * self.temporal_diversity +
            weights['generation'] * self.generation_diversity +
            weights['size'] * self.size_diversity +
            weights['label'] * self.energy_label_diversity +
            weights['occupancy'] * self.occupancy_diversity
        )
        
        return di * 10  # Scale to 0-10

@dataclass
class InterventionMetrics:
    """Metrics for assessing intervention opportunities"""
    retrofit_potential: float  # 0-1, buildings needing upgrades
    solar_potential: float  # 0-1, available roof area vs current
    heat_pump_suitability: float  # 0-1, buildings suitable for HP
    battery_opportunity: float  # 0-1, value from storage
    urgent_upgrades: int  # Count of E/F/G labels
    economic_viability: float  # 0-1, ROI potential
    
    @property
    def intervention_priority_score(self) -> float:
        """Calculate intervention priority (0-10)"""
        weights = {
            'retrofit': 0.30,
            'solar': 0.25,
            'heat_pump': 0.20,
            'battery': 0.15,
            'urgency': 0.10
        }
        
        urgency_factor = min(self.urgent_upgrades / 10, 1.0)
        
        score = (
            weights['retrofit'] * self.retrofit_potential +
            weights['solar'] * self.solar_potential +
            weights['heat_pump'] * self.heat_pump_suitability +
            weights['battery'] * self.battery_opportunity +
            weights['urgency'] * urgency_factor
        )
        
        return score * 10 * self.economic_viability

@dataclass
class GridMetrics:
    """Metrics for grid optimization potential"""
    peak_coincidence: float  # Current peak overlap (lower is better)
    transformer_loading: float  # Current utilization
    line_losses: float  # Technical losses
    voltage_stability: float  # Voltage deviation
    network_constraints: int  # Number of bottlenecks
    
    @property
    def grid_optimization_potential(self) -> float:
        """Calculate grid optimization potential (0-10)"""
        # Higher coincidence = more potential for reduction
        coincidence_potential = self.peak_coincidence
        
        # Medium loading is ideal (room for growth but not underutilized)
        loading_score = 1.0 - abs(0.7 - self.transformer_loading)
        
        # Higher losses = more improvement potential
        loss_potential = min(self.line_losses / 0.1, 1.0)
        
        # Poor stability = high improvement potential
        stability_potential = 1.0 - self.voltage_stability
        
        score = (
            0.4 * coincidence_potential +
            0.2 * loading_score +
            0.2 * loss_potential +
            0.2 * stability_potential
        )
        
        return score * 10

class LVGroupEvaluator:
    """Evaluates LV groups for suitability and intervention opportunities"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Energy label mappings (Netherlands specific)
        self.label_consumption = {
            'A++++': 15, 'A+++': 20, 'A++': 30, 'A+': 35,
            'A': 40, 'B': 50, 'C': 60, 'D': 75,
            'E': 90, 'F': 100, 'G': 110  # W/m²
        }
        
        # Building type categories
        self.building_types = {
            'residential_single': 0,
            'residential_multi': 1,
            'commercial_office': 2,
            'commercial_retail': 3,
            'industrial': 4,
            'educational': 5,
            'healthcare': 6,
            'mixed_use': 7
        }
        
    def _default_config(self) -> Dict:
        return {
            'min_buildings': 3,
            'max_buildings': 50,
            'min_diversity_score': 5.0,
            'intervention_threshold': 6.0,
            'grid_optimization_threshold': 5.0
        }
    
    def evaluate_lv_group(self, lv_group_data: Dict) -> Dict:
        """
        Comprehensive evaluation of an LV group
        
        Args:
            lv_group_data: Dictionary containing building data and network info
            
        Returns:
            Evaluation report with scores and recommendations
        """
        buildings = lv_group_data.get('buildings', [])
        
        if len(buildings) < self.config['min_buildings']:
            return {
                'lv_group_id': lv_group_data.get('id'),
                'status': 'insufficient_buildings',
                'building_count': len(buildings),
                'evaluation': None
            }
        
        # Calculate all metrics
        diversity = self._calculate_diversity_metrics(buildings)
        intervention = self._calculate_intervention_metrics(buildings)
        grid = self._calculate_grid_metrics(lv_group_data)
        
        # Calculate suitability scores
        complementarity_score = self._calculate_complementarity_suitability(
            diversity, grid
        )
        
        # Determine classification
        classification = self._classify_lv_group(
            diversity, intervention, grid, complementarity_score
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            diversity, intervention, grid, classification
        )
        
        # Enhanced district assessment if building data is available
        district_assessment = None
        if lv_group_data.get('buildings_data'):
            district_assessment = self.assess_district_characteristics(lv_group_data['buildings_data'])
        
        evaluation_result = {
            'lv_group_id': lv_group_data.get('id'),
            'status': 'evaluated',
            'building_count': len(buildings),
            'buildings_data': buildings,  # Include for report generation
            'evaluation': {
                'diversity_metrics': diversity.__dict__,
                'intervention_metrics': intervention.__dict__,
                'grid_metrics': grid.__dict__,
                'scores': {
                    'diversity_index': diversity.overall_diversity_index,
                    'intervention_priority': intervention.intervention_priority_score,
                    'grid_optimization_potential': grid.grid_optimization_potential,
                    'complementarity_suitability': complementarity_score
                },
                'classification': classification,
                'recommendations': recommendations
            }
        }
        
        # Add district assessment if available
        if district_assessment:
            evaluation_result['evaluation']['district_assessment'] = district_assessment
            # Update recommendations based on district type
            enhanced_recommendations = self._generate_district_specific_recommendations(
                district_assessment, diversity, intervention, grid
            )
            evaluation_result['evaluation']['recommendations'].extend(enhanced_recommendations)
        
        return evaluation_result
    
    def _calculate_diversity_metrics(self, buildings: List[Dict]) -> DiversityMetrics:
        """Calculate diversity metrics for buildings"""
        
        # Building type diversity (Shannon entropy)
        types = [b.get('type', 'residential_single') for b in buildings]
        type_counts = pd.Series(types).value_counts()
        type_entropy = -sum((c/len(types)) * np.log(c/len(types) + 1e-10) 
                           for c in type_counts)
        type_diversity = type_entropy / np.log(len(self.building_types))
        
        # Temporal diversity (peak hour variation)
        peak_hours = [b.get('peak_hour', 18) for b in buildings]
        hour_std = np.std(peak_hours) / 12  # Normalize by half day
        temporal_diversity = min(hour_std, 1.0)
        
        # Generation diversity (prosumer ratio)
        has_solar = sum(1 for b in buildings if b.get('has_solar', False))
        prosumer_ratio = has_solar / len(buildings)
        generation_diversity = 2 * prosumer_ratio * (1 - prosumer_ratio)  # Max at 50/50
        
        # Size diversity
        areas = [b.get('area', 100) for b in buildings]
        size_cv = np.std(areas) / (np.mean(areas) + 1e-10)  # Coefficient of variation
        size_diversity = min(size_cv, 1.0)
        
        # Energy label diversity
        labels = [b.get('energy_label', 'D') for b in buildings]
        label_counts = pd.Series(labels).value_counts()
        label_entropy = -sum((c/len(labels)) * np.log(c/len(labels) + 1e-10) 
                            for c in label_counts)
        label_diversity = label_entropy / np.log(7)  # 7 main label categories
        
        # Occupancy diversity
        occupancies = [b.get('occupancy', 2) for b in buildings]
        occ_cv = np.std(occupancies) / (np.mean(occupancies) + 1e-10)
        occupancy_diversity = min(occ_cv, 1.0)
        
        return DiversityMetrics(
            building_type_diversity=type_diversity,
            temporal_diversity=temporal_diversity,
            generation_diversity=generation_diversity,
            size_diversity=size_diversity,
            energy_label_diversity=label_diversity,
            occupancy_diversity=occupancy_diversity
        )
    
    def _calculate_intervention_metrics(self, buildings: List[Dict]) -> InterventionMetrics:
        """Calculate intervention opportunity metrics"""
        
        # Retrofit potential (poor labels)
        poor_labels = sum(1 for b in buildings 
                         if b.get('energy_label', 'D') in ['E', 'F', 'G'])
        retrofit_potential = poor_labels / len(buildings)
        
        # Solar potential (roof area without solar)
        total_roof = sum(b.get('roof_area', 0) for b in buildings)
        used_roof = sum(b.get('roof_area', 0) for b in buildings 
                       if b.get('has_solar', False))
        solar_potential = 1.0 - (used_roof / (total_roof + 1e-10))
        
        # Heat pump suitability (good insulation)
        suitable_hp = sum(1 for b in buildings 
                         if b.get('energy_label', 'D') in ['A', 'B', 'C'])
        hp_without = sum(1 for b in buildings 
                        if b.get('energy_label', 'D') in ['A', 'B', 'C'] 
                        and not b.get('has_heat_pump', False))
        heat_pump_suitability = hp_without / (suitable_hp + 1e-10)
        
        # Battery opportunity (solar without battery)
        solar_no_battery = sum(1 for b in buildings 
                             if b.get('has_solar', False) 
                             and not b.get('has_battery', False))
        battery_opportunity = solar_no_battery / (len(buildings) + 1e-10)
        
        # Urgent upgrades
        urgent_upgrades = sum(1 for b in buildings 
                             if b.get('energy_label', 'D') in ['E', 'F', 'G'])
        
        # Economic viability (simplified - based on building value)
        avg_area = np.mean([b.get('area', 100) for b in buildings])
        economic_viability = min(avg_area / 200, 1.0)  # Larger buildings = better ROI
        
        return InterventionMetrics(
            retrofit_potential=retrofit_potential,
            solar_potential=solar_potential,
            heat_pump_suitability=heat_pump_suitability,
            battery_opportunity=battery_opportunity,
            urgent_upgrades=urgent_upgrades,
            economic_viability=economic_viability
        )
    
    def _calculate_grid_metrics(self, lv_group_data: Dict) -> GridMetrics:
        """Calculate grid optimization metrics"""
        
        buildings = lv_group_data.get('buildings', [])
        
        # Peak coincidence (simplified - based on peak hour alignment)
        peak_hours = [b.get('peak_hour', 18) for b in buildings]
        mode_hour = max(set(peak_hours), key=peak_hours.count)
        coincidence = peak_hours.count(mode_hour) / len(peak_hours)
        
        # Transformer loading (if available)
        transformer_data = lv_group_data.get('transformer', {})
        capacity = transformer_data.get('capacity', 630)  # kVA
        peak_load = sum(b.get('peak_demand', 5) for b in buildings)
        transformer_loading = min(peak_load / capacity, 1.0)
        
        # Line losses (simplified estimate)
        avg_distance = np.mean([b.get('distance_to_transformer', 100) 
                               for b in buildings])
        line_losses = min(avg_distance / 1000 * 0.05, 0.2)  # 5% per km
        
        # Voltage stability (simplified)
        voltage_drops = [b.get('voltage_drop', 0.02) for b in buildings]
        avg_drop = np.mean(voltage_drops)
        voltage_stability = max(1.0 - avg_drop / 0.1, 0)  # 10% is poor
        
        # Network constraints
        network_constraints = lv_group_data.get('constraints', 0)
        
        return GridMetrics(
            peak_coincidence=coincidence,
            transformer_loading=transformer_loading,
            line_losses=line_losses,
            voltage_stability=voltage_stability,
            network_constraints=network_constraints
        )
    
    def _calculate_complementarity_suitability(
        self, diversity: DiversityMetrics, grid: GridMetrics
    ) -> float:
        """Calculate suitability for complementarity-based clustering"""
        
        # High diversity + high peak coincidence = high potential
        diversity_score = diversity.overall_diversity_index / 10
        coincidence_potential = grid.peak_coincidence
        
        # Sufficient scale (not too sparse)
        loading_factor = min(grid.transformer_loading / 0.3, 1.0)
        
        # Low losses (proximity)
        proximity_factor = 1.0 - min(grid.line_losses / 0.1, 1.0)
        
        suitability = (
            0.4 * diversity_score +
            0.3 * coincidence_potential +
            0.2 * loading_factor +
            0.1 * proximity_factor
        )
        
        return suitability * 10
    
    def _classify_lv_group(
        self, 
        diversity: DiversityMetrics,
        intervention: InterventionMetrics,
        grid: GridMetrics,
        complementarity_score: float
    ) -> str:
        """Classify LV group based on characteristics"""
        
        di = diversity.overall_diversity_index
        ip = intervention.intervention_priority_score
        gp = grid.grid_optimization_potential
        cs = complementarity_score
        
        # Priority classification
        if cs >= 7 and di >= 7:
            return "excellent_complementarity_candidate"
        elif ip >= 7 and intervention.urgent_upgrades > 5:
            return "urgent_intervention_target"
        elif cs >= 5 and ip >= 5:
            return "balanced_opportunity"
        elif gp >= 7:
            return "grid_optimization_priority"
        elif di >= 5:
            return "good_complementarity_potential"
        elif ip >= 5:
            return "intervention_opportunity"
        elif di < 3:
            return "homogeneous_limited_potential"
        else:
            return "standard_monitoring"
    
    def _generate_recommendations(
        self,
        diversity: DiversityMetrics,
        intervention: InterventionMetrics,
        grid: GridMetrics,
        classification: str
    ) -> List[str]:
        """Generate specific recommendations based on evaluation"""
        
        recommendations = []
        
        # Complementarity recommendations
        if diversity.overall_diversity_index >= 7:
            recommendations.append(
                "EXCELLENT for energy community formation - implement dynamic clustering"
            )
        elif diversity.overall_diversity_index >= 5:
            recommendations.append(
                "GOOD candidate for complementarity - consider pilot program"
            )
        elif diversity.temporal_diversity < 0.3:
            recommendations.append(
                "LIMITED complementarity - focus on individual optimizations"
            )
        
        # Intervention recommendations
        if intervention.urgent_upgrades > 5:
            recommendations.append(
                f"URGENT: {intervention.urgent_upgrades} buildings need immediate retrofit (E/F/G labels)"
            )
        
        if intervention.solar_potential > 0.7:
            recommendations.append(
                f"HIGH solar potential: {intervention.solar_potential*100:.0f}% roof area available"
            )
        
        if intervention.heat_pump_suitability > 0.5:
            recommendations.append(
                "Consider coordinated heat pump deployment program"
            )
        
        if intervention.battery_opportunity > 0.3:
            recommendations.append(
                "Battery storage would benefit existing solar installations"
            )
        
        # Grid recommendations
        if grid.peak_coincidence > 0.8:
            recommendations.append(
                "HIGH peak coincidence - demand response program recommended"
            )
        
        if grid.transformer_loading > 0.85:
            recommendations.append(
                "Transformer near capacity - prioritize load reduction"
            )
        elif grid.transformer_loading < 0.3:
            recommendations.append(
                "Transformer underutilized - room for electrification"
            )
        
        if grid.line_losses > 0.1:
            recommendations.append(
                "High line losses - consider local generation/storage"
            )
        
        # Classification-specific
        if classification == "excellent_complementarity_candidate":
            recommendations.insert(0, 
                "⭐ TOP PRIORITY: Ideal for demonstrating complementarity benefits"
            )
        elif classification == "urgent_intervention_target":
            recommendations.insert(0,
                "🚨 INTERVENTION PRIORITY: Immediate action needed for compliance"
            )
        
        return recommendations
    
    def _generate_district_specific_recommendations(
        self, 
        district_assessment: Dict[str, Any],
        diversity: DiversityMetrics,
        intervention: InterventionMetrics,
        grid: GridMetrics
    ) -> List[str]:
        """Generate recommendations specific to district characteristics"""
        
        recommendations = []
        
        # Mixed-use district recommendations
        mixed_use = district_assessment.get('mixed_use_potential', {})
        if mixed_use.get('is_excellent_mixed_use'):
            recommendations.append("⭐ EXCELLENT mixed-use district - implement time-of-use coordination between residential and commercial buildings")
        elif mixed_use.get('mixed_use_score', 0) > 0.5:
            recommendations.append("GOOD mixed-use potential - consider coordinated demand response programs")
        
        # Prosumer-rich district recommendations
        prosumer = district_assessment.get('prosumer_density', {})
        if prosumer.get('is_prosumer_rich'):
            recommendations.append("🌟 PROSUMER-RICH district - prioritize peer-to-peer energy trading and community battery systems")
            if prosumer.get('battery_storage_opportunity', 0) > 0.3:
                recommendations.append("HIGH battery storage opportunity - implement coordinated storage for prosumer optimization")
        elif prosumer.get('potential_prosumer_ratio', 0) > 0.5:
            recommendations.append("HIGH solar potential - coordinated solar installation program recommended")
        
        # Homogeneity concerns
        homogeneity = district_assessment.get('homogeneity_concerns', {})
        if homogeneity.get('is_homogeneous_problematic'):
            recommendations.append("⚠️ HOMOGENEOUS district - limited complementarity benefits, focus on individual building optimizations")
            recommendations.append("Consider expanding district boundaries to include more diverse building types")
        
        # Urgency-driven recommendations
        urgency = district_assessment.get('urgency_factors', {})
        if urgency.get('requires_immediate_action'):
            recommendations.append("🚨 URGENT intervention required - prioritize retrofit program for compliance")
            if urgency.get('compliance_risk'):
                recommendations.append("COMPLIANCE RISK - immediate action needed for EU energy efficiency regulations")
        
        # Intervention readiness recommendations
        readiness = district_assessment.get('intervention_readiness', {})
        if readiness.get('coordinated_intervention_potential', 0) > 0.3:
            recommendations.append("EXCELLENT coordinated intervention potential - implement multi-technology deployment")
        
        if readiness.get('solar_installation_readiness', 0) > 0.6:
            recommendations.append("HIGH solar readiness - bulk procurement and installation program recommended")
        
        if readiness.get('heat_pump_readiness', 0) > 0.5:
            recommendations.append("GOOD heat pump readiness - coordinate with retrofit and grid upgrade planning")
        
        # District type specific recommendations
        district_type = district_assessment.get('district_type')
        if district_type == 'mixed_use':
            recommendations.append("MIXED-USE optimization - implement differentiated time-of-use programs")
        elif district_type == 'predominantly_residential':
            recommendations.append("RESIDENTIAL focus - prioritize comfort, solar, and peak shaving interventions")
        elif district_type == 'predominantly_commercial':
            recommendations.append("COMMERCIAL focus - prioritize demand response and grid services")
        
        # Complementarity-driven recommendations
        comp_potential = district_assessment.get('complementarity_potential', {})
        if comp_potential.get('is_excellent_complementarity'):
            recommendations.append("⭐ EXCELLENT complementarity - ideal candidate for dynamic clustering pilot program")
        
        return recommendations
    
    def evaluate_portfolio(self, lv_groups: List[Dict]) -> pd.DataFrame:
        """Evaluate multiple LV groups and rank them"""
        
        evaluations = []
        
        for lv_group in lv_groups:
            eval_result = self.evaluate_lv_group(lv_group)
            
            if eval_result['status'] == 'evaluated':
                eval_data = eval_result['evaluation']
                evaluations.append({
                    'lv_group_id': eval_result['lv_group_id'],
                    'building_count': eval_result['building_count'],
                    'classification': eval_data['classification'],
                    'diversity_index': eval_data['scores']['diversity_index'],
                    'intervention_priority': eval_data['scores']['intervention_priority'],
                    'grid_optimization_potential': eval_data['scores']['grid_optimization_potential'],
                    'complementarity_score': eval_data['scores']['complementarity_suitability'],
                    'overall_score': (
                        eval_data['scores']['diversity_index'] * 0.3 +
                        eval_data['scores']['intervention_priority'] * 0.3 +
                        eval_data['scores']['grid_optimization_potential'] * 0.2 +
                        eval_data['scores']['complementarity_suitability'] * 0.2
                    )
                })
        
        df = pd.DataFrame(evaluations)
        df = df.sort_values('overall_score', ascending=False)
        
        return df
    
    def generate_evaluation_report(
        self, 
        evaluation: Dict,
        output_path: Optional[Path] = None
    ) -> str:
        """Generate detailed evaluation report"""
        
        report = []
        report.append("=" * 80)
        report.append(f"LV GROUP EVALUATION REPORT")
        report.append(f"LV Group ID: {evaluation['lv_group_id']}")
        report.append("=" * 80)
        
        if evaluation['status'] != 'evaluated':
            report.append(f"\nStatus: {evaluation['status']}")
            report.append(f"Building Count: {evaluation['building_count']}")
            report.append("Evaluation not possible - insufficient buildings")
        else:
            eval_data = evaluation['evaluation']
            
            # Summary
            report.append(f"\n📊 SUMMARY")
            report.append(f"Buildings: {evaluation['building_count']}")
            report.append(f"Classification: {eval_data['classification'].upper()}")
            report.append("")
            
            # Scores
            report.append("📈 SCORES (0-10 scale)")
            scores = eval_data['scores']
            report.append(f"  Diversity Index:          {scores['diversity_index']:.2f}")
            report.append(f"  Intervention Priority:    {scores['intervention_priority']:.2f}")
            report.append(f"  Grid Optimization:        {scores['grid_optimization_potential']:.2f}")
            if 'complementarity_score' in scores:
                report.append(f"  Complementarity Score:    {scores['complementarity_score']:.2f}")
            elif 'complementarity_potential' in scores:
                report.append(f"  Complementarity Score:    {scores['complementarity_potential']:.2f}")
            report.append("")
            
            # Diversity Details
            report.append("🌈 DIVERSITY METRICS")
            div = eval_data['diversity_metrics']
            report.append(f"  Building Types:    {div['building_type_diversity']:.2%}")
            report.append(f"  Temporal Patterns: {div['temporal_diversity']:.2%}")
            report.append(f"  Generation Mix:    {div['generation_diversity']:.2%}")
            report.append(f"  Size Variation:    {div['size_diversity']:.2%}")
            report.append(f"  Energy Labels:     {div['energy_label_diversity']:.2%}")
            report.append(f"  Occupancy:         {div['occupancy_diversity']:.2%}")
            report.append("")
            
            # Intervention Opportunities
            report.append("🔧 INTERVENTION OPPORTUNITIES")
            int_metrics = eval_data['intervention_metrics']
            report.append(f"  Retrofit Potential:   {int_metrics['retrofit_potential']:.1%}")
            report.append(f"  Solar Potential:      {int_metrics['solar_potential']:.1%}")
            report.append(f"  Heat Pump Ready:      {int_metrics['heat_pump_suitability']:.1%}")
            report.append(f"  Battery Opportunity:  {int_metrics['battery_opportunity']:.1%}")
            report.append(f"  Urgent Upgrades:      {int_metrics['urgent_upgrades']} buildings")
            report.append("")
            
            # Grid Metrics
            report.append("⚡ GRID METRICS")
            grid = eval_data['grid_metrics']
            report.append(f"  Peak Coincidence:     {grid['peak_coincidence']:.1%}")
            report.append(f"  Transformer Loading:  {grid['transformer_loading']:.1%}")
            report.append(f"  Line Losses:          {grid['line_losses']:.1%}")
            report.append(f"  Voltage Stability:    {grid['voltage_stability']:.1%}")
            report.append("")
            
            # Recommendations
            report.append("💡 RECOMMENDATIONS")
            for i, rec in enumerate(eval_data['recommendations'], 1):
                report.append(f"  {i}. {rec}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_path:
            output_path.write_text(report_text, encoding='utf-8')
            
        return report_text

    def assess_district_characteristics(self, buildings: List[Dict]) -> Dict[str, Any]:
        """
        Assess specific district characteristics for targeted analysis
        
        Returns comprehensive assessment for different district types
        """
        assessment = {
            'district_type': self._determine_district_type(buildings),
            'mixed_use_potential': self._assess_mixed_use_potential(buildings),
            'prosumer_density': self._assess_prosumer_density(buildings),
            'homogeneity_concerns': self._assess_homogeneity_concerns(buildings),
            'urgency_factors': self._assess_urgency_factors(buildings),
            'intervention_readiness': self._assess_intervention_readiness(buildings),
            'complementarity_potential': self._assess_complementarity_potential(buildings)
        }
        
        return assessment
    
    def _determine_district_type(self, buildings: List[Dict]) -> str:
        """Determine primary district characteristic"""
        types = [b.get('type', 'residential_single') for b in buildings]
        type_counts = pd.Series(types).value_counts()
        
        # Calculate percentages
        residential_pct = (
            type_counts.get('residential_single', 0) + 
            type_counts.get('residential_multi', 0)
        ) / len(buildings)
        commercial_pct = (
            type_counts.get('commercial_office', 0) + 
            type_counts.get('commercial_retail', 0)
        ) / len(buildings)
        
        if residential_pct > 0.8:
            return "predominantly_residential"
        elif commercial_pct > 0.6:
            return "predominantly_commercial"
        elif residential_pct > 0.4 and commercial_pct > 0.2:
            return "mixed_use"
        else:
            return "diverse_mixed"
    
    def _assess_mixed_use_potential(self, buildings: List[Dict]) -> Dict[str, Any]:
        """Assess mixed-use complementarity potential"""
        types = [b.get('type', 'residential_single') for b in buildings]
        
        # Count different use categories
        residential_count = sum(1 for t in types if 'residential' in t)
        commercial_count = sum(1 for t in types if 'commercial' in t)
        industrial_count = sum(1 for t in types if t == 'industrial')
        
        total = len(buildings)
        
        # Calculate complementarity potential
        res_ratio = residential_count / total
        com_ratio = commercial_count / total
        
        # Optimal mixed-use is around 60-70% residential, 20-30% commercial
        optimal_mixed = 1.0 - abs(0.65 - res_ratio) - abs(0.25 - com_ratio)
        
        return {
            'residential_ratio': res_ratio,
            'commercial_ratio': com_ratio,
            'industrial_ratio': industrial_count / total,
            'mixed_use_score': max(optimal_mixed, 0),
            'temporal_complementarity_potential': res_ratio * com_ratio * 4,  # Peak at 25% each
            'is_excellent_mixed_use': optimal_mixed > 0.7
        }
    
    def _assess_prosumer_density(self, buildings: List[Dict]) -> Dict[str, Any]:
        """Assess prosumer concentration and potential"""
        total_buildings = len(buildings)
        prosumers = sum(1 for b in buildings if b.get('has_solar', False))
        potential_prosumers = sum(1 for b in buildings 
                                 if b.get('roof_area', 0) > 50 and not b.get('has_solar', False))
        
        prosumer_ratio = prosumers / total_buildings
        potential_ratio = potential_prosumers / total_buildings
        
        # Assess if this is a prosumer-rich district
        is_prosumer_rich = prosumer_ratio > 0.3
        has_prosumer_potential = potential_ratio > 0.5
        
        return {
            'current_prosumer_ratio': prosumer_ratio,
            'potential_prosumer_ratio': potential_ratio,
            'prosumer_expansion_potential': potential_ratio,
            'is_prosumer_rich': is_prosumer_rich,
            'prosumer_clustering_benefit': prosumer_ratio * (1 - prosumer_ratio) * 4,
            'battery_storage_opportunity': prosumer_ratio * 0.8,  # 80% of prosumers could benefit
            'grid_benefits_from_prosumers': min(prosumer_ratio * 2, 1.0)
        }
    
    def _assess_homogeneity_concerns(self, buildings: List[Dict]) -> Dict[str, Any]:
        """Identify homogeneous areas with limited complementarity"""
        # Building type homogeneity
        types = [b.get('type', 'residential_single') for b in buildings]
        type_counts = pd.Series(types).value_counts()
        max_type_ratio = type_counts.max() / len(buildings)
        
        # Energy label homogeneity
        labels = [b.get('energy_label', 'D') for b in buildings]
        label_counts = pd.Series(labels).value_counts()
        max_label_ratio = label_counts.max() / len(buildings)
        
        # Size homogeneity
        areas = [b.get('area', 100) for b in buildings]
        area_cv = np.std(areas) / (np.mean(areas) + 1e-10)
        
        # Peak time homogeneity
        peak_hours = [b.get('peak_hour', 18) for b in buildings]
        peak_std = np.std(peak_hours)
        
        is_homogeneous = (max_type_ratio > 0.8 or 
                         max_label_ratio > 0.7 or 
                         area_cv < 0.2 or 
                         peak_std < 2)
        
        return {
            'building_type_homogeneity': max_type_ratio,
            'energy_label_homogeneity': max_label_ratio,
            'size_homogeneity': 1.0 - min(area_cv, 1.0),
            'temporal_homogeneity': 1.0 - min(peak_std / 12, 1.0),
            'overall_homogeneity': (max_type_ratio + max_label_ratio + 
                                  (1.0 - min(area_cv, 1.0)) + 
                                  (1.0 - min(peak_std / 12, 1.0))) / 4,
            'is_homogeneous_problematic': is_homogeneous,
            'complementarity_limitation': max_type_ratio if is_homogeneous else 0
        }
    
    def _assess_urgency_factors(self, buildings: List[Dict]) -> Dict[str, Any]:
        """Assess intervention urgency factors"""
        poor_labels = ['E', 'F', 'G']
        
        urgent_buildings = sum(1 for b in buildings 
                             if b.get('energy_label', 'D') in poor_labels)
        old_buildings = sum(1 for b in buildings 
                          if (2024 - b.get('construction_year', 2000)) > 40)
        
        total = len(buildings)
        urgent_ratio = urgent_buildings / total
        old_ratio = old_buildings / total
        
        # High urgency if > 30% poor labels or > 50% old buildings
        high_urgency = urgent_ratio > 0.3 or old_ratio > 0.5
        
        return {
            'poor_energy_label_ratio': urgent_ratio,
            'old_building_ratio': old_ratio,
            'urgent_building_count': urgent_buildings,
            'intervention_urgency_score': min((urgent_ratio + old_ratio) / 2 * 10, 10),
            'requires_immediate_action': high_urgency,
            'compliance_risk': urgent_ratio > 0.4  # EU regulations
        }
    
    def _assess_intervention_readiness(self, buildings: List[Dict]) -> Dict[str, Any]:
        """Assess readiness for different types of interventions"""
        total = len(buildings)
        
        # Solar readiness (good roof area, no existing solar)
        solar_ready = sum(1 for b in buildings 
                         if b.get('roof_area', 0) > 40 and not b.get('has_solar', False))
        
        # Heat pump readiness (good energy labels)
        hp_ready = sum(1 for b in buildings 
                      if b.get('energy_label', 'D') in ['A', 'B', 'C'] 
                      and not b.get('has_heat_pump', False))
        
        # Battery readiness (has solar, no battery)
        battery_ready = sum(1 for b in buildings 
                           if b.get('has_solar', False) and not b.get('has_battery', False))
        
        # Retrofit readiness (poor labels, sufficient size)
        retrofit_ready = sum(1 for b in buildings 
                           if b.get('energy_label', 'D') in ['D', 'E', 'F', 'G'] 
                           and b.get('area', 0) > 60)
        
        return {
            'solar_installation_readiness': solar_ready / total,
            'heat_pump_readiness': hp_ready / total,
            'battery_storage_readiness': battery_ready / total,
            'retrofit_readiness': retrofit_ready / total,
            'ready_building_counts': {
                'solar': solar_ready,
                'heat_pump': hp_ready,
                'battery': battery_ready,
                'retrofit': retrofit_ready
            },
            'overall_intervention_readiness': (solar_ready + hp_ready + battery_ready + retrofit_ready) / (4 * total),
            'coordinated_intervention_potential': min(solar_ready, hp_ready, battery_ready) / total
        }
    
    def _assess_complementarity_potential(self, buildings: List[Dict]) -> Dict[str, Any]:
        """Comprehensive complementarity assessment"""
        mixed_use = self._assess_mixed_use_potential(buildings)
        prosumer = self._assess_prosumer_density(buildings)
        homogeneity = self._assess_homogeneity_concerns(buildings)
        
        # Peak hours analysis
        peak_hours = [b.get('peak_hour', 18) for b in buildings]
        peak_spread = max(peak_hours) - min(peak_hours)
        temporal_complementarity = min(peak_spread / 12, 1.0)
        
        # Energy label diversity for retrofit complementarity
        labels = [b.get('energy_label', 'D') for b in buildings]
        label_diversity = len(set(labels)) / 7  # 7 possible labels
        
        overall_complementarity = (
            mixed_use['mixed_use_score'] * 0.3 +
            prosumer['prosumer_clustering_benefit'] * 0.25 +
            (1 - homogeneity['overall_homogeneity']) * 0.25 +
            temporal_complementarity * 0.2
        )
        
        return {
            'mixed_use_complementarity': mixed_use['mixed_use_score'],
            'prosumer_complementarity': prosumer['prosumer_clustering_benefit'],
            'temporal_complementarity': temporal_complementarity,
            'diversity_complementarity': 1 - homogeneity['overall_homogeneity'],
            'retrofit_complementarity': label_diversity,
            'overall_complementarity_index': overall_complementarity,
            'peak_hour_spread': peak_spread,
            'is_excellent_complementarity': overall_complementarity > 0.7
        }

def evaluate_and_select_lv_groups(
    lv_groups_data: List[Dict],
    selection_criteria: Optional[Dict] = None,
    top_n: int = 5
) -> Tuple[List[str], pd.DataFrame, List[Dict]]:
    """
    Evaluate all LV groups and select best candidates
    
    Returns:
        - List of selected LV group IDs
        - DataFrame with all evaluations
        - List of detailed evaluation reports
    """
    
    evaluator = LVGroupEvaluator()
    
    # Evaluate all groups
    evaluations = []
    reports = []
    
    for lv_group in lv_groups_data:
        eval_result = evaluator.evaluate_lv_group(lv_group)
        evaluations.append(eval_result)
        
        if eval_result['status'] == 'evaluated':
            report = evaluator.generate_evaluation_report(eval_result)
            reports.append({
                'lv_group_id': eval_result['lv_group_id'],
                'report': report
            })
    
    # Create portfolio summary
    portfolio_df = evaluator.evaluate_portfolio(lv_groups_data)
    
    # Apply selection criteria
    if selection_criteria:
        if 'min_diversity' in selection_criteria:
            portfolio_df = portfolio_df[
                portfolio_df['diversity_index'] >= selection_criteria['min_diversity']
            ]
        if 'classification' in selection_criteria:
            portfolio_df = portfolio_df[
                portfolio_df['classification'].isin(selection_criteria['classification'])
            ]
    
    # Select top candidates
    selected_ids = portfolio_df.head(top_n)['lv_group_id'].tolist()
    
    logger.info(f"Selected {len(selected_ids)} LV groups from {len(lv_groups_data)} evaluated")
    
    return selected_ids, portfolio_df, reports