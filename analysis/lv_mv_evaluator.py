"""
Enhanced LV/MV Group Evaluator for Initial Assessment
Focuses on diversity, energy labels, and planning potential
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class LVGroupMetrics:
    """Core metrics for LV group evaluation"""
    lv_group_id: str
    building_count: int
    
    # Diversity metrics (0-1 scale)
    function_diversity: float  # Mix of residential/commercial/industrial
    energy_label_diversity: float  # Variety in efficiency levels
    temporal_diversity: float  # Peak time variation
    size_diversity: float  # Building size variation
    
    # Quality metrics
    poor_label_ratio: float  # % of E/F/G labels
    avg_energy_label_score: float  # A=1, B=2, ..., G=7
    solar_readiness: float  # Buildings with good potential
    
    # Complementarity potential
    complementarity_score: float  # Overall potential for energy sharing
    peak_coincidence: float  # How aligned are peak demands
    
    def get_planning_score(self) -> float:
        """Calculate overall planning priority score (0-10)"""
        # High diversity + poor labels = good for intervention
        diversity_factor = (self.function_diversity + self.temporal_diversity) / 2
        intervention_need = self.poor_label_ratio
        solar_opportunity = self.solar_readiness
        
        score = (
            diversity_factor * 3.0 +  # Max 3 points for diversity
            intervention_need * 4.0 +  # Max 4 points for poor labels
            solar_opportunity * 2.0 +  # Max 2 points for solar potential
            (1 - self.peak_coincidence) * 1.0  # Max 1 point for non-coincident peaks
        )
        return min(score, 10.0)

@dataclass 
class MVGroupMetrics:
    """Aggregated metrics for MV station"""
    mv_station_id: str
    lv_group_count: int
    total_buildings: int
    
    # Aggregated from LV groups
    avg_diversity_score: float
    avg_poor_label_ratio: float
    avg_complementarity: float
    
    # MV-level metrics
    lv_group_variety: float  # Diversity among LV groups
    intervention_urgency: float  # Overall need for upgrades
    planning_priority: float  # Final priority score
    
    # Best/worst LV groups
    best_lv_groups: List[str]
    problematic_lv_groups: List[str]


class EnhancedLVMVEvaluator:
    """Simplified evaluator focusing on key metrics for planning"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or self._default_config()
        
        # Energy label scoring (lower is better)
        self.label_scores = {
            'A++++': 0.5, 'A+++': 0.7, 'A++': 0.9, 'A+': 1.0,
            'A': 1.0, 'B': 2.0, 'C': 3.0, 'D': 4.0,
            'E': 5.0, 'F': 6.0, 'G': 7.0
        }
        
        # Building type codes for diversity calculation
        self.building_types = {
            'residential': 0, 'commercial': 1, 'industrial': 2,
            'office': 1, 'retail': 1, 'mixed': 3
        }
        
    def _default_config(self) -> Dict:
        return {
            'min_buildings_per_lv': 5,
            'max_buildings_per_lv': 250,
            'poor_label_threshold': 0.3,  # 30% poor labels = urgent
            'diversity_threshold': 0.5,
            'solar_potential_formula': {
                'min_roof_area': 30,  # m²
                'good_orientation': ['S', 'SE', 'SW', 'E', 'W'],
                'label_bonus': {'A': 0.2, 'B': 0.1}  # Extra points for good labels
            }
        }
    
    def evaluate_lv_group(self, buildings: List[Dict]) -> LVGroupMetrics:
        """Evaluate a single LV group"""
        
        n_buildings = len(buildings)
        if n_buildings < self.config['min_buildings_per_lv']:
            logger.warning(f"LV group has only {n_buildings} buildings")
        
        # Calculate diversity metrics
        function_diversity = self._calculate_function_diversity(buildings)
        energy_label_diversity = self._calculate_label_diversity(buildings)
        temporal_diversity = self._calculate_temporal_diversity(buildings)
        size_diversity = self._calculate_size_diversity(buildings)
        
        # Calculate quality metrics
        poor_label_ratio = self._calculate_poor_label_ratio(buildings)
        avg_label_score = self._calculate_avg_label_score(buildings)
        solar_readiness = self._calculate_solar_readiness(buildings)
        
        # Calculate complementarity
        peak_coincidence = self._calculate_peak_coincidence(buildings)
        complementarity = self._calculate_complementarity_potential(
            function_diversity, temporal_diversity, peak_coincidence
        )
        
        return LVGroupMetrics(
            lv_group_id=buildings[0].get('lv_group_id', 'unknown'),
            building_count=n_buildings,
            function_diversity=function_diversity,
            energy_label_diversity=energy_label_diversity,
            temporal_diversity=temporal_diversity,
            size_diversity=size_diversity,
            poor_label_ratio=poor_label_ratio,
            avg_energy_label_score=avg_label_score,
            solar_readiness=solar_readiness,
            complementarity_score=complementarity,
            peak_coincidence=peak_coincidence
        )
    
    def _calculate_function_diversity(self, buildings: List[Dict]) -> float:
        """Calculate diversity of building functions using entropy"""
        types = []
        for b in buildings:
            # Use building_function field from KG (residential/non_residential)
            func = b.get('building_function')
            if func == 'residential':
                types.append('residential')
            elif func == 'non_residential':
                # Check non_residential_type for more detail
                nonres_type = b.get('non_residential_type', 'other')
                if nonres_type and 'Office' in str(nonres_type):
                    types.append('office')
                elif nonres_type and 'Retail' in str(nonres_type):
                    types.append('retail')
                elif nonres_type and 'Industrial' in str(nonres_type):
                    types.append('industrial')
                else:
                    types.append('other')
            else:
                # Default to residential if missing
                types.append('residential')
        
        # Calculate entropy
        type_counts = pd.Series(types).value_counts()
        proportions = type_counts / len(types)
        entropy = -sum(p * np.log(p + 1e-10) for p in proportions)
        
        # FIX: Use fixed number of possible types instead of actual count
        max_possible_types = 5  # residential, office, retail, industrial, other
        max_entropy = np.log(max_possible_types) if max_possible_types > 1 else 1.0
        
        # Ensure we don't divide by zero and result is between 0 and 1
        diversity = entropy / max_entropy if max_entropy > 0 else 0.0
        return max(0.0, min(1.0, diversity))  # Clamp between 0 and 1
    
    def _calculate_label_diversity(self, buildings: List[Dict]) -> float:
        """Calculate diversity of energy labels"""
        labels = [b.get('energy_label', 'D') for b in buildings]
        unique_labels = len(set(labels))
        max_labels = 7  # A through G
        return unique_labels / max_labels
    
    def _calculate_temporal_diversity(self, buildings: List[Dict]) -> float:
        """Calculate diversity in peak demand times"""
        peak_hours = []
        for b in buildings:
            ph = b.get('peak_hour', 18)
            # Handle if peak_hour is a list (from peak_hours field)
            if isinstance(ph, list):
                peak_hours.append(ph[0] if ph else 18)
            else:
                peak_hours.append(ph if ph is not None else 18)
        
        if len(set(peak_hours)) == 1:
            return 0.0  # All same peak
        
        # Calculate spread
        hour_std = np.std(peak_hours)
        max_std = 12  # Maximum possible std for 24-hour cycle
        return min(hour_std / max_std, 1.0)
    
    def _calculate_size_diversity(self, buildings: List[Dict]) -> float:
        """Calculate diversity in building sizes"""
        areas = [b.get('area', 100) for b in buildings]
        if len(set(areas)) == 1:
            return 0.0
        
        cv = np.std(areas) / (np.mean(areas) + 1e-10)
        return min(cv, 1.0)
    
    def _calculate_poor_label_ratio(self, buildings: List[Dict]) -> float:
        """Calculate ratio of buildings with poor energy labels (E/F/G)"""
        poor_labels = ['E', 'F', 'G']
        poor_count = sum(1 for b in buildings 
                        if b.get('energy_label', 'D') in poor_labels)
        return poor_count / len(buildings)
    
    def _calculate_avg_label_score(self, buildings: List[Dict]) -> float:
        """Calculate average energy label score"""
        scores = []
        for b in buildings:
            label = b.get('energy_label', 'D')
            scores.append(self.label_scores.get(label, 4.0))
        return np.mean(scores)
    
    def _calculate_solar_readiness(self, buildings: List[Dict]) -> float:
        """
        Calculate solar installation readiness based on:
        - Orientation (now available!)
        - Roof area 
        - Energy label (prioritize poor labels for intervention)
        - Has solar already
        """
        formula = self.config['solar_potential_formula']
        ready_count = 0
        
        for b in buildings:
            # Check if already has solar
            if b.get('has_solar', False):
                continue
            
            # Check roof area (now available from update)
            roof_area = b.get('roof_area', 0)
            if not roof_area:
                # Fallback to individual components
                flat_roof = b.get('flat_roof_area', 0) 
                sloped_roof = b.get('sloped_roof_area', 0)
                roof_area = (flat_roof or 0) + (sloped_roof or 0)
            
            if roof_area and roof_area >= formula['min_roof_area']:
                # Check orientation (NOW AVAILABLE!)
                orientation = b.get('building_orientation_cardinal') or b.get('orientation')
                
                # Score based on orientation
                orientation_score = 0
                if orientation in formula['good_orientation']:
                    if orientation in ['S', 'SE', 'SW']:
                        orientation_score = 1.0  # Best orientations
                    elif orientation in ['E', 'W']:
                        orientation_score = 0.7  # Good orientations
                else:
                    orientation_score = 0.3  # North facing, less ideal
                
                # Check energy label - prioritize poor labels
                label = b.get('energy_label', 'Unknown')
                label_score = 0
                if label in ['E', 'F', 'G']:
                    label_score = 1.0  # High priority for poor labels
                elif label in ['C', 'D']:
                    label_score = 0.7  # Medium priority
                elif label in ['A', 'B', 'A+', 'A++', 'A+++', 'A++++']:
                    label_score = 0.3  # Lower priority
                else:  # Unknown
                    label_score = 0.5
                
                # Combined score (orientation * label priority)
                ready_count += orientation_score * label_score
        
        return ready_count / len(buildings) if len(buildings) > 0 else 0.0
    
    def _calculate_peak_coincidence(self, buildings: List[Dict]) -> float:
        """Calculate how coincident peak demands are"""
        peak_hours = []
        for b in buildings:
            ph = b.get('peak_hour', 18)
            # Handle if peak_hour is a list (from peak_hours field)
            if isinstance(ph, list):
                peak_hours.append(ph[0] if ph else 18)
            else:
                peak_hours.append(ph if ph is not None else 18)
        
        # Find most common peak hour
        mode_hour = max(set(peak_hours), key=peak_hours.count)
        coincidence = peak_hours.count(mode_hour) / len(peak_hours)
        
        return coincidence
    
    def _calculate_complementarity_potential(
        self, function_div: float, temporal_div: float, peak_coin: float
    ) -> float:
        """Calculate overall complementarity potential"""
        # High diversity + low coincidence = high complementarity
        return (function_div * 0.4 + temporal_div * 0.4 + (1 - peak_coin) * 0.2)
    
    def evaluate_mv_station(
        self, lv_groups: List[List[Dict]], mv_station_id: str
    ) -> MVGroupMetrics:
        """Evaluate all LV groups under an MV station"""
        
        lv_metrics = []
        total_buildings = 0
        
        # Evaluate each LV group
        for lv_buildings in lv_groups:
            if not lv_buildings:
                continue
            metrics = self.evaluate_lv_group(lv_buildings)
            lv_metrics.append(metrics)
            total_buildings += metrics.building_count
        
        if not lv_metrics:
            logger.warning(f"No valid LV groups for MV station {mv_station_id}")
            return None
        
        # Aggregate metrics
        avg_diversity = np.mean([
            (m.function_diversity + m.temporal_diversity) / 2 
            for m in lv_metrics
        ])
        avg_poor_labels = np.mean([m.poor_label_ratio for m in lv_metrics])
        avg_complementarity = np.mean([m.complementarity_score for m in lv_metrics])
        
        # Calculate LV group variety (diversity among groups)
        planning_scores = [m.get_planning_score() for m in lv_metrics]
        lv_variety = np.std(planning_scores) / (np.mean(planning_scores) + 1e-10)
        
        # Calculate intervention urgency
        urgency = avg_poor_labels * 0.6 + (1 - avg_diversity) * 0.4
        
        # Overall planning priority
        priority = (
            avg_diversity * 2.0 +  # Diversity is good
            avg_poor_labels * 3.0 +  # Poor labels need intervention
            avg_complementarity * 2.0 +  # Complementarity potential
            lv_variety * 1.0 +  # Variety among groups
            urgency * 2.0  # Urgency factor
        )
        
        # Find best and problematic LV groups
        sorted_lv = sorted(lv_metrics, key=lambda x: x.get_planning_score(), reverse=True)
        best_lv = [m.lv_group_id for m in sorted_lv[:3]]
        problematic_lv = [m.lv_group_id for m in sorted_lv if m.poor_label_ratio > 0.5]
        
        return MVGroupMetrics(
            mv_station_id=mv_station_id,
            lv_group_count=len(lv_metrics),
            total_buildings=total_buildings,
            avg_diversity_score=avg_diversity,
            avg_poor_label_ratio=avg_poor_labels,
            avg_complementarity=avg_complementarity,
            lv_group_variety=lv_variety,
            intervention_urgency=urgency,
            planning_priority=min(priority, 10.0),
            best_lv_groups=best_lv,
            problematic_lv_groups=problematic_lv
        )
    
    def generate_assessment_report(
        self, 
        mv_metrics: MVGroupMetrics,
        lv_metrics_list: List[LVGroupMetrics],
        output_path: Optional[Path] = None
    ) -> str:
        """Generate comprehensive assessment report"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        report = []
        report.append("=" * 80)
        report.append("INITIAL ASSESSMENT REPORT - MV STATION & LV GROUPS")
        report.append(f"Generated: {timestamp}")
        report.append("=" * 80)
        
        # MV Station Summary
        report.append("\n📊 MV STATION OVERVIEW")
        report.append(f"Station ID: {mv_metrics.mv_station_id}")
        report.append(f"Total LV Groups: {mv_metrics.lv_group_count}")
        report.append(f"Total Buildings: {mv_metrics.total_buildings}")
        report.append("")
        
        # Key Metrics
        report.append("🎯 KEY METRICS")
        report.append(f"Planning Priority Score: {mv_metrics.planning_priority:.2f}/10")
        report.append(f"Average Diversity: {mv_metrics.avg_diversity_score:.1%}")
        report.append(f"Poor Energy Labels (E/F/G): {mv_metrics.avg_poor_label_ratio:.1%}")
        report.append(f"Complementarity Potential: {mv_metrics.avg_complementarity:.1%}")
        report.append(f"Intervention Urgency: {mv_metrics.intervention_urgency:.1%}")
        report.append("")
        
        # Strategic Assessment
        report.append("📋 STRATEGIC ASSESSMENT")
        
        if mv_metrics.planning_priority >= 7:
            report.append("✅ EXCELLENT candidate for energy community planning")
            report.append("   - High diversity enables complementary energy sharing")
            report.append("   - Significant upgrade needs justify intervention")
        elif mv_metrics.planning_priority >= 5:
            report.append("✅ GOOD candidate for targeted interventions")
            report.append("   - Moderate diversity offers some complementarity")
            report.append("   - Notable improvement potential")
        else:
            report.append("⚠️ LIMITED potential for community approach")
            report.append("   - Consider individual building optimizations")
            report.append("   - May benefit from boundary adjustments")
        report.append("")
        
        # Best LV Groups for Pilot
        report.append("🌟 TOP LV GROUPS FOR PILOT PROGRAM")
        for i, lv_id in enumerate(mv_metrics.best_lv_groups[:5], 1):
            lv_metric = next((m for m in lv_metrics_list if m.lv_group_id == lv_id), None)
            if lv_metric:
                report.append(f"{i}. {lv_id}")
                report.append(f"   - Buildings: {lv_metric.building_count}")
                report.append(f"   - Planning Score: {lv_metric.get_planning_score():.2f}/10")
                report.append(f"   - Diversity: {lv_metric.function_diversity:.1%}")
                report.append(f"   - Poor Labels: {lv_metric.poor_label_ratio:.1%}")
        report.append("")
        
        # Problematic LV Groups
        if mv_metrics.problematic_lv_groups:
            report.append("⚠️ LV GROUPS REQUIRING URGENT ATTENTION")
            for lv_id in mv_metrics.problematic_lv_groups[:5]:
                lv_metric = next((m for m in lv_metrics_list if m.lv_group_id == lv_id), None)
                if lv_metric:
                    report.append(f"• {lv_id}: {lv_metric.poor_label_ratio:.0%} poor labels")
        report.append("")
        
        # Recommendations
        report.append("💡 RECOMMENDATIONS")
        
        # Based on metrics, provide specific recommendations
        if mv_metrics.avg_poor_label_ratio > 0.3:
            report.append("1. URGENT: Implement retrofit program for E/F/G labeled buildings")
        
        if mv_metrics.avg_diversity_score > 0.6:
            report.append("2. OPPORTUNITY: Deploy dynamic clustering for energy sharing")
        
        if mv_metrics.avg_complementarity > 0.5:
            report.append("3. POTENTIAL: Establish peer-to-peer energy trading pilot")
        
        # Solar recommendations
        solar_ready_lv = [m for m in lv_metrics_list if m.solar_readiness > 0.3]
        if solar_ready_lv:
            report.append(f"4. SOLAR: {len(solar_ready_lv)} LV groups ready for solar deployment")
        
        # Detailed LV Analysis
        report.append("\n" + "=" * 80)
        report.append("DETAILED LV GROUP ANALYSIS")
        report.append("=" * 80)
        
        # Sort LV groups by planning score
        sorted_lv = sorted(lv_metrics_list, key=lambda x: x.get_planning_score(), reverse=True)
        
        for lv_metric in sorted_lv[:10]:  # Top 10 detailed
            report.append(f"\n📍 LV Group: {lv_metric.lv_group_id}")
            report.append(f"   Planning Score: {lv_metric.get_planning_score():.2f}/10")
            report.append(f"   Buildings: {lv_metric.building_count}")
            report.append(f"   Function Diversity: {lv_metric.function_diversity:.1%}")
            report.append(f"   Temporal Diversity: {lv_metric.temporal_diversity:.1%}")
            report.append(f"   Poor Labels: {lv_metric.poor_label_ratio:.1%}")
            report.append(f"   Solar Ready: {lv_metric.solar_readiness:.1%}")
            report.append(f"   Complementarity: {lv_metric.complementarity_score:.1%}")
            
            # Specific recommendations for this LV group
            if lv_metric.poor_label_ratio > 0.4:
                report.append("   → Priority: Retrofit program")
            if lv_metric.solar_readiness > 0.4:
                report.append("   → Opportunity: Solar installations")
            if lv_metric.complementarity_score > 0.6:
                report.append("   → Potential: Energy community formation")
        
        # Summary Statistics
        report.append("\n" + "=" * 80)
        report.append("SUMMARY STATISTICS")
        report.append("=" * 80)
        
        # Distribution analysis
        planning_scores = [m.get_planning_score() for m in lv_metrics_list]
        report.append(f"Planning Score Distribution:")
        report.append(f"  - Mean: {np.mean(planning_scores):.2f}")
        report.append(f"  - Std Dev: {np.std(planning_scores):.2f}")
        report.append(f"  - Min: {np.min(planning_scores):.2f}")
        report.append(f"  - Max: {np.max(planning_scores):.2f}")
        
        poor_ratios = [m.poor_label_ratio for m in lv_metrics_list]
        report.append(f"\nPoor Label Distribution:")
        report.append(f"  - >50% poor labels: {sum(1 for r in poor_ratios if r > 0.5)} LV groups")
        report.append(f"  - 30-50% poor labels: {sum(1 for r in poor_ratios if 0.3 <= r <= 0.5)} LV groups")
        report.append(f"  - <30% poor labels: {sum(1 for r in poor_ratios if r < 0.3)} LV groups")
        
        report.append("\n" + "=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        # Save if path provided
        if output_path:
            output_path.write_text(report_text, encoding='utf-8')
            logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def select_best_mv_station(
        self, mv_stations: List[MVGroupMetrics]
    ) -> Tuple[str, str]:
        """Select best MV station based on strategy"""
        
        # Sort by planning priority
        sorted_mv = sorted(mv_stations, key=lambda x: x.planning_priority, reverse=True)
        
        if not sorted_mv:
            return None, "No MV stations evaluated"
        
        best_mv = sorted_mv[0]
        
        # Generate selection rationale
        rationale = f"Selected {best_mv.mv_station_id} with priority score {best_mv.planning_priority:.2f}/10. "
        
        if best_mv.avg_diversity_score > 0.6 and best_mv.avg_poor_label_ratio > 0.3:
            rationale += "Excellent mix of diversity (enables sharing) and intervention needs."
        elif best_mv.avg_diversity_score > 0.6:
            rationale += "High diversity enables effective energy community formation."
        elif best_mv.avg_poor_label_ratio > 0.3:
            rationale += "Urgent intervention needs justify immediate action."
        else:
            rationale += "Balanced characteristics suitable for pilot program."
        
        return best_mv.mv_station_id, rationale


def run_initial_assessment(kg_connector, output_dir: Path = Path("reports")):
    """Run complete initial assessment of MV/LV groups"""
    
    logger.info("Starting initial assessment...")
    output_dir.mkdir(exist_ok=True)
    
    evaluator = EnhancedLVMVEvaluator()
    
    # Get all MV stations and their LV groups from knowledge graph
    # This is a simplified example - adapt to your actual KG structure
    mv_stations_data = kg_connector.get_mv_stations_with_lv_groups()
    
    all_mv_metrics = []
    all_reports = []
    
    for mv_data in mv_stations_data:
        mv_id = mv_data['id']
        lv_groups = mv_data['lv_groups']
        
        # Get buildings for each LV group
        lv_buildings_list = []
        lv_metrics_list = []
        
        for lv_group in lv_groups:
            buildings = kg_connector.get_buildings_in_lv_group(lv_group['id'])
            if buildings:
                lv_buildings_list.append(buildings)
                lv_metric = evaluator.evaluate_lv_group(buildings)
                lv_metrics_list.append(lv_metric)
        
        if lv_buildings_list:
            # Evaluate MV station
            mv_metrics = evaluator.evaluate_mv_station(lv_buildings_list, mv_id)
            if mv_metrics:
                all_mv_metrics.append(mv_metrics)
                
                # Generate report
                report = evaluator.generate_assessment_report(
                    mv_metrics, 
                    lv_metrics_list,
                    output_dir / f"assessment_{mv_id}.txt"
                )
                all_reports.append(report)
    
    # Select best MV station
    if all_mv_metrics:
        best_mv, rationale = evaluator.select_best_mv_station(all_mv_metrics)
        
        # Generate final summary
        summary_path = output_dir / "assessment_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("INITIAL ASSESSMENT SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Evaluated {len(all_mv_metrics)} MV stations\n\n")
            f.write(f"SELECTED MV STATION: {best_mv}\n")
            f.write(f"RATIONALE: {rationale}\n\n")
            
            # Top 5 MV stations
            f.write("TOP 5 MV STATIONS BY PLANNING PRIORITY:\n")
            sorted_mv = sorted(all_mv_metrics, key=lambda x: x.planning_priority, reverse=True)
            for i, mv in enumerate(sorted_mv[:5], 1):
                f.write(f"{i}. {mv.mv_station_id}: Score {mv.planning_priority:.2f}/10\n")
                f.write(f"   - LV Groups: {mv.lv_group_count}\n")
                f.write(f"   - Buildings: {mv.total_buildings}\n")
                f.write(f"   - Diversity: {mv.avg_diversity_score:.1%}\n")
                f.write(f"   - Poor Labels: {mv.avg_poor_label_ratio:.1%}\n\n")
        
        logger.info(f"Assessment complete. Selected MV station: {best_mv}")
        logger.info(f"Reports saved to {output_dir}")
        
        return best_mv, all_mv_metrics, all_reports
    
    return None, [], []