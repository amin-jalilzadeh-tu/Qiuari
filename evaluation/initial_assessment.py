"""
Initial Assessment Module for LV Groups and MV Stations
Evaluates grid hierarchy before clustering to identify:
1. High-potential LV groups for energy communities
2. Priority MV stations for interventions
3. Strategic opportunities across the network
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LVGroupMetrics:
    """Core metrics for LV group assessment"""
    lv_group_id: str
    building_count: int
    
    # Diversity metrics (0-1)
    function_diversity: float  # Mix of residential, commercial, industrial
    energy_label_distribution: Dict[str, int]  # A-G label counts
    poor_label_ratio: float  # Ratio of E, F, G labels
    size_variance: float  # Building size variation
    
    # Energy metrics
    total_demand_mwh: float  # Annual demand
    peak_demand_kw: float  # Peak load
    existing_solar_kw: float  # Current solar capacity
    solar_potential_kw: float  # Additional solar potential
    
    # Complementarity indicators
    temporal_diversity: float  # Peak hour spread
    prosumer_ratio: float  # Buildings with generation
    consumption_correlation: float  # Average correlation between buildings
    
    # Network metrics
    transformer_utilization: float  # Current loading %
    distance_spread: float  # Geographic spread of buildings
    
    def get_summary_score(self) -> float:
        """Calculate overall potential score"""
        diversity_score = self.function_diversity * 0.25
        intervention_need = self.poor_label_ratio * 0.25
        solar_opportunity = (self.solar_potential_kw / (self.solar_potential_kw + self.existing_solar_kw + 1e-6)) * 0.25
        complementarity = (1 - abs(self.consumption_correlation)) * 0.25
        return (diversity_score + intervention_need + solar_opportunity + complementarity) * 10


@dataclass 
class MVStationMetrics:
    """Aggregated metrics for MV station"""
    mv_station_id: str
    lv_group_count: int
    total_building_count: int
    
    # Aggregated diversity
    avg_function_diversity: float
    avg_poor_label_ratio: float
    total_poor_buildings: int
    
    # Energy totals
    total_demand_mwh: float
    peak_demand_kw: float
    total_solar_kw: float
    total_solar_potential_kw: float
    
    # Planning suitability
    high_potential_lv_groups: int  # LV groups with score > 7
    intervention_priority: float  # Based on poor labels and potential
    complementarity_index: float  # Average complementarity potential
    
    # Strategic classification
    strategy_recommendation: str  # 'energy_community', 'retrofit_focus', 'solar_priority', 'mixed'
    
    def get_priority_score(self) -> float:
        """Calculate MV station priority for interventions"""
        # High diversity + poor labels = great opportunity
        opportunity = self.avg_function_diversity * 0.3
        urgency = self.avg_poor_label_ratio * 0.3
        scale = min(self.total_building_count / 500, 1.0) * 0.2  # Prefer larger areas
        potential_lvs = (self.high_potential_lv_groups / max(self.lv_group_count, 1)) * 0.2
        return (opportunity + urgency + scale + potential_lvs) * 10


class InitialAssessment:
    """
    Performs initial assessment of LV groups and MV stations
    to guide clustering and intervention strategies
    """
    
    def __init__(self, kg_connector, config: Optional[Dict] = None):
        """
        Initialize assessment module
        
        Args:
            kg_connector: Neo4j connection
            config: Assessment configuration
        """
        self.kg = kg_connector
        self.config = config or self._default_config()
        
        # Thresholds
        self.poor_labels = ['E', 'F', 'G']
        self.good_labels = ['A', 'B', 'C']
        
        # Storage for results
        self.lv_assessments = {}
        self.mv_assessments = {}
        self.hierarchy_data = {}
        
    def _default_config(self) -> Dict:
        return {
            'min_buildings_per_lv': 3,
            'high_potential_threshold': 7.0,
            'diversity_weight': 0.3,
            'intervention_weight': 0.3,
            'complementarity_weight': 0.4
        }
    
    def _get_lv_groups_for_mv(self, mv_station_id: str) -> List[Dict]:
        """
        Get LV groups (cable groups) under an MV station.
        For now, simulate with reasonable data structure.
        """
        # In real implementation, this would query Neo4j for actual LV groups
        # For now, create a simple structure to test the assessment
        
        # Simulate 3-5 LV groups per MV station
        import random
        num_lv_groups = random.randint(3, 5)
        
        lv_groups = []
        for i in range(num_lv_groups):
            num_buildings = random.randint(10, 50)
            buildings = []
            
            for b in range(num_buildings):
                buildings.append({
                    'ogc_fid': f"{mv_station_id}_LV{i}_B{b}",
                    'area': random.uniform(60, 200),
                    'energy_label': random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G']),
                    'has_solar': random.random() < 0.2,
                    'has_heat_pump': random.random() < 0.1,
                    'has_battery': random.random() < 0.05,
                    'roof_area': random.uniform(30, 150),
                    'orientation': random.choice(['south', 'east', 'west', 'south-east', 'south-west']),
                    'type': random.choice(['residential', 'commercial', 'mixed']),
                    'peak_demand': random.uniform(3, 15),
                    'annual_consumption': random.uniform(2000, 15000),
                    'peak_hour': random.randint(7, 20),
                    'distance_to_transformer': random.uniform(50, 500)
                })
            
            lv_groups.append({
                'id': f"{mv_station_id}_LV_GROUP_{i:04d}",
                'buildings': buildings,
                'transformer_loading': random.uniform(0.3, 0.9)
            })
        
        return lv_groups
    
    def run_full_assessment(self) -> Tuple[Dict, pd.DataFrame]:
        """
        Run complete assessment of network hierarchy
        
        Returns:
            assessment_results: Dictionary with all metrics
            summary_df: Summary DataFrame for reporting
        """
        logger.info("Starting initial network assessment...")
        
        # Get full hierarchy
        self.hierarchy_data = self.kg.get_mv_lv_hierarchy()
        
        if not self.hierarchy_data:
            logger.error("No hierarchy data available")
            return {}, pd.DataFrame()
        
        # Process each HV substation
        for hv_name, hv_data in self.hierarchy_data.items():
            logger.info(f"Assessing HV substation: {hv_name}")
            
            for mv_station in hv_data.get('mv_stations', []):
                mv_id = mv_station['id']
                logger.info(f"  Processing MV station: {mv_id}")
                
                # Get LV groups for this MV station
                # Since hierarchy doesn't include full LV data, fetch it separately
                lv_groups = self._get_lv_groups_for_mv(mv_id)
                
                # Assess all LV groups under this MV
                lv_metrics = []
                for lv_group in lv_groups:
                    lv_assessment = self._assess_lv_group(lv_group)
                    if lv_assessment:
                        self.lv_assessments[lv_assessment.lv_group_id] = lv_assessment
                        lv_metrics.append(lv_assessment)
                
                # Aggregate to MV level
                if lv_metrics:
                    mv_assessment = self._aggregate_to_mv(mv_id, lv_metrics)
                    self.mv_assessments[mv_id] = mv_assessment
        
        # Generate summary
        assessment_results = {
            'timestamp': datetime.now().isoformat(),
            'hv_substations': len(self.hierarchy_data),
            'mv_stations': len(self.mv_assessments),
            'lv_groups': len(self.lv_assessments),
            'lv_assessments': {k: asdict(v) for k, v in self.lv_assessments.items()},
            'mv_assessments': {k: asdict(v) for k, v in self.mv_assessments.items()}
        }
        
        summary_df = self._create_summary_dataframe()
        
        logger.info(f"Assessment complete: {len(self.mv_assessments)} MV stations, "
                   f"{len(self.lv_assessments)} LV groups assessed")
        
        return assessment_results, summary_df
    
    def _assess_lv_group(self, lv_group: Dict) -> Optional[LVGroupMetrics]:
        """Assess individual LV group"""
        buildings = lv_group.get('buildings', [])
        
        if len(buildings) < self.config['min_buildings_per_lv']:
            return None
        
        # Calculate diversity metrics
        function_diversity = self._calculate_function_diversity(buildings)
        label_dist = self._get_label_distribution(buildings)
        poor_ratio = sum(label_dist.get(l, 0) for l in self.poor_labels) / len(buildings)
        
        # Building characteristics
        areas = [b.get('area', 100) for b in buildings]
        size_variance = np.std(areas) / (np.mean(areas) + 1e-6)
        
        # Energy metrics
        total_demand = sum(b.get('annual_consumption', 0) for b in buildings) / 1000  # MWh
        peak_demand = sum(b.get('peak_demand', 0) for b in buildings)
        
        # Solar assessment
        existing_solar = sum(b.get('solar_capacity', 0) for b in buildings 
                           if b.get('has_solar', False))
        
        # Calculate potential based on roof area and orientation
        solar_potential = self._calculate_solar_potential(buildings)
        
        # Complementarity metrics
        temporal_diversity = self._calculate_temporal_diversity(buildings)
        prosumer_ratio = sum(1 for b in buildings if b.get('has_solar', False)) / len(buildings)
        
        # Simple correlation estimate based on building types
        correlation = self._estimate_consumption_correlation(buildings)
        
        # Network metrics
        transformer_util = lv_group.get('transformer_loading', 0.5)
        distances = [b.get('distance_to_transformer', 100) for b in buildings]
        distance_spread = np.std(distances) / (np.mean(distances) + 1e-6)
        
        return LVGroupMetrics(
            lv_group_id=lv_group.get('id', 'unknown'),
            building_count=len(buildings),
            function_diversity=function_diversity,
            energy_label_distribution=label_dist,
            poor_label_ratio=poor_ratio,
            size_variance=min(size_variance, 1.0),
            total_demand_mwh=total_demand,
            peak_demand_kw=peak_demand,
            existing_solar_kw=existing_solar,
            solar_potential_kw=solar_potential,
            temporal_diversity=temporal_diversity,
            prosumer_ratio=prosumer_ratio,
            consumption_correlation=correlation,
            transformer_utilization=transformer_util,
            distance_spread=min(distance_spread, 1.0)
        )
    
    def _calculate_function_diversity(self, buildings: List[Dict]) -> float:
        """Calculate diversity of building functions"""
        types = [b.get('type', 'residential') for b in buildings]
        type_counts = pd.Series(types).value_counts()
        
        # Shannon entropy normalized
        entropy = -sum((c/len(types)) * np.log(c/len(types) + 1e-10) 
                      for c in type_counts.values)
        max_entropy = np.log(len(type_counts)) if len(type_counts) > 1 else 1
        
        return entropy / (max_entropy + 1e-10)
    
    def _get_label_distribution(self, buildings: List[Dict]) -> Dict[str, int]:
        """Get energy label distribution"""
        labels = [b.get('energy_label', 'D') for b in buildings]
        return dict(pd.Series(labels).value_counts())
    
    def _calculate_solar_potential(self, buildings: List[Dict]) -> float:
        """Calculate solar potential in kW"""
        potential_kw = 0
        
        for b in buildings:
            if not b.get('has_solar', False):
                roof_area = b.get('roof_area', 0)
                orientation_factor = self._get_orientation_factor(b.get('orientation', 'south'))
                
                # Prioritize buildings with good energy labels
                label = b.get('energy_label', 'D')
                label_bonus = 1.2 if label in ['A', 'B'] else 1.0
                
                # Estimate: 0.15 kW per m² with orientation adjustment
                potential_kw += roof_area * 0.15 * orientation_factor * label_bonus
        
        return potential_kw
    
    def _get_orientation_factor(self, orientation: str) -> float:
        """Get solar efficiency factor based on roof orientation"""
        factors = {
            'south': 1.0,
            'south-east': 0.95,
            'south-west': 0.95,
            'east': 0.85,
            'west': 0.85,
            'north-east': 0.65,
            'north-west': 0.65,
            'north': 0.5,
            'flat': 0.9
        }
        return factors.get(orientation, 0.8)
    
    def _calculate_temporal_diversity(self, buildings: List[Dict]) -> float:
        """Calculate temporal pattern diversity"""
        peak_hours = [b.get('peak_hour', 18) for b in buildings]
        
        if len(set(peak_hours)) == 1:
            return 0.0
        
        # Normalize by max possible spread (24 hours)
        spread = max(peak_hours) - min(peak_hours)
        return min(spread / 12, 1.0)
    
    def _estimate_consumption_correlation(self, buildings: List[Dict]) -> float:
        """Estimate average correlation between building consumptions"""
        types = [b.get('type', 'residential') for b in buildings]
        
        # Simple heuristic: same type = high correlation
        type_counts = pd.Series(types).value_counts()
        dominant_type_ratio = type_counts.max() / len(types)
        
        # High ratio = high correlation
        return dominant_type_ratio
    
    def _aggregate_to_mv(self, mv_id: str, lv_metrics: List[LVGroupMetrics]) -> MVStationMetrics:
        """Aggregate LV metrics to MV station level"""
        
        # Count high-potential LV groups
        high_potential = sum(1 for lv in lv_metrics 
                           if lv.get_summary_score() > self.config['high_potential_threshold'])
        
        # Calculate averages and totals
        avg_diversity = np.mean([lv.function_diversity for lv in lv_metrics])
        avg_poor_ratio = np.mean([lv.poor_label_ratio for lv in lv_metrics])
        total_poor = sum(lv.building_count * lv.poor_label_ratio for lv in lv_metrics)
        
        # Energy aggregation
        total_demand = sum(lv.total_demand_mwh for lv in lv_metrics)
        peak_demand = sum(lv.peak_demand_kw for lv in lv_metrics) * 0.8  # Diversity factor
        total_solar = sum(lv.existing_solar_kw for lv in lv_metrics)
        total_potential = sum(lv.solar_potential_kw for lv in lv_metrics)
        
        # Complementarity index
        avg_complementarity = np.mean([1 - abs(lv.consumption_correlation) for lv in lv_metrics])
        
        # Determine strategy
        strategy = self._determine_strategy(
            avg_diversity, avg_poor_ratio, avg_complementarity, total_potential
        )
        
        # Calculate intervention priority
        intervention_priority = (
            avg_poor_ratio * 0.4 +  # Urgent retrofits
            (total_potential / (total_potential + total_solar + 1e-6)) * 0.3 +  # Solar opportunity
            avg_diversity * 0.3  # Good for communities
        )
        
        return MVStationMetrics(
            mv_station_id=mv_id,
            lv_group_count=len(lv_metrics),
            total_building_count=sum(lv.building_count for lv in lv_metrics),
            avg_function_diversity=avg_diversity,
            avg_poor_label_ratio=avg_poor_ratio,
            total_poor_buildings=int(total_poor),
            total_demand_mwh=total_demand,
            peak_demand_kw=peak_demand,
            total_solar_kw=total_solar,
            total_solar_potential_kw=total_potential,
            high_potential_lv_groups=high_potential,
            intervention_priority=intervention_priority,
            complementarity_index=avg_complementarity,
            strategy_recommendation=strategy
        )
    
    def _determine_strategy(self, diversity: float, poor_ratio: float, 
                           complementarity: float, solar_potential: float) -> str:
        """Determine recommended strategy for MV station"""
        
        scores = {
            'energy_community': diversity * 0.5 + complementarity * 0.5,
            'retrofit_focus': poor_ratio,
            'solar_priority': min(solar_potential / 1000, 1.0),  # Normalize by 1MW
            'mixed': (diversity + poor_ratio + complementarity) / 3
        }
        
        return max(scores, key=scores.get)
    
    def _create_summary_dataframe(self) -> pd.DataFrame:
        """Create summary DataFrame for reporting"""
        
        # MV station summary
        mv_data = []
        for mv_id, metrics in self.mv_assessments.items():
            mv_data.append({
                'mv_station': mv_id,
                'lv_groups': metrics.lv_group_count,
                'buildings': metrics.total_building_count,
                'diversity': f"{metrics.avg_function_diversity:.2f}",
                'poor_labels': f"{metrics.avg_poor_label_ratio:.1%}",
                'poor_buildings': metrics.total_poor_buildings,
                'demand_mwh': f"{metrics.total_demand_mwh:.0f}",
                'solar_kw': f"{metrics.total_solar_kw:.0f}",
                'solar_potential_kw': f"{metrics.total_solar_potential_kw:.0f}",
                'high_potential_lvs': metrics.high_potential_lv_groups,
                'priority_score': f"{metrics.get_priority_score():.1f}",
                'strategy': metrics.strategy_recommendation
            })
        
        df = pd.DataFrame(mv_data)
        
        # Sort by priority
        if not df.empty:
            df['_priority'] = df['priority_score'].str.replace('', '').astype(float)
            df = df.sort_values('_priority', ascending=False)
            df = df.drop('_priority', axis=1)
        
        return df
    
    def generate_report(self, results: Dict, summary_df: pd.DataFrame, 
                       output_path: Optional[Path] = None) -> str:
        """Generate comprehensive assessment report"""
        
        report = []
        report.append("=" * 80)
        report.append("INITIAL NETWORK ASSESSMENT REPORT")
        report.append(f"Generated: {results['timestamp']}")
        report.append("=" * 80)
        report.append("")
        
        # Overview
        report.append("📊 NETWORK OVERVIEW")
        report.append(f"  HV Substations: {results['hv_substations']}")
        report.append(f"  MV Stations:    {results['mv_stations']}")
        report.append(f"  LV Groups:      {results['lv_groups']}")
        report.append("")
        
        # Top MV stations for intervention
        report.append("🎯 TOP PRIORITY MV STATIONS")
        report.append("")
        
        if not summary_df.empty:
            for idx, row in summary_df.head(5).iterrows():
                report.append(f"  {idx+1}. {row['mv_station']} (Score: {row['priority_score']})")
                report.append(f"     Strategy: {row['strategy'].upper()}")
                report.append(f"     Scale: {row['buildings']} buildings in {row['lv_groups']} LV groups")
                report.append(f"     Issues: {row['poor_buildings']} buildings need retrofit")
                report.append(f"     Solar: {row['solar_potential_kw']} kW potential")
                report.append("")
        
        # Strategy distribution
        report.append("📈 RECOMMENDED STRATEGIES")
        if not summary_df.empty:
            strategy_counts = summary_df['strategy'].value_counts()
            for strategy, count in strategy_counts.items():
                report.append(f"  {strategy}: {count} MV stations")
        report.append("")
        
        # Detailed MV analysis
        report.append("📋 DETAILED MV STATION ANALYSIS")
        report.append("-" * 80)
        
        for mv_id, metrics in sorted(self.mv_assessments.items(), 
                                    key=lambda x: x[1].get_priority_score(), 
                                    reverse=True)[:10]:
            report.append(f"\n{mv_id}")
            report.append(f"  Priority Score: {metrics.get_priority_score():.1f}/10")
            report.append(f"  Strategy: {metrics.strategy_recommendation}")
            report.append(f"  Buildings: {metrics.total_building_count} across {metrics.lv_group_count} LV groups")
            report.append(f"  Diversity Index: {metrics.avg_function_diversity:.2f}")
            report.append(f"  Poor Labels: {metrics.avg_poor_label_ratio:.1%} ({metrics.total_poor_buildings} buildings)")
            report.append(f"  Energy Demand: {metrics.total_demand_mwh:.0f} MWh/year")
            report.append(f"  Solar Installed: {metrics.total_solar_kw:.0f} kW")
            report.append(f"  Solar Potential: {metrics.total_solar_potential_kw:.0f} kW")
            report.append(f"  Complementarity: {metrics.complementarity_index:.2f}")
            report.append(f"  High-Potential LV Groups: {metrics.high_potential_lv_groups}")
        
        # Key findings
        report.append("")
        report.append("🔍 KEY FINDINGS")
        
        if not summary_df.empty:
            total_poor = sum(self.mv_assessments[mv].total_poor_buildings 
                           for mv in self.mv_assessments)
            total_solar_potential = sum(self.mv_assessments[mv].total_solar_potential_kw 
                                       for mv in self.mv_assessments)
            
            report.append(f"  • {total_poor} buildings need energy efficiency upgrades")
            report.append(f"  • {total_solar_potential:.0f} kW of untapped solar potential")
            report.append(f"  • {len([m for m in self.mv_assessments.values() if m.avg_function_diversity > 0.5])} "
                         f"MV stations have excellent diversity for energy communities")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if output_path:
            # Remove emojis for file saving to avoid encoding issues
            clean_report = report_text.replace('📊', '[DATA]').replace('🎯', '[TARGET]')
            clean_report = clean_report.replace('📈', '[CHART]').replace('📋', '[LIST]')
            clean_report = clean_report.replace('🔍', '[FIND]').replace('⭐', '[STAR]')
            clean_report = clean_report.replace('🌟', '[STAR]').replace('⚠️', '[WARN]')
            clean_report = clean_report.replace('🚨', '[ALERT]').replace('⚡', '[POWER]')
            clean_report = clean_report.replace('🔧', '[TOOLS]').replace('🌈', '[DIVERSE]')
            clean_report = clean_report.replace('💡', '[IDEA]')
            
            output_path.write_text(clean_report, encoding='utf-8')
            logger.info(f"Report saved to {output_path}")
        
        return report_text
    
    def select_mv_station(self, strategy: Optional[str] = None) -> Optional[str]:
        """
        Select best MV station based on strategy
        
        Args:
            strategy: 'energy_community', 'retrofit_focus', 'solar_priority', or None for best overall
            
        Returns:
            Selected MV station ID
        """
        if not self.mv_assessments:
            logger.warning("No MV assessments available")
            return None
        
        if strategy:
            # Filter by strategy
            candidates = [
                (mv_id, metrics) for mv_id, metrics in self.mv_assessments.items()
                if metrics.strategy_recommendation == strategy
            ]
            
            if not candidates:
                logger.warning(f"No MV stations with strategy: {strategy}")
                candidates = list(self.mv_assessments.items())
        else:
            candidates = list(self.mv_assessments.items())
        
        # Sort by priority score
        candidates.sort(key=lambda x: x[1].get_priority_score(), reverse=True)
        
        if candidates:
            selected = candidates[0][0]
            logger.info(f"Selected MV station: {selected} "
                       f"(score: {candidates[0][1].get_priority_score():.1f})")
            return selected
        
        return None
    
    def get_lv_groups_for_mv(self, mv_station_id: str) -> List[str]:
        """Get list of LV group IDs under specified MV station"""
        lv_groups = []
        
        for hv_data in self.hierarchy_data.values():
            for mv_station in hv_data.get('mv_stations', []):
                if mv_station['id'] == mv_station_id:
                    for lv_group in mv_station.get('lv_groups', []):
                        lv_groups.append(lv_group.get('id'))
        
        return lv_groups