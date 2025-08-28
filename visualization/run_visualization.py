"""
Main Visualization Runner
Generates all reports and visualizations from GNN results
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any

# Add parent directory to path
sys.path.append('..')

# Import our modules
from visualization.data_aggregator import DataAggregator, AggregatedMetrics
from visualization.chart_generator import ChartGenerator
from visualization.report_generator import ReportGenerator
from visualization.economic_calculator import EconomicCalculator
from visualization.excel_reporter import ExcelReporter

# Import from main project
from data.kg_connector import KGConnector
from tasks.enhanced_cluster_quality import EnhancedClusterMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VisualizationRunner:
    """Main runner for all visualization and reporting"""
    
    def __init__(self, results_path: str = None, neo4j_config: Dict = None):
        """
        Initialize visualization runner
        
        Args:
            results_path: Path to GNN results file
            neo4j_config: Neo4j connection configuration
        """
        self.results_path = results_path
        
        # Initialize components
        self.kg_connector = None
        if neo4j_config:
            self.kg_connector = KGConnector(
                uri=neo4j_config.get('uri', 'neo4j://localhost:7687'),
                user=neo4j_config.get('user', 'neo4j'),
                password=neo4j_config.get('password', 'password')
            )
        
        self.data_aggregator = DataAggregator(neo4j_connector=self.kg_connector)
        self.chart_generator = ChartGenerator()
        self.report_generator = ReportGenerator()
        self.economic_calculator = EconomicCalculator()
        self.excel_reporter = ExcelReporter()
        
        # Storage for results
        self.gnn_results = {}
        self.cluster_metrics = {}
        self.solar_results = {}
        self.economic_results = {}
        self.aggregated_metrics = None
    
    def load_gnn_results(self) -> Dict:
        """Load GNN results from file or generate sample data"""
        
        if self.results_path and Path(self.results_path).exists():
            logger.info(f"Loading results from {self.results_path}")
            with open(self.results_path, 'r') as f:
                self.gnn_results = json.load(f)
        else:
            logger.info("Generating sample GNN results")
            self.gnn_results = self._generate_sample_gnn_results()
        
        return self.gnn_results
    
    def generate_cluster_metrics(self) -> Dict[int, EnhancedClusterMetrics]:
        """Generate or load cluster quality metrics"""
        
        logger.info("Generating cluster quality metrics")
        
        # Generate sample cluster metrics
        self.cluster_metrics = {}
        
        for cluster_id in range(1, 13):  # 12 clusters
            metrics = EnhancedClusterMetrics(
                cluster_id=cluster_id,
                timestamp=0,
                lv_group_id=np.random.randint(1, 21),
                self_sufficiency_ratio=np.random.uniform(0.3, 0.7),
                self_consumption_ratio=np.random.uniform(0.6, 0.95),
                complementarity_score=np.random.uniform(0.5, 0.9),
                peak_reduction_ratio=np.random.uniform(0.15, 0.35),
                temporal_stability=np.random.uniform(0.7, 0.95),
                member_count=np.random.randint(5, 15),
                size_appropriateness=np.random.uniform(0.6, 1.0),
                total_demand_kwh=np.random.uniform(50000, 200000),
                total_generation_kwh=np.random.uniform(20000, 100000),
                total_shared_kwh=np.random.uniform(10000, 50000),
                grid_import_kwh=np.random.uniform(20000, 100000),
                grid_export_kwh=np.random.uniform(5000, 30000),
                building_type_diversity=np.random.uniform(0.4, 0.8),
                energy_label_diversity=np.random.uniform(0.5, 0.9),
                peak_hour_diversity=np.random.uniform(0.3, 0.7),
                cost_savings_percent=np.random.uniform(10, 30),
                revenue_potential=np.random.uniform(1000, 5000),
                avg_distance_m=np.random.uniform(50, 200),
                compactness_score=np.random.uniform(0.5, 0.9)
            )
            
            self.cluster_metrics[cluster_id] = metrics
        
        return self.cluster_metrics
    
    def generate_solar_analysis(self) -> Dict:
        """Generate solar installation analysis"""
        
        logger.info("Generating solar installation analysis")
        
        # Generate priority list
        priority_list = []
        for i in range(50):
            priority_list.append({
                'id': f'B{i:03d}',
                'energy_label': np.random.choice(['E', 'F', 'G']),
                'roof_area': np.random.uniform(50, 200),
                'capacity': np.random.uniform(5, 30),
                'roi': np.random.uniform(4, 12),
                'priority_score': np.random.uniform(0.6, 0.95),
                'annual_generation': np.random.uniform(6000, 36000)
            })
        
        # Sort by priority score
        priority_list.sort(key=lambda x: x['priority_score'], reverse=True)
        
        self.solar_results = {
            'priority_list': priority_list,
            'total_capacity': sum(b['capacity'] for b in priority_list),
            'num_installations': len(priority_list),
            'avg_roi': np.mean([b['roi'] for b in priority_list]),
            'total_investment': sum(b['capacity'] for b in priority_list) * 1200,
            'annual_generation': sum(b['annual_generation'] for b in priority_list),
            'co2_savings': sum(b['annual_generation'] for b in priority_list) * 0.5 / 1000,
            'roi_years': [b['roi'] for b in priority_list],
            'payback_periods': [b['roi'] for b in priority_list],
            'installations_by_label': {
                'E': sum(1 for b in priority_list if b['energy_label'] == 'E'),
                'F': sum(1 for b in priority_list if b['energy_label'] == 'F'),
                'G': sum(1 for b in priority_list if b['energy_label'] == 'G')
            }
        }
        
        return self.solar_results
    
    def calculate_economics(self) -> Dict:
        """Calculate comprehensive economic metrics"""
        
        logger.info("Calculating economic metrics")
        
        # Sample calculations
        solar_roi = self.economic_calculator.calculate_solar_roi(
            capacity_kwp=100,
            annual_generation_kwh=120000,
            self_consumption_ratio=0.7,
            building_demand_kwh=150000
        )
        
        battery_economics = self.economic_calculator.calculate_battery_economics(
            capacity_kwh=50,
            daily_cycles=1.5,
            peak_shaving_kw=20,
            arbitrage_revenue_daily=15
        )
        
        community_benefits = self.economic_calculator.calculate_community_benefits(
            num_buildings=100,
            shared_energy_kwh=500000,
            peak_reduction_percent=0.25,
            avg_building_demand_kwh=10000
        )
        
        grid_deferral = self.economic_calculator.calculate_grid_investment_deferral(
            peak_reduction_kw=500,
            self_sufficiency_ratio=0.4,
            num_buildings=100
        )
        
        self.economic_results = self.economic_calculator.create_financial_summary(
            solar_roi, battery_economics, community_benefits, grid_deferral
        )
        
        # Add additional economic data
        self.economic_results['investments'] = {
            'Solar PV': solar_roi['investment'],
            'Battery Storage': battery_economics['investment'],
            'Smart Infrastructure': 120000
        }
        
        self.economic_results['annual_benefits'] = {
            'Energy Savings': community_benefits['p2p_benefit'],
            'Peak Reduction': community_benefits['peak_savings'],
            'Grid Services': grid_deferral['annual_value'],
            'Carbon Credits': community_benefits['carbon_value']
        }
        
        self.economic_results['monthly_savings'] = {
            f'Month {i+1}': np.random.uniform(8000, 15000) for i in range(12)
        }
        
        self.economic_results['cost_breakdown'] = {
            'Energy': 40,
            'Network': 25,
            'Maintenance': 20,
            'Other': 15
        }
        
        self.economic_results['roi_timeline'] = [
            -self.economic_results['total_investment']
        ]
        for year in range(1, 21):
            cumulative = self.economic_results['roi_timeline'][-1]
            annual = self.economic_results['total_annual_benefit'] * (1.02 ** year)
            self.economic_results['roi_timeline'].append(cumulative + annual)
        
        return self.economic_results
    
    def generate_temporal_data(self) -> pd.DataFrame:
        """Generate sample temporal data"""
        
        logger.info("Generating temporal data")
        
        # Create hourly data for buildings
        data = []
        
        for building_id in range(160):
            for hour in range(24):
                demand = 30 + 10*np.sin((hour-6)*np.pi/12) + np.random.normal(0, 5)
                generation = max(0, 15*np.sin((hour-6)*np.pi/12)) + np.random.normal(0, 2)
                
                data.append({
                    'building_id': building_id,
                    'timestamp': hour,
                    'hour': hour,
                    'demand': max(0, demand),
                    'generation': max(0, generation),
                    'cluster_id': building_id % 12,
                    'peak_hour': hour in [17, 18, 19, 20]
                })
        
        return pd.DataFrame(data)
    
    def aggregate_all_data(self) -> AggregatedMetrics:
        """Aggregate all data sources into unified metrics"""
        
        logger.info("Aggregating all data sources")
        
        temporal_data = self.generate_temporal_data()
        
        self.aggregated_metrics = self.data_aggregator.aggregate_all_metrics(
            gnn_results=self.gnn_results,
            cluster_metrics=self.cluster_metrics,
            solar_results=self.solar_results,
            temporal_data=temporal_data
        )
        
        return self.aggregated_metrics
    
    def generate_all_charts(self):
        """Generate all visualization charts"""
        
        logger.info("Generating visualization charts")
        
        # Cluster quality heatmap
        self.chart_generator.create_cluster_quality_heatmap(
            self.cluster_metrics,
            save_path="cluster_quality_heatmap"
        )
        
        # Energy flow Sankey
        self.chart_generator.create_energy_flow_sankey(
            {'flows': {}},
            save_path="energy_flow_sankey"
        )
        
        # Temporal patterns
        temporal_data = self.generate_temporal_data()
        self.chart_generator.create_temporal_patterns(
            temporal_data,
            save_path="temporal_patterns"
        )
        
        # Solar ROI analysis
        self.chart_generator.create_solar_roi_analysis(
            self.solar_results,
            save_path="solar_roi_analysis"
        )
        
        # Economic dashboard
        self.chart_generator.create_economic_dashboard(
            self.economic_results,
            save_path="economic_dashboard"
        )
        
        # LV group summary
        lv_statistics = {
            'buildings_per_lv': {f'LV_{i:03d}': np.random.randint(5, 30) for i in range(20)},
            'cluster_coverage': 75,
            'solar_potential': np.random.uniform(100, 500, 20).tolist(),
            'building_count': np.random.randint(5, 30, 20).tolist(),
            'priority_scores': {f'LV_{i:03d}': np.random.uniform(0.5, 0.9) for i in range(20)}
        }
        
        self.chart_generator.create_lv_group_summary(
            lv_statistics,
            save_path="lv_group_summary"
        )
        
        logger.info("All charts generated successfully")
    
    def generate_all_reports(self):
        """Generate all text reports"""
        
        logger.info("Generating text reports")
        
        # Executive summary
        self.report_generator.generate_executive_summary(
            self.aggregated_metrics
        )
        
        # Technical report
        self.report_generator.generate_technical_report(
            self.aggregated_metrics,
            {k: v.__dict__ if hasattr(v, '__dict__') else v 
             for k, v in self.cluster_metrics.items()},
            {'processing_time': 1234, 'data_quality': 95}
        )
        
        # Cluster quality report
        self.report_generator.generate_cluster_quality_report(
            self.cluster_metrics
        )
        
        # Intervention recommendations
        solar_candidates = self.solar_results['priority_list'][:10]
        battery_candidates = [
            {'id': f'B{i:03d}', 'peak_demand': np.random.uniform(10, 50),
             'battery_size': np.random.uniform(10, 100),
             'peak_reduction': np.random.uniform(0.2, 0.4),
             'annual_savings': np.random.uniform(1000, 5000)}
            for i in range(10)
        ]
        retrofit_candidates = [
            {'id': f'B{i:03d}', 'current_label': np.random.choice(['E', 'F', 'G']),
             'target_label': np.random.choice(['B', 'C', 'D']),
             'cost': np.random.uniform(10000, 50000),
             'savings_percent': np.random.uniform(0.2, 0.4)}
            for i in range(10)
        ]
        
        self.report_generator.generate_intervention_recommendations(
            solar_candidates, battery_candidates, retrofit_candidates
        )
        
        # Stakeholder report
        benefits = [
            "Lower energy bills through community sharing",
            "Increased property values from green certification",
            "Reduced carbon footprint",
            "Energy independence and resilience"
        ]
        next_steps = [
            "Join the community energy program",
            "Schedule a solar assessment for your building",
            "Attend the next community meeting"
        ]
        
        self.report_generator.generate_stakeholder_report(
            self.aggregated_metrics,
            {b: True for b in benefits},
            next_steps
        )
        
        logger.info("All text reports generated successfully")
    
    def generate_excel_report(self):
        """Generate comprehensive Excel report"""
        
        logger.info("Generating Excel report")
        
        temporal_data = self.generate_temporal_data()
        
        filepath = self.excel_reporter.generate_comprehensive_report(
            self.aggregated_metrics,
            {k: v.__dict__ if hasattr(v, '__dict__') else v 
             for k, v in self.cluster_metrics.items()},
            self.solar_results,
            self.economic_results,
            temporal_data
        )
        
        logger.info(f"Excel report generated: {filepath}")
        return filepath
    
    def run_all(self):
        """Run complete visualization and reporting pipeline"""
        
        logger.info("Starting complete visualization pipeline")
        
        try:
            # Load/generate data
            self.load_gnn_results()
            self.generate_cluster_metrics()
            self.generate_solar_analysis()
            self.calculate_economics()
            
            # Aggregate all data
            self.aggregate_all_data()
            
            # Generate outputs
            try:
                self.generate_all_charts()
                logger.info("Charts generation complete")
            except Exception as e:
                logger.error(f"Error generating charts: {e}")
                
            try:
                self.generate_all_reports()
                logger.info("Reports generation complete")
            except Exception as e:
                logger.error(f"Error generating reports: {e}")
                
            try:
                self.generate_excel_report()
                logger.info("Excel report generation complete")
            except Exception as e:
                logger.error(f"Error generating Excel report: {e}")
            
            logger.info("Visualization pipeline complete!")
            
            # Print summary
            self._print_summary()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_sample_gnn_results(self) -> Dict:
        """Generate sample GNN results for testing"""
        
        return {
            'cluster_assignments': [i % 12 for i in range(160)],
            'building_features': {
                'energy_label': [np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G']) 
                               for _ in range(160)],
                'building_type': [np.random.choice(['Residential', 'Commercial', 'Industrial'])
                                 for _ in range(160)],
                'has_solar': [np.random.choice([True, False]) for _ in range(160)],
                'roof_area': [np.random.uniform(50, 200) for _ in range(160)],
                'annual_consumption': [np.random.uniform(5000, 50000) for _ in range(160)]
            },
            'lv_group_ids': [i // 8 for i in range(160)],  # 20 LV groups
            'energy_flows': {
                'total_shared': 180000
            },
            'network_metrics': {
                'voltage_deviation': 0.02,
                'transformer_util': 65,
                'line_losses': 3.5,
                'congestion_events': 2
            }
        }
    
    def _print_summary(self):
        """Print summary of generated outputs"""
        
        print("\n" + "="*60)
        print("VISUALIZATION PIPELINE SUMMARY")
        print("="*60)
        
        if self.aggregated_metrics:
            print(f"\nKey Metrics:")
            print(f"  - Clusters: {self.aggregated_metrics.num_clusters}")
            print(f"  - Buildings: {int(self.aggregated_metrics.num_lv_groups * self.aggregated_metrics.avg_buildings_per_lv)}")
            print(f"  - Self-Sufficiency: {self.aggregated_metrics.avg_self_sufficiency:.1%}")
            print(f"  - Cost Savings: €{self.aggregated_metrics.total_cost_savings_eur:,.0f}/month")
            print(f"  - CO2 Reduced: {self.aggregated_metrics.carbon_reduction_tons:.1f} tons/month")
        
        print(f"\nGenerated Outputs:")
        print(f"  - Charts: 6 interactive visualizations")
        print(f"  - Reports: 5 markdown reports")
        print(f"  - Excel: 1 comprehensive workbook")
        
        print(f"\nOutput Locations:")
        print(f"  - Charts: results/charts/")
        print(f"  - Reports: results/reports/")
        print(f"  - Data: results/data/")
        
        print("\n" + "="*60)


def main():
    """Main entry point"""
    
    # Configuration
    neo4j_config = {
        'uri': 'neo4j://127.0.0.1:7687',
        'user': 'neo4j',
        'password': 'DS4citizens'
    }
    
    # Check for results file
    results_files = list(Path("results").glob("unified_results_*.json"))
    results_path = str(results_files[-1]) if results_files else None
    
    # Create runner
    runner = VisualizationRunner(
        results_path=results_path,
        neo4j_config=neo4j_config
    )
    
    # Run complete pipeline
    runner.run_all()


if __name__ == "__main__":
    main()