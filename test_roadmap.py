"""
Test script for Solar Roadmap functionality
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import logging
from pathlib import Path
from gnn_main import UnifiedGNNSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_roadmap_generation():
    """Test the roadmap generation functionality"""
    
    logger.info("="*60)
    logger.info("TESTING SOLAR ROADMAP FUNCTIONALITY")
    logger.info("="*60)
    
    # Initialize system
    logger.info("\nInitializing GNN System...")
    system = UnifiedGNNSystem()
    
    # Run initial assessment if needed
    if not system.selected_mv_station:
        logger.info("\nRunning initial assessment...")
        system.run_initial_assessment()
    
    # Train the model briefly
    logger.info("\nTraining model (quick mode)...")
    system.config['training']['phases']['unsupervised'] = 2
    system.config['training']['phases']['semi_supervised'] = 2
    system.config['training']['phases']['fine_tuning'] = 2
    system.train()
    
    # Generate roadmap for 20% penetration in 5 years
    logger.info("\n" + "="*60)
    logger.info("GENERATING SOLAR ROADMAP")
    logger.info("="*60)
    
    roadmap = system.generate_solar_roadmap(
        target_penetration=0.2,  # 20% of available roof area
        timeframe_years=5,
        strategy='cascade_optimized'
    )
    
    if roadmap:
        logger.info("\n" + "="*60)
        logger.info("ROADMAP SUMMARY")
        logger.info("="*60)
        logger.info(f"Target Penetration: {roadmap.target_penetration:.1%}")
        logger.info(f"Timeframe: {roadmap.timeframe_years} years")
        logger.info(f"Strategy: {roadmap.optimization_strategy}")
        logger.info(f"Total Investment: €{roadmap.total_investment:,.0f}")
        logger.info(f"Total Capacity: {roadmap.expected_benefits.get('total_capacity_mw', 0):.2f} MW")
        logger.info(f"CO2 Reduction: {roadmap.expected_benefits.get('annual_co2_reduction_tons', 0):.0f} tons/year")
        
        logger.info("\nYearly Breakdown:")
        for plan in roadmap.yearly_plans:
            logger.info(f"  Year {plan.year}:")
            logger.info(f"    - Buildings: {len(plan.target_installations)}")
            logger.info(f"    - Capacity: {plan.total_capacity_mw:.2f} MW")
            logger.info(f"    - Investment: €{plan.budget_required:,.0f}")
            logger.info(f"    - Cumulative Penetration: {plan.cumulative_penetration:.1%}")
            logger.info(f"    - Expected Self-Sufficiency: {plan.expected_self_sufficiency:.1%}")
        
        # Test cluster evolution
        if roadmap.cluster_evolution:
            logger.info("\nCluster Evolution:")
            for evolution_point in roadmap.cluster_evolution:
                logger.info(f"  Year {evolution_point['year']}: {evolution_point['num_clusters']} clusters, "
                          f"Stability: {evolution_point['stability_score']:.1%}, "
                          f"Self-sufficiency: {evolution_point['self_sufficiency']:.1%}")
        
        # Export roadmap
        logger.info("\nExporting roadmap to Excel...")
        export_path = Path('results') / 'roadmap_test.xlsx'
        export_path.parent.mkdir(exist_ok=True)
        system.roadmap_planner.export_roadmap_to_excel(roadmap, str(export_path))
        logger.info(f"Roadmap exported to: {export_path}")
        
        # Test progress tracking (simulate Year 1 completion)
        logger.info("\n" + "="*60)
        logger.info("TESTING PROGRESS TRACKING")
        logger.info("="*60)
        
        if roadmap.yearly_plans:
            year_1_plan = roadmap.yearly_plans[0]
            simulated_installations = year_1_plan.target_installations[:5]  # Simulate partial completion
            
            progress = system.track_roadmap_progress(
                completed_installations=simulated_installations,
                years_elapsed=1.0
            )
            
            logger.info("\nProgress Report:")
            for metric, values in progress.items():
                if isinstance(values, dict) and 'percentage_complete' in values:
                    logger.info(f"  {metric}: {values['percentage_complete']:.1f}% complete")
        
        # Generate visualizations
        logger.info("\n" + "="*60)
        logger.info("GENERATING VISUALIZATIONS")
        logger.info("="*60)
        
        from visualization.roadmap_visualizer import RoadmapVisualizer
        visualizer = RoadmapVisualizer()
        
        # Create timeline
        timeline_fig = visualizer.create_timeline_gantt(roadmap)
        timeline_fig.write_html('results/roadmap_timeline.html')
        logger.info("Timeline saved to: results/roadmap_timeline.html")
        
        # Create penetration progress
        progress_fig = visualizer.create_penetration_progress(roadmap)
        progress_fig.write_html('results/penetration_progress.html')
        logger.info("Progress chart saved to: results/penetration_progress.html")
        
        # Create investment timeline
        investment_fig = visualizer.create_investment_timeline(roadmap)
        investment_fig.write_html('results/investment_timeline.html')
        logger.info("Investment timeline saved to: results/investment_timeline.html")
        
        # Create cluster evolution
        evolution_fig = visualizer.create_cluster_evolution(roadmap)
        evolution_fig.write_html('results/cluster_evolution.html')
        logger.info("Cluster evolution saved to: results/cluster_evolution.html")
        
        # Create benefits summary
        benefits_fig = visualizer.create_benefits_summary(roadmap)
        benefits_fig.write_html('results/benefits_summary.html')
        logger.info("Benefits summary saved to: results/benefits_summary.html")
        
        # Export comprehensive report
        visualizer.export_roadmap_report(
            roadmap,
            'results/roadmap_report.json',
            include_charts=True
        )
        logger.info("Comprehensive report saved to: results/roadmap_report.json and .html")
        
        logger.info("\n" + "="*60)
        logger.info("ROADMAP TESTING COMPLETE!")
        logger.info("="*60)
    else:
        logger.error("Failed to generate roadmap")

if __name__ == "__main__":
    test_roadmap_generation()