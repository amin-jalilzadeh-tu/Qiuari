"""
Test Model Validation and Roadmap Generation
Comprehensive testing script to validate GNN model and test roadmap functionality
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import logging
from pathlib import Path
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import system components
from gnn_main import UnifiedGNNSystem
from validation.model_requirements_validator import ModelRequirementsValidator
from visualization.roadmap_visualizer import RoadmapVisualizer


def test_model_validation(system: UnifiedGNNSystem):
    """
    Test model against all requirements
    """
    logger.info("\n" + "="*80)
    logger.info("TESTING MODEL VALIDATION")
    logger.info("="*80 + "\n")
    
    # Initialize validator
    validator = ModelRequirementsValidator(system.config)
    
    # Get test data
    test_data = system._prepare_data()
    
    # Run comprehensive validation
    report = validator.validate_model(
        model=system.model,
        test_data=test_data,
        gnn_system=system,
        detailed=True
    )
    
    # Save reports
    validator.save_report("reports/validation_report.json")
    validator.generate_html_report("reports/validation_report.html")
    
    # Print summary by category
    logger.info("\n" + "="*60)
    logger.info("VALIDATION SUMMARY BY CATEGORY")
    logger.info("="*60)
    
    from collections import defaultdict
    by_category = defaultdict(list)
    for check in report.requirements_checks:
        by_category[check.category].append(check)
    
    for category, checks in by_category.items():
        passed = sum(1 for c in checks if c.status == 'passed')
        failed = sum(1 for c in checks if c.status == 'failed')
        partial = sum(1 for c in checks if c.status == 'partial')
        avg_score = np.mean([c.score for c in checks])
        
        logger.info(f"\n{category}:")
        logger.info(f"  Passed: {passed}/{len(checks)}")
        logger.info(f"  Failed: {failed}/{len(checks)}")
        logger.info(f"  Partial: {partial}/{len(checks)}")
        logger.info(f"  Average Score: {avg_score:.1f}/100")
    
    return report


def test_roadmap_generation(system: UnifiedGNNSystem):
    """
    Test solar roadmap generation with different scenarios
    """
    logger.info("\n" + "="*80)
    logger.info("TESTING ROADMAP GENERATION")
    logger.info("="*80 + "\n")
    
    scenarios = [
        {
            'name': '20% in 5 years (Conservative)',
            'target_penetration': 0.2,
            'timeframe_years': 5,
            'strategy': 'linear'
        },
        {
            'name': '30% in 5 years (Moderate)',
            'target_penetration': 0.3,
            'timeframe_years': 5,
            'strategy': 'cascade_optimized'
        },
        {
            'name': '50% in 7 years (Ambitious)',
            'target_penetration': 0.5,
            'timeframe_years': 7,
            'strategy': 'accelerated'
        },
        {
            'name': '40% in 5 years (Cluster-balanced)',
            'target_penetration': 0.4,
            'timeframe_years': 5,
            'strategy': 'cluster_balanced'
        }
    ]
    
    roadmaps = []
    
    for scenario in scenarios:
        logger.info(f"\nScenario: {scenario['name']}")
        logger.info("-" * 40)
        
        try:
            # Generate roadmap
            roadmap = system.generate_solar_roadmap(
                target_penetration=scenario['target_penetration'],
                timeframe_years=scenario['timeframe_years'],
                strategy=scenario['strategy']
            )
            
            if roadmap:
                roadmaps.append((scenario, roadmap))
                
                # Log summary
                logger.info(f"✅ Roadmap generated successfully")
                logger.info(f"  Total Investment: €{roadmap.total_investment:,.0f}")
                logger.info(f"  Total Capacity: {roadmap.expected_benefits.get('total_capacity_mw', 0):.2f} MW")
                logger.info(f"  CO2 Reduction: {roadmap.expected_benefits.get('annual_co2_reduction_tons', 0):.0f} tons/year")
                logger.info(f"  Final Self-sufficiency: {roadmap.expected_benefits.get('final_self_sufficiency', 0):.1%}")
                
                # Export to Excel
                export_path = f"reports/roadmap_{scenario['strategy']}_{scenario['target_penetration']:.0%}.xlsx"
                system.roadmap_planner.export_roadmap_to_excel(roadmap, export_path)
                logger.info(f"  Exported to: {export_path}")
            else:
                logger.error(f"❌ Failed to generate roadmap")
                
        except Exception as e:
            logger.error(f"❌ Error generating roadmap: {e}")
    
    return roadmaps


def test_cluster_evolution(system: UnifiedGNNSystem):
    """
    Test how clusters evolve over time with solar installations
    """
    logger.info("\n" + "="*80)
    logger.info("TESTING CLUSTER EVOLUTION")
    logger.info("="*80 + "\n")
    
    # Generate a roadmap
    roadmap = system.generate_solar_roadmap(
        target_penetration=0.3,
        timeframe_years=5,
        strategy='cascade_optimized'
    )
    
    if not roadmap:
        logger.error("Could not generate roadmap for evolution testing")
        return None
    
    # Simulate yearly installations and track cluster changes
    evolution_data = []
    
    for year, plan in enumerate(roadmap.yearly_plans, 1):
        logger.info(f"\nYear {year}:")
        logger.info(f"  Installing solar on {len(plan.target_installations)} buildings")
        logger.info(f"  Total capacity: {plan.total_capacity_mw:.2f} MW")
        
        # Track progress
        progress = system.track_roadmap_progress(
            completed_installations=plan.target_installations,
            years_elapsed=year
        )
        
        if progress:
            evolution_data.append({
                'year': year,
                'installations': len(plan.target_installations),
                'penetration': plan.cumulative_penetration,
                'self_sufficiency': plan.expected_self_sufficiency,
                'num_clusters': len(set(plan.cluster_assignments.values())) if plan.cluster_assignments else 0,
                'on_track': progress.get('area', {}).get('on_track', False)
            })
            
            logger.info(f"  Progress: {progress.get('area', {}).get('percentage_complete', 0):.1f}% complete")
            logger.info(f"  Status: {'On track' if progress.get('area', {}).get('on_track', False) else 'Behind schedule'}")
    
    return evolution_data


def visualize_validation_results(report):
    """
    Create visualization of validation results
    """
    logger.info("\nGenerating validation visualizations...")
    
    # Prepare data
    categories = {}
    for check in report.requirements_checks:
        if check.category not in categories:
            categories[check.category] = {'passed': 0, 'failed': 0, 'partial': 0, 'not_tested': 0}
        categories[check.category][check.status] += 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Overall compliance gauge
    ax = axes[0, 0]
    compliance = report.overall_compliance
    colors = ['red' if compliance < 50 else 'orange' if compliance < 70 else 'yellow' if compliance < 90 else 'green']
    ax.pie([compliance, 100-compliance], 
           labels=[f'Compliant\n{compliance:.1f}%', ''],
           colors=[colors[0], 'lightgray'],
           startangle=90,
           counterclock=False)
    ax.set_title('Overall Compliance', fontsize=14, fontweight='bold')
    
    # 2. Category breakdown
    ax = axes[0, 1]
    category_names = list(categories.keys())
    status_counts = {
        'Passed': [categories[c]['passed'] for c in category_names],
        'Partial': [categories[c]['partial'] for c in category_names],
        'Failed': [categories[c]['failed'] for c in category_names],
        'Not Tested': [categories[c]['not_tested'] for c in category_names]
    }
    
    x = np.arange(len(category_names))
    width = 0.2
    colors_map = {'Passed': 'green', 'Partial': 'orange', 'Failed': 'red', 'Not Tested': 'gray'}
    
    for i, (status, counts) in enumerate(status_counts.items()):
        ax.bar(x + i*width, counts, width, label=status, color=colors_map[status], alpha=0.8)
    
    ax.set_xlabel('Category')
    ax.set_ylabel('Number of Requirements')
    ax.set_title('Requirements by Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(category_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Critical requirements status
    ax = axes[1, 0]
    critical_passed = sum(1 for c in report.requirements_checks 
                          if c.status == 'passed' and 'critical' in c.requirement_id)
    critical_failed = len(report.critical_failures)
    
    if critical_passed + critical_failed > 0:
        ax.pie([critical_passed, critical_failed],
               labels=[f'Passed\n{critical_passed}', f'Failed\n{critical_failed}'],
               colors=['green', 'red'],
               autopct='%1.0f%%',
               startangle=90)
        ax.set_title('Critical Requirements', fontsize=14, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No critical requirements data', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Critical Requirements', fontsize=14, fontweight='bold')
    
    # 4. Score distribution
    ax = axes[1, 1]
    scores = [check.score for check in report.requirements_checks]
    ax.hist(scores, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(scores), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(scores):.1f}')
    ax.set_xlabel('Score')
    ax.set_ylabel('Number of Requirements')
    ax.set_title('Score Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle(f'Model Validation Report - {report.model_name}', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('reports/validation_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info("Validation visualization saved to reports/validation_visualization.png")


def visualize_roadmap_comparison(roadmaps):
    """
    Compare different roadmap scenarios
    """
    if not roadmaps:
        logger.warning("No roadmaps to visualize")
        return
    
    logger.info("\nGenerating roadmap comparison visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Investment comparison
    ax = axes[0, 0]
    scenarios = [s['name'] for s, _ in roadmaps]
    investments = [r.total_investment for _, r in roadmaps]
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(roadmaps)))
    
    bars = ax.bar(range(len(scenarios)), investments, color=colors)
    ax.set_xlabel('Scenario')
    ax.set_ylabel('Total Investment (€)')
    ax.set_title('Investment Requirements', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.split('(')[0].strip() for s in scenarios], rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, inv in zip(bars, investments):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'€{inv/1e6:.1f}M', ha='center', va='bottom')
    
    # 2. Capacity timeline
    ax = axes[0, 1]
    for (scenario, roadmap), color in zip(roadmaps, colors):
        years = [p.year for p in roadmap.yearly_plans]
        cumulative_capacity = []
        total = 0
        for p in roadmap.yearly_plans:
            total += p.total_capacity_mw
            cumulative_capacity.append(total)
        
        ax.plot(years, cumulative_capacity, marker='o', 
                label=scenario['name'].split('(')[0].strip(),
                color=color, linewidth=2)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative Capacity (MW)')
    ax.set_title('Capacity Deployment Timeline', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 3. Penetration progression
    ax = axes[1, 0]
    for (scenario, roadmap), color in zip(roadmaps, colors):
        years = [0] + [p.year for p in roadmap.yearly_plans]
        penetration = [roadmap.current_penetration] + [p.cumulative_penetration for p in roadmap.yearly_plans]
        
        ax.plot(years, [p*100 for p in penetration], marker='s',
                label=f"{scenario['target_penetration']:.0%} target",
                color=color, linewidth=2)
    
    ax.set_xlabel('Year')
    ax.set_ylabel('Penetration Rate (%)')
    ax.set_title('Penetration Rate Evolution', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 4. Strategy comparison
    ax = axes[1, 1]
    strategies = [s['strategy'] for s, _ in roadmaps]
    final_self_suff = [r.expected_benefits.get('final_self_sufficiency', 0) for _, r in roadmaps]
    
    bars = ax.bar(range(len(strategies)), [s*100 for s in final_self_suff], color=colors)
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Final Self-Sufficiency (%)')
    ax.set_title('Expected Self-Sufficiency by Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(strategies)))
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, final_self_suff):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}', ha='center', va='bottom')
    
    plt.suptitle('Solar Roadmap Scenario Comparison', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('reports/roadmap_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    logger.info("Roadmap comparison saved to reports/roadmap_comparison.png")


def main():
    """
    Main test execution
    """
    logger.info("\n" + "="*80)
    logger.info("GNN MODEL VALIDATION AND ROADMAP TESTING")
    logger.info("="*80 + "\n")
    
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    # Initialize system
    logger.info("Initializing GNN system...")
    system = UnifiedGNNSystem(config_path='config/unified_config.yaml')
    
    # Option 1: Train a minimal model for testing
    logger.info("\nTraining model for validation (minimal epochs)...")
    system.train(num_epochs=3, run_assessment=False)  # Quick training for testing
    
    # Test 1: Model Validation
    logger.info("\n" + "="*60)
    logger.info("TEST 1: MODEL REQUIREMENTS VALIDATION")
    logger.info("="*60)
    validation_report = test_model_validation(system)
    visualize_validation_results(validation_report)
    
    # Test 2: Roadmap Generation
    logger.info("\n" + "="*60)
    logger.info("TEST 2: ROADMAP GENERATION")
    logger.info("="*60)
    roadmaps = test_roadmap_generation(system)
    if roadmaps:
        visualize_roadmap_comparison(roadmaps)
    
    # Test 3: Cluster Evolution
    logger.info("\n" + "="*60)
    logger.info("TEST 3: CLUSTER EVOLUTION")
    logger.info("="*60)
    evolution_data = test_cluster_evolution(system)
    
    if evolution_data:
        # Create evolution visualization
        df = pd.DataFrame(evolution_data)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Penetration over time
        ax = axes[0, 0]
        ax.plot(df['year'], df['penetration']*100, marker='o', linewidth=2, color='steelblue')
        ax.set_xlabel('Year')
        ax.set_ylabel('Penetration (%)')
        ax.set_title('Solar Penetration Progress')
        ax.grid(True, alpha=0.3)
        
        # Self-sufficiency over time
        ax = axes[0, 1]
        ax.plot(df['year'], df['self_sufficiency']*100, marker='s', linewidth=2, color='green')
        ax.set_xlabel('Year')
        ax.set_ylabel('Self-Sufficiency (%)')
        ax.set_title('Community Self-Sufficiency Evolution')
        ax.grid(True, alpha=0.3)
        
        # Annual installations
        ax = axes[1, 0]
        ax.bar(df['year'], df['installations'], color='orange', alpha=0.7)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Installations')
        ax.set_title('Annual Solar Installations')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Cluster evolution
        ax = axes[1, 1]
        if df['num_clusters'].sum() > 0:
            ax.plot(df['year'], df['num_clusters'], marker='D', linewidth=2, color='purple')
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Clusters')
            ax.set_title('Cluster Evolution')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No cluster data available', 
                    ha='center', va='center', transform=ax.transAxes)
        
        plt.suptitle('Solar Deployment Evolution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('reports/evolution_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        logger.info("Evolution analysis saved to reports/evolution_analysis.png")
    
    # Generate summary report
    logger.info("\n" + "="*80)
    logger.info("TESTING COMPLETE - SUMMARY")
    logger.info("="*80)
    
    logger.info(f"\n✅ Model Compliance: {validation_report.overall_compliance:.1f}%")
    logger.info(f"✅ Roadmaps Generated: {len(roadmaps)}")
    logger.info(f"✅ Critical Failures: {len(validation_report.critical_failures)}")
    
    if validation_report.overall_compliance >= 70:
        logger.info("\n🎉 Model PASSES basic requirements!")
    else:
        logger.info("\n⚠️ Model needs improvements to meet requirements")
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_compliance': validation_report.overall_compliance,
        'critical_failures': len(validation_report.critical_failures),
        'warnings': len(validation_report.warnings),
        'roadmaps_generated': len(roadmaps),
        'recommendations': validation_report.recommendations[:5]  # Top 5
    }
    
    with open('reports/test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("\nAll reports saved to 'reports/' directory")
    logger.info("Check validation_report.html for detailed results")


if __name__ == "__main__":
    main()