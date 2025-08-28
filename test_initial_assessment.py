"""
Test the initial assessment functionality
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from analysis.lv_mv_evaluator import EnhancedLVMVEvaluator, LVGroupMetrics, MVGroupMetrics
import numpy as np

def create_test_buildings(n_buildings=20, lv_id="LV_001"):
    """Create test building data"""
    buildings = []
    
    # Mix of building types for diversity
    types = ['residential'] * 12 + ['commercial'] * 5 + ['industrial'] * 3
    
    # Mix of energy labels (some poor for intervention need)
    labels = ['A'] * 2 + ['B'] * 3 + ['C'] * 4 + ['D'] * 5 + ['E'] * 3 + ['F'] * 2 + ['G'] * 1
    
    # Varied peak hours for temporal diversity
    peak_hours = [7, 8, 9, 12, 13, 17, 18, 18, 18, 19, 20, 21] + [18] * 8
    
    for i in range(n_buildings):
        building = {
            'id': f'BUILDING_{i:04d}',
            'lv_group_id': lv_id,
            'type': np.random.choice(types),
            'energy_label': np.random.choice(labels),
            'area': np.random.uniform(60, 200),
            'roof_area': np.random.uniform(30, 100),
            'orientation': np.random.choice(['N', 'S', 'E', 'W', 'SE', 'SW']),
            'has_solar': np.random.random() < 0.2,
            'peak_hour': np.random.choice(peak_hours)
        }
        buildings.append(building)
    
    return buildings

def test_lv_evaluation():
    """Test LV group evaluation"""
    print("Testing LV Group Evaluation")
    print("=" * 60)
    
    evaluator = EnhancedLVMVEvaluator()
    
    # Create test LV group with diverse buildings
    buildings = create_test_buildings(50, "LV_TEST_001")
    
    # Evaluate
    metrics = evaluator.evaluate_lv_group(buildings)
    
    print(f"LV Group: {metrics.lv_group_id}")
    print(f"Buildings: {metrics.building_count}")
    print(f"Planning Score: {metrics.get_planning_score():.2f}/10")
    print(f"Function Diversity: {metrics.function_diversity:.1%}")
    print(f"Temporal Diversity: {metrics.temporal_diversity:.1%}")
    print(f"Poor Labels (E/F/G): {metrics.poor_label_ratio:.1%}")
    print(f"Solar Readiness: {metrics.solar_readiness:.1%}")
    print(f"Complementarity: {metrics.complementarity_score:.1%}")
    
    return metrics

def test_mv_evaluation():
    """Test MV station evaluation"""
    print("\nTesting MV Station Evaluation")
    print("=" * 60)
    
    evaluator = EnhancedLVMVEvaluator()
    
    # Create multiple LV groups with varying characteristics
    lv_groups = []
    
    # Good diverse LV group
    lv1 = create_test_buildings(40, "LV_001")
    lv_groups.append(lv1)
    
    # Homogeneous residential group
    lv2 = [
        {
            'id': f'B_{i}',
            'lv_group_id': 'LV_002',
            'type': 'residential',
            'energy_label': 'C',
            'area': 120,
            'roof_area': 60,
            'orientation': 'S',
            'has_solar': False,
            'peak_hour': 18
        }
        for i in range(30)
    ]
    lv_groups.append(lv2)
    
    # Poor labels group needing intervention
    lv3 = create_test_buildings(25, "LV_003")
    for b in lv3:
        b['energy_label'] = np.random.choice(['E', 'F', 'G'])
    lv_groups.append(lv3)
    
    # Evaluate MV station
    mv_metrics = evaluator.evaluate_mv_station(lv_groups, "MV_TEST_001")
    
    print(f"MV Station: {mv_metrics.mv_station_id}")
    print(f"LV Groups: {mv_metrics.lv_group_count}")
    print(f"Total Buildings: {mv_metrics.total_buildings}")
    print(f"Planning Priority: {mv_metrics.planning_priority:.2f}/10")
    print(f"Avg Diversity: {mv_metrics.avg_diversity_score:.1%}")
    print(f"Avg Poor Labels: {mv_metrics.avg_poor_label_ratio:.1%}")
    print(f"Intervention Urgency: {mv_metrics.intervention_urgency:.1%}")
    print(f"Best LV Groups: {mv_metrics.best_lv_groups}")
    print(f"Problematic LV Groups: {mv_metrics.problematic_lv_groups}")
    
    return mv_metrics

def test_report_generation():
    """Test report generation"""
    print("\nTesting Report Generation")
    print("=" * 60)
    
    evaluator = EnhancedLVMVEvaluator()
    
    # Create test data
    lv_groups = []
    lv_metrics_list = []
    
    for i in range(5):
        buildings = create_test_buildings(30 + i*10, f"LV_{i:03d}")
        lv_groups.append(buildings)
        metrics = evaluator.evaluate_lv_group(buildings)
        lv_metrics_list.append(metrics)
    
    # Evaluate MV
    mv_metrics = evaluator.evaluate_mv_station(lv_groups, "MV_REPORT_TEST")
    
    # Generate report
    report_path = Path("reports/test_assessment.txt")
    report_path.parent.mkdir(exist_ok=True)
    
    report = evaluator.generate_assessment_report(
        mv_metrics,
        lv_metrics_list,
        report_path
    )
    
    print(f"Report generated: {report_path}")
    print("\nFirst 50 lines of report:")
    print("-" * 60)
    lines = report.split('\n')
    for line in lines[:50]:
        print(line)
    
    return report

def test_mv_selection():
    """Test MV station selection"""
    print("\nTesting MV Station Selection")
    print("=" * 60)
    
    evaluator = EnhancedLVMVEvaluator()
    
    # Create multiple MV stations
    mv_metrics_list = []
    
    for mv_idx in range(3):
        lv_groups = []
        lv_metrics = []
        
        # Vary characteristics for each MV
        if mv_idx == 0:
            # High diversity, moderate poor labels
            for i in range(4):
                buildings = create_test_buildings(40, f"MV{mv_idx}_LV{i}")
                lv_groups.append(buildings)
        elif mv_idx == 1:
            # Low diversity, high poor labels
            for i in range(3):
                buildings = create_test_buildings(30, f"MV{mv_idx}_LV{i}")
                for b in buildings:
                    b['type'] = 'residential'
                    if np.random.random() < 0.6:
                        b['energy_label'] = np.random.choice(['E', 'F', 'G'])
                lv_groups.append(buildings)
        else:
            # Balanced
            for i in range(5):
                buildings = create_test_buildings(25, f"MV{mv_idx}_LV{i}")
                lv_groups.append(buildings)
        
        mv_metrics = evaluator.evaluate_mv_station(lv_groups, f"MV_{mv_idx:03d}")
        if mv_metrics:
            mv_metrics_list.append(mv_metrics)
    
    # Select best
    best_mv, rationale = evaluator.select_best_mv_station(mv_metrics_list)
    
    print(f"Selected MV Station: {best_mv}")
    print(f"Rationale: {rationale}")
    print("\nAll MV Stations:")
    for mv in mv_metrics_list:
        print(f"  {mv.mv_station_id}: Priority {mv.planning_priority:.2f}/10")

if __name__ == "__main__":
    print("INITIAL ASSESSMENT TESTING")
    print("=" * 80)
    
    # Run tests
    lv_metrics = test_lv_evaluation()
    mv_metrics = test_mv_evaluation()
    report = test_report_generation()
    test_mv_selection()
    
    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)