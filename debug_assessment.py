"""
Debug script to find root causes of assessment issues
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from data.kg_connector import KGConnector
from analysis.lv_mv_evaluator import EnhancedLVMVEvaluator
import numpy as np

# Connect to KG
kg = KGConnector('neo4j://127.0.0.1:7687', 'neo4j', 'neo4jpassword')

# Get sample LV group data
query = """
MATCH (b:Building)-[:CONNECTED_TO]->(lv:CableGroup {group_id: 'LV_GROUP_0050'})
RETURN b.ogc_fid as id,
       b.construction_type as type,
       b.energy_label as energy_label,
       b.area as area,
       b.roof_area as roof_area,
       b.orientation as orientation,
       b.has_solar as has_solar,
       'LV_GROUP_0050' as lv_group_id
LIMIT 10
"""

try:
    buildings = kg.run(query)
    print(f"Retrieved {len(buildings)} buildings")
    
    # Check what data we actually have
    if buildings:
        print("\nSample building data:")
        for i, b in enumerate(buildings[:3]):
            print(f"\nBuilding {i+1}:")
            print(f"  ID: {b.get('id')}")
            print(f"  Type: {b.get('type')} (type: {type(b.get('type'))})")
            print(f"  Energy Label: {b.get('energy_label')}")
            print(f"  Area: {b.get('area')}")
            print(f"  Roof Area: {b.get('roof_area')}")
            print(f"  Orientation: {b.get('orientation')}")
            print(f"  Has Solar: {b.get('has_solar')}")
        
        # Test diversity calculation
        print("\n" + "="*60)
        print("TESTING DIVERSITY CALCULATION")
        print("="*60)
        
        # Extract types
        types = []
        for b in buildings:
            btype = b.get('type', 'residential')
            if btype is None:
                btype = 'residential'
            print(f"Building type: '{btype}'")
            types.append('residential')  # All become residential
        
        # Calculate entropy
        import pandas as pd
        type_counts = pd.Series(types).value_counts()
        print(f"\nType counts: {type_counts.to_dict()}")
        print(f"Number of unique types: {len(type_counts)}")
        
        proportions = type_counts / len(types)
        entropy = -sum(p * np.log(p + 1e-10) for p in proportions)
        print(f"Entropy: {entropy}")
        
        # The bug: when all same type, len(type_counts) = 1
        max_entropy = np.log(len(type_counts))
        print(f"Max entropy: {max_entropy} (log of {len(type_counts)})")
        
        if max_entropy == 0:
            print("BUG FOUND: max_entropy is 0 when all buildings same type!")
            print("This causes division by near-zero, creating negative diversity")
        
        diversity = entropy / (max_entropy + 1e-10)
        print(f"Diversity: {diversity}")
        
        # Test with evaluator
        print("\n" + "="*60)
        print("TESTING WITH EVALUATOR")
        print("="*60)
        
        evaluator = EnhancedLVMVEvaluator()
        metrics = evaluator.evaluate_lv_group(buildings)
        
        print(f"Function Diversity: {metrics.function_diversity}")
        print(f"Temporal Diversity: {metrics.temporal_diversity}")
        print(f"Complementarity Score: {metrics.complementarity_score}")
        print(f"Planning Score: {metrics.get_planning_score()}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()