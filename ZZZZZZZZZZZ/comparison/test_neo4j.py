import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.kg_connector import KGConnector

# Test Neo4j connection
print("Testing Neo4j connection...")

try:
    kg = KGConnector(
        uri="bolt://localhost:7687",
        user="neo4j", 
        password="12345678"
    )
    
    # Test basic query
    if kg.verify_connection():
        print("✓ Connection successful!")
        
        # Get statistics
        stats = kg.get_statistics()
        print(f"Database statistics: {stats}")
        
        # Get LV groups
        lv_groups = kg.get_all_lv_groups()
        print(f"Found {len(lv_groups)} LV groups")
        
        if lv_groups:
            print(f"Sample LV groups: {lv_groups[:5]}")
            
            # Get data for first group
            first_group = kg.get_lv_group_data(lv_groups[0])
            if first_group:
                print(f"First group has {len(first_group.get('buildings', []))} buildings")
    else:
        print("✗ Connection failed")
        
    kg.close()
    
except Exception as e:
    print(f"Error: {e}")
    print("\nPlease ensure:")
    print("1. Neo4j is running (neo4j console or neo4j start)")
    print("2. Password is correct (default may be 'neo4j' or '12345678')")
    print("3. Database is at bolt://localhost:7687")