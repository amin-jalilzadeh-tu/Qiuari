"""
Explore KG structure - nodes, edges, attributes
"""

from neo4j import GraphDatabase
import json

class KGStructureExplorer:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            "neo4j://127.0.0.1:7687",
            auth=("neo4j", "aminasad")
        )
    
    def close(self):
        self.driver.close()
    
    def explore_key_structures(self):
        with self.driver.session() as session:
            # Check Building node attributes
            print("=" * 60)
            print("BUILDING NODE ATTRIBUTES (Key ones for clustering):")
            print("=" * 60)
            result = session.run("""
                MATCH (b:Building)
                RETURN b
                LIMIT 1
            """)
            for record in result:
                building = record['b']
                important_attrs = [
                    'ogc_fid', 'x', 'y', 'lv_group_id', 'building_function',
                    'energy_label', 'solar_potential', 'solar_capacity_kwp',
                    'has_solar', 'has_battery', 'adjacency_type', 
                    'num_shared_walls', 'total_shared_length',
                    'avg_electricity_demand_kw', 'avg_heating_demand_kw',
                    'peak_electricity_demand_kw', 'area', 'suitable_roof_area'
                ]
                for attr in important_attrs:
                    if attr in building:
                        value = building[attr]
                        print(f"  {attr}: {value} ({type(value).__name__})")
            
            # Check LV Group structure
            print("\n" + "=" * 60)
            print("LV GROUP ORGANIZATION:")
            print("=" * 60)
            result = session.run("""
                MATCH (g:CableGroup)
                WHERE g.voltage_level = 'LV'
                OPTIONAL MATCH (b:Building)-[:CONNECTED_TO]->(g)
                RETURN g.group_id as lv_id, 
                       count(DISTINCT b) as building_count
                ORDER BY building_count DESC
                LIMIT 10
            """)
            for record in result:
                print(f"  {record['lv_id']}: {record['building_count']} buildings")
            
            # Check adjacency relationships
            print("\n" + "=" * 60)
            print("ADJACENCY RELATIONSHIPS:")
            print("=" * 60)
            result = session.run("""
                MATCH (b1:Building)-[r:ADJACENT_TO]->(b2:Building)
                RETURN r
                LIMIT 3
            """)
            for record in result:
                rel = record['r']
                print(f"  Adjacency relationship properties:")
                for key, value in rel.items():
                    print(f"    - {key}: {value}")
                break  # Just show one example
            
            # Check adjacency clusters
            print("\n" + "=" * 60)
            print("ADJACENCY CLUSTERS:")
            print("=" * 60)
            result = session.run("""
                MATCH (ac:AdjacencyCluster)
                OPTIONAL MATCH (b:Building)-[:IN_ADJACENCY_CLUSTER]->(ac)
                RETURN ac.cluster_id as cluster_id,
                       ac.cluster_type as type,
                       ac.pattern as pattern,
                       count(b) as building_count,
                       ac.lv_group_id as lv_group
                ORDER BY building_count DESC
                LIMIT 5
            """)
            for record in result:
                print(f"  Cluster: {record['cluster_id']}")
                print(f"    Type: {record['type']}, Pattern: {record['pattern']}")
                print(f"    Buildings: {record['building_count']}, LV: {record['lv_group']}")
            
            # Check energy states
            print("\n" + "=" * 60)
            print("ENERGY STATE EXAMPLE:")
            print("=" * 60)
            result = session.run("""
                MATCH (es:EnergyState)-[:FOR_BUILDING]->(b:Building)
                MATCH (es)-[:DURING]->(ts:TimeSlot)
                RETURN es, ts.hour_of_day as hour, ts.season as season
                LIMIT 1
            """)
            for record in result:
                state = record['es']
                print(f"  Energy State at hour {record['hour']} ({record['season']}):")
                energy_attrs = [
                    'electricity_demand_kw', 'heating_demand_kw', 
                    'solar_generation_kw', 'net_demand_kw',
                    'battery_soc_kwh', 'is_surplus'
                ]
                for attr in energy_attrs:
                    if attr in state:
                        print(f"    {attr}: {state[attr]}")
            
            # Check relationships overview
            print("\n" + "=" * 60)
            print("KEY RELATIONSHIPS FOR GNN:")
            print("=" * 60)
            result = session.run("""
                MATCH (a)-[r]->(b)
                WHERE labels(a)[0] IN ['Building', 'CableGroup', 'AdjacencyCluster']
                  AND labels(b)[0] IN ['Building', 'CableGroup', 'AdjacencyCluster']
                RETURN DISTINCT labels(a)[0] as from_node,
                       type(r) as relationship,
                       labels(b)[0] as to_node
                ORDER BY from_node, relationship
            """)
            for record in result:
                print(f"  {record['from_node']} --[{record['relationship']}]--> {record['to_node']}")

if __name__ == "__main__":
    explorer = KGStructureExplorer()
    try:
        explorer.explore_key_structures()
    finally:
        explorer.close()