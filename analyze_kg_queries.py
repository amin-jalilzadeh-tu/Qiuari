"""
Direct Cypher Query Analysis from kg_connector.py
Extracts and analyzes actual queries from the implementation
"""

import re
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Manually extract key queries from kg_connector.py for analysis
QUERIES_TO_ANALYZE = {
    "get_district_hierarchy": """
            MATCH (b:Building {district_name: $district})
            OPTIONAL MATCH (b)-[:CONNECTED_TO]->(cg:CableGroup)
            OPTIONAL MATCH (cg)-[:CONNECTS_TO]->(t:Transformer)
            OPTIONAL MATCH (t)-[:FEEDS_FROM]->(s:Substation)
            WITH s, t, cg, collect(DISTINCT b) as buildings
            WITH s, t, collect(DISTINCT {
                cable_group: cg,
                buildings: buildings
            }) as cg_data
            WITH s, collect(DISTINCT {
                transformer: t,
                cable_groups: cg_data
            }) as t_data
            RETURN s as substation, t_data as transformers
    """,
    
    "get_buildings_by_cable_group": """
            MATCH (b:Building)-[:CONNECTED_TO]->(cg:CableGroup {group_id: $group_id})
            OPTIONAL MATCH (b)-[:IN_ADJACENCY_CLUSTER]->(ac:AdjacencyCluster)
            RETURN b.ogc_fid as id,
                   b.energy_label as energy_label,
                   b.area as area,
                   b.building_function as building_function,
                   cg.group_id as lv_group,
                   ac.cluster_id as cluster_id
    """,
    
    "get_retrofit_candidates": """
            MATCH (b:Building {district_name: $district})
            WHERE b.energy_label IN $labels
            AND b.age_range CONTAINS $age_filter
            OPTIONAL MATCH (b)-[:CONNECTED_TO]->(cg:CableGroup)
            WITH cg, collect(b) as buildings
            RETURN cg.group_id as cable_group_id, 
                   size(buildings) as building_count,
                   buildings
            ORDER BY building_count DESC
    """,
    
    "get_adjacency_clusters": """
            MATCH (ac:AdjacencyCluster)<-[:IN_ADJACENCY_CLUSTER]-(b:Building)
            WHERE b.district_name = $district
            WITH ac, collect(b) as buildings
            WHERE size(buildings) >= $min_size
            WITH ac, buildings, size(buildings) as building_count,
                reduce(s = 0, b IN buildings | s + COALESCE(b.area, 0)) as total_area,
                reduce(s = 0, b IN buildings | s + CASE WHEN b.has_solar = true THEN 1 ELSE 0 END) as solar_count,
                reduce(s = 0, b IN buildings | s + CASE WHEN b.has_heat_pump = true THEN 1 ELSE 0 END) as hp_count
            RETURN 
                ac.cluster_id as cluster_id,
                ac.cluster_type as cluster_type,
                building_count,
                ac.energy_sharing_potential as sharing_potential
            ORDER BY ac.energy_sharing_potential DESC
    """,
    
    "get_grid_topology_nodes": """
            MATCH (b:Building {district_name: $district})
            WITH collect(DISTINCT b) as buildings
            OPTIONAL MATCH (cg:CableGroup)<-[:CONNECTED_TO]-(b2:Building {district_name: $district})
            WITH buildings, collect(DISTINCT cg) as cable_groups
            OPTIONAL MATCH (cg2:CableGroup)-[:CONNECTS_TO]->(t:Transformer)
            WITH buildings, cable_groups, collect(DISTINCT t) as transformers
            OPTIONAL MATCH (t2:Transformer)-[:FEEDS_FROM]->(s:Substation)
            WITH buildings, cable_groups, transformers, collect(DISTINCT s) as substations
            OPTIONAL MATCH (ac:AdjacencyCluster)<-[:IN_ADJACENCY_CLUSTER]-(b3:Building {district_name: $district})
            RETURN 
                buildings,
                cable_groups,
                transformers,
                substations,
                collect(DISTINCT ac) as clusters
    """,
    
    "get_building_time_series": """
            MATCH (b:Building)<-[:FOR_BUILDING]-(es:EnergyState)-[:DURING]->(ts:TimeSlot)
            WHERE toString(b.ogc_fid) IN $ids
            AND ts.timestamp > $start_time
            AND ts.timestamp <= $end_time
            WITH b, es, ts
            ORDER BY b.ogc_fid, ts.timestamp
            WITH toString(b.ogc_fid) as building_id,
                collect({
                    timestamp: ts.timestamp,
                    hour: COALESCE(ts.hour, 0),
                    day_of_week: COALESCE(ts.day_of_week, 0),
                    electricity: COALESCE(es.electricity_demand_kw, 0.0),
                    heating: COALESCE(es.heating_demand_kw, 0.0),
                    solar: COALESCE(es.solar_generation_kw, 0.0)
                }) as time_series
            RETURN building_id, time_series
    """,
    
    "aggregate_to_cable_group": """
            MATCH (cg:CableGroup {group_id: $group_id})<-[:CONNECTED_TO]-(b:Building)
            WITH cg, b
            RETURN 
                cg.group_id as group_id,
                cg.voltage_level as voltage_level,
                count(b) as building_count,
                sum(COALESCE(b.area, 0)) as total_area,
                sum(COALESCE(b.suitable_roof_area, 0)) as total_roof_area,
                avg(CASE b.energy_label
                    WHEN 'A' THEN 7 WHEN 'B' THEN 6 WHEN 'C' THEN 5
                    WHEN 'D' THEN 4 WHEN 'E' THEN 3 WHEN 'F' THEN 2
                    WHEN 'G' THEN 1 ELSE 0 END) as avg_energy_score,
                sum(CASE WHEN b.has_solar = true THEN 1 ELSE 0 END) as solar_count
    """,
    
    "get_statistics": """
            MATCH (b:Building)
            WITH count(b) as building_count
            MATCH (cg:CableGroup)
            WITH building_count, count(cg) as cable_group_count
            MATCH (t:Transformer)
            WITH building_count, cable_group_count, count(t) as transformer_count
            MATCH (s:Substation)
            WITH building_count, cable_group_count, transformer_count, count(s) as substation_count
            OPTIONAL MATCH (ac:AdjacencyCluster)
            WITH building_count, cable_group_count, transformer_count, substation_count, 
                 count(ac) as cluster_count
            OPTIONAL MATCH ()-[r]->()
            RETURN 
                building_count,
                cable_group_count,
                transformer_count,
                substation_count,
                cluster_count,
                count(r) as relationship_count
    """
}

def analyze_query_performance(query_name: str, query: str) -> Dict:
    """Analyze a single query for performance metrics."""
    metrics = {
        'name': query_name,
        'line_count': len(query.strip().split('\n')),
        'char_count': len(query),
        'operations': {},
        'complexity_score': 0,
        'issues': [],
        'optimizations': []
    }
    
    # Count operations
    operations = {
        'MATCH': query.count('MATCH') - query.count('OPTIONAL MATCH'),
        'OPTIONAL MATCH': query.count('OPTIONAL MATCH'),
        'WHERE': query.count('WHERE'),
        'WITH': query.count('WITH'),
        'RETURN': query.count('RETURN'),
        'ORDER BY': query.count('ORDER BY'),
        'collect': query.count('collect('),
        'sum': query.count('sum('),
        'avg': query.count('avg('),
        'count': query.count('count('),
        'reduce': query.count('reduce('),
        'CASE': query.count('CASE'),
        'COALESCE': query.count('COALESCE')
    }
    metrics['operations'] = operations
    
    # Calculate complexity score
    score = 0
    score += operations['MATCH'] * 2
    score += operations['OPTIONAL MATCH'] * 3
    score += operations['WITH'] * 1
    score += operations['WHERE'] * 1
    score += operations['collect'] * 2
    score += operations['reduce'] * 3
    score += operations['CASE'] * 2
    
    metrics['complexity_score'] = score
    
    # Identify issues
    if operations['MATCH'] > 3:
        metrics['issues'].append("Multiple MATCH clauses may cause performance issues")
    
    if operations['OPTIONAL MATCH'] > 2:
        metrics['issues'].append("Multiple OPTIONAL MATCH clauses increase complexity")
    
    if 'collect(DISTINCT' not in query and operations['collect'] > 0:
        metrics['issues'].append("Consider using collect(DISTINCT) to avoid duplicates")
    
    if operations['ORDER BY'] > 0 and 'LIMIT' not in query:
        metrics['issues'].append("ORDER BY without LIMIT can be expensive")
    
    # Check for Cartesian products
    if operations['MATCH'] > 1 and operations['WHERE'] == 0:
        metrics['issues'].append("Potential Cartesian product - no WHERE clause with multiple MATCHes")
    
    # Suggest optimizations
    if score > 15:
        metrics['optimizations'].append("Consider breaking this complex query into smaller parts")
    
    if operations['OPTIONAL MATCH'] > 0 and operations['COALESCE'] == 0:
        metrics['optimizations'].append("Add COALESCE for null handling with OPTIONAL MATCH")
    
    if '$' not in query:
        metrics['optimizations'].append("Use parameterized queries for better performance and security")
    
    return metrics

def main():
    """Run comprehensive query analysis."""
    print("="*80)
    print("NEO4J CYPHER QUERY PERFORMANCE ANALYSIS")
    print("="*80)
    
    all_metrics = []
    total_issues = 0
    
    for query_name, query in QUERIES_TO_ANALYZE.items():
        metrics = analyze_query_performance(query_name, query)
        all_metrics.append(metrics)
        total_issues += len(metrics['issues'])
        
    # Sort by complexity
    all_metrics.sort(key=lambda x: x['complexity_score'], reverse=True)
    
    print(f"\nAnalyzed {len(all_metrics)} key queries")
    print(f"Total issues found: {total_issues}")
    
    print("\n## QUERY COMPLEXITY RANKING")
    print("-"*40)
    for i, m in enumerate(all_metrics, 1):
        complexity = "Simple" if m['complexity_score'] < 10 else \
                    "Moderate" if m['complexity_score'] < 20 else \
                    "Complex" if m['complexity_score'] < 30 else "Very Complex"
        print(f"{i}. {m['name']}: {complexity} (score: {m['complexity_score']})")
    
    print("\n## DETAILED ANALYSIS")
    print("-"*40)
    
    for metrics in all_metrics:
        print(f"\n### {metrics['name']}")
        print(f"Complexity Score: {metrics['complexity_score']}")
        print(f"Size: {metrics['line_count']} lines, {metrics['char_count']} chars")
        
        # Show key operations
        key_ops = [(k, v) for k, v in metrics['operations'].items() if v > 0]
        if key_ops:
            print("Operations:", ", ".join([f"{k}({v})" for k, v in key_ops]))
        
        if metrics['issues']:
            print("Issues:")
            for issue in metrics['issues']:
                print(f"  - {issue}")
        
        if metrics['optimizations']:
            print("Optimizations:")
            for opt in metrics['optimizations']:
                print(f"  - {opt}")
    
    print("\n## PERFORMANCE RECOMMENDATIONS")
    print("-"*40)
    
    print("1. INDEX STRATEGY:")
    print("   Create indexes on:")
    print("   - Building(district_name)")
    print("   - Building(ogc_fid)")
    print("   - CableGroup(group_id)")
    print("   - Transformer(transformer_id)")
    print("   - TimeSlot(timestamp)")
    
    print("\n2. QUERY OPTIMIZATION PRIORITIES:")
    for m in all_metrics[:3]:  # Top 3 complex queries
        if m['issues']:
            print(f"   - {m['name']}: Address {len(m['issues'])} issues")
    
    print("\n3. CACHING CANDIDATES:")
    print("   - get_statistics: Cache for 5 minutes")
    print("   - get_district_hierarchy: Cache per district")
    print("   - get_grid_topology_nodes: Cache with district key")
    
    print("\n## ESTIMATED PERFORMANCE")
    print("-"*40)
    print("With proper indexing and ~10K nodes:")
    
    for m in all_metrics:
        if m['complexity_score'] < 10:
            est = "< 20ms"
        elif m['complexity_score'] < 20:
            est = "20-100ms"
        elif m['complexity_score'] < 30:
            est = "100-300ms"
        else:
            est = "> 300ms"
        print(f"  {m['name']}: {est}")
    
    print("\n" + "="*80)
    print("RECOMMENDATION SUMMARY")
    print("="*80)
    
    critical_queries = [m for m in all_metrics if len(m['issues']) > 2]
    if critical_queries:
        print(f"\nCRITICAL: {len(critical_queries)} queries need immediate optimization")
        for q in critical_queries:
            print(f"  - {q['name']}")
    
    moderate_queries = [m for m in all_metrics if 1 <= len(m['issues']) <= 2]
    if moderate_queries:
        print(f"\nMODERATE: {len(moderate_queries)} queries have minor issues")
    
    clean_queries = [m for m in all_metrics if len(m['issues']) == 0]
    if clean_queries:
        print(f"\nGOOD: {len(clean_queries)} queries are well-optimized")
    
    print("\nOVERALL ASSESSMENT: ", end="")
    if total_issues < 5:
        print("EXCELLENT - Queries are well-optimized")
    elif total_issues < 15:
        print("GOOD - Minor optimizations recommended")
    elif total_issues < 25:
        print("FAIR - Several optimizations needed")
    else:
        print("NEEDS WORK - Significant optimization required")
    
    print("="*80)

if __name__ == "__main__":
    main()