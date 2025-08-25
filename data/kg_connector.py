# kg_connector.py - UPDATED FOR YOUR SCHEMA
"""
Direct Neo4j connector for your energy grid knowledge graph.
Updated to match your actual schema: Building → CableGroup → Transformer → Substation
"""


from neo4j import GraphDatabase
import pandas as pd
import numpy as np  # ADD THIS LINE
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class KGConnector:
    """Direct Neo4j connection for your grid and building data."""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialize Neo4j connection.
        
        Args:
            uri: Neo4j connection URI (e.g., "bolt://localhost:7687")
            user: Neo4j username
            password: Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connected to Neo4j at {uri}")
        
    def verify_connection(self) -> bool:
        """Test if connection is active."""
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                return result.single()["test"] == 1
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    def get_district_hierarchy(self, district_name: str) -> Dict[str, Any]:
        """
        Get complete grid hierarchy for a district.
        Your schema: Building → CableGroup → Transformer → Substation
        
        Args:
            district_name: District name (e.g., "Buitenveldert-Oost")
            
        Returns:
            Hierarchical structure with grid components
        """
        with self.driver.session() as session:
            query = """
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
            """
            
            result = session.run(query, district=district_name).data()
            
            if not result:
                logger.warning(f"No data found for district {district_name}")
                return {}
                
            return self._format_hierarchy(result[0])
    
    def get_buildings_by_cable_group(self, group_id: str) -> List[Dict]:
        """
        Get all buildings connected to a specific cable group.
        Replaces the old get_buildings_by_lv() method.
        
        Args:
            group_id: Cable group identifier
            
        Returns:
            List of building dictionaries with attributes
        """
        with self.driver.session() as session:
            query = """
            MATCH (b:Building)-[:CONNECTED_TO]->(cg:CableGroup {group_id: $group_id})
            OPTIONAL MATCH (b)-[:IN_ADJACENCY_CLUSTER]->(ac:AdjacencyCluster)
            RETURN b.ogc_fid as id,
                   b.energy_label as energy_label,
                   b.area as area,
                   b.building_function as function,
                   b.age_range as age_range,
                   b.building_year as year,
                   b.suitable_roof_area as roof_area,
                   b.height as height,
                   b.solar_potential as solar_potential,
                   b.electrification_feasibility as electrification,
                   b.has_solar as has_solar,
                   b.has_battery as has_battery,
                   b.has_heat_pump as has_heat_pump,
                   ac.cluster_id as cluster_id
            """
            
            result = session.run(query, group_id=group_id).data()
            return result
    
    def get_retrofit_candidates(self, district_name: str, 
                              energy_labels: List[str] = ['E', 'F', 'G'],
                              age_filter: str = '19') -> Dict[str, List]:
        """
        Get buildings that are candidates for retrofitting.
        
        Args:
            district_name: District to search
            energy_labels: Poor energy labels to filter
            age_filter: Age range filter (buildings containing this string)
            
        Returns:
            Buildings grouped by cable group that need retrofitting
        """
        with self.driver.session() as session:
            query = """
            MATCH (b:Building {district_name: $district})
            WHERE b.energy_label IN $labels
            AND b.age_range CONTAINS $age_filter
            OPTIONAL MATCH (b)-[:CONNECTED_TO]->(cg:CableGroup)
            WITH cg, collect(b) as buildings
            RETURN cg.group_id as cable_group_id, 
                   size(buildings) as building_count,
                   buildings
            ORDER BY building_count DESC
            """
            
            result = session.run(
                query, 
                district=district_name,
                labels=energy_labels,
                age_filter=age_filter
            ).data()
            
            return {row['cable_group_id']: row['buildings'] for row in result if row['cable_group_id']}
    
    def get_adjacency_clusters(self, district_name: str, 
                            min_cluster_size: int = 3) -> List[Dict]:
        """
        Get adjacency clusters for energy sharing analysis.
        
        Args:
            district_name: District to analyze
            min_cluster_size: Minimum buildings in cluster
            
        Returns:
            Adjacency clusters with aggregated metrics
        """
        with self.driver.session() as session:
            # FIXED QUERY - using reduce for aggregation
            query = """
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
                ac.energy_sharing_potential as sharing_potential,
                ac.solar_penetration as solar_penetration,
                ac.hp_penetration as hp_penetration,
                ac.battery_penetration as battery_penetration,
                CASE WHEN building_count > 0 THEN total_area / building_count ELSE 0 END as avg_area,
                solar_count,
                hp_count,
                buildings
            ORDER BY ac.energy_sharing_potential DESC
            """
            
            result = session.run(
                query, 
                district=district_name,
                min_size=min_cluster_size
            ).data()
            return result
    


    # In kg_connector.py - Replace the get_grid_topology method

    def get_grid_topology(self, district_name: str) -> Dict[str, List]:
        """
        Get grid topology as nodes and edges for GNN.
        Fixed to properly capture CONNECTS_TO relationships.
        """
        with self.driver.session() as session:
            # Get all nodes
            nodes_query = """
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
            """
            
            nodes = session.run(nodes_query, district=district_name).single()
            
            # Get all edges - FIXED to use CONNECTS_TO
            edges_query = """
            MATCH (b:Building {district_name: $district})
            OPTIONAL MATCH (b)-[:CONNECTED_TO]->(cg:CableGroup)
            OPTIONAL MATCH (cg)-[:CONNECTS_TO]->(t:Transformer)
            OPTIONAL MATCH (t)-[:FEEDS_FROM]->(s:Substation)
            OPTIONAL MATCH (b)-[:IN_ADJACENCY_CLUSTER]->(ac:AdjacencyCluster)
            WITH 
                collect(DISTINCT CASE WHEN cg IS NOT NULL 
                    THEN {src: toString(b.ogc_fid), dst: cg.group_id} END) as b_to_cg,
                collect(DISTINCT CASE WHEN t IS NOT NULL AND cg IS NOT NULL
                    THEN {src: cg.group_id, dst: toString(t.transformer_id)} END) as cg_to_t,
                collect(DISTINCT CASE WHEN s IS NOT NULL AND t IS NOT NULL
                    THEN {src: toString(t.transformer_id), dst: s.station_id} END) as t_to_s,
                collect(DISTINCT CASE WHEN ac IS NOT NULL 
                    THEN {src: toString(b.ogc_fid), dst: toString(ac.cluster_id)} END) as b_to_ac
            RETURN 
                [e IN b_to_cg WHERE e IS NOT NULL] as building_to_cable,
                [e IN cg_to_t WHERE e IS NOT NULL] as cable_to_transformer,
                [e IN t_to_s WHERE e IS NOT NULL] as transformer_to_substation,
                [e IN b_to_ac WHERE e IS NOT NULL] as building_to_cluster
            """
            
            edges = session.run(edges_query, district=district_name).single()
            
            # Log edge counts for debugging
            logger.info(f"Edge counts - B->CG: {len(edges['building_to_cable'])}, "
                    f"CG->T: {len(edges['cable_to_transformer'])}, "
                    f"T->S: {len(edges['transformer_to_substation'])}, "
                    f"B->AC: {len(edges['building_to_cluster'])}")
            
            return {
                'nodes': {
                    'buildings': nodes['buildings'] or [],
                    'cable_groups': nodes['cable_groups'] or [],
                    'transformers': nodes['transformers'] or [],
                    'substations': nodes['substations'] or [],
                    'adjacency_clusters': nodes['clusters'] or []
                },
                'edges': {
                    'building_to_cable': edges['building_to_cable'] or [],
                    'cable_to_transformer': edges['cable_to_transformer'] or [],
                    'transformer_to_substation': edges['transformer_to_substation'] or [],
                    'building_to_cluster': edges['building_to_cluster'] or []
                }
            }
    
    def get_energy_states(self, building_ids: List[str], 
                         time_range: Optional[Dict] = None) -> pd.DataFrame:
        """
        Get energy time series data for buildings.
        New method for your EnergyState nodes.
        
        Args:
            building_ids: List of building OGC FIDs
            time_range: Optional time range filter
            
        Returns:
            DataFrame with energy states
        """
        with self.driver.session() as session:
            query = """
            MATCH (es:EnergyState)-[:DURING]->(ts:TimeSlot)
            WHERE toString(es.building_id) IN $ids
            RETURN es.building_id as building_id,
                   ts.timeslot_id as timeslot,
                   es.electricity_demand_kw as electricity_demand,
                   es.heating_demand_kw as heating_demand,
                   es.cooling_demand_kw as cooling_demand,
                   es.solar_generation_kw as solar_generation,
                   es.battery_soc_kwh as battery_soc,
                   es.net_demand_kw as net_demand,
                   es.export_potential_kw as export_potential
            ORDER BY es.building_id, ts.timeslot_id
            """
            
            # Convert building IDs to strings
            str_ids = [str(bid) for bid in building_ids]
            result = session.run(query, ids=str_ids).data()
            return pd.DataFrame(result) if result else pd.DataFrame()
    
    def aggregate_to_cable_group(self, group_id: str) -> Dict[str, Any]:
        """
        Aggregate building metrics at cable group level - FIXED VERSION.
        
        Args:
            group_id: Cable group ID
            
        Returns:
            Aggregated metrics for the cable group
        """
        with self.driver.session() as session:
            # First check if this cable group exists and get actual building count
            check_query = """
            MATCH (cg:CableGroup {group_id: $group_id})
            OPTIONAL MATCH (cg)<-[:CONNECTED_TO]-(b:Building)
            RETURN count(b) as actual_count
            """
            
            check_result = session.run(check_query, group_id=group_id).single()
            
            if not check_result or check_result['actual_count'] == 0:
                # No buildings connected to this cable group
                return {
                    'group_id': group_id,
                    'building_count': 0,
                    'total_area': 0,
                    'avg_energy_score': 0
                }
            
            # Now get the aggregated data
            query = """
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
                sum(CASE WHEN b.has_solar = true THEN 1 ELSE 0 END) as solar_count,
                sum(CASE WHEN b.has_battery = true THEN 1 ELSE 0 END) as battery_count,
                sum(CASE WHEN b.has_heat_pump = true THEN 1 ELSE 0 END) as hp_count
            """
            
            result = session.run(query, group_id=group_id).single()
            return dict(result) if result else {'group_id': group_id, 'building_count': 0}

    
    def get_systems_installed(self, district_name: str) -> Dict[str, Any]:
        """
        Get information about installed systems (solar, battery, heat pump).
        New method for your system nodes.
        
        Args:
            district_name: District name
            
        Returns:
            Dictionary with system statistics
        """
        with self.driver.session() as session:
            query = """
            MATCH (b:Building {district_name: $district})
            OPTIONAL MATCH (b)-[:HAS_INSTALLED]->(ss:SolarSystem)
            OPTIONAL MATCH (b)-[:HAS_INSTALLED]->(bs:BatterySystem)
            OPTIONAL MATCH (b)-[:HAS_INSTALLED]->(hp:HeatPumpSystem)
            RETURN 
                count(DISTINCT b) as total_buildings,
                count(DISTINCT ss) as solar_systems,
                count(DISTINCT bs) as battery_systems,
                count(DISTINCT hp) as heat_pump_systems,
                avg(ss.capacity_kwp) as avg_solar_capacity,
                avg(bs.installed_capacity_kwh) as avg_battery_capacity,
                avg(hp.heating_capacity_kw) as avg_hp_capacity
            """
            
            result = session.run(query, district=district_name).single()
            return dict(result) if result else {}
    
    
    def _format_hierarchy(self, raw_data: Dict) -> Dict:
        """Format raw hierarchy data into structured format."""
        formatted = {
            'substation': dict(raw_data['substation']) if raw_data['substation'] else {},
            'transformers': []
        }
        
        for t_item in raw_data['transformers']:
            if t_item['transformer']:
                t_dict = {
                    'transformer': dict(t_item['transformer']),
                    'cable_groups': []
                }
                
                for cg_item in t_item['cable_groups']:
                    if cg_item['cable_group']:
                        cg_dict = {
                            'cable_group': dict(cg_item['cable_group']),
                            'buildings': [dict(b) for b in cg_item['buildings'] if b]
                        }
                        t_dict['cable_groups'].append(cg_dict)
                
                formatted['transformers'].append(t_dict)
        
        return formatted

    # THESE METHODS SHOULD BE INDENTED TO BE PART OF THE CLASS:
    def get_building_time_series(self, building_ids: List[str], 
                                lookback_hours: int = 24,
                                end_time: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Get time series data for buildings.
        
        Args:
            building_ids: List of building IDs
            lookback_hours: Hours of history to fetch
            end_time: End timestamp (default: latest available)
        
        Returns:
            Dict mapping building_id -> temporal features array [hours, features]
        """
        # Check for empty input
        if not building_ids:
            logger.warning("No building IDs provided")
            return {}
        
        # Ensure all IDs are strings
        building_ids_str = [str(bid) for bid in building_ids]
        
        with self.driver.session() as session:
            # If no end_time specified, get the latest
            if end_time is None:
                latest_result = session.run("""
                    MATCH (ts:TimeSlot)
                    RETURN max(ts.timestamp) as latest
                """).single()
                
                if latest_result and latest_result['latest']:
                    end_time = latest_result['latest']
                else:
                    logger.warning("No time series data found in database")
                    return {}
            
            # Calculate start time
            start_time = end_time - (lookback_hours * 3600000)  # Convert hours to ms
            
            logger.info(f"Fetching time series for {len(building_ids_str)} buildings")
            logger.info(f"Time range: {start_time} to {end_time} ({lookback_hours} hours)")
            
            # Fetch time series - FIXED with proper null handling
            query = """
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
                    is_weekend: COALESCE(ts.is_weekend, false),
                    electricity: COALESCE(es.electricity_demand_kw, 0.0),
                    heating: COALESCE(es.heating_demand_kw, 0.0),
                    cooling: COALESCE(es.cooling_demand_kw, 0.0),
                    solar: COALESCE(es.solar_generation_kw, 0.0),
                    battery_soc: COALESCE(es.battery_soc_kwh, 0.0),
                    net_demand: COALESCE(es.net_demand_kw, 0.0),
                    export_potential: COALESCE(es.export_potential_kw, 0.0)
                }) as time_series
            RETURN building_id, time_series
            """
            
            try:
                result = session.run(
                    query, 
                    ids=building_ids_str,
                    start_time=start_time,
                    end_time=end_time
                ).data()
            except Exception as e:
                logger.error(f"Error fetching time series: {e}")
                return {}
            
            # Convert to numpy arrays
            time_series_dict = {}
            
            for row in result:
                bid = row['building_id']
                ts_data = row['time_series']
                
                if not ts_data:
                    logger.debug(f"No time series data for building {bid}")
                    continue
                
                # Check if we have enough data points
                if len(ts_data) < lookback_hours:
                    logger.warning(f"Building {bid} has only {len(ts_data)} hours of data (expected {lookback_hours})")
                    
                    # Pad with zeros if needed
                    if len(ts_data) > 0:
                        # Create partial array with available data
                        available_hours = min(len(ts_data), lookback_hours)
                        features = np.zeros((lookback_hours, 8))
                        
                        for i in range(available_hours):
                            t = ts_data[i]
                            features[i] = [
                                float(t.get('hour', 0)) / 24.0,  # Normalize hour
                                float(t.get('day_of_week', 0)) / 7.0,  # Normalize day
                                1.0 if t.get('is_weekend', False) else 0.0,
                                float(t.get('electricity', 0)),
                                float(t.get('heating', 0)),
                                float(t.get('solar', 0)),
                                float(t.get('net_demand', 0)),
                                float(t.get('export_potential', 0))
                            ]
                        
                        time_series_dict[bid] = features
                        logger.debug(f"Padded time series for building {bid}")
                else:
                    # We have enough data, take exactly lookback_hours
                    features = np.zeros((lookback_hours, 8))
                    
                    for i in range(lookback_hours):
                        t = ts_data[i]
                        features[i] = [
                            float(t.get('hour', 0)) / 24.0,  # Normalize hour
                            float(t.get('day_of_week', 0)) / 7.0,  # Normalize day
                            1.0 if t.get('is_weekend', False) else 0.0,
                            float(t.get('electricity', 0)),
                            float(t.get('heating', 0)),
                            float(t.get('solar', 0)),
                            float(t.get('net_demand', 0)),
                            float(t.get('export_potential', 0))
                        ]
                    
                    time_series_dict[bid] = features
            
            # Log summary
            logger.info(f"Retrieved time series for {len(time_series_dict)} buildings out of {len(building_ids_str)} requested")
            
            # For buildings without data, create zero arrays
            for bid in building_ids_str:
                if bid not in time_series_dict:
                    logger.debug(f"Creating zero array for building {bid} (no data found)")
                    time_series_dict[bid] = np.zeros((lookback_hours, 8))
            
            return time_series_dict

    def get_cluster_time_series(self, cluster_id: str, 
                                lookback_hours: int = 24) -> Dict[str, Any]:
        """
        Get aggregated time series for an adjacency cluster.
        Important for energy sharing analysis.
        """
        with self.driver.session() as session:
            query = """
            MATCH (ac:AdjacencyCluster {cluster_id: $cluster_id})
            MATCH (ac)<-[:IN_ADJACENCY_CLUSTER]-(b:Building)
            MATCH (b)<-[:FOR_BUILDING]-(es:EnergyState)-[:DURING]->(ts:TimeSlot)
            WHERE ts.timestamp > timestamp() - $lookback_ms
            WITH ts, 
                sum(es.electricity_demand_kw) as total_demand,
                sum(es.solar_generation_kw) as total_solar,
                sum(es.export_potential_kw) as total_export,
                sum(CASE WHEN es.net_demand_kw > 0 THEN es.net_demand_kw ELSE 0 END) as total_deficit,
                sum(CASE WHEN es.net_demand_kw < 0 THEN -es.net_demand_kw ELSE 0 END) as total_surplus
            ORDER BY ts.timestamp
            RETURN collect({
                hour: ts.hour,
                total_demand: total_demand,
                total_solar: total_solar,
                total_export: total_export,
                total_deficit: total_deficit,
                total_surplus: total_surplus,
                sharing_potential: total_surplus - total_deficit
            }) as cluster_series
            """
            
            result = session.run(
                query,
                cluster_id=cluster_id,
                lookback_ms=lookback_hours * 3600000
            ).single()
            
            return result['cluster_series'] if result else []

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about the knowledge graph.
        
        Returns:
            Dictionary with node and relationship counts
        """
        with self.driver.session() as session:
            query = """
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
            
            result = session.run(query).single()
            
            if result:
                return {
                    'nodes': {
                        'buildings': result['building_count'],
                        'cable_groups': result['cable_group_count'],
                        'transformers': result['transformer_count'],
                        'substations': result['substation_count'],
                        'clusters': result['cluster_count'],
                        'total': (result['building_count'] + result['cable_group_count'] + 
                                 result['transformer_count'] + result['substation_count'] + 
                                 result['cluster_count'])
                    },
                    'relationships': result['relationship_count']
                }
            else:
                return {
                    'nodes': {
                        'buildings': 0,
                        'cable_groups': 0,
                        'transformers': 0,
                        'substations': 0,
                        'clusters': 0,
                        'total': 0
                    },
                    'relationships': 0
                }

    def get_all_lv_groups(self) -> List[str]:
        """
        Get all LV cable group IDs (filtering for LV_GROUP prefix).
        
        Returns:
            List of LV cable group IDs
        """
        with self.driver.session() as session:
            query = """
            MATCH (cg:CableGroup)
            WHERE cg.group_id STARTS WITH 'LV_GROUP'
            RETURN DISTINCT cg.group_id as group_id
            ORDER BY group_id
            """
            
            result = session.run(query).data()
            return [r['group_id'] for r in result if r['group_id']]
    
    def get_lv_group_data(self, lv_group_id: str) -> Dict[str, Any]:
        """
        Get complete data for a cable group (LV group equivalent).
        
        Args:
            lv_group_id: Cable group ID
            
        Returns:
            Dictionary with buildings, edges, transformers, and energy profiles
        """
        # Get buildings in this cable group
        buildings = self.get_buildings_by_cable_group(lv_group_id)
        
        # Get transformer info
        with self.driver.session() as session:
            transformer_query = """
            MATCH (cg:CableGroup {group_id: $group_id})-[:CONNECTS_TO]->(t:Transformer)
            RETURN t.id as id, t.capacity as capacity
            """
            transformer_result = session.run(transformer_query, group_id=lv_group_id).data()
            
        # Create edge data (simplified for now)
        edges = []
        if buildings:
            # Create edges between buildings (simplified neighbor connections)
            for i, b1 in enumerate(buildings):
                for b2 in buildings[i+1:i+2]:  # Connect to next building only
                    edges.append({
                        'source': b1['id'],
                        'target': b2['id'],
                        'distance': 50.0,  # Default distance
                        'cable_capacity': 100.0  # Default capacity
                    })
        
        return {
            'buildings': buildings,
            'edges': edges,
            'transformers': transformer_result,
            'energy_profiles': {}  # Placeholder for energy profiles
        }
    
    def get_lv_groups_in_district(self, district_id: str) -> List[str]:
        """
        Get all cable group IDs in a specific district.
        
        Args:
            district_id: District identifier
            
        Returns:
            List of cable group IDs
        """
        with self.driver.session() as session:
            query = """
            MATCH (b:Building {district_name: $district})-[:CONNECTED_TO]->(cg:CableGroup)
            RETURN DISTINCT cg.group_id as group_id
            ORDER BY group_id
            """
            
            result = session.run(query, district=district_id).data()
            return [r['group_id'] for r in result if r['group_id']]
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()
        logger.info("Neo4j connection closed")