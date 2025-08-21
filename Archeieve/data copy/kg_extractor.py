# data/kg_extractor.py
"""
Extract graph data from Neo4j Knowledge Graph
Handles multiple node types and relationships
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import yaml
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KGExtractor:
    """Extract and process data from Neo4j Knowledge Graph"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize KG extractor with configuration"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Neo4j connection
        self.driver = GraphDatabase.driver(
            self.config['neo4j']['uri'],
            auth=(self.config['neo4j']['user'], 
                  self.config['neo4j']['password'])
        )
        
        # Data storage
        self.nodes = {}
        self.edges = {}
        self.features = {}
        self.temporal_data = {}
        
        logger.info(f"Connected to Neo4j at {self.config['neo4j']['uri']}")
    
    def close(self):
        """Close database connection"""
        self.driver.close()
        logger.info("Neo4j connection closed")
    
    def extract_all(self) -> Dict:
        """Extract complete graph from Neo4j"""
        logger.info("Starting complete graph extraction...")
        
        # Extract nodes
        self.extract_buildings()
        self.extract_transformers()
        self.extract_infrastructure()
        
        # Extract relationships
        self.extract_electrical_connections()
        self.extract_spatial_relationships()
        self.extract_complementarity()
        
        # Extract temporal data
        self.extract_energy_profiles()
        self.extract_baseline_metrics()
        
        # Compile results
        graph_data = {
            'nodes': self.nodes,
            'edges': self.edges,
            'features': self.features,
            'temporal': self.temporal_data,
            'metadata': self.extract_metadata()
        }
        
        logger.info(f"Extraction complete: {len(self.nodes['buildings'])} buildings, "
                   f"{sum(len(e) for e in self.edges.values())} edges")
        
        return graph_data
    
    def extract_buildings(self):
        """Extract building nodes with all properties"""
        with self.driver.session() as session:
            query = """
                MATCH (b:Building)
                RETURN b.ogc_fid as id,
                       b.x as x, b.y as y,
                       b.building_function as function,
                       b.residential_type as res_type,
                       b.non_residential_type as non_res_type,
                       b.area as area,
                       b.height as height,
                       b.age_range as age,
                       b.building_orientation_cardinal as orientation,
                       b.roof_area as roof_area,
                       b.suitable_roof_area as suitable_roof,
                       b.energy_label as energy_label,
                       b.insulation_quality as insulation,
                       b.solar_potential as solar_pot,
                       b.battery_readiness as battery_ready,
                       b.electrification_feasibility as heat_pump_ready,
                       b.has_solar as has_solar,
                       b.has_battery as has_battery,
                       b.has_heat_pump as has_heat_pump,
                       b.heating_system as heating,
                       b.solar_capacity_kwp as solar_kw,
                       b.peak_demand_kw as peak_demand,
                       b.avg_demand_kw as avg_demand,
                       b.load_factor as load_factor,
                       b.demand_variability as variability,
                       b.lv_component_id as lv_network,
                       b.num_shared_walls as shared_walls,
                       b.adjacency_type as adjacency_type,
                       b.total_shared_length as shared_length
                ORDER BY b.ogc_fid
            """
            
            result = session.run(query)
            buildings = []
            
            for record in result:
                building = dict(record)
                # Handle None values
                for key, value in building.items():
                    if value is None:
                        if key in ['area', 'height', 'roof_area']:
                            building[key] = 0.0
                        elif key in ['has_solar', 'has_battery', 'has_heat_pump']:
                            building[key] = False
                        elif key in ['solar_kw', 'peak_demand', 'avg_demand']:
                            building[key] = 0.0
                
                buildings.append(building)
            
            self.nodes['buildings'] = pd.DataFrame(buildings)
            logger.info(f"Extracted {len(buildings)} buildings")
    
    def extract_transformers(self):
        """Extract transformer and LV network nodes"""
        with self.driver.session() as session:
            # MV Transformers
            query_mv = """
                MATCH (t:MV_Transformer)
                RETURN t.id as id,
                       t.capacity_kva as capacity,
                       t.voltage_level as voltage,
                       t.x as x, t.y as y,
                       t.substation_id as substation
                ORDER BY t.id
            """
            
            result_mv = session.run(query_mv)
            mv_transformers = [dict(record) for record in result_mv]
            
            # LV Networks
            query_lv = """
                MATCH (l:LV_Network)
                RETURN l.network_id as id,
                       l.component_id as component_id,
                       l.capacity_kva as capacity,
                       l.mv_transformer_id as mv_transformer,
                       l.voltage_level as voltage,
                       l.baseline_building_count as building_count,
                       l.baseline_peak_kw as peak_kw,
                       l.baseline_avg_kw as avg_kw,
                       l.baseline_load_factor as load_factor,
                       l.baseline_solar_penetration as solar_pen,
                       l.baseline_battery_penetration as battery_pen
                ORDER BY l.network_id
            """
            
            result_lv = session.run(query_lv)
            lv_networks = [dict(record) for record in result_lv]
            
            self.nodes['mv_transformers'] = pd.DataFrame(mv_transformers)
            self.nodes['lv_networks'] = pd.DataFrame(lv_networks)
            
            logger.info(f"Extracted {len(mv_transformers)} MV transformers, "
                       f"{len(lv_networks)} LV networks")
    
    def extract_infrastructure(self):
        """Extract substation and other infrastructure"""
        with self.driver.session() as session:
            query = """
                MATCH (s:Substation)
                RETURN s.id as id,
                       s.capacity_mva as capacity,
                       s.voltage_level as voltage,
                       s.x as x, s.y as y
            """
            
            result = session.run(query)
            substations = [dict(record) for record in result]
            
            self.nodes['substations'] = pd.DataFrame(substations)
            logger.info(f"Extracted {len(substations)} substations")
    
    def extract_electrical_connections(self):
        """Extract electrical connectivity relationships"""
        with self.driver.session() as session:
            # Building to LV Network connections
            query_b2lv = """
                MATCH (b:Building)-[r:CONNECTED_TO]->(l:LV_Network)
                RETURN b.ogc_fid as source,
                       l.network_id as target,
                       'building_to_lv' as type,
                       r.distance as distance
            """
            
            # LV to MV connections
            query_lv2mv = """
                MATCH (l:LV_Network)-[r:CONNECTED_TO]->(m:MV_Transformer)
                RETURN l.network_id as source,
                       m.id as target,
                       'lv_to_mv' as type,
                       r.capacity_kva as capacity
            """
            
            # MV to Substation connections
            query_mv2s = """
                MATCH (m:MV_Transformer)-[r:CONNECTED_TO]->(s:Substation)
                RETURN m.id as source,
                       s.id as target,
                       'mv_to_substation' as type,
                       r.capacity_mva as capacity
            """
            
            edges_elec = []
            for query in [query_b2lv, query_lv2mv, query_mv2s]:
                result = session.run(query)
                edges_elec.extend([dict(record) for record in result])
            
            self.edges['electrical'] = pd.DataFrame(edges_elec)
            logger.info(f"Extracted {len(edges_elec)} electrical connections")
    
    def extract_spatial_relationships(self):
        """Extract spatial proximity and adjacency relationships"""
        with self.driver.session() as session:
            # Adjacent buildings (shared walls)
            query_adj = """
                MATCH (b1:Building)-[r:ADJACENT_TO]->(b2:Building)
                WHERE b1.ogc_fid < b2.ogc_fid  // Avoid duplicates
                RETURN b1.ogc_fid as source,
                       b2.ogc_fid as target,
                       r.shared_wall_length as shared_wall,
                       r.thermal_coupling as thermal_coupling,
                       r.distance as distance
            """
            
            result = session.run(query_adj)
            adjacencies = [dict(record) for record in result]
            
            # Buildings on same transformer (implicit spatial proximity)
            query_same_trans = """
                MATCH (b1:Building)-[:CONNECTED_TO]->(l:LV_Network)
                      <-[:CONNECTED_TO]-(b2:Building)
                WHERE b1.ogc_fid < b2.ogc_fid
                RETURN b1.ogc_fid as source,
                       b2.ogc_fid as target,
                       l.network_id as shared_lv,
                       'same_transformer' as relationship
            """
            
            result = session.run(query_same_trans)
            same_transformer = [dict(record) for record in result]
            
            self.edges['spatial'] = pd.DataFrame(adjacencies)
            self.edges['same_transformer'] = pd.DataFrame(same_transformer)
            
            logger.info(f"Extracted {len(adjacencies)} adjacencies, "
                       f"{len(same_transformer)} same-transformer pairs")
    
    def extract_complementarity(self):
        """Extract discovered complementarity relationships if they exist"""
        with self.driver.session() as session:
            query = """
                MATCH (b1:Building)-[r:COMPLEMENTS]->(b2:Building)
                RETURN b1.ogc_fid as source,
                       b2.ogc_fid as target,
                       r.score as complementarity_score,
                       r.correlation as correlation,
                       r.discovered_by as method,
                       r.timestamp as discovered_at
            """
            
            result = session.run(query)
            complements = [dict(record) for record in result]
            
            if complements:
                self.edges['complementarity'] = pd.DataFrame(complements)
                logger.info(f"Found {len(complements)} existing complementarity relationships")
            else:
                logger.info("No complementarity relationships found (will be discovered by GNN)")
    
    def extract_energy_profiles(self):
        """Extract temporal energy data"""
        with self.driver.session() as session:
            # Get sample of energy states for pattern analysis
            query = """
                MATCH (b:Building)-[r:HAS_STATE]->(e:EnergyState)
                WHERE e.timestamp >= datetime('2024-01-01T00:00:00')
                  AND e.timestamp <= datetime('2024-01-07T23:59:59')
                RETURN b.ogc_fid as building_id,
                       e.timestamp as timestamp,
                       e.consumption_kw as consumption,
                       e.generation_kw as generation,
                       e.net_load_kw as net_load,
                       e.hour_of_day as hour,
                       e.day_of_week as dow,
                       e.is_weekend as weekend
                ORDER BY b.ogc_fid, e.timestamp
                LIMIT 100000
            """
            
            result = session.run(query)
            energy_data = []
            
            for record in result:
                energy_data.append(dict(record))
            
            if energy_data:
                df_energy = pd.DataFrame(energy_data)
                
                # Pivot to create time series matrix
                self.temporal_data['energy_profiles'] = df_energy.pivot_table(
                    index='timestamp',
                    columns='building_id',
                    values='consumption',
                    fill_value=0
                )
                
                logger.info(f"Extracted {len(energy_data)} energy state records")
            else:
                logger.warning("No energy state data found - will use synthetic profiles")
                self.create_synthetic_profiles()
    
    def extract_baseline_metrics(self):
        """Extract pre-computed baseline metrics"""
        with self.driver.session() as session:
            query = """
                MATCH (l:LV_Network)
                OPTIONAL MATCH (l)<-[:AGGREGATED_TO]-(m:Metrics)
                RETURN l.network_id as lv_network,
                       l.baseline_peak_kw as peak,
                       l.baseline_avg_kw as average,
                       l.baseline_load_factor as load_factor,
                       l.baseline_self_sufficiency as self_suff,
                       l.baseline_peak_to_average as par,
                       m.timestamp as metric_time,
                       m.values as metric_values
            """
            
            result = session.run(query)
            baselines = [dict(record) for record in result]
            
            self.features['baseline_metrics'] = pd.DataFrame(baselines)
            logger.info(f"Extracted baseline metrics for {len(baselines)} LV networks")
    
    def create_synthetic_profiles(self):
        """Create synthetic energy profiles if real data not available"""
        logger.info("Creating synthetic energy profiles...")
        
        # Get building info
        buildings = self.nodes['buildings']
        
        # Create 24-hour profiles based on building type
        profiles = {
            'residential': np.array([2,2,2,2,3,5,8,9,7,5,4,4,5,6,7,8,9,8,7,6,5,4,3,2]),
            'office': np.array([1,1,1,1,2,3,5,8,15,18,20,20,18,18,20,18,15,10,5,3,2,1,1,1]),
            'retail': np.array([2,2,2,2,3,4,5,6,8,10,12,14,15,16,18,20,22,20,18,15,10,8,5,3]),
            'industrial': np.array([5,5,5,5,6,8,10,12,12,12,12,12,12,12,12,12,10,8,6,5,5,5,5,5])
        }
        
        # Generate profiles for each building
        building_profiles = {}
        for _, building in buildings.iterrows():
            # Determine building type
            if building['function'] == 'residential':
                base_profile = profiles['residential']
            elif building['non_res_type'] == 'Office':
                base_profile = profiles['office']
            elif building['non_res_type'] == 'Retail':
                base_profile = profiles['retail']
            else:
                base_profile = profiles['industrial']
            
            # Scale by building area
            scale = building['area'] / 100  # per 100m²
            
            # Add noise
            noise = np.random.normal(1, 0.1, 24)
            
            # Create week of data (168 hours)
            week_profile = np.tile(base_profile * scale * noise, 7)
            
            # Add weekly variation
            week_profile = week_profile * (1 + 0.1 * np.sin(np.arange(168) * 2 * np.pi / 168))
            
            # Convert to 15-min intervals
            profile_15min = np.repeat(week_profile, 4)  # 672 points
            
            building_profiles[building['id']] = profile_15min
        
        # Create DataFrame
        timestamps = pd.date_range('2024-01-01', periods=672, freq='15min')
        self.temporal_data['energy_profiles'] = pd.DataFrame(
            building_profiles,
            index=timestamps
        )
        
        logger.info(f"Created synthetic profiles for {len(building_profiles)} buildings")
    
    def extract_metadata(self) -> Dict:
        """Extract graph metadata and statistics"""
        with self.driver.session() as session:
            query = """
                MATCH (b:Building)
                WITH count(b) as num_buildings,
                     avg(b.area) as avg_area,
                     avg(b.peak_demand_kw) as avg_peak
                MATCH (l:LV_Network)
                WITH num_buildings, avg_area, avg_peak,
                     count(l) as num_lv_networks
                MATCH (m:MV_Transformer)
                WITH num_buildings, avg_area, avg_peak, num_lv_networks,
                     count(m) as num_transformers
                MATCH ()-[r]->()
                RETURN num_buildings, num_lv_networks, num_transformers,
                       count(r) as num_relationships,
                       avg_area, avg_peak,
                       datetime() as extraction_time
            """
            
            result = session.run(query).single()
            
            metadata = {
                'num_buildings': result['num_buildings'],
                'num_lv_networks': result['num_lv_networks'],
                'num_transformers': result['num_transformers'],
                'num_relationships': result['num_relationships'],
                'avg_building_area': result['avg_area'],
                'avg_peak_demand': result['avg_peak'],
                'extraction_time': str(result['extraction_time']),
                'graph_density': result['num_relationships'] / (result['num_buildings'] ** 2)
            }
            
            return metadata
    
    def validate_extraction(self) -> bool:
        """Validate extracted data for completeness and consistency"""
        is_valid = True
        
        # Check nodes exist
        if 'buildings' not in self.nodes or self.nodes['buildings'].empty:
            logger.error("No buildings extracted")
            is_valid = False
        
        # Check critical features
        if 'buildings' in self.nodes:
            required_cols = ['id', 'x', 'y', 'area', 'lv_network']
            missing = set(required_cols) - set(self.nodes['buildings'].columns)
            if missing:
                logger.error(f"Missing required columns: {missing}")
                is_valid = False
        
        # Check relationships
        if not self.edges or all(df.empty for df in self.edges.values()):
            logger.error("No relationships extracted")
            is_valid = False
        
        # Check temporal data
        if not self.temporal_data:
            logger.warning("No temporal data - will affect model performance")
        
        return is_valid

# Usage example
if __name__ == "__main__":
    # Extract graph from Neo4j
    extractor = KGExtractor()
    
    try:
        # Extract complete graph
        graph_data = extractor.extract_all()
        
        # Validate
        if extractor.validate_extraction():
            logger.info("Extraction successful and validated")
            
            # Save to disk for later use
            import pickle
            with open('processed_data/extracted_graph.pkl', 'wb') as f:
                pickle.dump(graph_data, f)
            logger.info("Graph data saved to processed_data/extracted_graph.pkl")
        else:
            logger.error("Extraction validation failed")
    
    finally:
        extractor.close()