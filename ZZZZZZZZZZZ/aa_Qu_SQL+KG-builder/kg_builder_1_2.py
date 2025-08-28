"""
KG Builder 1.2: Complete Electrical Grid Hierarchy
Includes FULL electrical grid hierarchy: HV Substations → MV Stations → LV Groups → Buildings
Based on SQL schema relationships discovered in grid analysis

Key Enhancements: 
- Adds HV (High Voltage) substation layer
- Adds MV (Medium Voltage) station relationships 
- Complete 4-level hierarchy that was missing in v1.0
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyKGBuilder_v1_2:
    """
    Complete Knowledge Graph Builder for Energy Data with HV-MV-LV Hierarchy
    Version 1.2: Adds complete voltage hierarchy (HV → MV → LV → Buildings)
    """
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 postgres_config: Optional[Dict[str, str]] = None):
        """
        Initialize KG Builder with database connections
        
        Args:
            neo4j_uri: Neo4j database URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            postgres_config: PostgreSQL connection config (optional)
        """
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.postgres_config = postgres_config
        self.stats = {
            'nodes': {'hv_substations': 0, 'mv_stations': 0, 'lv_groups': 0, 'buildings': 0, 
                     'transformers': 0, 'cable_groups': 0, 'substations': 0, 'adjacency_clusters': 0, 
                     'time_slots': 0},
            'relationships': {'HV_SUPPLIES_MV': 0, 'MV_SUPPLIES_LV': 0, 'LV_SUPPLIES_BUILDING': 0, 
                            'CONNECTED_TO': 0, 'BELONGS_TO': 0, 'ADJACENT_TO': 0, 'HAS_CONSUMPTION': 0,
                            'HAS_GENERATION': 0, 'MV_HAS_TRANSFORMER': 0, 'HV_HAS_SUBSTATION': 0}
        }
        logger.info("Initialized KG Builder v1.2 with complete HV-MV-LV hierarchy support")
    
    def clear_database(self) -> None:
        """Clear existing database (optional, use with caution)"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.warning("Database cleared")
    
    def create_hv_substations(self, hv_data: pd.DataFrame) -> None:
        """
        Create HV Substation nodes (highest level in hierarchy)
        
        Args:
            hv_data: DataFrame with columns:
                - substation_id: Unique HV substation identifier  
                - hv_group_id: HV group identifier
                - name: Substation name
                - capacity_mva: Substation capacity in MVA
                - voltage_kv: Voltage level in kV (typically 150kV)
                - location: Location information
        """
        logger.info(f"Creating {len(hv_data)} HV Substation nodes...")
        
        query = """
        UNWIND $batch AS row
        CREATE (hv:HVSubstation {
            substation_id: row.substation_id,
            group_id: coalesce(row.hv_group_id, row.group_id),
            name: coalesce(row.name, 'HV_' + toString(row.substation_id)),
            capacity_mva: coalesce(row.capacity_mva, 100.0),
            voltage_kv: coalesce(row.voltage_kv, 150.0),
            location: coalesce(row.location, ''),
            station_type: 'HV_SUBSTATION',
            hierarchy_level: 0,
            created_at: datetime()
        })
        """
        
        with self.driver.session() as session:
            batch = []
            for _, row in hv_data.iterrows():
                batch.append(row.to_dict())
                
                if len(batch) >= 1000:
                    session.run(query, batch=batch)
                    self.stats['nodes']['hv_substations'] += len(batch)
                    batch = []
            
            if batch:
                session.run(query, batch=batch)
                self.stats['nodes']['hv_substations'] += len(batch)
        
        logger.info(f"Created {self.stats['nodes']['hv_substations']} HV Substation nodes")
    
    def create_hv_mv_relationships(self, hv_mv_mapping: pd.DataFrame) -> None:
        """
        Create relationships between HV Substations and MV Stations
        
        Args:
            hv_mv_mapping: DataFrame with columns:
                - hv_substation_id: HV substation identifier
                - hv_group_id: HV group identifier
                - mv_station_id: MV station identifier
                - mv_group_id: MV group identifier
                - connection_type: Type of connection
                - distance_m: Distance between stations
        """
        logger.info("Creating HV-MV relationships...")
        
        query = """
        UNWIND $batch AS row
        MATCH (hv:HVSubstation {substation_id: row.hv_substation_id})
        MATCH (mv:MVStation {station_id: row.mv_station_id})
        CREATE (hv)-[r:HV_SUPPLIES_MV {
            hv_group_id: coalesce(row.hv_group_id, ''),
            mv_group_id: coalesce(row.mv_group_id, ''),
            connection_type: coalesce(row.connection_type, 'DIRECT'),
            distance_m: coalesce(row.distance_m, 0),
            created_at: datetime()
        }]->(mv)
        """
        
        with self.driver.session() as session:
            batch = []
            for _, row in hv_mv_mapping.iterrows():
                batch.append(row.to_dict())
                
                if len(batch) >= 500:
                    result = session.run(query, batch=batch)
                    self.stats['relationships']['HV_SUPPLIES_MV'] += result.summary().counters.relationships_created
                    batch = []
            
            if batch:
                result = session.run(query, batch=batch)
                self.stats['relationships']['HV_SUPPLIES_MV'] += result.summary().counters.relationships_created
        
        logger.info(f"Created {self.stats['relationships']['HV_SUPPLIES_MV']} HV-MV relationships")
    
    def create_mv_stations(self, mv_data: pd.DataFrame) -> None:
        """
        Create MV Station nodes (second level in hierarchy)
        
        Args:
            mv_data: DataFrame with columns:
                - mv_station_id: Unique MV station identifier
                - mv_group_id: MV group identifier
                - hv_substation_id: Parent HV substation (optional)
                - station_name: Station name (optional)
                - capacity_mva: Station capacity in MVA
                - voltage_kv: Voltage level in kV (typically 10kV)
                - location: Location information
        """
        logger.info(f"Creating {len(mv_data)} MV Station nodes...")
        
        query = """
        UNWIND $batch AS row
        CREATE (mv:MVStation {
            station_id: row.mv_station_id,
            group_id: row.mv_group_id,
            hv_parent: coalesce(row.hv_substation_id, ''),
            name: coalesce(row.station_name, 'MV_' + toString(row.mv_station_id)),
            capacity_mva: coalesce(row.capacity_mva, 10.0),
            voltage_kv: coalesce(row.voltage_kv, 10.0),
            location: coalesce(row.location, ''),
            station_type: 'MV',
            hierarchy_level: 1,
            created_at: datetime()
        })
        """
        
        with self.driver.session() as session:
            batch = []
            for _, row in mv_data.iterrows():
                batch.append(row.to_dict())
                
                if len(batch) >= 1000:
                    session.run(query, batch=batch)
                    self.stats['nodes']['mv_stations'] += len(batch)
                    batch = []
            
            if batch:
                session.run(query, batch=batch)
                self.stats['nodes']['mv_stations'] += len(batch)
        
        logger.info(f"Created {self.stats['nodes']['mv_stations']} MV Station nodes")
    
    def create_mv_lv_relationships(self, mv_lv_mapping: pd.DataFrame) -> None:
        """
        Create relationships between MV Stations and LV Groups
        
        Args:
            mv_lv_mapping: DataFrame with columns:
                - mv_station_id: MV station identifier
                - mv_group_id: MV group identifier
                - lv_group_id: LV group identifier
                - transformer_id: Transformer connecting them (optional)
                - connection_type: Type of connection
        """
        logger.info("Creating MV-LV relationships...")
        
        query = """
        UNWIND $batch AS row
        MATCH (mv:MVStation {station_id: row.mv_station_id})
        MATCH (lv:CableGroup {group_id: row.lv_group_id, voltage_level: 'LV'})
        CREATE (mv)-[r:MV_SUPPLIES_LV {
            transformer_id: coalesce(row.transformer_id, ''),
            connection_type: coalesce(row.connection_type, 'DIRECT'),
            mv_group_id: row.mv_group_id,
            created_at: datetime()
        }]->(lv)
        """
        
        with self.driver.session() as session:
            batch = []
            for _, row in mv_lv_mapping.iterrows():
                batch.append(row.to_dict())
                
                if len(batch) >= 500:
                    result = session.run(query, batch=batch)
                    self.stats['relationships']['MV_SUPPLIES_LV'] += result.summary().counters.relationships_created
                    batch = []
            
            if batch:
                result = session.run(query, batch=batch)
                self.stats['relationships']['MV_SUPPLIES_LV'] += result.summary().counters.relationships_created
        
        logger.info(f"Created {self.stats['relationships']['MV_SUPPLIES_LV']} MV-LV relationships")
    
    def create_lv_groups_with_hierarchy(self, lv_data: pd.DataFrame) -> None:
        """
        Create LV Cable Groups with MV hierarchy information
        
        Enhanced from v1.0 to include MV parent information
        """
        logger.info(f"Creating {len(lv_data)} LV Cable Groups with hierarchy...")
        
        query = """
        UNWIND $batch AS row
        MERGE (cg:CableGroup {group_id: row.group_id})
        SET cg += {
            station_fid: row.station_fid,
            voltage_level: 'LV',
            mv_parent_station: coalesce(row.mv_station_id, ''),
            mv_parent_group: coalesce(row.mv_group_id, ''),
            total_connections: coalesce(row.total_connections, 0),
            residential_connections: coalesce(row.residential_connections, 0),
            non_residential_connections: coalesce(row.non_residential_connections, 0),
            avg_distance_m: coalesce(row.avg_distance_m, 0),
            max_distance_m: coalesce(row.max_distance_m, 0),
            district: coalesce(row.district, ''),
            neighborhood: coalesce(row.neighborhood, ''),
            created_at: datetime()
        }
        """
        
        with self.driver.session() as session:
            batch = []
            for _, row in lv_data.iterrows():
                batch.append(row.to_dict())
                
                if len(batch) >= 1000:
                    session.run(query, batch=batch)
                    self.stats['nodes']['lv_groups'] += len(batch)
                    batch = []
            
            if batch:
                session.run(query, batch=batch)
                self.stats['nodes']['lv_groups'] += len(batch)
        
        logger.info(f"Created/Updated {self.stats['nodes']['lv_groups']} LV Groups")
    
    def create_buildings_with_full_hierarchy(self, building_data: pd.DataFrame) -> None:
        """
        Create Building nodes with complete MV-LV hierarchy
        
        Enhanced to include MV station information
        """
        logger.info(f"Creating {len(building_data)} Building nodes with hierarchy...")
        
        query = """
        UNWIND $batch AS row
        CREATE (b:Building {
            ogc_fid: row.building_id,
            lv_group: row.lv_group_id,
            mv_station: coalesce(row.mv_station_id, ''),
            mv_group: coalesce(row.mv_group_id, ''),
            building_function: row.building_function,
            building_type: coalesce(row.building_type, 'unknown'),
            building_area: coalesce(row.building_area, 100),
            connection_type: row.connection_type,
            connection_distance_m: coalesce(row.connection_distance_m, 0),
            is_mv_capable: coalesce(row.is_mv_capable, false),
            has_mv_nearby: coalesce(row.has_mv_nearby, false),
            is_problematic: coalesce(row.is_problematic, false),
            age_range: coalesce(row.age_range, 'unknown'),
            energy_label: coalesce(row.energy_label, 'D'),
            housing_type: coalesce(row.housing_type, ''),
            district_name: coalesce(row.district_name, ''),
            neighborhood_name: coalesce(row.neighborhood_name, ''),
            has_solar: coalesce(row.has_solar, false),
            has_battery: coalesce(row.has_battery, false),
            has_heat_pump: coalesce(row.has_heat_pump, false),
            created_at: datetime()
        })
        """
        
        with self.driver.session() as session:
            batch = []
            for _, row in building_data.iterrows():
                batch.append(row.to_dict())
                
                if len(batch) >= 1000:
                    session.run(query, batch=batch)
                    self.stats['nodes']['buildings'] += len(batch)
                    batch = []
            
            if batch:
                session.run(query, batch=batch)
                self.stats['nodes']['buildings'] += len(batch)
        
        logger.info(f"Created {self.stats['nodes']['buildings']} Building nodes")
    
    def create_transformers_with_mv_links(self, transformer_data: pd.DataFrame) -> None:
        """
        Create Transformer nodes and link them to MV stations
        
        New in v1.2: Explicit MV station relationships
        """
        logger.info("Creating Transformers with MV links...")
        
        # Create transformer nodes
        create_query = """
        UNWIND $batch AS row
        CREATE (t:Transformer {
            ogc_fid: row.ogc_fid,
            station_fid: row.station_fid,
            mv_station_id: coalesce(row.mv_station_id, ''),
            mv_group_id: coalesce(row.mv_group_id, ''),
            name: coalesce(row.name, 'T_' + toString(row.ogc_fid)),
            capacity_kva: coalesce(row.capacity_kva, 630),
            primary_voltage_kv: coalesce(row.primary_voltage_kv, 10.0),
            secondary_voltage_kv: coalesce(row.secondary_voltage_kv, 0.4),
            transformer_type: coalesce(row.transformer_type, 'distribution'),
            loading_percent: coalesce(row.loading_percent, 50.0),
            created_at: datetime()
        })
        """
        
        # Link transformers to MV stations
        link_query = """
        UNWIND $batch AS row
        MATCH (t:Transformer {ogc_fid: row.ogc_fid})
        MATCH (mv:MVStation {station_id: row.mv_station_id})
        CREATE (mv)-[r:MV_HAS_TRANSFORMER {
            transformer_type: coalesce(row.transformer_type, 'distribution'),
            capacity_kva: coalesce(row.capacity_kva, 630)
        }]->(t)
        """
        
        with self.driver.session() as session:
            batch = []
            for _, row in transformer_data.iterrows():
                batch.append(row.to_dict())
                
                if len(batch) >= 500:
                    # Create nodes
                    session.run(create_query, batch=batch)
                    self.stats['nodes']['transformers'] += len(batch)
                    
                    # Create relationships if MV station exists
                    if 'mv_station_id' in transformer_data.columns:
                        result = session.run(link_query, batch=batch)
                        self.stats['relationships']['MV_HAS_TRANSFORMER'] += result.summary().counters.relationships_created
                    
                    batch = []
            
            if batch:
                session.run(create_query, batch=batch)
                self.stats['nodes']['transformers'] += len(batch)
                
                if 'mv_station_id' in transformer_data.columns:
                    result = session.run(link_query, batch=batch)
                    self.stats['relationships']['MV_HAS_TRANSFORMER'] += result.summary().counters.relationships_created
        
        logger.info(f"Created {self.stats['nodes']['transformers']} Transformers with {self.stats['relationships']['MV_HAS_TRANSFORMER']} MV links")
    
    def create_hierarchical_supply_chain(self) -> None:
        """
        Create the complete supply chain: MV → Transformer → LV → Building
        
        New in v1.2: Complete hierarchical relationships
        """
        logger.info("Creating hierarchical supply chain relationships...")
        
        # LV Groups to Buildings (existing)
        lv_building_query = """
        MATCH (cg:CableGroup {voltage_level: 'LV'})
        MATCH (b:Building {lv_group: cg.group_id})
        MERGE (cg)-[r:LV_SUPPLIES_BUILDING]->(b)
        SET r.created_at = datetime()
        RETURN count(r) as created
        """
        
        # Transformers to LV Groups
        transformer_lv_query = """
        MATCH (t:Transformer)
        MATCH (lv:CableGroup {station_fid: t.station_fid, voltage_level: 'LV'})
        MERGE (t)-[r:SUPPLIES_LV_GROUP]->(lv)
        SET r.created_at = datetime()
        RETURN count(r) as created
        """
        
        with self.driver.session() as session:
            # Create LV to Building relationships
            result = session.run(lv_building_query)
            lv_building_count = result.single()['created']
            self.stats['relationships']['LV_SUPPLIES_BUILDING'] = lv_building_count
            logger.info(f"Created {lv_building_count} LV→Building relationships")
            
            # Create Transformer to LV relationships
            result = session.run(transformer_lv_query)
            transformer_lv_count = result.single()['created']
            logger.info(f"Created {transformer_lv_count} Transformer→LV relationships")
    
    def add_grid_hierarchy_properties(self) -> None:
        """
        Add computed hierarchy properties for easier querying
        
        New in v1.2: Complete hierarchy depth and path properties (HV→MV→LV→Building)
        """
        logger.info("Adding grid hierarchy properties...")
        
        queries = [
            # Add hierarchy level to nodes (HV=0, MV=1, LV=2, Building=3)
            """
            MATCH (hv:HVSubstation)
            SET hv.hierarchy_level = 0
            """,
            """
            MATCH (mv:MVStation)
            SET mv.hierarchy_level = 1
            """,
            """
            MATCH (t:Transformer)
            SET t.hierarchy_level = 2
            """,
            """
            MATCH (lv:CableGroup {voltage_level: 'LV'})
            SET lv.hierarchy_level = 3
            """,
            """
            MATCH (b:Building)
            SET b.hierarchy_level = 4
            """,
            
            # Add supply path for buildings
            """
            MATCH path = (mv:MVStation)-[:MV_SUPPLIES_LV|MV_HAS_TRANSFORMER|SUPPLIES_LV_GROUP*1..3]->(lv:CableGroup)-[:LV_SUPPLIES_BUILDING]->(b:Building)
            WITH b, mv, lv, length(path) as path_length
            SET b.supply_path_length = path_length,
                b.upstream_mv_station = mv.station_id,
                b.upstream_lv_group = lv.group_id
            """,
            
            # Calculate aggregates for MV stations
            """
            MATCH (mv:MVStation)-[:MV_SUPPLIES_LV]->(lv:CableGroup)-[:LV_SUPPLIES_BUILDING]->(b:Building)
            WITH mv, count(DISTINCT lv) as lv_count, count(DISTINCT b) as building_count,
                 sum(CASE WHEN b.has_solar THEN 1 ELSE 0 END) as solar_count,
                 avg(b.connection_distance_m) as avg_distance
            SET mv.total_lv_groups = lv_count,
                mv.total_buildings = building_count,
                mv.solar_installations = solar_count,
                mv.avg_connection_distance = avg_distance
            """
        ]
        
        with self.driver.session() as session:
            for query in queries:
                result = session.run(query)
                logger.info(f"Updated {result.summary().counters.properties_set} properties")
    
    def load_from_postgres(self) -> Dict[str, pd.DataFrame]:
        """
        Load data from PostgreSQL following the SQL schema
        
        Returns dict of DataFrames matching the complete HV-MV-LV hierarchical structure
        """
        if not self.postgres_config:
            logger.warning("No PostgreSQL config provided, using sample data")
            return self._generate_sample_data()
        
        import psycopg2
        import pandas as pd
        
        conn = psycopg2.connect(**self.postgres_config)
        
        # Load HV-MV mapping (from STEPS 2-3.sql lines 281-307)
        hv_mv_query = """
        WITH mv_to_hv AS (
            SELECT DISTINCT
                s.name as hv_substation_id,
                hv.group_id as hv_group_id,
                mv.group_id as mv_group_id,
                mv.station_fid as mv_station_id,
                mv.distance_m + hv.distance_m as total_distance,
                (mv.confidence_score + hv.confidence_score) / 2 as avg_confidence
            FROM amin_grid.tlip_group_stations mv
            JOIN amin_grid.tlip_group_stations hv
                ON mv.station_fid = hv.station_fid
                AND mv.station_type = 'SUBSTATION'
                AND hv.station_type = 'SUBSTATION'
                AND mv.voltage_level = 'MV'
                AND hv.voltage_level = 'HV'
            JOIN amin_grid.tlip_substations s
                ON hv.station_fid = s.ogc_fid
        )
        SELECT * FROM mv_to_hv
        """
        
        # Load MV-LV mapping (from HIERARCHICAL ELECTRICAL GRID SUMMARY.sql)
        mv_lv_query = """
        WITH mv_lv_mapping AS (
            SELECT DISTINCT
                mv_gs.station_fid as mv_station_id,
                mv_gs.group_id as mv_group_id,
                lv_gs.group_id as lv_group_id
            FROM amin_grid.tlip_group_stations mv_gs
            JOIN amin_grid.tlip_group_stations lv_gs
                ON mv_gs.station_fid = lv_gs.station_fid
                AND mv_gs.station_type = 'TRANSFORMER'
                AND lv_gs.station_type = 'TRANSFORMER'
                AND mv_gs.voltage_level = 'MV'
                AND lv_gs.voltage_level = 'LV'
        )
        SELECT * FROM mv_lv_mapping
        """
        
        # Load building data with hierarchy
        building_query = """
        WITH mv_lv_mapping AS (
            SELECT DISTINCT
                mv_gs.station_fid as mv_station_id,
                mv_gs.group_id as mv_group_id,
                lv_gs.group_id as lv_group_id
            FROM amin_grid.tlip_group_stations mv_gs
            JOIN amin_grid.tlip_group_stations lv_gs
                ON mv_gs.station_fid = lv_gs.station_fid
                AND mv_gs.station_type = 'TRANSFORMER'
                AND lv_gs.station_type = 'TRANSFORMER'
                AND mv_gs.voltage_level = 'MV'
                AND lv_gs.voltage_level = 'LV'
        )
        SELECT 
            b.building_id,
            b.connected_group_id as lv_group_id,
            m.mv_station_id,
            m.mv_group_id,
            b.building_function,
            b.building_type,
            b.building_area,
            b.connection_type,
            b.connection_distance_m,
            b.is_mv_capable,
            b.has_mv_nearby,
            b.is_problematic,
            bd.age_range,
            bd.meestvoorkomendelabel as energy_label,
            bd.woningtype as housing_type,
            bd.wijknaam as district_name,
            bd.buurtnaam as neighborhood_name
        FROM amin_grid.tlip_building_connections b
        LEFT JOIN amin_grid.tlip_buildings_1_deducted bd
            ON b.building_id = bd.ogc_fid
        LEFT JOIN mv_lv_mapping m
            ON b.connected_group_id = m.lv_group_id
        """
        
        data = {
            'mv_lv_mapping': pd.read_sql(mv_lv_query, conn),
            'buildings': pd.read_sql(building_query, conn)
        }
        
        # Extract unique MV stations
        data['mv_stations'] = data['mv_lv_mapping'][['mv_station_id', 'mv_group_id']].drop_duplicates()
        
        # Extract LV groups with counts
        lv_stats = data['buildings'].groupby('lv_group_id').agg({
            'building_id': 'count',
            'building_function': lambda x: (x == 'residential').sum(),
            'connection_distance_m': ['mean', 'max']
        }).reset_index()
        lv_stats.columns = ['group_id', 'total_connections', 'residential_connections', 'avg_distance_m', 'max_distance_m']
        
        # Merge with MV mapping
        data['lv_groups'] = pd.merge(
            lv_stats,
            data['mv_lv_mapping'][['lv_group_id', 'mv_station_id', 'mv_group_id']].drop_duplicates(),
            left_on='group_id',
            right_on='lv_group_id',
            how='left'
        )
        
        conn.close()
        logger.info(f"Loaded data from PostgreSQL: {len(data['mv_stations'])} MV stations, {len(data['lv_groups'])} LV groups, {len(data['buildings'])} buildings")
        
        return data
    
    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate sample data with complete HV-MV-LV hierarchy for testing"""
        np.random.seed(42)
        
        # Create HV substations (2 substations)
        hv_substations = pd.DataFrame({
            'substation_id': [f'HV_SUB_{i:02d}' for i in range(1, 3)],
            'hv_group_id': [f'HV_GROUP_{i:02d}' for i in range(1, 3)],
            'name': [f'HV Substation {i}' for i in range(1, 3)],
            'capacity_mva': [100, 150],
            'voltage_kv': [150.0, 150.0]
        })
        
        # Create MV stations (5 stations, distributed between HV substations)
        mv_stations = pd.DataFrame({
            'mv_station_id': [f'MV_STATION_{i:03d}' for i in range(1, 6)],
            'mv_group_id': [f'MV_GROUP_{i:03d}' for i in range(1, 6)],
            'hv_substation_id': [f'HV_SUB_{(i-1)//3 + 1:02d}' for i in range(1, 6)],  # Distribute between HV subs
            'station_name': [f'MV Station {i}' for i in range(1, 6)],
            'capacity_mva': np.random.uniform(5, 20, 5),
            'voltage_kv': [10.0] * 5
        })
        
        # Create HV-MV mapping
        hv_mv_mapping = pd.DataFrame({
            'hv_substation_id': [f'HV_SUB_{(i-1)//3 + 1:02d}' for i in range(1, 6)],
            'hv_group_id': [f'HV_GROUP_{(i-1)//3 + 1:02d}' for i in range(1, 6)],
            'mv_station_id': [f'MV_STATION_{i:03d}' for i in range(1, 6)],
            'mv_group_id': [f'MV_GROUP_{i:03d}' for i in range(1, 6)],
            'connection_type': ['DIRECT'] * 5,
            'distance_m': np.random.uniform(100, 1000, 5)
        })
        
        # Create LV groups (20 groups, 4 per MV station)
        lv_groups = []
        mv_lv_mapping = []
        for i, mv_row in mv_stations.iterrows():
            for j in range(4):
                lv_id = f'LV_GROUP_{i*4 + j + 1:03d}'
                lv_groups.append({
                    'group_id': lv_id,
                    'station_fid': f'STATION_{i*4 + j + 1:03d}',
                    'total_connections': np.random.randint(10, 50),
                    'residential_connections': np.random.randint(5, 40),
                    'avg_distance_m': np.random.uniform(10, 100),
                    'max_distance_m': np.random.uniform(100, 500)
                })
                mv_lv_mapping.append({
                    'mv_station_id': mv_row['mv_station_id'],
                    'mv_group_id': mv_row['mv_group_id'],
                    'lv_group_id': lv_id,
                    'transformer_id': f'T_{i*4 + j + 1:03d}'
                })
        
        lv_groups_df = pd.DataFrame(lv_groups)
        mv_lv_mapping_df = pd.DataFrame(mv_lv_mapping)
        
        # Merge MV info into LV groups
        lv_groups_df = pd.merge(
            lv_groups_df,
            mv_lv_mapping_df,
            left_on='group_id',
            right_on='lv_group_id',
            how='left'
        )
        
        # Create buildings (200 buildings distributed across LV groups)
        buildings = []
        for _, lv_row in lv_groups_df.iterrows():
            num_buildings = np.random.randint(5, 15)
            for b in range(num_buildings):
                buildings.append({
                    'building_id': f'B_{len(buildings) + 1:05d}',
                    'lv_group_id': lv_row['group_id'],
                    'mv_station_id': lv_row['mv_station_id'],
                    'mv_group_id': lv_row['mv_group_id'],
                    'building_function': np.random.choice(['residential', 'non_residential'], p=[0.85, 0.15]),
                    'building_type': np.random.choice(['vrijstaand', 'twee_onder_1_kap', 'rijtjeswoning', 'appartement']),
                    'building_area': np.random.uniform(50, 300),
                    'connection_type': np.random.choice(['ENDED', 'ENTERED', 'CROSSED', 'BY_DISTANCE']),
                    'connection_distance_m': np.random.uniform(5, 200),
                    'is_mv_capable': np.random.random() > 0.9,
                    'has_mv_nearby': np.random.random() > 0.8,
                    'is_problematic': np.random.random() > 0.95,
                    'age_range': np.random.choice(['< 1945', '1945-1975', '1975-1990', '1990-2005', '2005-2015', '> 2015']),
                    'energy_label': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], p=[0.1, 0.15, 0.25, 0.25, 0.15, 0.07, 0.03]),
                    'has_solar': np.random.random() > 0.8,
                    'has_battery': np.random.random() > 0.95,
                    'has_heat_pump': np.random.random() > 0.85
                })
        
        buildings_df = pd.DataFrame(buildings)
        
        # Create transformers
        transformers = []
        for _, mapping in mv_lv_mapping_df.iterrows():
            transformers.append({
                'ogc_fid': mapping['transformer_id'],
                'station_fid': f'STATION_{len(transformers) + 1:03d}',
                'mv_station_id': mapping['mv_station_id'],
                'mv_group_id': mapping['mv_group_id'],
                'name': f'Transformer {mapping["transformer_id"]}',
                'capacity_kva': np.random.choice([250, 400, 630, 1000]),
                'primary_voltage_kv': 10.0,
                'secondary_voltage_kv': 0.4,
                'transformer_type': 'distribution',
                'loading_percent': np.random.uniform(30, 80)
            })
        
        transformers_df = pd.DataFrame(transformers)
        
        return {
            'hv_substations': hv_substations,
            'mv_stations': mv_stations,
            'lv_groups': lv_groups_df,
            'buildings': buildings_df,
            'hv_mv_mapping': hv_mv_mapping,
            'mv_lv_mapping': mv_lv_mapping_df,
            'transformers': transformers_df
        }
    
    def build_complete_graph(self, data: Optional[Dict[str, pd.DataFrame]] = None) -> None:
        """
        Build the complete knowledge graph with HV-MV-LV hierarchy
        
        Args:
            data: Optional dict of DataFrames. If None, will load from PostgreSQL or generate sample
        """
        logger.info("Building complete energy knowledge graph v1.2 with HV-MV-LV hierarchy...")
        start_time = datetime.now()
        
        # Load data if not provided
        if data is None:
            data = self.load_from_postgres()
        
        # Create nodes in hierarchical order (top-down)
        if 'hv_substations' in data:
            self.create_hv_substations(data['hv_substations'])
        
        self.create_mv_stations(data['mv_stations'])
        self.create_lv_groups_with_hierarchy(data['lv_groups'])
        self.create_buildings_with_full_hierarchy(data['buildings'])
        
        if 'transformers' in data:
            self.create_transformers_with_mv_links(data['transformers'])
        
        # Create relationships
        if 'hv_mv_mapping' in data:
            self.create_hv_mv_relationships(data['hv_mv_mapping'])
        
        self.create_mv_lv_relationships(data['mv_lv_mapping'])
        self.create_hierarchical_supply_chain()
        
        # Add computed properties
        self.add_grid_hierarchy_properties()
        
        # Create indexes for performance
        self._create_indexes()
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Graph construction completed in {elapsed:.2f} seconds")
        self.print_statistics()
    
    def _create_indexes(self) -> None:
        """Create indexes for better query performance"""
        indexes = [
            ("MVStation", "station_id"),
            ("MVStation", "group_id"),
            ("CableGroup", "group_id"),
            ("Building", "ogc_fid"),
            ("Building", "lv_group"),
            ("Building", "mv_station"),
            ("Transformer", "ogc_fid"),
            ("Transformer", "mv_station_id")
        ]
        
        with self.driver.session() as session:
            for label, property in indexes:
                query = f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.{property})"
                session.run(query)
        
        logger.info(f"Created {len(indexes)} indexes")
    
    def verify_hierarchy(self) -> Dict[str, Any]:
        """
        Verify the complete HV-MV-LV-Building hierarchy is correctly built
        
        Returns statistics about the hierarchy
        """
        logger.info("Verifying complete hierarchy...")
        
        queries = {
            'hv_substation_count': "MATCH (hv:HVSubstation) RETURN count(hv) as count",
            'mv_station_count': "MATCH (mv:MVStation) RETURN count(mv) as count",
            'lv_group_count': "MATCH (lv:CableGroup {voltage_level: 'LV'}) RETURN count(lv) as count",
            'building_count': "MATCH (b:Building) RETURN count(b) as count",
            'hv_mv_relationships': "MATCH (:HVSubstation)-[r:HV_SUPPLIES_MV]->() RETURN count(r) as count",
            'mv_lv_relationships': "MATCH (:MVStation)-[r:MV_SUPPLIES_LV]->() RETURN count(r) as count",
            'lv_building_relationships': "MATCH (:CableGroup)-[r:LV_SUPPLIES_BUILDING]->() RETURN count(r) as count",
            'complete_hv_paths': """
                MATCH path = (hv:HVSubstation)-[:HV_SUPPLIES_MV]->(mv:MVStation)-[:MV_SUPPLIES_LV]->(lv:CableGroup)-[:LV_SUPPLIES_BUILDING]->(b:Building)
                RETURN count(DISTINCT b) as buildings_with_complete_hv_path
            """,
            'complete_mv_paths': """
                MATCH path = (mv:MVStation)-[:MV_SUPPLIES_LV]->(lv:CableGroup)-[:LV_SUPPLIES_BUILDING]->(b:Building)
                RETURN count(DISTINCT b) as buildings_with_complete_mv_path
            """,
            'orphan_buildings': """
                MATCH (b:Building)
                WHERE NOT EXISTS((b)<-[:LV_SUPPLIES_BUILDING]-())
                RETURN count(b) as count
            """,
            'mv_statistics': """
                MATCH (mv:MVStation)
                OPTIONAL MATCH (mv)-[:MV_SUPPLIES_LV]->(lv:CableGroup)
                OPTIONAL MATCH (lv)-[:LV_SUPPLIES_BUILDING]->(b:Building)
                WITH mv, count(DISTINCT lv) as lv_count, count(DISTINCT b) as building_count
                RETURN avg(lv_count) as avg_lv_per_mv, avg(building_count) as avg_buildings_per_mv
            """
        }
        
        results = {}
        with self.driver.session() as session:
            for name, query in queries.items():
                result = session.run(query).single()
                if result:
                    results[name] = dict(result)
                else:
                    results[name] = {}
        
        # Print verification results
        logger.info("=== COMPLETE HIERARCHY VERIFICATION ===")
        logger.info(f"HV Substations: {results.get('hv_substation_count', {}).get('count', 0)}")
        logger.info(f"MV Stations: {results.get('mv_station_count', {}).get('count', 0)}")
        logger.info(f"LV Groups: {results.get('lv_group_count', {}).get('count', 0)}")
        logger.info(f"Buildings: {results.get('building_count', {}).get('count', 0)}")
        logger.info(f"HV→MV Relationships: {results.get('hv_mv_relationships', {}).get('count', 0)}")
        logger.info(f"MV→LV Relationships: {results.get('mv_lv_relationships', {}).get('count', 0)}")
        logger.info(f"LV→Building Relationships: {results.get('lv_building_relationships', {}).get('count', 0)}")
        logger.info(f"Buildings with complete HV→MV→LV→Building path: {results.get('complete_hv_paths', {}).get('buildings_with_complete_hv_path', 0)}")
        logger.info(f"Buildings with MV→LV→Building path: {results.get('complete_mv_paths', {}).get('buildings_with_complete_mv_path', 0)}")
        logger.info(f"Orphan buildings: {results.get('orphan_buildings', {}).get('count', 0)}")
        
        if 'mv_statistics' in results and results['mv_statistics']:
            logger.info(f"Average LV groups per MV: {results['mv_statistics'].get('avg_lv_per_mv', 0):.1f}")
            logger.info(f"Average buildings per MV: {results['mv_statistics'].get('avg_buildings_per_mv', 0):.1f}")
        
        return results
    
    def print_statistics(self) -> None:
        """Print construction statistics"""
        print("\n" + "="*70)
        print("KG BUILDER v1.2 CONSTRUCTION STATISTICS")
        print("="*70)
        print("\nNODES CREATED:")
        for node_type, count in self.stats['nodes'].items():
            if count > 0:
                print(f"  {node_type}: {count:,}")
        
        print("\nRELATIONSHIPS CREATED:")
        for rel_type, count in self.stats['relationships'].items():
            if count > 0:
                print(f"  {rel_type}: {count:,}")
        
        print("\nCOMPLETE HIERARCHY:")
        print("  HV Substations (150kV)")
        print("       ↓")
        print("  MV Stations (10kV)")
        print("       ↓")
        print("  LV Groups (0.4kV)")
        print("       ↓")
        print("  Buildings")
        print("="*70)
    
    def close(self):
        """Close database connection"""
        self.driver.close()
        logger.info("Database connection closed")


def main():
    """Example usage of KG Builder v1.2"""
    
    # Configuration
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "aminasad"
    
    # PostgreSQL config (optional)
    postgres_config = {
        'host': 'localhost',
        'port': 5432,
        'database': 'neo4j',
        'user': 'neo4j',
        'password': 'aminasad'
    }
    
    # Initialize builder
    builder = EnergyKGBuilder_v1_2(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD,
        postgres_config=postgres_config  # Or None for sample data
    )
    
    try:
        # Optional: Clear existing data
        # builder.clear_database()
        
        # Build complete graph
        builder.build_complete_graph()
        
        # Verify hierarchy
        verification = builder.verify_hierarchy()
        
        print("\n✅ Knowledge Graph v1.2 with complete HV-MV-LV hierarchy built successfully!")
        print("   Hierarchy: HV Substations → MV Stations → LV Groups → Buildings")
        
    finally:
        builder.close()


if __name__ == "__main__":
    main()