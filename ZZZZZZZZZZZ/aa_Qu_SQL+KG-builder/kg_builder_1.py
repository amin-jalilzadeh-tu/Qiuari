"""
Knowledge Graph Builder for Energy District Analysis - PRE-GNN VERSION
Updated to use actual PostgreSQL database schema with rich grid infrastructure
Creates the foundational KG with raw data, infrastructure, and potential
Complementarity and clustering will be added AFTER GNN analysis
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
from decimal import Decimal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnergyKnowledgeGraphBuilder:
    """Build Knowledge Graph from energy district data - Pre-GNN version"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 pg_host: str, pg_database: str, pg_user: str, pg_password: str):
        """Initialize Neo4j and PostgreSQL connections"""
        # Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # PostgreSQL connection
        self.pg_conn = psycopg2.connect(
            host=pg_host,
            database=pg_database,
            user=pg_user,
            password=pg_password,
            port=5433
        )
        
        self.stats = {
            'nodes_created': {},
            'relationships_created': {},
            'processing_time': {}
        }
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
        logger.info(f"Connected to PostgreSQL database {pg_database}")
    
    def close(self):
        """Close database connections"""
        self.driver.close()
        self.pg_conn.close()
        logger.info("Database connections closed")
    
    def clear_database(self):
        """Clear all nodes and relationships"""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            logger.info("Neo4j database cleared")
    
    def convert_decimals(self, record):
        """Convert Decimal types to float for Neo4j compatibility"""
        if isinstance(record, dict):
            return {k: float(v) if isinstance(v, Decimal) else v for k, v in record.items()}
        elif isinstance(record, list):
            return [self.convert_decimals(item) for item in record]
        else:
            return float(record) if isinstance(record, Decimal) else record

    # ============================================
    # STEP 1: SCHEMA SETUP
    # ============================================
    
    def create_schema(self):
        """Create constraints and indexes for pre-GNN KG"""
        logger.info("Creating schema constraints and indexes...")
        
        constraints = [
            # Infrastructure
            "CREATE CONSTRAINT IF NOT EXISTS FOR (s:Substation) REQUIRE s.station_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Transformer) REQUIRE t.transformer_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:LVCabinet) REQUIRE c.cabinet_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:CableGroup) REQUIRE g.group_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (seg:CableSegment) REQUIRE seg.segment_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (b:Building) REQUIRE b.ogc_fid IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (cp:ConnectionPoint) REQUIRE cp.point_id IS UNIQUE",
            
            # Temporal
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:TimeSlot) REQUIRE t.slot_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:EnergyState) REQUIRE e.state_id IS UNIQUE",
            
            # Assets
            "CREATE CONSTRAINT IF NOT EXISTS FOR (sol:SolarSystem) REQUIRE sol.system_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (bat:BatterySystem) REQUIRE bat.system_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (hp:HeatPumpSystem) REQUIRE hp.system_id IS UNIQUE",
        ]
        
        indexes = [
            # Performance indexes
            "CREATE INDEX IF NOT EXISTS FOR (g:CableGroup) ON (g.voltage_level)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Building) ON (b.lv_group_id)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Building) ON (b.building_function)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Building) ON (b.district_name)",
            "CREATE INDEX IF NOT EXISTS FOR (b:Building) ON (b.has_solar)",
            "CREATE INDEX IF NOT EXISTS FOR (seg:CableSegment) ON (seg.group_id)",
            "CREATE INDEX IF NOT EXISTS FOR (cp:ConnectionPoint) ON (cp.building_id)",
            "CREATE INDEX IF NOT EXISTS FOR (t:TimeSlot) ON (t.timestamp)",
            "CREATE INDEX IF NOT EXISTS FOR (t:TimeSlot) ON (t.hour_of_day)",
            "CREATE INDEX IF NOT EXISTS FOR (e:EnergyState) ON (e.building_id)",
        ]
        
        with self.driver.session() as session:
            for constraint in constraints:
                session.run(constraint)
            for index in indexes:
                session.run(index)
        
        logger.info("Schema created successfully")
    
    # ============================================
    # STEP 2: LOAD GRID INFRASTRUCTURE
    # ============================================
    
    def load_grid_infrastructure(self):
        """Load complete grid topology from PostgreSQL"""
        logger.info("Loading grid infrastructure from PostgreSQL...")
        
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            
            # Load Substations
            cursor.execute("""
                SELECT 
                    fid as station_id,
                    ST_X(clipped_geom) as x,
                    ST_Y(clipped_geom) as y,
                    ST_AsText(clipped_geom) as geom_wkt
                FROM amin_grid.tlip_onderstations
                WHERE clipped_geom IS NOT NULL
            """)
            substations = [self.convert_decimals(row) for row in cursor.fetchall()]
            
            # Load Transformers (MV installations)
            cursor.execute("""
                SELECT 
                    fid as transformer_id,
                    ST_X(clipped_geom) as x,
                    ST_Y(clipped_geom) as y,
                    ST_AsText(clipped_geom) as geom_wkt
                FROM amin_grid.tlip_middenspanningsinstallaties
                WHERE clipped_geom IS NOT NULL
            """)
            transformers = [self.convert_decimals(row) for row in cursor.fetchall()]
            
            # Load LV Cabinets
            cursor.execute("""
                SELECT 
                    fid as cabinet_id,
                    ST_X(clipped_geom) as x,
                    ST_Y(clipped_geom) as y,
                    ST_AsText(clipped_geom) as geom_wkt
                FROM amin_grid.tlip_laagspanningsverdeelkasten
                WHERE clipped_geom IS NOT NULL
            """)
            lv_cabinets = [self.convert_decimals(row) for row in cursor.fetchall()]
            
            # Load Cable Groups (connected components)
            cursor.execute("""
                SELECT 
                    group_id,
                    voltage_level,
                    segment_count,
                    total_length_m,
                    ST_X(centroid) as x,
                    ST_Y(centroid) as y,
                    ST_AsText(bbox) as bbox_wkt
                FROM amin_grid.tlip_connected_groups
            """)
            cable_groups = [self.convert_decimals(row) for row in cursor.fetchall()]
            
            # Load Cable Segments
            cursor.execute("""
                SELECT 
                    segment_id,
                    original_fid,
                    voltage_level,
                    group_id,
                    length_m,
                    ST_X(start_point) as start_x,
                    ST_Y(start_point) as start_y,
                    ST_X(end_point) as end_x,
                    ST_Y(end_point) as end_y
                FROM amin_grid.tlip_cable_segments
                WHERE group_id IS NOT NULL
            """)
            cable_segments = [self.convert_decimals(row) for row in cursor.fetchall()]
            
            # Load hierarchy relationships
            cursor.execute("""
                SELECT 
                    child_group_id,
                    child_voltage,
                    parent_group_id,
                    parent_voltage,
                    connection_via,
                    via_station_fid,
                    confidence_score
                FROM amin_grid.tlip_group_hierarchy
            """)
            hierarchy = [self.convert_decimals(row) for row in cursor.fetchall()]
            
            # Load group-station connections
            cursor.execute("""
                SELECT 
                    group_id,
                    voltage_level,
                    station_type,
                    station_fid,
                    connection_type,
                    distance_m,
                    confidence_score
                FROM amin_grid.tlip_group_stations
            """)
            group_stations = [self.convert_decimals(row) for row in cursor.fetchall()]
        
        # Create nodes in Neo4j
        with self.driver.session() as session:
            # Create Substations
            for substation in substations:
                session.run("""
                    CREATE (s:GridComponent:Substation {
                        station_id: $station_id,
                        x: $x,
                        y: $y,
                        voltage_level: 'HV',
                        component_type: 'substation',
                        geom_wkt: $geom_wkt
                    })
                """, **substation)
            self.stats['nodes_created']['Substation'] = len(substations)
            
            # Create Transformers
            for transformer in transformers:
                session.run("""
                    CREATE (t:GridComponent:Transformer {
                        transformer_id: $transformer_id,
                        x: $x,
                        y: $y,
                        voltage_level: 'MV',
                        component_type: 'transformer',
                        geom_wkt: $geom_wkt
                    })
                """, **transformer)
            self.stats['nodes_created']['Transformer'] = len(transformers)
            
            # Create LV Cabinets
            for cabinet in lv_cabinets:
                session.run("""
                    CREATE (c:GridComponent:LVCabinet {
                        cabinet_id: $cabinet_id,
                        x: $x,
                        y: $y,
                        voltage_level: 'LV',
                        component_type: 'lv_cabinet',
                        geom_wkt: $geom_wkt
                    })
                """, **cabinet)
            self.stats['nodes_created']['LVCabinet'] = len(lv_cabinets)
            
            # Create Cable Groups
            for group in cable_groups:
                session.run("""
                    CREATE (g:GridComponent:CableGroup {
                        group_id: $group_id,
                        voltage_level: $voltage_level,
                        segment_count: $segment_count,
                        total_length_m: $total_length_m,
                        x: $x,
                        y: $y,
                        bbox_wkt: $bbox_wkt,
                        component_type: 'cable_group'
                    })
                """, **group)
            self.stats['nodes_created']['CableGroup'] = len(cable_groups)
            
            # Create Cable Segments
            segment_batch = []
            for segment in cable_segments:
                segment_batch.append({
                    'segment_id': segment['segment_id'],
                    'original_fid': segment['original_fid'],
                    'voltage_level': segment['voltage_level'],
                    'group_id': segment['group_id'],
                    'length_m': float(segment['length_m']),
                    'start_x': float(segment['start_x']),
                    'start_y': float(segment['start_y']),
                    'end_x': float(segment['end_x']),
                    'end_y': float(segment['end_y'])
                })
                
                if len(segment_batch) >= 1000:
                    session.run("""
                        UNWIND $segments as seg
                        CREATE (s:CableSegment {
                            segment_id: seg.segment_id,
                            original_fid: seg.original_fid,
                            voltage_level: seg.voltage_level,
                            group_id: seg.group_id,
                            length_m: seg.length_m,
                            start_x: seg.start_x,
                            start_y: seg.start_y,
                            end_x: seg.end_x,
                            end_y: seg.end_y
                        })
                    """, segments=segment_batch)
                    segment_batch = []
            
            if segment_batch:
                session.run("""
                    UNWIND $segments as seg
                    CREATE (s:CableSegment {
                        segment_id: seg.segment_id,
                        original_fid: seg.original_fid,
                        voltage_level: seg.voltage_level,
                        group_id: seg.group_id,
                        length_m: seg.length_m,
                        start_x: seg.start_x,
                        start_y: seg.start_y,
                        end_x: seg.end_x,
                        end_y: seg.end_y
                    })
                """, segments=segment_batch)
            
            self.stats['nodes_created']['CableSegment'] = len(cable_segments)
            
            # Create hierarchy relationships
            for rel in hierarchy:
                session.run("""
                    MATCH (child:CableGroup {group_id: $child_group_id})
                    MATCH (parent:CableGroup {group_id: $parent_group_id})
                    CREATE (child)-[:FEEDS_FROM {
                        connection_via: $connection_via,
                        via_station_fid: $via_station_fid,
                        confidence_score: $confidence_score
                    }]->(parent)
                """, **rel)
            self.stats['relationships_created']['FEEDS_FROM'] = len(hierarchy)
            
            # Create group-station relationships
            for conn in group_stations:
                if conn['station_type'] == 'TRANSFORMER':
                    session.run("""
                        MATCH (g:CableGroup {group_id: $group_id})
                        MATCH (t:Transformer {transformer_id: $station_fid})
                        CREATE (g)-[:CONNECTS_TO {
                            connection_type: $connection_type,
                            distance_m: $distance_m,
                            confidence_score: $confidence_score
                        }]->(t)
                    """, **conn)
                elif conn['station_type'] == 'SUBSTATION':
                    session.run("""
                        MATCH (g:CableGroup {group_id: $group_id})
                        MATCH (s:Substation {station_id: $station_fid})
                        CREATE (g)-[:CONNECTS_TO {
                            connection_type: $connection_type,
                            distance_m: $distance_m,
                            confidence_score: $confidence_score
                        }]->(s)
                    """, **conn)
                elif conn['station_type'] == 'LV_CABINET':
                    session.run("""
                        MATCH (g:CableGroup {group_id: $group_id})
                        MATCH (c:LVCabinet {cabinet_id: $station_fid})
                        CREATE (g)-[:CONNECTS_TO {
                            connection_type: $connection_type,
                            distance_m: $distance_m,
                            confidence_score: $confidence_score
                        }]->(c)
                    """, **conn)
            
            # Create segment to group relationships
            session.run("""
                MATCH (s:CableSegment)
                MATCH (g:CableGroup {group_id: s.group_id})
                CREATE (s)-[:PART_OF]->(g)
            """)
            
        logger.info(f"Created {sum(self.stats['nodes_created'].values())} infrastructure nodes")
    
    # ============================================
    # STEP 3: LOAD AND ENHANCE BUILDINGS
    # ============================================
    
    def load_buildings(self):
        """Load buildings with all connection metadata"""
        logger.info("Loading building data from PostgreSQL...")
        
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Load buildings with connection data
            cursor.execute("""
                SELECT 
                    b.ogc_fid,
                    b.x,
                    b.y,
                    b.building_function,
                    b.residential_type,
                    b.non_residential_type,
                    b.area,
                    b.height,
                    b.age_range,
                    b.building_orientation_cardinal,
                    b.meestvoorkomendelabel as energy_label,
                    b.woningtype as housing_type,
                    b.wijknaam as district_name,
                    b.buurtnaam as neighborhood_name,
                    b.bouwjaar as building_year,
                    b.b3_opp_dak_plat as flat_roof_area,
                    b.b3_opp_dak_schuin as sloped_roof_area,
                    -- Connection data
                    bc.connected_group_id as lv_group_id,
                    bc.connected_segment_id,
                    bc.connection_distance_m,
                    bc.connection_type,
                    bc.is_mv_capable,
                    bc.has_mv_nearby,
                    bc.nearest_mv_distance_m,
                    bc.is_problematic,
                    bc.connection_reason,
                    -- Energy indicators
                    b.ndvi_mean_100m,
                    b.ntl_mean_500m,
                    b.ndwi_mean_250m
                FROM amin.buildings_1_deducted b
                LEFT JOIN amin_grid.tlip_building_connections bc
                    ON b.ogc_fid = bc.building_id
                WHERE b.area > 10 
                    AND b.pand_geom IS NOT NULL
                    AND bc.connected_group_id IS NOT NULL
            """)
            buildings = [self.convert_decimals(row) for row in cursor.fetchall()]
            
            # Load connection points
            cursor.execute("""
                SELECT 
                    connection_point_id as point_id,
                    building_id,
                    segment_id,
                    group_id,
                    connection_type,
                    ST_X(point_on_line) as point_x,
                    ST_Y(point_on_line) as point_y,
                    distance_along_segment,
                    segment_fraction,
                    connection_distance_m,
                    is_direct_connection
                FROM amin_grid.tlip_building_connection_points
            """)
            connection_points = [self.convert_decimals(row) for row in cursor.fetchall()]
        
        # Process buildings and add derived features
        buildings_df = pd.DataFrame(buildings)
        buildings_df = self._calculate_building_features(buildings_df)
        
        # Create nodes in Neo4j
        with self.driver.session() as session:
            # Create building nodes with all features
            for _, building in buildings_df.iterrows():
                session.run("""
                    CREATE (b:Building {
                        ogc_fid: $ogc_fid,
                        x: $x,
                        y: $y,
                        building_function: $function,
                        residential_type: $res_type,
                        non_residential_type: $non_res_type,
                        area: $area,
                        height: $height,
                        age_range: $age,
                        building_year: $year,
                        building_orientation_cardinal: $orientation,
                        district_name: $district,
                        neighborhood_name: $neighborhood,
                        housing_type: $housing_type,
                        
                        // Roof data
                        flat_roof_area: $flat_roof,
                        sloped_roof_area: $sloped_roof,
                        suitable_roof_area: $suitable_roof,
                        
                        // Connection data
                        lv_group_id: $lv_group,
                        connection_segment_id: $segment_id,
                        connection_type: $conn_type,
                        connection_distance_m: $conn_distance,
                        is_mv_capable: $mv_capable,
                        has_mv_nearby: $mv_nearby,
                        nearest_mv_distance_m: $mv_distance,
                        is_problematic: $problematic,
                        connection_reason: $conn_reason,
                        
                        // Energy features
                        energy_label: $energy_label,
                        energy_label_simple: $energy_label_simple,
                        insulation_quality: $insulation,
                        solar_potential: $solar_pot,
                        solar_capacity_kwp: $solar_kwp,
                        battery_readiness: $battery_ready,
                        electrification_feasibility: $elec_feasible,
                        expected_cop: $cop,
                        
                        // Environmental indicators
                        vegetation_index: $ndvi,
                        nighttime_lights: $ntl,
                        water_index: $ndwi,
                        
                        // Current assets (initially false/none)
                        has_solar: $has_solar,
                        has_battery: $has_battery,
                        has_heat_pump: $has_hp,
                        heating_system: $heating
                    })
                """,
                ogc_fid=int(building['ogc_fid']),
                x=float(building['x']),
                y=float(building['y']),
                function=building['building_function'],
                res_type=building['residential_type'] if pd.notna(building['residential_type']) else 'None',
                non_res_type=building['non_residential_type'] if pd.notna(building['non_residential_type']) else 'None',
                area=float(building['area']),
                height=float(building['height']) if pd.notna(building['height']) else 3.0,
                age=building['age_range'] if pd.notna(building['age_range']) else 'Unknown',
                year=int(building['building_year']) if pd.notna(building['building_year']) else 0,
                orientation=building['building_orientation_cardinal'] if pd.notna(building['building_orientation_cardinal']) else 'Unknown',
                district=building['district_name'] if pd.notna(building['district_name']) else 'Unknown',
                neighborhood=building['neighborhood_name'] if pd.notna(building['neighborhood_name']) else 'Unknown',
                housing_type=building['housing_type'] if pd.notna(building['housing_type']) else 'Unknown',
                flat_roof=float(building['flat_roof_area']) if pd.notna(building['flat_roof_area']) else 0.0,
                sloped_roof=float(building['sloped_roof_area']) if pd.notna(building['sloped_roof_area']) else 0.0,
                suitable_roof=float(building['suitable_roof_area']),
                lv_group=building['lv_group_id'],
                segment_id=int(building['connected_segment_id']) if pd.notna(building['connected_segment_id']) else 0,
                conn_type=building['connection_type'],
                conn_distance=float(building['connection_distance_m']),
                mv_capable=bool(building['is_mv_capable']),
                mv_nearby=bool(building['has_mv_nearby']),
                mv_distance=float(building['nearest_mv_distance_m']) if pd.notna(building['nearest_mv_distance_m']) else 999.0,
                problematic=bool(building['is_problematic']),
                conn_reason=building['connection_reason'] if pd.notna(building['connection_reason']) else '',
                energy_label=building['energy_label'] if pd.notna(building['energy_label']) else 'Unknown',
                energy_label_simple=building['energy_label_simple'],
                insulation=building['insulation_quality'],
                solar_pot=building['solar_potential'],
                solar_kwp=float(building['solar_capacity_kwp']),
                battery_ready=building['battery_readiness'],
                elec_feasible=building['electrification_feasibility'],
                cop=float(building['expected_cop']),
                ndvi=float(building['ndvi_mean_100m']) if pd.notna(building['ndvi_mean_100m']) else 0.0,
                ntl=float(building['ntl_mean_500m']) if pd.notna(building['ntl_mean_500m']) else 0.0,
                ndwi=float(building['ndwi_mean_250m']) if pd.notna(building['ndwi_mean_250m']) else 0.0,
                has_solar=bool(building['has_solar']),
                has_battery=bool(building['has_battery']),
                has_hp=bool(building['has_heat_pump']),
                heating=building['heating_system']
                )
            
            self.stats['nodes_created']['Building'] = len(buildings_df)
            
            # Create Connection Point nodes
            for point in connection_points:
                session.run("""
                    CREATE (cp:ConnectionPoint {
                        point_id: $point_id,
                        building_id: $building_id,
                        segment_id: $segment_id,
                        group_id: $group_id,
                        connection_type: $conn_type,
                        x: $x,
                        y: $y,
                        distance_along_segment: $dist_along,
                        segment_fraction: $fraction,
                        connection_distance_m: $conn_dist,
                        is_direct: $is_direct
                    })
                """,
                point_id=int(point['point_id']),
                building_id=int(point['building_id']),
                segment_id=int(point['segment_id']),
                group_id=point['group_id'],
                conn_type=point['connection_type'],
                x=float(point['point_x']),
                y=float(point['point_y']),
                dist_along=float(point['distance_along_segment']),
                fraction=float(point['segment_fraction']),
                conn_dist=float(point['connection_distance_m']),
                is_direct=bool(point['is_direct_connection'])
                )
            
            self.stats['nodes_created']['ConnectionPoint'] = len(connection_points)
            
            # Create relationships
            # Building -> LV Cable Group
            result = session.run("""
                MATCH (b:Building)
                MATCH (g:CableGroup {group_id: b.lv_group_id})
                CREATE (b)-[:CONNECTED_TO {
                    connection_type: b.connection_type,
                    distance_m: b.connection_distance_m,
                    is_problematic: b.is_problematic
                }]->(g)
                RETURN count(*) as count
            """)
            self.stats['relationships_created']['CONNECTED_TO'] = result.single()['count']
            
            # Building -> Connection Point
            result = session.run("""
                MATCH (b:Building)
                MATCH (cp:ConnectionPoint {building_id: b.ogc_fid})
                CREATE (b)-[:HAS_CONNECTION_POINT]->(cp)
                RETURN count(*) as count
            """)
            self.stats['relationships_created']['HAS_CONNECTION_POINT'] = result.single()['count']
            
            # Connection Point -> Cable Segment
            result = session.run("""
                MATCH (cp:ConnectionPoint)
                MATCH (s:CableSegment {segment_id: cp.segment_id})
                CREATE (cp)-[:ON_SEGMENT {
                    fraction: cp.segment_fraction,
                    distance_along: cp.distance_along_segment
                }]->(s)
                RETURN count(*) as count
            """)
            self.stats['relationships_created']['ON_SEGMENT'] = result.single()['count']
            
            # Building NEAR_MV for MV-capable buildings
            # Building NEAR_MV for MV-capable buildings
            # Building NEAR_MV for MV-capable buildings
            result = session.run("""
                MATCH (b:Building {is_mv_capable: true, has_mv_nearby: true})
                MATCH (g:CableGroup {voltage_level: 'MV'})
                WHERE point.distance(point({x: b.x, y: b.y}), point({x: g.x, y: g.y})) < 200
                WITH b, g, point.distance(point({x: b.x, y: b.y}), point({x: g.x, y: g.y})) as dist
                ORDER BY dist
                WITH b, head(collect(g)) as nearest_mv, min(dist) as min_dist
                CREATE (b)-[:NEAR_MV {distance_m: min_dist}]->(nearest_mv)
                RETURN count(*) as count
            """)
            self.stats['relationships_created']['NEAR_MV'] = result.single()['count']
            
        logger.info(f"Created {len(buildings_df)} building nodes with {len(connection_points)} connection points")
    
    def _calculate_building_features(self, buildings: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived building features"""
        
        # Simplify energy label
        def simplify_label(label):
            if pd.isna(label):
                return 'Unknown'
            if label in ['A', 'A+', 'A++', 'A+++', 'A++++']:
                return 'A'
            elif label in ['B', 'C', 'D', 'E', 'F', 'G']:
                return label
            else:
                return 'Unknown'
        
        buildings['energy_label_simple'] = buildings['energy_label'].apply(simplify_label)
        
        # If no energy label, estimate from age
        def estimate_label(row):
            if row['energy_label_simple'] != 'Unknown':
                return row['energy_label_simple']
            
            age_labels = {
                '< 1945': 'F',
                '1945-1975': 'E', 
                '1975-1990': 'D',
                '1990-2005': 'C',
                '2005-2015': 'B',
                '> 2015': 'A'
            }
            return age_labels.get(row['age_range'], 'D')
        
        buildings['energy_label_simple'] = buildings.apply(estimate_label, axis=1)
        
        # Insulation quality
        label_insulation = {
            'A': 'excellent', 'B': 'good', 'C': 'fair',
            'D': 'fair', 'E': 'poor', 'F': 'poor', 
            'G': 'very_poor', 'Unknown': 'unknown'
        }
        buildings['insulation_quality'] = buildings['energy_label_simple'].map(label_insulation)
        
        # Calculate suitable roof area
        buildings['suitable_roof_area'] = buildings['flat_roof_area'].fillna(0) + \
            buildings.apply(lambda x: x['sloped_roof_area'] * 0.6 
                          if x['building_orientation_cardinal'] in ['S', 'SE', 'SW'] 
                          else x['sloped_roof_area'] * 0.3 if pd.notna(x['sloped_roof_area']) else 0, axis=1)
        
        # Solar potential
        def get_solar_potential(row):
            if row['suitable_roof_area'] > 100:
                return 'high'
            elif row['suitable_roof_area'] > 50:
                return 'medium'
            elif row['suitable_roof_area'] > 25:
                return 'low'
            else:
                return 'none'
        
        buildings['solar_potential'] = buildings.apply(get_solar_potential, axis=1)
        
        # Solar capacity potential
        # Solar capacity potential - Updated with validated values
        # Power density: 175 Wp/m² (average current technology)
        # System efficiency: 0.85 (includes inverter losses, temperature derating)
        buildings['solar_capacity_kwp'] = buildings['suitable_roof_area'] * 0.175 * 0.85
        
        # Battery readiness (based on solar potential and building type)
        def get_battery_readiness(row):
            if row['solar_potential'] in ['high', 'medium']:
                return 'ready'
            elif row['is_mv_capable'] or row['building_function'] == 'non_residential':
                return 'conditional'
            else:
                return 'not_ready'
        
        buildings['battery_readiness'] = buildings.apply(get_battery_readiness, axis=1)
        
        # Electrification feasibility
        def get_electrification_feasibility(row):
            if row['energy_label_simple'] in ['F', 'G']:
                return 'upgrade_needed'
            elif row['energy_label_simple'] in ['D', 'E']:
                return 'conditional'
            else:
                return 'immediate'
        
        buildings['electrification_feasibility'] = buildings.apply(get_electrification_feasibility, axis=1)
        
        # Expected COP for heat pumps - Validated for Dutch conditions
        # Based on typical leaving water temperatures and insulation levels
        cop_by_label = {
            'A': 4.2,    # Well-insulated, 35°C water temp
            'B': 3.7,    # Good insulation, 40°C water temp
            'C': 3.2,    # Minimum for hybrid, 45°C water temp
            'D': 2.7,    # Requires upgrade, 50°C water temp
            'E': 2.3,    # Poor insulation, 55°C water temp
            'F': 2.0,    # Very poor, needs major upgrade
            'G': 1.7,    # Not suitable without renovation
            'Unknown': 2.5  # Conservative estimate
        }
        buildings['expected_cop'] = buildings['energy_label_simple'].map(cop_by_label)
        
        # Initial asset assignment (most buildings don't have assets yet)
        buildings['has_solar'] = False
        buildings['has_battery'] = False
        buildings['has_heat_pump'] = (buildings['energy_label_simple'].isin(['A', 'B'])) & \
                                     (np.random.random(len(buildings)) > 0.8)
        buildings['heating_system'] = buildings.apply(
            lambda x: 'heat_pump' if x['has_heat_pump'] else 'gas', axis=1
        )
        
        # Updated with actual Dutch market penetration rates (2024)
        # 25-30% of Dutch homes have solar panels
        high_potential_mask = buildings['solar_potential'].isin(['high', 'medium'])
        buildings.loc[high_potential_mask, 'has_solar'] = np.random.random(high_potential_mask.sum()) > 0.72  # ~28% adoption

        # Batteries growing rapidly but still low penetration (~20% of solar homes)
        buildings.loc[buildings['has_solar'], 'has_battery'] = np.random.random(buildings['has_solar'].sum()) > 0.80  # 20% of solar homes

        # Heat pumps still low but concentrated in newer, efficient homes
        buildings['has_heat_pump'] = (buildings['energy_label_simple'].isin(['A', 'B'])) & \
                                    (np.random.random(len(buildings)) > 0.92)  # ~8% of A/B homes


        logger.info(f"Building features calculated: {len(buildings)} buildings processed")
        logger.info(f"  Solar potential - High: {(buildings['solar_potential']=='high').sum()}, "
                   f"Medium: {(buildings['solar_potential']=='medium').sum()}")
        logger.info(f"  Existing installations - Solar: {buildings['has_solar'].sum()}, "
                   f"Battery: {buildings['has_battery'].sum()}, "
                   f"Heat Pump: {buildings['has_heat_pump'].sum()}")
        
        return buildings
    
    # ============================================
    # STEP 4: CREATE EXISTING ASSET NODES
    # ============================================
    
    def create_existing_assets(self):
        """Create asset nodes for buildings that already have installations"""
        logger.info("Creating existing asset nodes...")
        
        with self.driver.session() as session:
            # Existing solar systems
            result = session.run("""
                MATCH (b:Building {has_solar: true})
                CREATE (s:SolarSystem {
                    system_id: 'SOLAR_EXISTING_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'existing',
                    installed_capacity_kwp: b.solar_capacity_kwp,
                    installation_year: 2023,
                    degradation_factor: 0.98,
                    orientation_efficiency: 
                        CASE b.building_orientation_cardinal
                            WHEN 'S' THEN 1.0
                            WHEN 'SE' THEN 0.95
                            WHEN 'SW' THEN 0.95
                            ELSE 0.8
                        END
                })
                CREATE (b)-[:HAS_INSTALLED {
                    install_date: date('2023-01-01')
                }]->(s)
                RETURN count(*) as count
            """)
            solar_count = result.single()['count']
            
            # Existing battery systems
            # Existing battery systems - Updated with validated Dutch market sizing
            result = session.run("""
                MATCH (b:Building {has_battery: true})
                CREATE (bat:BatterySystem {
                    system_id: 'BATTERY_EXISTING_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'existing',
                    installed_capacity_kwh: 
                        CASE 
                            WHEN b.area < 100 THEN 5.0    -- Small homes/apartments
                            WHEN b.area < 150 THEN 7.5    -- Average Dutch home
                            WHEN b.area < 200 THEN 10.0   -- Larger family home
                            WHEN b.area < 300 THEN 13.5   -- Big home (Tesla Powerwall size)
                            ELSE 15.0                     -- Very large homes (capped)
                        END,
                    power_rating_kw: 
                        CASE 
                            WHEN b.area < 100 THEN 2.5    -- 0.5C rate
                            WHEN b.area < 150 THEN 3.5    -- ~0.5C rate
                            WHEN b.area < 200 THEN 5.0    -- 0.5C rate
                            WHEN b.area < 300 THEN 6.5    -- ~0.5C rate
                            ELSE 7.5                      -- 0.5C rate
                        END,
                    round_trip_efficiency: 0.9,
                    estimated_cycles_per_year: 300  -- Daily cycling typical
                })
                CREATE (b)-[:HAS_INSTALLED {
                    install_date: date('2023-01-01')
                }]->(bat)
                RETURN count(*) as count
            """)
            battery_count = result.single()['count']
            
            # Existing heat pump systems
            result = session.run("""
                MATCH (b:Building {has_heat_pump: true})
                CREATE (hp:HeatPumpSystem {
                    system_id: 'HP_EXISTING_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'existing',
                    expected_cop: b.expected_cop,
                    heating_capacity_kw: CASE 
                        WHEN b.age_range = '< 1945' THEN b.area * 0.075
                        WHEN b.age_range = '1945-1975' THEN b.area * 0.065
                        WHEN b.age_range = '1975-1990' THEN b.area * 0.055
                        WHEN b.age_range = '1990-2005' THEN b.area * 0.045
                        WHEN b.age_range = '2005-2015' THEN b.area * 0.035
                        WHEN b.age_range = '> 2015' THEN b.area * 0.025
                        ELSE b.area * 0.050
                    END,
                    installation_year: 2022
                })
                CREATE (b)-[:HAS_INSTALLED {
                    install_date: date('2022-01-01')
                }]->(hp)
                RETURN count(*) as count
            """)
            hp_count = result.single()['count']
            
            self.stats['nodes_created']['ExistingSolar'] = solar_count
            self.stats['nodes_created']['ExistingBattery'] = battery_count  
            self.stats['nodes_created']['ExistingHeatPump'] = hp_count
            self.stats['relationships_created']['HAS_INSTALLED'] = solar_count + battery_count + hp_count
            
        logger.info(f"Created existing assets: {solar_count} solar, {battery_count} battery, {hp_count} heat pumps")
    
    # ============================================
    # STEP 5: LOAD TEMPORAL ENERGY DATA (UNCHANGED)
    # ============================================
    
    def load_temporal_data(self, data_path: str):
        """Load time-series energy profiles from parquet"""
        logger.info("Loading temporal energy data...")
        
        # Load energy profiles from parquet
        profiles = pd.read_parquet(f"{data_path}/energy_profiles.parquet")
        
        # Get unique timestamps
        timestamps = profiles['timestamp'].unique()
        
        with self.driver.session() as session:
            # Create TimeSlot nodes
            for i, ts in enumerate(timestamps):
                dt = pd.to_datetime(ts)
                session.run("""
                    CREATE (t:TimeSlot {
                        slot_id: $slot_id,
                        timestamp: datetime($timestamp),
                        hour_of_day: $hour,
                        day_of_week: $dow,
                        is_weekend: $weekend,
                        season: $season,
                        time_of_day: $tod
                    })
                """,
                slot_id=f"TS_{i}",
                timestamp=dt.isoformat(),
                hour=dt.hour,
                dow=dt.dayofweek,
                weekend=dt.dayofweek >= 5,
                season=self._get_season(dt),
                tod=self._get_time_of_day(dt.hour)
                )
            
            self.stats['nodes_created']['TimeSlot'] = len(timestamps)
            
            # Create EnergyState nodes in batches
            logger.info("Creating energy states (this may take a moment)...")
            
            batch_size = 1000
            state_count = 0
            
            for building_id in profiles['building_id'].unique():
                building_data = profiles[profiles['building_id'] == building_id]
                
                states = []
                for idx, row in building_data.iterrows():
                    dt = pd.to_datetime(row['timestamp'])
                    slot_id = f"TS_{list(timestamps).index(row['timestamp'])}"
                    
                    # Calculate net position
                    net_demand = row['electricity_demand_kw'] - row['solar_generation_kw'] + \
                                row['battery_discharge_kw'] - row['battery_charge_kw']
                    
                    states.append({
                        'state_id': f"ES_{building_id}_{slot_id}",
                        'building_id': int(building_id),
                        'timeslot_id': slot_id,
                        'electricity_demand_kw': float(row['electricity_demand_kw']),
                        'heating_demand_kw': float(row['heating_demand_kw']),
                        'cooling_demand_kw': float(row['cooling_demand_kw']),
                        'solar_generation_kw': float(row['solar_generation_kw']),
                        'battery_soc_kwh': float(row['battery_soc_kwh']),
                        'battery_charge_kw': float(row['battery_charge_kw']),
                        'battery_discharge_kw': float(row['battery_discharge_kw']),
                        'net_demand_kw': float(net_demand),
                        'is_surplus': net_demand < 0,
                        'export_potential_kw': max(0, -net_demand),
                        'import_need_kw': max(0, net_demand)
                    })
                    
                    if len(states) >= batch_size:
                        self._create_energy_states_batch(session, states)
                        state_count += len(states)
                        states = []
                
                if states:
                    self._create_energy_states_batch(session, states)
                    state_count += len(states)
            
            self.stats['nodes_created']['EnergyState'] = state_count
            
            # Create relationships
            logger.info("Creating temporal relationships...")
            
            # Building -> EnergyState
            result = session.run("""
                MATCH (b:Building), (e:EnergyState {building_id: b.ogc_fid})
                CREATE (b)-[:HAS_STATE_AT]->(e)
                RETURN count(*) as count
            """)
            self.stats['relationships_created']['HAS_STATE_AT'] = result.single()['count']
            
            # EnergyState -> TimeSlot
            result = session.run("""
                MATCH (e:EnergyState), (t:TimeSlot {slot_id: e.timeslot_id})
                CREATE (e)-[:DURING]->(t)
                RETURN count(*) as count
            """)
            self.stats['relationships_created']['DURING'] = result.single()['count']
            
        logger.info(f"Created {state_count} energy states with temporal relationships")
    
    def _create_energy_states_batch(self, session, states):
        """Create energy state nodes in batch"""
        session.run("""
            UNWIND $states as state
            CREATE (e:EnergyState {
                state_id: state.state_id,
                building_id: state.building_id,
                timeslot_id: state.timeslot_id,
                electricity_demand_kw: state.electricity_demand_kw,
                heating_demand_kw: state.heating_demand_kw,
                cooling_demand_kw: state.cooling_demand_kw,
                solar_generation_kw: state.solar_generation_kw,
                battery_soc_kwh: state.battery_soc_kwh,
                battery_charge_kw: state.battery_charge_kw,
                battery_discharge_kw: state.battery_discharge_kw,
                net_demand_kw: state.net_demand_kw,
                is_surplus: state.is_surplus,
                export_potential_kw: state.export_potential_kw,
                import_need_kw: state.import_need_kw
            })
        """, states=states)
    
    def _get_season(self, dt):
        """Get season from datetime"""
        month = dt.month
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'fall'
    
    def _get_time_of_day(self, hour):
        """Categorize time of day"""
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'
    
    # ============================================
    # STEP 6: IDENTIFY ASSET OPPORTUNITIES
    # ============================================
    
    def identify_asset_opportunities(self):
        """Create nodes for potential solar, battery, and electrification"""
        logger.info("Identifying asset deployment opportunities...")
        
        with self.driver.session() as session:
            # Solar opportunities (for buildings without solar)
            result = session.run("""
                MATCH (b:Building)
                WHERE b.solar_potential IN ['high', 'medium']
                  AND b.has_solar = false
                CREATE (s:SolarSystem {
                    system_id: 'SOLAR_POTENTIAL_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'potential',
                    potential_capacity_kwp: b.solar_capacity_kwp,
                    recommended_capacity_kwp: 
                        CASE 
                            WHEN b.area < 150 THEN 4.0
                            WHEN b.area < 300 THEN 6.0
                            ELSE 10.0
                        END,
                    orientation_efficiency: 
                        CASE b.building_orientation_cardinal
                            WHEN 'S' THEN 1.0
                            WHEN 'SE' THEN 0.95
                            WHEN 'SW' THEN 0.95
                            WHEN 'E' THEN 0.85
                            WHEN 'W' THEN 0.85
                            ELSE 0.7
                        END,
                    payback_years: 8.5
                })
                CREATE (b)-[:CAN_INSTALL {
                    feasibility_score: 
                        CASE b.solar_potential
                            WHEN 'high' THEN 0.9
                            WHEN 'medium' THEN 0.7
                            ELSE 0.5
                        END,
                    priority: 'medium'
                }]->(s)
                RETURN count(*) as count
            """)
            
            solar_count = result.single()['count']
            self.stats['nodes_created']['SolarPotential'] = solar_count
            
            # Battery opportunities
            result = session.run("""
                MATCH (b:Building)
                WHERE (b.solar_potential IN ['high', 'medium'] OR b.building_function = 'non_residential')
                  AND b.has_battery = false
                WITH b, 
                    CASE 
                        -- Based on Dutch average: 16 kWh daily, 70% night usage
                        -- Battery = daily_consumption * 0.7 * 1.2 (20% reserve)
                        WHEN b.area < 100 THEN 5.0   -- Small homes
                        WHEN b.area < 150 THEN 7.5   -- Average homes
                        WHEN b.area < 200 THEN 10.0  -- Larger homes
                        WHEN b.area < 300 THEN 13.5  -- Big homes (like Tesla Powerwall)
                        ELSE 15.0  -- Cap at 15 kWh for residential
                    END as battery_size
                CREATE (bat:BatterySystem {
                    system_id: 'BATTERY_POTENTIAL_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'potential',
                    recommended_capacity_kwh: battery_size,
                    power_rating_kw: battery_size / 4,
                    round_trip_efficiency: 0.9,
                    estimated_cycles_per_year: 250
                })
                CREATE (b)-[:CAN_INSTALL {
                    feasibility_score: 
                        CASE 
                            WHEN b.has_solar OR b.solar_potential IN ['high', 'medium'] THEN 0.8
                            ELSE 0.6
                        END,
                    requires_solar: NOT b.has_solar,
                    priority: 'low'
                }]->(bat)
                RETURN count(*) as count
            """)
            
            battery_count = result.single()['count']
            self.stats['nodes_created']['BatteryPotential'] = battery_count
            
            # Electrification opportunities (heat pumps)
            result = session.run("""
                MATCH (b:Building)
                WHERE b.energy_label_simple IN ['D', 'E', 'F', 'G']
                  AND b.has_heat_pump = false
                CREATE (hp:HeatPumpSystem {
                    system_id: 'HP_POTENTIAL_' + toString(b.ogc_fid),
                    building_id: b.ogc_fid,
                    status: 'potential',
                    expected_cop: b.expected_cop,
                    heating_capacity_kw: CASE 
                        WHEN b.age_range = '< 1945' THEN b.area * 0.075
                        WHEN b.age_range = '1945-1975' THEN b.area * 0.065
                        WHEN b.age_range = '1975-1990' THEN b.area * 0.055
                        WHEN b.age_range = '1990-2005' THEN b.area * 0.045
                        WHEN b.age_range = '2005-2015' THEN b.area * 0.035
                        WHEN b.age_range = '> 2015' THEN b.area * 0.025
                        ELSE b.area * 0.050
                    END,
                    upgrade_required: b.electrification_feasibility = 'upgrade_needed',
                    estimated_annual_savings_euro: 
                        CASE b.energy_label_simple
                            WHEN 'G' THEN 2000
                            WHEN 'F' THEN 1500
                            WHEN 'E' THEN 1000
                            ELSE 500
                        END
                })
                CREATE (b)-[:SHOULD_ELECTRIFY {
                    priority: 
                        CASE b.energy_label_simple
                            WHEN 'G' THEN 1
                            WHEN 'F' THEN 2
                            WHEN 'E' THEN 3
                            WHEN 'D' THEN 4
                            ELSE 5
                        END,
                    expected_cop: b.expected_cop,
                    requires_insulation_upgrade: b.energy_label_simple IN ['F', 'G']
                }]->(hp)
                RETURN count(*) as count
            """)
            
            hp_count = result.single()['count']
            self.stats['nodes_created']['HeatPumpPotential'] = hp_count
            
        logger.info(f"Identified opportunities: {solar_count} solar, {battery_count} battery, {hp_count} heat pumps")
    
    # ============================================
    # STEP 7: CALCULATE BASELINE METRICS
    # ============================================
    
    def calculate_baseline_metrics(self):
        """Calculate current state metrics for comparison after GNN optimization"""
        logger.info("Calculating baseline metrics...")
        
        with self.driver.session() as session:
            # Building-level statistics
            session.run("""
                MATCH (b:Building)-[:HAS_STATE_AT]->(e:EnergyState)
                WITH b, 
                    max(e.electricity_demand_kw) as peak_demand,
                    avg(e.electricity_demand_kw) as avg_demand,
                    stdev(e.electricity_demand_kw) as demand_std,
                    max(e.solar_generation_kw) as peak_solar,
                    avg(e.net_demand_kw) as avg_net_demand
                SET b.peak_demand_kw = peak_demand,
                    b.avg_demand_kw = avg_demand,
                    b.load_factor = avg_demand / CASE WHEN peak_demand > 0 THEN peak_demand ELSE 1 END,
                    b.demand_variability = demand_std / CASE WHEN avg_demand > 0 THEN avg_demand ELSE 1 END,
                    b.peak_solar_kw = peak_solar,
                    b.avg_net_demand_kw = avg_net_demand,
                    b.self_consumption_ratio = 
                        CASE WHEN b.has_solar 
                        THEN (avg_demand - avg_net_demand) / CASE WHEN avg_demand > 0 THEN avg_demand ELSE 1 END
                        ELSE 0 END
            """)
            
            # LV Cable Group baseline statistics
            session.run("""
                MATCH (g:CableGroup {voltage_level: 'LV'})<-[:CONNECTED_TO]-(b:Building)
                WITH g, 
                    count(b) as building_count,
                    sum(b.peak_demand_kw) as total_peak_demand,
                    avg(b.peak_demand_kw) as avg_building_peak,
                    sum(b.avg_demand_kw) as total_avg_demand,
                    sum(CASE WHEN b.has_solar THEN 1 ELSE 0 END) as solar_count,
                    sum(CASE WHEN b.has_battery THEN 1 ELSE 0 END) as battery_count,
                    sum(CASE WHEN b.has_heat_pump THEN 1 ELSE 0 END) as hp_count,
                    collect(DISTINCT b.building_function) as building_types
                SET g.baseline_building_count = building_count,
                    g.baseline_peak_kw = total_peak_demand,
                    g.baseline_avg_demand_kw = total_avg_demand,
                    g.baseline_load_factor = total_avg_demand / CASE WHEN total_peak_demand > 0 THEN total_peak_demand ELSE 1 END,
                    g.baseline_solar_penetration = toFloat(solar_count) / building_count,
                    g.baseline_battery_penetration = toFloat(battery_count) / building_count,
                    g.baseline_hp_penetration = toFloat(hp_count) / building_count,
                    g.baseline_diversity = size(building_types)
            """)
            
            # Transformer baseline
            session.run("""
                MATCH (t:Transformer)<-[:CONNECTS_TO]-(g:CableGroup {voltage_level: 'LV'})
                WITH t, 
                    count(g) as lv_count,
                    sum(g.baseline_peak_kw) as total_peak,
                    sum(g.baseline_building_count) as total_buildings
                SET t.baseline_lv_count = lv_count,
                    t.baseline_peak_kw = total_peak,
                    t.baseline_building_count = total_buildings
            """)
            
            # System-wide baseline
            result = session.run("""
                MATCH (b:Building)
                WITH count(b) as total_buildings,
                    sum(b.peak_demand_kw) as system_peak,
                    avg(b.load_factor) as avg_load_factor,
                    sum(CASE WHEN b.has_solar THEN 1 ELSE 0 END) as solar_buildings,
                    sum(CASE WHEN b.has_battery THEN 1 ELSE 0 END) as battery_buildings,
                    sum(CASE WHEN b.has_heat_pump THEN 1 ELSE 0 END) as hp_buildings
                CREATE (s:SystemBaseline {
                    id: 'BASELINE_' + toString(datetime()),
                    timestamp: datetime(),
                    total_buildings: total_buildings,
                    system_peak_kw: system_peak,
                    avg_load_factor: avg_load_factor,
                    solar_penetration: toFloat(solar_buildings) / total_buildings,
                    battery_penetration: toFloat(battery_buildings) / total_buildings,
                    hp_penetration: toFloat(hp_buildings) / total_buildings,
                    description: 'Pre-GNN optimization baseline'
                })
                RETURN total_buildings, system_peak, avg_load_factor
            """)
            
            baseline = result.single()
            if baseline:
                total_buildings = baseline['total_buildings'] or 0
                system_peak = baseline['system_peak'] or 0
                avg_load_factor = baseline['avg_load_factor'] or 0
                
                logger.info(f"Baseline: {total_buildings} buildings, "
                        f"{system_peak:.0f} kW peak, "
                        f"{avg_load_factor:.3f} load factor")
            
    # ============================================
    # STEP 8: ADD METADATA
    # ============================================
    
    def add_metadata(self):
        """Add metadata about the KG creation"""
        logger.info("Adding metadata...")
        
        with self.driver.session() as session:
            metadata = {
                'creation_timestamp': datetime.now().isoformat(),
                'total_nodes': sum(self.stats['nodes_created'].values()),
                'total_relationships': sum(self.stats['relationships_created'].values()),
                'node_types': self.stats['nodes_created'],
                'relationship_types': self.stats['relationships_created']
            }
            
            session.run("""
                CREATE (m:Metadata {
                    id: 'PRE_GNN_METADATA',
                    created_at: datetime($timestamp),
                    total_nodes: $nodes,
                    total_relationships: $rels,
                    node_breakdown: $node_types,
                    relationship_breakdown: $rel_types,
                    data_source: 'PostgreSQL',
                    version: '2.0',
                    stage: 'pre_gnn',
                    description: 'Knowledge graph with rich grid infrastructure before GNN optimization'
                })
            """,
            timestamp=metadata['creation_timestamp'],
            nodes=metadata['total_nodes'],
            rels=metadata['total_relationships'],
            node_types=json.dumps(metadata['node_types']),
            rel_types=json.dumps(metadata['relationship_types'])
            )
            
        logger.info("Metadata added")
    
    # ============================================
    # MAIN EXECUTION METHOD
    # ============================================
    
    def build_complete_graph(self, data_path: str, clear_first: bool = True):
        """Build complete knowledge graph from PostgreSQL database (Pre-GNN version)"""
        
        start_time = datetime.now()
        logger.info("="*50)
        logger.info("Starting Knowledge Graph Construction (Pre-GNN)")
        logger.info("="*50)
        
        try:
            # Clear database if requested
            if clear_first:
                self.clear_database()
            
            # Build graph step by step
            self.create_schema()
            self.load_grid_infrastructure()  # From PostgreSQL
            self.load_buildings()  # From PostgreSQL
            self.create_existing_assets()
            self.load_temporal_data(data_path)  # Still from parquet
            self.identify_asset_opportunities()
            self.calculate_baseline_metrics()
            self.add_metadata()
            
            # Calculate total time
            end_time = datetime.now()
            total_time = (end_time - start_time).total_seconds()
            
            # Print summary
            logger.info("="*50)
            logger.info("Pre-GNN Knowledge Graph Construction Complete!")
            logger.info("="*50)
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info("\nNodes created:")
            for node_type, count in self.stats['nodes_created'].items():
                logger.info(f"  {node_type}: {count:,}")
            logger.info(f"  TOTAL: {sum(self.stats['nodes_created'].values()):,}")
            
            logger.info("\nRelationships created:")
            for rel_type, count in self.stats['relationships_created'].items():
                logger.info(f"  {rel_type}: {count:,}")
            logger.info(f"  TOTAL: {sum(self.stats['relationships_created'].values()):,}")
            
            logger.info("\n⚠️ Note: Complementarity analysis and clustering will be added after GNN processing")
            
            return self.stats
            
        except Exception as e:
            logger.error(f"Error building knowledge graph: {str(e)}")
            raise

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "aminasad"  # Change this to your password
    
    # PostgreSQL configuration
    PG_HOST = "localhost"
    PG_DATABASE = ""  # Your database name
    PG_USER = ""
    PG_PASSWORD = "!"  # Change this

    DATA_PATH = ""  # For parquet files
    
    # Create builder and construct graph
    builder = EnergyKnowledgeGraphBuilder(
        NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
        PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD
    )
    
    try:
        # Build complete knowledge graph
        stats = builder.build_complete_graph(DATA_PATH, clear_first=True)
        
        # Run validation queries
        with builder.driver.session() as session:
            # Check grid infrastructure
            result = session.run("""
                MATCH (n:GridComponent)
                RETURN n.component_type as type, count(*) as count
                ORDER BY count DESC
            """)
            
            print("\n" + "="*50)
            print("Grid Infrastructure Summary:")
            print("="*50)
            for record in result:
                print(f"{record['type']}: {record['count']} components")
            
            # Check LV groups with buildings
            result = session.run("""
                MATCH (g:CableGroup {voltage_level: 'LV'})
                OPTIONAL MATCH (g)<-[:CONNECTED_TO]-(b:Building)
                WITH g, count(b) as buildings
                RETURN g.group_id as group_id,
                       buildings,
                       g.baseline_peak_kw as peak_kw,
                       g.baseline_solar_penetration as solar_pen
                ORDER BY buildings DESC
                LIMIT 10
            """)
            
            print("\n" + "="*50)
            print("Top 10 LV Groups by Building Count:")
            print("="*50)
            for record in result:
                print(f"{record['group_id']}: {record['buildings']} buildings, "
                      f"Peak: {record['peak_kw']:.0f} kW, "
                      f"Solar: {record['solar_pen']:.1%}" if record['peak_kw'] else "No metrics yet")
            
            # Check building connection quality
            result = session.run("""
                MATCH (b:Building)
                RETURN b.connection_type as type, 
                       count(*) as count,
                       avg(b.connection_distance_m) as avg_distance
                ORDER BY count DESC
            """)
            
            print("\n" + "="*50)
            print("Building Connection Quality:")
            print("="*50)
            for record in result:
                print(f"{record['type']}: {record['count']} buildings, "
                      f"Avg distance: {record['avg_distance']:.1f}m")
            
            # Check opportunities
            result = session.run("""
                OPTIONAL MATCH (s:SolarSystem {status: 'potential'})
                WITH count(s) as solar_opp
                OPTIONAL MATCH (b:BatterySystem {status: 'potential'})
                WITH solar_opp, count(b) as battery_opp
                OPTIONAL MATCH (h:HeatPumpSystem {status: 'potential'})
                RETURN solar_opp, battery_opp, count(h) as hp_opp
            """)
            
            opp = result.single()
            print("\n" + "="*50)
            print("Deployment Opportunities Identified:")
            print("="*50)
            if opp:
                print(f"Solar: {opp['solar_opp']} buildings")
                print(f"Battery: {opp['battery_opp']} buildings")
                print(f"Heat Pump: {opp['hp_opp']} buildings")
        
    except Exception as e:
        print(f"Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        builder.close()
    
    print("\n✅ Pre-GNN Knowledge Graph construction complete!")
    print("Ready for GNN processing to discover complementarity and optimal clustering")
    print("You can explore the graph in Neo4j Browser at http://localhost:7474")
    
    """
    # After GNN runs, you'll add:
    1. COMPLEMENTS relationships (discovered by GNN)
    2. EnergyCluster nodes (optimized groupings)
    3. TRADES_ENERGY_WITH relationships
    4. DeploymentScenario nodes (GNN recommendations)
    5. Performance metrics comparing to baseline
    """