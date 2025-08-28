"""
Adjacency Module for Energy KG - Part 2
Updates existing Knowledge Graph with adjacency relationships
Uses PostgreSQL database (same as Part 1) instead of CSV
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Optional
from decimal import Decimal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdjacencyUpdater:
    """Add adjacency relationships to existing Knowledge Graph using PostgreSQL data"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str,
                 pg_host: str, pg_database: str, pg_user: str, pg_password: str):
        """Initialize Neo4j and PostgreSQL connections (same as Part 1)"""
        # Neo4j connection
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        # PostgreSQL connection (same as Part 1)
        self.pg_conn = psycopg2.connect(
            host=pg_host,
            database=pg_database,
            user=pg_user,
            password=pg_password,
            port=5433
        )
        
        self.stats = {
            'relationships_created': 0,
            'clusters_created': 0,
            'nodes_updated': 0,
            'validation_results': {}
        }
        logger.info(f"Connected to Neo4j and PostgreSQL for adjacency update")
    
    def close(self):
        """Close database connections"""
        self.driver.close()
        self.pg_conn.close()
        logger.info("Connections closed")
    
    def convert_decimals(self, record):
        """Convert Decimal types to float for Neo4j compatibility (from Part 1)"""
        if isinstance(record, dict):
            return {k: float(v) if isinstance(v, Decimal) else v for k, v in record.items()}
        elif isinstance(record, list):
            return [self.convert_decimals(item) for item in record]
        else:
            return float(record) if isinstance(record, Decimal) else record
    
    def safe_float(self, value, default=0.0) -> float:
        """Safely convert value to float, handling None and other edge cases"""
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    def safe_int(self, value, default=0) -> int:
        """Safely convert value to int, handling None and other edge cases"""
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    
    def check_kg_status(self) -> Dict:
        """Check if KG exists and has required data"""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (b:Building)
                RETURN 
                    count(b) as building_count,
                    count(b.north_shared_length) as has_shared_wall_data,
                    exists((b)-[:ADJACENT_TO]-()) as has_adjacencies
                LIMIT 1
            """)
            
            status = result.single()
            
            return {
                'buildings_exist': status['building_count'] > 0,
                'has_shared_walls': status['has_shared_wall_data'] > 0,
                'adjacencies_exist': status['has_adjacencies'],
                'building_count': status['building_count']
            }
    
    def update_shared_wall_data_from_postgres(self):
        """Update existing Neo4j buildings with shared wall data from PostgreSQL"""
        logger.info("Updating buildings with shared wall data from PostgreSQL...")
        
        with self.pg_conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Query buildings with shared wall data from PostgreSQL
            cursor.execute("""
                SELECT 
                    b.ogc_fid,
                    -- Shared wall lengths (handle NULLs with COALESCE)
                    COALESCE(b.north_shared_length, 0) as north_shared_length,
                    COALESCE(b.south_shared_length, 0) as south_shared_length,
                    COALESCE(b.east_shared_length, 0) as east_shared_length,
                    COALESCE(b.west_shared_length, 0) as west_shared_length,
                    -- Facade lengths (handle NULLs with COALESCE)
                    COALESCE(b.north_facade_length, 10) as north_facade_length,
                    COALESCE(b.south_facade_length, 10) as south_facade_length,
                    COALESCE(b.east_facade_length, 10) as east_facade_length,
                    COALESCE(b.west_facade_length, 10) as west_facade_length,
                    -- Calculate derived fields
                    (CASE WHEN b.north_shared_length > 0 THEN 1 ELSE 0 END +
                     CASE WHEN b.south_shared_length > 0 THEN 1 ELSE 0 END +
                     CASE WHEN b.east_shared_length > 0 THEN 1 ELSE 0 END +
                     CASE WHEN b.west_shared_length > 0 THEN 1 ELSE 0 END) as num_shared_walls,
                    (COALESCE(b.north_shared_length, 0) + 
                     COALESCE(b.south_shared_length, 0) + 
                     COALESCE(b.east_shared_length, 0) + 
                     COALESCE(b.west_shared_length, 0)) as total_shared_length,
                    -- Adjacency type classification
                    CASE 
                        WHEN (CASE WHEN b.north_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.south_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.east_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.west_shared_length > 0 THEN 1 ELSE 0 END) = 0 
                        THEN 'ISOLATED'
                        
                        WHEN (b.north_shared_length > 0 AND b.south_shared_length > 0) OR
                             (b.east_shared_length > 0 AND b.west_shared_length > 0)
                        THEN 'MIDDLE_ROW'
                        
                        WHEN (CASE WHEN b.north_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.south_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.east_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.west_shared_length > 0 THEN 1 ELSE 0 END) = 2
                        THEN 'CORNER'
                        
                        WHEN (CASE WHEN b.north_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.south_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.east_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.west_shared_length > 0 THEN 1 ELSE 0 END) = 1
                        THEN 'END_ROW'
                        
                        WHEN (CASE WHEN b.north_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.south_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.east_shared_length > 0 THEN 1 ELSE 0 END +
                              CASE WHEN b.west_shared_length > 0 THEN 1 ELSE 0 END) >= 3
                        THEN 'COURTYARD'
                        
                        ELSE 'UNKNOWN'
                    END as adjacency_type,
                    -- Additional context
                    b.woningtype,
                    b.area
                FROM amin.buildings_1_deducted b
                JOIN amin_grid.tlip_building_connections bc 
                    ON b.ogc_fid = bc.building_id
                WHERE b.area > 10 
                    AND b.pand_geom IS NOT NULL
                    AND bc.connected_group_id IS NOT NULL
            """)
            
            buildings = [self.convert_decimals(row) for row in cursor.fetchall()]
        
        # Update Neo4j buildings with shared wall data
        with self.driver.session() as session:
            updated = 0
            batch_size = 500
            batch = []
            
            for building in buildings:
                # Use safe conversion functions to handle None values
                batch.append({
                    'ogc_fid': self.safe_int(building['ogc_fid']),
                    'north_shared': self.safe_float(building['north_shared_length'], 0),
                    'south_shared': self.safe_float(building['south_shared_length'], 0),
                    'east_shared': self.safe_float(building['east_shared_length'], 0),
                    'west_shared': self.safe_float(building['west_shared_length'], 0),
                    'north_facade': self.safe_float(building['north_facade_length'], 10),
                    'south_facade': self.safe_float(building['south_facade_length'], 10),
                    'east_facade': self.safe_float(building['east_facade_length'], 10),
                    'west_facade': self.safe_float(building['west_facade_length'], 10),
                    'num_shared': self.safe_int(building['num_shared_walls'], 0),
                    'total_shared': self.safe_float(building['total_shared_length'], 0),
                    'adj_type': building.get('adjacency_type', 'UNKNOWN') or 'UNKNOWN',
                    'housing_type': building.get('woningtype', 'unknown') or 'unknown'
                })
                
                if len(batch) >= batch_size:
                    result = session.run("""
                        UNWIND $batch as building
                        MATCH (b:Building {ogc_fid: building.ogc_fid})
                        SET b.north_shared_length = building.north_shared,
                            b.south_shared_length = building.south_shared,
                            b.east_shared_length = building.east_shared,
                            b.west_shared_length = building.west_shared,
                            b.north_facade_length = building.north_facade,
                            b.south_facade_length = building.south_facade,
                            b.east_facade_length = building.east_facade,
                            b.west_facade_length = building.west_facade,
                            b.num_shared_walls = building.num_shared,
                            b.total_shared_length = building.total_shared,
                            b.adjacency_type = building.adj_type,
                            b.woningtype = building.housing_type
                        RETURN count(b) as updated_count
                    """, batch=batch)
                    
                    single_result = result.single()
                    if single_result:
                        updated += single_result['updated_count']
                    batch = []
            
            # Process remaining batch
            if batch:
                result = session.run("""
                    UNWIND $batch as building
                    MATCH (b:Building {ogc_fid: building.ogc_fid})
                    SET b.north_shared_length = building.north_shared,
                        b.south_shared_length = building.south_shared,
                        b.east_shared_length = building.east_shared,
                        b.west_shared_length = building.west_shared,
                        b.north_facade_length = building.north_facade,
                        b.south_facade_length = building.south_facade,
                        b.east_facade_length = building.east_facade,
                        b.west_facade_length = building.west_facade,
                        b.num_shared_walls = building.num_shared,
                        b.total_shared_length = building.total_shared,
                        b.adjacency_type = building.adj_type,
                        b.woningtype = building.housing_type
                    RETURN count(b) as updated_count
                """, batch=batch)
                
                single_result = result.single()
                if single_result:
                    updated += single_result['updated_count']
            
            self.stats['nodes_updated'] = updated
            logger.info(f"Updated {updated} buildings with shared wall data from PostgreSQL")
    
    def create_adjacency_relationships(self):
        """Find and create ADJACENT_TO relationships between buildings"""
        logger.info("Creating adjacency relationships...")
        
        with self.driver.session() as session:
            # Clear any existing adjacency relationships first
            session.run("MATCH ()-[r:ADJACENT_TO]-() DELETE r")
            
            # Create relationships using spatial proximity and shared walls
            result = session.run("""
                // Find all buildings with shared walls
                MATCH (b1:Building)
                WHERE b1.num_shared_walls > 0
                
                // Find potential neighbors in same LV group
                MATCH (b2:Building)
                WHERE b1.ogc_fid < b2.ogc_fid  // Process each pair only once
                AND b1.lv_group_id = b2.lv_group_id  // Same LV network
                AND b2.num_shared_walls > 0
                
                // Calculate distance (must be very close for true adjacency)
                WITH b1, b2, 
                    sqrt((b2.x - b1.x)^2 + (b2.y - b1.y)^2) as distance
                WHERE distance < 5  // Reduced to 5m for true adjacency
                
                // Check for matching shared walls
                WITH b1, b2, distance,
                    CASE 
                        // North-South match
                        WHEN b1.north_shared_length > 0 AND b2.south_shared_length > 0 
                            AND b2.y > b1.y AND abs(b2.x - b1.x) < 3
                        THEN {
                            pair: 'north-south',
                            b1_wall: 'north',
                            b2_wall: 'south',
                            b1_length: b1.north_shared_length,
                            b2_length: b2.south_shared_length,
                            match_quality: 
                                CASE 
                                    WHEN b1.north_shared_length < b2.south_shared_length 
                                    THEN b1.north_shared_length / b2.south_shared_length
                                    ELSE b2.south_shared_length / b1.north_shared_length
                                END
                        }
                        // South-North match
                        WHEN b1.south_shared_length > 0 AND b2.north_shared_length > 0
                            AND b2.y < b1.y AND abs(b2.x - b1.x) < 3
                        THEN {
                            pair: 'south-north',
                            b1_wall: 'south',
                            b2_wall: 'north',
                            b1_length: b1.south_shared_length,
                            b2_length: b2.north_shared_length,
                            match_quality: 
                                CASE 
                                    WHEN b1.south_shared_length < b2.north_shared_length 
                                    THEN b1.south_shared_length / b2.north_shared_length
                                    ELSE b2.north_shared_length / b1.south_shared_length
                                END
                        }
                        // East-West match
                        WHEN b1.east_shared_length > 0 AND b2.west_shared_length > 0
                            AND b2.x > b1.x AND abs(b2.y - b1.y) < 3
                        THEN {
                            pair: 'east-west',
                            b1_wall: 'east',
                            b2_wall: 'west',
                            b1_length: b1.east_shared_length,
                            b2_length: b2.west_shared_length,
                            match_quality: 
                                CASE 
                                    WHEN b1.east_shared_length < b2.west_shared_length 
                                    THEN b1.east_shared_length / b2.west_shared_length
                                    ELSE b2.west_shared_length / b1.east_shared_length
                                END
                        }
                        // West-East match
                        WHEN b1.west_shared_length > 0 AND b2.east_shared_length > 0
                            AND b2.x < b1.x AND abs(b2.y - b1.y) < 3
                        THEN {
                            pair: 'west-east',
                            b1_wall: 'west',
                            b2_wall: 'east',
                            b1_length: b1.west_shared_length,
                            b2_length: b2.east_shared_length,
                            match_quality: 
                                CASE 
                                    WHEN b1.west_shared_length < b2.east_shared_length 
                                    THEN b1.west_shared_length / b2.east_shared_length
                                    ELSE b2.east_shared_length / b1.west_shared_length
                                END
                        }
                        ELSE null
                    END as wall_match
                
                WHERE wall_match IS NOT NULL
                AND wall_match.match_quality > 0.5  // Walls must be similar length
                
                // Create bidirectional relationships
                CREATE (b1)-[r1:ADJACENT_TO {
                    wall_pair: wall_match.pair,
                    my_wall: wall_match.b1_wall,
                    their_wall: wall_match.b2_wall,
                    my_shared_length: wall_match.b1_length,
                    their_shared_length: wall_match.b2_length,
                    match_quality: wall_match.match_quality,
                    distance_m: distance,
                    adjacency_strength: (wall_match.b1_length + wall_match.b2_length) / 2,
                    
                    // Energy implications
                    thermal_coupling: true,
                    cable_distance: distance * 0.5,
                    transmission_loss_factor: 0.001 * distance,  -- 0.1% per meter for proper LV cabling
                    energy_sharing_viable: distance < 3,
                    
                    // Complementarity based on function
                    function_diversity: CASE 
                        WHEN b1.building_function <> b2.building_function THEN 2.0
                        WHEN b1.residential_type <> b2.residential_type THEN 1.5
                        ELSE 1.0
                    END,
                    
                    // Solar diversity
                    solar_diversity: CASE
                        WHEN b1.has_solar <> b2.has_solar THEN 2.0
                        WHEN b1.solar_potential <> b2.solar_potential THEN 1.5
                        ELSE 1.0
                    END,
                    
                    created_at: datetime()
                }]->(b2)
                
                CREATE (b2)-[r2:ADJACENT_TO {
                    wall_pair: wall_match.b2_wall + '-' + wall_match.b1_wall,
                    my_wall: wall_match.b2_wall,
                    their_wall: wall_match.b1_wall,
                    my_shared_length: wall_match.b2_length,
                    their_shared_length: wall_match.b1_length,
                    match_quality: wall_match.match_quality,
                    distance_m: distance,
                    adjacency_strength: (wall_match.b1_length + wall_match.b2_length) / 2,
                    
                    // Same energy implications
                    thermal_coupling: true,
                    cable_distance: distance * 0.5,
                    transmission_loss_factor: 0.001 * distance,  -- 0.1% per meter for proper LV cabling
                    energy_sharing_viable: distance < 3,
                    
                    function_diversity: CASE 
                        WHEN b1.building_function <> b2.building_function THEN 2.0
                        WHEN b1.residential_type <> b2.residential_type THEN 1.5
                        ELSE 1.0
                    END,
                    
                    solar_diversity: CASE
                        WHEN b1.has_solar <> b2.has_solar THEN 2.0
                        WHEN b1.solar_potential <> b2.solar_potential THEN 1.5
                        ELSE 1.0
                    END,
                    
                    created_at: datetime()
                }]->(b1)
                
                RETURN count(DISTINCT r1) as relationships_created
            """)
            
            single_result = result.single()
            count = single_result['relationships_created'] if single_result else 0
            self.stats['relationships_created'] = count * 2  # Both directions
            logger.info(f"Created {count * 2} adjacency relationships ({count} pairs)")
            
            return count * 2
    
    def validate_adjacencies(self) -> Dict:
        """Validate adjacency patterns match expectations based on housing types"""
        logger.info("Validating adjacency patterns...")
        
        with self.driver.session() as session:
            validations = {}
            
            # Validation 1: Row houses (rijtjeswoning) should have 2 neighbors
            result = session.run("""
                MATCH (b:Building)
                WHERE b.woningtype = 'rijtjeswoning'
                OPTIONAL MATCH (b)-[r:ADJACENT_TO]-()
                WITH b, count(DISTINCT r) as neighbor_count
                RETURN 
                    count(b) as total,
                    sum(CASE WHEN neighbor_count = 2 THEN 1 ELSE 0 END) as correct,
                    avg(neighbor_count) as avg_neighbors
            """)
            row = result.single()
            if row and row['total'] > 0:
                validations['row_houses'] = {
                    'total': row['total'],
                    'correct': row['correct'] or 0,
                    'avg_neighbors': row['avg_neighbors'] or 0,
                    'accuracy': (row['correct'] or 0) / row['total']
                }
            
            # Validation 2: Detached (vrijstaand) should have 0 neighbors
            result = session.run("""
                MATCH (b:Building)
                WHERE b.woningtype = 'vrijstaand'
                OPTIONAL MATCH (b)-[r:ADJACENT_TO]-()
                WITH b, count(DISTINCT r) as neighbor_count
                RETURN 
                    count(b) as total,
                    sum(CASE WHEN neighbor_count = 0 THEN 1 ELSE 0 END) as correct,
                    avg(neighbor_count) as avg_neighbors
            """)
            row = result.single()
            if row and row['total'] > 0:
                validations['detached'] = {
                    'total': row['total'],
                    'correct': row['correct'] or 0,
                    'avg_neighbors': row['avg_neighbors'] or 0,
                    'accuracy': (row['correct'] or 0) / row['total']
                }
            
            # Validation 3: Semi-detached (twee_onder_1_kap) should have 1 neighbor
            result = session.run("""
                MATCH (b:Building)
                WHERE b.woningtype = 'twee_onder_1_kap'
                OPTIONAL MATCH (b)-[r:ADJACENT_TO]-()
                WITH b, count(DISTINCT r) as neighbor_count
                RETURN 
                    count(b) as total,
                    sum(CASE WHEN neighbor_count = 1 THEN 1 ELSE 0 END) as correct,
                    avg(neighbor_count) as avg_neighbors
            """)
            row = result.single()
            if row and row['total'] > 0:
                validations['semi_detached'] = {
                    'total': row['total'],
                    'correct': row['correct'] or 0,
                    'avg_neighbors': row['avg_neighbors'] or 0,
                    'accuracy': (row['correct'] or 0) / row['total']
                }
            
            # Validation 4: MIDDLE_ROW adjacency type should match
            result = session.run("""
                MATCH (b:Building {adjacency_type: 'MIDDLE_ROW'})
                OPTIONAL MATCH (b)-[r:ADJACENT_TO]-()
                WITH b, count(DISTINCT r) as neighbor_count
                RETURN 
                    count(b) as total,
                    sum(CASE WHEN neighbor_count = 2 THEN 1 ELSE 0 END) as correct,
                    avg(neighbor_count) as avg_neighbors
            """)
            row = result.single()
            if row and row['total'] > 0:
                validations['middle_row_type'] = {
                    'total': row['total'],
                    'correct': row['correct'] or 0,
                    'avg_neighbors': row['avg_neighbors'] or 0,
                    'accuracy': (row['correct'] or 0) / row['total']
                }
            
            # Validation 5: Reciprocal relationships
            result = session.run("""
                MATCH (b1:Building)-[r:ADJACENT_TO]->(b2:Building)
                WHERE b1.ogc_fid < b2.ogc_fid
                OPTIONAL MATCH (b2)-[r2:ADJACENT_TO]->(b1)
                RETURN 
                    count(r) as total_relationships,
                    count(r2) as reciprocal_relationships
            """)
            row = result.single()
            if row and row['total_relationships'] > 0:
                validations['reciprocal'] = {
                    'total': row['total_relationships'],
                    'reciprocal': row['reciprocal_relationships'] or 0,
                    'accuracy': (row['reciprocal_relationships'] or 0) / row['total_relationships']
                }
            
            self.stats['validation_results'] = validations
            return validations
    
    def create_adjacency_clusters(self):
        """Create natural clusters from adjacency patterns"""
        logger.info("Creating natural adjacency clusters...")
        
        with self.driver.session() as session:
            # Clear existing adjacency clusters
            session.run("MATCH (ac:AdjacencyCluster) DETACH DELETE ac")
            
            total_clusters = 0
            
            # Method 1: Row House Clusters - FIXED to use elementId()
            logger.info("Creating row house clusters...")
            
            result = session.run("""
                // Find unprocessed row houses
                MATCH (start:Building)
                WHERE (start.adjacency_type = 'MIDDLE_ROW' OR start.adjacency_type = 'END_ROW')
                AND NOT EXISTS((start)<-[:IN_ADJACENCY_CLUSTER]-())
                
                // Find connected row of buildings
                MATCH path = (start)-[:ADJACENT_TO*1..20]-(connected:Building)
                WHERE connected.adjacency_type IN ['MIDDLE_ROW', 'END_ROW', 'CORNER']
                
                WITH start, collect(DISTINCT connected) + [start] as row_buildings
                WHERE size(row_buildings) >= 3
                
                // Get LV group for cluster
                WITH row_buildings, row_buildings[0].lv_group_id as lv_group,
                    row_buildings[0].district_name as district,
                    row_buildings[0].ogc_fid as first_building_id
                
                CREATE (ac:AdjacencyCluster {
                    cluster_id: 'ROW_' + lv_group + '_' + toString(first_building_id),
                    cluster_type: 'ROW_HOUSES',
                    member_count: size(row_buildings),
                    lv_group_id: lv_group,
                    district_name: district,
                    created_at: datetime(),
                    pattern: 'LINEAR',
                    thermal_benefit: 'HIGH',
                    cable_savings: 'HIGH',
                    avg_shared_walls: 2.0
                })
                
                WITH ac, row_buildings
                UNWIND row_buildings as building
                CREATE (building)-[:IN_ADJACENCY_CLUSTER {
                    joined_at: datetime()
                }]->(ac)
                
                RETURN count(DISTINCT ac) as clusters_created
            """)
            
            row_result = result.single()
            row_count = row_result['clusters_created'] if row_result else 0
            logger.info(f"Created {row_count} row house clusters")
            
            # Method 2: Corner/Courtyard Clusters
            logger.info("Creating corner/courtyard clusters...")
            
            result = session.run("""
                // Find unprocessed corner/courtyard buildings
                MATCH (center:Building)
                WHERE center.adjacency_type IN ['CORNER', 'COURTYARD']
                AND NOT EXISTS((center)<-[:IN_ADJACENCY_CLUSTER]-())
                
                // Find immediate neighbors
                MATCH (center)-[:ADJACENT_TO]-(neighbor:Building)
                WHERE NOT EXISTS((neighbor)<-[:IN_ADJACENCY_CLUSTER]-())
                
                WITH center, collect(DISTINCT neighbor) as neighbors
                WHERE size(neighbors) >= 2
                
                CREATE (ac:AdjacencyCluster {
                    cluster_id: 'CORNER_' + center.lv_group_id + '_' + toString(center.ogc_fid),
                    cluster_type: CASE 
                        WHEN center.adjacency_type = 'COURTYARD' THEN 'COURTYARD_BLOCK'
                        ELSE 'CORNER_BLOCK'
                    END,
                    member_count: size(neighbors) + 1,
                    lv_group_id: center.lv_group_id,
                    district_name: center.district_name,
                    created_at: datetime(),
                    pattern: CASE 
                        WHEN center.adjacency_type = 'COURTYARD' THEN 'ENCLOSED'
                        ELSE 'L_SHAPE'
                    END,
                    thermal_benefit: 'MEDIUM',
                    cable_savings: 'MEDIUM',
                    avg_shared_walls: 1.5
                })
                
                CREATE (center)-[:IN_ADJACENCY_CLUSTER {
                    role: 'CENTER',
                    joined_at: datetime()
                }]->(ac)
                
                WITH ac, neighbors
                UNWIND neighbors as neighbor
                CREATE (neighbor)-[:IN_ADJACENCY_CLUSTER {
                    role: 'MEMBER',
                    joined_at: datetime()
                }]->(ac)
                
                RETURN count(DISTINCT ac) as created
            """)
            
            corner_result = result.single()
            corner_count = corner_result['created'] if corner_result else 0
            logger.info(f"Created {corner_count} corner/courtyard clusters")
            
            # Method 3: Apartment Complex Clusters - FIXED to use ogc_fid
            logger.info("Creating apartment clusters...")
            
            result = session.run("""
                // Find apartment buildings in same LV group and location
                MATCH (b1:Building)
                WHERE b1.woningtype = 'appartement'
                AND NOT EXISTS((b1)<-[:IN_ADJACENCY_CLUSTER]-())
                
                MATCH (b2:Building)
                WHERE b2.woningtype = 'appartement'
                AND b1.ogc_fid < b2.ogc_fid
                AND b1.lv_group_id = b2.lv_group_id
                AND b1.district_name = b2.district_name
                AND NOT EXISTS((b2)<-[:IN_ADJACENCY_CLUSTER]-())
                AND sqrt((b2.x - b1.x)^2 + (b2.y - b1.y)^2) < 50  // Within 50m
                
                WITH b1.lv_group_id as lv_group, 
                    b1.district_name as district,
                    collect(DISTINCT b1) + collect(DISTINCT b2) as apartments,
                    min(b1.ogc_fid) as min_building_id
                WHERE size(apartments) >= 2
                
                CREATE (ac:AdjacencyCluster {
                    cluster_id: 'APT_' + lv_group + '_' + toString(min_building_id),
                    cluster_type: 'APARTMENT_COMPLEX',
                    member_count: size(apartments),
                    lv_group_id: lv_group,
                    district_name: district,
                    created_at: datetime(),
                    pattern: 'VERTICAL',
                    thermal_benefit: 'LOW',
                    cable_savings: 'VERY_HIGH',
                    avg_shared_walls: 1.0
                })
                
                WITH ac, apartments
                UNWIND apartments as apt
                CREATE (apt)-[:IN_ADJACENCY_CLUSTER {
                    joined_at: datetime()
                }]->(ac)
                
                RETURN count(DISTINCT ac) as created
            """)
            
            apt_result = result.single()
            apt_count = apt_result['created'] if apt_result else 0
            logger.info(f"Created {apt_count} apartment clusters")
            
            total_clusters = row_count + corner_count + apt_count
            self.stats['clusters_created'] = total_clusters
            
            logger.info(f"Created {total_clusters} adjacency clusters total")
            
            return total_clusters
    
    def enhance_building_metrics(self):
        """Add adjacency-based metrics to buildings"""
        logger.info("Enhancing building metrics with adjacency data...")
        
        with self.driver.session() as session:
            # Add adjacency counts and metrics
            session.run("""
                MATCH (b:Building)
                OPTIONAL MATCH (b)-[adj:ADJACENT_TO]-()
                WITH b, 
                     count(DISTINCT adj) as adjacency_count,
                     avg(adj.adjacency_strength) as avg_strength,
                     max(adj.complementarity_potential) as max_complementarity,
                     collect(DISTINCT adj.wall_pair) as shared_walls
                
                SET b.adjacency_count = adjacency_count,
                    b.avg_adjacency_strength = COALESCE(avg_strength, 0),
                    b.max_complementarity = COALESCE(max_complementarity, 0),
                    b.has_adjacent_neighbors = adjacency_count > 0,
                    b.shared_wall_directions = shared_walls,
                    b.isolation_factor = CASE 
                        WHEN adjacency_count = 0 THEN 1.0      -- Detached
                        WHEN adjacency_count = 1 THEN 0.85     -- Semi-detached (15% reduction)
                        WHEN adjacency_count = 2 THEN 0.70     -- Row house (30% reduction)
                        ELSE 0.60                              -- Apartment (40% reduction)
                    END,
                    b.thermal_efficiency_boost = CASE
                        WHEN adjacency_count = 0 THEN 1.0
                        WHEN adjacency_count = 1 THEN 1.15     -- 15% improvement
                        WHEN adjacency_count = 2 THEN 1.30     -- 30% improvement
                        ELSE 1.40                              -- 40% improvement
                    END
            """)
            
            # Add complementarity potential for adjacent pairs
            session.run("""
                MATCH (b1:Building)-[adj:ADJACENT_TO]-(b2:Building)
                WHERE b1.ogc_fid < b2.ogc_fid
                
                WITH b1, b2, adj,
                     adj.function_diversity * adj.solar_diversity * adj.match_quality as potential
                
                SET adj.complementarity_potential = potential,
                    adj.priority_for_sharing = CASE
                        WHEN potential > 2.5 THEN 'VERY_HIGH'
                        WHEN potential > 2.0 THEN 'HIGH'
                        WHEN potential > 1.5 THEN 'MEDIUM'
                        ELSE 'LOW'
                    END,
                    adj.thermal_resistance_reduction = 0.1 * adj.match_quality
            """)
            
            # Update cluster metrics
            session.run("""
                MATCH (ac:AdjacencyCluster)<-[:IN_ADJACENCY_CLUSTER]-(b:Building)
                WITH ac, 
                     collect(b) as members,
                     avg(b.solar_capacity_kwp) as avg_solar_potential,
                     sum(CASE WHEN b.has_solar THEN 1 ELSE 0 END) as solar_count,
                     sum(CASE WHEN b.has_battery THEN 1 ELSE 0 END) as battery_count,
                     sum(CASE WHEN b.has_heat_pump THEN 1 ELSE 0 END) as hp_count,
                     collect(DISTINCT b.building_function) as functions
                     
                SET ac.avg_solar_potential_kwp = avg_solar_potential,
                    ac.solar_penetration = toFloat(solar_count) / size(members),
                    ac.battery_penetration = toFloat(battery_count) / size(members),
                    ac.hp_penetration = toFloat(hp_count) / size(members),
                    ac.function_diversity = size(functions),
                    ac.energy_sharing_potential = CASE
                        WHEN size(functions) > 1 AND solar_count > 0 THEN 'HIGH'
                        WHEN solar_count > 0 OR battery_count > 0 THEN 'MEDIUM'
                        ELSE 'LOW'
                    END
            """)
            
            logger.info("Building metrics enhanced with adjacency data")
    
    def generate_report(self):
        """Generate comprehensive adjacency report"""
        logger.info("\n" + "="*60)
        logger.info("ADJACENCY UPDATE REPORT")
        logger.info("="*60)
        
        with self.driver.session() as session:
            # Overall statistics
            result = session.run("""
                MATCH (b:Building)
                RETURN 
                    count(b) as total_buildings,
                    sum(CASE WHEN b.num_shared_walls > 0 THEN 1 ELSE 0 END) as with_shared_walls,
                    sum(CASE WHEN b.adjacency_count > 0 THEN 1 ELSE 0 END) as with_adjacencies,
                    avg(b.adjacency_count) as avg_adjacencies
            """)
            stats = result.single()
            
            if stats:
                print(f"\nBuilding Statistics:")
                print(f"  Total buildings: {stats['total_buildings']}")
                print(f"  With shared walls: {stats['with_shared_walls']}")
                print(f"  With adjacencies found: {stats['with_adjacencies']}")
                if stats['avg_adjacencies']:
                    print(f"  Average adjacencies: {stats['avg_adjacencies']:.2f}")
            
            # Adjacency type distribution
            result = session.run("""
                MATCH (b:Building)
                WHERE b.adjacency_type IS NOT NULL
                RETURN b.adjacency_type as type, count(b) as count
                ORDER BY count DESC
            """)
            
            print(f"\nAdjacency Type Distribution:")
            for record in result:
                print(f"  {record['type']}: {record['count']}")
            
            # Housing type analysis - FIXED
            result = session.run("""
                MATCH (b:Building)
                WHERE b.woningtype IS NOT NULL
                OPTIONAL MATCH (b)-[adj:ADJACENT_TO]-()
                WITH b, b.woningtype as housing_type, count(adj) as adjacency_count
                WITH housing_type, 
                    count(DISTINCT b) as count,
                    avg(adjacency_count) as avg_adjacencies
                RETURN housing_type, count, avg_adjacencies
                ORDER BY count DESC
            """)
            
            print(f"\nHousing Type Adjacency Analysis:")
            for record in result:
                if record['avg_adjacencies'] is not None:
                    print(f"  {record['housing_type']}: {record['count']} buildings, "
                        f"avg {record['avg_adjacencies']:.1f} adjacencies")
                else:
                    print(f"  {record['housing_type']}: {record['count']} buildings, no adjacencies")
            
            # Relationship statistics
            result = session.run("""
                MATCH ()-[adj:ADJACENT_TO]->()
                RETURN 
                    count(adj)/2 as total_relationships,
                    avg(adj.distance_m) as avg_distance,
                    avg(adj.adjacency_strength) as avg_strength,
                    avg(adj.match_quality) as avg_match_quality,
                    avg(adj.complementarity_potential) as avg_potential
            """)
            rel_stats = result.single()
            
            print(f"\nAdjacency Relationships:")
            if rel_stats and rel_stats['total_relationships']:
                print(f"  Total pairs: {rel_stats['total_relationships']}")
                if rel_stats['avg_distance']:
                    print(f"  Avg distance: {rel_stats['avg_distance']:.2f} m")
                if rel_stats['avg_strength']:
                    print(f"  Avg strength: {rel_stats['avg_strength']:.2f}")
                if rel_stats['avg_match_quality']:
                    print(f"  Avg match quality: {rel_stats['avg_match_quality']:.2f}")
                if rel_stats['avg_potential']:
                    print(f"  Avg complementarity: {rel_stats['avg_potential']:.2f}")
            
            # Cluster statistics
            result = session.run("""
                MATCH (ac:AdjacencyCluster)
                RETURN 
                    ac.cluster_type as type,
                    count(ac) as count,
                    avg(ac.member_count) as avg_size,
                    avg(ac.solar_penetration) as avg_solar_pen
                ORDER BY count DESC
            """)
            
            print(f"\nAdjacency Clusters:")
            has_clusters = False
            for record in result:
                has_clusters = True
                solar_pen_str = f"{record['avg_solar_pen']:.1%}" if record['avg_solar_pen'] is not None else "N/A"
                print(f"  {record['type']}: {record['count']} clusters, "
                    f"avg size: {record['avg_size']:.1f}, "
                    f"solar pen: {solar_pen_str}")
            if not has_clusters:
                print("  No clusters created")
            
            # LV Group adjacency summary
            result = session.run("""
                MATCH (b:Building)-[:ADJACENT_TO]-()
                WITH b.lv_group_id as lv_group, count(DISTINCT b) as adjacent_buildings
                MATCH (b2:Building {lv_group_id: lv_group})
                WITH lv_group, 
                    adjacent_buildings,
                    count(DISTINCT b2) as total_buildings
                RETURN lv_group,
                    adjacent_buildings,
                    total_buildings,
                    toFloat(adjacent_buildings) / total_buildings as adjacency_ratio
                ORDER BY adjacency_ratio DESC
                LIMIT 10
            """)
            
            print(f"\nTop LV Groups by Adjacency Ratio:")
            for record in result:
                print(f"  {record['lv_group']}: {record['adjacent_buildings']}/{record['total_buildings']} "
                    f"({record['adjacency_ratio']:.1%})")
            
            # Top complementarity pairs
            result = session.run("""
                MATCH (b1:Building)-[adj:ADJACENT_TO]-(b2:Building)
                WHERE b1.ogc_fid < b2.ogc_fid
                AND adj.complementarity_potential > 1.5
                RETURN 
                    b1.ogc_fid as building1,
                    b2.ogc_fid as building2,
                    b1.building_function as func1,
                    b2.building_function as func2,
                    b1.woningtype as type1,
                    b2.woningtype as type2,
                    adj.wall_pair as walls,
                    adj.complementarity_potential as potential
                ORDER BY potential DESC
                LIMIT 5
            """)
            
            print(f"\nTop Complementarity Pairs:")
            has_pairs = False
            for record in result:
                has_pairs = True
                print(f"  Buildings {record['building1']}-{record['building2']}: "
                    f"{record['func1']}/{record['func2']}, "
                    f"{record['type1']}/{record['type2']}, "
                    f"potential: {record['potential']:.2f}")
            if not has_pairs:
                print("  No high-complementarity pairs found")
            
            # Validation results
            if self.stats['validation_results']:
                print(f"\nValidation Results:")
                for key, val in self.stats['validation_results'].items():
                    if key == 'reciprocal':
                        print(f"  {key}: {val.get('reciprocal', 0)}/{val.get('total', 0)} "
                            f"({val.get('accuracy', 0):.1%} accuracy)")
                    else:
                        print(f"  {key}: {val.get('correct', 0)}/{val.get('total', 0)} "
                            f"({val.get('accuracy', 0):.1%} accuracy, "
                            f"avg: {val.get('avg_neighbors', 0):.1f})")
            
            print("\n" + "="*60)
            print("ADJACENCY UPDATE COMPLETE")
            print("="*60)
    
    def run_full_update(self):
        """Run complete adjacency update process using PostgreSQL data"""
        start_time = datetime.now()
        
        try:
            # Check KG status
            status = self.check_kg_status()
            logger.info(f"KG Status: {status}")
            
            if not status['buildings_exist']:
                logger.error("No buildings found in KG. Run main KG builder (Part 1) first.")
                return False
            
            if status['adjacencies_exist']:
                logger.warning("Adjacencies already exist. They will be recreated.")
            
            # Update shared wall data from PostgreSQL
            if not status['has_shared_walls']:
                logger.info("Updating shared wall data from PostgreSQL...")
                self.update_shared_wall_data_from_postgres()
            else:
                logger.info("Shared wall data already exists in Neo4j")
            
            # Create adjacency relationships
            self.create_adjacency_relationships()
            
            # Validate
            validation = self.validate_adjacencies()
            
            # Create clusters
            self.create_adjacency_clusters()
            
            # Enhance metrics
            self.enhance_building_metrics()
            
            # Generate report
            self.generate_report()
            
            # Calculate time
            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"Total processing time: {elapsed:.2f} seconds")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during adjacency update: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Configuration (SAME AS PART 1)
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "aminasad"
    
    # PostgreSQL configuration (SAME AS PART 1)
    PG_HOST = "localhost"
    PG_DATABASE = "research"
    PG_USER = "aminj"
    PG_PASSWORD = "Aminej@geodan!"
    
    print("Starting Adjacency Module (Part 2) using PostgreSQL data...")
    print(f"Connecting to PostgreSQL database: {PG_DATABASE}")
    print(f"Connecting to Neo4j at: {NEO4J_URI}")
    
    # Create updater with both connections
    updater = AdjacencyUpdater(
        NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD,
        PG_HOST, PG_DATABASE, PG_USER, PG_PASSWORD
    )
    
    try:
        # Run full update using PostgreSQL data
        success = updater.run_full_update()
        
        if success:
            print("\n✅ Adjacency update completed successfully!")
            print("Your KG now has:")
            print("  - ADJACENT_TO relationships with energy implications")
            print("  - Natural adjacency clusters (row houses, corners, apartments)")
            print("  - Enhanced complementarity potential for adjacent buildings")
            print("  - Thermal coupling metrics")
            print("  - Validation against housing types (woningtype)")
            print("\nReady for GNN processing with adjacency-aware features!")
        
    finally:
        updater.close()