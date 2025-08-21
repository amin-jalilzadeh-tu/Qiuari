"""
Standalone Adjacency Module for Energy KG
Updates existing Knowledge Graph with adjacency relationships
Can be run independently after main KG construction
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from datetime import datetime
import logging
from typing import Dict, List, Tuple
import os
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AdjacencyUpdater:
    """Add adjacency relationships to existing Knowledge Graph"""
    
    def __init__(self, uri: str, user: str, password: str):
        """Initialize Neo4j connection"""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.stats = {
            'relationships_created': 0,
            'clusters_created': 0,
            'nodes_updated': 0,
            'validation_results': {}
        }
        logger.info(f"Connected to Neo4j for adjacency update")
    
    def close(self):
        """Close database connection"""
        self.driver.close()
        logger.info("Connection closed")
    
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
    
    def update_shared_wall_data(self, buildings_csv_path: str):
        """Update existing buildings with shared wall data from CSV"""
        logger.info("Updating buildings with shared wall data...")
        
        # Load the CSV with shared wall data
        buildings_df = pd.read_csv(buildings_csv_path)
        
        with self.driver.session() as session:
            updated = 0
            for _, building in buildings_df.iterrows():
                result = session.run("""
                    MATCH (b:Building {ogc_fid: $ogc_fid})
                    SET b.north_shared_length = $north_shared,
                        b.south_shared_length = $south_shared,
                        b.east_shared_length = $east_shared,
                        b.west_shared_length = $west_shared,
                        b.north_facade_length = $north_facade,
                        b.south_facade_length = $south_facade,
                        b.east_facade_length = $east_facade,
                        b.west_facade_length = $west_facade,
                        b.num_shared_walls = $num_shared,
                        b.total_shared_length = $total_shared,
                        b.adjacency_type = $adj_type
                    RETURN b.ogc_fid as updated_id
                """,
                ogc_fid=int(building['ogc_fid']),
                north_shared=float(building.get('north_shared_length', 0)),
                south_shared=float(building.get('south_shared_length', 0)),
                east_shared=float(building.get('east_shared_length', 0)),
                west_shared=float(building.get('west_shared_length', 0)),
                north_facade=float(building.get('north_facade_length', 10)),
                south_facade=float(building.get('south_facade_length', 10)),
                east_facade=float(building.get('east_facade_length', 10)),
                west_facade=float(building.get('west_facade_length', 10)),
                num_shared=int(building.get('num_shared_walls', 0)),
                total_shared=float(building.get('total_shared_length', 0)),
                adj_type=building.get('adjacency_type', 'UNKNOWN')
                )
                
                if result.single():
                    updated += 1
            
            self.stats['nodes_updated'] = updated
            logger.info(f"Updated {updated} buildings with shared wall data")
    
    def create_adjacency_relationships(self):
        """Find and create ADJACENT_TO relationships"""
        logger.info("Creating adjacency relationships...")
        
        with self.driver.session() as session:
            # Clear any existing adjacency relationships first
            session.run("MATCH ()-[r:ADJACENT_TO]-() DELETE r")
            
            # Create relationships using a single comprehensive query
            result = session.run("""
                // Find all buildings with shared walls
                MATCH (b1:Building)
                WHERE b1.num_shared_walls > 0
                
                // Find potential neighbors in same LV network
                MATCH (b2:Building)
                WHERE b1.ogc_fid < b2.ogc_fid  // Process each pair only once
                AND b1.lv_network_id = b2.lv_network_id
                AND b2.num_shared_walls > 0
                
                // Calculate distance
                WITH b1, b2, 
                    sqrt((b2.x - b1.x)^2 + (b2.y - b1.y)^2) as distance
                WHERE distance < 20  // Must be within 20m
                
                // Check for matching shared walls
                WITH b1, b2, distance,
                    CASE 
                        // North-South match
                        WHEN b1.north_shared_length > 0 AND b2.south_shared_length > 0 
                            AND b2.y > b1.y AND abs(b2.x - b1.x) < 10 
                        THEN {
                            pair: 'north-south',
                            b1_wall: 'north',
                            b2_wall: 'south',
                            b1_length: b1.north_shared_length,
                            b2_length: b2.south_shared_length
                        }
                        // South-North match
                        WHEN b1.south_shared_length > 0 AND b2.north_shared_length > 0
                            AND b2.y < b1.y AND abs(b2.x - b1.x) < 10
                        THEN {
                            pair: 'south-north',
                            b1_wall: 'south',
                            b2_wall: 'north',
                            b1_length: b1.south_shared_length,
                            b2_length: b2.north_shared_length
                        }
                        // East-West match
                        WHEN b1.east_shared_length > 0 AND b2.west_shared_length > 0
                            AND b2.x > b1.x AND abs(b2.y - b1.y) < 10
                        THEN {
                            pair: 'east-west',
                            b1_wall: 'east',
                            b2_wall: 'west',
                            b1_length: b1.east_shared_length,
                            b2_length: b2.west_shared_length
                        }
                        // West-East match
                        WHEN b1.west_shared_length > 0 AND b2.east_shared_length > 0
                            AND b2.x < b1.x AND abs(b2.y - b1.y) < 10
                        THEN {
                            pair: 'west-east',
                            b1_wall: 'west',
                            b2_wall: 'east',
                            b1_length: b1.west_shared_length,
                            b2_length: b2.east_shared_length
                        }
                        ELSE null
                    END as wall_match
                
                WHERE wall_match IS NOT NULL
                
                // Create BOTH directional relationships
                CREATE (b1)-[r1:ADJACENT_TO {
                    wall_pair: wall_match.pair,
                    my_wall: wall_match.b1_wall,
                    their_wall: wall_match.b2_wall,
                    my_shared_length: wall_match.b1_length,
                    their_shared_length: wall_match.b2_length,
                    distance_m: distance,
                    adjacency_strength: (wall_match.b1_length + wall_match.b2_length) / 2,
                    
                    // Energy implications
                    thermal_coupling: true,
                    cable_distance: distance * 0.5,
                    transmission_loss_factor: 0.02,
                    energy_sharing_viable: true,
                    
                    // Complementarity
                    function_diversity: CASE 
                        WHEN b1.building_function <> b2.building_function THEN 1.5
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
                    distance_m: distance,
                    adjacency_strength: (wall_match.b1_length + wall_match.b2_length) / 2,
                    
                    // Same energy implications
                    thermal_coupling: true,
                    cable_distance: distance * 0.5,
                    transmission_loss_factor: 0.02,
                    energy_sharing_viable: true,
                    
                    function_diversity: CASE 
                        WHEN b1.building_function <> b2.building_function THEN 1.5
                        ELSE 1.0
                    END,
                    
                    created_at: datetime()
                }]->(b1)
                
                RETURN count(DISTINCT r1) as relationships_created
            """)
            
            # FIX: Store result once
            single_result = result.single()
            count = single_result['relationships_created'] if single_result else 0
            self.stats['relationships_created'] = count * 2  # Multiply by 2 since we create pairs
            logger.info(f"Created {count * 2} adjacency relationships ({count} pairs)")
            
            return count * 2
    
    def validate_adjacencies(self) -> Dict:
        """Validate adjacency patterns match expectations"""
        logger.info("Validating adjacency patterns...")
        
        with self.driver.session() as session:
            validations = {}
            
            # Validation 1: MIDDLE_ROW should have exactly 2 neighbors
            result = session.run("""
                MATCH (b:Building {adjacency_type: 'MIDDLE_ROW'})
                OPTIONAL MATCH (b)-[r:ADJACENT_TO]-()
                WITH b, count(r) as neighbor_count
                RETURN 
                    count(b) as total,
                    sum(CASE WHEN neighbor_count = 2 THEN 1 ELSE 0 END) as correct,
                    collect(CASE WHEN neighbor_count <> 2 
                           THEN {id: b.ogc_fid, neighbors: neighbor_count} 
                           END) as issues
            """)
            row = result.single()
            if row:
                validations['middle_row'] = {
                    'total': row['total'],
                    'correct': row['correct'],
                    'accuracy': row['correct'] / row['total'] if row['total'] > 0 else 0,
                    'issues': [i for i in row['issues'] if i is not None]
                }
            
            # Validation 2: CORNER should have 2 perpendicular neighbors
            result = session.run("""
                MATCH (b:Building {adjacency_type: 'CORNER'})
                OPTIONAL MATCH (b)-[r:ADJACENT_TO]-()
                WITH b, count(r) as neighbor_count
                RETURN 
                    count(b) as total,
                    sum(CASE WHEN neighbor_count = 2 THEN 1 ELSE 0 END) as correct
            """)
            row = result.single()
            if row:
                validations['corner'] = {
                    'total': row['total'],
                    'correct': row['correct'],
                    'accuracy': row['correct'] / row['total'] if row['total'] > 0 else 0
                }
            
            # Validation 3: ISOLATED should have no neighbors
            result = session.run("""
                MATCH (b:Building {adjacency_type: 'ISOLATED'})
                OPTIONAL MATCH (b)-[r:ADJACENT_TO]-()
                WITH b, count(r) as neighbor_count
                RETURN 
                    count(b) as total,
                    sum(CASE WHEN neighbor_count = 0 THEN 1 ELSE 0 END) as correct
            """)
            row = result.single()
            if row:
                validations['isolated'] = {
                    'total': row['total'],
                    'correct': row['correct'],
                    'accuracy': row['correct'] / row['total'] if row['total'] > 0 else 0
                }
            
            # Validation 4: Reciprocal check (different structure)
            result = session.run("""
                MATCH (b1:Building)-[r:ADJACENT_TO]->(b2:Building)
                WHERE b1.ogc_fid < b2.ogc_fid
                OPTIONAL MATCH (b2)-[r2:ADJACENT_TO]->(b1)
                RETURN 
                    count(r) as total_relationships,
                    count(r2) as reciprocal_relationships
            """)
            row = result.single()
            if row:
                validations['reciprocal'] = {
                    'total': row['total_relationships'],
                    'reciprocal': row['reciprocal_relationships'],
                    'accuracy': row['reciprocal_relationships'] / row['total_relationships'] 
                               if row['total_relationships'] > 0 else 0
                }
            
            self.stats['validation_results'] = validations
            return validations
    
    def create_adjacency_clusters(self):
        """Create natural clusters from adjacency patterns with batch processing"""
        logger.info("Creating natural adjacency clusters...")
        
        with self.driver.session() as session:
            # Clear existing adjacency clusters
            session.run("MATCH (ac:AdjacencyCluster) DETACH DELETE ac")
            
            total_clusters = 0
            
            # Method 1: Row House Clusters (SIMPLIFIED)
            logger.info("Creating row house clusters...")
            
            # Create clusters for connected MIDDLE_ROW buildings
            result = session.run("""
                MATCH (start:Building {adjacency_type: 'MIDDLE_ROW'})
                WHERE NOT EXISTS((start)<-[:IN_ADJACENCY_CLUSTER]-())
                MATCH path = (start)-[:ADJACENT_TO*1..10]-(connected:Building {adjacency_type: 'MIDDLE_ROW'})
                WITH start, collect(DISTINCT connected) + [start] as row_buildings
                WHERE size(row_buildings) >= 3
                
                CREATE (ac:AdjacencyCluster {
                    cluster_id: 'ROW_' + toString(start.ogc_fid) + '_' + toString(toInteger(rand() * 10000)),
                    cluster_type: 'ROW_HOUSES',
                    member_count: size(row_buildings),
                    created_at: datetime(),
                    pattern: 'LINEAR',
                    thermal_benefit: 'HIGH',
                    cable_savings: 'HIGH'
                })
                
                WITH ac, row_buildings
                UNWIND row_buildings as building
                MERGE (building)-[:IN_ADJACENCY_CLUSTER]->(ac)
                
                RETURN count(DISTINCT ac) as clusters_created
            """)
            
            # FIX: Store result.single() once
            row_result = result.single()
            row_count = row_result['clusters_created'] if row_result else 0
            logger.info(f"Total row house clusters: {row_count}")
            
            # Method 2: Corner Block Clusters (SIMPLIFIED)
            logger.info("Creating corner block clusters...")
            
            result = session.run("""
                MATCH (corner:Building {adjacency_type: 'CORNER'})
                WHERE NOT EXISTS((corner)<-[:IN_ADJACENCY_CLUSTER]-())
                OPTIONAL MATCH (corner)-[:ADJACENT_TO]-(neighbor:Building)
                WITH corner, collect(DISTINCT neighbor) as neighbors
                WHERE size(neighbors) >= 2
                
                CREATE (ac:AdjacencyCluster {
                    cluster_id: 'CORNER_' + toString(corner.ogc_fid) + '_' + toString(toInteger(rand() * 1000)),
                    cluster_type: 'CORNER_BLOCK',
                    member_count: size(neighbors) + 1,
                    created_at: datetime(),
                    pattern: 'L_SHAPE',
                    thermal_benefit: 'MEDIUM',
                    cable_savings: 'MEDIUM'
                })
                
                CREATE (corner)-[:IN_ADJACENCY_CLUSTER]->(ac)
                WITH ac, neighbors
                UNWIND neighbors as neighbor
                CREATE (neighbor)-[:IN_ADJACENCY_CLUSTER]->(ac)
                
                RETURN count(DISTINCT ac) as created
            """)
            
            # FIX: Store result.single() once
            corner_result = result.single()
            corner_count = corner_result['created'] if corner_result else 0
            logger.info(f"Total corner block clusters: {corner_count}")
            
            total_clusters = row_count + corner_count
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
                     count(adj) as adjacency_count,
                     avg(adj.adjacency_strength) as avg_strength,
                     sum(adj.thermal_resistance_reduction) as thermal_benefit,
                     collect(DISTINCT adj.wall_pair) as shared_walls
                
                SET b.adjacency_count = adjacency_count,
                    b.avg_adjacency_strength = COALESCE(avg_strength, 0),
                    b.thermal_efficiency_boost = 1 + COALESCE(thermal_benefit, 0),
                    b.has_adjacent_neighbors = adjacency_count > 0,
                    b.shared_wall_directions = shared_walls,
                    b.isolation_factor = CASE 
                        WHEN adjacency_count = 0 THEN 1.0
                        WHEN adjacency_count = 1 THEN 0.7
                        WHEN adjacency_count = 2 THEN 0.5
                        ELSE 0.3
                    END
            """)
            
            # Add complementarity potential for adjacent pairs
            session.run("""
                MATCH (b1:Building)-[adj:ADJACENT_TO]-(b2:Building)
                WHERE b1.ogc_fid < b2.ogc_fid
                
                WITH b1, b2, adj,
                     adj.function_diversity * adj.solar_diversity * 1.5 as potential
                
                SET adj.complementarity_potential = potential,
                    adj.priority_for_sharing = CASE
                        WHEN potential > 2.0 THEN 'HIGH'
                        WHEN potential > 1.5 THEN 'MEDIUM'
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
            
            # Relationship statistics
            result = session.run("""
                MATCH ()-[adj:ADJACENT_TO]->()
                RETURN 
                    count(adj)/2 as total_relationships,
                    avg(adj.distance_m) as avg_distance,
                    avg(adj.adjacency_strength) as avg_strength,
                    avg(adj.complementarity_potential) as avg_potential
            """)
            rel_stats = result.single()
            
            print(f"\nAdjacency Relationships:")
            if rel_stats and rel_stats['total_relationships']:
                print(f"  Total: {rel_stats['total_relationships']}")
                if rel_stats['avg_distance']:
                    print(f"  Avg distance: {rel_stats['avg_distance']:.2f} m")
                if rel_stats['avg_strength']:
                    print(f"  Avg strength: {rel_stats['avg_strength']:.2f}")
                if rel_stats['avg_potential']:
                    print(f"  Avg complementarity potential: {rel_stats['avg_potential']:.2f}")
            else:
                print("  No adjacency relationships found")
            
            # Cluster statistics
            result = session.run("""
                MATCH (ac:AdjacencyCluster)
                RETURN 
                    ac.cluster_type as type,
                    count(ac) as count,
                    avg(ac.member_count) as avg_size
                ORDER BY count DESC
            """)
            
            print(f"\nAdjacency Clusters:")
            has_clusters = False
            for record in result:
                has_clusters = True
                print(f"  {record['type']}: {record['count']} clusters, "
                      f"avg size: {record['avg_size']:.1f}")
            if not has_clusters:
                print("  No clusters created")
            
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
                      f"{record['walls']}, potential: {record['potential']:.2f}")
            if not has_pairs:
                print("  No high-complementarity pairs found")
            
            # Validation results - FIXED to handle different structures
            if self.stats['validation_results']:
                print(f"\nValidation Results:")
                for key, val in self.stats['validation_results'].items():
                    if key == 'reciprocal':
                        # Special handling for reciprocal validation
                        print(f"  {key}: {val.get('reciprocal', 0)}/{val.get('total', 0)} "
                              f"({val.get('accuracy', 0):.1%} accuracy)")
                    else:
                        # Standard validation format
                        print(f"  {key}: {val.get('correct', 0)}/{val.get('total', 0)} "
                              f"({val.get('accuracy', 0):.1%} accuracy)")
            
            print("\n" + "="*60)
            print("ADJACENCY UPDATE COMPLETE")
            print("="*60)
    
    def run_full_update(self, buildings_csv_path: str = None):
        """Run complete adjacency update process"""
        start_time = datetime.now()
        
        try:
            # Check KG status
            status = self.check_kg_status()
            logger.info(f"KG Status: {status}")
            
            if not status['buildings_exist']:
                logger.error("No buildings found in KG. Run main KG builder first.")
                return False
            
            if status['adjacencies_exist']:
                logger.warning("Adjacencies already exist. They will be recreated.")
            
            # Update shared wall data if CSV provided
            if buildings_csv_path and not status['has_shared_walls']:
                self.update_shared_wall_data(buildings_csv_path)
            elif not status['has_shared_walls']:
                logger.error("No shared wall data found. Provide CSV path.")
                return False
            
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
            raise

# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "aminasad"
    
    # Platform-independent path
    BUILDINGS_CSV = os.path.join("mimic_data", "buildings.csv")
    
    # Verify file exists
    if not os.path.exists(BUILDINGS_CSV):
        print(f"Error: File not found at {BUILDINGS_CSV}")
        print(f"Current directory: {os.getcwd()}")
        print(f"Files in mimic_data: {os.listdir('mimic_data') if os.path.exists('mimic_data') else 'Directory not found'}")
    else:
        print(f"✓ Found buildings.csv at {BUILDINGS_CSV}")
        
        # Create updater
        updater = AdjacencyUpdater(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        try:
            # Run full update
            success = updater.run_full_update(BUILDINGS_CSV)
            
            if success:
                print("\n✅ Adjacency update completed successfully!")
                print("Your KG now has:")
                print("  - ADJACENT_TO relationships with energy implications")
                print("  - Natural adjacency clusters (row houses, corners, courtyards)")
                print("  - Enhanced complementarity potential for adjacent buildings")
                print("  - Thermal coupling metrics")
                print("\nReady for GNN processing with adjacency-aware features!")
            
        finally:
            updater.close()