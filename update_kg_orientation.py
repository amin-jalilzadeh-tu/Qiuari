"""
Script to update Neo4j KG with orientation data from PostgreSQL
This will add the missing orientation data to existing Building nodes
"""

import psycopg2
from neo4j import GraphDatabase
import yaml
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def update_orientation_data():
    """Update Neo4j Building nodes with orientation from PostgreSQL"""
    
    # Load Neo4j config
    with open('config/unified_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # PostgreSQL connection
    pg_conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="research", 
        user="aminj",
        password="Aminej@geodan!"
    )
    pg_cursor = pg_conn.cursor()
    
    # Neo4j connection
    neo4j_driver = GraphDatabase.driver(
        config['kg']['uri'],
        auth=(config['kg']['user'], config['kg']['password'])
    )
    
    try:
        logger.info("Starting orientation data update...")
        
        # Step 1: Get orientation data from PostgreSQL
        logger.info("Fetching orientation data from PostgreSQL...")
        
        query = """
        SELECT 
            b.ogc_fid,
            b.building_orientation,
            b.building_orientation_cardinal,
            b.b3_opp_dak_plat as flat_roof_area,
            b.b3_opp_dak_schuin as sloped_roof_area,
            COALESCE(b.b3_opp_dak_plat, 0) + COALESCE(b.b3_opp_dak_schuin, 0) as total_roof_area
        FROM amin_grid.amsterdam_buildings_1_deducted b
        WHERE b.building_orientation_cardinal IS NOT NULL
        """
        
        pg_cursor.execute(query)
        buildings_data = pg_cursor.fetchall()
        
        logger.info(f"Found {len(buildings_data)} buildings with orientation data")
        
        # Step 2: Update Neo4j in batches
        logger.info("Updating Neo4j Building nodes...")
        
        batch_size = 1000
        total_updated = 0
        
        with neo4j_driver.session() as session:
            
            # First, check how many Building nodes exist
            result = session.run("MATCH (b:Building) RETURN count(b) as count")
            total_buildings = result.single()['count']
            logger.info(f"Total Building nodes in Neo4j: {total_buildings}")
            
            # Process in batches
            for i in tqdm(range(0, len(buildings_data), batch_size), desc="Updating buildings"):
                batch = buildings_data[i:i+batch_size]
                
                # Prepare batch data
                batch_updates = []
                for row in batch:
                    ogc_fid, orientation, cardinal, flat_roof, sloped_roof, total_roof = row
                    
                    batch_updates.append({
                        'ogc_fid': ogc_fid,
                        'orientation': float(orientation) if orientation is not None else None,
                        'cardinal': cardinal,
                        'flat_roof': float(flat_roof) if flat_roof is not None else 0.0,
                        'sloped_roof': float(sloped_roof) if sloped_roof is not None else 0.0,
                        'total_roof': float(total_roof) if total_roof is not None else 0.0
                    })
                
                # Update Neo4j
                update_query = """
                UNWIND $updates as update
                MATCH (b:Building {ogc_fid: update.ogc_fid})
                SET b.building_orientation = update.orientation,
                    b.building_orientation_cardinal = update.cardinal,
                    b.orientation = update.cardinal,  // Add simplified field
                    b.flat_roof_area = update.flat_roof,
                    b.sloped_roof_area = update.sloped_roof,
                    b.roof_area = update.total_roof  // Add total roof area
                RETURN count(b) as updated_count
                """
                
                result = session.run(update_query, updates=batch_updates)
                batch_updated = result.single()['updated_count']
                total_updated += batch_updated
        
        logger.info(f"Successfully updated {total_updated} Building nodes with orientation data")
        
        # Step 3: Verify the update
        logger.info("Verifying update...")
        
        with neo4j_driver.session() as session:
            # Check buildings with orientation
            result = session.run("""
                MATCH (b:Building)
                WHERE b.building_orientation_cardinal IS NOT NULL
                RETURN count(b) as with_orientation
            """)
            with_orientation = result.single()['with_orientation']
            
            # Check distribution
            result = session.run("""
                MATCH (b:Building)
                WHERE b.building_orientation_cardinal IS NOT NULL
                RETURN b.building_orientation_cardinal as direction, count(b) as count
                ORDER BY count DESC
            """)
            
            logger.info(f"Buildings with orientation: {with_orientation}")
            logger.info("Orientation distribution:")
            for record in result:
                logger.info(f"  {record['direction']}: {record['count']} buildings")
            
            # Check roof area update
            result = session.run("""
                MATCH (b:Building)
                WHERE b.roof_area > 0
                RETURN count(b) as with_roof
            """)
            with_roof = result.single()['with_roof']
            logger.info(f"Buildings with roof area: {with_roof}")
        
        # Step 4: Add indexes for better query performance
        logger.info("Creating indexes for orientation fields...")
        
        with neo4j_driver.session() as session:
            # Create indexes
            try:
                session.run("CREATE INDEX building_orientation IF NOT EXISTS FOR (b:Building) ON (b.building_orientation_cardinal)")
                session.run("CREATE INDEX building_roof IF NOT EXISTS FOR (b:Building) ON (b.roof_area)")
                logger.info("Indexes created successfully")
            except Exception as e:
                logger.warning(f"Index creation note: {e}")
        
        logger.info("✅ Orientation update complete!")
        
    except Exception as e:
        logger.error(f"Error updating orientation data: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        pg_cursor.close()
        pg_conn.close()
        neo4j_driver.close()

def verify_orientation_in_kg():
    """Quick verification of orientation data in KG"""
    
    with open('config/unified_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    driver = GraphDatabase.driver(
        config['kg']['uri'],
        auth=(config['kg']['user'], config['kg']['password'])
    )
    
    with driver.session() as session:
        # Sample buildings with orientation
        result = session.run("""
            MATCH (b:Building)
            WHERE b.building_orientation_cardinal IS NOT NULL
            RETURN b.ogc_fid as id,
                   b.building_orientation as angle,
                   b.building_orientation_cardinal as cardinal,
                   b.energy_label as label,
                   b.roof_area as roof,
                   b.has_solar as solar
            LIMIT 10
        """)
        
        print("\nSample buildings with orientation:")
        print("-" * 60)
        for record in result:
            print(f"Building {record['id']}:")
            print(f"  Orientation: {record['angle']:.1f}° ({record['cardinal']})")
            print(f"  Energy Label: {record['label']}")
            print(f"  Roof Area: {record['roof']:.1f} m²")
            print(f"  Has Solar: {record['solar']}")
    
    driver.close()

if __name__ == "__main__":
    print("=" * 60)
    print("UPDATING NEO4J WITH ORIENTATION DATA")
    print("=" * 60)
    
    # Run the update
    update_orientation_data()
    
    # Verify the results
    print("\n" + "=" * 60)
    print("VERIFICATION")
    print("=" * 60)
    verify_orientation_in_kg()