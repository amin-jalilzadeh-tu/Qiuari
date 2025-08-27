"""
Test Neo4j connection and check current database state
"""

from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_connection():
    """Test Neo4j connection and get database stats."""
    
    # Connection parameters - update these if needed
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"  # Common default passwords to try
    
    passwords_to_try = ["12345678", "password", "neo4j", "admin", "neo4jadmin"]
    
    for pwd in passwords_to_try:
        try:
            logger.info(f"Trying to connect to Neo4j at {NEO4J_URI}...")
            driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, pwd))
            
            with driver.session() as session:
                # Test connection
                result = session.run("RETURN 1 as test")
                if result.single():
                    logger.info(f"✓ Connected successfully with password: {pwd[:2]}***")
                    
                    # Get node counts
                    logger.info("\nCurrent database state:")
                    logger.info("-" * 40)
                    
                    # Count nodes by label
                    labels_query = """
                    CALL db.labels() YIELD label
                    RETURN label
                    """
                    labels = session.run(labels_query)
                    
                    for record in labels:
                        label = record['label']
                        count_query = f"MATCH (n:{label}) RETURN count(n) as count"
                        count_result = session.run(count_query)
                        count = count_result.single()['count']
                        logger.info(f"  {label}: {count} nodes")
                    
                    # Count relationships
                    rel_query = """
                    MATCH ()-[r]->()
                    RETURN type(r) as type, count(r) as count
                    ORDER BY count DESC
                    """
                    relationships = session.run(rel_query)
                    
                    logger.info("\nRelationship types:")
                    for record in relationships:
                        logger.info(f"  {record['type']}: {record['count']} relationships")
                    
                    # Check for existing indexes
                    logger.info("\nExisting indexes:")
                    index_query = "SHOW INDEXES"
                    try:
                        indexes = session.run(index_query)
                        index_count = 0
                        for record in indexes:
                            index_count += 1
                            logger.info(f"  - {record.get('name', 'unnamed')} on {record.get('labelsOrTypes', [])} ({record.get('properties', [])})")
                        if index_count == 0:
                            logger.info("  No indexes found")
                    except:
                        logger.info("  Could not retrieve index information")
                    
                    driver.close()
                    return True
                    
        except Exception as e:
            if "authentication" not in str(e).lower():
                logger.error(f"Connection failed: {e}")
                break
            else:
                continue
    
    logger.error("Could not connect to Neo4j. Please check:")
    logger.error("1. Neo4j is running (check services or Neo4j Desktop)")
    logger.error("2. Connection URI is correct (default: bolt://localhost:7687)")
    logger.error("3. Username/password are correct")
    return False

if __name__ == "__main__":
    if test_connection():
        logger.info("\n✓ Neo4j is ready for optimization!")
    else:
        logger.error("\n✗ Please fix Neo4j connection before running optimizer")