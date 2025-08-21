"""
Real Knowledge Graph connector for Neo4j
"""

from neo4j import GraphDatabase
import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class KGConnector:
    """Real Neo4j connector for Knowledge Graph operations"""
    
    def __init__(self, uri: str, user: str, password: str, **kwargs):
        """Initialize KG connector"""
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        self.connected = False
        
        # Auto-connect
        self.connect()
    
    def connect(self) -> bool:
        """Connect to Neo4j database"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password)
            )
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.connected = True
            logger.info("Connected to Neo4j successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.connected = False
            return False
    
    def query(self, cypher_query: str) -> List[Dict]:
        """Execute Cypher query and return results"""
        if not self.connected:
            logger.error("Not connected to database")
            return []
        
        try:
            with self.driver.session() as session:
                result = session.run(cypher_query)
                return [dict(record) for record in result]
        except Exception as e:
            logger.error(f"Query failed: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, int]:
        """Get database statistics"""
        if not self.connected:
            return {"total_nodes": 0}
        
        try:
            with self.driver.session() as session:
                # Total nodes
                result = session.run("MATCH (n) RETURN COUNT(n) as count")
                total_nodes = result.single()['count']
                
                # Building count
                result = session.run("MATCH (b:Building) RETURN COUNT(b) as count")
                buildings = result.single()['count']
                
                # LV Network count
                result = session.run("MATCH (lv:LV_Network) RETURN COUNT(lv) as count")
                lv_networks = result.single()['count']
                
                # Relationships
                result = session.run("MATCH ()-[r]->() RETURN COUNT(r) as count")
                relationships = result.single()['count']
                
                return {
                    "total_nodes": total_nodes,
                    "buildings": buildings,
                    "lv_networks": lv_networks,
                    "relationships": relationships
                }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"total_nodes": 0}
    
    def close(self):
        """Close the connection"""
        if self.driver:
            self.driver.close()
            self.connected = False
            logger.info("Disconnected from Neo4j")
    
    def __del__(self):
        """Cleanup on deletion"""
        self.close()