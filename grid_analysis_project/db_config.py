"""
Database configuration and connection management
"""
import psycopg2
from psycopg2 import sql
from typing import Optional, Dict, Any
import logging

class DatabaseConfig:
    """Database configuration and connection management"""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 5433,
                 database: str = "research",
                 user: str = "aminj",
                 password: str = "Aminej@geodan!"):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.connection = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def connect(self) -> psycopg2.extensions.connection:
        """Establish database connection"""
        try:
            if self.connection and not self.connection.closed:
                return self.connection
                
            self.connection = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password
            )
            self.logger.info(f"Connected to database: {self.database}")
            return self.connection
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and not self.connection.closed:
            self.connection.close()
            self.logger.info("Database connection closed")
    
    def execute_query(self, query: str, params: Optional[tuple] = None, 
                     fetch: bool = False) -> Optional[list]:
        """Execute a SQL query"""
        conn = self.connect()
        cursor = conn.cursor()
        
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch:
                result = cursor.fetchall()
                return result
            else:
                conn.commit()
                return None
                
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Query execution failed: {e}")
            raise
        finally:
            cursor.close()
    
    def check_table_exists(self, schema: str, table_name: str) -> bool:
        """Check if a table exists in the database"""
        query = """
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = %s AND table_name = %s
            )
        """
        result = self.execute_query(query, (schema, table_name), fetch=True)
        return result[0][0] if result else False
    
    def get_building_info(self, ogc_fid: int) -> Dict[str, Any]:
        """Get building information for a given ogc_fid"""
        query = """
            SELECT 
                ogc_fid,
                ST_AsText(ST_Centroid(pand_geom)) as centroid,
                ST_XMin(pand_geom) as xmin,
                ST_YMin(pand_geom) as ymin,
                ST_XMax(pand_geom) as xmax,
                ST_YMax(pand_geom) as ymax,
                ST_X(ST_Centroid(pand_geom)) as centroid_x,
                ST_Y(ST_Centroid(pand_geom)) as centroid_y
            FROM amin.buildings_1_deducted
            WHERE ogc_fid = %s
        """
        result = self.execute_query(query, (ogc_fid,), fetch=True)
        
        if result:
            row = result[0]
            return {
                'ogc_fid': row[0],
                'centroid_wkt': row[1],
                'xmin': row[2],
                'ymin': row[3],
                'xmax': row[4],
                'ymax': row[5],
                'centroid_x': row[6],
                'centroid_y': row[7]
            }
        return None