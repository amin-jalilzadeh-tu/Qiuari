"""
SQL execution utilities with table prefix management
"""
import re
from typing import List, Dict, Any, Optional
from db_config import DatabaseConfig
import logging

class SQLExecutor:
    """Handles SQL execution with dynamic table naming"""
    
    def __init__(self, db_config: DatabaseConfig, prefix: str):
        self.db = db_config
        self.prefix = prefix
        self.logger = logging.getLogger(__name__)
        self.table_mappings = {}
        self.schema = "amin_grid"
        
    def get_prefixed_table_name(self, base_name: str, step: Optional[str] = None) -> str:
        """Generate prefixed table name with optional step identifier"""
        if step:
            return f"{self.prefix}_{step}_{base_name}"
        return f"{self.prefix}_{base_name}"
    
    def replace_table_names(self, sql_content: str, step: Optional[str] = None) -> str:
        """Replace table names in SQL with prefixed versions"""
        # Define base table names that need prefixing
        base_tables = [
            'tlip_box',
            'tlip_onderstations',
            'tlip_middenspanningsinstallaties',
            'tlip_laagspanningsverdeelkasten',
            'tlip_middenspanningskabels',
            'tlip_laagspanningskabels',
            'tlip_hoogspanningskabels',
            'tlip_buildings_1_deducted',
            'tlip_cable_segments',
            'tlip_segment_endpoints',
            'tlip_connected_groups',
            'tlip_group_segments',
            'tlip_group_stations',
            'tlip_group_hierarchy',
            'tlip_voltage_transitions',
            'tlip_building_connections',
            'tlip_building_connection_points',
            'tlip_segment_connections',
            'tlip_grid_summary',
            'v_tlip_hierarchy_tree',
            'v_tlip_grid_overview',
            'v_tlip_connection_types',
            'v_tlip_problematic_connections',
            'v_tlip_group_summary'
        ]
        
        # Replace each table name with prefixed version
        modified_sql = sql_content
        for table in base_tables:
            # Create the new table name
            new_name = self.get_prefixed_table_name(table.replace('tlip_', ''), step)
            
            # Replace in various contexts (CREATE, DROP, INSERT, SELECT, etc.)
            patterns = [
                (f'amin_grid\\.{table}', f'amin_grid.{new_name}'),
                (f'"{table}"', f'"{new_name}"'),
                (f"'{table}'", f"'{new_name}'"),
                (f' {table} ', f' {new_name} '),
                (f' {table}(', f' {new_name}('),
                (f'\\n{table} ', f'\\n{new_name} '),
            ]
            
            for pattern, replacement in patterns:
                modified_sql = re.sub(pattern, replacement, modified_sql)
            
            # Store mapping for reference
            self.table_mappings[table] = new_name
        
        return modified_sql
    
    def clean_existing_tables(self, step: Optional[str] = None):
        """Drop existing prefixed tables for a clean start"""
        self.logger.info(f"Cleaning existing tables with prefix: {self.prefix}")
        
        # Get list of tables to clean
        query = """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = %s 
            AND table_name LIKE %s
        """
        
        pattern = f"{self.prefix}_%"
        if step:
            pattern = f"{self.prefix}_{step}_%"
            
        tables = self.db.execute_query(query, (self.schema, pattern), fetch=True)
        
        if tables:
            for table in tables:
                table_name = table[0]
                drop_query = f"DROP TABLE IF EXISTS {self.schema}.{table_name} CASCADE"
                self.db.execute_query(drop_query)
                self.logger.info(f"Dropped table: {table_name}")
        
        # Also drop views
        query_views = """
            SELECT table_name 
            FROM information_schema.views 
            WHERE table_schema = %s 
            AND table_name LIKE %s
        """
        views = self.db.execute_query(query_views, (self.schema, pattern), fetch=True)
        
        if views:
            for view in views:
                view_name = view[0]
                drop_query = f"DROP VIEW IF EXISTS {self.schema}.{view_name} CASCADE"
                self.db.execute_query(drop_query)
                self.logger.info(f"Dropped view: {view_name}")
    
    def execute_sql_script(self, sql_content: str, step: Optional[str] = None, 
                          description: str = ""):
        """Execute SQL script with proper error handling"""
        self.logger.info(f"Executing: {description or 'SQL script'}")
        
        # Replace table names with prefixed versions
        modified_sql = self.replace_table_names(sql_content, step)
        
        # Split by semicolons but preserve those within functions/procedures
        statements = self.split_sql_statements(modified_sql)
        
        conn = self.db.connect()
        cursor = conn.cursor()
        
        try:
            for i, statement in enumerate(statements, 1):
                statement = statement.strip()
                if not statement:
                    continue
                    
                try:
                    cursor.execute(statement)
                    conn.commit()
                    
                    # Log progress for long scripts
                    if len(statements) > 10 and i % 10 == 0:
                        self.logger.info(f"  Progress: {i}/{len(statements)} statements executed")
                        
                except Exception as e:
                    self.logger.error(f"Error in statement {i}: {str(e)[:200]}")
                    conn.rollback()
                    # Continue with next statement for non-critical errors
                    if "NOTICE" not in str(e) and "already exists" not in str(e):
                        raise
        
        finally:
            cursor.close()
        
        self.logger.info(f"✓ Completed: {description or 'SQL script'}")
    
    def split_sql_statements(self, sql: str) -> List[str]:
        """Split SQL into individual statements, preserving DO blocks and functions"""
        # Handle DO blocks and CREATE FUNCTION blocks specially
        statements = []
        current = []
        in_do_block = False
        in_function = False
        
        lines = sql.split('\n')
        
        for line in lines:
            line_upper = line.upper().strip()
            
            # Check for DO block start
            if line_upper.startswith('DO $$'):
                in_do_block = True
            
            # Check for CREATE FUNCTION/PROCEDURE start
            if ('CREATE FUNCTION' in line_upper or 'CREATE PROCEDURE' in line_upper or
                'CREATE OR REPLACE FUNCTION' in line_upper or 
                'CREATE OR REPLACE VIEW' in line_upper):
                in_function = True
            
            current.append(line)
            
            # Check for end of DO block
            if in_do_block and line.strip().endswith('$$;'):
                in_do_block = False
                statements.append('\n'.join(current))
                current = []
            
            # Check for end of function/view
            elif in_function and line.strip().endswith(';') and not in_do_block:
                # Simple heuristic: if line ends with ; and we're not in a DO block
                if 'END;' in line_upper or 'END ;' in line_upper or \
                   line.strip() == ';' or 'FROM' in line_upper:
                    in_function = False
                    statements.append('\n'.join(current))
                    current = []
            
            # Regular statement end
            elif not in_do_block and not in_function and line.strip().endswith(';'):
                statements.append('\n'.join(current))
                current = []
        
        # Add any remaining content
        if current:
            statements.append('\n'.join(current))
        
        return [s for s in statements if s.strip()]
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics after processing"""
        stats = {}
        
        # Count buildings connected
        query = f"""
            SELECT 
                COUNT(*) as total_buildings,
                COUNT(CASE WHEN building_function = 'residential' THEN 1 END) as residential,
                COUNT(CASE WHEN building_function = 'non_residential' THEN 1 END) as non_residential,
                COUNT(CASE WHEN is_mv_capable = TRUE THEN 1 END) as mv_capable
            FROM {self.schema}.{self.get_prefixed_table_name('building_connections', 's4')}
        """
        
        try:
            result = self.db.execute_query(query, fetch=True)
            if result:
                stats['buildings'] = {
                    'total': result[0][0],
                    'residential': result[0][1],
                    'non_residential': result[0][2],
                    'mv_capable': result[0][3]
                }
        except:
            pass
        
        return stats