"""
Main grid analysis workflow orchestration
"""
from typing import Tuple, Optional
from db_config import DatabaseConfig
from sql_executor import SQLExecutor
import logging
from datetime import datetime

class GridAnalysisWorkflow:
    """Orchestrates the complete grid analysis workflow"""
    
    def __init__(self, prefix: str, building1_id: int, building2_id: int,
                 clean_start: bool = True):
        self.prefix = prefix
        self.building1_id = building1_id
        self.building2_id = building2_id
        self.clean_start = clean_start
        
        # Initialize database and SQL executor
        self.db = DatabaseConfig()
        self.executor = SQLExecutor(self.db, prefix)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Track workflow state
        self.workflow_state = {
            'clip': False,
            'step1': False,
            'step2_3': False,
            'step4_8': False,
            'summary': False,
            'analysis': False
        }
        
    def validate_buildings(self) -> bool:
        """Validate that both building IDs exist"""
        self.logger.info(f"Validating building IDs: {self.building1_id}, {self.building2_id}")
        
        building1 = self.db.get_building_info(self.building1_id)
        building2 = self.db.get_building_info(self.building2_id)
        
        if not building1:
            self.logger.error(f"Building {self.building1_id} not found")
            return False
        if not building2:
            self.logger.error(f"Building {self.building2_id} not found")
            return False
            
        self.building1_info = building1
        self.building2_info = building2
        
        self.logger.info(f"✓ Building 1: {self.building1_id} - Centroid: ({building1['centroid_x']:.2f}, {building1['centroid_y']:.2f})")
        self.logger.info(f"✓ Building 2: {self.building2_id} - Centroid: ({building2['centroid_x']:.2f}, {building2['centroid_y']:.2f})")
        
        return True
    
    def execute_clip(self):
        """Execute the clipping step"""
        self.logger.info("=" * 60)
        self.logger.info("STEP: CLIPPING DATA")
        self.logger.info("=" * 60)
        
        # Get bounding box coordinates
        xmin = min(self.building1_info['centroid_x'], self.building2_info['centroid_x'])
        xmax = max(self.building1_info['centroid_x'], self.building2_info['centroid_x'])
        ymin = min(self.building1_info['centroid_y'], self.building2_info['centroid_y'])
        ymax = max(self.building1_info['centroid_y'], self.building2_info['centroid_y'])
        
        clip_sql = self.generate_clip_sql(xmin, ymin, xmax, ymax)
        
        self.executor.execute_sql_script(
            clip_sql, 
            step='clip',
            description=f"Clipping data between buildings {self.building1_id} and {self.building2_id}"
        )
        
        self.workflow_state['clip'] = True
        self.logger.info("✓ Clipping completed successfully")
    
    def generate_clip_sql(self, xmin: float, ymin: float, xmax: float, ymax: float) -> str:
        """Generate the clipping SQL"""
        return f"""
        -- Create clipping box table
        CREATE TABLE IF NOT EXISTS amin_grid.tlip_box (
            id SERIAL PRIMARY KEY,
            geom geometry(Polygon, 28992)
        );

        -- Clear and insert new clipping box
        TRUNCATE amin_grid.tlip_box;
        
        INSERT INTO amin_grid.tlip_box (geom)
        SELECT ST_MakeEnvelope(
            {xmin}, {ymin}, {xmax}, {ymax}, 28992
        );

        -- Drop existing clipped tables if they exist
        DROP TABLE IF EXISTS amin_grid.tlip_onderstations CASCADE;
        DROP TABLE IF EXISTS amin_grid.tlip_middenspanningsinstallaties CASCADE;
        DROP TABLE IF EXISTS amin_grid.tlip_laagspanningsverdeelkasten CASCADE;
        DROP TABLE IF EXISTS amin_grid.tlip_middenspanningskabels CASCADE;
        DROP TABLE IF EXISTS amin_grid.tlip_laagspanningskabels CASCADE;
        DROP TABLE IF EXISTS amin_grid.tlip_hoogspanningskabels CASCADE;
        DROP TABLE IF EXISTS amin_grid.tlip_buildings_1_deducted CASCADE;

        -- Clip onderstations
        CREATE TABLE amin_grid.tlip_onderstations AS
        SELECT 
            o.*,
            ST_Intersection(o.geom, t.geom) as clipped_geom
        FROM amin_grid.onderstations o
        JOIN amin_grid.tlip_box t ON ST_Intersects(o.geom, t.geom);

        -- Clip middenspanningsinstallaties
        CREATE TABLE amin_grid.tlip_middenspanningsinstallaties AS
        SELECT 
            m.*,
            ST_Intersection(m.geom, t.geom) as clipped_geom
        FROM amin_grid.middenspanningsinstallaties m
        JOIN amin_grid.tlip_box t ON ST_Intersects(m.geom, t.geom);

        -- Clip laagspanningsverdeelkasten
        CREATE TABLE amin_grid.tlip_laagspanningsverdeelkasten AS
        SELECT 
            l.*,
            ST_Intersection(l.geom, t.geom) as clipped_geom
        FROM amin_grid.laagspanningsverdeelkasten l
        JOIN amin_grid.tlip_box t ON ST_Intersects(l.geom, t.geom);

        -- Clip middenspanningskabels
        CREATE TABLE amin_grid.tlip_middenspanningskabels AS
        SELECT 
            m.*,
            ST_Intersection(m.geom, t.geom) as clipped_geom
        FROM amin_grid.middenspanningskabels m
        JOIN amin_grid.tlip_box t ON ST_Intersects(m.geom, t.geom);

        -- Clip laagspanningskabels
        CREATE TABLE amin_grid.tlip_laagspanningskabels AS
        SELECT 
            l.*,
            ST_Intersection(l.geom, t.geom) as clipped_geom
        FROM amin_grid.laagspanningskabels l
        JOIN amin_grid.tlip_box t ON ST_Intersects(l.geom, t.geom);

        -- Clip hoogspanningskabels
        CREATE TABLE amin_grid.tlip_hoogspanningskabels AS
        SELECT 
            h.*,
            ST_Intersection(h.geom, t.geom) as clipped_geom
        FROM amin_grid.hoogspanningskabels h
        JOIN amin_grid.tlip_box t ON ST_Intersects(h.geom, t.geom);

        -- Clip buildings
        CREATE TABLE amin_grid.tlip_buildings_1_deducted AS
        SELECT 
            b.*,
            ST_Intersection(b.pand_geom, t.geom) as clipped_geom
        FROM amin.buildings_1_deducted b
        JOIN amin_grid.tlip_box t ON ST_Intersects(b.pand_geom, t.geom);

        -- Verify clip results
        SELECT 'Clipping Results:' as info;
        SELECT 'onderstations' as table_name, COUNT(*) as count FROM amin_grid.tlip_onderstations
        UNION ALL
        SELECT 'middenspanningsinstallaties', COUNT(*) FROM amin_grid.tlip_middenspanningsinstallaties
        UNION ALL
        SELECT 'laagspanningsverdeelkasten', COUNT(*) FROM amin_grid.tlip_laagspanningsverdeelkasten
        UNION ALL
        SELECT 'middenspanningskabels', COUNT(*) FROM amin_grid.tlip_middenspanningskabels
        UNION ALL
        SELECT 'laagspanningskabels', COUNT(*) FROM amin_grid.tlip_laagspanningskabels
        UNION ALL
        SELECT 'hoogspanningskabels', COUNT(*) FROM amin_grid.tlip_hoogspanningskabels
        UNION ALL
        SELECT 'buildings', COUNT(*) FROM amin_grid.tlip_buildings_1_deducted;
        """
    
    def execute_step1(self):
        """Execute Step 1: Cable Segments and Groups"""
        if not self.workflow_state['clip']:
            self.logger.warning("Clipping not completed, running it first...")
            self.execute_clip()
        
        self.logger.info("=" * 60)
        self.logger.info("STEP 1: CABLE SEGMENTS AND GROUPS")
        self.logger.info("=" * 60)
        
        # Read and execute Step 1 SQL
        step1_sql = self.read_sql_file('STEP 1.sql')
        if step1_sql:
            self.executor.execute_sql_script(
                step1_sql,
                step='s1',
                description="Creating cable segments and connected groups"
            )
            self.workflow_state['step1'] = True
            self.logger.info("✓ Step 1 completed successfully")
    
    def execute_steps_2_3(self):
        """Execute Steps 2-3: Hierarchical Connections"""
        if not self.workflow_state['step1']:
            self.logger.warning("Step 1 not completed, running it first...")
            self.execute_step1()
        
        self.logger.info("=" * 60)
        self.logger.info("STEPS 2-3: HIERARCHICAL CONNECTIONS")
        self.logger.info("=" * 60)
        
        # Read and execute Steps 2-3 SQL
        steps23_sql = self.read_sql_file('STEPS 2-3.sql')
        if steps23_sql:
            self.executor.execute_sql_script(
                steps23_sql,
                step='s23',
                description="Establishing hierarchical connections"
            )
            self.workflow_state['step2_3'] = True
            self.logger.info("✓ Steps 2-3 completed successfully")
    
    def execute_steps_4_8(self):
        """Execute Steps 4-8: Building Connections"""
        if not self.workflow_state['step2_3']:
            self.logger.warning("Steps 2-3 not completed, running them first...")
            self.execute_steps_2_3()
        
        self.logger.info("=" * 60)
        self.logger.info("STEPS 4-8: BUILDING CONNECTIONS")
        self.logger.info("=" * 60)
        
        # Read and execute Steps 4-8 SQL
        steps48_sql = self.read_sql_file('STEPS 4-8.sql')
        if steps48_sql:
            self.executor.execute_sql_script(
                steps48_sql,
                step='s4',
                description="Creating building connections"
            )
            self.workflow_state['step4_8'] = True
            self.logger.info("✓ Steps 4-8 completed successfully")
    
    def execute_summary(self):
        """Execute Hierarchical Grid Summary"""
        if not self.workflow_state['step4_8']:
            self.logger.warning("Steps 4-8 not completed, running them first...")
            self.execute_steps_4_8()
        
        self.logger.info("=" * 60)
        self.logger.info("HIERARCHICAL GRID SUMMARY")
        self.logger.info("=" * 60)
        
        summary_sql = self.read_sql_file('HIERARCHICAL ELECTRICAL GRID SUMMARY.sql')
        if summary_sql:
            self.executor.execute_sql_script(
                summary_sql,
                step='summary',
                description="Generating hierarchical grid summary"
            )
            self.workflow_state['summary'] = True
            self.logger.info("✓ Summary completed successfully")
    
    def execute_analysis(self):
        """Execute MV-LV Based Analysis"""
        if not self.workflow_state['step4_8']:
            self.logger.warning("Steps 4-8 not completed, running them first...")
            self.execute_steps_4_8()
        
        self.logger.info("=" * 60)
        self.logger.info("MV-LV BASED ANALYSIS")
        self.logger.info("=" * 60)
        
        analysis_sql = self.read_sql_file('MV-LV-Based Analysis.sql')
        if analysis_sql:
            self.executor.execute_sql_script(
                analysis_sql,
                step='analysis',
                description="Performing MV-LV based analysis"
            )
            self.workflow_state['analysis'] = True
            self.logger.info("✓ Analysis completed successfully")
    
    def run_complete_workflow(self):
        """Execute the complete workflow"""
        start_time = datetime.now()
        
        self.logger.info("=" * 60)
        self.logger.info(f"STARTING GRID ANALYSIS WORKFLOW")
        self.logger.info(f"Prefix: {self.prefix}")
        self.logger.info(f"Buildings: {self.building1_id} <-> {self.building2_id}")

        self.logger.info("=" * 60)
        
        # Validate buildings
        if not self.validate_buildings():
            self.logger.error("Building validation failed. Aborting workflow.")
            return False
        
        # Clean existing tables if requested
        if self.clean_start:
            self.logger.info("Cleaning existing tables...")
            self.executor.clean_existing_tables()
        
        try:
            # Execute all steps in sequence
            self.execute_clip()
            self.execute_step1()
            self.execute_steps_2_3()
            self.execute_steps_4_8()
            self.execute_summary()
            self.execute_analysis()
            
            # Get final statistics
            stats = self.executor.get_summary_stats()
            
            # Print summary
            elapsed = datetime.now() - start_time
            self.logger.info("=" * 60)
            self.logger.info("WORKFLOW COMPLETED SUCCESSFULLY")
            self.logger.info(f"Total time: {elapsed}")
            
            if stats and 'buildings' in stats:
                self.logger.info(f"Buildings connected: {stats['buildings']['total']}")
                self.logger.info(f"  - Residential: {stats['buildings']['residential']}")
                self.logger.info(f"  - Non-residential: {stats['buildings']['non_residential']}")
                self.logger.info(f"  - MV Capable: {stats['buildings']['mv_capable']}")
            
            self.logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            self.logger.error("Check logs for details")
            return False
        
        finally:
            self.db.disconnect()
    
    def read_sql_file(self, filename: str) -> Optional[str]:
        """Read SQL file content (placeholder - replace with actual file reading)"""
        # In production, read from actual files
        # For now, return None to indicate file should be provided
        self.logger.warning(f"SQL file '{filename}' not found - using embedded SQL")
        return None
    
    def get_workflow_status(self) -> dict:
        """Get current workflow status"""
        return self.workflow_state.copy()