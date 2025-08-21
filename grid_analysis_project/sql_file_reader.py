"""
SQL file reading utilities
This module handles reading SQL files from disk and provides
the SQL content to the workflow.
"""
import os
from typing import Dict, Optional
import logging

class SQLFileReader:
    """Manages reading SQL files from disk"""
    
    def __init__(self, sql_directory: str = "sql_scripts"):
        self.sql_directory = sql_directory
        self.logger = logging.getLogger(__name__)
        self.sql_cache = {}
        
        # Define SQL file mappings
        self.sql_files = {
            'clip': 'Clip.sql',
            'step1': 'STEP 1.sql',
            'steps23': 'STEPS 2-3.sql',
            'steps48': 'STEPS 4-8.sql',
            'summary': 'HIERARCHICAL ELECTRICAL GRID SUMMARY.sql',
            'analysis': 'MV-LV-Based Analysis.sql'
        }
        
        # Verify SQL directory exists
        if not os.path.exists(self.sql_directory):
            self.logger.warning(f"SQL directory '{self.sql_directory}' not found. Creating it...")
            os.makedirs(self.sql_directory)
    
    def read_sql_file(self, step_name: str) -> Optional[str]:
        """Read SQL content for a specific step"""
        
        # Check cache first
        if step_name in self.sql_cache:
            return self.sql_cache[step_name]
        
        # Get file name
        if step_name not in self.sql_files:
            self.logger.error(f"Unknown step: {step_name}")
            return None
        
        file_name = self.sql_files[step_name]
        file_path = os.path.join(self.sql_directory, file_name)
        
        # Check if file exists
        if not os.path.exists(file_path):
            self.logger.error(f"SQL file not found: {file_path}")
            return None
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Cache the content
            self.sql_cache[step_name] = content
            
            self.logger.info(f"Loaded SQL file: {file_name} ({len(content)} bytes)")
            return content
            
        except Exception as e:
            self.logger.error(f"Error reading SQL file {file_path}: {e}")
            return None
    
    def validate_sql_files(self) -> Dict[str, bool]:
        """Check which SQL files are available"""
        status = {}
        
        for step, filename in self.sql_files.items():
            file_path = os.path.join(self.sql_directory, filename)
            exists = os.path.exists(file_path)
            status[step] = exists
            
            if exists:
                size = os.path.getsize(file_path)
                self.logger.info(f"✓ {filename}: {size:,} bytes")
            else:
                self.logger.warning(f"✗ {filename}: NOT FOUND")
        
        return status
    
    def create_sample_sql_files(self):
        """Create sample SQL files with basic structure"""
        
        samples = {
            'Clip.sql': """-- Clipping SQL
-- This file should contain the clipping logic
-- Placeholder content - replace with actual SQL

SELECT 'Replace this with actual Clip.sql content' as message;
""",
            'STEP 1.sql': """-- Step 1: Cable Segments
-- This file should contain Step 1 logic
-- Placeholder content - replace with actual SQL

SELECT 'Replace this with actual STEP 1.sql content' as message;
""",
            'STEPS 2-3.sql': """-- Steps 2-3: Hierarchical Connections
-- This file should contain Steps 2-3 logic
-- Placeholder content - replace with actual SQL

SELECT 'Replace this with actual STEPS 2-3.sql content' as message;
""",
            'STEPS 4-8.sql': """-- Steps 4-8: Building Connections
-- This file should contain Steps 4-8 logic
-- Placeholder content - replace with actual SQL

SELECT 'Replace this with actual STEPS 4-8.sql content' as message;
""",
            'HIERARCHICAL ELECTRICAL GRID SUMMARY.sql': """-- Hierarchical Grid Summary
-- This file should contain summary logic
-- Placeholder content - replace with actual SQL

SELECT 'Replace this with actual summary SQL content' as message;
""",
            'MV-LV-Based Analysis.sql': """-- MV-LV Based Analysis
-- This file should contain analysis logic
-- Placeholder content - replace with actual SQL

SELECT 'Replace this with actual analysis SQL content' as message;
"""
        }
        
        created = 0
        for filename, content in samples.items():
            file_path = os.path.join(self.sql_directory, filename)
            
            if not os.path.exists(file_path):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"Created sample file: {filename}")
                created += 1
            else:
                self.logger.info(f"File already exists: {filename}")
        
        if created > 0:
            self.logger.info(f"Created {created} sample SQL files in '{self.sql_directory}/'")
            self.logger.info("Please replace these with your actual SQL files")
    
    def get_clip_sql_template(self, xmin: float, ymin: float, 
                             xmax: float, ymax: float) -> str:
        """
        Generate parameterized clipping SQL
        This can be used if the Clip.sql file needs dynamic bounds
        """
        
        # Try to read the template first
        template = self.read_sql_file('clip')
        
        if template:
            # Replace placeholders with actual values
            replacements = {
                '{{XMIN}}': str(xmin),
                '{{YMIN}}': str(ymin),
                '{{XMAX}}': str(xmax),
                '{{YMAX}}': str(ymax),
                '{{SRID}}': '28992'
            }
            
            for placeholder, value in replacements.items():
                template = template.replace(placeholder, value)
            
            return template
        
        # Return a default clipping SQL if no template found
        return self.get_default_clip_sql(xmin, ymin, xmax, ymax)
    
    def get_default_clip_sql(self, xmin: float, ymin: float, 
                            xmax: float, ymax: float) -> str:
        """Generate default clipping SQL"""
        return f"""
        -- Auto-generated clipping SQL
        -- Bounds: ({xmin:.2f}, {ymin:.2f}) to ({xmax:.2f}, {ymax:.2f})
        
        -- Create clipping box
        CREATE TABLE IF NOT EXISTS amin_grid.tlip_box (
            id SERIAL PRIMARY KEY,
            geom geometry(Polygon, 28992)
        );
        
        TRUNCATE amin_grid.tlip_box;
        
        INSERT INTO amin_grid.tlip_box (geom)
        SELECT ST_MakeEnvelope({xmin}, {ymin}, {xmax}, {ymax}, 28992);
        
        -- Add your clipping logic here
        SELECT 'Clipping area created' as status;
        """


# Modified grid_analysis.py integration
def integrate_with_workflow(workflow_instance):
    """
    Integrate SQLFileReader with GridAnalysisWorkflow
    Add this to your grid_analysis.py
    """
    
    # Import at the top of grid_analysis.py:
    # from sql_file_reader import SQLFileReader
    
    # In GridAnalysisWorkflow.__init__, add:
    # self.sql_reader = SQLFileReader()
    
    # Replace the read_sql_file method:
    def read_sql_file(self, filename: str) -> Optional[str]:
        """Read SQL file content using SQLFileReader"""
        # Map filename to step name
        step_map = {
            'Clip.sql': 'clip',
            'STEP 1.sql': 'step1',
            'STEPS 2-3.sql': 'steps23',
            'STEPS 4-8.sql': 'steps48',
            'HIERARCHICAL ELECTRICAL GRID SUMMARY.sql': 'summary',
            'MV-LV-Based Analysis.sql': 'analysis'
        }
        
        step_name = step_map.get(filename)
        if not step_name:
            self.logger.error(f"Unknown SQL file: {filename}")
            return None
        
        return self.sql_reader.read_sql_file(step_name)
    
    # For the clipping step, use:
    # clip_sql = self.sql_reader.get_clip_sql_template(xmin, ymin, xmax, ymax)
    
    workflow_instance.read_sql_file = read_sql_file


if __name__ == "__main__":
    """Test the SQL file reader"""
    
    logging.basicConfig(level=logging.INFO)
    reader = SQLFileReader()
    
    print("\n" + "=" * 60)
    print("SQL File Reader Test")
    print("=" * 60)
    
    # Check which files exist
    print("\nChecking SQL files...")
    status = reader.validate_sql_files()
    
    # Count missing files
    missing = [step for step, exists in status.items() if not exists]
    
    if missing:
        print(f"\nMissing {len(missing)} SQL files")
        response = input("Create sample SQL files? (y/n): ").strip().lower()
        
        if response == 'y':
            reader.create_sample_sql_files()
            print("\nSample files created. Please replace with actual SQL content.")
    else:
        print("\nAll SQL files found!")
    
    # Test reading a file
    print("\nTesting file reading...")
    content = reader.read_sql_file('step1')
    if content:
        print(f"Successfully read Step 1 SQL ({len(content)} bytes)")
    
    # Test clip SQL generation
    print("\nTesting clip SQL generation...")
    clip_sql = reader.get_clip_sql_template(100000, 400000, 120000, 420000)
    if clip_sql:
        print(f"Generated clipping SQL ({len(clip_sql)} bytes)")