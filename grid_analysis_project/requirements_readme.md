# Grid Analysis Workflow Automation

## Requirements File (requirements.txt)
```
psycopg2-binary==2.9.9
```

## Installation
```bash
pip install -r requirements.txt
```

## Project Structure
```
grid_analysis_project/
│
├── db_config.py          # Database configuration and connection
├── sql_executor.py       # SQL execution with prefix management
├── grid_analysis.py      # Main workflow orchestration
├── main.py              # Entry point and CLI
├── batch_config.json    # Example batch configuration
├── requirements.txt     # Python dependencies
│
└── sql_scripts/         # Place your SQL files here
    ├── Clip.sql
    ├── STEP 1.sql
    ├── STEPS 2-3.sql
    ├── STEPS 4-8.sql
    ├── HIERARCHICAL ELECTRICAL GRID SUMMARY.sql
    └── MV-LV-Based Analysis.sql
```

## Usage Examples

### 1. Interactive Mode
Run without arguments for interactive prompts:
```bash
python main.py
```

### 2. Command Line Mode
Specify all parameters via command line:
```bash
python main.py --building1 4804870 --building2 4794514 --prefix test_run_01 --clean
```

### 3. Test Mode
Run with default test building IDs:
```bash
python main.py --test
```

### 4. Batch Mode
Process multiple areas from configuration file:
```bash
python main.py --batch batch_config.json
```

### 5. List Available Buildings
Get sample building IDs from database:
```bash
python main.py --list-buildings
```

## Table Naming Convention

The system uses a hierarchical naming convention for better organization:

**Format:** `{prefix}_{step}_{table_name}`

Examples:
- `test01_clip_box` - Clipping boundary box
- `test01_clip_buildings_1_deducted` - Clipped buildings
- `test01_s1_cable_segments` - Step 1 cable segments
- `test01_s1_connected_groups` - Step 1 connected groups
- `test01_s23_group_stations` - Steps 2-3 group stations
- `test01_s4_building_connections` - Step 4 building connections
- `test01_summary_grid_hierarchy` - Summary tables
- `test01_analysis_mv_district_metrics` - Analysis tables

## Features

### Automatic Workflow Management
- **Dependency Handling:** Steps automatically run prerequisites if needed
- **Clean Start Option:** Remove existing tables before processing
- **Progress Logging:** Detailed logs with timestamps
- **Error Recovery:** Continue processing on non-critical errors

### Table Prefix Management
- All tables are prefixed for easy identification
- Multiple runs can coexist in the same schema
- Easy cleanup of specific runs

### Building Validation
- Verifies building IDs exist before processing
- Calculates bounding box from building centroids
- Reports building coordinates and area coverage

### Comprehensive Logging
- Console output for real-time monitoring
- File logging for detailed debugging
- Progress indicators for long operations
- Summary statistics at completion

## SQL File Integration

### Option 1: Embedded SQL (Current)
The SQL is embedded in the Python code for portability.

### Option 2: External SQL Files
Modify `grid_analysis.py` to read from files:

```python
def read_sql_file(self, filename: str) -> Optional[str]:
    """Read SQL file content"""
    import os
    sql_path = os.path.join('sql_scripts', filename)
    
    if os.path.exists(sql_path):
        with open(sql_path, 'r') as f:
            return f.read()
    else:
        self.logger.error(f"SQL file not found: {sql_path}")
        return None
```

## Database Configuration

Default connection parameters in `db_config.py`:
- Host: localhost
- Port: 5433
- Database: research
- User: aminj
- Password: (specified in code)

To modify, edit the `DatabaseConfig` class initialization.

## Output Tables

After successful completion, the following table groups are created:

### Clipped Data (prefix_clip_*)
- Infrastructure components within bounding box
- Buildings within area

### Network Analysis (prefix_s1_*, prefix_s23_*)
- Cable segments and groups
- Hierarchical connections
- Voltage transitions

### Building Connections (prefix_s4_*)
- Building-to-grid connections
- Connection types and distances
- MV capability flags

### Analysis Results (prefix_summary_*, prefix_analysis_*)
- Grid hierarchy summaries
- MV-LV district metrics
- Energy community potential
- Intervention priorities

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Check database credentials
   - Verify PostgreSQL is running
   - Check network/firewall settings

2. **Building Not Found**
   - Verify OGC FID exists in database
   - Check schema name (amin.buildings_1_deducted)

3. **Table Already Exists**
   - Use `--clean` flag to remove existing tables
   - Or use different prefix

4. **Insufficient Data in Clipped Area**
   - Buildings may be too close together
   - Try buildings further apart for larger area

### Debug Mode
For detailed debugging, modify logging level in `db_config.py`:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Tips

1. **Index Management**
   - Script automatically creates necessary indexes
   - Monitor PostgreSQL performance for large areas

2. **Batch Processing**
   - Use batch mode for multiple areas
   - Run overnight for large datasets

3. **Memory Usage**
   - Large clipping areas may require more memory
   - Consider processing in smaller chunks

## Example Workflow Output

```
============================================================
STARTING GRID ANALYSIS WORKFLOW
Prefix: test_area_01
Buildings: 4804870 ↔ 4794514
============================================================
Validating building IDs: 4804870, 4794514
✓ Building 1: 4804870 - Centroid: (118998.07, 482179.63)
✓ Building 2: 4794514 - Centroid: (121436.53, 483907.64)
============================================================
STEP: CLIPPING DATA
============================================================
✓ Clipping completed successfully
============================================================
STEP 1: CABLE SEGMENTS AND GROUPS
============================================================
✓ Step 1 completed successfully
[... continues for all steps ...]
============================================================
WORKFLOW COMPLETED SUCCESSFULLY
Total time: 0:02:34
Buildings connected: 485
  - Residential: 412
  - Non-residential: 73
  - MV Capable: 15
============================================================
```