"""
Check PostgreSQL database for orientation data
"""
import psycopg2
import pandas as pd

# Connection parameters
conn_params = {
    "host": "localhost",
    "port": 5433,
    "database": "research",
    "user": "aminj",
    "password": "Aminej@geodan!",
}

try:
    # Connect to PostgreSQL
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    
    print("Connected to PostgreSQL database")
    print("=" * 60)
    
    # Check table structure
    print("\n1. CHECKING TABLE STRUCTURE:")
    query_columns = """
    SELECT column_name, data_type 
    FROM information_schema.columns 
    WHERE table_schema = 'amin_grid' 
    AND table_name = 'buildings_1_deducted'
    ORDER BY ordinal_position;
    """
    
    cur.execute(query_columns)
    columns = cur.fetchall()
    
    print("\nColumns in buildings_1_deducted:")
    for col_name, col_type in columns:
        if 'orient' in col_name.lower() or 'azimuth' in col_name.lower() or 'direction' in col_name.lower():
            print(f"  * {col_name}: {col_type} <-- ORIENTATION FIELD")
        else:
            print(f"    {col_name}: {col_type}")
    
    # Check for orientation-related columns
    print("\n2. ORIENTATION-RELATED COLUMNS:")
    orientation_cols = [col[0] for col in columns if 'orient' in col[0].lower() or 'azimuth' in col[0].lower()]
    
    if orientation_cols:
        print(f"Found orientation columns: {orientation_cols}")
        
        # Check data in orientation columns
        for col in orientation_cols:
            query_check = f"""
            SELECT 
                COUNT(*) as total,
                COUNT({col}) as non_null,
                COUNT(DISTINCT {col}) as unique_values
            FROM amin_grid.buildings_1_deducted;
            """
            cur.execute(query_check)
            result = cur.fetchone()
            print(f"\n  {col}:")
            print(f"    Total rows: {result[0]}")
            print(f"    Non-null values: {result[1]}")
            print(f"    Unique values: {result[2]}")
            
            # Get sample values
            query_sample = f"""
            SELECT DISTINCT {col}
            FROM amin_grid.buildings_1_deducted
            WHERE {col} IS NOT NULL
            LIMIT 10;
            """
            cur.execute(query_sample)
            samples = cur.fetchall()
            if samples:
                print(f"    Sample values: {[s[0] for s in samples]}")
    else:
        print("No orientation columns found!")
    
    # Check roof-related columns
    print("\n3. ROOF-RELATED COLUMNS:")
    roof_cols = [col[0] for col in columns if 'roof' in col[0].lower()]
    
    if roof_cols:
        print(f"Found roof columns: {roof_cols}")
        for col in roof_cols[:5]:  # Check first 5
            query_check = f"""
            SELECT 
                COUNT({col}) as non_null,
                MIN({col}) as min_val,
                MAX({col}) as max_val,
                AVG({col}) as avg_val
            FROM amin_grid.buildings_1_deducted
            WHERE {col} IS NOT NULL;
            """
            try:
                cur.execute(query_check)
                result = cur.fetchone()
                if result[0] > 0:
                    print(f"  {col}: {result[0]} non-null, range [{result[1]:.1f} - {result[2]:.1f}], avg: {result[3]:.1f}")
            except:
                pass
    
    # Check solar-related columns
    print("\n4. SOLAR-RELATED COLUMNS:")
    solar_cols = [col[0] for col in columns if 'solar' in col[0].lower() or 'pv' in col[0].lower()]
    
    if solar_cols:
        print(f"Found solar columns: {solar_cols}")
    
    # Sample full data
    print("\n5. SAMPLE BUILDING DATA:")
    query_sample = """
    SELECT *
    FROM amin_grid.buildings_1_deducted
    LIMIT 1;
    """
    
    # Use pandas for better display
    df = pd.read_sql_query(query_sample, conn)
    
    # Find orientation column if it exists
    orient_cols = [c for c in df.columns if 'orient' in c.lower()]
    roof_cols = [c for c in df.columns if 'roof' in c.lower()]
    solar_cols = [c for c in df.columns if 'solar' in c.lower()]
    
    print("\nSample building (relevant fields):")
    for col in orient_cols + roof_cols + solar_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].iloc[0]}")
    
    conn.close()
    print("\n" + "=" * 60)
    print("Database check complete!")
    
except Exception as e:
    print(f"Error connecting to database: {e}")
    import traceback
    traceback.print_exc()