"""
Find correct table names in PostgreSQL
"""
import psycopg2

conn_params = {
    "host": "localhost",
    "port": 5433,
    "database": "research",
    "user": "aminj",
    "password": "Aminej@geodan!",
}

try:
    conn = psycopg2.connect(**conn_params)
    cur = conn.cursor()
    
    print("Connected to PostgreSQL")
    print("=" * 60)
    
    # List all schemas
    print("\n1. AVAILABLE SCHEMAS:")
    cur.execute("""
        SELECT schema_name 
        FROM information_schema.schemata 
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
        ORDER BY schema_name;
    """)
    schemas = cur.fetchall()
    for schema in schemas:
        print(f"  - {schema[0]}")
    
    # Check tables in amin_grid schema
    print("\n2. TABLES IN 'amin_grid' SCHEMA:")
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'amin_grid'
        ORDER BY table_name;
    """)
    tables = cur.fetchall()
    
    if tables:
        for table in tables:
            print(f"  - {table[0]}")
    else:
        print("  No tables found in amin_grid schema")
    
    # Search for building-related tables in all schemas
    print("\n3. SEARCHING FOR BUILDING TABLES IN ALL SCHEMAS:")
    cur.execute("""
        SELECT table_schema, table_name 
        FROM information_schema.tables 
        WHERE table_name LIKE '%building%' 
           OR table_name LIKE '%Building%'
           OR table_name LIKE '%BUILDING%'
        ORDER BY table_schema, table_name;
    """)
    building_tables = cur.fetchall()
    
    if building_tables:
        for schema, table in building_tables:
            print(f"  - {schema}.{table}")
            
            # Check if it has orientation data
            cur.execute(f"""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_schema = '{schema}' 
                AND table_name = '{table}'
                AND (column_name ILIKE '%orient%' 
                     OR column_name ILIKE '%azimuth%'
                     OR column_name ILIKE '%direction%')
                LIMIT 5;
            """)
            orient_cols = cur.fetchall()
            if orient_cols:
                print(f"    Has orientation columns: {[c[0] for c in orient_cols]}")
    else:
        print("  No building tables found")
    
    # Check public schema
    print("\n4. TABLES IN 'public' SCHEMA (common default):")
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
        AND (table_name LIKE '%building%' OR table_name LIKE '%grid%')
        ORDER BY table_name
        LIMIT 20;
    """)
    public_tables = cur.fetchall()
    
    if public_tables:
        for table in public_tables:
            print(f"  - public.{table[0]}")
    
    # Search in all schemas for deducted tables
    print("\n5. SEARCHING FOR 'deducted' TABLES:")
    cur.execute("""
        SELECT table_schema, table_name 
        FROM information_schema.tables 
        WHERE table_name LIKE '%deducted%'
        ORDER BY table_schema, table_name;
    """)
    deducted_tables = cur.fetchall()
    
    if deducted_tables:
        for schema, table in deducted_tables:
            print(f"  - {schema}.{table}")
    else:
        print("  No 'deducted' tables found")
    
    conn.close()
    print("\n" + "=" * 60)
    
except Exception as e:
    print(f"Error: {e}")