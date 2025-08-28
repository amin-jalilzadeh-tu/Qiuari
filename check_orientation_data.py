"""
Check orientation data in the correct PostgreSQL table
"""
import psycopg2
import pandas as pd

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
    
    print("Checking orientation data in PostgreSQL")
    print("=" * 60)
    
    # Use the Amsterdam table which seems to be the main one
    table_name = "amsterdam_buildings_1_deducted"
    schema_name = "amin_grid"
    
    print(f"\nTable: {schema_name}.{table_name}")
    
    # Check orientation columns
    print("\n1. ORIENTATION DATA:")
    query = f"""
    SELECT 
        COUNT(*) as total_buildings,
        COUNT(building_orientation) as has_orientation,
        COUNT(building_orientation_cardinal) as has_cardinal,
        COUNT(DISTINCT building_orientation_cardinal) as unique_cardinals
    FROM {schema_name}.{table_name};
    """
    
    cur.execute(query)
    result = cur.fetchone()
    
    print(f"  Total buildings: {result[0]}")
    print(f"  Has building_orientation: {result[1]}")
    print(f"  Has building_orientation_cardinal: {result[2]}")
    print(f"  Unique cardinal directions: {result[3]}")
    
    # Get unique cardinal values
    print("\n2. CARDINAL DIRECTIONS DISTRIBUTION:")
    query = f"""
    SELECT 
        building_orientation_cardinal,
        COUNT(*) as count
    FROM {schema_name}.{table_name}
    WHERE building_orientation_cardinal IS NOT NULL
    GROUP BY building_orientation_cardinal
    ORDER BY count DESC;
    """
    
    cur.execute(query)
    directions = cur.fetchall()
    
    for direction, count in directions:
        print(f"  {direction}: {count} buildings")
    
    # Check numeric orientation values
    print("\n3. NUMERIC ORIENTATION VALUES:")
    query = f"""
    SELECT 
        MIN(building_orientation) as min_orientation,
        MAX(building_orientation) as max_orientation,
        AVG(building_orientation) as avg_orientation,
        STDDEV(building_orientation) as std_orientation
    FROM {schema_name}.{table_name}
    WHERE building_orientation IS NOT NULL;
    """
    
    cur.execute(query)
    result = cur.fetchone()
    
    if result[0] is not None:
        print(f"  Min: {result[0]:.1f}°")
        print(f"  Max: {result[1]:.1f}°")
        print(f"  Average: {result[2]:.1f}°")
        print(f"  Std Dev: {result[3]:.1f}°")
    
    # Check roof area data
    print("\n4. ROOF AREA DATA:")
    query = f"""
    SELECT 
        COUNT(flat_roof_area) as has_flat_roof,
        COUNT(sloped_roof_area) as has_sloped_roof,
        COUNT(suitable_roof_area) as has_suitable_roof
    FROM {schema_name}.{table_name};
    """
    
    cur.execute(query)
    result = cur.fetchone()
    
    print(f"  Has flat_roof_area: {result[0]}")
    print(f"  Has sloped_roof_area: {result[1]}")
    print(f"  Has suitable_roof_area: {result[2]}")
    
    # Sample data with orientation
    print("\n5. SAMPLE BUILDINGS WITH ORIENTATION:")
    query = f"""
    SELECT 
        ogc_fid,
        building_orientation,
        building_orientation_cardinal,
        energy_label,
        flat_roof_area,
        sloped_roof_area,
        suitable_roof_area,
        has_solar
    FROM {schema_name}.{table_name}
    WHERE building_orientation_cardinal IS NOT NULL
    LIMIT 5;
    """
    
    cur.execute(query)
    samples = cur.fetchall()
    
    for sample in samples:
        print(f"\n  Building {sample[0]}:")
        print(f"    Orientation: {sample[1]:.1f}° ({sample[2]})")
        print(f"    Energy Label: {sample[3]}")
        print(f"    Roof Areas - Flat: {sample[4]}, Sloped: {sample[5]}, Suitable: {sample[6]}")
        print(f"    Has Solar: {sample[7]}")
    
    conn.close()
    print("\n" + "=" * 60)
    print("Orientation data EXISTS in PostgreSQL!")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()