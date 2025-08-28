"""
Check actual KG data structure to understand what fields we have
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from data.kg_connector import KGConnector
import yaml

# Load config to get correct credentials
with open('config/unified_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

kg = KGConnector(
    uri=config['kg']['uri'],
    user=config['kg']['user'],
    password=config['kg']['password']
)

print("Checking KG Data Structure")
print("="*60)

# 1. Check what fields Building nodes actually have
print("\n1. BUILDING NODE PROPERTIES:")
query_props = """
MATCH (b:Building) 
RETURN keys(b) as properties
LIMIT 1
"""
result = kg.run(query_props)
if result:
    print("Available properties:", result[0]['properties'])

# 2. Check sample building data
print("\n2. SAMPLE BUILDING DATA:")
query_sample = """
MATCH (b:Building)
RETURN b.ogc_fid as id,
       b.construction_type as construction_type,
       b.type as type,
       b.building_type as building_type,
       b.energy_label as energy_label,
       b.area as area,
       b.roof_area as roof_area,
       b.orientation as orientation,
       b.has_solar as has_solar,
       b.num_occupants as occupants,
       b.district_name as district
LIMIT 5
"""
buildings = kg.run(query_sample)
for i, b in enumerate(buildings):
    print(f"\nBuilding {i+1}:")
    for key, value in b.items():
        if value is not None:
            print(f"  {key}: {value}")

# 3. Check unique values for key fields
print("\n3. UNIQUE VALUES FOR KEY FIELDS:")

# Construction types
query_types = """
MATCH (b:Building)
WHERE b.construction_type IS NOT NULL
RETURN DISTINCT b.construction_type as type, count(*) as count
ORDER BY count DESC
"""
types = kg.run(query_types)
print("\nConstruction Types:")
for t in types:
    print(f"  {t['type']}: {t['count']} buildings")

# Energy labels
query_labels = """
MATCH (b:Building)
WHERE b.energy_label IS NOT NULL
RETURN DISTINCT b.energy_label as label, count(*) as count
ORDER BY label
"""
labels = kg.run(query_labels)
print("\nEnergy Labels:")
for l in labels:
    print(f"  {l['label']}: {l['count']} buildings")

# Check if orientation exists
query_orientation = """
MATCH (b:Building)
RETURN 
    count(b) as total,
    count(b.orientation) as with_orientation,
    collect(DISTINCT b.orientation) as orientations
"""
orient = kg.run(query_orientation)
if orient:
    print(f"\nOrientation Data:")
    print(f"  Total buildings: {orient[0]['total']}")
    print(f"  With orientation: {orient[0]['with_orientation']}")
    print(f"  Unique values: {orient[0]['orientations']}")

# Check roof area
query_roof = """
MATCH (b:Building)
WHERE b.roof_area IS NOT NULL
RETURN 
    count(*) as count,
    min(b.roof_area) as min_roof,
    max(b.roof_area) as max_roof,
    avg(b.roof_area) as avg_roof
"""
roof = kg.run(query_roof)
if roof:
    print(f"\nRoof Area Data:")
    print(f"  Buildings with roof area: {roof[0]['count']}")
    print(f"  Min: {roof[0]['min_roof']:.1f} m²")
    print(f"  Max: {roof[0]['max_roof']:.1f} m²")
    print(f"  Avg: {roof[0]['avg_roof']:.1f} m²")

# Check solar status
query_solar = """
MATCH (b:Building)
RETURN 
    count(b) as total,
    count(CASE WHEN b.has_solar = true THEN 1 END) as with_solar,
    count(CASE WHEN b.has_solar = false THEN 1 END) as without_solar,
    count(CASE WHEN b.has_solar IS NULL THEN 1 END) as null_solar
"""
solar = kg.run(query_solar)
if solar:
    print(f"\nSolar Status:")
    print(f"  Total: {solar[0]['total']}")
    print(f"  With solar: {solar[0]['with_solar']}")
    print(f"  Without solar: {solar[0]['without_solar']}")
    print(f"  Null values: {solar[0]['null_solar']}")

# 4. Check alternative type fields
print("\n4. CHECKING ALTERNATIVE TYPE FIELDS:")
query_alt_types = """
MATCH (b:Building)
RETURN 
    b.type as type_field,
    b.building_type as building_type_field,
    b.construction_type as construction_type_field,
    b.usage_type as usage_type_field,
    b.function as function_field
LIMIT 10
"""
alt_types = kg.run(query_alt_types)
print("\nAlternative type fields found:")
type_fields = set()
for b in alt_types:
    for key, val in b.items():
        if val is not None:
            type_fields.add(key)
print(f"  Fields with data: {type_fields}")

# 5. Check district data
print("\n5. DISTRICT AND LV GROUP DISTRIBUTION:")
query_districts = """
MATCH (b:Building)-[:CONNECTED_TO]->(lv:CableGroup)
RETURN 
    b.district_name as district,
    count(DISTINCT lv.group_id) as lv_groups,
    count(b) as buildings
ORDER BY buildings DESC
LIMIT 5
"""
districts = kg.run(query_districts)
for d in districts:
    print(f"  {d['district']}: {d['buildings']} buildings in {d['lv_groups']} LV groups")