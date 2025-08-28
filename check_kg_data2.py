"""
Check more KG data fields that actually exist
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from data.kg_connector import KGConnector
import yaml

with open('config/unified_config.yaml', 'r') as f:
    config = yaml.safe_load(f)

kg = KGConnector(
    uri=config['kg']['uri'],
    user=config['kg']['user'],
    password=config['kg']['password']
)

print("Checking Actual Available Data")
print("="*60)

# Check building function (found in properties)
print("\n1. BUILDING FUNCTION (for diversity):")
query_func = """
MATCH (b:Building)
WHERE b.building_function IS NOT NULL
RETURN DISTINCT b.building_function as function, count(*) as count
ORDER BY count DESC
"""
funcs = kg.run(query_func)
for f in funcs:
    print(f"  {f['function']}: {f['count']} buildings")

# Check residential vs non-residential types
print("\n2. RESIDENTIAL TYPES:")
query_res = """
MATCH (b:Building)
WHERE b.residential_type IS NOT NULL
RETURN DISTINCT b.residential_type as type, count(*) as count
ORDER BY count DESC
LIMIT 10
"""
res = kg.run(query_res)
for r in res:
    print(f"  {r['type']}: {r['count']}")

print("\n3. NON-RESIDENTIAL TYPES:")
query_nonres = """
MATCH (b:Building)
WHERE b.non_residential_type IS NOT NULL
RETURN DISTINCT b.non_residential_type as type, count(*) as count
ORDER BY count DESC
LIMIT 10
"""
nonres = kg.run(query_nonres)
for n in nonres:
    print(f"  {n['type']}: {n['count']}")

# Check roof areas
print("\n4. ROOF AREA FIELDS:")
query_roofs = """
MATCH (b:Building)
RETURN 
    count(b.flat_roof_area) as with_flat_roof,
    count(b.sloped_roof_area) as with_sloped_roof,
    count(b.suitable_roof_area) as with_suitable_roof,
    avg(b.flat_roof_area) as avg_flat,
    avg(b.sloped_roof_area) as avg_sloped,
    avg(b.suitable_roof_area) as avg_suitable
"""
roofs = kg.run(query_roofs)
if roofs:
    r = roofs[0]
    print(f"  Flat roof: {r['with_flat_roof']} buildings (avg: {r['avg_flat']:.1f if r['avg_flat'] else 0:.1f} m²)")
    print(f"  Sloped roof: {r['with_sloped_roof']} buildings (avg: {r['avg_sloped']:.1f if r['avg_sloped'] else 0:.1f} m²)")
    print(f"  Suitable roof: {r['with_suitable_roof']} buildings (avg: {r['avg_suitable']:.1f if r['avg_suitable'] else 0:.1f} m²)")

# Check orientation field  
print("\n5. BUILDING ORIENTATION:")
query_orient = """
MATCH (b:Building)
WHERE b.building_orientation_cardinal IS NOT NULL
RETURN DISTINCT b.building_orientation_cardinal as orientation, count(*) as count
ORDER BY count DESC
"""
orient = kg.run(query_orient)
for o in orient:
    print(f"  {o['orientation']}: {o['count']} buildings")

# Check solar potential
print("\n6. SOLAR POTENTIAL:")
query_solar_pot = """
MATCH (b:Building)
WHERE b.solar_potential IS NOT NULL
RETURN 
    count(*) as count,
    min(b.solar_potential) as min_pot,
    max(b.solar_potential) as max_pot,
    avg(b.solar_potential) as avg_pot
"""
solar_pot = kg.run(query_solar_pot)
if solar_pot and solar_pot[0]['count'] > 0:
    s = solar_pot[0]
    print(f"  Buildings with data: {s['count']}")
    print(f"  Range: {s['min_pot']:.1f} - {s['max_pot']:.1f}")
    print(f"  Average: {s['avg_pot']:.1f}")

# Check peak hours
print("\n7. PEAK HOURS/DEMANDS:")
query_peaks = """
MATCH (b:Building)
RETURN 
    count(b.peak_hours) as with_peak_hours,
    count(b.peak_demands) as with_peak_demands,
    count(b.peak_electricity_demand_kw) as with_peak_elec,
    avg(b.peak_electricity_demand_kw) as avg_peak_elec
"""
peaks = kg.run(query_peaks)
if peaks:
    p = peaks[0]
    print(f"  Peak hours field: {p['with_peak_hours']} buildings")
    print(f"  Peak demands field: {p['with_peak_demands']} buildings")
    print(f"  Peak electricity: {p['with_peak_elec']} buildings")
    if p['avg_peak_elec']:
        print(f"  Avg peak elec: {p['avg_peak_elec']:.1f} kW")

# Sample complete building data
print("\n8. SAMPLE COMPLETE BUILDING:")
query_complete = """
MATCH (b:Building)
WHERE b.building_function IS NOT NULL
RETURN 
    b.ogc_fid as id,
    b.building_function as function,
    b.residential_type as res_type,
    b.non_residential_type as nonres_type,
    b.energy_label as label,
    b.area as area,
    b.flat_roof_area as flat_roof,
    b.sloped_roof_area as sloped_roof,
    b.suitable_roof_area as suitable_roof,
    b.building_orientation_cardinal as orientation,
    b.solar_potential as solar_pot,
    b.has_solar as has_solar,
    b.peak_electricity_demand_kw as peak_elec
LIMIT 3
"""
complete = kg.run(query_complete)
for i, b in enumerate(complete):
    print(f"\nBuilding {i+1} (ID: {b['id']}):")
    for key, val in b.items():
        if val is not None and key != 'id':
            print(f"  {key}: {val}")