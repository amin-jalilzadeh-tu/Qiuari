"""
Test script to verify energy data integration with Neo4j KG.
"""

import pandas as pd
from neo4j import GraphDatabase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_energy_data_match():
    """Test that energy profile building IDs match KG building IDs."""
    
    # Neo4j configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "aminasad"
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    
    try:
        # Get building IDs from KG
        with driver.session() as session:
            result = session.run("""
                MATCH (b:Building)
                WHERE b.ogc_fid IS NOT NULL
                RETURN collect(b.ogc_fid) as kg_building_ids
            """).single()
            
            kg_building_ids = set(result['kg_building_ids'])
            logger.info(f"Found {len(kg_building_ids)} buildings in KG")
        
        # Load energy profiles
        energy_df = pd.read_parquet('mimic_data/energy_profiles_matched.parquet')
        energy_building_ids = set(energy_df['building_id'].unique())
        logger.info(f"Found {len(energy_building_ids)} buildings in energy profiles")
        
        # Check overlap
        matching_ids = kg_building_ids.intersection(energy_building_ids)
        logger.info(f"Matching building IDs: {len(matching_ids)}")
        
        # Check for mismatches
        in_kg_not_energy = kg_building_ids - energy_building_ids
        in_energy_not_kg = energy_building_ids - kg_building_ids
        
        if in_kg_not_energy:
            logger.warning(f"Buildings in KG but not in energy data: {len(in_kg_not_energy)}")
            logger.warning(f"Sample: {list(in_kg_not_energy)[:5]}")
        
        if in_energy_not_kg:
            logger.warning(f"Buildings in energy data but not in KG: {len(in_energy_not_kg)}")
            logger.warning(f"Sample: {list(in_energy_not_kg)[:5]}")
        
        # Test a sample integration
        if matching_ids:
            sample_id = list(matching_ids)[0]
            logger.info(f"\nTesting integration with building {sample_id}...")
            
            # Get building from KG
            with driver.session() as session:
                building = session.run("""
                    MATCH (b:Building {ogc_fid: $id})
                    RETURN b.area as area, 
                           b.energy_label as label,
                           b.has_solar as has_solar
                """, id=sample_id).single()
                
                logger.info(f"  KG data: Area={building['area']}, Label={building['label']}, Solar={building['has_solar']}")
            
            # Get energy data
            sample_energy = energy_df[energy_df['building_id'] == sample_id].head(5)
            logger.info(f"  Energy data shape: {sample_energy.shape}")
            logger.info(f"  Avg electricity demand: {sample_energy['electricity_demand_kw'].mean():.2f} kW")
            logger.info(f"  Avg heating demand: {sample_energy['heating_demand_kw'].mean():.2f} kW")
            
            logger.info("\n✅ Integration test successful!")
        
        return len(matching_ids) == len(energy_building_ids)
        
    finally:
        driver.close()


def check_data_quality():
    """Check the quality of the generated energy data."""
    
    logger.info("\n" + "="*60)
    logger.info("DATA QUALITY CHECK")
    logger.info("="*60)
    
    # Load energy profiles
    df = pd.read_parquet('mimic_data/energy_profiles_matched.parquet')
    
    # Check for nulls
    null_counts = df.isnull().sum()
    if null_counts.any():
        logger.warning("Null values found:")
        logger.warning(null_counts[null_counts > 0])
    else:
        logger.info("✅ No null values")
    
    # Check for negative values (except net_demand can be negative)
    for col in ['electricity_demand_kw', 'heating_demand_kw', 'cooling_demand_kw', 
                'solar_generation_kw', 'battery_soc_kwh']:
        if col in df.columns:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                logger.warning(f"❌ {col} has {neg_count} negative values")
            else:
                logger.info(f"✅ {col} has no negative values")
    
    # Check reasonable ranges
    logger.info("\nValue ranges:")
    for col in df.columns:
        if col not in ['building_id', 'timestamp']:
            logger.info(f"  {col}: [{df[col].min():.3f}, {df[col].max():.3f}]")
    
    # Check time coverage
    timestamps = pd.to_datetime(df['timestamp'], unit='ms')
    logger.info(f"\nTime coverage: {timestamps.min()} to {timestamps.max()}")
    logger.info(f"Total hours: {(timestamps.max() - timestamps.min()).total_seconds() / 3600:.0f}")
    
    # Check data completeness
    expected_records_per_building = len(df) / df['building_id'].nunique()
    logger.info(f"\nRecords per building: {expected_records_per_building:.0f}")
    
    # Check if all buildings have same number of records
    records_per_building = df.groupby('building_id').size()
    if records_per_building.nunique() == 1:
        logger.info("✅ All buildings have equal number of time steps")
    else:
        logger.warning(f"❌ Unequal time steps: min={records_per_building.min()}, max={records_per_building.max()}")


if __name__ == "__main__":
    # Run tests
    match_success = test_energy_data_match()
    check_data_quality()
    
    if match_success:
        print("\n" + "="*60)
        print("SUCCESS: Energy data is ready for KG integration!")
        print("="*60)
        print("\nNext steps:")
        print("1. Run kg_builder_3.py to add energy data to KG")
        print("2. Use the matched parquet file: mimic_data/energy_profiles_matched.parquet")
    else:
        print("\n⚠️ Warning: Some building IDs don't match between KG and energy data")
        print("This is expected if the energy data was generated from a subset of buildings")