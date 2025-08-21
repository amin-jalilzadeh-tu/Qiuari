"""
Create energy profiles parquet file with building IDs matching Neo4j KG.
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from datetime import datetime, timedelta
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnergyProfileGenerator:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize with Neo4j connection."""
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        logger.info(f"Connected to Neo4j at {neo4j_uri}")
        
    def get_buildings_from_kg(self, limit: int = None) -> pd.DataFrame:
        """Get building data from Neo4j KG."""
        with self.driver.session() as session:
            query = """
            MATCH (b:Building)
            WHERE b.ogc_fid IS NOT NULL
            RETURN 
                b.ogc_fid as building_id,
                b.area as area,
                b.energy_label as energy_label,
                b.building_function as function,
                b.has_solar as has_solar,
                b.has_battery as has_battery,
                b.has_heat_pump as has_heat_pump,
                b.solar_capacity_kwp as solar_capacity,
                b.expected_cop as heat_pump_cop,
                b.height as height,
                b.district_name as district
            ORDER BY b.ogc_fid
            """
            
            if limit:
                query += f" LIMIT {limit}"
            
            result = session.run(query).data()
            df = pd.DataFrame(result)
            logger.info(f"Retrieved {len(df)} buildings from KG")
            return df
    
    def generate_energy_profiles(self, buildings_df: pd.DataFrame, 
                                days: int = 7,
                                hourly_resolution: bool = True) -> pd.DataFrame:
        """
        Generate realistic energy profiles for buildings.
        
        Args:
            buildings_df: DataFrame with building characteristics
            days: Number of days to simulate
            hourly_resolution: If True, hourly data; if False, 15-minute intervals
        
        Returns:
            DataFrame with energy time series
        """
        
        # Time parameters
        intervals_per_hour = 1 if hourly_resolution else 4
        total_intervals = days * 24 * intervals_per_hour
        
        # Create timestamp range
        start_time = datetime(2024, 1, 1, 0, 0, 0)  # Start from Jan 1, 2024
        if hourly_resolution:
            timestamps = pd.date_range(start_time, periods=total_intervals, freq='H')
        else:
            timestamps = pd.date_range(start_time, periods=total_intervals, freq='15min')
        
        profiles = []
        
        logger.info(f"Generating {total_intervals} time intervals for {len(buildings_df)} buildings")
        
        for _, building in tqdm(buildings_df.iterrows(), total=len(buildings_df), desc="Generating profiles"):
            building_id = building['building_id']
            area = building.get('area', 150) or 150
            energy_label = building.get('energy_label', 'D')
            function = building.get('function', 'residential')
            has_solar = building.get('has_solar', False)
            has_battery = building.get('has_battery', False)
            has_heat_pump = building.get('has_heat_pump', False)
            solar_capacity = building.get('solar_capacity', 0) or 0
            
            # Energy efficiency factor based on label
            efficiency_map = {
                'A': 0.5, 'B': 0.6, 'C': 0.7, 'D': 0.85,
                'E': 1.0, 'F': 1.2, 'G': 1.4
            }
            efficiency_factor = efficiency_map.get(energy_label, 1.0)
            
            # Base consumption (kWh/m²/year)
            if function == 'residential':
                base_elec_annual = 30  # kWh/m²/year
                base_heat_annual = 80  # kWh/m²/year
            elif function == 'office':
                base_elec_annual = 50
                base_heat_annual = 60
            else:
                base_elec_annual = 40
                base_heat_annual = 70
            
            # Convert to hourly demand
            base_elec_hourly = (base_elec_annual * area * efficiency_factor) / 8760
            base_heat_hourly = (base_heat_annual * area * efficiency_factor) / 8760
            
            for ts in timestamps:
                hour = ts.hour
                day_of_week = ts.dayofweek
                month = ts.month
                
                # Time-of-day factors
                if function == 'residential':
                    # Residential pattern: peaks morning and evening
                    if 6 <= hour < 9:
                        elec_factor = 1.5
                    elif 17 <= hour < 22:
                        elec_factor = 1.8
                    elif 22 <= hour or hour < 6:
                        elec_factor = 0.4
                    else:
                        elec_factor = 0.8
                else:
                    # Commercial pattern: peaks during work hours
                    if 8 <= hour < 18 and day_of_week < 5:
                        elec_factor = 1.5
                    else:
                        elec_factor = 0.3
                
                # Seasonal heating factor
                seasonal_heat = {
                    1: 1.5, 2: 1.4, 3: 1.2, 4: 0.8,
                    5: 0.5, 6: 0.2, 7: 0.1, 8: 0.1,
                    9: 0.3, 10: 0.7, 11: 1.1, 12: 1.4
                }
                heat_factor = seasonal_heat[month]
                
                # Add random variation (±20%)
                random_factor = np.random.uniform(0.8, 1.2)
                
                # Calculate demands
                electricity_demand = base_elec_hourly * elec_factor * random_factor
                heating_demand = base_heat_hourly * heat_factor * random_factor
                
                # Cooling (inverse of heating, simplified)
                cooling_demand = max(0, (0.3 - heat_factor) * base_elec_hourly * random_factor)
                
                # Solar generation (if installed)
                solar_generation = 0
                if has_solar and solar_capacity > 0:
                    # Solar production curve
                    if 6 <= hour <= 18:
                        solar_factor = np.sin((hour - 6) * np.pi / 12)
                        # Season adjustment
                        season_solar = {
                            1: 0.3, 2: 0.4, 3: 0.6, 4: 0.8,
                            5: 0.9, 6: 1.0, 7: 1.0, 8: 0.9,
                            9: 0.7, 10: 0.5, 11: 0.3, 12: 0.2
                        }
                        solar_generation = (solar_capacity * solar_factor * 
                                          season_solar[month] * 
                                          np.random.uniform(0.7, 1.0))
                
                # Battery operation (simplified)
                battery_soc = 0
                battery_charge = 0
                battery_discharge = 0
                
                if has_battery:
                    battery_capacity = min(13.5, area * 0.05)  # kWh
                    
                    # Simple battery logic
                    net_demand = electricity_demand - solar_generation
                    
                    if net_demand < 0:  # Excess solar
                        battery_charge = min(-net_demand, battery_capacity * 0.2)
                        battery_soc = min(battery_capacity, battery_charge * 0.9)
                    elif net_demand > 0 and hour >= 17:  # Evening discharge
                        battery_discharge = min(net_demand * 0.5, battery_capacity * 0.2)
                        battery_soc = max(0, battery_capacity * 0.5)
                
                profiles.append({
                    'building_id': int(building_id),
                    'timestamp': ts,
                    'electricity_demand_kw': round(electricity_demand, 3),
                    'heating_demand_kw': round(heating_demand, 3),
                    'cooling_demand_kw': round(cooling_demand, 3),
                    'solar_generation_kw': round(solar_generation, 3),
                    'battery_soc_kwh': round(battery_soc, 3),
                    'battery_charge_kw': round(battery_charge, 3),
                    'battery_discharge_kw': round(battery_discharge, 3)
                })
        
        df = pd.DataFrame(profiles)
        logger.info(f"Generated {len(df)} energy records for {df['building_id'].nunique()} buildings")
        
        return df
    
    def save_to_parquet(self, df: pd.DataFrame, output_path: str):
        """Save DataFrame to parquet file."""
        # Convert timestamp to milliseconds for compatibility
        df['timestamp'] = df['timestamp'].astype('int64') // 10**6  # Convert to milliseconds
        
        df.to_parquet(output_path, index=False, compression='snappy')
        logger.info(f"Saved {len(df)} records to {output_path}")
        
        # Verify
        test_df = pd.read_parquet(output_path)
        logger.info(f"Verification - Shape: {test_df.shape}, Buildings: {test_df['building_id'].nunique()}")
        logger.info(f"Sample building IDs: {sorted(test_df['building_id'].unique())[:10]}")
    
    def close(self):
        """Close Neo4j connection."""
        self.driver.close()


def main():
    # Neo4j configuration
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "aminasad"
    
    # Output configuration
    OUTPUT_PATH = "mimic_data/energy_profiles_matched.parquet"
    DAYS_TO_SIMULATE = 28  # 4 weeks of data
    BUILDING_LIMIT = None  # Set to None to get all buildings, or a number to limit
    
    logger.info("="*60)
    logger.info("Starting Energy Profile Generation with KG Building IDs")
    logger.info("="*60)
    
    # Create generator
    generator = EnergyProfileGenerator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    try:
        # Step 1: Get buildings from KG
        logger.info("Step 1: Fetching buildings from Knowledge Graph...")
        buildings_df = generator.get_buildings_from_kg(limit=BUILDING_LIMIT)
        
        if buildings_df.empty:
            logger.error("No buildings found in KG!")
            return
        
        logger.info(f"Found {len(buildings_df)} buildings in KG")
        logger.info(f"Building ID range: {buildings_df['building_id'].min()} - {buildings_df['building_id'].max()}")
        logger.info(f"Districts: {buildings_df['district'].value_counts().head()}")
        
        # Step 2: Generate energy profiles
        logger.info(f"\nStep 2: Generating {DAYS_TO_SIMULATE} days of energy profiles...")
        energy_df = generator.generate_energy_profiles(
            buildings_df, 
            days=DAYS_TO_SIMULATE,
            hourly_resolution=True
        )
        
        # Step 3: Save to parquet
        logger.info("\nStep 3: Saving to parquet file...")
        generator.save_to_parquet(energy_df, OUTPUT_PATH)
        
        # Print summary statistics
        logger.info("\n" + "="*60)
        logger.info("SUMMARY STATISTICS")
        logger.info("="*60)
        logger.info(f"Total records: {len(energy_df):,}")
        logger.info(f"Unique buildings: {energy_df['building_id'].nunique()}")
        logger.info(f"Time range: {DAYS_TO_SIMULATE} days")
        logger.info(f"Avg electricity demand: {energy_df['electricity_demand_kw'].mean():.2f} kW")
        logger.info(f"Avg heating demand: {energy_df['heating_demand_kw'].mean():.2f} kW")
        logger.info(f"Total solar generation: {energy_df['solar_generation_kw'].sum():.0f} kWh")
        logger.info(f"Output file: {OUTPUT_PATH}")
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        generator.close()
        logger.info("\nGeneration complete!")


if __name__ == "__main__":
    main()