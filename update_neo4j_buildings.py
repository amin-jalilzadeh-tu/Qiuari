"""
Update Neo4j buildings with additional attributes from CSV
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from neo4j import GraphDatabase
import pandas as pd
import yaml
import logging
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_buildings():
    """Update building nodes with additional attributes"""
    
    # Load config
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Connect to Neo4j
    driver = GraphDatabase.driver(
        config['neo4j']['uri'],
        auth=(config['neo4j']['user'], config['neo4j']['password'])
    )
    
    # Load buildings data
    logger.info("Loading buildings CSV...")
    buildings_df = pd.read_csv('mimic_data/buildings.csv')
    
    # Load energy profiles for aggregated stats
    logger.info("Loading energy profiles...")
    energy_df = pd.read_parquet('mimic_data/energy_profiles.parquet')
    
    # Calculate peak and average demand per building
    demand_stats = energy_df.groupby('building_id').agg({
        'electricity_demand_kw': ['max', 'mean'],
        'solar_generation_kw': 'max'
    }).reset_index()
    demand_stats.columns = ['building_id', 'peak_demand', 'avg_demand', 'max_solar']
    
    with driver.session() as session:
        for idx, row in buildings_df.iterrows():
            building_id = f"B_{row['ogc_fid']}"
            
            # Get demand stats for this building
            building_stats = demand_stats[demand_stats['building_id'] == row['ogc_fid']]
            if not building_stats.empty:
                peak_demand = float(building_stats.iloc[0]['peak_demand'])
                avg_demand = float(building_stats.iloc[0]['avg_demand'])
            else:
                # Estimate based on building area
                peak_demand = row['area'] * 0.05  # 50W/m2 peak
                avg_demand = row['area'] * 0.02   # 20W/m2 average
            
            # Update building node with all attributes
            session.run("""
                MATCH (b:Building {id: $building_id})
                SET b.x = $x,
                    b.y = $y,
                    b.height = $height,
                    b.age_range = $age_range,
                    b.solar_capacity_kwp = $solar_capacity,
                    b.battery_capacity_kwh = $battery_capacity,
                    b.has_solar = $has_solar,
                    b.has_battery = $has_battery,
                    b.peak_demand = $peak_demand,
                    b.avg_demand = $avg_demand,
                    b.suitable_roof_area = $suitable_roof_area,
                    b.lv_network = $lv_network,
                    b.adjacency_type = $adjacency_type,
                    b.num_shared_walls = $num_shared_walls
            """, {
                'building_id': building_id,
                'x': float(row['x']),
                'y': float(row['y']),
                'height': float(row['height']) if pd.notna(row['height']) else 10.0,
                'age_range': str(row['age_range']) if pd.notna(row['age_range']) else 'unknown',
                'solar_capacity': float(row['solar_capacity_kwp']) if pd.notna(row['solar_capacity_kwp']) else 0.0,
                'battery_capacity': float(row['battery_capacity_kwh']) if pd.notna(row['battery_capacity_kwh']) else 0.0,
                'has_solar': bool(row['has_solar']),
                'has_battery': bool(row['has_battery']),
                'peak_demand': peak_demand,
                'avg_demand': avg_demand,
                'suitable_roof_area': float(row['sloped_roof_area']) if pd.notna(row['sloped_roof_area']) else float(row['roof_area']) * 0.7,
                'lv_network': str(row['lv_network_id']),
                'adjacency_type': str(row['adjacency_type']) if pd.notna(row['adjacency_type']) else 'UNKNOWN',
                'num_shared_walls': int(row['num_shared_walls']) if pd.notna(row['num_shared_walls']) else 0
            })
            
            if (idx + 1) % 20 == 0:
                logger.info(f"Updated {idx + 1}/{len(buildings_df)} buildings...")
    
    # Verify updates
    with driver.session() as session:
        result = session.run("""
            MATCH (b:Building)
            WHERE b.peak_demand IS NOT NULL
            RETURN COUNT(b) as count
        """)
        count = result.single()['count']
        logger.info(f"Buildings with peak_demand: {count}")
        
        result = session.run("""
            MATCH (b:Building)
            WHERE b.has_solar = true
            RETURN COUNT(b) as count
        """)
        solar_count = result.single()['count']
        logger.info(f"Buildings with solar: {solar_count}")
    
    driver.close()
    logger.info("Building updates complete!")

if __name__ == "__main__":
    update_buildings()