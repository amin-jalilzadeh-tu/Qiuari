# mimic_data_generator.py
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import networkx as nx
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================
# 1. CREATE MIMIC GRID TOPOLOGY
# ============================================

def create_grid_topology():
    """Create a realistic grid with 3 LV networks under 1 MV transformer"""
    
    grid_data = {
        'substations': [],
        'mv_transformers': [],
        'lv_networks': [],
        'lv_cables': [],
        'mv_cables': []
    }
    
    # Create 1 substation
    substation = {
        'id': 'SUB_001',
        'x': 100000,
        'y': 450000,
        'capacity_mva': 50,
        'geometry': Point(100000, 450000)
    }
    grid_data['substations'].append(substation)
    
    # Create 2 MV transformers
    mv_positions = [(100200, 450100), (100400, 449900)]
    for i, pos in enumerate(mv_positions):
        mv_transformer = {
            'id': f'MV_TRANS_{i+1:03d}',
            'x': pos[0],
            'y': pos[1],
            'capacity_kva': 1000,
            'substation_id': 'SUB_001',
            'geometry': Point(pos[0], pos[1])
        }
        grid_data['mv_transformers'].append(mv_transformer)
        
        # MV cable from substation to transformer
        mv_cable = {
            'id': f'MV_CABLE_{i+1:03d}',
            'from_node': 'SUB_001',
            'to_node': f'MV_TRANS_{i+1:03d}',
            'length_m': np.sqrt((pos[0]-100000)**2 + (pos[1]-450000)**2),
            'geometry': LineString([Point(100000, 450000), Point(pos[0], pos[1])])
        }
        grid_data['mv_cables'].append(mv_cable)
    
    # Create 3 LV networks per MV transformer
    lv_id = 1
    for mv_idx, mv_trans in enumerate(grid_data['mv_transformers']):
        for lv_net in range(3):
            # Position LV networks around MV transformer
            angle = lv_net * 120 * np.pi / 180
            lv_x = mv_trans['x'] + 150 * np.cos(angle)
            lv_y = mv_trans['y'] + 150 * np.sin(angle)
            
            lv_network = {
                'id': f'LV_NET_{lv_id:03d}',
                'component_id': lv_id,
                'x': lv_x,
                'y': lv_y,
                'mv_transformer_id': mv_trans['id'],
                'capacity_kva': 250,
                'geometry': Point(lv_x, lv_y)
            }
            grid_data['lv_networks'].append(lv_network)
            
            # Create LV cables (simplified as single line)
            lv_cable = {
                'id': f'LV_CABLE_{lv_id:03d}',
                'component_id': lv_id,
                'from_node': mv_trans['id'],
                'to_node': f'LV_NET_{lv_id:03d}',
                'length_m': 150,
                'geometry': LineString([Point(mv_trans['x'], mv_trans['y']), 
                                       Point(lv_x, lv_y)])
            }
            grid_data['lv_cables'].append(lv_cable)
            lv_id += 1
    
    return grid_data

# ============================================
# 2. CREATE MIMIC BUILDINGS
# ============================================

def create_buildings(grid_data):
    """Create diverse buildings assigned to LV networks"""
    
    buildings = []
    building_id = 1
    
    # Building type distributions
    building_configs = [
        # LV_NET_001: Residential heavy (morning/evening peaks)
        {'lv_id': 'LV_NET_001', 'residential': 20, 'office': 2, 'retail': 3, 'industrial': 1},
        # LV_NET_002: Mixed use (good complementarity)
        {'lv_id': 'LV_NET_002', 'residential': 10, 'office': 8, 'retail': 4, 'industrial': 2},
        # LV_NET_003: Commercial heavy (daytime peaks)
        {'lv_id': 'LV_NET_003', 'residential': 5, 'office': 12, 'retail': 6, 'industrial': 2},
        # LV_NET_004: Residential cluster
        {'lv_id': 'LV_NET_004', 'residential': 25, 'office': 1, 'retail': 2, 'industrial': 0},
        # LV_NET_005: Industrial mixed
        {'lv_id': 'LV_NET_005', 'residential': 8, 'office': 4, 'retail': 2, 'industrial': 6},
        # LV_NET_006: Small mixed
        {'lv_id': 'LV_NET_006', 'residential': 12, 'office': 3, 'retail': 3, 'industrial': 1}
    ]
    
    for config in building_configs:
        lv_network = next(lv for lv in grid_data['lv_networks'] if lv['id'] == config['lv_id'])
        
        # Generate buildings around LV network center
        for building_type, count in config.items():
            if building_type == 'lv_id':
                continue
                
            for i in range(count):
                # Random position around LV network (within 100m)
                angle = random.uniform(0, 2*np.pi)
                distance = random.uniform(10, 100)
                x = lv_network['x'] + distance * np.cos(angle)
                y = lv_network['y'] + distance * np.sin(angle)
                
                # Building attributes based on type
                if building_type == 'residential':
                    area = random.choice([80, 100, 120, 150, 200])
                    height = random.choice([3, 6, 9, 12])
                    residential_subtype = random.choice(['Detached', 'Semi-detached', 
                                                        'Terrace', 'Apartment'])
                    orientation = random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
                    roof_area = area * 0.6 if residential_subtype != 'Apartment' else area * 0.3
                    
                elif building_type == 'office':
                    area = random.choice([200, 400, 600, 800, 1000])
                    height = random.choice([12, 15, 18, 21])
                    residential_subtype = None
                    orientation = random.choice(['N', 'E', 'S', 'W'])
                    roof_area = area * 0.4
                    
                elif building_type == 'retail':
                    area = random.choice([150, 300, 500, 750])
                    height = random.choice([6, 9, 12])
                    residential_subtype = None
                    orientation = random.choice(['N', 'E', 'S', 'W'])
                    roof_area = area * 0.5
                    
                else:  # industrial
                    area = random.choice([500, 1000, 1500, 2000])
                    height = random.choice([9, 12, 15])
                    residential_subtype = None
                    orientation = 'Flat'
                    roof_area = area * 0.7
                
                building = {
                    'ogc_fid': building_id,
                    'x': x,
                    'y': y,
                    'building_function': 'residential' if building_type == 'residential' else 'non_residential',
                    'residential_type': residential_subtype,
                    'non_residential_type': building_type.capitalize() if building_type != 'residential' else None,
                    'area': area,
                    'height': height,
                    'age_range': random.choice(['<1945', '1945-1975', '1975-1995', '1995-2015', '>2015']),
                    'building_orientation_cardinal': orientation,
                    'roof_area': roof_area,
                    'flat_roof_area': roof_area if orientation == 'Flat' else roof_area * 0.3,
                    'sloped_roof_area': 0 if orientation == 'Flat' else roof_area * 0.7,
                    'lv_component_id': lv_network['component_id'],
                    'lv_network_id': lv_network['id'],
                    'geometry': Point(x, y)
                }
                buildings.append(building)
                building_id += 1
    
    return pd.DataFrame(buildings)
















def add_shared_wall_data(buildings_df):
    """
    Add realistic shared wall data based on building positions
    Creates row houses, corner configurations, and isolated buildings
    """
    
    # Initialize all wall columns
    for side in ['north', 'south', 'east', 'west']:
        buildings_df[f'{side}_side'] = True  # All buildings have 4 walls
        buildings_df[f'{side}_shared_length'] = 0.0
        buildings_df[f'{side}_facade_length'] = 10.0  # Default 10m per wall
    
    # Group buildings by LV network for local processing
    for lv_network in buildings_df['lv_network_id'].unique():
        lv_buildings = buildings_df[buildings_df['lv_network_id'] == lv_network].copy()
        
        # Create different adjacency patterns based on building types
        residential_buildings = lv_buildings[lv_buildings['building_function'] == 'residential']
        
        # Pattern 1: Row Houses (Terraces)
        # Find clusters of residential buildings that could form rows
        if len(residential_buildings) >= 3:
            # Sort by x coordinate to create east-west rows
            sorted_res = residential_buildings.sort_values('x').head(6)  # Take up to 6 for a row
            
            for i in range(len(sorted_res) - 1):
                idx_current = sorted_res.iloc[i].name
                idx_next = sorted_res.iloc[i + 1].name
                
                # Check if buildings are close enough to be row houses
                if abs(sorted_res.iloc[i]['x'] - sorted_res.iloc[i + 1]['x']) < 15:
                    # Current building shares east wall
                    buildings_df.at[idx_current, 'east_shared_length'] = 8.0
                    buildings_df.at[idx_current, 'east_facade_length'] = 2.0
                    
                    # Next building shares west wall
                    buildings_df.at[idx_next, 'west_shared_length'] = 8.0
                    buildings_df.at[idx_next, 'west_facade_length'] = 2.0
        
        # Pattern 2: Corner Configuration (L-shapes)
        # Find buildings that could form corner clusters
        for idx, building in lv_buildings.iterrows():
            # Find potential neighbors
            nearby = lv_buildings[
                (lv_buildings.index != idx) &
                (((lv_buildings['x'] - building['x']).abs() < 15) |
                 ((lv_buildings['y'] - building['y']).abs() < 15))
            ]
            
            for n_idx, neighbor in nearby.iterrows():
                dx = neighbor['x'] - building['x']
                dy = neighbor['y'] - building['y']
                
                # North-South adjacency
                if abs(dx) < 5 and dy > 0 and dy < 15:  # Neighbor is north
                    if buildings_df.at[idx, 'north_shared_length'] == 0:  # Not already shared
                        shared_length = 6.0 if building['residential_type'] == 'Apartment' else 8.0
                        buildings_df.at[idx, 'north_shared_length'] = shared_length
                        buildings_df.at[idx, 'north_facade_length'] = 10.0 - shared_length
                        buildings_df.at[n_idx, 'south_shared_length'] = shared_length
                        buildings_df.at[n_idx, 'south_facade_length'] = 10.0 - shared_length
                
                elif abs(dx) < 5 and dy < 0 and dy > -15:  # Neighbor is south
                    if buildings_df.at[idx, 'south_shared_length'] == 0:
                        shared_length = 6.0 if building['residential_type'] == 'Apartment' else 8.0
                        buildings_df.at[idx, 'south_shared_length'] = shared_length
                        buildings_df.at[idx, 'south_facade_length'] = 10.0 - shared_length
                        buildings_df.at[n_idx, 'north_shared_length'] = shared_length
                        buildings_df.at[n_idx, 'north_facade_length'] = 10.0 - shared_length
                
                # East-West adjacency (if not already in row house)
                elif abs(dy) < 5 and dx > 0 and dx < 15:  # Neighbor is east
                    if buildings_df.at[idx, 'east_shared_length'] == 0:
                        shared_length = 7.0
                        buildings_df.at[idx, 'east_shared_length'] = shared_length
                        buildings_df.at[idx, 'east_facade_length'] = 10.0 - shared_length
                        buildings_df.at[n_idx, 'west_shared_length'] = shared_length
                        buildings_df.at[n_idx, 'west_facade_length'] = 10.0 - shared_length
                
                elif abs(dy) < 5 and dx < 0 and dx > -15:  # Neighbor is west
                    if buildings_df.at[idx, 'west_shared_length'] == 0:
                        shared_length = 7.0
                        buildings_df.at[idx, 'west_shared_length'] = shared_length
                        buildings_df.at[idx, 'west_facade_length'] = 10.0 - shared_length
                        buildings_df.at[n_idx, 'east_shared_length'] = shared_length
                        buildings_df.at[n_idx, 'east_facade_length'] = 10.0 - shared_length
    
    # Pattern 3: Apartment blocks (multiple shared walls)
    apartments = buildings_df[buildings_df['residential_type'] == 'Apartment']
    for idx, apt in apartments.iterrows():
        # Apartments often share 2-3 walls
        nearby_apts = apartments[
            (apartments.index != idx) &
            (((apartments['x'] - apt['x']).abs() < 10) &
             ((apartments['y'] - apt['y']).abs() < 10))
        ]
        
        if len(nearby_apts) >= 2:
            # Create a more complex sharing pattern
            buildings_df.at[idx, 'north_shared_length'] = 9.0
            buildings_df.at[idx, 'north_facade_length'] = 1.0
            buildings_df.at[idx, 'east_shared_length'] = 9.0
            buildings_df.at[idx, 'east_facade_length'] = 1.0
    
    # Add summary statistics
    buildings_df['num_shared_walls'] = (
        (buildings_df['north_shared_length'] > 0).astype(int) +
        (buildings_df['south_shared_length'] > 0).astype(int) +
        (buildings_df['east_shared_length'] > 0).astype(int) +
        (buildings_df['west_shared_length'] > 0).astype(int)
    )
    
    buildings_df['total_shared_length'] = (
        buildings_df['north_shared_length'] +
        buildings_df['south_shared_length'] +
        buildings_df['east_shared_length'] +
        buildings_df['west_shared_length']
    )
    
    buildings_df['adjacency_type'] = buildings_df.apply(classify_adjacency_type, axis=1)
    
    return buildings_df

def classify_adjacency_type(row):
    """Classify building adjacency configuration"""
    num_shared = row['num_shared_walls']
    
    if num_shared == 0:
        return 'ISOLATED'
    elif num_shared == 1:
        return 'END_UNIT'
    elif num_shared == 2:
        # Check if opposite or corner
        if (row['north_shared_length'] > 0 and row['south_shared_length'] > 0) or \
           (row['east_shared_length'] > 0 and row['west_shared_length'] > 0):
            return 'MIDDLE_ROW'
        else:
            return 'CORNER'
    elif num_shared == 3:
        return 'THREE_SIDED'
    else:
        return 'ENCLOSED'











# ============================================
# 3. CREATE ENERGY PROFILES
# ============================================

# Fixed version of create_energy_profiles function
def create_energy_profiles(buildings_df, days=7):
    """Create realistic 15-minute energy profiles for each building"""
    
    # Time series index (15-minute intervals for 7 days)
    time_index = pd.date_range(start='2024-01-15', periods=days*24*4, freq='15min')
    
    profiles = {}
    
    for _, building in buildings_df.iterrows():
        building_id = building['ogc_fid']
        
        # Base load patterns by building type
        if building['building_function'] == 'residential':
            # Residential: morning and evening peaks
            base_pattern = np.array([
                0.3, 0.3, 0.25, 0.25,  # 00:00-01:00 (night)
                0.25, 0.25, 0.25, 0.25,  # 01:00-02:00
                0.25, 0.25, 0.25, 0.25,  # 02:00-03:00
                0.25, 0.25, 0.25, 0.25,  # 03:00-04:00
                0.3, 0.35, 0.4, 0.45,   # 04:00-05:00 (early morning)
                0.5, 0.6, 0.7, 0.8,     # 05:00-06:00
                0.9, 1.0, 0.95, 0.9,    # 06:00-07:00 (morning peak)
                0.85, 0.8, 0.75, 0.7,   # 07:00-08:00
                0.6, 0.5, 0.45, 0.4,    # 08:00-09:00
                0.35, 0.35, 0.35, 0.35, # 09:00-10:00 (daytime low)
                0.35, 0.35, 0.35, 0.35, # 10:00-11:00
                0.35, 0.4, 0.45, 0.5,   # 11:00-12:00
                0.55, 0.6, 0.6, 0.55,   # 12:00-13:00 (lunch)
                0.5, 0.45, 0.4, 0.4,    # 13:00-14:00
                0.4, 0.4, 0.4, 0.4,     # 14:00-15:00
                0.4, 0.45, 0.5, 0.55,   # 15:00-16:00
                0.6, 0.65, 0.7, 0.75,   # 16:00-17:00
                0.8, 0.85, 0.9, 0.95,   # 17:00-18:00 (evening rise)
                1.0, 1.0, 0.95, 0.9,    # 18:00-19:00 (evening peak)
                0.85, 0.8, 0.75, 0.7,   # 19:00-20:00
                0.65, 0.6, 0.55, 0.5,   # 20:00-21:00
                0.45, 0.4, 0.35, 0.35,  # 21:00-22:00
                0.35, 0.35, 0.3, 0.3,   # 22:00-23:00
                0.3, 0.3, 0.3, 0.3      # 23:00-00:00
            ])
            peak_demand = building['area'] * 0.03  # 30W/m2 peak
            
        elif building['non_residential_type'] == 'Office':
            # Office: strong daytime peak, minimal night/weekend
            base_pattern = np.array([
                0.15, 0.15, 0.15, 0.15,  # 00:00-01:00 (night minimum)
                0.15, 0.15, 0.15, 0.15,  # 01:00-02:00
                0.15, 0.15, 0.15, 0.15,  # 02:00-03:00
                0.15, 0.15, 0.15, 0.15,  # 03:00-04:00
                0.15, 0.15, 0.15, 0.15,  # 04:00-05:00
                0.15, 0.15, 0.2, 0.25,   # 05:00-06:00
                0.3, 0.4, 0.5, 0.6,      # 06:00-07:00 (startup)
                0.7, 0.8, 0.85, 0.9,     # 07:00-08:00
                0.95, 1.0, 1.0, 1.0,     # 08:00-09:00 (work hours)
                1.0, 1.0, 1.0, 1.0,      # 09:00-10:00
                1.0, 1.0, 1.0, 1.0,      # 10:00-11:00
                1.0, 1.0, 1.0, 1.0,      # 11:00-12:00
                0.9, 0.85, 0.85, 0.9,    # 12:00-13:00 (lunch dip)
                0.95, 1.0, 1.0, 1.0,     # 13:00-14:00
                1.0, 1.0, 1.0, 1.0,      # 14:00-15:00
                1.0, 1.0, 0.95, 0.9,     # 15:00-16:00
                0.85, 0.8, 0.7, 0.6,     # 16:00-17:00 (end of day)
                0.5, 0.4, 0.3, 0.25,     # 17:00-18:00
                0.2, 0.2, 0.18, 0.18,    # 18:00-19:00
                0.17, 0.17, 0.16, 0.16,  # 19:00-20:00
                0.15, 0.15, 0.15, 0.15,  # 20:00-21:00
                0.15, 0.15, 0.15, 0.15,  # 21:00-22:00
                0.15, 0.15, 0.15, 0.15,  # 22:00-23:00
                0.15, 0.15, 0.15, 0.15   # 23:00-00:00
            ])
            peak_demand = building['area'] * 0.05  # 50W/m2 peak
            
        elif building['non_residential_type'] == 'Retail':
            # Retail: late morning to evening
            base_pattern = np.array([
                0.1, 0.1, 0.1, 0.1,      # 00:00-01:00
                0.1, 0.1, 0.1, 0.1,      # 01:00-02:00
                0.1, 0.1, 0.1, 0.1,      # 02:00-03:00
                0.1, 0.1, 0.1, 0.1,      # 03:00-04:00
                0.1, 0.1, 0.1, 0.1,      # 04:00-05:00
                0.1, 0.15, 0.2, 0.25,    # 05:00-06:00
                0.3, 0.35, 0.4, 0.45,    # 06:00-07:00
                0.5, 0.55, 0.6, 0.65,    # 07:00-08:00
                0.7, 0.75, 0.8, 0.85,    # 08:00-09:00
                0.9, 0.95, 1.0, 1.0,     # 09:00-10:00 (opening)
                1.0, 1.0, 1.0, 1.0,      # 10:00-11:00
                1.0, 1.0, 1.0, 1.0,      # 11:00-12:00
                1.0, 1.0, 1.0, 1.0,      # 12:00-13:00
                1.0, 1.0, 1.0, 1.0,      # 13:00-14:00
                1.0, 1.0, 1.0, 1.0,      # 14:00-15:00
                1.0, 1.0, 1.0, 1.0,      # 15:00-16:00
                1.0, 1.0, 1.0, 1.0,      # 16:00-17:00
                1.0, 1.0, 0.95, 0.9,     # 17:00-18:00
                0.85, 0.8, 0.75, 0.7,    # 18:00-19:00
                0.65, 0.6, 0.5, 0.4,     # 19:00-20:00 (closing)
                0.3, 0.25, 0.2, 0.15,    # 20:00-21:00
                0.12, 0.12, 0.11, 0.11,  # 21:00-22:00
                0.1, 0.1, 0.1, 0.1,      # 22:00-23:00
                0.1, 0.1, 0.1, 0.1       # 23:00-00:00
            ])
            peak_demand = building['area'] * 0.06  # 60W/m2 peak
            
        else:  # Industrial
            # Industrial: relatively constant with slight day variation
            base_pattern = np.array([
                0.7, 0.7, 0.7, 0.7,      # 00:00-01:00 (24/7 operation)
                0.7, 0.7, 0.7, 0.7,      # 01:00-02:00
                0.7, 0.7, 0.7, 0.7,      # 02:00-03:00
                0.7, 0.7, 0.7, 0.7,      # 03:00-04:00
                0.7, 0.7, 0.7, 0.7,      # 04:00-05:00
                0.7, 0.75, 0.8, 0.85,    # 05:00-06:00
                0.9, 0.95, 1.0, 1.0,     # 06:00-07:00 (shift start)
                1.0, 1.0, 1.0, 1.0,      # 07:00-08:00
                1.0, 1.0, 1.0, 1.0,      # 08:00-09:00
                1.0, 1.0, 1.0, 1.0,      # 09:00-10:00
                1.0, 1.0, 1.0, 1.0,      # 10:00-11:00
                1.0, 1.0, 1.0, 1.0,      # 11:00-12:00
                1.0, 1.0, 1.0, 1.0,      # 12:00-13:00
                1.0, 1.0, 1.0, 1.0,      # 13:00-14:00
                1.0, 1.0, 1.0, 1.0,      # 14:00-15:00
                1.0, 0.95, 0.9, 0.85,    # 15:00-16:00 (shift change)
                0.8, 0.8, 0.8, 0.8,      # 16:00-17:00
                0.8, 0.8, 0.8, 0.8,      # 17:00-18:00
                0.8, 0.8, 0.8, 0.8,      # 18:00-19:00
                0.8, 0.8, 0.8, 0.8,      # 19:00-20:00
                0.8, 0.75, 0.75, 0.75,   # 20:00-21:00
                0.75, 0.75, 0.7, 0.7,    # 21:00-22:00
                0.7, 0.7, 0.7, 0.7,      # 22:00-23:00
                0.7, 0.7, 0.7, 0.7       # 23:00-00:00
            ])
            peak_demand = building['area'] * 0.08  # 80W/m2 peak
        
        # Generate full time series
        daily_pattern = base_pattern * peak_demand
        
        # Add some randomness and weekly variation
        full_profile = []
        for day in range(days):
            if day % 7 in [5, 6]:  # Weekend
                weekend_factor = 0.6 if building['non_residential_type'] in ['Office', 'Retail'] else 1.1
            else:
                weekend_factor = 1.0
            
            daily_values = daily_pattern * weekend_factor
            # Add noise
            daily_values = daily_values * (1 + np.random.normal(0, 0.05, len(daily_values)))
            daily_values = np.maximum(daily_values, 0)  # No negative values
            full_profile.extend(daily_values)
        
        # Convert to numpy array for multiplication
        full_profile = np.array(full_profile)
        
        profiles[building_id] = {
            'timestamps': time_index[:len(full_profile)],
            'electricity_demand_kw': full_profile,
            'heating_demand_kw': full_profile * 0.5 * (1.5 if building['age_range'] == '<1945' else 1.0),
            'cooling_demand_kw': full_profile * 0.3 * (0.5 if building['building_function'] == 'residential' else 1.0)
        }
    
    return profiles

# ============================================
# 4. CREATE SOLAR & BATTERY PROFILES
# ============================================

def create_solar_profiles(buildings_df, profiles_dict):
    """Add solar generation profiles for suitable buildings"""
    
    solar_profiles = {}
    
    for _, building in buildings_df.iterrows():
        building_id = building['ogc_fid']
        
        # Solar potential based on roof and orientation
        if building['roof_area'] > 50:  # Minimum 50m2 roof
            # Solar capacity (kWp) - 0.15 kW/m2 panel efficiency, 0.7 usable roof area
            if building['building_orientation_cardinal'] in ['S', 'SE', 'SW']:
                solar_capacity_kwp = building['roof_area'] * 0.15 * 0.7
            elif building['building_orientation_cardinal'] in ['E', 'W']:
                solar_capacity_kwp = building['roof_area'] * 0.15 * 0.5
            elif building['building_orientation_cardinal'] == 'Flat':
                solar_capacity_kwp = building['flat_roof_area'] * 0.15 * 0.8
            else:
                solar_capacity_kwp = building['roof_area'] * 0.15 * 0.3
            
            # Generate daily solar pattern (simplified)
            solar_pattern = np.array([
                0, 0, 0, 0,              # 00:00-01:00
                0, 0, 0, 0,              # 01:00-02:00
                0, 0, 0, 0,              # 02:00-03:00
                0, 0, 0, 0,              # 03:00-04:00
                0, 0, 0, 0,              # 04:00-05:00
                0, 0, 0.02, 0.05,        # 05:00-06:00 (sunrise)
                0.1, 0.15, 0.2, 0.25,    # 06:00-07:00
                0.3, 0.35, 0.4, 0.45,    # 07:00-08:00
                0.5, 0.55, 0.6, 0.65,    # 08:00-09:00
                0.7, 0.75, 0.8, 0.85,    # 09:00-10:00
                0.9, 0.93, 0.96, 0.98,   # 10:00-11:00
                1.0, 1.0, 1.0, 1.0,      # 11:00-12:00 (peak)
                1.0, 1.0, 0.98, 0.96,    # 12:00-13:00
                0.93, 0.9, 0.85, 0.8,    # 13:00-14:00
                0.75, 0.7, 0.65, 0.6,    # 14:00-15:00
                0.55, 0.5, 0.45, 0.4,    # 15:00-16:00
                0.35, 0.3, 0.25, 0.2,    # 16:00-17:00
                0.15, 0.1, 0.05, 0.02,   # 17:00-18:00 (sunset)
                0, 0, 0, 0,              # 18:00-19:00
                0, 0, 0, 0,              # 19:00-20:00
                0, 0, 0, 0,              # 20:00-21:00
                0, 0, 0, 0,              # 21:00-22:00
                0, 0, 0, 0,              # 22:00-23:00
                0, 0, 0, 0               # 23:00-00:00
            ])
            
            # Full profile with weather variation
            days = len(profiles_dict[building_id]['timestamps']) // 96
            full_solar = []
            for day in range(days):
                weather_factor = random.choice([0.2, 0.5, 0.8, 0.9, 1.0])  # Cloud cover
                daily_solar = solar_pattern * solar_capacity_kwp * weather_factor
                full_solar.extend(daily_solar)
            
            solar_profiles[building_id] = {
                'solar_capacity_kwp': solar_capacity_kwp,
                'solar_generation_kw': full_solar[:len(profiles_dict[building_id]['timestamps'])]
            }
        else:
            solar_profiles[building_id] = {
                'solar_capacity_kwp': 0,
                'solar_generation_kw': np.zeros(len(profiles_dict[building_id]['timestamps']))
            }
    
    return solar_profiles

def create_battery_profiles(buildings_df, profiles_dict, solar_profiles):
    """Add battery storage and dispatch profiles"""
    
    battery_profiles = {}
    
    for _, building in buildings_df.iterrows():
        building_id = building['ogc_fid']
        
        # Battery sizing (only for buildings with solar)
        if solar_profiles[building_id]['solar_capacity_kwp'] > 0:
            # Battery capacity: 2 kWh per kWp of solar
            battery_capacity_kwh = solar_profiles[building_id]['solar_capacity_kwp'] * 2
            battery_power_kw = battery_capacity_kwh / 4  # C/4 rate
            
            # Simple dispatch logic
            battery_soc = []  # State of charge
            battery_charge = []
            battery_discharge = []
            
            current_soc = battery_capacity_kwh * 0.5  # Start at 50%
            
            for i in range(len(profiles_dict[building_id]['timestamps'])):
                demand = profiles_dict[building_id]['electricity_demand_kw'][i]
                solar = solar_profiles[building_id]['solar_generation_kw'][i]
                
                net_demand = demand - solar
                
                if net_demand < 0:  # Excess solar - charge battery
                    charge_power = min(-net_demand, battery_power_kw, 
                                      battery_capacity_kwh - current_soc)
                    discharge_power = 0
                    current_soc = min(current_soc + charge_power * 0.25, battery_capacity_kwh)
                    
                elif net_demand > 0 and current_soc > battery_capacity_kwh * 0.2:  # Need power - discharge
                    discharge_power = min(net_demand, battery_power_kw, 
                                         current_soc - battery_capacity_kwh * 0.2)
                    charge_power = 0
                    current_soc = max(current_soc - discharge_power * 0.25, 0)
                else:
                    charge_power = 0
                    discharge_power = 0
                
                battery_soc.append(current_soc)
                battery_charge.append(charge_power)
                battery_discharge.append(discharge_power)
            
            battery_profiles[building_id] = {
                'battery_capacity_kwh': battery_capacity_kwh,
                'battery_power_kw': battery_power_kw,
                'battery_soc_kwh': battery_soc,
                'battery_charge_kw': battery_charge,
                'battery_discharge_kw': battery_discharge
            }
        else:
            battery_profiles[building_id] = {
                'battery_capacity_kwh': 0,
                'battery_power_kw': 0,
                'battery_soc_kwh': np.zeros(len(profiles_dict[building_id]['timestamps'])),
                'battery_charge_kw': np.zeros(len(profiles_dict[building_id]['timestamps'])),
                'battery_discharge_kw': np.zeros(len(profiles_dict[building_id]['timestamps']))
            }
    
    return battery_profiles

# ============================================
# 5. GENERATE ALL DATA
# ============================================

print("Creating mimic data...")

# Generate grid topology
grid_data = create_grid_topology()
print(f"[OK] Created grid: {len(grid_data['substations'])} substations, "
      f"{len(grid_data['mv_transformers'])} MV transformers, "
      f"{len(grid_data['lv_networks'])} LV networks")

# Generate buildings
buildings_df = create_buildings(grid_data)
print(f"[OK] Created {len(buildings_df)} buildings")

# ADD SHARED WALL DATA
buildings_df = add_shared_wall_data(buildings_df)
print(f"[OK] Added shared wall data:")
print(f"  - Isolated buildings: {(buildings_df['adjacency_type'] == 'ISOLATED').sum()}")
print(f"  - End units: {(buildings_df['adjacency_type'] == 'END_UNIT').sum()}")
print(f"  - Middle row: {(buildings_df['adjacency_type'] == 'MIDDLE_ROW').sum()}")
print(f"  - Corner buildings: {(buildings_df['adjacency_type'] == 'CORNER').sum()}")
print(f"  - Three-sided: {(buildings_df['adjacency_type'] == 'THREE_SIDED').sum()}")


# Generate energy profiles
energy_profiles = create_energy_profiles(buildings_df, days=7)
print(f"[OK] Created energy profiles for {len(energy_profiles)} buildings")

# Generate solar profiles
solar_profiles = create_solar_profiles(buildings_df, energy_profiles)
solar_count = sum(1 for v in solar_profiles.values() if v['solar_capacity_kwp'] > 0)
print(f"[OK] Created solar profiles ({solar_count} buildings with solar)")

# Generate battery profiles
battery_profiles = create_battery_profiles(buildings_df, energy_profiles, solar_profiles)
battery_count = sum(1 for v in battery_profiles.values() if v['battery_capacity_kwh'] > 0)
print(f"[OK] Created battery profiles ({battery_count} buildings with batteries)")




# Add this section BEFORE saving the buildings CSV (around line 800 in paste.txt)
# This should be added RIGHT AFTER creating solar_profiles and battery_profiles
# and BEFORE the "Save to files" section

# ============================================
# INTEGRATE SOLAR & BATTERY DATA INTO BUILDINGS
# ============================================

# Add solar and battery capacity to buildings dataframe
buildings_df['solar_capacity_kwp'] = 0.0
buildings_df['battery_capacity_kwh'] = 0.0
buildings_df['battery_power_kw'] = 0.0
buildings_df['has_solar'] = False
buildings_df['has_battery'] = False

for building_id in buildings_df['ogc_fid']:
    if building_id in solar_profiles:
        solar_cap = solar_profiles[building_id]['solar_capacity_kwp']
        buildings_df.loc[buildings_df['ogc_fid'] == building_id, 'solar_capacity_kwp'] = solar_cap
        buildings_df.loc[buildings_df['ogc_fid'] == building_id, 'has_solar'] = solar_cap > 0
    
    if building_id in battery_profiles:
        battery_cap = battery_profiles[building_id]['battery_capacity_kwh']
        battery_power = battery_profiles[building_id]['battery_power_kw']
        buildings_df.loc[buildings_df['ogc_fid'] == building_id, 'battery_capacity_kwh'] = battery_cap
        buildings_df.loc[buildings_df['ogc_fid'] == building_id, 'battery_power_kw'] = battery_power
        buildings_df.loc[buildings_df['ogc_fid'] == building_id, 'has_battery'] = battery_cap > 0

print(f"[OK] Integrated solar/battery data into buildings:")
print(f"  - Buildings with solar: {buildings_df['has_solar'].sum()}")
print(f"  - Buildings with battery: {buildings_df['has_battery'].sum()}")
print(f"  - Average solar capacity: {buildings_df[buildings_df['has_solar']]['solar_capacity_kwp'].mean():.2f} kWp")
print(f"  - Average battery capacity: {buildings_df[buildings_df['has_battery']]['battery_capacity_kwh'].mean():.2f} kWh")









# ============================================
# 6. SAVE DATA
# ============================================

# Save to files
buildings_df.to_csv('mimic_data/buildings.csv', index=False)
pd.DataFrame(grid_data['lv_networks']).to_csv('mimic_data/lv_networks.csv', index=False)
pd.DataFrame(grid_data['mv_transformers']).to_csv('mimic_data/mv_transformers.csv', index=False)

# Save profiles as parquet for efficiency
profiles_df = pd.DataFrame()
for building_id, profile in energy_profiles.items():
    df_temp = pd.DataFrame({
        'building_id': building_id,
        'timestamp': profile['timestamps'],
        'electricity_demand_kw': profile['electricity_demand_kw'],
        'heating_demand_kw': profile['heating_demand_kw'],
        'cooling_demand_kw': profile['cooling_demand_kw'],
        'solar_generation_kw': solar_profiles[building_id]['solar_generation_kw'],
        'battery_soc_kwh': battery_profiles[building_id]['battery_soc_kwh'],
        'battery_charge_kw': battery_profiles[building_id]['battery_charge_kw'],
        'battery_discharge_kw': battery_profiles[building_id]['battery_discharge_kw']
    })
    profiles_df = pd.concat([profiles_df, df_temp], ignore_index=True)

profiles_df.to_parquet('mimic_data/energy_profiles.parquet', index=False)

print("\n[DONE] Mimic data generation complete!")
print(f"[FILES] Files saved in 'mimic_data/' directory")

# ============================================
# 7. DATA SUMMARY
# ============================================

print("\n" + "="*50)
print("DATA SUMMARY")
print("="*50)

# LV Network summary
lv_summary = buildings_df.groupby('lv_network_id').agg({
    'ogc_fid': 'count',
    'building_function': lambda x: (x == 'residential').sum(),
    'area': 'sum',
    'roof_area': 'sum'
}).rename(columns={
    'ogc_fid': 'total_buildings',
    'building_function': 'residential_count',
    'area': 'total_area_m2',
    'roof_area': 'total_roof_area_m2'
})

lv_summary['non_residential_count'] = lv_summary['total_buildings'] - lv_summary['residential_count']
lv_summary['res_nonres_ratio'] = lv_summary['residential_count'] / lv_summary['non_residential_count'].replace(0, 1)

print("\nLV Network Statistics:")
print(lv_summary)

print("\n" + "="*50)


print("Creating mimic data...")

# Generate grid topology
grid_data = create_grid_topology()
print(f"[OK] Created grid: {len(grid_data['substations'])} substations, "
      f"{len(grid_data['mv_transformers'])} MV transformers, "
      f"{len(grid_data['lv_networks'])} LV networks")

# Generate buildings
buildings_df = create_buildings(grid_data)
print(f"[OK] Created {len(buildings_df)} buildings")

# ADD SHARED WALL DATA
buildings_df = add_shared_wall_data(buildings_df)
print(f"[OK] Added shared wall data:")
print(f"  - Isolated buildings: {(buildings_df['adjacency_type'] == 'ISOLATED').sum()}")
print(f"  - End units: {(buildings_df['adjacency_type'] == 'END_UNIT').sum()}")
print(f"  - Middle row: {(buildings_df['adjacency_type'] == 'MIDDLE_ROW').sum()}")
print(f"  - Corner buildings: {(buildings_df['adjacency_type'] == 'CORNER').sum()}")
print(f"  - Three-sided: {(buildings_df['adjacency_type'] == 'THREE_SIDED').sum()}")
