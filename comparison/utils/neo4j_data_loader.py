import numpy as np
import pandas as pd
import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.kg_connector import KGConnector
import torch

def load_neo4j_data():
    """Load real data from Neo4j database"""
    print("Connecting to Neo4j database...")
    
    kg_connector = KGConnector(
        uri="bolt://localhost:7687",
        user="neo4j",
        password="12345678"
    )
    
    try:
        # First, try to get statistics to understand the data
        stats = kg_connector.get_statistics()
        print(f"Database statistics: {stats}")
        
        # Get all LV groups
        lv_groups = kg_connector.get_all_lv_groups()
        print(f"Found {len(lv_groups)} LV groups")
        
        if not lv_groups:
            print("No LV groups found, using synthetic data")
            kg_connector.close()
            return generate_synthetic_from_stats(stats)
        
        # Collect buildings and their data
        all_buildings = []
        building_to_idx = {}
        consumption_list = []
        generation_list = []
        lv_group_assignments = {}
        
        # Process a subset of LV groups for testing
        max_groups = min(10, len(lv_groups))  # Process up to 10 groups
        
        for group_idx, lv_group_id in enumerate(lv_groups[:max_groups]):
            print(f"Processing LV group {group_idx+1}/{max_groups}: {lv_group_id}")
            
            # Get LV group data
            lv_data = kg_connector.get_lv_group_data(lv_group_id)
            
            if lv_data and 'buildings' in lv_data:
                for building in lv_data['buildings']:
                    building_id = str(building.get('ogc_fid', building.get('id', '')))
                    
                    if building_id and building_id not in building_to_idx:
                        idx = len(all_buildings)
                        building_to_idx[building_id] = idx
                        all_buildings.append(building)
                        
                        # Assign to LV group
                        if lv_group_id not in lv_group_assignments:
                            lv_group_assignments[lv_group_id] = []
                        lv_group_assignments[lv_group_id].append(idx)
                        
                        # Generate synthetic consumption/generation for now
                        # (since time series might not be available)
                        consumption = generate_consumption_profile()
                        generation = generate_generation_profile(building)
                        
                        consumption_list.append(consumption)
                        generation_list.append(generation)
        
        if not all_buildings:
            print("No buildings found, using synthetic data")
            kg_connector.close()
            return generate_synthetic_from_stats(stats)
        
        n_buildings = len(all_buildings)
        print(f"Loaded {n_buildings} buildings from {len(lv_group_assignments)} LV groups")
        
        # Convert to numpy arrays
        consumption_matrix = np.array(consumption_list).T  # Shape: (96, n_buildings)
        generation_matrix = np.array(generation_list).T    # Shape: (96, n_buildings)
        
        # Create network graph based on LV group connections
        G = nx.Graph()
        for i in range(n_buildings):
            G.add_node(i)
        
        # Connect buildings within same LV group
        for lv_group, building_indices in lv_group_assignments.items():
            for i in range(len(building_indices)):
                for j in range(i+1, len(building_indices)):
                    G.add_edge(building_indices[i], building_indices[j])
        
        # Add some inter-group connections for network connectivity
        lv_group_list = list(lv_group_assignments.values())
        for i in range(len(lv_group_list)-1):
            if lv_group_list[i] and lv_group_list[i+1]:
                G.add_edge(lv_group_list[i][0], lv_group_list[i+1][0])
        
        # Extract building features and locations
        energy_labels = []
        areas = []
        roof_areas = []
        locations = []
        
        for building in all_buildings:
            energy_labels.append(building.get('energy_label', 'D'))
            areas.append(float(building.get('living_area', 100)))
            roof_areas.append(float(building.get('suitable_roof_area', 30)))
            
            # Extract location (lat, lon)
            lat = building.get('latitude', building.get('lat', 52.3676 + np.random.randn() * 0.01))
            lon = building.get('longitude', building.get('lon', 4.9041 + np.random.randn() * 0.01))
            locations.append([lat, lon])
        
        # Convert LV group assignments to list format
        lv_groups_list = list(lv_group_assignments.values())
        
        input_data = {
            'consumption': consumption_matrix,
            'generation': generation_matrix,
            'grid_topology': G,
            'building_features': {
                'energy_label': energy_labels,
                'area': np.array(areas),
                'roof_area': np.array(roof_areas)
            },
            'building_locations': np.array(locations),  # Add location data
            'constraints': {
                'lv_groups': lv_groups_list,
                'transformer_capacity': 250.0
            },
            'n_buildings': n_buildings
        }
        
        kg_connector.close()
        
        print(f"Successfully loaded real data: {n_buildings} buildings, {G.number_of_edges()} edges")
        return input_data
        
    except Exception as e:
        print(f"Error loading from Neo4j: {e}")
        kg_connector.close()
        
        # Fall back to synthetic data
        return generate_synthetic_from_stats({})

def generate_consumption_profile():
    """Generate realistic consumption profile (96 timesteps for 24h at 15min)"""
    # Base load pattern (residential)
    base_pattern = np.array([
        0.3, 0.3, 0.3, 0.3,  # 00:00 - 01:00
        0.3, 0.3, 0.3, 0.3,  # 01:00 - 02:00
        0.3, 0.3, 0.3, 0.3,  # 02:00 - 03:00
        0.3, 0.3, 0.3, 0.3,  # 03:00 - 04:00
        0.3, 0.3, 0.3, 0.3,  # 04:00 - 05:00
        0.4, 0.4, 0.5, 0.5,  # 05:00 - 06:00
        0.6, 0.7, 0.8, 0.8,  # 06:00 - 07:00
        0.7, 0.6, 0.5, 0.5,  # 07:00 - 08:00
        0.4, 0.4, 0.4, 0.4,  # 08:00 - 09:00
        0.4, 0.4, 0.4, 0.4,  # 09:00 - 10:00
        0.4, 0.4, 0.4, 0.4,  # 10:00 - 11:00
        0.5, 0.5, 0.5, 0.5,  # 11:00 - 12:00
        0.6, 0.6, 0.5, 0.5,  # 12:00 - 13:00
        0.5, 0.5, 0.5, 0.5,  # 13:00 - 14:00
        0.5, 0.5, 0.5, 0.5,  # 14:00 - 15:00
        0.5, 0.5, 0.5, 0.5,  # 15:00 - 16:00
        0.6, 0.6, 0.6, 0.6,  # 16:00 - 17:00
        0.7, 0.8, 0.9, 0.9,  # 17:00 - 18:00
        0.9, 0.9, 0.8, 0.8,  # 18:00 - 19:00
        0.7, 0.7, 0.6, 0.6,  # 19:00 - 20:00
        0.6, 0.6, 0.5, 0.5,  # 20:00 - 21:00
        0.5, 0.5, 0.4, 0.4,  # 21:00 - 22:00
        0.4, 0.4, 0.3, 0.3,  # 22:00 - 23:00
        0.3, 0.3, 0.3, 0.3   # 23:00 - 00:00
    ])
    
    # Add randomness
    scale = np.random.uniform(5, 15)  # kW peak
    noise = np.random.normal(0, 0.1, 96)
    
    return np.maximum(0, base_pattern * scale + noise)

def generate_generation_profile(building):
    """Generate solar generation profile based on building properties"""
    roof_area = building.get('suitable_roof_area', 0)
    
    if roof_area == 0 or np.random.random() > 0.3:  # 30% of buildings have solar
        return np.zeros(96)
    
    # Solar generation pattern (peak at noon)
    hours = np.arange(96) / 4  # Convert to hours
    solar_pattern = np.zeros(96)
    
    for i, hour in enumerate(hours):
        if 6 <= hour <= 18:  # Daylight hours
            solar_pattern[i] = np.sin((hour - 6) * np.pi / 12)
    
    # Scale by roof area (assuming 0.2 kW/m² efficiency)
    capacity = roof_area * 0.2 / 1000  # Convert to kW
    noise = np.random.normal(0, 0.05, 96)
    
    return np.maximum(0, solar_pattern * capacity + noise)

def generate_synthetic_from_stats(stats):
    """Generate synthetic data based on database statistics"""
    n_buildings = stats.get('total_buildings', 100)
    
    print(f"Generating synthetic data for {n_buildings} buildings based on Neo4j statistics")
    
    # Generate consumption and generation
    consumption_matrix = np.zeros((96, n_buildings))
    generation_matrix = np.zeros((96, n_buildings))
    
    for i in range(n_buildings):
        consumption_matrix[:, i] = generate_consumption_profile()
        
        # Synthetic building for generation
        synthetic_building = {'suitable_roof_area': np.random.uniform(0, 100)}
        generation_matrix[:, i] = generate_generation_profile(synthetic_building)
    
    # Create network
    G = nx.watts_strogatz_graph(n_buildings, 6, 0.3)
    
    # Create LV groups
    lv_groups = []
    group_size = 10
    for i in range(0, n_buildings, group_size):
        lv_groups.append(list(range(i, min(i + group_size, n_buildings))))
    
    # Generate features
    energy_labels = np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_buildings)
    areas = np.random.uniform(50, 200, n_buildings)
    roof_areas = np.random.uniform(0, 100, n_buildings)
    
    return {
        'consumption': consumption_matrix,
        'generation': generation_matrix,
        'grid_topology': G,
        'building_features': {
            'energy_label': list(energy_labels),
            'area': areas,
            'roof_area': roof_areas
        },
        'constraints': {
            'lv_groups': lv_groups,
            'transformer_capacity': 250.0
        },
        'n_buildings': n_buildings
    }