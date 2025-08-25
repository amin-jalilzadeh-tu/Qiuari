import numpy as np
import pandas as pd
import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.data_loader import EnergyDataLoader as OriginalDataLoader
from data.feature_processor import FeatureProcessor
from data.kg_connector import KGConnector
import torch

def prepare_comparison_data():
    # For now, we'll use synthetic data for the comparison
    return generate_synthetic_data(n_buildings=100)

def generate_synthetic_data(n_buildings=100, n_timesteps=96):
    np.random.seed(42)
    
    base_profiles = {
        'residential': np.array([0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.7, 0.8, 
                                0.6, 0.5, 0.4, 0.4, 0.5, 0.6, 0.5, 0.5,
                                0.6, 0.8, 0.9, 0.8, 0.7, 0.5, 0.4, 0.3]),
        'commercial': np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.8,
                               0.9, 0.9, 0.9, 0.8, 0.8, 0.9, 0.9, 0.9,
                               0.8, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2])
    }
    
    consumption = np.zeros((n_timesteps, n_buildings))
    generation = np.zeros((n_timesteps, n_buildings))
    
    for i in range(n_buildings):
        profile_type = 'residential' if i % 3 != 0 else 'commercial'
        base = base_profiles[profile_type]
        
        hourly_profile = base * np.random.uniform(5, 15)
        noise = np.random.normal(0, 0.1, 24)
        hourly_profile = np.maximum(0, hourly_profile + noise)
        
        for h in range(24):
            consumption[h*4:(h+1)*4, i] = hourly_profile[h]
        
        if i % 3 == 0:
            solar_peak = np.random.uniform(3, 8)
            for t in range(n_timesteps):
                hour = t / 4
                if 6 <= hour <= 18:
                    generation[t, i] = solar_peak * np.sin((hour - 6) * np.pi / 12)
    
    G = nx.watts_strogatz_graph(n_buildings, 6, 0.3)
    
    building_features = {
        'energy_label': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'], n_buildings),
        'area': np.random.uniform(50, 200, n_buildings),
        'roof_area': np.random.uniform(20, 100, n_buildings)
    }
    
    lv_groups = []
    group_size = 10
    for i in range(0, n_buildings, group_size):
        lv_groups.append(list(range(i, min(i + group_size, n_buildings))))
    
    return {
        'consumption': consumption,
        'generation': generation,
        'grid_topology': G,
        'building_features': building_features,
        'constraints': {
            'lv_groups': lv_groups,
            'transformer_capacity': 250.0
        }
    }