# data/feature_engineering.py
"""
Advanced feature engineering for energy GNN
Creates rich node and edge features from raw data
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional
from scipy import signal, stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Create advanced features for GNN model"""
    
    def __init__(self, config: Dict):
        """Initialize feature engineer"""
        self.config = config
        self.scalers = {}
        
    def create_node_features(self, nodes: pd.DataFrame, 
                            temporal_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Create comprehensive node feature matrix
        
        Args:
            nodes: Building nodes DataFrame
            temporal_data: Energy consumption time series
            
        Returns:
            Feature matrix (n_nodes x n_features)
        """
        features = []
        
        # 1. Static features
        static_features = self._extract_static_features(nodes)
        features.append(static_features)
        
        # 2. Temporal features
        if temporal_data is not None:
            temporal_features = self._extract_temporal_features(nodes, temporal_data)
            features.append(temporal_features)
        
        # 3. Spatial features
        spatial_features = self._extract_spatial_features(nodes)
        features.append(spatial_features)
        
        # 4. Building physics features
        physics_features = self._extract_physics_features(nodes)
        features.append(physics_features)
        
        # 5. Economic features
        economic_features = self._extract_economic_features(nodes)
        features.append(economic_features)
        
        # Concatenate all features
        feature_matrix = np.hstack(features)
        
        logger.info(f"Created feature matrix: {feature_matrix.shape}")
        return feature_matrix
    
    def _extract_static_features(self, nodes: pd.DataFrame) -> np.ndarray:
        """Extract static building features"""
        features = []
        
        # Geometric features
        features.append(nodes['area'].fillna(0).values.reshape(-1, 1))
        features.append(nodes['height'].fillna(0).values.reshape(-1, 1))
        features.append(nodes['roof_area'].fillna(0).values.reshape(-1, 1))
        features.append(nodes['suitable_roof'].fillna(0).values.reshape(-1, 1))
        
        # Calculate additional geometric features
        volume = nodes['area'] * nodes['height']
        surface_area = 2 * nodes['area'] + 4 * np.sqrt(nodes['area']) * nodes['height']
        compactness = volume / surface_area
        
        features.append(volume.fillna(0).values.reshape(-1, 1))
        features.append(surface_area.fillna(0).values.reshape(-1, 1))
        features.append(compactness.fillna(0).values.reshape(-1, 1))
        
        # Age features (convert to numeric)
        age_mapping = {
            'pre_1945': 80,
            '1945_1975': 60,
            '1975_1995': 40,
            '1995_2010': 20,
            'post_2010': 10
        }
        age_numeric = nodes['age'].map(age_mapping).fillna(30)
        features.append(age_numeric.values.reshape(-1, 1))
        
        # Orientation features (convert to sine/cosine)
        orientation_mapping = {
            'N': 0, 'NE': 45, 'E': 90, 'SE': 135,
            'S': 180, 'SW': 225, 'W': 270, 'NW': 315
        }
        orientation_deg = nodes['orientation'].map(orientation_mapping).fillna(0)
        orientation_rad = np.deg2rad(orientation_deg)
        features.append(np.sin(orientation_rad).values.reshape(-1, 1))
        features.append(np.cos(orientation_rad).values.reshape(-1, 1))
        
        # Energy label (ordinal encoding)
        label_mapping = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
        energy_label = nodes['energy_label'].map(label_mapping).fillna(3)
        features.append(energy_label.values.reshape(-1, 1))
        
        # Equipment binary features
        features.append(nodes['has_solar'].astype(float).values.reshape(-1, 1))
        features.append(nodes['has_battery'].astype(float).values.reshape(-1, 1))
        features.append(nodes['has_heat_pump'].astype(float).values.reshape(-1, 1))
        
        # Scale features
        static_array = np.hstack(features)
        scaler = StandardScaler()
        static_scaled = scaler.fit_transform(static_array)
        self.scalers['static'] = scaler
        
        return static_scaled
    
    def _extract_temporal_features(self, nodes: pd.DataFrame, 
                                  temporal_data: pd.DataFrame) -> np.ndarray:
        """Extract temporal features from energy profiles"""
        features = []
        
        for building_id in nodes['id']:
            if building_id in temporal_data.columns:
                profile = temporal_data[building_id].values
                
                # Statistical features
                feat = [
                    np.mean(profile),
                    np.std(profile),
                    np.min(profile),
                    np.max(profile),
                    np.percentile(profile, 25),
                    np.percentile(profile, 50),
                    np.percentile(profile, 75),
                    stats.skew(profile),
                    stats.kurtosis(profile)
                ]
                
                # Load duration curve features
                sorted_profile = np.sort(profile)[::-1]
                feat.extend([
                    sorted_profile[int(0.01 * len(profile))],  # 1% peak
                    sorted_profile[int(0.05 * len(profile))],  # 5% peak
                    sorted_profile[int(0.10 * len(profile))]   # 10% peak
                ])
                
                # Ramp rates
                diff = np.diff(profile)
                feat.extend([
                    np.mean(np.abs(diff)),  # Mean absolute change
                    np.max(diff),  # Max ramp up
                    np.min(diff)   # Max ramp down
                ])
                
                # Frequency domain features (FFT)
                fft = np.fft.fft(profile)
                freqs = np.fft.fftfreq(len(profile))
                
                # Find dominant frequencies
                magnitude = np.abs(fft)
                dominant_freq_idx = np.argsort(magnitude)[-5:]  # Top 5 frequencies
                feat.extend(freqs[dominant_freq_idx].tolist())
                feat.extend(magnitude[dominant_freq_idx].tolist())
                
                # Autocorrelation features
                autocorr_24h = np.correlate(profile[:96], profile[96:192], mode='valid')[0]
                autocorr_7d = np.correlate(profile[:96], profile[576:672], mode='valid')[0]
                feat.extend([autocorr_24h, autocorr_7d])
                
                features.append(feat)
            else:
                # No temporal data - use zeros
                features.append(np.zeros(27))  # Match feature count
        
        temporal_array = np.array(features)
        
        # Scale features
        scaler = StandardScaler()
        temporal_scaled = scaler.fit_transform(temporal_array)
        self.scalers['temporal'] = scaler
        
        return temporal_scaled
    
    def _extract_spatial_features(self, nodes: pd.DataFrame) -> np.ndarray:
        """Extract spatial and neighborhood features"""
        features = []
        
        # Coordinates
        x_coords = nodes['x'].fillna(0).values
        y_coords = nodes['y'].fillna(0).values
        
        # Normalize coordinates
        x_norm = (x_coords - x_coords.mean()) / (x_coords.std() + 1e-6)
        y_norm = (y_coords - y_coords.mean()) / (y_coords.std() + 1e-6)
        
        features.append(x_norm.reshape(-1, 1))
        features.append(y_norm.reshape(-1, 1))
        
        # Distance to centroid
        centroid_x = x_coords.mean()
        centroid_y = y_coords.mean()
        dist_to_center = np.sqrt((x_coords - centroid_x)**2 + (y_coords - centroid_y)**2)
        features.append(dist_to_center.reshape(-1, 1))
        
        # Density features (buildings within radius)
        densities = []
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            distances = np.sqrt((x_coords - x)**2 + (y_coords - y)**2)
            density_100m = np.sum(distances < 100)
            density_250m = np.sum(distances < 250)
            density_500m = np.sum(distances < 500)
            densities.append([density_100m, density_250m, density_500m])
        
        features.append(np.array(densities))
        
        # Adjacency features
        shared_walls = nodes['shared_walls'].fillna(0).values
        shared_length = nodes['shared_length'].fillna(0).values
        
        features.append(shared_walls.reshape(-1, 1))
        features.append(shared_length.reshape(-1, 1))
        
        spatial_array = np.hstack(features)
        
        # Scale features
        scaler = StandardScaler()
        spatial_scaled = scaler.fit_transform(spatial_array)
        self.scalers['spatial'] = scaler
        
        return spatial_scaled
    
    def _extract_physics_features(self, nodes: pd.DataFrame) -> np.ndarray:
        """Extract building physics and energy features"""
        features = []
        
        # Heat loss coefficient estimate
        area = nodes['area'].fillna(100)
        height = nodes['height'].fillna(3)
        age_factor = nodes['age'].map({
            'pre_1945': 2.0, '1945_1975': 1.8, '1975_1995': 1.5,
            '1995_2010': 1.2, 'post_2010': 1.0
        }).fillna(1.5)
        
        # Simplified heat loss calculation
        wall_area = 4 * np.sqrt(area) * height
        u_value = 0.3 * age_factor  # W/m²K
        heat_loss = (area + wall_area) * u_value
        
        features.append(heat_loss.values.reshape(-1, 1))
        
        # Thermal mass estimate
        thermal_mass = area * height * 200  # kg (assuming 200 kg/m³)
        features.append(thermal_mass.values.reshape(-1, 1))
        
        # Solar gain potential
        roof_area = nodes['roof_area'].fillna(0)
        orientation_factor = nodes['orientation'].map({
            'S': 1.0, 'SE': 0.95, 'SW': 0.95, 'E': 0.85,
            'W': 0.85, 'NE': 0.7, 'NW': 0.7, 'N': 0.6
        }).fillna(0.8)
        
        solar_potential = roof_area * orientation_factor * 5  # kWh/m²/day * 5 hours
        features.append(solar_potential.values.reshape(-1, 1))
        
        # Electrification readiness score
        elec_score = (
            nodes['heat_pump_ready'].map({'ready': 1, 'conditional': 0.5, 'upgrade_needed': 0}).fillna(0) * 0.4 +
            nodes['insulation'].map({'excellent': 1, 'good': 0.75, 'fair': 0.5, 'poor': 0.25}).fillna(0.5) * 0.3 +
            (nodes['area'] < 200).astype(float) * 0.3  # Smaller buildings easier to electrify
        )
        features.append(elec_score.values.reshape(-1, 1))
        
        physics_array = np.hstack(features)
        
        # Scale features
        scaler = StandardScaler()
        physics_scaled = scaler.fit_transform(physics_array)
        self.scalers['physics'] = scaler
        
        return physics_scaled
    
    def _extract_economic_features(self, nodes: pd.DataFrame) -> np.ndarray:
        """Extract economic and investment features"""
        features = []
        
        # Investment potential based on building type and size
        area = nodes['area'].fillna(100)
        
        type_factor = nodes['function'].map({
            'residential': 1.0,
            'commercial': 1.5,
            'industrial': 1.2
        }).fillna(1.0)
        
        investment_potential = area * type_factor * 100  # €/m² rough estimate
        features.append(investment_potential.values.reshape(-1, 1))
        
        # Energy cost estimate
        peak_demand = nodes['peak_demand'].fillna(10)
        avg_demand = nodes['avg_demand'].fillna(5)
        
        # Annual energy cost estimate
        annual_consumption = avg_demand * 8760  # kWh/year
        energy_cost = annual_consumption * 0.25  # €0.25/kWh average
        features.append(energy_cost.values.reshape(-1, 1))
        
        # Payback sensitivity
        has_solar = nodes['has_solar'].astype(float)
        solar_capacity = nodes['solar_kw'].fillna(0)
        
        # Existing solar value
        solar_value = solar_capacity * 1200 * 0.25  # kWh/kWp/year * price
        features.append(solar_value.values.reshape(-1, 1))
        
        # Demand flexibility value
        load_factor = nodes['load_factor'].fillna(0.5)
        variability = nodes['variability'].fillna(0.2)
        
        flexibility_score = (1 - load_factor) * variability  # Higher is more flexible
        features.append(flexibility_score.values.reshape(-1, 1))
        
        economic_array = np.hstack(features)
        
        # Scale features
        scaler = StandardScaler()
        economic_scaled = scaler.fit_transform(economic_array)
        self.scalers['economic'] = scaler
        
        return economic_scaled
    
    def create_edge_features(self, edges: pd.DataFrame, 
                            nodes: pd.DataFrame) -> np.ndarray:
        """
        Create edge feature matrix
        
        Args:
            edges: Edge DataFrame with source, target
            nodes: Node DataFrame for additional info
            
        Returns:
            Edge feature matrix (n_edges x n_features)
        """
        features = []
        
        for _, edge in edges.iterrows():
            source_id = edge['source']
            target_id = edge['target']
            
            # Get node info
            source_node = nodes[nodes['id'] == source_id].iloc[0]
            target_node = nodes[nodes['id'] == target_id].iloc[0]
            
            edge_feat = []
            
            # Distance
            dist = np.sqrt((source_node['x'] - target_node['x'])**2 + 
                          (source_node['y'] - target_node['y'])**2)
            edge_feat.append(dist)
            
            # Shared wall length (if adjacent)
            shared_wall = edge.get('shared_wall', 0)
            edge_feat.append(shared_wall)
            
            # Type compatibility
            same_type = float(source_node['function'] == target_node['function'])
            edge_feat.append(same_type)
            
            # Size ratio
            size_ratio = min(source_node['area'], target_node['area']) / \
                        max(source_node['area'], target_node['area'])
            edge_feat.append(size_ratio)
            
            # Peak demand difference
            peak_diff = abs(source_node['peak_demand'] - target_node['peak_demand'])
            edge_feat.append(peak_diff)
            
            # Load factor similarity
            lf_diff = abs(source_node['load_factor'] - target_node['load_factor'])
            edge_feat.append(lf_diff)
            
            features.append(edge_feat)
        
        edge_array = np.array(features)
        
        # Scale features
        scaler = StandardScaler()
        edge_scaled = scaler.fit_transform(edge_array)
        self.scalers['edges'] = scaler
        
        return edge_scaled
    
    def create_global_features(self, nodes: pd.DataFrame, 
                              temporal_data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """Create global graph-level features"""
        features = []
        
        # Graph statistics
        features.append(len(nodes))  # Number of nodes
        features.append(nodes['area'].sum())  # Total area
        features.append(nodes['peak_demand'].sum())  # Total peak demand
        features.append(nodes['has_solar'].sum())  # Solar installations
        features.append(nodes['has_battery'].sum())  # Battery installations
        
        # Diversity metrics
        building_types = nodes['function'].value_counts()
        type_entropy = stats.entropy(building_types)
        features.append(type_entropy)
        
        # Energy metrics
        if temporal_data is not None:
            total_profile = temporal_data.sum(axis=1)
            features.append(total_profile.max())  # System peak
            features.append(total_profile.mean())  # System average
            features.append(total_profile.std())  # System variability
            
            # System load factor
            system_lf = total_profile.mean() / total_profile.max()
            features.append(system_lf)
        
        return np.array(features)

# Usage example
if __name__ == "__main__":
    import yaml
    import pickle
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load extracted data
    with open('processed_data/extracted_graph.pkl', 'rb') as f:
        extracted_data = pickle.load(f)
    
    # Create feature engineer
    engineer = FeatureEngineer(config)
    
    # Create node features
    nodes = extracted_data['nodes']['buildings']
    temporal = extracted_data['temporal'].get('energy_profiles')
    
    node_features = engineer.create_node_features(nodes, temporal)
    print(f"Node features shape: {node_features.shape}")
    
    # Create edge features
    if 'spatial' in extracted_data['edges']:
        edges = extracted_data['edges']['spatial']
        edge_features = engineer.create_edge_features(edges, nodes)
        print(f"Edge features shape: {edge_features.shape}")
    
    # Create global features
    global_features = engineer.create_global_features(nodes, temporal)
    print(f"Global features shape: {global_features.shape}")