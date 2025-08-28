"""
Data Preprocessing Pipeline for KG-Based Energy Complementarity Clustering
Extracts and prepares all necessary data from Neo4j Knowledge Graph
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from scipy.stats import pearsonr
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from data.kg_connector import KGConnector

logger = logging.getLogger(__name__)

class KGDataPreprocessor:
    """
    Extract and preprocess data from Knowledge Graph for clustering algorithms.
    Focuses on complementarity rather than similarity.
    """
    
    def __init__(self, kg_connector: KGConnector):
        """
        Initialize preprocessor with KG connection.
        
        Args:
            kg_connector: Connected KGConnector instance
        """
        self.kg = kg_connector
        self.cache = {}
        logger.info("Initialized KG Data Preprocessor")
    
    def prepare_data_from_kg(self, district_name: str, 
                            lookback_hours: int = 168) -> Dict[str, Any]:
        """
        Extract and prepare all necessary data from KG.
        
        Args:
            district_name: District to analyze
            lookback_hours: Hours of time series history (default 1 week)
            
        Returns:
            Dictionary with all preprocessed data for clustering
        """
        logger.info(f"Preparing data for district: {district_name}")
        
        # 1. Get topology
        topology = self._get_topology(district_name)
        
        # 2. Get time series data
        building_ids = [b['ogc_fid'] for b in topology['nodes']['buildings']]
        time_series = self._get_time_series(building_ids, lookback_hours)
        
        # 3. Get adjacency relationships
        adjacency = self._get_adjacency(district_name)
        
        # 4. Create constraint matrices
        constraints = self._create_constraints(topology)
        
        # 5. Calculate complementarity scores
        complementarity = self._calculate_complementarity(time_series)
        
        # 6. Extract building features
        building_features = self._extract_building_features(topology['nodes']['buildings'])
        
        # 7. Calculate electrical distances
        electrical_distances = self._calculate_electrical_distances(topology)
        
        return {
            'district_name': district_name,
            'topology': topology,
            'time_series': time_series,
            'adjacency': adjacency,
            'constraints': constraints,
            'complementarity': complementarity,
            'building_features': building_features,
            'electrical_distances': electrical_distances,
            'metadata': {
                'n_buildings': len(building_ids),
                'n_cable_groups': len(topology['nodes']['cable_groups']),
                'n_transformers': len(topology['nodes']['transformers']),
                'lookback_hours': lookback_hours,
                'timestamp': datetime.now().isoformat()
            }
        }
    
    def _get_topology(self, district_name: str) -> Dict[str, Any]:
        """Get grid topology from KG."""
        logger.info("Fetching grid topology...")
        topology = self.kg.get_grid_topology(district_name)
        
        # Convert node objects to dictionaries with all properties
        for node_type in topology['nodes']:
            if topology['nodes'][node_type]:
                topology['nodes'][node_type] = [
                    dict(node) if hasattr(node, '__dict__') else node 
                    for node in topology['nodes'][node_type]
                ]
        
        logger.info(f"Found {len(topology['nodes']['buildings'])} buildings, "
                   f"{len(topology['nodes']['cable_groups'])} cable groups, "
                   f"{len(topology['nodes']['transformers'])} transformers")
        
        return topology
    
    def _get_time_series(self, building_ids: List[str], 
                        lookback_hours: int) -> Dict[str, np.ndarray]:
        """Get time series data for buildings."""
        logger.info(f"Fetching {lookback_hours} hours of time series for {len(building_ids)} buildings...")
        
        time_series = self.kg.get_building_time_series(
            building_ids, 
            lookback_hours=lookback_hours
        )
        
        # Ensure all buildings have data (create synthetic if missing)
        for bid in building_ids:
            bid_str = str(bid)
            if bid_str not in time_series or time_series[bid_str] is None:
                logger.warning(f"No time series for building {bid}, creating synthetic data")
                time_series[bid_str] = self._create_synthetic_profile(lookback_hours)
        
        logger.info(f"Retrieved time series for {len(time_series)} buildings")
        return time_series
    
    def _create_synthetic_profile(self, hours: int) -> np.ndarray:
        """Create synthetic energy profile for buildings without data."""
        # Create realistic daily patterns
        daily_pattern = np.array([
            0.5, 0.4, 0.35, 0.3, 0.3, 0.35,  # Night (0-6)
            0.6, 0.8, 0.9, 0.85, 0.8, 0.75,  # Morning (6-12)
            0.7, 0.65, 0.6, 0.65, 0.7, 0.8,  # Afternoon (12-18)
            0.9, 0.95, 0.85, 0.7, 0.6, 0.55  # Evening (18-24)
        ])
        
        # Repeat for number of days needed
        n_days = hours // 24 + 1
        pattern = np.tile(daily_pattern, n_days)[:hours]
        
        # Add some noise
        noise = np.random.normal(0, 0.05, hours)
        pattern = np.clip(pattern + noise, 0.1, 1.0)
        
        # Scale to realistic kW values (10-50 kW for residential)
        base_load = np.random.uniform(10, 30)
        peak_factor = np.random.uniform(1.5, 3.0)
        electricity = pattern * base_load * peak_factor
        
        # Create feature array [hour_norm, dow_norm, weekend, electricity, heating, solar, net, export]
        features = np.zeros((hours, 8))
        for i in range(hours):
            features[i] = [
                (i % 24) / 24.0,  # Normalized hour
                ((i // 24) % 7) / 7.0,  # Normalized day of week
                1.0 if (i // 24) % 7 >= 5 else 0.0,  # Weekend flag
                electricity[i],  # Electricity demand
                electricity[i] * 0.3,  # Heating (correlated)
                max(0, 20 * np.sin(np.pi * (i % 24 - 6) / 12)) if 6 <= i % 24 <= 18 else 0,  # Solar
                electricity[i],  # Net demand (no solar/battery)
                0.0  # Export potential
            ]
        
        return features
    
    def _get_adjacency(self, district_name: str) -> Dict[str, List]:
        """Get adjacency clusters and relationships."""
        logger.info("Fetching adjacency clusters...")
        clusters = self.kg.get_adjacency_clusters(district_name)
        
        # Build adjacency matrix
        adjacency_matrix = {}
        for cluster in clusters:
            if cluster.get('buildings'):
                for i, b1 in enumerate(cluster['buildings']):
                    for b2 in cluster['buildings'][i+1:]:
                        key = tuple(sorted([str(b1.get('ogc_fid', b1)), 
                                          str(b2.get('ogc_fid', b2))]))
                        adjacency_matrix[key] = {
                            'cluster_id': cluster['cluster_id'],
                            'sharing_potential': cluster.get('sharing_potential', 0)
                        }
        
        logger.info(f"Found {len(clusters)} adjacency clusters")
        return {
            'clusters': clusters,
            'adjacency_matrix': adjacency_matrix
        }
    
    def _create_constraints(self, topology: Dict) -> Dict[str, Any]:
        """Create constraint matrices from topology."""
        logger.info("Creating constraint matrices...")
        
        buildings = topology['nodes']['buildings']
        n_buildings = len(buildings)
        
        # Create building ID to index mapping
        bid_to_idx = {str(b['ogc_fid']): i for i, b in enumerate(buildings)}
        
        # Initialize constraint matrices
        same_cable_group = np.zeros((n_buildings, n_buildings), dtype=bool)
        same_transformer = np.zeros((n_buildings, n_buildings), dtype=bool)
        adjacent_buildings = np.zeros((n_buildings, n_buildings), dtype=bool)
        
        # Fill same cable group constraint
        cable_groups = {}
        for b in buildings:
            cg_id = b.get('lv_group_id') or b.get('cable_group_id')
            if cg_id:
                if cg_id not in cable_groups:
                    cable_groups[cg_id] = []
                cable_groups[cg_id].append(bid_to_idx[str(b['ogc_fid'])])
        
        for cg_buildings in cable_groups.values():
            for i in cg_buildings:
                for j in cg_buildings:
                    if i != j:
                        same_cable_group[i, j] = True
        
        # Get transformer connections from edges
        transformer_groups = self._get_transformer_groups(topology)
        for t_buildings in transformer_groups.values():
            indices = [bid_to_idx[str(bid)] for bid in t_buildings if str(bid) in bid_to_idx]
            for i in indices:
                for j in indices:
                    if i != j:
                        same_transformer[i, j] = True
        
        # Get transformer capacities
        transformer_capacities = {}
        for t in topology['nodes']['transformers']:
            t_id = str(t.get('transformer_id', t.get('id', '')))
            # Standard Dutch transformer sizes
            capacity = t.get('capacity_kva', 630)  # Default 630 kVA
            transformer_capacities[t_id] = capacity
        
        logger.info(f"Created constraints for {n_buildings} buildings")
        
        return {
            'same_cable_group': same_cable_group,
            'same_transformer': same_transformer,
            'adjacent_buildings': adjacent_buildings,
            'transformer_capacity': transformer_capacities,
            'cable_groups': cable_groups,
            'transformer_groups': transformer_groups,
            'bid_to_idx': bid_to_idx,
            'idx_to_bid': {v: k for k, v in bid_to_idx.items()}
        }
    
    def _get_transformer_groups(self, topology: Dict) -> Dict[str, List]:
        """Map transformers to their connected buildings."""
        transformer_groups = {}
        
        # Get cable group to transformer mapping
        cg_to_transformer = {}
        for edge in topology['edges'].get('cable_to_transformer', []):
            cg_id = edge['src']
            t_id = edge['dst']
            cg_to_transformer[cg_id] = t_id
        
        # Map buildings through cable groups to transformers
        for b in topology['nodes']['buildings']:
            cg_id = b.get('lv_group_id') or b.get('cable_group_id')
            if cg_id and cg_id in cg_to_transformer:
                t_id = cg_to_transformer[cg_id]
                if t_id not in transformer_groups:
                    transformer_groups[t_id] = []
                transformer_groups[t_id].append(b['ogc_fid'])
        
        return transformer_groups
    
    def _calculate_complementarity(self, time_series: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Calculate complementarity matrix from time series.
        Complementarity = 1 - correlation (negative correlation is good).
        """
        logger.info("Calculating complementarity matrix...")
        
        building_ids = sorted(time_series.keys())
        n_buildings = len(building_ids)
        complementarity = np.zeros((n_buildings, n_buildings))
        
        for i, bid1 in enumerate(building_ids):
            for j, bid2 in enumerate(building_ids):
                if i < j:  # Calculate only upper triangle
                    # Use net demand (column 6) for complementarity
                    ts1 = time_series[bid1][:, 6] if len(time_series[bid1]) > 0 else np.array([0])
                    ts2 = time_series[bid2][:, 6] if len(time_series[bid2]) > 0 else np.array([0])
                    
                    if len(ts1) > 1 and len(ts2) > 1:
                        # Calculate Pearson correlation
                        try:
                            corr, _ = pearsonr(ts1, ts2)
                            # Complementarity: high when correlation is negative
                            comp = 1 - corr  # Range [0, 2], where 2 is perfect complementarity
                            comp = comp / 2  # Normalize to [0, 1]
                        except:
                            comp = 0.5  # Default to neutral
                    else:
                        comp = 0.5
                    
                    complementarity[i, j] = comp
                    complementarity[j, i] = comp  # Symmetric
        
        # Set diagonal to 0 (no self-complementarity)
        np.fill_diagonal(complementarity, 0)
        
        logger.info(f"Complementarity matrix calculated. Mean: {np.mean(complementarity):.3f}, "
                   f"Max: {np.max(complementarity):.3f}")
        
        return complementarity
    
    def _extract_building_features(self, buildings: List[Dict]) -> pd.DataFrame:
        """Extract and normalize building features."""
        logger.info("Extracting building features...")
        
        features = []
        for b in buildings:
            # Extract relevant features
            feature_dict = {
                'ogc_fid': str(b['ogc_fid']),
                'area': float(b.get('area', 100)),
                'height': float(b.get('height', 3)),
                'energy_label': b.get('energy_label', 'D'),
                'energy_label_num': self._encode_energy_label(b.get('energy_label', 'D')),
                'has_solar': bool(b.get('has_solar', False)),
                'has_battery': bool(b.get('has_battery', False)),
                'has_heat_pump': bool(b.get('has_heat_pump', False)),
                'solar_potential': b.get('solar_potential', 'medium'),
                'solar_capacity_kwp': float(b.get('solar_capacity_kwp', 0)),
                'suitable_roof_area': float(b.get('suitable_roof_area', 0)),
                'building_function': b.get('building_function', 'residential'),
                'is_residential': b.get('building_function', 'residential') == 'residential',
                'age_range': b.get('age_range', 'Unknown'),
                'insulation_quality': b.get('insulation_quality', 'fair'),
                'expected_cop': float(b.get('expected_cop', 2.5)),
                'x': float(b.get('x', 0)),
                'y': float(b.get('y', 0))
            }
            features.append(feature_dict)
        
        df = pd.DataFrame(features)
        logger.info(f"Extracted features for {len(df)} buildings")
        
        return df
    
    def _encode_energy_label(self, label: str) -> int:
        """Encode energy label as numeric value."""
        label_map = {
            'A': 7, 'B': 6, 'C': 5, 'D': 4,
            'E': 3, 'F': 2, 'G': 1, 'Unknown': 0
        }
        return label_map.get(label, 0)
    
    def _calculate_electrical_distances(self, topology: Dict) -> np.ndarray:
        """
        Calculate electrical distances between buildings.
        Distance increases when crossing cable groups or transformers.
        """
        logger.info("Calculating electrical distances...")
        
        buildings = topology['nodes']['buildings']
        n_buildings = len(buildings)
        distances = np.full((n_buildings, n_buildings), np.inf)
        
        # Create building ID to index mapping
        bid_to_idx = {str(b['ogc_fid']): i for i, b in enumerate(buildings)}
        
        # Group buildings by cable group
        cable_groups = {}
        for b in buildings:
            cg_id = b.get('lv_group_id') or b.get('cable_group_id')
            if cg_id:
                if cg_id not in cable_groups:
                    cable_groups[cg_id] = []
                cable_groups[cg_id].append(bid_to_idx[str(b['ogc_fid'])])
        
        # Same cable group: distance = 1
        for cg_buildings in cable_groups.values():
            for i in cg_buildings:
                for j in cg_buildings:
                    if i != j:
                        distances[i, j] = 1
        
        # Get transformer groups
        transformer_groups = self._get_transformer_groups(topology)
        
        # Same transformer but different cable group: distance = 2
        for t_buildings in transformer_groups.values():
            indices = [bid_to_idx[str(bid)] for bid in t_buildings if str(bid) in bid_to_idx]
            for i in indices:
                for j in indices:
                    if i != j and distances[i, j] > 2:
                        distances[i, j] = 2
        
        # Different transformer: distance = 3
        distances[distances == np.inf] = 3
        
        # Set diagonal to 0
        np.fill_diagonal(distances, 0)
        
        logger.info("Electrical distance matrix calculated")
        return distances
    
    def get_prosumer_consumer_lists(self, building_features: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Identify prosumers and consumers for stable matching.
        
        Args:
            building_features: DataFrame with building features
            
        Returns:
            Tuple of (prosumer_ids, consumer_ids)
        """
        prosumers = building_features[
            (building_features['has_solar'] == True) |
            (building_features['has_battery'] == True) |
            (building_features['solar_potential'].isin(['high', 'medium']))
        ]['ogc_fid'].tolist()
        
        consumers = building_features[
            ~building_features['ogc_fid'].isin(prosumers)
        ]['ogc_fid'].tolist()
        
        logger.info(f"Identified {len(prosumers)} prosumers and {len(consumers)} consumers")
        
        return prosumers, consumers
    
    def calculate_diversity_index(self, cluster_buildings: List[str], 
                                 building_features: pd.DataFrame) -> float:
        """
        Calculate diversity index for a cluster (Shannon entropy).
        
        Args:
            cluster_buildings: List of building IDs in cluster
            building_features: DataFrame with building features
            
        Returns:
            Diversity index [0, 1]
        """
        if len(cluster_buildings) == 0:
            return 0
        
        cluster_df = building_features[
            building_features['ogc_fid'].isin([str(b) for b in cluster_buildings])
        ]
        
        # Calculate diversity across multiple dimensions
        diversities = []
        
        # Building function diversity
        if 'building_function' in cluster_df.columns:
            func_counts = cluster_df['building_function'].value_counts()
            func_probs = func_counts / len(cluster_df)
            func_entropy = -np.sum(func_probs * np.log(func_probs + 1e-10))
            func_max_entropy = np.log(len(func_counts))
            if func_max_entropy > 0:
                diversities.append(func_entropy / func_max_entropy)
        
        # Energy label diversity
        if 'energy_label' in cluster_df.columns:
            label_counts = cluster_df['energy_label'].value_counts()
            label_probs = label_counts / len(cluster_df)
            label_entropy = -np.sum(label_probs * np.log(label_probs + 1e-10))
            label_max_entropy = np.log(7)  # Max 7 labels (A-G)
            if label_max_entropy > 0:
                diversities.append(label_entropy / label_max_entropy)
        
        # Asset diversity (solar, battery, heat pump)
        asset_diversity = 0
        for asset in ['has_solar', 'has_battery', 'has_heat_pump']:
            if asset in cluster_df.columns:
                has_asset = cluster_df[asset].sum()
                p_asset = has_asset / len(cluster_df)
                if 0 < p_asset < 1:
                    asset_diversity += -p_asset * np.log(p_asset) - (1-p_asset) * np.log(1-p_asset)
        if asset_diversity > 0:
            diversities.append(asset_diversity / (3 * np.log(2)))  # Normalize by max entropy
        
        return np.mean(diversities) if diversities else 0