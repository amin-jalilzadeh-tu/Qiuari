# feature_processor.py - CORRECTED VERSION
"""
Feature processing and engineering for energy grid data.
Updated to match actual KG structure and feature ordering.
"""

import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from typing import Dict, List, Any, Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FeatureProcessor:
    """Processes and engineers features for GNN - Updated for actual schema."""
    
    def __init__(self):
        """Initialize feature processor with scalers and encoders."""
        
        # Scalers for different feature types
        self.scalers = {
            'building': StandardScaler(),
            'building_energy': MinMaxScaler(),  # For energy features
            'cable_group': StandardScaler(),
            'transformer': StandardScaler(),
            'substation': StandardScaler(),
            'adjacency_cluster': StandardScaler()
        }
        
        # Feature statistics (for new data)
        self.feature_stats = {}
        
        # Define feature indices for building features from graph_constructor
        # Total 17 features from graph_constructor.py
        self.building_feature_indices = {
            'area': 0,
            'energy_score': 1,
            'solar_score': 2,
            'electrify_score': 3,
            'age': 4,
            'roof_area': 5,
            'height': 6,
            'has_solar': 7,
            'has_battery': 8,
            'has_heat_pump': 9,
            'shared_walls': 10,
            'x_coord': 11,
            'y_coord': 12,
            'avg_electricity': 13,
            'avg_heating': 14,
            'peak_electricity': 15,
            'energy_intensity': 16
        }
    
    def process_graph_features(self, graph, fit: bool = True) -> None:
        """
        Process all features in a HeteroData graph in-place.
        
        Args:
            graph: PyTorch Geometric HeteroData
            fit: Whether to fit scalers (True for training)
        """
        logger.info("Processing graph features")
        
        # Process each node type that exists in graph
        for node_type in ['building', 'cable_group', 'transformer', 'substation', 'adjacency_cluster']:
            if node_type in graph.node_types:
                if hasattr(graph[node_type], 'x') and graph[node_type].x is not None:
                    features = graph[node_type].x
                    processed = self.process_node_features(
                        features, node_type, fit=fit
                    )
                    graph[node_type].x = processed
                    
                    # Add engineered features
                    engineered = self.engineer_node_features(
                        processed, node_type, graph
                    )
                    if engineered is not None:
                        graph[node_type].x_engineered = engineered
                        logger.info(f"Added engineered features for {node_type}: shape {engineered.shape}")
    
    def process_node_features(self, features: torch.Tensor, 
                             node_type: str,
                             fit: bool = True) -> torch.Tensor:
        """
        Process features for a specific node type.
        
        Args:
            features: Raw feature tensor
            node_type: Type of node
            fit: Whether to fit scalers
            
        Returns:
            Processed feature tensor
        """
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        
        if node_type == 'building':
            processed = self._process_building_features(features_np, fit)
        elif node_type == 'cable_group':
            processed = self._process_cable_group_features(features_np, fit)
        elif node_type == 'transformer':
            processed = self._process_transformer_features(features_np, fit)
        elif node_type == 'substation':
            processed = self._process_substation_features(features_np, fit)
        elif node_type == 'adjacency_cluster':
            processed = self._process_cluster_features(features_np, fit)
        else:
            processed = features_np
        
        return torch.tensor(processed, dtype=torch.float)
    
    def _process_building_features(self, features: np.ndarray, 
                                  fit: bool = True) -> np.ndarray:
        """
        Process building features - UPDATED FOR 17 FEATURES.
        
        Features from graph_constructor:
        0: area
        1: energy_score (1-7)
        2: solar_score (0-3)
        3: electrify_score (0-3)
        4: age
        5: roof_area
        6: height
        7: has_solar (0/1)
        8: has_battery (0/1)
        9: has_heat_pump (0/1)
        10: shared_walls
        11: x_coord
        12: y_coord
        13: avg_electricity_demand
        14: avg_heating_demand
        15: peak_electricity_demand
        16: energy_intensity
        """
        processed = features.copy()
        
        # Normalize continuous features (not binary or already normalized)
        continuous_indices = [0, 4, 5, 6, 10, 11, 12]  # area, age, roof, height, walls, x, y
        if features[:, continuous_indices].shape[1] > 0:
            if fit:
                self.scalers['building'].fit(features[:, continuous_indices])
            processed[:, continuous_indices] = self.scalers['building'].transform(
                features[:, continuous_indices]
            )
        
        # Normalize energy features separately (keep positive)
        energy_indices = [13, 14, 15, 16]  # avg_elec, avg_heat, peak_elec, energy_intensity
        if features[:, energy_indices].shape[1] > 0:
            if fit:
                self.scalers['building_energy'].fit(features[:, energy_indices])
            processed[:, energy_indices] = self.scalers['building_energy'].transform(
                features[:, energy_indices]
            )
        
        # Normalize scores to [0, 1]
        # IN feature_processor.py (around line 145-147):
        # REMOVE these lines since values are already in correct scale from graph_constructor:
        # T remove these lines, we commnted them 
        # processed[:, 1] = processed[:, 1] / 7.0  # DELETE THIS
        # processed[:, 2] = processed[:, 2] / 3.0  # DELETE THIS
        # processed[:, 3] = processed[:, 3] / 3.0  # DELETE THIS

        # REPLACE WITH just validation:
        processed[:, 1] = np.clip(processed[:, 1], 0, 7)  # energy_score already 0-7
        processed[:, 2] = np.clip(processed[:, 2], 0, 3)  # solar_score already 0-3
        processed[:, 3] = np.clip(processed[:, 3], 0, 3)  # electrify_score already 0-3
        # Binary features (7,8,9) are already 0/1, no processing needed
        
        return processed
    
    def _process_cable_group_features(self, features: np.ndarray, 
                                     fit: bool = True) -> np.ndarray:
        """
        Process cable group features.
        
        Features from graph_constructor:
        0: total_length_m
        1: segment_count
        2: building_count
        3: avg_energy_score
        4: total_area
        5: total_roof_area
        6: solar_count
        7: battery_count
        8: hp_count
        9: avg_electricity_demand
        10: peak_electricity_demand
        11: demand_diversity_factor
        """
        processed = features.copy()
        
        if fit:
            self.scalers['cable_group'].fit(processed)
        processed = self.scalers['cable_group'].transform(processed)
        
        return processed
    
    def _process_transformer_features(self, features: np.ndarray, 
                                     fit: bool = True) -> np.ndarray:
        """Process transformer features."""
        processed = features.copy()
        
        if fit:
            self.scalers['transformer'].fit(processed)
        processed = self.scalers['transformer'].transform(processed)
        
        return processed
    
    def _process_substation_features(self, features: np.ndarray,
                                    fit: bool = True) -> np.ndarray:
        """Process substation features."""
        processed = features.copy()
        
        if fit:
            self.scalers['substation'].fit(processed)
        processed = self.scalers['substation'].transform(processed)
        
        return processed
    
    def _process_cluster_features(self, features: np.ndarray, 
                                 fit: bool = True) -> np.ndarray:
        """
        Process adjacency cluster features.
        
        Features from graph_constructor:
        0: member_count
        1: energy_sharing_potential
        2: solar_penetration
        3: hp_penetration
        4: battery_penetration
        5: thermal_benefit
        6: cable_savings
        7: total_demand_kw
        8: export_potential_kw
        9: self_sufficiency_ratio
        10: sharing_benefit_kwh
        """
        processed = features.copy()
        
        if fit:
            self.scalers['adjacency_cluster'].fit(processed)
        processed = self.scalers['adjacency_cluster'].transform(processed)
        
        return processed
    
    def engineer_node_features(self, features: torch.Tensor,
                              node_type: str,
                              graph) -> Optional[torch.Tensor]:
        """
        Create engineered features for nodes.
        
        Args:
            features: Processed features
            node_type: Type of node
            graph: Full graph for context
            
        Returns:
            Engineered features or None
        """
        if node_type == 'building':
            return self._engineer_building_features(features, graph)
        elif node_type == 'cable_group':
            return self._engineer_cable_group_features(features, graph)
        elif node_type == 'adjacency_cluster':
            return self._engineer_cluster_features(features, graph)
        else:
            return None
    
    def _engineer_building_features(self, features: torch.Tensor,
                                   graph) -> torch.Tensor:
        """
        Engineer building-level features - UPDATED FOR CORRECT INDICES.
        
        New features:
        - Energy intensity ratio
        - Retrofit potential score
        - Solar suitability
        - Electrification readiness
        - System integration score
        """
        features_np = features.numpy()
        engineered = []
        
        # Use correct indices from self.building_feature_indices
        idx = self.building_feature_indices
        
        # Energy intensity ratio (normalized demand vs area)
        area_norm = features_np[:, idx['area']]
        elec_norm = features_np[:, idx['avg_electricity']]
        heat_norm = features_np[:, idx['avg_heating']]
        
        energy_intensity_ratio = (elec_norm + heat_norm) / (area_norm + 1e-6)
        engineered.append(energy_intensity_ratio.reshape(-1, 1))
        
        # Retrofit potential score
        # Poor energy score + old age = high retrofit potential
        energy_score_norm = features_np[:, idx['energy_score']]  # Lower is worse after normalization
        age_norm = features_np[:, idx['age']]
        
        retrofit_score = (1 - energy_score_norm) * 0.6 + age_norm * 0.4
        engineered.append(retrofit_score.reshape(-1, 1))
        
        # Solar suitability
        # Based on roof area, current solar status, and solar score
        roof_norm = features_np[:, idx['roof_area']]
        solar_score_norm = features_np[:, idx['solar_score']]
        has_solar = features_np[:, idx['has_solar']]
        
        solar_suit = roof_norm * solar_score_norm * (1 - has_solar * 0.5)  # Reduce if already has solar
        engineered.append(solar_suit.reshape(-1, 1))
        
        # Electrification readiness
        # High heating demand + good efficiency + electrification score
        electrif_score_norm = features_np[:, idx['electrify_score']]
        has_hp = features_np[:, idx['has_heat_pump']]
        
        electrif_ready = electrif_score_norm * heat_norm * (1 - has_hp * 0.5)
        engineered.append(electrif_ready.reshape(-1, 1))
        
        # System integration score (how many systems installed)
        system_score = (features_np[:, idx['has_solar']] + 
                       features_np[:, idx['has_battery']] + 
                       features_np[:, idx['has_heat_pump']]) / 3.0
        engineered.append(system_score.reshape(-1, 1))
        
        # Peak to average ratio (demand flexibility indicator)
        peak_norm = features_np[:, idx['peak_electricity']]
        avg_demand = (elec_norm + heat_norm) / 2
        peak_ratio = peak_norm / (avg_demand + 1e-6)
        engineered.append(peak_ratio.reshape(-1, 1))
        
        # Shared wall factor (for thermal benefits in clusters)
        shared_walls_norm = features_np[:, idx['shared_walls']]
        engineered.append(shared_walls_norm.reshape(-1, 1))
        
        return torch.tensor(np.hstack(engineered), dtype=torch.float)
    
    def _engineer_cable_group_features(self, features: torch.Tensor,
                                      graph) -> torch.Tensor:
        """
        Engineer cable group features.
        
        New features:
        - Diversity factor efficiency
        - Self-sufficiency potential
        - Grid stress indicator
        - DER penetration rate
        """
        features_np = features.numpy()
        engineered = []
        
        # Indices for cable group features
        # [length, segments, buildings, avg_energy, total_area, roof_area, 
        #  solar_count, battery_count, hp_count, avg_elec, peak_elec, diversity]
        
        # Diversity factor efficiency
        diversity = features_np[:, 11]  # demand_diversity_factor
        building_count = features_np[:, 2]
        diversity_eff = diversity * np.log1p(building_count) / 10
        engineered.append(diversity_eff.reshape(-1, 1))
        
        # Self-sufficiency potential (roof area vs demand)
        roof_area = features_np[:, 5]
        avg_demand = features_np[:, 9]
        self_suff = roof_area / (avg_demand * 100 + 1e-6)  # Rough solar generation potential
        engineered.append(self_suff.reshape(-1, 1))
        
        # Grid stress (peak demand vs assumed capacity)
        peak_demand = features_np[:, 10]
        length = features_np[:, 0]
        stress = peak_demand / (length + 1e-6)  # Demand per meter as proxy
        engineered.append(stress.reshape(-1, 1))
        
        # DER penetration rate
        solar_count = features_np[:, 6]
        battery_count = features_np[:, 7]
        hp_count = features_np[:, 8]
        der_penetration = (solar_count + battery_count + hp_count) / (building_count + 1e-6)
        engineered.append(der_penetration.reshape(-1, 1))
        
        return torch.tensor(np.hstack(engineered), dtype=torch.float)
    
    def _engineer_cluster_features(self, features: torch.Tensor,
                                  graph) -> torch.Tensor:
        """
        Engineer adjacency cluster features.
        
        New features:
        - Self-sufficiency ratio
        - System penetration index
        - Sharing efficiency
        - Economic benefit score
        """
        features_np = features.numpy()
        engineered = []
        
        # Indices for cluster features
        # [members, sharing_pot, solar_pen, hp_pen, battery_pen, thermal, cable_savings,
        #  total_demand, export_pot, self_suff_ratio, sharing_benefit]
        
        # Enhanced self-sufficiency ratio
        self_suff_ratio = features_np[:, 9]  # Already calculated
        solar_pen = features_np[:, 2]
        battery_pen = features_np[:, 4]
        enhanced_self_suff = self_suff_ratio * (1 + solar_pen + battery_pen) / 3
        engineered.append(enhanced_self_suff.reshape(-1, 1))
        
        # System penetration index
        system_pen = (features_np[:, 2] + features_np[:, 3] + features_np[:, 4]) / 3
        engineered.append(system_pen.reshape(-1, 1))
        
        # Sharing efficiency (export potential vs total demand)
        export_pot = features_np[:, 8]
        total_demand = features_np[:, 7]
        sharing_eff = export_pot / (total_demand + 1e-6)
        engineered.append(sharing_eff.reshape(-1, 1))
        
        # Economic benefit score (thermal + cable savings + sharing benefit)
        thermal_benefit = features_np[:, 5]
        cable_savings = features_np[:, 6]
        sharing_benefit = features_np[:, 10]
        economic_score = (thermal_benefit + cable_savings + sharing_benefit / 1000) / 3
        engineered.append(economic_score.reshape(-1, 1))
        
        # Member density factor
        member_count = features_np[:, 0]
        density_factor = np.log1p(np.maximum(member_count, 0)) / 10  # Ensure non-negative

        engineered.append(density_factor.reshape(-1, 1))
        
        return torch.tensor(np.hstack(engineered), dtype=torch.float)
    
    def create_temporal_features(self, 
                                demands: pd.DataFrame,
                                window_size: int = 24) -> torch.Tensor:
        """
        Create temporal features from time-series demands.
        
        Args:
            demands: DataFrame with hourly demands
            window_size: Hours to look back
            
        Returns:
            Temporal feature tensor
        """
        temporal_features = []
        
        # Rolling statistics
        for col in ['electricity', 'heating']:
            if col in demands.columns:
                rolling = demands[col].rolling(window=window_size)
                
                temporal_features.append(rolling.mean().fillna(0).values)
                temporal_features.append(rolling.std().fillna(0).values)
                temporal_features.append(rolling.max().fillna(0).values)
                temporal_features.append(rolling.min().fillna(0).values)
        
        # Time-based features
        if 'timestamp' in demands.columns:
            timestamps = pd.to_datetime(demands['timestamp'])
            
            # Hour of day (sin/cos encoding)
            hour = timestamps.dt.hour
            temporal_features.append(np.sin(2 * np.pi * hour / 24))
            temporal_features.append(np.cos(2 * np.pi * hour / 24))
            
            # Day of week (sin/cos encoding)
            dow = timestamps.dt.dayofweek
            temporal_features.append(np.sin(2 * np.pi * dow / 7))
            temporal_features.append(np.cos(2 * np.pi * dow / 7))
            
            # Month (sin/cos encoding)
            month = timestamps.dt.month
            temporal_features.append(np.sin(2 * np.pi * month / 12))
            temporal_features.append(np.cos(2 * np.pi * month / 12))
        
        return torch.tensor(np.column_stack(temporal_features), dtype=torch.float)
    
    def aggregate_features(self, 
                          node_features: torch.Tensor,
                          edge_index: torch.Tensor,
                          aggregation: str = 'mean') -> torch.Tensor:
        """
        Aggregate features along edges.
        
        Args:
            node_features: Features to aggregate
            edge_index: Edge connections
            aggregation: Type of aggregation ('mean', 'sum', 'max')
            
        Returns:
            Aggregated features
        """
        from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
        
        if aggregation == 'mean':
            return global_mean_pool(node_features, edge_index[1])
        elif aggregation == 'sum':
            return global_add_pool(node_features, edge_index[1])
        elif aggregation == 'max':
            return global_max_pool(node_features, edge_index[1])
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    
    def process_kg_data(self, kg_data: Dict) -> Dict:
        """
        Process raw KG data into features for GNN.
        
        Args:
            kg_data: Raw data from KGConnector
            
        Returns:
            Processed data with features
        """
        processed = {}
        
        # Process building features
        if 'buildings' in kg_data and kg_data['buildings']:
            building_features = []
            for building in kg_data['buildings']:
                features = [
                    building.get('area', 100.0),
                    building.get('height', 10.0),
                    building.get('age', 20),
                    self._encode_energy_label(building.get('energy_label', 'D')),
                    building.get('roof_area', 50.0),
                    building.get('orientation', 0),
                    building.get('occupancy', 4)
                ]
                building_features.append(features)
            processed['building_features'] = np.array(building_features)
        
        # Process edges
        processed['edges'] = kg_data.get('edges', [])
        
        # Process transformers
        processed['transformers'] = kg_data.get('transformers', [])
        
        # Energy profiles (synthetic if not available)
        if not kg_data.get('energy_profiles'):
            num_buildings = len(kg_data.get('buildings', []))
            processed['energy_profiles'] = self._generate_synthetic_profiles(num_buildings)
        else:
            processed['energy_profiles'] = kg_data['energy_profiles']
        
        return processed
    
    def _encode_energy_label(self, label: str) -> float:
        """Encode energy label A-G as numeric."""
        mapping = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        return mapping.get(label, 4)  # Default to D
    
    def _generate_synthetic_profiles(self, num_buildings: int):
        """Generate synthetic energy profiles for testing."""
        profiles = {}
        for i in range(num_buildings):
            # Simple sinusoidal pattern
            hours = np.arange(24)
            base_load = np.random.uniform(1, 5)
            peak_load = np.random.uniform(5, 15)
            profile = base_load + peak_load * np.sin(2 * np.pi * (hours - 6) / 24)
            profiles[f'building_{i}'] = profile.clip(min=0)
        return profiles
    
    def create_task_specific_features(self, 
                                     graph,
                                     task: str) -> Dict[str, torch.Tensor]:
        """
        Create features specific to a task.
        
        Args:
            graph: HeteroData graph
            task: Task type
            
        Returns:
            Dictionary of task-specific features
        """
        task_features = {}
        
        if task == 'retrofit':
            # Create retrofit-specific features
            if 'building' in graph.node_types and hasattr(graph['building'], 'x_engineered'):
                building_features = graph['building'].x_engineered
                # Retrofit score is second engineered feature (index 1)
                task_features['retrofit_priority'] = building_features[:, 1]
                
                # Add cost estimation
                area = graph['building'].x[:, self.building_feature_indices['area']]
                cost_per_m2 = 500  # EUR/m2 for retrofit
                task_features['retrofit_cost'] = area * cost_per_m2
        
        elif task == 'energy_sharing':
            # Create sharing-specific features
            if 'adjacency_cluster' in graph.node_types and hasattr(graph['adjacency_cluster'], 'x_engineered'):
                cluster_features = graph['adjacency_cluster'].x_engineered
                # Enhanced self-sufficiency is first engineered feature
                task_features['self_sufficiency'] = cluster_features[:, 0]
                # System penetration is second
                task_features['system_penetration'] = cluster_features[:, 1]
                # Sharing efficiency is third
                task_features['sharing_efficiency'] = cluster_features[:, 2]
        
        elif task == 'solar':
            # Create solar-specific features
            if 'building' in graph.node_types and hasattr(graph['building'], 'x_engineered'):
                building_features = graph['building'].x_engineered
                # Solar suitability is third engineered feature (index 2)
                task_features['solar_suitability'] = building_features[:, 2]
                
                # Estimate generation potential
                roof_area = graph['building'].x[:, self.building_feature_indices['roof_area']]
                task_features['solar_generation'] = roof_area * 0.15 * 1000  # kWh/year
        
        elif task == 'electrification':
            # Create electrification features
            if 'building' in graph.node_types:
                building_features = graph['building'].x_engineered if hasattr(graph['building'], 'x_engineered') else None
                
                if building_features is not None:
                    # Electrification readiness is fourth engineered feature (index 3)
                    task_features['electrification_ready'] = building_features[:, 3]
                
                # Has heat pump from original features
                task_features['has_heat_pump'] = graph['building'].x[:, self.building_feature_indices['has_heat_pump']]
        
        return task_features
    
    def process_temporal_features(self, temporal_tensor: torch.Tensor,
                                 normalize: bool = True) -> torch.Tensor:
        """
        Process temporal features for GNN.
        
        Args:
            temporal_tensor: [n_nodes, time_steps, features]
            normalize: Whether to normalize
        
        Returns:
            Processed temporal tensor
        """
        if temporal_tensor is None:
            return None
        
        n_nodes, time_steps, n_features = temporal_tensor.shape
        
        if normalize:
            # Normalize per feature across all nodes and time
            reshaped = temporal_tensor.reshape(-1, n_features)
            
            # Skip first 3 features (hour, day_of_week, is_weekend) - already normalized
            for i in range(3, n_features):
                feature_vals = reshaped[:, i]
                mean = feature_vals.mean()
                std = feature_vals.std() + 1e-6
                reshaped[:, i] = (feature_vals - mean) / std
            
            temporal_tensor = reshaped.reshape(n_nodes, time_steps, n_features)
        
        return temporal_tensor
    
    def create_temporal_embeddings(self, temporal_tensor: torch.Tensor,
                                  embedding_dim: int = 32) -> torch.Tensor:
        """
        Create learned embeddings from temporal features.
        
        Args:
            temporal_tensor: [n_nodes, time_steps, features]
            embedding_dim: Size of embedding
        
        Returns:
            Temporal embeddings [n_nodes, embedding_dim]
        """
        import torch.nn as nn
        
        if temporal_tensor is None:
            return None
        
        n_nodes, time_steps, n_features = temporal_tensor.shape
        
        # Simple LSTM encoder
        encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        with torch.no_grad():
            encoder.eval()
            _, (h_n, _) = encoder(temporal_tensor)
            embeddings = h_n.squeeze(0)  # [n_nodes, embedding_dim]
        
        return embeddings
    

    def extract_temporal_patterns(self, temporal_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract specific temporal patterns for energy sharing.
        
        Returns dict with:
        - peak_hours: When does each building peak?
        - valley_hours: When is demand lowest?
        - export_hours: When can building export?
        - import_hours: When does building need import?
        """
        if temporal_tensor is None:
            return {}
        
        # Check the shape to determine if it's building or cluster data
        n_nodes, time_steps, n_features = temporal_tensor.shape
        patterns = {}
        
        # Building temporal features have 8 features: [hour, day, weekend, elec, heat, solar, net, export]
        # Cluster temporal features have 7 features: [hour, demand, solar, export, deficit, surplus, sharing]
        
        if n_features == 8:
            # Building features
            # Find peak demand hours (highest electricity demand)
            elec_demand = temporal_tensor[:, :, 3]  # [n_nodes, time_steps]
            patterns['peak_hours'] = elec_demand.argmax(dim=1)
            patterns['valley_hours'] = elec_demand.argmin(dim=1)
            
            # Find export potential hours (export_potential > 0)
            export_potential = temporal_tensor[:, :, 7]
            patterns['can_export'] = (export_potential > 0).float().sum(dim=1)
            
            # Find import need hours (net_demand > 0)
            net_demand = temporal_tensor[:, :, 6]
            patterns['needs_import'] = (net_demand > 0).float().sum(dim=1)
            
            # Mismatch score (how often building's peak doesn't align with solar)
            solar_gen = temporal_tensor[:, :, 5]
            solar_peak = solar_gen.argmax(dim=1)
            patterns['mismatch_score'] = (patterns['peak_hours'] - solar_peak).abs().float()
            
        elif n_features == 7:
            # Cluster features
            # Find peak demand hours (highest total demand)
            total_demand = temporal_tensor[:, :, 1]  # [n_nodes, time_steps]
            patterns['peak_hours'] = total_demand.argmax(dim=1)
            patterns['valley_hours'] = total_demand.argmin(dim=1)
            
            # Find export potential hours (export > 0)
            export_potential = temporal_tensor[:, :, 3]
            patterns['can_export'] = (export_potential > 0).float().sum(dim=1)
            
            # Find deficit hours (deficit > 0)
            deficit = temporal_tensor[:, :, 4]
            patterns['needs_import'] = (deficit > 0).float().sum(dim=1)
            
            # Sharing potential score
            sharing_potential = temporal_tensor[:, :, 6]
            patterns['sharing_score'] = sharing_potential.mean(dim=1)
        
        return patterns
    
    def save_processors(self, path: str):
        """Save fitted scalers and encoders."""
        import pickle
        
        processors = {
            'scalers': self.scalers,
            'feature_stats': self.feature_stats,
            'building_feature_indices': self.building_feature_indices
        }
        
        with open(path, 'wb') as f:
            pickle.dump(processors, f)
        
        logger.info(f"Saved processors to {path}")
    
    def load_processors(self, path: str):
        """Load fitted scalers and encoders."""
        import pickle
        
        with open(path, 'rb') as f:
            processors = pickle.load(f)
        
        self.scalers = processors['scalers']
        self.feature_stats = processors['feature_stats']
        self.building_feature_indices = processors.get('building_feature_indices', self.building_feature_indices)
        
        logger.info(f"Loaded processors from {path}")