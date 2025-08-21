# data/graph_builder.py
"""
Build PyTorch Geometric datasets from extracted KG data
Handles heterogeneous graphs with multiple node and edge types
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData, Data
from torch_geometric.utils import from_networkx, to_undirected
from typing import Dict, List, Tuple, Optional
import networkx as nx
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)

class GraphBuilder:
    """Build PyTorch Geometric graphs from KG data"""
    
    def __init__(self, extracted_data: Dict, config: Dict):
        """
        Initialize graph builder
        
        Args:
            extracted_data: Output from KGExtractor
            config: Configuration dictionary
        """
        self.data = extracted_data
        self.config = config
        
        # Feature scalers
        self.scalers = {}
        self.encoders = {}
        
        # Graph data
        self.hetero_graph = None
        self.homo_graph = None
        
        # Mappings
        self.node_mappings = {}
        self.edge_mappings = {}
        
        logger.info("Graph builder initialized")
    
    def build_heterogeneous_graph(self) -> HeteroData:
        """Build heterogeneous graph for multi-type nodes and edges"""
        logger.info("Building heterogeneous graph...")
        
        data = HeteroData()
        
        # Add building nodes
        building_features, building_mapping = self._process_building_nodes()
        data['building'].x = building_features
        data['building'].node_id = torch.tensor(list(building_mapping.values()))
        self.node_mappings['building'] = building_mapping
        
        # Add transformer nodes
        if 'lv_networks' in self.data['nodes']:
            lv_features, lv_mapping = self._process_transformer_nodes()
            data['lv_network'].x = lv_features
            data['lv_network'].node_id = torch.tensor(list(lv_mapping.values()))
            self.node_mappings['lv_network'] = lv_mapping
        
        # Add edges
        self._add_electrical_edges(data)
        self._add_spatial_edges(data)
        self._add_temporal_edges(data)
        
        # Add global features
        data.num_nodes = len(building_mapping)
        if 'lv_networks' in self.data['nodes']:
            data.num_nodes += len(lv_mapping)
        
        self.hetero_graph = data
        logger.info(f"Built heterogeneous graph with {data.num_nodes} nodes")
        
        return data
    
    def build_homogeneous_graph(self) -> Data:
        """Build homogeneous graph (buildings only) for simpler models"""
        logger.info("Building homogeneous graph...")
        
        # Get building features
        building_features, building_mapping = self._process_building_nodes()
        
        # Store the mapping
        self.node_mappings['building'] = building_mapping
        
        # Create edge index from all relationships
        edge_index = self._build_combined_edge_index()
        
        # Create edge attributes
        edge_attr = self._build_edge_attributes(edge_index)
        
        # Create Data object
        data = Data(
            x=building_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(building_mapping)
        )
        
        # Add additional attributes
        if 'temporal' in self.data and 'energy_profiles' in self.data['temporal']:
            data.temporal = self._process_temporal_features()
        
        # Add labels if available (for supervised tasks)
        data.y = self._create_pseudo_labels()
        
        self.homo_graph = data
        logger.info(f"Built homogeneous graph with {data.num_nodes} nodes, "
                   f"{data.edge_index.shape[1]} edges")
        
        return data
    
    def _process_building_nodes(self) -> Tuple[torch.Tensor, Dict]:
        """Process building nodes and create feature matrix"""
        buildings = self.data['nodes']['buildings']
        
        # Create node ID mapping
        node_mapping = {row['id']: idx for idx, row in buildings.iterrows()}
        
        # Numerical features (to be scaled)
        numerical_features = [
            'area', 'height', 'roof_area', 'suitable_roof',
            'peak_demand', 'avg_demand', 'load_factor', 'variability',
            'x', 'y'
        ]
        
        # Categorical features (to be encoded)
        categorical_features = [
            'function', 'res_type', 'non_res_type', 'age',
            'orientation', 'energy_label', 'insulation',
            'solar_pot', 'battery_ready', 'heat_pump_ready',
            'heating', 'adjacency_type'
        ]
        
        # Binary features
        binary_features = [
            'has_solar', 'has_battery', 'has_heat_pump'
        ]
        
        # Process numerical features
        num_data = buildings[numerical_features].fillna(0).values
        scaler = StandardScaler()
        num_scaled = scaler.fit_transform(num_data)
        self.scalers['building_numerical'] = scaler
        
        # Process categorical features
        cat_encoded = []
        for feat in categorical_features:
            if feat in buildings.columns:
                encoder = LabelEncoder()
                # Handle NaN as a category
                values = buildings[feat].fillna('unknown').astype(str)
                encoded = encoder.fit_transform(values)
                cat_encoded.append(encoded.reshape(-1, 1))
                self.encoders[f'building_{feat}'] = encoder
        
        cat_data = np.hstack(cat_encoded) if cat_encoded else np.array([])
        
        # Process binary features (only those that exist)
        existing_binary = [f for f in binary_features if f in buildings.columns]
        if existing_binary:
            bin_data = buildings[existing_binary].fillna(0).astype(float).values
        else:
            bin_data = np.zeros((len(buildings), 0))  # Empty array if no binary features
        
        # Combine all features
        if cat_data.size > 0:
            features = np.hstack([num_scaled, cat_data, bin_data])
        else:
            features = np.hstack([num_scaled, bin_data])
        
        # Convert to tensor
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        
        logger.info(f"Processed {len(buildings)} buildings with "
                   f"{feature_tensor.shape[1]} features")
        
        return feature_tensor, node_mapping
    
    def _process_transformer_nodes(self) -> Tuple[torch.Tensor, Dict]:
        """Process LV network/transformer nodes"""
        lv_networks = self.data['nodes']['lv_networks']
        
        # Create node ID mapping
        node_mapping = {row['id']: idx for idx, row in lv_networks.iterrows()}
        
        # Features to use
        features = [
            'capacity', 'building_count', 'peak_kw', 'avg_kw',
            'load_factor', 'solar_pen', 'battery_pen'
        ]
        
        # Extract and scale features
        feat_data = lv_networks[features].fillna(0).values
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat_data)
        self.scalers['lv_network'] = scaler
        
        # Convert to tensor
        feature_tensor = torch.tensor(feat_scaled, dtype=torch.float32)
        
        logger.info(f"Processed {len(lv_networks)} LV networks with "
                   f"{feature_tensor.shape[1]} features")
        
        return feature_tensor, node_mapping
    
    def _add_electrical_edges(self, data: HeteroData):
        """Add electrical connection edges"""
        if 'electrical' not in self.data['edges']:
            return
        
        edges = self.data['edges']['electrical']
        
        # Building to LV edges
        b2lv_edges = edges[edges['type'] == 'building_to_lv']
        if not b2lv_edges.empty:
            source_idx = [self.node_mappings['building'][sid] 
                         for sid in b2lv_edges['source']]
            target_idx = [self.node_mappings['lv_network'][tid] 
                         for tid in b2lv_edges['target']]
            
            edge_index = torch.tensor([source_idx, target_idx], dtype=torch.long)
            data['building', 'connects_to', 'lv_network'].edge_index = edge_index
            
            logger.info(f"Added {len(source_idx)} building->LV edges")
    
    def _add_spatial_edges(self, data: HeteroData):
        """Add spatial proximity edges"""
        # Adjacent buildings
        if 'spatial' in self.data['edges'] and not self.data['edges']['spatial'].empty:
            edges = self.data['edges']['spatial']
            
            source_idx = []
            target_idx = []
            edge_weights = []
            
            for _, edge in edges.iterrows():
                if edge['source'] in self.node_mappings['building'] and \
                   edge['target'] in self.node_mappings['building']:
                    source_idx.append(self.node_mappings['building'][edge['source']])
                    target_idx.append(self.node_mappings['building'][edge['target']])
                    
                    # Edge weight based on shared wall length
                    weight = edge.get('shared_wall', 1.0)
                    edge_weights.append(weight)
            
            if source_idx:
                edge_index = torch.tensor([source_idx, target_idx], dtype=torch.long)
                edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
                
                data['building', 'adjacent_to', 'building'].edge_index = edge_index
                data['building', 'adjacent_to', 'building'].edge_attr = edge_attr
                
                logger.info(f"Added {len(source_idx)} adjacency edges")
        
        # Same transformer edges
        if 'same_transformer' in self.data['edges']:
            edges = self.data['edges']['same_transformer']
            
            source_idx = []
            target_idx = []
            
            for _, edge in edges.iterrows():
                if edge['source'] in self.node_mappings['building'] and \
                   edge['target'] in self.node_mappings['building']:
                    source_idx.append(self.node_mappings['building'][edge['source']])
                    target_idx.append(self.node_mappings['building'][edge['target']])
            
            if source_idx:
                edge_index = torch.tensor([source_idx, target_idx], dtype=torch.long)
                data['building', 'shares_transformer', 'building'].edge_index = edge_index
                
                logger.info(f"Added {len(source_idx)} same-transformer edges")
    
    def _add_temporal_edges(self, data: HeteroData):
        """Add temporal correlation edges"""
        if 'energy_profiles' not in self.data['temporal']:
            return
        
        profiles = self.data['temporal']['energy_profiles']
        
        # Compute correlation matrix
        corr_matrix = profiles.corr()
        
        # Threshold for creating edges
        threshold = self.config['graph']['temporal']['correlation_threshold']
        
        source_idx = []
        target_idx = []
        edge_weights = []
        
        for i, building_i in enumerate(corr_matrix.columns):
            for j, building_j in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_value = corr_matrix.iloc[i, j]
                    
                    # Create edge if correlation is significant
                    if abs(corr_value) > threshold:
                        if building_i in self.node_mappings['building'] and \
                           building_j in self.node_mappings['building']:
                            source_idx.append(self.node_mappings['building'][building_i])
                            target_idx.append(self.node_mappings['building'][building_j])
                            edge_weights.append(corr_value)
        
        if source_idx:
            edge_index = torch.tensor([source_idx, target_idx], dtype=torch.long)
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
            
            # Make undirected
            edge_index = to_undirected(edge_index)
            edge_attr = edge_attr.repeat(2, 1)
            
            data['building', 'correlates_with', 'building'].edge_index = edge_index
            data['building', 'correlates_with', 'building'].edge_attr = edge_attr
            
            logger.info(f"Added {edge_index.shape[1]} temporal correlation edges")
    
    def _build_combined_edge_index(self) -> torch.Tensor:
        """Build combined edge index for homogeneous graph"""
        all_edges = []
        
        # Collect all building-to-building edges
        for edge_type, edges_df in self.data['edges'].items():
            if edges_df.empty:
                continue
                
            if edge_type in ['spatial', 'same_transformer', 'complementarity']:
                for _, edge in edges_df.iterrows():
                    if edge['source'] in self.node_mappings['building'] and \
                       edge['target'] in self.node_mappings['building']:
                        source_idx = self.node_mappings['building'][edge['source']]
                        target_idx = self.node_mappings['building'][edge['target']]
                        all_edges.append([source_idx, target_idx])
                        all_edges.append([target_idx, source_idx])  # Make undirected
        
        if all_edges:
            edge_index = torch.tensor(all_edges, dtype=torch.long).t()
        else:
            # Create a minimal connected graph if no edges
            logger.warning("No edges found, creating minimal connectivity")
            n = len(self.node_mappings['building'])
            edges = [[i, (i+1) % n] for i in range(n)]
            edges += [[(i+1) % n, i] for i in range(n)]  # Make undirected
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        return edge_index
    
    def _build_edge_attributes(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Build edge attribute matrix"""
        num_edges = edge_index.shape[1]
        edge_attr = []
        
        # For now, create simple edge weights
        # In practice, these would come from the relationship properties
        for i in range(num_edges):
            # Default weight
            weight = 1.0
            
            # Could add distance-based weights, capacity-based weights, etc.
            edge_attr.append([weight])
        
        return torch.tensor(edge_attr, dtype=torch.float32)
    
    def _process_temporal_features(self) -> torch.Tensor:
        """Process temporal energy profiles"""
        if 'energy_profiles' not in self.data['temporal']:
            return None
        
        profiles = self.data['temporal']['energy_profiles']
        
        # Convert to tensor (buildings x timesteps)
        profile_array = profiles.values.T  # Transpose to get buildings x time
        
        # Normalize
        scaler = StandardScaler()
        profile_scaled = scaler.fit_transform(profile_array.T).T
        self.scalers['temporal'] = scaler
        
        return torch.tensor(profile_scaled, dtype=torch.float32)
    
    def _create_pseudo_labels(self) -> torch.Tensor:
        """Create pseudo labels for semi-supervised learning"""
        buildings = self.data['nodes']['buildings']
        
        # Create labels based on building type for initial training
        # 0: Residential, 1: Office, 2: Retail, 3: Other
        labels = []
        
        for _, building in buildings.iterrows():
            if building['function'] == 'residential':
                labels.append(0)
            elif building.get('non_res_type') == 'Office':
                labels.append(1)
            elif building.get('non_res_type') == 'Retail':
                labels.append(2)
            else:
                labels.append(3)
        
        return torch.tensor(labels, dtype=torch.long)
    
    def add_complementarity_scores(self):
        """Calculate and add complementarity scores as edge features"""
        if 'energy_profiles' not in self.data['temporal']:
            logger.warning("No energy profiles for complementarity calculation")
            return
        
        profiles = self.data['temporal']['energy_profiles']
        
        # Calculate correlation matrix
        corr_matrix = profiles.corr()
        
        # Convert to complementarity scores (negative correlation is good)
        comp_matrix = (1 - corr_matrix) / 2  # Scale to [0, 1]
        
        # Store for later use
        self.complementarity_matrix = comp_matrix
        
        logger.info("Calculated complementarity scores")
    
    def save_graphs(self, output_dir: str = "processed_data"):
        """Save processed graphs to disk"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save heterogeneous graph
        if self.hetero_graph:
            torch.save(self.hetero_graph, output_path / "hetero_graph.pt")
            logger.info(f"Saved heterogeneous graph to {output_path}/hetero_graph.pt")
        
        # Save homogeneous graph
        if self.homo_graph:
            torch.save(self.homo_graph, output_path / "homo_graph.pt")
            logger.info(f"Saved homogeneous graph to {output_path}/homo_graph.pt")
        
        # Save mappings and scalers
        metadata = {
            'node_mappings': self.node_mappings,
            'edge_mappings': self.edge_mappings,
            'scalers': self.scalers,
            'encoders': self.encoders
        }
        
        with open(output_path / "graph_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"Saved metadata to {output_path}/graph_metadata.pkl")

# Usage example
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load extracted data
    with open('processed_data/extracted_graph.pkl', 'rb') as f:
        extracted_data = pickle.load(f)
    
    # Build graphs
    builder = GraphBuilder(extracted_data, config)
    
    # Build heterogeneous graph
    hetero_graph = builder.build_heterogeneous_graph()
    print(f"Heterogeneous graph: {hetero_graph}")
    
    # Build homogeneous graph
    homo_graph = builder.build_homogeneous_graph()
    print(f"Homogeneous graph: {homo_graph}")
    
    # Add complementarity
    builder.add_complementarity_scores()
    
    # Save graphs
    builder.save_graphs()