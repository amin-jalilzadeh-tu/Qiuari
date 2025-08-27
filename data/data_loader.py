"""
Data loader for Energy GNN
Focuses on complete LV groups and complementarity sampling
"""

import torch
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.loader import NeighborLoader, ClusterData, ClusterLoader
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
from sklearn.preprocessing import StandardScaler


from utils.feature_mapping import feature_mapper
from utils.data_validator import DimensionValidator
from utils.constants import *


class EnergyDataLoader:
    """
    Custom data loader for energy GNN focusing on LV groups
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        mode: str = 'train'
    ):
        """
        Initialize data loader
        
        Args:
            config: Configuration dictionary
            mode: One of 'train', 'val', 'test'
        """
        self.config = config
        self.mode = mode
        self.batch_size = config.get('batch_size', 32)
        self.num_workers = config.get('num_workers', 4)
        self.lv_group_only = config.get('lv_group_only', True)
        self.complementarity_sampling = config.get('complementarity_sampling', True)
        
        # Scalers for normalization
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
    def create_lv_group_data(
        self,
        buildings: pd.DataFrame,
        grid_topology: Dict,
        temporal_data: pd.DataFrame,
        lv_group_id: str
    ) -> Data:
        """
        Create PyG Data object for a single LV group
        
        Args:
            buildings: Building features dataframe
            grid_topology: Grid topology information
            temporal_data: Temporal consumption/generation data
            lv_group_id: ID of the LV group
            
        Returns:
            PyG Data object for the LV group
        """
        # Filter buildings in this LV group
        lv_buildings = buildings[buildings['lv_group'] == lv_group_id].copy()
        
        if len(lv_buildings) < self.config.get('min_cluster_size', 3):
            return None  # Skip small LV groups
            
        # Create node features
        node_features = self._create_node_features(lv_buildings)
        
        # Create edge index (electrical connections within LV group)
        edge_index = self._create_edge_index(lv_buildings, grid_topology)
        
        # Create edge attributes
        edge_attr = self._create_edge_attributes(edge_index, lv_buildings, grid_topology)
        
        # Extract temporal profiles
        temporal_profiles = self._extract_temporal_profiles(
            lv_buildings.index.tolist(),
            temporal_data
        )
        
        # Calculate targets
        targets = self._calculate_targets(temporal_profiles, lv_buildings)
        
        # Create Data object
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float32),  # Always include edge_attr
            temporal_profiles=torch.tensor(temporal_profiles, dtype=torch.float32),
            **targets,
            lv_group=lv_group_id,
            building_ids=lv_buildings.index.tolist()
        )
        
        # Add placeholder y for compatibility (will be replaced by pseudo-labels)
        # Set to None to indicate unsupervised mode
        data.y = None
        
        # Add complementarity pairs if enabled
        if self.complementarity_sampling:
            comp_pairs = self._find_complementarity_pairs(temporal_profiles)
            data.complementary_pairs = torch.tensor(comp_pairs, dtype=torch.long)
            
        return data
    
    def _create_node_features(self, buildings: pd.DataFrame) -> np.ndarray:
        """Create node feature matrix using feature mapper"""
        # Use feature mapper for Dutch data
        if self.config.get('dutch_data_mode', True):
            return feature_mapper.get_feature_vector(buildings)
        
        # Fallback to original implementation
        return self._create_node_features_legacy(buildings)
    
    def _create_node_features_legacy(self, buildings: pd.DataFrame) -> np.ndarray:
        """Legacy feature creation (kept for compatibility)"""
    
    def _create_edge_index(
        self,
        buildings: pd.DataFrame,
        grid_topology: Dict
    ) -> np.ndarray:
        """
        Create edge index for electrical connections
        """
        edges = []
        building_ids = buildings.index.tolist()
        id_to_idx = {bid: idx for idx, bid in enumerate(building_ids)}
        
        # Add electrical connections from grid topology
        if 'connections' in grid_topology:
            for connection in grid_topology['connections']:
                if connection['from'] in id_to_idx and connection['to'] in id_to_idx:
                    edges.append([id_to_idx[connection['from']], id_to_idx[connection['to']]])
                    # Add reverse edge for undirected graph
                    edges.append([id_to_idx[connection['to']], id_to_idx[connection['from']]])
                    
        # Add adjacency connections (buildings sharing walls)
        if 'adjacency' in grid_topology:
            for adj in grid_topology['adjacency']:
                if adj['building1'] in id_to_idx and adj['building2'] in id_to_idx:
                    edges.append([id_to_idx[adj['building1']], id_to_idx[adj['building2']]])
                    edges.append([id_to_idx[adj['building2']], id_to_idx[adj['building1']]])
                    
        # If no edges found, create a fully connected graph (small LV group)
        if not edges and len(building_ids) < 10:
            for i in range(len(building_ids)):
                for j in range(i + 1, len(building_ids)):
                    edges.append([i, j])
                    edges.append([j, i])
                    
        return np.array(edges, dtype=np.int64).T if edges else np.array([[], []], dtype=np.int64)
    
    def _create_edge_attributes(
        self,
        edge_index: np.ndarray,
        buildings: pd.DataFrame,
        grid_topology: Dict
    ) -> np.ndarray:
        """Create edge attributes (distance, cable capacity, etc.)"""
        if edge_index.shape[1] == 0:
            # Return empty array with correct shape for consistency
            return np.array([], dtype=np.float32).reshape(0, 3)
            
        num_edges = edge_index.shape[1]
        edge_attr = np.zeros((num_edges, 3))  # [distance, capacity, resistance]
        
        # Default values
        edge_attr[:, 0] = 10.0  # Default distance 10m
        edge_attr[:, 1] = 100.0  # Default capacity 100kW
        edge_attr[:, 2] = 0.01  # Default resistance
        
        # Update with actual values from grid topology if available
        if 'cable_properties' in grid_topology:
            building_ids = buildings.index.tolist()
            for i in range(num_edges):
                from_idx = edge_index[0, i]
                to_idx = edge_index[1, i]
                from_id = building_ids[from_idx]
                to_id = building_ids[to_idx]
                
                cable_key = f"{from_id}_{to_id}"
                if cable_key in grid_topology['cable_properties']:
                    props = grid_topology['cable_properties'][cable_key]
                    edge_attr[i, 0] = props.get('distance', 10.0)
                    edge_attr[i, 1] = props.get('capacity', 100.0)
                    edge_attr[i, 2] = props.get('resistance', 0.01)
                    
        return edge_attr
    
    def _extract_temporal_profiles(
        self,
        building_ids: List[str],
        temporal_data: pd.DataFrame
    ) -> np.ndarray:
        """Extract temporal consumption/generation profiles"""
        # Assume temporal_data has columns: building_id, timestamp, consumption, generation
        profiles = []
        
        for bid in building_ids:
            # Handle both DataFrame and dict formats
            if isinstance(temporal_data, dict):
                # temporal_data is a dict with building_id as key
                if bid in temporal_data:
                    building_data = temporal_data[bid]
                    # Assume first column is consumption
                    consumption = building_data[:, 0] if len(building_data.shape) > 1 else building_data
                    consumption = consumption[:96]  # Get 96 timesteps
                    
                    # Pad if necessary
                    if len(consumption) < 96:
                        consumption = np.pad(consumption, (0, 96 - len(consumption)), 'mean')
                        
                    profiles.append(consumption)
                else:
                    # Use average profile if no data
                    profiles.append(np.ones(96) * 10.0)  # Default 10kW
            else:
                # Original DataFrame logic
                building_data = temporal_data[temporal_data['building_id'] == bid]
                
                if len(building_data) > 0:
                    # Get 96 timesteps (24 hours * 4 quarters)
                    consumption = building_data['consumption'].values[:96]
                    
                    # Pad if necessary
                    if len(consumption) < 96:
                        consumption = np.pad(consumption, (0, 96 - len(consumption)), 'mean')
                        
                    profiles.append(consumption)
                else:
                    # Use average profile if no data
                    profiles.append(np.ones(96) * 10.0)  # Default 10kW
                
        return np.array(profiles)
    
    def _calculate_targets(
        self,
        temporal_profiles: np.ndarray,
        buildings: pd.DataFrame
    ) -> Dict[str, torch.Tensor]:
        """Calculate target values for training"""
        targets = {}
        
        # Individual peaks
        targets['individual_peaks'] = temporal_profiles.max(axis=1)
        
        # Generation (if buildings have solar)
        if 'has_solar' in buildings.columns:
            # Simplified: assume solar generates during day hours (8am-6pm)
            generation = np.zeros_like(temporal_profiles)
            solar_mask = buildings['has_solar'].values
            day_hours = slice(32, 72)  # 8am-6pm in 15-min intervals
            
            for i, has_solar in enumerate(solar_mask):
                if has_solar:
                    # Simple solar profile
                    generation[i, day_hours] = temporal_profiles[i, day_hours].mean() * 0.3
                    
            targets['generation'] = generation
        else:
            targets['generation'] = np.zeros_like(temporal_profiles)
            
        # Demand is consumption
        targets['demand'] = temporal_profiles
        
        # We don't have actual power flow or transformer capacity data
        # These would require detailed grid simulation
        # Removing placeholder values that cause issues
        
        # Convert to tensors
        for key, value in targets.items():
            targets[key] = torch.tensor(value, dtype=torch.float32)
            
        return targets
    
    def _find_complementarity_pairs(
        self,
        temporal_profiles: np.ndarray,
        threshold: float = -0.3
    ) -> List[Tuple[int, int]]:
        """
        Find pairs of buildings with complementary consumption patterns
        
        Args:
            temporal_profiles: Consumption profiles [N, T]
            threshold: Correlation threshold for complementarity
            
        Returns:
            List of complementary pairs (indices)
        """
        n_buildings = temporal_profiles.shape[0]
        pairs = []
        
        # Calculate correlation matrix
        profiles_norm = (temporal_profiles - temporal_profiles.mean(axis=1, keepdims=True)) / (
            temporal_profiles.std(axis=1, keepdims=True) + 1e-8
        )
        corr_matrix = np.matmul(profiles_norm, profiles_norm.T) / temporal_profiles.shape[1]
        
        # Find complementary pairs (negative correlation)
        for i in range(n_buildings):
            for j in range(i + 1, n_buildings):
                if corr_matrix[i, j] < threshold:
                    pairs.append((i, j))
                    
        return pairs
    
    def filter_complete_lv_groups(
        self,
        graph_data: List[Data]
    ) -> List[Data]:
        """
        Filter to keep only complete LV groups
        
        Args:
            graph_data: List of Data objects
            
        Returns:
            Filtered list with complete LV groups
        """
        if not self.lv_group_only:
            return graph_data
            
        filtered = []
        for data in graph_data:
            # Check if LV group is complete
            if self._is_complete_lv_group(data):
                filtered.append(data)
                
        print(f"Filtered {len(graph_data)} to {len(filtered)} complete LV groups")
        return filtered
    
    def _is_complete_lv_group(self, data: Data) -> bool:
        """Check if LV group is complete (has transformer connection)"""
        # Check minimum size (relaxed for testing)
        if data.x.shape[0] < self.config.get('min_cluster_size', 2):
            return False
            
        # Check maximum size
        if data.x.shape[0] > self.config.get('max_cluster_size', 100):
            return False
            
        # Check if has edges (connected) - allow graphs without edges for testing
        # if data.edge_index.shape[1] == 0:
        #     return False
            
        # Check if has transformer capacity (indicates connection to MV)
        if hasattr(data, 'transformer_capacity'):
            if data.transformer_capacity.item() <= 0:
                return False
                
        return True
    
    def create_dataloaders(
        self,
        train_data: List[Data],
        val_data: List[Data],
        test_data: Optional[List[Data]] = None
    ) -> Tuple[PyGDataLoader, PyGDataLoader, Optional[PyGDataLoader]]:
        """
        Create PyTorch Geometric dataloaders
        
        Args:
            train_data: Training data
            val_data: Validation data
            test_data: Test data (optional)
            
        Returns:
            Train, validation, and test dataloaders
        """
        # Filter for complete LV groups
        train_data = self.filter_complete_lv_groups(train_data)
        val_data = self.filter_complete_lv_groups(val_data)
        if test_data:
            test_data = self.filter_complete_lv_groups(test_data)
            
        # Create dataloaders
        # Adjust batch size if we have fewer samples
        actual_batch_size = min(self.batch_size, len(train_data)) if train_data else 1
        
        train_loader = PyGDataLoader(
            train_data,
            batch_size=actual_batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            drop_last=False  # Don't drop last batch when we have few samples
        )
        
        val_batch_size = min(self.batch_size, len(val_data)) if val_data else 1
        
        val_loader = PyGDataLoader(
            val_data,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )
        
        test_loader = None
        if test_data:
            test_batch_size = min(self.batch_size, len(test_data)) if test_data else 1
            test_loader = PyGDataLoader(
                test_data,
                batch_size=test_batch_size,
                shuffle=False,
                num_workers=self.num_workers
            )
            
        return train_loader, val_loader, test_loader
    
    def create_cluster_loader(
        self,
        data: Data,
        num_parts: int = 10
    ) -> ClusterLoader:
        """
        Create cluster-based loader for large graphs
        
        Args:
            data: Large graph data
            num_parts: Number of clusters to partition into
            
        Returns:
            ClusterLoader
        """
        cluster_data = ClusterData(
            data,
            num_parts=num_parts,
            recursive=False,
            save_dir='./cluster_data'
        )
        
        cluster_loader = ClusterLoader(
            cluster_data,
            batch_size=1,
            shuffle=True,
            num_workers=self.num_workers
        )
        
        return cluster_loader
    
    def create_neighbor_loader(
        self,
        data: HeteroData,
        num_neighbors: List[int] = [15, 10, 5],
        batch_size: int = 128
    ) -> NeighborLoader:
        """
        Create neighbor sampling loader for large heterogeneous graphs
        
        Args:
            data: Heterogeneous graph data
            num_neighbors: Number of neighbors to sample per layer
            batch_size: Batch size for sampling
            
        Returns:
            NeighborLoader
        """
        loader = NeighborLoader(
            data,
            num_neighbors=num_neighbors,
            batch_size=batch_size,
            shuffle=True if self.mode == 'train' else False,
            num_workers=self.num_workers
        )
        
        return loader
    
    def augment_with_complementarity(
        self,
        data: Data
    ) -> Data:
        """
        Augment data with complementarity-based features
        
        Args:
            data: Original data
            
        Returns:
            Augmented data
        """
        if not hasattr(data, 'temporal_profiles'):
            return data
            
        profiles = data.temporal_profiles
        
        # Calculate pairwise complementarity scores
        n_nodes = profiles.shape[0]
        comp_matrix = torch.zeros((n_nodes, n_nodes))
        
        # Normalize profiles
        profiles_norm = (profiles - profiles.mean(dim=1, keepdim=True)) / (
            profiles.std(dim=1, keepdim=True) + 1e-8
        )
        
        # Correlation matrix
        comp_matrix = torch.matmul(profiles_norm, profiles_norm.t()) / profiles.shape[1]
        
        # Add as node feature (average complementarity with neighbors)
        edge_index = data.edge_index
        comp_features = []
        
        for node_idx in range(n_nodes):
            # Get neighbors
            neighbors = edge_index[1, edge_index[0] == node_idx]
            if len(neighbors) > 0:
                # Average complementarity with neighbors (negative correlation is good)
                avg_comp = -comp_matrix[node_idx, neighbors].mean()
                comp_features.append(avg_comp)
            else:
                comp_features.append(0.0)
                
        # Add to node features
        comp_features = torch.tensor(comp_features).unsqueeze(1)
        data.x = torch.cat([data.x, comp_features], dim=1)
        
        # Store complementarity matrix for loss calculation
        data.complementarity_matrix = comp_matrix
        
        return data


def collate_energy_batch(batch: List[Data]) -> Batch:
    """
    Custom collate function for energy data batches
    
    Args:
        batch: List of Data objects
        
    Returns:
        Batched Data object
    """
    # Filter out None values (incomplete LV groups)
    batch = [data for data in batch if data is not None]
    
    if not batch:
        return None
        
    # Use default PyG batching
    return Batch.from_data_list(batch)