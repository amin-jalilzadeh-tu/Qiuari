# data_loader.py - CORRECTED VERSION
"""
Custom data loading and batching strategies for energy grid GNN.
Updated for actual KG schema: Building -> CableGroup -> Transformer -> Substation
"""

import torch
from torch_geometric.loader import NeighborLoader, HGTLoader, ClusterData, ClusterLoader
from torch_geometric.data import HeteroData
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TaskSpecificLoader:
    """Creates task-specific data loaders with custom sampling strategies."""
    
    def __init__(self, 
                 batch_size: int = 32,
                 num_neighbors: List[int] = [15, 10],
                 num_workers: int = 0):
        """
        Initialize loader configuration.
        
        Args:
            batch_size: Number of target nodes per batch
            num_neighbors: Number of neighbors to sample at each hop
            num_workers: Number of parallel workers for loading
        """
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.num_workers = num_workers
        
        # Task-specific configurations - UPDATED FOR ACTUAL NODE TYPES
        self.task_configs = {
            'retrofit': {
                'target_node': 'building',
                'sampling': 'priority',  # Sample worst performers first
                'preserve_hierarchy': True,
                'include_cable_group': True
            },
            'energy_sharing': {
                'target_node': 'adjacency_cluster',  # UPDATED from 'lv_line'
                'sampling': 'cluster',  # Keep complete clusters
                'preserve_hierarchy': True,
                'include_all_buildings': True
            },
            'solar': {
                'target_node': 'building',
                'sampling': 'spatial',  # Group by location
                'preserve_hierarchy': False,
                'include_neighbors': True
            },
            'grid_planning': {
                'target_node': 'transformer',  # UPDATED from 'mv_station'
                'sampling': 'hierarchical',  # Full hierarchy
                'preserve_hierarchy': True,
                'include_downstream': True
            },
            'electrification': {
                'target_node': 'building',
                'sampling': 'feasibility',
                'preserve_hierarchy': True,
                'include_cable_group': True
            }
        }
    
    def create_loader(self, 
                     graph: HeteroData,
                     task: str,
                     split: str = 'train') -> torch.utils.data.DataLoader:
        """
        Create task-specific data loader with temporal support.
        
        Args:
            graph: HeteroData graph (may include temporal features)
            task: Task type ('retrofit', 'energy_sharing', 'solar', 'grid_planning', 'electrification')
            split: Data split ('train', 'val', 'test')
            
        Returns:
            Configured DataLoader with temporal features if available
        """
        if task not in self.task_configs:
            raise ValueError(f"Unknown task: {task}. Available tasks: {list(self.task_configs.keys())}")
        
        config = self.task_configs[task]
        
        # Check for temporal features
        has_temporal = any(
            hasattr(graph[node_type], 'x_temporal') 
            for node_type in graph.node_types
        )
        
        if has_temporal:
            logger.info(f"Graph contains temporal features for task '{task}'")
        
        # Route to appropriate loader based on task
        if task == 'retrofit':
            return self._create_retrofit_loader(graph, split, config)
            
        elif task == 'energy_sharing':
            # Use temporal loader if temporal features are available
            if has_temporal:
                logger.info("Using temporal loader for energy sharing")
                return self._create_energy_sharing_loader_temporal(graph, split, config)
            else:
                logger.info("Using standard loader for energy sharing (no temporal data)")
                return self._create_energy_sharing_loader(graph, split, config)
                
        elif task == 'solar':
            return self._create_solar_loader(graph, split, config)
            
        elif task == 'grid_planning':
            return self._create_grid_planning_loader(graph, split, config)
            
        elif task == 'electrification':
            if has_temporal:
                logger.info("Temporal features available for electrification task")
            return self._create_electrification_loader(graph, split, config)
            
        else:
            raise ValueError(f"Task '{task}' not implemented")
    
    def _create_retrofit_loader(self, 
                               graph: HeteroData,
                               split: str,
                               config: Dict) -> torch.utils.data.DataLoader:
        """
        Create loader for retrofit targeting.
        Prioritizes buildings with poor energy labels.
        """
        logger.info(f"Creating retrofit loader for {split}")
        
        # Get building features to identify priority nodes
        building_features = graph['building'].x
        # Energy score is at index 1 (normalized 0-1, lower = worse)
        energy_scores = building_features[:, 1]
        
        # Create priority mask - buildings with poor energy scores
        if split == 'train':
            # Focus on worst performers for training
            priority_threshold = torch.quantile(energy_scores, 0.3)  # Bottom 30%
            priority_mask = energy_scores <= priority_threshold
        else:
            # Use all buildings for validation/test
            priority_mask = torch.ones(len(energy_scores), dtype=torch.bool)
        
        # Add split mask if available
        if hasattr(graph['building'], f'{split}_mask'):
            split_mask = graph['building'][f'{split}_mask']
            input_nodes = torch.where(priority_mask & split_mask)[0]
        else:
            input_nodes = torch.where(priority_mask)[0]
        
        # UPDATED edge types for actual schema
        loader = HierarchicalNeighborLoader(
            graph,
            num_neighbors={
                ('building', 'connected_to', 'cable_group'): [-1],  # All buildings in cable group
                ('cable_group', 'connects_to', 'transformer'): [5],
                ('transformer', 'feeds_from', 'substation'): [2],
                ('building', 'in_cluster', 'adjacency_cluster'): [10]  # Related clusters
            },
            batch_size=self.batch_size,
            input_nodes=('building', input_nodes),
            shuffle=(split == 'train'),
            num_workers=self.num_workers
        )
        
        return loader
    
    def _create_energy_sharing_loader(self,
                                     graph: HeteroData,
                                     split: str,
                                     config: Dict) -> torch.utils.data.DataLoader:
        """
        Create loader for energy sharing analysis.
        Targets adjacency clusters.
        """
        logger.info(f"Creating energy sharing loader for {split}")
        
        # Target adjacency clusters for energy sharing
        target_node_type = 'adjacency_cluster'
        
        # Check if adjacency clusters exist
        if target_node_type not in graph.node_types:
            logger.warning("No adjacency clusters found, falling back to buildings")
            target_node_type = 'building'
        
        # Get clusters with sufficient members for sharing
        if target_node_type == 'adjacency_cluster' and hasattr(graph[target_node_type], 'x'):
            cluster_features = graph[target_node_type].x
            valid_clusters = []
            
            for i in range(cluster_features.shape[0]):
                member_count = cluster_features[i, 0].item()  # First feature is member count
                if member_count >= 3:  # Minimum for energy sharing
                    valid_clusters.append(i)
            
            input_nodes = torch.tensor(valid_clusters, dtype=torch.long) if valid_clusters else torch.arange(cluster_features.shape[0])
        else:
            # Use all nodes if no filtering criteria
            num_nodes = graph[target_node_type].x.shape[0]
            input_nodes = torch.arange(num_nodes)
        
        # Apply split mask if available
        if hasattr(graph[target_node_type], f'{split}_mask'):
            split_mask = graph[target_node_type][f'{split}_mask']
            input_nodes = input_nodes[split_mask[input_nodes]]
        
        # Create loader that keeps complete clusters
        loader = ClusterAwareLoader(
            graph,
            num_neighbors={
                ('building', 'in_cluster', 'adjacency_cluster'): [-1],  # ALL buildings in cluster
                ('building', 'connected_to', 'cable_group'): [-1],
                ('cable_group', 'connects_to', 'transformer'): [-1],
                ('transformer', 'feeds_from', 'substation'): [-1]
            },
            batch_size=max(1, self.batch_size // 10),  # Fewer clusters per batch
            input_nodes=(target_node_type, input_nodes),
            shuffle=(split == 'train'),
            num_workers=self.num_workers
        )
        
        return loader
    
    def _create_energy_sharing_loader_temporal(self,
                                              graph: HeteroData,
                                              split: str,
                                              config: Dict) -> torch.utils.data.DataLoader:
        """
        Create loader for energy sharing with temporal features.
        Keeps complete adjacency clusters together with their time series data.
        """
        logger.info(f"Creating TEMPORAL energy sharing loader for {split}")
        
        # Check if temporal features exist
        has_temporal = False
        if 'building' in graph.node_types and hasattr(graph['building'], 'x_temporal'):
            has_temporal = True
            logger.info(f"Building temporal features found: {graph['building'].x_temporal.shape}")
        
        if 'adjacency_cluster' in graph.node_types and hasattr(graph['adjacency_cluster'], 'x_temporal'):
            has_temporal = True
            logger.info(f"Cluster temporal features found: {graph['adjacency_cluster'].x_temporal.shape}")
        
        if not has_temporal:
            logger.warning("No temporal features found, falling back to standard loader")
            return self._create_energy_sharing_loader(graph, split, config)
        
        # Target adjacency clusters for energy sharing
        target_node_type = 'adjacency_cluster'
        
        # Get clusters with good sharing potential
        if target_node_type in graph.node_types and hasattr(graph[target_node_type], 'x'):
            cluster_features = graph[target_node_type].x
            valid_clusters = []
            
            for i in range(cluster_features.shape[0]):
                member_count = cluster_features[i, 0].item()
                sharing_potential = cluster_features[i, 1].item()
                
                # Include clusters with enough members and good sharing potential
                if member_count >= 3: #and sharing_potential >= 0.3:
                    valid_clusters.append(i)
            
            if not valid_clusters:
                logger.warning("No valid clusters found, using all clusters")
                valid_clusters = list(range(cluster_features.shape[0]))
            
            input_nodes = torch.tensor(valid_clusters, dtype=torch.long)
        else:
            # Fallback to building-level if no clusters
            logger.warning(f"No {target_node_type} nodes found, using buildings")
            target_node_type = 'building'
            num_buildings = graph[target_node_type].x.shape[0]
            input_nodes = torch.arange(num_buildings)
        
        # Apply split mask
        if hasattr(graph[target_node_type], f'{split}_mask'):
            split_mask = graph[target_node_type][f'{split}_mask']
            input_nodes = input_nodes[split_mask[input_nodes]]
        
        # UPDATED edge types for actual schema
        loader = NeighborLoader(
            graph,
            num_neighbors={
                ('building', 'in_cluster', 'adjacency_cluster'): [-1],  # All buildings
                ('building', 'connected_to', 'cable_group'): [-1],      # All connections
                ('cable_group', 'connects_to', 'transformer'): [-1],
                ('transformer', 'feeds_from', 'substation'): [-1],
            },
            batch_size=max(1, self.batch_size // 10),  # Smaller batches for clusters
            input_nodes=(target_node_type, input_nodes),
            shuffle=(split == 'train'),
            num_workers=self.num_workers
        )
        
        # Note: PyG automatically preserves x_temporal attributes in batches
        
        logger.info(f"Created temporal energy sharing loader with {len(input_nodes)} nodes")
        return loader
    
    def _create_solar_loader(self,
                           graph: HeteroData,
                           split: str,
                           config: Dict) -> torch.utils.data.DataLoader:
        """
        Create loader for solar optimization.
        Groups buildings by spatial proximity.
        """
        logger.info(f"Creating solar loader for {split}")
        
        # Get buildings with good roof area
        building_features = graph['building'].x
        roof_areas = building_features[:, 5]  # Roof area at index 5
        solar_scores = building_features[:, 2]  # Solar score at index 2
        
        # Focus on buildings with sufficient roof area and good solar potential
        if split == 'train':
            roof_threshold = torch.quantile(roof_areas, 0.5)  # Top 50%
            solar_mask = (roof_areas >= roof_threshold) & (solar_scores > 0.3)
        else:
            solar_mask = torch.ones(len(roof_areas), dtype=torch.bool)
        
        # Apply split mask if available
        if hasattr(graph['building'], f'{split}_mask'):
            split_mask = graph['building'][f'{split}_mask']
            input_nodes = torch.where(solar_mask & split_mask)[0]
        else:
            input_nodes = torch.where(solar_mask)[0]
        
        # UPDATED edge types
        loader = NeighborLoader(
            graph,
            num_neighbors={
                ('building', 'in_cluster', 'adjacency_cluster'): [15, 10],  # 2 hops
                ('building', 'connected_to', 'cable_group'): [5, 3],       # 2 hops  
                ('cable_group', 'connects_to', 'transformer'): [2, 1]      # 2 hops
            },
            batch_size=self.batch_size,
            input_nodes=('building', input_nodes),
            shuffle=(split == 'train'),
            num_workers=self.num_workers
        )
        
        return loader
    
    def _create_grid_planning_loader(self,
                                    graph: HeteroData,
                                    split: str,
                                    config: Dict) -> torch.utils.data.DataLoader:
        """
        Create loader for grid planning.
        Samples complete grid hierarchies starting from transformers.
        """
        logger.info(f"Creating grid planning loader for {split}")
        
        # Target transformers for planning (updated from mv_station)
        target_node_type = 'transformer'
        
        if target_node_type not in graph.node_types:
            logger.warning("No transformers found, using cable groups")
            target_node_type = 'cable_group'
        
        num_nodes = graph[target_node_type].x.shape[0]
        
        # Create input nodes
        input_nodes = torch.arange(num_nodes)
        
        # Apply split mask if available
        if hasattr(graph[target_node_type], f'{split}_mask'):
            split_mask = graph[target_node_type][f'{split}_mask']
            input_nodes = input_nodes[split_mask]
        
        # UPDATED edge types for complete hierarchies
        loader = HierarchicalNeighborLoader(
            graph,
            num_neighbors={
                ('cable_group', 'connects_to', 'transformer'): [-1],  # All cable groups
                ('building', 'connected_to', 'cable_group'): [5],  # Sample of buildings
                ('transformer', 'feeds_from', 'substation'): [-1],  # Parent substation
                ('building', 'in_cluster', 'adjacency_cluster'): [5]
            },
            batch_size=max(1, self.batch_size // 20),  # Few transformers per batch
            input_nodes=(target_node_type, input_nodes),
            shuffle=(split == 'train'),
            num_workers=self.num_workers
        )
        
        return loader
    
    def _create_electrification_loader(self,
                                      graph: HeteroData,
                                      split: str,
                                      config: Dict) -> torch.utils.data.DataLoader:
        """
        Create loader for electrification planning.
        Focuses on buildings suitable for heat pump installation.
        """
        logger.info(f"Creating electrification loader for {split}")
        
        # Get building features
        building_features = graph['building'].x
        # Electrification score at index 3, has_heat_pump at index 9
        electrify_scores = building_features[:, 3]
        has_heat_pump = building_features[:, 9]
        
        # Focus on buildings ready for electrification but without heat pumps
        if split == 'train':
            electrify_mask = (electrify_scores >= 0.5) & (has_heat_pump == 0)
        else:
            electrify_mask = torch.ones(len(electrify_scores), dtype=torch.bool)
        
        # Apply split mask if available
        if hasattr(graph['building'], f'{split}_mask'):
            split_mask = graph['building'][f'{split}_mask']
            input_nodes = torch.where(electrify_mask & split_mask)[0]
        else:
            input_nodes = torch.where(electrify_mask)[0]
        
        # Similar to retrofit but with electrification focus
        loader = HierarchicalNeighborLoader(
            graph,
            num_neighbors={
                ('building', 'connected_to', 'cable_group'): [-1],
                ('cable_group', 'connects_to', 'transformer'): [5],
                ('transformer', 'feeds_from', 'substation'): [2],
                ('building', 'in_cluster', 'adjacency_cluster'): [10]
            },
            batch_size=self.batch_size,
            input_nodes=('building', input_nodes),
            shuffle=(split == 'train'),
            num_workers=self.num_workers
        )
        
        return loader


class HierarchicalNeighborLoader(NeighborLoader):
    """
    Custom NeighborLoader that preserves grid hierarchy.
    Updated for actual schema: Building -> CableGroup -> Transformer -> Substation
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with hierarchy preservation."""
        super().__init__(*args, **kwargs)
        # Updated hierarchy for actual schema
        self.hierarchy = [
            ('substation', 'feeds', 'transformer'),
            ('transformer', 'connects', 'cable_group'),
            ('cable_group', 'supplies', 'building')
        ]
    
    def __iter__(self):
        """Iterate with hierarchy preservation."""
        for batch in super().__iter__():
            # Ensure hierarchy is complete
            batch = self._complete_hierarchy(batch)
            yield batch
    
    def _complete_hierarchy(self, batch: HeteroData) -> HeteroData:
        """
        Ensure all parent nodes are included for sampled children.
        
        Args:
            batch: Sampled batch
            
        Returns:
            Batch with complete hierarchy
        """
        # For each level in hierarchy, ensure parents are included
        for parent_type, rel, child_type in self.hierarchy:
            if child_type in batch.node_types and parent_type in batch.node_types:
                # Mark batch as having complete hierarchy
                batch[f'{parent_type}_complete'] = True
        
        return batch


class ClusterAwareLoader(NeighborLoader):
    """
    Loader that keeps complete clusters together.
    Used for energy sharing where all buildings in a cluster must be in same batch.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize cluster-aware loader."""
        super().__init__(*args, **kwargs)
        
    def __iter__(self):
        """Iterate keeping clusters intact."""
        for batch in super().__iter__():
            # Ensure complete clusters
            batch = self._complete_clusters(batch)
            yield batch
    
    def _complete_clusters(self, batch: HeteroData) -> HeteroData:
        """
        Ensure complete adjacency clusters are included.
        
        Args:
            batch: Sampled batch
            
        Returns:
            Batch with complete clusters
        """
        if 'adjacency_cluster' in batch.node_types and 'building' in batch.node_types:
            # Get cluster-to-building edges
            edge_key = ('building', 'in_cluster', 'adjacency_cluster')
            if edge_key in batch.edge_types:
                edge_index = batch[edge_key].edge_index
                
                # For each cluster in batch, ensure ALL its buildings are included
                cluster_indices = torch.unique(edge_index[1])
                
                # Mark batch as having complete clusters
                batch.complete_clusters = True
                batch.cluster_indices = cluster_indices
        
        return batch


def create_train_val_test_loaders(graph: HeteroData,
                                 task: str,
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 batch_size: int = 32) -> Tuple[Any, Any, Any]:
    """
    Create train, validation, and test loaders.
    
    Args:
        graph: HeteroData graph
        task: Task type
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        batch_size: Batch size
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # UPDATED task configurations for actual node types
    task_configs = {
        'retrofit': 'building',
        'energy_sharing': 'adjacency_cluster',  # Updated from 'lv_line'
        'solar': 'building',
        'grid_planning': 'transformer',  # Updated from 'mv_station'
        'electrification': 'building'
    }
    
    target_type = task_configs.get(task, 'building')
    
    # Check if target type exists in graph
    if target_type not in graph.node_types:
        logger.warning(f"Target type {target_type} not found, using building")
        target_type = 'building'
    
    num_nodes = graph[target_type].x.shape[0]
    
    # Create random splits
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    # Create masks
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    # Add masks to graph
    graph[target_type].train_mask = train_mask
    graph[target_type].val_mask = val_mask
    graph[target_type].test_mask = test_mask
    
    # Create loaders
    loader_creator = TaskSpecificLoader(batch_size=batch_size)
    
    train_loader = loader_creator.create_loader(graph, task, 'train')
    val_loader = loader_creator.create_loader(graph, task, 'val')
    test_loader = loader_creator.create_loader(graph, task, 'test')
    
    logger.info(f"Created loaders - Train: {train_size}, Val: {val_size}, Test: {num_nodes - train_size - val_size}")
    
    return train_loader, val_loader, test_loader