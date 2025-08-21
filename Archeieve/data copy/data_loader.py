# data/data_loader.py
"""
Data loader for batch processing and temporal handling
Supports both static and dynamic graph learning
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, HeteroData, Batch
from torch_geometric.loader import DataLoader as GeometricDataLoader
from torch_geometric.loader import NeighborLoader, HGTLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class EnergyDataLoader:
    """Main data loader class for energy system data"""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize data loader with configuration"""
        self.config = config or {}
        self.data_path = Path(self.config.get('data_path', 'data'))
        logger.info(f"EnergyDataLoader initialized with path: {self.data_path}")
    
    def load_data(self) -> Dict[str, Any]:
        """Load energy system data"""
        logger.info("Loading energy system data")
        return {
            "status": "loaded",
            "records": 0,
            "path": str(self.data_path)
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get data loader status"""
        return {
            "initialized": True,
            "data_path": str(self.data_path),
            "config": self.config
        }
    
    def load_all_data(self) -> Union[Data, Dict[str, Any]]:
        """Load all data from mimic_data directory and convert to graph format"""
        logger.info("Loading all data from mimic_data directory")
        
        # Check if mimic_data directory exists
        mimic_path = Path(self.config.get('paths', {}).get('mimic_data', 'mimic_data'))
        if not mimic_path.exists():
            logger.warning(f"Mimic data directory not found: {mimic_path}")
            # Return a simple graph structure
            return Data(
                x=torch.randn(100, 10),  # 100 nodes with 10 features
                edge_index=torch.randint(0, 100, (2, 200)),  # 200 random edges
                num_nodes=100
            )
        
        data = {}
        
        # Load CSV files
        csv_files = {
            'buildings': 'buildings.csv',
            'transformers': 'mv_transformers.csv',
            'networks': 'lv_networks.csv'
        }
        
        for key, filename in csv_files.items():
            file_path = mimic_path / filename
            if file_path.exists():
                data[key] = pd.read_csv(file_path)
                logger.info(f"Loaded {len(data[key])} records from {filename}")
            else:
                data[key] = pd.DataFrame()
                logger.warning(f"File not found: {file_path}")
        
        # Load parquet file for energy profiles
        parquet_file = mimic_path / 'energy_profiles.parquet'
        if parquet_file.exists():
            data['energy_profiles'] = pd.read_parquet(parquet_file)
            logger.info(f"Loaded energy profiles with shape {data['energy_profiles'].shape}")
        else:
            data['energy_profiles'] = pd.DataFrame()
            logger.warning(f"Energy profiles not found: {parquet_file}")
        
        # Convert to PyTorch Geometric Data object
        if any(not df.empty for df in data.values()):
            # Create a simple graph from the data
            num_buildings = len(data.get('buildings', pd.DataFrame()))
            num_transformers = len(data.get('transformers', pd.DataFrame()))
            num_networks = len(data.get('networks', pd.DataFrame()))
            
            total_nodes = max(num_buildings + num_transformers + num_networks, 100)
            
            # Create node features (random for now, should be extracted from data)
            x = torch.randn(total_nodes, 10)
            
            # Create edges (simple connectivity for now)
            edge_index = torch.randint(0, total_nodes, (2, total_nodes * 2))
            
            return Data(
                x=x,
                edge_index=edge_index,
                num_nodes=total_nodes,
                raw_data=data  # Store original data for reference
            )
        else:
            # Return a simple default graph
            return Data(
                x=torch.randn(100, 10),
                edge_index=torch.randint(0, 100, (2, 200)),
                num_nodes=100
            )

class TemporalEnergyDataset(Dataset):
    """Dataset for temporal energy graph data"""
    
    def __init__(self, 
                 graph_data: Union[Data, HeteroData],
                 temporal_data: Optional[torch.Tensor] = None,
                 sequence_length: int = 96,
                 prediction_horizon: int = 24,
                 stride: int = 1):
        """
        Initialize temporal dataset
        
        Args:
            graph_data: Static graph structure
            temporal_data: Temporal features (nodes x time)
            sequence_length: Input sequence length (96 = 24 hours at 15-min)
            prediction_horizon: Prediction horizon
            stride: Sliding window stride
        """
        self.graph = graph_data
        self.temporal_data = temporal_data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.stride = stride
        
        # Calculate number of samples
        if temporal_data is not None:
            total_length = temporal_data.shape[1]
            self.num_samples = (total_length - sequence_length - prediction_horizon) // stride + 1
        else:
            self.num_samples = 1
            
        logger.info(f"Created temporal dataset with {self.num_samples} samples")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """Get a temporal sample"""
        if self.temporal_data is not None:
            # Calculate time window
            start_idx = idx * self.stride
            end_idx = start_idx + self.sequence_length
            target_start = end_idx
            target_end = target_start + self.prediction_horizon
            
            # Extract temporal features
            x_temporal = self.temporal_data[:, start_idx:end_idx]
            y_temporal = self.temporal_data[:, target_start:target_end]
            
            # Create data object
            data = Data(
                x=self.graph.x,
                edge_index=self.graph.edge_index,
                edge_attr=self.graph.edge_attr if hasattr(self.graph, 'edge_attr') else None,
                x_temporal=x_temporal,
                y_temporal=y_temporal,
                time_idx=torch.tensor([start_idx, end_idx])
            )
        else:
            # No temporal data - return static graph
            data = self.graph
            
        return data

class MultiTaskDataLoader:
    """Data loader for multi-task learning"""
    
    def __init__(self, 
                 graph_data: Union[Data, HeteroData],
                 task_configs: Dict,
                 batch_size: int = 32,
                 shuffle: bool = True):
        """
        Initialize multi-task data loader
        
        Args:
            graph_data: Graph data
            task_configs: Task-specific configurations
            batch_size: Batch size
            shuffle: Whether to shuffle data
        """
        self.graph = graph_data
        self.task_configs = task_configs
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Create task-specific data loaders
        self.loaders = {}
        self._create_task_loaders()
    
    def _create_task_loaders(self):
        """Create data loaders for each task"""
        
        # Clustering task - full graph
        if self.task_configs.get('clustering', {}).get('enabled', False):
            self.loaders['clustering'] = self._create_clustering_loader()
        
        # Solar optimization - node-level prediction
        if self.task_configs.get('solar_optimization', {}).get('enabled', False):
            self.loaders['solar'] = self._create_node_prediction_loader()
        
        # P2P trading - link prediction
        if self.task_configs.get('p2p_trading', {}).get('enabled', False):
            self.loaders['p2p'] = self._create_link_prediction_loader()
        
        # Congestion prediction - temporal
        if self.task_configs.get('congestion_prediction', {}).get('enabled', False):
            self.loaders['congestion'] = self._create_temporal_loader()
    
    def _create_clustering_loader(self):
        """Create loader for clustering task"""
        # For clustering, we typically use the full graph
        return GeometricDataLoader([self.graph], batch_size=1, shuffle=False)
    
    def _create_node_prediction_loader(self):
        """Create loader for node-level prediction tasks"""
        # Sample neighborhoods for scalability
        if isinstance(self.graph, HeteroData):
            loader = HGTLoader(
                self.graph,
                num_samples={key: [25, 10] for key in self.graph.node_types},
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=0
            )
        else:
            loader = NeighborLoader(
                self.graph,
                num_neighbors=[25, 10],
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=0
            )
        return loader
    
    def _create_link_prediction_loader(self):
        """Create loader for link prediction tasks"""
        # Create positive and negative edges
        edge_index = self.graph.edge_index
        num_nodes = self.graph.num_nodes
        
        # Sample negative edges
        num_neg_samples = edge_index.shape[1]
        neg_edge_index = self._sample_negative_edges(num_nodes, num_neg_samples)
        
        # Create labels
        pos_labels = torch.ones(edge_index.shape[1])
        neg_labels = torch.zeros(neg_edge_index.shape[1])
        
        # Combine edges
        all_edges = torch.cat([edge_index, neg_edge_index], dim=1)
        all_labels = torch.cat([pos_labels, neg_labels])
        
        # Create dataset
        dataset = torch.utils.data.TensorDataset(all_edges.t(), all_labels)
        
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
    
    def _create_temporal_loader(self):
        """Create loader for temporal prediction tasks"""
        # Assuming we have temporal data attached to the graph
        if hasattr(self.graph, 'temporal'):
            dataset = TemporalEnergyDataset(
                self.graph,
                self.graph.temporal,
                sequence_length=96,
                prediction_horizon=24
            )
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        else:
            logger.warning("No temporal data found for congestion prediction")
            return None
    
    def _sample_negative_edges(self, num_nodes: int, num_samples: int) -> torch.Tensor:
        """Sample negative edges for link prediction"""
        neg_edges = []
        
        while len(neg_edges) < num_samples:
            # Random sample
            src = torch.randint(0, num_nodes, (num_samples,))
            dst = torch.randint(0, num_nodes, (num_samples,))
            
            # Remove self-loops
            mask = src != dst
            src = src[mask]
            dst = dst[mask]
            
            # Add to negative edges
            neg_edges.append(torch.stack([src, dst]))
            
        neg_edge_index = torch.cat(neg_edges, dim=1)[:, :num_samples]
        return neg_edge_index
    
    def get_loader(self, task: str):
        """Get loader for specific task"""
        return self.loaders.get(task)
    
    def iterate_tasks(self):
        """Iterate over all task loaders"""
        for task, loader in self.loaders.items():
            if loader is not None:
                yield task, loader

class EnergyGraphSampler:
    """Custom sampler for energy graphs with constraints"""
    
    def __init__(self, graph: Union[Data, HeteroData], config: Dict):
        """
        Initialize sampler with energy-specific constraints
        
        Args:
            graph: Graph data
            config: Sampling configuration
        """
        self.graph = graph
        self.config = config
        
    def sample_by_transformer(self, num_samples: int) -> List[Data]:
        """Sample subgraphs by transformer boundaries"""
        samples = []
        
        # Group buildings by transformer
        if hasattr(self.graph, 'lv_network'):
            # Get LV network assignments
            lv_assignments = self.graph.x[:, -1]  # Assuming last feature is LV ID
            
            unique_lvs = torch.unique(lv_assignments)
            
            for lv_id in unique_lvs[:num_samples]:
                # Get buildings in this LV network
                mask = lv_assignments == lv_id
                node_idx = torch.where(mask)[0]
                
                # Create subgraph
                subgraph = self._create_subgraph(node_idx)
                samples.append(subgraph)
        
        return samples
    
    def sample_by_complementarity(self, num_samples: int, 
                                 correlation_matrix: np.ndarray) -> List[Data]:
        """Sample subgraphs based on complementarity patterns"""
        samples = []
        
        # Find highly complementary pairs (negative correlation)
        neg_corr = correlation_matrix < -0.3
        complementary_pairs = np.where(neg_corr)
        
        for i in range(min(num_samples, len(complementary_pairs[0]))):
            # Get pair indices
            src = complementary_pairs[0][i]
            dst = complementary_pairs[1][i]
            
            # Expand to k-hop neighborhood
            neighbors = self._get_k_hop_neighbors([src, dst], k=2)
            
            # Create subgraph
            subgraph = self._create_subgraph(neighbors)
            samples.append(subgraph)
        
        return samples
    
    def _create_subgraph(self, node_idx: torch.Tensor) -> Data:
        """Create subgraph from node indices"""
        # Get induced subgraph
        edge_mask = torch.isin(self.graph.edge_index[0], node_idx) & \
                   torch.isin(self.graph.edge_index[1], node_idx)
        
        # Filter edges
        sub_edge_index = self.graph.edge_index[:, edge_mask]
        
        # Remap node indices
        node_mapping = {int(old): new for new, old in enumerate(node_idx)}
        sub_edge_index = torch.tensor([[node_mapping[int(i)] for i in sub_edge_index[0]],
                                       [node_mapping[int(i)] for i in sub_edge_index[1]]])
        
        # Create subgraph data
        subgraph = Data(
            x=self.graph.x[node_idx],
            edge_index=sub_edge_index,
            edge_attr=self.graph.edge_attr[edge_mask] if hasattr(self.graph, 'edge_attr') else None,
            y=self.graph.y[node_idx] if hasattr(self.graph, 'y') else None
        )
        
        return subgraph
    
    def _get_k_hop_neighbors(self, seed_nodes: List[int], k: int = 2) -> torch.Tensor:
        """Get k-hop neighbors of seed nodes"""
        neighbors = set(seed_nodes)
        
        for _ in range(k):
            new_neighbors = set()
            for node in neighbors:
                # Get 1-hop neighbors
                mask = self.graph.edge_index[0] == node
                one_hop = self.graph.edge_index[1][mask].tolist()
                new_neighbors.update(one_hop)
            
            neighbors.update(new_neighbors)
        
        return torch.tensor(list(neighbors))

def create_data_loaders(config: Dict) -> Dict[str, DataLoader]:
    """
    Create all necessary data loaders based on configuration
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of data loaders
    """
    loaders = {}
    
    # Load saved graphs
    processed_path = Path(config['paths']['processed_data'])
    
    # Load heterogeneous graph if exists
    hetero_path = processed_path / "hetero_graph.pt"
    if hetero_path.exists():
        hetero_graph = torch.load(hetero_path)
        logger.info(f"Loaded heterogeneous graph from {hetero_path}")
        
        # Create multi-task loader
        task_configs = config.get('tasks', {})
        multi_loader = MultiTaskDataLoader(
            hetero_graph,
            task_configs,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )
        
        loaders['multi_task'] = multi_loader
    
    # Load homogeneous graph if exists
    homo_path = processed_path / "homo_graph.pt"
    if homo_path.exists():
        homo_graph = torch.load(homo_path)
        logger.info(f"Loaded homogeneous graph from {homo_path}")
        
        # Create standard loader
        loaders['standard'] = GeometricDataLoader(
            [homo_graph],
            batch_size=1,
            shuffle=False
        )
        
        # Create temporal loader if temporal data exists
        if hasattr(homo_graph, 'temporal'):
            temporal_dataset = TemporalEnergyDataset(
                homo_graph,
                homo_graph.temporal,
                sequence_length=config['model']['temporal']['sequence_length'],
                prediction_horizon=24
            )
            
            loaders['temporal'] = DataLoader(
                temporal_dataset,
                batch_size=config['training']['batch_size'],
                shuffle=True
            )
    
    logger.info(f"Created {len(loaders)} data loaders")
    
    return loaders

# Usage example
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load task configuration
    with open('config/tasks_config.yaml', 'r') as f:
        task_config = yaml.safe_load(f)
    
    # Create data loaders
    loaders = create_data_loaders(config)
    
    # Test multi-task loader
    if 'multi_task' in loaders:
        multi_loader = loaders['multi_task']
        
        for task, loader in multi_loader.iterate_tasks():
            print(f"\nTask: {task}")
            for batch in loader:
                print(f"  Batch: {batch}")
                break  # Just show first batch
    
    # Test temporal loader
    if 'temporal' in loaders:
        temporal_loader = loaders['temporal']
        
        print("\nTemporal loader:")
        for batch in temporal_loader:
            print(f"  Input shape: {batch.x_temporal.shape}")
            print(f"  Target shape: {batch.y_temporal.shape}")
            break  # Just show first batch