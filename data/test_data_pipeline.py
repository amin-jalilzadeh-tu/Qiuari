# test_pipeline_consolidated.py
"""
Consolidated test file with all components included.
Run this single file to test the complete pipeline without separate imports.

IMPORTANT FIX: NeighborLoader requires all node attributes to be tensors.
We store IDs and aggregates on the GraphConstructor instance instead of 
as node attributes to avoid "invalid feature tensor type" errors.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from torch_geometric.data import HeteroData
from torch_geometric.loader import NeighborLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# GRAPH CONSTRUCTOR (Simplified Version)
# ============================================================================

class GraphConstructor:
    """Constructs PyTorch Geometric graphs from KG data."""
    
    def __init__(self, kg_connector):
        self.kg = kg_connector
        self.node_types = ['hv_station', 'mv_station', 'lv_line', 'building']
        self.edge_types = [
            ('hv_station', 'feeds', 'mv_station'),
            ('mv_station', 'feeds', 'lv_line'),
            ('lv_line', 'supplies', 'building'),
            ('building', 'shares_lv_with', 'building'),
        ]
        self.node_mappings = {}
    
    def build_hetero_graph(self, region_id: str, include_energy_sharing: bool = True) -> HeteroData:
        """Build complete heterogeneous graph for a region."""
        logger.info(f"Building graph for region {region_id}")
        
        topology = self.kg.get_grid_topology(region_id)
        graph = HeteroData()
        
        self._add_nodes_to_graph(graph, topology['nodes'], region_id)
        self._add_edges_to_graph(graph, topology['edges'])
        
        if include_energy_sharing:
            self._add_energy_sharing_edges(graph, topology)
        
        graph.region_id = region_id
        graph.num_nodes_dict = {
            node_type: graph[node_type].x.shape[0] 
            for node_type in self.node_types if node_type in graph.node_types
        }
        
        return graph
    
    def build_subgraph_for_task(self, region_id: str, task_type: str, **kwargs) -> HeteroData:
        """Build task-specific subgraph."""
        if task_type == 'retrofit':
            return self._build_retrofit_graph(region_id, **kwargs)
        elif task_type == 'energy_sharing':
            return self._build_energy_sharing_graph(region_id, **kwargs)
        elif task_type == 'solar':
            return self._build_solar_graph(region_id, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
    
    def _add_nodes_to_graph(self, graph: HeteroData, nodes: Dict[str, List], region_id: str):
        """Add nodes with features to graph."""
        
        # HV stations
        hv_stations = nodes.get('hv_stations', [])
        if hv_stations:
            hv_features = []
            hv_ids = []
            for i, hv in enumerate(hv_stations):
                hv_ids.append(hv.get('station_id', f'HV_{i}'))
                hv_features.append([
                    hv.get('capacity', 50000),
                    0.75,  # load_factor
                    150  # voltage_level
                ])
            graph['hv_station'].x = torch.tensor(hv_features, dtype=torch.float)
            # Store ids as tensor of indices instead of list
            graph['hv_station'].node_ids = torch.arange(len(hv_ids))
            # Store the actual IDs mapping separately
            self.hv_ids = hv_ids
            self.node_mappings['hv_station'] = {id_: i for i, id_ in enumerate(hv_ids)}
        
        # MV stations  
        mv_stations = nodes.get('mv_stations', [])
        if mv_stations:
            mv_features = []
            mv_ids = []
            for i, mv in enumerate(mv_stations):
                mv_ids.append(mv.get('station_id', f'MV_{i}'))
                mv_features.append([
                    mv.get('capacity', 10000),
                    0.7,  # load_factor
                    10,   # voltage_level
                    2     # transformer_count
                ])
            graph['mv_station'].x = torch.tensor(mv_features, dtype=torch.float)
            # Store ids as tensor of indices instead of list
            graph['mv_station'].node_ids = torch.arange(len(mv_ids))
            # Store the actual IDs mapping separately
            self.mv_ids = mv_ids
            self.node_mappings['mv_station'] = {id_: i for i, id_ in enumerate(mv_ids)}
        
        # LV lines
        lv_lines = nodes.get('lv_lines', [])
        if lv_lines:
            lv_features = []
            lv_ids = []
            lv_aggregates = {}
            
            for i, lv in enumerate(lv_lines):
                lv_id = lv.get('line_id', f'LV_{i}')
                lv_ids.append(lv_id)
                agg = self.kg.aggregate_to_lv_level(lv_id)
                lv_aggregates[lv_id] = agg
                
                lv_features.append([
                    lv.get('capacity', 2000),
                    agg.get('load_factor', 0.5),
                    agg.get('building_count', 0),
                    agg.get('avg_energy_score', 4),
                    agg.get('total_floor_area', 0),
                    agg.get('total_roof_area', 0)
                ])
            
            graph['lv_line'].x = torch.tensor(lv_features, dtype=torch.float)
            # Store ids as tensor of indices instead of list
            graph['lv_line'].node_ids = torch.arange(len(lv_ids))
            # Store the actual IDs mapping separately
            self.lv_ids = lv_ids
            # Don't store aggregates as node attribute (it's a dict)
            self.lv_aggregates = lv_aggregates  # Store on self instead
            self.node_mappings['lv_line'] = {id_: i for i, id_ in enumerate(lv_ids)}
        
        # Buildings
        buildings = nodes.get('buildings', [])
        if buildings:
            building_features = []
            building_ids = []
            
            all_building_ids = [b.get('building_id') for b in buildings]
            ubem_df = self.kg.get_ubem_results(all_building_ids)
            ubem_dict = ubem_df.set_index('building_id').to_dict('index') if not ubem_df.empty else {}
            
            for i, building in enumerate(buildings):
                b_id = building.get('building_id', f'B_{i}')
                building_ids.append(b_id)
                ubem = ubem_dict.get(b_id, {})
                
                # Mock some building features for testing
                building_features.append([
                    150 + i * 10,  # floor_area
                    4,  # energy_score  
                    0,  # building_type
                    30 + i,  # age
                    100 + i * 5,  # roof_area
                    10 + i,  # height
                    4,  # occupants
                    ubem.get('electricity_demand', 5000),
                    ubem.get('heating_demand', 8000),
                    ubem.get('peak_load', 10)
                ])
            
            graph['building'].x = torch.tensor(building_features, dtype=torch.float)
            # Store ids as tensor of indices instead of list
            graph['building'].node_ids = torch.arange(len(building_ids))
            # Store the actual IDs mapping separately
            self.building_ids = building_ids
            self.node_mappings['building'] = {id_: i for i, id_ in enumerate(building_ids)}
    
    def _add_edges_to_graph(self, graph: HeteroData, edges: Dict[str, List]):
        """Add edges to graph."""
        
        # HV to MV
        if edges.get('hv_to_mv'):
            edge_index = self._create_edge_index(edges['hv_to_mv'], 'hv_station', 'mv_station')
            if edge_index is not None:
                graph['hv_station', 'feeds', 'mv_station'].edge_index = edge_index
        
        # MV to LV
        if edges.get('mv_to_lv'):
            edge_index = self._create_edge_index(edges['mv_to_lv'], 'mv_station', 'lv_line')
            if edge_index is not None:
                graph['mv_station', 'feeds', 'lv_line'].edge_index = edge_index
        
        # LV to Building
        if edges.get('lv_to_building'):
            edge_index = self._create_edge_index(edges['lv_to_building'], 'lv_line', 'building')
            if edge_index is not None:
                graph['lv_line', 'supplies', 'building'].edge_index = edge_index
    
    def _add_energy_sharing_edges(self, graph: HeteroData, topology: Dict):
        """Add edges between buildings on same LV line."""
        lv_to_buildings = {}
        for edge in topology['edges'].get('lv_to_building', []):
            lv_id = edge['src']
            building_id = edge['dst']
            if lv_id not in lv_to_buildings:
                lv_to_buildings[lv_id] = []
            lv_to_buildings[lv_id].append(building_id)
        
        sharing_edges = []
        for lv_id, building_ids in lv_to_buildings.items():
            for i, b1 in enumerate(building_ids):
                for b2 in building_ids[i+1:]:
                    if b1 in self.node_mappings.get('building', {}) and b2 in self.node_mappings.get('building', {}):
                        idx1 = self.node_mappings['building'][b1]
                        idx2 = self.node_mappings['building'][b2]
                        sharing_edges.append([idx1, idx2])
                        sharing_edges.append([idx2, idx1])
        
        if sharing_edges:
            edge_index = torch.tensor(sharing_edges, dtype=torch.long).t()
            graph['building', 'shares_lv_with', 'building'].edge_index = edge_index
    
    def _create_edge_index(self, edge_list: List[Dict], src_type: str, dst_type: str) -> Optional[torch.Tensor]:
        """Convert edge list to PyG edge_index tensor."""
        if not edge_list:
            return None
        
        edges = []
        src_mapping = self.node_mappings.get(src_type, {})
        dst_mapping = self.node_mappings.get(dst_type, {})
        
        for edge in edge_list:
            src_id = edge['src']
            dst_id = edge['dst']
            if src_id in src_mapping and dst_id in dst_mapping:
                edges.append([src_mapping[src_id], dst_mapping[dst_id]])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        return None
    
    def _build_retrofit_graph(self, region_id: str, min_age: int = 30, max_label: str = 'D') -> HeteroData:
        """Build graph for retrofit targeting task."""
        candidates = self.kg.get_retrofit_candidates(region_id, min_age, max_label)
        graph = self.build_hetero_graph(region_id)
        
        # Use the stored building_ids from self
        building_ids = self.building_ids if hasattr(self, 'building_ids') else []
        retrofit_labels = []
        
        for b_id in building_ids:
            is_candidate = any(
                any(b.get('building_id') == b_id for b in buildings)
                for buildings in candidates.values()
            )
            retrofit_labels.append(1 if is_candidate else 0)
        
        graph['building'].y = torch.tensor(retrofit_labels, dtype=torch.float)
        graph.task = 'retrofit'
        return graph
    
    def _build_energy_sharing_graph(self, region_id: str, min_cluster_size: int = 5) -> HeteroData:
        """Build graph for energy sharing analysis."""
        clusters = self.kg.get_energy_sharing_clusters(region_id, min_cluster_size)
        graph = self.build_hetero_graph(region_id, include_energy_sharing=True)
        
        # Use the stored lv_ids from self
        lv_ids = self.lv_ids if hasattr(self, 'lv_ids') else []
        sharing_scores = []
        
        for lv_id in lv_ids:
            cluster_data = next((c for c in clusters if c['lv_id'] == lv_id), None)
            if cluster_data:
                solar_potential = cluster_data.get('solar_potential_kwh', 0)
                total_demand = cluster_data.get('total_elec_demand', 1)
                score = min(solar_potential / max(total_demand, 1), 1.0)
                sharing_scores.append(score)
            else:
                sharing_scores.append(0.0)
        
        graph['lv_line'].y = torch.tensor(sharing_scores, dtype=torch.float)
        graph.task = 'energy_sharing'
        return graph
    
    def _build_solar_graph(self, region_id: str) -> HeteroData:
        """Build graph for solar optimization."""
        graph = self.build_hetero_graph(region_id)
        building_features = graph['building'].x
        roof_areas = building_features[:, 4]
        solar_potential = roof_areas * 0.15 * 1000
        graph['building'].y = solar_potential
        graph.task = 'solar'
        return graph


# ============================================================================
# FEATURE PROCESSOR (Simplified Version)
# ============================================================================

class FeatureProcessor:
    """Processes and engineers features for GNN."""
    
    def __init__(self):
        self.scalers = {
            'building': StandardScaler(),
            'lv_line': StandardScaler(),
            'mv_station': StandardScaler(),
            'hv_station': StandardScaler(),
            'energy': MinMaxScaler()
        }
        self.feature_stats = {}
    
    def process_graph_features(self, graph, fit: bool = True) -> None:
        """Process all features in a HeteroData graph in-place."""
        logger.info("Processing graph features")
        
        for node_type in ['building', 'lv_line', 'mv_station', 'hv_station']:
            if node_type in graph.node_types:
                features = graph[node_type].x
                processed = self.process_node_features(features, node_type, fit=fit)
                graph[node_type].x = processed
                
                engineered = self.engineer_node_features(processed, node_type, graph)
                if engineered is not None:
                    graph[node_type].x_engineered = engineered
    
    def process_node_features(self, features: torch.Tensor, node_type: str, fit: bool = True) -> torch.Tensor:
        """Process features for a specific node type."""
        features_np = features.numpy() if isinstance(features, torch.Tensor) else features
        
        if node_type == 'building':
            processed = self._process_building_features(features_np, fit)
        else:
            # Simple normalization for other node types
            if fit:
                self.scalers[node_type].fit(features_np)
            processed = self.scalers[node_type].transform(features_np)
        
        return torch.tensor(processed, dtype=torch.float)
    
    def _process_building_features(self, features: np.ndarray, fit: bool = True) -> np.ndarray:
        """Process building features with one-hot encoding."""
        processed = features.copy()
        
        # Normalize continuous features
        continuous_indices = [0, 3, 4, 5, 6, 9]
        if fit:
            self.scalers['building'].fit(processed[:, continuous_indices])
        processed[:, continuous_indices] = self.scalers['building'].transform(processed[:, continuous_indices])
        
        # Normalize energy demands
        energy_indices = [7, 8]
        if fit:
            self.scalers['energy'].fit(processed[:, energy_indices])
        processed[:, energy_indices] = self.scalers['energy'].transform(processed[:, energy_indices])
        
        # Energy score normalization
        processed[:, 1] = (processed[:, 1] - 1) / 6
        
        # Building type one-hot encoding
        n_types = 7
        building_types = processed[:, 2].astype(int)
        type_one_hot = np.zeros((len(building_types), n_types))
        type_one_hot[np.arange(len(building_types)), building_types] = 1
        
        # Combine features
        processed = np.hstack([
            processed[:, [0, 1]],
            type_one_hot,
            processed[:, 3:]
        ])
        
        return processed
    
    def engineer_node_features(self, features: torch.Tensor, node_type: str, graph) -> Optional[torch.Tensor]:
        """Create engineered features for nodes."""
        if node_type == 'building':
            return self._engineer_building_features(features)
        elif node_type == 'lv_line':
            return self._engineer_lv_features(features)
        return None
    
    def _engineer_building_features(self, features: torch.Tensor) -> torch.Tensor:
        """Engineer building-level features."""
        features_np = features.numpy()
        engineered = []
        
        # Energy intensity
        area_norm = features_np[:, 0]
        elec_demand_norm = features_np[:, -3]
        heat_demand_norm = features_np[:, -2]
        energy_intensity = (elec_demand_norm + heat_demand_norm) / (area_norm + 1e-6)
        engineered.append(energy_intensity.reshape(-1, 1))
        
        # Retrofit score
        energy_score_norm = features_np[:, 1]
        age_norm = features_np[:, 9] if features_np.shape[1] > 9 else features_np[:, 3]
        retrofit_score = (1 - energy_score_norm) * 0.6 + age_norm * 0.4
        engineered.append(retrofit_score.reshape(-1, 1))
        
        # Solar suitability
        roof_area_norm = features_np[:, 10] if features_np.shape[1] > 10 else features_np[:, 4]
        solar_suit = roof_area_norm
        engineered.append(solar_suit.reshape(-1, 1))
        
        # Electrification readiness
        electrif_ready = heat_demand_norm * energy_score_norm
        engineered.append(electrif_ready.reshape(-1, 1))
        
        # Peak ratio
        peak_norm = features_np[:, -1]
        avg_demand = (elec_demand_norm + heat_demand_norm) / 2
        peak_ratio = peak_norm / (avg_demand + 1e-6)
        engineered.append(peak_ratio.reshape(-1, 1))
        
        return torch.tensor(np.hstack(engineered), dtype=torch.float)
    
    def _engineer_lv_features(self, features: torch.Tensor) -> torch.Tensor:
        """Engineer LV-level features."""
        features_np = features.numpy()
        engineered = []
        
        # Diversity factor
        load_factor = features_np[:, 1]
        building_count_norm = features_np[:, 2]
        diversity = load_factor / (building_count_norm + 1e-6)
        engineered.append(diversity.reshape(-1, 1))
        
        # Self-sufficiency potential
        roof_norm = features_np[:, 5]
        floor_norm = features_np[:, 4]
        self_suff = roof_norm / (floor_norm + 1e-6)
        engineered.append(self_suff.reshape(-1, 1))
        
        # Grid stress
        capacity_norm = features_np[:, 0]
        stress = load_factor / (capacity_norm + 1e-6)
        engineered.append(stress.reshape(-1, 1))
        
        return torch.tensor(np.hstack(engineered), dtype=torch.float)
    
    def create_task_specific_features(self, graph, task: str) -> Dict[str, torch.Tensor]:
        """Create features specific to a task."""
        task_features = {}
        
        if task == 'retrofit' and 'building' in graph.node_types:
            if hasattr(graph['building'], 'x_engineered'):
                task_features['retrofit_priority'] = graph['building'].x_engineered[:, 1]
                area = graph['building'].x[:, 0]
                task_features['retrofit_cost'] = area * 500
        
        elif task == 'energy_sharing' and 'lv_line' in graph.node_types:
            if hasattr(graph['lv_line'], 'x_engineered'):
                task_features['self_sufficiency'] = graph['lv_line'].x_engineered[:, 1]
                task_features['diversity_factor'] = graph['lv_line'].x_engineered[:, 0]
        
        elif task == 'solar' and 'building' in graph.node_types:
            if hasattr(graph['building'], 'x_engineered'):
                task_features['solar_suitability'] = graph['building'].x_engineered[:, 2]
                roof_area = graph['building'].x[:, 10] if graph['building'].x.shape[1] > 10 else graph['building'].x[:, 4]
                task_features['solar_generation'] = roof_area * 0.15 * 1000
        
        # Ensure all outputs are tensors
        for key in list(task_features.keys()):
            if not isinstance(task_features[key], torch.Tensor):
                task_features[key] = torch.tensor(task_features[key], dtype=torch.float)
        
        return task_features


# ============================================================================
# DATA LOADER (Simplified Version)
# ============================================================================

class TaskSpecificLoader:
    """Creates task-specific data loaders with custom sampling strategies."""
    
    def __init__(self, batch_size: int = 32, num_neighbors: List[int] = [15, 10], num_workers: int = 0):
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.num_workers = num_workers
    
    def create_loader(self, graph: HeteroData, task: str, split: str = 'train'):
        """Create a simple task-specific loader."""
        logger.info(f"Creating {task} loader for {split}")
        
        # Determine target node type
        target_types = {
            'retrofit': 'building',
            'energy_sharing': 'lv_line', 
            'solar': 'building',
            'grid_planning': 'mv_station'
        }
        
        target_type = target_types.get(task, 'building')
        num_nodes = graph[target_type].x.shape[0]
        
        # Create or use existing masks
        if hasattr(graph[target_type], f'{split}_mask'):
            input_nodes = torch.where(graph[target_type][f'{split}_mask'])[0]
        else:
            # Simple split for testing
            if split == 'train':
                input_nodes = torch.arange(0, int(num_nodes * 0.7))
            elif split == 'val':
                input_nodes = torch.arange(int(num_nodes * 0.7), int(num_nodes * 0.85))
            else:
                input_nodes = torch.arange(int(num_nodes * 0.85), num_nodes)
        
        # Ensure input_nodes is not empty
        if len(input_nodes) == 0:
            # If no nodes in this split, use at least one node
            input_nodes = torch.tensor([0])
        
        # Create basic NeighborLoader with consistent hop counts
        num_neighbors_dict = {}
        for edge_type in graph.edge_types:
            if task == 'energy_sharing' and 'supplies' in str(edge_type):
                # For energy sharing, get all connected buildings (but same hop count)
                num_neighbors_dict[edge_type] = [-1, -1]  # 2 hops, all neighbors
            else:
                num_neighbors_dict[edge_type] = self.num_neighbors  # Default [15, 10]
        
        loader = NeighborLoader(
            graph,
            num_neighbors=num_neighbors_dict,
            batch_size=self.batch_size,
            input_nodes=(target_type, input_nodes),
            shuffle=(split == 'train'),
            num_workers=self.num_workers
        )
        
        return loader


def create_train_val_test_loaders(graph: HeteroData, task: str, batch_size: int = 32):
    """Create train, validation, and test loaders."""
    target_types = {
        'retrofit': 'building',
        'energy_sharing': 'lv_line',
        'solar': 'building', 
        'grid_planning': 'mv_station'
    }
    
    target_type = target_types.get(task, 'building')
    num_nodes = graph[target_type].x.shape[0]
    
    # Create splits
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * 0.7)
    val_size = int(num_nodes * 0.15)
    
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


# ============================================================================
# MOCK KG CONNECTOR
# ============================================================================

class MockKGConnector:
    """Mock KG connector for testing without Neo4j."""
    
    def __init__(self):
        self.region_id = "Amsterdam_North"
        logger.info("Using MockKGConnector for testing")
    
    def verify_connection(self) -> bool:
        return True
    
    def get_grid_hierarchy(self, region_id: str) -> Dict[str, Any]:
        """Return mock grid hierarchy."""
        return {
            'hv_station': {'station_id': 'HV001', 'region_id': region_id, 'capacity': 50000},
            'mv_stations': [
                {
                    'station': {'station_id': 'MV001', 'capacity': 10000},
                    'lv_lines': [
                        {
                            'line': {'line_id': 'LV001', 'capacity': 2000},
                            'buildings': [{'building_id': f'B00{i}'} for i in range(5)]
                        },
                        {
                            'line': {'line_id': 'LV002', 'capacity': 1800},
                            'buildings': [{'building_id': f'B01{i}'} for i in range(4)]
                        }
                    ]
                },
                {
                    'station': {'station_id': 'MV002', 'capacity': 8000},
                    'lv_lines': [
                        {
                            'line': {'line_id': 'LV003', 'capacity': 1500},
                            'buildings': [{'building_id': f'B02{i}'} for i in range(3)]
                        }
                    ]
                }
            ]
        }
    
    def get_grid_topology(self, region_id: str) -> Dict[str, Any]:
        """Return mock topology."""
        return {
            'nodes': {
                'hv_stations': [{'station_id': 'HV001', 'capacity': 50000}],
                'mv_stations': [
                    {'station_id': 'MV001', 'capacity': 10000},
                    {'station_id': 'MV002', 'capacity': 8000}
                ],
                'lv_lines': [
                    {'line_id': 'LV001', 'capacity': 2000},
                    {'line_id': 'LV002', 'capacity': 1800},
                    {'line_id': 'LV003', 'capacity': 1500}
                ],
                'buildings': [{'building_id': f'B{i:03d}'} for i in range(12)]
            },
            'edges': {
                'hv_to_mv': [
                    {'src': 'HV001', 'dst': 'MV001'},
                    {'src': 'HV001', 'dst': 'MV002'}
                ],
                'mv_to_lv': [
                    {'src': 'MV001', 'dst': 'LV001'},
                    {'src': 'MV001', 'dst': 'LV002'},
                    {'src': 'MV002', 'dst': 'LV003'}
                ],
                'lv_to_building': [
                    {'src': 'LV001', 'dst': f'B00{i}'} for i in range(5)
                ] + [
                    {'src': 'LV002', 'dst': f'B01{i}'} for i in range(4)
                ] + [
                    {'src': 'LV003', 'dst': f'B02{i}'} for i in range(3)
                ]
            }
        }
    
    def get_ubem_results(self, building_ids: list) -> pd.DataFrame:
        """Return mock UBEM results."""
        data = []
        for b_id in building_ids:
            idx = int(b_id[1:]) if b_id[1:].isdigit() else 0
            data.append({
                'building_id': b_id,
                'heating_demand': 8000 + idx * 500,
                'electricity_demand': 5000 + idx * 300,
                'peak_load': 10 + idx * 0.5
            })
        return pd.DataFrame(data)
    
    def aggregate_to_lv_level(self, lv_line_id: str) -> Dict[str, Any]:
        """Return mock LV aggregation."""
        lv_data = {
            'LV001': {'building_count': 5, 'avg_energy_score': 4.2, 'load_factor': 0.68,
                     'total_floor_area': 850, 'total_roof_area': 550},
            'LV002': {'building_count': 4, 'avg_energy_score': 3.0, 'load_factor': 0.62,
                     'total_floor_area': 860, 'total_roof_area': 512},
            'LV003': {'building_count': 3, 'avg_energy_score': 1.5, 'load_factor': 0.55,
                     'total_floor_area': 600, 'total_roof_area': 480}
        }
        return lv_data.get(lv_line_id, {'building_count': 0, 'avg_energy_score': 4, 
                                         'load_factor': 0.5, 'total_floor_area': 0, 
                                         'total_roof_area': 0})
    
    def get_retrofit_candidates(self, region_id: str, min_age: int = 30, max_label: str = 'D'):
        """Return mock retrofit candidates."""
        return {
            'LV002': [{'building_id': 'B010'}, {'building_id': 'B011'}],
            'LV003': [{'building_id': 'B020'}, {'building_id': 'B021'}, {'building_id': 'B022'}]
        }
    
    def get_energy_sharing_clusters(self, region_id: str, min_cluster_size: int = 3):
        """Return mock energy sharing clusters."""
        return [
            {'lv_id': 'LV001', 'building_count': 5, 'solar_potential_kwh': 82500,
             'total_elec_demand': 27500},
            {'lv_id': 'LV002', 'building_count': 4, 'solar_potential_kwh': 76800,
             'total_elec_demand': 22400}
        ]
    
    def close(self):
        logger.info("MockKGConnector closed")


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_full_pipeline():
    """Test the complete data pipeline."""
    print("\n" + "="*80)
    print("COMPLETE DATA PIPELINE TEST (CONSOLIDATED)")
    print("="*80)
    
    try:
        # Step 1: KG Connection
        print("\n" + "="*80)
        print("TESTING KG CONNECTOR")
        print("="*80)
        
        kg = MockKGConnector()
        assert kg.verify_connection(), "Connection should be verified"
        print("✓ Connection verified")
        
        hierarchy = kg.get_grid_hierarchy("Amsterdam_North")
        print(f"✓ Grid Hierarchy Retrieved:")
        print(f"  - HV Station: {hierarchy['hv_station']['station_id']}")
        print(f"  - MV Stations: {len(hierarchy['mv_stations'])}")
        
        retrofit = kg.get_retrofit_candidates("Amsterdam_North")
        print(f"✓ Retrofit Candidates: {sum(len(b) for b in retrofit.values())} buildings")
        
        clusters = kg.get_energy_sharing_clusters("Amsterdam_North")
        print(f"✓ Energy Sharing Clusters: {len(clusters)} clusters identified")
        
        # Step 2: Graph Construction
        print("\n" + "="*80)
        print("TESTING GRAPH CONSTRUCTOR")
        print("="*80)
        
        constructor = GraphConstructor(kg)
        graph = constructor.build_hetero_graph("Amsterdam_North", include_energy_sharing=True)
        
        print(f"✓ Graph Structure:")
        print(f"  Node types: {graph.node_types}")
        print(f"  Edge types: {[str(e) for e in graph.edge_types][:2]}...")
        
        print(f"✓ Node Counts:")
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'x'):
                print(f"  - {node_type}: {graph[node_type].x.shape[0]} nodes, "
                      f"{graph[node_type].x.shape[1]} features")
        
        # Test task-specific graphs
        retrofit_graph = constructor.build_subgraph_for_task("Amsterdam_North", "retrofit")
        sharing_graph = constructor.build_subgraph_for_task("Amsterdam_North", "energy_sharing")
        solar_graph = constructor.build_subgraph_for_task("Amsterdam_North", "solar")
        
        print(f"✓ Task-Specific Graphs Created:")
        print(f"  - Retrofit: {retrofit_graph['building'].y.sum().item():.0f} candidates")
        print(f"  - Energy Sharing: Avg score {sharing_graph['lv_line'].y.mean().item():.3f}")
        print(f"  - Solar: Total potential {solar_graph['building'].y.sum().item():.0f} kWh")
        
        # Step 3: Feature Processing
        print("\n" + "="*80)
        print("TESTING FEATURE PROCESSOR")
        print("="*80)
        
        processor = FeatureProcessor()
        processor.process_graph_features(graph, fit=True)
        
        print(f"✓ Processed Feature Shapes:")
        for node_type in graph.node_types:
            if hasattr(graph[node_type], 'x'):
                print(f"  - {node_type}: {graph[node_type].x.shape}")
                if hasattr(graph[node_type], 'x_engineered'):
                    print(f"    Engineered: {graph[node_type].x_engineered.shape}")
        
        task_features = processor.create_task_specific_features(graph, 'retrofit')
        print(f"✓ Task-Specific Features: {list(task_features.keys())}")
        
        # Step 4: Data Loading
        print("\n" + "="*80)
        print("TESTING DATA LOADER")
        print("="*80)
        
        for task in ['retrofit', 'energy_sharing', 'solar']:
            train_loader, val_loader, test_loader = create_train_val_test_loaders(
                graph, task, batch_size=4
            )
            
            # Test one batch
            for batch in train_loader:
                print(f"✓ {task.upper()} loader: Batch with {len(batch.node_types)} node types")
                break
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_full_pipeline()