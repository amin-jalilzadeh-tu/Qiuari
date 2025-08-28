# models/base_gnn.py
"""
Base heterogeneous GNN architecture for energy systems with multi-task support
Complete implementation with task heads and model factory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, HeteroData, Batch
from typing import Dict, Tuple, Optional, Any, Union, List
import logging
from models.pooling_layers import ConstrainedDiffPool
from models.temporal_layers_integrated import IntegratedTemporalProcessor

logger = logging.getLogger(__name__)


# ============================================
# ENCODER MODULES
# ============================================

class BuildingEncoder(nn.Module):
    """Encodes building features into embeddings"""
    
    def __init__(self, input_dim: Optional[int] = None, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim  # Store hidden_dim as instance variable

        # Auto-detect input dimension if not provided
        if input_dim is None:
            logger.warning("Input dimension not specified, will be set on first forward pass")
            self.input_dim = None
            self.input_projection = None  # Will be created dynamically
        else:
            self.input_dim = input_dim
            self._create_encoders(input_dim)
    
    def _create_encoders(self, input_dim: int):
        """Create encoders with known dimension"""
        self.input_dim = input_dim
            
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),  # Use self.hidden_dim
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor):
        # Create encoder on first forward pass if needed
        if self.input_projection is None:
            self._create_encoders(x.shape[-1])
            # Move to same device as input
            self.input_projection = self.input_projection.to(x.device)
            logger.info(f"Auto-detected BuildingEncoder input dimension: {x.shape[-1]}")
        return self.input_projection(x)


class LVGroupEncoder(nn.Module):
    """Encodes LV cable group features"""
    
    def __init__(self, input_dim: int = 12, hidden_dim: int = 128):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor):
        return self.input_projection(x)


class TransformerEncoder(nn.Module):
    """Encodes transformer features"""
    
    def __init__(self, input_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor):
        return self.input_projection(x)


class AdjacencyClusterEncoder(nn.Module):
    """Encodes adjacency cluster features"""
    
    def __init__(self, input_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor):
        return self.input_projection(x)


# ============================================
# MESSAGE PASSING MODULES
# ============================================

class HierarchicalMessagePassing(nn.Module):
    """Handles message passing through grid hierarchy"""
    
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4):
        super().__init__()
        
        # Bottom-up: Building -> LV Group
        self.building_to_lv = GATConv(
            hidden_dim, hidden_dim // num_heads,
            heads=num_heads, concat=True, dropout=0.1, add_self_loops=False
        )
        
        # Bottom-up: LV Group -> Transformer
        self.lv_to_transformer = GATConv(
            hidden_dim, hidden_dim // num_heads,
            heads=num_heads, concat=True, dropout=0.1, add_self_loops=False
        )
        
        # Top-down: Transformer -> LV Group
        self.transformer_to_lv = SAGEConv(hidden_dim, hidden_dim)
        
        # Top-down: LV Group -> Building
        self.lv_to_building = SAGEConv(hidden_dim, hidden_dim)
        
        # Lateral: Building <-> Building (adjacency)
        self.building_adjacency = GCNConv(hidden_dim, hidden_dim, add_self_loops=True)
        
    def forward(self, x_dict: Dict, edge_index_dict: Dict):
        """Perform hierarchical message passing"""
        h_dict = x_dict.copy()
        
        # Bottom-up: Buildings -> LV Groups
        if ('building', 'connected_to', 'cable_group') in edge_index_dict:
            edge_index = edge_index_dict[('building', 'connected_to', 'cable_group')]
            if edge_index.shape[1] > 0 and 'building' in h_dict and 'cable_group' in h_dict:
                h_lv_from_buildings = self.building_to_lv(
                    (h_dict['building'], h_dict['cable_group']),
                    edge_index
                )
                h_dict['cable_group'] = h_dict['cable_group'] + 0.5 * h_lv_from_buildings
        
        # Bottom-up: LV Groups -> Transformers
        if ('cable_group', 'connects_to', 'transformer') in edge_index_dict:
            edge_index = edge_index_dict[('cable_group', 'connects_to', 'transformer')]
            if edge_index.shape[1] > 0 and 'cable_group' in h_dict and 'transformer' in h_dict:
                h_trans_from_lv = self.lv_to_transformer(
                    (h_dict['cable_group'], h_dict['transformer']),
                    edge_index
                )
                h_dict['transformer'] = h_dict['transformer'] + 0.5 * h_trans_from_lv
        
        # Lateral: Building adjacency
        if ('building', 'adjacent_to', 'building') in edge_index_dict:
            edge_index = edge_index_dict[('building', 'adjacent_to', 'building')]
            if edge_index.shape[1] > 0 and 'building' in h_dict:
                h_building_adj = self.building_adjacency(
                    h_dict['building'],
                    edge_index
                )
                h_dict['building'] = h_dict['building'] + 0.2 * h_building_adj
        
        return h_dict


# ============================================
# TASK-SPECIFIC OUTPUT HEADS
# ============================================

class TaskHeads(nn.Module):
    """Multi-task output heads for different objectives"""
    
    def __init__(self, hidden_dim: int = 128, config: Dict = None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.config = config or {}
        
        # Clustering head (community formation)
        self.clustering_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, config.get('num_clusters', 10))
        )
        
        # Retrofit head (priority scoring)
        self.retrofit_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Retrofit priority score
        )
        
        # Solar optimization head (placement ranking)
        self.solar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Solar suitability score
        )
        
        # Electrification head (feasibility classification)
        self.electrification_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 3)  # Classes: infeasible, partial, full
        )
        
        # Battery placement head
        self.battery_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2)  # Battery size recommendation
        )
        
        # Congestion prediction head (for cable groups)
        self.congestion_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)  # Congestion level
        )
        
        # Thermal sharing head (for adjacent buildings)
        self.thermal_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Concatenated pair features
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)  # Thermal sharing potential
        )
        
        # Temporal processor (if enabled)
        self.use_temporal = config.get('use_temporal', False)
        if self.use_temporal:
            self.temporal_processor = IntegratedTemporalProcessor(config)
            logger.info("Temporal processing enabled with pattern extraction and complementarity scoring")
        
        # Store temporal outputs for loss calculation
        self.temporal_outputs = None
        
    def forward(self, embeddings: Union[torch.Tensor, Dict], task: str, 
                edge_index: Optional[torch.Tensor] = None):
        """
        Forward pass for specific task
        
        Args:
            embeddings: Node embeddings (tensor or dict by node type)
            task: Task name
            edge_index: Optional edge indices for pair-wise tasks
            
        Returns:
            Task-specific output
        """
        # Handle both tensor and dict inputs
        if isinstance(embeddings, dict):
            # Extract appropriate embeddings based on task
            if task in ['clustering', 'retrofit', 'solar', 'electrification', 'battery']:
                x = embeddings.get('building', embeddings.get('x'))
            elif task == 'congestion':
                x = embeddings.get('cable_group', embeddings.get('x'))
            elif task == 'thermal':
                x = embeddings.get('building', embeddings.get('x'))
            else:
                x = embeddings.get('building', embeddings.get('x'))
        else:
            x = embeddings
        
        # Apply task-specific head - ALWAYS return dict for consistency
        if task == 'clustering':
            return {'cluster_logits': self.clustering_head(x)}
        elif task == 'retrofit':
            return {'retrofit_scores': self.retrofit_head(x)}
        elif task == 'solar':
            # Return proper solar ROI classification logits
            solar_scores = self.solar_head(x)
            # Expand to 4-class logits using learnable transformation
            # This maintains gradient flow
            scores_expanded = solar_scores.expand(-1, 4)
            # Create differentiable logits using continuous functions
            scores_norm = torch.sigmoid(solar_scores)
            solar_logits = torch.cat([
                1.0 - scores_norm,  # Excellent ROI (high when score is low)
                0.5 - torch.abs(scores_norm - 0.5),  # Good ROI (high when score ~0.5)
                0.5 - torch.abs(scores_norm - 0.75),  # Fair ROI (high when score ~0.75)
                scores_norm  # Poor ROI (high when score is high)
            ], dim=1)
            return {'solar': solar_logits, 'solar_scores': solar_scores}
        elif task == 'electrification':
            return {'electrification_logits': self.electrification_head(x)}
        elif task == 'battery':
            return {'battery_recommendations': self.battery_head(x)}
        elif task == 'congestion':
            return {'congestion_scores': self.congestion_head(x)}
        elif task == 'thermal':
            # For thermal, we need pairs of adjacent buildings
            if edge_index is not None and edge_index.shape[1] > 0:
                x_i = x[edge_index[0]]
                x_j = x[edge_index[1]]
                x_pairs = torch.cat([x_i, x_j], dim=-1)
                return {'thermal_sharing': self.thermal_head(x_pairs)}
            else:
                return {'thermal_sharing': torch.zeros(x.shape[0], 1).to(x.device)}
        else:
            # Return empty dict for unknown tasks instead of error
            return {}


# ============================================
# MAIN GNN MODELS
# ============================================

class HeteroEnergyGNN(nn.Module):
    """Heterogeneous GNN for energy systems with multi-task support"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        
        # Feature dimensions from actual data - use None for auto-detection
        node_features = {
            'building': config.get('building_features'),  # None for auto-detection
            'cable_group': config.get('cable_group_features') or 12,
            'transformer': config.get('transformer_features') or 8,
            'adjacency_cluster': config.get('cluster_features') or 11
        }
        
        # Node encoders
        self.encoders = nn.ModuleDict({
            'building': BuildingEncoder(node_features['building'], self.hidden_dim),
            'cable_group': LVGroupEncoder(node_features['cable_group'], self.hidden_dim),
            'transformer': TransformerEncoder(node_features['transformer'], self.hidden_dim),
            'adjacency_cluster': AdjacencyClusterEncoder(node_features['adjacency_cluster'], 64)
        })
        
        # ADD: Hierarchical positional encoding for 3-level structure
        self.pos_embed = nn.ModuleDict({
            'building': nn.Embedding(1, self.hidden_dim),      # Level 0
            'cable_group': nn.Embedding(1, self.hidden_dim),   # Level 1  
            'transformer': nn.Embedding(1, self.hidden_dim)    # Level 2
        })
        
        # ADD: Constrained DiffPool for dynamic clustering
        self.diffpool = ConstrainedDiffPool(
            input_dim=self.hidden_dim,
            max_clusters=config.get('max_clusters', 20),
            min_cluster_size=config.get('min_cluster_size', 3),
            max_cluster_size=config.get('max_cluster_size', 20)
        )
        
        # Message passing layers
        self.mp_layers = nn.ModuleList([
            HierarchicalMessagePassing(self.hidden_dim)
            for _ in range(self.num_layers)
        ])
        
        # Layer normalization
        self.layer_norms = nn.ModuleDict({
            'building': nn.LayerNorm(self.hidden_dim),
            'cable_group': nn.LayerNorm(self.hidden_dim),
            'transformer': nn.LayerNorm(self.hidden_dim),
            'adjacency_cluster': nn.LayerNorm(64)
        })
        
        # Task heads
        self.task_heads = TaskHeads(self.hidden_dim, config)
        
        # ADD: New discovery heads
        from models.task_heads import ComplementarityScoreHead, NetworkCentralityHead, EnergyFlowHead
        self.complementarity_head = ComplementarityScoreHead(self.hidden_dim)
        self.centrality_head = NetworkCentralityHead(self.hidden_dim)
        self.flow_head = EnergyFlowHead(self.hidden_dim)
        
        # Attention module (optional)
        if config.get('use_attention', True):
            from models.attention_layers import ComplementarityAttention
            self.attention = ComplementarityAttention(
                in_channels=self.hidden_dim,
                out_channels=self.hidden_dim
            )
        else:
            self.attention = None
        
        # Temporal processor (if enabled)
        self.use_temporal = config.get('use_temporal', False)
        if self.use_temporal:
            self.temporal_processor = IntegratedTemporalProcessor(config)
            logger.info("Temporal processing enabled with pattern extraction and complementarity scoring")
        
        # Store temporal outputs for loss calculation
        self.temporal_outputs = None
        
        logger.info(f"Initialized HeteroEnergyGNN with {self.num_layers} layers and positional encoding")
    
    def forward(self, data: Union[Data, HeteroData, Dict], task: str = None):
        """
        Forward pass through the GNN
        
        Args:
            data: Input data (can be Data, HeteroData, or dict)
            task: Task name for task-specific output
            
        Returns:
            Task-specific output or embeddings
        """
        # Handle different input formats
        if isinstance(data, HeteroData):
            x_dict = {ntype: data[ntype].x for ntype in data.node_types if hasattr(data[ntype], 'x')}
            edge_index_dict = {etype: data[etype].edge_index for etype in data.edge_types}
        elif isinstance(data, Data):
            # Convert homogeneous to hetero format
            x_dict = {'building': data.x}
            edge_index_dict = {('building', 'connected', 'building'): data.edge_index}
        elif isinstance(data, dict):
            x_dict = data.get('x_dict', data)
            edge_index_dict = data.get('edge_index_dict', {})
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Step 1: Encode node features
        h_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.encoders:
                h_dict[node_type] = self.encoders[node_type](x)
        
        # Step 1.5: Apply hierarchical positional encoding
        if 'building' in h_dict:
            pos_tensor = torch.zeros(h_dict['building'].size(0), dtype=torch.long, device=h_dict['building'].device)
            h_dict['building'] = h_dict['building'] + self.pos_embed['building'](pos_tensor)
        if 'cable_group' in h_dict:
            pos_tensor = torch.zeros(h_dict['cable_group'].size(0), dtype=torch.long, device=h_dict['cable_group'].device)
            h_dict['cable_group'] = h_dict['cable_group'] + self.pos_embed['cable_group'](pos_tensor)
        if 'transformer' in h_dict:
            pos_tensor = torch.zeros(h_dict['transformer'].size(0), dtype=torch.long, device=h_dict['transformer'].device)
            h_dict['transformer'] = h_dict['transformer'] + self.pos_embed['transformer'](pos_tensor)
        
        # Step 1.6: Apply temporal processing if enabled
        if self.use_temporal and 'building' in h_dict:
            # Extract temporal data if available
            temporal_data = None
            season = 0
            is_weekend = False
            current_hour = 12
            
            if isinstance(data, HeteroData):
                if hasattr(data['building'], 'x_temporal'):
                    temporal_data = data['building'].x_temporal
                elif hasattr(data['building'], 'temporal_features'):
                    temporal_data = data['building'].temporal_features
                season = data.season if hasattr(data, 'season') else 0
                is_weekend = data.is_weekend if hasattr(data, 'is_weekend') else False
                current_hour = data.current_hour if hasattr(data, 'current_hour') else 12
            elif isinstance(data, dict):
                temporal_data = data.get('temporal_features')
                season = data.get('season', 0)
                is_weekend = data.get('is_weekend', False)
                current_hour = data.get('current_hour', 12)
            
            if temporal_data is not None:
                # Process temporal features
                temporal_output = self.temporal_processor(
                    h_dict['building'],
                    temporal_data,
                    season=season,
                    is_weekend=is_weekend,
                    current_hour=current_hour
                )
                
                # Update building embeddings with temporal-aware ones
                h_dict['building'] = temporal_output['embeddings']
                
                # Store temporal outputs for loss calculation
                self.temporal_outputs = {
                    'temporal_complementarity': temporal_output['temporal_complementarity'],
                    'peak_hours': temporal_output['peak_hours'],
                    'peak_probabilities': temporal_output['peak_probabilities'],
                    'pattern_types': temporal_output['pattern_types']
                }
        
        # Step 2: Message passing
        for i, mp_layer in enumerate(self.mp_layers):
            h_dict_new = mp_layer(h_dict, edge_index_dict)
            
            # Residual connection and normalization
            for node_type in h_dict_new:
                if node_type in h_dict and node_type in self.layer_norms:
                    if node_type == 'adjacency_cluster':
                        continue  # Keep clusters static
                    
                    h_dict_new[node_type] = h_dict[node_type] + h_dict_new[node_type]
                    h_dict_new[node_type] = self.layer_norms[node_type](h_dict_new[node_type])
                    h_dict_new[node_type] = F.dropout(
                        h_dict_new[node_type], p=self.dropout, training=self.training
                    )
            
            h_dict = h_dict_new
        
        # Step 3: Apply attention (optional)
        if self.attention is not None and 'building' in h_dict:
            # Get building edges
            building_edges = edge_index_dict.get(
                ('building', 'connected', 'building'),
                torch.tensor([[0], [0]], dtype=torch.long, device=h_dict['building'].device)
            )
            # Apply attention (returns tuple: embeddings, attention_weights)
            attention_output, _ = self.attention(h_dict['building'], building_edges)
            h_dict['building'] = attention_output
        
        # Step 4: Apply DiffPool if available for clustering
        predictions = {}
        if hasattr(self, 'diffpool') and 'building' in h_dict:
            # Get building edges for pooling
            building_edges = edge_index_dict.get(
                ('building', 'connected', 'building'),
                edge_index_dict.get(('building', 'adjacent_to', 'building'), 
                                  torch.tensor([[0], [0]], dtype=torch.long, device=h_dict['building'].device))
            )
            
            # Get LV group IDs if available for boundary enforcement
            lv_group_ids = None
            if isinstance(data, HeteroData) and hasattr(data['building'], 'lv_group_ids'):
                lv_group_ids = data['building'].lv_group_ids
            elif isinstance(data, Data) and hasattr(data, 'lv_group_ids'):
                lv_group_ids = data.lv_group_ids
            elif isinstance(data, dict) and 'lv_group_ids' in data:
                lv_group_ids = data['lv_group_ids']
            
            # Apply DiffPool with LV constraints
            x_pooled, adj_pooled, S, aux_loss = self.diffpool(
                h_dict['building'], 
                building_edges,
                lv_group_ids=lv_group_ids  # Pass LV group IDs
            )
            predictions['clusters'] = S
            predictions['aux_loss'] = aux_loss
        
        # Step 5: Apply discovery heads if available
        if hasattr(self, 'complementarity_head') and 'building' in h_dict:
            building_edges = edge_index_dict.get(
                ('building', 'connected', 'building'),
                edge_index_dict.get(('building', 'adjacent_to', 'building'))
            )
            if building_edges is not None:
                predictions['complementarity'] = self.complementarity_head(
                    h_dict['building'], building_edges
                )
        
        if hasattr(self, 'centrality_head') and 'building' in h_dict:
            building_edges = edge_index_dict.get(
                ('building', 'connected', 'building'),
                edge_index_dict.get(('building', 'adjacent_to', 'building'))
            )
            if building_edges is not None:
                predictions['centrality'] = self.centrality_head(
                    h_dict['building'], building_edges, 
                    predictions.get('clusters')
                )
        
        if hasattr(self, 'flow_head') and 'building' in h_dict:
            building_edges = edge_index_dict.get(
                ('building', 'connected', 'building'),
                edge_index_dict.get(('building', 'adjacent_to', 'building'))
            )
            if building_edges is not None:
                predictions['energy_flow'] = self.flow_head(
                    h_dict['building'], building_edges
                )
        
        # Step 6: Task-specific output (legacy support)
        if task:
            adjacency_edges = edge_index_dict.get(('building', 'adjacent_to', 'building'))
            output = self.task_heads(h_dict, task, adjacency_edges)
            # TaskHeads now always returns a dict
            if predictions:
                output.update(predictions)
            return output
        elif predictions:
            # Return predictions from new heads
            predictions['embeddings'] = h_dict
            return predictions
        else:
            # Return embeddings if no task specified
            return h_dict


class HomoEnergyGNN(nn.Module):
    """Homogeneous GNN for simplified energy systems"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        input_dim = config.get('input_dim', 17)
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        conv_type = config.get('conv_type', 'sage')
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            if conv_type == 'sage':
                conv = SAGEConv(self.hidden_dim, self.hidden_dim)
            elif conv_type == 'gat':
                conv = GATConv(self.hidden_dim, self.hidden_dim // 4, heads=4, concat=True)
            elif conv_type == 'gcn':
                conv = GCNConv(self.hidden_dim, self.hidden_dim)
            else:
                raise ValueError(f"Unknown conv type: {conv_type}")
            
            self.convs.append(conv)
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Task heads
        self.task_heads = TaskHeads(self.hidden_dim, config)
        
        # ADD: Discovery components for HomoGNN too
        if config.get('use_discovery_mode', True):
            from models.pooling_layers import ConstrainedDiffPool
            from models.task_heads import ComplementarityScoreHead, NetworkCentralityHead, EnergyFlowHead
            
            self.diffpool = ConstrainedDiffPool(
                input_dim=self.hidden_dim,
                max_clusters=config.get('max_clusters', 20),
                min_cluster_size=config.get('min_cluster_size', 3),
                max_cluster_size=config.get('max_cluster_size', 20)
            )
            self.complementarity_head = ComplementarityScoreHead(self.hidden_dim)
            self.centrality_head = NetworkCentralityHead(self.hidden_dim)
            self.flow_head = EnergyFlowHead(self.hidden_dim)
        
        logger.info(f"Initialized HomoEnergyGNN with {self.num_layers} {conv_type} layers")
    
    def forward(self, data: Union[Data, Dict], task: str = None):
        """
        Forward pass through homogeneous GNN
        
        Args:
            data: Input data (Data object or dict)
            task: Task name for task-specific output
            
        Returns:
            Task-specific output or embeddings
        """
        # Extract features and edges
        if isinstance(data, Data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch if hasattr(data, 'batch') else None
        elif isinstance(data, dict):
            x = data['x']
            edge_index = data['edge_index']
            batch = data.get('batch')
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
        
        # Initial projection
        x = self.input_projection(x)
        
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x = x + x_new  # Residual connection
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Apply discovery components if available
        predictions = {}
        if hasattr(self, 'diffpool'):
            x_pooled, adj_pooled, S, aux_loss = self.diffpool(x, edge_index)
            predictions['clusters'] = S
            predictions['aux_loss'] = aux_loss
        
        if hasattr(self, 'complementarity_head'):
            predictions['complementarity'] = self.complementarity_head(x, edge_index)
        
        if hasattr(self, 'centrality_head'):
            predictions['centrality'] = self.centrality_head(
                x, edge_index, predictions.get('clusters')
            )
        
        if hasattr(self, 'flow_head'):
            predictions['energy_flow'] = self.flow_head(x, edge_index)
        
        # Task-specific output (legacy support)
        if task:
            output = self.task_heads(x, task, edge_index)
            if predictions:
                output.update(predictions)
            return output
        elif predictions:
            predictions['embeddings'] = x
            return predictions
        else:
            return x


# ============================================
# FACTORY FUNCTION
# ============================================

def create_gnn_model(model_type: str, config: Dict) -> nn.Module:
    """
    Factory function to create GNN models
    
    Args:
        model_type: Type of model ('hetero', 'homo', 'multi_rel', 'adaptive')
        config: Model configuration
        
    Returns:
        Initialized GNN model
    """
    if model_type == 'hetero':
        model = HeteroEnergyGNN(config)
    elif model_type == 'homo':
        model = HomoEnergyGNN(config)
    elif model_type == 'multi_rel':
        # Could add more sophisticated multi-relational model
        model = HeteroEnergyGNN(config)
    elif model_type == 'adaptive':
        # Could add adaptive architecture selection
        model = HeteroEnergyGNN(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Log model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created {model_type} GNN model")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model


# Backward compatibility
create_energy_gnn_base = lambda config: create_gnn_model('hetero', config)


# ============================================
# TESTING
# ============================================

if __name__ == "__main__":
    # Test model creation
    config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.1,
        'num_clusters': 10,
        'building_features': 17,
        'cable_group_features': 12,
        'transformer_features': 8,
        'cluster_features': 11,
        'input_dim': 17,
        'conv_type': 'sage'
    }
    
    # Test heterogeneous model
    print("Testing Heterogeneous GNN...")
    hetero_model = create_gnn_model('hetero', config)
    
    # Create dummy hetero data
    hetero_data = HeteroData()
    hetero_data['building'].x = torch.randn(100, 17)
    hetero_data['cable_group'].x = torch.randn(10, 12)
    hetero_data['transformer'].x = torch.randn(5, 8)
    hetero_data[('building', 'connected_to', 'cable_group')].edge_index = torch.randint(0, 10, (2, 200))
    
    # Test forward pass
    output = hetero_model(hetero_data, task='clustering')
    print(f"Clustering output shape: {output.shape}")
    
    # Test homogeneous model
    print("\nTesting Homogeneous GNN...")
    homo_model = create_gnn_model('homo', config)
    
    # Create dummy homo data
    homo_data = Data(
        x=torch.randn(100, 17),
        edge_index=torch.randint(0, 100, (2, 500))
    )
    
    # Test forward pass
    output = homo_model(homo_data, task='retrofit')
    print(f"Retrofit output shape: {output.shape}")
    
    print("\n✅ All models created and tested successfully!")