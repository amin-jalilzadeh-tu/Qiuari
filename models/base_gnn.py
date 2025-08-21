# models/base_gnn.py - FIXED DIMENSION ISSUE
"""
Base heterogeneous GNN architecture for energy systems
Fixed cluster dimension mismatch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from typing import Dict, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class BuildingEncoder(nn.Module):
    """Encodes building features into embeddings"""
    
    def __init__(self, input_dim: int = 17, hidden_dim: int = 128):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor):
        return self.input_projection(x)


class LVGroupEncoder(nn.Module):
    """Encodes LV cable group features"""
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 128):
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
    
    def __init__(self, input_dim: int = 3, hidden_dim: int = 128):
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
    
    def __init__(self, input_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor):
        return self.input_projection(x)


class TemporalEncoder(nn.Module):
    """Encodes temporal context"""
    
    def __init__(self, hidden_dim: int = 36):
        super().__init__()
        
        self.season_embed = nn.Embedding(4, 12)
        self.daytype_embed = nn.Embedding(2, 8)
        self.hour_embed = nn.Embedding(24, 16)
        
        self.temporal_fusion = nn.Sequential(
            nn.Linear(36, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, season: torch.Tensor, is_weekend: torch.Tensor, hour: torch.Tensor):
        season_emb = self.season_embed(season)
        daytype_emb = self.daytype_embed(is_weekend.long())
        hour_emb = self.hour_embed(hour)
        
        temporal = torch.cat([season_emb, daytype_emb, hour_emb], dim=-1)
        return self.temporal_fusion(temporal)


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
        
        # FIX: Remove building_to_cluster since clusters are static
        # We'll use cluster info differently
        
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
        
        # Top-down: Transformers -> LV Groups
        if ('transformer', 'feeds', 'cable_group') in edge_index_dict:
            edge_index = edge_index_dict[('transformer', 'feeds', 'cable_group')]
            if edge_index.shape[1] > 0 and 'transformer' in h_dict and 'cable_group' in h_dict:
                h_lv_from_trans = self.transformer_to_lv(
                    (h_dict['transformer'], h_dict['cable_group']),
                    edge_index
                )
                h_dict['cable_group'] = h_dict['cable_group'] + 0.3 * h_lv_from_trans
        
        # Top-down: LV Groups -> Buildings
        if ('cable_group', 'supplies', 'building') in edge_index_dict:
            edge_index = edge_index_dict[('cable_group', 'supplies', 'building')]
            if edge_index.shape[1] > 0 and 'cable_group' in h_dict and 'building' in h_dict:
                h_building_from_lv = self.lv_to_building(
                    (h_dict['cable_group'], h_dict['building']),
                    edge_index
                )
                h_dict['building'] = h_dict['building'] + 0.3 * h_building_from_lv
        
        # Lateral: Building adjacency
        if ('building', 'adjacent_to', 'building') in edge_index_dict:
            edge_index = edge_index_dict[('building', 'adjacent_to', 'building')]
            if edge_index.shape[1] > 0 and 'building' in h_dict:
                h_building_adj = self.building_adjacency(
                    h_dict['building'],
                    edge_index
                )
                h_dict['building'] = h_dict['building'] + 0.2 * h_building_adj
        
        # FIX: Don't update adjacency clusters - they remain static
        # The cluster information is used as context, not updated
        
        return h_dict


class EnergyGNNBase(nn.Module):
    """Main base GNN model for energy grid"""
    
    def __init__(self, 
                 config: Dict[str, Any],
                 node_features: Optional[Dict[str, int]] = None):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.1)
        
        # Feature dimensions
        if node_features is None:
            node_features = {
                'building': 17,
                'cable_group': 4,
                'transformer': 3,
                'adjacency_cluster': 4
            }
        
        # Node encoders
        self.building_encoder = BuildingEncoder(
            node_features.get('building', 17), self.hidden_dim
        )
        self.lv_group_encoder = LVGroupEncoder(
            node_features.get('cable_group', 4), self.hidden_dim
        )
        self.transformer_encoder = TransformerEncoder(
            node_features.get('transformer', 3), self.hidden_dim
        )
        self.cluster_encoder = AdjacencyClusterEncoder(
            node_features.get('adjacency_cluster', 4), 64
        )
        
        # Temporal encoder
        self.temporal_encoder = TemporalEncoder(36)
        
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
        
        # Output projections
        self.output_projections = nn.ModuleDict({
            'building': nn.Linear(self.hidden_dim + 36, self.hidden_dim),
            'cable_group': nn.Linear(self.hidden_dim + 36, self.hidden_dim),
            'transformer': nn.Linear(self.hidden_dim, self.hidden_dim),
            'adjacency_cluster': nn.Linear(64, 64)
        })
        
        logger.info(f"Initialized EnergyGNNBase with {self.num_layers} layers")
    
    def forward(self, 
                x_dict: Dict[str, torch.Tensor],
                edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
                temporal_context: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GNN
        
        Args:
            x_dict: Node features by type
            edge_index_dict: Edge indices by relation type  
            temporal_context: Optional temporal features
            
        Returns:
            Dictionary of embeddings by node type
        """
        
        # STEP 1: Initial encoding
        h_dict = {}
        
        if 'building' in x_dict:
            h_dict['building'] = self.building_encoder(x_dict['building'])
        
        if 'cable_group' in x_dict:
            h_dict['cable_group'] = self.lv_group_encoder(x_dict['cable_group'])
        
        if 'transformer' in x_dict:
            h_dict['transformer'] = self.transformer_encoder(x_dict['transformer'])
        
        if 'adjacency_cluster' in x_dict:
            h_dict['adjacency_cluster'] = self.cluster_encoder(x_dict['adjacency_cluster'])
        
        # STEP 2: Add reverse edges
        edge_index_dict_bi = self._add_reverse_edges(edge_index_dict)
        
        # STEP 3: Message passing (but not for clusters)
        for i, mp_layer in enumerate(self.mp_layers):
            h_dict_new = mp_layer(h_dict, edge_index_dict_bi)
            
            # Residual connection and normalization
            for node_type in h_dict_new:
                if node_type in h_dict and node_type in self.layer_norms:
                    # Skip adjacency_cluster updates
                    if node_type == 'adjacency_cluster':
                        h_dict_new[node_type] = h_dict[node_type]  # Keep original
                        continue
                        
                    # Residual
                    h_dict_new[node_type] = h_dict[node_type] + h_dict_new[node_type]
                    # Normalize
                    h_dict_new[node_type] = self.layer_norms[node_type](h_dict_new[node_type])
                    # Dropout
                    h_dict_new[node_type] = F.dropout(
                        h_dict_new[node_type], p=self.dropout, training=self.training
                    )
            
            h_dict = h_dict_new
        
        # STEP 4: Add temporal context (optional)
        if temporal_context is not None:
            temporal_embed = self.temporal_encoder(
                temporal_context['season'],
                temporal_context['is_weekend'],
                temporal_context['hour']
            )
            
            # Add to buildings
            if 'building' in h_dict:
                num_buildings = h_dict['building'].shape[0]
                temporal_expand = temporal_embed.expand(num_buildings, -1)
                h_dict['building'] = torch.cat([h_dict['building'], temporal_expand], dim=-1)
                h_dict['building'] = self.output_projections['building'](h_dict['building'])
            
            # Add to cable groups
            if 'cable_group' in h_dict:
                num_groups = h_dict['cable_group'].shape[0]
                temporal_expand = temporal_embed.expand(num_groups, -1)
                h_dict['cable_group'] = torch.cat([h_dict['cable_group'], temporal_expand], dim=-1)
                h_dict['cable_group'] = self.output_projections['cable_group'](h_dict['cable_group'])
        
        # STEP 5: Final projections
        output_dict = {}
        for node_type in h_dict:
            if node_type in ['transformer', 'adjacency_cluster']:
                if node_type in self.output_projections:
                    output_dict[node_type] = self.output_projections[node_type](h_dict[node_type])
                else:
                    output_dict[node_type] = h_dict[node_type]
            else:
                output_dict[node_type] = h_dict[node_type]
        
        return output_dict
    
    def _add_reverse_edges(self, edge_index_dict: Dict) -> Dict:
        """Add reverse edges for bidirectional message passing"""
        edge_index_dict_bi = edge_index_dict.copy()
        
        # Map forward to reverse relationships
        reverse_mappings = [
            (('building', 'connected_to', 'cable_group'), 
             ('cable_group', 'supplies', 'building')),
            (('cable_group', 'connects_to', 'transformer'),
             ('transformer', 'feeds', 'cable_group')),
        ]
        
        for forward_key, reverse_key in reverse_mappings:
            if forward_key in edge_index_dict:
                forward_edges = edge_index_dict[forward_key]
                if forward_edges.shape[1] > 0:
                    reverse_edges = torch.stack([forward_edges[1], forward_edges[0]], dim=0)
                    edge_index_dict_bi[reverse_key] = reverse_edges
        
        return edge_index_dict_bi


def create_energy_gnn_base(config: Dict) -> EnergyGNNBase:
    """Factory function to create the base GNN model"""
    model = EnergyGNNBase(config)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created EnergyGNNBase with {total_params:,} parameters")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model