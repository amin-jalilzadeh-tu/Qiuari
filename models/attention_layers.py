"""
Attention layers for Energy GNN with complementarity awareness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, GATv2Conv
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.utils import softmax, add_self_loops
from typing import Optional, Tuple, Dict, List, Any
import math


class ComplementarityAttention(MessagePassing):
    """
    Graph attention layer that incorporates complementarity scores
    Learns to focus on complementary neighbors for energy sharing
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.1,
        edge_dim: Optional[int] = None,
        complementarity_weight: float = 0.5,
        negative_slope: float = 0.2,
        add_self_loops: bool = True,
        bias: bool = True
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.complementarity_weight = complementarity_weight
        self.negative_slope = negative_slope
        self.add_self_loops_flag = add_self_loops
        
        # Multi-head projections
        self.lin_key = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_query = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_value = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Edge feature projection (if provided)
        if edge_dim is not None:
            self.lin_edge = nn.Linear(edge_dim, heads * out_channels, bias=False)
        else:
            self.lin_edge = None
            
        # Complementarity scoring
        self.complementarity_net = nn.Sequential(
            nn.Linear(2 * in_channels, heads),
            nn.Tanh()  # Output between -1 and 1
        )
        
        # Attention parameters
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        # Output projection
        self.lin_out = nn.Linear(heads * out_channels, out_channels, bias=bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize parameters"""
        nn.init.xavier_uniform_(self.lin_key.weight)
        nn.init.xavier_uniform_(self.lin_query.weight)
        nn.init.xavier_uniform_(self.lin_value.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.xavier_uniform_(self.lin_out.weight)
        
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        temporal_profiles: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with complementarity-aware attention
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Edge indices [2, E]
            edge_attr: Edge features [E, edge_dim]
            temporal_profiles: Temporal consumption profiles for complementarity [N, T]
            return_attention_weights: Whether to return attention weights
            
        Returns:
            Updated node features and optionally attention weights
        """
        H, C = self.heads, self.out_channels
        N = x.size(0)
        
        # Add self-loops
        if self.add_self_loops_flag:
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=N
            )
            
        # Linear transformations
        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)
        
        # Calculate complementarity scores if temporal profiles provided
        if temporal_profiles is not None:
            comp_scores = self._calculate_complementarity_scores(
                x, edge_index, temporal_profiles
            )
        else:
            comp_scores = None
            
        # Propagate with complementarity
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            comp_scores=comp_scores,
            size=None
        )
        
        # Reshape and apply output projection
        out = out.view(N, H * C)
        out = self.lin_out(out)
        
        if return_attention_weights:
            # Calculate attention weights for analysis
            alpha = self._get_attention_weights(
                query, key, edge_index, comp_scores
            )
            return out, alpha
        else:
            return out, None
            
    def message(
        self,
        query_i: torch.Tensor,
        key_j: torch.Tensor,
        value_j: torch.Tensor,
        edge_attr: Optional[torch.Tensor],
        comp_scores: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int]
    ) -> torch.Tensor:
        """
        Message function with complementarity-aware attention
        """
        # Calculate attention scores
        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        
        # Add edge features if provided
        if self.lin_edge is not None and edge_attr is not None:
            edge_feat = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
            alpha = alpha + (edge_feat * key_j).sum(dim=-1)
            
        # Incorporate complementarity scores
        if comp_scores is not None:
            # Reshape comp_scores to match heads
            comp_scores = comp_scores.unsqueeze(1).expand(-1, self.heads)
            # Weight attention by complementarity (negative correlation is good)
            alpha = alpha + self.complementarity_weight * (-comp_scores)
            
        # Apply softmax
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.dropout_layer(alpha)
        
        # Return weighted values
        return value_j * alpha.unsqueeze(-1)
        
    def _calculate_complementarity_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_profiles: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate complementarity scores for edges based on temporal profiles
        """
        row, col = edge_index
        
        # Get temporal profiles for source and target nodes
        profiles_i = temporal_profiles[row]
        profiles_j = temporal_profiles[col]
        
        # Calculate correlation
        profiles_i_norm = (profiles_i - profiles_i.mean(dim=1, keepdim=True)) / (
            profiles_i.std(dim=1, keepdim=True) + 1e-8
        )
        profiles_j_norm = (profiles_j - profiles_j.mean(dim=1, keepdim=True)) / (
            profiles_j.std(dim=1, keepdim=True) + 1e-8
        )
        
        # Correlation coefficient
        correlation = (profiles_i_norm * profiles_j_norm).mean(dim=1)
        
        # Also use learned complementarity from node features
        x_concat = torch.cat([x[row], x[col]], dim=1)
        learned_comp = self.complementarity_net(x_concat).mean(dim=1)
        
        # Combine correlation and learned complementarity
        comp_scores = 0.5 * correlation + 0.5 * learned_comp
        
        return comp_scores
        
    def _get_attention_weights(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        edge_index: torch.Tensor,
        comp_scores: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Get attention weights for visualization"""
        row, col = edge_index
        
        # Calculate raw attention scores
        alpha = (query[row] * key[col]).sum(dim=-1) / math.sqrt(self.out_channels)
        
        if comp_scores is not None:
            comp_scores = comp_scores.unsqueeze(1).expand(-1, self.heads)
            alpha = alpha + self.complementarity_weight * (-comp_scores)
            
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, row)
        
        return alpha.mean(dim=1)  # Average over heads


class TemporalAttention(nn.Module):
    """
    Temporal attention layer for time-series consumption patterns
    Learns which time periods are most important for clustering
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Multi-head temporal attention
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Positional encoding for time steps
        self.positional_encoding = self._create_positional_encoding(96, input_dim)  # 96 timesteps
        
        # Time-of-day embedding
        self.time_embedding = nn.Embedding(24, hidden_dim)  # 24 hours
        
        # Output projection
        self.output_proj = nn.Linear(input_dim + hidden_dim, input_dim)
        
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
            
        return pe.unsqueeze(0)  # Add batch dimension
        
    def forward(
        self,
        temporal_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention
        
        Args:
            temporal_features: Temporal features [B, T, D]
            mask: Attention mask [B, T]
            
        Returns:
            Attended features and attention weights
        """
        B, T, D = temporal_features.shape
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :T, :].to(temporal_features.device)
        temporal_features = temporal_features + pos_encoding
        
        # Apply temporal attention
        attended, attention_weights = self.temporal_attention(
            temporal_features,
            temporal_features,
            temporal_features,
            key_padding_mask=mask,
            need_weights=True
        )
        
        # Add time-of-day embeddings
        hours = torch.arange(24, device=temporal_features.device).repeat(B, T // 24)
        time_embeds = self.time_embedding(hours).view(B, T, -1)
        
        # Combine with time embeddings
        combined = torch.cat([attended, time_embeds], dim=-1)
        output = self.output_proj(combined)
        
        return output, attention_weights


class SpatialAttention(nn.Module):
    """
    Spatial attention for geographical/topological relationships
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.spatial_conv = TransformerConv(
            in_channels=input_dim,
            out_channels=hidden_dim,
            heads=4,
            dropout=dropout,
            edge_dim=3  # [distance, capacity, resistance]
        )
        
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply spatial attention
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes (distance, capacity, etc.)
            
        Returns:
            Spatially attended features
        """
        # Apply transformer convolution
        out = self.spatial_conv(x, edge_index, edge_attr)
        
        # Project back to original dimension
        out = self.output_proj(out)
        
        return out


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention for multi-level grid structure
    (Building -> LV Group -> Transformer -> Substation)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_levels: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_levels = num_levels
        
        # Level-specific attention modules
        self.level_attentions = nn.ModuleList([
            GATv2Conv(
                in_channels=input_dim if i == 0 else hidden_dim,
                out_channels=hidden_dim,
                heads=4,
                dropout=dropout,
                edge_dim=3,
                add_self_loops=True
            )
            for i in range(num_levels)
        ])
        
        # Cross-level attention
        self.cross_level_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim * num_levels, input_dim)
        
    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str], torch.Tensor],
        edge_attr_dict: Optional[Dict[Tuple[str, str], torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply hierarchical attention
        
        Args:
            x_dict: Node features per level
            edge_index_dict: Edge indices per level
            edge_attr_dict: Edge attributes per level
            
        Returns:
            Updated node features per level
        """
        level_outputs = {}
        
        # Process each level
        for level_idx, (node_type, x) in enumerate(x_dict.items()):
            # Find edges for this node type
            relevant_edges = [(k, v) for k, v in edge_index_dict.items() 
                             if k[0] == node_type]
            
            if relevant_edges:
                edge_key, edge_index = relevant_edges[0]
                edge_attr = edge_attr_dict.get(edge_key) if edge_attr_dict else None
                
                # Apply level-specific attention
                out = self.level_attentions[level_idx](
                    x, edge_index, edge_attr
                )
                level_outputs[node_type] = out
            else:
                level_outputs[node_type] = x
                
        # Cross-level attention (aggregate information across hierarchy)
        all_features = list(level_outputs.values())
        if len(all_features) > 1:
            # Stack features for cross-attention
            stacked = torch.stack(all_features, dim=1)  # [N, L, D]
            
            # Apply cross-level attention
            attended, _ = self.cross_level_attention(
                stacked, stacked, stacked
            )
            
            # Update level outputs
            for i, node_type in enumerate(level_outputs.keys()):
                level_outputs[node_type] = attended[:, i, :]
                
        return level_outputs


class UnifiedAttentionModule(nn.Module):
    """
    Unified attention module combining all attention mechanisms
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_complementarity: bool = True,
        use_temporal: bool = True,
        use_spatial: bool = True
    ):
        super().__init__()
        
        self.use_complementarity = use_complementarity
        self.use_temporal = use_temporal
        self.use_spatial = use_spatial
        
        # Complementarity attention
        if use_complementarity:
            self.comp_attention = ComplementarityAttention(
                in_channels=input_dim,
                out_channels=hidden_dim,
                heads=num_heads,
                dropout=dropout,
                complementarity_weight=0.5
            )
            
        # Temporal attention
        if use_temporal:
            self.temp_attention = TemporalAttention(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_heads=num_heads // 2,
                dropout=dropout
            )
            
        # Spatial attention
        if use_spatial:
            self.spatial_attention = SpatialAttention(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout
            )
            
        # Fusion layer
        num_components = sum([use_complementarity, use_temporal, use_spatial])
        self.fusion = nn.Linear(hidden_dim * num_components, input_dim)
        
        # Layer norm and dropout
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        temporal_features: Optional[torch.Tensor] = None,
        temporal_profiles: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Apply unified attention
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_attr: Edge attributes
            temporal_features: Temporal features for temporal attention
            temporal_profiles: Consumption profiles for complementarity
            
        Returns:
            Updated features and attention weights
        """
        outputs = []
        attention_weights = {}
        
        # Apply complementarity attention
        if self.use_complementarity:
            comp_out, comp_weights = self.comp_attention(
                x, edge_index, edge_attr, temporal_profiles,
                return_attention_weights=True
            )
            outputs.append(comp_out)
            attention_weights['complementarity'] = comp_weights
            
        # Apply temporal attention
        if self.use_temporal and temporal_features is not None:
            temp_out, temp_weights = self.temp_attention(temporal_features)
            # Average over time dimension for node features
            temp_out = temp_out.mean(dim=1)
            outputs.append(temp_out)
            attention_weights['temporal'] = temp_weights
            
        # Apply spatial attention
        if self.use_spatial:
            spatial_out = self.spatial_attention(x, edge_index, edge_attr)
            outputs.append(spatial_out)
            
        # Fuse all attention outputs
        if outputs:
            combined = torch.cat(outputs, dim=-1)
            fused = self.fusion(combined)
            
            # Residual connection and normalization
            output = self.norm(x + self.dropout(fused))
        else:
            output = x
            
        return output, attention_weights