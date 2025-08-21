# models/attention_layers.py - FIXED VERSION
"""
Attention mechanisms for discovering energy complementarity and relationships
Fixed argument passing issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class MultiHeadComplementarityAttention(nn.Module):
    """
    Multi-head attention to discover complementary energy patterns
    """
    
    def __init__(self, 
                 embed_dim: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Building embeddings [batch_size, num_buildings, embed_dim]
            mask: Optional attention mask
        
        Returns:
            enhanced_embeddings, attention_weights
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        Q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores / self.temperature
        
        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(~mask, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attended = torch.matmul(attention_weights, V)
        
        # Reshape and project output
        attended = attended.transpose(1, 2).reshape(B, N, C)
        output = self.out_proj(attended)
        
        # Residual connection and layer norm
        enhanced = self.layer_norm(x + output)
        
        return enhanced, attention_weights


class ComplementarityScorer(nn.Module):
    """Scores complementarity between building pairs"""
    
    def __init__(self, embed_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, embeddings):
        B, N, D = embeddings.shape
        
        # Expand for pairwise computation
        embed_i = embeddings.unsqueeze(2).expand(B, N, N, D)
        embed_j = embeddings.unsqueeze(1).expand(B, N, N, D)
        
        # Concatenate pairs
        pairs = torch.cat([embed_i, embed_j], dim=-1)
        
        # Compute scores
        scores = self.scorer(pairs).squeeze(-1)
        
        # Make symmetric
        scores = (scores + scores.transpose(1, 2)) / 2
        
        return scores


class SpatialAttention(nn.Module):
    """Attention based on spatial proximity"""
    
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        
        self.spatial_proj = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.embed_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, embeddings, positions, adjacency=None):
        B, N, _ = embeddings.shape
        
        # Compute pairwise distances
        pos_i = positions.unsqueeze(2)
        pos_j = positions.unsqueeze(1)
        distances = torch.norm(pos_i - pos_j, dim=-1)
        
        # Normalize distances
        max_dist = distances.max() if distances.numel() > 0 else 1.0
        distances = distances / (max_dist + 1e-6)
        
        # Create spatial features
        spatial_features = torch.stack([
            1.0 - distances,
            adjacency.float() if adjacency is not None else torch.zeros_like(distances),
            torch.zeros_like(distances)
        ], dim=-1)
        
        # Compute spatial scores
        spatial_scores = self.spatial_proj(spatial_features).squeeze(-1)
        
        # Compute embedding-based scores
        embed_i = embeddings.unsqueeze(2).expand(B, N, N, -1)
        embed_j = embeddings.unsqueeze(1).expand(B, N, N, -1)
        embed_pairs = torch.cat([embed_i, embed_j], dim=-1)
        embed_scores = self.embed_scorer(embed_pairs).squeeze(-1)
        
        # Combine scores
        attention = torch.sigmoid(spatial_scores + embed_scores)
        
        return attention


class TemporalComplementarityAttention(nn.Module):
    """Discovers temporal complementarity patterns"""
    
    def __init__(self, embed_dim: int = 128, temporal_dim: int = 24):
        super().__init__()
        
        self.pattern_scorer = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
    def forward(self, embeddings, temporal_features=None):
        B, N, D = embeddings.shape
        
        # Use embeddings directly for now
        temporal_encoded = embeddings
        
        # Compute pairwise pattern similarity
        embed_i = temporal_encoded.unsqueeze(2).expand(B, N, N, D)
        embed_j = temporal_encoded.unsqueeze(1).expand(B, N, N, D)
        pairs = torch.cat([embed_i, embed_j], dim=-1)
        
        # Score patterns
        pattern_scores = self.pattern_scorer(pairs).squeeze(-1)
        
        # Make symmetric and convert to complementarity
        pattern_scores = (pattern_scores + pattern_scores.transpose(1, 2)) / 2
        complementarity = -pattern_scores
        
        return complementarity


class LVGroupConstraintMask(nn.Module):
    """Creates attention masks for LV group boundaries"""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, lv_group_ids):
        B, N = lv_group_ids.shape
        
        # Expand for pairwise comparison
        ids_i = lv_group_ids.unsqueeze(2).expand(B, N, N)
        ids_j = lv_group_ids.unsqueeze(1).expand(B, N, N)
        
        # Create mask
        mask = (ids_i == ids_j)
        
        return mask


class AttentionAggregator(nn.Module):
    """Aggregates multiple attention mechanisms"""
    
    def __init__(self, num_attention_types: int = 4):
        super().__init__()
        self.attention_weights = nn.Parameter(torch.ones(num_attention_types) / num_attention_types)
        
    def forward(self, attention_list):
        # Normalize weights
        weights = F.softmax(self.attention_weights, dim=0)
        
        # Weighted sum
        combined = torch.zeros_like(attention_list[0])
        for i, attention in enumerate(attention_list):
            combined = combined + weights[i] * attention
        
        return combined


class EnergyComplementarityAttention(nn.Module):
    """Main attention module combining all mechanisms"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        embed_dim = config.get('hidden_dim', 128)
        num_heads = config.get('attention_heads', 8)
        dropout = config.get('dropout', 0.1)
        
        # Different attention mechanisms
        self.multi_head_attention = MultiHeadComplementarityAttention(embed_dim, num_heads, dropout)
        self.complementarity_scorer = ComplementarityScorer(embed_dim)
        self.spatial_attention = SpatialAttention(embed_dim)
        self.temporal_attention = TemporalComplementarityAttention(embed_dim)
        self.lv_mask_generator = LVGroupConstraintMask()
        self.aggregator = AttentionAggregator(num_attention_types=4)
        
        self.output_projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        logger.info("Initialized EnergyComplementarityAttention")
    
    def forward(self, embeddings_dict, edge_index_dict, temporal_features=None, return_attention=True):
        """
        Apply all attention mechanisms
        """
        # Get building embeddings
        building_embeddings = embeddings_dict.get('building')
        
        if building_embeddings is None:
            raise ValueError("Building embeddings not found")
        
        # Add batch dimension if needed
        if building_embeddings.dim() == 2:
            building_embeddings = building_embeddings.unsqueeze(0)
        
        B, N, D = building_embeddings.shape
        
        # Create placeholders for now
        positions = torch.randn(B, N, 2).to(building_embeddings.device)
        lv_group_ids = torch.zeros(B, N).long().to(building_embeddings.device)
        
        # Generate masks
        lv_mask = self.lv_mask_generator(lv_group_ids)
        
        # 1. Multi-head attention - FIX: pass mask as positional argument
        enhanced_mha, attention_mha = self.multi_head_attention(building_embeddings, lv_mask)
        
        # 2. Complementarity scores
        comp_scores = self.complementarity_scorer(building_embeddings)
        comp_scores = comp_scores * lv_mask.float()
        
        # 3. Spatial attention
        spatial_att = self.spatial_attention(building_embeddings, positions)
        spatial_att = spatial_att * lv_mask.float()
        
        # 4. Temporal complementarity
        temporal_comp = self.temporal_attention(building_embeddings, temporal_features)
        temporal_comp = temporal_comp * lv_mask.float()
        
        # Aggregate all attention types
        attention_types = [
            attention_mha.mean(dim=1),  # Average over heads
            comp_scores,
            spatial_att,
            torch.sigmoid(temporal_comp)
        ]
        
        final_attention = self.aggregator(attention_types)
        
        # Apply attention through matrix multiplication
        enhanced = torch.bmm(final_attention, building_embeddings)
        
        # Add residual connection
        enhanced = building_embeddings + 0.5 * enhanced
        enhanced = self.output_projection(enhanced)
        
        # Update embeddings dict
        output_embeddings = embeddings_dict.copy()
        output_embeddings['building'] = enhanced.squeeze(0) if B == 1 else enhanced
        
        # Prepare output
        output = {
            'embeddings': output_embeddings,
            'complementarity_matrix': comp_scores.squeeze(0) if B == 1 else comp_scores,
        }
        
        if return_attention:
            output['attention_weights'] = {
                'multi_head': attention_mha,
                'complementarity': comp_scores,
                'spatial': spatial_att,
                'temporal': temporal_comp,
                'final': final_attention
            }
        
        return output


def create_attention_module(config: Dict) -> EnergyComplementarityAttention:
    """Create attention module with config"""
    return EnergyComplementarityAttention(config)