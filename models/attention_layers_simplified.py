"""
Simplified Attention Layers for Energy GNN
Keeping only essential attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from typing import Optional, Tuple


class SimpleMultiHeadAttention(nn.Module):
    """
    Simple multi-head attention for importance weighting
    Much simpler than the original complex version
    """
    
    def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        # Simple linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention
        
        Args:
            x: Input features [N, hidden_dim]
            mask: Optional attention mask
            
        Returns:
            Attended features [N, hidden_dim]
        """
        N = x.size(0)
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        attended = attended.view(N, self.hidden_dim)
        output = self.out_proj(attended)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + self.dropout(output))
        
        return output


class GraphAttentionLayer(nn.Module):
    """
    Simplified graph attention layer
    Wrapper around PyG's GATConv with residual connection
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True
    ):
        super().__init__()
        
        self.gat = GATConv(
            in_channels=in_channels,
            out_channels=out_channels // heads,
            heads=heads,
            dropout=dropout,
            concat=True
        )
        
        self.residual = residual
        if residual and in_channels != out_channels:
            self.residual_proj = nn.Linear(in_channels, out_channels)
        else:
            self.residual_proj = None
            
        self.layer_norm = nn.LayerNorm(out_channels)
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply graph attention
        
        Args:
            x: Node features
            edge_index: Graph connectivity
            
        Returns:
            (attended_features, attention_weights)
        """
        # Apply GAT
        out = self.gat(x, edge_index)
        
        # Residual connection
        if self.residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(x)
            else:
                residual = x
            out = out + residual
        
        # Layer norm
        out = self.layer_norm(out)
        
        # Get attention weights if needed (for explainability)
        # Note: GATConv doesn't directly expose attention weights in newer versions
        # This is a placeholder - actual implementation would need to access internal states
        attention_weights = torch.ones(edge_index.size(1), device=x.device)
        
        return out, attention_weights


class TemporalAttention(nn.Module):
    """
    Simple temporal attention for time series
    Identifies important time steps
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal attention
        
        Args:
            x: Temporal features [N, T, D]
            
        Returns:
            (weighted_features, attention_weights)
        """
        # Calculate attention scores
        scores = self.attention(x)  # [N, T, 1]
        weights = F.softmax(scores, dim=1)  # [N, T, 1]
        
        # Apply attention
        weighted = x * weights  # [N, T, D]
        output = weighted.sum(dim=1)  # [N, D]
        
        return output, weights.squeeze(-1)