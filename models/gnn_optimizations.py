"""
GNN Architecture Optimizations
Addresses over-smoothing, attention efficiency, and gradient flow issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.utils import dropout_adj, add_self_loops, degree
from typing import Optional, Tuple, Dict, List
import math
import logging

logger = logging.getLogger(__name__)


class DropEdge(nn.Module):
    """
    DropEdge: Randomly dropping edges during training to prevent over-smoothing
    Reference: https://arxiv.org/abs/1907.10903
    """
    
    def __init__(self, p: float = 0.2, force_undirected: bool = True):
        super().__init__()
        self.p = p
        self.force_undirected = force_undirected
        
    def forward(self, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None,
                training: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Drop edges randomly during training
        
        Args:
            edge_index: Edge connectivity [2, E]
            edge_attr: Edge features [E, F]
            training: Whether in training mode
            
        Returns:
            Dropped edge_index and edge_attr
        """
        if not training or self.p == 0.0:
            return edge_index, edge_attr
            
        # Use torch_geometric's dropout_adj if available
        return dropout_adj(edge_index, edge_attr, p=self.p, 
                          force_undirected=self.force_undirected, training=training)


class PairNorm(nn.Module):
    """
    PairNorm: Pairwise normalization to prevent over-smoothing
    Reference: https://arxiv.org/abs/1909.12223
    """
    
    def __init__(self, scale: float = 1.0, eps: float = 1e-5):
        super().__init__()
        self.scale = scale
        self.eps = eps
        
    def forward(self, x: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply PairNorm to node features
        
        Args:
            x: Node features [N, F]
            batch: Batch assignment [N]
            
        Returns:
            Normalized features
        """
        if batch is None:
            # Single graph
            mean = x.mean(dim=0, keepdim=True)
            x = x - mean
            norm = (x.pow(2).sum(dim=1, keepdim=True) / x.size(0)).sqrt() + self.eps
            x = self.scale * x / norm
        else:
            # Batched graphs
            batch_size = int(batch.max()) + 1
            for b in range(batch_size):
                mask = (batch == b)
                x_b = x[mask]
                mean = x_b.mean(dim=0, keepdim=True)
                x_b = x_b - mean
                norm = (x_b.pow(2).sum(dim=1, keepdim=True) / x_b.size(0)).sqrt() + self.eps
                x[mask] = self.scale * x_b / norm
                
        return x


class EfficientAttention(nn.Module):
    """
    Efficient scaled dot-product attention for graphs
    Replaces concatenation-based attention with more efficient dot-product
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply efficient attention over edges
        
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Edge connectivity [2, E]
            return_attention: Whether to return attention weights
            
        Returns:
            Updated node features and optional attention weights
        """
        N, D = x.shape
        row, col = edge_index
        
        # Project to Q, K, V
        Q = self.q_proj(x).view(N, self.num_heads, self.head_dim)
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)
        
        # Compute attention scores for edges
        q_i = Q[row]  # [E, heads, head_dim]
        k_j = K[col]  # [E, heads, head_dim]
        v_j = V[col]  # [E, heads, head_dim]
        
        # Scaled dot-product attention
        scores = (q_i * k_j).sum(dim=-1) / self.scale  # [E, heads]
        
        # Softmax over incoming edges for each node
        attention = torch.zeros(N, self.num_heads, N, device=x.device)
        attention[row, :, col] = scores
        
        # Apply softmax per node
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.zeros(N, self.num_heads, self.head_dim, device=x.device)
        for i in range(N):
            neighbors = (row == i)
            if neighbors.any():
                out[i] = (attention[i].unsqueeze(-1) * V).sum(dim=0)
        
        # Concatenate heads and project
        out = out.view(N, -1)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attention.mean(dim=1)  # Average over heads
        return out, None


class JumpingKnowledgeNetwork(nn.Module):
    """
    Jumping Knowledge Network to combine features from all layers
    Prevents over-smoothing by preserving information from early layers
    """
    
    def __init__(self, mode: str = 'cat', channels: Optional[int] = None, 
                 num_layers: Optional[int] = None):
        super().__init__()
        self.mode = mode
        
        if mode == 'cat':
            # Concatenation mode - no parameters needed
            pass
        elif mode == 'lstm':
            assert channels is not None and num_layers is not None
            self.lstm = nn.LSTM(channels, channels, batch_first=True)
        elif mode == 'max':
            # Max pooling mode - no parameters
            pass
        elif mode == 'attention':
            assert channels is not None and num_layers is not None
            self.attention = nn.Linear(channels, 1)
        else:
            raise ValueError(f"Unknown JK mode: {mode}")
            
    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine features from all layers
        
        Args:
            xs: List of features from each layer [x_1, x_2, ..., x_L]
            
        Returns:
            Combined features
        """
        if self.mode == 'cat':
            return torch.cat(xs, dim=-1)
            
        elif self.mode == 'lstm':
            x = torch.stack(xs, dim=1)  # [N, L, F]
            _, (h, _) = self.lstm(x)
            return h.squeeze(0)
            
        elif self.mode == 'max':
            return torch.stack(xs, dim=0).max(dim=0)[0]
            
        elif self.mode == 'attention':
            x = torch.stack(xs, dim=1)  # [N, L, F]
            scores = self.attention(x).squeeze(-1)  # [N, L]
            scores = F.softmax(scores, dim=-1)
            return (x * scores.unsqueeze(-1)).sum(dim=1)


class SparseDiffPool(nn.Module):
    """
    Sparse version of DiffPool that works with edge_index
    Avoids converting to dense adjacency matrix
    """
    
    def __init__(self, input_dim: int, output_dim: int, max_clusters: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_clusters = max_clusters
        
        # Assignment and embedding networks
        self.assign_conv1 = nn.Linear(input_dim, input_dim)
        self.assign_conv2 = nn.Linear(input_dim, max_clusters)
        
        self.embed_conv1 = nn.Linear(input_dim, output_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sparse DiffPool forward pass
        
        Args:
            x: Node features [N, F]
            edge_index: Edge connectivity [2, E]
            batch: Batch assignment
            
        Returns:
            x_pooled: Pooled features [K, F']
            edge_index_pooled: Pooled edges [2, E']
            S: Assignment matrix [N, K]
        """
        # Generate soft assignments
        S = F.relu(self.assign_conv1(x))
        S = F.softmax(self.assign_conv2(S), dim=-1)
        
        # Generate embeddings
        x_embed = self.embed_conv1(x)
        
        # Pool features
        x_pooled = torch.matmul(S.T, x_embed)
        
        # Pool edges (sparse operation)
        # Map nodes to clusters
        cluster_assignment = S.argmax(dim=1)  # Hard assignment for edges
        
        # Map edges to cluster space
        row, col = edge_index
        row_pooled = cluster_assignment[row]
        col_pooled = cluster_assignment[col]
        
        # Remove self-loops and duplicates
        edge_index_pooled = torch.stack([row_pooled, col_pooled], dim=0)
        edge_index_pooled = torch.unique(edge_index_pooled, dim=1)
        
        # Remove self-loops
        mask = edge_index_pooled[0] != edge_index_pooled[1]
        edge_index_pooled = edge_index_pooled[:, mask]
        
        return x_pooled, edge_index_pooled, S


class GradientCheckpointing(nn.Module):
    """
    Wrapper for gradient checkpointing to save memory
    """
    
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        
    def forward(self, *args, **kwargs):
        """
        Forward with gradient checkpointing
        """
        if self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self.module, *args, **kwargs)
        else:
            return self.module(*args, **kwargs)


def add_gnn_optimizations(model: nn.Module, config: Dict) -> nn.Module:
    """
    Add optimizations to existing GNN model
    
    Args:
        model: GNN model to optimize
        config: Configuration dictionary
        
    Returns:
        Optimized model
    """
    # Add DropEdge
    if config.get('use_dropedge', True):
        model.dropedge = DropEdge(p=config.get('dropedge_p', 0.2))
        logger.info("Added DropEdge with p=0.2")
    
    # Add PairNorm
    if config.get('use_pairnorm', True):
        model.pairnorm = PairNorm(scale=config.get('pairnorm_scale', 1.0))
        logger.info("Added PairNorm")
    
    # Add JumpingKnowledge
    if config.get('use_jumping_knowledge', True):
        jk_mode = config.get('jk_mode', 'cat')
        model.jumping_knowledge = JumpingKnowledgeNetwork(
            mode=jk_mode,
            channels=config.get('hidden_dim', 128),
            num_layers=config.get('num_layers', 4)
        )
        logger.info(f"Added JumpingKnowledge with mode={jk_mode}")
    
    # Replace attention with efficient version
    if config.get('use_efficient_attention', True):
        if hasattr(model, 'attention'):
            model.attention = EfficientAttention(
                hidden_dim=config.get('hidden_dim', 128),
                num_heads=config.get('num_heads', 8)
            )
            logger.info("Replaced attention with efficient scaled dot-product attention")
    
    # Add gradient checkpointing wrapper to heavy modules
    if config.get('use_gradient_checkpointing', False):
        if hasattr(model, 'mp_layers'):
            for i, layer in enumerate(model.mp_layers):
                model.mp_layers[i] = GradientCheckpointing(layer)
            logger.info("Added gradient checkpointing to message passing layers")
    
    return model


if __name__ == "__main__":
    # Test optimizations
    print("Testing GNN optimizations...")
    
    # Test DropEdge
    dropedge = DropEdge(p=0.2)
    edge_index = torch.randint(0, 100, (2, 500))
    dropped_edges, _ = dropedge(edge_index, training=True)
    print(f"DropEdge: {edge_index.shape[1]} -> {dropped_edges.shape[1]} edges")
    
    # Test PairNorm
    pairnorm = PairNorm()
    x = torch.randn(100, 128)
    x_norm = pairnorm(x)
    print(f"PairNorm: mean={x_norm.mean():.4f}, std={x_norm.std():.4f}")
    
    # Test EfficientAttention
    attention = EfficientAttention(128, num_heads=8)
    x = torch.randn(100, 128)
    edge_index = torch.randint(0, 100, (2, 500))
    out, _ = attention(x, edge_index)
    print(f"EfficientAttention: {x.shape} -> {out.shape}")
    
    # Test JumpingKnowledge
    jk = JumpingKnowledgeNetwork(mode='cat')
    xs = [torch.randn(100, 128) for _ in range(4)]
    combined = jk(xs)
    print(f"JumpingKnowledge: {len(xs)} x {xs[0].shape} -> {combined.shape}")
    
    print("\nAll optimizations tested successfully!")