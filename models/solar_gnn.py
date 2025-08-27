"""
Simplified Solar Recommendation GNN
Focus: Discover self-sufficient clusters and recommend solar installations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from typing import Dict, Optional, Tuple


class SolarRecommendationGNN(nn.Module):
    """
    Simplified GNN for solar recommendations with semi-supervised learning
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_clusters: int = 15,
        temporal_dim: int = 24,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # 1. Simple feature encoder (no complex multi-type encoders)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. Basic GNN layers (just GCN, no complex attention)
        self.gnn_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # 3. Temporal processor (simple LSTM)
        self.temporal = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True
        )
        
        # 4. Clustering head for self-sufficient districts
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_clusters)
        )
        
        # 5. Solar recommendation head (binary: good/bad for solar)
        self.solar_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for node + cluster features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 6. LV group constraint layer
        self.lv_constraint = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = dropout
        self.num_clusters = num_clusters
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_features: Optional[torch.Tensor] = None,
        lv_group_mask: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Node features [N, F]
            edge_index: Graph connectivity [2, E]
            temporal_features: Time series data [N, T, F]
            lv_group_mask: Mask for LV group constraints [N, G]
            batch: Batch assignment for nodes
            
        Returns:
            Dictionary with cluster assignments and solar recommendations
        """
        
        # 1. Encode features
        h = self.encoder(x)
        
        # 2. GNN message passing (within LV groups)
        for i, conv in enumerate(self.gnn_layers):
            h_new = conv(h, edge_index)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)
            
            # Apply LV group constraints
            if lv_group_mask is not None:
                h_new = self._apply_lv_constraints(h_new, lv_group_mask)
            
            # Skip connection
            if i > 0:
                h = h + h_new
            else:
                h = h_new
        
        # 3. Process temporal features if available
        if temporal_features is not None:
            # Simple temporal processing
            lstm_out, _ = self.temporal(temporal_features)
            # Take last timestep
            temporal_repr = lstm_out[:, -1, :]
            # Combine with spatial features
            h = h + temporal_repr
        
        # 4. Cluster assignment (soft clustering)
        cluster_logits = self.cluster_head(h)
        cluster_probs = F.softmax(cluster_logits, dim=-1)
        
        # 5. Generate cluster-level features
        if batch is not None:
            # Pooling within batches
            cluster_features = global_mean_pool(h, batch)
        else:
            # Global pooling
            cluster_features = h.mean(dim=0, keepdim=True).expand(h.size(0), -1)
        
        # 6. Solar recommendations
        # Combine node and cluster features
        combined = torch.cat([h, cluster_features], dim=-1)
        solar_scores = self.solar_head(combined).squeeze(-1)
        
        return {
            'cluster_logits': cluster_logits,
            'cluster_probs': cluster_probs,
            'solar_scores': solar_scores,
            'node_embeddings': h
        }
    
    def _apply_lv_constraints(
        self,
        h: torch.Tensor,
        lv_group_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply LV group constraints to ensure clustering within groups
        """
        # Project features through LV constraint layer
        h_constrained = self.lv_constraint(h)
        
        # Mask features based on LV group membership
        if lv_group_mask.dim() == 2:
            # lv_group_mask: [N, G] where G is number of LV groups
            # Apply group-wise normalization
            for g in range(lv_group_mask.size(1)):
                group_nodes = lv_group_mask[:, g] > 0
                if group_nodes.any():
                    h_constrained[group_nodes] = F.normalize(
                        h_constrained[group_nodes], p=2, dim=-1
                    )
        
        return h_constrained
    
    def get_solar_recommendations(
        self,
        solar_scores: torch.Tensor,
        cluster_probs: torch.Tensor,
        top_k: int = 10,
        min_cluster_confidence: float = 0.7
    ) -> Dict[str, torch.Tensor]:
        """
        Get top-k solar recommendations with cluster awareness
        
        Args:
            solar_scores: Solar suitability scores [N]
            cluster_probs: Cluster assignment probabilities [N, K]
            top_k: Number of buildings to recommend
            min_cluster_confidence: Minimum cluster confidence
            
        Returns:
            Dictionary with recommendations per cluster
        """
        # Find dominant cluster for each node
        max_probs, cluster_ids = cluster_probs.max(dim=1)
        
        # Filter by confidence
        confident_mask = max_probs > min_cluster_confidence
        
        recommendations = {}
        for c in range(self.num_clusters):
            cluster_mask = (cluster_ids == c) & confident_mask
            if cluster_mask.any():
                cluster_scores = solar_scores[cluster_mask]
                # Get top-k within cluster
                if len(cluster_scores) > 0:
                    top_values, top_indices = torch.topk(
                        cluster_scores, 
                        min(top_k, len(cluster_scores))
                    )
                    # Map back to original indices
                    original_indices = torch.where(cluster_mask)[0][top_indices]
                    recommendations[f'cluster_{c}'] = {
                        'indices': original_indices,
                        'scores': top_values
                    }
        
        return recommendations


class SemiSupervisedSolarGNN(SolarRecommendationGNN):
    """
    Extension with semi-supervised learning capabilities
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Label propagation for semi-supervised learning
        self.label_prop_steps = 5
        self.confidence_threshold = 0.8
        
    def propagate_labels(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        edge_index: torch.Tensor,
        num_steps: int = 5
    ) -> torch.Tensor:
        """
        Propagate labels from labeled to unlabeled nodes
        
        Args:
            predictions: Current predictions [N]
            labels: Ground truth labels [N] (-1 for unlabeled)
            mask: Mask for labeled nodes [N]
            edge_index: Graph connectivity
            num_steps: Number of propagation steps
            
        Returns:
            Pseudo-labels for all nodes
        """
        pseudo_labels = predictions.clone()
        
        # Initialize with known labels
        pseudo_labels[mask] = labels[mask].float()
        
        # Create adjacency matrix
        N = predictions.size(0)
        adj = torch.zeros(N, N, device=predictions.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        
        # Normalize adjacency matrix
        degree = adj.sum(dim=1, keepdim=True)
        degree[degree == 0] = 1.0
        adj_norm = adj / degree
        
        # Iterative label propagation
        for _ in range(num_steps):
            # Propagate
            pseudo_labels = torch.matmul(adj_norm, pseudo_labels.unsqueeze(-1)).squeeze(-1)
            # Reset known labels
            pseudo_labels[mask] = labels[mask].float()
            # Apply confidence threshold
            confident = (pseudo_labels > self.confidence_threshold) | (pseudo_labels < 1 - self.confidence_threshold)
            
        return pseudo_labels
    
    def compute_consistency_loss(
        self,
        predictions: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Consistency regularization: neighbors should have similar predictions
        """
        src, dst = edge_index
        pred_src = predictions[src]
        pred_dst = predictions[dst]
        
        # Smooth L1 loss between connected nodes
        consistency_loss = F.smooth_l1_loss(pred_src, pred_dst)
        
        return consistency_loss