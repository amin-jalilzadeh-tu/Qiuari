"""
Task heads for Energy GNN - Focused on Clustering and Pattern Discovery
Simplified to remove intervention prediction, focusing on complementarity discovery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from typing import Dict, Tuple, Optional

# Try to import torch_scatter, fall back to manual implementation if not available
try:
    from torch_scatter import scatter_add
except (ImportError, OSError):
    def scatter_add(src, index, dim=0, dim_size=None):
        """Manual scatter add implementation"""
        if dim_size is None:
            dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        if index.numel() > 0:
            index_expanded = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
            out.scatter_add_(dim, index_expanded, src)
        return out


class ClusteringHead(nn.Module):
    """
    Clustering head that discovers energy communities with complementary patterns
    Outputs soft cluster assignments and complementarity scores
    """
    
    def __init__(
        self,
        input_dim: int,
        num_clusters: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        
        self.num_clusters = num_clusters
        self.temperature = temperature
        
        # Cluster assignment network
        self.cluster_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_clusters)
        )
        
        # Complementarity scoring network
        self.complementarity_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # Output between -1 (perfect complementarity) and 1 (perfect correlation)
        )
        
        # Cluster prototype embeddings (learnable)
        self.cluster_prototypes = nn.Parameter(
            torch.randn(num_clusters, input_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for clustering
        
        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Edge connectivity
            batch: Batch assignment for nodes
            
        Returns:
            Dictionary containing:
                - cluster_assignments: Soft cluster assignments [num_nodes, num_clusters]
                - cluster_probs: Normalized cluster probabilities
                - complementarity_matrix: Pairwise complementarity scores
                - cluster_centers: Learned cluster prototypes
        """
        
        # Get cluster assignments (soft)
        cluster_logits = self.cluster_net(x)
        cluster_probs = F.softmax(cluster_logits / self.temperature, dim=-1)
        
        # Calculate pairwise complementarity scores for connected nodes
        complementarity_scores = self._calculate_complementarity(x, edge_index)
        
        # Calculate cluster centers based on soft assignments
        cluster_centers = self._update_cluster_centers(x, cluster_probs)
        
        # Create complementarity matrix for all pairs (for analysis)
        complementarity_matrix = self._build_complementarity_matrix(x)
        
        return {
            'cluster_assignments': cluster_logits,
            'cluster_probs': cluster_probs,
            'complementarity_scores': complementarity_scores,
            'complementarity_matrix': complementarity_matrix,
            'cluster_centers': cluster_centers,
            'cluster_prototypes': self.cluster_prototypes
        }
    
    def _calculate_complementarity(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate complementarity scores for edges"""
        row, col = edge_index
        
        # Concatenate features of connected nodes
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        
        # Get complementarity score (negative = complementary, positive = similar)
        scores = self.complementarity_net(edge_features).squeeze(-1)
        
        return scores
    
    def _build_complementarity_matrix(
        self,
        x: torch.Tensor,
        max_nodes: int = 1000
    ) -> Optional[torch.Tensor]:
        """Build full complementarity matrix for analysis (limited size for memory)"""
        num_nodes = x.size(0)
        
        if num_nodes > max_nodes:
            return None
            
        # Expand features for pairwise comparison
        x_i = x.unsqueeze(1).expand(-1, num_nodes, -1)
        x_j = x.unsqueeze(0).expand(num_nodes, -1, -1)
        
        # Concatenate pairs
        pairs = torch.cat([x_i, x_j], dim=-1)
        
        # Calculate complementarity for all pairs
        matrix = self.complementarity_net(pairs.view(-1, pairs.size(-1)))
        matrix = matrix.view(num_nodes, num_nodes)
        
        return matrix
    
    def _update_cluster_centers(
        self,
        x: torch.Tensor,
        cluster_probs: torch.Tensor
    ) -> torch.Tensor:
        """Update cluster centers based on soft assignments"""
        # Weighted average of node features
        weighted_sum = torch.matmul(cluster_probs.t(), x)
        
        # Normalize by total weight
        weights_sum = cluster_probs.sum(dim=0, keepdim=True).t() + 1e-8
        cluster_centers = weighted_sum / weights_sum
        
        return cluster_centers


class EnergyPredictionHead(nn.Module):
    """
    Energy prediction head for validation and pattern analysis
    Predicts energy metrics to validate clustering quality
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 4,  # [peak_demand, total_consumption, self_sufficiency, peak_time]
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Separate heads for different metrics
        self.peak_head = nn.Linear(output_dim, 1)
        self.consumption_head = nn.Linear(output_dim, 1)
        self.self_sufficiency_head = nn.Linear(output_dim, 1)
        self.peak_time_head = nn.Linear(output_dim, 24)  # 24 hours
        
    def forward(
        self,
        x: torch.Tensor,
        cluster_probs: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict energy metrics for clusters
        
        Args:
            x: Node features or cluster features
            cluster_probs: Cluster assignment probabilities
            batch: Batch assignment
            
        Returns:
            Dictionary of energy predictions
        """
        
        # Get base predictions
        features = self.predictor(x)
        
        # Predict different metrics
        peak_demand = self.peak_head(features)
        total_consumption = self.consumption_head(features)
        self_sufficiency = torch.sigmoid(self.self_sufficiency_head(features))
        peak_time_logits = self.peak_time_head(features)
        
        return {
            'peak_demand': peak_demand,
            'total_consumption': total_consumption,
            'self_sufficiency': self_sufficiency,
            'peak_time_probs': F.softmax(peak_time_logits, dim=-1),
            'peak_time_logits': peak_time_logits
        }


class NetworkImportanceHead(nn.Module):
    """
    Network importance scoring head
    Identifies critical nodes for intervention planning
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.importance_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # [centrality, cascade_potential, intervention_priority]
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate network importance scores
        
        Args:
            x: Node features
            
        Returns:
            Dictionary of importance scores
        """
        scores = self.importance_net(x)
        
        return {
            'centrality_score': torch.sigmoid(scores[:, 0]),
            'cascade_potential': torch.sigmoid(scores[:, 1]),
            'intervention_priority': torch.sigmoid(scores[:, 2])
        }


class UnifiedTaskHead(nn.Module):
    """
    Unified task head combining clustering, energy prediction, and network analysis
    Simplified from multi-task to focus on complementarity discovery
    """
    
    def __init__(
        self,
        input_dim: int,
        num_clusters: int = 10,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        temperature: float = 1.0
    ):
        super().__init__()
        
        # Main task: Clustering with complementarity
        self.clustering_head = ClusteringHead(
            input_dim=input_dim,
            num_clusters=num_clusters,
            hidden_dim=hidden_dim,
            dropout=dropout,
            temperature=temperature
        )
        
        # Validation task: Energy prediction
        self.energy_head = EnergyPredictionHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # Analysis task: Network importance
        self.network_head = NetworkImportanceHead(
            input_dim=input_dim,
            hidden_dim=hidden_dim // 2
        )
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        return_all: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through all task heads
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            batch: Batch assignment
            return_all: Whether to return all head outputs
            
        Returns:
            Dictionary containing all task outputs
        """
        outputs = {}
        
        # Primary task: Clustering
        clustering_output = self.clustering_head(x, edge_index, batch)
        outputs.update({f'clustering_{k}': v for k, v in clustering_output.items()})
        
        if return_all:
            # Energy predictions for validation
            energy_output = self.energy_head(
                x,
                cluster_probs=clustering_output['cluster_probs'],
                batch=batch
            )
            outputs.update({f'energy_{k}': v for k, v in energy_output.items()})
            
            # Network importance for planning
            network_output = self.network_head(x)
            outputs.update({f'network_{k}': v for k, v in network_output.items()})
        
        return outputs
    
    def get_cluster_assignments(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        hard: bool = False
    ) -> torch.Tensor:
        """
        Get cluster assignments (hard or soft)
        
        Args:
            x: Node features
            edge_index: Edge connectivity
            hard: Whether to return hard assignments
            
        Returns:
            Cluster assignments
        """
        output = self.clustering_head(x, edge_index)
        
        if hard:
            return torch.argmax(output['cluster_assignments'], dim=-1)
        else:
            return output['cluster_probs']
    
    def get_complementarity_scores(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Get complementarity scores for edges"""
        output = self.clustering_head(x, edge_index)
        return output['complementarity_scores']
    
    def get_network_importance(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get network importance scores for planning"""
        return self.network_head(x)


class ComplementarityScoreHead(nn.Module):
    """
    Calculates pairwise complementarity scores between buildings
    Output: NxN correlation matrix (-1 to 1, negative = complementary)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Calculate complementarity scores
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            
        Returns:
            Correlation matrix [N, N]
        """
        h = self.embed(x)
        # Normalize embeddings
        h_norm = F.normalize(h, p=2, dim=1)
        # Compute correlation matrix
        correlation = torch.matmul(h_norm, h_norm.T)
        # Apply mask for connected nodes only
        mask = torch.zeros_like(correlation)
        mask[edge_index[0], edge_index[1]] = 1
        return correlation * mask


class NetworkCentralityHead(nn.Module):
    """
    Calculates network importance scores for buildings
    Used for intervention targeting
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.centrality_net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),  # +1 for degree feature
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                cluster_assignment: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate centrality scores
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            cluster_assignment: Optional cluster assignments [N, K]
            
        Returns:
            Centrality scores [N]
        """
        # Calculate degree centrality
        degree = scatter_add(
            torch.ones(edge_index.size(1), device=x.device),
            edge_index[0],
            dim=0,
            dim_size=x.size(0)
        )
        degree = degree.unsqueeze(1) / edge_index.size(1)
        
        # Combine with learned features
        features = torch.cat([x, degree], dim=1)
        centrality = self.centrality_net(features)
        
        return centrality.squeeze()


class EnergyFlowHead(nn.Module):
    """
    Predicts building-to-building energy flows
    Output: Edge weights representing energy transfer (kW)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        self.flow_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Predict energy flows
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            
        Returns:
            Flow magnitudes [E]
        """
        # Get source and target node features
        src = x[edge_index[0]]
        dst = x[edge_index[1]]
        
        # Concatenate features
        edge_features = torch.cat([src, dst], dim=1)
        
        # Predict flow magnitude
        flows = self.flow_net(edge_features)
        
        return flows.squeeze()