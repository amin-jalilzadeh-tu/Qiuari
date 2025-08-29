# models/pooling_layers.py
"""
Pooling layers for dynamic energy community discovery
Includes ConstrainedDiffPool with transformer boundary constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

# Try to import torch_scatter, fall back to manual implementation if not available
try:
    from torch_scatter import scatter_add, scatter_mean
except (ImportError, OSError):
    # Manual fallback implementations
    def scatter_add(src, index, dim=0, dim_size=None):
        """Manual scatter add implementation"""
        if dim_size is None:
            dim_size = int(index.max()) + 1
        out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
        index_expanded = index.view(-1, *([1] * (src.dim() - 1))).expand_as(src)
        out.scatter_add_(dim, index_expanded, src)
        return out
    
    def scatter_mean(src, index, dim=0, dim_size=None):
        """Manual scatter mean implementation"""
        if dim_size is None:
            dim_size = int(index.max()) + 1
        sum_out = scatter_add(src, index, dim, dim_size)
        count = scatter_add(torch.ones_like(src), index, dim, dim_size)
        return sum_out / count.clamp(min=1)


class ConstrainedDiffPool(nn.Module):
    """
    DiffPool with transformer boundary constraints
    Ensures clusters don't cross physical grid boundaries
    """
    
    def __init__(self, input_dim: int, max_clusters: int = 20, 
                 min_cluster_size: int = 3, max_cluster_size: int = 20):
        """
        Args:
            input_dim: Input feature dimension
            max_clusters: Maximum number of clusters to form
            min_cluster_size: Minimum buildings per cluster
            max_cluster_size: Maximum buildings per cluster
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.max_clusters = max_clusters
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
        # Assignment network - learns soft cluster assignments
        self.assign_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, max_clusters)
        )
        
        # Embedding network - learns cluster representations
        self.embed_net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, input_dim)
        )
        
        # Initialize assignment network with balanced outputs
        self._init_balanced_assignment()
    
    def _init_balanced_assignment(self):
        """Initialize the final layer of assignment network to encourage balanced clusters"""
        final_layer = self.assign_net[-1]
        if isinstance(final_layer, nn.Linear):
            # Initialize with small random weights to avoid strong initial preferences
            nn.init.normal_(final_layer.weight, mean=0.0, std=0.1)
            if final_layer.bias is not None:
                # Initialize bias to encourage uniform distribution across clusters
                nn.init.constant_(final_layer.bias, 0.0)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                batch: Optional[torch.Tensor] = None,
                transformer_mask: Optional[torch.Tensor] = None,
                lv_group_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass with LV group constraints
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
            batch: Batch assignment for multiple graphs
            transformer_mask: Binary mask for valid assignments [N, K] (legacy)
            lv_group_ids: LV group assignment for each node [N]
        
        Returns:
            x_pooled: Pooled node features [K, F]
            adj_pooled: Pooled adjacency [K, K]
            S: Soft assignment matrix [N, K]
            aux_loss: Auxiliary losses (link + entropy)
        """
        N = x.size(0)
        
        # Generate soft assignment matrix
        S_logits = self.assign_net(x)  # [N, K]
        
        # Apply LV group constraints if provided
        if lv_group_ids is not None:
            # Create mask that respects LV boundaries
            lv_mask = self._create_lv_mask(lv_group_ids)
            S_logits = S_logits.masked_fill(lv_mask == 0, -1e9)
        elif transformer_mask is not None:
            # Legacy: Apply transformer constraints if provided
            S_logits = S_logits.masked_fill(transformer_mask == 0, -1e9)
        
        # Add load balancing regularization to encourage uniform distribution
        S_logits = self._add_load_balancing_regularization(S_logits)
        
        # Apply softmax to get probabilities
        S = F.softmax(S_logits, dim=1)  # [N, K]
        
        # Apply cluster size constraints (iterative balancing)
        S = self._apply_size_constraints(S)
        
        # Apply hard rebalancing if needed
        S = self._hard_rebalance(S)
        
        # Generate new embeddings
        x_embed = self.embed_net(x)  # [N, F]
        
        # Pool features
        x_pooled = torch.matmul(S.T, x_embed)  # [K, F]
        
        # Pool adjacency matrix
        adj = self._edge_index_to_adj(edge_index, N)
        adj_pooled = torch.matmul(torch.matmul(S.T, adj), S)  # [K, K]
        
        # Calculate auxiliary losses
        link_loss = self._link_prediction_loss(adj, S)
        entropy_loss = self._entropy_loss(S)
        balance_loss = self._balance_loss(S)
        aux_loss = link_loss + 0.1 * entropy_loss + 0.5 * balance_loss
        
        return x_pooled, adj_pooled, S, aux_loss
    
    def _create_lv_mask(self, lv_group_ids: torch.Tensor) -> torch.Tensor:
        """
        Create mask ensuring clusters respect LV boundaries but allow multiple clusters per LV
        
        Args:
            lv_group_ids: LV group ID for each node [N]
            
        Returns:
            Mask tensor [N, K] where 1 = allowed, 0 = forbidden
        """
        N = lv_group_ids.size(0)
        device = lv_group_ids.device
        
        # Get unique LV groups
        unique_lvs = torch.unique(lv_group_ids)
        num_lvs = len(unique_lvs)
        
        # Allow more clusters per LV group for flexibility
        min_clusters_per_lv = 1
        max_clusters_per_lv = max(2, self.max_clusters // max(1, num_lvs - 5))  # Allow overlap
        
        # Create mask (start with all allowed)
        mask = torch.ones(N, self.max_clusters, device=device)
        
        # For very large LV groups, enforce some boundaries
        for i, lv_id in enumerate(unique_lvs):
            lv_mask = (lv_group_ids == lv_id)
            lv_size = lv_mask.sum().item()
            
            # Only apply strict boundaries for very large LV groups
            if lv_size > 30:  # Large LV groups get some restrictions
                # Assign a preferred cluster range but don't forbid others completely
                start_cluster = (i * 2) % self.max_clusters
                end_cluster = min(start_cluster + max_clusters_per_lv, self.max_clusters)
                
                # Reduce probability for non-preferred clusters rather than forbidding
                preferred_mask = torch.zeros(self.max_clusters, device=device)
                preferred_mask[start_cluster:end_cluster] = 1.0
                
                # Apply soft constraint (0.3 for non-preferred, 1.0 for preferred)
                constraint = 0.3 + 0.7 * preferred_mask
                mask[lv_mask] = mask[lv_mask] * constraint.unsqueeze(0)
        
        return mask
    
    def _apply_size_constraints(self, S: torch.Tensor) -> torch.Tensor:
        """
        Apply min/max size constraints to clusters using stronger penalties
        
        Args:
            S: Soft assignment matrix [N, K]
        
        Returns:
            Constrained assignment matrix
        """
        # Calculate current cluster sizes (expected counts)
        cluster_sizes = torch.sum(S, dim=0)  # [K]
        
        # Create penalty based on size violations
        size_penalty = torch.zeros_like(cluster_sizes)
        
        # Strong penalty for oversized clusters
        oversized_mask = cluster_sizes > self.max_cluster_size
        size_penalty[oversized_mask] = -torch.log(cluster_sizes[oversized_mask] / self.max_cluster_size + 1e-8)
        
        # Moderate penalty for undersized clusters (but allow some small clusters)
        undersized_mask = (cluster_sizes > 0) & (cluster_sizes < self.min_cluster_size)
        size_penalty[undersized_mask] = -0.5 * torch.log(cluster_sizes[undersized_mask] / self.min_cluster_size + 1e-8)
        
        # Apply penalty to logits before softmax
        if torch.any(size_penalty != 0):
            # Convert S back to logits, apply penalty, then re-normalize
            S_logits = torch.log(S + 1e-8)
            S_logits = S_logits + size_penalty.unsqueeze(0)  # Add penalty to logits
            S_adjusted = F.softmax(S_logits, dim=1)
            
            # Additional hard constraint: if cluster is very oversized, redistribute
            very_oversized = cluster_sizes > self.max_cluster_size * 1.5
            if torch.any(very_oversized):
                for k in torch.where(very_oversized)[0]:
                    # Find nodes most weakly assigned to this cluster
                    cluster_probs = S_adjusted[:, k]
                    threshold = torch.quantile(cluster_probs, 0.7)  # Keep top 70% assignments
                    
                    # Redistribute bottom 30% to other clusters
                    weak_assignments = cluster_probs < threshold
                    if weak_assignments.any():
                        # Zero out weak assignments to oversized cluster
                        S_adjusted[weak_assignments, k] = 0.0
                        # Renormalize rows
                        S_adjusted[weak_assignments] = F.normalize(S_adjusted[weak_assignments], p=1, dim=1)
            
            return S_adjusted
        
        return S
    
    def _edge_index_to_adj(self, edge_index: torch.Tensor, N: int) -> torch.Tensor:
        """
        Convert edge index to adjacency matrix
        
        Args:
            edge_index: Edge indices [2, E]
            N: Number of nodes
        
        Returns:
            Adjacency matrix [N, N]
        """
        adj = torch.zeros((N, N), device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1.0
        return adj
    
    def _link_prediction_loss(self, adj: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """
        Link prediction loss to preserve graph structure
        
        Args:
            adj: Original adjacency matrix [N, N]
            S: Soft assignment matrix [N, K]
        
        Returns:
            Link prediction loss
        """
        # Predict links based on cluster assignments
        S_prod = torch.matmul(S, S.T)  # [N, N]
        
        # Binary cross entropy between original and predicted
        link_loss = F.binary_cross_entropy(
            S_prod.view(-1),
            adj.view(-1),
            reduction='mean'
        )
        
        return link_loss
    
    def _entropy_loss(self, S: torch.Tensor) -> torch.Tensor:
        """
        Entropy regularization for crisp assignments
        
        Args:
            S: Soft assignment matrix [N, K]
        
        Returns:
            Entropy loss (lower = more crisp)
        """
        entropy = -torch.mean(torch.sum(S * torch.log(S + 1e-8), dim=1))
        return entropy
    
    def _balance_loss(self, S: torch.Tensor) -> torch.Tensor:
        """
        Balance loss to encourage even cluster sizes
        
        Args:
            S: Soft assignment matrix [N, K]
            
        Returns:
            Balance loss (lower = more balanced)
        """
        N, K = S.shape
        target_size = N / K  # Ideal cluster size
        
        # Calculate actual cluster sizes
        cluster_sizes = S.sum(dim=0)  # [K]
        
        # Penalize deviation from target size
        size_deviations = torch.abs(cluster_sizes - target_size)
        balance_loss = size_deviations.mean() / target_size
        
        return balance_loss
    
    def _hard_rebalance(self, S: torch.Tensor) -> torch.Tensor:
        """
        Hard rebalancing to enforce cluster size limits more aggressively
        
        Args:
            S: Soft assignment matrix [N, K]
            
        Returns:
            Rebalanced assignment matrix
        """
        N, K = S.shape
        max_iterations = 5
        
        for iteration in range(max_iterations):
            cluster_sizes = S.sum(dim=0)
            
            # Check if any cluster is oversized (be more strict)
            very_oversized = cluster_sizes > self.max_cluster_size
            
            if not torch.any(very_oversized):
                break
            
            # For each oversized cluster, find weakest members and redistribute
            for k in torch.where(very_oversized)[0]:
                excess = cluster_sizes[k] - self.max_cluster_size
                if excess <= 0:
                    continue
                
                # Find nodes most weakly assigned to this cluster
                cluster_probs = S[:, k]
                
                # Get indices of weakest assignments (take more aggressive approach)
                num_to_redistribute = min(int(excess.ceil().item()) + 5, len(cluster_probs))
                _, weak_indices = torch.topk(cluster_probs, 
                                           num_to_redistribute, 
                                           largest=False)
                
                # Zero out assignments to oversized cluster
                S[weak_indices, k] = 1e-8
                
                # Redistribute to other clusters based on their current capacity
                available_capacity = torch.clamp(
                    self.max_cluster_size - cluster_sizes, 
                    min=0
                )
                
                # Normalize available capacity to get redistribution weights
                redistrib_weights = available_capacity / (available_capacity.sum() + 1e-8)
                
                # Redistribute the weak assignments
                for i, node_idx in enumerate(weak_indices):
                    S[node_idx] = S[node_idx] + redistrib_weights * S[node_idx, k]
                
                # Renormalize the affected rows
                S[weak_indices] = F.normalize(S[weak_indices], p=1, dim=1)
        
        return S
    
    def _add_load_balancing_regularization(self, S_logits: torch.Tensor) -> torch.Tensor:
        """
        Add load balancing regularization to logits to encourage uniform cluster sizes
        
        Args:
            S_logits: Assignment logits [N, K]
            
        Returns:
            Regularized logits
        """
        N, K = S_logits.shape
        target_size = N / K
        
        # Get current soft cluster sizes
        current_probs = F.softmax(S_logits, dim=1)
        current_sizes = current_probs.sum(dim=0)  # [K]
        
        # Calculate load balancing penalty
        # Penalize clusters that are over-represented
        load_penalties = torch.log(current_sizes / target_size + 1e-8) * 1.0  # Increased from 0.5
        
        # Apply penalty to discourage assignment to over-represented clusters
        S_logits = S_logits - load_penalties.unsqueeze(0)
        
        return S_logits


class TransformerConstrainedPooling(nn.Module):
    """
    Specialized pooling that strictly respects transformer boundaries
    Uses hard constraints rather than soft penalties
    """
    
    def __init__(self, input_dim: int, max_clusters_per_transformer: int = 5):
        """
        Args:
            input_dim: Input feature dimension
            max_clusters_per_transformer: Max clusters within each transformer region
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.max_clusters_per_transformer = max_clusters_per_transformer
        
        # Local assignment networks (per transformer region)
        self.local_assign = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Linear(input_dim // 2, max_clusters_per_transformer)
        )
        
    def forward(self, x: torch.Tensor, transformer_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pool nodes within transformer boundaries
        
        Args:
            x: Node features [N, F]
            transformer_ids: Transformer assignment for each node [N]
        
        Returns:
            S: Global assignment matrix [N, total_clusters]
            cluster_ids: Cluster ID for each node [N]
        """
        device = x.device
        N = x.size(0)
        unique_transformers = torch.unique(transformer_ids)
        
        # Initialize global assignment matrix
        total_clusters = len(unique_transformers) * self.max_clusters_per_transformer
        S = torch.zeros((N, total_clusters), device=device)
        
        # Process each transformer region independently
        offset = 0
        for t_id in unique_transformers:
            # Get nodes in this transformer region
            mask = (transformer_ids == t_id)
            indices = torch.where(mask)[0]
            
            if len(indices) == 0:
                continue
            
            # Local features
            x_local = x[indices]
            
            # Local assignment
            S_local = self.local_assign(x_local)
            S_local = F.softmax(S_local, dim=1)
            
            # Map to global assignment matrix
            for i, idx in enumerate(indices):
                S[idx, offset:offset + self.max_clusters_per_transformer] = S_local[i]
            
            offset += self.max_clusters_per_transformer
        
        # Get hard cluster assignments
        cluster_ids = torch.argmax(S, dim=1)
        
        return S, cluster_ids


class AdaptivePooling(nn.Module):
    """
    Adaptive pooling that adjusts number of clusters based on data
    """
    
    def __init__(self, input_dim: int, min_clusters: int = 3, max_clusters: int = 20):
        """
        Args:
            input_dim: Input feature dimension
            min_clusters: Minimum number of clusters
            max_clusters: Maximum number of clusters
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        
        # Network to predict optimal number of clusters
        self.cluster_predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Assignment networks for different cluster numbers
        self.assign_nets = nn.ModuleDict({
            str(k): nn.Linear(input_dim, k)
            for k in range(min_clusters, max_clusters + 1)
        })
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Adaptively pool nodes
        
        Args:
            x: Node features [N, F]
            edge_index: Edge indices [2, E]
        
        Returns:
            S: Soft assignment matrix [N, K]
            num_clusters: Predicted optimal number of clusters
        """
        # Aggregate features to predict cluster count
        x_agg = torch.mean(x, dim=0, keepdim=True)
        
        # Predict normalized cluster count
        cluster_ratio = self.cluster_predictor(x_agg).item()
        
        # Convert to actual cluster count
        num_clusters = int(
            self.min_clusters + cluster_ratio * (self.max_clusters - self.min_clusters)
        )
        num_clusters = max(self.min_clusters, min(self.max_clusters, num_clusters))
        
        # Get appropriate assignment network
        S = self.assign_nets[str(num_clusters)](x)
        S = F.softmax(S, dim=1)
        
        return S, num_clusters


class HierarchicalPooling(nn.Module):
    """
    Multi-level hierarchical pooling following grid structure
    Building -> Cable Group -> Transformer
    """
    
    def __init__(self, building_dim: int, cable_dim: int, transformer_dim: int):
        """
        Args:
            building_dim: Building feature dimension
            cable_dim: Cable group feature dimension
            transformer_dim: Transformer feature dimension
        """
        super().__init__()
        
        # Building to cable group pooling
        self.building_to_cable = nn.Sequential(
            nn.Linear(building_dim, cable_dim),
            nn.ReLU(),
            nn.Linear(cable_dim, cable_dim)
        )
        
        # Cable group to transformer pooling
        self.cable_to_transformer = nn.Sequential(
            nn.Linear(cable_dim, transformer_dim),
            nn.ReLU(),
            nn.Linear(transformer_dim, transformer_dim)
        )
        
    def forward(self, x_dict: Dict[str, torch.Tensor], 
                hierarchy: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Hierarchical pooling through grid levels
        
        Args:
            x_dict: Features by node type
            hierarchy: Hierarchical relationships
        
        Returns:
            Pooled features at each level
        """
        pooled = {}
        
        # Pool buildings to cable groups
        if 'building' in x_dict and 'building_to_cable' in hierarchy:
            assignment = hierarchy['building_to_cable']
            x_cable = scatter_mean(
                self.building_to_cable(x_dict['building']),
                assignment,
                dim=0
            )
            pooled['cable_group'] = x_cable
        
        # Pool cable groups to transformers
        if 'cable_group' in pooled and 'cable_to_transformer' in hierarchy:
            assignment = hierarchy['cable_to_transformer']
            x_transformer = scatter_mean(
                self.cable_to_transformer(pooled['cable_group']),
                assignment,
                dim=0
            )
            pooled['transformer'] = x_transformer
        
        return pooled


if __name__ == "__main__":
    # Test ConstrainedDiffPool
    print("Testing ConstrainedDiffPool...")
    
    # Create dummy data
    N = 100  # Number of buildings
    F = 128  # Feature dimension
    K = 10   # Number of clusters
    
    x = torch.randn(N, F)
    edge_index = torch.randint(0, N, (2, 500))
    
    # Create transformer mask (simulate 3 transformer regions)
    transformer_mask = torch.ones(N, K)
    # Block cross-transformer assignments
    transformer_mask[:33, 4:] = 0  # Transformer 1 can only use clusters 0-3
    transformer_mask[33:66, :4] = 0  # Transformer 2 can only use clusters 4-7
    transformer_mask[33:66, 8:] = 0
    transformer_mask[66:, :8] = 0  # Transformer 3 can only use clusters 8-9
    
    # Initialize and test
    pool = ConstrainedDiffPool(F, max_clusters=K)
    x_pooled, adj_pooled, S, aux_loss = pool(x, edge_index, transformer_mask=transformer_mask)
    
    print(f"Input shape: {x.shape}")
    print(f"Pooled shape: {x_pooled.shape}")
    print(f"Assignment shape: {S.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.4f}")
    
    # Verify constraints
    cluster_sizes = torch.sum(S, dim=0)
    print(f"Cluster sizes: {cluster_sizes.tolist()}")
    
    # Check transformer constraints
    for i in range(3):
        start = i * 33
        end = min((i + 1) * 33, N)
        node_assignments = S[start:end].sum(dim=0)
        print(f"Transformer {i+1} uses clusters: {torch.where(node_assignments > 0.01)[0].tolist()}")
    
    print("\n✅ ConstrainedDiffPool test successful!")