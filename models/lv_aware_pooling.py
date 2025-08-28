# models/lv_aware_pooling.py
"""
LV-aware pooling layers with dynamic cluster discovery
Enforces hard constraints and discovers optimal cluster count per LV group
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
import numpy as np

class LVAwareDiffPool(nn.Module):
    """
    Dynamic clustering that:
    1. Discovers optimal cluster count per LV group (not fixed)
    2. Enforces hard LV boundary constraints
    3. Maintains cluster size limits (3-20 buildings)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 min_cluster_size: int = 3, max_cluster_size: int = 20):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        
        # Cluster count predictor - learns optimal K per LV group
        self.cluster_count_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Outputs single value per LV
            nn.Sigmoid()  # Between 0 and 1
        )
        
        # Assignment network - adaptive to cluster count
        self.assign_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, max_cluster_size)  # Max possible clusters per LV
        )
        
        # Quality predictor - estimates cluster quality
        self.quality_net = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def create_lv_boundary_mask(self, lv_group_ids: torch.Tensor, 
                                num_clusters_per_lv: Dict[int, int]) -> torch.Tensor:
        """
        Create mask that enforces LV boundaries
        
        Args:
            lv_group_ids: LV group ID for each building [N]
            num_clusters_per_lv: Number of clusters for each LV group
            
        Returns:
            mask: [N, total_clusters] with -inf for invalid assignments
        """
        N = lv_group_ids.size(0)
        unique_lvs = torch.unique(lv_group_ids).tolist()
        
        # Calculate total clusters and their LV assignments
        cluster_to_lv = {}
        cluster_offset = 0
        for lv_id in sorted(unique_lvs):
            n_clusters = num_clusters_per_lv.get(lv_id, 5)  # Default 5 clusters
            for c in range(n_clusters):
                cluster_to_lv[cluster_offset + c] = lv_id
            cluster_offset += n_clusters
        
        total_clusters = cluster_offset
        
        # Create mask
        mask = torch.full((N, total_clusters), float('-inf'), device=lv_group_ids.device)
        
        for building_idx in range(N):
            building_lv = lv_group_ids[building_idx].item()
            for cluster_idx, cluster_lv in cluster_to_lv.items():
                if building_lv == cluster_lv:
                    mask[building_idx, cluster_idx] = 0.0  # Valid assignment
        
        return mask
    
    def discover_cluster_counts(self, x: torch.Tensor, lv_group_ids: torch.Tensor) -> Dict[int, int]:
        """
        Discover optimal number of clusters for each LV group
        
        Args:
            x: Building features [N, F]
            lv_group_ids: LV group for each building [N]
            
        Returns:
            Dictionary mapping LV ID to optimal cluster count
        """
        unique_lvs = torch.unique(lv_group_ids)
        cluster_counts = {}
        
        for lv_id in unique_lvs:
            # Get buildings in this LV group
            lv_mask = (lv_group_ids == lv_id)
            lv_buildings = x[lv_mask]
            n_buildings = lv_buildings.size(0)
            
            if n_buildings < self.min_cluster_size:
                # Too few buildings for any clusters
                cluster_counts[lv_id.item()] = 0
                continue
            
            # Aggregate LV features
            lv_features = lv_buildings.mean(dim=0, keepdim=True)
            
            # Predict optimal cluster ratio (0 to 1)
            cluster_ratio = self.cluster_count_net(lv_features).item()
            
            # Convert to actual count
            # More buildings = potentially more clusters
            max_possible = min(n_buildings // self.min_cluster_size, self.max_cluster_size)
            min_possible = max(1, n_buildings // self.max_cluster_size)
            
            # Scale ratio to actual count
            optimal_k = int(min_possible + cluster_ratio * (max_possible - min_possible))
            optimal_k = max(min_possible, min(optimal_k, max_possible))
            
            cluster_counts[lv_id.item()] = optimal_k
        
        return cluster_counts
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                lv_group_ids: torch.Tensor, 
                batch: Optional[torch.Tensor] = None) -> Dict:
        """
        Forward pass with dynamic clustering per LV group
        
        Args:
            x: Building features [N, F]
            edge_index: Building connections [2, E]
            lv_group_ids: LV group for each building [N]
            batch: Batch assignment (for multiple graphs)
            
        Returns:
            Dictionary with:
                - cluster_assignments: Hard cluster assignments [N]
                - soft_assignments: Soft assignment matrix [N, K]
                - cluster_counts: Clusters per LV group
                - cluster_features: Pooled features [K, F]
                - quality_scores: Predicted quality per cluster
        """
        N = x.size(0)
        device = x.device
        
        # Step 1: Discover optimal cluster counts per LV
        cluster_counts = self.discover_cluster_counts(x, lv_group_ids)
        total_clusters = sum(cluster_counts.values())
        
        if total_clusters == 0:
            # No valid clusters possible
            return {
                'cluster_assignments': torch.zeros(N, dtype=torch.long, device=device),
                'soft_assignments': torch.zeros(N, 1, device=device),
                'cluster_counts': cluster_counts,
                'cluster_features': x.mean(dim=0, keepdim=True),
                'quality_scores': torch.zeros(1, device=device)
            }
        
        # Step 2: Generate assignment logits
        assignment_logits = self.assign_net(x)  # [N, max_clusters]
        
        # Step 3: Create and apply LV boundary mask
        lv_mask = self.create_lv_boundary_mask(lv_group_ids, cluster_counts)
        
        # Trim logits to actual cluster count
        assignment_logits = assignment_logits[:, :total_clusters]
        masked_logits = assignment_logits + lv_mask
        
        # Step 4: Apply softmax to get assignments
        soft_assignments = F.softmax(masked_logits, dim=1)  # [N, K]
        
        # Step 5: Enforce cluster size constraints
        cluster_sizes = soft_assignments.sum(dim=0)  # [K]
        
        # Penalize too-small clusters by redistributing
        for cluster_idx in range(total_clusters):
            if cluster_sizes[cluster_idx] < self.min_cluster_size:
                # Find which LV this cluster belongs to
                cluster_lv = None
                offset = 0
                for lv_id, count in cluster_counts.items():
                    if cluster_idx < offset + count:
                        cluster_lv = lv_id
                        break
                    offset += count
                
                if cluster_lv is not None:
                    # Redistribute to other clusters in same LV
                    lv_mask_bool = (lv_group_ids == cluster_lv)
                    soft_assignments[lv_mask_bool, cluster_idx] *= 0.1  # Reduce probability
        
        # Step 6: Get hard assignments
        cluster_assignments = soft_assignments.argmax(dim=1)
        
        # Step 7: Pool features by cluster
        cluster_features = torch.zeros(total_clusters, x.size(1), device=device)
        for c in range(total_clusters):
            cluster_mask = (cluster_assignments == c)
            if cluster_mask.any():
                cluster_features[c] = x[cluster_mask].mean(dim=0)
            else:
                # Empty cluster - use mean of LV group
                # Find LV for this cluster
                offset = 0
                for lv_id, count in cluster_counts.items():
                    if c < offset + count:
                        lv_mask = (lv_group_ids == lv_id)
                        if lv_mask.any():
                            cluster_features[c] = x[lv_mask].mean(dim=0)
                        break
                    offset += count
        
        # Step 8: Predict cluster quality
        quality_scores = []
        for c in range(total_clusters):
            cluster_mask = (cluster_assignments == c)
            if cluster_mask.sum() >= 2:
                # Get pairwise features for quality estimation
                cluster_buildings = x[cluster_mask]
                mean_features = cluster_buildings.mean(dim=0)
                std_features = cluster_buildings.std(dim=0)
                quality_input = torch.cat([mean_features, std_features])
                quality = self.quality_net(quality_input.unsqueeze(0))
                quality_scores.append(quality.squeeze())
            else:
                quality_scores.append(torch.tensor(0.0, device=device))
        
        quality_scores = torch.stack(quality_scores) if quality_scores else torch.zeros(1, device=device)
        
        return {
            'cluster_assignments': cluster_assignments,
            'soft_assignments': soft_assignments,
            'cluster_counts': cluster_counts,
            'cluster_features': cluster_features,
            'quality_scores': quality_scores,
            'total_clusters': total_clusters
        }
    
    def compute_auxiliary_loss(self, soft_assignments: torch.Tensor,
                               cluster_counts: Dict[int, int],
                               lv_group_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary losses for training
        
        Returns:
            Combined auxiliary loss
        """
        losses = []
        
        # Entropy regularization for crisp assignments
        entropy = -torch.mean(soft_assignments * torch.log(soft_assignments + 1e-8))
        losses.append(entropy * 0.1)
        
        # Size constraint loss
        cluster_sizes = soft_assignments.sum(dim=0)
        size_penalties = torch.relu(self.min_cluster_size - cluster_sizes) + \
                        torch.relu(cluster_sizes - self.max_cluster_size)
        losses.append(size_penalties.mean())
        
        # Coverage loss - ensure all buildings are assigned
        max_assignment = soft_assignments.max(dim=1)[0]
        orphan_penalty = torch.relu(0.5 - max_assignment).mean()
        losses.append(orphan_penalty)
        
        return sum(losses)