"""
Simplified loss function for solar recommendation GNN
Focus: Self-sufficiency clustering and solar impact
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SolarRecommendationLoss(nn.Module):
    """
    Unified loss for solar recommendation with semi-supervised learning
    """
    
    def __init__(
        self,
        clustering_weight: float = 1.0,
        solar_weight: float = 2.0,
        physics_weight: float = 0.5,
        semi_supervised_weight: float = 0.3,
        consistency_weight: float = 0.2
    ):
        super().__init__()
        self.clustering_weight = clustering_weight
        self.solar_weight = solar_weight
        self.physics_weight = physics_weight
        self.semi_supervised_weight = semi_supervised_weight
        self.consistency_weight = consistency_weight
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: Optional[Dict[str, torch.Tensor]] = None,
        edge_index: Optional[torch.Tensor] = None,
        consumption_profiles: Optional[torch.Tensor] = None,
        lv_group_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total loss
        
        Args:
            outputs: Model outputs (cluster_probs, solar_scores, embeddings)
            labels: Ground truth labels (if available)
            edge_index: Graph connectivity
            consumption_profiles: Energy consumption profiles [N, T]
            lv_group_mask: LV group membership
            
        Returns:
            Total loss and loss components
        """
        losses = {}
        total_loss = 0.0
        
        # 1. Clustering Loss: Maximize self-sufficiency within clusters
        if 'cluster_probs' in outputs:
            cluster_loss = self._compute_clustering_loss(
                outputs['cluster_probs'],
                consumption_profiles,
                edge_index
            )
            losses['clustering'] = cluster_loss.item()
            total_loss += self.clustering_weight * cluster_loss
        
        # 2. Solar Loss: Supervised loss for labeled solar recommendations
        if labels is not None and 'solar_labels' in labels:
            mask = labels['solar_labels'] != -1  # -1 indicates unlabeled
            if mask.any():
                solar_loss = F.binary_cross_entropy(
                    outputs['solar_scores'][mask],
                    labels['solar_labels'][mask].float()
                )
                losses['solar_supervised'] = solar_loss.item()
                total_loss += self.solar_weight * solar_loss
        
        # 3. Physics Constraints: Respect LV group boundaries
        if lv_group_mask is not None:
            physics_loss = self._compute_physics_loss(
                outputs['cluster_probs'],
                lv_group_mask
            )
            losses['physics'] = physics_loss.item()
            total_loss += self.physics_weight * physics_loss
        
        # 4. Semi-supervised Loss: Pseudo-labeling and consistency
        if edge_index is not None:
            # Consistency regularization
            consistency_loss = self._compute_consistency_loss(
                outputs['solar_scores'],
                edge_index
            )
            losses['consistency'] = consistency_loss.item()
            total_loss += self.consistency_weight * consistency_loss
            
            # Pseudo-label loss (if confidence is high)
            if 'pseudo_labels' in labels:
                pseudo_loss = self._compute_pseudo_loss(
                    outputs['solar_scores'],
                    labels['pseudo_labels'],
                    labels.get('confidence', None)
                )
                losses['pseudo'] = pseudo_loss.item()
                total_loss += self.semi_supervised_weight * pseudo_loss
        
        return total_loss, losses
    
    def _compute_clustering_loss(
        self,
        cluster_probs: torch.Tensor,
        consumption_profiles: Optional[torch.Tensor],
        edge_index: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Clustering loss that promotes self-sufficiency
        
        Key idea: Buildings in same cluster should have complementary consumption
        (one produces when other consumes)
        """
        loss = 0.0
        
        # Get hard cluster assignments
        cluster_assignments = cluster_probs.argmax(dim=1)
        num_clusters = cluster_probs.size(1)
        
        if consumption_profiles is not None:
            # Compute complementarity within clusters
            for c in range(num_clusters):
                cluster_mask = cluster_assignments == c
                if cluster_mask.sum() > 1:
                    cluster_profiles = consumption_profiles[cluster_mask]
                    
                    # Calculate correlation matrix
                    profiles_norm = F.normalize(cluster_profiles, p=2, dim=1)
                    corr_matrix = torch.matmul(profiles_norm, profiles_norm.t())
                    
                    # We want negative correlation (complementarity)
                    # Penalize positive correlation
                    complementarity_loss = torch.relu(corr_matrix).mean()
                    loss += complementarity_loss
        
        # Add entropy regularization to avoid trivial solutions
        entropy = -(cluster_probs * torch.log(cluster_probs + 1e-8)).sum(dim=1).mean()
        loss -= 0.1 * entropy  # Encourage confident assignments
        
        return loss
    
    def _compute_physics_loss(
        self,
        cluster_probs: torch.Tensor,
        lv_group_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure clusters respect LV group boundaries
        
        Buildings in different LV groups should not be in same cluster
        """
        loss = 0.0
        
        # Get cluster similarity matrix
        cluster_sim = torch.matmul(cluster_probs, cluster_probs.t())
        
        # Get LV group similarity (1 if same group, 0 otherwise)
        if lv_group_mask.dim() == 2:
            # lv_group_mask: [N, G]
            lv_sim = torch.matmul(lv_group_mask, lv_group_mask.t())
            lv_sim = (lv_sim > 0).float()
        else:
            # lv_group_mask: [N] with group IDs
            lv_sim = (lv_group_mask.unsqueeze(0) == lv_group_mask.unsqueeze(1)).float()
        
        # Penalize high cluster similarity for different LV groups
        different_lv_mask = 1 - lv_sim
        physics_loss = (cluster_sim * different_lv_mask).mean()
        
        return physics_loss
    
    def _compute_consistency_loss(
        self,
        solar_scores: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Neighboring buildings should have similar solar potential
        (spatial smoothness assumption)
        """
        src, dst = edge_index
        score_diff = torch.abs(solar_scores[src] - solar_scores[dst])
        return score_diff.mean()
    
    def _compute_pseudo_loss(
        self,
        predictions: torch.Tensor,
        pseudo_labels: torch.Tensor,
        confidence: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Loss for pseudo-labeled data
        """
        if confidence is not None:
            # Weight by confidence
            weights = confidence
        else:
            # Use fixed threshold
            weights = (pseudo_labels > 0.8) | (pseudo_labels < 0.2)
            weights = weights.float()
        
        # Binary cross-entropy with confidence weighting
        loss = F.binary_cross_entropy(
            predictions,
            pseudo_labels,
            weight=weights,
            reduction='mean'
        )
        
        return loss


class SelfSufficiencyMetric(nn.Module):
    """
    Metric to evaluate cluster self-sufficiency
    Used for validation and model selection
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        cluster_assignments: torch.Tensor,
        production: torch.Tensor,
        consumption: torch.Tensor
    ) -> float:
        """
        Calculate self-sufficiency ratio for clusters
        
        Args:
            cluster_assignments: Cluster ID for each building [N]
            production: Solar production profiles [N, T]
            consumption: Energy consumption profiles [N, T]
            
        Returns:
            Average self-sufficiency across clusters
        """
        unique_clusters = cluster_assignments.unique()
        self_sufficiency_scores = []
        
        for cluster_id in unique_clusters:
            mask = cluster_assignments == cluster_id
            if mask.sum() > 0:
                cluster_production = production[mask].sum(dim=0)  # [T]
                cluster_consumption = consumption[mask].sum(dim=0)  # [T]
                
                # Self-sufficiency = min(production, consumption) / consumption
                # Averaged over time
                self_consumed = torch.minimum(cluster_production, cluster_consumption)
                self_sufficiency = (self_consumed / (cluster_consumption + 1e-8)).mean()
                
                self_sufficiency_scores.append(self_sufficiency.item())
        
        return sum(self_sufficiency_scores) / len(self_sufficiency_scores) if self_sufficiency_scores else 0.0