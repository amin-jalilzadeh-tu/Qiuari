"""
Balanced Multi-Objective Loss for Energy Communities
Prevents cluster collapse by balancing competing objectives
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class BalancedEnergyLoss(nn.Module):
    """
    Balanced loss that prevents cluster collapse while optimizing energy objectives
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Target number of clusters (prevent collapse)
        self.min_clusters = config.get('min_clusters', 4)
        self.max_clusters = config.get('max_clusters', 8)
        self.target_cluster_size = config.get('target_cluster_size', 20)  # buildings per cluster
        
        # Loss weights (normalized to sum to 1)
        weights = config.get('loss_weights', {})
        self.w_clustering = weights.get('clustering', 0.3)
        self.w_diversity = weights.get('diversity', 0.2)
        self.w_solar = weights.get('solar', 0.2)
        self.w_balance = weights.get('balance', 0.2)
        self.w_physics = weights.get('physics', 0.1)
        
        # Normalize weights
        total_weight = sum([self.w_clustering, self.w_diversity, self.w_solar, 
                           self.w_balance, self.w_physics])
        self.w_clustering /= total_weight
        self.w_diversity /= total_weight
        self.w_solar /= total_weight
        self.w_balance /= total_weight
        self.w_physics /= total_weight
        
        # Temperature for entropy regularization
        self.temperature = config.get('temperature', 1.0)
        
    def forward(
        self,
        cluster_logits: torch.Tensor,
        embeddings: torch.Tensor,
        building_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        lv_group_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute balanced loss preventing collapse
        
        Args:
            cluster_logits: Soft cluster assignments [N, K]
            embeddings: Node embeddings [N, D]
            building_features: Original features [N, F]
            edge_index: Graph edges [2, E]
            lv_group_ids: LV group assignments [N]
            
        Returns:
            Total loss and component dictionary
        """
        losses = {}
        
        # Get soft assignments
        cluster_probs = F.softmax(cluster_logits / self.temperature, dim=1)
        
        # 1. Clustering Loss (encourage tight clusters)
        losses['clustering'] = self._clustering_loss(cluster_probs, embeddings)
        
        # 2. Diversity Loss (prevent collapse)
        losses['diversity'] = self._diversity_loss(cluster_probs)
        
        # 3. Solar Optimization (but bounded)
        losses['solar'] = self._bounded_solar_loss(cluster_probs, building_features)
        
        # 4. Balance Loss (encourage equal-sized clusters)
        losses['balance'] = self._balance_loss(cluster_probs)
        
        # 5. Physics Constraints (respect grid limits)
        losses['physics'] = self._physics_loss(cluster_probs, building_features, lv_group_ids)
        
        # Normalize losses to similar scales
        for key in losses:
            if torch.isnan(losses[key]) or torch.isinf(losses[key]):
                losses[key] = torch.tensor(0.0, device=cluster_logits.device)
        
        # Combine with adaptive weights
        total_loss = (
            self.w_clustering * torch.clamp(losses['clustering'], 0, 10) +
            self.w_diversity * torch.clamp(losses['diversity'], 0, 10) +
            self.w_solar * torch.clamp(losses['solar'], -10, 10) +
            self.w_balance * torch.clamp(losses['balance'], 0, 10) +
            self.w_physics * torch.clamp(losses['physics'], 0, 10)
        )
        
        # Add STRONG collapse penalty
        num_effective_clusters = self._count_effective_clusters(cluster_probs)
        if num_effective_clusters < self.min_clusters:
            # Much stronger penalty - exponential growth
            collapse_penalty = ((self.min_clusters - num_effective_clusters) ** 2) * 2.0
            total_loss += collapse_penalty
            losses['collapse_penalty'] = collapse_penalty
        elif num_effective_clusters < 2:
            # EXTREME penalty for single cluster
            collapse_penalty = 10.0
            total_loss += collapse_penalty
            losses['collapse_penalty'] = collapse_penalty
        
        return total_loss, losses
    
    def _clustering_loss(self, cluster_probs: torch.Tensor, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Minimize intra-cluster distances
        """
        # Compute cluster centroids
        centroids = torch.matmul(cluster_probs.T, embeddings) / (cluster_probs.sum(0, keepdim=True).T + 1e-8)
        
        # Compute distances to assigned centroids
        expanded_centroids = torch.matmul(cluster_probs, centroids)
        distances = torch.norm(embeddings - expanded_centroids, dim=1)
        
        return distances.mean()
    
    def _diversity_loss(self, cluster_probs: torch.Tensor) -> torch.Tensor:
        """
        Maximize entropy to prevent collapse - ENHANCED
        """
        # Cluster distribution entropy
        cluster_sizes = cluster_probs.sum(0)
        cluster_dist = cluster_sizes / (cluster_sizes.sum() + 1e-8)
        
        # Calculate entropy
        entropy = -(cluster_dist * torch.log(cluster_dist + 1e-8)).sum()
        
        # We want high entropy (diverse clusters)
        max_entropy = torch.log(torch.tensor(cluster_probs.shape[1], dtype=torch.float32, device=cluster_probs.device))
        
        # Stronger penalty for low entropy
        entropy_ratio = entropy / (max_entropy + 1e-8)
        
        # Exponential penalty for very low entropy (approaching collapse)
        if entropy_ratio < 0.5:
            diversity_loss = (max_entropy - entropy) * (2.0 - entropy_ratio) ** 2
        else:
            diversity_loss = max_entropy - entropy
        
        # Add penalty for any cluster being too dominant
        max_cluster_ratio = cluster_dist.max()
        if max_cluster_ratio > 0.5:  # One cluster has >50% of buildings
            diversity_loss += (max_cluster_ratio - 0.5) * 10.0
        
        return diversity_loss
    
    def _bounded_solar_loss(self, cluster_probs: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """
        Solar optimization but with diminishing returns for large clusters
        """
        # Extract solar potential and consumption
        if features.shape[1] >= 2:
            consumption = features[:, 0]
            solar_potential = features[:, 1] if features.shape[1] > 1 else torch.ones_like(consumption)
        else:
            consumption = torch.ones(features.shape[0])
            solar_potential = torch.ones(features.shape[0])
        
        # Calculate cluster-wise solar benefits with diminishing returns
        cluster_sizes = cluster_probs.sum(0)
        
        # Sigmoid to cap benefits for large clusters
        size_factor = torch.sigmoid((self.target_cluster_size - cluster_sizes) / 10)
        
        # Solar benefit per cluster
        weighted_solar = torch.matmul(cluster_probs.T, solar_potential)
        weighted_consumption = torch.matmul(cluster_probs.T, consumption) + 1e-8
        
        solar_efficiency = (weighted_solar / weighted_consumption) * size_factor
        
        # We want high efficiency
        return -solar_efficiency.mean()
    
    def _balance_loss(self, cluster_probs: torch.Tensor) -> torch.Tensor:
        """
        Encourage balanced cluster sizes
        """
        cluster_sizes = cluster_probs.sum(0)
        
        # Target size
        n_buildings = cluster_probs.shape[0]
        n_clusters = cluster_probs.shape[1]
        target_size = n_buildings / n_clusters
        
        # Normalized deviation (0-1 scale)
        size_deviation = torch.abs(cluster_sizes - target_size) / (target_size + 1e-8)
        
        # Extra penalty for very small clusters (normalized)
        small_penalty = torch.where(
            cluster_sizes < 5,
            (5 - cluster_sizes) / 5,  # Normalized to 0-1
            torch.zeros_like(cluster_sizes)
        )
        
        return size_deviation.mean() + small_penalty.mean()
    
    def _physics_loss(
        self,
        cluster_probs: torch.Tensor,
        features: torch.Tensor,
        lv_group_ids: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Enforce physical grid constraints
        """
        if lv_group_ids is None:
            return torch.tensor(0.0, device=cluster_probs.device)
        
        # Penalize clusters that span multiple LV groups
        physics_loss = 0
        
        for cluster_idx in range(cluster_probs.shape[1]):
            cluster_weight = cluster_probs[:, cluster_idx]
            
            # For each LV group, check concentration
            unique_groups = torch.unique(lv_group_ids)
            for group_id in unique_groups:
                group_mask = (lv_group_ids == group_id).float()
                
                # Weight in this cluster from this LV group
                group_cluster_weight = (cluster_weight * group_mask).sum()
                total_cluster_weight = cluster_weight.sum() + 1e-8
                
                # Concentration (should be close to 0 or 1)
                concentration = group_cluster_weight / total_cluster_weight
                
                # Penalty for partial concentration (want all or nothing)
                physics_loss += concentration * (1 - concentration)
        
        return physics_loss
    
    def _count_effective_clusters(self, cluster_probs: torch.Tensor) -> float:
        """
        Count effective number of clusters (using entropy)
        """
        cluster_sizes = cluster_probs.sum(0)
        cluster_dist = cluster_sizes / cluster_sizes.sum()
        
        # Remove very small clusters from count
        effective_dist = torch.where(cluster_dist > 0.01, cluster_dist, torch.zeros_like(cluster_dist))
        
        # Count non-zero clusters
        return (effective_dist > 0).sum().float()


class AdaptiveLossScheduler:
    """
    Dynamically adjust loss weights during training to prevent collapse
    """
    
    def __init__(self, initial_weights: Dict):
        self.weights = initial_weights.copy()
        self.history = []
        
    def update(self, metrics: Dict, epoch: int):
        """
        Adjust weights based on training progress
        """
        # If clusters are collapsing, increase diversity weight
        if metrics.get('num_clusters', 1) < metrics.get('target_clusters', 4):
            self.weights['diversity'] *= 1.1
            self.weights['solar'] *= 0.9  # Reduce solar pressure
            
        # If clusters are too many, increase clustering weight
        elif metrics.get('num_clusters', 1) > metrics.get('target_clusters', 4) * 1.5:
            self.weights['clustering'] *= 1.1
            self.weights['diversity'] *= 0.9
            
        # If loss not decreasing, reduce learning rate implicitly via weights
        if len(self.history) > 3:
            recent_losses = [h['loss'] for h in self.history[-3:]]
            if np.std(recent_losses) < 0.001:  # Plateaued
                # Scale all weights down (implicit LR decay)
                for key in self.weights:
                    self.weights[key] *= 0.95
        
        self.history.append({
            'epoch': epoch,
            'weights': self.weights.copy(),
            **metrics
        })
        
        return self.weights