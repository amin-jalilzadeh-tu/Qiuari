# models/task_heads.py
"""
Task-specific output heads for energy community planning
Produces clusters, sharing schedules, metrics, and intervention recommendations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================
# Data Classes for Structured Outputs
# ============================================

@dataclass
class ClusterAssignment:
    """Represents cluster assignments for buildings"""
    building_ids: List[int]
    cluster_ids: List[int]
    cluster_probs: torch.Tensor  # Soft assignments
    hour: int
    lv_group: str
    
@dataclass
class EnergyFlow:
    """Represents energy sharing between buildings"""
    from_building: int
    to_building: int
    amount_kw: float
    hour: int
    efficiency: float
    
@dataclass
class ClusterMetrics:
    """Performance metrics for a cluster"""
    cluster_id: int
    self_sufficiency: float
    peak_reduction: float
    diversity_index: float
    sharing_efficiency: float
    carbon_saved: float
    member_count: int
    
@dataclass
class InterventionRecommendation:
    """Recommendation for adding solar/battery/retrofit"""
    building_id: int
    intervention_type: str  # 'solar', 'battery', 'retrofit'
    capacity: float
    impact_ssr: float  # Impact on self-sufficiency
    impact_peak: float  # Impact on peak reduction
    roi_years: float
    confidence: float


# ============================================
# Task Head 1: Dynamic Sub-Clustering
# ============================================

class DynamicSubClusteringHead(nn.Module):
    """Creates time-varying sub-clusters within LV groups"""
    
    def __init__(self, 
                 embed_dim: int = 128,
                 min_cluster_size: int = 3,
                 max_cluster_size: int = 15,
                 temperature: float = 1.0):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Cluster assignment network
        self.cluster_encoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim // 2)
        )
        
        # Complementarity scorer for clustering
        self.complementarity_net = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh()  # -1 to 1, negative = complementary
        )
        
        # Size regularization
        self.size_penalty_weight = nn.Parameter(torch.tensor(1.0))
        
        logger.info("Initialized DynamicSubClusteringHead")
    
    def forward(self, 
                embeddings: torch.Tensor,
                lv_group_ids: torch.Tensor,
                complementarity_matrix: Optional[torch.Tensor] = None,
                current_hour: Optional[int] = None) -> Dict:
        """
        Create sub-clusters within each LV group
        
        Args:
            embeddings: Building embeddings [batch, N, embed_dim] or [N, embed_dim]
            lv_group_ids: LV group assignment for each building [N]
            complementarity_matrix: Pre-computed complementarity [N, N]
            current_hour: Hour for time-specific clustering (0-23)
            
        Returns:
            Dictionary with clustering results
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        batch_size, num_buildings, embed_dim = embeddings.shape
        device = embeddings.device
        
        # Encode for clustering
        cluster_features = self.cluster_encoder(embeddings)  # [B, N, D/2]
        
        # Get unique LV groups
        unique_lv_groups = torch.unique(lv_group_ids[lv_group_ids >= 0])  # Skip -1 (orphaned)
        
        all_clusters = []
        all_assignments = torch.zeros(batch_size, num_buildings, dtype=torch.long, device=device)
        all_probs = torch.zeros(batch_size, num_buildings, num_buildings, device=device)
        cluster_count = 0
        
        # Process each LV group separately
        for lv_group in unique_lv_groups:
            # Get buildings in this LV group
            group_mask = (lv_group_ids == lv_group)
            group_indices = torch.where(group_mask)[0]
            
            if len(group_indices) < self.min_cluster_size:
                # Too small for clustering - keep as single cluster
                all_assignments[:, group_indices] = cluster_count
                cluster_count += 1
                continue
            
            # Extract features for this group
            group_embeddings = embeddings[:, group_indices, :]
            group_cluster_features = cluster_features[:, group_indices, :]
            
            # Determine number of clusters for this group
            num_group_buildings = len(group_indices)
            num_clusters = min(
                max(num_group_buildings // self.max_cluster_size, 1),
                num_group_buildings // self.min_cluster_size
            )
            num_clusters = max(num_clusters, 1)
            
            # Compute pairwise complementarity within group
            if complementarity_matrix is not None:
                # Use provided complementarity
                group_comp = complementarity_matrix[group_indices][:, group_indices]
            else:
                # Compute complementarity
                group_comp = self._compute_complementarity(group_embeddings[0])
            
            # Perform soft clustering using complementarity
            cluster_assignments, cluster_probs = self._cluster_by_complementarity(
                group_cluster_features[0],
                group_comp,
                num_clusters
            )
            
            # Apply size constraints
            cluster_assignments = self._enforce_size_constraints(
                cluster_assignments,
                cluster_probs,
                self.min_cluster_size,
                self.max_cluster_size
            )
            
            # Store results
            for i, idx in enumerate(group_indices):
                all_assignments[:, idx] = cluster_assignments[i] + cluster_count
            
            # Store probabilities
            for i, idx_i in enumerate(group_indices):
                for j, idx_j in enumerate(group_indices):
                    all_probs[:, idx_i, idx_j] = cluster_probs[i, j]
            
            cluster_count += num_clusters
        
        # Calculate cluster metrics
        cluster_metrics = self._calculate_cluster_metrics(
            all_assignments[0],
            embeddings[0],
            lv_group_ids
        )
        
        return {
            'cluster_assignments': all_assignments,  # [B, N]
            'cluster_probabilities': all_probs,  # [B, N, N]
            'num_clusters': cluster_count,
            'cluster_metrics': cluster_metrics,
            'hour': current_hour
        }
    
    def _compute_complementarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute complementarity scores between buildings"""
        num_buildings = embeddings.shape[0]
        device = embeddings.device
        
        # Compute pairwise complementarity
        comp_matrix = torch.zeros(num_buildings, num_buildings, device=device)
        
        for i in range(num_buildings):
            for j in range(i+1, num_buildings):
                # Concatenate embeddings
                pair = torch.cat([embeddings[i], embeddings[j]], dim=0)
                # Get complementarity score
                score = self.complementarity_net(pair)
                comp_matrix[i, j] = score
                comp_matrix[j, i] = score
        
        return comp_matrix
    
    def _cluster_by_complementarity(self,
                                   features: torch.Tensor,
                                   complementarity: torch.Tensor,
                                   num_clusters: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cluster buildings to maximize complementarity"""
        num_buildings = features.shape[0]
        device = features.device
        
        # Initialize cluster centers using furthest point sampling
        centers = self._initialize_centers(features, num_clusters)
        
        # Iterative clustering to maximize complementarity
        max_iters = 10
        for _ in range(max_iters):
            # Assign buildings to clusters
            distances = torch.cdist(features, centers)
            
            # Adjust distances by complementarity
            # Buildings with high complementarity should cluster together
            cluster_probs = F.softmax(-distances / self.temperature, dim=1)
            
            # Weight by average complementarity to cluster members
            for k in range(num_clusters):
                cluster_mask = cluster_probs[:, k:k+1]  # [N, 1]
                # Average complementarity to cluster k members
                avg_comp = (complementarity * cluster_mask.T).sum(dim=1) / (cluster_mask.sum() + 1e-6)
                # Adjust probabilities - higher complementarity = higher probability
                cluster_probs[:, k] *= torch.exp(-avg_comp)  # Negative comp is good
            
            # Normalize probabilities
            cluster_probs = cluster_probs / (cluster_probs.sum(dim=1, keepdim=True) + 1e-6)
            
            # Hard assignments
            cluster_assignments = torch.argmax(cluster_probs, dim=1)
            
            # Update centers
            for k in range(num_clusters):
                mask = (cluster_assignments == k)
                if mask.any():
                    centers[k] = features[mask].mean(dim=0)
        
        # Compute final probability matrix (which buildings are in same cluster)
        prob_matrix = torch.zeros(num_buildings, num_buildings, device=device)
        for k in range(num_clusters):
            mask = (cluster_assignments == k).float()
            prob_matrix += mask.unsqueeze(1) * mask.unsqueeze(0)
        
        return cluster_assignments, prob_matrix
    
    def _initialize_centers(self, features: torch.Tensor, num_clusters: int) -> torch.Tensor:
        """Initialize cluster centers using furthest point sampling"""
        num_buildings, feat_dim = features.shape
        device = features.device
        
        centers = torch.zeros(num_clusters, feat_dim, device=device)
        
        # Start with random point
        indices = [torch.randint(num_buildings, (1,)).item()]
        centers[0] = features[indices[0]]
        
        # Add furthest points iteratively
        for k in range(1, num_clusters):
            distances = torch.cdist(features, centers[:k])
            min_distances = distances.min(dim=1)[0]
            furthest = torch.argmax(min_distances)
            indices.append(furthest.item())
            centers[k] = features[furthest]
        
        return centers
    
    def _enforce_size_constraints(self,
                                 assignments: torch.Tensor,
                                 probs: torch.Tensor,
                                 min_size: int,
                                 max_size: int) -> torch.Tensor:
        """Enforce cluster size constraints"""
        unique_clusters = torch.unique(assignments)
        
        for cluster_id in unique_clusters:
            mask = (assignments == cluster_id)
            cluster_size = mask.sum().item()
            
            if cluster_size < min_size:
                # Merge with nearest cluster
                # Find cluster with highest average probability
                best_merge = -1
                best_prob = -1
                
                for other_cluster in unique_clusters:
                    if other_cluster != cluster_id:
                        other_mask = (assignments == other_cluster)
                        avg_prob = probs[mask][:, other_mask].mean().item()
                        if avg_prob > best_prob:
                            best_prob = avg_prob
                            best_merge = other_cluster
                
                if best_merge >= 0:
                    assignments[mask] = best_merge
            
            elif cluster_size > max_size:
                # Split cluster
                cluster_indices = torch.where(mask)[0]
                # Use probabilities to split
                sub_probs = probs[cluster_indices][:, cluster_indices]
                # Find least connected half
                connectivity = sub_probs.sum(dim=1)
                median_conn = connectivity.median()
                split_mask = connectivity < median_conn
                
                # Assign to new cluster
                new_cluster_id = assignments.max() + 1
                assignments[cluster_indices[split_mask]] = new_cluster_id
        
        return assignments
    
    def _calculate_cluster_metrics(self,
                                  assignments: torch.Tensor,
                                  embeddings: torch.Tensor,
                                  lv_group_ids: torch.Tensor) -> Dict:
        """Calculate metrics for each cluster"""
        metrics = {}
        unique_clusters = torch.unique(assignments)
        
        for cluster_id in unique_clusters:
            mask = (assignments == cluster_id)
            cluster_size = mask.sum().item()
            
            # Get LV group for this cluster
            cluster_lv_groups = lv_group_ids[mask]
            primary_lv = cluster_lv_groups[0].item()  # Should all be same
            
            metrics[cluster_id.item()] = {
                'size': cluster_size,
                'lv_group': primary_lv,
                'cohesion': self._calculate_cohesion(embeddings[mask]),
                'separation': self._calculate_separation(embeddings[mask], embeddings[~mask])
            }
        
        return metrics
    
    def _calculate_cohesion(self, cluster_embeddings: torch.Tensor) -> float:
        """Calculate within-cluster cohesion"""
        if len(cluster_embeddings) < 2:
            return 1.0
        
        # Average pairwise distance within cluster
        distances = torch.cdist(cluster_embeddings, cluster_embeddings)
        # Exclude diagonal
        mask = ~torch.eye(len(cluster_embeddings), dtype=torch.bool, device=distances.device)
        avg_distance = distances[mask].mean().item()
        
        # Convert to cohesion score (inverse of distance)
        cohesion = 1.0 / (1.0 + avg_distance)
        return cohesion
    
    def _calculate_separation(self, 
                            cluster_embeddings: torch.Tensor,
                            other_embeddings: torch.Tensor) -> float:
        """Calculate cluster separation from others"""
        if len(other_embeddings) == 0:
            return 1.0
        
        # Average distance to other clusters
        distances = torch.cdist(cluster_embeddings, other_embeddings)
        avg_distance = distances.mean().item()
        
        # Separation score (normalized distance)
        separation = avg_distance / (avg_distance + 1.0)
        return separation


# ============================================
# Task Head 2: Energy Sharing Predictor
# ============================================

class EnergySharingPredictor(nn.Module):
    """Predicts energy flows between buildings within clusters"""
    
    def __init__(self, 
                 embed_dim: int = 128,
                 temporal_dim: int = 24):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.temporal_dim = temporal_dim
        
        # Flow prediction network
        self.flow_predictor = nn.Sequential(
            nn.Linear(embed_dim * 2 + 2, 128),  # 2 embeddings + distance + time
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus()  # Positive flow only
        )
        
        # Efficiency predictor (based on distance)
        self.efficiency_net = nn.Sequential(
            nn.Linear(1, 16),  # Distance input
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # 0 to 1 efficiency
        )
        
        # Priority scorer (who should share first)
        self.priority_scorer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        logger.info("Initialized EnergySharingPredictor")
    
    # Fixed EnergySharingPredictor class in task_heads.py
    # Replace the forward method around line 500-530

    def forward(self,
                embeddings: torch.Tensor,
                cluster_assignments: torch.Tensor,
                generation: torch.Tensor,
                consumption: torch.Tensor,
                positions: torch.Tensor,
                current_hour: int = 0) -> Dict:
        """
        Predict energy sharing flows within clusters - FIXED FOR SINGLE BUILDING CASE
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        if generation.dim() == 1:
            generation = generation.unsqueeze(0)
        if consumption.dim() == 1:
            consumption = consumption.unsqueeze(0)
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        
        batch_size, num_buildings, _ = embeddings.shape
        device = embeddings.device
        
        # Calculate net position (surplus/deficit)
        net_position = generation - consumption  # Positive = surplus, Negative = deficit
        
        # If temporal dimension exists, select current hour
        if net_position.dim() == 3:
            net_position = net_position[:, :, current_hour]
        
        # Initialize sharing matrix
        sharing_matrix = torch.zeros(batch_size, num_buildings, num_buildings, device=device)
        efficiency_matrix = torch.ones(batch_size, num_buildings, num_buildings, device=device)
        
        # Get unique clusters
        unique_clusters = torch.unique(cluster_assignments)
        
        all_flows = []
        
        # Process each cluster
        for cluster_id in unique_clusters:
            # Get buildings in this cluster
            cluster_mask = (cluster_assignments[0] == cluster_id)
            cluster_indices = torch.where(cluster_mask)[0]
            
            if len(cluster_indices) < 2:
                continue  # Need at least 2 buildings to share
            
            # Split into surplus and deficit buildings
            cluster_net = net_position[0, cluster_indices]
            surplus_mask = cluster_net > 0
            deficit_mask = cluster_net < 0
            
            surplus_indices = cluster_indices[surplus_mask]
            deficit_indices = cluster_indices[deficit_mask]
            
            if len(surplus_indices) == 0 or len(deficit_indices) == 0:
                continue  # Need both surplus and deficit
            
            # Calculate priorities
            surplus_priorities = self.priority_scorer(embeddings[0, surplus_indices])
            deficit_priorities = self.priority_scorer(embeddings[0, deficit_indices])
            
            # FIX: Handle single building case properly
            if len(surplus_indices) == 1:
                surplus_sorted = surplus_indices
            else:
                surplus_order = torch.argsort(surplus_priorities.squeeze(), descending=True)
                surplus_sorted = surplus_indices[surplus_order]
            
            if len(deficit_indices) == 1:
                deficit_sorted = deficit_indices
            else:
                deficit_order = torch.argsort(deficit_priorities.squeeze(), descending=True)
                deficit_sorted = deficit_indices[deficit_order]
            
            # Match surplus to deficit
            for surplus_idx in surplus_sorted:
                surplus_available = net_position[0, surplus_idx].item()
                
                for deficit_idx in deficit_sorted:
                    if surplus_available <= 0:
                        break
                    
                    deficit_needed = -net_position[0, deficit_idx].item()
                    
                    if deficit_needed <= 0:
                        continue
                    
                    # Calculate distance and efficiency
                    distance = torch.norm(
                        positions[0, surplus_idx] - positions[0, deficit_idx]
                    )
                    efficiency = self._calculate_efficiency(distance)
                    
                    # Predict flow amount
                    flow_features = torch.cat([
                        embeddings[0, surplus_idx],
                        embeddings[0, deficit_idx],
                        distance.unsqueeze(0),
                        torch.tensor([current_hour / 24.0], device=device)
                    ])
                    
                    predicted_flow = self.flow_predictor(flow_features).item()
                    
                    # Actual flow is minimum of available, needed, and predicted
                    actual_flow = min(surplus_available, deficit_needed, predicted_flow)
                    
                    # Apply efficiency
                    delivered_flow = actual_flow * efficiency
                    
                    # Update sharing matrix
                    sharing_matrix[0, surplus_idx, deficit_idx] = actual_flow
                    efficiency_matrix[0, surplus_idx, deficit_idx] = efficiency
                    
                    # Update available/needed
                    surplus_available -= actual_flow
                    net_position[0, deficit_idx] += delivered_flow
                    
                    # Record flow
                    flow = EnergyFlow(
                        from_building=surplus_idx.item(),
                        to_building=deficit_idx.item(),
                        amount_kw=actual_flow,
                        hour=current_hour,
                        efficiency=efficiency
                    )
                    all_flows.append(flow)
        
        # Calculate total shared energy
        total_shared = sharing_matrix.sum().item()
        
        # Calculate sharing balance per building
        energy_sent = sharing_matrix.sum(dim=2)  # Energy sent by each building
        energy_received = (sharing_matrix * efficiency_matrix).sum(dim=1)  # Energy received
        
        return {
            'sharing_matrix': sharing_matrix,  # [B, N, N]
            'efficiency_matrix': efficiency_matrix,  # [B, N, N]
            'energy_flows': all_flows,
            'total_shared_kw': total_shared,
            'energy_sent': energy_sent,  # [B, N]
            'energy_received': energy_received,  # [B, N]
            'net_position_after': net_position  # [B, N]
        }
    
    def _calculate_efficiency(self, distance: torch.Tensor) -> float:
        """Calculate transmission efficiency based on distance"""
        # Normalize distance (assume max 1000m)
        normalized_dist = distance / 1000.0
        efficiency = self.efficiency_net(normalized_dist.unsqueeze(0))
        
        # Ensure reasonable efficiency (85% to 98%)
        efficiency = 0.85 + 0.13 * efficiency
        
        return efficiency.item()


# ============================================
# Task Head 3: Self-Sufficiency Metrics
# ============================================

class SelfSufficiencyMetricsCalculator(nn.Module):
    """Calculates performance metrics for clusters"""
    
    def __init__(self, carbon_intensity: float = 0.4):  # kg CO2/kWh
        super().__init__()
        
        self.carbon_intensity = carbon_intensity
        
        # Learnable importance weights for metrics
        self.metric_weights = nn.ParameterDict({
            'ssr': nn.Parameter(torch.tensor(1.0)),
            'peak': nn.Parameter(torch.tensor(0.8)),
            'diversity': nn.Parameter(torch.tensor(0.6)),
            'efficiency': nn.Parameter(torch.tensor(0.7)),
            'carbon': nn.Parameter(torch.tensor(0.9))
        })
        
        logger.info("Initialized SelfSufficiencyMetricsCalculator")
    
    def forward(self,
                cluster_assignments: torch.Tensor,
                generation: torch.Tensor,
                consumption: torch.Tensor,
                sharing_matrix: torch.Tensor,
                efficiency_matrix: torch.Tensor,
                building_types: Optional[torch.Tensor] = None) -> Dict:
        """
        Calculate comprehensive metrics for each cluster
        
        Args:
            cluster_assignments: Cluster assignment per building [batch, N]
            generation: Generation per building [batch, N] or [batch, N, T]
            consumption: Consumption per building [batch, N] or [batch, N, T]
            sharing_matrix: Energy sharing flows [batch, N, N]
            efficiency_matrix: Transmission efficiency [batch, N, N]
            building_types: Building type categories [N] (optional)
            
        Returns:
            Dictionary with metrics per cluster
        """
        if cluster_assignments.dim() == 1:
            cluster_assignments = cluster_assignments.unsqueeze(0)
        
        batch_size = cluster_assignments.shape[0]
        device = cluster_assignments.device
        
        # Get unique clusters
        unique_clusters = torch.unique(cluster_assignments)
        
        cluster_metrics = {}
        
        for cluster_id in unique_clusters:
            # Get buildings in this cluster
            cluster_mask = (cluster_assignments[0] == cluster_id)
            cluster_indices = torch.where(cluster_mask)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Extract cluster data
            cluster_gen = generation[:, cluster_indices]
            cluster_cons = consumption[:, cluster_indices]
            
            # Handle temporal dimension
            if cluster_gen.dim() == 3:
                # Average over time
                cluster_gen = cluster_gen.mean(dim=2)
                cluster_cons = cluster_cons.mean(dim=2)
            
            # 1. Self-Sufficiency Rate (SSR)
            total_generation = cluster_gen.sum().item()
            total_consumption = cluster_cons.sum().item()
            
            # Energy used locally (within cluster)
            cluster_sharing = sharing_matrix[0, cluster_indices][:, cluster_indices]
            local_energy_used = min(total_generation, total_consumption)
            
            ssr = local_energy_used / (total_consumption + 1e-6)
            
            # 2. Peak Reduction
            # Peak without sharing
            peak_without = cluster_cons.max().item()
            
            # Peak with sharing (after receiving energy)
            energy_received = (sharing_matrix[0, :, cluster_indices] * 
                             efficiency_matrix[0, :, cluster_indices]).sum(dim=0)
            net_consumption = cluster_cons[0] - energy_received
            peak_with = net_consumption.max().item()
            
            peak_reduction = (peak_without - peak_with) / (peak_without + 1e-6)
            
            # 3. Diversity Index
            if building_types is not None:
                cluster_types = building_types[cluster_indices]
                unique_types = torch.unique(cluster_types).numel()
                diversity_index = unique_types / len(cluster_indices)
            else:
                # Use consumption patterns as proxy for diversity
                patterns = cluster_cons[0]
                pattern_std = patterns.std().item()
                pattern_mean = patterns.mean().item()
                diversity_index = pattern_std / (pattern_mean + 1e-6)
            
            # 4. Sharing Efficiency
            total_sent = cluster_sharing.sum().item()
            if total_sent > 0:
                cluster_efficiency = efficiency_matrix[0, cluster_indices][:, cluster_indices]
                total_delivered = (cluster_sharing * cluster_efficiency).sum().item()
                sharing_efficiency = total_delivered / total_sent
            else:
                sharing_efficiency = 1.0
            
            # 5. Carbon Saved
            # Grid energy avoided by local sharing
            grid_avoided = local_energy_used
            carbon_saved = grid_avoided * self.carbon_intensity
            
            # 6. Overall Score (weighted combination)
            overall_score = (
                self.metric_weights['ssr'] * ssr +
                self.metric_weights['peak'] * peak_reduction +
                self.metric_weights['diversity'] * diversity_index +
                self.metric_weights['efficiency'] * sharing_efficiency +
                self.metric_weights['carbon'] * (carbon_saved / 100.0)  # Normalize
            )
            
            # Store metrics
            metrics = ClusterMetrics(
                cluster_id=cluster_id.item(),
                self_sufficiency=ssr,
                peak_reduction=peak_reduction,
                diversity_index=diversity_index,
                sharing_efficiency=sharing_efficiency,
                carbon_saved=carbon_saved,
                member_count=len(cluster_indices)
            )
            
            cluster_metrics[cluster_id.item()] = {
                'metrics': metrics,
                'overall_score': overall_score.item(),
                'member_buildings': cluster_indices.tolist()
            }
        
        return cluster_metrics


# ============================================
# Task Head 4: Intervention Recommender
# ============================================

class InterventionRecommender(nn.Module):
    """Recommends solar, battery, and retrofit interventions"""
    
    def __init__(self, 
                 embed_dim: int = 128,
                 max_recommendations: int = 10):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_recommendations = max_recommendations
        
        # Solar potential scorer
        self.solar_scorer = nn.Sequential(
            nn.Linear(embed_dim + 3, 64),  # Embeddings + roof area + orientation + current solar
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Battery value scorer
        self.battery_scorer = nn.Sequential(
            nn.Linear(embed_dim + 3, 64),  # Embeddings + consumption variance + peak + solar
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Retrofit impact scorer
        self.retrofit_scorer = nn.Sequential(
            nn.Linear(embed_dim + 3, 64),  # Embeddings + age + energy label + consumption
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Impact predictor
        self.impact_predictor = nn.Sequential(
            nn.Linear(embed_dim + 7, 128),  # Embeddings + intervention type + capacity + cluster context
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # SSR impact, peak impact, ROI years
        )
        
        logger.info("Initialized InterventionRecommender")
    
    def forward(self,
                embeddings: torch.Tensor,
                cluster_assignments: torch.Tensor,
                cluster_metrics: Dict,
                building_features: Dict,
                current_assets: Dict) -> List[InterventionRecommendation]:
        """
        Recommend interventions for maximum impact
        
        Args:
            embeddings: Building embeddings [batch, N, embed_dim]
            cluster_assignments: Cluster assignments [batch, N]
            cluster_metrics: Current performance metrics per cluster
            building_features: Dictionary with roof_area, age, etc.
            current_assets: Dictionary with has_solar, has_battery, etc.
            
        Returns:
            List of intervention recommendations
        """
        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)
        
        batch_size, num_buildings, _ = embeddings.shape
        device = embeddings.device
        
        recommendations = []
        
        # Score each building for each intervention type
        for building_idx in range(num_buildings):
            building_embed = embeddings[0, building_idx]
            
            # Get cluster context
            cluster_id = cluster_assignments[0, building_idx].item()
            if cluster_id in cluster_metrics:
                cluster_perf = cluster_metrics[cluster_id]['metrics']
                cluster_context = torch.tensor([
                    cluster_perf.self_sufficiency,
                    cluster_perf.peak_reduction,
                    cluster_perf.member_count / 20.0  # Normalize
                ], device=device)
            else:
                cluster_context = torch.zeros(3, device=device)
            
            # 1. Solar recommendation
            if not current_assets.get('has_solar', {}).get(building_idx, False):
                solar_features = torch.cat([
                    building_embed,
                    torch.tensor([
                        building_features.get('roof_area', {}).get(building_idx, 0) / 500.0,
                        building_features.get('orientation_score', {}).get(building_idx, 0.5),
                        0.0  # No current solar
                    ], device=device)
                ])
                
                solar_score = self.solar_scorer(solar_features).item()
                
                if solar_score > 0.5:  # Threshold for recommendation
                    # Determine capacity based on roof area
                    roof_area = building_features.get('roof_area', {}).get(building_idx, 100)
                    capacity_kwp = min(roof_area * 0.15, 100)  # 150W/m², max 100kWp
                    
                    # Predict impact
                    impact = self._predict_impact(
                        building_embed,
                        'solar',
                        capacity_kwp,
                        cluster_context
                    )
                    
                    rec = InterventionRecommendation(
                        building_id=building_idx,
                        intervention_type='solar',
                        capacity=capacity_kwp,
                        impact_ssr=impact['ssr'],
                        impact_peak=impact['peak'],
                        roi_years=impact['roi'],
                        confidence=solar_score
                    )
                    recommendations.append(rec)
            
            # 2. Battery recommendation
            if not current_assets.get('has_battery', {}).get(building_idx, False):
                # Calculate consumption variance
                if 'consumption_history' in building_features:
                    cons_history = building_features['consumption_history'][building_idx]
                    cons_variance = cons_history.std().item() if cons_history.numel() > 1 else 0
                else:
                    cons_variance = 0.5
                
                battery_features = torch.cat([
                    building_embed,
                    torch.tensor([
                        cons_variance,
                        building_features.get('peak_demand', {}).get(building_idx, 10) / 50.0,
                        1.0 if current_assets.get('has_solar', {}).get(building_idx, False) else 0.0
                    ], device=device)
                ])
                
                battery_score = self.battery_scorer(battery_features).item()
                
                if battery_score > 0.5:
                    # Determine capacity based on consumption pattern
                    peak_demand = building_features.get('peak_demand', {}).get(building_idx, 10)
                    capacity_kwh = min(peak_demand * 2, 50)  # 2 hours of peak, max 50kWh
                    
                    # Predict impact
                    impact = self._predict_impact(
                        building_embed,
                        'battery',
                        capacity_kwh,
                        cluster_context
                    )
                    
                    rec = InterventionRecommendation(
                        building_id=building_idx,
                        intervention_type='battery',
                        capacity=capacity_kwh,
                        impact_ssr=impact['ssr'],
                        impact_peak=impact['peak'],
                        roi_years=impact['roi'],
                        confidence=battery_score
                    )
                    recommendations.append(rec)
            
            # 3. Retrofit recommendation
            energy_label = building_features.get('energy_label', {}).get(building_idx, 'D')
            if energy_label in ['D', 'E', 'F', 'G']:
                retrofit_features = torch.cat([
                    building_embed,
                    torch.tensor([
                        building_features.get('building_age', {}).get(building_idx, 30) / 100.0,
                        ord(energy_label) - ord('A'),  # Convert label to number
                        building_features.get('heating_demand', {}).get(building_idx, 100) / 200.0
                    ], device=device)
                ])
                
                retrofit_score = self.retrofit_scorer(retrofit_features).item()
                
                if retrofit_score > 0.5:
                    # Estimate retrofit level
                    capacity = 1.0  # Represents full retrofit
                    
                    # Predict impact
                    impact = self._predict_impact(
                        building_embed,
                        'retrofit',
                        capacity,
                        cluster_context
                    )
                    
                    rec = InterventionRecommendation(
                        building_id=building_idx,
                        intervention_type='retrofit',
                        capacity=capacity,
                        impact_ssr=impact['ssr'],
                        impact_peak=impact['peak'],
                        roi_years=impact['roi'],
                        confidence=retrofit_score
                    )
                    recommendations.append(rec)
        
        # Sort by impact and return top recommendations
        recommendations.sort(key=lambda x: x.impact_ssr + x.impact_peak, reverse=True)
        
        return recommendations[:self.max_recommendations]
    
    def _predict_impact(self,
                       building_embed: torch.Tensor,
                       intervention_type: str,
                       capacity: float,
                       cluster_context: torch.Tensor) -> Dict:
        """Predict impact of intervention"""
        
        # Encode intervention type
        type_encoding = torch.zeros(3, device=building_embed.device)
        if intervention_type == 'solar':
            type_encoding[0] = 1
        elif intervention_type == 'battery':
            type_encoding[1] = 1
        else:  # retrofit
            type_encoding[2] = 1
        
        # Combine features
        impact_features = torch.cat([
            building_embed,
            type_encoding,
            torch.tensor([capacity / 100.0], device=building_embed.device),
            cluster_context
        ])
        
        # Predict impacts
        predictions = self.impact_predictor(impact_features)
        
        # Apply activation functions
        ssr_impact = torch.sigmoid(predictions[0]).item()  # 0 to 1
        peak_impact = torch.sigmoid(predictions[1]).item()  # 0 to 1
        roi_years = torch.exp(predictions[2]).item()  # Positive
        
        # Clip ROI to reasonable range
        roi_years = min(max(roi_years, 3), 20)
        
        return {
            'ssr': ssr_impact,
            'peak': peak_impact,
            'roi': roi_years
        }


# ============================================
# Main Task Heads Module
# ============================================

class EnergyTaskHeads(nn.Module):
    """Combined task heads for energy community planning"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        embed_dim = config.get('hidden_dim', 128)
        
        # Initialize all task heads
        self.clustering_head = DynamicSubClusteringHead(
            embed_dim=embed_dim,
            min_cluster_size=config.get('min_cluster_size', 3),
            max_cluster_size=config.get('max_cluster_size', 15)
        )
        
        self.sharing_head = EnergySharingPredictor(
            embed_dim=embed_dim,
            temporal_dim=config.get('temporal_dim', 24)
        )
        
        self.metrics_calculator = SelfSufficiencyMetricsCalculator(
            carbon_intensity=config.get('carbon_intensity', 0.4)
        )
        
        self.intervention_recommender = InterventionRecommender(
            embed_dim=embed_dim,
            max_recommendations=config.get('max_recommendations', 10)
        )
        
        logger.info("Initialized EnergyTaskHeads with all components")
    
    def forward(self,
                embeddings_dict: Dict,
                metadata: Dict,
                current_hour: Optional[int] = None) -> Dict:
        """
        Run all task heads
        
        Args:
            embeddings_dict: Dictionary of embeddings from previous layers
            metadata: Dictionary containing:
                - lv_group_ids: LV group assignments
                - positions: Building positions
                - generation: Generation data
                - consumption: Consumption data
                - building_features: Roof area, age, etc.
                - current_assets: Existing solar, batteries, etc.
                - complementarity_matrix: From attention layer
            current_hour: Current hour for time-specific outputs
            
        Returns:
            Dictionary with all task outputs
        """
        
        # Get building embeddings
        building_embeddings = embeddings_dict.get('building')
        if building_embeddings is None:
            raise ValueError("Building embeddings not found")
        
        # 1. Dynamic clustering
        cluster_output = self.clustering_head(
            building_embeddings,
            metadata['lv_group_ids'],
            metadata.get('complementarity_matrix'),
            current_hour
        )
        
        # 2. Energy sharing prediction
        sharing_output = self.sharing_head(
            building_embeddings,
            cluster_output['cluster_assignments'],
            metadata['generation'],
            metadata['consumption'],
            metadata['positions'],
            current_hour or 0
        )
        
        # 3. Calculate metrics
        metrics_output = self.metrics_calculator(
            cluster_output['cluster_assignments'],
            metadata['generation'],
            metadata['consumption'],
            sharing_output['sharing_matrix'],
            sharing_output['efficiency_matrix'],
            metadata.get('building_types')
        )
        
        # 4. Recommend interventions
        recommendations = self.intervention_recommender(
            building_embeddings,
            cluster_output['cluster_assignments'],
            metrics_output,
            metadata.get('building_features', {}),
            metadata.get('current_assets', {})
        )
        
        return {
            'clustering': cluster_output,
            'sharing': sharing_output,
            'metrics': metrics_output,
            'recommendations': recommendations,
            'summary': self._create_summary(
                cluster_output,
                sharing_output,
                metrics_output,
                recommendations
            )
        }
    
    def _create_summary(self, clustering, sharing, metrics, recommendations) -> Dict:
        """Create executive summary of results"""
        
        # Overall statistics
        num_clusters = clustering['num_clusters']
        total_shared = sharing['total_shared_kw']
        
        # Average metrics across clusters
        avg_ssr = np.mean([m['metrics'].self_sufficiency for m in metrics.values()])
        avg_peak_reduction = np.mean([m['metrics'].peak_reduction for m in metrics.values()])
        total_carbon_saved = sum([m['metrics'].carbon_saved for m in metrics.values()])
        
        # Top recommendations
        top_recs = []
        for rec in recommendations[:3]:
            top_recs.append({
                'building': rec.building_id,
                'type': rec.intervention_type,
                'impact': f"+{rec.impact_ssr*100:.1f}% SSR"
            })
        
        return {
            'num_clusters': num_clusters,
            'total_energy_shared_kw': total_shared,
            'avg_self_sufficiency': avg_ssr,
            'avg_peak_reduction': avg_peak_reduction,
            'total_carbon_saved_kg': total_carbon_saved,
            'top_interventions': top_recs
        }


def create_energy_task_heads(config: Dict) -> EnergyTaskHeads:
    """Factory function to create task heads"""
    return EnergyTaskHeads(config)