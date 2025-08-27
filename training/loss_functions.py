"""
Loss functions for Energy GNN training
Focuses on complementarity, energy metrics, and physics constraints
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class ComplementarityLoss(nn.Module):
    """
    Loss function that rewards complementary patterns (negative correlation)
    and penalizes similar patterns within clusters
    """
    
    def __init__(
        self,
        correlation_weight: float = 1.0,
        separation_weight: float = 0.5,
        diversity_weight: float = 0.3,
        temperature: float = 0.1
    ):
        super().__init__()
        self.correlation_weight = correlation_weight
        self.separation_weight = separation_weight
        self.diversity_weight = diversity_weight
        self.temperature = temperature
        
    def forward(
        self,
        embeddings: torch.Tensor,
        cluster_probs: torch.Tensor,
        temporal_profiles: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate complementarity loss
        
        Args:
            embeddings: Node embeddings [N, D]
            cluster_probs: Soft cluster assignments [N, K]
            temporal_profiles: Time series profiles [N, T] for correlation
            edge_index: Graph connectivity for local complementarity
            
        Returns:
            Total loss and component dictionary
        """
        losses = {}
        
        # 1. Negative Correlation Loss (main complementarity metric)
        if temporal_profiles is not None:
            corr_loss = self._negative_correlation_loss(
                temporal_profiles, cluster_probs
            )
            losses['correlation'] = corr_loss
        else:
            corr_loss = 0
            
        # 2. Cluster Separation Loss (different clusters should be different)
        sep_loss = self._cluster_separation_loss(embeddings, cluster_probs)
        losses['separation'] = sep_loss
        
        # 3. Diversity Loss (encourage diversity within clusters)
        div_loss = self._diversity_loss(embeddings, cluster_probs)
        losses['diversity'] = div_loss
        
        # Combine losses
        total_loss = (
            self.correlation_weight * corr_loss +
            self.separation_weight * sep_loss +
            self.diversity_weight * div_loss
        )
        
        # Assert loss is non-negative
        assert total_loss >= 0, f"ComplementarityLoss returned negative value: {total_loss}"
        
        return total_loss, losses
    
    def _negative_correlation_loss(
        self,
        profiles: torch.Tensor,
        cluster_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Reward negative correlation within clusters
        L = -∑_{i,j∈C} ρ(x_i, x_j) for same cluster
        """
        N, T = profiles.shape
        K = cluster_probs.shape[1]
        
        # Normalize profiles
        profiles_norm = (profiles - profiles.mean(dim=1, keepdim=True)) / (
            profiles.std(dim=1, keepdim=True) + 1e-8
        )
        
        # Compute correlation matrix
        corr_matrix = torch.matmul(profiles_norm, profiles_norm.t()) / T
        
        # Weight by cluster co-membership probability
        cluster_comembership = torch.matmul(cluster_probs, cluster_probs.t())
        
        # Loss: minimize correlation within clusters (want negative correlation)
        # We want negative correlation, so positive correlation should have high loss
        # Shift correlation values: -1 (perfect complementarity) -> 0, +1 (bad) -> 2
        weighted_corr = (corr_matrix + 1.0) * cluster_comembership
        
        # Exclude self-correlation
        mask = 1 - torch.eye(N, device=profiles.device)
        weighted_corr = weighted_corr * mask
        
        # Average correlation per cluster assignment strength
        # This will be 0 for perfect negative correlation, 2 for perfect positive correlation
        loss = weighted_corr.sum() / (cluster_comembership * mask).sum().clamp(min=1)
        
        # Ensure loss is always non-negative
        loss = torch.abs(loss)  # Safety check
        
        return loss
    
    def _cluster_separation_loss(
        self,
        embeddings: torch.Tensor,
        cluster_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure clusters are well-separated in embedding space
        """
        # Compute cluster centers
        cluster_centers = torch.matmul(cluster_probs.t(), embeddings)
        cluster_sizes = cluster_probs.sum(dim=0, keepdim=True).t() + 1e-8
        cluster_centers = cluster_centers / cluster_sizes
        
        # Compute pairwise distances between cluster centers
        K = cluster_centers.shape[0]
        center_distances = torch.cdist(cluster_centers, cluster_centers, p=2)
        
        # Create mask for upper triangle (avoid counting pairs twice)
        mask = torch.triu(torch.ones(K, K, device=embeddings.device), diagonal=1)
        
        # Loss: maximize minimum distance between clusters
        # We want large distances, so penalize small distances
        # Use exp(-distance) so small distances -> high loss, large distances -> low loss
        separation_loss = (torch.exp(-center_distances) * mask).sum() / mask.sum()
        
        return separation_loss
    
    def _diversity_loss(
        self,
        embeddings: torch.Tensor,
        cluster_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Encourage diversity within clusters using determinantal point process
        """
        K = cluster_probs.shape[1]
        total_loss = 0
        
        for k in range(K):
            # Get weighted embeddings for cluster k
            weights = cluster_probs[:, k:k+1]  # [N, 1]
            weighted_embeddings = embeddings * weights
            
            # Compute kernel matrix (similarity)
            kernel = torch.matmul(weighted_embeddings, weighted_embeddings.t())
            
            # Diversity loss: we want high diversity (high determinant)
            # So minimize negative log determinant -> maximize determinant
            # Add small diagonal for numerical stability
            kernel = kernel + torch.eye(kernel.shape[0], device=kernel.device) * 1e-6
            
            # Use log determinant for diversity
            # We want to MAXIMIZE diversity (high determinant)
            # So we minimize -log(det) which is equivalent to maximizing log(det)
            # But we need to ensure the loss is positive and bounded
            
            # First, normalize the kernel to [0, 1] range
            kernel_norm = kernel / (kernel.max() + 1e-8)
            
            # Calculate determinant (add identity for stability)
            stable_kernel = kernel_norm + torch.eye(kernel.shape[0], device=kernel.device) * 0.1
            logdet = torch.logdet(stable_kernel)
            
            # Convert to loss: we want high diversity (high logdet)
            # Use sigmoid to bound the loss between 0 and 1
            # High logdet -> low loss
            diversity = torch.sigmoid(-logdet)
            total_loss = total_loss + diversity
            
        return total_loss / K


class EnergyBalanceLoss(nn.Module):
    """
    Energy balance loss for community formation (without electrical grid constraints)
    """
    
    def __init__(
        self,
        balance_weight: float = 1.0,
        spatial_weight: float = 0.5,
        clustering_weight: float = 0.3
    ):
        super().__init__()
        self.balance_weight = balance_weight
        self.spatial_weight = spatial_weight
        self.clustering_weight = clustering_weight
        
    def forward(
        self,
        energy_sharing: torch.Tensor,
        cluster_assignments: torch.Tensor,
        node_positions: Optional[torch.Tensor] = None,
        consumption: Optional[torch.Tensor] = None,
        generation: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate energy balance and spatial losses
        
        Args:
            energy_sharing: Energy sharing between nodes [N, N] or [N, N, T]
            cluster_assignments: Cluster assignments [N] or [N, K] probabilities
            node_positions: Spatial coordinates [N, 2]
            consumption: Energy consumption [N] or [N, T]
            generation: Energy generation [N] or [N, T]
            
        Returns:
            Total loss and components
        """
        losses = {}
        
        # Energy balance within communities
        balance_loss = self._energy_balance_loss(energy_sharing, consumption, generation)
        losses['balance'] = balance_loss
        
        # Spatial compactness loss
        if node_positions is not None:
            spatial_loss = self._spatial_compactness_loss(
                cluster_assignments, node_positions
            )
            losses['spatial'] = spatial_loss
        else:
            spatial_loss = 0
            
        # Clustering quality loss
        clustering_loss = self._clustering_quality_loss(
            energy_sharing, cluster_assignments
        )
        losses['clustering'] = clustering_loss
        
        total_loss = (
            self.balance_weight * balance_loss +
            self.spatial_weight * spatial_loss +
            self.clustering_weight * clustering_loss
        )
        
        return total_loss, losses
    
    def _energy_balance_loss(
        self, 
        energy_sharing: torch.Tensor,
        consumption: Optional[torch.Tensor] = None,
        generation: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Energy balance within each community"""
        if consumption is None or generation is None:
            # If no consumption/generation data, just ensure sharing is balanced
            imbalance = energy_sharing.sum(dim=-2) - energy_sharing.sum(dim=-1)
            if imbalance.dim() > 1:
                imbalance = imbalance.mean(dim=-1)  # Average over time if present
        else:
            # Check that consumption - generation = net import
            net_demand = consumption - generation if generation is not None else consumption
            net_import = energy_sharing.sum(dim=-2) - energy_sharing.sum(dim=-1)
            if net_import.dim() > net_demand.dim():
                net_import = net_import.mean(dim=-1)
            imbalance = net_demand - net_import
        
        loss = (imbalance ** 2).mean()
        return loss
    
    def _spatial_compactness_loss(
        self,
        cluster_assignments: torch.Tensor,
        node_positions: torch.Tensor
    ) -> torch.Tensor:
        """Penalize spatially dispersed clusters"""
        # Convert cluster assignments to hard assignments if soft
        if cluster_assignments.dim() > 1:
            cluster_ids = cluster_assignments.argmax(dim=-1)
        else:
            cluster_ids = cluster_assignments
            
        num_clusters = cluster_ids.max().item() + 1
        total_loss = 0
        
        for c in range(num_clusters):
            mask = (cluster_ids == c).float()
            if mask.sum() < 2:
                continue
                
            # Get positions of nodes in this cluster
            cluster_positions = node_positions * mask.unsqueeze(-1)
            
            # Calculate centroid
            centroid = cluster_positions.sum(dim=0) / (mask.sum() + 1e-6)
            
            # Calculate average distance to centroid
            distances = torch.norm(cluster_positions - centroid.unsqueeze(0), dim=-1)
            avg_distance = (distances * mask).sum() / (mask.sum() + 1e-6)
            
            total_loss = total_loss + avg_distance
            
        return total_loss / (num_clusters + 1e-6)
    
    def _clustering_quality_loss(
        self,
        energy_sharing: torch.Tensor,
        cluster_assignments: torch.Tensor
    ) -> torch.Tensor:
        """Encourage energy sharing within clusters, discourage between clusters"""
        # Get cluster affinity matrix
        if cluster_assignments.dim() > 1:
            # Soft assignments: use probability product
            affinity = torch.matmul(cluster_assignments, cluster_assignments.T)
        else:
            # Hard assignments: binary same-cluster matrix
            cluster_ids = cluster_assignments.unsqueeze(1)
            affinity = (cluster_ids == cluster_ids.T).float()
        
        # Normalize energy sharing
        if energy_sharing.dim() > 2:
            energy_sharing = energy_sharing.mean(dim=-1)  # Average over time
            
        # Encourage within-cluster sharing, discourage between-cluster
        within_cluster = (energy_sharing * affinity).sum()
        between_cluster = (energy_sharing * (1 - affinity)).sum()
        
        # We want to maximize within and minimize between
        loss = -within_cluster + between_cluster
        return loss / (energy_sharing.numel() + 1e-6)


class PeakReductionLoss(nn.Module):
    """
    Loss for reducing peak demand through clustering
    """
    
    def __init__(self, reduction_target: float = 0.25):
        super().__init__()
        self.reduction_target = reduction_target
        
    def forward(
        self,
        individual_peaks: torch.Tensor,
        cluster_peaks: torch.Tensor,
        cluster_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate peak reduction loss
        
        Args:
            individual_peaks: Peak demand per building [N]
            cluster_peaks: Peak demand per cluster [K]
            cluster_probs: Cluster assignments [N, K]
            
        Returns:
            Peak reduction loss
        """
        # Calculate expected peak reduction per cluster
        K = cluster_probs.shape[1]
        
        total_loss = 0
        for k in range(K):
            # Buildings in this cluster (weighted by probability)
            weights = cluster_probs[:, k]
            
            # Sum of individual peaks (weighted)
            individual_sum = (individual_peaks * weights).sum()
            
            # Cluster peak should be less than sum of individuals
            if individual_sum > 0:
                reduction_ratio = cluster_peaks[k] / (individual_sum + 1e-8)
                
                # Loss: penalize if reduction is less than target
                loss = F.relu(reduction_ratio - (1 - self.reduction_target))
                total_loss = total_loss + loss
                
        return total_loss / K


class SelfSufficiencyLoss(nn.Module):
    """
    Loss for maximizing self-sufficiency within clusters
    """
    
    def __init__(self, target_sufficiency: float = 0.65):
        super().__init__()
        self.target = target_sufficiency
        
    def forward(
        self,
        generation: torch.Tensor,
        demand: torch.Tensor,
        cluster_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate self-sufficiency loss
        
        Args:
            generation: Generation profiles [N, T]
            demand: Demand profiles [N, T]
            cluster_probs: Cluster assignments [N, K]
            
        Returns:
            Self-sufficiency loss
        """
        N, T = demand.shape
        K = cluster_probs.shape[1]
        
        total_loss = 0
        for k in range(K):
            weights = cluster_probs[:, k:k+1]  # [N, 1]
            
            # Weighted generation and demand
            cluster_gen = (generation * weights).sum(dim=0)  # [T]
            cluster_demand = (demand * weights).sum(dim=0)  # [T]
            
            # Self-sufficiency: min(generation, demand) / demand
            self_consumed = torch.minimum(cluster_gen, cluster_demand)
            sufficiency = self_consumed.sum() / (cluster_demand.sum() + 1e-8)
            
            # Loss: penalize if below target
            loss = F.relu(self.target - sufficiency)
            total_loss = total_loss + loss
            
        return total_loss / K


class ClusterQualityLoss(nn.Module):
    """
    Loss for maintaining cluster quality (size, balance, modularity)
    """
    
    def __init__(
        self,
        min_size: int = 3,
        max_size: int = 20,
        balance_weight: float = 0.5
    ):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.balance_weight = balance_weight
        
    def forward(
        self,
        cluster_probs: torch.Tensor,
        adjacency: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate cluster quality losses
        
        Args:
            cluster_probs: Soft cluster assignments [N, K]
            adjacency: Adjacency matrix for modularity
            
        Returns:
            Total loss and components
        """
        losses = {}
        
        # Cluster sizes (soft)
        cluster_sizes = cluster_probs.sum(dim=0)
        
        # Size constraint loss
        size_loss = (
            F.relu(self.min_size - cluster_sizes).sum() +
            F.relu(cluster_sizes - self.max_size).sum()
        ) / len(cluster_sizes)
        losses['size'] = size_loss
        
        # Balance loss (encourage equal-sized clusters)
        mean_size = cluster_sizes.mean()
        balance_loss = ((cluster_sizes - mean_size) ** 2).mean() / (mean_size ** 2 + 1e-8)
        losses['balance'] = balance_loss
        
        # Modularity loss (if adjacency provided)
        if adjacency is not None:
            modularity_loss = self._modularity_loss(cluster_probs, adjacency)
            losses['modularity'] = modularity_loss
        else:
            modularity_loss = 0
            
        total_loss = (
            size_loss +
            self.balance_weight * balance_loss +
            modularity_loss
        )
        
        return total_loss, losses
    
    def _modularity_loss(
        self,
        cluster_probs: torch.Tensor,
        adjacency: torch.Tensor
    ) -> torch.Tensor:
        """
        Modularity-based loss for cluster quality
        High modularity = good cluster structure
        """
        # Check if adjacency is edge_index format (2, E) or adjacency matrix (N, N)
        if adjacency.dim() == 2 and adjacency.shape[0] == 2:
            # Convert edge_index to adjacency matrix
            num_nodes = cluster_probs.shape[0]
            adj_matrix = torch.zeros(num_nodes, num_nodes, device=adjacency.device)
            edge_index = adjacency.long()
            adj_matrix[edge_index[0], edge_index[1]] = 1.0
            # Make symmetric if not already
            adjacency = adj_matrix + adj_matrix.t()
            adjacency[adjacency > 1] = 1  # Remove duplicate edges
        
        # Compute modularity matrix
        degrees = adjacency.sum(dim=1)
        m = adjacency.sum() / 2
        
        # Modularity matrix: B_ij = A_ij - (d_i * d_j) / (2m)
        expected = torch.outer(degrees, degrees) / (2 * m + 1e-8)
        modularity_matrix = adjacency - expected
        
        # Trace of S^T B S where S is cluster assignment
        cluster_modularity = torch.matmul(
            torch.matmul(cluster_probs.t(), modularity_matrix),
            cluster_probs
        )
        
        # We want to maximize modularity, but loss should be positive
        # Use 1 - modularity so high modularity -> low loss
        modularity_score = torch.trace(cluster_modularity) / (2 * m + 1e-8)
        loss = 1.0 - torch.clamp(modularity_score, -1.0, 1.0)  # Clamp to valid range
        
        return loss


class UnifiedEnergyLoss(nn.Module):
    """
    Unified loss combining all energy GNN objectives
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        config = config or {}
        
        # Initialize component losses
        self.complementarity_loss = ComplementarityLoss(
            correlation_weight=config.get('correlation_weight', 1.0),
            separation_weight=config.get('separation_weight', 0.5),
            diversity_weight=config.get('diversity_weight', 0.3)
        )
        
        self.energy_balance_loss = EnergyBalanceLoss(
            balance_weight=config.get('balance_weight', 1.0),
            spatial_weight=config.get('spatial_weight', 0.5),
            clustering_weight=config.get('clustering_weight', 0.3)
        )
        
        self.peak_reduction_loss = PeakReductionLoss(
            reduction_target=config.get('peak_reduction_target', 0.25)
        )
        
        self.self_sufficiency_loss = SelfSufficiencyLoss(
            target_sufficiency=config.get('self_sufficiency_target', 0.65)
        )
        
        self.cluster_quality_loss = ClusterQualityLoss(
            min_size=config.get('min_cluster_size', 3),
            max_size=config.get('max_cluster_size', 20)
        )
        
        # Loss weights
        self.weights = {
            'complementarity': config.get('complementarity_weight', 1.0),
            'physics': config.get('physics_weight', 0.5),
            'peak': config.get('peak_weight', 0.3),
            'sufficiency': config.get('sufficiency_weight', 0.3),
            'quality': config.get('quality_weight', 0.2)
        }
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        graph_data: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate total loss
        
        Args:
            predictions: Model predictions
            targets: Target values
            graph_data: Additional graph information
            
        Returns:
            Total loss and component dictionary
        """
        all_losses = {}
        total_loss = 0
        
        # Complementarity loss (main objective)
        if 'clustering_cluster_probs' in predictions:
            comp_loss, comp_components = self.complementarity_loss(
                embeddings=predictions.get('embeddings'),
                cluster_probs=predictions['clustering_cluster_probs'],
                temporal_profiles=targets.get('temporal_profiles')
            )
            all_losses['complementarity'] = comp_loss
            all_losses.update({f'comp_{k}': v for k, v in comp_components.items()})
            total_loss += self.weights['complementarity'] * comp_loss
        
        # Energy balance and spatial constraints
        if self.weights.get('physics', 0) > 0:
            phys_loss, phys_components = self.energy_balance_loss(
                energy_sharing=predictions.get('energy_sharing', torch.zeros(1)),
                cluster_assignments=predictions.get('clustering_cluster_probs'),
                node_positions=graph_data.get('positions') if graph_data else None,
                consumption=targets.get('consumption'),
                generation=targets.get('generation')
            )
            all_losses['physics'] = phys_loss
            all_losses.update({f'phys_{k}': v for k, v in phys_components.items()})
            total_loss += self.weights['physics'] * phys_loss
        else:
            all_losses['physics'] = torch.tensor(0.0, device=predictions.get('clustering_cluster_probs', torch.zeros(1)).device)
        
        # Peak reduction
        if 'individual_peaks' in targets and 'cluster_peaks' in predictions:
            peak_loss = self.peak_reduction_loss(
                individual_peaks=targets['individual_peaks'],
                cluster_peaks=predictions['cluster_peaks'],
                cluster_probs=predictions['clustering_cluster_probs']
            )
            all_losses['peak_reduction'] = peak_loss
            total_loss += self.weights['peak'] * peak_loss
        
        # Self-sufficiency
        if 'generation' in targets and 'demand' in targets:
            suff_loss = self.self_sufficiency_loss(
                generation=targets['generation'],
                demand=targets['demand'],
                cluster_probs=predictions['clustering_cluster_probs']
            )
            all_losses['self_sufficiency'] = suff_loss
            total_loss += self.weights['sufficiency'] * suff_loss
        
        # Cluster quality
        if 'clustering_cluster_probs' in predictions:
            quality_loss, quality_components = self.cluster_quality_loss(
                cluster_probs=predictions['clustering_cluster_probs'],
                adjacency=graph_data.get('adjacency') if graph_data else None
            )
            all_losses['cluster_quality'] = quality_loss
            all_losses.update({f'quality_{k}': v for k, v in quality_components.items()})
            total_loss += self.weights['quality'] * quality_loss
        
        all_losses['total'] = total_loss
        
        return total_loss, all_losses


class DiscoveryLoss(nn.Module):
    """
    Unsupervised loss for energy community discovery
    No ground truth required - focuses on pattern discovery
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        config = config or {}
        
        self.alpha_comp = config.get('alpha_complementarity', 2.0)
        self.alpha_physics = config.get('alpha_physics', 1.0)
        self.alpha_quality = config.get('alpha_clustering', 1.5)
        self.alpha_peak = config.get('alpha_peak', 1.0)
        self.alpha_coverage = config.get('alpha_coverage', 0.5)
        self.alpha_temporal = config.get('alpha_temporal', 0.3)
        
        # Component losses
        self.complementarity_loss = ComplementarityLoss()
        self.cluster_quality_loss = ClusterQualityLoss(
            min_size=config.get('min_cluster_size', 3),
            max_size=config.get('max_cluster_size', 20)
        )
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        physics_data: Dict[str, torch.Tensor],
        batch: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate discovery loss without ground truth
        
        Args:
            predictions: Model predictions including clusters, flows, complementarity
            physics_data: Physical measurements (demand, generation, etc.)
            batch: Batch information including edge indices
            
        Returns:
            Total loss and component dictionary
        """
        losses = {}
        
        # 1. Complementarity Loss (maximize negative correlation)
        if 'complementarity' in predictions:
            comp_matrix = predictions['complementarity']
            # Want negative values (complementary patterns)
            # But loss should be positive: use ReLU on positive correlations
            losses['complementarity'] = torch.mean(
                torch.relu(comp_matrix)  # Penalize positive correlations
            )
        
        # 2. Physics Constraint Loss
        if 'energy_flow' in predictions and batch is not None:
            losses['physics'] = self._physics_constraint_loss(
                predictions, physics_data, batch
            )
        
        # 3. Clustering Quality Loss
        if 'clusters' in predictions:
            S = predictions['clusters']  # Soft assignment matrix
            
            # Size constraints (3-20 buildings per cluster)
            cluster_sizes = torch.sum(S, dim=0)
            size_penalty = torch.relu(3 - cluster_sizes) + torch.relu(cluster_sizes - 20)
            losses['size'] = torch.mean(size_penalty)
            
            # Entropy regularization for crisp assignments
            entropy = -torch.mean(S * torch.log(S + 1e-8))
            losses['entropy'] = entropy
        
        # 4. Peak Reduction Loss (target 25% reduction)
        if 'clusters' in predictions and 'demand' in physics_data:
            losses['peak'] = self._peak_reduction_loss(
                physics_data['demand'], predictions['clusters']
            )
        
        # 5. Coverage Loss (penalize orphan buildings)
        if 'clusters' in predictions:
            max_assignment = torch.max(predictions['clusters'], dim=1)[0]
            orphans = torch.sum(max_assignment < 0.1) / len(max_assignment)
            losses['coverage'] = orphans
        
        # 6. Temporal Stability Loss (if multiple timesteps)
        if 'clusters_prev' in predictions and 'clusters' in predictions:
            losses['temporal'] = self._temporal_stability_loss(
                predictions['clusters_prev'], predictions['clusters']
            )
        
        # Combine losses
        total_loss = (
            self.alpha_comp * losses.get('complementarity', 0) +
            self.alpha_physics * losses.get('physics', 0) +
            self.alpha_quality * (losses.get('size', 0) + losses.get('entropy', 0)) +
            self.alpha_peak * losses.get('peak', 0) +
            self.alpha_coverage * losses.get('coverage', 0) +
            self.alpha_temporal * losses.get('temporal', 0)
        )
        
        losses['total'] = total_loss
        return total_loss, losses


class SolarROILoss(nn.Module):
    """
    Loss function for solar ROI category prediction
    Trains model to predict payback period categories
    """
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        # Categories: excellent (<5yr), good (5-7yr), fair (7-10yr), poor (>10yr)
        self.class_weights = class_weights
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate ROI category loss
        
        Args:
            predictions: Predicted ROI categories [N, 4]
            targets: True ROI categories [N] (0=excellent, 1=good, 2=fair, 3=poor, -1=unknown)
        """
        # Filter out unknown labels
        mask = targets >= 0
        if not mask.any():
            return torch.tensor(0.0, device=predictions.device)
        
        return F.cross_entropy(
            predictions[mask],
            targets[mask],
            weight=self.class_weights
        )


class ClusterQualityLoss(nn.Module):
    """
    Semi-supervised loss for cluster quality
    Learns what makes a good energy community from labeled examples
    """
    
    def __init__(self):
        super().__init__()
        self.label_weights = {
            'excellent': 1.0,
            'good': 0.7,
            'fair': 0.4,
            'poor': 0.1
        }
        
    def forward(
        self,
        cluster_assignments: torch.Tensor,
        cluster_labels: Dict[int, str],
        cluster_metrics: Optional[Dict[int, float]] = None
    ) -> torch.Tensor:
        """
        Calculate cluster quality loss
        
        Args:
            cluster_assignments: Soft cluster assignments [N, K]
            cluster_labels: Quality labels for clusters {cluster_id: label}
            cluster_metrics: Optional metrics for each cluster
        """
        if not cluster_labels:
            return torch.tensor(0.0, device=cluster_assignments.device)
        
        loss = 0.0
        K = cluster_assignments.shape[1]
        
        for cluster_id in range(K):
            if cluster_id in cluster_labels:
                label = cluster_labels[cluster_id]
                target_weight = self.label_weights[label]
                
                # Cluster size (soft membership sum)
                cluster_size = cluster_assignments[:, cluster_id].sum()
                
                # Encourage good clusters to be larger, poor clusters to be smaller
                size_loss = -torch.log(cluster_size / cluster_assignments.shape[0] + 1e-8) * target_weight
                loss += size_loss
                
                # If metrics available, use them
                if cluster_metrics and cluster_id in cluster_metrics:
                    metric_value = cluster_metrics[cluster_id]
                    # Encourage high metrics for good clusters
                    metric_loss = (1 - metric_value) * target_weight
                    loss += metric_loss * 0.5
        
        return loss / K
    
    def _physics_constraint_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        physics_data: Dict[str, torch.Tensor],
        batch: Dict
    ) -> torch.Tensor:
        """
        Ensure physical constraints are satisfied
        """
        loss = 0.0
        
        # Energy balance per node
        if 'energy_flow' in predictions:
            flows = predictions['energy_flow']
            edge_index = batch.get('edge_index')
            
            if edge_index is not None:
                # Calculate net flow per node
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
                
                # Outgoing flows
                out_flow = scatter_add(
                    flows,
                    edge_index[0],
                    dim=0,
                    dim_size=physics_data['demand'].size(0)
                )
                
                # Incoming flows
                in_flow = scatter_add(
                    flows,
                    edge_index[1],
                    dim=0,
                    dim_size=physics_data['demand'].size(0)
                )
                
                # Net load should match flow difference
                net_load = physics_data.get('demand', 0) - physics_data.get('generation', 0)
                balance_violation = torch.abs(in_flow - out_flow - net_load)
                loss = torch.mean(balance_violation)
        
        # Transformer capacity constraints
        if 'transformer_mask' in batch and 'clusters' in predictions:
            loss += self._check_transformer_violations(
                predictions['clusters'], batch['transformer_mask']
            )
        
        return loss
    
    def _peak_reduction_loss(
        self,
        demand: torch.Tensor,
        clusters: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate peak reduction loss
        """
        # Original peak (sum of individual peaks)
        peak_original = torch.max(demand, dim=-1)[0] if demand.dim() > 1 else torch.max(demand)
        
        # Clustered peak (peak of aggregated demand)
        if clusters.dim() == 2:  # Soft assignments
            # Aggregate demand per cluster
            cluster_demand = torch.matmul(clusters.T, demand.unsqueeze(-1)).squeeze()
            peak_clustered = torch.max(cluster_demand)
        else:
            peak_clustered = peak_original  # Fallback
        
        # Calculate reduction
        peak_reduction = (peak_original - peak_clustered) / (peak_original + 1e-8)
        
        # Penalize if reduction is less than 25%
        return torch.relu(0.25 - peak_reduction)
    
    def _check_transformer_violations(
        self,
        clusters: torch.Tensor,
        transformer_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Check for transformer boundary violations
        """
        # Clusters should not cross transformer boundaries
        # transformer_mask[i, j] = 0 if nodes i and j are in different transformers
        
        if clusters.dim() == 2:  # Soft assignments
            # Check co-membership probability
            co_membership = torch.matmul(clusters, clusters.T)
            violations = co_membership * (1 - transformer_mask)
            return torch.mean(violations)
        
        return torch.tensor(0.0, device=clusters.device)
    
    def _temporal_stability_loss(
        self,
        clusters_prev: torch.Tensor,
        clusters_curr: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure temporal stability between consecutive timesteps
        """
        # Convert soft to hard assignments
        if clusters_prev.dim() == 2:
            prev_hard = torch.argmax(clusters_prev, dim=1)
            curr_hard = torch.argmax(clusters_curr, dim=1)
        else:
            prev_hard = clusters_prev
            curr_hard = clusters_curr
        
        # Calculate similarity (simple version - could use ARI)
        same_cluster = (prev_hard == curr_hard).float()
        stability = torch.mean(same_cluster)
        
        # Want high stability, so minimize 1 - stability
        return 1.0 - stability