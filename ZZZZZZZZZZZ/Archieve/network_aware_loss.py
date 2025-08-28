"""
Network-aware loss functions that consider multi-hop effects
Proves GNN value beyond simple correlation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class NetworkImpactLoss(nn.Module):
    """
    Loss that considers multi-hop network effects
    Not just internal cluster quality but network-wide impact
    """
    
    def __init__(
        self,
        hop_weights: List[float] = [1.0, 0.5, 0.25],
        congestion_weight: float = 0.5,
        boundary_weight: float = 0.3
    ):
        super().__init__()
        self.hop_weights = hop_weights
        self.congestion_weight = congestion_weight
        self.boundary_weight = boundary_weight
        
    def forward(
        self,
        cluster_assignments: torch.Tensor,
        network_embeddings: torch.Tensor,
        hop_features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        transformer_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate network impact loss
        
        Args:
            cluster_assignments: Soft cluster assignments [N, K]
            network_embeddings: Node embeddings [N, D]
            hop_features: Features at each hop distance
            edge_index: Graph connectivity
            transformer_mask: Mask for transformer boundaries
            
        Returns:
            Total loss and component dictionary
        """
        losses = {}
        
        # 1. Multi-hop impact loss
        hop_loss = self._multi_hop_impact_loss(
            cluster_assignments, hop_features, edge_index
        )
        losses['multi_hop'] = hop_loss
        
        # 2. Network congestion relief loss
        congestion_loss = self._congestion_relief_loss(
            cluster_assignments, network_embeddings, edge_index
        )
        losses['congestion'] = congestion_loss
        
        # 3. Transformer boundary respect loss
        if transformer_mask is not None:
            boundary_loss = self._boundary_respect_loss(
                cluster_assignments, transformer_mask
            )
            losses['boundary'] = boundary_loss
        else:
            boundary_loss = 0
        
        # 4. Cross-cluster information flow loss
        flow_loss = self._information_flow_loss(
            cluster_assignments, hop_features
        )
        losses['information_flow'] = flow_loss
        
        # Combine losses
        total_loss = (
            hop_loss +
            self.congestion_weight * congestion_loss +
            self.boundary_weight * boundary_loss +
            flow_loss
        )
        
        return total_loss, losses
    
    def _multi_hop_impact_loss(
        self,
        cluster_assignments: torch.Tensor,
        hop_features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure clusters have impact beyond immediate neighbors
        """
        total_loss = 0
        K = cluster_assignments.shape[1]
        
        for k in range(K):
            cluster_weights = cluster_assignments[:, k:k+1]
            
            # Calculate impact at each hop
            for hop_idx, (hop_name, hop_feat) in enumerate(hop_features.items()):
                if 'hop_' in hop_name and 'features' in hop_name:
                    # Weight features by cluster membership
                    weighted_features = hop_feat * cluster_weights
                    
                    # Calculate variance (want high variance = diverse impact)
                    variance = torch.var(weighted_features, dim=0).mean()
                    
                    # Loss decreases with variance (we want high variance)
                    hop_loss = 1.0 / (variance + 1e-8)
                    
                    # Weight by hop distance
                    if hop_idx < len(self.hop_weights):
                        hop_loss *= self.hop_weights[hop_idx]
                    
                    total_loss += hop_loss
        
        return total_loss / K
    
    def _congestion_relief_loss(
        self,
        cluster_assignments: torch.Tensor,
        network_embeddings: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Reward clusters that relieve network congestion
        """
        # Estimate congestion from embeddings
        # Higher similarity between connected nodes = higher congestion
        row, col = edge_index
        edge_similarity = F.cosine_similarity(
            network_embeddings[row], 
            network_embeddings[col],
            dim=-1
        )
        
        # Calculate congestion per node (average edge similarity)
        node_congestion = torch.zeros(network_embeddings.size(0), device=network_embeddings.device)
        node_congestion.index_add_(0, row, edge_similarity)
        node_degrees = torch.bincount(row, minlength=network_embeddings.size(0)).float() + 1e-8
        node_congestion = node_congestion / node_degrees
        
        # Clusters should reduce congestion
        K = cluster_assignments.shape[1]
        total_loss = 0
        
        for k in range(K):
            cluster_weights = cluster_assignments[:, k]
            
            # Weighted congestion in cluster
            cluster_congestion = (node_congestion * cluster_weights).sum() / (cluster_weights.sum() + 1e-8)
            
            # Penalize high congestion
            total_loss += cluster_congestion
        
        return total_loss / K
    
    def _boundary_respect_loss(
        self,
        cluster_assignments: torch.Tensor,
        transformer_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure clusters respect transformer boundaries
        """
        # Co-membership probability
        co_membership = torch.matmul(cluster_assignments, cluster_assignments.t())
        
        # Violations: high co-membership across transformer boundaries
        violations = co_membership * (1 - transformer_mask)
        
        return violations.mean()
    
    def _information_flow_loss(
        self,
        cluster_assignments: torch.Tensor,
        hop_features: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Ensure information flows properly through cluster hierarchy
        """
        # Information should decay with hop distance
        flow_loss = 0
        prev_norm = None
        
        for hop_idx, (hop_name, hop_feat) in enumerate(hop_features.items()):
            if 'hop_' in hop_name and 'features' in hop_name:
                current_norm = torch.norm(hop_feat, dim=-1).mean()
                
                if prev_norm is not None:
                    # Should decay
                    if current_norm > prev_norm:
                        flow_loss += (current_norm - prev_norm)
                
                prev_norm = current_norm
        
        return flow_loss


class CascadePredictionLoss(nn.Module):
    """
    Loss for predicting intervention cascade effects
    Trains GNN to understand how changes propagate
    """
    
    def __init__(
        self,
        hop_weights: List[float] = [1.0, 0.5, 0.25],
        energy_weight: float = 1.0,
        economic_weight: float = 0.5,
        temporal_weight: float = 0.3
    ):
        super().__init__()
        self.hop_weights = hop_weights
        self.energy_weight = energy_weight
        self.economic_weight = economic_weight
        self.temporal_weight = temporal_weight
        
    def forward(
        self,
        predicted_cascade: Dict[str, torch.Tensor],
        actual_cascade: Dict[str, torch.Tensor],
        intervention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate cascade prediction loss
        
        Args:
            predicted_cascade: GNN predictions of cascade effects
            actual_cascade: Actual simulated cascade effects
            intervention_mask: Which nodes were intervened
            
        Returns:
            Total loss and components
        """
        losses = {}
        
        # 1. Hop-wise cascade prediction
        hop_loss = 0
        for hop in range(len(self.hop_weights)):
            hop_key = f'hop_{hop+1}_impact'
            hop_key_alt = f'hop_{hop+1}'  # Alternative format
            
            # Check both formats
            pred_hop = None
            actual_hop = None
            
            # Get predicted hop values
            if hop_key in predicted_cascade:
                pred_hop = predicted_cascade[hop_key]
            elif hop_key_alt in predicted_cascade:
                # Sum across effect types if dict format
                if isinstance(predicted_cascade[hop_key_alt], dict):
                    # Get a reference tensor for shape and device
                    ref_tensor = None
                    for effect_type in ['energy_impact', 'congestion_relief', 'economic_value']:
                        if effect_type in predicted_cascade[hop_key_alt]:
                            ref_tensor = predicted_cascade[hop_key_alt][effect_type]
                            if isinstance(ref_tensor, torch.Tensor):
                                break
                    
                    if ref_tensor is not None:
                        pred_hop = torch.zeros_like(ref_tensor, dtype=torch.float32)
                        for effect_type in ['energy_impact', 'congestion_relief', 'economic_value']:
                            if effect_type in predicted_cascade[hop_key_alt]:
                                effect = predicted_cascade[hop_key_alt][effect_type]
                                if isinstance(effect, torch.Tensor):
                                    # Ensure matching dimensions
                                    if effect.shape != pred_hop.shape:
                                        if effect.numel() == 1:  # Scalar tensor
                                            effect = effect.expand_as(pred_hop)
                                        elif pred_hop.numel() == 1:  # pred_hop is scalar
                                            pred_hop = pred_hop.expand_as(effect)
                                    pred_hop = pred_hop + effect.abs()
                    else:
                        pred_hop = None
                else:
                    pred_hop = predicted_cascade[hop_key_alt]
            
            # Get actual hop values
            if hop_key in actual_cascade:
                actual_hop = actual_cascade[hop_key]
            elif hop_key_alt in actual_cascade:
                # Sum across effect types if dict format
                if isinstance(actual_cascade[hop_key_alt], dict):
                    # Get a reference tensor for shape and device
                    ref_tensor = None
                    for effect_type in ['energy_impact', 'congestion_relief', 'economic_value']:
                        if effect_type in actual_cascade[hop_key_alt]:
                            ref_tensor = actual_cascade[hop_key_alt][effect_type]
                            if isinstance(ref_tensor, torch.Tensor):
                                break
                    
                    if ref_tensor is not None:
                        actual_hop = torch.zeros_like(ref_tensor, dtype=torch.float32)
                        for effect_type in ['energy_impact', 'congestion_relief', 'economic_value']:
                            if effect_type in actual_cascade[hop_key_alt]:
                                effect = actual_cascade[hop_key_alt][effect_type]
                                if isinstance(effect, torch.Tensor):
                                    # Ensure matching dimensions
                                    if effect.shape != actual_hop.shape:
                                        if effect.numel() == 1:  # Scalar tensor
                                            effect = effect.expand_as(actual_hop)
                                        elif actual_hop.numel() == 1:  # actual_hop is scalar
                                            actual_hop = actual_hop.expand_as(effect)
                                    actual_hop = actual_hop + effect.abs()
                    else:
                        actual_hop = None
                else:
                    actual_hop = actual_cascade[hop_key_alt]
            
            if pred_hop is not None and actual_hop is not None:
                # Ensure tensors
                if not isinstance(pred_hop, torch.Tensor):
                    pred_hop = torch.tensor(pred_hop, dtype=torch.float32, device=intervention_mask.device)
                if not isinstance(actual_hop, torch.Tensor):
                    actual_hop = torch.tensor(actual_hop, dtype=torch.float32, device=intervention_mask.device)
                
                # Ensure matching shapes for MSE loss
                if pred_hop.shape != actual_hop.shape:
                    # Debug logging
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.debug(f"Shape mismatch - pred: {pred_hop.shape}, actual: {actual_hop.shape}")
                    
                    # Handle different scenarios
                    if pred_hop.dim() == 2 and actual_hop.dim() == 1:
                        # Model outputs [N, 3] (3 impact types), simulator outputs [N]
                        # Sum across impact types to get single value per node
                        pred_hop = pred_hop.sum(dim=-1)
                    elif actual_hop.dim() == 2 and pred_hop.dim() == 1:
                        # Opposite case
                        actual_hop = actual_hop.sum(dim=-1)
                    elif pred_hop.dim() == 2 and actual_hop.dim() == 2:
                        # Both are 2D but different shapes
                        if pred_hop.shape[-1] != actual_hop.shape[-1]:
                            # Different number of features - sum to reduce
                            pred_hop = pred_hop.sum(dim=-1)
                            actual_hop = actual_hop.sum(dim=-1)
                    
                    # After dimension reduction, check again
                    if pred_hop.shape != actual_hop.shape:
                        # If one is scalar and other is not, broadcast
                        if pred_hop.numel() == 1 and actual_hop.numel() > 1:
                            pred_hop = pred_hop.expand_as(actual_hop)
                        elif actual_hop.numel() == 1 and pred_hop.numel() > 1:
                            actual_hop = actual_hop.expand_as(pred_hop)
                        elif pred_hop.shape[0] != actual_hop.shape[0]:
                            # Different batch sizes - take the minimum
                            min_size = min(pred_hop.shape[0], actual_hop.shape[0])
                            pred_hop = pred_hop[:min_size]
                            actual_hop = actual_hop[:min_size]
                    
                # MSE weighted by hop distance
                loss = F.mse_loss(pred_hop, actual_hop)
                weighted_loss = loss * self.hop_weights[hop]
                hop_loss += weighted_loss
                losses[f'{hop_key}_loss'] = weighted_loss
        
        losses['hop_cascade'] = hop_loss
        
        # 2. Total network impact prediction
        if 'total_network_impact' in predicted_cascade and 'total_network_impact' in actual_cascade:
            impact_loss = F.mse_loss(
                predicted_cascade['total_network_impact'],
                actual_cascade['total_network_impact']
            )
            losses['network_impact'] = impact_loss
        else:
            impact_loss = 0
        
        # 3. Temporal evolution (if available)
        if 'temporal_evolution' in predicted_cascade:
            temporal_loss = self._temporal_consistency_loss(
                predicted_cascade['temporal_evolution'],
                actual_cascade.get('temporal_evolution')
            )
            losses['temporal'] = temporal_loss
        else:
            temporal_loss = 0
        
        # 4. Intervention value ranking loss
        value_loss = self._intervention_ranking_loss(
            predicted_cascade, actual_cascade, intervention_mask
        )
        losses['ranking'] = value_loss
        
        # Combine losses
        total_loss = (
            self.energy_weight * hop_loss +
            self.economic_weight * impact_loss +
            self.temporal_weight * temporal_loss +
            value_loss
        )
        
        return total_loss, losses
    
    def _temporal_consistency_loss(
        self,
        predicted_temporal: torch.Tensor,
        actual_temporal: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """
        Ensure temporal predictions are consistent
        """
        if actual_temporal is None:
            # Use smoothness as proxy
            diff = predicted_temporal[:, 1:] - predicted_temporal[:, :-1]
            return torch.mean(diff ** 2)
        else:
            return F.mse_loss(predicted_temporal, actual_temporal)
    
    def _intervention_ranking_loss(
        self,
        predicted: Dict[str, torch.Tensor],
        actual: Dict[str, torch.Tensor],
        intervention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Ensure GNN ranks interventions correctly by network value
        """
        # Get intervention values
        if 'intervention_values' in predicted:
            pred_values = predicted['intervention_values']
        else:
            # Calculate predicted values from hop impacts
            pred_values = None
            for i in range(3):
                hop_key = f'hop_{i+1}'
                if hop_key in predicted:
                    # Sum over effect types for this hop if dict-like
                    if isinstance(predicted[hop_key], dict):
                        hop_impact = None
                        for effect_type in ['energy_impact', 'congestion_relief', 'economic_value']:
                            if effect_type in predicted[hop_key]:
                                effect = predicted[hop_key][effect_type]
                                # Ensure effect is a tensor
                                if not isinstance(effect, torch.Tensor):
                                    # Convert scalar to tensor if needed
                                    device = intervention_mask.device if isinstance(intervention_mask, torch.Tensor) else 'cpu'
                                    effect = torch.tensor(effect, device=device, dtype=torch.float32)
                                # Ensure effect has proper dimensions
                                if effect.dim() == 0:  # scalar tensor
                                    effect = effect.unsqueeze(0)
                                if hop_impact is None:
                                    hop_impact = effect.abs()
                                else:
                                    hop_impact = hop_impact + effect.abs()
                    else:
                        # Direct tensor
                        hop_impact = predicted[hop_key]
                        if not isinstance(hop_impact, torch.Tensor):
                            device = intervention_mask.device if isinstance(intervention_mask, torch.Tensor) else 'cpu'
                            hop_impact = torch.tensor(hop_impact, device=device, dtype=torch.float32)
                        hop_impact = hop_impact.abs()
                    
                    if hop_impact is not None:
                        if pred_values is None:
                            pred_values = hop_impact
                        else:
                            pred_values = pred_values + hop_impact
            
            # If still no values, create zeros
            if pred_values is None:
                # Get device from intervention mask
                device = intervention_mask.device if isinstance(intervention_mask, torch.Tensor) else 'cpu'
                pred_values = torch.zeros(intervention_mask.shape[0], device=device)
        
        if 'intervention_values' in actual:
            actual_values = actual['intervention_values']
        else:
            # Calculate actual values from hop impacts
            actual_values = None
            for i in range(3):
                hop_key = f'hop_{i+1}'
                if hop_key in actual:
                    # Sum over effect types for this hop
                    hop_impact = None
                    for effect_type in ['energy_impact', 'congestion_relief', 'economic_value']:
                        if effect_type in actual[hop_key]:
                            effect = actual[hop_key][effect_type]
                            # Ensure effect is a tensor
                            if not isinstance(effect, torch.Tensor):
                                # Convert scalar to tensor if needed
                                device = intervention_mask.device if isinstance(intervention_mask, torch.Tensor) else 'cpu'
                                effect = torch.tensor(effect, device=device, dtype=torch.float32)
                            # Ensure effect has proper dimensions
                            if effect.dim() == 0:  # scalar tensor
                                effect = effect.unsqueeze(0)
                            if hop_impact is None:
                                hop_impact = effect.abs()
                            else:
                                hop_impact = hop_impact + effect.abs()
                    
                    if hop_impact is not None:
                        if actual_values is None:
                            actual_values = hop_impact
                        else:
                            actual_values = actual_values + hop_impact
            
            # If still no values, create zeros
            if actual_values is None:
                actual_values = torch.zeros_like(pred_values)
        
        # Ranking loss (pairwise)
        n = intervention_mask.sum()
        if n > 1:
            # Get values for intervened nodes
            pred_intervened = pred_values[intervention_mask.bool()]
            actual_intervened = actual_values[intervention_mask.bool()]
            
            # Pairwise ranking loss
            pred_diff = pred_intervened.unsqueeze(1) - pred_intervened.unsqueeze(0)
            actual_diff = actual_intervened.unsqueeze(1) - actual_intervened.unsqueeze(0)
            
            # Margin ranking loss
            ranking_loss = F.relu(1.0 - pred_diff * actual_diff.sign()).mean()
            
            return ranking_loss
        else:
            return torch.tensor(0.0, device=pred_values.device)


class NetworkAwareDiscoveryLoss(nn.Module):
    """
    Combined loss for network-aware unsupervised discovery
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        config = config or {}
        
        # Import existing losses
        from training.loss_functions import ComplementarityLoss, ClusterQualityLoss
        
        # Component losses
        self.complementarity_loss = ComplementarityLoss()
        self.cluster_quality_loss = ClusterQualityLoss(
            min_size=config.get('min_cluster_size', 3),
            max_size=config.get('max_cluster_size', 20)
        )
        self.network_impact_loss = NetworkImpactLoss()
        self.cascade_prediction_loss = CascadePredictionLoss()
        
        # Loss weights (network effects weighted higher)
        self.weights = {
            'complementarity': config.get('complementarity_weight', 1.0),
            'quality': config.get('quality_weight', 0.5),
            'network_impact': config.get('network_impact_weight', 2.0),  # Higher weight
            'cascade': config.get('cascade_weight', 1.5)
        }
        
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        network_data: Dict[str, torch.Tensor],
        cascade_data: Optional[Dict[str, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate total network-aware loss
        
        Args:
            predictions: Model predictions including clusters, network impacts
            network_data: Network state and topology
            cascade_data: Cascade effects if available
            
        Returns:
            Total loss and components
        """
        all_losses = {}
        total_loss = 0
        
        # 1. Original complementarity loss
        if 'clusters' in predictions:
            comp_loss, comp_components = self.complementarity_loss(
                embeddings=predictions.get('embeddings'),
                cluster_probs=predictions['clusters'],
                temporal_profiles=network_data.get('temporal_profiles')
            )
            all_losses['complementarity'] = comp_loss
            all_losses.update({f'comp_{k}': v for k, v in comp_components.items()})
            total_loss += self.weights['complementarity'] * comp_loss
        
        # 2. Cluster quality
        if 'clusters' in predictions:
            quality_loss, quality_components = self.cluster_quality_loss(
                cluster_probs=predictions['clusters']
            )
            all_losses['quality'] = quality_loss
            all_losses.update({f'quality_{k}': v for k, v in quality_components.items()})
            total_loss += self.weights['quality'] * quality_loss
        
        # 3. Network impact loss (NEW - key differentiator)
        if 'hop_features' in predictions:
            network_loss, network_components = self.network_impact_loss(
                cluster_assignments=predictions.get('clusters'),
                network_embeddings=predictions.get('embeddings'),
                hop_features=predictions['hop_features'],
                edge_index=network_data.get('edge_index'),
                transformer_mask=network_data.get('transformer_mask')
            )
            all_losses['network_impact'] = network_loss
            all_losses.update({f'network_{k}': v for k, v in network_components.items()})
            total_loss += self.weights['network_impact'] * network_loss
        
        # 4. Cascade prediction loss (if training with interventions)
        if cascade_data is not None and 'cascade_effects' in predictions:
            cascade_loss, cascade_components = self.cascade_prediction_loss(
                predicted_cascade=predictions['cascade_effects'],
                actual_cascade=cascade_data,
                intervention_mask=network_data.get('intervention_mask', torch.zeros(1))
            )
            all_losses['cascade'] = cascade_loss
            all_losses.update({f'cascade_{k}': v for k, v in cascade_components.items()})
            total_loss += self.weights['cascade'] * cascade_loss
        
        all_losses['total'] = total_loss
        
        return total_loss, all_losses