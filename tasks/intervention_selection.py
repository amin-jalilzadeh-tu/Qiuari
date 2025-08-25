"""
GNN-based intervention selection that considers network value
NOT rule-based - proves GNN adds value through network reasoning
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class NetworkAwareInterventionSelector:
    """
    Selects interventions based on network value predicted by GNN
    Key differentiator: considers multi-hop effects, not just local features
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize selector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Selection parameters
        self.local_weight = config.get('local_weight', 0.3)
        self.network_weight = config.get('network_weight', 0.7)  # Higher weight on network
        self.diversity_bonus = config.get('diversity_bonus', 0.2)
        self.boundary_penalty = config.get('boundary_penalty', 0.5)
        
        # Constraints
        self.min_spacing = config.get('min_spacing', 2)  # Min hops between interventions
        self.respect_boundaries = config.get('respect_boundaries', True)
        
    def rank_interventions(
        self,
        gnn_outputs: Dict[str, torch.Tensor],
        building_features: torch.Tensor,
        edge_index: torch.Tensor,
        existing_interventions: Optional[List[int]] = None
    ) -> torch.Tensor:
        """
        Rank all nodes by intervention value using GNN predictions
        
        Args:
            gnn_outputs: Output from NetworkAwareGNN including:
                - intervention_values: Direct value predictions
                - network_impacts: Multi-hop impact predictions
                - hop_features: Features at different hop distances
                - embeddings: Node embeddings
            building_features: Original building features
            edge_index: Network connectivity
            existing_interventions: Already intervened nodes
            
        Returns:
            Ranking scores for all nodes
        """
        N = building_features.size(0)
        device = building_features.device
        
        # 1. Local value (based on building features)
        local_scores = self._calculate_local_value(building_features)
        
        # 2. Network value from GNN (key differentiator)
        network_scores = self._calculate_network_value(gnn_outputs)
        
        # 3. Strategic position value
        position_scores = self._calculate_position_value(
            gnn_outputs.get('embeddings'),
            edge_index
        )
        
        # 4. Cascade potential
        cascade_scores = self._calculate_cascade_potential(
            gnn_outputs.get('hop_features', {}),
            edge_index
        )
        
        # 5. Complementarity with existing interventions
        if existing_interventions:
            comp_scores = self._calculate_complementarity_bonus(
                gnn_outputs.get('embeddings'),
                existing_interventions,
                edge_index
            )
        else:
            comp_scores = torch.zeros(N, device=device)
        
        # Combine scores (network value weighted higher)
        total_scores = (
            self.local_weight * local_scores +
            self.network_weight * network_scores +
            0.2 * position_scores +
            0.3 * cascade_scores +
            self.diversity_bonus * comp_scores
        )
        
        # Apply constraints
        if existing_interventions:
            total_scores = self._apply_spacing_constraint(
                total_scores, existing_interventions, edge_index
            )
        
        if self.respect_boundaries and 'boundary_violations' in gnn_outputs:
            total_scores = self._apply_boundary_penalty(
                total_scores, gnn_outputs['boundary_violations']
            )
        
        return total_scores
    
    def select_optimal_set(
        self,
        ranking_scores: torch.Tensor,
        k: int = 5,
        edge_index: Optional[torch.Tensor] = None,
        constraints: Optional[Dict] = None
    ) -> List[int]:
        """
        Select k interventions that maximize network benefit
        Considers interactions between interventions
        
        Args:
            ranking_scores: Scores for all nodes
            k: Number of interventions to select
            edge_index: Network connectivity for interaction constraints
            constraints: Additional constraints (budget, capacity, etc.)
            
        Returns:
            List of selected node indices
        """
        selected = []
        remaining_scores = ranking_scores.clone()
        
        for _ in range(k):
            # Select highest scoring node
            if len(selected) == 0:
                best_idx = torch.argmax(remaining_scores).item()
            else:
                # Consider interaction effects with already selected
                interaction_adjusted = self._adjust_for_interactions(
                    remaining_scores, selected, edge_index
                )
                best_idx = torch.argmax(interaction_adjusted).item()
            
            selected.append(best_idx)
            
            # Update scores considering new selection
            remaining_scores = self._update_scores_after_selection(
                remaining_scores, best_idx, edge_index
            )
            
            # Zero out selected node
            remaining_scores[best_idx] = -float('inf')
        
        return selected
    
    def evaluate_selection(
        self,
        selected_nodes: List[int],
        gnn_outputs: Dict[str, torch.Tensor],
        baseline_selection: List[int],
        edge_index: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compare GNN selection to baseline (e.g., rule-based)
        
        Args:
            selected_nodes: GNN-selected nodes
            gnn_outputs: GNN predictions
            baseline_selection: Baseline selection (e.g., by energy label)
            edge_index: Network connectivity
            
        Returns:
            Comparison metrics
        """
        metrics = {}
        
        # 1. Total network impact
        gnn_impact = self._calculate_total_impact(
            selected_nodes, gnn_outputs, edge_index
        )
        baseline_impact = self._calculate_total_impact(
            baseline_selection, gnn_outputs, edge_index
        )
        
        metrics['gnn_network_impact'] = gnn_impact
        metrics['baseline_network_impact'] = baseline_impact
        # Calculate improvement safely, handling edge cases
        if baseline_impact > 0:
            metrics['network_improvement'] = (gnn_impact - baseline_impact) / baseline_impact
        else:
            # If baseline is 0 or negative, use absolute difference
            metrics['network_improvement'] = gnn_impact - baseline_impact
        
        # 2. Multi-hop cascade value
        gnn_cascade = self._calculate_cascade_value(
            selected_nodes, gnn_outputs.get('hop_features', {}), edge_index
        )
        baseline_cascade = self._calculate_cascade_value(
            baseline_selection, gnn_outputs.get('hop_features', {}), edge_index
        )
        
        metrics['gnn_cascade_value'] = gnn_cascade
        metrics['baseline_cascade_value'] = baseline_cascade
        # Calculate cascade improvement safely
        if baseline_cascade > 0:
            metrics['cascade_improvement'] = (gnn_cascade - baseline_cascade) / baseline_cascade
        else:
            metrics['cascade_improvement'] = gnn_cascade - baseline_cascade
        
        # 3. Coverage (how many nodes benefit)
        gnn_coverage = self._calculate_coverage(selected_nodes, edge_index, max_hops=3)
        baseline_coverage = self._calculate_coverage(baseline_selection, edge_index, max_hops=3)
        
        metrics['gnn_coverage'] = gnn_coverage
        metrics['baseline_coverage'] = baseline_coverage
        
        # 4. Diversity of selection
        gnn_diversity = self._calculate_selection_diversity(
            selected_nodes, gnn_outputs.get('embeddings')
        )
        baseline_diversity = self._calculate_selection_diversity(
            baseline_selection, gnn_outputs.get('embeddings')
        )
        
        metrics['gnn_diversity'] = gnn_diversity
        metrics['baseline_diversity'] = baseline_diversity
        
        return metrics
    
    def _calculate_local_value(self, features: torch.Tensor) -> torch.Tensor:
        """Calculate value based on local features only"""
        # Simple scoring based on features
        # Assuming features include: energy_label, roof_area, etc.
        
        # Energy label score (assuming it's one-hot or ordinal)
        energy_score = features[:, 0]  # Adjust index as needed
        
        # Roof area score
        roof_score = features[:, 1] / features[:, 1].max() if features[:, 1].max() > 0 else features[:, 1]
        
        # Combine
        local_value = 0.5 * energy_score + 0.5 * roof_score
        
        return local_value
    
    def _calculate_network_value(self, gnn_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract network value from GNN predictions"""
        if 'intervention_values' in gnn_outputs:
            # Direct prediction from GNN
            return gnn_outputs['intervention_values'].squeeze()
        elif 'network_impacts' in gnn_outputs:
            # Sum multi-hop impacts
            impacts = gnn_outputs['network_impacts']
            return impacts.sum(dim=-1)
        else:
            # Fallback: use embedding magnitude
            embeddings = gnn_outputs.get('embeddings')
            if embeddings is not None:
                return torch.norm(embeddings, dim=-1)
            else:
                return torch.zeros(1)
    
    def _calculate_position_value(
        self,
        embeddings: Optional[torch.Tensor],
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate strategic position value in network"""
        if embeddings is None:
            return torch.zeros(1)
        
        N = embeddings.size(0)
        device = embeddings.device
        
        # Calculate degree centrality
        degrees = torch.bincount(edge_index[0], minlength=N).float()
        degree_score = degrees / degrees.max() if degrees.max() > 0 else degrees
        
        # Calculate betweenness proxy (nodes connecting different embeddings)
        row, col = edge_index
        edge_diversity = 1 - F.cosine_similarity(embeddings[row], embeddings[col])
        
        betweenness_score = torch.zeros(N, device=device)
        betweenness_score.index_add_(0, row, edge_diversity)
        betweenness_score = betweenness_score / betweenness_score.max() if betweenness_score.max() > 0 else betweenness_score
        
        # Combine centrality measures
        position_value = 0.5 * degree_score + 0.5 * betweenness_score
        
        return position_value
    
    def _calculate_cascade_potential(
        self,
        hop_features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate potential for cascade effects"""
        if not hop_features:
            return torch.zeros(1)
        
        cascade_scores = None
        
        # Sum variance across hops (high variance = high cascade potential)
        for hop_name, features in hop_features.items():
            if 'features' in hop_name:
                variance = torch.var(features, dim=-1)
                if cascade_scores is None:
                    cascade_scores = variance
                else:
                    cascade_scores += variance * 0.5  # Decay with hop distance
        
        if cascade_scores is not None:
            return cascade_scores / cascade_scores.max() if cascade_scores.max() > 0 else cascade_scores
        else:
            return torch.zeros(1)
    
    def _calculate_complementarity_bonus(
        self,
        embeddings: Optional[torch.Tensor],
        existing: List[int],
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate bonus for complementing existing interventions"""
        if embeddings is None or len(existing) == 0:
            return torch.zeros(1)
        
        N = embeddings.size(0)
        device = embeddings.device
        
        # Average embedding of existing interventions
        existing_embed = embeddings[existing].mean(dim=0, keepdim=True)
        
        # Diversity from existing (want different embeddings)
        diversity = 1 - F.cosine_similarity(embeddings, existing_embed.expand(N, -1))
        
        # Distance bonus (prefer nodes far from existing)
        distance_bonus = self._calculate_min_distance(existing, edge_index, N)
        
        # Combine
        comp_bonus = 0.7 * diversity + 0.3 * distance_bonus
        
        return comp_bonus
    
    def _apply_spacing_constraint(
        self,
        scores: torch.Tensor,
        existing: List[int],
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Apply minimum spacing constraint between interventions"""
        # First, completely exclude already selected nodes
        if existing:
            scores[existing] = float('-inf')  # Ensure they can't be selected again
        
        if self.min_spacing <= 0:
            return scores
        
        # Calculate distance from existing interventions
        distances = self._calculate_min_distance(existing, edge_index, scores.size(0))
        
        # Penalty for being too close
        penalty_mask = distances < self.min_spacing
        scores[penalty_mask] *= 0.1  # Severe penalty
        
        return scores
    
    def _apply_boundary_penalty(
        self,
        scores: torch.Tensor,
        boundary_violations: torch.Tensor
    ) -> torch.Tensor:
        """Penalize selections that violate transformer boundaries"""
        # boundary_violations should be [E] for edges
        # Need to aggregate to nodes
        
        # For now, simple penalty
        if boundary_violations.dim() == 1 and boundary_violations.size(0) == scores.size(0):
            scores = scores * (1 - self.boundary_penalty * boundary_violations)
        
        return scores
    
    def _adjust_for_interactions(
        self,
        scores: torch.Tensor,
        selected: List[int],
        edge_index: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Adjust scores considering interactions with already selected nodes"""
        if edge_index is None:
            return scores
        
        adjusted = scores.clone()
        
        for node in selected:
            # Find neighbors of selected node
            neighbors = edge_index[1][edge_index[0] == node]
            
            # Reduce scores of immediate neighbors (avoid clustering)
            adjusted[neighbors] *= 0.7
            
            # Boost scores of 2-hop neighbors (good spread)
            two_hop = []
            for n in neighbors:
                two_hop.extend(edge_index[1][edge_index[0] == n].tolist())
            two_hop = list(set(two_hop) - set(selected) - set(neighbors.tolist()))
            
            if two_hop:
                adjusted[two_hop] *= 1.2
        
        return adjusted
    
    def _update_scores_after_selection(
        self,
        scores: torch.Tensor,
        selected_idx: int,
        edge_index: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Update scores after a node is selected"""
        if edge_index is None:
            return scores
        
        updated = scores.clone()
        
        # Reduce scores of neighbors (diminishing returns)
        neighbors = edge_index[1][edge_index[0] == selected_idx]
        updated[neighbors] *= 0.8
        
        return updated
    
    def _calculate_min_distance(
        self,
        source_nodes: List[int],
        edge_index: torch.Tensor,
        num_nodes: int
    ) -> torch.Tensor:
        """Calculate minimum hop distance from source nodes to all nodes"""
        device = edge_index.device
        distances = torch.full((num_nodes,), float('inf'), device=device)
        distances[source_nodes] = 0
        
        # BFS to calculate distances
        current_level = source_nodes
        current_dist = 0
        visited = set(source_nodes)
        
        while current_level and current_dist < 10:  # Max 10 hops
            next_level = []
            for node in current_level:
                neighbors = edge_index[1][edge_index[0] == node].tolist()
                for n in neighbors:
                    if n not in visited:
                        visited.add(n)
                        next_level.append(n)
                        distances[n] = min(distances[n], current_dist + 1)
            
            current_level = next_level
            current_dist += 1
        
        # Normalize
        distances[distances == float('inf')] = 10
        distances = distances / 10
        
        return distances
    
    def _calculate_total_impact(
        self,
        nodes: List[int],
        gnn_outputs: Dict[str, torch.Tensor],
        edge_index: torch.Tensor
    ) -> float:
        """Calculate total network impact of selected nodes"""
        if 'network_impacts' in gnn_outputs:
            return gnn_outputs['network_impacts'][nodes].sum().item()
        else:
            # Fallback: count affected nodes
            coverage = self._calculate_coverage(nodes, edge_index, max_hops=3)
            return float(coverage)
    
    def _calculate_cascade_value(
        self,
        nodes: List[int],
        hop_features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor
    ) -> float:
        """Calculate total cascade value across hops"""
        if not hop_features:
            return 0.0
        
        total_value = 0.0
        hop_weights = [1.0, 0.5, 0.25]  # Decay with distance
        
        for hop_idx, (hop_name, features) in enumerate(hop_features.items()):
            if 'features' in hop_name and hop_idx < len(hop_weights):
                # Sum feature magnitudes for selected nodes
                node_values = torch.norm(features[nodes], dim=-1).sum()
                total_value += hop_weights[hop_idx] * node_values.item()
        
        return total_value
    
    def _calculate_coverage(
        self,
        nodes: List[int],
        edge_index: torch.Tensor,
        max_hops: int = 3
    ) -> int:
        """Calculate how many nodes are within max_hops of selected nodes"""
        covered = set(nodes)
        current_level = set(nodes)
        
        for _ in range(max_hops):
            next_level = set()
            for node in current_level:
                neighbors = edge_index[1][edge_index[0] == node].tolist()
                next_level.update(neighbors)
            
            covered.update(next_level)
            current_level = next_level - covered
        
        return len(covered)
    
    def _calculate_selection_diversity(
        self,
        nodes: List[int],
        embeddings: Optional[torch.Tensor]
    ) -> float:
        """Calculate diversity of selected nodes"""
        if embeddings is None or len(nodes) < 2:
            return 0.0
        
        selected_embeds = embeddings[nodes]
        
        # Pairwise distances
        distances = torch.cdist(selected_embeds, selected_embeds)
        
        # Average non-diagonal distances
        mask = 1 - torch.eye(len(nodes), device=distances.device)
        avg_distance = (distances * mask).sum() / mask.sum()
        
        return avg_distance.item()