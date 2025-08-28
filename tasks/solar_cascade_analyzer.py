"""
Solar Cascade Effects Analyzer
Implements multi-hop impact propagation for solar installations
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SolarCascadeAnalyzer:
    """
    Analyzes cascade effects of solar installations across the network
    """
    
    def __init__(self, config: Dict):
        """
        Initialize cascade analyzer
        
        Args:
            config: Configuration with cascade parameters
        """
        self.max_hops = config.get('max_hops', 3)
        self.impact_decay = config.get('impact_decay', 0.7)  # Decay per hop
        self.min_impact = config.get('min_impact', 0.1)  # Minimum impact threshold
        
        logger.info(f"Initialized SolarCascadeAnalyzer with {self.max_hops} hops")
    
    def analyze_cascade(
        self,
        building_id: int,
        capacity_kwp: float,
        graph_data: Dict,
        edge_index: torch.Tensor,
        building_features: torch.Tensor
    ) -> Dict:
        """
        Analyze cascade effects of solar installation
        
        Args:
            building_id: Building to install solar on
            capacity_kwp: Solar capacity in kWp
            graph_data: Graph structure
            edge_index: Building connections
            building_features: Building feature matrix
            
        Returns:
            Cascade analysis results
        """
        device = building_features.device
        num_buildings = building_features.shape[0]
        
        # Initialize impact scores
        impact_scores = torch.zeros(num_buildings, device=device)
        impact_scores[building_id] = 1.0  # Direct impact
        
        # Track affected buildings by hop distance
        hops_tracker = {0: [building_id]}
        
        # Propagate impact through network
        current_impact = impact_scores.clone()
        
        for hop in range(1, self.max_hops + 1):
            # Create adjacency matrix for propagation
            adj_matrix = self._create_adjacency(edge_index, num_buildings, device)
            
            # Propagate impact to neighbors
            next_impact = torch.matmul(adj_matrix, current_impact.unsqueeze(-1)).squeeze()
            
            # Apply decay
            next_impact = next_impact * self.impact_decay
            
            # Track buildings affected at this hop
            newly_affected = ((next_impact > self.min_impact) & (impact_scores == 0)).nonzero(as_tuple=True)[0]
            if len(newly_affected) > 0:
                hops_tracker[hop] = newly_affected.tolist()
            
            # Accumulate impact
            impact_scores = torch.maximum(impact_scores, next_impact)
            current_impact = next_impact
        
        # Calculate network-wide benefits
        benefits = self._calculate_benefits(
            building_id,
            capacity_kwp,
            impact_scores,
            hops_tracker,
            building_features
        )
        
        return {
            'impact_scores': impact_scores,
            'hops_tracker': hops_tracker,
            'total_affected': (impact_scores > self.min_impact).sum().item(),
            'network_benefits': benefits,
            'cascade_radius': max(hops_tracker.keys()) if hops_tracker else 0
        }
    
    def _create_adjacency(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create normalized adjacency matrix from edge index
        
        Args:
            edge_index: Edge connections
            num_nodes: Number of nodes
            device: Computation device
            
        Returns:
            Normalized adjacency matrix
        """
        adj = torch.zeros(num_nodes, num_nodes, device=device)
        
        if edge_index.shape[1] > 0:
            adj[edge_index[0], edge_index[1]] = 1.0
            
            # Normalize by degree
            degree = adj.sum(dim=1, keepdim=True)
            degree[degree == 0] = 1.0
            adj = adj / degree
        
        return adj
    
    def _calculate_benefits(
        self,
        source_building: int,
        capacity_kwp: float,
        impact_scores: torch.Tensor,
        hops_tracker: Dict,
        features: torch.Tensor
    ) -> Dict:
        """
        Calculate network-wide benefits from cascade
        
        Args:
            source_building: Building with solar
            capacity_kwp: Solar capacity
            impact_scores: Impact on each building
            hops_tracker: Buildings by hop distance
            features: Building features
            
        Returns:
            Network benefits dictionary
        """
        # Peak reduction at transformer
        peak_reduction_kw = capacity_kwp * 0.6  # 60% peak coincidence
        
        # Voltage improvement
        voltage_improvement = 0.0
        for hop, buildings in hops_tracker.items():
            if hop == 0:
                voltage_improvement += 0.02  # 2% at source
            elif hop == 1:
                voltage_improvement += 0.01 * len(buildings) / 100  # 1% per 100 buildings
            elif hop == 2:
                voltage_improvement += 0.005 * len(buildings) / 100  # 0.5% per 100 buildings
        
        # Energy loss reduction
        # Solar reduces line losses by reducing power flow
        affected_buildings = impact_scores > self.min_impact
        avg_distance = features[affected_buildings, 6].mean() if affected_buildings.any() else 0
        loss_reduction_kwh = capacity_kwp * 100 * (1 - avg_distance / 1000)  # Rough estimate
        
        # Grid congestion relief
        congestion_relief = peak_reduction_kw / 1000  # MW relief
        
        # Economic benefits
        total_impact = impact_scores.sum().item()
        network_value = total_impact * capacity_kwp * 100  # €100 per kWp per unit impact
        
        return {
            'peak_reduction_kw': peak_reduction_kw,
            'voltage_improvement_percent': min(5.0, voltage_improvement * 100),
            'loss_reduction_kwh_annual': max(0, loss_reduction_kwh),
            'congestion_relief_mw': congestion_relief,
            'network_value_euro': network_value,
            'benefited_buildings': {
                f"hop_{k}": len(v) for k, v in hops_tracker.items()
            }
        }
    
    def rank_buildings_by_cascade_potential(
        self,
        building_features: torch.Tensor,
        edge_index: torch.Tensor,
        existing_solar: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Rank buildings by their cascade effect potential
        
        Args:
            building_features: Building feature matrix
            edge_index: Building connections
            existing_solar: Binary mask of buildings with existing solar
            
        Returns:
            Ranking scores for each building
        """
        num_buildings = building_features.shape[0]
        device = building_features.device
        
        # Calculate network centrality (simple degree centrality)
        adj = self._create_adjacency(edge_index, num_buildings, device)
        degree_centrality = adj.sum(dim=1)
        
        # Calculate distance from transformer (using position features)
        # Assuming features[6] and features[7] are x, y coordinates
        positions = building_features[:, 6:8]
        transformer_pos = positions.mean(dim=0)  # Approximate transformer at center
        distance_from_transformer = torch.norm(positions - transformer_pos, dim=1)
        
        # Normalize distance (closer is better for cascade)
        distance_score = 1.0 - (distance_from_transformer / distance_from_transformer.max())
        
        # Consider existing solar (avoid clustering)
        if existing_solar is not None:
            # Reduce score near existing solar
            solar_adj = adj @ existing_solar.float()
            solar_penalty = solar_adj * 0.5  # 50% penalty if neighbor has solar
        else:
            solar_penalty = 0
        
        # Combine factors
        cascade_score = (
            0.4 * degree_centrality +  # Network connectivity
            0.3 * distance_score +      # Proximity to transformer
            0.3 * building_features[:, 9] / 100  # Suitable roof area (normalized)
        ) - solar_penalty
        
        return cascade_score
    
    def simulate_cumulative_cascade(
        self,
        selected_buildings: List[int],
        capacities: List[float],
        building_features: torch.Tensor,
        edge_index: torch.Tensor
    ) -> Dict:
        """
        Simulate cumulative cascade effects of multiple installations
        
        Args:
            selected_buildings: Buildings to install solar on
            capacities: Solar capacities for each building
            building_features: Building feature matrix
            edge_index: Building connections
            
        Returns:
            Cumulative cascade analysis
        """
        num_buildings = building_features.shape[0]
        device = building_features.device
        
        # Accumulate impacts
        cumulative_impact = torch.zeros(num_buildings, device=device)
        individual_cascades = []
        
        for building_id, capacity in zip(selected_buildings, capacities):
            cascade = self.analyze_cascade(
                building_id,
                capacity,
                {},  # graph_data not needed here
                edge_index,
                building_features
            )
            
            individual_cascades.append(cascade)
            cumulative_impact += cascade['impact_scores']
        
        # Calculate synergy effects
        synergy = cumulative_impact.max() / len(selected_buildings) if selected_buildings else 0
        
        # Total network benefits
        total_peak_reduction = sum(c['network_benefits']['peak_reduction_kw'] 
                                  for c in individual_cascades)
        total_value = sum(c['network_benefits']['network_value_euro'] 
                         for c in individual_cascades)
        
        # Check for overconcentration
        concentration_score = cumulative_impact.std() / cumulative_impact.mean() if cumulative_impact.mean() > 0 else 0
        well_distributed = concentration_score < 1.0  # Lower is better distributed
        
        return {
            'cumulative_impact': cumulative_impact,
            'synergy_factor': synergy,
            'total_peak_reduction_kw': total_peak_reduction,
            'total_network_value': total_value,
            'well_distributed': well_distributed,
            'concentration_score': concentration_score,
            'individual_cascades': individual_cascades
        }