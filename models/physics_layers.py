# models/physics_layers.py
"""
Physics constraint layers for energy system
Enforces energy balance, LV boundaries, and distance-based losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import logging
from utils.constants import CONFIDENCE_THRESHOLD

logger = logging.getLogger(__name__)


class LVGroupBoundaryEnforcer(nn.Module):
    """Ensures energy sharing only within same LV group"""
    
    def __init__(self):
        super().__init__()
        self.violation_penalty_weight = nn.Parameter(torch.tensor(10.0))
        
    def forward(self, 
                sharing_matrix: torch.Tensor,
                lv_group_ids: torch.Tensor,
                valid_lv_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply LV group constraints to sharing matrix
        
        Args:
            sharing_matrix: [batch, N, N] or [batch, N, N, T] proposed sharing
            lv_group_ids: [batch, N] or [N] LV group assignment for each building
            valid_lv_mask: [batch, N] or [N] mask for buildings in valid LV groups
            
        Returns:
            masked_sharing: Sharing matrix with invalid connections zeroed
            boundary_penalty: Penalty for attempted cross-boundary sharing
        """
        # Handle different input dimensions
        if lv_group_ids.dim() == 1:
            lv_group_ids = lv_group_ids.unsqueeze(0)
        
        batch_size, num_buildings = lv_group_ids.shape
        device = sharing_matrix.device
        
        # Create mask for same LV group
        lv_i = lv_group_ids.unsqueeze(2)  # [B, N, 1]
        lv_j = lv_group_ids.unsqueeze(1)  # [B, 1, N]
        same_lv_mask = (lv_i == lv_j).float()  # [B, N, N]
        
        # Apply valid LV mask if provided (skip orphaned groups)
        if valid_lv_mask is not None:
            if valid_lv_mask.dim() == 1:
                valid_lv_mask = valid_lv_mask.unsqueeze(0)
            valid_i = valid_lv_mask.unsqueeze(2)  # [B, N, 1]
            valid_j = valid_lv_mask.unsqueeze(1)  # [B, 1, N]
            valid_pair_mask = valid_i * valid_j  # Both buildings must be valid
            same_lv_mask = same_lv_mask * valid_pair_mask
        
        # Calculate penalty for violations (before masking)
        if sharing_matrix.dim() == 4:  # Has time dimension
            same_lv_mask = same_lv_mask.unsqueeze(-1)  # [B, N, N, 1]
            violations = sharing_matrix * (1 - same_lv_mask)
        else:
            violations = sharing_matrix * (1 - same_lv_mask)
        
        # Soft penalty (squared violations)
        boundary_penalty = (violations ** 2).sum() / (num_buildings ** 2)
        boundary_penalty = boundary_penalty * self.violation_penalty_weight
        
        # Apply mask to zero out invalid connections
        masked_sharing = sharing_matrix * same_lv_mask
        
        return masked_sharing, boundary_penalty


class DistanceBasedLossCalculator(nn.Module):
    """Calculates energy losses based on distance between buildings"""
    
    def __init__(self, 
                 base_efficiency: float = 0.98,
                 loss_per_meter: float = 0.0001):
        super().__init__()
        self.base_efficiency = base_efficiency
        self.loss_per_meter = loss_per_meter
        self.max_distance_penalty = nn.Parameter(torch.tensor(1000.0))
        
    def forward(self,
                sharing_matrix: torch.Tensor,
                positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply distance-based losses to sharing
        
        Args:
            sharing_matrix: [batch, N, N] or [batch, N, N, T] energy sharing
            positions: [batch, N, 2] or [N, 2] building x,y coordinates
            
        Returns:
            loss_adjusted_sharing: Sharing with efficiency losses applied
            distance_loss: Total loss due to distance
        """
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        
        batch_size, num_buildings, _ = positions.shape
        device = positions.device
        
        # Calculate pairwise distances
        pos_i = positions.unsqueeze(2)  # [B, N, 1, 2]
        pos_j = positions.unsqueeze(1)  # [B, 1, N, 2]
        distances = torch.norm(pos_i - pos_j, dim=-1)  # [B, N, N]
        
        # Calculate efficiency based on distance
        # Efficiency decreases with distance
        efficiency = torch.clamp(
            self.base_efficiency - self.loss_per_meter * distances,
            min=CONFIDENCE_THRESHOLD,  # Minimum 85% efficiency
            max=1.0    # Maximum 100% efficiency
        )
        
        # Apply efficiency to sharing
        if sharing_matrix.dim() == 4:  # Has time dimension
            efficiency = efficiency.unsqueeze(-1)  # [B, N, N, 1]
        
        loss_adjusted_sharing = sharing_matrix * efficiency
        
        # Calculate total energy lost
        energy_lost = sharing_matrix - loss_adjusted_sharing
        distance_loss = energy_lost.abs().sum() / (num_buildings ** 2)
        
        # Add penalty for very long distance sharing
        long_distance_mask = (distances > self.max_distance_penalty).float()
        if sharing_matrix.dim() == 4:
            long_distance_mask = long_distance_mask.unsqueeze(-1)
        long_distance_penalty = (sharing_matrix * long_distance_mask).abs().sum()
        
        total_loss = distance_loss + 0.1 * long_distance_penalty
        
        return loss_adjusted_sharing, total_loss


class EnergyBalanceChecker(nn.Module):
    """Ensures energy conservation within each LV group"""
    
    def __init__(self, tolerance: float = 0.05):
        super().__init__()
        self.tolerance = tolerance
        self.imbalance_penalty_weight = nn.Parameter(torch.tensor(5.0))
        
    def forward(self,
                consumption: torch.Tensor,
                generation: torch.Tensor,
                sharing_matrix: torch.Tensor,
                lv_group_ids: torch.Tensor,
                valid_lv_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Check energy balance per LV group
        
        Args:
            consumption: [batch, N] or [batch, N, T] consumption per building
            generation: [batch, N] or [batch, N, T] generation per building
            sharing_matrix: [batch, N, N] or [batch, N, N, T] energy flows
            lv_group_ids: [batch, N] or [N] LV group assignments
            valid_lv_mask: [batch, N] or [N] mask for valid buildings
            
        Returns:
            balance_penalty: Penalty for energy imbalance
            balance_info: Dictionary with balance details per LV group
        """
        if lv_group_ids.dim() == 1:
            lv_group_ids = lv_group_ids.unsqueeze(0)
        
        batch_size = consumption.shape[0]
        device = consumption.device
        
        # Get unique LV groups
        unique_lv_groups = torch.unique(lv_group_ids)
        
        total_imbalance = torch.tensor(0.0, device=device)
        balance_info = {}
        
        for lv_group in unique_lv_groups:
            # Skip invalid groups (e.g., -1 for orphaned)
            if lv_group < 0:
                continue
                
            # Get buildings in this LV group
            group_mask = (lv_group_ids == lv_group).float()
            
            # Apply valid mask if provided
            if valid_lv_mask is not None:
                if valid_lv_mask.dim() == 1:
                    valid_lv_mask = valid_lv_mask.unsqueeze(0)
                group_mask = group_mask * valid_lv_mask
            
            if group_mask.sum() == 0:
                continue
            
            # Calculate group consumption and generation
            if consumption.dim() == 3:  # Has time dimension
                group_mask_t = group_mask.unsqueeze(-1)
                group_consumption = (consumption * group_mask_t).sum(dim=1)
                group_generation = (generation * group_mask_t).sum(dim=1)
            else:
                group_consumption = (consumption * group_mask).sum(dim=1)
                group_generation = (generation * group_mask).sum(dim=1)
            
            # Calculate net sharing for the group
            # Positive sharing = export, negative = import
            if sharing_matrix.dim() == 4:  # Has time dimension
                group_mask_expanded = group_mask.unsqueeze(2).unsqueeze(-1)
                # Net export from group = sum of exports - sum of imports
                exports = (sharing_matrix * group_mask_expanded).sum(dim=1)
                imports = (sharing_matrix * group_mask_expanded.transpose(1, 2)).sum(dim=1)
                net_sharing = exports.sum(dim=1) - imports.sum(dim=1)
            else:
                group_mask_expanded = group_mask.unsqueeze(2)
                exports = (sharing_matrix * group_mask_expanded).sum(dim=1)
                imports = (sharing_matrix * group_mask_expanded.transpose(1, 2)).sum(dim=1)
                net_sharing = exports.sum(dim=1) - imports.sum(dim=1)
            
            # Energy balance: consumption = generation + import - export
            # Or: consumption - generation - net_import = 0
            net_import_needed = group_consumption - group_generation
            imbalance = (net_import_needed + net_sharing).abs()
            
            # Relative imbalance
            total_energy = group_consumption + group_generation + 1e-6
            relative_imbalance = imbalance / total_energy
            
            # Penalty for imbalance beyond tolerance
            penalty = F.relu(relative_imbalance - self.tolerance)
            total_imbalance = total_imbalance + penalty.sum()
            
            # Store info
            balance_info[f'lv_group_{lv_group.item()}'] = {
                'consumption': group_consumption.mean().item(),
                'generation': group_generation.mean().item(),
                'imbalance': imbalance.mean().item(),
                'relative_imbalance': relative_imbalance.mean().item()
            }
        
        balance_penalty = total_imbalance * self.imbalance_penalty_weight / len(unique_lv_groups)
        
        return balance_penalty, balance_info


class TemporalConsistencyValidator(nn.Module):
    """Ensures temporal feasibility of energy flows"""
    
    def __init__(self):
        super().__init__()
        self.temporal_penalty_weight = nn.Parameter(torch.tensor(3.0))
        
    def forward(self,
                energy_states: torch.Tensor,
                battery_states: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Check temporal consistency of energy flows
        
        Args:
            energy_states: [batch, N, T, features] temporal energy states
            battery_states: [batch, N, T] battery state of charge (optional)
            
        Returns:
            temporal_penalty: Penalty for temporal violations
        """
        batch_size, num_buildings, time_steps, _ = energy_states.shape
        device = energy_states.device
        
        total_penalty = torch.tensor(0.0, device=device)
        
        # Check ramp rate constraints (can't change too quickly)
        if time_steps > 1:
            # Calculate change between consecutive time steps
            energy_diff = energy_states[:, :, 1:, 0] - energy_states[:, :, :-1, 0]
            
            # Penalize very large changes (more than 50% change)
            max_change = 0.5 * (energy_states[:, :, 1:, 0].abs() + energy_states[:, :, :-1, 0].abs()) / 2
            ramp_violations = F.relu(energy_diff.abs() - max_change)
            total_penalty = total_penalty + ramp_violations.mean()
        
        # Check battery consistency if provided
        if battery_states is not None and battery_states.shape[-1] > 1:
            # Battery discharge can't exceed stored energy
            discharge = F.relu(-torch.diff(battery_states, dim=-1))  # Negative diff = discharge
            stored = battery_states[:, :, :-1]
            
            # Penalty for discharging more than stored
            battery_violations = F.relu(discharge - stored)
            total_penalty = total_penalty + battery_violations.mean()
            
            # Battery can't charge beyond capacity (assume normalized to 1.0)
            overcharge = F.relu(battery_states - 1.0)
            total_penalty = total_penalty + overcharge.mean()
        
        return total_penalty * self.temporal_penalty_weight


class ViolationPenaltyAggregator(nn.Module):
    """Aggregates all physics constraint violations into training penalty"""
    
    def __init__(self, 
                 boundary_weight: float = 10.0,
                 balance_weight: float = 5.0,
                 distance_weight: float = 1.0,
                 temporal_weight: float = 3.0):
        super().__init__()
        
        self.weights = nn.ParameterDict({
            'boundary': nn.Parameter(torch.tensor(boundary_weight)),
            'balance': nn.Parameter(torch.tensor(balance_weight)),
            'distance': nn.Parameter(torch.tensor(distance_weight)),
            'temporal': nn.Parameter(torch.tensor(temporal_weight))
        })
        
        # Learnable temperature for soft penalties
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, penalties: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict]:
        """
        Aggregate multiple penalties into single loss
        
        Args:
            penalties: Dictionary of individual penalties
            
        Returns:
            total_penalty: Weighted sum of all penalties
            penalty_info: Dictionary with weighted penalties
        """
        total_penalty = torch.tensor(0.0, device=next(iter(penalties.values())).device)
        penalty_info = {}
        
        for name, penalty in penalties.items():
            if name in self.weights:
                weighted_penalty = self.weights[name] * penalty / self.temperature
                total_penalty = total_penalty + weighted_penalty
                penalty_info[f'{name}_weighted'] = weighted_penalty.item()
                penalty_info[f'{name}_raw'] = penalty.item()
        
        penalty_info['total'] = total_penalty.item()
        
        return total_penalty, penalty_info


class PhysicsConstraintLayer(nn.Module):
    """Main physics constraint layer combining all components"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Components
        self.boundary_enforcer = LVGroupBoundaryEnforcer()
        self.distance_calculator = DistanceBasedLossCalculator()
        self.balance_checker = EnergyBalanceChecker()
        self.temporal_validator = TemporalConsistencyValidator()
        self.penalty_aggregator = ViolationPenaltyAggregator()
        
        # Configuration
        self.enforce_hard_boundaries = config.get('enforce_hard_boundaries', True)
        self.check_balance = config.get('check_balance', True)
        self.apply_losses = config.get('apply_losses', True)
        self.validate_temporal = config.get('validate_temporal', True)
        
        logger.info("Initialized PhysicsConstraintLayer")
    
    def forward(self,
                embeddings_dict: Dict,
                sharing_proposals: torch.Tensor,
                consumption_data: torch.Tensor,
                generation_data: torch.Tensor,
                metadata: Dict) -> Dict:
        """
        Apply all physics constraints
        
        Args:
            embeddings_dict: Embeddings from temporal processor
            sharing_proposals: [batch, N, N] or [batch, N, N, T] proposed sharing
            consumption_data: [batch, N] or [batch, N, T] consumption
            generation_data: [batch, N] or [batch, N, T] generation
            metadata: Dictionary containing:
                - lv_group_ids: LV group assignments
                - valid_lv_mask: Mask for valid buildings
                - positions: Building x,y coordinates
                - temporal_states: Optional temporal energy states
                
        Returns:
            Dictionary containing:
                - feasible_sharing: Physically feasible sharing matrix
                - feasible_embeddings: Adjusted embeddings
                - total_penalty: Sum of all constraint violations
                - penalty_breakdown: Individual penalties
                - balance_info: Energy balance details
        """
        device = sharing_proposals.device
        penalties = {}
        
        # Extract metadata
        lv_group_ids = metadata['lv_group_ids']
        valid_lv_mask = metadata.get('valid_lv_mask', None)
        positions = metadata['positions']
        temporal_states = metadata.get('temporal_states', None)
        
        # Start with proposed sharing
        feasible_sharing = sharing_proposals
        
        # 1. Apply LV group boundaries
        if self.enforce_hard_boundaries:
            feasible_sharing, boundary_penalty = self.boundary_enforcer(
                feasible_sharing, lv_group_ids, valid_lv_mask
            )
            penalties['boundary'] = boundary_penalty
        
        # 2. Apply distance-based losses
        if self.apply_losses and positions is not None:
            feasible_sharing, distance_loss = self.distance_calculator(
                feasible_sharing, positions
            )
            penalties['distance'] = distance_loss
        
        # 3. Check energy balance
        if self.check_balance:
            balance_penalty, balance_info = self.balance_checker(
                consumption_data, generation_data, feasible_sharing,
                lv_group_ids, valid_lv_mask
            )
            penalties['balance'] = balance_penalty
        else:
            balance_info = {}
        
        # 4. Validate temporal consistency
        if self.validate_temporal and temporal_states is not None:
            temporal_penalty = self.temporal_validator(temporal_states)
            penalties['temporal'] = temporal_penalty
        
        # 5. Aggregate penalties
        total_penalty, penalty_info = self.penalty_aggregator(penalties)
        
        # 6. Adjust embeddings based on feasibility
        # Reduce embedding magnitude for high-violation buildings
        building_embeddings = embeddings_dict.get('building')
        if building_embeddings is not None:
            # Calculate violation score per building
            violation_score = torch.zeros_like(building_embeddings[:, :, 0])
            
            if 'boundary' in penalties:
                # Buildings trying to share across boundaries
                cross_boundary = (sharing_proposals != feasible_sharing).float()
                violation_score += cross_boundary.sum(dim=-1).mean(dim=-1) if cross_boundary.dim() > 2 else cross_boundary.sum(dim=-1)
            
            # Apply soft suppression to embeddings
            suppression = torch.exp(-violation_score.unsqueeze(-1))
            feasible_embeddings = embeddings_dict.copy()
            feasible_embeddings['building'] = building_embeddings * suppression
        else:
            feasible_embeddings = embeddings_dict
        
        return {
            'feasible_sharing': feasible_sharing,
            'feasible_embeddings': feasible_embeddings,
            'total_penalty': total_penalty,
            'penalty_breakdown': penalty_info,
            'balance_info': balance_info,
            'violation_scores': violation_score if 'violation_score' in locals() else None
        }


def create_physics_constraint_layer(config: Dict) -> PhysicsConstraintLayer:
    """Factory function to create physics constraint layer"""
    return PhysicsConstraintLayer(config)


# Test function
def test_physics_layer():
    """Test physics constraint layer with dummy data"""
    
    print("\n" + "="*60)
    print("TESTING PHYSICS CONSTRAINT LAYER")
    print("="*60 + "\n")
    
    # Configuration
    config = {
        'enforce_hard_boundaries': True,
        'check_balance': True,
        'apply_losses': True,
        'validate_temporal': True
    }
    
    # Create dummy data
    batch_size = 1
    num_buildings = 100
    time_steps = 24
    
    # Embeddings (from temporal processor)
    embeddings_dict = {
        'building': torch.randn(batch_size, num_buildings, 128),
        'cable_group': torch.randn(batch_size, 20, 128)
    }
    
    # Proposed sharing matrix
    sharing_proposals = torch.rand(batch_size, num_buildings, num_buildings) * 10
    sharing_proposals = (sharing_proposals + sharing_proposals.transpose(1, 2)) / 2  # Symmetric
    
    # Consumption and generation
    consumption = torch.rand(batch_size, num_buildings) * 20 + 5
    generation = torch.rand(batch_size, num_buildings) * 5
    
    # Metadata
    lv_group_ids = torch.randint(0, 10, (num_buildings,))
    valid_lv_mask = torch.ones(num_buildings)
    valid_lv_mask[80:] = 0  # Last 20 buildings are invalid (orphaned)
    
    positions = torch.randn(num_buildings, 2) * 100  # Random positions
    temporal_states = torch.randn(batch_size, num_buildings, time_steps, 4)
    
    metadata = {
        'lv_group_ids': lv_group_ids,
        'valid_lv_mask': valid_lv_mask,
        'positions': positions,
        'temporal_states': temporal_states
    }
    
    # Create and test layer
    physics_layer = create_physics_constraint_layer(config)
    physics_layer.eval()
    
    with torch.no_grad():
        output = physics_layer(
            embeddings_dict,
            sharing_proposals,
            consumption,
            generation,
            metadata
        )
    
    print("Output keys:", output.keys())
    print(f"Feasible sharing shape: {output['feasible_sharing'].shape}")
    print(f"Total penalty: {output['total_penalty'].item():.4f}")
    print("\nPenalty breakdown:")
    for key, value in output['penalty_breakdown'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\n✅ Physics constraint layer test successful!")
    
    return output


if __name__ == "__main__":
    test_physics_layer()