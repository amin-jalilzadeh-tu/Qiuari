# tasks/energy_flow_tracker.py
"""
Detailed energy flow tracking between buildings
Tracks who shares with whom, how much, and when
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging

logger = logging.getLogger(__name__)

@dataclass
class EnergyFlow:
    """Single energy flow transaction"""
    timestamp: int
    from_building: int
    to_building: int
    energy_kwh: float
    flow_type: str  # 'direct', 'battery_mediated', 'grid_export'
    distance_m: float
    efficiency: float
    cluster_id: int
    lv_group_id: int

@dataclass 
class FlowSummary:
    """Summary statistics for a time period"""
    total_shared_kwh: float
    total_transactions: int
    avg_efficiency: float
    self_sufficiency_ratio: float
    peak_flow_kwh: float
    most_active_prosumer: int
    most_active_consumer: int
    cluster_internal_ratio: float
    lv_internal_ratio: float


class EnergyFlowTracker:
    """
    Tracks detailed energy flows between buildings with constraints
    """
    
    def __init__(self, config: Dict):
        """
        Initialize flow tracker
        
        Args:
            config: Configuration with flow parameters
        """
        self.distance_loss_per_100m = config.get('distance_loss_per_100m', 0.02)  # 2% per 100m
        self.battery_efficiency = config.get('battery_efficiency', 0.9)
        self.inverter_efficiency = config.get('inverter_efficiency', 0.95)
        self.max_flow_distance = config.get('max_flow_distance', 1000)  # meters
        
        # Storage for flows
        self.flow_history = []
        self.flow_matrices = {}  # timestamp -> flow matrix
        
        # Statistics
        self.building_stats = {}  # building_id -> stats
    
    def calculate_feasible_flows(self,
                                generation: torch.Tensor,
                                demand: torch.Tensor,
                                cluster_assignments: torch.Tensor,
                                lv_group_ids: torch.Tensor,
                                distance_matrix: Optional[torch.Tensor] = None,
                                timestamp: int = 0) -> torch.Tensor:
        """
        Calculate feasible energy flows for a single timestep
        
        Args:
            generation: Generation per building [N]
            demand: Demand per building [N]
            cluster_assignments: Cluster assignment per building [N]
            lv_group_ids: LV group per building [N]
            distance_matrix: Pairwise distances [N, N]
            timestamp: Current timestep
            
        Returns:
            flow_matrix: Energy flows [N, N] where [i,j] = flow from i to j
        """
        N = generation.size(0)
        device = generation.device
        
        # Calculate surplus and deficit
        surplus = torch.relu(generation - demand)  # Buildings with excess
        deficit = torch.relu(demand - generation)  # Buildings with need
        
        # Initialize flow matrix
        flow_matrix = torch.zeros(N, N, device=device)
        
        # Create masks for valid flows
        # 1. Same LV group constraint
        lv_mask = lv_group_ids.unsqueeze(0) == lv_group_ids.unsqueeze(1)  # [N, N]
        
        # 2. Same cluster preference (not hard constraint, but prioritized)
        cluster_mask = cluster_assignments.unsqueeze(0) == cluster_assignments.unsqueeze(1)
        
        # 3. Distance constraint (if provided)
        if distance_matrix is not None:
            distance_valid = distance_matrix <= self.max_flow_distance
        else:
            distance_valid = torch.ones(N, N, dtype=torch.bool, device=device)
        
        # Combined valid flow mask
        valid_flows = lv_mask & distance_valid
        
        # Calculate efficiency matrix based on distance
        if distance_matrix is not None:
            efficiency_matrix = 1.0 - (distance_matrix / 100.0 * self.distance_loss_per_100m)
            efficiency_matrix = torch.clamp(efficiency_matrix, min=0.0, max=1.0)
        else:
            efficiency_matrix = torch.ones(N, N, device=device) * 0.95
        
        # Apply inverter efficiency
        efficiency_matrix *= self.inverter_efficiency
        
        # Greedy matching algorithm with priorities
        remaining_surplus = surplus.clone()
        remaining_deficit = deficit.clone()
        
        # Priority 1: Same cluster, close distance
        priority_mask = valid_flows & cluster_mask
        if distance_matrix is not None:
            priority_mask &= (distance_matrix < 200)  # Within 200m
        
        flow_matrix = self._match_and_flow(
            remaining_surplus, remaining_deficit,
            priority_mask, efficiency_matrix, flow_matrix
        )
        
        # Priority 2: Same cluster, any distance
        secondary_mask = valid_flows & cluster_mask & ~priority_mask
        flow_matrix = self._match_and_flow(
            remaining_surplus, remaining_deficit,
            secondary_mask, efficiency_matrix, flow_matrix
        )
        
        # Priority 3: Same LV, different cluster
        tertiary_mask = valid_flows & ~cluster_mask
        flow_matrix = self._match_and_flow(
            remaining_surplus, remaining_deficit,
            tertiary_mask, efficiency_matrix, flow_matrix
        )
        
        # Store the flow matrix
        self.flow_matrices[timestamp] = flow_matrix.cpu().numpy()
        
        # Record individual flows
        self._record_flows(flow_matrix, cluster_assignments, lv_group_ids, 
                          distance_matrix, timestamp)
        
        return flow_matrix
    
    def _match_and_flow(self, surplus: torch.Tensor, deficit: torch.Tensor,
                       mask: torch.Tensor, efficiency: torch.Tensor,
                       flow_matrix: torch.Tensor) -> torch.Tensor:
        """
        Match surplus to deficit buildings and calculate flows
        """
        N = surplus.size(0)
        
        for i in range(N):
            if surplus[i] <= 0:
                continue
                
            # Find valid receivers
            valid_receivers = mask[i] & (deficit > 0)
            
            if not valid_receivers.any():
                continue
            
            # Sort by efficiency (prefer closer/more efficient)
            receiver_efficiency = efficiency[i] * valid_receivers.float()
            sorted_receivers = torch.argsort(receiver_efficiency, descending=True)
            
            for j in sorted_receivers:
                if receiver_efficiency[j] == 0:
                    break
                    
                # Calculate possible flow
                available = surplus[i]
                needed = deficit[j] / efficiency[i, j]  # Account for losses
                flow = min(available, needed)
                
                if flow > 0:
                    # Update flow matrix
                    flow_matrix[i, j] = flow
                    
                    # Update remaining amounts
                    surplus[i] -= flow
                    deficit[j] -= flow * efficiency[i, j]
                    
                    if surplus[i] <= 0:
                        break
        
        return flow_matrix
    
    def _record_flows(self, flow_matrix: torch.Tensor,
                     cluster_assignments: torch.Tensor,
                     lv_group_ids: torch.Tensor,
                     distance_matrix: Optional[torch.Tensor],
                     timestamp: int):
        """Record individual flows for detailed tracking"""
        N = flow_matrix.size(0)
        
        for i in range(N):
            for j in range(N):
                if flow_matrix[i, j] > 0:
                    distance = distance_matrix[i, j].item() if distance_matrix is not None else 100.0
                    efficiency = self.inverter_efficiency * (1.0 - distance / 100.0 * self.distance_loss_per_100m)
                    
                    flow = EnergyFlow(
                        timestamp=timestamp,
                        from_building=i,
                        to_building=j,
                        energy_kwh=flow_matrix[i, j].item(),
                        flow_type='direct',
                        distance_m=distance,
                        efficiency=efficiency,
                        cluster_id=cluster_assignments[i].item(),
                        lv_group_id=lv_group_ids[i].item()
                    )
                    
                    self.flow_history.append(flow)
                    
                    # Update building statistics
                    self._update_building_stats(i, j, flow.energy_kwh)
    
    def _update_building_stats(self, from_building: int, to_building: int, energy: float):
        """Update building-level statistics"""
        if from_building not in self.building_stats:
            self.building_stats[from_building] = {
                'energy_shared': 0, 'energy_received': 0,
                'transactions_out': 0, 'transactions_in': 0
            }
        if to_building not in self.building_stats:
            self.building_stats[to_building] = {
                'energy_shared': 0, 'energy_received': 0,
                'transactions_out': 0, 'transactions_in': 0
            }
        
        self.building_stats[from_building]['energy_shared'] += energy
        self.building_stats[from_building]['transactions_out'] += 1
        self.building_stats[to_building]['energy_received'] += energy
        self.building_stats[to_building]['transactions_in'] += 1
    
    def get_flow_summary(self, start_time: int = 0, end_time: Optional[int] = None) -> FlowSummary:
        """
        Get summary statistics for a time period
        
        Args:
            start_time: Start timestamp
            end_time: End timestamp (None = latest)
            
        Returns:
            FlowSummary with aggregated statistics
        """
        if end_time is None:
            end_time = max([f.timestamp for f in self.flow_history]) if self.flow_history else 0
        
        # Filter flows by time
        period_flows = [f for f in self.flow_history 
                       if start_time <= f.timestamp <= end_time]
        
        if not period_flows:
            return FlowSummary(0, 0, 0, 0, 0, -1, -1, 0, 1.0)
        
        # Calculate statistics
        total_shared = sum(f.energy_kwh for f in period_flows)
        total_transactions = len(period_flows)
        avg_efficiency = np.mean([f.efficiency for f in period_flows])
        peak_flow = max(f.energy_kwh for f in period_flows)
        
        # Find most active buildings
        prosumer_energy = {}
        consumer_energy = {}
        for f in period_flows:
            prosumer_energy[f.from_building] = prosumer_energy.get(f.from_building, 0) + f.energy_kwh
            consumer_energy[f.to_building] = consumer_energy.get(f.to_building, 0) + f.energy_kwh
        
        most_active_prosumer = max(prosumer_energy, key=prosumer_energy.get) if prosumer_energy else -1
        most_active_consumer = max(consumer_energy, key=consumer_energy.get) if consumer_energy else -1
        
        # Calculate internal ratios
        same_cluster_flows = sum(1 for f in period_flows 
                                if f.from_building != f.to_building and
                                f.cluster_id == f.cluster_id)
        cluster_ratio = same_cluster_flows / total_transactions if total_transactions > 0 else 0
        
        # All flows are within same LV by design
        lv_ratio = 1.0
        
        return FlowSummary(
            total_shared_kwh=total_shared,
            total_transactions=total_transactions,
            avg_efficiency=avg_efficiency,
            self_sufficiency_ratio=0.0,  # Calculate separately with demand data
            peak_flow_kwh=peak_flow,
            most_active_prosumer=most_active_prosumer,
            most_active_consumer=most_active_consumer,
            cluster_internal_ratio=cluster_ratio,
            lv_internal_ratio=lv_ratio
        )
    
    def get_flow_matrix_at_time(self, timestamp: int) -> Optional[np.ndarray]:
        """Get flow matrix for specific timestamp"""
        return self.flow_matrices.get(timestamp)
    
    def save_flow_history(self, filepath: str):
        """Save flow history to file"""
        flow_dicts = [asdict(f) for f in self.flow_history]
        df = pd.DataFrame(flow_dicts)
        
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(flow_dicts, f, indent=2)
        else:
            df.to_parquet(filepath)
        
        logger.info(f"Saved {len(self.flow_history)} flows to {filepath}")
    
    def generate_flow_report(self) -> str:
        """Generate human-readable flow report"""
        summary = self.get_flow_summary()
        
        report = []
        report.append("=" * 60)
        report.append("ENERGY FLOW TRACKING REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append(f"Total Energy Shared: {summary.total_shared_kwh:.2f} kWh")
        report.append(f"Total Transactions: {summary.total_transactions}")
        report.append(f"Average Efficiency: {summary.avg_efficiency:.1%}")
        report.append(f"Peak Single Flow: {summary.peak_flow_kwh:.2f} kWh")
        report.append("")
        
        report.append("FLOW PATTERNS:")
        report.append(f"  Cluster-Internal Flows: {summary.cluster_internal_ratio:.1%}")
        report.append(f"  LV-Internal Flows: {summary.lv_internal_ratio:.1%}")
        report.append("")
        
        report.append("TOP BUILDINGS:")
        report.append(f"  Most Active Prosumer: Building {summary.most_active_prosumer}")
        report.append(f"  Most Active Consumer: Building {summary.most_active_consumer}")
        report.append("")
        
        # Building statistics
        if self.building_stats:
            report.append("BUILDING STATISTICS:")
            sorted_stats = sorted(self.building_stats.items(), 
                                key=lambda x: x[1]['energy_shared'], 
                                reverse=True)[:5]
            for building_id, stats in sorted_stats:
                report.append(f"  Building {building_id}:")
                report.append(f"    Shared: {stats['energy_shared']:.1f} kWh")
                report.append(f"    Received: {stats['energy_received']:.1f} kWh")
        
        return "\n".join(report)