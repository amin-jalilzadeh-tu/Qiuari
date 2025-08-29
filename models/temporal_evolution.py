"""
Temporal Evolution Module
Predicts how clusters and energy patterns evolve as solar penetration increases
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TemporalEvolutionPredictor(nn.Module):
    """
    Predicts cluster evolution and energy flow changes over time
    as solar installations are deployed
    """
    
    def __init__(self, config: Dict):
        """
        Initialize temporal evolution predictor
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.2)
        
        # Feature dimensions
        self.building_features = config.get('building_features', 14)
        self.solar_features = 4  # capacity, generation, self-consumption, export
        
        # Temporal encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 2)
        )
        
        # Solar impact encoder
        self.solar_encoder = nn.Sequential(
            nn.Linear(self.solar_features, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # Cluster evolution predictor - dynamic input size
        # Will be initialized on first forward pass
        self.cluster_predictor = None
        self._gru_input_size = None
        
        # Output heads
        self.cluster_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 30)  # Max 30 clusters
        )
        
        self.energy_flow_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 3)  # import, export, self-consumption
        )
        
        self.stability_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 1),
            nn.Sigmoid()  # 0-1 stability score
        )
        
        logger.info("Initialized TemporalEvolutionPredictor")
    
    def forward(
        self,
        building_features: torch.Tensor,
        solar_installations: torch.Tensor,
        time_step: float,
        current_clusters: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict evolution of clusters and energy flows
        
        Args:
            building_features: Building features [N, F]
            solar_installations: Solar installation details [N, 4]
            time_step: Time step (0-1 normalized over planning horizon)
            current_clusters: Current cluster assignments [N]
            edge_index: Building connections
            
        Returns:
            Dictionary with predictions
        """
        batch_size = building_features.shape[0]
        device = building_features.device
        
        # Encode time
        time_input = torch.tensor([[time_step]], dtype=torch.float32, device=device).expand(batch_size, 1)
        time_features = self.time_encoder(time_input)
        
        # Encode solar impact
        solar_features = self.solar_encoder(solar_installations)
        
        # Combine features
        combined = torch.cat([
            building_features,
            solar_features,
            time_features.expand(batch_size, -1)
        ], dim=-1)
        
        # Initialize GRU dynamically if needed
        if self.cluster_predictor is None or self._gru_input_size != combined.shape[-1]:
            self._gru_input_size = combined.shape[-1]
            self.cluster_predictor = nn.GRU(
                input_size=self._gru_input_size,
                hidden_size=self.hidden_dim,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout if self.num_layers > 1 else 0
            ).to(device)
        
        # Process through GRU
        # Add temporal dimension
        combined = combined.unsqueeze(1)  # [N, 1, F]
        
        gru_out, hidden = self.cluster_predictor(combined)
        gru_out = gru_out.squeeze(1)  # [N, H]
        
        # Generate predictions
        cluster_logits = self.cluster_head(gru_out)
        energy_flows = self.energy_flow_head(gru_out)
        stability = self.stability_head(gru_out)
        
        # Apply graph constraints if available
        if edge_index is not None and current_clusters is not None:
            cluster_logits = self._apply_graph_constraints(
                cluster_logits,
                current_clusters,
                edge_index
            )
        
        return {
            'cluster_logits': cluster_logits,
            'cluster_assignments': cluster_logits.argmax(dim=-1),
            'energy_flows': energy_flows,
            'stability_scores': stability,
            'hidden_state': hidden
        }
    
    def predict_trajectory(
        self,
        building_features: torch.Tensor,
        solar_roadmap: List[Dict],
        initial_clusters: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None
    ) -> List[Dict]:
        """
        Predict complete evolution trajectory over planning horizon
        
        Args:
            building_features: Building features
            solar_roadmap: Planned solar installations per year
            initial_clusters: Starting cluster assignments
            edge_index: Building connections
            
        Returns:
            List of predictions per time step
        """
        trajectory = []
        current_clusters = initial_clusters.clone()
        current_solar = torch.zeros(
            building_features.shape[0], 
            self.solar_features,
            device=building_features.device
        )
        
        for year_idx, year_plan in enumerate(solar_roadmap):
            # Normalize time
            time_step = (year_idx + 1) / len(solar_roadmap)
            
            # Update solar installations
            for building_id, capacity in year_plan.items():
                current_solar[building_id, 0] = capacity  # Capacity
                current_solar[building_id, 1] = capacity * 1200  # Annual generation
                current_solar[building_id, 2] = capacity * 1200 * 0.3  # Self-consumption
                current_solar[building_id, 3] = capacity * 1200 * 0.7  # Export
            
            # Predict evolution
            predictions = self.forward(
                building_features,
                current_solar,
                time_step,
                current_clusters,
                edge_index
            )
            
            # Update clusters for next iteration
            current_clusters = predictions['cluster_assignments']
            
            # Store trajectory point
            trajectory.append({
                'year': year_idx + 1,
                'clusters': current_clusters.cpu().numpy(),
                'energy_flows': predictions['energy_flows'].cpu().numpy(),
                'stability': predictions['stability_scores'].mean().item(),
                'num_clusters': current_clusters.unique().shape[0]
            })
        
        return trajectory
    
    def _apply_graph_constraints(
        self,
        cluster_logits: torch.Tensor,
        current_clusters: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply graph constraints to maintain cluster connectivity
        
        Args:
            cluster_logits: Raw cluster predictions
            current_clusters: Current assignments
            edge_index: Building connections
            
        Returns:
            Constrained cluster logits
        """
        # Encourage connected buildings to be in same cluster
        if edge_index.shape[1] > 0:
            # Create adjacency bonus
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i], edge_index[1, i]
                
                # If neighbors are in same cluster, boost that cluster's logit
                if current_clusters[src] == current_clusters[dst]:
                    cluster_id = current_clusters[src]
                    cluster_logits[src, cluster_id] += 0.5
                    cluster_logits[dst, cluster_id] += 0.5
        
        return cluster_logits


class ClusterStabilityAnalyzer:
    """
    Analyzes cluster stability and reorganization patterns
    """
    
    def __init__(self):
        """Initialize stability analyzer"""
        self.history = []
        logger.info("Initialized ClusterStabilityAnalyzer")
    
    def analyze_transition(
        self,
        clusters_before: torch.Tensor,
        clusters_after: torch.Tensor,
        solar_changes: Dict[int, float]
    ) -> Dict:
        """
        Analyze cluster transition after solar installations
        
        Args:
            clusters_before: Cluster assignments before solar
            clusters_after: Cluster assignments after solar
            solar_changes: Buildings that got solar and capacities
            
        Returns:
            Transition analysis metrics
        """
        # Calculate stability metrics
        unchanged = (clusters_before == clusters_after).float().mean().item()
        
        # Find buildings that changed clusters
        changed_mask = clusters_before != clusters_after
        num_changed = changed_mask.sum().item()
        
        # Analyze solar impact on changes
        solar_buildings = set(solar_changes.keys())
        changed_indices = changed_mask.nonzero(as_tuple=True)[0].tolist()
        solar_induced_changes = len(solar_buildings.intersection(changed_indices))
        
        # Calculate cluster size changes
        unique_before = clusters_before.unique()
        unique_after = clusters_after.unique()
        
        cluster_sizes_before = {
            c.item(): (clusters_before == c).sum().item() 
            for c in unique_before
        }
        
        cluster_sizes_after = {
            c.item(): (clusters_after == c).sum().item() 
            for c in unique_after
        }
        
        # Identify splits and merges
        splits = []
        merges = []
        
        for c in unique_before:
            c_val = c.item()
            if c_val not in cluster_sizes_after:
                # Cluster disappeared - likely merged
                merges.append(c_val)
            elif cluster_sizes_after[c_val] < cluster_sizes_before[c_val] * 0.5:
                # Cluster significantly shrunk - likely split
                splits.append(c_val)
        
        analysis = {
            'stability_score': unchanged,
            'num_buildings_changed': num_changed,
            'solar_induced_changes': solar_induced_changes,
            'num_clusters_before': len(unique_before),
            'num_clusters_after': len(unique_after),
            'cluster_splits': splits,
            'cluster_merges': merges,
            'avg_cluster_size_before': np.mean(list(cluster_sizes_before.values())),
            'avg_cluster_size_after': np.mean(list(cluster_sizes_after.values()))
        }
        
        # Store in history
        self.history.append(analysis)
        
        return analysis
    
    def predict_stability(
        self,
        current_clusters: torch.Tensor,
        planned_solar: List[int],
        building_features: torch.Tensor
    ) -> float:
        """
        Predict cluster stability if solar is installed
        
        Args:
            current_clusters: Current cluster assignments
            planned_solar: Buildings planned for solar
            building_features: Building features
            
        Returns:
            Predicted stability score (0-1)
        """
        # Simple heuristic: more solar in diverse clusters = less stable
        affected_clusters = current_clusters[planned_solar].unique()
        
        # Calculate diversity of affected clusters
        cluster_diversity = len(affected_clusters) / (current_clusters.max().item() + 1)
        
        # Calculate energy impact
        total_consumption = building_features[:, 3].sum().item()
        solar_generation = len(planned_solar) * 5 * 1200  # 5kWp average, 1200 kWh/kWp
        energy_impact = solar_generation / total_consumption
        
        # Stability decreases with diversity and energy impact
        stability = 1.0 - (cluster_diversity * 0.3 + energy_impact * 0.7)
        stability = max(0.1, min(1.0, stability))
        
        return stability


class EnergyFlowEvolution:
    """
    Models evolution of energy flows as solar penetration increases
    """
    
    def __init__(self, config: Dict):
        """
        Initialize energy flow evolution model
        
        Args:
            config: Configuration parameters
        """
        self.self_consumption_base = config.get('self_consumption_base', 0.3)
        self.sharing_efficiency = config.get('sharing_efficiency', 0.85)
        self.grid_loss_factor = config.get('grid_loss_factor', 0.03)
        
        logger.info("Initialized EnergyFlowEvolution")
    
    def calculate_flows(
        self,
        solar_capacity: Dict[int, float],
        consumption_profiles: torch.Tensor,
        cluster_assignments: torch.Tensor,
        time_of_day: Optional[int] = None
    ) -> Dict:
        """
        Calculate energy flows with current solar deployment
        
        Args:
            solar_capacity: Solar capacity per building (kWp)
            consumption_profiles: Consumption patterns [N, T]
            cluster_assignments: Cluster assignments [N]
            time_of_day: Hour of day (0-23) for temporal analysis
            
        Returns:
            Energy flow metrics
        """
        num_buildings = consumption_profiles.shape[0]
        device = consumption_profiles.device
        
        # Initialize flows
        flows = {
            'self_consumption': torch.zeros(num_buildings, device=device),
            'peer_sharing': torch.zeros(num_buildings, device=device),
            'grid_export': torch.zeros(num_buildings, device=device),
            'grid_import': torch.zeros(num_buildings, device=device)
        }
        
        # Calculate solar generation
        solar_generation = torch.zeros(num_buildings, device=device)
        for building_id, capacity in solar_capacity.items():
            if time_of_day is not None:
                # Hourly generation profile
                hour_factor = self._solar_hour_factor(time_of_day)
                solar_generation[building_id] = capacity * hour_factor
            else:
                # Daily average
                solar_generation[building_id] = capacity * 4.8  # kWh/day per kWp
        
        # Process each cluster
        unique_clusters = cluster_assignments.unique()
        
        for cluster_id in unique_clusters:
            cluster_mask = cluster_assignments == cluster_id
            cluster_indices = cluster_mask.nonzero(as_tuple=True)[0]
            
            if len(cluster_indices) == 0:
                continue
            
            # Cluster totals
            cluster_generation = solar_generation[cluster_mask].sum()
            cluster_consumption = consumption_profiles[cluster_mask].sum()
            
            # Self-consumption within buildings
            for idx in cluster_indices:
                building_gen = solar_generation[idx].item()
                building_cons = consumption_profiles[idx].item() if consumption_profiles.dim() == 1 else consumption_profiles[idx].mean().item()
                
                self_consumed = min(building_gen, building_cons * self.self_consumption_base)
                flows['self_consumption'][idx] = self_consumed
                
                # Remaining generation available for sharing
                available = building_gen - self_consumed
                
                if available > 0:
                    # Share within cluster
                    cluster_need = cluster_consumption - cluster_generation
                    if cluster_need > 0:
                        shared = min(available, cluster_need / len(cluster_indices))
                        flows['peer_sharing'][idx] = shared * self.sharing_efficiency
                        available -= shared
                    
                    # Export remainder to grid
                    if available > 0:
                        flows['grid_export'][idx] = available * (1 - self.grid_loss_factor)
                
                # Import needs
                import_need = building_cons - self_consumed - flows['peer_sharing'][idx]
                if import_need > 0:
                    flows['grid_import'][idx] = import_need
        
        # Calculate summary metrics
        total_generation = solar_generation.sum().item()
        total_consumption = consumption_profiles.sum().item() if consumption_profiles.dim() == 1 else consumption_profiles.mean(dim=1).sum().item()
        
        metrics = {
            'total_self_consumption': flows['self_consumption'].sum().item(),
            'total_peer_sharing': flows['peer_sharing'].sum().item(),
            'total_grid_export': flows['grid_export'].sum().item(),
            'total_grid_import': flows['grid_import'].sum().item(),
            'self_sufficiency_rate': (flows['self_consumption'].sum() + flows['peer_sharing'].sum()) / (total_consumption + 1e-10),
            'sharing_rate': flows['peer_sharing'].sum() / (total_generation + 1e-10),
            'export_rate': flows['grid_export'].sum() / (total_generation + 1e-10),
            'flows_per_building': {
                k: v.cpu().numpy() for k, v in flows.items()
            }
        }
        
        return metrics
    
    def _solar_hour_factor(self, hour: int) -> float:
        """
        Solar generation factor for given hour
        
        Args:
            hour: Hour of day (0-23)
            
        Returns:
            Generation factor (0-1)
        """
        if hour < 6 or hour > 20:
            return 0.0
        elif hour < 9:
            return 0.3
        elif hour < 11:
            return 0.7
        elif hour < 14:
            return 1.0
        elif hour < 16:
            return 0.8
        elif hour < 18:
            return 0.5
        else:
            return 0.2
    
    def project_evolution(
        self,
        roadmap_years: List[Dict],
        initial_consumption: torch.Tensor,
        initial_clusters: torch.Tensor
    ) -> List[Dict]:
        """
        Project energy flow evolution over roadmap timeline
        
        Args:
            roadmap_years: Yearly installation plans
            initial_consumption: Initial consumption profiles
            initial_clusters: Initial cluster assignments
            
        Returns:
            Projected flow evolution
        """
        evolution = []
        cumulative_solar = {}
        
        for year_idx, year_plan in enumerate(roadmap_years):
            # Update cumulative solar
            for building_id, capacity in year_plan.items():
                cumulative_solar[building_id] = cumulative_solar.get(building_id, 0) + capacity
            
            # Calculate flows for this year
            flows = self.calculate_flows(
                cumulative_solar,
                initial_consumption,
                initial_clusters
            )
            
            flows['year'] = year_idx + 1
            flows['total_solar_capacity'] = sum(cumulative_solar.values())
            flows['num_buildings_with_solar'] = len(cumulative_solar)
            
            evolution.append(flows)
        
        return evolution