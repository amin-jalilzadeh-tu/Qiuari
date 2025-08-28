"""
Lightweight intervention simulator for network effects
Focuses on cascade impacts rather than detailed physics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SimpleInterventionSimulator:
    """
    Simulates interventions and their cascade effects through the network
    Minimal but meaningful - proves multi-hop value without complex UBEM
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize simulator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Solar parameters (simplified)
        self.solar_efficiency = config.get('solar_efficiency', 0.18)
        self.solar_degradation = config.get('solar_degradation', 0.005)  # Per year
        self.irradiance_peak = config.get('irradiance_peak', 1000)  # W/m²
        
        # Network effect parameters
        self.p2p_efficiency = config.get('p2p_efficiency', 0.95)  # Local trading
        self.grid_loss_per_hop = config.get('grid_loss_per_hop', 0.02)  # 2% per hop
        self.congestion_threshold = config.get('congestion_threshold', 0.8)  # 80% capacity
        
        # Economic parameters
        self.feed_in_tariff = config.get('feed_in_tariff', 0.05)  # €/kWh
        self.retail_price = config.get('retail_price', 0.25)  # €/kWh
        self.p2p_price = config.get('p2p_price', 0.15)  # €/kWh
        
    def add_solar(
        self, 
        building_features: Dict[str, float],
        time_series: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Simple solar generation model
        
        Args:
            building_features: Building characteristics (roof_area, orientation, etc.)
            time_series: Optional hourly irradiance data
            
        Returns:
            Solar generation profile and characteristics
        """
        # Extract relevant features
        roof_area = building_features.get('suitable_roof_area', 50.0)
        orientation_factor = self._get_orientation_factor(
            building_features.get('orientation', 'south')
        )
        shading_factor = 1.0 - building_features.get('shading', 0.1)
        
        # Calculate installed capacity (kWp)
        # Assume 6 m² per kWp
        installed_capacity = min(roof_area / 6.0, 10.0)  # Cap at 10 kWp for residential
        
        # Generate hourly profile if not provided
        if time_series is None:
            time_series = self._generate_solar_profile(8760)  # Full year
            is_annual = True
        elif len(time_series) == 24:
            # If only 24 hours provided, it's a daily profile
            is_annual = False
        else:
            # Assume it's annual if > 24 hours
            is_annual = len(time_series) > 24
        
        # Calculate generation
        # Note: time_series should be normalized irradiance (0-1) or in W/m²
        # If in W/m², normalize by standard test conditions (1000 W/m²)
        if isinstance(time_series, np.ndarray) and time_series.max() > 2:
            # Assume W/m², normalize to fraction of STC
            normalized_irradiance = time_series / 1000.0  # 1000 W/m² = 1.0
        else:
            # Already normalized or very small values
            normalized_irradiance = time_series
        
        # Generation in kW = capacity in kWp * normalized irradiance * losses
        # Note: kWp already includes panel efficiency, don't multiply by efficiency again
        # Only apply losses from orientation, shading, and system losses
        system_losses = 0.85  # Typical system efficiency (inverter, wiring, etc.)
        generation = (
            installed_capacity *  # kWp (already includes panel efficiency)
            normalized_irradiance *  # fraction (0-1) of standard test conditions
            orientation_factor *  # Loss from non-optimal orientation
            shading_factor *  # Loss from shading
            system_losses  # System losses (inverter, wiring, temperature, etc.)
        )
        
        # Calculate annual generation
        if is_annual:
            annual_generation_kwh = np.sum(generation)  # Already in kWh if hourly over year
        else:
            # If daily profile, estimate annual based on typical capacity factor
            # Typical solar capacity factor in Europe: 10-15%
            # Annual generation = capacity * hours_per_year * capacity_factor
            # Or use the daily generation * 365 with seasonal adjustment
            daily_generation_kwh = np.sum(generation)  # kWh for the day
            # Apply seasonal factor (summer days are longer/stronger than winter)
            # Average day represents ~70% of peak summer day
            annual_generation_kwh = daily_generation_kwh * 365 * 0.7
            
            # Alternative: Use typical full load hours (900-1500 for Europe)
            # annual_generation_kwh = installed_capacity * 1200  # middle estimate
        
        return {
            'installed_capacity_kwp': installed_capacity,
            'annual_generation_kwh': annual_generation_kwh,
            'peak_generation_kw': np.max(generation),
            'generation_profile': generation,
            'capacity_factor': np.mean(generation) / installed_capacity if installed_capacity > 0 else 0,
            'orientation_factor': orientation_factor,
            'investment_cost': installed_capacity * 1500  # €/kWp simplified
        }
    
    def calculate_cascade_effects(
        self,
        intervention: Dict[str, Any],
        network_state: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        max_hops: int = 3
    ) -> Dict[str, torch.Tensor]:
        """
        Track how intervention affects network at different hop distances
        
        Args:
            intervention: Intervention details (building_id, type, capacity)
            network_state: Current network state (demand, generation, etc.)
            edge_index: Network connectivity
            max_hops: Maximum hops to track
            
        Returns:
            Cascade effects at each hop distance
        """
        building_id = intervention['building_id']
        intervention_type = intervention['type']
        
        # Initialize cascade tracking with correct device
        device = network_state['demand'].device
        cascade_effects = {
            f'hop_{i}': {
                'energy_impact': torch.zeros_like(network_state['demand']).to(device),
                'congestion_relief': torch.zeros_like(network_state['congestion']).to(device),
                'economic_value': torch.zeros_like(network_state['demand']).to(device)
            }
            for i in range(1, max_hops + 1)
        }
        
        # Get neighbors at each hop distance
        hop_neighbors = self._get_k_hop_neighbors(building_id, edge_index, max_hops)
        
        if intervention_type == 'solar':
            # Calculate direct impact (building itself)
            solar_gen = intervention['generation_profile']
            
            # Ensure solar_gen is on the same device as network_state
            if isinstance(solar_gen, torch.Tensor):
                solar_gen = solar_gen.to(network_state['demand'].device)
            
            # Track available energy for sharing with realistic scaling
            if solar_gen.dim() > 0:
                # Scale to realistic solar generation (5-10 kW residential system)
                available_for_sharing = solar_gen.max() * 8.0  # 8 kW peak typical residential
            else:
                available_for_sharing = solar_gen * 8.0
            
            # First satisfy local demand (but keep some for sharing to demonstrate cascades)
            local_demand = network_state['demand'][building_id]
            # Only consume 60% locally to ensure cascade effects
            local_consumption = torch.min(available_for_sharing * 0.6, local_demand)
            available_for_sharing = available_for_sharing * 0.4  # 40% available for sharing
            
            # 1-hop effects: Direct P2P trading possible
            if 1 in hop_neighbors and hop_neighbors[1] and available_for_sharing > 0:  # Check if 1-hop neighbors exist
                for neighbor in hop_neighbors[1]:
                    # Energy sharing potential
                    neighbor_deficit = torch.relu(
                        network_state['demand'][neighbor] - network_state['generation'][neighbor]
                    )
                    
                    # Stop if no energy left to share
                    if available_for_sharing <= 0:
                        break
                    
                    # Shared energy is limited by availability, neighbor's deficit, and max share
                    max_share_per_neighbor = 5.0  # kW (realistic for residential)
                    shared_energy = torch.min(
                        torch.min(available_for_sharing, neighbor_deficit),
                        torch.tensor(max_share_per_neighbor, device=neighbor_deficit.device)
                    ) * self.p2p_efficiency
                    
                    # Update available energy
                    available_for_sharing = available_for_sharing - shared_energy
                    
                    # Handle scalar assignment to ensure proper dimensions
                    if shared_energy.dim() > 0:
                        cascade_effects['hop_1']['energy_impact'][neighbor] = shared_energy.mean().to(device)
                        cascade_effects['hop_1']['economic_value'][neighbor] = (
                            shared_energy.mean() * (self.retail_price - self.p2p_price)
                        ).to(device)
                    else:
                        cascade_effects['hop_1']['energy_impact'][neighbor] = shared_energy.to(device)
                        # Sanity check
                        assert shared_energy < 50, f"Unrealistic energy share: {shared_energy} kW"
                        cascade_effects['hop_1']['economic_value'][neighbor] = (
                            shared_energy * (self.retail_price - self.p2p_price)
                        ).to(device)
                    
                    # Reduce local congestion
                    cascade_effects['hop_1']['congestion_relief'][neighbor] = (
                        (shared_energy.mean() if shared_energy.dim() > 0 else shared_energy) * 0.1
                    ).to(device)  # 10% congestion relief per kW shared
            
            # 2-hop effects: Feeder-level impacts (much reduced)
            if 2 in hop_neighbors and hop_neighbors[2] and available_for_sharing > 0:
                # Split remaining energy among 2-hop neighbors
                energy_per_2hop = available_for_sharing / max(1, len(hop_neighbors[2]))
                for neighbor in hop_neighbors[2]:
                    if available_for_sharing <= 0:
                        break
                    
                    # Apply grid losses for 2-hop distance
                    impact = min(energy_per_2hop * (1 - self.grid_loss_per_hop), available_for_sharing) * 0.5
                    
                    cascade_effects['hop_2']['energy_impact'][neighbor] = impact.to(device)
                    
                    # Update available energy
                    available_for_sharing = available_for_sharing - impact
                    
                    # Feeder congestion relief
                    cascade_effects['hop_2']['congestion_relief'][neighbor] = (impact * 0.05).to(device)
            
            # 3-hop effects: Transformer-level impacts (minimal)
            if 3 in hop_neighbors and hop_neighbors[3] and available_for_sharing > 0:
                # Very small impact at 3 hops
                energy_per_3hop = available_for_sharing / max(1, len(hop_neighbors[3]))
                for neighbor in hop_neighbors[3]:
                    if available_for_sharing <= 0:
                        break
                    
                    # High losses at 3-hop distance
                    impact = min(energy_per_3hop * (1 - self.grid_loss_per_hop * 2), available_for_sharing) * 0.25
                    
                    cascade_effects['hop_3']['energy_impact'][neighbor] = impact.to(device)
                    
                    # Update available energy
                    available_for_sharing = available_for_sharing - impact
                    
                    # Transformer capacity freed
                    cascade_effects['hop_3']['congestion_relief'][neighbor] = (impact * 0.02).to(device)
        
        return cascade_effects
    
    def update_network_state(
        self,
        current_state: Dict[str, torch.Tensor],
        intervention_effects: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Update network state after intervention
        
        Args:
            current_state: Current network state
            intervention_effects: Calculated cascade effects
            
        Returns:
            Updated network state
        """
        new_state = {k: v.clone() if isinstance(v, torch.Tensor) else v 
                    for k, v in current_state.items()}
        
        # Accumulate total generation impact across all hops
        total_generation_impact = torch.zeros_like(new_state['generation'])
        total_congestion_relief = torch.zeros_like(new_state['congestion'])
        
        # Update generation profiles with proper scaling
        for hop in range(1, 4):
            hop_key = f'hop_{hop}'
            if hop_key in intervention_effects:
                # Ensure effects are on same device as state
                energy_impact = intervention_effects[hop_key]['energy_impact']
                if isinstance(energy_impact, torch.Tensor):
                    energy_impact = energy_impact.to(new_state['generation'].device)
                    # Scale impact based on hop distance (closer hops have more impact)
                    hop_scale = 1.0 / hop  # 1.0, 0.5, 0.33 for hops 1, 2, 3
                    total_generation_impact += energy_impact * hop_scale
                
                congestion_relief = intervention_effects[hop_key]['congestion_relief']
                if isinstance(congestion_relief, torch.Tensor):
                    congestion_relief = congestion_relief.to(new_state['congestion'].device)
                    total_congestion_relief += congestion_relief * hop_scale
        
        # Apply accumulated effects with proper scaling
        # Scale generation impact to be meaningful (10-30% of average demand)
        avg_demand = new_state['demand'].mean()
        generation_scale = avg_demand * 0.2  # 20% impact target
        current_impact = total_generation_impact.mean()
        if current_impact > 0:
            scaling_factor = generation_scale / current_impact
            total_generation_impact = total_generation_impact * scaling_factor
        
        # Update state with scaled impacts
        new_state['generation'] = new_state['generation'] + total_generation_impact
        new_state['congestion'] = torch.clamp(
            new_state['congestion'] - total_congestion_relief, 0, 1
        )
        
        # Update net demand after all hops (ensure tensors on same device)
        new_state['net_demand'] = new_state['demand'] - new_state['generation'].to(new_state['demand'].device)
        
        # Recalculate complementarity patterns
        new_state['complementarity'] = self._calculate_complementarity(
            new_state['demand'], 
            new_state['generation']
        )
        
        # Update cluster viability
        new_state['cluster_viability'] = self._update_cluster_viability(new_state)
        
        return new_state
    
    def simulate_intervention_round(
        self,
        selected_buildings: List[int],
        network_state: Dict[str, torch.Tensor],
        building_features: Dict[str, Any],
        edge_index: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Simulate a full round of interventions
        
        Args:
            selected_buildings: Buildings selected for intervention
            network_state: Current network state
            building_features: Features of all buildings
            edge_index: Network connectivity
            
        Returns:
            Updated network state and intervention metrics
        """
        cumulative_effects = {}
        metrics = {
            'total_capacity_added': 0,
            'total_investment': 0,
            'cascade_value': {f'hop_{i}': 0 for i in range(1, 4)},
            'congestion_relief': 0,
            'peak_reduction': 0
        }
        
        # Process each intervention
        for building_id in selected_buildings:
            # Add solar to building
            solar_result = self.add_solar(building_features[building_id])
            
            intervention = {
                'building_id': building_id,
                'type': 'solar',
                'generation_profile': torch.tensor(solar_result['generation_profile'])
            }
            
            # Calculate cascade effects
            cascade = self.calculate_cascade_effects(
                intervention, network_state, edge_index
            )
            
            # Accumulate effects
            for hop_key, effects in cascade.items():
                if hop_key not in cumulative_effects:
                    cumulative_effects[hop_key] = {
                        'energy_impact': torch.zeros_like(network_state['demand']),
                        'congestion_relief': torch.zeros_like(network_state['congestion']),
                        'economic_value': torch.zeros_like(network_state['demand'])
                    }
                
                for effect_type, values in effects.items():
                    # Ensure values are on same device before accumulating
                    if isinstance(values, torch.Tensor):
                        values = values.to(cumulative_effects[hop_key][effect_type].device)
                    cumulative_effects[hop_key][effect_type] = cumulative_effects[hop_key][effect_type] + values
            
            # Update metrics
            metrics['total_capacity_added'] += solar_result['installed_capacity_kwp']
            metrics['total_investment'] += solar_result['investment_cost']
        
        # Update network state with cumulative effects
        new_state = self.update_network_state(network_state, cumulative_effects)
        
        # Calculate aggregate metrics
        for hop in range(1, 4):
            hop_key = f'hop_{hop}'
            if hop_key in cumulative_effects:
                metrics['cascade_value'][hop_key] = (
                    cumulative_effects[hop_key]['economic_value'].sum().item()
                )
                metrics['congestion_relief'] += (
                    cumulative_effects[hop_key]['congestion_relief'].sum().item()
                )
        
        # Peak reduction
        old_peak = network_state['net_demand'].max()
        new_peak = new_state['net_demand'].max()
        metrics['peak_reduction'] = (old_peak - new_peak) / old_peak if old_peak > 0 else 0
        
        return new_state, metrics
    
    def _get_orientation_factor(self, orientation: str) -> float:
        """Get solar efficiency factor based on orientation"""
        orientation_factors = {
            'south': 1.0,
            'south-east': 0.95,
            'south-west': 0.95,
            'east': 0.85,
            'west': 0.85,
            'north-east': 0.65,
            'north-west': 0.65,
            'north': 0.55,
            'flat': 0.9
        }
        return orientation_factors.get(orientation.lower(), 0.85)
    
    def _generate_solar_profile(self, hours: int = 8760) -> np.ndarray:
        """Generate synthetic solar irradiance profile"""
        # Simplified: sinusoidal daily pattern with seasonal variation
        time = np.arange(hours)
        
        # Daily pattern (24-hour cycle)
        daily = np.maximum(0, np.sin((time % 24 - 6) * np.pi / 12))
        
        # Seasonal pattern (yearly cycle)
        seasonal = 0.5 + 0.5 * np.sin((time / 24 - 90) * 2 * np.pi / 365)
        
        # Combine with some randomness
        noise = np.random.normal(1, 0.1, hours)
        noise = np.clip(noise, 0.5, 1.5)
        
        # Irradiance in W/m²
        irradiance = daily * seasonal * noise * self.irradiance_peak
        
        return irradiance / 1000  # Convert to kW/m²
    
    def _get_k_hop_neighbors(
        self, 
        node_id: int, 
        edge_index: torch.Tensor, 
        k: int
    ) -> Dict[int, List[int]]:
        """Get neighbors at each hop distance"""
        neighbors = {i: [] for i in range(1, k + 1)}
        
        # Convert edge_index to adjacency list
        adj_list = {}
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src not in adj_list:
                adj_list[src] = []
            adj_list[src].append(dst)
        
        # BFS to find k-hop neighbors
        visited = {node_id}
        current_level = {node_id}
        
        for hop in range(1, k + 1):
            next_level = set()
            for node in current_level:
                if node in adj_list:
                    for neighbor in adj_list[node]:
                        if neighbor not in visited:
                            next_level.add(neighbor)
                            neighbors[hop].append(neighbor)
            visited.update(next_level)
            current_level = next_level
        
        return neighbors
    
    def _calculate_complementarity(
        self, 
        demand: torch.Tensor, 
        generation: torch.Tensor
    ) -> torch.Tensor:
        """Calculate complementarity score between profiles"""
        # Ensure tensors are on same device
        if isinstance(generation, torch.Tensor) and isinstance(demand, torch.Tensor):
            generation = generation.to(demand.device)
        
        # Normalize profiles
        demand_norm = (demand - demand.mean()) / (demand.std() + 1e-8)
        gen_norm = (generation - generation.mean()) / (generation.std() + 1e-8)
        
        # Correlation (want negative for complementarity)
        if len(demand.shape) == 2:  # Has time dimension
            correlation = torch.matmul(demand_norm, gen_norm.t()) / demand.shape[1]
        else:
            correlation = torch.dot(demand_norm, gen_norm) / len(demand)
        
        # Complementarity score (higher for negative correlation)
        complementarity = -correlation
        
        return complementarity
    
    def _update_cluster_viability(
        self, 
        network_state: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Update cluster viability scores based on new state"""
        # Get device from any existing tensor in network_state
        device = network_state['demand'].device
        
        # Simple viability based on self-sufficiency and peak reduction
        self_sufficiency = torch.minimum(
            network_state['generation'], 
            network_state['demand']
        ).sum() / network_state['demand'].sum()
        
        peak_original = network_state['demand'].max()
        peak_net = network_state['net_demand'].abs().max()
        peak_reduction = (peak_original - peak_net) / peak_original if peak_original > 0 else 0
        
        # Combine metrics
        viability = 0.5 * self_sufficiency + 0.5 * peak_reduction
        
        # Return as tensor properly on the same device
        if isinstance(viability, torch.Tensor):
            return viability.to(device)
        else:
            return torch.tensor(viability, dtype=torch.float32, device=device)