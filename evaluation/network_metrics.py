"""
Network effect evaluation metrics
Measures multi-hop impacts to prove GNN value
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class NetworkEffectEvaluator:
    """
    Evaluates multi-hop network effects to demonstrate GNN value
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.metrics_history = []
        
    def compare_to_baseline(
        self,
        gnn_interventions: List[int],
        baseline_interventions: List[int],
        network_state: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        cascade_simulator
    ) -> Dict[str, float]:
        """
        Compare GNN selections to baseline
        
        Args:
            gnn_interventions: Nodes selected by GNN
            baseline_interventions: Nodes selected by baseline method
            network_state: Current network state
            edge_index: Network connectivity
            cascade_simulator: Simulator for cascade effects
            
        Returns:
            Comparison metrics
        """
        # Simulate both intervention sets
        gnn_cascades = self._simulate_cascades(
            gnn_interventions, network_state, edge_index, cascade_simulator
        )
        
        baseline_cascades = self._simulate_cascades(
            baseline_interventions, network_state, edge_index, cascade_simulator
        )
        
        # Calculate metrics
        metrics = {
            'gnn_total_impact': self._calculate_total_impact(gnn_cascades),
            'baseline_total_impact': self._calculate_total_impact(baseline_cascades),
            'gnn_multi_hop_ratio': self._calculate_multi_hop_ratio(gnn_cascades),
            'baseline_multi_hop_ratio': self._calculate_multi_hop_ratio(baseline_cascades),
            'gnn_coverage': self._calculate_coverage(gnn_cascades),
            'baseline_coverage': self._calculate_coverage(baseline_cascades),
            'gnn_efficiency': self._calculate_efficiency(gnn_cascades, len(gnn_interventions)),
            'baseline_efficiency': self._calculate_efficiency(baseline_cascades, len(baseline_interventions))
        }
        
        # Calculate improvements
        if metrics['baseline_total_impact'] > 0:
            metrics['impact_improvement'] = (
                (metrics['gnn_total_impact'] - metrics['baseline_total_impact']) / 
                metrics['baseline_total_impact']
            )
        else:
            metrics['impact_improvement'] = 0
        
        metrics['multi_hop_improvement'] = (
            metrics['gnn_multi_hop_ratio'] - metrics['baseline_multi_hop_ratio']
        )
        
        return metrics
    
    def measure_cascade_magnitude(
        self,
        intervention: Dict[str, int],
        cascade_effects: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Quantify cascade effects at each hop distance
        
        Args:
            intervention: Intervention details
            cascade_effects: Cascade effects by hop
            
        Returns:
            Magnitude metrics by hop
        """
        magnitudes = {}
        
        for hop in range(1, 4):
            hop_key = f'hop_{hop}'
            if hop_key in cascade_effects:
                effects = cascade_effects[hop_key]
                
                # Energy impact magnitude
                energy_magnitude = effects['energy_impact'].abs().sum().item()
                
                # Congestion relief magnitude
                congestion_magnitude = effects['congestion_relief'].abs().sum().item()
                
                # Economic value
                economic_magnitude = effects['economic_value'].abs().sum().item()
                
                magnitudes[hop_key] = {
                    'energy': energy_magnitude,
                    'congestion': congestion_magnitude,
                    'economic': economic_magnitude,
                    'total': energy_magnitude + congestion_magnitude + economic_magnitude
                }
        
        # Calculate decay rate
        if magnitudes.get('hop_1', {}).get('total', 0) > 0:
            magnitudes['decay_rate'] = [
                magnitudes.get(f'hop_{i}', {}).get('total', 0) / 
                magnitudes.get('hop_1', {}).get('total', 1)
                for i in range(2, 4)
            ]
        
        return magnitudes
    
    def track_network_evolution(
        self,
        network_evolution: List[Dict],
        save_path: Optional[Path] = None
    ) -> Dict[str, List[float]]:
        """
        Track how network dynamics change over intervention rounds
        
        Args:
            network_evolution: Network states over rounds
            save_path: Optional path to save visualization
            
        Returns:
            Evolution metrics
        """
        metrics = {
            'peak_demand': [],
            'self_sufficiency': [],
            'congestion': [],
            'complementarity': [],
            'cluster_quality': []
        }
        
        for state_dict in network_evolution:
            state = state_dict['state']
            
            # Peak demand
            peak = state['net_demand'].max().item()
            metrics['peak_demand'].append(peak)
            
            # Self-sufficiency
            if state['demand'].sum() > 0:
                self_suff = torch.minimum(
                    state['generation'], state['demand']
                ).sum() / state['demand'].sum()
                metrics['self_sufficiency'].append(self_suff.item())
            else:
                metrics['self_sufficiency'].append(0)
            
            # Average congestion
            metrics['congestion'].append(state.get('congestion', torch.tensor(0)).mean().item())
            
            # Average complementarity
            metrics['complementarity'].append(
                state.get('complementarity', torch.tensor(0)).mean().item()
            )
            
            # Cluster quality (if available)
            if 'clusters' in state_dict:
                clusters = state_dict['clusters']
                # Simple quality metric: variance of cluster sizes
                if isinstance(clusters, torch.Tensor) and clusters.dim() == 2:
                    cluster_sizes = clusters.sum(dim=0)
                    quality = 1.0 / (cluster_sizes.std().item() + 1)
                    metrics['cluster_quality'].append(quality)
                else:
                    metrics['cluster_quality'].append(0)
        
        # Visualize if path provided
        if save_path:
            self._visualize_evolution(metrics, save_path)
        
        return metrics
    
    def validate_multi_hop_effects(
        self,
        cascade_effects: Dict[str, Dict[str, torch.Tensor]],
        threshold: float = 0.3
    ) -> bool:
        """
        Validate that multi-hop effects are significant
        
        Args:
            cascade_effects: Cascade effects by hop
            threshold: Minimum ratio for multi-hop effects
            
        Returns:
            True if multi-hop effects exceed threshold
        """
        total_value = 0
        multi_hop_value = 0
        
        for hop_key, effects in cascade_effects.items():
            if 'hop_' in hop_key:
                hop_num = int(hop_key.split('_')[1])
                value = sum(
                    e.abs().sum().item() 
                    for e in effects.values() 
                    if isinstance(e, torch.Tensor)
                )
                
                total_value += value
                if hop_num > 1:
                    multi_hop_value += value
        
        if total_value > 0:
            ratio = multi_hop_value / total_value
            logger.info(f"Multi-hop ratio: {ratio:.2%} (threshold: {threshold:.0%})")
            return ratio >= threshold
        
        return False
    
    def calculate_network_centrality_impact(
        self,
        selected_nodes: List[int],
        edge_index: torch.Tensor,
        node_features: torch.Tensor
    ) -> Dict[str, float]:
        """
        Calculate impact based on network centrality
        
        Args:
            selected_nodes: Selected intervention nodes
            edge_index: Network connectivity
            node_features: Node feature matrix
            
        Returns:
            Centrality-based impact metrics
        """
        n_nodes = node_features.size(0)
        
        # Calculate various centrality measures
        degrees = torch.bincount(edge_index[0], minlength=n_nodes).float()
        
        # Selected nodes' centralities
        selected_degrees = degrees[selected_nodes]
        
        metrics = {
            'avg_degree_centrality': selected_degrees.mean().item(),
            'max_degree_centrality': selected_degrees.max().item(),
            'total_degree_coverage': selected_degrees.sum().item(),
            'centrality_diversity': selected_degrees.std().item()
        }
        
        # Calculate betweenness proxy
        betweenness = self._calculate_betweenness_proxy(edge_index, n_nodes)
        selected_betweenness = betweenness[selected_nodes]
        
        metrics['avg_betweenness'] = selected_betweenness.mean().item()
        metrics['strategic_position_score'] = (
            0.5 * metrics['avg_degree_centrality'] / degrees.mean().item() +
            0.5 * metrics['avg_betweenness']
        )
        
        return metrics
    
    def _simulate_cascades(
        self,
        interventions: List[int],
        network_state: Dict[str, torch.Tensor],
        edge_index: torch.Tensor,
        simulator
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Simulate cascade effects for interventions"""
        cumulative_cascades = {}
        
        for node in interventions:
            # Simple intervention
            intervention = {
                'building_id': node,
                'type': 'solar',
                'generation_profile': torch.randn(24) * 5 + 10
            }
            
            cascades = simulator.calculate_cascade_effects(
                intervention,
                network_state,
                edge_index,
                max_hops=3
            )
            
            # Accumulate
            for hop_key, effects in cascades.items():
                if hop_key not in cumulative_cascades:
                    cumulative_cascades[hop_key] = {}
                
                for effect_type, values in effects.items():
                    if effect_type not in cumulative_cascades[hop_key]:
                        cumulative_cascades[hop_key][effect_type] = torch.zeros_like(values)
                    cumulative_cascades[hop_key][effect_type] += values
        
        return cumulative_cascades
    
    def _calculate_total_impact(self, cascades: Dict) -> float:
        """Calculate total impact across all hops"""
        total = 0
        for hop_key, effects in cascades.items():
            if 'hop_' in hop_key:
                for effect_type, values in effects.items():
                    if isinstance(values, torch.Tensor):
                        total += values.abs().sum().item()
        return total
    
    def _calculate_multi_hop_ratio(self, cascades: Dict) -> float:
        """Calculate ratio of multi-hop effects"""
        hop_values = [0, 0, 0]
        
        for i in range(3):
            hop_key = f'hop_{i+1}'
            if hop_key in cascades:
                for values in cascades[hop_key].values():
                    if isinstance(values, torch.Tensor):
                        hop_values[i] += values.abs().sum().item()
        
        total = sum(hop_values)
        if total > 0:
            return (hop_values[1] + hop_values[2]) / total
        return 0
    
    def _calculate_coverage(self, cascades: Dict) -> int:
        """Calculate number of affected nodes"""
        affected = set()
        
        for hop_key, effects in cascades.items():
            if 'hop_' in hop_key and 'energy_impact' in effects:
                impact = effects['energy_impact']
                affected.update(torch.where(impact > 0)[0].tolist())
        
        return len(affected)
    
    def _calculate_efficiency(self, cascades: Dict, num_interventions: int) -> float:
        """Calculate impact per intervention"""
        total_impact = self._calculate_total_impact(cascades)
        return total_impact / num_interventions if num_interventions > 0 else 0
    
    def _calculate_betweenness_proxy(
        self,
        edge_index: torch.Tensor,
        n_nodes: int
    ) -> torch.Tensor:
        """Simple betweenness centrality proxy"""
        betweenness = torch.zeros(n_nodes)
        
        # Count paths through each node (simplified)
        for src in range(n_nodes):
            # BFS from source
            distances = torch.full((n_nodes,), float('inf'))
            distances[src] = 0
            queue = [src]
            paths = {src: 1}
            
            while queue:
                current = queue.pop(0)
                neighbors = edge_index[1][edge_index[0] == current].tolist()
                
                for neighbor in neighbors:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
                        paths[neighbor] = paths.get(neighbor, 0) + paths.get(current, 1)
            
            # Update betweenness
            for node in range(n_nodes):
                if node != src and paths.get(node, 0) > 0:
                    betweenness[node] += 1.0 / paths[node]
        
        # Normalize
        if n_nodes > 2:
            betweenness = betweenness / ((n_nodes - 1) * (n_nodes - 2))
        
        return betweenness
    
    def _visualize_evolution(self, metrics: Dict[str, List[float]], save_path: Path):
        """Visualize network evolution metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        rounds = list(range(len(metrics['peak_demand'])))
        
        # Peak demand
        axes[0, 0].plot(rounds, metrics['peak_demand'], 'o-')
        axes[0, 0].set_title('Peak Demand Evolution')
        axes[0, 0].set_xlabel('Round')
        axes[0, 0].set_ylabel('Peak (kW)')
        axes[0, 0].grid(True)
        
        # Self-sufficiency
        axes[0, 1].plot(rounds, metrics['self_sufficiency'], 's-')
        axes[0, 1].set_title('Self-Sufficiency')
        axes[0, 1].set_xlabel('Round')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].grid(True)
        
        # Congestion
        axes[0, 2].plot(rounds, metrics['congestion'], '^-')
        axes[0, 2].set_title('Average Congestion')
        axes[0, 2].set_xlabel('Round')
        axes[0, 2].set_ylabel('Congestion Level')
        axes[0, 2].grid(True)
        
        # Complementarity
        axes[1, 0].plot(rounds, metrics['complementarity'], 'd-')
        axes[1, 0].set_title('Network Complementarity')
        axes[1, 0].set_xlabel('Round')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].grid(True)
        
        # Cluster quality
        if metrics['cluster_quality']:
            axes[1, 1].plot(rounds, metrics['cluster_quality'], 'p-')
            axes[1, 1].set_title('Cluster Quality')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Quality Score')
            axes[1, 1].grid(True)
        
        # Summary: Improvement rates
        improvements = {
            'Peak Reduction': (metrics['peak_demand'][0] - metrics['peak_demand'][-1]) / metrics['peak_demand'][0] * 100,
            'Self-Suff Gain': (metrics['self_sufficiency'][-1] - metrics['self_sufficiency'][0]) * 100,
            'Congestion Relief': (metrics['congestion'][0] - metrics['congestion'][-1]) / metrics['congestion'][0] * 100 if metrics['congestion'][0] > 0 else 0
        }
        
        axes[1, 2].bar(improvements.keys(), improvements.values())
        axes[1, 2].set_title('Overall Improvements (%)')
        axes[1, 2].set_ylabel('Improvement %')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()