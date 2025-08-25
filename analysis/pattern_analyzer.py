"""
Pattern Analyzer for Energy GNN
Analyzes discovered clusters to identify opportunities and gaps
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import networkx as nx
from scipy import stats
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class ClusterMetrics:
    """Data class for cluster performance metrics"""
    cluster_id: int
    num_buildings: int
    self_sufficiency: float
    self_consumption: float
    peak_reduction: float
    load_factor: float
    diversity_factor: float
    complementarity_score: float
    network_stress: float
    carbon_reduction: float
    economic_benefit: float


@dataclass
class EnergyGap:
    """Data class for identified energy gaps"""
    cluster_id: int
    timestamp: int
    gap_type: str  # 'generation', 'storage', 'demand_reduction'
    magnitude: float  # kW
    duration: int  # hours
    affected_buildings: List[str]
    priority: float
    recommended_intervention: str


@dataclass
class NetworkBottleneck:
    """Data class for network bottlenecks"""
    location: str  # transformer or cable ID
    type: str  # 'transformer', 'cable', 'substation'
    peak_load: float
    capacity: float
    utilization: float
    affected_clusters: List[int]
    criticality: float


class PatternAnalyzer:
    """
    Comprehensive analyzer for GNN-discovered patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize analyzer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.peak_hours = config.get('peak_hours', [17, 18, 19, 20])  # 5-8 PM
        self.solar_hours = config.get('solar_hours', list(range(9, 17)))  # 9 AM - 5 PM
        self.carbon_intensity = config.get('carbon_intensity', 0.5)  # kg CO2/kWh
        self.electricity_price = config.get('electricity_price', 0.15)  # $/kWh
        
    def analyze_clusters(
        self,
        cluster_assignments: torch.Tensor,
        temporal_profiles: torch.Tensor,
        building_features: torch.Tensor,
        edge_index: torch.Tensor,
        complementarity_matrix: Optional[torch.Tensor] = None,
        generation_profiles: Optional[torch.Tensor] = None,
        network_data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive cluster analysis
        
        Args:
            cluster_assignments: Cluster assignments [N]
            temporal_profiles: Consumption profiles [N, T]
            building_features: Building features [N, D]
            edge_index: Graph connectivity
            complementarity_matrix: Pairwise complementarity scores [N, N]
            generation_profiles: Generation profiles [N, T]
            network_data: Grid network information
            
        Returns:
            Comprehensive analysis results
        """
        results = {
            'cluster_metrics': [],
            'energy_gaps': [],
            'network_bottlenecks': [],
            'optimization_opportunities': [],
            'temporal_patterns': {},
            'network_effects': {}
        }
        
        # Convert to numpy for easier manipulation
        clusters = cluster_assignments.cpu().numpy()
        profiles = temporal_profiles.cpu().numpy()
        features = building_features.cpu().numpy()
        
        if generation_profiles is not None:
            generation = generation_profiles.cpu().numpy()
        else:
            generation = np.zeros_like(profiles)
            
        unique_clusters = np.unique(clusters)
        
        # Analyze each cluster
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_profiles = profiles[mask]
            cluster_generation = generation[mask]
            cluster_features = features[mask]
            
            # Calculate cluster metrics
            metrics = self._calculate_cluster_metrics(
                cluster_id,
                cluster_profiles,
                cluster_generation,
                complementarity_matrix[mask][:, mask] if complementarity_matrix is not None else None
            )
            results['cluster_metrics'].append(metrics)
            
            # Identify gaps
            gaps = self._identify_gaps(
                cluster_id,
                cluster_profiles,
                cluster_generation,
                np.where(mask)[0]
            )
            results['energy_gaps'].extend(gaps)
            
        # Analyze network bottlenecks
        if network_data:
            bottlenecks = self._analyze_network_bottlenecks(
                clusters,
                profiles,
                network_data
            )
            results['network_bottlenecks'] = bottlenecks
            
        # Identify optimization opportunities
        opportunities = self._identify_opportunities(
            results['cluster_metrics'],
            results['energy_gaps']
        )
        results['optimization_opportunities'] = opportunities
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(
            clusters,
            profiles,
            generation
        )
        results['temporal_patterns'] = temporal_patterns
        
        # Quantify network effects
        network_effects = self._quantify_network_effects(
            cluster_assignments,
            edge_index,
            temporal_profiles
        )
        results['network_effects'] = network_effects
        
        return results
    
    def _calculate_cluster_metrics(
        self,
        cluster_id: int,
        consumption: np.ndarray,
        generation: np.ndarray,
        complementarity_matrix: Optional[np.ndarray]
    ) -> ClusterMetrics:
        """
        Calculate comprehensive metrics for a cluster
        
        Args:
            cluster_id: Cluster identifier
            consumption: Consumption profiles [n, T]
            generation: Generation profiles [n, T]
            complementarity_matrix: Complementarity scores [n, n]
            
        Returns:
            ClusterMetrics object
        """
        n_buildings = consumption.shape[0]
        
        # Aggregate profiles
        total_consumption = consumption.sum(axis=0)
        total_generation = generation.sum(axis=0)
        
        # Self-sufficiency: how much demand is met by local generation
        self_consumed = np.minimum(total_consumption, total_generation)
        self_sufficiency = self_consumed.sum() / total_consumption.sum() if total_consumption.sum() > 0 else 0
        
        # Self-consumption: how much generation is used locally
        self_consumption = self_consumed.sum() / total_generation.sum() if total_generation.sum() > 0 else 1
        
        # Peak reduction
        individual_peaks = consumption.max(axis=1).sum()
        cluster_peak = total_consumption.max()
        peak_reduction = 1 - (cluster_peak / individual_peaks) if individual_peaks > 0 else 0
        
        # Load factor (average load / peak load)
        avg_load = total_consumption.mean()
        load_factor = avg_load / cluster_peak if cluster_peak > 0 else 0
        
        # Diversity factor (sum of individual peaks / group peak)
        diversity_factor = individual_peaks / cluster_peak if cluster_peak > 0 else 1
        
        # Complementarity score (average negative correlation)
        if complementarity_matrix is not None:
            # Convert to numpy if tensor
            if hasattr(complementarity_matrix, 'cpu'):
                complementarity_matrix = complementarity_matrix.cpu().numpy()
            # Lower triangular to avoid double counting
            lower_tri = np.tril(complementarity_matrix, k=-1)
            comp_scores = lower_tri[lower_tri != 0]
            complementarity_score = -comp_scores.mean() if len(comp_scores) > 0 else 0
        else:
            complementarity_score = self._calculate_complementarity(consumption)
            
        # Network stress (simplified - ratio of peak to average)
        network_stress = cluster_peak / avg_load if avg_load > 0 else 1
        
        # Carbon reduction (from increased self-sufficiency)
        grid_consumption = np.maximum(0, total_consumption - total_generation)
        carbon_saved = (total_consumption.sum() - grid_consumption.sum()) * self.carbon_intensity
        carbon_reduction = carbon_saved / (total_consumption.sum() * self.carbon_intensity) if total_consumption.sum() > 0 else 0
        
        # Economic benefit (simplified)
        economic_benefit = (
            peak_reduction * 100 +  # Peak charge reduction
            self_sufficiency * total_consumption.sum() * self.electricity_price  # Energy cost savings
        )
        
        return ClusterMetrics(
            cluster_id=cluster_id,
            num_buildings=n_buildings,
            self_sufficiency=self_sufficiency,
            self_consumption=self_consumption,
            peak_reduction=peak_reduction,
            load_factor=load_factor,
            diversity_factor=diversity_factor,
            complementarity_score=complementarity_score,
            network_stress=network_stress,
            carbon_reduction=carbon_reduction,
            economic_benefit=economic_benefit
        )
    
    def _calculate_complementarity(self, profiles: np.ndarray) -> float:
        """
        Calculate complementarity score from consumption profiles
        
        Args:
            profiles: Consumption profiles [n, T]
            
        Returns:
            Complementarity score (higher is better)
        """
        n_buildings = profiles.shape[0]
        if n_buildings < 2:
            return 0
            
        # Normalize profiles
        profiles_norm = (profiles - profiles.mean(axis=1, keepdims=True)) / (
            profiles.std(axis=1, keepdims=True) + 1e-8
        )
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(profiles_norm)
        
        # Get lower triangular (exclude diagonal)
        lower_tri = np.tril(corr_matrix, k=-1)
        correlations = lower_tri[lower_tri != 0]
        
        # Complementarity is negative correlation
        return -correlations.mean() if len(correlations) > 0 else 0
    
    def _identify_gaps(
        self,
        cluster_id: int,
        consumption: np.ndarray,
        generation: np.ndarray,
        building_indices: np.ndarray
    ) -> List[EnergyGap]:
        """
        Identify energy gaps in the cluster
        
        Args:
            cluster_id: Cluster identifier
            consumption: Consumption profiles [n, T]
            generation: Generation profiles [n, T]
            building_indices: Indices of buildings in cluster
            
        Returns:
            List of identified gaps
        """
        gaps = []
        
        # Aggregate profiles
        total_consumption = consumption.sum(axis=0)
        total_generation = generation.sum(axis=0)
        
        # Net demand
        net_demand = total_consumption - total_generation
        
        # Analyze each hour
        timesteps_per_hour = 4  # 15-minute intervals
        for hour in range(24):
            hour_slice = slice(hour * timesteps_per_hour, (hour + 1) * timesteps_per_hour)
            hour_demand = net_demand[hour_slice].mean()
            
            if hour_demand > 0:  # Generation gap
                # Check if it's during solar hours
                if hour in self.solar_hours:
                    gap_type = 'generation'
                    intervention = 'solar_pv'
                elif hour in self.peak_hours:
                    gap_type = 'storage'
                    intervention = 'battery'
                else:
                    gap_type = 'demand_reduction'
                    intervention = 'load_shifting'
                    
                # Find buildings with highest demand at this time
                hour_consumption = consumption[:, hour_slice].mean(axis=1)
                top_consumers = building_indices[np.argsort(hour_consumption)[-3:]]
                
                gap = EnergyGap(
                    cluster_id=cluster_id,
                    timestamp=hour,
                    gap_type=gap_type,
                    magnitude=hour_demand,
                    duration=1,
                    affected_buildings=top_consumers.tolist(),
                    priority=hour_demand / total_consumption.max(),  # Normalized priority
                    recommended_intervention=intervention
                )
                gaps.append(gap)
                
        # Merge consecutive gaps of same type
        gaps = self._merge_consecutive_gaps(gaps)
        
        return gaps
    
    def _merge_consecutive_gaps(self, gaps: List[EnergyGap]) -> List[EnergyGap]:
        """Merge consecutive gaps of the same type"""
        if not gaps:
            return gaps
            
        merged = []
        current_gap = gaps[0]
        
        for gap in gaps[1:]:
            if (gap.gap_type == current_gap.gap_type and 
                gap.timestamp == current_gap.timestamp + current_gap.duration):
                # Merge gaps
                current_gap.duration += 1
                current_gap.magnitude = max(current_gap.magnitude, gap.magnitude)
                current_gap.affected_buildings = list(set(
                    current_gap.affected_buildings + gap.affected_buildings
                ))
            else:
                merged.append(current_gap)
                current_gap = gap
                
        merged.append(current_gap)
        return merged
    
    def _analyze_network_bottlenecks(
        self,
        clusters: np.ndarray,
        profiles: np.ndarray,
        network_data: Dict
    ) -> List[NetworkBottleneck]:
        """
        Analyze network bottlenecks
        
        Args:
            clusters: Cluster assignments
            profiles: Consumption profiles
            network_data: Network topology and capacity data
            
        Returns:
            List of bottlenecks
        """
        bottlenecks = []
        
        # Analyze transformer loading
        if 'transformers' in network_data:
            for transformer in network_data['transformers']:
                transformer_id = transformer['id']
                capacity = transformer['capacity']
                
                # Find buildings connected to this transformer
                connected_buildings = transformer.get('connected_buildings', [])
                if not connected_buildings:
                    continue
                    
                # Calculate peak load
                connected_profiles = profiles[connected_buildings]
                peak_load = connected_profiles.sum(axis=0).max()
                
                utilization = peak_load / capacity if capacity > 0 else 1.0
                
                if utilization > 0.8:  # Bottleneck threshold
                    # Find affected clusters
                    affected_clusters = list(set(clusters[connected_buildings]))
                    
                    bottleneck = NetworkBottleneck(
                        location=transformer_id,
                        type='transformer',
                        peak_load=peak_load,
                        capacity=capacity,
                        utilization=utilization,
                        affected_clusters=affected_clusters,
                        criticality=utilization * len(affected_clusters)
                    )
                    bottlenecks.append(bottleneck)
                    
        # Analyze cable loading
        if 'cables' in network_data:
            for cable in network_data['cables']:
                cable_id = cable['id']
                capacity = cable.get('capacity', 100)  # Default 100kW
                
                # Simplified: assume uniform distribution
                connected_buildings = cable.get('connected_buildings', [])
                if not connected_buildings:
                    continue
                    
                cable_load = profiles[connected_buildings].sum(axis=0).max()
                utilization = cable_load / capacity if capacity > 0 else 1.0
                
                if utilization > 0.8:
                    affected_clusters = list(set(clusters[connected_buildings]))
                    
                    bottleneck = NetworkBottleneck(
                        location=cable_id,
                        type='cable',
                        peak_load=cable_load,
                        capacity=capacity,
                        utilization=utilization,
                        affected_clusters=affected_clusters,
                        criticality=utilization * len(connected_buildings)
                    )
                    bottlenecks.append(bottleneck)
                    
        # Sort by criticality
        bottlenecks.sort(key=lambda x: x.criticality, reverse=True)
        
        return bottlenecks
    
    def _identify_opportunities(
        self,
        cluster_metrics: List[ClusterMetrics],
        energy_gaps: List[EnergyGap]
    ) -> List[Dict[str, Any]]:
        """
        Identify optimization opportunities
        
        Args:
            cluster_metrics: List of cluster metrics
            energy_gaps: List of energy gaps
            
        Returns:
            List of opportunities
        """
        opportunities = []
        
        # Opportunity 1: Clusters with low self-sufficiency but high complementarity
        for metrics in cluster_metrics:
            if metrics.self_sufficiency < 0.3 and metrics.complementarity_score > 0.5:
                opportunities.append({
                    'type': 'high_potential_cluster',
                    'cluster_id': metrics.cluster_id,
                    'reason': 'High complementarity but low self-sufficiency',
                    'potential_improvement': (0.7 - metrics.self_sufficiency) * 100,
                    'priority': metrics.complementarity_score * (1 - metrics.self_sufficiency)
                })
                
        # Opportunity 2: Large generation gaps during solar hours
        solar_gaps = [g for g in energy_gaps if g.gap_type == 'generation' and g.timestamp in self.solar_hours]
        for gap in sorted(solar_gaps, key=lambda x: x.magnitude, reverse=True)[:5]:
            opportunities.append({
                'type': 'solar_opportunity',
                'cluster_id': gap.cluster_id,
                'magnitude': gap.magnitude,
                'duration': gap.duration,
                'affected_buildings': gap.affected_buildings,
                'estimated_solar_size': gap.magnitude * 1.2,  # 20% oversizing
                'priority': gap.priority
            })
            
        # Opportunity 3: Peak shaving with storage
        peak_gaps = [g for g in energy_gaps if g.gap_type == 'storage']
        for gap in sorted(peak_gaps, key=lambda x: x.magnitude, reverse=True)[:3]:
            opportunities.append({
                'type': 'storage_opportunity',
                'cluster_id': gap.cluster_id,
                'peak_reduction_potential': gap.magnitude,
                'battery_size': gap.magnitude * gap.duration,  # kWh
                'priority': gap.priority
            })
            
        # Opportunity 4: Clusters with poor load factor
        for metrics in cluster_metrics:
            if metrics.load_factor < 0.4:
                opportunities.append({
                    'type': 'load_management',
                    'cluster_id': metrics.cluster_id,
                    'current_load_factor': metrics.load_factor,
                    'improvement_potential': 0.6 - metrics.load_factor,
                    'strategy': 'demand_response',
                    'priority': 1 - metrics.load_factor
                })
                
        # Sort by priority
        opportunities.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return opportunities
    
    def _analyze_temporal_patterns(
        self,
        clusters: np.ndarray,
        consumption: np.ndarray,
        generation: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze temporal patterns in clusters
        
        Args:
            clusters: Cluster assignments
            consumption: Consumption profiles
            generation: Generation profiles
            
        Returns:
            Temporal pattern analysis
        """
        patterns = {}
        unique_clusters = np.unique(clusters)
        
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_consumption = consumption[mask].sum(axis=0)
            cluster_generation = generation[mask].sum(axis=0)
            
            # Reshape to hours (assuming 15-min intervals)
            hourly_consumption = cluster_consumption.reshape(24, -1).mean(axis=1)
            hourly_generation = cluster_generation.reshape(24, -1).mean(axis=1)
            
            patterns[f'cluster_{cluster_id}'] = {
                'peak_hours': np.argsort(hourly_consumption)[-3:].tolist(),
                'valley_hours': np.argsort(hourly_consumption)[:3].tolist(),
                'max_generation_hours': np.argsort(hourly_generation)[-3:].tolist(),
                'net_export_hours': np.where(hourly_generation > hourly_consumption)[0].tolist(),
                'net_import_hours': np.where(hourly_consumption > hourly_generation)[0].tolist(),
                'daily_pattern_stability': np.std(hourly_consumption),
                'consumption_generation_correlation': np.corrcoef(
                    hourly_consumption, hourly_generation
                )[0, 1] if hourly_generation.sum() > 0 else 0
            }
            
        return patterns
    
    def _quantify_network_effects(
        self,
        cluster_assignments: torch.Tensor,
        edge_index: torch.Tensor,
        temporal_profiles: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Quantify network effects discovered by GNN
        
        Args:
            cluster_assignments: Cluster assignments
            edge_index: Graph connectivity
            temporal_profiles: Consumption profiles
            
        Returns:
            Network effect metrics
        """
        # Convert to NetworkX for analysis
        G = nx.Graph()
        edge_list = edge_index.cpu().numpy().T
        G.add_edges_from(edge_list)
        
        clusters = cluster_assignments.cpu().numpy()
        profiles = temporal_profiles.cpu().numpy()
        
        effects = {}
        
        # 1. Multi-hop effects
        multi_hop_effects = self._calculate_multi_hop_effects(G, clusters, profiles)
        effects['multi_hop'] = multi_hop_effects
        
        # 2. Cascade potential
        cascade_potential = self._calculate_cascade_potential(G, clusters, profiles)
        effects['cascade'] = cascade_potential
        
        # 3. Network centrality of clusters
        cluster_centrality = self._calculate_cluster_centrality(G, clusters)
        effects['centrality'] = cluster_centrality
        
        # 4. Information flow efficiency
        flow_efficiency = self._calculate_flow_efficiency(G, clusters)
        effects['flow_efficiency'] = flow_efficiency
        
        return effects
    
    def _calculate_multi_hop_effects(
        self,
        G: nx.Graph,
        clusters: np.ndarray,
        profiles: np.ndarray
    ) -> Dict[str, float]:
        """Calculate multi-hop network effects"""
        effects = {}
        
        # Calculate correlation decay with distance
        correlations_by_distance = {1: [], 2: [], 3: []}
        
        for node in G.nodes():
            # Get neighbors at different distances
            for distance in [1, 2, 3]:
                neighbors = [n for n in nx.single_source_shortest_path_length(
                    G, node, cutoff=distance
                ).keys() if n != node]
                
                if neighbors:
                    # Calculate average correlation with neighbors at this distance
                    node_profile = profiles[node]
                    for neighbor in neighbors:
                        if neighbor < len(profiles):
                            corr = np.corrcoef(node_profile, profiles[neighbor])[0, 1]
                            correlations_by_distance[distance].append(corr)
                            
        # Average correlation by distance
        for distance, corrs in correlations_by_distance.items():
            if corrs:
                effects[f'correlation_at_{distance}_hop'] = np.mean(corrs)
                
        # Calculate how clustering preserves multi-hop relationships
        cluster_coherence = 0
        for cluster_id in np.unique(clusters):
            cluster_nodes = np.where(clusters == cluster_id)[0]
            if len(cluster_nodes) > 1:
                # Check if cluster nodes are within 2 hops
                subgraph = G.subgraph(cluster_nodes)
                if nx.is_connected(subgraph):
                    avg_path_length = nx.average_shortest_path_length(subgraph)
                    cluster_coherence += 1 / (1 + avg_path_length)
                    
        effects['cluster_network_coherence'] = cluster_coherence / len(np.unique(clusters))
        
        return effects
    
    def _calculate_cascade_potential(
        self,
        G: nx.Graph,
        clusters: np.ndarray,
        profiles: np.ndarray
    ) -> Dict[str, float]:
        """Calculate cascade potential of interventions"""
        cascade_metrics = {}
        
        # Identify hub nodes (high betweenness centrality)
        betweenness = nx.betweenness_centrality(G)
        hub_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate potential cascade from hubs
        total_cascade = 0
        for hub, centrality in hub_nodes:
            # Simulate intervention at hub
            affected_nodes = nx.single_source_shortest_path_length(G, hub, cutoff=3).keys()
            cascade_size = len(affected_nodes)
            
            # Weight by cluster diversity
            affected_clusters = len(set(clusters[list(affected_nodes)]))
            
            total_cascade += cascade_size * affected_clusters * centrality
            
        cascade_metrics['total_cascade_potential'] = total_cascade
        cascade_metrics['avg_cascade_size'] = total_cascade / len(hub_nodes)
        
        return cascade_metrics
    
    def _calculate_cluster_centrality(
        self,
        G: nx.Graph,
        clusters: np.ndarray
    ) -> Dict[str, float]:
        """Calculate network centrality of clusters"""
        centrality_metrics = {}
        
        # Calculate various centrality measures
        degree_centrality = nx.degree_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Average centrality per cluster
        for cluster_id in np.unique(clusters):
            cluster_nodes = np.where(clusters == cluster_id)[0]
            
            avg_degree = np.mean([degree_centrality.get(n, 0) for n in cluster_nodes])
            avg_closeness = np.mean([closeness_centrality.get(n, 0) for n in cluster_nodes])
            
            centrality_metrics[f'cluster_{cluster_id}_degree'] = avg_degree
            centrality_metrics[f'cluster_{cluster_id}_closeness'] = avg_closeness
            
        return centrality_metrics
    
    def _calculate_flow_efficiency(
        self,
        G: nx.Graph,
        clusters: np.ndarray
    ) -> Dict[str, float]:
        """Calculate energy flow efficiency metrics"""
        efficiency_metrics = {}
        
        # Global efficiency
        efficiency_metrics['global_efficiency'] = nx.global_efficiency(G)
        
        # Local efficiency (resilience)
        efficiency_metrics['local_efficiency'] = nx.local_efficiency(G)
        
        # Cluster-based efficiency
        for cluster_id in np.unique(clusters):
            cluster_nodes = np.where(clusters == cluster_id)[0]
            if len(cluster_nodes) > 1:
                subgraph = G.subgraph(cluster_nodes)
                if len(subgraph) > 0:
                    efficiency_metrics[f'cluster_{cluster_id}_efficiency'] = nx.global_efficiency(subgraph)
                    
        return efficiency_metrics
    
    def generate_report(
        self,
        analysis_results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate comprehensive analysis report
        
        Args:
            analysis_results: Results from analyze_clusters
            output_path: Optional path to save report
            
        Returns:
            Report as DataFrame
        """
        # Create summary DataFrame
        metrics_df = pd.DataFrame([vars(m) for m in analysis_results['cluster_metrics']])
        
        # Add gap summary
        gap_summary = pd.DataFrame(analysis_results['energy_gaps'])
        if not gap_summary.empty:
            gap_stats = gap_summary.groupby('cluster_id').agg({
                'magnitude': ['sum', 'max', 'mean'],
                'duration': 'sum',
                'gap_type': lambda x: x.mode()[0] if len(x) > 0 else 'none'
            })
            # Flatten MultiIndex columns created by multiple agg functions
            gap_stats.columns = ['_'.join(col).strip() if col[1] else col[0] 
                                 for col in gap_stats.columns.values]
            # Reset index to make cluster_id a regular column
            gap_stats = gap_stats.reset_index()
            metrics_df = metrics_df.merge(gap_stats, on='cluster_id', how='left')
            
        # Add network effects
        network_effects = analysis_results['network_effects']
        metrics_df['multi_hop_correlation'] = network_effects['multi_hop'].get('correlation_at_2_hop', 0)
        metrics_df['cascade_potential'] = network_effects['cascade'].get('total_cascade_potential', 0)
        
        # Sort by performance
        metrics_df['overall_score'] = (
            metrics_df['self_sufficiency'] * 0.3 +
            metrics_df['peak_reduction'] * 0.3 +
            metrics_df['complementarity_score'] * 0.2 +
            (1 - metrics_df['network_stress']) * 0.2
        )
        metrics_df = metrics_df.sort_values('overall_score', ascending=False)
        
        if output_path:
            metrics_df.to_csv(output_path, index=False)
            print(f"Report saved to {output_path}")
            
        return metrics_df
    
    def visualize_patterns(
        self,
        analysis_results: Dict[str, Any],
        save_path: Optional[str] = None
    ):
        """
        Create visualizations of discovered patterns
        
        Args:
            analysis_results: Results from analyze_clusters
            save_path: Optional path to save visualizations
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Cluster performance comparison
        metrics_df = pd.DataFrame([vars(m) for m in analysis_results['cluster_metrics']])
        
        ax = axes[0, 0]
        metrics_df[['self_sufficiency', 'peak_reduction', 'complementarity_score']].plot(
            kind='bar', ax=ax
        )
        ax.set_title('Cluster Performance Metrics')
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Score')
        ax.legend(loc='best')
        
        # 2. Gap distribution
        gap_df = pd.DataFrame(analysis_results['energy_gaps'])
        if not gap_df.empty:
            ax = axes[0, 1]
            gap_summary = gap_df.groupby('gap_type')['magnitude'].sum()
            gap_summary.plot(kind='pie', ax=ax, autopct='%1.1f%%')
            ax.set_title('Energy Gap Distribution')
            
        # 3. Temporal patterns
        temporal_patterns = analysis_results['temporal_patterns']
        if temporal_patterns:
            ax = axes[0, 2]
            first_cluster = list(temporal_patterns.keys())[0]
            peak_hours = temporal_patterns[first_cluster]['peak_hours']
            valley_hours = temporal_patterns[first_cluster]['valley_hours']
            
            hours = list(range(24))
            pattern = [1 if h in peak_hours else -1 if h in valley_hours else 0 for h in hours]
            ax.bar(hours, pattern, color=['red' if p > 0 else 'green' if p < 0 else 'gray' for p in pattern])
            ax.set_title('Peak and Valley Hours')
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Load Level')
            
        # 4. Network bottlenecks
        if analysis_results['network_bottlenecks']:
            ax = axes[1, 0]
            bottlenecks_df = pd.DataFrame([vars(b) for b in analysis_results['network_bottlenecks'][:5]])
            bottlenecks_df.plot(x='location', y='utilization', kind='barh', ax=ax)
            ax.set_title('Top Network Bottlenecks')
            ax.set_xlabel('Utilization')
            ax.axvline(x=0.8, color='r', linestyle='--', label='Threshold')
            
        # 5. Opportunity priorities
        if analysis_results['optimization_opportunities']:
            ax = axes[1, 1]
            opps = analysis_results['optimization_opportunities'][:5]
            opp_types = [o['type'] for o in opps]
            opp_priorities = [o.get('priority', 0) for o in opps]
            ax.bar(range(len(opps)), opp_priorities)
            ax.set_xticks(range(len(opps)))
            ax.set_xticklabels(opp_types, rotation=45, ha='right')
            ax.set_title('Top Optimization Opportunities')
            ax.set_ylabel('Priority Score')
            
        # 6. Network effects
        network_effects = analysis_results['network_effects']
        if network_effects:
            ax = axes[1, 2]
            multi_hop = network_effects['multi_hop']
            distances = []
            correlations = []
            for key, value in multi_hop.items():
                if 'correlation_at' in key:
                    dist = int(key.split('_')[2])
                    distances.append(dist)
                    correlations.append(value)
            if distances:
                ax.plot(distances, correlations, 'o-')
                ax.set_title('Correlation Decay with Network Distance')
                ax.set_xlabel('Network Distance (hops)')
                ax.set_ylabel('Average Correlation')
                ax.grid(True)
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualizations saved to {save_path}")
            
        plt.show()