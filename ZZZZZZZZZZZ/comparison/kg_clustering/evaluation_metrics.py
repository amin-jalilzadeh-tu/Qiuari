"""
Evaluation Metrics for Energy Complementarity Clustering
Comprehensive metrics aligned with research papers and KG structure
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import logging
from scipy.stats import entropy
from datetime import datetime

logger = logging.getLogger(__name__)

class ClusteringEvaluator:
    """
    Evaluate clustering results based on energy complementarity objectives.
    """
    
    def __init__(self, preprocessed_data: Dict[str, Any]):
        """
        Initialize evaluator with preprocessed data.
        
        Args:
            preprocessed_data: Output from KGDataPreprocessor
        """
        self.data = preprocessed_data
        self.topology = preprocessed_data['topology']
        self.time_series = preprocessed_data['time_series']
        self.complementarity = preprocessed_data['complementarity']
        self.constraints = preprocessed_data['constraints']
        self.building_features = preprocessed_data['building_features']
        self.electrical_distances = preprocessed_data['electrical_distances']
        
        logger.info("Initialized clustering evaluator")
    
    def evaluate_clustering(self, clusters: Dict[str, List[str]], 
                           method_name: str = "Unknown") -> Dict[str, Any]:
        """
        Comprehensive evaluation of clustering results.
        
        Args:
            clusters: Dictionary mapping cluster_id -> list of building_ids
            method_name: Name of clustering method
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info(f"Evaluating clustering from {method_name}")
        
        metrics = {
            'method': method_name,
            'timestamp': datetime.now().isoformat(),
            'cluster_count': len(clusters),
            'clustered_buildings': sum(len(c) for c in clusters.values()),
            
            # Primary metrics
            'peak_reduction': self._calculate_peak_reduction(clusters),
            'self_sufficiency': self._calculate_self_sufficiency(clusters),
            'complementarity_score': self._calculate_complementarity_score(clusters),
            'constraint_violations': self._count_constraint_violations(clusters),
            
            # Secondary metrics
            'network_losses': self._estimate_network_losses(clusters),
            'transformer_loading': self._calculate_transformer_loading(clusters),
            'diversity_index': self._calculate_diversity_index(clusters),
            'fairness_index': self._calculate_fairness_index(clusters),
            
            # Clustering quality metrics
            'modularity': self._calculate_modularity(clusters),
            'conductance': self._calculate_conductance(clusters),
            'coverage': self._calculate_coverage(clusters),
            'stability': self._calculate_stability(clusters),
            
            # Detailed breakdown
            'cluster_sizes': self._get_cluster_size_distribution(clusters),
            'per_cluster_metrics': self._calculate_per_cluster_metrics(clusters)
        }
        
        # Add summary statistics
        metrics['summary'] = self._create_summary(metrics)
        
        logger.info(f"Evaluation complete. Peak reduction: {metrics['peak_reduction']:.1%}, "
                   f"Self-sufficiency: {metrics['self_sufficiency']:.1%}, "
                   f"Violations: {metrics['constraint_violations']}")
        
        return metrics
    
    def _calculate_peak_reduction(self, clusters: Dict[str, List[str]]) -> float:
        """
        Calculate peak demand reduction from clustering.
        Compare aggregated peak vs sum of individual peaks.
        """
        if not clusters:
            return 0.0
        
        total_individual_peak = 0
        total_aggregated_peak = 0
        
        for cluster_id, building_ids in clusters.items():
            if not building_ids:
                continue
            
            # Get time series for cluster buildings
            cluster_peaks = []
            cluster_aggregated = None
            
            for bid in building_ids:
                bid_str = str(bid)
                if bid_str in self.time_series:
                    ts = self.time_series[bid_str]
                    if len(ts) > 0:
                        # Column 3 is electricity demand
                        demand = ts[:, 3]
                        cluster_peaks.append(np.max(demand))
                        
                        if cluster_aggregated is None:
                            cluster_aggregated = demand.copy()
                        else:
                            cluster_aggregated += demand
            
            if cluster_peaks and cluster_aggregated is not None:
                total_individual_peak += sum(cluster_peaks)
                total_aggregated_peak += np.max(cluster_aggregated)
        
        if total_individual_peak > 0:
            reduction = 1 - (total_aggregated_peak / total_individual_peak)
            return max(0, reduction)  # Ensure non-negative
        
        return 0.0
    
    def _calculate_self_sufficiency(self, clusters: Dict[str, List[str]]) -> float:
        """
        Calculate average self-sufficiency ratio across clusters.
        Self-sufficiency = local generation / local consumption.
        """
        if not clusters:
            return 0.0
        
        cluster_ratios = []
        
        for cluster_id, building_ids in clusters.items():
            if not building_ids:
                continue
            
            total_generation = 0
            total_consumption = 0
            
            for bid in building_ids:
                bid_str = str(bid)
                if bid_str in self.time_series:
                    ts = self.time_series[bid_str]
                    if len(ts) > 0:
                        # Column 5 is solar generation
                        total_generation += np.sum(ts[:, 5])
                        # Column 3 is electricity demand
                        total_consumption += np.sum(ts[:, 3])
            
            if total_consumption > 0:
                ratio = min(1.0, total_generation / total_consumption)
                cluster_ratios.append(ratio)
        
        return np.mean(cluster_ratios) if cluster_ratios else 0.0
    
    def _calculate_complementarity_score(self, clusters: Dict[str, List[str]]) -> float:
        """
        Calculate average complementarity within clusters.
        """
        if not clusters:
            return 0.0
        
        bid_to_idx = self.constraints['bid_to_idx']
        cluster_scores = []
        
        for cluster_id, building_ids in clusters.items():
            if len(building_ids) < 2:
                continue
            
            # Get indices for buildings
            indices = []
            for bid in building_ids:
                bid_str = str(bid)
                if bid_str in bid_to_idx:
                    indices.append(bid_to_idx[bid_str])
            
            if len(indices) < 2:
                continue
            
            # Calculate average pairwise complementarity
            comp_sum = 0
            count = 0
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    comp_sum += self.complementarity[indices[i], indices[j]]
                    count += 1
            
            if count > 0:
                cluster_scores.append(comp_sum / count)
        
        return np.mean(cluster_scores) if cluster_scores else 0.0
    
    def _count_constraint_violations(self, clusters: Dict[str, List[str]]) -> int:
        """
        Count violations of electrical constraints.
        Buildings in same cluster must be in same cable group.
        """
        violations = 0
        bid_to_idx = self.constraints['bid_to_idx']
        
        for cluster_id, building_ids in clusters.items():
            if len(building_ids) < 2:
                continue
            
            # Check if all buildings are in same cable group
            cable_groups = set()
            for bid in building_ids:
                bid_str = str(bid)
                for cg_id, cg_buildings in self.constraints['cable_groups'].items():
                    if bid_str in bid_to_idx and bid_to_idx[bid_str] in cg_buildings:
                        cable_groups.add(cg_id)
                        break
            
            # Violation if buildings span multiple cable groups
            if len(cable_groups) > 1:
                violations += len(cable_groups) - 1
        
        return violations
    
    def _estimate_network_losses(self, clusters: Dict[str, List[str]]) -> float:
        """
        Estimate network losses based on electrical distances.
        Lower is better.
        """
        if not clusters:
            return 0.0
        
        bid_to_idx = self.constraints['bid_to_idx']
        total_loss = 0
        total_flow = 0
        
        for cluster_id, building_ids in clusters.items():
            if len(building_ids) < 2:
                continue
            
            # Calculate average electrical distance within cluster
            indices = []
            for bid in building_ids:
                bid_str = str(bid)
                if bid_str in bid_to_idx:
                    indices.append(bid_to_idx[bid_str])
            
            if len(indices) < 2:
                continue
            
            # Estimate power flow and losses
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    distance = self.electrical_distances[indices[i], indices[j]]
                    # Assume 1% loss per unit distance
                    flow = 1.0  # Normalized flow
                    loss = flow * distance * 0.01
                    total_loss += loss
                    total_flow += flow
        
        return total_loss / total_flow if total_flow > 0 else 0.0
    
    def _calculate_transformer_loading(self, clusters: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Calculate loading factor for each transformer.
        """
        transformer_loads = {}
        
        for t_id, t_buildings in self.constraints['transformer_groups'].items():
            # Get peak load for this transformer's buildings
            peak_load = 0
            for bid in t_buildings:
                bid_str = str(bid)
                if bid_str in self.time_series:
                    ts = self.time_series[bid_str]
                    if len(ts) > 0:
                        peak_load += np.max(ts[:, 3])  # Electricity demand
            
            # Get transformer capacity (default 630 kVA)
            capacity = self.constraints['transformer_capacity'].get(t_id, 630)
            
            # Calculate loading factor
            loading = peak_load / capacity if capacity > 0 else 1.0
            transformer_loads[t_id] = min(1.0, loading)  # Cap at 100%
        
        return transformer_loads
    
    def _calculate_diversity_index(self, clusters: Dict[str, List[str]]) -> float:
        """
        Calculate average diversity index across clusters.
        """
        if not clusters:
            return 0.0
        
        diversity_scores = []
        
        for cluster_id, building_ids in clusters.items():
            if not building_ids:
                continue
            
            # Get building features for cluster
            cluster_df = self.building_features[
                self.building_features['ogc_fid'].isin([str(b) for b in building_ids])
            ]
            
            if len(cluster_df) < 2:
                continue
            
            # Calculate Shannon entropy for building types
            type_counts = cluster_df['building_function'].value_counts()
            if len(type_counts) > 1:
                type_probs = type_counts / len(cluster_df)
                diversity = entropy(type_probs)
                max_entropy = np.log(len(type_counts))
                if max_entropy > 0:
                    diversity_scores.append(diversity / max_entropy)
        
        return np.mean(diversity_scores) if diversity_scores else 0.0
    
    def _calculate_fairness_index(self, clusters: Dict[str, List[str]]) -> float:
        """
        Calculate Jain's fairness index for benefit distribution.
        """
        if not clusters:
            return 0.0
        
        cluster_benefits = []
        
        for cluster_id, building_ids in clusters.items():
            if not building_ids:
                continue
            
            # Calculate average benefit per building in cluster
            # Benefit = peak reduction + self-sufficiency gain
            benefit = len(building_ids)  # Simplified: benefit proportional to cluster size
            cluster_benefits.append(benefit)
        
        if not cluster_benefits:
            return 0.0
        
        # Jain's fairness index
        benefits = np.array(cluster_benefits)
        numerator = np.sum(benefits) ** 2
        denominator = len(benefits) * np.sum(benefits ** 2)
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def _calculate_modularity(self, clusters: Dict[str, List[str]]) -> float:
        """
        Calculate modularity based on complementarity network.
        """
        if not clusters:
            return 0.0
        
        bid_to_idx = self.constraints['bid_to_idx']
        total_weight = np.sum(self.complementarity)
        
        if total_weight == 0:
            return 0.0
        
        modularity = 0
        
        for cluster_id, building_ids in clusters.items():
            if len(building_ids) < 2:
                continue
            
            # Get indices
            indices = []
            for bid in building_ids:
                bid_str = str(bid)
                if bid_str in bid_to_idx:
                    indices.append(bid_to_idx[bid_str])
            
            if len(indices) < 2:
                continue
            
            # Calculate modularity contribution
            for i in indices:
                for j in indices:
                    if i != j:
                        actual = self.complementarity[i, j]
                        expected = (np.sum(self.complementarity[i, :]) * 
                                  np.sum(self.complementarity[:, j])) / total_weight
                        modularity += actual - expected
        
        return modularity / total_weight if total_weight > 0 else 0.0
    
    def _calculate_conductance(self, clusters: Dict[str, List[str]]) -> float:
        """
        Calculate conductance (inter-cluster vs intra-cluster connections).
        Lower is better.
        """
        if not clusters or len(clusters) < 2:
            return 0.0
        
        bid_to_idx = self.constraints['bid_to_idx']
        
        # Create cluster assignment
        node_to_cluster = {}
        for cluster_id, building_ids in clusters.items():
            for bid in building_ids:
                bid_str = str(bid)
                if bid_str in bid_to_idx:
                    node_to_cluster[bid_to_idx[bid_str]] = cluster_id
        
        # Calculate inter and intra cluster edges
        inter_cluster = 0
        intra_cluster = 0
        
        for i in range(len(self.complementarity)):
            for j in range(i + 1, len(self.complementarity)):
                weight = self.complementarity[i, j]
                
                if i in node_to_cluster and j in node_to_cluster:
                    if node_to_cluster[i] == node_to_cluster[j]:
                        intra_cluster += weight
                    else:
                        inter_cluster += weight
        
        total = inter_cluster + intra_cluster
        return inter_cluster / total if total > 0 else 0.0
    
    def _calculate_coverage(self, clusters: Dict[str, List[str]]) -> float:
        """
        Calculate coverage (fraction of buildings clustered).
        """
        total_buildings = len(self.building_features)
        clustered_buildings = sum(len(c) for c in clusters.values())
        
        return clustered_buildings / total_buildings if total_buildings > 0 else 0.0
    
    def _calculate_stability(self, clusters: Dict[str, List[str]]) -> float:
        """
        Calculate stability metric (resistance to perturbation).
        Simplified: based on cluster size distribution.
        """
        if not clusters:
            return 0.0
        
        sizes = [len(c) for c in clusters.values()]
        if not sizes:
            return 0.0
        
        # Stability higher when clusters are of similar size
        mean_size = np.mean(sizes)
        std_size = np.std(sizes)
        
        # Coefficient of variation (lower is more stable)
        cv = std_size / mean_size if mean_size > 0 else 1.0
        
        # Convert to stability score (higher is better)
        stability = 1 / (1 + cv)
        
        return stability
    
    def _get_cluster_size_distribution(self, clusters: Dict[str, List[str]]) -> Dict[str, Any]:
        """Get statistics about cluster sizes."""
        sizes = [len(c) for c in clusters.values()]
        
        if not sizes:
            return {'min': 0, 'max': 0, 'mean': 0, 'std': 0}
        
        return {
            'min': int(np.min(sizes)),
            'max': int(np.max(sizes)),
            'mean': float(np.mean(sizes)),
            'std': float(np.std(sizes)),
            'distribution': dict(pd.Series(sizes).value_counts().sort_index())
        }
    
    def _calculate_per_cluster_metrics(self, clusters: Dict[str, List[str]]) -> Dict[str, Dict]:
        """Calculate detailed metrics for each cluster."""
        per_cluster = {}
        
        for cluster_id, building_ids in clusters.items():
            if not building_ids:
                continue
            
            cluster_metrics = {
                'size': len(building_ids),
                'building_ids': building_ids[:5],  # Sample for brevity
                'peak_reduction': 0,
                'self_sufficiency': 0,
                'avg_complementarity': 0,
                'has_prosumer': False,
                'asset_mix': {}
            }
            
            # Check for prosumers
            cluster_df = self.building_features[
                self.building_features['ogc_fid'].isin([str(b) for b in building_ids])
            ]
            
            if len(cluster_df) > 0:
                cluster_metrics['has_prosumer'] = (
                    cluster_df['has_solar'].any() or 
                    cluster_df['has_battery'].any()
                )
                
                cluster_metrics['asset_mix'] = {
                    'solar': int(cluster_df['has_solar'].sum()),
                    'battery': int(cluster_df['has_battery'].sum()),
                    'heat_pump': int(cluster_df['has_heat_pump'].sum())
                }
            
            per_cluster[cluster_id] = cluster_metrics
        
        return per_cluster
    
    def _create_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of key metrics."""
        return {
            'overall_score': self._calculate_overall_score(metrics),
            'key_findings': {
                'peak_reduction': f"{metrics['peak_reduction']:.1%}",
                'self_sufficiency': f"{metrics['self_sufficiency']:.1%}",
                'violations': metrics['constraint_violations'],
                'avg_cluster_size': metrics['cluster_sizes']['mean']
            },
            'strengths': self._identify_strengths(metrics),
            'weaknesses': self._identify_weaknesses(metrics)
        }
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """
        Calculate weighted overall score.
        """
        # Define weights for different metrics
        weights = {
            'peak_reduction': 0.25,
            'self_sufficiency': 0.25,
            'complementarity_score': 0.20,
            'constraint_violations': -0.15,  # Negative weight (penalty)
            'fairness_index': 0.10,
            'diversity_index': 0.05
        }
        
        score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                value = metrics[metric]
                if metric == 'constraint_violations':
                    # Normalize violations (0 violations = score of 1)
                    value = 1 / (1 + value)
                score += value * abs(weight)
        
        return min(1.0, max(0.0, score))
    
    def _identify_strengths(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify strengths of the clustering."""
        strengths = []
        
        if metrics['peak_reduction'] > 0.15:
            strengths.append(f"High peak reduction ({metrics['peak_reduction']:.1%})")
        
        if metrics['self_sufficiency'] > 0.30:
            strengths.append(f"Good self-sufficiency ({metrics['self_sufficiency']:.1%})")
        
        if metrics['constraint_violations'] == 0:
            strengths.append("No constraint violations")
        
        if metrics['fairness_index'] > 0.8:
            strengths.append("Fair benefit distribution")
        
        return strengths
    
    def _identify_weaknesses(self, metrics: Dict[str, Any]) -> List[str]:
        """Identify weaknesses of the clustering."""
        weaknesses = []
        
        if metrics['peak_reduction'] < 0.05:
            weaknesses.append("Low peak reduction")
        
        if metrics['constraint_violations'] > 5:
            weaknesses.append(f"High constraint violations ({metrics['constraint_violations']})")
        
        if metrics['coverage'] < 0.8:
            weaknesses.append(f"Low coverage ({metrics['coverage']:.1%})")
        
        if metrics['cluster_sizes']['std'] > metrics['cluster_sizes']['mean']:
            weaknesses.append("Unbalanced cluster sizes")
        
        return weaknesses