# training/evaluation_metrics.py
"""
Evaluation metrics for multi-task energy GNN
Focuses on relative improvements and physics validation
No ground truth required - uses self-supervised and comparative metrics
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime
import logging
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class MetricConfig:
    """Configuration for evaluation metrics"""
    # Clustering metrics
    min_cluster_size: int = 3
    max_cluster_size: int = 20
    target_self_sufficiency: float = 0.7
    target_peak_reduction: float = 0.3
    
    # Solar metrics
    min_solar_viability_score: float = 0.6
    target_solar_coverage: float = 0.3
    
    # Retrofit metrics
    target_energy_reduction: float = 0.25
    max_acceptable_payback: float = 15
    
    # Physics constraints
    transformer_safety_margin: float = 0.9  # Don't exceed 90% of capacity
    max_voltage_deviation: float = 0.05  # 5% voltage tolerance
    max_line_loss: float = 0.03  # 3% distribution losses
    
    # Computational efficiency
    max_inference_time: float = 1.0  # seconds per LV group
    
    # Baseline comparison
    compare_with_random: bool = True
    compare_with_kmeans: bool = True
    compare_with_greedy: bool = True


class EvaluationMetrics:
    """
    Comprehensive evaluation metrics for energy GNN tasks
    Designed to work without ground truth labels
    """
    
    def __init__(self, config: Optional[MetricConfig] = None):
        """
        Initialize evaluation metrics
        
        Args:
            config: Metric configuration
        """
        self.config = config or MetricConfig()
        self.baseline_cache = {}
        self.metric_history = defaultdict(list)
        
        logger.info("Initialized EvaluationMetrics for unsupervised evaluation")
    
    # ============================================
    # CLUSTERING METRICS (No Ground Truth Needed)
    # ============================================
    
    def evaluate_clustering(self,
                          clusters: List[List[int]],
                          graph_data: Dict,
                          temporal_data: Optional[pd.DataFrame] = None,
                          embeddings: Optional[torch.Tensor] = None) -> Dict:
        """
        Evaluate clustering quality without ground truth
        
        Args:
            clusters: List of clusters (building IDs)
            graph_data: Building and grid information
            temporal_data: Energy consumption profiles
            embeddings: Building embeddings from GNN
            
        Returns:
            Dictionary of clustering metrics
        """
        metrics = {}
        
        # 1. Structural Quality Metrics
        structural = self._evaluate_cluster_structure(clusters, graph_data)
        metrics.update(structural)
        
        # 2. Energy Performance Metrics
        if temporal_data is not None:
            energy = self._evaluate_energy_performance(clusters, graph_data, temporal_data)
            metrics.update(energy)
        
        # 3. Constraint Satisfaction
        constraints = self._evaluate_constraints(clusters, graph_data)
        metrics.update(constraints)
        
        # 4. Statistical Quality Metrics
        if embeddings is not None:
            statistical = self._evaluate_statistical_quality(clusters, embeddings, graph_data)
            metrics.update(statistical)
        
        # 5. Comparative Metrics (vs baselines)
        if self.config.compare_with_random:
            baseline_comparison = self._compare_with_baselines(
                clusters, graph_data, temporal_data
            )
            metrics.update(baseline_comparison)
        
        # 6. Stability Metrics
        stability = self._evaluate_temporal_stability(clusters)
        metrics.update(stability)
        
        return metrics
    
    def _evaluate_cluster_structure(self,
                                   clusters: List[List[int]],
                                   graph_data: Dict) -> Dict:
        """Evaluate structural properties of clusters"""
        
        buildings = graph_data.get('buildings', {})
        
        # Cluster size distribution
        cluster_sizes = [len(c) for c in clusters]
        
        # Coverage
        total_buildings = len(buildings)
        clustered_buildings = sum(cluster_sizes)
        
        # Size compliance
        valid_sizes = sum(
            1 for size in cluster_sizes 
            if self.config.min_cluster_size <= size <= self.config.max_cluster_size
        )
        
        # Spatial cohesion
        spatial_cohesion = self._calculate_spatial_cohesion(clusters, buildings)
        
        # LV group compliance
        lv_compliance = self._check_lv_compliance(clusters, buildings)
        
        return {
            'cluster_count': len(clusters),
            'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
            'cluster_size_std': np.std(cluster_sizes) if cluster_sizes else 0,
            'building_coverage': clustered_buildings / total_buildings if total_buildings > 0 else 0,
            'valid_size_ratio': valid_sizes / len(clusters) if clusters else 0,
            'spatial_cohesion': spatial_cohesion,
            'lv_compliance_ratio': lv_compliance,
            'singleton_ratio': sum(1 for s in cluster_sizes if s == 1) / len(clusters) if clusters else 0
        }
    
    def _evaluate_energy_performance(self,
                                    clusters: List[List[int]],
                                    graph_data: Dict,
                                    temporal_data: pd.DataFrame) -> Dict:
        """Evaluate energy-related performance metrics"""
        
        buildings = graph_data.get('buildings', {})
        
        # Initialize aggregated metrics
        total_self_sufficiency = 0
        total_peak_reduction = 0
        total_complementarity = 0
        valid_clusters = 0
        
        for cluster in clusters:
            if len(cluster) < self.config.min_cluster_size:
                continue
            
            valid_clusters += 1
            
            # Self-sufficiency ratio
            ssr = self._calculate_self_sufficiency(cluster, buildings, temporal_data)
            total_self_sufficiency += ssr
            
            # Peak reduction
            peak_red = self._calculate_peak_reduction(cluster, buildings, temporal_data)
            total_peak_reduction += peak_red
            
            # Complementarity score
            comp = self._calculate_complementarity_score(cluster, temporal_data)
            total_complementarity += comp
        
        # Load factor improvement
        load_factor_improvement = self._calculate_load_factor_improvement(
            clusters, temporal_data
        )
        
        # Grid exchange reduction
        grid_exchange_reduction = self._calculate_grid_exchange_reduction(
            clusters, buildings, temporal_data
        )
        
        return {
            'avg_self_sufficiency': total_self_sufficiency / valid_clusters if valid_clusters > 0 else 0,
            'avg_peak_reduction': total_peak_reduction / valid_clusters if valid_clusters > 0 else 0,
            'avg_complementarity': total_complementarity / valid_clusters if valid_clusters > 0 else 0,
            'load_factor_improvement': load_factor_improvement,
            'grid_exchange_reduction': grid_exchange_reduction,
            'clusters_above_ssr_target': sum(
                1 for c in clusters 
                if self._calculate_self_sufficiency(c, buildings, temporal_data) > self.config.target_self_sufficiency
            ) / len(clusters) if clusters else 0
        }
    
    def _calculate_self_sufficiency(self,
                                   cluster: List[int],
                                   buildings: Dict,
                                   temporal_data: pd.DataFrame) -> float:
        """Calculate self-sufficiency ratio for a cluster"""
        
        if not cluster or temporal_data is None:
            return 0.0
        
        try:
            # Get cluster consumption profile
            cluster_consumption = pd.Series(0, index=temporal_data.index)
            cluster_generation = pd.Series(0, index=temporal_data.index)
            
            for building_id in cluster:
                # Consumption
                if building_id in temporal_data.columns:
                    cluster_consumption += temporal_data[building_id]
                
                # Generation (if has solar)
                building = buildings.get(building_id, {})
                if building.get('has_solar', False):
                    # Simplified solar generation profile
                    solar_capacity = building.get('solar_capacity_kw', 10)
                    hours = temporal_data.index.hour
                    solar_profile = np.where(
                        (hours >= 6) & (hours <= 18),
                        solar_capacity * 0.7 * np.sin((hours - 6) * np.pi / 12),
                        0
                    )
                    cluster_generation += solar_profile
            
            # Calculate self-sufficiency
            internal_use = np.minimum(cluster_generation, cluster_consumption)
            
            if cluster_consumption.sum() > 0:
                ssr = internal_use.sum() / cluster_consumption.sum()
            else:
                ssr = 0.0
            
            return min(1.0, ssr)
            
        except Exception as e:
            logger.warning(f"Error calculating self-sufficiency: {e}")
            return 0.0
    
    def _calculate_peak_reduction(self,
                                 cluster: List[int],
                                 buildings: Dict,
                                 temporal_data: pd.DataFrame) -> float:
        """Calculate peak reduction from clustering"""
        
        if not cluster or temporal_data is None:
            return 0.0
        
        try:
            # Individual peaks
            individual_peaks = []
            cluster_profile = pd.Series(0, index=temporal_data.index)
            
            for building_id in cluster:
                if building_id in temporal_data.columns:
                    building_profile = temporal_data[building_id]
                    individual_peaks.append(building_profile.max())
                    cluster_profile += building_profile
                else:
                    # Use average if not in temporal data
                    building = buildings.get(building_id, {})
                    individual_peaks.append(building.get('peak_demand_kw', 10))
            
            # Calculate reduction
            sum_individual_peaks = sum(individual_peaks)
            cluster_peak = cluster_profile.max() if len(cluster_profile) > 0 else sum_individual_peaks
            
            if sum_individual_peaks > 0:
                reduction = (sum_individual_peaks - cluster_peak) / sum_individual_peaks
                return max(0, min(1, reduction))
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating peak reduction: {e}")
            return 0.0
    
    def _calculate_complementarity_score(self,
                                        cluster: List[int],
                                        temporal_data: pd.DataFrame) -> float:
        """Calculate average complementarity within cluster"""
        
        if not cluster or len(cluster) < 2 or temporal_data is None:
            return 0.0
        
        try:
            # Calculate pairwise correlations
            correlations = []
            
            for i, building_i in enumerate(cluster):
                for j, building_j in enumerate(cluster):
                    if i >= j:  # Skip diagonal and duplicate pairs
                        continue
                    
                    if building_i in temporal_data.columns and building_j in temporal_data.columns:
                        corr = temporal_data[building_i].corr(temporal_data[building_j])
                        # Negative correlation is good (complementary)
                        if corr < 0:
                            correlations.append(-corr)  # Convert to positive score
                        else:
                            correlations.append(0)  # No complementarity
            
            return np.mean(correlations) if correlations else 0.0
            
        except Exception as e:
            logger.warning(f"Error calculating complementarity: {e}")
            return 0.0
    
    def _evaluate_constraints(self,
                             clusters: List[List[int]],
                             graph_data: Dict) -> Dict:
        """Evaluate physics and grid constraints"""
        
        buildings = graph_data.get('buildings', {})
        transformers = graph_data.get('transformers', {})
        
        # Transformer capacity violations
        transformer_violations = 0
        total_lv_groups = 0
        
        # Group clusters by LV group
        lv_groups = defaultdict(list)
        for cluster in clusters:
            for building_id in cluster:
                building = buildings.get(building_id, {})
                lv_group = building.get('lv_group_id')
                if lv_group:
                    lv_groups[lv_group].append(building_id)
        
        # Check each LV group
        for lv_group, group_buildings in lv_groups.items():
            total_lv_groups += 1
            
            # Calculate peak for this LV group
            peak_demand = sum(
                buildings.get(b_id, {}).get('peak_demand_kw', 10)
                for b_id in group_buildings
            )
            
            # Check transformer capacity
            transformer = transformers.get(lv_group, {})
            capacity_kva = transformer.get('capacity_kva', 250)
            capacity_kw = capacity_kva * 0.9  # Power factor assumption
            
            if peak_demand > capacity_kw * self.config.transformer_safety_margin:
                transformer_violations += 1
        
        # Size constraint violations
        size_violations = sum(
            1 for c in clusters
            if len(c) < self.config.min_cluster_size or len(c) > self.config.max_cluster_size
        )
        
        return {
            'transformer_violation_ratio': transformer_violations / total_lv_groups if total_lv_groups > 0 else 0,
            'size_violation_ratio': size_violations / len(clusters) if clusters else 0,
            'max_peak_to_capacity_ratio': self._calculate_max_peak_ratio(lv_groups, buildings, transformers),
            'energy_balance_error': 0.0  # Would need actual measurements
        }
    
    def _evaluate_statistical_quality(self,
                                     clusters: List[List[int]],
                                     embeddings: torch.Tensor,
                                     graph_data: Dict) -> Dict:
        """Evaluate statistical quality metrics"""
        
        if embeddings is None or len(clusters) == 0:
            return {}
        
        # Create cluster labels
        labels = np.zeros(len(embeddings))
        for cluster_id, cluster in enumerate(clusters):
            for building_idx in cluster:
                if building_idx < len(labels):
                    labels[building_idx] = cluster_id
        
        # Only evaluate for clustered buildings
        clustered_mask = labels >= 0
        
        if clustered_mask.sum() < 2:
            return {}
        
        embeddings_np = embeddings.cpu().numpy()
        
        try:
            # Silhouette score (cluster separation)
            silhouette = silhouette_score(
                embeddings_np[clustered_mask],
                labels[clustered_mask]
            ) if clustered_mask.sum() > 1 else 0
            
            # Calinski-Harabasz score (between-cluster vs within-cluster variance)
            ch_score = calinski_harabasz_score(
                embeddings_np[clustered_mask],
                labels[clustered_mask]
            ) if clustered_mask.sum() > len(clusters) else 0
            
            # Modularity (for graph-based clustering)
            modularity = self._calculate_modularity(clusters, graph_data)
            
            return {
                'silhouette_score': silhouette,
                'calinski_harabasz_score': ch_score / 1000,  # Normalize
                'modularity': modularity
            }
            
        except Exception as e:
            logger.warning(f"Error calculating statistical metrics: {e}")
            return {}
    
    def _calculate_modularity(self,
                             clusters: List[List[int]],
                             graph_data: Dict) -> float:
        """Calculate graph modularity"""
        
        # Build graph from building connections
        G = nx.Graph()
        buildings = graph_data.get('buildings', {})
        
        # Add nodes
        for building_id in buildings.keys():
            G.add_node(building_id)
        
        # Add edges based on shared LV group or adjacency
        for b1_id, b1_data in buildings.items():
            for b2_id, b2_data in buildings.items():
                if b1_id >= b2_id:
                    continue
                
                # Same LV group
                if b1_data.get('lv_group_id') == b2_data.get('lv_group_id'):
                    G.add_edge(b1_id, b2_id, weight=1.0)
                
                # Adjacent buildings
                if b2_id in b1_data.get('shared_walls', []):
                    if G.has_edge(b1_id, b2_id):
                        G[b1_id][b2_id]['weight'] += 0.5
                    else:
                        G.add_edge(b1_id, b2_id, weight=0.5)
        
        # Create partition for modularity
        partition = {}
        for cluster_id, cluster in enumerate(clusters):
            for building_id in cluster:
                partition[building_id] = cluster_id
        
        # Calculate modularity
        try:
            from networkx.algorithms.community import modularity
            communities = defaultdict(list)
            for node, comm in partition.items():
                communities[comm].append(node)
            
            mod = modularity(G, communities.values())
            return mod
        except:
            return 0.0
    
    def _compare_with_baselines(self,
                               clusters: List[List[int]],
                               graph_data: Dict,
                               temporal_data: Optional[pd.DataFrame]) -> Dict:
        """Compare with baseline methods"""
        
        # Generate baseline clusters if not cached
        if 'random' not in self.baseline_cache:
            self.baseline_cache['random'] = self._generate_random_clusters(
                graph_data, len(clusters)
            )
        
        if 'kmeans' not in self.baseline_cache and temporal_data is not None:
            self.baseline_cache['kmeans'] = self._generate_kmeans_clusters(
                graph_data, temporal_data, len(clusters)
            )
        
        metrics = {}
        
        # Compare with random
        if 'random' in self.baseline_cache:
            random_clusters = self.baseline_cache['random']
            
            # Energy performance comparison
            if temporal_data is not None:
                gnn_ssr = np.mean([
                    self._calculate_self_sufficiency(c, graph_data['buildings'], temporal_data)
                    for c in clusters
                ])
                random_ssr = np.mean([
                    self._calculate_self_sufficiency(c, graph_data['buildings'], temporal_data)
                    for c in random_clusters
                ])
                
                metrics['improvement_vs_random_ssr'] = (gnn_ssr - random_ssr) / (random_ssr + 1e-6)
                
                gnn_peak = np.mean([
                    self._calculate_peak_reduction(c, graph_data['buildings'], temporal_data)
                    for c in clusters
                ])
                random_peak = np.mean([
                    self._calculate_peak_reduction(c, graph_data['buildings'], temporal_data)
                    for c in random_clusters
                ])
                
                metrics['improvement_vs_random_peak'] = (gnn_peak - random_peak) / (random_peak + 1e-6)
        
        # Compare with k-means
        if 'kmeans' in self.baseline_cache and temporal_data is not None:
            kmeans_clusters = self.baseline_cache['kmeans']
            
            gnn_comp = np.mean([
                self._calculate_complementarity_score(c, temporal_data)
                for c in clusters
            ])
            kmeans_comp = np.mean([
                self._calculate_complementarity_score(c, temporal_data)
                for c in kmeans_clusters
            ])
            
            metrics['improvement_vs_kmeans_complementarity'] = (gnn_comp - kmeans_comp) / (kmeans_comp + 1e-6)
        
        return metrics
    
    def _evaluate_temporal_stability(self, clusters: List[List[int]]) -> Dict:
        """Evaluate temporal stability of clusters"""
        
        # Store current clusters in history
        self.metric_history['clusters'].append(clusters)
        
        if len(self.metric_history['clusters']) < 2:
            return {'temporal_stability': 1.0}
        
        # Compare with previous clustering
        prev_clusters = self.metric_history['clusters'][-2]
        
        # Calculate Jaccard similarity between clusterings
        stability_scores = []
        
        for curr_cluster in clusters:
            best_similarity = 0
            for prev_cluster in prev_clusters:
                intersection = len(set(curr_cluster) & set(prev_cluster))
                union = len(set(curr_cluster) | set(prev_cluster))
                if union > 0:
                    similarity = intersection / union
                    best_similarity = max(best_similarity, similarity)
            stability_scores.append(best_similarity)
        
        return {
            'temporal_stability': np.mean(stability_scores) if stability_scores else 1.0,
            'cluster_change_ratio': 1.0 - np.mean(stability_scores) if stability_scores else 0.0
        }
    
    # ============================================
    # SOLAR OPTIMIZATION METRICS
    # ============================================
    
    def evaluate_solar_optimization(self,
                                   solar_candidates: List,
                                   graph_data: Dict,
                                   clustering_results: Dict) -> Dict:
        """
        Evaluate solar optimization without ground truth ROI
        
        Uses ranking quality and physical feasibility metrics
        """
        
        if not solar_candidates:
            return {}
        
        buildings = graph_data.get('buildings', {})
        
        # 1. Coverage metrics
        coverage = self._evaluate_solar_coverage(solar_candidates, buildings)
        
        # 2. Ranking quality (using domain knowledge)
        ranking_quality = self._evaluate_solar_ranking(solar_candidates, buildings)
        
        # 3. Cluster coordination
        cluster_coordination = self._evaluate_solar_cluster_coordination(
            solar_candidates, clustering_results
        )
        
        # 4. Physical feasibility
        feasibility = self._evaluate_solar_feasibility(solar_candidates, buildings)
        
        # 5. Expected impact
        impact = self._evaluate_solar_impact(solar_candidates, buildings)
        
        metrics = {}
        metrics.update(coverage)
        metrics.update(ranking_quality)
        metrics.update(cluster_coordination)
        metrics.update(feasibility)
        metrics.update(impact)
        
        return metrics
    
    def _evaluate_solar_coverage(self,
                                candidates: List,
                                buildings: Dict) -> Dict:
        """Evaluate solar coverage metrics"""
        
        total_buildings = len(buildings)
        buildings_with_solar = sum(1 for b in buildings.values() if b.get('has_solar', False))
        new_solar = len(candidates)
        
        # Coverage by building type
        type_distribution = defaultdict(int)
        for candidate in candidates:
            building = buildings.get(candidate.building_id, {})
            b_type = building.get('building_function', 'unknown')
            type_distribution[b_type] += 1
        
        return {
            'solar_coverage_current': buildings_with_solar / total_buildings if total_buildings > 0 else 0,
            'solar_coverage_planned': (buildings_with_solar + new_solar) / total_buildings if total_buildings > 0 else 0,
            'solar_building_diversity': len(type_distribution) / len(candidates) if candidates else 0,
            'solar_candidates_count': new_solar
        }
    
    def _evaluate_solar_ranking(self,
                               candidates: List,
                               buildings: Dict) -> Dict:
        """Evaluate ranking quality using domain heuristics"""
        
        # Domain heuristic: good solar buildings have:
        # - Large roof area
        # - Good orientation (south)
        # - No existing solar
        # - High consumption
        
        heuristic_scores = []
        gnn_scores = []
        
        for i, candidate in enumerate(candidates):
            building = buildings.get(candidate.building_id, {})
            
            # Heuristic score
            h_score = 0
            h_score += min(1.0, building.get('suitable_roof_area', 0) / 200)  # Roof area
            h_score += 1.0 if building.get('building_orientation_cardinal', '') in ['south', 'flat'] else 0.5
            h_score += 0.0 if building.get('has_solar', False) else 1.0
            h_score += min(1.0, building.get('avg_demand_kw', 0) / 20)  # Consumption
            heuristic_scores.append(h_score / 4)  # Normalize
            
            # GNN score (ranking position)
            gnn_scores.append(1.0 - i / len(candidates))  # Higher rank = higher score
        
        # Calculate ranking correlation
        if len(heuristic_scores) > 1:
            correlation, _ = spearmanr(heuristic_scores, gnn_scores)
        else:
            correlation = 0.0
        
        # Check if top candidates make sense
        top_10_quality = np.mean(heuristic_scores[:10]) if len(heuristic_scores) >= 10 else np.mean(heuristic_scores)
        
        return {
            'solar_ranking_correlation': correlation,
            'solar_top10_quality': top_10_quality,
            'solar_ranking_monotonicity': self._check_monotonicity([c.total_score for c in candidates])
        }
    
    def _evaluate_solar_cluster_coordination(self,
                                            candidates: List,
                                            clustering_results: Dict) -> Dict:
        """Evaluate how well solar placement coordinates with clusters"""
        
        clusters = clustering_results.get('clusters', [])
        
        # Count solar distribution across clusters
        solar_per_cluster = defaultdict(int)
        buildings_per_cluster = defaultdict(int)
        
        for cluster_id, cluster in enumerate(clusters):
            buildings_per_cluster[cluster_id] = len(cluster)
            for candidate in candidates:
                if candidate.building_id in cluster:
                    solar_per_cluster[cluster_id] += 1
        
        # Calculate balance (solar should be distributed, not concentrated)
        if buildings_per_cluster:
            solar_ratios = [
                solar_per_cluster[cid] / buildings_per_cluster[cid]
                for cid in buildings_per_cluster
            ]
            solar_balance = 1.0 - np.std(solar_ratios) if solar_ratios else 0.0
        else:
            solar_balance = 0.0
        
        # Check complementarity focus
        complementarity_scores = [c.complementarity_score for c in candidates if hasattr(c, 'complementarity_score')]
        avg_complementarity = np.mean(complementarity_scores) if complementarity_scores else 0.0
        
        return {
            'solar_cluster_balance': solar_balance,
            'solar_complementarity_focus': avg_complementarity,
            'clusters_with_solar': len(solar_per_cluster) / len(clusters) if clusters else 0
        }
    
    # ============================================
    # RETROFIT TARGETING METRICS
    # ============================================
    
    def evaluate_retrofit_targeting(self,
                                   retrofit_candidates: List,
                                   graph_data: Dict,
                                   clustering_results: Dict) -> Dict:
        """
        Evaluate retrofit targeting without ground truth costs
        
        Uses urgency ranking and expected impact metrics
        """
        
        if not retrofit_candidates:
            return {}
        
        buildings = graph_data.get('buildings', {})
        
        # 1. Targeting accuracy (worst buildings first)
        targeting = self._evaluate_retrofit_targeting_accuracy(retrofit_candidates, buildings)
        
        # 2. Expected impact
        impact = self._evaluate_retrofit_impact(retrofit_candidates, buildings)
        
        # 3. Cluster coordination
        coordination = self._evaluate_retrofit_coordination(
            retrofit_candidates, clustering_results
        )
        
        # 4. Feasibility
        feasibility = self._evaluate_retrofit_feasibility(retrofit_candidates)
        
        metrics = {}
        metrics.update(targeting)
        metrics.update(impact)
        metrics.update(coordination)
        metrics.update(feasibility)
        
        return metrics
    
    def _evaluate_retrofit_targeting_accuracy(self,
                                             candidates: List,
                                             buildings: Dict) -> Dict:
        """Check if worst buildings are targeted first"""
        
        # Energy labels in order (worst to best)
        label_order = ['G', 'F', 'E', 'D', 'C', 'B', 'A']
        
        # Check if candidates are ordered by urgency
        candidate_labels = []
        for candidate in candidates:
            candidate_labels.append(candidate.current_label)
        
        # Calculate how well ordered (worst first)
        inversions = 0
        for i in range(len(candidate_labels)):
            for j in range(i + 1, len(candidate_labels)):
                if label_order.index(candidate_labels[i]) > label_order.index(candidate_labels[j]):
                    inversions += 1
        
        max_inversions = len(candidates) * (len(candidates) - 1) / 2
        targeting_accuracy = 1.0 - (inversions / max_inversions) if max_inversions > 0 else 1.0
        
        # Check if worst labels are prioritized
        worst_labels_ratio = sum(
            1 for c in candidates[:10] if c.current_label in ['F', 'G']
        ) / min(10, len(candidates)) if candidates else 0
        
        return {
            'retrofit_targeting_accuracy': targeting_accuracy,
            'retrofit_worst_first_ratio': worst_labels_ratio,
            'retrofit_urgency_distribution': self._get_label_distribution(candidates)
        }
    
    def _evaluate_retrofit_impact(self,
                                 candidates: List,
                                 buildings: Dict) -> Dict:
        """Evaluate expected retrofit impact"""
        
        total_energy_reduction = sum(c.annual_energy_savings_kwh for c in candidates)
        total_co2_reduction = sum(c.co2_reduction_tons_annual for c in candidates)
        
        # System-wide impact
        system_consumption = sum(
            b.get('avg_demand_kw', 10) * 8760 for b in buildings.values()
        )
        
        # Average improvements
        avg_intensity_reduction = np.mean([
            c.potential_reduction_percent for c in candidates
        ]) if candidates else 0
        
        return {
            'retrofit_total_energy_reduction_kwh': total_energy_reduction,
            'retrofit_system_reduction_percent': total_energy_reduction / system_consumption if system_consumption > 0 else 0,
            'retrofit_avg_intensity_reduction': avg_intensity_reduction,
            'retrofit_total_co2_reduction_tons': total_co2_reduction
        }
    
    # ============================================
    # COMPUTATIONAL EFFICIENCY METRICS
    # ============================================
    
    def evaluate_computational_efficiency(self,
                                         inference_time: float,
                                         model_params: int,
                                         baseline_time: Optional[float] = None) -> Dict:
        """Evaluate computational efficiency"""
        
        metrics = {
            'inference_time_seconds': inference_time,
            'model_parameters': model_params,
            'inference_within_target': inference_time <= self.config.max_inference_time
        }
        
        if baseline_time:
            metrics['speedup_vs_baseline'] = baseline_time / inference_time if inference_time > 0 else 0
        
        return metrics
    
    # ============================================
    # SYSTEM-WIDE METRICS
    # ============================================
    
    def evaluate_system_performance(self,
                                   clustering_metrics: Dict,
                                   solar_metrics: Dict,
                                   retrofit_metrics: Dict) -> Dict:
        """Combine all metrics for system-wide evaluation"""
        
        # Key performance indicators
        kpis = {
            'overall_self_sufficiency': clustering_metrics.get('avg_self_sufficiency', 0),
            'overall_peak_reduction': clustering_metrics.get('avg_peak_reduction', 0),
            'overall_solar_coverage': solar_metrics.get('solar_coverage_planned', 0),
            'overall_energy_reduction': retrofit_metrics.get('retrofit_system_reduction_percent', 0),
            'overall_constraint_satisfaction': 1.0 - clustering_metrics.get('transformer_violation_ratio', 0)
        }
        
        # Weighted score (configurable weights)
        weights = {
            'self_sufficiency': 0.25,
            'peak_reduction': 0.25,
            'solar_coverage': 0.20,
            'energy_reduction': 0.20,
            'constraints': 0.10
        }
        
        overall_score = sum(
            weights[key.replace('overall_', '')] * value
            for key, value in kpis.items()
            if key.replace('overall_', '') in weights
        )
        
        kpis['overall_score'] = overall_score
        
        return kpis
    
    # ============================================
    # HELPER METHODS
    # ============================================
    
    def _calculate_spatial_cohesion(self,
                                   clusters: List[List[int]],
                                   buildings: Dict) -> float:
        """Calculate spatial cohesion of clusters"""
        
        cohesion_scores = []
        
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            # Calculate average pairwise distance
            distances = []
            for i, b1_id in enumerate(cluster):
                for j, b2_id in enumerate(cluster):
                    if i >= j:
                        continue
                    
                    b1 = buildings.get(b1_id, {})
                    b2 = buildings.get(b2_id, {})
                    
                    x1, y1 = b1.get('x_coord', 0), b1.get('y_coord', 0)
                    x2, y2 = b2.get('x_coord', 0), b2.get('y_coord', 0)
                    
                    dist = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                    distances.append(dist)
            
            if distances:
                avg_distance = np.mean(distances)
                # Convert to cohesion score (inverse of distance, normalized)
                cohesion = 1.0 / (1.0 + avg_distance / 100)  # 100m normalization
                cohesion_scores.append(cohesion)
        
        return np.mean(cohesion_scores) if cohesion_scores else 0.0
    
    def _check_lv_compliance(self,
                            clusters: List[List[int]],
                            buildings: Dict) -> float:
        """Check if clusters respect LV boundaries"""
        
        compliant_clusters = 0
        
        for cluster in clusters:
            lv_groups = set()
            for building_id in cluster:
                building = buildings.get(building_id, {})
                lv_group = building.get('lv_group_id')
                if lv_group:
                    lv_groups.add(lv_group)
            
            # Cluster should only have one LV group
            if len(lv_groups) <= 1:
                compliant_clusters += 1
        
        return compliant_clusters / len(clusters) if clusters else 0.0
    
    def _calculate_load_factor_improvement(self,
                                          clusters: List[List[int]],
                                          temporal_data: Optional[pd.DataFrame]) -> float:
        """Calculate load factor improvement from clustering"""
        
        if temporal_data is None:
            return 0.0
        
        # Calculate system load factor before and after
        total_profile_before = temporal_data.sum(axis=1)
        avg_before = total_profile_before.mean()
        peak_before = total_profile_before.max()
        lf_before = avg_before / peak_before if peak_before > 0 else 0
        
        # After clustering (simplified - assumes perfect coordination)
        lf_after = min(1.0, lf_before * 1.2)  # Assume 20% improvement max
        
        return (lf_after - lf_before) / lf_before if lf_before > 0 else 0
    
    def _calculate_grid_exchange_reduction(self,
                                          clusters: List[List[int]],
                                          buildings: Dict,
                                          temporal_data: Optional[pd.DataFrame]) -> float:
        """Calculate reduction in grid exchange from local sharing"""
        
        if temporal_data is None:
            return 0.0
        
        # This is simplified - in practice would need detailed simulation
        # Assume clustering reduces grid exchange by enabling local sharing
        
        avg_cluster_size = np.mean([len(c) for c in clusters]) if clusters else 1
        
        # Larger clusters = more sharing potential
        reduction = min(0.3, avg_cluster_size * 0.02)  # 2% per building, max 30%
        
        return reduction
    
    def _calculate_max_peak_ratio(self,
                                 lv_groups: Dict,
                                 buildings: Dict,
                                 transformers: Dict) -> float:
        """Calculate maximum peak to capacity ratio"""
        
        max_ratio = 0.0
        
        for lv_group, group_buildings in lv_groups.items():
            peak = sum(
                buildings.get(b_id, {}).get('peak_demand_kw', 10)
                for b_id in group_buildings
            )
            
            transformer = transformers.get(lv_group, {})
            capacity = transformer.get('capacity_kva', 250) * 0.9  # Power factor
            
            if capacity > 0:
                ratio = peak / capacity
                max_ratio = max(max_ratio, ratio)
        
        return max_ratio
    
    def _generate_random_clusters(self,
                                 graph_data: Dict,
                                 num_clusters: int) -> List[List[int]]:
        """Generate random baseline clusters"""
        
        buildings = list(graph_data.get('buildings', {}).keys())
        np.random.shuffle(buildings)
        
        # Create roughly equal-sized clusters
        clusters = []
        cluster_size = len(buildings) // num_clusters
        
        for i in range(num_clusters):
            start = i * cluster_size
            end = start + cluster_size if i < num_clusters - 1 else len(buildings)
            cluster = buildings[start:end]
            if cluster:
                clusters.append(cluster)
        
        return clusters
    
    def _generate_kmeans_clusters(self,
                                 graph_data: Dict,
                                 temporal_data: pd.DataFrame,
                                 num_clusters: int) -> List[List[int]]:
        """Generate k-means baseline clusters"""
        
        from sklearn.cluster import KMeans
        
        buildings = list(graph_data.get('buildings', {}).keys())
        
        # Create feature matrix from temporal data
        features = []
        valid_buildings = []
        
        for building_id in buildings:
            if building_id in temporal_data.columns:
                features.append(temporal_data[building_id].values)
                valid_buildings.append(building_id)
        
        if not features:
            return self._generate_random_clusters(graph_data, num_clusters)
        
        features = np.array(features)
        
        # Run k-means
        kmeans = KMeans(n_clusters=min(num_clusters, len(features)), random_state=42)
        labels = kmeans.fit_predict(features)
        
        # Create clusters
        clusters = defaultdict(list)
        for building_id, label in zip(valid_buildings, labels):
            clusters[label].append(building_id)
        
        return list(clusters.values())
    
    def _check_monotonicity(self, scores: List[float]) -> float:
        """Check if scores are monotonically decreasing"""
        
        if len(scores) < 2:
            return 1.0
        
        decreasing_pairs = sum(
            1 for i in range(len(scores) - 1)
            if scores[i] >= scores[i + 1]
        )
        
        return decreasing_pairs / (len(scores) - 1)
    
    def _get_label_distribution(self, candidates: List) -> Dict:
        """Get distribution of energy labels"""
        
        distribution = defaultdict(int)
        for candidate in candidates:
            distribution[candidate.current_label] += 1
        
        return dict(distribution)
    
    def _evaluate_solar_feasibility(self,
                                   candidates: List,
                                   buildings: Dict) -> Dict:
        """Check physical feasibility of solar installations"""
        
        feasible_count = 0
        
        for candidate in candidates:
            building = buildings.get(candidate.building_id, {})
            
            # Check feasibility criteria
            roof_ok = building.get('suitable_roof_area', 0) >= 20
            no_existing = not building.get('has_solar', False)
            capacity_reasonable = candidate.recommended_capacity_kw <= 100
            
            if roof_ok and no_existing and capacity_reasonable:
                feasible_count += 1
        
        return {
            'solar_feasibility_ratio': feasible_count / len(candidates) if candidates else 0
        }
    
    def _evaluate_solar_impact(self,
                              candidates: List,
                              buildings: Dict) -> Dict:
        """Evaluate expected solar impact"""
        
        total_capacity = sum(c.recommended_capacity_kw for c in candidates)
        total_generation = sum(c.expected_generation_annual_kwh for c in candidates)
        
        # System consumption
        system_consumption = sum(
            b.get('avg_demand_kw', 10) * 8760 for b in buildings.values()
        )
        
        return {
            'solar_total_capacity_kw': total_capacity,
            'solar_renewable_fraction': total_generation / system_consumption if system_consumption > 0 else 0
        }
    
    def _evaluate_retrofit_coordination(self,
                                       candidates: List,
                                       clustering_results: Dict) -> Dict:
        """Evaluate retrofit coordination with clusters"""
        
        clusters = clustering_results.get('clusters', [])
        
        # Count retrofits per cluster
        retrofits_per_cluster = defaultdict(int)
        
        for candidate in candidates:
            for cluster_id, cluster in enumerate(clusters):
                if candidate.building_id in cluster:
                    retrofits_per_cluster[cluster_id] += 1
                    break
        
        # Check if retrofits are well-distributed
        if retrofits_per_cluster:
            distribution_score = 1.0 - np.std(list(retrofits_per_cluster.values())) / (np.mean(list(retrofits_per_cluster.values())) + 1e-6)
        else:
            distribution_score = 0.0
        
        return {
            'retrofit_cluster_distribution': distribution_score,
            'clusters_with_retrofits': len(retrofits_per_cluster) / len(clusters) if clusters else 0
        }
    
    def _evaluate_retrofit_feasibility(self, candidates: List) -> Dict:
        """Check feasibility of retrofit recommendations"""
        
        if not candidates:
            return {}
        
        # Check payback periods
        acceptable_payback = sum(
            1 for c in candidates
            if c.simple_payback_years <= self.config.max_acceptable_payback
        )
        
        # Check reduction potential
        significant_reduction = sum(
            1 for c in candidates
            if c.potential_reduction_percent >= self.config.target_energy_reduction
        )
        
        return {
            'retrofit_acceptable_payback_ratio': acceptable_payback / len(candidates),
            'retrofit_significant_reduction_ratio': significant_reduction / len(candidates),
            'retrofit_avg_payback_years': np.mean([c.simple_payback_years for c in candidates])
        }
    
    def print_summary(self, metrics: Dict, name: str = "Evaluation"):
        """Print formatted summary of metrics"""
        
        print("\n" + "="*60)
        print(f"{name} Metrics Summary")
        print("="*60)
        
        # Group metrics by category
        categories = defaultdict(dict)
        
        for key, value in metrics.items():
            if 'cluster' in key or 'self_sufficiency' in key or 'peak' in key:
                categories['Clustering'][key] = value
            elif 'solar' in key:
                categories['Solar Optimization'][key] = value
            elif 'retrofit' in key:
                categories['Retrofit Targeting'][key] = value
            elif 'overall' in key:
                categories['System Performance'][key] = value
            else:
                categories['Other'][key] = value
        
        # Print by category
        for category, cat_metrics in categories.items():
            if cat_metrics:
                print(f"\n{category}:")
                print("-" * 40)
                for key, value in cat_metrics.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.3f}")
                    else:
                        print(f"  {key}: {value}")


# Utility function for testing
def test_evaluation_metrics():
    """Test evaluation metrics with dummy data"""
    
    # Create dummy data
    clusters = [
        [1, 2, 3, 4],
        [5, 6, 7],
        [8, 9, 10, 11, 12]
    ]
    
    graph_data = {
        'buildings': {
            i: {
                'lv_group_id': f'LV_{i//5}',
                'x_coord': np.random.rand() * 100,
                'y_coord': np.random.rand() * 100,
                'peak_demand_kw': 10 + np.random.rand() * 20,
                'avg_demand_kw': 5 + np.random.rand() * 10,
                'has_solar': np.random.rand() > 0.7,
                'suitable_roof_area': 50 + np.random.rand() * 150,
                'energy_label_simple': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G'])
            }
            for i in range(1, 13)
        },
        'transformers': {
            'LV_0': {'capacity_kva': 250},
            'LV_1': {'capacity_kva': 250},
            'LV_2': {'capacity_kva': 250}
        }
    }
    
    # Create evaluator
    evaluator = EvaluationMetrics()
    
    # Test clustering metrics
    clustering_metrics = evaluator.evaluate_clustering(
        clusters=clusters,
        graph_data=graph_data,
        temporal_data=None,
        embeddings=None
    )
    
    # Print results
    evaluator.print_summary(clustering_metrics, "Clustering")
    
    print("\n✅ Evaluation metrics implementation complete!")
    
    return clustering_metrics


if __name__ == "__main__":
    test_results = test_evaluation_metrics()