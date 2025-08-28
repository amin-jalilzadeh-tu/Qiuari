"""
Cluster Evaluation Module
Evaluates clustering performance without ground truth
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class ClusterEvaluator:
    """
    Evaluates energy community clusters using internal metrics
    No ground truth required - focuses on objective performance
    """
    
    def __init__(self, config: Dict):
        """
        Initialize evaluator with configuration
        
        Args:
            config: Evaluation configuration
        """
        self.config = config
        self.metrics_to_track = config.get('metrics', [])
        
        # History tracking
        self.evaluation_history = []
        self.best_scores = defaultdict(float)
        
        logger.info(f"Initialized ClusterEvaluator tracking {len(self.metrics_to_track)} metrics")
    
    def evaluate(
        self,
        clusters: torch.Tensor,
        temporal_data: pd.DataFrame,
        building_features: Dict,
        lv_group_ids: torch.Tensor,
        adjacency_matrix: Optional[np.ndarray] = None,
        previous_clusters: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Comprehensive cluster evaluation
        
        Args:
            clusters: Soft cluster assignments [N, K]
            temporal_data: Energy consumption/generation data
            building_features: Building characteristics
            lv_group_ids: LV group assignment for each building
            adjacency_matrix: Physical adjacency between buildings
            previous_clusters: Previous cluster assignments for stability
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Convert soft assignments to hard clusters
        hard_clusters = self._get_hard_clusters(clusters)
        
        # 1. Self-sufficiency metrics
        if 'avg_self_sufficiency' in self.metrics_to_track:
            metrics['avg_self_sufficiency'] = self._evaluate_self_sufficiency(
                hard_clusters, temporal_data
            )
        
        # 2. Complementarity metrics
        if 'avg_complementarity' in self.metrics_to_track:
            metrics['avg_complementarity'] = self._evaluate_complementarity(
                hard_clusters, temporal_data
            )
        
        # 3. Peak reduction metrics
        if 'avg_peak_reduction' in self.metrics_to_track:
            metrics['avg_peak_reduction'] = self._evaluate_peak_reduction(
                hard_clusters, temporal_data
            )
        
        # 4. Constraint violations
        if 'lv_boundary_violations' in self.metrics_to_track:
            metrics['lv_boundary_violations'] = self._check_lv_violations(
                hard_clusters, lv_group_ids
            )
        
        if 'size_violations' in self.metrics_to_track:
            metrics['size_violations'] = self._check_size_violations(hard_clusters)
        
        # 5. Cluster stability
        if 'cluster_stability' in self.metrics_to_track and previous_clusters is not None:
            metrics['cluster_stability'] = self._evaluate_stability(
                clusters, previous_clusters
            )
        
        # 6. Coverage metrics
        if 'orphan_buildings_ratio' in self.metrics_to_track:
            metrics['orphan_buildings_ratio'] = self._evaluate_coverage(clusters)
        
        # 7. Physical compactness
        if 'avg_compactness' in self.metrics_to_track:
            metrics['avg_compactness'] = self._evaluate_compactness(
                hard_clusters, building_features, adjacency_matrix
            )
        
        # 8. Overall cluster quality score
        if 'avg_cluster_quality_score' in self.metrics_to_track:
            metrics['avg_cluster_quality_score'] = self._calculate_overall_score(metrics)
        
        # 9. Energy balance
        metrics['energy_balance'] = self._evaluate_energy_balance(
            hard_clusters, temporal_data
        )
        
        # 10. Load diversity
        metrics['load_diversity'] = self._evaluate_load_diversity(
            hard_clusters, temporal_data
        )
        
        # Store evaluation
        evaluation = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'num_clusters': len(hard_clusters),
            'total_buildings': clusters.shape[0]
        }
        self.evaluation_history.append(evaluation)
        
        # Update best scores
        for metric, value in metrics.items():
            if metric not in ['lv_boundary_violations', 'size_violations', 'orphan_buildings_ratio']:
                # Higher is better for most metrics
                self.best_scores[metric] = max(self.best_scores[metric], value)
            else:
                # Lower is better for violations
                if metric not in self.best_scores:
                    self.best_scores[metric] = value
                else:
                    self.best_scores[metric] = min(self.best_scores[metric], value)
        
        return metrics
    
    def _get_hard_clusters(self, soft_clusters: torch.Tensor) -> Dict[int, List[int]]:
        """
        Convert soft assignments to hard clusters
        
        Args:
            soft_clusters: Soft assignments [N, K]
            
        Returns:
            Dictionary of cluster_id -> building indices
        """
        hard_assignments = torch.argmax(soft_clusters, dim=1)
        clusters = defaultdict(list)
        
        for building_idx, cluster_id in enumerate(hard_assignments):
            # Only include if assignment confidence is high enough
            if soft_clusters[building_idx, cluster_id] > 0.3:
                clusters[cluster_id.item()].append(building_idx)
        
        # Filter out too-small clusters
        valid_clusters = {
            cid: members for cid, members in clusters.items()
            if len(members) >= 3
        }
        
        return valid_clusters
    
    def _evaluate_self_sufficiency(
        self,
        clusters: Dict[int, List[int]],
        temporal_data: pd.DataFrame
    ) -> float:
        """
        Evaluate average self-sufficiency across clusters
        """
        if not clusters or temporal_data is None:
            return 0.0
        
        self_sufficiencies = []
        
        for cluster_id, members in clusters.items():
            if len(members) < 3:
                continue
                
            # Get cluster data
            cluster_data = temporal_data[temporal_data['building_id'].isin(members)]
            
            if cluster_data.empty:
                continue
            
            # Aggregate demand and generation
            hourly = cluster_data.groupby('timestamp').agg({
                'demand_kw': 'sum',
                'generation_kw': 'sum'
            })
            
            # Calculate self-consumed energy
            hourly['self_consumed'] = hourly[['demand_kw', 'generation_kw']].min(axis=1)
            
            # Self-sufficiency ratio
            total_demand = hourly['demand_kw'].sum()
            if total_demand > 0:
                self_sufficiency = hourly['self_consumed'].sum() / total_demand
                self_sufficiencies.append(min(self_sufficiency, 1.0))
        
        return np.mean(self_sufficiencies) if self_sufficiencies else 0.0
    
    def _evaluate_complementarity(
        self,
        clusters: Dict[int, List[int]],
        temporal_data: pd.DataFrame
    ) -> float:
        """
        Evaluate average complementarity (negative correlation) in clusters
        """
        if not clusters or temporal_data is None:
            return 0.0
        
        complementarities = []
        
        for cluster_id, members in clusters.items():
            if len(members) < 2:
                continue
            
            # Get demand profiles
            profiles = []
            for building_id in members:
                building_data = temporal_data[temporal_data['building_id'] == building_id]
                if not building_data.empty:
                    profile = building_data.groupby('hour')['demand_kw'].mean().values
                    if len(profile) == 24:
                        profiles.append(profile)
            
            if len(profiles) >= 2:
                # Calculate average correlation
                corr_matrix = np.corrcoef(profiles)
                upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                avg_correlation = np.mean(upper_triangle)
                
                # Negative correlation is good
                complementarities.append(-avg_correlation)
        
        return np.mean(complementarities) if complementarities else 0.0
    
    def _evaluate_peak_reduction(
        self,
        clusters: Dict[int, List[int]],
        temporal_data: pd.DataFrame
    ) -> float:
        """
        Evaluate average peak reduction from clustering
        """
        if not clusters or temporal_data is None:
            return 0.0
        
        peak_reductions = []
        
        for cluster_id, members in clusters.items():
            cluster_data = temporal_data[temporal_data['building_id'].isin(members)]
            
            if cluster_data.empty:
                continue
            
            # Individual peaks
            individual_peaks = []
            for building_id in members:
                building_data = cluster_data[cluster_data['building_id'] == building_id]
                if not building_data.empty:
                    peak = building_data['demand_kw'].max()
                    individual_peaks.append(peak)
            
            if individual_peaks:
                sum_peaks = sum(individual_peaks)
                
                # Cluster peak
                hourly = cluster_data.groupby('timestamp')['demand_kw'].sum()
                cluster_peak = hourly.max()
                
                if sum_peaks > 0:
                    reduction = 1 - (cluster_peak / sum_peaks)
                    peak_reductions.append(max(0, reduction))
        
        return np.mean(peak_reductions) if peak_reductions else 0.0
    
    def _check_lv_violations(
        self,
        clusters: Dict[int, List[int]],
        lv_group_ids: torch.Tensor
    ) -> int:
        """
        Count clusters that violate LV boundaries
        """
        violations = 0
        
        for cluster_id, members in clusters.items():
            # Get LV groups for cluster members
            member_lv_groups = set()
            for building_idx in members:
                if building_idx < len(lv_group_ids):
                    lv_group = lv_group_ids[building_idx].item()
                    member_lv_groups.add(lv_group)
            
            # Violation if cluster spans multiple LV groups
            if len(member_lv_groups) > 1:
                violations += 1
                logger.warning(f"Cluster {cluster_id} violates LV boundaries: {member_lv_groups}")
        
        return violations
    
    def _check_size_violations(self, clusters: Dict[int, List[int]]) -> int:
        """
        Count clusters that violate size constraints
        """
        min_size = 3
        max_size = 20
        violations = 0
        
        for cluster_id, members in clusters.items():
            size = len(members)
            if size < min_size or size > max_size:
                violations += 1
                logger.warning(f"Cluster {cluster_id} size violation: {size} buildings")
        
        return violations
    
    def _evaluate_stability(
        self,
        current_clusters: torch.Tensor,
        previous_clusters: torch.Tensor
    ) -> float:
        """
        Evaluate cluster stability (buildings staying in same clusters)
        """
        # Get hard assignments
        current_hard = torch.argmax(current_clusters, dim=1)
        previous_hard = torch.argmax(previous_clusters, dim=1)
        
        # Count buildings that stayed in same cluster
        stable = (current_hard == previous_hard).float().mean()
        
        return stable.item()
    
    def _evaluate_coverage(self, clusters: torch.Tensor) -> float:
        """
        Evaluate ratio of orphan buildings (not strongly assigned)
        """
        # Maximum assignment probability for each building
        max_probs = torch.max(clusters, dim=1)[0]
        
        # Buildings with low assignment confidence are orphans
        orphans = (max_probs < 0.3).float().mean()
        
        return orphans.item()
    
    def _evaluate_compactness(
        self,
        clusters: Dict[int, List[int]],
        building_features: Dict,
        adjacency_matrix: Optional[np.ndarray] = None
    ) -> float:
        """
        Evaluate physical compactness of clusters
        """
        if not clusters:
            return 0.0
        
        compactness_scores = []
        
        for cluster_id, members in clusters.items():
            if len(members) <= 1:
                compactness_scores.append(1.0)
                continue
            
            # Check adjacency
            if adjacency_matrix is not None:
                adjacency_count = 0
                for i, b1 in enumerate(members):
                    for b2 in members[i+1:]:
                        if b1 < adjacency_matrix.shape[0] and b2 < adjacency_matrix.shape[1]:
                            if adjacency_matrix[b1, b2] > 0:
                                adjacency_count += 1
                
                total_pairs = len(members) * (len(members) - 1) / 2
                if total_pairs > 0:
                    adjacency_ratio = adjacency_count / total_pairs
                    compactness_scores.append(min(1.0, adjacency_ratio * 2))
            else:
                # Use distance-based compactness
                positions = []
                for building_idx in members:
                    if building_idx in building_features:
                        feat = building_features[building_idx]
                        if 'x' in feat and 'y' in feat:
                            positions.append([feat['x'], feat['y']])
                
                if len(positions) >= 2:
                    positions = np.array(positions)
                    distances = []
                    for i in range(len(positions)):
                        for j in range(i+1, len(positions)):
                            dist = np.linalg.norm(positions[i] - positions[j])
                            distances.append(dist)
                    
                    avg_distance = np.mean(distances)
                    compactness = max(0, 1 - avg_distance / 500)
                    compactness_scores.append(compactness)
        
        return np.mean(compactness_scores) if compactness_scores else 0.0
    
    def _evaluate_energy_balance(
        self,
        clusters: Dict[int, List[int]],
        temporal_data: pd.DataFrame
    ) -> float:
        """
        Evaluate energy balance in clusters
        """
        if not clusters or temporal_data is None:
            return 0.5
        
        balance_scores = []
        
        for cluster_id, members in clusters.items():
            cluster_data = temporal_data[temporal_data['building_id'].isin(members)]
            
            if not cluster_data.empty:
                total_demand = cluster_data['demand_kw'].sum()
                total_generation = cluster_data['generation_kw'].sum()
                
                if total_demand > 0:
                    ratio = total_generation / total_demand
                    # Perfect balance at ratio=1
                    if ratio <= 1:
                        balance = ratio
                    else:
                        balance = max(0, 2 - ratio)
                    balance_scores.append(balance)
        
        return np.mean(balance_scores) if balance_scores else 0.5
    
    def _evaluate_load_diversity(
        self,
        clusters: Dict[int, List[int]],
        temporal_data: pd.DataFrame
    ) -> float:
        """
        Evaluate load diversity in clusters
        """
        if not clusters or temporal_data is None:
            return 0.5
        
        diversity_scores = []
        
        for cluster_id, members in clusters.items():
            if len(members) < 2:
                continue
            
            peak_hours = []
            for building_id in members:
                building_data = temporal_data[temporal_data['building_id'] == building_id]
                if not building_data.empty:
                    hourly = building_data.groupby('hour')['demand_kw'].mean()
                    if not hourly.empty:
                        peak_hour = hourly.idxmax()
                        peak_hours.append(peak_hour)
            
            if len(peak_hours) >= 2:
                unique_peaks = len(set(peak_hours))
                diversity = unique_peaks / len(peak_hours)
                diversity_scores.append(diversity)
        
        return np.mean(diversity_scores) if diversity_scores else 0.5
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall cluster quality score
        """
        # Weights for different metrics
        weights = {
            'avg_self_sufficiency': 0.3,
            'avg_complementarity': 0.3,
            'avg_peak_reduction': 0.2,
            'avg_compactness': 0.1,
            'energy_balance': 0.05,
            'load_diversity': 0.05
        }
        
        score = 0
        total_weight = 0
        
        for metric, weight in weights.items():
            if metric in metrics:
                # Handle negative metrics (complementarity)
                if metric == 'avg_complementarity':
                    # More negative = better
                    value = max(0, -metrics[metric]) if metrics[metric] < 0 else 0
                else:
                    value = metrics[metric]
                
                score += weight * value
                total_weight += weight
        
        # Penalize violations
        if 'lv_boundary_violations' in metrics and metrics['lv_boundary_violations'] > 0:
            score *= 0.1  # Heavy penalty
        
        if 'size_violations' in metrics and metrics['size_violations'] > 0:
            score *= 0.8  # Moderate penalty
        
        if total_weight > 0:
            score = score / total_weight
        
        return min(max(score, 0), 1)
    
    def compare_to_baseline(
        self,
        clusters: torch.Tensor,
        baseline_method: str,
        temporal_data: pd.DataFrame,
        building_features: Dict
    ) -> Dict[str, float]:
        """
        Compare clustering to baseline methods
        
        Args:
            clusters: Current clustering
            baseline_method: Baseline to compare against
            temporal_data: Energy data
            building_features: Building features
            
        Returns:
            Comparison metrics
        """
        # Implement baseline clustering
        if baseline_method == 'random':
            baseline_clusters = self._random_clustering(clusters.shape[0])
        elif baseline_method == 'geographic':
            baseline_clusters = self._geographic_clustering(building_features)
        elif baseline_method == 'consumption':
            baseline_clusters = self._consumption_clustering(temporal_data)
        else:
            return {}
        
        # Evaluate baseline
        baseline_metrics = self.evaluate(
            baseline_clusters,
            temporal_data,
            building_features,
            torch.zeros(clusters.shape[0], dtype=torch.long)
        )
        
        # Calculate improvement
        current_metrics = self.evaluation_history[-1]['metrics'] if self.evaluation_history else {}
        
        improvements = {}
        for metric in current_metrics:
            if metric in baseline_metrics:
                if metric in ['lv_boundary_violations', 'size_violations', 'orphan_buildings_ratio']:
                    # Lower is better
                    improvement = baseline_metrics[metric] - current_metrics[metric]
                else:
                    # Higher is better
                    improvement = current_metrics[metric] - baseline_metrics[metric]
                
                improvements[f'{metric}_improvement'] = improvement
        
        return improvements
    
    def _random_clustering(self, num_buildings: int) -> torch.Tensor:
        """Generate random clustering baseline"""
        num_clusters = min(20, num_buildings // 5)
        clusters = torch.zeros(num_buildings, num_clusters)
        
        for i in range(num_buildings):
            cluster_id = np.random.randint(0, num_clusters)
            clusters[i, cluster_id] = 1.0
        
        return clusters
    
    def _geographic_clustering(self, building_features: Dict) -> torch.Tensor:
        """Generate geographic clustering baseline"""
        # Simple k-means on coordinates
        # Implementation depends on available features
        pass
    
    def _consumption_clustering(self, temporal_data: pd.DataFrame) -> torch.Tensor:
        """Generate consumption-based clustering baseline"""
        # Cluster based on consumption patterns
        # Implementation depends on data format
        pass
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of evaluation history
        
        Returns:
            Summary statistics
        """
        if not self.evaluation_history:
            return {}
        
        latest = self.evaluation_history[-1]
        
        # Calculate trends
        if len(self.evaluation_history) > 1:
            previous = self.evaluation_history[-2]
            trends = {}
            for metric in latest['metrics']:
                if metric in previous['metrics']:
                    change = latest['metrics'][metric] - previous['metrics'][metric]
                    trends[f'{metric}_trend'] = change
        else:
            trends = {}
        
        return {
            'latest_metrics': latest['metrics'],
            'best_scores': dict(self.best_scores),
            'trends': trends,
            'num_evaluations': len(self.evaluation_history),
            'timestamp': latest['timestamp']
        }