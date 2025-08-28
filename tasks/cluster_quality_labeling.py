"""
Cluster Quality Labeling System
Generates quality labels for discovered clusters based on performance metrics
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClusterLabel:
    """Label for a discovered cluster"""
    cluster_id: int
    timestamp: datetime
    quality_score: float  # 0-100
    quality_category: str  # excellent/good/fair/poor
    confidence: float  # 0-1
    
    # Detailed metrics
    metrics: Dict[str, float]
    
    # Metadata
    num_buildings: int
    lv_group_id: str
    observation_hours: int
    
    # Explanation
    explanation: str


class ClusterQualityLabeler:
    """
    Generates quality labels for energy community clusters
    Evaluates clusters based on multiple performance criteria
    """
    
    def __init__(self, config: Dict):
        """
        Initialize labeler with configuration
        
        Args:
            config: Cluster metrics configuration
        """
        self.config = config
        
        # Metric weights
        self.weights = config['weights']
        
        # Category thresholds
        self.thresholds = config['thresholds']
        
        # Confidence parameters
        self.min_observation_days = config.get('min_observation_days', 7)
        self.confidence_increase_per_week = config.get('confidence_increase_per_week', 0.1)
        
        # Track labeled clusters
        self.labeled_clusters = {}
        self.label_history = []
        
        logger.info("Initialized ClusterQualityLabeler")
    
    def evaluate_cluster(
        self,
        cluster_id: int,
        cluster_members: List[int],
        temporal_data: pd.DataFrame,
        building_features: Dict,
        lv_group_id: str,
        adjacency_matrix: Optional[np.ndarray] = None
    ) -> ClusterLabel:
        """
        Evaluate a cluster and generate quality label
        
        Args:
            cluster_id: Cluster identifier
            cluster_members: List of building IDs in cluster
            temporal_data: Energy consumption/generation data
            building_features: Building characteristics
            lv_group_id: LV group this cluster belongs to
            adjacency_matrix: Physical adjacency between buildings
            
        Returns:
            ClusterLabel with quality assessment
        """
        logger.debug(f"Evaluating cluster {cluster_id} with {len(cluster_members)} buildings")
        
        # Calculate individual metrics
        metrics = {}
        
        # 1. Self-sufficiency
        metrics['self_sufficiency'] = self._calculate_self_sufficiency(
            cluster_members, temporal_data
        )
        
        # 2. Complementarity
        metrics['complementarity'] = self._calculate_complementarity(
            cluster_members, temporal_data
        )
        
        # 3. Peak reduction
        metrics['peak_reduction'] = self._calculate_peak_reduction(
            cluster_members, temporal_data
        )
        
        # 4. Physical compactness
        metrics['compactness'] = self._calculate_compactness(
            cluster_members, building_features, adjacency_matrix
        )
        
        # 5. Size appropriateness
        metrics['size_appropriateness'] = self._calculate_size_appropriateness(
            len(cluster_members)
        )
        
        # 6. Energy balance
        metrics['energy_balance'] = self._calculate_energy_balance(
            cluster_members, temporal_data
        )
        
        # 7. Load diversity
        metrics['load_diversity'] = self._calculate_load_diversity(
            cluster_members, temporal_data
        )
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(metrics)
        
        # Determine category
        quality_category = self._categorize_quality(quality_score)
        
        # Calculate confidence
        observation_hours = len(temporal_data) if temporal_data is not None else 0
        confidence = self._calculate_confidence(observation_hours, metrics)
        
        # Generate explanation
        explanation = self._generate_explanation(
            quality_category, metrics, cluster_members
        )
        
        # Create label
        label = ClusterLabel(
            cluster_id=cluster_id,
            timestamp=datetime.now(),
            quality_score=quality_score * 100,  # Convert to 0-100
            quality_category=quality_category,
            confidence=confidence,
            metrics=metrics,
            num_buildings=len(cluster_members),
            lv_group_id=lv_group_id,
            observation_hours=observation_hours,
            explanation=explanation
        )
        
        # Store label
        self.labeled_clusters[cluster_id] = label
        self.label_history.append(label)
        
        logger.info(f"Cluster {cluster_id} labeled as {quality_category} "
                   f"(score: {quality_score*100:.1f}, confidence: {confidence:.2f})")
        
        return label
    
    def _calculate_self_sufficiency(
        self,
        cluster_members: List[int],
        temporal_data: pd.DataFrame
    ) -> float:
        """
        Calculate cluster self-sufficiency ratio
        
        Args:
            cluster_members: Building IDs in cluster
            temporal_data: Energy data
            
        Returns:
            Self-sufficiency score (0-1)
        """
        if temporal_data is None or temporal_data.empty:
            return 0.5  # Default if no data
        
        # Filter data for cluster members
        cluster_data = temporal_data[temporal_data['building_id'].isin(cluster_members)]
        
        if cluster_data.empty:
            return 0.5
        
        # Aggregate by timestamp
        hourly = cluster_data.groupby('timestamp').agg({
            'demand_kw': 'sum',
            'generation_kw': 'sum'
        })
        
        # Calculate self-sufficiency for each hour
        hourly['self_consumed'] = hourly[['demand_kw', 'generation_kw']].min(axis=1)
        
        # Overall self-sufficiency
        total_demand = hourly['demand_kw'].sum()
        total_self_consumed = hourly['self_consumed'].sum()
        
        if total_demand > 0:
            self_sufficiency = total_self_consumed / total_demand
        else:
            self_sufficiency = 0
        
        return min(self_sufficiency, 1.0)
    
    def _calculate_complementarity(
        self,
        cluster_members: List[int],
        temporal_data: pd.DataFrame
    ) -> float:
        """
        Calculate complementarity of consumption patterns
        
        Args:
            cluster_members: Building IDs in cluster
            temporal_data: Energy data
            
        Returns:
            Complementarity score (-1 to 1, negative is better)
        """
        if temporal_data is None or len(cluster_members) < 2:
            return 0  # No complementarity for single building
        
        # Get demand profiles for each building
        profiles = []
        for building_id in cluster_members:
            building_data = temporal_data[temporal_data['building_id'] == building_id]
            if not building_data.empty:
                profile = building_data.groupby('hour')['demand_kw'].mean().values
                if len(profile) == 24:  # Full day profile
                    profiles.append(profile)
        
        if len(profiles) < 2:
            return 0
        
        # Calculate pairwise correlations
        profiles_array = np.array(profiles)
        correlation_matrix = np.corrcoef(profiles_array)
        
        # Get upper triangle (excluding diagonal)
        upper_triangle = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]
        
        # Average correlation (negative is better for complementarity)
        avg_correlation = np.mean(upper_triangle) if len(upper_triangle) > 0 else 0
        
        # Convert to complementarity score (invert)
        complementarity = -avg_correlation
        
        return complementarity
    
    def _calculate_peak_reduction(
        self,
        cluster_members: List[int],
        temporal_data: pd.DataFrame
    ) -> float:
        """
        Calculate peak demand reduction from clustering
        
        Args:
            cluster_members: Building IDs in cluster
            temporal_data: Energy data
            
        Returns:
            Peak reduction ratio (0-1)
        """
        if temporal_data is None or temporal_data.empty:
            return 0
        
        cluster_data = temporal_data[temporal_data['building_id'].isin(cluster_members)]
        
        if cluster_data.empty:
            return 0
        
        # Individual peaks
        individual_peaks = []
        for building_id in cluster_members:
            building_data = cluster_data[cluster_data['building_id'] == building_id]
            if not building_data.empty:
                peak = building_data['demand_kw'].max()
                individual_peaks.append(peak)
        
        if not individual_peaks:
            return 0
        
        sum_individual_peaks = sum(individual_peaks)
        
        # Cluster peak (aggregate demand)
        hourly_cluster = cluster_data.groupby('timestamp')['demand_kw'].sum()
        cluster_peak = hourly_cluster.max() if not hourly_cluster.empty else 0
        
        # Peak reduction
        if sum_individual_peaks > 0:
            peak_reduction = 1 - (cluster_peak / sum_individual_peaks)
        else:
            peak_reduction = 0
        
        return max(0, min(peak_reduction, 1))
    
    def _calculate_compactness(
        self,
        cluster_members: List[int],
        building_features: Dict,
        adjacency_matrix: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate physical compactness of cluster
        
        Args:
            cluster_members: Building IDs in cluster
            building_features: Building characteristics
            adjacency_matrix: Physical adjacency
            
        Returns:
            Compactness score (0-1)
        """
        if len(cluster_members) <= 1:
            return 1.0  # Single building is maximally compact
        
        # Check if buildings share walls (best case)
        if adjacency_matrix is not None:
            adjacency_count = 0
            for i, b1 in enumerate(cluster_members):
                for b2 in cluster_members[i+1:]:
                    if adjacency_matrix[b1, b2] > 0:
                        adjacency_count += 1
            
            # Ratio of adjacent pairs to total possible pairs
            total_pairs = len(cluster_members) * (len(cluster_members) - 1) / 2
            if total_pairs > 0:
                adjacency_ratio = adjacency_count / total_pairs
            else:
                adjacency_ratio = 0
            
            # High adjacency = high compactness
            if adjacency_ratio > 0.5:
                return 0.9 + 0.1 * adjacency_ratio
        
        # Fall back to distance-based compactness
        if building_features:
            positions = []
            for building_id in cluster_members:
                features = building_features.get(building_id, {})
                if 'x' in features and 'y' in features:
                    positions.append([features['x'], features['y']])
            
            if len(positions) >= 2:
                positions = np.array(positions)
                # Calculate average pairwise distance
                distances = []
                for i in range(len(positions)):
                    for j in range(i+1, len(positions)):
                        dist = np.linalg.norm(positions[i] - positions[j])
                        distances.append(dist)
                
                avg_distance = np.mean(distances) if distances else 0
                
                # Convert to compactness score (closer = better)
                # Assume 500m is far, 50m is close
                compactness = max(0, 1 - avg_distance / 500)
                return compactness
        
        return 0.5  # Default if no spatial info
    
    def _calculate_size_appropriateness(self, cluster_size: int) -> float:
        """
        Calculate if cluster size is appropriate
        
        Args:
            cluster_size: Number of buildings
            
        Returns:
            Size score (0-1)
        """
        min_size = 3
        max_size = 20
        optimal_size = 8
        
        if cluster_size < min_size or cluster_size > max_size:
            return 0
        elif cluster_size == optimal_size:
            return 1.0
        else:
            # Linear decay from optimal
            if cluster_size < optimal_size:
                return 0.5 + 0.5 * (cluster_size - min_size) / (optimal_size - min_size)
            else:
                return 0.5 + 0.5 * (max_size - cluster_size) / (max_size - optimal_size)
    
    def _calculate_energy_balance(
        self,
        cluster_members: List[int],
        temporal_data: pd.DataFrame
    ) -> float:
        """
        Calculate energy balance (generation vs demand)
        
        Args:
            cluster_members: Building IDs in cluster
            temporal_data: Energy data
            
        Returns:
            Balance score (0-1)
        """
        if temporal_data is None or temporal_data.empty:
            return 0.5
        
        cluster_data = temporal_data[temporal_data['building_id'].isin(cluster_members)]
        
        if cluster_data.empty:
            return 0.5
        
        total_demand = cluster_data['demand_kw'].sum()
        total_generation = cluster_data['generation_kw'].sum()
        
        if total_demand > 0:
            ratio = total_generation / total_demand
            # Perfect balance at ratio=1, score decreases for imbalance
            if ratio <= 1:
                balance = ratio  # Under-generation
            else:
                balance = max(0, 2 - ratio)  # Over-generation penalty
        else:
            balance = 0.5
        
        return min(balance, 1.0)
    
    def _calculate_load_diversity(
        self,
        cluster_members: List[int],
        temporal_data: pd.DataFrame
    ) -> float:
        """
        Calculate load diversity factor
        
        Args:
            cluster_members: Building IDs in cluster
            temporal_data: Energy data
            
        Returns:
            Diversity score (0-1)
        """
        if temporal_data is None or len(cluster_members) < 2:
            return 0.5
        
        # Get peak times for each building
        peak_hours = []
        for building_id in cluster_members:
            building_data = temporal_data[temporal_data['building_id'] == building_id]
            if not building_data.empty:
                hourly = building_data.groupby('hour')['demand_kw'].mean()
                if not hourly.empty:
                    peak_hour = hourly.idxmax()
                    peak_hours.append(peak_hour)
        
        if len(peak_hours) < 2:
            return 0.5
        
        # Diversity = unique peak hours / total buildings
        unique_peaks = len(set(peak_hours))
        diversity = unique_peaks / len(peak_hours)
        
        return diversity
    
    def _calculate_quality_score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate overall quality score from metrics
        
        Args:
            metrics: Individual metric scores
            
        Returns:
            Weighted quality score (0-1)
        """
        score = 0
        total_weight = 0
        
        for metric_name, weight in self.weights.items():
            if metric_name in metrics:
                # Handle negative metrics (complementarity)
                if metric_name == 'complementarity':
                    # Convert from [-1, 1] to [0, 1] where negative is better
                    metric_value = max(0, -metrics[metric_name])
                else:
                    metric_value = metrics[metric_name]
                
                score += weight * metric_value
                total_weight += weight
        
        if total_weight > 0:
            score = score / total_weight
        
        return min(max(score, 0), 1)
    
    def _categorize_quality(self, quality_score: float) -> str:
        """
        Categorize quality score
        
        Args:
            quality_score: Overall quality (0-1)
            
        Returns:
            Category string
        """
        if quality_score >= self.thresholds['excellent']:
            return 'excellent'
        elif quality_score >= self.thresholds['good']:
            return 'good'
        elif quality_score >= self.thresholds['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _calculate_confidence(
        self,
        observation_hours: int,
        metrics: Dict[str, float]
    ) -> float:
        """
        Calculate confidence in label
        
        Args:
            observation_hours: Hours of data observed
            metrics: Calculated metrics
            
        Returns:
            Confidence score (0-1)
        """
        # Base confidence on observation time
        observation_days = observation_hours / 24
        min_days = self.min_observation_days
        
        if observation_days >= min_days:
            time_confidence = min(1.0, 0.7 + observation_days / (min_days * 4))
        else:
            time_confidence = observation_days / min_days * 0.7
        
        # Adjust based on metric consistency
        # If metrics are extreme (very good or very bad), higher confidence
        metric_values = [v for k, v in metrics.items() if k != 'complementarity']
        if metric_values:
            avg_metric = np.mean(metric_values)
            if avg_metric > 0.8 or avg_metric < 0.2:
                metric_confidence = 0.9
            else:
                metric_confidence = 0.7
        else:
            metric_confidence = 0.5
        
        # Combined confidence
        confidence = 0.7 * time_confidence + 0.3 * metric_confidence
        
        return min(confidence, 0.95)  # Cap at 0.95
    
    def _generate_explanation(
        self,
        quality_category: str,
        metrics: Dict[str, float],
        cluster_members: List[int]
    ) -> str:
        """
        Generate human-readable explanation
        
        Args:
            quality_category: Quality category
            metrics: Calculated metrics
            cluster_members: Building IDs
            
        Returns:
            Explanation string
        """
        explanation_parts = [f"This cluster is rated '{quality_category}' because:"]
        
        # Identify strongest and weakest points
        sorted_metrics = sorted(metrics.items(), key=lambda x: x[1], reverse=True)
        
        # Top strengths
        strengths = []
        for metric, value in sorted_metrics[:2]:
            if metric == 'complementarity':
                if value < -0.5:
                    strengths.append("excellent complementary consumption patterns")
            elif metric == 'self_sufficiency' and value > 0.7:
                strengths.append(f"high self-sufficiency ({value*100:.0f}%)")
            elif metric == 'peak_reduction' and value > 0.3:
                strengths.append(f"significant peak reduction ({value*100:.0f}%)")
            elif metric == 'compactness' and value > 0.7:
                strengths.append("physically compact layout")
        
        if strengths:
            explanation_parts.append(f"Strengths: {', '.join(strengths)}")
        
        # Weaknesses
        weaknesses = []
        for metric, value in reversed(sorted_metrics[-2:]):
            if metric == 'size_appropriateness' and value < 0.5:
                weaknesses.append(f"suboptimal size ({len(cluster_members)} buildings)")
            elif metric == 'energy_balance' and value < 0.3:
                weaknesses.append("poor energy balance")
            elif metric == 'load_diversity' and value < 0.3:
                weaknesses.append("low load diversity")
        
        if weaknesses:
            explanation_parts.append(f"Weaknesses: {', '.join(weaknesses)}")
        
        # Recommendation
        if quality_category == 'excellent':
            explanation_parts.append("Recommendation: Maintain cluster, prioritize for incentives")
        elif quality_category == 'good':
            explanation_parts.append("Recommendation: Consider adding solar/storage for improvement")
        elif quality_category == 'fair':
            explanation_parts.append("Recommendation: Review cluster composition, may need adjustment")
        else:
            explanation_parts.append("Recommendation: Reconfigure cluster for better performance")
        
        return " ".join(explanation_parts)
    
    def generate_batch_labels(
        self,
        clusters: Dict[int, List[int]],
        temporal_data: pd.DataFrame,
        building_features: Dict,
        lv_groups: Dict[int, str],
        adjacency_matrix: Optional[np.ndarray] = None
    ) -> Dict[int, ClusterLabel]:
        """
        Generate labels for multiple clusters
        
        Args:
            clusters: Dictionary of cluster_id -> member list
            temporal_data: Energy data
            building_features: Building characteristics
            lv_groups: Mapping of cluster to LV group
            adjacency_matrix: Physical adjacency
            
        Returns:
            Dictionary of labels by cluster ID
        """
        labels = {}
        
        for cluster_id, members in clusters.items():
            if len(members) >= 3:  # Only label valid clusters
                lv_group_id = lv_groups.get(cluster_id, 'unknown')
                
                label = self.evaluate_cluster(
                    cluster_id,
                    members,
                    temporal_data,
                    building_features,
                    lv_group_id,
                    adjacency_matrix
                )
                
                labels[cluster_id] = label
        
        # Log summary statistics
        categories = {'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0}
        for label in labels.values():
            categories[label.quality_category] += 1
        
        logger.info(f"Generated {len(labels)} cluster labels: {categories}")
        
        return labels