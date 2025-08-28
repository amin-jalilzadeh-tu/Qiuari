# tasks/enhanced_cluster_quality.py
"""
Enhanced Cluster Quality Metrics and Labeling
Comprehensive quality assessment with temporal stability
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnhancedClusterMetrics:
    """Comprehensive cluster quality metrics"""
    cluster_id: int
    timestamp: int
    lv_group_id: int
    
    # Core performance metrics (0-1 scale)
    self_sufficiency_ratio: float  # local_generation / local_demand
    self_consumption_ratio: float  # local_consumed / local_generation
    complementarity_score: float  # 1 - abs(correlation)
    peak_reduction_ratio: float  # (peak_before - peak_after) / peak_before
    
    # Stability and consistency
    temporal_stability: float  # 1 - membership_change_rate
    member_count: int
    size_appropriateness: float  # Score based on 3-20 range
    
    # Energy balance
    total_demand_kwh: float
    total_generation_kwh: float
    total_shared_kwh: float
    grid_import_kwh: float
    grid_export_kwh: float
    
    # Diversity metrics
    building_type_diversity: float  # Shannon entropy
    energy_label_diversity: float  # Distribution spread
    peak_hour_diversity: float  # Temporal spread
    
    # Economic metrics
    cost_savings_percent: float
    revenue_potential: float
    
    # Network metrics
    avg_distance_m: float
    compactness_score: float
    
    def get_overall_score(self) -> float:
        """Calculate weighted overall score (0-100)"""
        weights = {
            'self_sufficiency': 0.20,
            'complementarity': 0.20,
            'peak_reduction': 0.15,
            'temporal_stability': 0.15,
            'self_consumption': 0.10,
            'diversity': 0.10,
            'size': 0.05,
            'compactness': 0.05
        }
        
        # Aggregate diversity
        diversity_score = np.mean([
            self.building_type_diversity,
            self.energy_label_diversity,
            self.peak_hour_diversity
        ])
        
        score = (
            weights['self_sufficiency'] * self.self_sufficiency_ratio +
            weights['complementarity'] * self.complementarity_score +
            weights['peak_reduction'] * self.peak_reduction_ratio +
            weights['temporal_stability'] * self.temporal_stability +
            weights['self_consumption'] * self.self_consumption_ratio +
            weights['diversity'] * diversity_score +
            weights['size'] * self.size_appropriateness +
            weights['compactness'] * self.compactness_score
        )
        
        return min(score * 100, 100)  # Cap at 100
    
    def get_quality_label(self) -> str:
        """Get quality category"""
        score = self.get_overall_score()
        if score >= 80:
            return "excellent"
        elif score >= 60:
            return "good"  
        elif score >= 40:
            return "fair"
        else:
            return "poor"
    
    def get_explanation(self) -> str:
        """Generate human-readable explanation"""
        score = self.get_overall_score()
        label = self.get_quality_label()
        
        strengths = []
        weaknesses = []
        
        # Identify strengths
        if self.self_sufficiency_ratio > 0.7:
            strengths.append(f"High self-sufficiency ({self.self_sufficiency_ratio:.1%})")
        if self.complementarity_score > 0.8:
            strengths.append(f"Excellent complementarity ({self.complementarity_score:.2f})")
        if self.peak_reduction_ratio > 0.3:
            strengths.append(f"Significant peak reduction ({self.peak_reduction_ratio:.1%})")
        if self.temporal_stability > 0.9:
            strengths.append("Very stable membership")
        
        # Identify weaknesses
        if self.self_sufficiency_ratio < 0.3:
            weaknesses.append(f"Low self-sufficiency ({self.self_sufficiency_ratio:.1%})")
        if self.complementarity_score < 0.5:
            weaknesses.append(f"Poor complementarity ({self.complementarity_score:.2f})")
        if self.temporal_stability < 0.7:
            weaknesses.append("Unstable membership")
        if self.size_appropriateness < 0.5:
            weaknesses.append(f"Suboptimal size ({self.member_count} buildings)")
        
        explanation = f"Cluster {self.cluster_id} - {label.upper()} (Score: {score:.1f}/100)\n"
        
        if strengths:
            explanation += "Strengths: " + ", ".join(strengths) + "\n"
        if weaknesses:
            explanation += "Weaknesses: " + ", ".join(weaknesses)
        
        return explanation


class EnhancedQualityLabeler:
    """
    Enhanced cluster quality assessment with temporal tracking
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Thresholds
        self.min_cluster_size = config.get('min_cluster_size', 3)
        self.max_cluster_size = config.get('max_cluster_size', 20)
        self.min_observation_hours = config.get('min_observation_hours', 24)
        
        # Tracking
        self.cluster_history = defaultdict(list)  # cluster_id -> list of members over time
        self.metrics_history = defaultdict(list)  # cluster_id -> list of metrics
        self.stability_scores = defaultdict(float)  # cluster_id -> stability
    
    def evaluate_clusters(self,
                         cluster_assignments: torch.Tensor,
                         building_features: Dict[str, torch.Tensor],
                         temporal_data: pd.DataFrame,
                         lv_group_ids: torch.Tensor,
                         distance_matrix: Optional[torch.Tensor] = None,
                         timestamp: int = 0) -> Dict[int, EnhancedClusterMetrics]:
        """
        Evaluate all clusters at a timestep
        
        Args:
            cluster_assignments: Cluster ID per building [N]
            building_features: Dictionary of building features
            temporal_data: Time series energy data
            lv_group_ids: LV group per building [N]
            distance_matrix: Pairwise distances [N, N]
            timestamp: Current timestep
            
        Returns:
            Dictionary mapping cluster_id to metrics
        """
        unique_clusters = torch.unique(cluster_assignments)
        cluster_metrics = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = (cluster_assignments == cluster_id)
            cluster_buildings = torch.where(cluster_mask)[0]
            
            if len(cluster_buildings) < self.min_cluster_size:
                continue
            
            # Calculate metrics
            metrics = self._calculate_cluster_metrics(
                cluster_buildings,
                building_features,
                temporal_data,
                lv_group_ids,
                distance_matrix,
                cluster_id.item(),
                timestamp
            )
            
            cluster_metrics[cluster_id.item()] = metrics
            
            # Update history
            self._update_cluster_history(cluster_id.item(), cluster_buildings.tolist(), timestamp)
            self.metrics_history[cluster_id.item()].append(metrics)
        
        return cluster_metrics
    
    def _calculate_cluster_metrics(self,
                                  cluster_buildings: torch.Tensor,
                                  building_features: Dict,
                                  temporal_data: pd.DataFrame,
                                  lv_group_ids: torch.Tensor,
                                  distance_matrix: Optional[torch.Tensor],
                                  cluster_id: int,
                                  timestamp: int) -> EnhancedClusterMetrics:
        """Calculate comprehensive metrics for a cluster"""
        
        n_buildings = len(cluster_buildings)
        building_indices = cluster_buildings.cpu().numpy()
        
        # Get temporal data for cluster
        cluster_temporal = temporal_data[temporal_data['building_id'].isin(building_indices)]
        
        # Energy metrics
        total_demand = cluster_temporal['demand'].sum() if 'demand' in cluster_temporal else 0
        total_generation = cluster_temporal['generation'].sum() if 'generation' in cluster_temporal else 0
        
        self_sufficiency = min(total_generation / (total_demand + 1e-6), 1.0)
        self_consumption = min(total_demand / (total_generation + 1e-6), 1.0) if total_generation > 0 else 0
        
        # Complementarity (correlation of consumption patterns)
        if len(cluster_temporal) > 1 and 'demand' in cluster_temporal:
            demand_pivot = cluster_temporal.pivot(index='timestamp', columns='building_id', values='demand')
            if len(demand_pivot.columns) > 1:
                corr_matrix = demand_pivot.corr()
                avg_correlation = corr_matrix.values[np.triu_indices(len(corr_matrix), k=1)].mean()
                complementarity = 1 - abs(avg_correlation)
            else:
                complementarity = 0.5
        else:
            complementarity = 0.5
        
        # Peak reduction
        if 'demand' in cluster_temporal:
            individual_peaks = cluster_temporal.groupby('building_id')['demand'].max().sum()
            collective_peak = cluster_temporal.groupby('timestamp')['demand'].sum().max()
            peak_reduction = (individual_peaks - collective_peak) / (individual_peaks + 1e-6)
            peak_reduction = max(0, min(peak_reduction, 1))
        else:
            peak_reduction = 0
        
        # Temporal stability
        stability = self._calculate_temporal_stability(cluster_id)
        
        # Size appropriateness
        if self.min_cluster_size <= n_buildings <= self.max_cluster_size:
            if 5 <= n_buildings <= 15:  # Optimal range
                size_score = 1.0
            else:
                size_score = 0.7
        else:
            size_score = 0.3
        
        # Diversity metrics
        if 'type' in building_features:
            building_types = building_features['type'][building_indices]
            type_diversity = self._calculate_diversity(building_types)
        else:
            type_diversity = 0.5
        
        if 'energy_label' in building_features:
            labels = building_features['energy_label'][building_indices]
            label_diversity = self._calculate_diversity(labels)
        else:
            label_diversity = 0.5
        
        # Peak hour diversity
        if 'peak_hour' in cluster_temporal:
            peak_hours = cluster_temporal.groupby('building_id')['peak_hour'].first()
            peak_diversity = peak_hours.std() / 12 if len(peak_hours) > 1 else 0
        else:
            peak_diversity = 0.5
        
        # Compactness
        if distance_matrix is not None:
            cluster_distances = distance_matrix[building_indices][:, building_indices]
            avg_distance = cluster_distances.mean().item()
            # Normalize (closer is better)
            compactness = 1.0 / (1.0 + avg_distance / 100)
        else:
            avg_distance = 100
            compactness = 0.5
        
        # Economic metrics (simplified)
        cost_savings = peak_reduction * 0.2 + self_sufficiency * 0.3  # Rough estimate
        revenue_potential = (total_generation - total_demand) * 0.08 if total_generation > total_demand else 0
        
        # Get LV group
        cluster_lv = lv_group_ids[cluster_buildings[0]].item()
        
        return EnhancedClusterMetrics(
            cluster_id=cluster_id,
            timestamp=timestamp,
            lv_group_id=cluster_lv,
            self_sufficiency_ratio=self_sufficiency,
            self_consumption_ratio=self_consumption,
            complementarity_score=complementarity,
            peak_reduction_ratio=peak_reduction,
            temporal_stability=stability,
            member_count=n_buildings,
            size_appropriateness=size_score,
            total_demand_kwh=total_demand,
            total_generation_kwh=total_generation,
            total_shared_kwh=min(total_generation, total_demand) * complementarity,
            grid_import_kwh=max(0, total_demand - total_generation),
            grid_export_kwh=max(0, total_generation - total_demand),
            building_type_diversity=type_diversity,
            energy_label_diversity=label_diversity,
            peak_hour_diversity=peak_diversity,
            cost_savings_percent=cost_savings,
            revenue_potential=revenue_potential,
            avg_distance_m=avg_distance,
            compactness_score=compactness
        )
    
    def _calculate_diversity(self, categories: np.ndarray) -> float:
        """Calculate Shannon entropy for diversity"""
        unique, counts = np.unique(categories, return_counts=True)
        probabilities = counts / len(categories)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        max_entropy = np.log(len(unique)) if len(unique) > 1 else 1
        return entropy / max_entropy
    
    def _calculate_temporal_stability(self, cluster_id: int) -> float:
        """Calculate how stable cluster membership is over time"""
        history = self.cluster_history[cluster_id]
        
        if len(history) < 2:
            return 1.0  # New cluster, assume stable
        
        # Calculate Jaccard similarity between consecutive timesteps
        similarities = []
        for i in range(1, min(len(history), 10)):  # Look at last 10 timesteps
            prev_members = set(history[-i-1])
            curr_members = set(history[-i])
            
            if len(prev_members) == 0 and len(curr_members) == 0:
                similarity = 1.0
            else:
                intersection = len(prev_members & curr_members)
                union = len(prev_members | curr_members)
                similarity = intersection / union if union > 0 else 0
            
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 1.0
    
    def _update_cluster_history(self, cluster_id: int, members: List[int], timestamp: int):
        """Update cluster membership history"""
        self.cluster_history[cluster_id].append(members)
        
        # Keep only recent history (last 100 timesteps)
        if len(self.cluster_history[cluster_id]) > 100:
            self.cluster_history[cluster_id] = self.cluster_history[cluster_id][-100:]
    
    def generate_labels(self, cluster_metrics: Dict[int, EnhancedClusterMetrics],
                       confidence_boost: float = 0.0) -> Dict[int, Dict]:
        """
        Generate labels from metrics
        
        Returns:
            Dictionary with cluster_id -> label info
        """
        labels = {}
        
        for cluster_id, metrics in cluster_metrics.items():
            # Calculate confidence based on observation time
            observation_hours = len(self.metrics_history[cluster_id])
            base_confidence = min(observation_hours / self.min_observation_hours, 1.0)
            confidence = min(base_confidence + confidence_boost, 1.0)
            
            labels[cluster_id] = {
                'quality_score': metrics.get_overall_score(),
                'quality_label': metrics.get_quality_label(),
                'confidence': confidence,
                'explanation': metrics.get_explanation(),
                'metrics': {
                    'self_sufficiency': metrics.self_sufficiency_ratio,
                    'complementarity': metrics.complementarity_score,
                    'peak_reduction': metrics.peak_reduction_ratio,
                    'temporal_stability': metrics.temporal_stability,
                    'member_count': metrics.member_count
                }
            }
        
        return labels