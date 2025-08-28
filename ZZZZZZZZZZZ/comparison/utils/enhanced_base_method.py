import numpy as np
import pandas as pd
import networkx as nx
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import time
import logging
from dataclasses import dataclass
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

@dataclass
class EnhancedClusteringResult:
    """Enhanced result tracking location and temporal dynamics"""
    clusters: np.ndarray  # Final cluster assignments
    metrics: Dict[str, float]
    computation_time: float
    additional: Dict[str, Any]
    method_name: str
    
    # Location tracking
    building_locations: Optional[np.ndarray] = None  # (n_buildings, 2) lat/lon
    cluster_centers: Optional[np.ndarray] = None  # Geographic centers
    
    # Dynamic clustering over time
    temporal_clusters: Optional[np.ndarray] = None  # (n_timesteps, n_buildings)
    cluster_evolution: Optional[Dict[int, List[int]]] = None  # How clusters change
    
    # Hierarchical structure
    subclusters: Optional[Dict[int, List[int]]] = None  # Cluster -> subclusters
    hierarchy_levels: Optional[int] = None
    
    # Stability metrics
    temporal_stability: Optional[float] = None  # ARI between consecutive timesteps
    spatial_coherence: Optional[float] = None  # Geographic compactness
    
class EnhancedBaseClusteringMethod(ABC):
    """Enhanced base class with location and temporal tracking"""
    
    def __init__(self, name: str, track_dynamics: bool = True):
        self.name = name
        self.track_dynamics = track_dynamics
        self.logger = logging.getLogger(name)
        
    @abstractmethod
    def fit_predict(self, input_data: Dict[str, Any]) -> np.ndarray:
        """Core clustering method - must be implemented by subclasses"""
        pass
    
    def run(self, input_data: Dict[str, Any]) -> EnhancedClusteringResult:
        """Enhanced run method with temporal and spatial tracking"""
        start_time = time.time()
        
        # Extract location data if available
        locations = self._extract_locations(input_data)
        
        # Run main clustering
        clusters = self.fit_predict(input_data)
        
        # Track temporal dynamics if enabled
        temporal_clusters = None
        cluster_evolution = None
        temporal_stability = None
        
        if self.track_dynamics:
            temporal_clusters, cluster_evolution = self._track_temporal_dynamics(
                input_data, clusters
            )
            temporal_stability = self._calculate_temporal_stability(temporal_clusters)
        
        # Detect subclusters
        subclusters, hierarchy_levels = self._detect_subclusters(
            input_data, clusters
        )
        
        # Calculate cluster centers
        cluster_centers = self._calculate_cluster_centers(clusters, locations)
        
        # Calculate spatial coherence
        spatial_coherence = self._calculate_spatial_coherence(
            clusters, locations
        ) if locations is not None else None
        
        computation_time = time.time() - start_time
        
        # Evaluate clusters
        metrics = self.evaluate_clusters(input_data, clusters)
        
        # Add enhanced metrics
        metrics.update({
            'temporal_stability': temporal_stability or 0,
            'spatial_coherence': spatial_coherence or 0,
            'n_subclusters': sum(len(sc) for sc in subclusters.values()) if subclusters else 0,
            'hierarchy_levels': hierarchy_levels or 1
        })
        
        additional = self.get_additional_info()
        
        return EnhancedClusteringResult(
            clusters=clusters,
            metrics=metrics,
            computation_time=computation_time,
            additional=additional,
            method_name=self.name,
            building_locations=locations,
            cluster_centers=cluster_centers,
            temporal_clusters=temporal_clusters,
            cluster_evolution=cluster_evolution,
            subclusters=subclusters,
            hierarchy_levels=hierarchy_levels,
            temporal_stability=temporal_stability,
            spatial_coherence=spatial_coherence
        )
    
    def _extract_locations(self, input_data: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract building locations from input data"""
        if 'building_locations' in input_data:
            return input_data['building_locations']
        
        # Try to extract from building features
        if 'building_features' in input_data:
            features = input_data['building_features']
            if 'lat' in features and 'lon' in features:
                lat = features['lat']
                lon = features['lon']
                return np.column_stack([lat, lon])
        
        return None
    
    def _track_temporal_dynamics(
        self, 
        input_data: Dict[str, Any], 
        final_clusters: np.ndarray
    ) -> Tuple[np.ndarray, Dict]:
        """Track how clusters evolve over time"""
        consumption = input_data['consumption']
        generation = input_data.get('generation', np.zeros_like(consumption))
        
        n_timesteps, n_buildings = consumption.shape
        temporal_clusters = np.zeros((n_timesteps, n_buildings), dtype=int)
        
        # Simple approach: cluster at each timestep
        # More sophisticated methods can override this
        window_size = 4  # 1 hour windows (4 * 15min)
        
        for t in range(0, n_timesteps, window_size):
            t_end = min(t + window_size, n_timesteps)
            
            # Get data for this time window
            window_consumption = consumption[t:t_end, :].mean(axis=0)
            window_generation = generation[t:t_end, :].mean(axis=0)
            
            # Cluster based on net load in this window
            net_load = window_consumption - window_generation
            
            # Simple k-means style assignment based on similarity
            n_clusters = len(np.unique(final_clusters))
            cluster_centers = np.zeros(n_clusters)
            
            for c in range(n_clusters):
                mask = final_clusters == c
                if mask.any():
                    cluster_centers[c] = net_load[mask].mean()
            
            # Assign to nearest cluster
            for b in range(n_buildings):
                distances = np.abs(net_load[b] - cluster_centers)
                temporal_clusters[t:t_end, b] = np.argmin(distances)
        
        # Track evolution
        cluster_evolution = {}
        for t in range(1, n_timesteps):
            changes = np.where(temporal_clusters[t, :] != temporal_clusters[t-1, :])[0]
            if len(changes) > 0:
                cluster_evolution[t] = changes.tolist()
        
        return temporal_clusters, cluster_evolution
    
    def _detect_subclusters(
        self, 
        input_data: Dict[str, Any], 
        clusters: np.ndarray
    ) -> Tuple[Dict[int, List[int]], int]:
        """Detect subclusters within main clusters"""
        subclusters = {}
        max_depth = 1
        
        consumption = input_data['consumption']
        
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_size = cluster_mask.sum()
            
            if cluster_size > 5:  # Only subdivide if cluster is large enough
                # Get consumption patterns for this cluster
                cluster_consumption = consumption[:, cluster_mask]
                
                # Calculate correlation matrix
                corr_matrix = np.corrcoef(cluster_consumption.T)
                
                # Simple threshold-based subcluster detection
                threshold = 0.7
                visited = set()
                local_subclusters = []
                
                for i in range(cluster_size):
                    if i not in visited:
                        subcluster = [i]
                        visited.add(i)
                        
                        for j in range(i+1, cluster_size):
                            if j not in visited and corr_matrix[i, j] > threshold:
                                subcluster.append(j)
                                visited.add(j)
                        
                        if len(subcluster) > 1:
                            local_subclusters.append(subcluster)
                
                if local_subclusters:
                    subclusters[cluster_id] = local_subclusters
                    max_depth = 2
        
        return subclusters, max_depth
    
    def _calculate_cluster_centers(
        self, 
        clusters: np.ndarray, 
        locations: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """Calculate geographic centers of clusters"""
        if locations is None:
            return None
        
        n_clusters = len(np.unique(clusters))
        centers = np.zeros((n_clusters, 2))
        
        for c in range(n_clusters):
            mask = clusters == c
            if mask.any():
                centers[c, :] = locations[mask, :].mean(axis=0)
        
        return centers
    
    def _calculate_spatial_coherence(
        self, 
        clusters: np.ndarray, 
        locations: np.ndarray
    ) -> float:
        """Calculate how spatially coherent clusters are"""
        if locations is None:
            return 0.0
        
        total_coherence = 0.0
        n_clusters = len(np.unique(clusters))
        
        for c in range(n_clusters):
            mask = clusters == c
            if mask.sum() > 1:
                cluster_locs = locations[mask, :]
                
                # Calculate pairwise distances
                distances = []
                for i in range(len(cluster_locs)):
                    for j in range(i+1, len(cluster_locs)):
                        dist = np.linalg.norm(cluster_locs[i] - cluster_locs[j])
                        distances.append(dist)
                
                if distances:
                    # Coherence is inverse of average distance
                    avg_dist = np.mean(distances)
                    coherence = 1.0 / (1.0 + avg_dist)
                    total_coherence += coherence
        
        return total_coherence / n_clusters if n_clusters > 0 else 0.0
    
    def _calculate_temporal_stability(
        self, 
        temporal_clusters: np.ndarray
    ) -> float:
        """Calculate stability of clusters over time"""
        if temporal_clusters is None or len(temporal_clusters) < 2:
            return 0.0
        
        stabilities = []
        for t in range(1, len(temporal_clusters)):
            ari = adjusted_rand_score(
                temporal_clusters[t-1, :], 
                temporal_clusters[t, :]
            )
            stabilities.append(ari)
        
        return np.mean(stabilities) if stabilities else 0.0
    
    def evaluate_clusters(self, input_data: Dict[str, Any], clusters: np.ndarray) -> Dict[str, float]:
        """Enhanced evaluation with dynamic metrics"""
        consumption = input_data['consumption']
        generation = input_data.get('generation', np.zeros_like(consumption))
        
        n_timesteps, n_buildings = consumption.shape
        n_clusters = len(np.unique(clusters))
        
        # Time-resolved metrics
        hourly_self_sufficiency = []
        hourly_peak_reduction = []
        
        for hour in range(24):
            hour_start = hour * 4
            hour_end = (hour + 1) * 4
            
            hour_consumption = consumption[hour_start:hour_end, :].sum(axis=0)
            hour_generation = generation[hour_start:hour_end, :].sum(axis=0)
            
            hour_ssr = 0.0
            hour_peak_red = 0.0
            
            for cluster_id in np.unique(clusters):
                cluster_mask = clusters == cluster_id
                
                cluster_cons = hour_consumption[cluster_mask].sum()
                cluster_gen = hour_generation[cluster_mask].sum()
                
                if cluster_cons > 0:
                    hour_ssr += min(cluster_gen, cluster_cons) / cluster_cons
                
                individual_peak = hour_consumption[cluster_mask].max()
                cluster_peak = hour_consumption[cluster_mask].sum()
                if individual_peak > 0:
                    hour_peak_red += (individual_peak * cluster_mask.sum() - cluster_peak) / (individual_peak * cluster_mask.sum())
            
            hourly_self_sufficiency.append(hour_ssr / n_clusters)
            hourly_peak_reduction.append(hour_peak_red / n_clusters)
        
        # Overall metrics
        total_self_sufficiency = 0.0
        total_peak_reduction = 0.0
        total_energy_shared = 0.0
        
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_consumption = consumption[:, cluster_mask].sum(axis=1)
            cluster_generation = generation[:, cluster_mask].sum(axis=1)
            
            energy_shared = np.minimum(cluster_consumption, cluster_generation).sum()
            total_consumption = cluster_consumption.sum()
            
            if total_consumption > 0:
                cluster_self_sufficiency = energy_shared / total_consumption
                total_self_sufficiency += cluster_self_sufficiency * cluster_mask.sum()
            
            original_peak = consumption[:, cluster_mask].max(axis=1).sum()
            cluster_peak = cluster_consumption.max()
            if original_peak > 0:
                peak_reduction = (original_peak - cluster_peak) / original_peak
                total_peak_reduction += peak_reduction * cluster_mask.sum()
            
            total_energy_shared += energy_shared
        
        metrics = {
            'self_sufficiency': total_self_sufficiency / n_buildings,
            'peak_reduction': total_peak_reduction / n_buildings,
            'energy_shared': total_energy_shared,
            'n_clusters': n_clusters,
            'violations': self.check_violations(input_data, clusters),
            'avg_hourly_ssr': np.mean(hourly_self_sufficiency),
            'std_hourly_ssr': np.std(hourly_self_sufficiency),
            'peak_hour_ssr': np.max(hourly_self_sufficiency),
            'min_hour_ssr': np.min(hourly_self_sufficiency)
        }
        
        return metrics
    
    def check_violations(self, input_data: Dict[str, Any], clusters: np.ndarray) -> int:
        """Check constraint violations"""
        violations = 0
        
        if 'constraints' in input_data:
            constraints = input_data['constraints']
            
            if 'lv_groups' in constraints:
                lv_groups = constraints['lv_groups']
                for group in lv_groups:
                    cluster_ids = clusters[group]
                    if len(np.unique(cluster_ids)) > 1:
                        violations += 1
        
        return violations
    
    def get_additional_info(self) -> Dict[str, Any]:
        """Get method-specific additional information"""
        return {}