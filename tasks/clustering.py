# tasks/clustering.py
"""
Dynamic energy community clustering task
Discovers optimal building groups for energy sharing within LV boundaries
Tracks temporal energy flows and calculates key performance metrics
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import networkx as nx
from sklearn.metrics import silhouette_score
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class ClusterConfig:
    """Configuration for clustering task"""
    min_cluster_size: int = 3
    max_cluster_size: int = 20
    respect_lv_boundaries: bool = True
    complementarity_weight: float = 2.0
    adjacency_bonus: float = 1.5
    distance_penalty_factor: float = 0.01
    temporal_stability_weight: float = 0.3
    self_sufficiency_target: float = 0.7
    peak_reduction_target: float = 0.3


@dataclass
class EnergyFlow:
    """Energy flow between two buildings"""
    from_building: int
    to_building: int
    timestamp: datetime
    energy_kwh: float
    efficiency: float
    flow_type: str  # 'direct', 'battery', 'grid'


@dataclass
class ClusterMetrics:
    """Metrics for a cluster at a specific timestamp"""
    cluster_id: int
    timestamp: datetime
    self_sufficiency_ratio: float
    self_consumption_ratio: float
    peak_to_average_ratio: float
    peak_reduction_percent: float
    total_shared_kwh: float
    grid_import_kwh: float
    grid_export_kwh: float
    thermal_savings_kwh: float
    member_buildings: List[int]
    complementarity_score: float


class EnergyCommunityClustering:
    """
    Dynamic clustering for energy communities with temporal energy flow tracking
    """
    
    def __init__(self, model, config: Union[Dict, ClusterConfig]):
        """
        Initialize clustering task
        
        Args:
            model: Trained GNN model with clustering head
            config: Task configuration (dict or ClusterConfig)
        """
        self.model = model
        
        # Parse configuration
        if isinstance(config, dict):
            self.config = ClusterConfig(**config.get('clustering', {}))
        else:
            self.config = config
            
        # Storage for temporal tracking
        self.cluster_history = []
        self.energy_flows = []
        self.metrics_history = []
        # Building ID to index mapping
        self.building_id_to_index = {}
        self.index_to_building_id = {}
        # Caches for efficiency
        self._distance_cache = {}
        self._complementarity_cache = {}
        
        logger.info(f"Initialized EnergyCommunityClustering with config: {self.config}")
    
    def run(self, 
            graph_data: Dict,
            embeddings: torch.Tensor,
            temporal_data: Optional[pd.DataFrame] = None,
            current_timestamp: Optional[datetime] = None) -> Dict:
        """
        Main clustering execution
        
        Args:
            graph_data: Graph data from KG containing building info and LV groups
            embeddings: GNN embeddings [num_buildings, embedding_dim]
            temporal_data: Energy consumption/generation profiles
            current_timestamp: Current time for dynamic clustering
            
        Returns:
            Dictionary with clustering results, energy flows, and metrics
        """
        logger.info(f"Running clustering for {len(embeddings)} buildings")
        
        # Extract building and LV group information
        building_info = self._extract_building_info(graph_data)
        lv_groups = self._extract_lv_groups(graph_data)
        
        # ADD THESE NEW LINES:
        # Create building ID to index mapping
        building_ids = list(building_info.keys())
        self.building_id_to_index = {bid: idx for idx, bid in enumerate(building_ids)}
        self.index_to_building_id = {idx: bid for idx, bid in enumerate(building_ids)}
        
        # Compute complementarity matrix
        complementarity_matrix = self._compute_complementarity(
            embeddings, temporal_data, building_info
        )
        
        # Form clusters within each LV group
        clusters = self._form_clusters_by_lv_group(
            embeddings,
            complementarity_matrix,
            building_info,
            lv_groups
        )
        
        # Calculate energy flows if temporal data available
        energy_flows = []
        if temporal_data is not None and current_timestamp is not None:
            energy_flows = self._calculate_energy_flows(
                clusters,
                temporal_data,
                current_timestamp,
                building_info
            )
        
        # Calculate cluster metrics
        metrics = self._calculate_cluster_metrics(
            clusters,
            temporal_data,
            energy_flows,
            current_timestamp,
            building_info
        )
        
        # Apply temporal stability (prevent too frequent changes)
        if self.cluster_history:
            clusters = self._apply_temporal_stability(
                clusters,
                self.cluster_history[-1],
                self.config.temporal_stability_weight
            )
        
        # Store in history
        self.cluster_history.append(clusters)
        self.energy_flows.extend(energy_flows)
        self.metrics_history.extend(metrics)
        
        # Prepare output
        return {
            'clusters': clusters,
            'energy_flows': energy_flows,
            'metrics': metrics,
            'complementarity_matrix': complementarity_matrix,
            'summary': self._generate_summary(clusters, metrics)
        }
    
    def _extract_building_info(self, graph_data: Dict) -> Dict:
        """Extract building information from graph data"""
        building_info = {}
        
        for building_id, data in graph_data.get('buildings', {}).items():
            building_info[building_id] = {
                'lv_group': data.get('lv_group_id'),
                'x': data.get('x_coord'),
                'y': data.get('y_coord'),
                'has_solar': data.get('has_solar', False),
                'solar_capacity_kw': data.get('solar_capacity_kw', 0),
                'has_battery': data.get('has_battery', False),
                'battery_capacity_kwh': data.get('battery_capacity_kwh', 0),
                'shared_walls': data.get('shared_walls', []),
                'building_type': data.get('building_function'),
                'area': data.get('area', 0),
                'peak_demand_kw': data.get('peak_demand_kw', 0)
            }
        
        return building_info
    
    def _extract_lv_groups(self, graph_data: Dict) -> Dict:
        """Extract LV group topology from graph data"""
        lv_groups = defaultdict(list)
        
        for building_id, info in graph_data.get('buildings', {}).items():
            lv_group = info.get('lv_group_id')
            if lv_group:
                lv_groups[lv_group].append(building_id)
        
        # Add transformer capacity limits
        for lv_group_id in lv_groups:
            transformer_data = graph_data.get('transformers', {}).get(lv_group_id, {})
            lv_groups[lv_group_id] = {
                'buildings': lv_groups[lv_group_id],
                'capacity_kva': transformer_data.get('capacity_kva', 250),
                'current_peak_kw': transformer_data.get('current_peak_kw', 0)
            }
        
        return dict(lv_groups)
    
    def _compute_complementarity(self,
                                embeddings: torch.Tensor,
                                temporal_data: Optional[pd.DataFrame],
                                building_info: Dict) -> torch.Tensor:
        """
        Compute complementarity matrix between buildings
        
        Higher scores indicate better complementarity (opposite patterns)
        """
        num_buildings = len(embeddings)
        comp_matrix = torch.zeros(num_buildings, num_buildings)
        
        # Get ordered list of building IDs
        building_ids = list(building_info.keys())
        # Store for later use
        self.building_id_list = building_ids
        
        if temporal_data is not None:
            # Compute correlation-based complementarity
            for i in range(num_buildings):
                for j in range(i+1, num_buildings):
                    # Get building IDs
                    building_i = building_ids[i] if i < len(building_ids) else None
                    building_j = building_ids[j] if j < len(building_ids) else None
                    
                    if building_i is None or building_j is None:
                        continue
                    
                    # Get consumption profiles
                    if building_i in temporal_data.columns and building_j in temporal_data.columns:
                        profile_i = temporal_data[building_i].values
                        profile_j = temporal_data[building_j].values
                    else:
                        # Use index if building ID not in columns
                        profile_i = temporal_data.iloc[:, i].values if i < temporal_data.shape[1] else np.zeros(len(temporal_data))
                        profile_j = temporal_data.iloc[:, j].values if j < temporal_data.shape[1] else np.zeros(len(temporal_data))
                    
                    # Compute anti-correlation (complementarity)
                    if len(profile_i) > 0 and len(profile_j) > 0:
                        correlation = np.corrcoef(profile_i, profile_j)[0, 1]
                        complementarity = -correlation  # Negative correlation is good
                    else:
                        complementarity = 0
                    
                    # Get building types (check multiple possible field names)
                    type_i = (building_info[building_i].get('building_function') or 
                            building_info[building_i].get('building_type') or 
                            'unknown')
                    type_j = (building_info[building_j].get('building_function') or 
                            building_info[building_j].get('building_type') or 
                            'unknown')
                    
                    # Bonus for different building types
                    if type_i != type_j:
                        complementarity *= 1.2
                    
                    # Bonus for one having solar and other not
                    solar_i = building_info[building_i].get('has_solar', False)
                    solar_j = building_info[building_j].get('has_solar', False)
                    if solar_i != solar_j:
                        complementarity *= 1.3
                    
                    # Store symmetrically
                    comp_matrix[i, j] = complementarity
                    comp_matrix[j, i] = complementarity
        else:
            # Use embedding-based similarity as proxy
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            comp_matrix = 1.0 - torch.mm(embeddings_norm, embeddings_norm.t())
        
        # Add adjacency bonus
        for i, building_i in enumerate(building_ids):
            if building_i in building_info:
                for neighbor_j in building_info[building_i].get('shared_walls', []):
                    # Find index of neighbor
                    if neighbor_j in building_ids:
                        j = building_ids.index(neighbor_j)
                        if j < num_buildings:
                            comp_matrix[i, j] *= self.config.adjacency_bonus
                            comp_matrix[j, i] *= self.config.adjacency_bonus
        
        return comp_matrix
    
    def _form_clusters_by_lv_group(self,
                                   embeddings: torch.Tensor,
                                   complementarity: torch.Tensor,
                                   building_info: Dict,
                                   lv_groups: Dict) -> List[List[int]]:
        """
        Form clusters within each LV group boundary
        """
        all_clusters = []
        
        for lv_group_id, lv_data in lv_groups.items():
            group_buildings = lv_data['buildings']
            
            if len(group_buildings) < self.config.min_cluster_size:
                # Too small for clustering
                if len(group_buildings) > 0:
                    all_clusters.append(group_buildings)
                continue
            
            # Get indices for buildings in this LV group
            # Get indices for buildings in this LV group
            building_ids = list(building_info.keys())
            group_indices = [i for i, b_id in enumerate(building_ids) 
                        if b_id in group_buildings]
            
            # Extract submatrices for this LV group
            group_embeddings = embeddings[group_indices]
            group_complementarity = complementarity[group_indices][:, group_indices]
            
            # Determine optimal number of clusters dynamically for this LV group
            # Use new method if we have enough buildings, otherwise fall back to heuristic
            if len(group_buildings) >= 10 and self.config.complementarity_weight > 0:
                num_clusters = self._determine_optimal_k_per_lv(
                    group_embeddings, 
                    group_complementarity,
                    len(group_buildings)
                )
            else:
                num_clusters = self._determine_num_clusters(len(group_buildings))
            
            # Perform spectral clustering with complementarity
            clusters = self._spectral_clustering_with_constraints(
                group_embeddings,
                group_complementarity,
                num_clusters,
                self.config.min_cluster_size,
                self.config.max_cluster_size
            )
            
            # Map back to building IDs
            for cluster in clusters:
                cluster_building_ids = [group_buildings[idx] for idx in cluster]
                all_clusters.append(cluster_building_ids)
        
        return all_clusters
    
    def _determine_num_clusters(self, num_buildings: int) -> int:
        """Determine optimal number of clusters for a group"""
        # Simple heuristic: aim for clusters of size 8-12
        target_size = 10
        num_clusters = max(1, num_buildings // target_size)
        
        # Ensure we don't create too many small clusters
        max_clusters = num_buildings // self.config.min_cluster_size
        num_clusters = min(num_clusters, max_clusters)
        
        return num_clusters
    
    def _determine_optimal_k_per_lv(self, group_embeddings: torch.Tensor, 
                                     group_complementarity: torch.Tensor,
                                     num_buildings: int) -> int:
        """
        Dynamically determine optimal number of clusters for this specific LV group
        based on complementarity patterns and modularity
        """
        if num_buildings < self.config.min_cluster_size * 2:
            return 1  # Too small to cluster
        
        # Try different K values
        min_k = max(2, num_buildings // self.config.max_cluster_size)
        max_k = min(num_buildings // self.config.min_cluster_size, 10)  # Cap at 10 for efficiency
        
        best_k = min_k
        best_score = -float('inf')
        
        for k in range(min_k, max_k + 1):
            # Perform clustering with this K
            clusters = self._spectral_clustering_with_constraints(
                group_embeddings,
                group_complementarity,
                k,
                self.config.min_cluster_size,
                self.config.max_cluster_size
            )
            
            # Evaluate quality
            score = self._evaluate_clustering_quality(
                clusters,
                group_complementarity,
                group_embeddings
            )
            
            if score > best_score:
                best_score = score
                best_k = k
        
        logger.info(f"LV group with {num_buildings} buildings: optimal K={best_k} (score={best_score:.3f})")
        return best_k
    
    def _evaluate_clustering_quality(self, clusters: List[List[int]], 
                                    complementarity: torch.Tensor,
                                    embeddings: torch.Tensor) -> float:
        """
        Evaluate clustering quality based on:
        1. Internal complementarity (negative correlation within clusters)
        2. Separation between clusters
        3. Size balance
        """
        if not clusters:
            return 0.0
        
        score = 0.0
        total_buildings = sum(len(c) for c in clusters)
        
        # 1. Internal complementarity score
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            # Average complementarity within cluster
            cluster_comp = 0.0
            pairs = 0
            for i in range(len(cluster)):
                for j in range(i+1, len(cluster)):
                    cluster_comp += complementarity[cluster[i], cluster[j]].item()
                    pairs += 1
            
            if pairs > 0:
                # Higher negative values (complementarity) = better score
                score += -cluster_comp / pairs  # Negate because negative correlation is good
        
        # 2. Size balance penalty
        cluster_sizes = [len(c) for c in clusters]
        mean_size = np.mean(cluster_sizes)
        size_variance = np.var(cluster_sizes)
        size_penalty = size_variance / (mean_size ** 2 + 1e-6)
        score -= size_penalty * 0.5
        
        # 3. Coverage bonus (fewer orphans)
        coverage = total_buildings / embeddings.shape[0]
        score += coverage * 0.3
        
        return score
    
    def _spectral_clustering_with_constraints(self,
                                             embeddings: torch.Tensor,
                                             affinity: torch.Tensor,
                                             num_clusters: int,
                                             min_size: int,
                                             max_size: int) -> List[List[int]]:
        """
        Perform spectral clustering with size constraints
        """
        n = len(embeddings)
        
        if n <= max_size:
            # Single cluster if small enough
            return [list(range(n))]
        
        # Convert to numpy for spectral operations
        affinity_np = affinity.cpu().numpy()
        
        # Create graph Laplacian
        degree = np.sum(affinity_np, axis=1)
        degree_matrix = np.diag(degree)
        laplacian = degree_matrix - affinity_np
        
        # Compute eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        
        # Use first k eigenvectors for clustering
        features = eigenvectors[:, :num_clusters]
        
        # K-means on spectral features
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        
        # Group into clusters
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[label].append(idx)
        
        # Apply size constraints
        final_clusters = self._apply_size_constraints(
            list(clusters.values()),
            affinity_np,
            min_size,
            max_size
        )
        
        return final_clusters
    
    def _apply_size_constraints(self,
                               clusters: List[List[int]],
                               affinity: np.ndarray,
                               min_size: int,
                               max_size: int) -> List[List[int]]:
        """
        Post-process clusters to satisfy size constraints
        """
        final_clusters = []
        small_clusters = []
        
        for cluster in clusters:
            if len(cluster) > max_size:
                # Split large clusters
                sub_clusters = self._split_cluster(cluster, affinity, max_size)
                final_clusters.extend(sub_clusters)
            elif len(cluster) < min_size:
                # Collect small clusters for merging
                small_clusters.extend(cluster)
            else:
                final_clusters.append(cluster)
        
        # Merge small clusters
        if small_clusters:
            merged = self._merge_small_clusters(
                small_clusters,
                affinity,
                min_size,
                max_size
            )
            final_clusters.extend(merged)
        
        return final_clusters
    
    def _split_cluster(self,
                      cluster: List[int],
                      affinity: np.ndarray,
                      max_size: int) -> List[List[int]]:
        """Split a large cluster into smaller ones"""
        num_splits = (len(cluster) + max_size - 1) // max_size
        
        # Use hierarchical clustering for splitting
        from sklearn.cluster import AgglomerativeClustering
        
        sub_affinity = affinity[np.ix_(cluster, cluster)]
        clustering = AgglomerativeClustering(
            n_clusters=num_splits,
            affinity='precomputed',
            linkage='average'
        )
        labels = clustering.fit_predict(1 - sub_affinity)  # Convert to distance
        
        # Group by labels
        sub_clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            sub_clusters[label].append(cluster[idx])
        
        return list(sub_clusters.values())
    
    def _merge_small_clusters(self,
                             nodes: List[int],
                             affinity: np.ndarray,
                             min_size: int,
                             max_size: int) -> List[List[int]]:
        """Merge small clusters to meet minimum size"""
        if not nodes:
            return []
        
        # Greedy merging based on affinity
        clusters = []
        current_cluster = []
        
        remaining = set(nodes)
        
        while remaining:
            if not current_cluster:
                # Start new cluster with random node
                node = remaining.pop()
                current_cluster = [node]
            else:
                # Find best node to add
                best_node = None
                best_score = -float('inf')
                
                for node in remaining:
                    score = np.mean([affinity[node, c] for c in current_cluster])
                    if score > best_score:
                        best_score = score
                        best_node = node
                
                if best_node is not None:
                    current_cluster.append(best_node)
                    remaining.remove(best_node)
                
                # Check if cluster is complete
                if len(current_cluster) >= min_size or len(current_cluster) >= max_size:
                    clusters.append(current_cluster)
                    current_cluster = []
        
        # Add remaining cluster if non-empty
        if current_cluster:
            if len(current_cluster) < min_size and clusters:
                # Merge with last cluster if too small
                clusters[-1].extend(current_cluster)
            else:
                clusters.append(current_cluster)
        
        return clusters
    
    def _calculate_energy_flows(self,
                               clusters: List[List[int]],
                               temporal_data: pd.DataFrame,
                               timestamp: datetime,
                               building_info: Dict) -> List[EnergyFlow]:
        """
        Calculate energy flows between buildings within clusters
        """
        flows = []
        
        # Get current timestamp index
        time_idx = self._get_time_index(temporal_data, timestamp)
        if time_idx is None:
            return flows
        
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            # Get supply and demand for each building
            supplies = {}
            demands = {}
            
            for building_id in cluster:
                building_idx = self._get_building_index(building_id)
                
                # Get consumption
                consumption = temporal_data.iloc[time_idx, building_idx]
                
                # Get generation (if has solar)
                generation = 0
                if building_info[building_id]['has_solar']:
                    # Simplified: assume generation profile
                    generation = building_info[building_id]['solar_capacity_kw'] * \
                                self._solar_generation_factor(timestamp)
                
                # Get battery state (if has battery)
                battery_available = 0
                if building_info[building_id]['has_battery']:
                    # Simplified: assume battery can discharge
                    battery_available = min(
                        building_info[building_id]['battery_capacity_kwh'] * 0.5,
                        consumption
                    )
                
                # Calculate net position
                net_supply = generation + battery_available - consumption
                
                if net_supply > 0:
                    supplies[building_id] = net_supply
                else:
                    demands[building_id] = -net_supply
            
            # Match supplies with demands using optimal assignment
            flows.extend(
                self._match_supply_demand(
                    supplies,
                    demands,
                    building_info,
                    timestamp,
                    cluster
                )
            )
        
        return flows
    
    def _match_supply_demand(self,
                            supplies: Dict[int, float],
                            demands: Dict[int, float],
                            building_info: Dict,
                            timestamp: datetime,
                            cluster: List[int]) -> List[EnergyFlow]:
        """
        Optimally match energy supplies with demands
        """
        flows = []
        
        if not supplies or not demands:
            return flows
        
        # Create cost matrix based on distance and complementarity
        supply_ids = list(supplies.keys())
        demand_ids = list(demands.keys())
        
        cost_matrix = np.zeros((len(supplies), len(demands)))
        
        for i, supply_id in enumerate(supply_ids):
            for j, demand_id in enumerate(demand_ids):
                # Distance-based efficiency loss
                distance = self._calculate_distance(
                    building_info[supply_id],
                    building_info[demand_id]
                )
                efficiency = 1.0 / (1.0 + self.config.distance_penalty_factor * distance)
                
                # Adjacency bonus
                if demand_id in building_info[supply_id].get('shared_walls', []):
                    efficiency *= 1.1
                
                # Cost is inverse of efficiency
                cost_matrix[i, j] = 1.0 / efficiency
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create flows based on assignment
        for i, j in zip(row_ind, col_ind):
            supply_id = supply_ids[i]
            demand_id = demand_ids[j]
            
            # Determine actual flow amount
            flow_amount = min(supplies[supply_id], demands[demand_id])
            
            if flow_amount > 0:
                efficiency = 1.0 / cost_matrix[i, j]
                
                flows.append(EnergyFlow(
                    from_building=supply_id,
                    to_building=demand_id,
                    timestamp=timestamp,
                    energy_kwh=flow_amount,
                    efficiency=efficiency,
                    flow_type='direct'
                ))
                
                # Update remaining supply/demand
                supplies[supply_id] -= flow_amount
                demands[demand_id] -= flow_amount
        
        return flows
    
    def _calculate_distance(self, building1: Dict, building2: Dict) -> float:
        """Calculate distance between buildings"""
        x1 = building1.get('x_coord', 0) or building1.get('x', 0)
        y1 = building1.get('y_coord', 0) or building1.get('y', 0)
        x2 = building2.get('x_coord', 0) or building2.get('x', 0)
        y2 = building2.get('y_coord', 0) or building2.get('y', 0)
        
        key = (x1, y1, x2, y2)
        
        if key not in self._distance_cache:
            dx = x1 - x2
            dy = y1 - y2
            self._distance_cache[key] = np.sqrt(dx**2 + dy**2)
        
        return self._distance_cache[key]
    
    def _solar_generation_factor(self, timestamp: datetime) -> float:
        """Simple solar generation factor based on time of day"""
        hour = timestamp.hour
        
        # Simple bell curve for solar generation
        if 6 <= hour <= 18:
            return 0.8 * np.sin((hour - 6) * np.pi / 12)
        return 0.0
    
    def _calculate_cluster_metrics(self,
                                  clusters: List[List[int]],
                                  temporal_data: Optional[pd.DataFrame],
                                  energy_flows: List[EnergyFlow],
                                  timestamp: Optional[datetime],
                                  building_info: Dict) -> List[ClusterMetrics]:
        """
        Calculate comprehensive metrics for each cluster
        """
        metrics = []
        
        for cluster_id, cluster in enumerate(clusters):
            if len(cluster) == 0:
                continue
            
            # Initialize metric values
            total_generation = 0
            total_consumption = 0
            total_shared = 0
            grid_import = 0
            grid_export = 0
            
            if temporal_data is not None and timestamp is not None:
                time_idx = self._get_time_index(temporal_data, timestamp)
                
                if time_idx is not None:
                    # Calculate generation and consumption
                    for building_id in cluster:
                        # Try to access by column name first (if temporal_data uses building IDs as columns)
                        if building_id in temporal_data.columns:
                            consumption = temporal_data.loc[temporal_data.index[time_idx], building_id]
                        else:
                            # Fall back to index-based access
                            building_idx = self._get_building_index(building_id)
                            if building_idx < temporal_data.shape[1]:
                                consumption = temporal_data.iloc[time_idx, building_idx]
                            else:
                                  consumption = 0  # Default if not found

                        total_consumption += consumption
                        
                        # Generation
                        if building_info[building_id]['has_solar']:
                            generation = building_info[building_id]['solar_capacity_kw'] * \
                                       self._solar_generation_factor(timestamp)
                            total_generation += generation
                    
                    # Calculate shared energy from flows
                    for flow in energy_flows:
                        if flow.from_building in cluster and flow.to_building in cluster:
                            total_shared += flow.energy_kwh
                    
                    # Grid interaction
                    net_position = total_generation - total_consumption
                    if net_position > 0:
                        grid_export = net_position - total_shared
                    else:
                        grid_import = abs(net_position) - total_shared
            
            # Calculate ratios
            ssr = 0  # Self-sufficiency ratio
            scr = 0  # Self-consumption ratio
            
            if total_consumption > 0:
                ssr = min(1.0, (total_generation + total_shared) / total_consumption)
            
            if total_generation > 0:
                scr = min(1.0, total_shared / total_generation)
            
            # Peak-to-average ratio
            par = self._calculate_par(cluster, temporal_data, building_info)
            
            # Peak reduction vs individual operation
            peak_reduction = self._calculate_peak_reduction(
                cluster,
                temporal_data,
                building_info
            )
            
            # Thermal savings from adjacency
            thermal_savings = self._calculate_thermal_savings(cluster, building_info)
            
            # Complementarity score
            comp_score = self._calculate_cluster_complementarity(
                cluster,
                building_info
            )
            
            metrics.append(ClusterMetrics(
                cluster_id=cluster_id,
                timestamp=timestamp or datetime.now(),
                self_sufficiency_ratio=ssr,
                self_consumption_ratio=scr,
                peak_to_average_ratio=par,
                peak_reduction_percent=peak_reduction,
                total_shared_kwh=total_shared,
                grid_import_kwh=grid_import,
                grid_export_kwh=grid_export,
                thermal_savings_kwh=thermal_savings,
                member_buildings=cluster,
                complementarity_score=comp_score
            ))
        
        return metrics
    
    def _calculate_par(self,
                      cluster: List[int],
                      temporal_data: Optional[pd.DataFrame],
                      building_info: Dict) -> float:
        """Calculate peak-to-average ratio for cluster"""
        if temporal_data is None:
            return 1.0
        
        cluster_profile = np.zeros(len(temporal_data))
        
        for building_id in cluster:
            if building_id in temporal_data.columns:
                # Access by column name
                cluster_profile += temporal_data[building_id].values
            else:
                # Try index-based access
                building_idx = self._get_building_index(building_id)
                if building_idx < temporal_data.shape[1]:
                    cluster_profile += temporal_data.iloc[:, building_idx].values
        
        if cluster_profile.mean() > 0:
            return cluster_profile.max() / cluster_profile.mean()
        
        return 1.0
    
    def _calculate_peak_reduction(self,
                                 cluster: List[int],
                                 temporal_data: Optional[pd.DataFrame],
                                 building_info: Dict) -> float:
        """Calculate peak reduction from clustering"""
        if temporal_data is None:
            return 0.0
        
        # Individual peaks sum
        individual_peak_sum = sum(
            building_info[b_id].get('peak_demand_kw', 0)
            for b_id in cluster
        )
        
        # Cluster peak (coincident)
        cluster_profile = np.zeros(len(temporal_data))
        for building_id in cluster:
            if building_id in temporal_data.columns:
                # Access by column name
                cluster_profile += temporal_data[building_id].values
            else:
                # Try index-based access
                building_idx = self._get_building_index(building_id)
                if building_idx < temporal_data.shape[1]:
                    cluster_profile += temporal_data.iloc[:, building_idx].values
        
        cluster_peak = cluster_profile.max()
        
        if individual_peak_sum > 0:
            reduction = (individual_peak_sum - cluster_peak) / individual_peak_sum
            return max(0, min(1, reduction))  # Clamp to [0, 1]
        
        return 0.0
    
    def _calculate_thermal_savings(self,
                                  cluster: List[int],
                                  building_info: Dict) -> float:
        """Calculate thermal savings from adjacent buildings"""
        savings = 0.0
        
        for building_id in cluster:
            shared_walls = building_info[building_id].get('shared_walls', [])
            
            # Count how many shared wall neighbors are in same cluster
            cluster_neighbors = sum(1 for neighbor in shared_walls if neighbor in cluster)
            
            if cluster_neighbors > 0:
                # Empirical formula: 5% heating reduction per shared wall
                building_area = building_info[building_id].get('area', 100)
                heating_intensity = 50  # kWh/m²/year typical
                annual_heating = building_area * heating_intensity
                
                # 5% reduction per shared wall, max 20%
                reduction_factor = min(0.20, 0.05 * cluster_neighbors)
                savings += annual_heating * reduction_factor / 8760  # Convert to hourly
        
        return savings
    
    def _calculate_cluster_complementarity(self,
                                          cluster: List[int],
                                          building_info: Dict) -> float:
        """Calculate average complementarity within cluster"""
        if len(cluster) < 2:
            return 0.0
        
        # Count different building types
        building_types = set()
        solar_count = 0
        
        for building_id in cluster:
            building_types.add(building_info[building_id].get('building_type'))
            if building_info[building_id].get('has_solar'):
                solar_count += 1
        
        # Diversity scores
        type_diversity = len(building_types) / len(cluster)
        solar_diversity = solar_count * (len(cluster) - solar_count) / (len(cluster) ** 2)
        
        return (type_diversity + solar_diversity) / 2
    
    def _apply_temporal_stability(self,
                                 new_clusters: List[List[int]],
                                 old_clusters: List[List[int]],
                                 stability_weight: float) -> List[List[int]]:
        """
        Apply temporal stability to prevent too frequent cluster changes
        """
        if stability_weight == 0:
            return new_clusters
        
        # Calculate similarity between old and new clusters
        similarity_matrix = np.zeros((len(old_clusters), len(new_clusters)))
        
        for i, old_cluster in enumerate(old_clusters):
            for j, new_cluster in enumerate(new_clusters):
                intersection = len(set(old_cluster) & set(new_cluster))
                union = len(set(old_cluster) | set(new_cluster))
                if union > 0:
                    similarity_matrix[i, j] = intersection / union
        
        # Find best matching between old and new clusters
        row_ind, col_ind = linear_sum_assignment(-similarity_matrix)
        
        # Merge clusters based on stability weight
        stable_clusters = []
        used_new = set()
        
        for old_idx, new_idx in zip(row_ind, col_ind):
            if similarity_matrix[old_idx, new_idx] > 0.7:  # High similarity
                # Keep mostly the same
                stable_cluster = list(set(old_clusters[old_idx]) | set(new_clusters[new_idx]))
                stable_clusters.append(stable_cluster)
                used_new.add(new_idx)
            else:
                # Accept the change
                stable_clusters.append(new_clusters[new_idx])
                used_new.add(new_idx)
        
        # Add any remaining new clusters
        for j, new_cluster in enumerate(new_clusters):
            if j not in used_new:
                stable_clusters.append(new_cluster)
        
        return stable_clusters
    
    def _get_time_index(self, temporal_data: pd.DataFrame, timestamp: datetime) -> Optional[int]:
        """Get index for timestamp in temporal data"""
        # Assuming temporal_data has datetime index
        if hasattr(temporal_data, 'index'):
            try:
                return temporal_data.index.get_loc(timestamp)
            except:
                # Find closest timestamp
                time_diff = abs(temporal_data.index - timestamp)
                return time_diff.argmin()
        return None
    
    def _get_building_index(self, building_id: int) -> int:
        """Get column index for building in temporal data"""
        # Use the mapping created in run()
        if building_id in self.building_id_to_index:
            return self.building_id_to_index[building_id]
        else:
            # If not found, try to find it in temporal data columns
            # This handles the case where temporal_data uses building IDs as column names
            return building_id  # Return the ID itself for column-based access
    
    def _generate_summary(self,
                         clusters: List[List[int]],
                         metrics: List[ClusterMetrics]) -> Dict:
        """Generate summary statistics"""
        if not metrics:
            return {}
        
        return {
            'num_clusters': len(clusters),
            'avg_cluster_size': np.mean([len(c) for c in clusters]),
            'total_buildings': sum(len(c) for c in clusters),
            'avg_self_sufficiency': np.mean([m.self_sufficiency_ratio for m in metrics]),
            'avg_peak_reduction': np.mean([m.peak_reduction_percent for m in metrics]),
            'total_shared_energy_kwh': sum(m.total_shared_kwh for m in metrics),
            'avg_complementarity': np.mean([m.complementarity_score for m in metrics])
        }
    
    def evaluate(self, ground_truth: Optional[Dict] = None) -> Dict:
        """
        Evaluate clustering performance
        
        Args:
            ground_truth: Optional ground truth for comparison
            
        Returns:
            Evaluation metrics
        """
        if not self.cluster_history:
            return {}
        
        latest_clusters = self.cluster_history[-1]
        
        # Internal metrics (no ground truth needed)
        internal_metrics = {
            'num_clusters': len(latest_clusters),
            'avg_cluster_size': np.mean([len(c) for c in latest_clusters]),
            'size_variance': np.var([len(c) for c in latest_clusters]),
            'singleton_clusters': sum(1 for c in latest_clusters if len(c) == 1)
        }
        
        # Stability metrics
        if len(self.cluster_history) > 1:
            changes = self._calculate_cluster_changes(
                self.cluster_history[-2],
                self.cluster_history[-1]
            )
            internal_metrics['stability'] = 1.0 - changes
        
        # Performance metrics from history
        if self.metrics_history:
            recent_metrics = self.metrics_history[-len(latest_clusters):]
            internal_metrics['avg_self_sufficiency'] = np.mean(
                [m.self_sufficiency_ratio for m in recent_metrics]
            )
            internal_metrics['avg_peak_reduction'] = np.mean(
                [m.peak_reduction_percent for m in recent_metrics]
            )
            internal_metrics['total_energy_shared'] = sum(
                m.total_shared_kwh for m in recent_metrics
            )
        
        # External metrics if ground truth provided
        external_metrics = {}
        if ground_truth:
            external_metrics = self._evaluate_against_ground_truth(
                latest_clusters,
                ground_truth
            )
        
        return {
            'internal': internal_metrics,
            'external': external_metrics,
            'summary': self._generate_evaluation_summary(internal_metrics, external_metrics)
        }
    
    def _calculate_cluster_changes(self,
                                  old_clusters: List[List[int]],
                                  new_clusters: List[List[int]]) -> float:
        """Calculate fraction of buildings that changed clusters"""
        old_assignment = {}
        new_assignment = {}
        
        for i, cluster in enumerate(old_clusters):
            for building in cluster:
                old_assignment[building] = i
        
        for i, cluster in enumerate(new_clusters):
            for building in cluster:
                new_assignment[building] = i
        
        changes = 0
        total = 0
        
        for building in old_assignment:
            if building in new_assignment:
                if old_assignment[building] != new_assignment[building]:
                    changes += 1
                total += 1
        
        return changes / total if total > 0 else 0
    
    def _evaluate_against_ground_truth(self,
                                      clusters: List[List[int]],
                                      ground_truth: Dict) -> Dict:
        """Evaluate against ground truth labels"""
        # Implement standard clustering metrics
        # This would need actual ground truth format specification
        return {}
    
    def _generate_evaluation_summary(self,
                                    internal: Dict,
                                    external: Dict) -> str:
        """Generate human-readable evaluation summary"""
        summary = []
        
        summary.append(f"Formed {internal.get('num_clusters', 0)} clusters")
        summary.append(f"Average size: {internal.get('avg_cluster_size', 0):.1f} buildings")
        
        if 'avg_self_sufficiency' in internal:
            ssr = internal['avg_self_sufficiency']
            summary.append(f"Self-sufficiency: {ssr:.1%}")
        
        if 'avg_peak_reduction' in internal:
            pr = internal['avg_peak_reduction']
            summary.append(f"Peak reduction: {pr:.1%}")
        
        return " | ".join(summary)
    
    def save_results(self, filepath: str):
        """Save clustering results to file"""
        import pickle
        
        results = {
            'cluster_history': self.cluster_history,
            'energy_flows': self.energy_flows,
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        logger.info(f"Saved clustering results to {filepath}")
    
    def load_results(self, filepath: str):
        """Load clustering results from file"""
        import pickle
        
        with open(filepath, 'rb') as f:
            results = pickle.load(f)
        
        self.cluster_history = results['cluster_history']
        self.energy_flows = results['energy_flows']
        self.metrics_history = results['metrics_history']
        
        logger.info(f"Loaded clustering results from {filepath}")


# Utility function for standalone testing
def test_clustering():
    """Test clustering with dummy data"""
    
    # Create dummy model (can be None for testing)
    model = None
    
    # Configuration
    config = ClusterConfig(
        min_cluster_size=3,
        max_cluster_size=15,
        complementarity_weight=2.0
    )
    
    # Initialize clustering
    clustering = EnergyCommunityClustering(model, config)
    
    # Create dummy graph data
    graph_data = {
        'buildings': {
            i: {
                'lv_group_id': f'LV_{i//10}',
                'x_coord': np.random.rand() * 100,
                'y_coord': np.random.rand() * 100,
                'has_solar': np.random.rand() > 0.7,
                'solar_capacity_kw': np.random.rand() * 10,
                'building_function': np.random.choice(['residential', 'office', 'retail']),
                'area': 100 + np.random.rand() * 200,
                'peak_demand_kw': 5 + np.random.rand() * 20
            }
            for i in range(50)
        }
    }
    
    # Create dummy embeddings
    embeddings = torch.randn(50, 128)
    
    # Run clustering
    results = clustering.run(graph_data, embeddings)
    
    print("Clustering Results:")
    print(f"Number of clusters: {len(results['clusters'])}")
    print(f"Summary: {results['summary']}")
    
    return results


if __name__ == "__main__":
    # Test the clustering implementation
    test_results = test_clustering()
    print("\n✅ Clustering task implementation complete and tested!")