"""
Method 3: Information-Theoretic Synergy Clustering
Based on: "Information-Theoretic Measures for Microgrid Clustering" 
(Ghosh et al., IEEE Trans. Sustainable Energy, 2021)

Calculate multi-information synergy:
1. Discretize load profiles into states
2. Calculate entropy reduction from aggregation
3. Find groups with maximum synergy
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy.stats import entropy
from sklearn.preprocessing import KBinsDiscretizer
from itertools import combinations
from collections import defaultdict
from base_method import BaseClusteringMethod

logger = logging.getLogger(__name__)

class InformationSynergyClusteringKG(BaseClusteringMethod):
    """
    Information-theoretic clustering based on synergy in load aggregation.
    Finds building groups that maximize entropy reduction.
    """
    
    def __init__(self, n_bins: int = 10, max_cluster_size: int = 8,
                 synergy_threshold: float = 0.2):
        """
        Initialize Information Synergy Clustering.
        
        Args:
            n_bins: Number of bins for discretizing continuous values
            max_cluster_size: Maximum buildings per cluster
            synergy_threshold: Minimum synergy to form cluster
        """
        super().__init__(
            name="Information-Theoretic Synergy Clustering",
            paper_reference="Ghosh et al., IEEE Trans. Sustainable Energy, 2021"
        )
        
        self.n_bins = n_bins
        self.max_cluster_size = max_cluster_size
        self.synergy_threshold = synergy_threshold
        
        self.discretized_profiles = None
        self.synergy_matrix = None
        
        logger.info(f"Initialized with n_bins={n_bins}, max_cluster_size={max_cluster_size}, "
                   f"synergy_threshold={synergy_threshold}")
    
    def _perform_clustering(self, **kwargs) -> Dict[str, List[str]]:
        """
        Perform information-theoretic synergy clustering.
        
        Returns:
            Dictionary mapping cluster_id -> list of building_ids
        """
        # Update parameters if provided
        self.n_bins = kwargs.get('n_bins', self.n_bins)
        self.max_cluster_size = kwargs.get('max_cluster_size', self.max_cluster_size)
        self.synergy_threshold = kwargs.get('synergy_threshold', self.synergy_threshold)
        
        # Discretize time series data
        self.discretized_profiles = self._discretize_time_series()
        
        # Calculate pairwise synergy
        self.synergy_matrix = self._calculate_synergy_matrix()
        
        # Perform greedy synergy search
        clusters = self._greedy_synergy_search()
        
        return clusters
    
    def _discretize_time_series(self) -> Dict[str, np.ndarray]:
        """
        Discretize continuous load profiles into states.
        From paper's Section III: Convert to discrete states for entropy calculation.
        """
        logger.info(f"Discretizing time series into {self.n_bins} bins...")
        
        time_series = self.preprocessed_data['time_series']
        discretized = {}
        
        # Collect all demand values for global binning
        all_demands = []
        for bid, ts in time_series.items():
            if len(ts) > 0:
                # Use electricity demand (column 3)
                all_demands.extend(ts[:, 3])
        
        if not all_demands:
            logger.warning("No demand data found for discretization")
            return {}
        
        # Create bins based on global distribution
        all_demands = np.array(all_demands).reshape(-1, 1)
        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, 
            encode='ordinal', 
            strategy='quantile'
        )
        discretizer.fit(all_demands)
        
        # Discretize each building's profile
        for bid, ts in time_series.items():
            if len(ts) > 0:
                demand = ts[:, 3].reshape(-1, 1)
                discretized[bid] = discretizer.transform(demand).flatten().astype(int)
            else:
                # Empty profile
                discretized[bid] = np.zeros(1, dtype=int)
        
        logger.info(f"Discretized {len(discretized)} building profiles")
        
        return discretized
    
    def _calculate_synergy_with_constraints(self, group: List[str]) -> float:
        """
        Calculate synergy for a group with KG constraints.
        Paper's Equation (12) with KG constraints:
        S(G) = H(Σ_i∈G X_i) - (1/|G|) * Σ_i∈G H(X_i)
        
        Additional constraints:
        - Return -∞ if group spans multiple CableGroups
        - Add bonus for building type diversity from KG
        """
        if len(group) < 2:
            return 0.0
        
        # Check cable group constraint
        constraints = self.preprocessed_data['constraints']
        bid_to_idx = constraints['bid_to_idx']
        
        cable_groups = set()
        for bid in group:
            if bid in bid_to_idx:
                idx = bid_to_idx[bid]
                for cg_id, cg_indices in constraints['cable_groups'].items():
                    if idx in cg_indices:
                        cable_groups.add(cg_id)
                        break
        
        # Violates constraint if spans multiple cable groups
        if len(cable_groups) > 1:
            return -np.inf
        
        # Calculate synergy
        profiles = []
        for bid in group:
            if bid in self.discretized_profiles:
                profiles.append(self.discretized_profiles[bid])
        
        if len(profiles) < 2:
            return 0.0
        
        # Ensure all profiles have same length
        min_len = min(len(p) for p in profiles)
        profiles = [p[:min_len] for p in profiles]
        
        # Calculate individual entropies
        individual_entropies = []
        for profile in profiles:
            # Calculate probability distribution
            counts = np.bincount(profile, minlength=self.n_bins)
            probs = counts / len(profile)
            individual_entropies.append(entropy(probs))
        
        # Calculate aggregated entropy
        # Sum profiles (represents aggregated load)
        aggregated = np.sum(profiles, axis=0)
        
        # Discretize aggregated profile
        # Use wider range for aggregated values
        agg_min, agg_max = aggregated.min(), aggregated.max()
        if agg_max > agg_min:
            agg_bins = np.linspace(agg_min, agg_max, self.n_bins + 1)
            agg_discretized = np.digitize(aggregated, agg_bins) - 1
            agg_discretized = np.clip(agg_discretized, 0, self.n_bins - 1)
        else:
            agg_discretized = np.zeros_like(aggregated)
        
        # Calculate aggregated entropy
        agg_counts = np.bincount(agg_discretized, minlength=self.n_bins)
        agg_probs = agg_counts / len(agg_discretized)
        agg_entropy = entropy(agg_probs)
        
        # Calculate synergy (entropy reduction)
        avg_individual_entropy = np.mean(individual_entropies)
        synergy = avg_individual_entropy - agg_entropy / len(group)
        
        # Add diversity bonus
        building_features = self.preprocessed_data['building_features']
        group_features = building_features[building_features['ogc_fid'].isin(group)]
        
        if len(group_features) > 0:
            # Diversity in building types
            type_diversity = len(group_features['building_function'].unique()) / len(group)
            
            # Diversity in assets
            has_solar = group_features['has_solar'].sum() > 0
            has_battery = group_features['has_battery'].sum() > 0
            has_no_assets = (~group_features['has_solar'] & ~group_features['has_battery']).sum() > 0
            
            asset_diversity = (int(has_solar) + int(has_battery) + int(has_no_assets)) / 3
            
            # Apply diversity bonus
            diversity_bonus = 0.1 * (type_diversity + asset_diversity) / 2
            synergy += diversity_bonus
        
        return synergy
    
    def _calculate_synergy_matrix(self) -> np.ndarray:
        """
        Calculate pairwise synergy for all building pairs.
        """
        logger.info("Calculating pairwise synergy matrix...")
        
        building_ids = list(self.discretized_profiles.keys())
        n_buildings = len(building_ids)
        synergy_matrix = np.zeros((n_buildings, n_buildings))
        
        for i in range(n_buildings):
            for j in range(i + 1, n_buildings):
                pair = [building_ids[i], building_ids[j]]
                synergy = self._calculate_synergy_with_constraints(pair)
                
                if synergy != -np.inf:
                    synergy_matrix[i, j] = synergy
                    synergy_matrix[j, i] = synergy
        
        logger.info(f"Synergy matrix calculated. Mean: {np.mean(synergy_matrix[synergy_matrix > 0]):.3f}, "
                   f"Max: {np.max(synergy_matrix):.3f}")
        
        return synergy_matrix
    
    def _greedy_synergy_search(self) -> Dict[str, List[str]]:
        """
        Greedy algorithm to find groups with maximum synergy.
        Algorithm from Section IV-B:
        1. Start with highest synergy pairs
        2. Expand groups while synergy increases
        3. Respect transformer capacity limits
        """
        logger.info("Performing greedy synergy search...")
        
        building_ids = list(self.discretized_profiles.keys())
        n_buildings = len(building_ids)
        
        # Track which buildings are already clustered
        clustered = set()
        clusters = {}
        cluster_id = 0
        
        # Find all pairs with synergy above threshold
        synergy_pairs = []
        for i in range(n_buildings):
            for j in range(i + 1, n_buildings):
                if self.synergy_matrix[i, j] > self.synergy_threshold:
                    synergy_pairs.append((
                        self.synergy_matrix[i, j],
                        building_ids[i],
                        building_ids[j]
                    ))
        
        # Sort by synergy (descending)
        synergy_pairs.sort(reverse=True)
        
        # Start with highest synergy pairs
        for synergy_val, bid1, bid2 in synergy_pairs:
            if bid1 in clustered or bid2 in clustered:
                continue
            
            # Start new cluster
            current_cluster = [bid1, bid2]
            current_synergy = synergy_val
            
            # Try to expand cluster
            improved = True
            while improved and len(current_cluster) < self.max_cluster_size:
                improved = False
                best_addition = None
                best_new_synergy = current_synergy
                
                # Try adding each unclustered building
                for bid in building_ids:
                    if bid in clustered or bid in current_cluster:
                        continue
                    
                    # Calculate synergy with addition
                    test_cluster = current_cluster + [bid]
                    new_synergy = self._calculate_synergy_with_constraints(test_cluster)
                    
                    # Check if this improves synergy
                    if new_synergy > best_new_synergy:
                        # Also check transformer capacity
                        if self._check_cluster_capacity(test_cluster):
                            best_addition = bid
                            best_new_synergy = new_synergy
                            improved = True
                
                # Add best building if found
                if improved and best_addition:
                    current_cluster.append(best_addition)
                    current_synergy = best_new_synergy
            
            # Save cluster if it meets minimum size
            if len(current_cluster) >= 3:
                clusters[f"synergy_{cluster_id}"] = current_cluster
                clustered.update(current_cluster)
                cluster_id += 1
                
                logger.debug(f"Created cluster {cluster_id} with {len(current_cluster)} buildings, "
                           f"synergy={current_synergy:.3f}")
        
        # Handle unclustered buildings
        unclustered = set(building_ids) - clustered
        if unclustered:
            logger.info(f"Handling {len(unclustered)} unclustered buildings...")
            
            # Group by cable group
            constraints = self.preprocessed_data['constraints']
            cable_group_unclustered = defaultdict(list)
            
            for bid in unclustered:
                if bid in constraints['bid_to_idx']:
                    idx = constraints['bid_to_idx'][bid]
                    for cg_id, cg_indices in constraints['cable_groups'].items():
                        if idx in cg_indices:
                            cable_group_unclustered[cg_id].append(bid)
                            break
            
            # Create clusters for unclustered buildings in same cable group
            for cg_id, buildings in cable_group_unclustered.items():
                if len(buildings) >= 3:
                    # Check synergy for this group
                    group_synergy = self._calculate_synergy_with_constraints(buildings)
                    if group_synergy > 0:
                        clusters[f"synergy_unclustered_{cg_id}"] = buildings
                        cluster_id += 1
        
        logger.info(f"Greedy search complete. Created {len(clusters)} clusters")
        
        return clusters
    
    def _check_cluster_capacity(self, cluster: List[str]) -> bool:
        """
        Check if cluster respects transformer capacity.
        """
        constraints = self.preprocessed_data['constraints']
        
        # Find transformer for this cluster
        transformer_id = None
        for t_id, t_buildings in constraints['transformer_groups'].items():
            if all(bid in [str(b) for b in t_buildings] for bid in cluster):
                transformer_id = t_id
                break
        
        if transformer_id:
            return self._check_transformer_capacity(cluster, transformer_id)
        
        return True
    
    def _calculate_interaction_information(self, group: List[str]) -> float:
        """
        Calculate interaction information (multi-information) for a group.
        From paper: I(A;B;C) measures the synergy beyond pairwise interactions.
        """
        if len(group) < 3:
            return 0.0
        
        profiles = []
        for bid in group:
            if bid in self.discretized_profiles:
                profiles.append(self.discretized_profiles[bid])
        
        if len(profiles) < 3:
            return 0.0
        
        # Ensure same length
        min_len = min(len(p) for p in profiles)
        profiles = [p[:min_len] for p in profiles]
        
        # Calculate multi-information using inclusion-exclusion
        n = len(profiles)
        total_mi = 0
        
        # Sum over all non-empty subsets
        for r in range(1, n + 1):
            for subset in combinations(range(n), r):
                # Calculate joint entropy for subset
                if len(subset) == 1:
                    # Single variable entropy
                    profile = profiles[subset[0]]
                    counts = np.bincount(profile, minlength=self.n_bins)
                    probs = counts / len(profile)
                    h = entropy(probs)
                else:
                    # Joint entropy
                    joint_profile = np.column_stack([profiles[i] for i in subset])
                    # Convert to tuple for counting
                    joint_tuples = [tuple(row) for row in joint_profile]
                    unique, counts = np.unique(joint_tuples, return_counts=True, axis=0)
                    probs = counts / len(joint_tuples)
                    h = entropy(probs)
                
                # Add or subtract based on inclusion-exclusion
                sign = (-1) ** (len(subset) + 1)
                total_mi += sign * h
        
        return total_mi
    
    def get_method_specific_metrics(self) -> Dict[str, Any]:
        """
        Get method-specific metrics for information synergy clustering.
        """
        if not self.clusters:
            return {}
        
        metrics = {
            'avg_cluster_synergy': 0,
            'total_entropy_reduction': 0,
            'max_synergy': 0,
            'min_synergy': float('inf')
        }
        
        synergies = []
        for cluster_id, members in self.clusters.items():
            if len(members) >= 2:
                synergy = self._calculate_synergy_with_constraints(members)
                if synergy != -np.inf:
                    synergies.append(synergy)
                    metrics['max_synergy'] = max(metrics['max_synergy'], synergy)
                    metrics['min_synergy'] = min(metrics['min_synergy'], synergy)
        
        if synergies:
            metrics['avg_cluster_synergy'] = np.mean(synergies)
            metrics['total_entropy_reduction'] = np.sum(synergies)
        
        # Add interaction information for larger clusters
        interaction_info = []
        for cluster_id, members in self.clusters.items():
            if len(members) >= 3:
                ii = self._calculate_interaction_information(members)
                interaction_info.append(ii)
        
        if interaction_info:
            metrics['avg_interaction_information'] = np.mean(interaction_info)
        
        return metrics