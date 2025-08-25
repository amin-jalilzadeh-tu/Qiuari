import numpy as np
from itertools import combinations
from scipy.stats import entropy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_method import BaseClusteringMethod

class InformationSynergyMethod(BaseClusteringMethod):
    def __init__(self, n_clusters=10, n_bins=10, max_group_size=4):
        super().__init__("Information-Theoretic Synergy")
        self.n_clusters = n_clusters
        self.n_bins = n_bins
        self.max_group_size = max_group_size
        self.synergy_scores = None
        
    def fit_predict(self, input_data):
        consumption = input_data['consumption']
        generation = input_data.get('generation', np.zeros_like(consumption))
        
        net_load = consumption - generation
        n_buildings = net_load.shape[1]
        
        discretized = self._discretize_signals(net_load)
        
        self.synergy_scores = {}
        best_groups = []
        
        sample_size = min(n_buildings, 30)
        sample_indices = np.random.choice(n_buildings, sample_size, replace=False)
        
        for size in range(2, min(self.max_group_size + 1, sample_size + 1)):
            for group in combinations(sample_indices, size):
                synergy = self._calculate_synergy(discretized, group)
                self.synergy_scores[group] = synergy
                best_groups.append((synergy, group))
        
        best_groups.sort(key=lambda x: x[0], reverse=True)
        
        clusters = np.full(n_buildings, -1, dtype=int)
        cluster_id = 0
        used_buildings = set()
        
        for synergy, group in best_groups[:self.n_clusters * 2]:
            if all(b not in used_buildings for b in group):
                for b in group:
                    clusters[b] = cluster_id
                    used_buildings.add(b)
                cluster_id += 1
                
                if cluster_id >= self.n_clusters:
                    break
        
        for i in range(n_buildings):
            if clusters[i] == -1:
                if cluster_id < self.n_clusters:
                    clusters[i] = cluster_id
                    cluster_id += 1
                else:
                    clusters[i] = np.random.randint(0, self.n_clusters)
        
        return clusters
    
    def _discretize_signals(self, signals):
        discretized = np.zeros_like(signals, dtype=int)
        for i in range(signals.shape[1]):
            bins = np.histogram_bin_edges(signals[:, i], bins=self.n_bins)
            discretized[:, i] = np.digitize(signals[:, i], bins) - 1
        return discretized
    
    def _calculate_synergy(self, discretized, group):
        if len(group) < 2:
            return 0
        
        aggregate = discretized[:, list(group)].sum(axis=1)
        
        H_aggregate = self._entropy(aggregate)
        
        H_individuals = sum(self._entropy(discretized[:, i]) for i in group)
        
        synergy = H_aggregate - H_individuals / len(group)
        
        if len(group) >= 3:
            for subset in combinations(group, 2):
                subset_aggregate = discretized[:, list(subset)].sum(axis=1)
                H_subset = self._entropy(subset_aggregate)
                synergy += (H_aggregate - H_subset) / (len(group) - 1)
        
        return synergy
    
    def _entropy(self, signal):
        _, counts = np.unique(signal, return_counts=True)
        probs = counts / len(signal)
        return entropy(probs, base=2)
    
    def _mutual_information(self, x, y):
        xy = np.column_stack([x, y])
        unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
        p_xy = counts_xy / len(xy)
        
        _, counts_x = np.unique(x, return_counts=True)
        p_x = counts_x / len(x)
        
        _, counts_y = np.unique(y, return_counts=True)
        p_y = counts_y / len(y)
        
        mi = 0
        for i, (xi, yi) in enumerate(unique_xy):
            p_joint = p_xy[i]
            p_xi = p_x[xi] if xi < len(p_x) else 1/len(x)
            p_yi = p_y[yi] if yi < len(p_y) else 1/len(y)
            if p_joint > 0 and p_xi > 0 and p_yi > 0:
                mi += p_joint * np.log2(p_joint / (p_xi * p_yi))
        
        return max(0, mi)
    
    def get_additional_info(self):
        if self.synergy_scores is not None:
            all_scores = list(self.synergy_scores.values())
            positive_scores = [s for s in all_scores if s > 0]
            
            return {
                'avg_synergy': float(np.mean(all_scores)) if all_scores else 0,
                'max_synergy': float(np.max(all_scores)) if all_scores else 0,
                'n_positive_synergies': len(positive_scores),
                'n_groups_evaluated': len(self.synergy_scores),
                'n_bins': self.n_bins
            }
        return {}