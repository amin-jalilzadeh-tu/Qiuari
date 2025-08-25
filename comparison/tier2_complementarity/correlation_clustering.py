import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_method import BaseClusteringMethod

class CorrelationClusteringMethod(BaseClusteringMethod):
    def __init__(self, n_clusters=10):
        super().__init__("Correlation-Based Clustering (Anti-correlation)")
        self.n_clusters = n_clusters
        self.correlation_matrix = None
        self.distance_matrix = None
        
    def fit_predict(self, input_data):
        consumption = input_data['consumption']
        generation = input_data.get('generation', np.zeros_like(consumption))
        
        net_load = consumption - generation
        
        n_buildings = net_load.shape[1]
        self.correlation_matrix = np.corrcoef(net_load.T)
        
        self.distance_matrix = np.zeros_like(self.correlation_matrix)
        for i in range(n_buildings):
            for j in range(n_buildings):
                corr = self.correlation_matrix[i, j]
                if i == j:
                    self.distance_matrix[i, j] = 0
                elif corr < 0:
                    self.distance_matrix[i, j] = 1 - abs(corr)
                else:
                    self.distance_matrix[i, j] = 2
        
        condensed_distance = squareform(self.distance_matrix)
        
        linkage_matrix = linkage(condensed_distance, method='ward')
        
        clusters = fcluster(linkage_matrix, self.n_clusters, criterion='maxclust') - 1
        
        return clusters
    
    def get_additional_info(self):
        if self.correlation_matrix is not None:
            negative_corrs = self.correlation_matrix[self.correlation_matrix < 0]
            return {
                'avg_negative_correlation': float(negative_corrs.mean()) if len(negative_corrs) > 0 else 0,
                'n_negative_pairs': len(negative_corrs) // 2,
                'strongest_anti_correlation': float(negative_corrs.min()) if len(negative_corrs) > 0 else 0,
                'complementarity_score': float(abs(negative_corrs).mean()) if len(negative_corrs) > 0 else 0
            }
        return {}