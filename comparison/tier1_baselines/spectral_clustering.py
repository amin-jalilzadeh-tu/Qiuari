import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_method import BaseClusteringMethod

class SpectralMethod(BaseClusteringMethod):
    def __init__(self, n_clusters=10, random_state=42):
        super().__init__("Spectral Clustering")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.spectral = None
        self.affinity_matrix = None
        
    def fit_predict(self, input_data):
        consumption = input_data['consumption']
        grid_topology = input_data['grid_topology']
        
        n_buildings = consumption.shape[1]
        
        consumption_similarity = cosine_similarity(consumption.T)
        
        adjacency_matrix = nx.adjacency_matrix(grid_topology).todense()
        if adjacency_matrix.shape[0] != n_buildings:
            adjacency_matrix = np.eye(n_buildings)
        
        self.affinity_matrix = 0.7 * consumption_similarity + 0.3 * adjacency_matrix
        
        self.affinity_matrix = (self.affinity_matrix + self.affinity_matrix.T) / 2
        np.fill_diagonal(self.affinity_matrix, 0)
        
        self.spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='precomputed',
            random_state=self.random_state,
            n_init=10
        )
        
        clusters = self.spectral.fit_predict(self.affinity_matrix)
        
        return clusters
    
    def get_additional_info(self):
        if self.affinity_matrix is not None:
            eigenvalues = np.linalg.eigvalsh(self.affinity_matrix)
            return {
                'affinity_matrix_shape': self.affinity_matrix.shape,
                'top_eigenvalues': eigenvalues[-10:].tolist() if len(eigenvalues) >= 10 else eigenvalues.tolist(),
                'spectral_gap': float(eigenvalues[-1] - eigenvalues[-2]) if len(eigenvalues) >= 2 else 0
            }
        return {}