import numpy as np
import networkx as nx
from networkx.algorithms import community
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_method import BaseClusteringMethod

class LouvainMethod(BaseClusteringMethod):
    def __init__(self, resolution=1.0, random_state=42):
        super().__init__("Louvain Algorithm")
        self.resolution = resolution
        self.random_state = random_state
        self.modularity = None
        self.communities = None
        
    def fit_predict(self, input_data):
        consumption = input_data['consumption']
        grid_topology = input_data['grid_topology']
        
        n_buildings = consumption.shape[1]
        
        G = nx.Graph()
        
        if grid_topology.number_of_nodes() == n_buildings:
            G = grid_topology.copy()
        else:
            for i in range(n_buildings):
                G.add_node(i)
            for i in range(n_buildings):
                for j in range(i+1, min(i+6, n_buildings)):
                    G.add_edge(i, j)
        
        for i, j in G.edges():
            corr = np.corrcoef(consumption[:, i], consumption[:, j])[0, 1]
            
            weight = 1.0 - abs(corr) if corr < 0 else 0.1
            G[i][j]['weight'] = weight
        
        self.communities = community.louvain_communities(
            G, 
            weight='weight',
            resolution=self.resolution,
            seed=self.random_state
        )
        
        clusters = np.zeros(n_buildings, dtype=int)
        for comm_id, comm in enumerate(self.communities):
            for node in comm:
                if node < n_buildings:
                    clusters[node] = comm_id
        
        self.modularity = community.modularity(G, self.communities, weight='weight')
        
        return clusters
    
    def get_additional_info(self):
        if self.communities is not None:
            return {
                'modularity': float(self.modularity),
                'n_communities': len(self.communities),
                'community_sizes': [len(c) for c in self.communities],
                'resolution': self.resolution
            }
        return {}