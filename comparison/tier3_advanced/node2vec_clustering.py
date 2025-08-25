import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_method import BaseClusteringMethod

class Node2VecMethod(BaseClusteringMethod):
    def __init__(self, n_clusters=10, embedding_dim=32, walk_length=5, num_walks=20, p=1, q=1, random_state=42):
        super().__init__("Node2vec + Clustering")
        self.n_clusters = n_clusters
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.random_state = random_state
        self.embeddings = None
        
    def fit_predict(self, input_data):
        consumption = input_data['consumption']
        grid_topology = input_data['grid_topology']
        
        n_buildings = consumption.shape[1]
        
        if grid_topology.number_of_nodes() != n_buildings:
            G = nx.watts_strogatz_graph(n_buildings, 6, 0.3)
        else:
            G = grid_topology.copy()
        
        for i, j in G.edges():
            corr = np.corrcoef(consumption[:, i], consumption[:, j])[0, 1]
            weight = 1.0 + abs(corr)
            G[i][j]['weight'] = weight
        
        self.embeddings = self._node2vec_embed(G)
        
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(self.embeddings)
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        clusters = kmeans.fit_predict(embeddings_scaled)
        
        return clusters
    
    def _node2vec_embed(self, G):
        n_nodes = G.number_of_nodes()
        
        walks = self._generate_walks(G)
        
        embeddings = self._learn_embeddings(walks, n_nodes)
        
        return embeddings
    
    def _generate_walks(self, G):
        walks = []
        nodes = list(G.nodes())
        
        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self._node2vec_walk(G, node)
                walks.append(walk)
        
        return walks
    
    def _node2vec_walk(self, G, start_node):
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            cur = walk[-1]
            neighbors = list(G.neighbors(cur))
            
            if len(neighbors) == 0:
                break
            
            if len(walk) == 1:
                walk.append(np.random.choice(neighbors))
            else:
                prev = walk[-2]
                next_node = self._get_next_node(G, prev, cur, neighbors)
                walk.append(next_node)
        
        return walk
    
    def _get_next_node(self, G, prev, cur, neighbors):
        probs = []
        
        for neighbor in neighbors:
            if neighbor == prev:
                prob = 1 / self.p
            elif G.has_edge(neighbor, prev):
                prob = 1
            else:
                prob = 1 / self.q
            
            if G.has_edge(cur, neighbor) and 'weight' in G[cur][neighbor]:
                prob *= G[cur][neighbor]['weight']
            
            probs.append(prob)
        
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        return np.random.choice(neighbors, p=probs)
    
    def _learn_embeddings(self, walks, n_nodes):
        from collections import defaultdict
        
        vocab = defaultdict(int)
        for walk in walks:
            for node in walk:
                vocab[node] += 1
        
        embeddings = np.random.randn(n_nodes, self.embedding_dim) * 0.01
        
        window_size = 5
        learning_rate = 0.025
        min_learning_rate = 0.0001
        num_epochs = 2
        
        for epoch in range(num_epochs):
            lr = learning_rate * (1 - epoch / num_epochs) + min_learning_rate
            
            for walk in walks:
                for i, center_node in enumerate(walk):
                    context_start = max(0, i - window_size)
                    context_end = min(len(walk), i + window_size + 1)
                    
                    for j in range(context_start, context_end):
                        if i != j:
                            context_node = walk[j]
                            
                            dot_product = np.dot(embeddings[center_node], embeddings[context_node])
                            sigmoid = 1 / (1 + np.exp(-dot_product))
                            
                            gradient = lr * (1 - sigmoid)
                            embeddings[center_node] += gradient * embeddings[context_node]
                            embeddings[context_node] += gradient * embeddings[center_node]
                            
                            neg_samples = np.random.choice(n_nodes, 5)
                            for neg_node in neg_samples:
                                if neg_node not in walk[context_start:context_end]:
                                    dot_product = np.dot(embeddings[center_node], embeddings[neg_node])
                                    sigmoid = 1 / (1 + np.exp(-dot_product))
                                    
                                    gradient = lr * sigmoid
                                    embeddings[center_node] -= gradient * embeddings[neg_node]
                                    embeddings[neg_node] -= gradient * embeddings[center_node]
        
        return embeddings
    
    def get_additional_info(self):
        if self.embeddings is not None:
            return {
                'embedding_dim': self.embedding_dim,
                'embeddings_shape': self.embeddings.shape,
                'walk_length': self.walk_length,
                'num_walks': self.num_walks,
                'p': self.p,
                'q': self.q,
                'avg_embedding_norm': float(np.linalg.norm(self.embeddings, axis=1).mean())
            }
        return {}