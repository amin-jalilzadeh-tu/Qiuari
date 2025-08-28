import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.base_method import BaseClusteringMethod

class KMeansMethod(BaseClusteringMethod):
    def __init__(self, n_clusters=10, random_state=42):
        super().__init__("K-means Clustering")
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        
    def fit_predict(self, input_data):
        consumption = input_data['consumption']
        
        X = consumption.T
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        
        clusters = self.kmeans.fit_predict(X_scaled)
        
        return clusters
    
    def get_additional_info(self):
        if self.kmeans is not None:
            return {
                'inertia': float(self.kmeans.inertia_),
                'n_iter': int(self.kmeans.n_iter_),
                'cluster_centers_shape': self.kmeans.cluster_centers_.shape
            }
        return {}