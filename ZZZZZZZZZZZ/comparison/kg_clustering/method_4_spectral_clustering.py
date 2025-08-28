"""
Method 4: Spectral Clustering with Laplacian Constraints
Based on: "Network-Aware Coordination of Residential Distributed Energy Resources" 
(Papadaskalopoulos et al., IEEE Trans. Smart Grid, 2019)

Construct Laplacian matrix incorporating:
1. Electrical network topology from KG
2. Complementarity weights
3. Physical adjacency benefits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from base_method import BaseClusteringMethod

logger = logging.getLogger(__name__)

class NetworkAwareSpectralClustering(BaseClusteringMethod):
    """
    Spectral clustering that respects electrical network topology.
    Uses graph Laplacian with complementarity weights and electrical constraints.
    """
    
    def __init__(self, n_clusters: int = 10, lambda_decay: float = 0.5,
                 use_normalized_laplacian: bool = True):
        """
        Initialize Network-Aware Spectral Clustering.
        
        Args:
            n_clusters: Number of clusters to form
            lambda_decay: Decay factor for electrical distance (exp(-λ * distance))
            use_normalized_laplacian: Whether to use normalized Laplacian
        """
        super().__init__(
            name="Network-Aware Spectral Clustering",
            paper_reference="Papadaskalopoulos et al., IEEE Trans. Smart Grid, 2019"
        )
        
        self.n_clusters = n_clusters
        self.lambda_decay = lambda_decay
        self.use_normalized_laplacian = use_normalized_laplacian
        
        self.affinity_matrix = None
        self.laplacian = None
        self.eigenvectors = None
        
        logger.info(f"Initialized with n_clusters={n_clusters}, λ={lambda_decay}, "
                   f"normalized={use_normalized_laplacian}")
    
    def _perform_clustering(self, **kwargs) -> Dict[str, List[str]]:
        """
        Perform spectral clustering with network constraints.
        
        Returns:
            Dictionary mapping cluster_id -> list of building_ids
        """
        # Update parameters if provided
        self.n_clusters = kwargs.get('n_clusters', self.n_clusters)
        self.lambda_decay = kwargs.get('lambda_decay', self.lambda_decay)
        self.use_normalized_laplacian = kwargs.get('use_normalized_laplacian', 
                                                   self.use_normalized_laplacian)
        
        # Construct weighted affinity matrix
        self.affinity_matrix = self._construct_weighted_affinity()
        
        # Construct Laplacian
        self.laplacian = self._construct_laplacian()
        
        # Compute eigenvectors
        self.eigenvectors = self._compute_eigenvectors()
        
        # Apply k-means in spectral space
        clusters = self._spectral_kmeans()
        
        # Post-process to ensure feasibility
        clusters = self._ensure_feasibility(clusters)
        
        return clusters
    
    def _construct_weighted_affinity(self) -> np.ndarray:
        """
        Construct affinity matrix with electrical and complementarity weights.
        Paper's Equation (15):
        W_ij = exp(-λ * electrical_distance) * (1 - |corr_ij|) * capacity_factor
        
        Using KG data:
        - electrical_distance from graph traversal
        - correlation from EnergyState time series
        - capacity_factor from Transformer limits
        """
        logger.info("Constructing weighted affinity matrix...")
        
        # Get data
        complementarity = self.preprocessed_data['complementarity']
        electrical_distances = self.preprocessed_data['electrical_distances']
        constraints = self.preprocessed_data['constraints']
        building_features = self.preprocessed_data['building_features']
        adjacency = self.preprocessed_data.get('adjacency', {})
        
        n_buildings = len(building_features)
        affinity = np.zeros((n_buildings, n_buildings))
        
        # Building ID to index mapping
        bid_to_idx = constraints['bid_to_idx']
        idx_to_bid = constraints['idx_to_bid']
        
        # Calculate affinity for each pair
        for i in range(n_buildings):
            for j in range(i + 1, n_buildings):
                # Base complementarity weight
                base_weight = complementarity[i, j]
                
                # Electrical distance factor
                elec_dist = electrical_distances[i, j]
                distance_factor = np.exp(-self.lambda_decay * elec_dist)
                
                # Capacity factor (penalize if different transformer)
                same_cable = constraints['same_cable_group'][i, j]
                same_transformer = constraints['same_transformer'][i, j]
                
                if same_cable:
                    capacity_factor = 1.0
                elif same_transformer:
                    capacity_factor = 0.7
                else:
                    capacity_factor = 0.1  # Heavily penalize cross-transformer
                
                # Physical adjacency bonus
                adjacency_bonus = 0
                if idx_to_bid[i] in adjacency.get('adjacency_matrix', {}):
                    adj_key = tuple(sorted([idx_to_bid[i], idx_to_bid[j]]))
                    if adj_key in adjacency['adjacency_matrix']:
                        adjacency_bonus = 0.2 * adjacency['adjacency_matrix'][adj_key].get(
                            'sharing_potential', 0
                        )
                
                # Calculate final affinity
                affinity[i, j] = (base_weight * distance_factor * capacity_factor + 
                                 adjacency_bonus)
                affinity[j, i] = affinity[i, j]  # Symmetric
        
        # Add small diagonal to ensure positive definiteness
        np.fill_diagonal(affinity, 0.01)
        
        logger.info(f"Affinity matrix constructed. Density: {np.count_nonzero(affinity) / (n_buildings**2):.3f}")
        
        return affinity
    
    def _construct_laplacian(self) -> np.ndarray:
        """
        Construct graph Laplacian from affinity matrix.
        L = D - W where D is degree matrix
        """
        logger.info("Constructing Laplacian matrix...")
        
        # Degree matrix
        degrees = np.sum(self.affinity_matrix, axis=1)
        D = np.diag(degrees)
        
        # Laplacian
        L = D - self.affinity_matrix
        
        if self.use_normalized_laplacian:
            # Normalized Laplacian: L_norm = D^(-1/2) * L * D^(-1/2)
            # Avoid division by zero
            d_sqrt_inv = np.diag(1.0 / np.sqrt(degrees + 1e-10))
            L = d_sqrt_inv @ L @ d_sqrt_inv
        
        logger.info(f"Laplacian constructed. Using {'normalized' if self.use_normalized_laplacian else 'unnormalized'} form")
        
        return L
    
    def _compute_eigenvectors(self) -> np.ndarray:
        """
        Compute eigenvectors of the Laplacian.
        Modified Algorithm 3 from paper:
        1. Compute eigenvectors of constrained Laplacian
        2. Use smallest k eigenvalues (Fiedler vectors)
        """
        logger.info(f"Computing {self.n_clusters} eigenvectors...")
        
        # Convert to sparse for efficiency
        L_sparse = csr_matrix(self.laplacian)
        
        # Compute smallest k eigenvalues and eigenvectors
        # Note: eigsh finds largest eigenvalues, so we use 'SM' for smallest magnitude
        try:
            eigenvalues, eigenvectors = eigsh(
                L_sparse, 
                k=min(self.n_clusters, self.laplacian.shape[0] - 1),
                which='SM',
                maxiter=1000
            )
        except Exception as e:
            logger.warning(f"Sparse eigendecomposition failed: {e}. Using dense method.")
            # Fall back to dense method
            eigenvalues, eigenvectors = np.linalg.eigh(self.laplacian)
            # Take smallest k
            idx = eigenvalues.argsort()[:self.n_clusters]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
        
        # Log spectral gap (indicator of cluster quality)
        if len(eigenvalues) > 1:
            spectral_gap = eigenvalues[1] - eigenvalues[0]
            logger.info(f"Spectral gap: {spectral_gap:.4f}")
        
        return eigenvectors
    
    def _spectral_kmeans(self) -> Dict[str, List[str]]:
        """
        Apply k-means clustering in spectral embedding space.
        Post-process to ensure feasibility.
        """
        logger.info("Applying k-means in spectral space...")
        
        # Normalize rows of eigenvector matrix
        row_norms = np.linalg.norm(self.eigenvectors, axis=1, keepdims=True)
        normalized_eigenvectors = self.eigenvectors / (row_norms + 1e-10)
        
        # Apply k-means
        kmeans = KMeans(
            n_clusters=min(self.n_clusters, len(normalized_eigenvectors)),
            random_state=42,
            n_init=10
        )
        labels = kmeans.fit_predict(normalized_eigenvectors)
        
        # Convert to cluster dictionary
        clusters = {}
        building_features = self.preprocessed_data['building_features']
        building_ids = building_features['ogc_fid'].tolist()
        
        for cluster_idx in range(kmeans.n_clusters):
            cluster_members = [building_ids[i] for i in range(len(labels)) 
                             if labels[i] == cluster_idx]
            if cluster_members:
                clusters[f"spectral_{cluster_idx}"] = cluster_members
        
        logger.info(f"K-means complete. Created {len(clusters)} initial clusters")
        
        return clusters
    
    def _ensure_feasibility(self, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Post-process clusters to ensure electrical feasibility.
        Split clusters that violate constraints.
        """
        logger.info("Ensuring electrical feasibility of clusters...")
        
        constraints = self.preprocessed_data['constraints']
        bid_to_idx = constraints['bid_to_idx']
        
        feasible_clusters = {}
        split_count = 0
        
        for cluster_id, members in clusters.items():
            # Check if cluster violates cable group constraints
            cable_groups = set()
            member_by_cg = defaultdict(list)
            
            for bid in members:
                if bid in bid_to_idx:
                    idx = bid_to_idx[bid]
                    for cg_id, cg_indices in constraints['cable_groups'].items():
                        if idx in cg_indices:
                            cable_groups.add(cg_id)
                            member_by_cg[cg_id].append(bid)
                            break
            
            if len(cable_groups) <= 1:
                # No violation, keep cluster
                feasible_clusters[cluster_id] = members
            else:
                # Split by cable group
                logger.debug(f"Splitting cluster {cluster_id} across {len(cable_groups)} cable groups")
                split_count += 1
                
                for i, (cg_id, cg_members) in enumerate(member_by_cg.items()):
                    if len(cg_members) >= 3:  # Minimum cluster size
                        new_id = f"{cluster_id}_cg{i}"
                        feasible_clusters[new_id] = cg_members
        
        # Merge small clusters if possible
        feasible_clusters = self._merge_small_clusters(feasible_clusters)
        
        logger.info(f"Feasibility ensured. Split {split_count} clusters. "
                   f"Final count: {len(feasible_clusters)}")
        
        return feasible_clusters
    
    def _merge_small_clusters(self, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Merge small clusters within same cable group.
        """
        constraints = self.preprocessed_data['constraints']
        bid_to_idx = constraints['bid_to_idx']
        
        # Group clusters by cable group
        clusters_by_cg = defaultdict(list)
        
        for cluster_id, members in clusters.items():
            if len(members) < 3:  # Small cluster
                # Find cable group
                for bid in members:
                    if bid in bid_to_idx:
                        idx = bid_to_idx[bid]
                        for cg_id, cg_indices in constraints['cable_groups'].items():
                            if idx in cg_indices:
                                clusters_by_cg[cg_id].append((cluster_id, members))
                                break
                        break
        
        # Merge small clusters in same cable group
        merged_clusters = {k: v for k, v in clusters.items() if len(v) >= 3}
        
        for cg_id, small_clusters in clusters_by_cg.items():
            if len(small_clusters) > 1:
                # Merge all small clusters in this cable group
                merged_members = []
                for _, members in small_clusters:
                    merged_members.extend(members)
                
                if len(merged_members) >= 3:
                    merged_clusters[f"spectral_merged_{cg_id}"] = merged_members
            elif len(small_clusters) == 1:
                # Keep single small cluster if no merge possible
                cluster_id, members = small_clusters[0]
                merged_clusters[cluster_id] = members
        
        return merged_clusters
    
    def _calculate_cut_value(self) -> float:
        """
        Calculate the normalized cut value of the clustering.
        Lower is better.
        """
        if not self.clusters or self.affinity_matrix is None:
            return float('inf')
        
        total_cut = 0
        total_volume = 0
        
        # Create cluster assignment
        n_buildings = self.affinity_matrix.shape[0]
        assignment = np.zeros(n_buildings)
        
        building_features = self.preprocessed_data['building_features']
        building_ids = building_features['ogc_fid'].tolist()
        
        for cluster_idx, (cluster_id, members) in enumerate(self.clusters.items()):
            for bid in members:
                if bid in building_ids:
                    idx = building_ids.index(bid)
                    assignment[idx] = cluster_idx
        
        # Calculate cut for each cluster
        for cluster_idx in range(len(self.clusters)):
            cluster_mask = (assignment == cluster_idx)
            
            # Cut: sum of edges between cluster and rest
            cut = np.sum(self.affinity_matrix[cluster_mask, :][:, ~cluster_mask])
            
            # Volume: sum of all edges from cluster
            volume = np.sum(self.affinity_matrix[cluster_mask, :])
            
            if volume > 0:
                total_cut += cut / volume
                total_volume += 1
        
        return total_cut / total_volume if total_volume > 0 else float('inf')
    
    def get_method_specific_metrics(self) -> Dict[str, Any]:
        """
        Get method-specific metrics for spectral clustering.
        """
        if not self.clusters:
            return {}
        
        metrics = {
            'normalized_cut': self._calculate_cut_value(),
            'n_final_clusters': len(self.clusters)
        }
        
        # Add spectral properties if available
        if self.laplacian is not None:
            # Calculate algebraic connectivity (2nd smallest eigenvalue)
            try:
                L_sparse = csr_matrix(self.laplacian)
                eigenvalues, _ = eigsh(L_sparse, k=min(3, self.laplacian.shape[0] - 1), 
                                      which='SM')
                if len(eigenvalues) > 1:
                    metrics['algebraic_connectivity'] = float(eigenvalues[1])
                    metrics['spectral_gap'] = float(eigenvalues[1] - eigenvalues[0])
            except:
                pass
        
        # Add cluster balance metric
        cluster_sizes = [len(members) for members in self.clusters.values()]
        if cluster_sizes:
            metrics['cluster_balance'] = np.std(cluster_sizes) / np.mean(cluster_sizes)
        
        return metrics