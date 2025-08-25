"""
Method 6: Correlation Clustering with Side Information
Based on: "Correlation Clustering with Local Objectives" 
(Charikar et al., NeurIPS, 2017)

Minimize disagreement function:
disagreement = Σ_same_cluster (1 - complementarity_ij) + 
               Σ_diff_cluster complementarity_ij

With KG side information as hard constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy.optimize import linear_sum_assignment
import cvxpy as cp
from base_method import BaseClusteringMethod

logger = logging.getLogger(__name__)

class CorrelationClusteringKG(BaseClusteringMethod):
    """
    Correlation clustering that minimizes disagreement with complementarity scores.
    Uses KG constraints as must-link and cannot-link constraints.
    """
    
    def __init__(self, max_iterations: int = 100, convergence_threshold: float = 0.01,
                 use_sdp_relaxation: bool = False):
        """
        Initialize Correlation Clustering with KG constraints.
        
        Args:
            max_iterations: Maximum iterations for optimization
            convergence_threshold: Convergence threshold for objective
            use_sdp_relaxation: Whether to use SDP relaxation (more accurate but slower)
        """
        super().__init__(
            name="Correlation Clustering with KG Constraints",
            paper_reference="Charikar et al., NeurIPS, 2017"
        )
        
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.use_sdp_relaxation = use_sdp_relaxation
        
        self.must_link = None
        self.cannot_link = None
        self.disagreement_matrix = None
        
        logger.info(f"Initialized with max_iter={max_iterations}, "
                   f"convergence={convergence_threshold}, SDP={use_sdp_relaxation}")
    
    def _perform_clustering(self, **kwargs) -> Dict[str, List[str]]:
        """
        Perform correlation clustering with constraints.
        
        Returns:
            Dictionary mapping cluster_id -> list of building_ids
        """
        # Update parameters
        self.max_iterations = kwargs.get('max_iterations', self.max_iterations)
        self.use_sdp_relaxation = kwargs.get('use_sdp_relaxation', self.use_sdp_relaxation)
        
        # Extract constraints from KG
        self.must_link, self.cannot_link = self._get_kg_constraints()
        
        # Build disagreement matrix
        self.disagreement_matrix = self._build_disagreement_matrix()
        
        # Solve clustering problem
        if self.use_sdp_relaxation:
            clusters = self._solve_with_sdp()
        else:
            clusters = self._solve_with_local_search()
        
        return clusters
    
    def _get_kg_constraints(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Extract must-link and cannot-link constraints from KG.
        
        Must-link: Buildings in same adjacency cluster with high sharing potential
        Cannot-link: Buildings in different transformers
        """
        logger.info("Extracting constraints from KG...")
        
        constraints = self.preprocessed_data['constraints']
        adjacency = self.preprocessed_data.get('adjacency', {})
        bid_to_idx = constraints['bid_to_idx']
        
        must_link = []
        cannot_link = []
        
        # Must-link: Adjacent buildings with high sharing potential
        for adj_cluster in adjacency.get('clusters', []):
            if adj_cluster.get('sharing_potential', 0) > 0.7:
                buildings = adj_cluster.get('buildings', [])
                # Create must-link for buildings in high-potential clusters
                for i in range(len(buildings)):
                    for j in range(i + 1, len(buildings)):
                        bid1 = str(buildings[i].get('ogc_fid', buildings[i]))
                        bid2 = str(buildings[j].get('ogc_fid', buildings[j]))
                        if bid1 in bid_to_idx and bid2 in bid_to_idx:
                            must_link.append((bid_to_idx[bid1], bid_to_idx[bid2]))
        
        # Cannot-link: Different transformers
        for i in range(len(constraints['same_transformer'])):
            for j in range(i + 1, len(constraints['same_transformer'])):
                if not constraints['same_transformer'][i, j]:
                    # Check if they're in different cable groups too
                    if not constraints['same_cable_group'][i, j]:
                        cannot_link.append((i, j))
        
        logger.info(f"Found {len(must_link)} must-link and {len(cannot_link)} cannot-link constraints")
        
        return must_link, cannot_link
    
    def _build_disagreement_matrix(self) -> np.ndarray:
        """
        Build disagreement matrix from complementarity scores.
        
        disagreement[i,j] = cost of putting i,j in same cluster
                         = 1 - complementarity[i,j]
        """
        logger.info("Building disagreement matrix...")
        
        complementarity = self.preprocessed_data['complementarity']
        n_buildings = complementarity.shape[0]
        
        # Base disagreement is inverse of complementarity
        disagreement = 1 - complementarity
        
        # Apply constraints
        # Must-link: Set disagreement to -inf (strong preference to cluster together)
        for i, j in self.must_link:
            disagreement[i, j] = -1.0
            disagreement[j, i] = -1.0
        
        # Cannot-link: Set disagreement to inf (prohibit clustering together)
        for i, j in self.cannot_link:
            disagreement[i, j] = np.inf
            disagreement[j, i] = np.inf
        
        return disagreement
    
    def _solve_with_local_search(self) -> Dict[str, List[str]]:
        """
        Solve correlation clustering using local search heuristic.
        Faster but potentially suboptimal.
        """
        logger.info("Solving with local search heuristic...")
        
        n_buildings = self.disagreement_matrix.shape[0]
        building_features = self.preprocessed_data['building_features']
        building_ids = building_features['ogc_fid'].tolist()
        
        # Initialize with greedy clustering
        clusters = self._greedy_initialization()
        
        # Local search improvements
        prev_cost = self._calculate_total_disagreement(clusters)
        
        for iteration in range(self.max_iterations):
            improved = False
            
            # Try moving each building to a different cluster
            for bid in building_ids:
                best_move = None
                best_improvement = 0
                
                # Find current cluster
                current_cluster = None
                for c_id, members in clusters.items():
                    if bid in members:
                        current_cluster = c_id
                        break
                
                if not current_cluster:
                    continue
                
                # Try moving to each other cluster
                for target_cluster in clusters:
                    if target_cluster == current_cluster:
                        continue
                    
                    # Calculate improvement
                    improvement = self._calculate_move_improvement(
                        bid, current_cluster, target_cluster, clusters
                    )
                    
                    if improvement > best_improvement:
                        best_move = target_cluster
                        best_improvement = improvement
                
                # Execute best move if found
                if best_move and best_improvement > 0:
                    clusters[current_cluster].remove(bid)
                    if not clusters[current_cluster]:
                        del clusters[current_cluster]
                    clusters[best_move].append(bid)
                    improved = True
            
            # Check convergence
            current_cost = self._calculate_total_disagreement(clusters)
            if abs(prev_cost - current_cost) < self.convergence_threshold:
                logger.info(f"Converged after {iteration + 1} iterations")
                break
            
            prev_cost = current_cost
            
            if not improved:
                break
        
        # Filter small clusters
        final_clusters = {}
        for c_id, members in clusters.items():
            if len(members) >= 3:
                final_clusters[f"correlation_{c_id}"] = members
        
        logger.info(f"Local search complete. Found {len(final_clusters)} clusters")
        
        return final_clusters
    
    def _greedy_initialization(self) -> Dict[int, List[str]]:
        """
        Greedy initialization for correlation clustering.
        Start with each building as its own cluster, then merge.
        """
        building_features = self.preprocessed_data['building_features']
        building_ids = building_features['ogc_fid'].tolist()
        bid_to_idx = self.preprocessed_data['constraints']['bid_to_idx']
        
        # Start with singleton clusters
        clusters = {i: [bid] for i, bid in enumerate(building_ids)}
        
        # Merge clusters with negative disagreement (high complementarity)
        merged = True
        while merged:
            merged = False
            best_merge = None
            best_score = 0
            
            cluster_ids = list(clusters.keys())
            for i in range(len(cluster_ids)):
                for j in range(i + 1, len(cluster_ids)):
                    c1 = cluster_ids[i]
                    c2 = cluster_ids[j]
                    
                    if c1 not in clusters or c2 not in clusters:
                        continue
                    
                    # Calculate merge score
                    score = self._calculate_merge_score(clusters[c1], clusters[c2])
                    
                    if score < best_score:  # Negative score is good
                        best_merge = (c1, c2)
                        best_score = score
            
            # Execute best merge
            if best_merge and best_score < -0.1:
                c1, c2 = best_merge
                clusters[c1].extend(clusters[c2])
                del clusters[c2]
                merged = True
        
        return clusters
    
    def _calculate_merge_score(self, cluster1: List[str], cluster2: List[str]) -> float:
        """
        Calculate score for merging two clusters.
        Lower (more negative) is better.
        """
        bid_to_idx = self.preprocessed_data['constraints']['bid_to_idx']
        score = 0
        
        for bid1 in cluster1:
            for bid2 in cluster2:
                if bid1 in bid_to_idx and bid2 in bid_to_idx:
                    idx1 = bid_to_idx[bid1]
                    idx2 = bid_to_idx[bid2]
                    
                    if self.disagreement_matrix[idx1, idx2] == np.inf:
                        return np.inf  # Cannot merge
                    
                    score += self.disagreement_matrix[idx1, idx2]
        
        return score
    
    def _calculate_move_improvement(self, building: str, from_cluster: int,
                                   to_cluster: int, clusters: Dict) -> float:
        """
        Calculate improvement from moving building to different cluster.
        """
        bid_to_idx = self.preprocessed_data['constraints']['bid_to_idx']
        
        if building not in bid_to_idx:
            return 0
        
        b_idx = bid_to_idx[building]
        improvement = 0
        
        # Cost reduction from leaving current cluster
        for other_bid in clusters[from_cluster]:
            if other_bid != building and other_bid in bid_to_idx:
                other_idx = bid_to_idx[other_bid]
                improvement -= self.disagreement_matrix[b_idx, other_idx]
        
        # Cost increase from joining new cluster
        for other_bid in clusters[to_cluster]:
            if other_bid in bid_to_idx:
                other_idx = bid_to_idx[other_bid]
                if self.disagreement_matrix[b_idx, other_idx] == np.inf:
                    return -np.inf  # Invalid move
                improvement += self.disagreement_matrix[b_idx, other_idx]
        
        return -improvement  # Negative because we want to minimize disagreement
    
    def _calculate_total_disagreement(self, clusters: Dict) -> float:
        """
        Calculate total disagreement for clustering.
        """
        bid_to_idx = self.preprocessed_data['constraints']['bid_to_idx']
        total = 0
        
        # Within-cluster disagreement
        for c_id, members in clusters.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    if members[i] in bid_to_idx and members[j] in bid_to_idx:
                        idx1 = bid_to_idx[members[i]]
                        idx2 = bid_to_idx[members[j]]
                        if self.disagreement_matrix[idx1, idx2] != np.inf:
                            total += self.disagreement_matrix[idx1, idx2]
        
        return total
    
    def _solve_with_sdp(self) -> Dict[str, List[str]]:
        """
        Solve using semidefinite programming relaxation.
        More accurate but computationally intensive.
        """
        logger.info("Solving with SDP relaxation (this may take a while)...")
        
        n_buildings = self.disagreement_matrix.shape[0]
        
        # Limit size for SDP (computationally expensive)
        if n_buildings > 50:
            logger.warning(f"Too many buildings ({n_buildings}) for SDP. Using local search instead.")
            return self._solve_with_local_search()
        
        # SDP formulation
        X = cp.Variable((n_buildings, n_buildings), symmetric=True)
        
        # Objective: minimize disagreement
        objective = 0
        for i in range(n_buildings):
            for j in range(i + 1, n_buildings):
                if self.disagreement_matrix[i, j] != np.inf:
                    # X[i,j] = 1 if in same cluster, 0 otherwise
                    objective += self.disagreement_matrix[i, j] * X[i, j]
                    objective += (1 - self.disagreement_matrix[i, j]) * (1 - X[i, j])
        
        # Constraints
        constraints = [
            X >> 0,  # Positive semidefinite
            cp.diag(X) == 1,  # Diagonal elements are 1
        ]
        
        # Add must-link constraints
        for i, j in self.must_link:
            constraints.append(X[i, j] == 1)
        
        # Add cannot-link constraints
        for i, j in self.cannot_link:
            constraints.append(X[i, j] == 0)
        
        # Triangle inequalities for valid metric
        for i in range(n_buildings):
            for j in range(i + 1, n_buildings):
                for k in range(j + 1, n_buildings):
                    constraints.append(X[i, j] + X[j, k] - X[i, k] <= 1)
                    constraints.append(X[i, k] + X[j, k] - X[i, j] <= 1)
                    constraints.append(X[i, j] + X[i, k] - X[j, k] <= 1)
        
        # Solve
        prob = cp.Problem(cp.Minimize(objective), constraints)
        
        try:
            prob.solve(solver=cp.SCS, max_iters=1000)
            
            if prob.status not in ["optimal", "optimal_inaccurate"]:
                logger.warning(f"SDP solver status: {prob.status}. Using local search.")
                return self._solve_with_local_search()
            
            # Round solution to get clustering
            clusters = self._round_sdp_solution(X.value)
            
            logger.info(f"SDP solution found. Objective: {prob.value:.3f}")
            
            return clusters
            
        except Exception as e:
            logger.error(f"SDP solving failed: {e}. Using local search.")
            return self._solve_with_local_search()
    
    def _round_sdp_solution(self, X: np.ndarray) -> Dict[str, List[str]]:
        """
        Round SDP solution to discrete clustering.
        """
        n_buildings = X.shape[0]
        building_features = self.preprocessed_data['building_features']
        building_ids = building_features['ogc_fid'].tolist()
        
        # Use threshold rounding
        threshold = 0.5
        clusters = {}
        assigned = set()
        cluster_id = 0
        
        for i in range(n_buildings):
            if i in assigned:
                continue
            
            # Start new cluster
            cluster_members = [building_ids[i]]
            assigned.add(i)
            
            # Add buildings with high correlation
            for j in range(i + 1, n_buildings):
                if j not in assigned and X[i, j] > threshold:
                    cluster_members.append(building_ids[j])
                    assigned.add(j)
            
            if len(cluster_members) >= 3:
                clusters[f"correlation_{cluster_id}"] = cluster_members
                cluster_id += 1
        
        return clusters
    
    def get_method_specific_metrics(self) -> Dict[str, Any]:
        """
        Get correlation clustering specific metrics.
        """
        if not self.clusters:
            return {}
        
        metrics = {
            'total_disagreement': self._calculate_total_disagreement(self.clusters),
            'must_link_satisfied': 0,
            'cannot_link_satisfied': 0,
            'constraint_satisfaction_rate': 0
        }
        
        # Check constraint satisfaction
        bid_to_idx = self.preprocessed_data['constraints']['bid_to_idx']
        
        # Check must-link
        for i, j in self.must_link:
            idx_to_bid = {v: k for k, v in bid_to_idx.items()}
            bid1 = idx_to_bid.get(i)
            bid2 = idx_to_bid.get(j)
            
            # Check if in same cluster
            in_same = False
            for c_id, members in self.clusters.items():
                if bid1 in members and bid2 in members:
                    in_same = True
                    break
            
            if in_same:
                metrics['must_link_satisfied'] += 1
        
        # Check cannot-link
        for i, j in self.cannot_link:
            idx_to_bid = {v: k for k, v in bid_to_idx.items()}
            bid1 = idx_to_bid.get(i)
            bid2 = idx_to_bid.get(j)
            
            # Check if in different clusters
            in_different = True
            for c_id, members in self.clusters.items():
                if bid1 in members and bid2 in members:
                    in_different = False
                    break
            
            if in_different:
                metrics['cannot_link_satisfied'] += 1
        
        total_constraints = len(self.must_link) + len(self.cannot_link)
        if total_constraints > 0:
            metrics['constraint_satisfaction_rate'] = (
                (metrics['must_link_satisfied'] + metrics['cannot_link_satisfied']) / 
                total_constraints
            )
        
        return metrics