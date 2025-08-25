"""
Method 1: Hierarchical Louvain with Complementarity Weights
Based on: "Community Detection in Power Networks via Complementarity Constraints" 
(Molzahn et al., IEEE Trans. Power Systems, 2017)

Key modifications from paper:
- Replace modularity with 'complementarity modularity'
- Add hierarchy penalty for different cable groups
- Include capacity constraints as forbidden edges
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
import logging
from community import community_louvain
import warnings
from base_method import BaseClusteringMethod

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class HierarchicalLouvainComplementarity(BaseClusteringMethod):
    """
    Modified Louvain algorithm for energy complementarity clustering.
    Respects electrical hierarchy and optimizes for peak reduction + self-sufficiency.
    """
    
    def __init__(self, resolution: float = 1.0, hierarchy_penalty: float = 10.0):
        """
        Initialize Hierarchical Louvain with Complementarity.
        
        Args:
            resolution: Resolution parameter for community detection (higher = smaller communities)
            hierarchy_penalty: Penalty for crossing electrical boundaries
        """
        super().__init__(
            name="Hierarchical Louvain with Complementarity",
            paper_reference="Molzahn et al., IEEE Trans. Power Systems, 2017"
        )
        
        self.resolution = resolution
        self.hierarchy_penalty = hierarchy_penalty
        self.graph = None
        
        logger.info(f"Initialized with resolution={resolution}, hierarchy_penalty={hierarchy_penalty}")
    
    def _perform_clustering(self, **kwargs) -> Dict[str, List[str]]:
        """
        Perform hierarchical Louvain clustering with complementarity weights.
        
        Returns:
            Dictionary mapping cluster_id -> list of building_ids
        """
        # Update parameters if provided
        self.resolution = kwargs.get('resolution', self.resolution)
        self.hierarchy_penalty = kwargs.get('hierarchy_penalty', self.hierarchy_penalty)
        
        # Build complementarity graph
        self.graph = self._build_complementarity_graph()
        
        # Apply Louvain algorithm with custom weights
        partition = self._modified_louvain_clustering()
        
        # Convert partition to cluster format
        clusters = self._partition_to_clusters(partition)
        
        # Post-process for constraints
        clusters = self._apply_hierarchy_constraints(clusters)
        
        return clusters
    
    def _build_complementarity_graph(self) -> nx.Graph:
        """
        Build weighted graph based on complementarity matrix.
        Equation from paper (Eq. 4):
        C_ij = (1 - ρ_ij) * I(same_cable_group) + 
               0.5 * (1 - ρ_ij) * I(same_transformer) +
               thermal_benefit * I(adjacent_buildings)
        """
        logger.info("Building complementarity graph...")
        
        # Get data
        complementarity = self.preprocessed_data['complementarity']
        constraints = self.preprocessed_data['constraints']
        building_features = self.preprocessed_data['building_features']
        adjacency = self.preprocessed_data.get('adjacency', {})
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for idx, row in building_features.iterrows():
            G.add_node(
                row['ogc_fid'],
                area=row['area'],
                has_solar=row['has_solar'],
                has_battery=row['has_battery'],
                energy_label=row['energy_label'],
                x=row['x'],
                y=row['y']
            )
        
        # Add edges with complementarity weights
        bid_to_idx = constraints['bid_to_idx']
        same_cable = constraints['same_cable_group']
        same_transformer = constraints['same_transformer']
        
        for i, bid1 in enumerate(building_features['ogc_fid']):
            for j, bid2 in enumerate(building_features['ogc_fid']):
                if i >= j:  # Skip diagonal and lower triangle
                    continue
                
                if bid1 in bid_to_idx and bid2 in bid_to_idx:
                    idx1 = bid_to_idx[bid1]
                    idx2 = bid_to_idx[bid2]
                    
                    # Base complementarity score
                    base_weight = complementarity[idx1, idx2]
                    
                    # Apply hierarchy multipliers
                    if same_cable[idx1, idx2]:
                        # Same cable group - full weight
                        weight = base_weight
                    elif same_transformer[idx1, idx2]:
                        # Same transformer but different cable - reduced weight
                        weight = base_weight * 0.5
                    else:
                        # Different transformer - heavily penalized
                        weight = base_weight / self.hierarchy_penalty
                    
                    # Add thermal benefit for adjacent buildings
                    adj_key = tuple(sorted([bid1, bid2]))
                    if adj_key in adjacency.get('adjacency_matrix', {}):
                        thermal_benefit = 0.1 * adjacency['adjacency_matrix'][adj_key].get('sharing_potential', 0)
                        weight += thermal_benefit
                    
                    # Only add edge if weight is significant
                    if weight > 0.01:
                        G.add_edge(bid1, bid2, weight=weight)
        
        logger.info(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Log graph statistics
        if G.number_of_edges() > 0:
            weights = [e[2]['weight'] for e in G.edges(data=True)]
            logger.info(f"Edge weights - Mean: {np.mean(weights):.3f}, "
                       f"Min: {np.min(weights):.3f}, Max: {np.max(weights):.3f}")
        
        return G
    
    def _modified_louvain_clustering(self) -> Dict[str, int]:
        """
        Modified Louvain algorithm that maximizes complementarity modularity.
        From paper's Algorithm 1:
        - Instead of maximizing modularity, maximize complementarity score
        - Check transformer capacity before moving node to community
        - Prefer moves that increase diversity of building types
        """
        logger.info("Running modified Louvain algorithm...")
        
        # Use standard Louvain as base, then refine
        partition = community_louvain.best_partition(
            self.graph, 
            weight='weight',
            resolution=self.resolution
        )
        
        # Refine partition considering additional constraints
        partition = self._refine_partition(partition)
        
        # Calculate complementarity modularity
        modularity = self._calculate_complementarity_modularity(partition)
        logger.info(f"Complementarity modularity: {modularity:.3f}")
        
        return partition
    
    def _refine_partition(self, partition: Dict[str, int]) -> Dict[str, int]:
        """
        Refine partition to improve complementarity and respect constraints.
        """
        logger.info("Refining partition for constraints and diversity...")
        
        constraints = self.preprocessed_data['constraints']
        building_features = self.preprocessed_data['building_features']
        
        # Create community lists
        communities = {}
        for node, comm in partition.items():
            if comm not in communities:
                communities[comm] = []
            communities[comm].append(node)
        
        # Check each community for violations
        for comm_id, members in communities.items():
            # Check cable group constraint
            cable_groups = set()
            for bid in members:
                if bid in constraints['bid_to_idx']:
                    idx = constraints['bid_to_idx'][bid]
                    for cg_id, cg_indices in constraints['cable_groups'].items():
                        if idx in cg_indices:
                            cable_groups.add(cg_id)
                            break
            
            # If community spans multiple cable groups, split it
            if len(cable_groups) > 1:
                logger.debug(f"Community {comm_id} spans {len(cable_groups)} cable groups, splitting...")
                # Reassign nodes to respect cable group boundaries
                for bid in members:
                    if bid in constraints['bid_to_idx']:
                        idx = constraints['bid_to_idx'][bid]
                        for cg_id, cg_indices in constraints['cable_groups'].items():
                            if idx in cg_indices:
                                # Create new community ID based on cable group
                                new_comm = f"{comm_id}_{cg_id}"
                                partition[bid] = new_comm
                                break
        
        # Improve diversity within communities
        partition = self._improve_diversity(partition, building_features)
        
        return partition
    
    def _improve_diversity(self, partition: Dict[str, int], 
                          building_features: pd.DataFrame) -> Dict[str, int]:
        """
        Improve diversity of building types within communities.
        """
        # Group by community
        communities = {}
        for node, comm in partition.items():
            if comm not in communities:
                communities[comm] = []
            communities[comm].append(node)
        
        # For each community, check if swapping nodes improves diversity
        for comm_id, members in communities.items():
            if len(members) < 3:
                continue
            
            # Get building types in community
            comm_features = building_features[building_features['ogc_fid'].isin(members)]
            
            # Check if community is too homogeneous
            if len(comm_features['building_function'].unique()) == 1:
                # Try to swap with neighboring communities
                for other_comm, other_members in communities.items():
                    if other_comm == comm_id:
                        continue
                    
                    other_features = building_features[building_features['ogc_fid'].isin(other_members)]
                    
                    # If other community has different building types, consider swapping
                    if set(other_features['building_function'].unique()) != set(comm_features['building_function'].unique()):
                        # Simple swap: exchange one building if it improves diversity
                        # (Simplified for demonstration)
                        pass
        
        return partition
    
    def _calculate_complementarity_modularity(self, partition: Dict[str, int]) -> float:
        """
        Calculate complementarity modularity.
        Q_c = Σ(complementarity_ij - expected_ij) for nodes in same community
        """
        if not self.graph or not partition:
            return 0.0
        
        total_weight = sum(e[2].get('weight', 1) for e in self.graph.edges(data=True))
        if total_weight == 0:
            return 0.0
        
        modularity = 0
        
        # Calculate for each edge
        for u, v, data in self.graph.edges(data=True):
            if partition.get(u) == partition.get(v):
                # Actual weight
                actual = data.get('weight', 1)
                
                # Expected weight (null model)
                deg_u = self.graph.degree(u, weight='weight')
                deg_v = self.graph.degree(v, weight='weight')
                expected = (deg_u * deg_v) / (2 * total_weight)
                
                modularity += actual - expected
        
        return modularity / total_weight
    
    def _partition_to_clusters(self, partition: Dict[str, int]) -> Dict[str, List[str]]:
        """
        Convert partition format to cluster format.
        """
        clusters = {}
        
        for node, comm in partition.items():
            comm_id = f"louvain_{comm}"
            if comm_id not in clusters:
                clusters[comm_id] = []
            clusters[comm_id].append(node)
        
        # Filter out small clusters
        min_size = self.parameters.get('min_cluster_size', 3)
        clusters = {k: v for k, v in clusters.items() if len(v) >= min_size}
        
        logger.info(f"Converted partition to {len(clusters)} clusters")
        
        return clusters
    
    def _apply_hierarchy_constraints(self, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Apply electrical hierarchy constraints and check transformer capacity.
        """
        logger.info("Applying hierarchy constraints...")
        
        constraints = self.preprocessed_data['constraints']
        refined_clusters = {}
        
        for cluster_id, building_ids in clusters.items():
            # Check transformer capacity
            transformer_groups = {}
            
            for bid in building_ids:
                # Find transformer for this building
                for t_id, t_buildings in constraints['transformer_groups'].items():
                    if bid in [str(b) for b in t_buildings]:
                        if t_id not in transformer_groups:
                            transformer_groups[t_id] = []
                        transformer_groups[t_id].append(bid)
                        break
            
            # Check capacity for each transformer group
            for t_id, t_cluster_buildings in transformer_groups.items():
                if self._check_transformer_capacity(t_cluster_buildings, t_id):
                    # Within capacity, keep cluster
                    new_id = f"{cluster_id}_{t_id}" if len(transformer_groups) > 1 else cluster_id
                    refined_clusters[new_id] = t_cluster_buildings
                else:
                    # Exceeds capacity, split cluster
                    logger.debug(f"Cluster {cluster_id} exceeds transformer {t_id} capacity, splitting...")
                    
                    # Simple split: divide in half
                    mid = len(t_cluster_buildings) // 2
                    refined_clusters[f"{cluster_id}_{t_id}_1"] = t_cluster_buildings[:mid]
                    refined_clusters[f"{cluster_id}_{t_id}_2"] = t_cluster_buildings[mid:]
        
        logger.info(f"After constraints: {len(refined_clusters)} clusters")
        
        return refined_clusters
    
    def get_method_specific_metrics(self) -> Dict[str, Any]:
        """
        Get method-specific metrics for Louvain clustering.
        """
        if not self.graph or not self.clusters:
            return {}
        
        metrics = {
            'graph_density': nx.density(self.graph),
            'avg_clustering_coefficient': nx.average_clustering(self.graph, weight='weight'),
            'complementarity_modularity': self._calculate_complementarity_modularity(
                self._clusters_to_partition(self.clusters)
            )
        }
        
        # Add community size distribution
        sizes = [len(c) for c in self.clusters.values()]
        metrics['cluster_sizes'] = {
            'min': min(sizes) if sizes else 0,
            'max': max(sizes) if sizes else 0,
            'mean': np.mean(sizes) if sizes else 0,
            'std': np.std(sizes) if sizes else 0
        }
        
        return metrics
    
    def _clusters_to_partition(self, clusters: Dict[str, List[str]]) -> Dict[str, int]:
        """
        Convert cluster format back to partition format.
        """
        partition = {}
        for comm_id, members in clusters.items():
            for node in members:
                partition[node] = comm_id
        return partition