"""
Method 7: Node2Vec with Electrical Flow Paths
Based on: "node2vec: Scalable Feature Learning for Networks" (Grover & Leskovec, KDD 2016) 
+ "Power Flow Based Graph Neural Networks" (Chen et al., IEEE Trans. Power Systems, 2022)

Modify random walks to follow electrical paths:
1. Walks preferentially follow CableGroup connections
2. Return parameter (p) based on electrical distance
3. In-out parameter (q) based on complementarity
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Any, Tuple, Optional
import logging
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict
import random
from base_method import BaseClusteringMethod

logger = logging.getLogger(__name__)

class ElectricalNode2Vec(BaseClusteringMethod):
    """
    Node2Vec clustering modified for electrical networks.
    Random walks follow electrical flow paths with complementarity-based transitions.
    """
    
    def __init__(self, embedding_dim: int = 64, walk_length: int = 30,
                 num_walks: int = 20, p: float = 1.0, q: float = 1.0,
                 n_clusters: int = 15, window_size: int = 5):
        """
        Initialize Electrical Node2Vec.
        
        Args:
            embedding_dim: Dimension of node embeddings
            walk_length: Length of each random walk
            num_walks: Number of walks per node
            p: Return parameter (controls likelihood of returning to previous node)
            q: In-out parameter (controls exploration vs exploitation)
            n_clusters: Number of clusters to form
            window_size: Context window for Word2Vec
        """
        super().__init__(
            name="Electrical Node2Vec Clustering",
            paper_reference="Grover & Leskovec, KDD 2016 + Chen et al., IEEE Trans. Power Systems, 2022"
        )
        
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.n_clusters = n_clusters
        self.window_size = window_size
        
        self.graph = None
        self.walks = []
        self.embeddings = None
        
        logger.info(f"Initialized with dim={embedding_dim}, walks={num_walks}x{walk_length}, "
                   f"p={p}, q={q}, clusters={n_clusters}")
    
    def _perform_clustering(self, **kwargs) -> Dict[str, List[str]]:
        """
        Perform Node2Vec clustering with electrical flow paths.
        
        Returns:
            Dictionary mapping cluster_id -> list of building_ids
        """
        # Update parameters
        self.n_clusters = kwargs.get('n_clusters', self.n_clusters)
        self.p = kwargs.get('p', self.p)
        self.q = kwargs.get('q', self.q)
        
        # Build electrical graph
        self.graph = self._build_electrical_graph()
        
        # Generate biased random walks
        self.walks = self._generate_electrical_walks()
        
        # Learn embeddings
        self.embeddings = self._learn_embeddings()
        
        # Cluster based on embeddings
        clusters = self._cluster_embeddings()
        
        # Post-process for constraints
        clusters = self._enforce_electrical_constraints(clusters)
        
        return clusters
    
    def _build_electrical_graph(self) -> nx.Graph:
        """
        Build graph with electrical topology and complementarity weights.
        Modified from original node2vec to incorporate electrical constraints.
        """
        logger.info("Building electrical flow graph...")
        
        G = nx.Graph()
        
        # Get data
        building_features = self.preprocessed_data['building_features']
        topology = self.preprocessed_data['topology']
        complementarity = self.preprocessed_data['complementarity']
        constraints = self.preprocessed_data['constraints']
        electrical_distances = self.preprocessed_data['electrical_distances']
        
        # Add building nodes
        for _, row in building_features.iterrows():
            G.add_node(
                row['ogc_fid'],
                node_type='building',
                area=row['area'],
                has_solar=row['has_solar'],
                has_battery=row['has_battery'],
                energy_label=row['energy_label'],
                cable_group=row.get('lv_group_id', 'unknown')
            )
        
        # Add infrastructure nodes (cable groups, transformers)
        for cg in topology['nodes'].get('cable_groups', []):
            G.add_node(
                f"CG_{cg.get('group_id', cg)}",
                node_type='cable_group',
                voltage_level=cg.get('voltage_level', 'LV')
            )
        
        for t in topology['nodes'].get('transformers', []):
            G.add_node(
                f"T_{t.get('transformer_id', t)}",
                node_type='transformer'
            )
        
        # Add edges based on electrical connections
        bid_to_idx = constraints['bid_to_idx']
        
        # Building to Cable Group edges
        for edge in topology['edges'].get('building_to_cable', []):
            if 'src' in edge and 'dst' in edge:
                src = edge['src']
                dst = f"CG_{edge['dst']}"
                if G.has_node(src) and G.has_node(dst):
                    G.add_edge(src, dst, weight=1.0, edge_type='connects_to')
        
        # Cable Group to Transformer edges
        for edge in topology['edges'].get('cable_to_transformer', []):
            if 'src' in edge and 'dst' in edge:
                src = f"CG_{edge['src']}"
                dst = f"T_{edge['dst']}"
                if G.has_node(src) and G.has_node(dst):
                    G.add_edge(src, dst, weight=0.8, edge_type='feeds_from')
        
        # Building to Building edges based on complementarity
        building_ids = building_features['ogc_fid'].tolist()
        
        for i, bid1 in enumerate(building_ids):
            for j, bid2 in enumerate(building_ids[i+1:], i+1):
                if bid1 in bid_to_idx and bid2 in bid_to_idx:
                    idx1 = bid_to_idx[bid1]
                    idx2 = bid_to_idx[bid2]
                    
                    # Check electrical feasibility
                    elec_dist = electrical_distances[idx1, idx2]
                    
                    if elec_dist <= 2:  # Same transformer or closer
                        # Weight based on complementarity and electrical distance
                        comp_weight = complementarity[idx1, idx2]
                        dist_weight = 1.0 / (1 + elec_dist)
                        
                        # Combined weight
                        edge_weight = comp_weight * dist_weight
                        
                        if edge_weight > 0.1:  # Threshold to limit graph density
                            G.add_edge(bid1, bid2, 
                                     weight=edge_weight,
                                     edge_type='potential_sharing',
                                     electrical_distance=elec_dist,
                                     complementarity=comp_weight)
        
        logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Log graph statistics
        degrees = dict(G.degree())
        avg_degree = np.mean(list(degrees.values()))
        logger.info(f"Average degree: {avg_degree:.2f}")
        
        return G
    
    def _generate_electrical_walks(self) -> List[List[str]]:
        """
        Generate random walks biased by electrical flow paths.
        Modified from original node2vec:
        P(next = x | current = v) ∝ w_vx * α_pq(t, x)
        
        Where α_pq considers:
        - Electrical feasibility from KG
        - Complementarity score
        - Transformer capacity remaining
        """
        logger.info(f"Generating {self.num_walks} walks of length {self.walk_length} per node...")
        
        walks = []
        building_nodes = [n for n in self.graph.nodes() 
                         if self.graph.nodes[n].get('node_type') == 'building']
        
        for _ in range(self.num_walks):
            # Shuffle nodes for each iteration
            random.shuffle(building_nodes)
            
            for node in building_nodes:
                walk = self._electrical_random_walk(node)
                walks.append(walk)
        
        logger.info(f"Generated {len(walks)} walks")
        
        return walks
    
    def _electrical_random_walk(self, start_node: str) -> List[str]:
        """
        Perform biased random walk following electrical paths.
        """
        walk = [start_node]
        
        for _ in range(self.walk_length - 1):
            current = walk[-1]
            neighbors = list(self.graph.neighbors(current))
            
            if not neighbors:
                break
            
            # Get previous node for calculating p, q parameters
            prev = walk[-2] if len(walk) > 1 else None
            
            # Calculate transition probabilities
            probs = self._get_transition_probabilities(prev, current, neighbors)
            
            # Sample next node
            if sum(probs) > 0:
                probs = np.array(probs) / sum(probs)
                next_node = np.random.choice(neighbors, p=probs)
                walk.append(next_node)
            else:
                break
        
        return walk
    
    def _get_transition_probabilities(self, prev: Optional[str], 
                                     current: str, 
                                     neighbors: List[str]) -> List[float]:
        """
        Calculate transition probabilities considering electrical constraints.
        """
        probs = []
        
        for neighbor in neighbors:
            # Get edge weight
            edge_data = self.graph.get_edge_data(current, neighbor, default={})
            weight = edge_data.get('weight', 1.0)
            
            # Calculate alpha based on p, q parameters
            if prev is None:
                alpha = 1.0
            elif neighbor == prev:
                # Return to previous node
                alpha = 1.0 / self.p
            elif self.graph.has_edge(neighbor, prev):
                # Move to node connected to previous (BFS-like)
                alpha = 1.0
            else:
                # Move to distant node (DFS-like)
                alpha = 1.0 / self.q
            
            # Additional factors for electrical networks
            
            # Prefer same cable group
            current_cg = self.graph.nodes[current].get('cable_group')
            neighbor_cg = self.graph.nodes[neighbor].get('cable_group')
            if current_cg and neighbor_cg and current_cg == neighbor_cg:
                alpha *= 2.0
            
            # Consider node types
            current_type = self.graph.nodes[current].get('node_type')
            neighbor_type = self.graph.nodes[neighbor].get('node_type')
            
            if current_type == 'building' and neighbor_type == 'cable_group':
                # Transition from building to infrastructure
                alpha *= 1.5
            elif current_type == 'cable_group' and neighbor_type == 'building':
                # Return from infrastructure to building
                alpha *= 1.2
            
            # Consider complementarity for building-to-building
            if current_type == 'building' and neighbor_type == 'building':
                comp = edge_data.get('complementarity', 0.5)
                alpha *= (1 + comp)
            
            probs.append(weight * alpha)
        
        return probs
    
    def _learn_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Learn node embeddings using Word2Vec on walks.
        """
        logger.info(f"Learning {self.embedding_dim}-dimensional embeddings...")
        
        # Convert walks to strings for Word2Vec
        walk_strings = [[str(node) for node in walk] for walk in self.walks]
        
        # Train Word2Vec model
        model = Word2Vec(
            sentences=walk_strings,
            vector_size=self.embedding_dim,
            window=self.window_size,
            min_count=1,
            sg=1,  # Skip-gram
            workers=4,
            epochs=10
        )
        
        # Extract embeddings for building nodes only
        embeddings = {}
        building_nodes = [n for n in self.graph.nodes() 
                         if self.graph.nodes[n].get('node_type') == 'building']
        
        for node in building_nodes:
            if str(node) in model.wv:
                embeddings[node] = model.wv[str(node)]
            else:
                # Use average embedding for unseen nodes
                embeddings[node] = np.mean(list(model.wv.vectors), axis=0)
        
        logger.info(f"Learned embeddings for {len(embeddings)} buildings")
        
        return embeddings
    
    def _cluster_embeddings(self) -> Dict[str, List[str]]:
        """
        Cluster buildings based on learned embeddings.
        """
        logger.info(f"Clustering embeddings into {self.n_clusters} clusters...")
        
        # Prepare data for clustering
        building_ids = list(self.embeddings.keys())
        embedding_matrix = np.array([self.embeddings[bid] for bid in building_ids])
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embedding_matrix)
        
        # Create clusters
        clusters = defaultdict(list)
        for bid, label in zip(building_ids, labels):
            clusters[label].append(bid)
        
        # Convert to final format
        final_clusters = {}
        for label, members in clusters.items():
            if len(members) >= 3:  # Minimum cluster size
                final_clusters[f"node2vec_{label}"] = members
        
        logger.info(f"Created {len(final_clusters)} initial clusters")
        
        return final_clusters
    
    def _enforce_electrical_constraints(self, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Post-process clusters to ensure electrical constraints.
        """
        logger.info("Enforcing electrical constraints...")
        
        constraints = self.preprocessed_data['constraints']
        bid_to_idx = constraints['bid_to_idx']
        
        refined_clusters = {}
        
        for cluster_id, members in clusters.items():
            # Check cable group constraint
            cable_group_splits = defaultdict(list)
            
            for bid in members:
                if bid in bid_to_idx:
                    idx = bid_to_idx[bid]
                    
                    # Find cable group
                    for cg_id, cg_indices in constraints['cable_groups'].items():
                        if idx in cg_indices:
                            cable_group_splits[cg_id].append(bid)
                            break
            
            # Create separate clusters for each cable group if needed
            if len(cable_group_splits) == 1:
                # No split needed
                refined_clusters[cluster_id] = members
            else:
                # Split by cable group
                for i, (cg_id, cg_members) in enumerate(cable_group_splits.items()):
                    if len(cg_members) >= 3:
                        new_id = f"{cluster_id}_cg{i}"
                        refined_clusters[new_id] = cg_members
        
        logger.info(f"After constraint enforcement: {len(refined_clusters)} clusters")
        
        return refined_clusters
    
    def get_method_specific_metrics(self) -> Dict[str, Any]:
        """
        Get Node2Vec specific metrics.
        """
        if not self.embeddings or not self.clusters:
            return {}
        
        metrics = {
            'embedding_dim': self.embedding_dim,
            'total_walks': len(self.walks),
            'avg_walk_length': np.mean([len(w) for w in self.walks]) if self.walks else 0
        }
        
        # Calculate embedding quality metrics
        if self.embeddings:
            embeddings_array = np.array(list(self.embeddings.values()))
            
            # Variance explained by embeddings
            total_variance = np.sum(np.var(embeddings_array, axis=0))
            metrics['embedding_variance'] = float(total_variance)
            
            # Average cosine similarity within clusters
            similarities = []
            for cluster_id, members in self.clusters.items():
                if len(members) >= 2:
                    cluster_embeddings = [self.embeddings[bid] for bid in members 
                                        if bid in self.embeddings]
                    if len(cluster_embeddings) >= 2:
                        # Calculate pairwise similarities
                        for i in range(len(cluster_embeddings)):
                            for j in range(i + 1, len(cluster_embeddings)):
                                e1 = cluster_embeddings[i]
                                e2 = cluster_embeddings[j]
                                sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
                                similarities.append(sim)
            
            if similarities:
                metrics['avg_intra_cluster_similarity'] = float(np.mean(similarities))
        
        # Graph statistics
        if self.graph:
            metrics['graph_nodes'] = self.graph.number_of_nodes()
            metrics['graph_edges'] = self.graph.number_of_edges()
            metrics['graph_density'] = nx.density(self.graph)
        
        return metrics