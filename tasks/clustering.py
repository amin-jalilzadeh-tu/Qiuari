# tasks/clustering.py
"""
Dynamic energy community clustering task
Discovers optimal building groups for energy sharing
"""

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import silhouette_score, davies_bouldin_score
import networkx as nx
import logging

logger = logging.getLogger(__name__)

class EnergyCommunityClustering:
    """Dynamic clustering for energy communities"""
    
    def __init__(self, model, config: Dict):
        """
        Initialize clustering task
        
        Args:
            model: Trained GNN model with clustering head
            config: Task configuration
        """
        self.model = model
        self.config = config
        
        # Clustering parameters
        self.min_cluster_size = config.get('min_cluster_size', 3)
        self.max_cluster_size = config.get('max_cluster_size', 20)
        self.respect_transformer = config.get('respect_transformer_boundaries', True)
        self.complementarity_weight = config.get('complementarity', {}).get('negative_correlation_bonus', 2.0)
        
        logger.info("Initialized EnergyCommunityClustering task")
    
    def run(self, graph_data: Dict, temporal_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run clustering task
        
        Args:
            graph_data: Graph data from KG
            temporal_data: Energy consumption profiles
            
        Returns:
            Clustering results and metrics
        """
        logger.info("Running energy community clustering...")
        
        # Prepare data
        x, edge_index, adjacency = self._prepare_data(graph_data)
        
        # Compute complementarity if temporal data available
        if temporal_data is not None:
            complementarity = self._compute_complementarity(temporal_data)
        else:
            complementarity = None
        
        # Run model inference
        with torch.no_grad():
            self.model.eval()
            outputs = self.model.task_heads['clustering'](x, adjacency)
        
        # Post-process clusters
        clusters = self._post_process_clusters(
            outputs['soft_assignment'],
            outputs['hard_assignment'],
            graph_data,
            complementarity
        )
        
        # Apply constraints
        clusters = self._apply_constraints(clusters, graph_data)
        
        # Evaluate clustering quality
        metrics = self._evaluate_clustering(clusters, graph_data, temporal_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(clusters, metrics)
        
        return {
            'clusters': clusters,
            'metrics': metrics,
            'recommendations': recommendations,
            'model_outputs': outputs
        }
    
    def _prepare_data(self, graph_data: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare graph data for model"""
        # Extract features
        buildings = graph_data['nodes']['buildings']
        x = torch.tensor(buildings[self._get_feature_columns()].values, dtype=torch.float32)
        
        # Build adjacency matrix
        n_nodes = len(buildings)
        adjacency = torch.zeros(n_nodes, n_nodes)
        
        if 'electrical' in graph_data['edges']:
            edges = graph_data['edges']['electrical']
            for _, edge in edges.iterrows():
                if edge['source'] < n_nodes and edge['target'] < n_nodes:
                    adjacency[edge['source'], edge['target']] = 1
                    adjacency[edge['target'], edge['source']] = 1
        
        # Edge index for message passing
        edge_index = adjacency.nonzero().t()
        
        return x, edge_index, adjacency
    
    def _get_feature_columns(self) -> List[str]:
        """Get relevant feature columns"""
        return [
            'area', 'height', 'peak_demand', 'avg_demand',
            'load_factor', 'has_solar', 'has_battery'
        ]
    
    def _compute_complementarity(self, temporal_data: pd.DataFrame) -> np.ndarray:
        """Compute complementarity matrix from temporal profiles"""
        # Correlation matrix
        corr_matrix = temporal_data.corr().values
        
        # Convert to complementarity (negative correlation is good)
        complementarity = (1 - corr_matrix) / 2
        
        # Boost negative correlations
        negative_mask = corr_matrix < 0
        complementarity[negative_mask] *= self.complementarity_weight
        
        return complementarity
    
    def _post_process_clusters(self, soft_assignment: torch.Tensor,
                              hard_assignment: torch.Tensor,
                              graph_data: Dict,
                              complementarity: Optional[np.ndarray]) -> Dict:
        """Post-process clustering results"""
        clusters = {}
        
        # Get unique cluster IDs
        unique_clusters = torch.unique(hard_assignment)
        
        for cluster_id in unique_clusters:
            cluster_mask = hard_assignment == cluster_id
            building_indices = torch.where(cluster_mask)[0].tolist()
            
            if len(building_indices) < self.min_cluster_size:
                continue
            
            # Calculate cluster properties
            cluster_info = {
                'id': int(cluster_id),
                'buildings': building_indices,
                'size': len(building_indices),
                'confidence': soft_assignment[cluster_mask, cluster_id].mean().item()
            }
            
            # Add complementarity score if available
            if complementarity is not None:
                cluster_comp = complementarity[np.ix_(building_indices, building_indices)]
                cluster_info['avg_complementarity'] = np.mean(cluster_comp)
                cluster_info['min_complementarity'] = np.min(cluster_comp)
            
            clusters[int(cluster_id)] = cluster_info
        
        return clusters
    
    def _apply_constraints(self, clusters: Dict, graph_data: Dict) -> Dict:
        """Apply physical and operational constraints"""
        constrained_clusters = {}
        
        for cluster_id, cluster_info in clusters.items():
            building_indices = cluster_info['buildings']
            
            # Check transformer constraint
            if self.respect_transformer:
                if not self._check_transformer_constraint(building_indices, graph_data):
                    # Split cluster by transformer
                    sub_clusters = self._split_by_transformer(building_indices, graph_data)
                    for i, sub_cluster in enumerate(sub_clusters):
                        if len(sub_cluster) >= self.min_cluster_size:
                            new_id = f"{cluster_id}_{i}"
                            constrained_clusters[new_id] = {
                                **cluster_info,
                                'id': new_id,
                                'buildings': sub_cluster,
                                'size': len(sub_cluster),
                                'split_reason': 'transformer_boundary'
                            }
                    continue
            
            # Check size constraint
            if cluster_info['size'] > self.max_cluster_size:
                # Split large cluster
                sub_clusters = self._split_large_cluster(
                    building_indices, 
                    self.max_cluster_size
                )
                for i, sub_cluster in enumerate(sub_clusters):
                    new_id = f"{cluster_id}_{i}"
                    constrained_clusters[new_id] = {
                        **cluster_info,
                        'id': new_id,
                        'buildings': sub_cluster,
                        'size': len(sub_cluster),
                        'split_reason': 'size_limit'
                    }
            else:
                constrained_clusters[cluster_id] = cluster_info
        
        return constrained_clusters
    
    def _check_transformer_constraint(self, building_indices: List[int], 
                                     graph_data: Dict) -> bool:
        """Check if buildings share same transformer"""
        buildings = graph_data['nodes']['buildings']
        transformers = buildings.iloc[building_indices]['lv_network'].unique()
        return len(transformers) == 1
    
    def _split_by_transformer(self, building_indices: List[int], 
                             graph_data: Dict) -> List[List[int]]:
        """Split buildings by transformer assignment"""
        buildings = graph_data['nodes']['buildings']
        transformer_groups = {}
        
        for idx in building_indices:
            transformer = buildings.iloc[idx]['lv_network']
            if transformer not in transformer_groups:
                transformer_groups[transformer] = []
            transformer_groups[transformer].append(idx)
        
        return list(transformer_groups.values())
    
    def _split_large_cluster(self, building_indices: List[int], 
                            max_size: int) -> List[List[int]]:
        """Split large cluster into smaller ones"""
        n_splits = (len(building_indices) + max_size - 1) // max_size
        return [building_indices[i::n_splits] for i in range(n_splits)]
    
    def _evaluate_clustering(self, clusters: Dict, graph_data: Dict,
                           temporal_data: Optional[pd.DataFrame]) -> Dict:
        """Evaluate clustering quality"""
        metrics = {}
        
        # Basic statistics
        metrics['num_clusters'] = len(clusters)
        metrics['avg_cluster_size'] = np.mean([c['size'] for c in clusters.values()])
        metrics['total_buildings_clustered'] = sum(c['size'] for c in clusters.values())
        
        if temporal_data is not None:
            # Energy metrics
            peak_reductions = []
            self_sufficiencies = []
            
            for cluster_info in clusters.values():
                building_indices = cluster_info['buildings']
                
                # Aggregate profiles
                cluster_profile = temporal_data.iloc[:, building_indices].sum(axis=1)
                
                # Peak reduction
                individual_peaks = temporal_data.iloc[:, building_indices].max()
                aggregated_peak = cluster_profile.max()
                peak_reduction = 1 - (aggregated_peak / individual_peaks.sum())
                peak_reductions.append(peak_reduction)
                
                # Self-sufficiency (if solar present)
                buildings = graph_data['nodes']['buildings']
                has_solar = buildings.iloc[building_indices]['has_solar'].any()
                if has_solar:
                    # Simplified calculation
                    self_sufficiency = 0.3  # Placeholder
                    self_sufficiencies.append(self_sufficiency)
            
            metrics['avg_peak_reduction'] = np.mean(peak_reductions)
            metrics['max_peak_reduction'] = np.max(peak_reductions)
            
            if self_sufficiencies:
                metrics['avg_self_sufficiency'] = np.mean(self_sufficiencies)
        
        # Graph metrics
        if len(clusters) > 1:
            # Modularity
            metrics['modularity'] = self._calculate_modularity(clusters, graph_data)
            
            # Silhouette score (if features available)
            try:
                labels = np.zeros(len(graph_data['nodes']['buildings']))
                for cluster_id, cluster_info in clusters.items():
                    labels[cluster_info['buildings']] = cluster_id
                
                features = graph_data['nodes']['buildings'][self._get_feature_columns()].values
                valid_mask = labels > 0
                
                if valid_mask.sum() > 1:
                    metrics['silhouette_score'] = silhouette_score(
                        features[valid_mask], 
                        labels[valid_mask]
                    )
            except:
                pass
        
        return metrics
    
    def _calculate_modularity(self, clusters: Dict, graph_data: Dict) -> float:
        """Calculate modularity of clustering"""
        # Build NetworkX graph
        G = nx.Graph()
        
        if 'electrical' in graph_data['edges']:
            edges = graph_data['edges']['electrical']
            for _, edge in edges.iterrows():
                G.add_edge(edge['source'], edge['target'])
        
        # Create partition
        partition = {}
        for cluster_id, cluster_info in clusters.items():
            for building_idx in cluster_info['buildings']:
                partition[building_idx] = cluster_id
        
        # Calculate modularity
        try:
            from networkx.algorithms.community import modularity
            communities = [set(cluster['buildings']) for cluster in clusters.values()]
            return modularity(G, communities)
        except:
            return 0.0
    
    def _generate_recommendations(self, clusters: Dict, metrics: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overall clustering recommendation
        if metrics.get('avg_peak_reduction', 0) > 0.15:
            recommendations.append({
                'type': 'high_value_clustering',
                'priority': 'high',
                'message': f"Clustering achieves {metrics['avg_peak_reduction']:.1%} peak reduction",
                'action': 'Proceed with community formation'
            })
        
        # Individual cluster recommendations
        for cluster_id, cluster_info in clusters.items():
            if cluster_info.get('avg_complementarity', 0) > 0.7:
                recommendations.append({
                    'type': 'high_complementarity',
                    'cluster_id': cluster_id,
                    'priority': 'high',
                    'message': f"Cluster {cluster_id} has excellent complementarity",
                    'action': 'Prioritize for P2P trading implementation'
                })
            
            if cluster_info['size'] < self.min_cluster_size * 1.5:
                recommendations.append({
                    'type': 'small_cluster',
                    'cluster_id': cluster_id,
                    'priority': 'medium',
                    'message': f"Cluster {cluster_id} is small ({cluster_info['size']} buildings)",
                    'action': 'Consider merging with adjacent cluster'
                })
        
        # Constraint violations
        split_clusters = [c for c in clusters.values() if 'split_reason' in c]
        if split_clusters:
            recommendations.append({
                'type': 'constraint_splits',
                'priority': 'info',
                'message': f"{len(split_clusters)} clusters were split due to constraints",
                'action': 'Review transformer capacity for community expansion'
            })
        
        return recommendations
    
    def optimize_clusters(self, initial_clusters: Dict, 
                         optimization_steps: int = 10) -> Dict:
        """Iteratively optimize clusters"""
        current_clusters = initial_clusters.copy()
        
        for step in range(optimization_steps):
            # Evaluate current clustering
            current_score = self._score_clustering(current_clusters)
            
            # Try local modifications
            modified_clusters = self._local_search(current_clusters)
            modified_score = self._score_clustering(modified_clusters)
            
            # Accept if improved
            if modified_score > current_score:
                current_clusters = modified_clusters
                logger.info(f"Optimization step {step}: score improved to {modified_score:.3f}")
            else:
                break
        
        return current_clusters
    
    def _score_clustering(self, clusters: Dict) -> float:
        """Score clustering quality"""
        # Combine multiple objectives
        score = 0.0
        
        for cluster_info in clusters.values():
            # Size penalty
            size_score = 1.0 - abs(cluster_info['size'] - 10) / 10
            
            # Complementarity bonus
            comp_score = cluster_info.get('avg_complementarity', 0.5)
            
            # Confidence score
            conf_score = cluster_info.get('confidence', 0.5)
            
            score += size_score * 0.3 + comp_score * 0.5 + conf_score * 0.2
        
        return score / len(clusters) if clusters else 0.0
    
    def _local_search(self, clusters: Dict) -> Dict:
        """Local search for cluster improvement"""
        import random
        
        modified = clusters.copy()
        
        # Try moving a random building to another cluster
        if len(modified) > 1:
            source_cluster = random.choice(list(modified.values()))
            target_cluster = random.choice(list(modified.values()))
            
            if (source_cluster != target_cluster and 
                source_cluster['size'] > self.min_cluster_size and
                target_cluster['size'] < self.max_cluster_size):
                
                # Move one building
                building = random.choice(source_cluster['buildings'])
                source_cluster['buildings'].remove(building)
                source_cluster['size'] -= 1
                target_cluster['buildings'].append(building)
                target_cluster['size'] += 1
        
        return modified

# Usage example
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config/tasks_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Mock model
    class MockModel:
        def __init__(self):
            self.task_heads = {'clustering': self._mock_clustering_head}
            
        def _mock_clustering_head(self, x, adjacency):
            n_nodes = x.shape[0]
            return {
                'soft_assignment': torch.rand(n_nodes, 10),
                'hard_assignment': torch.randint(0, 10, (n_nodes,)),
                'modularity': torch.tensor(0.5),
                'num_active_clusters': 5
            }
        
        def eval(self):
            pass
    
    # Create task
    task = EnergyCommunityClustering(
        model=MockModel(),
        config=config.get('clustering', {})
    )
    
    # Mock graph data
    graph_data = {
        'nodes': {
            'buildings': pd.DataFrame({
                'area': np.random.rand(100) * 200,
                'height': np.random.rand(100) * 10,
                'peak_demand': np.random.rand(100) * 20,
                'avg_demand': np.random.rand(100) * 10,
                'load_factor': np.random.rand(100),
                'has_solar': np.random.choice([0, 1], 100),
                'has_battery': np.random.choice([0, 1], 100),
                'lv_network': np.random.choice(['LV1', 'LV2', 'LV3'], 100)
            })
        },
        'edges': {
            'electrical': pd.DataFrame({
                'source': np.random.randint(0, 100, 200),
                'target': np.random.randint(0, 100, 200)
            })
        }
    }
    
    # Run task
    results = task.run(graph_data)
    
    print("Clustering Results:")
    print(f"Number of clusters: {results['metrics']['num_clusters']}")
    print(f"Average cluster size: {results['metrics']['avg_cluster_size']:.1f}")
    print(f"Recommendations: {len(results['recommendations'])}")