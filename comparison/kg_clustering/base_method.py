"""
Base Class for KG-Aware Clustering Methods
All clustering methods should inherit from this class
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class BaseClusteringMethod(ABC):
    """
    Abstract base class for energy complementarity clustering methods.
    """
    
    def __init__(self, name: str, paper_reference: str):
        """
        Initialize base clustering method.
        
        Args:
            name: Method name
            paper_reference: Citation for the base paper
        """
        self.name = name
        self.paper_reference = paper_reference
        self.preprocessed_data = None
        self.clusters = None
        self.execution_time = None
        self.parameters = {}
        
        logger.info(f"Initialized {name} clustering method")
        logger.info(f"Based on: {paper_reference}")
    
    def fit(self, preprocessed_data: Dict[str, Any], **kwargs) -> 'BaseClusteringMethod':
        """
        Fit the clustering method to data.
        
        Args:
            preprocessed_data: Output from KGDataPreprocessor
            **kwargs: Method-specific parameters
            
        Returns:
            Self for chaining
        """
        logger.info(f"Fitting {self.name} to data...")
        
        self.preprocessed_data = preprocessed_data
        self.parameters = kwargs
        
        # Validate data
        self._validate_data()
        
        # Record start time
        start_time = datetime.now()
        
        # Run clustering
        self.clusters = self._perform_clustering(**kwargs)
        
        # Post-process to ensure constraints
        self.clusters = self._post_process_clusters(self.clusters)
        
        # Record execution time
        self.execution_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Clustering complete. Found {len(self.clusters)} clusters in {self.execution_time:.2f}s")
        
        return self
    
    @abstractmethod
    def _perform_clustering(self, **kwargs) -> Dict[str, List[str]]:
        """
        Perform the actual clustering.
        Must be implemented by subclasses.
        
        Args:
            **kwargs: Method-specific parameters
            
        Returns:
            Dictionary mapping cluster_id -> list of building_ids
        """
        pass
    
    def _validate_data(self):
        """Validate that required data is present."""
        required_keys = ['topology', 'time_series', 'constraints', 
                        'complementarity', 'building_features']
        
        for key in required_keys:
            if key not in self.preprocessed_data:
                raise ValueError(f"Missing required data: {key}")
        
        # Check data consistency
        n_buildings = len(self.preprocessed_data['building_features'])
        comp_shape = self.preprocessed_data['complementarity'].shape
        
        if comp_shape[0] != n_buildings or comp_shape[1] != n_buildings:
            raise ValueError(f"Inconsistent data dimensions: {n_buildings} buildings, "
                           f"but complementarity matrix is {comp_shape}")
    
    def _post_process_clusters(self, clusters: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """
        Post-process clusters to ensure constraints are satisfied.
        
        Args:
            clusters: Raw clustering results
            
        Returns:
            Processed clusters
        """
        logger.info("Post-processing clusters to ensure constraints...")
        
        processed = {}
        constraints = self.preprocessed_data['constraints']
        
        for cluster_id, building_ids in clusters.items():
            if not building_ids:
                continue
            
            # Split cluster if it violates cable group constraints
            cable_group_splits = self._split_by_cable_group(building_ids, constraints)
            
            for i, split in enumerate(cable_group_splits):
                if len(split) >= self.parameters.get('min_cluster_size', 3):
                    new_id = f"{cluster_id}_{i}" if len(cable_group_splits) > 1 else cluster_id
                    processed[new_id] = split
        
        return processed
    
    def _split_by_cable_group(self, building_ids: List[str], 
                             constraints: Dict) -> List[List[str]]:
        """
        Split buildings by cable group to ensure electrical feasibility.
        
        Args:
            building_ids: List of building IDs
            constraints: Constraint matrices and mappings
            
        Returns:
            List of building ID lists, one per cable group
        """
        cable_group_map = {}
        bid_to_idx = constraints['bid_to_idx']
        
        for bid in building_ids:
            bid_str = str(bid)
            if bid_str not in bid_to_idx:
                continue
            
            idx = bid_to_idx[bid_str]
            
            # Find cable group for this building
            cg_found = None
            for cg_id, cg_indices in constraints['cable_groups'].items():
                if idx in cg_indices:
                    cg_found = cg_id
                    break
            
            if cg_found:
                if cg_found not in cable_group_map:
                    cable_group_map[cg_found] = []
                cable_group_map[cg_found].append(bid)
        
        # Add uncategorized buildings to largest group
        uncategorized = [bid for bid in building_ids 
                        if not any(bid in group for group in cable_group_map.values())]
        
        if uncategorized and cable_group_map:
            largest_group = max(cable_group_map.keys(), 
                              key=lambda k: len(cable_group_map[k]))
            cable_group_map[largest_group].extend(uncategorized)
        elif uncategorized:
            cable_group_map['unknown'] = uncategorized
        
        return list(cable_group_map.values())
    
    def get_clusters(self) -> Optional[Dict[str, List[str]]]:
        """
        Get clustering results.
        
        Returns:
            Dictionary mapping cluster_id -> list of building_ids
        """
        return self.clusters
    
    def get_execution_info(self) -> Dict[str, Any]:
        """
        Get execution information.
        
        Returns:
            Dictionary with method info and execution details
        """
        return {
            'method': self.name,
            'paper_reference': self.paper_reference,
            'parameters': self.parameters,
            'execution_time': self.execution_time,
            'n_clusters': len(self.clusters) if self.clusters else 0,
            'n_buildings_clustered': sum(len(c) for c in self.clusters.values()) if self.clusters else 0
        }
    
    def save_results(self, filepath: str):
        """
        Save clustering results to file.
        
        Args:
            filepath: Path to save results
        """
        if not self.clusters:
            logger.warning("No clusters to save")
            return
        
        results = {
            'method_info': self.get_execution_info(),
            'clusters': {k: list(v) for k, v in self.clusters.items()},
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def load_results(self, filepath: str):
        """
        Load clustering results from file.
        
        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.clusters = results['clusters']
        self.parameters = results['method_info'].get('parameters', {})
        self.execution_time = results['method_info'].get('execution_time')
        
        logger.info(f"Results loaded from {filepath}")
    
    def visualize_clusters(self) -> Dict[str, Any]:
        """
        Create visualization data for clusters.
        Can be overridden by specific methods.
        
        Returns:
            Dictionary with visualization data
        """
        if not self.clusters or not self.preprocessed_data:
            return {}
        
        viz_data = {
            'clusters': [],
            'edges': [],
            'statistics': {}
        }
        
        # Get building positions
        building_features = self.preprocessed_data['building_features']
        
        for cluster_id, building_ids in self.clusters.items():
            cluster_buildings = []
            
            for bid in building_ids:
                bid_str = str(bid)
                building_row = building_features[building_features['ogc_fid'] == bid_str]
                
                if not building_row.empty:
                    cluster_buildings.append({
                        'id': bid_str,
                        'x': float(building_row.iloc[0]['x']),
                        'y': float(building_row.iloc[0]['y']),
                        'cluster': cluster_id,
                        'has_solar': bool(building_row.iloc[0]['has_solar']),
                        'has_battery': bool(building_row.iloc[0]['has_battery'])
                    })
            
            viz_data['clusters'].append({
                'id': cluster_id,
                'buildings': cluster_buildings,
                'size': len(cluster_buildings)
            })
        
        return viz_data
    
    def _calculate_cluster_complementarity(self, building_ids: List[str]) -> float:
        """
        Calculate average complementarity within a cluster.
        
        Args:
            building_ids: List of building IDs in cluster
            
        Returns:
            Average pairwise complementarity score
        """
        if len(building_ids) < 2:
            return 0.0
        
        bid_to_idx = self.preprocessed_data['constraints']['bid_to_idx']
        complementarity = self.preprocessed_data['complementarity']
        
        scores = []
        for i, bid1 in enumerate(building_ids):
            for bid2 in building_ids[i+1:]:
                if str(bid1) in bid_to_idx and str(bid2) in bid_to_idx:
                    idx1 = bid_to_idx[str(bid1)]
                    idx2 = bid_to_idx[str(bid2)]
                    scores.append(complementarity[idx1, idx2])
        
        return np.mean(scores) if scores else 0.0
    
    def _check_transformer_capacity(self, building_ids: List[str], 
                                   transformer_id: str) -> bool:
        """
        Check if buildings exceed transformer capacity.
        
        Args:
            building_ids: List of building IDs
            transformer_id: Transformer ID
            
        Returns:
            True if within capacity, False otherwise
        """
        # Get transformer capacity
        capacity = self.preprocessed_data['constraints']['transformer_capacity'].get(
            transformer_id, 630  # Default 630 kVA
        )
        
        # Calculate total peak demand
        total_peak = 0
        time_series = self.preprocessed_data['time_series']
        
        for bid in building_ids:
            bid_str = str(bid)
            if bid_str in time_series:
                ts = time_series[bid_str]
                if len(ts) > 0:
                    # Column 3 is electricity demand
                    total_peak += np.max(ts[:, 3])
        
        return total_peak <= capacity * 0.8  # 80% safety margin