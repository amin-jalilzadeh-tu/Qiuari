import numpy as np
import pandas as pd
import networkx as nx
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import time
import logging
from dataclasses import dataclass

@dataclass
class ClusteringResult:
    clusters: np.ndarray
    metrics: Dict[str, float]
    computation_time: float
    additional: Dict[str, Any]
    method_name: str
    
class BaseClusteringMethod(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        
    @abstractmethod
    def fit_predict(self, input_data: Dict[str, Any]) -> np.ndarray:
        pass
    
    def run(self, input_data: Dict[str, Any]) -> ClusteringResult:
        start_time = time.time()
        
        clusters = self.fit_predict(input_data)
        
        computation_time = time.time() - start_time
        
        metrics = self.evaluate_clusters(input_data, clusters)
        
        additional = self.get_additional_info()
        
        return ClusteringResult(
            clusters=clusters,
            metrics=metrics,
            computation_time=computation_time,
            additional=additional,
            method_name=self.name
        )
    
    def evaluate_clusters(self, input_data: Dict[str, Any], clusters: np.ndarray) -> Dict[str, float]:
        consumption = input_data['consumption']
        generation = input_data.get('generation', np.zeros_like(consumption))
        
        n_buildings = consumption.shape[1]
        n_clusters = len(np.unique(clusters))
        
        total_self_sufficiency = 0.0
        total_peak_reduction = 0.0
        total_energy_shared = 0.0
        
        for cluster_id in np.unique(clusters):
            cluster_mask = clusters == cluster_id
            cluster_consumption = consumption[:, cluster_mask].sum(axis=1)
            cluster_generation = generation[:, cluster_mask].sum(axis=1)
            
            energy_shared = np.minimum(cluster_consumption, cluster_generation).sum()
            total_consumption = cluster_consumption.sum()
            
            if total_consumption > 0:
                cluster_self_sufficiency = energy_shared / total_consumption
                total_self_sufficiency += cluster_self_sufficiency * cluster_mask.sum()
            
            original_peak = consumption[:, cluster_mask].max(axis=1).sum()
            cluster_peak = cluster_consumption.max()
            if original_peak > 0:
                peak_reduction = (original_peak - cluster_peak) / original_peak
                total_peak_reduction += peak_reduction * cluster_mask.sum()
            
            total_energy_shared += energy_shared
        
        metrics = {
            'self_sufficiency': total_self_sufficiency / n_buildings,
            'peak_reduction': total_peak_reduction / n_buildings,
            'energy_shared': total_energy_shared,
            'n_clusters': n_clusters,
            'violations': self.check_violations(input_data, clusters)
        }
        
        return metrics
    
    def check_violations(self, input_data: Dict[str, Any], clusters: np.ndarray) -> int:
        violations = 0
        
        if 'constraints' in input_data:
            constraints = input_data['constraints']
            
            if 'lv_groups' in constraints:
                lv_groups = constraints['lv_groups']
                for group in lv_groups:
                    cluster_ids = clusters[group]
                    if len(np.unique(cluster_ids)) > 1:
                        violations += 1
        
        return violations
    
    def get_additional_info(self) -> Dict[str, Any]:
        return {}