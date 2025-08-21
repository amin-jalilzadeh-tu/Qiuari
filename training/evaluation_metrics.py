# training/evaluation_metrics.py
"""
Task-specific evaluation metrics for energy GNN
Includes energy, clustering, and prediction metrics
"""

import torch
import numpy as np
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support, roc_auc_score,
    accuracy_score, confusion_matrix
)
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class EvaluationMetrics:
    """Comprehensive evaluation metrics for multi-task energy GNN"""
    
    def __init__(self, config: Dict):
        """
        Initialize metrics calculator
        
        Args:
            config: Configuration with task settings
        """
        self.config = config
        self.reset()
        
        logger.info("Initialized EvaluationMetrics")
    
    def reset(self):
        """Reset accumulated metrics"""
        self.predictions = {task: [] for task in self.config.keys()}
        self.targets = {task: [] for task in self.config.keys()}
        self.features = []
        self.clusters = []
        self.energy_profiles = []
    
    def update(self, outputs: Dict, batch: torch.Tensor):
        """
        Update metrics with batch predictions
        
        Args:
            outputs: Model outputs
            batch: Input batch with ground truth
        """
        # Store predictions for each task
        for task, task_outputs in outputs.items():
            if task in self.predictions:
                self.predictions[task].append(task_outputs)
        
        # Store targets if available
        if hasattr(batch, 'y'):
            for task in self.targets.keys():
                if hasattr(batch, f'{task}_gt'):
                    self.targets[task].append(getattr(batch, f'{task}_gt'))
        
        # Store features for clustering metrics
        if hasattr(batch, 'x'):
            self.features.append(batch.x)
    
    def compute(self) -> Dict:
        """
        Compute all metrics
        
        Returns:
            Dictionary of metrics for each task
        """
        metrics = {}
        
        # Clustering metrics
        if 'clustering' in self.predictions and self.predictions['clustering']:
            metrics['clustering'] = self._compute_clustering_metrics()
        
        # Solar optimization metrics
        if 'solar' in self.predictions and self.predictions['solar']:
            metrics['solar'] = self._compute_solar_metrics()
        
        # Retrofit metrics
        if 'retrofit' in self.predictions and self.predictions['retrofit']:
            metrics['retrofit'] = self._compute_retrofit_metrics()
        
        # Electrification metrics
        if 'electrification' in self.predictions and self.predictions['electrification']:
            metrics['electrification'] = self._compute_electrification_metrics()
        
        # Battery placement metrics
        if 'battery' in self.predictions and self.predictions['battery']:
            metrics['battery'] = self._compute_battery_metrics()
        
        # P2P trading metrics
        if 'p2p' in self.predictions and self.predictions['p2p']:
            metrics['p2p'] = self._compute_p2p_metrics()
        
        # Congestion prediction metrics
        if 'congestion' in self.predictions and self.predictions['congestion']:
            metrics['congestion'] = self._compute_congestion_metrics()
        
        # Energy-specific metrics
        metrics['energy'] = self._compute_energy_metrics()
        
        return metrics
    
    def _compute_clustering_metrics(self) -> Dict:
        """Compute clustering quality metrics"""
        metrics = {}
        
        # Concatenate all predictions
        all_assignments = []
        all_features = []
        
        for pred_batch in self.predictions['clustering']:
            if 'hard_assignment' in pred_batch:
                all_assignments.append(pred_batch['hard_assignment'])
        
        if self.features:
            all_features = torch.cat(self.features, dim=0).cpu().numpy()
        
        if all_assignments and len(all_features) > 0:
            assignments = torch.cat(all_assignments).cpu().numpy()
            
            # Only compute if we have multiple clusters
            n_clusters = len(np.unique(assignments))
            
            if n_clusters > 1 and n_clusters < len(assignments):
                try:
                    # Silhouette score
                    metrics['silhouette_score'] = silhouette_score(
                        all_features, assignments
                    )
                    
                    # Davies-Bouldin score (lower is better)
                    metrics['davies_bouldin_score'] = davies_bouldin_score(
                        all_features, assignments
                    )
                    
                    # Calinski-Harabasz score (higher is better)
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                        all_features, assignments
                    )
                except:
                    logger.warning("Could not compute clustering metrics")
            
            # Number of clusters
            metrics['num_clusters'] = n_clusters
            
            # Cluster size statistics
            cluster_sizes = np.bincount(assignments)
            metrics['avg_cluster_size'] = np.mean(cluster_sizes)
            metrics['std_cluster_size'] = np.std(cluster_sizes)
            metrics['min_cluster_size'] = np.min(cluster_sizes)
            metrics['max_cluster_size'] = np.max(cluster_sizes)
        
        # Modularity if available
        for pred_batch in self.predictions['clustering']:
            if 'modularity' in pred_batch:
                if 'modularity_scores' not in metrics:
                    metrics['modularity_scores'] = []
                metrics['modularity_scores'].append(pred_batch['modularity'].item())
        
        if 'modularity_scores' in metrics:
            metrics['avg_modularity'] = np.mean(metrics['modularity_scores'])
        
        return metrics
    
    def _compute_solar_metrics(self) -> Dict:
        """Compute solar optimization metrics"""
        metrics = {}
        
        # Aggregate predictions
        all_scores = []
        all_capacities = []
        all_roi = []
        
        for pred_batch in self.predictions['solar']:
            if 'solar_score' in pred_batch:
                all_scores.append(pred_batch['solar_score'])
            if 'capacity_kwp' in pred_batch:
                all_capacities.append(pred_batch['capacity_kwp'])
            if 'roi_years' in pred_batch:
                all_roi.append(pred_batch['roi_years'])
        
        if all_capacities:
            capacities = torch.cat(all_capacities).cpu().numpy()
            metrics['total_capacity_kwp'] = np.sum(capacities)
            metrics['avg_capacity_kwp'] = np.mean(capacities)
            metrics['num_installations'] = np.sum(capacities > 0)
        
        if all_roi:
            roi = torch.cat(all_roi).cpu().numpy()
            metrics['avg_roi_years'] = np.mean(roi)
            metrics['num_viable'] = np.sum(roi < 10)  # ROI < 10 years
            metrics['viability_rate'] = metrics['num_viable'] / len(roi)
        
        # If we have ground truth
        if self.targets['solar']:
            # Ranking metrics
            if all_scores and 'solar_ranking_gt' in self.targets:
                scores = torch.cat(all_scores).cpu().numpy()
                gt_ranking = torch.cat(self.targets['solar_ranking_gt']).cpu().numpy()
                
                # Spearman correlation
                from scipy.stats import spearmanr
                correlation, _ = spearmanr(scores, gt_ranking)
                metrics['ranking_correlation'] = correlation
        
        return metrics
    
    def _compute_retrofit_metrics(self) -> Dict:
        """Compute retrofit targeting metrics"""
        metrics = {}
        
        # Aggregate predictions
        all_scores = []
        all_savings = []
        all_costs = []
        
        for pred_batch in self.predictions['retrofit']:
            if 'retrofit_score' in pred_batch:
                all_scores.append(pred_batch['retrofit_score'])
            if 'energy_savings' in pred_batch:
                all_savings.append(pred_batch['energy_savings'])
            if 'retrofit_cost' in pred_batch:
                all_costs.append(pred_batch['retrofit_cost'])
        
        if all_savings and all_costs:
            savings = torch.cat(all_savings).cpu().numpy()
            costs = torch.cat(all_costs).cpu().numpy()
            
            metrics['total_savings_potential'] = np.sum(savings)
            metrics['avg_savings_percent'] = np.mean(savings) * 100
            metrics['total_investment_needed'] = np.sum(costs)
            
            # Cost-effectiveness
            cost_per_saving = costs / (savings + 1e-6)
            metrics['avg_cost_per_percent_saved'] = np.mean(cost_per_saving)
        
        return metrics
    
    def _compute_electrification_metrics(self) -> Dict:
        """Compute electrification readiness metrics"""
        metrics = {}
        
        # Aggregate predictions
        all_classes = []
        all_capacities = []
        
        for pred_batch in self.predictions['electrification']:
            if 'readiness_class' in pred_batch:
                all_classes.append(pred_batch['readiness_class'])
            if 'hp_capacity_kw' in pred_batch:
                all_capacities.append(pred_batch['hp_capacity_kw'])
        
        if all_classes:
            classes = torch.cat(all_classes).cpu().numpy()
            
            # Readiness distribution
            unique, counts = np.unique(classes, return_counts=True)
            readiness_dist = dict(zip(['ready', 'conditional', 'major_upgrade'], 
                                     counts / len(classes)))
            metrics['readiness_distribution'] = readiness_dist
            metrics['num_ready'] = np.sum(classes == 0)
        
        if all_capacities:
            capacities = torch.cat(all_capacities).cpu().numpy()
            metrics['total_hp_capacity_kw'] = np.sum(capacities)
            metrics['avg_hp_capacity_kw'] = np.mean(capacities)
        
        return metrics
    
    def _compute_battery_metrics(self) -> Dict:
        """Compute battery placement metrics"""
        metrics = {}
        
        # Aggregate predictions
        all_values = []
        all_capacities = []
        
        for pred_batch in self.predictions['battery']:
            if 'total_value_score' in pred_batch:
                all_values.append(pred_batch['total_value_score'])
            if 'capacity_kwh' in pred_batch:
                all_capacities.append(pred_batch['capacity_kwh'])
        
        if all_capacities:
            capacities = torch.cat(all_capacities).cpu().numpy()
            metrics['total_storage_capacity_kwh'] = np.sum(capacities)
            metrics['avg_battery_size_kwh'] = np.mean(capacities)
            metrics['num_batteries'] = np.sum(capacities > 0)
        
        if all_values:
            values = torch.cat(all_values).cpu().numpy()
            metrics['avg_value_score'] = np.mean(values)
            metrics['high_value_locations'] = np.sum(values > 0.7)
        
        return metrics
    
    def _compute_p2p_metrics(self) -> Dict:
        """Compute P2P trading metrics"""
        metrics = {}
        
        # Aggregate predictions
        all_compatibilities = []
        all_volumes = []
        all_values = []
        
        for pred_batch in self.predictions['p2p']:
            if 'compatibility' in pred_batch:
                all_compatibilities.append(pred_batch['compatibility'])
            if 'trading_volume' in pred_batch:
                all_volumes.append(pred_batch['trading_volume'])
            if 'trading_value' in pred_batch:
                all_values.append(pred_batch['trading_value'])
        
        if all_volumes:
            volumes = torch.cat(all_volumes).cpu().numpy()
            metrics['total_trading_volume_kwh'] = np.sum(volumes)
            metrics['avg_trade_size_kwh'] = np.mean(volumes[volumes > 0])
        
        if all_values:
            values = torch.cat(all_values).cpu().numpy()
            metrics['total_trading_value_eur'] = np.sum(values)
        
        if all_compatibilities:
            compatibilities = torch.cat(all_compatibilities).cpu().numpy()
            metrics['num_viable_pairs'] = np.sum(compatibilities > 0.5)
            metrics['avg_compatibility'] = np.mean(compatibilities)
        
        return metrics
    
    def _compute_congestion_metrics(self) -> Dict:
        """Compute congestion prediction metrics"""
        metrics = {}
        
        # Aggregate predictions
        all_probs = []
        all_alerts = {'yellow': 0, 'orange': 0, 'red': 0}
        
        for pred_batch in self.predictions['congestion']:
            if 'congestion_probs' in pred_batch:
                all_probs.append(pred_batch['congestion_probs'])
            
            for alert_type in all_alerts.keys():
                if f'{alert_type}_alert' in pred_batch:
                    all_alerts[alert_type] += pred_batch[f'{alert_type}_alert'].sum().item()
        
        if all_probs:
            probs = torch.cat(all_probs).cpu().numpy()
            
            # Average congestion probability by horizon
            metrics['avg_congestion_1d'] = np.mean(probs[:, 0]) if probs.shape[1] > 0 else 0
            metrics['avg_congestion_1w'] = np.mean(probs[:, 1]) if probs.shape[1] > 1 else 0
            metrics['avg_congestion_1y'] = np.mean(probs[:, 2]) if probs.shape[1] > 2 else 0
        
        # Alert statistics
        total_nodes = sum(all_alerts.values())
        if total_nodes > 0:
            metrics['alert_distribution'] = {
                k: v / total_nodes for k, v in all_alerts.items()
            }
        
        return metrics
    
    def _compute_energy_metrics(self) -> Dict:
        """Compute overall energy system metrics"""
        metrics = {}
        
        # Peak reduction potential
        if self.energy_profiles:
            profiles = np.concatenate(self.energy_profiles, axis=0)
            
            # Individual peaks vs aggregated peak
            individual_peaks = np.max(profiles, axis=1)
            aggregated_profile = np.sum(profiles, axis=0)
            aggregated_peak = np.max(aggregated_profile)
            
            metrics['peak_reduction_potential'] = 1 - (aggregated_peak / np.sum(individual_peaks))
            
            # Load factor
            metrics['system_load_factor'] = np.mean(aggregated_profile) / aggregated_peak
            
            # Variability
            metrics['demand_variability'] = np.std(aggregated_profile) / np.mean(aggregated_profile)
        
        # Self-sufficiency estimation (if solar deployed)
        if 'solar' in self.predictions and 'total_capacity_kwp' in self._compute_solar_metrics():
            solar_capacity = self._compute_solar_metrics()['total_capacity_kwp']
            
            # Simple estimation: 1kWp generates ~1200kWh/year in Europe
            annual_generation = solar_capacity * 1200
            
            # Estimate consumption (simplified)
            if self.energy_profiles:
                annual_consumption = np.sum(self.energy_profiles) * 365 / len(self.energy_profiles)
                metrics['estimated_self_sufficiency'] = min(1.0, annual_generation / annual_consumption)
        
        return metrics

class RegressionMetrics:
    """Regression metrics for continuous predictions"""
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Compute regression metrics"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        }

class ClassificationMetrics:
    """Classification metrics for discrete predictions"""
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray, 
                average: str = 'weighted') -> Dict:
        """Compute classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred)
        }
        
        # Multi-class metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average=average
        )
        
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        return metrics

# Usage example
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config/tasks_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create metrics calculator
    metrics = EvaluationMetrics(config)
    
    # Mock outputs
    outputs = {
        'clustering': {
            'hard_assignment': torch.randint(0, 5, (100,)),
            'modularity': torch.tensor(0.5)
        },
        'solar': {
            'capacity_kwp': torch.rand(100) * 100,
            'roi_years': torch.rand(100) * 15
        }
    }
    
    # Mock batch
    batch = type('obj', (object,), {
        'x': torch.randn(100, 45)
    })()
    
    # Update metrics
    metrics.update(outputs, batch)
    
    # Compute metrics
    results = metrics.compute()
    
    print("Metrics:")
    for task, task_metrics in results.items():
        print(f"\n{task}:")
        for metric, value in task_metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")