"""
Uncertainty Analysis for Energy GNN
Implements MC Dropout and confidence estimation
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


class UncertaintyAnalyzer:
    """
    Analyzes prediction uncertainty using MC Dropout and other techniques
    """
    
    def __init__(self, config: Dict):
        """
        Initialize uncertainty analyzer
        
        Args:
            config: Configuration for uncertainty analysis
        """
        self.mc_iterations = config.get('mc_iterations', 20)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        self.dropout_rate = config.get('dropout_rate', 0.2)
        
        logger.info(f"Initialized UncertaintyAnalyzer with {self.mc_iterations} MC iterations")
    
    def analyze_clustering_uncertainty(
        self,
        model: nn.Module,
        data: Dict,
        task: str = 'clustering'
    ) -> Dict:
        """
        Analyze uncertainty in clustering predictions using MC Dropout
        
        Args:
            model: GNN model
            data: Input data
            task: Task type
            
        Returns:
            Uncertainty analysis results
        """
        model.train()  # Keep dropout active
        device = next(model.parameters()).device
        
        # Run multiple forward passes
        predictions = []
        cluster_assignments = []
        
        for i in range(self.mc_iterations):
            with torch.no_grad():
                outputs = model(data, task=task)
                
                if 'cluster_logits' in outputs:
                    logits = outputs['cluster_logits']
                elif 'cluster_assignments' in outputs:
                    logits = outputs['cluster_assignments']
                else:
                    logits = outputs.get('clusters', torch.zeros(1))
                
                predictions.append(logits)
                cluster_assignments.append(logits.argmax(dim=-1))
        
        # Stack predictions
        predictions = torch.stack(predictions)  # [mc_iterations, num_buildings, num_clusters]
        cluster_assignments = torch.stack(cluster_assignments)  # [mc_iterations, num_buildings]
        
        # Calculate uncertainty metrics
        metrics = self._calculate_uncertainty_metrics(predictions, cluster_assignments)
        
        # Identify uncertain buildings
        uncertain_buildings = self._identify_uncertain_nodes(metrics)
        
        # Calculate cluster stability
        cluster_stability = self._calculate_cluster_stability(cluster_assignments)
        
        return {
            'mean_prediction': predictions.mean(dim=0),
            'std_prediction': predictions.std(dim=0),
            'entropy': metrics['entropy'],
            'mutual_information': metrics['mutual_information'],
            'variation_ratio': metrics['variation_ratio'],
            'uncertain_buildings': uncertain_buildings,
            'cluster_stability': cluster_stability,
            'confidence_scores': metrics['confidence']
        }
    
    def analyze_solar_uncertainty(
        self,
        model: nn.Module,
        data: Dict,
        building_ids: List[int]
    ) -> Dict:
        """
        Analyze uncertainty in solar recommendations
        
        Args:
            model: GNN model
            data: Input data
            building_ids: Buildings to analyze
            
        Returns:
            Solar recommendation uncertainty
        """
        model.train()  # Keep dropout active
        
        # Run multiple forward passes
        solar_scores = []
        roi_predictions = []
        
        for i in range(self.mc_iterations):
            with torch.no_grad():
                outputs = model(data, task='solar')
                
                # Extract solar scores
                if isinstance(outputs, dict):
                    scores = outputs.get('solar_scores', outputs.get('solar', torch.zeros(1)))
                else:
                    scores = outputs
                
                solar_scores.append(scores)
                
                # Simulate ROI (would come from solar simulator in practice)
                roi = 5 + torch.randn_like(scores) * 2  # Mock ROI years
                roi_predictions.append(roi)
        
        # Stack predictions
        solar_scores = torch.stack(solar_scores)  # [mc_iterations, num_buildings]
        roi_predictions = torch.stack(roi_predictions)
        
        # Calculate uncertainty for each building
        results = {}
        for building_id in building_ids:
            building_scores = solar_scores[:, building_id]
            building_roi = roi_predictions[:, building_id]
            
            results[building_id] = {
                'mean_score': building_scores.mean().item(),
                'std_score': building_scores.std().item(),
                'confidence': 1.0 - (building_scores.std() / building_scores.mean()).item(),
                'roi_mean': building_roi.mean().item(),
                'roi_std': building_roi.std().item(),
                'roi_interval_95': [
                    building_roi.mean().item() - 1.96 * building_roi.std().item(),
                    building_roi.mean().item() + 1.96 * building_roi.std().item()
                ],
                'recommendation': 'high' if building_scores.mean() > 0.7 else 'medium' if building_scores.mean() > 0.4 else 'low',
                'certainty': 'certain' if building_scores.std() < 0.1 else 'uncertain'
            }
        
        return results
    
    def _calculate_uncertainty_metrics(
        self,
        predictions: torch.Tensor,
        assignments: torch.Tensor
    ) -> Dict:
        """
        Calculate various uncertainty metrics
        
        Args:
            predictions: MC predictions [iterations, nodes, classes]
            assignments: Cluster assignments [iterations, nodes]
            
        Returns:
            Dictionary of uncertainty metrics
        """
        # Softmax probabilities
        probs = torch.softmax(predictions, dim=-1)
        mean_probs = probs.mean(dim=0)
        
        # Entropy (uncertainty)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=-1)
        
        # Mutual Information (epistemic uncertainty)
        expected_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean(dim=0)
        mutual_information = entropy - expected_entropy
        
        # Variation Ratio (disagreement)
        mode_assignment = assignments.mode(dim=0).values
        variation_ratio = 1.0 - (assignments == mode_assignment.unsqueeze(0)).float().mean(dim=0)
        
        # Confidence (max probability)
        confidence = mean_probs.max(dim=-1).values
        
        return {
            'entropy': entropy,
            'mutual_information': mutual_information,
            'variation_ratio': variation_ratio,
            'confidence': confidence
        }
    
    def _identify_uncertain_nodes(self, metrics: Dict, threshold: float = None) -> torch.Tensor:
        """
        Identify nodes with high uncertainty
        
        Args:
            metrics: Uncertainty metrics
            threshold: Confidence threshold
            
        Returns:
            Indices of uncertain nodes
        """
        if threshold is None:
            threshold = self.confidence_threshold
        
        # Nodes are uncertain if confidence is low or entropy is high
        uncertain = (metrics['confidence'] < threshold) | (metrics['entropy'] > 1.0)
        
        return uncertain.nonzero(as_tuple=True)[0]
    
    def _calculate_cluster_stability(self, assignments: torch.Tensor) -> Dict:
        """
        Calculate cluster stability across MC iterations
        
        Args:
            assignments: Cluster assignments [iterations, nodes]
            
        Returns:
            Cluster stability metrics
        """
        num_iterations, num_nodes = assignments.shape
        num_clusters = assignments.max().item() + 1
        
        # Calculate how often each node stays in the same cluster
        mode_assignment = assignments.mode(dim=0).values
        stability_per_node = (assignments == mode_assignment.unsqueeze(0)).float().mean(dim=0)
        
        # Calculate cluster-wise stability
        cluster_stability = {}
        for c in range(num_clusters):
            cluster_mask = mode_assignment == c
            if cluster_mask.any():
                cluster_stability[f'cluster_{c}'] = stability_per_node[cluster_mask].mean().item()
        
        return {
            'overall_stability': stability_per_node.mean().item(),
            'min_stability': stability_per_node.min().item(),
            'max_stability': stability_per_node.max().item(),
            'cluster_stability': cluster_stability,
            'unstable_nodes': (stability_per_node < 0.7).sum().item()
        }
    
    def calibrate_confidence(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 10
    ) -> Dict:
        """
        Calibrate confidence scores using reliability diagrams
        
        Args:
            predictions: Model predictions
            labels: True labels
            num_bins: Number of calibration bins
            
        Returns:
            Calibration metrics
        """
        probs = torch.softmax(predictions, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        accuracy = (predicted == labels).float()
        
        # Create bins
        bin_boundaries = torch.linspace(0, 1, num_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0  # Expected Calibration Error
        bin_metrics = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            
            if in_bin.any():
                bin_conf = confidence[in_bin].mean()
                bin_acc = accuracy[in_bin].mean()
                bin_size = in_bin.float().mean()
                
                ece += torch.abs(bin_conf - bin_acc) * bin_size
                
                bin_metrics.append({
                    'range': (bin_lower.item(), bin_upper.item()),
                    'confidence': bin_conf.item(),
                    'accuracy': bin_acc.item(),
                    'count': in_bin.sum().item()
                })
        
        return {
            'expected_calibration_error': ece.item(),
            'bin_metrics': bin_metrics,
            'mean_confidence': confidence.mean().item(),
            'mean_accuracy': accuracy.mean().item()
        }
    
    def generate_uncertainty_report(
        self,
        clustering_uncertainty: Dict,
        solar_uncertainty: Dict
    ) -> str:
        """
        Generate human-readable uncertainty report
        
        Args:
            clustering_uncertainty: Clustering uncertainty analysis
            solar_uncertainty: Solar recommendation uncertainty
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("UNCERTAINTY ANALYSIS REPORT")
        report.append("=" * 60)
        
        # Clustering uncertainty
        report.append("\n1. CLUSTERING UNCERTAINTY")
        report.append("-" * 40)
        report.append(f"Overall cluster stability: {clustering_uncertainty['cluster_stability']['overall_stability']:.2%}")
        report.append(f"Uncertain buildings: {len(clustering_uncertainty['uncertain_buildings'])}")
        report.append(f"Unstable nodes: {clustering_uncertainty['cluster_stability']['unstable_nodes']}")
        
        # Per-cluster stability
        report.append("\nCluster Stability:")
        for cluster, stability in clustering_uncertainty['cluster_stability']['cluster_stability'].items():
            report.append(f"  {cluster}: {stability:.2%}")
        
        # Solar uncertainty
        report.append("\n2. SOLAR RECOMMENDATION UNCERTAINTY")
        report.append("-" * 40)
        
        certain_count = sum(1 for b in solar_uncertainty.values() if b['certainty'] == 'certain')
        total_count = len(solar_uncertainty)
        
        report.append(f"Certain recommendations: {certain_count}/{total_count}")
        report.append("\nTop recommendations with confidence:")
        
        # Sort by score
        sorted_buildings = sorted(solar_uncertainty.items(), 
                                key=lambda x: x[1]['mean_score'], 
                                reverse=True)[:5]
        
        for building_id, metrics in sorted_buildings:
            report.append(f"  Building {building_id}:")
            report.append(f"    Score: {metrics['mean_score']:.3f} ± {metrics['std_score']:.3f}")
            report.append(f"    ROI: {metrics['roi_mean']:.1f} years [{metrics['roi_interval_95'][0]:.1f}, {metrics['roi_interval_95'][1]:.1f}]")
            report.append(f"    Confidence: {metrics['confidence']:.2%}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)