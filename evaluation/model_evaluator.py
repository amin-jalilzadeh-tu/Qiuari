"""
Model Evaluation Module
Simple metrics tracking and visualization for GNN training
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Track and visualize essential model performance metrics
    """
    
    def __init__(self, output_dir: str = "results/model_evaluation"):
        """Initialize evaluator with output directory"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.training_history = {
            'phase_1_discovery': {'losses': [], 'cluster_counts': [], 'building_stability': []},
            'phase_2_semi_supervised': {'losses': [], 'cluster_counts': [], 'building_stability': []},
            'phase_3_solar': {'losses': [], 'cluster_counts': [], 'building_stability': []}
        }
        
        self.cluster_metrics = []
        self.evaluation_results = {}
        
        # Track building assignments for stability
        self.previous_assignments = {}
        self.stability_window = 5  # Compare with assignments from 5 epochs ago
        
    def log_epoch(
        self,
        phase: str,
        epoch: int,
        loss: float,
        cluster_assignments: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None
    ):
        """
        Log metrics for each training epoch
        
        Args:
            phase: Training phase name
            epoch: Epoch number
            loss: Training loss
            cluster_assignments: Current cluster assignments
            embeddings: Optional node embeddings for analysis
        """
        # Store loss
        phase_key = f'phase_{phase}'
        if phase_key in self.training_history:
            self.training_history[phase_key]['losses'].append(loss)
            
            # Count unique clusters
            unique_clusters = len(torch.unique(cluster_assignments))
            self.training_history[phase_key]['cluster_counts'].append(unique_clusters)
            
            # Calculate BUILDING-LEVEL stability (not just cluster count!)
            building_stability = self._calculate_building_stability(phase_key, cluster_assignments)
            self.training_history[phase_key]['building_stability'].append(building_stability)
            
            # Calculate cluster distribution
            cluster_sizes = self._get_cluster_sizes(cluster_assignments)
            
            # Log basic info
            logger.debug(f"{phase} Epoch {epoch}: Loss={loss:.4f}, Clusters={unique_clusters}, "
                        f"Min_size={min(cluster_sizes)}, Max_size={max(cluster_sizes)}, "
                        f"Building_stability={building_stability:.1%}")
    
    def _get_cluster_sizes(self, cluster_assignments: torch.Tensor) -> List[int]:
        """Get the size of each cluster"""
        unique_clusters = torch.unique(cluster_assignments)
        sizes = []
        for cluster_id in unique_clusters:
            size = (cluster_assignments == cluster_id).sum().item()
            sizes.append(size)
        return sizes
    
    def calculate_cluster_quality(
        self,
        embeddings: torch.Tensor,
        cluster_assignments: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Calculate basic cluster quality metrics
        
        Args:
            embeddings: Node embeddings
            cluster_assignments: Cluster assignments
            features: Optional original features for comparison
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        # Convert to numpy for easier calculation
        embeddings_np = embeddings.detach().cpu().numpy()
        clusters_np = cluster_assignments.detach().cpu().numpy()
        
        # 1. Cluster sizes and balance
        unique_clusters, counts = np.unique(clusters_np, return_counts=True)
        metrics['num_clusters'] = len(unique_clusters)
        metrics['cluster_sizes'] = counts.tolist()
        metrics['size_std'] = np.std(counts)
        metrics['size_balance'] = min(counts) / max(counts) if max(counts) > 0 else 0
        
        # 2. Intra-cluster cohesion (average distance within clusters)
        cohesion_scores = []
        for cluster_id in unique_clusters:
            mask = clusters_np == cluster_id
            cluster_embeddings = embeddings_np[mask]
            if len(cluster_embeddings) > 1:
                # Pairwise distances within cluster
                centroid = cluster_embeddings.mean(axis=0)
                distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
                cohesion_scores.append(distances.mean())
        
        metrics['avg_cohesion'] = np.mean(cohesion_scores) if cohesion_scores else 0
        
        # 3. Inter-cluster separation (distance between cluster centroids)
        if len(unique_clusters) > 1:
            centroids = []
            for cluster_id in unique_clusters:
                mask = clusters_np == cluster_id
                centroid = embeddings_np[mask].mean(axis=0)
                centroids.append(centroid)
            
            separations = []
            for i in range(len(centroids)):
                for j in range(i+1, len(centroids)):
                    dist = np.linalg.norm(centroids[i] - centroids[j])
                    separations.append(dist)
            
            metrics['avg_separation'] = np.mean(separations)
            metrics['separation_cohesion_ratio'] = (
                metrics['avg_separation'] / metrics['avg_cohesion'] 
                if metrics['avg_cohesion'] > 0 else 0
            )
        else:
            metrics['avg_separation'] = 0
            metrics['separation_cohesion_ratio'] = 0
        
        # 4. Cluster collapse detection
        metrics['is_collapsed'] = metrics['num_clusters'] == 1
        metrics['largest_cluster_ratio'] = max(counts) / sum(counts)
        
        return metrics
    
    def plot_training_curves(self):
        """Generate and save training loss curves"""
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))
        fig.suptitle('GNN Training Metrics', fontsize=14, fontweight='bold')
        
        phases = ['phase_1_discovery', 'phase_2_semi_supervised', 'phase_3_solar']
        phase_names = ['Discovery', 'Semi-Supervised', 'Solar Optimization']
        
        for idx, (phase, name) in enumerate(zip(phases, phase_names)):
            # Loss curve
            ax_loss = axes[0, idx]
            losses = self.training_history[phase]['losses']
            if losses:
                ax_loss.plot(losses, 'b-', linewidth=2)
                ax_loss.set_title(f'{name} - Loss')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.grid(True, alpha=0.3)
                
                # Add trend line
                if len(losses) > 1:
                    z = np.polyfit(range(len(losses)), losses, 1)
                    p = np.poly1d(z)
                    ax_loss.plot(range(len(losses)), p(range(len(losses))), 
                               "r--", alpha=0.5, label=f'Trend: {z[0]:.4f}')
                    ax_loss.legend()
            
            # Cluster count
            ax_clusters = axes[1, idx]
            cluster_counts = self.training_history[phase]['cluster_counts']
            if cluster_counts:
                ax_clusters.plot(cluster_counts, 'g-', linewidth=2)
                ax_clusters.set_title(f'{name} - Clusters')
                ax_clusters.set_xlabel('Epoch')
                ax_clusters.set_ylabel('Number of Clusters')
                ax_clusters.grid(True, alpha=0.3)
                ax_clusters.set_ylim(bottom=0)
            
            # Building stability (NEW!)
            ax_stability = axes[2, idx]
            building_stability = self.training_history[phase].get('building_stability', [])
            if building_stability:
                ax_stability.plot(np.array(building_stability) * 100, 'purple', linewidth=2)
                ax_stability.set_title(f'{name} - Building Stability')
                ax_stability.set_xlabel('Epoch')
                ax_stability.set_ylabel('% Buildings Stable')
                ax_stability.grid(True, alpha=0.3)
                ax_stability.set_ylim(0, 105)
                ax_stability.axhline(y=70, color='r', linestyle='--', alpha=0.5, label='Target: 70%')
                ax_stability.legend()
        
        plt.tight_layout()
        output_path = self.output_dir / f'training_curves_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved to {output_path}")
    
    def plot_cluster_distribution(self, cluster_assignments: torch.Tensor, phase: str = "final"):
        """Plot histogram of cluster sizes"""
        cluster_sizes = self._get_cluster_sizes(cluster_assignments)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f'Cluster Distribution - {phase}', fontsize=14, fontweight='bold')
        
        # Histogram
        ax1.bar(range(len(cluster_sizes)), cluster_sizes, color='skyblue', edgecolor='navy')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Number of Buildings')
        ax1.set_title('Cluster Sizes')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add mean line
        mean_size = np.mean(cluster_sizes)
        ax1.axhline(y=mean_size, color='r', linestyle='--', 
                   label=f'Mean: {mean_size:.1f}')
        ax1.legend()
        
        # Pie chart for proportions
        ax2.pie(cluster_sizes, labels=[f'C{i}' for i in range(len(cluster_sizes))],
                autopct='%1.1f%%', startangle=90)
        ax2.set_title('Cluster Proportions')
        
        plt.tight_layout()
        output_path = self.output_dir / f'cluster_distribution_{phase}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()
        logger.info(f"Cluster distribution saved to {output_path}")
    
    def save_evaluation_report(self, final_metrics: Dict):
        """
        Save a simple evaluation report
        
        Args:
            final_metrics: Final evaluation metrics
        """
        # Convert numpy/torch types to Python native types for JSON serialization
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # Torch tensors
                return obj.item()
            else:
                return obj
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'training_summary': convert_to_native(self._get_training_summary()),
            'cluster_quality': convert_to_native(final_metrics),
            'convergence_analysis': convert_to_native(self._analyze_convergence()),
            'recommendations': self._generate_recommendations(final_metrics)
        }
        
        # Save JSON report
        json_path = self.output_dir / f'evaluation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save text summary
        text_path = self.output_dir / f'evaluation_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(text_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("GNN MODEL EVALUATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("TRAINING PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            for phase, summary in report['training_summary'].items():
                f.write(f"{phase}:\n")
                for key, value in summary.items():
                    if 'stability' in key:
                        f.write(f"  {key}: {value:.1%}\n")
                    elif isinstance(value, float):
                        f.write(f"  {key}: {value:.4f}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            f.write("CLUSTER QUALITY:\n")
            f.write("-" * 30 + "\n")
            for key, value in final_metrics.items():
                if not isinstance(value, list):
                    f.write(f"  {key}: {value:.4f}\n" if isinstance(value, float) else f"  {key}: {value}\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            f.write("-" * 30 + "\n")
            for i, rec in enumerate(report['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        logger.info(f"Evaluation report saved to {json_path}")
        logger.info(f"Evaluation summary saved to {text_path}")
    
    def _get_training_summary(self) -> Dict:
        """Summarize training performance"""
        summary = {}
        for phase, history in self.training_history.items():
            if history['losses']:
                # Calculate average building stability over last 10 epochs
                building_stabilities = history.get('building_stability', [])
                avg_building_stability = (
                    np.mean(building_stabilities[-10:]) 
                    if len(building_stabilities) > 0 else 0.0
                )
                
                summary[phase] = {
                    'initial_loss': history['losses'][0],
                    'final_loss': history['losses'][-1],
                    'loss_reduction': history['losses'][0] - history['losses'][-1],
                    'loss_reduction_pct': ((history['losses'][0] - history['losses'][-1]) / 
                                          history['losses'][0] * 100) if history['losses'][0] > 0 else 0,
                    'final_clusters': history['cluster_counts'][-1] if history['cluster_counts'] else 0,
                    'cluster_count_stability': self._calculate_stability(history['cluster_counts']),
                    'building_assignment_stability': avg_building_stability
                }
        return summary
    
    def _calculate_building_stability(self, phase_key: str, current_assignments: torch.Tensor) -> float:
        """
        Calculate how many buildings stayed in the same cluster
        This is the REAL stability metric - not just cluster count!
        """
        # Handle both tensor and numpy array inputs
        if isinstance(current_assignments, torch.Tensor):
            current_np = current_assignments.detach().cpu().numpy()
        else:
            current_np = np.array(current_assignments)
        
        # Ensure it's 1D array
        if current_np.ndim > 1:
            current_np = current_np.flatten()
        
        # Store assignments history
        if phase_key not in self.previous_assignments:
            self.previous_assignments[phase_key] = []
        
        # Compare with previous assignment if exists
        stability = 0.0
        if len(self.previous_assignments[phase_key]) > 0:
            # Compare with most recent assignment
            prev_assignments = self.previous_assignments[phase_key][-1]
            
            # Count how many buildings stayed in same cluster
            same_cluster = (current_np == prev_assignments).sum()
            stability = same_cluster / current_np.size  # Use .size instead of len()
        
        # Store current for next comparison
        self.previous_assignments[phase_key].append(current_np.copy())
        
        # Keep only last N assignments to save memory
        if len(self.previous_assignments[phase_key]) > self.stability_window:
            self.previous_assignments[phase_key].pop(0)
        
        return stability
    
    def _calculate_stability(self, cluster_counts: List[int]) -> float:
        """Calculate stability of cluster count over epochs
        DEPRECATED: This only checks if NUMBER of clusters changes, not building assignments!
        """
        if len(cluster_counts) < 2:
            return 0.0
        
        changes = sum(1 for i in range(1, len(cluster_counts)) 
                     if cluster_counts[i] != cluster_counts[i-1])
        return 1.0 - (changes / (len(cluster_counts) - 1))
    
    def _analyze_convergence(self) -> Dict:
        """Analyze convergence behavior"""
        analysis = {}
        for phase, history in self.training_history.items():
            if len(history['losses']) > 3:
                # Check if loss is decreasing
                losses = history['losses']
                is_decreasing = losses[-1] < losses[0]
                
                # Check if loss plateaued (last 3 epochs similar)
                if len(losses) >= 3:
                    recent_std = np.std(losses[-3:])
                    is_plateaued = recent_std < 0.01
                else:
                    is_plateaued = False
                
                analysis[phase] = {
                    'converged': is_decreasing and is_plateaued,
                    'is_decreasing': is_decreasing,
                    'is_plateaued': is_plateaued
                }
        return analysis
    
    def _generate_recommendations(self, final_metrics: Dict) -> List[str]:
        """Generate recommendations based on metrics"""
        recommendations = []
        
        # Check for cluster collapse
        if final_metrics.get('is_collapsed', False):
            recommendations.append("CRITICAL: Model is collapsing to single cluster - reduce regularization or adjust loss weights")
        
        # Check cluster balance
        if final_metrics.get('size_balance', 0) < 0.2:
            recommendations.append("Cluster sizes are highly imbalanced - consider balanced sampling or size penalties")
        
        # Check separation
        if final_metrics.get('separation_cohesion_ratio', 0) < 1.5:
            recommendations.append("Poor cluster separation - increase embedding dimension or adjust clustering loss")
        
        # Check convergence
        convergence = self._analyze_convergence()
        for phase, conv in convergence.items():
            if not conv.get('is_decreasing', True):
                recommendations.append(f"{phase}: Loss not decreasing - check learning rate or model architecture")
        
        # Check stability
        for phase, history in self.training_history.items():
            # Check BUILDING assignment stability (the real metric!)
            building_stabilities = history.get('building_stability', [])
            if len(building_stabilities) > 5:
                recent_stability = np.mean(building_stabilities[-5:])
                if recent_stability < 0.7:  # Less than 70% of buildings staying put
                    recommendations.append(f"{phase}: Buildings jumping between clusters (only {recent_stability:.0%} stable) - increase momentum or add temporal smoothing")
            
            # Also check cluster count stability
            if history['cluster_counts']:
                count_stability = self._calculate_stability(history['cluster_counts'])
                if count_stability < 0.5:
                    recommendations.append(f"{phase}: Number of clusters fluctuating - stabilize cluster formation")
        
        if not recommendations:
            recommendations.append("Model performance appears stable - consider longer training for refinement")
        
        return recommendations