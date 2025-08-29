"""
Training utilities to stabilize GNN training
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
import logging

logger = logging.getLogger(__name__)


class ClusterMomentum:
    """
    Add momentum to cluster assignments to prevent thrashing
    """
    
    def __init__(self, momentum: float = 0.9, history_size: int = 5):
        self.momentum = momentum
        self.history = deque(maxlen=history_size)
        self.smooth_assignments = None
        
    def update(self, cluster_logits: torch.Tensor) -> torch.Tensor:
        """
        Apply momentum to stabilize cluster assignments
        
        Args:
            cluster_logits: Current cluster logits [N, K]
            
        Returns:
            Smoothed cluster logits
        """
        if self.smooth_assignments is None:
            self.smooth_assignments = cluster_logits.clone()
        else:
            # Exponential moving average
            self.smooth_assignments = (
                self.momentum * self.smooth_assignments + 
                (1 - self.momentum) * cluster_logits
            )
        
        self.history.append(cluster_logits.clone())
        return self.smooth_assignments
    
    def get_stability_score(self) -> float:
        """
        Calculate stability of recent assignments
        """
        if len(self.history) < 2:
            return 1.0
        
        # Compare consecutive assignments
        changes = 0
        for i in range(1, len(self.history)):
            prev_clusters = self.history[i-1].argmax(dim=1)
            curr_clusters = self.history[i].argmax(dim=1)
            changes += (prev_clusters != curr_clusters).float().mean().item()
        
        stability = 1.0 - (changes / (len(self.history) - 1))
        return stability


class EarlyStopping:
    """
    Early stopping to prevent overfitting and waste
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.best_epoch = 0
        
    def __call__(self, score: float, epoch: int) -> bool:
        """
        Check if should stop training
        
        Returns:
            True if should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            logger.info(f"Early stopping triggered. Best epoch: {self.best_epoch}")
            return True
        
        return False


class LearningRateScheduler:
    """
    Adaptive learning rate scheduling
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        mode: str = 'plateau',
        factor: float = 0.5,
        patience: int = 5,
        min_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_score = None
        self.counter = 0
        
    def step(self, score: float):
        """
        Adjust learning rate based on score
        """
        if self.best_score is None:
            self.best_score = score
            return
        
        if self.mode == 'plateau':
            # Reduce LR if no improvement
            if abs(score - self.best_score) < 0.001:
                self.counter += 1
                if self.counter >= self.patience:
                    self._reduce_lr()
                    self.counter = 0
            else:
                self.counter = 0
                if score < self.best_score:
                    self.best_score = score
    
    def _reduce_lr(self):
        """
        Reduce learning rate
        """
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.factor, self.min_lr)
            param_group['lr'] = new_lr
            logger.info(f"Reduced learning rate from {old_lr:.6f} to {new_lr:.6f}")


class GradientClipper:
    """
    Gradient clipping to prevent explosion
    """
    
    def __init__(self, max_norm: float = 1.0):
        self.max_norm = max_norm
        self.history = []
        
    def clip(self, model: nn.Module):
        """
        Clip gradients and track norm
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_norm
        )
        self.history.append(total_norm.item())
        
        if total_norm > self.max_norm * 5:
            logger.warning(f"Large gradient norm: {total_norm:.2f}")
        
        return total_norm


class ClusterValidator:
    """
    Validate cluster quality during training
    """
    
    def __init__(self, min_size: int = 5, max_size: int = 50):
        self.min_size = min_size
        self.max_size = max_size
        
    def validate(
        self,
        cluster_assignments: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Validate cluster assignments
        
        Returns:
            Validation metrics
        """
        unique_clusters, counts = torch.unique(cluster_assignments, return_counts=True)
        
        metrics = {
            'num_clusters': len(unique_clusters),
            'min_cluster_size': counts.min().item(),
            'max_cluster_size': counts.max().item(),
            'avg_cluster_size': counts.float().mean().item(),
            'size_std': counts.float().std().item(),
            'has_tiny_clusters': (counts < self.min_size).any().item(),
            'has_huge_clusters': (counts > self.max_size).any().item(),
            'is_collapsed': len(unique_clusters) == 1,
            'balance_score': counts.min().item() / counts.max().item() if counts.max() > 0 else 0
        }
        
        # Check for degenerate cases
        if metrics['is_collapsed']:
            metrics['warning'] = "COLLAPSED: All nodes in single cluster"
        elif metrics['has_tiny_clusters']:
            tiny_count = (counts < self.min_size).sum().item()
            metrics['warning'] = f"Has {tiny_count} tiny clusters (< {self.min_size} nodes)"
        elif metrics['balance_score'] < 0.1:
            metrics['warning'] = "Severely imbalanced clusters"
        
        return metrics


class TrainingMonitor:
    """
    Comprehensive training monitoring
    """
    
    def __init__(self, log_interval: int = 5):
        self.log_interval = log_interval
        self.metrics_history = []
        
    def log_epoch(
        self,
        epoch: int,
        phase: str,
        loss: float,
        cluster_metrics: Dict,
        lr: float,
        stability: float
    ):
        """
        Log comprehensive training metrics
        """
        metrics = {
            'epoch': epoch,
            'phase': phase,
            'loss': loss,
            'lr': lr,
            'stability': stability,
            **cluster_metrics
        }
        
        self.metrics_history.append(metrics)
        
        if epoch % self.log_interval == 0:
            self._print_summary(metrics)
    
    def _print_summary(self, metrics: Dict):
        """
        Print training summary
        """
        logger.info(
            f"[{metrics['phase']}] Epoch {metrics['epoch']}: "
            f"Loss={metrics['loss']:.4f}, "
            f"Clusters={metrics.get('num_clusters', 0)}, "
            f"Stability={metrics.get('stability', 0):.2%}, "
            f"Balance={metrics.get('balance_score', 0):.2f}, "
            f"LR={metrics.get('lr', 0):.6f}"
        )
        
        if 'warning' in metrics:
            logger.warning(f"  ⚠️  {metrics['warning']}")
    
    def get_summary(self) -> Dict:
        """
        Get training summary
        """
        if not self.metrics_history:
            return {}
        
        recent = self.metrics_history[-10:]  # Last 10 epochs
        
        return {
            'avg_loss': np.mean([m['loss'] for m in recent]),
            'loss_trend': self._calculate_trend([m['loss'] for m in recent]),
            'avg_clusters': np.mean([m.get('num_clusters', 0) for m in recent]),
            'avg_stability': np.mean([m.get('stability', 0) for m in recent]),
            'has_warnings': any('warning' in m for m in recent)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """
        Calculate trend direction
        """
        if len(values) < 2:
            return 'stable'
        
        # Simple linear regression
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope < -0.01:
            return 'decreasing'
        elif slope > 0.01:
            return 'increasing'
        else:
            return 'stable'