"""
Discovery-focused GNN Trainer for Energy Community Discovery
Simplified trainer focusing on unsupervised discovery without ground truth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict
from tqdm import tqdm
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DiscoveryGNNTrainer:
    """
    Simplified trainer focused on unsupervised discovery
    No ground truth needed - pure pattern discovery
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize discovery trainer
        
        Args:
            model: GNN model with discovery heads
            config: Training configuration
            device: Device to use
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Discovery-only loss
        from training.loss_functions import DiscoveryLoss
        self.criterion = DiscoveryLoss(config)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('epochs', 100),
            eta_min=1e-6
        )
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {
            'self_sufficiency': 0,
            'peak_reduction': 0,
            'avg_complementarity': 1.0  # Start high (want negative)
        }
        self.training_history = []
        
        logger.info(f"Initialized DiscoveryGNNTrainer on {device}")
        
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average loss and loss components
        """
        self.model.train()
        total_loss = 0
        loss_components = defaultdict(float)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch = batch.to(self.device)
            
            # Forward pass with DiffPool
            predictions = self._forward_with_diffpool(batch)
            
            # Calculate physics data from batch
            physics_data = self._extract_physics_data(batch)
            
            # Calculate loss
            loss, components = self.criterion(predictions, physics_data, batch)
            
            # Add auxiliary loss from DiffPool if present
            if 'aux_loss' in predictions:
                loss = loss + 0.1 * predictions['aux_loss']
                components['aux_loss'] = predictions['aux_loss'].item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            for k, v in components.items():
                loss_components[k] += v.item() if torch.is_tensor(v) else v
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'comp': f"{components.get('complementarity', 0):.4f}"
            })
        
        # Average metrics
        avg_loss = total_loss / len(train_loader)
        avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
        
        self.current_epoch += 1
        self.scheduler.step()
        
        return avg_loss, avg_components
    
    def _forward_with_diffpool(self, batch: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass including DiffPool clustering
        
        Args:
            batch: Input batch
            
        Returns:
            Predictions dictionary
        """
        # Get model output directly - it handles everything
        output = self.model(batch)
        
        # If output is already a dict with predictions, return it
        if isinstance(output, dict):
            return output
        
        # Otherwise, create predictions dict
        predictions = {}
        
        # Extract embeddings
        if isinstance(output, dict):
            h = output.get('building', output.get('embeddings', output.get('x')))
        else:
            h = output
        
        predictions['embeddings'] = h
        
        return predictions
    
    def _extract_physics_data(self, batch: Data) -> Dict[str, torch.Tensor]:
        """
        Extract physics data from batch
        
        Args:
            batch: Input batch
            
        Returns:
            Physics data dictionary
        """
        physics_data = {}
        
        # Assuming features are ordered as:
        # [demand, generation, ...]
        if hasattr(batch, 'x') and batch.x.size(1) >= 2:
            physics_data['demand'] = batch.x[:, 0]
            physics_data['generation'] = batch.x[:, 1]
            physics_data['net_load'] = batch.x[:, 0] - batch.x[:, 1]
        
        return physics_data
    
    def _create_transformer_mask(self, batch: Data) -> Optional[torch.Tensor]:
        """
        Create transformer boundary mask
        
        Args:
            batch: Input batch
            
        Returns:
            Transformer mask or None
        """
        if not hasattr(batch, 'transformer_id'):
            return None
        
        N = batch.x.size(0)
        K = self.config.get('max_clusters', 20)
        
        # Create mask
        mask = torch.ones(N, K, device=self.device)
        
        # Group nodes by transformer
        unique_transformers = torch.unique(batch.transformer_id)
        clusters_per_transformer = K // len(unique_transformers)
        
        # Assign cluster ranges to transformers
        for i, t_id in enumerate(unique_transformers):
            nodes_mask = (batch.transformer_id == t_id)
            cluster_start = i * clusters_per_transformer
            cluster_end = min((i + 1) * clusters_per_transformer, K)
            
            # Block assignments outside transformer's cluster range
            mask[nodes_mask, :cluster_start] = 0
            mask[nodes_mask, cluster_end:] = 0
        
        return mask
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Validation metrics
        """
        self.model.eval()
        metrics = defaultdict(list)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self._forward_with_diffpool(batch)
                physics_data = self._extract_physics_data(batch)
                
                # Calculate discovery metrics (no ground truth needed)
                metrics['self_sufficiency'].append(
                    self._calculate_self_sufficiency(predictions, physics_data)
                )
                metrics['peak_reduction'].append(
                    self._calculate_peak_reduction(predictions, physics_data)
                )
                
                if 'complementarity' in predictions:
                    metrics['avg_complementarity'].append(
                        torch.mean(predictions['complementarity']).item()
                    )
                
                if 'clusters' in predictions:
                    cluster_sizes = self._get_cluster_sizes(predictions['clusters'])
                    metrics['avg_cluster_size'].append(np.mean(cluster_sizes))
                    metrics['min_cluster_size'].append(np.min(cluster_sizes))
                    metrics['max_cluster_size'].append(np.max(cluster_sizes))
        
        # Average metrics
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _calculate_self_sufficiency(
        self,
        predictions: Dict[str, torch.Tensor],
        physics_data: Dict[str, torch.Tensor]
    ) -> float:
        """
        Calculate self-sufficiency metric
        
        Args:
            predictions: Model predictions
            physics_data: Physics data
            
        Returns:
            Self-sufficiency ratio
        """
        if 'clusters' not in predictions or 'demand' not in physics_data:
            return 0.0
        
        S = predictions['clusters']  # Soft assignments [N, K]
        demand = physics_data.get('demand', torch.zeros_like(S[:, 0]))
        generation = physics_data.get('generation', torch.zeros_like(S[:, 0]))
        
        # Aggregate per cluster
        cluster_demand = torch.matmul(S.T, demand)
        cluster_generation = torch.matmul(S.T, generation)
        
        # Calculate self-sufficiency
        self_consumed = torch.min(cluster_demand, cluster_generation)
        self_sufficiency = torch.sum(self_consumed) / (torch.sum(cluster_demand) + 1e-8)
        
        return self_sufficiency.item()
    
    def _calculate_peak_reduction(
        self,
        predictions: Dict[str, torch.Tensor],
        physics_data: Dict[str, torch.Tensor]
    ) -> float:
        """
        Calculate peak reduction metric
        
        Args:
            predictions: Model predictions
            physics_data: Physics data
            
        Returns:
            Peak reduction ratio
        """
        if 'clusters' not in predictions or 'demand' not in physics_data:
            return 0.0
        
        S = predictions['clusters']
        demand = physics_data['demand']
        
        # Original peak (sum of individual peaks)
        peak_original = torch.sum(demand)
        
        # Clustered peak
        cluster_demand = torch.matmul(S.T, demand)
        peak_clustered = torch.max(cluster_demand)
        
        # Reduction ratio
        reduction = (peak_original - peak_clustered) / (peak_original + 1e-8)
        
        return reduction.item()
    
    def _get_cluster_sizes(self, clusters: torch.Tensor) -> List[int]:
        """
        Get cluster sizes from soft assignments
        
        Args:
            clusters: Soft assignment matrix [N, K]
            
        Returns:
            List of cluster sizes
        """
        # Convert to hard assignments
        hard_assignments = torch.argmax(clusters, dim=1).cpu().numpy()
        
        # Count members per cluster
        unique, counts = np.unique(hard_assignments, return_counts=True)
        
        return counts.tolist()
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """
        Save training checkpoint
        
        Args:
            path: Save path
            is_best: Whether this is the best model
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metrics': self.best_metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = Path(path).parent / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint
        
        Args:
            path: Checkpoint path
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metrics = checkpoint['best_metrics']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        save_dir: str = './checkpoints'
    ):
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            save_dir: Directory to save checkpoints
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Train epoch
            train_loss, train_components = self.train_epoch(train_loader)
            
            # Log training metrics
            logger.info(f"Epoch {epoch}: Loss={train_loss:.4f}")
            for k, v in train_components.items():
                logger.info(f"  {k}: {v:.4f}")
            
            # Validate if loader provided
            if val_loader:
                val_metrics = self.validate(val_loader)
                
                logger.info("Validation metrics:")
                for k, v in val_metrics.items():
                    logger.info(f"  {k}: {v:.4f}")
                
                # Check if best model
                is_best = False
                if val_metrics.get('self_sufficiency', 0) > self.best_metrics['self_sufficiency']:
                    self.best_metrics['self_sufficiency'] = val_metrics['self_sufficiency']
                    is_best = True
                
                if val_metrics.get('avg_complementarity', 1) < self.best_metrics['avg_complementarity']:
                    self.best_metrics['avg_complementarity'] = val_metrics['avg_complementarity']
                    is_best = True
                
                # Save checkpoint
                if epoch % 10 == 0 or is_best:
                    checkpoint_path = save_path / f'checkpoint_epoch_{epoch}.pt'
                    self.save_checkpoint(checkpoint_path, is_best=is_best)
            
            # Store history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_components': train_components,
                'val_metrics': val_metrics if val_loader else None
            })
        
        logger.info("Training completed!")
        logger.info(f"Best metrics: {self.best_metrics}")


if __name__ == "__main__":
    # Test trainer
    from models.base_gnn import create_gnn_model
    
    # Create dummy config
    config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'max_clusters': 10,
        'min_cluster_size': 3,
        'max_cluster_size': 20,
        'learning_rate': 1e-3,
        'epochs': 100,
        'alpha_complementarity': 2.0,
        'alpha_physics': 1.0,
        'alpha_clustering': 1.5,
        'alpha_peak': 1.0,
        'alpha_coverage': 0.5
    }
    
    # Create model
    model = create_gnn_model('hetero', config)
    
    # Create trainer
    trainer = DiscoveryGNNTrainer(model, config)
    
    print("✅ Discovery trainer initialized successfully!")