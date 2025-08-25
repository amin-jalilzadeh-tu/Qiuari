"""
Unified GNN Trainer for Energy Community Discovery
Simplified to focus on complementarity-based clustering
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from tqdm import tqdm
import wandb
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class UnifiedGNNTrainer:
    """
    Trainer for Energy GNN focusing on complementarity discovery
    """
    
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        config: Dict[str, Any],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        use_wandb: bool = True
    ):
        """
        Initialize trainer
        
        Args:
            model: GNN model with task heads
            loss_fn: Loss function (UnifiedEnergyLoss)
            config: Training configuration
            device: Device to use
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.config = config
        self.device = device
        self.use_wandb = use_wandb
        
        # Optimizer
        self.optimizer = self._build_optimizer()
        
        # Scheduler
        self.scheduler = self._build_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {
            'complementarity': float('inf'),
            'self_sufficiency': 0,
            'peak_reduction': 0
        }
        self.training_history = []
        
        # Initialize wandb if requested
        if self.use_wandb:
            self._init_wandb()
            
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer based on config"""
        opt_config = self.config.get('optimizer', {})
        opt_type = opt_config.get('type', 'adam')
        
        params = self.model.parameters()
        
        if opt_type.lower() == 'adam':
            return optim.Adam(
                params,
                lr=opt_config.get('lr', 1e-3),
                weight_decay=opt_config.get('weight_decay', 1e-5)
            )
        elif opt_type.lower() == 'adamw':
            return optim.AdamW(
                params,
                lr=opt_config.get('lr', 1e-3),
                weight_decay=opt_config.get('weight_decay', 1e-5)
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
            
    def _build_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler"""
        sched_config = self.config.get('scheduler', {})
        
        if not sched_config:
            return None
            
        sched_type = sched_config.get('type', 'cosine')
        
        if sched_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=sched_config.get('T_max', 100),
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10)
            )
        else:
            return None
            
    def _init_wandb(self):
        """Initialize Weights & Biases logging"""
        wandb.init(
            project=self.config.get('project_name', 'energy-gnn'),
            name=self.config.get('run_name', f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'),
            config=self.config
        )
        wandb.watch(self.model)
        
    def train_epoch(
        self,
        train_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        epoch_metrics = {
            'total_loss': 0,
            'complementarity_loss': 0,
            'physics_loss': 0,
            'peak_loss': 0,
            'sufficiency_loss': 0,
            'quality_loss': 0
        }
        
        num_batches = 0
        
        with tqdm(train_loader, desc=f'Epoch {self.current_epoch}') as pbar:
            for batch_idx, batch in enumerate(pbar):
                batch = batch.to(self.device)
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch)
                
                # Prepare targets from batch
                targets = self._prepare_targets(batch)
                
                # Calculate loss
                loss, loss_components = self.loss_fn(
                    predictions=outputs,
                    targets=targets,
                    graph_data={'adjacency': batch.edge_index}
                )
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                # Optimizer step
                self.optimizer.step()
                
                # Update metrics
                for key in epoch_metrics:
                    if key == 'total_loss':
                        epoch_metrics[key] += loss.item()
                    elif key.replace('_loss', '') in loss_components:
                        epoch_metrics[key] += loss_components[key.replace('_loss', '')].item()
                
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': loss.item(),
                    'comp': loss_components.get('complementarity', 0).item()
                })
                
                # Log batch metrics
                if self.use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'batch_idx': batch_idx + self.current_epoch * len(train_loader)
                    })
        
        # Average metrics
        if num_batches > 0:
            for key in epoch_metrics:
                epoch_metrics[key] /= num_batches
        else:
            logger.warning("No batches in training loader - returning zero metrics")
            
        return epoch_metrics
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        val_metrics = {
            'total_loss': 0,
            'complementarity_score': 0,
            'self_sufficiency': 0,
            'peak_reduction': 0,
            'physics_violations': 0,
            'avg_cluster_size': 0
        }
        
        num_batches = 0
        all_clusters = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                outputs = self.model(batch)
                
                # Prepare targets
                targets = self._prepare_targets(batch)
                
                # Calculate loss
                loss, loss_components = self.loss_fn(
                    predictions=outputs,
                    targets=targets,
                    graph_data={'adjacency': batch.edge_index}
                )
                
                # Calculate additional metrics
                metrics = self._calculate_validation_metrics(outputs, targets, batch)
                
                # Update metrics
                val_metrics['total_loss'] += loss.item()
                val_metrics['complementarity_score'] += metrics['complementarity']
                val_metrics['self_sufficiency'] += metrics['self_sufficiency']
                val_metrics['peak_reduction'] += metrics['peak_reduction']
                val_metrics['physics_violations'] += metrics['physics_violations']
                val_metrics['avg_cluster_size'] += metrics['avg_cluster_size']
                
                # Store cluster assignments for analysis
                if 'clustering_cluster_probs' in outputs:
                    clusters = torch.argmax(outputs['clustering_cluster_probs'], dim=-1)
                    all_clusters.append(clusters.cpu())
                
                num_batches += 1
        
        # Average metrics
        if num_batches > 0:
            for key in val_metrics:
                val_metrics[key] /= num_batches
        else:
            logger.warning("No batches in validation loader - returning zero metrics")
            
        # Calculate cluster statistics
        if all_clusters:
            all_clusters = torch.cat(all_clusters)
            val_metrics['num_clusters'] = len(torch.unique(all_clusters))
            
        return val_metrics
    
    def _calculate_validation_metrics(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch: Data
    ) -> Dict[str, float]:
        """
        Calculate detailed validation metrics
        
        Args:
            outputs: Model outputs
            targets: Target values
            batch: Batch data
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Get cluster assignments
        if 'clustering_cluster_probs' in outputs:
            cluster_probs = outputs['clustering_cluster_probs']
            clusters = torch.argmax(cluster_probs, dim=-1)
            
            # Average cluster size
            unique_clusters = torch.unique(clusters)
            avg_size = len(clusters) / len(unique_clusters) if len(unique_clusters) > 0 else 0
            metrics['avg_cluster_size'] = avg_size
        else:
            metrics['avg_cluster_size'] = 0
            
        # Complementarity score (average negative correlation)
        if 'clustering_complementarity_scores' in outputs:
            comp_scores = outputs['clustering_complementarity_scores']
            metrics['complementarity'] = -comp_scores.mean().item()  # Negative for complementarity
        else:
            metrics['complementarity'] = 0
            
        # Self-sufficiency calculation
        if 'generation' in targets and 'demand' in targets:
            gen = targets['generation']
            demand = targets['demand']
            
            # Calculate per cluster
            self_suff = 0
            for k in unique_clusters:
                mask = clusters == k
                if mask.sum() > 0:
                    cluster_gen = gen[mask].sum()
                    cluster_demand = demand[mask].sum()
                    if cluster_demand > 0:
                        self_suff += min(cluster_gen, cluster_demand) / cluster_demand
                        
            metrics['self_sufficiency'] = self_suff / len(unique_clusters) if len(unique_clusters) > 0 else 0
        else:
            metrics['self_sufficiency'] = 0
            
        # Peak reduction
        if 'individual_peaks' in targets:
            individual_peaks = targets['individual_peaks']
            
            peak_reduction = 0
            for k in unique_clusters:
                mask = clusters == k
                if mask.sum() > 0:
                    individual_sum = individual_peaks[mask].sum()
                    # Simplified: assume cluster peak is 75% of sum (to be refined)
                    cluster_peak = 0.75 * individual_sum
                    reduction = 1 - (cluster_peak / individual_sum)
                    peak_reduction += reduction
                    
            metrics['peak_reduction'] = peak_reduction / len(unique_clusters) if len(unique_clusters) > 0 else 0
        else:
            metrics['peak_reduction'] = 0
            
        # Physics violations
        if 'power_flow' in targets:
            power_flow = targets['power_flow']
            
            # Check energy balance
            imbalance = power_flow.sum(dim=-1).abs()
            violations = (imbalance > 0.01).float().mean()
            
            # Check transformer limits if available
            if 'transformer_capacity' in targets:
                capacity = targets['transformer_capacity']
                overload = (power_flow.abs() > capacity.unsqueeze(-1)).float().mean()
                violations += overload
                
            metrics['physics_violations'] = violations.item()
        else:
            metrics['physics_violations'] = 0
            
        return metrics
    
    def _prepare_targets(self, batch: Data) -> Dict[str, torch.Tensor]:
        """
        Prepare target tensors from batch
        
        Args:
            batch: Batch data
            
        Returns:
            Dictionary of target tensors
        """
        targets = {}
        
        # Extract available targets from batch
        if hasattr(batch, 'temporal_profiles'):
            targets['temporal_profiles'] = batch.temporal_profiles
            
        if hasattr(batch, 'power_flow'):
            targets['power_flow'] = batch.power_flow
            
        if hasattr(batch, 'generation'):
            targets['generation'] = batch.generation
            
        if hasattr(batch, 'demand'):
            targets['demand'] = batch.demand
            
        if hasattr(batch, 'individual_peaks'):
            targets['individual_peaks'] = batch.individual_peaks
            
        if hasattr(batch, 'transformer_capacity'):
            targets['transformer_capacity'] = batch.transformer_capacity
            
        if hasattr(batch, 'line_capacity'):
            targets['line_capacity'] = batch.line_capacity
            
        return targets
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int
    ) -> Dict[str, Any]:
        """
        Full training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs to train
            
        Returns:
            Training history and best model
        """
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_metrics = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
                    
            # Log metrics
            metrics = {
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.training_history.append(metrics)
            
            # Print progress
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            print(f"Val Loss: {val_metrics['total_loss']:.4f}")
            print(f"Complementarity: {val_metrics['complementarity_score']:.4f}")
            print(f"Self-Sufficiency: {val_metrics['self_sufficiency']:.4f}")
            print(f"Peak Reduction: {val_metrics['peak_reduction']:.4f}")
            print(f"Physics Violations: {val_metrics['physics_violations']:.4f}")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total_loss'],
                    'val_loss': val_metrics['total_loss'],
                    'val_complementarity': val_metrics['complementarity_score'],
                    'val_self_sufficiency': val_metrics['self_sufficiency'],
                    'val_peak_reduction': val_metrics['peak_reduction'],
                    'val_physics_violations': val_metrics['physics_violations'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
                
            # Check for best model
            if self._is_best_model(val_metrics):
                self.best_metrics = val_metrics.copy()
                self.save_checkpoint('best_model.pt')
                print("[SAVED] New best model saved!")
                
            # Early stopping check
            if self._should_stop_early(val_metrics):
                print("Early stopping triggered")
                break
                
        print("\nTraining completed!")
        return {
            'history': self.training_history,
            'best_metrics': self.best_metrics
        }
    
    def _is_best_model(self, metrics: Dict[str, float]) -> bool:
        """Check if current model is best so far"""
        # Prioritize complementarity and self-sufficiency
        score = (
            -metrics.get('complementarity_score', 0) * 0.4 +  # Want negative correlation
            metrics.get('self_sufficiency', 0) * 0.3 +
            metrics.get('peak_reduction', 0) * 0.2 -
            metrics.get('physics_violations', 0) * 0.1
        )
        
        best_score = (
            -self.best_metrics.get('complementarity', 0) * 0.4 +
            self.best_metrics.get('self_sufficiency', 0) * 0.3 +
            self.best_metrics.get('peak_reduction', 0) * 0.2 -
            self.best_metrics.get('physics_violations', float('inf')) * 0.1
        )
        
        return score > best_score
    
    def _should_stop_early(self, metrics: Dict[str, float]) -> bool:
        """Check early stopping criteria"""
        if not self.config.get('early_stopping', {}).get('enabled', False):
            return False
            
        patience = self.config['early_stopping'].get('patience', 20)
        
        # Check if we haven't improved for patience epochs
        if len(self.training_history) < patience:
            return False
            
        recent_losses = [h['val']['total_loss'] for h in self.training_history[-patience:]]
        
        # Stop if loss hasn't decreased
        return all(loss >= recent_losses[0] for loss in recent_losses[1:])
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_dir = Path(self.config.get('checkpoint_dir', 'checkpoints'))
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metrics': self.best_metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_dir / filename)
        
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        checkpoint_path = Path(self.config.get('checkpoint_dir', 'checkpoints')) / filename
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        self.current_epoch = checkpoint['epoch']
        self.best_metrics = checkpoint['best_metrics']
        
        print(f"Loaded checkpoint from epoch {self.current_epoch}")