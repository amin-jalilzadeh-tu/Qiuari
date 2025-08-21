# training/multi_task_trainer.py
"""
Multi-objective training loop for energy GNN
Handles multiple tasks with dynamic weight balancing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging
import time
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)

class MultiTaskTrainer:
    """Multi-task trainer for energy GNN"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 device: str = 'cuda'):
        """
        Initialize multi-task trainer
        
        Args:
            model: Multi-task GNN model
            config: Training configuration
            device: Device to train on
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Training parameters
        self.epochs = config['training']['epochs']
        self.learning_rate = config['training']['learning_rate']
        self.batch_size = config['training']['batch_size']
        self.early_stopping_patience = config['training']['early_stopping_patience']
        
        # Task weights (will be dynamically adjusted)
        self.task_weights = config['training']['task_weights'].copy()
        self.initial_weights = self.task_weights.copy()
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Loss functions
        from .loss_functions import MultiTaskLoss
        self.loss_fn = MultiTaskLoss(config)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.task_losses = {task: [] for task in self.task_weights.keys()}
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Tensorboard
        self.writer = SummaryWriter(f"runs/energy_gnn_{time.strftime('%Y%m%d_%H%M%S')}")
        
        # Checkpointing
        self.checkpoint_dir = Path(config['paths']['model_checkpoints'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized MultiTaskTrainer on {device}")
        logger.info(f"Training for {self.epochs} epochs with batch size {self.batch_size}")
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer"""
        optimizer_type = self.config['training']['optimizer']
        weight_decay = self.config['training']['weight_decay']
        
        if optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        scheduler_type = self.config['training']['scheduler']
        
        if scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=1e-6
            )
        elif scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.5
            )
        elif scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10
            )
        else:
            return optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1.0
            )
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              test_loader: Optional[DataLoader] = None) -> Dict:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            
        Returns:
            Training history and final metrics
        """
        logger.info("Starting training...")
        
        for epoch in range(self.epochs):
            # Training phase
            train_loss, train_task_losses = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            if val_loader is not None:
                val_loss, val_task_losses, val_metrics = self._validate(val_loader, epoch)
                
                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self._save_checkpoint('best_model.pth', epoch, val_loss)
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            else:
                val_loss = train_loss
                val_task_losses = train_task_losses
                val_metrics = {}
            
            # Update task weights dynamically
            self._update_task_weights(train_task_losses, epoch)
            
            # Learning rate scheduling
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()
            
            # Logging
            self._log_epoch(epoch, train_loss, val_loss, train_task_losses, val_task_losses)
            
            # Periodic checkpointing
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_loss)
        
        # Final evaluation
        final_metrics = {}
        if test_loader is not None:
            logger.info("Running final evaluation on test set...")
            test_loss, test_task_losses, test_metrics = self._validate(test_loader, self.epochs)
            final_metrics['test'] = {
                'loss': test_loss,
                'task_losses': test_task_losses,
                'metrics': test_metrics
            }
        
        # Save final model
        self._save_checkpoint('final_model.pth', self.epochs, val_loss)
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'task_losses': self.task_losses,
            'final_metrics': final_metrics
        }
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Tuple[float, Dict]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        task_losses = {task: 0 for task in self.task_weights.keys()}
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.epochs}")
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            # Check if model expects separate x and edge_index or a batch object
            if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                outputs = self.model(batch.x, batch.edge_index)
            else:
                outputs = self.model(batch)
            
            # Wrap tensor output in dictionary if needed
            if isinstance(outputs, torch.Tensor):
                outputs = {'predictions': outputs}
            
            # Calculate losses
            losses = self.loss_fn(outputs, batch, self.task_weights)
            
            # Total weighted loss
            total_batch_loss = losses['total']
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )
            
            # Optimizer step
            self.optimizer.step()
            
            # Track losses
            total_loss += total_batch_loss.item()
            for task in task_losses.keys():
                if task in losses:
                    task_losses[task] += losses[task].item()
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': total_batch_loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
        
        self.train_losses.append(avg_loss)
        for task, loss in avg_task_losses.items():
            self.task_losses[task].append(loss)
        
        return avg_loss, avg_task_losses
    
    def _validate(self, val_loader: DataLoader, epoch: int) -> Tuple[float, Dict, Dict]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        task_losses = {task: 0 for task in self.task_weights.keys()}
        num_batches = 0
        
        # Metrics accumulation
        from .evaluation_metrics import EvaluationMetrics
        metrics_calculator = EvaluationMetrics(self.config)
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                # Forward pass
                # Check if model expects separate x and edge_index or a batch object
                if hasattr(batch, 'x') and hasattr(batch, 'edge_index'):
                    outputs = self.model(batch.x, batch.edge_index)
                else:
                    outputs = self.model(batch)
                
                # Wrap tensor output in dictionary if needed
                if isinstance(outputs, torch.Tensor):
                    outputs = {'predictions': outputs}
                
                # Calculate losses
                losses = self.loss_fn(outputs, batch, self.task_weights)
                
                # Track losses
                total_loss += losses['total'].item()
                for task in task_losses.keys():
                    if task in losses:
                        task_losses[task] += losses[task].item()
                
                # Calculate metrics
                metrics_calculator.update(outputs, batch)
                
                num_batches += 1
        
        # Average losses
        avg_loss = total_loss / num_batches
        avg_task_losses = {task: loss / num_batches for task, loss in task_losses.items()}
        
        # Compute final metrics
        metrics = metrics_calculator.compute()
        
        self.val_losses.append(avg_loss)
        
        return avg_loss, avg_task_losses, metrics
    
    def _update_task_weights(self, task_losses: Dict, epoch: int):
        """Dynamically update task weights based on loss magnitudes"""
        if epoch < 10:  # Don't update in early epochs
            return
        
        # Calculate relative loss magnitudes
        loss_values = np.array(list(task_losses.values()))
        loss_values = loss_values[loss_values > 0]  # Filter out zero losses
        
        if len(loss_values) > 0:
            # Normalize by initial weights to get relative progress
            relative_progress = {}
            for task, loss in task_losses.items():
                if task in self.initial_weights and loss > 0:
                    relative_progress[task] = loss / self.initial_weights[task]
            
            if relative_progress:
                # Tasks with higher relative loss get more weight
                mean_progress = np.mean(list(relative_progress.values()))
                
                for task in self.task_weights.keys():
                    if task in relative_progress:
                        # Gradual adjustment
                        adjustment = relative_progress[task] / mean_progress
                        adjustment = np.clip(adjustment, 0.5, 2.0)  # Limit adjustment range
                        
                        # Exponential moving average
                        alpha = 0.1
                        self.task_weights[task] = (
                            (1 - alpha) * self.task_weights[task] +
                            alpha * self.initial_weights[task] * adjustment
                        )
                
                # Normalize weights
                total_weight = sum(self.task_weights.values())
                for task in self.task_weights:
                    self.task_weights[task] /= total_weight
    
    def _log_epoch(self, epoch: int, train_loss: float, val_loss: float,
                   train_task_losses: Dict, val_task_losses: Dict):
        """Log epoch metrics"""
        # Console logging
        logger.info(f"Epoch {epoch+1}/{self.epochs}")
        logger.info(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"  Task Weights: {self.task_weights}")
        
        # Tensorboard logging
        self.writer.add_scalar('Loss/train', train_loss, epoch)
        self.writer.add_scalar('Loss/validation', val_loss, epoch)
        self.writer.add_scalar('Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
        
        for task, loss in train_task_losses.items():
            self.writer.add_scalar(f'TaskLoss_train/{task}', loss, epoch)
        
        for task, loss in val_task_losses.items():
            self.writer.add_scalar(f'TaskLoss_val/{task}', loss, epoch)
        
        for task, weight in self.task_weights.items():
            self.writer.add_scalar(f'TaskWeight/{task}', weight, epoch)
    
    def _save_checkpoint(self, filename: str, epoch: int, val_loss: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'task_weights': self.task_weights,
            'config': self.config
        }
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.task_weights = checkpoint['task_weights']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch'], checkpoint['val_loss']

class AdaptiveMultiTaskTrainer(MultiTaskTrainer):
    """Advanced trainer with adaptive strategies"""
    
    def __init__(self, model: nn.Module, config: Dict, device: str = 'cuda'):
        super().__init__(model, config, device)
        
        # Additional adaptive features
        self.use_uncertainty_weighting = config['training'].get('uncertainty_weighting', False)
        self.use_gradient_surgery = config['training'].get('gradient_surgery', False)
        self.use_curriculum_learning = config['training'].get('curriculum_learning', False)
        
        if self.use_uncertainty_weighting:
            self._init_uncertainty_weights()
    
    def _init_uncertainty_weights(self):
        """Initialize learnable uncertainty weights"""
        self.log_vars = nn.ParameterDict({
            task: nn.Parameter(torch.zeros(1))
            for task in self.task_weights.keys()
        })
        
        # Add to optimizer
        self.optimizer.add_param_group({
            'params': self.log_vars.parameters(),
            'lr': self.learning_rate * 0.1
        })
    
    def _calculate_uncertainty_weighted_loss(self, losses: Dict) -> torch.Tensor:
        """Calculate loss with uncertainty weighting"""
        total_loss = 0
        
        for task, loss in losses.items():
            if task in self.log_vars:
                precision = torch.exp(-self.log_vars[task])
                total_loss += precision * loss + self.log_vars[task]
        
        return total_loss
    
    def _apply_gradient_surgery(self):
        """Apply gradient surgery to handle conflicting gradients"""
        # Get gradients for each task
        task_gradients = {}
        
        for task in self.task_weights.keys():
            task_grad = []
            for param in self.model.parameters():
                if param.grad is not None:
                    task_grad.append(param.grad.clone().flatten())
            
            if task_grad:
                task_gradients[task] = torch.cat(task_grad)
        
        # Check for conflicts and project gradients
        for task1 in task_gradients:
            for task2 in task_gradients:
                if task1 != task2:
                    # Calculate cosine similarity
                    cos_sim = torch.dot(task_gradients[task1], task_gradients[task2])
                    cos_sim /= (task_gradients[task1].norm() * task_gradients[task2].norm() + 1e-8)
                    
                    # If gradients conflict (negative cosine similarity)
                    if cos_sim < 0:
                        # Project task2's gradient
                        proj = torch.dot(task_gradients[task1], task_gradients[task2])
                        proj /= (task_gradients[task1].norm() ** 2 + 1e-8)
                        task_gradients[task2] = task_gradients[task2] - proj * task_gradients[task1]
        
        # Apply modified gradients back
        idx = 0
        for param in self.model.parameters():
            if param.grad is not None:
                param_size = param.grad.numel()
                param.grad = task_gradients[list(task_gradients.keys())[0]][idx:idx+param_size].view_as(param.grad)
                idx += param_size

# Usage example
if __name__ == "__main__":
    import yaml
    from models.base_gnn import create_gnn_model
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    model = create_gnn_model('homo', config['model'])
    
    # Create trainer
    trainer = MultiTaskTrainer(model, config)
    
    # Mock data loaders
    from torch_geometric.data import Data, Batch
    
    def create_mock_batch():
        data_list = []
        for _ in range(32):
            x = torch.randn(100, 45)
            edge_index = torch.randint(0, 100, (2, 200))
            y = torch.randn(100, 1)
            data = Data(x=x, edge_index=edge_index, y=y)
            data_list.append(data)
        return Batch.from_data_list(data_list)
    
    train_loader = [create_mock_batch() for _ in range(10)]
    val_loader = [create_mock_batch() for _ in range(5)]
    
    # Train
    history = trainer.train(train_loader, val_loader)
    print(f"Training complete. Final validation loss: {history['val_losses'][-1]:.4f}")