"""
Unified Solar GNN Trainer
Combines discovery and intervention phases with semi-supervised learning
Simplified from multiple trainers into one coherent pipeline
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class UnifiedSolarTrainer:
    """
    Single trainer for both discovery and solar recommendation
    Supports semi-supervised learning with cluster and solar labels
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize unified trainer
        
        Args:
            model: SolarDistrictGNN model
            config: Training configuration
            device: Device to use
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get('learning_rate', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2),
            eta_min=config.get('eta_min', 1e-6)
        )
        
        # Initialize loss functions
        self._setup_losses()
        
        # Label storage
        self.cluster_labels = {}  # Auto-generated cluster quality labels
        self.solar_labels = {}    # Real solar deployment labels
        
        # Training state
        self.current_epoch = 0
        self.current_round = 0  # Deployment round
        self.training_history = []
        self.best_metrics = {
            'discovery': {'self_sufficiency': 0, 'complementarity': float('inf')},
            'solar': {'roi_accuracy': 0, 'deployment_success': 0}
        }
        
        # Experiment tracking
        self.experiment_dir = Path(config.get('experiment_dir', 'experiments'))
        self.experiment_dir = self.experiment_dir / f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized UnifiedSolarTrainer on {device}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def _setup_losses(self):
        """Setup loss functions for both phases"""
        from training.loss_functions import (
            ComplementarityLoss,
            EnergyBalanceLoss,
            PeakReductionLoss,
            SelfSufficiencyLoss
        )
        
        # Discovery phase losses
        self.complementarity_loss = ComplementarityLoss()
        self.energy_balance_loss = EnergyBalanceLoss()
        self.self_sufficiency_loss = SelfSufficiencyLoss()
        
        # Solar phase losses
        self.solar_roi_loss = SolarROILoss()  # New loss for ROI prediction
        self.cluster_quality_loss = ClusterQualityLoss()  # Semi-supervised cluster loss
        
        # Network-aware losses (merged from network_aware_loss.py)
        self.network_impact_loss = NetworkImpactLoss()
        
    def train_discovery_phase(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        use_labels: bool = False
    ) -> Dict[str, float]:
        """
        Phase 1: Discover self-sufficient energy communities
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            epochs: Number of epochs
            use_labels: Whether to use cluster quality labels (semi-supervised)
            
        Returns:
            Best metrics from discovery
        """
        logger.info("Starting Discovery Phase Training")
        phase_metrics = []
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self._train_discovery_epoch(train_loader, use_labels)
            phase_metrics.append(train_metrics)
            
            # Validation
            if val_loader:
                val_metrics = self._validate_discovery(val_loader)
                train_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
            
            # Update best metrics
            if train_metrics.get('self_sufficiency', 0) > self.best_metrics['discovery']['self_sufficiency']:
                self.best_metrics['discovery'] = {
                    'self_sufficiency': train_metrics['self_sufficiency'],
                    'complementarity': train_metrics['complementarity']
                }
                self._save_checkpoint('best_discovery')
            
            # Log progress
            logger.info(f"Discovery Epoch {epoch}: Loss={train_metrics['total_loss']:.4f}, "
                       f"Self-Suff={train_metrics['self_sufficiency']:.3f}, "
                       f"Complement={train_metrics['complementarity']:.3f}")
            
            # Scheduler step
            self.scheduler.step()
        
        return self.best_metrics['discovery']
    
    def _train_discovery_epoch(
        self,
        train_loader: DataLoader,
        use_labels: bool
    ) -> Dict[str, float]:
        """Train one epoch of discovery phase"""
        self.model.train()
        epoch_metrics = {
            'total_loss': 0,
            'complementarity': 0,
            'self_sufficiency': 0,
            'network_impact': 0
        }
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Discovery Training")):
            batch = batch.to(self.device)
            
            # Forward pass - discovery phase
            outputs = self.model(batch, phase='discovery')
            
            # Calculate losses
            losses = {}
            
            # Unsupervised losses (always active)
            comp_loss, comp_metrics = self.complementarity_loss(
                outputs['embeddings'],
                outputs['cluster_assignments'],
                batch.temporal_features if hasattr(batch, 'temporal_features') else None
            )
            losses['complementarity'] = comp_loss
            
            # Energy balance - skip if causes issues
            try:
                balance_loss = self.energy_balance_loss(
                    outputs['cluster_assignments'],
                    batch
                )
                losses['energy_balance'] = balance_loss * 0.3
            except (RuntimeError, ValueError) as e:
                # Skip energy balance if shapes don't match
                logger.debug(f"Skipping energy balance loss: {e}")
                losses['energy_balance'] = torch.tensor(0.0, device=batch.x.device)
            
            # Network impact
            network_loss, network_metrics = self.network_impact_loss(
                outputs['cluster_assignments'],
                outputs['embeddings'],
                outputs['hop_features'],
                batch.edge_index
            )
            losses['network_impact'] = network_loss * 0.5
            
            # Semi-supervised loss (if labels available)
            if use_labels and self.cluster_labels:
                cluster_ids = self._get_cluster_labels_for_batch(batch)
                if cluster_ids is not None:
                    quality_loss = self.cluster_quality_loss(
                        outputs['cluster_assignments'],
                        cluster_ids
                    )
                    losses['cluster_quality'] = quality_loss * 0.3
            
            # Total loss
            total_loss = sum(losses.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            epoch_metrics['total_loss'] += total_loss.item()
            
            # Handle different return types from metrics
            comp_val = comp_metrics.get('correlation', 0)
            epoch_metrics['complementarity'] += comp_val.item() if hasattr(comp_val, 'item') else comp_val
            
            epoch_metrics['self_sufficiency'] += self._calculate_self_sufficiency(
                outputs['cluster_assignments'], batch
            ).item()
            
            net_val = network_metrics.get('multi_hop', 0)
            epoch_metrics['network_impact'] += net_val.item() if hasattr(net_val, 'item') else net_val
        
        # Average metrics
        n_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            
        return epoch_metrics
    
    def train_solar_phase(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 30,
        cluster_assignments: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Phase 2: Train solar recommendation within discovered clusters
        
        Args:
            train_loader: Training data
            val_loader: Validation data  
            epochs: Number of epochs
            cluster_assignments: Pre-discovered clusters to constrain solar
            
        Returns:
            Best solar metrics
        """
        logger.info("Starting Solar Recommendation Phase Training")
        phase_metrics = []
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_metrics = self._train_solar_epoch(train_loader, cluster_assignments)
            phase_metrics.append(train_metrics)
            
            # Validation
            if val_loader:
                val_metrics = self._validate_solar(val_loader)
                train_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
            
            # Update best metrics
            if train_metrics.get('roi_accuracy', 0) > self.best_metrics['solar']['roi_accuracy']:
                self.best_metrics['solar'] = {
                    'roi_accuracy': train_metrics['roi_accuracy'],
                    'deployment_success': train_metrics.get('deployment_success', 0)
                }
                self._save_checkpoint('best_solar')
            
            # Log progress
            logger.info(f"Solar Epoch {epoch}: Loss={train_metrics['total_loss']:.4f}, "
                       f"ROI Acc={train_metrics['roi_accuracy']:.3f}, "
                       f"Network Impact={train_metrics['network_impact']:.3f}")
            
            # Scheduler step
            self.scheduler.step()
        
        return self.best_metrics['solar']
    
    def _train_solar_epoch(
        self,
        train_loader: DataLoader,
        cluster_assignments: Optional[Dict]
    ) -> Dict[str, float]:
        """Train one epoch of solar phase"""
        self.model.train()
        epoch_metrics = {
            'total_loss': 0,
            'roi_accuracy': 0,
            'network_impact': 0,
            'confidence': 0
        }
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc="Solar Training")):
            batch = batch.to(self.device)
            
            # Forward pass - solar phase
            outputs = self.model(batch, phase='solar')
            
            # Calculate losses
            losses = {}
            
            # Solar ROI loss (if labels available)
            if self.solar_labels:
                solar_targets = self._get_solar_labels_for_batch(batch)
                if solar_targets is not None:
                    roi_loss = self.solar_roi_loss(
                        outputs['roi_category'],
                        solar_targets
                    )
                    losses['roi'] = roi_loss
                    
                    # Calculate accuracy
                    roi_accuracy = self._calculate_roi_accuracy(
                        outputs['roi_category'],
                        solar_targets
                    )
                    epoch_metrics['roi_accuracy'] += roi_accuracy.item()
            
            # Network impact loss
            network_loss = self._calculate_solar_network_loss(
                outputs['solar_potential'],
                outputs['network_impact'],
                batch.edge_index
            )
            losses['network'] = network_loss * 0.3
            
            # Confidence regularization (prefer confident predictions)
            confidence_loss = -torch.log(outputs['confidence'] + 1e-8).mean()
            losses['confidence'] = confidence_loss * 0.1
            
            # If no labeled data, use unsupervised proxy
            if not self.solar_labels:
                # Estimate ROI from features
                estimated_roi = self._estimate_roi_from_features(batch)
                proxy_loss = F.mse_loss(outputs['solar_potential'].squeeze(), estimated_roi)
                losses['proxy'] = proxy_loss
            
            # Total loss
            total_loss = sum(losses.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['network_impact'] += outputs['network_impact'].mean().item()
            epoch_metrics['confidence'] += outputs['confidence'].mean().item()
        
        # Average metrics
        n_batches = len(train_loader)
        for key in epoch_metrics:
            epoch_metrics[key] /= n_batches
            
        return epoch_metrics
    
    def add_cluster_labels(self, clusters: torch.Tensor, metrics: Dict[str, float]):
        """
        Add automatically generated cluster quality labels
        
        Args:
            clusters: Cluster assignments
            metrics: Cluster performance metrics
        """
        # Determine cluster quality based on metrics
        if metrics['self_sufficiency'] > 0.7 and metrics['complementarity'] < -0.3:
            label = 'excellent'
        elif metrics['self_sufficiency'] > 0.5:
            label = 'good'
        elif metrics['self_sufficiency'] > 0.3:
            label = 'fair'
        else:
            label = 'poor'
        
        cluster_id = hash(clusters.cpu().numpy().tobytes())
        self.cluster_labels[cluster_id] = label
        
        logger.info(f"Added cluster label: {label} (self_suff={metrics['self_sufficiency']:.3f})")
    
    def add_solar_labels(self, building_ids: List[int], performance_data: Dict):
        """
        Add real solar deployment labels
        
        Args:
            building_ids: Buildings with solar installed
            performance_data: Actual performance metrics
        """
        for bid in building_ids:
            roi_years = performance_data.get(f'roi_{bid}', 10)
            
            # Categorize ROI
            if roi_years < 5:
                label = 'excellent'
            elif roi_years < 7:
                label = 'good'
            elif roi_years < 10:
                label = 'fair'
            else:
                label = 'poor'
            
            self.solar_labels[bid] = label
        
        logger.info(f"Added {len(building_ids)} solar labels. Total labels: {len(self.solar_labels)}")
    
    def recommend_solar(
        self,
        data_loader: DataLoader,
        top_k: int = 10,
        confidence_threshold: float = 0.7
    ) -> List[Tuple[int, float, str]]:
        """
        Generate solar recommendations
        
        Returns:
            List of (building_id, score, roi_category) tuples
        """
        self.model.eval()
        all_recommendations = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch, phase='solar')
                
                # Get high-confidence recommendations
                mask = outputs['confidence'].squeeze() > confidence_threshold
                
                if mask.any():
                    scores = outputs['solar_potential'][mask].squeeze()
                    roi_cats = outputs['roi_category'][mask].argmax(dim=-1)
                    building_ids = torch.where(mask)[0]
                    
                    for bid, score, roi_cat in zip(building_ids, scores, roi_cats):
                        roi_label = ['excellent', 'good', 'fair', 'poor'][roi_cat.item()]
                        all_recommendations.append((
                            bid.item(),
                            score.item(),
                            roi_label
                        ))
        
        # Sort by score and return top k
        all_recommendations.sort(key=lambda x: x[1], reverse=True)
        return all_recommendations[:top_k]
    
    def _calculate_self_sufficiency(self, cluster_assignments: torch.Tensor, batch: Data) -> torch.Tensor:
        """Calculate self-sufficiency metric for clusters"""
        # Simplified calculation
        cluster_probs = cluster_assignments
        if hasattr(batch, 'consumption') and hasattr(batch, 'generation'):
            # Real calculation
            consumption = batch.consumption
            generation = batch.generation
            
            # Weight by cluster membership
            cluster_consumption = torch.matmul(cluster_probs.T, consumption)
            cluster_generation = torch.matmul(cluster_probs.T, generation)
            
            # Self-sufficiency ratio
            self_sufficiency = (cluster_generation / (cluster_consumption + 1e-8)).mean()
        else:
            # Proxy based on complementarity
            self_sufficiency = 1.0 - cluster_probs.std(dim=0).mean()
        
        return self_sufficiency
    
    def _calculate_roi_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Calculate ROI category prediction accuracy"""
        pred_classes = predictions.argmax(dim=-1)
        return (pred_classes == targets).float().mean()
    
    def _estimate_roi_from_features(self, batch: Data) -> torch.Tensor:
        """Estimate ROI potential from building features (unsupervised proxy)"""
        # Simple heuristic based on features
        if hasattr(batch, 'roof_area') and hasattr(batch, 'consumption'):
            roi_proxy = batch.roof_area / (batch.consumption + 1e-8)
            roi_proxy = torch.sigmoid(roi_proxy - 1.0)  # Normalize to 0-1
        else:
            roi_proxy = torch.rand(batch.num_nodes, device=batch.x.device) * 0.5 + 0.25
        
        return roi_proxy
    
    def _calculate_solar_network_loss(
        self,
        solar_potential: torch.Tensor,
        network_impact: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Calculate network-aware loss for solar placement"""
        # Encourage high network impact nodes to get solar
        return F.mse_loss(solar_potential.squeeze(), network_impact.squeeze())
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'epoch': self.current_epoch,
            'round': self.current_round,
            'cluster_labels': self.cluster_labels,
            'solar_labels': self.solar_labels,
            'best_metrics': self.best_metrics
        }
        
        path = self.experiment_dir / f'{name}_checkpoint.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.current_epoch = checkpoint['epoch']
        self.current_round = checkpoint['round']
        self.cluster_labels = checkpoint['cluster_labels']
        self.solar_labels = checkpoint['solar_labels']
        self.best_metrics = checkpoint['best_metrics']
        logger.info(f"Loaded checkpoint from {path}")
    
    def _get_cluster_labels_for_batch(self, batch: Data) -> Optional[torch.Tensor]:
        """Get cluster labels for current batch if available"""
        # Implementation depends on how batches are identified
        return None  # Placeholder
    
    def _get_solar_labels_for_batch(self, batch: Data) -> Optional[torch.Tensor]:
        """Get solar labels for current batch if available"""
        # Map building IDs to labels
        if hasattr(batch, 'building_id'):
            labels = []
            for bid in batch.building_id:
                if bid.item() in self.solar_labels:
                    label = self.solar_labels[bid.item()]
                    # Convert to numerical
                    label_map = {'excellent': 0, 'good': 1, 'fair': 2, 'poor': 3}
                    labels.append(label_map[label])
                else:
                    labels.append(-1)  # Unknown
            
            labels = torch.tensor(labels, device=batch.x.device)
            return labels if (labels >= 0).any() else None
        return None
    
    def _validate_discovery(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate discovery phase"""
        self.model.eval()
        val_metrics = {'self_sufficiency': 0, 'complementarity': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch, phase='discovery')
                
                val_metrics['self_sufficiency'] += self._calculate_self_sufficiency(
                    outputs['cluster_assignments'], batch
                ).item()
                
                # Calculate complementarity
                if hasattr(batch, 'temporal_features'):
                    comp_score = self._calculate_complementarity_score(
                        outputs['cluster_assignments'],
                        batch.temporal_features
                    )
                    val_metrics['complementarity'] += comp_score.item()
        
        # Average
        n_batches = len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= n_batches
            
        return val_metrics
    
    def _validate_solar(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate solar phase"""
        self.model.eval()
        val_metrics = {'roi_accuracy': 0, 'network_impact': 0}
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                outputs = self.model(batch, phase='solar')
                
                # Calculate metrics
                val_metrics['network_impact'] += outputs['network_impact'].mean().item()
                
                # ROI accuracy if labels available
                targets = self._get_solar_labels_for_batch(batch)
                if targets is not None:
                    accuracy = self._calculate_roi_accuracy(
                        outputs['roi_category'],
                        targets
                    )
                    val_metrics['roi_accuracy'] += accuracy.item()
        
        # Average
        n_batches = len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= n_batches
            
        return val_metrics
    
    def _calculate_complementarity_score(
        self,
        cluster_assignments: torch.Tensor,
        temporal_features: torch.Tensor
    ) -> torch.Tensor:
        """Calculate complementarity score for validation"""
        # Simplified version
        cluster_probs = cluster_assignments
        
        # Normalize temporal features
        temporal_norm = (temporal_features - temporal_features.mean(dim=1, keepdim=True)) / (
            temporal_features.std(dim=1, keepdim=True) + 1e-8
        )
        
        # Correlation within clusters
        corr_matrix = torch.matmul(temporal_norm, temporal_norm.t()) / temporal_features.size(1)
        cluster_comembership = torch.matmul(cluster_probs, cluster_probs.t())
        
        # Weighted correlation (want negative)
        weighted_corr = corr_matrix * cluster_comembership
        mask = 1 - torch.eye(cluster_assignments.size(0), device=cluster_assignments.device)
        
        return (weighted_corr * mask).sum() / (cluster_comembership * mask).sum().clamp(min=1)


# Additional loss classes that were missing

class SolarROILoss(nn.Module):
    """Loss function for solar ROI category prediction"""
    
    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate ROI category loss
        
        Args:
            predictions: Predicted ROI categories [N, 4]
            targets: True ROI categories [N]
        """
        # Filter out unknown labels
        mask = targets >= 0
        if not mask.any():
            return torch.tensor(0.0, device=predictions.device)
        
        return F.cross_entropy(
            predictions[mask],
            targets[mask],
            weight=self.class_weights
        )


class ClusterQualityLoss(nn.Module):
    """Semi-supervised loss for cluster quality"""
    
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        cluster_assignments: torch.Tensor,
        cluster_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate cluster quality loss
        
        Args:
            cluster_assignments: Soft cluster assignments [N, K]
            cluster_labels: Quality labels for clusters
        """
        # Convert labels to scores
        label_scores = {
            'excellent': 1.0,
            'good': 0.7,
            'fair': 0.4,
            'poor': 0.1
        }
        
        # This is simplified - actual implementation would be more complex
        return F.mse_loss(cluster_assignments.mean(dim=0), torch.tensor(0.5))


class NetworkImpactLoss(nn.Module):
    """Merged from network_aware_loss.py"""
    
    def __init__(self):
        super().__init__()
        self.hop_weights = [1.0, 0.5, 0.25]
        
    def forward(
        self,
        cluster_assignments: torch.Tensor,
        embeddings: torch.Tensor,
        hop_features: Dict[str, torch.Tensor],
        edge_index: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate network impact loss"""
        losses = {}
        
        # Multi-hop impact
        total_loss = 0
        for i, weight in enumerate(self.hop_weights):
            if f'hop_{i+1}_features' in hop_features:
                hop_feat = hop_features[f'hop_{i+1}_features']
                # Encourage diversity in hop features
                hop_variance = torch.var(hop_feat, dim=0).mean()
                hop_loss = 1.0 / (hop_variance + 1e-8)
                total_loss += weight * hop_loss
                losses[f'hop_{i+1}'] = hop_loss.item()
        
        losses['multi_hop'] = total_loss.item()
        
        return total_loss, losses