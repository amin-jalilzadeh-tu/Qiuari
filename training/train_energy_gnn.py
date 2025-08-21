# training/train_energy_gnn.py
"""
Training framework for Energy Planning GNN
Self-supervised learning with physics-informed objectives
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import wandb  # Optional: for experiment tracking
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import all model components
from models.base_gnn import create_energy_gnn_base
from models.attention_layers import EnergyComplementarityAttention
from models.temporal_layers import TemporalProcessor
from models.physics_layers import PhysicsConstraintLayer
from models.task_heads import create_energy_task_heads

logger = logging.getLogger(__name__)


class EnergyGNNLoss(nn.Module):
    """Combined loss function for energy planning GNN"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Loss weights (learnable for automatic balancing)
        self.weights = nn.ParameterDict({
            'self_sufficiency': nn.Parameter(torch.tensor(5.0)),
            'peak_reduction': nn.Parameter(torch.tensor(3.0)),
            'complementarity': nn.Parameter(torch.tensor(4.0)),
            'physics_penalty': nn.Parameter(torch.tensor(10.0)),
            'sharing_fairness': nn.Parameter(torch.tensor(2.0)),
            'cluster_size': nn.Parameter(torch.tensor(1.0)),
            'distance_penalty': nn.Parameter(torch.tensor(1.5))
        })
        
        self.config = config
        
    def forward(self, outputs: Dict, data: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Calculate total loss and individual components
        
        Args:
            outputs: Model outputs from all layers
            data: Input data including consumption, generation, etc.
            
        Returns:
            total_loss: Combined loss for backprop
            loss_dict: Individual loss components for monitoring
        """
        losses = {}
        
        # 1. Self-Sufficiency Loss (maximize local energy use)
        losses['self_sufficiency'] = self._self_sufficiency_loss(
            outputs['sharing']['sharing_matrix'],
            data['generation'],
            data['consumption']
        )
        
        # 2. Peak Reduction Loss (minimize peak demand)
        losses['peak_reduction'] = self._peak_reduction_loss(
            outputs['sharing']['sharing_matrix'],
            outputs['sharing']['efficiency_matrix'],
            data['consumption']
        )
        
        # 3. Complementarity Loss (cluster opposite patterns)
        losses['complementarity'] = self._complementarity_loss(
            outputs['clustering']['cluster_assignments'],
            outputs['complementarity_matrix']
        )
        
        # 4. Physics Penalty (from physics layer)
        losses['physics_penalty'] = outputs.get('physics_penalty', torch.tensor(0.0))
        
        # 5. Sharing Fairness Loss (avoid one building supplying all)
        losses['sharing_fairness'] = self._sharing_fairness_loss(
            outputs['sharing']['energy_sent'],
            outputs['sharing']['energy_received']
        )
        
        # 6. Cluster Size Loss (enforce size constraints)
        losses['cluster_size'] = self._cluster_size_loss(
            outputs['clustering']['cluster_assignments'],
            self.config['min_cluster_size'],
            self.config['max_cluster_size']
        )
        
        # 7. Distance Penalty (minimize long-distance sharing)
        losses['distance_penalty'] = self._distance_penalty_loss(
            outputs['sharing']['sharing_matrix'],
            data['positions']
        )
        
        # Weighted combination
        total_loss = torch.tensor(0.0, device=data['positions'].device)
        for key, loss in losses.items():
            if key in self.weights:
                weighted_loss = self.weights[key] * loss
                total_loss = total_loss + weighted_loss
                losses[f'{key}_weighted'] = weighted_loss.item()
        
        losses['total'] = total_loss.item()
        
        return total_loss, losses
    
    def _self_sufficiency_loss(self, sharing_matrix, generation, consumption):
        """Maximize local energy use within clusters"""
        # Calculate how much generated energy is used locally
        total_generation = generation.sum()
        total_consumption = consumption.sum()
        
        # Energy that could be used locally
        potential_local_use = torch.min(total_generation, total_consumption)
        
        # Energy actually shared (indicates local use when within cluster)
        total_shared = sharing_matrix.sum()
        
        # Loss: minimize difference between potential and actual local use
        # Lower sharing = higher local use = better self-sufficiency
        loss = 1.0 - (total_shared / (potential_local_use + 1e-6))
        
        return -loss  # Negative because we want to maximize
    
    def _peak_reduction_loss(self, sharing_matrix, efficiency_matrix, consumption):
        """Minimize peak demand through sharing"""
        # Calculate net consumption after sharing
        energy_received = (sharing_matrix * efficiency_matrix).sum(dim=1)
        net_consumption = consumption - energy_received
        
        # Peak before and after
        peak_before = consumption.max()
        peak_after = net_consumption.max()
        
        # We want to minimize peak_after
        loss = peak_after / (peak_before + 1e-6)
        
        return loss
    
    def _complementarity_loss(self, cluster_assignments, complementarity_matrix):
        """Ensure clusters have complementary buildings"""
        if cluster_assignments.dim() > 1:
            cluster_assignments = cluster_assignments[0]
        
        unique_clusters = torch.unique(cluster_assignments)
        total_complementarity = torch.tensor(0.0, device=cluster_assignments.device)
        
        for cluster_id in unique_clusters:
            mask = (cluster_assignments == cluster_id)
            if mask.sum() < 2:
                continue
            
            # Get complementarity within cluster
            cluster_comp = complementarity_matrix[mask][:, mask]
            
            # Average complementarity (negative is good)
            avg_comp = cluster_comp.mean()
            
            # We want negative (complementary) values
            total_complementarity = total_complementarity + avg_comp
        
        # Average across clusters
        loss = total_complementarity / len(unique_clusters)
        
        return loss  # Lower (more negative) is better
    
    def _sharing_fairness_loss(self, energy_sent, energy_received):
        """Ensure fair distribution of sharing burden"""
        # Variance in energy sent (we want low variance = fairness)
        sent_variance = energy_sent.var()
        
        # Variance in net position (sent - received)
        net_position = energy_sent - energy_received
        net_variance = net_position.var()
        
        loss = sent_variance + 0.5 * net_variance
        
        return loss
    
    def _cluster_size_loss(self, cluster_assignments, min_size, max_size):
        """Enforce cluster size constraints"""
        if cluster_assignments.dim() > 1:
            cluster_assignments = cluster_assignments[0]
        
        unique_clusters = torch.unique(cluster_assignments)
        size_penalty = torch.tensor(0.0, device=cluster_assignments.device)
        
        for cluster_id in unique_clusters:
            cluster_size = (cluster_assignments == cluster_id).sum().float()
            
            # Penalty for too small
            if cluster_size < min_size:
                size_penalty = size_penalty + (min_size - cluster_size) ** 2
            
            # Penalty for too large
            if cluster_size > max_size:
                size_penalty = size_penalty + (cluster_size - max_size) ** 2
        
        return size_penalty / len(unique_clusters)
    
    def _distance_penalty_loss(self, sharing_matrix, positions):
        """Minimize long-distance energy sharing"""
        if positions.dim() == 2:
            positions = positions.unsqueeze(0)
        
        # Calculate pairwise distances
        distances = torch.cdist(positions, positions)
        
        # Weight sharing by distance
        weighted_sharing = sharing_matrix * distances
        
        # Total distance-weighted sharing (minimize this)
        loss = weighted_sharing.sum() / (sharing_matrix.sum() + 1e-6)
        
        return loss


class EnergyGNNModel(nn.Module):
    """Complete Energy Planning GNN Model"""
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Initialize all layers
        self.base_gnn = create_energy_gnn_base(config)
        self.attention_layer = EnergyComplementarityAttention(config)
        self.temporal_processor = TemporalProcessor(config)
        self.physics_layer = PhysicsConstraintLayer(config)
        self.task_heads = create_energy_task_heads(config)
        
        self.config = config
        
    def forward(self, data: Dict, current_hour: int = 14) -> Dict:
        """Forward pass through all layers"""
        
        # 1. Base GNN
        base_output = self.base_gnn(
            data['node_features'],
            data['edge_indices']
        )
        
        # 2. Attention Layer
        attention_output = self.attention_layer(
            base_output,
            data['edge_indices'],
            return_attention=False
        )
        
        # 3. Temporal Processor
        temporal_output = self.temporal_processor(
            attention_output['embeddings'],
            temporal_data=data.get('temporal_data'),
            current_hour=current_hour,
            return_all_hours=False
        )
        
        # Ensure batch dimension
        if temporal_output['embeddings']['building'].dim() == 2:
            for key in temporal_output['embeddings']:
                temporal_output['embeddings'][key] = temporal_output['embeddings'][key].unsqueeze(0)
        
        # 4. Physics Layer
        # Create initial sharing proposal based on complementarity
        num_buildings = data['num_buildings']
        sharing_proposals = self._create_smart_sharing_proposals(
            temporal_output['embeddings']['building'],
            attention_output['complementarity_matrix'],
            data['generation'],
            data['consumption']
        )
        
        physics_metadata = {
            'lv_group_ids': data['lv_group_ids'],
            'valid_lv_mask': data['valid_lv_mask'],
            'positions': data['positions'],
            'temporal_states': temporal_output.get('temporal_encoding')
        }
        
        physics_output = self.physics_layer(
            temporal_output['embeddings'],
            sharing_proposals,
            data['consumption'],
            data['generation'],
            physics_metadata
        )
        
        # 5. Task Heads
        task_metadata = {
            'lv_group_ids': data['lv_group_ids'],
            'positions': data['positions'],
            'generation': data['generation'],
            'consumption': data['consumption'],
            'complementarity_matrix': attention_output['complementarity_matrix'],
            'building_types': data.get('building_types'),
            'building_features': data.get('building_features', {}),
            'current_assets': data.get('current_assets', {})
        }
        
        task_output = self.task_heads(
            physics_output['feasible_embeddings'],
            task_metadata,
            current_hour=current_hour
        )
        
        # Combine all outputs
        return {
            'base': base_output,
            'attention': attention_output,
            'temporal': temporal_output,
            'physics': physics_output,
            'tasks': task_output,
            # Key outputs for loss
            'clustering': task_output['clustering'],
            'sharing': task_output['sharing'],
            'metrics': task_output['metrics'],
            'complementarity_matrix': attention_output['complementarity_matrix'],
            'physics_penalty': physics_output['total_penalty']
        }
    
    def _create_smart_sharing_proposals(self, embeddings, complementarity, generation, consumption):
        """Create intelligent sharing proposals based on complementarity"""
        batch_size = embeddings.shape[0]
        num_buildings = embeddings.shape[1]
        
        # Start with complementarity as base
        if complementarity.dim() == 2:
            sharing = complementarity.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            sharing = complementarity
        
        # Make positive and symmetric
        sharing = torch.abs(sharing)
        sharing = (sharing + sharing.transpose(1, 2)) / 2
        
        # Scale by available generation/consumption
        net_position = generation - consumption
        surplus_mask = (net_position > 0).float()
        deficit_mask = (net_position < 0).float()
        
        # Only share from surplus to deficit
        for b in range(batch_size):
            sharing[b] = sharing[b] * surplus_mask[b].unsqueeze(1) * deficit_mask[b].unsqueeze(0)
        
        # Scale to reasonable values
        sharing = sharing * 10.0  # Adjust scale as needed
        
        return sharing


class EnergyGNNTrainer:
    """Trainer for Energy Planning GNN"""
    
    def __init__(self, model: EnergyGNNModel, config: Dict, data_loader=None):
        self.model = model
        self.config = config
        self.data_loader = data_loader
        
        # Loss function
        self.criterion = EnergyGNNLoss(config)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(model.parameters()) + list(self.criterion.parameters()),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        # History tracking
        self.train_history = {
            'loss': [],
            'metrics': {}
        }
        
        # Best model tracking
        self.best_loss = float('inf')
        self.best_model_state = None
        
    def train_epoch(self, data: Dict) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(data)
        
        # Calculate loss
        loss, loss_components = self.criterion(outputs, data)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        # Calculate metrics
        metrics = self._calculate_metrics(outputs, data)
        metrics.update(loss_components)
        
        return {
            'loss': loss.item(),
            'metrics': metrics
        }
    
    def validate(self, data: Dict) -> Dict:
        """Validation step"""
        self.model.eval()
        
        with torch.no_grad():
            outputs = self.model(data)
            loss, loss_components = self.criterion(outputs, data)
            metrics = self._calculate_metrics(outputs, data)
            metrics.update(loss_components)
        
        return {
            'loss': loss.item(),
            'metrics': metrics
        }
    
    def train(self, train_data: Dict, val_data: Optional[Dict] = None, epochs: int = 100):
        """Main training loop"""
        
        print("\n" + "="*60)
        print("TRAINING ENERGY PLANNING GNN")
        print("="*60 + "\n")
        
        for epoch in range(epochs):
            # Training
            train_result = self.train_epoch(train_data)
            self.train_history['loss'].append(train_result['loss'])
            
            # Validation
            if val_data is not None:
                val_result = self.validate(val_data)
                
                # Learning rate scheduling
                self.scheduler.step(val_result['loss'])
                
                # Save best model
                if val_result['loss'] < self.best_loss:
                    self.best_loss = val_result['loss']
                    self.best_model_state = self.model.state_dict()
            else:
                val_result = {'loss': 0, 'metrics': {}}
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{epochs}")
                print(f"  Train Loss: {train_result['loss']:.4f}")
                if val_data is not None:
                    print(f"  Val Loss:   {val_result['loss']:.4f}")
                
                # Print key metrics
                print("  Metrics:")
                for key in ['self_sufficiency', 'peak_reduction', 'complementarity']:
                    if key in train_result['metrics']:
                        print(f"    {key}: {train_result['metrics'][key]:.4f}")
                
                # Print performance
                if 'tasks' in train_data:
                    summary = train_result['metrics'].get('summary', {})
                    print(f"  Performance:")
                    print(f"    Avg SSR: {summary.get('avg_self_sufficiency', 0):.1%}")
                    print(f"    Avg Peak Reduction: {summary.get('avg_peak_reduction', 0):.1%}")
                print("-" * 40)
        
        print("\n✅ Training Complete!")
        
        # Load best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print(f"Loaded best model (val loss: {self.best_loss:.4f})")
        
        return self.train_history
    
    def _calculate_metrics(self, outputs: Dict, data: Dict) -> Dict:
        """Calculate performance metrics"""
        metrics = {}
        
        # From task outputs
        if 'tasks' in outputs and 'summary' in outputs['tasks']:
            summary = outputs['tasks']['summary']
            metrics['avg_ssr'] = summary.get('avg_self_sufficiency', 0)
            metrics['avg_peak_reduction'] = summary.get('avg_peak_reduction', 0)
            metrics['total_carbon_saved'] = summary.get('total_carbon_saved_kg', 0)
            metrics['total_shared'] = summary.get('total_energy_shared_kw', 0)
        
        return metrics
    
    def plot_training_history(self):
        """Plot training curves"""
        plt.figure(figsize=(12, 4))
        
        # Loss curve
        plt.subplot(1, 3, 1)
        plt.plot(self.train_history['loss'])
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        # SSR curve
        if 'avg_ssr' in self.train_history['metrics']:
            plt.subplot(1, 3, 2)
            plt.plot(self.train_history['metrics']['avg_ssr'])
            plt.title('Average Self-Sufficiency')
            plt.xlabel('Epoch')
            plt.ylabel('SSR')
            plt.grid(True)
        
        # Peak reduction curve
        if 'avg_peak_reduction' in self.train_history['metrics']:
            plt.subplot(1, 3, 3)
            plt.plot(self.train_history['metrics']['avg_peak_reduction'])
            plt.title('Average Peak Reduction')
            plt.xlabel('Epoch')
            plt.ylabel('Peak Reduction')
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()


# ========================================
# MAIN TRAINING SCRIPT
# ========================================

def train_energy_gnn(neo4j_data: Dict, config: Dict, epochs: int = 100):
    """Main function to train the Energy GNN"""
    
    # Create model
    model = EnergyGNNModel(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Create trainer
    trainer = EnergyGNNTrainer(model, config)
    
    # Prepare data (add temporal data)
    neo4j_data['temporal_data'] = {
        'consumption_history': torch.randn(1, neo4j_data['num_buildings'], 24, 8),
        'season': torch.tensor(0),
        'is_weekend': torch.tensor(False)
    }
    
    # Train model
    history = trainer.train(
        train_data=neo4j_data,
        val_data=None,  # Could split data for validation
        epochs=epochs
    )
    
    # Plot results
    trainer.plot_training_history()
    
    # Save model
    torch.save({
        'model_state': model.state_dict(),
        'config': config,
        'history': history
    }, 'trained_energy_gnn.pth')
    
    print("\n✅ Model saved to 'trained_energy_gnn.pth'")
    
    return model, trainer


# ========================================
# RUN TRAINING
# ========================================

if __name__ == "__main__":
    # Your configuration
    config = {
        'num_building_features': 17,
        'num_cable_features': 8,
        'num_transformer_features': 5,
        'num_cluster_features': 5,
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.1,
        'attention_heads': 8,
        'min_cluster_size': 3,
        'max_cluster_size': 15,
        'max_recommendations': 10,
        'carbon_intensity': 0.4,
        'temporal_dim': 24,
        # Training specific
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        # Physics
        'enforce_hard_boundaries': True,
        'check_balance': True,
        'apply_losses': True,
        'validate_temporal': False
    }
    
    # Load your Neo4j data
    print("Loading data from Neo4j...")
    # [Use your data loading code here]
    
    # Train model
    model, trainer = train_energy_gnn(neo4j_data, config, epochs=100)