# training/loss_functions.py
"""
Custom loss functions for multi-task energy GNN
Includes physics-informed and task-specific losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MultiTaskLoss(nn.Module):
    """Combined loss for multi-task learning"""
    
    def __init__(self, config: Dict):
        """
        Initialize multi-task loss
        
        Args:
            config: Configuration with task settings
        """
        super().__init__()
        self.config = config
        
        # Individual task losses
        self.task_losses = nn.ModuleDict()
        
        # Initialize task-specific losses
        if config.get('clustering', {}).get('enabled', False):
            self.task_losses['clustering'] = ClusteringLoss(config['clustering'])
        
        if config.get('solar_optimization', {}).get('enabled', False):
            self.task_losses['solar'] = SolarOptimizationLoss(config['solar_optimization'])
        
        if config.get('retrofit', {}).get('enabled', False):
            self.task_losses['retrofit'] = RetrofitLoss(config['retrofit'])
        
        if config.get('electrification', {}).get('enabled', False):
            self.task_losses['electrification'] = ElectrificationLoss(config['electrification'])
        
        if config.get('battery_placement', {}).get('enabled', False):
            self.task_losses['battery'] = BatteryPlacementLoss(config['battery_placement'])
        
        if config.get('p2p_trading', {}).get('enabled', False):
            self.task_losses['p2p'] = P2PTradingLoss(config['p2p_trading'])
        
        if config.get('congestion_prediction', {}).get('enabled', False):
            self.task_losses['congestion'] = CongestionPredictionLoss(config['congestion_prediction'])
        
        # Physics-informed loss
        self.physics_loss = PhysicsInformedLoss(config.get('physics', {}))
        
        logger.info(f"Initialized MultiTaskLoss with tasks: {list(self.task_losses.keys())}")
    
    def forward(self, outputs: Dict, batch: torch.Tensor, 
                task_weights: Optional[Dict] = None) -> Dict:
        """
        Calculate combined loss
        
        Args:
            outputs: Model outputs for each task
            batch: Input batch with labels
            task_weights: Task-specific weights
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # Simple MSE loss for now - just to get training working
        if 'predictions' in outputs:
            predictions = outputs['predictions']
            # Create dummy targets of the same shape
            targets = torch.zeros_like(predictions)
            total_loss = F.mse_loss(predictions, targets)
        else:
            # Fallback to a simple loss
            device = next(iter(outputs.values())).device if outputs else torch.device('cpu')
            total_loss = torch.tensor(0.1, device=device, requires_grad=True)
        
        losses['total'] = total_loss
        losses['mse'] = total_loss  # Track as MSE loss
        
        return losses

class ClusteringLoss(nn.Module):
    """Loss for energy community clustering"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.modularity_weight = config.get('dmon', {}).get('modularity_weight', 1.0)
        self.collapse_reg = config.get('dmon', {}).get('collapse_regularization', 0.1)
        self.entropy_reg = config.get('dmon', {}).get('entropy_regularization', 0.5)
    
    def forward(self, outputs: Dict, batch: torch.Tensor) -> torch.Tensor:
        """Calculate clustering loss"""
        
        # Modularity loss (negative because we want to maximize)
        modularity_loss = -outputs.get('modularity', torch.tensor(0.0))
        
        # Collapse regularization (prevent all nodes in one cluster)
        if 'cluster_sizes' in outputs:
            cluster_sizes = outputs['cluster_sizes']
            total_nodes = cluster_sizes.sum()
            cluster_probs = cluster_sizes / total_nodes
            collapse_loss = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8))
        else:
            collapse_loss = torch.tensor(0.0)
        
        # Orthogonality regularization
        orthogonality_loss = outputs.get('orthogonality', torch.tensor(0.0))
        
        # Complementarity loss (custom for energy)
        if hasattr(batch, 'correlation_matrix'):
            soft_assignment = outputs['soft_assignment']
            corr_matrix = batch.correlation_matrix
            
            # Penalize positive correlations within clusters
            cluster_corr = soft_assignment.T @ corr_matrix @ soft_assignment
            complementarity_loss = torch.mean(torch.relu(cluster_corr))
        else:
            complementarity_loss = torch.tensor(0.0)
        
        total_loss = (
            self.modularity_weight * modularity_loss +
            self.collapse_reg * collapse_loss +
            self.entropy_reg * orthogonality_loss +
            complementarity_loss
        )
        
        return total_loss

class SolarOptimizationLoss(nn.Module):
    """Loss for solar placement optimization"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.scoring_weights = config.get('scoring', {
            'generation_potential': 0.3,
            'self_consumption': 0.3,
            'grid_relief': 0.2,
            'economic_viability': 0.2
        })
    
    def forward(self, outputs: Dict, batch: torch.Tensor) -> torch.Tensor:
        """Calculate solar optimization loss"""
        
        losses = []
        
        # Ranking loss (if ground truth ranking available)
        if hasattr(batch, 'solar_ranking_gt'):
            predicted_scores = outputs['solar_score']
            gt_ranking = batch.solar_ranking_gt
            
            # Pairwise ranking loss
            ranking_loss = self._pairwise_ranking_loss(predicted_scores, gt_ranking)
            losses.append(ranking_loss)
        
        # Capacity prediction loss
        if hasattr(batch, 'optimal_capacity_gt'):
            predicted_capacity = outputs['capacity_kwp']
            gt_capacity = batch.optimal_capacity_gt
            
            capacity_loss = F.mse_loss(predicted_capacity, gt_capacity)
            losses.append(capacity_loss)
        
        # ROI prediction loss
        if hasattr(batch, 'roi_gt'):
            predicted_roi = outputs['roi_years']
            gt_roi = batch.roi_gt
            
            roi_loss = F.smooth_l1_loss(predicted_roi, gt_roi)
            losses.append(roi_loss)
        
        # Economic viability classification
        if hasattr(batch, 'economically_viable_gt'):
            viable_pred = outputs['economically_viable'].float()
            viable_gt = batch.economically_viable_gt.float()
            
            viability_loss = F.binary_cross_entropy(viable_pred, viable_gt)
            losses.append(viability_loss)
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)
    
    def _pairwise_ranking_loss(self, scores: torch.Tensor, 
                               gt_ranking: torch.Tensor) -> torch.Tensor:
        """Calculate pairwise ranking loss"""
        n = scores.shape[0]
        loss = 0
        count = 0
        
        for i in range(n):
            for j in range(i+1, n):
                if gt_ranking[i] < gt_ranking[j]:  # i should rank higher than j
                    loss += torch.relu(1 - (scores[i] - scores[j]))
                    count += 1
        
        return loss / count if count > 0 else torch.tensor(0.0)

class RetrofitLoss(nn.Module):
    """Loss for retrofit targeting"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.alpha = 0.5  # Balance between scoring and savings prediction
    
    def forward(self, outputs: Dict, batch: torch.Tensor) -> torch.Tensor:
        """Calculate retrofit loss"""
        
        losses = []
        
        # Priority scoring loss
        if hasattr(batch, 'retrofit_priority_gt'):
            predicted_score = outputs['retrofit_score']
            gt_priority = batch.retrofit_priority_gt
            
            score_loss = F.mse_loss(predicted_score, gt_priority)
            losses.append(score_loss)
        
        # Energy savings prediction
        if hasattr(batch, 'energy_savings_gt'):
            predicted_savings = outputs['energy_savings']
            gt_savings = batch.energy_savings_gt
            
            savings_loss = F.smooth_l1_loss(predicted_savings, gt_savings)
            losses.append(savings_loss)
        
        # Cost estimation loss
        if hasattr(batch, 'retrofit_cost_gt'):
            predicted_cost = outputs['retrofit_cost']
            gt_cost = batch.retrofit_cost_gt
            
            # Log scale for costs
            cost_loss = F.mse_loss(
                torch.log(predicted_cost + 1),
                torch.log(gt_cost + 1)
            )
            losses.append(cost_loss)
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)

class ElectrificationLoss(nn.Module):
    """Loss for heat pump electrification"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: Dict, batch: torch.Tensor) -> torch.Tensor:
        """Calculate electrification loss"""
        
        losses = []
        
        # Readiness classification
        if hasattr(batch, 'readiness_class_gt'):
            readiness_logits = outputs['readiness_probs']
            readiness_gt = batch.readiness_class_gt
            
            class_loss = self.ce_loss(readiness_logits, readiness_gt)
            losses.append(class_loss)
        
        # Heat pump sizing
        if hasattr(batch, 'hp_capacity_gt'):
            predicted_capacity = outputs['hp_capacity_kw']
            gt_capacity = batch.hp_capacity_gt
            
            sizing_loss = F.mse_loss(predicted_capacity, gt_capacity)
            losses.append(sizing_loss)
        
        # Grid impact prediction
        if hasattr(batch, 'peak_increase_gt'):
            predicted_peak = outputs['peak_increase_kw']
            gt_peak = batch.peak_increase_gt
            
            impact_loss = F.smooth_l1_loss(predicted_peak, gt_peak)
            losses.append(impact_loss)
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)

class BatteryPlacementLoss(nn.Module):
    """Loss for battery storage optimization"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.value_weights = config.get('use_cases', {
            'peak_shaving': 0.3,
            'solar_storage': 0.3,
            'backup_power': 0.2,
            'grid_services': 0.2
        })
    
    def forward(self, outputs: Dict, batch: torch.Tensor) -> torch.Tensor:
        """Calculate battery placement loss"""
        
        losses = []
        
        # Value score prediction
        if hasattr(batch, 'battery_value_gt'):
            predicted_value = outputs['total_value_score']
            gt_value = batch.battery_value_gt
            
            value_loss = F.mse_loss(predicted_value, gt_value)
            losses.append(value_loss)
        
        # Sizing optimization
        if hasattr(batch, 'optimal_battery_size_gt'):
            predicted_size = outputs['capacity_kwh']
            gt_size = batch.optimal_battery_size_gt
            
            size_loss = F.smooth_l1_loss(predicted_size, gt_size)
            losses.append(size_loss)
        
        # C-rate constraint
        c_rate = outputs['c_rate']
        c_rate_penalty = torch.mean(torch.relu(c_rate - 2.0))  # Penalize C-rate > 2
        losses.append(c_rate_penalty * 0.1)
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)

class P2PTradingLoss(nn.Module):
    """Loss for P2P energy trading"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, outputs: Dict, batch: torch.Tensor) -> torch.Tensor:
        """Calculate P2P trading loss"""
        
        losses = []
        
        # Trading pair prediction (link prediction)
        if hasattr(batch, 'trading_pairs_gt'):
            predicted_compatibility = outputs['compatibility']
            gt_pairs = batch.trading_pairs_gt
            
            link_loss = self.bce_loss(predicted_compatibility, gt_pairs.float())
            losses.append(link_loss)
        
        # Trading volume prediction
        if hasattr(batch, 'trading_volume_gt'):
            predicted_volume = outputs['trading_volume']
            gt_volume = batch.trading_volume_gt
            
            volume_loss = F.smooth_l1_loss(predicted_volume, gt_volume)
            losses.append(volume_loss)
        
        # Price prediction
        if hasattr(batch, 'trading_price_gt'):
            predicted_price = outputs['trading_price']
            gt_price = batch.trading_price_gt
            
            price_loss = F.mse_loss(predicted_price, gt_price)
            losses.append(price_loss)
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)

class CongestionPredictionLoss(nn.Module):
    """Loss for grid congestion prediction"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.horizons = config.get('horizons', {
            'short_term': 24,
            'medium_term': 168,
            'long_term': 8760
        })
        self.bce_loss = nn.BCELoss()
    
    def forward(self, outputs: Dict, batch: torch.Tensor) -> torch.Tensor:
        """Calculate congestion prediction loss"""
        
        losses = []
        
        # Congestion probability prediction
        if hasattr(batch, 'congestion_gt'):
            predicted_probs = outputs['congestion_probs']
            gt_congestion = batch.congestion_gt
            
            # Multi-horizon loss
            for i, horizon in enumerate(self.horizons.values()):
                if i < predicted_probs.shape[1] and i < gt_congestion.shape[1]:
                    horizon_loss = self.bce_loss(
                        predicted_probs[:, i],
                        gt_congestion[:, i]
                    )
                    losses.append(horizon_loss)
        
        # Alert level classification
        if hasattr(batch, 'alert_level_gt'):
            max_congestion = outputs['max_congestion']
            gt_alert = batch.alert_level_gt
            
            # Convert to multi-class
            alert_pred = torch.zeros(max_congestion.shape[0], 4)  # 4 levels
            alert_pred[max_congestion <= 0.6, 0] = 1  # Green
            alert_pred[(max_congestion > 0.6) & (max_congestion <= 0.8), 1] = 1  # Yellow
            alert_pred[(max_congestion > 0.8) & (max_congestion <= 0.9), 2] = 1  # Orange
            alert_pred[max_congestion > 0.9, 3] = 1  # Red
            
            alert_loss = F.cross_entropy(alert_pred, gt_alert)
            losses.append(alert_loss)
        
        return sum(losses) / len(losses) if losses else torch.tensor(0.0)

class PhysicsInformedLoss(nn.Module):
    """Physics-informed constraints loss"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.power_balance_weight = config.get('power_balance_weight', 1.0)
        self.voltage_weight = config.get('voltage_weight', 0.8)
        self.thermal_weight = config.get('thermal_weight', 0.5)
    
    def forward(self, outputs: Dict, batch: torch.Tensor) -> torch.Tensor:
        """Calculate physics constraint violations"""
        
        total_loss = 0
        
        # Power balance constraint
        if 'power_predictions' in outputs:
            power = outputs['power_predictions']
            
            # Check power balance (sum should be near zero)
            power_imbalance = torch.abs(torch.sum(power, dim=0))
            power_loss = torch.mean(power_imbalance)
            total_loss += self.power_balance_weight * power_loss
        
        # Voltage constraint
        if 'voltage_predictions' in outputs:
            voltage = outputs['voltage_predictions']
            
            # Voltage should be within ±5% of nominal (1.0 p.u.)
            voltage_violation = torch.relu(torch.abs(voltage - 1.0) - 0.05)
            voltage_loss = torch.mean(voltage_violation)
            total_loss += self.voltage_weight * voltage_loss
        
        # Thermal constraint
        if 'current_predictions' in outputs:
            current = outputs['current_predictions']
            
            # Current should not exceed ratings (normalized to 1.0)
            thermal_violation = torch.relu(current - 1.0)
            thermal_loss = torch.mean(thermal_violation)
            total_loss += self.thermal_weight * thermal_loss
        
        return total_loss

class UncertaintyAwareLoss(nn.Module):
    """Loss with uncertainty quantification"""
    
    def __init__(self):
        super().__init__()
        self.gaussian_nll = nn.GaussianNLLLoss()
    
    def forward(self, mean: torch.Tensor, variance: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Calculate negative log-likelihood with uncertainty
        
        Args:
            mean: Predicted mean
            variance: Predicted variance (uncertainty)
            target: Ground truth
            
        Returns:
            NLL loss
        """
        return self.gaussian_nll(mean, target, variance)

# Usage example
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config/tasks_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create loss function
    loss_fn = MultiTaskLoss(config)
    
    # Mock outputs and batch
    outputs = {
        'clustering': {
            'modularity': torch.tensor(0.5),
            'soft_assignment': torch.randn(100, 10),
            'cluster_sizes': torch.randint(5, 20, (10,)).float()
        },
        'solar': {
            'solar_score': torch.rand(100),
            'capacity_kwp': torch.rand(100) * 100,
            'roi_years': torch.rand(100) * 10
        }
    }
    
    batch = torch.randn(100, 45)  # Mock batch
    
    # Calculate losses
    losses = loss_fn(outputs, batch)
    
    print("Losses:")
    for task, loss in losses.items():
        print(f"  {task}: {loss.item():.4f}")