"""
Uncertainty quantification module for Energy GNN.
Implements MC Dropout, ensemble methods, and confidence estimation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)


class MCDropout(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    Keeps dropout active during inference for uncertainty quantification.
    """
    
    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p
        
    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Apply dropout that stays active during inference.
        
        Args:
            x: Input tensor
            training: Ignored - dropout always active for MC sampling
            
        Returns:
            Tensor with dropout applied
        """
        return F.dropout(x, p=self.p, training=True)


class BayesianGNNLayer(nn.Module):
    """
    Bayesian GNN layer with learnable uncertainty.
    Estimates both aleatoric and epistemic uncertainty.
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # Mean and variance networks
        self.mean_layer = nn.Linear(in_features, out_features)
        self.log_var_layer = nn.Linear(in_features, out_features)
        
        # MC Dropout for epistemic uncertainty
        self.dropout = MCDropout(dropout_rate)
        
        # Learned temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Args:
            x: Input features
            sample: Whether to sample from distribution
            
        Returns:
            Mean and log variance of predictions
        """
        # Apply dropout for epistemic uncertainty
        x_dropped = self.dropout(x)
        
        # Compute mean and log variance
        mean = self.mean_layer(x_dropped)
        log_var = self.log_var_layer(x_dropped)
        
        # Clamp log variance for numerical stability
        log_var = torch.clamp(log_var, min=-20, max=2)
        
        if sample:
            # Sample from distribution
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + eps * std, log_var
        
        return mean, log_var


class UncertaintyQuantifier(nn.Module):
    """
    Comprehensive uncertainty quantification for GNN predictions.
    Combines multiple uncertainty estimation methods.
    """
    
    def __init__(self, base_model: nn.Module, config: Dict[str, Any] = None,
                 num_classes: int = None, mc_samples: int = 20, 
                 temperature: float = 1.0):
        super().__init__()
        self.base_model = base_model
        self.config = config or {}
        
        # Extract dimensions from config or use defaults
        self.num_classes = num_classes if num_classes is not None else (
            config.get('num_clusters', config.get('num_classes', 10)) if config else 10
        )
        self.mc_samples = mc_samples
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Store device
        self.device = next(base_model.parameters()).device
        
        # Add MC Dropout to existing model
        self._add_mc_dropout(base_model)
        
        # Uncertainty estimation heads
        hidden_dim = config.get('hidden_dim', 128) if config else 128
        
        # Aleatoric uncertainty predictor
        self.aleatoric_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            MCDropout(0.1),
            nn.Linear(hidden_dim // 2, self.num_classes)
        )
        
        # Epistemic uncertainty aggregator
        self.epistemic_aggregator = nn.Sequential(
            nn.Linear(self.num_classes * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus()  # Ensure positive uncertainty
        )
        
        # Confidence calibration network - fixed input size
        # Using just max_prob and entropy for stability
        calibration_input_dim = 2  # Only max_prob and entropy
        self.calibration_net = nn.Sequential(
            nn.Linear(calibration_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def _add_mc_dropout(self, model: nn.Module):
        """Replace regular dropout with MC Dropout in the model."""
        for name, module in model.named_children():
            if isinstance(module, nn.Dropout):
                setattr(model, name, MCDropout(module.p))
            else:
                self._add_mc_dropout(module)
    
    def forward(self, data: Data, return_all_samples: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with uncertainty quantification.
        
        Args:
            data: Input graph data
            return_all_samples: Whether to return all MC samples
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        # Enable dropout for MC sampling
        self.base_model.train()
        
        # Collect MC samples
        mc_predictions = []
        mc_embeddings = []
        
        with torch.no_grad():
            for i in range(self.mc_samples):
                outputs = self.base_model(data)
                
                # Extract predictions based on model output structure
                if 'clustering_cluster_assignments' in outputs:
                    preds = outputs['clustering_cluster_assignments']
                elif 'clusters' in outputs:
                    preds = outputs['clusters']
                else:
                    preds = outputs.get('predictions', outputs.get('output'))
                
                # Ensure predictions is a tensor with correct shape
                if preds is not None and isinstance(preds, torch.Tensor):
                    # Check if predictions need reshaping
                    if len(preds.shape) == 1:  # Class indices
                        # Convert to one-hot or logits format
                        batch_size = preds.shape[0]
                        logits = torch.zeros(batch_size, self.num_classes, device=preds.device)
                        logits.scatter_(1, preds.unsqueeze(1), 1.0)
                        mc_predictions.append(logits)
                    elif len(preds.shape) == 2:  # Already logits
                        mc_predictions.append(preds)
                    else:
                        # Handle other shapes
                        mc_predictions.append(preds.view(-1, self.num_classes))
                else:
                    # Create dummy predictions if needed
                    batch_size = data.x.shape[0] if hasattr(data, 'x') else 1
                    dummy_preds = torch.randn(batch_size, self.num_classes, device=self.device) * 0.1
                    mc_predictions.append(dummy_preds)
                
                # Store embeddings if available (ensure it's a tensor)
                if 'embeddings' in outputs and isinstance(outputs['embeddings'], torch.Tensor):
                    mc_embeddings.append(outputs['embeddings'])
        
        # Stack predictions
        mc_predictions = torch.stack(mc_predictions)  # [mc_samples, batch, classes]
        
        # Calculate mean and variance
        mean_prediction = mc_predictions.mean(dim=0)
        variance = mc_predictions.var(dim=0)
        
        # Temperature scaling for calibration
        calibrated_mean = mean_prediction / self.temperature
        
        # Softmax probabilities
        probs = F.softmax(calibrated_mean, dim=-1)
        
        # Epistemic uncertainty (variance across MC samples)
        epistemic_uncertainty = variance.mean(dim=-1)
        
        # Aleatoric uncertainty (inherent data uncertainty)
        if mc_embeddings:
            mean_embedding = torch.stack(mc_embeddings).mean(dim=0)
            aleatoric_logits = self.aleatoric_head(mean_embedding)
            aleatoric_uncertainty = F.softplus(aleatoric_logits).mean(dim=-1)
        else:
            # Estimate from prediction entropy
            aleatoric_uncertainty = self._entropy(probs)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        # Confidence scores
        max_probs, predicted_classes = torch.max(probs, dim=-1)
        entropy = self._entropy(probs)
        
        # Calibrated confidence - using only max_probs and entropy for stability
        confidence_input = torch.cat([
            max_probs.unsqueeze(-1),
            entropy.unsqueeze(-1)
        ], dim=-1)
        calibrated_confidence = self.calibration_net(confidence_input).squeeze(-1)
        
        # Uncertainty metrics
        results = {
            'predictions': predicted_classes,
            'probabilities': probs,
            'mean_logits': mean_prediction,
            'epistemic_uncertainty': epistemic_uncertainty,
            'aleatoric_uncertainty': aleatoric_uncertainty,
            'total_uncertainty': total_uncertainty,
            'confidence': calibrated_confidence,
            'raw_confidence': max_probs,
            'entropy': entropy,
            'variance': variance,
            'temperature': self.temperature.item()
        }
        
        if return_all_samples:
            results['mc_samples'] = mc_predictions
        
        # Add uncertainty statistics
        results['uncertainty_stats'] = self._compute_uncertainty_stats(
            epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty
        )
        
        return results
    
    def _entropy(self, probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Calculate entropy of probability distribution."""
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)
    
    def _compute_uncertainty_stats(self, epistemic: torch.Tensor, 
                                  aleatoric: torch.Tensor,
                                  total: torch.Tensor) -> Dict[str, float]:
        """Compute statistics about uncertainty estimates."""
        return {
            'epistemic_mean': epistemic.mean().item(),
            'epistemic_std': epistemic.std().item(),
            'aleatoric_mean': aleatoric.mean().item(),
            'aleatoric_std': aleatoric.std().item(),
            'total_mean': total.mean().item(),
            'total_std': total.std().item(),
            'epistemic_ratio': (epistemic / (total + 1e-8)).mean().item()
        }


class EnsembleUncertainty(nn.Module):
    """
    Ensemble-based uncertainty estimation.
    Trains multiple models and aggregates predictions.
    """
    
    def __init__(self, model_class: type, model_configs: List[Dict],
                 num_models: int = 5):
        super().__init__()
        self.num_models = num_models
        
        # Ensure we have enough configs
        if len(model_configs) < num_models:
            # Duplicate the first config if not enough provided
            model_configs = model_configs + [model_configs[0]] * (num_models - len(model_configs))
        
        # Create ensemble of models
        self.models = nn.ModuleList([
            model_class(config) for config in model_configs[:num_models]
        ])
        
        # Ensemble aggregation methods
        self.aggregation_method = 'weighted_average'  # or 'voting', 'stacking'
        
        # Learned weights for weighted averaging
        self.ensemble_weights = nn.Parameter(torch.ones(num_models) / num_models)
        
        # Stacking meta-learner
        hidden_dim = model_configs[0].get('hidden_dim', 128)
        self.meta_learner = nn.Sequential(
            nn.Linear(hidden_dim * num_models, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, model_configs[0].get('num_classes', 10))
        )
        
    def forward(self, data: Data, method: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble with uncertainty estimation.
        
        Args:
            data: Input graph data
            method: Aggregation method to use
            
        Returns:
            Ensemble predictions with uncertainty
        """
        method = method or self.aggregation_method
        
        # Collect predictions from all models
        all_predictions = []
        all_embeddings = []
        
        for model in self.models:
            outputs = model(data)
            
            # Extract predictions
            if 'clustering_cluster_assignments' in outputs:
                preds = outputs['clustering_cluster_assignments']
            else:
                preds = outputs.get('predictions', outputs.get('clusters'))
            
            all_predictions.append(preds)
            
            if 'embeddings' in outputs:
                all_embeddings.append(outputs['embeddings'])
        
        # Stack predictions
        ensemble_preds = torch.stack(all_predictions)  # [num_models, batch, classes]
        
        # Aggregate based on method
        if method == 'voting':
            # Majority voting
            probs = F.softmax(ensemble_preds, dim=-1)
            mean_probs = probs.mean(dim=0)
            predictions = torch.argmax(mean_probs, dim=-1)
            
        elif method == 'weighted_average':
            # Weighted average with learned weights
            weights = F.softmax(self.ensemble_weights, dim=0)
            weighted_preds = torch.sum(
                ensemble_preds * weights.view(-1, 1, 1),
                dim=0
            )
            probs = F.softmax(weighted_preds, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
        elif method == 'stacking' and all_embeddings:
            # Stacking with meta-learner
            stacked_embeddings = torch.cat(all_embeddings, dim=-1)
            meta_output = self.meta_learner(stacked_embeddings)
            probs = F.softmax(meta_output, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        else:
            # Default to mean
            mean_preds = ensemble_preds.mean(dim=0)
            probs = F.softmax(mean_preds, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
        
        # Calculate uncertainty from ensemble disagreement
        ensemble_probs = F.softmax(ensemble_preds, dim=-1)
        
        # Variance across ensemble (epistemic uncertainty)
        variance = ensemble_probs.var(dim=0)
        epistemic_uncertainty = variance.sum(dim=-1)
        
        # Entropy of mean predictions (aleatoric uncertainty)
        mean_probs = ensemble_probs.mean(dim=0)
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        
        # Disagreement metric
        disagreement = self._calculate_disagreement(ensemble_preds)
        
        return {
            'predictions': predictions,
            'probabilities': probs,
            'ensemble_variance': variance,
            'epistemic_uncertainty': epistemic_uncertainty,
            'entropy': entropy,
            'disagreement': disagreement,
            'confidence': probs.max(dim=-1)[0],
            'ensemble_weights': F.softmax(self.ensemble_weights, dim=0)
        }
    
    def _calculate_disagreement(self, ensemble_preds: torch.Tensor) -> torch.Tensor:
        """Calculate disagreement among ensemble members."""
        # Get predictions from each model
        preds = torch.argmax(ensemble_preds, dim=-1)  # [num_models, batch]
        
        # Calculate pairwise disagreement
        disagreement_sum = 0
        for i in range(self.num_models):
            for j in range(i + 1, self.num_models):
                disagreement_sum += (preds[i] != preds[j]).float()
        
        # Normalize
        num_pairs = self.num_models * (self.num_models - 1) / 2
        disagreement = disagreement_sum / num_pairs
        
        return disagreement


class ConfidenceCalibrator(nn.Module):
    """
    Calibrates model confidence scores to match actual accuracy.
    Implements temperature scaling and Platt scaling.
    """
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        
        # Temperature scaling parameter
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Platt scaling parameters
        self.platt_scale = nn.Parameter(torch.ones(1))
        self.platt_bias = nn.Parameter(torch.zeros(1))
        
        # Histogram binning for ECE calculation
        self.n_bins = 15
        
    def forward(self, logits: torch.Tensor, method: str = 'temperature') -> torch.Tensor:
        """
        Calibrate confidence scores.
        
        Args:
            logits: Raw model outputs
            method: Calibration method ('temperature', 'platt', or 'both')
            
        Returns:
            Calibrated probabilities
        """
        if method == 'temperature' or method == 'both':
            logits = logits / self.temperature
        
        if method == 'platt' or method == 'both':
            logits = logits * self.platt_scale + self.platt_bias
        
        return F.softmax(logits, dim=-1)
    
    def compute_ece(self, probs: torch.Tensor, labels: torch.Tensor) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        Args:
            probs: Predicted probabilities
            labels: True labels
            
        Returns:
            ECE score (lower is better)
        """
        confidences, predictions = torch.max(probs, dim=-1)
        accuracies = (predictions == labels).float()
        
        # Bin confidences
        bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.float().mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece.item()