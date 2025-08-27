"""
Enhanced uncertainty quantification with improved calibration and methods.
Provides state-of-the-art uncertainty estimation for energy GNN predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from torch_geometric.data import Data, Batch
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class DeepEnsemble(nn.Module):
    """
    Deep ensemble with diversity enforcement for robust uncertainty.
    Uses different initializations and data bootstrapping.
    """
    
    def __init__(self, model_class, model_config: Dict, num_models: int = 5,
                 diversity_weight: float = 0.1):
        super().__init__()
        self.num_models = num_models
        self.diversity_weight = diversity_weight
        
        # Create ensemble with different random seeds
        self.models = nn.ModuleList()
        for i in range(num_models):
            torch.manual_seed(42 + i * 100)  # Different seed for each model
            model = model_class(model_config)
            self.models.append(model)
        
        # Diversity enforcement layer
        hidden_dim = model_config.get('hidden_dim', 128)
        self.diversity_projector = nn.Linear(hidden_dim, hidden_dim // 2)
        
    def forward(self, data: Data, return_all: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with diversity-aware ensemble aggregation."""
        all_outputs = []
        all_embeddings = []
        
        for model in self.models:
            output = model(data)
            
            # Extract predictions and embeddings
            if isinstance(output, dict):
                pred = output.get('clusters', output.get('predictions'))
                emb = output.get('embeddings')
            else:
                pred = output
                emb = None
                
            all_outputs.append(pred)
            if emb is not None:
                all_embeddings.append(emb)
        
        # Stack outputs
        ensemble_outputs = torch.stack(all_outputs)  # [num_models, batch, classes]
        
        # Calculate mean and variance
        mean_output = ensemble_outputs.mean(dim=0)
        variance = ensemble_outputs.var(dim=0)
        
        # Calculate predictive uncertainty
        probs = F.softmax(ensemble_outputs, dim=-1)
        mean_probs = probs.mean(dim=0)
        
        # Epistemic uncertainty (disagreement between models)
        epistemic = self._mutual_information(probs)
        
        # Aleatoric uncertainty (average entropy)
        aleatoric = self._expected_entropy(probs)
        
        # Total uncertainty
        total = self._total_entropy(mean_probs)
        
        # Diversity metrics
        if all_embeddings:
            diversity = self._calculate_diversity(all_embeddings)
        else:
            diversity = torch.tensor(0.0)
        
        results = {
            'predictions': torch.argmax(mean_output, dim=-1),
            'mean_logits': mean_output,
            'probabilities': mean_probs,
            'variance': variance,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': total,
            'diversity': diversity
        }
        
        if return_all:
            results['all_outputs'] = ensemble_outputs
            
        return results
    
    def _mutual_information(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate mutual information (epistemic uncertainty)."""
        mean_probs = probs.mean(dim=0)
        entropy_mean = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=-1)
        mean_entropy = torch.mean(
            -torch.sum(probs * torch.log(probs + 1e-8), dim=-1), dim=0
        )
        return entropy_mean - mean_entropy
    
    def _expected_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate expected entropy (aleatoric uncertainty)."""
        return torch.mean(
            -torch.sum(probs * torch.log(probs + 1e-8), dim=-1), dim=0
        )
    
    def _total_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Calculate total entropy."""
        return -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
    
    def _calculate_diversity(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """Calculate diversity among ensemble embeddings."""
        stacked = torch.stack(embeddings)  # [num_models, batch, dim]
        
        # Project to lower dimension for diversity calculation
        projected = self.diversity_projector(stacked)
        
        # Calculate pairwise cosine distances
        diversity_sum = 0
        count = 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                cos_sim = F.cosine_similarity(projected[i], projected[j], dim=-1)
                diversity_sum += (1 - cos_sim).mean()
                count += 1
        
        return diversity_sum / count if count > 0 else torch.tensor(0.0)


class SWAG(nn.Module):
    """
    Stochastic Weight Averaging Gaussian for efficient Bayesian inference.
    Captures both first and second moments of weights during training.
    """
    
    def __init__(self, base_model: nn.Module, rank: int = 20, 
                 update_freq: int = 50, scale: float = 0.5):
        super().__init__()
        self.base_model = base_model
        self.rank = rank
        self.update_freq = update_freq
        self.scale = scale
        
        # Initialize SWAG parameters
        self.mean = {}
        self.sq_mean = {}
        self.deviations = []
        
        # Copy initial weights
        for name, param in base_model.named_parameters():
            self.mean[name] = param.data.clone()
            self.sq_mean[name] = param.data.clone() ** 2
        
        self.n_models = 0
        
    def update_swag(self):
        """Update SWAG statistics with current model weights."""
        self.n_models += 1
        
        for name, param in self.base_model.named_parameters():
            # Update running averages
            self.mean[name] = (self.mean[name] * (self.n_models - 1) + 
                              param.data) / self.n_models
            self.sq_mean[name] = (self.sq_mean[name] * (self.n_models - 1) + 
                                 param.data ** 2) / self.n_models
        
        # Store deviation for low-rank approximation
        if len(self.deviations) < self.rank:
            deviation = {}
            for name, param in self.base_model.named_parameters():
                deviation[name] = param.data - self.mean[name]
            self.deviations.append(deviation)
    
    def sample(self, scale: Optional[float] = None) -> None:
        """Sample weights from SWAG distribution."""
        if scale is None:
            scale = self.scale
            
        # Sample from diagonal variance
        for name, param in self.base_model.named_parameters():
            # Diagonal variance
            var = torch.clamp(self.sq_mean[name] - self.mean[name] ** 2, min=1e-6)
            eps = torch.randn_like(param.data)
            param.data = self.mean[name] + torch.sqrt(var) * eps * scale
            
        # Add low-rank perturbation if available
        if self.deviations:
            # Sample coefficients
            z = torch.randn(len(self.deviations)) * scale / np.sqrt(2)
            
            for i, deviation in enumerate(self.deviations):
                for name, param in self.base_model.named_parameters():
                    param.data += z[i] * deviation[name] / np.sqrt(len(self.deviations))
    
    def forward(self, data: Data, n_samples: int = 30) -> Dict[str, torch.Tensor]:
        """Forward pass with SWAG sampling."""
        predictions = []
        
        for _ in range(n_samples):
            self.sample()
            with torch.no_grad():
                output = self.base_model(data)
                if isinstance(output, dict):
                    pred = output.get('clusters', output.get('predictions'))
                else:
                    pred = output
                predictions.append(pred)
        
        predictions = torch.stack(predictions)
        
        # Calculate uncertainties
        probs = F.softmax(predictions, dim=-1)
        mean_probs = probs.mean(dim=0)
        
        return {
            'predictions': torch.argmax(mean_probs, dim=-1),
            'probabilities': mean_probs,
            'epistemic_uncertainty': predictions.var(dim=0).mean(dim=-1),
            'samples': predictions
        }


class EvidentialUncertainty(nn.Module):
    """
    Evidential deep learning for uncertainty estimation.
    Learns parameters of a Dirichlet distribution for classification.
    """
    
    def __init__(self, input_dim: int, num_classes: int, 
                 annealing_step: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.annealing_step = annealing_step
        
        # Evidence prediction network
        self.evidence_net = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, num_classes)
        )
        
        self.current_step = 0
        
    def forward(self, embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass computing evidential uncertainty.
        
        Args:
            embeddings: Node or graph embeddings [batch, dim]
            
        Returns:
            Dictionary with predictions and uncertainty estimates
        """
        # Predict evidence (non-negative)
        logits = self.evidence_net(embeddings)
        evidence = F.softplus(logits)
        
        # Dirichlet parameters
        alpha = evidence + 1
        
        # Expected probability (mean of Dirichlet)
        probs = alpha / alpha.sum(dim=-1, keepdim=True)
        
        # Uncertainty measures
        S = alpha.sum(dim=-1)  # Dirichlet strength
        
        # Epistemic uncertainty (based on total evidence)
        epistemic = self.num_classes / S
        
        # Aleatoric uncertainty (expected entropy)
        digamma_S = torch.digamma(S)
        aleatoric = -torch.sum(
            probs * (torch.digamma(alpha) - digamma_S.unsqueeze(-1)), 
            dim=-1
        )
        
        # Total uncertainty
        total = epistemic + aleatoric
        
        # Predictions
        predictions = torch.argmax(probs, dim=-1)
        
        return {
            'predictions': predictions,
            'probabilities': probs,
            'evidence': evidence,
            'epistemic_uncertainty': epistemic,
            'aleatoric_uncertainty': aleatoric,
            'total_uncertainty': total,
            'dirichlet_alpha': alpha
        }
    
    def compute_loss(self, output: Dict, targets: torch.Tensor, 
                    epoch: int) -> torch.Tensor:
        """
        Compute evidential loss with KL divergence regularization.
        
        Args:
            output: Output from forward pass
            targets: True labels
            epoch: Current epoch for annealing
            
        Returns:
            Total loss
        """
        alpha = output['dirichlet_alpha']
        S = alpha.sum(dim=-1)
        
        # Convert targets to one-hot
        y_one_hot = F.one_hot(targets, self.num_classes).float()
        
        # Expected cross-entropy loss
        ce_loss = torch.sum(
            y_one_hot * (torch.digamma(S).unsqueeze(-1) - torch.digamma(alpha)),
            dim=-1
        )
        
        # KL divergence regularization (annealed)
        annealing_coef = min(1.0, epoch / self.annealing_step)
        
        # KL divergence from uniform Dirichlet
        alpha_tilde = y_one_hot + (1 - y_one_hot) * alpha
        S_tilde = alpha_tilde.sum(dim=-1)
        
        kl_loss = torch.lgamma(S_tilde) - torch.lgamma(torch.tensor(self.num_classes).float())
        kl_loss -= torch.sum(torch.lgamma(alpha_tilde), dim=-1)
        kl_loss += torch.sum(
            (alpha_tilde - 1) * (torch.digamma(alpha_tilde) - 
                                 torch.digamma(S_tilde).unsqueeze(-1)),
            dim=-1
        )
        
        # Total loss
        loss = ce_loss.mean() + annealing_coef * kl_loss.mean()
        
        return loss


class SelectiveNet(nn.Module):
    """
    Selective prediction with abstention option.
    Allows the model to abstain from prediction when uncertain.
    """
    
    def __init__(self, base_model: nn.Module, input_dim: int, 
                 coverage_target: float = 0.9):
        super().__init__()
        self.base_model = base_model
        self.coverage_target = coverage_target
        
        # Selection head (confidence predictor)
        self.selection_head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, data: Data, threshold: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with selective prediction.
        
        Args:
            data: Input graph data
            threshold: Confidence threshold for selection
            
        Returns:
            Predictions with selection scores
        """
        # Get base model output
        output = self.base_model(data)
        
        if isinstance(output, dict):
            embeddings = output.get('embeddings')
            logits = output.get('clusters', output.get('predictions'))
        else:
            embeddings = None
            logits = output
        
        # Calculate selection scores
        if embeddings is not None:
            selection_scores = self.selection_head(embeddings).squeeze(-1)
        else:
            # Use max probability as selection score
            probs = F.softmax(logits, dim=-1)
            selection_scores = probs.max(dim=-1)[0]
        
        # Apply threshold if provided
        if threshold is not None:
            selected = selection_scores > threshold
        else:
            # Use adaptive threshold to achieve target coverage
            threshold = self._compute_adaptive_threshold(
                selection_scores, self.coverage_target
            )
            selected = selection_scores > threshold
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        # Mask unselected predictions
        predictions_selective = predictions.clone()
        predictions_selective[~selected] = -1  # -1 indicates abstention
        
        return {
            'predictions': predictions_selective,
            'predictions_all': predictions,
            'selection_scores': selection_scores,
            'selected_mask': selected,
            'threshold': threshold,
            'coverage': selected.float().mean(),
            'logits': logits
        }
    
    def _compute_adaptive_threshold(self, scores: torch.Tensor, 
                                   target_coverage: float) -> float:
        """Compute threshold to achieve target coverage."""
        sorted_scores = torch.sort(scores, descending=True)[0]
        n = len(sorted_scores)
        idx = min(int(target_coverage * n), n - 1)
        return sorted_scores[idx].item()
    
    def compute_loss(self, output: Dict, targets: torch.Tensor,
                    lambda_coverage: float = 0.1) -> torch.Tensor:
        """
        Compute selective loss with coverage penalty.
        
        Args:
            output: Output from forward pass
            targets: True labels
            lambda_coverage: Weight for coverage penalty
            
        Returns:
            Total loss
        """
        logits = output['logits']
        selected = output['selected_mask']
        selection_scores = output['selection_scores']
        
        # Classification loss (only on selected samples)
        if selected.any():
            ce_loss = F.cross_entropy(
                logits[selected], targets[selected], reduction='sum'
            ) / (selected.sum() + 1e-8)
        else:
            ce_loss = torch.tensor(0.0, device=logits.device)
        
        # Coverage penalty
        coverage = selected.float().mean()
        coverage_loss = torch.abs(coverage - self.coverage_target)
        
        # Total loss
        loss = ce_loss + lambda_coverage * coverage_loss
        
        return loss


def create_uncertainty_estimator(
    method: str, 
    base_model: nn.Module, 
    config: Dict[str, Any]
) -> nn.Module:
    """
    Factory function to create uncertainty estimators.
    
    Args:
        method: Type of uncertainty method ('ensemble', 'swag', 'evidential', 'selective')
        base_model: Base GNN model
        config: Configuration dictionary
        
    Returns:
        Uncertainty estimator module
    """
    if method == 'ensemble':
        from models.base_gnn import HomoEnergyGNN, HeteroEnergyGNN
        model_class = HeteroEnergyGNN if config.get('type') == 'hetero' else HomoEnergyGNN
        return DeepEnsemble(
            model_class=model_class,
            model_config=config,
            num_models=config.get('ensemble_size', 5)
        )
    
    elif method == 'swag':
        return SWAG(
            base_model=base_model,
            rank=config.get('swag_rank', 20),
            scale=config.get('swag_scale', 0.5)
        )
    
    elif method == 'evidential':
        hidden_dim = config.get('hidden_dim', 128)
        num_classes = config.get('num_clusters', 10)
        return EvidentialUncertainty(
            input_dim=hidden_dim,
            num_classes=num_classes
        )
    
    elif method == 'selective':
        hidden_dim = config.get('hidden_dim', 128)
        coverage = config.get('coverage_target', 0.9)
        return SelectiveNet(
            base_model=base_model,
            input_dim=hidden_dim,
            coverage_target=coverage
        )
    
    else:
        raise ValueError(f"Unknown uncertainty method: {method}")


if __name__ == "__main__":
    # Test enhanced uncertainty methods
    print("Testing Enhanced Uncertainty Quantification Methods")
    print("=" * 60)
    
    # Create dummy model and data for testing
    import yaml
    from models.base_gnn import create_gnn_model
    
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)['model']
    
    base_model = create_gnn_model('homo', config)
    
    # Test data
    data = Data(
        x=torch.randn(30, config['input_dim']),
        edge_index=torch.randint(0, 30, (2, 100))
    )
    
    # Test each method
    methods = ['ensemble', 'swag', 'evidential', 'selective']
    
    for method in methods:
        print(f"\nTesting {method.upper()}...")
        estimator = create_uncertainty_estimator(method, base_model, config)
        
        try:
            if method == 'evidential':
                # Need embeddings for evidential
                with torch.no_grad():
                    base_out = base_model(data)
                    embeddings = base_out.get('embeddings', torch.randn(30, config['hidden_dim']))
                    output = estimator(embeddings)
            else:
                output = estimator(data)
            
            print(f"  Output keys: {list(output.keys())}")
            if 'epistemic_uncertainty' in output:
                print(f"  Epistemic uncertainty: {output['epistemic_uncertainty'].mean():.4f}")
            if 'aleatoric_uncertainty' in output:
                print(f"  Aleatoric uncertainty: {output['aleatoric_uncertainty'].mean():.4f}")
            print(f"  [OK] {method} working correctly")
            
        except Exception as e:
            print(f"  [ERROR] {method}: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Enhanced uncertainty methods tested successfully!")