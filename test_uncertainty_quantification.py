"""
Comprehensive test suite for uncertainty quantification modules.
Tests calibration, MC Dropout, ensemble methods, and confidence intervals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data
import yaml
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.uncertainty_quantification import (
    MCDropout, BayesianGNNLayer, UncertaintyQuantifier,
    EnsembleUncertainty, ConfidenceCalibrator
)
from models.enhanced_uncertainty import (
    DeepEnsemble, SWAG, EvidentialUncertainty, 
    SelectiveNet, create_uncertainty_estimator
)
from models.base_gnn import create_gnn_model
import warnings
warnings.filterwarnings('ignore')


class UncertaintyTestSuite:
    """Comprehensive test suite for uncertainty quantification."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize test suite with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model_config = self.config['model']
        # Ensure input_dim is set
        if self.model_config.get('input_dim') is None:
            self.model_config['input_dim'] = 17  # Default building features
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def create_synthetic_data(self, num_nodes: int = 100, 
                            num_edges: int = 300,
                            num_features: int = None) -> Data:
        """Create synthetic graph data for testing."""
        if num_features is None:
            num_features = self.model_config.get('input_dim', 17)
        
        # Create random graph
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        x = torch.randn(num_nodes, num_features)
        
        # Add some structure for more realistic testing
        # Create clusters in the feature space
        num_clusters = 5
        for i in range(num_clusters):
            cluster_nodes = range(i * (num_nodes // num_clusters), 
                                 (i + 1) * (num_nodes // num_clusters))
            cluster_center = torch.randn(num_features)
            for node in cluster_nodes:
                if node < num_nodes:
                    x[node] = cluster_center + torch.randn(num_features) * 0.3
        
        # Create ground truth labels based on clusters
        y = torch.tensor([i // (num_nodes // num_clusters) 
                         for i in range(num_nodes)])
        
        data = Data(x=x, edge_index=edge_index, y=y)
        data = data.to(self.device)
        
        return data
    
    def test_mc_dropout(self) -> Dict:
        """Test Monte Carlo Dropout functionality."""
        print("\n" + "="*60)
        print("Testing Monte Carlo Dropout")
        print("="*60)
        
        results = {
            'passed': [],
            'failed': [],
            'metrics': {}
        }
        
        # Test 1: Dropout stays active during inference
        print("\n1. Testing dropout behavior during inference...")
        mc_dropout = MCDropout(p=0.5)
        x = torch.ones(10, 20)
        
        # Collect multiple forward passes
        outputs = []
        for _ in range(100):
            with torch.no_grad():
                out = mc_dropout(x, training=False)  # Should still drop despite training=False
                outputs.append(out)
        
        outputs = torch.stack(outputs)
        variance = outputs.var(dim=0).mean().item()
        
        if variance > 0.1:  # Should have variance due to dropout
            print("  [PASS] MC Dropout stays active during inference")
            results['passed'].append('MC Dropout activation')
        else:
            print("  [FAIL] MC Dropout not working properly (variance too low)")
            results['failed'].append('MC Dropout activation')
        
        results['metrics']['dropout_variance'] = variance
        
        # Test 2: Uncertainty increases with higher dropout rate
        print("\n2. Testing uncertainty vs dropout rate...")
        dropout_rates = [0.1, 0.3, 0.5, 0.7]
        uncertainties = []
        
        for p in dropout_rates:
            mc = MCDropout(p=p)
            outputs = []
            for _ in range(50):
                with torch.no_grad():
                    outputs.append(mc(x))
            outputs = torch.stack(outputs)
            uncertainties.append(outputs.var(dim=0).mean().item())
        
        # Check if uncertainty increases with dropout rate
        if all(uncertainties[i] < uncertainties[i+1] for i in range(len(uncertainties)-1)):
            print("  [PASS] Uncertainty increases with dropout rate")
            results['passed'].append('Dropout rate correlation')
        else:
            print("  [FAIL] Uncertainty doesn't correlate with dropout rate")
            results['failed'].append('Dropout rate correlation')
        
        results['metrics']['dropout_uncertainties'] = dict(zip(dropout_rates, uncertainties))
        
        return results
    
    def test_bayesian_layer(self) -> Dict:
        """Test Bayesian GNN layer with aleatoric and epistemic uncertainty."""
        print("\n" + "="*60)
        print("Testing Bayesian GNN Layer")
        print("="*60)
        
        results = {
            'passed': [],
            'failed': [],
            'metrics': {}
        }
        
        # Test 1: Layer produces mean and variance
        print("\n1. Testing mean and variance output...")
        layer = BayesianGNNLayer(64, 32, dropout_rate=0.2)
        x = torch.randn(50, 64)
        
        mean, log_var = layer(x, sample=True)
        
        if mean.shape == (50, 32) and log_var.shape == (50, 32):
            print("  [PASS] Correct output shapes for mean and log_var")
            results['passed'].append('Bayesian layer shapes')
        else:
            print("  [FAIL] Incorrect output shapes")
            results['failed'].append('Bayesian layer shapes')
        
        # Test 2: Sampling produces different outputs
        print("\n2. Testing stochastic sampling...")
        samples = []
        for _ in range(10):
            sample, _ = layer(x, sample=True)
            samples.append(sample)
        
        samples = torch.stack(samples)
        sample_variance = samples.var(dim=0).mean().item()
        
        if sample_variance > 0.01:
            print("  [PASS] Sampling produces stochastic outputs")
            results['passed'].append('Stochastic sampling')
        else:
            print("  [FAIL] Sampling not stochastic enough")
            results['failed'].append('Stochastic sampling')
        
        results['metrics']['sample_variance'] = sample_variance
        
        # Test 3: Log variance is properly bounded
        print("\n3. Testing log variance bounds...")
        if log_var.min() >= -20 and log_var.max() <= 2:
            print("  [PASS] Log variance properly bounded")
            results['passed'].append('Variance bounds')
        else:
            print("  [FAIL] Log variance out of bounds")
            results['failed'].append('Variance bounds')
        
        results['metrics']['log_var_range'] = (log_var.min().item(), log_var.max().item())
        
        return results
    
    def test_uncertainty_quantifier(self) -> Dict:
        """Test comprehensive uncertainty quantification."""
        print("\n" + "="*60)
        print("Testing Uncertainty Quantifier")
        print("="*60)
        
        results = {
            'passed': [],
            'failed': [],
            'metrics': {}
        }
        
        # Create base model and uncertainty quantifier
        base_model = create_gnn_model('homo', self.model_config)
        base_model = base_model.to(self.device)
        
        uq = UncertaintyQuantifier(
            base_model, 
            config=self.model_config,
            num_classes=10,
            mc_samples=20
        ).to(self.device)
        
        # Create test data
        data = self.create_synthetic_data(num_nodes=50)
        
        # Test 1: Forward pass produces all uncertainty metrics
        print("\n1. Testing uncertainty metrics output...")
        try:
            output = uq(data, return_all_samples=True)
            
            required_keys = [
                'predictions', 'probabilities', 'epistemic_uncertainty',
                'aleatoric_uncertainty', 'total_uncertainty', 'confidence',
                'entropy', 'mc_samples'
            ]
            
            missing_keys = [k for k in required_keys if k not in output]
            if not missing_keys:
                print("  [PASS] All uncertainty metrics present")
                results['passed'].append('UQ metrics completeness')
            else:
                print(f"  [FAIL] Missing keys: {missing_keys}")
                results['failed'].append('UQ metrics completeness')
            
        except Exception as e:
            print(f"  [FAIL] Error in forward pass: {e}")
            results['failed'].append('UQ forward pass')
            return results
        
        # Test 2: Epistemic vs Aleatoric uncertainty relationship
        print("\n2. Testing uncertainty decomposition...")
        epistemic = output['epistemic_uncertainty'].mean().item()
        aleatoric = output['aleatoric_uncertainty'].mean().item()
        total = output['total_uncertainty'].mean().item()
        
        # Total should approximately equal sum of components
        if abs(total - (epistemic + aleatoric)) < 0.5:
            print("  [PASS] Uncertainty decomposition is consistent")
            results['passed'].append('Uncertainty decomposition')
        else:
            print("  [FAIL] Uncertainty decomposition inconsistent")
            results['failed'].append('Uncertainty decomposition')
        
        results['metrics']['epistemic_mean'] = epistemic
        results['metrics']['aleatoric_mean'] = aleatoric
        results['metrics']['total_mean'] = total
        
        # Test 3: MC samples variance
        print("\n3. Testing MC sample diversity...")
        mc_samples = output['mc_samples']  # [mc_samples, batch, classes]
        sample_variance = mc_samples.var(dim=0).mean().item()
        
        if sample_variance > 0.01:
            print("  [PASS] MC samples show diversity")
            results['passed'].append('MC sample diversity')
        else:
            print("  [FAIL] MC samples too similar")
            results['failed'].append('MC sample diversity')
        
        results['metrics']['mc_sample_variance'] = sample_variance
        
        # Test 4: Confidence calibration
        print("\n4. Testing confidence calibration...")
        confidence = output['confidence'].mean().item()
        raw_confidence = output['raw_confidence'].mean().item()
        
        if 0 <= confidence <= 1 and 0 <= raw_confidence <= 1:
            print("  [PASS] Confidence scores in valid range")
            results['passed'].append('Confidence range')
        else:
            print("  [FAIL] Confidence scores out of range")
            results['failed'].append('Confidence range')
        
        results['metrics']['calibrated_confidence'] = confidence
        results['metrics']['raw_confidence'] = raw_confidence
        
        return results
    
    def test_ensemble_uncertainty(self) -> Dict:
        """Test ensemble-based uncertainty estimation."""
        print("\n" + "="*60)
        print("Testing Ensemble Uncertainty")
        print("="*60)
        
        results = {
            'passed': [],
            'failed': [],
            'metrics': {}
        }
        
        # Create ensemble with smaller models for testing
        small_config = self.model_config.copy()
        small_config['hidden_dim'] = 32
        
        print("\n1. Creating ensemble models...")
        try:
            # Create multiple configs for ensemble
            configs = [small_config.copy() for _ in range(3)]
            
            from models.base_gnn import HomoEnergyGNN
            ensemble = EnsembleUncertainty(
                model_class=HomoEnergyGNN,
                model_configs=configs,
                num_models=3
            ).to(self.device)
            
            print("  [PASS] Ensemble created successfully")
            results['passed'].append('Ensemble creation')
        except Exception as e:
            print(f"  [FAIL] Failed to create ensemble: {e}")
            results['failed'].append('Ensemble creation')
            return results
        
        # Test forward pass
        print("\n2. Testing ensemble forward pass...")
        data = self.create_synthetic_data(num_nodes=30)
        
        try:
            # Test different aggregation methods
            methods = ['voting', 'weighted_average']
            
            for method in methods:
                output = ensemble(data, method=method)
                
                if 'predictions' in output and 'epistemic_uncertainty' in output:
                    print(f"  [PASS] {method} aggregation works")
                    results['passed'].append(f'Ensemble {method}')
                else:
                    print(f"  [FAIL] {method} aggregation failed")
                    results['failed'].append(f'Ensemble {method}')
            
        except Exception as e:
            print(f"  [FAIL] Ensemble forward pass failed: {e}")
            results['failed'].append('Ensemble forward')
        
        # Test 3: Ensemble disagreement
        print("\n3. Testing ensemble disagreement metric...")
        if 'disagreement' in output:
            disagreement = output['disagreement'].mean().item()
            if 0 <= disagreement <= 1:
                print("  [PASS] Disagreement metric in valid range")
                results['passed'].append('Ensemble disagreement')
            else:
                print("  [FAIL] Disagreement metric invalid")
                results['failed'].append('Ensemble disagreement')
            results['metrics']['ensemble_disagreement'] = disagreement
        
        return results
    
    def test_calibration(self) -> Dict:
        """Test confidence calibration methods."""
        print("\n" + "="*60)
        print("Testing Confidence Calibration")
        print("="*60)
        
        results = {
            'passed': [],
            'failed': [],
            'metrics': {}
        }
        
        # Create calibrator
        calibrator = ConfidenceCalibrator(num_classes=10).to(self.device)
        
        # Test 1: Temperature scaling
        print("\n1. Testing temperature scaling...")
        logits = torch.randn(100, 10).to(self.device)
        
        # Before calibration
        probs_before = F.softmax(logits, dim=-1)
        
        # After calibration
        probs_after = calibrator(logits, method='temperature')
        
        # Check if probabilities are valid
        if torch.allclose(probs_after.sum(dim=-1), torch.ones(100).to(self.device), atol=1e-6):
            print("  [PASS] Temperature scaling produces valid probabilities")
            results['passed'].append('Temperature scaling')
        else:
            print("  [FAIL] Temperature scaling invalid")
            results['failed'].append('Temperature scaling')
        
        # Test 2: ECE calculation
        print("\n2. Testing Expected Calibration Error (ECE)...")
        labels = torch.randint(0, 10, (100,)).to(self.device)
        
        try:
            ece_before = calibrator.compute_ece(probs_before, labels)
            ece_after = calibrator.compute_ece(probs_after, labels)
            
            print(f"  ECE before: {ece_before:.4f}")
            print(f"  ECE after:  {ece_after:.4f}")
            
            results['metrics']['ece_before'] = ece_before
            results['metrics']['ece_after'] = ece_after
            
            if 0 <= ece_before <= 1 and 0 <= ece_after <= 1:
                print("  [PASS] ECE calculation successful")
                results['passed'].append('ECE calculation')
            else:
                print("  [FAIL] ECE values invalid")
                results['failed'].append('ECE calculation')
                
        except Exception as e:
            print(f"  [FAIL] ECE calculation failed: {e}")
            results['failed'].append('ECE calculation')
        
        # Test 3: Platt scaling
        print("\n3. Testing Platt scaling...")
        probs_platt = calibrator(logits, method='platt')
        
        if torch.allclose(probs_platt.sum(dim=-1), torch.ones(100).to(self.device), atol=1e-6):
            print("  [PASS] Platt scaling produces valid probabilities")
            results['passed'].append('Platt scaling')
        else:
            print("  [FAIL] Platt scaling invalid")
            results['failed'].append('Platt scaling')
        
        return results
    
    def test_enhanced_methods(self) -> Dict:
        """Test enhanced uncertainty methods from enhanced_uncertainty.py."""
        print("\n" + "="*60)
        print("Testing Enhanced Uncertainty Methods")
        print("="*60)
        
        results = {
            'passed': [],
            'failed': [],
            'metrics': {}
        }
        
        # Test 1: Deep Ensemble with diversity
        print("\n1. Testing Deep Ensemble...")
        try:
            from models.base_gnn import HomoEnergyGNN
            deep_ensemble = DeepEnsemble(
                model_class=HomoEnergyGNN,
                model_config=self.model_config,
                num_models=3,
                diversity_weight=0.1
            ).to(self.device)
            
            data = self.create_synthetic_data(num_nodes=30)
            output = deep_ensemble(data, return_all=True)
            
            if 'diversity' in output:
                diversity = output['diversity'].item() if torch.is_tensor(output['diversity']) else output['diversity']
                print(f"  Ensemble diversity: {diversity:.4f}")
                results['metrics']['ensemble_diversity'] = diversity
                results['passed'].append('Deep Ensemble')
            else:
                results['failed'].append('Deep Ensemble - no diversity')
                
        except Exception as e:
            print(f"  [FAIL] Deep Ensemble failed: {e}")
            results['failed'].append('Deep Ensemble')
        
        # Test 2: Evidential Uncertainty
        print("\n2. Testing Evidential Uncertainty...")
        try:
            evidential = EvidentialUncertainty(
                input_dim=128,
                num_classes=10,
                annealing_step=10
            ).to(self.device)
            
            embeddings = torch.randn(50, 128).to(self.device)
            output = evidential(embeddings)
            
            if 'dirichlet_alpha' in output:
                alpha = output['dirichlet_alpha']
                # Check if Dirichlet parameters are positive
                if (alpha > 0).all():
                    print("  [PASS] Valid Dirichlet parameters")
                    results['passed'].append('Evidential uncertainty')
                else:
                    print("  [FAIL] Invalid Dirichlet parameters")
                    results['failed'].append('Evidential uncertainty')
                    
                # Test uncertainty decomposition
                epistemic = output['epistemic_uncertainty'].mean().item()
                aleatoric = output['aleatoric_uncertainty'].mean().item()
                print(f"  Epistemic: {epistemic:.4f}, Aleatoric: {aleatoric:.4f}")
                results['metrics']['evidential_epistemic'] = epistemic
                results['metrics']['evidential_aleatoric'] = aleatoric
                
        except Exception as e:
            print(f"  [FAIL] Evidential uncertainty failed: {e}")
            results['failed'].append('Evidential uncertainty')
        
        # Test 3: Selective Network
        print("\n3. Testing Selective Network...")
        try:
            base_model = create_gnn_model('homo', self.model_config).to(self.device)
            selective = SelectiveNet(
                base_model=base_model,
                input_dim=128,
                coverage_target=0.8
            ).to(self.device)
            
            data = self.create_synthetic_data(num_nodes=30)
            output = selective(data, threshold=0.5)
            
            if 'selected_mask' in output:
                coverage = output['coverage'].item()
                print(f"  Coverage: {coverage:.2%}")
                
                # Check abstention
                predictions = output['predictions']
                abstained = (predictions == -1).sum().item()
                print(f"  Abstained on {abstained} samples")
                
                results['metrics']['selective_coverage'] = coverage
                results['metrics']['selective_abstentions'] = abstained
                results['passed'].append('Selective Network')
            else:
                results['failed'].append('Selective Network')
                
        except Exception as e:
            print(f"  [FAIL] Selective Network failed: {e}")
            results['failed'].append('Selective Network')
        
        return results
    
    def test_confidence_correlation(self) -> Dict:
        """Test if confidence scores correlate with prediction accuracy."""
        print("\n" + "="*60)
        print("Testing Confidence-Accuracy Correlation")
        print("="*60)
        
        results = {
            'passed': [],
            'failed': [],
            'metrics': {}
        }
        
        # Create model with uncertainty
        base_model = create_gnn_model('homo', self.model_config).to(self.device)
        uq = UncertaintyQuantifier(
            base_model,
            config=self.model_config,
            num_classes=5,  # Fewer classes for clearer testing
            mc_samples=10
        ).to(self.device)
        
        # Generate data with varying difficulty
        easy_data = self.create_synthetic_data(num_nodes=50)
        hard_data = self.create_synthetic_data(num_nodes=50)
        
        # Add noise to hard data
        hard_data.x += torch.randn_like(hard_data.x) * 2.0
        
        print("\n1. Testing on easy vs hard data...")
        
        # Get predictions and confidence for both
        with torch.no_grad():
            easy_output = uq(easy_data)
            hard_output = uq(hard_data)
        
        easy_conf = easy_output['confidence'].mean().item()
        hard_conf = hard_output['confidence'].mean().item()
        
        easy_unc = easy_output['total_uncertainty'].mean().item()
        hard_unc = hard_output['total_uncertainty'].mean().item()
        
        print(f"  Easy data - Confidence: {easy_conf:.4f}, Uncertainty: {easy_unc:.4f}")
        print(f"  Hard data - Confidence: {hard_conf:.4f}, Uncertainty: {hard_unc:.4f}")
        
        # Confidence should be higher for easy data
        if easy_conf > hard_conf:
            print("  [PASS] Confidence higher for easy data")
            results['passed'].append('Confidence ordering')
        else:
            print("  [FAIL] Confidence not properly ordered")
            results['failed'].append('Confidence ordering')
        
        # Uncertainty should be lower for easy data
        if easy_unc < hard_unc:
            print("  [PASS] Uncertainty lower for easy data")
            results['passed'].append('Uncertainty ordering')
        else:
            print("  [FAIL] Uncertainty not properly ordered")
            results['failed'].append('Uncertainty ordering')
        
        results['metrics']['easy_confidence'] = easy_conf
        results['metrics']['hard_confidence'] = hard_conf
        results['metrics']['easy_uncertainty'] = easy_unc
        results['metrics']['hard_uncertainty'] = hard_unc
        
        # Test 2: Confidence vs correctness correlation
        print("\n2. Testing confidence-correctness correlation...")
        
        # Generate multiple samples
        confidences = []
        correctness = []
        
        for _ in range(10):
            data = self.create_synthetic_data(num_nodes=30)
            with torch.no_grad():
                output = uq(data)
            
            predictions = output['predictions']
            confidence = output['confidence']
            
            # Check correctness (using synthetic labels)
            correct = (predictions == data.y).float()
            
            confidences.extend(confidence.cpu().numpy())
            correctness.extend(correct.cpu().numpy())
        
        # Calculate correlation
        from scipy.stats import spearmanr
        corr, p_value = spearmanr(confidences, correctness)
        
        print(f"  Spearman correlation: {corr:.4f} (p={p_value:.4f})")
        
        if corr > 0.1:  # Positive correlation expected
            print("  [PASS] Positive confidence-accuracy correlation")
            results['passed'].append('Confidence correlation')
        else:
            print("  [FAIL] No positive correlation found")
            results['failed'].append('Confidence correlation')
        
        results['metrics']['confidence_accuracy_correlation'] = corr
        results['metrics']['correlation_p_value'] = p_value
        
        return results
    
    def test_confidence_intervals(self) -> Dict:
        """Test confidence interval calculations."""
        print("\n" + "="*60)
        print("Testing Confidence Intervals")
        print("="*60)
        
        results = {
            'passed': [],
            'failed': [],
            'metrics': {}
        }
        
        # Create model
        base_model = create_gnn_model('homo', self.model_config).to(self.device)
        uq = UncertaintyQuantifier(
            base_model,
            config=self.model_config,
            mc_samples=100  # More samples for better intervals
        ).to(self.device)
        
        data = self.create_synthetic_data(num_nodes=30)
        
        print("\n1. Calculating prediction intervals...")
        with torch.no_grad():
            output = uq(data, return_all_samples=True)
        
        mc_samples = output['mc_samples']  # [mc_samples, batch, classes]
        
        # Calculate percentiles for confidence intervals
        probs = F.softmax(mc_samples, dim=-1)
        
        # 95% confidence interval
        lower_95 = torch.quantile(probs, 0.025, dim=0)
        upper_95 = torch.quantile(probs, 0.975, dim=0)
        
        # 90% confidence interval  
        lower_90 = torch.quantile(probs, 0.05, dim=0)
        upper_90 = torch.quantile(probs, 0.95, dim=0)
        
        # Check interval properties
        mean_probs = probs.mean(dim=0)
        
        # Test 1: Mean should be within intervals
        within_95 = ((mean_probs >= lower_95) & (mean_probs <= upper_95)).all()
        within_90 = ((mean_probs >= lower_90) & (mean_probs <= upper_90)).all()
        
        if within_95 and within_90:
            print("  [PASS] Mean predictions within confidence intervals")
            results['passed'].append('Interval consistency')
        else:
            print("  [FAIL] Mean predictions outside intervals")
            results['failed'].append('Interval consistency')
        
        # Test 2: 95% interval should be wider than 90%
        width_95 = (upper_95 - lower_95).mean().item()
        width_90 = (upper_90 - lower_90).mean().item()
        
        print(f"  95% CI width: {width_95:.4f}")
        print(f"  90% CI width: {width_90:.4f}")
        
        if width_95 > width_90:
            print("  [PASS] Interval widths properly ordered")
            results['passed'].append('Interval ordering')
        else:
            print("  [FAIL] Interval widths incorrectly ordered")
            results['failed'].append('Interval ordering')
        
        results['metrics']['ci_95_width'] = width_95
        results['metrics']['ci_90_width'] = width_90
        
        # Test 3: Coverage validation
        print("\n2. Validating coverage rates...")
        
        # For each sample, check if true label falls within CI
        # (Using predicted as proxy since we don't have true probabilities)
        predictions = output['predictions']
        
        coverage_95 = 0
        coverage_90 = 0
        
        for i in range(len(predictions)):
            pred_class = predictions[i]
            # Check if predicted class has high probability in CI
            if lower_95[i, pred_class] > 0.1:  # Reasonable threshold
                coverage_95 += 1
            if lower_90[i, pred_class] > 0.1:
                coverage_90 += 1
        
        coverage_95 = coverage_95 / len(predictions)
        coverage_90 = coverage_90 / len(predictions)
        
        print(f"  Empirical 95% coverage: {coverage_95:.2%}")
        print(f"  Empirical 90% coverage: {coverage_90:.2%}")
        
        results['metrics']['empirical_coverage_95'] = coverage_95
        results['metrics']['empirical_coverage_90'] = coverage_90
        
        return results
    
    def visualize_calibration(self, save_path: str = 'calibration_plot.png'):
        """Create calibration plots."""
        print("\n" + "="*60)
        print("Creating Calibration Visualizations")
        print("="*60)
        
        # Create model
        base_model = create_gnn_model('homo', self.model_config).to(self.device)
        uq = UncertaintyQuantifier(
            base_model,
            config=self.model_config,
            num_classes=5,
            mc_samples=20
        ).to(self.device)
        
        # Generate predictions on multiple batches
        all_confidences = []
        all_accuracies = []
        all_predictions = []
        all_labels = []
        
        for _ in range(20):
            data = self.create_synthetic_data(num_nodes=50)
            with torch.no_grad():
                output = uq(data)
            
            predictions = output['predictions'].cpu()
            confidences = output['confidence'].cpu()
            labels = data.y.cpu()
            
            all_predictions.extend(predictions.numpy())
            all_confidences.extend(confidences.numpy())
            all_labels.extend(labels.numpy())
            
            correct = (predictions == labels).float()
            all_accuracies.extend(correct.numpy())
        
        # Create calibration plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Reliability diagram
        ax = axes[0, 0]
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        accuracies = []
        confidences = []
        counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (all_confidences >= bin_lower) & (all_confidences < bin_upper)
            if in_bin.sum() > 0:
                accuracies.append(np.mean(np.array(all_accuracies)[in_bin]))
                confidences.append(np.mean(np.array(all_confidences)[in_bin]))
                counts.append(in_bin.sum())
            else:
                accuracies.append(0)
                confidences.append((bin_lower + bin_upper) / 2)
                counts.append(0)
        
        # Plot reliability diagram
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax.bar(confidences, accuracies, width=1.0/n_bins, alpha=0.7, 
               edgecolor='black', label='Model')
        ax.set_xlabel('Mean Confidence')
        ax.set_ylabel('Accuracy')
        ax.set_title('Reliability Diagram')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Confidence histogram
        ax = axes[0, 1]
        ax.hist(all_confidences, bins=20, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution')
        ax.grid(True, alpha=0.3)
        
        # 3. Accuracy vs Confidence scatter
        ax = axes[1, 0]
        ax.scatter(all_confidences, all_accuracies, alpha=0.3, s=10)
        ax.set_xlabel('Confidence')
        ax.set_ylabel('Correct (0/1)')
        ax.set_title('Confidence vs Correctness')
        ax.grid(True, alpha=0.3)
        
        # 4. ECE over time (simulate training)
        ax = axes[1, 1]
        calibrator = ConfidenceCalibrator(num_classes=5).to(self.device)
        
        eces = []
        for epoch in range(10):
            data = self.create_synthetic_data(num_nodes=50)
            logits = torch.randn(50, 5).to(self.device)
            probs = F.softmax(logits / (1 + epoch * 0.1), dim=-1)  # Simulate improving calibration
            labels = data.y[:50].to(self.device)
            ece = calibrator.compute_ece(probs, labels)
            eces.append(ece)
        
        ax.plot(range(10), eces, 'o-')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('ECE')
        ax.set_title('Expected Calibration Error Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        print(f"  Calibration plots saved to {save_path}")
        
        return save_path
    
    def run_all_tests(self) -> Dict:
        """Run complete test suite."""
        print("\n" + "="*80)
        print("COMPREHENSIVE UNCERTAINTY QUANTIFICATION TEST SUITE")
        print("="*80)
        
        all_results = {}
        
        # Run individual test suites
        all_results['mc_dropout'] = self.test_mc_dropout()
        all_results['bayesian_layer'] = self.test_bayesian_layer()
        all_results['uncertainty_quantifier'] = self.test_uncertainty_quantifier()
        all_results['ensemble'] = self.test_ensemble_uncertainty()
        all_results['calibration'] = self.test_calibration()
        all_results['enhanced_methods'] = self.test_enhanced_methods()
        all_results['confidence_correlation'] = self.test_confidence_correlation()
        all_results['confidence_intervals'] = self.test_confidence_intervals()
        
        # Create visualization
        viz_path = self.visualize_calibration('uncertainty_calibration.png')
        
        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        total_passed = 0
        total_failed = 0
        
        for test_name, results in all_results.items():
            passed = len(results.get('passed', []))
            failed = len(results.get('failed', []))
            total_passed += passed
            total_failed += failed
            
            status = "PASS" if failed == 0 else "FAIL"
            print(f"\n{test_name}: {status}")
            print(f"  Passed: {passed}, Failed: {failed}")
            
            if results.get('metrics'):
                print("  Key Metrics:")
                for metric, value in list(results['metrics'].items())[:3]:
                    if isinstance(value, float):
                        print(f"    - {metric}: {value:.4f}")
                    else:
                        print(f"    - {metric}: {value}")
        
        print("\n" + "="*80)
        print(f"OVERALL: {total_passed} passed, {total_failed} failed")
        
        if total_failed == 0:
            print("[SUCCESS] ALL TESTS PASSED - Uncertainty quantification working correctly!")
        else:
            print("[WARNING] SOME TESTS FAILED - Review failed components")
        
        print("="*80)
        
        return all_results


def main():
    """Run uncertainty quantification tests."""
    tester = UncertaintyTestSuite()
    results = tester.run_all_tests()
    
    # Save results to file
    import json
    with open('uncertainty_test_results.json', 'w') as f:
        # Convert numpy/torch values to Python types
        clean_results = {}
        for test_name, test_results in results.items():
            clean_results[test_name] = {
                'passed': test_results.get('passed', []),
                'failed': test_results.get('failed', []),
                'metrics': {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v 
                          for k, v in test_results.get('metrics', {}).items()}
            }
        json.dump(clean_results, f, indent=2)
    
    print(f"\nResults saved to uncertainty_test_results.json")
    
    return results


if __name__ == "__main__":
    main()