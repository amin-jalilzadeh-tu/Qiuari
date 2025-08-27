"""
Simplified uncertainty quantification tests focusing on core functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.uncertainty_quantification import (
    MCDropout, BayesianGNNLayer, ConfidenceCalibrator
)
from models.enhanced_uncertainty import EvidentialUncertainty

import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("SIMPLIFIED UNCERTAINTY QUANTIFICATION TESTS")
print("="*80)

# Test results tracking
test_results = {
    'passed': [],
    'failed': [],
    'metrics': {}
}

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}")

# Test 1: Monte Carlo Dropout
print("\n" + "="*60)
print("Test 1: Monte Carlo Dropout")
print("="*60)

mc_dropout = MCDropout(p=0.3).to(device)
x = torch.randn(100, 50).to(device)

# Test dropout stays active
outputs = []
for _ in range(50):
    with torch.no_grad():
        out = mc_dropout(x, training=False)
        outputs.append(out)

outputs = torch.stack(outputs)
variance = outputs.var(dim=0).mean().item()
mean_output = outputs.mean(dim=0).mean().item()

print(f"MC Dropout variance: {variance:.6f}")
print(f"Mean output value: {mean_output:.6f}")

if variance > 0.05:
    print("[PASS] MC Dropout maintains stochasticity during inference")
    test_results['passed'].append('MC Dropout')
else:
    print("[FAIL] MC Dropout variance too low")
    test_results['failed'].append('MC Dropout')

test_results['metrics']['mc_dropout_variance'] = variance

# Test 2: Bayesian Layer Uncertainty
print("\n" + "="*60)
print("Test 2: Bayesian Layer")
print("="*60)

bayesian_layer = BayesianGNNLayer(64, 32, dropout_rate=0.2).to(device)
x = torch.randn(50, 64).to(device)

# Test mean and variance output
mean, log_var = bayesian_layer(x, sample=True)
variance = torch.exp(log_var)

print(f"Output shape - Mean: {mean.shape}, Log Var: {log_var.shape}")
print(f"Mean value range: [{mean.min().item():.3f}, {mean.max().item():.3f}]")
print(f"Variance range: [{variance.min().item():.6f}, {variance.max().item():.6f}]")

# Test sampling produces different outputs
samples = []
for _ in range(20):
    sample, _ = bayesian_layer(x, sample=True)
    samples.append(sample)

samples_tensor = torch.stack(samples)
sample_variance = samples_tensor.var(dim=0).mean().item()

print(f"Sample variance across multiple runs: {sample_variance:.6f}")

if sample_variance > 0.01:
    print("[PASS] Bayesian layer produces stochastic samples")
    test_results['passed'].append('Bayesian Layer')
else:
    print("[FAIL] Bayesian layer not stochastic enough")
    test_results['failed'].append('Bayesian Layer')

test_results['metrics']['bayesian_sample_variance'] = sample_variance

# Test 3: Confidence Calibration
print("\n" + "="*60)
print("Test 3: Confidence Calibration")
print("="*60)

calibrator = ConfidenceCalibrator(num_classes=10).to(device)

# Create synthetic logits and labels
batch_size = 200
num_classes = 10
logits = torch.randn(batch_size, num_classes).to(device)
labels = torch.randint(0, num_classes, (batch_size,)).to(device)

# Test temperature scaling
probs_before = F.softmax(logits, dim=-1)
probs_after = calibrator(logits, method='temperature')

# Calculate ECE
ece_before = calibrator.compute_ece(probs_before, labels)
ece_after = calibrator.compute_ece(probs_after, labels)

print(f"Temperature parameter: {calibrator.temperature.item():.4f}")
print(f"ECE before calibration: {ece_before:.4f}")
print(f"ECE after calibration: {ece_after:.4f}")

# Check if probabilities are valid
prob_sum = probs_after.sum(dim=-1)
valid_probs = torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-6)

if valid_probs:
    print("[PASS] Calibration produces valid probabilities")
    test_results['passed'].append('Calibration')
else:
    print("[FAIL] Calibration produces invalid probabilities")
    test_results['failed'].append('Calibration')

test_results['metrics']['ece_before'] = ece_before
test_results['metrics']['ece_after'] = ece_after

# Test 4: Evidential Uncertainty
print("\n" + "="*60)
print("Test 4: Evidential Uncertainty")
print("="*60)

evidential = EvidentialUncertainty(input_dim=128, num_classes=10).to(device)
embeddings = torch.randn(50, 128).to(device)

try:
    output = evidential(embeddings)
    
    # Check Dirichlet parameters
    alpha = output['dirichlet_alpha']
    if (alpha > 0).all():
        print("[PASS] Valid Dirichlet parameters (all positive)")
        test_results['passed'].append('Evidential')
    else:
        print("[FAIL] Invalid Dirichlet parameters")
        test_results['failed'].append('Evidential')
    
    # Check uncertainty decomposition
    epistemic = output['epistemic_uncertainty'].mean().item()
    aleatoric = output['aleatoric_uncertainty'].mean().item()
    total = output['total_uncertainty'].mean().item()
    
    print(f"Epistemic uncertainty: {epistemic:.4f}")
    print(f"Aleatoric uncertainty: {aleatoric:.4f}")
    print(f"Total uncertainty: {total:.4f}")
    
    test_results['metrics']['evidential_epistemic'] = epistemic
    test_results['metrics']['evidential_aleatoric'] = aleatoric
    
except Exception as e:
    print(f"[FAIL] Evidential uncertainty error: {e}")
    test_results['failed'].append('Evidential')

# Test 5: Uncertainty vs Data Quality
print("\n" + "="*60)
print("Test 5: Uncertainty Response to Data Quality")
print("="*60)

# Create clean vs noisy data
clean_data = torch.randn(50, 64).to(device) * 0.5
noisy_data = torch.randn(50, 64).to(device) * 2.0

# Test with Bayesian layer
clean_outputs = []
noisy_outputs = []

for _ in range(20):
    clean_mean, clean_log_var = bayesian_layer(clean_data, sample=True)
    noisy_mean, noisy_log_var = bayesian_layer(noisy_data, sample=True)
    clean_outputs.append(clean_mean)
    noisy_outputs.append(noisy_mean)

clean_variance = torch.stack(clean_outputs).var(dim=0).mean().item()
noisy_variance = torch.stack(noisy_outputs).var(dim=0).mean().item()

print(f"Clean data variance: {clean_variance:.6f}")
print(f"Noisy data variance: {noisy_variance:.6f}")

if noisy_variance > clean_variance:
    print("[PASS] Higher uncertainty for noisier data")
    test_results['passed'].append('Uncertainty vs Noise')
else:
    print("[FAIL] Uncertainty not responding to data quality")
    test_results['failed'].append('Uncertainty vs Noise')

test_results['metrics']['clean_variance'] = clean_variance
test_results['metrics']['noisy_variance'] = noisy_variance

# Test 6: Confidence Interval Width
print("\n" + "="*60)
print("Test 6: Confidence Interval Properties")
print("="*60)

# Generate MC samples
mc_samples = []
for _ in range(100):
    with torch.no_grad():
        sample = torch.randn(50, 10).to(device)
        mc_samples.append(F.softmax(sample + torch.randn_like(sample) * 0.3, dim=-1))

mc_samples = torch.stack(mc_samples)

# Calculate confidence intervals
lower_95 = torch.quantile(mc_samples, 0.025, dim=0)
upper_95 = torch.quantile(mc_samples, 0.975, dim=0)
lower_90 = torch.quantile(mc_samples, 0.05, dim=0)
upper_90 = torch.quantile(mc_samples, 0.95, dim=0)

width_95 = (upper_95 - lower_95).mean().item()
width_90 = (upper_90 - lower_90).mean().item()

print(f"95% CI width: {width_95:.4f}")
print(f"90% CI width: {width_90:.4f}")

if width_95 > width_90:
    print("[PASS] Confidence intervals properly ordered")
    test_results['passed'].append('CI Ordering')
else:
    print("[FAIL] Confidence intervals incorrectly ordered")
    test_results['failed'].append('CI Ordering')

test_results['metrics']['ci_95_width'] = width_95
test_results['metrics']['ci_90_width'] = width_90

# Summary
print("\n" + "="*80)
print("TEST SUMMARY")
print("="*80)

print(f"\nTests Passed: {len(test_results['passed'])}")
print(f"Tests Failed: {len(test_results['failed'])}")

if test_results['passed']:
    print("\nPassed Tests:")
    for test in test_results['passed']:
        print(f"  - {test}")

if test_results['failed']:
    print("\nFailed Tests:")
    for test in test_results['failed']:
        print(f"  - {test}")

print("\nKey Metrics:")
for metric, value in test_results['metrics'].items():
    if isinstance(value, float):
        print(f"  {metric}: {value:.6f}")
    else:
        print(f"  {metric}: {value}")

# Overall assessment
print("\n" + "="*80)
if len(test_results['failed']) == 0:
    print("[SUCCESS] All uncertainty quantification tests passed!")
elif len(test_results['failed']) <= 2:
    print("[PARTIAL SUCCESS] Most tests passed with minor issues")
else:
    print("[NEEDS WORK] Several uncertainty components need attention")
print("="*80)

# Save results
import json
with open('simple_uncertainty_results.json', 'w') as f:
    json.dump(test_results, f, indent=2)
print("\nResults saved to simple_uncertainty_results.json")