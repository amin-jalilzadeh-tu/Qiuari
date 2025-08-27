"""
Neural Network Training Analysis and Optimization Script
Analyzes current training setup and identifies bottlenecks
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import yaml
import json
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration"""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def analyze_model_architecture():
    """Analyze model architecture and parameter counts"""
    from models.base_gnn import HeteroEnergyGNN
    
    config = load_config()
    
    # Create dummy model to analyze
    model = HeteroEnergyGNN(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Analyze layers
    layer_info = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTM, nn.GRU)):
            params = sum(p.numel() for p in module.parameters())
            layer_info[name] = {
                'type': module.__class__.__name__,
                'params': params,
                'trainable': all(p.requires_grad for p in module.parameters())
            }
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layers': layer_info
    }

def analyze_loss_weights():
    """Analyze loss weight configuration"""
    config = load_config()
    
    # Extract all loss weights
    loss_weights = {}
    
    # Training loss weights
    if 'training' in config and 'loss_weights' in config['training']:
        loss_weights['training'] = config['training']['loss_weights']
    
    # Loss section weights
    if 'loss' in config:
        loss_weights['loss_config'] = config['loss']
    
    # Calculate relative importance
    if 'training' in loss_weights:
        training_weights = loss_weights['training']
        total = sum(training_weights.values())
        loss_weights['normalized'] = {k: v/total for k, v in training_weights.items()}
    
    return loss_weights

def analyze_learning_dynamics():
    """Analyze learning rate scheduling and optimization"""
    config = load_config()
    
    training_config = config.get('training', {})
    
    analysis = {
        'optimizer': training_config.get('optimizer', {}),
        'scheduler': training_config.get('scheduler', {}),
        'batch_size': training_config.get('batch_size', 32),
        'num_epochs': training_config.get('num_epochs', 100),
        'learning_rate': training_config.get('learning_rate', 0.001),
        'weight_decay': training_config.get('weight_decay', 1e-5),
        'grad_clip': training_config.get('grad_clip', 1.0),
        'early_stopping_patience': training_config.get('early_stopping_patience', 20)
    }
    
    # Analyze scheduler behavior
    if analysis['scheduler'].get('type') == 'CosineAnnealingWarmRestarts':
        T_0 = analysis['scheduler'].get('T_0', 10)
        T_mult = analysis['scheduler'].get('T_mult', 2)
        min_lr = analysis['scheduler'].get('min_lr', 1e-6)
        
        # Calculate restart points
        restarts = []
        current = T_0
        for i in range(5):  # First 5 restarts
            restarts.append(current)
            current = current + T_0 * (T_mult ** (i+1))
            if current > analysis['num_epochs']:
                break
        
        analysis['scheduler_restarts'] = restarts
    
    return analysis

def run_mini_training_test():
    """Run a mini training session to test convergence"""
    logger.info("Running mini training test...")
    
    from models.base_gnn import HeteroEnergyGNN
    from training.loss_functions import UnifiedEnergyLoss
    
    config = load_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model and loss
    model = HeteroEnergyGNN(config).to(device)
    loss_fn = UnifiedEnergyLoss(config.get('loss', {}))
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create dummy data
    batch_size = 32
    num_nodes = 200
    num_features = 10
    num_edges = 500
    
    # Training metrics
    loss_history = []
    grad_norms = []
    
    # Run mini training
    model.train()
    for epoch in range(10):
        # Create random batch
        x = torch.randn(num_nodes, num_features).to(device)
        edge_index = torch.randint(0, num_nodes, (2, num_edges)).to(device)
        
        # Create batch
        from torch_geometric.data import Data
        batch = Data(x=x, edge_index=edge_index)
        batch.batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        try:
            outputs = model(batch)
            
            # Create dummy targets
            targets = {
                'temporal_profiles': torch.randn(num_nodes, 24).to(device),
                'demand': torch.randn(num_nodes, 24).to(device),
                'generation': torch.randn(num_nodes, 24).to(device),
                'individual_peaks': torch.randn(num_nodes).to(device)
            }
            
            # Calculate loss
            loss, loss_components = loss_fn(
                predictions=outputs,
                targets=targets,
                graph_data={'adjacency': edge_index}
            )
            
            # Backward pass
            loss.backward()
            
            # Calculate gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1./2.)
            grad_norms.append(total_norm)
            
            # Gradient clipping
            if config['training'].get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config['training']['grad_clip']
                )
            
            optimizer.step()
            
            loss_history.append({
                'total': loss.item(),
                'components': {k: v.item() if torch.is_tensor(v) else v 
                             for k, v in loss_components.items()}
            })
            
            logger.info(f"Epoch {epoch}: Loss = {loss.item():.4f}, Grad Norm = {total_norm:.4f}")
            
        except Exception as e:
            logger.error(f"Error in training: {e}")
            return {'error': str(e)}
    
    return {
        'loss_history': loss_history,
        'grad_norms': grad_norms,
        'final_loss': loss_history[-1]['total'] if loss_history else None,
        'convergence_rate': (loss_history[0]['total'] - loss_history[-1]['total']) / loss_history[0]['total'] if loss_history else 0
    }

def analyze_data_pipeline():
    """Analyze data loading and preprocessing"""
    config = load_config()
    
    data_config = config.get('data_loader', {})
    
    analysis = {
        'batch_size': data_config.get('batch_size', 32),
        'num_workers': data_config.get('num_workers', 0),
        'pin_memory': data_config.get('pin_memory', True),
        'shuffle': data_config.get('shuffle', True),
        'min_cluster_size': data_config.get('min_cluster_size', 3),
        'max_cluster_size': data_config.get('max_cluster_size', 20)
    }
    
    # Check for data availability
    data_path = Path('data/processed')
    if data_path.exists():
        files = list(data_path.glob('*.pt'))
        analysis['available_data'] = [f.name for f in files]
    else:
        analysis['available_data'] = []
    
    return analysis

def identify_training_issues():
    """Identify potential training issues"""
    issues = []
    recommendations = []
    
    config = load_config()
    
    # Check learning rate
    lr = config['training']['learning_rate']
    if lr > 0.01:
        issues.append("Learning rate too high - may cause instability")
        recommendations.append("Reduce learning rate to 0.001-0.005 range")
    elif lr < 1e-5:
        issues.append("Learning rate too low - slow convergence")
        recommendations.append("Increase learning rate to 1e-4 to 1e-3 range")
    
    # Check batch size
    batch_size = config['training']['batch_size']
    if batch_size < 16:
        issues.append("Small batch size - high gradient variance")
        recommendations.append("Increase batch size to 32-64 for stability")
    elif batch_size > 128:
        issues.append("Large batch size - may reduce generalization")
        recommendations.append("Consider batch size of 32-64")
    
    # Check loss weights
    loss_weights = config['training'].get('loss_weights', {})
    if 'physics' in loss_weights and loss_weights['physics'] > 5:
        issues.append("Physics loss weight too high - may dominate training")
        recommendations.append("Balance physics weight with other objectives (1-2 range)")
    
    # Check gradient clipping
    if config['training'].get('grad_clip', 0) == 0:
        issues.append("No gradient clipping - risk of gradient explosion")
        recommendations.append("Enable gradient clipping (1.0-5.0)")
    
    # Check early stopping
    patience = config['training'].get('early_stopping_patience', 20)
    if patience < 10:
        issues.append("Early stopping too aggressive")
        recommendations.append("Increase patience to 15-30 epochs")
    
    # Check scheduler
    scheduler = config['training'].get('scheduler', {})
    if scheduler.get('type') == 'CosineAnnealingWarmRestarts':
        T_0 = scheduler.get('T_0', 10)
        if T_0 < 5:
            issues.append("Scheduler restarts too frequent")
            recommendations.append("Increase T_0 to 10-20 epochs")
    
    return issues, recommendations

def generate_optimization_report():
    """Generate comprehensive optimization report"""
    logger.info("Generating optimization report...")
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_analysis': analyze_model_architecture(),
        'loss_weights': analyze_loss_weights(),
        'learning_dynamics': analyze_learning_dynamics(),
        'data_pipeline': analyze_data_pipeline(),
        'mini_training_test': run_mini_training_test()
    }
    
    # Identify issues
    issues, recommendations = identify_training_issues()
    report['issues'] = issues
    report['recommendations'] = recommendations
    
    # Calculate optimization scores
    scores = {
        'model_efficiency': 0.0,
        'loss_balance': 0.0,
        'convergence_quality': 0.0,
        'data_efficiency': 0.0
    }
    
    # Model efficiency (based on parameter count)
    if report['model_analysis']['total_params'] < 1e6:
        scores['model_efficiency'] = 0.9
    elif report['model_analysis']['total_params'] < 10e6:
        scores['model_efficiency'] = 0.7
    else:
        scores['model_efficiency'] = 0.4
    
    # Loss balance (based on weight distribution)
    if 'normalized' in report['loss_weights']:
        weights = list(report['loss_weights']['normalized'].values())
        std = np.std(weights)
        scores['loss_balance'] = max(0, 1 - std * 2)
    
    # Convergence quality
    if 'convergence_rate' in report['mini_training_test']:
        scores['convergence_quality'] = min(1.0, report['mini_training_test']['convergence_rate'] * 2)
    
    # Data efficiency
    if report['data_pipeline']['num_workers'] > 0:
        scores['data_efficiency'] += 0.3
    if report['data_pipeline']['pin_memory']:
        scores['data_efficiency'] += 0.2
    if 16 <= report['data_pipeline']['batch_size'] <= 64:
        scores['data_efficiency'] += 0.5
    
    report['optimization_scores'] = scores
    report['overall_score'] = np.mean(list(scores.values()))
    
    return report

def save_report(report):
    """Save optimization report"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON report
    report_path = Path('reports') / f'training_optimization_{timestamp}.json'
    report_path.parent.mkdir(exist_ok=True)
    
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Report saved to {report_path}")
    
    return report_path

def main():
    """Main analysis function"""
    logger.info("Starting Neural Network Training Analysis...")
    
    # Generate report
    report = generate_optimization_report()
    
    # Save report
    report_path = save_report(report)
    
    # Print summary
    print("\n" + "="*50)
    print("NEURAL NETWORK TRAINING ANALYSIS SUMMARY")
    print("="*50)
    
    print(f"\nModel Parameters: {report['model_analysis']['total_params']:,}")
    print(f"Trainable Parameters: {report['model_analysis']['trainable_params']:,}")
    
    print("\nOptimization Scores:")
    for key, value in report['optimization_scores'].items():
        print(f"  {key}: {value:.2%}")
    print(f"  Overall: {report['overall_score']:.2%}")
    
    print("\nIdentified Issues:")
    for issue in report['issues']:
        print(f"  ⚠️  {issue}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  ✓ {rec}")
    
    print(f"\nFull report saved to: {report_path}")
    
    return report

if __name__ == "__main__":
    report = main()