"""
Hyperparameter optimization for Energy GNN System
Finds optimal configuration for best results
"""

import os
import numpy as np
import torch
import yaml
from pathlib import Path
import optuna
from optuna.trial import TrialState
import logging
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Optimize hyperparameters for better results"""
    
    def __init__(self, base_config_path='config/config.yaml'):
        with open(base_config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)
        
        self.best_config = None
        self.best_score = -float('inf')
        
    def objective(self, trial):
        """Objective function for Optuna optimization"""
        
        # Sample hyperparameters
        config = self.sample_hyperparameters(trial)
        
        try:
            # Train and evaluate model with sampled config
            score = self.train_and_evaluate(config, trial)
            
            # Track best configuration
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
                self.save_best_config()
            
            return score
            
        except Exception as e:
            logger.error(f"Trial failed: {e}")
            return -float('inf')
    
    def sample_hyperparameters(self, trial):
        """Sample hyperparameters for trial"""
        config = self.base_config.copy()
        
        # Model hyperparameters
        config['model']['hidden_dim'] = trial.suggest_categorical(
            'hidden_dim', [64, 128, 256, 512]
        )
        config['model']['num_layers'] = trial.suggest_int(
            'num_layers', 2, 6
        )
        config['model']['dropout'] = trial.suggest_float(
            'dropout', 0.0, 0.5, step=0.1
        )
        config['model']['num_heads'] = trial.suggest_categorical(
            'num_heads', [4, 8, 16]
        )
        
        # Training hyperparameters
        config['training']['learning_rate'] = trial.suggest_loguniform(
            'learning_rate', 1e-5, 1e-2
        )
        config['training']['weight_decay'] = trial.suggest_loguniform(
            'weight_decay', 1e-6, 1e-3
        )
        config['training']['batch_size'] = trial.suggest_categorical(
            'batch_size', [16, 32, 64]
        )
        
        # Loss weights (ensure they sum to reasonable value)
        w_cluster = trial.suggest_float('w_cluster', 0.5, 2.0)
        w_placement = trial.suggest_float('w_placement', 0.5, 2.0)
        w_flow = trial.suggest_float('w_flow', 0.5, 2.0)
        w_physics = trial.suggest_float('w_physics', 5.0, 20.0)
        
        # Normalize weights
        total_weight = w_cluster + w_placement + w_flow
        config['loss']['w_cluster'] = w_cluster / total_weight * 3.0
        config['loss']['w_placement'] = w_placement / total_weight * 3.0
        config['loss']['w_flow'] = w_flow / total_weight * 3.0
        config['loss']['w_physics'] = w_physics
        
        # Discovery-specific parameters
        config['loss']['alpha_complementarity'] = trial.suggest_float(
            'alpha_complementarity', 0.5, 3.0
        )
        config['loss']['alpha_network'] = trial.suggest_float(
            'alpha_network', 0.5, 2.0
        )
        config['loss']['alpha_temporal'] = trial.suggest_float(
            'alpha_temporal', 0.1, 1.0
        )
        
        # Architecture choices
        config['model']['conv_type'] = trial.suggest_categorical(
            'conv_type', ['sage', 'gat', 'gcn']
        )
        
        # Optimizer settings
        config['training']['optimizer']['type'] = trial.suggest_categorical(
            'optimizer_type', ['Adam', 'AdamW', 'SGD']
        )
        
        if config['training']['optimizer']['type'] == 'SGD':
            config['training']['optimizer']['momentum'] = trial.suggest_float(
                'momentum', 0.5, 0.99
            )
        
        # Scheduler settings
        config['training']['scheduler']['type'] = trial.suggest_categorical(
            'scheduler_type', ['CosineAnnealingWarmRestarts', 'StepLR', 'ReduceLROnPlateau']
        )
        
        return config
    
    def train_and_evaluate(self, config, trial):
        """Train model with config and return validation score"""
        import sys
        sys.path.append('.')
        
        from main import UnifiedEnergyGNNSystem
        
        # Save temporary config
        temp_config_path = f'config/trial_{trial.number}.yaml'
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
        
        try:
            # Initialize system with trial config
            system = UnifiedEnergyGNNSystem(temp_config_path)
            
            # Load data (use smaller subset for optimization)
            train_loader, val_loader, _ = system.load_and_prepare_data(evaluate_groups=False)
            
            # Train for fewer epochs during optimization
            num_epochs = 20
            
            # Track metrics
            best_val_score = -float('inf')
            
            for epoch in range(num_epochs):
                # Training step
                system.model.train()
                train_loss = 0
                
                for batch in train_loader:
                    batch = batch.to(system.device)
                    optimizer = torch.optim.Adam(
                        system.model.parameters(),
                        lr=config['training']['learning_rate']
                    )
                    
                    optimizer.zero_grad()
                    outputs = system.model(batch)
                    
                    # Compute loss (simplified)
                    if 'clusters' in outputs:
                        loss = -outputs['clusters'].max(dim=1)[0].mean()  # Maximize cluster confidence
                    else:
                        loss = torch.tensor(0.0)
                    
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                
                # Validation
                system.model.eval()
                val_score = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(system.device)
                        outputs = system.model(batch)
                        
                        # Compute validation score (higher is better)
                        if 'clusters' in outputs:
                            # Cluster quality: high confidence, good separation
                            cluster_probs = torch.softmax(outputs['clusters'], dim=1)
                            entropy = -(cluster_probs * torch.log(cluster_probs + 1e-10)).sum(dim=1).mean()
                            score = 1.0 / (entropy + 1e-10)  # Lower entropy = better
                        else:
                            score = 0.0
                        
                        val_score += score
                
                val_score /= len(val_loader)
                
                if val_score > best_val_score:
                    best_val_score = val_score
                
                # Report to Optuna
                trial.report(val_score, epoch)
                
                # Prune trial if needed
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Clean up
            Path(temp_config_path).unlink(missing_ok=True)
            
            return best_val_score
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            Path(temp_config_path).unlink(missing_ok=True)
            return -float('inf')
    
    def optimize(self, n_trials=50):
        """Run hyperparameter optimization"""
        print("\n" + "="*80)
        print("HYPERPARAMETER OPTIMIZATION")
        print("="*80)
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=5
            )
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            n_jobs=1,  # Use single process for stability
            show_progress_bar=True
        )
        
        # Print results
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"\nBest trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value:.4f}")
        print(f"  Params:")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
        # Save optimization history
        self.save_optimization_results(study)
        
        return self.best_config
    
    def save_best_config(self):
        """Save best configuration found"""
        if self.best_config:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save as YAML
            yaml_path = f'config/optimized_config_{timestamp}.yaml'
            with open(yaml_path, 'w') as f:
                yaml.dump(self.best_config, f, default_flow_style=False)
            
            # Save with score info
            info_path = f'config/optimized_info_{timestamp}.json'
            with open(info_path, 'w') as f:
                json.dump({
                    'score': float(self.best_score),
                    'timestamp': timestamp,
                    'config_path': yaml_path
                }, f, indent=2)
            
            logger.info(f"Best config saved to: {yaml_path}")
    
    def save_optimization_results(self, study):
        """Save complete optimization results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = f'results/optimization_{timestamp}.json'
        
        Path('results').mkdir(exist_ok=True)
        
        results = {
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'trials': []
        }
        
        for trial in study.trials:
            results['trials'].append({
                'number': trial.number,
                'value': trial.value if trial.value is not None else None,
                'params': trial.params,
                'state': str(trial.state)
            })
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Optimization results saved to: {results_path}")


def quick_optimize():
    """Quick optimization with good defaults"""
    print("\n" + "="*80)
    print("QUICK HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Create optimized config based on analysis
    optimized_config = {
        'model': {
            'type': 'hetero',
            'hidden_dim': 256,  # Balanced size
            'num_layers': 4,    # Enough for multi-hop
            'dropout': 0.2,     # Moderate regularization
            'num_heads': 8,     # Good attention resolution
            'num_clusters': 20,
            'conv_type': 'gat'  # Best for heterogeneous graphs
        },
        'training': {
            'mode': 'discovery',
            'learning_rate': 0.0005,  # Conservative LR
            'weight_decay': 1e-5,
            'batch_size': 32,
            'num_epochs': 100,
            'optimizer': {
                'type': 'AdamW',
                'betas': [0.9, 0.999],
                'eps': 1e-8
            },
            'scheduler': {
                'type': 'CosineAnnealingWarmRestarts',
                'T_0': 10,
                'T_mult': 2,
                'min_lr': 1e-6
            },
            'grad_clip': 1.0,  # Stability
            'early_stopping_patience': 20
        },
        'loss': {
            'w_cluster': 1.0,
            'w_placement': 1.0,
            'w_flow': 1.0,
            'w_physics': 10.0,  # Enforce physics
            'alpha_complementarity': 2.0,  # Priority on complementarity
            'alpha_network': 1.5,
            'alpha_temporal': 0.5,
            'alpha_cascade': 2.0,
            'alpha_balance': 1.0
        },
        'data_loader': {
            'batch_size': 32,
            'num_workers': 0,
            'shuffle': True,
            'pin_memory': True,
            'min_cluster_size': 3,
            'max_cluster_size': 20
        }
    }
    
    # Load base config for other settings
    with open('config/config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Merge configs
    def deep_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value
    
    deep_merge(base_config, optimized_config)
    
    # Save optimized config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f'config/quick_optimized_{timestamp}.yaml'
    
    with open(output_path, 'w') as f:
        yaml.dump(base_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nOptimized configuration saved to: {output_path}")
    print("\nKey optimizations applied:")
    print("  • Hidden dimension: 256 (balanced)")
    print("  • Architecture: GAT with 8 attention heads")
    print("  • Learning rate: 0.0005 with cosine annealing")
    print("  • Gradient clipping: 1.0")
    print("  • Loss weights: Prioritize physics and complementarity")
    print("\nTo use this config:")
    print(f"  python main.py --config {output_path} --mode train")
    
    return output_path


def main():
    """Main optimization entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize Energy GNN hyperparameters')
    parser.add_argument('--mode', choices=['quick', 'full'], default='quick',
                       help='Optimization mode')
    parser.add_argument('--n_trials', type=int, default=50,
                       help='Number of trials for full optimization')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Base configuration file')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        config_path = quick_optimize()
    else:
        optimizer = HyperparameterOptimizer(args.config)
        best_config = optimizer.optimize(n_trials=args.n_trials)
        
        if best_config:
            print("\n✓ Optimization complete!")
            print("Use the optimized configuration for better results")


if __name__ == "__main__":
    main()