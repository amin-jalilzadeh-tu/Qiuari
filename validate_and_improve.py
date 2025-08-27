"""
Comprehensive validation and improvement script for Energy GNN System
Run this to identify issues and automatically apply fixes for better results
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemValidator:
    """Validates and improves the Energy GNN system"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.config_path = config_path
        self.issues = []
        self.fixes_applied = []
        self.performance_metrics = {}
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def run_full_validation(self):
        """Run comprehensive validation and apply fixes"""
        print("\n" + "="*80)
        print("ENERGY GNN SYSTEM VALIDATION & IMPROVEMENT")
        print("="*80)
        
        # 1. Validate configuration
        print("\n[1/7] Validating Configuration...")
        self.validate_configuration()
        
        # 2. Check data pipeline
        print("\n[2/7] Checking Data Pipeline...")
        self.validate_data_pipeline()
        
        # 3. Validate model architecture
        print("\n[3/7] Validating Model Architecture...")
        self.validate_model_architecture()
        
        # 4. Check training stability
        print("\n[4/7] Testing Training Stability...")
        self.validate_training_stability()
        
        # 5. Validate loss functions
        print("\n[5/7] Validating Loss Functions...")
        self.validate_loss_functions()
        
        # 6. Check output quality
        print("\n[6/7] Checking Output Quality...")
        self.validate_output_quality()
        
        # 7. Apply automatic fixes
        print("\n[7/7] Applying Automatic Fixes...")
        self.apply_fixes()
        
        # Generate report
        self.generate_validation_report()
        
        return len(self.issues) == 0
    
    def validate_configuration(self):
        """Validate and fix configuration issues"""
        issues_found = []
        
        # Check for null/missing critical values
        critical_params = [
            ('model.input_dim', 'auto'),
            ('model.building_features', 'auto'),
            ('model.hidden_dim', 256),
            ('training.learning_rate', 0.001),
            ('loss.w_cluster', 1.0)
        ]
        
        for param_path, default_value in critical_params:
            keys = param_path.split('.')
            value = self.config
            for key in keys:
                if key not in value:
                    issues_found.append(f"Missing parameter: {param_path}")
                    # Auto-fix by adding default
                    self._set_nested_config(param_path, default_value)
                    break
                value = value[key]
                if value is None or (isinstance(value, str) and value.lower() in ['null', 'none']):
                    issues_found.append(f"Null parameter: {param_path}")
                    self._set_nested_config(param_path, default_value)
                    break
        
        # Check loss weights sum to reasonable value
        loss_weights = self.config.get('loss', {})
        weight_sum = sum([v for k, v in loss_weights.items() if 'weight' in k and isinstance(v, (int, float))])
        if weight_sum > 100:
            issues_found.append(f"Loss weights sum too high: {weight_sum}")
            # Normalize weights
            for key in loss_weights:
                if 'weight' in key and isinstance(loss_weights[key], (int, float)):
                    loss_weights[key] = loss_weights[key] / weight_sum * 10
            self.fixes_applied.append("Normalized loss weights")
        
        # Check batch size consistency
        batch_sizes = []
        if 'data_loader' in self.config and 'batch_size' in self.config['data_loader']:
            batch_sizes.append(self.config['data_loader']['batch_size'])
        if 'training' in self.config and 'batch_size' in self.config['training']:
            batch_sizes.append(self.config['training']['batch_size'])
        
        if len(set(batch_sizes)) > 1:
            issues_found.append(f"Inconsistent batch sizes: {batch_sizes}")
            # Use minimum batch size for safety
            min_batch = min(batch_sizes)
            self.config['data_loader']['batch_size'] = min_batch
            self.config['training']['batch_size'] = min_batch
            self.fixes_applied.append(f"Unified batch size to {min_batch}")
        
        if issues_found:
            self.issues.extend(issues_found)
            logger.warning(f"Found {len(issues_found)} configuration issues")
        else:
            logger.info("✓ Configuration validated successfully")
    
    def validate_data_pipeline(self):
        """Check data loading and processing"""
        try:
            from data.data_loader import EnergyDataLoader
            from data.kg_connector import KGConnector
            
            # Test data loader initialization
            loader = EnergyDataLoader(self.config['data_loader'], mode='train')
            
            # Check if we can connect to Neo4j
            if self.config['data']['use_kg']:
                try:
                    kg = KGConnector(
                        uri=self.config['kg']['uri'],
                        user=self.config['kg']['user'],
                        password=self.config['kg']['password']
                    )
                    # Test query
                    result = kg.run_query("MATCH (n) RETURN count(n) as count LIMIT 1")
                    if result:
                        logger.info(f"✓ Neo4j connection successful")
                    kg.close()
                except Exception as e:
                    self.issues.append(f"Neo4j connection failed: {e}")
                    logger.warning("Neo4j unavailable, will use synthetic data")
            
            # Test data creation
            test_data = self._create_test_data()
            if test_data is not None:
                logger.info("✓ Data pipeline validated")
            else:
                self.issues.append("Failed to create test data")
                
        except Exception as e:
            self.issues.append(f"Data pipeline error: {e}")
            logger.error(f"Data pipeline validation failed: {e}")
    
    def validate_model_architecture(self):
        """Validate model can be created and run forward pass"""
        try:
            from models.base_gnn import create_gnn_model
            
            # Create model
            model = create_gnn_model(
                self.config['model']['type'],
                self.config['model']
            )
            
            # Test forward pass with dummy data
            test_data = self._create_test_data()
            if test_data is not None:
                model.eval()
                with torch.no_grad():
                    output = model(test_data)
                    
                    # Check output validity
                    if isinstance(output, dict):
                        for key, value in output.items():
                            if torch.is_tensor(value):
                                if torch.isnan(value).any():
                                    self.issues.append(f"NaN in model output: {key}")
                                if torch.isinf(value).any():
                                    self.issues.append(f"Inf in model output: {key}")
                    
                    logger.info("✓ Model architecture validated")
            
            # Check model size
            total_params = sum(p.numel() for p in model.parameters())
            if total_params > 100_000_000:  # 100M parameters
                self.issues.append(f"Model too large: {total_params:,} parameters")
                logger.warning("Model may be over-parameterized")
            
        except Exception as e:
            self.issues.append(f"Model creation failed: {e}")
            logger.error(f"Model validation failed: {e}")
    
    def validate_training_stability(self):
        """Test if training is numerically stable"""
        try:
            from models.base_gnn import create_gnn_model
            from training.loss_functions import DiscoveryLoss
            
            # Create model and loss
            model = create_gnn_model(
                self.config['model']['type'],
                self.config['model']
            )
            loss_fn = DiscoveryLoss(self.config['loss'])
            
            # Create optimizer
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=self.config['training']['learning_rate']
            )
            
            # Run a few training steps
            model.train()
            test_data = self._create_test_data()
            
            losses = []
            for i in range(10):
                optimizer.zero_grad()
                output = model(test_data)
                
                # Create mock targets
                targets = self._create_mock_targets(test_data, output)
                
                # Compute loss
                loss, loss_dict = loss_fn(output, targets)
                
                if torch.isnan(loss):
                    self.issues.append(f"NaN loss at step {i}")
                    break
                if torch.isinf(loss):
                    self.issues.append(f"Inf loss at step {i}")
                    break
                    
                losses.append(loss.item())
                
                # Check gradient
                loss.backward()
                
                # Check for gradient explosion
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                if total_norm > 100:
                    self.issues.append(f"Gradient explosion: norm={total_norm:.2f}")
                    # Apply gradient clipping fix
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    self.fixes_applied.append("Applied gradient clipping")
                
                optimizer.step()
            
            # Check if loss is decreasing
            if len(losses) > 5:
                if losses[-1] > losses[0]:
                    self.issues.append("Loss not decreasing")
                    logger.warning("Training may be unstable")
                else:
                    logger.info(f"✓ Training stability verified (loss: {losses[0]:.4f} -> {losses[-1]:.4f})")
            
        except Exception as e:
            self.issues.append(f"Training validation failed: {e}")
            logger.error(f"Training stability check failed: {e}")
    
    def validate_loss_functions(self):
        """Validate loss functions are numerically stable"""
        try:
            from training.loss_functions import DiscoveryLoss, UnifiedEnergyLoss
            
            # Test DiscoveryLoss
            discovery_loss = DiscoveryLoss(self.config['loss'])
            
            # Create test inputs with edge cases
            test_cases = [
                # Normal case
                {'clusters': torch.randn(10, 5), 'complementarity': torch.randn(10, 10)},
                # Small values
                {'clusters': torch.randn(10, 5) * 1e-6, 'complementarity': torch.randn(10, 10) * 1e-6},
                # Large values  
                {'clusters': torch.randn(10, 5) * 1e6, 'complementarity': torch.randn(10, 10) * 1e6},
                # Mixed
                {'clusters': torch.cat([torch.zeros(5, 5), torch.ones(5, 5)]), 
                 'complementarity': torch.eye(10)}
            ]
            
            for i, predictions in enumerate(test_cases):
                targets = {'cluster_labels': torch.randint(0, 5, (10,))}
                
                try:
                    loss, components = discovery_loss(predictions, targets)
                    
                    if torch.isnan(loss):
                        self.issues.append(f"NaN loss in test case {i}")
                    if torch.isinf(loss):
                        self.issues.append(f"Inf loss in test case {i}")
                    if loss < 0:
                        self.issues.append(f"Negative loss in test case {i}")
                        
                except Exception as e:
                    self.issues.append(f"Loss computation failed for case {i}: {e}")
            
            if not any("loss" in issue.lower() for issue in self.issues[-4:]):
                logger.info("✓ Loss functions validated")
                
        except Exception as e:
            self.issues.append(f"Loss validation failed: {e}")
            logger.error(f"Loss function validation failed: {e}")
    
    def validate_output_quality(self):
        """Check if outputs and reports can be generated"""
        try:
            from analysis.comprehensive_reporter import ComprehensiveReporter
            from analysis.pattern_analyzer import PatternAnalyzer
            
            # Test reporter initialization
            reporter = ComprehensiveReporter(self.config)
            analyzer = PatternAnalyzer(self.config.get('analysis', {}))
            
            # Create mock data
            mock_clusters = np.random.randint(0, 5, 20)
            mock_building_data = {
                'building_ids': list(range(20)),
                'lv_group': 'TEST_LV_001',
                'consumption_profiles': np.random.rand(20, 96),
                'generation_profiles': np.random.rand(20, 96)
            }
            mock_gnn_outputs = {
                'cluster_assignments': mock_clusters,
                'cluster_probs': np.random.rand(20, 5)
            }
            mock_intervention_plan = {}
            
            # Try to generate report
            try:
                reports = reporter.generate_full_report(
                    mock_clusters,
                    mock_building_data,
                    mock_gnn_outputs,
                    mock_intervention_plan,
                    save_dir="reports/test"
                )
                
                if reports and len(reports) > 0:
                    logger.info("✓ Report generation validated")
                else:
                    self.issues.append("Empty report generated")
                    
            except Exception as e:
                self.issues.append(f"Report generation failed: {e}")
                
        except Exception as e:
            self.issues.append(f"Output validation failed: {e}")
            logger.error(f"Output quality validation failed: {e}")
    
    def apply_fixes(self):
        """Apply automatic fixes for common issues"""
        fixes_count = 0
        
        # Fix 1: Ensure output directories exist
        dirs_to_create = [
            'checkpoints', 'results', 'results/analysis', 
            'results/interventions', 'results/comparisons',
            'results/visualizations', 'reports', 'logs'
        ]
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        fixes_count += 1
        
        # Fix 2: Update config with validated values
        if self.fixes_applied:
            self.save_fixed_config()
            fixes_count += 1
        
        # Fix 3: Create default feature mapper if missing
        feature_mapper_path = Path('utils/feature_mapping.py')
        if not feature_mapper_path.exists():
            self._create_default_feature_mapper()
            fixes_count += 1
        
        # Fix 4: Set environment variables for stability
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        os.environ['PYTHONHASHSEED'] = '0'
        fixes_count += 1
        
        logger.info(f"✓ Applied {fixes_count} automatic fixes")
        logger.info(f"  Specific fixes: {', '.join(self.fixes_applied)}")
    
    def generate_validation_report(self):
        """Generate comprehensive validation report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = Path(f"reports/validation_report_{timestamp}.txt")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("ENERGY GNN SYSTEM VALIDATION REPORT\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write("="*80 + "\n\n")
            
            # Summary
            f.write("SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Issues Found: {len(self.issues)}\n")
            f.write(f"Fixes Applied: {len(self.fixes_applied)}\n")
            f.write(f"System Status: {'✓ READY' if len(self.issues) == 0 else '⚠ NEEDS ATTENTION'}\n\n")
            
            # Issues
            if self.issues:
                f.write("ISSUES FOUND\n")
                f.write("-"*40 + "\n")
                for i, issue in enumerate(self.issues, 1):
                    f.write(f"{i}. {issue}\n")
                f.write("\n")
            
            # Fixes
            if self.fixes_applied:
                f.write("FIXES APPLIED\n")
                f.write("-"*40 + "\n")
                for i, fix in enumerate(self.fixes_applied, 1):
                    f.write(f"{i}. {fix}\n")
                f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            recommendations = self._generate_recommendations()
            for i, rec in enumerate(recommendations, 1):
                f.write(f"{i}. {rec}\n")
            
        logger.info(f"Validation report saved to: {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("VALIDATION COMPLETE")
        print("="*80)
        print(f"Issues Found: {len(self.issues)}")
        print(f"Fixes Applied: {len(self.fixes_applied)}")
        print(f"Report Location: {report_path}")
        
        if len(self.issues) == 0:
            print("\n✓ System is ready for training!")
            print("Run: python main.py --mode train")
        else:
            print(f"\n⚠ {len(self.issues)} issues need attention")
            print("See validation report for details")
    
    def _create_test_data(self):
        """Create test data for validation"""
        from torch_geometric.data import Data
        
        try:
            num_nodes = 20
            num_edges = 60
            
            x = torch.randn(num_nodes, self.config['model'].get('input_dim', 17))
            edge_index = torch.randint(0, num_nodes, (2, num_edges))
            
            data = Data(x=x, edge_index=edge_index)
            return data
        except Exception as e:
            logger.error(f"Failed to create test data: {e}")
            return None
    
    def _create_mock_targets(self, data, output):
        """Create mock targets for loss computation"""
        num_nodes = data.x.shape[0]
        
        targets = {
            'cluster_labels': torch.randint(0, 10, (num_nodes,)),
            'complementarity_matrix': torch.rand(num_nodes, num_nodes),
            'energy_balance': torch.zeros(num_nodes)
        }
        
        return targets
    
    def _set_nested_config(self, path, value):
        """Set nested configuration value"""
        keys = path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
        self.fixes_applied.append(f"Set {path} = {value}")
    
    def save_fixed_config(self):
        """Save fixed configuration"""
        backup_path = Path(self.config_path).with_suffix('.backup.yaml')
        
        # Backup original
        import shutil
        shutil.copy(self.config_path, backup_path)
        
        # Save fixed config
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Configuration updated (backup: {backup_path})")
    
    def _create_default_feature_mapper(self):
        """Create default feature mapper if missing"""
        code = '''"""Default feature mapper for energy data"""
import numpy as np
import torch

class FeatureMapper:
    def get_feature_vector(self, data):
        """Convert data to feature vector"""
        if isinstance(data, dict):
            features = []
            for key in sorted(data.keys()):
                if isinstance(data[key], (int, float)):
                    features.append(data[key])
            return np.array(features)
        return np.array(data)
    
    def get_feature_dim(self):
        """Get feature dimension"""
        return 17  # Default for energy features

feature_mapper = FeatureMapper()
'''
        
        Path('utils').mkdir(exist_ok=True)
        with open('utils/feature_mapping.py', 'w') as f:
            f.write(code)
        
        self.fixes_applied.append("Created default feature mapper")
    
    def _generate_recommendations(self):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Based on issues found
        if any('NaN' in issue for issue in self.issues):
            recommendations.append("Enable gradient clipping: set training.grad_clip = 1.0")
            
        if any('loss' in issue.lower() for issue in self.issues):
            recommendations.append("Reduce learning rate: try training.learning_rate = 0.0001")
            
        if any('Neo4j' in issue for issue in self.issues):
            recommendations.append("Use synthetic data: set data.use_kg = false")
            
        if any('memory' in issue.lower() for issue in self.issues):
            recommendations.append("Reduce batch size: set data_loader.batch_size = 16")
        
        # General recommendations
        if not recommendations:
            recommendations.append("Start with smaller dataset for testing")
            recommendations.append("Monitor training loss carefully for first 10 epochs")
            recommendations.append("Use tensorboard for visualization: tensorboard --logdir=runs")
        
        return recommendations


def main():
    """Main validation entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate and improve Energy GNN system')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--fix', action='store_true',
                       help='Automatically apply fixes')
    
    args = parser.parse_args()
    
    # Run validation
    validator = SystemValidator(args.config)
    success = validator.run_full_validation()
    
    # Return exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()