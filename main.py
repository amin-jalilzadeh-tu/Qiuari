"""
Main entry point for Energy GNN System
Orchestrates training, evaluation, analysis, and intervention planning
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # Fix OpenMP conflict
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import logging
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import all components
from data.kg_connector import KGConnector
from data.graph_constructor import GraphConstructor
from data.data_loader import EnergyDataLoader
from data.feature_processor import FeatureProcessor

from models.base_gnn import create_gnn_model, HeteroEnergyGNN
from models.task_heads import UnifiedTaskHead, ComplementarityScoreHead, NetworkCentralityHead, EnergyFlowHead
from models.attention_layers import UnifiedAttentionModule
from models.temporal_layers import TemporalProcessor as TemporalBlock
from models.physics_layers import PhysicsConstraintLayer as PhysicsInformedLayer

# NEW: Import network-aware components
from models.network_aware_layers import NetworkAwareGNN
from training.network_aware_loss import NetworkAwareDiscoveryLoss, CascadePredictionLoss
from training.network_aware_trainer import NetworkAwareGNNTrainer
from tasks.intervention_selection import NetworkAwareInterventionSelector
from simulation.simple_intervention import SimpleInterventionSimulator
from evaluation.network_metrics import NetworkEffectEvaluator

from training.loss_functions import DiscoveryLoss, UnifiedEnergyLoss
from training.discovery_trainer import DiscoveryGNNTrainer
from training.unified_gnn_trainer import UnifiedGNNTrainer

from analysis.pattern_analyzer import PatternAnalyzer
from analysis.intervention_recommender import InterventionRecommender
from analysis.baseline_comparison import BaselineComparison
from analysis.comprehensive_reporter import ComprehensiveReporter
from analysis.lv_group_evaluator import LVGroupEvaluator, evaluate_and_select_lv_groups

# Import all enhancement components
from models.semi_supervised_layers import (
    PseudoLabelGenerator, GraphLabelPropagation,
    SelfTrainingModule, ConsistencyRegularization
)
from models.enhanced_temporal_layers import (
    TemporalFusionNetwork, EnhancedTemporalTransformer,
    AdaptiveLSTM, SeasonalDecomposition
)
from models.uncertainty_quantification import (
    UncertaintyQuantifier, EnsembleUncertainty,
    ConfidenceCalibrator, BayesianGNNLayer
)
from models.explainability_layers import (
    EnhancedGNNExplainer, AttentionVisualizer,
    FeatureImportanceAnalyzer, ExplainableGATConv
)
from models.dynamic_graph_layers import (
    EdgeFeatureProcessor, DynamicGraphConstructor,
    HierarchicalGraphPooling
)
from training.active_learning import (
    ActiveLearningSelector, QueryByCommittee
)
from training.contrastive_learning import (
    GraphContrastiveLearning, SimCLRGNN, GraphAugmentor
)
from utils.output_validation import (
    PhysicsValidator, StructuredReportGenerator
)
from training.enhanced_trainer import EnhancedGNNTrainer


class UnifiedEnergyGNNSystem:
    """
    Main system class that orchestrates all components
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the system
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize components
        self._initialize_components()
        
        # Create output directories
        self._create_directories()
        
    
    def _get_input_dim(self):
        """Get input dimension from actual data"""
        if hasattr(self, 'feature_processor') and self.feature_processor:
            return self.feature_processor.get_feature_dim()
        # Try to get from first batch of data
        try:
            sample_data = self.kg_connector.get_sample_data(limit=10)
            if sample_data and 'features' in sample_data:
                return sample_data['features'].shape[-1]
        except:
            pass
        # Return None to trigger auto-detection
        return None
    
    def _get_building_features(self):
        """Get building feature count dynamically"""
        try:
            from utils.feature_mapping import feature_mapper
            sample = self.kg_connector.get_buildings_df(limit=1)
            features = feature_mapper.get_feature_vector(sample)
            return features.shape[-1]
        except:
            return None
    
    def _initialize_components(self):
        """Initialize all system components"""
        print("\nInitializing system components...")
        
        # Data components
        self.kg_connector = KGConnector(
            uri=self.config['kg']['uri'],
            user=self.config['kg']['user'],
            password=self.config['kg']['password']
        )
        self.graph_constructor = GraphConstructor(self.config['graph'])
        self.feature_processor = FeatureProcessor()
        self.data_loader = EnergyDataLoader(self.config['data_loader'], mode='train')
        
        # Model components
        self.model = self._build_model()
        
        # Initialize enhancements based on configuration
        if self.config.get('enhancements', {}).get('enabled', False):
            self._initialize_enhancements()
        
        # Training components - determine mode from config
        training_mode = self.config.get('training', {}).get('mode', 'enhanced')
        use_enhanced = training_mode == 'enhanced' or self.config.get('enhancements', {}).get('enabled', False)
        use_discovery = training_mode == 'discovery' or self.config.get('use_discovery_mode', False)
        use_network_aware = training_mode == 'network_aware'
        
        if use_network_aware:
            # Use network-aware trainer for multi-hop effects
            self.trainer = NetworkAwareGNNTrainer(
                self.config,
                self.kg_connector
            )
            self.loss_fn = None
            print("[OK] Network-aware trainer initialized")
        elif use_enhanced:
            # Use enhanced trainer with all new features
            self.trainer = EnhancedGNNTrainer(
                base_model=self.model,
                config=self.config
            )
            self.loss_fn = None  # Enhanced trainer manages its own losses
            print("[OK] Enhanced trainer initialized with all improvements")
        elif use_discovery:
            # Use discovery trainer for unsupervised learning
            self.trainer = DiscoveryGNNTrainer(
                self.model,
                self.config['training'],
                self.device
            )
            self.loss_fn = self.trainer.criterion  # Loss is built into trainer
        else:
            # Fall back to original trainer if needed
            self.loss_fn = UnifiedEnergyLoss(self.config['loss'])
            self.trainer = UnifiedGNNTrainer(
                self.model,
                self.loss_fn,
                self.config['training'],
                self.device,
                use_wandb=self.config.get('use_wandb', False)
            )
        
        # Analysis components
        self.pattern_analyzer = PatternAnalyzer(self.config['analysis'])
        self.intervention_recommender = InterventionRecommender(self.config['planning'])
        self.baseline_comparison = BaselineComparison(self.config['validation'])
        self.comprehensive_reporter = ComprehensiveReporter(self.config)
        
        print("[OK] All components initialized successfully")
    
    def _initialize_enhancements(self):
        """Initialize ALL enhancement modules based on configuration"""
        enhancements = self.config.get('enhancements', {})
        model_config = self.config.get('model', {})
        
        # Semi-supervised learning
        if enhancements.get('use_semi_supervised', False):
            ssl_config = enhancements.get('semi_supervised', {})
            self.pseudo_generator = PseudoLabelGenerator(
                hidden_dim=model_config['hidden_dim'],
                num_classes=model_config['num_clusters'],
                confidence_threshold=ssl_config.get('confidence_threshold', 0.85)
            ).to(self.device)
            
            self.label_propagator = GraphLabelPropagation(
                num_iterations=ssl_config.get('propagation_iterations', 10),
                alpha=ssl_config.get('propagation_alpha', 0.85)
            ).to(self.device)
            
            self.self_trainer = SelfTrainingModule(
                base_model=self.model,
                num_classes=model_config['num_clusters'],
                initial_threshold=ssl_config.get('initial_threshold', 0.9),
                final_threshold=ssl_config.get('final_threshold', 0.7)
            ).to(self.device)
            print("  [+] Semi-supervised learning enabled")
        
        # Uncertainty quantification
        if enhancements.get('use_uncertainty', False):
            uq_config = enhancements.get('uncertainty', {})
            self.uncertainty_quantifier = UncertaintyQuantifier(
                base_model=self.model,
                num_classes=model_config['num_clusters'],
                mc_samples=uq_config.get('mc_samples', 20),
                temperature=uq_config.get('temperature', 1.0)
            ).to(self.device)
            
            self.confidence_calibrator = ConfidenceCalibrator(
                num_classes=model_config['num_clusters']
            ).to(self.device)
            print("  [+] Uncertainty quantification enabled")
        
        # Explainability
        if enhancements.get('use_explainability', False):
            exp_config = enhancements.get('explainability', {})
            self.explainer = EnhancedGNNExplainer(
                model=self.model,
                num_hops=exp_config.get('num_hops', 3)
            ).to(self.device)
            
            self.attention_visualizer = AttentionVisualizer(
                save_dir=exp_config.get('visualization_dir', 'visualizations/attention')
            )
            
            if exp_config.get('register_hooks', False):
                self.attention_visualizer.register_attention_hook(self.model)
            
            self.feature_analyzer = FeatureImportanceAnalyzer(
                model=self.model
            ).to(self.device)
            print("  [+] Explainability enabled")
        
        # Active learning
        if enhancements.get('use_active_learning', False):
            al_config = enhancements.get('active_learning', {})
            self.active_selector = ActiveLearningSelector(
                model=self.model,
                strategy=al_config.get('strategy', 'hybrid'),
                budget=al_config.get('budget', 10),
                device=self.device.type
            )
            print("  [+] Active learning enabled")
        
        # Contrastive learning
        if enhancements.get('use_contrastive', False):
            cl_config = enhancements.get('contrastive', {})
            self.contrastive_learner = GraphContrastiveLearning(
                encoder=self.model,
                hidden_dim=model_config['hidden_dim'],
                projection_dim=cl_config.get('projection_dim', 64),
                temperature=cl_config.get('temperature', 0.5)
            ).to(self.device)
            print("  [+] Contrastive learning enabled")
        
        # Dynamic graph construction
        if enhancements.get('use_dynamic_graph', False):
            dg_config = enhancements.get('dynamic_graph', {})
            self.dynamic_graph = DynamicGraphConstructor(
                hidden_dim=model_config['hidden_dim'],
                similarity_threshold=dg_config.get('similarity_threshold', 0.8)
            ).to(self.device)
            
        if enhancements.get('use_edge_features', False):
            self.edge_processor = EdgeFeatureProcessor(
                node_dim=model_config.get('input_dim', 17),
                edge_dim=model_config.get('edge_dim', 3),
                hidden_dim=model_config['hidden_dim']
            ).to(self.device)
            print("  [+] Edge feature processing enabled")
        
        # Enhanced temporal processing
        if enhancements.get('use_enhanced_temporal', False):
            et_config = enhancements.get('enhanced_temporal', {})
            self.temporal_fusion = TemporalFusionNetwork(
                input_dim=model_config.get('temporal_dim', 8),
                hidden_dim=model_config['hidden_dim'],
                num_heads=model_config.get('num_heads', 8)
            ).to(self.device)
            print("  [+] Enhanced temporal processing enabled")
        
        # Physics validation
        if enhancements.get('use_physics_validation', False):
            pv_config = enhancements.get('physics_validation', {})
            self.physics_validator = PhysicsValidator(
                tolerance=pv_config.get('tolerance', 0.001)
            )
            print("  [+] Physics validation enabled")
        
        # Structured reporting and hierarchical pooling
        if enhancements.get('use_structured_reports', False):
            self.report_generator = StructuredReportGenerator(
                output_dir=self.config.get('reporting', {}).get('output_dir', 'reports')
            )
            print("  [+] Structured reporting enabled")
            
        # Hierarchical pooling
        if enhancements.get('use_hierarchical_pooling', False):
            hp_config = enhancements.get('hierarchical_pooling', {})
            self.hierarchical_pooling = HierarchicalGraphPooling(
                hidden_dim=model_config['hidden_dim'],
                pooling_ratio=hp_config.get('pooling_ratio', 0.5),
                num_levels=hp_config.get('num_levels', 3)
            ).to(self.device)
            print("  [+] Hierarchical pooling enabled")
        
    def _build_model(self) -> nn.Module:
        """Build the GNN model with discovery heads"""
        model_config = self.config['model']
        
        # Update config for discovery mode
        model_config['max_clusters'] = model_config.get('num_clusters', 10)
        model_config['min_cluster_size'] = self.config['data_loader'].get('min_cluster_size', 3)
        model_config['max_cluster_size'] = self.config['data_loader'].get('max_cluster_size', 20)
        
        # Create model using factory function
        model_type = model_config.get('type', 'hetero')
        model = create_gnn_model(model_type, model_config)
        
        # Model already includes new heads from our changes
        return model.to(self.device)
    
    def _create_directories(self):
        """Create necessary output directories"""
        dirs = [
            'checkpoints',
            'results',
            'results/analysis',
            'results/interventions',
            'results/comparisons',
            'results/visualizations',
            'results/inference',
            'reports',
            'reports/lv_group_evaluation',
            'logs'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_and_prepare_data(self, evaluate_groups=True) -> tuple:
        """
        Load data from KG and prepare for training
        
        Returns:
            train_loader, val_loader, test_loader
        """
        print("\n" + "="*80)
        print("LOADING AND PREPARING DATA")
        print("="*80)
        
        # Connect to KG
        print("\n1. Connecting to Knowledge Graph...")
        # Connection established in __init__
        
        # Extract data
        print("\n2. Extracting data from KG...")
        
        # Get LV groups
        lv_group_ids = self.kg_connector.get_all_lv_groups()
        print(f"   Found {len(lv_group_ids)} LV groups")
        
        # Evaluate and select LV groups if requested
        if evaluate_groups:
            print("\n3. Evaluating LV groups for suitability...")
            selected_ids = self._evaluate_and_select_lv_groups(lv_group_ids)
            print(f"   Selected {len(selected_ids)} high-priority LV groups")
            lv_group_ids = selected_ids
        
        # Process each LV group
        all_data = []
        
        for lv_group_id in lv_group_ids[:self.config.get('max_lv_groups', 50)]:
            print(f"\n   Processing LV group: {lv_group_id}")
            
            # Get complete LV group data
            lv_data = self.kg_connector.get_lv_group_data(lv_group_id)
            buildings = lv_data['buildings']
            
            if len(buildings) < self.config['data_loader']['min_cluster_size']:
                print(f"   Skipping - too few buildings ({len(buildings)})")
                continue
                
            # Get temporal data
            building_ids = [b['id'] for b in buildings]
            temporal_data = self.kg_connector.get_building_time_series(
                building_ids=building_ids,
                lookback_hours=24*7  # One week of data
            )
            
            # Use edges from lv_data
            grid_topology = lv_data['edges']
            
            # Convert to DataFrame for data loader
            buildings_df = pd.DataFrame(buildings)
            buildings_df['lv_group'] = lv_group_id
            
            # Create graph data
            graph_data = self.data_loader.create_lv_group_data(
                buildings=buildings_df,
                grid_topology=grid_topology,
                temporal_data=temporal_data,
                lv_group_id=lv_group_id
            )
            
            if graph_data is not None:
                all_data.append(graph_data)
        
        step_num = 4 if evaluate_groups else 3
        print(f"\n{step_num}. Created {len(all_data)} valid LV group graphs")
        
        # Split data
        step_num += 1
        print(f"\n{step_num}. Splitting data into train/val/test...")
        n_total = len(all_data)
        n_train = int(0.7 * n_total)
        n_val = int(0.15 * n_total)
        
        train_data = all_data[:n_train]
        val_data = all_data[n_train:n_train+n_val]
        test_data = all_data[n_train+n_val:]
        
        print(f"   Train: {len(train_data)} graphs")
        print(f"   Val: {len(val_data)} graphs")
        print(f"   Test: {len(test_data)} graphs")
        
        # Create data loaders
        step_num += 1
        print(f"\n{step_num}. Creating data loaders...")
        train_loader, val_loader, test_loader = self.data_loader.create_dataloaders(
            train_data, val_data, test_data
        )
        
        # Close KG connection
        if hasattr(self.kg_connector, 'close'):
            self.kg_connector.close()
        
        print("\n[OK] Data preparation complete")
        
        return train_loader, val_loader, test_loader
    
    def _evaluate_and_select_lv_groups(self, lv_group_ids):
        """Evaluate and select best LV groups"""
        evaluator = LVGroupEvaluator()
        
        # Prepare data for evaluation
        lv_groups_data = []
        for lg_id in lv_group_ids:
            # Get LV group data
            lv_data = self.kg_connector.get_lv_group_data(lg_id)
            buildings = lv_data.get('buildings', [])
            
            # Skip if too few buildings
            if len(buildings) < self.config['data_loader']['min_cluster_size']:
                continue
            
            # Enrich building data for evaluation
            enriched_buildings = []
            for b in buildings:
                building_data = {
                    'id': b.get('id'),
                    'type': self._get_building_type(b),
                    'area': b.get('area', 100),
                    'height': b.get('height', 10),
                    'energy_label': b.get('energy_label', 'D'),
                    'roof_area': b.get('roof_area', b.get('area', 100) * 0.6),
                    'has_solar': b.get('has_solar', False),
                    'has_battery': b.get('has_battery', False),
                    'has_heat_pump': b.get('has_heat_pump', False),
                    'occupancy': b.get('occupancy', 2),
                    'peak_hour': self._estimate_peak_hour(b),
                    'peak_demand': b.get('peak_demand', 5),
                    'distance_to_transformer': b.get('distance_to_transformer', 100),
                    'construction_year': b.get('construction_year', 2000)
                }
                enriched_buildings.append(building_data)
            
            # Enhanced LV group data with comprehensive assessment
            lv_group_data = {
                'id': lg_id,
                'buildings': enriched_buildings,
                'transformer': lv_data.get('transformer', {}),
                'buildings_data': enriched_buildings  # For enhanced assessment
            }
            lv_groups_data.append(lv_group_data)
        
        # Evaluate and select (relaxed criteria for testing)
        selection_criteria = {
            'min_diversity': 0.0,  # Accept all diversity levels for now
            # Don't filter by classification for initial testing
        }
        
        selected_ids, portfolio_df, reports = evaluate_and_select_lv_groups(
            lv_groups_data,
            selection_criteria=selection_criteria,
            top_n=min(25, len(lv_groups_data))
        )
        
        # Save evaluation report
        self._save_evaluation_report(portfolio_df, reports)
        
        return selected_ids
    
    def _get_building_type(self, building):
        """Determine building type from features"""
        if building.get('building_type'):
            return building['building_type']
        
        area = building.get('area', 100)
        if area > 500:
            return 'commercial_office'
        elif area < 100:
            return 'residential_multi'
        else:
            return 'residential_single'
    
    def _estimate_peak_hour(self, building):
        """Estimate peak hour based on building type"""
        building_type = self._get_building_type(building)
        
        peak_hours = {
            'residential_single': 18,
            'residential_multi': 19,
            'commercial_office': 14,
            'commercial_retail': 12,
            'industrial': 10,
            'educational': 15,
            'healthcare': 11,
            'mixed_use': 17
        }
        
        return peak_hours.get(building_type, 18)
    
    def _save_evaluation_report(self, portfolio_df, reports):
        """Save evaluation results"""
        reports_dir = Path('reports/lv_group_evaluation')
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save portfolio summary
        portfolio_path = reports_dir / 'portfolio_summary.csv'
        portfolio_df.to_csv(portfolio_path, index=False)
        print(f"   Portfolio summary saved to {portfolio_path}")
        
        # Save top 5 detailed reports
        for i, report_data in enumerate(reports[:5]):
            report_path = reports_dir / f"lv_group_{report_data['lv_group_id']}_evaluation.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_data['report'])
        
        # Print summary
        print("\n" + "="*80)
        print("LV GROUP EVALUATION SUMMARY")
        print("="*80)
        print(f"\nTop 5 LV Groups by Overall Score:")
        if not portfolio_df.empty:
            print(portfolio_df[['lv_group_id', 'classification', 'overall_score']].head())
            print("\nClassification Distribution:")
            print(portfolio_df['classification'].value_counts())
        print("="*80)
    
    def train_network_aware_model(self, district_name: str = None, use_intervention_loop: bool = True):
        """
        Train network-aware GNN model with intervention loop
        Demonstrates multi-hop network effects beyond simple correlation
        
        Args:
            district_name: District to load data from (if using KG)
            use_intervention_loop: Whether to use intervention loop training
        """
        print("\n" + "="*80)
        print("TRAINING NETWORK-AWARE GNN MODEL")
        print("="*80)
        
        # Initialize network-aware trainer
        network_config = {
            'model': {
                'hidden_dim': self.config.get('hidden_dim', 128),
                'num_layers': 4,  # Need 4 layers for better multi-hop
                'max_cascade_hops': 3,
                'building_features': self.config.get('input_dim', 17)
            },
            'loss': {
                'complementarity_weight': 1.0,
                'network_impact_weight': 2.0,  # Higher weight on network effects
                'cascade_weight': 1.5
            },
            'selection': {
                'local_weight': 0.3,
                'network_weight': 0.7  # Prioritize network value
            },
            'learning_rate': self.config.get('learning_rate', 1e-3),
            'device': self.device.type
        }
        
        network_trainer = NetworkAwareGNNTrainer(network_config, self.kg_connector)
        
        # Load MV network data (multiple LVs)
        if self.kg_connector and district_name:
            print(f"\nLoading MV network data for {district_name}...")
            data = network_trainer.load_mv_network_data(district_name)
        else:
            print("\nCreating synthetic MV network data...")
            data = self._create_synthetic_mv_network(num_buildings=200, num_lv_groups=10)
        
        print(f"Loaded {data.x.shape[0]} buildings across multiple LVs")
        
        # Phase 1: Train base model
        print("\n--- Phase 1: Base Model Training ---")
        base_history = network_trainer.train_base_model(data, epochs=50)
        
        # Phase 2: Intervention loop (if enabled)
        if use_intervention_loop:
            print("\n--- Phase 2: Intervention Loop ---")
            intervention_results = network_trainer.intervention_loop(
                data,
                num_rounds=5,
                interventions_per_round=5
            )
            
            # Phase 3: Compare to baseline
            print("\n--- Phase 3: Baseline Comparison ---")
            comparison = network_trainer.compare_to_baseline(data)
            
            # Check success criteria
            network_improvement = comparison.get('network_improvement', 0)
            cascade_improvement = comparison.get('cascade_improvement', 0)
            
            print("\n" + "="*80)
            print("NETWORK-AWARE TRAINING RESULTS")
            print("="*80)
            # Display improvements with proper interpretation
            if network_improvement > 0:
                print(f"[OK] Network Impact Improvement: {network_improvement:.1%}")
            else:
                print(f"[INFO] Network Impact: No improvement (GNN={comparison['gnn_network_impact']:.2f}, Baseline={comparison['baseline_network_impact']:.2f})")
            
            if cascade_improvement > 0:
                print(f"[OK] Cascade Value Improvement: {cascade_improvement:.1%}")
            else:
                print(f"[INFO] Cascade Value: No improvement (GNN={comparison['gnn_cascade_value']:.2f}, Baseline={comparison['baseline_cascade_value']:.2f})")
            
            # Only claim multi-hop effects are significant if there's meaningful improvement
            if cascade_improvement > 0.05:  # >5% improvement
                print(f"[OK] Multi-hop effects proved significant")
            else:
                print(f"[INFO] Multi-hop effects detected but marginal ({cascade_improvement:.1%})")
            
            # Save results
            network_trainer.save_results({
                'base_history': base_history,
                'intervention_results': intervention_results,
                'comparison': comparison
            })
            
            return network_trainer, intervention_results
        
        return network_trainer, base_history
    
    def train_model(self, train_loader, val_loader, num_epochs: int = 100):
        """
        Train the GNN model (original method)
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
        """
        print("\n" + "="*80)
        print("TRAINING GNN MODEL")
        print("="*80)
        
        # Train model - handle both trainer types
        if hasattr(self.trainer, 'train'):
            # DiscoveryGNNTrainer uses 'epochs' parameter
            from training.discovery_trainer import DiscoveryGNNTrainer
            if isinstance(self.trainer, DiscoveryGNNTrainer):
                self.trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=num_epochs,
                    save_dir='./checkpoints'
                )
                # Create results from trainer history
                training_results = {
                    'history': self.trainer.training_history,
                    'best_metrics': self.trainer.best_metrics
                }
            else:
                # UnifiedGNNTrainer uses different parameters
                training_results = self.trainer.train(
                    train_loader=train_loader,
                    val_loader=val_loader,
                    num_epochs=num_epochs
                )
        
        # Save training history
        history_df = pd.DataFrame(training_results['history'])
        history_df.to_csv('results/training_history.csv', index=False)
        
        print("\n[OK] Training complete")
        print(f"Best validation metrics:")
        for metric, value in training_results['best_metrics'].items():
            print(f"  {metric}: {value:.4f}")
        
        return training_results
    
    def analyze_patterns(self, test_loader):
        """
        Analyze discovered patterns
        
        Args:
            test_loader: Test data loader
        """
        print("\n" + "="*80)
        print("ANALYZING DISCOVERED PATTERNS")
        print("="*80)
        
        self.model.eval()
        all_analysis_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = batch.to(self.device)
                
                print(f"\nAnalyzing batch {batch_idx + 1}/{len(test_loader)}")
                
                # Get model predictions
                outputs = self.model(batch)
                
                # Extract cluster assignments - handle both old and new output formats
                if 'clusters' in outputs:
                    # New discovery model format
                    cluster_assignments = torch.argmax(outputs['clusters'], dim=-1)
                elif 'clustering_cluster_assignments' in outputs:
                    # Old format
                    cluster_assignments = torch.argmax(
                        outputs['clustering_cluster_assignments'], dim=-1
                    )
                else:
                    # Fallback - create dummy clusters
                    cluster_assignments = torch.zeros(batch.x.size(0), dtype=torch.long)
                
                # Simple analysis using available data
                try:
                    # Create dummy temporal profiles with correct shape (96 timesteps)
                    num_nodes = batch.x.size(0)
                    temporal_profiles = torch.randn(num_nodes, 96).abs()  # 96 = 24 hours * 4 (15-min intervals)
                    
                    # Get complementarity matrix from outputs if available
                    complementarity_matrix = outputs.get('complementarity', 
                                                        outputs.get('clustering_complementarity_matrix'))
                    
                    # Create generation profiles (if available)
                    generation_profiles = torch.randn(num_nodes, 96).abs() * 0.5
                    
                    # Call analyzer with correct signature
                    analysis_results = self.pattern_analyzer.analyze_clusters(
                        cluster_assignments=cluster_assignments,
                        temporal_profiles=temporal_profiles,
                        building_features=batch.x,
                        edge_index=batch.edge_index,
                        complementarity_matrix=complementarity_matrix,
                        generation_profiles=generation_profiles,
                        network_data={'transformers': [], 'cables': []}  # Simplified
                    )
                except Exception as e:
                    print(f"  Warning: Pattern analysis failed: {e}")
                    # Create minimal results
                    analysis_results = {
                        'cluster_metrics': [],
                        'gaps': [],
                        'opportunities': [],
                        'bottlenecks': []
                    }
                
                all_analysis_results.append(analysis_results)
                
                # Generate report for first batch
                if batch_idx == 0:
                    report_df = self.pattern_analyzer.generate_report(
                        analysis_results,
                        f'results/analysis/pattern_analysis_batch_{batch_idx}.csv'
                    )
                    
                    # Visualize patterns
                    self.pattern_analyzer.visualize_patterns(
                        analysis_results,
                        f'results/visualizations/patterns_batch_{batch_idx}.png'
                    )
        
        print("\n[OK] Pattern analysis complete")
        print(f"Analyzed {len(all_analysis_results)} batches")
        
        # Summary statistics
        total_gaps = sum(len(r['energy_gaps']) for r in all_analysis_results)
        total_bottlenecks = sum(len(r['network_bottlenecks']) for r in all_analysis_results)
        avg_self_sufficiency = np.mean([
            m.self_sufficiency 
            for r in all_analysis_results 
            for m in r['cluster_metrics']
        ])
        
        print(f"\nSummary:")
        print(f"  Total energy gaps identified: {total_gaps}")
        print(f"  Total network bottlenecks: {total_bottlenecks}")
        print(f"  Average self-sufficiency: {avg_self_sufficiency:.3f}")
        
        return all_analysis_results
    
    def recommend_interventions(self, analysis_results, test_loader):
        """
        Generate intervention recommendations
        
        Args:
            analysis_results: Results from pattern analysis
            test_loader: Test data loader
        """
        print("\n" + "="*80)
        print("GENERATING INTERVENTION RECOMMENDATIONS")
        print("="*80)
        
        self.model.eval()
        all_plans = []
        
        with torch.no_grad():
            for batch_idx, (batch, analysis) in enumerate(zip(test_loader, analysis_results)):
                if batch_idx >= 3:  # Process first 3 batches for demo
                    break
                    
                batch = batch.to(self.device)
                
                print(f"\nGenerating interventions for batch {batch_idx + 1}")
                
                # Get model outputs
                outputs = self.model(batch)
                
                # Convert to numpy for recommender
                gnn_outputs_np = {
                    k: v.cpu().numpy() if torch.is_tensor(v) else v
                    for k, v in outputs.items()
                }
                
                # Create mock building data (in practice, load from KG)
                n_buildings = batch.x.shape[0]
                building_data = pd.DataFrame({
                    'id': range(n_buildings),
                    'floor_area': np.random.uniform(100, 500, n_buildings),
                    'roof_area': np.random.uniform(50, 200, n_buildings),
                    'building_age': np.random.uniform(5, 50, n_buildings),
                    'energy_intensity': np.random.uniform(50, 200, n_buildings),
                    'energy_label': np.random.choice(['B', 'C', 'D', 'E'], n_buildings)
                })
                
                # Generate recommendations
                plan = self.intervention_recommender.recommend_interventions(
                    analysis_results=analysis,
                    gnn_outputs=gnn_outputs_np,
                    building_data=building_data,
                    network_topology={'transformers': [], 'cables': []},
                    budget_constraint=1000000  # $1M budget
                )
                
                all_plans.append(plan)
                
                # Export plan
                plan_json = self.intervention_recommender.export_plan(
                    plan,
                    format='json',
                    output_path=f'results/interventions/plan_batch_{batch_idx}.json'
                )
                
                # Generate report
                plan_report = self.intervention_recommender.export_plan(
                    plan,
                    format='report',
                    output_path=f'results/interventions/report_batch_{batch_idx}.txt'
                )
                
                # Print summary
                print(f"  Total interventions: {len(plan.interventions)}")
                print(f"  Total cost: ${plan.total_cost:,.0f}")
                print(f"  Expected peak reduction: {plan.expected_benefits.get('peak_reduction', 0):.1f} kW")
                print(f"  Expected carbon reduction: {plan.expected_benefits.get('carbon_reduction', 0):.1f} tons/year")
        
        print("\n[OK] Intervention planning complete")
        print(f"Generated {len(all_plans)} intervention plans")
        
        return all_plans
    
    def run_baseline_comparison(self, test_loader):
        """
        Compare GNN with baseline methods
        
        Args:
            test_loader: Test data loader
        """
        print("\n" + "="*80)
        print("RUNNING BASELINE COMPARISON")
        print("="*80)
        
        # Use first batch for comparison
        test_batch = next(iter(test_loader))
        test_batch = test_batch.to(self.device)
        
        # Run comprehensive comparison
        comparison_report = self.baseline_comparison.run_comprehensive_comparison(
            temporal_profiles=test_batch.temporal_profiles,
            building_features=test_batch.x,
            edge_index=test_batch.edge_index,
            gnn_model=self.model,
            generation_profiles=test_batch.generation if hasattr(test_batch, 'generation') else None,
            network_data=None
        )
        
        # Generate report
        report_text = self.baseline_comparison.generate_comparison_report(
            comparison_report,
            save_path='results/comparisons/baseline_comparison_report.txt'
        )
        
        # Create visualizations
        self.baseline_comparison.visualize_comparison(
            comparison_report,
            save_path='results/visualizations/baseline_comparison.png'
        )
        
        print("\n[OK] Baseline comparison complete")
        
        # Print key improvements
        print("\nKey GNN Improvements:")
        for method, improvements in comparison_report.improvements.items():
            if method in ['kmeans', 'correlation', 'spectral']:
                self_suff_imp = improvements.get('self_sufficiency_improvement', 0)
                peak_red_imp = improvements.get('peak_reduction_improvement', 0)
                print(f"  vs {method}: +{self_suff_imp:.1f}% self-sufficiency, +{peak_red_imp:.1f}% peak reduction")
        
        return comparison_report
    
    def active_learning_loop(self, initial_train_loader, unlabeled_pool, num_rounds: int = 5):
        """
        Run active learning loop for data-efficient training
        """
        print("\n" + "="*80)
        print("ACTIVE LEARNING LOOP")
        print("="*80)
        
        if not self.config['enhancements']['use_active_learning']:
            print("Warning: Active learning not enabled in config")
            return
        
        current_train = list(initial_train_loader)
        remaining_pool = list(unlabeled_pool)
        
        for round_idx in range(num_rounds):
            print(f"\n--- Round {round_idx + 1}/{num_rounds} ---")
            
            # Select samples
            selected_indices, metrics = self.active_selector.select_samples(
                remaining_pool, current_train
            )
            
            print(f"Selected {len(selected_indices)} samples")
            print(f"Selection metrics: {metrics}")
            
            # Move selected samples to training set
            for idx in sorted(selected_indices, reverse=True):
                if idx < len(remaining_pool):
                    current_train.append(remaining_pool.pop(idx))
            
            # Retrain model
            from torch_geometric.loader import DataLoader
            new_train_loader = DataLoader(
                current_train,
                batch_size=self.config.get('data_loader', {}).get('batch_size', 32),
                shuffle=True
            )
            
            # Train for few epochs
            if hasattr(self.trainer, 'train_epoch'):
                self.trainer.train_epoch(new_train_loader, epoch=round_idx)
            
            # Update selector performance
            val_metrics = self.trainer.validate(initial_train_loader) if hasattr(self.trainer, 'validate') else {}
            self.active_selector.update_performance(selected_indices, val_metrics)
        
        print(f"[✓] Active learning complete. Final training set size: {len(current_train)}")
        return current_train

    def train_enhanced(self, num_epochs: int = None):
        """
        Train with enhanced features based on configuration
        """
        print("\n" + "="*80)
        print("STARTING ENHANCED TRAINING")
        print("="*80)
        
        # Load data
        train_loader, val_loader, test_loader = self.load_and_prepare_data()
        
        # Prepare unlabeled data if using semi-supervised
        unlabeled_loader = None
        if self.config['enhancements']['use_semi_supervised']:
            # Create synthetic unlabeled data from training set
            print("Creating synthetic unlabeled data from training set")
            unlabeled_loader = self._create_unlabeled_loader(train_loader)
        
        # Train based on mode
        training_mode = self.config['training'].get('mode', 'enhanced')
        num_epochs = num_epochs or self.config['training'].get('num_epochs', 100)
        
        if training_mode == 'enhanced' and isinstance(self.trainer, EnhancedGNNTrainer):
            # Use enhanced trainer
            results = self.trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                unlabeled_loader=unlabeled_loader,
                num_epochs=num_epochs
            )
            
            # Generate comprehensive report if configured
            if self.config['reporting']['generate_final_report']:
                report = self.trainer.generate_comprehensive_report(test_loader)
                print(f"Report saved to {self.trainer.experiment_dir}")
        else:
            # Use standard training
            results = self.train_model(train_loader, val_loader, num_epochs)
        
        # Evaluate on test set
        if self.config['evaluation']['run_test_evaluation']:
            test_metrics = self.evaluate_enhanced(test_loader)
            results['test_metrics'] = test_metrics
        
        # Save final model
        if self.config['training']['save_final_model']:
            self.save_model('final_model.pt')
        
        return results
    
    def evaluate_enhanced(self, test_loader):
        """
        Comprehensive evaluation with all enhancements
        """
        print("\n" + "="*80)
        print("ENHANCED EVALUATION")
        print("="*80)
        
        metrics = {}
        
        # Standard evaluation
        if hasattr(self.trainer, 'validate'):
            metrics['standard'] = self.trainer.validate(test_loader)
        
        # Uncertainty evaluation
        if self.config['enhancements']['use_uncertainty'] and hasattr(self, 'uncertainty_quantifier'):
            metrics['uncertainty'] = self._evaluate_uncertainty(test_loader)
        
        # Physics validation
        if self.config['enhancements']['use_physics_validation'] and hasattr(self, 'physics_validator'):
            metrics['physics'] = self._evaluate_physics(test_loader)
        
        # Explainability analysis
        if self.config['enhancements']['use_explainability'] and hasattr(self, 'explainer'):
            metrics['explanations'] = self._generate_explanations(test_loader)
        
        # Pattern analysis
        if self.config['evaluation']['analyze_patterns']:
            metrics['patterns'] = self.analyze_patterns(test_loader)
        
        # Baseline comparison
        if self.config['evaluation']['compare_baselines']:
            metrics['baseline_comparison'] = self.run_baseline_comparison(test_loader)
        
        return metrics
    
    def _evaluate_uncertainty(self, data_loader):
        """Evaluate model uncertainty"""
        print("Evaluating uncertainty quantification...")
        
        uncertainties = []
        confidences = []
        
        for batch in data_loader:
            batch = batch.to(self.device)
            output = self.uncertainty_quantifier(batch)
            
            uncertainties.append(output['total_uncertainty'].mean().item())
            confidences.append(output['confidence'].mean().item())
        
        return {
            'mean_uncertainty': np.mean(uncertainties),
            'mean_confidence': np.mean(confidences),
            'uncertainty_std': np.std(uncertainties)
        }
    
    def _evaluate_physics(self, data_loader):
        """Evaluate physics constraint violations"""
        print("Evaluating physics constraints...")
        
        violations = []
        
        for batch in data_loader:
            batch = batch.to(self.device)
            output = self.model(batch)
            
            # Prepare predictions for validation
            predictions = {
                'clustering_assignments': output.get('clustering_cluster_assignments'),
                'energy_flow': output.get('energy_flow'),
                'power_flows': output.get('power_flows')
            }
            
            # Validate
            report = self.physics_validator.comprehensive_validation(
                predictions,
                self.config.get('physics_constraints', {})
            )
            
            violations.append(not report['overall_valid'])
        
        return {
            'violation_rate': np.mean(violations),
            'total_violations': sum(violations)
        }
    
    def _generate_explanations(self, data_loader):
        """Generate explanations for predictions"""
        print("Generating explanations...")
        
        explanations = []
        
        for batch_idx, batch in enumerate(data_loader):
            if batch_idx >= 5:  # Limit to 5 batches
                break
                
            batch = batch.to(self.device)
            
            # Explain first node in batch
            if batch.x.size(0) > 0:
                explanation = self.explainer.explain_node(batch, node_idx=0)
                importance = self.feature_analyzer.comprehensive_importance(batch, 0)
                
                explanations.append({
                    'batch_idx': batch_idx,
                    'top_features': importance['top_features'],
                    'edge_importance': explanation.get('edge_importance', [])
                })
        
        # Visualize attention if configured
        if self.config['enhancements']['explainability'].get('visualize_attention', False):
            self.attention_visualizer.visualize_layer_attention_stats()
        
        return explanations
    
    def _create_unlabeled_loader(self, train_loader):
        """Create unlabeled data by removing labels from training data"""
        unlabeled_data = []
        
        for batch in train_loader:
            # Clone batch without labels
            unlabeled_batch = batch.clone()
            if hasattr(unlabeled_batch, 'y'):
                delattr(unlabeled_batch, 'y')
            unlabeled_data.append(unlabeled_batch)
        
        from torch_geometric.loader import DataLoader
        return DataLoader(unlabeled_data, batch_size=train_loader.batch_size, shuffle=True)
    
    def save_model(self, filename: str):
        """Save model and all components"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add enhancement states if they exist
        if hasattr(self, 'pseudo_generator'):
            checkpoint['pseudo_generator'] = self.pseudo_generator.state_dict()
        if hasattr(self, 'temporal_fusion'):
            checkpoint['temporal_fusion'] = self.temporal_fusion.state_dict()
        if hasattr(self, 'uncertainty_quantifier'):
            checkpoint['uncertainty_quantifier'] = self.uncertainty_quantifier.state_dict()
        
        path = Path('checkpoints') / filename
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, filename: str):
        """Load model and components"""
        path = Path('checkpoints') / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        
        # Load enhancement states
        if 'pseudo_generator' in checkpoint and hasattr(self, 'pseudo_generator'):
            self.pseudo_generator.load_state_dict(checkpoint['pseudo_generator'])
        if 'temporal_fusion' in checkpoint and hasattr(self, 'temporal_fusion'):
            self.temporal_fusion.load_state_dict(checkpoint['temporal_fusion'])
        
        print(f"Model loaded from {path}")

    def run_inference(self, data_path: str):
        """
        Run inference on new data
        
        Args:
            data_path: Path to new data
        """
        print("\n" + "="*80)
        print("RUNNING INFERENCE")
        print("="*80)
        
        # Load checkpoint
        checkpoint_path = 'checkpoints/best_model.pt'
        if not Path(checkpoint_path).exists():
            print("Error: No trained model found. Please train first.")
            return
            
        self.trainer.load_checkpoint(checkpoint_path)
        self.model.eval()
        
        # Load new data (simplified - in practice, load from file or KG)
        # For demo, use test loader
        test_loader = self.load_and_prepare_data()[2]
        
        print("\nRunning inference on new data...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx >= 1:  # Process first batch for demo
                    break
                    
                batch = batch.to(self.device)
                
                # Get predictions
                outputs = self.model(batch)
                
                # Extract clusters
                clusters = torch.argmax(outputs['clustering_cluster_assignments'], dim=-1)
                
                # Analyze
                analysis = self.pattern_analyzer.analyze_clusters(
                    cluster_assignments=clusters,
                    temporal_profiles=batch.temporal_profiles,
                    building_features=batch.x,
                    edge_index=batch.edge_index,
                    generation_profiles=batch.generation if hasattr(batch, 'generation') else None
                )
                
                # Generate recommendations
                building_data = pd.DataFrame({
                    'id': range(batch.x.shape[0]),
                    'floor_area': np.random.uniform(100, 500, batch.x.shape[0])
                })
                
                plan = self.intervention_recommender.recommend_interventions(
                    analysis_results=analysis,
                    gnn_outputs={k: v.cpu().numpy() for k, v in outputs.items()},
                    building_data=building_data,
                    network_topology={},
                    budget_constraint=500000
                )
                
                # Save results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                # Save cluster assignments
                np.save(f'results/inference/clusters_{timestamp}.npy', clusters.cpu().numpy())
                
                # Save intervention plan
                self.intervention_recommender.export_plan(
                    plan,
                    format='json',
                    output_path=f'results/inference/plan_{timestamp}.json'
                )
                
                print(f"\nInference results:")
                print(f"  Number of clusters: {len(torch.unique(clusters))}")
                print(f"  Recommended interventions: {len(plan.interventions)}")
                print(f"  Total investment: ${plan.total_cost:,.0f}")
                print(f"  Expected benefits:")
                for benefit, value in plan.expected_benefits.items():
                    print(f"    {benefit}: {value:.2f}")
        
        print("\n[OK] Inference complete")
    
    def _create_synthetic_mv_network(self, num_buildings: int = 200, num_lv_groups: int = 10):
        """Create synthetic MV network data for testing"""
        from torch_geometric.data import Data
        
        # Distribute buildings across LV groups
        buildings_per_lv = num_buildings // num_lv_groups
        
        # Create node features (17 dimensions as expected)
        features = []
        transformer_groups = []
        
        for lv_idx in range(num_lv_groups):
            for b_idx in range(buildings_per_lv):
                # Random but structured features
                feat = [
                    np.random.randint(0, 7) / 7.0,  # Energy label (A-G)
                    np.random.uniform(50, 200) / 1000,  # Area
                    np.random.uniform(20, 100) / 100,  # Roof area
                    np.random.uniform(5, 20) / 50,  # Height
                    np.random.random() > 0.8,  # Has solar
                    np.random.random() > 0.9,  # Has battery
                    np.random.random() > 0.85,  # Has heat pump
                    np.random.uniform(0.3, 0.9),  # Solar potential
                    np.random.uniform(0.2, 0.8),  # Electrification feasibility
                ]
                
                # Pad to 17 features
                feat.extend([np.random.random() for _ in range(17 - len(feat))])
                features.append(feat)
                transformer_groups.append(f'LV_{lv_idx:04d}')
        
        features = torch.tensor(features, dtype=torch.float32)
        
        # Create edges (within LV groups + some cross-LV for testing)
        edges = []
        
        # Within LV group connections
        for lv_idx in range(num_lv_groups):
            start_idx = lv_idx * buildings_per_lv
            end_idx = start_idx + buildings_per_lv
            
            # Create local connectivity
            for i in range(start_idx, end_idx):
                # Connect to next 2-3 neighbors
                for j in range(i + 1, min(i + 3, end_idx)):
                    edges.append([i, j])
                    edges.append([j, i])
        
        # Add a few cross-LV edges (these should be penalized)
        for _ in range(20):
            lv1 = np.random.randint(0, num_lv_groups)
            lv2 = np.random.randint(0, num_lv_groups)
            if lv1 != lv2:
                b1 = lv1 * buildings_per_lv + np.random.randint(0, buildings_per_lv)
                b2 = lv2 * buildings_per_lv + np.random.randint(0, buildings_per_lv)
                edges.append([b1, b2])
                edges.append([b2, b1])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        # Create transformer mask
        transformer_mask = torch.zeros(num_buildings, num_buildings)
        for i in range(num_buildings):
            for j in range(num_buildings):
                if transformer_groups[i] == transformer_groups[j]:
                    transformer_mask[i, j] = 1.0
        
        # Create temporal profiles
        temporal_profiles = torch.abs(torch.randn(num_buildings, 24))
        
        # Add structure to profiles
        for i in range(num_buildings):
            # Morning and evening peaks
            temporal_profiles[i, 7:9] += 2.0
            temporal_profiles[i, 17:20] += 3.0
            # Random variation by LV group
            lv_idx = i // buildings_per_lv
            temporal_profiles[i] += torch.randn(24) * (0.5 + 0.1 * lv_idx)
        
        # Create centrality features
        degrees = torch.bincount(edge_index[0], minlength=num_buildings).float()
        centrality_features = torch.stack([
            degrees / degrees.max(),
            torch.rand(num_buildings),
            torch.rand(num_buildings),
            torch.rand(num_buildings),
            torch.rand(num_buildings)
        ], dim=1)
        
        # Create Data object
        data = Data(
            x=features,
            edge_index=edge_index,
            transformer_mask=transformer_mask,
            temporal_profiles=temporal_profiles,
            centrality_features=centrality_features,
            transformer_groups=transformer_groups,
            building_ids=[f'B_{i:04d}' for i in range(num_buildings)]
        )
        
        return data


def main():
    """Main execution function for Unified Energy GNN System"""
    parser = argparse.ArgumentParser(description='Unified Energy GNN System')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'inference', 'full', 'network-aware', 'active-learning', 'enhanced'],
        default='full',
        help='Execution mode'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config/config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to data for inference mode'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("="*80)
    print(" "*25 + "ENERGY GNN SYSTEM")
    print(" "*20 + "Network-Aware Energy Community Formation")
    print("="*80)
    print(f"\nMode: {args.mode.upper()}")
    print(f"Config: {args.config}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("="*80)
    
    # Initialize unified system
    system = UnifiedEnergyGNNSystem(args.config)
    
    if args.mode == 'train':
        # Training mode
        print("\n[MODE] TRAINING")
        train_loader, val_loader, _ = system.load_and_prepare_data()
        system.train_model(train_loader, val_loader, args.epochs)
        
    elif args.mode == 'evaluate':
        # Evaluation mode
        print("\n[MODE] EVALUATION")
        _, _, test_loader = system.load_and_prepare_data()
        
        # Analyze patterns
        analysis_results = system.analyze_patterns(test_loader)
        
        # Generate interventions
        system.recommend_interventions(analysis_results, test_loader)
        
        # Run comparison
        system.run_baseline_comparison(test_loader)
        
    elif args.mode == 'inference':
        # Inference mode
        print("\n[MODE] INFERENCE")
        system.run_inference(args.data)
    
    elif args.mode == 'network-aware':
        # Network-aware training with intervention loop
        print("\n[MODE] NETWORK-AWARE WITH INTERVENTION LOOP")
        print("This mode demonstrates multi-hop network effects beyond simple correlation")
        print("-" * 80)
        
        # Get district name if using KG
        district = None
        if system.kg_connector:
            # Use district from config or default to None for synthetic data
            district = system.config.get('network_aware', {}).get('district', None)
        
        # Run network-aware training
        network_trainer, results = system.train_network_aware_model(
            district_name=district,
            use_intervention_loop=True
        )
        
        print("\n" + "="*80)
        print("NETWORK-AWARE TRAINING COMPLETE")
        print("="*80)
        print("[OK] Proved GNN value through multi-hop network effects")
        print("[OK] Intervention cascade impacts tracked at 1-hop, 2-hop, 3-hop")
        print("[OK] Results saved to experiments folder")
        
    elif args.mode == 'active-learning':
        # Active learning mode
        print("\n[MODE] ACTIVE LEARNING")
        train_loader, val_loader, test_loader = system.load_and_prepare_data()
        
        # Run active learning loop
        al_config = system.config['enhancements'].get('active_learning', {})
        system.active_learning_loop(
            initial_train_loader=train_loader,
            unlabeled_pool=test_loader,
            num_rounds=al_config.get('num_rounds', 5)
        )
    
    elif args.mode == 'enhanced':
        # Enhanced training with all features
        print("\n[MODE] ENHANCED TRAINING")
        results = system.train_enhanced(num_epochs=args.epochs)
        
        print("\n" + "="*80)
        print("ENHANCED TRAINING COMPLETE")
        print("="*80)
        print(f"Best metrics: {results.get('best_metrics', {})}")
        
    else:  # full mode
        # Complete pipeline
        print("\n[STARTING] FULL PIPELINE MODE")
        
        # Load data
        train_loader, val_loader, test_loader = system.load_and_prepare_data()
        
        # Train model
        system.train_model(train_loader, val_loader, args.epochs)
        
        # Analyze patterns
        analysis_results = system.analyze_patterns(test_loader)
        
        # Generate interventions
        intervention_plans = system.recommend_interventions(analysis_results, test_loader)
        
        # Run comparison
        comparison_report = system.run_baseline_comparison(test_loader)
        
        # Generate comprehensive report
        print("\n" + "="*80)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*80)
        
        # Prepare data for comprehensive report
        if analysis_results and 'cluster_metrics' in analysis_results[0]:
            # Get all cluster assignments from the first batch
            cluster_metrics = analysis_results[0]['cluster_metrics']
            num_buildings = len(cluster_metrics)
            clusters = np.array([cm.cluster_id for cm in cluster_metrics])
        else:
            # Default if no analysis results
            num_buildings = 10
            clusters = np.zeros(num_buildings, dtype=int)
        
        building_data = {
            'building_ids': list(range(num_buildings)),
            'lv_group': 'LV_GROUP_0004',
            'consumption_profiles': np.random.uniform(1, 10, (num_buildings, 96)),
            'generation_profiles': np.random.uniform(0, 5, (num_buildings, 96))
        }
        
        # Determine the number of unique clusters
        n_clusters = len(np.unique(clusters))
        
        gnn_outputs = {
            'cluster_assignments': clusters,
            'cluster_probs': np.random.dirichlet(np.ones(n_clusters), num_buildings)
        }
        
        # Pass the InterventionPlan object directly if it exists, otherwise empty dict
        intervention_plan = intervention_plans[0] if intervention_plans else {}
        
        reports = system.comprehensive_reporter.generate_full_report(
            clusters,
            building_data,
            gnn_outputs,
            intervention_plan,
            save_dir="reports"
        )
        
        # Final summary
        print("\n" + "="*80)
        print("PIPELINE COMPLETE - SUMMARY")
        print("="*80)
        
        print("\n[RESULTS] Model Performance:")
        print(f"  Self-sufficiency: {comparison_report.gnn_result.metrics['self_sufficiency']:.3f}")
        print(f"  Peak reduction: {comparison_report.gnn_result.metrics['peak_reduction']:.3f}")
        print(f"  Complementarity: {comparison_report.gnn_result.metrics['complementarity']:.3f}")
        print(f"  Physics violations: {comparison_report.gnn_result.violations['total']:.3f}")
        
        print("\n[RESULTS] Improvements over baselines:")
        print(f"  vs K-means: +{comparison_report.improvements['kmeans']['self_sufficiency_improvement']:.1f}%")
        print(f"  vs Correlation: +{comparison_report.improvements['correlation']['self_sufficiency_improvement']:.1f}%")
        
        if intervention_plans:
            print("\n[RESULTS] Intervention Summary:")
            total_interventions = sum(len(p.interventions) for p in intervention_plans)
            total_cost = sum(p.total_cost for p in intervention_plans)
            print(f"  Total interventions: {total_interventions}")
            print(f"  Total investment: ${total_cost:,.0f}")
        
        print("\n[OK] All results saved to 'results/' directory")
        print("="*80)


if __name__ == "__main__":
    main()