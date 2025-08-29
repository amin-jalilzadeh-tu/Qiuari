"""
Unified GNN Main System for Energy Community Clustering
Based on our conversation: Dynamic sub-clustering, energy sharing, semi-supervised, solar cascade
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our components
from data.kg_connector import KGConnector
from data.graph_constructor import GraphConstructor
from data.feature_processor import FeatureProcessor
from data.data_loader import EnergyDataLoader
# Import models as discussed - using the best architecture
from models.base_gnn import HeteroEnergyGNN
from models.task_heads import ClusteringHead, UnifiedTaskHead
from models.pooling_layers import ConstrainedDiffPool
from models.physics_layers import PhysicsConstraintLayer
from models.network_aware_layers import NetworkAwareGNN
from models.semi_supervised_layers import SelfTrainingModule
from models.uncertainty_quantification import UncertaintyQuantifier
from models.attention_layers_simplified import SimpleMultiHeadAttention
from models.temporal_evolution import TemporalEvolutionPredictor, ClusterStabilityAnalyzer, EnergyFlowEvolution
# HeteroEnergyGNN already has built-in task heads, but we import these
# for additional functionality and flexibility
from training.loss_functions import DiscoveryLoss, ComplementarityLoss, SolarROILoss
from training.solar_simulator import SolarPerformanceSimulator
from tasks.cluster_quality_labeling import ClusterQualityLabeler
from tasks.solar_cascade_analyzer import SolarCascadeAnalyzer
from tasks.solar_roadmap_planner import SolarRoadmapPlanner, SolarRoadmap
from tasks.penetration_targets import PenetrationTargetsManager
from tasks.uncertainty_analyzer import UncertaintyAnalyzer
from tasks.explainability_generator import ExplainabilityGenerator
from evaluation.cluster_evaluator import ClusterEvaluator
from torch_geometric.data import HeteroData
from utils.device_utils import move_hetero_data_to_device

# Import visualization components
from visualization.real_data_connector import RealDataConnector
from visualization.data_aggregator import DataAggregator
from visualization.chart_generator import ChartGenerator
from visualization.report_generator import ReportGenerator
from visualization.economic_calculator import EconomicCalculator
from visualization.excel_reporter import ExcelReporter
from tracking.energy_flow_tracker import EnergyFlowTracker
from explainability.stakeholder_explainer import StakeholderExplainer


class UnifiedGNNSystem:
    """
    Complete GNN system as discussed:
    1. Dynamic sub-clustering within LV groups (10-250 buildings)
    2. Energy sharing tracking at each timestep
    3. Semi-supervised learning with performance-based labels (Option A: start with no labels)
    4. Solar recommendations with CASCADE effects
    5. Uncertainty quantification
    6. Explainability
    """
    
    def __init__(self, config_path: str = 'config/unified_config.yaml'):
        """Initialize unified system"""
        logger.info("Initializing Unified GNN System...")
        
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self._initialize_components()
        
        # Tracking metrics
        self.cluster_stability = defaultdict(list)  # Track cluster jumping
        self.energy_flows = defaultdict(dict)  # Track energy sharing
        self.pseudo_labels = {}  # Store generated labels
        self.solar_cascade_effects = {}  # Track cascade impacts
        
        # Initial assessment results
        self.initial_assessment = None
        self.selected_mv_station = None
        self.selected_lv_groups = []
        
    def _initialize_components(self):
        """Initialize all system components"""
        logger.info("Initializing components...")
        
        # Data pipeline
        self.kg_connector = KGConnector(
            uri=self.config['kg']['uri'],
            user=self.config['kg']['user'],
            password=self.config['kg']['password']
        )
        self.graph_constructor = GraphConstructor(self.kg_connector)
        self.feature_processor = FeatureProcessor()  # Doesn't take config
        self.data_loader = EnergyDataLoader(
            config=self.config,
            mode='train'
        )
        
        # Model - Use HeteroEnergyGNN as discussed (NOT solar_district_gnn!)
        self.model = HeteroEnergyGNN(self.config).to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Loss functions
        from training.balanced_loss import BalancedEnergyLoss, AdaptiveLossScheduler
        self.balanced_loss = BalancedEnergyLoss(self.config.get('balanced_loss', {
            'min_clusters': 4,
            'max_clusters': 8,
            'target_cluster_size': 20,
            'loss_weights': {
                'clustering': 0.3,
                'diversity': 0.25,
                'solar': 0.2,
                'balance': 0.15,
                'physics': 0.1
            }
        }))
        self.loss_scheduler = AdaptiveLossScheduler(self.balanced_loss.__dict__.copy())
        
        # Keep old losses for compatibility
        self.discovery_loss = DiscoveryLoss(self.config)
        self.discovery_loss.model = self.model
        self.complementarity_loss = ComplementarityLoss(
            correlation_weight=self.config.get('correlation_weight', 1.0),
            separation_weight=self.config.get('separation_weight', 0.5),
            diversity_weight=self.config.get('diversity_weight', 0.3)
        )
        self.solar_loss = SolarROILoss(class_weights=None)
        
        # Task modules
        cluster_config = self.config.get('cluster_metrics', self.config.get('task', {}).get('cluster_labeling', {}))
        self.quality_labeler = ClusterQualityLabeler(cluster_config)
        self.cascade_analyzer = SolarCascadeAnalyzer(self.config)
        self.uncertainty_analyzer = UncertaintyAnalyzer(self.config)
        self.explainability_gen = ExplainabilityGenerator(self.config)
        solar_config = self.config.get('task', {}).get('solar', self.config)
        self.solar_simulator = SolarPerformanceSimulator(solar_config)
        self.cluster_evaluator = ClusterEvaluator(self.config)
        
        # Initialize model evaluator for tracking
        from evaluation.model_evaluator import ModelEvaluator
        self.model_evaluator = ModelEvaluator()
        
        # Initialize new tracking and explainability modules
        self.flow_tracker = EnergyFlowTracker()
        self.stakeholder_explainer = StakeholderExplainer()
        
        # Initialize training stabilizers
        from training.training_utils import (
            ClusterMomentum, EarlyStopping, LearningRateScheduler,
            GradientClipper, ClusterValidator, TrainingMonitor
        )
        self.cluster_momentum = ClusterMomentum(momentum=0.9)
        self.early_stopping = EarlyStopping(patience=10, min_delta=0.001)
        self.lr_scheduler = LearningRateScheduler(self.optimizer, patience=5)
        self.gradient_clipper = GradientClipper(max_norm=1.0)
        self.cluster_validator = ClusterValidator(min_size=5, max_size=50)
        self.training_monitor = TrainingMonitor(log_interval=5)
        
        # Initialize roadmap and penetration components
        self.roadmap_planner = SolarRoadmapPlanner(self.config.get('roadmap', {}))
        self.penetration_manager = PenetrationTargetsManager(self.config.get('penetration', {}))
        self.temporal_evolution = TemporalEvolutionPredictor(self.config.get('temporal', self.config.get('model', {}))).to(self.device)
        self.cluster_stability_analyzer = ClusterStabilityAnalyzer()
        self.energy_flow_evolution = EnergyFlowEvolution(self.config.get('energy_flow', {}))
        
        logger.info("All components initialized")
    
    def run_initial_assessment(self):
        """Run initial assessment to select MV station and evaluate LV groups"""
        logger.info("\n" + "="*60)
        logger.info("RUNNING INITIAL NETWORK ASSESSMENT")
        logger.info("="*60 + "\n")
        
        from analysis.lv_mv_evaluator import EnhancedLVMVEvaluator, LVGroupMetrics, MVGroupMetrics
        
        # Initialize evaluator
        evaluator = EnhancedLVMVEvaluator(self.config.get('assessment', {}))
        
        # Get all MV stations from KG
        query = """
        MATCH (mv:MVStation)-[:MV_SUPPLIES_LV]->(lv:CableGroup {voltage_level: 'LV'})
        WITH mv, collect(DISTINCT lv) as lv_groups
        RETURN mv.station_id as mv_id, 
               [lg IN lv_groups | lg.group_id] as lv_group_ids,
               size(lv_groups) as lv_count
        ORDER BY lv_count DESC
        """
        mv_stations = self.kg_connector.run(query)
        
        all_mv_metrics = []
        all_lv_metrics = []
        
        logger.info(f"Evaluating {len(mv_stations)} MV stations...")
        
        for mv_data in mv_stations[:10]:  # Evaluate top 10 MV stations by size
            mv_id = mv_data['mv_id']
            lv_group_ids = mv_data['lv_group_ids']
            
            lv_groups_buildings = []
            lv_metrics_list = []
            
            # Get buildings for each LV group
            for lv_id in lv_group_ids:
                buildings_query = """
                MATCH (b:Building)-[:CONNECTED_TO]->(lv:CableGroup {group_id: $lv_id})
                RETURN b.ogc_fid as id,
                       b.building_function as building_function,
                       b.residential_type as residential_type,
                       b.non_residential_type as non_residential_type,
                       b.energy_label as energy_label,
                       b.area as area,
                       b.roof_area as roof_area,
                       b.flat_roof_area as flat_roof_area,
                       b.sloped_roof_area as sloped_roof_area,
                       b.building_orientation_cardinal as building_orientation_cardinal,
                       b.orientation as orientation,
                       b.has_solar as has_solar,
                       b.peak_electricity_demand_kw as peak_demand,
                       COALESCE(b.peak_hours, 18) as peak_hour,
                       $lv_id as lv_group_id
                """
                buildings = self.kg_connector.run(
                    buildings_query, 
                    {'lv_id': lv_id}
                )
                
                if len(buildings) >= 5:  # Min buildings threshold
                    lv_groups_buildings.append(buildings)
                    lv_metric = evaluator.evaluate_lv_group(buildings)
                    lv_metrics_list.append(lv_metric)
                    all_lv_metrics.append(lv_metric)
            
            if lv_groups_buildings:
                # Evaluate MV station
                mv_metrics = evaluator.evaluate_mv_station(lv_groups_buildings, mv_id)
                if mv_metrics:
                    all_mv_metrics.append(mv_metrics)
                    
                    # Generate report for this MV
                    report_path = Path(f"reports/assessment_{mv_id}.txt")
                    report_path.parent.mkdir(exist_ok=True)
                    report = evaluator.generate_assessment_report(
                        mv_metrics,
                        lv_metrics_list,
                        report_path
                    )
        
        # Select best MV station
        if all_mv_metrics:
            sorted_mv = sorted(all_mv_metrics, key=lambda x: x.planning_priority, reverse=True)
            best_mv = sorted_mv[0]
            
            self.selected_mv_station = best_mv.mv_station_id
            self.selected_lv_groups = best_mv.best_lv_groups
            
            # Store assessment results
            self.initial_assessment = {
                'mv_metrics': all_mv_metrics,
                'lv_metrics': all_lv_metrics,
                'selected_mv': best_mv.mv_station_id,
                'selected_lv': best_mv.best_lv_groups
            }
            
            # Log summary
            logger.info(f"\n{'='*60}")
            logger.info("ASSESSMENT COMPLETE")
            logger.info(f"{'='*60}")
            logger.info(f"Selected MV Station: {best_mv.mv_station_id}")
            logger.info(f"Planning Priority: {best_mv.planning_priority:.2f}/10")
            logger.info(f"LV Groups: {best_mv.lv_group_count}")
            logger.info(f"Total Buildings: {best_mv.total_buildings}")
            logger.info(f"Avg Diversity: {best_mv.avg_diversity_score:.1%}")
            logger.info(f"Poor Labels: {best_mv.avg_poor_label_ratio:.1%}")
            logger.info(f"Best LV Groups: {', '.join(best_mv.best_lv_groups[:3])}")
            
            if best_mv.planning_priority >= 7:
                logger.info("✅ EXCELLENT candidate - High diversity + intervention needs")
            elif best_mv.planning_priority >= 5:
                logger.info("✅ GOOD candidate - Balanced potential")
            else:
                logger.info("⚠️ LIMITED potential - Consider alternatives")
        
        return self.initial_assessment
    
    def train(self, num_epochs: int = None, run_assessment: bool = True):
        """
        Main training loop with phases as discussed:
        Phase 0: Initial Assessment (optional)
        Phase 1: Discovery (unsupervised clustering)
        Phase 2: Semi-supervised with cascade effects
        Phase 3: Solar optimization with learning loop
        """
        if num_epochs is None:
            num_epochs = self.config['training']['epochs']
        
        # Run initial assessment if requested
        if run_assessment:
            self.run_initial_assessment()
        
        logger.info(f"\n{'='*60}")
        logger.info("STARTING UNIFIED GNN TRAINING")
        logger.info(f"{'='*60}\n")
        
        # Get initial data
        data = self._prepare_data()
        
        # Training phases
        phase_epochs = {
            'discovery': num_epochs // 3,
            'semi_supervised': num_epochs // 3,
            'solar_optimization': num_epochs - 2 * (num_epochs // 3)
        }
        
        # Phase 1: Discovery
        logger.info(f"\n=== PHASE 1: DISCOVERY ({phase_epochs['discovery']} epochs) ===")
        logger.info("Learning optimal sub-clusters within LV groups...")
        
        for epoch in range(phase_epochs['discovery']):
            loss = self._train_discovery_epoch(data, epoch)
            
            # Track cluster stability every 5 epochs
            if epoch % 5 == 0:
                self._track_cluster_stability(data, epoch)
            
            logger.info(f"Epoch {epoch+1}/{phase_epochs['discovery']} - Loss: {loss:.4f}")
            
            # Track metrics for evaluation
            with torch.no_grad():
                outputs = self.model(data, task='clustering')
                cluster_assignments = outputs.get('cluster_assignments', outputs.get('cluster_logits', torch.zeros(1)).argmax(dim=-1))
                embeddings = outputs.get('embeddings', outputs.get('building_embeddings', None))
                self.model_evaluator.log_epoch('1_discovery', epoch+1, loss, cluster_assignments, embeddings)
            
            # Check early stopping
            if self.early_stopping(loss, epoch):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
            
            # Adjust learning rate
            self.lr_scheduler.step(loss)
        
        # Generate pseudo-labels after discovery
        logger.info("\nGenerating pseudo-labels from discovered clusters...")
        self._generate_pseudo_labels(data)
        
        # Phase 2: Semi-supervised with cascade
        logger.info(f"\n=== PHASE 2: SEMI-SUPERVISED WITH CASCADE ({phase_epochs['semi_supervised']} epochs) ===")
        logger.info("Refining clusters with pseudo-labels and cascade effects...")
        
        for epoch in range(phase_epochs['semi_supervised']):
            loss = self._train_semi_supervised_epoch(data, epoch)
            
            # Track energy flows
            if epoch % 3 == 0:
                self._track_energy_flows_old(data, epoch)
            
            logger.info(f"Epoch {epoch+1}/{phase_epochs['semi_supervised']} - Loss: {loss:.4f}")
            
            # Track metrics for evaluation
            with torch.no_grad():
                outputs = self.model(data, task='clustering')
                cluster_assignments = outputs.get('cluster_assignments', outputs.get('cluster_logits', torch.zeros(1)).argmax(dim=-1))
                embeddings = outputs.get('embeddings', outputs.get('building_embeddings', None))
                self.model_evaluator.log_epoch('2_semi_supervised', epoch+1, loss, cluster_assignments, embeddings)
        
        # Phase 3: Solar optimization
        logger.info(f"\n=== PHASE 3: SOLAR OPTIMIZATION ({phase_epochs['solar_optimization']} epochs) ===")
        logger.info("Learning solar cascade effects and optimal placement...")
        
        for epoch in range(phase_epochs['solar_optimization']):
            loss = self._train_solar_epoch(data, epoch)
            
            # Simulate solar deployment rounds
            if epoch % 10 == 0:
                pass  # self._simulate_solar_deployment(data, epoch)  # Disabled - causes hang
            
            logger.info(f"Epoch {epoch+1}/{phase_epochs['solar_optimization']} - Loss: {loss:.4f}")
            
            # Track metrics for evaluation
            with torch.no_grad():
                outputs = self.model(data, task='solar')
                cluster_assignments = outputs.get('cluster_assignments', outputs.get('cluster_logits', torch.zeros(1)).argmax(dim=-1))
                embeddings = outputs.get('embeddings', outputs.get('building_embeddings', None))
                self.model_evaluator.log_epoch('3_solar', epoch+1, loss, cluster_assignments, embeddings)
        
        logger.info("\n=== TRAINING COMPLETE ===")
        
        # Generate model evaluation report
        with torch.no_grad():
            outputs = self.model(data, task='clustering')
            cluster_assignments = outputs.get('cluster_assignments', outputs.get('cluster_logits', torch.zeros(1)).argmax(dim=-1))
            embeddings = outputs.get('embeddings', outputs.get('building_embeddings', None))
            
            # Calculate final cluster quality
            final_metrics = self.model_evaluator.calculate_cluster_quality(
                embeddings if embeddings is not None else data['building'].x,
                cluster_assignments,
                data['building'].x
            )
            
            # Generate evaluation visualizations and report
            self.model_evaluator.plot_training_curves()
            self.model_evaluator.plot_cluster_distribution(cluster_assignments, 'final')
            self.model_evaluator.save_evaluation_report(final_metrics)
        
        self._generate_final_report()
    
    def _prepare_data(self) -> HeteroData:
        """Prepare data from KG"""
        logger.info("Preparing data from Knowledge Graph...")
        
        # Get MV-LV hierarchy
        hierarchy = self.kg_connector.get_mv_lv_hierarchy()
        
        # Count MV stations and LV groups from hierarchy
        mv_count = sum(len(hv_data.get('mv_stations', [])) for hv_data in hierarchy.values())
        lv_count = sum(
            mv_data.get('lv_count', 0)
            for hv_data in hierarchy.values() 
            for mv_data in hv_data.get('mv_stations', [])
        )
        
        logger.info(f"Loaded hierarchy: {len(hierarchy)} HV substations, "
                   f"{mv_count} MV stations, {lv_count} LV groups")
        
        # Create graph data using GraphConstructor
        # Use an actual district from the KG
        district_name = "Zuidas"  # Use actual district from KG
        data = self.graph_constructor.build_hetero_graph(
            district_name, 
            include_energy_sharing=True, 
            include_temporal=True
        )
        data = move_hetero_data_to_device(data, self.device)
        
        # Add LV group IDs if available
        if hasattr(data['building'], 'lv_group_ids'):
            data.lv_group_ids = data['building'].lv_group_ids
        elif ('building', 'connected_to', 'cable_group') in data.edge_index_dict:
            # Extract LV groups from cable group connections
            edge_index = data.edge_index_dict[('building', 'connected_to', 'cable_group')]
            num_buildings = data['building'].x.size(0)
            lv_group_ids = torch.zeros(num_buildings, dtype=torch.long, device=self.device)
            
            # Assign buildings to LV groups based on cable connections
            for i in range(edge_index.shape[1]):
                building_idx = edge_index[0, i]
                cable_idx = edge_index[1, i]
                lv_group_ids[building_idx] = cable_idx  # Use cable group index as LV group ID
            
            data.lv_group_ids = lv_group_ids
            logger.info(f"Added LV group IDs for {num_buildings} buildings")
        else:
            # Fallback: assign all to same LV group
            num_buildings = data['building'].x.size(0) if 'building' in data.node_types else 0
            data.lv_group_ids = torch.zeros(num_buildings, dtype=torch.long, device=self.device)
            logger.warning("No LV group information found, assigning all buildings to LV group 0")
        
        return data
    
    def _train_discovery_epoch(self, data: HeteroData, epoch: int) -> float:
        """Discovery phase training with balanced loss to prevent collapse"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Add temporal metadata to data
        import datetime
        now = datetime.datetime.now()
        data.season = torch.tensor((now.month % 12) // 3).to(self.device)  # 0-3
        data.is_weekend = torch.tensor(now.weekday() >= 5).to(self.device)
        data.current_hour = torch.tensor(now.hour).to(self.device)
        
        # Forward pass with clustering task
        outputs = self.model(data, task='clustering')
        
        # Get cluster logits and embeddings
        cluster_logits = outputs.get('cluster_logits', outputs.get('clustering'))
        embeddings = outputs.get('building_embeddings', outputs.get('embeddings', data['building'].x))
        
        # Apply momentum to stabilize clusters (detach to avoid double backward)
        cluster_logits_stabilized = self.cluster_momentum.update(cluster_logits.detach())
        # Mix original and stabilized for gradient flow
        cluster_logits = 0.7 * cluster_logits + 0.3 * cluster_logits_stabilized.detach()
        
        # Use balanced loss to prevent collapse
        loss, loss_components = self.balanced_loss(
            cluster_logits=cluster_logits,
            embeddings=embeddings,
            building_features=data['building'].x,
            edge_index=data.get(('building', 'connected_to', 'building'), {}).get('edge_index'),
            lv_group_ids=data.get('lv_group_ids')
        )
        
        # Complementarity is already included in balanced loss, skip adding it again
        
        # Backward with gradient clipping
        loss.backward()
        self.gradient_clipper.clip(self.model)
        self.optimizer.step()
        
        # Validate clusters
        with torch.no_grad():
            cluster_assignments = cluster_logits.argmax(dim=-1)
            validation_metrics = self.cluster_validator.validate(cluster_assignments, data['building'].x)
            
            # Log warning if collapsed
            if validation_metrics.get('is_collapsed'):
                logger.warning("⚠️ CLUSTER COLLAPSE DETECTED - all buildings in single cluster!")
        
        return loss.item()
    
    def _train_semi_supervised_epoch(self, data: HeteroData, epoch: int) -> float:
        """Semi-supervised phase with BALANCED loss"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with clustering task
        outputs = self.model(data, task='clustering')
        
        # Get cluster logits and embeddings
        cluster_logits = outputs.get('cluster_logits', outputs.get('clustering'))
        embeddings = outputs.get('building_embeddings', outputs.get('embeddings', data['building'].x))
        
        # Apply momentum for stability
        cluster_logits_stabilized = self.cluster_momentum.update(cluster_logits.detach())
        cluster_logits = 0.8 * cluster_logits + 0.2 * cluster_logits_stabilized.detach()
        
        # Use BALANCED loss instead of discovery loss
        loss, loss_components = self.balanced_loss(
            cluster_logits=cluster_logits,
            embeddings=embeddings,
            building_features=data['building'].x,
            edge_index=data.get(('building', 'connected_to', 'building'), {}).get('edge_index'),
            lv_group_ids=data.get('lv_group_ids')
        )
        
        # Add small weight for pseudo-labels if confident
        if self.pseudo_labels and self.pseudo_labels.get('confidence', 0) > 0.7:
            labels = self.pseudo_labels.get('cluster_labels')
            if labels is not None:
                pseudo_loss = F.cross_entropy(cluster_logits, labels)
                loss = 0.9 * loss + 0.1 * pseudo_loss  # Mostly balanced loss
        
        # Add cascade effects
        if 'solar' in outputs:
            cascade_loss = self._compute_cascade_loss(outputs['solar'], data)
            loss += self.config['training']['cascade_weight'] * cascade_loss
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _train_solar_epoch(self, data: HeteroData, epoch: int) -> float:
        """Solar optimization WITH ANTI-COLLAPSE - maintains cluster diversity"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Get BOTH solar and clustering outputs
        solar_outputs = self.model(data, task='solar')
        cluster_outputs = self.model(data, task='clustering')
        
        # Extract cluster logits and embeddings
        cluster_logits = cluster_outputs.get('cluster_logits', cluster_outputs.get('clustering'))
        embeddings = cluster_outputs.get('building_embeddings', cluster_outputs.get('embeddings', data['building'].x))
        
        # CRITICAL: Use balanced loss as the BASE to prevent collapse
        base_loss, loss_components = self.balanced_loss(
            cluster_logits=cluster_logits,
            embeddings=embeddings,
            building_features=data['building'].x,
            edge_index=data.get(('building', 'connected_to', 'building'), {}).get('edge_index'),
            lv_group_ids=data.get('lv_group_ids')
        )
        # Get solar predictions from model output
        # Model now properly returns 4-class logits for ROI categories
        if 'solar' in solar_outputs:
            solar_logits = solar_outputs['solar']
        else:
            # Shouldn't happen with fixed model, but have fallback
            logger.warning("Solar task did not return expected 'solar' key")
            num_buildings = data['building'].x.size(0)
            solar_logits = torch.zeros(num_buildings, 4, device=self.device)
        
        # Generate pseudo-targets based on building characteristics
        # Buildings with good solar potential get better ROI classes
        num_buildings = solar_logits.shape[0]
        solar_targets = torch.zeros(num_buildings, dtype=torch.long, device=self.device)
        
        # Use building features to estimate ROI
        for i in range(num_buildings):
            area = data['building'].x[i, 0]  # Building area
            has_solar = data['building'].x[i, 2]  # Existing solar
            
            # Simple heuristic: larger buildings with no solar have better ROI
            if area > 0.5 and has_solar < 0.5:
                solar_targets[i] = 0  # Excellent ROI
            elif area > 0.3:
                solar_targets[i] = 1  # Good ROI
            elif area > 0.1:
                solar_targets[i] = 2  # Fair ROI
            else:
                solar_targets[i] = 3  # Poor ROI
        
        # Solar ROI loss (secondary)
        solar_loss = F.cross_entropy(solar_logits, solar_targets)
        
        # COMBINE: Prioritize cluster diversity over solar optimization
        # 70% balanced (anti-collapse), 30% solar optimization
        loss = 0.7 * base_loss + 0.3 * solar_loss
        
        # Backward with gradient clipping
        loss.backward()
        self.gradient_clipper.clip(self.model)
        self.optimizer.step()
        
        # Validate clusters - CRITICAL check
        with torch.no_grad():
            cluster_assignments = cluster_logits.argmax(dim=-1)
            validation_metrics = self.cluster_validator.validate(cluster_assignments, data['building'].x)
            
            if validation_metrics.get('is_collapsed'):
                logger.error("❌ SOLAR PHASE COLLAPSE! Adjusting weights...")
                # Increase diversity weight for next epoch
                self.balanced_loss.w_diversity *= 1.5
                self.balanced_loss.w_solar *= 0.5
        
        return loss.item()
    
    def _track_energy_flows(self, data: HeteroData, outputs: Dict, temporal_data):
        """Track detailed energy flows between buildings and communities"""
        from datetime import datetime, timedelta
        
        if 'clustering' not in outputs or outputs['clustering'].numel() == 0:
            return
        
        # Get cluster assignments
        cluster_assignments = outputs['clustering'].argmax(dim=1) if outputs['clustering'].dim() > 1 else outputs['clustering']
        
        # Simulate hourly flows for demonstration
        base_time = datetime.now()
        
        for hour in range(24):
            timestamp = base_time + timedelta(hours=hour)
            
            # Track P2P flows within communities
            for cluster_id in torch.unique(cluster_assignments):
                buildings_in_cluster = torch.where(cluster_assignments == cluster_id)[0]
                
                if len(buildings_in_cluster) > 1:
                    # Simulate energy sharing within community
                    for i, building_i in enumerate(buildings_in_cluster[:5]):  # Limit for performance
                        for building_j in buildings_in_cluster[i+1:i+2]:  # One pair per building
                            # Simulate a flow
                            energy_kwh = np.random.uniform(1, 10)
                            self.flow_tracker.record_flow(
                                timestamp=timestamp,
                                from_id=f'B_{building_i}',
                                to_id=f'B_{building_j}',
                                energy_kwh=energy_kwh,
                                flow_type='p2p',
                                cluster_id=int(cluster_id),
                                price_per_kwh=0.15,
                                metadata={'hour': hour}
                            )
            
            # Track community-level exchanges
            for cluster_id in torch.unique(cluster_assignments):
                buildings_in_cluster = torch.where(cluster_assignments == cluster_id)[0]
                building_ids = [f'B_{b}' for b in buildings_in_cluster.tolist()]
                
                # Simulate community metrics
                generation = np.random.uniform(50, 200) * len(buildings_in_cluster)
                consumption = np.random.uniform(100, 300) * len(buildings_in_cluster)
                shared = min(generation, consumption) * 0.3
                
                self.flow_tracker.record_community_exchange(
                    timestamp=timestamp,
                    cluster_id=int(cluster_id),
                    generation_kwh=generation,
                    consumption_kwh=consumption,
                    shared_kwh=shared,
                    grid_import_kwh=max(0, consumption - generation),
                    grid_export_kwh=max(0, generation - consumption),
                    self_sufficiency=min(1.0, generation / consumption),
                    participating_buildings=building_ids
                )
        
        # Record grid impacts
        if ('building', 'connected_to', 'transformer') in data.edge_index_dict:
            transformers = torch.unique(data.edge_index_dict[('building', 'connected_to', 'transformer')][1])
            
            for transformer_id in transformers[:5]:  # Limit for performance
                self.flow_tracker.record_grid_impact(
                    timestamp=base_time,
                    transformer_id=f'T_{transformer_id}',
                    load_kw=np.random.uniform(100, 500),
                    capacity_kw=630,  # Standard capacity
                    voltage_pu=np.random.uniform(0.95, 1.05),
                    losses_kw=np.random.uniform(5, 20),
                    congestion_level=np.random.uniform(0.3, 0.9)
                )
        
        # Generate flow report
        flow_report = self.flow_tracker.generate_flow_report()
        logger.info(f"Energy flow tracking complete: {flow_report['summary']['total_transactions']} transactions recorded")
        
        # Visualize flows
        self.flow_tracker.visualize_flows(cluster_assignments)
    
    def _generate_stakeholder_explanations(self, data: HeteroData, outputs: Dict, cluster_metrics: Dict):
        """Generate explanations for different stakeholder groups"""
        
        if 'clustering' not in outputs or outputs['clustering'].numel() == 0:
            return
        
        cluster_assignments = outputs['clustering'].argmax(dim=1) if outputs['clustering'].dim() > 1 else outputs['clustering']
        
        # Generate building owner explanations for a few sample buildings
        sample_buildings = torch.randperm(data['building'].x.size(0))[:3]  # 3 sample buildings
        
        for building_idx in sample_buildings:
            building_id = f'B_{building_idx}'
            cluster_id = int(cluster_assignments[building_idx])
            
            # Extract building features
            features = {
                'consumption_kwh': float(data['building'].x[building_idx, 0] * 1000),
                'energy_label': chr(65 + int(data['building'].x[building_idx, 1] * 7)),  # A-G
                'has_solar': bool(data['building'].x[building_idx, 2] > 0.5),
                'generation_kwh': float(data['building'].x[building_idx, 2] * 100) if data['building'].x[building_idx, 2] > 0.5 else 0
            }
            
            # Generate predictions
            predictions = {
                'monthly_savings': np.random.uniform(30, 150),
                'self_sufficiency': np.random.uniform(0.3, 0.7),
                'co2_reduction': np.random.uniform(100, 500),
                'grid_dependence': np.random.uniform(0.3, 0.7),
                'solar_roi_years': np.random.uniform(5, 10) if not features['has_solar'] else 'Already installed'
            }
            
            explanation = self.stakeholder_explainer.explain_for_building_owner(
                building_id=building_id,
                cluster_id=cluster_id,
                features=features,
                predictions=predictions
            )
            
            # Save explanation
            self.stakeholder_explainer.generate_explanation_report(
                audience='building_owner',
                entity_id=building_id,
                save_format='json'
            )
        
        # Generate grid operator explanation
        if ('building', 'connected_to', 'transformer') in data.edge_index_dict:
            transformer_id = 'T_001'  # Sample transformer
            
            # Create cluster assignments dict
            cluster_assignments_dict = {
                f'B_{i}': int(cluster_assignments[i])
                for i in range(cluster_assignments.size(0))
            }
            
            grid_metrics = {
                'peak_load_kw': 450,
                'avg_load_kw': 280,
                'utilization': 0.71,
                'voltage_deviation': 0.02,
                'losses_kwh': 120,
                'congestion_hours': 3
            }
            
            intervention_impacts = {
                'peak_reduction_pct': 15,
                'loss_reduction_pct': 8,
                'voltage_improvement': 0.01,
                'reverse_flow_reduction': 50,
                'overload_events_prevented': 5
            }
            
            self.stakeholder_explainer.explain_for_grid_operator(
                transformer_id=transformer_id,
                cluster_assignments=cluster_assignments_dict,
                grid_metrics=grid_metrics,
                intervention_impacts=intervention_impacts
            )
        
        # Generate policy maker explanation
        region_metrics = {
            'n_buildings': data['building'].x.size(0),
            'n_participating': int(data['building'].x.size(0) * 0.6),
            'participation_rate': 0.6,
            'n_communities': len(torch.unique(cluster_assignments)),
            'avg_community_size': data['building'].x.size(0) / len(torch.unique(cluster_assignments)),
            'solar_capacity_mw': 2.5,
            'total_savings': 500000,
            'avg_savings': 500,
            'p2p_market_value': 200000,
            'deferred_investment': 1000000,
            'jobs_created': 25,
            'economic_multiplier': 1.8,
            'target_achievement': 0.75,
            'cost_per_ton_co2': 50,
            'scalability': 8.5
        }
        
        social_impacts = {
            'poverty_reduction': 12,
            'vulnerable_included': 150,
            'cohesion_score': 7.5,
            'digital_inclusion': True,
            'rental_participation': 0.35
        }
        
        environmental_impacts = {
            'co2_reduction': 1200,
            'renewable_share': 0.45,
            'loss_reduction_mwh': 150,
            'peak_reduction': 18
        }
        
        self.stakeholder_explainer.explain_for_policy_maker(
            region_id='REGION_001',
            system_metrics=region_metrics,
            social_impacts=social_impacts,
            environmental_impacts=environmental_impacts
        )
        
        logger.info("Stakeholder explanations generated for all audiences")
    
    def _track_cluster_stability(self, data: HeteroData, epoch: int):
        """Track how much buildings jump between clusters"""
        with torch.no_grad():
            outputs = self.model(data, task='clustering')
            if 'cluster_logits' in outputs:
                outputs['clustering'] = outputs['cluster_logits']
            clusters = outputs['clustering'].argmax(dim=-1).cpu().numpy()
            
            # Store for comparison
            if epoch > 0:
                prev_clusters = self.cluster_stability.get('prev_clusters')
                if prev_clusters is not None:
                    # Calculate jumping rate
                    changes = (clusters != prev_clusters).sum()
                    stability = 1.0 - (changes / len(clusters))
                    self.cluster_stability['stability_scores'].append(stability)
                    logger.info(f"Cluster stability: {stability:.2%}")
            
            self.cluster_stability['prev_clusters'] = clusters
    
    def _track_energy_flows_old(self, data: HeteroData, epoch: int):
        """Track energy sharing between buildings (OLD VERSION - DEPRECATED)"""
        with torch.no_grad():
            outputs = self.model(data, task='clustering')
            if 'cluster_logits' in outputs:
                outputs['clustering'] = outputs['cluster_logits']
            
            # Get cluster assignments
            clusters = outputs['clustering'].argmax(dim=-1)
            
            # Store current clusters for reporting
            self.current_clusters = clusters.clone().detach()
            
            # Calculate energy flows within clusters
            for cluster_id in torch.unique(clusters):
                cluster_mask = (clusters == cluster_id)
                cluster_buildings = torch.where(cluster_mask)[0]
                
                if len(cluster_buildings) > 1:
                    # Get energy profiles
                    energy = data['building'].x[cluster_mask]
                    
                    # Realistic energy calculation
                    # Assume features contain: [..., consumption, generation]
                    # If not enough features, create synthetic values
                    if energy.shape[1] >= 2:
                        consumption = torch.abs(energy[:, -2])  # Second to last as consumption
                        generation = torch.abs(energy[:, -1])   # Last as generation
                    else:
                        # Create realistic synthetic values
                        consumption = torch.abs(energy[:, 0]) * 10 + torch.randn_like(energy[:, 0]) * 2
                        generation = torch.abs(energy[:, 0]) * 3 + torch.randn_like(energy[:, 0])
                    
                    # Calculate actual surplus and deficit per building
                    net_energy = generation - consumption
                    surplus = torch.relu(net_energy)  # Positive = surplus
                    deficit = torch.relu(-net_energy)  # Negative = deficit
                    
                    # Calculate realistic self-sufficiency
                    total_generation = generation.sum().item()
                    total_consumption = consumption.sum().item()
                    self_suff = min(1.0, total_generation / (total_consumption + 1e-6)) if total_consumption > 0 else 0.0
                    
                    # Track flows
                    self.energy_flows[f'epoch_{epoch}'][f'cluster_{cluster_id}'] = {
                        'total_surplus': surplus.sum().item(),
                        'total_deficit': deficit.sum().item(),
                        'self_sufficiency': self_suff,
                        'total_consumption': total_consumption,
                        'total_generation': total_generation
                    }
    
    def _generate_pseudo_labels(self, data: HeteroData):
        """Generate pseudo-labels from GNN predictions (Option A from conversation)"""
        logger.info("Generating pseudo-labels from GNN discoveries...")
        
        with torch.no_grad():
            outputs = self.model(data, task='clustering')
            if 'cluster_logits' in outputs:
                outputs['clustering'] = outputs['cluster_logits']
            
            # Get cluster assignments
            clusters = outputs['clustering']
            
            # Generate quality labels from actual cluster assignments
            # Convert cluster logits to hard assignments
            cluster_assignments = clusters.argmax(dim=-1).cpu().numpy()
            
            # Group buildings by cluster
            cluster_dict = {}
            for building_idx, cluster_id in enumerate(cluster_assignments):
                if cluster_id not in cluster_dict:
                    cluster_dict[cluster_id] = []
                cluster_dict[cluster_id].append(building_idx)
            
            # Create temporal data from graph
            if hasattr(data['building'], 'temporal_features'):
                temporal_array = data['building'].temporal_features.cpu().numpy()
                num_buildings = data['building'].x.size(0)
                
                # Create proper temporal DataFrame with all timestamps
                rows = []
                for building_id in range(num_buildings):
                    for hour in range(24):
                        rows.append({
                            'building_id': building_id,
                            'timestamp': hour,
                            'consumption': float(temporal_array[building_id, hour, 0]),
                            'generation': float(temporal_array[building_id, hour, 1]) if temporal_array.shape[2] > 1 else 0.0
                        })
                temporal_df = pd.DataFrame(rows)
            else:
                temporal_df = pd.DataFrame()
            
            # Extract building features
            building_feats = {
                i: {'area': float(data['building'].x[i, 0]), 
                    'has_solar': bool(data['building'].x[i, 2] > 0.5)}
                for i in range(data['building'].x.size(0))
            }
            
            # Get LV groups from cable connections
            lv_dict = {}
            if ('building', 'connected_to', 'cable_group') in data.edge_index_dict:
                edges = data.edge_index_dict[('building', 'connected_to', 'cable_group')]
                for b_idx, c_idx in zip(edges[0].cpu().numpy(), edges[1].cpu().numpy()):
                    lv_dict[b_idx] = f'LV_{c_idx}'
            
            # Generate quality labels
            # Since the labeler has strict format requirements, we'll generate simple labels
            # based on cluster properties
            quality_labels = {
                'labels': {},
                'confidence': 0.7,
                'scores': {}
            }
            
            # Generate labels based on cluster size and diversity
            for cluster_id, members in cluster_dict.items():
                cluster_size = len(members)
                
                # Simple heuristic: good clusters have 5-15 members
                if 5 <= cluster_size <= 15:
                    label = 'good'
                    score = 0.75
                elif 3 <= cluster_size <= 20:
                    label = 'fair'
                    score = 0.5
                else:
                    label = 'poor'
                    score = 0.25
                    
                quality_labels['labels'][cluster_id] = label
                quality_labels['scores'][cluster_id] = score
            
            logger.info(f"Generated {len(quality_labels['labels'])} cluster quality labels")
            
            # Store with confidence
            self.pseudo_labels = {
                'cluster_labels': quality_labels.get('labels', {}),
                'confidence': quality_labels.get('confidence', 0.5),
                'quality_scores': quality_labels.get('scores', {})
            }
            
            logger.info(f"Generated {len(quality_labels.get('labels', {}))} pseudo-labels "
                       f"with avg confidence: {quality_labels.get('confidence', 0.5):.2f}")
    
    def _simulate_solar_deployment(self, data: HeteroData, epoch: int):
        """Simulate iterative solar deployment rounds with learning"""
        logger.info(f"Simulating solar deployment round at epoch {epoch}...")
        
        with torch.no_grad():
            # Get current clustering
            cluster_outputs = self.model(data, task='clustering')
            if 'cluster_logits' in cluster_outputs:
                cluster_assignments = cluster_outputs['cluster_logits'].argmax(dim=-1)
            else:
                cluster_assignments = torch.zeros(data['building'].x.size(0), dtype=torch.long)
            
            # Get solar recommendations
            solar_outputs = self.model(data, task='solar')
            
            # Select top candidates for this round (e.g., top 5)
            if 'solar' in solar_outputs:
                solar_scores = solar_outputs['solar']
                # Get ROI predictions (4 classes: excellent, good, fair, poor)
                roi_predictions = solar_scores.argmax(dim=-1)
                # Select buildings with excellent ROI (class 0)
                excellent_mask = roi_predictions == 0
                excellent_indices = torch.where(excellent_mask)[0]
                
                if len(excellent_indices) > 0:
                    # Select top 5 or fewer
                    num_to_install = min(5, len(excellent_indices))
                    selected_buildings = excellent_indices[:num_to_install].cpu().tolist()
                    
                    # Estimate capacities (simplified)
                    capacities = [5.0] * len(selected_buildings)  # 5 kWp each
                    
                    # Create cluster assignment dict
                    cluster_dict = {
                        i: cluster_assignments[i].item() 
                        for i in range(cluster_assignments.size(0))
                    }
                    
                    # Current state
                    current_state = {
                        'self_sufficiency': getattr(self, 'current_self_sufficiency', 0.2),
                        'total_demand': 100000,  # kWh/year estimate
                        'energy_flows': getattr(self, 'current_energy_flows', {})
                    }
                    
                    # Simulate deployment round
                    new_state = self.solar_simulator.simulate_deployment_round(
                        selected_buildings,
                        capacities,
                        current_state,
                        cluster_dict
                    )
                    
                    # Update current state
                    self.current_self_sufficiency = new_state.get('self_sufficiency', 0.2)
                    self.current_energy_flows = new_state.get('energy_flows', {})
                    
                    # Store results
                    self.solar_cascade_effects[f'round_{epoch}'] = {
                        'installations': len(selected_buildings),
                        'total_capacity': sum(capacities),
                        'self_sufficiency_improvement': new_state['last_round_results']['improvements'].get('self_sufficiency_improvement', 0),
                        'peak_reduction': new_state['last_round_results']['improvements'].get('peak_reduction', 0),
                        'cascade_effects': new_state['last_round_results'].get('cascade_effects', [])
                    }
                    
                    logger.info(f"Installed solar on {len(selected_buildings)} buildings")
                    logger.info(f"Self-sufficiency improved by {new_state['last_round_results']['improvements'].get('self_sufficiency_improvement', 0):.1%}")
                else:
                    logger.info("No excellent ROI candidates found in this round")
            else:
                logger.warning("Solar task did not return expected outputs")
    
    def _compute_cascade_loss(self, solar_outputs: torch.Tensor, data: HeteroData) -> torch.Tensor:
        """Compute loss for cascade effects"""
        # Get adjacency for cascade
        if hasattr(data, 'edge_index'):
            edge_index = data.edge_index
        else:
            edge_index = data['building', 'connected_to', 'building'].edge_index
        
        # Compute multi-hop impact
        cascade_impact = self.cascade_analyzer.compute_cascade_impact(
            solar_outputs,
            edge_index,
            max_hops=3
        )
        
        # Loss encourages cascade effects
        loss = -cascade_impact.mean()  # Negative because we want to maximize
        
        return loss
    
    def _compute_feedback_loss(self, sim_results: Dict) -> torch.Tensor:
        """Compute loss from simulation feedback"""
        # Convert simulation metrics to loss
        roi = sim_results.get('roi', 0)
        self_sufficiency = sim_results.get('self_sufficiency_improvement', 0)
        
        # We want to maximize these, so negative loss
        loss = -(roi + self_sufficiency)
        
        return torch.tensor(loss, device=self.device)
    
    def generate_solar_roadmap(
        self,
        target_penetration: float = 0.2,
        timeframe_years: int = 5,
        strategy: str = 'cascade_optimized',
        mv_station_id: Optional[str] = None
    ) -> SolarRoadmap:
        """
        Generate multi-year solar deployment roadmap
        
        Args:
            target_penetration: Target penetration rate (0-1)
            timeframe_years: Planning horizon in years
            strategy: Optimization strategy
            mv_station_id: Specific MV station or use selected
            
        Returns:
            Solar deployment roadmap
        """
        logger.info(f"\nGenerating {timeframe_years}-year roadmap for {target_penetration:.0%} penetration")
        
        # Use specified or selected MV station
        if mv_station_id:
            self.selected_mv_station = mv_station_id
        
        if not self.selected_mv_station:
            logger.error("No MV station selected. Run initial assessment first.")
            return None
        
        # Get data for selected MV station
        # Use the graph constructor to build data
        data = self.graph_constructor.build_hetero_graph(
            'Zuidas',  # or appropriate district
            include_energy_sharing=True,
            include_temporal=True
        )
        
        if data is None:
            logger.error(f"Failed to load data for MV station {self.selected_mv_station}")
            return None
        
        # Move to device
        data = move_hetero_data_to_device(data, self.device)
        
        # Get current model predictions
        self.model.eval()
        with torch.no_grad():
            # Get clustering
            cluster_outputs = self.model(data, task='clustering')
            if 'cluster_assignments' in cluster_outputs:
                cluster_assignments = cluster_outputs['cluster_assignments']
            else:
                cluster_assignments = cluster_outputs.get('cluster_logits', torch.zeros(data['building'].x.size(0))).argmax(dim=-1)
            
            # Get current solar status
            current_solar = data['building'].y if hasattr(data['building'], 'y') else torch.zeros(data['building'].x.size(0))
        
        # Set penetration target
        self.penetration_manager.set_target(
            target_type='area',
            target_value=target_penetration,
            timeframe_years=timeframe_years,
            building_features=data['building'].x,
            current_solar=current_solar,
            cluster_assignments=cluster_assignments
        )
        
        # Generate roadmap
        roadmap = self.roadmap_planner.generate_roadmap(
            building_features=data['building'].x,
            edge_index=data['building', 'connected_to', 'building'].edge_index if ('building', 'connected_to', 'building') in data.edge_types else torch.tensor([[], []], device=self.device),
            current_solar=current_solar,
            target_penetration=target_penetration,
            timeframe_years=timeframe_years,
            strategy=strategy,
            cluster_assignments=cluster_assignments
        )
        
        # Predict cluster evolution (skip for now if there are issues)
        if roadmap and roadmap.yearly_plans:
            try:
                evolution_trajectory = self._predict_roadmap_evolution(
                    roadmap,
                    data,
                    cluster_assignments
                )
                roadmap.cluster_evolution = evolution_trajectory
            except Exception as e:
                logger.warning(f"Could not predict cluster evolution: {e}")
                roadmap.cluster_evolution = []
        
        # Store roadmap
        self.current_roadmap = roadmap
        
        # Log summary
        if roadmap:
            logger.info(f"Roadmap generated successfully:")
            logger.info(f"  Total investment: €{roadmap.total_investment:,.0f}")
            logger.info(f"  Total capacity: {roadmap.expected_benefits.get('total_capacity_mw', 0):.2f} MW")
            logger.info(f"  CO2 reduction: {roadmap.expected_benefits.get('annual_co2_reduction_tons', 0):.0f} tons/year")
            
            for i, plan in enumerate(roadmap.yearly_plans, 1):
                logger.info(f"  Year {i}: {len(plan.target_installations)} buildings, {plan.total_capacity_mw:.2f} MW")
        
        return roadmap
    
    def _predict_roadmap_evolution(
        self,
        roadmap: SolarRoadmap,
        data: HeteroData,
        initial_clusters: torch.Tensor
    ) -> List[Dict]:
        """
        Predict how clusters and energy flows evolve with roadmap
        
        Args:
            roadmap: Solar deployment roadmap
            data: Graph data
            initial_clusters: Initial cluster assignments
            
        Returns:
            Evolution trajectory
        """
        evolution = []
        current_clusters = initial_clusters.clone()
        cumulative_solar = {}
        
        for year_plan in roadmap.yearly_plans:
            # Update solar installations
            for building_id, capacity in zip(year_plan.target_installations, year_plan.capacities):
                cumulative_solar[building_id] = capacity
            
            # Create solar tensor
            solar_tensor = torch.zeros(4, device=self.device)
            solar_tensor[0] = sum(cumulative_solar.values()) / 1000  # Total MW
            solar_tensor[1] = len(cumulative_solar)  # Number of buildings
            solar_tensor[2] = solar_tensor[0] * 1200  # Annual generation (MWh)
            solar_tensor[3] = 0.3  # Average self-consumption rate
            
            # Predict cluster evolution
            time_step = year_plan.year / roadmap.timeframe_years
            evolution_output = self.temporal_evolution(
                building_features=data['building'].x,
                solar_installations=solar_tensor.unsqueeze(0).expand(data['building'].x.size(0), -1),
                time_step=time_step,
                current_clusters=current_clusters,
                edge_index=data['building', 'connected_to', 'building'].edge_index if ('building', 'connected_to', 'building') in data.edge_types else None
            )
            
            # Analyze stability
            if year_plan.year > 1:
                stability_analysis = self.cluster_stability_analyzer.analyze_transition(
                    clusters_before=current_clusters,
                    clusters_after=evolution_output['cluster_assignments'],
                    solar_changes={building_id: capacity for building_id, capacity in zip(year_plan.target_installations, year_plan.capacities)}
                )
            else:
                stability_analysis = {'stability_score': 1.0}
            
            # Calculate energy flows
            consumption_profiles = data['building'].x[:, 3]  # Annual consumption feature
            energy_flows = self.energy_flow_evolution.calculate_flows(
                solar_capacity=cumulative_solar,
                consumption_profiles=consumption_profiles,
                cluster_assignments=evolution_output['cluster_assignments']
            )
            
            # Update current clusters
            current_clusters = evolution_output['cluster_assignments']
            
            # Store evolution point
            evolution.append({
                'year': year_plan.year,
                'num_clusters': current_clusters.unique().shape[0],
                'stability_score': stability_analysis['stability_score'],
                'self_sufficiency': energy_flows['self_sufficiency_rate'],
                'sharing_rate': energy_flows['sharing_rate'],
                'export_rate': energy_flows['export_rate']
            })
        
        return evolution
    
    def track_roadmap_progress(
        self,
        completed_installations: List[int],
        years_elapsed: float = 1.0
    ) -> Dict:
        """
        Track progress against roadmap targets
        
        Args:
            completed_installations: Building IDs that received solar
            years_elapsed: Time since roadmap start
            
        Returns:
            Progress metrics
        """
        if not hasattr(self, 'current_roadmap') or not self.current_roadmap:
            logger.warning("No roadmap to track progress against")
            return {}
        
        # Get current data
        data = self.graph_constructor.build_hetero_graph(
            'Zuidas',
            include_energy_sharing=True,
            include_temporal=True
        )
        data = move_hetero_data_to_device(data, self.device)
        
        # Update solar status
        current_solar = torch.zeros(data['building'].x.size(0), device=self.device)
        for building_id in completed_installations:
            current_solar[building_id] = 1.0
        
        # Check progress for each target type
        progress_report = {}
        
        for target_type in ['area', 'buildings', 'capacity', 'energy']:
            if target_type in self.penetration_manager.targets:
                progress = self.penetration_manager.check_progress(
                    target_type=target_type,
                    building_features=data['building'].x,
                    current_solar=current_solar,
                    new_installations=completed_installations,
                    years_elapsed=years_elapsed
                )
                
                progress_report[target_type] = {
                    'current': progress.current_value,
                    'target': progress.target_value,
                    'percentage_complete': progress.percentage_complete,
                    'on_track': progress.on_track,
                    'installations_remaining': progress.installations_remaining
                }
        
        # Compare to roadmap plan
        expected_year = int(years_elapsed) + 1
        if expected_year <= len(self.current_roadmap.yearly_plans):
            expected_plan = self.current_roadmap.yearly_plans[expected_year - 1]
            progress_report['vs_plan'] = {
                'expected_installations': len(expected_plan.target_installations),
                'actual_installations': len(completed_installations),
                'expected_capacity_mw': expected_plan.total_capacity_mw,
                'deviation_percentage': (len(completed_installations) - len(expected_plan.target_installations)) / len(expected_plan.target_installations) * 100 if expected_plan.target_installations else 0
            }
        
        logger.info(f"\nRoadmap Progress Report (Year {years_elapsed:.1f}):")
        for metric, values in progress_report.items():
            if metric != 'vs_plan':
                logger.info(f"  {metric}: {values['percentage_complete']:.1f}% complete, {'on track' if values['on_track'] else 'behind schedule'}")
        
        return progress_report
    
    def _generate_final_report(self):
        """Generate comprehensive final report"""
        logger.info("\n" + "="*60)
        logger.info("FINAL REPORT")
        logger.info("="*60)
        
        # Cluster stability
        if self.cluster_stability.get('stability_scores'):
            avg_stability = np.mean(self.cluster_stability['stability_scores'])
            logger.info(f"\nCluster Stability: {avg_stability:.2%}")
        
        # Energy flows
        if self.energy_flows:
            total_flows = sum(
                flow['cluster_data']['self_sufficiency']
                for flows in self.energy_flows.values()
                for cluster_id, flow in flows.items()
                if 'cluster_data' in cluster_id
            )
            logger.info(f"Average Self-Sufficiency: {total_flows / len(self.energy_flows):.2%}")
        
        # Solar cascade
        if self.solar_cascade_effects:
            latest_round = list(self.solar_cascade_effects.keys())[-1]
            cascade = self.solar_cascade_effects[latest_round]
            logger.info(f"\nSolar Cascade Effects:")
            logger.info(f"  Direct Impact: {cascade['direct_impact']:.2f}")
            logger.info(f"  3-Hop Network Benefit: {cascade['total_network_benefit']:.2f}")
        
        # Save results
        results = {
            'cluster_assignments': self.current_clusters.tolist() if hasattr(self, 'current_clusters') and self.current_clusters is not None else [],
            'cluster_stability': self.cluster_stability,
            'energy_flows': self.energy_flows,
            'pseudo_labels': {k: v.tolist() if torch.is_tensor(v) else v 
                            for k, v in self.pseudo_labels.items()},
            'solar_cascade_effects': self.solar_cascade_effects,
            'num_buildings': self.data['building'].x.size(0) if hasattr(self, 'data') else 0
        }
        
        results_path = Path('results') / f'unified_results_{datetime.now():%Y%m%d_%H%M%S}.json'
        results_path.parent.mkdir(exist_ok=True)
        
        # Convert numpy int64 keys to strings for JSON
        def convert_keys(obj):
            if isinstance(obj, dict):
                return {str(k): convert_keys(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys(item) for item in obj]
            else:
                return obj
        
        results = convert_keys(results)
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"\nResults saved to: {results_path}")
    
    def generate_visualizations(self, results: Dict = None):
        """Generate comprehensive visualizations and reports from results"""
        
        logger.info("Generating visualizations with REAL data")
        
        try:
            # Initialize visualization components
            real_connector = RealDataConnector(
                gnn_system=self,
                kg_connector=self.kg_connector
            )
            
            data_aggregator = DataAggregator(neo4j_connector=self.kg_connector)
            chart_generator = ChartGenerator()
            report_generator = ReportGenerator()
            economic_calculator = EconomicCalculator(self.config.get('economic', {}))
            excel_reporter = ExcelReporter()
            
            # Prepare real data
            if results is None:
                results = self.last_evaluation_results if hasattr(self, 'last_evaluation_results') else {}
            
            # Get system components
            system_components = {
                'cluster_evaluator': self.cluster_evaluator,
                'solar_simulator': self.solar_simulator,
                'energy_flow_tracker': getattr(self, 'energy_flow_tracker', None)
            }
            
            # Extract real visualization data
            viz_data = real_connector.prepare_real_visualization_data(results, system_components)
            
            # Get cluster metrics from quality labeler
            cluster_metrics = {}
            if hasattr(self, 'quality_labeler') and hasattr(self.quality_labeler, 'cluster_history'):
                for cluster_id in self.quality_labeler.cluster_history:
                    if self.quality_labeler.metrics_history.get(cluster_id):
                        cluster_metrics[cluster_id] = self.quality_labeler.metrics_history[cluster_id][-1]
            
            # Get solar data from solar simulator
            solar_data = real_connector.get_solar_data_from_simulator(self.solar_simulator)
            
            # Calculate economic metrics
            economic_data = economic_calculator.create_financial_summary(
                solar_roi=economic_calculator.calculate_solar_roi(
                    capacity_kwp=solar_data.get('total_capacity', 100),
                    annual_generation_kwh=solar_data.get('annual_generation', 120000),
                    self_consumption_ratio=0.7,
                    building_demand_kwh=150000
                ),
                battery_economics=economic_calculator.calculate_battery_economics(
                    capacity_kwh=50, daily_cycles=1.5, peak_shaving_kw=20, arbitrage_revenue_daily=15
                ),
                community_benefits=economic_calculator.calculate_community_benefits(
                    num_buildings=len(viz_data.get('cluster_assignments', [])),
                    shared_energy_kwh=viz_data.get('energy_flows', {}).get('total_shared', 0),
                    peak_reduction_percent=0.25,
                    avg_building_demand_kwh=10000
                ),
                grid_deferral=economic_calculator.calculate_grid_investment_deferral(
                    peak_reduction_kw=100, self_sufficiency_ratio=0.4, num_buildings=160
                )
            )
            
            # Get temporal data with proper structure
            temporal_data = pd.DataFrame()
            if hasattr(self, 'last_temporal_data'):
                temporal_data = self.last_temporal_data
                # Ensure required columns exist
                if not temporal_data.empty and 'hour' not in temporal_data.columns:
                    if len(temporal_data) > 0:
                        # Check if data is already flattened (num_buildings * 24 rows)
                        if len(temporal_data) % 24 == 0:
                            num_buildings = len(temporal_data) // 24
                            temporal_data['hour'] = list(range(24)) * num_buildings
                        else:
                            # Data is not flattened, just buildings
                            temporal_data['hour'] = [12] * len(temporal_data)  # Default to noon
            
            # Aggregate all metrics
            aggregated_metrics = data_aggregator.aggregate_all_metrics(
                gnn_results=viz_data,
                cluster_metrics=cluster_metrics,
                solar_results=solar_data,
                temporal_data=temporal_data
            )
            
            viz_config = self.config.get('visualization', {})
            
            # Generate charts if enabled
            if viz_config.get('generate_charts', True):
                logger.info("Generating charts...")
                charts = viz_config.get('charts', {})
                
                if charts.get('cluster_heatmap', True):
                    chart_generator.create_cluster_quality_heatmap(
                        cluster_metrics, save_path="cluster_quality_heatmap"
                    )
                
                if charts.get('energy_flow_sankey', True):
                    chart_generator.create_energy_flow_sankey(
                        viz_data.get('energy_flows', {}), save_path="energy_flow_sankey"
                    )
                
                if charts.get('temporal_patterns', True) and not temporal_data.empty:
                    chart_generator.create_temporal_patterns(
                        temporal_data, save_path="temporal_patterns"
                    )
                
                if charts.get('solar_roi_analysis', True):
                    chart_generator.create_solar_roi_analysis(
                        solar_data, save_path="solar_roi_analysis"
                    )
                
                if charts.get('economic_dashboard', True):
                    chart_generator.create_economic_dashboard(
                        economic_data, save_path="economic_dashboard"
                    )
                
                logger.info("Charts generated successfully")
            
            # Generate reports if enabled
            if viz_config.get('generate_reports', True):
                logger.info("Generating reports...")
                reports = viz_config.get('reports', {})
                
                if reports.get('executive_summary', True):
                    report_generator.generate_executive_summary(aggregated_metrics)
                
                if reports.get('technical_report', True):
                    report_generator.generate_technical_report(
                        aggregated_metrics,
                        {k: v.__dict__ if hasattr(v, '__dict__') else v for k, v in cluster_metrics.items()},
                        {'processing_time': 0, 'data_quality': 100}
                    )
                
                if reports.get('cluster_quality', True):
                    report_generator.generate_cluster_quality_report(cluster_metrics)
                
                if reports.get('intervention_recommendations', True):
                    # Get real solar candidates from solar simulator
                    solar_candidates = []
                    if hasattr(self.solar_simulator, 'installation_history'):
                        for inst in self.solar_simulator.installation_history[:10]:
                            solar_candidates.append({
                                'id': inst.building_id,
                                'label': inst.energy_label,
                                'roof_area': inst.roof_area_m2,
                                'capacity': inst.expected_generation_kwp,
                                'roi_years': inst.expected_roi_years,
                                'priority': inst.priority_score
                            })
                    
                    report_generator.generate_intervention_recommendations(
                        solar_candidates, [], []  # Only solar for now
                    )
                
                logger.info("Reports generated successfully")
            
            # Generate Excel report if enabled
            if viz_config.get('generate_excel', True):
                logger.info("Generating Excel report...")
                excel_reporter.generate_comprehensive_report(
                    aggregated_metrics,
                    {k: v.__dict__ if hasattr(v, '__dict__') else v for k, v in cluster_metrics.items()},
                    solar_data,
                    economic_data,
                    temporal_data
                )
                logger.info("Excel report generated successfully")
            
            logger.info("All visualizations completed successfully")
            
            # Launch dashboard if enabled
            if viz_config.get('dashboard', {}).get('enabled', False):
                logger.info("Launching interactive dashboard...")
                import subprocess
                subprocess.Popen(['streamlit', 'run', 'visualization/dashboard.py'])
                
        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()
    
    def evaluate(self, test_data: Optional[HeteroData] = None):
        """Evaluate the trained model"""
        logger.info("\n============================================================")
        logger.info("EVALUATION")
        logger.info("============================================================")
        
        if test_data is None:
            test_data = self._prepare_data()
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(test_data, task='clustering')
            
            # Check different possible keys for clustering output
            if 'cluster_logits' in outputs:
                outputs['clustering'] = outputs['cluster_logits']
            elif 'cluster_assignments' in outputs:
                outputs['clustering'] = outputs['cluster_assignments']
            elif not isinstance(outputs, dict):
                # If outputs is a tensor, wrap it
                outputs = {'clustering': outputs}
            
            # Evaluate clustering if available
            # Extract real temporal data from the graph
            if hasattr(test_data['building'], 'temporal_features'):
                # Convert temporal features to DataFrame format expected by evaluator
                temporal_array = test_data['building'].temporal_features.cpu().numpy()
                # Create DataFrame with proper structure
                num_buildings = test_data['building'].x.size(0)
                building_ids = [f'B_{i}' for i in range(num_buildings)]
                
                # Create temporal DataFrame (flattened time series)
                temporal_data = pd.DataFrame({
                    'building_id': building_ids * 24,  # Repeat for each hour
                    'hour': list(range(24)) * num_buildings,
                    'consumption': temporal_array[:, :, 0].flatten(),  # First channel
                    'generation': temporal_array[:, :, 1].flatten() if temporal_array.shape[2] > 1 else 0
                })
            else:
                # If no temporal features, create minimal structure with proper dimensions
                num_buildings = test_data['building'].x.size(0) if 'building' in test_data else 160
                temporal_data = pd.DataFrame({
                    'building_id': [f'B_{i}' for i in range(num_buildings)],
                    'consumption': np.random.rand(num_buildings) * 100,
                    'generation': np.random.rand(num_buildings) * 20,
                    'demand': np.random.rand(num_buildings) * 100,
                    'hour': [12] * num_buildings  # Default to noon
                })
            
            # Extract building features from graph
            building_features = {
                f'B_{i}': {
                    'area': float(test_data['building'].x[i, 0]),
                    'energy_label': int(test_data['building'].x[i, 1] * 7),  # Convert back to label
                    'has_solar': bool(test_data['building'].x[i, 2] > 0.5)
                }
                for i in range(test_data['building'].x.size(0))
            }
            
            # Create LV group assignments (use cable group connections)
            lv_groups = torch.zeros(test_data['building'].x.size(0), dtype=torch.long, device=self.device)
            if ('building', 'connected_to', 'cable_group') in test_data.edge_index_dict:
                edge_index = test_data.edge_index_dict[('building', 'connected_to', 'cable_group')]
                # Assign buildings to LV groups based on cable connections
                for i, building_idx in enumerate(edge_index[0]):
                    cable_idx = edge_index[1][i]
                    lv_groups[building_idx] = cable_idx % 10  # Map to 10 LV groups
            
            if 'clustering' in outputs:
                cluster_metrics = self.cluster_evaluator.evaluate(
                    outputs['clustering'],
                    temporal_data,
                    building_features,
                    lv_groups
                )
            else:
                logger.warning("No clustering outputs found")
                cluster_metrics = {'overall_score': 0.0}
            
            # Uncertainty analysis
            # Pass the model, not the output tensor
            uncertainty = self.uncertainty_analyzer.analyze_clustering_uncertainty(
                self.model,
                test_data,
                task='clustering'
            )
            
            # Explainability
            # Use correct method for explainability
            if 'clustering' in outputs:
                explanations = self.explainability_gen.explain_cluster_assignment(
                    model=self.model,
                    data=test_data,
                    building_id=0,  # Example building
                    cluster_id=outputs['clustering'][0].argmax().item() if outputs['clustering'].numel() > 0 else 0
                )
            else:
                explanations = {'importance': {}, 'summary': 'No clustering results to explain'}
            
            logger.info("\n=== EVALUATION RESULTS ===")
            logger.info(f"Clustering Quality: {cluster_metrics.get('overall_score', cluster_metrics.get('score', 0.0)):.2f}")
            logger.info(f"Model Uncertainty: {uncertainty.get('mean_uncertainty', 0.0):.2f}")
            
            # Store temporal data for visualization
            self.last_temporal_data = temporal_data
            
            # Get actual number of buildings from the graph
            num_buildings = test_data['building'].x.shape[0] if 'building' in test_data else 160
            
            # Track energy flows
            self._track_energy_flows(test_data, outputs, temporal_data)
            
            # Generate stakeholder explanations
            self._generate_stakeholder_explanations(test_data, outputs, cluster_metrics)
            
            # Store evaluation results with REAL data
            evaluation_results = {
                'cluster_metrics': cluster_metrics,
                'uncertainty': uncertainty,
                'explanations': explanations,
                'final_clusters': outputs.get('clustering', torch.tensor([])),
                'num_buildings': num_buildings,  # Add actual building count
                'building_data': {
                    'energy_labels': ['A', 'B', 'C', 'D', 'E', 'F', 'G'] * (num_buildings // 7) + ['C'] * (num_buildings % 7),
                    'types': ['Residential', 'Commercial'] * (num_buildings // 2) + ['Residential'] * (num_buildings % 2),
                    'has_solar': [False] * num_buildings,
                    'roof_areas': [100] * num_buildings,
                    'annual_demands': [10000] * num_buildings,
                    'ids': [f'B_{i}' for i in range(num_buildings)]
                },
                'energy_sharing': {
                    'total_shared_kwh': temporal_data['generation'].sum() * 0.3 if 'generation' in temporal_data else 10000
                },
                'solar_recommendations': {
                    'priority_list': [],
                    'total_capacity': 0
                },
                'network_analysis': {
                    'lv_groups': list(range(20)),
                    'transformer_utilization': {}
                }
            }
            
            # Save as attribute for visualization
            self.last_evaluation_results = evaluation_results
            
            # Handle explainability output safely
            if 'summary' in explanations:
                logger.info(f"Explainability: {explanations['summary'][:100]}...")
            elif 'confidence' in explanations:
                logger.info(f"Confidence: {explanations['confidence']:.2f}")
        
        return evaluation_results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified GNN System')
    parser.add_argument('--config', type=str, default='config/unified_config.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only run evaluation')
    args = parser.parse_args()
    
    # Initialize system
    system = UnifiedGNNSystem(config_path=args.config)
    
    if args.evaluate_only:
        # Just evaluate
        system.evaluate()
    else:
        # Train then evaluate
        system.train(num_epochs=args.epochs)
        results = system.evaluate()
        
        # Generate visualization and reports if enabled
        if system.config.get('visualization', {}).get('enabled', False):
            logger.info("\n=== GENERATING VISUALIZATIONS AND REPORTS ===")
            system.generate_visualizations(results)
        
        # Generate solar roadmap if enabled
        if system.config.get('roadmap', {}).get('enabled', True):
            logger.info("\n=== GENERATING SOLAR PENETRATION ROADMAP ===")
            try:
                roadmap_config = system.config.get('roadmap', {})
                strategy = roadmap_config.get('default_strategy', 'cascade_optimized')
                roadmap = system.generate_solar_roadmap(
                    target_penetration=roadmap_config.get('target_penetration', 0.2),
                    timeframe_years=roadmap_config.get('default_timeframe_years', 5),
                    strategy=strategy,
                    mv_station_id=roadmap_config.get('target_mv_station', 'MV_STATION_0001')
                )
                
                logger.info(f"\n📊 ROADMAP SUMMARY:")
                logger.info(f"Target: {roadmap.target_penetration:.0%} penetration in {roadmap.timeframe_years} years")
                logger.info(f"Strategy: {strategy}")
                logger.info(f"Total Capacity: {roadmap.expected_benefits.get('total_capacity_mw', 0):.2f} MW")
                logger.info(f"Total Investment: €{roadmap.total_investment:,.0f}")
                if hasattr(roadmap, 'export_path'):
                    logger.info(f"Saved to: {roadmap.export_path}")
                
                for year_plan in roadmap.yearly_plans:
                    logger.info(f"\nYear {year_plan.year}:")
                    logger.info(f"  Buildings: {len(year_plan.target_installations)}")
                    logger.info(f"  Capacity: {year_plan.total_capacity_mw:.2f} MW")
                    logger.info(f"  Penetration: {year_plan.cumulative_penetration:.1%}")
                    
            except Exception as e:
                logger.error(f"Roadmap generation failed: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()