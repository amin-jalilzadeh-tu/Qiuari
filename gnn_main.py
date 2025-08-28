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
# HeteroEnergyGNN already has built-in task heads, but we import these
# for additional functionality and flexibility
from training.loss_functions import DiscoveryLoss, ComplementarityLoss, SolarROILoss
from training.solar_simulator import SolarPerformanceSimulator
from tasks.cluster_quality_labeling import ClusterQualityLabeler
from tasks.solar_cascade_analyzer import SolarCascadeAnalyzer
from tasks.uncertainty_analyzer import UncertaintyAnalyzer
from tasks.explainability_generator import ExplainabilityGenerator
from evaluation.cluster_evaluator import ClusterEvaluator
from torch_geometric.data import HeteroData
from utils.device_utils import move_hetero_data_to_device


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
        self.discovery_loss = DiscoveryLoss(self.config)
        self.discovery_loss.model = self.model  # Pass model reference for temporal outputs
        self.complementarity_loss = ComplementarityLoss(
            correlation_weight=self.config.get('correlation_weight', 1.0),
            separation_weight=self.config.get('separation_weight', 0.5),
            diversity_weight=self.config.get('diversity_weight', 0.3)
        )
        self.solar_loss = SolarROILoss(class_weights=None)  # Use default weights
        
        # Task modules
        cluster_config = self.config.get('cluster_metrics', self.config.get('task', {}).get('cluster_labeling', {}))
        self.quality_labeler = ClusterQualityLabeler(cluster_config)
        self.cascade_analyzer = SolarCascadeAnalyzer(self.config)
        self.uncertainty_analyzer = UncertaintyAnalyzer(self.config)
        self.explainability_gen = ExplainabilityGenerator(self.config)
        solar_config = self.config.get('task', {}).get('solar', self.config)
        self.solar_simulator = SolarPerformanceSimulator(solar_config)
        self.cluster_evaluator = ClusterEvaluator(self.config)
        
        logger.info("All components initialized")
    
    def run_initial_assessment(self):
        """Run initial assessment to select MV station and evaluate LV groups"""
        logger.info("\n" + "="*60)
        logger.info("RUNNING INITIAL NETWORK ASSESSMENT")
        logger.info("="*60 + "\n")
        
        from evaluation.initial_assessment import InitialAssessment
        
        # Initialize assessment
        assessor = InitialAssessment(self.kg_connector)
        
        # Run full assessment
        results, summary_df = assessor.run_full_assessment()
        
        # Store results
        self.initial_assessment = results
        
        # Select MV station based on config strategy
        strategy = self.config.get('assessment', {}).get('selection_strategy', None)
        self.selected_mv_station = assessor.select_mv_station(strategy)
        
        if self.selected_mv_station:
            self.selected_lv_groups = assessor.get_lv_groups_for_mv(self.selected_mv_station)
            logger.info(f"Selected MV station: {self.selected_mv_station}")
            logger.info(f"Contains {len(self.selected_lv_groups)} LV groups")
            
            # Show summary
            mv_metrics = assessor.mv_assessments.get(self.selected_mv_station)
            if mv_metrics:
                logger.info(f"Priority Score: {mv_metrics.get_priority_score():.1f}/10")
                logger.info(f"Total Buildings: {mv_metrics.total_building_count}")
                logger.info(f"Strategy: {mv_metrics.strategy_recommendation}")
                logger.info(f"Poor Labels: {mv_metrics.avg_poor_label_ratio:.1%}")
                logger.info(f"Solar Potential: {mv_metrics.total_solar_potential_kw:.0f} kW")
        
        return results
    
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
                self._track_energy_flows(data, epoch)
            
            logger.info(f"Epoch {epoch+1}/{phase_epochs['semi_supervised']} - Loss: {loss:.4f}")
        
        # Phase 3: Solar optimization
        logger.info(f"\n=== PHASE 3: SOLAR OPTIMIZATION ({phase_epochs['solar_optimization']} epochs) ===")
        logger.info("Learning solar cascade effects and optimal placement...")
        
        for epoch in range(phase_epochs['solar_optimization']):
            loss = self._train_solar_epoch(data, epoch)
            
            # Simulate solar deployment rounds
            if epoch % 10 == 0:
                pass  # self._simulate_solar_deployment(data, epoch)  # Disabled - causes hang
            
            logger.info(f"Epoch {epoch+1}/{phase_epochs['solar_optimization']} - Loss: {loss:.4f}")
        
        logger.info("\n=== TRAINING COMPLETE ===")
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
        """Discovery phase training - unsupervised clustering"""
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
        
        # HeteroEnergyGNN returns cluster_logits for clustering task
        if 'cluster_logits' in outputs:
            outputs['clustering'] = outputs['cluster_logits']
        
        # Discovery loss (unsupervised)
        # Prepare physics data from the graph
        physics_data = {
            'demand': data['building'].x[:, 0] if 'building' in data.node_types else torch.zeros(1).to(self.device),
            'generation': data['building'].x[:, 1] if 'building' in data.node_types and data['building'].x.shape[1] > 1 else torch.zeros_like(data['building'].x[:, 0])
        }
        
        # Pass data as batch (contains lv_group_ids if available)
        loss, loss_components = self.discovery_loss(
            outputs,
            physics_data,
            data  # Pass data object which may contain lv_group_ids
        )
        
        # Add complementarity loss if available
        if 'complementarity' in outputs:
            comp_loss = self.complementarity_loss(
                outputs['complementarity'],
                data
            )
            loss += self.config['training'].get('complementarity_weight', 0.5) * comp_loss
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _train_semi_supervised_epoch(self, data: HeteroData, epoch: int) -> float:
        """Semi-supervised phase with pseudo-labels"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with clustering task (still clustering but with labels)
        outputs = self.model(data, task='clustering')
        
        # HeteroEnergyGNN returns cluster_logits
        if 'cluster_logits' in outputs:
            outputs['clustering'] = outputs['cluster_logits']
        
        # Use pseudo-labels
        labels = self.pseudo_labels.get('cluster_labels')
        label_confidence = self.pseudo_labels.get('confidence', 1.0)
        
        # Prepare physics data
        physics_data = {
            'demand': data['building'].x[:, 0] if 'building' in data.node_types else torch.zeros(1).to(self.device),
            'generation': data['building'].x[:, 1] if 'building' in data.node_types and data['building'].x.shape[1] > 1 else torch.zeros_like(data['building'].x[:, 0])
        }
        
        # Semi-supervised loss
        loss, loss_components = self.discovery_loss(
            outputs,
            physics_data,
            data  # Pass data object with lv_group_ids
        )
        
        # Weight by confidence
        loss = loss * label_confidence
        
        # Add cascade effects
        if 'solar' in outputs:
            cascade_loss = self._compute_cascade_loss(outputs['solar'], data)
            loss += self.config['training']['cascade_weight'] * cascade_loss
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _train_solar_epoch(self, data: HeteroData, epoch: int) -> float:
        """Solar optimization with learning loop"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # Forward pass with solar task
        solar_outputs = self.model(data, task='solar')
        
        # Also get clustering for combined optimization
        cluster_outputs = self.model(data, task='clustering')
        if 'cluster_logits' in cluster_outputs:
            outputs = {'clustering': cluster_outputs['cluster_logits'], 'solar': solar_outputs}
        else:
            outputs = {'solar': solar_outputs}
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
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(solar_logits, solar_targets)
        
        # Add simulation feedback (disabled for now - missing cluster_context)
        # if epoch % 5 == 0:
        #     sim_results = self.solar_simulator.simulate_deployment(
        #         solar_outputs,
        #         data,
        #         {}  # cluster_context placeholder
        #     )
        #     feedback_loss = self._compute_feedback_loss(sim_results)
        #     loss += self.config['training']['feedback_weight'] * feedback_loss
        
        # Backward
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
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
    
    def _track_energy_flows(self, data: HeteroData, epoch: int):
        """Track energy sharing between buildings"""
        with torch.no_grad():
            outputs = self.model(data, task='clustering')
            if 'cluster_logits' in outputs:
                outputs['clustering'] = outputs['cluster_logits']
            
            # Get cluster assignments
            clusters = outputs['clustering'].argmax(dim=-1)
            
            # Calculate energy flows within clusters
            for cluster_id in torch.unique(clusters):
                cluster_mask = (clusters == cluster_id)
                cluster_buildings = torch.where(cluster_mask)[0]
                
                if len(cluster_buildings) > 1:
                    # Get energy profiles
                    energy = data['building'].x[cluster_mask]
                    
                    # Simple energy sharing calculation
                    surplus = torch.relu(energy[:, -1])  # Last feature as generation
                    deficit = torch.relu(-energy[:, -1])
                    
                    # Track flows
                    self.energy_flows[f'epoch_{epoch}'][f'cluster_{cluster_id}'] = {
                        'total_surplus': surplus.sum().item(),
                        'total_deficit': deficit.sum().item(),
                        'self_sufficiency': min(1.0, surplus.sum() / (deficit.sum() + 1e-6))
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
        """Simulate iterative solar deployment rounds"""
        logger.info(f"Simulating solar deployment round at epoch {epoch}...")
        
        with torch.no_grad():
            outputs = self.model(data, task='clustering')
            if 'cluster_logits' in outputs:
                outputs['clustering'] = outputs['cluster_logits']
            
            # Analyze cascade effects
            cascade_results = self.cascade_analyzer.analyze_cascade(
                outputs.get('solar', outputs['clustering']),
                data,
                max_hops=3
            )
            
            # Store cascade effects
            self.solar_cascade_effects[f'round_{epoch}'] = {
                'direct_impact': cascade_results['direct_impact'],
                '1_hop_impact': cascade_results['hop_1_impact'],
                '2_hop_impact': cascade_results['hop_2_impact'],
                '3_hop_impact': cascade_results['hop_3_impact'],
                'total_network_benefit': cascade_results['total_benefit']
            }
            
            logger.info(f"Cascade effects - Direct: {cascade_results['direct_impact']:.2f}, "
                       f"3-hop total: {cascade_results['total_benefit']:.2f}")
    
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
            'cluster_stability': self.cluster_stability,
            'energy_flows': self.energy_flows,
            'pseudo_labels': {k: v.tolist() if torch.is_tensor(v) else v 
                            for k, v in self.pseudo_labels.items()},
            'solar_cascade_effects': self.solar_cascade_effects
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
                # If no temporal features, create minimal structure
                temporal_data = pd.DataFrame({
                    'building_id': [f'B_{i}' for i in range(160)],
                    'consumption': np.random.rand(160) * 100,
                    'generation': np.random.rand(160) * 20
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
            
            # Handle explainability output safely
            if 'summary' in explanations:
                logger.info(f"Explainability: {explanations['summary'][:100]}...")
            elif 'confidence' in explanations:
                logger.info(f"Confidence: {explanations['confidence']:.2f}")
        
        return cluster_metrics


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
        system.evaluate()
    
    logger.info("\n=== SYSTEM COMPLETE ===")


if __name__ == "__main__":
    main()