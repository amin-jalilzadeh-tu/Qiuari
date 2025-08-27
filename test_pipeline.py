"""
Test script for the simplified pipeline with proper environment setup
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import json
from pathlib import Path
import logging
import traceback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all imports"""
    try:
        logger.info("Testing imports...")
        
        # Models
        from models.solar_district_gnn import SolarDistrictGNN
        logger.info("✓ SolarDistrictGNN imported")
        
        # Training
        from training.unified_solar_trainer import UnifiedSolarTrainer
        logger.info("✓ UnifiedSolarTrainer imported")
        
        # Tasks
        from tasks.solar_labeling import SolarPerformanceLabeler, SolarInstallation
        logger.info("✓ Solar labeling imported")
        
        from tasks.solar_optimization import SolarOptimization
        logger.info("✓ Solar optimization imported")
        
        # Data
        from data.data_loader import EnergyDataLoader
        logger.info("✓ Data loader imported")
        
        return True
    except Exception as e:
        logger.error(f"Import error: {e}")
        traceback.print_exc()
        return False

def test_model_initialization():
    """Test model initialization"""
    try:
        logger.info("\nTesting model initialization...")
        
        from models.solar_district_gnn import SolarDistrictGNN
        
        # Simple config
        config = {
            'hidden_dim': 128,
            'num_clusters': 10,
            'max_hops': 3,
            'temporal_window': 24,
            'building_feat_dim': 15,
            'lv_feat_dim': 12,
            'transformer_feat_dim': 8,
            'temporal_feat_dim': 8
        }
        
        model = SolarDistrictGNN(config)
        logger.info(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model, config
    except Exception as e:
        logger.error(f"Model initialization error: {e}")
        traceback.print_exc()
        return None, None

def test_trainer_initialization(model, config):
    """Test trainer initialization"""
    try:
        logger.info("\nTesting trainer initialization...")
        
        from training.unified_solar_trainer import UnifiedSolarTrainer
        
        training_config = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-4,
            'T_0': 10,
            'T_mult': 2,
            'eta_min': 1e-6,
            'experiment_dir': 'experiments_test'
        }
        
        trainer = UnifiedSolarTrainer(model, training_config)
        logger.info("✓ Trainer initialized")
        
        return trainer
    except Exception as e:
        logger.error(f"Trainer initialization error: {e}")
        traceback.print_exc()
        return None

def test_data_creation():
    """Create synthetic test data"""
    try:
        logger.info("\nCreating synthetic test data...")
        
        from torch_geometric.data import Data, DataLoader
        
        # Create synthetic data
        num_buildings = 100
        num_edges = 300
        
        # Node features
        x = torch.randn(num_buildings, 15)  # 15 features per building
        
        # Edge index (random connections)
        edge_index = torch.randint(0, num_buildings, (2, num_edges))
        
        # Temporal features (batch_size=1, num_buildings, timesteps=24, features=8)
        # Adding batch dimension for TemporalSequenceEncoder
        temporal_features = torch.randn(1, num_buildings, 24, 8)
        
        # LV group IDs (10 groups)
        lv_group_ids = torch.randint(0, 10, (num_buildings,))
        
        # Create Data object
        data = Data(
            x=x,
            edge_index=edge_index,
            temporal_features=temporal_features,
            lv_group_ids=lv_group_ids,
            num_nodes=num_buildings
        )
        
        # Create DataLoader
        loader = DataLoader([data], batch_size=1, shuffle=False)
        
        logger.info(f"✓ Created synthetic data with {num_buildings} buildings, {num_edges} edges")
        
        return loader
    except Exception as e:
        logger.error(f"Data creation error: {e}")
        traceback.print_exc()
        return None

def test_forward_pass(model, data_loader):
    """Test model forward pass"""
    try:
        logger.info("\nTesting model forward pass...")
        
        device = next(model.parameters()).device
        model.eval()
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to same device as model
                batch = batch.to(device)
                
                # Test discovery phase
                outputs_discovery = model(batch, phase='discovery')
                logger.info(f"✓ Discovery phase output keys: {outputs_discovery.keys()}")
                
                if 'cluster_assignments' in outputs_discovery:
                    clusters = outputs_discovery['cluster_assignments']
                    logger.info(f"  - Cluster assignments shape: {clusters.shape}")
                
                # Test solar phase
                outputs_solar = model(batch, phase='solar')
                logger.info(f"✓ Solar phase output keys: {outputs_solar.keys()}")
                
                if 'solar_potential' in outputs_solar:
                    potential = outputs_solar['solar_potential']
                    logger.info(f"  - Solar potential shape: {potential.shape}")
                
                break
        
        return True
    except Exception as e:
        logger.error(f"Forward pass error: {e}")
        traceback.print_exc()
        return False

def test_training_step(trainer, data_loader):
    """Test one training step"""
    try:
        logger.info("\nTesting training step...")
        
        # Try one epoch of discovery training
        metrics = trainer._train_discovery_epoch(data_loader, use_labels=False)
        logger.info(f"✓ Discovery training metrics: {metrics}")
        
        # Try one epoch of solar training
        metrics = trainer._train_solar_epoch(data_loader, cluster_assignments=None)
        logger.info(f"✓ Solar training metrics: {metrics}")
        
        return True
    except Exception as e:
        logger.error(f"Training step error: {e}")
        traceback.print_exc()
        return False

def test_labeling_system():
    """Test labeling system"""
    try:
        logger.info("\nTesting labeling system...")
        
        from tasks.solar_labeling import SolarPerformanceLabeler, SolarInstallation
        from datetime import datetime
        
        labeler = SolarPerformanceLabeler()
        
        # Create test installation
        installation = SolarInstallation(
            building_id=1,
            cluster_id=0,
            installation_date=datetime.now(),
            capacity_kw=10.0,
            installation_cost=10000,
            daily_generation_kwh=list(np.random.uniform(30, 50, 90)),
            self_consumption_rate=0.7,
            export_revenue=100,
            peak_reduction_percent=0.3
        )
        
        labeler.add_installation(installation)
        label, confidence = labeler.label_installation(1, force=True)
        
        logger.info(f"✓ Labeling system working: Label={label}, Confidence={confidence:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"Labeling system error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    logger.info("="*60)
    logger.info("Testing Simplified Pipeline Components")
    logger.info("="*60)
    
    # Test imports
    if not test_imports():
        logger.error("Import test failed. Stopping.")
        return
    
    # Test model initialization
    model, config = test_model_initialization()
    if model is None:
        logger.error("Model initialization failed. Stopping.")
        return
    
    # Test trainer initialization
    trainer = test_trainer_initialization(model, config)
    if trainer is None:
        logger.error("Trainer initialization failed. Stopping.")
        return
    
    # Create test data
    data_loader = test_data_creation()
    if data_loader is None:
        logger.error("Data creation failed. Stopping.")
        return
    
    # Test forward pass
    if not test_forward_pass(model, data_loader):
        logger.error("Forward pass failed. Stopping.")
        return
    
    # Test training step
    if not test_training_step(trainer, data_loader):
        logger.error("Training step failed. Stopping.")
        return
    
    # Test labeling system
    if not test_labeling_system():
        logger.error("Labeling system failed. Stopping.")
        return
    
    logger.info("\n" + "="*60)
    logger.info("✓ All tests passed successfully!")
    logger.info("="*60)

if __name__ == "__main__":
    main()