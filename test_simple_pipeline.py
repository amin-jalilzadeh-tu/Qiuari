"""
Simplified test to verify the core pipeline works
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_model():
    """Test basic model functionality"""
    try:
        from models.solar_district_gnn import SolarDistrictGNN
        
        # Minimal config
        config = {
            'hidden_dim': 64,
            'num_clusters': 5,
            'max_hops': 2,
            'temporal_window': 24,
            'building_feat_dim': 10,
            'lv_feat_dim': 8,
            'transformer_feat_dim': 6,
            'temporal_feat_dim': 4
        }
        
        # Create model
        model = SolarDistrictGNN(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        logger.info(f"✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Create minimal test data
        num_nodes = 50
        num_edges = 100
        
        data = Data(
            x=torch.randn(num_nodes, 10).to(device),  # Node features
            edge_index=torch.randint(0, num_nodes, (2, num_edges)).to(device),  # Edges
            num_nodes=num_nodes
        )
        
        # Test forward pass - discovery
        model.eval()
        with torch.no_grad():
            outputs = model(data, phase='discovery')
            logger.info(f"✓ Discovery outputs: {list(outputs.keys())}")
            
            if 'cluster_assignments' in outputs:
                clusters = outputs['cluster_assignments']
                logger.info(f"  Cluster shape: {clusters.shape}")
                logger.info(f"  Clusters range: [{clusters.min():.3f}, {clusters.max():.3f}]")
        
        # Test forward pass - solar
        with torch.no_grad():
            outputs = model(data, phase='solar')
            logger.info(f"✓ Solar outputs: {list(outputs.keys())}")
            
            if 'solar_potential' in outputs:
                potential = outputs['solar_potential']
                logger.info(f"  Solar potential shape: {potential.shape}")
                logger.info(f"  Potential range: [{potential.min():.3f}, {potential.max():.3f}]")
        
        return model, data
        
    except Exception as e:
        logger.error(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_basic_training(model, data):
    """Test basic training loop"""
    try:
        from training.unified_solar_trainer import UnifiedSolarTrainer
        
        # Simple training config
        config = {
            'learning_rate': 1e-3,
            'weight_decay': 1e-5,
            'T_0': 10,
            'experiment_dir': 'test_experiments'
        }
        
        trainer = UnifiedSolarTrainer(model, config)
        logger.info("✓ Trainer initialized")
        
        # Create simple dataloader
        loader = DataLoader([data], batch_size=1)
        
        # Test one training step
        model.train()
        optimizer = trainer.optimizer
        
        # Simple forward-backward pass
        outputs = model(data, phase='discovery')
        
        # Create simple loss (just for testing)
        if 'cluster_assignments' in outputs:
            loss = outputs['cluster_assignments'].sum() * 0.001  # Dummy loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            logger.info(f"✓ Training step completed, loss: {loss.item():.6f}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"Training test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_solar_labeling():
    """Test labeling system"""
    try:
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
            daily_generation_kwh=[40.0] * 90,  # 90 days of data
            self_consumption_rate=0.7,
            export_revenue=100,
            peak_reduction_percent=0.3
        )
        
        labeler.add_installation(installation)
        roi = labeler.calculate_roi(installation)
        label, confidence = labeler.label_installation(1, force=True)
        
        logger.info(f"✓ Labeling works: ROI={roi:.1f} years, Label={label}, Confidence={confidence:.2f}")
        
        # Test cluster labeling
        cluster_metrics = {
            0: {'self_sufficiency': 0.6, 'complementarity': -0.2, 'peak_reduction': 0.3, 'network_losses': 0.1},
            1: {'self_sufficiency': 0.3, 'complementarity': 0.1, 'peak_reduction': 0.1, 'network_losses': 0.2}
        }
        
        cluster_labels = labeler.generate_cluster_labels(cluster_metrics)
        logger.info(f"✓ Cluster labels: {cluster_labels}")
        
        return labeler
        
    except Exception as e:
        logger.error(f"Labeling test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_integration():
    """Test basic integration between components"""
    try:
        logger.info("\nTesting Component Integration...")
        
        # Create model and data
        model, data = test_basic_model()
        if model is None:
            return False
        
        # Test training
        trainer = test_basic_training(model, data)
        if trainer is None:
            return False
        
        # Test labeling
        labeler = test_solar_labeling()
        if labeler is None:
            return False
        
        # Test recommendation
        model.eval()
        with torch.no_grad():
            outputs = model(data, phase='solar')
            
            if 'solar_potential' in outputs:
                potential = outputs['solar_potential'].squeeze()
                top_k = 5
                top_values, top_indices = torch.topk(potential, min(top_k, len(potential)))
                
                logger.info(f"✓ Top {top_k} buildings for solar:")
                for i, (idx, score) in enumerate(zip(top_indices, top_values)):
                    logger.info(f"   {i+1}. Building {idx}: {score:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    logger.info("="*60)
    logger.info("SIMPLIFIED PIPELINE TEST")
    logger.info("="*60)
    
    # Test model
    model, data = test_basic_model()
    if model is None:
        logger.error("❌ Model test failed")
        return
    
    logger.info("\n" + "-"*40)
    
    # Test training
    trainer = test_basic_training(model, data)
    if trainer is None:
        logger.error("❌ Training test failed")
        return
    
    logger.info("\n" + "-"*40)
    
    # Test labeling
    labeler = test_solar_labeling()
    if labeler is None:
        logger.error("❌ Labeling test failed")
        return
    
    logger.info("\n" + "-"*40)
    
    # Test integration
    if test_integration():
        logger.info("\n" + "="*60)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("The simplified pipeline is working correctly")
        logger.info("="*60)
    else:
        logger.error("❌ Integration test failed")

if __name__ == "__main__":
    main()