"""
Quick test of balanced training to verify anti-collapse measures
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import logging
import yaml

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_balanced_training():
    """Test if balanced loss prevents cluster collapse"""
    
    # Import system
    from gnn_main import UnifiedGNNSystem
    
    # Initialize with config path (not dict)
    logger.info("Initializing GNN system with balanced loss...")
    system = UnifiedGNNSystem('config/unified_config.yaml')
    
    # Override epochs for quick test
    system.config['training']['epochs'] = 15  # Quick test with 5 epochs per phase
    
    # Run training (skip assessment for speed)
    logger.info("Starting training with anti-collapse measures...")
    system.train(num_epochs=15, run_assessment=False)
    
    # Check final clusters
    logger.info("\n" + "="*60)
    logger.info("CHECKING FINAL CLUSTER DISTRIBUTION")
    logger.info("="*60)
    
    # Get final cluster assignments
    data = system._prepare_data()
    with torch.no_grad():
        outputs = system.model(data, task='clustering')
        cluster_logits = outputs.get('cluster_logits', outputs.get('clustering'))
        cluster_assignments = cluster_logits.argmax(dim=-1)
        
        # Count clusters
        unique_clusters, counts = torch.unique(cluster_assignments, return_counts=True)
        
        logger.info(f"Number of clusters: {len(unique_clusters)}")
        logger.info("Cluster sizes:")
        for i, (cluster_id, count) in enumerate(zip(unique_clusters, counts)):
            percentage = (count.item() / len(cluster_assignments)) * 100
            logger.info(f"  Cluster {cluster_id}: {count} buildings ({percentage:.1f}%)")
        
        # Check for collapse
        if len(unique_clusters) == 1:
            logger.error("❌ FAILED: Model still collapsed to single cluster!")
            return False
        elif len(unique_clusters) < 3:
            logger.warning("⚠️  WARNING: Only {len(unique_clusters)} clusters formed (target: 4-8)")
            return False
        else:
            logger.info(f"✅ SUCCESS: Model formed {len(unique_clusters)} distinct clusters!")
            
            # Check balance
            balance_score = counts.min().item() / counts.max().item()
            logger.info(f"Balance score: {balance_score:.3f} (1.0 = perfectly balanced)")
            
            if balance_score < 0.1:
                logger.warning("⚠️  Clusters are still quite imbalanced")
            
            return True

if __name__ == "__main__":
    success = test_balanced_training()
    
    if success:
        logger.info("\n✅ Balanced training test PASSED - collapse prevention working!")
    else:
        logger.info("\n❌ Balanced training test FAILED - further tuning needed")