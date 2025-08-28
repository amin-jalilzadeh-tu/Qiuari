"""
Test GNN training with small epochs to check for issues
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from gnn_main import UnifiedGNNSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    print("="*60)
    print("TESTING GNN TRAINING WITH 3 EPOCHS")
    print("="*60)
    
    # Initialize system
    system = UnifiedGNNSystem()
    
    # Run training with just 3 epochs to test
    try:
        system.train(num_epochs=3, run_assessment=False)  # Skip assessment for speed
        print("\nTraining completed successfully!")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
    
    # Check results
    if hasattr(system, 'cluster_stability'):
        print(f"\nCluster stability scores: {system.cluster_stability}")
    
    if hasattr(system, 'energy_flows'):
        print(f"Energy flow data collected: {len(system.energy_flows)} timesteps")
    
    if hasattr(system, 'pseudo_labels'):
        print(f"Pseudo labels generated: {len(system.pseudo_labels)} sets")