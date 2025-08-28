"""
Quick test to check GNN main issues
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Test imports
print("Testing imports...")
try:
    from gnn_main import UnifiedGNNSystem
    print("OK: UnifiedGNNSystem imported")
except Exception as e:
    print(f"ERROR importing UnifiedGNNSystem: {e}")
    exit(1)

# Test initialization
print("\nTesting initialization...")
try:
    system = UnifiedGNNSystem()
    print("OK: System initialized")
except Exception as e:
    print(f"ERROR initializing system: {e}")
    exit(1)

# Test data preparation
print("\nTesting data preparation...")
try:
    data = system._prepare_data()
    print(f"OK: Data prepared: {data}")
    print(f"  Building nodes: {data['building'].x.shape if 'building' in data.node_types else 'N/A'}")
    print(f"  LV group IDs: {data.lv_group_ids.shape if hasattr(data, 'lv_group_ids') else 'N/A'}")
except Exception as e:
    print(f"ERROR preparing data: {e}")
    import traceback
    traceback.print_exc()

# Test one training epoch
print("\nTesting discovery epoch...")
try:
    if data:
        loss = system._train_discovery_epoch(data, 0)
        print(f"OK: Discovery epoch completed, loss: {loss:.4f}")
except Exception as e:
    print(f"ERROR in discovery epoch: {e}")
    import traceback
    traceback.print_exc()

print("\nQuick test complete!")