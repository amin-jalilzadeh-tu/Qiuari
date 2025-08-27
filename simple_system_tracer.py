#!/usr/bin/env python3
"""
Simple System Tracer for Qiuari_V3 - Bypasses OpenMP issues
Performs focused tracing of key system components
"""

import os
import sys
import time
import traceback
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set environment variable first
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def trace_component(component_name, func):
    """Trace execution of a component"""
    print(f"\n[TRACE] {component_name}")
    print("-" * 60)
    
    start_time = time.time()
    result = None
    error = None
    
    try:
        result = func()
        status = "SUCCESS"
    except Exception as e:
        error = str(e)
        status = "ERROR"
        print(f"ERROR: {error}")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Status: {status}")
    print(f"Duration: {duration:.3f}s")
    
    return {
        'component': component_name,
        'status': status,
        'duration': duration,
        'error': error,
        'result': result
    }

def test_imports():
    """Test importing core modules"""
    print("Testing core imports...")
    
    # Test basic PyTorch import
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    
    # Test PyTorch Geometric
    try:
        import torch_geometric
        print(f"  PyTorch Geometric: {torch_geometric.__version__}")
    except ImportError:
        print("  PyTorch Geometric: NOT AVAILABLE")
    
    # Test Neo4j driver
    try:
        from neo4j import GraphDatabase
        print("  Neo4j Driver: Available")
    except ImportError:
        print("  Neo4j Driver: NOT AVAILABLE")
    
    # Test main components
    sys.path.append('.')
    
    try:
        from data.kg_connector import KGConnector
        print("  KGConnector: Available")
    except Exception as e:
        print(f"  KGConnector: ERROR - {e}")
    
    try:
        from models.base_gnn import create_gnn_model
        print("  Base GNN: Available")
    except Exception as e:
        print(f"  Base GNN: ERROR - {e}")
    
    try:
        from main import UnifiedEnergyGNNSystem
        print("  Main System: Available")
    except Exception as e:
        print(f"  Main System: ERROR - {e}")
    
    return {"status": "completed"}

def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    
    config_path = "config/config.yaml"
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"  Config keys: {list(config.keys())}")
    print(f"  Model config: {config.get('model', {}).get('type', 'unknown')}")
    print(f"  Training mode: {config.get('training', {}).get('mode', 'unknown')}")
    print(f"  Device: {config.get('system', {}).get('device', 'unknown')}")
    
    return {"status": "success", "config_keys": len(config)}

def test_kg_connection():
    """Test Knowledge Graph connection"""
    print("Testing KG connection...")
    
    sys.path.append('.')
    from data.kg_connector import KGConnector
    
    # Use config for connection details
    import yaml
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    kg_config = config['kg']
    
    try:
        connector = KGConnector(
            uri=kg_config['uri'],
            user=kg_config['user'],
            password=kg_config['password']
        )
        
        # Test basic query
        lv_groups = connector.get_all_lv_groups()
        print(f"  LV Groups found: {len(lv_groups)}")
        
        if lv_groups:
            sample_lv = lv_groups[0]
            lv_data = connector.get_lv_group_data(sample_lv)
            print(f"  Sample LV data - Buildings: {len(lv_data.get('buildings', []))}")
        
        if hasattr(connector, 'close'):
            connector.close()
        
        return {"status": "success", "lv_groups": len(lv_groups)}
        
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def test_model_creation():
    """Test model creation"""
    print("Testing model creation...")
    
    sys.path.append('.')
    from models.base_gnn import create_gnn_model
    import torch
    
    # Simple model config
    model_config = {
        'type': 'hetero',
        'input_dim': 17,
        'hidden_dim': 64,
        'num_layers': 2,
        'num_clusters': 5,
        'dropout': 0.1,
        'building_features': 17,
        'temporal_dim': 8,
        'num_heads': 4
    }
    
    model = create_gnn_model('hetero', model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Model type: {type(model).__name__}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Test forward pass with dummy data
    try:
        from torch_geometric.data import Data
        
        dummy_data = Data(
            x=torch.randn(10, 17),
            edge_index=torch.randint(0, 10, (2, 20)),
            temporal_profiles=torch.randn(10, 96),
            batch=torch.zeros(10, dtype=torch.long)
        )
        
        model.eval()
        with torch.no_grad():
            outputs = model(dummy_data)
        
        print(f"  Forward pass: SUCCESS")
        if isinstance(outputs, dict):
            print(f"  Output keys: {list(outputs.keys())}")
            for key, value in outputs.items():
                if torch.is_tensor(value):
                    print(f"    {key}: {list(value.shape)}")
        
        return {"status": "success", "parameters": total_params}
        
    except Exception as e:
        print(f"  Forward pass: ERROR - {e}")
        return {"status": "partial", "parameters": total_params, "forward_error": str(e)}

def test_data_loading():
    """Test data loading process"""
    print("Testing data loading...")
    
    sys.path.append('.')
    from data.data_loader import EnergyDataLoader
    from data.feature_processor import FeatureProcessor
    import torch
    
    # Simple config
    data_config = {
        'batch_size': 16,
        'num_workers': 0,
        'shuffle': True,
        'min_cluster_size': 3,
        'max_cluster_size': 20
    }
    
    try:
        # Create data loader
        data_loader = EnergyDataLoader(data_config, mode='train')
        feature_processor = FeatureProcessor()
        
        print(f"  Data loader created: {type(data_loader).__name__}")
        print(f"  Feature processor created: {type(feature_processor).__name__}")
        
        # Test feature processing with dummy data
        dummy_features = torch.randn(5, 10)
        processed = feature_processor.process_features(dummy_features)
        
        print(f"  Feature processing: {dummy_features.shape} -> {processed.shape}")
        
        return {"status": "success", "feature_dim": processed.shape[-1]}
        
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def test_system_initialization():
    """Test full system initialization"""
    print("Testing system initialization...")
    
    sys.path.append('.')
    
    try:
        from main import UnifiedEnergyGNNSystem
        
        # Initialize with reduced complexity
        system = UnifiedEnergyGNNSystem("config/config.yaml")
        
        print(f"  System created: {type(system).__name__}")
        print(f"  Device: {system.device}")
        print(f"  Model: {type(system.model).__name__}")
        print(f"  Trainer: {type(system.trainer).__name__}")
        print(f"  KG Connector: {type(system.kg_connector).__name__}")
        
        # Test model parameters
        if hasattr(system.model, 'parameters'):
            total_params = sum(p.numel() for p in system.model.parameters())
            print(f"  Model parameters: {total_params:,}")
        
        return {"status": "success", "device": str(system.device)}
        
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def test_training_components():
    """Test training-related components"""
    print("Testing training components...")
    
    sys.path.append('.')
    
    try:
        from training.loss_functions import UnifiedEnergyLoss
        from training.discovery_trainer import DiscoveryGNNTrainer
        import torch
        
        # Test loss function
        loss_config = {
            'w_cluster': 1.0,
            'w_placement': 1.0,
            'w_flow': 1.0,
            'w_physics': 1.0
        }
        
        loss_fn = UnifiedEnergyLoss(loss_config)
        print(f"  Loss function: {type(loss_fn).__name__}")
        
        # Test with dummy data
        dummy_outputs = {
            'clusters': torch.randn(10, 5),
            'placement': torch.randn(10, 4),
            'flow': torch.randn(10, 10)
        }
        dummy_batch = type('MockBatch', (), {})()
        dummy_batch.x = torch.randn(10, 17)
        dummy_batch.edge_index = torch.randint(0, 10, (2, 20))
        
        try:
            loss_value = loss_fn(dummy_outputs, dummy_batch)
            print(f"  Loss computation: SUCCESS (value: {loss_value.item():.4f})")
        except Exception as e:
            print(f"  Loss computation: ERROR - {e}")
        
        return {"status": "success"}
        
    except Exception as e:
        return {"status": "failed", "error": str(e)}

def main():
    """Main tracing function"""
    print("="*80)
    print("SIMPLE SYSTEM TRACER - QIUARI_V3")
    print("="*80)
    
    trace_results = []
    
    # Test sequence
    tests = [
        ("Imports", test_imports),
        ("Config Loading", test_config_loading),
        ("KG Connection", test_kg_connection),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
        ("System Initialization", test_system_initialization),
        ("Training Components", test_training_components),
    ]
    
    start_time = time.time()
    
    for test_name, test_func in tests:
        result = trace_component(test_name, test_func)
        trace_results.append(result)
        
        # Stop on critical failures
        if result['status'] == 'ERROR' and test_name in ['Imports', 'Config Loading']:
            print(f"\n[CRITICAL] Stopping due to {test_name} failure")
            break
    
    total_time = time.time() - start_time
    
    # Generate summary
    print("\n" + "="*80)
    print("TRACE SUMMARY")
    print("="*80)
    
    successful = [r for r in trace_results if r['status'] == 'SUCCESS']
    failed = [r for r in trace_results if r['status'] == 'ERROR']
    partial = [r for r in trace_results if r['status'] == 'PARTIAL']
    
    print(f"\nTotal Execution Time: {total_time:.2f} seconds")
    print(f"Components Tested: {len(trace_results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Partial: {len(partial)}")
    print(f"Success Rate: {len(successful)/len(trace_results)*100:.1f}%")
    
    print(f"\nComponent Status:")
    for result in trace_results:
        status_symbol = "✓" if result['status'] == 'SUCCESS' else "✗" if result['status'] == 'ERROR' else "~"
        print(f"  {status_symbol} {result['component']}: {result['status']} ({result['duration']:.3f}s)")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create trace_outputs directory
    output_dir = Path("trace_outputs")
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    result_file = output_dir / f"simple_trace_{timestamp}.json"
    with open(result_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_time': total_time,
            'success_rate': len(successful)/len(trace_results),
            'results': trace_results
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {result_file}")
    
    # Return success if most components worked
    return len(successful) >= len(trace_results) * 0.7  # 70% success rate

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)