#!/usr/bin/env python3
"""
Runtime System Tracer - Captures actual runtime execution data
"""

import os
import sys
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class RuntimeTracer:
    def __init__(self):
        self.traces = []
        self.start_time = time.time()
        
    def trace_stage(self, stage_name, data=None):
        stage_info = {
            'stage': stage_name,
            'timestamp': datetime.now().isoformat(),
            'elapsed_from_start': time.time() - self.start_time,
            'data': data
        }
        self.traces.append(stage_info)
        print(f"[{stage_info['elapsed_from_start']:.2f}s] {stage_name}")
        if data:
            for key, value in data.items():
                print(f"  {key}: {value}")
    
    def run_end_to_end_test(self):
        """Run actual end-to-end test with real data"""
        try:
            sys.path.append('.')
            
            # Stage 1: System initialization
            self.trace_stage("System Initialization")
            from main import UnifiedEnergyGNNSystem
            system = UnifiedEnergyGNNSystem("config/config.yaml")
            
            self.trace_stage("System Ready", {
                'device': str(system.device),
                'model_type': type(system.model).__name__,
                'trainer_type': type(system.trainer).__name__
            })
            
            # Stage 2: Data loading with actual KG data
            self.trace_stage("Starting Data Load")
            data_start = time.time()
            
            # Get basic LV group info first
            lv_groups = system.kg_connector.get_all_lv_groups()
            self.trace_stage("LV Groups Retrieved", {
                'total_lv_groups': len(lv_groups),
                'sample_ids': lv_groups[:5] if lv_groups else []
            })
            
            # Load limited data for testing
            try:
                train_loader, val_loader, test_loader = system.load_and_prepare_data(evaluate_groups=False)
                data_time = time.time() - data_start
                
                self.trace_stage("Data Loading Complete", {
                    'load_time_seconds': data_time,
                    'train_batches': len(train_loader) if train_loader else 0,
                    'val_batches': len(val_loader) if val_loader else 0,
                    'test_batches': len(test_loader) if test_loader else 0
                })
                
                # Stage 3: Analyze first batch
                if train_loader and len(train_loader) > 0:
                    batch = next(iter(train_loader))
                    self.trace_stage("First Batch Analysis", {
                        'node_features_shape': list(batch.x.shape) if hasattr(batch, 'x') else 'N/A',
                        'edges_shape': list(batch.edge_index.shape) if hasattr(batch, 'edge_index') else 'N/A',
                        'temporal_shape': list(batch.temporal_profiles.shape) if hasattr(batch, 'temporal_profiles') else 'N/A',
                        'num_nodes': batch.x.shape[0] if hasattr(batch, 'x') else 0
                    })
                    
                    # Stage 4: Model forward pass test
                    system.model.eval()
                    batch = batch.to(system.device)
                    
                    forward_start = time.time()
                    with torch.no_grad():
                        outputs = system.model(batch)
                    forward_time = time.time() - forward_start
                    
                    output_info = {}
                    if isinstance(outputs, dict):
                        for key, value in outputs.items():
                            if torch.is_tensor(value):
                                output_info[f'{key}_shape'] = list(value.shape)
                    
                    self.trace_stage("Model Forward Pass", {
                        'forward_time_ms': forward_time * 1000,
                        'outputs': output_info
                    })
                    
                    # Stage 5: Limited training test (2 epochs)
                    if hasattr(system.trainer, 'train_epoch'):
                        self.trace_stage("Starting Training Test")
                        
                        epoch_times = []
                        for epoch in range(2):  # Just 2 epochs for testing
                            epoch_start = time.time()
                            
                            try:
                                # Simple training loop
                                system.model.train()
                                batch_count = 0
                                for batch in train_loader:
                                    if batch_count >= 3:  # Limit batches
                                        break
                                    batch = batch.to(system.device)
                                    outputs = system.model(batch)
                                    batch_count += 1
                                
                                epoch_time = time.time() - epoch_start
                                epoch_times.append(epoch_time)
                                
                            except Exception as e:
                                epoch_times.append(-1)  # Mark failed epoch
                        
                        self.trace_stage("Training Test Complete", {
                            'epochs_attempted': 2,
                            'epoch_times_seconds': epoch_times,
                            'avg_epoch_time': sum(t for t in epoch_times if t > 0) / len([t for t in epoch_times if t > 0]) if epoch_times else 0
                        })
                
                # Stage 6: Pattern analysis simulation
                self.trace_stage("Pattern Analysis Simulation")
                if test_loader and len(test_loader) > 0:
                    system.model.eval()
                    analysis_results = []
                    
                    with torch.no_grad():
                        for batch_idx, batch in enumerate(test_loader):
                            if batch_idx >= 2:  # Limit to 2 batches
                                break
                            
                            batch = batch.to(system.device)
                            outputs = system.model(batch)
                            
                            # Extract cluster information
                            cluster_info = {}
                            if 'clusters' in outputs:
                                cluster_logits = outputs['clusters']
                                if cluster_logits.dim() > 1:
                                    cluster_assignments = torch.argmax(cluster_logits, dim=-1)
                                    unique_clusters = torch.unique(cluster_assignments)
                                    cluster_info = {
                                        'num_nodes': batch.x.shape[0],
                                        'num_clusters': len(unique_clusters),
                                        'cluster_sizes': [(cluster_assignments == c).sum().item() for c in unique_clusters]
                                    }
                            
                            analysis_results.append(cluster_info)
                    
                    self.trace_stage("Pattern Analysis Complete", {
                        'batches_analyzed': len(analysis_results),
                        'sample_results': analysis_results
                    })
                
                return True
                
            except Exception as e:
                self.trace_stage("Data Loading Failed", {
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                return False
                
        except Exception as e:
            self.trace_stage("System Error", {
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return False
    
    def save_trace_data(self):
        """Save trace data to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        output_dir = Path("trace_outputs")
        output_dir.mkdir(exist_ok=True)
        
        trace_file = output_dir / f"runtime_trace_{timestamp}.json"
        
        with open(trace_file, 'w') as f:
            json.dump({
                'total_execution_time': time.time() - self.start_time,
                'trace_count': len(self.traces),
                'traces': self.traces
            }, f, indent=2, default=str)
        
        print(f"\nTrace data saved to: {trace_file}")
        return trace_file

def main():
    print("="*80)
    print("RUNTIME SYSTEM TRACER")
    print("="*80)
    
    tracer = RuntimeTracer()
    
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*80)
    
    success = tracer.run_end_to_end_test()
    
    total_time = time.time() - tracer.start_time
    
    print("\n" + "="*80)
    print("RUNTIME TRACE COMPLETE")
    print("="*80)
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Stages Traced: {len(tracer.traces)}")
    print(f"Status: {'SUCCESS' if success else 'PARTIAL'}")
    
    # Save trace data
    trace_file = tracer.save_trace_data()
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)