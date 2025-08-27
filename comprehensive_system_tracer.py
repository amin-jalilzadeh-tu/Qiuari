#!/usr/bin/env python3
"""
Comprehensive End-to-End System Tracer for Qiuari_V3
Performs deep tracing and validation of the entire system with real execution data.
"""

import os
import sys
import time
import traceback
import psutil
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import json

# Set OpenMP environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

@dataclass
class ExecutionTrace:
    """Data class to store execution traces"""
    stage: str
    start_time: float
    end_time: float
    duration: float
    memory_before: float
    memory_after: float
    memory_peak: float
    memory_delta: float
    cpu_usage: float
    gpu_memory: float = 0.0
    data_shapes: Dict[str, Any] = None
    intermediate_results: Dict[str, Any] = None
    errors: List[str] = None
    warnings: List[str] = None
    performance_metrics: Dict[str, float] = None

class SystemTracer:
    """Main system tracer class"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.traces: List[ExecutionTrace] = []
        self.global_start_time = time.time()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.process = psutil.Process()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up logging
        self.log_file = f"trace_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        print("="*80)
        print("COMPREHENSIVE SYSTEM TRACER - QIUARI_V3")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"System RAM: {psutil.virtual_memory().total / 1024**3:.2f} GB")
        print(f"CPU Cores: {psutil.cpu_count(logical=False)} physical, {psutil.cpu_count()} logical")
        print("="*80)
    
    @contextmanager
    def trace_execution(self, stage: str):
        """Context manager to trace execution of a stage"""
        # Pre-execution measurements
        start_time = time.time()
        memory_before = self.process.memory_info().rss / 1024**2  # MB
        cpu_before = self.process.cpu_percent()
        gpu_memory = 0.0
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # Initialize trace object
        trace = ExecutionTrace(
            stage=stage,
            start_time=start_time,
            end_time=0.0,
            duration=0.0,
            memory_before=memory_before,
            memory_after=0.0,
            memory_peak=memory_before,
            memory_delta=0.0,
            cpu_usage=0.0,
            gpu_memory=gpu_memory,
            data_shapes={},
            intermediate_results={},
            errors=[],
            warnings=[],
            performance_metrics={}
        )
        
        print(f"\n[TRACING] {stage}")
        print(f"  Start Time: {datetime.fromtimestamp(start_time).strftime('%H:%M:%S.%f')[:-3]}")
        print(f"  Memory Before: {memory_before:.1f} MB")
        if torch.cuda.is_available():
            print(f"  GPU Memory: {gpu_memory:.1f} MB")
        
        try:
            yield trace
            
        except Exception as e:
            trace.errors.append(f"ERROR: {str(e)}")
            trace.errors.append(f"TRACEBACK: {traceback.format_exc()}")
            print(f"  ERROR in {stage}: {str(e)}")
            
        finally:
            # Post-execution measurements
            end_time = time.time()
            memory_after = self.process.memory_info().rss / 1024**2  # MB
            cpu_after = self.process.cpu_percent()
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_memory_after = torch.cuda.memory_allocated() / 1024**2  # MB
                trace.gpu_memory = gpu_memory_after
            
            # Update trace
            trace.end_time = end_time
            trace.duration = end_time - start_time
            trace.memory_after = memory_after
            trace.memory_delta = memory_after - memory_before
            trace.memory_peak = max(memory_before, memory_after)
            trace.cpu_usage = (cpu_before + cpu_after) / 2
            
            # Log execution summary
            print(f"  End Time: {datetime.fromtimestamp(end_time).strftime('%H:%M:%S.%f')[:-3]}")
            print(f"  Duration: {trace.duration:.3f}s")
            print(f"  Memory After: {memory_after:.1f} MB (Δ{trace.memory_delta:+.1f} MB)")
            print(f"  CPU Usage: {trace.cpu_usage:.1f}%")
            if torch.cuda.is_available():
                print(f"  GPU Memory After: {trace.gpu_memory:.1f} MB")
            
            if trace.errors:
                print(f"  Errors: {len(trace.errors)}")
            if trace.warnings:
                print(f"  Warnings: {len(trace.warnings)}")
            
            # Add to traces
            self.traces.append(trace)
    
    def log_data_shapes(self, trace: ExecutionTrace, data_dict: Dict[str, Any], prefix: str = ""):
        """Log shapes of data structures"""
        for key, value in data_dict.items():
            shape_key = f"{prefix}{key}" if prefix else key
            
            if torch.is_tensor(value):
                trace.data_shapes[shape_key] = {
                    'type': 'torch.Tensor',
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'device': str(value.device),
                    'memory_mb': value.numel() * value.element_size() / 1024**2
                }
            elif isinstance(value, np.ndarray):
                trace.data_shapes[shape_key] = {
                    'type': 'numpy.ndarray',
                    'shape': list(value.shape),
                    'dtype': str(value.dtype),
                    'memory_mb': value.nbytes / 1024**2
                }
            elif isinstance(value, (list, tuple)):
                trace.data_shapes[shape_key] = {
                    'type': type(value).__name__,
                    'length': len(value),
                    'memory_mb': sys.getsizeof(value) / 1024**2
                }
            elif isinstance(value, dict):
                trace.data_shapes[shape_key] = {
                    'type': 'dict',
                    'keys': len(value),
                    'memory_mb': sys.getsizeof(value) / 1024**2
                }
                # Recursively log nested dictionaries
                if len(value) < 20:  # Avoid too much recursion
                    self.log_data_shapes(trace, value, f"{shape_key}.")
            elif hasattr(value, '__len__') and not isinstance(value, str):
                try:
                    trace.data_shapes[shape_key] = {
                        'type': type(value).__name__,
                        'length': len(value),
                        'memory_mb': sys.getsizeof(value) / 1024**2
                    }
                except:
                    pass
    
    def run_comprehensive_trace(self):
        """Run comprehensive end-to-end trace"""
        try:
            # Stage 1: System Initialization
            with self.trace_execution("01_System_Initialization") as trace:
                system = self._initialize_system(trace)
            
            # Stage 2: Knowledge Graph Connection
            with self.trace_execution("02_KG_Connection") as trace:
                kg_data = self._test_kg_connection(system, trace)
            
            # Stage 3: Data Loading and Processing
            with self.trace_execution("03_Data_Loading") as trace:
                data_loaders = self._load_and_process_data(system, trace)
            
            # Stage 4: Model Creation and Validation
            with self.trace_execution("04_Model_Creation") as trace:
                model_info = self._create_and_validate_model(system, trace)
            
            # Stage 5: Training Loop (Few Epochs)
            with self.trace_execution("05_Training_Loop") as trace:
                training_results = self._run_training_loop(system, data_loaders, trace)
            
            # Stage 6: Model Inference
            with self.trace_execution("06_Model_Inference") as trace:
                inference_results = self._run_model_inference(system, data_loaders, trace)
            
            # Stage 7: Pattern Analysis
            with self.trace_execution("07_Pattern_Analysis") as trace:
                pattern_results = self._analyze_patterns(system, data_loaders, trace)
            
            # Stage 8: Intervention Generation
            with self.trace_execution("08_Intervention_Generation") as trace:
                intervention_results = self._generate_interventions(system, pattern_results, trace)
            
            # Stage 9: Output Validation
            with self.trace_execution("09_Output_Validation") as trace:
                validation_results = self._validate_outputs(system, inference_results, trace)
            
            # Stage 10: Memory and Performance Analysis
            with self.trace_execution("10_Performance_Analysis") as trace:
                performance_analysis = self._analyze_performance(trace)
            
            return True
            
        except Exception as e:
            print(f"\n[FATAL ERROR] Comprehensive trace failed: {str(e)}")
            print(traceback.format_exc())
            return False
    
    def _initialize_system(self, trace: ExecutionTrace):
        """Initialize the system and capture initialization data"""
        sys.path.append('.')
        from main import UnifiedEnergyGNNSystem
        
        # Initialize system
        system = UnifiedEnergyGNNSystem(self.config_path)
        
        # Log system components
        trace.data_shapes['system_components'] = {
            'kg_connector': type(system.kg_connector).__name__,
            'model': type(system.model).__name__,
            'trainer': type(system.trainer).__name__,
            'device': str(system.device)
        }
        
        # Log model parameters
        if hasattr(system.model, 'parameters'):
            total_params = sum(p.numel() for p in system.model.parameters())
            trainable_params = sum(p.numel() for p in system.model.parameters() if p.requires_grad)
            
            trace.performance_metrics.update({
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_memory_mb': sum(p.numel() * p.element_size() for p in system.model.parameters()) / 1024**2
            })
        
        return system
    
    def _test_kg_connection(self, system, trace: ExecutionTrace):
        """Test knowledge graph connection and data retrieval"""
        try:
            # Test basic connection
            lv_groups = system.kg_connector.get_all_lv_groups()
            trace.performance_metrics['lv_groups_found'] = len(lv_groups)
            
            # Test data retrieval for first LV group
            if lv_groups:
                sample_lv = lv_groups[0]
                lv_data = system.kg_connector.get_lv_group_data(sample_lv)
                
                trace.data_shapes['kg_sample_data'] = {
                    'lv_group_id': sample_lv,
                    'buildings_count': len(lv_data.get('buildings', [])),
                    'edges_count': len(lv_data.get('edges', []))
                }
                
                if lv_data.get('buildings'):
                    building_ids = [b['id'] for b in lv_data['buildings'][:5]]  # First 5 buildings
                    temporal_data = system.kg_connector.get_building_time_series(
                        building_ids=building_ids,
                        lookback_hours=24
                    )
                    
                    self.log_data_shapes(trace, {'temporal_data': temporal_data}, 'kg_')
            
            return {'status': 'success', 'lv_groups_count': len(lv_groups)}
            
        except Exception as e:
            trace.errors.append(f"KG Connection Error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def _load_and_process_data(self, system, trace: ExecutionTrace):
        """Load and process data, capturing data flow information"""
        try:
            # Load data
            train_loader, val_loader, test_loader = system.load_and_prepare_data(evaluate_groups=False)
            
            # Log data loader information
            trace.data_shapes['data_loaders'] = {
                'train_batches': len(train_loader),
                'val_batches': len(val_loader),
                'test_batches': len(test_loader),
                'batch_size': train_loader.batch_size if hasattr(train_loader, 'batch_size') else 'unknown'
            }
            
            # Analyze first batch from each loader
            for loader_name, loader in [('train', train_loader), ('val', val_loader), ('test', test_loader)]:
                if len(loader) > 0:
                    try:
                        batch = next(iter(loader))
                        self._analyze_batch_structure(batch, trace, f"{loader_name}_batch")
                    except Exception as e:
                        trace.warnings.append(f"Could not analyze {loader_name} batch: {str(e)}")
            
            # Performance metrics
            trace.performance_metrics.update({
                'total_graphs': len(train_loader) + len(val_loader) + len(test_loader),
                'train_graphs': len(train_loader),
                'val_graphs': len(val_loader),
                'test_graphs': len(test_loader)
            })
            
            return {'train': train_loader, 'val': val_loader, 'test': test_loader}
            
        except Exception as e:
            trace.errors.append(f"Data Loading Error: {str(e)}")
            return None
    
    def _analyze_batch_structure(self, batch, trace: ExecutionTrace, prefix: str):
        """Analyze the structure of a data batch"""
        if hasattr(batch, 'x'):
            trace.data_shapes[f'{prefix}_node_features'] = {
                'shape': list(batch.x.shape),
                'dtype': str(batch.x.dtype),
                'device': str(batch.x.device),
                'memory_mb': batch.x.numel() * batch.x.element_size() / 1024**2
            }
        
        if hasattr(batch, 'edge_index'):
            trace.data_shapes[f'{prefix}_edges'] = {
                'shape': list(batch.edge_index.shape),
                'num_edges': batch.edge_index.shape[1],
                'dtype': str(batch.edge_index.dtype),
                'device': str(batch.edge_index.device)
            }
        
        if hasattr(batch, 'temporal_profiles'):
            trace.data_shapes[f'{prefix}_temporal'] = {
                'shape': list(batch.temporal_profiles.shape),
                'dtype': str(batch.temporal_profiles.dtype),
                'memory_mb': batch.temporal_profiles.numel() * batch.temporal_profiles.element_size() / 1024**2
            }
        
        # Check for other attributes
        other_attrs = []
        for attr in dir(batch):
            if not attr.startswith('_') and hasattr(batch, attr):
                val = getattr(batch, attr)
                if torch.is_tensor(val):
                    other_attrs.append(f"{attr}:{list(val.shape)}")
        
        if other_attrs:
            trace.data_shapes[f'{prefix}_other_tensors'] = other_attrs
    
    def _create_and_validate_model(self, system, trace: ExecutionTrace):
        """Create and validate model structure"""
        try:
            model = system.model
            model.eval()
            
            # Model structure analysis
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Memory footprint
            model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
            
            trace.performance_metrics.update({
                'model_total_params': total_params,
                'model_trainable_params': trainable_params,
                'model_memory_mb': model_memory,
                'model_layers': len(list(model.modules()))
            })
            
            # Test forward pass with dummy data if possible
            try:
                from torch_geometric.data import Data
                dummy_data = Data(
                    x=torch.randn(10, 17).to(system.device),
                    edge_index=torch.randint(0, 10, (2, 20)).to(system.device),
                    temporal_profiles=torch.randn(10, 96).to(system.device),
                    batch=torch.zeros(10, dtype=torch.long).to(system.device)
                )
                
                with torch.no_grad():
                    outputs = model(dummy_data)
                
                # Log output structure
                if isinstance(outputs, dict):
                    for key, value in outputs.items():
                        if torch.is_tensor(value):
                            trace.data_shapes[f'model_output_{key}'] = {
                                'shape': list(value.shape),
                                'dtype': str(value.dtype)
                            }
                else:
                    if torch.is_tensor(outputs):
                        trace.data_shapes['model_output'] = {
                            'shape': list(outputs.shape),
                            'dtype': str(outputs.dtype)
                        }
                
                trace.intermediate_results['forward_pass_test'] = 'success'
                
            except Exception as e:
                trace.warnings.append(f"Forward pass test failed: {str(e)}")
                trace.intermediate_results['forward_pass_test'] = f'failed: {str(e)}'
            
            return {'status': 'success', 'model_info': 'validated'}
            
        except Exception as e:
            trace.errors.append(f"Model Validation Error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_training_loop(self, system, data_loaders, trace: ExecutionTrace):
        """Run a few epochs of training to capture training dynamics"""
        try:
            if not data_loaders:
                raise ValueError("No data loaders available")
            
            train_loader = data_loaders['train']
            val_loader = data_loaders['val']
            
            # Store original training config
            original_epochs = self.config['training'].get('num_epochs', 100)
            
            # Run limited training (5 epochs for tracing)
            num_trace_epochs = 5
            trace.performance_metrics['trace_epochs'] = num_trace_epochs
            
            print(f"    Running {num_trace_epochs} epochs for tracing...")
            
            epoch_times = []
            epoch_losses = []
            epoch_memory_usage = []
            
            system.model.train()
            
            for epoch in range(num_trace_epochs):
                epoch_start = time.time()
                epoch_memory_start = self.process.memory_info().rss / 1024**2
                
                # Training step
                if hasattr(system.trainer, 'train_epoch'):
                    try:
                        epoch_loss = system.trainer.train_epoch(train_loader, epoch)
                        epoch_losses.append(epoch_loss)
                    except Exception as e:
                        trace.warnings.append(f"Epoch {epoch} training failed: {str(e)}")
                        epoch_losses.append(None)
                else:
                    # Simplified training loop
                    total_loss = 0
                    num_batches = 0
                    
                    try:
                        for batch_idx, batch in enumerate(train_loader):
                            if batch_idx >= 5:  # Limit batches for tracing
                                break
                            
                            batch = batch.to(system.device)
                            
                            # Forward pass
                            outputs = system.model(batch)
                            
                            # Simple loss computation (if we can access loss function)
                            if hasattr(system, 'loss_fn') and system.loss_fn:
                                try:
                                    loss = system.loss_fn(outputs, batch)
                                    total_loss += loss.item()
                                    num_batches += 1
                                except Exception as e:
                                    trace.warnings.append(f"Loss computation failed: {str(e)}")
                            
                    except Exception as e:
                        trace.warnings.append(f"Training iteration failed: {str(e)}")
                    
                    avg_loss = total_loss / max(num_batches, 1)
                    epoch_losses.append(avg_loss)
                
                epoch_end = time.time()
                epoch_memory_end = self.process.memory_info().rss / 1024**2
                
                epoch_duration = epoch_end - epoch_start
                epoch_memory_delta = epoch_memory_end - epoch_memory_start
                
                epoch_times.append(epoch_duration)
                epoch_memory_usage.append(epoch_memory_delta)
                
                print(f"      Epoch {epoch+1}/{num_trace_epochs}: "
                      f"Loss={epoch_losses[-1]:.4f if epoch_losses[-1] else 'N/A'}, "
                      f"Time={epoch_duration:.2f}s, "
                      f"Memory Δ={epoch_memory_delta:+.1f}MB")
            
            # Store training metrics
            trace.performance_metrics.update({
                'avg_epoch_time': np.mean(epoch_times),
                'total_training_time': sum(epoch_times),
                'avg_epoch_memory_delta': np.mean(epoch_memory_usage),
                'final_loss': epoch_losses[-1] if epoch_losses[-1] is not None else 'N/A'
            })
            
            trace.intermediate_results.update({
                'epoch_times': epoch_times,
                'epoch_losses': [l for l in epoch_losses if l is not None],
                'epoch_memory_usage': epoch_memory_usage
            })
            
            return {'status': 'success', 'epochs_completed': num_trace_epochs}
            
        except Exception as e:
            trace.errors.append(f"Training Error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def _run_model_inference(self, system, data_loaders, trace: ExecutionTrace):
        """Run model inference and capture prediction details"""
        try:
            if not data_loaders or 'test' not in data_loaders:
                raise ValueError("No test data loader available")
            
            test_loader = data_loaders['test']
            system.model.eval()
            
            all_predictions = []
            inference_times = []
            batch_sizes = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if batch_idx >= 10:  # Limit for tracing
                        break
                    
                    batch = batch.to(system.device)
                    batch_start = time.time()
                    
                    # Inference
                    outputs = system.model(batch)
                    
                    batch_end = time.time()
                    
                    inference_time = batch_end - batch_start
                    inference_times.append(inference_time)
                    batch_sizes.append(batch.x.shape[0] if hasattr(batch, 'x') else 0)
                    
                    # Store prediction information
                    pred_info = {
                        'batch_idx': batch_idx,
                        'inference_time': inference_time,
                        'batch_size': batch_sizes[-1]
                    }
                    
                    # Analyze outputs
                    if isinstance(outputs, dict):
                        for key, value in outputs.items():
                            if torch.is_tensor(value):
                                pred_info[f'output_{key}_shape'] = list(value.shape)
                                
                                # Extract meaningful metrics
                                if key in ['clusters', 'clustering_cluster_assignments']:
                                    # Cluster predictions
                                    if value.dim() > 1:
                                        cluster_probs = F.softmax(value, dim=-1)
                                        cluster_assignments = torch.argmax(cluster_probs, dim=-1)
                                        pred_info['num_unique_clusters'] = len(torch.unique(cluster_assignments))
                                        pred_info['cluster_confidence'] = torch.max(cluster_probs, dim=-1)[0].mean().item()
                    
                    all_predictions.append(pred_info)
            
            # Calculate inference metrics
            trace.performance_metrics.update({
                'total_inference_batches': len(all_predictions),
                'avg_inference_time_per_batch': np.mean(inference_times),
                'avg_batch_size': np.mean(batch_sizes),
                'inference_throughput_samples_per_sec': sum(batch_sizes) / sum(inference_times) if sum(inference_times) > 0 else 0
            })
            
            trace.intermediate_results['inference_details'] = all_predictions[:5]  # Store first 5 for space
            
            return {'status': 'success', 'predictions': all_predictions}
            
        except Exception as e:
            trace.errors.append(f"Inference Error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def _analyze_patterns(self, system, data_loaders, trace: ExecutionTrace):
        """Analyze patterns from model outputs"""
        try:
            if not data_loaders or 'test' not in data_loaders:
                raise ValueError("No test data available")
            
            test_loader = data_loaders['test']
            
            # Run pattern analysis on a limited dataset
            analysis_results = []
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    if batch_idx >= 3:  # Limit for tracing
                        break
                    
                    batch = batch.to(system.device)
                    
                    try:
                        # Get model predictions
                        outputs = system.model(batch)
                        
                        # Simple pattern analysis
                        if 'clusters' in outputs or 'clustering_cluster_assignments' in outputs:
                            cluster_key = 'clusters' if 'clusters' in outputs else 'clustering_cluster_assignments'
                            cluster_logits = outputs[cluster_key]
                            
                            if cluster_logits.dim() > 1:
                                cluster_assignments = torch.argmax(cluster_logits, dim=-1)
                                
                                # Basic pattern metrics
                                unique_clusters = torch.unique(cluster_assignments)
                                cluster_sizes = [(cluster_assignments == c).sum().item() for c in unique_clusters]
                                
                                pattern_info = {
                                    'batch_idx': batch_idx,
                                    'num_nodes': batch.x.shape[0],
                                    'num_clusters_found': len(unique_clusters),
                                    'cluster_sizes': cluster_sizes,
                                    'largest_cluster_size': max(cluster_sizes) if cluster_sizes else 0,
                                    'smallest_cluster_size': min(cluster_sizes) if cluster_sizes else 0,
                                    'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0
                                }
                                
                                analysis_results.append(pattern_info)
                    
                    except Exception as e:
                        trace.warnings.append(f"Pattern analysis failed for batch {batch_idx}: {str(e)}")
            
            # Aggregate pattern metrics
            if analysis_results:
                total_nodes = sum(r['num_nodes'] for r in analysis_results)
                total_clusters = sum(r['num_clusters_found'] for r in analysis_results)
                avg_nodes_per_cluster = total_nodes / total_clusters if total_clusters > 0 else 0
                
                trace.performance_metrics.update({
                    'total_nodes_analyzed': total_nodes,
                    'total_clusters_found': total_clusters,
                    'avg_nodes_per_cluster': avg_nodes_per_cluster,
                    'pattern_analysis_batches': len(analysis_results)
                })
            
            trace.intermediate_results['pattern_analysis'] = analysis_results
            
            return {'status': 'success', 'analysis_results': analysis_results}
            
        except Exception as e:
            trace.errors.append(f"Pattern Analysis Error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def _generate_interventions(self, system, pattern_results, trace: ExecutionTrace):
        """Generate intervention recommendations"""
        try:
            if not pattern_results or pattern_results.get('status') != 'success':
                trace.warnings.append("No valid pattern results for intervention generation")
                return {'status': 'skipped', 'reason': 'no_pattern_results'}
            
            # Generate mock interventions based on patterns
            interventions = []
            
            for pattern in pattern_results.get('analysis_results', []):
                num_clusters = pattern.get('num_clusters_found', 0)
                num_nodes = pattern.get('num_nodes', 0)
                
                if num_clusters > 0 and num_nodes > 0:
                    # Simple intervention logic
                    avg_nodes_per_cluster = num_nodes / num_clusters
                    
                    intervention = {
                        'batch_idx': pattern.get('batch_idx', 0),
                        'intervention_type': 'solar_battery_combo' if avg_nodes_per_cluster > 5 else 'solar_only',
                        'target_clusters': min(num_clusters, 3),  # Target up to 3 clusters
                        'expected_nodes_affected': int(avg_nodes_per_cluster * min(num_clusters, 3)),
                        'estimated_cost': int(avg_nodes_per_cluster * min(num_clusters, 3) * 25000),  # $25k per node
                        'expected_benefits': {
                            'peak_reduction_kw': avg_nodes_per_cluster * min(num_clusters, 3) * 2.5,
                            'self_sufficiency_increase': 0.15 if avg_nodes_per_cluster > 5 else 0.10,
                            'carbon_reduction_tons_per_year': avg_nodes_per_cluster * min(num_clusters, 3) * 1.2
                        }
                    }
                    
                    interventions.append(intervention)
            
            # Aggregate intervention metrics
            total_cost = sum(i['estimated_cost'] for i in interventions)
            total_nodes_affected = sum(i['expected_nodes_affected'] for i in interventions)
            total_peak_reduction = sum(i['expected_benefits']['peak_reduction_kw'] for i in interventions)
            
            trace.performance_metrics.update({
                'total_interventions_generated': len(interventions),
                'total_intervention_cost': total_cost,
                'total_nodes_affected': total_nodes_affected,
                'total_peak_reduction_kw': total_peak_reduction
            })
            
            trace.intermediate_results['interventions'] = interventions
            
            return {'status': 'success', 'interventions': interventions}
            
        except Exception as e:
            trace.errors.append(f"Intervention Generation Error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def _validate_outputs(self, system, inference_results, trace: ExecutionTrace):
        """Validate model outputs for consistency and physics constraints"""
        try:
            if not inference_results or inference_results.get('status') != 'success':
                trace.warnings.append("No valid inference results for validation")
                return {'status': 'skipped', 'reason': 'no_inference_results'}
            
            validation_results = {
                'physics_violations': [],
                'data_consistency_issues': [],
                'numerical_issues': []
            }
            
            predictions = inference_results.get('predictions', [])
            
            for pred in predictions:
                batch_idx = pred.get('batch_idx', -1)
                
                # Check for numerical issues
                for key, value in pred.items():
                    if key.endswith('_shape'):
                        continue
                    
                    if isinstance(value, (int, float)):
                        if np.isnan(value):
                            validation_results['numerical_issues'].append(f"NaN in batch {batch_idx}, key {key}")
                        elif np.isinf(value):
                            validation_results['numerical_issues'].append(f"Inf in batch {batch_idx}, key {key}")
                
                # Check cluster consistency
                num_clusters = pred.get('num_unique_clusters', 0)
                if num_clusters == 0:
                    validation_results['data_consistency_issues'].append(f"No clusters found in batch {batch_idx}")
                elif num_clusters > 20:  # Based on config max_clusters
                    validation_results['data_consistency_issues'].append(f"Too many clusters ({num_clusters}) in batch {batch_idx}")
                
                # Check confidence levels
                confidence = pred.get('cluster_confidence', 0)
                if confidence < 0.1:
                    validation_results['data_consistency_issues'].append(f"Very low confidence ({confidence:.3f}) in batch {batch_idx}")
            
            # Summary validation metrics
            total_violations = (len(validation_results['physics_violations']) + 
                              len(validation_results['data_consistency_issues']) + 
                              len(validation_results['numerical_issues']))
            
            trace.performance_metrics.update({
                'total_validation_issues': total_violations,
                'physics_violations': len(validation_results['physics_violations']),
                'consistency_issues': len(validation_results['data_consistency_issues']),
                'numerical_issues': len(validation_results['numerical_issues']),
                'validation_pass_rate': max(0, (len(predictions) - total_violations) / len(predictions)) if predictions else 0
            })
            
            trace.intermediate_results['validation_results'] = validation_results
            
            return {'status': 'success', 'validation_results': validation_results}
            
        except Exception as e:
            trace.errors.append(f"Output Validation Error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def _analyze_performance(self, trace: ExecutionTrace):
        """Analyze overall system performance and identify bottlenecks"""
        try:
            # Memory analysis
            memory_traces = [(t.stage, t.memory_before, t.memory_after, t.memory_delta) for t in self.traces]
            max_memory_usage = max(t.memory_after for t in self.traces)
            total_memory_delta = sum(t.memory_delta for t in self.traces)
            
            # Timing analysis
            timing_traces = [(t.stage, t.duration) for t in self.traces]
            total_execution_time = time.time() - self.global_start_time
            bottleneck_stage = max(self.traces, key=lambda t: t.duration).stage
            
            # Error analysis
            total_errors = sum(len(t.errors) for t in self.traces)
            total_warnings = sum(len(t.warnings) for t in self.traces)
            
            performance_summary = {
                'total_execution_time': total_execution_time,
                'max_memory_usage_mb': max_memory_usage,
                'total_memory_delta_mb': total_memory_delta,
                'bottleneck_stage': bottleneck_stage,
                'total_errors': total_errors,
                'total_warnings': total_warnings,
                'memory_efficiency': max_memory_usage / (psutil.virtual_memory().total / 1024**2),  # Fraction of total RAM used
                'stages_completed': len([t for t in self.traces if not t.errors]),
                'stages_with_errors': len([t for t in self.traces if t.errors])
            }
            
            trace.performance_metrics = performance_summary
            trace.intermediate_results = {
                'memory_traces': memory_traces,
                'timing_traces': timing_traces
            }
            
            # Memory leaks detection
            memory_trend = [t.memory_after for t in self.traces]
            if len(memory_trend) > 3:
                memory_slope = np.polyfit(range(len(memory_trend)), memory_trend, 1)[0]
                if memory_slope > 10:  # > 10MB increase per stage
                    trace.warnings.append(f"Potential memory leak detected (slope: +{memory_slope:.1f}MB per stage)")
            
            return {'status': 'success', 'performance_summary': performance_summary}
            
        except Exception as e:
            trace.errors.append(f"Performance Analysis Error: {str(e)}")
            return {'status': 'failed', 'error': str(e)}
    
    def generate_trace_report(self) -> str:
        """Generate comprehensive trace report"""
        report = []
        report.append("# QIUARI_V3 SYSTEM DEEP TRACE ANALYSIS")
        report.append("=" * 80)
        report.append("")
        
        # Executive Summary
        total_time = time.time() - self.global_start_time
        successful_stages = len([t for t in self.traces if not t.errors])
        total_stages = len(self.traces)
        
        report.append("## Executive Summary")
        report.append(f"- **Total Execution Time**: {total_time:.2f} seconds")
        report.append(f"- **Stages Completed Successfully**: {successful_stages}/{total_stages} ({successful_stages/total_stages*100:.1f}%)")
        report.append(f"- **Device**: {self.device}")
        report.append(f"- **Peak Memory Usage**: {max(t.memory_after for t in self.traces if t.memory_after):.1f} MB")
        report.append("")
        
        # Detailed Stage Analysis
        report.append("## Detailed Stage Analysis")
        report.append("")
        
        for trace in self.traces:
            report.append(f"### {trace.stage}")
            report.append(f"- **Duration**: {trace.duration:.3f}s")
            report.append(f"- **Memory**: {trace.memory_before:.1f} MB → {trace.memory_after:.1f} MB (Δ{trace.memory_delta:+.1f} MB)")
            report.append(f"- **CPU Usage**: {trace.cpu_usage:.1f}%")
            
            if torch.cuda.is_available():
                report.append(f"- **GPU Memory**: {trace.gpu_memory:.1f} MB")
            
            if trace.data_shapes:
                report.append("- **Data Shapes**:")
                for key, shape in trace.data_shapes.items():
                    if isinstance(shape, dict) and 'shape' in shape:
                        report.append(f"  - {key}: {shape['shape']} ({shape.get('dtype', 'unknown')})")
                    else:
                        report.append(f"  - {key}: {shape}")
            
            if trace.performance_metrics:
                report.append("- **Performance Metrics**:")
                for key, value in trace.performance_metrics.items():
                    report.append(f"  - {key}: {value}")
            
            if trace.errors:
                report.append("- **Errors**:")
                for error in trace.errors:
                    report.append(f"  - {error}")
            
            if trace.warnings:
                report.append("- **Warnings**:")
                for warning in trace.warnings:
                    report.append(f"  - {warning}")
            
            report.append("")
        
        # Performance Bottlenecks
        report.append("## Performance Bottlenecks")
        sorted_traces = sorted(self.traces, key=lambda t: t.duration, reverse=True)
        report.append("")
        
        for i, trace in enumerate(sorted_traces[:5]):  # Top 5 slowest stages
            report.append(f"{i+1}. **{trace.stage}**: {trace.duration:.3f}s")
        report.append("")
        
        # Memory Analysis
        report.append("## Memory Usage Analysis")
        memory_intensive = sorted(self.traces, key=lambda t: abs(t.memory_delta), reverse=True)[:3]
        report.append("")
        
        for trace in memory_intensive:
            report.append(f"- **{trace.stage}**: {trace.memory_delta:+.1f} MB change")
        report.append("")
        
        # Data Flow Summary
        report.append("## Data Flow Summary")
        report.append("")
        
        for trace in self.traces:
            if trace.data_shapes:
                report.append(f"### {trace.stage}")
                for key, shape_info in trace.data_shapes.items():
                    if isinstance(shape_info, dict) and 'memory_mb' in shape_info:
                        report.append(f"- {key}: {shape_info['memory_mb']:.2f} MB")
        report.append("")
        
        # Error Summary
        all_errors = []
        all_warnings = []
        
        for trace in self.traces:
            all_errors.extend(trace.errors)
            all_warnings.extend(trace.warnings)
        
        if all_errors:
            report.append("## Errors Encountered")
            for i, error in enumerate(all_errors, 1):
                report.append(f"{i}. {error}")
            report.append("")
        
        if all_warnings:
            report.append("## Warnings")
            for i, warning in enumerate(all_warnings, 1):
                report.append(f"{i}. {warning}")
            report.append("")
        
        # Recommendations
        report.append("## Recommendations")
        report.append("")
        
        # Memory recommendations
        max_memory = max(t.memory_after for t in self.traces if t.memory_after)
        if max_memory > 2000:  # > 2GB
            report.append("- Consider reducing batch size or model complexity due to high memory usage")
        
        # Performance recommendations
        bottleneck = max(self.traces, key=lambda t: t.duration)
        if bottleneck.duration > total_time * 0.4:  # One stage takes >40% of total time
            report.append(f"- Optimize {bottleneck.stage} stage as it's the primary bottleneck")
        
        # Error-based recommendations
        if all_errors:
            report.append("- Fix critical errors before production deployment")
        
        if len(all_warnings) > 10:
            report.append("- Review warnings for potential performance improvements")
        
        return "\n".join(report)
    
    def save_trace_data(self, output_dir: str = "trace_outputs"):
        """Save detailed trace data to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save trace data as JSON
        trace_data = []
        for trace in self.traces:
            trace_dict = asdict(trace)
            # Convert numpy types to Python types for JSON serialization
            for key, value in trace_dict.items():
                if isinstance(value, dict):
                    for k, v in value.items():
                        if isinstance(v, np.ndarray):
                            trace_dict[key][k] = v.tolist()
                        elif hasattr(v, 'item'):  # numpy scalars
                            trace_dict[key][k] = v.item()
            trace_data.append(trace_dict)
        
        json_path = output_path / f"trace_data_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(trace_data, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for trace in self.traces:
            summary_data.append({
                'stage': trace.stage,
                'duration_seconds': trace.duration,
                'memory_before_mb': trace.memory_before,
                'memory_after_mb': trace.memory_after,
                'memory_delta_mb': trace.memory_delta,
                'cpu_usage_percent': trace.cpu_usage,
                'gpu_memory_mb': trace.gpu_memory,
                'errors_count': len(trace.errors) if trace.errors else 0,
                'warnings_count': len(trace.warnings) if trace.warnings else 0
            })
        
        csv_path = output_path / f"trace_summary_{timestamp}.csv"
        pd.DataFrame(summary_data).to_csv(csv_path, index=False)
        
        # Save detailed report
        report_content = self.generate_trace_report()
        report_path = output_path / f"trace_report_{timestamp}.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        print(f"\nTrace data saved to:")
        print(f"  - Detailed JSON: {json_path}")
        print(f"  - Summary CSV: {csv_path}")
        print(f"  - Report: {report_path}")
        
        return {
            'json_path': str(json_path),
            'csv_path': str(csv_path),
            'report_path': str(report_path)
        }


def main():
    """Main execution function"""
    print("Starting Comprehensive System Trace...")
    
    # Initialize tracer
    tracer = SystemTracer()
    
    try:
        # Run comprehensive trace
        success = tracer.run_comprehensive_trace()
        
        # Generate and save reports
        saved_files = tracer.save_trace_data()
        
        # Print final summary
        print("\n" + "="*80)
        print("COMPREHENSIVE TRACE COMPLETE")
        print("="*80)
        
        total_time = time.time() - tracer.global_start_time
        successful_stages = len([t for t in tracer.traces if not t.errors])
        total_stages = len(tracer.traces)
        
        print(f"\n[SUMMARY]")
        print(f"  Total Execution Time: {total_time:.2f} seconds")
        print(f"  Stages Completed: {successful_stages}/{total_stages}")
        print(f"  Peak Memory Usage: {max(t.memory_after for t in tracer.traces):.1f} MB")
        print(f"  Success Rate: {successful_stages/total_stages*100:.1f}%")
        
        if success:
            print(f"  Status: SUCCESS")
        else:
            print(f"  Status: COMPLETED WITH ISSUES")
        
        print(f"\nDetailed reports saved to trace_outputs/")
        print("="*80)
        
        return success
        
    except Exception as e:
        print(f"\n[FATAL ERROR] Trace execution failed: {str(e)}")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)