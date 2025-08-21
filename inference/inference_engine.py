# inference/inference_engine.py
"""
Inference engine for running trained GNN model
Handles model loading, preprocessing, and prediction
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData, Batch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import pickle
import yaml
import time

logger = logging.getLogger(__name__)

class InferenceEngine:
    """Main inference engine for energy GNN"""
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = "config/config.yaml",
                 device: str = None):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(device)
        
        # Load model
        self.model_path = model_path  # Store model path
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # Load preprocessors
        self._load_preprocessors()
        
        # Cache for repeated queries
        self.cache = {}
        self.cache_size = 100
        
        logger.info(f"Initialized InferenceEngine on {self.device}")
    
    def _load_model(self, model_path: str) -> nn.Module:
        """Load trained model from checkpoint"""
        from models.base_gnn import create_gnn_model
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create model architecture
        model_config = checkpoint.get('config', self.config)['model']
        model = create_gnn_model('homo', model_config)  # Adjust type as needed
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        
        logger.info(f"Loaded model from {model_path}")
        
        return model
    
    def _load_preprocessors(self):
        """Load data preprocessors (scalers, encoders)"""
        preprocessor_path = Path(self.config['paths']['processed_data']) / 'graph_metadata.pkl'
        
        if preprocessor_path.exists():
            with open(preprocessor_path, 'rb') as f:
                metadata = pickle.load(f)
                self.scalers = metadata.get('scalers', {})
                self.encoders = metadata.get('encoders', {})
                self.node_mappings = metadata.get('node_mappings', {})
        else:
            logger.warning("Preprocessors not found, using defaults")
            self.scalers = {}
            self.encoders = {}
            self.node_mappings = {}
    
    def predict(self, 
                graph_data: Union[Data, HeteroData, Dict],
                task: str = 'all',
                return_embeddings: bool = False) -> Dict:
        """
        Run inference on graph data
        
        Args:
            graph_data: Input graph data
            task: Specific task or 'all'
            return_embeddings: Whether to return node embeddings
            
        Returns:
            Prediction results
        """
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(graph_data, task)
        if cache_key in self.cache:
            logger.info(f"Using cached result for {task}")
            return self.cache[cache_key]
        
        # Preprocess data
        processed_data = self._preprocess(graph_data)
        
        # Move to device
        if isinstance(processed_data, Data):
            processed_data = processed_data.to(self.device)
        elif isinstance(processed_data, HeteroData):
            processed_data = processed_data.to(self.device)
        
        # Run inference
        with torch.no_grad():
            if task == 'all':
                outputs = self.model(processed_data)
            else:
                # Run specific task head
                outputs = self._run_task(processed_data, task)
        
        # Post-process outputs
        results = self._postprocess(outputs, task)
        
        # Add embeddings if requested
        if return_embeddings:
            results['embeddings'] = self._extract_embeddings(outputs)
        
        # Add metadata
        results['inference_time'] = time.time() - start_time
        results['device'] = str(self.device)
        results['model_path'] = str(self.model_path)
        
        # Update cache
        self._update_cache(cache_key, results)
        
        logger.info(f"Inference completed in {results['inference_time']:.2f}s")
        
        return results
    
    def _preprocess(self, graph_data: Union[Data, Dict]) -> Union[Data, HeteroData]:
        """Preprocess input data"""
        if isinstance(graph_data, (Data, HeteroData)):
            return graph_data
        
        # Convert dictionary to PyTorch Geometric format
        from data.graph_builder import GraphBuilder
        
        builder = GraphBuilder(graph_data, self.config)
        
        if 'buildings' in graph_data.get('nodes', {}):
            # Homogeneous graph
            graph = builder.build_homogeneous_graph()
        else:
            # Heterogeneous graph
            graph = builder.build_heterogeneous_graph()
        
        return graph
    
    def _run_task(self, data: Union[Data, HeteroData], task: str) -> Dict:
        """Run specific task head"""
        # Get base embeddings
        if hasattr(self.model, 'encode'):
            embeddings = self.model.encode(data)
        else:
            embeddings = self.model(data.x, data.edge_index)
        
        # Run task-specific head
        task_outputs = {}
        
        if hasattr(self.model, 'task_heads') and task in self.model.task_heads:
            task_head = self.model.task_heads[task]
            
            if task == 'clustering':
                # Need adjacency matrix for clustering
                adjacency = self._build_adjacency(data)
                task_outputs = task_head(embeddings, adjacency)
            elif task in ['thermal', 'p2p']:
                # Need edge index for these tasks
                task_outputs = task_head(embeddings, data.edge_index)
            else:
                task_outputs = task_head(embeddings)
        else:
            logger.warning(f"Task {task} not found, returning embeddings only")
            task_outputs = {'embeddings': embeddings}
        
        return task_outputs
    
    def _build_adjacency(self, data: Data) -> torch.Tensor:
        """Build adjacency matrix from edge index"""
        num_nodes = data.num_nodes
        adjacency = torch.zeros(num_nodes, num_nodes, device=self.device)
        
        edge_index = data.edge_index
        adjacency[edge_index[0], edge_index[1]] = 1
        
        return adjacency
    
    def _postprocess(self, outputs: Dict, task: str) -> Dict:
        """Post-process model outputs"""
        results = {}
        
        # Task-specific post-processing
        if task == 'clustering' or 'clustering' in outputs:
            results['clustering'] = self._postprocess_clustering(
                outputs.get('clustering', outputs)
            )
        
        if task == 'solar' or 'solar' in outputs:
            results['solar'] = self._postprocess_solar(
                outputs.get('solar', outputs)
            )
        
        if task == 'retrofit' or 'retrofit' in outputs:
            results['retrofit'] = self._postprocess_retrofit(
                outputs.get('retrofit', outputs)
            )
        
        # Add more task-specific postprocessing as needed
        
        return results
    
    def _postprocess_clustering(self, outputs: Dict) -> Dict:
        """Post-process clustering outputs"""
        results = {}
        
        if 'hard_assignment' in outputs:
            assignments = outputs['hard_assignment'].cpu().numpy()
            
            # Group buildings by cluster
            clusters = {}
            for idx, cluster_id in enumerate(assignments):
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(idx)
            
            results['clusters'] = clusters
            results['num_clusters'] = len(clusters)
        
        if 'soft_assignment' in outputs:
            results['confidence'] = outputs['soft_assignment'].max(dim=1)[0].mean().item()
        
        if 'modularity' in outputs:
            results['modularity'] = outputs['modularity'].item()
        
        return results
    
    def _postprocess_solar(self, outputs: Dict) -> Dict:
        """Post-process solar optimization outputs"""
        results = {}
        
        if 'solar_score' in outputs:
            scores = outputs['solar_score'].cpu().numpy()
            
            # Rank buildings
            ranking = np.argsort(scores)[::-1]
            results['ranking'] = ranking.tolist()
            results['top_10'] = ranking[:10].tolist()
        
        if 'capacity_kwp' in outputs:
            capacities = outputs['capacity_kwp'].cpu().numpy()
            results['total_capacity'] = float(np.sum(capacities))
            results['capacities'] = capacities.tolist()
        
        if 'roi_years' in outputs:
            roi = outputs['roi_years'].cpu().numpy()
            results['avg_roi'] = float(np.mean(roi))
            results['viable_count'] = int(np.sum(roi < 10))
        
        return results
    
    def _postprocess_retrofit(self, outputs: Dict) -> Dict:
        """Post-process retrofit outputs"""
        results = {}
        
        if 'retrofit_score' in outputs:
            scores = outputs['retrofit_score'].cpu().numpy()
            ranking = np.argsort(scores)[::-1]
            results['priority_ranking'] = ranking.tolist()
        
        if 'energy_savings' in outputs:
            savings = outputs['energy_savings'].cpu().numpy()
            results['total_savings_potential'] = float(np.sum(savings))
            results['avg_savings'] = float(np.mean(savings))
        
        if 'retrofit_cost' in outputs:
            costs = outputs['retrofit_cost'].cpu().numpy()
            results['total_investment'] = float(np.sum(costs))
        
        return results
    
    def _extract_embeddings(self, outputs: Dict) -> np.ndarray:
        """Extract node embeddings from outputs"""
        if 'embeddings' in outputs:
            return outputs['embeddings'].cpu().numpy()
        
        # Look for embeddings in task outputs
        for key, value in outputs.items():
            if isinstance(value, dict) and 'embeddings' in value:
                return value['embeddings'].cpu().numpy()
            elif isinstance(value, torch.Tensor) and value.dim() == 2:
                # Assume 2D tensors are embeddings
                return value.cpu().numpy()
        
        return None
    
    def _get_cache_key(self, graph_data: Any, task: str) -> str:
        """Generate cache key for query"""
        # Simple hash based on data size and task
        if isinstance(graph_data, Data):
            key = f"{graph_data.num_nodes}_{graph_data.num_edges}_{task}"
        elif isinstance(graph_data, dict):
            key = f"{len(graph_data.get('nodes', {}))}_{task}"
        else:
            key = f"{hash(str(graph_data))}_{task}"
        
        return key
    
    def _update_cache(self, key: str, value: Dict):
        """Update result cache"""
        # Implement LRU cache
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    def batch_predict(self, 
                      graph_list: List[Union[Data, Dict]],
                      task: str = 'all',
                      batch_size: int = 32) -> List[Dict]:
        """
        Run batch inference on multiple graphs
        
        Args:
            graph_list: List of graph data
            task: Task to run
            batch_size: Batch size for inference
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i in range(0, len(graph_list), batch_size):
            batch = graph_list[i:i+batch_size]
            
            # Process batch
            batch_data = []
            for graph in batch:
                processed = self._preprocess(graph)
                batch_data.append(processed)
            
            # Create batch
            if isinstance(batch_data[0], Data):
                batched = Batch.from_data_list(batch_data)
            else:
                # Handle heterogeneous data
                batched = batch_data  # Process individually for now
            
            # Run inference
            if isinstance(batched, list):
                # Process individually
                for data in batched:
                    result = self.predict(data, task)
                    results.append(result)
            else:
                # Batch processing
                with torch.no_grad():
                    outputs = self.model(batched.to(self.device))
                
                # Split results
                batch_results = self._split_batch_results(outputs, batch_size)
                results.extend(batch_results)
        
        return results
    
    def _split_batch_results(self, outputs: Dict, batch_size: int) -> List[Dict]:
        """Split batched outputs into individual results"""
        results = []
        
        # Simple splitting - needs refinement based on actual output structure
        for i in range(batch_size):
            result = {}
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() > 0 and value.shape[.0] == batch_size:
                        result[key] = value[i]
                    else:
                        result[key] = value
                elif isinstance(value, dict):
                    result[key] = {k: v[i] if isinstance(v, torch.Tensor) and v.dim() > 0 else v 
                                  for k, v in value.items()}
            results.append(result)
        
        return results
    
    def explain_prediction(self, 
                          graph_data: Union[Data, Dict],
                          task: str,
                          node_idx: Optional[int] = None) -> Dict:
        """
        Generate explanation for predictions
        
        Args:
            graph_data: Input graph
            task: Task to explain
            node_idx: Specific node to explain (optional)
            
        Returns:
            Explanation dictionary
        """
        from captum.attr import IntegratedGradients, GradientShap
        
        # Prepare data
        data = self._preprocess(graph_data).to(self.device)
        
        # Get base prediction
        self.model.eval()
        with torch.no_grad():
            outputs = self._run_task(data, task)
        
        # Create explainer
        def model_forward(x):
            data.x = x
            return self._run_task(data, task)['score']  # Adjust based on task
        
        # Calculate attributions
        ig = IntegratedGradients(model_forward)
        
        if node_idx is not None:
            # Explain specific node
            attributions = ig.attribute(
                data.x[node_idx:node_idx+1],
                target=0,
                n_steps=50
            )
        else:
            # Explain all nodes
            attributions = ig.attribute(
                data.x,
                target=0,
                n_steps=50
            )
        
        # Process attributions
        feature_importance = torch.abs(attributions).mean(dim=0).cpu().numpy()
        
        explanation = {
            'feature_importance': feature_importance.tolist(),
            'top_features': np.argsort(feature_importance)[-10:][::-1].tolist(),
            'prediction': outputs,
            'confidence': self._calculate_confidence(outputs)
        }
        
        return explanation
    
    def _calculate_confidence(self, outputs: Dict) -> float:
        """Calculate prediction confidence"""
        if 'confidence' in outputs:
            return outputs['confidence'].item()
        
        # Task-specific confidence calculation
        confidence = 0.5  # Default
        
        if 'soft_assignment' in outputs:
            # For clustering - entropy of soft assignments
            probs = outputs['soft_assignment']
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
            confidence = 1 - (entropy / np.log(probs.shape[1]))  # Normalize
        
        return float(confidence)

class StreamingInference:
    """Handle streaming/real-time inference"""
    
    def __init__(self, engine: InferenceEngine):
        """
        Initialize streaming inference
        
        Args:
            engine: Base inference engine
        """
        self.engine = engine
        self.buffer = []
        self.buffer_size = 10
        self.last_update = time.time()
        self.update_interval = 60  # seconds
    
    def add_data(self, new_data: Dict):
        """Add new data point to buffer"""
        self.buffer.append(new_data)
        
        # Check if we should run inference
        if len(self.buffer) >= self.buffer_size or \
           time.time() - self.last_update > self.update_interval:
            return self.process_buffer()
        
        return None
    
    def process_buffer(self) -> Dict:
        """Process accumulated data"""
        if not self.buffer:
            return None
        
        # Combine buffer data
        combined_data = self._combine_buffer_data()
        
        # Run inference
        results = self.engine.predict(combined_data)
        
        # Clear buffer
        self.buffer = []
        self.last_update = time.time()
        
        return results
    
    def _combine_buffer_data(self) -> Dict:
        """Combine buffered data points"""
        # Implementation depends on data structure
        combined = {
            'nodes': {},
            'edges': {},
            'temporal': []
        }
        
        for data in self.buffer:
            # Merge nodes
            for node_type, nodes in data.get('nodes', {}).items():
                if node_type not in combined['nodes']:
                    combined['nodes'][node_type] = []
                combined['nodes'][node_type].extend(nodes)
            
            # Merge edges
            for edge_type, edges in data.get('edges', {}).items():
                if edge_type not in combined['edges']:
                    combined['edges'][edge_type] = []
                combined['edges'][edge_type].extend(edges)
            
            # Add temporal data
            if 'temporal' in data:
                combined['temporal'].append(data['temporal'])
        
        return combined

# Usage example
if __name__ == "__main__":
    # Initialize inference engine
    engine = InferenceEngine(
        model_path="checkpoints/best_model.pth",
        config_path="config/config.yaml"
    )
    
    # Mock graph data
    graph_data = {
        'nodes': {
            'buildings': pd.DataFrame({
                'area': np.random.rand(100) * 200,
                'peak_demand': np.random.rand(100) * 20,
                'has_solar': np.random.choice([0, 1], 100)
            })
        },
        'edges': {
            'electrical': pd.DataFrame({
                'source': np.random.randint(0, 100, 200),
                'target': np.random.randint(0, 100, 200)
            })
        }
    }
    
    # Run inference
    results = engine.predict(graph_data, task='clustering')
    
    print("Inference Results:")
    print(f"  Clusters: {results.get('clustering', {}).get('num_clusters', 0)}")
    print(f"  Inference time: {results['inference_time']:.2f}s")