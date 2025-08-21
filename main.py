# main.py
"""
Main entry point for the Energy GNN System
Orchestrates data processing, training, inference, and task execution
"""

import argparse
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
import yaml
import json
import logging
from typing import Dict, Optional, List, Any
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
import pandas as pd
from datetime import datetime

# Import custom modules
from utils.logger import setup_logging, TaskLogger, ExperimentLogger
from utils.metrics_tracker import MetricsTracker, PerformanceMonitor, Timer
from utils.visualization import GraphVisualizer

from data.kg_connector import KGConnector
from data.data_loader import EnergyDataLoader
from data.graph_builder import GraphBuilder
from data.preprocessor import DataPreprocessor

from models.base_gnn import create_gnn_model

from training.multi_task_trainer import MultiTaskTrainer, AdaptiveMultiTaskTrainer
from training.evaluation_metrics import EvaluationMetrics
from training.validation import ComprehensiveValidator

from inference.query_processor import QueryProcessor, QueryValidator
from inference.inference_engine import InferenceEngine, StreamingInference
from inference.kg_updater import KGUpdater

from tasks.clustering import EnergyCommunityClustering
from tasks.solar_optimization import SolarOptimization
from tasks.retrofit_targeting import RetrofitTargeting

class EnergyGNNSystem:
    """Main system orchestrator for Energy GNN"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the Energy GNN System
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup logging
        self.logger = setup_logging(self.config.get('logging', {}))
        self.logger.info("="*50)
        self.logger.info("Energy GNN System Initialized")
        self.logger.info(f"Configuration loaded from {config_path}")
        
        # Initialize metrics tracker
        self.metrics_tracker = MetricsTracker(
            db_path=self.config['paths']['metrics_db']
        )
        self.performance_monitor = PerformanceMonitor(self.metrics_tracker)
        
        # Initialize components (lazy loading)
        self.kg_connector = None
        self.data_loader = None
        self.model = None
        self.inference_engine = None
        self.query_processor = None
        self.visualizer = None
        
        # Track system state
        self.state = {
            'model_loaded': False,
            'kg_connected': False,
            'last_training': None,
            'last_inference': None
        }
        
        self.logger.info("System initialization complete")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load task configuration
        task_config_path = Path(config_path).parent / 'tasks_config.yaml'
        if task_config_path.exists():
            with open(task_config_path, 'r') as f:
                config['tasks'] = yaml.safe_load(f)
        
        return config
    
    def connect_kg(self) -> bool:
        """Connect to Neo4j Knowledge Graph"""
        try:
            self.logger.info("Connecting to Neo4j Knowledge Graph...")
            
            self.kg_connector = KGConnector(
                uri=self.config['neo4j']['uri'],
                user=self.config['neo4j']['user'],
                password=self.config['neo4j']['password']
            )
            
            # Test connection
            stats = self.kg_connector.get_statistics()
            self.logger.info(f"Connected to KG with {stats['total_nodes']} nodes")
            
            self.state['kg_connected'] = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to KG: {e}")
            return False
    
    def prepare_data(self, force_rebuild: bool = False) -> Dict:
        """
        Prepare data for training/inference
        
        Args:
            force_rebuild: Force rebuilding of graph data
            
        Returns:
            Prepared graph data
        """
        self.logger.info("Preparing data...")
        
        with Timer(self.metrics_tracker, 'data_preparation'):
            # Check if processed data exists
            processed_path = Path(self.config['paths']['processed_data'])
            graph_file = processed_path / 'graph_data.pt'
            
            if graph_file.exists() and not force_rebuild:
                self.logger.info("Loading existing graph data...")
                graph_data = torch.load(graph_file)
            else:
                # Load from KG if connected
                if self.state['kg_connected']:
                    self.logger.info("Fetching data from Knowledge Graph...")
                    
                    # Fetch building data with all properties
                    buildings_query = """
                        MATCH (b:Building)
                        RETURN b.id as id, 
                               b.id as building_id,
                               b.area as area,
                               b.roof_area as roof_area, 
                               b.suitable_roof_area as suitable_roof_area,
                               b.orientation as building_orientation_cardinal,
                               b.height as height,
                               b.has_solar as has_solar,
                               b.peak_demand as peak_demand,
                               b.avg_demand as avg_demand,
                               b.lv_network as lv_network,
                               b.solar_capacity_kwp as solar_capacity_kwp,
                               b.x as x,
                               b.y as y,
                               b.function as function,
                               b.type as type,
                               0.0 as load_factor,
                               0.0 as variability,
                               COALESCE(b.suitable_roof_area, b.roof_area * 0.7) as suitable_roof,
                               COALESCE(b.has_battery, false) as has_battery,
                               false as has_heat_pump
                    """
                    buildings_data = self.kg_connector.query(buildings_query)
                    
                    # Fetch relationships (connections between buildings and LV networks)
                    edges_query = """
                        MATCH (n1)-[r:CONNECTED_TO]-(n2)
                        RETURN id(n1) as source, id(n2) as target, 
                               type(r) as type, r.distance as distance
                    """
                    edges_data = self.kg_connector.query(edges_query)
                    
                    # Convert query results to DataFrame
                    buildings_df = pd.DataFrame(buildings_data)
                    edges_df = pd.DataFrame(edges_data) if edges_data else pd.DataFrame()
                    
                    # Build graph
                    extracted_data = {
                        'nodes': {'buildings': buildings_df},
                        'edges': {'connections': edges_df} if not edges_df.empty else {}
                    }
                    graph_builder = GraphBuilder(extracted_data, self.config)
                    graph_data = graph_builder.build_homogeneous_graph()
                    
                    # Attach building data to graph for later use
                    graph_data.building_data = buildings_df
                else:
                    # Load from files
                    self.logger.info("Loading data from files...")
                    self.data_loader = EnergyDataLoader(self.config)
                    graph_data = self.data_loader.load_all_data()
                
                # Save processed data
                torch.save(graph_data, graph_file)
                self.logger.info(f"Saved graph data to {graph_file}")
            
            # Log data statistics
            self.metrics_tracker.track_metric(
                'data', 'num_nodes', 
                graph_data.num_nodes if hasattr(graph_data, 'num_nodes') else len(graph_data['nodes'])
            )
            
            return graph_data
    
    def train_model(self, 
                    graph_data: Optional[Dict] = None,
                    resume_from: Optional[str] = None) -> Dict:
        """
        Train the GNN model
        
        Args:
            graph_data: Prepared graph data
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training results
        """
        self.logger.info("Starting model training...")
        
        # Prepare data if not provided
        if graph_data is None:
            graph_data = self.prepare_data()
        
        # Create data loaders
        from torch_geometric.loader import DataLoader
        
        # Split data (simplified - should use proper train/val/test split)
        train_size = int(0.8 * len(graph_data))
        val_size = int(0.1 * len(graph_data))
        
        train_loader = DataLoader([graph_data], batch_size=32, shuffle=True)
        val_loader = DataLoader([graph_data], batch_size=32)
        
        # Create model
        self.logger.info("Creating model architecture...")
        self.model = create_gnn_model(
            model_type=self.config['model']['type'],
            config=self.config['model']
        )
        
        # Create trainer
        if self.config['training'].get('adaptive', False):
            trainer = AdaptiveMultiTaskTrainer(
                model=self.model,
                config=self.config,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        else:
            trainer = MultiTaskTrainer(
                model=self.model,
                config=self.config,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        
        # Resume from checkpoint if specified
        if resume_from:
            self.logger.info(f"Resuming from checkpoint: {resume_from}")
            epoch, val_loss = trainer.load_checkpoint(resume_from)
            self.logger.info(f"Resumed from epoch {epoch} with val_loss {val_loss:.4f}")
        
        # Setup experiment logger
        exp_logger = ExperimentLogger(
            f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        exp_logger.log_config(self.config)
        
        # Train model
        self.logger.info("Training model...")
        with Timer(self.metrics_tracker, 'model_training'):
            history = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader
            )
        
        # Log results
        self.logger.info("Training completed")
        self.logger.info(f"Final validation loss: {history['val_losses'][-1]:.4f}")
        
        # Save training history
        exp_logger.log_summary()
        
        # Update state
        self.state['last_training'] = datetime.now()
        self.state['model_loaded'] = True
        
        # Track metrics
        self.metrics_tracker.track_model_performance(
            task='training',
            metrics={
                'final_loss': history['val_losses'][-1],
                'epochs_trained': len(history['train_losses'])
            }
        )
        
        return history
    
    def run_inference(self, 
                      query: str,
                      graph_data: Optional[Dict] = None) -> Dict:
        """
        Run inference based on natural language query
        
        Args:
            query: Natural language query
            graph_data: Graph data (optional)
            
        Returns:
            Inference results
        """
        self.logger.info(f"Processing query: {query}")
        
        # Validate query
        validator = QueryValidator()
        is_valid, error = validator.validate(query)
        if not is_valid:
            self.logger.error(f"Invalid query: {error}")
            return {'error': error}
        
        # Process query
        if self.query_processor is None:
            self.query_processor = QueryProcessor(self.config)
        
        intent = self.query_processor.process(query)
        self.logger.info(f"Detected task: {intent.task.value} (confidence: {intent.confidence:.2f})")
        
        # Load model if needed
        if not self.state['model_loaded']:
            self.load_model()
        
        # Initialize inference engine
        if self.inference_engine is None:
            self.inference_engine = InferenceEngine(
                model_path=self.config['paths']['model_checkpoints'] + '/best_model.pth',
                config_path='config/config.yaml'
            )
        
        # Prepare data if needed
        if graph_data is None:
            graph_data = self.prepare_data()
        
        # Run inference
        with Timer(self.metrics_tracker, 'inference'):
            results = self.inference_engine.predict(
                graph_data,
                task=intent.task.value
            )
        
        # Run specific task processing
        task_results = self.run_task(
            intent.task.value,
            results,
            intent.parameters,
            graph_data
        )
        
        # Update KG if connected
        if self.state['kg_connected']:
            self.update_kg(task_results, intent.task.value)
        
        # Visualize results
        if self.config.get('visualization', {}).get('auto_visualize', True):
            self.visualize_results(task_results, intent.task.value)
        
        # Update state
        self.state['last_inference'] = datetime.now()
        
        # Track metrics
        self.metrics_tracker.track_event(
            'inference_completed',
            f"Task: {intent.task.value}",
            metadata={'query': query, 'confidence': intent.confidence}
        )
        
        return task_results
    
    def run_task(self, 
                 task_name: str,
                 model_outputs: Dict,
                 parameters: Dict,
                 graph_data = None) -> Dict:
        """
        Run specific task processing
        
        Args:
            task_name: Task to run
            model_outputs: Model outputs
            parameters: Task parameters
            
        Returns:
            Task results
        """
        task_logger = TaskLogger(self.logger, task_name)
        task_logger.log_start()
        
        try:
            if task_name == 'clustering':
                task = EnergyCommunityClustering(self.model, self.config['tasks']['clustering'])
                results = task.run(graph_data if graph_data is not None else model_outputs)
                
            elif task_name == 'solar_optimization':
                task = SolarOptimization(self.model, self.config['tasks']['solar_optimization'])
                results = task.run(graph_data if graph_data is not None else model_outputs)
                
            elif task_name == 'retrofit':
                task = RetrofitTargeting(self.model, self.config['tasks']['retrofit'])
                results = task.run(graph_data if graph_data is not None else model_outputs)
                
            else:
                self.logger.warning(f"Task {task_name} not implemented, returning raw outputs")
                results = model_outputs
            
            task_logger.log_complete(duration=0, success=True)
            
            # Validate results
            validator = ComprehensiveValidator(self.config)
            validation = validator.validate(results, model_outputs)
            results['validation'] = validation
            
            return results
            
        except Exception as e:
            task_logger.log_complete(duration=0, success=False)
            self.logger.error(f"Task {task_name} failed: {e}")
            raise
    
    def update_kg(self, results: Dict, task: str):
        """Update Knowledge Graph with results"""
        try:
            self.logger.info("Updating Knowledge Graph...")
            
            updater = KGUpdater(
                uri=self.config['neo4j']['uri'],
                user=self.config['neo4j']['user'],
                password=self.config['neo4j']['password']
            )
            
            stats = updater.update(results, task)
            
            self.logger.info(f"KG updated: {stats}")
            
            updater.close()
            
        except Exception as e:
            self.logger.error(f"Failed to update KG: {e}")
    
    def visualize_results(self, results: Dict, task: str):
        """Visualize task results"""
        try:
            self.logger.info("Creating visualizations...")
            
            if self.visualizer is None:
                self.visualizer = GraphVisualizer(style='light')
            
            # Create dashboard
            dashboard = self.visualizer.create_dashboard(results, task)
            
            # Save visualization
            output_path = Path(self.config['paths']['visualizations']) / \
                         f"{task}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            
            self.visualizer.export_visualization(dashboard, str(output_path))
            
            self.logger.info(f"Visualization saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Visualization failed: {e}")
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """Load trained model"""
        if checkpoint_path is None:
            checkpoint_path = Path(self.config['paths']['model_checkpoints']) / 'best_model.pth'
        
        self.logger.info(f"Loading model from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create model
        self.model = create_gnn_model(
            model_type=self.config['model']['type'],
            config=checkpoint.get('config', self.config)['model']
        )
        
        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.state['model_loaded'] = True
        self.logger.info("Model loaded successfully")
    
    def run_batch_analysis(self, queries: List[str]) -> List[Dict]:
        """Run batch analysis for multiple queries"""
        results = []
        
        for query in queries:
            self.logger.info(f"Processing query {len(results)+1}/{len(queries)}")
            result = self.run_inference(query)
            results.append(result)
        
        return results
    
    def export_results(self, results: Dict, format: str = 'json', 
                      output_path: Optional[str] = None):
        """Export results to file"""
        if output_path is None:
            output_path = Path(self.config['paths']['results']) / \
                         f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format == 'csv':
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
        elif format == 'excel':
            df = pd.DataFrame(results)
            df.to_excel(output_path, index=False)
        
        self.logger.info(f"Results exported to {output_path}")
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            'state': self.state,
            'metrics': self.metrics_tracker.get_summary(last_n_minutes=60),
            'performance': {
                'cpu_percent': self.metrics_tracker.cpu_percent,
                'memory_percent': self.metrics_tracker.memory_percent,
                'gpu_percent': self.metrics_tracker.gpu_percent
            }
        }
        
        return status
    
    def shutdown(self):
        """Cleanup and shutdown system"""
        self.logger.info("Shutting down Energy GNN System...")
        
        # Flush metrics
        self.metrics_tracker.flush()
        
        # Close KG connection
        if self.kg_connector:
            self.kg_connector.close()
        
        self.logger.info("Shutdown complete")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Energy GNN System - Graph Neural Networks for Building Energy Optimization'
    )
    
    # Main commands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the GNN model')
    train_parser.add_argument('--config', default='config/config.yaml', help='Configuration file')
    train_parser.add_argument('--resume', help='Resume from checkpoint')
    train_parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild graph data')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('query', help='Natural language query')
    infer_parser.add_argument('--config', default='config/config.yaml', help='Configuration file')
    infer_parser.add_argument('--model', help='Model checkpoint path')
    infer_parser.add_argument('--export', help='Export results to file')
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Run batch analysis')
    batch_parser.add_argument('queries_file', help='File with queries (one per line)')
    batch_parser.add_argument('--config', default='config/config.yaml', help='Configuration file')
    batch_parser.add_argument('--output', help='Output file for results')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model performance')
    eval_parser.add_argument('--config', default='config/config.yaml', help='Configuration file')
    eval_parser.add_argument('--model', help='Model checkpoint path')
    eval_parser.add_argument('--test-data', help='Test data path')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Visualize results')
    viz_parser.add_argument('results_file', help='Results file to visualize')
    viz_parser.add_argument('--task', required=True, help='Task type')
    viz_parser.add_argument('--output', help='Output visualization file')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Get system status')
    status_parser.add_argument('--config', default='config/config.yaml', help='Configuration file')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Interactive mode')
    interactive_parser.add_argument('--config', default='config/config.yaml', help='Configuration file')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    # Initialize system
    config_path = getattr(args, 'config', 'config/config.yaml')
    system = EnergyGNNSystem(config_path)
    
    try:
        # Connect to KG if needed
        if args.command in ['train', 'infer', 'batch', 'evaluate']:
            system.connect_kg()
        
        # Execute command
        if args.command == 'train':
            system.train_model(
                resume_from=args.resume,
                graph_data=system.prepare_data(force_rebuild=args.force_rebuild)
            )
            
        elif args.command == 'infer':
            if args.model:
                system.load_model(args.model)
            
            results = system.run_inference(args.query)
            
            print("\n" + "="*50)
            print("RESULTS:")
            print("="*50)
            print(json.dumps(results, indent=2, default=str))
            
            if args.export:
                system.export_results(results, output_path=args.export)
        
        elif args.command == 'batch':
            with open(args.queries_file, 'r') as f:
                queries = [line.strip() for line in f if line.strip()]
            
            results = system.run_batch_analysis(queries)
            
            if args.output:
                system.export_results(results, output_path=args.output)
            else:
                print(json.dumps(results, indent=2, default=str))
        
        elif args.command == 'evaluate':
            # Load model
            if args.model:
                system.load_model(args.model)
            
            # Prepare test data
            if args.test_data:
                test_data = torch.load(args.test_data)
            else:
                test_data = system.prepare_data()
            
            # Evaluate
            metrics = EvaluationMetrics(system.config['tasks'])
            
            # Run inference
            outputs = system.inference_engine.predict(test_data)
            metrics.update(outputs, test_data)
            
            # Compute metrics
            results = metrics.compute()
            
            print("\n" + "="*50)
            print("EVALUATION RESULTS:")
            print("="*50)
            for task, task_metrics in results.items():
                print(f"\n{task.upper()}:")
                for metric, value in task_metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
        
        elif args.command == 'visualize':
            # Load results
            with open(args.results_file, 'r') as f:
                results = json.load(f)
            
            # Create visualization
            visualizer = GraphVisualizer()
            dashboard = visualizer.create_dashboard(results, args.task)
            
            # Save
            output = args.output or f"{args.task}_visualization.html"
            visualizer.export_visualization(dashboard, output)
            
            print(f"Visualization saved to {output}")
        
        elif args.command == 'status':
            status = system.get_system_status()
            
            print("\n" + "="*50)
            print("SYSTEM STATUS:")
            print("="*50)
            print(json.dumps(status, indent=2, default=str))
        
        elif args.command == 'interactive':
            print("\n" + "="*50)
            print("ENERGY GNN INTERACTIVE MODE")
            print("="*50)
            print("Type 'help' for commands, 'exit' to quit\n")
            
            while True:
                try:
                    query = input(">>> ").strip()
                    
                    if query.lower() == 'exit':
                        break
                    elif query.lower() == 'help':
                        print("""
Available commands:
  - Natural language queries (e.g., "What buildings need solar panels?")
  - status: Show system status
  - train: Start model training
  - export <filename>: Export last results
  - clear: Clear screen
  - exit: Quit
                        """)
                    elif query.lower() == 'status':
                        status = system.get_system_status()
                        print(json.dumps(status, indent=2, default=str))
                    elif query.lower() == 'train':
                        system.train_model()
                    elif query.lower().startswith('export'):
                        parts = query.split()
                        if len(parts) > 1:
                            system.export_results(
                                system.state.get('last_results', {}),
                                output_path=parts[1]
                            )
                    elif query.lower() == 'clear':
                        os.system('cls' if os.name == 'nt' else 'clear')
                    else:
                        # Process as natural language query
                        results = system.run_inference(query)
                        system.state['last_results'] = results
                        print(json.dumps(results, indent=2, default=str))
                        
                except KeyboardInterrupt:
                    print("\nUse 'exit' to quit")
                except Exception as e:
                    print(f"Error: {e}")
    
    except Exception as e:
        system.logger.error(f"System error: {e}", exc_info=True)
        sys.exit(1)
    
    finally:
        system.shutdown()

if __name__ == "__main__":
    main()