"""
Simplified Main Pipeline for Solar Energy Optimization
Iterative semi-supervised learning with real deployment feedback
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Import simplified components
from models.solar_district_gnn import SolarDistrictGNN
from training.unified_solar_trainer import UnifiedSolarTrainer
from tasks.solar_labeling import SolarPerformanceLabeler, SolarInstallation
from tasks.solar_optimization import SolarOptimization
from data.data_loader import EnergyDataLoader
from analysis.comprehensive_reporter import ComprehensiveReporter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SolarOptimizationPipeline:
    """
    Main pipeline for iterative solar optimization
    Combines discovery, recommendation, deployment, and learning
    """
    
    def __init__(self, config_path: str):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                self.config = yaml.safe_load(f)
            else:
                self.config = json.load(f)
        
        # Initialize components
        self.model = SolarDistrictGNN(self.config['model'])
        self.trainer = UnifiedSolarTrainer(self.model, self.config['training'])
        self.labeler = SolarPerformanceLabeler(self.config.get('labeling', {}))
        self.solar_optimizer = SolarOptimization(self.model, self.config.get('solar', {}))
        
        # Data management
        self.data_loader = EnergyDataLoader(self.config['data'])
        self.cluster_labels = {}
        self.solar_labels = {}
        
        # Tracking
        self.current_round = 0
        self.deployment_history = []
        self.performance_metrics = []
        
        # Output directory
        self.output_dir = Path(self.config.get('output_dir', 'outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized Solar Optimization Pipeline")
    
    def run_full_pipeline(self, num_rounds: int = 5):
        """
        Run complete iterative pipeline
        
        Args:
            num_rounds: Number of deployment rounds
        """
        logger.info(f"Starting {num_rounds}-round optimization pipeline")
        
        for round_idx in range(num_rounds):
            self.current_round = round_idx + 1
            logger.info(f"\n{'='*50}")
            logger.info(f"ROUND {self.current_round}/{num_rounds}")
            logger.info(f"{'='*50}")
            
            # Run one iteration
            round_results = self.run_iteration()
            self.performance_metrics.append(round_results)
            
            # Save intermediate results
            self.save_round_results(round_results)
            
            # Report progress
            self.report_progress(round_results)
        
        # Final report
        self.generate_final_report()
    
    def run_iteration(self) -> Dict:
        """
        Run one complete iteration of the pipeline
        
        Returns:
            Results dictionary for this round
        """
        results = {
            'round': self.current_round,
            'timestamp': datetime.now().isoformat()
        }
        
        # Phase 1: Discovery (find self-sufficient clusters)
        logger.info("\n--- Phase 1: Discovery ---")
        discovery_results = self.run_discovery_phase()
        results['discovery'] = discovery_results
        
        # Phase 2: Solar Recommendation
        logger.info("\n--- Phase 2: Solar Recommendation ---")
        recommendations = self.run_solar_phase(discovery_results['clusters'])
        results['recommendations'] = recommendations
        
        # Phase 3: Deployment (simulated or real)
        logger.info("\n--- Phase 3: Deployment ---")
        deployment_results = self.deploy_solar(recommendations['top_buildings'])
        results['deployment'] = deployment_results
        
        # Phase 4: Measurement & Labeling
        logger.info("\n--- Phase 4: Measurement & Labeling ---")
        new_labels = self.measure_and_label(deployment_results['installations'])
        results['new_labels'] = new_labels
        
        # Phase 5: Model Update
        logger.info("\n--- Phase 5: Model Update ---")
        update_results = self.update_model(new_labels)
        results['model_update'] = update_results
        
        return results
    
    def run_discovery_phase(self) -> Dict:
        """
        Discover self-sufficient energy communities
        """
        # Load data
        train_loader, val_loader = self.data_loader.get_dataloaders()
        
        # Train discovery phase
        use_labels = len(self.cluster_labels) > 0
        epochs = self.config['training'].get('discovery_epochs', 30)
        
        if use_labels:
            logger.info(f"Training with {len(self.cluster_labels)} cluster labels")
            self.trainer.cluster_labels = self.cluster_labels
        
        discovery_metrics = self.trainer.train_discovery_phase(
            train_loader,
            val_loader,
            epochs=epochs,
            use_labels=use_labels
        )
        
        # Get cluster assignments
        self.model.eval()
        all_clusters = []
        
        with torch.no_grad():
            for batch in train_loader:
                batch = batch.to(self.trainer.device)
                outputs = self.model(batch, phase='discovery')
                clusters = outputs['cluster_assignments'].argmax(dim=-1)
                all_clusters.append(clusters)
        
        all_clusters = torch.cat(all_clusters)
        
        # Generate cluster quality labels
        cluster_metrics = self._evaluate_clusters(all_clusters, train_loader)
        new_cluster_labels = self.labeler.generate_cluster_labels(cluster_metrics)
        self.cluster_labels.update(new_cluster_labels)
        
        return {
            'clusters': all_clusters,
            'num_clusters': len(torch.unique(all_clusters)),
            'metrics': discovery_metrics,
            'cluster_quality': cluster_metrics
        }
    
    def run_solar_phase(self, clusters: torch.Tensor) -> Dict:
        """
        Recommend solar installations within discovered clusters
        """
        # Load data
        train_loader, val_loader = self.data_loader.get_dataloaders()
        
        # Train solar phase
        epochs = self.config['training'].get('solar_epochs', 20)
        
        if self.solar_labels:
            logger.info(f"Training with {len(self.solar_labels)} solar labels")
            self.trainer.solar_labels = self.solar_labels
        
        solar_metrics = self.trainer.train_solar_phase(
            train_loader,
            val_loader,
            epochs=epochs,
            cluster_assignments={'clusters': clusters}
        )
        
        # Get recommendations
        top_k = self.config.get('deployment', {}).get('buildings_per_round', 10)
        recommendations = self.trainer.recommend_solar(
            train_loader,
            top_k=top_k,
            confidence_threshold=0.6
        )
        
        return {
            'top_buildings': recommendations,
            'num_recommendations': len(recommendations),
            'metrics': solar_metrics
        }
    
    def deploy_solar(self, recommendations: List[Tuple[int, float, str]]) -> Dict:
        """
        Deploy solar (simulated or trigger real deployment)
        """
        installations = []
        
        for building_id, score, roi_category in recommendations:
            # In real system, this would trigger actual installation
            # Here we simulate
            installation = SolarInstallation(
                building_id=building_id,
                cluster_id=0,  # Would get from cluster assignments
                installation_date=datetime.now(),
                capacity_kw=np.random.uniform(5, 20),  # Simulated
                installation_cost=np.random.uniform(5000, 20000),  # Simulated
                daily_generation_kwh=[],  # Will be filled by measurement
                self_consumption_rate=0.0,
                export_revenue=0.0,
                peak_reduction_percent=0.0
            )
            
            installations.append(installation)
            self.labeler.add_installation(installation)
            
            logger.info(f"Deployed {installation.capacity_kw:.1f}kW solar at building {building_id}")
        
        self.deployment_history.extend(installations)
        
        return {
            'installations': installations,
            'total_capacity_kw': sum(i.capacity_kw for i in installations),
            'total_cost': sum(i.installation_cost for i in installations)
        }
    
    def measure_and_label(self, installations: List[SolarInstallation]) -> Dict:
        """
        Measure performance and generate labels
        In real system, this would wait for actual data
        """
        new_solar_labels = {}
        
        for installation in installations:
            # Simulate performance data
            # In reality, would wait and collect real data
            daily_generation = np.random.uniform(
                installation.capacity_kw * 3,
                installation.capacity_kw * 5,
                size=90  # 90 days of data
            ).tolist()
            
            self_consumption = np.random.uniform(0.3, 0.8)
            peak_reduction = np.random.uniform(0.1, 0.4)
            
            # Update installation with "measured" data
            self.labeler.update_performance_data(
                installation.building_id,
                daily_generation,
                self_consumption,
                peak_reduction
            )
            
            # Generate label
            label, confidence = self.labeler.label_installation(
                installation.building_id,
                force=True  # Force labeling for demo
            )
            
            new_solar_labels[installation.building_id] = (label, confidence)
            self.solar_labels[installation.building_id] = label
        
        # Get statistics
        stats = self.labeler.get_statistics()
        
        return {
            'new_labels': new_solar_labels,
            'total_solar_labels': len(self.solar_labels),
            'label_statistics': stats['label_distribution']
        }
    
    def update_model(self, new_labels: Dict) -> Dict:
        """
        Update model with new labels
        """
        # Add new labels to trainer
        self.trainer.solar_labels = self.solar_labels
        self.trainer.cluster_labels = self.cluster_labels
        
        # Quick fine-tuning with new labels
        train_loader, val_loader = self.data_loader.get_dataloaders()
        
        # Just a few epochs of fine-tuning
        finetune_epochs = 5
        
        logger.info(f"Fine-tuning model with {len(new_labels['new_labels'])} new labels")
        
        # Train both phases with new labels
        discovery_metrics = self.trainer.train_discovery_phase(
            train_loader,
            val_loader,
            epochs=finetune_epochs,
            use_labels=True
        )
        
        solar_metrics = self.trainer.train_solar_phase(
            train_loader,
            val_loader,
            epochs=finetune_epochs
        )
        
        return {
            'discovery_improvement': discovery_metrics,
            'solar_improvement': solar_metrics,
            'total_labels_used': len(self.solar_labels) + len(self.cluster_labels)
        }
    
    def _evaluate_clusters(self, clusters: torch.Tensor, data_loader: DataLoader) -> Dict:
        """
        Evaluate cluster quality metrics
        """
        cluster_metrics = {}
        unique_clusters = torch.unique(clusters)
        
        for cluster_id in unique_clusters:
            cluster_mask = clusters == cluster_id
            cluster_size = cluster_mask.sum().item()
            
            # Calculate metrics (simplified)
            # In reality, would calculate from actual consumption data
            metrics = {
                'self_sufficiency': np.random.uniform(0.3, 0.8),
                'complementarity': np.random.uniform(-0.5, 0.2),
                'peak_reduction': np.random.uniform(0.1, 0.4),
                'network_losses': np.random.uniform(0.05, 0.15)
            }
            
            cluster_metrics[cluster_id.item()] = metrics
        
        return cluster_metrics
    
    def save_round_results(self, results: Dict):
        """Save results for current round"""
        filepath = self.output_dir / f"round_{self.current_round}_results.json"
        with open(filepath, 'w') as f:
            # Convert tensors to lists for JSON serialization
            def convert_tensors(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_tensors(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_tensors(v) for v in obj]
                return obj
            
            json.dump(convert_tensors(results), f, indent=2, default=str)
        
        logger.info(f"Saved round results to {filepath}")
    
    def report_progress(self, round_results: Dict):
        """Report progress for current round"""
        print(f"\n{'='*50}")
        print(f"Round {self.current_round} Summary")
        print(f"{'='*50}")
        print(f"Clusters found: {round_results['discovery']['num_clusters']}")
        print(f"Solar deployed: {len(round_results['deployment']['installations'])}")
        print(f"Total capacity: {round_results['deployment']['total_capacity_kw']:.1f} kW")
        print(f"New labels generated: {len(round_results['new_labels']['new_labels'])}")
        print(f"Label distribution: {round_results['new_labels']['label_statistics']}")
        print(f"{'='*50}")
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        report = {
            'total_rounds': self.current_round,
            'total_installations': len(self.deployment_history),
            'total_capacity_kw': sum(i.capacity_kw for i in self.deployment_history),
            'total_investment': sum(i.installation_cost for i in self.deployment_history),
            'label_statistics': self.labeler.get_statistics(),
            'performance_by_round': self.performance_metrics
        }
        
        # Save report
        report_path = self.output_dir / "final_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Print summary
        print(f"\n{'='*60}")
        print("FINAL REPORT")
        print(f"{'='*60}")
        print(f"Total rounds completed: {report['total_rounds']}")
        print(f"Total solar installations: {report['total_installations']}")
        print(f"Total capacity deployed: {report['total_capacity_kw']:.1f} kW")
        print(f"Total investment: €{report['total_investment']:.2f}")
        print(f"\nLabel Statistics:")
        print(f"  Excellent: {report['label_statistics']['label_distribution']['excellent']}")
        print(f"  Good: {report['label_statistics']['label_distribution']['good']}")
        print(f"  Fair: {report['label_statistics']['label_distribution']['fair']}")
        print(f"  Poor: {report['label_statistics']['label_distribution']['poor']}")
        
        if report['label_statistics']['average_roi']:
            print(f"\nAverage ROI: {report['label_statistics']['average_roi']:.1f} years")
        
        print(f"\nFinal report saved to: {report_path}")
        print(f"{'='*60}")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Solar Energy Optimization Pipeline")
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--rounds', type=int, default=5,
                       help='Number of deployment rounds')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'discovery', 'solar', 'test'],
                       help='Pipeline mode')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SolarOptimizationPipeline(args.config)
    
    # Run based on mode
    if args.mode == 'full':
        pipeline.run_full_pipeline(num_rounds=args.rounds)
    elif args.mode == 'discovery':
        results = pipeline.run_discovery_phase()
        print(f"Discovery completed: {results['num_clusters']} clusters found")
    elif args.mode == 'solar':
        # Need to run discovery first
        discovery = pipeline.run_discovery_phase()
        results = pipeline.run_solar_phase(discovery['clusters'])
        print(f"Solar recommendations: {results['num_recommendations']} buildings selected")
    elif args.mode == 'test':
        # Quick test run
        results = pipeline.run_iteration()
        print(f"Test iteration completed successfully")


if __name__ == "__main__":
    main()