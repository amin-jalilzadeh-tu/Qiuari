"""
Main script to run and compare all KG-aware clustering methods
"""

import sys
import os
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Import KG connector and preprocessor
from data.kg_connector import KGConnector
from data_preprocessor import KGDataPreprocessor
from evaluation_metrics import ClusteringEvaluator

# Import clustering methods
from method_1_hierarchical_louvain import HierarchicalLouvainComplementarity
from method_2_stable_matching import StableMatchingEnergySharing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ClusteringComparison:
    """
    Compare multiple KG-aware clustering methods on energy complementarity.
    """
    
    def __init__(self, kg_connector: KGConnector):
        """
        Initialize comparison framework.
        
        Args:
            kg_connector: Connected KGConnector instance
        """
        self.kg = kg_connector
        self.preprocessor = KGDataPreprocessor(kg_connector)
        self.methods = {}
        self.results = {}
        
        logger.info("Initialized clustering comparison framework")
    
    def add_method(self, name: str, method_instance):
        """
        Add a clustering method to compare.
        
        Args:
            name: Method identifier
            method_instance: Instance of clustering method
        """
        self.methods[name] = method_instance
        logger.info(f"Added method: {name}")
    
    def run_comparison(self, district_name: str, 
                       lookback_hours: int = 168,
                       save_results: bool = True) -> pd.DataFrame:
        """
        Run all clustering methods and compare results.
        
        Args:
            district_name: District to analyze
            lookback_hours: Hours of time series data
            save_results: Whether to save results to file
            
        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Starting comparison for district: {district_name}")
        
        # Preprocess data
        logger.info("Preprocessing data from KG...")
        preprocessed_data = self.preprocessor.prepare_data_from_kg(
            district_name, 
            lookback_hours
        )
        
        # Initialize evaluator
        evaluator = ClusteringEvaluator(preprocessed_data)
        
        # Run each method
        for method_name, method in self.methods.items():
            logger.info(f"\n{'='*50}")
            logger.info(f"Running {method_name}")
            logger.info(f"{'='*50}")
            
            try:
                # Run clustering
                method.fit(preprocessed_data)
                clusters = method.get_clusters()
                
                # Evaluate results
                metrics = evaluator.evaluate_clustering(clusters, method_name)
                
                # Add method-specific metrics
                if hasattr(method, 'get_method_specific_metrics'):
                    metrics['method_specific'] = method.get_method_specific_metrics()
                
                # Store results
                self.results[method_name] = {
                    'clusters': clusters,
                    'metrics': metrics,
                    'execution_info': method.get_execution_info()
                }
                
                logger.info(f"{method_name} complete - Score: {metrics['summary']['overall_score']:.3f}")
                
            except Exception as e:
                logger.error(f"Error running {method_name}: {str(e)}")
                self.results[method_name] = {
                    'error': str(e),
                    'clusters': {},
                    'metrics': {}
                }
        
        # Create comparison table
        comparison_df = self._create_comparison_table()
        
        # Save results if requested
        if save_results:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._save_results(f"comparison_results_{timestamp}")
        
        return comparison_df
    
    def _create_comparison_table(self) -> pd.DataFrame:
        """
        Create comparison table of all methods.
        """
        rows = []
        
        for method_name, result in self.results.items():
            if 'error' in result:
                continue
            
            metrics = result['metrics']
            exec_info = result['execution_info']
            
            row = {
                'Method': method_name,
                'Clusters': metrics.get('cluster_count', 0),
                'Buildings': metrics.get('clustered_buildings', 0),
                'Peak Reduction': metrics.get('peak_reduction', 0),
                'Self-Sufficiency': metrics.get('self_sufficiency', 0),
                'Complementarity': metrics.get('complementarity_score', 0),
                'Violations': metrics.get('constraint_violations', 0),
                'Network Losses': metrics.get('network_losses', 0),
                'Diversity': metrics.get('diversity_index', 0),
                'Fairness': metrics.get('fairness_index', 0),
                'Coverage': metrics.get('coverage', 0),
                'Overall Score': metrics.get('summary', {}).get('overall_score', 0),
                'Execution Time': exec_info.get('execution_time', 0)
            }
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Sort by overall score
        df = df.sort_values('Overall Score', ascending=False)
        
        # Format percentages
        pct_cols = ['Peak Reduction', 'Self-Sufficiency', 'Complementarity', 
                   'Coverage', 'Diversity', 'Fairness']
        for col in pct_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: f"{x:.1%}")
        
        # Format other numbers
        df['Network Losses'] = df['Network Losses'].apply(lambda x: f"{x:.3f}")
        df['Overall Score'] = df['Overall Score'].apply(lambda x: f"{x:.3f}")
        df['Execution Time'] = df['Execution Time'].apply(lambda x: f"{x:.2f}s")
        
        return df
    
    def _save_results(self, prefix: str):
        """
        Save detailed results to files.
        """
        output_dir = "comparison/kg_clustering/results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save comparison table
        comparison_df = self._create_comparison_table()
        comparison_df.to_csv(f"{output_dir}/{prefix}_comparison.csv", index=False)
        
        # Save detailed results
        with open(f"{output_dir}/{prefix}_detailed.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = {}
            for method_name, result in self.results.items():
                serializable_results[method_name] = {
                    'metrics': result['metrics'],
                    'execution_info': result['execution_info'],
                    'n_clusters': len(result.get('clusters', {}))
                }
            
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {output_dir}/{prefix}_*")
    
    def visualize_comparison(self):
        """
        Create visualization of comparison results.
        """
        if not self.results:
            logger.warning("No results to visualize")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('KG-Aware Clustering Methods Comparison', fontsize=16)
        
        # Prepare data
        methods = []
        peak_reductions = []
        self_sufficiencies = []
        complementarities = []
        violations = []
        coverages = []
        overall_scores = []
        
        for method_name, result in self.results.items():
            if 'error' not in result:
                metrics = result['metrics']
                methods.append(method_name.replace(' ', '\n'))
                peak_reductions.append(metrics.get('peak_reduction', 0))
                self_sufficiencies.append(metrics.get('self_sufficiency', 0))
                complementarities.append(metrics.get('complementarity_score', 0))
                violations.append(metrics.get('constraint_violations', 0))
                coverages.append(metrics.get('coverage', 0))
                overall_scores.append(metrics.get('summary', {}).get('overall_score', 0))
        
        # Plot 1: Peak Reduction
        axes[0, 0].bar(methods, peak_reductions, color='skyblue')
        axes[0, 0].set_title('Peak Reduction')
        axes[0, 0].set_ylabel('Reduction (%)')
        axes[0, 0].set_ylim(0, max(peak_reductions) * 1.2 if peak_reductions else 1)
        
        # Plot 2: Self-Sufficiency
        axes[0, 1].bar(methods, self_sufficiencies, color='lightgreen')
        axes[0, 1].set_title('Self-Sufficiency')
        axes[0, 1].set_ylabel('Ratio')
        axes[0, 1].set_ylim(0, max(self_sufficiencies) * 1.2 if self_sufficiencies else 1)
        
        # Plot 3: Complementarity Score
        axes[0, 2].bar(methods, complementarities, color='coral')
        axes[0, 2].set_title('Complementarity Score')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_ylim(0, max(complementarities) * 1.2 if complementarities else 1)
        
        # Plot 4: Constraint Violations
        axes[1, 0].bar(methods, violations, color='salmon')
        axes[1, 0].set_title('Constraint Violations')
        axes[1, 0].set_ylabel('Count')
        
        # Plot 5: Coverage
        axes[1, 1].bar(methods, coverages, color='gold')
        axes[1, 1].set_title('Coverage')
        axes[1, 1].set_ylabel('Fraction')
        axes[1, 1].set_ylim(0, 1.1)
        
        # Plot 6: Overall Score
        axes[1, 2].bar(methods, overall_scores, color='mediumpurple')
        axes[1, 2].set_title('Overall Score')
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_ylim(0, 1.1)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_dir = "comparison/kg_clustering/results"
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{output_dir}/comparison_viz_{timestamp}.png", dpi=300, bbox_inches='tight')
        
        plt.show()
        
        logger.info("Visualization saved")

def main():
    """
    Main execution function.
    """
    # Configuration
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "aminasad"
    
    # District to analyze
    DISTRICT_NAME = "Buitenveldert-Oost"  # Update with your district
    
    try:
        # Connect to Neo4j
        logger.info("Connecting to Neo4j...")
        kg = KGConnector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        
        if not kg.verify_connection():
            raise ConnectionError("Failed to connect to Neo4j")
        
        # Get KG statistics
        stats = kg.get_statistics()
        logger.info(f"KG contains {stats['nodes']['buildings']} buildings, "
                   f"{stats['nodes']['cable_groups']} cable groups")
        
        # Initialize comparison framework
        comparison = ClusteringComparison(kg)
        
        # Add all clustering methods
        
        # Method 1: Hierarchical Louvain
        comparison.add_method(
            "Hierarchical Louvain",
            HierarchicalLouvainComplementarity(resolution=1.0, hierarchy_penalty=10.0)
        )
        
        # Method 2: Stable Matching
        comparison.add_method(
            "Stable Matching",
            StableMatchingEnergySharing(alpha=0.5, beta=0.3, gamma=0.2, max_partners=5)
        )
        
        # Method 3: Information Synergy
        from method_3_information_synergy import InformationSynergyClusteringKG
        comparison.add_method(
            "Information Synergy",
            InformationSynergyClusteringKG(n_bins=10, max_cluster_size=8, synergy_threshold=0.2)
        )
        
        # Method 4: Spectral Clustering
        from method_4_spectral_clustering import NetworkAwareSpectralClustering
        comparison.add_method(
            "Spectral Clustering",
            NetworkAwareSpectralClustering(n_clusters=15, lambda_decay=0.5)
        )
        
        # Method 5: Deep RL (optional - requires GPU)
        try:
            from method_5_deep_rl import RLClusteringWithKG
            comparison.add_method(
                "Deep RL",
                RLClusteringWithKG(n_episodes=50, max_steps=100, epsilon=0.1)
            )
        except ImportError:
            logger.warning("Deep RL method not available (missing torch dependencies)")
        
        # Method 6: Correlation Clustering
        from method_6_correlation_clustering import CorrelationClusteringKG
        comparison.add_method(
            "Correlation Clustering",
            CorrelationClusteringKG(max_iterations=100, use_sdp_relaxation=False)
        )
        
        # Method 7: Node2Vec
        from method_7_node2vec import ElectricalNode2Vec
        comparison.add_method(
            "Node2Vec Electrical",
            ElectricalNode2Vec(embedding_dim=64, walk_length=30, num_walks=20, n_clusters=15)
        )
        
        # Run comparison
        logger.info("\n" + "="*60)
        logger.info("STARTING CLUSTERING COMPARISON")
        logger.info("="*60)
        
        results_df = comparison.run_comparison(
            DISTRICT_NAME,
            lookback_hours=168,  # 1 week
            save_results=True
        )
        
        # Display results
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(results_df.to_string())
        
        # Create visualization
        comparison.visualize_comparison()
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if not results_df.empty:
            best_method = results_df.iloc[0]['Method']
            best_score = results_df.iloc[0]['Overall Score']
            print(f"Best Method: {best_method}")
            print(f"Overall Score: {best_score}")
            
            # Print strengths of best method
            best_result = comparison.results.get(best_method, {})
            if best_result and 'metrics' in best_result:
                strengths = best_result['metrics'].get('summary', {}).get('strengths', [])
                if strengths:
                    print(f"Strengths: {', '.join(strengths)}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise
    
    finally:
        # Close connection
        if 'kg' in locals():
            kg.close()
            logger.info("Neo4j connection closed")

if __name__ == "__main__":
    main()