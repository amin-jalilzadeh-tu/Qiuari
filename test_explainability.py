"""
Comprehensive test script for GNN explainability capabilities.
Tests attention visualization, feature importance, gradient-based explanations,
and subgraph extraction for clustering decisions.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from torch_geometric.data import Data
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.explainability_layers import (
    ExplainableGATConv, EnhancedGNNExplainer,
    AttentionVisualizer, FeatureImportanceAnalyzer
)
from models.base_gnn import EnergyGNN
from data.data_loader import UnifiedDataLoader
from config.config import load_config
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directory for visualizations
OUTPUT_DIR = Path("explainability_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


class ExplainabilityTester:
    """Comprehensive tester for GNN explainability features."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the tester with configuration."""
        self.config = load_config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.data_loader = None
        self.test_data = None
        self.explainer = None
        self.attention_viz = None
        self.feature_analyzer = None
        
    def setup(self):
        """Setup model and data."""
        logger.info("Setting up model and data...")
        
        # Load data
        try:
            self.data_loader = UnifiedDataLoader(self.config)
            train_loader, val_loader, test_loader = self.data_loader.get_loaders()
            
            # Get a batch of test data
            for batch in test_loader:
                self.test_data = batch
                break
                
            if self.test_data is None:
                # Create synthetic data for testing
                self.test_data = self.create_synthetic_data()
                
            self.test_data = self.test_data.to(self.device)
            logger.info(f"Loaded test data with {self.test_data.num_nodes} nodes")
            
        except Exception as e:
            logger.warning(f"Could not load real data: {e}. Creating synthetic data...")
            self.test_data = self.create_synthetic_data().to(self.device)
        
        # Load or create model
        try:
            # Try to load existing model
            checkpoint_path = Path("checkpoints/best_model.pt")
            if checkpoint_path.exists():
                logger.info(f"Loading model from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Initialize model
                self.model = EnergyGNN(self.config)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model = self.model.to(self.device)
                self.model.eval()
                logger.info("Model loaded successfully")
            else:
                logger.info("Creating new model for testing...")
                self.model = self.create_test_model()
        except Exception as e:
            logger.warning(f"Could not load model: {e}. Creating test model...")
            self.model = self.create_test_model()
        
        # Initialize explainability components
        self.explainer = EnhancedGNNExplainer(self.model, num_hops=3)
        self.attention_viz = AttentionVisualizer(save_dir=str(OUTPUT_DIR / "attention"))
        self.feature_analyzer = FeatureImportanceAnalyzer(self.model)
        
        # Register attention hooks
        self.attention_viz.register_attention_hook(self.model)
        
        logger.info("Setup complete!")
        
    def create_synthetic_data(self) -> Data:
        """Create synthetic test data."""
        num_nodes = 100
        num_edges = 300
        num_features = 17
        
        # Create node features
        x = torch.randn(num_nodes, num_features)
        
        # Create edges (random graph)
        edge_index = torch.randint(0, num_nodes, (2, num_edges))
        
        # Create labels for clustering
        y = torch.randint(0, 5, (num_nodes,))
        
        return Data(x=x, edge_index=edge_index, y=y)
    
    def create_test_model(self) -> nn.Module:
        """Create a simple test model."""
        class SimpleGNN(nn.Module):
            def __init__(self, input_dim=17, hidden_dim=64, num_classes=5):
                super().__init__()
                self.config = {'hidden_dim': hidden_dim}
                
                # Use explainable GAT layer
                self.conv1 = ExplainableGATConv(input_dim, hidden_dim, heads=4)
                self.conv2 = ExplainableGATConv(hidden_dim * 4, hidden_dim, heads=4)
                self.conv3 = ExplainableGATConv(hidden_dim * 4, hidden_dim, heads=1)
                
                self.classifier = nn.Linear(hidden_dim, num_classes)
                
            def forward(self, data):
                x, edge_index = data.x, data.edge_index
                
                # First layer with attention
                x, att1 = self.conv1(x, edge_index)
                x = torch.relu(x)
                
                # Second layer
                x, att2 = self.conv2(x, edge_index)
                x = torch.relu(x)
                
                # Third layer
                x, att3 = self.conv3(x, edge_index)
                
                # Classification
                out = self.classifier(x)
                
                return {
                    'predictions': out,
                    'clustering_cluster_assignments': torch.softmax(out, dim=-1),
                    'attention_weights': [att1, att2, att3]
                }
        
        model = SimpleGNN().to(self.device)
        return model
    
    def test_attention_visualization(self):
        """Test attention weight extraction and visualization."""
        logger.info("\n" + "="*50)
        logger.info("Testing Attention Visualization")
        logger.info("="*50)
        
        try:
            # Forward pass to generate attention weights
            with torch.no_grad():
                output = self.model(self.test_data)
            
            # Check if attention weights were captured
            if self.attention_viz.layer_attentions:
                logger.info(f"Captured attention from {len(self.attention_viz.layer_attentions)} layers")
                
                # Visualize attention for a sample node
                sample_node = 0
                self.attention_viz.visualize_node_attention(sample_node)
                logger.info(f"Created attention visualization for node {sample_node}")
                
                # Visualize overall attention statistics
                self.attention_viz.visualize_layer_attention_stats()
                logger.info("Created attention statistics visualization")
                
                # Analyze attention patterns
                attention_analysis = self.analyze_attention_patterns()
                logger.info("Attention Pattern Analysis:")
                for key, value in attention_analysis.items():
                    logger.info(f"  {key}: {value:.4f}")
                    
                return True, attention_analysis
            else:
                logger.warning("No attention weights captured. Model may not have attention layers.")
                return False, {}
                
        except Exception as e:
            logger.error(f"Attention visualization failed: {e}")
            return False, {}
    
    def analyze_attention_patterns(self) -> dict:
        """Analyze captured attention patterns."""
        analysis = {}
        
        for layer_name, attention in self.attention_viz.layer_attentions.items():
            if attention is not None:
                att_numpy = attention.numpy()
                analysis[f"{layer_name}_mean"] = float(np.mean(att_numpy))
                analysis[f"{layer_name}_std"] = float(np.std(att_numpy))
                analysis[f"{layer_name}_max"] = float(np.max(att_numpy))
                analysis[f"{layer_name}_sparsity"] = float(np.mean(att_numpy < 0.1))
                
        return analysis
    
    def test_feature_importance(self):
        """Test feature importance extraction methods."""
        logger.info("\n" + "="*50)
        logger.info("Testing Feature Importance Analysis")
        logger.info("="*50)
        
        try:
            # Select a sample node for explanation
            sample_node = 5
            
            # Test gradient-based importance
            logger.info("Testing gradient-based feature importance...")
            grad_importance = self.feature_analyzer.gradient_importance(
                self.test_data, sample_node
            )
            
            # Test perturbation-based importance
            logger.info("Testing perturbation-based feature importance...")
            perturb_importance = self.feature_analyzer.perturbation_importance(
                self.test_data, sample_node, n_samples=50
            )
            
            # Test integrated gradients
            logger.info("Testing integrated gradients...")
            ig_importance = self.feature_analyzer.integrated_gradients(
                self.test_data, sample_node, n_steps=30
            )
            
            # Get comprehensive importance
            logger.info("Computing comprehensive feature importance...")
            comprehensive = self.feature_analyzer.comprehensive_importance(
                self.test_data, sample_node
            )
            
            # Visualize feature importance
            self.visualize_feature_importance(comprehensive, sample_node)
            
            # Log top features
            logger.info(f"\nTop 5 Important Features for Node {sample_node}:")
            for i, (name, score) in enumerate(comprehensive['feature_ranking'][:5], 1):
                logger.info(f"  {i}. {name}: {score:.4f}")
            
            return True, comprehensive
            
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
            return False, {}
    
    def visualize_feature_importance(self, importance_data: dict, node_idx: int):
        """Create visualization for feature importance."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        feature_names = self.feature_analyzer.feature_names
        
        # Plot different importance methods
        methods = ['gradient', 'perturbation', 'integrated_gradients', 'learned']
        titles = ['Gradient-based', 'Perturbation-based', 'Integrated Gradients', 'Learned Aggregation']
        
        for idx, (method, title) in enumerate(zip(methods, titles)):
            ax = axes[idx // 2, idx % 2]
            
            if method in importance_data:
                scores = importance_data[method].cpu().numpy()
                
                # Create bar plot
                y_pos = np.arange(len(feature_names))
                ax.barh(y_pos, scores)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(feature_names, fontsize=8)
                ax.set_xlabel('Importance Score')
                ax.set_title(f'{title} Feature Importance')
                ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Feature Importance Analysis for Node {node_idx}', fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f'feature_importance_node_{node_idx}.png', dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature importance visualization to {OUTPUT_DIR}")
    
    def test_gnn_explainer(self):
        """Test GNNExplainer for subgraph and edge importance."""
        logger.info("\n" + "="*50)
        logger.info("Testing GNNExplainer for Subgraph Extraction")
        logger.info("="*50)
        
        try:
            # Select a node to explain
            sample_node = 10
            
            logger.info(f"Explaining prediction for node {sample_node}...")
            explanation = self.explainer.explain_node(
                self.test_data, 
                sample_node
            )
            
            # Log explanation details
            logger.info(f"Explanation for Node {sample_node}:")
            logger.info(f"  Predicted class: {explanation['target_class']}")
            logger.info(f"  Subgraph size: {len(explanation['subgraph_nodes'])} nodes")
            logger.info(f"  Important edges: {len(explanation['top_edges']['indices'])}")
            
            # Top features
            logger.info("  Top 5 important features:")
            for i, (idx, name, score) in enumerate(zip(
                explanation['top_features']['indices'],
                explanation['top_features']['names'],
                explanation['top_features']['values']
            ), 1):
                logger.info(f"    {i}. {name} (idx {idx}): {score:.4f}")
            
            # Visualize the explanation
            self.visualize_subgraph_explanation(explanation)
            
            return True, explanation
            
        except Exception as e:
            logger.error(f"GNNExplainer failed: {e}")
            return False, {}
    
    def visualize_subgraph_explanation(self, explanation: dict):
        """Visualize the subgraph explanation."""
        import networkx as nx
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Create subgraph
        G = nx.Graph()
        edge_list = explanation['subgraph_edges'].cpu().t().numpy()
        for i in range(edge_list.shape[0]):
            G.add_edge(int(edge_list[i, 0]), int(edge_list[i, 1]))
        
        # Plot 1: Subgraph structure
        ax = axes[0]
        pos = nx.spring_layout(G)
        
        # Color nodes based on importance
        node_colors = ['red' if n == explanation['node_idx'] else 'lightblue' 
                      for n in G.nodes()]
        
        nx.draw(G, pos, ax=ax, node_color=node_colors, 
                node_size=300, with_labels=True, font_size=8)
        ax.set_title(f"Subgraph for Node {explanation['node_idx']}")
        
        # Plot 2: Edge importance
        ax = axes[1]
        edge_importance = explanation['edge_importance'].cpu().numpy()
        ax.bar(range(len(edge_importance)), sorted(edge_importance, reverse=True))
        ax.set_xlabel('Edge Index')
        ax.set_ylabel('Importance Score')
        ax.set_title('Edge Importance Distribution')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Feature importance heatmap
        ax = axes[2]
        feature_imp = explanation['feature_importance'].cpu().numpy()
        feature_imp_2d = feature_imp.reshape(-1, 1)
        sns.heatmap(feature_imp_2d.T, annot=False, cmap='YlOrRd', 
                   ax=ax, cbar_kws={'label': 'Importance'})
        ax.set_xlabel('Feature Index')
        ax.set_title('Feature Importance Heatmap')
        
        plt.suptitle(f"GNNExplainer Results - Node {explanation['node_idx']} "
                    f"(Class {explanation['target_class']})", fontsize=14)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"gnn_explanation_node_{explanation['node_idx']}.png", 
                   dpi=100, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved GNNExplainer visualization to {OUTPUT_DIR}")
    
    def test_clustering_explanation(self):
        """Test explanation of clustering decisions."""
        logger.info("\n" + "="*50)
        logger.info("Testing Clustering Decision Explanation")
        logger.info("="*50)
        
        try:
            # Get model predictions
            with torch.no_grad():
                output = self.model(self.test_data)
            
            if 'clustering_cluster_assignments' in output:
                cluster_probs = output['clustering_cluster_assignments']
            else:
                cluster_probs = torch.softmax(output['predictions'], dim=-1)
            
            # Find nodes with high confidence clustering
            max_probs, cluster_assignments = cluster_probs.max(dim=1)
            high_confidence_nodes = torch.where(max_probs > 0.7)[0]
            
            if len(high_confidence_nodes) > 0:
                # Select a high-confidence node
                node_idx = high_confidence_nodes[0].item()
                cluster_id = cluster_assignments[node_idx].item()
                confidence = max_probs[node_idx].item()
                
                logger.info(f"\nExplaining clustering decision for node {node_idx}")
                logger.info(f"  Assigned to cluster: {cluster_id}")
                logger.info(f"  Confidence: {confidence:.3f}")
                
                # Generate comprehensive explanation
                explanation = self.generate_clustering_explanation(
                    node_idx, cluster_id, confidence
                )
                
                # Create natural language explanation
                nl_explanation = self.create_natural_language_explanation(
                    node_idx, cluster_id, explanation
                )
                
                logger.info("\nNatural Language Explanation:")
                logger.info(nl_explanation)
                
                # Save explanation
                self.save_clustering_explanation(node_idx, cluster_id, explanation, nl_explanation)
                
                return True, explanation
            else:
                logger.warning("No high-confidence clustering decisions found")
                return False, {}
                
        except Exception as e:
            logger.error(f"Clustering explanation failed: {e}")
            return False, {}
    
    def generate_clustering_explanation(self, node_idx: int, cluster_id: int, 
                                       confidence: float) -> dict:
        """Generate comprehensive explanation for clustering decision."""
        explanation = {
            'node_idx': node_idx,
            'cluster_id': cluster_id,
            'confidence': confidence
        }
        
        # Get feature importance
        feature_importance = self.feature_analyzer.comprehensive_importance(
            self.test_data, node_idx
        )
        explanation['feature_importance'] = feature_importance
        
        # Get subgraph explanation
        subgraph_explanation = self.explainer.explain_node(
            self.test_data, node_idx, target_class=cluster_id
        )
        explanation['subgraph'] = subgraph_explanation
        
        # Analyze neighbor influence
        neighbor_influence = self.analyze_neighbor_influence(node_idx)
        explanation['neighbor_influence'] = neighbor_influence
        
        # Get cluster characteristics
        cluster_chars = self.analyze_cluster_characteristics(cluster_id)
        explanation['cluster_characteristics'] = cluster_chars
        
        return explanation
    
    def analyze_neighbor_influence(self, node_idx: int) -> dict:
        """Analyze influence of neighbors on node's clustering."""
        edge_index = self.test_data.edge_index
        
        # Find neighbors
        neighbors = []
        for i in range(edge_index.size(1)):
            if edge_index[0, i] == node_idx:
                neighbors.append(edge_index[1, i].item())
            elif edge_index[1, i] == node_idx:
                neighbors.append(edge_index[0, i].item())
        
        # Get predictions for neighbors
        with torch.no_grad():
            output = self.model(self.test_data)
            
        if 'clustering_cluster_assignments' in output:
            cluster_probs = output['clustering_cluster_assignments']
        else:
            cluster_probs = torch.softmax(output['predictions'], dim=-1)
        
        neighbor_clusters = []
        for n in neighbors[:10]:  # Limit to 10 neighbors
            if n < cluster_probs.size(0):
                cluster = cluster_probs[n].argmax().item()
                prob = cluster_probs[n].max().item()
                neighbor_clusters.append({
                    'node': n,
                    'cluster': cluster,
                    'confidence': prob
                })
        
        # Calculate influence metrics
        if neighbor_clusters:
            same_cluster = sum(1 for nc in neighbor_clusters 
                             if nc['cluster'] == cluster_probs[node_idx].argmax().item())
            influence_score = same_cluster / len(neighbor_clusters)
        else:
            influence_score = 0
        
        return {
            'num_neighbors': len(neighbors),
            'analyzed_neighbors': neighbor_clusters,
            'same_cluster_ratio': influence_score
        }
    
    def analyze_cluster_characteristics(self, cluster_id: int) -> dict:
        """Analyze characteristics of a cluster."""
        with torch.no_grad():
            output = self.model(self.test_data)
            
        if 'clustering_cluster_assignments' in output:
            cluster_probs = output['clustering_cluster_assignments']
        else:
            cluster_probs = torch.softmax(output['predictions'], dim=-1)
        
        # Find all nodes in this cluster
        cluster_assignments = cluster_probs.argmax(dim=1)
        cluster_nodes = torch.where(cluster_assignments == cluster_id)[0]
        
        if len(cluster_nodes) > 0:
            # Calculate cluster statistics
            cluster_features = self.test_data.x[cluster_nodes]
            
            return {
                'size': len(cluster_nodes),
                'feature_mean': cluster_features.mean(dim=0).cpu().tolist()[:5],  # First 5 features
                'feature_std': cluster_features.std(dim=0).cpu().tolist()[:5],
                'avg_confidence': cluster_probs[cluster_nodes, cluster_id].mean().item()
            }
        else:
            return {
                'size': 0,
                'feature_mean': [],
                'feature_std': [],
                'avg_confidence': 0
            }
    
    def create_natural_language_explanation(self, node_idx: int, cluster_id: int,
                                           explanation: dict) -> str:
        """Create a natural language explanation of the clustering decision."""
        nl_parts = []
        
        # Introduction
        nl_parts.append(f"Building {node_idx} was assigned to Cluster {cluster_id} "
                       f"with {explanation.get('confidence', 0):.1%} confidence.")
        
        # Top features
        if 'feature_importance' in explanation:
            top_features = explanation['feature_importance'].get('top_features', {})
            if top_features and 'names' in top_features:
                features_str = ", ".join(top_features['names'][:3])
                nl_parts.append(f"The most influential features were: {features_str}.")
        
        # Neighbor influence
        if 'neighbor_influence' in explanation:
            neighbor_info = explanation['neighbor_influence']
            ratio = neighbor_info.get('same_cluster_ratio', 0)
            num_neighbors = neighbor_info.get('num_neighbors', 0)
            nl_parts.append(f"Among {num_neighbors} connected buildings, "
                          f"{ratio:.0%} are in the same cluster, indicating "
                          f"{'strong' if ratio > 0.7 else 'moderate' if ratio > 0.4 else 'weak'} "
                          f"local clustering consistency.")
        
        # Cluster characteristics
        if 'cluster_characteristics' in explanation:
            cluster_info = explanation['cluster_characteristics']
            size = cluster_info.get('size', 0)
            avg_conf = cluster_info.get('avg_confidence', 0)
            nl_parts.append(f"Cluster {cluster_id} contains {size} buildings "
                          f"with an average assignment confidence of {avg_conf:.1%}.")
        
        # Practical implications
        nl_parts.append("\nPractical Implications:")
        nl_parts.append("- Buildings in this cluster share similar energy patterns")
        nl_parts.append("- They would benefit from coordinated energy interventions")
        nl_parts.append("- Potential for shared resources like community batteries")
        
        return "\n".join(nl_parts)
    
    def save_clustering_explanation(self, node_idx: int, cluster_id: int,
                                   explanation: dict, nl_explanation: str):
        """Save clustering explanation to file."""
        output_file = OUTPUT_DIR / f"clustering_explanation_node_{node_idx}.json"
        
        # Prepare data for JSON serialization
        save_data = {
            'node_idx': node_idx,
            'cluster_id': cluster_id,
            'confidence': float(explanation.get('confidence', 0)),
            'natural_language_explanation': nl_explanation,
            'feature_importance': {
                'top_features': explanation.get('feature_importance', {}).get('top_features', {})
            },
            'neighbor_influence': explanation.get('neighbor_influence', {}),
            'cluster_characteristics': explanation.get('cluster_characteristics', {})
        }
        
        # Convert tensors to lists
        def convert_tensors(obj):
            if torch.is_tensor(obj):
                return obj.cpu().tolist()
            elif isinstance(obj, dict):
                return {k: convert_tensors(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors(v) for v in obj]
            return obj
        
        save_data = convert_tensors(save_data)
        
        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Saved clustering explanation to {output_file}")
    
    def run_all_tests(self):
        """Run all explainability tests."""
        logger.info("\n" + "="*60)
        logger.info("STARTING COMPREHENSIVE EXPLAINABILITY TESTS")
        logger.info("="*60)
        
        results = {
            'attention_visualization': {'success': False},
            'feature_importance': {'success': False},
            'gnn_explainer': {'success': False},
            'clustering_explanation': {'success': False}
        }
        
        # Setup
        self.setup()
        
        # Test 1: Attention Visualization
        success, attention_results = self.test_attention_visualization()
        results['attention_visualization'] = {
            'success': success,
            'results': attention_results
        }
        
        # Test 2: Feature Importance
        success, importance_results = self.test_feature_importance()
        results['feature_importance'] = {
            'success': success,
            'results': importance_results if success else {}
        }
        
        # Test 3: GNN Explainer
        success, explainer_results = self.test_gnn_explainer()
        results['gnn_explainer'] = {
            'success': success,
            'results': explainer_results if success else {}
        }
        
        # Test 4: Clustering Explanation
        success, clustering_results = self.test_clustering_explanation()
        results['clustering_explanation'] = {
            'success': success,
            'results': clustering_results if success else {}
        }
        
        # Generate summary report
        self.generate_summary_report(results)
        
        return results
    
    def generate_summary_report(self, results: dict):
        """Generate a summary report of all tests."""
        logger.info("\n" + "="*60)
        logger.info("EXPLAINABILITY TEST SUMMARY")
        logger.info("="*60)
        
        # Calculate overall success rate
        total_tests = len(results)
        successful_tests = sum(1 for r in results.values() if r['success'])
        success_rate = (successful_tests / total_tests) * 100
        
        logger.info(f"\nOverall Success Rate: {success_rate:.1f}% ({successful_tests}/{total_tests})")
        
        # Individual test results
        logger.info("\nIndividual Test Results:")
        for test_name, test_result in results.items():
            status = "PASSED" if test_result['success'] else "FAILED"
            logger.info(f"  {test_name}: {status}")
        
        # Key findings
        logger.info("\nKey Findings:")
        
        if results['attention_visualization']['success']:
            logger.info("  - Attention mechanisms are properly captured and visualized")
        else:
            logger.info("  - Attention visualization needs improvement or model lacks attention layers")
        
        if results['feature_importance']['success']:
            logger.info("  - Feature importance can be extracted using multiple methods")
        else:
            logger.info("  - Feature importance extraction encountered issues")
        
        if results['gnn_explainer']['success']:
            logger.info("  - GNNExplainer successfully identifies important subgraphs")
        else:
            logger.info("  - GNNExplainer needs debugging or optimization")
        
        if results['clustering_explanation']['success']:
            logger.info("  - Clustering decisions can be explained comprehensively")
        else:
            logger.info("  - Clustering explanation generation needs work")
        
        # Save summary to file
        summary_file = OUTPUT_DIR / "explainability_test_summary.json"
        
        # Prepare results for JSON
        def clean_for_json(obj):
            if isinstance(obj, (torch.Tensor, np.ndarray)):
                return None
            elif isinstance(obj, dict):
                return {k: clean_for_json(v) for k, v in obj.items() if v is not None}
            elif isinstance(obj, list):
                return [clean_for_json(v) for v in obj if v is not None]
            return obj
        
        clean_results = clean_for_json(results)
        clean_results['summary'] = {
            'total_tests': total_tests,
            'successful_tests': successful_tests,
            'success_rate': success_rate,
            'output_directory': str(OUTPUT_DIR)
        }
        
        with open(summary_file, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        logger.info(f"\nResults saved to: {summary_file}")
        logger.info(f"Visualizations saved to: {OUTPUT_DIR}/")


def main():
    """Main function to run explainability tests."""
    tester = ExplainabilityTester()
    results = tester.run_all_tests()
    
    # Print final status
    print("\n" + "="*60)
    print("EXPLAINABILITY TESTING COMPLETE")
    print("="*60)
    
    # Return success/failure code
    all_passed = all(r['success'] for r in results.values())
    return 0 if all_passed else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)