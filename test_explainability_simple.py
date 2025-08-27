"""
Simple explainability test script that avoids OpenMP conflicts.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import warnings
warnings.filterwarnings('ignore')

# Setup paths
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Create output directory
OUTPUT_DIR = Path("explainability_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def create_synthetic_data():
    """Create synthetic building data for testing."""
    num_nodes = 50
    num_features = 17
    
    # Feature names matching the actual building features
    feature_names = [
        'area', 'energy_score', 'solar_score', 'electrify_score', 'age',
        'roof_area', 'height', 'has_solar', 'has_battery', 'has_heat_pump',
        'shared_walls', 'x_coord', 'y_coord', 'avg_electricity_demand',
        'avg_heating_demand', 'peak_electricity_demand', 'energy_intensity'
    ]
    
    # Create realistic synthetic features
    features = torch.zeros(num_nodes, num_features)
    
    # Area (50-500 m²)
    features[:, 0] = torch.rand(num_nodes) * 450 + 50
    
    # Energy score (0-1)
    features[:, 1] = torch.rand(num_nodes)
    
    # Solar score (0-1)
    features[:, 2] = torch.rand(num_nodes)
    
    # Electrification score (0-1)
    features[:, 3] = torch.rand(num_nodes)
    
    # Age (0-100 years)
    features[:, 4] = torch.rand(num_nodes) * 100
    
    # Roof area (20-200 m²)
    features[:, 5] = torch.rand(num_nodes) * 180 + 20
    
    # Height (3-30 m)
    features[:, 6] = torch.rand(num_nodes) * 27 + 3
    
    # Binary features (has_solar, has_battery, has_heat_pump)
    features[:, 7:10] = torch.bernoulli(torch.ones(num_nodes, 3) * 0.3)
    
    # Shared walls (0-3)
    features[:, 10] = torch.randint(0, 4, (num_nodes,))
    
    # Coordinates
    features[:, 11:13] = torch.randn(num_nodes, 2) * 100
    
    # Demand features (positive values)
    features[:, 13:17] = torch.abs(torch.randn(num_nodes, 4)) * 100
    
    # Create edges (spatial proximity)
    edges = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            dist = torch.norm(features[i, 11:13] - features[j, 11:13])
            if dist < 50:  # Connect if within 50 units
                edges.append([i, j])
                edges.append([j, i])
    
    if len(edges) == 0:
        # Fallback: create random edges
        num_edges = num_nodes * 3
        for _ in range(num_edges):
            i, j = torch.randint(0, num_nodes, (2,))
            if i != j:
                edges.append([i.item(), j.item()])
    
    edge_index = torch.tensor(edges).t()
    
    # Create cluster assignments based on feature similarity
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=5, random_state=42)
    cluster_labels = kmeans.fit_predict(features.numpy())
    
    return features, edge_index, torch.tensor(cluster_labels), feature_names


def test_simple_attention():
    """Test simple attention mechanism."""
    logger.info("\n" + "="*50)
    logger.info("Testing Simple Attention Mechanism")
    logger.info("="*50)
    
    # Create simple attention layer
    class SimpleAttention(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.W = nn.Linear(dim, dim)
            self.a = nn.Linear(2 * dim, 1)
            
        def forward(self, x, edge_index):
            # Transform features
            h = self.W(x)
            
            # Compute attention for each edge
            num_edges = edge_index.size(1)
            attention_scores = []
            
            for e in range(min(num_edges, 100)):  # Limit for testing
                i, j = edge_index[0, e], edge_index[1, e]
                edge_feat = torch.cat([h[i], h[j]])
                score = torch.sigmoid(self.a(edge_feat))
                attention_scores.append(score.item())
            
            return attention_scores
    
    # Test with synthetic data
    features, edge_index, _, _ = create_synthetic_data()
    
    att_layer = SimpleAttention(features.size(1))
    attention_scores = att_layer(features, edge_index)
    
    # Visualize attention distribution
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(attention_scores, bins=20, edgecolor='black', alpha=0.7)
    plt.xlabel('Attention Score')
    plt.ylabel('Frequency')
    plt.title('Attention Score Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(sorted(attention_scores, reverse=True), 'o-', markersize=3)
    plt.xlabel('Edge Rank')
    plt.ylabel('Attention Score')
    plt.title('Sorted Attention Scores')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'simple_attention_test.png', dpi=100)
    plt.close()
    
    logger.info(f"Generated {len(attention_scores)} attention scores")
    logger.info(f"Mean attention: {np.mean(attention_scores):.4f}")
    logger.info(f"Std attention: {np.std(attention_scores):.4f}")
    logger.info(f"Max attention: {np.max(attention_scores):.4f}")
    logger.info(f"Min attention: {np.min(attention_scores):.4f}")
    
    return True


def test_gradient_feature_importance():
    """Test gradient-based feature importance."""
    logger.info("\n" + "="*50)
    logger.info("Testing Gradient-Based Feature Importance")
    logger.info("="*50)
    
    # Create simple model
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, num_classes):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 32)
            self.fc2 = nn.Linear(32, num_classes)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            return self.fc2(x)
    
    # Get data
    features, _, cluster_labels, feature_names = create_synthetic_data()
    features.requires_grad = True
    
    # Create and run model
    model = SimpleModel(features.size(1), 5)
    output = model(features)
    
    # Select a node to explain
    node_idx = 0
    target_class = cluster_labels[node_idx]
    
    # Compute gradients
    loss = output[node_idx, target_class]
    loss.backward()
    
    # Get feature importance
    importance = features.grad[node_idx].abs()
    importance = importance / importance.sum()
    
    # Get top features
    top_k = 5
    top_values, top_indices = torch.topk(importance, k=top_k)
    
    # Visualize
    plt.figure(figsize=(12, 5))
    
    # Bar plot of all features
    plt.subplot(1, 2, 1)
    plt.bar(range(len(importance)), importance.detach().numpy())
    plt.xlabel('Feature Index')
    plt.ylabel('Importance Score')
    plt.title('All Feature Importance Scores')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    # Top features
    plt.subplot(1, 2, 2)
    top_names = [feature_names[i] for i in top_indices]
    plt.barh(range(top_k), top_values.detach().numpy())
    plt.yticks(range(top_k), top_names)
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_k} Important Features')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'gradient_importance_node_{node_idx}.png', dpi=100)
    plt.close()
    
    logger.info(f"Explaining node {node_idx} (cluster {target_class.item()})")
    logger.info("Top 5 important features:")
    for i, (idx, name, score) in enumerate(zip(top_indices, top_names, top_values), 1):
        logger.info(f"  {i}. {name}: {score.item():.4f}")
    
    return True


def test_perturbation_importance():
    """Test perturbation-based feature importance."""
    logger.info("\n" + "="*50)
    logger.info("Testing Perturbation-Based Feature Importance")
    logger.info("="*50)
    
    # Simple predictor
    def predict(features, weights=None):
        if weights is None:
            weights = torch.randn(features.size(1))
        scores = features @ weights
        return torch.softmax(scores.unsqueeze(1).repeat(1, 5), dim=1)
    
    # Get data
    features, _, _, feature_names = create_synthetic_data()
    
    # Fixed weights for consistency
    weights = torch.randn(features.size(1))
    
    # Original prediction for node 0
    node_idx = 0
    original_pred = predict(features, weights)[node_idx]
    original_class = original_pred.argmax()
    original_conf = original_pred.max()
    
    # Perturb each feature
    importance_scores = []
    n_samples = 20
    
    for feat_idx in range(features.size(1)):
        changes = []
        original_val = features[node_idx, feat_idx].clone()
        
        for _ in range(n_samples):
            # Perturb feature
            noise = torch.randn(1) * features[:, feat_idx].std()
            features[node_idx, feat_idx] = original_val + noise
            
            # New prediction
            new_pred = predict(features, weights)[node_idx]
            change = abs(new_pred.max() - original_conf).item()
            changes.append(change)
        
        # Restore original
        features[node_idx, feat_idx] = original_val
        
        # Average change as importance
        importance_scores.append(np.mean(changes))
    
    # Normalize
    importance_scores = np.array(importance_scores)
    importance_scores = importance_scores / (importance_scores.sum() + 1e-8)
    
    # Visualize
    plt.figure(figsize=(10, 6))
    
    # Create color map based on importance
    colors = plt.cm.YlOrRd(importance_scores / importance_scores.max())
    
    plt.bar(range(len(importance_scores)), importance_scores, color=colors)
    plt.xlabel('Feature')
    plt.ylabel('Importance Score')
    plt.title(f'Perturbation-Based Feature Importance (Node {node_idx})')
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'perturbation_importance_node_{node_idx}.png', dpi=100)
    plt.close()
    
    # Log results
    top_k = 5
    top_indices = np.argsort(importance_scores)[-top_k:][::-1]
    
    logger.info(f"Original prediction: class {original_class} (conf: {original_conf:.3f})")
    logger.info(f"Top {top_k} important features by perturbation:")
    for i, idx in enumerate(top_indices, 1):
        logger.info(f"  {i}. {feature_names[idx]}: {importance_scores[idx]:.4f}")
    
    return True


def test_subgraph_explanation():
    """Test subgraph-based explanation."""
    logger.info("\n" + "="*50)
    logger.info("Testing Subgraph-Based Explanation")
    logger.info("="*50)
    
    features, edge_index, cluster_labels, feature_names = create_synthetic_data()
    
    # Select a node to explain
    node_idx = 10
    cluster = cluster_labels[node_idx].item()
    
    # Extract k-hop subgraph
    def get_k_hop_subgraph(edge_index, node_idx, k=2):
        subgraph_nodes = {node_idx}
        
        for _ in range(k):
            new_nodes = set()
            for node in subgraph_nodes:
                # Find neighbors
                mask = (edge_index[0] == node) | (edge_index[1] == node)
                neighbors = edge_index[:, mask].unique().tolist()
                new_nodes.update(neighbors)
            subgraph_nodes.update(new_nodes)
        
        return list(subgraph_nodes)
    
    subgraph_nodes = get_k_hop_subgraph(edge_index, node_idx, k=2)
    
    # Analyze subgraph
    subgraph_features = features[subgraph_nodes]
    subgraph_clusters = cluster_labels[subgraph_nodes]
    
    # Calculate statistics
    same_cluster = (subgraph_clusters == cluster).sum().item()
    cluster_consistency = same_cluster / len(subgraph_nodes)
    
    # Feature similarity
    node_feat = features[node_idx]
    similarities = []
    for i, idx in enumerate(subgraph_nodes):
        if idx != node_idx:
            sim = torch.cosine_similarity(node_feat.unsqueeze(0), 
                                         features[idx].unsqueeze(0))
            similarities.append(sim.item())
    
    avg_similarity = np.mean(similarities) if similarities else 0
    
    # Visualize subgraph
    import networkx as nx
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Create subgraph network
    G = nx.Graph()
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        if src in subgraph_nodes and dst in subgraph_nodes:
            G.add_edge(src, dst)
    
    # Plot 1: Subgraph structure
    ax = axes[0]
    pos = nx.spring_layout(G, seed=42)
    
    node_colors = []
    for node in G.nodes():
        if node == node_idx:
            node_colors.append('red')
        elif cluster_labels[node] == cluster:
            node_colors.append('lightgreen')
        else:
            node_colors.append('lightblue')
    
    nx.draw(G, pos, ax=ax, node_color=node_colors, 
            node_size=300, with_labels=True, font_size=8)
    ax.set_title(f'Subgraph around Node {node_idx}\n'
                f'(Red: target, Green: same cluster, Blue: different)')
    
    # Plot 2: Feature comparison
    ax = axes[1]
    
    # Compare average features of same vs different cluster nodes
    same_cluster_mask = subgraph_clusters == cluster
    if same_cluster_mask.any() and (~same_cluster_mask).any():
        same_feat_avg = subgraph_features[same_cluster_mask].mean(0)
        diff_feat_avg = subgraph_features[~same_cluster_mask].mean(0)
        
        x_pos = np.arange(5)  # Show first 5 features
        width = 0.35
        
        ax.bar(x_pos - width/2, same_feat_avg[:5].numpy(), width, 
              label='Same Cluster', color='lightgreen')
        ax.bar(x_pos + width/2, diff_feat_avg[:5].numpy(), width, 
              label='Different Cluster', color='lightblue')
        
        ax.set_xlabel('Feature')
        ax.set_ylabel('Average Value')
        ax.set_title('Feature Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names[:5], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Cluster distribution
    ax = axes[2]
    unique_clusters, counts = torch.unique(subgraph_clusters, return_counts=True)
    colors_pie = ['red' if c == cluster else 'lightblue' 
                  for c in unique_clusters]
    
    ax.pie(counts.numpy(), labels=[f'C{c}' for c in unique_clusters.numpy()],
           colors=colors_pie, autopct='%1.0f%%', startangle=90)
    ax.set_title(f'Cluster Distribution in Subgraph\n'
                f'(Target cluster: C{cluster})')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'subgraph_explanation_node_{node_idx}.png', dpi=100)
    plt.close()
    
    # Log results
    logger.info(f"Subgraph analysis for node {node_idx} (cluster {cluster}):")
    logger.info(f"  Subgraph size: {len(subgraph_nodes)} nodes")
    logger.info(f"  Cluster consistency: {cluster_consistency:.1%}")
    logger.info(f"  Average feature similarity: {avg_similarity:.3f}")
    logger.info(f"  Same cluster nodes: {same_cluster}/{len(subgraph_nodes)}")
    
    return True


def generate_clustering_explanation():
    """Generate comprehensive clustering explanation."""
    logger.info("\n" + "="*50)
    logger.info("Generating Comprehensive Clustering Explanation")
    logger.info("="*50)
    
    features, edge_index, cluster_labels, feature_names = create_synthetic_data()
    
    # Select a node
    node_idx = 5
    cluster = cluster_labels[node_idx].item()
    
    # 1. Identify key features for this cluster
    cluster_mask = cluster_labels == cluster
    cluster_features = features[cluster_mask]
    other_features = features[~cluster_mask]
    
    # Calculate feature differences
    cluster_mean = cluster_features.mean(0)
    other_mean = other_features.mean(0)
    feature_diff = (cluster_mean - other_mean).abs()
    feature_diff = feature_diff / feature_diff.sum()
    
    # 2. Neighbor analysis
    neighbors = []
    for i in range(edge_index.size(1)):
        if edge_index[0, i] == node_idx:
            neighbors.append(edge_index[1, i].item())
        elif edge_index[1, i] == node_idx:
            neighbors.append(edge_index[0, i].item())
    
    neighbor_clusters = [cluster_labels[n].item() for n in neighbors]
    same_cluster_neighbors = sum(1 for c in neighbor_clusters if c == cluster)
    
    # 3. Create natural language explanation
    explanation_parts = []
    
    explanation_parts.append(f"CLUSTERING DECISION EXPLANATION FOR BUILDING {node_idx}")
    explanation_parts.append("=" * 60)
    explanation_parts.append("")
    
    explanation_parts.append(f"Assignment: Cluster {cluster}")
    explanation_parts.append(f"Cluster Size: {cluster_mask.sum().item()} buildings")
    explanation_parts.append("")
    
    # Key features
    top_k = 3
    top_feat_indices = torch.topk(feature_diff, k=top_k).indices
    
    explanation_parts.append("KEY DISTINGUISHING FEATURES:")
    for i, idx in enumerate(top_feat_indices, 1):
        feat_name = feature_names[idx]
        cluster_val = cluster_mean[idx].item()
        other_val = other_mean[idx].item()
        diff_pct = abs(cluster_val - other_val) / (other_val + 1e-8) * 100
        
        explanation_parts.append(f"  {i}. {feat_name}:")
        explanation_parts.append(f"     - This cluster avg: {cluster_val:.2f}")
        explanation_parts.append(f"     - Other clusters avg: {other_val:.2f}")
        explanation_parts.append(f"     - Difference: {diff_pct:.1f}%")
    
    explanation_parts.append("")
    
    # Neighbor influence
    explanation_parts.append("NEIGHBOR INFLUENCE:")
    explanation_parts.append(f"  - Total neighbors: {len(neighbors)}")
    explanation_parts.append(f"  - Same cluster: {same_cluster_neighbors} ({same_cluster_neighbors/max(1,len(neighbors))*100:.1f}%)")
    
    if same_cluster_neighbors > len(neighbors) * 0.5:
        explanation_parts.append("  - Strong local clustering (majority of neighbors in same cluster)")
    else:
        explanation_parts.append("  - Weak local clustering (mixed neighbor clusters)")
    
    explanation_parts.append("")
    
    # Building characteristics
    node_features = features[node_idx]
    explanation_parts.append("BUILDING CHARACTERISTICS:")
    
    # Energy features
    if node_features[7] > 0.5:  # has_solar
        explanation_parts.append("  - Has solar panels installed")
    if node_features[8] > 0.5:  # has_battery
        explanation_parts.append("  - Has battery storage")
    if node_features[9] > 0.5:  # has_heat_pump
        explanation_parts.append("  - Has heat pump")
    
    explanation_parts.append(f"  - Building area: {node_features[0].item():.0f} m²")
    explanation_parts.append(f"  - Energy score: {node_features[1].item():.2f}")
    explanation_parts.append(f"  - Solar potential: {node_features[2].item():.2f}")
    
    explanation_parts.append("")
    
    # Practical implications
    explanation_parts.append("PRACTICAL IMPLICATIONS:")
    
    if cluster_mean[2] > 0.7:  # High solar potential
        explanation_parts.append("  - High solar potential cluster: prioritize for PV installation")
    
    if cluster_mean[13] > other_mean[13]:  # High electricity demand
        explanation_parts.append("  - High electricity demand: consider community battery")
    
    if same_cluster_neighbors > len(neighbors) * 0.7:
        explanation_parts.append("  - Tight spatial cluster: ideal for microgrid development")
    
    explanation_parts.append("  - Coordinate interventions with cluster members for efficiency")
    explanation_parts.append("  - Share resources like batteries or heat networks")
    
    # Save explanation
    explanation_text = "\n".join(explanation_parts)
    
    with open(OUTPUT_DIR / f"clustering_explanation_node_{node_idx}.txt", 'w') as f:
        f.write(explanation_text)
    
    # Also create visual summary
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Feature comparison
    ax = axes[0, 0]
    x_pos = np.arange(len(top_feat_indices))
    top_feat_names = [feature_names[i] for i in top_feat_indices]
    cluster_vals = [cluster_mean[i].item() for i in top_feat_indices]
    other_vals = [other_mean[i].item() for i in top_feat_indices]
    
    width = 0.35
    ax.bar(x_pos - width/2, cluster_vals, width, label=f'Cluster {cluster}', color='green')
    ax.bar(x_pos + width/2, other_vals, width, label='Other Clusters', color='gray')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Average Value')
    ax.set_title('Key Distinguishing Features')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(top_feat_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: All features heatmap
    ax = axes[0, 1]
    feature_comparison = torch.stack([cluster_mean[:10], other_mean[:10]])
    sns.heatmap(feature_comparison.numpy(), annot=True, fmt='.2f', 
               xticklabels=feature_names[:10], 
               yticklabels=[f'Cluster {cluster}', 'Others'],
               cmap='YlOrRd', ax=ax)
    ax.set_title('Feature Heatmap Comparison')
    
    # Plot 3: Neighbor clusters
    ax = axes[1, 0]
    if neighbors:
        unique_neighbor_clusters, counts = np.unique(neighbor_clusters, return_counts=True)
        colors_bar = ['green' if c == cluster else 'lightgray' for c in unique_neighbor_clusters]
        ax.bar(unique_neighbor_clusters, counts, color=colors_bar)
        ax.set_xlabel('Cluster ID')
        ax.set_ylabel('Count')
        ax.set_title(f'Neighbor Cluster Distribution\n(Node {node_idx} in green cluster)')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: Cluster statistics
    ax = axes[1, 1]
    cluster_stats = {
        'Size': cluster_mask.sum().item(),
        'Avg Area': cluster_features[:, 0].mean().item(),
        'Solar %': (cluster_features[:, 7].mean().item() * 100),
        'Battery %': (cluster_features[:, 8].mean().item() * 100),
        'Heat Pump %': (cluster_features[:, 9].mean().item() * 100)
    }
    
    ax.barh(range(len(cluster_stats)), list(cluster_stats.values()), color='green', alpha=0.7)
    ax.set_yticks(range(len(cluster_stats)))
    ax.set_yticklabels(list(cluster_stats.keys()))
    ax.set_xlabel('Value')
    ax.set_title(f'Cluster {cluster} Statistics')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Clustering Explanation for Building {node_idx}', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'clustering_visual_explanation_node_{node_idx}.png', dpi=100)
    plt.close()
    
    # Log summary
    logger.info(f"\nExplanation generated for building {node_idx}:")
    logger.info(f"  Assigned to cluster {cluster} with {cluster_mask.sum().item()} buildings")
    logger.info(f"  Key features: {', '.join(top_feat_names)}")
    logger.info(f"  Neighbor consistency: {same_cluster_neighbors}/{len(neighbors)}")
    logger.info(f"  Full explanation saved to {OUTPUT_DIR}")
    
    return True


def run_all_tests():
    """Run all explainability tests."""
    logger.info("\n" + "="*60)
    logger.info("RUNNING EXPLAINABILITY VALIDATION TESTS")
    logger.info("="*60)
    
    results = {}
    
    # Test 1: Simple Attention
    try:
        results['attention'] = test_simple_attention()
        logger.info("✓ Attention test passed")
    except Exception as e:
        logger.error(f"✗ Attention test failed: {e}")
        results['attention'] = False
    
    # Test 2: Gradient Feature Importance
    try:
        results['gradient_importance'] = test_gradient_feature_importance()
        logger.info("✓ Gradient importance test passed")
    except Exception as e:
        logger.error(f"✗ Gradient importance test failed: {e}")
        results['gradient_importance'] = False
    
    # Test 3: Perturbation Importance
    try:
        results['perturbation_importance'] = test_perturbation_importance()
        logger.info("✓ Perturbation importance test passed")
    except Exception as e:
        logger.error(f"✗ Perturbation importance test failed: {e}")
        results['perturbation_importance'] = False
    
    # Test 4: Subgraph Explanation
    try:
        results['subgraph'] = test_subgraph_explanation()
        logger.info("✓ Subgraph explanation test passed")
    except Exception as e:
        logger.error(f"✗ Subgraph explanation test failed: {e}")
        results['subgraph'] = False
    
    # Test 5: Comprehensive Clustering Explanation
    try:
        results['clustering'] = generate_clustering_explanation()
        logger.info("✓ Clustering explanation test passed")
    except Exception as e:
        logger.error(f"✗ Clustering explanation test failed: {e}")
        results['clustering'] = False
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    logger.info(f"Tests passed: {passed}/{total} ({passed/total*100:.1f}%)")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Save summary
    summary = {
        'tests_run': total,
        'tests_passed': passed,
        'success_rate': passed/total,
        'results': results,
        'output_directory': str(OUTPUT_DIR)
    }
    
    with open(OUTPUT_DIR / 'test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    return results


if __name__ == "__main__":
    results = run_all_tests()
    print(f"\nAll tests completed. Results saved to {OUTPUT_DIR}/")