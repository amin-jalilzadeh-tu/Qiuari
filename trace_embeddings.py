"""
Trace data transformation and embedding pipeline
"""
import torch
import numpy as np
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def trace_data_pipeline():
    """Trace the complete data transformation pipeline"""
    
    print("=" * 80)
    print("TRACING DATA TRANSFORMATION & EMBEDDING PIPELINE")
    print("=" * 80)
    
    # 1. Import necessary modules
    from data.data_loader import EnergyDataLoader
    from data.feature_processor import FeatureProcessor
    from data.graph_constructor import GraphConstructor
    from models.base_gnn import HeteroEnergyGNN
    from models.network_aware_layers import NetworkAwareGNN
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # 2. Load and trace raw data
    print("\n" + "="*60)
    print("STEP 1: RAW DATA LOADING")
    print("="*60)
    
    # Generate synthetic buildings data
    n_buildings = 50
    buildings = []
    for i in range(n_buildings):
        buildings.append({
            'building_id': f'B{i:03d}',
            'energy_label': chr(65 + np.random.randint(0, 7)),  # A-G
            'area': 100 + np.random.randn() * 50,
            'roof_area': 50 + np.random.randn() * 20,
            'height': 10 + np.random.randn() * 5,
            'has_solar': np.random.random() > 0.7,
            'has_battery': np.random.random() > 0.9,
            'has_heat_pump': np.random.random() > 0.8,
            'solar_potential': np.random.random(),
            'electrification': np.random.random(),
            'lv_group': f'LV_{i // 10}',
            'transformer': f'TR_{i // 20}'
        })
    
    # Generate temporal profiles
    temporal_profiles = torch.randn(n_buildings, 24)  # 24 hours
    # Add some structure
    for i in range(n_buildings):
        # Morning peak
        temporal_profiles[i, 7:9] += 1.0
        # Evening peak  
        temporal_profiles[i, 17:20] += 1.5
        # Night time low
        temporal_profiles[i, 0:6] *= 0.3
    temporal_profiles = torch.abs(temporal_profiles)
    
    print(f"✓ Loaded {len(buildings)} buildings")
    print(f"✓ Raw features per building: {buildings[0].keys()}")
    print(f"✓ Temporal profile shape: {temporal_profiles.shape if temporal_profiles is not None else 'None'}")
    
    # Sample raw data
    sample_building = buildings[0]
    print(f"\nSample building raw data:")
    for key, value in sample_building.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    # 3. Trace feature processing
    print("\n" + "="*60)
    print("STEP 2: FEATURE PROCESSING")
    print("="*60)
    
    processor = FeatureProcessor()
    
    # Convert buildings to feature tensor
    node_features = []
    for b in buildings:
        feat = [
            ord(b['energy_label']) - ord('A'),  # Energy label as ordinal
            b['area'] / 200,  # Normalize area
            b['roof_area'] / 100,  # Normalize roof area
            b['height'] / 20,  # Normalize height  
            1.0 if b['has_solar'] else 0.0,
            1.0 if b['has_battery'] else 0.0,
            1.0 if b['has_heat_pump'] else 0.0,
            b['solar_potential'],
            b['electrification'],
            np.random.random(),  # Random features to fill dimension
            np.random.random(),
            np.random.random(),
            np.random.random(),
            np.random.random(),
            np.random.random(),
            np.random.random(),
            np.random.random()
        ]
        node_features.append(feat)
    
    node_features = torch.tensor(node_features, dtype=torch.float32)
    edge_features = None  # No edge features for now
    
    print(f"✓ Node features shape: {node_features.shape}")
    print(f"✓ Feature dimensions: {node_features.shape[1]}")
    print(f"✓ Edge features: {edge_features}")
    
    # Check feature statistics
    print(f"\nFeature statistics:")
    print(f"  Mean: {node_features.mean(dim=0)[:5].numpy()}")  # First 5 features
    print(f"  Std:  {node_features.std(dim=0)[:5].numpy()}")
    print(f"  Min:  {node_features.min(dim=0).values[:5].numpy()}")
    print(f"  Max:  {node_features.max(dim=0).values[:5].numpy()}")
    
    # Check for NaN or Inf
    has_nan = torch.isnan(node_features).any()
    has_inf = torch.isinf(node_features).any()
    print(f"\n✓ Contains NaN: {has_nan}")
    print(f"✓ Contains Inf: {has_inf}")
    
    # 4. Trace graph construction
    print("\n" + "="*60)
    print("STEP 3: GRAPH CONSTRUCTION")
    print("="*60)
    
    # Create graph manually without GraphConstructor
    from torch_geometric.data import Data
    
    # Create edge index (connect nearby buildings)
    edge_list = []
    for i in range(n_buildings):
        # Connect to neighbors in same LV group
        for j in range(n_buildings):
            if i != j and buildings[i]['lv_group'] == buildings[j]['lv_group']:
                edge_list.append([i, j])
    
    if not edge_list:
        # Fallback: create a simple grid connectivity
        for i in range(n_buildings):
            if i > 0:
                edge_list.append([i, i-1])
                edge_list.append([i-1, i])
            if i < n_buildings - 1:
                edge_list.append([i, i+1])
                edge_list.append([i+1, i])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Calculate centrality features
    degrees = torch.bincount(edge_index[0], minlength=n_buildings).float()
    centrality_features = torch.stack([
        degrees / degrees.max() if degrees.max() > 0 else degrees,
        degrees / degrees.max() if degrees.max() > 0 else degrees,  # Repeat for simplicity
        torch.randn(n_buildings) * 0.1,
        torch.randn(n_buildings) * 0.1,
        torch.randn(n_buildings) * 0.1
    ], dim=1)
    
    data = Data(
        x=node_features,
        edge_index=edge_index,
        temporal_profiles=temporal_profiles,
        centrality_features=centrality_features
    )
    
    print(f"✓ Graph nodes: {data.x.shape[0]}")
    print(f"✓ Graph edges: {data.edge_index.shape[1]}")
    print(f"✓ Average degree: {data.edge_index.shape[1] / data.x.shape[0]:.2f}")
    
    # Check connectivity
    degrees = torch.bincount(data.edge_index[0])
    print(f"✓ Min degree: {degrees.min().item()}")
    print(f"✓ Max degree: {degrees.max().item()}")
    print(f"✓ Isolated nodes: {(degrees == 0).sum().item()}")
    
    # 5. Trace through GNN model
    print("\n" + "="*60)
    print("STEP 4: GNN EMBEDDING GENERATION")
    print("="*60)
    
    # Test with base GNN
    model_config = {
        'hidden_dim': 128,
        'num_layers': 3,
        'dropout': 0.1,
        'building_features': node_features.shape[1],
        'cable_group_features': 12,
        'transformer_features': 8,
        'use_positional_encoding': True
    }
    base_model = HeteroEnergyGNN(model_config).to(device)
    
    data = data.to(device)
    base_model.eval()
    
    with torch.no_grad():
        # Forward pass - the model expects Data object or dict
        outputs = base_model(data)
        
        # Extract embeddings per layer (if available)
        embeddings_per_layer = []
        if isinstance(outputs, dict):
            if 'embeddings' in outputs:
                final_embeddings = outputs['embeddings']
                if isinstance(final_embeddings, dict):
                    # Take building embeddings if hetero
                    final_embeddings = final_embeddings.get('building', list(final_embeddings.values())[0])
            else:
                final_embeddings = None
        else:
            final_embeddings = outputs
    
    if embeddings_per_layer:
        print(f"✓ Number of embedding layers captured: {len(embeddings_per_layer)}")
        for i, emb in enumerate(embeddings_per_layer):
            print(f"  Layer {i+1} embedding shape: {emb.shape}")
            print(f"    Mean: {emb.mean():.4f}, Std: {emb.std():.4f}")
            print(f"    Min: {emb.min():.4f}, Max: {emb.max():.4f}")
    
    # Process final embeddings
    if final_embeddings is not None and torch.is_tensor(final_embeddings):
        final_embeddings = final_embeddings.cpu()
    
    if final_embeddings is not None:
        print(f"\n✓ Final embedding shape: {final_embeddings.shape}")
        print(f"  Embedding dimension: {final_embeddings.shape[1]}")
        
        # Check embedding quality
        print(f"\nEmbedding quality metrics:")
        
        # 1. Variance per dimension
        var_per_dim = final_embeddings.var(dim=0)
        print(f"  Average variance per dimension: {var_per_dim.mean():.4f}")
        print(f"  Min variance: {var_per_dim.min():.4f}")
        print(f"  Max variance: {var_per_dim.max():.4f}")
        
        # 2. Dead neurons (dimensions with very low variance)
        dead_neurons = (var_per_dim < 0.01).sum()
        print(f"  Dead neurons (var < 0.01): {dead_neurons}/{final_embeddings.shape[1]}")
        
        # 3. Embedding distances
        pairwise_dist = torch.cdist(final_embeddings, final_embeddings)
        print(f"  Average pairwise distance: {pairwise_dist.mean():.4f}")
        print(f"  Min distance (non-zero): {pairwise_dist[pairwise_dist > 0].min():.4f}")
        print(f"  Max distance: {pairwise_dist.max():.4f}")
        
        # 4. Clustering tendency (ratio of inter vs intra distances)
        # Simple k-means like clustering
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(5, len(buildings)//10), random_state=42)
        clusters = kmeans.fit_predict(final_embeddings.numpy())
        
        print(f"\n✓ Clustering analysis (k={kmeans.n_clusters}):")
        for i in range(kmeans.n_clusters):
            cluster_mask = clusters == i
            print(f"  Cluster {i}: {cluster_mask.sum()} nodes")
    
    # 6. Test Network-Aware model embeddings
    print("\n" + "="*60)
    print("STEP 5: NETWORK-AWARE EMBEDDINGS")
    print("="*60)
    
    network_config = {
        'hidden_dim': 128,
        'num_layers': 4,
        'building_features': node_features.shape[1],
        'use_cascade_tracking': True
    }
    network_model = NetworkAwareGNN(network_config).to(device)
    
    network_model.eval()
    
    # Add required features for network-aware model
    data.centrality_features = torch.randn(data.x.shape[0], 5).to(device)
    boundary_mask = torch.zeros(data.x.shape[0]).to(device)
    grid_level = torch.zeros(data.x.shape[0], dtype=torch.long).to(device)
    
    with torch.no_grad():
        network_outputs = network_model(
            data.x,
            data.edge_index,
            centrality_features=data.centrality_features,
            boundary_mask=boundary_mask,
            grid_level=grid_level
        )
    
    if 'embeddings' in network_outputs:
        network_embeddings = network_outputs['embeddings'].cpu()
        print(f"✓ Network-aware embedding shape: {network_embeddings.shape}")
        
        # Compare with base embeddings
        if final_embeddings is not None:
            similarity = torch.nn.functional.cosine_similarity(
                final_embeddings.mean(dim=0),
                network_embeddings.mean(dim=0),
                dim=0
            )
            print(f"✓ Cosine similarity between base and network embeddings: {similarity:.4f}")
    
    # 7. Visualize embeddings
    print("\n" + "="*60)
    print("STEP 6: EMBEDDING VISUALIZATION")
    print("="*60)
    
    if final_embeddings is not None and len(final_embeddings) > 2:
        # Use t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(final_embeddings)-1))
        embeddings_2d = tsne.fit_transform(final_embeddings.numpy())
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Colored by node index
        scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=range(len(embeddings_2d)), cmap='viridis')
        axes[0].set_title('Embeddings colored by node index')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # Plot 2: Colored by energy label
        energy_labels = [ord(b['energy_label']) - ord('A') for b in buildings]
        scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                                   c=energy_labels, cmap='RdYlGn_r')
        axes[1].set_title('Embeddings colored by energy label')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=axes[1])
        
        plt.tight_layout()
        plt.savefig('embedding_visualization.png', dpi=150)
        print("✓ Saved embedding visualization to 'embedding_visualization.png'")
        plt.close()
    
    # 8. Check for common issues
    print("\n" + "="*60)
    print("VALIDATION CHECKS")
    print("="*60)
    
    issues = []
    
    # Check 1: Feature normalization
    if node_features.std(dim=0).min() < 0.1:
        issues.append("⚠ Some features have very low variance - consider better normalization")
    
    # Check 2: Graph connectivity
    if (degrees == 0).any():
        issues.append("⚠ Graph has isolated nodes - may affect message passing")
    
    # Check 3: Embedding collapse
    if final_embeddings is not None:
        if var_per_dim.min() < 0.001:
            issues.append("⚠ Some embedding dimensions have collapsed (very low variance)")
    
    # Check 4: Gradient flow (if in training mode)
    if final_embeddings is not None and final_embeddings.requires_grad:
        if (final_embeddings.grad == 0).all():
            issues.append("⚠ No gradients flowing to embeddings")
    
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✓ All validation checks passed!")
    
    print("\n" + "="*80)
    print("EMBEDDING PIPELINE TRACE COMPLETE")
    print("="*80)
    
    return {
        'node_features': node_features,
        'embeddings': final_embeddings,
        'network_embeddings': network_embeddings if 'network_embeddings' in locals() else None,
        'data': data
    }

if __name__ == "__main__":
    results = trace_data_pipeline()
    print("\n✓ Tracing complete. Results stored in 'results' variable.")