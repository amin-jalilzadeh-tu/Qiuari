#!/usr/bin/env python3
"""
Test clustering behavior to identify cluster collapse issue
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from models.base_gnn import HeteroEnergyGNN
from torch_geometric.data import HeteroData
import yaml

def test_clustering():
    # Load config
    with open('config/unified_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print('Config num_clusters:', config['model']['num_clusters'])

    # Create model
    model = HeteroEnergyGNN(config['model'])

    # Create test data with realistic structure
    data = HeteroData()
    data['building'].x = torch.randn(160, 17)
    data['cable_group'].x = torch.randn(20, 12) 
    data['transformer'].x = torch.randn(5, 8)

    # Create edges - ensure proper connectivity
    building_ids = torch.arange(160)
    cable_ids = torch.randint(0, 20, (160,))  # Each building connects to one cable group
    edge_index = torch.stack([building_ids, cable_ids])
    data[('building', 'connected_to', 'cable_group')].edge_index = edge_index

    # Add LV group assignments (match cable connections)
    data.lv_group_ids = cable_ids

    print('Buildings:', data['building'].x.shape[0])
    print('Cable groups:', data['cable_group'].x.shape[0])

    # Test clustering
    model.eval()
    with torch.no_grad():
        outputs = model(data, task='clustering')
        print('Clustering output keys:', outputs.keys())
        
        if 'cluster_logits' in outputs:
            cluster_logits = outputs['cluster_logits']
            print('\n=== CLUSTER ANALYSIS ===')
            print('Cluster logits shape:', cluster_logits.shape)
            print('Cluster logits range:', f'{cluster_logits.min().item():.4f} to {cluster_logits.max().item():.4f}')
            
            # Check if logits are all similar (causing collapse)
            cluster_probs = torch.softmax(cluster_logits, dim=1)
            print('\nCluster probabilities stats:')
            print('  Max prob per building:', f'{cluster_probs.max(dim=1)[0].mean().item():.4f}')
            print('  Min prob per building:', f'{cluster_probs.min(dim=1)[0].mean().item():.4f}')
            entropy = (-cluster_probs * torch.log(cluster_probs + 1e-8)).sum(dim=1).mean().item()
            print('  Entropy (higher=more diverse):', f'{entropy:.4f}')
            
            # Get hard assignments
            cluster_assignments = cluster_logits.argmax(dim=-1)
            unique_clusters = torch.unique(cluster_assignments)
            print(f'\nUnique cluster IDs: {unique_clusters.tolist()}')
            
            # Count buildings per cluster
            cluster_counts = torch.bincount(cluster_assignments, minlength=config['model']['num_clusters'])
            non_empty_clusters = (cluster_counts > 0).sum().item()
            print(f'Non-empty clusters: {non_empty_clusters}/{config["model"]["num_clusters"]}')
            
            # Show only non-zero cluster counts
            non_zero_clusters = [(i, count.item()) for i, count in enumerate(cluster_counts) if count > 0]
            print('Buildings per cluster (non-zero only):')
            for cluster_id, count in non_zero_clusters:
                print(f'  Cluster {cluster_id}: {count} buildings')
                
            # Check if cluster sizes are reasonable
            max_cluster_size = max(count for _, count in non_zero_clusters)
            min_cluster_size = min(count for _, count in non_zero_clusters)
            avg_cluster_size = len(cluster_assignments) / len(non_zero_clusters) if non_zero_clusters else 0
            
            print(f'Cluster size stats: min={min_cluster_size}, max={max_cluster_size}, avg={avg_cluster_size:.1f}')
            
            if len(unique_clusters) == 1:
                print(f'\nPROBLEM: All {len(cluster_assignments)} buildings assigned to cluster {unique_clusters[0].item()}')
                
                # Analyze the issue
                print('\n=== DEBUGGING CLUSTER COLLAPSE ===')
                
                # Check DiffPool output if available
                if 'clusters' in outputs:
                    diffpool_S = outputs['clusters']
                    print('DiffPool S matrix shape:', diffpool_S.shape)
                    print('DiffPool S max values per cluster:', diffpool_S.max(dim=0)[0])
                    print('DiffPool S sum per cluster:', diffpool_S.sum(dim=0))
                
                # Check TaskHeads clustering output
                print('TaskHeads clustering logits per cluster:')
                for i in range(min(10, cluster_logits.shape[1])):
                    cluster_i_logits = cluster_logits[:, i]
                    print(f'  Cluster {i}: mean={cluster_i_logits.mean():.4f}, std={cluster_i_logits.std():.4f}')
                
                # Check building feature variance
                building_var = data['building'].x.var(dim=0).mean().item()
                print(f'Building feature variance: {building_var:.6f}')
                
                if building_var < 1e-6:
                    print('WARNING: Buildings have very low feature variance - may cause collapse')
                
                # Check model initialization
                print('\n=== MODEL ANALYSIS ===')
                
                # Check TaskHeads clustering head
                if hasattr(model.task_heads, 'clustering_head'):
                    ch = model.task_heads.clustering_head
                    print('Clustering head output dim:', ch[-1].out_features)
                    
                    # Check if last layer weights are initialized properly
                    final_weights = ch[-1].weight
                    print('Final layer weight stats:')
                    print(f'  Mean: {final_weights.mean().item():.6f}')
                    print(f'  Std: {final_weights.std().item():.6f}')
                    print(f'  Min: {final_weights.min().item():.6f}')
                    print(f'  Max: {final_weights.max().item():.6f}')
                    
                    if final_weights.std().item() < 1e-4:
                        print('WARNING: Final layer weights have very low variance - may cause collapse')
                
                return False
                
            else:
                print(f'\nGOOD: Buildings distributed across {len(unique_clusters)} clusters')
                return True

if __name__ == "__main__":
    test_clustering()