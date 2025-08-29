#!/usr/bin/env python3
"""
Test ConstrainedDiffPool specifically to identify cluster imbalance
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from models.pooling_layers import ConstrainedDiffPool

def test_diffpool():
    print("Testing ConstrainedDiffPool cluster balance...")
    
    # Create test data
    N = 160  # buildings
    F = 128  # feature dim
    K = 10   # clusters
    
    # Generate diverse building features
    x = torch.randn(N, F) * 0.5  # Moderate variance
    
    # Add some structure to features to encourage clustering
    cluster_centers = torch.randn(5, F) * 2  # 5 natural clusters
    for i in range(N):
        center_idx = i % 5
        x[i] = cluster_centers[center_idx] + torch.randn(F) * 0.3
    
    # Create random edges
    edge_index = torch.randint(0, N, (2, N * 3))  # Each node has ~3 edges on average
    
    # Create LV group assignments (20 groups)
    lv_group_ids = torch.randint(0, 20, (N,))
    
    # Initialize DiffPool
    pool = ConstrainedDiffPool(
        input_dim=F, 
        max_clusters=K,
        min_cluster_size=5,
        max_cluster_size=25
    )
    
    print(f"Input: {N} buildings, {F} features, {K} max clusters")
    print(f"LV groups: {len(torch.unique(lv_group_ids))} unique groups")
    
    # Test DiffPool
    x_pooled, adj_pooled, S, aux_loss = pool(x, edge_index, lv_group_ids=lv_group_ids)
    
    print("\n=== DIFFPOOL RESULTS ===")
    print(f"Assignment matrix S shape: {S.shape}")
    print(f"Auxiliary loss: {aux_loss.item():.4f}")
    
    # Analyze assignments
    hard_assignments = S.argmax(dim=1)
    unique_clusters = torch.unique(hard_assignments)
    cluster_counts = torch.bincount(hard_assignments, minlength=K)
    
    print(f"Active clusters: {len(unique_clusters)}/{K}")
    print(f"Cluster assignments: {unique_clusters.tolist()}")
    print(f"Buildings per cluster: {cluster_counts.tolist()}")
    
    # Check assignment confidence
    max_probs = S.max(dim=1)[0]
    print(f"Assignment confidence: min={max_probs.min():.3f}, mean={max_probs.mean():.3f}, max={max_probs.max():.3f}")
    
    # Check entropy (diversity of assignments)
    entropy = -(S * torch.log(S + 1e-8)).sum(dim=1).mean()
    print(f"Assignment entropy: {entropy:.3f} (higher = more uncertain)")
    
    # Analyze LV group constraints
    print("\n=== LV GROUP ANALYSIS ===")
    for lv_id in torch.unique(lv_group_ids)[:5]:  # Show first 5 LV groups
        lv_mask = (lv_group_ids == lv_id)
        lv_buildings = torch.where(lv_mask)[0]
        lv_assignments = hard_assignments[lv_mask]
        lv_unique_clusters = torch.unique(lv_assignments)
        
        print(f"LV group {lv_id.item()}: {len(lv_buildings)} buildings -> {len(lv_unique_clusters)} clusters {lv_unique_clusters.tolist()}")
    
    # Check if size constraints are working
    oversized_clusters = (cluster_counts > pool.max_cluster_size).sum()
    undersized_clusters = (cluster_counts > 0) & (cluster_counts < pool.min_cluster_size)
    undersized_count = undersized_clusters.sum()
    
    print(f"\n=== SIZE CONSTRAINT ANALYSIS ===")
    print(f"Oversized clusters (>{pool.max_cluster_size}): {oversized_clusters}")
    print(f"Undersized clusters (<{pool.min_cluster_size}): {undersized_count}")
    
    # Identify issues
    if len(unique_clusters) < 3:
        print("\nISSUE: Too few clusters formed")
    
    max_cluster_size = cluster_counts.max()
    if max_cluster_size > pool.max_cluster_size * 2:
        print(f"ISSUE: Cluster too large ({max_cluster_size} vs limit {pool.max_cluster_size})")
    
    # Check if certain clusters dominate
    largest_cluster_ratio = cluster_counts.max().float() / N
    if largest_cluster_ratio > 0.5:
        print(f"ISSUE: Largest cluster contains {largest_cluster_ratio:.1%} of all buildings")
    
    return S, hard_assignments, cluster_counts

def test_without_constraints():
    """Test what happens without LV constraints"""
    print("\n" + "="*50)
    print("Testing WITHOUT LV constraints...")
    
    N = 160
    F = 128
    K = 10
    
    x = torch.randn(N, F)
    edge_index = torch.randint(0, N, (2, N * 3))
    
    pool = ConstrainedDiffPool(input_dim=F, max_clusters=K)
    
    # No LV constraints
    x_pooled, adj_pooled, S, aux_loss = pool(x, edge_index, lv_group_ids=None)
    
    hard_assignments = S.argmax(dim=1)
    unique_clusters = torch.unique(hard_assignments)
    cluster_counts = torch.bincount(hard_assignments, minlength=K)
    
    print(f"Active clusters: {len(unique_clusters)}/{K}")
    print(f"Buildings per cluster: {cluster_counts.tolist()}")
    
    return cluster_counts

if __name__ == "__main__":
    test_diffpool()
    test_without_constraints()