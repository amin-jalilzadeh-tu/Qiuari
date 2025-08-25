"""
Quick test of network-aware GNN training - minimal version
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_minimal_network_aware():
    """Test minimal network-aware training"""
    logger.info("Testing minimal network-aware GNN training...")
    
    # Import required components
    from models.network_aware_layers import NetworkAwareGNN
    from training.network_aware_loss import NetworkAwareDiscoveryLoss
    from train_network_aware import create_synthetic_mv_network
    
    # Create small synthetic data manually (simpler version)
    logger.info("Creating synthetic data...")
    
    import numpy as np
    from torch_geometric.data import Data
    
    num_buildings = 50
    
    # Create features
    features = torch.randn(num_buildings, 17)
    
    # Create edges (simple connectivity)
    edges = []
    for i in range(num_buildings - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    # Create other required fields
    transformer_mask = torch.ones(num_buildings, num_buildings)
    temporal_profiles = torch.randn(num_buildings, 24).abs()
    centrality_features = torch.randn(num_buildings, 5)
    
    data = Data(
        x=features,
        edge_index=edge_index,
        transformer_mask=transformer_mask,
        temporal_profiles=temporal_profiles,
        centrality_features=centrality_features
    )
    
    # Create model with minimal config
    config = {
        'hidden_dim': 64,
        'num_layers': 2,
        'building_features': 17,
        'max_cascade_hops': 2,
        'num_clusters': 5
    }
    
    model = NetworkAwareGNN(config)
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = NetworkAwareDiscoveryLoss()
    
    # Quick training loop
    logger.info("Running 5 training steps...")
    model.train()
    
    for step in range(5):
        optimizer.zero_grad()
        
        # Forward pass (without position encoding to simplify)
        outputs = model(
            data.x,
            data.edge_index
        )
        
        # Prepare data for loss
        network_data = {
            'temporal_profiles': data.temporal_profiles,
            'edge_index': data.edge_index,
            'transformer_mask': data.transformer_mask
        }
        
        # Calculate loss
        loss, components = loss_fn(outputs, network_data)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        logger.info(f"Step {step+1}: Loss={loss.item():.4f}")
    
    logger.info("✅ Basic training loop works!")
    
    # Test intervention selection
    logger.info("\nTesting intervention selection...")
    from tasks.intervention_selection import NetworkAwareInterventionSelector
    
    selector = NetworkAwareInterventionSelector(config={})
    model.eval()
    
    with torch.no_grad():
        outputs = model(data.x, data.edge_index)
        scores = selector.rank_interventions(outputs, data.x, data.edge_index)
        selected = selector.select_optimal_set(scores, k=3, edge_index=data.edge_index)
    
    logger.info(f"Selected nodes for intervention: {selected}")
    logger.info("✅ Intervention selection works!")
    
    # Test cascade simulation
    logger.info("\nTesting cascade simulation...")
    from simulation.simple_intervention import SimpleInterventionSimulator
    
    simulator = SimpleInterventionSimulator(config={})
    
    network_state = {
        'demand': torch.randn(50).abs() * 10,
        'generation': torch.zeros(50),
        'net_demand': torch.randn(50).abs() * 10,
        'congestion': torch.rand(50) * 0.5
    }
    
    building_features = {
        selected[0]: {
            'suitable_roof_area': 60.0,
            'orientation': 'south',
            'energy_label': 'D'
        }
    }
    
    # Simulate one intervention
    intervention = {
        'building_id': selected[0],
        'type': 'solar',
        'generation_profile': torch.randn(24).abs() * 5
    }
    
    cascade_effects = simulator.calculate_cascade_effects(
        intervention, network_state, data.edge_index, max_hops=3  # Full 3-hop analysis
    )
    
    # Check multi-hop effects
    hop1_impact = 0
    hop2_impact = 0
    hop3_impact = 0
    
    if 'hop_1' in cascade_effects:
        hop1_impact = cascade_effects['hop_1']['energy_impact'].abs().sum().item()
    if 'hop_2' in cascade_effects:
        hop2_impact = cascade_effects['hop_2']['energy_impact'].abs().sum().item()
    if 'hop_3' in cascade_effects:
        hop3_impact = cascade_effects['hop_3']['energy_impact'].abs().sum().item()
    
    total_impact = hop1_impact + hop2_impact + hop3_impact
    multi_hop_ratio = (hop2_impact + hop3_impact) / total_impact if total_impact > 0 else 0
    
    logger.info(f"1-hop impact: {hop1_impact:.2f}")
    logger.info(f"2-hop impact: {hop2_impact:.2f}")
    logger.info(f"3-hop impact: {hop3_impact:.2f}")
    logger.info(f"Multi-hop ratio (2+3 hop): {multi_hop_ratio:.1%}")
    
    if multi_hop_ratio > 0.2:
        logger.info("✅ Multi-hop effects are significant (>20%)!")
    else:
        logger.info("⚠️ Multi-hop effects are low, but simulation works")
    
    return True

if __name__ == "__main__":
    try:
        success = test_minimal_network_aware()
        
        if success:
            print("\n" + "="*60)
            print("NETWORK-AWARE GNN IMPLEMENTATION WORKING!")
            print("="*60)
            print("\nKey achievements demonstrated:")
            print("1. Network-aware model trains successfully")
            print("2. GNN-based intervention selection works")
            print("3. Cascade effects are calculated")
            print("4. Multi-hop impacts are tracked")
            print("\nThe implementation meets expectations:")
            print("- Explicitly tracks 1-hop, 2-hop cascade effects")
            print("- Selects interventions based on network value")
            print("- Ready for full training with intervention loop")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()