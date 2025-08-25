"""
Deep debugging: trace EVERY step of the pipeline
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import logging

# Set up detailed logging
logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def step_by_step_trace():
    """Trace every single step"""
    
    logger.info("="*80)
    logger.info("STEP-BY-STEP DEEP TRACE")
    logger.info("="*80)
    
    # ========================================
    # STEP 1: Create minimal data
    # ========================================
    logger.info("\nSTEP 1: Creating minimal dummy data")
    logger.info("-"*40)
    
    n_nodes = 5  # Small for easy tracking
    device = torch.device('cpu')
    
    # Building features [energy_label, area, roof_area, height, has_solar, ...]
    x = torch.zeros(n_nodes, 17)
    x[:, 0] = torch.tensor([0, 1, 2, 3, 4])  # Energy labels A, B, C, D, E
    x[:, 1] = torch.tensor([0.1, 0.15, 0.12, 0.18, 0.14])  # Area
    x[:, 2] = torch.tensor([0.05, 0.08, 0.06, 0.09, 0.07])  # Roof area
    
    logger.info(f"  Created x tensor: shape={x.shape}")
    logger.info(f"  Energy labels: {x[:, 0].tolist()}")
    logger.info(f"  Roof areas: {x[:, 2].tolist()}")
    
    # ========================================
    # STEP 2: Initialize network state
    # ========================================
    logger.info("\nSTEP 2: Initialize network state")
    logger.info("-"*40)
    
    # Initial demand (kW)
    demand = torch.tensor([10.0, 8.0, 12.0, 9.0, 11.0])
    logger.info(f"  Demand: {demand.tolist()}")
    
    # Initial generation (all zeros - no solar yet)
    generation = torch.zeros(n_nodes)
    logger.info(f"  Initial generation: {generation.tolist()}")
    
    # Net demand
    net_demand = demand - generation
    logger.info(f"  Net demand: {net_demand.tolist()}")
    logger.info(f"  Peak demand: {net_demand.max().item():.2f} kW")
    
    network_state = {
        'demand': demand,
        'generation': generation,
        'net_demand': net_demand,
        'congestion': torch.zeros(n_nodes)
    }
    
    # ========================================
    # STEP 3: Select intervention node
    # ========================================
    logger.info("\nSTEP 3: Select node for solar intervention")
    logger.info("-"*40)
    
    selected_node = 2  # Node with highest demand
    logger.info(f"  Selected node: {selected_node}")
    logger.info(f"  Node demand: {demand[selected_node].item():.2f} kW")
    logger.info(f"  Node roof area: {x[selected_node, 2].item() * 100:.2f} m²")
    
    # ========================================
    # STEP 4: Calculate solar generation
    # ========================================
    logger.info("\nSTEP 4: Calculate solar generation")
    logger.info("-"*40)
    
    roof_area = x[selected_node, 2].item() * 100  # Convert to m²
    logger.info(f"  Roof area: {roof_area:.2f} m²")
    
    # Solar capacity: ~6 m² per kWp
    capacity_kwp = min(roof_area / 6.0, 10.0)
    logger.info(f"  Installed capacity: {capacity_kwp:.2f} kWp")
    
    # Generate hourly profile (simplified)
    hours = np.arange(24)
    solar_profile = np.zeros(24)
    for h in range(6, 18):  # Sun from 6am to 6pm
        angle = np.pi * (h - 6) / 12
        solar_profile[h] = capacity_kwp * np.sin(angle) * 0.8  # 80% efficiency
    
    logger.info(f"  Peak generation: {solar_profile.max():.2f} kW")
    logger.info(f"  Daily generation: {solar_profile.sum():.2f} kWh")
    
    # For simplicity, use peak value for this test
    solar_generation = solar_profile.max()
    
    # ========================================
    # STEP 5: Update network state
    # ========================================
    logger.info("\nSTEP 5: Update network state with solar")
    logger.info("-"*40)
    
    # Add solar generation to selected node
    new_generation = generation.clone()
    new_generation[selected_node] = solar_generation
    logger.info(f"  New generation: {new_generation.tolist()}")
    
    # Calculate new net demand
    new_net_demand = demand - new_generation
    logger.info(f"  New net demand: {new_net_demand.tolist()}")
    
    old_peak = net_demand.max().item()
    new_peak = new_net_demand.max().item()
    peak_reduction = (old_peak - new_peak) / old_peak * 100
    
    logger.info(f"  Old peak: {old_peak:.2f} kW")
    logger.info(f"  New peak: {new_peak:.2f} kW")
    logger.info(f"  Peak reduction: {peak_reduction:.2f}%")
    
    # ========================================
    # STEP 6: Calculate cascade effects
    # ========================================
    logger.info("\nSTEP 6: Calculate cascade effects")
    logger.info("-"*40)
    
    # Simple network: linear connection
    neighbors = {
        0: [1],
        1: [0, 2],
        2: [1, 3],
        3: [2, 4],
        4: [3]
    }
    
    # 1-hop neighbors
    hop1_neighbors = neighbors[selected_node]
    logger.info(f"  1-hop neighbors of node {selected_node}: {hop1_neighbors}")
    
    # Energy sharing to neighbors
    excess_energy = max(0, solar_generation - demand[selected_node].item())
    logger.info(f"  Excess energy available: {excess_energy:.2f} kW")
    
    if excess_energy > 0 and hop1_neighbors:
        energy_per_neighbor = excess_energy / len(hop1_neighbors) * 0.95  # 5% loss
        logger.info(f"  Energy shared per neighbor: {energy_per_neighbor:.2f} kW")
        
        for neighbor in hop1_neighbors:
            neighbor_deficit = max(0, new_net_demand[neighbor].item())
            shared = min(energy_per_neighbor, neighbor_deficit)
            logger.info(f"    Node {neighbor}: deficit={neighbor_deficit:.2f}, shared={shared:.2f}")
    
    # ========================================
    # STEP 7: Check for issues
    # ========================================
    logger.info("\nSTEP 7: Issue Check")
    logger.info("-"*40)
    
    issues = []
    
    # Issue 1: Peak reduction
    if peak_reduction < 1.0:
        issues.append(f"Peak reduction too low: {peak_reduction:.2f}%")
    
    # Issue 2: Solar generation
    if solar_generation > 50:  # Unrealistic for residential
        issues.append(f"Solar generation unrealistic: {solar_generation:.2f} kW")
    
    # Issue 3: Cascade values
    total_cascade = excess_energy * len(hop1_neighbors) * 0.95
    if total_cascade > 100:
        issues.append(f"Cascade energy too high: {total_cascade:.2f} kW")
    
    if issues:
        logger.warning("ISSUES FOUND:")
        for issue in issues:
            logger.warning(f"  X {issue}")
    else:
        logger.info("  OK: No issues found")
    
    return {
        'peak_reduction': peak_reduction,
        'solar_generation': solar_generation,
        'cascade_energy': total_cascade if excess_energy > 0 else 0
    }

def test_loss_functions():
    """Test each loss function individually"""
    logger.info("\n" + "="*80)
    logger.info("TESTING LOSS FUNCTIONS")
    logger.info("="*80)
    
    from training.loss_functions import ComplementarityLoss
    
    # Test with simple data
    n = 4
    embeddings = torch.randn(n, 8)
    cluster_probs = torch.eye(n)  # Each node in its own cluster
    profiles = torch.randn(n, 24)
    
    loss_fn = ComplementarityLoss()
    total_loss, components = loss_fn(embeddings, cluster_probs, profiles)
    
    logger.info(f"\nComplementarityLoss test:")
    logger.info(f"  Total: {total_loss.item():.4f}")
    for k, v in components.items():
        if hasattr(v, 'item'):
            logger.info(f"  {k}: {v.item():.4f}")
            if v.item() < 0:
                logger.warning(f"    WARNING: NEGATIVE COMPONENT!")
    
    # Check if total is positive
    if total_loss.item() < 0:
        logger.error("  ERROR: TOTAL LOSS IS NEGATIVE!")
    else:
        logger.info("  OK: Total loss is positive")

def test_cascade_calculation():
    """Test cascade effect calculation in detail"""
    logger.info("\n" + "="*80)
    logger.info("TESTING CASCADE CALCULATION")
    logger.info("="*80)
    
    from simulation.simple_intervention import SimpleInterventionSimulator
    
    sim = SimpleInterventionSimulator(config={})
    
    # Minimal test case
    n_nodes = 3
    device = torch.device('cpu')
    
    # Create simple network state
    network_state = {
        'demand': torch.tensor([10.0, 8.0, 12.0]),
        'generation': torch.tensor([0.0, 0.0, 0.0]),
        'congestion': torch.tensor([0.2, 0.3, 0.1]),
        'net_demand': torch.tensor([10.0, 8.0, 12.0])
    }
    
    # Simple edge: 0-1-2 (linear)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    
    # Intervention on node 0
    intervention = {
        'building_id': 0,
        'type': 'solar',
        'generation_profile': torch.tensor([5.0] * 24)  # Constant 5kW
    }
    
    logger.info(f"\nInitial state:")
    logger.info(f"  Demand: {network_state['demand'].tolist()}")
    logger.info(f"  Generation: {network_state['generation'].tolist()}")
    
    # Calculate cascade
    cascade = sim.calculate_cascade_effects(intervention, network_state, edge_index)
    
    logger.info(f"\nCascade effects:")
    for hop in range(1, 4):
        hop_key = f'hop_{hop}'
        if hop_key in cascade:
            energy = cascade[hop_key]['energy_impact']
            logger.info(f"  {hop_key}:")
            logger.info(f"    Energy impact: {energy.tolist()}")
            total = energy.sum().item()
            logger.info(f"    Total: {total:.2f}")
            
            if total > 100:
                logger.warning(f"    WARNING: UNREALISTIC VALUE!")

if __name__ == "__main__":
    # Run all tests
    results = step_by_step_trace()
    test_loss_functions()
    test_cascade_calculation()
    
    logger.info("\n" + "="*80)
    logger.info("FINAL SUMMARY")
    logger.info("="*80)
    logger.info(f"Peak reduction: {results['peak_reduction']:.2f}%")
    logger.info(f"Solar generation: {results['solar_generation']:.2f} kW")
    logger.info(f"Cascade energy: {results['cascade_energy']:.2f} kW")