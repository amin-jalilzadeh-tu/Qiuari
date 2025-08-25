"""
Trace the cascade value issue - why are we getting 10000+ values?
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def trace_cascade_accumulation():
    """Trace how cascade values accumulate to huge numbers"""
    
    logger.info("="*80)
    logger.info("TRACING CASCADE VALUE ACCUMULATION")
    logger.info("="*80)
    
    from simulation.simple_intervention import SimpleInterventionSimulator
    from training.network_aware_trainer import NetworkAwareGNNTrainer
    
    # Simulate what happens in intervention loop
    n_nodes = 200  # Realistic size
    n_selected = 5  # 5 nodes selected per round
    
    logger.info(f"\nSetup: {n_nodes} nodes, {n_selected} selected")
    
    # Create network state
    network_state = {
        'demand': torch.rand(n_nodes) * 10 + 5,  # 5-15 kW
        'generation': torch.zeros(n_nodes),
        'congestion': torch.rand(n_nodes) * 0.5,
        'net_demand': torch.rand(n_nodes) * 10 + 5
    }
    
    # Create edge index (each node connected to ~3 neighbors)
    edge_list = []
    for i in range(n_nodes):
        for j in range(max(0, i-2), min(n_nodes, i+3)):
            if i != j:
                edge_list.append([i, j])
    edge_index = torch.tensor(edge_list).t()
    
    logger.info(f"Network: {edge_index.shape[1]} edges")
    
    # Simulate interventions
    simulator = SimpleInterventionSimulator(config={})
    
    cumulative_effects = {
        f'hop_{i}': {
            'energy_impact': torch.zeros(n_nodes),
            'congestion_relief': torch.zeros(n_nodes),
            'economic_value': torch.zeros(n_nodes)
        }
        for i in range(1, 4)
    }
    
    selected_nodes = torch.randperm(n_nodes)[:n_selected].tolist()
    logger.info(f"Selected nodes: {selected_nodes}")
    
    for idx, node_id in enumerate(selected_nodes):
        logger.info(f"\n--- Processing node {node_id} ---")
        
        # Create building features
        building_features = {
            'suitable_roof_area': 50.0,  # 50 m2
            'orientation': 'south',
            'shading': 0.1
        }
        
        # Generate solar
        solar_result = simulator.add_solar(building_features)
        logger.info(f"  Solar capacity: {solar_result['installed_capacity_kwp']:.2f} kWp")
        
        # Create intervention
        intervention = {
            'building_id': node_id,
            'type': 'solar',
            'generation_profile': torch.tensor(solar_result['generation_profile'][:24])
        }
        
        # Calculate cascade
        cascade = simulator.calculate_cascade_effects(
            intervention, network_state, edge_index
        )
        
        # Accumulate
        for hop_key in cascade:
            if hop_key in cumulative_effects:
                for effect_type in cascade[hop_key]:
                    if effect_type in cumulative_effects[hop_key]:
                        cumulative_effects[hop_key][effect_type] += cascade[hop_key][effect_type]
        
        # Log current totals
        hop1_energy = cumulative_effects['hop_1']['energy_impact'].sum().item()
        hop2_energy = cumulative_effects['hop_2']['energy_impact'].sum().item()
        hop3_energy = cumulative_effects['hop_3']['energy_impact'].sum().item()
        
        logger.info(f"  Cumulative after {idx+1} nodes:")
        logger.info(f"    Hop 1: {hop1_energy:.2f} kW")
        logger.info(f"    Hop 2: {hop2_energy:.2f} kW")
        logger.info(f"    Hop 3: {hop3_energy:.2f} kW")
        logger.info(f"    TOTAL: {hop1_energy + hop2_energy + hop3_energy:.2f} kW")
    
    # Final check
    total = sum([
        cumulative_effects[f'hop_{i}']['energy_impact'].sum().item()
        for i in range(1, 4)
    ])
    
    logger.info(f"\n" + "="*40)
    logger.info(f"FINAL TOTAL CASCADE: {total:.2f} kW")
    
    if total > 1000:
        logger.error("ERROR: Cascade value is unrealistically high!")
        
        # Analyze why
        logger.info("\nAnalyzing the issue:")
        
        # Check individual node impacts
        for hop in range(1, 4):
            hop_key = f'hop_{hop}'
            impacts = cumulative_effects[hop_key]['energy_impact']
            max_impact = impacts.max().item()
            mean_impact = impacts.mean().item()
            nonzero = (impacts > 0).sum().item()
            
            logger.info(f"  Hop {hop}:")
            logger.info(f"    Max impact on single node: {max_impact:.2f} kW")
            logger.info(f"    Mean impact: {mean_impact:.2f} kW")
            logger.info(f"    Nodes affected: {nonzero}/{n_nodes}")
    else:
        logger.info("Cascade values seem reasonable")
    
    return total

if __name__ == "__main__":
    total_cascade = trace_cascade_accumulation()
    
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSIS")
    logger.info("="*80)
    
    if total_cascade > 1000:
        logger.info("The issue is likely:")
        logger.info("1. Solar generation profiles are too large (check kWp calculations)")
        logger.info("2. Cascade effects are not properly bounded")
        logger.info("3. Accumulation across multiple interventions compounds the issue")
    else:
        logger.info("No major issues detected in cascade calculation")