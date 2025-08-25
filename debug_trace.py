"""
Debug script to trace the entire flow with dummy data
"""
import torch
import numpy as np
import logging
from torch_geometric.data import Data

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

def create_dummy_data(n_nodes=10, n_features=17, n_timesteps=24):
    """Create dummy data for testing"""
    logger.info("="*60)
    logger.info("CREATING DUMMY DATA")
    logger.info("="*60)
    
    # Node features (buildings)
    x = torch.randn(n_nodes, n_features)
    x[:, 0] = torch.randint(0, 7, (n_nodes,)).float()  # Energy labels (A-G)
    x[:, 1] = torch.rand(n_nodes) * 0.2 + 0.05  # Area (normalized)
    x[:, 2] = torch.rand(n_nodes) * 0.1 + 0.02  # Roof area (normalized)
    
    # Create a simple network (each node connected to 2-3 neighbors)
    edge_list = []
    for i in range(n_nodes):
        # Connect to next 2-3 nodes (with wraparound)
        for j in range(1, min(4, n_nodes)):
            neighbor = (i + j) % n_nodes
            edge_list.append([i, neighbor])
            edge_list.append([neighbor, i])  # Bidirectional
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    # Temporal profiles (energy demand over 24 hours)
    profiles = torch.zeros(n_nodes, n_timesteps)
    for i in range(n_nodes):
        # Create diverse profiles
        base = torch.randn(n_timesteps) * 0.5 + 1.0
        base[7:9] += 0.5   # Morning peak
        base[17:20] += 1.0  # Evening peak
        base[0:6] *= 0.3    # Night low
        profiles[i] = torch.abs(base)
    
    data = Data(x=x, edge_index=edge_index, temporal_profiles=profiles)
    data.centrality_features = torch.rand(n_nodes, 5)
    
    logger.info(f"Created {n_nodes} nodes with {n_features} features")
    logger.info(f"Edge connections: {edge_index.shape[1]//2} bidirectional edges")
    logger.info(f"Temporal profiles shape: {profiles.shape}")
    
    return data

def trace_loss_calculation():
    """Trace loss calculation with dummy data"""
    from training.loss_functions import ComplementarityLoss
    
    logger.info("\n" + "="*60)
    logger.info("TRACING LOSS CALCULATION")
    logger.info("="*60)
    
    # Create dummy inputs
    n_nodes = 10
    n_clusters = 3
    n_features = 32
    n_timesteps = 24
    
    embeddings = torch.randn(n_nodes, n_features)
    cluster_probs = torch.softmax(torch.randn(n_nodes, n_clusters), dim=1)
    temporal_profiles = torch.randn(n_nodes, n_timesteps)
    
    # Make some profiles complementary (negative correlation)
    temporal_profiles[0] = torch.sin(torch.linspace(0, 2*np.pi, n_timesteps))
    temporal_profiles[1] = -temporal_profiles[0]  # Perfect negative correlation
    temporal_profiles[2] = temporal_profiles[0] * 0.8  # High positive correlation
    
    # Set cluster assignments to test
    cluster_probs = torch.zeros(n_nodes, n_clusters)
    cluster_probs[0, 0] = 1.0  # Node 0 in cluster 0
    cluster_probs[1, 0] = 1.0  # Node 1 in cluster 0 (should be complementary)
    cluster_probs[2, 0] = 1.0  # Node 2 in cluster 0 (not complementary)
    cluster_probs[3:, 1] = 1.0  # Other nodes in cluster 1
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Cluster probs shape: {cluster_probs.shape}")
    logger.info(f"Temporal profiles shape: {temporal_profiles.shape}")
    
    # Calculate loss
    loss_fn = ComplementarityLoss()
    total_loss, components = loss_fn(embeddings, cluster_probs, temporal_profiles)
    
    logger.info(f"\nLoss Components:")
    logger.info(f"  Total loss: {total_loss.item():.4f}")
    for key, value in components.items():
        if hasattr(value, 'item'):
            logger.info(f"  {key}: {value.item():.4f}")
        else:
            logger.info(f"  {key}: {value}")
    
    # Check correlation manually
    profiles_norm = (temporal_profiles - temporal_profiles.mean(dim=1, keepdim=True)) / (
        temporal_profiles.std(dim=1, keepdim=True) + 1e-8
    )
    corr_matrix = torch.matmul(profiles_norm, profiles_norm.t()) / n_timesteps
    
    logger.info(f"\nCorrelation between nodes 0 and 1: {corr_matrix[0, 1].item():.4f} (should be ~-1)")
    logger.info(f"Correlation between nodes 0 and 2: {corr_matrix[0, 2].item():.4f} (should be ~0.8)")
    
    return total_loss.item()

def trace_intervention_flow():
    """Trace the intervention simulation flow"""
    from simulation.simple_intervention import SimpleInterventionSimulator
    
    logger.info("\n" + "="*60)
    logger.info("TRACING INTERVENTION FLOW")
    logger.info("="*60)
    
    # Create simulator with config
    simulator = SimpleInterventionSimulator(config={})
    
    # Create dummy building features
    building_features = {
        'suitable_roof_area': 50.0,  # 50 m²
        'orientation': 'south',
        'shading': 0.1,
        'energy_label': 'C'
    }
    
    # Generate solar profile
    logger.info("\n1. SOLAR GENERATION:")
    solar_result = simulator.add_solar(building_features)
    logger.info(f"  Installed capacity: {solar_result['installed_capacity_kwp']:.2f} kWp")
    logger.info(f"  Annual generation: {solar_result['annual_generation_kwh']:.2f} kWh")
    logger.info(f"  Peak generation: {solar_result['peak_generation_kw']:.2f} kW")
    
    # Create dummy network state
    n_nodes = 10
    device = torch.device('cpu')
    network_state = {
        'demand': torch.rand(n_nodes) * 10 + 5,  # 5-15 kW demand
        'generation': torch.rand(n_nodes) * 2,    # 0-2 kW existing generation
        'congestion': torch.rand(n_nodes) * 0.5,  # 0-50% congestion
        'net_demand': torch.rand(n_nodes) * 8 + 3  # Net demand
    }
    
    # Create simple edge index
    edge_list = []
    for i in range(n_nodes):
        if i < n_nodes - 1:
            edge_list.extend([[i, i+1], [i+1, i]])
    edge_index = torch.tensor(edge_list).t()
    
    # Simulate intervention on node 0
    intervention = {
        'building_id': 0,
        'type': 'solar',
        'generation_profile': torch.tensor(solar_result['generation_profile'][:24])
    }
    
    logger.info("\n2. CASCADE EFFECTS:")
    cascade_effects = simulator.calculate_cascade_effects(
        intervention, network_state, edge_index, max_hops=3
    )
    
    for hop in range(1, 4):
        hop_key = f'hop_{hop}'
        if hop_key in cascade_effects:
            energy = cascade_effects[hop_key]['energy_impact'].sum().item()
            congestion = cascade_effects[hop_key]['congestion_relief'].sum().item()
            economic = cascade_effects[hop_key]['economic_value'].sum().item()
            logger.info(f"  {hop_key}: energy={energy:.2f}, congestion={congestion:.2f}, economic={economic:.2f}")
    
    # Update network state
    logger.info("\n3. NETWORK STATE UPDATE:")
    old_peak = network_state['net_demand'].max().item()
    logger.info(f"  Old peak demand: {old_peak:.2f} kW")
    
    new_state = simulator.update_network_state(network_state, cascade_effects)
    new_peak = new_state['net_demand'].max().item()
    logger.info(f"  New peak demand: {new_peak:.2f} kW")
    logger.info(f"  Peak reduction: {(old_peak - new_peak) / old_peak * 100:.2f}%")
    
    # Check actual generation change
    gen_change = (new_state['generation'] - network_state['generation']).sum().item()
    logger.info(f"  Total generation added: {gen_change:.2f} kW")
    
    return cascade_effects

def trace_full_pipeline():
    """Trace the complete pipeline"""
    logger.info("\n" + "="*60)
    logger.info("TRACING FULL PIPELINE")
    logger.info("="*60)
    
    # Create dummy data
    data = create_dummy_data(n_nodes=20)
    
    # Trace loss
    loss = trace_loss_calculation()
    
    # Trace intervention
    cascade = trace_intervention_flow()
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Final loss value: {loss:.4f}")
    logger.info(f"Cascade effects created: {len(cascade)} hops")
    
    # Check for issues
    issues = []
    if loss < 0:
        issues.append("Loss is negative!")
    
    total_cascade_energy = sum([
        cascade[f'hop_{i}']['energy_impact'].sum().item() 
        for i in range(1, 4) if f'hop_{i}' in cascade
    ])
    if total_cascade_energy > 1000:
        issues.append(f"Cascade energy unrealistic: {total_cascade_energy:.2f} kW")
    
    if issues:
        logger.warning("\nISSUES FOUND:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("\nNo major issues detected")

if __name__ == "__main__":
    trace_full_pipeline()