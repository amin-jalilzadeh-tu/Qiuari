"""
Test and fix physical parameters
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def test_solar_generation():
    """Test solar generation calculations"""
    from simulation.simple_intervention import SimpleInterventionSimulator
    
    logger.info("="*60)
    logger.info("TESTING SOLAR GENERATION")
    logger.info("="*60)
    
    sim = SimpleInterventionSimulator(config={})
    
    # Typical residential building
    building = {
        'suitable_roof_area': 50.0,  # 50 m² roof
        'orientation': 'south',
        'shading': 0.1
    }
    
    # Generate 24-hour irradiance profile (W/m²)
    hours = np.arange(24)
    irradiance = np.zeros(24)
    for h in range(24):
        if 6 <= h <= 18:  # Sun from 6am to 6pm
            angle = np.pi * (h - 6) / 12
            irradiance[h] = 1000 * np.sin(angle)  # Peak 1000 W/m²
    
    logger.info(f"\nIrradiance profile (W/m²):")
    logger.info(f"  6am-9am: {irradiance[6:9]}")
    logger.info(f"  12pm: {irradiance[12]:.0f}")
    logger.info(f"  6pm: {irradiance[18]:.0f}")
    logger.info(f"  Daily total: {irradiance.sum():.0f} Wh/m²")
    
    # Test solar generation
    result = sim.add_solar(building, time_series=irradiance)
    
    logger.info(f"\nSolar system specs:")
    logger.info(f"  Roof area: {building['suitable_roof_area']} m²")
    logger.info(f"  Installed capacity: {result['installed_capacity_kwp']:.2f} kWp")
    logger.info(f"  Peak generation: {result['peak_generation_kw']:.2f} kW")
    logger.info(f"  Annual generation: {result['annual_generation_kwh']:.2f} kWh")
    
    # Check if realistic
    issues = []
    if result['installed_capacity_kwp'] > 10:
        issues.append(f"Capacity too high for residential: {result['installed_capacity_kwp']:.2f} kWp")
    
    if result['peak_generation_kw'] > result['installed_capacity_kwp']:
        issues.append(f"Peak generation exceeds capacity: {result['peak_generation_kw']:.2f} > {result['installed_capacity_kwp']:.2f}")
    
    # Expected annual: capacity * 1000-1500 hours (depending on location)
    expected_annual_min = result['installed_capacity_kwp'] * 900
    expected_annual_max = result['installed_capacity_kwp'] * 1500
    
    if not (expected_annual_min <= result['annual_generation_kwh'] <= expected_annual_max):
        issues.append(f"Annual generation unrealistic: {result['annual_generation_kwh']:.0f} kWh")
    
    if issues:
        logger.error("ISSUES FOUND:")
        for issue in issues:
            logger.error(f"  - {issue}")
    else:
        logger.info("  OK: All values realistic")
    
    return result

def test_cascade_calculation():
    """Test cascade effect calculations"""
    from simulation.simple_intervention import SimpleInterventionSimulator
    
    logger.info("\n" + "="*60)
    logger.info("TESTING CASCADE CALCULATIONS")
    logger.info("="*60)
    
    sim = SimpleInterventionSimulator(config={})
    
    # Simple 3-node network
    n_nodes = 3
    network_state = {
        'demand': torch.tensor([2.0, 8.0, 6.0]),  # kW - node 0 has lower demand for surplus
        'generation': torch.tensor([0.0, 0.0, 0.0]),  # kW
        'congestion': torch.tensor([0.2, 0.3, 0.1]),
        'net_demand': torch.tensor([2.0, 8.0, 6.0])
    }
    
    # Linear network: 0-1-2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    
    # Add 5 kW solar to node 0 (more than its 2 kW demand = 3 kW surplus)
    intervention = {
        'building_id': 0,
        'type': 'solar',
        'generation_profile': torch.tensor([5.0] * 24)  # Constant 5 kW
    }
    
    logger.info("\nNetwork state:")
    logger.info(f"  Demand: {network_state['demand'].tolist()} kW")
    logger.info(f"  Solar on node 0: 5.0 kW")
    
    # Calculate cascade
    cascade = sim.calculate_cascade_effects(intervention, network_state, edge_index)
    
    logger.info("\nCascade effects:")
    for hop in range(1, 4):
        hop_key = f'hop_{hop}'
        if hop_key in cascade:
            energy = cascade[hop_key]['energy_impact']
            total = energy.sum().item()
            logger.info(f"  Hop {hop}: {energy.tolist()} kW (total: {total:.2f} kW)")
    
    # Check realism
    total_cascade = sum([
        cascade[f'hop_{h}']['energy_impact'].sum().item() 
        for h in range(1, 4) if f'hop_{h}' in cascade
    ])
    
    logger.info(f"\nTotal cascade energy: {total_cascade:.2f} kW")
    
    # Check that cascade doesn't exceed surplus (generation - local demand)
    local_demand = network_state['demand'][0].item()
    generation = intervention['generation_profile'][0].item()
    max_shareable = generation - local_demand
    
    logger.info(f"  Generation: {generation:.2f} kW")
    logger.info(f"  Local demand: {local_demand:.2f} kW")
    logger.info(f"  Max shareable: {max_shareable:.2f} kW")
    
    if total_cascade > max_shareable + 0.1:  # Small tolerance for rounding
        logger.error(f"ERROR: Cascade ({total_cascade:.2f}) exceeds shareable energy ({max_shareable:.2f})!")
    else:
        logger.info("  OK: Cascade within shareable energy limits")

def test_peak_reduction():
    """Test peak reduction calculation"""
    logger.info("\n" + "="*60)
    logger.info("TESTING PEAK REDUCTION")
    logger.info("="*60)
    
    # Morning peak scenario
    demand = torch.tensor([
        [3, 4, 5, 6, 8, 10, 12, 11, 9, 8, 7, 6, 5, 5, 6, 7, 9, 11, 10, 8, 6, 5, 4, 3],  # Node 0
        [2, 3, 4, 5, 7, 9, 11, 10, 8, 7, 6, 5, 4, 4, 5, 6, 8, 10, 9, 7, 5, 4, 3, 2],    # Node 1
        [4, 5, 6, 7, 9, 11, 13, 12, 10, 9, 8, 7, 6, 6, 7, 8, 10, 12, 11, 9, 7, 6, 5, 4]  # Node 2
    ], dtype=torch.float32)
    
    # Solar generation (peaks at noon)
    solar = torch.zeros_like(demand)
    for h in range(24):
        if 6 <= h <= 18:
            solar[:, h] = 5 * np.sin(np.pi * (h - 6) / 12)  # Max 5 kW at noon
    
    # Calculate peaks
    old_peak = demand.sum(dim=0).max().item()
    net_demand = demand - solar
    new_peak = net_demand.sum(dim=0).max().item()
    
    reduction_pct = (old_peak - new_peak) / old_peak * 100
    
    logger.info(f"\nDemand profile:")
    logger.info(f"  Total demand peak: {old_peak:.1f} kW at hour {demand.sum(dim=0).argmax().item()}")
    logger.info(f"  Solar peak: {solar.sum(dim=0).max().item():.1f} kW at hour {solar.sum(dim=0).argmax().item()}")
    logger.info(f"  New peak: {new_peak:.1f} kW")
    logger.info(f"  Peak reduction: {reduction_pct:.1f}%")
    
    if reduction_pct < 5:
        logger.warning("WARNING: Peak reduction too low - timing mismatch?")
    else:
        logger.info("  OK: Significant peak reduction achieved")

if __name__ == "__main__":
    # Run all tests
    solar_result = test_solar_generation()
    test_cascade_calculation()
    test_peak_reduction()
    
    logger.info("\n" + "="*60)
    logger.info("PHYSICAL PARAMETER ANALYSIS")
    logger.info("="*60)
    
    logger.info("\nKey findings:")
    logger.info("1. Solar capacity should be ~6 m²/kWp, max 10 kWp residential")
    logger.info("2. Peak generation should not exceed installed capacity")
    logger.info("3. Cascade energy cannot exceed available generation")
    logger.info("4. Peak reduction depends on demand/generation timing alignment")
    logger.info("5. Units must be consistent: kW for power, kWh for energy")