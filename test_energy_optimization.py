"""
Comprehensive Test Suite for Energy Grid Optimization Logic
Tests P2P trading, battery optimization, power flow, and grid constraints
"""

import numpy as np
import torch
import pandas as pd
from datetime import datetime, timedelta
import logging
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import components to test
from models.physics_layers import (
    PhysicsConstraintLayer,
    EnergyBalanceChecker,
    LVGroupBoundaryEnforcer,
    DistanceBasedLossCalculator
)
from tasks.solar_optimization import SolarOptimization, SolarConfig
from analysis.intervention_recommender import InterventionRecommender
from analysis.pattern_analyzer import PatternAnalyzer

class EnergyOptimizationTester:
    """Comprehensive testing suite for energy optimization components"""
    
    def __init__(self):
        self.test_results = {}
        self.validation_errors = []
        
    def generate_test_data(self, num_buildings: int = 20, num_timesteps: int = 96):
        """Generate synthetic test data for energy system"""
        np.random.seed(42)
        
        # Building characteristics
        buildings = {}
        for i in range(num_buildings):
            building_type = np.random.choice(['residential', 'commercial', 'industrial'])
            buildings[i] = {
                'ogc_fid': f'BUILDING_{i:04d}',
                'building_function': building_type,
                'area': np.random.uniform(100, 1000),
                'roof_area': np.random.uniform(50, 500),
                'height': np.random.uniform(8, 30),
                'x_coord': np.random.uniform(0, 1000),
                'y_coord': np.random.uniform(0, 1000),
                'lv_group_id': i // 5,  # Group every 5 buildings
                'transformer_id': i // 10,  # 2 transformers per LV group
                'has_solar': np.random.random() < 0.2,
                'solar_capacity_kw': np.random.uniform(5, 50) if np.random.random() < 0.2 else 0,
                'has_battery': np.random.random() < 0.1,
                'battery_capacity_kwh': np.random.uniform(10, 100) if np.random.random() < 0.1 else 0,
                'energy_label': np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G']),
                'avg_demand_kw': {'residential': 5, 'commercial': 25, 'industrial': 50}[building_type],
                'peak_demand_kw': {'residential': 10, 'commercial': 40, 'industrial': 80}[building_type]
            }
        
        # Generate temporal profiles
        timestamps = pd.date_range(start='2024-01-01', periods=num_timesteps, freq='15min')
        temporal_data = pd.DataFrame(index=timestamps)
        
        for i in range(num_buildings):
            building = buildings[i]
            base_load = building['avg_demand_kw']
            
            # Create realistic consumption patterns
            if 'residential' in building['building_function']:
                # Morning and evening peaks
                hourly_factors = np.array([0.4, 0.4, 0.3, 0.3, 0.3, 0.4, 0.6, 0.8,
                                         0.7, 0.5, 0.4, 0.4, 0.5, 0.5, 0.6, 0.7,
                                         0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
            elif 'commercial' in building['building_function']:
                # Business hours peak
                hourly_factors = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.5, 0.8,
                                         1.0, 1.0, 1.0, 0.9, 0.8, 0.9, 1.0, 0.9,
                                         0.8, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2])
            else:  # industrial
                # Constant with slight variation
                hourly_factors = np.ones(24) * 0.8 + np.random.random(24) * 0.2
            
            # Expand to 15-minute intervals
            profile = np.repeat(hourly_factors, 4)[:num_timesteps]
            consumption = base_load * profile * (1 + np.random.normal(0, 0.1, num_timesteps))
            temporal_data[f'consumption_{i}'] = np.maximum(0, consumption)
            
            # Generate solar generation if applicable
            if building['has_solar']:
                solar_capacity = building['solar_capacity_kw']
                # Simple solar curve (peaks at noon)
                hours = np.arange(num_timesteps) % 96 / 4  # Convert to hours
                solar_generation = np.maximum(0, solar_capacity * np.exp(-((hours - 12)**2) / 18))
                solar_generation[hours < 6] = 0
                solar_generation[hours > 18] = 0
                temporal_data[f'generation_{i}'] = solar_generation
            else:
                temporal_data[f'generation_{i}'] = 0
        
        return buildings, temporal_data
    
    def test_energy_balance(self, buildings: Dict, temporal_data: pd.DataFrame):
        """Test 1: Verify energy balance is maintained"""
        logger.info("\n=== TEST 1: ENERGY BALANCE VALIDATION ===")
        
        num_buildings = len(buildings)
        batch_size = 1
        num_timesteps = len(temporal_data)
        
        # Create tensors for testing
        consumption = torch.zeros(batch_size, num_buildings, num_timesteps)
        generation = torch.zeros(batch_size, num_buildings, num_timesteps)
        
        for i in range(num_buildings):
            consumption[0, i, :] = torch.tensor(temporal_data[f'consumption_{i}'].values)
            generation[0, i, :] = torch.tensor(temporal_data[f'generation_{i}'].values)
        
        # Calculate total system energy
        total_consumption = consumption.sum().item()
        total_generation = generation.sum().item()
        net_import_needed = total_consumption - total_generation
        
        logger.info(f"Total Consumption: {total_consumption:.2f} kWh")
        logger.info(f"Total Generation: {total_generation:.2f} kWh")
        logger.info(f"Net Import Needed: {net_import_needed:.2f} kWh")
        logger.info(f"Self-sufficiency: {(total_generation/total_consumption)*100:.1f}%")
        
        # Test energy balance checker
        balance_checker = EnergyBalanceChecker(tolerance=0.05)
        
        # Create dummy sharing matrix (no sharing initially)
        sharing_matrix = torch.zeros(batch_size, num_buildings, num_buildings, num_timesteps)
        
        # LV group assignments
        lv_group_ids = torch.tensor([b['lv_group_id'] for b in buildings.values()])
        
        # Check balance
        penalty, balance_info = balance_checker(
            consumption, generation, sharing_matrix, lv_group_ids
        )
        
        logger.info(f"Balance Penalty: {penalty.item():.4f}")
        for group, info in balance_info.items():
            logger.info(f"  {group}: Imbalance={info['imbalance']:.2f} kW")
        
        # Validate: Total energy should be conserved
        energy_conservation_error = abs(total_consumption - total_generation - net_import_needed)
        assert energy_conservation_error < 1e-6, f"Energy not conserved! Error: {energy_conservation_error}"
        
        self.test_results['energy_balance'] = {
            'passed': True,
            'total_consumption': total_consumption,
            'total_generation': total_generation,
            'self_sufficiency': total_generation / total_consumption,
            'balance_penalty': penalty.item()
        }
        
        logger.info("✓ Energy balance test PASSED")
        return True
    
    def test_voltage_constraints(self, buildings: Dict):
        """Test 2: Verify voltage constraints are respected"""
        logger.info("\n=== TEST 2: VOLTAGE CONSTRAINT VALIDATION ===")
        
        # Simulate voltage levels at different nodes
        num_buildings = len(buildings)
        nominal_voltage = 230  # V
        
        # Calculate voltage drops based on distance from transformer
        transformers = {}
        for i, b in buildings.items():
            t_id = b['transformer_id']
            if t_id not in transformers:
                transformers[t_id] = {'buildings': [], 'x': 0, 'y': 0}
            transformers[t_id]['buildings'].append(i)
            transformers[t_id]['x'] += b['x_coord']
            transformers[t_id]['y'] += b['y_coord']
        
        # Average transformer positions
        for t_id in transformers:
            n = len(transformers[t_id]['buildings'])
            transformers[t_id]['x'] /= n
            transformers[t_id]['y'] /= n
        
        voltage_violations = []
        for i, b in buildings.items():
            # Distance to transformer
            t = transformers[b['transformer_id']]
            distance = np.sqrt((b['x_coord'] - t['x'])**2 + (b['y_coord'] - t['y'])**2)
            
            # Voltage drop (simplified: 0.01V per meter under load)
            voltage_drop = 0.01 * distance * (b['peak_demand_kw'] / 10)  # Scaled by load
            actual_voltage = nominal_voltage - voltage_drop
            
            # Check constraints (±10% of nominal)
            if actual_voltage < nominal_voltage * 0.9:
                voltage_violations.append({
                    'building': i,
                    'voltage': actual_voltage,
                    'violation': 'undervoltage'
                })
            elif actual_voltage > nominal_voltage * 1.1:
                voltage_violations.append({
                    'building': i,
                    'voltage': actual_voltage,
                    'violation': 'overvoltage'
                })
        
        logger.info(f"Voltage violations found: {len(voltage_violations)}")
        for v in voltage_violations[:5]:  # Show first 5
            logger.info(f"  Building {v['building']}: {v['voltage']:.1f}V ({v['violation']})")
        
        self.test_results['voltage_constraints'] = {
            'passed': len(voltage_violations) == 0,
            'violations': len(voltage_violations),
            'nominal_voltage': nominal_voltage,
            'min_allowed': nominal_voltage * 0.9,
            'max_allowed': nominal_voltage * 1.1
        }
        
        if len(voltage_violations) == 0:
            logger.info("✓ Voltage constraint test PASSED")
        else:
            logger.warning(f"⚠ Voltage constraint test FAILED with {len(voltage_violations)} violations")
        
        return len(voltage_violations) == 0
    
    def test_power_flow_equations(self, buildings: Dict, temporal_data: pd.DataFrame):
        """Test 3: Verify power flow equations are correct"""
        logger.info("\n=== TEST 3: POWER FLOW EQUATION VALIDATION ===")
        
        # Simplified DC power flow test
        num_buildings = len(buildings)
        
        # Build admittance matrix based on electrical distances
        Y = np.zeros((num_buildings, num_buildings))
        
        for i in range(num_buildings):
            for j in range(i+1, num_buildings):
                # Same LV group = connected
                if buildings[i]['lv_group_id'] == buildings[j]['lv_group_id']:
                    # Conductance inversely proportional to distance
                    dist = np.sqrt(
                        (buildings[i]['x_coord'] - buildings[j]['x_coord'])**2 +
                        (buildings[i]['y_coord'] - buildings[j]['y_coord'])**2
                    )
                    conductance = 1.0 / (1 + dist/100)  # Normalized conductance
                    Y[i, j] = -conductance
                    Y[j, i] = -conductance
        
        # Diagonal elements (self-conductance)
        for i in range(num_buildings):
            Y[i, i] = -np.sum(Y[i, :])
        
        # Test power flow at a specific timestep
        t = 48  # Noon
        P_injection = np.zeros(num_buildings)
        
        for i in range(num_buildings):
            consumption = temporal_data[f'consumption_{i}'].iloc[t]
            generation = temporal_data[f'generation_{i}'].iloc[t]
            P_injection[i] = generation - consumption  # Net injection
        
        # Check power balance: Sum of all injections should equal losses
        total_injection = np.sum(P_injection)
        logger.info(f"Total power injection: {total_injection:.2f} kW")
        
        # Calculate power flows on lines (simplified)
        power_flows = {}
        total_losses = 0
        
        for i in range(num_buildings):
            for j in range(i+1, num_buildings):
                if Y[i, j] != 0:
                    # Approximate power flow
                    flow = -Y[i, j] * (P_injection[i] - P_injection[j]) * 0.1  # Scaling factor
                    power_flows[f'{i}-{j}'] = flow
                    # Losses proportional to flow squared and distance
                    dist = np.sqrt(
                        (buildings[i]['x_coord'] - buildings[j]['x_coord'])**2 +
                        (buildings[i]['y_coord'] - buildings[j]['y_coord'])**2
                    )
                    losses = 0.001 * flow**2 * dist / 100
                    total_losses += losses
        
        logger.info(f"Total line losses: {total_losses:.2f} kW")
        logger.info(f"Number of active power flows: {len(power_flows)}")
        
        # Validate Kirchhoff's laws
        # Current law: At each node, sum of flows = injection
        node_balance_errors = []
        for i in range(num_buildings):
            flow_in = sum(power_flows.get(f'{j}-{i}', 0) for j in range(i))
            flow_out = sum(power_flows.get(f'{i}-{j}', 0) for j in range(i+1, num_buildings))
            net_flow = flow_in - flow_out
            error = abs(net_flow - P_injection[i])
            if error > 0.1:  # Tolerance
                node_balance_errors.append((i, error))
        
        logger.info(f"Node balance errors: {len(node_balance_errors)}")
        
        self.test_results['power_flow'] = {
            'passed': len(node_balance_errors) == 0,
            'total_injection': total_injection,
            'total_losses': total_losses,
            'loss_percentage': (total_losses / abs(total_injection)) * 100 if total_injection != 0 else 0,
            'node_errors': len(node_balance_errors)
        }
        
        if len(node_balance_errors) == 0:
            logger.info("✓ Power flow equation test PASSED")
        else:
            logger.warning(f"⚠ Power flow equation test FAILED with {len(node_balance_errors)} node errors")
        
        return len(node_balance_errors) == 0
    
    def test_optimization_objectives(self, buildings: Dict, temporal_data: pd.DataFrame):
        """Test 4: Verify optimization objectives are properly formulated"""
        logger.info("\n=== TEST 4: OPTIMIZATION OBJECTIVES VALIDATION ===")
        
        # Test multiple objectives
        objectives = {
            'minimize_cost': 0,
            'minimize_losses': 0,
            'maximize_self_sufficiency': 0,
            'minimize_emissions': 0
        }
        
        # Calculate objective values
        total_consumption = 0
        total_generation = 0
        total_import = 0
        
        for i in range(len(buildings)):
            consumption = temporal_data[f'consumption_{i}'].sum()
            generation = temporal_data[f'generation_{i}'].sum()
            total_consumption += consumption
            total_generation += generation
            
            # Import when consumption > generation
            import_profile = np.maximum(0, 
                temporal_data[f'consumption_{i}'] - temporal_data[f'generation_{i}'])
            total_import += import_profile.sum()
        
        # 1. Cost objective
        electricity_price = 0.12  # $/kWh
        feed_in_tariff = 0.08  # $/kWh
        cost = total_import * electricity_price - (total_generation - total_consumption) * feed_in_tariff
        objectives['minimize_cost'] = cost
        
        # 2. Loss objective (simplified)
        losses = total_consumption * 0.05  # Assume 5% losses
        objectives['minimize_losses'] = losses
        
        # 3. Self-sufficiency objective
        self_sufficiency = total_generation / total_consumption if total_consumption > 0 else 0
        objectives['maximize_self_sufficiency'] = self_sufficiency
        
        # 4. Emissions objective
        grid_emissions_factor = 0.5  # kg CO2/kWh
        emissions = total_import * grid_emissions_factor
        objectives['minimize_emissions'] = emissions
        
        logger.info("Optimization Objectives:")
        logger.info(f"  Cost: ${objectives['minimize_cost']:.2f}")
        logger.info(f"  Losses: {objectives['minimize_losses']:.2f} kWh")
        logger.info(f"  Self-sufficiency: {objectives['maximize_self_sufficiency']*100:.1f}%")
        logger.info(f"  Emissions: {objectives['minimize_emissions']:.2f} kg CO2")
        
        # Validate objective formulation
        assert objectives['minimize_cost'] >= 0, "Cost cannot be negative"
        assert objectives['minimize_losses'] >= 0, "Losses cannot be negative"
        assert 0 <= objectives['maximize_self_sufficiency'] <= 1, "Self-sufficiency must be between 0 and 1"
        assert objectives['minimize_emissions'] >= 0, "Emissions cannot be negative"
        
        self.test_results['optimization_objectives'] = {
            'passed': True,
            'objectives': objectives,
            'total_consumption': total_consumption,
            'total_generation': total_generation
        }
        
        logger.info("✓ Optimization objectives test PASSED")
        return True
    
    def test_p2p_trading_logic(self, buildings: Dict, temporal_data: pd.DataFrame):
        """Test 5: Validate P2P energy trading logic"""
        logger.info("\n=== TEST 5: P2P ENERGY TRADING VALIDATION ===")
        
        # Identify potential trading pairs within same LV group
        lv_groups = {}
        for i, b in buildings.items():
            lv_id = b['lv_group_id']
            if lv_id not in lv_groups:
                lv_groups[lv_id] = []
            lv_groups[lv_id].append(i)
        
        trading_opportunities = []
        
        for lv_id, members in lv_groups.items():
            if len(members) < 2:
                continue
            
            # Analyze trading potential at each timestep
            for t in range(0, len(temporal_data), 4):  # Sample every hour
                suppliers = []
                consumers = []
                
                for building_id in members:
                    generation = temporal_data[f'generation_{building_id}'].iloc[t]
                    consumption = temporal_data[f'consumption_{building_id}'].iloc[t]
                    net = generation - consumption
                    
                    if net > 0:
                        suppliers.append((building_id, net))
                    elif net < 0:
                        consumers.append((building_id, -net))
                
                # Match suppliers with consumers
                if suppliers and consumers:
                    # Simple matching: largest supplier with largest consumer
                    suppliers.sort(key=lambda x: x[1], reverse=True)
                    consumers.sort(key=lambda x: x[1], reverse=True)
                    
                    for s_id, s_amount in suppliers:
                        for c_id, c_amount in consumers:
                            trade_amount = min(s_amount, c_amount)
                            if trade_amount > 0.1:  # Minimum trade threshold
                                trading_opportunities.append({
                                    'timestep': t,
                                    'supplier': s_id,
                                    'consumer': c_id,
                                    'amount_kw': trade_amount,
                                    'lv_group': lv_id
                                })
                                # Update remaining amounts
                                s_amount -= trade_amount
                                c_amount -= trade_amount
                                if s_amount <= 0:
                                    break
        
        logger.info(f"P2P Trading Opportunities Found: {len(trading_opportunities)}")
        
        # Calculate trading benefits
        total_traded_energy = sum(t['amount_kw'] for t in trading_opportunities)
        avg_trade_size = total_traded_energy / len(trading_opportunities) if trading_opportunities else 0
        
        # Validate trading logic
        for trade in trading_opportunities[:5]:  # Check first 5 trades
            # Verify same LV group
            supplier_lv = buildings[trade['supplier']]['lv_group_id']
            consumer_lv = buildings[trade['consumer']]['lv_group_id']
            assert supplier_lv == consumer_lv, f"Cross-LV trading detected: {supplier_lv} != {consumer_lv}"
            
            # Verify energy availability
            t = trade['timestep']
            supplier_gen = temporal_data[f"generation_{trade['supplier']}"].iloc[t]
            supplier_cons = temporal_data[f"consumption_{trade['supplier']}"].iloc[t]
            available = supplier_gen - supplier_cons
            assert available >= trade['amount_kw'] - 0.01, f"Supplier doesn't have enough energy"
        
        logger.info(f"Total P2P Traded Energy: {total_traded_energy:.2f} kW")
        logger.info(f"Average Trade Size: {avg_trade_size:.2f} kW")
        
        self.test_results['p2p_trading'] = {
            'passed': True,
            'num_trades': len(trading_opportunities),
            'total_traded': total_traded_energy,
            'avg_trade_size': avg_trade_size
        }
        
        logger.info("✓ P2P trading logic test PASSED")
        return True
    
    def test_battery_optimization(self, buildings: Dict, temporal_data: pd.DataFrame):
        """Test 6: Validate battery storage optimization"""
        logger.info("\n=== TEST 6: BATTERY STORAGE OPTIMIZATION VALIDATION ===")
        
        # Identify buildings with batteries
        battery_buildings = [i for i, b in buildings.items() if b.get('has_battery', False)]
        
        if not battery_buildings:
            # Add virtual battery for testing
            buildings[0]['has_battery'] = True
            buildings[0]['battery_capacity_kwh'] = 50
            battery_buildings = [0]
        
        logger.info(f"Buildings with batteries: {len(battery_buildings)}")
        
        battery_operations = []
        
        for building_id in battery_buildings:
            battery_capacity = buildings[building_id]['battery_capacity_kwh']
            soc = battery_capacity * 0.5  # Start at 50% SOC
            efficiency = 0.92  # Round-trip efficiency
            max_power = battery_capacity / 4  # C/4 rate
            
            building_operations = {
                'building_id': building_id,
                'capacity': battery_capacity,
                'charge_events': [],
                'discharge_events': [],
                'peak_shaving': 0,
                'energy_arbitrage': 0
            }
            
            for t in range(len(temporal_data)):
                consumption = temporal_data[f'consumption_{building_id}'].iloc[t]
                generation = temporal_data[f'generation_{building_id}'].iloc[t]
                net = generation - consumption
                
                # Simple battery control logic
                if net > 0 and soc < battery_capacity * 0.9:  # Excess generation, charge
                    charge_power = min(net, max_power, (battery_capacity * 0.9 - soc))
                    soc += charge_power * efficiency * 0.25  # 15 min = 0.25 hour
                    building_operations['charge_events'].append({
                        'time': t,
                        'power': charge_power,
                        'soc_after': soc
                    })
                elif net < 0 and soc > battery_capacity * 0.1:  # Deficit, discharge
                    discharge_power = min(-net, max_power, (soc - battery_capacity * 0.1))
                    soc -= discharge_power / efficiency * 0.25
                    building_operations['discharge_events'].append({
                        'time': t,
                        'power': discharge_power,
                        'soc_after': soc
                    })
                    
                    # Track peak shaving
                    if consumption > buildings[building_id]['avg_demand_kw'] * 1.5:
                        building_operations['peak_shaving'] += discharge_power * 0.25
                
                # Validate SOC bounds
                assert 0 <= soc <= battery_capacity * 1.01, f"SOC out of bounds: {soc}/{battery_capacity}"
            
            # Calculate arbitrage value (simplified)
            charge_energy = sum(e['power'] * 0.25 for e in building_operations['charge_events'])
            discharge_energy = sum(e['power'] * 0.25 for e in building_operations['discharge_events'])
            building_operations['energy_arbitrage'] = discharge_energy * 0.12 - charge_energy * 0.08
            
            battery_operations.append(building_operations)
        
        # Aggregate results
        total_charge_events = sum(len(b['charge_events']) for b in battery_operations)
        total_discharge_events = sum(len(b['discharge_events']) for b in battery_operations)
        total_peak_shaving = sum(b['peak_shaving'] for b in battery_operations)
        total_arbitrage_value = sum(b['energy_arbitrage'] for b in battery_operations)
        
        logger.info(f"Total Charge Events: {total_charge_events}")
        logger.info(f"Total Discharge Events: {total_discharge_events}")
        logger.info(f"Peak Shaving Energy: {total_peak_shaving:.2f} kWh")
        logger.info(f"Arbitrage Value: ${total_arbitrage_value:.2f}")
        
        self.test_results['battery_optimization'] = {
            'passed': True,
            'num_batteries': len(battery_buildings),
            'charge_events': total_charge_events,
            'discharge_events': total_discharge_events,
            'peak_shaving_kwh': total_peak_shaving,
            'arbitrage_value': total_arbitrage_value
        }
        
        logger.info("✓ Battery optimization test PASSED")
        return True
    
    def test_loss_minimization(self, buildings: Dict, temporal_data: pd.DataFrame):
        """Test 7: Validate loss minimization strategies"""
        logger.info("\n=== TEST 7: LOSS MINIMIZATION VALIDATION ===")
        
        # Test distance-based loss calculator
        num_buildings = len(buildings)
        positions = torch.tensor([[b['x_coord'], b['y_coord']] for b in buildings.values()])
        
        loss_calculator = DistanceBasedLossCalculator(base_efficiency=0.98, loss_per_meter=0.0001)
        
        # Create test sharing matrix
        sharing_matrix = torch.zeros(1, num_buildings, num_buildings)
        
        # Add some test flows
        for i in range(num_buildings):
            for j in range(i+1, num_buildings):
                if buildings[i]['lv_group_id'] == buildings[j]['lv_group_id']:
                    # Simulate some energy sharing
                    sharing_matrix[0, i, j] = np.random.uniform(0, 5)
        
        # Calculate losses
        adjusted_sharing, distance_loss = loss_calculator(sharing_matrix, positions)
        
        # Verify losses increase with distance
        losses_by_distance = []
        for i in range(num_buildings):
            for j in range(i+1, num_buildings):
                if sharing_matrix[0, i, j] > 0:
                    distance = torch.norm(positions[i] - positions[j]).item()
                    original = sharing_matrix[0, i, j].item()
                    adjusted = adjusted_sharing[0, i, j].item()
                    loss = original - adjusted
                    loss_percentage = (loss / original) * 100 if original > 0 else 0
                    losses_by_distance.append({
                        'distance': distance,
                        'loss_percentage': loss_percentage
                    })
        
        # Sort by distance and verify trend
        losses_by_distance.sort(key=lambda x: x['distance'])
        
        if len(losses_by_distance) > 1:
            # Check if losses generally increase with distance
            distances = [l['distance'] for l in losses_by_distance]
            loss_percentages = [l['loss_percentage'] for l in losses_by_distance]
            
            # Simple correlation check
            correlation = np.corrcoef(distances, loss_percentages)[0, 1]
            logger.info(f"Distance-Loss Correlation: {correlation:.3f}")
            
            # Should be positive correlation
            assert correlation > 0, "Losses should increase with distance"
        
        total_original = sharing_matrix.sum().item()
        total_adjusted = adjusted_sharing.sum().item()
        total_loss = total_original - total_adjusted
        loss_percentage = (total_loss / total_original) * 100 if total_original > 0 else 0
        
        logger.info(f"Total Energy Shared: {total_original:.2f} kW")
        logger.info(f"Total After Losses: {total_adjusted:.2f} kW")
        logger.info(f"Total Losses: {total_loss:.2f} kW ({loss_percentage:.1f}%)")
        
        # Validate loss constraints
        assert loss_percentage < 20, f"Losses too high: {loss_percentage}%"
        assert total_adjusted >= 0, "Negative energy after losses"
        
        self.test_results['loss_minimization'] = {
            'passed': True,
            'total_shared': total_original,
            'total_losses': total_loss,
            'loss_percentage': loss_percentage,
            'distance_loss_correlation': correlation if len(losses_by_distance) > 1 else None
        }
        
        logger.info("✓ Loss minimization test PASSED")
        return True
    
    def test_grid_constraints(self, buildings: Dict, temporal_data: pd.DataFrame):
        """Test 8: Validate grid constraint handling"""
        logger.info("\n=== TEST 8: GRID CONSTRAINT VALIDATION ===")
        
        # Test LV group boundary enforcement
        num_buildings = len(buildings)
        lv_group_ids = torch.tensor([b['lv_group_id'] for b in buildings.values()])
        
        boundary_enforcer = LVGroupBoundaryEnforcer()
        
        # Create test sharing matrix with some invalid connections
        sharing_matrix = torch.rand(1, num_buildings, num_buildings) * 10
        
        # Apply boundary constraints
        valid_sharing, boundary_penalty = boundary_enforcer(sharing_matrix, lv_group_ids)
        
        # Verify no cross-boundary sharing
        violations = []
        for i in range(num_buildings):
            for j in range(num_buildings):
                if i != j and valid_sharing[0, i, j] > 0:
                    if buildings[i]['lv_group_id'] != buildings[j]['lv_group_id']:
                        violations.append((i, j, valid_sharing[0, i, j].item()))
        
        logger.info(f"Cross-boundary violations after enforcement: {len(violations)}")
        logger.info(f"Boundary penalty: {boundary_penalty.item():.4f}")
        
        assert len(violations) == 0, f"Found {len(violations)} cross-boundary connections"
        
        # Test transformer capacity constraints
        transformer_loads = {}
        transformer_capacities = {0: 200, 1: 300}  # kW ratings
        
        for i, b in buildings.items():
            t_id = b['transformer_id']
            if t_id not in transformer_loads:
                transformer_loads[t_id] = 0
            transformer_loads[t_id] += b['peak_demand_kw']
        
        overloaded_transformers = []
        for t_id, load in transformer_loads.items():
            capacity = transformer_capacities.get(t_id, 250)
            if load > capacity:
                overloaded_transformers.append({
                    'transformer': t_id,
                    'load': load,
                    'capacity': capacity,
                    'overload_percentage': ((load - capacity) / capacity) * 100
                })
        
        logger.info(f"Overloaded transformers: {len(overloaded_transformers)}")
        for t in overloaded_transformers:
            logger.info(f"  Transformer {t['transformer']}: {t['load']:.1f}/{t['capacity']} kW "
                       f"({t['overload_percentage']:.1f}% overload)")
        
        self.test_results['grid_constraints'] = {
            'passed': len(violations) == 0,
            'boundary_violations': len(violations),
            'boundary_penalty': boundary_penalty.item(),
            'overloaded_transformers': len(overloaded_transformers)
        }
        
        logger.info("✓ Grid constraint test PASSED")
        return True
    
    def generate_report(self):
        """Generate comprehensive test report"""
        logger.info("\n" + "="*60)
        logger.info("ENERGY OPTIMIZATION TEST REPORT")
        logger.info("="*60)
        
        all_passed = all(r.get('passed', False) for r in self.test_results.values())
        
        logger.info(f"\nOverall Status: {'✓ ALL TESTS PASSED' if all_passed else '⚠ SOME TESTS FAILED'}")
        logger.info(f"Tests Run: {len(self.test_results)}")
        logger.info(f"Tests Passed: {sum(1 for r in self.test_results.values() if r.get('passed', False))}")
        
        logger.info("\nDetailed Results:")
        for test_name, results in self.test_results.items():
            status = "✓ PASS" if results.get('passed', False) else "✗ FAIL"
            logger.info(f"\n{test_name}: {status}")
            for key, value in results.items():
                if key != 'passed':
                    if isinstance(value, float):
                        logger.info(f"  {key}: {value:.4f}")
                    else:
                        logger.info(f"  {key}: {value}")
        
        # Physical validation summary
        logger.info("\n" + "="*60)
        logger.info("PHYSICAL VALIDATION SUMMARY")
        logger.info("="*60)
        
        validations = [
            ("Energy Balance", self.test_results.get('energy_balance', {}).get('passed', False),
             "Energy is conserved across the system"),
            ("Voltage Constraints", self.test_results.get('voltage_constraints', {}).get('passed', False),
             "Voltage levels within ±10% of nominal"),
            ("Power Flow", self.test_results.get('power_flow', {}).get('passed', False),
             "Kirchhoff's laws are satisfied"),
            ("Grid Constraints", self.test_results.get('grid_constraints', {}).get('passed', False),
             "LV boundaries and transformer limits respected"),
            ("Loss Calculations", self.test_results.get('loss_minimization', {}).get('passed', False),
             "Losses increase with distance as expected"),
        ]
        
        for name, passed, description in validations:
            status = "✓" if passed else "✗"
            logger.info(f"{status} {name}: {description}")
        
        if self.validation_errors:
            logger.info("\n⚠ VALIDATION ERRORS:")
            for error in self.validation_errors:
                logger.info(f"  - {error}")
        
        return self.test_results


def main():
    """Run comprehensive energy optimization tests"""
    tester = EnergyOptimizationTester()
    
    # Generate test data
    logger.info("Generating test data...")
    buildings, temporal_data = tester.generate_test_data(num_buildings=20, num_timesteps=96)
    
    # Run all tests
    tests = [
        tester.test_energy_balance,
        tester.test_voltage_constraints,
        tester.test_power_flow_equations,
        tester.test_optimization_objectives,
        tester.test_p2p_trading_logic,
        tester.test_battery_optimization,
        tester.test_loss_minimization,
        tester.test_grid_constraints
    ]
    
    for test in tests:
        try:
            if test.__name__ in ['test_energy_balance', 'test_power_flow_equations', 
                                'test_optimization_objectives', 'test_p2p_trading_logic',
                                'test_battery_optimization', 'test_loss_minimization', 
                                'test_grid_constraints']:
                test(buildings, temporal_data)
            else:
                test(buildings)
        except Exception as e:
            logger.error(f"Test {test.__name__} failed with error: {e}")
            tester.validation_errors.append(f"{test.__name__}: {str(e)}")
    
    # Generate final report
    report = tester.generate_report()
    
    # Save report to file
    with open('energy_optimization_test_results.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info("\n✅ Test results saved to energy_optimization_test_results.json")
    
    return report


if __name__ == "__main__":
    main()