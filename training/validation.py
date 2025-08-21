# training/validation.py
"""
Physics and economic validation for model outputs
Ensures predictions are feasible and economically viable
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PhysicsValidator:
    """Validate model outputs against physical constraints"""
    
    def __init__(self, config: Dict):
        """
        Initialize physics validator
        
        Args:
            config: Configuration with grid parameters
        """
        self.config = config
        
        # Grid parameters
        self.voltage_tolerance = config.get('voltage_tolerance', 0.05)  # ±5%
        self.power_factor_min = config.get('power_factor_min', 0.85)
        self.frequency_tolerance = config.get('frequency_tolerance', 0.5)  # Hz
        
        # Transformer limits
        self.transformer_capacities = config.get('transformer_capacities', {
            'LV': 250,  # kVA
            'MV': 1000,  # kVA
            'HV': 10000  # kVA
        })
        
        logger.info("Initialized PhysicsValidator")
    
    def validate(self, predictions: Dict, graph_data: Dict) -> Dict:
        """
        Validate predictions against physics constraints
        
        Args:
            predictions: Model predictions
            graph_data: Graph structure and parameters
            
        Returns:
            Validation results and violations
        """
        results = {
            'valid': True,
            'violations': [],
            'warnings': [],
            'metrics': {}
        }
        
        # Power balance validation
        power_results = self._validate_power_balance(predictions, graph_data)
        results['metrics']['power_balance'] = power_results
        if not power_results['valid']:
            results['violations'].append('Power balance violation')
            results['valid'] = False
        
        # Voltage constraints
        voltage_results = self._validate_voltage(predictions, graph_data)
        results['metrics']['voltage'] = voltage_results
        if not voltage_results['valid']:
            results['violations'].append('Voltage constraint violation')
            results['valid'] = False
        
        # Transformer capacity
        transformer_results = self._validate_transformer_capacity(predictions, graph_data)
        results['metrics']['transformer'] = transformer_results
        if not transformer_results['valid']:
            results['violations'].append('Transformer overload')
            results['valid'] = False
        
        # Line capacity
        line_results = self._validate_line_capacity(predictions, graph_data)
        results['metrics']['line_capacity'] = line_results
        if not line_results['valid']:
            results['warnings'].append('Line capacity exceeded')
        
        # Renewable integration limits
        renewable_results = self._validate_renewable_integration(predictions, graph_data)
        results['metrics']['renewable'] = renewable_results
        if renewable_results['penetration'] > 0.5:
            results['warnings'].append('High renewable penetration may cause stability issues')
        
        return results
    
    def _validate_power_balance(self, predictions: Dict, graph_data) -> Dict:
        """Validate power balance (generation = consumption + losses)"""
        results = {'valid': True}
        
        if 'power_flows' in predictions:
            power_flows = predictions['power_flows']
            
            # Sum of all power injections should be close to zero
            total_injection = torch.sum(power_flows)
            
            # Allow for losses (typically 3-5%)
            max_imbalance = 0.05 * torch.abs(power_flows).sum()
            
            results['total_injection'] = total_injection.item()
            results['max_allowed_imbalance'] = max_imbalance.item()
            results['valid'] = torch.abs(total_injection) <= max_imbalance
        
        return results
    
    def _validate_voltage(self, predictions: Dict, graph_data: Dict) -> Dict:
        """Validate voltage constraints"""
        results = {'valid': True}
        
        if 'voltages' in predictions:
            voltages = predictions['voltages']
            
            # Check voltage deviations (should be within ±5% of nominal)
            voltage_violations = torch.abs(voltages - 1.0) > self.voltage_tolerance
            
            results['num_violations'] = voltage_violations.sum().item()
            results['max_deviation'] = torch.max(torch.abs(voltages - 1.0)).item()
            results['valid'] = not voltage_violations.any()
        
        return results
    
    def _validate_transformer_capacity(self, predictions: Dict, graph_data) -> Dict:
        """Validate transformer loading"""
        results = {'valid': True, 'transformers': []}
        
        if 'clustering' in predictions and 'clusters' in predictions['clustering']:
            clusters = predictions['clustering']['clusters']
            buildings = graph_data['nodes']['buildings']
            
            for cluster_id, cluster_info in clusters.items():
                building_indices = cluster_info['buildings']
                
                # Get transformer assignment
                transformers = buildings.iloc[building_indices]['lv_network'].unique()
                
                for transformer in transformers:
                    transformer_buildings = buildings[buildings['lv_network'] == transformer]
                    
                    # Calculate total load
                    total_load = transformer_buildings['peak_demand'].sum()
                    capacity = self.transformer_capacities['LV']
                    
                    utilization = total_load / capacity
                    
                    transformer_result = {
                        'transformer': transformer,
                        'load_kw': total_load,
                        'capacity_kva': capacity,
                        'utilization': utilization,
                        'overloaded': utilization > 1.0
                    }
                    
                    results['transformers'].append(transformer_result)
                    
                    if utilization > 1.0:
                        results['valid'] = False
        
        return results
    
    def _validate_line_capacity(self, predictions: Dict, graph_data) -> Dict:
        """Validate line current limits"""
        results = {'valid': True, 'lines': []}
        
        if 'line_flows' in predictions:
            line_flows = predictions['line_flows']
            
            # Typical LV cable rating: 200A
            max_current = 200  # Amperes
            voltage = 400  # Volts (LV)
            max_power = max_current * voltage * np.sqrt(3) / 1000  # kW
            
            for i, flow in enumerate(line_flows):
                if flow > max_power:
                    results['lines'].append({
                        'line_id': i,
                        'flow_kw': flow.item(),
                        'capacity_kw': max_power,
                        'overload_percent': (flow.item() / max_power - 1) * 100
                    })
                    results['valid'] = False
        
        return results
    
    def _validate_renewable_integration(self, predictions: Dict, graph_data) -> Dict:
        """Validate renewable energy integration limits"""
        results = {}
        
        if 'solar' in predictions and 'total_capacity_kwp' in predictions['solar']:
            solar_capacity = predictions['solar']['total_capacity_kwp']
            
            # Get total system capacity
            if hasattr(graph_data, 'x'):
                # PyTorch Geometric format - use estimated demand
                num_nodes = graph_data.x.shape[0]
                total_demand = num_nodes * 10  # Assume 10kW average peak per building
            elif isinstance(graph_data, dict) and 'nodes' in graph_data:
                buildings = graph_data['nodes']['buildings']
                total_demand = buildings['peak_demand'].sum()
            else:
                total_demand = 1000  # Default 1MW
            
            penetration = solar_capacity / total_demand if total_demand > 0 else 0
            
            results['solar_capacity_kwp'] = solar_capacity
            results['peak_demand_kw'] = total_demand
            results['penetration'] = penetration
            
            # Check if penetration is reasonable
            if penetration > 0.3:
                results['requires_storage'] = True
                results['recommended_storage_kwh'] = solar_capacity * 2  # 2 hours storage
        else:
            # Default values if solar not in predictions
            results['solar_capacity_kwp'] = 0
            results['peak_demand_kw'] = 1000
            results['penetration'] = 0
        
        return results

class EconomicValidator:
    """Validate economic viability of recommendations"""
    
    def __init__(self, config: Dict):
        """
        Initialize economic validator
        
        Args:
            config: Economic parameters
        """
        self.config = config
        
        # Cost parameters
        self.costs = {
            'solar_per_kwp': 1000,  # €/kWp
            'battery_per_kwh': 500,  # €/kWh
            'heat_pump_per_kw': 800,  # €/kW
            'retrofit_per_m2': 100,  # €/m²
            'grid_upgrade_per_kva': 200  # €/kVA
        }
        
        # Economic parameters
        self.discount_rate = config.get('discount_rate', 0.05)
        self.electricity_price = config.get('electricity_price', 0.25)  # €/kWh
        self.feed_in_tariff = config.get('feed_in_tariff', 0.08)  # €/kWh
        self.carbon_price = config.get('carbon_price', 50)  # €/tCO2
        
        logger.info("Initialized EconomicValidator")
    
    def validate(self, predictions: Dict, graph_data: Dict) -> Dict:
        """
        Validate economic viability
        
        Args:
            predictions: Model predictions
            graph_data: Graph structure
            
        Returns:
            Economic validation results
        """
        results = {
            'total_investment': 0,
            'annual_savings': 0,
            'payback_period': float('inf'),
            'npv': 0,
            'irr': 0,
            'interventions': []
        }
        
        # Solar economics
        if 'solar' in predictions:
            solar_results = self._validate_solar_economics(predictions['solar'])
            results['interventions'].append(solar_results)
            results['total_investment'] += solar_results['investment']
            results['annual_savings'] += solar_results['annual_savings']
        
        # Battery economics
        if 'battery' in predictions:
            battery_results = self._validate_battery_economics(predictions['battery'])
            results['interventions'].append(battery_results)
            results['total_investment'] += battery_results['investment']
            results['annual_savings'] += battery_results['annual_savings']
        
        # Retrofit economics
        if 'retrofit' in predictions:
            retrofit_results = self._validate_retrofit_economics(predictions['retrofit'], graph_data)
            results['interventions'].append(retrofit_results)
            results['total_investment'] += retrofit_results['investment']
            results['annual_savings'] += retrofit_results['annual_savings']
        
        # Calculate overall economics
        if results['annual_savings'] > 0:
            results['payback_period'] = results['total_investment'] / results['annual_savings']
            results['npv'] = self._calculate_npv(
                results['total_investment'],
                results['annual_savings'],
                20  # 20 year lifetime
            )
            results['irr'] = self._calculate_irr(
                results['total_investment'],
                results['annual_savings'],
                20
            )
        
        # Viability assessment
        results['economically_viable'] = (
            results['payback_period'] < 10 and
            results['npv'] > 0 and
            results['irr'] > self.discount_rate
        )
        
        return results
    
    def _validate_solar_economics(self, solar_predictions: Dict) -> Dict:
        """Validate solar installation economics"""
        results = {'type': 'solar'}
        
        if 'total_capacity_kwp' in solar_predictions:
            capacity = solar_predictions['total_capacity_kwp']
            
            # Investment
            results['investment'] = capacity * self.costs['solar_per_kwp']
            
            # Annual generation (Europe average)
            annual_generation = capacity * 1200  # kWh/year
            
            # Savings (30% self-consumption, 70% export)
            self_consumption = annual_generation * 0.3
            export = annual_generation * 0.7
            
            results['annual_savings'] = (
                self_consumption * self.electricity_price +
                export * self.feed_in_tariff
            )
            
            # Carbon savings (0.4 kgCO2/kWh avoided)
            results['carbon_savings_tonnes'] = annual_generation * 0.4 / 1000
            results['carbon_value'] = results['carbon_savings_tonnes'] * self.carbon_price
        
        return results
    
    def _validate_battery_economics(self, battery_predictions: Dict) -> Dict:
        """Validate battery storage economics"""
        results = {'type': 'battery'}
        
        if 'total_storage_capacity' in battery_predictions:
            capacity = battery_predictions['total_storage_capacity']
            
            # Investment
            results['investment'] = capacity * self.costs['battery_per_kwh']
            
            # Annual savings (peak shaving + solar storage)
            # Simplified: 1 cycle per day, 90% efficiency
            daily_throughput = capacity * 0.9
            annual_throughput = daily_throughput * 250  # Operating days
            
            # Value from price arbitrage
            price_differential = 0.10  # €/kWh peak vs off-peak
            results['annual_savings'] = annual_throughput * price_differential
        
        return results
    
    def _validate_retrofit_economics(self, retrofit_predictions: Dict, 
                                    graph_data: Dict) -> Dict:
        """Validate retrofit economics"""
        results = {'type': 'retrofit'}
        
        if 'num_viable_retrofits' in retrofit_predictions:
            num_retrofits = retrofit_predictions['num_viable_retrofits']
            
            # Average building size
            buildings = graph_data['nodes']['buildings']
            avg_area = buildings['area'].mean()
            
            # Investment
            results['investment'] = num_retrofits * avg_area * self.costs['retrofit_per_m2']
            
            # Savings (30% energy reduction average)
            avg_consumption = buildings['avg_demand'].mean() * 8760  # kWh/year
            energy_saved = avg_consumption * 0.3 * num_retrofits
            
            results['annual_savings'] = energy_saved * self.electricity_price * 0.5  # Heating is cheaper
            
            # Carbon savings
            results['carbon_savings_tonnes'] = energy_saved * 0.2 / 1000  # Gas heating emissions
        
        return results
    
    def _calculate_npv(self, investment: float, annual_savings: float, 
                      lifetime: int) -> float:
        """Calculate Net Present Value"""
        npv = -investment
        
        for year in range(1, lifetime + 1):
            discounted_cashflow = annual_savings / ((1 + self.discount_rate) ** year)
            npv += discounted_cashflow
        
        return npv
    
    def _calculate_irr(self, investment: float, annual_savings: float,
                      lifetime: int) -> float:
        """Calculate Internal Rate of Return (simplified)"""
        # Newton-Raphson method
        irr = 0.1  # Initial guess
        
        for _ in range(100):
            npv = -investment
            dnpv = 0
            
            for year in range(1, lifetime + 1):
                npv += annual_savings / ((1 + irr) ** year)
                dnpv -= year * annual_savings / ((1 + irr) ** (year + 1))
            
            if abs(npv) < 0.01:
                break
            
            irr = irr - npv / dnpv
        
        return irr

class SafetyValidator:
    """Validate safety aspects of recommendations"""
    
    def __init__(self):
        """Initialize safety validator"""
        self.safety_factors = {
            'transformer_loading': 0.8,  # Max 80% loading
            'line_loading': 0.7,  # Max 70% loading
            'voltage_drop': 0.03,  # Max 3% drop
            'n_minus_1': True  # N-1 contingency
        }
    
    def validate(self, predictions: Dict, graph_data: Dict) -> Dict:
        """
        Validate safety constraints
        
        Args:
            predictions: Model predictions
            graph_data: Graph structure
            
        Returns:
            Safety validation results
        """
        results = {
            'safe': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check loading factors
        if 'transformer_utilization' in predictions:
            for transformer in predictions['transformer_utilization']:
                if transformer['utilization'] > self.safety_factors['transformer_loading']:
                    results['issues'].append(
                        f"Transformer {transformer['id']} overloaded: "
                        f"{transformer['utilization']:.1%}"
                    )
                    results['safe'] = False
        
        # N-1 contingency check
        if self.safety_factors['n_minus_1']:
            contingency_results = self._check_n_minus_1(predictions, graph_data)
            if not contingency_results['secure']:
                results['issues'].append("System not N-1 secure")
                results['recommendations'].append(
                    "Add redundancy or reduce critical loads"
                )
        
        return results
    
    def _check_n_minus_1(self, predictions: Dict, graph_data: Dict) -> Dict:
        """Check N-1 contingency security"""
        results = {'secure': True}
        
        # Simplified: Check if losing any single component causes overload
        # In practice, would run power flow for each contingency
        
        if 'clusters' in predictions:
            for cluster in predictions['clusters'].values():
                if len(cluster.get('transformers', [])) < 2:
                    results['secure'] = False
                    results['vulnerable_clusters'] = cluster['id']
        
        return results

# Combined validator
class ComprehensiveValidator:
    """Combined physics, economic, and safety validation"""
    
    def __init__(self, config: Dict):
        """Initialize comprehensive validator"""
        self.physics_validator = PhysicsValidator(config)
        self.economic_validator = EconomicValidator(config)
        self.safety_validator = SafetyValidator()
    
    def validate(self, predictions: Dict, graph_data: Dict) -> Dict:
        """
        Perform comprehensive validation
        
        Args:
            predictions: Model predictions
            graph_data: Graph structure
            
        Returns:
            Complete validation results
        """
        results = {
            'physics': self.physics_validator.validate(predictions, graph_data),
            'economics': self.economic_validator.validate(predictions, graph_data),
            'safety': self.safety_validator.validate(predictions, graph_data)
        }
        
        # Overall assessment
        results['overall_valid'] = (
            results['physics']['valid'] and
            results['economics']['economically_viable'] and
            results['safety']['safe']
        )
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> str:
        """Generate validation summary"""
        summary = []
        
        if results['overall_valid']:
            summary.append("✅ All validations passed")
        else:
            summary.append("❌ Validation issues detected:")
            
            if not results['physics']['valid']:
                summary.append(f"  - Physics: {results['physics']['violations']}")
            
            if not results['economics']['economically_viable']:
                summary.append(f"  - Economics: Payback {results['economics']['payback_period']:.1f} years")
            
            if not results['safety']['safe']:
                summary.append(f"  - Safety: {results['safety']['issues']}")
        
        return "\n".join(summary)

# Usage example
if __name__ == "__main__":
    import yaml
    
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create validator
    validator = ComprehensiveValidator(config)
    
    # Mock predictions
    predictions = {
        'clustering': {
            'clusters': {
                0: {'buildings': [0, 1, 2], 'transformers': ['T1']},
                1: {'buildings': [3, 4, 5], 'transformers': ['T2']}
            }
        },
        'solar': {
            'total_capacity_kwp': 500
        },
        'voltages': torch.tensor([0.98, 1.02, 0.97, 1.03, 0.99])
    }
    
    # Mock graph data
    graph_data = {
        'nodes': {
            'buildings': pd.DataFrame({
                'peak_demand': [10, 15, 20, 12, 18, 25],
                'area': [150, 200, 180, 160, 220, 190],
                'lv_network': ['T1', 'T1', 'T1', 'T2', 'T2', 'T2']
            })
        }
    }
    
    # Validate
    results = validator.validate(predictions, graph_data)
    
    print("Validation Results:")
    print(results['summary'])