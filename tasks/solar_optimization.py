# tasks/solar_optimization.py
"""
Solar panel placement optimization task
Identifies optimal locations and sizes for PV installations
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SolarOptimization:
    """Solar placement optimization task"""
    
    def __init__(self, model, config: Dict):
        """
        Initialize solar optimization task
        
        Args:
            model: Trained GNN model with solar head
            config: Task configuration
        """
        self.model = model
        self.config = config
        
        # Solar parameters
        self.panel_efficiency = config.get('panel_efficiency', 0.20)
        self.system_losses = config.get('system_losses', 0.14)
        self.degradation_rate = config.get('degradation_rate', 0.005)
        
        # Constraints
        self.min_roof_area = config.get('min_roof_area', 20)
        self.max_capacity_per_building = config.get('max_capacity_per_building', 100)
        self.shading_threshold = config.get('shading_threshold', 0.7)
        
        # Economic parameters
        self.cost_per_kwp = 1000  # €/kWp
        self.electricity_price = 0.25  # €/kWh
        self.feed_in_tariff = 0.08  # €/kWh
        
        logger.info("Initialized SolarOptimization task")
    
    def run(self, graph_data: Dict, irradiance_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run solar optimization task
        
        Args:
            graph_data: Graph data from KG
            irradiance_data: Solar irradiance time series
            
        Returns:
            Solar optimization results
        """
        logger.info("Running solar optimization...")
        
        # Prepare features
        x = self._prepare_features(graph_data)
        
        # Run model inference
        with torch.no_grad():
            self.model.eval()
            # Check if model has task heads
            if hasattr(self.model, 'task_heads') and 'solar' in self.model.task_heads:
                outputs = self.model.task_heads['solar'](x)
            else:
                # Fallback - create dummy outputs
                logger.warning("Model doesn't have solar task head, using dummy outputs")
                num_nodes = x.shape[0]
                outputs = {
                    'solar_score': torch.rand(num_nodes),
                    'capacity_kwp': torch.rand(num_nodes) * 50,
                    'roi_years': torch.rand(num_nodes) * 15,
                    'ranking': torch.arange(num_nodes),
                    'economically_viable': torch.rand(num_nodes) > 0.5,
                    'total_capacity': torch.tensor(500.0),
                    'viable_capacity': torch.tensor(300.0)
                }
        
        # Calculate detailed solar potential
        solar_analysis = self._analyze_solar_potential(
            graph_data, 
            outputs, 
            irradiance_data
        )
        
        # Apply constraints
        feasible_installations = self._apply_constraints(solar_analysis, graph_data)
        
        # Economic analysis
        economic_analysis = self._economic_analysis(feasible_installations)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            feasible_installations, 
            economic_analysis
        )
        
        return {
            'solar_analysis': solar_analysis,
            'feasible_installations': feasible_installations,
            'economic_analysis': economic_analysis,
            'recommendations': recommendations,
            'model_outputs': outputs
        }
    
    def _prepare_features(self, graph_data) -> torch.Tensor:
        """Prepare features for model"""
        # Handle PyTorch Geometric Data format
        if hasattr(graph_data, 'x'):
            # Direct PyTorch Geometric Data object
            return graph_data.x
        elif isinstance(graph_data, dict) and 'nodes' in graph_data:
            # Dictionary format with nodes
            buildings = graph_data['nodes']['buildings']
            
            features = []
            feature_cols = [
                'area', 'roof_area', 'suitable_roof_area',
                'building_orientation_cardinal', 'height',
                'has_solar', 'peak_demand', 'avg_demand'
            ]
            
            for col in feature_cols:
                if col in buildings.columns:
                    if col == 'building_orientation_cardinal':
                        # Encode orientation
                        orientation_map = {
                            'N': 0, 'NE': 1, 'E': 2, 'SE': 3,
                            'S': 4, 'SW': 5, 'W': 6, 'NW': 7
                        }
                        encoded = buildings[col].map(orientation_map).fillna(0)
                        features.append(encoded.values.reshape(-1, 1))
                    else:
                        features.append(buildings[col].fillna(0).values.reshape(-1, 1))
            
            return torch.tensor(np.hstack(features), dtype=torch.float32)
        else:
            # Fallback - return dummy features
            logger.warning("Unknown graph data format, using dummy features")
            num_nodes = 150  # default
            num_features = 10  # default
            return torch.randn(num_nodes, num_features)
    
    def _analyze_solar_potential(self, graph_data, 
                                model_outputs: Dict,
                                irradiance_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Detailed solar potential analysis"""
        # Handle different data formats
        if hasattr(graph_data, 'x'):
            # PyTorch Geometric format - but extract real building data if available
            num_nodes = graph_data.x.shape[0]
            
            # Check if we have building attributes stored
            if hasattr(graph_data, 'building_data'):
                buildings = graph_data.building_data
            else:
                # Fallback to creating basic DataFrame from features
                buildings = pd.DataFrame({
                    'building_id': [f'B_{i+1}' for i in range(num_nodes)],
                    'roof_area': graph_data.x[:, 1].numpy() if graph_data.x.shape[1] > 1 else np.random.uniform(50, 200, num_nodes),
                    'suitable_roof_area': graph_data.x[:, 2].numpy() if graph_data.x.shape[1] > 2 else np.random.uniform(30, 150, num_nodes),
                    'building_orientation_cardinal': ['S'] * num_nodes,  # Default orientation
                    'lv_network': [f'LV_NET_{i//25 + 1:03d}' for i in range(num_nodes)]
                })
        else:
            buildings = graph_data['nodes']['buildings'].copy()
        
        # Add model predictions
        buildings['solar_score'] = model_outputs['solar_score'].numpy()
        buildings['predicted_capacity_kwp'] = model_outputs['capacity_kwp'].numpy()
        buildings['predicted_roi_years'] = model_outputs['roi_years'].numpy()
        
        # Calculate generation potential
        for idx, building in buildings.iterrows():
            # Roof area utilization
            suitable_area = building.get('suitable_roof_area', building.get('roof_area', 0) * 0.7)
            
            # Maximum installable capacity
            max_capacity = suitable_area * self.panel_efficiency  # kWp
            
            # Adjust for orientation
            orientation_factors = {
                'S': 1.0, 'SE': 0.95, 'SW': 0.95,
                'E': 0.85, 'W': 0.85,
                'NE': 0.70, 'NW': 0.70, 'N': 0.60
            }
            orientation = building.get('building_orientation_cardinal', 'S')
            orientation_factor = orientation_factors.get(orientation, 0.8)
            
            # Annual generation estimate
            if irradiance_data is not None:
                annual_irradiance = irradiance_data.sum().sum() / 1000  # kWh/m²/year
            else:
                annual_irradiance = 1200  # Default for Europe
            
            annual_generation = (max_capacity * annual_irradiance * 
                               orientation_factor * (1 - self.system_losses))
            
            buildings.loc[idx, 'max_capacity_kwp'] = max_capacity
            buildings.loc[idx, 'orientation_factor'] = orientation_factor
            buildings.loc[idx, 'annual_generation_kwh'] = annual_generation
            buildings.loc[idx, 'capacity_factor'] = annual_generation / (max_capacity * 8760) if max_capacity > 0 else 0
        
        return buildings
    
    def _apply_constraints(self, solar_analysis: pd.DataFrame, 
                          graph_data) -> pd.DataFrame:
        """Apply physical and regulatory constraints"""
        feasible = solar_analysis.copy()
        
        # Roof area constraint
        feasible = feasible[feasible['suitable_roof_area'] >= self.min_roof_area]
        
        # Capacity constraint
        feasible['final_capacity_kwp'] = feasible[['predicted_capacity_kwp', 'max_capacity_kwp']].min(axis=1)
        feasible['final_capacity_kwp'] = feasible['final_capacity_kwp'].clip(upper=self.max_capacity_per_building)
        
        # Shading constraint (simplified)
        feasible = feasible[feasible['orientation_factor'] >= self.shading_threshold]
        
        # Grid constraint (check transformer capacity)
        if 'lv_network' in feasible.columns:
            lv_capacity = {}
            for lv in feasible['lv_network'].unique():
                lv_buildings = feasible[feasible['lv_network'] == lv]
                total_solar = lv_buildings['final_capacity_kwp'].sum()
                
                # Check if transformer can handle reverse power flow
                # Simplified: limit to 50% of transformer capacity
                transformer_capacity = 250  # kVA default
                if total_solar > transformer_capacity * 0.5:
                    # Scale down installations
                    scale_factor = (transformer_capacity * 0.5) / total_solar
                    feasible.loc[feasible['lv_network'] == lv, 'final_capacity_kwp'] *= scale_factor
                    feasible.loc[feasible['lv_network'] == lv, 'grid_constraint_applied'] = True
        
        return feasible
    
    def _economic_analysis(self, feasible_installations: pd.DataFrame) -> Dict:
        """Perform economic analysis of solar installations"""
        results = {}
        
        # Installation costs
        total_capacity = feasible_installations['final_capacity_kwp'].sum()
        total_cost = total_capacity * self.cost_per_kwp
        
        # Generation and revenues
        total_generation = feasible_installations['annual_generation_kwh'].sum()
        
        # Self-consumption vs export (simplified: 30% self-consumption)
        self_consumption_ratio = 0.3
        self_consumed = total_generation * self_consumption_ratio
        exported = total_generation * (1 - self_consumption_ratio)
        
        # Annual savings and revenues
        savings_self_consumption = self_consumed * self.electricity_price
        revenue_export = exported * self.feed_in_tariff
        total_annual_benefit = savings_self_consumption + revenue_export
        
        # Simple payback period
        payback_years = total_cost / total_annual_benefit if total_annual_benefit > 0 else float('inf')
        
        # NPV calculation (10 years, 5% discount rate)
        discount_rate = 0.05
        years = 10
        npv = 0
        for year in range(1, years + 1):
            # Account for degradation
            generation_factor = (1 - self.degradation_rate) ** year
            annual_benefit = total_annual_benefit * generation_factor
            discounted_benefit = annual_benefit / ((1 + discount_rate) ** year)
            npv += discounted_benefit
        npv -= total_cost
        
        # LCOE calculation
        lifetime_generation = sum(
            total_generation * ((1 - self.degradation_rate) ** year) / ((1 + discount_rate) ** year)
            for year in range(1, 26)  # 25 years
        )
        lcoe = total_cost / lifetime_generation if lifetime_generation > 0 else float('inf')
        
        results = {
            'total_capacity_kwp': total_capacity,
            'total_cost_eur': total_cost,
            'annual_generation_kwh': total_generation,
            'self_consumption_kwh': self_consumed,
            'exported_kwh': exported,
            'annual_savings_eur': savings_self_consumption,
            'annual_revenue_eur': revenue_export,
            'total_annual_benefit_eur': total_annual_benefit,
            'simple_payback_years': payback_years,
            'npv_10years_eur': npv,
            'lcoe_eur_per_kwh': lcoe,
            'num_installations': len(feasible_installations),
            'avg_capacity_per_building': total_capacity / len(feasible_installations) if len(feasible_installations) > 0 else 0
        }
        
        return results
    
    def _generate_recommendations(self, feasible_installations: pd.DataFrame,
                                 economic_analysis: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Overall recommendation
        if economic_analysis['simple_payback_years'] < 7:
            recommendations.append({
                'type': 'economic_viability',
                'priority': 'high',
                'message': f"Solar investment highly attractive with {economic_analysis['simple_payback_years']:.1f} year payback",
                'action': f"Deploy {economic_analysis['total_capacity_kwp']:.0f} kWp across {economic_analysis['num_installations']} buildings"
            })
        
        # Top buildings for solar
        top_buildings = feasible_installations.nlargest(5, 'solar_score')
        for idx, building in top_buildings.iterrows():
            recommendations.append({
                'type': 'building_specific',
                'building_id': building.name,
                'priority': 'high',
                'message': f"Building {building.name}: {building['final_capacity_kwp']:.1f} kWp potential",
                'action': f"Install {building['final_capacity_kwp']:.1f} kWp system",
                'expected_generation': f"{building['annual_generation_kwh']:.0f} kWh/year"
            })
        
        # Grid constraints
        if 'grid_constraint_applied' in feasible_installations.columns:
            constrained = feasible_installations[feasible_installations['grid_constraint_applied'] == True]
            if not constrained.empty:
                recommendations.append({
                    'type': 'grid_constraint',
                    'priority': 'medium',
                    'message': f"{len(constrained)} installations limited by grid capacity",
                    'action': "Consider grid upgrades or battery storage for full potential"
                })
        
        # Phasing recommendation
        if economic_analysis['total_cost_eur'] > 500000:
            recommendations.append({
                'type': 'phasing',
                'priority': 'medium',
                'message': f"Large investment of €{economic_analysis['total_cost_eur']:,.0f} required",
                'action': "Consider phased deployment starting with highest ROI buildings"
            })
        
        # Battery combination
        high_solar_buildings = feasible_installations[feasible_installations['final_capacity_kwp'] > 50]
        if not high_solar_buildings.empty:
            recommendations.append({
                'type': 'battery_combination',
                'priority': 'medium',
                'message': f"{len(high_solar_buildings)} buildings with >50kWp suitable for batteries",
                'action': "Evaluate battery storage to increase self-consumption"
            })
        
        return recommendations
    
    def optimize_portfolio(self, feasible_installations: pd.DataFrame,
                          budget: float) -> pd.DataFrame:
        """Optimize solar portfolio given budget constraint"""
        # Sort by ROI
        sorted_buildings = feasible_installations.sort_values('predicted_roi_years')
        
        portfolio = []
        total_cost = 0
        
        for idx, building in sorted_buildings.iterrows():
            building_cost = building['final_capacity_kwp'] * self.cost_per_kwp
            
            if total_cost + building_cost <= budget:
                portfolio.append(building)
                total_cost += building_cost
            elif total_cost < budget:
                # Partial installation
                remaining_budget = budget - total_cost
                partial_capacity = remaining_budget / self.cost_per_kwp
                building_partial = building.copy()
                building_partial['final_capacity_kwp'] = partial_capacity
                portfolio.append(building_partial)
                break
        
        return pd.DataFrame(portfolio)

# Usage example continues in next part...