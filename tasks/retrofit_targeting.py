# tasks/retrofit_targeting.py
"""
Building retrofit prioritization task
Identifies buildings needing energy efficiency improvements
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class RetrofitTargeting:
    """Retrofit targeting and prioritization task"""
    
    def __init__(self, model, config: Dict):
        """
        Initialize retrofit targeting task
        
        Args:
            model: Trained GNN model
            config: Task configuration
        """
        self.model = model
        self.config = config
        
        # Age categories and priorities
        self.age_priorities = config.get('age_categories', {
            'pre_1945': 1.5,
            '1945_1975': 1.3,
            '1975_1995': 1.1,
            '1995_2010': 0.9,
            'post_2010': 0.5
        })
        
        # Performance targets
        self.target_energy_label = config.get('performance', {}).get('target_energy_label', 'B')
        self.max_heating_demand = config.get('performance', {}).get('max_heating_demand', 50)
        
        # Economic parameters
        self.investment_cap = config.get('economics', {}).get('investment_cap', 50000)
        self.payback_max = config.get('economics', {}).get('payback_period_max', 15)
        
        logger.info("Initialized RetrofitTargeting task")
    
    def run(self, graph_data: Dict, energy_data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Run retrofit targeting task
        
        Args:
            graph_data: Graph data from KG
            energy_data: Historical energy consumption
            
        Returns:
            Retrofit targeting results
        """
        logger.info("Running retrofit targeting...")
        
        # Prepare features
        x = self._prepare_features(graph_data)
        
        # Run model inference
        with torch.no_grad():
            self.model.eval()
            outputs = self.model.task_heads['retrofit'](x)
        
        # Analyze retrofit needs
        retrofit_analysis = self._analyze_retrofit_needs(graph_data, outputs, energy_data)
        
        # Calculate intervention packages
        intervention_packages = self._design_interventions(retrofit_analysis)
        
        # Economic assessment
        economic_assessment = self._economic_assessment(intervention_packages)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            retrofit_analysis,
            intervention_packages,
            economic_assessment
        )
        
        return {
            'retrofit_analysis': retrofit_analysis,
            'intervention_packages': intervention_packages,
            'economic_assessment': economic_assessment,
            'recommendations': recommendations,
            'model_outputs': outputs
        }
    
    def _prepare_features(self, graph_data: Dict) -> torch.Tensor:
        """Prepare building features for model"""
        buildings = graph_data['nodes']['buildings']
        
        features = []
        
        # Building characteristics
        features.append(buildings['area'].fillna(100).values.reshape(-1, 1))
        features.append(buildings['height'].fillna(3).values.reshape(-1, 1))
        
        # Age encoding
        age_map = {age: i for i, age in enumerate(self.age_priorities.keys())}
        age_encoded = buildings['age_range'].map(age_map).fillna(2).values
        features.append(age_encoded.reshape(-1, 1))
        
        # Energy performance
        label_map = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
        label_encoded = buildings['energy_label'].map(label_map).fillna(3).values
        features.append(label_encoded.reshape(-1, 1))
        
        # Insulation quality
        insulation_map = {'excellent': 4, 'good': 3, 'fair': 2, 'poor': 1}
        insulation_encoded = buildings['insulation_quality'].map(insulation_map).fillna(2).values
        features.append(insulation_encoded.reshape(-1, 1))
        
        return torch.tensor(np.hstack(features), dtype=torch.float32)
    
    def _analyze_retrofit_needs(self, graph_data: Dict, 
                               model_outputs: Dict,
                               energy_data: Optional[pd.DataFrame]) -> pd.DataFrame:
        """Analyze retrofit needs for each building"""
        buildings = graph_data['nodes']['buildings'].copy()
        
        # Add model predictions
        buildings['retrofit_score'] = model_outputs['retrofit_score'].numpy()
        buildings['predicted_savings'] = model_outputs['energy_savings'].numpy()
        buildings['estimated_cost'] = model_outputs['retrofit_cost'].numpy()
        
        # Calculate current performance
        for idx, building in buildings.iterrows():
            # Estimated heating demand (simplified)
            building_age_factor = self.age_priorities.get(building.get('age_range', '1975_1995'), 1.0)
            insulation_factor = {'poor': 2.0, 'fair': 1.5, 'good': 1.0, 'excellent': 0.7}.get(
                building.get('insulation_quality', 'fair'), 1.5
            )
            
            base_demand = 100  # kWh/m²/year baseline
            current_demand = base_demand * building_age_factor * insulation_factor
            buildings.loc[idx, 'current_heating_demand'] = current_demand
            
            # Gap to target
            buildings.loc[idx, 'demand_gap'] = max(0, current_demand - self.max_heating_demand)
            
            # Priority calculation
            priority = (
                buildings.loc[idx, 'retrofit_score'] * 0.4 +
                (buildings.loc[idx, 'demand_gap'] / 100) * 0.3 +
                buildings.loc[idx, 'predicted_savings'] * 0.3
            )
            buildings.loc[idx, 'priority'] = priority
        
        # Rank by priority
        buildings['rank'] = buildings['priority'].rank(ascending=False, method='dense')
        
        return buildings
    
    def _design_interventions(self, retrofit_analysis: pd.DataFrame) -> pd.DataFrame:
        """Design specific intervention packages"""
        interventions = []
        
        for idx, building in retrofit_analysis.iterrows():
            package = {
                'building_id': idx,
                'current_demand': building['current_heating_demand'],
                'target_demand': self.max_heating_demand,
                'measures': []
            }
            
            # Determine needed measures based on current state
            
            # Insulation upgrade
            if building.get('insulation_quality') in ['poor', 'fair']:
                package['measures'].append({
                    'type': 'insulation',
                    'description': 'Wall and roof insulation upgrade',
                    'cost': building['area'] * 50,  # €50/m²
                    'savings': building['current_heating_demand'] * 0.3  # 30% reduction
                })
            
            # Window replacement
            if building.get('age_range') in ['pre_1945', '1945_1975']:
                package['measures'].append({
                    'type': 'windows',
                    'description': 'Triple glazing installation',
                    'cost': building['area'] * 0.2 * 300,  # 20% window area, €300/m²
                    'savings': building['current_heating_demand'] * 0.15  # 15% reduction
                })
            
            # Heating system upgrade
            if building.get('heating_system') == 'gas' and building.get('energy_label') in ['E', 'F', 'G']:
                package['measures'].append({
                    'type': 'heating',
                    'description': 'High-efficiency boiler or heat pump',
                    'cost': 10000 + building['area'] * 20,
                    'savings': building['current_heating_demand'] * 0.25  # 25% reduction
                })
            
            # Ventilation with heat recovery
            if building.get('insulation_quality') in ['good', 'excellent']:
                package['measures'].append({
                    'type': 'ventilation',
                    'description': 'Mechanical ventilation with heat recovery',
                    'cost': 5000 + building['area'] * 10,
                    'savings': building['current_heating_demand'] * 0.1  # 10% reduction
                })
            
            # Calculate totals
            package['total_cost'] = sum(m['cost'] for m in package['measures'])
            package['total_savings_kwh'] = sum(m['savings'] for m in package['measures'])
            package['final_demand'] = building['current_heating_demand'] - package['total_savings_kwh']
            
            interventions.append(package)
        
        return pd.DataFrame(interventions)
    
    def _economic_assessment(self, intervention_packages: pd.DataFrame) -> Dict:
        """Assess economic viability of interventions"""
        results = {}
        
        # Energy prices
        energy_price = 0.10  # €/kWh for heating
        
        # Calculate economics for each package
        viable_packages = []
        
        for _, package in intervention_packages.iterrows():
            if package['total_cost'] <= self.investment_cap:
                annual_savings = package['total_savings_kwh'] * energy_price
                payback = package['total_cost'] / annual_savings if annual_savings > 0 else float('inf')
                
                if payback <= self.payback_max:
                    viable_packages.append({
                        'building_id': package['building_id'],
                        'cost': package['total_cost'],
                        'annual_savings': annual_savings,
                        'payback_years': payback,
                        'roi': (annual_savings * 20 - package['total_cost']) / package['total_cost']  # 20-year ROI
                    })
        
        viable_df = pd.DataFrame(viable_packages)
        
        if not viable_df.empty:
            results = {
                'num_viable_retrofits': len(viable_df),
                'total_investment': viable_df['cost'].sum(),
                'total_annual_savings': viable_df['annual_savings'].sum(),
                'avg_payback': viable_df['payback_years'].mean(),
                'best_roi_building': viable_df.loc[viable_df['roi'].idxmax(), 'building_id'],
                'best_roi': viable_df['roi'].max()
            }
        else:
            results = {
                'num_viable_retrofits': 0,
                'total_investment': 0,
                'total_annual_savings': 0
            }
        
        return results
    
    def _generate_recommendations(self, retrofit_analysis: pd.DataFrame,
                                 intervention_packages: pd.DataFrame,
                                 economic_assessment: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # High priority buildings
        top_priority = retrofit_analysis.nsmallest(10, 'rank')
        
        for _, building in top_priority.iterrows():
            package = intervention_packages[intervention_packages['building_id'] == building.name].iloc[0]
            
            if package['total_cost'] <= self.investment_cap:
                recommendations.append({
                    'type': 'building_retrofit',
                    'building_id': building.name,
                    'priority': 'high' if building['rank'] <= 5 else 'medium',
                    'current_performance': f"{building['current_heating_demand']:.0f} kWh/m²/year",
                    'measures': package['measures'],
                    'investment': f"€{package['total_cost']:,.0f}",
                    'savings': f"{package['total_savings_kwh']:.0f} kWh/year"
                })
        
        # Neighborhood approach
        if 'lv_network' in retrofit_analysis.columns:
            for lv in retrofit_analysis['lv_network'].unique():
                lv_buildings = retrofit_analysis[retrofit_analysis['lv_network'] == lv]
                if (lv_buildings['priority'] > 0.7).sum() > 5:
                    recommendations.append({
                        'type': 'neighborhood_retrofit',
                        'area': lv,
                        'priority': 'high',
                        'message': f"{(lv_buildings['priority'] > 0.7).sum()} buildings need retrofit in {lv}",
                        'action': "Consider bulk procurement for cost reduction"
                    })
        
        # Economic summary
        if economic_assessment['num_viable_retrofits'] > 0:
            recommendations.append({
                'type': 'economic_summary',
                'priority': 'info',
                'viable_retrofits': economic_assessment['num_viable_retrofits'],
                'total_investment': f"€{economic_assessment['total_investment']:,.0f}",
                'annual_savings': f"€{economic_assessment['total_annual_savings']:,.0f}",
                'avg_payback': f"{economic_assessment['avg_payback']:.1f} years"
            })
        
        return recommendations