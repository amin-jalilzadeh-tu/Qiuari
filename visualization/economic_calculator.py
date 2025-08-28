"""
Economic Calculator Module
Comprehensive financial analysis for energy communities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class EconomicCalculator:
    """Calculate economic metrics and financial projections"""
    
    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        
        # Energy prices (EUR/kWh)
        self.electricity_price = config.get('electricity_price', 0.25)
        self.feed_in_tariff = config.get('feed_in_tariff', 0.08)
        self.p2p_price = config.get('p2p_price', 0.15)  # P2P trading price
        
        # Peak charges (EUR/kW/month)
        self.peak_charge = config.get('peak_charge', 50)
        
        # Solar costs
        self.solar_cost_per_kwp = config.get('solar_cost_per_kwp', 1200)
        self.solar_maintenance_per_kwp = config.get('solar_maintenance', 20)  # Annual
        
        # Battery costs
        self.battery_cost_per_kwh = config.get('battery_cost_per_kwh', 500)
        self.battery_lifetime_cycles = config.get('battery_cycles', 5000)
        
        # Financial parameters
        self.discount_rate = config.get('discount_rate', 0.05)
        self.inflation_rate = config.get('inflation_rate', 0.02)
        self.project_lifetime = config.get('project_lifetime', 20)  # Years
        
        # Subsidies and incentives
        self.solar_subsidy_percent = config.get('solar_subsidy', 0.3)  # 30%
        self.community_bonus = config.get('community_bonus', 0.1)  # 10% extra
        
        # Carbon pricing
        self.carbon_price_per_ton = config.get('carbon_price', 50)
        self.grid_carbon_intensity = config.get('grid_carbon', 0.5)  # kg CO2/kWh
    
    def calculate_solar_roi(self, 
                           capacity_kwp: float,
                           annual_generation_kwh: float,
                           self_consumption_ratio: float,
                           building_demand_kwh: float) -> Dict:
        """Calculate ROI for solar installation"""
        
        # Initial investment
        gross_cost = capacity_kwp * self.solar_cost_per_kwp
        subsidy = gross_cost * self.solar_subsidy_percent
        net_investment = gross_cost - subsidy
        
        # Annual revenues
        self_consumed = min(annual_generation_kwh * self_consumption_ratio, building_demand_kwh)
        exported = annual_generation_kwh - self_consumed
        
        # Savings and revenues
        self_consumption_savings = self_consumed * self.electricity_price
        export_revenue = exported * self.feed_in_tariff
        annual_revenue = self_consumption_savings + export_revenue
        
        # Annual costs
        annual_maintenance = capacity_kwp * self.solar_maintenance_per_kwp
        net_annual_benefit = annual_revenue - annual_maintenance
        
        # Simple payback
        simple_payback = net_investment / net_annual_benefit if net_annual_benefit > 0 else float('inf')
        
        # NPV calculation
        cash_flows = [-net_investment]
        for year in range(1, self.project_lifetime + 1):
            # Adjust for inflation
            adjusted_benefit = net_annual_benefit * ((1 + self.inflation_rate) ** year)
            # Discount to present value
            pv = adjusted_benefit / ((1 + self.discount_rate) ** year)
            cash_flows.append(pv)
        
        npv = sum(cash_flows)
        
        # IRR calculation (simplified)
        irr = self._calculate_irr(cash_flows)
        
        # Profitability index
        pv_benefits = sum(cash_flows[1:])  # Exclude initial investment
        profitability_index = pv_benefits / net_investment if net_investment > 0 else 0
        
        return {
            'capacity_kwp': capacity_kwp,
            'investment': net_investment,
            'annual_revenue': annual_revenue,
            'annual_cost': annual_maintenance,
            'net_annual_benefit': net_annual_benefit,
            'simple_payback_years': simple_payback,
            'npv': npv,
            'irr': irr,
            'profitability_index': profitability_index,
            'lifetime_revenue': annual_revenue * self.project_lifetime,
            'carbon_saved_tons': (annual_generation_kwh * self.grid_carbon_intensity * 
                                 self.project_lifetime) / 1000
        }
    
    def calculate_battery_economics(self,
                                   capacity_kwh: float,
                                   daily_cycles: float,
                                   peak_shaving_kw: float,
                                   arbitrage_revenue_daily: float) -> Dict:
        """Calculate economics of battery storage"""
        
        # Investment
        investment = capacity_kwh * self.battery_cost_per_kwh
        
        # Battery lifetime in years
        annual_cycles = daily_cycles * 365
        battery_lifetime = min(self.battery_lifetime_cycles / annual_cycles, 10)  # Max 10 years
        
        # Annual benefits
        peak_savings_monthly = peak_shaving_kw * self.peak_charge
        peak_savings_annual = peak_savings_monthly * 12
        
        arbitrage_annual = arbitrage_revenue_daily * 365
        
        total_annual_benefit = peak_savings_annual + arbitrage_annual
        
        # Simple payback
        simple_payback = investment / total_annual_benefit if total_annual_benefit > 0 else float('inf')
        
        # NPV over battery lifetime
        npv = -investment
        for year in range(1, int(battery_lifetime) + 1):
            pv = total_annual_benefit / ((1 + self.discount_rate) ** year)
            npv += pv
        
        return {
            'capacity_kwh': capacity_kwh,
            'investment': investment,
            'lifetime_years': battery_lifetime,
            'annual_benefit': total_annual_benefit,
            'peak_savings': peak_savings_annual,
            'arbitrage_revenue': arbitrage_annual,
            'simple_payback': simple_payback,
            'npv': npv,
            'benefit_cost_ratio': (total_annual_benefit * battery_lifetime) / investment
        }
    
    def calculate_community_benefits(self,
                                    num_buildings: int,
                                    shared_energy_kwh: float,
                                    peak_reduction_percent: float,
                                    avg_building_demand_kwh: float) -> Dict:
        """Calculate economic benefits of energy community formation"""
        
        # P2P trading benefits
        p2p_savings = shared_energy_kwh * (self.electricity_price - self.p2p_price)
        p2p_revenue = shared_energy_kwh * (self.p2p_price - self.feed_in_tariff)
        total_p2p_benefit = p2p_savings + p2p_revenue
        
        # Peak reduction benefits
        total_demand = num_buildings * avg_building_demand_kwh
        avg_peak_kw = total_demand / (365 * 24) * 3  # Assume peak is 3x average
        peak_reduction_kw = avg_peak_kw * peak_reduction_percent
        peak_savings = peak_reduction_kw * self.peak_charge * 12
        
        # Community bonus (for renewable integration)
        renewable_bonus = shared_energy_kwh * self.electricity_price * self.community_bonus
        
        # Network benefits (reduced losses, deferred upgrades)
        network_benefits = total_demand * 0.02 * self.electricity_price  # 2% of total
        
        # Total community benefits
        total_annual_benefit = (total_p2p_benefit + peak_savings + 
                               renewable_bonus + network_benefits)
        
        # Per building benefits
        benefit_per_building = total_annual_benefit / num_buildings if num_buildings > 0 else 0
        
        # Carbon benefits
        carbon_saved = shared_energy_kwh * self.grid_carbon_intensity / 1000  # tons
        carbon_value = carbon_saved * self.carbon_price_per_ton
        
        return {
            'num_buildings': num_buildings,
            'shared_energy_kwh': shared_energy_kwh,
            'p2p_benefit': total_p2p_benefit,
            'peak_savings': peak_savings,
            'community_bonus': renewable_bonus,
            'network_benefits': network_benefits,
            'total_annual_benefit': total_annual_benefit,
            'benefit_per_building': benefit_per_building,
            'carbon_saved_tons': carbon_saved,
            'carbon_value': carbon_value,
            'total_with_carbon': total_annual_benefit + carbon_value
        }
    
    def calculate_retrofit_economics(self,
                                    current_label: str,
                                    target_label: str,
                                    building_area_m2: float,
                                    current_consumption_kwh: float) -> Dict:
        """Calculate economics of building retrofit"""
        
        # Retrofit costs per m2 based on improvement level
        label_values = {'G': 7, 'F': 6, 'E': 5, 'D': 4, 'C': 3, 'B': 2, 'A': 1}
        current_val = label_values.get(current_label, 4)
        target_val = label_values.get(target_label, 2)
        improvement_levels = current_val - target_val
        
        # Cost increases with improvement level
        cost_per_m2 = improvement_levels * 150  # EUR/m2
        total_cost = building_area_m2 * cost_per_m2
        
        # Energy savings
        savings_percent = improvement_levels * 0.15  # 15% per level
        annual_savings_kwh = current_consumption_kwh * savings_percent
        annual_savings_eur = annual_savings_kwh * self.electricity_price
        
        # Simple payback
        simple_payback = total_cost / annual_savings_eur if annual_savings_eur > 0 else float('inf')
        
        # NPV over 20 years
        npv = -total_cost
        for year in range(1, 21):
            adjusted_savings = annual_savings_eur * ((1 + self.inflation_rate) ** year)
            pv = adjusted_savings / ((1 + self.discount_rate) ** year)
            npv += pv
        
        # Carbon savings
        carbon_saved_annual = annual_savings_kwh * self.grid_carbon_intensity / 1000
        carbon_value_annual = carbon_saved_annual * self.carbon_price_per_ton
        
        return {
            'current_label': current_label,
            'target_label': target_label,
            'investment': total_cost,
            'annual_energy_savings_kwh': annual_savings_kwh,
            'annual_cost_savings': annual_savings_eur,
            'simple_payback': simple_payback,
            'npv': npv,
            'carbon_saved_tons_annual': carbon_saved_annual,
            'carbon_value_annual': carbon_value_annual,
            'total_annual_benefit': annual_savings_eur + carbon_value_annual
        }
    
    def calculate_grid_investment_deferral(self,
                                          peak_reduction_kw: float,
                                          self_sufficiency_ratio: float,
                                          num_buildings: int) -> Dict:
        """Calculate value of deferred grid investments"""
        
        # Grid upgrade costs (EUR/kW)
        transformer_upgrade_cost = 200
        line_upgrade_cost = 150
        
        # Deferred transformer upgrade
        transformer_deferral = peak_reduction_kw * transformer_upgrade_cost
        
        # Deferred line upgrades (based on self-sufficiency)
        line_capacity_saved = num_buildings * 10 * self_sufficiency_ratio  # kW
        line_deferral = line_capacity_saved * line_upgrade_cost
        
        # Annual value (assuming 10-year deferral)
        annual_value = (transformer_deferral + line_deferral) / 10
        
        # Reduced maintenance
        maintenance_savings = num_buildings * 50 * self_sufficiency_ratio  # EUR/year
        
        return {
            'peak_reduction_kw': peak_reduction_kw,
            'transformer_deferral': transformer_deferral,
            'line_deferral': line_deferral,
            'total_deferral': transformer_deferral + line_deferral,
            'annual_value': annual_value,
            'maintenance_savings': maintenance_savings,
            'total_annual_benefit': annual_value + maintenance_savings
        }
    
    def create_financial_summary(self,
                                solar_roi: Dict,
                                battery_economics: Dict,
                                community_benefits: Dict,
                                grid_deferral: Dict) -> Dict:
        """Create comprehensive financial summary"""
        
        # Total investments
        total_investment = (solar_roi.get('investment', 0) + 
                          battery_economics.get('investment', 0))
        
        # Total annual benefits
        total_annual_benefit = (
            solar_roi.get('net_annual_benefit', 0) +
            battery_economics.get('annual_benefit', 0) +
            community_benefits.get('total_annual_benefit', 0) +
            grid_deferral.get('total_annual_benefit', 0)
        )
        
        # Combined metrics
        overall_payback = total_investment / total_annual_benefit if total_annual_benefit > 0 else float('inf')
        
        # Total carbon savings
        total_carbon = (
            solar_roi.get('carbon_saved_tons', 0) +
            community_benefits.get('carbon_saved_tons', 0)
        )
        
        return {
            'total_investment': total_investment,
            'total_annual_benefit': total_annual_benefit,
            'overall_payback_years': overall_payback,
            'roi_percent': (total_annual_benefit / total_investment * 100) if total_investment > 0 else 0,
            'total_carbon_saved': total_carbon,
            'carbon_value': total_carbon * self.carbon_price_per_ton,
            'breakdown': {
                'solar': solar_roi.get('net_annual_benefit', 0),
                'battery': battery_economics.get('annual_benefit', 0),
                'community': community_benefits.get('total_annual_benefit', 0),
                'grid': grid_deferral.get('total_annual_benefit', 0)
            }
        }
    
    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return using Newton's method"""
        
        # Initial guess
        rate = 0.1
        tolerance = 0.0001
        max_iterations = 100
        
        for _ in range(max_iterations):
            # Calculate NPV and its derivative
            npv = 0
            dnpv = 0
            
            for t, cf in enumerate(cash_flows):
                npv += cf / ((1 + rate) ** t)
                if t > 0:
                    dnpv -= t * cf / ((1 + rate) ** (t + 1))
            
            # Newton's method update
            if abs(dnpv) < tolerance:
                break
                
            new_rate = rate - npv / dnpv
            
            if abs(new_rate - rate) < tolerance:
                return new_rate
                
            rate = new_rate
        
        return rate