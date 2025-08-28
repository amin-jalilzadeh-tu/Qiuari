"""
Solar Performance Simulator
Generates synthetic solar performance data for training
"""

import numpy as np
import torch
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class SolarPerformanceSimulator:
    """
    Simulates solar panel performance for semi-supervised learning
    """
    
    def __init__(self, config: Dict):
        """
        Initialize simulator with configuration
        
        Args:
            config: Solar configuration parameters
        """
        self.config = config
        
        # Cost parameters
        self.cost_per_kwp = config.get('cost_per_kwp', 1200)
        self.maintenance_per_kwp = config.get('maintenance_per_kwp_year', 20)
        
        # Energy prices
        self.electricity_price = config.get('electricity_price', 0.25)
        self.feed_in_tariff = config.get('feed_in_tariff', 0.08)
        
        # Performance parameters
        self.annual_generation_per_kwp = config.get('annual_generation_per_kwp', 1200)
        self.degradation_rate = config.get('degradation_rate', 0.005)
        
        # ROI thresholds
        self.roi_categories = config.get('roi_categories', {
            'excellent': 5,
            'good': 7,
            'fair': 10,
            'poor': 15
        })
        
        # Solar generation curve (hourly, normalized)
        self.solar_curve = self._create_solar_curve()
        
        logger.info("Initialized SolarPerformanceSimulator")
    
    def _create_solar_curve(self) -> np.ndarray:
        """
        Create normalized daily solar generation curve
        
        Returns:
            24-hour normalized generation curve
        """
        hours = np.arange(24)
        # Peak at noon, zero at night
        curve = np.maximum(0, np.cos((hours - 12) * np.pi / 12))
        curve[:6] = 0  # No generation before 6am
        curve[20:] = 0  # No generation after 8pm
        
        # Normalize so sum = 1
        if curve.sum() > 0:
            curve = curve / curve.sum()
        
        return curve
    
    def simulate_deployment(
        self,
        building_id: int,
        building_features: Dict,
        cluster_context: Dict,
        temporal_data: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Simulate solar panel deployment on a building
        
        Args:
            building_id: Building identifier
            building_features: Building characteristics
            cluster_context: Cluster membership and complementarity
            temporal_data: Historical consumption patterns
            
        Returns:
            Simulated performance metrics and labels
        """
        # Estimate capacity based on roof area
        roof_area = building_features.get('suitable_roof_area', 50)
        capacity_kwp = min(roof_area * 0.15, 10)  # 150W/m², max 10kWp
        
        # Adjust for orientation if available
        orientation_factor = self._get_orientation_factor(building_features)
        capacity_kwp *= orientation_factor
        
        # Calculate annual generation
        annual_generation = capacity_kwp * self.annual_generation_per_kwp
        
        # Estimate self-consumption based on demand patterns
        if temporal_data is not None:
            self_consumption_rate = self._calculate_self_consumption(
                temporal_data,
                capacity_kwp,
                cluster_context
            )
        else:
            # Base estimate
            self_consumption_rate = self.config.get('self_consumption_base', 0.3)
            
            # Adjust based on cluster complementarity
            if cluster_context.get('avg_complementarity', 0) < -0.5:
                self_consumption_rate += 0.2  # Better consumption in complementary cluster
        
        # Calculate financial metrics
        installation_cost = capacity_kwp * self.cost_per_kwp
        
        # Annual savings
        self_consumed_value = annual_generation * self_consumption_rate * self.electricity_price
        exported_value = annual_generation * (1 - self_consumption_rate) * self.feed_in_tariff
        annual_savings = self_consumed_value + exported_value - (capacity_kwp * self.maintenance_per_kwp)
        
        # Simple payback period
        if annual_savings > 0:
            roi_years = installation_cost / annual_savings
        else:
            roi_years = 100  # Very long payback
        
        # Determine ROI category
        roi_category = self._categorize_roi(roi_years)
        
        # Calculate network benefits (cascade effects)
        network_benefits = self._simulate_network_impact(
            building_id,
            capacity_kwp,
            cluster_context
        )
        
        # Generate time series (90 days of hourly data)
        if temporal_data is not None:
            daily_generation = self._generate_time_series(capacity_kwp, 90)
        else:
            daily_generation = [capacity_kwp * 5] * 90  # Simple estimate: 5 hours equivalent per day
        
        # Calculate confidence based on data quality
        confidence = self._calculate_confidence(building_features, temporal_data)
        
        return {
            'building_id': building_id,
            'capacity_kwp': capacity_kwp,
            'installation_cost': installation_cost,
            'annual_generation_kwh': annual_generation,
            'self_consumption_rate': self_consumption_rate,
            'roi_years': roi_years,
            'roi_category': roi_category,
            'network_benefits': network_benefits,
            'daily_generation_kwh': daily_generation,
            'confidence': confidence,
            'timestamp': datetime.now()
        }
    
    def _get_orientation_factor(self, building_features: Dict) -> float:
        """
        Get orientation factor for solar generation
        
        Args:
            building_features: Building characteristics
            
        Returns:
            Factor between 0.5 and 1.0
        """
        orientation = building_features.get('roof_orientation', 'unknown')
        
        orientation_factors = {
            'south': 1.0,
            'south_east': 0.95,
            'south_west': 0.95,
            'east': 0.85,
            'west': 0.85,
            'flat': 0.9,
            'north_east': 0.7,
            'north_west': 0.7,
            'north': 0.6,
            'unknown': 0.85
        }
        
        return orientation_factors.get(orientation, 0.85)
    
    def _calculate_self_consumption(
        self,
        temporal_data: pd.DataFrame,
        capacity_kwp: float,
        cluster_context: Dict
    ) -> float:
        """
        Calculate self-consumption rate based on demand patterns
        
        Args:
            temporal_data: Historical consumption
            capacity_kwp: Solar capacity
            cluster_context: Cluster information
            
        Returns:
            Self-consumption rate (0-1)
        """
        if temporal_data is None or temporal_data.empty:
            return 0.3
        
        # Get average daily demand profile
        hourly_demand = temporal_data.groupby('hour')['demand_kw'].mean()
        
        # Calculate overlap with solar generation
        solar_generation = self.solar_curve * capacity_kwp * 5  # 5 peak sun hours
        
        # How much solar can be consumed locally
        hourly_consumption = np.minimum(hourly_demand, solar_generation)
        total_consumption = hourly_consumption.sum()
        total_generation = solar_generation.sum()
        
        if total_generation > 0:
            base_rate = total_consumption / total_generation
        else:
            base_rate = 0.3
        
        # Adjust for cluster energy sharing
        if cluster_context.get('cluster_size', 1) > 1:
            # More buildings in cluster = higher consumption potential
            cluster_factor = min(1.5, 1 + cluster_context['cluster_size'] * 0.05)
            base_rate = min(0.9, base_rate * cluster_factor)
        
        return base_rate
    
    def _categorize_roi(self, roi_years: float) -> str:
        """
        Categorize ROI into performance categories
        
        Args:
            roi_years: Payback period in years
            
        Returns:
            Category string
        """
        if roi_years < self.roi_categories['excellent']:
            return 'excellent'
        elif roi_years < self.roi_categories['good']:
            return 'good'
        elif roi_years < self.roi_categories['fair']:
            return 'fair'
        else:
            return 'poor'
    
    def _simulate_network_impact(
        self,
        building_id: int,
        capacity_kwp: float,
        cluster_context: Dict
    ) -> Dict:
        """
        Simulate network-wide impact of solar installation
        
        Args:
            building_id: Building identifier
            capacity_kwp: Solar capacity
            cluster_context: Cluster information
            
        Returns:
            Network impact metrics
        """
        # Base impact on capacity
        direct_impact = capacity_kwp * 0.8  # 80% of capacity reduces grid load
        
        # Cascade effects based on network position
        centrality = cluster_context.get('building_centrality', {}).get(building_id, 0.5)
        cascade_multiplier = 1 + centrality  # More central = more impact
        
        # Number of buildings affected
        cluster_size = cluster_context.get('cluster_size', 1)
        affected_buildings = min(cluster_size - 1, int(cascade_multiplier * 5))
        
        # Peak reduction potential
        peak_reduction_kw = capacity_kwp * 0.6 * cascade_multiplier
        
        # Grid loss reduction (rough estimate)
        loss_reduction_kwh = capacity_kwp * 100 * centrality  # Annual
        
        return {
            'direct_impact_kw': direct_impact,
            'cascade_multiplier': cascade_multiplier,
            'affected_buildings': affected_buildings,
            'peak_reduction_kw': peak_reduction_kw,
            'loss_reduction_kwh_annual': loss_reduction_kwh,
            'transformer_relief_percent': min(20, capacity_kwp * 2)  # Max 20% relief
        }
    
    def _generate_time_series(self, capacity_kwp: float, days: int) -> List[float]:
        """
        Generate synthetic daily generation time series
        
        Args:
            capacity_kwp: Solar capacity
            days: Number of days to simulate
            
        Returns:
            Daily generation values
        """
        daily_generation = []
        
        for day in range(days):
            # Base generation
            base_generation = capacity_kwp * 5  # 5 peak sun hours
            
            # Add weather variation
            weather_factor = np.random.normal(1.0, 0.2)  # 20% variation
            weather_factor = np.clip(weather_factor, 0.3, 1.3)
            
            # Seasonal variation (simple sine wave)
            seasonal_factor = 0.8 + 0.4 * np.sin(day * 2 * np.pi / 365)
            
            daily_kwh = base_generation * weather_factor * seasonal_factor
            daily_generation.append(max(0, daily_kwh))
        
        return daily_generation
    
    def _calculate_confidence(
        self,
        building_features: Dict,
        temporal_data: Optional[pd.DataFrame]
    ) -> float:
        """
        Calculate confidence in simulation
        
        Args:
            building_features: Building characteristics
            temporal_data: Historical data availability
            
        Returns:
            Confidence score (0-1)
        """
        confidence = 0.5  # Base confidence for simulation
        
        # Increase confidence for known features
        if building_features.get('suitable_roof_area'):
            confidence += 0.1
        if building_features.get('roof_orientation'):
            confidence += 0.1
        if building_features.get('energy_label'):
            confidence += 0.05
        
        # Increase confidence for available temporal data
        if temporal_data is not None and not temporal_data.empty:
            data_days = len(temporal_data) / 24 if len(temporal_data) > 0 else 0
            confidence += min(0.2, data_days / 100)  # Max 0.2 for 100+ days
        
        return min(confidence, 0.9)  # Cap at 0.9 for simulated data
    
    def generate_batch_labels(
        self,
        recommendations: List[int],
        building_features: Dict,
        cluster_contexts: Dict,
        temporal_data: Optional[pd.DataFrame] = None
    ) -> Dict[int, Dict]:
        """
        Generate labels for multiple solar recommendations
        
        Args:
            recommendations: List of building IDs
            building_features: Features for all buildings
            cluster_contexts: Cluster information for buildings
            temporal_data: Historical consumption data
            
        Returns:
            Dictionary of labels by building ID
        """
        labels = {}
        
        for building_id in recommendations:
            features = building_features.get(building_id, {})
            context = cluster_contexts.get(building_id, {})
            
            performance = self.simulate_deployment(
                building_id,
                features,
                context,
                temporal_data
            )
            
            labels[building_id] = {
                'roi_category': performance['roi_category'],
                'roi_years': performance['roi_years'],
                'confidence': performance['confidence'],
                'network_benefits': performance['network_benefits'],
                'capacity_kwp': performance['capacity_kwp']
            }
            
            logger.debug(f"Generated label for building {building_id}: "
                        f"{performance['roi_category']} ({performance['roi_years']:.1f} years)")
        
        return labels