"""
Solar Performance Labeling System
Generates labels from actual solar installation performance
Key component for semi-supervised learning loop
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


@dataclass
class SolarInstallation:
    """Record of a solar installation and its performance"""
    building_id: int
    cluster_id: int
    installation_date: datetime
    capacity_kw: float
    installation_cost: float
    
    # Performance metrics (measured after deployment)
    daily_generation_kwh: List[float]  # Actual daily generation
    self_consumption_rate: float  # Percentage consumed locally
    export_revenue: float  # Revenue from exported energy
    peak_reduction_percent: float  # Peak demand reduction achieved
    
    # Calculated metrics
    roi_years: Optional[float] = None
    performance_label: Optional[str] = None
    confidence: Optional[float] = None


class SolarPerformanceLabeler:
    """
    Generates training labels from actual solar deployment performance
    Critical for semi-supervised learning improvement
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize labeler
        
        Args:
            config: Configuration parameters
        """
        self.config = config if config is not None else {}
        
        # ROI thresholds for labeling
        self.roi_thresholds = {
            'excellent': 5.0,  # < 5 years payback
            'good': 7.0,       # 5-7 years
            'fair': 10.0,      # 7-10 years
            'poor': float('inf')  # > 10 years
        }
        
        # Performance tracking
        self.installations = {}
        self.labeled_buildings = set()
        self.label_statistics = {
            'excellent': 0,
            'good': 0,
            'fair': 0,
            'poor': 0
        }
        
        # Minimum observation period before labeling
        self.min_observation_days = self.config.get('min_observation_days', 90)
        
    def add_installation(self, installation: SolarInstallation):
        """
        Register a new solar installation for tracking
        
        Args:
            installation: Solar installation record
        """
        self.installations[installation.building_id] = installation
        logger.info(f"Added installation for building {installation.building_id}: "
                   f"{installation.capacity_kw}kW, Cost: €{installation.installation_cost}")
    
    def calculate_roi(self, installation: SolarInstallation) -> float:
        """
        Calculate return on investment in years
        
        Args:
            installation: Solar installation to evaluate
            
        Returns:
            ROI in years
        """
        # Annual generation estimate
        if len(installation.daily_generation_kwh) >= self.min_observation_days:
            # Use actual data
            avg_daily_generation = np.mean(installation.daily_generation_kwh)
            annual_generation = avg_daily_generation * 365
        else:
            # Estimate if not enough data
            logger.warning(f"Insufficient data for building {installation.building_id}, using estimates")
            annual_generation = installation.capacity_kw * 1200  # Rough estimate: 1200 kWh/kWp/year
        
        # Calculate savings
        electricity_price = self.config.get('electricity_price', 0.25)  # €/kWh
        feed_in_tariff = self.config.get('feed_in_tariff', 0.08)  # €/kWh
        
        # Energy consumed locally saves full electricity price
        self_consumed_value = annual_generation * installation.self_consumption_rate * electricity_price
        
        # Exported energy earns feed-in tariff
        exported_value = annual_generation * (1 - installation.self_consumption_rate) * feed_in_tariff
        
        # Total annual savings
        annual_savings = self_consumed_value + exported_value + installation.export_revenue
        
        # Subtract maintenance costs
        maintenance_cost = installation.capacity_kw * self.config.get('maintenance_cost_per_kw', 20)
        net_annual_savings = annual_savings - maintenance_cost
        
        # Calculate simple payback period
        if net_annual_savings > 0:
            roi_years = installation.installation_cost / net_annual_savings
        else:
            roi_years = float('inf')
        
        return roi_years
    
    def label_installation(
        self,
        building_id: int,
        force: bool = False
    ) -> Tuple[str, float]:
        """
        Generate label for a solar installation based on performance
        
        Args:
            building_id: Building to label
            force: Force labeling even with insufficient data
            
        Returns:
            (label, confidence) tuple
        """
        if building_id not in self.installations:
            raise ValueError(f"No installation record for building {building_id}")
        
        installation = self.installations[building_id]
        
        # Check if enough observation time has passed
        days_observed = len(installation.daily_generation_kwh)
        if days_observed < self.min_observation_days and not force:
            logger.info(f"Building {building_id}: Only {days_observed} days observed, "
                       f"need {self.min_observation_days}")
            return 'unknown', 0.0
        
        # Calculate ROI
        roi_years = self.calculate_roi(installation)
        installation.roi_years = roi_years
        
        # Determine label based on ROI
        if roi_years < self.roi_thresholds['excellent']:
            label = 'excellent'
        elif roi_years < self.roi_thresholds['good']:
            label = 'good'
        elif roi_years < self.roi_thresholds['fair']:
            label = 'fair'
        else:
            label = 'poor'
        
        # Calculate confidence based on data quality
        confidence = min(days_observed / self.min_observation_days, 1.0)
        
        # Consider other factors for confidence
        if installation.peak_reduction_percent > 0.3:
            confidence *= 1.1  # Boost confidence if good peak reduction
        if installation.self_consumption_rate > 0.7:
            confidence *= 1.1  # Boost confidence if high self-consumption
        
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        # Store label
        installation.performance_label = label
        installation.confidence = confidence
        self.labeled_buildings.add(building_id)
        self.label_statistics[label] += 1
        
        logger.info(f"Labeled building {building_id}: {label} (ROI: {roi_years:.1f} years, "
                   f"Confidence: {confidence:.2f})")
        
        return label, confidence
    
    def generate_cluster_labels(
        self,
        cluster_metrics: Dict[int, Dict[str, float]]
    ) -> Dict[int, str]:
        """
        Generate labels for discovered clusters based on their metrics
        These are automatic labels, not from real deployment
        
        Args:
            cluster_metrics: Metrics for each cluster
            
        Returns:
            Cluster labels
        """
        cluster_labels = {}
        
        for cluster_id, metrics in cluster_metrics.items():
            # Evaluate cluster quality
            self_sufficiency = metrics.get('self_sufficiency', 0)
            complementarity = metrics.get('complementarity', 0)
            peak_reduction = metrics.get('peak_reduction', 0)
            network_losses = metrics.get('network_losses', 1)
            
            # Scoring system
            score = 0
            
            # Self-sufficiency contribution (max 40 points)
            score += min(self_sufficiency * 40, 40)
            
            # Complementarity contribution (max 30 points)
            # Negative correlation is good
            if complementarity < 0:
                score += min(abs(complementarity) * 30, 30)
            
            # Peak reduction contribution (max 20 points)
            score += min(peak_reduction * 20, 20)
            
            # Network losses penalty (max -10 points)
            score -= min(network_losses * 10, 10)
            
            # Determine label based on score
            if score >= 70:
                label = 'excellent'
            elif score >= 50:
                label = 'good'
            elif score >= 30:
                label = 'fair'
            else:
                label = 'poor'
            
            cluster_labels[cluster_id] = label
            
            logger.info(f"Cluster {cluster_id} labeled as {label} (score: {score:.1f})")
        
        return cluster_labels
    
    def get_training_labels(
        self,
        min_confidence: float = 0.7
    ) -> Dict[int, Tuple[str, float]]:
        """
        Get all high-confidence labels for training
        
        Args:
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dict of building_id -> (label, confidence)
        """
        training_labels = {}
        
        for building_id in self.labeled_buildings:
            installation = self.installations[building_id]
            
            if installation.confidence >= min_confidence:
                training_labels[building_id] = (
                    installation.performance_label,
                    installation.confidence
                )
        
        logger.info(f"Retrieved {len(training_labels)} high-confidence labels for training")
        
        return training_labels
    
    def update_performance_data(
        self,
        building_id: int,
        new_generation_data: List[float],
        new_self_consumption: Optional[float] = None,
        new_peak_reduction: Optional[float] = None
    ):
        """
        Update performance data for an installation
        
        Args:
            building_id: Building to update
            new_generation_data: New daily generation values
            new_self_consumption: Updated self-consumption rate
            new_peak_reduction: Updated peak reduction
        """
        if building_id not in self.installations:
            raise ValueError(f"No installation record for building {building_id}")
        
        installation = self.installations[building_id]
        
        # Append new generation data
        installation.daily_generation_kwh.extend(new_generation_data)
        
        # Update rates if provided
        if new_self_consumption is not None:
            installation.self_consumption_rate = new_self_consumption
        if new_peak_reduction is not None:
            installation.peak_reduction_percent = new_peak_reduction
        
        # Re-label if we now have enough data
        if len(installation.daily_generation_kwh) >= self.min_observation_days:
            if installation.performance_label is None:
                self.label_installation(building_id)
            elif len(installation.daily_generation_kwh) % 30 == 0:
                # Re-label every 30 days to update
                old_label = installation.performance_label
                new_label, confidence = self.label_installation(building_id)
                if old_label != new_label:
                    logger.info(f"Building {building_id} label changed: {old_label} -> {new_label}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get labeling statistics
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_installations': len(self.installations),
            'labeled_buildings': len(self.labeled_buildings),
            'label_distribution': self.label_statistics,
            'average_roi': None,
            'best_performer': None,
            'worst_performer': None
        }
        
        # Calculate average ROI
        rois = [inst.roi_years for inst in self.installations.values() 
                if inst.roi_years is not None and inst.roi_years < float('inf')]
        
        if rois:
            stats['average_roi'] = np.mean(rois)
            
            # Find best and worst performers
            best_roi = min(rois)
            worst_roi = max(rois)
            
            for bid, inst in self.installations.items():
                if inst.roi_years == best_roi:
                    stats['best_performer'] = {
                        'building_id': bid,
                        'roi_years': best_roi,
                        'label': inst.performance_label
                    }
                if inst.roi_years == worst_roi:
                    stats['worst_performer'] = {
                        'building_id': bid,
                        'roi_years': worst_roi,
                        'label': inst.performance_label
                    }
        
        return stats
    
    def export_labels(self, filepath: str):
        """
        Export labels to file for persistence
        
        Args:
            filepath: Path to save labels
        """
        labels_data = []
        
        for building_id, installation in self.installations.items():
            if installation.performance_label is not None:
                labels_data.append({
                    'building_id': building_id,
                    'cluster_id': installation.cluster_id,
                    'label': installation.performance_label,
                    'roi_years': installation.roi_years,
                    'confidence': installation.confidence,
                    'capacity_kw': installation.capacity_kw,
                    'self_consumption_rate': installation.self_consumption_rate,
                    'peak_reduction_percent': installation.peak_reduction_percent,
                    'days_observed': len(installation.daily_generation_kwh)
                })
        
        df = pd.DataFrame(labels_data)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(labels_data)} labels to {filepath}")
    
    def import_labels(self, filepath: str):
        """
        Import previously generated labels
        
        Args:
            filepath: Path to labels file
        """
        df = pd.read_csv(filepath)
        
        for _, row in df.iterrows():
            # Recreate installation record
            installation = SolarInstallation(
                building_id=row['building_id'],
                cluster_id=row['cluster_id'],
                installation_date=datetime.now(),  # Placeholder
                capacity_kw=row['capacity_kw'],
                installation_cost=0,  # Not stored
                daily_generation_kwh=[],  # Not stored
                self_consumption_rate=row['self_consumption_rate'],
                export_revenue=0,  # Not stored
                peak_reduction_percent=row['peak_reduction_percent'],
                roi_years=row['roi_years'],
                performance_label=row['label'],
                confidence=row['confidence']
            )
            
            self.installations[row['building_id']] = installation
            self.labeled_buildings.add(row['building_id'])
            self.label_statistics[row['label']] += 1
        
        logger.info(f"Imported {len(df)} labels from {filepath}")