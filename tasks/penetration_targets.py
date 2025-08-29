"""
Penetration Targets Manager
Manages and tracks solar penetration goals and progress
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class PenetrationTarget:
    """Solar penetration target definition"""
    target_type: str  # 'area', 'capacity', 'buildings'
    target_value: float  # Target value (percentage or absolute)
    baseline: float  # Current/baseline value
    timeframe_years: int
    sub_targets: Dict[str, float] = field(default_factory=dict)  # Per cluster/LV group
    constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass  
class PenetrationProgress:
    """Tracks progress toward penetration targets"""
    current_value: float
    target_value: float
    percentage_complete: float
    installations_completed: int
    installations_remaining: int
    time_elapsed_years: float
    time_remaining_years: float
    on_track: bool
    projected_completion_date: Optional[datetime] = None
    

class PenetrationTargetsManager:
    """
    Manages solar penetration targets and tracks progress
    """
    
    def __init__(self, config: Dict):
        """
        Initialize targets manager
        
        Args:
            config: Configuration parameters
        """
        self.config = config
        
        # Target types and their calculation methods
        self.target_calculators = {
            'area': self._calculate_area_penetration,
            'capacity': self._calculate_capacity_penetration,
            'buildings': self._calculate_building_penetration,
            'energy': self._calculate_energy_penetration
        }
        
        # Constraints
        self.max_penetration = config.get('max_penetration', 0.8)  # 80% max
        self.min_cluster_penetration = config.get('min_cluster_penetration', 0.1)  # 10% min per cluster
        
        # Tracking
        self.targets = {}
        self.progress_history = []
        
        logger.info("Initialized PenetrationTargetsManager")
    
    def set_target(
        self,
        target_type: str,
        target_value: float,
        timeframe_years: int,
        building_features: torch.Tensor,
        current_solar: torch.Tensor,
        cluster_assignments: Optional[torch.Tensor] = None,
        constraints: Optional[Dict] = None
    ) -> PenetrationTarget:
        """
        Set a new penetration target
        
        Args:
            target_type: Type of target ('area', 'capacity', 'buildings', 'energy')
            target_value: Target value (0-1 for percentage, absolute for counts)
            timeframe_years: Years to achieve target
            building_features: Building feature matrix
            current_solar: Current solar installations
            cluster_assignments: Optional cluster assignments for sub-targets
            constraints: Additional constraints
            
        Returns:
            PenetrationTarget object
        """
        if target_type not in self.target_calculators:
            raise ValueError(f"Unknown target type: {target_type}")
        
        # Calculate baseline
        baseline = self.target_calculators[target_type](
            building_features,
            current_solar,
            None  # No new installations for baseline
        )
        
        # Validate target
        if target_type in ['area', 'buildings', 'energy']:
            # Percentage targets
            if not 0 <= target_value <= 1:
                raise ValueError(f"Percentage target must be between 0 and 1, got {target_value}")
            if target_value > self.max_penetration:
                logger.warning(f"Target {target_value:.1%} exceeds max penetration {self.max_penetration:.1%}")
                target_value = self.max_penetration
        
        # Create sub-targets if clusters provided
        sub_targets = {}
        if cluster_assignments is not None:
            sub_targets = self._distribute_to_clusters(
                target_type,
                target_value,
                building_features,
                current_solar,
                cluster_assignments
            )
        
        # Create target object
        target = PenetrationTarget(
            target_type=target_type,
            target_value=target_value,
            baseline=baseline,
            timeframe_years=timeframe_years,
            sub_targets=sub_targets,
            constraints=constraints or {}
        )
        
        # Store target
        self.targets[target_type] = target
        
        logger.info(f"Set {target_type} target: {target_value:.2f} in {timeframe_years} years")
        logger.info(f"Baseline: {baseline:.2f}")
        
        return target
    
    def check_progress(
        self,
        target_type: str,
        building_features: torch.Tensor,
        current_solar: torch.Tensor,
        new_installations: Optional[List[int]] = None,
        years_elapsed: float = 0
    ) -> PenetrationProgress:
        """
        Check progress toward a target
        
        Args:
            target_type: Type of target to check
            building_features: Building features
            current_solar: Current solar state
            new_installations: New installations to consider
            years_elapsed: Years since target was set
            
        Returns:
            Progress tracking object
        """
        if target_type not in self.targets:
            raise ValueError(f"No target set for {target_type}")
        
        target = self.targets[target_type]
        
        # Calculate current value
        current_value = self.target_calculators[target_type](
            building_features,
            current_solar,
            new_installations
        )
        
        # Calculate progress metrics
        progress_made = current_value - target.baseline
        progress_needed = target.target_value - target.baseline
        percentage_complete = (progress_made / progress_needed * 100) if progress_needed > 0 else 100
        
        # Estimate remaining installations
        if target_type == 'buildings':
            installations_completed = int(current_value - target.baseline)
            installations_remaining = int(target.target_value - current_value)
        else:
            # Estimate based on average installation size
            avg_installation_impact = progress_made / max(len(new_installations) if new_installations else 1, 1)
            installations_completed = len(new_installations) if new_installations else 0
            installations_remaining = int((target.target_value - current_value) / max(avg_installation_impact, 0.01))
        
        # Time tracking
        time_remaining = target.timeframe_years - years_elapsed
        
        # Check if on track
        expected_progress = years_elapsed / target.timeframe_years
        actual_progress = percentage_complete / 100
        on_track = actual_progress >= expected_progress * 0.9  # 90% of expected
        
        # Project completion
        if progress_made > 0 and years_elapsed > 0:
            rate = progress_made / years_elapsed
            years_to_complete = (target.target_value - current_value) / rate
            from datetime import timedelta
            projected_completion = datetime.now() + timedelta(days=years_to_complete * 365)
        else:
            projected_completion = None
        
        progress = PenetrationProgress(
            current_value=current_value,
            target_value=target.target_value,
            percentage_complete=percentage_complete,
            installations_completed=installations_completed,
            installations_remaining=installations_remaining,
            time_elapsed_years=years_elapsed,
            time_remaining_years=time_remaining,
            on_track=on_track,
            projected_completion_date=projected_completion
        )
        
        # Store in history
        self.progress_history.append({
            'timestamp': datetime.now(),
            'target_type': target_type,
            'progress': progress
        })
        
        return progress
    
    def _calculate_area_penetration(
        self,
        building_features: torch.Tensor,
        current_solar: torch.Tensor,
        new_installations: Optional[List[int]] = None
    ) -> float:
        """Calculate roof area penetration"""
        roof_areas = building_features[:, 5].cpu().numpy()  # Roof area feature
        total_area = roof_areas.sum()
        
        # Current solar area (rough estimate)
        solar_mask = current_solar.cpu().numpy() > 0
        used_area = (roof_areas * solar_mask).sum()
        
        # Add new installations
        if new_installations:
            for idx in new_installations:
                if not solar_mask[idx]:
                    used_area += roof_areas[idx]
        
        return used_area / total_area if total_area > 0 else 0
    
    def _calculate_capacity_penetration(
        self,
        building_features: torch.Tensor,
        current_solar: torch.Tensor,
        new_installations: Optional[List[int]] = None
    ) -> float:
        """Calculate capacity penetration in MW"""
        # Current capacity
        current_capacity = current_solar.sum().item() * 10 / 1000  # kWp to MW
        
        # Add new installations
        if new_installations:
            roof_areas = building_features[:, 5].cpu().numpy()
            for idx in new_installations:
                capacity_kwp = min(roof_areas[idx] * 0.15, 10)  # 150W/m², max 10kWp
                current_capacity += capacity_kwp / 1000
        
        return current_capacity
    
    def _calculate_building_penetration(
        self,
        building_features: torch.Tensor,
        current_solar: torch.Tensor,
        new_installations: Optional[List[int]] = None
    ) -> float:
        """Calculate building penetration (percentage with solar)"""
        total_buildings = building_features.shape[0]
        
        # Current installations
        buildings_with_solar = (current_solar.cpu().numpy() > 0).sum()
        
        # Add new installations
        if new_installations:
            buildings_with_solar += len(new_installations)
        
        return buildings_with_solar / total_buildings if total_buildings > 0 else 0
    
    def _calculate_energy_penetration(
        self,
        building_features: torch.Tensor,
        current_solar: torch.Tensor,
        new_installations: Optional[List[int]] = None
    ) -> float:
        """Calculate energy penetration (solar generation / consumption)"""
        # Total consumption
        consumption = building_features[:, 3].sum().item()  # Annual consumption
        
        # Current solar generation
        solar_generation = current_solar.sum().item() * 10 * 1200  # kWp * kWh/kWp/year
        
        # Add new installations
        if new_installations:
            roof_areas = building_features[:, 5].cpu().numpy()
            for idx in new_installations:
                capacity_kwp = min(roof_areas[idx] * 0.15, 10)
                solar_generation += capacity_kwp * 1200
        
        return solar_generation / consumption if consumption > 0 else 0
    
    def _distribute_to_clusters(
        self,
        target_type: str,
        target_value: float,
        building_features: torch.Tensor,
        current_solar: torch.Tensor,
        cluster_assignments: torch.Tensor
    ) -> Dict[str, float]:
        """
        Distribute target across clusters
        
        Args:
            target_type: Type of target
            target_value: Overall target
            building_features: Building features
            current_solar: Current installations
            cluster_assignments: Cluster assignments
            
        Returns:
            Sub-targets per cluster
        """
        sub_targets = {}
        unique_clusters = cluster_assignments.unique()
        
        for cluster_id in unique_clusters:
            cluster_mask = (cluster_assignments == cluster_id).cpu().numpy()
            cluster_id_str = f"cluster_{cluster_id.item()}"
            
            if target_type == 'area':
                # Proportional to cluster roof area
                cluster_area = building_features[cluster_mask, 5].sum().item()
                total_area = building_features[:, 5].sum().item()
                sub_targets[cluster_id_str] = target_value * (cluster_area / total_area)
                
            elif target_type == 'buildings':
                # Proportional to cluster size
                cluster_size = cluster_mask.sum()
                total_size = len(cluster_mask)
                sub_targets[cluster_id_str] = target_value * (cluster_size / total_size)
                
            elif target_type == 'capacity':
                # Proportional to consumption
                cluster_consumption = building_features[cluster_mask, 3].sum().item()
                total_consumption = building_features[:, 3].sum().item()
                sub_targets[cluster_id_str] = target_value * (cluster_consumption / total_consumption)
                
            elif target_type == 'energy':
                # Same target for all clusters
                sub_targets[cluster_id_str] = target_value
            
            # Apply minimum constraint
            if target_type in ['area', 'buildings', 'energy']:
                sub_targets[cluster_id_str] = max(
                    sub_targets[cluster_id_str],
                    self.min_cluster_penetration
                )
        
        return sub_targets
    
    def get_cluster_progress(
        self,
        target_type: str,
        cluster_id: int,
        building_features: torch.Tensor,
        current_solar: torch.Tensor,
        cluster_assignments: torch.Tensor
    ) -> Dict:
        """
        Get progress for a specific cluster
        
        Args:
            target_type: Type of target
            cluster_id: Cluster ID to check
            building_features: Building features
            current_solar: Current installations
            cluster_assignments: Cluster assignments
            
        Returns:
            Cluster-specific progress metrics
        """
        if target_type not in self.targets:
            return {'error': 'No target set'}
        
        target = self.targets[target_type]
        cluster_key = f"cluster_{cluster_id}"
        
        if cluster_key not in target.sub_targets:
            return {'error': 'No sub-target for cluster'}
        
        # Get cluster mask
        cluster_mask = (cluster_assignments == cluster_id).cpu().numpy()
        
        # Calculate cluster-specific penetration
        if target_type == 'area':
            cluster_area = building_features[cluster_mask, 5].sum().item()
            solar_area = (building_features[cluster_mask, 5] * 
                         (current_solar[cluster_mask] > 0)).sum().item()
            current = solar_area / cluster_area if cluster_area > 0 else 0
            
        elif target_type == 'buildings':
            total = cluster_mask.sum()
            with_solar = (current_solar[cluster_mask] > 0).sum().item()
            current = with_solar / total if total > 0 else 0
            
        elif target_type == 'capacity':
            current = (current_solar[cluster_mask].sum().item() * 10) / 1000  # MW
            
        else:  # energy
            consumption = building_features[cluster_mask, 3].sum().item()
            generation = current_solar[cluster_mask].sum().item() * 10 * 1200
            current = generation / consumption if consumption > 0 else 0
        
        target_value = target.sub_targets[cluster_key]
        progress = (current / target_value * 100) if target_value > 0 else 100
        
        return {
            'cluster_id': cluster_id,
            'current': current,
            'target': target_value,
            'progress_percentage': progress,
            'on_track': progress >= (1 / target.timeframe_years * 100 * 0.9)  # 90% of expected
        }
    
    def recommend_next_installations(
        self,
        target_type: str,
        building_features: torch.Tensor,
        current_solar: torch.Tensor,
        cluster_assignments: Optional[torch.Tensor] = None,
        num_recommendations: int = 10
    ) -> List[int]:
        """
        Recommend next buildings for installation to meet targets
        
        Args:
            target_type: Target type to optimize for
            building_features: Building features
            current_solar: Current installations
            cluster_assignments: Optional cluster assignments
            num_recommendations: Number of recommendations
            
        Returns:
            List of building indices to install
        """
        if target_type not in self.targets:
            logger.warning(f"No target set for {target_type}")
            return []
        
        target = self.targets[target_type]
        
        # Get buildings without solar
        no_solar_mask = current_solar.cpu().numpy() < 0.1
        available_indices = np.where(no_solar_mask)[0]
        
        if len(available_indices) == 0:
            return []
        
        # Score buildings based on target type
        scores = np.zeros(len(available_indices))
        
        if target_type == 'area':
            # Prioritize large roofs
            roof_areas = building_features[available_indices, 5].cpu().numpy()
            scores = roof_areas
            
        elif target_type == 'buildings':
            # Equal priority
            scores = np.ones(len(available_indices))
            
        elif target_type == 'capacity':
            # Prioritize high potential capacity
            roof_areas = building_features[available_indices, 5].cpu().numpy()
            orientations = building_features[available_indices, 7].cpu().numpy()
            scores = roof_areas * (1 + orientations * 0.2)
            
        elif target_type == 'energy':
            # Prioritize high consumption buildings
            consumption = building_features[available_indices, 3].cpu().numpy()
            scores = consumption
        
        # Adjust for cluster balance if provided
        if cluster_assignments is not None and target.sub_targets:
            for i, idx in enumerate(available_indices):
                cluster_id = cluster_assignments[idx].item()
                cluster_key = f"cluster_{cluster_id}"
                
                if cluster_key in target.sub_targets:
                    # Check cluster progress
                    cluster_progress = self.get_cluster_progress(
                        target_type,
                        cluster_id,
                        building_features,
                        current_solar,
                        cluster_assignments
                    )
                    
                    # Boost score if cluster is behind
                    if not cluster_progress.get('on_track', True):
                        scores[i] *= 1.5
        
        # Sort and select top N
        sorted_indices = np.argsort(scores)[::-1]
        recommendations = available_indices[sorted_indices[:num_recommendations]]
        
        return recommendations.tolist()
    
    def export_targets(self, filepath: str):
        """Export targets to JSON file"""
        export_data = {}
        
        for target_type, target in self.targets.items():
            export_data[target_type] = {
                'target_value': target.target_value,
                'baseline': target.baseline,
                'timeframe_years': target.timeframe_years,
                'sub_targets': target.sub_targets,
                'constraints': target.constraints,
                'created_at': target.created_at.isoformat()
            }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported targets to {filepath}")
    
    def load_targets(self, filepath: str):
        """Load targets from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for target_type, target_data in data.items():
            self.targets[target_type] = PenetrationTarget(
                target_type=target_type,
                target_value=target_data['target_value'],
                baseline=target_data['baseline'],
                timeframe_years=target_data['timeframe_years'],
                sub_targets=target_data.get('sub_targets', {}),
                constraints=target_data.get('constraints', {}),
                created_at=datetime.fromisoformat(target_data['created_at'])
            )
        
        logger.info(f"Loaded {len(self.targets)} targets from {filepath}")