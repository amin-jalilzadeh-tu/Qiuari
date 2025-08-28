"""
Explainability Generator for Energy GNN
Generates human-interpretable explanations for model decisions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ExplainabilityGenerator:
    """
    Generates explanations for GNN decisions
    """
    
    def __init__(self, config: Dict):
        """
        Initialize explainability generator
        
        Args:
            config: Configuration for explainability
        """
        self.feature_names = config.get('feature_names', [
            'floor_area', 'num_dwellings', 'construction_year', 'has_solar',
            'num_shared_walls', 'x_coord', 'y_coord', 'energy_label',
            'suitable_roof_area', 'peak_electricity_kw', 'peak_heating_kw',
            'has_battery', 'has_heat_pump', 'annual_consumption'
        ])
        
        self.importance_threshold = config.get('importance_threshold', 0.1)
        
        logger.info("Initialized ExplainabilityGenerator")
    
    def explain_cluster_assignment(
        self,
        model: nn.Module,
        data: Dict,
        building_id: int,
        cluster_id: int
    ) -> Dict:
        """
        Explain why a building was assigned to a specific cluster
        
        Args:
            model: GNN model
            data: Input data
            building_id: Building to explain
            cluster_id: Assigned cluster
            
        Returns:
            Explanation dictionary
        """
        model.eval()
        
        # Get building features
        building_features = data['building'].x[building_id]
        
        # Get cluster assignment probabilities
        with torch.no_grad():
            outputs = model(data, task='clustering')
            if 'cluster_logits' in outputs:
                logits = outputs['cluster_logits']
            else:
                logits = outputs.get('cluster_assignments', torch.zeros(1))
            
            probs = torch.softmax(logits[building_id], dim=0)
        
        # Find neighbors in the same cluster
        # Check if building-to-building edges exist, otherwise create from cable groups
        if ('building', 'connected_to', 'building') in data.edge_types:
            edge_index = data[('building', 'connected_to', 'building')].edge_index
        elif ('building', 'connected_to', 'cable_group') in data.edge_types:
            # Infer building connections from shared cable groups
            b2c_edges = data[('building', 'connected_to', 'cable_group')].edge_index
            # Find buildings that share cable groups
            edge_list = []
            cable_groups = {}
            for i in range(b2c_edges.shape[1]):
                b_idx = b2c_edges[0, i].item()
                c_idx = b2c_edges[1, i].item()
                if c_idx not in cable_groups:
                    cable_groups[c_idx] = []
                cable_groups[c_idx].append(b_idx)
            
            # Create edges between buildings in same cable group
            for buildings in cable_groups.values():
                for i in range(len(buildings)):
                    for j in range(i+1, len(buildings)):
                        edge_list.append([buildings[i], buildings[j]])
                        edge_list.append([buildings[j], buildings[i]])
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
        else:
            # No connectivity information available
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        
        neighbors = self._get_neighbors(building_id, edge_index)
        
        if len(neighbors) > 0:
            # Ensure neighbors tensor is on same device as logits
            neighbors = neighbors.to(logits.device)
            neighbor_clusters = logits[neighbors].argmax(dim=-1)
            same_cluster_neighbors = neighbors[neighbor_clusters == cluster_id]
        else:
            same_cluster_neighbors = torch.tensor([], dtype=torch.long)
        
        # Calculate feature importance using gradient
        feature_importance = self._calculate_feature_importance(
            model, data, building_id, 'clustering'
        )
        
        # Identify complementary buildings
        complementarity_scores = self._calculate_complementarity_with_cluster(
            building_features, 
            data['building'].x[same_cluster_neighbors] if len(same_cluster_neighbors) > 0 else None
        )
        
        # Generate explanation
        explanation = {
            'building_id': building_id,
            'cluster_id': cluster_id,
            'confidence': probs[cluster_id].item(),
            'top_features': self._get_top_features(feature_importance),
            'neighbors_in_cluster': len(same_cluster_neighbors),
            'total_neighbors': len(neighbors),
            'complementarity': {
                'avg_score': complementarity_scores.mean().item() if complementarity_scores is not None else 0,
                'interpretation': self._interpret_complementarity(complementarity_scores)
            },
            'summary': self._generate_cluster_summary(
                building_features,
                cluster_id,
                probs[cluster_id].item(),
                complementarity_scores
            )
        }
        
        return explanation
    
    def explain_solar_recommendation(
        self,
        model: nn.Module,
        data: Dict,
        building_id: int,
        cascade_analyzer
    ) -> Dict:
        """
        Explain solar panel recommendation for a building
        
        Args:
            model: GNN model
            data: Input data
            building_id: Building to explain
            cascade_analyzer: Solar cascade analyzer
            
        Returns:
            Solar recommendation explanation
        """
        model.eval()
        
        # Get building features
        building_features = data['building'].x[building_id]
        
        # Check if building-to-building edges exist, otherwise create from cable groups
        if ('building', 'connected_to', 'building') in data.edge_types:
            edge_index = data[('building', 'connected_to', 'building')].edge_index
        elif ('building', 'connected_to', 'cable_group') in data.edge_types:
            # Infer building connections from shared cable groups
            b2c_edges = data[('building', 'connected_to', 'cable_group')].edge_index
            # Find buildings that share cable groups
            edge_list = []
            cable_groups = {}
            for i in range(b2c_edges.shape[1]):
                b_idx = b2c_edges[0, i].item()
                c_idx = b2c_edges[1, i].item()
                if c_idx not in cable_groups:
                    cable_groups[c_idx] = []
                cable_groups[c_idx].append(b_idx)
            
            # Create edges between buildings in same cable group
            for buildings in cable_groups.values():
                for i in range(len(buildings)):
                    for j in range(i+1, len(buildings)):
                        edge_list.append([buildings[i], buildings[j]])
                        edge_list.append([buildings[j], buildings[i]])
            
            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t()
            else:
                edge_index = torch.zeros(2, 0, dtype=torch.long)
        else:
            # No connectivity information available
            edge_index = torch.zeros(2, 0, dtype=torch.long)
        
        # Get solar score
        with torch.no_grad():
            outputs = model(data, task='solar')
            if isinstance(outputs, dict):
                solar_score = outputs.get('solar', torch.zeros(1))[building_id]
            else:
                solar_score = outputs[building_id] if len(outputs.shape) > 0 else outputs
        
        # Calculate cascade effects
        cascade_effects = cascade_analyzer.analyze_cascade(
            building_id,
            building_features[8].item(),  # suitable_roof_area as proxy for capacity
            data,
            edge_index,
            data['building'].x
        )
        
        # Feature importance
        feature_importance = self._calculate_feature_importance(
            model, data, building_id, 'solar'
        )
        
        # Local factors
        local_factors = {
            'roof_area': building_features[8].item(),
            'energy_label': chr(int(building_features[7].item()) + ord('A')),
            'has_battery': bool(building_features[11].item()),
            'annual_consumption': building_features[9].item() * 8760  # peak * hours
        }
        
        # Network impact
        network_impact = {
            'affected_buildings': cascade_effects['total_affected'],
            'peak_reduction_kw': cascade_effects['network_benefits']['peak_reduction_kw'],
            'voltage_improvement': cascade_effects['network_benefits']['voltage_improvement_percent'],
            'network_value': cascade_effects['network_benefits']['network_value_euro']
        }
        
        explanation = {
            'building_id': building_id,
            'recommendation_score': solar_score.item(),
            'local_factors': local_factors,
            'network_impact': network_impact,
            'cascade_radius': cascade_effects['cascade_radius'],
            'top_features': self._get_top_features(feature_importance),
            'recommendation': 'Highly Recommended' if solar_score > 0.7 else 'Recommended' if solar_score > 0.4 else 'Consider',
            'summary': self._generate_solar_summary(
                local_factors,
                network_impact,
                solar_score.item()
            )
        }
        
        return explanation
    
    def explain_energy_flow(
        self,
        model: nn.Module,
        data: Dict,
        source_building: int,
        target_building: int,
        timestep: int
    ) -> Dict:
        """
        Explain energy flow between buildings
        
        Args:
            model: GNN model
            data: Input data
            source_building: Energy source building
            target_building: Energy target building
            timestep: Time of energy flow
            
        Returns:
            Energy flow explanation
        """
        # Get building features
        source_features = data['building'].x[source_building]
        target_features = data['building'].x[target_building]
        
        # Calculate complementarity
        time_factor = np.sin(2 * np.pi * timestep / 24)  # Daily pattern
        
        source_generation = source_features[3].item() * 5 * max(0, time_factor)  # Solar generation estimate
        target_demand = target_features[9].item() * (1 + 0.3 * time_factor)  # Demand with daily variation
        
        # Energy flow potential
        flow_potential = min(source_generation, target_demand)
        
        # Distance factor
        distance = torch.norm(source_features[5:7] - target_features[5:7]).item()
        efficiency = max(0.8, 1.0 - distance / 1000)  # Loss per km
        
        # LV group check
        same_lv = data['building'].lv_group_ids[source_building] == data['building'].lv_group_ids[target_building]
        
        explanation = {
            'source': source_building,
            'target': target_building,
            'timestep': timestep,
            'flow_potential_kw': flow_potential,
            'efficiency': efficiency,
            'distance_m': distance,
            'same_lv_group': same_lv,
            'feasible': same_lv and flow_potential > 0,
            'source_type': 'Solar' if source_features[3] > 0 else 'Grid',
            'target_profile': 'Residential' if target_features[1] > 0 else 'Commercial',
            'summary': self._generate_flow_summary(
                flow_potential,
                efficiency,
                same_lv,
                timestep
            )
        }
        
        return explanation
    
    def _calculate_feature_importance(
        self,
        model: nn.Module,
        data: Dict,
        node_id: int,
        task: str
    ) -> torch.Tensor:
        """
        Calculate feature importance using gradient-based method
        
        Args:
            model: GNN model
            data: Input data
            node_id: Node to analyze
            task: Task type
            
        Returns:
            Feature importance scores
        """
        try:
            # Set model to eval but enable gradient computation
            model.eval()
            
            # Enable gradients for input
            original_features = data['building'].x.clone()
            data['building'].x = original_features.detach().requires_grad_(True)
            
            # Forward pass with gradient tracking
            with torch.enable_grad():
                outputs = model(data, task=task)
                
                if task == 'clustering':
                    if 'cluster_logits' in outputs:
                        target = outputs['cluster_logits'][node_id].max()
                    else:
                        target = outputs.get('cluster_assignments', torch.zeros(1))[node_id].max()
                else:
                    target = outputs.get(task, outputs)[node_id] if isinstance(outputs, dict) else outputs[node_id]
                
                # Check if target has gradient
                if not target.requires_grad:
                    # Return zero importance if gradients not available
                    return torch.zeros(original_features.shape[1])
                
                # Backward pass
                model.zero_grad()
                target.backward(retain_graph=True)
            
            # Get gradients
            if data['building'].x.grad is not None:
                gradients = data['building'].x.grad[node_id].abs()
            else:
                gradients = torch.zeros(original_features.shape[1])
            
            # Restore original features
            data['building'].x = original_features
            
            return gradients
            
        except Exception as e:
            # Return zero importance if calculation fails
            logger.warning(f"Feature importance calculation failed: {e}")
            return torch.zeros(data['building'].x.shape[1])
    
    def _get_neighbors(self, node_id: int, edge_index: torch.Tensor) -> torch.Tensor:
        """Get neighbors of a node"""
        mask = edge_index[0] == node_id
        return edge_index[1, mask]
    
    def _calculate_complementarity_with_cluster(
        self,
        building_features: torch.Tensor,
        cluster_features: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Calculate complementarity with cluster members"""
        if cluster_features is None or len(cluster_features) == 0:
            return None
        
        # Simple complementarity: negative correlation in consumption patterns
        # Using peak demands as proxy
        building_pattern = building_features[9:11]  # electricity and heating peaks
        cluster_patterns = cluster_features[:, 9:11]
        
        # Calculate correlation
        complementarity = -torch.cosine_similarity(
            building_pattern.unsqueeze(0),
            cluster_patterns,
            dim=1
        )
        
        return complementarity
    
    def _get_top_features(self, importance: torch.Tensor, top_k: int = 5) -> List[Dict]:
        """Get top important features"""
        top_indices = importance.argsort(descending=True)[:top_k]
        
        top_features = []
        for idx in top_indices:
            if importance[idx] > self.importance_threshold:
                top_features.append({
                    'feature': self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}',
                    'importance': importance[idx].item()
                })
        
        return top_features
    
    def _interpret_complementarity(self, scores: Optional[torch.Tensor]) -> str:
        """Interpret complementarity scores"""
        if scores is None:
            return "No cluster members to compare"
        
        avg_score = scores.mean().item()
        
        if avg_score < -0.5:
            return "Highly complementary - excellent for energy sharing"
        elif avg_score < -0.2:
            return "Moderately complementary - good for energy sharing"
        elif avg_score < 0.2:
            return "Neutral - limited sharing potential"
        else:
            return "Similar profiles - compete for same resources"
    
    def _generate_cluster_summary(
        self,
        features: torch.Tensor,
        cluster_id: int,
        confidence: float,
        complementarity: Optional[torch.Tensor]
    ) -> str:
        """Generate human-readable cluster explanation"""
        
        summary = f"Building assigned to Cluster {cluster_id} with {confidence:.1%} confidence. "
        
        # Key reasons
        reasons = []
        
        if features[8] > 70:  # Large roof area
            reasons.append("large solar potential")
        
        if features[7] < 2:  # Good energy label (A or B)
            reasons.append("energy efficient")
        
        if complementarity is not None and complementarity.mean() < -0.3:
            reasons.append("complementary consumption patterns")
        
        if features[11] > 0:  # Has battery
            reasons.append("storage capability")
        
        if reasons:
            summary += f"Key factors: {', '.join(reasons)}. "
        
        summary += "This clustering enables efficient energy sharing within the LV network."
        
        return summary
    
    def _generate_solar_summary(
        self,
        local_factors: Dict,
        network_impact: Dict,
        score: float
    ) -> str:
        """Generate human-readable solar recommendation"""
        
        if score > 0.7:
            recommendation = "Highly recommended for solar installation. "
        elif score > 0.4:
            recommendation = "Good candidate for solar installation. "
        else:
            recommendation = "Solar installation should be considered. "
        
        summary = recommendation
        
        # Local benefits
        if local_factors['roof_area'] > 70:
            summary += f"Excellent roof area ({local_factors['roof_area']}m²). "
        
        # Network benefits
        if network_impact['affected_buildings'] > 10:
            summary += f"Installation would benefit {network_impact['affected_buildings']} nearby buildings. "
        
        if network_impact['peak_reduction_kw'] > 5:
            summary += f"Reduces peak load by {network_impact['peak_reduction_kw']:.1f}kW. "
        
        summary += f"Estimated network value: €{network_impact['network_value']:.0f}."
        
        return summary
    
    def _generate_flow_summary(
        self,
        flow_potential: float,
        efficiency: float,
        same_lv: bool,
        timestep: int
    ) -> str:
        """Generate human-readable energy flow explanation"""
        
        hour = timestep % 24
        time_of_day = "morning" if 6 <= hour < 12 else "afternoon" if 12 <= hour < 18 else "evening" if 18 <= hour < 22 else "night"
        
        if not same_lv:
            return f"Energy sharing not possible - buildings in different LV groups."
        
        if flow_potential > 0:
            return f"Can share {flow_potential:.1f}kW during {time_of_day} with {efficiency:.0%} efficiency."
        else:
            return f"No energy available for sharing at this time ({time_of_day})."
    
    def generate_explainability_report(
        self,
        cluster_explanations: List[Dict],
        solar_explanations: List[Dict]
    ) -> str:
        """
        Generate comprehensive explainability report
        
        Args:
            cluster_explanations: List of cluster explanations
            solar_explanations: List of solar explanations
            
        Returns:
            Formatted report
        """
        report = []
        report.append("=" * 60)
        report.append("EXPLAINABILITY REPORT")
        report.append("=" * 60)
        
        # Clustering explanations
        report.append("\n1. CLUSTERING EXPLANATIONS")
        report.append("-" * 40)
        
        for exp in cluster_explanations[:3]:  # Top 3
            report.append(f"\nBuilding {exp['building_id']} → Cluster {exp['cluster_id']}")
            report.append(f"Confidence: {exp['confidence']:.1%}")
            report.append(f"Complementarity: {exp['complementarity']['interpretation']}")
            report.append(f"Summary: {exp['summary']}")
        
        # Solar explanations
        report.append("\n2. SOLAR RECOMMENDATIONS")
        report.append("-" * 40)
        
        for exp in solar_explanations[:3]:  # Top 3
            report.append(f"\nBuilding {exp['building_id']}")
            report.append(f"Recommendation: {exp['recommendation']}")
            report.append(f"Network impact: {exp['network_impact']['affected_buildings']} buildings")
            report.append(f"Summary: {exp['summary']}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)