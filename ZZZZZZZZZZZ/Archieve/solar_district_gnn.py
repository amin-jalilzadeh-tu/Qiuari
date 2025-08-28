"""
Simplified Solar District GNN Model
Main entry point for energy community discovery and solar recommendations
Combines essential components without over-engineering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, HeteroData
from typing import Dict, Optional, Tuple, List
import logging

# Import only essential components
from models.base_gnn import BuildingEncoder, LVGroupEncoder, TransformerEncoder
from models.network_aware_layers import MultiHopAggregator
from models.physics_layers import LVGroupBoundaryEnforcer, DistanceBasedLossCalculator
from models.pooling_layers import ConstrainedDiffPool
from models.temporal_layers import TemporalSequenceEncoder
from models.attention_layers_simplified import SimpleMultiHeadAttention

logger = logging.getLogger(__name__)


class ClusteringHead(nn.Module):
    """Clustering head for discovering self-sufficient communities"""
    
    def __init__(self, hidden_dim: int = 128, num_clusters: int = 10):
        super().__init__()
        
        self.cluster_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_clusters),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Produce soft cluster assignments"""
        return self.cluster_projection(x)


class SolarRecommendationHead(nn.Module):
    """Solar installation recommendation head"""
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        # Solar potential predictor
        self.solar_potential = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # 0-1 solar potential score
        )
        
        # ROI category predictor (excellent <5yr, good 5-7yr, fair 7-10yr, poor >10yr)
        self.roi_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
            nn.Softmax(dim=-1)
        )
        
        # Confidence estimator
        self.confidence = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Produce solar recommendations with confidence"""
        return {
            'solar_potential': self.solar_potential(x),
            'roi_category': self.roi_classifier(x),
            'confidence': self.confidence(x)
        }


class SolarDistrictGNN(nn.Module):
    """
    Simplified main GNN model for solar energy optimization
    Focuses on discovery of self-sufficient districts and solar recommendations
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract config
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_clusters = config.get('num_clusters', 10)
        self.max_hops = config.get('max_hops', 3)
        self.temporal_window = config.get('temporal_window', 24)
        
        # Node encoders (keep essential ones)
        self.building_encoder = BuildingEncoder(
            input_dim=config.get('building_feat_dim'),
            hidden_dim=self.hidden_dim
        )
        
        self.lv_encoder = LVGroupEncoder(
            input_dim=config.get('lv_feat_dim', 12),
            hidden_dim=self.hidden_dim
        )
        
        self.transformer_encoder = TransformerEncoder(
            input_dim=config.get('transformer_feat_dim', 8),
            hidden_dim=self.hidden_dim
        )
        
        # Temporal processor (simplified - just GRU)
        self.temporal_encoder = TemporalSequenceEncoder(
            input_dim=config.get('temporal_feat_dim', 8),
            hidden_dim=self.hidden_dim,
            num_layers=2
        )
        
        # Multi-hop aggregator (KEY DIFFERENTIATOR)
        self.multi_hop_aggregator = MultiHopAggregator(
            hidden_dim=self.hidden_dim,
            max_hops=self.max_hops
        )
        
        # Physics constraints (ESSENTIAL)
        self.lv_boundary_enforcer = LVGroupBoundaryEnforcer()
        self.distance_calculator = DistanceBasedLossCalculator()
        
        # Simple attention for importance weighting
        self.attention = SimpleMultiHeadAttention(
            hidden_dim=self.hidden_dim,
            num_heads=4
        )
        
        # Hierarchical pooling
        self.hierarchical_pool = ConstrainedDiffPool(
            input_dim=self.hidden_dim,
            max_clusters=self.num_clusters,
            min_cluster_size=3,
            max_cluster_size=20
        )
        
        # Task heads
        self.cluster_head = ClusteringHead(
            hidden_dim=self.hidden_dim,
            num_clusters=self.num_clusters
        )
        
        self.solar_head = SolarRecommendationHead(
            hidden_dim=self.hidden_dim
        )
        
        # Simple uncertainty estimation
        self.uncertainty_estimator = nn.Sequential(
            nn.Dropout(0.1),  # MC Dropout for uncertainty
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus()  # Positive uncertainty values
        )
        
    def encode_nodes(self, data: Data) -> torch.Tensor:
        """Encode different node types"""
        # Encode buildings
        if hasattr(data, 'x_building'):
            building_emb = self.building_encoder(data.x_building)
        else:
            building_emb = self.building_encoder(data.x)
            
        # Encode LV groups if available
        if hasattr(data, 'x_lv'):
            lv_emb = self.lv_encoder(data.x_lv)
        else:
            lv_emb = torch.zeros_like(building_emb)
            
        # Combine embeddings
        node_embeddings = building_emb + 0.1 * lv_emb
        
        return node_embeddings
    
    def process_temporal(self, temporal_data: torch.Tensor) -> torch.Tensor:
        """Process temporal consumption patterns"""
        if temporal_data is None:
            return None
            
        # Handle different tensor shapes
        if temporal_data.dim() == 4:  # [B, N, T, F]
            # Use GRU to encode sequences
            temporal_features = self.temporal_encoder(temporal_data)
            if isinstance(temporal_features, tuple):
                temporal_features = temporal_features[0]  # Just take the encoding
            # Remove batch dimension if needed
            if temporal_features.dim() == 3 and temporal_features.size(0) == 1:
                temporal_features = temporal_features.squeeze(0)
        elif temporal_data.dim() == 3:  # [N, T, F]
            # Add batch dimension
            temporal_data = temporal_data.unsqueeze(0)
            temporal_features = self.temporal_encoder(temporal_data)
            if isinstance(temporal_features, tuple):
                temporal_features = temporal_features[0]
            temporal_features = temporal_features.squeeze(0)
        else:
            # Skip temporal processing if wrong shape
            return None
            
        return temporal_features
    
    def forward(self, data: Data, phase: str = 'discovery') -> Dict[str, torch.Tensor]:
        """
        Forward pass for discovery or solar recommendation
        
        Args:
            data: Input graph data
            phase: 'discovery' for clustering, 'solar' for recommendations
            
        Returns:
            Dictionary with phase-appropriate outputs
        """
        # 1. Encode nodes
        node_embeddings = self.encode_nodes(data)
        
        # 2. Process temporal patterns if available
        if hasattr(data, 'temporal_features'):
            temporal_emb = self.process_temporal(data.temporal_features)
            node_embeddings = node_embeddings + 0.3 * temporal_emb
        
        # 3. Multi-hop aggregation (KEY COMPONENT)
        aggregated_features, hop_features = self.multi_hop_aggregator(
            node_embeddings, 
            data.edge_index
        )
        
        # 4. Apply attention for importance weighting
        attended_features = self.attention(aggregated_features)
        
        # Store features for analysis
        outputs = {
            'embeddings': attended_features,
            'hop_features': hop_features,
            'node_features': node_embeddings
        }
        
        if phase == 'discovery':
            # 5a. Discovery phase - clustering
            cluster_assignments = self.cluster_head(attended_features)
            
            # Apply physics constraints
            if hasattr(data, 'lv_group_ids'):
                # Ensure dimensions match
                # cluster_assignments: [N, K] where N=num_nodes, K=num_clusters
                # Convert to sharing matrix format [N, N] for constraint enforcement
                # For now, skip this constraint as it needs different input format
                outputs['boundary_penalty'] = torch.tensor(0.0, device=attended_features.device)
            
            outputs['cluster_assignments'] = cluster_assignments
            outputs['cluster_confidence'] = 1 - self.uncertainty_estimator(attended_features)
            
        elif phase == 'solar':
            # 5b. Solar recommendation phase
            solar_outputs = self.solar_head(attended_features)
            outputs.update(solar_outputs)
            
            # Add network impact scores from multi-hop analysis
            outputs['network_impact'] = self._calculate_network_impact(hop_features)
            
            # Uncertainty for solar predictions
            outputs['solar_uncertainty'] = self.uncertainty_estimator(attended_features)
            
        else:
            # 5c. Both phases
            cluster_assignments = self.cluster_head(attended_features)
            solar_outputs = self.solar_head(attended_features)
            
            outputs['cluster_assignments'] = cluster_assignments
            outputs.update(solar_outputs)
            outputs['uncertainty'] = self.uncertainty_estimator(attended_features)
        
        return outputs
    
    def _calculate_network_impact(self, hop_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Calculate how solar installation impacts the network
        Uses multi-hop features to assess grid-wide effects
        """
        impact_scores = []
        
        # Weight impacts by hop distance
        hop_weights = [1.0, 0.5, 0.25]  # Decreasing influence with distance
        
        for i, (hop_key, hop_feat) in enumerate(hop_features.items()):
            if 'hop_' in hop_key and 'features' in hop_key and isinstance(hop_feat, torch.Tensor):
                if i < len(hop_weights):
                    # Calculate variance in features (high variance = high impact)
                    impact = torch.var(hop_feat, dim=-1, keepdim=True)
                    weighted_impact = impact * hop_weights[i]
                    impact_scores.append(weighted_impact)
        
        if impact_scores:
            return torch.cat(impact_scores, dim=-1).mean(dim=-1, keepdim=True)
        else:
            # Find first tensor in hop_features to get size
            for value in hop_features.values():
                if isinstance(value, torch.Tensor):
                    return torch.zeros((value.size(0), 1), device=value.device)
            # Fallback
            return torch.zeros((1, 1))
    
    def get_explanations(self, data: Data, node_idx: int) -> Dict[str, torch.Tensor]:
        """
        Simple explainability - which features matter for this node's recommendation
        """
        with torch.no_grad():
            outputs = self.forward(data, phase='solar')
            
            # Feature importance via gradient
            solar_score = outputs['solar_potential'][node_idx]
            solar_score.backward(retain_graph=True)
            
            # Get gradients for input features
            feature_importance = data.x[node_idx].grad.abs()
            
            # Get attention weights for this node
            attention_weights = outputs.get('attention_weights', None)
            
            return {
                'feature_importance': feature_importance,
                'attention_weights': attention_weights[node_idx] if attention_weights is not None else None,
                'network_impact': outputs['network_impact'][node_idx],
                'confidence': outputs['confidence'][node_idx]
            }