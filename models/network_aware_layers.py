"""
Network-aware layers for multi-hop effects and intervention cascade prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.utils import softmax, add_self_loops
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class MultiHopAggregator(nn.Module):
    """
    Explicitly tracks multi-hop information flow through the network
    Essential for proving GNN value beyond simple correlation
    """
    
    def __init__(self, hidden_dim: int = 128, max_hops: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_hops = max_hops
        
        # Separate GNN layers for each hop distance
        self.hop_layers = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim, add_self_loops=False)
            for _ in range(max_hops)
        ])
        
        # Attention mechanism to weight hop importance
        self.hop_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Gate to control information flow from different hops
        self.hop_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            ) for _ in range(max_hops)
        ])
        
        # Final aggregation
        self.aggregator = nn.Sequential(
            nn.Linear(hidden_dim * max_hops, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass tracking multi-hop effects
        
        Returns:
            Aggregated features and hop-wise features for analysis
        """
        hop_features = []
        hop_importance = {}
        
        # Track information at each hop distance
        h_current = x
        for hop in range(self.max_hops):
            # Apply hop-specific transformation
            h_hop = self.hop_layers[hop](h_current, edge_index)
            
            # Gate based on original features (selective information flow)
            gate = self.hop_gates[hop](torch.cat([x, h_hop], dim=-1))
            h_gated = gate * h_hop + (1 - gate) * h_current
            
            hop_features.append(h_gated)
            hop_importance[f'hop_{hop+1}'] = gate.mean().item()
            
            # Prepare for next hop
            h_current = h_gated
        
        # Stack hop features [N, max_hops, D]
        hop_stack = torch.stack(hop_features, dim=1)
        
        # Apply attention across hops
        attended, attention_weights = self.hop_attention(
            hop_stack, hop_stack, hop_stack
        )
        
        # Concatenate all hop information
        hop_concat = torch.cat(hop_features, dim=-1)
        
        # Final aggregation
        output = self.aggregator(hop_concat)
        
        # Store hop-wise features for cascade tracking
        hop_dict = {
            f'hop_{i+1}_features': hop_features[i] 
            for i in range(self.max_hops)
        }
        hop_dict['hop_importance'] = hop_importance
        hop_dict['attention_weights'] = attention_weights
        
        return output, hop_dict


class InterventionImpactLayer(nn.Module):
    """
    Predicts how interventions cascade through the network
    Critical for proving multi-hop value
    """
    
    def __init__(self, hidden_dim: int = 128, max_cascade_hops: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_cascade_hops = max_cascade_hops
        
        # Intervention encoder
        self.intervention_encoder = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),  # +3 for intervention type
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Cascade propagation using GATConv for attention-based spread
        self.cascade_layers = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim // 4, heads=4, concat=True, dropout=0.1)
            for _ in range(max_cascade_hops)
        ])
        
        # LSTM for temporal cascade dynamics
        self.cascade_lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim, 
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Impact prediction heads for each hop
        self.impact_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim // 2, 3)  # 3 impact types: energy, cost, emissions
            ) for _ in range(max_cascade_hops)
        ])
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        intervention_mask: torch.Tensor,
        intervention_type: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Predict cascade effects of interventions
        
        Args:
            x: Node features [N, D]
            edge_index: Graph connectivity
            intervention_mask: Binary mask of intervened nodes [N]
            intervention_type: Type of intervention (solar, battery, etc.) [N, 3]
            
        Returns:
            Dictionary with cascade predictions at each hop
        """
        # Encode intervention
        if intervention_type is None:
            # Default: solar intervention
            intervention_type = torch.zeros(x.size(0), 3, device=x.device)
            intervention_type[:, 0] = 1.0  # Solar flag
            
        # Combine features with intervention info
        x_intervened = self.intervention_encoder(
            torch.cat([x, intervention_type], dim=-1)
        )
        
        # Apply intervention mask
        x_cascade = x_intervened * intervention_mask.unsqueeze(-1)
        
        cascade_effects = {}
        hop_impacts = []
        
        # Propagate cascade through network
        h_current = x_cascade
        for hop in range(self.max_cascade_hops):
            # Spread impact through graph
            h_cascade = self.cascade_layers[hop](h_current, edge_index)
            
            # Predict impact at this hop distance
            impact = self.impact_heads[hop](h_cascade)
            
            cascade_effects[f'hop_{hop+1}_impact'] = impact
            hop_impacts.append(h_cascade.unsqueeze(1))  # Add time dimension
            
            # Update for next hop
            h_current = h_cascade
        
        # Temporal dynamics of cascade
        hop_sequence = torch.cat(hop_impacts, dim=1)  # [N, hops, D]
        temporal_cascade, (h_n, c_n) = self.cascade_lstm(hop_sequence)
        
        # Final cascade prediction
        cascade_effects['temporal_evolution'] = temporal_cascade
        cascade_effects['final_state'] = h_n[-1]  # Last layer hidden state
        
        # Calculate total network impact
        total_impact = sum([
            cascade_effects[f'hop_{i+1}_impact'].abs().mean() * (0.5 ** i)
            for i in range(self.max_cascade_hops)
        ])
        cascade_effects['total_network_impact'] = total_impact
        
        return cascade_effects


class NetworkPositionEncoder(nn.Module):
    """
    Encodes network position beyond just features
    Captures strategic value of nodes in the network
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Positional encoding based on network topology
        self.centrality_encoder = nn.Sequential(
            nn.Linear(5, hidden_dim // 4),  # 5 centrality measures
            nn.ReLU()
        )
        
        # Transformer boundary encoding
        self.boundary_encoder = nn.Embedding(2, hidden_dim // 4)  # Binary: boundary or not
        
        # Grid level encoding
        self.level_encoder = nn.Embedding(3, hidden_dim // 4)  # Building, LV, MV level
        
        # Combine encodings
        self.position_mixer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        centrality_features: torch.Tensor,
        boundary_mask: torch.Tensor,
        grid_level: torch.Tensor
    ) -> torch.Tensor:
        """
        Add network position information to node features
        """
        # Encode different aspects of network position
        centrality_enc = self.centrality_encoder(centrality_features)
        boundary_enc = self.boundary_encoder(boundary_mask.long())
        level_enc = self.level_encoder(grid_level)
        
        # Concatenate position encodings
        position_features = torch.cat([
            centrality_enc,
            boundary_enc,
            level_enc,
            torch.zeros(x.size(0), self.hidden_dim // 4, device=x.device)  # Padding
        ], dim=-1)
        
        # Mix with original features
        x_positioned = x + self.position_mixer(position_features)
        
        return x_positioned


class CrossLVBoundaryAttention(MessagePassing):
    """
    Special attention layer for cross-LV connections
    Learns which connections are impossible (transformer boundaries)
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__(aggr='add')
        self.hidden_dim = hidden_dim
        
        # Attention for valid connections
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.LeakyReLU(0.2)
        )
        
        # Boundary violation detector
        self.boundary_detector = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor,
        transformer_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with boundary learning
        
        Returns:
            Updated features and boundary violation scores
        """
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Start propagating messages
        out = self.propagate(edge_index, x=x, transformer_mask=transformer_mask)
        
        # Detect boundary violations
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col]], dim=-1)
        boundary_scores = self.boundary_detector(edge_features)
        
        return out, boundary_scores
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        """Construct messages"""
        # Compute attention scores
        edge_features = torch.cat([x_i, x_j], dim=-1)
        alpha = self.attention(edge_features)
        alpha = softmax(alpha, index=index, dim=0)
        
        # Weight messages by attention
        return alpha * x_j
    
    def update(self, aggr_out: torch.Tensor) -> torch.Tensor:
        """Update node features"""
        return aggr_out


class NetworkAwareGNN(nn.Module):
    """
    Enhanced GNN with explicit network awareness and cascade tracking
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 4)  # Increased to 4 for better multi-hop
        
        # Import base model components
        from models.base_gnn import BuildingEncoder, TaskHeads
        
        # Encoders
        self.building_encoder = BuildingEncoder(
            config.get('building_features', 17),
            self.hidden_dim
        )
        
        # Network position encoding
        self.position_encoder = NetworkPositionEncoder(self.hidden_dim)
        
        # Multi-hop aggregator
        self.multi_hop_agg = MultiHopAggregator(
            self.hidden_dim,
            max_hops=config.get('max_cascade_hops', 3)
        )
        
        # Cross-LV boundary attention
        self.boundary_attention = CrossLVBoundaryAttention(self.hidden_dim)
        
        # Intervention impact predictor
        self.intervention_impact = InterventionImpactLayer(
            self.hidden_dim,
            max_cascade_hops=config.get('max_cascade_hops', 3)
        )
        
        # Standard GNN layers
        self.gnn_layers = nn.ModuleList([
            GATConv(self.hidden_dim, self.hidden_dim // 4, heads=4, concat=True)
            for _ in range(self.num_layers)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers)
        ])
        
        # Modified task heads with network impact prediction
        self.task_heads = TaskHeads(self.hidden_dim, config)
        
        # NEW: Network impact prediction head
        self.network_impact_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),  # Predict 1-hop, 2-hop, 3-hop impacts
            nn.ReLU()  # Ensure non-negative impacts
        )
        
        # NEW: Intervention value head
        self.intervention_value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)  # Network-wide value score
        )
        
        # Cluster head for soft assignments
        self.cluster_head = nn.Linear(self.hidden_dim, config.get('num_clusters', 10))
        
        logger.info(f"Initialized NetworkAwareGNN with {self.num_layers} layers")
        
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        intervention_mask: Optional[torch.Tensor] = None,
        centrality_features: Optional[torch.Tensor] = None,
        boundary_mask: Optional[torch.Tensor] = None,
        grid_level: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with network-aware processing
        """
        # Encode building features
        h = self.building_encoder(x)
        
        # Add network position encoding if available
        if centrality_features is not None:
            h = self.position_encoder(
                h, 
                centrality_features,
                boundary_mask if boundary_mask is not None else torch.zeros(x.size(0)),
                grid_level if grid_level is not None else torch.zeros(x.size(0))
            )
        
        # Standard GNN message passing
        for i, (conv, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            h_new = conv(h, edge_index)
            h = h + h_new  # Residual
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=0.1, training=self.training)
        
        # Multi-hop aggregation
        h_multihop, hop_features = self.multi_hop_agg(h, edge_index)
        
        # Cross-LV boundary learning
        h_boundary, boundary_violations = self.boundary_attention(h_multihop, edge_index)
        
        # Combine features
        h_final = h_multihop + 0.5 * h_boundary
        
        # Predict intervention impacts if mask provided
        cascade_effects = None
        if intervention_mask is not None:
            cascade_effects = self.intervention_impact(
                h_final, edge_index, intervention_mask
            )
        
        # Task-specific outputs
        outputs = {
            'embeddings': h_final,
            'hop_features': hop_features,
            'boundary_violations': boundary_violations
        }
        
        # Add clustering output (soft assignments)
        outputs['clusters'] = F.softmax(self.cluster_head(h_final), dim=-1)
        
        # Network impact predictions
        network_impacts = self.network_impact_head(h_final)
        outputs['network_impacts'] = network_impacts
        
        # Intervention value predictions
        intervention_values = self.intervention_value_head(h_final)
        outputs['intervention_values'] = intervention_values
        
        if cascade_effects is not None:
            outputs['cascade_effects'] = cascade_effects
        
        return outputs