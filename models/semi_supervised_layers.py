"""
Semi-supervised learning components for the Energy GNN system.
Implements pseudo-label generation, label propagation, and confidence-based learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LabelPropagation, MessagePassing
from torch_geometric.utils import softmax, add_self_loops, degree
from typing import Dict, Tuple, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PseudoLabelGenerator(nn.Module):
    """
    Generates high-quality pseudo-labels for unlabeled data with confidence scoring.
    Uses temporal consistency and graph structure for validation.
    """
    
    def __init__(self, hidden_dim: int = 128, num_classes: int = 10, 
                 confidence_threshold: float = 0.85):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        
        # Confidence predictor network
        self.confidence_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Temporal consistency checker
        self.temporal_consistency = nn.LSTM(
            input_size=num_classes,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            batch_first=True,
            dropout=0.1
        )
        
        # Graph consistency validator using GAT
        self.graph_validator = GATConv(
            in_channels=num_classes,
            out_channels=num_classes,
            heads=4,
            concat=False,
            dropout=0.1
        )
        
    def forward(self, embeddings: torch.Tensor, predictions: torch.Tensor,
                edge_index: torch.Tensor, temporal_features: Optional[torch.Tensor] = None) -> Dict:
        """
        Generate pseudo-labels with confidence scores and validation.
        
        Args:
            embeddings: Node embeddings from GNN [N, hidden_dim]
            predictions: Raw predictions from model [N, num_classes]
            edge_index: Graph connectivity [2, E]
            temporal_features: Temporal data for consistency check [N, T, D]
            
        Returns:
            Dictionary containing pseudo-labels, confidence scores, and masks
        """
        batch_size = predictions.size(0)
        device = predictions.device
        
        # 1. Get initial pseudo-labels and probabilities
        probs = F.softmax(predictions, dim=-1)
        max_probs, pseudo_labels = torch.max(probs, dim=-1)
        
        # 2. Calculate confidence scores from embeddings
        embedding_confidence = self.confidence_net(embeddings).squeeze(-1)
        
        # 3. Check temporal consistency if temporal features available
        temporal_confidence = torch.ones(batch_size, device=device)
        if temporal_features is not None and temporal_features.size(1) > 1:
            # Reshape predictions for temporal analysis
            pred_sequence = predictions.unsqueeze(1).expand(-1, temporal_features.size(1), -1)
            lstm_out, _ = self.temporal_consistency(pred_sequence)
            
            # Calculate temporal stability (low variance = high confidence)
            temporal_variance = torch.var(lstm_out, dim=1).mean(dim=-1)
            temporal_confidence = 1.0 / (1.0 + temporal_variance)
        
        # 4. Validate using graph structure
        graph_validated_probs = self.graph_validator(probs, edge_index)
        graph_consistency = F.cosine_similarity(probs, graph_validated_probs, dim=-1)
        graph_confidence = (graph_consistency + 1.0) / 2.0  # Normalize to [0, 1]
        
        # 5. Combine all confidence scores
        combined_confidence = (
            0.4 * max_probs +  # Original prediction confidence
            0.2 * embedding_confidence +  # Embedding-based confidence
            0.2 * temporal_confidence +  # Temporal consistency
            0.2 * graph_confidence  # Graph structure consistency
        )
        
        # 6. Create confidence mask
        confident_mask = combined_confidence > self.confidence_threshold
        
        # 7. Additional validation: check neighborhood agreement
        neighborhood_agreement = self._check_neighborhood_agreement(
            pseudo_labels, edge_index, min_agreement=0.6
        )
        
        # Final mask combines confidence and neighborhood agreement
        final_mask = confident_mask & neighborhood_agreement
        
        # 8. Entropy-based uncertainty for analysis
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        uncertainty = entropy / np.log(self.num_classes)  # Normalize to [0, 1]
        
        return {
            'pseudo_labels': pseudo_labels,
            'confidence_scores': combined_confidence,
            'confident_mask': final_mask,
            'max_probs': max_probs,
            'embedding_confidence': embedding_confidence,
            'temporal_confidence': temporal_confidence,
            'graph_confidence': graph_confidence,
            'uncertainty': uncertainty,
            'num_confident': final_mask.sum().item(),
            'confidence_stats': {
                'mean': combined_confidence.mean().item(),
                'std': combined_confidence.std().item(),
                'min': combined_confidence.min().item(),
                'max': combined_confidence.max().item()
            }
        }
    
    def _check_neighborhood_agreement(self, labels: torch.Tensor, 
                                     edge_index: torch.Tensor,
                                     min_agreement: float = 0.6) -> torch.Tensor:
        """
        Check if predicted labels agree with neighborhood majority.
        
        Args:
            labels: Predicted labels [N]
            edge_index: Graph edges [2, E]
            min_agreement: Minimum agreement ratio required
            
        Returns:
            Boolean mask indicating nodes with sufficient neighborhood agreement
        """
        num_nodes = labels.size(0)
        device = labels.device
        
        # Add self-loops for stability
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        
        # Calculate degree for each node
        row, col = edge_index_with_loops
        deg = degree(row, num_nodes, dtype=torch.float)
        
        # Create one-hot encoding of labels
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Aggregate neighborhood labels
        neighborhood_sum = torch.zeros_like(labels_one_hot)
        for i in range(edge_index_with_loops.size(1)):
            src, dst = edge_index_with_loops[:, i]
            neighborhood_sum[dst] += labels_one_hot[src]
        
        # Calculate agreement ratio
        neighborhood_avg = neighborhood_sum / (deg.unsqueeze(-1) + 1e-8)
        own_label_agreement = neighborhood_avg[torch.arange(num_nodes), labels]
        
        return own_label_agreement >= min_agreement


class GraphLabelPropagation(MessagePassing):
    """
    Advanced label propagation that considers edge features and node confidence.
    Propagates labels from high-confidence to low-confidence nodes.
    """
    
    def __init__(self, num_iterations: int = 10, alpha: float = 0.9):
        super().__init__(aggr='mean')
        self.num_iterations = num_iterations
        self.alpha = alpha  # Weight for original labels vs propagated
        
        # Edge weight predictor
        self.edge_weight_net = nn.Sequential(
            nn.Linear(3, 16),  # Assuming 3 edge features
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
    def forward(self, labels: torch.Tensor, confidence: torch.Tensor,
                edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Propagate labels through the graph based on confidence.
        
        Args:
            labels: Current labels (can be pseudo-labels) [N, C] or [N]
            confidence: Confidence scores for each node [N]
            edge_index: Graph connectivity [2, E]
            edge_attr: Edge features [E, D]
            mask: Mask indicating which nodes have known labels [N]
            
        Returns:
            Refined labels after propagation
        """
        device = labels.device
        
        # Convert labels to one-hot if needed
        if labels.dim() == 1:
            num_classes = labels.max().item() + 1
            labels = F.one_hot(labels, num_classes=num_classes).float()
        
        # Calculate edge weights if edge features provided
        if edge_attr is not None:
            edge_weights = self.edge_weight_net(edge_attr).squeeze(-1)
        else:
            edge_weights = torch.ones(edge_index.size(1), device=device)
        
        # Store original labels
        original_labels = labels.clone()
        
        # Iterative propagation
        for iteration in range(self.num_iterations):
            # Weight labels by confidence
            weighted_labels = labels * confidence.unsqueeze(-1)
            
            # Propagate through graph
            out = self.propagate(edge_index, x=weighted_labels, 
                                edge_weight=edge_weights,
                                confidence=confidence)
            
            # Combine with original labels
            if mask is not None:
                # Keep known labels fixed
                labels = torch.where(
                    mask.unsqueeze(-1),
                    original_labels,
                    self.alpha * original_labels + (1 - self.alpha) * out
                )
            else:
                labels = self.alpha * original_labels + (1 - self.alpha) * out
            
            # Normalize to maintain probability distribution
            labels = F.softmax(labels, dim=-1)
        
        return labels
    
    def message(self, x_j: torch.Tensor, edge_weight: torch.Tensor,
                confidence_j: torch.Tensor) -> torch.Tensor:
        """
        Construct messages from source to target nodes.
        
        Args:
            x_j: Source node features (labels) [E, C]
            edge_weight: Weight for each edge [E]
            confidence_j: Confidence of source nodes [E]
            
        Returns:
            Weighted messages [E, C]
        """
        # Weight by both edge weight and source confidence
        weight = edge_weight * confidence_j
        return x_j * weight.unsqueeze(-1)


class SelfTrainingModule(nn.Module):
    """
    Implements self-training with iterative pseudo-label refinement.
    Includes curriculum learning and label quality assessment.
    """
    
    def __init__(self, base_model: nn.Module, num_classes: int = 10,
                 initial_threshold: float = 0.9, final_threshold: float = 0.7):
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.initial_threshold = initial_threshold
        self.final_threshold = final_threshold
        
        # Pseudo-label generator
        self.pseudo_generator = PseudoLabelGenerator(
            hidden_dim=base_model.config.get('hidden_dim', 128),
            num_classes=num_classes,
            confidence_threshold=initial_threshold
        )
        
        # Label propagation
        self.label_propagator = GraphLabelPropagation(
            num_iterations=10,
            alpha=0.85
        )
        
        # Label quality assessor
        self.quality_assessor = nn.Sequential(
            nn.Linear(num_classes * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Track training statistics
        self.training_stats = {
            'pseudo_label_accuracy': [],
            'confidence_evolution': [],
            'label_stability': []
        }
        
    def generate_and_refine_labels(self, data, labeled_mask: Optional[torch.Tensor] = None):
        """
        Generate pseudo-labels and refine them through propagation.
        
        Args:
            data: Input graph data
            labeled_mask: Boolean mask indicating labeled nodes
            
        Returns:
            Refined pseudo-labels and confidence scores
        """
        # Get base model predictions
        self.base_model.eval()
        with torch.no_grad():
            outputs = self.base_model(data)
            embeddings = outputs.get('embeddings', outputs.get('clustering_embeddings'))
            predictions = outputs.get('clustering_cluster_assignments', outputs.get('clusters'))
        
        # Generate initial pseudo-labels
        pseudo_results = self.pseudo_generator(
            embeddings=embeddings,
            predictions=predictions,
            edge_index=data.edge_index,
            temporal_features=data.x_temporal if hasattr(data, 'x_temporal') else None
        )
        
        # Refine through label propagation
        if data.edge_index.size(1) > 0:  # Only if edges exist
            refined_labels = self.label_propagator(
                labels=pseudo_results['pseudo_labels'],
                confidence=pseudo_results['confidence_scores'],
                edge_index=data.edge_index,
                edge_attr=data.edge_attr if hasattr(data, 'edge_attr') else None,
                mask=labeled_mask
            )
            
            # Convert back to class labels
            refined_labels = torch.argmax(refined_labels, dim=-1)
        else:
            refined_labels = pseudo_results['pseudo_labels']
        
        # Assess label quality
        quality_scores = self._assess_label_quality(
            refined_labels, 
            pseudo_results['confidence_scores'],
            data.edge_index
        )
        
        return {
            'labels': refined_labels,
            'confidence': pseudo_results['confidence_scores'],
            'quality': quality_scores,
            'mask': pseudo_results['confident_mask'],
            'stats': pseudo_results['confidence_stats']
        }
    
    def _assess_label_quality(self, labels: torch.Tensor, 
                             confidence: torch.Tensor,
                             edge_index: torch.Tensor) -> torch.Tensor:
        """
        Assess the quality of generated labels.
        
        Args:
            labels: Generated labels
            confidence: Confidence scores
            edge_index: Graph structure
            
        Returns:
            Quality scores for each label
        """
        num_nodes = labels.size(0)
        device = labels.device
        
        # One-hot encode labels
        labels_one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        
        # Calculate label distribution in neighborhood
        neighborhood_dist = torch.zeros_like(labels_one_hot)
        row, col = edge_index
        for i in range(edge_index.size(1)):
            neighborhood_dist[col[i]] += labels_one_hot[row[i]]
        
        # Normalize
        deg = degree(row, num_nodes, dtype=torch.float)
        neighborhood_dist = neighborhood_dist / (deg.unsqueeze(-1) + 1e-8)
        
        # Concatenate label and neighborhood distribution
        quality_input = torch.cat([labels_one_hot, neighborhood_dist], dim=-1)
        
        # Predict quality
        quality = self.quality_assessor(quality_input).squeeze(-1)
        
        # Combine with confidence
        final_quality = 0.6 * quality + 0.4 * confidence
        
        return final_quality
    
    def curriculum_threshold(self, epoch: int, max_epochs: int) -> float:
        """
        Implement curriculum learning by gradually decreasing confidence threshold.
        
        Args:
            epoch: Current epoch
            max_epochs: Total training epochs
            
        Returns:
            Current confidence threshold
        """
        progress = epoch / max_epochs
        threshold = self.initial_threshold - (self.initial_threshold - self.final_threshold) * progress
        return max(threshold, self.final_threshold)


class ConsistencyRegularization(nn.Module):
    """
    Implements consistency regularization for semi-supervised learning.
    Ensures predictions are consistent under different augmentations.
    """
    
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, pred_weak: torch.Tensor, pred_strong: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Calculate consistency loss between weak and strong augmentations.
        
        Args:
            pred_weak: Predictions from weakly augmented data
            pred_strong: Predictions from strongly augmented data
            mask: Optional mask for selective consistency
            
        Returns:
            Consistency loss
        """
        # Sharpen the weak predictions (teacher)
        prob_weak = F.softmax(pred_weak / self.temperature, dim=-1)
        prob_weak = prob_weak.detach()  # Stop gradient
        
        # Student predictions
        log_prob_strong = F.log_softmax(pred_strong, dim=-1)
        
        # KL divergence loss
        loss = F.kl_div(log_prob_strong, prob_weak, reduction='none').sum(dim=-1)
        
        if mask is not None:
            loss = loss * mask
            return loss.sum() / (mask.sum() + 1e-8)
        
        return loss.mean()