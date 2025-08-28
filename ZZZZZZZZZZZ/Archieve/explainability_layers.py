"""
Explainability layers for Energy GNN.
Implements GNNExplainer, attention visualization, and feature importance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
try:
    from torch_geometric.explain import GNNExplainer as BaseGNNExplainer
except ImportError:
    from torch_geometric.nn import GNNExplainer as BaseGNNExplainer
from torch_geometric.utils import softmax, add_self_loops
from torch_geometric.data import Data
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ExplainableGATConv(MessagePassing):
    """
    GAT layer that returns attention weights for explainability.
    Tracks and visualizes attention patterns.
    """
    
    def __init__(self, in_channels: int, out_channels: int, 
                 heads: int = 8, dropout: float = 0.1):
        super().__init__(node_dim=0, aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.dropout = dropout
        
        # Attention parameters
        self.W = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_channels))
        
        # Bias and dropout
        self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        self.dropout_layer = nn.Dropout(dropout)
        
        # Store attention weights for visualization
        self.attention_weights = None
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                return_attention: bool = True) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with attention weight tracking.
        
        Args:
            x: Node features [N, in_channels]
            edge_index: Graph connectivity [2, E]
            return_attention: Whether to return attention weights
            
        Returns:
            Updated node features and optionally attention weights
        """
        # Linear transformation
        x = self.W(x).view(-1, self.heads, self.out_channels)
        
        # Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Propagate with attention
        out = self.propagate(edge_index, x=x, size=None)
        
        # Apply bias and reshape
        out = out.view(-1, self.heads * self.out_channels)
        out = out + self.bias
        
        if return_attention and self.attention_weights is not None:
            return out, self.attention_weights
        
        return out, None
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor, 
                index: torch.Tensor, ptr: Optional[torch.Tensor],
                size_i: Optional[int]) -> torch.Tensor:
        """
        Compute messages with attention mechanism.
        
        Args:
            x_i: Target node features
            x_j: Source node features
            index: Target node indices
            ptr: Batch pointer
            size_i: Number of target nodes
            
        Returns:
            Weighted messages
        """
        # Compute attention scores
        x_cat = torch.cat([x_i, x_j], dim=-1)
        alpha = (x_cat * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, 0.2)
        alpha = softmax(alpha, index, ptr, size_i)
        
        # Store attention weights
        self.attention_weights = alpha
        
        # Apply dropout
        alpha = self.dropout_layer(alpha)
        
        # Weight messages
        return x_j * alpha.unsqueeze(-1)


class EnhancedGNNExplainer(nn.Module):
    """
    Enhanced GNNExplainer for Energy GNN with domain-specific constraints.
    Explains predictions by identifying important nodes and edges.
    """
    
    def __init__(self, model: nn.Module, num_hops: int = 3):
        super().__init__()
        self.model = model
        self.num_hops = num_hops
        
        # Edge and node mask parameters
        self.edge_mask = None
        self.node_feat_mask = None
        
        # Domain-specific importance weights
        self.feature_importance = nn.Parameter(torch.ones(17))  # For 17 building features
        
        # Explanation network
        self.explanation_net = nn.Sequential(
            nn.Linear(model.config.get('hidden_dim', 128), 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def explain_node(self, data: Data, node_idx: int, 
                    target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        Explain prediction for a specific node.
        
        Args:
            data: Input graph data
            node_idx: Index of node to explain
            target_class: Target class to explain (if None, use predicted)
            
        Returns:
            Dictionary with explanation components
        """
        self.model.eval()
        
        # Get subgraph around node
        subgraph_nodes, subgraph_edge_index, mapping = self._get_subgraph(
            data.edge_index, node_idx, self.num_hops
        )
        
        # Initialize masks
        num_edges = subgraph_edge_index.size(1)
        num_features = data.x.size(1)
        
        self.edge_mask = nn.Parameter(torch.ones(num_edges))
        self.node_feat_mask = nn.Parameter(torch.ones(len(subgraph_nodes), num_features))
        
        # Optimizer for masks
        optimizer = torch.optim.Adam([self.edge_mask, self.node_feat_mask], lr=0.01)
        
        # Get original prediction
        with torch.no_grad():
            original_output = self.model(data)
            if target_class is None:
                if 'clustering_cluster_assignments' in original_output:
                    pred = original_output['clustering_cluster_assignments'][node_idx]
                else:
                    pred = original_output['predictions'][node_idx]
                target_class = pred.argmax().item()
        
        # Optimization loop to find important edges and features
        for epoch in range(200):
            optimizer.zero_grad()
            
            # Apply masks
            masked_x = data.x[subgraph_nodes] * torch.sigmoid(self.node_feat_mask)
            edge_weight = torch.sigmoid(self.edge_mask)
            
            # Create masked subgraph data
            masked_data = Data(
                x=masked_x,
                edge_index=subgraph_edge_index,
                edge_attr=edge_weight
            )
            
            # Forward pass
            output = self.model(masked_data)
            
            # Get prediction for explained node
            node_output = output['predictions'][mapping] if 'predictions' in output else output[mapping]
            
            # Loss: maximize target class probability while minimizing mask size
            pred_loss = -F.log_softmax(node_output, dim=-1)[target_class]
            mask_loss = self.edge_mask.sigmoid().sum() * 0.01
            feat_loss = self.node_feat_mask.sigmoid().sum() * 0.001
            
            loss = pred_loss + mask_loss + feat_loss
            loss.backward()
            optimizer.step()
        
        # Get final masks
        edge_importance = self.edge_mask.sigmoid().detach()
        feature_importance = self.node_feat_mask.sigmoid().detach()
        
        # Identify most important features
        avg_feature_importance = feature_importance.mean(dim=0)
        top_features = torch.topk(avg_feature_importance, k=5)
        
        # Identify most important edges
        top_edges = torch.topk(edge_importance, k=min(10, num_edges))
        
        return {
            'node_idx': node_idx,
            'target_class': target_class,
            'subgraph_nodes': subgraph_nodes,
            'subgraph_edges': subgraph_edge_index,
            'edge_importance': edge_importance,
            'feature_importance': avg_feature_importance,
            'top_features': {
                'indices': top_features.indices.tolist(),
                'values': top_features.values.tolist(),
                'names': self._get_feature_names(top_features.indices)
            },
            'top_edges': {
                'indices': top_edges.indices.tolist(),
                'values': top_edges.values.tolist()
            }
        }
    
    def _get_subgraph(self, edge_index: torch.Tensor, node_idx: int, 
                     num_hops: int) -> Tuple[List[int], torch.Tensor, int]:
        """Extract k-hop subgraph around a node."""
        device = edge_index.device
        
        # Start with the target node
        subgraph_nodes = {node_idx}
        
        # Expand by k hops
        for _ in range(num_hops):
            new_nodes = set()
            for node in subgraph_nodes:
                # Find neighbors
                mask = (edge_index[0] == node) | (edge_index[1] == node)
                neighbors = edge_index[:, mask].unique().tolist()
                new_nodes.update(neighbors)
            subgraph_nodes.update(new_nodes)
        
        subgraph_nodes = list(subgraph_nodes)
        node_mapping = {node: i for i, node in enumerate(subgraph_nodes)}
        
        # Extract edges within subgraph
        subgraph_edges = []
        for i in range(edge_index.size(1)):
            src, dst = edge_index[0, i].item(), edge_index[1, i].item()
            if src in node_mapping and dst in node_mapping:
                subgraph_edges.append([node_mapping[src], node_mapping[dst]])
        
        subgraph_edge_index = torch.tensor(subgraph_edges, device=device).t()
        
        return subgraph_nodes, subgraph_edge_index, node_mapping[node_idx]
    
    def _get_feature_names(self, indices: torch.Tensor) -> List[str]:
        """Map feature indices to names."""
        feature_names = [
            'area', 'energy_score', 'solar_score', 'electrify_score', 'age',
            'roof_area', 'height', 'has_solar', 'has_battery', 'has_heat_pump',
            'shared_walls', 'x_coord', 'y_coord', 'avg_electricity_demand',
            'avg_heating_demand', 'peak_electricity_demand', 'energy_intensity'
        ]
        return [feature_names[i] for i in indices if i < len(feature_names)]


class AttentionVisualizer(nn.Module):
    """
    Visualizes attention weights and patterns in the GNN.
    Creates interpretable visualizations for analysis.
    """
    
    def __init__(self, save_dir: str = 'visualizations/attention'):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Store attention weights from different layers
        self.layer_attentions = {}
        
    def register_attention_hook(self, model: nn.Module):
        """Register hooks to capture attention weights."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) > 1:
                    # Assume second element is attention weights
                    if output[1] is not None:
                        self.layer_attentions[name] = output[1].detach().cpu()
            return hook
        
        # Register hooks for attention layers
        for name, module in model.named_modules():
            if 'attention' in name.lower() or isinstance(module, ExplainableGATConv):
                module.register_forward_hook(hook_fn(name))
    
    def visualize_node_attention(self, node_idx: int, layer_name: str = None):
        """
        Visualize attention weights for a specific node.
        
        Args:
            node_idx: Index of node to visualize
            layer_name: Specific layer to visualize (if None, visualize all)
        """
        if layer_name:
            layers_to_vis = {layer_name: self.layer_attentions[layer_name]}
        else:
            layers_to_vis = self.layer_attentions
        
        for name, attention in layers_to_vis.items():
            if attention is None:
                continue
                
            # Create visualization
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Attention distribution
            if attention.dim() == 2:
                node_attention = attention[node_idx] if node_idx < attention.size(0) else attention[0]
            else:
                node_attention = attention.mean(dim=0)[node_idx] if node_idx < attention.size(1) else attention.mean(dim=0)[0]
            
            axes[0].bar(range(len(node_attention)), node_attention.numpy())
            axes[0].set_title(f'Attention Weights for Node {node_idx}')
            axes[0].set_xlabel('Neighbor Index')
            axes[0].set_ylabel('Attention Weight')
            
            # Attention heatmap
            if attention.dim() >= 2:
                attention_matrix = attention.mean(dim=0) if attention.dim() > 2 else attention
                sns.heatmap(attention_matrix.numpy()[:20, :20], ax=axes[1], cmap='YlOrRd')
                axes[1].set_title(f'Attention Pattern - {name}')
            
            plt.tight_layout()
            plt.savefig(self.save_dir / f'attention_{name}_{node_idx}.png', dpi=100, bbox_inches='tight')
            plt.close()
    
    def visualize_layer_attention_stats(self):
        """Visualize statistics of attention across all layers."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (name, attention) in enumerate(self.layer_attentions.items()):
            if idx >= 4 or attention is None:
                break
            
            # Flatten attention weights
            attention_flat = attention.flatten().numpy()
            
            # Plot distribution
            axes[idx].hist(attention_flat, bins=50, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{name} - Attention Distribution')
            axes[idx].set_xlabel('Attention Weight')
            axes[idx].set_ylabel('Frequency')
            
            # Add statistics
            mean_att = attention_flat.mean()
            std_att = attention_flat.std()
            axes[idx].axvline(mean_att, color='red', linestyle='--', label=f'Mean: {mean_att:.3f}')
            axes[idx].axvline(mean_att + std_att, color='orange', linestyle='--', label=f'Std: {std_att:.3f}')
            axes[idx].axvline(mean_att - std_att, color='orange', linestyle='--')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'attention_statistics.png', dpi=100, bbox_inches='tight')
        plt.close()


class FeatureImportanceAnalyzer(nn.Module):
    """
    Analyzes and ranks feature importance for model predictions.
    Uses multiple methods including gradient-based and perturbation-based approaches.
    """
    
    def __init__(self, model: nn.Module, feature_names: Optional[List[str]] = None):
        super().__init__()
        self.model = model
        self.feature_names = feature_names or self._default_feature_names()
        
        # Feature importance aggregator
        self.importance_aggregator = nn.Sequential(
            nn.Linear(len(self.feature_names), 64),
            nn.ReLU(),
            nn.Linear(64, len(self.feature_names)),
            nn.Softmax(dim=-1)
        )
        
    def _default_feature_names(self) -> List[str]:
        """Default feature names for building data."""
        return [
            'area', 'energy_label', 'solar_potential', 'electrification',
            'age', 'roof_area', 'height', 'has_solar', 'has_battery',
            'has_heat_pump', 'shared_walls', 'x_coord', 'y_coord',
            'avg_electricity', 'avg_heating', 'peak_electricity', 'energy_intensity'
        ]
    
    def gradient_importance(self, data: Data, target_idx: int) -> torch.Tensor:
        """
        Calculate feature importance using gradients.
        
        Args:
            data: Input graph data
            target_idx: Index of target node
            
        Returns:
            Feature importance scores
        """
        self.model.eval()
        data.x.requires_grad = True
        
        # Forward pass
        output = self.model(data)
        
        # Get prediction for target node
        if 'predictions' in output:
            target_output = output['predictions'][target_idx]
        else:
            target_output = output[target_idx]
        
        # Get gradient with respect to input features
        target_class = target_output.argmax()
        target_output[target_class].backward()
        
        # Calculate importance as absolute gradient
        gradients = data.x.grad[target_idx].abs()
        
        # Normalize
        importance = gradients / (gradients.sum() + 1e-8)
        
        return importance.detach()
    
    def perturbation_importance(self, data: Data, target_idx: int,
                              n_samples: int = 100) -> torch.Tensor:
        """
        Calculate feature importance by perturbation analysis.
        
        Args:
            data: Input graph data
            target_idx: Target node index
            n_samples: Number of perturbation samples
            
        Returns:
            Feature importance scores
        """
        self.model.eval()
        
        with torch.no_grad():
            # Get original prediction
            original_output = self.model(data)
            if 'predictions' in original_output:
                original_pred = original_output['predictions'][target_idx]
            else:
                original_pred = original_output[target_idx]
            original_prob = F.softmax(original_pred, dim=-1).max()
        
        # Calculate importance for each feature
        num_features = data.x.size(1)
        importance = torch.zeros(num_features)
        
        for feat_idx in range(num_features):
            # Store original value
            original_value = data.x[target_idx, feat_idx].clone()
            
            # Perturbation samples
            changes = []
            for _ in range(n_samples):
                # Randomly perturb feature
                noise = torch.randn(1) * data.x[:, feat_idx].std()
                data.x[target_idx, feat_idx] = original_value + noise
                
                # Get new prediction
                with torch.no_grad():
                    perturbed_output = self.model(data)
                    if 'predictions' in perturbed_output:
                        perturbed_pred = perturbed_output['predictions'][target_idx]
                    else:
                        perturbed_pred = perturbed_output[target_idx]
                    perturbed_prob = F.softmax(perturbed_pred, dim=-1).max()
                
                # Calculate change
                changes.append(abs(original_prob - perturbed_prob).item())
            
            # Restore original value
            data.x[target_idx, feat_idx] = original_value
            
            # Average importance
            importance[feat_idx] = np.mean(changes)
        
        # Normalize
        importance = importance / (importance.sum() + 1e-8)
        
        return importance
    
    def integrated_gradients(self, data: Data, target_idx: int,
                            n_steps: int = 50) -> torch.Tensor:
        """
        Calculate feature importance using integrated gradients.
        
        Args:
            data: Input graph data
            target_idx: Target node index
            n_steps: Number of integration steps
            
        Returns:
            Feature importance scores
        """
        self.model.eval()
        
        # Create baseline (zeros)
        baseline = torch.zeros_like(data.x[target_idx])
        
        # Generate interpolated inputs
        alphas = torch.linspace(0, 1, n_steps)
        integrated_grads = torch.zeros_like(data.x[target_idx])
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (data.x[target_idx] - baseline)
            
            # Create modified data
            data_copy = data.clone()
            data_copy.x = data.x.clone()
            data_copy.x[target_idx] = interpolated
            data_copy.x.requires_grad = True
            
            # Forward pass
            output = self.model(data_copy)
            if 'predictions' in output:
                target_output = output['predictions'][target_idx]
            else:
                target_output = output[target_idx]
            
            # Backward pass
            target_class = target_output.argmax()
            self.model.zero_grad()
            target_output[target_class].backward()
            
            # Accumulate gradients
            integrated_grads += data_copy.x.grad[target_idx] / n_steps
        
        # Multiply by input difference
        importance = integrated_grads * (data.x[target_idx] - baseline)
        importance = importance.abs()
        
        # Normalize
        importance = importance / (importance.sum() + 1e-8)
        
        return importance.detach()
    
    def comprehensive_importance(self, data: Data, target_idx: int) -> Dict[str, Any]:
        """
        Calculate comprehensive feature importance using multiple methods.
        
        Args:
            data: Input graph data
            target_idx: Target node index
            
        Returns:
            Dictionary with importance scores from different methods
        """
        # Calculate importance using different methods
        grad_importance = self.gradient_importance(data, target_idx)
        perturb_importance = self.perturbation_importance(data, target_idx)
        ig_importance = self.integrated_gradients(data, target_idx)
        
        # Aggregate importances
        combined_importance = (grad_importance + perturb_importance + ig_importance) / 3
        
        # Learn aggregated importance
        learned_importance = self.importance_aggregator(combined_importance.unsqueeze(0)).squeeze(0)
        
        # Get top features
        top_k = 5
        top_features = torch.topk(learned_importance, k=top_k)
        
        return {
            'gradient': grad_importance,
            'perturbation': perturb_importance,
            'integrated_gradients': ig_importance,
            'combined': combined_importance,
            'learned': learned_importance,
            'top_features': {
                'indices': top_features.indices.tolist(),
                'names': [self.feature_names[i] for i in top_features.indices],
                'scores': top_features.values.tolist()
            },
            'feature_ranking': sorted(
                zip(self.feature_names, learned_importance.tolist()),
                key=lambda x: x[1],
                reverse=True
            )
        }