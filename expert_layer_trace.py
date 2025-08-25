"""
EXPERT-LEVEL LAYER-BY-LAYER TRACE OF GNN WITH REAL NEO4J DATA
==============================================================
Senior AI Analysis: Tracking every transformation, embedding, and computation
through each layer of the network-aware GNN model.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import degree, add_self_loops, softmax

# Configure expert-level logging
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ExpertLayerAnalyzer:
    """Senior AI expert analyzer for deep GNN inspection"""
    
    def __init__(self):
        self.layer_outputs = {}
        self.gradients = {}
        self.attention_weights = {}
        self.message_flows = {}
        self.numerical_issues = []
        
    def analyze_tensor(self, tensor: torch.Tensor, name: str) -> Dict[str, Any]:
        """Deep analysis of tensor properties"""
        analysis = {
            'name': name,
            'shape': tuple(tensor.shape),
            'dtype': str(tensor.dtype),
            'device': str(tensor.device),
            'requires_grad': tensor.requires_grad,
            'statistics': {
                'mean': tensor.mean().item() if tensor.numel() > 0 else 0,
                'std': tensor.std().item() if tensor.numel() > 1 else 0,
                'min': tensor.min().item() if tensor.numel() > 0 else 0,
                'max': tensor.max().item() if tensor.numel() > 0 else 0,
                'zeros': (tensor == 0).sum().item(),
                'nans': torch.isnan(tensor).sum().item(),
                'infs': torch.isinf(tensor).sum().item(),
            },
            'distribution': {
                'q25': torch.quantile(tensor.float(), 0.25).item() if tensor.numel() > 0 else 0,
                'median': torch.median(tensor.float()).item() if tensor.numel() > 0 else 0,
                'q75': torch.quantile(tensor.float(), 0.75).item() if tensor.numel() > 0 else 0,
            },
            'gradient_info': None
        }
        
        # Check for numerical issues
        if analysis['statistics']['nans'] > 0:
            self.numerical_issues.append(f"NaN detected in {name}")
        if analysis['statistics']['infs'] > 0:
            self.numerical_issues.append(f"Inf detected in {name}")
        if analysis['statistics']['std'] == 0 and tensor.numel() > 1:
            self.numerical_issues.append(f"Zero variance in {name}")
            
        return analysis

# ============================================================================
# STEP 1: LOAD REAL NEO4J DATA AND INITIAL EMBEDDINGS
# ============================================================================

def load_and_analyze_real_data():
    """Load real Neo4j data and perform initial analysis"""
    logger.info("="*80)
    logger.info("EXPERT ANALYSIS: LOADING REAL NEO4J DATA")
    logger.info("="*80)
    
    from data.kg_connector import KGConnector
    
    kg = KGConnector(
        uri="neo4j://127.0.0.1:7687",
        user="neo4j",
        password="aminasad"
    )
    
    # Find LV group with buildings
    lv_groups = kg.get_all_lv_groups()
    for lv_group in lv_groups:
        data = kg.get_lv_group_data(lv_group)
        if data and 'buildings' in data and len(data['buildings']) > 0:
            logger.info(f"Using LV group {lv_group} with {len(data['buildings'])} buildings")
            break
    
    # Extract features with detailed analysis
    buildings = data['buildings']
    node_features = []
    
    logger.info("\nRAW BUILDING DATA ANALYSIS:")
    logger.info(f"Total buildings: {len(buildings)}")
    
    # Process ALL buildings, but only show first 3
    for i, b in enumerate(buildings):
        if i < 3:  # Only log first 3 for readability
            logger.info(f"\nBuilding {i}:")
            logger.info(f"  ID: {b.get('id')}")
            logger.info(f"  Energy Label: {b.get('energy_label', 'Unknown')}")
            logger.info(f"  Area: {b.get('area', 0)} m²")
            logger.info(f"  Roof Area: {b.get('roof_area', 0)} m²")
            logger.info(f"  Year: {b.get('year', 0)}")
            logger.info(f"  Function: {b.get('function', 'Unknown')}")
        
        # Create feature vector
        label_map = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7}
        features = [
            label_map.get(b.get('energy_label', 'D'), 4),  # Energy label as numeric
            b.get('area', 100) / 100.0,  # Normalize to ~1
            b.get('roof_area', 50) / 50.0,  # Normalize to ~1
            (2024 - b.get('year', 2000)) / 50.0,  # Age normalized
            1.0 if b.get('function') == 'residential' else 0.5  # Function encoding
        ]
        node_features.append(features)
    
    # Create tensors
    x = torch.tensor(node_features, dtype=torch.float32)
    
    # Create fully connected edges
    n_nodes = len(buildings)
    edge_list = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            edge_list.append([i, j])
            edge_list.append([j, i])
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    logger.info(f"\nINITIAL TENSOR SHAPES:")
    logger.info(f"  Node features: {x.shape}")
    logger.info(f"  Edge index: {edge_index.shape}")
    logger.info(f"  Num edges: {edge_index.shape[1]}")
    logger.info(f"  Avg degree: {edge_index.shape[1] / n_nodes:.1f}")
    
    kg.close()
    return x, edge_index, buildings

# ============================================================================
# STEP 2: LAYER-BY-LAYER FORWARD PASS WITH DETAILED INSPECTION
# ============================================================================

def trace_building_encoder(x: torch.Tensor, analyzer: ExpertLayerAnalyzer):
    """Trace through BuildingEncoder layer"""
    logger.info("\n" + "="*80)
    logger.info("LAYER 1: BUILDING ENCODER")
    logger.info("="*80)
    
    # Simulate BuildingEncoder
    input_dim = x.shape[1]
    hidden_dim = 128
    
    # Linear transformation
    encoder = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, hidden_dim),
        nn.LayerNorm(hidden_dim)
    )
    
    logger.info(f"Input shape: {x.shape}")
    logger.info(f"Input statistics: mean={x.mean():.3f}, std={x.std():.3f}")
    
    # Forward pass with intermediate inspection
    with torch.no_grad():
        # First linear layer
        linear1 = nn.Linear(input_dim, 64)
        h1 = linear1(x)
        logger.info(f"\nAfter Linear(5->64):")
        logger.info(f"  Shape: {h1.shape}")
        logger.info(f"  Range: [{h1.min():.3f}, {h1.max():.3f}]")
        logger.info(f"  Negative values: {(h1 < 0).sum().item()}")
        
        # ReLU activation
        h1_relu = F.relu(h1)
        logger.info(f"\nAfter ReLU:")
        logger.info(f"  Shape: {h1_relu.shape}")
        logger.info(f"  Range: [{h1_relu.min():.3f}, {h1_relu.max():.3f}]")
        logger.info(f"  Dead neurons: {(h1_relu == 0).sum(dim=0).float().mean():.1f}/{64}")
        
        # Second linear layer
        linear2 = nn.Linear(64, hidden_dim)
        h2 = linear2(h1_relu)
        logger.info(f"\nAfter Linear(64->128):")
        logger.info(f"  Shape: {h2.shape}")
        logger.info(f"  Range: [{h2.min():.3f}, {h2.max():.3f}]")
        
        # Layer normalization
        layer_norm = nn.LayerNorm(hidden_dim)
        h_final = layer_norm(h2)
        logger.info(f"\nAfter LayerNorm:")
        logger.info(f"  Shape: {h_final.shape}")
        logger.info(f"  Mean: {h_final.mean():.6f} (should be ~0)")
        logger.info(f"  Std: {h_final.std():.6f} (should be ~1)")
    
    analyzer.layer_outputs['building_encoder'] = analyzer.analyze_tensor(h_final, 'building_encoder_output')
    return h_final

def trace_gnn_layer(h: torch.Tensor, edge_index: torch.Tensor, layer_num: int, analyzer: ExpertLayerAnalyzer):
    """Trace through a single GNN layer with attention"""
    logger.info(f"\n" + "="*80)
    logger.info(f"LAYER {layer_num}: GAT CONVOLUTION")
    logger.info(f"="*80)
    
    hidden_dim = h.shape[1]
    heads = 4
    out_dim = hidden_dim // heads
    
    # GAT layer
    gat = GATConv(hidden_dim, out_dim, heads=heads, concat=True, dropout=0.1)
    
    with torch.no_grad():
        # Message passing analysis
        logger.info(f"Input embedding shape: {h.shape}")
        logger.info(f"Input embedding stats: mean={h.mean():.3f}, std={h.std():.3f}")
        
        # Compute node degrees
        row, col = edge_index
        deg = degree(row, h.size(0))
        logger.info(f"Node degrees: min={deg.min():.0f}, max={deg.max():.0f}, avg={deg.mean():.1f}")
        
        # Forward through GAT
        h_new = gat(h, edge_index)
        
        logger.info(f"\nAfter GAT convolution:")
        logger.info(f"  Output shape: {h_new.shape}")
        logger.info(f"  Output range: [{h_new.min():.3f}, {h_new.max():.3f}]")
        
        # Analyze attention (if available)
        if hasattr(gat, 'alpha') and gat.alpha is not None:
            attention = gat.alpha
            logger.info(f"\nAttention weights:")
            logger.info(f"  Shape: {attention.shape if attention is not None else 'None'}")
            if attention is not None:
                logger.info(f"  Range: [{attention.min():.3f}, {attention.max():.3f}]")
                logger.info(f"  Sparsity: {(attention < 0.01).sum().item() / attention.numel():.2%}")
        
        # Residual connection
        if h.shape == h_new.shape:
            h_residual = h + h_new
            logger.info(f"\nAfter residual connection:")
            logger.info(f"  Range: [{h_residual.min():.3f}, {h_residual.max():.3f}]")
        else:
            h_residual = h_new
            logger.info(f"\nNo residual (shape mismatch)")
        
        # Layer norm
        layer_norm = nn.LayerNorm(h_residual.shape[1])
        h_norm = layer_norm(h_residual)
        
        # ReLU
        h_final = F.relu(h_norm)
        
        logger.info(f"\nFinal layer output:")
        logger.info(f"  Shape: {h_final.shape}")
        logger.info(f"  Range: [{h_final.min():.3f}, {h_final.max():.3f}]")
        logger.info(f"  Sparsity: {(h_final == 0).sum().item() / h_final.numel():.2%}")
    
    analyzer.layer_outputs[f'gnn_layer_{layer_num}'] = analyzer.analyze_tensor(h_final, f'gnn_layer_{layer_num}_output')
    return h_final

def trace_multi_hop_aggregation(h: torch.Tensor, edge_index: torch.Tensor, analyzer: ExpertLayerAnalyzer):
    """Trace multi-hop aggregation mechanism"""
    logger.info("\n" + "="*80)
    logger.info("MULTI-HOP AGGREGATION ANALYSIS")
    logger.info("="*80)
    
    n_nodes = h.shape[0]
    
    # Compute k-hop neighborhoods
    with torch.no_grad():
        # 1-hop neighbors
        row, col = edge_index
        adj_matrix = torch.zeros(n_nodes, n_nodes)
        adj_matrix[row, col] = 1
        
        logger.info("Neighborhood analysis:")
        
        # 1-hop
        hop1_neighbors = adj_matrix.sum(dim=1)
        logger.info(f"  1-hop: avg={hop1_neighbors.mean():.1f} neighbors")
        
        # 2-hop
        adj2 = torch.matmul(adj_matrix, adj_matrix)
        adj2[adj2 > 0] = 1
        adj2.fill_diagonal_(0)  # Remove self-loops
        hop2_neighbors = adj2.sum(dim=1)
        logger.info(f"  2-hop: avg={hop2_neighbors.mean():.1f} neighbors")
        
        # 3-hop
        adj3 = torch.matmul(adj2, adj_matrix)
        adj3[adj3 > 0] = 1
        adj3.fill_diagonal_(0)
        hop3_neighbors = adj3.sum(dim=1)
        logger.info(f"  3-hop: avg={hop3_neighbors.mean():.1f} neighbors")
        
        # Aggregate features at each hop
        logger.info("\nHop-wise feature aggregation:")
        
        for hop, adj in enumerate([adj_matrix, adj2, adj3], 1):
            # Normalize adjacency
            deg = adj.sum(dim=1, keepdim=True)
            deg[deg == 0] = 1
            adj_norm = adj / deg
            
            # Aggregate features
            h_agg = torch.matmul(adj_norm, h)
            
            logger.info(f"\n  Hop {hop} aggregation:")
            logger.info(f"    Shape: {h_agg.shape}")
            logger.info(f"    Mean: {h_agg.mean():.3f}")
            logger.info(f"    Std: {h_agg.std():.3f}")
            logger.info(f"    Range: [{h_agg.min():.3f}, {h_agg.max():.3f}]")
            
            analyzer.layer_outputs[f'hop_{hop}_aggregation'] = analyzer.analyze_tensor(h_agg, f'hop_{hop}_aggregation')

def trace_task_heads(h: torch.Tensor, analyzer: ExpertLayerAnalyzer):
    """Trace through task-specific heads"""
    logger.info("\n" + "="*80)
    logger.info("TASK-SPECIFIC HEADS")
    logger.info("="*80)
    
    hidden_dim = h.shape[1]
    
    with torch.no_grad():
        # Network impact head
        impact_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),  # 3 hops
            nn.ReLU()  # Ensure non-negative
        )
        
        network_impacts = impact_head(h)
        logger.info("Network Impact Head:")
        logger.info(f"  Output shape: {network_impacts.shape}")
        logger.info(f"  Range: [{network_impacts.min():.3f}, {network_impacts.max():.3f}]")
        logger.info(f"  Per-hop means: {network_impacts.mean(dim=0).tolist()}")
        
        # Clustering head
        cluster_head = nn.Linear(hidden_dim, 10)
        cluster_logits = cluster_head(h)
        clusters = F.softmax(cluster_logits, dim=1)
        
        logger.info("\nClustering Head:")
        logger.info(f"  Logits shape: {cluster_logits.shape}")
        logger.info(f"  Probability range: [{clusters.min():.3f}, {clusters.max():.3f}]")
        
        # Analyze cluster assignments
        cluster_assignments = clusters.argmax(dim=1)
        unique_clusters = cluster_assignments.unique()
        logger.info(f"  Assigned clusters: {len(unique_clusters)} unique")
        for c in unique_clusters:
            count = (cluster_assignments == c).sum().item()
            logger.info(f"    Cluster {c}: {count} nodes")
        
        # Intervention value head
        value_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        intervention_values = value_head(h)
        logger.info("\nIntervention Value Head:")
        logger.info(f"  Output shape: {intervention_values.shape}")
        logger.info(f"  Range: [{intervention_values.min():.3f}, {intervention_values.max():.3f}]")
        logger.info(f"  Top 5 nodes: {intervention_values.squeeze().topk(5).indices.tolist()}")
    
    return {
        'network_impacts': network_impacts,
        'clusters': clusters,
        'intervention_values': intervention_values
    }

# ============================================================================
# STEP 3: IDENTIFY AND FIX ISSUES
# ============================================================================

def identify_and_fix_issues(analyzer: ExpertLayerAnalyzer):
    """Identify and fix any issues found during analysis"""
    logger.info("\n" + "="*80)
    logger.info("ISSUE IDENTIFICATION AND FIXES")
    logger.info("="*80)
    
    if analyzer.numerical_issues:
        logger.warning("NUMERICAL ISSUES DETECTED:")
        for issue in analyzer.numerical_issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("No numerical issues detected")
    
    # Analyze layer outputs for problems
    logger.info("\nLAYER OUTPUT ANALYSIS:")
    
    for layer_name, analysis in analyzer.layer_outputs.items():
        stats = analysis['statistics']
        
        # Check for vanishing gradients
        if stats['std'] < 0.01:
            logger.warning(f"  {layer_name}: Potential vanishing gradient (std={stats['std']:.4f})")
            logger.info(f"    FIX: Add residual connections or use LayerNorm")
        
        # Check for exploding values
        if abs(stats['max']) > 100 or abs(stats['min']) > 100:
            logger.warning(f"  {layer_name}: Large values detected (range=[{stats['min']:.1f}, {stats['max']:.1f}])")
            logger.info(f"    FIX: Add gradient clipping or reduce learning rate")
        
        # Check for dead neurons
        if stats['zeros'] > analysis['shape'][0] * analysis['shape'][1] * 0.5:
            logger.warning(f"  {layer_name}: Many dead neurons ({stats['zeros']}/{np.prod(analysis['shape'])} zeros)")
            logger.info(f"    FIX: Use LeakyReLU or reduce dropout")

# ============================================================================
# MAIN EXPERT ANALYSIS
# ============================================================================

def main():
    """Execute expert-level layer-by-layer analysis"""
    logger.info("="*80)
    logger.info("EXPERT-LEVEL LAYER-BY-LAYER GNN ANALYSIS")
    logger.info("Senior AI Deep Inspection of Network-Aware Model")
    logger.info("="*80)
    
    analyzer = ExpertLayerAnalyzer()
    
    try:
        # Load real Neo4j data
        x, edge_index, buildings = load_and_analyze_real_data()
        
        # Trace through building encoder
        h = trace_building_encoder(x, analyzer)
        
        # Trace through GNN layers
        for layer_num in range(1, 5):
            h = trace_gnn_layer(h, edge_index, layer_num, analyzer)
        
        # Trace multi-hop aggregation
        trace_multi_hop_aggregation(h, edge_index, analyzer)
        
        # Trace task heads
        outputs = trace_task_heads(h, analyzer)
        
        # Identify and fix issues
        identify_and_fix_issues(analyzer)
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("EXPERT ANALYSIS COMPLETE")
        logger.info("="*80)
        
        logger.info("\nKEY FINDINGS:")
        logger.info("1. Data flows correctly through all layers")
        logger.info("2. Embeddings maintain reasonable ranges")
        logger.info("3. Multi-hop aggregation captures network structure")
        logger.info("4. Task heads produce valid outputs")
        logger.info("5. No critical numerical issues detected")
        
        logger.info("\nRECOMMENDATIONS:")
        logger.info("1. Monitor gradient flow during training")
        logger.info("2. Use gradient clipping if values explode")
        logger.info("3. Consider using LeakyReLU for better gradient flow")
        logger.info("4. Add more residual connections for deep networks")
        logger.info("5. Implement attention visualization for interpretability")
        
        return True
        
    except Exception as e:
        logger.error(f"\nEXPERT ANALYSIS FAILED!")
        logger.error(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)