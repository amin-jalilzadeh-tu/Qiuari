# Energy GNN System Refactoring Plan

## Table 1: Complete File Analysis and Actions

### Models Directory (models/)

| File | Current Purpose | Action | Reason | New Purpose |
|------|----------------|--------|---------|-------------|
| **base_gnn.py** | Core heterogeneous GNN architecture | **KEEP & SIMPLIFY** | Essential foundation | Simplified core GNN with building/LV/transformer encoders |
| **network_aware_layers.py** | Multi-hop aggregation, cascade tracking | **KEEP** | KEY DIFFERENTIATOR - proves GNN value | Multi-hop network effects for solar impact |
| **physics_layers.py** | LV boundaries, distance losses | **KEEP** | ESSENTIAL grid constraints | Enforce physical energy sharing rules |
| **temporal_layers.py** | Consumption pattern extraction, GRU/LSTM | **MERGE** | Duplicate with enhanced version | Merge into enhanced_temporal_layers.py |
| **enhanced_temporal_layers.py** | Advanced temporal (Fusion, Transformer) | **SIMPLIFY** | Over-complex, keep GRU only | Simple GRU-based temporal encoding |
| **attention_layers.py** | Multi-head attention mechanisms | **SIMPLIFY** | Keep basic attention only | Basic attention for importance weighting |
| **pooling_layers.py** | ConstrainedDiffPool for clustering | **KEEP** | Needed for hierarchical structure | Hierarchical clustering with constraints |
| **task_heads.py** | Multiple prediction heads | **SIMPLIFY** | Too many tasks | Keep only: ClusteringHead, SolarHead |
| **semi_supervised_layers.py** | Pseudo-labeling, label propagation | **KEEP & SIMPLIFY** | Needed for semi-supervised | Simple pseudo-label generator for unlabeled data |
| **uncertainty_quantification.py** | Bayesian layers, MC dropout | **MERGE** | Duplicate functionality | Merge best parts into enhanced_uncertainty.py |
| **enhanced_uncertainty.py** | Ensemble uncertainty, calibration | **SIMPLIFY** | Keep simple uncertainty only | Basic uncertainty estimation for confidence |
| **explainability_layers.py** | GNN explainer, attention viz | **SIMPLIFY** | Keep basic explainer only | Simple feature importance for solar decisions |
| **dynamic_graph_layers.py** | Edge features, dynamic construction | **REMOVE** | Over-complex for current needs | - |
| **sparse_utils.py** | Sparse matrix utilities | **KEEP** | Performance optimization | Efficient sparse operations |
| **gnn_optimizations.py** | Performance optimizations | **KEEP** | Critical for speed | Memory and compute optimizations |
| **optimized_base_gnn.py** | Optimized version of base | **MERGE** | Duplicate with base_gnn | Merge optimizations into base_gnn.py |

### Training Directory (training/)

| File | Current Purpose | Action | Reason | New Purpose |
|------|----------------|--------|---------|-------------|
| **unified_gnn_trainer.py** | Main trainer for clustering | **KEEP & ENHANCE** | Best starting point | Unified trainer for discovery + solar |
| **network_aware_trainer.py** | Multi-hop training with interventions | **MERGE** | Best features into unified | Merge network-aware logic into unified |
| **discovery_trainer.py** | Unsupervised pattern discovery | **MERGE** | Discovery phase of unified | Merge discovery logic into unified |
| **enhanced_trainer.py** | Kitchen sink - all features | **REMOVE** | Over-complex | - |
| **enhanced_trainer_minimal.py** | Slightly simpler enhanced | **REMOVE** | Still too complex | - |
| **loss_functions.py** | Core energy losses | **KEEP & ENHANCE** | Central loss repository | Add SolarROILoss, ClusterQualityLoss |
| **network_aware_loss.py** | Network impact losses | **MERGE** | Important but merge | Merge into loss_functions.py |
| **active_learning.py** | Query strategies | **REMOVE** | Not needed for current approach | - |
| **contrastive_learning.py** | Self-supervised contrastive | **REMOVE** | Over-complex for current needs | - |
| **evaluation_metrics.py** | Comprehensive metrics | **SIMPLIFY** | Too many metrics | Keep only essential energy metrics |
| **enhanced_hooks.py** | Training hooks/callbacks | **REMOVE** | Unnecessary complexity | - |

### Tasks Directory (tasks/)

| File | Current Purpose | Action | Reason | New Purpose |
|------|----------------|--------|---------|-------------|
| **solar_optimization.py** | Solar placement optimization | **KEEP & ENHANCE** | Core task | Add semi-supervised label support |
| **intervention_selection.py** | Network-aware intervention ranking | **KEEP** | Good network reasoning | Select solar based on multi-hop impact |
| **clustering.py** | Energy community clustering | **MODIFY** | Add label generation | Add cluster quality label generation |
| **retrofit_targeting.py** | Building retrofit recommendations | **REMOVE** | Not core to solar focus | - |
| **Additional Task Implementations.py** | Unclear/various tasks | **REMOVE** | Unclear purpose | - |

### NEW Files to Create

| File | Purpose | Why Needed |
|------|---------|------------|
| **models/solar_district_gnn.py** | Simplified main model | Single clear model entry point |
| **training/unified_solar_trainer.py** | Combined discovery + solar trainer | Unified training pipeline |
| **tasks/solar_labeling.py** | Generate labels from solar performance | Semi-supervised learning loop |
| **utils/label_generator.py** | Auto-generate cluster quality labels | Automated cluster evaluation |

## Table 2: Simplified Architecture Components

### Core Components to Keep

| Component | Purpose | Why Essential |
|-----------|---------|---------------|
| **Multi-hop GNN** | Track network effects 2-3 hops away | Proves GNN value beyond correlation |
| **Physics Constraints** | LV boundary enforcement | Can't share energy across boundaries |
| **Temporal GRU** | Process consumption patterns | Need temporal for complementarity |
| **Hierarchical Pooling** | Building → LV → District | Natural grid hierarchy |
| **Simple Uncertainty** | Confidence in predictions | Know when unsure about solar |
| **Basic Explainer** | Why recommend solar here? | Trustworthy decisions |
| **Semi-supervised** | Learn from deployed solar | Iterative improvement |

### Loss Functions Structure

| Loss Type | Purpose | When Used |
|-----------|---------|-----------|
| **ComplementarityLoss** | Find negative correlation patterns | Discovery phase |
| **NetworkImpactLoss** | Multi-hop grid effects | Both phases |
| **PhysicsConstraintLoss** | Respect LV boundaries | Always active |
| **SolarROILoss** (NEW) | Optimize solar returns | Solar phase |
| **ClusterQualityLoss** (NEW) | Semi-supervised cluster learning | When labels available |

### Training Pipeline Phases

| Phase | Input | Output | Labels Used |
|-------|-------|--------|-------------|
| **Discovery** | Temporal consumption data | Self-sufficient clusters | Cluster quality (auto) |
| **Solar Planning** | Clusters + building features | Solar recommendations | Solar success (real) |
| **Deployment** | Top N recommendations | Installed solar | - |
| **Measurement** | Actual performance data | Performance metrics | - |
| **Labeling** | Performance metrics | New training labels | Generate new labels |

## Table 3: Code Reduction Summary

| Category | Current Files | After Refactor | Lines Saved |
|----------|--------------|----------------|-------------|
| **Models** | 16 files | 10 files | ~3,000 lines |
| **Training** | 11 files | 5 files | ~2,500 lines |
| **Tasks** | 5 files | 3 files | ~1,500 lines |
| **Total** | 32 files | 18 files | ~7,000 lines (44% reduction) |

## Table 4: Semi-Supervised Learning Components

| Component | Function | Label Type | Source |
|-----------|----------|------------|--------|
| **Cluster Evaluator** | Rate cluster quality | Automatic | Calculated from metrics |
| **Solar Labeler** | Rate solar success | Manual | Real deployment data |
| **Pseudo-Label Generator** | Label unlabeled nodes | Automatic | High-confidence predictions |
| **Label Propagation** | Spread labels in graph | Automatic | Graph structure |

## Implementation Priority

### Phase 1: Core Simplification (Week 1)
1. Merge duplicate files (temporal, uncertainty, base_gnn)
2. Create unified_solar_trainer.py
3. Simplify loss functions

### Phase 2: Semi-Supervised Integration (Week 2)
1. Add solar_labeling.py
2. Add cluster quality labels
3. Implement label generation

### Phase 3: Testing & Validation (Week 3)
1. Test simplified pipeline
2. Validate multi-hop value
3. Measure performance improvement

## Key Benefits of This Refactoring

1. **Clear Purpose**: Each file has single responsibility
2. **No Duplication**: Merged all overlapping functionality
3. **Maintainable**: 44% less code to maintain
4. **Semi-Supervised Ready**: Clear label integration points
5. **Preserves Innovation**: Keeps multi-hop GNN value
6. **Production Ready**: Simplified for deployment

## What Makes This Architecture Special

1. **Multi-hop Network Reasoning**: Unlike simple clustering, considers grid-wide effects
2. **Physics-Aware**: Respects real grid constraints (LV boundaries)
3. **Iterative Learning**: Gets better with each deployment round
4. **Dual Labels**: Both automatic (clusters) and real (solar) feedback
5. **Practical Focus**: Optimizes actual ROI, not abstract metrics