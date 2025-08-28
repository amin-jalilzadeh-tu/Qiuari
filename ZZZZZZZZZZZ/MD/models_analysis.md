# Models Analysis

## Core Models (Essential)
1. **base_gnn.py** - Main GNN architecture (HeteroEnergyGNN)
   - Building/LV/Transformer encoders
   - Message passing layers
   - Already includes ConstrainedDiffPool for clustering!

2. **task_heads.py** - Output heads
   - ClusteringHead (soft assignments)
   - EnergyPredictionHead
   - ComplementarityScoreHead

3. **solar_district_gnn.py** - Simplified version
   - Combines essential components
   - Focused on solar recommendations

## Feature Layers (Important)
4. **physics_layers.py** - Constraints
   - LVGroupBoundaryEnforcer (respects LV boundaries!)
   - EnergyBalanceChecker
   - DistanceBasedLossCalculator

5. **network_aware_layers.py** - Cascade effects
   - MultiHopAggregator (1-3 hop impacts)
   - InterventionImpactLayer
   - CrossLVBoundaryAttention

6. **pooling_layers.py** - Clustering
   - ConstrainedDiffPool (dynamic clustering within constraints!)
   - Respects min/max cluster sizes

## Enhancement Layers (Optional)
7. **attention_layers.py** / **attention_layers_simplified.py**
   - Complementarity attention
   - Temporal attention

8. **temporal_layers.py** 
   - Consumption pattern extraction
   - Seasonal adapters

9. **semi_supervised_layers.py**
   - PseudoLabelGenerator
   - SelfTrainingModule

10. **uncertainty_quantification.py**
    - MCDropout
    - Confidence calibration

11. **explainability_layers.py**
    - GNNExplainer
    - Attention visualizer

## Likely Unnecessary
- enhanced_temporal_layers.py (over-engineered)
- enhanced_uncertainty.py (too complex)
- gnn_optimizations.py (premature optimization)
- optimized_base_gnn.py (duplicate)
- sparse_utils.py (utility only)