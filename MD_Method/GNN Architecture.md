# GNN Architecture

GNN Architecture


## 1. Input Layer

- **Node Features** (from KG):
- **Edge Features**:


## 2. -Aware Spatial Layer

## 3. Temporal Dynamics Layer

## 4. Multi-Granular Pooling Layer

## 5. Constraint-Aware Loss Function

## 6. Evaluation Metrics Beyond Standard

📊 Complementarity-Specific Metrics
1. Energy Synergy Score
synergy_score = (1 - PAR_cluster/PAR_individual) * 100

2. Temporal Stability
stability = 1 - (cluster_changes_per_hour / total_buildings)

3. Spatial Efficiency
spatial_eff = cluster_area / convex_hull_area

4. Constraint Satisfaction
constraint_score = buildings_in_same_LV / total_clustered

5. Diversity Index
diversity = shannon_entropy(building_types_in_cluster)


# **3. Key Differentiators from Existing Work**

| Aspect | Existing Papers | Your Innovation |
|--------|----------------|-----------------|
| **Objective** | Similarity clustering or load forecasting | Complementarity-based dynamic clustering |
| **Graph Type** | Static or simple temporal | Dynamic with hourly cluster evolution |
| **Node Relations** | Homophily (similar nodes connect) | Heterophily (opposite profiles connect) |
| **Constraints** | Often ignored or simplified | Strict LV/MV hierarchy enforcement |
| **KG Integration** | Separate from GNN | Unified KG features + GNN learning |
| **Evaluation** | Accuracy metrics | Energy synergy metrics (PAR, self-sufficiency) |

### **4. Novel Contributions to Highlight**

## 🌟 **Your Unique Contributions**

1. **Complementarity-First Design**
   - Unlike papers focusing on prediction accuracy, you optimize for energy synergy
   - Novel loss function combining negative correlation + peak reduction

2. **Dynamic Cluster Tracking**
   - Track how building clusters evolve throughout the day
   - Visualize "cluster jumping" patterns

3. **KG-GNN Deep Integration**
   - Not just using KG for features, but updating KG with GNN discoveries
   - Bidirectional learning loop

4. **Geomatics Perspective**
   - Spatial clustering quality metrics from geography
   - Convex hull efficiency for cluster compactness
   - Multi-scale spatial analysis (building → block → district)

5. **Real Grid Constraints**
   - PostgreSQL grid topology directly constrains GNN
   - Physical feasibility guaranteed by design

### **5. Implementation Roadmap**

## 📋 **Step-by-Step GNN Implementation**

```markdown
## Phase 1: Baseline Heterophily-Aware GNN
1. Implement GAT with negative attention for dissimilar nodes
2. Test on your mimic data (6 LV networks)
3. Baseline: correlation-based complementarity

## Phase 2: Add Temporal Dynamics
1. Integrate GRU/LSTM layers
2. 15-min resolution, 24h windows
3. Track cluster stability metrics

## Phase 3: Hierarchical Pooling
1. Implement DiffPool for soft cluster assignment
2. Add LV boundary constraints
3. Multi-level aggregation

## Phase 4: KG Integration
1. Pre-train KG embeddings (PyKEEN)
2. Concatenate with raw features
3. Update KG with discovered clusters

## Phase 5: Advanced Features
1. Solar dispatch co-optimization
2. EV charging pattern integration
3. Demand response signals