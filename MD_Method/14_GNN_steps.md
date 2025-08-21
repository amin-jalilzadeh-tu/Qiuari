## 🏘️ **DUMMY DATASET: 6 Buildings, 2 Transformers**

Let me create a minimal example and walk through EVERY step with exact inputs, formulas, and outputs.

---

## **📊 INITIAL DUMMY DATA**

### **Buildings (6 nodes):**
```
B1: Residential, 150m², Morning peak (7-9am)
B2: Residential, 120m², Morning peak (7-9am)  
B3: Office, 500m², Day peak (9am-5pm)
B4: Retail, 300m², Evening peak (5-8pm)
B5: Restaurant, 200m², Lunch+Dinner peaks (12-2pm, 6-9pm)
B6: Residential, 180m², Morning peak (7-9am)
```

### **Transformers (2 nodes):**
```
T1: 250 kVA capacity, serves B1, B2, B3
T2: 250 kVA capacity, serves B4, B5, B6
```

### **Time Series (24 hours, hourly):**
```python
# Simplified load profiles (kW)
L_B1 = [2,2,2,2,3,4,6,8,7,5,4,4,5,6,7,6,5,4,3,3,2,2,2,2]  # Residential
L_B2 = [2,2,2,2,3,4,5,7,6,4,3,3,4,5,6,5,4,3,3,2,2,2,2,2]  # Residential  
L_B3 = [1,1,1,1,2,3,5,8,15,18,20,20,18,18,20,18,15,10,5,3,2,1,1,1]  # Office
L_B4 = [2,2,2,2,3,4,5,6,8,10,12,14,15,16,18,20,22,20,18,15,10,8,5,3]  # Retail
L_B5 = [3,3,3,3,4,5,6,7,8,10,15,20,18,12,10,15,20,25,22,18,12,8,5,4]  # Restaurant
L_B6 = [2,2,2,2,3,5,7,9,8,6,5,5,6,7,8,7,6,5,4,3,3,2,2,2]  # Residential
```

---

## **🔄 STEP-BY-STEP PROCESS**

### **STEP 1: Multi-Source Data Integration**

**INPUT:**
- Building attributes (CSV)
- Grid topology (JSON)
- Time series (CSV)

**PROCESS:**
```python
# Data harmonization
buildings_df = {
    'id': ['B1','B2','B3','B4','B5','B6'],
    'type': ['res','res','office','retail','restaurant','res'],
    'area': [150,120,500,300,200,180],
    'x': [0,50,100,200,250,300],  # coordinates
    'y': [0,0,0,0,0,0],
    'transformer': ['T1','T1','T1','T2','T2','T2']
}

grid_topology = {
    'T1': {'capacity': 250, 'buildings': ['B1','B2','B3']},
    'T2': {'capacity': 250, 'buildings': ['B4','B5','B6']}
}
```

**OUTPUT:** Structured data ready for KG

---

### **STEP 2: Knowledge Graph Construction**

**INPUT:** Harmonized data from Step 1

**PROCESS:**
```cypher
// Create nodes
CREATE (b1:Building {id:'B1', type:'res', area:150, x:0, y:0})
CREATE (b2:Building {id:'B2', type:'res', area:120, x:50, y:0})
...
CREATE (t1:Transformer {id:'T1', capacity:250})
CREATE (t2:Transformer {id:'T2', capacity:250})

// Create relationships
CREATE (b1)-[:CONNECTED_TO]->(t1)
CREATE (b2)-[:CONNECTED_TO]->(t1)
...
```

**FORMULA:** Distance-based spatial relationships
$$d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

For B1-B2: $d_{12} = \sqrt{(0-50)^2 + (0-0)^2} = 50m$

**OUTPUT:** Neo4j graph with 8 nodes, 12+ edges

---

### **STEP 3: Temporal Feature Extraction**

**INPUT:** Time series data L_B1 to L_B6

**PROCESS & FORMULAS:**

**Mean Load:**
$$\mu_{B1} = \frac{1}{24}\sum_{t=1}^{24} L_{B1}(t) = \frac{106}{24} = 4.42 \text{ kW}$$

**Standard Deviation:**
$$\sigma_{B1} = \sqrt{\frac{1}{24}\sum_{t=1}^{24}(L_{B1}(t) - \mu_{B1})^2} = 2.15 \text{ kW}$$

**Peak Load:**
$$P_{B1} = \max(L_{B1}) = 8 \text{ kW}$$

**Load Factor:**
$$LF_{B1} = \frac{\mu_{B1}}{P_{B1}} = \frac{4.42}{8} = 0.55$$

**Ramp Rate:**
$$RR_{B1} = \max|L_{B1}(t) - L_{B1}(t-1)| = |8-6| = 2 \text{ kW/h}$$

**OUTPUT Matrix:**
```
Features = [
    [4.42, 2.15, 8, 0.55, 2],  # B1
    [3.96, 1.89, 7, 0.57, 2],  # B2
    [9.58, 7.23, 20, 0.48, 5], # B3
    [10.5, 6.12, 22, 0.48, 4], # B4
    [11.3, 6.89, 25, 0.45, 6], # B5
    [4.75, 2.31, 9, 0.53, 2]   # B6
]
```

---

### **STEP 4: Semantic Embedding Generation**

**INPUT:** KG structure from Step 2

**PROCESS:** TransE embedding algorithm

**FORMULA:** 
$$\mathcal{L} = \sum_{(h,r,t) \in S} \sum_{(h',r,t') \in S'} [\gamma + d(h+r,t) - d(h'+r,t')]_+$$

Where:
- $d(h+r,t) = ||h + r - t||_2$ (L2 distance)
- $\gamma$ = margin parameter (typically 1.0)

**Example Calculation:**
For triple (B1, CONNECTED_TO, T1):
```
h_B1 = [0.2, 0.5, 0.1]  # Random initialization
r_conn = [0.1, 0.0, 0.3]  # Relation embedding
t_T1 = [0.3, 0.5, 0.4]  # Target embedding

Loss = ||h_B1 + r_conn - t_T1||² = ||(0.3,0.5,0.4) - (0.3,0.5,0.4)||² = 0 (ideal)
```

**OUTPUT:** Embedding vectors (dim=32)
```
E_B1 = [0.21, 0.53, 0.14, ..., 0.32]  # 32-dim
E_B2 = [0.19, 0.48, 0.16, ..., 0.29]
...
```

---

### **STEP 5: KG to GNN Graph Construction**

**INPUT:** KG + Features + Embeddings

**PROCESS:** Build multi-layer adjacency matrices

**Electrical Adjacency:**
```
A_elec = [
  [0,0,0,0,0,0],  # B1
  [0,0,0,0,0,0],  # B2
  [0,0,0,0,0,0],  # B3
  [0,0,0,0,0,0],  # B4
  [0,0,0,0,0,0],  # B5
  [0,0,0,0,0,0]   # B6
]
# Buildings connected to same transformer
A_elec[0,1] = A_elec[1,0] = 1  # B1-B2 (same T1)
A_elec[0,2] = A_elec[2,0] = 1  # B1-B3 (same T1)
A_elec[1,2] = A_elec[2,1] = 1  # B2-B3 (same T1)
# Similar for T2 buildings
```

**Spatial Adjacency (distance-based):**
$$A_{spatial}[i,j] = \exp(-d_{ij}/\sigma)$$

For B1-B2: $A_{spatial}[0,1] = \exp(-50/100) = 0.606$

**Feature Matrix:**
$$X = [X_{static} || X_{temporal} || E_{KG}]$$

```
X_B1 = [150, 1, 0, 0, | 4.42, 2.15, 8, 0.55, 2 | 0.21, 0.53, ...]
       [area, res, off, ret | temporal features | KG embedding]
```

**OUTPUT:** PyTorch Geometric Data object

---

### **STEP 6: Complementarity Graph Mining**

**INPUT:** Time series from all buildings

**PROCESS:** Compute correlation matrix

**FORMULA:** Pearson correlation
$$\rho_{ij} = \frac{\sum_{t=1}^{24}(L_i(t)-\mu_i)(L_j(t)-\mu_j)}{\sqrt{\sum_{t=1}^{24}(L_i(t)-\mu_i)^2}\sqrt{\sum_{t=1}^{24}(L_j(t)-\mu_j)^2}}$$

**Example Calculation B1-B3:**
```python
# B1 (residential) vs B3 (office)
ρ_13 = correlation(L_B1, L_B3) = -0.72  # Negative = complementary!
```

**Full Correlation Matrix:**
```
ρ = [
  [ 1.00, 0.92,-0.72,-0.45,-0.38, 0.95],  # B1
  [ 0.92, 1.00,-0.68,-0.52,-0.41, 0.89],  # B2
  [-0.72,-0.68, 1.00, 0.65, 0.43,-0.70],  # B3
  [-0.45,-0.52, 0.65, 1.00, 0.71,-0.48],  # B4
  [-0.38,-0.41, 0.43, 0.71, 1.00,-0.40],  # B5
  [ 0.95, 0.89,-0.70,-0.48,-0.40, 1.00]   # B6
]
```

**Complementarity Score:**
$$C_{ij} = \begin{cases} 
\frac{1-\rho_{ij}}{2} & \text{if same transformer} \\
0 & \text{otherwise}
\end{cases}$$

For B1-B3: $C_{13} = \frac{1-(-0.72)}{2} = 0.86$ (High complementarity!)

**OUTPUT:** Complementarity adjacency matrix

---

### **STEP 7: GNN Model Training**

**INPUT:** Graph from Step 5, Labels from Step 6

**PROCESS:** Forward pass through GNN

**Layer 1 - Message Passing:**
$$h_i^{(1)} = \sigma\left(W_1 \cdot \text{AGG}\left(\{x_j : j \in \mathcal{N}(i)\}\right)\right)$$

For B1 with neighbors B2, B3:
```
h_B1^(1) = ReLU(W1 · mean([X_B1, X_B2, X_B3]))
        = ReLU(W1 · [avg_features])
        = [0.34, 0.67, ..., 0.23]  # 64-dim hidden
```

**Layer 2 - Attention:**
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

Where $e_{ij} = \text{LeakyReLU}(a^T[Wh_i || Wh_j])$

**Loss Function:**
$$\mathcal{L} = \lambda_1 \underbrace{\sum_{i,j} C_{ij} \cdot d(h_i, h_j)}_{\text{Complementarity}} + \lambda_2 \underbrace{\frac{\max(\sum L_c)}{\text{avg}(\sum L_c)}}_{\text{Peak-to-Average}}$$

**OUTPUT:** Trained model weights, embeddings

---

### **STEP 8: Differentiable Clustering**

**INPUT:** Node embeddings from Step 7

**PROCESS:** DMoN clustering

**Soft Assignment Matrix:**
$$S = \text{softmax}(MLP(H))$$

```
S = [
  [0.9, 0.1],  # B1 → Cluster 1 (90%)
  [0.85,0.15], # B2 → Cluster 1
  [0.8, 0.2],  # B3 → Cluster 1
  [0.1, 0.9],  # B4 → Cluster 2
  [0.15,0.85], # B5 → Cluster 2
  [0.92,0.08]  # B6 → Cluster 1
]
```

**Modularity Loss:**
$$\mathcal{L}_{mod} = -\frac{1}{2m} \text{Tr}(S^T B S)$$

Where $B_{ij} = A_{ij} - \frac{k_i k_j}{2m}$

**Hard Clustering:**
```
Cluster 1: {B1, B2, B3, B6} - Mixed residential + office
Cluster 2: {B4, B5} - Commercial (retail + restaurant)
```

**OUTPUT:** Cluster assignments

---

### **STEP 9: Intervention Planning**

**INPUT:** Clusters from Step 8

**PROCESS:** Evaluate intervention impacts

**Solar Potential Score:**
$$S_{solar} = \text{area} \times \text{irradiance} \times \eta \times (1 - \text{shading})$$

For B3 (office, 500m²):
$$S_{B3} = 500 \times 0.15 \times 5.2 \times 0.85 = 331.5 \text{ kWh/day}$$

**Battery Value Score:**
$$V_{battery} = \int_t \max(0, L(t) - \bar{L}) \times \text{price}(t) dt$$

**Ranking Algorithm:**
```
Interventions = [
  {B3: Solar 75kW, ROI: 6.2 years},  # Best for cluster 1
  {B4: Battery 50kWh, ROI: 8.1 years}, # Best for cluster 2
  {B1: HeatPump 5kW, Savings: 30%}
]
```

**OUTPUT:** Ranked intervention list

---

### **STEP 10: Performance Validation**

**INPUT:** Clusters + Interventions

**PROCESS:** Physics and economic validation

**Energy Balance Check:**
$$\sum_{i \in c} L_i(t) = \sum_{i \in c} G_i(t) + P_{grid}(t) \quad \forall t$$

**Peak Reduction:**
Original: $P_{cluster1} = 8+7+20+9 = 44$ kW
Optimized: $P_{cluster1}^* = \max_t(\sum L_i - \sum G_i) = 28$ kW
Reduction: $(44-28)/44 = 36\%$

**Self-Sufficiency:**
$$SSR = \frac{\int \min(G_{solar}, L_{cluster})dt}{\int L_{cluster}dt} = \frac{280}{520} = 54\%$$

**Grid Constraint Check:**
$$\max_t \sum_{i \in T1} L_i(t) = 35 \text{ kW} < 250 \text{ kVA} \checkmark$$

**OUTPUT:** Validation metrics

---

### **STEP 11: KG Enrichment with Results**

**INPUT:** GNN outputs

**PROCESS:** Update Neo4j

```cypher
// Add complementarity relationships
MATCH (b1:Building {id:'B1'}), (b3:Building {id:'B3'})
CREATE (b1)-[:COMPLEMENTS {score:0.86, correlation:-0.72}]->(b3)

// Create clusters
CREATE (c1:EnergyCluster {
  id: 'C1',
  buildings: ['B1','B2','B3','B6'],
  peak_reduction: 0.36,
  self_sufficiency: 0.54
})

// Link buildings to clusters
MATCH (b:Building), (c:EnergyCluster)
WHERE b.id IN c.buildings
CREATE (b)-[:BELONGS_TO]->(c)
```

**OUTPUT:** Enriched KG

---

### **STEP 12: Continuous Learning Loop**

**INPUT:** New daily data

**PROCESS:** Incremental update

**Online Learning Formula:**
$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}(x_{new}, \theta_t)$$

**Concept Drift Detection:**
$$D_{KL}(P_t || P_{t-1}) > \epsilon \implies \text{retrain}$$

**OUTPUT:** Updated model

---

### **STEP 13: Query Interface Development**

**INPUT:** User queries

**PROCESS:** Query translation

**Natural Language → Cypher:**
```
"Which buildings complement each other?"
↓
MATCH (b1:Building)-[r:COMPLEMENTS]->(b2:Building)
WHERE r.score > 0.7
RETURN b1.id, b2.id, r.score
```

**Result:**
```
B1 ← 0.86 → B3  (Residential ← → Office)
B2 ← 0.84 → B3  (Residential ← → Office)
```

**OUTPUT:** Query results

---

### **STEP 14: Decision Support System**

**INPUT:** All analyses

**PROCESS:** Generate recommendations

**Decision Matrix:**
```
Scenario Analysis:
1. Do Nothing: Cost=0, Savings=0, Carbon=100%
2. Solar on B3: Cost=75k€, Savings=12k€/yr, Carbon=75%
3. Full optimization: Cost=150k€, Savings=28k€/yr, Carbon=45%

Recommendation: Start with Scenario 2, expand to 3 in Phase 2
```

**OUTPUT:** Action plan

---

### **STEP 15: Real-World Implementation**

**INPUT:** Approved plan

**PROCESS:** Deployment steps

```
Timeline:
Month 1: Install 75kW solar on B3
Month 2: Configure P2P energy sharing
Month 3: Deploy monitoring system
Month 4-6: Measure and optimize
```

**OUTPUT:** Operational energy community

---

## **📊 COMPLETE DATA FLOW SUMMARY**

```
STEP    INPUT                    KEY FORMULA                OUTPUT
====    =====                    ===========                ======
1       Raw data                 Harmonization              Structured data
2       Structured data          Graph creation             Neo4j KG
3       Time series              μ, σ, peak, LF            Feature matrix
4       KG structure             TransE: h+r≈t              Embeddings
5       KG + Features            Multi-layer adj            GNN Graph
6       Time series              ρ = cov(X,Y)/σxσy         Complement matrix
7       Graph + Labels           h = σ(W·AGG(N))           Trained GNN
8       Embeddings               S = softmax(MLP(H))        Clusters
9       Clusters                 ROI = Savings/Cost         Interventions
10      Clusters + Plans         Energy balance             Validation
11      GNN outputs              Cypher updates             Enriched KG
12      New data                 θ_new = θ - α∇L           Updated model
13      User queries             NL → Cypher                Query results
14      All analyses             Decision matrix            Recommendations
15      Approved plan            Implementation             Real community
```

This complete walkthrough with 6 buildings shows EXACTLY what happens at each step, with real formulas and concrete outputs!