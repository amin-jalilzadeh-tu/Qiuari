how to formulate energy complementarity for your GNN clustering project:

## How Your Energy Types Are Complementary



### 2. **Key Complementarity Formulations for Your Project**

**Basic Complementarity Score** (from your document snippets 9, 63):
```python
# Pearson correlation approach (negative = complementary)
complementarity_score = 1 - correlation(building_i_profile, building_j_profile)

# Peak-to-Average Ratio (PAR) reduction
PAR_individual = max(profile) / mean(profile)
PAR_cluster = max(sum(profiles)) / mean(sum(profiles))
complementarity_benefit = 1 - (PAR_cluster / mean(PAR_individual))
```

**Multi-Energy Complementarity Index** (based on research findings):
```python
# Combine all three energy types
total_complementarity = w1 * heating_comp + w2 * cooling_comp + w3 * elec_comp

# Where each component uses:
heating_comp = -correlation(heat_i, heat_j) * diversity_factor(heat_i, heat_j)
cooling_comp = -correlation(cool_i, cool_j) * diversity_factor(cool_i, cool_j)
```

**Advanced Total Variation Metric** (from MDPI paper on complementarity):
```python
# Captures rate of change complementarity
TV_complementarity = 1 / (1 + abs(TV(profile_i) - TV(profile_j)))
# Where TV = sum of absolute differences between consecutive time steps
```

### 3. **Specific Articles You Should Read**

Based on your needs, prioritize these:

1. **"Total Variation-Based Metrics for Assessing Complementarity in Energy Resources Time Series"** (MDPI, 2022)
   - Provides mathematical formulations for complementarity beyond correlation
   - Includes Python code examples

2. **"A review on the complementarity of renewable energy sources"** (Solar Energy, 2020) 
   - Comprehensive metrics review
   - Section 3 specifically covers multi-energy complementarity

3. **"Multi-objective optimization of integrated energy system considering installation configuration"** (Energy, 2023)
   - Shows how to formulate multi-energy objectives
   - Includes heating/cooling/electricity integration

4. From your literature review document:
   - References #9, #39, #40: Cover complementarity definitions
   - References #63, #64: Discuss PAR and load factor optimization
   - Reference #76: Multi-objective optimization for complementarity

### 4. **Practical Implementation Steps**

**Step 1: Feature Engineering for Complementarity**
```python
# For each building pair, calculate:
features = {

```

**Step 2: GNN Loss Function Design**
From your method document, adapt the loss function to include all energy types:
```
```

**Step 3: Thermal-Electric Synergy Metrics**
Consider heat pump potential:
```python
# Buildings with excess cooling can provide heat via heat pumps
heat_pump_synergy = min(cooling_load_i, heating_load_j) * COP
```

