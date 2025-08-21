## **`kg_builder.py` - Complete Design Specification** 🏗️

### **📥 INPUTS**

#### **1. Mimic Data Files**
```
mimic_data/
├── buildings.csv (150 buildings)
│   ├── Building attributes: ogc_fid, x, y, area, height
│   ├── Type info: building_function, residential_type, non_residential_type
│   ├── Energy features: roof_area, orientation, age_range
│   └── Grid assignment: lv_component_id, lv_network_id
│
├── lv_networks.csv (6 LV networks)
│   ├── Network info: component_id, x, y
│   ├── Hierarchy: mv_transformer_id
│   └── Capacity: capacity_kva
│
├── mv_transformers.csv (2 MV transformers)
│   ├── Transformer info: id, x, y
│   ├── Hierarchy: substation_id
│   └── Capacity: capacity_kva
│
└── energy_profiles.parquet (7 days × 96 intervals × 150 buildings)
    ├── Demand: electricity_demand_kw, heating_demand_kw, cooling_demand_kw
    ├── Generation: solar_generation_kw
    └── Storage: battery_soc_kwh, battery_charge_kw, battery_discharge_kw
```

#### **2. Configuration Parameters**
```yaml
config:
  neo4j_uri: "bolt://localhost:7687"
  neo4j_user: "neo4j"
  neo4j_password: "password"
  
  temporal:
    resolution: "15min"  # or "1hour"
    time_window: "7days"
    seasons: ["winter", "spring", "summer", "fall"]
  
  thresholds:
    complementarity_correlation: -0.3  # Negative correlation threshold
    peak_offset_hours: 4  # Minimum hours between peaks
    spatial_proximity: 200  # meters for potential sharing
    
  clustering:
    min_cluster_size: 3
    max_cluster_size: 20
    respect_transformer_boundary: true
```

### **⚙️ PROCESS**

#### **Core Transformation Logic**

```
1. Grid Infrastructure Layer
   CSV Grid Data → Graph Topology with Hierarchy
   
2. Building Layer  
   CSV Building Data → Enhanced Building Nodes with Derived Features
   
3. Temporal Layer
   Parquet Time-Series → Time-Indexed Energy States
   
4. Intelligence Layer
   Statistical Analysis → Complementarity Relationships & Initial Clusters
   
5. Asset Layer
   Potential Calculations → Solar/Battery/Electrification Opportunities
```

### **📤 OUTPUTS (Neo4j Graph Structure)**

#### **Final Knowledge Graph**
```
Nodes Created:
├── Substation (1 node)
├── MV_Transformer (2 nodes)  
├── LV_Network (6 nodes)
├── Building (150 nodes)
├── TimeSlot (672 nodes: 7 days × 96 intervals)
├── EnergyState (100,800 nodes: 150 buildings × 672 timeslots)
├── SolarSystem (60-80 nodes: buildings with solar potential)
├── BatterySystem (40-60 nodes: buildings with batteries)
├── EnergyCluster (10-20 initial clusters)
└── ComplementarityProfile (200-500 pairs)

Relationships Created:
├── FEEDS_FROM (9 edges: grid hierarchy)
├── CONNECTED_TO (150 edges: building to LV)
├── HAS_STATE_AT (100,800 edges: building to energy states)
├── DURING (100,800 edges: energy state to timeslot)
├── COMPLEMENTS (200-500 edges: building pairs)
├── COULD_SHARE_WITH (100-300 edges: proximity + complementarity)
├── CAN_INSTALL (100-200 edges: solar/battery potential)
└── MEMBER_OF (150 edges: initial cluster assignments)
```

### **📋 DETAILED STEPS**

#### **Step 1: Initialize Neo4j Connection & Schema**
```
Purpose: Set up database structure
Process:
1.1 Connect to Neo4j instance
1.2 Clear existing data (if requested)
1.3 Create constraints:
    - Unique IDs for all node types
    - Index on frequently queried properties
1.4 Define node labels and relationship types
Output: Empty graph with schema ready
```

#### **Step 2: Load Grid Infrastructure Hierarchy**
```
Purpose: Create electrical network topology
Process:
2.1 Create Substation node:
    - Properties: id, location, capacity_mva
2.2 Create MV_Transformer nodes:
    - Properties: id, location, capacity_kva
    - Link to substation via FEEDS_FROM
2.3 Create LV_Network nodes:
    - Properties: component_id, location, capacity_kva
    - Link to MV transformers via FEEDS_FROM
2.4 Validate hierarchy completeness
Output: Complete grid topology graph
```

#### **Step 3: Load and Enhance Buildings**
```
Purpose: Create building nodes with derived features
Process:
3.1 Create Building nodes with raw attributes:
    - Static: ogc_fid, area, height, type, orientation
    - Location: x, y coordinates
3.2 Calculate derived features:
    - Energy label (from age + type)
    - Insulation quality (from age + label)
    - Solar potential (from roof + orientation)
    - Electrification feasibility
3.3 Assign to LV networks:
    - Create CONNECTED_TO relationships
    - Add distance_to_transformer property
3.4 Calculate network position metrics:
    - Buildings on same transformer
    - Electrical centrality
Output: 150 building nodes with 30+ properties each
```

#### **Step 4: Load Temporal Energy Data**
```
Purpose: Create time-series representation
Process:
4.1 Create TimeSlot nodes:
    - One per 15-min interval (672 total)
    - Properties: timestamp, hour, day_of_week, is_weekend
4.2 Create EnergyState nodes:
    - One per building per timeslot
    - Properties: demands (elec/heat/cool), generation, storage
4.3 Calculate state metrics:
    - Net demand (demand - generation)
    - Is surplus (generation > demand)
    - Export potential
4.4 Link states:
    - Building HAS_STATE_AT EnergyState
    - EnergyState DURING TimeSlot
4.5 Add temporal sequences:
    - EnergyState FOLLOWS next_EnergyState
Output: Complete temporal graph layer
```

#### **Step 5: Compute Complementarity Relationships**
```
Purpose: Identify synergistic building pairs
Process:
5.1 For each LV network:
    - Get all building pairs
5.2 Extract demand profiles:
    - 7-day sequences for each building
5.3 Calculate complementarity metrics:
    - Correlation coefficient
    - Peak time offset
    - Average anti-correlation strength
5.4 Create COMPLEMENTS relationships:
    - If correlation < -0.3
    - Properties: score, peak_offset, type
5.5 Identify sharing potential:
    - If spatial_distance < 200m
    - AND has complementarity
    - Create COULD_SHARE_WITH
Output: Network of complementary relationships
```

#### **Step 6: Identify Asset Opportunities**
```
Purpose: Mark deployment potential
Process:
6.1 Solar potential assessment:
    - If roof_area > 50m² AND good orientation
    - Create SolarSystem node (status: 'potential')
    - Link: Building CAN_INSTALL SolarSystem
    - Calculate: recommended_capacity_kwp
6.2 Battery readiness:
    - If has solar OR high evening demand
    - Create BatterySystem node
    - Calculate: recommended_capacity_kwh
6.3 Electrification priority:
    - If poor energy label AND gas heating
    - Mark SHOULD_ELECTRIFY
    - Estimate: expected_COP, upgrade_needed
Output: Asset opportunity nodes and relationships
```

#### **Step 7: Create Initial Clusters**
```
Purpose: Bootstrap clustering for GNN training
Process:
7.1 Simple complementarity clustering:
    - Group buildings with strong COMPLEMENTS edges
    - Respect LV network boundaries
7.2 Create EnergyCluster nodes:
    - Properties: cluster_id, lv_network_id
7.3 Calculate cluster metrics:
    - Member diversity (building types)
    - Baseline peak (sum of individuals)
    - Expected peak reduction
7.4 Create MEMBER_OF relationships:
    - Building to Cluster
    - Add confidence score
Output: 10-20 initial clusters for analysis
```

#### **Step 8: Add Metadata and Statistics**
```
Purpose: Store computed metrics for querying
Process:
8.1 LV Network statistics:
    - Total buildings, diversity index
    - Current peak load, available capacity
8.2 Building statistics:
    - Load factor, peak demand time
    - Complementarity degree (# of COMPLEMENTS)
8.3 System-wide metrics:
    - Total solar potential
    - Electrification candidates
    - Average complementarity score
8.4 Create MetaData node:
    - Creation timestamp
    - Data quality metrics
    - Configuration used
Output: Searchable statistics layer
```

### **🔍 VALIDATION CHECKS**

After each step, validate:
```
✓ Node counts match expected
✓ All relationships have required properties  
✓ No orphaned nodes (except intentional)
✓ Hierarchy is complete (every building → LV → MV → Substation)
✓ Temporal sequence is continuous
✓ Complementarity scores are symmetric
✓ Clusters respect transformer boundaries
✓ Total energy is conserved in states
```

### **📊 SUMMARY METRICS TO REPORT**

After completion, output:
```
Graph Statistics:
- Total nodes: ~102,000
- Total relationships: ~102,000
- Memory usage: ~100-200 MB

Network Statistics:
- Buildings per LV: min/max/avg
- Complementarity pairs found: count
- Solar potential: total kWp
- Electrification candidates: count

Data Quality:
- Missing data points: count
- Constraint violations: count
- Processing time: seconds
```

**Ready to implement this step by step? Should we start with Step 1 - setting up the Neo4j connection and schema?** 🚀