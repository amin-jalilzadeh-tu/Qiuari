# Knowledge Graph Structure Schema

## Graph Schema Overview

```
Knowledge Graph for Energy System
├── Electrical Grid Hierarchy
├── Temporal Energy Data
├── Spatial Relationships
└── Asset Management
```

## Node Types and Their Properties

### 1. **HVSubstation** (High Voltage Substation)
```python
HVSubstation {
    substation_id: String       # Unique ID (e.g., "HV_SUB_001")
    name: String                 # Display name
    voltage_kv: Float           # 150.0 kV
    capacity_mva: Float         # Capacity in MVA
    group_id: String            # Grid group identifier
    hierarchy_level: Integer    # 0 (top of hierarchy)
    created_at: DateTime        
    added_by: String            
}
```

### 2. **MVStation** (Medium Voltage Station)
```python
MVStation {
    station_id: String          # Unique ID (e.g., "MV_STATION_0001")
    name: String                
    voltage_kv: Float           # 10.0 kV
    capacity_mva: Float         
    group_id: String            # From tlip_group_stations
    hv_parent: String           # Reference to parent HVSubstation
    hierarchy_level: Integer    # 1
    created_at: DateTime
    added_by: String
}
```

### 3. **CableGroup** (LV Cable Groups)
```python
CableGroup {
    group_id: String            # Unique ID (e.g., "LV_GROUP_0001")
    voltage_level: String       # "LV" (0.4kV)
    total_length: Float         # Total cable length in meters
    num_cables: Integer         # Number of cables in group
    mv_parent: String           # Reference to parent MVStation
    hierarchy_level: Integer    # 3
}
```

### 4. **Building**
```python
Building {
    # Identity
    ogc_fid: String             # Unique building ID
    district_name: String       # District location
    
    # Spatial Properties
    x: Float                    # X coordinate
    y: Float                    # Y coordinate
    
    # Building Characteristics
    building_function: String   # Residential/Commercial/Industrial
    build_year: Integer         # Construction year
    area_m2: Float              # Floor area
    height: Float               # Building height
    num_floors: Integer         # Number of floors
    
    # Energy Properties
    energy_label: String        # A/B/C/D/E/F/G
    insulation_quality: String  # Good/Average/Poor
    annual_consumption_kwh: Float
    solar_potential_kw: Float   # Rooftop solar potential
    
    # Asset Flags
    has_solar: Boolean          # Existing solar panels
    has_battery: Boolean        # Existing battery storage
    has_heat_pump: Boolean      # Existing heat pump
    
    # Hierarchy References
    upstream_mv_station: String # Supplying MV station
    upstream_lv_group: String   # Supplying LV group
    upstream_hv_substation: String # Root HV substation
    has_complete_hierarchy: Boolean
    hierarchy_level: Integer    # 4 (bottom of hierarchy)
    hierarchy_depth: Integer    # Path length from HV
}
```

### 5. **Transformer**
```python
Transformer {
    ogc_fid: String             # Unique transformer ID
    capacity_kva: Float         # Capacity in kVA
    type: String                # Transformer type
    voltage_primary: Float      # Primary voltage
    voltage_secondary: Float    # Secondary voltage
}
```

### 6. **Substation**
```python
Substation {
    name: String                # Substation name
    type: String                # Substation type
    location: String            # Physical location
}
```

### 7. **TimeSlot**
```python
TimeSlot {
    timestamp: DateTime         # Full timestamp
    hour: Integer              # Hour of day (0-23)
    day_of_week: Integer       # Day of week (0-6)
    month: Integer             # Month (1-12)
    season: String             # Season name
}
```

### 8. **AdjacencyCluster**
```python
AdjacencyCluster {
    cluster_id: String         # Unique cluster ID
    num_buildings: Integer     # Buildings in cluster
    avg_distance: Float        # Average distance between buildings
    cluster_type: String       # Cluster classification
}
```

### 9. **EnergyState** (Implicit from relationships)
```python
EnergyState {
    consumption_kwh: Float     # Energy consumption
    generation_kwh: Float      # Energy generation (if any)
    net_load_kwh: Float       # Net load (consumption - generation)
}
```

## Relationship Types and Structure

### Electrical Hierarchy Relationships
```cypher
(HVSubstation)-[:HV_SUPPLIES_MV]->(MVStation)
(MVStation)-[:MV_SUPPLIES_LV]->(CableGroup)
(CableGroup)-[:LV_SUPPLIES_BUILDING]->(Building)
(Building)-[:CONNECTED_TO]->(CableGroup)  # Original connection
```

### Temporal Relationships
```cypher
(EnergyState)-[:DURING]->(TimeSlot)
(EnergyState)-[:FOR_BUILDING]->(Building)
(ConsumptionProfile)-[:PROFILE_FOR]->(Building)
```

### Spatial Relationships
```cypher
(Building)-[:ADJACENT_TO]-(Building)  # Bidirectional
(Building)-[:IN_ADJACENCY_CLUSTER]->(AdjacencyCluster)
(Building)-[:NEAR_MV]->(MVStation)
```

### Infrastructure Relationships
```cypher
(CableSegment)-[:PART_OF]->(CableGroup)
(Building)-[:HAS_CONNECTION_POINT]->(ConnectionPoint)
(Building)-[:ON_SEGMENT]->(CableSegment)
(CableGroup)-[:FEEDS_FROM]->(CableGroup)  # LV hierarchy
(CableGroup)-[:CONNECTS_TO]->(Transformer)
```

### Asset Management Relationships
```cypher
(Building)-[:CAN_INSTALL {
    asset_type: String,        # "solar"/"battery"/"heat_pump"
    capacity_kw: Float,         # Potential capacity
    priority: Integer           # Installation priority
}]->(Asset)

(Building)-[:HAS_INSTALLED {
    installation_date: Date,
    capacity_kw: Float
}]->(Asset)

(Building)-[:SHOULD_ELECTRIFY {
    priority: Integer,
    potential_savings: Float
}]->(HeatingSystem)
```

## Query Patterns

### Hierarchical Traversal
```cypher
// Complete path from HV to Building
MATCH path = (hv:HVSubstation)-[:HV_SUPPLIES_MV]->(mv:MVStation)
            -[:MV_SUPPLIES_LV]->(lv:CableGroup)
            -[:LV_SUPPLIES_BUILDING]->(b:Building)
WHERE b.ogc_fid = $building_id
RETURN path
```

### Temporal Analysis
```cypher
// Get consumption profile for a building
MATCH (b:Building)<-[:FOR_BUILDING]-(es:EnergyState)-[:DURING]->(ts:TimeSlot)
WHERE b.ogc_fid = $building_id 
  AND ts.timestamp >= $start_date 
  AND ts.timestamp <= $end_date
RETURN ts.timestamp, es.consumption_kwh
```

### Spatial Clustering
```cypher
// Find adjacent buildings with similar characteristics
MATCH (b1:Building)-[:ADJACENT_TO]-(b2:Building)
WHERE b1.energy_label = b2.energy_label
  AND b1.building_function = b2.building_function
RETURN b1, b2
```

### Asset Optimization
```cypher
// Find buildings suitable for solar installation
MATCH (b:Building)-[r:CAN_INSTALL]->(a:Asset)
WHERE a.type = 'solar' 
  AND b.solar_potential_kw > 10
  AND NOT b.has_solar
RETURN b, r.capacity_kw, r.priority
ORDER BY r.priority
```

## Data Flow Architecture

```
SQL Database (PostgreSQL)
     ↓
KG Builder 1: Infrastructure & Buildings
     ↓
KG Builder 2: Spatial Relationships
     ↓
KG Builder 3: Temporal Data
     ↓
KG Hierarchy Updater: MV/HV Layers
     ↓
Connection Fixer: Complete Hierarchy
     ↓
KG Optimizer: Indexes & Constraints
     ↓
Neo4j Knowledge Graph (Complete)
```

## Graph Features for GNN

### Node Features Available
- **Buildings**: 15+ features (area, height, energy_label, consumption, etc.)
- **Grid Nodes**: voltage_level, capacity, hierarchy_level
- **Temporal**: hour, day_of_week, season
- **Spatial**: x, y coordinates, adjacency

### Edge Features Available
- **Hierarchy**: voltage drops, power flow direction
- **Temporal**: time-based consumption patterns
- **Spatial**: distance, adjacency strength
- **Assets**: capacity, installation potential

## Usage in GNN Pipeline

```python
# Node types for heterogeneous GNN
node_types = ['Building', 'CableGroup', 'MVStation', 'HVSubstation']

# Edge types for message passing
edge_types = [
    ('Building', 'CONNECTED_TO', 'CableGroup'),
    ('CableGroup', 'MV_SUPPLIES_LV', 'MVStation'),
    ('MVStation', 'HV_SUPPLIES_MV', 'HVSubstation'),
    ('Building', 'ADJACENT_TO', 'Building')
]

# Features for each node type
feature_dims = {
    'Building': 15,      # All building properties
    'CableGroup': 4,     # Electrical properties
    'MVStation': 5,      # Station properties
    'HVSubstation': 4    # Substation properties
}
```