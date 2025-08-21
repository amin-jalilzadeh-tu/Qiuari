# Comprehensive Knowledge Graph Construction Documentation

## Energy District Knowledge Graph: Architecture and Implementation

This document provides a detailed technical explanation of how the Energy District Knowledge Graph is constructed from PostgreSQL data sources, incorporating grid infrastructure, building characteristics, energy assets, and adjacency relationships.

## 1. DATA INPUTS

### Primary Data Sources (PostgreSQL Database)

#### Grid Infrastructure Tables:

- **`amin_grid.tlip_onderstations`** - Electrical substations (HV level)
- **`amin_grid.tlip_middenspanningsinstallaties`** - Medium voltage transformers
- **`amin_grid.tlip_laagspanningsverdeelkasten`** - Low voltage distribution cabinets
- **`amin_grid.tlip_connected_groups`** - Cable groups (connected components)
- **`amin_grid.tlip_cable_segments`** - Individual cable segments
- **`amin_grid.tlip_group_hierarchy`** - Hierarchical relationships between cable groups
- **`amin_grid.tlip_group_stations`** - Connections between cable groups and stations
- **`amin_grid.tlip_building_connections`** - Building-to-grid connection metadata
- **`amin_grid.tlip_building_connection_points`** - Precise connection points on cables

#### Building Data Tables:

- **`amin.buildings_1_deducted`** - Main building characteristics including:
  - Spatial data (x, y coordinates, geometry)
  - Physical attributes (area, height, roof areas)
  - Energy labels and building year
  - Environmental indices (NDVI, NTL, NDWI)
  - Housing types and functions

#### Temporal Data (Parquet Files):

- **`energy_profiles.parquet`** - Time-series energy consumption/generation data

## 2. NODE TYPES AND ATTRIBUTES

### Infrastructure Nodes

#### **Substation** (GridComponent)

```cypher
{
  station_id: INTEGER (unique),
  x, y: FLOAT (coordinates),
  voltage_level: 'HV',
  component_type: 'substation',
  geom_wkt: STRING (geometry)
}
```

#### **Transformer** (GridComponent)

```cypher
{
  transformer_id: INTEGER (unique),
  x, y: FLOAT,
  voltage_level: 'MV',
  component_type: 'transformer',
  geom_wkt: STRING,
  baseline_lv_count: INTEGER,
  baseline_peak_kw: FLOAT,
  baseline_building_count: INTEGER
}
```

#### **LVCabinet** (GridComponent)

```cypher
{
  cabinet_id: INTEGER (unique),
  x, y: FLOAT,
  voltage_level: 'LV',
  component_type: 'lv_cabinet',
  geom_wkt: STRING
}
```

#### **CableGroup** (GridComponent)

```cypher
{
  group_id: STRING (unique),
  voltage_level: STRING ('HV'/'MV'/'LV'),
  segment_count: INTEGER,
  total_length_m: FLOAT,
  x, y: FLOAT (centroid),
  bbox_wkt: STRING,
  component_type: 'cable_group',
  
  // Baseline metrics (for LV groups)
  baseline_building_count: INTEGER,
  baseline_peak_kw: FLOAT,
  baseline_avg_demand_kw: FLOAT,
  baseline_load_factor: FLOAT,
  baseline_solar_penetration: FLOAT,
  baseline_battery_penetration: FLOAT,
  baseline_hp_penetration: FLOAT,
  baseline_diversity: INTEGER
}
```

#### **CableSegment**

```cypher
{
  segment_id: INTEGER (unique),
  original_fid: INTEGER,
  voltage_level: STRING,
  group_id: STRING,
  length_m: FLOAT,
  start_x, start_y: FLOAT,
  end_x, end_y: FLOAT
}
```

#### **ConnectionPoint**

```cypher
{
  point_id: INTEGER (unique),
  building_id: INTEGER,
  segment_id: INTEGER,
  group_id: STRING,
  connection_type: STRING,
  x, y: FLOAT,
  distance_along_segment: FLOAT,
  segment_fraction: FLOAT,
  connection_distance_m: FLOAT,
  is_direct: BOOLEAN
}
```

### Building Nodes

#### **Building**

```cypher
{
  // Core identifiers
  ogc_fid: INTEGER (unique),
  x, y: FLOAT,
  
  // Physical characteristics
  building_function: STRING,
  residential_type: STRING,
  non_residential_type: STRING,
  area: FLOAT (m²),
  height: FLOAT (m),
  age_range: STRING,
  building_year: INTEGER,
  building_orientation_cardinal: STRING,
  district_name: STRING,
  neighborhood_name: STRING,
  housing_type: STRING (woningtype),
  
  // Roof data
  flat_roof_area: FLOAT,
  sloped_roof_area: FLOAT,
  suitable_roof_area: FLOAT, // Calculated
  
  // Grid connection
  lv_group_id: STRING,
  connection_segment_id: INTEGER,
  connection_type: STRING,
  connection_distance_m: FLOAT,
  is_mv_capable: BOOLEAN,
  has_mv_nearby: BOOLEAN,
  nearest_mv_distance_m: FLOAT,
  is_problematic: BOOLEAN,
  connection_reason: STRING,
  
  // Energy characteristics
  energy_label: STRING,
  energy_label_simple: STRING, // Simplified A-G
  insulation_quality: STRING,
  solar_potential: STRING ('high'/'medium'/'low'/'none'),
  solar_capacity_kwp: FLOAT, // Calculated
  battery_readiness: STRING,
  electrification_feasibility: STRING,
  expected_cop: FLOAT, // Heat pump efficiency
  heating_system: STRING,
  
  // Environmental indices
  vegetation_index: FLOAT (NDVI),
  nighttime_lights: FLOAT (NTL),
  water_index: FLOAT (NDWI),
  
  // Current assets
  has_solar: BOOLEAN,
  has_battery: BOOLEAN,
  has_heat_pump: BOOLEAN,
  
  // Performance metrics
  peak_demand_kw: FLOAT,
  avg_demand_kw: FLOAT,
  load_factor: FLOAT,
  demand_variability: FLOAT,
  peak_solar_kw: FLOAT,
  avg_net_demand_kw: FLOAT,
  self_consumption_ratio: FLOAT,
  
  // Adjacency data (from Part 2)
  north_shared_length: FLOAT,
  south_shared_length: FLOAT,
  east_shared_length: FLOAT,
  west_shared_length: FLOAT,
  north_facade_length: FLOAT,
  south_facade_length: FLOAT,
  east_facade_length: FLOAT,
  west_facade_length: FLOAT,
  num_shared_walls: INTEGER,
  total_shared_length: FLOAT,
  adjacency_type: STRING, // 'ISOLATED'/'MIDDLE_ROW'/'END_ROW'/'CORNER'/'COURTYARD'
  adjacency_count: INTEGER,
  avg_adjacency_strength: FLOAT,
  max_complementarity: FLOAT,
  has_adjacent_neighbors: BOOLEAN,
  shared_wall_directions: LIST,
  isolation_factor: FLOAT,
  thermal_efficiency_boost: FLOAT
}
```

### Asset Nodes

#### **SolarSystem**

```cypher
{
  system_id: STRING (unique),
  building_id: INTEGER,
  status: STRING ('existing'/'potential'),
  
  // For existing systems
  installed_capacity_kwp: FLOAT,
  installation_year: INTEGER,
  degradation_factor: FLOAT (0.98),
  orientation_efficiency: FLOAT,
  
  // For potential systems
  potential_capacity_kwp: FLOAT,
  recommended_capacity_kwp: FLOAT,
  payback_years: FLOAT
}
```

#### **BatterySystem**

```cypher
{
  system_id: STRING (unique),
  building_id: INTEGER,
  status: STRING ('existing'/'potential'),
  
  // Capacity sizing
  installed_capacity_kwh: FLOAT,
  recommended_capacity_kwh: FLOAT,
  power_rating_kw: FLOAT,
  
  // Performance
  round_trip_efficiency: FLOAT (0.9),
  estimated_cycles_per_year: INTEGER
}
```

#### **HeatPumpSystem**

```cypher
{
  system_id: STRING (unique),
  building_id: INTEGER,
  status: STRING ('existing'/'potential'),
  
  expected_cop: FLOAT,
  heating_capacity_kw: FLOAT,
  installation_year: INTEGER,
  
  // For potential systems
  upgrade_required: BOOLEAN,
  estimated_annual_savings_euro: FLOAT
}
```

### Temporal Nodes

#### **TimeSlot**

```cypher
{
  slot_id: STRING (unique),
  timestamp: DATETIME,
  hour_of_day: INTEGER,
  day_of_week: INTEGER,
  is_weekend: BOOLEAN,
  season: STRING,
  time_of_day: STRING
}
```

#### **EnergyState**

```cypher
{
  state_id: STRING (unique),
  building_id: INTEGER,
  timeslot_id: STRING,
  
  // Energy flows (kW)
  electricity_demand_kw: FLOAT,
  heating_demand_kw: FLOAT,
  cooling_demand_kw: FLOAT,
  solar_generation_kw: FLOAT,
  battery_soc_kwh: FLOAT,
  battery_charge_kw: FLOAT,
  battery_discharge_kw: FLOAT,
  
  // Net position
  net_demand_kw: FLOAT,
  is_surplus: BOOLEAN,
  export_potential_kw: FLOAT,
  import_need_kw: FLOAT
}
```

### Cluster Nodes (Part 2)

#### **AdjacencyCluster**

```cypher
{
  cluster_id: STRING (unique),
  cluster_type: STRING ('ROW_HOUSES'/'CORNER_BLOCK'/'COURTYARD_BLOCK'/'APARTMENT_COMPLEX'),
  member_count: INTEGER,
  lv_group_id: STRING,
  district_name: STRING,
  created_at: DATETIME,
  pattern: STRING ('LINEAR'/'L_SHAPE'/'ENCLOSED'/'VERTICAL'),
  thermal_benefit: STRING,
  cable_savings: STRING,
  avg_shared_walls: FLOAT,
  
  // Energy metrics
  avg_solar_potential_kwp: FLOAT,
  solar_penetration: FLOAT,
  battery_penetration: FLOAT,
  hp_penetration: FLOAT,
  function_diversity: INTEGER,
  energy_sharing_potential: STRING
}
```

## 3. RELATIONSHIPS AND ATTRIBUTES

### Grid Infrastructure Relationships

#### **FEEDS_FROM** (CableGroup → CableGroup)

```cypher
{
  connection_via: STRING,
  via_station_fid: INTEGER,
  confidence_score: FLOAT
}
```

#### **CONNECTS_TO** (CableGroup → Transformer/Substation/LVCabinet)

```cypher
{
  connection_type: STRING,
  distance_m: FLOAT,
  confidence_score: FLOAT
}
```

#### **PART_OF** (CableSegment → CableGroup)

No attributes - membership relationship

### Building-Grid Relationships

#### **CONNECTED_TO** (Building → CableGroup)

```cypher
{
  connection_type: STRING,
  distance_m: FLOAT,
  is_problematic: BOOLEAN
}
```

#### **HAS_CONNECTION_POINT** (Building → ConnectionPoint)

No attributes

#### **ON_SEGMENT** (ConnectionPoint → CableSegment)

```cypher
{
  fraction: FLOAT,
  distance_along: FLOAT
}
```

#### **NEAR_MV** (Building → CableGroup[MV])

```cypher
{
  distance_m: FLOAT
}
```

### Asset Relationships

#### **HAS_INSTALLED** (Building → SolarSystem/BatterySystem/HeatPumpSystem)

```cypher
{
  install_date: DATE
}
```

#### **CAN_INSTALL** (Building → SolarSystem/BatterySystem)

```cypher
{
  feasibility_score: FLOAT,
  priority: STRING,
  requires_solar: BOOLEAN (for batteries)
}
```

#### **SHOULD_ELECTRIFY** (Building → HeatPumpSystem)

```cypher
{
  priority: INTEGER (1-5),
  expected_cop: FLOAT,
  requires_insulation_upgrade: BOOLEAN
}
```

### Temporal Relationships

#### **HAS_STATE_AT** (Building → EnergyState)

No attributes

#### **DURING** (EnergyState → TimeSlot)

No attributes

### Adjacency Relationships (Part 2)

#### **ADJACENT_TO** (Building ↔ Building)

Bidirectional relationship with:

```cypher
{
  // Physical connection
  wall_pair: STRING (e.g., 'north-south'),
  my_wall: STRING,
  their_wall: STRING,
  my_shared_length: FLOAT,
  their_shared_length: FLOAT,
  match_quality: FLOAT,
  distance_m: FLOAT,
  adjacency_strength: FLOAT,
  
  // Energy implications
  thermal_coupling: BOOLEAN,
  cable_distance: FLOAT,
  transmission_loss_factor: FLOAT (0.001 * distance),
  energy_sharing_viable: BOOLEAN,
  thermal_resistance_reduction: FLOAT,
  
  // Diversity metrics
  function_diversity: FLOAT (1.0-2.0),
  solar_diversity: FLOAT (1.0-2.0),
  complementarity_potential: FLOAT,
  priority_for_sharing: STRING,
  
  created_at: DATETIME
}
```

#### **IN_ADJACENCY_CLUSTER** (Building → AdjacencyCluster)

```cypher
{
  role: STRING ('CENTER'/'MEMBER'),
  joined_at: DATETIME
}
```

## 4. KEY FORMULAS AND CALCULATIONS

### Building Energy Features

#### Suitable Roof Area

```python
suitable_roof_area = flat_roof_area + (
    sloped_roof_area * 0.6 if orientation in ['S', 'SE', 'SW']
    else sloped_roof_area * 0.3
)
```

#### Solar Capacity Potential

```python
solar_capacity_kwp = suitable_roof_area * 0.175 * 0.85
# 0.175 kWp/m² = power density
# 0.85 = system efficiency (inverter losses, temperature derating)
```

#### Expected Heat Pump COP by Energy Label

```python
cop_mapping = {
    'A': 4.2,  # 35°C water temp
    'B': 3.7,  # 40°C water temp
    'C': 3.2,  # 45°C water temp
    'D': 2.7,  # 50°C water temp
    'E': 2.3,  # 55°C water temp
    'F': 2.0,  # Needs major upgrade
    'G': 1.7   # Not suitable
}
```

#### Heating Capacity Requirements

```python
heating_capacity_kw = area * age_factor
where age_factor = {
    '< 1945': 0.075,
    '1945-1975': 0.065,
    '1975-1990': 0.055,
    '1990-2005': 0.045,
    '2005-2015': 0.035,
    '> 2015': 0.025
}
```

#### Battery Sizing (Dutch Market Standards)

```python
battery_capacity_kwh = {
    area < 100: 5.0,    # Small homes
    area < 150: 7.5,    # Average homes
    area < 200: 10.0,   # Larger homes  
    area < 300: 13.5,   # Big homes (Tesla Powerwall size)
    else: 15.0          # Capped at 15 kWh
}
power_rating_kw = battery_capacity_kwh / 4  # 0.25C rate
```

### Grid Metrics

#### Load Factor

```python
load_factor = avg_demand_kw / peak_demand_kw
```

#### Demand Variability

```python
demand_variability = stdev(demand) / avg(demand)
```

#### Self-Consumption Ratio

```python
self_consumption_ratio = (avg_demand - avg_net_demand) / avg_demand
# Only for buildings with solar
```

#### Transmission Loss Factor

```python
transmission_loss_factor = 0.001 * distance_m  # 0.1% per meter for LV cables
```

### Adjacency Metrics

#### Isolation Factor (Thermal)

```python
isolation_factor = {
    0 adjacencies: 1.0,    # Detached (baseline)
    1 adjacency: 0.85,     # Semi-detached (15% reduction)
    2 adjacencies: 0.70,   # Row house (30% reduction)
    3+ adjacencies: 0.60   # Apartment (40% reduction)
}
```

#### Thermal Efficiency Boost

```python
thermal_efficiency_boost = {
    0 adjacencies: 1.0,    # No improvement
    1 adjacency: 1.15,     # 15% improvement
    2 adjacencies: 1.30,   # 30% improvement
    3+ adjacencies: 1.40   # 40% improvement
}
```

#### Wall Match Quality

```python
match_quality = min(wall1_length, wall2_length) / max(wall1_length, wall2_length)
# Must be > 0.5 for valid adjacency
```

#### Complementarity Potential

```python
complementarity_potential = function_diversity * solar_diversity * match_quality
where:
  function_diversity = 2.0 if different building functions, else 1.0
  solar_diversity = 2.0 if one has solar and other doesn't, else 1.0
```

#### Adjacency Type Classification

```python
adjacency_type = {
    num_shared_walls == 0: 'ISOLATED',
    opposite_walls_shared: 'MIDDLE_ROW',  # N-S or E-W
    num_shared_walls == 1: 'END_ROW',
    num_shared_walls == 2: 'CORNER',
    num_shared_walls >= 3: 'COURTYARD'
}
```

## 5. PROCESSING LOGIC FLOW

### Part 1: Main KG Construction

1. **Schema Creation**

   - Create unique constraints for all node types
   - Create performance indexes
2. **Grid Infrastructure Loading**

   - Load substations, transformers, LV cabinets from PostgreSQL
   - Create cable groups and segments
   - Establish hierarchy (FEEDS_FROM relationships)
   - Connect groups to stations (CONNECTS_TO)
3. **Building Data Processing**

   - Load buildings with connection metadata
   - Calculate derived features (suitable roof area, solar potential)
   - Estimate energy labels from age if missing
   - Create connection points
4. **Asset Creation**

   - Identify existing assets based on rules:
     - Solar: 28% of high/medium potential buildings
     - Battery: 20% of solar-equipped buildings
     - Heat pumps: 8% of A/B labeled buildings
   - Create potential assets for remaining buildings
5. **Temporal Data Integration**

   - Load time-series from parquet
   - Create TimeSlot nodes for each timestamp
   - Create EnergyState nodes with net position calculations
6. **Baseline Metrics**

   - Calculate building-level statistics
   - Aggregate to LV group level
   - Roll up to transformer level
   - Create system-wide baseline

### Part 2: Adjacency Enhancement

1. **Data Update**

   - Load shared wall data from PostgreSQL
   - Update existing building nodes with wall lengths
2. **Adjacency Detection**

   - Find buildings in same LV group within 5m
   - Match shared walls (north-south, east-west pairs)
   - Validate match quality (>50% length similarity)
   - Create bidirectional ADJACENT_TO relationships
3. **Cluster Formation**

   - **Row Houses**: Connected chains of MIDDLE_ROW/END_ROW buildings
   - **Corner/Courtyard**: Buildings with 2+ adjacencies
   - **Apartments**: Same building type within 50m
4. **Metric Enhancement**

   - Add adjacency counts to buildings
   - Calculate thermal benefits
   - Identify high-complementarity pairs
   - Update cluster-level metrics

## 6. DATA VALIDATION LOGIC

### Housing Type Validation

```python
expected_adjacencies = {
    'rijtjeswoning' (row house): 2,
    'vrijstaand' (detached): 0,
    'twee_onder_1_kap' (semi-detached): 1,
    'appartement': varies
}
```

### Reciprocal Relationship Check

- Every ADJACENT_TO relationship must have a reverse relationship
- Wall pairs must be complementary (north-south, east-west)

### Distance Validation

- True adjacencies must be within 5m
- Energy sharing viable only within 3m
- MV capability checked within 200m radius
