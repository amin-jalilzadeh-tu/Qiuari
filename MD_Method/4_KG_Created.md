# Your Knowledge Graph Structure - What You Actually Have

## 📊 **NODES IN YOUR KG**

### 1. **Infrastructure Nodes**
```
🏭 Substation (1 node)
├── id: 'SUB_001'
├── capacity_mva: 50
├── voltage_level: 'HV'
└── x, y: coordinates

⚡ MV_Transformer (2 nodes)
├── id: 'MV_TRANS_001', 'MV_TRANS_002'
├── capacity_kva: 1000
├── substation_id: reference
└── voltage_level: 'MV'

🔌 LV_Network (6 nodes)
├── component_id: 1-6
├── network_id: 'LV_NET_001' to 'LV_NET_006'
├── capacity_kva: 250
├── mv_transformer_id: reference
└── voltage_level: 'LV'
```

### 2. **Building Nodes (~145 nodes)**
```
🏠 Building
├── Identity
│   ├── ogc_fid: unique ID
│   ├── x, y: coordinates
│   └── lv_component_id: which LV network
│
├── Physical Attributes
│   ├── building_function: 'residential'/'non_residential'
│   ├── residential_type: 'Detached'/'Semi-detached'/'Terrace'/'Apartment'
│   ├── non_residential_type: 'Office'/'Retail'/'Industrial'
│   ├── area: floor area in m²
│   ├── height: building height
│   ├── age_range: '<1945'/'1945-1975'/etc.
│   └── building_orientation_cardinal: 'N'/'S'/'E'/'W'/etc.
│
├── Roof Data
│   ├── roof_area: total roof m²
│   ├── flat_roof_area: flat portion
│   ├── sloped_roof_area: sloped portion
│   └── suitable_roof_area: usable for solar
│
├── Energy Features
│   ├── energy_label: 'A' to 'G' (derived)
│   ├── insulation_quality: 'poor'/'fair'/'good'/'excellent'
│   ├── solar_potential: 'high'/'medium'/'low'/'none'
│   ├── battery_readiness: 'ready'/'conditional'
│   ├── electrification_feasibility: 'immediate'/'conditional'/'upgrade_needed'
│   └── expected_cop: 1.5 to 4.0 (heat pump efficiency)
│
├── Current Assets
│   ├── has_solar: true/false
│   ├── has_battery: true/false
│   ├── has_heat_pump: true/false
│   ├── heating_system: 'gas'/'heat_pump'
│   └── solar_capacity_kwp: installed capacity
│
├── Calculated Metrics (from baseline)
│   ├── peak_demand_kw: maximum demand
│   ├── avg_demand_kw: average demand
│   ├── load_factor: avg/peak ratio
│   ├── demand_variability: coefficient of variation
│   └── self_consumption_ratio: if has solar
│
└── Adjacency Data (if adjacency module run)
    ├── num_shared_walls: 0-4
    ├── adjacency_type: 'ISOLATED'/'END_UNIT'/'MIDDLE_ROW'/'CORNER'
    ├── north/south/east/west_shared_length: meters
    └── total_shared_length: sum of shared walls
```

### 3. **Asset Nodes**
```
☀️ SolarSystem
├── system_id: 'SOLAR_EXISTING_X' or 'SOLAR_POTENTIAL_X'
├── building_id: reference
├── status: 'existing'/'potential'
├── installed_capacity_kwp: actual if existing
├── potential_capacity_kwp: calculated max
├── orientation_efficiency: 0.7-1.0
└── installation_year: if existing

🔋 BatterySystem
├── system_id: 'BATTERY_EXISTING_X' or 'BATTERY_POTENTIAL_X'
├── status: 'existing'/'potential'
├── installed_capacity_kwh: 5/10/15 kWh
├── power_rating_kw: C/4 rate
└── round_trip_efficiency: 0.9

🔥 HeatPumpSystem
├── system_id: 'HP_EXISTING_X' or 'HP_POTENTIAL_X'
├── status: 'existing'/'potential'
├── expected_cop: efficiency rating
├── heating_capacity_kw: area * 0.05
└── upgrade_required: true/false
```

### 4. **Temporal Nodes**
```
⏰ TimeSlot (672 nodes for 7 days @ 15-min)
├── slot_id: 'TS_0' to 'TS_671'
├── timestamp: datetime
├── hour_of_day: 0-23
├── day_of_week: 0-6
├── is_weekend: true/false
├── season: 'winter'/'spring'/'summer'/'fall'
└── time_of_day: 'morning'/'afternoon'/'evening'/'night'

📊 EnergyState (~97,440 nodes: 145 buildings × 672 timeslots)
├── state_id: 'ES_buildingID_slotID'
├── building_id: reference
├── timeslot_id: reference
├── electricity_demand_kw: actual demand
├── heating_demand_kw: thermal demand
├── cooling_demand_kw: cooling demand
├── solar_generation_kw: if has solar
├── battery_soc_kwh: state of charge
├── battery_charge_kw: charging rate
├── battery_discharge_kw: discharge rate
├── net_demand_kw: demand - generation
├── is_surplus: true if generation > demand
├── export_potential_kw: excess generation
└── import_need_kw: grid import needed
```

### 5. **System Nodes**
```
📈 SystemBaseline (1 node)
├── total_buildings: count
├── system_peak_kw: sum of peaks
├── avg_load_factor: system average
├── solar_penetration: % with solar
├── battery_penetration: % with batteries
└── hp_penetration: % with heat pumps

📋 Metadata (1 node)
├── created_at: timestamp
├── total_nodes: count
├── total_relationships: count
└── stage: 'pre_gnn'
```

## 🔗 **RELATIONSHIPS IN YOUR KG**

### Infrastructure Hierarchy
```
Substation <-[FEEDS_FROM]- MV_Transformer
    ↑ capacity_kva: 1000

MV_Transformer <-[FEEDS_FROM]- LV_Network
    ↑ capacity_kva: 250

LV_Network <-[CONNECTED_TO]- Building
    ↑ distance_m: calculated
    ↑ cable_type: 'underground'
```

### Asset Relationships
```
Building -[HAS_INSTALLED]-> SolarSystem (existing)
    ↓ install_date: '2023-01-01'

Building -[HAS_INSTALLED]-> BatterySystem (existing)
    ↓ install_date: '2023-01-01'

Building -[HAS_INSTALLED]-> HeatPumpSystem (existing)
    ↓ install_date: '2022-01-01'

Building -[CAN_INSTALL]-> SolarSystem (potential)
    ↓ feasibility_score: 0.5-0.9
    ↓ priority: 'high'/'medium'/'low'

Building -[CAN_INSTALL]-> BatterySystem (potential)
    ↓ feasibility_score: 0.6-0.8
    ↓ requires_solar: true/false

Building -[SHOULD_ELECTRIFY]-> HeatPumpSystem (potential)
    ↓ priority: 1-5 (1=highest)
    ↓ expected_cop: efficiency
    ↓ requires_insulation_upgrade: true/false
```

### Temporal Relationships
```
Building -[HAS_STATE_AT]-> EnergyState
EnergyState -[DURING]-> TimeSlot
```

### Adjacency Relationships (if module run)
```
Building -[ADJACENT_TO]-> Building
    ↓ wall_pair: 'north-south'/'east-west'/etc
    ↓ my_wall: which wall is shared
    ↓ shared_length: meters
    ↓ distance_m: physical distance
    ↓ thermal_coupling: true
    ↓ energy_sharing_viable: true
    ↓ function_diversity: 1.0-1.5
    ↓ complementarity_potential: calculated

Building -[IN_ADJACENCY_CLUSTER]-> AdjacencyCluster
```

## 📊 **SUMMARY STATISTICS**

Your KG contains approximately:
- **~98,300 total nodes**:
  - 9 infrastructure nodes
  - 145 buildings
  - ~300 asset nodes (existing + potential)
  - 672 timeslot nodes
  - ~97,440 energy state nodes
  - 2 system nodes

- **~195,500 total relationships**:
  - 9 FEEDS_FROM (hierarchy)
  - 145 CONNECTED_TO (building to LV)
  - ~100 HAS_INSTALLED (existing assets)
  - ~200 CAN_INSTALL/SHOULD_ELECTRIFY (potential)
  - 97,440 HAS_STATE_AT (building to states)
  - 97,440 DURING (states to timeslots)
  - ~300 ADJACENT_TO (if adjacency run)

## 🎯 **What's Ready for GNN**

Your KG provides everything needed for GNN processing:
1. **Graph topology** via infrastructure relationships
2. **Rich node features** from building attributes
3. **Temporal patterns** from energy states
4. **Deployment opportunities** identified
5. **Baseline metrics** for comparison
6. **Physical constraints** encoded in hierarchy

The GNN will use this to discover:
- COMPLEMENTS relationships (building pairs)
- EnergyCluster nodes (optimal groupings)
- DeploymentScenario recommendations
- Performance improvements vs baseline