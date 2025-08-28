# Comprehensive Documentation: Electrical Grid Hierarchy Analysis System

[A review on the complementarity of renewable energy sources: Concept, metrics, application and future research directions - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0038092X19311831)

## Project Overview

This project develops a sophisticated analytical framework for understanding electrical distribution networks in the Netherlands, focusing on identifying optimal locations for energy communities and prioritizing areas for energy efficiency interventions. The analysis is structured around the physical electrical infrastructure hierarchy, recognizing that energy communities must share common electrical infrastructure to effectively exchange energy.

## Core Problem Statement

The Dutch energy transition requires:

1. **Formation of Energy Communities**: Groups of buildings that can share renewable energy and balance each other's consumption patterns
2. **Targeted Retrofitting**: Upgrading buildings with poor energy efficiency (labels E, F, G) before 2030 regulatory deadlines
3. **Infrastructure-Based Planning**: Recognizing that energy sharing is only possible within the same electrical network segments

The challenge is identifying which areas under which transformers are:

- **Best suited for energy communities** (high diversity, complementary consumption patterns)
- **Most in need of intervention** (poor energy labels, old buildings)
- **Optimal for combined approaches** (both community formation and retrofitting)

## Data Architecture and Hierarchy

### 1. **Electrical Infrastructure Hierarchy**

The analysis follows the actual electrical grid structure:

```
HV (High Voltage) Network
    ↓
Substations
    ↓
MV (Medium Voltage) Network → MV Cable Groups
    ↓
Transformers (MV/LV Stations)
    ↓
LV (Low Voltage) Network → LV Cable Groups
    ↓
LV Cabinets
    ↓
Individual Buildings
```

### 2. **Why This Hierarchy Matters**

- **Energy Sharing Constraints**: Buildings can only share energy if connected to the same transformer
- **Load Balancing**: Diversity benefits occur at the transformer level
- **Investment Efficiency**: Upgrades benefit all buildings under the same infrastructure
- **Grid Stability**: Managing peak loads requires understanding aggregated demand at each level

## Database Tables Created

### **Step 1: Cable Network Segmentation**

**Tables Created:**

- `tlip_cable_segments`: Individual cable segments from the original network data
- `tlip_connected_groups`: Cables grouped by electrical continuity
- `tlip_segment_endpoints`: Connection points between cable segments

**Logic:**

- Original cable data often contains disconnected segments
- Used 0.5m tolerance to identify physically connected cables
- Created groups of electrically continuous cable networks
- Each group represents a single electrical circuit

**Purpose:**

- Understand which buildings share the same electrical path
- Identify isolated network sections (expected in clipped study area)
- Foundation for tracing electrical connectivity

### **Step 2-3: Station and Hierarchy Connections**

**Tables Created:**

- `tlip_group_stations`: Links cable groups to their stations (transformers, substations)
- `tlip_group_hierarchy`: Parent-child relationships (HV→MV→LV)
- `tlip_voltage_transitions`: Transformation points between voltage levels

**Logic:**

- Transformers connect MV and LV networks
- Substations connect HV and MV networks
- Buildings on the same LV group share a transformer
- Proximity-based matching (transformers to cable groups)

**Purpose:**

- Establish which buildings can form energy communities (same transformer)
- Understand power flow hierarchy
- Identify infrastructure bottlenecks

### **Step 4-8: Building Connections**

**Tables Created:**

- `tlip_building_connections`: Links each building to its serving cable
- `tlip_building_connection_points`: Exact connection points on cables
- `tlip_segment_connections`: Statistics per cable segment

**Key Features Added:**

- **Connection Types**:

  - ENDED: Cable terminates at building
  - ENTERED: Cable enters building
  - CROSSED: Cable passes through building
  - BY_DISTANCE: Connected by proximity
  - TOO_FAR: >150m distance (problematic)
- **MV Capability Flags**:

  - `is_mv_capable`: Large non-residential >3000m²
  - `has_mv_nearby`: MV cable within 100m

**Logic:**

- All buildings connect to LV (low voltage) by default
- Flag large buildings that could potentially use MV
- Identify problematic connections (>150m) needing infrastructure investment

## Analytical Tables for Decision Support

### **1. Hierarchical Grid Summary**

**Purpose**: Understand complete electrical path from transmission to end-user

**Key Queries Created:**

- MV stations ranked by buildings served
- LV groups under each MV station
- Building characteristics aggregated at each level
- Cross-tabulation of infrastructure and building properties

**Insights Provided:**

- Which transformers serve the most diverse building stock
- Load distribution across the network
- Infrastructure utilization patterns

### **2. District Analysis by Electrical Hierarchy**

**Purpose**: Evaluate districts for energy community potential and intervention needs

**MV-District Metrics Table:**
Combines geographical districts with electrical infrastructure, calculating:

**Energy Community (EC) Potential Score:**

- **Type Diversity** (0-10): Count of unique building types × 1.5
- **Size Heterogeneity** (0-10): Standard deviation of building areas
- **Functional Mix** (0-10): Balance between residential and non-residential
- **Temporal Diversity** (0-10): Presence of complementary pairs (office+residential)
- **Scale Factor** (0-10): Sufficient buildings for viable community

**Intervention Priority Score:**

- **Energy Efficiency Need** (0-10): % of D-G energy labels
- **Building Age Need** (0-10): % of pre-1975 buildings
- Higher scores = MORE urgent need for help

**Combined Classifications:**

- **EXCELLENT ⭐⭐⭐**: High EC potential + High intervention need (best targets)
- **INTERVENTION FOCUS**: Low EC potential + High intervention need
- **EC READY**: High EC potential + Low intervention need
- **LOW PRIORITY**: Low EC potential + Low intervention need

### **3. Building Summary Statistics**

**Purpose**: Comprehensive overview of building stock characteristics

**Aggregations Include:**

- Building functions (residential vs non-residential)
- Building types (detached, apartment, office, retail, etc.)
- Age distributions (6 periods from pre-1945 to post-2015)
- Energy label distributions (A++++ through G)
- Size categories (<50m² to >5000m²)
- Connection quality metrics

## Key Analytical Insights

### **Why Hierarchy Matters for Energy Communities**

1. **Electrical Constraints**: Energy can only be shared within the same transformer zone
2. **Load Balancing**: Diversity benefits occur at specific infrastructure points
3. **Investment Efficiency**: Infrastructure upgrades benefit all connected buildings
4. **Regulatory Compliance**: Meeting 2030 requirements for rental properties (E-G labels)

### **Optimal Target Identification**

The best areas for intervention have:

1. **High Building Diversity**: Mixed functions create temporal complementarity
2. **Poor Current Efficiency**: High percentage of E-G labels means maximum improvement potential
3. **Sufficient Scale**: Enough buildings to form viable community
4. **Common Infrastructure**: All served by same transformer

### **Complementarity Patterns Identified**

- **Office + Residential**: Day vs evening peak consumption
- **Retail + Residential**: Weekend vs weekday patterns
- **School + Residential**: School hours vs evening demand
- **Industrial + Any**: Potential 24/7 base load

## Decision Support Framework

### **For Energy Community Formation:**

1. Query MV stations with highest EC scores
2. Verify sufficient building diversity
3. Check for temporal complementarity patterns
4. Ensure common transformer connection

### **For Retrofit Prioritization:**

1. Identify areas with highest intervention scores
2. Focus on E-G label concentrations (2030 deadline)
3. Consider building age (insulation quality)
4. Evaluate upgrade cost-effectiveness

### **For Combined Programs:**

1. Target "HIGH-HIGH" areas (diverse but inefficient)
2. Implement community formation with retrofitting
3. Maximize both social and technical benefits
4. Achieve economies of scale

## Technical Implementation Notes

### **Spatial Considerations:**

- Used Dutch coordinate system (EPSG:28992)
- 0.5m tolerance for cable connections
- 150m threshold for problematic building connections
- Proximity-based station matching with distance limits

### **Data Quality Handling:**

- Filtered buildings <10m² as likely errors
- Handled NULL values in building characteristics
- Validated geometries before spatial operations
- Managed orphaned network segments (expected in clipped area)

### **Performance Optimizations:**

- Spatial indexes on all geometry columns
- Materialized views for complex aggregations
- Temporary tables for multi-step analyses
- Batch processing for large building sets

## Research Applications

This framework supports:

1. **GNN Training**: Labeled examples of good/bad energy community locations
2. **Policy Analysis**: Impact assessment of regulatory changes
3. **Investment Planning**: Optimal allocation of retrofit budgets
4. **Grid Planning**: Understanding load diversity and upgrade needs
5. **Social Equity**: Prioritizing help for inefficient buildings

## Key Outcomes

The analysis provides:

- **Ranked lists** of MV stations and districts by EC potential and intervention need
- **Detailed metrics** for each hierarchy level
- **Pattern identification** for complementary building groups
- **Clear classifications** for decision-making
- **Quantitative scores** for objective comparison

This comprehensive framework enables data-driven decisions for the Dutch energy transition, identifying where to form energy communities and where to prioritize efficiency interventions, all while respecting the physical constraints of the electrical infrastructure.
