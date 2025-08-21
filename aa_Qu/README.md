# Energy Grid Analysis - SQL & Knowledge Graph Components

## Initial Files for Review

I'm sharing the SQL processing and Knowledge Graph (KG) creation components of our energy grid analysis system. These files represent the data extraction and hierarchical structuring phase of our project.

## 🎯 What's Included Today

### 1. SQL Data Processing (`grid_analysis_project/`)

These files handle the hierarchical selection and processing of grid data:

#### SQL Scripts (`grid_analysis_project/sql_scripts/`):

- **`STEP 1.sql`** - Initial data extraction from raw tables
- **`STEPS 2-3.sql`** - Data transformation and cleaning
- **`STEPS 4-8.sql`** - Hierarchical relationship building
- **`MV-LV-Based Analysis.sql`** - Network-level analysis
- **`HIERARCHICAL ELECTRICAL GRID SUMMARY.sql`** - Complete hierarchy overview

### 2. Knowledge Graph Builders

Two versions for different use cases:

- **`kg_builder_1.py`**

  - Creates nodes: Buildings, Transformers, Networks
  - Basic relationships: CONNECTED_TO, PART_OF
- **`kg_builder_2.py`**

  - Additional relationships: ADJACENT_TO, SHARES_TRANSFORMER
  - Includes spatial indexing and richer properties
  - Better handling of hierarchical grid structure

## 📊 Data Hierarchy Flow

```
Raw Data (SQL Tables)
    ↓
SQL Scripts (Hierarchical Selection)
    ↓
Structured Grid Data
    ↓
KG Builders (Neo4j Graph Creation)
    ↓
Knowledge Graph
```

## 🔄 Current Status

### ✅ Ready for Review:

- SQL processing pipeline
- Hierarchical data selection logic
- KG creation structure
- Test data generation

## 💡 Key Points to Understand

1. **SQL Files**: Define how we extract and structure the hierarchical grid data from source databases
2. **KG Builders**: Transform structured data into Neo4j graph format for GNN processing
3. **Hierarchy**: MV Transformer → LV Network → Buildings (with all relationships preserved)

## 🔜 Coming Next

I'll send the following components once finalized:

- Created database in neo4j
- UBEM simulation results
- GNN final model
- Figures and data in ase you need
- more documents

Please review these files to understand the data processing and KG creation approach. Let me know if you have any questions about the hierarchical selection logic or KG structure.

Best regards,
Amin

---
