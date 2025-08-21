# 📋 EXECUTION ORDER OF PYTHON FILES

## 🔄 Complete System Flow

### **PHASE 1: DATA GENERATION & SETUP** (One-time setup)
**Purpose:** Generate synthetic data and populate Neo4j database

```
1. mimic_data_generator.py
   ├── Creates synthetic building data
   ├── Generates energy profiles
   ├── Creates grid topology
   └── Outputs: CSVs + kg_cypher_commands.txt

2. kg_builder.py (or kg_builder_1.py)
   ├── Reads CSV files
   ├── Calculates additional features
   ├── Creates Neo4j schema
   └── Populates Neo4j database

3. load_mimic_data_to_neo4j.py
   ├── Executes Cypher commands
   └── Loads data into Neo4j

4. update_neo4j_buildings.py
   ├── Updates building nodes
   └── Adds energy profile statistics
```

---

### **PHASE 2: MAIN EXECUTION** (Runtime)
**Purpose:** Train model or run inference

```
5. main.py [ENTRY POINT]
   ├── Parses command line arguments
   ├── Initializes system
   └── Routes to: train, infer, or evaluate

   When main.py runs, it imports and uses:
   
   6. utils/logger.py
      └── Sets up logging system
   
   7. utils/metrics_tracker.py
      └── Tracks performance metrics
   
   8. data/kg_connector.py
      ├── Connects to Neo4j
      └── Queries graph data
   
   9. data/kg_extractor.py
      ├── Extracts graph structure
      └── Formats for processing
   
   10. data/graph_builder.py
       ├── Converts Neo4j data to PyTorch
       └── Creates node features (14 dims)
   
   11. data/feature_engineering.py
       ├── Creates additional features
       └── Normalizes data
   
   12. data/preprocessor.py
       └── Preprocesses raw data
   
   13. data/data_loader.py
       ├── Creates data batches
       └── Handles train/val/test splits
```

---

### **PHASE 3A: TRAINING PATH** (If command = train)
```
14. models/base_gnn.py
    ├── Defines GNN architecture
    └── GraphSAGE layers

15. models/attention_layers.py
    └── Attention mechanisms

16. models/temporal_layers.py
    └── Time series processing

17. models/physics_layers.py
    └── Physics constraints

18. models/task_heads.py
    └── Task-specific outputs

19. training/multi_task_trainer.py
    ├── Training loop
    └── Manages epochs

20. training/loss_functions.py
    └── Calculates losses

21. training/evaluation_metrics.py
    └── Computes metrics

22. training/validation.py
    ├── Validates physics
    └── Checks economics
```

---

### **PHASE 3B: INFERENCE PATH** (If command = infer)
```
14. inference/query_processor.py
    ├── Processes natural language
    └── Maps to task

15. inference/inference_engine.py
    ├── Loads trained model
    └── Runs forward pass

16. tasks/solar_optimization.py (or other task file)
    ├── Task-specific logic
    ├── Calculates recommendations
    └── Economic analysis

17. tasks/clustering.py
    └── Energy community detection

18. tasks/retrofit_targeting.py
    └── Building retrofit priority

19. training/validation.py
    ├── Validates results
    └── Checks constraints

20. inference/kg_updater.py
    └── Writes results to Neo4j

21. utils/visualization.py
    └── Creates visualizations
```

---

### **PHASE 4: TESTING & DEBUGGING** (As needed)
```
test_neo4j_data.py
├── Tests Neo4j connection
└── Verifies data

check_features.py
├── Checks feature dimensions
└── Validates tensor shapes

check_data.py
└── Inspects graph structure

check_graph_structure.py
└── Verifies graph topology

simple_inference.py
└── Simplified testing
```

---

## 📊 **EXECUTION EXAMPLES**

### **Example 1: Fresh Setup + Training**
```bash
# 1. Generate data
python mimic_data_generator.py

# 2. Load to Neo4j
python load_mimic_data_to_neo4j.py
python update_neo4j_buildings.py

# 3. Train model
python main.py train --epochs 200
```

**File execution order:**
1. mimic_data_generator.py
2. load_mimic_data_to_neo4j.py
3. update_neo4j_buildings.py
4. main.py
5. utils/logger.py
6. data/kg_connector.py
7. data/graph_builder.py
8. models/base_gnn.py
9. training/multi_task_trainer.py
10. training/loss_functions.py

### **Example 2: Inference Query**
```bash
python main.py infer "What buildings need solar panels?"
```

**File execution order:**
1. main.py
2. utils/logger.py
3. data/kg_connector.py
4. data/graph_builder.py
5. inference/query_processor.py
6. inference/inference_engine.py
7. models/base_gnn.py
8. tasks/solar_optimization.py
9. training/validation.py
10. utils/visualization.py

---

## 🔑 **KEY DEPENDENCIES**

### **Core Dependencies:**
```
main.py
├── data/kg_connector.py      [Always]
├── data/graph_builder.py     [Always]
├── models/base_gnn.py        [Always]
├── utils/logger.py           [Always]
└── utils/metrics_tracker.py  [Always]
```

### **Training Dependencies:**
```
main.py --train
└── training/multi_task_trainer.py
    ├── training/loss_functions.py
    ├── training/evaluation_metrics.py
    └── training/validation.py
```

### **Inference Dependencies:**
```
main.py --infer
└── inference/inference_engine.py
    ├── inference/query_processor.py
    ├── tasks/*.py (based on query)
    └── inference/kg_updater.py
```

---

## 📈 **FREQUENCY OF USE**

| File | Usage Frequency | When Used |
|------|----------------|-----------|
| main.py | Every run | Entry point |
| data/kg_connector.py | Every run | Neo4j connection |
| data/graph_builder.py | Every run | Graph creation |
| models/base_gnn.py | Every run | Model definition |
| mimic_data_generator.py | Once | Initial setup |
| kg_builder.py | Once | Initial setup |
| training/*.py | During training | Model training |
| inference/*.py | During inference | Query processing |
| tasks/*.py | Task-specific | Based on query |
| test_*.py | Debugging only | Testing |

---

## 🎯 **CRITICAL PATH**

**Minimal files needed for inference:**
1. main.py
2. data/kg_connector.py
3. data/graph_builder.py
4. models/base_gnn.py
5. inference/inference_engine.py
6. One task file (e.g., tasks/solar_optimization.py)

**Total: 6 core files** for basic operation

---

## 📝 **NOTES**

- **kg_builder.py** is NOT used during runtime - only for initial setup
- **main.py** is the orchestrator that calls everything else
- **data/graph_builder.py** is different from kg_builder.py
- Files in **utils/** are helper modules used throughout
- Files in **tasks/** are selected based on the query
- **Training** and **inference** follow different paths after main.py