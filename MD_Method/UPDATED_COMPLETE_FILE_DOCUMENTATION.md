# Complete File Documentation - Energy GNN System (Updated)
## Last Updated: 2025-08-18

## Project Structure Overview

```
Qiuari/
├── main.py                           # Primary entry point for training and inference
├── main_with_kg_builders.py          # Alternative entry with KG builder integration
├── simple_inference.py               # Simplified inference pipeline
├── mimic_data_generator.py           # Generate synthetic data (buildings, solar, battery)
├── kg_builder_1.py                   # Knowledge graph builder v1
├── kg_builder_2.py                   # Knowledge graph builder v2 (enhanced)
├── check_features.py                 # Feature validation utility
├── check_data.py                     # Data integrity checker
├── check_graph_structure.py          # Graph structure validator
├── test_neo4j_data.py               # Neo4j data testing
├── update_neo4j_buildings.py        # Neo4j building data updater
│
├── config/
│   ├── config.yaml                  # Main configuration
│   └── tasks_config.yaml            # Task-specific configurations
│
├── data/
│   ├── data_loader.py               # Data loading and batching
│   ├── feature_engineering.py       # Feature extraction and engineering
│   ├── graph_builder.py             # Graph construction from raw data
│   ├── kg_connector.py              # Knowledge graph connection interface
│   ├── kg_connector_real.py         # Real Neo4j KG connector
│   ├── kg_connector_stub.py         # Stub KG connector for testing
│   ├── kg_extractor.py              # Extract data from knowledge graph
│   └── preprocessor.py              # Data preprocessing pipeline
│
├── models/
│   ├── base_gnn.py                  # Base GNN architecture
│   ├── task_heads.py                # Multi-task prediction heads
│   ├── attention_layers.py          # Attention mechanisms for GNN
│   ├── temporal_layers.py           # Temporal processing layers
│   └── physics_layers.py            # Physics-informed neural layers
│
├── training/
│   ├── multi_task_trainer.py        # Multi-task training orchestrator
│   ├── loss_functions.py            # Custom loss functions
│   ├── evaluation_metrics.py        # Evaluation metrics
│   └── validation.py                # Validation utilities
│
├── inference/
│   ├── inference_engine.py          # Main inference engine
│   ├── query_processor.py           # Process user queries
│   └── kg_updater.py                # Update KG with predictions
│
├── tasks/
│   ├── solar_optimization.py        # Solar panel placement optimization
│   ├── clustering.py                # Energy profile clustering
│   ├── retrofit_targeting.py        # Building retrofit recommendations
│   └── Additional Task Implementations.py  # Extended task implementations
│
├── utils/
│   ├── logger.py                    # Logging configuration
│   ├── metrics_tracker.py           # Track training/inference metrics
│   └── visualization.py             # Visualization utilities
│
├── grid_analysis_project/
│   ├── main.py                      # Grid analysis entry point
│   ├── grid_analysis.py             # Core grid analysis logic
│   ├── db_config.py                 # Database configuration
│   ├── sql_executor.py              # SQL execution engine
│   ├── sql_file_reader.py           # SQL file parser
│   └── sql_scripts/                 # SQL analysis scripts
│
├── mimic_data/                      # Generated synthetic data
│   ├── buildings.csv                # Building characteristics
│   ├── lv_networks.csv              # LV network topology
│   ├── mv_transformers.csv          # MV transformer data
│   ├── energy_profiles.parquet      # Time-series energy profiles
│   └── knowledge_graph.html         # KG visualization
│
├── checkpoints/                     # Model checkpoints
├── experiments/                     # Experiment configurations
├── logs/                           # Training/inference logs
├── processed_data/                 # Processed graph data
└── runs/                          # TensorBoard logs
```

## Core Components Documentation

### 1. Main Entry Points

#### `main.py`
- **Purpose**: Primary orchestrator for training and inference
- **Key Functions**:
  - `main()`: Parse arguments, initialize components, run pipeline
  - Supports both training and inference modes
  - Integrates with KG through `kg_connector`

#### `main_with_kg_builders.py`
- **Purpose**: Alternative entry point with KG builder selection
- **Features**:
  - Choose between `kg_builder_1.py` or `kg_builder_2.py`
  - Direct KG construction before training

#### `simple_inference.py`
- **Purpose**: Simplified inference for quick predictions
- **Use Case**: Production deployment, API endpoints

### 2. Data Generation & Processing

#### `mimic_data_generator.py`
- **Purpose**: Generate realistic synthetic energy grid data
- **Key Functions**:
  - `create_grid_topology()`: Generate grid infrastructure
  - `create_buildings()`: Create diverse building types
  - `add_shared_wall_data()`: Add building adjacency
  - `create_energy_profiles()`: Generate 15-min energy profiles
  - `create_solar_profiles()`: Generate solar generation patterns
  - `create_battery_profiles()`: Simulate battery charge/discharge
- **Output**: Complete synthetic dataset in `mimic_data/`

#### `kg_builder_1.py` & `kg_builder_2.py`
- **Purpose**: Build knowledge graph from data
- **kg_builder_1**: Basic KG construction
  - Node types: Buildings, Transformers, Networks
  - Relationships: CONNECTED_TO, PART_OF
- **kg_builder_2**: Enhanced version
  - Additional relationships: ADJACENT_TO, SHARES_TRANSFORMER
  - Richer node properties
  - Better spatial indexing

### 3. Data Pipeline (`data/`)

#### `data_loader.py`
- **Classes**:
  - `EnergyDataLoader`: Load and batch graph data
  - `TemporalDataLoader`: Handle time-series data
- **Features**:
  - PyTorch Geometric data handling
  - Mini-batch generation
  - Train/val/test splitting

#### `feature_engineering.py`
- **Purpose**: Extract and engineer features
- **Key Features**:
  - Building characteristics (area, height, age)
  - Energy profiles (peak, average, variance)
  - Solar/battery capacity features
  - Temporal features (hour, day, season)

#### `graph_builder.py`
- **Purpose**: Construct graph from raw data
- **Functions**:
  - `build_graph()`: Create PyG Data object
  - `create_edge_index()`: Build adjacency matrix
  - `add_edge_features()`: Add edge attributes

#### `kg_extractor.py`
- **Purpose**: Extract data from Neo4j KG
- **Key Methods**:
  - `extract_buildings()`: Get building nodes
  - `extract_networks()`: Get network topology
  - `extract_relationships()`: Get connections

### 4. Model Architecture (`models/`)

#### `base_gnn.py`
- **Class**: `EnergyGNN`
- **Architecture**:
  - Multiple GCN/GAT layers
  - Skip connections
  - Dropout regularization
- **Forward Pass**:
  - Node embedding
  - Message passing
  - Feature aggregation

#### `task_heads.py`
- **Purpose**: Task-specific prediction heads
- **Heads**:
  - `DemandHead`: Energy demand prediction
  - `SolarHead`: Solar potential estimation
  - `ClusterHead`: Load profile clustering
  - `RetrofitHead`: Retrofit recommendations

#### `attention_layers.py`
- **Purpose**: Attention mechanisms
- **Classes**:
  - `SpatialAttention`: Node-level attention
  - `TemporalAttention`: Time-series attention
  - `CrossAttention`: Multi-modal attention

#### `physics_layers.py`
- **Purpose**: Physics-informed layers
- **Features**:
  - Power flow constraints
  - Energy balance equations
  - Voltage regulation

### 5. Training System (`training/`)

#### `multi_task_trainer.py`
- **Class**: `MultiTaskTrainer`
- **Features**:
  - Multi-task learning orchestration
  - Dynamic task weighting
  - Gradient balancing
  - Early stopping
  - Checkpoint management

#### `loss_functions.py`
- **Custom Losses**:
  - `PhysicsInformedLoss`: Incorporate physical constraints
  - `TemporalConsistencyLoss`: Ensure temporal smoothness
  - `ComplementarityLoss`: Maximize load complementarity

#### `evaluation_metrics.py`
- **Metrics**:
  - MAE, RMSE, MAPE for regression
  - Accuracy, F1 for classification
  - Silhouette score for clustering
  - Custom energy metrics

### 6. Inference System (`inference/`)

#### `inference_engine.py`
- **Class**: `InferenceEngine`
- **Methods**:
  - `predict()`: Single prediction
  - `batch_predict()`: Batch processing
  - `stream_predict()`: Real-time streaming

#### `query_processor.py`
- **Purpose**: Process natural language queries
- **Features**:
  - Query parsing
  - Intent recognition
  - Result formatting

### 7. Task Implementations (`tasks/`)

#### `solar_optimization.py`
- **Class**: `SolarOptimization`
- **Functions**:
  - Identify optimal solar locations
  - Calculate ROI and payback
  - Consider grid constraints
  - Generate recommendations

#### `clustering.py`
- **Purpose**: Energy profile clustering
- **Methods**:
  - K-means clustering
  - Hierarchical clustering
  - Profile similarity metrics

#### `retrofit_targeting.py`
- **Purpose**: Building retrofit analysis
- **Features**:
  - Energy efficiency scoring
  - Retrofit impact prediction
  - Cost-benefit analysis

### 8. Grid Analysis (`grid_analysis_project/`)

#### `grid_analysis.py`
- **Purpose**: Analyze grid topology and constraints
- **Features**:
  - Load flow analysis
  - Capacity assessment
  - Bottleneck identification

#### SQL Scripts
- **STEP 1.sql**: Initial data extraction
- **STEPS 2-3.sql**: Data transformation
- **STEPS 4-8.sql**: Analysis queries
- **MV-LV-Based Analysis.sql**: Network-level analysis

### 9. Utilities (`utils/`)

#### `logger.py`
- **Purpose**: Centralized logging
- **Features**:
  - Multi-level logging
  - File and console output
  - Structured log format

#### `metrics_tracker.py`
- **Purpose**: Track metrics during training/inference
- **Features**:
  - TensorBoard integration
  - Custom metric logging
  - Performance monitoring

#### `visualization.py`
- **Purpose**: Visualization utilities
- **Functions**:
  - `plot_graph()`: Visualize graph structure
  - `plot_predictions()`: Show prediction results
  - `plot_energy_profiles()`: Display time-series

## Data Flow

```
1. Raw Data → mimic_data_generator.py → mimic_data/
2. mimic_data/ → kg_builder_*.py → Neo4j KG
3. Neo4j KG → kg_extractor.py → PyG Data
4. PyG Data → data_loader.py → Batches
5. Batches → EnergyGNN → Predictions
6. Predictions → task_heads.py → Task Results
7. Task Results → inference_engine.py → Output
```

## Key Updates Since Last Documentation

### New Files Added:
1. **`mimic_data_generator.py`**: Complete synthetic data generation including:
   - Battery storage profiles
   - Solar generation patterns
   - Building adjacency data
   - 15-minute resolution energy profiles

2. **`check_features.py`**: Feature validation and debugging

3. **`test_neo4j_data.py`**: Neo4j connection and data testing

4. **`update_neo4j_buildings.py`**: Update Neo4j with new building properties

5. **Grid Analysis Project**: Complete SQL-based grid analysis pipeline

### Removed Files:
- `load_mimic_data_to_neo4j.py` (functionality integrated into kg_builders)

### Major Changes:
1. **Enhanced Data Generation**:
   - Added battery and solar mimicked data
   - Improved building diversity
   - Added shared wall relationships

2. **Multi-entry Points**:
   - `main.py` for standard operation
   - `main_with_kg_builders.py` for KG-first approach
   - `simple_inference.py` for production

3. **Improved Task System**:
   - Solar optimization with ROI analysis
   - Clustering with complementarity metrics
   - Retrofit targeting with cost-benefit

## Configuration Files

### `config/config.yaml`
```yaml
model:
  hidden_dim: 128
  num_layers: 3
  dropout: 0.2
  
training:
  epochs: 200
  learning_rate: 0.001
  batch_size: 32
  
tasks:
  - demand_prediction
  - solar_optimization
  - clustering
```

### `PROJECT_STRUCTURE.ini`
- Project metadata
- Directory structure
- File descriptions

## Usage Examples

### Training:
```bash
python main.py --mode train --config config/config.yaml
```

### Inference:
```bash
python simple_inference.py --model checkpoints/best_model.pth
```

### Data Generation:
```bash
python mimic_data_generator.py
```

### KG Building:
```bash
python kg_builder_2.py --data mimic_data/
```

## Dependencies
- PyTorch & PyTorch Geometric
- Neo4j Python Driver
- Pandas, NumPy, NetworkX
- Scikit-learn
- TensorBoard