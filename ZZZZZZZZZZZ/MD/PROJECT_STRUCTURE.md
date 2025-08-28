# Unified Energy GNN System - Project Structure

## Core Files
```
Qiuari_V3/
│
├── main.py                       # Unified main entry point (all modes)
├── config/
│   └── config.yaml              # Unified configuration file
│
## Data Processing
├── data/
│   ├── kg_connector.py          # Neo4j Knowledge Graph connection
│   ├── graph_constructor.py     # Graph structure builder
│   ├── data_loader.py           # Data loading and batching
│   └── feature_processor.py     # Feature engineering
│
## Model Architecture
├── models/
│   ├── base_gnn.py              # Core GNN architecture
│   ├── task_heads.py            # Task-specific output heads
│   ├── attention_layers.py      # Attention mechanisms
│   ├── temporal_layers.py       # Temporal processing
│   ├── physics_layers.py        # Physics constraints
│   ├── network_aware_layers.py  # Network-aware components
│   ├── pooling_layers.py        # Graph pooling
│   │
│   ## Enhancement Modules
│   ├── semi_supervised_layers.py    # Semi-supervised learning
│   ├── enhanced_temporal_layers.py  # Advanced temporal processing
│   ├── uncertainty_quantification.py # Uncertainty estimation
│   ├── explainability_layers.py     # Model explainability
│   ├── dynamic_graph_layers.py      # Dynamic graph construction
│   └── sparse_utils.py               # Sparse operations utilities
│
## Training Pipeline
├── training/
│   ├── discovery_trainer.py     # Discovery mode trainer
│   ├── enhanced_trainer.py      # Enhanced mode trainer
│   ├── network_aware_trainer.py # Network-aware trainer
│   ├── unified_gnn_trainer.py   # Standard trainer
│   ├── loss_functions.py        # Loss function definitions
│   ├── network_aware_loss.py    # Network-specific losses
│   ├── evaluation_metrics.py    # Evaluation metrics
│   ├── active_learning.py       # Active learning strategies
│   └── contrastive_learning.py  # Contrastive learning
│
## Analysis & Evaluation
├── analysis/
│   ├── pattern_analyzer.py      # Pattern discovery
│   ├── intervention_recommender.py # Intervention planning
│   ├── baseline_comparison.py   # Baseline comparisons
│   ├── comprehensive_reporter.py # Report generation
│   └── lv_group_evaluator.py    # LV group evaluation
│
├── evaluation/
│   └── network_metrics.py       # Network effect metrics
│
├── simulation/
│   └── simple_intervention.py   # Intervention simulation
│
├── tasks/
│   └── intervention_selection.py # Intervention selection
│
├── utils/
│   └── output_validation.py     # Physics validation & reporting
│
## Data Storage
├── data/
│   └── processed/               # Processed datasets
│       ├── train_dataset.pt
│       ├── val_dataset.pt
│       └── test_dataset.pt
│
## Outputs
├── checkpoints/                 # Model checkpoints
│   └── best_model.pt
│
├── experiments/                 # Experiment results
│   └── exp_*/
│
├── results/                     # Analysis results
│   ├── analysis/
│   ├── interventions/
│   ├── comparisons/
│   ├── visualizations/
│   └── inference/
│
├── reports/                     # Generated reports
│   └── lv_group_evaluation/
│
└── logs/                        # Training logs
```

## Essential Configuration Sections

### config/config.yaml
- **experiment**: Experiment settings and metadata
- **data**: Data source configuration (KG or files)
- **kg**: Knowledge Graph connection
- **model**: Model architecture parameters
- **training**: Training mode and hyperparameters
- **enhancements**: Optional enhancement features
- **loss**: Loss function weights
- **evaluation**: Evaluation settings
- **reporting**: Report generation config

## Execution Modes

### 1. Standard Training
```bash
python main.py --mode train --epochs 100
```

### 2. Discovery Mode
```bash
python main.py --mode train  # with training.mode: "discovery" in config
```

### 3. Network-Aware Mode
```bash
python main.py --mode network-aware
```

### 4. Enhanced Mode (All Features)
```bash
python main.py --mode enhanced
```

### 5. Active Learning
```bash
python main.py --mode active-learning
```

### 6. Evaluation
```bash
python main.py --mode evaluate
```

### 7. Full Pipeline
```bash
python main.py --mode full
```

## Key Features
- **Unified Entry Point**: Single main.py handles all modes
- **Unified Configuration**: One config.yaml controls everything
- **Modular Architecture**: Clean separation of concerns
- **Multiple Training Modes**: Discovery, Enhanced, Network-aware
- **Enhancement Modules**: Optional features can be enabled/disabled
- **Comprehensive Analysis**: Pattern discovery, intervention planning
- **Physics Validation**: Ensures physical constraints are met

## Data Flow
1. **KG/Files** → Data Loader → Graph Constructor
2. **Graph Data** → GNN Model → Task Heads
3. **Predictions** → Analysis → Interventions
4. **Results** → Reports & Visualizations

## Files NOT Included (Extra/Debug)
- *.pyc files
- __pycache__ directories
- Test scripts (run_simple_test.py, etc.)
- Debug traces (expert_layer_trace.py, etc.)
- Temporary files
- Old/backup versions