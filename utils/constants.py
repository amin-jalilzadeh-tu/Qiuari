"""
Constants and configuration values for Energy GNN System
All magic numbers should be defined here with documentation
"""

# ==================== MODEL ARCHITECTURE ====================
# Neural Network Parameters
DROPOUT_RATE = 0.1  # Standard dropout for regularization, prevents overfitting
DEFAULT_HIDDEN_DIM = 128  # Hidden dimension for GNN layers, balance between capacity and efficiency
NUM_GNN_LAYERS = 4  # Number of message passing layers, captures 4-hop neighborhoods
NUM_ATTENTION_HEADS = 8  # Multi-head attention for GAT layers
NEGATIVE_SLOPE = 0.2  # LeakyReLU negative slope for non-linearity

# Clustering Parameters
MIN_CLUSTER_SIZE = 3  # Minimum buildings for valid energy community (regulatory requirement)
MAX_CLUSTER_SIZE = 20  # Maximum buildings per community (grid capacity limit)
DEFAULT_NUM_CLUSTERS = 20  # Default number of clusters to discover

# ==================== PHYSICS & ENERGY ====================
# Energy Efficiency
BATTERY_EFFICIENCY = 0.95  # Lithium-ion round-trip efficiency (industry standard)
P2P_SHARING_EFFICIENCY = 0.98  # Local energy sharing efficiency (minimal losses)
BASE_GRID_EFFICIENCY = 0.98  # Base efficiency for grid transmission
ENERGY_LOSS_PER_METER = 0.0001  # Energy loss per meter distance (0.01% per 100m)

# Battery Specifications
BATTERY_MAX_SOC = 100.0  # Maximum state of charge (kWh)
BATTERY_MAX_CHARGE_RATE = 10.0  # Maximum charging rate (kW)
BATTERY_MAX_DISCHARGE_RATE = 10.0  # Maximum discharging rate (kW)

# Solar Generation
SOLAR_PEAK_EFFICIENCY = 0.18  # Solar panel efficiency (18% is typical)
SOLAR_CAPACITY_FACTOR = 0.7  # Typical capacity factor during peak hours
SOLAR_DAYLIGHT_HOURS = 8  # Average productive daylight hours

# ==================== SPATIAL CONSTRAINTS ====================
# Distance Limits
MAX_COMMUNITY_RADIUS_M = 500  # Maximum radius for energy community (meters)
MAX_SHARING_DISTANCE_M = 1000  # Maximum distance for P2P energy sharing
MIN_BUILDING_DISTANCE_M = 5  # Minimum distance between buildings

# Distance Penalties
DISTANCE_PENALTY_FACTOR = 0.001  # Loss penalty per meter
LONG_DISTANCE_PENALTY = 1000.0  # Penalty for exceeding max distance
SAME_CABLE_GROUP_BONUS = 2.0  # Multiplier for same cable group sharing
CROSS_CABLE_PENALTY = 0.5  # Penalty for different cable groups

# ==================== TRAINING PARAMETERS ====================
# Learning Rates
DEFAULT_LEARNING_RATE = 0.001  # AdamW optimizer learning rate
MIN_LEARNING_RATE = 1e-6  # Minimum learning rate for scheduling
WEIGHT_DECAY = 1e-5  # L2 regularization strength

# Training Settings
DEFAULT_BATCH_SIZE = 32  # Batch size for training
DEFAULT_NUM_EPOCHS = 100  # Default number of training epochs
EARLY_STOPPING_PATIENCE = 20  # Epochs without improvement before stopping
GRADIENT_CLIP_VALUE = 1.0  # Gradient clipping to prevent exploding gradients
VALIDATION_FREQUENCY = 1  # Validate every N epochs

# Loss Weights
COMPLEMENTARITY_WEIGHT = 1.0  # Weight for energy complementarity loss
BALANCE_WEIGHT = 1.0  # Weight for energy balance loss
SPATIAL_WEIGHT = 0.5  # Weight for spatial compactness loss
CLUSTERING_WEIGHT = 0.3  # Weight for clustering quality loss
PEAK_WEIGHT = 0.3  # Weight for peak reduction loss
SUFFICIENCY_WEIGHT = 0.3  # Weight for self-sufficiency loss

# ==================== THRESHOLDS & TARGETS ====================
# Confidence Thresholds
CONFIDENCE_THRESHOLD = 0.85  # Minimum confidence for pseudo-labels
INITIAL_CONFIDENCE = 0.9  # Initial confidence threshold (starts high)
FINAL_CONFIDENCE = 0.7  # Final confidence threshold (relaxes over time)

# Performance Targets
TARGET_SELF_SUFFICIENCY = 0.65  # Target 65% energy self-sufficiency
TARGET_PEAK_REDUCTION = 0.25  # Target 25% peak demand reduction
TARGET_ENERGY_REDUCTION = 0.25  # Target 25% energy consumption reduction
MAX_ACCEPTABLE_PAYBACK_YEARS = 15  # Maximum payback period for interventions

# Validation Thresholds
ENERGY_BALANCE_TOLERANCE = 0.05  # 5% tolerance for energy balance
MAX_PHYSICS_VIOLATIONS = 0.05  # Maximum 5% physics violations allowed
MIN_ACCURACY = 0.8  # Minimum acceptable accuracy

# ==================== DATA PROCESSING ====================
# Feature Dimensions (calculated dynamically, these are defaults)
DEFAULT_BUILDING_FEATURES = 17  # Default number of building features
DEFAULT_CABLE_FEATURES = 12  # Default cable group features
DEFAULT_TRANSFORMER_FEATURES = 8  # Default transformer features
DEFAULT_EDGE_FEATURES = 3  # [distance, same_cable_group, connection_type]

# Temporal Settings
TEMPORAL_WINDOW_HOURS = 24  # Hours to consider for temporal patterns
TEMPORAL_RESOLUTION_MINUTES = 15  # Time series resolution (15-minute intervals)
FORECAST_HORIZON_HOURS = 24  # Prediction horizon

# ==================== SEMI-SUPERVISED LEARNING ====================
# Pseudo-labeling
PROPAGATION_ITERATIONS = 10  # Label propagation iterations
PROPAGATION_ALPHA = 0.85  # Label propagation smoothing factor
PSEUDO_LABEL_WEIGHT = 0.5  # Weight for pseudo-labeled samples

# Active Learning
ACTIVE_LEARNING_BUDGET = 10  # Samples to label per round
UNCERTAINTY_WEIGHT = 0.5  # Weight for uncertainty sampling
DIVERSITY_WEIGHT = 0.3  # Weight for diversity sampling
REPRESENTATIVENESS_WEIGHT = 0.2  # Weight for representativeness

# ==================== EVALUATION METRICS ====================
# Statistical Significance
MIN_SAMPLES_FOR_SIGNIFICANCE = 30  # Minimum samples for t-test
SIGNIFICANCE_LEVEL = 0.05  # P-value threshold

# Baseline Comparison
RANDOM_BASELINE_RUNS = 10  # Number of random baseline runs
KMEANS_N_INIT = 10  # Number of k-means initializations

# ==================== SYSTEM CONFIGURATION ====================
# Computational Limits
MAX_INFERENCE_TIME_S = 1.0  # Maximum inference time per batch
MAX_MEMORY_GB = 8.0  # Maximum GPU memory usage
NUM_WORKERS = 0  # DataLoader workers (0 for Windows)

# Logging
LOG_FREQUENCY = 10  # Log every N batches
TENSORBOARD_UPDATE_FREQ = 100  # Update TensorBoard every N steps
CHECKPOINT_INTERVAL = 10  # Save checkpoint every N epochs

# ==================== NEO4J CONFIGURATION ====================
# Query Limits
NEO4J_BATCH_SIZE = 1000  # Batch size for Neo4j queries
NEO4J_TIMEOUT_S = 30  # Query timeout in seconds
NEO4J_MAX_RETRIES = 3  # Maximum query retries

# ==================== INTERVENTION PLANNING ====================
# Intervention Sizes
SOLAR_SIZES_KW = [10, 20, 30, 50, 100]  # Available solar installation sizes
BATTERY_SIZES_KWH = [10, 25, 50, 100]  # Available battery sizes
RETROFIT_REDUCTIONS_PCT = [10, 20, 30]  # Retrofit efficiency improvements
HEAT_PUMP_SIZES_KW = [5, 10, 15]  # Heat pump capacities

# Planning Horizon
PLANNING_HORIZON_YEARS = 10  # Strategic planning horizon
DISCOUNT_RATE = 0.05  # Financial discount rate
MAX_INTERVENTIONS_PER_YEAR = 50  # Maximum interventions per year
MIN_ROI = 0.1  # Minimum return on investment

# ==================== DUTCH BUILDING SPECIFICS ====================
# Building Age Categories (Dutch context)
AGE_CATEGORIES = {
    'pre_1945': 0,
    '1945_1975': 1,
    '1975_1990': 2, 
    '1990_2005': 3,
    'post_2005': 4
}

# Energy Labels (Dutch system)
ENERGY_LABELS = {
    'A': 7, 'B': 6, 'C': 5, 'D': 4,
    'E': 3, 'F': 2, 'G': 1, 'unknown': 0
}

# Building Types
BUILDING_TYPES = {
    'residential': 0,
    'office': 1,
    'retail': 2,
    'industrial': 3,
    'mixed': 4,
    'other': 5
}

# ==================== VALIDATION CONSTANTS ====================
# Dimension Validation
MAX_INPUT_DIM = 200  # Maximum expected input dimension
MIN_INPUT_DIM = 5  # Minimum expected input dimension
MAX_SEQUENCE_LENGTH = 96  # Maximum temporal sequence (24h at 15min)

# Value Ranges
MAX_BUILDING_HEIGHT_M = 200  # Maximum building height
MAX_BUILDING_AREA_M2 = 10000  # Maximum building area
MAX_CONSUMPTION_KWH_DAY = 1000  # Maximum daily consumption
MAX_GENERATION_KW = 500  # Maximum generation capacity

# ==================== ERROR MESSAGES ====================
ERR_NO_NEO4J_CONNECTION = "Failed to connect to Neo4j. Check credentials and server status."
ERR_NO_BUILDINGS = "No buildings found in Neo4j database."
ERR_DIMENSION_MISMATCH = "Feature dimension mismatch: expected {expected}, got {actual}"
ERR_INVALID_CLUSTER_SIZE = f"Cluster size must be between {MIN_CLUSTER_SIZE} and {MAX_CLUSTER_SIZE}"
ERR_NO_TEMPORAL_DATA = "No temporal data available for buildings"
ERR_PHYSICS_VIOLATION = "Physics constraints violated: {violation_type}"

# ==================== WARNING MESSAGES ====================
WARN_FEW_SAMPLES = "Warning: Only {n} samples found, results may be unreliable"
WARN_HIGH_LOSS = "Warning: Loss is not decreasing, check learning rate"
WARN_MEMORY_USAGE = "Warning: High memory usage detected"
WARN_SLOW_TRAINING = "Warning: Training is slower than expected"