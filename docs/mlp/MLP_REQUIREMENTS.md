# The Oracle (MLP) - Hyperparameter Recommendation Requirements

**Document Version**: 1.0
**Last Updated**: 2026-01-11
**Component**: MLP - "The Oracle"
**Purpose**: Initial hyperparameter recommendation based on dataset characteristics

---

## Executive Summary

**Current Implementation Status**: Phase 1 is **NOT STARTED** - meta-dataset infrastructure not implemented.

### What's Implemented ✅

**EA Infrastructure** (provides training data):
- ✅ Complete hyperparameter space exploration
- ✅ Results CSV with all parameters and metrics
- ✅ UID-based tracking

### What's Missing ❌

**Meta-Dataset Generation**:
- ❌ Dataset meta-feature extraction
- ❌ Meta-dataset aggregation across EA runs
- ❌ Feature engineering for dataset characteristics

**Oracle Model**:
- ❌ MLP architecture definition
- ❌ Multi-output regression setup
- ❌ Training pipeline
- ❌ Hyperparameter recommendation interface

**Integration**:
- ❌ OracleRecommender class
- ❌ EA initialization integration
- ❌ LSTM initialization (future)

### Phase Roadmap

- **Phase 1**: Meta-dataset generation ❌ (0% complete)
- **Phase 2**: MLP model training ❌ (0% complete)
- **Phase 3**: Integration with EA/LSTM ❌ (0% complete)

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Meta-Dataset Generation](#phase-1-meta-dataset-generation)
3. [Phase 2: MLP Model Development](#phase-2-mlp-model-development)
4. [Phase 3: Integration](#phase-3-integration)
5. [Requirements Traceability Matrix](#requirements-traceability-matrix)
6. [Document Information](#document-information)

---

## Overview

**The Oracle (MLP)** is a multi-layer perceptron that recommends initial SOM hyperparameters and LSTM configurations based on dataset characteristics, replacing random initialization with informed suggestions.

**Primary Objectives**:
1. **SOM Hyperparameter Recommendation**: Suggest good starting parameters for SOM training
2. **LSTM Configuration Recommendation**: Suggest controller settings (future)
3. **EA Initialization**: Seed EA population with Oracle recommendations

**Key Insight**: Different datasets (size, dimensionality, complexity) benefit from different hyperparameter configurations. The Oracle learns this mapping from EA campaign results.

---

## Phase 1: Meta-Dataset Generation

### 1.1 Meta-Feature Extraction

#### FR-MLP-1.1.1: Dataset Characteristics ❌

**Requirement**: Extract comprehensive meta-features describing dataset properties.

**Status**: NOT IMPLEMENTED

**Meta-Features to Extract**:

**Basic Statistics**:
- `num_samples`: Total number of samples
- `num_dimensions`: Number of features
- `num_categorical`: Number of categorical features
- `num_numerical`: Number of numerical features

**Distribution Properties**:
- `mean_values`: Mean per feature
- `std_values`: Standard deviation per feature
- `skewness`: Distribution skewness per feature
- `kurtosis`: Distribution kurtosis per feature

**Complexity Metrics**:
- `intrinsic_dimensionality`: Estimated intrinsic dimension (e.g., via PCA)
- `correlation_mean`: Mean absolute correlation between features
- `outlier_ratio`: Percentage of outliers (e.g., 3-sigma rule)

**Clustering Characteristics** (optional, computationally expensive):
- `estimated_num_clusters`: K-means elbow method estimate
- `silhouette_score`: Clustering quality metric
- `hopkins_statistic`: Clustering tendency measure

**Extraction Script**:
```python
# app/mlp/extract_meta_features.py

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

def extract_meta_features(data: np.ndarray, categorical_mask: np.ndarray = None) -> dict:
    """
    Extract meta-features from dataset.

    Args:
        data: Normalized dataset (N, D)
        categorical_mask: Boolean mask for categorical features

    Returns:
        Dictionary of meta-features
    """
    N, D = data.shape

    if categorical_mask is None:
        categorical_mask = np.zeros(D, dtype=bool)

    numerical_data = data[:, ~categorical_mask]

    meta_features = {
        # Basic
        'num_samples': N,
        'num_dimensions': D,
        'num_categorical': categorical_mask.sum(),
        'num_numerical': (~categorical_mask).sum(),

        # Distribution (numerical features only)
        'mean_mean': np.mean(numerical_data.mean(axis=0)),
        'mean_std': np.mean(numerical_data.std(axis=0)),
        'mean_skewness': np.mean(skew(numerical_data, axis=0)),
        'mean_kurtosis': np.mean(kurtosis(numerical_data, axis=0)),

        # Complexity
        'intrinsic_dim': estimate_intrinsic_dimension(numerical_data),
        'correlation_mean': np.abs(np.corrcoef(numerical_data.T)).mean(),
        'outlier_ratio': compute_outlier_ratio(numerical_data),

        # Clustering (optional)
        'estimated_clusters': estimate_num_clusters(numerical_data, max_k=10),
    }

    return meta_features

def estimate_intrinsic_dimension(data: np.ndarray, variance_threshold: float = 0.95) -> int:
    """Estimate intrinsic dimensionality via PCA."""
    pca = PCA()
    pca.fit(data)

    cumsum = np.cumsum(pca.explained_variance_ratio_)
    intrinsic_dim = np.argmax(cumsum >= variance_threshold) + 1

    return intrinsic_dim

def compute_outlier_ratio(data: np.ndarray, sigma: float = 3.0) -> float:
    """Compute percentage of outliers using 3-sigma rule."""
    mean = data.mean(axis=0)
    std = data.std(axis=0)

    # Count outliers per feature
    outliers = np.abs(data - mean) > sigma * std
    outlier_ratio = outliers.any(axis=1).mean()  # Samples with any outlier feature

    return outlier_ratio

def estimate_num_clusters(data: np.ndarray, max_k: int = 10) -> int:
    """Estimate number of clusters using elbow method."""
    if len(data) < max_k * 2:
        return min(3, len(data) // 2)

    inertias = []
    for k in range(2, min(max_k + 1, len(data) // 2)):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)

    # Simple elbow detection (derivative)
    diffs = np.diff(inertias)
    elbow_idx = np.argmax(diffs[:-1] - diffs[1:]) + 2

    return elbow_idx
```

**Acceptance Criteria**:
- [ ] Extract 15+ meta-features per dataset
- [ ] Handle both numerical and categorical features
- [ ] Computationally efficient (< 10 seconds per dataset)
- [ ] Normalize features to similar scales

**Verified**: NOT IMPLEMENTED

---

#### FR-MLP-1.1.2: Meta-Dataset Aggregation ❌

**Requirement**: Aggregate meta-features from multiple EA runs into single meta-dataset.

**Status**: NOT IMPLEMENTED

**Aggregation Process**:
```python
# app/mlp/create_meta_dataset.py

def create_meta_dataset(ea_run_dirs: List[str]) -> pd.DataFrame:
    """
    Create meta-dataset from multiple EA runs.

    Args:
        ea_run_dirs: List of EA run directories

    Returns:
        DataFrame with columns:
        - dataset_id: Unique dataset identifier
        - meta_feature_1, meta_feature_2, ...: Dataset meta-features
        - best_param_1, best_param_2, ...: Best hyperparameters for this dataset
        - best_mqe, best_te, ...: Best achieved metrics
    """
    meta_dataset = []

    for ea_dir in ea_run_dirs:
        # 1. Load dataset
        dataset_path = find_dataset_path(ea_dir)  # From preprocessing info
        data = load_dataset(dataset_path)

        # 2. Extract meta-features
        meta_features = extract_meta_features(data)

        # 3. Find best EA individual
        results_csv = pd.read_csv(os.path.join(ea_dir, 'results.csv'))
        best_idx = results_csv['best_mqe'].idxmin()
        best_row = results_csv.iloc[best_idx]

        # 4. Combine into meta-dataset row
        meta_row = {
            'dataset_id': get_dataset_id(dataset_path),
            **meta_features,  # Unpack meta-features
            **extract_best_hyperparameters(best_row),  # Best SOM params
            'best_mqe': best_row['best_mqe'],
            'best_te': best_row['topographic_error'],
            'best_dead_ratio': best_row['dead_neuron_ratio']
        }

        meta_dataset.append(meta_row)

    return pd.DataFrame(meta_dataset)

def extract_best_hyperparameters(row: pd.Series) -> dict:
    """Extract relevant hyperparameters from results row."""
    return {
        'best_map_size_0': row['map_size'][0] if isinstance(row['map_size'], list) else None,
        'best_map_size_1': row['map_size'][1] if isinstance(row['map_size'], list) else None,
        'best_processing_type': row['processing_type'],
        'best_start_lr': row['start_learning_rate'],
        'best_end_lr': row['end_learning_rate'],
        'best_lr_decay_type': row['lr_decay_type'],
        'best_start_radius_ratio': row['start_radius_init_ratio'],
        'best_radius_decay_type': row['radius_decay_type'],
        'best_batch_growth_type': row['batch_growth_type'],
        'best_epoch_multiplier': row['epoch_multiplier'],
        'best_num_batches': row['num_batches']
    }
```

**Acceptance Criteria**:
- [ ] Support multiple EA runs (different datasets)
- [ ] Link meta-features to best hyperparameters
- [ ] Handle missing/incomplete EA runs
- [ ] Save meta-dataset to CSV

**Note**: Requires EA runs on **diverse datasets** (not just Iris)

---

#### FR-MLP-1.1.3: Feature Engineering ❌

**Requirement**: Engineer additional meta-features for better Oracle predictions.

**Status**: NOT IMPLEMENTED

**Derived Features**:
- `samples_per_dim`: `num_samples / num_dimensions` (dataset density)
- `categorical_ratio`: `num_categorical / num_dimensions`
- `complexity_score`: Composite metric combining intrinsic_dim, correlation, etc.
- `recommended_map_area`: Heuristic estimate (e.g., `sqrt(num_samples) * intrinsic_dim`)

**Acceptance Criteria**:
- [ ] Create 5-10 derived features
- [ ] Feature selection (remove low-variance features)
- [ ] Correlation analysis (remove redundant features)

---

### 1.2 Dataset Requirements

#### FR-MLP-1.2.1: Minimum Dataset Diversity ❌

**Requirement**: Ensure sufficient diversity in meta-dataset for generalization.

**Status**: NOT IMPLEMENTED

**Diversity Requirements**:
- **Minimum datasets**: 20-50 different datasets
- **Size range**: 100 samples to 100,000+ samples
- **Dimensionality range**: 2D to 100+D
- **Complexity range**: Simple (2-3 clusters) to complex (10+ clusters)

**Recommended Datasets**:
- **UCI ML Repository**: Iris, Wine, Breast Cancer, Digits, etc.
- **Synthetic datasets**: Generated with varying complexity
- **Real-world datasets**: Domain-specific datasets

**Acceptance Criteria**:
- [ ] At least 20 unique datasets
- [ ] Wide range of num_samples (2+ orders of magnitude)
- [ ] Wide range of num_dimensions (10x range)
- [ ] Diverse clustering structures

**Current Status**: Only Iris dataset tested (insufficient for Oracle training)

---

## Phase 2: MLP Model Development

### 2.1 Model Architecture

#### FR-MLP-2.1.1: MLP Architecture ❌

**Requirement**: Define MLP architecture for hyperparameter recommendation.

**Status**: NOT IMPLEMENTED

**Architecture**:
```python
import torch.nn as nn

class OracleMLP(nn.Module):
    """Multi-layer perceptron for hyperparameter recommendation."""

    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64], output_size: int = 20):
        """
        Initialize Oracle MLP.

        Args:
            input_size: Number of meta-features
            hidden_sizes: Hidden layer dimensions
            output_size: Number of hyperparameters to predict
        """
        super().__init__()

        layers = []
        prev_size = input_size

        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

        # Separate output heads for different parameter types
        self.output_map = {
            'continuous': [0, 1, 2, 3, 4, 5],  # LRs, radius ratio, epoch multiplier, etc.
            'categorical': [6, 7, 8, 9, 10]    # Processing type, decay types, etc.
        }

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: (batch, input_size) meta-features

        Returns:
            (batch, output_size) predicted hyperparameters
        """
        return self.network(x)
```

**Output Mapping**:
```python
# Predicted hyperparameters (indices in output vector)
output_params = {
    0: 'map_size_ratio',       # Ratio of dataset size (continuous)
    1: 'start_learning_rate',  # Continuous
    2: 'end_learning_rate',    # Continuous
    3: 'start_radius_ratio',   # Continuous
    4: 'epoch_multiplier',     # Continuous
    5: 'num_batches',          # Discrete (round to int)
    6: 'processing_type',      # Categorical (argmax over 3 classes)
    7: 'lr_decay_type',        # Categorical (argmax over 5 classes)
    8: 'radius_decay_type',    # Categorical (argmax over 5 classes)
    9: 'batch_growth_type',    # Categorical (argmax over 3 classes)
    10: 'normalize_weights',   # Binary (sigmoid > 0.5)
}
```

**Acceptance Criteria**:
- [ ] Support configurable hidden layers
- [ ] Batch normalization for stability
- [ ] Dropout for regularization
- [ ] Mixed output types (continuous, categorical, binary)

---

#### FR-MLP-2.1.2: Multi-Output Regression ❌

**Requirement**: Implement multi-output regression for simultaneous parameter prediction.

**Status**: NOT IMPLEMENTED

**Loss Function**:
```python
def oracle_loss(predictions: torch.Tensor, targets: torch.Tensor,
                output_map: dict, alpha: dict = None) -> torch.Tensor:
    """
    Combined loss for multi-output regression.

    Args:
        predictions: (batch, num_outputs) predicted hyperparameters
        targets: (batch, num_outputs) true best hyperparameters
        output_map: Dictionary mapping output indices to parameter types
        alpha: Weights for each parameter type (optional)

    Returns:
        Combined loss
    """
    if alpha is None:
        alpha = {'continuous': 1.0, 'categorical': 0.5, 'binary': 0.3}

    # Continuous parameters: MSE loss
    continuous_indices = output_map['continuous']
    continuous_loss = F.mse_loss(predictions[:, continuous_indices],
                                  targets[:, continuous_indices])

    # Categorical parameters: Cross-entropy loss
    categorical_indices = output_map['categorical']
    # Assume targets are class indices for categoricals
    categorical_loss = 0
    for idx in categorical_indices:
        categorical_loss += F.cross_entropy(predictions[:, idx], targets[:, idx].long())

    # Combine
    total_loss = (alpha['continuous'] * continuous_loss +
                  alpha['categorical'] * categorical_loss)

    return total_loss, continuous_loss, categorical_loss
```

**Acceptance Criteria**:
- [ ] MSE loss for continuous parameters
- [ ] Cross-entropy loss for categorical parameters
- [ ] Weighted combination of losses
- [ ] Configurable loss weights

---

### 2.2 Training Pipeline

#### FR-MLP-2.2.1: Training Script ❌

**Requirement**: Implement MLP training pipeline with validation.

**Status**: NOT IMPLEMENTED

**Training Loop**:
```python
# app/mlp/train.py

class OracleTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model = OracleMLP(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            output_size=config['output_size']
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_loader, val_loader):
        """Main training loop."""
        best_val_loss = float('inf')

        for epoch in range(self.config['epochs']):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate(val_loader)

            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, 'best')

    def _train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            meta_features, hyperparameters = batch
            meta_features = meta_features.to(self.device)
            hyperparameters = hyperparameters.to(self.device)

            # Forward pass
            predictions = self.model(meta_features)

            # Loss
            loss, _, _ = oracle_loss(predictions, hyperparameters, self.model.output_map)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)
```

**Acceptance Criteria**:
- [ ] Data loading from meta-dataset CSV
- [ ] Training/validation split
- [ ] Multi-output loss calculation
- [ ] Checkpointing best model
- [ ] Metric logging

---

#### FR-MLP-2.2.2: Model Evaluation ❌

**Requirement**: Evaluate Oracle recommendations on validation set.

**Status**: NOT IMPLEMENTED

**Evaluation Metrics**:

**Continuous Parameters**:
- MAE, RMSE for each parameter
- R² score per parameter

**Categorical Parameters**:
- Accuracy per parameter
- Top-k accuracy (e.g., top-2 for decay types)

**Overall Quality**:
- Correlation between predicted and actual final MQE
- Recommendation quality: % of recommendations in top 25% of search space

**Acceptance Criteria**:
- [ ] Evaluate on held-out validation set
- [ ] Per-parameter metrics
- [ ] Overall recommendation quality metrics
- [ ] Comparison to random baseline

---

## Phase 3: Integration

### 3.1 Recommendation Interface

#### FR-MLP-3.1.1: OracleRecommender Class ❌

**Requirement**: Encapsulate trained Oracle in simple recommendation interface.

**Status**: NOT IMPLEMENTED

**Class Interface**:
```python
# app/mlp/recommender.py

class OracleRecommender:
    """Oracle-based hyperparameter recommendation."""

    def __init__(self, model_checkpoint: str, config: dict = None):
        """
        Initialize Oracle recommender.

        Args:
            model_checkpoint: Path to trained MLP checkpoint
            config: Optional configuration
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_checkpoint, config)
        self.model.eval()

    def recommend(self, data: np.ndarray, categorical_mask: np.ndarray = None) -> dict:
        """
        Recommend SOM hyperparameters for dataset.

        Args:
            data: Dataset (N, D) - normalized
            categorical_mask: Boolean mask for categorical features

        Returns:
            Dictionary of recommended hyperparameters
        """
        # 1. Extract meta-features
        meta_features = extract_meta_features(data, categorical_mask)

        # 2. Convert to tensor
        feature_vector = self._meta_features_to_tensor(meta_features)
        feature_vector = feature_vector.unsqueeze(0).to(self.device)  # (1, num_features)

        # 3. Predict
        with torch.no_grad():
            predictions = self.model(feature_vector)

        # 4. Decode predictions to hyperparameters
        recommendations = self._decode_predictions(predictions.squeeze(0))

        return recommendations

    def recommend_batch(self, datasets: List[np.ndarray]) -> List[dict]:
        """Recommend for multiple datasets."""
        # ... batch recommendation

    def _decode_predictions(self, predictions: torch.Tensor) -> dict:
        """
        Decode model output to hyperparameter dictionary.

        Args:
            predictions: (num_outputs,) tensor

        Returns:
            Dictionary of hyperparameters compatible with SOM
        """
        predictions_np = predictions.cpu().numpy()

        recommendations = {
            # Continuous parameters
            'start_learning_rate': float(predictions_np[1]),
            'end_learning_rate': float(predictions_np[2]),
            'start_radius_init_ratio': float(predictions_np[3]),
            'epoch_multiplier': float(predictions_np[4]),

            # Discrete parameters
            'num_batches': int(round(predictions_np[5])),

            # Map size (derived from ratio)
            'map_size': self._compute_map_size(predictions_np[0], ...),

            # Categorical parameters (argmax)
            'processing_type': self._decode_categorical(predictions_np[6], ['stochastic', 'deterministic', 'hybrid']),
            'lr_decay_type': self._decode_categorical(predictions_np[7], ['static', 'linear-drop', 'exp-drop', 'log-drop', 'step-down']),
            'radius_decay_type': self._decode_categorical(predictions_np[8], ['linear-drop', 'exp-drop', 'log-drop', 'step-down']),
            'batch_growth_type': self._decode_categorical(predictions_np[9], ['linear-growth', 'exp-growth', 'log-growth']),

            # Binary parameters
            'normalize_weights_flag': bool(predictions_np[10] > 0.5)
        }

        return recommendations

    def _compute_map_size(self, ratio: float, num_samples: int, intrinsic_dim: int) -> List[int]:
        """Compute map size from predicted ratio."""
        # Heuristic: map_area = ratio * sqrt(num_samples) * intrinsic_dim
        map_area = int(ratio * np.sqrt(num_samples) * intrinsic_dim)

        # Convert to square-ish map
        side_length = int(np.sqrt(map_area))
        return [side_length, side_length]

    def _decode_categorical(self, logits: np.ndarray, classes: List[str]) -> str:
        """Decode categorical prediction."""
        # For single-value categorical, just use the value as index
        # In real implementation, use proper softmax classification
        idx = int(round(logits * (len(classes) - 1)))
        return classes[idx]

    def _load_model(self, checkpoint_path: str, config: dict):
        """Load trained MLP."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if config is None:
            config = checkpoint['config']

        model = OracleMLP(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            output_size=config['output_size']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        self.meta_feature_scaler = checkpoint['scaler']

        return model

    def _meta_features_to_tensor(self, meta_features: dict) -> torch.Tensor:
        """Convert meta-feature dict to normalized tensor."""
        # Extract features in correct order
        feature_vector = np.array([meta_features[key] for key in sorted(meta_features.keys())])

        # Normalize
        feature_vector_norm = self.meta_feature_scaler.transform(feature_vector.reshape(1, -1))

        return torch.FloatTensor(feature_vector_norm.squeeze(0))
```

**Acceptance Criteria**:
- [ ] Simple `recommend(data)` interface
- [ ] Automatic meta-feature extraction
- [ ] Prediction decoding to hyperparameter dict
- [ ] Map size computation from heuristics
- [ ] SOM-compatible output format

---

### 3.2 EA Integration

#### FR-MLP-3.2.1: EA Population Initialization ❌

**Requirement**: Seed EA initial population with Oracle recommendations.

**Status**: NOT IMPLEMENTED

**Integration Point**: [app/ea/ea.py](../app/ea/ea.py) `run_evolution()` function

**Modified Initialization**:
```python
# In run_evolution()

def run_evolution(ea_config: dict, data: np.ndarray, ignore_mask: np.ndarray,
                  oracle: OracleRecommender = None) -> None:
    """
    Main EA loop with optional Oracle initialization.

    Args:
        ea_config: EA configuration
        data: Training data
        ignore_mask: Feature mask
        oracle: Optional Oracle recommender
    """
    population_size = ea_config["EA_SETTINGS"]["population_size"]
    search_space = ea_config["SEARCH_SPACE"]

    # Initialize population
    if oracle is not None:
        # Use Oracle for initial recommendations
        oracle_recommendations = oracle.recommend(data, ignore_mask)

        # Create multiple variations of Oracle recommendation
        population = []

        # Add Oracle recommendation directly
        population.append(oracle_recommendations)

        # Add variations (slight mutations)
        for i in range(min(population_size - 1, 5)):
            variation = mutate_recommendation(oracle_recommendations, search_space, mutation_rate=0.1)
            population.append(variation)

        # Fill remaining with random
        while len(population) < population_size:
            population.append(random_config(search_space))

        print(f"INFO: Population initialized with Oracle recommendation + {len(population)-1} variations")
    else:
        # Random initialization (existing behavior)
        population = [random_config(search_space) for _ in range(population_size)]

    # ... rest of EA loop unchanged
```

**Helper Function**:
```python
def mutate_recommendation(config: dict, search_space: dict, mutation_rate: float = 0.1) -> dict:
    """
    Create variation of Oracle recommendation.

    Args:
        config: Oracle recommendation
        search_space: Valid parameter ranges
        mutation_rate: Probability of mutating each parameter

    Returns:
        Mutated configuration
    """
    mutated = config.copy()

    for key in search_space:
        if random.random() < mutation_rate:
            # Mutate this parameter
            if isinstance(search_space[key], list):
                mutated[key] = random.choice(search_space[key])
            # Keep continuous parameters as-is (Oracle already optimized)

    return mutated
```

**Acceptance Criteria**:
- [ ] Oracle parameter added to run_evolution()
- [ ] Oracle recommendation seeded as first individual
- [ ] Variations created via mild mutation
- [ ] Remaining population random
- [ ] Backward compatibility (oracle=None)

---

#### FR-MLP-3.2.2: Recommendation Logging ❌

**Requirement**: Log Oracle recommendations for analysis.

**Status**: NOT IMPLEMENTED

**Logging**:
```python
# Save Oracle recommendation to file
if oracle is not None:
    oracle_log_path = os.path.join(WORKING_DIR, "oracle_recommendation.json")

    with open(oracle_log_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'meta_features': meta_features,
            'recommendation': oracle_recommendations
        }, f, indent=2)

    # Also log to pareto front log
    log_message("ORACLE", f"Recommended: {oracle_recommendations}", WORKING_DIR)
```

**Acceptance Criteria**:
- [ ] Save recommendation to JSON
- [ ] Include meta-features for traceability
- [ ] Log to EA execution log

---

## Requirements Traceability Matrix

| Requirement ID | Description | Status | Implementation | Verified |
|---------------|-------------|--------|----------------|----------|
| **FR-MLP-1.1.1** | Meta-feature extraction | ❌ | Not implemented | - |
| **FR-MLP-1.1.2** | Meta-dataset aggregation | ❌ | Not implemented | - |
| **FR-MLP-1.1.3** | Feature engineering | ❌ | Not implemented | - |
| **FR-MLP-1.2.1** | Dataset diversity | ❌ | Not implemented | - |
| **FR-MLP-2.1.1** | MLP architecture | ❌ | Not implemented | - |
| **FR-MLP-2.1.2** | Multi-output regression | ❌ | Not implemented | - |
| **FR-MLP-2.2.1** | Training script | ❌ | Not implemented | - |
| **FR-MLP-2.2.2** | Model evaluation | ❌ | Not implemented | - |
| **FR-MLP-3.1.1** | OracleRecommender class | ❌ | Not implemented | - |
| **FR-MLP-3.2.1** | EA initialization | ❌ | Not implemented | - |
| **FR-MLP-3.2.2** | Recommendation logging | ❌ | Not implemented | - |

**Summary**: 0/11 requirements implemented (0%)

---

## Document Information

### Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-11 | Initial MLP requirements document | Claude Sonnet 4.5 |

### Implementation Summary

**Phase 1 (Meta-Dataset)**: 0% complete
- ❌ Meta-feature extraction not implemented
- ❌ Meta-dataset aggregation not implemented
- ⚠️ **Blocker**: Requires EA runs on diverse datasets (currently only Iris)

**Phase 2 (MLP Model)**: 0% complete
- ❌ MLP architecture not defined
- ❌ Training pipeline not implemented

**Phase 3 (Integration)**: 0% complete
- ❌ OracleRecommender not implemented
- ❌ EA integration not complete

**Critical Dependency**: Requires **20-50 diverse datasets** with EA campaigns

**Next Immediate Steps**:
1. **Collect diverse datasets**: UCI ML Repository, synthetic datasets
2. **Run EA campaigns**: One campaign per dataset (save best configs)
3. **Extract meta-features**: Implement extraction script
4. **Create meta-dataset**: Aggregate all EA results
5. **Define MLP architecture**: Simple 2-3 layer network
6. **Train Oracle**: Supervised learning on meta-dataset
7. **Implement OracleRecommender**: Wrap trained model
8. **Test integration**: Evaluate recommendations on new datasets

---

**Recommendation**: The Oracle is **lowest priority** among the three neural networks. Focus on:
1. **CNN (The Eye)** first - immediate value for EA Phase 2
2. **LSTM (The Brain)** second - enables dynamic control
3. **MLP (The Oracle)** third - nice-to-have for initialization

The Oracle provides marginal benefit compared to random initialization when search space is well-designed. Only implement after CNN and LSTM are working.

---

**End of MLP Requirements Specification**
