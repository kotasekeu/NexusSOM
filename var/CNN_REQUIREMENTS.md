# The Eye (CNN) - Visual Quality Assessment Requirements

**Document Version**: 1.0
**Last Updated**: 2026-01-11
**Component**: CNN - "The Eye"
**Purpose**: Visual quality assessment of SOM maps through deep learning

---

## Executive Summary

**Current Implementation Status**: Phase 1 is **40% complete** - data infrastructure ready, CNN model not yet implemented.

### What's Implemented ✅

**Data Infrastructure**:
- ✅ RGB map generation from EA (U-Matrix + Distance + Dead Neurons)
- ✅ UID-based file naming and tracking
- ✅ Centralized dataset directory (`maps_dataset/`)
- ✅ Results CSV with quality metrics
- ✅ `prepare_data.py` script for CNN dataset preparation

**Data Preparation Script**: [app/cnn/prepare_data.py](../app/cnn/prepare_data.py)
- ✅ EA run directory parsing
- ✅ RGB image path collection
- ✅ Quality score calculation from metrics
- ✅ Dataset CSV generation with train/val/test splits
- ✅ Image resizing and normalization utilities

### What's Missing ❌

**CNN Model**:
- ❌ CNN architecture definition (custom or transfer learning)
- ❌ Multi-task learning setup (regression + classification)
- ❌ Training pipeline implementation
- ❌ Model checkpointing and versioning
- ❌ Inference wrapper class

**Human Annotation Pipeline**:
- ❌ Annotation tool for binary labeling (bad/not_bad)
- ❌ Multi-annotator aggregation logic
- ❌ Annotation database/file format

**Integration**:
- ❌ CNNQualityEvaluator class for EA integration
- ❌ Prediction caching mechanism
- ❌ Error handling for missing/corrupt images

### Phase Roadmap

- **Phase 1**: Data preparation and annotation ✅ (40% complete)
- **Phase 2**: CNN model training ❌ (0% complete)
- **Phase 3**: EA integration ❌ (0% complete)

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Data Preparation](#phase-1-data-preparation)
3. [Phase 2: CNN Model Development](#phase-2-cnn-model-development)
4. [Phase 3: Integration with EA](#phase-3-integration-with-ea)
5. [Requirements Traceability Matrix](#requirements-traceability-matrix)
6. [Document Information](#document-information)

---

## Overview

**The Eye (CNN)** is a convolutional neural network designed to visually assess the quality of Self-Organizing Maps by analyzing RGB images combining three map types:
- **R channel**: U-Matrix (topological structure)
- **G channel**: Distance Map (quantization error)
- **B channel**: Dead Neurons Map (neuron activity)

**Primary Objectives**:
1. **Regression**: Predict continuous quality score (0.0-1.0)
2. **Classification** (optional): Predict binary label (bad/not_bad)

**Integration Point**: Provides `cnn_quality_score` to EA for Phase 2 multi-objective optimization

---

## Phase 1: Data Preparation

### 1.1 Data Source Requirements

#### FR-CNN-1.1.1: EA Campaign Dependency ✅

**Requirement**: Utilize large-scale EA campaign output as primary training data source.

**Implementation**: [app/cnn/prepare_data.py](../app/cnn/prepare_data.py)

**Data Source Structure**:
```
ea_run_directory/
├── maps_dataset/
│   ├── rgb/
│   │   ├── {uid}_rgb.png     # 3-channel RGB images
│   │   └── ...
│   ├── {uid}_u_matrix.png    # Individual channel maps (backup)
│   ├── {uid}_distance_map.png
│   └── {uid}_dead_neurons_map.png
└── results.csv               # Quality metrics + hyperparameters
```

**Acceptance Criteria**:
- ✅ Parse EA results directory structure
- ✅ Load results.csv with quality metrics
- ✅ Locate RGB images by UID
- ✅ Handle multiple EA runs for dataset aggregation

**Verified**: `prepare_data.py` successfully processes EA run `20260110_220147`

---

#### FR-CNN-1.1.2: Minimum Dataset Size 🔜

**Requirement**: Ensure sufficient training data through large-scale EA campaign.

**Recommended Sizes**:
- **Minimum**: 1,000 unique SOM configurations
- **Recommended**: 5,000-10,000 configurations
- **Optimal**: 20,000+ configurations for robust generalization

**Acceptance Criteria**:
- [ ] Validate dataset size before training
- [ ] Warn if dataset < 1,000 samples
- [ ] Support multi-run aggregation for larger datasets

**Current Status**: Test run has 36 samples (insufficient for training)

---

### 1.2 Quality Score Calculation

#### FR-CNN-1.2.1: Composite Quality Score ✅

**Requirement**: Calculate normalized quality score from SOM metrics.

**Implementation**: [app/cnn/prepare_data.py](../app/cnn/prepare_data.py) lines 64-117

**Formula**:
```python
def calculate_quality_score(row: pd.Series, weights: dict) -> float:
    # Normalize metrics to 0-1 range (lower is better)
    norm_mqe = row['best_mqe'] / max_mqe
    norm_te = row['topographic_error']  # Already 0-1
    norm_dead = row['dead_neuron_ratio']  # Already 0-1

    # Weighted combination
    composite = (
        weights['mqe'] * norm_mqe +
        weights['te'] * norm_te +
        weights['dead'] * norm_dead
    )

    # Invert: higher score = better quality
    quality_score = 1.0 - composite

    return quality_score
```

**Default Weights**: Configurable via JSON
```json
{
  "quality_weights": {
    "mqe": 0.5,
    "te": 0.3,
    "dead": 0.2
  }
}
```

**Acceptance Criteria**:
- ✅ Normalize all metrics to 0-1 range
- ✅ Configurable weights for each metric
- ✅ Quality score range: 0.0 (worst) to 1.0 (best)
- ✅ Handle missing metrics gracefully

**Verified**: `prepare_data.py` calculates quality scores for all 36 samples

---

#### FR-CNN-1.2.2: Additional Metrics Support 🔜

**Requirement**: Support optional additional metrics in quality calculation.

**Optional Metrics**:
- `u_matrix_mean`: Cluster separation indicator
- `u_matrix_std`: Topology smoothness
- `training_duration`: Efficiency metric
- `epochs_ran`: Convergence indicator

**Acceptance Criteria**:
- [ ] Extensible weight configuration
- [ ] Automatic detection of available metrics
- [ ] Fallback to core metrics if optional ones missing

---

### 1.3 Human Annotation Pipeline

#### FR-CNN-1.3.1: Binary Annotation Tool ❌

**Requirement**: Simple tool for manual binary labeling of map quality.

**Status**: NOT IMPLEMENTED

**Planned Features**:
- Display RGB map image
- Annotator provides binary label: `bad` (1) or `not_bad` (0)
- Save annotations to CSV/JSON file
- Support batch annotation (multiple images per session)
- Progress tracking and resume capability

**Acceptance Criteria**:
- [ ] GUI or CLI tool for rapid annotation
- [ ] Keyboard shortcuts (e.g., 'b' = bad, 'g' = good, 's' = skip)
- [ ] Annotation persistence (CSV/JSON format)
- [ ] Image display with UID and quality metrics

**Planned Implementation**:
```python
# app/cnn/annotate.py
class AnnotationTool:
    def __init__(self, image_dir: str, annotations_file: str):
        """Load images and existing annotations."""

    def display_next_image(self):
        """Show next unlabeled image."""

    def save_annotation(self, uid: str, label: int):
        """Save binary label for UID."""

    def export_annotations(self) -> pd.DataFrame:
        """Export all annotations as DataFrame."""
```

---

#### FR-CNN-1.3.2: Multi-Annotator Aggregation ❌

**Requirement**: Aggregate annotations from multiple annotators for consensus labeling.

**Status**: NOT IMPLEMENTED

**Aggregation Strategies**:
1. **Majority Vote**: Label = mode of all annotations
2. **Soft Labels**: Label = mean of all annotations (0.0-1.0 continuous)
3. **Weighted Average**: Weight by annotator agreement rate
4. **Consensus Threshold**: Require N% agreement for labeling

**Acceptance Criteria**:
- [ ] Support multiple annotation files per UID
- [ ] Configurable aggregation strategy
- [ ] Inter-annotator agreement metrics (Cohen's Kappa)
- [ ] Filter samples with low agreement

**Planned Implementation**:
```python
def aggregate_annotations(annotations: List[pd.DataFrame],
                         strategy: str = 'majority') -> pd.DataFrame:
    """
    Aggregate multiple annotation files.

    Returns:
        DataFrame with columns: uid, label, confidence, num_annotators
    """
```

---

#### FR-CNN-1.3.3: Annotation Quality Metrics 🔜

**Requirement**: Track and report annotation quality statistics.

**Metrics**:
- Inter-annotator agreement (Cohen's Kappa, Fleiss' Kappa)
- Per-annotator consistency (re-annotation agreement)
- Label distribution (class imbalance)
- Annotation coverage (% of dataset labeled)

**Acceptance Criteria**:
- [ ] Generate annotation quality report
- [ ] Identify potentially mislabeled samples (low agreement)
- [ ] Flag annotators with low consistency

---

### 1.4 Dataset Preparation

#### FR-CNN-1.4.1: Multi-Channel Image Loading ✅

**Requirement**: Load and combine three map types into single RGB tensor.

**Implementation**: [app/cnn/prepare_data.py](../app/cnn/prepare_data.py) lines 119-170

**Loading Process**:
```python
def load_rgb_image(uid: str, maps_dir: str, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Load and preprocess RGB map image.

    Process:
    1. Load {uid}_rgb.png (already combined by EA)
    2. Resize to target_size (e.g., 224x224 for transfer learning)
    3. Normalize to [0, 1] range
    4. Return as (H, W, 3) numpy array
    """
```

**Alternative**: Load individual channels if RGB not available
```python
def load_individual_channels(uid: str, maps_dir: str) -> np.ndarray:
    """
    Load three separate grayscale maps and combine into RGB.

    Channels:
    - R: {uid}_u_matrix.png
    - G: {uid}_distance_map.png
    - B: {uid}_dead_neurons_map.png
    """
```

**Acceptance Criteria**:
- ✅ Support both RGB and individual channel loading
- ✅ Configurable target image size
- ✅ Normalization to [0, 1] range
- ✅ Error handling for missing/corrupt images

**Verified**: `prepare_data.py` successfully loads and processes RGB images

---

#### FR-CNN-1.4.2: Dataset CSV Generation ✅

**Requirement**: Create final dataset CSV linking images to labels and scores.

**Implementation**: [app/cnn/prepare_data.py](../app/cnn/prepare_data.py) lines 172-250

**Dataset CSV Structure**:
```csv
uid,image_path,quality_score,best_mqe,topographic_error,dead_neuron_ratio,split,label
041f5bf...,maps_dataset/rgb/041f5bf_rgb.png,0.423,0.424,0.027,0.9,train,1
6945148...,maps_dataset/rgb/6945148_rgb.png,0.782,0.135,0.013,0.578,train,0
...
```

**Columns**:
- `uid`: Unique identifier
- `image_path`: Path to RGB image (relative to EA run dir)
- `quality_score`: Calculated composite score (0-1)
- `best_mqe`, `topographic_error`, `dead_neuron_ratio`: Raw metrics
- `split`: train/val/test (stratified split)
- `label`: Human annotation (bad=1, not_bad=0) - optional, NaN if not annotated

**Acceptance Criteria**:
- ✅ All UIDs from results.csv included
- ✅ Quality score calculated for all samples
- ✅ Train/val/test split (default: 70/15/15)
- ✅ Stratified split by quality score bins
- ⚠️ Support for missing annotations (NaN handling)

**Verified**: Test dataset CSV generated with 36 samples

---

#### FR-CNN-1.4.3: Data Augmentation Pipeline 🔜

**Requirement**: Implement augmentation strategies to increase training data diversity.

**Status**: NOT IMPLEMENTED

**Augmentation Strategies**:
- **Geometric**: Rotation (90°, 180°, 270°), horizontal/vertical flip
- **Color**: Brightness/contrast adjustment (mild, preserve structure)
- **Noise**: Gaussian noise addition (mild)
- **NOT recommended**: Cropping, scaling (preserves spatial structure)

**Acceptance Criteria**:
- [ ] Configurable augmentation pipeline
- [ ] Apply only to training set
- [ ] Preserve validation/test sets (no augmentation)
- [ ] Generate augmented samples on-the-fly or pre-generate

**Planned Implementation**:
```python
# Using albumentations or torchvision.transforms
augmentation = A.Compose([
    A.Rotate(limit=90, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
    A.GaussNoise(var_limit=(0.0, 0.01), p=0.2)
])
```

---

#### FR-CNN-1.4.4: Dataset Splitting Strategy ✅

**Requirement**: Split dataset into train/validation/test sets with proper stratification.

**Implementation**: [app/cnn/prepare_data.py](../app/cnn/prepare_data.py) lines 200-230

**Splitting Strategy**:
```python
# Stratify by quality score bins
quality_bins = pd.cut(df['quality_score'], bins=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])

# Split: 70% train, 15% val, 15% test
train, temp = train_test_split(df, test_size=0.3, stratify=quality_bins, random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp['quality_bin'], random_state=42)
```

**Acceptance Criteria**:
- ✅ Stratified split by quality score
- ✅ Configurable split ratios
- ✅ Fixed random seed for reproducibility
- ✅ No data leakage between splits

**Verified**: Dataset split correctly in test run

---

## Phase 2: CNN Model Development

### 2.1 Model Architecture

#### FR-CNN-2.1.1: Architecture Selection ❌

**Requirement**: Define CNN architecture suitable for 3-channel map analysis.

**Status**: NOT IMPLEMENTED

**Architecture Options**:

**Option 1: Custom CNN**
```python
class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc_regression = nn.Linear(128, 1)  # Quality score
        self.fc_classification = nn.Linear(128, 1)  # Bad/not_bad

    def forward(self, x):
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)

        quality_score = torch.sigmoid(self.fc_regression(features))
        bad_label = self.fc_classification(features)

        return quality_score, bad_label
```

**Option 2: Transfer Learning (Recommended)**
```python
import torchvision.models as models

class TransferCNN(nn.Module):
    def __init__(self, base_model='mobilenet_v2', pretrained=True):
        super().__init__()
        # Load pretrained backbone
        if base_model == 'mobilenet_v2':
            backbone = models.mobilenet_v2(pretrained=pretrained)
            num_features = backbone.last_channel
            self.features = backbone.features
        elif base_model == 'resnet18':
            backbone = models.resnet18(pretrained=pretrained)
            num_features = backbone.fc.in_features
            self.features = nn.Sequential(*list(backbone.children())[:-1])

        # Dual heads
        self.fc_regression = nn.Linear(num_features, 1)
        self.fc_classification = nn.Linear(num_features, 1)

    def forward(self, x):
        features = self.features(x)
        features = features.view(features.size(0), -1)

        quality_score = torch.sigmoid(self.fc_regression(features))
        bad_label = self.fc_classification(features)

        return quality_score, bad_label
```

**Acceptance Criteria**:
- [ ] Support both custom and transfer learning architectures
- [ ] Configurable backbone selection
- [ ] Input size: (batch, 3, H, W) where H, W typically 224
- [ ] Dual output heads (regression + classification)

---

#### FR-CNN-2.1.2: Multi-Task Learning Setup ❌

**Requirement**: Train CNN on dual objectives (regression + classification).

**Status**: NOT IMPLEMENTED

**Loss Function**:
```python
def combined_loss(quality_pred, quality_true, label_pred, label_true, alpha=0.7):
    """
    Combined loss for multi-task learning.

    Args:
        quality_pred: Predicted quality scores (0-1)
        quality_true: True quality scores
        label_pred: Predicted logits for bad/not_bad
        label_true: True binary labels (0 or 1)
        alpha: Weight for regression loss (1-alpha for classification)

    Returns:
        Combined loss
    """
    regression_loss = F.mse_loss(quality_pred, quality_true)
    classification_loss = F.binary_cross_entropy_with_logits(label_pred, label_true)

    total_loss = alpha * regression_loss + (1 - alpha) * classification_loss

    return total_loss, regression_loss, classification_loss
```

**Acceptance Criteria**:
- [ ] MSE loss for regression task
- [ ] Binary cross-entropy for classification task
- [ ] Configurable loss weighting (alpha parameter)
- [ ] Handle missing labels (skip classification loss if NaN)

---

#### FR-CNN-2.1.3: Model Configuration ❌

**Requirement**: Support flexible model configuration via JSON/YAML.

**Status**: NOT IMPLEMENTED

**Configuration Structure**:
```json
{
  "model": {
    "architecture": "transfer",
    "backbone": "mobilenet_v2",
    "pretrained": true,
    "input_size": [224, 224],
    "dropout": 0.5
  },
  "training": {
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "scheduler": "reduce_on_plateau",
    "early_stopping_patience": 10
  },
  "loss": {
    "alpha": 0.7
  },
  "data_augmentation": {
    "enabled": true,
    "rotation_limit": 90,
    "flip_probability": 0.5
  }
}
```

**Acceptance Criteria**:
- [ ] Load configuration from JSON/YAML
- [ ] Override config with command-line arguments
- [ ] Validate configuration completeness
- [ ] Save config with trained model

---

### 2.2 Training Pipeline

#### FR-CNN-2.2.1: Training Script ❌

**Requirement**: Implement complete training pipeline with monitoring and checkpointing.

**Status**: NOT IMPLEMENTED

**Training Script Structure**:
```python
# app/cnn/train.py

class CNNTrainer:
    def __init__(self, config: dict):
        """Initialize trainer with configuration."""
        self.config = config
        self.model = self._build_model()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_loader, val_loader):
        """Main training loop."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate(val_loader)

            # Logging
            self._log_metrics(epoch, train_loss, val_loss, val_metrics)

            # Checkpointing
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(epoch, 'best')
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break

            # Learning rate scheduling
            self.scheduler.step(val_loss)

    def _train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0

        for batch in tqdm(train_loader):
            images, quality_scores, labels = batch
            images = images.to(self.device)

            # Forward pass
            quality_pred, label_pred = self.model(images)

            # Loss calculation
            loss, reg_loss, cls_loss = combined_loss(
                quality_pred, quality_scores, label_pred, labels, alpha=self.config['alpha']
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)

    def _validate(self, val_loader):
        """Validate on validation set."""
        self.model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                images, quality_scores, labels = batch
                images = images.to(self.device)

                quality_pred, label_pred = self.model(images)

                loss, _, _ = combined_loss(
                    quality_pred, quality_scores, label_pred, labels
                )

                val_loss += loss.item()
                all_preds.extend(quality_pred.cpu().numpy())
                all_targets.extend(quality_scores.numpy())

        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(all_targets, all_preds),
            'r2': r2_score(all_targets, all_preds)
        }

        return val_loss / len(val_loader), metrics
```

**Acceptance Criteria**:
- [ ] Data loading with configurable batch size
- [ ] Training/validation loop with progress bars
- [ ] Multi-task loss calculation
- [ ] Optimizer and scheduler support
- [ ] Early stopping based on validation loss
- [ ] Checkpointing (best model + periodic saves)
- [ ] Metric logging (loss, MAE, R²)

---

#### FR-CNN-2.2.2: Model Checkpointing ❌

**Requirement**: Save model checkpoints during training with versioning.

**Status**: NOT IMPLEMENTED

**Checkpoint Structure**:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'val_metrics': val_metrics,
    'config': config,
    'timestamp': datetime.now().isoformat()
}

torch.save(checkpoint, f'checkpoints/model_epoch_{epoch}.pt')
```

**Checkpoint Types**:
- **Best model**: Saved when validation loss improves
- **Latest model**: Saved every N epochs
- **Final model**: Saved at end of training

**Acceptance Criteria**:
- [ ] Save best model based on validation loss
- [ ] Periodic checkpoint saving (every N epochs)
- [ ] Include training state for resuming
- [ ] Checkpoint directory organization
- [ ] Model versioning (timestamp, git commit hash)

---

#### FR-CNN-2.2.3: Training Monitoring ❌

**Requirement**: Comprehensive logging and visualization of training progress.

**Status**: NOT IMPLEMENTED

**Logging Tools**:
- **TensorBoard**: Loss curves, metric plots
- **CSV logs**: Epoch-wise metrics
- **Console output**: Real-time progress

**Metrics to Track**:
- Training loss (total, regression, classification)
- Validation loss (total, regression, classification)
- Regression metrics: MAE, MSE, R²
- Classification metrics: Accuracy, Precision, Recall, F1 (if labels available)
- Learning rate (from scheduler)

**Acceptance Criteria**:
- [ ] TensorBoard integration
- [ ] CSV log export
- [ ] Real-time console updates with tqdm
- [ ] Plot generation (loss curves, metric evolution)

---

### 2.3 Model Evaluation

#### FR-CNN-2.3.1: Test Set Evaluation ❌

**Requirement**: Evaluate final model performance on held-out test set.

**Status**: NOT IMPLEMENTED

**Evaluation Metrics**:

**Regression Task**:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score
- Median Absolute Error

**Classification Task** (if labels available):
- Accuracy
- Precision, Recall, F1
- ROC-AUC
- Confusion Matrix

**Acceptance Criteria**:
- [ ] Load best checkpoint
- [ ] Evaluate on test set
- [ ] Generate comprehensive metrics report
- [ ] Save predictions for error analysis

**Planned Implementation**:
```python
def evaluate_model(model, test_loader, checkpoint_path):
    """Evaluate model on test set."""
    # Load best checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Collect predictions
    all_quality_pred = []
    all_quality_true = []
    all_label_pred = []
    all_label_true = []
    all_uids = []

    with torch.no_grad():
        for batch in test_loader:
            images, quality_scores, labels, uids = batch
            quality_pred, label_pred = model(images.to(device))

            all_quality_pred.extend(quality_pred.cpu().numpy())
            all_quality_true.extend(quality_scores.numpy())
            # ... collect all predictions

    # Calculate metrics
    metrics = {
        'regression': {
            'mae': mean_absolute_error(all_quality_true, all_quality_pred),
            'rmse': np.sqrt(mean_squared_error(all_quality_true, all_quality_pred)),
            'r2': r2_score(all_quality_true, all_quality_pred),
            'median_ae': median_absolute_error(all_quality_true, all_quality_pred)
        },
        'classification': {
            'accuracy': accuracy_score(all_label_true, all_label_pred),
            # ... other metrics
        }
    }

    # Save predictions
    predictions_df = pd.DataFrame({
        'uid': all_uids,
        'quality_true': all_quality_true,
        'quality_pred': all_quality_pred,
        'label_true': all_label_true,
        'label_pred': all_label_pred
    })
    predictions_df.to_csv('test_predictions.csv', index=False)

    return metrics, predictions_df
```

---

#### FR-CNN-2.3.2: Error Analysis ❌

**Requirement**: Identify and analyze prediction errors for model improvement.

**Status**: NOT IMPLEMENTED

**Analysis Types**:
1. **Worst Predictions**: Samples with highest prediction error
2. **Error Distribution**: Histogram of prediction errors
3. **Correlation Analysis**: Predicted vs actual scatter plot
4. **Failure Mode Analysis**: Visual inspection of misclassified samples

**Acceptance Criteria**:
- [ ] Generate error analysis report
- [ ] Visualize worst predictions with images
- [ ] Identify patterns in errors (e.g., specific metric ranges)
- [ ] Recommend data collection strategies

---

## Phase 3: Integration with EA

### 3.1 Inference Wrapper

#### FR-CNN-3.1.1: CNNQualityEvaluator Class ❌

**Requirement**: Encapsulate CNN inference in simple interface for EA integration.

**Status**: NOT IMPLEMENTED

**Class Interface**:
```python
# app/cnn/evaluator.py

class CNNQualityEvaluator:
    """CNN-based quality evaluator for SOM maps."""

    def __init__(self, model_checkpoint: str, config: dict = None):
        """
        Initialize evaluator with trained model.

        Args:
            model_checkpoint: Path to trained model checkpoint
            config: Optional model configuration (loaded from checkpoint if None)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_checkpoint, config)
        self.model.eval()
        self.transform = self._get_transform()
        self.prediction_cache = {}  # UID -> quality_score cache

    def predict(self, image_path: str, uid: str = None) -> float:
        """
        Predict quality score for RGB map image.

        Args:
            image_path: Path to RGB map image
            uid: Optional UID for caching

        Returns:
            Quality score (0.0-1.0, higher is better)
        """
        # Check cache
        if uid and uid in self.prediction_cache:
            return self.prediction_cache[uid]

        # Load and preprocess image
        image = self._load_image(image_path)
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            quality_score, _ = self.model(image_tensor)

        score = quality_score.item()

        # Cache prediction
        if uid:
            self.prediction_cache[uid] = score

        return score

    def predict_batch(self, image_paths: List[str], uids: List[str] = None) -> List[float]:
        """Batch prediction for efficiency."""
        # ... batch inference implementation

    def _load_model(self, checkpoint_path: str, config: dict):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if config is None:
            config = checkpoint['config']

        model = build_model(config)  # Uses FR-CNN-2.1.1
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        return model

    def _get_transform(self):
        """Get image preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def _load_image(self, image_path: str) -> Image:
        """Load RGB image with error handling."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert('RGB')
        return image
```

**Acceptance Criteria**:
- [ ] Simple `predict(image_path)` interface
- [ ] Automatic model loading from checkpoint
- [ ] Prediction caching by UID
- [ ] Batch inference support
- [ ] Error handling (missing files, corrupt images)
- [ ] GPU support with CPU fallback

---

#### FR-CNN-3.1.2: EA Integration ❌

**Requirement**: Integrate CNNQualityEvaluator into EA Phase 2 for fitness augmentation.

**Status**: NOT IMPLEMENTED

**Integration Point**: [app/ea/ea.py](../app/ea/ea.py) `evaluate_individual()` function

**Modified Evaluation**:
```python
# In evaluate_individual()

# After SOM training and metric calculation
training_results['topographic_error'] = som.calculate_topographic_error(data, mask=ignore_mask)
dead_count, dead_ratio = som.calculate_dead_neurons(data)
training_results['dead_neuron_count'] = dead_count
training_results['dead_neuron_ratio'] = dead_ratio

# CNN quality prediction
if cnn_evaluator is not None:
    rgb_image_path = os.path.join(individual_dir, "visualizations", f"{uid}_rgb.png")

    # Ensure RGB image exists
    if os.path.exists(rgb_image_path):
        cnn_quality_score = cnn_evaluator.predict(rgb_image_path, uid=uid)
        training_results['cnn_quality_score'] = cnn_quality_score
    else:
        # Fallback if RGB not generated yet
        training_results['cnn_quality_score'] = None
```

**Updated NSGA-II Objectives**:
```python
# In run_evolution()

objectives = np.array([
    [
        res['best_mqe'],
        res['duration'],
        res.get('topographic_error', 1.0),
        res.get('dead_neuron_ratio', 1.0),
        1.0 - res.get('cnn_quality_score', 0.0)  # Invert: minimize (1 - quality)
    ]
    for cfg, res in combined_population
])
```

**Acceptance Criteria**:
- [ ] Load CNNQualityEvaluator at EA start
- [ ] Predict CNN quality score during evaluation
- [ ] Add `cnn_quality_score` to results.csv
- [ ] Include CNN score in NSGA-II objectives (5th objective)
- [ ] Handle missing CNN predictions gracefully

---

#### FR-CNN-3.1.3: Prediction Caching ❌

**Requirement**: Cache CNN predictions to avoid re-computation for duplicate configurations.

**Status**: NOT IMPLEMENTED (basic structure in CNNQualityEvaluator)

**Caching Strategy**:
```python
class PredictionCache:
    """Persistent cache for CNN predictions."""

    def __init__(self, cache_file: str = 'cnn_predictions_cache.json'):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        """Load cache from disk."""
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def get(self, uid: str) -> Optional[float]:
        """Get cached prediction."""
        return self.cache.get(uid)

    def set(self, uid: str, score: float):
        """Save prediction to cache."""
        self.cache[uid] = score

    def save(self):
        """Persist cache to disk."""
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=2)
```

**Acceptance Criteria**:
- [ ] In-memory cache (dict)
- [ ] Persistent cache (JSON file)
- [ ] Automatic save on EA completion
- [ ] Cache invalidation mechanism (model version tracking)

---

## Requirements Traceability Matrix

| Requirement ID | Description | Status | Implementation | Verified |
|---------------|-------------|--------|----------------|----------|
| **FR-CNN-1.1.1** | EA campaign dependency | ✅ | prepare_data.py | Test run parsed |
| **FR-CNN-1.1.2** | Minimum dataset size | 🔜 | Validation needed | - |
| **FR-CNN-1.2.1** | Quality score calculation | ✅ | prepare_data.py L64-117 | 36 samples |
| **FR-CNN-1.2.2** | Additional metrics | 🔜 | Extensible weights | - |
| **FR-CNN-1.3.1** | Binary annotation tool | ❌ | Not implemented | - |
| **FR-CNN-1.3.2** | Multi-annotator aggregation | ❌ | Not implemented | - |
| **FR-CNN-1.3.3** | Annotation quality metrics | 🔜 | Planning | - |
| **FR-CNN-1.4.1** | Multi-channel image loading | ✅ | prepare_data.py L119-170 | RGB loaded |
| **FR-CNN-1.4.2** | Dataset CSV generation | ✅ | prepare_data.py L172-250 | CSV created |
| **FR-CNN-1.4.3** | Data augmentation | 🔜 | Planning | - |
| **FR-CNN-1.4.4** | Dataset splitting | ✅ | prepare_data.py L200-230 | Stratified split |
| **FR-CNN-2.1.1** | Architecture selection | ❌ | Not implemented | - |
| **FR-CNN-2.1.2** | Multi-task learning | ❌ | Not implemented | - |
| **FR-CNN-2.1.3** | Model configuration | ❌ | Not implemented | - |
| **FR-CNN-2.2.1** | Training script | ❌ | Not implemented | - |
| **FR-CNN-2.2.2** | Model checkpointing | ❌ | Not implemented | - |
| **FR-CNN-2.2.3** | Training monitoring | ❌ | Not implemented | - |
| **FR-CNN-2.3.1** | Test set evaluation | ❌ | Not implemented | - |
| **FR-CNN-2.3.2** | Error analysis | ❌ | Not implemented | - |
| **FR-CNN-3.1.1** | CNNQualityEvaluator | ❌ | Not implemented | - |
| **FR-CNN-3.1.2** | EA integration | ❌ | Not implemented | - |
| **FR-CNN-3.1.3** | Prediction caching | ❌ | Not implemented | - |

**Summary**: 6/23 requirements implemented (26%)

---

## Document Information

### Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-11 | Initial CNN requirements document | Claude Sonnet 4.5 |

### Implementation Summary

**Phase 1 (Data Preparation)**: 40% complete
- ✅ EA data pipeline working
- ✅ Quality score calculation implemented
- ✅ Dataset CSV generation working
- ❌ Human annotation tool not implemented

**Phase 2 (CNN Model)**: 0% complete
- ❌ Model architecture not defined
- ❌ Training pipeline not implemented
- ❌ Evaluation framework not implemented

**Phase 3 (Integration)**: 0% complete
- ❌ CNNQualityEvaluator not implemented
- ❌ EA integration not complete

**Next Immediate Steps**:
1. Run large-scale EA campaign (5,000+ configurations)
2. Implement human annotation tool
3. Annotate subset of dataset (~1,000 samples)
4. Define CNN architecture (recommend transfer learning)
5. Implement training pipeline
6. Train initial CNN model
7. Implement CNNQualityEvaluator
8. Integrate with EA Phase 2

---

**End of CNN Requirements Specification**
