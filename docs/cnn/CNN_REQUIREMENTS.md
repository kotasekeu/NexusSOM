# The Eye (CNN) - Visual Quality Assessment Requirements

**Document Version**: 2.1
**Last Updated**: 2026-02-02
**Component**: CNN - "The Eye"
**Purpose**: Visual quality assessment of SOM maps through deep learning

---

## Executive Summary

**Current Implementation Status**: Phase 2 is **77% complete** (10/13 requirements) - Ready for first training run.

### What's Implemented ✅

**CNN Model** (`app/cnn/src/model.py`):
- ✅ GAP (Global Average Pooling) architecture for variable input sizes
- ✅ Support for SOM maps 5x5 to 30x30 (1 neuron = 1 pixel, no interpolation)
- ✅ Standard model (4 conv blocks, 256 features)
- ✅ Lightweight model (3 conv blocks, 128 features)
- ✅ MSE loss for regression (quality score 0-1)

**Training Pipeline** (`app/cnn/src/train.py`):
- ✅ GroupedSizeDataLoader - batches images by size (no interpolation artifacts)
- ✅ Data augmentation (horizontal/vertical flip only)
- ✅ Model checkpointing (best + periodic)
- ✅ EarlyStopping, ReduceLROnPlateau
- ✅ TensorBoard logging
- ✅ CSV training log

**SOM Converter** (`app/cnn/src/som_converter.py`):
- ✅ SOM weights → RGB image conversion
- ✅ Multiple visualization methods (RGB, U-matrix, combined)
- ✅ Native size and fixed size output

**Data Pipeline** (`app/cnn/src/prepare_dataset.py`):
- ✅ Auto-labeling based on metrics (dead_neuron_ratio, topographic_error)
- ✅ Pseudo-labeling with trained model
- ✅ Dataset preparation from EA results
- ✅ Integration with `data/cnn/` directory structure
- ✅ Image organization by size (5x5/, 10x10/, etc.)
- ✅ Dataset CSV generation

**Tested with BreastCancer dataset**:
- ✅ 1000 EA results parsed successfully
- ✅ 884 samples auto-labeled (88.4%)
- ✅ Images organized in size-specific directories
- ✅ Dataset CSV created: `data/cnn/datasets/dataset_v1.csv`

### What's Missing ❌

**Inference**:
- ❌ Inference script for prediction (`predict.py`)
- ❌ CNNQualityEvaluator class for EA integration (`evaluator.py`)

---

## Directory Structure

```
NexusSom/
├── app/cnn/
│   └── src/
│       ├── model.py          # CNN architecture (GAP)
│       ├── train.py          # Training script with GroupedSizeDataLoader
│       ├── som_converter.py  # SOM weights → image conversion
│       ├── predict.py        # Inference script (TODO)
│       └── prepare_dataset.py # Dataset preparation (TODO)
│
├── data/
│   ├── datasets/
│   │   ├── BreastCancer/
│   │   │   └── results/EA/results.csv  # EA results with map_size
│   │   └── LungCancer/
│   │       └── results/EA/results.csv
│   │
│   └── cnn/                   # CNN training data (TODO)
│       ├── images/            # SOM map images organized by size
│       │   ├── 5x5/
│       │   ├── 10x10/
│       │   └── ...
│       ├── labels/            # Labels (auto + manual)
│       │   ├── auto_labels.csv
│       │   └── manual_labels.csv
│       └── datasets/          # Prepared training datasets
│           └── dataset_v1.csv
```

---

## Phase 1: Data Preparation

### 1.1 Data Sources

#### FR-CNN-1.1.1: EA Results Integration ✅

**Source**: `data/datasets/{DatasetName}/results/EA/results.csv`

**Available columns**:
```
uid, best_mqe, duration, topographic_error, u_matrix_mean, u_matrix_std,
u_matrix_max, distance_map_max, total_weight_updates, epochs_ran,
dead_neuron_count, dead_neuron_ratio, map_size, processing_type,
start_learning_rate, end_learning_rate, lr_decay_type, ...
```

**Key fields for labeling**:
- `map_size`: `"[5, 5]"` to `"[30, 30]"` - determines image size
- `best_mqe`: Lower is better (quantization error)
- `topographic_error`: Lower is better (topology preservation)
- `dead_neuron_ratio`: Lower is better (neuron utilization)

---

### 1.2 Pseudo-Labeling Pipeline

#### FR-CNN-1.2.1: Auto-Labeling Based on Metrics ✅

**Requirement**: Automatically label extreme cases (clearly good/bad maps).

**Implementation** (`app/cnn/src/prepare_dataset.py`):

```python
class AutoLabeler:
    def label(self, row: pd.Series) -> Tuple[Optional[float], str]:
        """
        Auto-label based on metrics. Returns (score, label_source) or (None, 'unlabeled').

        Label scale: 0.0 (worst) to 1.0 (best)
        """
        dead = row.get('dead_neuron_ratio', 0)
        topo = row.get('topographic_error', 0)

        # Clearly BAD maps
        if dead > 0.30:  # >30% dead neurons
            return 0.1, 'auto_dead'
        if topo > 0.50:  # >50% topographic error
            return 0.1, 'auto_topo'

        # Clearly GOOD maps
        if dead < 0.05 and topo < 0.10:
            return 0.9, 'auto_good'

        # Uncertain - needs pseudo-labeling
        return None, 'unlabeled'
```

**Acceptance Criteria**:
- [x] Auto-label extreme cases with high confidence
- [x] Return `None` for uncertain cases
- [x] Track label source (auto_dead, auto_topo, auto_good, unlabeled)

**Test Results** (BreastCancer dataset, 1000 samples):
- `auto_topo`: 469 (46.9%)
- `auto_dead`: 414 (41.4%)
- `auto_good`: 1 (0.1%)
- `unlabeled`: 116 (11.6%)

---

#### FR-CNN-1.2.2: Iterative Pseudo-Labeling ✅

**Requirement**: Use trained CNN to label uncertain cases, retrain with expanded dataset.

**Implementation** (`app/cnn/src/prepare_dataset.py`):

```python
class PseudoLabeler:
    def pseudo_label(self, unlabeled_df, model_path, confidence_threshold=0.15):
        """
        Use trained CNN to label uncertain cases.

        Args:
            confidence_threshold: Distance from 0.5 to consider confident
                                 (e.g., 0.15 means <0.35 or >0.65 is confident)
        """
        # Load trained model
        # Predict on unlabeled samples
        # Add confident predictions to training set
        # Track pseudo-label iteration
```

**Usage**:
```bash
# Step 1: Prepare initial dataset with auto-labels
python app/cnn/src/prepare_dataset.py --output data/cnn/datasets/dataset_v1.csv

# Step 2: Train CNN on auto-labeled data
python app/cnn/src/train.py --dataset data/cnn/datasets/dataset_v1.csv

# Step 3: Pseudo-label uncertain cases
python app/cnn/src/prepare_dataset.py --pseudo-label --model app/cnn/models/best.keras

# Step 4: Retrain with expanded dataset (repeat as needed)
```

**Acceptance Criteria**:
- [x] Confidence threshold for pseudo-labels
- [x] Track pseudo-label iterations (label_source: pseudo_v1, pseudo_v2, etc.)
- [x] Prevent label drift (original auto-labels never change)

---

### 1.3 Image Generation

#### FR-CNN-1.3.1: SOM Weights to Image Conversion ✅

**Implementation**: `app/cnn/src/som_converter.py`

**Process**:
```python
from som_converter import SOMToImageConverter

converter = SOMToImageConverter(target_size=None, method='rgb')
image = converter.convert(som_weights)  # (rows, cols, 3)
converter.save_image(som_weights, 'path/to/image.png')
```

**Key Features**:
- Native size (1 neuron = 1 pixel) - no interpolation
- RGB from first 3 weight dimensions
- U-matrix visualization option
- Combined RGB + U-matrix option

---

#### FR-CNN-1.3.2: Image Organization by Size ✅

**Requirement**: Organize images by map size for GroupedSizeDataLoader.

**Implementation**: Automatically handled by `prepare_dataset.py`

**Directory Structure**:
```
data/cnn/images/
├── 5x5/
│   ├── {uid}.png
│   └── ...
├── 10x10/
│   ├── {uid}.png
│   └── ...
└── 30x30/
    └── ...
```

**Acceptance Criteria**:
- [x] Parse map_size from results.csv (`"[10, 10]"` → `10x10`)
- [x] Create size-specific directories automatically
- [x] Generate images from SOM weights or copy from EA results

**Test Results** (BreastCancer dataset):
- Created 16 size-specific directories (5x5 to 20x20)
- 1000 images organized successfully
- Example: `data/cnn/images/19x19/` contains 129 images

---

### 1.4 Dataset CSV Format

#### FR-CNN-1.4.1: Training Dataset CSV ✅

**Location**: `data/cnn/datasets/dataset_v1.csv`

**Implementation**: Generated by `prepare_dataset.py`

**Format**:
```csv
filepath,quality_score,map_width,uid,label_source,dataset_name,dead_neuron_ratio,topographic_error,best_mqe
data/cnn/images/5x5/a8894a53.png,0.1,5,a8894a53,auto_topo,BreastCancer,0.0,2.32,5.93
data/cnn/images/10x10/5e787849.png,0.1,10,5e787849,auto_dead,BreastCancer,0.62,0.30,1.73
```

**Columns**:
- `filepath`: Relative path to image from project root
- `quality_score`: 0.0-1.0 (target for CNN regression)
- `map_width`: Integer (5-30) - extracted from map_size
- `uid`: Unique identifier from EA results
- `label_source`: `auto_topo`, `auto_dead`, `auto_good`, `unlabeled`, `pseudo_v1`, etc.
- `dataset_name`: Source dataset (BreastCancer, LungCancer, etc.)
- `dead_neuron_ratio`: Original metric from EA
- `topographic_error`: Original metric from EA
- `best_mqe`: Original metric from EA

**Test Results**:
- BreastCancer dataset: 884 labeled samples (116 unlabeled)
- CSV saved to: `data/cnn/datasets/dataset_v1.csv`
- Labels also saved to: `data/cnn/labels/auto_labels.csv`

---

## Phase 2: CNN Model

### 2.1 Architecture

#### FR-CNN-2.1.1: GAP Architecture ✅

**Implementation**: `app/cnn/src/model.py`

**Key Design Decisions**:
1. **No MaxPooling**: Preserves spatial info for small maps (5x5)
2. **padding='same'**: Maintains dimensions through conv layers
3. **GlobalAveragePooling2D**: Handles any input size → fixed 256 features
4. **Regression output**: Single neuron with sigmoid (0-1)

**Standard Model**:
```
Input (None, None, 3)
├── Conv Block 1: 32 filters, BN, ReLU, Dropout(0.25)
├── Conv Block 2: 64 filters, BN, ReLU, Dropout(0.25)
├── Conv Block 3: 128 filters, BN, ReLU, Dropout(0.3)
├── Conv Block 4: 256 filters, BN, ReLU, Dropout(0.3)
├── GlobalAveragePooling2D → 256 features
├── Dense 256 + BN + ReLU + Dropout(0.5)
├── Dense 128 + BN + ReLU + Dropout(0.5)
└── Dense 1 + Sigmoid → quality_score
```

---

### 2.2 Training

#### FR-CNN-2.2.1: GroupedSizeDataLoader ✅

**Implementation**: `app/cnn/src/train.py`

**Key Features**:
- Groups images by size (5x5, 10x10, etc.)
- Batches contain only same-sized images
- No interpolation artifacts
- Efficient GPU utilization (batch_size > 1)

**Usage**:
```bash
cd /path/to/NexusSom
python app/cnn/src/train.py \
    --dataset data/cnn/datasets/dataset_v1.csv \
    --model standard \
    --epochs 50 \
    --batch-size 32 \
    --learning-rate 0.001
```

---

#### FR-CNN-2.2.2: Callbacks ✅

**Implemented**:
- `ModelCheckpoint`: Best model + periodic saves (`.keras` format)
- `EarlyStopping`: patience=15 on val_loss
- `ReduceLROnPlateau`: factor=0.5, patience=5
- `TensorBoard`: Loss curves, histograms
- `CSVLogger`: Epoch-wise metrics

---

### 2.3 Data Augmentation

#### FR-CNN-2.3.1: Augmentation Strategy ✅

**Implementation**: `app/cnn/src/train.py` (ImageDataGenerator)

**Applied augmentations**:
- Horizontal flip (p=0.5)
- Vertical flip (p=0.5)

**NOT applied** (would distort small maps):
- Rotation (meaningless for grid topology)
- Zoom/crop (loses neurons)
- Color jitter (changes semantic meaning)

---

## Phase 3: Inference & Integration

### 3.1 Inference Script

#### FR-CNN-3.1.1: Predict Script ❌

**Requirement**: Standalone prediction script.

**Planned Location**: `app/cnn/src/predict.py`

**Usage**:
```bash
python app/cnn/src/predict.py \
    --model models/som_quality_best.keras \
    --image path/to/som_map.png
# Output: Quality score: 0.847

python app/cnn/src/predict.py \
    --model models/som_quality_best.keras \
    --weights path/to/som_weights.npy
# Output: Quality score: 0.847
```

---

#### FR-CNN-3.1.2: CNNQualityEvaluator ❌

**Requirement**: Class for EA integration.

**Planned Location**: `app/cnn/src/evaluator.py`

```python
class CNNQualityEvaluator:
    def __init__(self, model_path: str):
        self.model = keras.models.load_model(model_path)
        self.converter = SOMToImageConverter()

    def predict_from_weights(self, weights: np.ndarray) -> float:
        """Predict quality from SOM weight matrix."""
        image = self.converter.convert(weights)
        image = np.expand_dims(image, axis=0)  # Add batch dim
        return self.model.predict(image, verbose=0)[0][0]

    def predict_from_image(self, image_path: str) -> float:
        """Predict quality from saved image."""
        image = load_image(image_path)
        image = np.expand_dims(image, axis=0)
        return self.model.predict(image, verbose=0)[0][0]
```

---

### 3.2 EA Integration

#### FR-CNN-3.2.1: Quality Score in EA ❌

**Requirement**: Add CNN quality score to EA evaluation.

**Integration point**: `app/ea/ea.py` in `evaluate_individual()`

```python
# After SOM training
if cnn_evaluator is not None:
    cnn_score = cnn_evaluator.predict_from_weights(som.weights)
    results['cnn_quality_score'] = cnn_score
```

---

## Requirements Traceability Matrix

| Requirement ID | Description | Status | Implementation | Test Status |
|---------------|-------------|--------|----------------|-------------|
| **FR-CNN-1.1.1** | EA results integration | ✅ | results.csv parsing | ✅ Tested |
| **FR-CNN-1.2.1** | Auto-labeling | ✅ | prepare_dataset.py | ✅ Tested |
| **FR-CNN-1.2.2** | Iterative pseudo-labeling | ✅ | prepare_dataset.py | 🔄 Ready |
| **FR-CNN-1.3.1** | SOM to image conversion | ✅ | som_converter.py | ✅ Tested |
| **FR-CNN-1.3.2** | Image organization by size | ✅ | prepare_dataset.py | ✅ Tested |
| **FR-CNN-1.4.1** | Dataset CSV format | ✅ | prepare_dataset.py | ✅ Tested |
| **FR-CNN-2.1.1** | GAP architecture | ✅ | model.py | ✅ Tested |
| **FR-CNN-2.2.1** | GroupedSizeDataLoader | ✅ | train.py | 🔄 Ready |
| **FR-CNN-2.2.2** | Training callbacks | ✅ | train.py | 🔄 Ready |
| **FR-CNN-2.3.1** | Data augmentation | ✅ | train.py | 🔄 Ready |
| **FR-CNN-3.1.1** | Predict script | ❌ | predict.py | ⏸️ Pending |
| **FR-CNN-3.1.2** | CNNQualityEvaluator | ❌ | evaluator.py | ⏸️ Pending |
| **FR-CNN-3.2.1** | EA integration | ❌ | ea.py | ⏸️ Pending |

**Summary**: 10/13 requirements implemented (77%)

---

## Next Steps

### Immediate (Ready to Execute)

1. ✅ **Create `data/cnn/` directory structure** - DONE
2. ✅ **Implement `app/cnn/src/prepare_dataset.py`** - DONE
   - ✅ Parse all EA results.csv files
   - ✅ Auto-label extreme cases
   - ✅ Generate/copy images organized by size
   - ✅ Create dataset CSV
3. **Run first training with auto-labels**:
   ```bash
   python3 app/cnn/src/train.py \
       --dataset data/cnn/datasets/dataset_v1.csv \
       --model standard \
       --epochs 50 \
       --batch-size 32
   ```

### Short-term (After First Training)

4. **Iterate pseudo-labeling**:
   ```bash
   python3 app/cnn/src/prepare_dataset.py \
       --pseudo-label \
       --model app/cnn/models/best.keras
   ```
5. **Retrain with expanded dataset**

### Medium-term (Integration)

6. **Implement `app/cnn/src/predict.py`** - inference script
7. **Implement `app/cnn/src/evaluator.py`** - CNNQualityEvaluator for EA
8. **Integrate CNN into EA pipeline** - add `cnn_quality_score` to objectives

---

**End of CNN Requirements Specification v2.0**
