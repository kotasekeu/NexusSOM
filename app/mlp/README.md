# MLP - "The Prophet"

**Hyperparameter Quality Predictor**

Multi-Layer Perceptron that predicts SOM quality metrics from hyperparameters WITHOUT running SOM training.

## Overview

The Prophet analyzes hyperparameter configurations and predicts:
- **MQE (Mean Quantization Error)** - How well data is represented
- **Topographic Error** - How well topology is preserved
- **Dead Neuron Ratio** - Proportion of unused neurons

This enables the EA to evaluate candidate configurations instantly, speeding up evolution dramatically.

---

## Quick Start

### 1. Prepare Dataset

```bash
cd /Users/tomas/OSU/Python/NexusSom/app

# From single EA run
python3 mlp/prepare_dataset.py --results_dir ./test/results/20260112_140511

# From multiple EA runs (better training data)
python3 mlp/prepare_dataset.py \
    --results_dir ./test/results/20260112_140511,./test/results/20260112_143240 \
    --output ./mlp/data/combined_dataset.csv
```

### 2. Train Model

```bash
cd mlp

# Standard model
python3 src/train.py --dataset data/dataset.csv --epochs 100

# Lightweight model (faster inference)
python3 src/train.py --dataset data/dataset.csv --model lite --epochs 100

# Custom hyperparameters
python3 src/train.py --dataset data/dataset.csv \
    --epochs 200 --batch-size 64 --learning-rate 0.001
```

### 3. Evaluate Model

```bash
# Evaluate on test set
python3 evaluate_model.py \
    --model models/mlp_prophet_standard_20260112_220000_best.keras \
    --scaler models/mlp_prophet_standard_20260112_220000_scaler.pkl \
    --dataset data/dataset.csv
```

---

## Architecture

### Standard Model
```
Input (20 features)
    ↓
Dense(256) + BatchNorm + Dropout(0.3)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + BatchNorm + Dropout(0.2)
    ↓
Dense(32) + Dropout(0.2)
    ↓
Output (3: mqe, topo_error, dead_ratio)
```

**Total Parameters:** ~100K

### Lightweight Model
```
Input (20 features)
    ↓
Dense(128) + Dropout(0.2)
    ↓
Dense(64) + Dropout(0.2)
    ↓
Dense(32)
    ↓
Output (3)
```

**Total Parameters:** ~30K

---

## Input Features

After encoding, approximately 20 features:

### Numeric Features:
- `map_rows`, `map_cols` (from map_size)
- `start_learning_rate`, `end_learning_rate`
- `start_radius`, `end_radius`
- `start_batch_percent`, `end_batch_percent`
- `epoch_multiplier`
- `growth_g`
- `num_batches`
- `max_epochs_without_improvement`
- `normalize_weights_flag` (0/1)

### Categorical Features (One-Hot Encoded):
- `lr_decay_type`: linear, exp-drop, log-drop
- `radius_decay_type`: linear, exp-drop, log-drop
- `batch_growth_type`: linear, exp-growth, log-growth

---

## Output Targets

3 quality metrics (continuous values):

1. **best_mqe** - Mean Quantization Error
   - Lower is better
   - Typical range: 0.001 - 1.0

2. **topographic_error** - Topology Preservation
   - Lower is better
   - Range: 0.0 - 1.0

3. **dead_neuron_ratio** - Dead Neurons
   - Lower is better
   - Range: 0.0 - 1.0

---

## Files Structure

```
mlp/
├── README.md                   # This file
├── prepare_dataset.py          # Dataset preparation script
├── evaluate_model.py           # Model evaluation script
├── src/
│   ├── model.py               # Model architectures
│   └── train.py               # Training script
├── data/                      # Training data (generated)
│   ├── dataset.csv
│   └── dataset_metadata.json
├── models/                    # Trained models (generated)
│   ├── mlp_prophet_*_best.keras
│   ├── mlp_prophet_*_final.keras
│   ├── mlp_prophet_*_scaler.pkl
│   └── mlp_prophet_*_metadata.json
└── logs/                      # Training logs (generated)
    └── mlp_prophet_*/
```

---

## Dataset Preparation

The `prepare_dataset.py` script:

1. Loads `results.csv` from EA run(s)
2. Extracts hyperparameter columns
3. Parses `map_size` into separate `map_rows`, `map_cols`
4. One-hot encodes categorical features
5. Extracts target quality metrics
6. Saves dataset CSV and metadata JSON

**Output:**
- `dataset.csv` - Combined features and targets
- `dataset_metadata.json` - Feature names, encoders, statistics

---

## Training

The training script:

1. Loads dataset and splits into train/val/test (70/15/15)
2. Standardizes features using StandardScaler
3. Creates MLP model
4. Trains with callbacks:
   - ModelCheckpoint (saves best model)
   - EarlyStopping (patience=20)
   - ReduceLROnPlateau (patience=10)
   - TensorBoard logging
5. Evaluates on test set
6. Saves model, scaler, and metadata

**Outputs:**
- `models/mlp_prophet_*_best.keras` - Best model
- `models/mlp_prophet_*_final.keras` - Final model
- `models/mlp_prophet_*_scaler.pkl` - Fitted scaler
- `models/mlp_prophet_*_metadata.json` - Model info
- `logs/mlp_prophet_*/` - Training logs

---

## Evaluation

Metrics calculated:
- **MAE (Mean Absolute Error)** - Average prediction error
- **RMSE (Root Mean Squared Error)** - Penalizes large errors
- Per-target metrics for each output

---

## Integration with EA

### Use Case: Fast Fitness Estimation

Instead of running SOM training for every candidate:

```python
# Traditional EA (slow)
def evaluate_candidate(config):
    som = train_som(config)  # 10-60 seconds
    return calculate_metrics(som)

# With The Prophet (fast)
def evaluate_candidate(config):
    features = encode_config(config)
    prediction = prophet_model.predict(features)  # < 1ms
    return prediction
```

### Benefits:
- **1000× faster** - Milliseconds vs seconds
- **More generations** - Explore larger search space
- **Better solutions** - More evolution time

### Hybrid Approach:
1. Use Prophet for initial filtering (eliminate obviously bad configs)
2. Run actual SOM training only on promising candidates
3. Use results to retrain Prophet (active learning)

---

## Performance Expectations

Based on typical EA results:

**Expected Metrics:**
- MAE: 0.01 - 0.05 (depending on data quality)
- RMSE: 0.02 - 0.08

**Good Performance Indicators:**
- MAE < 0.03 for MQE prediction
- MAE < 0.05 for topographic error
- MAE < 0.10 for dead neuron ratio

---

## Troubleshooting

### "No module named 'joblib'"
```bash
pip install joblib
```

### "Feature mismatch"
Ensure the scaler was trained with the same features as the model.
Check `metadata.json` for feature column names.

### "Low accuracy"
- **Collect more data**: Run more EA experiments
- **Feature engineering**: Add interaction features
- **Tune hyperparameters**: Adjust learning rate, dropout, layers

### "Overfitting"
- **Increase dropout**: Try 0.4-0.5
- **Add regularization**: L2 penalty on weights
- **Early stopping**: Reduce patience
- **More data**: Collect from diverse EA runs

---

## Next Steps

1. **Collect More Data**: Run EA with diverse datasets and configurations
2. **Feature Engineering**: Add derived features (e.g., learning_rate_ratio)
3. **Ensemble Models**: Combine multiple MLPs for better predictions
4. **Active Learning**: Retrain on EA feedback
5. **Integration**: Use in EA for real-time fitness estimation

---

## References

- [TensorFlow Keras Documentation](https://www.tensorflow.org/api_docs/python/tf/keras)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)
- [MLP Best Practices](https://cs231n.github.io/neural-networks-1/)
