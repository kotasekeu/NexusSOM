# LSTM - "The Oracle"

**Training Progress Predictor**

LSTM network that predicts final SOM quality from early training progress, enabling intelligent early stopping.

## Overview

The Oracle analyzes time-series training metrics and predicts final quality:
- **MQE trajectory** → Final MQE
- **Topographic error progression** → Final topology quality
- **Dead neuron evolution** → Final dead neuron ratio

This enables terminating bad training runs early, saving 50-90% of training time.

---

## Quick Start

### 1. Collect Training Data

**NOTE:** This proof-of-concept uses simulated data. For production, modify SOM training code to log intermediate metrics.

```bash
cd /Users/tomas/OSU/Python/NexusSom/app

# Collect from single EA run
python3 lstm/collect_training_data.py --results_dir ./test/results/20260112_140511

# Custom checkpoints
python3 lstm/collect_training_data.py \
    --results_dir ./test/results/20260112_140511 \
    --output ./lstm/data/dataset.csv \
    --checkpoints 20
```

### 2. Train Model

```bash
cd lstm

# Standard LSTM
python3 src/train.py --dataset data/dataset.csv --epochs 100

# Bidirectional LSTM (better accuracy)
python3 src/train.py --dataset data/dataset.csv --model bidirectional --epochs 100

# Lightweight LSTM (faster inference)
python3 src/train.py --dataset data/dataset.csv --model lite --epochs 100
```

### 3. Evaluate Model

```bash
# Evaluate on test sequences
python3 evaluate_model.py \
    --model models/lstm_oracle_standard_20260112_220000_best.keras \
    --dataset data/dataset.csv

# Predict from partial sequence (early stopping)
python3 evaluate_model.py \
    --model models/lstm_oracle_standard_20260112_220000_best.keras \
    --sequence path/to/partial_sequence.json
```

---

## Architecture

### Standard LSTM
```
Input (sequence_length, 3)
    ↓
LSTM(128, return_sequences=True) + Dropout(0.3)
    ↓
LSTM(64, return_sequences=True) + Dropout(0.3)
    ↓
LSTM(32, return_sequences=False) + Dropout(0.2)
    ↓
Dense(32) + Dropout(0.2)
    ↓
Dense(16)
    ↓
Output (3: final_mqe, final_topo_error, final_dead_ratio)
```

**Total Parameters:** ~150K

### Bidirectional LSTM
```
Input (sequence_length, 3)
    ↓
Bidirectional LSTM(64) + Dropout(0.3)
    ↓
Bidirectional LSTM(32) + Dropout(0.2)
    ↓
Dense(32) + Dropout(0.2)
    ↓
Output (3)
```

**Total Parameters:** ~80K
**Advantage:** Processes sequences forward and backward for better context

### Lightweight LSTM
```
Input (sequence_length, 3)
    ↓
LSTM(64, return_sequences=True) + Dropout(0.2)
    ↓
LSTM(32, return_sequences=False)
    ↓
Dense(16)
    ↓
Output (3)
```

**Total Parameters:** ~40K

---

## Input Format

Time-series sequences of training metrics at checkpoints:

```json
{
  "epochs": [10, 50, 100, 200, 500, 1000],
  "mqe": [0.5, 0.3, 0.2, 0.15, 0.12, 0.10],
  "topographic_error": [0.1, 0.08, 0.06, 0.05, 0.04, 0.04],
  "dead_neuron_ratio": [0.15, 0.12, 0.10, 0.08, 0.06, 0.05]
}
```

**Shape:** `(sequence_length, 3)`
- Sequence length = number of checkpoints
- 3 features per checkpoint: mqe, topo_error, dead_ratio

---

## Output Targets

3 final quality metrics (what the training will achieve):

1. **final_mqe** - Predicted final MQE
2. **final_topographic_error** - Predicted final topology quality
3. **final_dead_neuron_ratio** - Predicted final dead neurons

---

## Files Structure

```
lstm/
├── README.md                   # This file
├── collect_training_data.py    # Data collection script
├── evaluate_model.py           # Model evaluation script
├── src/
│   ├── model.py               # Model architectures
│   └── train.py               # Training script
├── data/                      # Training data (generated)
│   ├── dataset.csv
│   └── dataset_metadata.json
├── models/                    # Trained models (generated)
│   ├── lstm_oracle_*_best.keras
│   ├── lstm_oracle_*_final.keras
│   └── lstm_oracle_*_metadata.json
└── logs/                      # Training logs (generated)
    └── lstm_oracle_*/
```

---

## Data Collection

### Current: Simulated Data (Proof-of-Concept)

The `collect_training_data.py` script simulates training histories based on final results.

**Simulation approach:**
- Start with high MQE, exponentially decay to final value
- Add noise to topographic error progression
- Model typical dead neuron evolution patterns

**Limitations:**
- Not real training dynamics
- May not capture all failure modes
- Good for testing, not production

### Production: Real Training Logs

To collect real data, modify SOM training code:

```python
# In som_trainer.py or equivalent
def train_som(data, config):
    som = SOM(config)

    # Checkpoints to log
    checkpoints = [10, 50, 100, 200, 500, 1000]
    training_history = {'epochs': [], 'mqe': [], 'topo_error': [], 'dead_ratio': []}

    for epoch in range(config['max_epochs']):
        som.train_epoch(data)

        if epoch in checkpoints:
            # Log intermediate metrics
            training_history['epochs'].append(epoch)
            training_history['mqe'].append(calculate_mqe(som))
            training_history['topo_error'].append(calculate_topo_error(som))
            training_history['dead_ratio'].append(calculate_dead_ratio(som))

    # Save training history to JSON
    with open(f'training_logs/{uid}_history.json', 'w') as f:
        json.dump(training_history, f)

    return som
```

---

## Training

The training script:

1. Loads dataset with training sequences
2. Parses JSON histories into numpy arrays
3. Splits into train/val/test (70/15/15)
4. Creates LSTM model
5. Trains with callbacks:
   - ModelCheckpoint (saves best)
   - EarlyStopping (patience=20)
   - ReduceLROnPlateau (patience=10)
   - TensorBoard logging
6. Evaluates on test sequences
7. Saves model and metadata

**Outputs:**
- `models/lstm_oracle_*_best.keras` - Best model
- `models/lstm_oracle_*_final.keras` - Final model
- `models/lstm_oracle_*_metadata.json` - Model info
- `logs/lstm_oracle_*/` - Training logs

---

## Evaluation

### Full Sequence Evaluation

Tests on complete training sequences:

```bash
python3 evaluate_model.py --model models/lstm_oracle_standard_*_best.keras --dataset data/dataset.csv
```

**Metrics:**
- MAE per target (mqe, topo_error, dead_ratio)
- RMSE per target
- Overall MAE/RMSE

### Early Prediction (Real Use Case)

Predicts final quality from partial sequence:

```bash
python3 evaluate_model.py --model models/lstm_oracle_standard_*_best.keras --sequence partial.json
```

**Example partial sequence (after 100 epochs):**
```json
{
  "epochs": [10, 50, 100],
  "mqe": [0.5, 0.3, 0.25],
  "topographic_error": [0.1, 0.09, 0.08],
  "dead_neuron_ratio": [0.2, 0.15, 0.12]
}
```

**Output:**
- Predicted final quality
- Early stopping recommendation

---

## Integration with SOM Training

### Use Case: Intelligent Early Stopping

```python
def train_som_with_oracle(data, config, oracle_model):
    som = SOM(config)
    training_history = {'epochs': [], 'mqe': [], 'topo_error': [], 'dead_ratio': []}

    checkpoints = [10, 50, 100, 200]

    for epoch in range(config['max_epochs']):
        som.train_epoch(data)

        if epoch in checkpoints:
            # Log metrics
            training_history['epochs'].append(epoch)
            training_history['mqe'].append(calculate_mqe(som))
            training_history['topo_error'].append(calculate_topo_error(som))
            training_history['dead_ratio'].append(calculate_dead_ratio(som))

            # After 100 epochs, check if we should stop early
            if epoch == 100:
                # Prepare sequence for Oracle
                sequence = np.array([
                    training_history['mqe'],
                    training_history['topo_error'],
                    training_history['dead_ratio']
                ]).T

                # Predict final quality
                prediction = oracle_model.predict(np.expand_dims(sequence, axis=0))[0]
                final_mqe, final_topo, final_dead = prediction

                # Quality score (lower is better)
                quality_score = final_mqe * 1.0 + final_topo * 1.0 + final_dead * 0.5

                if quality_score > 1.0:
                    print(f"Early stopping at epoch {epoch}: predicted quality {quality_score:.4f}")
                    return None  # Terminate training

    return som
```

### Benefits:
- **Save 50-90% training time** on bad configurations
- **More EA generations** in same time
- **Better resource utilization**

---

## Performance Expectations

**Expected Metrics:**
- MAE: 0.02 - 0.06 (depending on sequence length)
- RMSE: 0.03 - 0.10

**Good Performance Indicators:**
- MAE < 0.05 for MQE prediction after 100 epochs
- Early stopping accuracy > 80%

**Factors Affecting Accuracy:**
- **Sequence length**: More checkpoints = better predictions
- **Checkpoint placement**: Logarithmically spaced works well
- **Data diversity**: Train on varied configurations
- **Real vs simulated**: Real data will perform better

---

## Troubleshooting

### "Sequence length mismatch"
All sequences must have the same number of checkpoints.
Use padding or interpolation if sequences vary.

### "Poor early predictions"
- **Collect more early checkpoints**: Add checkpoints at epochs 5, 10, 20
- **Use bidirectional LSTM**: Captures patterns better
- **More training data**: Collect from multiple EA runs

### "Overfitting"
- **Increase dropout**: Try 0.4-0.5
- **Reduce model size**: Use lightweight version
- **Early stopping**: Reduce patience
- **More data**: Collect diverse training sequences

---

## Next Steps

1. **Collect Real Data**: Modify SOM training to log checkpoints
2. **Optimize Checkpoints**: Find minimal set for accurate predictions
3. **Multi-Step Prediction**: Predict quality at multiple future points
4. **Attention Mechanism**: Focus on critical training phases
5. **Integration**: Deploy in EA for real-time early stopping

---

## Production Deployment

### Step 1: Modify SOM Training

Add checkpoint logging to your SOM trainer (see Data Collection section).

### Step 2: Collect Training Data

Run EA experiments and collect training histories:

```bash
python3 lstm/collect_training_data.py --results_dir ./test/results/*
```

### Step 3: Train Oracle

```bash
cd lstm
python3 src/train.py --dataset data/real_training_data.csv --epochs 200 --model bidirectional
```

### Step 4: Deploy

Integrate trained model into EA or SOM training pipeline.

---

## References

- [LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Time Series Prediction with LSTM](https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/)
- [Early Stopping Techniques](https://arxiv.org/abs/1703.09580)
