# The Brain (LSTM/RNN) - Dynamic SOM Control Requirements

**Document Version**: 1.0
**Last Updated**: 2026-01-11
**Component**: LSTM/RNN - "The Brain"
**Purpose**: Real-time dynamic control of SOM training parameters

---

## Executive Summary

**Current Implementation Status**: Phase 1 is **NOT STARTED** - SOM architecture prepared, controller not implemented.

### What's Implemented ✅

**SOM Infrastructure** (ready for controller integration):
- ✅ Baseline parameter calculation separated from application
- ✅ Complete training history tracking (LR, radius, batch size, MQE)
- ✅ Modular `get_decay_value()` method
- ✅ Training state accessible each iteration

**Architecture Readiness**:
- ✅ SOM can provide state dictionary for controller
- ✅ History tracking generates time-series data
- ✅ Parameter update mechanism supports factor-based modification

### What's Missing ❌

**Controller Implementation**:
- ❌ LSTM/RNN model architecture
- ❌ Controller interface definition
- ❌ Training data generation from EA runs
- ❌ Reinforcement learning or imitation learning pipeline
- ❌ State representation design
- ❌ Action space definition
- ❌ Reward function formulation

**Training Infrastructure**:
- ❌ Time-series dataset preparation
- ❌ Training/validation split strategy
- ❌ Model training pipeline
- ❌ Evaluation metrics

**Integration**:
- ❌ NNController class
- ❌ SOM integration (controller parameter to train())
- ❌ Fallback mechanism for controller failures

### Phase Roadmap

- **Phase 1**: Data preparation from EA history ❌ (0% complete)
- **Phase 2**: LSTM model training ❌ (0% complete)
- **Phase 3**: SOM integration ❌ (0% complete)

---

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Data Preparation](#phase-1-data-preparation)
3. [Phase 2: LSTM Model Development](#phase-2-lstm-model-development)
4. [Phase 3: Integration with SOM](#phase-3-integration-with-som)
5. [Requirements Traceability Matrix](#requirements-traceability-matrix)
6. [Document Information](#document-information)

---

## Overview

**The Brain (LSTM/RNN)** is a recurrent neural network designed to dynamically control SOM training parameters in real-time, replacing static schedules with adaptive decision-making.

**Control Objectives**:
1. **Learning Rate Adjustment**: Modify LR based on training progress
2. **Radius Adjustment**: Adapt neighborhood size dynamically
3. **Batch Size Control**: Adjust sampling strategy (hybrid mode)
4. **Early Stopping**: Intelligent convergence detection

**Training Paradigm**: Reinforcement Learning or Imitation Learning from successful EA runs

**Integration Point**: Provides real-time parameter factors to SOM during training loop

---

## Phase 1: Data Preparation

### 1.1 Training Data Generation

#### FR-LSTM-1.1.1: EA History Extraction ❌

**Requirement**: Extract complete training histories from EA runs to create time-series dataset.

**Status**: NOT IMPLEMENTED

**Data Source**: EA individual directories with training results

**History Data Structure** (from SOM):
```python
# From training_results['history']
history = {
    'learning_rate': [(iteration, lr), ...],
    'radius': [(iteration, radius), ...],
    'batch_size': [(iteration, batch_size), ...],
    'mqe': [(iteration, mqe), ...]
}
```

**Extraction Script**:
```python
# app/lstm/prepare_training_data.py

def extract_ea_histories(ea_run_dir: str) -> pd.DataFrame:
    """
    Extract training histories from all EA individuals.

    Returns:
        DataFrame with columns:
        - uid: Individual identifier
        - iteration: Training iteration
        - learning_rate, radius, batch_size, mqe: Current values
        - final_mqe: Final quality (for labeling)
        - final_te, final_dead_ratio: Final metrics
    """
    results_csv = pd.read_csv(os.path.join(ea_run_dir, 'results.csv'))

    all_histories = []

    for uid in results_csv['uid']:
        individual_dir = os.path.join(ea_run_dir, 'individuals', uid)

        # Load history (would need to be saved by SOM)
        history_file = os.path.join(individual_dir, 'training_history.json')

        if os.path.exists(history_file):
            with open(history_file, 'r') as f:
                history = json.load(f)

            # Convert to time-series DataFrame
            for i, (iter_num, lr) in enumerate(history['learning_rate']):
                all_histories.append({
                    'uid': uid,
                    'iteration': iter_num,
                    'learning_rate': lr,
                    'radius': history['radius'][i][1],
                    'batch_size': history['batch_size'][i][1] if 'batch_size' in history else None,
                    'mqe': history['mqe'][i][1] if i < len(history['mqe']) else None,
                    'final_mqe': results_csv.loc[results_csv['uid'] == uid, 'best_mqe'].values[0],
                    'final_te': results_csv.loc[results_csv['uid'] == uid, 'topographic_error'].values[0],
                    'final_dead_ratio': results_csv.loc[results_csv['uid'] == uid, 'dead_neuron_ratio'].values[0]
                })

    return pd.DataFrame(all_histories)
```

**Acceptance Criteria**:
- [ ] Extract histories from all EA individuals
- [ ] Align time-series data (different training lengths)
- [ ] Link to final quality metrics
- [ ] Handle missing/incomplete histories

**Note**: Requires SOM to save `training_history.json` (currently only in `training_results` dict)

---

#### FR-LSTM-1.1.2: Time-Series Windowing ❌

**Requirement**: Convert continuous histories into fixed-length sequential windows for LSTM input.

**Status**: NOT IMPLEMENTED

**Windowing Strategy**:
```python
def create_time_windows(history_df: pd.DataFrame, window_size: int = 10, stride: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from training history.

    Args:
        history_df: Time-series DataFrame from extract_ea_histories()
        window_size: Number of timesteps per window
        stride: Step size for sliding window

    Returns:
        X: Input sequences (num_windows, window_size, num_features)
        y: Target values (num_windows, num_targets)
    """
    # Feature columns
    feature_cols = ['learning_rate', 'radius', 'batch_size', 'mqe']

    # Group by UID (each training run)
    sequences_X = []
    sequences_y = []

    for uid, group in history_df.groupby('uid'):
        # Sort by iteration
        group = group.sort_values('iteration')

        # Extract features
        features = group[feature_cols].values  # Shape: (num_iterations, num_features)
        final_quality = group['final_mqe'].iloc[0]

        # Create sliding windows
        for i in range(0, len(features) - window_size, stride):
            window = features[i:i+window_size]

            # Input: historical window
            sequences_X.append(window)

            # Target: improvement in next window (or final quality)
            next_mqe = group.iloc[i+window_size]['mqe'] if i+window_size < len(group) else final_quality
            current_mqe = group.iloc[i+window_size-1]['mqe']

            improvement = current_mqe - next_mqe  # Positive = improvement

            sequences_y.append([improvement, final_quality])

    return np.array(sequences_X), np.array(sequences_y)
```

**Acceptance Criteria**:
- [ ] Fixed-length windows (e.g., 10-50 timesteps)
- [ ] Sliding window with configurable stride
- [ ] Handle variable-length training runs
- [ ] Normalize features to [0, 1] range

---

#### FR-LSTM-1.1.3: State Representation ❌

**Requirement**: Define comprehensive state representation for LSTM input.

**Status**: NOT IMPLEMENTED

**State Features**:

**Historical Metrics** (time-series, window_size timesteps):
- `mqe_history`: Last N MQE values
- `learning_rate_history`: Last N LR values
- `radius_history`: Last N radius values
- `batch_size_history`: Last N batch sizes (hybrid mode)

**Derived Features** (per timestep):
- `mqe_improvement`: Current - Previous MQE (delta)
- `lr_decay_rate`: Current / Previous LR (ratio)
- `radius_decay_rate`: Current / Previous radius

**Context Features** (constant per training run):
- `total_iterations`: Total training length
- `num_samples`: Dataset size
- `map_size`: SOM dimensions (flattened)
- `processing_type`: One-hot encoded (stochastic/deterministic/hybrid)

**State Tensor Shape**:
```
(batch, window_size, num_time_features + num_context_features)
```

**Acceptance Criteria**:
- [ ] Time-series features (MQE, LR, radius, batch size)
- [ ] Derived features (deltas, ratios)
- [ ] Context features (dataset metadata)
- [ ] Feature normalization

---

#### FR-LSTM-1.1.4: Label Strategy ❌

**Requirement**: Define target labels for supervised or imitation learning.

**Status**: NOT IMPLEMENTED

**Labeling Options**:

**Option 1: Imitation Learning** (Recommended for initial version)
- **Target**: Actual parameter adjustments from successful EA runs
- **Filter**: Only use training runs from top 10% by final quality
- **Labels**: `[lr_factor, radius_factor, batch_factor, stop_decision]`

**Option 2: Reinforcement Learning**
- **Reward**: Improvement in MQE at next timestep
- **Terminal Reward**: Final quality score
- **Labels**: Computed via RL algorithm (Q-learning, PPO)

**Imitation Learning Label Extraction**:
```python
def extract_imitation_labels(history_df: pd.DataFrame, top_percentile: float = 0.1) -> np.ndarray:
    """
    Extract parameter adjustment actions from best EA runs.

    Returns:
        labels: (num_windows, num_actions)
        actions: [lr_factor, radius_factor, batch_factor, stop]
    """
    # Filter to top performers
    quality_threshold = history_df['final_mqe'].quantile(top_percentile)
    best_runs = history_df[history_df['final_mqe'] <= quality_threshold]

    labels = []

    for uid, group in best_runs.groupby('uid'):
        group = group.sort_values('iteration')

        for i in range(len(group) - 1):
            current_iter = group.iloc[i]
            next_iter = group.iloc[i+1]

            # Compute factors (actual / baseline)
            baseline_lr = get_baseline_lr(current_iter['iteration'], ...)  # From static schedule
            actual_lr = next_iter['learning_rate']
            lr_factor = actual_lr / baseline_lr

            # Similarly for radius and batch size
            radius_factor = compute_radius_factor(...)
            batch_factor = compute_batch_factor(...)

            # Stop decision: 1 if this is final iteration, 0 otherwise
            stop = 1 if i == len(group) - 2 else 0

            labels.append([lr_factor, radius_factor, batch_factor, stop])

    return np.array(labels)
```

**Acceptance Criteria**:
- [ ] Filter to successful training runs
- [ ] Compute adjustment factors (not absolute values)
- [ ] Handle early stopping decisions
- [ ] Validate label ranges (factors typically 0.5-2.0)

---

### 1.2 Dataset Preparation

#### FR-LSTM-1.2.1: Training/Validation Split ❌

**Requirement**: Split time-series data into training and validation sets.

**Status**: NOT IMPLEMENTED

**Splitting Strategy**:
- **Training**: 80% of EA runs (entire time-series per run)
- **Validation**: 20% of EA runs
- **Test**: Evaluate on live SOM training (not pre-split)

**Important**: Split by UID (entire training runs), NOT by timesteps

**Acceptance Criteria**:
- [ ] Stratified split by final quality (ensure diversity)
- [ ] No data leakage between train/val
- [ ] Preserve temporal order within runs

---

#### FR-LSTM-1.2.2: Data Normalization ❌

**Requirement**: Normalize all features to similar scales for LSTM training.

**Status**: NOT IMPLEMENTED

**Normalization Strategy**:
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Option 1: Standardization (mean=0, std=1)
scaler = StandardScaler()
features_normalized = scaler.fit_transform(features)

# Option 2: Min-Max scaling (range [0, 1])
scaler = MinMaxScaler()
features_normalized = scaler.fit_transform(features)
```

**Features Requiring Normalization**:
- `mqe`: Scale by dataset max MQE
- `learning_rate`: Scale to [0, 1]
- `radius`: Scale to [0, 1]
- `batch_size`: Scale to [0, 1]

**Acceptance Criteria**:
- [ ] Fit scaler on training set only
- [ ] Apply same scaler to validation set
- [ ] Save scaler for inference
- [ ] Inverse transform for interpretability

---

## Phase 2: LSTM Model Development

### 2.1 Model Architecture

#### FR-LSTM-2.1.1: LSTM Architecture ❌

**Requirement**: Define LSTM/GRU architecture for sequential decision-making.

**Status**: NOT IMPLEMENTED

**Architecture Options**:

**Option 1: Vanilla LSTM**
```python
import torch.nn as nn

class BrainLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_actions=4):
        """
        LSTM controller for SOM parameter adjustment.

        Args:
            input_size: Number of features per timestep
            hidden_size: LSTM hidden dimension
            num_layers: Number of stacked LSTM layers
            num_actions: Output dimension (lr_factor, radius_factor, batch_factor, stop)
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Output heads
        self.fc_factors = nn.Linear(hidden_size, 3)  # LR, radius, batch factors
        self.fc_stop = nn.Linear(hidden_size, 1)  # Stop decision (sigmoid)

    def forward(self, x, hidden=None):
        """
        Args:
            x: (batch, window_size, input_size)
            hidden: Optional previous hidden state

        Returns:
            factors: (batch, 3) - adjustment factors
            stop_prob: (batch, 1) - probability of stopping
            hidden: Updated hidden state
        """
        lstm_out, hidden = self.lstm(x, hidden)

        # Use last timestep output
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)

        # Predict adjustment factors (typically 0.5-2.0 range)
        factors = torch.sigmoid(self.fc_factors(last_output)) * 1.5 + 0.5  # Range: [0.5, 2.0]

        # Predict stop probability
        stop_prob = torch.sigmoid(self.fc_stop(last_output))

        return factors, stop_prob, hidden
```

**Option 2: GRU** (lighter, faster)
```python
class BrainGRU(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_actions=4):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        self.fc_factors = nn.Linear(hidden_size, 3)
        self.fc_stop = nn.Linear(hidden_size, 1)

    def forward(self, x, hidden=None):
        gru_out, hidden = self.gru(x, hidden)
        last_output = gru_out[:, -1, :]

        factors = torch.sigmoid(self.fc_factors(last_output)) * 1.5 + 0.5
        stop_prob = torch.sigmoid(self.fc_stop(last_output))

        return factors, stop_prob, hidden
```

**Acceptance Criteria**:
- [ ] Support both LSTM and GRU
- [ ] Configurable hidden size and layer count
- [ ] Dual output heads (factors + stop decision)
- [ ] Hidden state support for stateful training

---

#### FR-LSTM-2.1.2: Model Configuration ❌

**Requirement**: Flexible model configuration via JSON/YAML.

**Status**: NOT IMPLEMENTED

**Configuration Structure**:
```json
{
  "model": {
    "type": "lstm",
    "input_size": 16,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "num_actions": 4
  },
  "training": {
    "batch_size": 64,
    "epochs": 100,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "scheduler": "reduce_on_plateau",
    "early_stopping_patience": 15
  },
  "data": {
    "window_size": 20,
    "stride": 5,
    "top_percentile": 0.1
  }
}
```

**Acceptance Criteria**:
- [ ] Load from JSON/YAML
- [ ] Command-line overrides
- [ ] Validation and defaults

---

### 2.2 Training Pipeline

#### FR-LSTM-2.2.1: Imitation Learning Training ❌

**Requirement**: Train LSTM to mimic successful EA parameter schedules.

**Status**: NOT IMPLEMENTED

**Training Loop**:
```python
# app/lstm/train.py

class LSTMTrainer:
    def __init__(self, config: dict):
        self.config = config
        self.model = build_lstm_model(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, train_loader, val_loader):
        """Main training loop."""
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.config['epochs']):
            train_loss = self._train_epoch(train_loader)
            val_loss, val_metrics = self._validate(val_loader)

            print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

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

    def _train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            sequences, targets = batch  # sequences: (batch, window, features), targets: (batch, 4)
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            factors, stop_prob, _ = self.model(sequences)
            predicted_actions = torch.cat([factors, stop_prob], dim=1)  # (batch, 4)

            # Loss: MSE for factor regression + BCE for stop decision
            factor_loss = F.mse_loss(predicted_actions[:, :3], targets[:, :3])
            stop_loss = F.binary_cross_entropy(predicted_actions[:, 3], targets[:, 3])

            loss = factor_loss + stop_loss

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # Gradient clipping
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(train_loader)
```

**Acceptance Criteria**:
- [ ] MSE loss for factor prediction
- [ ] BCE loss for stop decision
- [ ] Gradient clipping for stability
- [ ] Checkpointing best model
- [ ] Early stopping

---

#### FR-LSTM-2.2.2: Reinforcement Learning Training (Future) 🔜

**Requirement**: Train LSTM using RL for direct quality optimization.

**Status**: FUTURE ENHANCEMENT

**RL Approach**:
- **Algorithm**: PPO (Proximal Policy Optimization) or DQN
- **Environment**: SOM training process
- **State**: Current training metrics
- **Action**: Parameter adjustment factors
- **Reward**: Improvement in MQE (or composite quality)

**Acceptance Criteria**:
- [ ] RL environment wrapper for SOM
- [ ] Reward function design
- [ ] Policy network training
- [ ] Exploration strategy

**Note**: Start with imitation learning, move to RL once baseline established

---

### 2.3 Evaluation

#### FR-LSTM-2.3.1: Offline Evaluation ❌

**Requirement**: Evaluate LSTM predictions on validation set.

**Status**: NOT IMPLEMENTED

**Metrics**:
- **Factor Prediction**: MAE, MSE for adjustment factors
- **Stop Decision**: Accuracy, F1 for early stopping
- **Sequence Prediction**: Correlation between predicted and actual trajectories

**Acceptance Criteria**:
- [ ] Evaluate on validation set
- [ ] Generate prediction plots (predicted vs actual)
- [ ] Compute correlation metrics

---

#### FR-LSTM-2.3.2: Online Evaluation (Integration Test) ❌

**Requirement**: Evaluate LSTM controller in live SOM training.

**Status**: NOT IMPLEMENTED

**Evaluation Protocol**:
1. Train SOM with LSTM controller on test datasets
2. Compare final quality vs static schedule baseline
3. Measure training efficiency (iterations to convergence)

**Metrics**:
- Final MQE comparison (LSTM vs baseline)
- Training duration
- Parameter stability (variance of adjustments)

**Acceptance Criteria**:
- [ ] Outperform static schedule on average
- [ ] Stable training (no divergence)
- [ ] Reasonable training duration

---

## Phase 3: Integration with SOM

### 3.1 Controller Interface

#### FR-LSTM-3.1.1: SOMController Interface Definition ❌

**Requirement**: Define formal interface for SOM controllers.

**Status**: NOT IMPLEMENTED (planned in SOM Phase 2)

**Interface**:
```python
# app/som/controller.py

from abc import ABC, abstractmethod

class SOMController(ABC):
    """Abstract base class for SOM parameter controllers."""

    @abstractmethod
    def get_lr_factor(self, state: dict) -> float:
        """
        Get learning rate adjustment factor.

        Args:
            state: Current training state (iteration, mqe, etc.)

        Returns:
            Factor to multiply baseline LR (0.5-2.0 range)
        """
        pass

    @abstractmethod
    def get_radius_factor(self, state: dict) -> float:
        """Get radius adjustment factor."""
        pass

    @abstractmethod
    def get_batch_size_factor(self, state: dict) -> float:
        """Get batch size adjustment factor (hybrid mode)."""
        pass

    @abstractmethod
    def should_stop(self, state: dict) -> bool:
        """Decide whether to stop training early."""
        pass

    def record_metrics(self, iteration: int, metrics: dict) -> None:
        """
        Record training metrics for stateful controllers.

        Optional method for controllers that maintain history.
        """
        pass
```

**Acceptance Criteria**:
- [ ] Abstract base class defined
- [ ] All essential methods specified
- [ ] State dictionary format documented

---

#### FR-LSTM-3.1.2: NNController Implementation ❌

**Requirement**: Implement controller using trained LSTM model.

**Status**: NOT IMPLEMENTED

**Implementation**:
```python
# app/lstm/nn_controller.py

import torch
from som.controller import SOMController

class NNController(SOMController):
    """Neural network-based SOM controller using trained LSTM."""

    def __init__(self, model_checkpoint: str, config: dict = None):
        """
        Initialize controller with trained LSTM.

        Args:
            model_checkpoint: Path to trained model checkpoint
            config: Optional configuration
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_checkpoint, config)
        self.model.eval()

        # State tracking
        self.window_size = config.get('window_size', 20)
        self.history = {
            'mqe': [],
            'learning_rate': [],
            'radius': [],
            'batch_size': []
        }
        self.hidden_state = None  # LSTM hidden state

    def get_lr_factor(self, state: dict) -> float:
        """Get learning rate adjustment factor from LSTM."""
        factors, _, self.hidden_state = self._predict(state)
        return factors[0].item()  # First factor = LR

    def get_radius_factor(self, state: dict) -> float:
        """Get radius adjustment factor from LSTM."""
        factors, _, _ = self._predict(state)
        return factors[1].item()  # Second factor = radius

    def get_batch_size_factor(self, state: dict) -> float:
        """Get batch size adjustment factor from LSTM."""
        factors, _, _ = self._predict(state)
        return factors[2].item()  # Third factor = batch size

    def should_stop(self, state: dict) -> bool:
        """Decide early stopping based on LSTM."""
        _, stop_prob, _ = self._predict(state)
        return stop_prob.item() > 0.5  # Stop if probability > 50%

    def record_metrics(self, iteration: int, metrics: dict):
        """Record metrics for window construction."""
        self.history['mqe'].append(metrics.get('mqe', 0))
        self.history['learning_rate'].append(metrics.get('learning_rate', 0))
        self.history['radius'].append(metrics.get('radius', 0))
        self.history['batch_size'].append(metrics.get('batch_size', 0))

        # Keep only last window_size values
        for key in self.history:
            if len(self.history[key]) > self.window_size:
                self.history[key] = self.history[key][-self.window_size:]

    def _predict(self, state: dict):
        """Internal prediction method."""
        # Construct input window from history
        if len(self.history['mqe']) < self.window_size:
            # Not enough history, return neutral factors
            return torch.tensor([1.0, 1.0, 1.0]), torch.tensor([0.0]), self.hidden_state

        # Build feature matrix
        features = np.array([
            self.history['mqe'][-self.window_size:],
            self.history['learning_rate'][-self.window_size:],
            self.history['radius'][-self.window_size:],
            self.history['batch_size'][-self.window_size:]
        ]).T  # Shape: (window_size, num_features)

        # Normalize
        features_norm = self.scaler.transform(features)

        # Convert to tensor
        input_tensor = torch.FloatTensor(features_norm).unsqueeze(0).to(self.device)  # (1, window, features)

        # Predict
        with torch.no_grad():
            factors, stop_prob, hidden = self.model(input_tensor, self.hidden_state)

        return factors.squeeze(0), stop_prob.squeeze(0), hidden

    def _load_model(self, checkpoint_path: str, config: dict):
        """Load trained LSTM model."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if config is None:
            config = checkpoint['config']

        model = build_lstm_model(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        self.scaler = checkpoint['scaler']  # Load saved scaler

        return model
```

**Acceptance Criteria**:
- [ ] Implements SOMController interface
- [ ] Loads trained LSTM model
- [ ] Maintains rolling window of metrics
- [ ] Stateful prediction with hidden state
- [ ] Fallback to neutral factors if insufficient history

---

#### FR-LSTM-3.1.3: SOM Integration ❌

**Requirement**: Integrate NNController into SOM training loop.

**Status**: NOT IMPLEMENTED (SOM Phase 2)

**Modified SOM Training**:
```python
# In som.py train() method

def train(self, data: np.ndarray, controller: SOMController = None,
          ignore_mask: np.ndarray = None, working_dir: str = '.') -> dict:
    """
    Train SOM with optional controller.

    Args:
        data: Training data
        controller: Optional SOMController for dynamic parameter adjustment
        ignore_mask: Feature mask
        working_dir: Output directory
    """
    # ... existing setup ...

    for iteration in tqdm(range(total_iterations)):
        # Baseline parameters (static schedule)
        baseline_lr = self.get_decay_value(iteration, total_iterations,
                                           self.start_learning_rate, self.end_learning_rate,
                                           self.lr_decay_type)

        baseline_radius = self.get_decay_value(iteration, total_iterations,
                                               self.start_radius, self.end_radius,
                                               self.radius_decay_type)

        # Controller adjustment
        if controller is not None:
            state = {
                'iteration': iteration,
                'total_iterations': total_iterations,
                'mqe': self.mqe_history[-1] if self.mqe_history else None,
                'learning_rate': baseline_lr,
                'radius': baseline_radius
            }

            try:
                lr_factor = controller.get_lr_factor(state)
                radius_factor = controller.get_radius_factor(state)

                current_lr = baseline_lr * lr_factor
                current_radius = baseline_radius * radius_factor

                # Check for early stopping
                if controller.should_stop(state):
                    print(f"Controller requested early stop at iteration {iteration}")
                    break

            except Exception as e:
                # Fallback to baseline on error
                print(f"Controller error: {e}, using baseline parameters")
                current_lr = baseline_lr
                current_radius = baseline_radius
        else:
            current_lr = baseline_lr
            current_radius = baseline_radius

        # ... rest of training loop ...

        # Record metrics for controller
        if controller is not None and iteration % self.mqe_evaluations_per_run == 0:
            current_mqe = self.compute_quantization_error(data, ignore_mask)
            controller.record_metrics(iteration, {
                'mqe': current_mqe,
                'learning_rate': current_lr,
                'radius': current_radius,
                'batch_size': current_batch_size
            })
```

**Acceptance Criteria**:
- [ ] Controller parameter added to train()
- [ ] Baseline parameters computed first
- [ ] Controller factors applied to baseline
- [ ] Error handling with fallback
- [ ] Backward compatibility (controller=None)

---

## Requirements Traceability Matrix

| Requirement ID | Description | Status | Implementation | Verified |
|---------------|-------------|--------|----------------|----------|
| **FR-LSTM-1.1.1** | EA history extraction | ❌ | Not implemented | - |
| **FR-LSTM-1.1.2** | Time-series windowing | ❌ | Not implemented | - |
| **FR-LSTM-1.1.3** | State representation | ❌ | Not implemented | - |
| **FR-LSTM-1.1.4** | Label strategy | ❌ | Not implemented | - |
| **FR-LSTM-1.2.1** | Train/val split | ❌ | Not implemented | - |
| **FR-LSTM-1.2.2** | Data normalization | ❌ | Not implemented | - |
| **FR-LSTM-2.1.1** | LSTM architecture | ❌ | Not implemented | - |
| **FR-LSTM-2.1.2** | Model configuration | ❌ | Not implemented | - |
| **FR-LSTM-2.2.1** | Imitation learning | ❌ | Not implemented | - |
| **FR-LSTM-2.2.2** | RL training | 🔜 | Future | - |
| **FR-LSTM-2.3.1** | Offline evaluation | ❌ | Not implemented | - |
| **FR-LSTM-2.3.2** | Online evaluation | ❌ | Not implemented | - |
| **FR-LSTM-3.1.1** | Controller interface | ❌ | Planned (SOM Phase 2) | - |
| **FR-LSTM-3.1.2** | NNController class | ❌ | Not implemented | - |
| **FR-LSTM-3.1.3** | SOM integration | ❌ | Planned (SOM Phase 2) | - |

**Summary**: 0/15 requirements implemented (0%)

---

## Document Information

### Version History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-01-11 | Initial LSTM requirements document | Claude Sonnet 4.5 |

### Implementation Summary

**Phase 1 (Data Preparation)**: 0% complete
- ❌ EA history extraction not implemented
- ❌ Time-series dataset not created
- ⚠️ **Blocker**: SOM doesn't save training history to JSON yet

**Phase 2 (LSTM Model)**: 0% complete
- ❌ Model architecture not defined
- ❌ Training pipeline not implemented

**Phase 3 (Integration)**: 0% complete
- ❌ Controller interface not defined
- ❌ NNController not implemented
- ❌ SOM integration not complete

**Critical Dependency**: Requires SOM to save `training_history.json` in individual directories

**Next Immediate Steps**:
1. **Modify SOM**: Save training history to JSON file
2. **Run large EA campaign**: Generate diverse training data
3. **Extract EA histories**: Implement prepare_training_data.py
4. **Create time-series dataset**: Windowing and labeling
5. **Define LSTM architecture**: Start with simple 2-layer LSTM
6. **Train baseline model**: Imitation learning on top 10% runs
7. **Implement NNController**: Wrap trained model
8. **Test integration**: Evaluate on test datasets

---

**End of LSTM Requirements Specification**
