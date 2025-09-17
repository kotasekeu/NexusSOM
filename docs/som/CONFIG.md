# SOM Algorithm Configuration Guide

This document outlines the configuration for a single run of the Hybrid Kohonen Self-Organizing Map (SOM). All settings are managed through a single, flat `config.json` file.

## Usage

This configuration file is a required argument when executing the main SOM script. For detailed instructions on script execution and command-line syntax, please refer to the `RUN.md` file.

### Configuration Philosophy

The configuration file uses a flat, single-level structure for simplicity and clarity. Each key represents a specific hyperparameter or setting for the SOM training, analysis, or preprocessing pipeline. This design ensures that a single file contains the complete, reproducible definition of one experiment.

---

## `config.json` File Structure

The configuration file is a single JSON object. It is logically divided into three groups of parameters, even though they are all at the same level:

1.  **Data Preprocessing Parameters**: Settings that control how the input CSV data is cleaned, analyzed, and prepared.
2.  **SOM Core Parameters**: Hyperparameters that directly define the architecture and training process of the Self-Organizing Map.
3.  **Analysis Parameters**: Settings for the post-training analysis, such as outlier detection.

---

## Parameter Reference

### 1. Data Preprocessing Parameters (`PREPROCES_DATA`)

These settings are used by the `preprocess.py` module.

*   `"primary_id"`: (String) The name of the column to be treated as a unique identifier. This column will be present in the data for analysis but ignored during SOM training.
*   `"delimiter"`: (String) The delimiter used in the input CSV file (e.g., `","`).
*   `"categorical_threshold_numeric"`: (Integer) The maximum number of unique values for a numeric column to be classified as categorical.
*   `"categorical_threshold_text"`: (Integer) The maximum number of unique values for a text column to be classified as categorical.
*   `"noise_threshold_ratio"`: (Float, 0-1) The ratio of unique values to total rows above which a text column is considered "noise" and completely ignored during processing.

### 2. SOM Core Parameters

These hyperparameters directly influence the SOM's training behavior.

**Core Algorithm Settings**
*   `"processing_type"`: (String) The training mode.
    *   `"stochastic"`: Updates weights after each single, randomly chosen data sample. Fast but can be unstable.
    *   `"deterministic"`: Processes the entire dataset in each iteration. Slow but stable.
    *   `"hybrid"`: Starts with small batches of data and gradually increases the batch size. A balanced approach.
*   `"epoch_multiplier"`: (Float) A multiplier that determines the total number of training iterations, calculated as `number_of_samples * epoch_multiplier`.

**Map Structure**
*   `"m"`: (Integer) The height of the SOM grid.
*   `"n"`: (Integer) The width of the SOM grid.
*   `"map_type"`: (String) The topology of the SOM grid. Supported values are `"square"` or `"hex"`.

**Learning Rate Parameters**
*   `"start_learning_rate"`: (Float, 0-1) The initial magnitude of weight updates at the beginning of training.
*   `"end_learning_rate"`: (Float, 0-1) The final learning rate at the end of training.
*   `"lr_decay_type"`: (String) The function defining how the learning rate decreases over time (e.g., `"linear-drop"`, `"exp-drop"`, `"log-drop"`, `"step-down"`).

**Neighborhood Radius Parameters**
*   `"start_radius"`: (Float) The initial neighborhood radius, defining the initial size of the area where neurons are updated.
*   `"end_radius"`: (Float) The final radius at the end of training. A value of `1.0` or `0.5` is typical.
*   `"radius_decay_type"`: (String) The decay function for the radius.

**Hybrid Mode Parameters (only used if `processing_type` is `"hybrid"`)**
*   `"start_batch_percent"`: (Float) The initial batch size as a percentage of the total dataset.
*   `"end_batch_percent"`: (Float) The final batch size as a percentage.
*   `"batch_growth_type"`: (String) The function governing how the batch size increases (`"linear-growth"`, `"exp-growth"`, `"log-growth"`).
*   `"num_batches"`: (Integer) The number of sections the dataset is split into for hybrid processing.

**General & Advanced Parameters**
*   `"normalize_weights_flag"`: (Boolean) If `true`, neuron weight vectors are L2-normalized after each update.
*   `"growth_g"`: (Float) The `G` parameter controlling the steepness of `exp-` and `log-` decay/growth curves. A higher value means a steeper curve.
*   `"random_seed"`: (Integer or `null`) The seed for the random number generator to ensure reproducibility.
*   `"mqe_evaluations_per_run"`: (Integer) How many times the Mean Quantization Error (MQE) should be evaluated during the training run. This controls the frequency of checks for early stopping.
*   `"early_stopping_window"`: (Integer) The number of recent MQE evaluations to average for the moving average early stopping mechanism. A value of `1` effectively disables moving average.
*   `"max_epochs_without_improvement"`: (Integer) The number of checks without improvement in the moving average before stopping training. This acts as the "patience" parameter. Set to a very high number to disable early stopping.

### 3. Analysis Parameters

Settings for the post-training analysis phase.

*   `"std_threshold"`: (Float) The number of standard deviations from the mean used to classify a value as an outlier during extreme value detection. A typical value is `2.5` or `3.0`.

---

## Complete `config.json` Example

```json
{
    "primary_id": "primary_id",
    "delimiter": ",",
    "categorical_threshold_numeric": 30,
    "categorical_threshold_text": 30,
    "noise_threshold_ratio": 0.2,

    "processing_type": "hybrid",
    "epoch_multiplier": 10.0,

    "m": 15,
    "n": 15,
    "map_type": "hex",

    "start_learning_rate": 0.8,
    "end_learning_rate": 0.1,
    "lr_decay_type": "linear-drop",

    "start_radius": 7.5,
    "end_radius": 1.0,
    "radius_decay_type": "linear-drop",

    "start_batch_percent": 5.0,
    "end_batch_percent": 100.0,
    "batch_growth_type": "exp-growth",
    "num_batches": 5,

    "normalize_weights_flag": false,
    "growth_g": 5.0,
    "random_seed": 42,
    
    "mqe_evaluations_per_run": 50,
    "early_stopping_window": 5,
    "max_epochs_without_improvement": 10,

    "std_threshold": 2.5
}
```