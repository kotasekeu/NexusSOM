# Evolutionary Algorithm Configuration Guide

This document outlines the configuration of the evolutionary algorithm used for optimizing the parameters of a Modified Kohonen Self-Organizing Map (SOM). All settings are managed through the `config.json` file.


## Usage
This configuration file is utilized by passing its path as a command-line argument to the evolutionary algorithm script. For detailed instructions on script execution, please refer to the `RUN.md` file.

### Configuration Design Philosophy

The configuration is structured into distinct sections (EA_SETTINGS, SEARCH_SPACE, FIXED_PARAMS, etc.) to provide a clear and robust definition of the experimental setup. This explicit separation ensures that parameters intended for optimization are distinctly identified from static run-time constants.

## Core Concept: Search Space vs. Fixed Parameters
The `config.json` file defines both the fixed parameters and the search space for hyperparameters. The logic is simple:

*   **Array of Values** in SEARCH_SPACE: If a parameter is assigned an array of values, it defines the search space for that hyperparameter. The EA will explore combinations of these values.
*   **Single Value** in FIXED_PARAMS: If a parameter is assigned a single value, it is treated as a fixed constant for the entire run.

#### Configuration Examples:

**NUMERICAL VALUE**
```json
"start_learning_rate": 0.5
```
> The algorithm will use a fixed `start_learning_rate` of 0.5.

```json
"start_learning_rate": [0.1, 0.5, 0.9]
```
> The evolutionary algorithm will explore these three distinct values for the `start_learning_rate`.

**STRING VALUE**
```json
"lr_decay_type": "linear-drop"
```
> Only the linear decay function will be used.

```json
"lr_decay_type": ["linear-drop", "exp-drop"]
```
> The algorithm will test both decay function types.

**DIMENSIONAL VALUE (e.g., Map Size)**
```json
"map_size": [20, 20]
```
> A single, fixed map size of 20x20 will be used.

```json
"map_size": [[10, 10], [20, 20]]
```
> The evolution will explore two different map sizes.

---

## `config.json` File Structure

The configuration file is organized into five primary sections:

1.  **`EA_SETTINGS`**: Controls the evolutionary process itself (e.g., population size, number of generations).
2. **`DATA_PARAMS`**: (Optional) Parameters for auto-generated datasets. Ignored if a real dataset is provided via the command line.
3. **`PREPROCES_DATA`**: Defines fixed parameters for the data preprocessing step.
4. **`SEARCH_SPACE`**: The core section defining which SOM parameters and their respective ranges the algorithm should optimize.
5.  **`FIXED_PARAMS`**: Defines SOM constants that remain unchanged throughout the evolution.

---

## Parameter Reference

### 1. `EA_SETTINGS`

Parameters that govern the behavior of the evolutionary algorithm.

*   `"population_size"`: (Integer) The number of individuals (configurations) in each generation.
*   `"generations"`: (Integer) The total number of generations the evolution will run.

### 2. `DATA_PARAMS`

Used only for synthetic data generation if no input file is specified.

*   `"sample_size"`: (Integer) Number of data points to generate.
*   `"input_dim"`: (Integer) Number of features (dimensions) for the generated data.

### 3. `PREPROCES_DATA`

Fixed settings for the data preprocessing pipeline.

*   `"primary_id"`: (String) The name of the column to be treated as a unique identifier. This column will be ignored during training via the ignore mask.
*   `"delimiter"`: (String) The delimiter used in the input CSV file (e.g., `","`).
*   `"categorical_threshold_numeric"` / `"categorical_threshold_text"`: (Integer) The maximum number of unique values for a numeric/text column to be considered categorical.
*   `"noise_threshold_ratio"`: (Float, 0-1) The ratio of unique values to total rows above which a text column is considered noise and ignored.

### 4. `SEARCH_SPACE` & `FIXED_PARAMS`

These two sections define the hyperparameter space for the SOM. Parameters can be placed in either section.

**Core Algorithm Parameters**
*   `"processing_type"`: (String) The training mode. Can be `"stochastic"`, `"deterministic"`, or `"hybrid"`.
*   `"epoch_multiplier"`: (Float) A multiplier that determines the total number of training iterations, calculated as `number_of_samples * epoch_multiplier`.

**Map Structure**
*   `"map_size"`: (Array of Arrays) Defines the map dimensions `[height, width]` to be tested.
*   `"map_type"`: (String) The topology of the SOM grid (`"square"` or `"hex"`).

**Learning Rate Parameters**
*   `"start_learning_rate"`: (Float, 0-1) The initial magnitude of weight updates.
*   `"end_learning_rate"`: (Float, 0-1) The final learning rate at the end of training.
*   `"lr_decay_type"`: (String) The decay function for the learning rate (e.g., `"linear-drop"`, `"exp-drop"`, `"log-drop"`, `"step-down"`).

**Neighborhood Radius Parameters**
*   `"start_radius_init_ratio"`: (Float) Initial neighborhood radius as a ratio of the map's largest dimension.
*   `"end_radius"`: (Float) The final radius at the end of training.
*   `"radius_decay_type"`: (String) The decay function for the radius.

**Hybrid Mode Parameters (ignored for other modes)**
*   `"start_batch_percent"`: (Float) The initial batch size as a percentage of the total dataset.
*   `"end_batch_percent"`: (Float) The final batch size as a percentage.
*   `"batch_growth_type"`: (String) The function governing how the batch size increases (`"linear-growth"`, `"exp-growth"`, `"log-growth"`).
*   `"num_batches"`: (Integer) The number of sections the dataset is split into for hybrid processing.

**General & Advanced Parameters**
*   `"normalize_weights_flag"`: (Boolean) If `true`, neuron weight vectors are normalized after each update.
*   `"growth_g"`: (Float) The `G` parameter controlling the steepness of `exp-` and `log-` decay/growth curves.
*   `"random_seed"`: (Integer or `null`) The seed for the random number generator.
*   `"mqe_evaluations_per_run"`: (Integer) How many times the MQE should be evaluated during a single training run.
*   `"early_stopping_window"`: (Integer) The number of recent MQE evaluations to average for the moving average early stopping.
*   `"max_epochs_without_improvement"`: (Integer) The number of checks without improvement in the moving average before stopping training (patience).

---

## Complete `config.json` Example

```json
{
  "EA_SETTINGS": {
    "population_size": 20,
    "generations": 30
  },
  "SEARCH_SPACE": {
    "map_size": [
      [8, 8],
      [10, 10],
      [15, 15]
    ],
    "processing_type": ["stochastic", "deterministic", "hybrid"],
    
    "start_learning_rate": [0.9, 0.8, 0.7, 0.6, 0.5],
    "end_learning_rate": [0.2, 0.1, 0.05, 0.01],
    "lr_decay_type": ["linear-drop", "exp-drop", "log-drop", "step-down"],

    "start_radius_init_ratio": [1.0, 0.75, 0.5, 0.25, 0.1],  
    "radius_decay_type": ["linear-drop", "exp-drop", "log-drop", "step-down"],

    "start_batch_percent": [0.025, 0.5, 1.0, 5.0, 10.0],
    "end_batch_percent": [3.0, 5.0, 7.5, 10.0, 15.0],
    "batch_growth_type": ["linear-growth", "exp-growth", "log-growth"],

    "epoch_multiplier": [5.0, 10.0 , 15.0],
    "normalize_weights_flag": [false, true],
    "growth_g": [1.0, 5.0, 15.0, 25.0, 35.0],
       
    "num_batches": [1, 3, 5, 10, 20],
    "map_type": ["hex","square"]
  },
  "FIXED_PARAMS": {
    "end_radius": 1.0,
    "random_seed": 42,
    "mqe_evaluations_per_run": 500,
    "max_epochs_without_improvement": 50,
    "early_stopping_window": 5
  },
  "PREPROCES_DATA": {
    "delimiter": ",",
    "categorical_threshold_numeric": 30,    
    "categorical_threshold_text": 30,
    "noise_threshold_ratio": 0.2,
    "primary_id": "primary_id"    
  },
  "DATA_PARAMS": {
    "sample_size": 1000,
    "niput_dim": 10
  }
}
```