# Evolutionary Algorithm Configuration Guide

This document outlines the configuration of the evolutionary algorithm used for optimizing the parameters of a Modified Kohonen Self-Organizing Map (SOM). All settings are managed through the `config.json` file.


## Usage

This configuration file is utilized by passing its path as a command-line argument to the evolutionary algorithm script. For detailed instructions on script execution, command-line syntax, and usage examples, please refer to the `RUN.md` file, located in the same directory.

### Configuration Design Philosophy

The configuration is deliberately structured into sections to provide a clear and unambiguous definition of the experimental setup. This explicit separation ensures that parameters intended for optimization (`SEARCH_SPACE`) are distinctly identified from static run-time constants (`FIXED_PARAMS`). This design choice enhances the configuration's readability and robustness, directly supporting the reproducibility of results.

## Core Concept

The `config.json` file defines both the fixed parameters for a run and the search space for hyperparameters to be optimized. The configuration follows a simple principle:

*   **Single Value**: If a parameter is assigned a single value (e.g., a number or a string), it is treated as a fixed constant for the entire run.
*   **Array of Values**: If a parameter is assigned an array of values, it defines the search space for that hyperparameter. The evolutionary algorithm will explore combinations of these values to find an optimal configuration.

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

The configuration file is organized into three primary sections:

1.  **`EA_SETTINGS`**: Controls the evolutionary process itself (e.g., population size, number of generations).
2. **`DATA_PARAMS`**: Contains parameters related to the dataset, such as sample size and input dimensions. This section is used only if the dataset is auto-generated. If a dataset file is provided, these parameters are ignored.
3. **`SEARCH_SPACE`**: The core section defining which SOM parameters and their respective ranges the algorithm should optimize. Arrays are typically used here.
3.  **`FIXED_PARAMS`**: Defines constants for the run that remain unchanged throughout the evolution, such as map topology or data generation settings.

---

## Parameter Reference

### 1. `EA_SETTINGS`

Parameters that govern the behavior of the evolutionary algorithm.

*   `"population_size"`: (Integer) The number of individuals (configurations) in each generation.
*   `"generations"`: (Integer) The total number of generations for which the evolution will run before terminating.

### 2. `DATA_PARAMS`
*   `"sample_size"`: (Integer) 
*   `"input_dim"`: (Integer) 

### 3. `SEARCH_SPACE & FIXED_PARAMS`

The hyperparameter space for the Kohonen SOM to be explored by the algorithm.

*   `"map_size"`: (Array of Arrays) Defines the map dimensions to be tested. Each inner array must be a `[width, height]` pair.
*   
*   `"start_learning_rate"`: (Array of Floats, 0-1) The initial learning rate, determining the magnitude of weight updates.
*   `"end_learning_rate"`: (Array of Floats, 0-1) The lower bound to which the `start_learning_rate` will decay over the course of training.
*   `"lr_decay_type"`: (Array of Strings) The decay function for the `learning_rate` (e.g., `"linear-drop"`, `"exp-drop"`).


*   `"start_radius_init_ratio"`: (Array of Floats) Initial neighborhood radius as a ratio of the map size, determining the initial size of the neighborhood for weight updates.
*   `"end_radius"`: (Array of Floats) Final neighborhood radius for the SOM, controlling the size of the neighborhood for weight updates.
*   `"radius_decay_type"`: (Array of Strings) The decay function for the neighborhood radius.


*   `"start_batch_percent"`: (Array of Floats) The initial batch size as a percentage of the total dataset size, determining how many samples are processed in each batch.
*   `"end_batch_percent"`: (Array of Floats) The final batch size as a percentage of the total dataset size, controlling how many samples are processed in each batch.
*   `"batch_growth_type"`: (Array of Strings) The function governing how the batch size increases over time (e.g., `"exp-growth"`, `"linear-growth"`).   


*   `"epoch_multiplier"`: (Array of Floats) A multiplier that determines the total number of training epochs, calculated as `number_of_samples * epoch_multiplier`.
*   `"normalize_weights_flag"`: (Array of Booleans) If `true`, neuron weight vectors are normalized after each weight update.
*   `"growth_g"`: (Array of Floats) The `G` parameter for the growth and decay function, controlling the steepness of the curve.  
*   `"random_seed"`: (Integer or `null`) The seed for the random number generator to ensure run reproducibility. Set to `null` for non-deterministic behavior.
*   `"map_type"`: (String) The topology of the SOM grid (e.g., `"square"`).
*   `"num_batches"`: (Integer) The number of batches to process within a single epoch.
*   `"max_epochs_without_improvement"`: (Integer) A safeguard for early stopping. Training for a given individual will terminate if the quantization error does not improve for this number of epochs.

---

## Complete `config.json` Example

```json
{
  "EA_SETTINGS": {
    "population_size": 10,
    "generations": 20
  },
  "DATA_PARAMS": {
    "sample_size": 1000,
    "input_dim": 10
  },
  "SEARCH_SPACE": {
    "map_size": [
      [10, 10],
      [15, 15],
      [20, 20],
      [25, 25],
      [30, 30]
    ],
    "start_learning_rate": [0.99, 0.9, 0.8, 0.7, 0.6, 0.5],
    "end_learning_rate": [0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.01],
    "lr_decay_type": ["linear-drop", "exp-drop"],

    "start_radius_init_ratio": [1, 0.5, 0.25],
    "end_radius": [2, 1, 0.5],
    "radius_decay_type": ["linear-drop", "exp-drop"],

    "start_batch_percent": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0],
    "end_batch_percent": [1.0, 2.0, 5.0, 7.5, 10.0, 15.0],
    "batch_growth_type": ["exp-growth", "linear-growth"],

    "epoch_multiplier": [0.5, 1.0, 2.0, 5.0],
    "normalize_weights_flag": [false, true],
    "growth_g": [5.0, 15.0, 30.0, 50.0]
  },
  "FIXED_PARAMS": {
    "random_seed": 42,
    "map_type": "square",
    "num_batches": 10,
    "max_epochs_without_improvement": 25
  }
}
```