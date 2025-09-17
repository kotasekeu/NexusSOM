# Interpreting the Pareto Front

The `pareto_front_log.txt` file contains the final output of the evolutionary algorithm: the **Pareto front**. This is not a single "best" solution, but a set of optimal, non-dominated trade-offs. This guide explains how to read and interpret its structure.

## What is the Pareto Front?

In multi-objective optimization, the Pareto front is a set of solutions where no single solution is definitively better than another across all objectives. To improve one objective (e.g., accuracy), you must necessarily sacrifice performance in another (e.g., speed). The EA's goal is to find this set of optimal compromises, allowing you to choose the one that best fits your specific needs.

The most important part of the log is the **final generation's section**, which represents the best solutions found throughout the entire run.

---

## Structure of a Single Entry

Each solution in the Pareto front log is presented in a structured format. Let's break down an example:

```
UID: a506531841bdb53cd900ffd66c6987bf
  - Objectives: QE=0.089854, TE=0.0133, Time=2.94s
  - U-Matrix:   Mean=0.3545, Std=0.1227
  - Params:
    - batch_growth_type: linear-growth
    - end_batch_percent: 5.0
    - end_learning_rate: 0.1
    - epoch_multiplier: 5.0
    - ... (other hyperparameters)
```

### 1. Key Information

*   **`UID`**: The unique identifier (hash) for this specific configuration. This UID corresponds to a subdirectory in the `individuals/` folder, where you can find detailed artifacts like training graphs and output maps (`u_matrix.png`, etc.).

### 2. Performance Metrics

This section summarizes the performance of the solution.

*   **`Objectives`**: These are the primary metrics the EA was actively optimizing.
    *   **`QE` (Quantization Error):** Measures data fidelity. Lower is better.
    *   **`TE` (Topographic Error):** Measures the quality of the map's topological structure. Lower is better.
    *   **`Time` (Duration):** The total training time in seconds. Lower is better.
*   **`U-Matrix`**: These are secondary, analytical metrics that describe the visual quality of the resulting U-Matrix.
    *   **`Mean`:** The average distance between adjacent neurons. A higher value suggests better-separated clusters.
    *   **`Std` (Standard Deviation):** The variance in distances. A higher value indicates a high-contrast map with both clear clusters ("valleys") and distinct boundaries ("mountains").

### 3. Hyperparameters (`Params`)

This section lists the exact set of hyperparameter values that produced the metrics above. This is the "recipe" for the solution. By analyzing the parameters of different "champion" solutions (e.g., the fastest vs. the most accurate), you can derive insights into how different settings affect the outcome.

---

## How to Analyze the Pareto Front

When reviewing the final generation in the log, look for specific types of solutions to understand the trade-offs:

1.  **The "Quality Champion":** Find the solution with the lowest **Topographic Error (TE)**. This configuration produced the most structurally correct map, often at the cost of longer training time.
2.  **The "Accuracy Champion":** Find the solution with the lowest **Quantization Error (QE)**. This configuration is the most faithful to the data points, but may not have the best topology.
3.  **The "Speed Champion":** Find the solution with the lowest **Time**. This is the fastest configuration, but it likely compromises on QE and TE.
4.  **The "Balanced Choice":** Look for a solution that does not excel in any single metric but scores very well across all of them (e.g., low QE, low TE, and reasonably low Time). This is often the most practical solution for real-world applications.

By comparing the parameters of these "champions", you can formulate conclusions about which hyperparameters are most critical for achieving speed, accuracy, or structural quality for your specific dataset.