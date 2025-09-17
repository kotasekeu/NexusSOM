# Evolutionary Hyperparameter Tuner for Self-Organizing Maps 🧬🧠

This project provides an advanced evolutionary algorithm for automatically finding optimal hyperparameters for Kohonen Self-Organizing Maps (SOMs). It leverages the **NSGA-II (Nondominated Sorting Genetic Algorithm II)** algorithm to perform multi-objective optimization, concurrently minimizing both the final quantization error (accuracy) and the training duration (performance).

The primary goal is to discover not a single "best" configuration, but a set of optimal trade-off solutions, known as the **Pareto Front**.

---

## 📚 Documentation

- [Configuration Guide](../docs/ea/CONFIG.md) ⚙️ — Details on how to set up and customize the evolutionary algorithm via `config.json`.
- [Run Instructions](../docs/ea/RUN.md) ▶️ — How to execute the algorithm, including command-line usage and output structure.
- [FAQ](../docs/ea/FAQ.md) ❓ — Frequently asked questions and troubleshooting.

---

## Core Concepts: Multi-Objective Optimization with NSGA-II

Tuning SOMs involves inherent trade-offs. A larger map or a higher number of training epochs may increase accuracy (lower quantization error) but at the cost of significantly longer computation time. Treating this as a multi-objective problem is more practical than using a simplistic single-score metric.

This algorithm addresses this challenge by seeking a set of solutions where:
-   **No solution is definitively better than another.** To improve one objective (e.g., accuracy), you must sacrifice performance in the other objective (e.g., speed).
-   The final output is a **Pareto Front**, allowing the user to select a configuration that best fits their specific needs—whether it's the fastest model, the most accurate one, or a balanced compromise.

NSGA-II is a state-of-the-art algorithm chosen for its efficiency and proven ability to find a well-distributed set of non-dominated solutions.

---

## ✨ Key Features

*   **Multi-Objective Optimization:** Concurrently optimizes for four key metrics:
    1.  **Quantization Error** (data fidelity)
    2.  **Topographic Error** (topological correctness)
    3.  **Training Time** (performance)
    4.  **Inactive Neuron Ratio** (map efficiency)
*   **Pareto Front Output:** Delivers a set of optimal, non-dominated solutions instead of a single, biased "winner".
*   **Efficient Parallel Processing:** Utilizes multiple CPU cores to evaluate the population in parallel. The data loading and preprocessing are performed only once to minimize I/O overhead and memory consumption.
*   **Diversity Preservation:** Implements *Crowding Distance* to ensure that the solutions on the Pareto front are well-distributed and cover a wide range of trade-offs.
*   **Robust and Valid Search:** A validation mechanism ensures that only logically valid hyperparameter combinations are generated and evaluated, preventing wasted computational resources.
*   **Flexible JSON-based Configuration:** The entire search space, evolutionary settings, and fixed SOM parameters are defined in an external JSON file, allowing for easy experimentation without code modification.

---

## 🛠️ Algorithm Breakdown

The evolutionary process follows these main steps in each generation:

1.  **Initialization:** An initial population of random, valid configurations is generated based on the `SEARCH_SPACE` defined in the configuration file.
2.  **Evaluation:** Each individual (configuration) in the population is used to train a SOM. Four objective values are recorded to assess the quality and efficiency of the resulting map:
    *   `best_mqe`: The best (lowest) mean quantization error achieved during training, measuring data fidelity.
    *   `duration`: The total wall-clock time for the training process, measuring performance.
    *   `topographic_error`: The percentage of data points for which the first and second best matching units are not adjacent, measuring topological correctness.
    *   `inactive_neuron_ratio`: The proportion of neurons that never won for any data point, measuring map size efficiency.
3.  **Selection (NSGA-II Core):** This is the heart of the algorithm, combining the current population with the best solutions found so far (the archive).
    *   **Non-Dominated Sorting:** The combined population is ranked into several fronts. The first front (rank 0) contains the best, non-dominated solutions (the current Pareto Front).
    *   **Crowding Distance Assignment:** To maintain diversity, a crowding score is calculated for each individual. Solutions in less populated regions of the objective space are preferred, preventing convergence to a single point.
    *   **Crowded Tournament Selection:** Parents for the next generation are selected based on their rank and crowding distance, favoring individuals with lower ranks and higher crowding distances.
4.  **Reproduction:** The selected parents create offspring using:
    *   **Uniform Crossover:** Each parameter in the child configuration is randomly inherited from one of the two parents.
    *   **Single-Gene Mutation:** A single parameter in the child's configuration is randomly changed to a new value from the search space.
5.  **Validation & Repair:** After reproduction, each new offspring is validated to ensure logical consistency (e.g., `start_learning_rate` >= `end_learning_rate`). If invalid, the configuration is repaired.
6.  **Elitism:** The best solutions found so far (the Pareto Front archive) are carried over to the next generation, ensuring that high-quality solutions are never lost.

---

## ▶️ Usage

To run the algorithm, use the following command structure:

```bash
python evolution.py --config <path_to_config.json> --input <path_to_data.csv>
```

-   `--config`: Path to the JSON configuration file. See [Configuration Guide](../docs/ea/CONFIG.md).
-   `--input`: (Optional) Path to the input data file. If omitted, synthetic data will be generated based on `DATA_PARAMS` in the config.

The results, including the generation-by-generation Pareto front, are logged in a new directory created in the same location as the input file. For details on output structure and further usage scenarios, refer to [Run Instructions](../docs/ea/RUN.md).

---

## 🚧 Limitations and Future Work

While the current implementation is robust and effective, there are several areas for improvement:

*   **Rudimentary Genetic Operators:** The current crossover and mutation operators are simple and designed for discrete parameter spaces (i.e., values from a list).
    *   **Improvement:** Implement more advanced operators like **Simulated Binary Crossover (SBX)** and **Polynomial Mutation**, which are better suited for continuous numerical ranges and can lead to more effective exploration.

*   **Code Structure:** The algorithm relies on several global variables (`WORKING_DIR`, `ARCHIVE`, etc.). This can hinder testability and maintainability.
    *   **Improvement:** Refactor the entire logic into an object-oriented structure (e.g., an `EvolutionOptimizer` class). This would encapsulate state, remove globals, and make the codebase cleaner and more modular.

*   **Limited Search Space Definition:** The `SEARCH_SPACE` only supports lists of discrete values.
    *   **Improvement:** Extend the configuration to support continuous ranges (e.g., `["float", 0.1, 0.9]`) and integer ranges, which would allow for a finer-grained search and require the more advanced genetic operators mentioned above.

---

For further details, troubleshooting, and updates, consult the [FAQ](../docs/ea/FAQ.md) or other documentation files in this directory.

