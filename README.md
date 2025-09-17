# NexusSOM: Autonomous Data Analysis Platform 🚀🧠

**NexusSOM** is an advanced, modular platform for **exploratory data analysis** and **unsupervised machine learning**. At its core, it leverages a powerful **Hybrid Self-Organizing Map (SOM)** to perform topological data analysis, coupled with a sophisticated **Evolutionary Algorithm (EA)** that automates the discovery of optimal hyperparameters.

The system is designed not just to cluster data, but to uncover its intrinsic structure, identify meaningful patterns, and detect anomalies, transforming raw data into actionable insights with minimal human intervention.

---

## 🎯 Core Philosophy & Vision

The project is built on the vision of creating an **autonomous, self-optimizing analytical engine**. Instead of requiring a data scientist to manually tune complex models, NexusSOM uses evolutionary computation to find the best analytical pipeline for any given dataset.

-   **Phase 1 (Current):** A robust Hybrid SOM and an EA that optimizes its hyperparameters based on data fidelity, topological correctness, performance, and map efficiency.
-   **Phase 2 (Future):** Integration of Deep Learning models (CNNs, Autoencoders) to automate the visual interpretation of maps and enhance anomaly detection.
-   **Phase 3 (Vision):** A fully autonomous, self-improving system that combines multiple analytical techniques (SOM, HDBSCAN, etc.) and uses LLMs to generate human-readable reports, moving from data to insight automatically.

---

## ✨ Key Features

-   **Hybrid SOM Implementation:** A flexible SOM algorithm that combines the speed of stochastic training with the stability of deterministic methods.
-   **Advanced Evolutionary Optimization:** A multi-objective NSGA-II algorithm that optimizes for **four criteria simultaneously**:
    1.  **Quantization Error** (Accuracy)
    2.  **Topographic Error** (Structural Quality)
    3.  **Performance** (Training Time)
    4.  **Map Efficiency** (Inactive Neuron Ratio)
-   **Automated Analysis Pipeline:** From preprocessing and normalization to training, analysis, and visualization, the entire workflow is streamlined.
-   **Rich Visualization Suite:** Generates a comprehensive set of plots and maps for deep analysis, including U-Matrix, Component Planes, Hit Maps, and Distance Maps.
-   **Modular and Extensible:** Designed with a clear separation of concerns (`som`, `ea`, `analysis`, `visualization`), making it easy to extend or modify.

---

## 📚 Documentation

The project is extensively documented to guide users and developers. All documentation is located in the `/docs` directory.

| Document | Description |
| :--- | :--- |
| **Quick Start** | |
| 📄 [`INSTALL.md`](./docs/INSTALL.md) | **Start here.** Step-by-step installation instructions. |
| **Single SOM Run (`som/`)** | |
| ⚙️ [`som/CONFIG.md`](./docs/som/CONFIG.md) | Guide to all parameters for a single SOM run. |
| ▶️ [`som/RUN.md`](./docs/som/RUN.md) | How to execute a single SOM analysis. |
| 🖼️ [`som/EXAMPLES.md`](./docs/som/EXAMPLES.md) | **(New)** A visual gallery of all generated maps and plots with explanations. |
| **Evolutionary Algorithm (`ea/`)** | |
| ⚙️ [`ea/CONFIG.md`](./docs/ea/CONFIG.md) | Guide to configuring the evolutionary optimization. |
| ▶️ [`ea/RUN.md`](./docs/ea/RUN.md) | How to execute an evolutionary run to find optimal parameters. |
| **Understanding Results** | |
| 📊 [`RESULTS.md`](./docs/RESULTS.md) | **Essential reading.** Detailed explanation of all output files for a single SOM run. |
| 📈 [`ea/RESULTS.md`](./docs/ea/RESULTS.md) | **(New)** Explanation of the output structure specific to an EA run. |
| 🏆 [`ea/PARETO.md`](./docs/ea/PARETO.md) | **(New)** A guide to interpreting the Pareto front—the final output of the EA. |
| **General** | |
| ❓ [`FAQ.md`](./docs/FAQ.md) | Frequently asked questions and troubleshooting tips. |
---

## 🚀 Quick Start Guide

1.  **Install the environment:**
    Follow the instructions in **[`INSTALL.md`](./docs/INSTALL.md)**.

2.  **Run a single SOM analysis:**
    Navigate to the project root and execute the `run_som.py` script via the `som` module. This is the best way to test your configuration on a dataset.

    ```bash
    python3 -m som.run -i path/to/data.csv -c path/to/config.json
    ```
    *See [`som/RUN.md`](./docs/som/RUN.md) for more details.*

3.  **Find optimal hyperparameters with the Evolutionary Algorithm:**
    To unleash the full power of the platform, run the `run_ea.py` script. This will start the multi-objective optimization process.

    ```bash
    python3 -m ea.ea -i path/to/data.csv -c path/to/ea_config.json
    ```
    *See [`/ea/RUN.md`](./docs/ea/RUN.md) for more details.*

---

## 📁 Project Structure

```
.
├── docs/             # All documentation files
├── ea/               # Evolutionary Algorithm module
├── som/              # Core SOM and Analysis module
├── test/             # Test datasets and configurations
├── run_som.py        # Executable script for single SOM runs
├── run_ea.py         # Executable script for evolutionary runs
└── requirements.txt  # Project dependencies
```