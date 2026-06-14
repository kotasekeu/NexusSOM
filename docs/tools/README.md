# Tools — overview

All tools live in [`app/tools/`](../../app/tools/).

| Tool | Documentation | Purpose |
|------|---------------|---------|
| `budget_saturation.py` | [doc](budget_saturation.md) | Experiment D driver + report: quality-vs-training-budget saturation curves per processing regime (SEARCH_SPACE step 2) |
| `coverage_sim.py` | [doc](coverage_sim.md) | Simulates the batch-sampling regime without SOM training; measures dataset coverage (SEARCH_SPACE step 1) |
| `generate_benchmark.py` | [doc](generate_benchmark.md) | Generates Swiss Roll and space-filling benchmarks for SOM verification |
| `generate_dataset.py` | [doc](generate_dataset.md) | Generates synthetic data following a real dataset's schema |
| `generate_trips_dataset.py` | [doc](generate_trips_dataset.md) | Generates a business-trips dataset with injected anomalies (ground truth) |
| `plot_dim_qe.py` | [doc](plot_dim_qe.md) | Per-dimension QE heatmaps on the SOM map |
| `plot_pareto_evolution.py` | [doc](plot_pareto_evolution.md) | Visualizes Pareto-front evolution across EA generations |
| `plot_som_topology.py` | [doc](plot_som_topology.md) | Projects SOM topology to 2D/3D/HTML |
| `verify_ea_run.py` | [doc](verify_ea_run.md) | EA run correctness diagnostics |
