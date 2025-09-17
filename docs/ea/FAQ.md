# Frequently Asked Questions (FAQ)

This document provides answers to common questions and troubleshooting tips for the NexusSOM platform.

---

### General Questions

**Q: What is the main difference between running `som.run` and `ea.ea`?**
**A:**
*   `som.run` executes a **single** analysis using one specific set of hyperparameters defined in its configuration file. It's used for direct analysis or for testing a specific configuration.
*   `ea.ea` runs the **evolutionary algorithm**, which automatically tests hundreds or thousands of different hyperparameter configurations to find a set of optimal solutions (the Pareto front). It's a tool for discovery, not for a single analysis.

**Q: Why do I need to run the scripts with `python3 -m som.run` instead of `python3 som/run.py`?**
**A:** Using the `-m` flag tells Python to run the script as a module within a package. This correctly sets up the project's internal import paths, allowing modules like `som.run` to find other modules like `som.utils`. Running it as a direct file path (`som/run.py`) breaks these imports. It is the standard and most robust way to execute scripts in a structured Python project.

**Q: The algorithm is running very slowly on my dataset. What can I do?**
**A:** Slow performance is usually caused by a few key parameters:
1.  **`map_size`:** An excessively large map (e.g., `[50, 50]`) dramatically increases computation. Ensure the map size is reasonable for your dataset size.
2.  **`epoch_multiplier`:** This directly controls the number of training iterations. A high value (e.g., `30.0`) will result in a long run. For a quick test, try a value like `1.0` or `5.0`.
3.  **`processing_type`:** The `"deterministic"` mode is the slowest as it processes the entire dataset in every iteration. For large datasets, `"hybrid"` or `"stochastic"` will be significantly faster.
4.  **`mqe_evaluations_per_run`:** Evaluating the MQE frequently on a large dataset can be a bottleneck. For initial runs, you can use a lower value (e.g., `10` or `20`).

---

### Troubleshooting & Errors

**Q: I'm getting a `ModuleNotFoundError` even though the file exists.**
**A:** This is almost always an import path issue. Ensure you are following these two rules:
1.  Run all scripts from the **root directory** of the project.
2.  Use the `python3 -m <package>.<module>` syntax (e.g., `python3 -m som.run ...`).

**Q: My maps look strange, like a single smooth gradient from one corner to another. What's wrong?**
**A:** This is a classic symptom of including a high-cardinality identifier column (like a `primary_id` or index) in the training data. This column creates an artificial, overly strong signal that dominates the entire organization process.
*   **Solution:** Make sure the name of your ID column is correctly specified in the `"primary_id"` field of your configuration file. The preprocessing step will then automatically mark it to be ignored during training.

**Q: My training stops very early, and the `mqe_evolution.png` graph is very short.**
**A:** This is likely due to the **early stopping** mechanism. It means the algorithm's moving average of the MQE did not improve for the number of checks specified by `"max_epochs_without_improvement"` (the "patience").
*   **Solution:** You can increase the `"max_epochs_without_improvement"` value in your config to give the algorithm more "patience". To disable it completely for a test run, set this value to a very high number (e.g., `50000`).

**Q: All my component planes look flat or random.**
**A:** This can be a symptom of the `primary_id` issue mentioned above, or it could mean the SOM did not have enough time to train properly.
*   **Solution:** First, verify the `primary_id` is correctly configured. If it is, try increasing the `"epoch_multiplier"` to allow for more training iterations.

---

### Evolutionary Algorithm (EA) Specifics

**Q: The EA finished, but my Pareto front only has one or two solutions. Is that normal?**
**A:** Yes, this can happen, especially with shorter runs or simpler datasets. It means the algorithm quickly converged to a small set of solutions that dominate all others. To get a more diverse front, you can try:
*   Increasing the `"population_size"` in `EA_SETTINGS`.
*   Running for more `"generations"`.
*   Expanding the ranges in the `"SEARCH_SPACE"` to allow for more diverse configurations.

**Q: How do I choose the "best" solution from the Pareto front?**
**A:** There is no single "best" solution—that's the point of Pareto optimization. You must choose based on your priorities:
*   For the **highest quality map**, look for the solution with the lowest **`topographic_error` (TE)**.
*   For the **fastest analysis**, look for the solution with the lowest **`duration`**.
*   For the most **faithful data representation**, look for the lowest **`best_mqe`**.
*   Often, the best practical choice is a **balanced solution** that scores well in all metrics without being the absolute best in any single one. Review the `pareto_front_log.txt` and inspect the generated maps for your top candidates.