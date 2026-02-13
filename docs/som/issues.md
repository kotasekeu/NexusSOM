# SOM Module — Known Issues

This document describes issues encountered during the development of the SOM module.
Each entry follows the format: what the issue is, why it occurs, and how it is (or can be) resolved.

---

## 1. Excessive Logging Slows Down Training

**Issue**
Every phase of the SOM pipeline writes log entries by calling `log_message()`. In `run.py` alone there are ~25 calls covering directory setup, validation, preprocessing, training start/end, analysis, plots, and visualizations. Additional calls appear inside `preprocess.py` for each processing step. During an EA run that executes hundreds of SOM trainings in parallel, this volume of I/O adds measurable overhead.

**Why**
`log_message()` in `utils.py` opens the log file, appends a single line, and closes it on every call. There is no buffering, no log level filtering, and no way to disable logging at runtime. The cost is not in any single call — it is in the cumulative effect across the full pipeline, especially when the EA spawns many parallel workers all writing to their own log files simultaneously.

**How It Can Be Solved**
The simplest fix is to replace per-call file opens with a buffered approach: open the file once per run, buffer writes in memory, and flush periodically or at the end. An alternative is to add a log level threshold so that routine `SYSTEM` messages are skipped unless explicitly requested, while `ERROR` and `FATAL` messages are always written. Neither change affects the algorithm itself — this is purely an I/O optimization.

---

## 2. Early Stopping Window Can Terminate Training Too Soon

**Issue**
The early stopping mechanism uses a moving average over a window of recent MQE evaluations. If the window is too small, normal short-term MQE fluctuations can look like a plateau, and the patience counter triggers a stop before the map has actually converged. This risks producing a suboptimal map, particularly in the early phases of training when MQE is still decreasing unevenly.

**Why**
MQE is not evaluated at every iteration — it is computed at a fixed interval (`mqe_compute_interval = total_iterations / mqe_evaluations_per_run`). The moving average is calculated over the last `early_stopping_window` evaluations using a `deque`. The stop fires when `epochs_without_improvement` reaches `early_stopping_patience`. If the window is small relative to the number of evaluations, or if MQE has a temporary spike followed by a dip, the moving average can stall even though the overall trend is still improving.

**Current State**
Both `early_stopping_window` and `early_stopping_patience` are currently set to a default of 50000, which effectively disables the feature — training always runs for the full `epoch_multiplier × num_samples` iterations. There are explicit `FIXME` comments in the code marking this as intentionally disabled until the interaction between window size, patience, and MQE evaluation frequency is properly tuned.

**How It Can Be Solved**
The parameters need to be calibrated relative to `mqe_evaluations_per_run` and the expected training length. A reasonable starting point is a window of 3–5 evaluations and patience of 2–3 checks. This should be validated empirically across different map sizes and datasets before re-enabling the feature.

---

## 3. Missing or Corrupted Sensor Data Tears the SOM Map Apart

**Issue**
When the source dataset contains missing values — caused either by sensor malfunction or by genuinely extreme/outlier readings — the SOM map can become distorted. Neurons near the missing-value samples get pulled toward artificial fill values, creating visible tears or dead zones in the resulting map topology.

**Why**
Missing values cannot simply be removed from a row because each sample must have the same dimensionality as the weight vectors. The current preprocessing pipeline fills NaN values with column medians (for numeric columns) or empty strings (for categorical columns). These fill values are synthetic and do not represent any real data point. Without a mechanism to exclude the filled dimensions from influencing the map, the SOM treats them as real observations during BMU selection and weight updates, which distorts the learned topology.

**How It Is Solved — The Ignore Mask**
A per-sample boolean mask (`ignore_mask`) is created during preprocessing from `training_df.isnull()`. Any dimension that was originally NaN is marked `True`. The primary ID column is also unconditionally masked. This mask is carried through the entire training loop and applied in four places:

- **BMU selection** (`find_bmu`): masked dimensions are zeroed out before computing Euclidean distance, so missing values do not influence which neuron wins.
- **Weight update** (`update_weights`): the update term is zeroed on masked dimensions, so neuron weights are not pulled toward fill values.
- **Quantization error** (`compute_quantization_error`): masked dimensions are excluded from the error calculation, keeping MQE honest.
- **Topographic error** (`calculate_topographic_error`): same zeroing applied before distance computation.

The result is that missing dimensions are effectively invisible to the algorithm. The SOM learns from the dimensions that are actually present for each sample.

---

## 4. The Mask Changes the Effective Data — Downstream Interpretation Required

**Issue**
By masking missing dimensions, we prevent the SOM from being corrupted, but we also change what the map represents. Two samples that look identical on their non-missing dimensions will map to the same neuron, even if they originally differed on a now-masked dimension. The clusters that emerge from the SOM no longer have a uniform feature basis — different neurons may have been shaped by different subsets of dimensions.

**Why**
This is an inherent consequence of the mask-based approach. The mask is necessary (without it the map tears apart, as described in Issue 3), but it introduces asymmetry: sample A might contribute to a neuron's weight in all 10 dimensions, while sample B only contributes in 7. The neuron's final weight vector is shaped by a mix of complete and incomplete observations. This is invisible in the raw SOM output — the weights and visualizations look normal.

**How It Must Be Handled — Data Mining Module**
This issue cannot be fully solved within the SOM module alone. It requires awareness at the interpretation stage. The future Data Mining module must:

- Track which dimensions were masked for each sample and which neurons they contributed to.
- Flag clusters where a significant portion of contributing samples had masked dimensions.
- Avoid drawing conclusions about masked dimensions from neuron weights that were not informed by those dimensions.
- Consider presenting cluster analyses separately for fully-observed samples versus samples with missing data, to make the distinction visible to the end user.

This is a system-level concern. Solving it requires changes that span from the SOM output format through to the final user-facing report.
