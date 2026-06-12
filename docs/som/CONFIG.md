# SOM Configuration Guide

Configuration for a single run of the Hybrid Kohonen Self-Organizing Map.
All settings live in one flat JSON file passed via `-c`. A single file is the
complete, reproducible definition of one experiment (plus the optional
`-s/--seed` CLI override for multi-seed studies).

Updated for the refactored module (see `REFACTOR_PLAN.md`, branch `4-som-cleanup`).

## Example

```json
{
    "preprocess_strategy": "nexus",
    "primary_id": "ID",
    "delimiter": ",",
    "categorical_threshold_numeric": 20,
    "categorical_threshold_text": 20,
    "noise_threshold_ratio": 0.2,

    "map_size": [30, 30],
    "map_type": "hex",
    "random_seed": 42,
    "epoch_multiplier": 5.0,

    "start_learning_rate": 0.9,
    "end_learning_rate": 0.1,
    "lr_decay_type": "linear-drop",

    "start_radius_init_ratio": 1.0,
    "end_radius": 1.0,
    "radius_decay_type": "linear-drop",

    "start_batch_percent": 0.01,
    "end_batch_percent": 0.01,
    "batch_growth_type": "static",
    "num_batches": 1,

    "mqe_evaluations_per_run": 500,
    "save_checkpoints": true,
    "checkpoint_count": 10,
    "track_sample_coverage": false,

    "save_training_plots": true,
    "save_visualizations": true
}
```

---

## 1. Preprocessing parameters

Used by `som/preprocess.py` (pure stage — returns a `PreprocessResult`,
artifacts are saved by `som/persistence.py`).

| Key | Default | Description |
|---|---|---|
| `preprocess_strategy` | `"nexus"` | How much of the pipeline is applied. `"nexus"` = full (noise exclusion, ignore mask, median fill, encoding, MinMax). `"scale-only"` = keep all columns, no mask, but normalize — isolates the mask contribution. `"none"` = encoding only, no normalization — ablation baseline; organization is expected to collapse. Recorded into `dataset_meta.json` as `ds_preprocess_strategy`. |
| `primary_id` | `"primary_id"` | Record-identifier column. Always masked for training and excluded from analyzable feature lists. |
| `delimiter` | `","` | CSV delimiter. |
| `selected_columns` | — | Optional list; restricts the input to these columns (missing ones raise an error). |
| `categorical_threshold_numeric` | `30` | Numeric column with ≤ N unique values is treated as categorical. |
| `categorical_threshold_text` | `30` | Text column with ≤ N unique values is treated as categorical. |
| `noise_threshold_ratio` | `0.2` | Text column whose unique-value ratio exceeds this is dropped as noise (`nexus` strategy only). |

Missing values: numeric NaN → column median, text NaN → empty string. Under
`nexus`, originally-missing cells are additionally marked in the **ignore
mask** so they are invisible to BMU selection, weight updates, and error
metrics (see `issues.md` #3–4 for why this matters).

## 2. SOM core parameters

Constructor arguments of `KohonenSOM` (`som/som.py`). The data dimension
(`dim`) is set automatically by the pipeline.

| Key | Default | Description |
|---|---|---|
| `map_size` | `[10, 10]` | Grid size `[width, height]`. |
| `map_type` | — | `"hex"` (6 neighbors, cube coordinates) or `"square"` (Moore neighborhood). |
| `random_seed` | `null` | Seeds numpy for weight init and batch sampling. Same seed → bit-identical run. Overridable per run via `run_pipeline(seed=...)` / CLI `-s`. |
| `epoch_multiplier` | — | Total iterations = `epoch_multiplier × n_samples`. |
| `start_learning_rate` / `end_learning_rate` | — | LR range over training. |
| `lr_decay_type` | — | Decay curve, see below. |
| `start_radius_init_ratio` | `1.0` | Initial radius = ratio × `max(map_width, map_height)`. |
| `end_radius` | — | Final neighborhood radius. |
| `radius_decay_type` | — | Decay curve, see below. |
| `start_batch_percent` / `end_batch_percent` | — | Percentage of the dataset sampled per iteration **from each section**. |
| `batch_growth_type` | — | Curve for batch size evolution (use a `*-growth` type to grow). |
| `sampling_method` | `reshuffle` | How samples are drawn per iteration: `reshuffle` (default since 2026-06-12) = without-replacement epoch shuffling — pointer over a per-epoch re-shuffled permutation; every sample hit equally often ±1, **guaranteed coverage**, and faster (O(1) per draw). `random` = legacy behavior: fresh `np.random.choice` each iteration (with replacement *across* iterations — coverage probabilistic, ~Poisson(λ), needs em ≈ 10 for 99.9 %). Measured basis: `docs/ea/SEARCH_SPACE.md` step 1, experiments B (coverage) and C (quality A/B: no regression at equal budget, 7–30 % faster). ⚠ Configs of runs recorded before the flip must state `"sampling_method": "random"` explicitly to stay replayable. `cycle` is a deprecated alias for `reshuffle`. |
| `num_batches` | — | Number of sections the shuffled dataset is split into; each iteration samples from every section. ⚠ **Measured useless for coverage** (matches the Poisson null model exactly at equal budget; raising it just multiplies per-iteration throughput) — keep at `1`; see `docs/ea/SEARCH_SPACE.md` experiment A. |
| `growth_g` | — | Steepness of `exp-*` and `log-*` curves. |
| `normalize_weights_flag` | — | Re-normalize weight vectors to unit norm each iteration. EA experience: consistently degrades organization (see `verify_ea_run.py`). |
| `mqe_evaluations_per_run` | `20` | How many times MQE (and stopping criteria) are evaluated during a run. |
| `max_epochs_without_improvement` | — | Early-stopping patience. ⚠ Together with `early_stopping_window` this feature is currently **effectively disabled** (defaults 50000) — see `issues.md` #2. |
| `save_checkpoints` | `false` | Record training checkpoints (progress, MQE, TE, dead ratio, LR, radius) — required for LSTM training data and the multi-seed MQE comparison. |
| `checkpoint_count` | `10` | Number of checkpoints per run. |
| `checkpoint_every_mqe` | `false` | Checkpoint at every MQE evaluation instead (dense curves, ~`mqe_evaluations_per_run` points). |
| `track_sample_coverage` | `false` | Count how many times each input vector is processed → `csv/sample_coverage.json`. Tracking verified correct 2026-06-12 (exact replay by `app/tools/coverage_sim.py verify`); see `article_implementation.md` item 3. |
| `show_progress` | `true` | tqdm progress bar. EA and batch tools disable it. |

### Decay types (`lr_decay_type`, `radius_decay_type`, `batch_growth_type`)

`static`, `linear-drop`, `linear-growth`, `exp-drop`, `exp-growth`,
`log-drop`, `log-growth`, `step-down` (10 steps, factor 0.7).

Domain constraint: LR and radius must **always decrease** during organization
— use `*-drop` types for them. "Dynamic" control (LSTM controller) modulates
the decay speed, never the direction. Growth types exist for the batch size.

### Training modes

There is no `processing_type` switch — hybrid sampling covers the extremes
(`issues.md` #6):

- **deterministic**: `num_batches: 1`, batch percent `100`
- **stochastic**: `num_batches: 1`, batch percent `≈ 1/n_samples` (1 sample)
- **hybrid**: anything between; growing batches

`num_batches > 1` (sections) was part of the original hybrid design but is
measured to add nothing to coverage — keep it at 1 (`docs/ea/SEARCH_SPACE.md`
experiment A). The default `sampling_method: reshuffle` gives stochastic
runs guaranteed coverage at a fraction of the passes (experiment B) with
no quality cost (experiment C); for deterministic runs (batch 100 %) the
method only changes the per-iteration processing order.

## 3. Analysis parameters

| Key | Default | Description |
|---|---|---|
| `std_threshold` | `2.5` | Z-score threshold for global and per-neuron (local) outlier detection in `som/analysis.py`. |

## 4. Output control

| Key | Default | Description |
|---|---|---|
| `save_training_plots` | `true` | MQE/LR/radius/batch evolution PNGs. Disable for batch runs. |
| `save_visualizations` | `true` | All maps (U-Matrix, hit, distance, dead neurons, cluster, component planes, pie). Disable for batch runs — maps can be re-rendered later from artifacts via `som.visualization.render_results_dir(results_dir)`. |

EA has its own per-individual switches in `FIXED_PARAMS`:
`generate_training_plots`, `generate_individual_maps`, `save_individual_weights`
(default behavior documented in `app/ea/ea.py`).

## 5. Neural-network section (optional)

```json
"NEURAL_NETWORKS": {
    "use_lstm": true,
    "lstm_model_path": "app/lstm/models/....keras",
    "lstm_scaler_path": "app/lstm/models/....pkl",
    "lstm_quality_threshold": 0.75,

    "use_lstm_controller": true,
    "lstm_controller_model_path": "app/lstm/models/....keras",
    "lstm_controller_scaler_path": "app/lstm/models/....pkl"
}
```

| Key | Description |
|---|---|
| `use_lstm` | LSTM early stopping — aborts runs predicted to end poorly (fires after ≥ 20 % of training). |
| `lstm_quality_threshold` | Stop when predicted quality score falls below this (default 0.75). |
| `use_lstm_controller` | LSTM Phase 3 controller — adjusts LR/radius decay speed per checkpoint (multiplicative cumulative factors). |

Requires the NN extras (`python/requirements_nn.txt`). When a model fails to
load, the run continues without the feature (warning only). The core pipeline
has no NN dependency.
