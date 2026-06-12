# EA Output Structure

Directory layout and file contents produced by one `app/run_ea.py` run.
Column semantics follow the objectives/constraints defined in `EA.md`.

## Run root — `data/datasets/<Name>/results/<YYYYMMDD_HHMMSS>/`

Shared by all seeds (preprocessing and calibration run once):

```
<timestamp>/
├── dataset_meta.json          # ds_* dataset stats (samples, dims, types, missing ratio)
├── calibration_probe.csv      # probe_idx, org_max — basis of org_threshold (EA.md §7)
├── calibration_probes/<i>/    # working dirs of the probe trainings
├── csv/
│   ├── original_input.csv     # input as loaded
│   ├── training_data.npy      # normalized training matrix (n × d)
│   ├── training_data_readable.csv
│   └── ignore_mask.csv        # per-cell ignore mask (primary ID, missing values)
├── json/
│   └── preprocessing_info.json
├── log.txt                    # validation/preprocess log
└── seed_<seed>/               # one independent evolution per seed
```

`dataset_meta.json` fields (`ds_n_samples`, `ds_n_active_dimensions`,
`ds_n_numeric`, `ds_n_categorical`, `ds_missing_ratio`, …) are injected into
every `results.csv` row — they are the dataset features for MLP training.

## Seed directory — `seed_<seed>/`

```
seed_42/
├── results.csv          # every evaluated individual (authoritative record)
├── pareto_front.csv     # archive snapshot per generation
├── pareto_metrics.csv   # HV / spacing / spread per generation
├── status.csv           # evaluation lifecycle
├── log.txt              # per-UID event log
├── individuals/<uid>/   # per-individual artifacts
└── maps_dataset/        # only with generate_individual_maps (CNN legacy)
```

### `results.csv` — one row per evaluation

UIDs are unique within a run (gene-only hashing + pre-evaluation dedup,
`issues.md` #89). Groups of columns:

| Group | Columns |
|---|---|
| Identity | `uid` (MD5 of genes = directory name in `individuals/`), `dataset_name` |
| **Raw objectives** (NSGA-II inputs) | `raw_mqe_improvement_ratio` (final/initial MQE, lower=better), `raw_topographic_error`, `raw_topological_correlation` (ρ; objective is `1−ρ`) |
| Raw support metrics | `raw_best_mqe`, `initial_mqe`, `dead_neuron_ratio`, `dead_neuron_count`, `u_matrix_mean/std/max`, `distance_map_max` |
| Constraints | `constraint_violation` (0 = feasible), `is_penalized`, `penalty_factor` (=1+CV, reference only), `penalty_reason` (`org(u=…,d=…)` / `dead=…%(thresh=…%)` / `mlp_prescreened`) |
| Legacy penalized values | `best_mqe`, `topographic_error`, `mqe_improvement_ratio` (today equal to raw — kept for cross-run comparability with historical CSVs) |
| Run stats | `duration`, `total_weight_updates`, `epochs_ran`, `map_m`, `map_n` |
| NN | `cnn_quality_score` (empty unless the closed CNN track is enabled) |
| Dataset context | all `ds_*` fields |
| Genes | every `SEARCH_SPACE` parameter (`map_size`, learning rates, decay types, …) |

### `pareto_front.csv` — archive snapshot per generation

One row per (generation, archive member); the archive is the capped rank-0
front (`max_archive_size`), **not** the survivor population. Columns:
`generation`, `uid`, `dataset_name`, the three raw objectives as
`raw_mqe_ratio` / `raw_te` / `raw_topo_corr`, `dead_ratio`,
`constraint_violation` + penalty metadata, `initial_mqe`, `pen_mqe_ratio`,
`pen_te`, `map_m`, `map_n`, U-matrix stats, `duration`, `ds_*`, and all
search-space parameters. The last generation block is the final result of
the run.

### `pareto_metrics.csv` — front quality per generation

`generation`, `front_size`, `hv` (hypervolume vs [1.1]³ in running-min/max
normalized space; empty without pymoo), `spacing` (0 = uniform),
`spread_mqe`, `spread_te`, `spread_dead` (≈1 = front spans the observed
range). Feasible archive members only. See `EA.md` §9.

### `status.csv` — evaluation lifecycle

`uid`, `population_id`, `generation` (0-based), `status`
(`started` / `completed` / `failed` / `mlp_skipped`), `start_time`,
`end_time`. Each evaluation produces a `started` row and a terminal row —
`completed` counts per generation are the quickest run-health check.

## Per-individual directory — `individuals/<uid>/`

```
<uid>/
└── csv/
    ├── training_checkpoints.json   # training time series (LSTM training data)
    ├── sample_coverage.json        # which samples hit which neurons
    └── weights.npy                 # only with save_individual_weights
```

With the opt-in flags, additionally `visualizations/` (U-matrix, distance
map, dead-neurons map + legends) and training plots (MQE evolution, LR /
radius decay, batch growth).

`training_checkpoints.json` is a list of per-checkpoint records:

```json
{ "iteration": 0, "progress": 0.0, "mqe": 0.659, "topographic_error": 0.146,
  "dead_neuron_ratio": 0.847, "learning_rate": 0.648, "radius": 8.61 }
```

`progress` ∈ [0, 1] normalizes differently long trainings onto one axis;
`checkpoints[0]` is the random-init baseline used by the MQE-ratio
objective. With `checkpoint_every_mqe: true` there are
`mqe_evaluations_per_run` records per individual.

## `maps_dataset/` (CNN legacy, opt-in)

`<uid>_u_matrix.png`, `<uid>_distance_map.png`, `<uid>_dead_neurons_map.png`
copies plus `rgb/<uid>_rgb.png` composites (channels R=U-matrix,
G=distance, B=dead neurons). Produced only with
`generate_individual_maps: true` — kept for regenerating the CNN-closure
exhibit data (`issues.md` #90).

## Data volume guideline

A 5-seed × 50 × 6 collection run ≈ 1 500 evaluations; with
`checkpoint_every_mqe` at 300 evaluations/run that is ~450 000 checkpoint
records (~the LSTM training corpus for one dataset). Weights and PNGs stay
off unless explicitly enabled.
