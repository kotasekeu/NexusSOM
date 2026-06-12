# EA Configuration Reference (`config-ea.json`)

Per-dataset configuration for `app/run_ea.py` — lives in
`data/datasets/<Name>/config-ea.json`. Every key below is verified against
the code (2026-06-11); keys not listed here are ignored by the EA. `comment`
keys are allowed anywhere (top level of a section or inside a parameter
spec) and are skipped by the operators.

Top-level sections:

| Section | Purpose |
|---|---|
| `use_nn` | Master switch for all neural-network features (bool) |
| `EA_SETTINGS` | Evolution scale: population, generations, seeds |
| `CALIBRATION` | Pre-G0 organization-threshold probe |
| `SEARCH_SPACE` | Typed hyperparameter specs the EA optimizes |
| `GENETIC_OPERATORS` | SBX/mutation/tournament parameters |
| `FIXED_PARAMS` | Constants passed to every SOM training + EA toggles |
| `NEURAL_NETWORKS` | Model paths and thresholds (active only with `use_nn: true`) |
| `PREPROCES_DATA` | Preprocessing settings (same semantics as SOM runs) |

---

## EA_SETTINGS

| Key | Default | Description |
|---|---|---|
| `population_size` | required | Individuals per generation. Realistic results need ≥ 30 (legacy `ISSUES.md` #18). |
| `generations` | required | Number of generations. Diversity decays after convergence (~gen 5–8) — prefer more seeds over more generations (`EA.md` §8). |
| `seeds` | `[FIXED_PARAMS.random_seed]` | List of seeds; each runs an independent evolution into `seed_<seed>/`. |

## CALIBRATION

| Key | Default | Description |
|---|---|---|
| `n_probes` | 15 | Quick SOM trainings before G0 to calibrate `org_threshold` (70th percentile of `max(u_matrix_max, dist_map_max)`). `0` disables → threshold 1.0. |
| `probe_epoch_multiplier` | 0.3 | Shortened training length for probes. |

## SEARCH_SPACE

Each entry is a typed spec; the operators per type are described in
`EA.md` §5.

```json
"start_learning_rate": { "type": "float", "min": 0.5, "max": 1.0, "log_scale": true }
"lr_decay_type":       { "type": "categorical", "values": ["linear-drop", "exp-drop", "log-drop", "step-down"] }
"num_batches":         { "type": "int", "min": 1, "max": 20 }
"map_size":            { "type": "discrete_int_pair" }
```

- `float` — uniform sampling/SBX/polynomial mutation in [min, max];
  `"log_scale": true` moves all three to log-space (use for learning rates
  and radius ratios — see `EA.md` §5).
- `int` — SBX/mutation + rounding.
- `categorical` — uniform swap / random replacement from `values`.
- `discrete_int_pair` — square map side `[s, s]`.
- A non-dict value is passed through as a fixed gene (not searched).

**Dynamic bounds**: `map_size` and `epoch_multiplier` may omit `min`/`max` —
`apply_dynamic_search_space()` always overwrites them from the dataset size
(Vesanto corridor; epoch-multiplier anchor table in `EA.md` §5). Any values
you write there are ignored.

Searched SOM hyperparameters in the standard configs:
`map_size`, `start_learning_rate`, `end_learning_rate`, `lr_decay_type`,
`start_radius_init_ratio`, `radius_decay_type`, `start_batch_percent`,
`end_batch_percent`, `batch_growth_type`, `epoch_multiplier`, `growth_g`,
`num_batches`. Parameter meanings are SOM semantics — see
`docs/som/CONFIG.md`.

Deliberately **not** searched (results of earlier experiments):
`map_type` (hex, fixed), `normalize_weights_flag` (true ⇒ 100 % penalized,
legacy `ISSUES.md` #26), `end_radius` (1.0), and the lower bounds of
`start_learning_rate`/`start_radius_init_ratio` are kept ≥ 0.5 so every
candidate retains a global organization phase (legacy `ISSUES.md` #52).

## GENETIC_OPERATORS

| Key | Default | Description |
|---|---|---|
| `sbx_eta` | 20.0 | SBX distribution index — higher = children closer to parents. |
| `mutation_eta` | 20.0 | Polynomial-mutation distribution index. |
| `mutation_prob` | 0.1 | Per-gene mutation probability. `0` disables mutation (ablation switch). |
| `tournament_k` | `max(2, population_size // 10)` | Tournament size; clamped to the population size at selection time (`issues.md` #91). |

(`crossover_prob` was removed 2026-06-11 — it was never read; SBX applies
per-gene with a fixed internal 50 % probability, `issues.md` #88.)

## FIXED_PARAMS

Everything here is merged into each individual's SOM parameters
(`{**genes, **FIXED_PARAMS}` — fixed params win on conflict). SOM-semantic
keys (`end_radius`, `map_type`, `mqe_evaluations_per_run`,
`early_stopping_window`, `max_epochs_without_improvement`,
`normalize_weights_flag`, `save_checkpoints`, `checkpoint_every_mqe`,
`checkpoint_count`, `random_seed`) are documented in `docs/som/CONFIG.md`.
EA-specific keys:

| Key | Default | Description |
|---|---|---|
| `max_archive_size` | 0 (off) | Cap on the result Pareto archive; trimmed by crowding distance. Reporting concern only — elitism does not depend on it (`issues.md` #87). Typical: 10–25. |
| `generate_training_plots` | false | Per-individual MQE/LR/radius/batch PNG plots (opt-in). |
| `generate_individual_maps` | false | Per-individual U-matrix/distance/dead-neuron PNGs + `maps_dataset/` + RGB composites. **CNN-legacy path** — the CNN track is closed; enable only to regenerate archival example data (`issues.md` #90). |
| `save_individual_weights` | false | Persist `weights.npy` per individual (nothing downstream reads it since the CNN closure). |

Notes:
- `save_checkpoints: true` + `checkpoint_every_mqe: true` is the standard
  setting — checkpoints are the LSTM training data and the source of
  `initial_mqe` for the MQE-ratio objective (without checkpoints the
  objective falls back to absolute `raw_best_mqe`).
- `org_threshold` is set internally from the calibration probe — do not put
  it in the config.

## NEURAL_NETWORKS

Active only when top-level `use_nn` is `true`; each hook is independent
(see `NN_INTEGRATION.md`):

| Key | Default | Description |
|---|---|---|
| `use_mlp` | false | MLP pre-screen: skip SOM training for configs with low predicted quality. Requires `mlp_filter_bad_configs: true` to actually skip. |
| `use_lstm` | false | LSTM early stopping of weak SOM trainings. |
| `use_lstm_controller` | false | Phase 3 dynamic LR/radius control (experimental — credit assignment problem, legacy `ISSUES.md` #79). |
| `use_cnn` | false | CNN visual quality as a 4th objective. **Closed track**; requires `generate_individual_maps: true`, otherwise a warning is printed and the score stays empty. |
| `mlp_model_path`, `mlp_scaler_path`, `lstm_model_path`, `lstm_scaler_path`, `cnn_model_path`, `lstm_controller_model_path`, `lstm_controller_scaler_path` | null | Explicit model paths; `null` auto-detects `*_latest.keras` / newest `*_best.keras` in `app/{mlp,lstm,cnn}/models/`. |
| `mlp_filter_bad_configs` | false | Enable the actual skipping (with `use_mlp`). |
| `mlp_bad_quality_threshold` | 0.5 | Skip when predicted `raw_mqe_improvement_ratio` < threshold (**higher prediction = better** — the comparison direction matters, legacy `ISSUES.md` #58). |
| `lstm_quality_threshold` | 1.0 | Stop training when predicted badness score exceeds this (lower score = better; legacy `ISSUES.md` #57/#68). |
| `verbose` | false | NN status messages. |

## PREPROCES_DATA

Identical semantics to the SOM pipeline (`docs/som/CONFIG.md`): `delimiter`,
`primary_id`, `categorical_threshold_numeric`, `categorical_threshold_text`,
`noise_threshold_ratio`. Preprocessing runs once per EA run and is shared by
all seeds.
