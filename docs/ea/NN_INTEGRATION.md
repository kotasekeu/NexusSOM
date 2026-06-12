# Neural-Network Hooks in the EA

How the optional MLP/LSTM models plug into the EA evaluation loop.
Code: `app/ea/nn_integration.py` (loading + prediction wrappers) and
`evaluate_individual()` in `app/ea/ea.py` (hook call sites). The models
themselves (architectures, training pipelines) belong to `app/mlp/` and
`app/lstm/` — see `docs/mlp/` and `docs/lstm/`; this document covers the EA
side only.

All hooks are optional and independent; the EA is fully functional without
TensorFlow installed (`python/requirements_nn.txt`). Activation requires the
top-level `"use_nn": true` **and** the per-hook flag in `NEURAL_NETWORKS`
(see `CONFIG.md`).

| Hook | Flag | Effect in the EA |
|---|---|---|
| MLP "Oracle" pre-screen | `use_mlp` + `mlp_filter_bad_configs` | Skips SOM training for configurations with low predicted quality |
| LSTM early stopping | `use_lstm` | Aborts a running SOM training when the predicted final quality is poor |
| LSTM dynamic controller | `use_lstm_controller` | Adjusts LR/radius factors at every MQE checkpoint (Phase 3, experimental) |
| CNN visual quality | `use_cnn` | Adds `1 − cnn_quality_score` as a 4th objective — **closed track** |

## Model loading

`NeuralNetworkIntegration` lazy-loads TensorFlow only when a hook is
enabled; if TF or a model file is missing, the hook deactivates itself with
a warning instead of failing the run. Models are cached **per worker
process** (`_get_nn_integration` in `ea.py`) so each Pool worker loads them
once.

Path resolution when the config paths are `null`: stable name first
(`app/<module>/models/*_latest.keras` + `*_scaler_latest.pkl` +
`*_latest_metadata.json`), otherwise the newest `*_best.keras`. Metadata
JSON is required by the MLP for feature encoding (candidate-path list —
legacy `ISSUES.md` #59).

## MLP pre-screen (Phase 2)

Called before SOM training in `evaluate_individual()`:

1. `encode_config_for_mlp()` builds the feature vector from the metadata's
   `feature_columns`: numeric hyperparameters, `map_rows`/`map_cols` parsed
   from `map_size`, one-hot decay/growth types, and `ds_*` dataset stats
   (injected into `FIXED_PARAMS` from `dataset_meta.json`).
2. The model predicts `(raw_mqe_improvement_ratio, raw_topographic_error,
   dead_neuron_ratio)`; the EA uses prediction[0].
3. Skip condition: `pred < mlp_bad_quality_threshold` — **higher predicted
   ratio = better**; the inverted comparison was a real bug once (legacy
   `ISSUES.md` #58).

A skipped individual is logged to `status.csv` as `mlp_skipped` and enters
the population as infeasible (`constraint_violation = 999`,
`penalty_reason = mlp_prescreened`) — it can still pass genes on, but never
reaches the archive.

## LSTM early stopping (Phase 2)

`evaluate_individual()` builds a callback passed to `som.train()`. The SOM
collects checkpoints independently of `save_checkpoints` (legacy
`ISSUES.md` #55) and calls the callback once enough prefix exists
(≥ 20 % of `mqe_evaluations_per_run`; legacy `ISSUES.md` #56).

The callback normalizes the sequence — `(progress, mqe/mqe₀, TE,
dead_ratio, lr/lr₀, radius/radius₀)` per checkpoint — and feeds it to the
hybrid model together with the dataset context vector `[n_samples,
n_active_dims, n_numeric, n_categorical]` (scaled by the stored scaler;
sequence-only legacy models still work without it). The decision:

```
badness = (1 − pred_mqe_ratio) + pred_te + pred_dead × 0.5
stop when badness > lstm_quality_threshold
```

Lower badness = better predicted outcome; the formula direction and
threshold calibration have their own history (legacy `ISSUES.md` #57, #68).

## LSTM dynamic controller (Phase 3 — experimental)

`get_dynamic_schedule_fn()` returns a closure that accumulates the
checkpoint sequence and at each MQE checkpoint predicts
`(lr_factor, radius_factor)` for the most recent timestep; `som.py` applies
them multiplicatively with physical clamps. Status: technically functional
end-to-end, but the trained model collapses to ≈1.0 factors — trajectory-level
advantages cannot solve the per-timestep credit assignment problem (legacy
`ISSUES.md` #79, options A/B/C documented there). Treat as an open research
branch, not a production hook.

## CNN visual quality (closed)

With `use_cnn: true` the EA composes the three per-individual maps into an
RGB image and adds `1 − cnn_quality_score` as a 4th NSGA-II objective.
The CNN track is **closed** (`docs/cnn/CNN_REQUIREMENTS.md`); the hook needs
`generate_individual_maps: true` (default off since `issues.md` #90) and
exists only so the closure exhibit can be regenerated.

## Verifying a NN-enabled run

```bash
# How many individuals the MLP filtered:
grep -c mlp_skipped <run>/seed_*/status.csv

# LSTM stops are visible in the seed log:
grep -i "early stop" <run>/seed_*/log.txt

# Compare archive quality vs a no-NN baseline run:
# raw_mqe_ratio / raw_te / raw_topo_corr in pareto_front.csv must not degrade
```

The honest comparison protocol (NN-on vs NN-off on the same dataset and
seeds) is part of the ablation plan — `docs/global/ABLATION_STUDY.md`.
