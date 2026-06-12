# Running the Evolutionary Algorithm

One EA run = validate → preprocess (once) → dynamic search-space bounds →
calibration probe → N independent seeded NSGA-II evolutions, each writing
its own `seed_<seed>/` directory (see `RESULTS.md`). Algorithm details:
`EA.md`; configuration reference: `CONFIG.md`.

Always use the project venv interpreter `.venv/bin/python3` from the repo root.

## CLI

```bash
.venv/bin/python3 app/run_ea.py -i data/datasets/Iris/iris.csv -c data/datasets/Iris/config-ea.json
```

| Argument | Required | Description |
|---|---|---|
| `-i`, `--input` | yes | Input CSV file. Results land in `results/<YYYYMMDD_HHMMSS>/` next to it. |
| `-c`, `--config` | yes | JSON configuration (see `CONFIG.md`). |

Both arguments are required since the 2026-06-11 cleanup (`issues.md` #88) —
the synthetic-data fallback and the `ea_config.py` module fallback were
removed. A fatal error inside the evolution loop exits non-zero with a full
traceback (`issues.md` #91); `Ctrl+C` stops gracefully and keeps the CSVs
written so far.

## What happens before generation 0 (shared across seeds)

1. **Validation + preprocessing** (`som.preprocess`) — same pipeline as a
   SOM run; artifacts saved into the run root (`csv/`, `json/`,
   `dataset_meta.json`).
2. **Dynamic search space** — `map_size` bounds from the Vesanto heuristic,
   `epoch_multiplier` bounds from log-interpolated anchors (`EA.md` §5).
3. **Calibration probe** — `CALIBRATION.n_probes` quick SOM trainings
   calibrate `org_threshold` (70th percentile); `calibration_probe.csv`.
   Skip with `"n_probes": 0`.
4. **Dataset metadata injection** — `ds_*` stats from `dataset_meta.json`
   are attached to every `results.csv` row (training data for the MLP).

## Multi-seed runs

```json
"EA_SETTINGS": { "population_size": 50, "generations": 6, "seeds": [42, 1337, 7, 101, 2026] }
```

Each seed runs an independent evolution into `seed_<seed>/` (fresh archive,
fresh normalization stats). When `seeds` is omitted, the single
`FIXED_PARAMS.random_seed` is used. Rationale — post-convergence generations
produce low-diversity evaluations; several shorter independent runs cover
the space better (`EA.md` §8).

## Typical workflows

```bash
# Hyperparameter search for one dataset (fast: no per-individual artifacts beyond checkpoints)
.venv/bin/python3 app/run_ea.py -i data/datasets/BreastCancer/breast-cancer.csv \
    -c data/datasets/BreastCancer/config-ea.json

# NN training-data collection: 5 seeds × 50 × 6 (see EA.md §8)
# → ~1500 evaluations with checkpoints in individuals/<uid>/csv/

# Ablation A3: EA on a benchmark with ground truth, then verify the best
# configurations with the SOM topology tools (docs/som/RUN.md)
.venv/bin/python3 app/run_ea.py -i data/datasets/SwissRoll/swiss_roll.csv \
    -c data/datasets/SwissRoll/config-ea.json
```

Useful `FIXED_PARAMS` toggles for bulk runs (all default off — see
`CONFIG.md`): `generate_training_plots`, `generate_individual_maps`
(CNN-legacy PNG maps + `maps_dataset/` + RGB composites),
`save_individual_weights`.

## After the run

```bash
# Sanity verification of one seed directory (8 diagnostic sections)
.venv/bin/python3 app/tools/verify_ea_run.py <run>/seed_42

# End-to-end smoke test of the EA pipeline itself (used as the cleanup gate)
EA_SMOKE=1 .venv/bin/python3 -m pytest tests/integration/test_ea_smoke.py
```

See `VERIFICATION.md` for the full verification toolbox and `RESULTS.md`
for every output file.
