# SOM Module — Test Suite Guide

121 tests cover the SOM module (out of ~250 in the whole project). They were
written as a **baseline before the module refactor** (see `REFACTOR_PLAN.md`)
and now serve as the permanent safety net: any change to the module must keep
them green, and intentional contract changes must update them consciously.

## How to run

Always from the repo root, with the project venv:

```bash
# Whole project suite (~8 s)
.venv/bin/python3 -m pytest

# SOM module only
.venv/bin/python3 -m pytest tests/unit/test_som_core.py tests/unit/test_som_preprocess.py \
    tests/unit/test_som_analysis.py tests/unit/test_som_visualization.py \
    tests/unit/test_som_persistence.py tests/integration/test_som_pipeline.py \
    tests/integration/test_som_run_pipeline.py

# One file / one class / one test
.venv/bin/python3 -m pytest tests/unit/test_som_core.py
.venv/bin/python3 -m pytest tests/unit/test_som_core.py::TestDecay
.venv/bin/python3 -m pytest tests/unit/test_som_core.py::TestDecay::test_drop_types_monotonically_decrease
```

Expected result: **all green**, no skips, no network, no GPU. Everything runs
on synthetic data in pytest temp dirs — no project files are touched. The two
integration files spawn real pipeline runs and take a few seconds each;
everything else is sub-second.

## Test files

| File | Tests | What it covers |
|---|---|---|
| `tests/unit/test_som_core.py` | 43 | `KohonenSOM`: init (map size, radius, seed reproducibility, hex cube coords), all decay types incl. **monotonic-decrease guarantee** for `*-drop` curves (domain constraint: LR/radius must never rise), BMU search and weight updates with/without mask, metrics (QE, topographic error on organized vs. scrambled maps, dead neurons, U-matrix stats, topological correlation), and the `train()` contract: returned keys, MQE improvement, checkpoints, coverage counts, **no disk writes**, fully-masked weight dims zeroed, partially-masked dims keep training. |
| `tests/unit/test_som_preprocess.py` | 22 | `preprocess_data`: input validation (missing file/columns, empty CSV), purity (no disk writes, no config mutation), column classification (numeric/categorical/noise, primary ID excluded from features), NaN masking + median fill, normalization range, `dataset_stats` contract, fully-masked columns zeroed, and the three **`preprocess_strategy`** behaviors (`nexus` / `scale-only` / `none`). |
| `tests/unit/test_som_analysis.py` | 15 | `perform_analysis` outputs (clusters, QE, extremes, sample assignments incl. per-dim QE), BMU assignment **honoring the ignore mask** (a median-filled value must not move a sample to a different neuron than training), masked dims having zero per-dim QE, outlier detection, plus `som/utils.py` (config loading, logging, working dir). |
| `tests/unit/test_som_visualization.py` | 12 | Smoke rendering of every map type from `(weights, map_type)` — square and hex — legends, `generate_all_maps` orchestration, training plots, and **`render_results_dir`**: re-rendering a stored run purely from artifacts (the UI/ablation path). File existence is asserted, not pixel content. |
| `tests/unit/test_som_persistence.py` | 11 | The persistence layer: weights npy + readable CSV roundtrip, checkpoints/coverage/run-metrics writes, no-op behavior on empty inputs, `save_preprocess_artifacts` layout — plus `compute_u_matrix` **equivalence with the original loop implementation** for both square and hex grids. |
| `tests/integration/test_som_pipeline.py` | 8 | **End-to-end CLI run** (`app/run_som.py` as subprocess) on a synthetic two-cluster dataset: exit code, the complete **results-directory layout** (this test *is* the output contract — see `RESULTS.md`), run-metrics content, weights shape, cluster coverage, cluster separation on the map, `llm_context.json` contract incl. `spatial_quality_score`. |
| `tests/integration/test_som_run_pipeline.py` | 10 | The programmatic API: `run_pipeline()` artifacts, seed override without config mutation, **same seed → bit-identical weights**, different seeds differ, config-as-path, `preprocess_strategy` recorded in metadata — and the **multi-seed tool** end to end (3 seeds via subprocess: metrics CSV, summary with mean/std and pairwise ARI, MQE comparison plot). |

## Conventions

- **Characterization first.** Tests capture observable behavior and contracts
  (return keys, file layouts, metric semantics), not implementation details.
  When a refactor breaks one, first ask whether the *contract* was supposed
  to change.
- **Contracts with teeth.** Two tests are deliberately strict and should not
  be loosened casually: `test_som_pipeline.py::test_output_layout` (the
  results-dir layout consumed by analysis/API/UI/tools) and
  `test_train_writes_nothing_to_disk` (core purity — persistence is the
  caller's job).
- **Shared helpers.** `make_som(**overrides)` in `test_som_core.py` builds a
  small fully-configured SOM (6×6, `show_progress=False`) and is reused by
  other test files; `fast_config()` in `test_som_run_pipeline.py` is the
  minimal fast pipeline config (plots/maps disabled).
- Tests insert `app/` into `sys.path` themselves — they run with plain
  `pytest` from the repo root (`pytest.ini` sets `testpaths = tests`).

## What is intentionally NOT tested

- Pixel content of rendered maps (only that files render without error).
- LSTM/MLP integration paths in `run.py` (optional NN dependencies; covered
  by graceful-degradation behavior — a missing model logs a warning and the
  run continues).
- The disabled early-stopping mechanism (`issues.md` #2) — to be covered when
  it is recalibrated and re-enabled.
