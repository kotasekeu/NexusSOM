# budget_saturation.py — experiment D driver + report

Measures **where map quality saturates as a function of the training
budget**, separately per processing regime, on real SOM runs. Built for
`docs/ea/SEARCH_SPACE.md` **step 2** (the `epoch_multiplier` decision);
the protocol and pre-registered decision criteria live there — this page
documents the tool.

## Why per regime

`total_iterations = N · epoch_multiplier` and each iteration processes
`ceil(N · pct / 100)` samples, so `epoch_multiplier` means something
different in every regime (see the table in SEARCH_SPACE step 2). The
comparable quantity is the **per-sample exposure budget P** (how many
times each sample updates the map; exact under the `reshuffle` default):

| Regime | Batch profile | Budget unit | em derivation |
|---|---|---|---|
| deterministic | 100 % static | epochs E (= exposures) | `em = (E + 0.5)/N` |
| stochastic | 0.01 % static (1 sample/iter) | em (= exposures) | `em = P` |
| hybrid | 1→5 % linear-growth | nominal exposures B | `em = (round(B·100/3) + 0.5)/N` |

The `+0.5` absorbs the `int()` truncation in `total_iterations` so the
iteration count lands exactly. Analysis always uses the **measured**
exposures (mean of `sample_coverage.json` counts), never the nominal
budget (hybrid `ceil` quantization inflates small datasets by ~10 %).

## `run` — execute the grid (resumable)

```bash
.venv/bin/python3 app/tools/budget_saturation.py run \
    [--datasets Iris,SwissRoll,Helix,WineQuality] \
    [--regimes det,stoch,hyb] [--seeds 10] [--dry-run]
```

Grid: deterministic E ∈ {3, 10, 30, 100, 300, 1000}, stochastic
em ∈ {1, …, 1000}, hybrid B ∈ {3, …, 1000}; 19 arms × 10 seeds per
dataset. Per arm it generates a config from the experiment-C base config
(`config-som-expC-reshuffle-em10.json` — only the batch profile,
`epoch_multiplier`, and bookkeeping keys are overridden; config snapshot
saved into the arm directory), runs `multi_seed_som.py`, and on
ground-truth datasets (SwissRoll, Helix) runs `verify_topology.py` on
every seed. **Resume**: arms with a finished `multi_seed_summary.json`
are skipped, missing `verify_topology.json` files are filled in — safe to
re-run after an interruption. Note `verify_topology.py` exits non-zero on
a FAIL *verdict* (normal on Helix); the driver detects real failures by
the missing report file instead.

Output: `data/datasets/<ds>/results/expD_<regime>-<budget>/`.

## `report` — aggregate, evaluate criteria, plot

```bash
.venv/bin/python3 app/tools/budget_saturation.py report [--datasets ...]
```

Reads whatever arms have finished and writes into `data/expD/`:

- `expD_metrics.csv` — one row per run (metrics + measured exposures +
  ground-truth ρ/R² where available),
- `expD_<dataset>_curves.png` — MQE, TE, dead_ratio, T&C, ground-truth ρ,
  ARI stability vs measured exposures (log x), median + IQR band per
  regime,
- `expD_<dataset>_overtraining.png` — the under/overtraining exhibit:
  MQE (left axis) vs TE and ground-truth ρ (right axis) per regime;
  divergence right of the MQE plateau = overtraining,
- `expD_report.md` — per-arm tables plus the **saturation verdict table**
  evaluating the pre-registered criteria: b\* = smallest budget from
  which every further grid step improves median MQE by < 2 % per
  doubling of budget with no topology metric significantly worse
  (Mann-Whitney p < 0.05) than at the best-MQE budget; `—` means
  saturation was not reached at the largest tested budget.

The report is purely mechanical; interpretation and the fixed-value
decision are recorded in `docs/ea/SEARCH_SPACE.md` (step 2 results).
