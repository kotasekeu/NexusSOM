# EA Module — NSGA-II Hyperparameter Optimization

Multi-objective evolutionary optimization of SOM training hyperparameters.
This document describes the algorithm **as implemented** — verified against the
code on 2026-06-11 (see `CLEANUP_PLAN.md` for the verification findings).

Code map:

| File | Role |
|---|---|
| `app/ea/ea.py` | NSGA-II loop, evaluation, constraints, calibration, logging, CLI (`main`) |
| `app/ea/genetic_operators.py` | SBX crossover, polynomial mutation, random init — mixed/log-scale variants |
| `app/ea/nn_integration.py` | Optional MLP/LSTM/controller hooks (see `NN_INTEGRATION.md`) |
| `app/ea/analyze_pareto_fronts.py` | Post-hoc comparison of fronts across runs |
| `app/run_ea.py` | Repo-root CLI wrapper |

Configuration lives per dataset in `data/datasets/<Name>/config-ea.json`
(see `CONFIG.md`). Entry point:

```bash
.venv/bin/python3 app/run_ea.py -i data/datasets/Iris/iris.csv -c data/datasets/Iris/config-ea.json
```

---

## 1. Problem formulation

**What is optimized:** SOM training hyperparameters — map size, learning-rate
schedule, neighborhood-radius schedule, epoch count, batch growth, decay curve
types.

**Why evolutionary:** the space is mixed (continuous floats, integers,
categoricals), the fitness is a black box (requires a full SOM training run),
and no gradient exists.

**Why multi-objective:** the three quality metrics are in a natural trade-off.
The result is a Pareto front of compromise configurations, not a single
"best" solution.

**Two roles in the project** (see `docs/global/COMPONENTS.md`):
1. Search/comparison tool for the ablation study (Pareto fronts, hypervolume,
   EA-optimized vs deterministic baselines — e.g. the SwissRoll/Helix
   layer-bridging question, `docs/som/BENCHMARKS.md`).
2. Bulk generator of training data for the MLP/LSTM models (multi-seed runs,
   per-individual checkpoints).

---

## 2. Objectives (Pareto goals)

All three objectives: **lower = better**. Raw measured values enter dominance —
never penalized/modified ones (penalties act through constraints, §3).

| # | Objective | Formula | Measures |
|---|---|---|---|
| 1 | `raw_mqe_ratio` | `MQE_final / MQE_initial` | Relative quantization improvement vs random init (checkpoint[0]); map-size and dataset independent |
| 2 | `raw_te` | share of samples whose 1st and 2nd BMU are not grid neighbors | Local topology preservation (hex: cube-distance 1; square: Moore) |
| 3 | `1 − ρ` | Spearman correlation of pairwise weight distances (data space) vs pairwise physical neuron distances (grid space) | Global topology / manifold unfolding — detects globally folded maps that TE cannot see |

`dead_neuron_ratio` is **not** an objective — it is a constraint (§3). As an
objective it created selection pressure toward small maps (legacy `ISSUES.md`
#85). With `use_cnn: true` a fourth objective `1 − cnn_quality_score` is added;
the CNN track is **closed** (see `docs/cnn/CNN_REQUIREMENTS.md`) and this path
is kept only for archival data regeneration.

Implementation: objectives assembled in `run_evolution()`; `raw_mqe_ratio`
computed in `evaluate_individual()` from checkpoint[0]; TE and ρ are
`KohonenSOM.calculate_topographic_error()` /
`calculate_topological_correlation()` in `app/som/som.py`.

---

## 3. Constraints — constrained dominance (Deb 2002)

Instead of multiplying penalties into objectives, infeasible solutions are
handled by **constrained dominance** in `_dominates()`:

1. A feasible solution (CV = 0) always dominates an infeasible one.
2. Between two infeasible: smaller `constraint_violation` dominates.
3. Between two feasible: standard Pareto dominance on the 3 raw objectives.

`compute_constraint_violation()` aggregates one scalar CV:

- **Organization constraint**: `org_cv = max(0, max(u_matrix_max,
  distance_map_max) − org_threshold)`. The threshold is calibrated per dataset
  by the analytic probe (§7) — a static 1.0 penalized ~100 % of normal runs.
- **Dead-neuron constraint** with a coverage-derived dynamic threshold
  `clamp(1 − coverage_ratio/10, 0.30, 0.85)` where
  `coverage_ratio = n_samples / (m·n)`, and graduated bands above it:

  | `dead_excess` | factor |
  |---|---|
  | < 0.2 | × 1.5 |
  | < 0.4 | × 2.5 |
  | ≥ 0.4 | × 5.0 |

`penalty_factor = 1 + CV` and `penalty_reason` are logged for diagnostics
only; raw objective values are never altered.

---

## 4. The generation loop

`run_evolution()` — canonical NSGA-II (μ + λ) with a result archive:

```
P₀ = population_size random configs (validate_and_repair applied), evaluated in parallel
survivors = ∅
for each generation:
  1. evaluate current population (Pool, apply_async; gen 0: P₀, later: offspring only)
  2. combined = evaluated ∪ survivors ∪ ARCHIVE, deduplicated by uid
     (survivors carry their results — no re-evaluation of parents)
  3. constrained non-dominated sort → fronts F₀, F₁, …
  4. crowding distance per front; rank + distance attached to each individual
  5. environmental selection: combined sorted by (rank, −crowding);
     first N become survivors P_{g+1} (kept WITH results)
  6. ARCHIVE = rank-0 members, deduplicated by uid, capped at
     max_archive_size by crowding distance; logged to pareto_front.csv
  7. mating pool: N × tournament_selection(P_{g+1}, k)
  8. offspring: SBX crossover + polynomial mutation + validate_and_repair,
     generated until N unique UIDs (or population_size×3 attempts)
  9. population ← offspring
```

Notes on the implementation:

- **Elitism** flows through `survivors` (full N best of P ∪ O, canonical
  NSGA-II) *and* the `ARCHIVE` (permanent rank-0 memory, capped). Until
  2026-06-11 survivors were discarded and only the capped archive re-entered
  the combine step — fixed, see `issues.md` #87.
- **Tournament selection**: lower rank wins; tie → larger crowding distance.
  `tournament_k` from config (`GENETIC_OPERATORS.tournament_k`), default
  `max(2, population_size // 10)`.
- **Evaluation failures** (exception in a worker) drop that individual from
  the generation; the loop continues with the rest.
- There is no cross-process evaluation cache — duplicates are prevented
  *before* evaluation by the UID dedup (§6); `results.csv` is the
  authoritative record of what was evaluated.

---

## 5. Search space

Defined in `config-ea.json` → `SEARCH_SPACE` as typed specs
(`{"type": ..., "min": ..., "max": ...}`); see `CONFIG.md` for the full
reference. Parameter types and their operators:

| Type | Init | Crossover | Mutation |
|---|---|---|---|
| `float` | uniform | SBX (η = `sbx_eta`) | polynomial (η = `mutation_eta`) |
| `float` + `log_scale: true` | log-uniform | SBX on `log(x)`, result `exp(c)` | polynomial on `log(x)` |
| `int` | randint | SBX + round | polynomial + round |
| `categorical` | choice | uniform swap (50 % per gene) | random replacement |
| `discrete_int_pair` (`map_size`) | square `[s, s]` | SBX on side + round | polynomial on side + round |

**Log-scale sampling** is used for `start/end_learning_rate` and
`start_radius_init_ratio`: the difference 0.01→0.02 is +100 % while
0.51→0.52 is +2 %; linear-space operators would undersample small values.

**Repair** (`validate_and_repair`, applied after init, crossover and
mutation): swap `start_lr < end_lr`, swap `start_batch > end_batch`, swap
radius bounds, clamp `epoch_multiplier ≥ 0.1` (float — dynamic bounds go
below 1.0 for large datasets), `growth_g = 0` iff all decay/growth curves are
linear (prevents functionally identical duplicates), `num_batches ≥ 1`.

### Dynamic bounds (computed once per run from the dataset)

`apply_dynamic_search_space()` overrides two specs:

- **`map_size`** — Vesanto & Alhoniemi (2000) heuristic: optimal neuron count
  `U = 5√n_samples`, side `√U`, search corridor
  `[max(8, 0.7·side), 1.3·side]`. Example: 569 samples → [8, 14].
- **`epoch_multiplier`** — log-linear interpolation between empirical anchors
  (this replaces two outdated descriptions in older docs):

  | n_samples | em_min | em_max |
  |---|---|---|
  | ≤ 100 | 1.0 | 5.0 |
  | 500 | 1.0 | 5.0 |
  | 5 000 | 0.5 | 3.0 |
  | 20 000 | 0.3 | 1.0 |
  | ≥ 50 000 | 0.1 | 0.3 |

  Values between anchors are interpolated in log-n space; total iterations =
  `epoch_multiplier × n_samples` stay in a comparable budget across dataset
  sizes.

**Fixed (not searched)**: `map_type` (hex), `end_radius = 1.0`,
`normalize_weights_flag = false` (searching `true` produced 100 % penalized
runs — legacy `ISSUES.md` #26), `start_radius_init_ratio` min raised to 0.5
so every run keeps a global organization phase (legacy `ISSUES.md` #52).

---

## 6. Deduplication and UIDs

Every configuration has a deterministic **gene-only** UID = MD5 of its sorted
parameter items (`get_uid`); NSGA-II selection metadata (`rank`,
`crowding_distance`) is excluded from the hash, so the same genes hash
identically before and after ranking (`issues.md` #89). Offspring are
generated **inside** the reproduction loop until exactly `population_size`
unique UIDs exist (duplicates are replaced, not dropped — legacy `ISSUES.md`
#51), seeded with the UIDs of current survivors and archive members so no
living configuration is ever re-evaluated. The combine step deduplicates
survivors/archive/offspring overlap by UID as a safety net.

---

## 7. Calibration probe (before G0)

`run_calibration_probe()` — `n_probes` (default 15) quick SOM trainings
(`epoch_multiplier` forced to `probe_epoch_multiplier`, default 0.3) on random
configs from the same search space, in parallel. `org_threshold` = 70th
percentile of `max(u_matrix_max, dist_map_max)` over the probes; results in
`calibration_probe.csv`. Disabled with `n_probes: 0` → fallback threshold 1.0.

Rationale: typical healthy values are 1.02–1.16; a static threshold 1.0
penalized everything (legacy `ISSUES.md` #39/#40/#46). The threshold is
calibrated once and stays constant for the whole run — dynamic per-generation
thresholds are architecturally incompatible with NSGA-II (archive solutions
would be ranked under different rules; legacy `ISSUES.md` #36).

---

## 8. Multi-seed strategy

`EA_SETTINGS.seeds` (e.g. `[42, 1337, 7, 101, 2026]`) runs the whole
evolution N× into `results/<timestamp>/seed_<seed>/`. Preprocessing, dynamic
search space and the calibration probe run **once**, shared by all seeds.

Rationale: after convergence (~gen 5–8) SBX produces offspring in a narrow
region; one long run yields many low-diversity evaluations. Several shorter
independent runs cover the space better — important for NN training data
(legacy `ISSUES.md` #53/#54).

---

## 9. Pareto metrics

Logged per generation into `pareto_metrics.csv` from **feasible** archive
members (`_compute_pareto_metrics`):

| Metric | Description |
|---|---|
| `hv` | Hypervolume vs reference point [1.1, 1.1, 1.1] in normalized space (pymoo) |
| `spacing` | Std of nearest-neighbor distances on the front; 0 = perfectly uniform |
| `spread_mqe/te/dead` | max − min per dimension in normalized space; ≈1 = front spans the observed range |

Normalization uses a **global running min/max** updated each generation
(reset per seed) — per-generation min-max would break cross-generation
comparability, raw clipping would let dimensions with larger absolute ranges
dominate HV (legacy `ISSUES.md` #65).

---

## 10. Evaluation of one individual

`evaluate_individual()` (runs in a Pool worker):

1. Optional MLP pre-screen: skip SOM training when predicted
   `raw_mqe_improvement_ratio < mlp_bad_quality_threshold` (higher = better).
2. Train `KohonenSOM` with `{**genes, **FIXED_PARAMS}`; optional LSTM
   early-stop and LSTM-controller callbacks attached.
3. Persist per-individual artifacts into `individuals/<uid>/`: training
   checkpoints (LSTM training data), sample coverage; weights only when
   `save_individual_weights: true`.
4. Measure: TE, topological correlation ρ, U-matrix metrics, dead neurons,
   distance-map max; compute CV and improvement ratios.
5. Log to `results.csv` / `status.csv` / `log.txt`; per-individual training
   plots and PNG maps are **opt-in** (`generate_training_plots`,
   `generate_individual_maps`, both default `false` — the maps feed only the
   closed CNN track; enable for archival data regeneration).

A failed evaluation drops only that individual; a fatal error in the
generation loop itself re-raises and fails the run with a non-zero exit
(`issues.md` #91).

---

## 11. Output files (per seed directory)

| File | Content |
|---|---|
| `pareto_front.csv` | Archive snapshot per generation — raw objectives, CV, penalty metadata, map dims, `ds_*` dataset stats, all search-space params |
| `pareto_metrics.csv` | HV, spacing, spread per generation |
| `results.csv` | Every evaluated individual (one row per evaluation) |
| `status.csv` | Evaluation lifecycle (started/completed/failed/cached) |
| `log.txt` | Per-UID event log |
| `individuals/<uid>/` | Checkpoints, coverage, optional plots/maps/weights |
| `maps_dataset/` | U-matrix/distance/dead-neuron PNGs + RGB composites (CNN legacy path; active only with `generate_individual_maps`) |

Shared in the run root: `calibration_probe.csv`, preprocess artifacts,
`dataset_meta.json`.

Post-hoc tools: `app/tools/verify_ea_run.py` (run sanity verification),
`app/ea/analyze_pareto_fronts.py` (cross-run front comparison).

---

## 12. References

- Deb, K. et al. (2002). *A Fast and Elitist Multiobjective Genetic
  Algorithm: NSGA-II.* IEEE TEC 6(2). — algorithm + constrained dominance
  (Section III-B).
- Deb, K. & Agrawal, R. B. (1995). *Simulated Binary Crossover for Continuous
  Search Space.* Complex Systems 9(2). — SBX + polynomial mutation.
- Vesanto, J. & Alhoniemi, E. (2000). *Clustering of the Self-Organizing
  Map.* IEEE TNN 11(3). — `5√n` map-size heuristic.
- Kiviluoto, K. (1996). *Topology Preservation in Self-Organizing Maps.*
  ICNN. — topographic error.
