# EA System Design

Evolutionary optimization of Kohonen SOM hyperparameters using NSGA-II with constrained dominance.
Implementation: `app/ea/ea.py`, operators: `app/ea/genetic_operators.py`.

---

## Architecture Overview

Standard NSGA-II (Deb et al. 2002) with three modifications: constrained dominance instead of penalty
multiplication, calibrated constraint thresholds from pre-run probe, and dynamic search space bounds
derived from dataset size. Evaluation is a full SOM training cycle per individual; parallelized via
`multiprocessing.Pool`.

---

## Execution Flow

```
main()
  ├── preprocess_data()                  # normalize, build ignore_mask
  ├── apply_dynamic_search_space()       # adjust map_size bounds from n_samples
  ├── run_calibration_probe()            # N quick SOM runs → org_threshold (p70)
  └── run_evolution()
        for each generation:
          ├── deduplicate population by UID
          ├── Pool.map(evaluate_individual)
          ├── combined_population = evaluated + ARCHIVE
          ├── non_dominated_sort(objectives, violations)
          ├── crowding_distance_assignment()
          ├── ARCHIVE ← rank==0, UID-deduped, capped at max_archive_size
          ├── log_pareto_front()         → pareto_front.csv
          └── tournament_selection → SBX crossover → polynomial mutation → repair
```

---

## Pre-run Initialization

### Dynamic Search Space
Map size bounds are not fixed. Before G0, `U = 5·√n_samples` (Vesanto heuristic) gives the
optimal neuron count; map side range is set to `[max(8, round(√U·0.7)), round(√U·1.3)]`.
Eliminates structurally inappropriate extremes (e.g. 5×5 or 25×25 for a 569-sample dataset).

### Calibration Probe
15 quick SOM runs (configurable; `epoch_multiplier=0.3` injected via `fixed_params`) sampled from
the adjusted search space. The probe overrides `epoch_multiplier` through `fixed_params`, which
takes precedence over `ind` in `som_params = {**ind, **fixed_params}` — the repair function
operates only on search space individuals and does not affect this override.
The 70th percentile of `max(u_matrix_max, dist_map_max)` across probes becomes `org_threshold`,
injected into `fixed_params` before the main loop. This adapts the organization constraint to the
dataset and map size instead of using a fixed value. Results logged to `calibration_probe.csv`.

---

## Search Space & Encoding

| Parameter | Type | Notes |
|---|---|---|
| `map_size` | `discrete_int_pair` | Square maps only; bounds set dynamically |
| `start_learning_rate`, `end_learning_rate` | `float` (log-scale) | SBX + mutation in log-space |
| `start_radius`, `end_radius` | `float` (log-scale) | Same |
| `epoch_multiplier`, `num_batches` | `int` / `float` | Standard polynomial mutation |
| `lr_decay_type`, `radius_decay_type`, `batch_growth_type` | `categorical` | Uniform swap crossover |
| `start_batch_percent`, `end_batch_percent` | `float` | Linear-space SBX |
| `growth_g` | `float` | Fixed to 0 when all decay types are linear (phenotype dedup) |

Continuous parameters with exponential influence (LR, radius) use log-space SBX and mutation to
preserve proportional symmetry across orders of magnitude.

---

## NSGA-II Objectives

Three raw (unmodified) metrics, all minimized:

| Objective | Meaning |
|---|---|
| `raw_mqe_improvement_ratio` | `final_mqe / initial_mqe` — ratio vs. random-init baseline (checkpoint[0]) |
| `raw_topographic_error` | Fraction of samples whose 2nd BMU is non-adjacent to 1st BMU |
| `dead_neuron_ratio` | Fraction of neurons that are BMU for zero training samples |

`mqe_improvement_ratio` is dataset- and map-size-agnostic; a value < 1 means the SOM improved
beyond random initialization. Duration is **not** an objective — it is logged as metadata only.

---

## Constrained Dominance (Deb 2002)

Replaces multiplicative penalty. A scalar `constraint_violation` (CV) is computed per individual:

```
cv = org_cv + dead_cv

org_cv  = max(0, max(u_matrix_max, dist_map_max) - org_threshold)

dead_threshold = clamp(1 - (n_samples / neurons) / 10, 0.30, 0.85)
dead_excess    = max(0, dead_ratio - dead_threshold)
dead_cv        = dead_excess × {1.5 if excess < 0.2; 2.5 if < 0.4; 5.0 otherwise}
```

`dead_threshold` is coverage-ratio based: small maps (many samples/neuron) get a strict threshold;
large maps (few samples/neuron) get a lenient one. The graduated `dead_cv` gives NSGA-II a
continuous gradient for ranking infeasible solutions rather than a binary filter.

Dominance rules:
- `cv_a = 0, cv_b > 0` → a dominates (feasible always beats infeasible)
- `cv_a > 0, cv_b > 0` → lower CV dominates (less constraint violation wins)
- `cv_a = 0, cv_b = 0` → standard Pareto on 3 raw objectives

Raw metric values are **never modified** — CV is stored separately in `results.csv` and
`pareto_front.csv`. `penalty_factor = 1 + cv` is logged as reference only.

---

## Archive & Elitism

- Archive = rank-0 front from `combined_population = evaluated ∪ ARCHIVE` (standard NSGA-II elitism)
- After sorting: duplicates removed by UID (keep highest crowding distance copy)
- Capped at `max_archive_size` by crowding distance (most diverse solutions retained)
- Stale-index bug fix: archive built via `rank == 0` filter on sorted list, not from `fronts[0]`
  indices (which are invalidated by the sort)

---

## Genetic Operators

**Crossover**: Simulated Binary Crossover (SBX, η=20) for continuous and integer parameters;
uniform random swap for categorical. Map size uses SBX on the single dimension (square maps).
Log-scale SBX transforms to log-space, applies standard SBX, transforms back.

**Mutation**: Polynomial mutation (η=20, per-gene probability configurable) with same log-space
variant for LR and radius.

**Repair** (after every crossover and mutation, applied to search space individuals only):
- `start_lr ≥ end_lr`, `start_radius ≥ end_radius` (swap if violated)
- `start_batch_percent ≤ end_batch_percent`
- `epoch_multiplier ≥ 0.1` (prevents zero/negative; actual minimum is defined by search space)
- `growth_g = 0` if all decay/growth types are linear (phenotype normalization for UID cache)

**Intra-generation deduplication**: UIDs computed before spawning workers; duplicate configs in the
same generation are dropped before evaluation (genetic collisions produce identical hashes). Only
unique individuals are sent to the Pool.

---

## Evaluation Pipeline (`evaluate_individual`)

Each individual is a full SOM training run:

1. `KohonenSOM.train()` — hybrid batch mode, `epoch_multiplier × n_samples` total iterations
2. `calculate_topographic_error()` — NumPy-vectorized, not per-sample loop
3. `calculate_u_matrix_metrics()` — u_matrix_max, mean, std
4. `compute_quantization_error()` → `distance_map_max`
5. `calculate_dead_neurons()` — count + ratio
6. Compute `constraint_violation`; store raw metrics; log to `results.csv`
7. Generate training plots and SOM visualizations
8. NN integration (if enabled): MLP pre-screen, LSTM early-stop callback, CNN visual scoring

**Determinism**: `np.random.seed(random_seed)` is called **before** `np.random.rand()` for weight
initialization. Same config always produces identical metrics.

**Cache**: `EVALUATED_CACHE` is a per-process dict (not shared across Pool workers). Intra-generation
dedup (above) is the primary mechanism to avoid redundant evaluation.

**Checkpoints**: stored at every MQE evaluation interval (~500 per run) for LSTM training data.
`checkpoint[0]` (pre-training, near-random-init state) is the `initial_mqe` baseline.

---

## Neural Network Integration (optional)

| Module | Role |
|---|---|
| MLP | Pre-screen: predict MQE before SOM training; skip if predicted > threshold |
| LSTM | Early stopping: predict convergence from checkpoint sequence during training |
| CNN | Visual quality score from RGB composite of u_matrix + distance_map + dead_neurons_map |

CNN score can be added as a 4th Pareto objective when enabled.

---

## Logging

| File | Content |
|---|---|
| `results.csv` | One row per unique evaluation: raw + penalized metrics, CV, map dims, config |
| `pareto_front.csv` | One row per (generation, archive solution): objectives, CV, config params |
| `calibration_probe.csv` | org_max per probe run; p70 = used org_threshold |
| `status.csv` | Per-individual evaluation lifecycle (started / completed / failed / cached) |
| `log.txt` | Free-form per-UID messages including constraint violations |
| `individuals/<uid>/` | Training plots, SOM visualizations, checkpoint data |
| `maps_dataset/` | u_matrix, distance_map, dead_neurons_map per UID for CNN training |

---

## Configuration

```json
{
  "EA_SETTINGS":   { "population_size": 30, "generations": 50 },
  "FIXED_PARAMS":  { "max_archive_size": 20, ... },
  "SEARCH_SPACE":  { "map_size": {"type": "discrete_int_pair", "min": 8, "max": 20}, ... },
  "GENETIC_OPERATORS": { "sbx_eta": 20, "mutation_eta": 20, "mutation_prob": 0.1, "tournament_k": 3 },
  "CALIBRATION":   { "n_probes": 15, "probe_epoch_multiplier": 0.3 },
  "NEURAL_NETWORKS": { "use_mlp": false, "use_lstm": false, "use_cnn": false }
}
```

Tournament size defaults to `max(2, population_size // 10)` if not set.

---

## Non-standard Design Choices

| Choice | Rationale |
|---|---|
| Constrained dominance instead of penalty multiplication | Penalty deforms raw objectives; constrained dominance preserves clean fitness landscape |
| Calibration probe for org_threshold | Fixed threshold (1.0) caused 100% penalization across all tested datasets |
| Dynamic map_size bounds from n_samples | Eliminates structurally impossible configurations before G0 |
| Coverage-ratio dead threshold | Dead neurons are a function of map/dataset ratio, not an absolute measure |
| MQE improvement ratio (final/initial) | Absolute MQE is not comparable across datasets and map sizes |
| Log-space SBX/mutation for LR and radius | Linear-space operators cluster offspring near the geometric center |
| `duration` removed as Pareto objective | Training time rewarded small fast maps regardless of quality; dominated archive |
| Intra-generation UID dedup before Pool spawn | Parallel workers have no shared cache; dedup prevents redundant evaluation |
