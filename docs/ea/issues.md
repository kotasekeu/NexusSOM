# EA Module — Known Issues

This document records problems encountered and decisions made during the
development of the EA module — raw material for the thesis text.

Entries **1–86** were migrated from the original Czech records on 2026-06-11
(cleanup phase 4, finding F12) with a validity check against the current
code; outdated claims carry an *[update …]* annotation. They keep their
original compact one-paragraph form. Entries from **#87** on follow the SOM
convention: English, full Issue / Why / Fix structure.

## Architecture and design

1. **NSGA-II instead of a single-objective EA** — originally four objectives
   (MQE ratio, TE, dead ratio, time); reduced to three after discovering that
   time as a Pareto objective attracts small fast maps regardless of quality
   (see #32); the Pareto front preserves a spectrum of quality trade-offs.

2. **SBX + polynomial mutation instead of DE or uniform crossover** — SBX
   provides exploitation near the parents plus proportional jumps across the
   whole interval; polynomial mutation respects bounds without clipping
   artifacts.

3. **Three separate layers: problem type / algorithm / operators** — NSGA-II
   does not dictate the crossover type; SBX would work in SPEA2 too. Mixing
   the layers up led to misunderstandings of the architecture.

4. **NSGA-II instead of SPEA2** — SPEA2 needs a preset external-archive
   size; with an unknown Pareto-front size the variable front of NSGA-II is
   preferable.

## Bugs in the EA code

5. **Archive built from stale indices after `sort()`** —
   `ARCHIVE = [combined_population[i] for i in fronts[0]]` was executed after
   `combined_population.sort()`, so the indices pointed at the wrong
   individuals; good solutions (low MQE) kept dropping out of the archive and
   were replaced by penalized ones. Fixed by filtering `rank == 0` from the
   sorted list.

6. **Hardcoded `tournament_k = 3` unsuitable for small populations** — for a
   population of 5, k=3 would eliminate 60 % of genetic diversity; replaced
   by a dynamic `max(2, population_size // 10)` default, overridable in the
   config. *[update 2026-06-11: k is additionally clamped to the population
   size at selection time — #91.]*

7. **`cnn_quality_score` logged as NaN in results.csv** —
   `log_result_to_csv` ran before the CNN assessment; moved after the CNN
   block.

8. **Stale `fronts[0]` indices broke elitism** — same root cause as #5;
   present in every run from the beginning, preventing genuine retention of
   the best configurations.

## Fitness and evaluation

9. **Absolute MQE is a misleading fitness signal** — MQE depends on map and
   dataset size; an 18×18 SOM on Iris improving MQE 0.1→0.095 looks great
   but the delta is negligible. Replaced by
   `mqe_improvement_ratio = final_mqe / initial_mqe` (from checkpoint[0]).

10. **Topographic error and dead ratio need no normalization** — both are
    naturally in [0, 1] as fractions, independent of dataset and map;
    normalizing them by initial values changed the character of the data with
    no benefit.

11. **Multiplying penalties into MQE distorts the metric** — when
    `u_matrix_max > 1.0` or `distance_map_max > 1.0` multiplied `best_mqe`
    by 10+, a penalized value (QE=14.8) was compared with a legitimate one
    (QE=0.55) on the same scale; this forced the move to improvement ratios
    (and later to constrained dominance, #43).

## Search space

12. **Linear sampling of learning rate distorts the space** — 0.01→0.02 is
    +100 % while 0.51→0.52 is +2 %; log sampling covers the space
    proportionally. Added `log_scale: true` for LR and radius in the config.

13. **SBX and polynomial mutation in linear space for LR** — operators in
    linear space concentrate children near the interval center; added the
    log-space variant: SBX works on `log(x)`, the result is `exp(c)`.

14. **A mixed search space needs typed operator branches** — `float`,
    `float(log)`, `int`, `categorical`, `int pair` each need their own
    operation; uniform swap for categoricals, SBX+round for ints.

15. **`processing_type` was a useless parameter** — the SOM always runs in
    hybrid mode; keeping it in the search space caused redundant
    diversification. Removed from all configs and from the SOM code.

16. **Repair must run after every crossover and mutation** — SBX operates
    per-gene independently and can break relations (LR_start < LR_end etc.);
    a deterministic repair (swap or constant) is applied consistently.

## Pareto front and diversity

17. **The Pareto front explodes with four objectives** — mathematically ~39 %
    of random solutions are non-dominated in 4D; with population 50 that
    means ~20 archive members with no meaningful discrimination. Addressed by
    the `max_archive_size` cap with crowding-distance selection.

18. **A small population yields a large front with no real coverage** — 5
    individuals × 3 generations = 15 evaluations; even penalized solutions
    (QE 11–15) were non-dominated because alternatives were missing. Real
    results need population ≥ 30.

## Efficiency and parallelism

19. **Parallelism on Windows requires `freeze_support` and spawn mode** —
    fork works on macOS/Linux; Windows always spawns a new process; the
    seemingly sequential behavior was spawn overhead, not broken parallelism.

20. **`EVALUATED_CACHE` was not shared across worker processes** — with
    `Pool.starmap` every worker had an empty cache; individuals were
    re-evaluated. *[update 2026-06-11: the cache was removed entirely —
    duplicates are now prevented before evaluation by gene-only UID dedup,
    #89.]*

21. **NN training data require MQE normalization across datasets** — the MQE
    of an 18×18 map on BreastCancer is not comparable with a 5×5 map on
    Iris; `mqe_improvement_ratio` normalizes both; TE and dead ratio need no
    normalization.

22. **Dataset as a feature for the NN models** — an MLP/LSTM trained on one
    dataset does not generalize; dataset statistics (`n_samples`,
    `n_features`, `n_categorical`, …) were added as input features
    (the `ds_*` columns).

## Objective architecture and penalties

32. **`duration` removed as a Pareto objective — replaced by logging** —
    training time as a full NSGA-II objective permanently favored small fast
    maps (5×5, 3–5 s) regardless of quality; they occupied the extreme
    positions of the TIME axis → infinite crowding distance → survived the
    cap and pushed quality solutions out. Time is now logged as metadata but
    does not enter dominance; objectives became
    `[raw_mqe_improvement_ratio, raw_topographic_error, dead_neuron_ratio]`
    (later `1−ρ` replaced dead ratio, #85).

33. **Penalties overwrote raw values — information loss** — multiplying
    `best_mqe *= penalty_factor` destroyed the measured values; it was
    impossible to distinguish "truly bad result" from "good result penalized
    for organization". Added separate columns `raw_best_mqe`,
    `raw_topographic_error`, `raw_mqe_improvement_ratio`, `penalty_factor`,
    `is_penalized`, `penalty_reason`.

34. **NSGA-II objectives switched to raw values** — dominance and ranking use
    raw metrics; penalized values are logged for reference only. Penalties
    still steer the EA through selection pressure but no longer corrupt the
    objectives.

35. **Penalties act as selection pressure, not as an archive filter** —
    penalized solutions stay in the combined population as potential parents
    (genetic diversity); they reach the archive only on raw merit.

## Dynamic penalties and thresholds

36. **Dynamic penalties are architecturally incompatible with NSGA-II** —
    changing penalty thresholds between generations changes the fitness
    landscape; archive members evaluated under G1 rules would then be ranked
    against G5 rules. NSGA-II assumes one consistent fitness function for the
    whole run; dynamic thresholds would force a full archive re-evaluation
    every generation. Penalties must be static from G0 (hence the one-off
    calibration probe, #46).

37. **Dead penalty and `dead_neuron_ratio` as an objective played different
    roles** — the objective minimized dead neurons inside the trade-off; the
    penalty signaled a fundamentally wrong map/dataset ratio and pushed the
    EA away before such configs reached the archive. Both kept (objective
    later replaced by ρ, #85; the constraint remains, #48).

38. **A static `DEAD_NEURON_THRESHOLD = 10 %` ignores the map/dataset
    relation** — dead ratio is primarily a function of
    `coverage_ratio = n_samples / (m·n)`; the threshold must follow it:
    `threshold = clamp(1 − coverage_ratio/10, 0.3, 0.85)` — strict for big
    data on small maps, lenient for small datasets on large maps.

39. **`ORGANIZATION_THRESHOLD = 1.0` was too strict for `dist_map_max`** —
    healthy trained SOMs typically score 1.02–1.16; a static 1.0 penalized
    nearly every run (100 % penalization rate observed). The penalty should
    catch genuinely broken organization (values 5–10), not normal variance.

40. **Calibration approach for the organization threshold** — run 10–20
    quick random SOM configs before the EA (~1–2 % of the budget) and take
    the 70th percentile of `dist_map_max` as the threshold; calibrated once
    before G0, constant afterwards (#36).

41. **Graduated dead-ratio penalty instead of a binary threshold** — one
    fixed threshold jumps from zero to full penalty; graduated bands
    (`excess < 0.2` → ×1.5, `< 0.4` → ×2.5, `≥ 0.4` → ×5.0) preserve the
    information of *how much* the map is oversized and give the EA a
    continuous gradient.

42. **Search-space restriction for `map_size` from dataset analysis** — map
    size should not be chosen freely by the EA; the `5√n` rule (Vesanto &
    Alhoniemi 2000) gives the optimal neuron count, the search corridor
    eliminates predictably unsuitable extremes (5×5 or 25×25).

43. **Constrained dominance (Deb 2002) replaces multiplicative penalties** —
    feasible always dominates infeasible regardless of raw quality; between
    two infeasible the smaller `constraint_violation` wins; between two
    feasible standard Pareto on the 3 raw objectives. Implemented in
    `non_dominated_sort(objectives, violations)` via `_dominates()`.

44. **`constraint_violation` as one scalar aggregating all constraints** —
    organization violation `max(0, max(u_matrix_max, distance_map_max) −
    ORG_THRESHOLD)` plus the graduated dead-neuron violation; `cv = 0` →
    feasible. The magnitude lets NSGA-II order infeasible solutions from
    "mildly violating" to "completely unsuitable". Logged as a column in
    `results.csv` and `pareto_front.csv`.

45. **`penalty_factor` kept as reference-only metadata** — logged as
    `1.0 + constraint_violation` for diagnostics and backward compatibility;
    `best_mqe` and `topographic_error` are always raw measured values now.

46. **Analytic probe before the EA calibrates the organization threshold** —
    `n_probes` (default 15) quick SOM trainings
    (`probe_epoch_multiplier = 0.3`, parallel) from random search-space
    configs; the 70th percentile of `max(u_matrix_max, dist_map_max)` becomes
    `org_threshold` (logged to `calibration_probe.csv`); `n_probes = 0`
    skips the probe (fallback 1.0).

47. **Dynamic search space for `map_size` (Vesanto rule)** — before the EA
    starts, `U = 5√n_samples`, `optimal_side = √U`, corridor
    `[max(8, 0.7·side), 1.3·side]`. BreastCancer (569 samples): U≈119,
    side≈10.9 → [8, 14]. `apply_dynamic_search_space()` returns an adjusted
    copy without mutating the original.

48. **Graduated dead threshold implemented in
    `compute_constraint_violation`** — the static 10 % threshold was replaced
    by the coverage formula (#38) with the graduated bands (#41); this
    eliminated the 100 % penalization rate — the EA now distinguishes a
    structurally wrong map/dataset combination from a normal result.

## Diagnostics and result correctness

26. **`normalize_weights_flag=True` leads to 100 % penalization** — in all
    712 runs with the flag on, the result was penalized (ratio ≥ 2); weight
    normalization consistently produces bad organization. `true` removed
    from the search space.

27. **`map_size min=5` made penalized runs dominate** — 79 % of 5×5 maps
    were penalized and they formed 48 % of all evaluations; the minimum was
    raised (now governed by the Vesanto corridor, #47).

28. **`max_archive_size=10` too small — good solutions crowded out** —
    penalized fast solutions occupied extreme TIME-axis positions with
    infinite crowding distance and ejected valid `ratio < 1` solutions (only
    4 of 39 good UIDs survived). A bigger archive does not fix it — the root
    was removed with `duration` as an objective (#32) and raw objectives
    (#34). *[update 2026-06-11: with canonical elitism the cap is a
    reporting concern only — #87.]*

29. **`np.random.seed()` applied after weight init — nondeterministic
    evaluation** — `self.weights = np.random.rand(...)` ran *before*
    `np.random.seed(self.random_seed)` in `som.py`; the same config evaluated
    twice gave different results (45 duplicate UIDs with different values in
    one 30×50 run). Fixed by swapping the order.

30. **`dead_neuron_ratio` missing from the Pareto log** — the (then) fourth
    objective was not logged, making dominance verification incomplete;
    added `Dead=` to the log. *(Historical — the text log was later replaced
    by `pareto_front.csv`.)*

31. **Diagnostic tool `verify_ea_run.py`** — created in `app/tools/`;
    sections: overview, penalties by map size, archive evolution, elitism/UID
    tracking, final-archive dominance verification, crowding-ejection
    analysis, parameter↔penalty correlation, recommendations. *[update
    2026-06-11: dominance check moved to the current objective triple with
    ρ — #92.]*

49. **`epoch_multiplier min=0` destroyed a 2-hour EA run** — the search
    space allowed 0; repair clamped to ~1, so every SOM trained only
    ~n_samples iterations; SBX from identical parents kept children at the
    same value and, without `duration` as an objective, nothing pushed the
    EA away — all generations converged to systematically undertrained SOMs.
    Fixed by raising the minimum. *[update 2026-06-11: the original fix
    note claimed the repair clamp became `max(1, int(...))`; the current
    intentional behavior is `max(0.1, float(...))` because dynamic bounds go
    below 1.0 for large datasets (#50) — covered by a phase-0
    characterization test.]*

50. **`epoch_multiplier` range must follow dataset size** — a fixed range
    (5–30) suits 569 samples but gives 450 000 iterations for 15 000
    samples. `apply_dynamic_search_space` now derives the range from a
    target total-iteration budget via log-interpolated anchors; `min`/`max`
    written in the config are ignored (always overwritten).

51. **Duplicate offspring were dropped without replacement** — intra-batch
    deduplication ran after generating the whole population, so the
    effective population could shrink. Fixed by moving dedup inside the
    generation loop — offspring are generated until exactly
    `population_size` unique individuals exist (or `population_size × 3`
    attempts). *[update 2026-06-11: the dedup set is now seeded with
    survivor + archive UIDs — #89.]*

52. **The EA converged to tiny LR and radius — skipping the global
    organization phase** — by gen 5 the median `start_learning_rate`
    dropped 0.13→0.03 and `start_radius_init_ratio` 0.12→0.07: small
    radius+LR optimizes `mqe_improvement_ratio` via fast local convergence,
    but the SOM never globally organizes. Neither log sampling nor the
    Gaussian neighborhood prevents it — the problem is selection pressure.
    Fixed by raising the lower bounds: `start_learning_rate min 0.01→0.5`,
    `start_radius_init_ratio min 0.05→0.5` (radius covers at least half the
    map).

53. **Post-convergence generations produce biased NN training data** — after
    convergence (~gen 5) SBX generates offspring in a narrow region;
    generations 6–15 add ~800 mutually similar evaluations, overfitting the
    MLP to the converged region. Longer runs increase the count but decrease
    the diversity of training data.

54. **Multi-seed strategy for NN data collection** — `5 × 50 × 6`
    (seeds × population × generations) instead of one 80×15 run; fixed seeds
    `[42, 1337, 7, 101, 2026]` in `EA_SETTINGS.seeds`; each seed gets its
    own `results/seed_<seed>/`; preprocessing and the calibration probe run
    once, shared. ~1 500 evaluations per dataset with far better space
    coverage.

## NN integration (MLP + LSTM) — Phase 2

55. **LSTM early stopping was gated behind `save_checkpoints` — it never
    ran** — the LSTM callback lived inside the
    `if self.save_checkpoints and …` block in `som.py`. Fixed with a
    separate `lstm_checkpoints` list independent of file-saving settings;
    the LSTM always receives the checkpoint stream.

56. **LSTM fired after 2 checkpoints — far too early (< 20 % of
    training)** — the model was trained on K-prefix windows with
    K ∈ {20,…,70} % of the sequence, but the callback activated after 2
    checkpoints (~2 %). Fixed with
    `lstm_min_checkpoints = max(2, mqe_evaluations_per_run // 5)` (= 20 % of
    training).

57. **Inverted `quality_score` — the LSTM stopped *good* runs** — the
    formula added `final_mqe_ratio` (higher = better) instead of
    `1 − final_mqe_ratio`; fixed to
    `quality_score = (1 − mqe_ratio) + te + dead × 0.5` (lower = better);
    threshold calibrated to p75 of the score distribution.

58. **Inverted MLP filter — it skipped good configurations** — the
    pre-screen condition was `if pred_mqe > threshold: skip`, but the
    predicted `raw_mqe_improvement_ratio` is higher = better; fixed to
    `if pred_mqe < threshold: skip`. The bug existed from the first
    implementation and optimized in the wrong direction.

59. **MLP metadata path failed for the stable `mlp_latest.keras` path** —
    metadata path was derived via `replace('_best.keras', …)`, which is a
    no-op for the stable name, so `json.load()` hit the binary `.keras`
    file and the MLP deactivated itself. Fixed with a candidate-path list
    (`mlp_latest_metadata.json` first).

## Phase 3 — LSTM dynamic controller

60. **Lambda layers in `model_controller.py` were not serializable** —
    Keras refuses to deserialize Python-lambda layers; replaced with proper
    `layers.Layer` subclasses `_TileContext` and `_ScaleSigmoid`
    (with `compute_output_shape()` and `get_config()`); model retrained.

61. **`sample_weight` shape mismatch in seq2seq training** — with
    `return_sequences=True` the internal loss has shape `(batch, time)`;
    passing `(N,)` weights fails. Fixed by tiling the advantage array to
    `(N, T)`.

62. **`_load_lstm_controller` lacked `custom_objects`** — loading the
    retrained model would fail with `Unknown layer: _TileContext`; fixed by
    importing and passing both custom layers to `load_model()`.

63. **Phase 3 model collapsed to a constant output ≈ 0.997** — on 24
    trajectories the predicted `lr_f` std was 0.005 vs target 0.079,
    r ≈ −0.10, MAE worse than the constant-1.0 baseline; root cause:
    70–75 % of timesteps have target 1.0 (`PERTURB_PROB = 0.4` + baseline
    trajectories); advantage weighting could not overcome the imbalance.

64. **Phase 3 factors were invisible in `som.py`** — the controller applied
    factors with no logging; added a periodic
    `LSTM ctrl @ XX%: step lr_f=… rad_f=… | cum_lr=… cum_rad=…` log line.

68. **Phase 2 confusion matrix showed 100 % accuracy — an imbalance
    artifact** — the test set contained no individual below the stop
    threshold, so "always STOP" was always right; the threshold (0.75) was
    not an informative decision boundary. Fix direction: calibrate the
    threshold as a training-set percentile and stratify the uid-level
    split.

69. **TE is structurally harder for the Phase 2 LSTM** — r=0.499 for TE vs
    r=0.970 for the MQE ratio; TE regimes differ by map size (context, not
    sequence input) and TE decreases in discrete jumps while MQE is smooth.
    Decision: optionally down-weight TE for early prefixes; left at full
    weight as an open Phase 2 question.

70. **Phase 3 perturbations ignored physical limits — radius > map, LR
    growing** — the random perturb functions applied ±25 % factors blindly;
    cumulative drift produced radius 80+ on 20×20 maps and rising LR, so
    training data contained physically meaningless trajectories. Replaced by
    a single `make_constrained_perturb_fn()` that reads current values,
    clips the proposed absolute value to `lr ∈ [1e-4, start_lr]`,
    `radius ∈ [1.0, max(m, n)]`, and returns `new/current`.

71. **60 % of Phase 3 timesteps had target exactly 1.0 — model collapse** —
    unperturbed timesteps dominated the loss; fix: mask/filter to perturbed
    timesteps (or raise `PERTURB_PROB`), and measure r/MAE only on perturbed
    timesteps.

72. **Padding zeros created a fake +0.5 residual spike** — padded timesteps
    have `y = [0, 0]` while the model predicts ~0.5; pure padding artifact.
    Fix: mask padded positions in evaluation and add
    `Masking(mask_value=0.0)` in training.

73. **Advantage quartile plot had an empty Q1 with many-tie zeros** —
    quartile boundaries computed from tied values produced an empty set;
    fixed by binning via `np.array_split(np.argsort(adv), 4)`.

74. **`visualize_model.py` missed `custom_objects` for controller models** —
    same class of failure as #62; fixed by conditional import + passing the
    custom layers.

75. **Static advantage weights let one component dominate across datasets** —
    `1.0·δMQE + 0.5·δTE + 0.3·δDead` with fixed coefficients: on one dataset
    δTE contributes nothing, on another it dominates. Fixed by Z-score
    normalization of each component across all records before summation.

76. **`load_splits` read masks but did not return them** — the return tuple
    omitted `msk_*`, causing `NameError: msk_train is not defined` at the
    first use; fixed by extending both tuples.

77. **Prediction display showed a padding position** — `last = -1` on padded
    sequences always pointed at `y = [0, 0]`, making the model look broken;
    fixed by locating the last perturbed timestep via the mask.

78. **Advantage clipping at 0 left 61 % of trajectories with zero weight** —
    negative advantages were clipped to 0, so the model effectively learned
    from ~10 % of timestep–batch combinations and collapsed to the output
    midpoint; fixed by min–max normalization (worst trajectory 0.0, best
    1.0, everything else non-zero).

79. **Phase 3 verdict: behavioral cloning with trajectory-level advantages
    does not work — credit assignment problem** — after all fixes
    (#70/#71/#78) the model still collapses to ≈1.0 (r = 0.03 for lr_factor,
    0.08 for radius_factor): a trajectory-level advantage rewards every
    perturbation in a trajectory equally, so averaging over ~152 random
    perturbed timesteps cancels any consistent gradient. Candidate
    solutions: **(A)** per-timestep advantage — one isolated perturbation
    per trajectory, measure its individual δMQE (structurally cleanest);
    **(B)** a global controller predicting one (lr_scale, radius_scale) pair
    per run; **(C)** an order of magnitude more data. This is the standing
    status of the Phase 3 controller.

80. **Phase 3 visualizations mixed padding and unperturbed timesteps** —
    74 % of plotted points were padding/no-ops, making Phase 3 plots look
    like Phase 2 plots; fixed by flattening only timesteps with
    `msk > 0` ("perturbed timesteps only" in titles).

## Pareto metrics — HV, Spacing, Spread

65. **`np.clip` without scale normalization distorted HV** — objectives with
    larger absolute ranges dominated the hypervolume; per-generation min–max
    would break cross-generation comparability, so a **global running
    min/max** (updated each generation, reset per seed) normalizes before
    HV; reference point [1.1]³ keeps its meaning across the run.

66. **HV and Spacing were missing as per-generation front metrics** —
    implemented `_compute_pareto_metrics()` (pymoo HV + nearest-neighbor
    Spacing) and `pareto_metrics.csv` logging; only feasible solutions
    enter.

67. **Spacing without Maximum Spread misses front coverage** — a perfectly
    uniform front can still cover 1 % of the objective space; added per-
    dimension `spread_mqe/te/dead` (max−min in normalized space).

85. **`dead_neuron_ratio` moved from objective to constraint — replaced by
    Spearman ρ as the global topology objective** — dead ratio as an
    objective pushed toward small maps and duplicated the constraint system;
    TE catches only local neighborhood errors, while a globally crumpled map
    can keep TE low. New third objective: `1 − ρ` where ρ = Spearman
    correlation between pairwise weight distances (data space) and pairwise
    physical neuron distances (grid space); dataset-size independent.
    Implemented as `calculate_topological_correlation()` in `som.py`;
    `raw_topo_corr` added to `pareto_front.csv`. *[update 2026-06-11:
    `raw_topological_correlation` was missing from `results.csv` until
    cleanup finding F17 added it.]*

86. **Hex grid distances for ρ switched from cube-Manhattan to physical
    Euclidean** — integer cube distances produce massive rank ties on the
    grid side of the Spearman correlation; physical coordinates
    (`x = j + 0.5·(i%2)`, `y = i·√3/2`) remove the ties. Isolated inside
    `calculate_topological_correlation()` — TE, the neighborhood function
    and `_grid_edges()` are untouched.

## SOM visualization and metrics (found during EA work)

81. **TE for hex maps used the Moore neighborhood (8 neighbors) instead of
    6** — `max(|Δi|, |Δj|) ≤ 1` accepted 8 neighbors for both topologies,
    systematically underestimating hex TE; fixed with cube-coordinate
    distance == 1 for hex.

82. **`_grid_edges` drew wrong hex edges** — parity-based logic matched a
    wrong cube convention: on a 3×3 grid 4 drawn edges had cube distance 2
    while 4 real neighbors were missing — the "spaghetti" in topology plots
    was a drawing bug, not a SOM error. Rewritten via cube directions and
    verified exhaustively on 3×3 and 8×8.

83. **Hex auto-detection in `plot_som_topology.py` never fired** — the tool
    read `hex_topology` from `dataset_meta.json`, but the topology is in
    `run_metrics.json` (`map_topology`); fixed by reading both.

84. **`plot_som_topology.py` projected masked dimensions** — the primary-ID
    column (monotonic for data, random for weights) contaminated PCA/UMAP
    projections; fixed by stripping always-masked columns before
    projection.

## Checkpoints and LSTM data

23. **Sparse checkpoints in long SOM trainings** — 15 000 iterations with 25
    checkpoints = 1 per 600 iterations, too little for LSTM training; added
    `checkpoint_every_mqe` (~one checkpoint per MQE evaluation).

24. **TE computation slowed logging dramatically** — a Python loop over all
    samples; replaced by NumPy broadcasting (~100× faster).

25. **Variable LSTM sequence lengths due to early stopping** — runs stopped
    early have fewer checkpoints; `collect_training_data.py` pads/truncates
    to a fixed length; the default was raised to match real data
    (~500 checkpoints).

---

## 87. Environmental Selection Survivors Were Discarded — Elitism Reduced to the Capped Archive

**Issue**
The generation loop computed the canonical NSGA-II environmental selection
(combined population sorted by rank and crowding distance, first N kept), but
then used those N survivors **only as the mating pool**. The population
carried into the next generation was the offspring alone, and the combine
step joined offspring only with `ARCHIVE` — the rank-0 front capped at
`max_archive_size` (20–25). Every survivor from fronts F₁ and beyond was
silently lost, and even rank-0 solutions beyond the archive cap disappeared.
Selection pressure and convergence behavior deviated from NSGA-II as
described in the documentation (and as cited from Deb 2002).

**Why**
Two intents collided in the same variable. `population` was first assigned
the environmental-selection survivors (correct NSGA-II), but at the end of
the loop reassigned to `next_gen_offspring` (a μ, λ replacement scheme).
The combine step `evaluated_population + ARCHIVE` then re-imported only the
archive, which had been deliberately capped for *reporting* reasons — the
user had reduced `max_archive_size` because a 50-member result Pareto front
full of near-identical learning rates is not useful output. The cap thus
quietly became the elitism bottleneck, which was never the intent (the
reporting-size concern is tracked separately as design question D1 in
`CLEANUP_PLAN.md`).

**Fix** *(2026-06-11)*
`run_evolution()` now keeps the survivors **with their evaluation results**
in `elite_survivors`. The combine step is
`evaluated_offspring + elite_survivors + ARCHIVE`, deduplicated by UID
(fresh evaluations take precedence), so parents are never re-evaluated and
the selection operates on the full P ∪ O as Deb 2002 prescribes. The archive
keeps its original role: a capped, permanent record of the best front for
logging and result reporting. Verified with a 6×3 smoke run on Iris: 6
evaluations per generation in `status.csv`, archive growth 2→3→5, all final
archive members feasible. The smoke run also exposed one duplicate
evaluation (17 unique UIDs of 18) — that is the separate archive-UID
mismatch, finding F8 in `CLEANUP_PLAN.md`.

## 88. Dead and Broken Code Paths Accumulated in ea.py

**Issue**
The module carried several unreachable or non-functional code paths: the
pre-continuous discrete operators `crossover()`, `random_config()`,
`mutate()` (the last referencing parameters `min/max_batch_percent` that no
longer exist); `load_input_data()` whose *first statement* was `sys.exit(0)`
— any caller would silently kill the process with exit code 0;
`log_progress()`, `log_final_best()`, and `extract_uid_from_path()` (which
searched for a `nxmpp` filename prefix from a long-abandoned naming scheme);
a synthetic-data path (`get_or_generate_data` + `DATA_PARAMS`) used only when
`-i` was omitted; and the `ea_config.py` fallback config whose list-style
`SEARCH_SPACE` was incompatible with `random_config_continuous` (a list spec
would be passed through as a fixed value — the whole list), still offered
`normalize_weights_flag: [False, True]` (removed per legacy entry #26), and
contained the typo `niput_dim`. The `GENETIC_OPERATORS.crossover_prob` config
key was never read by any code (SBX applies per-gene with a fixed 50 %
probability internally).

**Why**
Iterative development: the EA migrated from discrete value-list search spaces
to typed continuous specs, from Python-module config to per-dataset JSON, and
from synthetic blobs to real CSV input — each migration left the previous
mechanism behind instead of deleting it. None of it was caught earlier
because the module had no tests (fixed in cleanup phase 0).

**Fix** *(2026-06-11, cleanup phase 1)*
Removed all functions listed above, the unused `NORMALIZED_DATA` global and
`make_blobs` import; deleted `app/ea/ea_config.py`, the stale
`app/ea/ea-config.json`, and the manual print script
`app/ea/test_genetic_operators.py` (superseded by
`tests/unit/test_ea_operators.py`). `-i` and `-c` are now required CLI
arguments; `load_configuration()` loads JSON only. `crossover_prob` removed
from all 13 `config-ea*.json` files via JSON round-trip (validated). Full
suite green: 327 passed + the `EA_SMOKE=1` end-to-end run.

## 89. Selection Metadata Leaked into UID Hashing — Duplicate Evaluations and a Dead Cache

**Issue**
Two related defects around configuration identity. (1) The offspring
deduplication seeded its "already known" set with UIDs of archive members,
but those UIDs were computed from config dicts that the ranking step had
mutated in place with `rank` and `crowding_distance` — so they could never
match the gene-only UIDs of fresh offspring, and a duplicate of a living
solution could be silently re-evaluated (observed: 17 unique UIDs from 18
evaluations in a 6×3 run). (2) `EVALUATED_CACHE` pretended to prevent
re-evaluation, but a new `Pool` is created every generation, so worker
caches died with the pool and the parent-process cache was never written
(legacy entry #20); the cache only added code paths and misleading
"cache hit rate" statistics computed from assumptions instead of data.

**Why**
`get_uid()` hashed *all* dict items, and NSGA-II metadata is attached
directly to the gene dicts during ranking — identity and selection state
were entangled in one object. The cache predated the per-generation Pool
design and was never revisited after it.

**Fix** *(2026-06-11, cleanup phase 2)*
`get_uid()` now excludes `_NON_GENE_KEYS = {rank, crowding_distance}` —
the same genes hash identically before and after ranking (UIDs of pure gene
dicts are unchanged, so historical `results.csv` UIDs remain comparable).
The offspring dedup set is seeded with the UIDs of both the current
survivors and the archive, so no living configuration is re-generated.
`EVALUATED_CACHE`/`EVALUATION_STATS` removed entirely; the end-of-run
summary now reports total and unique evaluations measured from
`results.csv`. Verified: the previously failing 6×3 scenario now yields
18/18 unique UIDs; the `EA_SMOKE=1` end-to-end test gained a hard
`results.uid.is_unique` assertion; full suite 328 passed.

## 90. CNN Legacy Paths Ran by Default in Every EA Evaluation

**Issue**
Although the CNN track is closed (`docs/cnn/CNN_REQUIREMENTS.md`), every EA
evaluation still rendered per-individual PNG maps (U-matrix, distance map,
dead-neurons map) by default, copied them into a central `maps_dataset/`
directory, and after each seed run composed them into RGB images — artifacts
whose only consumer was CNN training. Hundreds of evaluations × 4 PNGs add
measurable runtime and disk usage for no analytical benefit (spatial
analysis operates on `som.weights` directly).

**Why**
`generate_training_plots` and `generate_individual_maps` defaulted to `True`
in `evaluate_individual`, and `combine_maps_to_rgb()` ran unconditionally
after every seed — defaults set when the CNN was still an active component
and never revisited after its closure.

**Fix** *(2026-06-11, cleanup phase 3)*
Both flags now default to `False`; `combine_maps_to_rgb()` runs only when
`generate_individual_maps` is explicitly enabled. The code paths are kept
intact for archival regeneration of CNN example data (thesis exhibit). When
`use_cnn: true` is combined with maps disabled, a warning explains that
`cnn_quality_score` will stay empty. Verified with a dataset-style config
(no flags set): `individuals/<uid>/` contains only checkpoint CSVs, no
`maps_dataset/` is created.

## 91. Tournament Crash on Small Populations Was Swallowed — Truncated Run Reported Success

**Issue**
An EA run with `population_size: 4` and `tournament_k: 5` (the value shipped
in dataset configs) crashed in generation 1: `random.sample(population, 5)`
raises "Sample larger than population". Worse, the crash was invisible: the
generation loop's blanket `except Exception` printed one line and returned,
after which the run printed "Evolution completed", produced
partial-but-plausible CSVs (generation 0 only), and exited with code 0. A
long optimization could silently lose most of its generations and still look
successful to scripts and the UI.

**Why**
`tournament_selection` passed the configured `k` straight to
`random.sample`, which requires `k ≤ len(population)`; nothing validated the
config value against the population size (and survivors can legitimately
shrink below `population_size` when evaluations fail). The `except
Exception` block was written to keep partial results accessible but
converted every programming or configuration error into a quiet truncation.

**Fix** *(2026-06-11, cleanup phase 3)*
`tournament_selection` clamps `k = max(1, min(k, len(population)))` — small
or degenerate populations degrade selection pressure instead of crashing.
The generation loop still prints the fatal error but now **re-raises** it,
so the process exits non-zero with a full traceback (`KeyboardInterrupt`
keeps its graceful-stop behavior). Regression test:
`test_tournament_k_larger_than_population_is_clamped`; the original failing
scenario (Iris config, population 4, k=5) now completes 8/8 evaluations.

## 92. Post-hoc Tools Lagged Behind the Output Format and the Objective Switch

**Issue**
Both post-hoc analysis tools had silently drifted from the EA they analyze.
`app/ea/analyze_pareto_fronts.py` was completely broken: it searched for
`pareto_front.csv` directly under timestamp directories (the file moved into
`seed_<seed>/` with the multi-seed strategy, legacy #54), aggregated columns
that no longer exist (`best_mqe`, `topographic_error`,
`dead_neuron_ratio`), mixed all generation snapshots instead of final
fronts, and ranked configurations by a weighted score including `duration`
and a multiplicative organization penalty — both concepts deliberately
removed from the EA (legacy #32, #43). `app/tools/verify_ea_run.py` worked,
but its section-5/6 dominance check tested the objective triple
`(mqe_ratio, te, dead_ratio)` although the third NSGA-II objective has been
`1 − ρ` since legacy #85 — a correct archive could be reported as
containing dominated solutions ("BUG" false positives), and dead ratio is a
constraint, not an objective.

**Why**
The output layout and the objective set evolved in several steps
(multi-seed directories #54, raw objectives #34, constrained dominance #43,
ρ objective #85) and the post-hoc tools were never part of any gate, so
nothing forced them to follow.

**Fix** *(2026-06-11, cleanup phase 4/b)*
`analyze_pareto_fronts.py` rewritten against the current format: discovers
fronts in seed directories (and run roots), uses only the final-generation
block per front, reports per-run raw-objective statistics + feasibility,
computes the **combined cross-run non-dominated front** with the same
constrained dominance as the EA (no weighted score), exports it, and plots
objective-pair scatters plus hypervolume evolution from
`pareto_metrics.csv`. `verify_ea_run.py` dominance switched to
`(raw_mqe_ratio, raw_te, 1−ρ)` — ρ read from `results.csv`
(`raw_topological_correlation`, present since F17) with `pareto_front.csv`
fallback; legacy runs without ρ degrade to a two-objective check instead of
failing. Verified end-to-end on a fresh 2-seed run: per-run stats, a 5-of-9
combined front, all plots, and a clean section-5 dominance check with the ρ
column displayed.
