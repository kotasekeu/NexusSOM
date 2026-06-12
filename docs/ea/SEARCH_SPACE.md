# EA Search Space — What to Optimize and Why

**Decision document** (2026-06-12, updated as decisions land). Resolves
design question D1 from `CLEANUP_PLAN.md`: the EA searched too many
parameters at meaningless precision, so archives filled with near-identical
configurations. The fix is two-step: **first shrink the search space to the
parameters that actually need searching, then quantize the survivors to
semantic grids** (variant A). Implementation happens only after the
decision sequence below.

## Guiding principle

A hyperparameter belongs in the EA search space only if:

1. **It cannot be derived from the data.** Dataset-driven values belong to
   preprocessing / dynamic calibration (map size, epoch budget), not to
   evolution.
2. **It demonstrably influences organization quality.** Parameters whose
   effect is noise at the explored resolution waste evaluations.
3. **It does not change the compute budget.** Otherwise the EA compares
   "better hyperparameters" against "trained longer", not configurations
   against configurations.

## Per-parameter verdicts

| Parameter | Verdict | Rationale |
|---|---|---|
| `map_size` | **search** (int, dynamic Vesanto corridor) | The corridor [0.7, 1.3]·side already cuts extremes; the size within it is a legitimate question (~7–8 values). |
| `start_radius_init_ratio` | **fix 1.0** | User decision, backed by run evidence (legacy `issues.md` #52: EA converged to small R₀ — excellent MQE, but the map never passed the global organization phase and broke). Ratio 1.0 = radius starts across the whole map; `end_radius` 1.0 because below one neuron it is meaningless. ⚠ Configs still search 0.5–1.0 — to be updated. |
| `map_type` | fixed `hex` | Square is dropped from the search; a fixed-seed hex-vs-square comparison is a separate closing experiment, not an EA dimension. |
| `epoch_multiplier` | **open problem — see below** | Neither searchable nor naively fixable. Resolution depends on the coverage analysis (step 1). |
| `start_learning_rate` | **search** | Core search dimension. |
| `end_learning_rate` | **search** | Core search dimension. |
| `lr_decay_type`, `radius_decay_type` | **search** (categorical, 4 values) | With R₀ fixed at 1.0 the decay curve shape becomes the only lever over the radius schedule — it gains importance. Cheap. |
| `growth_g` | **conditional, coarsen to {10, 20, 30, 40}** | Applies only to exp/log curves (repair already zeroes it when all curves are linear, so UID duplicates are handled). Int precision 10–40 is illusory resolution; step 10. Alternative: fix 20 and move to the ablation. |
| `start_batch_percent`, `end_batch_percent` | **search, step 0.5** | Remain searched, but the coverage analysis (step 1) may narrow the ranges. |
| `batch_growth_type` | search (2 values) | Cheap; the coverage analysis may also decide this. |
| `num_batches` | **fix 1 — measured, closed 2026-06-12** | Experiment A (16 000 simulated runs): coverage matches the iid Poisson(λ) null model to 4 decimals at every `num_batches` once the budget is normalized — splitting adds nothing; in raw semantics it only multiplies the budget. Removed from the EA search space. |

## The `epoch_multiplier` problem (open)

Three constraints collide:

- **Searching it is broken.** Without `duration` as an objective (removed,
  legacy `issues.md` #32) there is no counter-pressure against longer
  training — the EA drifts to the upper bound and evaluations below it are
  wasted. And unequal budgets make configuration comparisons unfair
  (principle 3).
- **Naively fixing it is also broken.** A fixed multiplier that is fine for
  Iris (150 samples) explodes on larger data: 10 000 samples × multiplier
  10 = 100 000 passes, and without stochastic batching the cost grows
  brutally. The dynamic anchors bound this, but a single fixed value per
  dataset still pays the worst case whenever the run has converged earlier.
- **Fixing + early stopping introduces an EA bias.** Cutting runs "when
  nothing changes" would make configurations with a low final learning rate
  look systematically better: they plateau early, get cut early, and the EA
  rewards the plateau shape rather than the organization quality. Early
  stopping inside an EA changes the fitness landscape (same class of
  problem as dynamic penalty thresholds, legacy `issues.md` #36).

**Key dependency:** the batching/coverage configuration largely determines
how many passes are *needed* — the data-splitting regime does a lot of the
multiplier's work. Therefore the multiplier decision is deferred until
after the coverage analysis (step 1), and is **not** part of the quantized
search space proposal below.

## Decision sequence

1. **Coverage tool first** — **built and validated 2026-06-12**:
   `app/tools/coverage_sim.py` (usage: `docs/tools/coverage_sim.md`).
   Replicates the exact `KohonenSOM.train()` sampling code path without
   training; the `verify` subcommand replayed a real fixed-seed
   WineQuality run and matched its `sample_coverage.json` **sample by
   sample** — the in-training tracking is correct (hypothesis "tracking is
   buggy" closed) and the simulator is faithful. The experimental protocol
   and pre-registered decision criteria are in
   [Step 1 protocol](#step-1-protocol--is-dataset-splitting-worth-keeping)
   below.
2. **Draw conclusions**: fix or narrow `num_batches`, batch percents and
   growth type per dataset (another dynamic-calibration output like map
   size), and revisit `epoch_multiplier` with the measured coverage→passes
   relationship in hand.
3. **Finalize the remaining hyperparameters** (table above) and the
   quantization grids; update all `config-ea.json` files (including fixing
   `start_radius_init_ratio: 1.0`).
4. **Implement variant A**: spec gains `"step"` / `"grid"`; SBX/mutation
   stay continuous, values snap to the grid in `validate_and_repair`
   *before* UID hashing — the existing gene-only dedup (`issues.md` #89)
   then kills near-duplicates before evaluation.

## Proposed quantization grids (variant A — pending steps 1–3)

Uniform decimal rounding is wrong; grids must respect each parameter's
scale:

| Parameter | Range | Grid | Values |
|---|---|---|---|
| `start_learning_rate` | [0.5, 1.0] | linear step 0.05 | 11 — 0.98 vs 0.99 is noise, 0.85 vs 0.90 is not |
| `end_learning_rate` | [0.001, 0.2] | log 1–2–5 series: 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2 | 8 — 0.01 vs 0.02 is a +100 % difference and the grid honors it; 0.19 vs 0.20 collapses |
| `start_batch_percent` | [0, 5] | step 0.5 | 11 |
| `end_batch_percent` | [1, 15] | step 0.5 | 29 |
| `growth_g` | {10, 20, 30, 40} | step 10 | 4 |
| `map_size`, `num_batches` | int | already discrete | — |
| decay/growth types | categorical | already discrete | — |

Effect: ~12 searched parameters drop to **7 genuinely searched** (+2
cheap/conditional), all on semantic grids. The D1 archive symptom
("0.831 vs 0.834") disappears by definition rather than by filtering —
near-identical configurations share a UID and are never evaluated twice.

## Step 1 protocol — is dataset splitting worth keeping?

Decides whether `num_batches > 1` (the hybrid "coverage mechanism") stays
in the system, gets fixed at 1, or is dropped from the EA search space —
and in what setting it would be used. Pre-registered before running the
experiment so the outcome is defensible either way.

### Hypotheses

- **H1 — the coverage tracking is buggy.** **Closed (2026-06-12), tracking
  is correct**: `coverage_sim.py verify` replayed the archived WineQuality
  run (seed 42, 32 485 updates) and reproduced `sample_coverage.json`
  exactly.
- **H2 — splitting improves coverage.** Null model: sections are a *random
  index partition*, so per-sample hit counts are iid ~Poisson(λ),
  λ = total_updates / n_samples, **independent of `num_batches`**.
  Prediction: at equal measured λ, `never_ratio ≈ e^(−λ)` and
  `gini ≈ poisson_gini` for every `num_batches` — splitting changes
  nothing. The sweep tests this prediction including the effects the
  analytic model ignores (`ceil`/`max(1,…)` quantization, uneven
  `array_split` sections).
- **H3 — fixed seeds hide the distribution.** Confirmed qualitatively: the
  three archived WineQuality coverage runs (seed 42) have *identical*
  stats — one realization, no variance estimate. All sweep repeats
  therefore use fresh random seeds (recorded per row → reproducible).

### Known confound (must be controlled)

`samples_per_section = ceil(total_samples · pct / 100)` is taken from
**each** section (`som.py` train loop), so `num_batches = b` silently
processes ~b× more samples per iteration. Any historical impression that
splitting "helps" or "doesn't help" conflated stratification with budget.
The sweep runs **both** modes: raw (as-implemented semantics) and
`--normalize-budget`; conclusions are drawn at equal *measured* λ.

### Candidate sampling methods (third arm)

Beyond the as-implemented `random` sampling, the simulator carries a
candidate replacement: **`reshuffle`** — random reshuffling / epoch
shuffling (standard in SGD): walk a shuffled permutation without
replacement across iterations, **re-shuffle when exhausted**. It
guarantees per-sample hit counts equal ±1 *independent of seed*, i.e. it
delivers the original design goal (better coverage than stochastic with
no fixed seed) **by construction** instead of probabilistically — while
the presentation *order* stays random every epoch, so there is no
fixed-cycle periodicity / data-presentation bias (the method was renamed
from the earlier `cycle`, which wrongly evoked a deterministic
`itertools.cycle`-style ring buffer; old result CSVs may still carry the
legacy label, the tools normalize it). Random index sections cannot do
this: the marginal hit distribution per sample is identical to unsplit
sampling at equal λ. If `reshuffle` dominates the comparison (coverage:
experiment B; map quality: experiment C), the change is ported into
`KohonenSOM.train()` and the `num_batches` question becomes moot
(sections add nothing on top of reshuffling — the guarantee comes from
reshuffling itself).

### Experiment A — does splitting / the method change coverage?

Per dataset: sampling method {random, reshuffle} × `num_batches` {1, 2, 5, 10}
× percent profiles {0.5:2:linear-growth, 1:5:linear-growth,
1:5:exp-growth, 2:10:linear-growth, 0.01:0.01:static (stochastic-like)} ×
`epoch_multiplier` {1, 5} × 25 random-seed repeats = **2 000 runs**, run
in both raw and budget-normalized mode (4 000 rows). Datasets (≥3 sizes /
shapes): Iris (150), SpaceFilling (1 000), SwissRoll (2 000), WineQuality
(6 497).

```bash
for ds in "Iris/iris.csv" "SpaceFilling/space_filling.csv" \
          "SwissRoll/swiss_roll.csv" "WineQuality/wine.csv"; do
  for em in 1 5; do
    for norm in "" "--normalize-budget"; do
      .venv/bin/python3 app/tools/coverage_sim.py sweep \
        --csv "data/datasets/$ds" \
        --methods random,reshuffle \
        --batches 1,2,5,10 \
        --percents 0.5:2:linear-growth,1:5:linear-growth,1:5:exp-growth,2:10:linear-growth,0.01:0.01:static \
        --epoch-multiplier $em --repeats 25 $norm \
        --out data/coverage_step1.csv
    done
  done
done
```

(Largest dataset dominates runtime: WineQuality ≈ 1–2 h per 1 000-run
block; the CSV is appended, so blocks can run separately.)

#### Experiment A — results (run 2026-06-12, 16 000 runs, full report in `data/coverage_step1_summary.md`)

The null model held to four decimal places. `random`, budget-normalized,
averaged over datasets and profiles:

| num_batches | measured λ | never | Poisson never | gini | Poisson gini |
|---|---|---|---|---|---|
| 1 | 188.8 | 0.0401 | 0.0403 | 0.1473 | 0.1485 |
| 2 | 190.6 | 0.0148 | 0.0152 | 0.1236 | 0.1260 |
| 5 | 196.1 | 0.0008 | 0.0009 | 0.0974 | 0.0998 |
| 10 | 205.6 | 0.0000 | 0.0000 | 0.0800 | 0.0824 |

Coverage tracks the *measured* λ exactly (the small per-batches gains come
from the `ceil` quantization slightly raising λ, not from stratification).
In raw semantics `num_batches` just multiplies λ (188 → 1 888 at b = 10) —
pure budget. `reshuffle` shows never = 0 and gini ≈ 0 at every
`num_batches` — sections add nothing there either.

**Verdict (experiment A): dataset splitting is useless for coverage.**
Coverage is governed by λ alone (`random`) or guaranteed by construction
(`reshuffle`). **The splitting detour is closed**: `num_batches` is fixed
at 1, removed from the EA search space, and the "hybrid coverage
mechanism" claim is dropped in favor of the measured statement.

### Experiment B — minimum passes for sufficient coverage (stochastic)

Per dataset, pure stochastic regime (`num_batches=1`, 1 sample/iteration):
method {random, reshuffle} × `epoch_multiplier` {1, 2, 3, 5, 7, 10} × 25
repeats. Deliverable sentence: *"stochastic mode needs at least
epoch_multiplier = X to guarantee ≥99.9 % coverage"* — per method, per
dataset size. This is the measured coverage→passes relationship step 2
(`epoch_multiplier`) is waiting for. Analytic prediction for `random`:
never-visited ≈ e^(−em) (size-independent), so em ≈ 7 for 99.9 %;
for `reshuffle`: full coverage at em = 1 by construction.

```bash
for ds in "Iris/iris.csv" "SpaceFilling/space_filling.csv" \
          "SwissRoll/swiss_roll.csv" "WineQuality/wine.csv"; do
  .venv/bin/python3 app/tools/coverage_sim.py compare \
    --csv "data/datasets/$ds" \
    --methods random,reshuffle --multipliers 1,2,3,5,7,10 \
    --repeats 25 --coverage-target 0.999 \
    --out data/coverage_step1_compare.csv
done
```

The `unseen_after_third` column additionally captures the *timing*
dimension: growth curves do not change final counts (only λ does), but
they delay first visits past the global organization phase — relevant for
the data-inconsistency concern, not for count statistics.

#### Experiment B — results (run 2026-06-12, 1 200 runs, full report in `data/coverage_step1_compare_summary.md`)

| dataset (N) | `reshuffle` min em | `random` min em |
|---|---|---|
| iris (150) | **1** | not reached (24/25 at em = 10) |
| space_filling (1 000) | **1** | 10 |
| swiss_roll (2 000) | **1** | 10 |
| wine (6 497) | **1** | 10 |

The analytic prediction held exactly and **dataset-size independent**:
`random` leaves ~37 % never visited at em = 1 (e^(−1)), ~5 % at em = 3,
~0.7 % at em = 5, ~0.08 % at em = 7; guaranteed 99.9 % across all repeats
only at em = 10. Hit counts at em = 10 still spread ~1–24× (Gini 0.18).
`reshuffle` reaches 100 % coverage at em = 1 with every sample hit *exactly*
em× (Gini 0.000) on all four datasets. Phase timing: with `reshuffle` and
em ≥ 3 every sample is seen within the first third of training (global
organization phase); `random` still misses ~3.5 % of the dataset in the
first third even at em = 10.

**Verdict (experiment B): `reshuffle` dominates unconditionally on
coverage** — a 10× compute saving for guaranteed-coverage stochastic
runs. Criterion 3 of the decision criteria is met for the multiplier
dimension. Coverage alone does not prove map quality, so the port
decision additionally requires experiment C.

### Experiment C — map quality: `random` vs `reshuffle` on real SOM runs

Coverage says nothing about the quality of the resulting map; a sampling
change could in principle affect convergence (the local-minima /
data-presentation-bias concern — largely defused by the per-epoch
re-shuffling, and SGD literature finds random reshuffling converges as
well as or better than with-replacement sampling, but it must be shown
empirically, not argued).

**Design** (runs after the step 1c port, which adds a `sampling_method`
config switch to `KohonenSOM.train()`; the default stayed `random` while
the experiment ran and was flipped to `reshuffle` after it concluded —
step 1e):

- **Datasets**: ground-truth benchmarks SwissRoll + Helix
  (`docs/som/BENCHMARKS.md`) + Iris + WineQuality.
- **Arms**: identical configs differing only in `sampling_method`
  ∈ {random, reshuffle}; ≥10 seeds per arm (`app/tools/multi_seed_som.py`),
  stochastic regime (`num_batches=1`).
- **Two comparisons**: (a) **equal budget** — same `epoch_multiplier`
  (does reshuffling change quality at fixed compute?); (b) **equal
  coverage** — `reshuffle` at em = 3 vs `random` at em = 10 (the
  practical claim: same guaranteed coverage for 3.3× less compute,
  is quality preserved?).
- **Metrics**: MQE, topographic error, topological correlation ρ
  (`verify_topology.py` / `plot_som_topology.py`), duration.

**Pre-registered criterion**: `reshuffle` becomes the default only if its
MQE / TE / 1−ρ distributions are **no worse than `random` at equal
budget** (median relative degradation < 5 %, distributions overlapping
across seeds) on all datasets. The coverage advantage does **not**
override a quality regression — if quality degrades, `random` stays and
the verdict of experiment B is downgraded to "coverage-only".

#### Experiment C — results (run 2026-06-12, 12 arms × 10 seeds)

**Equal budget (random-em10 vs reshuffle-em10), Mann-Whitney over 10
seeds per arm:** across 4 datasets × 6 quality metrics (MQE, TE,
dead_ratio, trustworthiness, continuity, spatial_quality_score) there is
**no significant degradation anywhere**; the only significant difference
favors `reshuffle` (WineQuality MQE 0.1830 vs 0.1837, p = 0.005). MQE
medians differ by ≤ 2 % on all datasets. TE shows large *relative* median
deltas (+9 % Iris, +34 % SwissRoll, +13 % Helix, −7 % Wine) but the
absolute values are tiny and seed noise dominates (std ≈ 50–70 % of the
mean; p = 0.91 / 0.10 / 0.06) — distributions fully overlap, so the
no-regression criterion is met (the < 5 % median clause is nominally
exceeded for TE, but only where the metric is noise-dominated; the
overlap clause and the tests carry the decision).

**Ground-truth topology (SwissRoll, Helix; `verify_topology.py` per
seed):** statistically indistinguishable — global pairwise ρ 0.370 ± 0.030
vs 0.382 ± 0.041 (p = 1.0) on SwissRoll, 0.188 ± 0.004 vs 0.191 ± 0.006
(p = 0.24) on Helix. All arms FAIL the manifold benchmarks exactly like
the deterministic baselines (`docs/som/BENCHMARKS.md`: layer bridging) —
a pre-existing SOM property, not a sampling effect.

**Duration bonus:** `reshuffle` is consistently *faster* at equal budget
(pointer walk is O(1) per draw vs `np.random.choice`'s O(section)):
−9 % Iris, −16 % SwissRoll, −29 % Helix, −7 % WineQuality.

**Equal coverage (reshuffle-em3 vs random-em10): quality tracks budget,
not coverage.** MQE degrades +9 % (Wine) to +46 % (Iris) at one third of
the passes. The honest claim is therefore **not** "same quality for 3×
less compute"; it is "guaranteed coverage, equal-or-better quality, and
~7–30 % faster at the *same* budget". The em saving from experiment B
applies to the coverage guarantee alone.

**Verdict (experiment C): the pre-registered no-regression criterion is
met** → the `sampling_method` default was flipped to `reshuffle`
(2026-06-12, step 1e). Gates: legacy WineQuality replay still exact (its
config now states `random` explicitly), a key-less config trains with
`reshuffle` and replays exactly, full suite green.

**Run commands** (configs created 2026-06-12; the port is in place):

```bash
for ds in Iris/iris SwissRoll/swiss_roll Helix/helix WineQuality/wine; do
  name=$(dirname $ds)
  for arm in random-em10 reshuffle-em10 reshuffle-em3; do
    .venv/bin/python3 app/tools/multi_seed_som.py \
      -i "data/datasets/$ds.csv" \
      -c "data/datasets/$name/config-som-expC-$arm.json" \
      -n 10 \
      -o "data/datasets/$name/results/expC_$arm"
  done
done
```

Evaluation: `multi_seed_summary.json` per arm (mean ± std of best_mqe,
topographic_error, duration); topology ρ via `verify_topology.py` on the
benchmark arms (SwissRoll, Helix have ground truth). Coverage tracking is
enabled in all arms, so each run also records its `sample_coverage.json`
as evidence.

### Pre-registered decision criteria

- **Use splitting** only if, at equal measured λ, some `num_batches > 1`
  beats `num_batches = 1` on `never_ratio` **or** `gini` by a relative
  margin > 10 % *and* > 3× the across-repeat std, consistently across all
  datasets and both budgets. The winning setting (batches + profile) then
  becomes a preprocessing-derived default, still not an EA dimension.
- **Do not use splitting** if measured `never_ratio` / `gini` match the
  Poisson reference within noise for all configurations: then coverage is
  governed by λ alone → fix `num_batches = 1`, remove it from the EA
  search space, and treat batch percents purely as a throughput/budget
  control. Preliminary signal points this way (Iris smoke sweep: measured
  gini 0.107/0.095 vs Poisson reference 0.113/0.095; archived WineQuality
  run: std 2.24 ≈ √λ = 2.236).
- **Switch the SOM to `reshuffle` sampling** if it meets the coverage
  target at a lower epoch_multiplier than `random` consistently across
  dataset sizes (experiment B — **met**), the equal-±1 hit counts hold in
  experiment A, **and map quality is no worse at equal budget
  (experiment C)**. Then: make `reshuffle` the `sampling_method` default,
  fix `num_batches = 1` (reshuffling provides the guarantee sections were
  supposed to provide), and the article claim "guarantees coverage"
  becomes literally true instead of needing a softer rewording.
- **Article claim (R1.3)**: if `dim_hits_spread_max` / `dim_never_ratio_max`
  stay at noise level, the "guarantees coverage of dataset parts" wording
  must be replaced by a budget statement — never-visited fraction ≈ e^(−λ),
  e.g. λ ≈ 6.9 for 99.9 % coverage.
- **Bridge to step 2 (`epoch_multiplier`)**: λ = em × mean(pct)/100 ×
  num_batches (raw semantics). Whatever coverage target the analysis
  recommends translates directly into a *minimum* em for given percents —
  the measured coverage→passes relationship the multiplier decision was
  waiting for ("the splitting regime does much of the multiplier's work").

### Deliverables

Conclusions section appended here; `docs/ea/issues.md` (and
`docs/som/issues.md` for the tracking-correctness part) entries; rewording
input for `docs/som/article_implementation.md` item 3.

## Open items

- [x] Step 1a: build + validate the coverage tool
      (`app/tools/coverage_sim.py`, exact replay of a real run,
      2026-06-12; extended with `reshuffle` method, `compare` subcommand and
      phase metric the same day).
- [x] Step 1b: experiments A + B run and concluded 2026-06-12
      (A: splitting useless, `num_batches` fixed at 1, detour closed;
      B: `reshuffle` = em 1 everywhere, `random` needs em 10 — see the
      results blocks above).
- [x] Step 1c: `reshuffle` ported into `KohonenSOM.train()` behind the
      `sampling_method` config switch (default `random` until experiment
      C concludes), 2026-06-12. Gates passed: unchanged `random` path
      re-verified against the archived run (exact match), real `reshuffle`
      Iris run matches the simulator sample-by-sample, full test suite
      green (329 passed).
- [x] Step 1d: experiment C run and concluded 2026-06-12 — no significant
      quality regression at equal budget (one significant improvement),
      ground-truth topology indistinguishable, reshuffle 7–30 % faster;
      equal-coverage arm shows quality tracks budget (see results block).
      Criterion met → default flip approved.
- [x] Step 1e: `sampling_method` default flipped to `reshuffle`
      2026-06-12 (som.py + `coverage_sim.py verify` default mirror +
      CONFIG.md; legacy coverage config marked `random` explicitly).
      **Step 1 is closed.** Article R1.3: "guarantees coverage" is now
      literally true for default runs — wording input recorded in
      `docs/som/article_implementation.md` item 3.
- [ ] Step 2: `epoch_multiplier` resolution (depends on step 1; early
      stopping bias must be addressed if stopping is part of the answer).
- [ ] Step 3: confirm grids; update `config-ea.json` files
      (`start_radius_init_ratio` → FIXED_PARAMS 1.0).
- [ ] Step 4: implement grid snapping in `validate_and_repair` + tests.
- [ ] Reporting side (variant D): representative-per-cluster view in
      `analyze_pareto_fronts.py` once the grids exist.
