# coverage_sim.py — batch-sampling coverage simulator

Simulates the **exact sample-selection code path** of `KohonenSOM.train()`
(`app/som/som.py`: shuffle → `array_split` into `num_batches` sections →
per-iteration batch-percent decay → `ceil` → `np.random.choice` per
section, without replacement) **without any SOM training**. A simulated
run costs seconds instead of minutes, so coverage questions can be
answered at statistical scale (hundreds of repeats with random seeds).

## Sampling methods

| Method | Mechanism | Coverage property |
|---|---|---|
| `random` | as implemented in `som.py` today: fresh `np.random.choice` from each section every iteration (with replacement *across* iterations) | per-sample hits ~ Poisson(λ); never-visited ≈ e^(−λ); spread min–max grows with λ as √λ |
| `reshuffle` | proposed alternative — **random reshuffling** (epoch shuffling, standard in SGD): walk a shuffled permutation of each section with a pointer, **re-shuffle when exhausted** | hit counts **equal ±1 by construction, independent of seed** — coverage is guaranteed, not probabilistic; presentation *order* stays random every epoch |

Naming: the method was originally called `cycle`, which wrongly evokes a
fixed deterministic ring buffer (`itertools.cycle`) and invites the
data-presentation-bias objection (map adapting to a periodic input
order). There is no fixed order — the permutation is re-drawn every
epoch; only the *counts* are periodic (each sample exactly once per
epoch). `cycle` is still accepted as a deprecated CLI alias and is
normalized when reading older result CSVs.

`reshuffle` is **ported into `KohonenSOM.train()` and is the default**
since 2026-06-12 (config key `sampling_method`; the quality A/B —
experiment C — confirmed no regression at equal budget; see
`docs/ea/SEARCH_SPACE.md` step 1). Parity is proven both ways: the
`verify` subcommand replays real runs of either method exactly (it reads
`sampling_method` from the config, defaulting to `reshuffle` like the
SOM — configs of pre-flip runs must state `"sampling_method": "random"`
explicitly to replay).

Built for `docs/ea/SEARCH_SPACE.md` step 1: decide whether dataset
splitting (`num_batches > 1`) measurably improves coverage — and
therefore whether the EA keeps, fixes, or drops it. The experimental
protocol and decision criteria live there; this page documents the tool.

## Validation status

**Verified 2026-06-12** against a real run: replaying
`WineQuality/results/20260531_153004` (seed 42, 32 485 iterations)
reproduces `sample_coverage.json` **exactly, sample by sample**. This
simultaneously proves:

1. the in-training `track_sample_coverage` counting is correct
   (hypothesis "tracking is buggy" — closed), and
2. the simulator faithfully replicates the training-loop sampling.

The replay works because the simulator mirrors the global RNG stream of a
fixed-seed run: `np.random.seed` → weight-init `rand(m, n, dim)` draw
(consumed via `--config` map size + `training_data.npy` dim) →
`permutation` → per-iteration `choice`. Re-run the check after any change
to the sampling code in `som.py` or in this tool (they must stay in sync —
both files carry a sync note).

## Subcommands

### `verify` — replay a real fixed-seed run

```bash
.venv/bin/python3 app/tools/coverage_sim.py verify \
    data/datasets/WineQuality/results/20260531_153004 \
    --config data/datasets/WineQuality/config-som-coverage.json
```

Needs a results dir with `csv/sample_coverage.json` + `csv/training_data.npy`
and the config the run was started with (`random_seed` must be set,
`track_sample_coverage: true`). Exits non-zero on mismatch. Caveat: a run
truncated by MQE early stopping would mismatch — early stopping is
effectively disabled (`docs/som/issues.md` #2), so this does not occur in
practice.

### `run` — one simulated configuration

```bash
.venv/bin/python3 app/tools/coverage_sim.py run --size 6497 \
    --num-batches 5 --percents 1:5:linear-growth --epoch-multiplier 5
```

`--csv <file>` instead of `--size` additionally computes data-space
metrics. `--seed` fixes the RNG; default is whatever state the process is
in (use `sweep` for proper randomized repeats).

### `sweep` — grid × repeats with random seeds

```bash
.venv/bin/python3 app/tools/coverage_sim.py sweep \
    --csv data/datasets/WineQuality/wine.csv \
    --batches 1,2,5,10 \
    --percents 0.5:2:linear-growth,1:5:linear-growth,1:5:exp-growth \
    --epoch-multiplier 5 --repeats 25 --normalize-budget \
    --out coverage_runs.csv
```

Each repeat draws a fresh random seed (recorded in the CSV → every row is
reproducible). Results are **appended** to `--out`, so one CSV can collect
multiple datasets/invocations. A grouped summary (mean ± std vs the
Poisson reference) is printed after each sweep.

Percent profiles are `start:end:growth_type` triplets; growth types are
the `get_decay_value` curves (`static`, `linear-growth`, `exp-growth`, …)
with `--growth-g` steepness for exp/log. `--methods random,reshuffle` adds the
sampling method as a grid dimension.

### `compare` — methods × epoch multipliers head-to-head

```bash
.venv/bin/python3 app/tools/coverage_sim.py compare \
    --csv data/datasets/WineQuality/wine.csv \
    --methods random,reshuffle --multipliers 1,2,3,5,7,10 \
    --repeats 25 --coverage-target 0.999 --out compare_runs.csv
```

The "headline" experiment: for each sampling method and each
`epoch_multiplier`, run N random-seed repeats and report the hit-count
distribution (mean of per-run min / p25 / median / p75 / max),
`never_ratio` mean ± std, Gini, `unseen_after_third`, and the fraction of
repeats reaching `--coverage-target`. The final block prints, per method,
the **minimum epoch_multiplier that reached the target coverage in every
repeat** — the direct answer to "stochastic mode needs at least X passes
to guarantee sufficient coverage".

Default profile is `0.01:0.01:static` with `--num-batches 1` — pure
stochastic, exactly 1 sample/iteration for datasets up to 10 000 rows
(`ceil` of a sub-1 value), so λ = epoch_multiplier and the analytic
prediction for `random` is never-visited ≈ e^(−em).

Example (Iris, N = 150, 10 repeats): `random` reaches 99.9 % coverage in
all repeats only at em = 10 (em = 1 leaves 37 % of the dataset untouched,
matching e^(−1)); `reshuffle` has every sample hit exactly em× already from
em = 1.

### `summarize` — readable markdown report

```bash
.venv/bin/python3 app/tools/coverage_sim.py summarize \
    data/coverage_step1_compare.csv [--coverage-target 0.999] [--md report.md]
```

Turns an accumulated results CSV (sweep- or compare-style, auto-detected
by columns) into a markdown report: per-dataset tables of never-visited %,
hit-count min/median/max, Gini, phase coverage and target-met counts per
method × multiplier, ending with a **conclusion table of the minimum
epoch_multiplier per dataset and method**. The coverage target can be
changed post-hoc (recomputed from `never_ratio`, no re-run needed).
Written to `<csv>_summary.md` by default.

`sweep` and `compare --out` also regenerate this report **automatically
after every invocation**, from *all* rows accumulated in the CSV so far —
so a multi-dataset experiment keeps one up-to-date summary next to its
data file.

## The budget confound and `--normalize-budget`

In `som.py`, `samples_per_section` is computed from **total** dataset
size: `ceil(total_samples · batch_percent / 100)` taken from *each*
section. Raising `num_batches` therefore multiplies the per-iteration
sample throughput by `num_batches` — any naive comparison of splitting
configurations compares *budgets*, not the splitting mechanism.
`--normalize-budget` divides the percents by `num_batches` to hold
throughput constant. Exact parity is still impossible on small datasets
(`max(1, ceil(...))` quantization — each section always contributes at
least one sample), so analyses must always compare at the **measured**
`lambda`, never the nominal one. The CSV records actual `total_updates`.

## Metrics (per run)

| Column | Meaning |
|---|---|
| `total_updates`, `lambda` | processed sample count; λ = updates / n_samples (the real coverage budget) |
| `mean/std/min/max` | per-sample hit-count stats |
| `never`, `never_ratio` | samples never processed |
| `poisson_never_ratio` | analytic iid reference e^(−λ) — the null model |
| `p25`, `median`, `p75` | hit-count quartiles |
| `gini` | hit-count inequality (0 = perfectly even) |
| `poisson_gini` | Monte-Carlo Gini of iid Poisson(λ) — the null model |
| `iters_to_99` | first iteration when ≥99 % of samples were visited (−1 = never) |
| `unseen_after_third` | fraction of samples not yet visited within the first third of training — proxy for "missed the global organization phase" (high LR, large radius); this is where the batch *growth curve* matters even though final counts don't change |
| `target_met` (`compare`) | run reached `--coverage-target` visited fraction |
| `dim_hits_spread_max` | worst-case per-dimension decile unevenness of hits (CSV mode) |
| `dim_never_ratio_max` | worst-case concentration of never-visited samples in any decile of any dimension (CSV mode) |

**Null model.** Sections are a *random index partition*, so per-sample hit
counts should follow iid Poisson(λ) regardless of `num_batches`. If
measured `never_ratio ≈ poisson_never_ratio` and `gini ≈ poisson_gini`,
splitting adds nothing beyond its (hidden) budget multiplication. The
useful coverage knob is then λ alone — which links coverage directly to
the `epoch_multiplier` question (`SEARCH_SPACE.md` step 2): reaching a
never-visited target `p` needs λ ≈ −ln p (e.g. 0.1 % → λ ≈ 6.9).

## Runtime

A run costs O(iterations × N) (the `choice` call mirrors `som.py` and is
O(section size)). WineQuality (N = 6 497, em = 5) ≈ 2–4 s per simulated
run; a 1 000-run sweep on that scale ≈ 1–2 h. Iris-scale is near-instant.
Sweeps append to the CSV, so large grids can be split across invocations.
