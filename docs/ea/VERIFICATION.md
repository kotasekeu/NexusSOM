# EA Verification Toolbox

How to prove that an EA run behaved correctly and how to compare runs.
Complements `RESULTS.md` (what the files contain) and
`docs/global/ABLATION_STUDY.md` (which experiments the thesis needs).

## Automated tests

| Layer | Where | Scope |
|---|---|---|
| Unit (fast, in suite) | `tests/unit/test_ea_core.py` | constrained dominance, non-dominated sort, crowding, tournament (incl. k-clamp), repair, constraint violation bands, objective normalization, dynamic search space anchors, gene-only UIDs |
| Unit (fast, in suite) | `tests/unit/test_ea_operators.py` | SBX/polynomial/categorical operators, bounds, log-uniform sampling proof |
| End-to-end (opt-in) | `tests/integration/test_ea_smoke.py` | full `run_ea.py` subprocess on a tiny dataset; asserts CSV consistency, per-generation evaluation counts, UID uniqueness, all three raw objectives present |

```bash
.venv/bin/python3 -m pytest tests/unit/test_ea_core.py tests/unit/test_ea_operators.py
EA_SMOKE=1 .venv/bin/python3 -m pytest tests/integration/test_ea_smoke.py
```

The smoke test is the per-phase gate of `CLEANUP_PLAN.md` — run it after
any change to `ea.py`.

## `verify_ea_run.py` — run diagnostics

```bash
.venv/bin/python3 app/tools/verify_ea_run.py <run>/seed_42
.venv/bin/python3 app/tools/verify_ea_run.py <run>/seed_42 --sections 1 5 8
```

Point it at a **seed directory** (where `results.csv` lives). Eight
sections: (1) overview + penalty distribution, (2) map size vs penalty
rate, (3) archive evolution per generation, (4) elitism/UID tracking,
(5) final-archive dominance check + constraint breakdown, (6) crowding
ejection analysis, (7) parameter↔penalty correlation,
(8) recommendations (detects known pathologies: always-penalized
`normalize_weights_flag`, too-small maps, archive regression,
non-deterministic re-evaluations).

The section-5/6 dominance check uses the current objective triple
`(raw_mqe_ratio, raw_te, 1 − ρ)` with constrained dominance — fixed
2026-06-11 (`issues.md` #92); legacy runs without recorded ρ degrade to a
two-objective check.

## Interpreting `pareto_metrics.csv`

Per-generation front quality (mechanics in `EA.md` §9):

- **Hypervolume (hv)** — overall front quality; should be non-decreasing
  over generations. With canonical elitism (`issues.md` #87) a drop is
  suspicious: check for constraint-threshold effects or archive-cap trims.
- **Spacing** — uniformity of the front (0 = perfectly even). High spacing
  with small `front_size` = clumped trade-offs.
- **Spread per dimension** — how much of each objective's observed range the
  front covers. Near-zero spread in one dimension means the front
  degenerated to one region — relevant to the open granularity question
  (`CLEANUP_PLAN.md` D1: archives full of near-identical learning rates).

Cross-generation comparability holds within one seed (running min/max
normalization is reset per seed) — compare *shapes* across seeds, not raw
HV values.

## Front evolution plots — `plot_pareto_evolution.py`

```bash
.venv/bin/python3 app/tools/plot_pareto_evolution.py <run>/seed_42
.venv/bin/python3 app/tools/plot_pareto_evolution.py <run>/seed_42 --fixed-range --csv
```

Reads `pareto_front.csv` and renders `pareto_evolution.png` (5-panel:
archive size + MQE evolution + three 2D objective projections) and
`pareto_3d.png` (scatter with wall-shadow projections). Axes are the
current objectives — `raw_mqe_ratio`, `raw_te`, `1 − ρ` (`topo_obj`;
pre-#85 runs fall back to `dead_ratio`). All axes are inverted so
convergence always moves visually up-right. `--fixed-range` pins axes to
[0, 1] for side-by-side comparison of runs; `--guide-lines`,
`--space-grid`, `--lattice`, `--elev/--azim` tune the 3D view; `--csv`
exports per-generation stats.

## Cross-run comparison — `analyze_pareto_fronts.py`

```bash
.venv/bin/python3 app/ea/analyze_pareto_fronts.py <run-or-results-dir> --plot --export combined.csv
```

Discovers `pareto_front.csv` in seed directories (and run roots) under the
given base, keeps only each front's **final generation**, prints per-run
raw-objective statistics + feasibility, and computes the **combined
cross-run non-dominated front** using the same constrained dominance as the
EA itself (no weighted scores). `--plot` adds objective-pair scatters per
run and a hypervolume-evolution comparison from `pareto_metrics.csv`.
Rewritten 2026-06-11 against the current output format (`issues.md` #92).

## Benchmark verification (ablation A3)

EA results on ground-truth benchmarks are verified with the SOM topology
tools (`docs/som/RUN.md`): re-run the best archive configurations via
`app/run_som.py`, then `app/tools/verify_topology.py` and
`plot_som_topology.py`. Baselines: deterministic runs of 2026-06-11 in
`docs/som/BENCHMARKS.md` (Swiss Roll global ρ 0.31, Helix 0.16). The open
question: does any EA-found configuration beat the layer-bridging FAILs?

## Planned ablation switches (not yet implemented)

Folded from the former `ea_tests_ablation_study.md`; the full experiment
matrix lives in `docs/global/ABLATION_STUDY.md`:

- **Mutation off** — works today (`mutation_prob: 0`).
- **Crossover off** — no switch exists (SBX has a fixed internal 50 %);
  needs a `crossover_enabled` flag if the operator ablation is run
  (requirements audit `ea_checklist.md` #46).
- **Objective subset** (e.g. MQE-only) — objectives are hardcoded; needs a
  config switch (`ea_checklist.md` #47).
- **Fitness sub-sampling** (train on a data fraction during evaluation) —
  not implemented (`ea_checklist.md` #26).

`ea_checklist.md` (50-requirement coverage audit) is the authoritative
list of implemented vs missing EA capabilities.
