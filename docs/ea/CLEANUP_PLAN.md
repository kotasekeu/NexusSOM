# EA Module — Cleanup Plan

**Date**: 2026-06-11
**Pattern**: same as the SOM cleanup (`docs/som/REFACTOR_PLAN.md`) —
characterization tests first, then phased changes, full test suite green
after every phase, every resolved problem recorded in `issues.md`.
**Baseline**: EA core has **no unit tests** (only API-level
`tests/integration/test_api_ea.py`); `app/ea/test_genetic_operators.py` is a
print script, not pytest.

---

## 1. Inventory — what is in the module

| File | Lines | State |
|---|---|---|
| `ea.py` | ~1700 | core, works; contains dead code, one fixed bug (#87), several silent inefficiencies |
| `genetic_operators.py` | 304 | healthy — SBX/polynomial/log-scale verified against config usage |
| `nn_integration.py` | 670 | functional; belongs to the MLP/LSTM testing effort (next module) |
| `analyze_pareto_fronts.py` | 330 | post-hoc tool; verify during docs phase 4 |
| `ea_config.py` | 59 | **broken fallback** — list-style search space incompatible with continuous operators |
| `test_genetic_operators.py` | 142 | print script in module dir — convert to pytest |
| `ea-config.json` | — | stale sample config in module dir |
| `README.md`, `NN_INTEGRATION_README.md` | — | outdated module-dir docs |

## 2. Findings (verification of 2026-06-11)

Status legend: ☐ open · ☒ done

- ☒ **F7 — Elitism broken** *(fixed 2026-06-11, `issues.md` #87)*:
  survivors of environmental selection were discarded; only the capped
  rank-0 ARCHIVE re-entered the combine step. Fixed to canonical NSGA-II
  (survivors carried with results; combined = offspring ∪ survivors ∪
  archive, uid-deduplicated). Verified by smoke run (archive 2→3→5, 6
  evals/gen).
- ☒ **F1 — `load_input_data()` is dead and booby-trapped** *(removed
  2026-06-11, `issues.md` #88)*: first statement was `sys.exit(0)`; zero
  callers.
- ☒ **F2 — Legacy discrete operators** *(removed 2026-06-11, #88)*:
  `crossover()`, `random_config()`, `mutate()` in `ea.py` — superseded by
  `genetic_operators.py`, zero callers, referenced nonexistent params.
- ☒ **F3 — More dead functions** *(removed 2026-06-11, #88)*:
  `log_progress()`, `log_final_best()`, `extract_uid_from_path()`.
- ☒ **F4 — `ea_config.py` fallback broken** *(removed 2026-06-11, #88)*:
  file + fallback deleted, `-c` and `-i` now required; synthetic-data path
  (`get_or_generate_data` + `DATA_PARAMS`) removed with it; stale
  `app/ea/ea-config.json` deleted too.
- ☒ **F5 — `crossover_prob` config key never read** *(removed 2026-06-11,
  #88)*: dropped from all 13 `config-ea*.json` files (JSON round-trip,
  validated).
- ☒ **F6 — No EA core tests** *(closed 2026-06-11, phase 0)*: characterization
  suite added — `tests/unit/test_ea_core.py` (38 tests),
  `tests/unit/test_ea_operators.py` (19 tests, supersedes the print script),
  `tests/integration/test_ea_smoke.py` (end-to-end, gated by `EA_SMOKE=1`).
  Deleting `app/ea/test_genetic_operators.py` itself happens in phase 1.
- ☒ **F8 — Offspring-vs-archive UID dedup ineffective** *(fixed 2026-06-11,
  `issues.md` #89)*: `get_uid()` now excludes selection metadata
  (`_NON_GENE_KEYS`); dedup set seeded with survivor + archive UIDs.
  Previously failing 6×3 scenario now 18/18 unique.
- ☒ **F9 — `EVALUATED_CACHE` effectively dead** *(removed 2026-06-11, #89)*:
  cache + `EVALUATION_STATS` plumbing deleted; end-of-run summary measured
  from `results.csv`.
- ☒ **F10 — CNN legacy paths active by default** *(fixed 2026-06-11,
  `issues.md` #90)*: `generate_training_plots` / `generate_individual_maps`
  default off; `combine_maps_to_rgb()` opt-in; warning when `use_cnn` is set
  without maps. Code paths kept for archival regeneration.
- ☒ **F13 — `tournament_k` > population crashed `random.sample`** *(found &
  fixed 2026-06-11 during the phase-3 gate, `issues.md` #91)*: k now clamped
  to population size.
- ☒ **F14 — Fatal errors in the generation loop were swallowed** *(fixed
  2026-06-11, `issues.md` #91)*: a crashed run printed "Evolution completed"
  and exited 0 with partial CSVs; the loop now re-raises after logging
  (KeyboardInterrupt keeps graceful stop).
- ☐ **F11 — Doc/code mismatches** (corrected in `EA.md`, propagate
  everywhere): `epoch_multiplier` calibration is anchor-interpolation (older
  docs claim `[3000, 20000]/n_samples`; config comments claim
  `target_iter=10_000_000//n_samples` — both wrong); `tournament_k` default
  is `max(2, N//10)`, not 5; CLAUDE.md lists objectives as
  `[mqe_ratio, TE, dead_ratio]` — actual: `[mqe_ratio, TE, 1−ρ]` with dead
  ratio as constraint.
- ☒ **F12 — Legacy `ISSUES.md` migrated to English** *(done 2026-06-11)*:
  all 86 Czech entries translated into `issues.md` with validity checks;
  outdated claims carry *[update …]* annotations (#20 cache removed, #28
  cap now reporting-only, #49 float clamp intentional, #31/#51/#85
  follow-ups). File renamed `ISSUES.md` → `issues.md` (two-step `git mv`,
  case-insensitive FS).
- ☒ **F15 — `analyze_pareto_fronts.py` broken** *(rewritten 2026-06-11,
  `issues.md` #92)*: now discovers fronts in seed directories, uses
  final-generation blocks, reports raw-objective stats + feasibility,
  computes the combined cross-run non-dominated front under constrained
  dominance (no weighted score), and plots objective pairs + HV evolution.
  Verified end-to-end on a fresh 2-seed run.
- ☒ **F16 — `verify_ea_run.py` checked the wrong objective triple**
  *(fixed 2026-06-11, #92)*: dominance now on `(raw_mqe_ratio, raw_te,
  1−ρ)`; ρ from `results.csv` (F17 column) with `pareto_front.csv`
  fallback; ρ column added to the final-archive table; stale
  `EVALUATED_CACHE` recommendation text replaced.
- ☒ **F17 — `results.csv` lacked the third objective** *(fixed 2026-06-11)*:
  `raw_topological_correlation` was never written to `results.csv` (only
  `pareto_front.csv` had it) — an omission from the #85 objective switch
  that would also have starved MLP training of the ρ target. Added to
  `base_fields`; smoke test now asserts the column exists and is finite.

### Open design question (not a bug — needs a decision)

**D1 — Pareto granularity / archive diversity in parameter space.** After
long runs the archive fills with configurations differing only in
insignificant decimals of `learning_rate` etc. — `max_archive_size` was
reduced as a stopgap, but crowding distance operates in *objective* space, so
near-identical *parameter* vectors survive.

**Resolution path decided 2026-06-12 — see `SEARCH_SPACE.md`** (the
authoritative decision document): first shrink the search space to the
parameters that genuinely need searching (R₀ ratio fixed at 1.0, map type
hex, `epoch_multiplier` an explicitly open problem tied to early-stopping
bias), then quantize the survivors to semantic per-parameter grids snapped
in `validate_and_repair` before UID hashing. Sequence: coverage tool
(~1000 simulated runs × ≥3 datasets) → conclusions on
`num_batches`/batch percents/`epoch_multiplier` → final grids → implement.

---

## 3. Phases

### Phase 0 — Characterization tests (before any further code change)

**Status: ☒ done 2026-06-11** — 57 unit tests green on first run, full suite
327 passed / 1 skipped (the gated smoke). Notable characterizations: log-scale
sampling proven log-uniform via median test; `epoch_multiplier` clamp
documented as intentionally float-0.1 (contradicts legacy `ISSUES.md` #49);
F8 captured as `test_metadata_changes_the_uid` (must be inverted in phase 2).

`tests/unit/test_ea_core.py` + `tests/unit/test_ea_operators.py`:

- `non_dominated_sort` / `_dominates`: plain Pareto cases; constrained
  dominance (feasible beats infeasible, CV ordering); known small fronts.
- `crowding_distance_assignment`: extremes get ∞, interior ordering.
- `tournament_selection`: rank wins, crowding breaks ties.
- `validate_and_repair`: swaps, clamps, `growth_g` zeroing rule.
- `compute_constraint_violation` + `_dead_neuron_threshold`: band boundaries,
  coverage formula examples from the docstring.
- `_normalize_objectives` / `_update_obj_running_stats`: running min/max,
  degenerate spans.
- `apply_dynamic_search_space`: Vesanto corridor + epoch_multiplier anchors
  (the table in `EA.md` §5 as test vectors); no mutation of the input.
- `get_uid`: stability, key-order independence, sensitivity to values.
- Operators (port of `test_genetic_operators.py`): bounds respected, log-scale
  operators stay in bounds and concentrate around parents, categorical swap,
  int rounding, square `map_size` invariant.
- One **slow integration smoke** (marked, optional): tiny EA run (pop 4–6,
  2–3 gens, probes off, maps/plots off) on a small CSV; asserts
  `results.csv`/`status.csv`/`pareto_front.csv` consistency — exactly the
  manual verification used for #87.

Gate: suite green; this is the safety net for everything below.

### Phase 1 — Dead code removal (F1, F2, F3, F4, F5)

**Status: ☒ done 2026-06-11** — recorded as `issues.md` #88. `ea.py` shrank
by ~200 lines; `ea_config.py`, `ea-config.json`, `test_genetic_operators.py`
deleted; `-i`/`-c` required. Gate passed: full suite 327 green +
`EA_SMOKE=1` end-to-end run.

Delete dead functions, the broken `ea_config.py` fallback (require `-c`),
the synthetic-data path (require `-i`), `crossover_prob` from configs, and
the stale `app/ea/ea-config.json`. Move `test_genetic_operators.py` content
into the pytest suite (done in phase 0) and delete the script. Record each
removal in `issues.md`.

### Phase 2 — Silent-bug fixes (F8, F9)

**Status: ☒ done 2026-06-11** — recorded as `issues.md` #89. Gate passed:
full suite 328 green; `EA_SMOKE=1` smoke now hard-asserts
`results.uid.is_unique`; the previously failing 6×3 Iris scenario yields
18/18 unique evaluations.

Gene-only UID hashing for the archive/survivor dedup (`get_uid` strips
`rank`/`crowding_distance`); dedup set seeded with survivor + archive UIDs;
`EVALUATED_CACHE`/`EVALUATION_STATS` removed — `results.csv` is the
authoritative evaluation record.

### Phase 3 — CNN-legacy defaults (F10) + crash robustness (F13, F14)

**Status: ☒ done 2026-06-11** — recorded as `issues.md` #90 and #91. The
phase-3 gate (run with a real dataset-style config lacking the flags)
exposed two pre-existing defects, both fixed: `tournament_k` clamping and
fatal-error re-raise. Gate passed: dataset-style 4×2 run completes 8/8
evaluations with no PNG/maps artifacts, exit 0; suite 329 green +
`EA_SMOKE=1`.

`generate_individual_maps` / `generate_training_plots` default to `False`
(`combine_maps_to_rgb` + `copy_maps_to_dataset` opt-in); explicit opt-in to
be documented in `CONFIG.md` (phase 4) for archival regeneration.

### Phase 4 — Documentation rebuild (file by file, code verified per file)

| New file | Replaces | Status |
|---|---|---|
| `EA.md` | EA_MODULE.md, EA_DESIGN.md, EA_REQUIREMENTS.md (algorithm parts) | ☒ 2026-06-11 |
| `RUN.md` | QUICKSTART_EA.md, RUN.md (old), parts of COMPARISON.md | ☒ 2026-06-11 (overwritten in place) |
| `CONFIG.md` | CONFIG.md (old), EA_FLOAT_PRECISION_AND_VALIDATION.md | ☒ 2026-06-11 (overwritten in place) |
| `RESULTS.md` | RESULTS.md (old), RESULTS_PREPROCESS.MD | ☒ 2026-06-11 (overwritten in place) |
| `NN_INTEGRATION.md` | NN_INTEGRATION.md (old), QUICK_START_NN.md, SOM_AND_EA_WITHOUT_NN.md, `app/ea/NN_INTEGRATION_README.md` | ☒ 2026-06-11 (overwritten in place; model internals belong to docs/mlp + docs/lstm) |
| `VERIFICATION.md` | PARETO.md, PARETO_METRICS.md, ea_tests_ablation_study.md (still-valid items folded in) | ☒ 2026-06-11 |
| `issues.md` | ISSUES.md (Czech, 86 entries) — migrated to English with validity checks (F12) | ☒ 2026-06-11 (#1–#92) |

`ea_checklist.md` stays as the requirements-coverage audit; entries
invalidated by phases 1–3 were updated (12, 25, 30, 43, 44, 48). Czech →
English conversion of the checklist goes together with the ISSUES.md
migration. CLAUDE.md EA bullet fixed (F11, 2026-06-11). During this phase
the code verification of the post-hoc tools surfaced F15/F16 (open) and
F17 (fixed).

### Phase 5 — Legacy docs removal (after user sign-off — destructive)

**Status: ☒ done 2026-06-11** (user approved the b → a → 5 sequence). All
files listed below removed via `git rm`, plus `docs/ea/VISUALIZATION.md`
(not in the original inventory — its `plot_pareto_evolution.py` section was
folded into `VERIFICATION.md`; the `plot_som_topology.py` part was
superseded by `docs/som/RUN.md`). `docs/ea/` now contains exactly: `EA.md`,
`RUN.md`, `CONFIG.md`, `RESULTS.md`, `NN_INTEGRATION.md`,
`VERIFICATION.md`, `issues.md`, `ea_checklist.md`, `CLEANUP_PLAN.md`.
External references updated: `CLAUDE.md` (docs/ea marked current),
`README.md` docs table, `docs/global/INTEGRATION_PLAN.md`,
`docs/mlp/MLP_VISUALIZATION.md`, `docs/tools/plot_som_topology.md`.
Remaining cosmetic debt: `ea_checklist.md` is still Czech (translate when
next updated).

Once phase 4 is complete and reviewed, delete the superseded files:
`EA_MODULE.md`, `EA_DESIGN.md`, `EA_REQUIREMENTS.md`, `EA_BUGFIX_ITERATION.md`,
`EA_CONTINUOUS_UPGRADE.md`, `EA_DEDUPLICATION.md`,
`EA_FLOAT_PRECISION_AND_VALIDATION.md`, `EA_SESSION_SUMMARY.md`,
`EA_UPDATE_SUMMARY.md`, `QUICKSTART_EA.md`, `QUICK_START_NN.md`, `FAQ.md`,
`COMPARISON.md`, `RESULTS_PREPROCESS.MD`, `SOM_AND_EA_WITHOUT_NN.md`,
`thesis.md`, old `RUN.md`/`CONFIG.md`/`RESULTS.md`/`NN_INTEGRATION.md`/
`PARETO.md`/`PARETO_METRICS.md`, `ISSUES.md` (after migration),
`app/ea/README.md`, `app/ea/NN_INTEGRATION_README.md`,
`docs/ea/ea_checklist.md`, `docs/ea/ea_tests_ablation_study.md` (fold the
still-valid items into `VERIFICATION.md` / `docs/global/ABLATION_STUDY.md`
first). Presentation files (PPT2, SLIDES) already deleted in the index.

### Phase 6 — D1 decision + EA benchmark runs

Resolve the Pareto-granularity question (D1) with the user, then run EA on
SwissRoll/Helix against the deterministic baselines of 2026-06-11
(`docs/som/BENCHMARKS.md`, ablation A3).
