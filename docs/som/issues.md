# SOM Module — Known Issues

This document describes issues encountered during the development of the SOM module.
Each entry follows the format: what the issue is, why it occurs, and how it is (or can be) resolved.

---

## 1. Excessive Logging Slows Down Training

**Issue**
Every phase of the SOM pipeline writes log entries by calling `log_message()`. In `run.py` alone there are ~25 calls covering directory setup, validation, preprocessing, training start/end, analysis, plots, and visualizations. Additional calls appear inside `preprocess.py` for each processing step. During an EA run that executes hundreds of SOM trainings in parallel, this volume of I/O adds measurable overhead.

**Why**
`log_message()` in `utils.py` opens the log file, appends a single line, and closes it on every call. There is no buffering, no log level filtering, and no way to disable logging at runtime. The cost is not in any single call — it is in the cumulative effect across the full pipeline, especially when the EA spawns many parallel workers all writing to their own log files simultaneously.

**How It Can Be Solved**
The simplest fix is to replace per-call file opens with a buffered approach: open the file once per run, buffer writes in memory, and flush periodically or at the end. An alternative is to add a log level threshold so that routine `SYSTEM` messages are skipped unless explicitly requested, while `ERROR` and `FATAL` messages are always written. Neither change affects the algorithm itself — this is purely an I/O optimization.

---

## 2. Early Stopping Window Can Terminate Training Too Soon

**Issue**
The early stopping mechanism uses a moving average over a window of recent MQE evaluations. If the window is too small, normal short-term MQE fluctuations can look like a plateau, and the patience counter triggers a stop before the map has actually converged. This risks producing a suboptimal map, particularly in the early phases of training when MQE is still decreasing unevenly.

**Why**
MQE is not evaluated at every iteration — it is computed at a fixed interval (`mqe_compute_interval = total_iterations / mqe_evaluations_per_run`). The moving average is calculated over the last `early_stopping_window` evaluations using a `deque`. The stop fires when `epochs_without_improvement` reaches `early_stopping_patience`. If the window is small relative to the number of evaluations, or if MQE has a temporary spike followed by a dip, the moving average can stall even though the overall trend is still improving.

**Current State**
Both `early_stopping_window` and `early_stopping_patience` are currently set to a default of 50000, which effectively disables the feature — training always runs for the full `epoch_multiplier × num_samples` iterations. There are explicit `FIXME` comments in the code marking this as intentionally disabled until the interaction between window size, patience, and MQE evaluation frequency is properly tuned.

**How It Can Be Solved**
The parameters need to be calibrated relative to `mqe_evaluations_per_run` and the expected training length. A reasonable starting point is a window of 3–5 evaluations and patience of 2–3 checks. This should be validated empirically across different map sizes and datasets before re-enabling the feature.

---

## 3. Missing or Corrupted Sensor Data Tears the SOM Map Apart

**Issue**
When the source dataset contains missing values — caused either by sensor malfunction or by genuinely extreme/outlier readings — the SOM map can become distorted. Neurons near the missing-value samples get pulled toward artificial fill values, creating visible tears or dead zones in the resulting map topology.

**Why**
Missing values cannot simply be removed from a row because each sample must have the same dimensionality as the weight vectors. The current preprocessing pipeline fills NaN values with column medians (for numeric columns) or empty strings (for categorical columns). These fill values are synthetic and do not represent any real data point. Without a mechanism to exclude the filled dimensions from influencing the map, the SOM treats them as real observations during BMU selection and weight updates, which distorts the learned topology.

**How It Is Solved — The Ignore Mask**
A per-sample boolean mask (`ignore_mask`) is created during preprocessing from `training_df.isnull()`. Any dimension that was originally NaN is marked `True`. The primary ID column is also unconditionally masked. This mask is carried through the entire training loop and applied in four places:

- **BMU selection** (`find_bmu`): masked dimensions are zeroed out before computing Euclidean distance, so missing values do not influence which neuron wins.
- **Weight update** (`update_weights`): the update term is zeroed on masked dimensions, so neuron weights are not pulled toward fill values.
- **Quantization error** (`compute_quantization_error`): masked dimensions are excluded from the error calculation, keeping MQE honest.
- **Topographic error** (`calculate_topographic_error`): same zeroing applied before distance computation.

The result is that missing dimensions are effectively invisible to the algorithm. The SOM learns from the dimensions that are actually present for each sample.

---

## 4. The Mask Changes the Effective Data — Downstream Interpretation Required

**Issue**
By masking missing dimensions, we prevent the SOM from being corrupted, but we also change what the map represents. Two samples that look identical on their non-missing dimensions will map to the same neuron, even if they originally differed on a now-masked dimension. The clusters that emerge from the SOM no longer have a uniform feature basis — different neurons may have been shaped by different subsets of dimensions.

**Why**
This is an inherent consequence of the mask-based approach. The mask is necessary (without it the map tears apart, as described in Issue 3), but it introduces asymmetry: sample A might contribute to a neuron's weight in all 10 dimensions, while sample B only contributes in 7. The neuron's final weight vector is shaped by a mix of complete and incomplete observations. This is invisible in the raw SOM output — the weights and visualizations look normal.

**How It Must Be Handled — Data Mining Module**
This issue cannot be fully solved within the SOM module alone. It requires awareness at the interpretation stage. The future Data Mining module must:

- Track which dimensions were masked for each sample and which neurons they contributed to.
- Flag clusters where a significant portion of contributing samples had masked dimensions.
- Avoid drawing conclusions about masked dimensions from neuron weights that were not informed by those dimensions.
- Consider presenting cluster analyses separately for fully-observed samples versus samples with missing data, to make the distinction visible to the end user.

This is a system-level concern. Solving it requires changes that span from the SOM output format through to the final user-facing report.

---

## Novější problémy a rozhodnutí (zkrácený formát)

5. **Výpočet topographic error v Python smyčce** — Python for-loop přes všechny vzorky způsoboval ~100× zpomalení logování; nahrazeno NumPy vektorizací s broadcasting.

6. **Hybrid batch mode jako jediný tréninkový mód** — tři módy (stochastic / deterministic / hybrid) byly redundantní; hybrid pokrývá krajní případy nastavením batch velikosti; `processing_type` odstraněno ze search space i kódu.

7. **Řídké checkpointy při dlouhém tréninku** — 25 checkpointů na 15 000 iterací nestačí pro LSTM trénink; přidán flag `checkpoint_every_mqe` pro záznam při každém výpočtu MQE (~500 bodů na běh).

8. **Checkpoint[0] jako baseline pro normalizaci MQE** — první checkpoint reprezentuje stav blízký náhodné inicializaci; `initial_mqe` z checkpoint[0] slouží jako dataset- a map-size-nezávislá baseline pro `mqe_improvement_ratio`.

9. **Dead ratio a topographic error nepotřebují normalizaci** — obě metriky jsou přirozeně v [0, 1] jako podíly; pokus o normalizaci ratiem initial hodnoty měnil charakter dat bez přínosu.

---

## Module cleanup and restructuring — short records

10. **Duplicate result analyzer** — the module accumulated a full copy of the post-run analysis (`som/result_analyzer.py`, 302 lines) after the standalone `app/analysis` module took over the same role. Nobody imported it anymore; deleted. Lesson: incremental rewrites leave orphaned predecessors behind unless usage is audited.

11. **Silently broken visualizations** — two rendering features had been dead for an unknown time without anyone noticing: the cluster map was never generated (it looked for `clusters.json` in the wrong directory after an output-layout change) and the best-MQE marker never appeared in the MQE plot (result key renamed, plot code not updated). Discovered only during a systematic usage audit; fixed and locked by an end-to-end output-layout test so the contract cannot rot silently again.

12. **Primary ID classified as an analyzable feature** — training itself was always protected (the ignore mask marks the ID column by default), but column *classification* treated the ID like any numeric column: it landed in `numerical_column`, so outlier detection reported the highest/lowest ID as a "global extreme" (filtered away downstream by string matching). The ID is now excluded from feature lists at the classification source. Open follow-up: the ID column is still normalized and carried as a dead dimension in the training matrix — see #20.

13. **Training entangled with disk IO** — `train()` wrote weights, checkpoints, and coverage files itself and logged into a working directory. This forced the EA to pay file-system overhead for every evaluated individual and made in-memory experimentation impossible. Resolved by making `train()` pure compute and extracting a persistence layer; the orchestrator decides what to save.

14. **Preprocessing mutated the experiment config** — `preprocess_data` injected derived column lists into the caller's config dict and returned a path to a file it had written, creating hidden coupling across the pipeline. Replaced by a pure function returning a `PreprocessResult` dataclass; artifacts are persisted explicitly.

15. **Visualizations required a live SOM object** — maps could only be rendered during the run that trained them, so stored results could not be re-rendered for the UI or ablation comparisons. All rendering now works from saved artifacts (`weights.npy` + map type), with `render_results_dir()` as the re-rendering entry point.

16. **Per-individual artifacts slowed the EA** — every EA individual wrote final weights, a readable CSV, and training plots even though nothing downstream consumed the weights after the CNN track closed. Per-individual weights are now opt-in and plot generation is configurable, removing needless IO from runs with hundreds of trainings.

17. **U-Matrix computed in nested Python loops** — per-neuron neighbor iteration was replaced by vectorized NumPy shifts (with row-parity offsets for hex grids); equivalence with the original loop is locked by tests. Note: the EA-side U-matrix *metric* intentionally keeps its older 4-neighbor approximation to stay comparable with thousands of already-evaluated individuals.

18. **No multi-seed capability for single SOM runs** — reviewers (R2.4) and the ablation study require mean ± std over independent runs, but only the EA supported multiple seeds. Added a programmatic `run_pipeline(..., seed=...)` API and a multi-seed tool comparing final metrics, MQE evolution curves, and clustering stability (pairwise Adjusted Rand Index) across seeds.

19. **Preprocessing was all-or-nothing** — the ablation study needs to quantify what preprocessing contributes (known effect: without normalization or the ignore mask, organization collapses completely). Introduced `preprocess_strategy` (`nexus` / `scale-only` / `none`) so pipeline stages can be switched off in controlled steps; the strategy is recorded in the run's metadata.

20. **The masked ID dimension leaked into unmasked computations** — the ID column was normalized and kept as a dimension of the training matrix, but because every sample masks it, its weight dimension never updated and stayed random initialization noise. Mask-aware computations (BMU during training, QE, TE) were unaffected, but several computations ran *without* the mask and absorbed this noise: the analysis-phase BMU assignment (`clusters.json` could disagree with training BMUs for borderline samples), dead-neuron counting, hit/dead maps, and the weight-only metrics (U-matrix stats, topological correlation) used as EA objectives. Simply skipping normalization of the ID column would have made it worse — raw ID magnitudes would dominate unmasked distances. **Resolved by making fully-masked dimensions inert on both sides**: the training matrix zeroes columns masked for all samples, `train()` zeroes their weight dimensions at start (zero here is not a data value — the dimension carries no information by construction; per-sample missing values keep the mask semantics untouched). Additionally, the analysis-phase BMU assignment, per-dimension QE, dead-neuron counting, and hit/dead maps now honor the per-sample mask, so a median-filled missing value can no longer pull a sample to a different neuron than training would. Linking organized results back to the original dataset is unaffected — it works through row order and the ID in `original_input.csv`, never through the training matrix.

21. **What a partially masked value means for weight updates** — when one sample has an invalid (masked) value in dimension *k*, the sample still competes for a BMU using its valid dimensions, and the update it triggers adjusts the BMU neighborhood in all dimensions *except k*. The neuron's dim-*k* weight is therefore shaped exclusively by samples that actually have a valid value there; the median fill exists only so the matrix contains no NaN and never enters training. Consequence for interpretation (see also #4): a neuron's dim-*k* weight can be based on a subset of its assigned samples.

22. **Topological collapse when fitness ignores topology** (historical, from the Phase 1 article experiments) — on the Student Habits dataset the EA, whose fitness was driven strictly by MQE and runtime, converged into a suboptimal local minimum with an excessively small initial radius. The result was a severe topological collapse: a dense cluster forced into the center of the map, with hybrid MQE visibly worse than stochastic. This artifact is what motivated adding explicit topographic metrics into the optimization — the current EA minimizes `[raw_mqe_ratio, topo_error, dead_ratio]` as separate NSGA-II objectives, which prevents this failure mode by construction.

23. **ISOMAP topology plots inverted expectations — and exposed two deceptive metrics** — on a deterministic Swiss Roll run, PCA showed a clean planar SOM grid while joint-fit ISOMAP collapsed into a ring, the opposite of the naive expectation ("ISOMAP unrolls a correct map"). Investigation: the dataset alone unrolls perfectly (|corr(t, embedding axis)| = 0.992 with ground truth), but adding the 225 SOM weight vectors into the ISOMAP neighbor graph drops it to 0.187 — **the SOM organized as a flat principal plane cutting through the roll's interior, and its weights bridged the spiral layers**, short-circuiting the geodesics. The clean PCA grid was the *symptom*, not success: the weights genuinely lie near a plane. Equally deceptive: MQE 0.078 and topographic error 0.04 looked fine, because TE measures local grid consistency, which a flat plane satisfies while completely ignoring the manifold. Consequences: (a) `--isomap-fit data|joint` added to the topology tool — data-only fit gives the honest verification view (manifold-following SOM lands as a coherent grid, layer-bridging SOM scatters with crossing edges); (b) interpretation guide written into `RUN.md`; (c) confirms the need for a ground-truth manifold-adherence metric (corr with unrolling parameter t) — `article_implementation.md` item 4.

24. **No single topology metric suffices — and UMAP/t-SNE are the wrong lens for grid verification** — the S-Curve benchmark run exposed both halves of this. (a) UMAP and t-SNE plots "always looked weird" — not a rendering bug: both are neighbor-embedding methods that tear a continuous manifold into clumps and discard global distances, so straight grid edges drawn between embedded neurons cross between fragments — projection artifacts indistinguishable from SOM errors. PCA (global linear) and ISOMAP (global geodesic) are the correct lenses for grid coherence; UMAP/t-SNE remain useful only for cluster-structured data. (b) The first adherence metric (linear grid→parameter R²) flagged the S-Curve `height` as FAIL (R²=0.10) although the map was excellent — the parameter was mapped along a *curved* grid direction (kNN R²=0.96, per-neuron ANOVA 0.98). Switching to kNN R² alone then over-praised the deterministic Swiss Roll run, which is locally coherent but *folded across the manifold*. Resolution: the verdict now requires a **local/global metric pair** — kNN `grid_param_R2` (local adherence) AND pairwise-distance Spearman on standardized ground truth (global structure; ideal map calibrates to ≈ 0.98). Resulting scale on real runs: ideal 0.98 → bent-but-correct S-Curve 0.52 (WARN) → folded Swiss Roll 0.31 (FAIL).

25. **Hex U-Matrix used mirrored row parity — a real computational error found by auditing the visualization chain** — the U-Matrix neighbor offsets for hexagonal maps had the even/odd row diagonals swapped relative to the cube-coordinate convention that the training (`update_weights`), topographic error, and the topology-plot grid edges all share (odd-r: odd rows shifted right). Two of the six neighbors per neuron were wrong, in both the original loop implementation and its faithful vectorization — so every hex U-Matrix ever rendered averaged a partially wrong neighbor set. Subtle visually, which is why it survived; caught only by cross-checking each component of the visualization chain against `KohonenSOM.cube_coords` as the single source of truth. Fixed; the equivalence test now references cube distance directly instead of the legacy loop. The aggregate `calculate_u_matrix_metrics()` (EA objective) was not affected — it uses its own documented 4-neighbor approximation.

26. **`map_type: "rect"` crashed training — one branch used the opposite default** — the codebase convention everywhere (training init, topographic error, visualizations, verification tools) is `if map_type == 'hex'` with everything else treated as square, so the `"rect"` value used by benchmark configs and the UI worked — except in `update_weights`, which tested `== 'square'` and sent `"rect"` down the hex path, crashing on the hex-only `cube_coords` attribute at the first weight update. Found by the Helix chain benchmark (first `rect` run after the cleanup). The branch now matches the convention; locked by a regression test. Lesson: a string enum decided by exclusion must use the *same* exclusion everywhere.

27. **Spatial stats crashed on chain maps (1×N)** — `compute_spatial_stats` called `np.gradient(plane)` over both grid axes, but a 1×100 chain map (helix, space-filling benchmarks) has a singleton row axis with no defined gradient, which aborted the analysis stage of the pipeline. The gradient is now computed per axis only where the axis has ≥ 2 elements (zero contribution otherwise); Moran's I, extrema counting, and boundary ratio already tolerated singleton axes. Chain maps had simply never passed through the full pipeline before the benchmark battery existed.

28. **Raw ARI condemned a perfect map — granularity mismatch in label verification** — on the blobs benchmark `verify_topology` reported ARI = 0.05 next to purity = 1.0 and a visually perfect 5-cluster separation. Not a SOM failure: ARI between the neuron-level partition (~190 micro-clusters) and 5 labels is near zero *by construction*, because the convention cluster = neuron makes the compared partitions differ in granularity. The verdict now uses ARI after merging neurons by their dominant label (blobs: 1.0) combined with neuron purity (either alone can be gamed); raw neuron-level ARI stays in the report as informational.

29. **Compare-plot colouring was meaningless noise** — `plot_som_topology --compare` coloured points by CSV row index, assuming rows are ordered by the manifold parameter ("≈ unrolling parameter t for Swiss Roll"). Generated benchmarks sample the parameter randomly, so the colours were pure speckle on every compare plot ever rendered, silently discarding the plot's main diagnostic (colour continuity along the grid reveals folding). Points are now coloured by the first column of the auto-discovered `*_groundtruth.csv` (neurons by the mean of their assigned samples), falling back to row index only when no ground truth exists — the colorbar states which. Panel titles also switched to English per the project convention.

30. **Space-filling chain gets a citable number** — the space-filling benchmark has no ground-truth parameter by design (uniform 2D points), so verification ended at NO-GROUNDTRUTH although its defining property — the chain winds without crossing itself — is exactly computable. Added a self-crossing count of the chain polyline in data space (proper segment intersections, non-adjacent segments) to `verify_topology` for 1×N maps with 2 active dims; the reference run scores 0 crossings over 99 segments → PASS.


31. **The "hybrid coverage mechanism" (dataset splitting) measured useless — replaced by guaranteed-coverage reshuffling** — the long-open coverage investigation (article R1.3, `article_implementation.md` item 3) was closed by simulation at statistical scale (`app/tools/coverage_sim.py`, 17 000+ simulated runs, 4 datasets, random seeds). Findings: (a) the `track_sample_coverage` counting was always correct — the simulator replays a real fixed-seed run *sample by sample*; (b) per-sample hit counts under the implemented `random` sampling match the iid Poisson(λ) null model to four decimal places at every `num_batches` once the budget is normalized — random index sections cannot improve coverage *by construction*, and the historical impression that splitting helps conflated stratification with a hidden budget multiplication (`samples_per_section` is derived from *total* dataset size, so `num_batches = b` silently processes b× more samples per iteration); (c) guaranteed coverage requires sampling *without replacement across iterations* — added as `sampling_method: reshuffle` (per-epoch re-shuffled permutation walked by a pointer; hit counts equal ±1 independent of seed; the order stays random every epoch, so no data-presentation bias). Measured: `reshuffle` reaches 100 % coverage at `epoch_multiplier = 1` where `random` needs ≈ 10 for 99.9 % (never-visited ≈ e^(−em), dataset-size independent) — a 10× pass saving for stochastic runs. `num_batches` is fixed at 1 and dropped from the EA search space; the default stays `sampling_method: random` until the map-quality A/B (MQE/TE/ρ, `SEARCH_SPACE.md` experiment C) confirms no regression. Full decision trail: `docs/ea/SEARCH_SPACE.md` step 1 protocol. *[update 2026-06-12, same day: experiment C (quality A/B, 12 arms × 10 seeds, Mann-Whitney) found no significant regression in any of 24 dataset×metric comparisons at equal budget — the only significant difference favors reshuffle (WineQuality MQE, p = 0.005); ground-truth topology indistinguishable; reshuffle is 7–30 % faster (O(1) pointer vs O(N) choice per draw). The equal-coverage arm showed quality tracks budget, not coverage (MQE +9…+46 % at 1/3 passes) — the defensible claim is "guaranteed coverage, equal-or-better quality, faster at the same budget", not "same quality for less compute". The default flipped to `sampling_method: reshuffle`; pre-flip configs must state `random` explicitly to stay replayable.]*
