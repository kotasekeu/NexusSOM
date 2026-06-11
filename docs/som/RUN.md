# Running the SOM Pipeline

One run = validate → preprocess → train → analyze → LLM context → plots → maps,
producing a self-contained results directory (see `RESULTS.md`).

Always use the project venv interpreter `.venv/bin/python3` from the repo root.

## CLI

```bash
.venv/bin/python3 app/run_som.py -i data/datasets/Iris/iris.csv -c data/datasets/Iris/config-som.json
```

| Argument | Required | Description |
|---|---|---|
| `-i`, `--input` | yes | Input CSV file. |
| `-c`, `--config` | yes | JSON configuration (see `CONFIG.md`). |
| `-o`, `--output` | no | Results directory. Default: timestamped `results/<YYYYMMDD_HHMMSS>/` next to the input CSV. |
| `-s`, `--seed` | no | Overrides `random_seed` from the config — for repeated runs of one configuration. |

Exit code is non-zero on failure; errors are printed and logged into
`<results_dir>/log.txt` when the directory already exists.

## Programmatic API

The same pipeline as a function — used by the multi-seed tool, ablation
tooling, and the UI (no subprocess needed):

```python
from som.run import run_pipeline   # app/ must be on sys.path

results_dir = run_pipeline(
    'data/datasets/Iris/iris.csv',
    'data/datasets/Iris/config-som.json',  # dict or path
    output_dir=None,                       # default: timestamped dir
    seed=7,                                # optional random_seed override
)
```

`run_pipeline` does not mutate the passed config dict and raises on failure.
The module's public API is defined in `app/som/__init__.py`
(`KohonenSOM`, `preprocess_data`, `validate_input_data`, `PreprocessResult`,
`persistence`).

## Multi-seed comparison

Statistical robustness of one configuration (mean ± std, MQE-curve overlay,
clustering stability via Adjusted Rand Index):

```bash
.venv/bin/python3 app/tools/multi_seed_som.py -i data.csv -c config.json -n 10
.venv/bin/python3 app/tools/multi_seed_som.py -i data.csv -c config.json --seeds 1 7 42
.venv/bin/python3 app/tools/multi_seed_som.py -i data.csv -c config.json -n 5 --with-maps
```

Outputs `multi_seed_metrics.csv`, `multi_seed_summary.json`, and
`mqe_evolution_comparison.png` next to the per-seed results dirs. Maps and
training plots are skipped by default for speed (`--with-maps` enables them).

## Re-rendering maps for a stored run

Maps can be (re)generated for any saved run purely from its artifacts —
no re-training, no live SOM object:

```python
from som.visualization import render_results_dir

viz_dir = render_results_dir('data/datasets/Iris/results/20260610_120000')
```

This is how the UI displays runs that were executed with
`save_visualizations: false`.

## Topology verification plots

Visual proof that the SOM preserved the data topology — the map grid is
projected into the data space next to the training samples. Primary
verification tool for manifold benchmarks (Swiss Roll, space filling):

```bash
# Everything at once: 2D for all projections, 3D where supported,
# interactive HTML per projection, and the PCA/ISOMAP compare grid
.venv/bin/python3 app/tools/plot_som_topology.py <results_dir> --all

# Single projection
.venv/bin/python3 app/tools/plot_som_topology.py <results_dir> --projection isomap
.venv/bin/python3 app/tools/plot_som_topology.py <results_dir> --projection umap --3d --html
.venv/bin/python3 app/tools/plot_som_topology.py <results_dir> --compare

# ROTATABLE interactive 3D (Plotly — drag rotates the camera; a single fixed
# viewpoint often hides whether the grid is a clean sheet or a folded one)
.venv/bin/python3 app/tools/plot_som_topology.py <results_dir> --projection isomap --html3d
.venv/bin/python3 app/tools/plot_som_topology.py <results_dir> --projection raw --html3d
```

| Projection | Character | When to use |
|---|---|---|
| `raw` | **no transformation at all** | the honest ground view for benchmark datasets with 2–3 active dims (Swiss Roll, S-Curve, helix, plane) — combined with `--html3d` you rotate the actual data space |
| `pca` | linear, fast | quick look. ⚠ Deceptive on manifolds: a SOM that ignores the Swiss Roll and spans a flat plane through it shows a *clean grid* in PCA — a nice PCA grid is NOT proof of manifold adherence |
| `umap` | non-linear neighbor embedding | ⚠ **unsuitable for grid-edge verification on continuous manifolds** — tears the sheet into clumps, so grid edges cross between fragments (projection artifacts, not SOM errors). Useful only for cluster-structured data (blobs). Requires `umap-learn` |
| `tsne` | non-linear neighbor embedding | same caveat as UMAP — preserves only local neighborhoods, global distances are meaningless; no 3D support |
| `isomap` | geodesic-distance preserving | **primary manifold verification** — see interpretation guide below |
| `--compare` | 2×2 grid: (data \| weights) × (PCA \| ISOMAP) | one-image ablation/verification summary |
| `--html` | interactive Plotly (zoom, hover with neuron info) | manual inspection; one file per projection |

### Interpreting ISOMAP on manifold benchmarks (Swiss Roll)

ISOMAP has two fit modes (`--isomap-fit`, see `issues.md` #23 for the full story):

- **`joint`** (default): data + weights in one neighbor graph. No out-of-sample
  artifacts, but a SOM whose weights cut through the roll's interior **bridges
  the spiral layers** — geodesics short-circuit and the embedding collapses
  into a **ring instead of unrolling**. Measured on the Swiss Roll benchmark:
  data alone unrolls with |corr(t, axis)| = 0.99; adding the weights of a
  layer-bridging SOM drops it to 0.19.
- **`data`**: fit on data only, weights mapped out-of-sample (Nyström —
  boundary neurons less precise). The manifold always unrolls faithfully here,
  so this is the honest verification view of the *SOM*.

Reading the verdict:

| Observation | Meaning |
|---|---|
| `data` fit: SOM grid lies as a coherent sheet over the unrolled data | ✅ SOM follows the manifold |
| `data` fit: SOM grid scattered with many crossing/stretched edges | ❌ SOM bridges manifold layers (e.g. flat principal plane through the roll) |
| `joint` fit: ring/annulus instead of an unrolled sheet | ❌ same failure — the SOM weights themselves blocked the unrolling |

⚠ Low MQE and low topographic error do **not** detect this failure — a flat
plane through the roll has excellent local grid consistency. Manifold adherence
needs the ground-truth metric (correlation with the unrolling parameter `t`
from `swiss_roll_groundtruth.csv`) — planned as `article_implementation.md`
item 4.

Outputs land in the results dir: `topology_2d_<method>.png`,
`topology_3d_<method>.png` (pca/umap/isomap/raw),
`topology_interactive_<method>.html` (2D),
`topology_interactive_3d_<method>.html` (rotatable 3D),
`topology_compare_pca_isomap.png`. `--all` generates all of them, including
the raw views when the dataset has 2 or 3 active dims.

Useful flags: `--grid-only` (hide samples, pure grid geometry), `--density`
(hexbin instead of scatter), `--hex` (force hex connectivity; normally read
from `run_metrics.json`), `--elev/--azim` (3D camera). With `--all`, a missing
optional dependency (umap-learn, plotly) skips that output instead of aborting
the batch. Fully-masked dimensions (primary ID) are excluded from projection
automatically.

## Benchmark datasets and quantitative verification

Ground-truth benchmarks prove that the SOM organization works and that there
is no mathematical error — each targets one property (full table in
`app/tools/generate_benchmark.py`):

```bash
# Generate one or all benchmarks (dataset + ground truth + tuned config-som.json)
.venv/bin/python3 app/tools/generate_benchmark.py all
.venv/bin/python3 app/tools/generate_benchmark.py s-curve --samples 3000
```

| Benchmark | Verifies |
|---|---|
| `swiss-roll` | manifold unfolding — hard case (layered spiral) |
| `s-curve` | manifold unfolding — easier case (no layers); a SOM failing here signals a real defect |
| `helix` | 1D chain ordering along a curve |
| `torus` | closed manifold — documented imperfection expected (planar sheet cannot wrap a torus) |
| `blobs` | cluster separation (label ground truth → ARI, purity) |
| `noisy-plane` | robustness to numeric noise dimensions (preprocess ablation input) |
| `uniform-cube` | **negative control** — no structure; the system must not invent clusters |
| `space-filling` | 1×N chain winding without self-crossings |

After running the SOM on a benchmark, produce the evidence:

```bash
.venv/bin/python3 app/tools/verify_topology.py <results_dir>          # auto-finds groundtruth
.venv/bin/python3 app/tools/verify_topology.py <results_dir> -g path/to/x_groundtruth.csv
```

The verdict combines two complementary views (`issues.md` #24 — neither alone
is sufficient):

- **Local adherence** per manifold parameter: `grid_param_R2` (cross-validated
  kNN regression grid → parameter, PASS ≥ 0.8; tolerates curved mappings),
  `neuron_anova_R2` (does the neuron determine the parameter), `linear_R2`
  (strict axis-aligned mapping), Spearman per axis.
- **Global structure**: Spearman of pairwise distances (standardized ground
  truth vs grid, PASS ≥ 0.7). Detects maps **folded across the manifold**,
  which local metrics cannot see. Calibration: an ideal 15×15 mapping scores
  ≈ 0.98 on both Swiss Roll and S-Curve.

Plus trustworthiness/continuity, ARI + purity for labeled benchmarks, and
hit-distribution stats (dead ratio, Gini) as the negative control. Console
verdict PASS/WARN/FAIL + machine-readable `json/verify_topology.json` —
citable in the ablation study.

Worked example — why the metric *pair* matters (both runs 15×15 hex,
healthy-looking MQE/TE/T&C):

| Run | local kNN R² (t / height) | linear R² | global pairwise | Verdict |
|---|---|---|---|---|
| S-Curve | 1.00 / 0.96 | 0.86 / 0.10 | 0.52 | **WARN** — locally correct sheet, globally bent (benign for clustering, documented) |
| Swiss Roll (deterministic) | 0.96 / 0.96 | 0.02 / 0.82 | 0.31 | **FAIL** — locally coherent patches but the map is folded across the spiral (confirmed visually by ISOMAP `--isomap-fit data`) |

Other verification tools (not tests):

```bash
# Per-dimension QE heatmaps — which feature drives the error where
.venv/bin/python3 app/tools/plot_dim_qe.py <results_dir>

# EA run sanity verification
.venv/bin/python3 app/tools/verify_ea_run.py <ea_results_dir>
```

## Typical workflows

```bash
# Fast batch run (no plots/maps), render maps later only for the winner
.venv/bin/python3 app/run_som.py -i data.csv -c config-fast.json -o out/run1
# config-fast.json: {"save_training_plots": false, "save_visualizations": false, ...}

# Ablation A1.1 — preprocessing ladder (see docs/global/ABLATION_STUDY.md)
# Same config, three values of "preprocess_strategy": nexus | scale-only | none
```
