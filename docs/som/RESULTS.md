# SOM Results Directory Guide

Every run produces one self-contained results directory (default:
`results/<YYYYMMDD_HHMMSS>/` next to the input CSV). The layout below is the
**stable contract** — it is enforced by
`tests/integration/test_som_pipeline.py::test_output_layout` and consumed by
the analysis module, the REST API import, the tools, and the UI.

```
<results_dir>/
├── log.txt                          timestamped log of the whole pipeline
├── run_metrics.json                 headline metrics of the run
├── dataset_meta.json                ds_* dataset statistics (NN features)
├── csv/
│   ├── original_input.csv           input data as received (after validation)
│   ├── training_data.npy            normalized (N, dim) training matrix
│   ├── training_data_readable.csv   same, human-readable
│   ├── ignore_mask.csv              boolean (N, dim); True = dim invisible
│   ├── weights.npy                  final weights (m, n, dim)
│   ├── weights_readable.csv         neuron_i, neuron_j, dim_0..dim_k
│   ├── sample_assignments.csv       per-sample BMU + QE + per-dimension QE
│   ├── training_checkpoints.json    [save_checkpoints] training trajectory
│   └── sample_coverage.json         [track_sample_coverage] per-vector counts
├── json/
│   ├── preprocessing_info.json      per-column classification + reasons
│   ├── clusters.json                {neuron_key: [sample_ids]} (cluster = BMU)
│   ├── quantization_errors.json     total + per-neuron QE
│   ├── extremes.json                {sample_id: [outlier reasons]}
│   ├── pie_data_<col>.json          per-categorical neuron distributions
│   └── llm_context.json             aggregated context for the LLM layer
└── visualizations/                  [save_visualizations / save_training_plots]
    ├── u_matrix.png                 cluster boundaries (neighbor distances)
    ├── hit_map.png                  samples per neuron (with counts)
    ├── distance_map.png             per-neuron quantization error
    ├── dead_neurons_map.png         binary activity map (black = dead)
    ├── cluster_map.png              active neurons, one color per cluster
    ├── component_<col>.png          one weight plane per training column
    ├── pie_map_<col>.png            categorical distribution per neuron
    ├── mqe_evolution.png            MQE over iterations (best point marked)
    ├── learning_rate_decay.png      effective LR curve
    ├── radius_decay.png             effective radius curve
    ├── batch_size_growth.png        processed samples per iteration
    └── legends/                     standalone colorbars/legends per map
```

Conditional files are marked with `[config key]`. With
`save_visualizations: false` the run is much faster and maps can be
regenerated later via `som.visualization.render_results_dir(results_dir)`.

## Key files

### `run_metrics.json`

```json
{
  "map_size": [30, 30],
  "map_topology": "hex",
  "best_mqe": 0.0421,
  "topographic_error": 0.0156,
  "duration": 12.84,
  "lstm_stopped": false,
  "lstm_stop_progress": null
}
```

`topographic_error` is computed directly on the final weights (share of
samples whose BMU and second-BMU are not grid neighbors).

### `dataset_meta.json`

`ds_*` statistics of the input (samples, dimensions, numeric/categorical/
ignored counts, missing ratio, `ds_preprocess_strategy`). Doubles as feature
input for the MLP Oracle.

### `csv/sample_assignments.csv`

One row per sample: `sample_id, bmu_i, bmu_j, bmu_key, qe,
qe_dim_<col>..., is_outlier`. The per-dimension QE columns explain *why* a
sample is anomalous (which feature contributes the error) — see
`app/tools/plot_dim_qe.py` for heatmaps.

### `json/clusters.json`

Clustering output. No separate clustering algorithm is used: **a cluster is
the set of samples assigned to one neuron (BMU)**. Neuron key format `"i_j"`.

### `json/llm_context.json`

Aggregated, LLM-ready context built by `app/analysis/` (`map` overview incl.
`spatial_quality_score`, T&C and silhouette; per-cluster stats with dominant
categories/purity/z-score deviations; ranked anomalies with annotated full
rows; global dimension stats; `spatial_analysis` — gradients, Moran's I,
dominant-category regions). This file is what the LLM report/chat consumes
today; the agentic direction queries the same data through the REST API.

### `csv/training_checkpoints.json`

Trajectory records `{iteration, progress, mqe, topographic_error,
dead_neuron_ratio, learning_rate, radius, lr_factor, radius_factor}` —
training data for the LSTM models and input for the multi-seed MQE
comparison.

### `csv/sample_coverage.json`

Per-input-vector processing counts plus summary stats (min/max/mean/std,
never-processed ratio). Basis for the hybrid-coverage claim — currently an
open investigation (`article_implementation.md` item 3).

## Multi-seed comparison outputs

`app/tools/multi_seed_som.py` writes one level above the per-seed dirs:

```
<base_dir>/
├── seed_<k>/                     full results dir per seed (layout above)
├── multi_seed_metrics.csv        one row per seed
├── multi_seed_summary.json       {metric: {mean, std, min, max, values}}
│                                 + clustering_stability_ari (pairwise ARI)
└── mqe_evolution_comparison.png  overlaid MQE curves
```
