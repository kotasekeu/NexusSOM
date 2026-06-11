"""
verify_topology.py — Quantitative verification of SOM topology preservation
against benchmark ground truth.

This is the evidence generator for the ablation study: it turns "the map looks
organized" into numbers. Visual plots (plot_som_topology.py) show *where* it
breaks; this tool proves *whether* and *how much*. Motivated by
docs/som/issues.md #23 — low MQE/TE do not detect manifold failures.

Metrics (per ground-truth parameter column):
  - grid_param_R2        cross-validated kNN regression (bmu grid position ->
                         parameter); the main manifold-adherence score. Robust
                         to curved/folded but coherent parametrizations — a
                         purely linear fit produced false negatives on the
                         S-Curve benchmark (height was mapped along a curved
                         grid direction: linear R2 = 0.10, kNN R2 = 0.96).
  - neuron_anova_R2      how much the neuron assignment alone determines the
                         parameter (1 - within-neuron variance / total) —
                         parametrization-free locality check.
  - linear_R2            strict axis-aligned fit; high only when the parameter
                         maps onto a straight grid direction.
  - best_axis_spearman   |Spearman| of the parameter vs the better grid axis.

Global metrics:
  - pairwise_distance_spearman   Spearman between sample-pair distances in
                                 ground-truth space and on the SOM grid.
  - trustworthiness / continuity grid positions vs ground-truth coordinates.
  - For label columns (categorical ground truth, e.g. blobs):
    adjusted_rand_index (label vs neuron) and mean neuron purity.
  - Always: hit-distribution stats (dead ratio, density Gini) — negative
    control for "the system must not invent structure".

Usage:
  python app/tools/verify_topology.py <results_dir>                  # auto-find groundtruth
  python app/tools/verify_topology.py <results_dir> -g path/to/x_groundtruth.csv
  python app/tools/verify_topology.py <results_dir> --pairs 5000 --neighbors 12

Output: console report + <results_dir>/json/verify_topology.json
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# Verdict thresholds — single source for the console report and the JSON.
# Local adherence (kNN R2) and global structure (pairwise distances) must BOTH
# pass: a map folded across the manifold is locally coherent (high kNN R2,
# high ANOVA) yet globally wrong (low pairwise correlation) — observed on the
# deterministic Swiss Roll run (docs/som/issues.md #23/#24).
R2_PASS = 0.80          # grid locally explains >= 80 % of a manifold parameter
R2_WARN = 0.50
GLOBAL_PASS = 0.70      # pairwise distance Spearman (standardized gt params)
GLOBAL_WARN = 0.50
ARI_PASS = 0.80
ARI_WARN = 0.50


# ─── Loading ──────────────────────────────────────────────────────────────────

def find_groundtruth(results_dir: str) -> str | None:
    """Look for *_groundtruth.csv next to the dataset (two levels up:
    <dataset_dir>/results/<timestamp>)."""
    dataset_dir = os.path.dirname(os.path.dirname(os.path.abspath(results_dir)))
    matches = sorted(glob.glob(os.path.join(dataset_dir, '*_groundtruth.csv')))
    return matches[0] if matches else None


def load_run(results_dir: str) -> tuple[pd.DataFrame, dict]:
    sa_path = os.path.join(results_dir, 'csv', 'sample_assignments.csv')
    if not os.path.isfile(sa_path):
        sys.exit(f'ERROR: {sa_path} not found — run the SOM pipeline first.')
    assignments = pd.read_csv(sa_path)

    rm_path = os.path.join(results_dir, 'run_metrics.json')
    run_metrics = {}
    if os.path.isfile(rm_path):
        with open(rm_path, encoding='utf-8') as f:
            run_metrics = json.load(f)
    return assignments, run_metrics


def grid_coords(assignments: pd.DataFrame, map_type: str) -> np.ndarray:
    """Physical 2D positions of each sample's BMU (hex rows are offset)."""
    i = assignments['bmu_i'].values.astype(float)
    j = assignments['bmu_j'].values.astype(float)
    if map_type == 'hex':
        x = j + 0.5 * (i % 2)
        y = i * (np.sqrt(3) / 2)
    else:
        x, y = j, i
    return np.stack([x, y], axis=1)


def split_gt_columns(gt: pd.DataFrame) -> tuple[list, list]:
    """Numeric continuous columns = manifold params; low-cardinality integer
    columns = cluster labels."""
    params, labels = [], []
    for col in gt.columns:
        if col == 'id':
            continue
        series = gt[col]
        if pd.api.types.is_integer_dtype(series) and series.nunique() <= 50:
            labels.append(col)
        elif pd.api.types.is_numeric_dtype(series):
            params.append(col)
    return params, labels


# ─── Metrics ──────────────────────────────────────────────────────────────────

def param_adherence(param: np.ndarray, grid_xy: np.ndarray) -> dict:
    """How well the grid position explains one manifold parameter."""
    from scipy.stats import spearmanr
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsRegressor

    # Primary score: cross-validated kNN regression on grid coordinates.
    # Tolerates curved/folded but coherent parametrizations of the grid.
    k = max(3, min(10, len(param) // 20))
    knn_r2 = float(cross_val_score(KNeighborsRegressor(n_neighbors=k),
                                   grid_xy, param, cv=5, scoring='r2').mean())

    # Secondary: strict linear fit (with intercept): param ~ a*x + b*y + c
    A = np.hstack([grid_xy, np.ones((len(grid_xy), 1))])
    coef, *_ = np.linalg.lstsq(A, param, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((param - pred) ** 2))
    ss_tot = float(np.sum((param - param.mean()) ** 2))
    linear_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    rho_x = abs(spearmanr(param, grid_xy[:, 0]).statistic)
    rho_y = abs(spearmanr(param, grid_xy[:, 1]).statistic)
    return {
        'grid_param_R2': round(max(0.0, knn_r2), 4),
        'linear_R2': round(linear_r2, 4),
        'best_axis_spearman': round(float(max(rho_x, rho_y)), 4),
    }


def neuron_anova_r2(param: np.ndarray, bmu_keys: pd.Series) -> float:
    """Share of parameter variance explained by the neuron assignment alone
    (1 - mean within-neuron variance / total variance)."""
    df = pd.DataFrame({'p': param, 'n': bmu_keys.values})
    total_var = float(df['p'].var(ddof=0))
    if total_var == 0:
        return 0.0
    within = float(df.groupby('n')['p'].var(ddof=0)
                   .mul(df.groupby('n').size()).sum() / len(df))
    return round(1.0 - within / total_var, 4)


def _standardize(params: np.ndarray) -> np.ndarray:
    """Z-score per column so no manifold parameter dominates the distances."""
    std = params.std(axis=0)
    std[std == 0] = 1.0
    return (params - params.mean(axis=0)) / std


def pairwise_distance_spearman(gt_params: np.ndarray, grid_xy: np.ndarray,
                               n_pairs: int = 2000, seed: int = 42) -> float:
    """Spearman between ground-truth-space and grid-space pair distances —
    the GLOBAL structure metric (detects maps folded across the manifold)."""
    from scipy.stats import spearmanr
    gt_params = _standardize(gt_params)
    rng = np.random.default_rng(seed)
    n = len(grid_xy)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    valid = a != b
    a, b = a[valid], b[valid]
    d_gt = np.linalg.norm(gt_params[a] - gt_params[b], axis=1)
    d_grid = np.linalg.norm(grid_xy[a] - grid_xy[b], axis=1)
    return round(float(spearmanr(d_gt, d_grid).statistic), 4)


def trustworthiness_continuity(gt_params: np.ndarray, grid_xy: np.ndarray,
                               n_neighbors: int = 10,
                               max_samples: int = 2000, seed: int = 42) -> dict:
    """T&C between the ground-truth manifold coordinates and grid positions."""
    from sklearn.manifold import trustworthiness
    rng = np.random.default_rng(seed)
    if len(grid_xy) > max_samples:
        idx = rng.choice(len(grid_xy), max_samples, replace=False)
        gt_params, grid_xy = gt_params[idx], grid_xy[idx]
    return {
        'trustworthiness': round(float(
            trustworthiness(gt_params, grid_xy, n_neighbors=n_neighbors)), 4),
        'continuity': round(float(
            trustworthiness(grid_xy, gt_params, n_neighbors=n_neighbors)), 4),
    }


def label_metrics(labels: np.ndarray, bmu_keys: pd.Series) -> dict:
    """Cluster separation vs ground-truth labels (cluster = neuron)."""
    from sklearn.metrics import adjusted_rand_score
    df = pd.DataFrame({'label': labels, 'neuron': bmu_keys.values})
    purity_per_neuron = df.groupby('neuron')['label'].agg(
        lambda s: s.value_counts().iloc[0] / len(s))
    return {
        'adjusted_rand_index': round(float(
            adjusted_rand_score(df['label'], df['neuron'])), 4),
        'mean_neuron_purity': round(float(purity_per_neuron.mean()), 4),
    }


def hit_distribution(assignments: pd.DataFrame, run_metrics: dict) -> dict:
    """Map utilization — negative-control metrics (no structure invented)."""
    map_size = run_metrics.get('map_size') or []
    total = map_size[0] * map_size[1] if len(map_size) == 2 else None
    counts = assignments['bmu_key'].value_counts().values.astype(float)
    if total:
        counts = np.concatenate([counts, np.zeros(total - len(counts))])
    sorted_c = np.sort(counts)
    n = len(sorted_c)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_c) / (n * sorted_c.sum())
            - (n + 1) / n) if sorted_c.sum() > 0 else 0.0
    return {
        'active_neurons': int((counts > 0).sum()),
        'total_neurons': total,
        'dead_ratio': round(float((counts == 0).sum() / n), 4) if total else None,
        'hit_gini': round(float(gini), 4),
    }


def verdict(value: float | None, pass_t: float, warn_t: float) -> str:
    if value is None:
        return 'N/A'
    if value >= pass_t:
        return 'PASS'
    if value >= warn_t:
        return 'WARN'
    return 'FAIL'


# ─── Main ─────────────────────────────────────────────────────────────────────

def verify(results_dir: str, groundtruth_path: str | None = None,
           n_pairs: int = 2000, n_neighbors: int = 10) -> dict:
    assignments, run_metrics = load_run(results_dir)
    map_type = run_metrics.get('map_topology', 'hex')
    grid_xy = grid_coords(assignments, map_type)

    report: dict = {
        'results_dir': os.path.abspath(results_dir),
        'run_metrics': {k: run_metrics.get(k) for k in
                        ('map_size', 'map_topology', 'best_mqe',
                         'topographic_error', 'duration')},
        'hit_distribution': hit_distribution(assignments, run_metrics),
        'note': ('Low MQE/TE alone do NOT prove manifold adherence '
                 '(docs/som/issues.md #23) — grid_param_R2 does.'),
    }

    if groundtruth_path is None:
        groundtruth_path = find_groundtruth(results_dir)

    if groundtruth_path is None:
        report['groundtruth'] = None
        report['verdict'] = 'NO-GROUNDTRUTH (hit distribution only — '
        report['verdict'] += 'negative-control mode)'
        return report

    gt = pd.read_csv(groundtruth_path)
    merged = assignments.merge(gt, left_on='sample_id', right_on='id',
                               how='inner')
    if merged.empty:
        sys.exit('ERROR: ground truth ids do not match sample_assignments ids.')
    grid_xy = grid_coords(merged, map_type)

    param_cols, label_cols = split_gt_columns(gt)
    report['groundtruth'] = os.path.abspath(groundtruth_path)
    report['n_samples_matched'] = int(len(merged))

    verdicts = []
    if param_cols:
        params_block = {}
        for col in param_cols:
            param = merged[col].values.astype(float)
            metrics = param_adherence(param, grid_xy)
            metrics['neuron_anova_R2'] = neuron_anova_r2(param, merged['bmu_key'])
            metrics['verdict'] = verdict(metrics['grid_param_R2'], R2_PASS, R2_WARN)
            verdicts.append(metrics['verdict'])
            params_block[col] = metrics
        report['manifold_params'] = params_block

        gt_matrix = merged[param_cols].values.astype(float)
        global_rho = pairwise_distance_spearman(gt_matrix, grid_xy, n_pairs)
        global_verdict = verdict(global_rho, GLOBAL_PASS, GLOBAL_WARN)
        verdicts.append(global_verdict)
        report['global_structure'] = {
            'pairwise_distance_spearman': global_rho,
            'verdict': global_verdict,
        }
        report['neighborhood'] = trustworthiness_continuity(
            _standardize(gt_matrix), grid_xy, n_neighbors)

    if label_cols:
        labels_block = {}
        for col in label_cols:
            metrics = label_metrics(merged[col].values, merged['bmu_key'])
            metrics['verdict'] = verdict(metrics['adjusted_rand_index'],
                                         ARI_PASS, ARI_WARN)
            verdicts.append(metrics['verdict'])
            labels_block[col] = metrics
        report['labels'] = labels_block

    if not verdicts:
        report['verdict'] = 'NO-METRICS (ground truth has no usable columns)'
    elif 'FAIL' in verdicts:
        report['verdict'] = 'FAIL'
    elif 'WARN' in verdicts:
        report['verdict'] = 'WARN'
    else:
        report['verdict'] = 'PASS'
    return report


def print_report(report: dict):
    print(f"\n=== Topology verification — {report['results_dir']} ===")
    rm = report['run_metrics']
    print(f"map {rm.get('map_size')} {rm.get('map_topology')}  "
          f"MQE={rm.get('best_mqe'):.4f}  TE={rm.get('topographic_error'):.4f}"
          if rm.get('best_mqe') is not None else "run metrics missing")
    hd = report['hit_distribution']
    print(f"hits: {hd['active_neurons']}/{hd['total_neurons']} neurons active, "
          f"dead_ratio={hd['dead_ratio']}, gini={hd['hit_gini']}")
    if report.get('groundtruth') is None:
        print('No ground truth found — negative-control mode only.')
    for col, m in report.get('manifold_params', {}).items():
        print(f"  param '{col}':  R2={m['grid_param_R2']} (kNN)  "
              f"linear={m['linear_R2']}  anova={m['neuron_anova_R2']}  "
              f"spearman={m['best_axis_spearman']}  → {m['verdict']}")
    if 'global_structure' in report:
        gs = report['global_structure']
        print(f"  global structure (pairwise distances): "
              f"spearman={gs['pairwise_distance_spearman']}  → {gs['verdict']}")
    if 'neighborhood' in report:
        nb = report['neighborhood']
        print(f"  trustworthiness={nb['trustworthiness']}  "
              f"continuity={nb['continuity']}")
    for col, m in report.get('labels', {}).items():
        print(f"  labels '{col}':  ARI={m['adjusted_rand_index']}  "
              f"purity={m['mean_neuron_purity']}  → {m['verdict']}")
    print(f"VERDICT: {report['verdict']}")


def main():
    parser = argparse.ArgumentParser(
        description='Verify SOM topology preservation against ground truth.')
    parser.add_argument('results_dir')
    parser.add_argument('-g', '--groundtruth', default=None,
                        help='Path to *_groundtruth.csv (default: auto-discover '
                             'next to the dataset)')
    parser.add_argument('--pairs', type=int, default=2000,
                        help='Sampled pairs for distance correlation (default 2000)')
    parser.add_argument('--neighbors', type=int, default=10,
                        help='k for trustworthiness/continuity (default 10)')
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        sys.exit(f'ERROR: directory not found: {args.results_dir}')

    report = verify(args.results_dir, args.groundtruth, args.pairs, args.neighbors)
    print_report(report)

    out_path = os.path.join(args.results_dir, 'json', 'verify_topology.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")
    return 0 if report['verdict'] in ('PASS', 'WARN') else 1


if __name__ == '__main__':
    sys.exit(main())
