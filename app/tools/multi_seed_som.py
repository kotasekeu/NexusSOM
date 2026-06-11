"""
multi_seed_som.py — Statistical robustness tool: run the same SOM configuration
with N different random seeds and compare everything that can vary between runs.

Answers reviewer concern R2.4 (mean ± std instead of a single run) and feeds
the ablation study (docs/global/ABLATION_STUDY.md, protocol section).

Compared per seed:
  - final metrics: best_mqe, topographic_error, duration (run_metrics.json)
  - map quality:   dead_ratio, silhouette, spatial_quality_score, T&C
                   (json/llm_context.json)
  - cluster count: active neurons (json/clusters.json)
  - MQE evolution: checkpoint curves overlaid in one plot
  - clustering stability: pairwise Adjusted Rand Index of sample→neuron
    assignments across seeds (csv/sample_assignments.csv)

Outputs into <output_dir>/:
  seed_<k>/                    full results dir per seed
  multi_seed_metrics.csv       one row per seed
  multi_seed_summary.json      {metric: {mean, std, values}} + ARI stats
  mqe_evolution_comparison.png overlaid MQE curves (when checkpoints enabled)

Usage:
  python app/tools/multi_seed_som.py -i data.csv -c config.json -n 10
  python app/tools/multi_seed_som.py -i data.csv -c config.json --seeds 1 7 42
  python app/tools/multi_seed_som.py -i data.csv -c config.json -n 5 --with-maps
"""

import argparse
import itertools
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Allow `som.*` imports when running from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from som.run import run_pipeline  # noqa: E402
from som.utils import load_configuration  # noqa: E402


COMPARED_METRICS = ('best_mqe', 'topographic_error', 'duration', 'dead_ratio',
                    'silhouette', 'spatial_quality_score', 'active_neurons')


def _load_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def collect_seed_metrics(results_dir: str, seed: int) -> dict:
    """Flatten the per-run artifacts into one comparable metrics row."""
    run_metrics = _load_json(os.path.join(results_dir, 'run_metrics.json'))
    llm_context = _load_json(os.path.join(results_dir, 'json', 'llm_context.json'))
    clusters = _load_json(os.path.join(results_dir, 'json', 'clusters.json'))
    map_section = llm_context.get('map', {})

    row = {
        'seed': seed,
        'results_dir': results_dir,
        'best_mqe': run_metrics.get('best_mqe'),
        'topographic_error': run_metrics.get('topographic_error'),
        'duration': run_metrics.get('duration'),
        'dead_ratio': map_section.get('dead_ratio'),
        'silhouette': map_section.get('silhouette'),
        'spatial_quality_score': map_section.get('spatial_quality_score'),
        'active_neurons': len(clusters),
    }
    tc = map_section.get('trustworthiness_continuity') or {}
    for key, value in tc.items():
        row[f'tc_{key}'] = value
    return row


def load_assignments(results_dir: str) -> pd.Series | None:
    """sample_id → bmu_key labels for clustering-stability comparison."""
    path = os.path.join(results_dir, 'csv', 'sample_assignments.csv')
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    return df.sort_values('sample_id').set_index('sample_id')['bmu_key']


def load_mqe_curve(results_dir: str) -> tuple | None:
    """(progress, mqe) arrays from training checkpoints, if saved."""
    cps = _load_json(os.path.join(results_dir, 'csv', 'training_checkpoints.json'))
    if not cps:
        return None
    return ([cp['progress'] for cp in cps], [cp['mqe'] for cp in cps])


def pairwise_ari(assignments: dict) -> dict:
    """Adjusted Rand Index for every seed pair — clustering stability."""
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return {}

    scores = {}
    for (seed_a, labels_a), (seed_b, labels_b) in itertools.combinations(
            assignments.items(), 2):
        common = labels_a.index.intersection(labels_b.index)
        if len(common) == 0:
            continue
        score = adjusted_rand_score(labels_a.loc[common], labels_b.loc[common])
        scores[f'{seed_a}_vs_{seed_b}'] = round(float(score), 4)
    return scores


def summarize(values_per_metric: dict, ari_scores: dict) -> dict:
    summary = {}
    for metric, values in values_per_metric.items():
        clean = [v for v in values if v is not None]
        if not clean:
            continue
        summary[metric] = {
            'mean': round(float(np.mean(clean)), 6),
            'std': round(float(np.std(clean)), 6),
            'min': round(float(np.min(clean)), 6),
            'max': round(float(np.max(clean)), 6),
            'values': clean,
        }
    if ari_scores:
        ari_values = list(ari_scores.values())
        summary['clustering_stability_ari'] = {
            'mean': round(float(np.mean(ari_values)), 4),
            'std': round(float(np.std(ari_values)), 4),
            'min': round(float(np.min(ari_values)), 4),
            'max': round(float(np.max(ari_values)), 4),
            'pairs': ari_scores,
        }
    return summary


def plot_mqe_curves(curves: dict, output_path: str):
    if not curves:
        return
    plt.figure(figsize=(10, 6))
    for seed, (progress, mqe) in sorted(curves.items()):
        plt.plot(progress, mqe, label=f'seed {seed}', alpha=0.8)
    plt.title('MQE Evolution Across Seeds')
    plt.xlabel('Training progress')
    plt.ylabel('MQE')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Run one SOM configuration with N seeds and compare results')
    parser.add_argument('-i', '--input', required=True, help='Input CSV file')
    parser.add_argument('-c', '--config', required=True, help='SOM config JSON')
    parser.add_argument('-n', '--n-seeds', type=int, default=10,
                        help='Number of seeds (1..N); ignored when --seeds is given')
    parser.add_argument('--seeds', type=int, nargs='+',
                        help='Explicit list of seeds to run')
    parser.add_argument('-o', '--output',
                        help='Output base directory (default: results/multi_seed_<timestamp> next to input)')
    parser.add_argument('--with-maps', action='store_true',
                        help='Also generate per-run visualizations (slower; metrics do not need them)')
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else list(range(1, args.n_seeds + 1))

    config = load_configuration(args.config)
    # Checkpoints are needed for the MQE-evolution comparison
    config.setdefault('save_checkpoints', True)
    config.setdefault('checkpoint_count', 20)
    config['show_progress'] = False
    if not args.with_maps:
        config['save_training_plots'] = False
        config['save_visualizations'] = False

    if args.output:
        base_dir = args.output
    else:
        from datetime import datetime
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)),
                                'results', f'multi_seed_{stamp}')
    os.makedirs(base_dir, exist_ok=True)

    rows = []
    assignments = {}
    curves = {}
    for seed in seeds:
        print(f'INFO: Running seed {seed} ({seeds.index(seed) + 1}/{len(seeds)})...')
        results_dir = run_pipeline(args.input, config,
                                   output_dir=os.path.join(base_dir, f'seed_{seed}'),
                                   seed=seed)
        rows.append(collect_seed_metrics(results_dir, seed))
        labels = load_assignments(results_dir)
        if labels is not None:
            assignments[seed] = labels
        curve = load_mqe_curve(results_dir)
        if curve is not None:
            curves[seed] = curve

    metrics_df = pd.DataFrame(rows)
    metrics_csv = os.path.join(base_dir, 'multi_seed_metrics.csv')
    metrics_df.to_csv(metrics_csv, index=False)

    values_per_metric = {m: metrics_df[m].tolist() for m in COMPARED_METRICS
                         if m in metrics_df.columns}
    ari_scores = pairwise_ari(assignments)
    summary = summarize(values_per_metric, ari_scores)
    summary_path = os.path.join(base_dir, 'multi_seed_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({'seeds': seeds, 'input': os.path.abspath(args.input),
                   'config': os.path.abspath(args.config), 'metrics': summary},
                  f, indent=2)

    plot_mqe_curves(curves, os.path.join(base_dir, 'mqe_evolution_comparison.png'))

    print(f'\n=== Multi-seed summary ({len(seeds)} runs) ===')
    for metric, stats in summary.items():
        if metric == 'clustering_stability_ari':
            print(f"  clustering stability (ARI): "
                  f"{stats['mean']:.4f} ± {stats['std']:.4f}")
        else:
            print(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f}")
    print(f'\nINFO: Metrics:  {metrics_csv}')
    print(f'INFO: Summary:  {summary_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
