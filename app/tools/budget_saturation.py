"""
budget_saturation.py — experiment D driver + report (docs/ea/SEARCH_SPACE.md
step 2): quality-vs-training-budget saturation curves per processing regime.

Measures where map quality saturates as a function of the per-sample
exposure budget P (how many times each sample updates the map), separately
for the three regimes — they couple schedule steps and exposures
differently (`total_iterations = N * epoch_multiplier`, each iteration
processes `ceil(N * pct / 100)` samples):

  deterministic  pct 100 static   exposures = N * em   (1 iteration = 1 epoch)
  stochastic     pct 0.01 static  exposures = em       (1 sample / iteration)
  hybrid         pct 1->5 growth  exposures = em * N * mean_pct / 100

Arms are generated from the experiment-C base configs with only the batch
profile and `epoch_multiplier` overridden; em is derived per dataset from
the measured CSV size so that the nominal budget lands exactly (the +0.5
nudge below absorbs the int() truncation in `total_iterations`). All
analysis uses the *measured* exposures (mean of `sample_coverage.json`
counts), never the nominal budget.

Usage:
  # full grid, resumable (arms with a finished multi_seed_summary.json are
  # skipped; ground-truth datasets also get verify_topology.py per seed):
  .venv/bin/python3 app/tools/budget_saturation.py run [--datasets Iris,Helix]
      [--seeds 10] [--regimes det,stoch,hyb] [--dry-run]

  # aggregate whatever has finished into data/expD/ (report + figures):
  .venv/bin/python3 app/tools/budget_saturation.py report [--datasets ...]

Pre-registered decision criteria live in docs/ea/SEARCH_SPACE.md ("Step 2
protocol"); this tool only measures and reports.
"""

import argparse
import ast
import json
import math
import os
import subprocess
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASETS = {
    'Iris': {
        'csv': 'data/datasets/Iris/iris.csv',
        'base_config': 'data/datasets/Iris/config-som-expC-reshuffle-em10.json',
        'groundtruth': None,
    },
    'SwissRoll': {
        'csv': 'data/datasets/SwissRoll/swiss_roll.csv',
        'base_config': 'data/datasets/SwissRoll/config-som-expC-reshuffle-em10.json',
        'groundtruth': 'data/datasets/SwissRoll/swiss_roll_groundtruth.csv',
    },
    'Helix': {
        'csv': 'data/datasets/Helix/helix.csv',
        'base_config': 'data/datasets/Helix/config-som-expC-reshuffle-em10.json',
        'groundtruth': 'data/datasets/Helix/helix_groundtruth.csv',
    },
    'WineQuality': {
        'csv': 'data/datasets/WineQuality/wine.csv',
        'base_config': 'data/datasets/WineQuality/config-som-expC-reshuffle-em10.json',
        'groundtruth': None,
    },
}

DET_EPOCHS = (3, 10, 30, 100, 300, 1000)
STOCH_EMS = (1, 3, 10, 30, 100, 300, 1000)
HYB_BUDGETS = (3, 10, 30, 100, 300, 1000)
HYB_MEAN_PCT = 3.0  # mean of the 1->5 linear-growth profile

REPORT_DIR = 'data/expD'

# Metrics shown in the per-dataset curve figures (column, label, lower_is_better)
CURVE_METRICS = (
    ('best_mqe', 'MQE', True),
    ('topographic_error', 'Topographic error', True),
    ('dead_ratio', 'Dead ratio', True),
    ('trustworthiness', 'Trustworthiness (k=10)', False),
    ('continuity', 'Continuity (k=10)', False),
    ('gt_pairwise_spearman', 'Ground-truth pairwise ρ', False),
)
# Topology metrics consulted by the saturation / overtraining criteria
TOPO_METRICS = (
    ('topographic_error', True),
    ('trustworthiness', False),
    ('continuity', False),
    ('gt_pairwise_spearman', False),
)


def build_arms(n_samples: int) -> list:
    """The experiment grid for one dataset of size n_samples."""
    arms = []
    for e in DET_EPOCHS:
        arms.append({
            'name': f'det-e{e}', 'regime': 'det', 'budget': e,
            'overrides': {
                # +0.5: int(N * em) must land exactly on e iterations
                'epoch_multiplier': (e + 0.5) / n_samples,
                'start_batch_percent': 100.0, 'end_batch_percent': 100.0,
                'batch_growth_type': 'static',
            },
        })
    for em in STOCH_EMS:
        arms.append({
            'name': f'stoch-em{em}', 'regime': 'stoch', 'budget': em,
            'overrides': {
                'epoch_multiplier': float(em),
                'start_batch_percent': 0.01, 'end_batch_percent': 0.01,
                'batch_growth_type': 'static',
            },
        })
    for b in HYB_BUDGETS:
        iters = round(b * 100.0 / HYB_MEAN_PCT)
        arms.append({
            'name': f'hyb-b{b}', 'regime': 'hyb', 'budget': b,
            'overrides': {
                'epoch_multiplier': (iters + 0.5) / n_samples,
                'start_batch_percent': 1.0, 'end_batch_percent': 5.0,
                'batch_growth_type': 'linear-growth',
            },
        })
    return arms


def count_samples(csv_path: str) -> int:
    with open(csv_path, encoding='utf-8') as f:
        return sum(1 for line in f if line.strip()) - 1  # minus header


def build_config(base_config_path: str, arm: dict) -> dict:
    with open(base_config_path, encoding='utf-8') as f:
        config = json.load(f)
    config.pop('_comment', None)
    config.update(arm['overrides'])
    config.update({
        '_comment': (f"Experiment D (docs/ea/SEARCH_SPACE.md step 2) — arm "
                     f"{arm['name']}: regime {arm['regime']}, nominal budget "
                     f"{arm['budget']} exposures/sample. Generated by "
                     f"app/tools/budget_saturation.py from "
                     f"{os.path.basename(base_config_path)}."),
        'sampling_method': 'reshuffle',
        'num_batches': 1,
        'track_sample_coverage': True,
        'mqe_evaluations_per_run': 100,
        'save_checkpoints': True,
        'checkpoint_count': 20,
        'show_progress': False,
    })
    return config


def arm_dir(ds_name: str, arm_name: str) -> str:
    return os.path.join(REPO_ROOT, 'data', 'datasets', ds_name, 'results',
                        f'expD_{arm_name}')


def run_verify_topology(out_dir: str, groundtruth: str) -> None:
    """verify_topology.py on every seed dir that does not have results yet."""
    for entry in sorted(os.listdir(out_dir)):
        seed_dir = os.path.join(out_dir, entry)
        if not (entry.startswith('seed_') and os.path.isdir(seed_dir)):
            continue
        marker = os.path.join(seed_dir, 'json', 'verify_topology.json')
        if os.path.isfile(marker):
            continue
        result = subprocess.run(
            [sys.executable, os.path.join(REPO_ROOT, 'app/tools/verify_topology.py'),
             seed_dir, '-g', os.path.join(REPO_ROOT, groundtruth)],
            capture_output=True, text=True)
        # exit code 1 only signals a FAIL verdict (expected on e.g. Helix);
        # real failure = the report file was not written
        if not os.path.isfile(marker):
            print(f'WARN: verify_topology failed for {seed_dir}:\n{result.stderr[-500:]}')


def cmd_run(args) -> int:
    datasets = parse_datasets(args.datasets)
    regimes = set(args.regimes.split(',')) if args.regimes else {'det', 'stoch', 'hyb'}
    failures = []
    for ds_name in datasets:
        spec = DATASETS[ds_name]
        csv_path = os.path.join(REPO_ROOT, spec['csv'])
        n_samples = count_samples(csv_path)
        arms = [a for a in build_arms(n_samples) if a['regime'] in regimes]
        print(f'=== {ds_name} (N={n_samples}, {len(arms)} arms) ===')
        for arm in arms:
            out_dir = arm_dir(ds_name, arm['name'])
            done = os.path.isfile(os.path.join(out_dir, 'multi_seed_summary.json'))
            if args.dry_run:
                iters = int(n_samples * arm['overrides']['epoch_multiplier'])
                status = 'SKIP (done)' if done else 'run'
                print(f"  {arm['name']:<12} em={arm['overrides']['epoch_multiplier']:.6f} "
                      f"iters={iters} -> {status}")
                continue
            if done:
                print(f"  {arm['name']:<12} already finished — skipping")
            else:
                os.makedirs(out_dir, exist_ok=True)
                config_path = os.path.join(out_dir, 'config.json')
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(build_config(os.path.join(REPO_ROOT, spec['base_config']),
                                           arm), f, indent=4)
                start = time.time()
                result = subprocess.run(
                    [sys.executable,
                     os.path.join(REPO_ROOT, 'app/tools/multi_seed_som.py'),
                     '-i', csv_path, '-c', config_path,
                     '-n', str(args.seeds), '-o', out_dir],
                    capture_output=True, text=True)
                if result.returncode != 0:
                    failures.append(f'{ds_name}/{arm["name"]}')
                    print(f"  {arm['name']:<12} FAILED:\n{result.stderr[-800:]}")
                    continue
                print(f"  {arm['name']:<12} done in {time.time() - start:.0f}s")
            if spec['groundtruth']:
                run_verify_topology(out_dir, spec['groundtruth'])
    if args.dry_run:
        return 0
    if failures:
        print(f'\nERROR: {len(failures)} arm(s) failed: {", ".join(failures)}')
        return 1
    print('\nAll requested arms finished.')
    return 0


# ---------------------------------------------------------------- report ---

def parse_datasets(arg: str | None) -> list:
    if not arg:
        return list(DATASETS)
    names = [n.strip() for n in arg.split(',') if n.strip()]
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        raise SystemExit(f'Unknown dataset(s): {unknown} (known: {list(DATASETS)})')
    return names


def parse_arm_name(dirname: str):
    """'expD_stoch-em30' -> ('stoch', 30); None for foreign dirs."""
    if not dirname.startswith('expD_'):
        return None
    name = dirname[len('expD_'):]
    try:
        regime, tag = name.split('-', 1)
        budget = int(''.join(ch for ch in tag if ch.isdigit()))
    except ValueError:
        return None
    if regime not in ('det', 'stoch', 'hyb'):
        return None
    return regime, budget


def collect_rows(ds_name: str) -> list:
    """One row per seed run: metrics + measured exposures (+ ground truth)."""
    results_root = os.path.join(REPO_ROOT, 'data', 'datasets', ds_name, 'results')
    rows = []
    if not os.path.isdir(results_root):
        return rows
    for entry in sorted(os.listdir(results_root)):
        parsed = parse_arm_name(entry)
        if parsed is None:
            continue
        regime, budget = parsed
        base = os.path.join(results_root, entry)
        metrics_csv = os.path.join(base, 'multi_seed_metrics.csv')
        if not os.path.isfile(metrics_csv):
            continue
        ari_mean = None
        summary_path = os.path.join(base, 'multi_seed_summary.json')
        if os.path.isfile(summary_path):
            with open(summary_path, encoding='utf-8') as f:
                summary = json.load(f)
            ari_mean = (summary.get('metrics', {})
                        .get('clustering_stability_ari', {}).get('mean'))
        df = pd.read_csv(metrics_csv)
        for _, mrow in df.iterrows():
            seed_dir = os.path.join(base, f"seed_{int(mrow['seed'])}")
            row = {
                'dataset': ds_name, 'regime': regime, 'budget': budget,
                'seed': int(mrow['seed']),
                'best_mqe': mrow.get('best_mqe'),
                'topographic_error': mrow.get('topographic_error'),
                'duration': mrow.get('duration'),
                'dead_ratio': mrow.get('dead_ratio'),
                'silhouette': mrow.get('silhouette'),
                'active_neurons': mrow.get('active_neurons'),
                'ari_mean': ari_mean,
                'exposures': None,
                'trustworthiness': None, 'continuity': None,
                'gt_pairwise_spearman': None, 'gt_grid_R2_mean': None,
            }
            cov_path = os.path.join(seed_dir, 'csv', 'sample_coverage.json')
            if os.path.isfile(cov_path):
                with open(cov_path, encoding='utf-8') as f:
                    row['exposures'] = json.load(f).get('mean')
            tc_raw = mrow.get('tc_10')
            if isinstance(tc_raw, str):
                try:
                    tc = ast.literal_eval(tc_raw)
                    row['trustworthiness'] = tc.get('trustworthiness')
                    row['continuity'] = tc.get('continuity')
                except (ValueError, SyntaxError):
                    pass
            vt_path = os.path.join(seed_dir, 'json', 'verify_topology.json')
            if os.path.isfile(vt_path):
                with open(vt_path, encoding='utf-8') as f:
                    vt = json.load(f)
                row['gt_pairwise_spearman'] = (vt.get('global_structure', {})
                                               .get('pairwise_distance_spearman'))
                params = vt.get('manifold_params', {})
                r2 = [p.get('grid_param_R2') for p in params.values()
                      if p.get('grid_param_R2') is not None]
                if r2:
                    row['gt_grid_R2_mean'] = float(np.mean(r2))
            rows.append(row)
    return rows


def mannwhitney_p(a, b):
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        return None
    a = [v for v in a if v is not None and not pd.isna(v)]
    b = [v for v in b if v is not None and not pd.isna(v)]
    if len(a) < 3 or len(b) < 3 or (np.std(a) == 0 and np.std(b) == 0):
        return None
    return float(mannwhitneyu(a, b, alternative='two-sided').pvalue)


def significantly_worse(vals, ref_vals, lower_is_better: bool) -> bool:
    """vals significantly worse than ref_vals (MW p<0.05 + worse median)."""
    p = mannwhitney_p(vals, ref_vals)
    if p is None or p >= 0.05:
        return False
    med, ref = np.nanmedian(vals), np.nanmedian(ref_vals)
    return med > ref if lower_is_better else med < ref


def evaluate_regime(df: pd.DataFrame) -> dict:
    """Saturation point + overtraining flags for one dataset x regime slice.

    Pre-registered criteria (SEARCH_SPACE.md step 2, marginal-gain form):
    b* = smallest budget from which every further grid step improves the
    median MQE by < 2 % per doubling of budget, with no topology metric
    significantly worse than at the best-MQE budget. MQE-vs-budget decays
    power-law-like without a hard plateau, so an absolute closeness
    criterion would always pull b* to the grid edge; the marginal form is
    honest on a finite grid — b* = None means "saturation not reached at
    the largest tested budget". Overtraining = budget > b* with MQE not
    worse but a topology metric significantly worse than at b*.
    """
    budgets = sorted(df['budget'].unique())
    by_budget = {b: df[df['budget'] == b] for b in budgets}
    mqe_medians = {b: np.nanmedian(by_budget[b]['best_mqe']) for b in budgets}
    best_budget = min(budgets, key=lambda b: mqe_medians[b])

    def gain_per_doubling(a, c):
        gain = (mqe_medians[a] - mqe_medians[c]) / mqe_medians[a]
        return max(gain, 0.0) / math.log2(c / a)

    b_star = None
    for i, b in enumerate(budgets[:-1]):
        if any(gain_per_doubling(budgets[j], budgets[j + 1]) >= 0.02
               for j in range(i, len(budgets) - 1)):
            continue
        topo_worse = [
            col for col, lower in TOPO_METRICS
            if by_budget[b][col].notna().sum() >= 3
            and significantly_worse(by_budget[b][col].tolist(),
                                    by_budget[best_budget][col].tolist(), lower)
        ]
        if not topo_worse:
            b_star = b
            break
    overtrained = []
    if b_star is not None:
        for b in [b for b in budgets if b > b_star]:
            if significantly_worse(by_budget[b]['best_mqe'].tolist(),
                                   by_budget[b_star]['best_mqe'].tolist(), True):
                continue  # MQE itself degraded -> not the overtraining pattern
            worse = [col for col, lower in TOPO_METRICS
                     if by_budget[b][col].notna().sum() >= 3
                     and significantly_worse(by_budget[b][col].tolist(),
                                             by_budget[b_star][col].tolist(), lower)]
            if worse:
                overtrained.append((b, worse))
    return {'budgets': budgets, 'mqe_medians': mqe_medians,
            'best_budget': best_budget, 'b_star': b_star,
            'overtrained': overtrained}


REGIME_STYLE = {'det': ('tab:blue', 'deterministic (budget = epochs)'),
                'stoch': ('tab:red', 'stochastic (budget = em)'),
                'hyb': ('tab:green', 'hybrid 1→5 % (budget = exposures)')}


def plot_curves(df: pd.DataFrame, ds_name: str, out_path: str) -> None:
    metrics = [(c, lbl) for c, lbl, _ in CURVE_METRICS if df[c].notna().any()]
    if df['ari_mean'].notna().any():
        metrics.append(('ari_mean', 'Clustering stability ARI (arm mean)'))
    ncols = 3
    nrows = math.ceil(len(metrics) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4 * nrows),
                             squeeze=False)
    for ax, (col, label) in zip(axes.flat, metrics):
        for regime, (color, rlabel) in REGIME_STYLE.items():
            sub = df[(df['regime'] == regime) & df[col].notna()
                     & df['exposures'].notna()]
            if sub.empty:
                continue
            stats = (sub.groupby('budget')
                     .agg(x=('exposures', 'median'), med=(col, 'median'),
                          p25=(col, lambda s: s.quantile(0.25)),
                          p75=(col, lambda s: s.quantile(0.75)))
                     .sort_values('x'))
            ax.plot(stats['x'], stats['med'], 'o-', color=color, label=rlabel)
            ax.fill_between(stats['x'], stats['p25'], stats['p75'],
                            color=color, alpha=0.15)
            ax.scatter(sub['exposures'], sub[col], color=color, s=8, alpha=0.25)
        ax.set_xscale('log')
        ax.set_xlabel('measured exposures per sample')
        ax.set_title(label)
        ax.grid(True, linestyle='--', alpha=0.5)
    for ax in axes.flat[len(metrics):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f'Experiment D — {ds_name}: quality vs training budget', y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_overtraining(df: pd.DataFrame, ds_name: str, out_path: str) -> None:
    """MQE (left axis) vs topology metrics (right axis) per regime — the
    under/overtraining exhibit: divergence right of the MQE plateau."""
    regimes = [r for r in REGIME_STYLE if not df[df['regime'] == r].empty]
    fig, axes = plt.subplots(1, len(regimes), figsize=(5.5 * len(regimes), 4.2),
                             squeeze=False)
    for ax, regime in zip(axes.flat, regimes):
        sub = df[(df['regime'] == regime) & df['exposures'].notna()]
        stats = (sub.groupby('budget')
                 .agg(x=('exposures', 'median'), mqe=('best_mqe', 'median'),
                      te=('topographic_error', 'median'),
                      rho=('gt_pairwise_spearman', 'median'))
                 .sort_values('x'))
        ax.plot(stats['x'], stats['mqe'], 'o-', color='black', label='MQE (left)')
        ax.set_ylabel('MQE')
        ax.set_xscale('log')
        ax.set_xlabel('measured exposures per sample')
        ax2 = ax.twinx()
        ax2.plot(stats['x'], stats['te'], 's--', color='tab:orange',
                 label='TE (right)')
        if stats['rho'].notna().any():
            ax2.plot(stats['x'], stats['rho'], '^--', color='tab:purple',
                     label='ground-truth ρ (right)')
        ax2.set_ylabel('TE / ρ')
        lines = ax.get_lines() + ax2.get_lines()
        ax.legend(lines, [ln.get_label() for ln in lines], fontsize=8)
        ax.set_title(REGIME_STYLE[regime][1])
        ax.grid(True, linestyle='--', alpha=0.5)
    fig.suptitle(f'Experiment D — {ds_name}: under/overtraining exhibit '
                 '(quantization vs topology)', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def fmt(v, digits=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '—'
    return f'{v:.{digits}f}'


def write_report(df: pd.DataFrame, out_dir: str) -> str:
    lines = [
        '# Experiment D — training-budget saturation (step 2)', '',
        f'Generated by `app/tools/budget_saturation.py report` from '
        f'{df["dataset"].nunique()} dataset(s), '
        f'{len(df.groupby(["dataset", "regime", "budget"]))} arms, '
        f'{len(df)} runs. Protocol and pre-registered criteria: '
        '`docs/ea/SEARCH_SPACE.md` "Step 2 protocol".', '',
        'X-axis everywhere: **measured** exposures per sample '
        '(`sample_coverage.json` mean), not the nominal budget.', '',
    ]
    verdict_rows = []
    for ds_name in df['dataset'].unique():
        ds = df[df['dataset'] == ds_name]
        lines += [f'## {ds_name}', '',
                  f'![curves](expD_{ds_name}_curves.png)', '',
                  f'![overtraining](expD_{ds_name}_overtraining.png)', '',
                  '| Regime | Budget | Exposures (meas.) | MQE med (IQR) | TE med '
                  '| Dead | Trust | Cont | GT ρ | ARI | Duration s |',
                  '|---|---|---|---|---|---|---|---|---|---|---|']
        for (regime, budget), arm in ds.groupby(['regime', 'budget']):
            mqe = arm['best_mqe']
            lines.append(
                f"| {regime} | {budget} | {fmt(arm['exposures'].median(), 1)} "
                f"| {fmt(mqe.median())} ({fmt(mqe.quantile(0.25))}–{fmt(mqe.quantile(0.75))}) "
                f"| {fmt(arm['topographic_error'].median())} "
                f"| {fmt(arm['dead_ratio'].median(), 3)} "
                f"| {fmt(arm['trustworthiness'].median(), 3)} "
                f"| {fmt(arm['continuity'].median(), 3)} "
                f"| {fmt(arm['gt_pairwise_spearman'].median(), 3)} "
                f"| {fmt(arm['ari_mean'].median(), 3)} "
                f"| {fmt(arm['duration'].median(), 1)} |")
        lines.append('')
        for regime in ('det', 'stoch', 'hyb'):
            sub = ds[ds['regime'] == regime]
            if sub.empty:
                continue
            res = evaluate_regime(sub)
            over = ('; '.join(f'budget {b}: {", ".join(cols)} worse'
                              for b, cols in res['overtrained'])
                    or 'not observed')
            verdict_rows.append(
                f"| {ds_name} | {regime} | {res['b_star'] if res['b_star'] is not None else '—'} "
                f"| {res['best_budget']} | {fmt(res['mqe_medians'][res['best_budget']])} "
                f"| {over} |")
    lines += [
        '## Saturation verdicts (pre-registered criteria)', '',
        'b\\* = smallest budget from which every further grid step improves '
        'median MQE by < 2 % per doubling of budget, with no topology metric '
        'significantly worse (Mann-Whitney p < 0.05) than at the best-MQE '
        'budget. "—" = saturation not reached at the largest tested budget. '
        'Overtraining = larger budget where MQE holds but a topology metric '
        'is significantly worse than at b\\*.', '',
        '| Dataset | Regime | b\\* | Best-MQE budget | Best MQE med | Overtraining |',
        '|---|---|---|---|---|---|',
        *verdict_rows, '',
        'Interpretation and the fixed-value decision belong in '
        '`docs/ea/SEARCH_SPACE.md` (step 2 results), not here.', '',
    ]
    path = os.path.join(out_dir, 'expD_report.md')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path


def cmd_report(args) -> int:
    import warnings
    # all-NaN slices (e.g. ground-truth columns on non-benchmark datasets)
    # are expected; medians over them correctly yield NaN -> rendered as '—'
    warnings.filterwarnings('ignore', message='Mean of empty slice')
    warnings.filterwarnings('ignore', message='All-NaN slice encountered')
    datasets = parse_datasets(args.datasets)
    rows = []
    for ds_name in datasets:
        rows.extend(collect_rows(ds_name))
    if not rows:
        print('ERROR: no finished expD arms found.')
        return 1
    df = pd.DataFrame(rows)
    out_dir = os.path.join(REPO_ROOT, REPORT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, 'expD_metrics.csv'), index=False)
    for ds_name in df['dataset'].unique():
        ds = df[df['dataset'] == ds_name]
        plot_curves(ds, ds_name, os.path.join(out_dir, f'expD_{ds_name}_curves.png'))
        plot_overtraining(ds, ds_name,
                          os.path.join(out_dir, f'expD_{ds_name}_overtraining.png'))
    path = write_report(df, out_dir)
    print(f'INFO: {len(df)} runs aggregated.')
    print(f'INFO: report:  {path}')
    print(f'INFO: figures: {out_dir}/expD_<dataset>_*.png')
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='Experiment D: quality-vs-budget saturation per regime')
    sub = parser.add_subparsers(dest='command', required=True)

    p_run = sub.add_parser('run', help='run the experiment grid (resumable)')
    p_run.add_argument('--datasets', help='comma list (default: all four)')
    p_run.add_argument('--regimes', help='comma list of det,stoch,hyb (default all)')
    p_run.add_argument('--seeds', type=int, default=10)
    p_run.add_argument('--dry-run', action='store_true',
                       help='print the arm plan without running')
    p_run.set_defaults(func=cmd_run)

    p_rep = sub.add_parser('report', help='aggregate results into data/expD/')
    p_rep.add_argument('--datasets', help='comma list (default: all four)')
    p_rep.set_defaults(func=cmd_report)

    args = parser.parse_args()
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
