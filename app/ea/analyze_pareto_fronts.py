#!/usr/bin/env python3
"""
Compare Pareto fronts across EA runs / seeds.

Works with the current output layout (pareto_front.csv inside seed_<N>/
directories, raw-objective columns) — see docs/ea/RESULTS.md. For each
discovered front only the FINAL generation block is used (earlier
generations are evolution snapshots, not results).

Usage:
    # All seeds of one run
    python3 app/ea/analyze_pareto_fronts.py data/datasets/Iris/results/20260611_200003

    # Several runs of one dataset at once
    python3 app/ea/analyze_pareto_fronts.py data/datasets/Iris/results

    # With plots and CSV export
    python3 app/ea/analyze_pareto_fronts.py <base> --plot --export combined.csv

Outputs:
- per-front summary statistics (raw objectives, feasibility)
- combined non-dominated front across all runs (constrained dominance on
  [raw_mqe_ratio, raw_te, 1−rho] — same rules as the EA itself)
- optional scatter plots and hypervolume-evolution comparison
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

OBJECTIVE_COLS = ['raw_mqe_ratio', 'raw_te', 'raw_topo_corr']


# ---------------------------------------------------------------------------
# Discovery and loading
# ---------------------------------------------------------------------------

def find_pareto_files(base_path: str) -> list:
    """
    Find pareto_front.csv files under base_path:
    directly, in seed_*/, in <run>/, and in <run>/seed_*/.
    """
    if not os.path.isdir(base_path):
        print(f"Error: base path does not exist: {base_path}")
        return []
    patterns = [
        os.path.join(base_path, 'pareto_front.csv'),
        os.path.join(base_path, 'seed_*', 'pareto_front.csv'),
        os.path.join(base_path, '*', 'pareto_front.csv'),
        os.path.join(base_path, '*', 'seed_*', 'pareto_front.csv'),
    ]
    found = sorted({p for pat in patterns for p in glob.glob(pat)})
    return found


def run_id_for(path: str, base_path: str) -> str:
    """Readable run identifier = path of the seed dir relative to base."""
    rel = os.path.relpath(os.path.dirname(path), base_path)
    return rel if rel != '.' else os.path.basename(os.path.abspath(base_path))


def load_final_front(path: str, run_id: str) -> pd.DataFrame:
    """Load pareto_front.csv and keep only the final-generation block."""
    df = pd.read_csv(path)
    if 'generation' in df.columns and len(df):
        df = df[df['generation'] == df['generation'].max()].copy()
    df['run_id'] = run_id
    return df


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def front_stats(df: pd.DataFrame, run_id: str) -> dict:
    """Summary statistics of one final front (raw objectives + feasibility)."""
    cv = pd.to_numeric(df.get('constraint_violation', 0.0), errors='coerce').fillna(0.0)
    stats = {
        'run_id': run_id,
        'front_size': len(df),
        'feasible': int((cv < 1e-9).sum()),
    }
    for col in OBJECTIVE_COLS + ['dead_ratio', 'duration']:
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if len(vals):
                stats[f'{col}_min'] = round(float(vals.min()), 4)
                stats[f'{col}_mean'] = round(float(vals.mean()), 4)
                stats[f'{col}_max'] = round(float(vals.max()), 4)
    return stats


def _objectives(row) -> tuple:
    """(raw_mqe_ratio, raw_te, 1 - rho) with NaN for missing values."""
    mqe = row.get('raw_mqe_ratio', np.nan)
    te = row.get('raw_te', np.nan)
    rho = row.get('raw_topo_corr', np.nan)
    one_minus_rho = 1.0 - rho if pd.notna(rho) else np.nan
    return float(mqe), float(te), one_minus_rho


def _dominates(a: tuple, cv_a: float, b: tuple, cv_b: float) -> bool:
    """Constrained dominance (Deb 2002); NaN objectives are skipped."""
    fa, fb = cv_a < 1e-9, cv_b < 1e-9
    if fa and not fb:
        return True
    if not fa and fb:
        return False
    if not fa and not fb:
        return cv_a < cv_b
    better = False
    for av, bv in zip(a, b):
        if np.isnan(av) or np.isnan(bv):
            continue
        if av > bv:
            return False
        if av < bv:
            better = True
    return better


def combined_non_dominated(all_fronts: pd.DataFrame) -> pd.DataFrame:
    """
    Union of all final fronts reduced to the cross-run non-dominated set
    (constrained dominance on the 3 raw objectives), deduplicated by uid.
    """
    df = all_fronts.drop_duplicates('uid').reset_index(drop=True)
    objs = [_objectives(row) for _, row in df.iterrows()]
    cvs = pd.to_numeric(df.get('constraint_violation', 0.0), errors='coerce') \
            .fillna(0.0).tolist()

    keep = []
    for i in range(len(df)):
        dominated = any(
            j != i and _dominates(objs[j], cvs[j], objs[i], cvs[i])
            for j in range(len(df))
        )
        if not dominated:
            keep.append(i)
    return df.iloc[keep].sort_values('raw_mqe_ratio').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(all_fronts: pd.DataFrame, base_path: str, output_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    runs = all_fronts['run_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(runs), 2)))

    pairs = [
        ('raw_mqe_ratio', 'raw_te', 'MQE ratio vs topographic error',
         'mqe_vs_te.png'),
        ('raw_mqe_ratio', 'raw_topo_corr', 'MQE ratio vs topological correlation ρ',
         'mqe_vs_rho.png'),
        ('raw_te', 'raw_topo_corr', 'Topographic error vs ρ',
         'te_vs_rho.png'),
    ]
    for x, y, title, fname in pairs:
        if x not in all_fronts.columns or y not in all_fronts.columns:
            continue
        plt.figure(figsize=(10, 7))
        for i, run in enumerate(runs):
            sub = all_fronts[all_fronts['run_id'] == run]
            plt.scatter(sub[x], sub[y], label=run, alpha=0.7, s=45,
                        color=colors[i % len(colors)])
        plt.xlabel(x)
        plt.ylabel(y)
        plt.title(f'Final Pareto fronts: {title}')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()

    # Hypervolume evolution per run (pareto_metrics.csv next to each front)
    plt.figure(figsize=(10, 7))
    plotted = False
    for i, run in enumerate(runs):
        metrics_path = os.path.join(base_path, run, 'pareto_metrics.csv')
        if not os.path.exists(metrics_path):
            continue
        m = pd.read_csv(metrics_path)
        if 'hv' in m.columns and m['hv'].notna().any():
            plt.plot(m['generation'], m['hv'], marker='o', label=run,
                     color=colors[i % len(colors)])
            plotted = True
    if plotted:
        plt.xlabel('generation')
        plt.ylabel('hypervolume (normalized space)')
        plt.title('Hypervolume evolution per run')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'hv_evolution.png'), dpi=150)
    plt.close()

    print(f"Plots saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare final Pareto fronts across EA runs/seeds.')
    parser.add_argument('base', help='Run dir, results dir, or seed dir')
    parser.add_argument('--top_n', type=int, default=10,
                        help='Rows of the combined front to display (default 10)')
    parser.add_argument('--export', help='Export the combined front to CSV')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--plot_dir', default=None,
                        help='Plot directory (default: <base>/pareto_analysis)')
    args = parser.parse_args()

    files = find_pareto_files(args.base)
    if not files:
        sys.exit(f"No pareto_front.csv found under {args.base}")

    fronts = []
    stats = []
    for path in files:
        run_id = run_id_for(path, args.base)
        df = load_final_front(path, run_id)
        if df.empty:
            continue
        fronts.append(df)
        stats.append(front_stats(df, run_id))
        print(f"loaded {run_id}: final front {len(df)} solutions")

    if not fronts:
        sys.exit("All discovered fronts are empty.")

    all_fronts = pd.concat(fronts, ignore_index=True)

    print(f"\n{'=' * 78}\nPER-RUN FINAL FRONT STATISTICS\n{'=' * 78}")
    print(pd.DataFrame(stats).to_string(index=False))

    combined = combined_non_dominated(all_fronts)
    print(f"\n{'=' * 78}\nCOMBINED NON-DOMINATED FRONT "
          f"({len(combined)} of {all_fronts['uid'].nunique()} unique solutions)\n{'=' * 78}")
    display_cols = [c for c in
                    ['run_id', 'uid', 'raw_mqe_ratio', 'raw_te',
                     'raw_topo_corr', 'dead_ratio', 'constraint_violation',
                     'map_m', 'map_n']
                    if c in combined.columns]
    print(combined[display_cols].head(args.top_n).to_string(index=False))

    if args.export:
        combined.to_csv(args.export, index=False)
        print(f"\nCombined non-dominated front exported to: {args.export}")

    if args.plot:
        plot_comparison(all_fronts, args.base,
                        args.plot_dir or os.path.join(args.base, 'pareto_analysis'))


if __name__ == '__main__':
    main()
