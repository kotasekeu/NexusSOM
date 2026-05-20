#!/usr/bin/env python3
"""
Business Trips Dataset Generator — with Anomaly Injection

Generates a synthetic business trips dataset with realistic correlations
between destination, distance, duration, and costs.

Two anomaly types:
  outlier   — random rows with extreme cost (8-15× normal) — measurement error
  subgroup  — nearby trips (~400 km) with intercontinental-level costs and duration
              the "Slovakia at Japan prices" ratio anomaly

SOM detection targets:
  outlier  → global_extreme, numeric outlier within cluster
  subgroup → multi-dim outlier (distance_km low in expensive cluster),
             categorical minority (nearby_eu in intercontinental-dominated cluster)

Output files:
  {base}.csv                 — clean CSV for SOM input (no label column)
  {base}_labeled.csv         — same data + _anomaly_label column (0=clean, 1=outlier, 2=subgroup)
  {base}_groundtruth.json    — anomaly IDs and metadata for evaluation
  {base}_config.json         — SOM run config template

Usage:
    python3 generate_trips_dataset.py --rows 800 --output data/datasets/Trips/dataset
    python3 generate_trips_dataset.py --rows 800 --inject both --outlier-fraction 0.03 \\
        --subgroup-fraction 0.04 --seed 42
    python3 generate_trips_dataset.py --rows 800 --inject outlier --seed 42
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


# ── Category parameters ────────────────────────────────────────────────────────

CATEGORIES  = ['domestic', 'nearby_eu', 'europe', 'intercontinental']
CAT_PROBS   = [0.40, 0.25, 0.20, 0.15]

# Per-category generation parameters
CAT_PARAMS = {
    'domestic': {
        'dist_mean': 280,  'dist_std': 140,  'dist_min': 50,   'dist_max': 800,
        'dur_mean':  1.8,  'dur_std':  0.7,  'dur_min': 1,     'dur_max': 4,
        'transport_rate':  10.0,   # CZK per km
        'daily_accommodation': 2200,  # CZK per night
    },
    'nearby_eu': {
        'dist_mean': 580,  'dist_std': 230,  'dist_min': 200,  'dist_max': 1400,
        'dur_mean':  2.8,  'dur_std':  1.1,  'dur_min': 1,     'dur_max': 6,
        'transport_rate':  16.0,
        'daily_accommodation': 3400,
    },
    'europe': {
        'dist_mean': 1600, 'dist_std': 600,  'dist_min': 800,  'dist_max': 4000,
        'dur_mean':  4.8,  'dur_std':  1.8,  'dur_min': 2,     'dur_max': 10,
        'transport_rate':  28.0,
        'daily_accommodation': 5000,
    },
    'intercontinental': {
        'dist_mean': 8200, 'dist_std': 2800, 'dist_min': 3500, 'dist_max': 15000,
        'dur_mean':  9.0,  'dur_std':  3.2,  'dur_min': 5,     'dur_max': 21,
        'transport_rate':  13.0,   # Cheaper per km — long-haul efficiency
        'daily_accommodation': 7800,
    },
}

PURPOSES      = ['conference', 'client_visit', 'training', 'other']
PURPOSE_PROBS = [0.30, 0.35, 0.20, 0.15]
LABEL_COL     = '_anomaly_label'


# ── Row generation ─────────────────────────────────────────────────────────────

def _clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, val))


def _generate_row(row_id: int, rng: np.random.Generator) -> dict:
    cat     = rng.choice(CATEGORIES, p=CAT_PROBS)
    p       = CAT_PARAMS[cat]
    purpose = rng.choice(PURPOSES, p=PURPOSE_PROBS)

    distance_km = _clamp(rng.normal(p['dist_mean'], p['dist_std']),
                         p['dist_min'], p['dist_max'])

    duration_days = int(_clamp(
        round(rng.normal(p['dur_mean'], p['dur_std'])),
        p['dur_min'], p['dur_max']))

    transport_cost    = int(distance_km * p['transport_rate']
                            * rng.lognormal(0.0, 0.28))
    accommodation_cost = int(duration_days * p['daily_accommodation']
                             * rng.lognormal(0.0, 0.38))
    total_cost = transport_cost + accommodation_cost

    cost_per_km = round(total_cost / distance_km, 1) if distance_km > 0 else 0.0

    return {
        'id':                  row_id,
        'destination_category': cat,
        'purpose':             purpose,
        'distance_km':         round(distance_km, 1),
        'duration_days':       duration_days,
        'transport_cost':      transport_cost,
        'accommodation_cost':  accommodation_cost,
        'total_cost':          total_cost,
        'cost_per_km':         cost_per_km,
    }


def generate_clean_dataset(n_rows: int, seed: int | None = None) -> pd.DataFrame:
    rng  = np.random.default_rng(seed)
    rows = [_generate_row(i + 1, rng) for i in range(n_rows)]
    return pd.DataFrame(rows)


# ── Anomaly injection ──────────────────────────────────────────────────────────

def inject_outliers(df: pd.DataFrame, fraction: float,
                    rng: np.random.Generator) -> tuple[pd.DataFrame, list, int]:
    """
    Extreme total_cost (8–15× normal) for random rows.
    Mimics: duplicate billing, currency conversion error, fraud.
    SOM detects via global_extreme + numeric outlier within cluster.
    """
    n   = max(1, int(len(df) * fraction))
    idx = rng.choice(len(df), size=n, replace=False)
    lbl = df.index[idx]

    df  = df.copy()
    mul = rng.uniform(8, 15, n)
    df.loc[lbl, 'transport_cost']     = (df.loc[lbl, 'transport_cost'] * mul).astype(int)
    df.loc[lbl, 'total_cost']         = (df.loc[lbl, 'transport_cost']
                                         + df.loc[lbl, 'accommodation_cost'])
    df.loc[lbl, 'cost_per_km']        = (df.loc[lbl, 'total_cost']
                                         / df.loc[lbl, 'distance_km']).round(1)
    df.loc[lbl, LABEL_COL]            = 1

    ids = df.loc[lbl, 'id'].tolist()
    print(f"  [outlier]  {n} rows injected — cost ×{mul.mean():.1f} avg")
    print(f"             cost range: {df.loc[lbl,'total_cost'].min():,} – "
          f"{df.loc[lbl,'total_cost'].max():,} CZK")
    return df, ids, n


def inject_subgroup(df: pd.DataFrame, fraction: float,
                    rng: np.random.Generator) -> tuple[pd.DataFrame, list, int]:
    """
    "Slovakia at Japan prices" — nearby_eu trips with intercontinental costs/duration.

    destination_category stays 'nearby_eu' (short distance ~400 km) but:
      - transport_cost   → intercontinental level (~40 000–60 000 CZK)
      - accommodation_cost → intercontinental level (7–12 nights × ~8 000 CZK)
      - duration_days    → extended to 7–12 days

    SOM placement: end up in the expensive cluster (by cost), but distance_km
    is anomalously low for that cluster → multi-dim outlier.
    destination_category='nearby_eu' is a categorical minority there.
    """
    ic = CAT_PARAMS['intercontinental']
    candidates = df[
        df['destination_category'].isin(['domestic', 'nearby_eu']) &
        (df[LABEL_COL] == 0)
    ].index
    if len(candidates) == 0:
        print("  [subgroup] WARNING: no domestic/nearby_eu rows — skipping")
        return df, [], 0

    n   = max(3, min(int(len(df) * fraction), len(candidates)))
    sel = rng.choice(candidates, size=n, replace=False)

    df  = df.copy()

    # Extended duration like an intercontinental trip
    dur = np.array([int(_clamp(round(rng.normal(ic['dur_mean'], ic['dur_std'])),
                               ic['dur_min'], ic['dur_max']))
                    for _ in sel])
    df.loc[sel, 'duration_days'] = dur

    # Intercontinental transport cost (flight-level, ignores actual distance)
    df.loc[sel, 'transport_cost'] = np.maximum(0,
        rng.normal(48000, 9000, n) * rng.lognormal(0, 0.2, n)).astype(int)

    # Intercontinental accommodation cost
    df.loc[sel, 'accommodation_cost'] = (
        dur * ic['daily_accommodation'] * rng.lognormal(0, 0.3, n)).astype(int)

    df.loc[sel, 'total_cost'] = (df.loc[sel, 'transport_cost']
                                  + df.loc[sel, 'accommodation_cost'])
    df.loc[sel, 'cost_per_km'] = (df.loc[sel, 'total_cost']
                                   / df.loc[sel, 'distance_km']).round(1)
    df.loc[sel, LABEL_COL]    = 2

    ids = df.loc[sel, 'id'].tolist()
    print(f"  [subgroup] {n} rows injected — nearby_eu distance, intercontinental cost")
    print(f"             distance: {df.loc[sel,'distance_km'].min():.0f} – "
          f"{df.loc[sel,'distance_km'].max():.0f} km  (nearby range)")
    print(f"             cost:     {df.loc[sel,'total_cost'].min():,} – "
          f"{df.loc[sel,'total_cost'].max():,} CZK  (intercontinental range)")
    return df, ids, n


# ── Output helpers ─────────────────────────────────────────────────────────────

def _som_config(csv_path: str) -> dict:
    return {
        "_comment": "SOM config for Trips dataset anomaly validation",
        "map_size": [15, 15],
        "start_learning_rate": 0.8,
        "end_learning_rate": 0.01,
        "lr_decay_type": "linear-drop",
        "start_radius_init_ratio": 0.8,
        "end_radius": 1.0,
        "radius_decay_type": "step-down",
        "start_batch_percent": 1.0,
        "end_batch_percent": 5.0,
        "batch_growth_type": "exp-growth",
        "epoch_multiplier": 1.0,
        "normalize_weights_flag": False,
        "growth_g": 15,
        "random_seed": 42,
        "map_type": "hex",
        "num_batches": 9,
        "max_epochs_without_improvement": 300,
        "mqe_evaluations_per_run": 300,
        "save_checkpoints": False,
        "delimiter": ",",
        "categorical_threshold_numeric": 30,
        "noise_threshold_ratio": 0.2,
        "categorical_threshold_text": 30,
        "primary_id": "id",
        "NEURAL_NETWORKS": {
            "use_lstm": False,
            "lstm_model_path": "app/lstm/models/lstm_latest.keras",
            "lstm_scaler_path": "app/lstm/models/lstm_scaler_latest.pkl",
            "lstm_quality_threshold": 0.75
        }
    }


def print_dataset_summary(df: pd.DataFrame):
    clean    = (df[LABEL_COL] == 0).sum() if LABEL_COL in df.columns else len(df)
    outliers = (df[LABEL_COL] == 1).sum() if LABEL_COL in df.columns else 0
    subgroup = (df[LABEL_COL] == 2).sum() if LABEL_COL in df.columns else 0

    print(f"\n  Total rows:     {len(df)}")
    print(f"  Clean:          {clean}")
    if outliers: print(f"  Outlier (×8-15 cost): {outliers}")
    if subgroup: print(f"  Subgroup (ratio):     {subgroup}")

    print(f"\n  Category distribution:")
    for cat, n in df['destination_category'].value_counts().items():
        print(f"    {cat:<20} {n:>4}  ({n/len(df)*100:.0f}%)")

    print(f"\n  Cost (total_cost) by category:")
    for cat in CATEGORIES:
        subset = df[df['destination_category'] == cat]['total_cost']
        if len(subset) > 0:
            print(f"    {cat:<20} median={subset.median():>8,.0f}  "
                  f"range=[{subset.min():>7,.0f} – {subset.max():>9,.0f}] CZK")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Generate business trips dataset with anomaly injection for SOM testing'
    )
    parser.add_argument('--rows',     '-r', type=int,   default=800,
                        help='Number of rows (default: 800)')
    parser.add_argument('--output',   '-o', type=str,   default='data/datasets/Trips/dataset',
                        help='Output base path (default: data/datasets/Trips/dataset)')
    parser.add_argument('--inject',         type=str,   default='both',
                        choices=['none', 'outlier', 'subgroup', 'both'],
                        help='Anomaly type to inject (default: both)')
    parser.add_argument('--outlier-fraction',  type=float, default=0.03,
                        help='Fraction of outlier rows (default: 0.03)')
    parser.add_argument('--subgroup-fraction', type=float, default=0.04,
                        help='Fraction of subgroup rows (default: 0.04)')
    parser.add_argument('--seed',     type=int,   default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--verbose',  action='store_true')
    args = parser.parse_args()

    print('=' * 60)
    print('TRIPS DATASET GENERATOR')
    print('=' * 60)
    print(f'  rows={args.rows}  inject={args.inject}  seed={args.seed}')

    rng = np.random.default_rng(args.seed)

    # Generate clean dataset
    print(f'\nGenerating {args.rows} clean rows...')
    df = generate_clean_dataset(args.rows, args.seed)
    df[LABEL_COL] = 0

    groundtruth = {
        'seed':    args.seed,
        'n_rows':  args.rows,
        'inject':  args.inject,
        'anomalies': {},
    }

    # Inject anomalies
    if args.inject in ('outlier', 'both'):
        print(f'\nInjecting outliers (fraction={args.outlier_fraction})...')
        df, ids, n = inject_outliers(df, args.outlier_fraction, rng)
        groundtruth['anomalies']['outlier'] = {
            'label':    1,
            'count':    n,
            'fraction': args.outlier_fraction,
            'ids':      ids,
            'description': 'Extreme transport_cost (8-15x normal) — measurement error',
        }

    if args.inject in ('subgroup', 'both'):
        print(f'\nInjecting subgroup (fraction={args.subgroup_fraction})...')
        df, ids, n = inject_subgroup(df, args.subgroup_fraction, rng)
        groundtruth['anomalies']['subgroup'] = {
            'label':    2,
            'count':    n,
            'fraction': args.subgroup_fraction,
            'ids':      ids,
            'description': 'nearby_eu destination with intercontinental cost/duration (ratio anomaly)',
        }

    print_dataset_summary(df)

    # Save outputs
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    som_df = df.drop(columns=[LABEL_COL])
    som_path = str(out) + '.csv'
    som_df.to_csv(som_path, index=False)

    labeled_path = str(out) + '_labeled.csv'
    df.to_csv(labeled_path, index=False)

    gt_path = str(out) + '_groundtruth.json'
    with open(gt_path, 'w') as f:
        json.dump(groundtruth, f, indent=2)

    cfg_path = str(out) + '_config.json'
    with open(cfg_path, 'w') as f:
        json.dump(_som_config(som_path), f, indent=2)

    print(f'\nOutput:')
    print(f'  SOM input:     {som_path}')
    print(f'  Labeled:       {labeled_path}')
    print(f'  Ground truth:  {gt_path}')
    print(f'  SOM config:    {cfg_path}')
    print()

    # Quick evaluation hint
    n_anomaly = sum(d['count'] for d in groundtruth['anomalies'].values())
    print(f'Evaluation:')
    print(f'  Run SOM with:  python3 app/som/run.py --dataset {som_path} '
          f'--config {cfg_path}')
    print(f'  Ground truth:  {n_anomaly} anomalous rows '
          f'({n_anomaly/args.rows*100:.1f}%)')
    print(f'  Compare SOM anomaly IDs against groundtruth.json[anomalies][*][ids]')


if __name__ == '__main__':
    main()
