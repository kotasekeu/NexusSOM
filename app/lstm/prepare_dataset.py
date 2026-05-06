#!/usr/bin/env python3
"""
LSTM Dataset Preparation — Phase 2 (Early Stopping Predictor)

Reads training_checkpoints.json from all EA individuals, normalizes the 6-feature
sequence, resamples to a fixed grid via the progress axis, generates K%-prefix
windows for multiple K values, and writes numpy arrays ready for LSTM training.

Output files (app/lstm/data/):
  sequences_X.npy        shape (N_windows, seq_len, 6)
  sequences_y.npy        shape (N_windows, 3)
  sequences_context.npy  shape (N_windows, 4)
  metadata.json

Usage:
    python3 prepare_dataset.py --results_root data/results
    python3 prepare_dataset.py --results_root data/results --resample_len 100
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

RESULTS_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'results'))

OUTPUT_DIR   = os.path.join(os.path.dirname(__file__), 'data')
RESAMPLE_LEN = 200       # number of evenly-spaced progress points (0..1)
K_FRACTIONS  = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
PARETO_OVERSAMPLE = 2    # repeat Pareto individuals this many times

TARGET_COLS = ['raw_mqe_improvement_ratio', 'raw_topographic_error', 'dead_neuron_ratio']
CONTEXT_COLS = ['ds_n_samples', 'ds_n_active_dimensions', 'ds_n_numeric', 'ds_n_categorical']


# ── normalization ─────────────────────────────────────────────────────────────

def normalize_checkpoints(checkpoints: list[dict]) -> np.ndarray:
    """
    Convert raw checkpoint list to (N, 6) float32 array, normalized so all
    features are comparable across different maps and datasets.

    Columns: progress, mqe_rel, topographic_error, dead_neuron_ratio,
             lr_rel, radius_rel
    """
    if len(checkpoints) < 2:
        return None

    progress = np.array([c['progress'] for c in checkpoints], dtype=np.float64)
    mqe      = np.array([c['mqe']      for c in checkpoints], dtype=np.float64)
    te       = np.array([c['topographic_error'] for c in checkpoints], dtype=np.float64)
    dead     = np.array([c['dead_neuron_ratio']  for c in checkpoints], dtype=np.float64)
    lr       = np.array([c['learning_rate']      for c in checkpoints], dtype=np.float64)
    radius   = np.array([c['radius']             for c in checkpoints], dtype=np.float64)

    init_mqe    = max(mqe[0],    1e-10)
    init_lr     = max(lr[0],     1e-10)
    init_radius = max(radius[0], 1e-10)

    seq = np.stack([
        progress,
        mqe    / init_mqe,
        te,
        dead,
        lr     / init_lr,
        radius / init_radius,
    ], axis=1).astype(np.float32)

    return seq   # (N, 6)


def resample_to_grid(seq: np.ndarray, n_points: int) -> np.ndarray:
    """
    Resample a (N, 6) sequence to (n_points, 6) using the progress column (col 0)
    as the x-axis.  Works even when the original sequence was stopped early and
    does not reach progress=1.
    """
    x_orig = seq[:, 0]           # progress values
    x_new  = np.linspace(x_orig[0], x_orig[-1], n_points)

    resampled = np.zeros((n_points, seq.shape[1]), dtype=np.float32)
    for col in range(seq.shape[1]):
        f = interp1d(x_orig, seq[:, col], kind='linear', bounds_error=False,
                     fill_value=(seq[0, col], seq[-1, col]))
        resampled[:, col] = f(x_new)

    return resampled   # (n_points, 6)


# ── data loading ──────────────────────────────────────────────────────────────

def find_all_seed_dirs(results_root: str) -> list[Path]:
    """Walk results_root and return all seed_* directories."""
    root = Path(results_root)
    seeds = []
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        results_dir = ds_dir / 'results'
        if not results_dir.exists():
            continue
        for run_dir in sorted(results_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            for seed_dir in sorted(run_dir.iterdir()):
                if seed_dir.is_dir() and seed_dir.name.startswith('seed_'):
                    seeds.append(seed_dir)
    return seeds


def load_seed(seed_dir: Path):
    """
    Load all individuals from one seed directory.

    Returns list of dicts:
        uid, targets (3-element array), context (4-element array),
        seq_norm (N,6 normalized), is_pareto
    """
    results_csv = seed_dir / 'results.csv'
    pareto_csv  = seed_dir / 'pareto_front.csv'
    ind_dir     = seed_dir / 'individuals'

    if not results_csv.exists() or not ind_dir.exists():
        return []

    df = pd.read_csv(results_csv)

    # Pareto set
    pareto_uids = set()
    if pareto_csv.exists():
        pf = pd.read_csv(pareto_csv)
        pareto_uids = set(pf['uid'].values)

    # Filter penalized and log-growth
    if 'is_penalized' in df.columns:
        df = df[~df['is_penalized']]
    if 'batch_growth_type' in df.columns:
        df = df[df['batch_growth_type'] != 'log-growth']

    # Check all target columns exist
    for col in TARGET_COLS:
        if col not in df.columns:
            print(f'  ⚠ Missing target column {col} in {results_csv}')
            return []

    records = []
    for _, row in df.iterrows():
        uid = row['uid']
        cp_path = ind_dir / uid / 'csv' / 'training_checkpoints.json'
        if not cp_path.exists():
            continue

        with open(cp_path) as f:
            raw = json.load(f)

        seq = normalize_checkpoints(raw)
        if seq is None or len(seq) < 4:
            continue

        targets = np.array([row[c] for c in TARGET_COLS], dtype=np.float32)
        context = np.array([
            row.get('ds_n_samples',           0),
            row.get('ds_n_active_dimensions', 0),
            row.get('ds_n_numeric',           0),
            row.get('ds_n_categorical',       0),
        ], dtype=np.float32)

        records.append({
            'uid':       uid,
            'targets':   targets,
            'context':   context,
            'seq_norm':  seq,
            'is_pareto': uid in pareto_uids,
            'dataset_name': row.get('dataset_name', seed_dir.parent.parent.name),
        })

    return records


# ── windowing ────────────────────────────────────────────────────────────────

def make_windows(record: dict, resample_len: int, k_fractions: list[float]):
    """
    For one individual, generate K%-prefix windows after resampling to resample_len.
    Returns list of (seq_window, targets, context) tuples.
    """
    full_seq = resample_to_grid(record['seq_norm'], resample_len)  # (L, 6)
    windows = []
    for k in k_fractions:
        cut = max(2, int(round(k * resample_len)))
        window = full_seq[:cut, :]      # (cut, 6)
        windows.append((window, record['targets'], record['context']))
    return windows


# ── main ──────────────────────────────────────────────────────────────────────

def build_dataset(results_root: str, resample_len: int, k_fractions: list[float],
                  pareto_oversample: int):
    seed_dirs = find_all_seed_dirs(results_root)
    print(f'Found {len(seed_dirs)} seed directories')

    all_records = []
    for sd in seed_dirs:
        recs = load_seed(sd)
        print(f'  {sd.parent.parent.name}/{sd.name}: {len(recs)} individuals')
        all_records.extend(recs)

    if not all_records:
        print('No records found — check results_root path')
        sys.exit(1)

    print(f'\nTotal individuals: {len(all_records)}')
    n_pareto = sum(r['is_pareto'] for r in all_records)
    print(f'  Pareto: {n_pareto}  |  oversampled ×{pareto_oversample}')

    # Pareto oversampling
    pareto_recs = [r for r in all_records if r['is_pareto']]
    extra = pareto_recs * (pareto_oversample - 1)
    all_records = all_records + extra

    # Build windows
    X_list, y_list, ctx_list, dataset_list = [], [], [], []
    for rec in all_records:
        for seq_w, tgt, ctx in make_windows(rec, resample_len, k_fractions):
            X_list.append(seq_w)
            y_list.append(tgt)
            ctx_list.append(ctx)
            dataset_list.append(rec['dataset_name'])

    # Stack — sequences have variable length so we use an object array first,
    # then verify they all have the right shapes.
    print(f'\nTotal windows: {len(X_list)}')
    seq_lens = [x.shape[0] for x in X_list]
    print(f'  Sequence lengths: min={min(seq_lens)} max={max(seq_lens)} '
          f'(K ∈ {[int(k*100) for k in k_fractions]}%)')

    X   = np.array(X_list,   dtype=object)  # will be ragged → store as numpy arrays
    y   = np.stack(y_list,   axis=0)
    ctx = np.stack(ctx_list, axis=0)

    return X, y, ctx, dataset_list


def split_by_individual(all_records, k_fractions, resample_len, pareto_oversample,
                         train_frac=0.70, val_frac=0.15, random_state=42):
    """
    Split at individual level (stratified by dataset), then expand to windows.
    Prevents data leakage from the same individual appearing in train and val/test.
    """
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(random_state)

    # Group by dataset
    datasets = sorted(set(r['dataset_name'] for r in all_records))
    train_recs, val_recs, test_recs = [], [], []

    for ds in datasets:
        ds_recs = [r for r in all_records if r['dataset_name'] == ds]
        n = len(ds_recs)
        n_test = max(1, int(round(n * (1 - train_frac - val_frac))))
        n_val  = max(1, int(round(n * val_frac)))
        n_train = n - n_test - n_val

        idx = np.arange(n)
        rng.shuffle(idx)
        train_recs.extend([ds_recs[i] for i in idx[:n_train]])
        val_recs.extend([  ds_recs[i] for i in idx[n_train:n_train+n_val]])
        test_recs.extend([ ds_recs[i] for i in idx[n_train+n_val:]])

    def expand(records, oversample_pareto=True):
        pool = list(records)
        if oversample_pareto:
            pool += [r for r in records if r['is_pareto']] * (pareto_oversample - 1)
        X_l, y_l, ctx_l = [], [], []
        for rec in pool:
            for seq_w, tgt, ctx in make_windows(rec, resample_len, k_fractions):
                X_l.append(seq_w)
                y_l.append(tgt)
                ctx_l.append(ctx)
        return (np.array(X_l, dtype=object),
                np.stack(y_l, axis=0),
                np.stack(ctx_l, axis=0))

    X_train, y_train, ctx_train = expand(train_recs)
    X_val,   y_val,   ctx_val   = expand(val_recs,   oversample_pareto=False)
    X_test,  y_test,  ctx_test  = expand(test_recs,  oversample_pareto=False)

    return (X_train, y_train, ctx_train,
            X_val,   y_val,   ctx_val,
            X_test,  y_test,  ctx_test,
            len(train_recs), len(val_recs), len(test_recs))


def main():
    parser = argparse.ArgumentParser(description='Prepare LSTM Phase 2 dataset')
    parser.add_argument('--results_root', default=RESULTS_ROOT)
    parser.add_argument('--resample_len', type=int, default=RESAMPLE_LEN,
                        help='Number of evenly-spaced points after resampling (default: 200)')
    parser.add_argument('--output', default=OUTPUT_DIR)
    args = parser.parse_args()

    print('=' * 70)
    print('LSTM DATASET PREPARATION — Phase 2')
    print('=' * 70)
    print(f'Results root : {args.results_root}')
    print(f'Resample len : {args.resample_len}')
    print(f'K fractions  : {[int(k*100) for k in K_FRACTIONS]}%')
    print(f'Pareto ×     : {PARETO_OVERSAMPLE}')
    print()

    seed_dirs = find_all_seed_dirs(args.results_root)
    print(f'Found {len(seed_dirs)} seed directories')

    all_records = []
    for sd in seed_dirs:
        recs = load_seed(sd)
        print(f'  {sd.parent.parent.name}/{sd.name}: {len(recs)}')
        all_records.extend(recs)

    if not all_records:
        print('No records found.')
        sys.exit(1)

    n_pareto = sum(r['is_pareto'] for r in all_records)
    print(f'\nTotal individuals : {len(all_records)}')
    print(f'Pareto individuals: {n_pareto}')

    (X_train, y_train, ctx_train,
     X_val,   y_val,   ctx_val,
     X_test,  y_test,  ctx_test,
     n_tr, n_va, n_te) = split_by_individual(
        all_records, K_FRACTIONS, args.resample_len, PARETO_OVERSAMPLE)

    print(f'\nSplit (individual level):')
    print(f'  Train: {n_tr} individuals → {len(X_train)} windows')
    print(f'  Val  : {n_va} individuals → {len(X_val)} windows')
    print(f'  Test : {n_te} individuals → {len(X_test)} windows')

    os.makedirs(args.output, exist_ok=True)

    np.save(os.path.join(args.output, 'X_train.npy'),   X_train)
    np.save(os.path.join(args.output, 'y_train.npy'),   y_train)
    np.save(os.path.join(args.output, 'ctx_train.npy'), ctx_train)

    np.save(os.path.join(args.output, 'X_val.npy'),     X_val)
    np.save(os.path.join(args.output, 'y_val.npy'),     y_val)
    np.save(os.path.join(args.output, 'ctx_val.npy'),   ctx_val)

    np.save(os.path.join(args.output, 'X_test.npy'),    X_test)
    np.save(os.path.join(args.output, 'y_test.npy'),    y_test)
    np.save(os.path.join(args.output, 'ctx_test.npy'),  ctx_test)

    meta = {
        'resample_len':      args.resample_len,
        'k_fractions':       K_FRACTIONS,
        'pareto_oversample': PARETO_OVERSAMPLE,
        'n_features':        6,
        'n_context':         4,
        'n_targets':         3,
        'feature_names':     ['progress', 'mqe_rel', 'topographic_error',
                              'dead_neuron_ratio', 'lr_rel', 'radius_rel'],
        'context_names':     CONTEXT_COLS,
        'target_names':      TARGET_COLS,
        'n_individuals':     len(all_records),
        'n_pareto':          n_pareto,
        'split': {
            'train_individuals': n_tr,
            'val_individuals':   n_va,
            'test_individuals':  n_te,
            'train_windows':     len(X_train),
            'val_windows':       len(X_val),
            'test_windows':      len(X_test),
        },
    }
    meta_path = os.path.join(args.output, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'\nSaved to: {args.output}/')
    print(f'  X_train.npy   {X_train.shape[0]} × variable × 6')
    print(f'  y_train.npy   {y_train.shape}')
    print(f'  ctx_train.npy {ctx_train.shape}')
    print(f'  (val + test analogously)')
    print(f'  metadata.json')
    print('\nDone.')


if __name__ == '__main__':
    main()
