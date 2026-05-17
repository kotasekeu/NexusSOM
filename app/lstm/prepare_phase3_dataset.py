#!/usr/bin/env python3
"""
LSTM Dataset Preparation — Phase 3 (Dynamic Schedule Controller)

Reads trajectories.json produced by generate_phase3_data.py and builds
advantage-weighted training arrays for the Phase 3 LSTM controller.

Paradigm: advantage-weighted behavioral cloning
  - Imitate actions (lr_factor, radius_factor) from trajectories that improved MQE
  - sample_weight = advantage = max(0, delta_mqe), normalized to [0, 1]
  - Baseline trajectories are excluded (all lr_factor=1 → no signal)

Output files (app/lstm/data/phase3/):
  X_train.npy        (N, T, 6)  — padded checkpoint sequences
  y_train.npy        (N, T, 2)  — padded target actions (lr_f, radius_f)
  ctx_train.npy      (N, 4)     — dataset context (constant per individual)
  adv_train.npy      (N,)       — advantage weights in [0, 1]
  (+ val/test analogues)
  metadata_p3.json

Usage:
    python3 app/lstm/prepare_phase3_dataset.py \\
        --trajectories app/lstm/data/phase3/trajectories.json \\
        --seed_dir     data/datasets/LungCancerDataset/results/20260513_181811/seed_42
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'data', 'phase3')

CONTEXT_COLS = ['ds_n_samples', 'ds_n_active_dimensions', 'ds_n_numeric', 'ds_n_categorical']


# ── normalization (identical to Phase 2) ─────────────────────────────────────

def normalize_checkpoints(checkpoints: list) -> tuple:
    """
    Returns (seq_norm, actions) where:
      seq_norm : (T, 6) float32 — same 6 features as Phase 2
      actions  : (T, 2) float32 — (lr_factor, radius_factor) at each checkpoint
    """
    if len(checkpoints) < 2:
        return None, None

    progress = np.array([c['progress']           for c in checkpoints], np.float64)
    mqe      = np.array([c['mqe']                for c in checkpoints], np.float64)
    te       = np.array([c['topographic_error']  for c in checkpoints], np.float64)
    dead     = np.array([c['dead_neuron_ratio']   for c in checkpoints], np.float64)
    lr       = np.array([c['learning_rate']       for c in checkpoints], np.float64)
    radius   = np.array([c['radius']              for c in checkpoints], np.float64)

    lr_f     = np.array([c.get('lr_factor',     1.0) for c in checkpoints], np.float32)
    radius_f = np.array([c.get('radius_factor', 1.0) for c in checkpoints], np.float32)

    # 1.0 for timesteps where a perturbation was actually applied, 0.0 otherwise.
    # Non-perturbed timesteps (60 % of data) get zero loss weight so the model
    # cannot collapse to predicting 1.0 for everything.
    perturb_mask = ((np.abs(lr_f - 1.0) > 0.01) | (np.abs(radius_f - 1.0) > 0.01)
                    ).astype(np.float32)

    init_mqe    = max(mqe[0],    1e-10)
    init_lr     = max(lr[0],     1e-10)
    init_radius = max(radius[0], 1e-10)

    seq_norm = np.stack([
        progress,
        mqe    / init_mqe,
        te,
        dead,
        lr     / init_lr,
        radius / init_radius,
    ], axis=1).astype(np.float32)

    actions = np.stack([lr_f, radius_f], axis=1)

    return seq_norm, actions, perturb_mask


def pad_ragged(arrays: list, pad_value: float = 0.0) -> np.ndarray:
    """Pad list of (T_i, F) arrays to (N, T_max, F)."""
    T_max = max(a.shape[0] for a in arrays)
    F     = arrays[0].shape[1]
    out   = np.full((len(arrays), T_max, F), pad_value, dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, :a.shape[0], :] = a
    return out


def pad_ragged_1d(arrays: list, pad_value: float = 0.0) -> np.ndarray:
    """Pad list of (T_i,) arrays to (N, T_max)."""
    T_max = max(a.shape[0] for a in arrays)
    out   = np.full((len(arrays), T_max), pad_value, dtype=np.float32)
    for i, a in enumerate(arrays):
        out[i, :a.shape[0]] = a
    return out


# ── data loading ──────────────────────────────────────────────────────────────

def load_context_from_results(seed_dirs: list) -> dict:
    """Build uid → context_array mapping from results.csv across all given seed dirs."""
    ctx_map = {}
    for sd in seed_dirs:
        csv_path = Path(sd) / 'results.csv'
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            ctx_map[row['uid']] = np.array([
                row.get('ds_n_samples',           0),
                row.get('ds_n_active_dimensions', 0),
                row.get('ds_n_numeric',           0),
                row.get('ds_n_categorical',       0),
            ], dtype=np.float32)
    return ctx_map


def _find_seed_dirs(results_root: str) -> list:
    """Scan results_root for all seed_* dirs that contain results.csv."""
    root = Path(results_root)
    dirs = []
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        results_dir = ds_dir / 'results'
        if not results_dir.exists():
            continue
        for ts_dir in sorted(results_dir.iterdir()):
            if not ts_dir.is_dir():
                continue
            for seed_dir in sorted(ts_dir.iterdir()):
                if seed_dir.is_dir() and seed_dir.name.startswith('seed_'):
                    if (seed_dir / 'results.csv').exists():
                        dirs.append(seed_dir)
    return dirs


def build_records(trajectories: list, ctx_map: dict) -> list:
    """
    Convert trajectories list into training records.
    Skips baseline variants (no perturbation signal).
    """
    records = []
    for traj in trajectories:
        if traj['variant'] == 'baseline':
            continue

        uid = traj['uid']
        if uid not in ctx_map:
            continue

        seq_norm, actions, perturb_mask = normalize_checkpoints(traj['checkpoints'])
        if seq_norm is None:
            continue

        # Support both old format (delta_mqe only) and new (multi-objective advantage)
        advantage = float(traj.get('advantage', traj.get('delta_mqe', 0.0)))

        records.append({
            'uid':          uid,
            'variant':      traj['variant'],
            'seq_norm':     seq_norm,       # (T, 6)
            'actions':      actions,        # (T, 2)
            'perturb_mask': perturb_mask,   # (T,) — 1.0 where perturbation applied
            'delta_mqe':    float(traj.get('delta_mqe', 0.0)),
            'delta_te':     float(traj.get('delta_te',  0.0)),
            'delta_dead':   float(traj.get('delta_dead', 0.0)),
            'advantage':    advantage,
            'context':      ctx_map[uid],   # (4,)
        })

    return records


def compute_advantages(records: list) -> np.ndarray:
    """
    Multi-objective advantage via Z-score normalization.

    Each component (delta_mqe, delta_te, delta_dead) is divided by its
    global standard deviation across ALL records before summing.  This
    prevents one component from dominating purely because its values happen
    to be on a larger scale on a given dataset.

    advantage = (Δmqe / σ_mqe) + (Δte / σ_te) + (Δdead / σ_dead)

    Negative advantages are clipped to 0 — model only imitates trajectories
    that improved all-objectives on net.  Result is normalized to [0, 1].
    """
    delta_mqe  = np.array([r['delta_mqe']  for r in records], dtype=np.float64)
    delta_te   = np.array([r['delta_te']   for r in records], dtype=np.float64)
    delta_dead = np.array([r['delta_dead'] for r in records], dtype=np.float64)

    std_mqe  = delta_mqe.std()  if delta_mqe.std()  > 1e-10 else 1.0
    std_te   = delta_te.std()   if delta_te.std()   > 1e-10 else 1.0
    std_dead = delta_dead.std() if delta_dead.std() > 1e-10 else 1.0

    print(f'  Advantage σ — mqe: {std_mqe:.5f}  te: {std_te:.5f}  dead: {std_dead:.5f}')

    raw = (delta_mqe / std_mqe) + (delta_te / std_te) + (delta_dead / std_dead)
    raw = np.clip(raw, 0.0, None).astype(np.float32)
    max_val = raw.max() if raw.max() > 0 else 1.0
    return raw / max_val


def split_records(records: list, train_frac=0.70, val_frac=0.15, seed=42):
    """Split at uid level to avoid leakage across variants of the same individual."""
    rng = np.random.default_rng(seed)
    uids = list({r['uid'] for r in records})
    rng.shuffle(uids)

    n = len(uids)
    n_val  = max(1, int(round(n * val_frac)))
    n_test = max(1, int(round(n * (1 - train_frac - val_frac))))

    train_uids = set(uids[:n - n_val - n_test])
    val_uids   = set(uids[n - n_val - n_test:n - n_test])
    test_uids  = set(uids[n - n_test:])

    train = [r for r in records if r['uid'] in train_uids]
    val   = [r for r in records if r['uid'] in val_uids]
    test  = [r for r in records if r['uid'] in test_uids]
    return train, val, test


# ── main ──────────────────────────────────────────────────────────────────────

def prepare(trajectories_path: str, seed_dirs: list, output_dir: str):
    print('=' * 60)
    print('PHASE 3 DATASET PREPARATION')
    print('=' * 60)

    with open(trajectories_path, encoding='utf-8') as f:
        trajectories = json.load(f)
    print(f'Loaded {len(trajectories)} trajectories from {trajectories_path}')

    ctx_map = load_context_from_results(seed_dirs)
    print(f'Context map: {len(ctx_map)} individuals from {len(seed_dirs)} seed dir(s)')

    records = build_records(trajectories, ctx_map)
    n_pert   = len(records)
    n_better = sum(1 for r in records if r['advantage'] > 0)
    print(f'\nPerturbed trajectories: {n_pert}')
    print(f'  Better than baseline (advantage > 0): {n_better}/{n_pert} '
          f'({n_better/max(1,n_pert)*100:.0f}%)')
    print(f'  advantage range: [{min(r["advantage"] for r in records):.4f}, '
          f'{max(r["advantage"] for r in records):.4f}]')
    print(f'  delta_mqe range: [{min(r["delta_mqe"] for r in records):.4f}, '
          f'{max(r["delta_mqe"] for r in records):.4f}]')

    if not records:
        print('ERROR: No valid records found.')
        sys.exit(1)

    train_recs, val_recs, test_recs = split_records(records)
    print(f'\nSplit (uid level):')
    print(f'  Train: {len(train_recs)} trajectories')
    print(f'  Val  : {len(val_recs)} trajectories')
    print(f'  Test : {len(test_recs)} trajectories')

    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_recs in [('train', train_recs), ('val', val_recs), ('test', test_recs)]:
        if not split_recs:
            continue
        adv = compute_advantages(split_recs)

        X   = pad_ragged([r['seq_norm']     for r in split_recs])   # (N, T, 6)
        y   = pad_ragged([r['actions']      for r in split_recs])   # (N, T, 2)
        msk = pad_ragged_1d([r['perturb_mask'] for r in split_recs])  # (N, T)
        ctx = np.stack([r['context'] for r in split_recs])            # (N, 4)

        np.save(os.path.join(output_dir, f'X_{split_name}.npy'),   X)
        np.save(os.path.join(output_dir, f'y_{split_name}.npy'),   y)
        np.save(os.path.join(output_dir, f'msk_{split_name}.npy'), msk)
        np.save(os.path.join(output_dir, f'ctx_{split_name}.npy'), ctx)
        np.save(os.path.join(output_dir, f'adv_{split_name}.npy'), adv)
        pct_active = msk.mean() * 100
        print(f'  {split_name}: X={X.shape}, y={y.shape}, '
              f'active timesteps={pct_active:.0f}%, adv mean={adv.mean():.3f}')

    meta = {
        'n_features':   6,
        'n_actions':    2,
        'n_context':    4,
        'feature_names': ['progress', 'mqe_rel', 'topographic_error',
                          'dead_neuron_ratio', 'lr_rel', 'radius_rel'],
        'action_names':  ['lr_factor', 'radius_factor'],
        'context_names': CONTEXT_COLS,
        'split': {
            'train': len(train_recs),
            'val':   len(val_recs),
            'test':  len(test_recs),
        },
        'n_better_than_baseline': n_better,
        'n_perturbed_total':      n_pert,
    }
    with open(os.path.join(output_dir, 'metadata_p3.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)

    print(f'\nSaved to: {output_dir}/')
    print('Done.')


def main():
    default_traj = os.path.join(OUTPUT_DIR, 'trajectories.json')

    parser = argparse.ArgumentParser(description='Prepare LSTM Phase 3 dataset')
    parser.add_argument('--trajectories', default=default_traj,
                        help=f'Path to trajectories.json (default: {default_traj})')
    parser.add_argument('--results_root', default=None,
                        help='Root of datasets dir — auto-discovers all seed dirs for context')
    parser.add_argument('--seed_dir', default=None,
                        help='Single EA seed directory for context (use with single-seed trajectories)')
    parser.add_argument('--output', default=OUTPUT_DIR,
                        help=f'Output directory (default: {OUTPUT_DIR})')
    args = parser.parse_args()

    if args.results_root:
        seed_dirs = _find_seed_dirs(args.results_root)
        if not seed_dirs:
            print(f'ERROR: No seed dirs found under {args.results_root}')
            sys.exit(1)
        print(f'Found {len(seed_dirs)} seed dir(s) under {args.results_root}')
    elif args.seed_dir:
        seed_dirs = [args.seed_dir]
    else:
        # Try to auto-detect from default results_root
        default_root = 'data/datasets'
        seed_dirs = _find_seed_dirs(default_root)
        if not seed_dirs:
            print('ERROR: No seed dirs found. Provide --results_root or --seed_dir.')
            sys.exit(1)
        print(f'Auto-detected {len(seed_dirs)} seed dir(s) from {default_root}')

    prepare(args.trajectories, seed_dirs, args.output)


if __name__ == '__main__':
    main()
