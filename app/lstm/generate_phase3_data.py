#!/usr/bin/env python3
"""
Generate Phase 3 LSTM Training Data — Perturbation Trajectories

For each Pareto individual from an EA run:
  - Run baseline (clean, no perturbation)
  - Run N_VARIANTS with random lr/radius perturbation at each checkpoint
  - Record checkpoint sequence + applied factors + delta_final_mqe vs baseline

Output: app/lstm/data/phase3/trajectories.json

Usage (single seed):
    python3 app/lstm/generate_phase3_data.py \\
        --seed_dir data/datasets/LungCancerDataset/results/20260513_181811/seed_42 \\
        --dataset  data/datasets/LungCancerDataset/dataset.csv \\
        --n_pareto 5 --n_variants 8 --output app/lstm/data/phase3

Usage (all datasets + all seeds — dataset.csv auto-detected):
    python3 app/lstm/generate_phase3_data.py \\
        --results_root data/datasets \\
        --n_pareto 5 --n_variants 8 --output app/lstm/data/phase3
"""

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from som.som import KohonenSOM
from som.preprocess import preprocess_data


# ── constants ─────────────────────────────────────────────────────────────────

LR_PERTURB     = 0.25   # ±25 % per checkpoint
RADIUS_PERTURB = 0.25
PERTURB_PROB   = 0.4    # probability of applying perturbation at each checkpoint

FIXED = {
    'end_radius':                  1.0,
    'mqe_evaluations_per_run':     300,
    'map_type':                    'hex',
    'save_checkpoints':            True,
    'checkpoint_every_mqe':        True,
    'checkpoint_count':            25,
    'early_stopping_window':       50,
    'max_epochs_without_improvement': 500,
    'normalize_weights_flag':      False,
    'early_stopping_patience':     500,
}

# Columns from results.csv that are SOM hyperparameters
SOM_PARAM_COLS = [
    'map_m', 'map_n', 'start_learning_rate', 'end_learning_rate', 'lr_decay_type',
    'start_radius_init_ratio', 'radius_decay_type', 'start_batch_percent',
    'end_batch_percent', 'batch_growth_type', 'epoch_multiplier', 'growth_g', 'num_batches',
]

# Fields from results.csv to drop (not SOM params)
DROP_FIELDS = {
    'uid', 'is_penalized', 'generation', 'population_id', 'rank', 'crowding_distance',
    'best_mqe', 'raw_best_mqe', 'raw_mqe_improvement_ratio', 'raw_topographic_error',
    'topographic_error', 'mqe_improvement_ratio', 'initial_mqe', 'dead_neuron_ratio',
    'dead_neuron_count', 'duration', 'cnn_quality_score', 'constraint_violation',
    'penalty_factor', 'penalty_reason', 'map_size', 'u_matrix_mean', 'u_matrix_std',
    'u_matrix_max', 'distance_map_max', 'total_weight_updates', 'epochs_ran',
    'ds_has_primary_id', 'ds_missing_ratio', 'ds_n_dimensions', 'ds_n_ignored',
    'ds_n_missing_values', 'ds_n_original_cols', 'ds_n_training_cols', 'dataset_name',
    'ds_n_samples', 'ds_n_active_dimensions', 'ds_n_numeric', 'ds_n_categorical',
    'comment', 'nn_config', 'sample_size', 'input_dim',
}


# ── perturbation functions ────────────────────────────────────────────────────

def make_constrained_perturb_fn(rng, max_radius: float, max_lr: float,
                                 apply_lr: bool = True, apply_radius: bool = True,
                                 lr_perturb: float = LR_PERTURB,
                                 radius_perturb: float = RADIUS_PERTURB,
                                 prob: float = PERTURB_PROB):
    """
    Physically-constrained perturbation function.

    Uses the checkpoint's actual current LR and radius to compute the proposed
    new absolute value, clips it to valid bounds, then returns the factor needed
    to reach that value.  This prevents cumulative drift into nonsensical ranges
    (e.g. radius > map size, or LR growing instead of declining).

    Bounds enforced per checkpoint:
      lr     ∈ [1e-4, max_lr]          — max_lr = start_learning_rate of the individual
      radius ∈ [1.0,  max_radius]      — max_radius = max(map_m, map_n)
    """
    def fn(checkpoint):
        lr_f = 1.0
        if apply_lr and rng.random() < prob:
            current_lr  = max(float(checkpoint.get('learning_rate', 0.1)), 1e-10)
            proposed_lr = current_lr * rng.uniform(1 - lr_perturb, 1 + lr_perturb)
            proposed_lr = float(np.clip(proposed_lr, 1e-4, max_lr))
            lr_f        = proposed_lr / current_lr

        rad_f = 1.0
        if apply_radius and rng.random() < prob:
            current_r  = max(float(checkpoint.get('radius', 5.0)), 1e-10)
            proposed_r = current_r * rng.uniform(1 - radius_perturb, 1 + radius_perturb)
            proposed_r = float(np.clip(proposed_r, 1.0, max_radius))
            rad_f      = proposed_r / current_r

        return lr_f, rad_f
    return fn


# ── SOM construction ──────────────────────────────────────────────────────────

def build_som(row: pd.Series, data: np.ndarray) -> KohonenSOM:
    """Construct KohonenSOM from a results.csv row."""
    params = {k: v for k, v in row.items() if k not in DROP_FIELDS}
    params.update(FIXED)
    # Remove any remaining non-SOM keys that crept in
    for k in list(params.keys()):
        if isinstance(params[k], float) and np.isnan(params[k]):
            params.pop(k)
    return KohonenSOM(dim=data.shape[1], **params)


def run_som(row: pd.Series, data: np.ndarray, ignore_mask: np.ndarray,
            dynamic_fn=None, seed: int = 42) -> dict:
    """Run one SOM training. Returns dict with checkpoints + final metrics."""
    np.random.seed(seed)
    som = build_som(row, data)
    import tempfile, shutil
    tmp = tempfile.mkdtemp()
    try:
        result = som.train(data, ignore_mask=ignore_mask, working_dir=tmp,
                           dynamic_schedule_fn=dynamic_fn)
        topo_error = som.calculate_topographic_error(data, mask=ignore_mask)
        _, dead_ratio = som.calculate_dead_neurons(data)
        cp_path = os.path.join(tmp, 'csv', 'training_checkpoints.json')
        checkpoints = json.loads(Path(cp_path).read_text()) if os.path.exists(cp_path) else []
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    return {
        'final_mqe':               result['best_mqe'],
        'final_topographic_error': topo_error,
        'final_dead_ratio':        dead_ratio,
        'checkpoints':             checkpoints,
    }


# ── parallel worker ───────────────────────────────────────────────────────────

def _worker(task: dict) -> dict:
    """
    Picklable worker for Pool. Reconstructs dynamic_fn from perturb_config dict.
    task keys: row_dict, data, ignore_mask, perturb_config, som_seed
    perturb_config: None (baseline) | {'type': str, 'seed': int}
    """
    row   = pd.Series(task['row_dict'])
    data  = task['data']
    mask  = task['ignore_mask']
    pc    = task['perturb_config']
    seed  = task['som_seed']

    if pc is None:
        dynamic_fn = None
    else:
        map_m      = int(task['row_dict'].get('map_m', 20))
        map_n      = int(task['row_dict'].get('map_n', 20))
        max_radius = float(max(map_m, map_n))
        max_lr     = float(task['row_dict'].get('start_learning_rate', 1.0))

        rng   = np.random.default_rng(pc['seed'])
        ptype = pc['type']
        if ptype == 'lr+radius':
            dynamic_fn = make_constrained_perturb_fn(rng, max_radius, max_lr)
        elif ptype == 'lr_only':
            dynamic_fn = make_constrained_perturb_fn(rng, max_radius, max_lr,
                                                      apply_radius=False)
        else:  # radius_only
            dynamic_fn = make_constrained_perturb_fn(rng, max_radius, max_lr,
                                                      apply_lr=False)

    result = run_som(row, data, mask, dynamic_fn=dynamic_fn, seed=seed)
    return {**result, 'uid': task['uid'], 'variant': task['variant'],
            'variant_seed': task['variant_seed'], 'perturb_type': pc['type'] if pc else 'baseline'}


# ── main pipeline ─────────────────────────────────────────────────────────────

def generate(seed_dir: str, dataset_path: str, n_pareto: int, n_variants: int,
             output_dir: str, n_workers: int = 0, _return_trajectories: bool = False):
    seed_dir = Path(seed_dir)
    results_csv = seed_dir / 'results.csv'
    pareto_csv  = seed_dir / 'pareto_front.csv'

    if not results_csv.exists():
        print(f'ERROR: results.csv not found in {seed_dir}')
        sys.exit(1)

    results = pd.read_csv(results_csv)
    pareto  = pd.read_csv(pareto_csv) if pareto_csv.exists() else results.head(n_pareto)

    valid       = results[~results.get('is_penalized', pd.Series(False, index=results.index))]
    pareto_uids = set(pareto['uid'].values)
    pareto_rows = valid[valid['uid'].isin(pareto_uids)].head(n_pareto)

    if len(pareto_rows) == 0:
        print('ERROR: No valid Pareto individuals found.')
        sys.exit(1)

    print(f'Found {len(pareto_rows)} Pareto individuals (requested {n_pareto})')

    # Preprocess dataset once — shared (read-only) across all workers
    print(f'Preprocessing dataset: {dataset_path}')
    import tempfile
    preprocess_config = {
        'PREPROCES_DATA': {
            'delimiter': ',',
            'categorical_threshold_numeric': 30,
            'noise_threshold_ratio': 0.2,
            'categorical_threshold_text': 30,
        }
    }
    with tempfile.TemporaryDirectory() as tmp:
        input_df = pd.read_csv(dataset_path)
        npy_path, _, ignore_mask, _ = preprocess_data(input_df, preprocess_config, tmp)
        data = np.load(npy_path)
    print(f'Dataset shape: {data.shape}')

    # Build task list for all individuals × variants
    rng_master = np.random.default_rng(42)
    tasks = []
    for _, row in pareto_rows.iterrows():
        uid           = row['uid']
        row_dict      = row.to_dict()
        variant_seeds = rng_master.integers(0, 100_000, size=n_variants)

        tasks.append({'uid': uid, 'row_dict': row_dict, 'data': data,
                      'ignore_mask': ignore_mask, 'perturb_config': None,
                      'som_seed': 42, 'variant': 'baseline', 'variant_seed': 42})

        for v_idx, v_seed in enumerate(variant_seeds):
            ptype = ['lr+radius', 'lr_only', 'radius_only'][v_idx % 3]
            tasks.append({'uid': uid, 'row_dict': row_dict, 'data': data,
                          'ignore_mask': ignore_mask,
                          'perturb_config': {'type': ptype, 'seed': int(v_seed)},
                          'som_seed': int(v_seed) % 10000,
                          'variant': ptype, 'variant_seed': int(v_seed)})

    total = len(tasks)
    workers = n_workers if n_workers > 0 else max(1, cpu_count() - 1)
    print(f'Running {total} SOM trains across {workers} workers...')

    t_start = time.time()
    with Pool(processes=workers) as pool:
        raw_results = pool.map(_worker, tasks)
    elapsed = time.time() - t_start
    print(f'All runs done in {elapsed:.1f}s  ({elapsed/total:.1f}s/run avg)')

    # Build trajectories — multi-objective advantage vs baseline
    # advantage = delta_mqe + α·delta_te + β·delta_dead  (all: positive = better)
    # Weights chosen so MQE dominates but topology destruction is penalized.
    ADV_W_MQE  = 1.0
    ADV_W_TE   = 0.5
    ADV_W_DEAD = 0.3

    baselines = {
        r['uid']: {
            'mqe':  r['final_mqe'],
            'te':   r['final_topographic_error'],
            'dead': r['final_dead_ratio'],
        }
        for r in raw_results if r['variant'] == 'baseline'
    }

    trajectories = []
    for r in raw_results:
        uid  = r['uid']
        base = baselines[uid]

        delta_mqe  = base['mqe']  - r['final_mqe']                  # positive = better
        delta_te   = base['te']   - r['final_topographic_error']     # positive = better
        delta_dead = base['dead'] - r['final_dead_ratio']            # positive = better
        advantage  = (ADV_W_MQE  * delta_mqe
                    + ADV_W_TE   * delta_te
                    + ADV_W_DEAD * delta_dead)

        trajectories.append({
            'uid':                  uid,
            'variant':              r['variant'],
            'variant_seed':         r['variant_seed'],
            'final_mqe':            r['final_mqe'],
            'final_topographic_error': r['final_topographic_error'],
            'final_dead_ratio':     r['final_dead_ratio'],
            'delta_mqe':            float(delta_mqe),
            'delta_te':             float(delta_te),
            'delta_dead':           float(delta_dead),
            'advantage':            float(advantage),
            'better_than_baseline': advantage > 0,
            'checkpoints':          r['checkpoints'],
        })

    # Per-individual summary (printed after pool completes — clean, no interleaving)
    from collections import defaultdict
    by_uid = defaultdict(list)
    for t in trajectories:
        by_uid[t['uid']].append(t)
    for uid, group in by_uid.items():
        base  = next((t for t in group if t['variant'] == 'baseline'), None)
        varts = [t for t in group if t['variant'] != 'baseline']
        if base is None:
            continue
        best_adv = max((t['advantage'] for t in varts), default=0.0)
        n_pos    = sum(1 for t in varts if t['better_than_baseline'])
        print(f'  ── Individual {uid[:8]} ── baseline MQE={base["final_mqe"]:.4f}  '
              f'best adv={best_adv:+.4f}  better: {n_pos}/{len(varts)}')

    n_better    = sum(1 for t in trajectories if t['better_than_baseline'])
    n_perturbed = sum(1 for t in trajectories if t['variant'] != 'baseline')
    print(f'  → {len(trajectories)} trajectories, '
          f'{n_better}/{n_perturbed} perturbed better than baseline '
          f'({n_better/max(1,n_perturbed)*100:.0f}%)')

    if _return_trajectories:
        return trajectories

    # Single-seed direct write
    class _NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.bool_): return bool(o)
            return super().default(o)

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'trajectories.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(trajectories, f, cls=_NpEncoder)
    print(f'\nSaved {len(trajectories)} trajectories → {out_path}')
    print(f'Total checkpoints: {sum(len(t["checkpoints"]) for t in trajectories):,}')


def _find_seed_dirs(results_root: str):
    """
    Scan results_root for all seed_* directories and their dataset CSV paths.
    Expected structure: <results_root>/<DS_NAME>/results/<TIMESTAMP>/seed_<N>/
    Dataset CSV expected at: <results_root>/<DS_NAME>/dataset.csv
    Returns list of (seed_dir_path, dataset_csv_path).
    """
    root = Path(results_root)
    pairs = []
    for ds_dir in sorted(root.iterdir()):
        if not ds_dir.is_dir():
            continue
        dataset_csv = ds_dir / 'dataset.csv'
        if not dataset_csv.exists():
            print(f'  SKIP {ds_dir.name}: dataset.csv not found')
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
                        pairs.append((seed_dir, dataset_csv))
    return pairs


def main():
    parser = argparse.ArgumentParser(description='Generate Phase 3 LSTM perturbation data')
    parser.add_argument('--seed_dir',     default=None,
                        help='Path to one EA seed_* directory (use with --dataset)')
    parser.add_argument('--dataset',      default=None,
                        help='Path to original dataset CSV (required with --seed_dir)')
    parser.add_argument('--results_root', default=None,
                        help='Root of datasets dir — auto-discovers all seed dirs and dataset CSVs')
    parser.add_argument('--n_pareto',     type=int, default=5,
                        help='Number of Pareto individuals per seed (default: 5)')
    parser.add_argument('--n_variants',   type=int, default=8,
                        help='Perturbed variants per individual (default: 8)')
    parser.add_argument('--output',       default='app/lstm/data/phase3',
                        help='Output directory (default: app/lstm/data/phase3)')
    parser.add_argument('--n_workers',    type=int, default=0,
                        help='Parallel workers (default: cpu_count-1)')
    args = parser.parse_args()

    if args.results_root is None and args.seed_dir is None:
        parser.error('Provide either --results_root or --seed_dir + --dataset')
    if args.seed_dir and args.dataset is None:
        parser.error('--seed_dir requires --dataset')

    print('=' * 60)
    print('PHASE 3 DATA GENERATION')
    print('=' * 60)

    if args.results_root:
        pairs = _find_seed_dirs(args.results_root)
        if not pairs:
            print(f'ERROR: No seed directories found under {args.results_root}')
            sys.exit(1)
        print(f'Results root: {args.results_root}')
        print(f'Seed dirs found: {len(pairs)}')
        for sd, ds in pairs:
            print(f'  {sd.relative_to(Path(args.results_root).parent)}  ←  {ds.name}')
    else:
        pairs = [(Path(args.seed_dir), Path(args.dataset))]
        print(f'Seed dir: {args.seed_dir}')
        print(f'Dataset:  {args.dataset}')

    workers_label = args.n_workers if args.n_workers > 0 else f'auto ({max(1, cpu_count()-1)})'
    print(f'Pareto:     {args.n_pareto} individuals × {args.n_variants} variants')
    print(f'Runs/seed:  {args.n_pareto * (1 + args.n_variants)}')
    print(f'Total runs: {len(pairs) * args.n_pareto * (1 + args.n_variants)}')
    print(f'Workers:    {workers_label}')
    print(f'Output:     {args.output}')
    print()

    # Accumulate trajectories from all seeds, then write once
    os.makedirs(args.output, exist_ok=True)
    all_trajectories = []

    for seed_dir, dataset_csv in pairs:
        print(f'\n{"─"*60}')
        print(f'Processing: {seed_dir}')
        print(f'Dataset:    {dataset_csv}')
        print(f'{"─"*60}')
        trajectories = generate(str(seed_dir), str(dataset_csv),
                                args.n_pareto, args.n_variants, args.output,
                                n_workers=args.n_workers, _return_trajectories=True)
        all_trajectories.extend(trajectories)
        print(f'Accumulated: {len(all_trajectories)} trajectories total')

    class _NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.bool_): return bool(o)
            return super().default(o)

    out_path = os.path.join(args.output, 'trajectories.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_trajectories, f, cls=_NpEncoder)

    n_better   = sum(1 for t in all_trajectories if t['better_than_baseline'])
    n_perturbed = sum(1 for t in all_trajectories if t['variant'] != 'baseline')
    total_ckpts = sum(len(t['checkpoints']) for t in all_trajectories)

    print(f'\n{"="*60}')
    print(f'DONE — {len(pairs)} seed(s) processed')
    print(f'Saved {len(all_trajectories)} trajectories → {out_path}')
    print(f'Perturbed better than baseline: {n_better}/{n_perturbed} '
          f'({n_better/max(1,n_perturbed)*100:.0f}%)')
    print(f'Total checkpoints: {total_ckpts:,}')


if __name__ == '__main__':
    main()
