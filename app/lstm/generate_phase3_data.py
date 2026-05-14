#!/usr/bin/env python3
"""
Generate Phase 3 LSTM Training Data — Perturbation Trajectories

For each Pareto individual from an EA run:
  - Run baseline (clean, no perturbation)
  - Run N_VARIANTS with random lr/radius perturbation at each checkpoint
  - Record checkpoint sequence + applied factors + delta_final_mqe vs baseline

Output: app/lstm/data/phase3/trajectories.json

Usage:
    python3 app/lstm/generate_phase3_data.py \\
        --seed_dir data/datasets/LungCancerDataset/results/20260513_181811/seed_42 \\
        --dataset  data/datasets/LungCancerDataset/dataset.csv \\
        --n_pareto 5 --n_variants 8 --output app/lstm/data/phase3
"""

import argparse
import json
import os
import sys
import time
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

def make_random_perturb_fn(rng, lr_perturb=LR_PERTURB, radius_perturb=RADIUS_PERTURB,
                           prob=PERTURB_PROB):
    """Returns a dynamic_schedule_fn with random perturbations at each checkpoint."""
    def fn(checkpoint):
        lr_f = rng.uniform(1 - lr_perturb, 1 + lr_perturb) if rng.random() < prob else 1.0
        rad_f = rng.uniform(1 - radius_perturb, 1 + radius_perturb) if rng.random() < prob else 1.0
        return float(lr_f), float(rad_f)
    return fn


def make_lr_only_fn(rng, lr_perturb=LR_PERTURB, prob=PERTURB_PROB):
    def fn(checkpoint):
        lr_f = rng.uniform(1 - lr_perturb, 1 + lr_perturb) if rng.random() < prob else 1.0
        return float(lr_f), 1.0
    return fn


def make_radius_only_fn(rng, radius_perturb=RADIUS_PERTURB, prob=PERTURB_PROB):
    def fn(checkpoint):
        rad_f = rng.uniform(1 - radius_perturb, 1 + radius_perturb) if rng.random() < prob else 1.0
        return 1.0, float(rad_f)
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


# ── main pipeline ─────────────────────────────────────────────────────────────

def generate(seed_dir: str, dataset_path: str, n_pareto: int, n_variants: int,
             output_dir: str):
    seed_dir = Path(seed_dir)
    results_csv = seed_dir / 'results.csv'
    pareto_csv  = seed_dir / 'pareto_front.csv'

    if not results_csv.exists():
        print(f'ERROR: results.csv not found in {seed_dir}')
        sys.exit(1)

    results = pd.read_csv(results_csv)
    pareto  = pd.read_csv(pareto_csv) if pareto_csv.exists() else results.head(n_pareto)

    # Pick top N Pareto individuals (non-penalized)
    valid   = results[~results.get('is_penalized', pd.Series(False, index=results.index))]
    pareto_uids = set(pareto['uid'].values)
    pareto_rows = valid[valid['uid'].isin(pareto_uids)].head(n_pareto)

    if len(pareto_rows) == 0:
        print('ERROR: No valid Pareto individuals found.')
        sys.exit(1)

    print(f'Found {len(pareto_rows)} Pareto individuals (requested {n_pareto})')

    # Preprocess dataset
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

    os.makedirs(output_dir, exist_ok=True)
    trajectories = []
    rng_master   = np.random.default_rng(42)

    total_runs = len(pareto_rows) * (1 + n_variants)
    run_idx = 0

    for _, row in pareto_rows.iterrows():
        uid = row['uid']
        print(f'\n── Individual {uid[:8]} ──')

        # Variant 0: baseline (no perturbation)
        print(f'  [0/{n_variants}] baseline ...', end=' ', flush=True)
        t0 = time.time()
        baseline = run_som(row, data, ignore_mask, dynamic_fn=None, seed=42)
        print(f'MQE={baseline["final_mqe"]:.4f}  ({time.time()-t0:.1f}s)')
        run_idx += 1

        trajectories.append({
            'uid':          uid,
            'variant':      'baseline',
            'variant_seed': 42,
            'final_mqe':    baseline['final_mqe'],
            'delta_mqe':    0.0,
            'better_than_baseline': False,
            'checkpoints':  baseline['checkpoints'],
        })

        # Variants 1-N: random perturbations
        variant_seeds = rng_master.integers(0, 100_000, size=n_variants)
        for v_idx, v_seed in enumerate(variant_seeds):
            v_rng = np.random.default_rng(int(v_seed))

            # Alternate perturbation types for variety
            if v_idx % 3 == 0:
                fn = make_random_perturb_fn(v_rng)
                vtype = 'lr+radius'
            elif v_idx % 3 == 1:
                fn = make_lr_only_fn(v_rng)
                vtype = 'lr_only'
            else:
                fn = make_radius_only_fn(v_rng)
                vtype = 'radius_only'

            print(f'  [{v_idx+1}/{n_variants}] {vtype} (seed={v_seed}) ...', end=' ', flush=True)
            t0 = time.time()
            result = run_som(row, data, ignore_mask, dynamic_fn=fn,
                             seed=int(v_seed) % 10000)
            delta = baseline['final_mqe'] - result['final_mqe']  # positive = perturbed is BETTER
            print(f'MQE={result["final_mqe"]:.4f}  Δ={delta:+.4f}  ({time.time()-t0:.1f}s)')
            run_idx += 1

            trajectories.append({
                'uid':          uid,
                'variant':      vtype,
                'variant_seed': int(v_seed),
                'final_mqe':    result['final_mqe'],
                'delta_mqe':    float(delta),
                'better_than_baseline': delta > 0,
                'checkpoints':  result['checkpoints'],
            })

    # Save — custom encoder for numpy scalars
    class _NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer): return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.bool_): return bool(o)
            return super().default(o)

    out_path = os.path.join(output_dir, 'trajectories.json')
    with open(out_path, 'w') as f:
        json.dump(trajectories, f, cls=_NpEncoder)

    # Summary
    n_better = sum(1 for t in trajectories if t['better_than_baseline'])
    n_perturbed = sum(1 for t in trajectories if t['variant'] != 'baseline')
    print(f'\n{"="*60}')
    print(f'Saved {len(trajectories)} trajectories → {out_path}')
    print(f'Perturbed runs better than baseline: {n_better}/{n_perturbed} '
          f'({n_better/max(1,n_perturbed)*100:.0f}%)')
    print(f'Total checkpoints: {sum(len(t["checkpoints"]) for t in trajectories):,}')


def main():
    parser = argparse.ArgumentParser(description='Generate Phase 3 LSTM perturbation data')
    parser.add_argument('--seed_dir',  required=True,
                        help='Path to EA seed_* directory (contains results.csv + pareto_front.csv)')
    parser.add_argument('--dataset',   required=True,
                        help='Path to original dataset CSV')
    parser.add_argument('--n_pareto',  type=int, default=5,
                        help='Number of Pareto individuals to use (default: 5)')
    parser.add_argument('--n_variants', type=int, default=8,
                        help='Perturbed variants per individual (default: 8)')
    parser.add_argument('--output',    default='app/lstm/data/phase3',
                        help='Output directory (default: app/lstm/data/phase3)')
    args = parser.parse_args()

    print('=' * 60)
    print('PHASE 3 DATA GENERATION')
    print('=' * 60)
    print(f'Seed dir:   {args.seed_dir}')
    print(f'Dataset:    {args.dataset}')
    print(f'Pareto:     {args.n_pareto} individuals × {args.n_variants} variants')
    print(f'Total runs: {args.n_pareto * (1 + args.n_variants)}')
    print(f'Output:     {args.output}')
    print()

    generate(args.seed_dir, args.dataset, args.n_pareto, args.n_variants, args.output)


if __name__ == '__main__':
    main()
