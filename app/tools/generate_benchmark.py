"""
generate_benchmark.py — Generate SOM topology benchmark datasets.

Swiss Roll  — 3D data lying on a 2D manifold (spiral surface).
              Correctly trained 20×20 hex SOM unfolds it to a flat rectangle.
              Expected result: Spearman ρ → 1.0, TE ≈ 0.

Space-filling — Uniform 2D points in [0,1]².
              Correctly trained 1×N rect SOM winds through the space without crossings.
              Expected result: TE ≈ 0, no self-intersections in topology plot.

Usage:
  python app/tools/generate_benchmark.py swiss-roll
  python app/tools/generate_benchmark.py space-filling
  python app/tools/generate_benchmark.py all

  python app/tools/generate_benchmark.py swiss-roll --samples 3000 --noise 0.15
  python app/tools/generate_benchmark.py space-filling --samples 2000 --chain 80
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

DATASETS_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'datasets'
)


# ─── Swiss Roll ───────────────────────────────────────────────────────────────

def generate_swiss_roll(n_samples: int = 2000, noise: float = 0.1,
                        random_seed: int = 42) -> pd.DataFrame:
    from sklearn.datasets import make_swiss_roll
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise,
                           random_state=random_seed)
    df = pd.DataFrame(X, columns=['x', 'y', 'z'])
    df.insert(0, 'id', range(1, len(df) + 1))
    # Include the unroll parameter as ground-truth column (not used by SOM,
    # but useful for validating that the output colour-codes correctly)
    df['t'] = np.round(t, 4)
    return df


def save_swiss_roll(n_samples: int = 2000, noise: float = 0.1,
                    random_seed: int = 42):
    out_dir = os.path.join(DATASETS_DIR, 'SwissRoll')
    os.makedirs(out_dir, exist_ok=True)

    df = generate_swiss_roll(n_samples, noise, random_seed)
    csv_path = os.path.join(out_dir, 'swiss_roll.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples → {csv_path}")

    # Vesanto rule: U = 5*sqrt(n), side = sqrt(U)
    import math
    side = max(10, round(math.sqrt(5 * math.sqrt(n_samples))))
    _write_swiss_roll_config(out_dir, side, n_samples, random_seed)


def _write_swiss_roll_config(out_dir: str, side: int, n_samples: int,
                              random_seed: int):
    import json, math

    # epoch_multiplier: target ~10 000 iterations
    em = max(3.0, round(10000 / n_samples, 1))

    cfg = {
        "_comment": (
            "Swiss Roll benchmark — 3D data on 2D manifold. "
            f"Map {side}×{side} hex ({side*side} neurons for {n_samples} samples). "
            "Correctly trained SOM unfolds the roll: high Spearman rho, TE ≈ 0."
        ),
        "processing_type": "hybrid",
        "map_size": [side, side],
        "start_learning_rate": 0.9,
        "end_learning_rate": 0.01,
        "lr_decay_type": "exp-drop",
        "start_radius_init_ratio": 1.0,
        "end_radius": 1.0,
        "radius_decay_type": "linear-drop",
        "start_batch_percent": 100.0,
        "end_batch_percent": 100.0,
        "batch_growth_type": "static",
        "epoch_multiplier": em,
        "normalize_weights_flag": False,
        "growth_g": 20.0,
        "random_seed": random_seed,
        "map_type": "hex",
        "num_batches": 1,
        "max_epochs_without_improvement": 500,
        "delimiter": ",",
        "categorical_threshold_numeric": 20,
        "noise_threshold_ratio": 0.2,
        "categorical_threshold_text": 20,
        "primary_id": "id",
        "mqe_evaluations_per_run": 500,
        "save_checkpoints": True,
        "checkpoint_count": 10
    }

    path = os.path.join(out_dir, 'config-som.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print(f"Saved config → {path}  (map {side}×{side}, epoch_multiplier={em})")


# ─── Space-filling ────────────────────────────────────────────────────────────

def generate_space_filling(n_samples: int = 1000,
                           random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    xy = rng.uniform(0.0, 1.0, size=(n_samples, 2))
    df = pd.DataFrame(xy, columns=['x', 'y'])
    df.insert(0, 'id', range(1, len(df) + 1))
    return df


def save_space_filling(n_samples: int = 1000, chain_len: int = 0,
                       random_seed: int = 42):
    out_dir = os.path.join(DATASETS_DIR, 'SpaceFilling')
    os.makedirs(out_dir, exist_ok=True)

    df = generate_space_filling(n_samples, random_seed)
    csv_path = os.path.join(out_dir, 'space_filling.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples → {csv_path}")

    # Chain length: ~n_samples / 10, capped at 200
    if chain_len <= 0:
        chain_len = min(200, max(20, n_samples // 10))

    _write_space_filling_config(out_dir, chain_len, n_samples, random_seed)


def _write_space_filling_config(out_dir: str, chain_len: int, n_samples: int,
                                 random_seed: int):
    import json

    em = max(5.0, round(15000 / n_samples, 1))

    cfg = {
        "_comment": (
            f"Space-filling benchmark — uniform 2D points in [0,1]². "
            f"Map 1×{chain_len} rect (chain of {chain_len} neurons). "
            "Correctly trained SOM winds through the square without crossings: TE ≈ 0."
        ),
        "processing_type": "hybrid",
        "map_size": [1, chain_len],
        "start_learning_rate": 0.9,
        "end_learning_rate": 0.01,
        "lr_decay_type": "exp-drop",
        "start_radius_init_ratio": 1.0,
        "end_radius": 1.0,
        "radius_decay_type": "linear-drop",
        "start_batch_percent": 100.0,
        "end_batch_percent": 100.0,
        "batch_growth_type": "static",
        "epoch_multiplier": em,
        "normalize_weights_flag": False,
        "growth_g": 20.0,
        "random_seed": random_seed,
        "map_type": "rect",
        "num_batches": 1,
        "max_epochs_without_improvement": 500,
        "delimiter": ",",
        "categorical_threshold_numeric": 20,
        "noise_threshold_ratio": 0.2,
        "categorical_threshold_text": 20,
        "primary_id": "id",
        "mqe_evaluations_per_run": 500,
        "save_checkpoints": True,
        "checkpoint_count": 10
    }

    path = os.path.join(out_dir, 'config-som.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=4, ensure_ascii=False)
    print(f"Saved config → {path}  (map 1×{chain_len}, epoch_multiplier={em})")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate SOM topology benchmark datasets.")
    parser.add_argument('benchmark',
                        choices=['swiss-roll', 'space-filling', 'all'],
                        help='Which benchmark to generate')
    parser.add_argument('--samples', type=int, default=0,
                        help='Number of data points (default: 2000 / 1000)')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Swiss Roll noise std (default: 0.1)')
    parser.add_argument('--chain', type=int, default=0,
                        help='Space-filling chain length (default: auto)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    do_swiss = args.benchmark in ('swiss-roll', 'all')
    do_space = args.benchmark in ('space-filling', 'all')

    if do_swiss:
        n = args.samples if args.samples > 0 else 2000
        print(f"\n── Swiss Roll  ({n} samples, noise={args.noise}) ──")
        save_swiss_roll(n_samples=n, noise=args.noise, random_seed=args.seed)

    if do_space:
        n = args.samples if args.samples > 0 else 1000
        print(f"\n── Space-filling  ({n} samples) ──")
        save_space_filling(n_samples=n, chain_len=args.chain,
                           random_seed=args.seed)


if __name__ == '__main__':
    main()
