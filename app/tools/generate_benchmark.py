"""
generate_benchmark.py — Generate SOM topology benchmark datasets with ground truth.

Each benchmark targets one specific property of a correct SOM implementation;
together they form the verification battery for the ablation study (A3/B4 in
docs/global/ABLATION_STUDY.md). Quantitative evaluation against the ground
truth is done by app/tools/verify_topology.py.

| Benchmark     | Structure                          | Verifies                                   |
|---------------|------------------------------------|--------------------------------------------|
| swiss-roll    | 2D manifold rolled in 3D           | manifold unfolding (hard: layered)         |
| s-curve       | 2D manifold bent in 3D             | manifold unfolding (easier: no layers)     |
| helix         | 1D manifold coiled in 3D           | 1D chain ordering along a curve            |
| torus         | closed 2D manifold in 3D           | behavior on closed surfaces (documented imperfection: a planar sheet cannot wrap a torus) |
| blobs         | 5 Gaussian clusters in 5D          | cluster separation (labels = ground truth) |
| noisy-plane   | 2D plane + pure-noise dimensions   | robustness to noise dimensions             |
| uniform-cube  | no structure at all                | negative control — the system must not invent clusters |
| space-filling | uniform 2D, 1×N chain              | space-filling without self-crossings       |

Usage:
  python app/tools/generate_benchmark.py swiss-roll
  python app/tools/generate_benchmark.py all
  python app/tools/generate_benchmark.py blobs --samples 3000 --seed 7
  python app/tools/generate_benchmark.py swiss-roll --samples 3000 --noise 0.15
  python app/tools/generate_benchmark.py space-filling --samples 2000 --chain 80
"""

import argparse
import json
import math
import os

import numpy as np
import pandas as pd

DATASETS_DIR = os.path.join(
    os.path.dirname(__file__), '..', '..', 'data', 'datasets'
)


# ─── Shared helpers ───────────────────────────────────────────────────────────

def _vesanto_side(n_samples: int) -> int:
    return max(10, round(math.sqrt(5 * math.sqrt(n_samples))))


def _write_config(out_dir: str, comment: str, map_size: list, map_type: str,
                  n_samples: int, updates_target: int, random_seed: int):
    em = max(3.0, round(updates_target / n_samples, 1))
    cfg = {
        "_comment": comment + f" Map {map_size[0]}x{map_size[1]} {map_type}, "
                              f"epoch_multiplier={em} -> {int(em * n_samples)} weight updates.",
        "map_size": map_size,
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
        "map_type": map_type,
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
    print(f"Saved config → {path}")


def _save(name: str, dir_name: str, df: pd.DataFrame, gt: pd.DataFrame | None,
          comment: str, map_size: list, map_type: str, updates_target: int,
          random_seed: int):
    out_dir = os.path.join(DATASETS_DIR, dir_name)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f'{name}.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} samples → {csv_path}  (columns: {list(df.columns)})")

    if gt is not None:
        gt_path = os.path.join(out_dir, f'{name}_groundtruth.csv')
        gt.to_csv(gt_path, index=False)
        print(f"Saved ground truth → {gt_path}  (NOT for training)")

    _write_config(out_dir, comment, map_size, map_type, len(df),
                  updates_target, random_seed)


def _with_id(X: np.ndarray, columns: list) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=columns)
    df.insert(0, 'id', range(1, len(df) + 1))
    return df


def _gt(columns: dict) -> pd.DataFrame:
    n = len(next(iter(columns.values())))
    gt = pd.DataFrame({'id': range(1, n + 1)})
    for name, values in columns.items():
        gt[name] = np.round(values, 4)
    return gt


# ─── Benchmarks ───────────────────────────────────────────────────────────────

def save_swiss_roll(n_samples: int = 2000, noise: float = 0.1, random_seed: int = 42):
    from sklearn.datasets import make_swiss_roll
    X, t = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=random_seed)
    # Manifold parameters: t (unrolling angle) and X[:,1] (height along the roll).
    # They must NOT enter training — kept in the ground-truth file only.
    side = _vesanto_side(n_samples)
    _save('swiss_roll', 'SwissRoll', _with_id(X, ['x', 'y', 'z']),
          _gt({'t': t, 'height': X[:, 1]}),
          "Swiss Roll benchmark — 2D manifold rolled in 3D. Correct SOM unfolds it: "
          "high grid<->t correlation (verify_topology.py), TE alone is NOT sufficient "
          "(see docs/som/issues.md #23).",
          [side, side], 'hex', 10000, random_seed)


def save_s_curve(n_samples: int = 2000, noise: float = 0.1, random_seed: int = 42):
    from sklearn.datasets import make_s_curve
    X, t = make_s_curve(n_samples=n_samples, noise=noise, random_state=random_seed)
    side = _vesanto_side(n_samples)
    _save('s_curve', 'SCurve', _with_id(X, ['x', 'y', 'z']),
          _gt({'t': t, 'height': X[:, 1]}),
          "S-Curve benchmark — 2D manifold bent in 3D (no overlapping layers, easier "
          "than Swiss Roll). Correct SOM flattens it: high grid<->(t,height) adherence.",
          [side, side], 'hex', 10000, random_seed)


def save_helix(n_samples: int = 2000, noise: float = 0.05, random_seed: int = 42,
               chain_len: int = 0):
    rng = np.random.default_rng(random_seed)
    t = np.sort(rng.uniform(0, 6 * np.pi, n_samples))
    X = np.stack([np.cos(t), np.sin(t), t / (6 * np.pi) * 3.0], axis=1)
    X += rng.normal(0, noise, X.shape)
    if chain_len <= 0:
        chain_len = min(200, max(30, n_samples // 20))
    _save('helix', 'Helix', _with_id(X, ['x', 'y', 'z']), _gt({'t': t}),
          "Helix benchmark — 1D manifold coiled in 3D. A 1xN chain SOM must order "
          "itself monotonically along t (grid position <-> t correlation ~ 1).",
          [1, chain_len], 'rect', 15000, random_seed)


def save_torus(n_samples: int = 2000, noise: float = 0.05, random_seed: int = 42):
    rng = np.random.default_rng(random_seed)
    u = rng.uniform(0, 2 * np.pi, n_samples)   # around the tube center
    v = rng.uniform(0, 2 * np.pi, n_samples)   # around the tube itself
    R, r = 3.0, 1.0
    X = np.stack([(R + r * np.cos(v)) * np.cos(u),
                  (R + r * np.cos(v)) * np.sin(u),
                  r * np.sin(v)], axis=1)
    X += rng.normal(0, noise, X.shape)
    side = _vesanto_side(n_samples)
    _save('torus', 'Torus', _with_id(X, ['x', 'y', 'z']),
          _gt({'u': u, 'v': v}),
          "Torus benchmark — CLOSED 2D manifold. A planar SOM sheet cannot wrap a "
          "torus perfectly: expect a documented seam (boundary TE), not a math error. "
          "Hard case for the ablation discussion.",
          [side, side], 'hex', 10000, random_seed)


def save_blobs(n_samples: int = 2000, centers: int = 5, n_features: int = 5,
               cluster_std: float = 1.0, random_seed: int = 42):
    from sklearn.datasets import make_blobs
    X, labels = make_blobs(n_samples=n_samples, centers=centers,
                           n_features=n_features, cluster_std=cluster_std,
                           random_state=random_seed)
    side = _vesanto_side(n_samples)
    df = _with_id(X, [f'f{i}' for i in range(n_features)])
    gt = pd.DataFrame({'id': range(1, n_samples + 1), 'label': labels})
    _save('blobs', 'Blobs', df, gt,
          f"Gaussian blobs benchmark — {centers} clusters in {n_features}D. Correct SOM "
          "separates them into disjoint neuron regions: high ARI(label, neuron), "
          "high purity, coherent regions in compute_regions.",
          [side, side], 'hex', 10000, random_seed)


def save_noisy_plane(n_samples: int = 2000, noise_dims: int = 4,
                     random_seed: int = 42):
    rng = np.random.default_rng(random_seed)
    u = rng.uniform(0, 1, n_samples)
    v = rng.uniform(0, 1, n_samples)
    signal = np.stack([u, v], axis=1)
    noise = rng.uniform(0, 1, (n_samples, noise_dims))
    X = np.hstack([signal, noise])
    side = _vesanto_side(n_samples)
    cols = ['u_signal', 'v_signal'] + [f'noise_{i}' for i in range(noise_dims)]
    _save('noisy_plane', 'NoisyPlane', _with_id(X, cols), _gt({'u': u, 'v': v}),
          f"Noisy plane benchmark — 2D structure + {noise_dims} pure-noise numeric "
          "dimensions (numeric noise is NOT excluded by preprocessing — only text "
          "noise is). Measures how noise dims degrade grid<->(u,v) adherence; input "
          "for the preprocessing ablation.",
          [side, side], 'hex', 10000, random_seed)


def save_uniform_cube(n_samples: int = 2000, n_dims: int = 5, random_seed: int = 42):
    rng = np.random.default_rng(random_seed)
    X = rng.uniform(0, 1, (n_samples, n_dims))
    side = _vesanto_side(n_samples)
    _save('uniform_cube', 'UniformCube',
          _with_id(X, [f'f{i}' for i in range(n_dims)]), None,
          f"Uniform cube — NO structure ({n_dims}D uniform noise). Negative control: "
          "the system must report low cluster coherence and must not invent clusters "
          "(uniform hit map, low density Gini). No ground-truth file by design.",
          [side, side], 'hex', 10000, random_seed)


def save_space_filling(n_samples: int = 1000, chain_len: int = 0, random_seed: int = 42):
    rng = np.random.default_rng(random_seed)
    xy = rng.uniform(0.0, 1.0, size=(n_samples, 2))
    if chain_len <= 0:
        chain_len = min(200, max(20, n_samples // 10))
    _save('space_filling', 'SpaceFilling', _with_id(xy, ['x', 'y']), None,
          f"Space-filling benchmark — uniform 2D points, 1x{chain_len} rect chain. "
          "Correct SOM winds through the square without self-crossings: TE ~ 0.",
          [1, chain_len], 'rect', 15000, random_seed)


BENCHMARKS = {
    'swiss-roll':   lambda a: save_swiss_roll(a.samples or 2000, a.noise, a.seed),
    's-curve':      lambda a: save_s_curve(a.samples or 2000, a.noise, a.seed),
    'helix':        lambda a: save_helix(a.samples or 2000, min(a.noise, 0.05), a.seed, a.chain),
    'torus':        lambda a: save_torus(a.samples or 2000, min(a.noise, 0.05), a.seed),
    'blobs':        lambda a: save_blobs(a.samples or 2000, random_seed=a.seed),
    'noisy-plane':  lambda a: save_noisy_plane(a.samples or 2000, random_seed=a.seed),
    'uniform-cube': lambda a: save_uniform_cube(a.samples or 2000, random_seed=a.seed),
    'space-filling': lambda a: save_space_filling(a.samples or 1000, a.chain, a.seed),
}


def main():
    parser = argparse.ArgumentParser(
        description="Generate SOM topology benchmark datasets with ground truth.")
    parser.add_argument('benchmark', choices=list(BENCHMARKS) + ['all'],
                        help='Which benchmark to generate')
    parser.add_argument('--samples', type=int, default=0,
                        help='Number of data points (default per benchmark)')
    parser.add_argument('--noise', type=float, default=0.1,
                        help='Noise std where applicable (default: 0.1)')
    parser.add_argument('--chain', type=int, default=0,
                        help='Chain length for helix/space-filling (default: auto)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    targets = list(BENCHMARKS) if args.benchmark == 'all' else [args.benchmark]
    for name in targets:
        print(f"\n── {name} ──")
        BENCHMARKS[name](args)


if __name__ == '__main__':
    main()
