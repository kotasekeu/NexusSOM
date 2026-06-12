"""
End-to-end EA smoke test — full run_ea.py subprocess on a tiny dataset.

Slow (tens of seconds): skipped by default, enable with

    EA_SMOKE=1 .venv/bin/python3 -m pytest tests/integration/test_ea_smoke.py

This is the automated form of the manual verification used for issues.md #87
(elitism fix) and the per-phase gate of docs/ea/CLEANUP_PLAN.md.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

POPULATION = 4
GENERATIONS = 2

CONFIG = {
    "use_nn": False,
    "EA_SETTINGS": {"population_size": POPULATION, "generations": GENERATIONS,
                    "seeds": [42]},
    "CALIBRATION": {"n_probes": 0},
    "SEARCH_SPACE": {
        "map_size": {"type": "discrete_int_pair"},
        "start_learning_rate": {"type": "float", "min": 0.5, "max": 1.0,
                                "log_scale": True},
        "end_learning_rate": {"type": "float", "min": 0.001, "max": 0.5,
                              "log_scale": True},
        "lr_decay_type": {"type": "categorical",
                          "values": ["linear-drop", "exp-drop"]},
        "start_radius_init_ratio": {"type": "float", "min": 0.5, "max": 1.0,
                                    "log_scale": True},
        "radius_decay_type": {"type": "categorical",
                              "values": ["linear-drop", "exp-drop"]},
        "start_batch_percent": {"type": "float", "min": 0.0, "max": 5.0},
        "end_batch_percent": {"type": "float", "min": 1.0, "max": 15.0},
        "batch_growth_type": {"type": "categorical",
                              "values": ["linear-growth", "exp-growth"]},
        "epoch_multiplier": {"type": "float"},
        "growth_g": {"type": "int", "min": 10, "max": 40},
        "num_batches": {"type": "int", "min": 1, "max": 5},
    },
    "GENETIC_OPERATORS": {"sbx_eta": 20.0, "mutation_eta": 20.0,
                          "mutation_prob": 0.1, "tournament_k": 2},
    "FIXED_PARAMS": {
        "end_radius": 1.0, "random_seed": 42, "max_archive_size": 10,
        "mqe_evaluations_per_run": 20, "map_type": "hex",
        "save_checkpoints": True, "checkpoint_count": 10,
        "early_stopping_window": 50, "max_epochs_without_improvement": 500,
        "normalize_weights_flag": False,
        "generate_training_plots": False, "generate_individual_maps": False,
    },
    "NEURAL_NETWORKS": {"use_mlp": False, "use_lstm": False, "use_cnn": False},
    "PREPROCES_DATA": {"delimiter": ",", "categorical_threshold_numeric": 30,
                       "categorical_threshold_text": 30,
                       "noise_threshold_ratio": 0.2, "primary_id": "id"},
}


@pytest.mark.skipif(not os.environ.get('EA_SMOKE'),
                    reason="slow EA end-to-end run — set EA_SMOKE=1 to enable")
def test_ea_smoke_run(tmp_path):
    # Tiny structured dataset: 3 Gaussian blobs in 3D + id column
    rng = np.random.default_rng(42)
    centers = np.array([[0.2, 0.2, 0.2], [0.8, 0.8, 0.2], [0.5, 0.2, 0.8]])
    points = np.vstack([rng.normal(c, 0.05, size=(30, 3)) for c in centers])
    df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    df.insert(0, 'id', range(1, len(df) + 1))
    csv_path = tmp_path / 'blobs.csv'
    df.to_csv(csv_path, index=False)

    config_path = tmp_path / 'config-ea.json'
    config_path.write_text(json.dumps(CONFIG))

    proc = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, 'app', 'run_ea.py'),
         '-i', str(csv_path), '-c', str(config_path)],
        capture_output=True, text=True, timeout=600, cwd=REPO_ROOT,
    )
    assert proc.returncode == 0, f"EA run failed:\n{proc.stdout}\n{proc.stderr}"

    results_root = tmp_path / 'results'
    run_dirs = list(results_root.iterdir())
    assert len(run_dirs) == 1
    seed_dir = run_dirs[0] / 'seed_42'

    for name in ('results.csv', 'status.csv', 'pareto_front.csv',
                 'pareto_metrics.csv'):
        assert (seed_dir / name).is_file(), f"missing {name}"

    results = pd.read_csv(seed_dir / 'results.csv')
    status = pd.read_csv(seed_dir / 'status.csv')
    front = pd.read_csv(seed_dir / 'pareto_front.csv')

    # All three raw objectives must be recorded per evaluation (F17: the
    # topological-correlation objective was missing from results.csv)
    assert 'raw_topological_correlation' in results.columns
    assert np.isfinite(results['raw_topological_correlation']).all()

    # Every generation evaluated 1..POPULATION individuals, none failed
    completed = status[status.status == 'completed']
    assert set(completed.generation) == set(range(GENERATIONS))
    per_gen = completed.groupby('generation').size()
    assert (per_gen >= 1).all() and (per_gen <= POPULATION).all()
    assert 'failed' not in set(status.status)
    assert len(results) == len(completed)

    # Archive consistency: snapshots per generation, members were evaluated,
    # objectives present and finite
    assert set(front.generation) <= set(range(1, GENERATIONS + 1))
    assert front.generation.max() == GENERATIONS
    assert set(front.uid) <= set(results.uid)
    last = front[front.generation == GENERATIONS]
    assert len(last) >= 1
    for col in ('raw_mqe_ratio', 'raw_te', 'raw_topo_corr',
                'constraint_violation'):
        assert col in front.columns
        assert np.isfinite(last[col]).all()

    # Phase 2 gate (CLEANUP_PLAN.md, F8): gene-only UID dedup means no
    # configuration may be evaluated twice within a run.
    assert results.uid.is_unique, "duplicate evaluation — offspring dedup broken"
