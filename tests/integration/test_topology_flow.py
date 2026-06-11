"""
End-to-end verification of the topology-validation flow:
generate benchmark dataset → SOM organization → quantitative ground-truth
verification → topology rendering.

This is the "no computational error in the whole chain" test: a clean 2D
plane MUST organize into a coherent map (local + global PASS) and all
rendering paths must produce their outputs.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'app'))
sys.path.insert(0, str(REPO_ROOT / 'app' / 'tools'))

from som.run import run_pipeline  # noqa: E402
from verify_topology import verify  # noqa: E402


@pytest.fixture(scope='module')
def organized_plane(tmp_path_factory):
    """Generate a 2D plane dataset with ground truth and organize it."""
    tmp = tmp_path_factory.mktemp('plane')
    rng = np.random.default_rng(42)
    n = 300
    u, v = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    pd.DataFrame({'id': range(1, n + 1), 'x': u, 'y': v}).to_csv(
        tmp / 'plane.csv', index=False)
    pd.DataFrame({'id': range(1, n + 1), 'u': u, 'v': v}).to_csv(
        tmp / 'plane_groundtruth.csv', index=False)

    config = {
        'map_size': [7, 7],
        'map_type': 'hex',
        'random_seed': 42,
        'epoch_multiplier': 10.0,
        'start_learning_rate': 0.9,
        'end_learning_rate': 0.05,
        'lr_decay_type': 'linear-drop',
        'start_radius_init_ratio': 1.0,
        'end_radius': 1.0,
        'radius_decay_type': 'linear-drop',
        'start_batch_percent': 2.0,
        'end_batch_percent': 2.0,
        'batch_growth_type': 'static',
        'num_batches': 1,
        'normalize_weights_flag': False,
        'growth_g': 1.0,
        'max_epochs_without_improvement': 500,
        'mqe_evaluations_per_run': 20,
        'primary_id': 'id',
        'categorical_threshold_numeric': 5,
        'categorical_threshold_text': 5,
        'show_progress': False,
        'save_training_plots': False,
        'save_visualizations': False,
    }
    results_dir = run_pipeline(str(tmp / 'plane.csv'), config,
                               output_dir=str(tmp / 'results' / 'run1'))
    return tmp, results_dir


class TestQuantitativeVerification:
    def test_clean_plane_passes_local_and_global(self, organized_plane):
        """A clean 2D plane must organize correctly — the canary for
        computational errors anywhere in the chain."""
        _, results_dir = organized_plane
        report = verify(results_dir)
        assert report['verdict'] == 'PASS', json.dumps(report, indent=2)

        for param in ('u', 'v'):
            metrics = report['manifold_params'][param]
            assert metrics['grid_param_R2'] > 0.85, (param, metrics)
            assert metrics['neuron_anova_R2'] > 0.85, (param, metrics)
        assert report['global_structure']['pairwise_distance_spearman'] > 0.75

    def test_groundtruth_autodiscovery(self, organized_plane):
        tmp, results_dir = organized_plane
        report = verify(results_dir)  # no explicit -g
        assert report['groundtruth'] == str(tmp / 'plane_groundtruth.csv')
        assert report['n_samples_matched'] == 300

    def test_scrambled_assignments_fail(self, organized_plane, tmp_path):
        """Sanity: the metric must FAIL when the organization is destroyed."""
        import shutil
        _, results_dir = organized_plane
        broken = tmp_path / 'results' / 'broken'
        broken.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(results_dir, broken)
        sa_path = broken / 'csv' / 'sample_assignments.csv'
        sa = pd.read_csv(sa_path)
        rng = np.random.default_rng(0)
        perm = rng.permutation(len(sa))
        sa[['bmu_i', 'bmu_j', 'bmu_key']] = sa[['bmu_i', 'bmu_j', 'bmu_key']].iloc[perm].values
        sa.to_csv(sa_path, index=False)
        report = verify(str(broken),
                        groundtruth_path=str(organized_plane[0] / 'plane_groundtruth.csv'))
        assert report['verdict'] == 'FAIL'


class TestRenderingFlow:
    @pytest.mark.parametrize('args,expected', [
        (['--projection', 'raw'], 'topology_2d_raw.png'),
        (['--projection', 'pca'], 'topology_2d_pca.png'),
        (['--projection', 'pca', '--html'], 'topology_interactive_pca.html'),
    ])
    def test_render_outputs(self, organized_plane, args, expected):
        _, results_dir = organized_plane
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / 'app' / 'tools' / 'plot_som_topology.py'),
             results_dir] + args,
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=180)
        assert proc.returncode == 0, proc.stderr
        assert (Path(results_dir) / expected).is_file(), expected

    def test_raw_html3d_rejects_wrong_dimensionality(self, organized_plane):
        """raw 3D interactive on a 2D dataset must fail loudly,
        not silently mis-render."""
        _, results_dir = organized_plane
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / 'app' / 'tools' / 'plot_som_topology.py'),
             results_dir, '--projection', 'raw', '--html3d'],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=120)
        assert proc.returncode != 0
        assert 'raw' in (proc.stdout + proc.stderr)
        assert not (Path(results_dir) / 'topology_interactive_3d_raw.html').exists()
