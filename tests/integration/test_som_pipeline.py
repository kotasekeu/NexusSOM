"""
End-to-end baseline test of the full SOM pipeline via the CLI
(app/run_som.py): preprocess → train → analysis → llm_context → plots → maps.

This is the primary characterization test for the SOM module refactor:
if the restructuring changes the run contract or output layout, this fails.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def pipeline_run(tmp_path_factory):
    """Run the full pipeline once on a tiny synthetic dataset."""
    tmp = tmp_path_factory.mktemp('som_e2e')
    rng = np.random.default_rng(42)
    n = 40
    df = pd.DataFrame({
        'ID': range(1, n + 1),
        'feat_a': np.concatenate([rng.normal(2, 0.3, n // 2), rng.normal(8, 0.3, n // 2)]),
        'feat_b': np.concatenate([rng.normal(5, 0.5, n // 2), rng.normal(1, 0.2, n // 2)]),
        'label': ['low'] * (n // 2) + ['high'] * (n // 2),
    })
    input_csv = tmp / 'data.csv'
    df.to_csv(input_csv, index=False)

    config = {
        'map_size': [6, 6],
        'start_learning_rate': 0.9,
        'end_learning_rate': 0.1,
        'lr_decay_type': 'linear-drop',
        'start_radius_init_ratio': 1.0,
        'end_radius': 1.0,
        'radius_decay_type': 'linear-drop',
        'start_batch_percent': 10.0,
        'end_batch_percent': 10.0,
        'batch_growth_type': 'static',
        'epoch_multiplier': 2.0,
        'normalize_weights_flag': False,
        'growth_g': 1.0,
        'random_seed': 42,
        'map_type': 'hex',
        'num_batches': 2,
        'max_epochs_without_improvement': 500,
        'mqe_evaluations_per_run': 10,
        'delimiter': ',',
        'categorical_threshold_numeric': 10,
        'categorical_threshold_text': 10,
        'noise_threshold_ratio': 0.2,
        'primary_id': 'ID',
        'save_checkpoints': True,
        'checkpoint_count': 5,
    }
    config_path = tmp / 'config.json'
    config_path.write_text(json.dumps(config))

    out_dir = tmp / 'results'
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / 'app' / 'run_som.py'),
         '-i', str(input_csv), '-c', str(config_path), '-o', str(out_dir)],
        capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300,
    )
    return proc, out_dir


class TestPipelineExitAndLayout:
    def test_exit_code_zero(self, pipeline_run):
        proc, _ = pipeline_run
        assert proc.returncode == 0, f'stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}'

    def test_output_layout(self, pipeline_run):
        _, out = pipeline_run
        expected = [
            'log.txt',
            'run_metrics.json',
            'dataset_meta.json',
            'csv/original_input.csv',
            'csv/training_data.npy',
            'csv/ignore_mask.csv',
            'csv/weights.npy',
            'csv/weights_readable.csv',
            'csv/sample_assignments.csv',
            'csv/training_checkpoints.json',
            'json/preprocessing_info.json',
            'json/clusters.json',
            'json/quantization_errors.json',
            'json/extremes.json',
            'json/llm_context.json',
            'visualizations/u_matrix.png',
            'visualizations/hit_map.png',
            'visualizations/distance_map.png',
            'visualizations/dead_neurons_map.png',
            'visualizations/mqe_evolution.png',
            'visualizations/learning_rate_decay.png',
            'visualizations/radius_decay.png',
            'visualizations/batch_size_growth.png',
            'visualizations/pie_map_label.png',
            'visualizations/cluster_map.png',
        ]
        missing = [rel for rel in expected if not (out / rel).is_file()]
        assert not missing, f'Missing outputs: {missing}'


class TestPipelineContent:
    def test_run_metrics_contract(self, pipeline_run):
        _, out = pipeline_run
        rm = json.loads((out / 'run_metrics.json').read_text())
        assert rm['map_size'] == [6, 6]
        assert rm['map_topology'] == 'hex'
        assert rm['best_mqe'] > 0
        assert rm['topographic_error'] is not None
        assert rm['duration'] > 0

    def test_weights_shape(self, pipeline_run):
        _, out = pipeline_run
        w = np.load(out / 'csv/weights.npy')
        # 4 training dims: ID + feat_a + feat_b + label
        assert w.shape == (6, 6, 4)

    def test_clusters_cover_all_samples(self, pipeline_run):
        _, out = pipeline_run
        clusters = json.loads((out / 'json/clusters.json').read_text())
        all_ids = sorted(sid for ids in clusters.values() for sid in ids)
        assert all_ids == list(range(1, 41))

    def test_two_cluster_structure_separated(self, pipeline_run):
        """The two synthetic clusters must map to disjoint neuron groups."""
        _, out = pipeline_run
        sa = pd.read_csv(out / 'csv/sample_assignments.csv')
        low = set(sa[sa['sample_id'] <= 20]['bmu_key'])
        high = set(sa[sa['sample_id'] > 20]['bmu_key'])
        assert not (low & high), 'clusters low/high share neurons — map not organized'

    def test_llm_context_contract(self, pipeline_run):
        _, out = pipeline_run
        ctx = json.loads((out / 'json/llm_context.json').read_text())
        for key in ('map', 'clusters', 'anomalies', 'dimension_stats'):
            assert key in ctx, key
        assert ctx['map']['size'] == [6, 6]
        # spatial analysis is wired into the context (CNN replacement)
        assert 'spatial_analysis' in ctx
        assert 0.0 <= ctx['map']['spatial_quality_score'] <= 1.0

    def test_sample_assignments_per_dim_qe(self, pipeline_run):
        _, out = pipeline_run
        sa = pd.read_csv(out / 'csv/sample_assignments.csv')
        assert 'qe_dim_feat_a' in sa.columns
        assert 'qe_dim_feat_b' in sa.columns
        assert len(sa) == 40
