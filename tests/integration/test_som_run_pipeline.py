"""
Integration tests for the programmatic pipeline API (som.run.run_pipeline)
and the multi-seed comparison tool (app/tools/multi_seed_som.py).

run_pipeline is the contract for the multi-seed tool, ablation tooling, and
the Streamlit UI — these tests call it in-process (no subprocess).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / 'app'))

from som.run import run_pipeline  # noqa: E402


def make_dataset(tmp_path) -> Path:
    rng = np.random.default_rng(42)
    n = 30
    df = pd.DataFrame({
        'ID': range(1, n + 1),
        'feat_a': np.concatenate([rng.normal(2, 0.3, n // 2), rng.normal(8, 0.3, n // 2)]),
        'feat_b': np.concatenate([rng.normal(5, 0.5, n // 2), rng.normal(1, 0.2, n // 2)]),
    })
    path = tmp_path / 'data.csv'
    df.to_csv(path, index=False)
    return path


def fast_config(**overrides) -> dict:
    config = {
        'map_size': [5, 5],
        'start_learning_rate': 0.9,
        'end_learning_rate': 0.1,
        'lr_decay_type': 'linear-drop',
        'start_radius_init_ratio': 1.0,
        'end_radius': 1.0,
        'radius_decay_type': 'linear-drop',
        'start_batch_percent': 10.0,
        'end_batch_percent': 10.0,
        'batch_growth_type': 'static',
        'epoch_multiplier': 1.0,
        'normalize_weights_flag': False,
        'growth_g': 1.0,
        'random_seed': 42,
        'map_type': 'square',
        'num_batches': 2,
        'max_epochs_without_improvement': 500,
        'mqe_evaluations_per_run': 5,
        'primary_id': 'ID',
        'categorical_threshold_numeric': 5,
        'categorical_threshold_text': 5,
        'show_progress': False,
        'save_training_plots': False,
        'save_visualizations': False,
        'save_checkpoints': True,
        'checkpoint_count': 5,
    }
    config.update(overrides)
    return config


class TestRunPipeline:
    def test_returns_results_dir_with_core_artifacts(self, tmp_path):
        input_csv = make_dataset(tmp_path)
        results_dir = run_pipeline(str(input_csv), fast_config(),
                                   output_dir=str(tmp_path / 'out'))
        for rel in ('run_metrics.json', 'csv/weights.npy',
                    'json/clusters.json', 'json/llm_context.json'):
            assert (Path(results_dir) / rel).is_file(), rel
        # visualizations disabled by config
        assert not (Path(results_dir) / 'visualizations').exists()

    def test_seed_override_does_not_mutate_config(self, tmp_path):
        input_csv = make_dataset(tmp_path)
        config = fast_config()
        run_pipeline(str(input_csv), config, output_dir=str(tmp_path / 'out'), seed=7)
        assert config['random_seed'] == 42  # caller's dict untouched
        assert 'num_samples' not in config

    def test_same_seed_reproduces_weights(self, tmp_path):
        input_csv = make_dataset(tmp_path)
        dir_a = run_pipeline(str(input_csv), fast_config(),
                             output_dir=str(tmp_path / 'a'), seed=7)
        dir_b = run_pipeline(str(input_csv), fast_config(),
                             output_dir=str(tmp_path / 'b'), seed=7)
        w_a = np.load(Path(dir_a) / 'csv' / 'weights.npy')
        w_b = np.load(Path(dir_b) / 'csv' / 'weights.npy')
        np.testing.assert_array_equal(w_a, w_b)

    def test_different_seeds_differ(self, tmp_path):
        input_csv = make_dataset(tmp_path)
        dir_a = run_pipeline(str(input_csv), fast_config(),
                             output_dir=str(tmp_path / 'a'), seed=1)
        dir_b = run_pipeline(str(input_csv), fast_config(),
                             output_dir=str(tmp_path / 'b'), seed=2)
        w_a = np.load(Path(dir_a) / 'csv' / 'weights.npy')
        w_b = np.load(Path(dir_b) / 'csv' / 'weights.npy')
        assert not np.array_equal(w_a, w_b)

    def test_config_path_accepted(self, tmp_path):
        input_csv = make_dataset(tmp_path)
        config_path = tmp_path / 'config.json'
        config_path.write_text(json.dumps(fast_config()))
        results_dir = run_pipeline(str(input_csv), str(config_path),
                                   output_dir=str(tmp_path / 'out'))
        assert (Path(results_dir) / 'run_metrics.json').is_file()

    def test_preprocess_strategy_recorded(self, tmp_path):
        input_csv = make_dataset(tmp_path)
        results_dir = run_pipeline(str(input_csv),
                                   fast_config(preprocess_strategy='scale-only'),
                                   output_dir=str(tmp_path / 'out'))
        meta = json.loads((Path(results_dir) / 'dataset_meta.json').read_text())
        assert meta['ds_preprocess_strategy'] == 'scale-only'


class TestMultiSeedTool:
    @pytest.fixture(scope='class')
    def tool_run(self, tmp_path_factory):
        tmp = tmp_path_factory.mktemp('multi_seed')
        input_csv = make_dataset(tmp)
        config_path = tmp / 'config.json'
        config_path.write_text(json.dumps(fast_config()))
        out_dir = tmp / 'comparison'
        proc = subprocess.run(
            [sys.executable, str(REPO_ROOT / 'app' / 'tools' / 'multi_seed_som.py'),
             '-i', str(input_csv), '-c', str(config_path),
             '--seeds', '1', '2', '3', '-o', str(out_dir)],
            capture_output=True, text=True, cwd=str(REPO_ROOT), timeout=300,
        )
        return proc, out_dir

    def test_exit_code_zero(self, tool_run):
        proc, _ = tool_run
        assert proc.returncode == 0, f'stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}'

    def test_outputs_exist(self, tool_run):
        _, out = tool_run
        assert (out / 'multi_seed_metrics.csv').is_file()
        assert (out / 'multi_seed_summary.json').is_file()
        assert (out / 'mqe_evolution_comparison.png').is_file()
        for seed in (1, 2, 3):
            assert (out / f'seed_{seed}' / 'run_metrics.json').is_file()

    def test_metrics_csv_has_row_per_seed(self, tool_run):
        _, out = tool_run
        df = pd.read_csv(out / 'multi_seed_metrics.csv')
        assert sorted(df['seed']) == [1, 2, 3]
        assert df['best_mqe'].notna().all()

    def test_summary_contains_stability_and_stats(self, tool_run):
        _, out = tool_run
        summary = json.loads((out / 'multi_seed_summary.json').read_text())
        metrics = summary['metrics']
        assert 'best_mqe' in metrics
        assert {'mean', 'std', 'min', 'max', 'values'} <= set(metrics['best_mqe'])
        ari = metrics['clustering_stability_ari']
        assert len(ari['pairs']) == 3  # 3 seeds → 3 pairs
        assert -1.0 <= ari['mean'] <= 1.0