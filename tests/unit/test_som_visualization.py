"""
Tests for SOM visualizations — app/som/visualization.py and graphs.py.

All map functions work from stored artifacts (weights + map_type), never from
a live KohonenSOM instance — verified here including the render_results_dir
entry point used by the UI/ablation tooling.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from som.visualization import (  # noqa: E402
    generate_u_matrix, generate_hit_map, generate_distance_map,
    generate_dead_neurons_map, generate_individual_maps, generate_all_maps,
    render_results_dir,
)
from som.graphs import generate_training_plots  # noqa: E402


@pytest.fixture
def weights_and_data():
    rng = np.random.default_rng(0)
    return rng.random((4, 4, 3)), rng.random((20, 3))


class TestIndividualMaps:
    def test_u_matrix(self, tmp_path, weights_and_data):
        weights, _ = weights_and_data
        out = tmp_path / 'u_matrix.png'
        generate_u_matrix(weights, 'square', str(out))
        assert out.is_file()

    def test_u_matrix_hex(self, tmp_path, weights_and_data):
        weights, _ = weights_and_data
        out = tmp_path / 'u_matrix_hex.png'
        generate_u_matrix(weights, 'hex', str(out))
        assert out.is_file()

    def test_hit_map(self, tmp_path, weights_and_data):
        weights, data = weights_and_data
        out = tmp_path / 'hit_map.png'
        generate_hit_map(weights, 'square', data, str(out))
        assert out.is_file()

    def test_distance_map(self, tmp_path, weights_and_data):
        weights, data = weights_and_data
        out = tmp_path / 'distance_map.png'
        generate_distance_map(weights, 'square', data, None, str(out))
        assert out.is_file()

    def test_dead_neurons_map(self, tmp_path, weights_and_data):
        weights, data = weights_and_data
        out = tmp_path / 'dead.png'
        generate_dead_neurons_map(weights, 'square', data, str(out))
        assert out.is_file()

    def test_individual_maps_for_ea(self, tmp_path, weights_and_data):
        weights, data = weights_and_data
        generate_individual_maps(weights, 'hex', data, None, str(tmp_path))
        viz = tmp_path / 'visualizations'
        for name in ('u_matrix.png', 'distance_map.png', 'dead_neurons_map.png'):
            assert (viz / name).is_file(), name

    def test_legends_created(self, tmp_path, weights_and_data):
        weights, _ = weights_and_data
        out = tmp_path / 'u_matrix.png'
        generate_u_matrix(weights, 'square', str(out))
        assert (tmp_path / 'legends' / 'u_matrix.png').is_file()


class TestGenerateAllMaps:
    def test_full_orchestration(self, tmp_path, weights_and_data):
        weights, data = weights_and_data
        original_df = pd.DataFrame({
            'ID': range(20),
            'value_a': data[:, 1],
            'category': ['x'] * 10 + ['y'] * 10,
        })
        config = {
            'primary_id': 'ID',
            'numerical_column': ['value_a'],
            'categorical_column': [],  # no pie data files on disk → skip pies
        }
        generate_all_maps(weights, 'square', original_df, data, config, None, str(tmp_path))
        viz = tmp_path / 'visualizations'
        assert (viz / 'u_matrix.png').is_file()
        assert (viz / 'hit_map.png').is_file()
        # component planes: one per non-ID column (3 columns match dim=3)
        assert (viz / 'component_value_a.png').is_file()
        assert (viz / 'component_category.png').is_file()
        assert not (viz / 'component_ID.png').exists()


class TestRenderResultsDir:
    """Re-rendering a stored run from artifacts only — the UI/ablation path."""

    def make_results_dir(self, tmp_path):
        rng = np.random.default_rng(0)
        weights = rng.random((4, 4, 3))
        data = rng.random((20, 3))
        csv_dir = tmp_path / 'csv'
        json_dir = tmp_path / 'json'
        csv_dir.mkdir()
        json_dir.mkdir()

        np.save(csv_dir / 'weights.npy', weights)
        np.save(csv_dir / 'training_data.npy', data)
        pd.DataFrame({
            'ID': range(20),
            'value_a': data[:, 1],
            'category': ['x'] * 10 + ['y'] * 10,
        }).to_csv(csv_dir / 'original_input.csv', index=False)
        pd.DataFrame(np.zeros((20, 3), dtype=bool)).to_csv(
            csv_dir / 'ignore_mask.csv', index=False, header=False)
        (tmp_path / 'run_metrics.json').write_text(json.dumps({'map_topology': 'hex'}))
        (json_dir / 'preprocessing_info.json').write_text(json.dumps({
            'ID': {'status': 'used', 'base_type': 'numeric', 'nunique_ratio': 1.0},
            'value_a': {'status': 'used', 'base_type': 'numeric',
                        'is_categorical': False, 'nunique_ratio': 0.9},
            'category': {'status': 'used', 'base_type': 'text',
                         'is_categorical': True, 'nunique_ratio': 0.1},
        }))
        return tmp_path

    def test_renders_maps_without_live_som(self, tmp_path):
        results_dir = self.make_results_dir(tmp_path)
        viz_dir = render_results_dir(str(results_dir))
        for name in ('u_matrix.png', 'hit_map.png', 'distance_map.png',
                     'dead_neurons_map.png', 'component_value_a.png',
                     'component_category.png'):
            assert os.path.isfile(os.path.join(viz_dir, name)), name

    def test_missing_weights_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            render_results_dir(str(tmp_path))


class TestTrainingPlots:
    def test_plots_from_history(self, tmp_path):
        results = {
            'history': {
                'mqe': [(0, 0.5), (10, 0.3), (20, 0.2)],
                'learning_rate': [(0, 0.9), (10, 0.5), (20, 0.1)],
                'radius': [(0, 5.0), (10, 3.0), (20, 1.0)],
                'batch_size': [(0, 4), (10, 4), (20, 4)],
            },
            'best_mqe': 0.2,
        }
        generate_training_plots(results, str(tmp_path))
        viz = tmp_path / 'visualizations'
        for name in ('mqe_evolution.png', 'learning_rate_decay.png',
                     'radius_decay.png', 'batch_size_growth.png'):
            assert (viz / name).is_file(), name

    def test_missing_history_is_noop(self, tmp_path):
        generate_training_plots({}, str(tmp_path))
        assert not (tmp_path / 'visualizations' / 'mqe_evolution.png').exists()
