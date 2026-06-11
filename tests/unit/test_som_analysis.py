"""
Baseline tests for post-training SOM analysis — app/som/analysis.py —
and the small utilities in app/som/utils.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from som.analysis import perform_analysis, _detect_extremes, _get_bmu_assignments  # noqa: E402
from som.utils import load_configuration, log_message, get_working_directory  # noqa: E402
from tests.unit.test_som_core import make_som  # noqa: E402


@pytest.fixture
def analysis_setup():
    """Small trained-ish SOM + matching original df and normalized data."""
    rng = np.random.default_rng(0)
    n = 40
    original_df = pd.DataFrame({
        'ID': range(1, n + 1),
        'value_a': rng.normal(10, 2, n),
        'category': rng.choice(['x', 'y'], n),
    })
    # Normalized training data: 3 dims (ID, value_a, category-coded)
    normalized = rng.random((n, 3))
    som = make_som(dim=3)
    config = {
        'primary_id': 'ID',
        'numerical_column': ['value_a'],
        'categorical_column': ['category'],
        'std_threshold': 2.5,
        'preprocessing_info': {
            'ID': {'status': 'used'},
            'value_a': {'status': 'used'},
            'category': {'status': 'used'},
        },
    }
    return som, original_df, normalized, config


class TestPerformAnalysis:
    def test_output_files_created(self, tmp_path, analysis_setup):
        som, original_df, normalized, config = analysis_setup
        (tmp_path / 'json').mkdir()
        perform_analysis(som, original_df, normalized, config, str(tmp_path))
        for rel in ('json/clusters.json', 'json/quantization_errors.json',
                    'json/extremes.json', 'csv/sample_assignments.csv',
                    'json/pie_data_category.json'):
            assert (tmp_path / rel).is_file(), rel

    def test_clusters_cover_all_samples(self, tmp_path, analysis_setup):
        som, original_df, normalized, config = analysis_setup
        (tmp_path / 'json').mkdir()
        perform_analysis(som, original_df, normalized, config, str(tmp_path))
        clusters = json.loads((tmp_path / 'json/clusters.json').read_text())
        all_ids = [sid for ids in clusters.values() for sid in ids]
        assert sorted(all_ids) == list(range(1, 41))

    def test_sample_assignments_have_per_dim_qe(self, tmp_path, analysis_setup):
        som, original_df, normalized, config = analysis_setup
        (tmp_path / 'json').mkdir()
        perform_analysis(som, original_df, normalized, config, str(tmp_path))
        df = pd.read_csv(tmp_path / 'csv/sample_assignments.csv')
        assert 'qe' in df.columns
        assert 'is_outlier' in df.columns
        qe_dim_cols = [c for c in df.columns if c.startswith('qe_dim_')]
        assert 'qe_dim_value_a' in qe_dim_cols
        assert 'qe_dim_category' in qe_dim_cols
        assert len(df) == 40

    def test_quantization_errors_structure(self, tmp_path, analysis_setup):
        som, original_df, normalized, config = analysis_setup
        (tmp_path / 'json').mkdir()
        perform_analysis(som, original_df, normalized, config, str(tmp_path))
        qe = json.loads((tmp_path / 'json/quantization_errors.json').read_text())
        assert 'total_quantization_error' in qe
        assert len(qe['neuron_quantization_errors']) == som.m * som.n


class TestBmuAssignments:
    def test_assignment_columns(self, analysis_setup):
        som, original_df, normalized, config = analysis_setup
        df_assigned, clusters = _get_bmu_assignments(
            som, normalized, original_df, 'ID',
            training_cols=['ID', 'value_a', 'category'])
        for col in ('bmu_i', 'bmu_j', 'bmu_key', 'qe'):
            assert col in df_assigned.columns
        assert sum(len(v) for v in clusters.values()) == 40

    def test_bmu_key_format(self, analysis_setup):
        som, original_df, normalized, config = analysis_setup
        df_assigned, _ = _get_bmu_assignments(som, normalized, original_df, 'ID')
        i, j = map(int, df_assigned['bmu_key'].iloc[0].split('_'))
        assert 0 <= i < som.m
        assert 0 <= j < som.n

    def test_assignment_respects_ignore_mask(self, analysis_setup):
        """Analysis-phase BMU must match training-time (masked) BMU selection:
        a median-filled masked value must not pull a sample to another neuron."""
        som, original_df, normalized, config = analysis_setup
        som.weights[:] = 1.0
        som.weights[2, 2] = [0.5, 0.5, 0.0]   # matches samples on valid dims
        som.weights[3, 3] = [0.7, 0.7, 0.9]   # would win via the filled dim 2
        data = np.tile([0.5, 0.5, 0.9], (len(original_df), 1))
        mask = np.zeros_like(data, dtype=bool)
        mask[:, 2] = True  # dim 2 is a filled missing value

        masked, _ = _get_bmu_assignments(som, data, original_df, 'ID',
                                         ignore_mask=mask)
        unmasked, _ = _get_bmu_assignments(som, data, original_df, 'ID')
        assert (masked['bmu_key'] == '2_2').all()
        assert not (unmasked['bmu_key'] == '2_2').all()

    def test_masked_dims_have_zero_per_dim_qe(self, analysis_setup):
        som, original_df, normalized, config = analysis_setup
        mask = np.zeros_like(normalized, dtype=bool)
        mask[:, 1] = True  # value_a masked for everyone
        df_assigned, _ = _get_bmu_assignments(
            som, normalized, original_df, 'ID',
            training_cols=['ID', 'value_a', 'category'], ignore_mask=mask)
        np.testing.assert_array_equal(df_assigned['qe_dim_value_a'].values, 0.0)


class TestDetectExtremes:
    def test_global_outlier_detected(self):
        n = 30
        df = pd.DataFrame({
            'ID': range(n),
            'value': [10.0] * (n - 1) + [1000.0],  # one massive outlier
            'bmu_key': ['0_0'] * n,
        })
        df['value'] += np.linspace(0, 1, n)  # avoid zero std
        extremes = _detect_extremes(df, ['value'], 'ID', std_threshold=2.5)
        assert n - 1 in extremes
        reasons = ' '.join(extremes[n - 1])
        assert 'global maximum' in reasons

    def test_empty_input_returns_empty(self):
        assert _detect_extremes(pd.DataFrame(), ['x'], 'ID', 2.5) == {}


class TestUtils:
    def test_log_message_appends(self, tmp_path):
        log_message(str(tmp_path), 'SYSTEM', 'hello')
        log_message(str(tmp_path), 'ERROR', 'world')
        content = (tmp_path / 'log.txt').read_text()
        assert '[SYSTEM] hello' in content
        assert '[ERROR] world' in content

    def test_load_configuration_missing_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_configuration(str(tmp_path / 'none.json'))

    def test_load_configuration_invalid_json_raises(self, tmp_path):
        bad = tmp_path / 'bad.json'
        bad.write_text('{not json')
        with pytest.raises(ValueError):
            load_configuration(str(bad))

    def test_load_configuration_roundtrip(self, tmp_path):
        path = tmp_path / 'config.json'
        path.write_text(json.dumps({'map_size': [5, 5]}))
        assert load_configuration(str(path)) == {'map_size': [5, 5]}

    def test_get_working_directory_creates_results_dir(self, tmp_path):
        input_file = tmp_path / 'data.csv'
        input_file.write_text('a,b\n1,2\n')
        wd = get_working_directory(str(input_file))
        assert os.path.isdir(wd)
        assert os.path.dirname(wd) == str(tmp_path / 'results')
