"""
Baseline tests for the SOM preprocessing pipeline — app/som/preprocess.py.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from som.preprocess import validate_input_data, preprocess_data  # noqa: E402


@pytest.fixture
def sample_df():
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame({
        'ID': range(1, n + 1),
        'value_a': rng.normal(10, 2, n),
        'value_b': rng.normal(100, 30, n),
        'category': rng.choice(['red', 'green', 'blue'], n),
        'free_text': [f'unique note {i}' for i in range(n)],  # noise: high cardinality
    })


@pytest.fixture
def config():
    return {
        'primary_id': 'ID',
        'categorical_threshold_numeric': 20,
        'categorical_threshold_text': 20,
        'noise_threshold_ratio': 0.2,
        'delimiter': ',',
    }


class TestValidateInputData:
    def test_missing_file_raises(self, tmp_path, config):
        with pytest.raises(FileNotFoundError):
            validate_input_data(str(tmp_path / 'nope.csv'), str(tmp_path), config)

    def test_loads_valid_csv(self, tmp_path, config, sample_df):
        path = tmp_path / 'data.csv'
        sample_df.to_csv(path, index=False)
        df = validate_input_data(str(path), str(tmp_path), config)
        assert len(df) == 50
        assert list(df.columns) == list(sample_df.columns)

    def test_missing_selected_columns_raises(self, tmp_path, config, sample_df):
        path = tmp_path / 'data.csv'
        sample_df.to_csv(path, index=False)
        config['selected_columns'] = ['ID', 'does_not_exist']
        with pytest.raises(ValueError, match='missing required columns'):
            validate_input_data(str(path), str(tmp_path), config)

    def test_selected_columns_subsets(self, tmp_path, config, sample_df):
        path = tmp_path / 'data.csv'
        sample_df.to_csv(path, index=False)
        config['selected_columns'] = ['ID', 'value_a']
        df = validate_input_data(str(path), str(tmp_path), config)
        assert list(df.columns) == ['ID', 'value_a']

    def test_empty_csv_raises(self, tmp_path, config):
        path = tmp_path / 'empty.csv'
        path.write_text('')
        with pytest.raises(ValueError):
            validate_input_data(str(path), str(tmp_path), config)


class TestPreprocessData:
    def test_is_pure_no_disk_writes_no_config_mutation(self, tmp_path, monkeypatch,
                                                       config, sample_df):
        monkeypatch.chdir(tmp_path)
        config_before = dict(config)
        preprocess_data(sample_df, config)
        assert list(tmp_path.iterdir()) == []
        assert config == config_before

    def test_normalization_to_unit_range(self, config, sample_df):
        result = preprocess_data(sample_df, config)
        eps = 1e-9  # MinMaxScaler float rounding
        assert result.training_data.min() >= -eps
        assert result.training_data.max() <= 1.0 + eps

    def test_noise_column_excluded_from_training(self, config, sample_df):
        result = preprocess_data(sample_df, config)
        # free_text dropped as noise: ID + value_a + value_b + category = 4 dims
        assert result.training_data.shape == (50, 4)
        assert result.preprocessing_info['free_text']['status'] == 'ignored'

    def test_column_classification(self, config, sample_df):
        result = preprocess_data(sample_df, config)
        assert 'value_a' in result.numerical_column
        assert 'value_b' in result.numerical_column
        assert 'category' in result.categorical_column
        assert 'ID' not in result.numerical_column
        assert 'ID' not in result.categorical_column

    def test_primary_id_masked(self, config, sample_df):
        result = preprocess_data(sample_df, config)
        id_col_idx = list(sample_df.columns).index('ID')  # ID is first training col
        assert result.ignore_mask[:, id_col_idx].all()

    def test_fully_masked_columns_zeroed_in_training_data(self, config, sample_df):
        # Fully-masked dims (primary ID) must be inert: zero in the matrix,
        # so they contribute nothing even to unmasked computations (issue #20).
        result = preprocess_data(sample_df, config)
        id_col_idx = list(sample_df.columns).index('ID')
        np.testing.assert_array_equal(result.training_data[:, id_col_idx], 0.0)

    def test_scale_only_does_not_zero_id(self, config, sample_df):
        # Without a mask there is nothing "fully masked" — ID trains as data.
        config['preprocess_strategy'] = 'scale-only'
        result = preprocess_data(sample_df, config)
        id_col_idx = list(sample_df.columns).index('ID')
        assert result.training_data[:, id_col_idx].any()

    def test_nan_values_masked_and_filled(self, config, sample_df):
        sample_df.loc[3, 'value_a'] = np.nan
        sample_df.loc[7, 'value_a'] = np.nan
        result = preprocess_data(sample_df, config)
        col_idx = 1  # value_a
        assert result.ignore_mask[3, col_idx]
        assert result.ignore_mask[7, col_idx]
        assert result.ignore_mask[:, col_idx].sum() == 2
        assert not np.isnan(result.training_data).any()  # median-filled before scaling

    def test_dataset_stats_contract(self, config, sample_df):
        stats = preprocess_data(sample_df, config).dataset_stats
        assert stats['ds_n_samples'] == 50
        assert stats['ds_n_original_cols'] == 5
        assert stats['ds_n_ignored'] == 1
        assert stats['ds_n_numeric'] == 2  # primary ID excluded from features
        assert stats['ds_n_categorical'] == 1
        assert stats['ds_has_primary_id'] == 1
        assert stats['ds_n_dimensions'] == 4

    def test_numeric_low_cardinality_is_categorical(self, config):
        df = pd.DataFrame({
            'ID': range(100),
            'rating': np.tile([1, 2, 3, 4, 5], 20),  # numeric, 5 unique → categorical
            'measure': np.linspace(0, 1, 100),
        })
        result = preprocess_data(df, config)
        assert 'rating' in result.categorical_column
        assert 'measure' in result.numerical_column

    def test_log_fn_receives_messages(self, config, sample_df):
        messages = []
        preprocess_data(sample_df, config, log_fn=messages.append)
        assert any('noise' in m for m in messages)


class TestPreprocessStrategies:
    """Ablation ladder: none → scale-only → nexus (see ABLATION_STUDY.md A1.1)."""

    def test_unknown_strategy_raises(self, config, sample_df):
        config['preprocess_strategy'] = 'bogus'
        with pytest.raises(ValueError, match='preprocess_strategy'):
            preprocess_data(sample_df, config)

    def test_default_strategy_is_nexus(self, config, sample_df):
        result = preprocess_data(sample_df, config)
        assert result.dataset_stats['ds_preprocess_strategy'] == 'nexus'

    def test_scale_only_keeps_noise_and_disables_mask(self, config, sample_df):
        config['preprocess_strategy'] = 'scale-only'
        result = preprocess_data(sample_df, config)
        # free_text NOT excluded → 5 dims instead of 4
        assert result.training_data.shape == (50, 5)
        assert not result.ignore_mask.any()
        # but still normalized
        eps = 1e-9
        assert result.training_data.max() <= 1.0 + eps

    def test_scale_only_does_not_mask_nan_or_id(self, config, sample_df):
        sample_df.loc[3, 'value_a'] = np.nan
        config['preprocess_strategy'] = 'scale-only'
        result = preprocess_data(sample_df, config)
        assert not result.ignore_mask.any()  # NaN filled but not masked

    def test_none_strategy_skips_normalization(self, config, sample_df):
        config['preprocess_strategy'] = 'none'
        result = preprocess_data(sample_df, config)
        # value_b ~ N(100, 30) stays in raw scale → values far outside [0, 1]
        assert result.training_data.max() > 10
        assert not result.ignore_mask.any()
        assert result.dataset_stats['ds_preprocess_strategy'] == 'none'

    def test_none_strategy_still_encodes_text(self, config, sample_df):
        # SOM needs a numeric matrix even without normalization
        config['preprocess_strategy'] = 'none'
        result = preprocess_data(sample_df, config)
        assert result.training_data.dtype == float
        assert not np.isnan(result.training_data).any()
