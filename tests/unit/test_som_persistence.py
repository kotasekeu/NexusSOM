"""
Tests for the SOM persistence layer (app/som/persistence.py) and the
vectorized U-Matrix computation (app/som/visualization.py).
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from som.persistence import (  # noqa: E402
    save_weights, save_training_checkpoints, save_sample_coverage,
    save_run_metrics, save_preprocess_artifacts,
)
from som.preprocess import preprocess_data  # noqa: E402
from som.visualization import compute_u_matrix  # noqa: E402


class TestSaveWeights:
    def test_npy_and_readable_csv(self, tmp_path):
        weights = np.random.default_rng(0).random((4, 5, 3))
        npy_path = save_weights(weights, str(tmp_path))
        loaded = np.load(npy_path)
        np.testing.assert_array_equal(loaded, weights)

        readable = pd.read_csv(tmp_path / 'csv' / 'weights_readable.csv')
        assert list(readable.columns) == ['neuron_i', 'neuron_j', 'dim_0', 'dim_1', 'dim_2']
        assert len(readable) == 20


class TestSaveCheckpointsAndCoverage:
    def test_checkpoints_roundtrip(self, tmp_path):
        cps = [{'iteration': 0, 'mqe': 0.5}, {'iteration': 10, 'mqe': 0.3}]
        path = save_training_checkpoints(cps, str(tmp_path))
        assert json.loads(open(path).read()) == cps

    def test_empty_checkpoints_noop(self, tmp_path):
        assert save_training_checkpoints([], str(tmp_path)) is None
        assert not (tmp_path / 'csv' / 'training_checkpoints.json').exists()

    def test_coverage_roundtrip(self, tmp_path):
        cov = {'min': 1, 'max': 5, 'total_samples': 3, 'counts': [1, 5, 3]}
        path = save_sample_coverage(cov, str(tmp_path))
        assert json.loads(open(path).read()) == cov

    def test_none_coverage_noop(self, tmp_path):
        assert save_sample_coverage(None, str(tmp_path)) is None

    def test_run_metrics(self, tmp_path):
        save_run_metrics({'best_mqe': 0.1}, str(tmp_path))
        assert json.loads((tmp_path / 'run_metrics.json').read_text()) == {'best_mqe': 0.1}


class TestSavePreprocessArtifacts:
    def test_full_artifact_layout(self, tmp_path):
        df = pd.DataFrame({
            'ID': range(1, 21),
            'value': np.linspace(0, 10, 20),
            'category': ['a', 'b'] * 10,
        })
        config = {'primary_id': 'ID', 'categorical_threshold_numeric': 5,
                  'categorical_threshold_text': 5, 'noise_threshold_ratio': 0.2}
        result = preprocess_data(df, config)
        npy_path = save_preprocess_artifacts(result, df, str(tmp_path))

        for rel in ('csv/original_input.csv', 'csv/training_data.npy',
                    'csv/training_data_readable.csv', 'csv/ignore_mask.csv',
                    'json/preprocessing_info.json', 'dataset_meta.json'):
            assert (tmp_path / rel).is_file(), rel
        np.testing.assert_array_equal(np.load(npy_path), result.training_data)
        meta = json.loads((tmp_path / 'dataset_meta.json').read_text())
        assert meta == result.dataset_stats


class TestComputeUMatrix:
    def test_constant_weights_zero(self):
        u = compute_u_matrix(np.ones((5, 5, 3)), 'square')
        np.testing.assert_allclose(u, 0.0)

    def test_square_known_values(self):
        # 1D ramp along columns: every horizontal neighbor distance is 1.
        weights = np.zeros((3, 3, 1))
        weights[:, :, 0] = [[0, 1, 2]] * 3
        u = compute_u_matrix(weights, 'square')
        # corner (0,0): neighbors right(1) and down(0) → mean 0.5
        assert u[0, 0] == pytest.approx(0.5)
        # center (1,1): neighbors 1,1 (horizontal) + 0,0 (vertical) → mean 0.5
        assert u[1, 1] == pytest.approx(0.5)

    def test_matches_legacy_loop_square(self):
        weights = np.random.default_rng(0).random((6, 7, 4))
        u = compute_u_matrix(weights, 'square')
        # Legacy reference implementation (pre-vectorization)
        m, n = 6, 7
        expected = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                dists = [np.linalg.norm(weights[i, j] - weights[ni, nj])
                         for ni, nj in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]
                         if 0 <= ni < m and 0 <= nj < n]
                expected[i, j] = np.mean(dists)
        np.testing.assert_allclose(u, expected)

    def test_matches_cube_convention_hex(self):
        """Hex U-matrix neighbors must equal cube-distance-1 neighbors —
        the same convention KohonenSOM uses for training and TE
        (the legacy implementation had row parities swapped, issues #25)."""
        weights = np.random.default_rng(1).random((6, 7, 3))
        u = compute_u_matrix(weights, 'hex')
        m, n = 6, 7
        # Authoritative cube coordinates (odd-r), as in KohonenSOM.__init__
        coords = np.indices((m, n)).transpose(1, 2, 0)
        q = coords[:, :, 1] - np.floor(coords[:, :, 0] / 2)
        z = coords[:, :, 0]
        cube = np.stack([q, -q - z, z], axis=-1)
        expected = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                dists = [np.linalg.norm(weights[i, j] - weights[ni, nj])
                         for ni in range(m) for nj in range(n)
                         if np.abs(cube[i, j] - cube[ni, nj]).sum() / 2 == 1]
                expected[i, j] = np.mean(dists)
        np.testing.assert_allclose(u, expected)
