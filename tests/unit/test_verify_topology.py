"""
Unit tests for the topology verification tool (app/tools/verify_topology.py).
A synthetic perfectly-organized map must PASS; a scrambled one must FAIL.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app', 'tools'))

from verify_topology import (  # noqa: E402
    param_adherence, pairwise_distance_spearman, label_metrics,
    grid_coords, split_gt_columns, hit_distribution, verdict,
)


@pytest.fixture
def organized():
    """Samples whose grid position perfectly encodes two parameters."""
    rng = np.random.default_rng(0)
    n = 400
    i = rng.integers(0, 10, n).astype(float)
    j = rng.integers(0, 10, n).astype(float)
    grid_xy = np.stack([j, i], axis=1)
    u = j / 9.0          # param mapped to grid x
    v = i / 9.0          # param mapped to grid y
    return grid_xy, u, v


class TestParamAdherence:
    def test_perfect_organization_passes(self, organized):
        grid_xy, u, _ = organized
        metrics = param_adherence(u, grid_xy)
        assert metrics['grid_param_R2'] > 0.99
        assert metrics['best_axis_spearman'] > 0.99

    def test_scrambled_organization_fails(self, organized):
        grid_xy, u, _ = organized
        rng = np.random.default_rng(1)
        scrambled = rng.permutation(u)
        metrics = param_adherence(scrambled, grid_xy)
        assert metrics['grid_param_R2'] < 0.1

    def test_diagonal_param_still_high_r2(self, organized):
        # A parameter along the grid diagonal is fine for the linear fit.
        grid_xy, u, v = organized
        metrics = param_adherence(u + v, grid_xy)
        assert metrics['grid_param_R2'] > 0.99


class TestPairwiseDistance:
    def test_perfect_map_high_correlation(self, organized):
        grid_xy, u, v = organized
        gt = np.stack([u, v], axis=1)
        assert pairwise_distance_spearman(gt, grid_xy) > 0.95

    def test_scrambled_low_correlation(self, organized):
        grid_xy, u, v = organized
        rng = np.random.default_rng(2)
        gt = rng.random((len(grid_xy), 2))
        assert pairwise_distance_spearman(gt, grid_xy) < 0.2


class TestLabels:
    def test_perfect_separation(self):
        labels = np.array([0] * 50 + [1] * 50)
        neurons = pd.Series(['0_0'] * 50 + ['5_5'] * 50)
        metrics = label_metrics(labels, neurons)
        assert metrics['adjusted_rand_index'] == 1.0
        assert metrics['mean_neuron_purity'] == 1.0

    def test_mixed_neurons_low_ari(self):
        rng = np.random.default_rng(3)
        labels = rng.integers(0, 2, 200)
        neurons = pd.Series(rng.choice(['0_0', '1_1', '2_2'], 200))
        metrics = label_metrics(labels, neurons)
        assert metrics['adjusted_rand_index'] < 0.1


class TestHelpers:
    def test_grid_coords_hex_offsets_odd_rows(self):
        df = pd.DataFrame({'bmu_i': [0, 1], 'bmu_j': [0, 0]})
        xy = grid_coords(df, 'hex')
        assert xy[0, 0] == 0.0
        assert xy[1, 0] == 0.5  # odd row shifted

    def test_grid_coords_square_no_offset(self):
        df = pd.DataFrame({'bmu_i': [1], 'bmu_j': [2]})
        xy = grid_coords(df, 'square')
        assert (xy[0] == [2.0, 1.0]).all()

    def test_split_gt_columns(self):
        gt = pd.DataFrame({
            'id': [1, 2, 3],
            't': [0.1, 0.5, 0.9],
            'label': np.array([0, 1, 0], dtype=int),
        })
        params, labels = split_gt_columns(gt)
        assert params == ['t']
        assert labels == ['label']

    def test_hit_distribution_counts_dead(self):
        assignments = pd.DataFrame({'bmu_key': ['0_0'] * 8 + ['1_1'] * 2})
        stats = hit_distribution(assignments, {'map_size': [2, 2]})
        assert stats['active_neurons'] == 2
        assert stats['dead_ratio'] == 0.5
        assert stats['hit_gini'] > 0.5  # very uneven

    def test_verdict_thresholds(self):
        assert verdict(0.9, 0.8, 0.5) == 'PASS'
        assert verdict(0.6, 0.8, 0.5) == 'WARN'
        assert verdict(0.1, 0.8, 0.5) == 'FAIL'
        assert verdict(None, 0.8, 0.5) == 'N/A'
