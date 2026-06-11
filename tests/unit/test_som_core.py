"""
Baseline (characterization) tests for KohonenSOM — app/som/som.py.

Captures current behavior BEFORE the SOM module refactor so regressions
introduced by the restructuring are visible immediately.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from som.som import KohonenSOM  # noqa: E402


def make_som(**overrides):
    params = {
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
        'epoch_multiplier': 1.0,
        'normalize_weights_flag': False,
        'growth_g': 1.0,
        'random_seed': 42,
        'map_type': 'square',
        'num_batches': 2,
        'max_epochs_without_improvement': 500,
        'mqe_evaluations_per_run': 5,
        'dim': 3,
        'show_progress': False,
    }
    params.update(overrides)
    return KohonenSOM(**params)


class TestInit:
    def test_map_dimensions(self):
        som = make_som(map_size=[4, 7])
        assert (som.m, som.n) == (4, 7)
        assert som.weights.shape == (4, 7, 3)

    def test_start_radius_from_ratio(self):
        som = make_som(map_size=[6, 10], start_radius_init_ratio=0.5)
        assert som.start_radius == 5.0  # 0.5 * max(6, 10)

    def test_default_radius_ratio_is_one(self):
        som = make_som(start_radius_init_ratio=None)
        assert som.start_radius == 6.0

    def test_seed_reproducibility(self):
        w1 = make_som(random_seed=7).weights
        w2 = make_som(random_seed=7).weights
        np.testing.assert_array_equal(w1, w2)

    def test_hex_map_has_cube_coords(self):
        som = make_som(map_type='hex')
        assert som.cube_coords.shape == (6, 6, 3)
        # cube coordinate invariant: x + y + z == 0
        assert np.all(som.cube_coords.sum(axis=2) == 0)


class TestDecay:
    @pytest.mark.parametrize('decay_type', [
        'static', 'linear-drop', 'linear-growth', 'exp-drop', 'exp-growth',
        'log-drop', 'log-growth', 'step-down',
    ])
    def test_decay_endpoints_within_range(self, decay_type):
        som = make_som()
        start, end = 0.9, 0.1
        v0 = som.get_decay_value(0, 100, start, end, decay_type)
        vN = som.get_decay_value(99, 100, start, end, decay_type)
        lo, hi = min(start, end), max(start, end)
        eps = 1e-6
        assert lo - eps <= v0 <= hi + eps
        assert lo - eps <= vN <= hi + eps

    def test_linear_drop_exact(self):
        som = make_som()
        assert som.get_decay_value(0, 101, 1.0, 0.0, 'linear-drop') == 1.0
        assert som.get_decay_value(100, 101, 1.0, 0.0, 'linear-drop') == 0.0
        assert som.get_decay_value(50, 101, 1.0, 0.0, 'linear-drop') == pytest.approx(0.5)

    def test_drop_types_monotonically_decrease(self):
        # Domain constraint: decay curves must never rise again.
        som = make_som()
        for decay_type in ('linear-drop', 'exp-drop', 'log-drop', 'step-down'):
            values = [som.get_decay_value(t, 200, 0.9, 0.1, decay_type)
                      for t in range(200)]
            diffs = np.diff(values)
            assert np.all(diffs <= 1e-9), f'{decay_type} is not monotonic'

    def test_static_returns_start(self):
        som = make_som()
        assert som.get_decay_value(50, 100, 0.7, 0.1, 'static') == 0.7

    def test_n_below_two_returns_start(self):
        som = make_som()
        assert som.get_decay_value(0, 1, 0.9, 0.1, 'linear-drop') == 0.9

    def test_unknown_decay_type_raises(self):
        som = make_som()
        with pytest.raises(ValueError):
            som.get_decay_value(0, 10, 0.9, 0.1, 'bogus')


class TestBmuAndUpdate:
    def test_find_bmu_exact_match(self):
        som = make_som()
        sample = som.weights[2, 3].copy()
        assert som.find_bmu(sample) == (2, 3)

    def test_find_bmu_respects_mask(self):
        som = make_som()
        som.weights[:] = 0.5
        som.weights[1, 1] = [0.0, 0.0, 0.5]
        # Sample matches neuron (1,1) on dims 0-1, masked dim 2 is wildly off.
        sample = np.array([0.0, 0.0, 99.0])
        mask = np.array([False, False, True])
        assert som.find_bmu(sample, mask=mask) == (1, 1)

    def test_update_moves_bmu_toward_sample(self):
        som = make_som()
        sample = np.array([1.0, 1.0, 1.0])
        bmu = som.find_bmu(sample)
        before = np.linalg.norm(som.weights[bmu] - sample)
        som.update_weights(sample, bmu, current_learning_rate=0.5, radius=1.0)
        after = np.linalg.norm(som.weights[bmu] - sample)
        assert after < before

    def test_update_weights_accepts_rect_alias(self):
        # 'rect' (used by benchmark configs and the UI) must take the square
        # path, not the hex path — regression for the cube_coords crash.
        som = make_som(map_type='rect')
        sample = np.array([1.0, 1.0, 1.0])
        bmu = som.find_bmu(sample)
        before = np.linalg.norm(som.weights[bmu] - sample)
        som.update_weights(sample, bmu, current_learning_rate=0.5, radius=1.0)
        assert np.linalg.norm(som.weights[bmu] - sample) < before

    def test_update_skips_masked_dimensions(self):
        som = make_som()
        before = som.weights[:, :, 2].copy()
        sample = np.array([1.0, 1.0, 1.0])
        mask = np.array([False, False, True])
        som.update_weights(sample, (0, 0), 0.5, 1.0, mask=mask)
        np.testing.assert_array_equal(som.weights[:, :, 2], before)


class TestMetrics:
    def test_qe_zero_for_perfect_match(self):
        som = make_som()
        data = som.weights.reshape(-1, 3)[:5].copy()
        _, total_qe = som.compute_quantization_error(data)
        assert total_qe == pytest.approx(0.0)

    def test_qe_known_value(self):
        som = make_som()
        som.weights[:] = 0.0
        data = np.array([[0.3, 0.0, 0.0]])
        _, total_qe = som.compute_quantization_error(data)
        assert total_qe == pytest.approx(0.3)

    def test_neuron_error_map_shape(self):
        som = make_som()
        data = np.random.default_rng(0).random((20, 3))
        error_map, _ = som.compute_quantization_error(data)
        assert error_map.shape == (som.m, som.n)

    def test_dead_neurons_all_on_one(self):
        som = make_som()
        som.weights[:] = 1.0
        som.weights[0, 0] = [0.0, 0.0, 0.0]
        data = np.zeros((10, 3))
        count, ratio = som.calculate_dead_neurons(data)
        assert count == som.m * som.n - 1
        assert ratio == pytest.approx((som.m * som.n - 1) / (som.m * som.n))

    def test_dead_neurons_empty_data(self):
        som = make_som()
        count, ratio = som.calculate_dead_neurons(np.empty((0, 3)))
        assert count == som.m * som.n
        assert ratio == 1.0

    def test_topographic_error_organized_square(self):
        som = make_som(dim=2)
        # Weights = grid coordinates: BMU and 2nd BMU are always grid neighbors.
        coords = np.indices((som.m, som.n)).transpose(1, 2, 0).astype(float)
        som.weights = coords.copy()
        data = coords.reshape(-1, 2) + 0.1
        assert som.calculate_topographic_error(data) == pytest.approx(0.0)

    def test_topographic_error_scrambled_is_high(self):
        som = make_som(dim=2)
        coords = np.indices((som.m, som.n)).transpose(1, 2, 0).astype(float)
        rng = np.random.default_rng(1)
        flat = coords.reshape(-1, 2)
        som.weights = rng.permutation(flat).reshape(som.m, som.n, 2)
        data = flat + 0.1
        assert som.calculate_topographic_error(data) > 0.3

    def test_topological_correlation_organized_map(self):
        som = make_som(dim=2)
        som.weights = np.indices((som.m, som.n)).transpose(1, 2, 0).astype(float)
        assert som.calculate_topological_correlation() > 0.9

    def test_topological_correlation_constant_weights(self):
        som = make_som()
        som.weights[:] = 0.5
        assert som.calculate_topological_correlation() == 0.0

    def test_u_matrix_constant_weights(self):
        som = make_som()
        som.weights[:] = 0.5
        metrics = som.calculate_u_matrix_metrics()
        assert metrics['u_matrix_mean'] == pytest.approx(0.0)
        assert metrics['u_matrix_max'] == pytest.approx(0.0)

    def test_normalize_weights_unit_norm(self):
        som = make_som()
        som.normalize_weights()
        norms = np.linalg.norm(som.weights, axis=2)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-10)


class TestTrain:
    def test_train_returns_expected_contract(self):
        som = make_som()
        rng = np.random.default_rng(0)
        data = rng.random((40, 3))
        results = som.train(data)

        for key in ('best_mqe', 'duration', 'total_weight_updates', 'epochs_ran',
                    'converged', 'lstm_stopped', 'history', 'checkpoints',
                    'sample_coverage'):
            assert key in results
        assert results['epochs_ran'] == 40  # epoch_multiplier 1.0 × 40 samples
        assert results['best_mqe'] < float('inf')

    def test_train_writes_nothing_to_disk(self, tmp_path, monkeypatch):
        # Core contract: train() is pure compute — persistence is the caller's job.
        monkeypatch.chdir(tmp_path)
        som = make_som()
        som.train(np.random.default_rng(0).random((30, 3)))
        assert list(tmp_path.iterdir()) == []

    def test_train_improves_mqe(self):
        som = make_som(epoch_multiplier=5.0)
        rng = np.random.default_rng(0)
        # Two well-separated clusters
        data = np.vstack([rng.normal(0.2, 0.03, (30, 3)),
                          rng.normal(0.8, 0.03, (30, 3))])
        results = som.train(data)
        mqe_series = [v for _, v in results['history']['mqe']]
        assert results['best_mqe'] < mqe_series[0]

    def test_train_with_coverage_tracking(self):
        som = make_som(track_sample_coverage=True)
        data = np.random.default_rng(0).random((30, 3))
        results = som.train(data)
        cov = results['sample_coverage']
        assert cov is not None
        assert cov['total_samples'] == 30
        assert len(cov['counts']) == 30

    def test_train_with_checkpoints(self):
        som = make_som(save_checkpoints=True, checkpoint_count=4)
        data = np.random.default_rng(0).random((30, 3))
        results = som.train(data)
        assert len(results['checkpoints']) > 0
        cp = results['checkpoints'][0]
        for key in ('iteration', 'progress', 'mqe', 'topographic_error',
                    'dead_neuron_ratio', 'learning_rate', 'radius'):
            assert key in cp

    def test_train_respects_ignore_mask(self):
        som = make_som()
        data = np.random.default_rng(0).random((20, 3))
        mask = np.zeros((20, 3), dtype=bool)
        mask[:, 2] = True  # dim 2 invisible for all samples
        results = som.train(data, ignore_mask=mask)
        assert results['best_mqe'] < float('inf')

    def test_train_zeroes_fully_masked_weight_dims(self):
        # A dim masked for ALL samples never updates — its weights must be
        # zeroed so it stays inert in unmasked computations (issue #20).
        som = make_som()
        data = np.random.default_rng(0).random((20, 3))
        mask = np.zeros((20, 3), dtype=bool)
        mask[:, 1] = True
        som.train(data, ignore_mask=mask)
        np.testing.assert_array_equal(som.weights[:, :, 1], 0.0)
        assert som.weights[:, :, 0].any()  # other dims trained normally

    def test_partially_masked_dim_keeps_training(self):
        # A dim masked for only SOME samples must keep learning from the rest.
        som = make_som()
        data = np.random.default_rng(0).random((20, 3))
        mask = np.zeros((20, 3), dtype=bool)
        mask[:5, 1] = True  # invalid value in 5 samples only
        before = som.weights[:, :, 1].copy()
        som.train(data, ignore_mask=mask)
        assert not np.array_equal(som.weights[:, :, 1], before)
        assert som.weights[:, :, 1].any()

    def test_dead_neurons_respect_mask(self):
        som = make_som()
        som.weights[:] = 1.0
        som.weights[1, 1] = [0.5, 0.5, 0.0]
        # All samples sit on neuron (1,1) only if the noisy dim 2 is masked
        data = np.tile([0.5, 0.5, 99.0], (10, 1))
        mask = np.zeros((10, 3), dtype=bool)
        mask[:, 2] = True
        _, ratio_masked = som.calculate_dead_neurons(data, mask=mask)
        assert ratio_masked == pytest.approx((som.m * som.n - 1) / (som.m * som.n))
        # the mask changes which neuron wins: without it dim 2 dominates
        assert som.find_bmu(data[0], mask=mask[0]) == (1, 1)
        assert som.find_bmu(data[0]) != (1, 1)

    def test_train_log_fn_receives_messages_with_coverage(self):
        messages = []
        som = make_som(track_sample_coverage=True)
        som.train(np.random.default_rng(0).random((20, 3)),
                  log_fn=messages.append)
        assert any('Sample coverage' in m for m in messages)
