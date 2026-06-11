"""
Unit tests for the spatial map analysis (mathematical CNN replacement) —
compute_spatial_stats / compute_regions in app/analysis/src/stats.py.
"""
import numpy as np
import pytest

from app.analysis.src.stats import (
    compute_spatial_stats,
    compute_regions,
    _morans_i,
    _boundary_ratio,
)


def make_data(weights=None, pie_data=None, run_metrics=None, original_df=None):
    return {
        'weights':      weights,
        'pie_data':     pie_data or {},
        'run_metrics':  run_metrics or {},
        'original_df':  original_df,
        'columns':      {'numeric': [], 'categorical': [], 'noise': [],
                         'primary_id': None},
    }


def smooth_weights(m=12, n=12, dim=2):
    """Linear ramps — perfectly organized map."""
    rows = np.linspace(0, 1, m)[:, None] * np.ones((1, n))
    cols = np.ones((m, 1)) * np.linspace(0, 1, n)[None, :]
    return np.stack([rows, cols][:dim] + [rows] * max(0, dim - 2), axis=-1)


def noisy_weights(m=12, n=12, dim=2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((m, n, dim))


# ─── compute_spatial_stats ────────────────────────────────────────────────────

class TestSpatialStats:
    def test_missing_weights_returns_empty(self):
        assert compute_spatial_stats(make_data(weights=None)) == {}

    def test_wrong_ndim_returns_empty(self):
        assert compute_spatial_stats(make_data(weights=np.zeros((5, 5)))) == {}

    def test_smooth_map_scores_higher_than_noise(self):
        smooth = compute_spatial_stats(make_data(smooth_weights()))
        noise = compute_spatial_stats(make_data(noisy_weights()))
        assert smooth['spatial_quality_score'] > noise['spatial_quality_score']

    def test_smooth_map_has_high_morans_i(self):
        result = compute_spatial_stats(make_data(smooth_weights()))
        assert result['morans_i_mean'] > 0.8

    def test_noise_map_has_low_morans_i(self):
        result = compute_spatial_stats(make_data(noisy_weights()))
        assert result['morans_i_mean'] < 0.3

    def test_per_feature_keys(self):
        result = compute_spatial_stats(make_data(smooth_weights(dim=2)))
        assert set(result['per_feature']) == {'dim_0', 'dim_1'}
        entry = result['per_feature']['dim_0']
        for key in ('gradient_mean', 'gradient_max', 'n_local_maxima',
                    'n_local_minima', 'roughness', 'morans_i'):
            assert key in entry

    def test_single_peak_detected(self):
        w = np.zeros((11, 11, 1))
        w[5, 5, 0] = 1.0
        result = compute_spatial_stats(make_data(w))
        assert result['per_feature']['dim_0']['n_local_maxima'] == 1

    def test_constant_plane_handled(self):
        result = compute_spatial_stats(make_data(np.ones((8, 8, 1))))
        entry = result['per_feature']['dim_0']
        assert 'morans_i' not in entry
        assert 'roughness' not in entry

    def test_chain_map_singleton_axis(self):
        # 1xN chain maps (helix, space-filling benchmarks) have no row
        # gradient — must not crash and must still produce a score.
        w = np.linspace(0.0, 1.0, 100).reshape(1, 100, 1)
        result = compute_spatial_stats(make_data(w))
        entry = result['per_feature']['dim_0']
        assert entry['gradient_mean'] > 0
        assert 0.0 <= result['spatial_quality_score'] <= 1.0

    def test_score_in_unit_range(self):
        for w in (smooth_weights(), noisy_weights()):
            score = compute_spatial_stats(make_data(w))['spatial_quality_score']
            assert 0.0 <= score <= 1.0


# ─── compute_regions ──────────────────────────────────────────────────────────

def two_halves_pie(m=6, n=6):
    """Left half dominant 'A', right half dominant 'B'."""
    counts = {}
    for r in range(m):
        for c in range(n):
            code = '0' if c < n // 2 else '1'
            counts[f'{r}_{c}'] = {code: 5}
    return {'label': {'categories': {'0': 'A', '1': 'B'}, 'counts': counts}}


class TestRegions:
    def test_two_halves_give_two_regions(self):
        data = make_data(weights=np.zeros((6, 6, 1)), pie_data=two_halves_pie())
        result = compute_regions(data)
        assert result['label']['n_regions'] == 2
        categories = {r['category'] for r in result['label']['regions']}
        assert categories == {'A', 'B'}

    def test_boundary_ratio_of_two_halves(self):
        data = make_data(weights=np.zeros((6, 6, 1)), pie_data=two_halves_pie())
        ratio = compute_regions(data)['label']['boundary_ratio']
        # only the single vertical seam mismatches: 6 of 60 rook pairs
        assert ratio == pytest.approx(0.1)

    def test_no_pie_data_returns_empty(self):
        assert compute_regions(make_data(weights=np.zeros((6, 6, 1)))) == {}

    def test_no_map_shape_returns_empty(self):
        assert compute_regions(make_data(weights=None)) == {}

    def test_region_sizes_sum_to_active_neurons(self):
        data = make_data(weights=np.zeros((6, 6, 1)), pie_data=two_halves_pie())
        regions = compute_regions(data)['label']['regions']
        assert sum(r['size'] for r in regions) == 36

    def test_spatial_stats_includes_regions_and_coherence(self):
        data = make_data(weights=smooth_weights(6, 6, 1),
                         pie_data=two_halves_pie())
        result = compute_spatial_stats(data)
        assert result['regions']['label']['n_regions'] == 2
        assert 'spatial_quality_score' in result


# ─── Internal helpers ─────────────────────────────────────────────────────────

class TestHelpers:
    def test_morans_i_constant_is_none(self):
        assert _morans_i(np.ones((5, 5))) is None

    def test_morans_i_checkerboard_is_negative(self):
        plane = np.indices((8, 8)).sum(axis=0) % 2
        assert _morans_i(plane.astype(float)) < -0.9

    def test_boundary_ratio_homogeneous_is_zero(self):
        assert _boundary_ratio(np.zeros((4, 4), dtype=int)) == 0.0

    def test_boundary_ratio_ignores_inactive(self):
        mat = np.full((4, 4), -1, dtype=int)
        assert _boundary_ratio(mat) is None
