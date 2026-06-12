"""
Characterization tests for the EA core pure functions — app/ea/ea.py.

Captures current behavior BEFORE the EA module cleanup (docs/ea/CLEANUP_PLAN.md
phase 0): constrained dominance, non-dominated sorting, crowding distance,
tournament selection, repair, constraint violation, objective normalization,
dynamic search space, and UID hashing. Safety net for phases 1–3.
"""
import os
import random
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from ea import ea  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_state():
    """Isolate module-level mutable state (running stats, RNG)."""
    random.seed(1234)
    saved = (ea._OBJ_RUNNING_MIN, ea._OBJ_RUNNING_MAX)
    ea._OBJ_RUNNING_MIN = None
    ea._OBJ_RUNNING_MAX = None
    yield
    ea._OBJ_RUNNING_MIN, ea._OBJ_RUNNING_MAX = saved


# ─── validate_and_repair ──────────────────────────────────────────────────────

class TestValidateAndRepair:
    def test_lr_order_swapped(self):
        out = ea.validate_and_repair(
            {'start_learning_rate': 0.1, 'end_learning_rate': 0.5})
        assert out['start_learning_rate'] == 0.5
        assert out['end_learning_rate'] == 0.1

    def test_batch_order_swapped(self):
        out = ea.validate_and_repair(
            {'start_batch_percent': 10.0, 'end_batch_percent': 2.0})
        assert out['start_batch_percent'] == 2.0
        assert out['end_batch_percent'] == 10.0

    def test_epoch_multiplier_clamped_to_float_min(self):
        # Dynamic search-space bounds go below 1.0 for large datasets — the
        # clamp must stay float with min 0.1 (legacy ISSUES.md #49 is outdated).
        assert ea.validate_and_repair({'epoch_multiplier': 0.01})['epoch_multiplier'] == 0.1
        assert ea.validate_and_repair({'epoch_multiplier': 0.25})['epoch_multiplier'] == 0.25

    def test_growth_g_zeroed_when_all_curves_linear(self):
        out = ea.validate_and_repair({
            'lr_decay_type': 'linear-drop',
            'radius_decay_type': 'linear-drop',
            'batch_growth_type': 'linear-growth',
            'growth_g': 25,
        })
        assert out['growth_g'] == 0

    def test_growth_g_min_one_when_nonlinear_curve_present(self):
        out = ea.validate_and_repair({
            'lr_decay_type': 'exp-drop',
            'radius_decay_type': 'linear-drop',
            'batch_growth_type': 'linear-growth',
            'growth_g': 0.2,
        })
        assert out['growth_g'] == 1.0

    def test_num_batches_min_one(self):
        assert ea.validate_and_repair({'num_batches': 0})['num_batches'] == 1

    def test_input_not_mutated(self):
        config = {'start_learning_rate': 0.1, 'end_learning_rate': 0.5}
        snapshot = dict(config)
        ea.validate_and_repair(config)
        assert config == snapshot


# ─── Constraint violation ─────────────────────────────────────────────────────

class TestConstraintViolation:
    def test_dead_threshold_formula(self):
        # clamp(1 - coverage/10, 0.30, 0.85) — docstring examples for n=569
        assert ea._dead_neuron_threshold(8, 8, 569) == pytest.approx(0.30)
        assert ea._dead_neuron_threshold(12, 12, 569) == pytest.approx(0.605, abs=1e-3)
        assert ea._dead_neuron_threshold(14, 14, 569) == pytest.approx(0.710, abs=1e-3)
        assert ea._dead_neuron_threshold(20, 20, 569) == pytest.approx(0.85)

    def base_results(self, **over):
        results = {'u_matrix_max': 0.9, 'distance_map_max': 0.8,
                   'dead_neuron_ratio': 0.0, 'map_m': 10, 'map_n': 10}
        results.update(over)
        return results

    def test_feasible_is_zero(self):
        # coverage 1000/100 = 10 → dead threshold 0.30; everything below limits
        cv = ea.compute_constraint_violation(self.base_results(), 1000, org_threshold=1.0)
        assert cv == 0.0

    def test_org_violation_is_excess(self):
        cv = ea.compute_constraint_violation(
            self.base_results(u_matrix_max=1.3), 1000, org_threshold=1.0)
        assert cv == pytest.approx(0.3)

    def test_dead_violation_graduated_bands(self):
        # threshold 0.30 → excess 0.1 / 0.3 / 0.55 hit the 1.5 / 2.5 / 5.0 bands
        for dead, expected in [(0.4, 0.1 * 1.5), (0.6, 0.3 * 2.5), (0.85, 0.55 * 5.0)]:
            cv = ea.compute_constraint_violation(
                self.base_results(dead_neuron_ratio=dead), 1000, org_threshold=1.0)
            assert cv == pytest.approx(expected), f"dead_ratio={dead}"

    def test_org_and_dead_sum(self):
        cv = ea.compute_constraint_violation(
            self.base_results(u_matrix_max=1.2, dead_neuron_ratio=0.4),
            1000, org_threshold=1.0)
        assert cv == pytest.approx(0.2 + 0.15)

    def test_missing_metrics_tolerated(self):
        assert ea.compute_constraint_violation({}, 1000, org_threshold=1.0) == 0.0


# ─── Constrained dominance + non-dominated sort ───────────────────────────────

class TestDominanceAndSort:
    def test_plain_pareto_dominance(self):
        objectives = np.array([[0.0, 0.0], [1.0, 1.0]])
        assert ea._dominates(objectives, None, 0, 1)
        assert not ea._dominates(objectives, None, 1, 0)

    def test_non_dominated_pair(self):
        objectives = np.array([[0.0, 1.0], [1.0, 0.0]])
        assert not ea._dominates(objectives, None, 0, 1)
        assert not ea._dominates(objectives, None, 1, 0)
        assert ea.non_dominated_sort(objectives) == [[0, 1]]

    def test_sort_layers(self):
        objectives = np.array([
            [0.0, 0.0],   # dominates everything
            [1.0, 2.0],   # front 1
            [2.0, 1.0],   # front 1
            [3.0, 3.0],   # front 2
        ])
        assert ea.non_dominated_sort(objectives) == [[0], [1, 2], [3]]

    def test_feasible_always_dominates_infeasible(self):
        # index 1 has the better raw objectives but is infeasible
        objectives = np.array([[5.0, 5.0], [0.0, 0.0]])
        violations = np.array([0.0, 0.7])
        assert ea._dominates(objectives, violations, 0, 1)
        assert not ea._dominates(objectives, violations, 1, 0)

    def test_between_infeasible_lower_cv_wins(self):
        objectives = np.array([[5.0, 5.0], [0.0, 0.0]])
        violations = np.array([0.1, 0.7])
        assert ea._dominates(objectives, violations, 0, 1)

    def test_constrained_sort_puts_infeasible_last(self):
        objectives = np.array([
            [0.0, 0.0],   # infeasible despite perfect objectives
            [1.0, 2.0],   # feasible
            [2.0, 1.0],   # feasible
        ])
        violations = np.array([0.5, 0.0, 0.0])
        assert ea.non_dominated_sort(objectives, violations) == [[1, 2], [0]]


# ─── Crowding distance + tournament ───────────────────────────────────────────

class TestCrowdingAndTournament:
    def test_extremes_infinite_interior_finite(self):
        objectives = np.array([[0.0, 3.0], [1.0, 2.0], [2.0, 1.0], [3.0, 0.0]])
        fronts = [[0, 1, 2, 3]]
        cd = ea.crowding_distance_assignment(objectives, fronts)
        assert np.isinf(cd[0]) and np.isinf(cd[3])
        assert np.isfinite(cd[1]) and np.isfinite(cd[2])
        assert cd[1] == pytest.approx(cd[2])  # symmetric interior points

    def test_two_member_front_both_infinite(self):
        objectives = np.array([[0.0, 1.0], [1.0, 0.0]])
        cd = ea.crowding_distance_assignment(objectives, [[0, 1]])
        assert np.isinf(cd).all()

    def test_tournament_lower_rank_wins(self):
        population = [
            {'name': 'worst', 'rank': 2, 'crowding_distance': 99.0},
            {'name': 'best', 'rank': 0, 'crowding_distance': 0.1},
            {'name': 'mid', 'rank': 1, 'crowding_distance': 5.0},
        ]
        for _ in range(20):
            assert ea.tournament_selection(population, k=3)['name'] == 'best'

    def test_tournament_k_larger_than_population_is_clamped(self):
        # tournament_k=5 from dataset configs must not crash a population of 2
        # (issues.md #91)
        population = [
            {'name': 'a', 'rank': 1, 'crowding_distance': 0.0},
            {'name': 'b', 'rank': 0, 'crowding_distance': 0.0},
        ]
        for _ in range(10):
            assert ea.tournament_selection(population, k=5)['name'] == 'b'

    def test_tournament_tie_broken_by_crowding(self):
        population = [
            {'name': 'a', 'rank': 0, 'crowding_distance': 0.1},
            {'name': 'b', 'rank': 0, 'crowding_distance': 5.0},
            {'name': 'c', 'rank': 0, 'crowding_distance': 1.0},
        ]
        for _ in range(20):
            assert ea.tournament_selection(population, k=3)['name'] == 'b'


# ─── Objective normalization (running min/max) ────────────────────────────────

class TestObjectiveNormalization:
    def test_fallback_clip_before_first_update(self):
        out = ea._normalize_objectives(np.array([[1.5, -0.5, 0.7]]))
        assert out.tolist() == [[1.1, 0.0, 0.7]]

    def test_normalizes_to_unit_range_after_update(self):
        ea._update_obj_running_stats(np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 8.0]]))
        out = ea._normalize_objectives(np.array([[1.0, 2.0, 4.0]]))
        assert out.tolist() == [[0.5, 0.5, 0.5]]
        assert ea._normalize_objectives(np.array([[2.0, 4.0, 8.0]])).tolist() == [[1.0, 1.0, 1.0]]

    def test_running_stats_only_widen(self):
        ea._update_obj_running_stats(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        ea._update_obj_running_stats(np.array([[0.5, 0.5, 0.5]]))  # inside range — no shrink
        assert ea._OBJ_RUNNING_MIN.tolist() == [0.0, 0.0, 0.0]
        assert ea._OBJ_RUNNING_MAX.tolist() == [1.0, 1.0, 1.0]

    def test_values_above_max_clipped(self):
        ea._update_obj_running_stats(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
        out = ea._normalize_objectives(np.array([[2.0, 2.0, 2.0]]))
        assert out.tolist() == [[1.1, 1.1, 1.1]]

    def test_degenerate_dimension_yields_no_nan(self):
        ea._update_obj_running_stats(np.array([[0.5, 0.0, 0.0], [0.5, 1.0, 1.0]]))
        out = ea._normalize_objectives(np.array([[0.5, 0.5, 0.5]]))
        assert not np.isnan(out).any()

    def test_empty_update_is_noop(self):
        ea._update_obj_running_stats(np.empty((0, 3)))
        assert ea._OBJ_RUNNING_MIN is None


# ─── Dynamic search space ─────────────────────────────────────────────────────

class TestDynamicSearchSpace:
    SPEC = {
        'map_size': {'type': 'discrete_int_pair'},
        'epoch_multiplier': {'type': 'float'},
        'num_batches': {'type': 'int', 'min': 1, 'max': 20},
        'fixed_thing': 'keep-me',
    }

    def test_vesanto_corridor_for_569_samples(self):
        out = ea.apply_dynamic_search_space(self.SPEC, 569)
        assert out['map_size']['min'] == 8
        assert out['map_size']['max'] == 14

    def test_epoch_multiplier_anchor_values(self):
        for n, em_min, em_max in [(100, 1.0, 5.0), (500, 1.0, 5.0),
                                  (20_000, 0.3, 1.0), (50_000, 0.1, 0.3)]:
            out = ea.apply_dynamic_search_space(self.SPEC, n)
            assert out['epoch_multiplier']['min'] == pytest.approx(em_min), f"n={n}"
            assert out['epoch_multiplier']['max'] == pytest.approx(em_max), f"n={n}"

    def test_epoch_multiplier_monotonically_decreasing(self):
        maxima = [ea.apply_dynamic_search_space(self.SPEC, n)['epoch_multiplier']['max']
                  for n in (100, 1_000, 10_000, 100_000)]
        assert maxima == sorted(maxima, reverse=True)

    def test_interp_log_midpoint(self):
        # n=10000 between anchors (5000, 3.0) and (20000, 1.0): t=0.5 in log-n
        assert ea._interp_log(10_000, ea._EM_MAX_ANCHORS) == pytest.approx(
            3.0 * (1.0 / 3.0) ** 0.5, abs=0.01)

    def test_other_keys_passed_through_and_input_not_mutated(self):
        out = ea.apply_dynamic_search_space(self.SPEC, 569)
        assert out['num_batches'] == self.SPEC['num_batches']
        assert out['fixed_thing'] == 'keep-me'
        assert 'min' not in self.SPEC['map_size']  # original spec untouched


# ─── UID hashing ──────────────────────────────────────────────────────────────

class TestGetUid:
    def test_key_order_independent(self):
        assert ea.get_uid({'a': 1, 'b': [10, 10]}) == ea.get_uid({'b': [10, 10], 'a': 1})

    def test_value_sensitive(self):
        assert ea.get_uid({'a': 1}) != ea.get_uid({'a': 2})

    def test_selection_metadata_does_not_change_the_uid(self):
        # Finding F8 (CLEANUP_PLAN.md phase 2): UIDs are gene-only — rank and
        # crowding_distance attached during selection must not affect the hash,
        # otherwise archive/survivor UIDs never match offspring UIDs.
        genes = {'a': 1, 'b': 2}
        assert ea.get_uid(genes) == ea.get_uid(dict(genes, rank=0, crowding_distance=1.0))

    def test_gene_keys_still_change_the_uid(self):
        genes = {'a': 1, 'b': 2}
        assert ea.get_uid(genes) != ea.get_uid(dict(genes, c=3))

    def test_deterministic_md5(self):
        uid = ea.get_uid({'a': 1})
        assert uid == ea.get_uid({'a': 1})
        assert len(uid) == 32
