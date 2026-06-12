"""
Characterization tests for the EA genetic operators — app/ea/genetic_operators.py.

Captures current behavior BEFORE the EA module cleanup (docs/ea/CLEANUP_PLAN.md
phase 0) so regressions introduced by later phases are visible immediately.
Replaces the manual print script app/ea/test_genetic_operators.py.
"""
import math
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'app'))

from ea.genetic_operators import (  # noqa: E402
    sbx_crossover,
    polynomial_mutation,
    random_config_continuous,
    crossover_mixed,
    mutate_mixed,
)

SPACE = {
    'map_size': {'type': 'discrete_int_pair', 'min': 8, 'max': 14},
    'start_learning_rate': {'type': 'float', 'min': 0.5, 'max': 1.0, 'log_scale': True},
    'end_learning_rate': {'type': 'float', 'min': 0.001, 'max': 0.5, 'log_scale': True},
    'start_batch_percent': {'type': 'float', 'min': 0.0, 'max': 5.0},
    'lr_decay_type': {'type': 'categorical',
                      'values': ['linear-drop', 'exp-drop', 'log-drop', 'step-down']},
    'num_batches': {'type': 'int', 'min': 1, 'max': 20},
}


@pytest.fixture(autouse=True)
def _seeded():
    random.seed(1234)


def assert_in_space(config):
    assert config['map_size'][0] == config['map_size'][1]
    assert 8 <= config['map_size'][0] <= 14
    assert 0.5 <= config['start_learning_rate'] <= 1.0
    assert 0.001 <= config['end_learning_rate'] <= 0.5
    assert 0.0 <= config['start_batch_percent'] <= 5.0
    assert config['lr_decay_type'] in SPACE['lr_decay_type']['values']
    assert isinstance(config['num_batches'], int)
    assert 1 <= config['num_batches'] <= 20


# ─── random_config_continuous ─────────────────────────────────────────────────

class TestRandomConfig:
    def test_bounds_and_types(self):
        for _ in range(100):
            assert_in_space(random_config_continuous(SPACE))

    def test_log_scale_sampling_is_log_uniform(self):
        # end_learning_rate spans [0.001, 0.5]: a log-uniform median sits near
        # sqrt(0.001 * 0.5) ≈ 0.022, a linear-uniform median near 0.25.
        values = [random_config_continuous(SPACE)['end_learning_rate']
                  for _ in range(2000)]
        values.sort()
        median = values[len(values) // 2]
        assert median < 0.1, f"median {median} suggests linear sampling"

    def test_fixed_value_passthrough(self):
        space = dict(SPACE, fixed_thing='keep-me')
        config = random_config_continuous(space)
        assert config['fixed_thing'] == 'keep-me'

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match='Unknown parameter type'):
            random_config_continuous({'x': {'type': 'mystery', 'min': 0, 'max': 1}})


# ─── sbx_crossover ────────────────────────────────────────────────────────────

class TestSbxCrossover:
    def test_children_within_bounds(self):
        for _ in range(500):
            c1, c2 = sbx_crossover(0.6, 0.9, eta=20.0, bounds=(0.5, 1.0))
            assert 0.5 <= c1 <= 1.0
            assert 0.5 <= c2 <= 1.0

    def test_identical_parents_unchanged(self):
        for _ in range(50):
            c1, c2 = sbx_crossover(0.7, 0.7, eta=20.0, bounds=(0.0, 1.0))
            assert c1 == 0.7 and c2 == 0.7

    def test_sum_preserved_when_unclipped(self):
        # SBX property: child1 + child2 == parent1 + parent2 (when no clipping)
        for _ in range(200):
            c1, c2 = sbx_crossover(0.4, 0.6, eta=20.0, bounds=(-100.0, 100.0))
            assert c1 + c2 == pytest.approx(1.0)

    def test_log_scale_children_within_bounds(self):
        for _ in range(500):
            c1, c2 = sbx_crossover(0.002, 0.4, eta=20.0,
                                   bounds=(0.001, 0.5), log_scale=True)
            assert 0.001 <= c1 <= 0.5
            assert 0.001 <= c2 <= 0.5


# ─── polynomial_mutation ──────────────────────────────────────────────────────

class TestPolynomialMutation:
    def test_within_bounds(self):
        for _ in range(500):
            m = polynomial_mutation(0.7, eta=20.0, bounds=(0.5, 1.0),
                                    mutation_prob=1.0)
            assert 0.5 <= m <= 1.0

    def test_prob_zero_is_identity(self):
        assert polynomial_mutation(0.7, bounds=(0.0, 1.0), mutation_prob=0.0) == 0.7

    def test_missing_bounds_raise(self):
        with pytest.raises(ValueError):
            polynomial_mutation(0.7, bounds=None, mutation_prob=1.0)

    def test_degenerate_bounds_return_value(self):
        assert polynomial_mutation(0.7, bounds=(0.7, 0.7), mutation_prob=1.0) == 0.7

    def test_log_scale_within_bounds(self):
        for _ in range(500):
            m = polynomial_mutation(0.01, eta=20.0, bounds=(0.001, 0.5),
                                    mutation_prob=1.0, log_scale=True)
            assert 0.001 <= m <= 0.5


# ─── crossover_mixed / mutate_mixed ───────────────────────────────────────────

class TestMixedOperators:
    def test_crossover_children_complete_and_bounded(self):
        p1 = random_config_continuous(SPACE)
        p2 = random_config_continuous(SPACE)
        for _ in range(100):
            c1, c2 = crossover_mixed(p1, p2, SPACE, eta=20.0)
            assert set(c1) == set(SPACE)
            assert set(c2) == set(SPACE)
            assert_in_space(c1)
            assert_in_space(c2)

    def test_crossover_categorical_comes_from_parents(self):
        p1 = dict(random_config_continuous(SPACE), lr_decay_type='linear-drop')
        p2 = dict(random_config_continuous(SPACE), lr_decay_type='step-down')
        for _ in range(50):
            c1, c2 = crossover_mixed(p1, p2, SPACE, eta=20.0)
            assert {c1['lr_decay_type'], c2['lr_decay_type']} == \
                   {'linear-drop', 'step-down'}

    def test_crossover_skips_comment_and_fixed_keys(self):
        space = dict(SPACE, comment='ignore me', fixed_thing='keep-me')
        p1 = random_config_continuous(space)
        p2 = random_config_continuous(space)
        c1, c2 = crossover_mixed(p1, p2, space, eta=20.0)
        assert 'comment' not in c1 and 'fixed_thing' not in c1
        assert set(c1) == set(SPACE)

    def test_mutate_within_bounds_when_forced(self):
        config = random_config_continuous(SPACE)
        for _ in range(100):
            m = mutate_mixed(config, SPACE, eta=20.0, mutation_prob=1.0)
            assert_in_space(m)

    def test_mutate_prob_zero_is_identity(self):
        # Values from random_config_continuous are already rounded to the
        # operator precision (2 / 4 decimals), so prob=0 must be a no-op.
        config = random_config_continuous(SPACE)
        assert mutate_mixed(config, SPACE, eta=20.0, mutation_prob=0.0) == config

    def test_mutate_does_not_mutate_input(self):
        config = random_config_continuous(SPACE)
        snapshot = dict(config)
        mutate_mixed(config, SPACE, eta=20.0, mutation_prob=1.0)
        assert config == snapshot
