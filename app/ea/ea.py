import numpy as np
import argparse
import os
import random
import copy
import json
import psutil
import csv
import hashlib
import time
import pandas as pd
import sys
import multiprocessing
import shutil
import warnings

try:
    from pymoo.indicators.hv import HV as _PyMooHV
    _PYMOO_AVAILABLE = True
except ImportError:
    _PYMOO_AVAILABLE = False

# Suppress resource tracker warnings on Python 3.14+
# These warnings appear because Python 3.14 has stricter resource tracking
# The semaphores are actually cleaned up properly at exit
warnings.filterwarnings('ignore', category=UserWarning, module='multiprocessing.resource_tracker')

# Monkey-patch warnings.warn to filter resource_tracker messages
_original_warn = warnings.warn
def _filtered_warn(message, category=UserWarning, stacklevel=1):
    """Filter out resource_tracker warnings while preserving others"""
    msg_str = str(message)
    if 'resource_tracker' not in msg_str and 'loky' not in msg_str:
        _original_warn(message, category, stacklevel + 1)

warnings.warn = _filtered_warn

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datetime import datetime
from sklearn.datasets import make_blobs
from multiprocessing import Pool, cpu_count

from som.preprocess import validate_input_data, preprocess_data
from som.som import KohonenSOM
from som.graphs import generate_training_plots
from som.visualization import generate_individual_maps
from ea.genetic_operators import (
    random_config_continuous,
    crossover_mixed,
    mutate_mixed
)
from ea.nn_integration import NeuralNetworkIntegration

# Global variables
INPUT_FILE = None
NORMALIZED_DATA = None
WORKING_DIR = None
EVALUATED_CACHE = {}  # Cache for evaluated individuals: {uid: (training_results, config)}
EVALUATION_STATS = {'total_requested': 0, 'cache_hits': 0, 'new_evaluations': 0}  # Track deduplication stats
_NN_INTEGRATION_CACHE = {}  # Per-process NN model cache: {cache_key: NeuralNetworkIntegration}
# Running min/max for objective normalization — updated each generation from all feasible solutions.
# Shape (3,): [mqe_ratio, topo_error, dead_ratio]. None until first update.
_OBJ_RUNNING_MIN = None
_OBJ_RUNNING_MAX = None


def _get_nn_integration(nn_config: dict) -> NeuralNetworkIntegration:
    """
    Return a cached NeuralNetworkIntegration for this process.
    Models are loaded once per subprocess and reused across evaluations.
    """
    global _NN_INTEGRATION_CACHE
    cache_key = (
        nn_config.get('use_mlp'), nn_config.get('use_lstm'), nn_config.get('use_cnn'),
        nn_config.get('use_lstm_controller'),
        nn_config.get('mlp_model_path'), nn_config.get('lstm_model_path'),
        nn_config.get('lstm_scaler_path'), nn_config.get('cnn_model_path'),
        nn_config.get('lstm_controller_model_path'),
    )
    if cache_key not in _NN_INTEGRATION_CACHE:
        _NN_INTEGRATION_CACHE[cache_key] = NeuralNetworkIntegration(
            use_mlp=nn_config.get('use_mlp', False),
            use_lstm=nn_config.get('use_lstm', False),
            use_cnn=nn_config.get('use_cnn', False),
            use_lstm_controller=nn_config.get('use_lstm_controller', False),
            mlp_model_path=nn_config.get('mlp_model_path'),
            mlp_scaler_path=nn_config.get('mlp_scaler_path'),
            lstm_model_path=nn_config.get('lstm_model_path'),
            lstm_scaler_path=nn_config.get('lstm_scaler_path'),
            cnn_model_path=nn_config.get('cnn_model_path'),
            lstm_controller_model_path=nn_config.get('lstm_controller_model_path'),
            lstm_controller_scaler_path=nn_config.get('lstm_controller_scaler_path'),
            verbose=nn_config.get('verbose', False),
        )
    return _NN_INTEGRATION_CACHE[cache_key]


def validate_and_repair(config: dict) -> dict:
    repaired_config = config.copy()

    # Ensure start_learning_rate >= end_learning_rate (LR should decrease during training)
    if 'start_learning_rate' in repaired_config and 'end_learning_rate' in repaired_config:
        if repaired_config['start_learning_rate'] < repaired_config['end_learning_rate']:
            repaired_config['start_learning_rate'], repaired_config['end_learning_rate'] = \
                repaired_config['end_learning_rate'], repaired_config['start_learning_rate']

    # Ensure end_batch_percent >= start_batch_percent (batch size should grow during training)
    if 'start_batch_percent' in repaired_config and 'end_batch_percent' in repaired_config:
        if repaired_config['start_batch_percent'] > repaired_config['end_batch_percent']:
            repaired_config['start_batch_percent'], repaired_config['end_batch_percent'] = \
                repaired_config['end_batch_percent'], repaired_config['start_batch_percent']

    # Ensure start_radius >= end_radius (radius should decrease during training)
    if 'start_radius' in repaired_config and 'end_radius' in repaired_config:
        if repaired_config['start_radius'] < repaired_config['end_radius']:
            repaired_config['start_radius'], repaired_config['end_radius'] = \
                repaired_config['end_radius'], repaired_config['start_radius']

    if 'epoch_multiplier' in repaired_config:
        repaired_config['epoch_multiplier'] = max(0.1, float(repaired_config['epoch_multiplier']))

    # Set growth_g = 0 ONLY if ALL decay/growth types are linear (where it's not used at all)
    # This prevents different individuals that are functionally identical
    all_linear = True

    # Check if any non-linear decay/growth type is used
    if 'lr_decay_type' in repaired_config and repaired_config['lr_decay_type'] != 'linear-drop':
        all_linear = False
    if 'radius_decay_type' in repaired_config and repaired_config['radius_decay_type'] != 'linear-drop':
        all_linear = False
    if 'batch_growth_type' in repaired_config and repaired_config['batch_growth_type'] != 'linear-growth':
        all_linear = False

    if all_linear:
        repaired_config['growth_g'] = 0
    else:
        # Ensure growth_g is at least 1.0 when it's actually used by non-linear curves
        if 'growth_g' in repaired_config:
            repaired_config['growth_g'] = max(1.0, repaired_config['growth_g'])

    # Ensure num_batches is at least 1
    if 'num_batches' in repaired_config:
        repaired_config['num_batches'] = max(1, int(repaired_config['num_batches']))

    return repaired_config

def _dead_neuron_threshold(map_m: int, map_n: int, n_samples: int) -> float:
    """
    Dynamic dead neuron threshold from map/dataset coverage ratio.
    High coverage (small map, many samples/neuron) → strict threshold (few dead expected).
    Low coverage (large map, few samples/neuron) → lenient threshold (many dead expected).
    Formula: clamp(1 - coverage_ratio/10, 0.30, 0.85)
    Examples (n=569): 8x8→0.30, 12x12→0.60, 14x14→0.71, 20x20→0.85
    """
    coverage_ratio = n_samples / max(1, map_m * map_n)
    return float(np.clip(1.0 - coverage_ratio / 10.0, 0.3, 0.85))


def compute_constraint_violation(results: dict, n_samples: int, org_threshold: float = 1.0) -> float:
    """
    Scalar constraint violation for constrained NSGA-II dominance (Deb 2002).
    Returns 0.0 for feasible solutions, >0 for infeasible.
    Organization violation: max(u_matrix_max, distance_map_max) - org_threshold.
    Dead neuron violation: graduated penalty above dataset-size-calibrated threshold.
    """
    u_matrix_max = results.get('u_matrix_max') or 0.0
    distance_map_max = results.get('distance_map_max') or 0.0
    org_cv = max(0.0, max(u_matrix_max, distance_map_max) - org_threshold)

    dead_ratio = results.get('dead_neuron_ratio') or 0.0
    m = results.get('map_m', 10)
    n_map = results.get('map_n', 10)
    dead_threshold = _dead_neuron_threshold(m, n_map, n_samples)  # m, n_map = map dimensions
    dead_excess = max(0.0, dead_ratio - dead_threshold)
    if dead_excess == 0.0:
        dead_cv = 0.0
    elif dead_excess < 0.2:
        dead_cv = dead_excess * 1.5
    elif dead_excess < 0.4:
        dead_cv = dead_excess * 2.5
    else:
        dead_cv = dead_excess * 5.0

    return org_cv + dead_cv


def _dominates(objectives: np.ndarray, violations, p: int, q: int) -> bool:
    """
    Constrained dominance (Deb 2002):
    - feasible always dominates infeasible
    - between two infeasible: less violation dominates
    - between two feasible: standard Pareto dominance
    Falls back to standard Pareto when violations is None.
    """
    if violations is not None:
        cv_p, cv_q = violations[p], violations[q]
        feasible_p = cv_p < 1e-9
        feasible_q = cv_q < 1e-9
        if feasible_p and not feasible_q:
            return True
        if not feasible_p and feasible_q:
            return False
        if not feasible_p and not feasible_q:
            return cv_p < cv_q
    p_obj, q_obj = objectives[p], objectives[q]
    return bool(np.all(p_obj <= q_obj) and np.any(p_obj < q_obj))


def non_dominated_sort(objectives: np.ndarray, violations: np.ndarray = None) -> list:
    """
    Perform non-dominated sorting for NSGA-II.
    When violations is provided, uses constrained dominance (Deb 2002).

    Args:
        objectives: Array of objective values for all individuals.
        violations: Optional array of constraint violation scalars (0 = feasible).

    Returns:
        List of fronts, each containing indices of individuals.
    """
    n_individuals, n_objectives = objectives.shape

    domination_count = np.zeros(n_individuals, dtype=int)
    dominated_solutions = [[] for _ in range(n_individuals)]

    for p in range(n_individuals):
        for q in range(p + 1, n_individuals):
            if _dominates(objectives, violations, p, q):
                dominated_solutions[p].append(q)
                domination_count[q] += 1
            elif _dominates(objectives, violations, q, p):
                dominated_solutions[q].append(p)
                domination_count[p] += 1

    fronts = [[]]
    for p in range(n_individuals):
        if domination_count[p] == 0:
            fronts[0].append(p)

    front_index = 0
    # The loop will run as long as the last created front contains any individuals
    while fronts[front_index]:
        next_front = []
        # Process individuals in the current (last created) front
        for p in fronts[front_index]:
            # Process all that dominate
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                # If no one else dominates individual q, it belongs to the next front
                if domination_count[q] == 0:
                    next_front.append(q)

        # Increase the index for the next iteration
        front_index += 1

        # Add the new front to the list of fronts.
        # If it was empty, the loop will end in the next iteration.
        fronts.append(next_front)

    # The last added front will always be empty, so we remove it
    return fronts[:-1]


def crowding_distance_assignment(objectives: np.ndarray, fronts: list) -> np.ndarray:
    """
    Calculate crowding distance for each individual to maintain diversity.

    Args:
        objectives: Same array as in non_dominated_sort.
        fronts: Output from non_dominated_sort.

    Returns:
        Numpy array with crowding distance values for each individual.
    """
    n_individuals, n_objectives = objectives.shape
    crowding_distances = np.zeros(n_individuals)

    for front in fronts:
        if not front:
            continue

        front_objectives = objectives[front, :]
        n_front_members = len(front)

        for m in range(n_objectives):
            # Sort individuals in the front by the current objective
            sorted_indices = np.argsort(front_objectives[:, m])

            # Extreme points have infinite distance to be always preferred
            crowding_distances[front[sorted_indices[0]]] = np.inf
            crowding_distances[front[sorted_indices[-1]]] = np.inf

            if n_front_members > 2:
                # Normalization factor
                obj_range = front_objectives[sorted_indices[-1], m] - front_objectives[sorted_indices[0], m]
                if obj_range < 1e-8:  # Avoid division by zero
                    continue

                # For other points
                for i in range(1, n_front_members - 1):
                    dist = front_objectives[sorted_indices[i + 1], m] - front_objectives[sorted_indices[i - 1], m]
                    crowding_distances[front[sorted_indices[i]]] += dist / obj_range

    return crowding_distances


def tournament_selection(population: list, k: int = 3) -> dict:
    """
    Tournament selection based on rank and crowding distance.

    Args:
        population: List of individuals (dicts) with 'rank' and 'crowding_distance'.
        k: Tournament size.

    Returns:
        Winning individual (dict).
    """
    # Randomly select tournament participants
    participants = random.sample(population, k)

    best_participant = participants[0]

    for i in range(1, k):
        p = participants[i]
        # Prefer better (lower) rank
        if p['rank'] < best_participant['rank']:
            best_participant = p
        # With the same rank, prefer larger crowding distance for diversity
        elif p['rank'] == best_participant['rank'] and \
                p['crowding_distance'] > best_participant['crowding_distance']:
            best_participant = p

    return best_participant


_HV_REFERENCE = np.array([1.1, 1.1, 1.1])


def _update_obj_running_stats(raw_objectives: np.ndarray):
    """
    Update global running min/max from a batch of raw objective vectors.
    Called once per generation with all feasible solutions evaluated so far.
    Shape: (N, 3).
    """
    global _OBJ_RUNNING_MIN, _OBJ_RUNNING_MAX
    if len(raw_objectives) == 0:
        return
    batch_min = raw_objectives.min(axis=0)
    batch_max = raw_objectives.max(axis=0)
    if _OBJ_RUNNING_MIN is None:
        _OBJ_RUNNING_MIN = batch_min.copy()
        _OBJ_RUNNING_MAX = batch_max.copy()
    else:
        _OBJ_RUNNING_MIN = np.minimum(_OBJ_RUNNING_MIN, batch_min)
        _OBJ_RUNNING_MAX = np.maximum(_OBJ_RUNNING_MAX, batch_max)


def _normalize_objectives(raw_objectives: np.ndarray) -> np.ndarray:
    """
    Normalize objectives to [0, 1] using global running min/max, then clip to [0, 1.1].
    Falls back to plain clip if running stats not available yet.
    """
    if _OBJ_RUNNING_MIN is None:
        return np.clip(raw_objectives, 0.0, 1.1)
    span = _OBJ_RUNNING_MAX - _OBJ_RUNNING_MIN
    # Avoid division by zero for degenerate dimensions (all values equal)
    span = np.where(span < 1e-9, 1.0, span)
    normalized = (raw_objectives - _OBJ_RUNNING_MIN) / span
    return np.clip(normalized, 0.0, 1.1)


def _compute_pareto_metrics(front_objectives: np.ndarray) -> dict:
    """
    Compute per-generation Pareto quality metrics from the non-dominated front.

    front_objectives must already be normalized via _normalize_objectives().
    Reference point is fixed at [1.1, 1.1, 1.1] in normalized space.

    Returns dict with keys: front_size, hv, spacing, spread_mqe, spread_te, spread_dead.
    hv=None if pymoo not available; spacing/spread=None if front has <2 solutions.
    """
    n = len(front_objectives)
    result = {
        'front_size': n,
        'hv': None,
        'spacing': None,
        'spread_mqe': None,
        'spread_te': None,
        'spread_dead': None,
    }

    if n == 0:
        return result

    obj = front_objectives  # already normalized + clipped

    if _PYMOO_AVAILABLE:
        try:
            ind = _PyMooHV(ref_point=_HV_REFERENCE)
            result['hv'] = float(ind.do(obj))
        except Exception:
            pass

    if n >= 2:
        # Spacing: sqrt(mean((d_bar - d_i)^2)) where d_i = nearest-neighbor distance
        dists = []
        for i in range(n):
            others = np.delete(obj, i, axis=0)
            nn_dist = float(np.min(np.linalg.norm(others - obj[i], axis=1)))
            dists.append(nn_dist)
        d = np.array(dists)
        result['spacing'] = float(np.sqrt(np.mean((d.mean() - d) ** 2)))

        # Maximum Spread per objective dimension — how much of the objective space is covered
        result['spread_mqe']  = float(obj[:, 0].max() - obj[:, 0].min())
        result['spread_te']   = float(obj[:, 1].max() - obj[:, 1].min())
        result['spread_dead'] = float(obj[:, 2].max() - obj[:, 2].min())

    return result


def _log_pareto_metrics(generation: int, metrics: dict):
    """Write per-generation HV + Spacing + Spread summary to pareto_metrics.csv."""
    global WORKING_DIR
    csv_path = os.path.join(WORKING_DIR, "pareto_metrics.csv")
    file_exists = os.path.isfile(csv_path)
    fieldnames = ['generation', 'front_size', 'hv', 'spacing', 'spread_mqe', 'spread_te', 'spread_dead']

    def _fmt(v):
        return round(v, 6) if v is not None else ''

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'generation':  generation + 1,
            'front_size':  metrics['front_size'],
            'hv':          _fmt(metrics['hv']),
            'spacing':     _fmt(metrics['spacing']),
            'spread_mqe':  _fmt(metrics['spread_mqe']),
            'spread_te':   _fmt(metrics['spread_te']),
            'spread_dead': _fmt(metrics['spread_dead']),
        })


def log_pareto_front(generation: int, search_space: dict):
    """
    Append the current Pareto archive to pareto_front.csv — one row per (generation, solution).
    Columns: generation, uid, raw_mqe_ratio, raw_te, dead_ratio,
             is_penalized, penalty_factor, penalty_reason,
             initial_mqe, pen_mqe_ratio, pen_te,
             u_matrix_mean, u_matrix_std, u_matrix_max, distance_map_max,
             duration, <search_space_params...>
    """
    global WORKING_DIR
    global ARCHIVE

    csv_path = os.path.join(WORKING_DIR, "pareto_front.csv")
    file_exists = os.path.isfile(csv_path)

    sorted_archive = sorted(
        ARCHIVE,
        key=lambda x: x[1].get('raw_mqe_improvement_ratio') or x[1].get('raw_best_mqe') or x[1]['best_mqe']
    )

    # Compute and log per-generation HV + Spacing + Spread from feasible front solutions.
    # Running min/max is updated first so normalization covers the full observed range.
    feasible_raw = np.array([
        [
            res.get('raw_mqe_improvement_ratio', 1.0) or 1.0,
            res.get('raw_topographic_error', res.get('topographic_error', 1.0)),
            res.get('dead_neuron_ratio', 1.0),
        ]
        for _, res in ARCHIVE
        if not res.get('is_penalized', False)
    ]) if ARCHIVE else np.empty((0, 3))
    _update_obj_running_stats(feasible_raw)
    feasible_norm = _normalize_objectives(feasible_raw) if len(feasible_raw) > 0 else feasible_raw
    _log_pareto_metrics(generation, _compute_pareto_metrics(feasible_norm))

    search_keys = sorted(search_space.keys())
    # Collect ds_* keys present in any archive result
    ds_keys = sorted({k for _, res in ARCHIVE for k in res if k.startswith('ds_')})
    fieldnames = [
        'generation', 'uid', 'dataset_name',
        'raw_mqe_ratio', 'raw_te', 'dead_ratio',
        'constraint_violation', 'is_penalized', 'penalty_factor', 'penalty_reason',
        'initial_mqe', 'pen_mqe_ratio', 'pen_te',
        'map_m', 'map_n',
        'u_matrix_mean', 'u_matrix_std', 'u_matrix_max', 'distance_map_max',
        'duration',
    ] + ds_keys + search_keys

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()

        for config, results in sorted_archive:
            row = {
                'generation':          generation + 1,
                'uid':                 results['uid'],
                'dataset_name':        results.get('dataset_name'),
                'raw_mqe_ratio':       results.get('raw_mqe_improvement_ratio'),
                'raw_te':              results.get('raw_topographic_error', results.get('topographic_error')),
                'dead_ratio':          results.get('dead_neuron_ratio'),
                'constraint_violation': results.get('constraint_violation', 0.0),
                'is_penalized':        results.get('is_penalized', False),
                'penalty_factor':      results.get('penalty_factor', 1.0),
                'penalty_reason':      results.get('penalty_reason', ''),
                'initial_mqe':         results.get('initial_mqe'),
                'pen_mqe_ratio':       results.get('mqe_improvement_ratio'),
                'pen_te':              results.get('topographic_error'),
                'map_m':               results.get('map_m'),
                'map_n':               results.get('map_n'),
                'u_matrix_mean':       results.get('u_matrix_mean'),
                'u_matrix_std':        results.get('u_matrix_std'),
                'u_matrix_max':        results.get('u_matrix_max'),
                'distance_map_max':    results.get('distance_map_max'),
                'duration':            results.get('training_duration', results.get('duration')),
            }
            for k in ds_keys:
                row[k] = results.get(k)
            for k in search_keys:
                row[k] = config.get(k)
            writer.writerow(row)


def get_working_directory(input_file: str = None) -> str:
    """
    Create and return a working directory for results, named by timestamp.
    """
    if input_file:
        base_dir = os.path.dirname(os.path.abspath(input_file))
    else:
        base_dir = os.getcwd()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = os.path.join(base_dir, "results", f"{timestamp}")

    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir

_EM_MAX_ANCHORS = [(100, 5.0), (500, 5.0), (5_000, 3.0), (20_000, 1.0), (50_000, 0.3)]
_EM_MIN_ANCHORS = [(100, 1.0), (500, 1.0), (5_000, 0.5), (20_000, 0.3), (50_000, 0.1)]


def _interp_log(n: int, anchors: list) -> float:
    """Log-linear interpolation between (n, value) anchor points."""
    import math
    if n <= anchors[0][0]:
        return anchors[0][1]
    if n >= anchors[-1][0]:
        return anchors[-1][1]
    for (n1, v1), (n2, v2) in zip(anchors, anchors[1:]):
        if n1 <= n <= n2:
            t = math.log(n / n1) / math.log(n2 / n1)
            return round(v1 * (v2 / v1) ** t, 2)


def apply_dynamic_search_space(search_space: dict, n_samples: int) -> dict:
    """
    Adjust map_size and epoch_multiplier bounds from dataset size.

    map_size:
      Vesanto heuristic U = 5*sqrt(n_samples); side range [0.7*sqrt(U), 1.3*sqrt(U)].

    epoch_multiplier:
      Log-linear interpolation between empirical anchors.
      em_max anchors: (100→5.0), (500→5.0), (5000→3.0), (20000→1.0), (50000→0.3)
      em_min anchors: (100→1.0), (500→1.0), (5000→0.5), (20000→0.3), (50000→0.1)

    Returns a modified copy — original is not mutated.
    """
    import math

    # --- map_size ---
    U = 5.0 * math.sqrt(n_samples)
    optimal_side = math.sqrt(U)
    map_new_min = max(8, round(optimal_side * 0.7))
    map_new_max = max(map_new_min + 2, round(optimal_side * 1.3))

    # --- epoch_multiplier ---
    em_max = _interp_log(n_samples, _EM_MAX_ANCHORS)
    em_min = _interp_log(n_samples, _EM_MIN_ANCHORS)
    if em_min >= em_max:
        em_min = round(em_max * 0.5, 2)

    adjusted = {}
    for key, spec in search_space.items():
        if key == 'map_size' and isinstance(spec, dict) and spec.get('type') == 'discrete_int_pair':
            new_spec = dict(spec)
            new_spec['min'] = map_new_min
            new_spec['max'] = map_new_max
            adjusted[key] = new_spec
            print(f"INFO: Dynamic map_size bounds [{map_new_min}, {map_new_max}] "
                  f"(optimal_side={optimal_side:.1f}, U={U:.0f}, n_samples={n_samples})")
        elif key == 'epoch_multiplier' and isinstance(spec, dict):
            new_spec = dict(spec)
            new_spec['min'] = em_min
            new_spec['max'] = em_max
            adjusted[key] = new_spec
            print(f"INFO: Dynamic epoch_multiplier bounds [{em_min}, {em_max}] "
                  f"(n_samples={n_samples})")
        else:
            adjusted[key] = spec
    return adjusted


def load_configuration(json_path: str = None) -> dict:
    """
        Load configuration. Priority: JSON > ea_config.py.

        Args:
            json_path: Path to JSON config file.

        Returns:
            Configuration dictionary.
        """
    if json_path:
        print(f"INFO: Using configuration file from argument: {json_path}")
        if not os.path.exists(json_path):
            print(f"ERROR: Configuration file {json_path} does not exist.")
            sys.exit(1)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: File {json_path} is not valid JSON: {e}")
            sys.exit(1)
    else:
        print("INFO: Argument --config not provided, searching for ea_config.py.")
        try:
            from ea_config import CONFIG
            print("INFO: ea_config.py found and loaded.")
            return CONFIG
        except ImportError:
            print("ERROR: No configuration file provided and default ea_config.py not found.")
            print("Solution: Create ea_config.py or use --config <path_to_json> as described in the documentation.")
            sys.exit(1)


def crossover(parent1: dict, parent2: dict, param_space: dict) -> dict:
    """
    Perform crossover between two parents.

    Args:
        parent1: First parent configuration.
        parent2: Second parent configuration.
        param_space: Search space dictionary.

    Returns:
        Child configuration dictionary.
    """
    child = {}
    for key in param_space:
        if isinstance(param_space[key], list):
            child[key] = random.choice([parent1[key], parent2[key]])
        else:
            child[key] = parent1[key]
    return child

def random_config(param_space: dict) -> dict:
    """
    Generate a random configuration from the search space.

    Args:
        param_space: Search space dictionary.

    Returns:
        Configuration dictionary.
    """
    config = {}
    for key, value in param_space.items():
        if isinstance(value, list):
            config[key] = random.choice(value)
        else:
            config[key] = value

    # Validate and fix min/max batch
    if 'min_batch_percent' in config and 'max_batch_percent' in config:
        if config['min_batch_percent'] > config['max_batch_percent']:
            # Simplest fix: swap values, variant generate until correct order
            config['min_batch_percent'], config['max_batch_percent'] = \
                config['max_batch_percent'], config['min_batch_percent']
    return config

def mutate(config: dict, param_space: dict) -> dict:
    """
    Mutate a configuration by randomly changing one parameter.

    Args:
        config: Configuration dictionary.
        param_space: Search space dictionary.

    Returns:
        Mutated configuration dictionary.
    """
    key = random.choice(list(param_space.keys()))
    if isinstance(param_space[key], list):
        config[key] = random.choice(param_space[key])
    return config

def _run_probe_worker(args: tuple):
    """
    Lightweight SOM training for calibration probe — no logging, no plots.
    Returns max(u_matrix_max, dist_map_max) or None on failure.
    """
    config, fixed_params, data, ignore_mask, probe_dir = args
    try:
        os.makedirs(probe_dir, exist_ok=True)
        som_params = {**config, **fixed_params}
        for k in ['sample_size', 'input_dim', 'rank', 'crowding_distance', 'comment',
                  'nn_config', 'org_threshold', 'max_archive_size']:
            som_params.pop(k, None)
        som_params['save_checkpoints'] = False
        som = KohonenSOM(dim=data.shape[1], **som_params)
        som.train(data, ignore_mask=ignore_mask, working_dir=probe_dir)
        u_metrics = som.calculate_u_matrix_metrics()
        distance_map, _ = som.compute_quantization_error(data, mask=ignore_mask)
        dist_max = float(np.max(distance_map)) if distance_map is not None else 0.0
        u_max = float(u_metrics.get('u_matrix_max', 0.0))
        return max(u_max, dist_max)
    except Exception:
        return None


def run_calibration_probe(ea_config: dict, data: np.ndarray, ignore_mask: np.ndarray,
                          working_dir: str) -> float:
    """
    Run N quick SOM trainings before EA to calibrate the organization threshold.
    Uses 70th percentile of max(u_matrix_max, dist_map_max) across all probes.
    Config section CALIBRATION: {n_probes: 15, probe_epoch_multiplier: 0.3}.
    Returns calibrated org_threshold (falls back to 1.0 on failure).
    """
    cal_cfg = ea_config.get('CALIBRATION', {})
    n_probes = cal_cfg.get('n_probes', 15)
    probe_multiplier = cal_cfg.get('probe_epoch_multiplier', 0.3)

    if n_probes <= 0:
        print("INFO: Calibration probe disabled (n_probes=0), using org_threshold=1.0")
        return 1.0

    search_space = ea_config['SEARCH_SPACE']
    probe_fixed = dict(ea_config['FIXED_PARAMS'])
    probe_fixed.pop('nn_config', None)
    probe_fixed['epoch_multiplier'] = probe_multiplier

    print(f"\nINFO: Running {n_probes} calibration probes (epoch_multiplier={probe_multiplier})...")
    probes_base = os.path.join(working_dir, 'calibration_probes')

    configs = [validate_and_repair(random_config_continuous(search_space)) for _ in range(n_probes)]
    args_list = [
        (cfg, probe_fixed, data, ignore_mask, os.path.join(probes_base, str(i)))
        for i, cfg in enumerate(configs)
    ]

    with Pool(processes=min(n_probes, cpu_count(), 10)) as pool:
        org_values_raw = pool.map(_run_probe_worker, args_list)

    org_values = [v for v in org_values_raw if v is not None and v > 0]

    if not org_values:
        print("WARNING: All calibration probes failed, using org_threshold=1.0")
        return 1.0

    org_threshold = float(np.percentile(org_values, 70))

    probe_csv = os.path.join(working_dir, 'calibration_probe.csv')
    with open(probe_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['probe_idx', 'org_max'])
        for i, v in enumerate(org_values):
            writer.writerow([i + 1, round(v, 6)])

    print(f"INFO: Calibration org_max — min={min(org_values):.3f}, "
          f"median={float(np.median(org_values)):.3f}, "
          f"p70={org_threshold:.3f}, max={max(org_values):.3f}")
    print(f"INFO: ORGANIZATION_THRESHOLD set to {org_threshold:.3f} (70th percentile)")
    return org_threshold


def run_evolution(ea_config: dict, data: np.ndarray, ignore_mask: np.ndarray) -> None:
    """
    Main loop of the evolutionary algorithm with Pareto optimization (NSGA-II).

    """
    global ARCHIVE
    global WORKING_DIR
    global EVALUATED_CACHE
    global EVALUATION_STATS

    # Clear cache and stats at the start of each evolution run
    EVALUATED_CACHE.clear()
    EVALUATION_STATS = {'total_requested': 0, 'cache_hits': 0, 'new_evaluations': 0}

    population_size = ea_config["EA_SETTINGS"]["population_size"]
    generations = ea_config["EA_SETTINGS"]["generations"]

    fixed_params = ea_config["FIXED_PARAMS"]
    search_space = ea_config["SEARCH_SPACE"]

    # Inject NN config into fixed_params so evaluate_individual subprocesses can load models
    use_nn = ea_config.get("use_nn", False)
    nn_cfg = ea_config.get("NEURAL_NETWORKS", {})
    use_cnn_objective = False
    _any_nn = (nn_cfg.get("use_mlp") or nn_cfg.get("use_lstm") or
               nn_cfg.get("use_cnn") or nn_cfg.get("use_lstm_controller"))
    if use_nn and _any_nn:
        fixed_params = dict(fixed_params)  # don't mutate the original
        fixed_params['nn_config'] = {
            'use_mlp':              nn_cfg.get('use_mlp', False),
            'use_lstm':             nn_cfg.get('use_lstm', False),
            'use_cnn':              nn_cfg.get('use_cnn', False),
            'use_lstm_controller':  nn_cfg.get('use_lstm_controller', False),
            'mlp_model_path':       nn_cfg.get('mlp_model_path'),
            'mlp_scaler_path':      nn_cfg.get('mlp_scaler_path'),
            'lstm_model_path':      nn_cfg.get('lstm_model_path'),
            'lstm_scaler_path':     nn_cfg.get('lstm_scaler_path'),
            'cnn_model_path':       nn_cfg.get('cnn_model_path'),
            'lstm_controller_model_path':  nn_cfg.get('lstm_controller_model_path'),
            'lstm_controller_scaler_path': nn_cfg.get('lstm_controller_scaler_path'),
            'lstm_quality_threshold':    nn_cfg.get('lstm_quality_threshold', 1.0),
            'mlp_filter_bad_configs':    nn_cfg.get('mlp_filter_bad_configs', False),
            'mlp_bad_quality_threshold': nn_cfg.get('mlp_bad_quality_threshold', 0.5),
            'verbose':              nn_cfg.get('verbose', False),
        }
        use_cnn_objective = nn_cfg.get('use_cnn', False)
        print(f"INFO: NN integration enabled — MLP={nn_cfg.get('use_mlp')}, "
              f"LSTM={nn_cfg.get('use_lstm')}, CNN={nn_cfg.get('use_cnn')}, "
              f"LSTM-Controller={nn_cfg.get('use_lstm_controller')}")
    else:
        print("INFO: NN integration disabled — running EA without neural networks")

    # Get genetic operator parameters
    genetic_params = ea_config.get("GENETIC_OPERATORS", {})
    sbx_eta = genetic_params.get("sbx_eta", 20.0)
    mutation_eta = genetic_params.get("mutation_eta", 20.0)
    mutation_prob = genetic_params.get("mutation_prob", 0.1)
    # Tournament size: configurable, default scales with population (k=2 is Deb's original)
    tournament_k = genetic_params.get("tournament_k", max(2, population_size // 10))

    # Initialize population using continuous operators
    # Generate initial population and validate/repair constraints
    population = [validate_and_repair(random_config_continuous(search_space)) for _ in range(population_size)]

    try:
        for gen in range(generations):
            print(f"Generation {gen + 1}/{generations}")

            # Population evaluation (parallel)
            with Pool(processes=min(10, cpu_count(), len(population))) as pool:
                args_list = [(ind, i, gen, fixed_params, data, ignore_mask, WORKING_DIR) for i, ind in enumerate(population)]
                results_async = [pool.apply_async(evaluate_individual, args=arg) for arg in args_list]
                evaluated_population = []
                for r in results_async:
                    try:
                        training_results, config = r.get(timeout=3600)
                        evaluated_population.append((config, training_results))
                    except Exception as e:
                        print(f"[ERROR] Individual failed: {e}")

            if len(evaluated_population) == 0:
                print("Error: No individual was successfully evaluated. Exiting.")
                return

            # Combine population and archive (elitism)
            # Combine current evaluated population with the best individuals from previous generations
            combined_population = evaluated_population + ARCHIVE

            # Fitness calculation using Non-dominated Sorting and Crowding Distance
            # Use improvement ratios (final/initial) instead of absolute values so that
            # results are comparable across different map sizes and datasets.
            # ratio < 1 = improvement, ratio > 1 = worse than random init (penalized maps).
            # Fall back to absolute values when ratios are unavailable (e.g. no checkpoints).
            # mqe_improvement_ratio normalizes MQE across map sizes and datasets.
            # topographic_error and dead_neuron_ratio are already in [0,1] — no normalization needed.
            # NSGA-II objectives use RAW values (before penalty) so dominance reflects true quality.
            # Penalized solutions are still in the population for genetic diversity but their raw
            # quality metrics determine ranking. Penalty metadata is logged separately.
            # Objectives: [raw_mqe_ratio, raw_topographic_error, dead_neuron_ratio]
            # Training duration is removed — it is dataset/config metadata, not a quality objective.
            def _mqe_obj(res):
                r = res.get('raw_mqe_improvement_ratio')
                if r is not None:
                    return r
                return res.get('raw_best_mqe', res['best_mqe'])

            if use_cnn_objective:
                objectives = np.array([
                    [
                        _mqe_obj(res),
                        res.get('raw_topographic_error', res.get('topographic_error', 1.0)),
                        res.get('dead_neuron_ratio', 1.0),
                        1.0 - (res.get('cnn_quality_score') or 0.0),
                    ]
                    for cfg, res in combined_population
                ])
            else:
                objectives = np.array([
                    [
                        _mqe_obj(res),
                        res.get('raw_topographic_error', res.get('topographic_error', 1.0)),
                        res.get('dead_neuron_ratio', 1.0),
                    ]
                    for cfg, res in combined_population
                ])

            # Constraint violation array for constrained dominance (Deb 2002)
            violations = np.array([
                res.get('constraint_violation', 0.0) or 0.0
                for cfg, res in combined_population
            ])

            # Sorting into Pareto fronts using constrained dominance
            fronts = non_dominated_sort(objectives, violations)

            # Calculate "distance to neighbors" to maintain diversity
            crowding_distances = crowding_distance_assignment(objectives, fronts)

            # Assign rank and crowding distance to each individual
            for i, front in enumerate(fronts):
                for individual_idx in front:
                    # Add fitness metrics directly to the configuration dictionary
                    combined_population[individual_idx][0]['rank'] = i
                    combined_population[individual_idx][0]['crowding_distance'] = crowding_distances[individual_idx]

            # Selection for next generation
            # Sort the combined population: primarily by rank (ascending), secondarily by crowding distance (descending)
            combined_population.sort(key=lambda x: (x[0]['rank'], -x[0]['crowding_distance']))

            # New population is the first N best individuals from the sorted list
            # Keep only the dictionaries with configuration, not results
            population = [cfg for cfg, res in combined_population[:population_size]]

            # Update archive - contains only individuals from the best front (rank 0).
            # NOTE: must be built from the sorted list using rank==0 filter, NOT from
            # fronts[0] indices — those index the pre-sort combined_population and are
            # stale after combined_population.sort() above.
            ARCHIVE = [(cfg, res) for cfg, res in combined_population if cfg.get('rank') == 0]

            # Deduplicate by UID — parallel workers may evaluate the same config multiple times
            # (EVALUATED_CACHE is not shared across processes). Keep the copy with the highest
            # crowding distance (most diverse position in objective space).
            seen_uids: set = set()
            deduped = []
            for cfg, res in sorted(ARCHIVE, key=lambda x: -x[0].get('crowding_distance', 0)):
                uid = res.get('uid')
                if uid not in seen_uids:
                    seen_uids.add(uid)
                    deduped.append((cfg, res))
            ARCHIVE = deduped

            max_archive = fixed_params.get('max_archive_size', 0)
            if max_archive > 0 and len(ARCHIVE) > max_archive:
                ARCHIVE.sort(key=lambda x: -x[0].get('crowding_distance', 0))
                ARCHIVE = ARCHIVE[:max_archive]
            print(f" Best Pareto front has {len(ARCHIVE)} solutions.")
            log_pareto_front(gen, search_space)  # Log the current best front

            # Reproduction (offspring creation)
            # Fill mating pool via tournament selection
            mating_pool = [tournament_selection(population, k=tournament_k)
                           for _ in range(population_size)]

            # Generate exactly population_size unique offspring.
            # If SBX/mutation produces a duplicate (already in this generation or archive),
            # retry with a new random pair — up to max_crossovers attempts.
            archive_uids = {get_uid(cfg) for cfg, _ in ARCHIVE}
            next_gen_offspring: list = []
            seen_offspring_uids: set = set(archive_uids)  # skip re-generating known archive members
            max_crossovers = population_size * 3
            crossover_attempts = 0

            while len(next_gen_offspring) < population_size and crossover_attempts < max_crossovers:
                crossover_attempts += 1
                p1_full = random.choice(mating_pool)
                p2_full = random.choice(mating_pool)

                p1_genes = {k: v for k, v in p1_full.items() if k in search_space}
                p2_genes = {k: v for k, v in p2_full.items() if k in search_space}

                child1, child2 = crossover_mixed(p1_genes, p2_genes, search_space, eta=sbx_eta)

                for child in (child1, child2):
                    if len(next_gen_offspring) >= population_size:
                        break
                    mutated = mutate_mixed(child, search_space, eta=mutation_eta, mutation_prob=mutation_prob)
                    repaired = validate_and_repair(mutated)
                    uid = get_uid(repaired)
                    if uid not in seen_offspring_uids:
                        seen_offspring_uids.add(uid)
                        next_gen_offspring.append(repaired)

            generated = len(next_gen_offspring)
            if generated < population_size:
                print(f"  INFO: Search space saturated — proceeding with {generated}/{population_size} "
                      f"unique offspring (no new unique configs found after {crossover_attempts} attempts)")

            # New population for the next iteration
            population = next_gen_offspring
    except KeyboardInterrupt:
        print("\nTerminating evolutionary algorithm...")
    except Exception as e:
        print(f"\nFatal error during execution of the evolutionary algorithm: {str(e)}")

    # Calculate deduplication statistics from results.csv
    # (Can't use EVALUATION_STATS due to multiprocessing - globals not shared between processes)
    csv_path = os.path.join(WORKING_DIR, "results.csv")
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        new_evaluations = len(df)  # Unique UIDs in results.csv
        total_requested = population_size * generations  # Total individual evaluations requested
        cache_hits = total_requested - new_evaluations  # Duplicates skipped
    except Exception as e:
        # Fallback to empty stats if file doesn't exist
        total_requested = 0
        new_evaluations = 0
        cache_hits = 0

    print("Evolution completed.")
    print(f"Deduplication stats: {new_evaluations} unique configurations evaluated, {cache_hits} duplicates skipped (from {total_requested} total requests)")
    if total_requested > 0:
        cache_hit_rate = (cache_hits / total_requested) * 100
        print(f"Cache hit rate: {cache_hit_rate:.1f}%")

def get_uid(config: dict) -> str:
    """
    Generate a unique identifier for a configuration.
    """
    config_str = str(sorted(config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()

def log_message(uid: str, message: str, working_dir: str = None) -> None:
    """
    Log a message to the log file.
    """
    if working_dir is None:
        global WORKING_DIR
        working_dir = WORKING_DIR
    log_path = os.path.join(working_dir, "log.txt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding='utf-8') as f:
        f.write(f"[{now}] [{uid}] {message}\n")

def log_result_to_csv(config: dict, results: dict, working_dir: str = None) -> None:
    """
    Log evaluation results to a CSV file.
    """
    if working_dir is None:
        global WORKING_DIR
        working_dir = WORKING_DIR
    csv_path = os.path.join(working_dir, "results.csv")
    uid = results['uid']

    file_exists = os.path.isfile(csv_path)
    base_fields = ['uid',
                   'dataset_name',
                   'raw_best_mqe', 'raw_topographic_error', 'raw_mqe_improvement_ratio',
                   'best_mqe', 'topographic_error', 'mqe_improvement_ratio',
                   'initial_mqe', 'constraint_violation', 'penalty_factor', 'is_penalized', 'penalty_reason',
                   'duration', 'dead_neuron_ratio', 'map_m', 'map_n',
                   'u_matrix_mean', 'u_matrix_std', 'u_matrix_max', 'distance_map_max',
                   'total_weight_updates', 'epochs_ran', 'dead_neuron_count',
                   'cnn_quality_score']
    ds_keys = sorted(k for k in results if k.startswith('ds_'))

    with open(csv_path, mode="a", newline="", encoding='utf-8') as f:
        row = {'uid': uid}
        for key in base_fields[1:]:
            row[key] = results.get(key)
        for key in ds_keys:
            row[key] = results.get(key)
        row.update(config)

        fieldnames = ['uid'] + [k for k in base_fields[1:]] + ds_keys + list(config.keys())

        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def log_progress(current: int, total: int) -> None:
    """
    Log progress to a file.
    """
    global WORKING_DIR
    progress_path = os.path.join(WORKING_DIR, "progress.log")
    with open(progress_path, "a", encoding='utf-8') as f:
        f.write(f"{current}/{total} completed\n")

def get_or_generate_data(sample_size: int, input_dim: int) -> np.ndarray:
    """
    Load or generate synthetic data for SOM training.

    Args:
        sample_size: Number of samples.
        input_dim: Number of features.

    Returns:
        Numpy array with data.
    """
    global WORKING_DIR
    file_name = f"data_{sample_size}x{input_dim}.npy"
    file_path = os.path.join(WORKING_DIR, file_name)

    if os.path.exists(file_path):
        return np.load(file_path)

    data, _ = make_blobs(n_samples=sample_size, n_features=input_dim, centers=5)
    np.save(file_path, data)
    log_message("SYSTEM", f"Generated new data: {file_name}")
    return data

def log_status_to_csv(uid: str, population_id: int, generation: int, status: str,
                     start_time: str = None, end_time: str = None, working_dir: str = None) -> None:
    """
    Log the status of an individual's evaluation to a CSV file.
    """
    if working_dir is None:
        global WORKING_DIR
        working_dir = WORKING_DIR
    csv_path = os.path.join(working_dir, "status.csv")
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, mode="a", newline="", encoding='utf-8') as f:
        fieldnames = ['uid', 'population_id', 'generation', 'status', 'start_time', 'end_time']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        
        row = {
            'uid': uid,
            'population_id': population_id,
            'generation': generation,
            'status': status,
            'start_time': start_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'end_time': end_time
        }
        writer.writerow(row)

def log_final_best(uid: str, config: dict, score: float, duration: float) -> None:
    """
    Log the best final result to a file.
    """
    global WORKING_DIR
    best_path = os.path.join(WORKING_DIR, "final_best.txt")
    with open(best_path, "a", encoding='utf-8') as f:
        f.write(f"UID: {uid}\n")
        f.write(f"Score (quantization error): {score:.6f}\n")
        f.write(f"Duration: {duration:.2f} s\n")
        f.write("Parameters:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")

def load_input_data(input_file: str) -> np.ndarray:
    sys.exit(0)
    """
    Load and normalize input data from a CSV file.

    Args:
        input_file: Path to input CSV file.

    Returns:
        Numpy array with normalized data.
    """
    global NORMALIZED_DATA
    global WORKING_DIR
    
    if NORMALIZED_DATA is not None:
        return NORMALIZED_DATA

    preprocess_file = os.path.join(WORKING_DIR, "preprocess-input.csv")
    # if not os.path.exists(preprocess_file):
    #     preprocess_file = normalize_data(input_file, {}, {})
    NORMALIZED_DATA = pd.read_csv(preprocess_file, delimiter=',').values
    log_message("SYSTEM", f"Loaded and normalized data from external file: {input_file}")
    return NORMALIZED_DATA

def extract_uid_from_path(file_path: str) -> str:
    """
    Extract UID from file path.
    """
    parts = file_path.split('/')
    for part in parts:
        if part.startswith('nxmpp'):
            return part
    return None


def copy_maps_to_dataset(uid: str, individual_dir: str, working_dir: str) -> None:
    """
    Copy generated maps (u_matrix, distance_map, dead_neurons_map) to centralized dataset directory.
    This creates a dataset for CNN training.

    Args:
        uid: Unique identifier for the individual
        individual_dir: Directory containing the individual's visualizations
        working_dir: Working directory for the EA run
    """
    maps_dataset_dir = os.path.join(working_dir, "maps_dataset")
    os.makedirs(maps_dataset_dir, exist_ok=True)

    # List of maps to copy
    map_files = ["u_matrix.png", "distance_map.png", "dead_neurons_map.png"]

    source_dir = os.path.join(individual_dir, "visualizations")

    for map_file in map_files:
        source_path = os.path.join(source_dir, map_file)
        if os.path.exists(source_path):
            # Create filename with UID prefix for uniqueness
            dest_filename = f"{uid}_{map_file}"
            dest_path = os.path.join(maps_dataset_dir, dest_filename)
            shutil.copy2(source_path, dest_path)


def evaluate_individual(ind: dict, population_id: int, generation: int,
                        fixed_params: dict, data: np.ndarray,
                        ignore_mask: np.ndarray, working_dir: str) -> tuple:
    """
    Evaluate a single individual (configuration) for SOM training.

    Args:
        ind: Individual configuration dictionary.
        population_id: Index in population.
        generation: Current generation number.
        fixed_params: Fixed parameters for SOM.
        data: Training data array.
        ignore_mask: Mask for ignored features.
        working_dir: Working directory path.

    Returns:
        Tuple of (training_results, configuration).
    """
    global EVALUATED_CACHE
    global EVALUATION_STATS

    EVALUATION_STATS['total_requested'] += 1

    start_time = time.monotonic()
    uid = get_uid(ind)

    # Check if this configuration has already been evaluated
    if uid in EVALUATED_CACHE:
        EVALUATION_STATS['cache_hits'] += 1
        cached_results, cached_config = EVALUATED_CACHE[uid]
        print(f"[GEN {generation + 1}] Skipping duplicate UID {uid[:8]}... (already evaluated)")
        log_status_to_csv(uid, population_id, generation, "cached",
                         start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), working_dir=working_dir)
        return cached_results, cached_config

    EVALUATION_STATS['new_evaluations'] += 1

    try:
        print(f"[GEN {generation + 1}] Total RAM used: {psutil.virtual_memory().used // (1024 ** 2)} MB")
        log_status_to_csv(uid, population_id, generation, "started",
                         start_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), working_dir=working_dir)

        individual_dir = os.path.join(working_dir, "individuals", uid)
        os.makedirs(individual_dir, exist_ok=True)

        # --- Neural Network integration ---
        nn_config = fixed_params.get('nn_config')
        nn = _get_nn_integration(nn_config) if nn_config else None

        som_params = {**ind, **fixed_params}

        # Remove non-SOM parameters
        som_params.pop('sample_size', None)
        som_params.pop('input_dim', None)
        som_params.pop('rank', None)  # Fitness metadata from NSGA-II
        som_params.pop('crowding_distance', None)  # Fitness metadata from NSGA-II
        som_params.pop('comment', None)  # Comment fields from config
        som_params.pop('nn_config', None)  # NN config is not a SOM param

        # MLP pre-screen: predict quality before full SOM training
        if nn is not None and nn.can_predict_fitness() and nn_config.get('mlp_filter_bad_configs', False):
            predicted = nn.predict_fitness(ind)
            if predicted is not None:
                pred_mqe = max(0.0, predicted[0])   # raw_mqe_improvement_ratio, higher=better
                threshold = nn_config.get('mlp_bad_quality_threshold', 0.5)
                if pred_mqe < threshold:             # skip when predicted improvement is LOW
                    log_message(uid, f"MLP pre-screen: predicted improvement={pred_mqe:.3f} < threshold={threshold:.3f}, skipping SOM training", working_dir)
                    penalty_results = {
                        'uid': uid, 'best_mqe': pred_mqe * 2.0, 'duration': 0.0,
                        'topographic_error': 1.0, 'dead_neuron_ratio': 1.0,
                        'dead_neuron_count': 0, 'training_duration': 0.0,
                        'u_matrix_mean': 0.0, 'u_matrix_std': 0.0, 'u_matrix_max': 0.0,
                        'distance_map_max': 0.0, 'total_weight_updates': 0, 'epochs_ran': 0,
                        'cnn_quality_score': None,
                        'raw_best_mqe': pred_mqe * 2.0, 'raw_topographic_error': 1.0,
                        'raw_mqe_improvement_ratio': None,
                        'map_m': 10, 'map_n': 10,
                        'constraint_violation': 999.0, 'penalty_factor': 1000.0,
                        'is_penalized': True, 'penalty_reason': 'mlp_prescreened',
                    }
                    log_status_to_csv(uid, population_id, generation, "mlp_skipped",
                                      start_time=datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                                      end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), working_dir=working_dir)
                    result = (penalty_results, copy.deepcopy(ind))
                    EVALUATED_CACHE[uid] = result
                    return result

        # Build LSTM early-stop callback for SOM training
        lstm_early_stop_fn = None
        if nn is not None and nn.can_check_early_stopping():
            lstm_threshold = nn_config.get('lstm_quality_threshold', 1.0)
            _ds_context = [
                fixed_params.get('ds_n_samples', 0),
                fixed_params.get('ds_n_active_dimensions', 0),
                fixed_params.get('ds_n_numeric', 0),
                fixed_params.get('ds_n_categorical', 0),
            ]
            def lstm_early_stop_fn(checkpoints):
                history = {
                    'progress':          [c['progress'] for c in checkpoints],
                    'mqe':               [c['mqe'] for c in checkpoints],
                    'topographic_error': [c['topographic_error'] for c in checkpoints],
                    'dead_neuron_ratio': [c['dead_neuron_ratio'] for c in checkpoints],
                    'learning_rate':     [c['learning_rate'] for c in checkpoints],
                    'radius':            [c['radius'] for c in checkpoints],
                }
                return nn.should_stop_early(history, lstm_threshold, dataset_context=_ds_context)

        # Build Phase 3 dynamic schedule callback
        dynamic_schedule_fn = None
        if nn is not None and nn.can_use_dynamic_schedule():
            _ds_context_ctrl = [
                fixed_params.get('ds_n_samples', 0),
                fixed_params.get('ds_n_active_dimensions', 0),
                fixed_params.get('ds_n_numeric', 0),
                fixed_params.get('ds_n_categorical', 0),
            ]
            dynamic_schedule_fn = nn.get_dynamic_schedule_fn(_ds_context_ctrl)

        som = KohonenSOM(dim=data.shape[1], **som_params)

        training_results = som.train(data, ignore_mask=ignore_mask, working_dir=individual_dir,
                                     lstm_early_stop_fn=lstm_early_stop_fn,
                                     dynamic_schedule_fn=dynamic_schedule_fn)

        training_results['uid'] = uid

        topographic_error = som.calculate_topographic_error(data, mask=ignore_mask)
        u_matrix_metrics = som.calculate_u_matrix_metrics()

        training_results['topographic_error'] = topographic_error
        training_results.update(u_matrix_metrics)

        training_results['training_duration'] = training_results.get('duration', None)

        dead_count, dead_ratio = som.calculate_dead_neurons(data)
        training_results['dead_neuron_count'] = dead_count
        training_results['dead_neuron_ratio'] = dead_ratio

        # Calculate distance map max for quality check
        distance_map, _ = som.compute_quantization_error(data, mask=ignore_mask)
        distance_map_max = np.max(distance_map) if distance_map is not None else 0.0
        training_results['distance_map_max'] = distance_map_max

        # Store raw metrics — never modified, used as NSGA-II objectives
        raw_best_mqe = training_results['best_mqe']
        raw_topographic_error = topographic_error
        training_results['topographic_error'] = topographic_error

        # Assess constraint violations (organization quality and dead neuron ratio)
        u_matrix_max = training_results.get('u_matrix_max', 0.0)
        ORGANIZATION_THRESHOLD = fixed_params.get('org_threshold', 1.0)
        penalty_reasons = []

        if u_matrix_max > ORGANIZATION_THRESHOLD or distance_map_max > ORGANIZATION_THRESHOLD:
            penalty_reasons.append(f"org(u={u_matrix_max:.3f},d={distance_map_max:.3f})")
            log_message(uid, f"Organization violation — U-Matrix max: {u_matrix_max:.3f}, Distance max: {distance_map_max:.3f}", working_dir)

        dead_threshold = _dead_neuron_threshold(som.m, som.n, data.shape[0])
        if dead_ratio > dead_threshold:
            penalty_reasons.append(f"dead={dead_ratio:.1%}(thresh={dead_threshold:.1%})")
            log_message(uid, f"Dead neuron violation: {dead_ratio:.1%} > threshold {dead_threshold:.1%}", working_dir)

        # Store raw values and map dimensions for constraint violation computation
        training_results['raw_best_mqe'] = round(raw_best_mqe, 8)
        training_results['raw_topographic_error'] = round(raw_topographic_error, 8)
        training_results['map_m'] = som.m
        training_results['map_n'] = som.n

        # Constraint violation scalar for constrained NSGA-II dominance (Deb 2002)
        # 0.0 = feasible; >0 = infeasible; magnitude ranks infeasible solutions
        cv = compute_constraint_violation(training_results, data.shape[0], ORGANIZATION_THRESHOLD)
        training_results['constraint_violation'] = round(cv, 6)
        training_results['is_penalized'] = cv > 0.0
        training_results['penalty_factor'] = round(1.0 + cv, 6)  # logged for reference; not applied to objectives
        training_results['penalty_reason'] = "|".join(penalty_reasons)

        # Compute improvement ratios vs random initialization (checkpoint[0] = pre-training baseline).
        # raw_mqe_improvement_ratio uses raw MQE — reflects true SOM quality, used as NSGA-II objective.
        # mqe_improvement_ratio uses penalized MQE — logged for reference only.
        ckpts = training_results.get('checkpoints', [])
        if ckpts:
            init_mqe = max(ckpts[0]['mqe'], 1e-10)
            training_results['initial_mqe'] = round(ckpts[0]['mqe'], 8)
            training_results['raw_mqe_improvement_ratio'] = round(raw_best_mqe / init_mqe, 6)
            training_results['mqe_improvement_ratio'] = round(training_results['best_mqe'] / init_mqe, 6)
        else:
            training_results['initial_mqe'] = None
            training_results['raw_mqe_improvement_ratio'] = None
            training_results['mqe_improvement_ratio'] = None

        cv_str = f" [INFEASIBLE cv={cv:.3f}: {training_results['penalty_reason']}]" if training_results['is_penalized'] else ""
        log_message(uid, f"Evaluated – raw_ratio={training_results['raw_mqe_improvement_ratio']}, raw_TE={raw_topographic_error:.4f}, Dead={dead_ratio:.2%}, Time={training_results['duration']:.2f}s{cv_str}", working_dir)

        # Propagate dataset metadata from fixed_params into results for CSV logging
        for key, value in fixed_params.items():
            if key.startswith('ds_') or key == 'dataset_name':
                training_results[key] = value

        generate_training_plots(training_results, individual_dir)

        # generate_individual_maps: controlled by FIXED_PARAMS.generate_individual_maps
        # Set to false when CNN visual quality assessment is not needed (spatial analysis
        # works directly on som.weights — no PNG files required).
        _gen_maps = fixed_params.get('generate_individual_maps', True)
        if _gen_maps:
            generate_individual_maps(som, data, ignore_mask, individual_dir)

        # CNN visual quality assessment (combine individual maps → RGB → CNN score)
        training_results['cnn_quality_score'] = None
        if _gen_maps and nn is not None and nn.can_assess_visual_quality():
            try:
                from PIL import Image as _PIL_Image
                vis_dir = os.path.join(individual_dir, "visualizations")
                u_path = os.path.join(vis_dir, "u_matrix.png")
                d_path = os.path.join(vis_dir, "distance_map.png")
                dn_path = os.path.join(vis_dir, "dead_neurons_map.png")
                if all(os.path.exists(p) for p in [u_path, d_path, dn_path]):
                    rgb_img = _PIL_Image.merge('RGB', (
                        _PIL_Image.open(u_path).convert('L'),
                        _PIL_Image.open(d_path).convert('L'),
                        _PIL_Image.open(dn_path).convert('L'),
                    ))
                    rgb_tmp = os.path.join(vis_dir, f"{uid}_rgb_tmp.png")
                    rgb_img.save(rgb_tmp)
                    cnn_result = nn.assess_visual_quality(rgb_tmp)
                    if cnn_result is not None:
                        cnn_score, cnn_label = cnn_result
                        training_results['cnn_quality_score'] = cnn_score
                        log_message(uid, f"CNN quality: {cnn_score:.3f} ({cnn_label})", working_dir)
            except Exception as e:
                log_message(uid, f"CNN assessment failed: {e}", working_dir)

        # Log results to CSV after CNN assessment so cnn_quality_score is included
        log_result_to_csv(ind, training_results, working_dir)

        log_status_to_csv(uid, population_id, generation, "completed",
                         start_time=datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                         end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), working_dir=working_dir)

        # Copy maps to centralized dataset for CNN training
        if _gen_maps:
            copy_maps_to_dataset(uid, individual_dir, working_dir)

        # Cache the results to avoid re-evaluation of duplicate configurations
        result = (training_results, copy.deepcopy(ind))
        EVALUATED_CACHE[uid] = result

        return result

    except Exception as e:
        import traceback
        log_message(uid, f"ERROR during evaluation: {e}", working_dir)
        log_message(uid, f"Full traceback:\n{traceback.format_exc()}", working_dir)

        log_status_to_csv(uid, population_id, generation, "failed",
                         start_time=datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                         end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), working_dir=working_dir)
        log_message(uid, f"Error during evaluation: {str(e)}", working_dir)
        raise e


def combine_maps_to_rgb(working_dir: str) -> None:
    """
    Combine three individual maps (U-Matrix, Distance Map, Dead Neurons Map) into a single RGB image.
    This function runs after EA completes and processes all maps in maps_dataset directory.

    RGB Channel mapping:
        - R (Red): U-Matrix (topological structure)
        - G (Green): Distance Map (quantization error)
        - B (Blue): Dead Neurons Map (neuron activity)

    Args:
        working_dir: Working directory containing maps_dataset folder
    """
    from PIL import Image

    maps_dataset_dir = os.path.join(working_dir, "maps_dataset")
    rgb_output_dir = os.path.join(maps_dataset_dir, "rgb")
    os.makedirs(rgb_output_dir, exist_ok=True)

    # Find all unique UIDs by looking for u_matrix files
    u_matrix_files = [f for f in os.listdir(maps_dataset_dir) if f.endswith('_u_matrix.png')]

    print(f"INFO: Combining {len(u_matrix_files)} map sets into RGB images...")

    for u_matrix_file in u_matrix_files:
        # Extract UID from filename
        uid = u_matrix_file.replace('_u_matrix.png', '')

        # Construct paths to all three maps
        u_matrix_path = os.path.join(maps_dataset_dir, f"{uid}_u_matrix.png")
        distance_map_path = os.path.join(maps_dataset_dir, f"{uid}_distance_map.png")
        dead_neurons_path = os.path.join(maps_dataset_dir, f"{uid}_dead_neurons_map.png")

        # Check if all three maps exist
        if not all(os.path.exists(p) for p in [u_matrix_path, distance_map_path, dead_neurons_path]):
            print(f"WARNING: Missing maps for UID {uid}, skipping...")
            continue

        try:
            # Load images as grayscale
            u_matrix_img = Image.open(u_matrix_path).convert('L')
            distance_map_img = Image.open(distance_map_path).convert('L')
            dead_neurons_img = Image.open(dead_neurons_path).convert('L')

            # Verify all images have the same size
            if not (u_matrix_img.size == distance_map_img.size == dead_neurons_img.size):
                print(f"WARNING: Map size mismatch for UID {uid}, skipping...")
                continue

            # Combine into RGB image
            rgb_image = Image.merge('RGB', (u_matrix_img, distance_map_img, dead_neurons_img))

            # Save RGB image
            rgb_output_path = os.path.join(rgb_output_dir, f"{uid}_rgb.png")
            rgb_image.save(rgb_output_path)

        except Exception as e:
            print(f"ERROR: Failed to create RGB image for UID {uid}: {str(e)}")

    print(f"INFO: RGB image generation completed. Saved to {rgb_output_dir}")


def main():
    global WORKING_DIR
    global ARCHIVE
    global INPUT_FILE
    global _OBJ_RUNNING_MIN, _OBJ_RUNNING_MAX

    parser = argparse.ArgumentParser(description='Evolutionary optimization of the SOM algorithm')
    parser.add_argument('-i', '--input', help='Path to the input CSV file')
    parser.add_argument('-c', '--config', help='Path to a custom configuration file (JSON)')
    args = parser.parse_args()

    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist.")
            sys.exit(1)
        INPUT_FILE = args.input

    config = load_configuration(args.config)
    preprocess_config = config.get("PREPROCES_DATA", {})

    # Preprocessing and search space adjustment happen once — shared across all seed runs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if INPUT_FILE:
        base_results_dir = os.path.join(os.path.dirname(os.path.abspath(INPUT_FILE)), "results", timestamp)
    else:
        base_results_dir = os.path.join(os.getcwd(), "results", timestamp)
    os.makedirs(base_results_dir, exist_ok=True)

    print("INFO: Preparing data for evolution...")
    if INPUT_FILE:
        input_data_df = validate_input_data(INPUT_FILE, base_results_dir, preprocess_config)
        training_data_path, _, ignore_mask, _ = preprocess_data(input_data_df, preprocess_config, base_results_dir)
        loaded_data = np.load(training_data_path)
        config['PREPROCES_DATA'] = preprocess_config
        print(f"INFO: Data loaded and normalized from file {training_data_path}. Shape: {loaded_data.shape}")
    else:
        data_params = config.get("DATA_PARAMS", {})
        sample_size = data_params.get("sample_size", 1000)
        input_dim = data_params.get("input_dim", 10)
        loaded_data = get_or_generate_data(sample_size, input_dim)
        ignore_mask = None
        print(f"INFO: Data has been generated. Number of samples: {sample_size}, dimension: {input_dim}")

    # Adjust search space bounds once from dataset size
    config['SEARCH_SPACE'] = apply_dynamic_search_space(config['SEARCH_SPACE'], loaded_data.shape[0])

    # Inject dataset metadata into fixed_params so every results.csv row carries dataset context
    config['FIXED_PARAMS'] = dict(config.get('FIXED_PARAMS', {}))
    if INPUT_FILE:
        config['FIXED_PARAMS']['dataset_name'] = os.path.basename(os.path.dirname(os.path.abspath(INPUT_FILE)))
        meta_path = os.path.join(base_results_dir, "dataset_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, encoding='utf-8') as f:
                for key, value in json.load(f).items():
                    config['FIXED_PARAMS'][key] = value
            print(f"INFO: Dataset metadata loaded from {meta_path}")
        else:
            print("INFO: dataset_meta.json not found — dataset columns will be empty")

    # Calibrate org_threshold once — depends on dataset/map size, not on seed
    org_threshold = run_calibration_probe(config, loaded_data, ignore_mask, base_results_dir)
    config['FIXED_PARAMS']['org_threshold'] = org_threshold

    # Seed list: EA_SETTINGS.seeds overrides FIXED_PARAMS.random_seed
    seeds = config.get('EA_SETTINGS', {}).get('seeds')
    if not seeds:
        seeds = [config['FIXED_PARAMS'].get('random_seed', 42)]

    print(f"\nINFO: Running {len(seeds)} independent EA runs — seeds: {seeds}")

    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"EA run {i+1}/{len(seeds)} — seed={seed}")
        print(f"{'='*60}")

        WORKING_DIR = os.path.join(base_results_dir, f"seed_{seed}")
        os.makedirs(WORKING_DIR, exist_ok=True)

        run_config = dict(config)
        run_config['FIXED_PARAMS'] = dict(config['FIXED_PARAMS'])
        run_config['FIXED_PARAMS']['random_seed'] = seed

        ARCHIVE = []
        _OBJ_RUNNING_MIN = None
        _OBJ_RUNNING_MAX = None
        run_evolution(run_config, loaded_data, ignore_mask)

        print(f"\nINFO: Generating RGB images for seed={seed}...")
        combine_maps_to_rgb(WORKING_DIR)

    print(f"\nINFO: All {len(seeds)} seed runs complete. Results in: {base_results_dir}")

if __name__ == "__main__":
    try:
        main()
    finally:
        # Clean up any leaked resources
        import gc
        gc.collect()

        # Force cleanup of multiprocessing resources
        try:
            from multiprocessing.util import _exit_function
            _exit_function()
        except Exception:
            pass
