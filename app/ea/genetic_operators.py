"""
Genetic operators for real-valued evolutionary optimization.

Implements:
- Simulated Binary Crossover (SBX) for continuous parameters
- Polynomial Mutation for continuous parameters
- Mixed operators for continuous + categorical parameters
"""

import random
import numpy as np
from typing import Tuple, Dict, Any


def sbx_crossover(parent1: float, parent2: float, eta: float = 20.0,
                  bounds: Tuple[float, float] = None) -> Tuple[float, float]:
    """
    Simulated Binary Crossover (SBX) for continuous parameters.

    Args:
        parent1: First parent value
        parent2: Second parent value
        eta: Distribution index (higher = more exploitative, typical: 15-20)
        bounds: (min, max) bounds for clipping

    Returns:
        (child1, child2) offspring values
    """
    # 50% chance of no crossover
    if random.random() > 0.5:
        return parent1, parent2

    # If parents are identical, no crossover
    if abs(parent1 - parent2) < 1e-9:
        return parent1, parent2

    # Ensure parent1 <= parent2
    if parent1 > parent2:
        parent1, parent2 = parent2, parent1

    # Generate spread factor beta
    u = random.random()

    if u <= 0.5:
        beta = (2.0 * u) ** (1.0 / (eta + 1.0))
    else:
        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

    # Calculate offspring
    child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
    child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)

    # Apply bounds if specified
    if bounds:
        min_val, max_val = bounds
        child1 = np.clip(child1, min_val, max_val)
        child2 = np.clip(child2, min_val, max_val)

    return child1, child2


def polynomial_mutation(value: float, eta: float = 20.0,
                       bounds: Tuple[float, float] = None,
                       mutation_prob: float = 1.0) -> float:
    """
    Polynomial mutation for continuous parameters.

    Args:
        value: Current parameter value
        eta: Distribution index (higher = smaller mutations, typical: 15-20)
        bounds: (min, max) bounds
        mutation_prob: Probability of mutation (default: 1.0 since called per-gene)

    Returns:
        Mutated value
    """
    if random.random() > mutation_prob:
        return value

    if bounds is None:
        raise ValueError("Bounds required for polynomial mutation")

    min_val, max_val = bounds

    # Normalize value to [0, 1]
    if max_val - min_val < 1e-9:
        return value

    delta_1 = (value - min_val) / (max_val - min_val)
    delta_2 = (max_val - value) / (max_val - min_val)

    u = random.random()

    if u < 0.5:
        xy = 1.0 - delta_1
        val = 2.0 * u + (1.0 - 2.0 * u) * (xy ** (eta + 1.0))
        delta_q = val ** (1.0 / (eta + 1.0)) - 1.0
    else:
        xy = 1.0 - delta_2
        val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (xy ** (eta + 1.0))
        delta_q = 1.0 - val ** (1.0 / (eta + 1.0))

    # Apply mutation
    mutated = value + delta_q * (max_val - min_val)

    # Clip to bounds
    mutated = np.clip(mutated, min_val, max_val)

    return mutated


def random_config_continuous(param_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate random configuration from mixed search space.

    Supports:
    - float: Continuous real-valued parameters
    - int: Discrete integer parameters
    - categorical: Discrete categorical parameters
    - discrete_int_pair: Pair of integers (e.g., map_size)

    Args:
        param_space: Search space specification

    Returns:
        Random configuration dictionary
    """
    config = {}

    for key, spec in param_space.items():
        # Skip comment fields
        if key == 'comment' or (isinstance(spec, dict) and 'comment' in spec):
            if isinstance(spec, dict):
                spec = {k: v for k, v in spec.items() if k != 'comment'}

        if not isinstance(spec, dict):
            # Fixed value (not in search space)
            config[key] = spec
            continue

        param_type = spec.get('type')

        if param_type == 'float':
            # Uniform random from [min, max], rounded to 2 decimal places
            config[key] = round(random.uniform(spec['min'], spec['max']), 2)

        elif param_type == 'int':
            # Random integer from [min, max]
            config[key] = random.randint(spec['min'], spec['max'])

        elif param_type == 'categorical':
            # Random choice from values
            config[key] = random.choice(spec['values'])

        elif param_type == 'discrete_int_pair':
            # Two integers for map_size (square maps: m == n)
            size = random.randint(spec['min'], spec['max'])
            config[key] = [size, size]

        else:
            raise ValueError(f"Unknown parameter type: {param_type} for parameter {key}")

    return config


def crossover_mixed(parent1: Dict[str, Any], parent2: Dict[str, Any],
                   param_space: Dict[str, Any], eta: float = 20.0) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Mixed crossover for continuous and categorical parameters.

    Args:
        parent1: First parent configuration
        parent2: Second parent configuration
        param_space: Search space specification
        eta: SBX distribution index

    Returns:
        (child1, child2) offspring configurations
    """
    child1 = {}
    child2 = {}

    for key, spec in param_space.items():
        # Skip comment fields
        if key == 'comment' or not isinstance(spec, dict):
            continue

        if 'comment' in spec:
            spec = {k: v for k, v in spec.items() if k != 'comment'}

        param_type = spec.get('type')

        if param_type == 'float':
            # SBX crossover for continuous parameters, rounded to 2 decimal places
            bounds = (spec['min'], spec['max'])
            c1, c2 = sbx_crossover(parent1[key], parent2[key], eta=eta, bounds=bounds)
            child1[key] = round(c1, 2)
            child2[key] = round(c2, 2)

        elif param_type == 'int':
            # SBX + rounding for integer parameters
            bounds = (spec['min'], spec['max'])
            c1, c2 = sbx_crossover(float(parent1[key]), float(parent2[key]), eta=eta, bounds=bounds)
            child1[key] = int(round(c1))
            child2[key] = int(round(c2))

        elif param_type == 'categorical':
            # Uniform crossover for categorical parameters
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        elif param_type == 'discrete_int_pair':
            # Crossover for square maps (m == n)
            bounds = (spec['min'], spec['max'])
            c1_size, c2_size = sbx_crossover(float(parent1[key][0]), float(parent2[key][0]), eta=eta, bounds=bounds)
            child1[key] = [int(round(c1_size)), int(round(c1_size))]
            child2[key] = [int(round(c2_size)), int(round(c2_size))]

    return child1, child2


def mutate_mixed(config: Dict[str, Any], param_space: Dict[str, Any],
                eta: float = 20.0, mutation_prob: float = 0.1) -> Dict[str, Any]:
    """
    Mixed mutation for continuous and categorical parameters.

    Args:
        config: Configuration to mutate
        param_space: Search space specification
        eta: Polynomial mutation distribution index
        mutation_prob: Probability of mutating each parameter

    Returns:
        Mutated configuration
    """
    mutated = config.copy()

    for key, spec in param_space.items():
        # Skip comment fields
        if key == 'comment' or not isinstance(spec, dict):
            continue

        if 'comment' in spec:
            spec = {k: v for k, v in spec.items() if k != 'comment'}

        param_type = spec.get('type')

        if param_type == 'float':
            # Polynomial mutation for continuous parameters, rounded to 2 decimal places
            bounds = (spec['min'], spec['max'])
            mutated[key] = round(polynomial_mutation(config[key], eta=eta, bounds=bounds,
                                              mutation_prob=mutation_prob), 2)

        elif param_type == 'int':
            # Polynomial mutation + rounding for integer parameters
            if random.random() < mutation_prob:
                bounds = (spec['min'], spec['max'])
                mutated_float = polynomial_mutation(float(config[key]), eta=eta, bounds=bounds,
                                                   mutation_prob=1.0)
                mutated[key] = int(round(mutated_float))

        elif param_type == 'categorical':
            # Random replacement for categorical parameters
            if random.random() < mutation_prob:
                mutated[key] = random.choice(spec['values'])

        elif param_type == 'discrete_int_pair':
            # Mutate square maps (m == n)
            if random.random() < mutation_prob:
                bounds = (spec['min'], spec['max'])
                size_mutated = polynomial_mutation(float(config[key][0]), eta=eta, bounds=bounds,
                                                   mutation_prob=1.0)
                mutated[key] = [int(round(size_mutated)), int(round(size_mutated))]

    return mutated
