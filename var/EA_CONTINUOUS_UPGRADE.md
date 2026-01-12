# EA Upgrade: Continuous Real-Valued Optimization

**Document Version**: 1.0
**Last Updated**: 2026-01-11
**Purpose**: Upgrade EA from discrete categorical to continuous real-valued optimization

---

## Problem Statement

**Current Limitation**: The EA currently supports only **discrete categorical selection** from predefined lists:

```json
{
  "start_learning_rate": [0.9, 0.8, 0.7, 0.6, 0.5],
  "end_learning_rate": [0.2, 0.1, 0.05, 0.01]
}
```

This severely limits the search space. For example:
- Only 5 possible values for `start_learning_rate` (instead of continuous range `[0.5, 0.9]`)
- Cannot explore intermediate values like `0.75` or `0.625`
- Total search space = product of list lengths (very sparse)

**Required Solution**: Support **continuous intervals** with real-valued genetic operators:

```json
{
  "start_learning_rate": {"type": "float", "min": 0.5, "max": 0.9},
  "end_learning_rate": {"type": "float", "min": 0.01, "max": 0.3},
  "processing_type": {"type": "categorical", "values": ["stochastic", "deterministic", "hybrid"]}
}
```

---

## Requirements

### FR-EA-CONT-1: Mixed Search Space Format ❌

**Requirement**: Support both continuous intervals and categorical lists in search space configuration.

**Status**: NOT IMPLEMENTED

**New Configuration Format**:
```json
{
  "SEARCH_SPACE": {
    "map_size": {
      "type": "discrete_int_pair",
      "min": 5,
      "max": 20,
      "step": 1
    },
    "processing_type": {
      "type": "categorical",
      "values": ["stochastic", "deterministic", "hybrid"]
    },
    "start_learning_rate": {
      "type": "float",
      "min": 0.5,
      "max": 0.9
    },
    "end_learning_rate": {
      "type": "float",
      "min": 0.01,
      "max": 0.3
    },
    "lr_decay_type": {
      "type": "categorical",
      "values": ["linear-drop", "exp-drop", "log-drop", "step-down"]
    },
    "start_radius_init_ratio": {
      "type": "float",
      "min": 0.1,
      "max": 1.0
    },
    "radius_decay_type": {
      "type": "categorical",
      "values": ["linear-drop", "exp-drop", "log-drop", "step-down"]
    },
    "start_batch_percent": {
      "type": "float",
      "min": 0.1,
      "max": 10.0
    },
    "end_batch_percent": {
      "type": "float",
      "min": 3.0,
      "max": 20.0
    },
    "batch_growth_type": {
      "type": "categorical",
      "values": ["linear-growth", "exp-growth", "log-growth"]
    },
    "epoch_multiplier": {
      "type": "float",
      "min": 5.0,
      "max": 30.0
    },
    "normalize_weights_flag": {
      "type": "categorical",
      "values": [false, true]
    },
    "growth_g": {
      "type": "float",
      "min": 1.0,
      "max": 50.0
    },
    "num_batches": {
      "type": "int",
      "min": 1,
      "max": 20
    }
  }
}
```

**Parameter Types**:
- `float`: Continuous real-valued (e.g., learning rates)
- `int`: Discrete integer (e.g., num_batches)
- `categorical`: Discrete categorical (e.g., decay types)
- `discrete_int_pair`: Pair of integers (e.g., map_size [m, n])

**Backward Compatibility**: Support old format (lists) by auto-converting to new format

**Acceptance Criteria**:
- [ ] Parse new interval-based format
- [ ] Detect parameter type automatically
- [ ] Validate min/max bounds
- [ ] Backward compatible with list format

---

### FR-EA-CONT-2: Continuous Random Initialization ❌

**Requirement**: Generate random configurations sampling from continuous intervals.

**Status**: NOT IMPLEMENTED

**Implementation**:
```python
def random_config_continuous(param_space: dict) -> dict:
    """
    Generate random configuration from mixed search space.

    Supports: float intervals, int intervals, categorical lists
    """
    config = {}

    for key, spec in param_space.items():
        param_type = spec.get('type')

        if param_type == 'float':
            # Uniform random from [min, max]
            config[key] = random.uniform(spec['min'], spec['max'])

        elif param_type == 'int':
            # Random integer from [min, max]
            config[key] = random.randint(spec['min'], spec['max'])

        elif param_type == 'categorical':
            # Random choice from values
            config[key] = random.choice(spec['values'])

        elif param_type == 'discrete_int_pair':
            # Two integers for map_size
            m = random.randint(spec['min'], spec['max'])
            n = random.randint(spec['min'], spec['max'])
            config[key] = [m, n]

        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return validate_and_repair(config)
```

**Acceptance Criteria**:
- [ ] Uniform sampling from float intervals
- [ ] Integer sampling from int intervals
- [ ] Categorical selection from lists
- [ ] Validation and repair after generation

---

### FR-EA-CONT-3: Simulated Binary Crossover (SBX) ❌

**Requirement**: Implement SBX for continuous parameter crossover.

**Status**: NOT IMPLEMENTED

**SBX Algorithm**:
```python
def sbx_crossover(parent1: float, parent2: float, eta: float = 20.0, bounds: tuple = None) -> tuple:
    """
    Simulated Binary Crossover (SBX) for continuous parameters.

    Args:
        parent1, parent2: Parent values
        eta: Distribution index (higher = more explorative)
        bounds: (min, max) bounds for clipping

    Returns:
        (child1, child2) offspring values
    """
    if random.random() > 0.5:
        # No crossover (50% chance)
        return parent1, parent2

    # SBX calculation
    if abs(parent1 - parent2) < 1e-9:
        return parent1, parent2

    # Generate random beta
    u = random.random()

    if u <= 0.5:
        beta = (2.0 * u) ** (1.0 / (eta + 1.0))
    else:
        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))

    # Calculate offspring
    child1 = 0.5 * ((1 + beta) * parent1 + (1 - beta) * parent2)
    child2 = 0.5 * ((1 - beta) * parent1 + (1 + beta) * parent2)

    # Apply bounds
    if bounds:
        child1 = np.clip(child1, bounds[0], bounds[1])
        child2 = np.clip(child2, bounds[0], bounds[1])

    return child1, child2
```

**Acceptance Criteria**:
- [ ] SBX operator for float parameters
- [ ] Configurable distribution index (eta)
- [ ] Bounds enforcement
- [ ] 50% crossover probability

---

### FR-EA-CONT-4: Polynomial Mutation ❌

**Requirement**: Implement polynomial mutation for continuous parameters.

**Status**: NOT IMPLEMENTED

**Polynomial Mutation Algorithm**:
```python
def polynomial_mutation(value: float, eta: float = 20.0, bounds: tuple = None,
                       mutation_prob: float = 0.1) -> float:
    """
    Polynomial mutation for continuous parameters.

    Args:
        value: Current parameter value
        eta: Distribution index (higher = smaller mutations)
        bounds: (min, max) bounds
        mutation_prob: Probability of mutation

    Returns:
        Mutated value
    """
    if random.random() > mutation_prob:
        return value  # No mutation

    if bounds is None:
        bounds = (0.0, 1.0)  # Default bounds

    min_val, max_val = bounds
    delta_1 = (value - min_val) / (max_val - min_val)
    delta_2 = (max_val - value) / (max_val - min_val)

    u = random.random()
    if u < 0.5:
        delta_q = (2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta_1) ** (eta + 1.0)) ** (1.0 / (eta + 1.0)) - 1.0
    else:
        delta_q = 1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta_2) ** (eta + 1.0)) ** (1.0 / (eta + 1.0))

    mutated = value + delta_q * (max_val - min_val)

    # Clip to bounds
    mutated = np.clip(mutated, min_val, max_val)

    return mutated
```

**Acceptance Criteria**:
- [ ] Polynomial mutation for float parameters
- [ ] Configurable distribution index (eta)
- [ ] Bounds enforcement
- [ ] Per-parameter mutation probability

---

### FR-EA-CONT-5: Mixed Crossover Operator ❌

**Requirement**: Combine SBX for continuous parameters with discrete crossover.

**Status**: NOT IMPLEMENTED

**Implementation**:
```python
def crossover_mixed(parent1: dict, parent2: dict, param_space: dict, eta: float = 20.0) -> tuple:
    """
    Mixed crossover for continuous and categorical parameters.

    Args:
        parent1, parent2: Parent configurations
        param_space: Search space specification
        eta: SBX distribution index

    Returns:
        (child1, child2) offspring configurations
    """
    child1 = {}
    child2 = {}

    for key, spec in param_space.items():
        param_type = spec.get('type')

        if param_type == 'float':
            # SBX crossover
            bounds = (spec['min'], spec['max'])
            c1, c2 = sbx_crossover(parent1[key], parent2[key], eta=eta, bounds=bounds)
            child1[key] = c1
            child2[key] = c2

        elif param_type == 'int':
            # SBX + rounding
            bounds = (spec['min'], spec['max'])
            c1, c2 = sbx_crossover(float(parent1[key]), float(parent2[key]), eta=eta, bounds=bounds)
            child1[key] = int(round(c1))
            child2[key] = int(round(c2))

        elif param_type == 'categorical':
            # Uniform crossover (random selection)
            if random.random() < 0.5:
                child1[key] = parent1[key]
                child2[key] = parent2[key]
            else:
                child1[key] = parent2[key]
                child2[key] = parent1[key]

        elif param_type == 'discrete_int_pair':
            # Crossover each dimension separately
            bounds = (spec['min'], spec['max'])
            c1_m, c2_m = sbx_crossover(float(parent1[key][0]), float(parent2[key][0]), eta=eta, bounds=bounds)
            c1_n, c2_n = sbx_crossover(float(parent1[key][1]), float(parent2[key][1]), eta=eta, bounds=bounds)
            child1[key] = [int(round(c1_m)), int(round(c1_n))]
            child2[key] = [int(round(c2_m)), int(round(c2_n))]

    return validate_and_repair(child1), validate_and_repair(child2)
```

**Acceptance Criteria**:
- [ ] SBX for float parameters
- [ ] SBX + rounding for int parameters
- [ ] Uniform crossover for categorical parameters
- [ ] Per-dimension crossover for int pairs

---

### FR-EA-CONT-6: Mixed Mutation Operator ❌

**Requirement**: Combine polynomial mutation for continuous with discrete mutation.

**Status**: NOT IMPLEMENTED

**Implementation**:
```python
def mutate_mixed(config: dict, param_space: dict, eta: float = 20.0, mutation_prob: float = 0.1) -> dict:
    """
    Mixed mutation for continuous and categorical parameters.

    Args:
        config: Configuration to mutate
        param_space: Search space specification
        eta: Polynomial mutation distribution index
        mutation_prob: Probability per parameter

    Returns:
        Mutated configuration
    """
    mutated = config.copy()

    for key, spec in param_space.items():
        param_type = spec.get('type')

        if param_type == 'float':
            # Polynomial mutation
            bounds = (spec['min'], spec['max'])
            mutated[key] = polynomial_mutation(config[key], eta=eta, bounds=bounds, mutation_prob=mutation_prob)

        elif param_type == 'int':
            # Polynomial mutation + rounding
            bounds = (spec['min'], spec['max'])
            mutated_float = polynomial_mutation(float(config[key]), eta=eta, bounds=bounds, mutation_prob=mutation_prob)
            mutated[key] = int(round(mutated_float))

        elif param_type == 'categorical':
            # Random replacement
            if random.random() < mutation_prob:
                mutated[key] = random.choice(spec['values'])

        elif param_type == 'discrete_int_pair':
            # Mutate each dimension separately
            bounds = (spec['min'], spec['max'])
            if random.random() < mutation_prob:
                m_mutated = polynomial_mutation(float(config[key][0]), eta=eta, bounds=bounds, mutation_prob=1.0)
                n_mutated = polynomial_mutation(float(config[key][1]), eta=eta, bounds=bounds, mutation_prob=1.0)
                mutated[key] = [int(round(m_mutated)), int(round(n_mutated))]

    return validate_and_repair(mutated)
```

**Acceptance Criteria**:
- [ ] Polynomial mutation for float parameters
- [ ] Polynomial mutation + rounding for int parameters
- [ ] Random replacement for categorical parameters
- [ ] Per-dimension mutation for int pairs

---

### FR-EA-CONT-7: Backward Compatibility ❌

**Requirement**: Support old list-based format by auto-converting to new interval format.

**Status**: NOT IMPLEMENTED

**Implementation**:
```python
def convert_legacy_search_space(legacy_space: dict) -> dict:
    """
    Convert old list-based format to new interval-based format.

    Old: {"start_learning_rate": [0.9, 0.8, 0.7, 0.6, 0.5]}
    New: {"start_learning_rate": {"type": "float", "min": 0.5, "max": 0.9}}
    """
    converted = {}

    for key, value in legacy_space.items():
        if isinstance(value, list):
            # Detect type from list contents
            if all(isinstance(v, str) for v in value):
                # Categorical
                converted[key] = {'type': 'categorical', 'values': value}

            elif all(isinstance(v, bool) for v in value):
                # Boolean categorical
                converted[key] = {'type': 'categorical', 'values': value}

            elif all(isinstance(v, list) for v in value):
                # Pairs (e.g., map_size)
                flat = [item for sublist in value for item in sublist]
                converted[key] = {'type': 'discrete_int_pair', 'min': min(flat), 'max': max(flat)}

            elif all(isinstance(v, int) for v in value):
                # Integer range
                converted[key] = {'type': 'int', 'min': min(value), 'max': max(value)}

            elif all(isinstance(v, (int, float)) for v in value):
                # Float range
                converted[key] = {'type': 'float', 'min': min(value), 'max': max(value)}

        else:
            # Already new format or fixed value
            converted[key] = value

    return converted
```

**Acceptance Criteria**:
- [ ] Auto-detect parameter type from list
- [ ] Convert to appropriate interval/categorical format
- [ ] Maintain semantics (min/max from list)
- [ ] Support mixed legacy and new formats

---

## Implementation Checklist

**Files to Modify**:
1. `app/ea/ea.py`:
   - ✅ Update `random_config()` → `random_config_continuous()`
   - ✅ Update `crossover()` → `crossover_mixed()`
   - ✅ Update `mutate()` → `mutate_mixed()`
   - ✅ Add SBX and polynomial mutation functions
   - ✅ Add legacy format converter

2. `app/test/ea-iris-config.json`:
   - ✅ Convert to new interval-based format

3. New file `app/ea/genetic_operators.py` (recommended):
   - ✅ Move `sbx_crossover()`
   - ✅ Move `polynomial_mutation()`
   - ✅ Move `crossover_mixed()`
   - ✅ Move `mutate_mixed()`

**Testing**:
- [ ] Test SBX crossover produces valid offspring
- [ ] Test polynomial mutation respects bounds
- [ ] Test mixed operators handle all parameter types
- [ ] Test backward compatibility with old configs
- [ ] Run EA on Iris with new continuous config
- [ ] Compare results: continuous vs discrete EA

---

## Example New Configuration

```json
{
  "EA_SETTINGS": {
    "population_size": 50,
    "generations": 100
  },
  "SEARCH_SPACE": {
    "map_size": {
      "type": "discrete_int_pair",
      "min": 5,
      "max": 20
    },
    "processing_type": {
      "type": "categorical",
      "values": ["stochastic", "deterministic", "hybrid"]
    },
    "start_learning_rate": {
      "type": "float",
      "min": 0.5,
      "max": 0.9
    },
    "end_learning_rate": {
      "type": "float",
      "min": 0.01,
      "max": 0.3
    },
    "lr_decay_type": {
      "type": "categorical",
      "values": ["linear-drop", "exp-drop", "log-drop", "step-down"]
    },
    "start_radius_init_ratio": {
      "type": "float",
      "min": 0.1,
      "max": 1.0
    },
    "radius_decay_type": {
      "type": "categorical",
      "values": ["linear-drop", "exp-drop", "log-drop", "step-down"]
    },
    "start_batch_percent": {
      "type": "float",
      "min": 0.1,
      "max": 10.0
    },
    "end_batch_percent": {
      "type": "float",
      "min": 3.0,
      "max": 20.0
    },
    "batch_growth_type": {
      "type": "categorical",
      "values": ["linear-growth", "exp-growth", "log-growth"]
    },
    "epoch_multiplier": {
      "type": "float",
      "min": 5.0,
      "max": 30.0
    },
    "normalize_weights_flag": {
      "type": "categorical",
      "values": [false, true]
    },
    "growth_g": {
      "type": "float",
      "min": 1.0,
      "max": 50.0
    },
    "num_batches": {
      "type": "int",
      "min": 1,
      "max": 20
    }
  },
  "GENETIC_OPERATORS": {
    "sbx_eta": 20.0,
    "mutation_eta": 20.0,
    "mutation_prob": 0.1,
    "crossover_prob": 0.9
  },
  "FIXED_PARAMS": {
    "end_radius": 1.0,
    "random_seed": 42,
    "mqe_evaluations_per_run": 500,
    "map_type": "hex"
  }
}
```

---

## Priority

**This upgrade is CRITICAL** before running large-scale EA campaigns for CNN training.

**Impact**:
- **Search space size**: Increases from ~10^8 to continuous infinity
- **Solution quality**: Can find optimal intermediate values
- **CNN training**: Provides more diverse and high-quality dataset

**Recommendation**: Implement this **immediately** before generating CNN training data.

---

**End of EA Continuous Upgrade Specification**
