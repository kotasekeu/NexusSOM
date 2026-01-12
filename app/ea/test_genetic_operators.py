"""
Quick test script for continuous genetic operators.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ea.genetic_operators import (
    random_config_continuous,
    crossover_mixed,
    mutate_mixed,
    sbx_crossover,
    polynomial_mutation
)

# Test search space (from config)
test_search_space = {
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
        "min": 0.0,
        "max": 1.0
    },
    "end_learning_rate": {
        "type": "float",
        "min": 0.0,
        "max": 1.0
    },
    "lr_decay_type": {
        "type": "categorical",
        "values": ["linear-drop", "exp-drop", "log-drop", "step-down"]
    },
    "num_batches": {
        "type": "int",
        "min": 1,
        "max": 30
    }
}

print("=" * 60)
print("Testing Continuous Genetic Operators")
print("=" * 60)

# Test 1: Random config generation
print("\n1. Testing random_config_continuous()...")
config1 = random_config_continuous(test_search_space)
config2 = random_config_continuous(test_search_space)

print("Config 1:", config1)
print("Config 2:", config2)

# Validate bounds
assert 0.0 <= config1['start_learning_rate'] <= 1.0, "LR out of bounds!"
assert 5 <= config1['map_size'][0] <= 20, "Map size out of bounds!"
assert config1['map_size'][0] == config1['map_size'][1], "Map is not square!"
assert config2['map_size'][0] == config2['map_size'][1], "Map is not square!"
assert config1['processing_type'] in ["stochastic", "deterministic", "hybrid"], "Invalid processing type!"
print("✓ Random generation successful, bounds respected, maps are square")

# Test 2: SBX Crossover
print("\n2. Testing sbx_crossover()...")
parent1_lr = 0.8
parent2_lr = 0.5
child1_lr, child2_lr = sbx_crossover(parent1_lr, parent2_lr, eta=20.0, bounds=(0.0, 1.0))

print(f"Parent 1: {parent1_lr:.3f}, Parent 2: {parent2_lr:.3f}")
print(f"Child 1: {child1_lr:.3f}, Child 2: {child2_lr:.3f}")
assert 0.0 <= child1_lr <= 1.0, "Child 1 out of bounds!"
assert 0.0 <= child2_lr <= 1.0, "Child 2 out of bounds!"
print("✓ SBX crossover successful, bounds respected")

# Test 3: Polynomial Mutation
print("\n3. Testing polynomial_mutation()...")
value = 0.7
mutated = polynomial_mutation(value, eta=20.0, bounds=(0.0, 1.0), mutation_prob=1.0)

print(f"Original: {value:.3f}, Mutated: {mutated:.3f}")
assert 0.0 <= mutated <= 1.0, "Mutated value out of bounds!"
print("✓ Polynomial mutation successful, bounds respected")

# Test 4: Mixed Crossover
print("\n4. Testing crossover_mixed()...")
child1, child2 = crossover_mixed(config1, config2, test_search_space, eta=20.0)

print("Child 1:", child1)
print("Child 2:", child2)

# Validate children
assert 0.0 <= child1['start_learning_rate'] <= 1.0, "Child 1 LR out of bounds!"
assert 0.0 <= child2['start_learning_rate'] <= 1.0, "Child 2 LR out of bounds!"
assert 1 <= child1['num_batches'] <= 30, "Child 1 num_batches out of bounds!"
assert child1['map_size'][0] == child1['map_size'][1], "Child 1 map is not square!"
assert child2['map_size'][0] == child2['map_size'][1], "Child 2 map is not square!"
assert child1['processing_type'] in ["stochastic", "deterministic", "hybrid"], "Invalid processing type!"
print("✓ Mixed crossover successful, all bounds respected, maps are square")

# Test 5: Mixed Mutation
print("\n5. Testing mutate_mixed()...")
mutated_config = mutate_mixed(config1, test_search_space, eta=20.0, mutation_prob=0.5)

print("Original:", config1)
print("Mutated:", mutated_config)

# Validate mutated config
assert 0.0 <= mutated_config['start_learning_rate'] <= 1.0, "Mutated LR out of bounds!"
assert 5 <= mutated_config['map_size'][0] <= 20, "Mutated map size out of bounds!"
assert mutated_config['map_size'][0] == mutated_config['map_size'][1], "Mutated map is not square!"
print("✓ Mixed mutation successful, all bounds respected, maps are square")

print("\n" + "=" * 60)
print("All tests passed! ✓")
print("=" * 60)

# Test 6: Full reproduction cycle
print("\n6. Testing full reproduction cycle (100 iterations)...")
population = [random_config_continuous(test_search_space) for _ in range(10)]

for i in range(100):
    # Select two random parents
    import random
    p1 = random.choice(population)
    p2 = random.choice(population)

    # Crossover
    c1, c2 = crossover_mixed(p1, p2, test_search_space, eta=20.0)

    # Mutation
    c1 = mutate_mixed(c1, test_search_space, eta=20.0, mutation_prob=0.1)
    c2 = mutate_mixed(c2, test_search_space, eta=20.0, mutation_prob=0.1)

    # Validate
    assert 0.0 <= c1['start_learning_rate'] <= 1.0, f"Iteration {i}: Invalid LR!"
    assert 0.0 <= c2['start_learning_rate'] <= 1.0, f"Iteration {i}: Invalid LR!"

print("✓ 100 reproduction cycles completed without errors")

print("\n" + "=" * 60)
print("Continuous EA operators ready for use!")
print("=" * 60)
