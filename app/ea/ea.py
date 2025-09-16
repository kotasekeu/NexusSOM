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

from datetime import datetime
from sklearn.datasets import make_blobs
from multiprocessing import Pool, cpu_count
from som.preprocess import validate_input_data, preprocess_data
from som.som import KohonenSOM
from som.graphs import generate_training_plots
from som.visualization import generate_individual_maps

# Global variables
INPUT_FILE = None
NORMALIZED_DATA = None
WORKING_DIR = None


def validate_and_repair(config: dict) -> dict:
    repaired_config = config.copy()

    if 'start_learning_rate' in repaired_config and 'end_learning_rate' in repaired_config:
        if repaired_config['start_learning_rate'] < repaired_config['end_learning_rate']:
            repaired_config['start_learning_rate'], repaired_config['end_learning_rate'] = \
                repaired_config['end_learning_rate'], repaired_config['start_learning_rate']

    if 'start_batch_percent' in repaired_config and 'end_batch_percent' in repaired_config:
        if repaired_config['start_batch_percent'] > repaired_config['end_batch_percent']:
            repaired_config['start_batch_percent'], repaired_config['end_batch_percent'] = \
                repaired_config['end_batch_percent'], repaired_config['start_batch_percent']

    if 'start_radius' in repaired_config and 'end_radius' in repaired_config:
        if repaired_config['start_radius'] < repaired_config['end_radius']:
            repaired_config['start_radius'], repaired_config['end_radius'] = \
                repaired_config['end_radius'], repaired_config['start_radius']

    return repaired_config

def non_dominated_sort(objectives: np.ndarray) -> list:
    """
    Perform non-dominated sorting for NSGA-II.

    Args:
        objectives: Array of objective values for all individuals.

    Returns:
        List of fronts, each containing indices of individuals.
    """
    n_individuals, n_objectives = objectives.shape

    domination_count = np.zeros(n_individuals, dtype=int)
    dominated_solutions = [[] for _ in range(n_individuals)]

    for p in range(n_individuals):
        for q in range(p + 1, n_individuals):
            p_obj = objectives[p]
            q_obj = objectives[q]

            if np.all(p_obj <= q_obj) and np.any(p_obj < q_obj):
                dominated_solutions[p].append(q)
                domination_count[q] += 1
            elif np.all(q_obj <= p_obj) and np.any(q_obj < p_obj):
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


def log_pareto_front(generation: int, search_space: dict):
    """
    Write the current Pareto front (archive) to a file.

    Args:
        generation: Current generation number.
        search_space: Dictionary of search space parameters.
    """
    global WORKING_DIR
    global ARCHIVE

    log_path = os.path.join(WORKING_DIR, "pareto_front_log.txt")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"--- Generation {generation + 1} | Number of solutions: {len(ARCHIVE)} ---\n")

        # Sort for clarity by the first goal (quantization error)
        sorted_archive = sorted(ARCHIVE, key=lambda x: x[1]['best_mqe'])

        for config, results in sorted_archive:
            uid = results['uid']
            qe = results['best_mqe']
            te = results.get('topographic_error', -1)
            duration = results['training_duration']
            f.write(f"UID: {uid} | QE: {qe:.6f} | TE: {te:.4f} | Time: {duration:.2f}s\n")

            # Print only parameters from the search space
            search_params = {k: v for k, v in config.items() if k in search_space}
            for key, val in sorted(search_params.items()):
                f.write(f"  - {key}: {val}\n")
            f.write("-" * 20 + "\n")
        f.write("\n")


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
            with open(json_path, 'r') as f:
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

def run_evolution(ea_config: dict, data: np.ndarray, ignore_mask: np.ndarray) -> None:
    """
    Main loop of the evolutionary algorithm with Pareto optimization (NSGA-II).

    """
    global ARCHIVE
    global WORKING_DIR

    population_size = ea_config["EA_SETTINGS"]["population_size"]
    generations = ea_config["EA_SETTINGS"]["generations"]

    fixed_params = ea_config["FIXED_PARAMS"]
    search_space = ea_config["SEARCH_SPACE"]

    population = [random_config(search_space) for _ in range(population_size)]

    try:
        for gen in range(generations):
            print(f"Generation {gen + 1}/{generations}")

            # Population evaluation (parallel)
            with Pool(processes=min(12, cpu_count(), population_size)) as pool:
                args = [(ind, i, gen, fixed_params, data, ignore_mask) for i, ind in enumerate(population)]
                results_async = [pool.apply_async(evaluate_individual, arg) for arg in args]
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
            # Get the objectives array (qe, duration) that we want to minimize
            objectives = np.array([
                [
                    res['best_mqe'],
                    res['duration'],
                    res.get('topographic_error', 1.0)  # Default to 1.0 (bad) if not present
                ]
                for cfg, res in combined_population
            ])

            # Sorting into Pareto fronts (ranks)
            fronts = non_dominated_sort(objectives)

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

            # Update archive - contains only individuals from the best front (rank 0)
            ARCHIVE = [combined_population[i] for i in fronts[0]]
            print(f" Best Pareto front has {len(ARCHIVE)} solutions.")
            log_pareto_front(gen, search_space)  # Log the current best front

            # Reproduction (offspring creation)
            mating_pool = []
            # Fill the "mating pool" using tournament selection
            for _ in range(population_size):
                winner = tournament_selection(population, k=3)
                mating_pool.append(winner)

            next_gen_offspring = []
            i = 0
            while i < population_size:
                p1_full = mating_pool[i]
                if i + 1 < population_size:
                    p2_full = mating_pool[i + 1]
                else:
                    # If the number is odd, pair the last one with a random other
                    p2_full = random.choice(mating_pool[:-1])

                # Remove fitness keys before crossover and mutation
                p1_genes = {k: v for k, v in p1_full.items() if k in search_space}
                p2_genes = {k: v for k, v in p2_full.items() if k in search_space}

                child1 = crossover(p1_genes, p2_genes, search_space)
                child2 = crossover(p2_genes, p1_genes, search_space)

                mutated_child1 = mutate(child1, search_space)
                repaired_child1 = validate_and_repair(mutated_child1)
                next_gen_offspring.append(repaired_child1)

                if len(next_gen_offspring) < population_size:
                    mutated_child2 = mutate(child2, search_space)
                    repaired_child2 = validate_and_repair(mutated_child2)
                    next_gen_offspring.append(repaired_child2)

                i += 2

            # New population for the next iteration
            population = next_gen_offspring[:population_size]
    except KeyboardInterrupt:
        print("\nTerminating evolutionary algorithm...")
    except Exception as e:
        print(f"\nFatal error during execution of the evolutionary algorithm: {str(e)}")

    print("Evolution completed.")

def get_uid(config: dict) -> str:
    """
    Generate a unique identifier for a configuration.
    """
    config_str = str(sorted(config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()

def log_message(uid: str, message: str) -> None:
    """
    Log a message to the log file.
    """
    global WORKING_DIR
    log_path = os.path.join(WORKING_DIR, "log.txt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{now}] [{uid}] {message}\n")

def log_result_to_csv(config: dict, results: dict) -> None:
    """
    Log evaluation results to a CSV file.
    """
    global WORKING_DIR
    csv_path = os.path.join(WORKING_DIR, "results.csv")
    uid = results['uid']

    file_exists = os.path.isfile(csv_path)
    base_fields = ['uid', 'best_mqe', 'duration', 'topographic_error',
                   'u_matrix_mean', 'u_matrix_std', 'total_weight_updates']
    with open(csv_path, mode="a", newline="") as f:
        row = {'uid': uid}
        for key in base_fields[1:]:
            row[key] = results.get(key)
        row.update(config)

        fieldnames = ['uid'] + [k for k in base_fields[1:]] + list(config.keys())

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
    with open(progress_path, "a") as f:
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
                     start_time: str = None, end_time: str = None) -> None:
    """
    Log the status of an individual's evaluation to a CSV file.
    """
    global WORKING_DIR
    csv_path = os.path.join(WORKING_DIR, "status.csv")
    write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0

    with open(csv_path, mode="a", newline="") as f:
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
    with open(best_path, "a") as f:
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

def evaluate_individual(ind: dict, population_id: int, generation: int,
                        fixed_params: dict, data: np.ndarray,
                        ignore_mask: np.ndarray) -> tuple:
    """
    Evaluate a single individual (configuration) for SOM training.

    Args:
        ind: Individual configuration dictionary.
        population_id: Index in population.
        generation: Current generation number.
        fixed_params: Fixed parameters for SOM.
        data_config: Data generation parameters.

    Returns:
        Tuple of (training_results, configuration).
    """
    start_time = time.monotonic()
    uid = get_uid(ind)
    
    try:
        print(f"[GEN {generation + 1}] Total RAM used: {psutil.virtual_memory().used // (1024 ** 2)} MB")
        log_status_to_csv(uid, population_id, generation, "started", 
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        global WORKING_DIR
        individual_dir = os.path.join(WORKING_DIR, "individuals", uid)
        os.makedirs(individual_dir, exist_ok=True)

        som_params = {**ind, **fixed_params}

        map_width, map_height = som_params['map_size']
        som_params['m'] = map_width
        som_params['n'] = map_height

        if 'start_radius_init_ratio' in som_params:
            som_params['start_radius'] = max(map_width, map_height) * som_params['start_radius_init_ratio']
            som_params.pop('start_radius_init_ratio')

        som_params.pop('sample_size', None)
        som_params.pop('input_dim', None)
        som_params.pop('map_size', None)

        som = KohonenSOM(dim=data.shape[1], **som_params)


        training_results = som.train(data, ignore_mask=ignore_mask, working_dir=individual_dir)

        training_results['uid'] = uid

        topographic_error = som.calculate_topographic_error(data, mask=ignore_mask)
        u_matrix_metrics = som.calculate_u_matrix_metrics()

        training_results['topographic_error'] = topographic_error
        training_results.update(u_matrix_metrics)

        training_results['training_duration'] = training_results.get('duration', None)

        log_message(uid, f"Evaluated – QE: {training_results['best_mqe']:.6f}, TE: {topographic_error:.4f}, Time: {training_results['duration']:.2f}s")
        log_result_to_csv(ind, training_results)

        log_status_to_csv(uid, population_id, generation, "completed", 
                         datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        generate_training_plots(training_results, individual_dir)

        generate_individual_maps(som, data, ignore_mask, individual_dir)

        return (training_results, copy.deepcopy(ind))

    except Exception as e:
        import traceback
        log_message(uid, f"ERROR during evaluation: {e}")
        log_message(uid, f"Full traceback:\n{traceback.format_exc()}")

        log_status_to_csv(uid, population_id, generation, "failed", 
                         datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        log_message(uid, f"Error during evaluation: {str(e)}")
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evolutionary optimization of the SOM algorithm')
    parser.add_argument('-i', '--input', help='Path to the input CSV file')
    parser.add_argument('-c', '--config', help='Path to a custom configuration file (JSON)')
    args = parser.parse_args()

    if args.input:
        if not os.path.exists(args.input):
            print(f"Error: Input file {args.input} does not exist.")
            sys.exit(1)
        INPUT_FILE = args.input

    WORKING_DIR = get_working_directory(INPUT_FILE)
    config = load_configuration(args.config)
    preprocess_config = config.get("PREPROCES_DATA", {})

    print("INFO: Preparing data for evolution...")
    if INPUT_FILE:
        input_data_df = validate_input_data(INPUT_FILE, WORKING_DIR, preprocess_config)
        training_data_path, _, ignore_mask = preprocess_data(input_data_df, preprocess_config, WORKING_DIR)
        loaded_data = np.load(training_data_path)
        config['PREPROCES_DATA'] = preprocess_config

        print(
            f"INFO: Data loaded and normalized from file {training_data_path}. Shape: {loaded_data.shape}")
    else:
        data_params = config.get("DATA_PARAMS", {})
        sample_size = data_params.get("sample_size", 1000)
        input_dim = data_params.get("input_dim", 10)
        loaded_data = get_or_generate_data(sample_size, input_dim)
        ignore_mask = None
        print(f"INFO: Data has been generated. Number of samples: {sample_size}, dimension: {input_dim}")

    ARCHIVE = []

    run_evolution(config, loaded_data, ignore_mask)