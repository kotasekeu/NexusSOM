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

from datetime import datetime
from sklearn.datasets import make_blobs
from som.kohonen import KohonenSOM
from multiprocessing import Pool, cpu_count

# import shutil
# from sklearn.metrics import pairwise_distances_argmin_min
# import sys
# from som.project_processor import normalize_data
# import pandas as pd
# from os.path import exists
# from som.utils import set_uid_hash

# Váhy pro multi-kriteriální fitness funkci
W_ERROR = 0.7  # váha pro kvantizační chybu
W_TIME = 0.3   # váha pro dobu výpočtu

# Globální proměnné
INPUT_FILE = None
NORMALIZED_DATA = None
WORKING_DIR = None


# --- NOVÉ POMOCNÉ FUNKCE (musíte je implementovat) ---
# Níže uvedené funkce jsou pro logiku NSGA-II klíčové.
# Můžete pro ně najít hotové implementace online nebo v knihovnách jako 'deap'.

def non_dominated_sort(objectives: np.ndarray) -> list:
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
    # Smyčka poběží, dokud bude poslední vytvořená fronta obsahovat nějaké jedince
    while fronts[front_index]:
        next_front = []
        # Projdeme jedince v aktuální (poslední vytvořené) frontě
        for p in fronts[front_index]:
            # Projdeme všechny, které dominují
            for q in dominated_solutions[p]:
                domination_count[q] -= 1
                # Pokud už jedince q nikdo další nedominuje, patří do další fronty
                if domination_count[q] == 0:
                    next_front.append(q)

        # Zvýšíme index pro další iteraci
        front_index += 1

        # Přidáme novou frontu do seznamu front.
        # Pokud byla prázdná, cyklus v další iteraci skončí.
        fronts.append(next_front)

    # Poslední přidaná fronta bude vždy prázdná, takže ji odstraníme
    return fronts[:-1]


def crowding_distance_assignment(objectives: np.ndarray, fronts: list) -> np.ndarray:
    """
    Vypočítá crowding distance pro každého jedince pro udržení diverzity.

    Args:
        objectives: Stejné pole jako v non_dominated_sort.
        fronts: Výstup z non_dominated_sort.

    Returns:
        Numpy pole (vektor) s hodnotami crowding distance pro každého jedince.
    """
    n_individuals, n_objectives = objectives.shape
    crowding_distances = np.zeros(n_individuals)

    for front in fronts:
        if not front:
            continue

        front_objectives = objectives[front, :]
        n_front_members = len(front)

        for m in range(n_objectives):
            # Seřadíme jedince ve frontě podle aktuálního cíle
            sorted_indices = np.argsort(front_objectives[:, m])

            # Extrémní body mají nekonečnou vzdálenost, aby byly vždy preferovány
            crowding_distances[front[sorted_indices[0]]] = np.inf
            crowding_distances[front[sorted_indices[-1]]] = np.inf

            if n_front_members > 2:
                # Normalizační faktor
                obj_range = front_objectives[sorted_indices[-1], m] - front_objectives[sorted_indices[0], m]
                if obj_range < 1e-8:  # Vyhneme se dělení nulou
                    continue

                # Pro ostatní body
                for i in range(1, n_front_members - 1):
                    dist = front_objectives[sorted_indices[i + 1], m] - front_objectives[sorted_indices[i - 1], m]
                    crowding_distances[front[sorted_indices[i]]] += dist / obj_range

    return crowding_distances


def tournament_selection(population: list, k: int = 3) -> dict:
    """
    Provede turnajovou selekci na základě ranku a crowding distance.

    Args:
        population: Seznam jedinců (slovníků), kde každý má klíče 'rank' a 'crowding_distance'.
        k: Velikost turnaje.

    Returns:
        Vítězný jedinec (slovník).
    """
    # Náhodně vybereme účastníky turnaje
    participants = random.sample(population, k)

    best_participant = participants[0]

    for i in range(1, k):
        # Porovnáme s aktuálně nejlepším
        p = participants[i]
        # Preferujeme lepší (nižší) rank
        if p['rank'] < best_participant['rank']:
            best_participant = p
        # Při stejném ranku preferujeme větší crowding distance pro diverzitu
        elif p['rank'] == best_participant['rank'] and \
                p['crowding_distance'] > best_participant['crowding_distance']:
            best_participant = p

    return best_participant


def log_pareto_front(generation: int, search_space: dict):
    """
    Zapíše aktuální Pareto frontu (archiv) do souboru.

    Args:
        archive: Seznam (config, results) tuples.
        generation: Aktuální číslo generace.
    """
    global WORKING_DIR
    global ARCHIVE

    log_path = os.path.join(WORKING_DIR, "pareto_front_log.txt")

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"--- Generace {generation + 1} | Počet řešení: {len(ARCHIVE)} ---\n")

        # Seřadíme pro přehlednost podle prvního cíle (kvantizační chyba)
        sorted_archive = sorted(ARCHIVE, key=lambda x: x[1]['final_mqe'])

        for config, results in sorted_archive:
            qe = results['final_mqe']
            duration = results['training_duration']
            f.write(f"QE: {qe:.8f} | Čas: {duration:.2f}s\n")

            # Vypíšeme pouze parametry z prohledávacího prostoru
            search_params = {k: v for k, v in config.items() if k in search_space}
            for key, val in sorted(search_params.items()):
                f.write(f"  - {key}: {val}\n")
            f.write("-" * 20 + "\n")
        f.write("\n")


def get_working_directory(input_file: str = None) -> str:
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
        Načte konfiguraci. Priorita: JSON > ea_config.py.
        """
    if json_path:
        print(f"INFO: Používám konfigurační soubor z argumentu: {json_path}")
        if not os.path.exists(json_path):
            print(f"ERROR: Konfigurační soubor {json_path} neexistuje.")
            sys.exit(1)
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"ERROR: Soubor {json_path} není validní JSON: {e}")
            sys.exit(1)
    else:
        print("INFO: Argument --config nebyl zadán, hledám ea_config.py.")
        try:
            from ea_config import CONFIG
            print("INFO: Soubor ea_config.py nalezen a načten.")
            return CONFIG
        except ImportError:
            print("ERROR: Nebyl zadán konfigurační soubor a výchozí soubor ea_config.py nebyl nalezen.")
            print("Řešení: Vytvořte ea_config.py nebo použijte argument --config <cesta_k_json> dle informací v dokumentaci.")
            sys.exit(1)


def crossover(parent1: dict, parent2: dict, param_space: dict) -> dict:
    child = {}
    for key in param_space:
        if isinstance(param_space[key], list):
            child[key] = random.choice([parent1[key], parent2[key]])
        else:
            child[key] = parent1[key]
    return child

def random_config(param_space: dict) -> dict:
    config = {}
    for key, value in param_space.items():
        if isinstance(value, list):
            config[key] = random.choice(value)
        else:
            config[key] = value

    # Validace a oprava min/max batch
    if 'min_batch_percent' in config and 'max_batch_percent' in config:
        if config['min_batch_percent'] > config['max_batch_percent']:
            # Nejjednodušší oprava: prohodit hodnoty, varianta generovat tak dlouho, dokud nebude správné pořadí
            config['min_batch_percent'], config['max_batch_percent'] = \
                config['max_batch_percent'], config['min_batch_percent']
    return config

def mutate(config: dict, param_space: dict) -> dict:

    key = random.choice(list(param_space.keys()))
    if isinstance(param_space[key], list):
        config[key] = random.choice(param_space[key])
    return config

def run_evolution(ea_config: dict) -> None:
    """
    Hlavní smyčka evolučního algoritmu s Pareto optimalizací (NSGA-II).

    Args:
        config: Slovník s konfigurací (obsahuje EA_SETTINGS, SEARCH_SPACE, FIXED_PARAMS).
    """
    global ARCHIVE
    global WORKING_DIR

    population_size = ea_config["EA_SETTINGS"]["population_size"]
    generations = ea_config["EA_SETTINGS"]["generations"]

    fixed_params = ea_config["FIXED_PARAMS"]
    search_space = ea_config["SEARCH_SPACE"]
    data_params = ea_config["DATA_PARAMS"]

    # Získání nastavení z nové struktury konfigurace


    # 1. Inicializace populace
    population = [random_config(search_space) for _ in range(population_size)]

    # try:
    for gen in range(generations):
        print(f"Generace {gen + 1}/{generations}")

        # 2. Vyhodnocení populace (paralelně)
        with Pool(processes=min(12, cpu_count(), population_size)) as pool:
            # Předáváme i fixní parametry do každého procesu
            args = [(ind, i, gen, fixed_params, data_params) for i, ind in enumerate(population)]
            results_async = [pool.apply_async(evaluate_individual, arg) for arg in args]
            evaluated_population = []
            for r in results_async:
                try:
                    # Očekáváme návrat (results_dict, config_dict)
                    training_results, config = r.get(timeout=3600)
                    evaluated_population.append((config, training_results))
                except Exception as e:
                    print(f"[CHYBA] Jedinec selhal: {e}")

        if not evaluated_population:
            print("Chyba: Žádný jedinec nebyl úspěšně vyhodnocen. Ukončuji.")
            return

        # 3. Kombinace populace a archivu (Elitářství)
        # Spojíme aktuální vyhodnocenou populaci s nejlepšími jedinci z minulých generací
        combined_population = evaluated_population + ARCHIVE

        # 4. Výpočet fitness pomocí Non-dominated Sorting a Crowding Distance
        # Získáme pole cílů (qe, duration), které chceme minimalizovat
        objectives = np.array([
            [res['final_mqe'], res['training_duration']]
            for cfg, res in combined_population
        ])

        # A. Seřazení do Pareto front (ranků)
        fronts = non_dominated_sort(objectives)

        # B. Výpočet "vzdálenosti od sousedů" pro udržení diverzity
        crowding_distances = crowding_distance_assignment(objectives, fronts)

        # C. Přiřazení ranku a crowding distance každému jedinci
        for i, front in enumerate(fronts):
            for individual_idx in front:
                # Přidáváme fitness metriky přímo do slovníku s konfigurací
                combined_population[individual_idx][0]['rank'] = i
                combined_population[individual_idx][0]['crowding_distance'] = crowding_distances[individual_idx]

        # 5. Selekce pro další generaci
        # Seřadíme kombinovanou populaci: primárně podle ranku (vzestupně), sekundárně podle crowding distance (sestupně)
        combined_population.sort(key=lambda x: (x[0]['rank'], -x[0]['crowding_distance']))

        # A. Nová populace je prvních N nejlepších jedinců z seřazeného seznamu
        # Uchováváme pouze slovníky s konfigurací, ne výsledky
        population = [cfg for cfg, res in combined_population[:population_size]]

        # B. Aktualizujeme archiv - obsahuje pouze jedince z nejlepší fronty (rank 0)
        ARCHIVE = [combined_population[i] for i in fronts[0]]
        print(f" Nejlepší Pareto fronta má {len(ARCHIVE)} řešení.")
        log_pareto_front(gen, search_space)  # Zápis aktuální nejlepší fronty

        # 6. Reprodukce (vytvoření potomků)
        mating_pool = []
        # Naplníme "mating pool" pomocí turnajové selekce
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
                # Pokud je počet lichý, spárujeme posledního s náhodným jiným
                p2_full = random.choice(mating_pool[:-1])

            # Před křížením a mutací odstraníme fitness klíče
            p1_genes = {k: v for k, v in p1_full.items() if k in search_space}
            p2_genes = {k: v for k, v in p2_full.items() if k in search_space}

            child1 = crossover(p1_genes, p2_genes, search_space)
            child2 = crossover(p2_genes, p1_genes, search_space)

            next_gen_offspring.append(mutate(child1, search_space))
            next_gen_offspring.append(mutate(child2, search_space))
            i += 2

        # 7. Nová populace pro další iteraci
        population = next_gen_offspring[:population_size]

    # except KeyboardInterrupt:
    #     print("\nUkončuji evoluční algoritmus...")
    # except Exception as e:
    #     print(f"\nFatální chyba při běhu evolučního algoritmu: {str(e)}")

    print("Evoluce dokončena.")

def get_uid(config: dict) -> str:

    config_str = str(sorted(config.items()))
    return hashlib.md5(config_str.encode()).hexdigest()

def log_message(uid: str, message: str) -> None:
    global WORKING_DIR
    log_path = os.path.join(WORKING_DIR, "log.txt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{now}] [{uid}] {message}\n")

def log_result_to_csv(uid: str, config: dict, score: float, duration: float, total_weight_updates: int) -> None:
    global WORKING_DIR
    csv_path = os.path.join(WORKING_DIR, "results.csv")

    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode="a", newline="") as f:
        fieldnames = ['uid', 'best MQE', 'duration', 'total_weight_updates'] + list(config.keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        row = {'uid': uid, 'best MQE': score, 'duration': duration, 'total_weight_updates': total_weight_updates, **config}
        writer.writerow(row)

def log_progress(current: int, total: int) -> None:
    global WORKING_DIR
    progress_path = os.path.join(WORKING_DIR, "progress.log")
    with open(progress_path, "a") as f:
        f.write(f"{current}/{total} dokončeno\n")

def get_or_generate_data(sample_size: int, input_dim: int) -> np.ndarray:
    global WORKING_DIR
    file_name = f"data_{sample_size}x{input_dim}.npy"
    file_path = os.path.join(WORKING_DIR, file_name)

    if os.path.exists(file_path):
        return np.load(file_path)

    data, _ = make_blobs(n_samples=sample_size, n_features=input_dim, centers=5)
    np.save(file_path, data)
    log_message("SYSTEM", f"Vygenerována nová data: {file_name}")
    return data

def log_status_to_csv(uid: str, population_id: int, generation: int, status: str, 
                     start_time: str = None, end_time: str = None) -> None:
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
    global NORMALIZED_DATA
    global WORKING_DIR
    
    if NORMALIZED_DATA is not None:
        return NORMALIZED_DATA

    preprocess_file = os.path.join(WORKING_DIR, "preprocess-input.csv")
    if not os.path.exists(preprocess_file):
        preprocess_file = normalize_data(input_file, {}, {})
    NORMALIZED_DATA = pd.read_csv(preprocess_file, delimiter=',').values
    log_message("SYSTEM", f"Načtena a normalizována data z externího souboru: {input_file}")
    return NORMALIZED_DATA

def extract_uid_from_path(file_path: str) -> str:
    parts = file_path.split('/')
    for part in parts:
        if part.startswith('nxmpp'):
            return part
    return None

def evaluate_individual(ind: dict, population_id: int, generation: int, fixed_params: dict, data_config: dict) -> tuple:

    start_time = time.monotonic()
    uid = get_uid(ind)
    
    try:
        print(f"[GEN {generation + 1}] Total RAM used: {psutil.virtual_memory().used // (1024 ** 2)} MB")
        log_status_to_csv(uid, population_id, generation, "started", 
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if INPUT_FILE:
            data = load_input_data(INPUT_FILE)
        else:
            if "sample_size" in data_config and "input_dim" in data_config:
                sample_size = data_config["sample_size"]
                input_dim = data_config["input_dim"]
                data = get_or_generate_data(sample_size, input_dim)
            else:
                raise ValueError("Není použit vstupní soubor ani nastaveny rozměry pro generování automaticky generovaného souboru")

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

        training_results = som.train(data)
        
        duration = time.monotonic() - start_time
        log_message(uid, f"Konfigurace vyhodnocena – kvantizační chyba: {som.best_mqe:.8f}, čas: {duration:.2f}s")
        log_result_to_csv(uid, ind, som.best_mqe, duration, som.total_weight_updates)
        
        log_status_to_csv(uid, population_id, generation, "completed", 
                         datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        return (training_results, copy.deepcopy(ind))

    except Exception as e:
        log_status_to_csv(uid, population_id, generation, "failed", 
                         datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S"),
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        log_message(uid, f"Chyba při vyhodnocování: {str(e)}")
        raise e

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evoluční optimalizace SOM algoritmu')
    parser.add_argument('-i', '--input', help='Cesta k vstupnímu CSV souboru')
    parser.add_argument('-c', '--config', help='Cesta k vlastnímu konfiguračnímu souboru (JSON)')
    args = parser.parse_args()

    if args.input:
        if not os.path.exists(args.input):
            print(f"Chyba: Vstupní soubor {args.input} neexistuje.")
            sys.exit(1)
        INPUT_FILE = args.input

    WORKING_DIR = get_working_directory(INPUT_FILE)

    config = load_configuration(args.config)

    ARCHIVE = []

    run_evolution(config)