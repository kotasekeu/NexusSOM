# visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from matplotlib.patches import Rectangle, RegularPolygon, Wedge, Circle, Patch
from matplotlib.colors import Normalize, ListedColormap
from som import KohonenSOM


# --- CENTRÁLNÍ KRESLÍCÍ FUNKCE ---

def _create_map(som: KohonenSOM, values: np.ndarray, title: str, output_file: str,
                cmap: str, cbar_label: str = None, show_text: list = None):
    """
    Univerzální funkce pro vykreslení jakékoliv SOM mapy (U-Matrix, Hitmap, atd.).
    """
    m, n, map_type = som.m, som.n, som.map_type

    fig, ax = plt.subplots(figsize=(n * 1.2, m * 1.2))  # Dynamická velikost
    ax.set_aspect('equal')
    ax.axis('off')

    # Výpočet pozic a velikostí
    if map_type == 'hex':
        side_len = 0.5
        radius = side_len / np.cos(np.pi / 6)
        dy = 2 * radius * (3 / 4)
        patches = []
        for i in range(m):
            for j in range(n):
                x = j * (2 * side_len) + (i % 2) * side_len
                y = i * dy
                patches.append(RegularPolygon((x, y), numVertices=6, radius=radius))
    else:  # square
        side_len = 1.0
        patches = []
        for i in range(m):
            for j in range(n):
                patches.append(Rectangle((j - side_len / 2, i - side_len / 2), side_len, side_len))

    # Vytvoření kolekce patchů s barvami
    collection = plt.matplotlib.collections.PatchCollection(patches)
    collection.set_array(values.flatten())
    collection.set_cmap(cmap)
    collection.set_edgecolor('white')
    ax.add_collection(collection)

    ax.autoscale_view()

    # Zobrazení textových hodnot (např. pro Hit Map)
    if show_text is not None:
        coords = np.array([p.get_center() for p in patches])
        for i, txt in enumerate(show_text):
            if txt > 0:
                ax.text(coords[i, 0], coords[i, 1], str(int(txt)),
                        color='red', ha='center', va='center', weight='bold')

    # Uložení hlavní mapy
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

    # Uložení samostatné legendy (colorbar)
    if cbar_label:
        fig_legend, ax_legend = plt.subplots(figsize=(1.5, 6))
        norm = Normalize(vmin=values.min(), vmax=values.max())
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_legend,
                            orientation='vertical', label=cbar_label)
        legend_path = os.path.join(os.path.dirname(output_file), "legends", os.path.basename(output_file))
        os.makedirs(os.path.dirname(legend_path), exist_ok=True)
        fig_legend.savefig(legend_path, dpi=150, bbox_inches='tight')
        plt.close(fig_legend)


# --- WRAPPER FUNKCE PRO JEDNOTLIVÉ MAPY ---

def generate_u_matrix(som: KohonenSOM, output_file: str):
    """Vektorizovaný výpočet a vykreslení U-Matrix."""
    m, n, weights = som.m, som.n, som.weights
    u_matrix = np.zeros((m, n))

    # Vektorizovaný výpočet vzdáleností k sousedům
    # Vertikální sousedi
    diffs_v = np.linalg.norm(weights[1:, :, :] - weights[:-1, :, :], axis=2)
    u_matrix[1:, :] += diffs_v
    u_matrix[:-1, :] += diffs_v
    # Horizontální sousedi
    diffs_h = np.linalg.norm(weights[:, 1:, :] - weights[:, :-1, :], axis=2)
    u_matrix[:, 1:] += diffs_h
    u_matrix[:, :-1] += diffs_h

    # Normalizace počtem sousedů (rohy 2, hrany 3, střed 4)
    counts = np.full((m, n), 4)
    counts[[0, -1], :] -= 1
    counts[:, [0, -1]] -= 1
    u_matrix /= counts

    _create_map(som, u_matrix, "U-Matrix", output_file, cmap='viridis', cbar_label="Průměrná vzdálenost k sousedům")


def generate_hit_map(som: KohonenSOM, normalized_data: np.ndarray, output_file: str):
    """Vektorizovaný výpočet a vykreslení Hit Mapy."""
    num_neurons = som.m * som.n
    flat_weights = som.weights.reshape(num_neurons, som.dim)

    dists = np.linalg.norm(normalized_data[:, np.newaxis, :] - flat_weights[np.newaxis, :, :], axis=2)
    bmu_indices = np.argmin(dists, axis=1)

    hit_counts = np.bincount(bmu_indices, minlength=num_neurons).reshape(som.m, som.n)

    _create_map(som, hit_counts, "Hit Map", output_file, cmap='Blues',
                cbar_label="Počet přiřazených vzorků", show_text=hit_counts.flatten())


def generate_component_planes(som: KohonenSOM, original_df: pd.DataFrame, config: dict, output_dir: str):
    """Vykreslí komponentní roviny pro všechny numerické sloupce."""
    numerical_cols = config.get('numerical_column', [])
    weights_reshaped = som.weights.reshape(som.m, som.n, -1)

    for i, col_name in enumerate(numerical_cols):
        plane_values = weights_reshaped[:, :, i]

        # De-normalizace hodnot pro lepší interpretaci v legendě
        col_min = original_df[col_name].min()
        col_max = original_df[col_name].max()
        de_normalized_values = plane_values * (col_max - col_min) + col_min

        output_file = os.path.join(output_dir, f"component_{col_name}.png")
        _create_map(som, de_normalized_values, f"Component Plane: {col_name}", output_file,
                    cmap='coolwarm', cbar_label=f"Hodnota váhy pro {col_name}")


def generate_pie_maps(som: KohonenSOM, config: dict, working_dir: str, output_dir: str):
    """Vykreslí koláčové mapy pro všechny kategorické sloupce."""
    # Implementace této funkce je komplexnější, prozatím placeholder
    print("INFO: Generování koláčových map (TODO)...")
    pass


def generate_component_planes(som: KohonenSOM, original_df: pd.DataFrame, config: dict, output_dir: str):
    """
    Vykreslí komponentní roviny pro VŠECHNY sloupce použité v tréninku.
    """
    # Získáme seznam všech sloupců, které reálně vstoupily do SOM
    # (po odstranění primary_id v preprocess.py)
    # TENTO SEZNAM MUSÍ ODPOVÍDAT POŘADÍ DIMENZÍ V SOM
    primary_id_col = config.get('primary_id')
    training_columns = [col for col in original_df.columns if col != primary_id_col]

    weights_reshaped = som.weights.reshape(som.m, som.n, -1)

    # Ujistíme se, že máme správný počet sloupců
    if len(training_columns) != som.dim:
        print(
            f"WARNING: Počet trénovacích sloupců ({len(training_columns)}) nesouhlasí s dimenzí SOM ({som.dim}). Component planes mohou mít špatné popisky.")
        training_columns = [f"dim_{i}" for i in range(som.dim)]

    for i, col_name in enumerate(training_columns):
        plane_values = weights_reshaped[:, :, i]

        # De-normalizace hodnot pro lepší interpretaci v legendě (pouze pro numerické sloupce)
        if col_name in config.get('numerical_column', []):
            col_min = original_df[col_name].min()
            col_max = original_df[col_name].max()
            de_normalized_values = plane_values * (col_max - col_min) + col_min
            cbar_label_text = f"Hodnota váhy pro {col_name}"
        else:
            # Pro kategorické sloupce zobrazíme normalizované váhy (0-1)
            de_normalized_values = plane_values
            cbar_label_text = f"Hodnota váhy pro {col_name} (kategorický)"

        output_file = os.path.join(output_dir, f"component_{col_name}.png")
        _create_map(som, de_normalized_values, f"Component Plane: {col_name}", output_file,
                    cmap='coolwarm', cbar_label=cbar_label_text)


def generate_cluster_map(som: KohonenSOM, clusters: dict, output_file: str):
    """Vykreslí mapu aktivních neuronů (shluků)."""
    m, n = som.m, som.n

    # Vytvoříme mapu, kde každý aktivní neuron bude mít unikátní celé číslo
    labels = np.full((m, n), -1, dtype=int)  # -1 pro neaktivní neurony
    active_neuron_keys = sorted(clusters.keys())

    for idx, key in enumerate(active_neuron_keys):
        i, j = map(int, key.split('_'))
        if 0 <= i < m and 0 <= j < n:
            labels[i, j] = idx

    # Vytvoříme diskrétní barevnou mapu
    num_clusters = len(active_neuron_keys)
    if num_clusters == 0:
        print("WARNING: Žádné aktivní shluky k vykreslení v Cluster Map.")
        return

    cmap = plt.get_cmap('viridis', num_clusters)

    _create_map(som, labels, "Cluster Map", output_file, cmap=cmap)


def generate_all_maps(som: KohonenSOM, original_df: pd.DataFrame, normalized_data: np.ndarray, config: dict,
                      working_dir: str):
    """Hlavní orchestrátor pro generování všech map."""
    maps_dir = os.path.join(working_dir, "visualizations")
    os.makedirs(maps_dir, exist_ok=True)

    print("INFO: Generuji vizualizace map...")

    # 1. U-Matrix
    generate_u_matrix(som, os.path.join(maps_dir, "u_matrix.png"))

    # 2. Hit Map
    generate_hit_map(som, normalized_data, os.path.join(maps_dir, "hit_map.png"))

    # 3. Component Planes (nyní pro všechny sloupce)
    generate_component_planes(som, original_df, config, maps_dir)

    # 4. Cluster Map (přidáno zpět)
    clusters_path = os.path.join(working_dir, "clusters.json")
    if os.path.exists(clusters_path):
        with open(clusters_path, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
        generate_cluster_map(som, clusters, os.path.join(maps_dir, "cluster_map.png"))

    # 5. Pie Maps
    generate_pie_maps(som, config, working_dir, maps_dir)

    print("INFO: Generování map dokončeno.")