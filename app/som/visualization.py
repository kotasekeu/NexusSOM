# visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from matplotlib.patches import Rectangle, RegularPolygon, Wedge, Circle, Patch
from matplotlib.colors import Normalize, ListedColormap
from som.som import KohonenSOM


# Central drawing function for SOM maps
def _create_map(som: KohonenSOM, values: np.ndarray, title: str, output_file: str,
                cmap: str, cbar_label: str = None, show_text: list = None, show_title: bool = True):
    """
    Universal function for rendering any SOM map (U-Matrix, Hitmap, etc.).

    Args:
        show_title: If True, display title above the map. Set to False for CNN training data.
    """
    m, n, map_type = som.m, som.n, som.map_type

    fig, ax = plt.subplots(figsize=(n * 1.2, m * 1.2))
    ax.set_aspect('equal')
    ax.axis('off')

    # Calculate neuron positions and sizes
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

    # Create patch collection with colors
    collection = plt.matplotlib.collections.PatchCollection(patches)
    collection.set_array(values.flatten())
    collection.set_cmap(cmap)
    collection.set_edgecolor('white')
    ax.add_collection(collection)

    ax.autoscale_view()

    # Display text values (e.g., for Hit Map)
    if show_text is not None:
        coords = np.array([p.xy for p in patches])
        for i, txt in enumerate(show_text):
            if txt > 0:
                ax.text(coords[i, 0], coords[i, 1], str(int(txt)),
                        color='red', ha='center', va='center', weight='bold')

    # Save main map
    if show_title:
        plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

    # Save separate legend (colorbar)
    if cbar_label:
        fig_legend, ax_legend = plt.subplots(figsize=(1.5, 6))
        norm = Normalize(vmin=values.min(), vmax=values.max())
        cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_legend,
                            orientation='vertical', label=cbar_label)
        legend_path = os.path.join(os.path.dirname(output_file), "legends", os.path.basename(output_file))
        os.makedirs(os.path.dirname(legend_path), exist_ok=True)
        fig_legend.savefig(legend_path, dpi=150, bbox_inches='tight')
        plt.close(fig_legend)


def generate_categorical_legend(categories: dict, cmap_name: str, title: str, output_file: str):
    """Generates a standalone legend for categorical data."""
    fig, ax = plt.subplots(figsize=(3, max(2, len(categories) * 0.4)))
    ax.axis('off')

    cmap = plt.get_cmap(cmap_name)
    handles = [
        Patch(facecolor=cmap(i / (len(categories) - 1 if len(categories) > 1 else 1)), label=label)
        for i, label in enumerate(categories.values())
    ]

    ax.legend(handles=handles, title=title, loc='center', frameon=False, fontsize='large')

    legend_path = os.path.join(os.path.dirname(output_file), "legends", os.path.basename(output_file))
    os.makedirs(os.path.dirname(legend_path), exist_ok=True)
    fig.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def generate_u_matrix(som: KohonenSOM, output_file: str, show_title: bool = True):
    """Calculates and visualizes the U-Matrix, supporting both square and hex grids."""
    m, n, weights = som.m, som.n, som.weights
    u_matrix = np.zeros((m, n))

    # Iterate over each neuron to calculate its average distance to neighbors
    for i in range(m):
        for j in range(n):
            neuron_weights = weights[i, j]
            neighbor_distances = []

            # Define neighbors based on map type
            if som.map_type == 'hex':
                # Hexagonal neighbors have complex offsets
                if i % 2 == 0:  # Even rows
                    neighbors = [(i - 1, j), (i - 1, j + 1), (i, j - 1), (i, j + 1), (i + 1, j), (i + 1, j + 1)]
                else:  # Odd rows
                    neighbors = [(i - 1, j - 1), (i - 1, j), (i, j - 1), (i, j + 1), (i + 1, j - 1), (i + 1, j)]
            else:  # Square neighbors
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]

            # Calculate distances to valid neighbors
            for ni, nj in neighbors:
                if 0 <= ni < m and 0 <= nj < n:
                    neighbor_weights = weights[ni, nj]
                    neighbor_distances.append(np.linalg.norm(neuron_weights - neighbor_weights))

            if neighbor_distances:
                u_matrix[i, j] = np.mean(neighbor_distances)

    _create_map(som, u_matrix, "U-Matrix", output_file, cmap='viridis',
                cbar_label="Average distance to neighbors", show_title=show_title)


def generate_hit_map(som: KohonenSOM, normalized_data: np.ndarray, output_file: str):
    """Vectorized calculation and rendering of Hit Map."""
    num_neurons = som.m * som.n
    flat_weights = som.weights.reshape(num_neurons, som.dim)

    dists = np.linalg.norm(normalized_data[:, np.newaxis, :] - flat_weights[np.newaxis, :, :], axis=2)
    bmu_indices = np.argmin(dists, axis=1)

    hit_counts = np.bincount(bmu_indices, minlength=num_neurons).reshape(som.m, som.n)

    _create_map(som, hit_counts, "Hit Map", output_file, cmap='Blues',
                cbar_label="Number of assigned samples", show_text=hit_counts.flatten())


def generate_component_planes(som: KohonenSOM, original_df: pd.DataFrame, config: dict, output_dir: str):
    """
    Render component planes for ALL columns used in training.
    """
    # Get all columns that were actually used for SOM training
    training_columns = [col for col in original_df.columns]
    primary_id_col = config.get('primary_id', 'primary_id')

    try:
        pid_index = training_columns.index(primary_id_col)
    except ValueError:
        pid_index = -1

    weights_reshaped = som.weights.reshape(som.m, som.n, -1)

    # Ensure correct number of columns
    if len(training_columns) != som.dim:
        print(
            f"WARNING: Number of training columns ({len(training_columns)}) does not match SOM dimension ({som.dim}). Component planes may have incorrect labels.")
        training_columns = [f"dim_{i}" for i in range(som.dim)]

    for i, col_name in enumerate(training_columns):
        if i == pid_index:
            continue
        plane_values = weights_reshaped[:, :, i]

        # Denormalize values for better legend interpretation (only for numerical columns)
        if col_name in config.get('numerical_column', []):
            col_min = original_df[col_name].min()
            col_max = original_df[col_name].max()
            de_normalized_values = plane_values * (col_max - col_min) + col_min
            cbar_label_text = f"Weight value for {col_name}"
        else:
            # For categorical columns show normalized weights (0-1)
            de_normalized_values = plane_values
            cbar_label_text = f"Weight value for {col_name} (categorical)"

        output_file = os.path.join(output_dir, f"component_{col_name}.png")
        _create_map(som, de_normalized_values, f"Component Plane: {col_name}", output_file,
                    cmap='coolwarm', cbar_label=cbar_label_text)


def generate_pie_map(som: KohonenSOM, pie_data: dict, output_file: str, cmap_name: str = 'tab20b'):
    """
    Visualizes categorical data distribution on the SOM grid using pie charts.
    """
    m, n, map_type = som.m, som.n, som.map_type

    fig, ax = plt.subplots(figsize=(n * 1.2, m * 1.2))
    ax.set_aspect('equal')
    ax.axis('off')

    if map_type == 'hex':
        side_len = 0.5;
        radius = side_len / np.cos(np.pi / 6);
        dy = 2 * radius * (3 / 4)
        patches = [RegularPolygon((j * (2 * side_len) + (i % 2) * side_len, i * dy), numVertices=6, radius=radius) for i
                   in range(m) for j in range(n)]
    else:  # square
        side_len = 1.0
        patches = [Rectangle((j - side_len / 2, i - side_len / 2), side_len, side_len) for i in range(m) for j in
                   range(n)]

    bg_collection = plt.matplotlib.collections.PatchCollection(patches, facecolor='#f0f0f0', edgecolor='white')
    ax.add_collection(bg_collection)
    ax.autoscale_view()

    categories = pie_data['categories']
    cat_keys = sorted(categories.keys(), key=int)
    num_categories = len(cat_keys)
    cmap = plt.get_cmap(cmap_name)

    coords = np.array([p.get_center() if map_type == 'square' else p.xy for p in patches]).reshape(m, n, 2)
    pie_radius = (coords[0, 1, 0] - coords[0, 0, 0]) * 0.45 if n > 1 else 0.45

    for pos, counts in pie_data['counts'].items():
        i, j = map(int, pos.split('_'))
        center_x, center_y = coords[i, j]

        valid_counts = {k: v for k, v in counts.items() if v > 0}

        if not valid_counts:
            continue

        total = sum(valid_counts.values())

        if len(valid_counts) == 1:
            key = list(valid_counts.keys())[0]
            color_idx = cat_keys.index(key)
            color = cmap(color_idx / (num_categories - 1 if num_categories > 1 else 1))

            circle = Circle((center_x, center_y), pie_radius,
                            facecolor=color,
                            edgecolor='white',
                            linewidth=0.5)
            ax.add_patch(circle)

        else:
            angle_start = 90
            for key in cat_keys:
                if key in valid_counts:
                    angle_end = angle_start + 360 * (valid_counts[key] / total)
                    color_idx = cat_keys.index(key)
                    color = cmap(color_idx / (num_categories - 1 if num_categories > 1 else 1))

                    wedge = Wedge((center_x, center_y), pie_radius,
                                  angle_start, angle_end,
                                  facecolor=color,
                                  edgecolor='white',
                                  linewidth=0.5)
                    ax.add_patch(wedge)
                    angle_start = angle_end

    plt.title(f"Pie Map: {os.path.basename(output_file).replace('pie_map_', '').replace('.png', '')}", fontsize=20)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)

    generate_categorical_legend(
        categories=categories,
        cmap_name=cmap_name,
        title=os.path.basename(output_file).replace('pie_map_', '').replace('.png', ''),
        output_file=output_file
    )

def generate_pie_maps(som: KohonenSOM, config: dict, working_dir: str, output_dir: str):
    """
    Loads pre-calculated pie data and generates a Pie Map for each categorical column.
    """
    categorical_cols = config.get('categorical_column', [])
    if not categorical_cols:
        return

    print("INFO: Generating Pie Maps...")
    for col in categorical_cols:
        json_path = os.path.join(working_dir, "json", f"pie_data_{col}.json")
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                pie_data = json.load(f)

            output_file = os.path.join(output_dir, f"pie_map_{col}.png")
            generate_pie_map(som, pie_data, output_file)
        else:
            print(f"WARNING: Pie data file not found for column '{col}' at '{json_path}'. Skipping Pie Map.")


def generate_cluster_map(som: KohonenSOM, clusters: dict, output_file: str):
    """Render map of active neurons (clusters)."""
    m, n = som.m, som.n

    # Create map where each active neuron has a unique integer label
    labels = np.full((m, n), -1, dtype=int)
    active_neuron_keys = sorted(clusters.keys())

    for idx, key in enumerate(active_neuron_keys):
        i, j = map(int, key.split('_'))
        if 0 <= i < m and 0 <= j < n:
            labels[i, j] = idx

    num_clusters = len(active_neuron_keys)
    if num_clusters == 0:
        print("WARNING: No active clusters to render in Cluster Map.")
        return

    cmap = plt.get_cmap('tab20', num_clusters)

    _create_map(som, labels, "Cluster Map", output_file, cmap=cmap)


def generate_distance_map(som: KohonenSOM, normalized_data: np.ndarray,
                          mask: np.ndarray, output_file: str, show_title: bool = True):

    neuron_error_map, _ = som.compute_quantization_error(normalized_data, mask=mask)

    if neuron_error_map is not None:
        _create_map(som, neuron_error_map, "Distance Map (Neuron QE)", output_file,
                    cmap='magma', cbar_label="Quantization Error", show_title=show_title)


def generate_dead_neurons_map(som: KohonenSOM, normalized_data: np.ndarray, output_file: str, show_title: bool = True):
    """
    Generates a map showing dead (inactive) neurons.
    Dead neurons are those that have not been assigned any data samples (hit count = 0).
    Uses binary colormap: black for dead neurons, white for active neurons.
    """
    num_neurons = som.m * som.n
    flat_weights = som.weights.reshape(num_neurons, som.dim)

    # Vectorized calculation of BMUs (same as in generate_hit_map)
    dists = np.linalg.norm(normalized_data[:, np.newaxis, :] - flat_weights[np.newaxis, :, :], axis=2)
    bmu_indices = np.argmin(dists, axis=1)

    # Count hits per neuron
    hit_counts = np.bincount(bmu_indices, minlength=num_neurons).reshape(som.m, som.n)

    # Create binary map: 0 = dead (black), 1 = active (white)
    activity_map = (hit_counts > 0).astype(float)

    _create_map(som, activity_map, "Dead Neurons Map", output_file,
                cmap='binary', cbar_label="Neuron activity (0=dead, 1=active)", show_title=show_title)


def generate_individual_maps(som: KohonenSOM, normalized_data: np.ndarray,
                             mask: np.ndarray, output_dir: str):
    """
    Generate individual maps (U-Matrix, Distance Map, Dead Neurons Map) for EA runs.
    Maps are generated WITHOUT titles to be suitable for CNN training.
    """
    maps_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(maps_dir, exist_ok=True)

    # Generate maps without titles for CNN compatibility
    generate_u_matrix(som, os.path.join(maps_dir, "u_matrix.png"), show_title=False)
    generate_distance_map(som, normalized_data, mask, os.path.join(maps_dir, "distance_map.png"), show_title=False)
    generate_dead_neurons_map(som, normalized_data, os.path.join(maps_dir, "dead_neurons_map.png"), show_title=False)


def generate_all_maps(som: KohonenSOM, original_df: pd.DataFrame, normalized_data: np.ndarray, config: dict,
                      mask: np.ndarray, output_dir: str):
    """Main orchestrator for generating all maps."""
    maps_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(maps_dir, exist_ok=True)

    print("INFO: Generating visualizations...")

    generate_individual_maps(som, normalized_data, mask, output_dir)

    # generate_u_matrix(som, os.path.join(maps_dir, "u_matrix.png"))
    generate_hit_map(som, normalized_data, os.path.join(maps_dir, "hit_map.png"))
    generate_component_planes(som, original_df, config, maps_dir)

    clusters_path = os.path.join(output_dir, "clusters.json")
    if os.path.exists(clusters_path):
        with open(clusters_path, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
        generate_cluster_map(som, clusters, os.path.join(maps_dir, "cluster_map.png"))

    generate_pie_maps(som, config, output_dir, maps_dir)

    print("INFO: Map generation completed.")
