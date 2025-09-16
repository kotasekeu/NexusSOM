# visualization.py
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
from matplotlib.patches import Rectangle, RegularPolygon, Wedge, Circle, Patch
from matplotlib.colors import Normalize, ListedColormap
from som import KohonenSOM


# Central drawing function for SOM maps
def _create_map(som: KohonenSOM, values: np.ndarray, title: str, output_file: str,
                cmap: str, cbar_label: str = None, show_text: list = None):
    """
    Universal function for rendering any SOM map (U-Matrix, Hitmap, etc.).
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


def generate_u_matrix(som: KohonenSOM, output_file: str):
    """Vectorized calculation and rendering of U-Matrix."""
    m, n, weights = som.m, som.n, som.weights
    u_matrix = np.zeros((m, n))

    # Vectorized calculation of neighbor distances
    diffs_v = np.linalg.norm(weights[1:, :, :] - weights[:-1, :, :], axis=2)
    u_matrix[1:, :] += diffs_v
    u_matrix[:-1, :] += diffs_v
    diffs_h = np.linalg.norm(weights[:, 1:, :] - weights[:, :-1, :], axis=2)
    u_matrix[:, 1:] += diffs_h
    u_matrix[:, :-1] += diffs_h

    # Normalize by number of neighbors (corners 2, edges 3, center 4)
    counts = np.full((m, n), 4)
    counts[[0, -1], :] -= 1
    counts[:, [0, -1]] -= 1
    u_matrix /= counts

    _create_map(som, u_matrix, "U-Matrix", output_file, cmap='viridis', cbar_label="Average distance to neighbors")


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


def generate_pie_maps(som: KohonenSOM, config: dict, working_dir: str, output_dir: str):
    """Render pie maps for all categorical columns."""
    # Placeholder for future implementation
    print("INFO: Pie map generation (TODO)...")
    pass


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

    cmap = plt.get_cmap('viridis', num_clusters)

    _create_map(som, labels, "Cluster Map", output_file, cmap=cmap)


def generate_all_maps(som: KohonenSOM, original_df: pd.DataFrame, normalized_data: np.ndarray, config: dict,
                      working_dir: str):
    """Main orchestrator for generating all maps."""
    maps_dir = os.path.join(working_dir, "visualizations")
    os.makedirs(maps_dir, exist_ok=True)

    print("INFO: Generating visualizations...")

    generate_u_matrix(som, os.path.join(maps_dir, "u_matrix.png"))
    generate_hit_map(som, normalized_data, os.path.join(maps_dir, "hit_map.png"))
    generate_component_planes(som, original_df, config, maps_dir)

    clusters_path = os.path.join(working_dir, "clusters.json")
    if os.path.exists(clusters_path):
        with open(clusters_path, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
        generate_cluster_map(som, clusters, os.path.join(maps_dir, "cluster_map.png"))

    generate_pie_maps(som, config, working_dir, maps_dir)

    print("INFO: Map generation completed.")
