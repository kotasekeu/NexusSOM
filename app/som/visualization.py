# visualization.py — SOM map rendering.
#
# All functions work from stored artifacts — a weight matrix (m, n, dim) plus
# map_type ('hex' | 'square') — never from a live KohonenSOM instance. This
# lets the UI and ablation tooling re-render any saved run from weights.npy
# without re-training (see render_results_dir).
import json
import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — required for multiprocessing on Windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle, RegularPolygon, Wedge, Circle, Patch


# ─── Shared numpy helpers ─────────────────────────────────────────────────────

def _bmu_indices(weights: np.ndarray, data: np.ndarray,
                 mask: np.ndarray = None) -> np.ndarray:
    """Flat BMU index for every sample (vectorized, mask-aware)."""
    flat_weights = weights.reshape(-1, weights.shape[2])
    diffs = data[:, np.newaxis, :] - flat_weights[np.newaxis, :, :]
    if mask is not None:
        diffs *= (~mask)[:, np.newaxis, :]
    dists = np.linalg.norm(diffs, axis=2)
    return np.argmin(dists, axis=1)


def _neuron_qe_map(weights: np.ndarray, data: np.ndarray,
                   mask: np.ndarray = None) -> np.ndarray:
    """Per-neuron mean quantization error as an (m, n) matrix."""
    m, n, dim = weights.shape
    flat_weights = weights.reshape(-1, dim)
    bmu_idx = _bmu_indices(weights, data)

    diffs = data - flat_weights[bmu_idx]
    if mask is not None:
        diffs = diffs * (~mask)
    errors = np.linalg.norm(diffs, axis=1)

    sum_errors = np.bincount(bmu_idx, weights=errors, minlength=m * n)
    counts = np.bincount(bmu_idx, minlength=m * n)
    neuron_errors = np.divide(sum_errors, counts,
                              out=np.zeros_like(sum_errors), where=counts != 0)
    return neuron_errors.reshape(m, n)


def compute_u_matrix(weights: np.ndarray, map_type: str) -> np.ndarray:
    """
    Vectorized U-Matrix: per-neuron mean distance to its grid neighbors.
    Square grid uses 4 neighbors; hex grid uses 6 with row-parity offsets.
    """
    m, n, _ = weights.shape
    if map_type == 'hex':
        # Odd-r convention (odd rows shifted right) — must match the cube
        # coordinates used by KohonenSOM for training/TE: even rows reach
        # diagonals at j-1/j, odd rows at j/j+1. The legacy implementation had
        # the two parities swapped (docs/som/issues.md #25).
        offsets_even = {(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)}
        offsets_odd = {(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)}
    else:
        offsets_even = offsets_odd = {(-1, 0), (1, 0), (0, -1), (0, 1)}

    row_is_even = (np.arange(m)[:, None] % 2 == 0) * np.ones((1, n), dtype=bool)

    dist_sum = np.zeros((m, n))
    counts = np.zeros((m, n))
    for di, dj in offsets_even | offsets_odd:
        # Distance to the (di, dj)-shifted neighbor over the valid overlap region
        i0, i1 = max(0, -di), m - max(0, di)
        j0, j1 = max(0, -dj), n - max(0, dj)
        if i0 >= i1 or j0 >= j1:
            continue
        dist = np.full((m, n), np.nan)
        dist[i0:i1, j0:j1] = np.linalg.norm(
            weights[i0 + di:i1 + di, j0 + dj:j1 + dj] - weights[i0:i1, j0:j1], axis=2)

        applicable = np.zeros((m, n), dtype=bool)
        if (di, dj) in offsets_even:
            applicable |= row_is_even
        if (di, dj) in offsets_odd:
            applicable |= ~row_is_even
        valid = applicable & ~np.isnan(dist)
        dist_sum[valid] += dist[valid]
        counts[valid] += 1

    return np.divide(dist_sum, counts, out=np.zeros((m, n)), where=counts > 0)


# ─── Central drawing function ─────────────────────────────────────────────────

def _grid_patches(m: int, n: int, map_type: str) -> list:
    """Matplotlib patches for the neuron grid, ordered row-major (i*n + j)."""
    if map_type == 'hex':
        side_len = 0.5
        radius = side_len / np.cos(np.pi / 6)
        dy = 2 * radius * (3 / 4)
        return [RegularPolygon((j * (2 * side_len) + (i % 2) * side_len, i * dy),
                               numVertices=6, radius=radius)
                for i in range(m) for j in range(n)]
    side_len = 1.0
    return [Rectangle((j - side_len / 2, i - side_len / 2), side_len, side_len)
            for i in range(m) for j in range(n)]


def _create_map(values: np.ndarray, map_type: str, title: str, output_file: str,
                cmap, cbar_label: str = None, show_text: list = None,
                show_title: bool = True, fixed_norm: bool = False, vmax: float = None):
    """
    Universal function for rendering any SOM map (U-Matrix, Hitmap, etc.).

    Args:
        values: (m, n) matrix of per-neuron values to color.
        map_type: 'hex' or 'square'.
        show_title: If True, display title above the map.
        fixed_norm: If True, normalize to [0, vmax] for consistent visualization.
        vmax: Maximum value for fixed normalization. If None, uses values.max().
    """
    m, n = values.shape

    fig, ax = plt.subplots(figsize=(n * 1.2, m * 1.2))
    ax.set_aspect('equal')
    ax.axis('off')

    patches = _grid_patches(m, n, map_type)

    collection = plt.matplotlib.collections.PatchCollection(patches)
    collection.set_array(values.flatten())
    collection.set_cmap(cmap)

    # Apply fixed normalization if requested (black=0, white=vmax)
    if fixed_norm:
        max_val = vmax if vmax is not None else values.max()
        collection.set_clim(vmin=0, vmax=max_val)

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

        if fixed_norm:
            max_val = vmax if vmax is not None else values.max()
            norm = Normalize(vmin=0, vmax=max_val)
        else:
            norm = Normalize(vmin=values.min(), vmax=values.max())

        plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax_legend,
                     orientation='vertical', label=cbar_label)
        legend_path = os.path.join(os.path.dirname(output_file), "legends",
                                   os.path.basename(output_file))
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

    legend_path = os.path.join(os.path.dirname(output_file), "legends",
                               os.path.basename(output_file))
    os.makedirs(os.path.dirname(legend_path), exist_ok=True)
    fig.savefig(legend_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─── Individual map generators ────────────────────────────────────────────────

def generate_u_matrix(weights: np.ndarray, map_type: str, output_file: str,
                      show_title: bool = True):
    """Calculates and visualizes the U-Matrix, supporting both square and hex grids."""
    u_matrix = compute_u_matrix(weights, map_type)
    _create_map(u_matrix, map_type, "U-Matrix", output_file, cmap='viridis',
                cbar_label="Average distance to neighbors", show_title=show_title)


def generate_hit_map(weights: np.ndarray, map_type: str, normalized_data: np.ndarray,
                     output_file: str, mask: np.ndarray = None):
    """Vectorized calculation and rendering of Hit Map."""
    m, n, _ = weights.shape
    hit_counts = np.bincount(_bmu_indices(weights, normalized_data, mask),
                             minlength=m * n).reshape(m, n)

    _create_map(hit_counts.astype(float), map_type, "Hit Map", output_file, cmap='Blues',
                cbar_label="Number of assigned samples", show_text=hit_counts.flatten())


def generate_component_planes(weights: np.ndarray, map_type: str,
                              original_df: pd.DataFrame, config: dict, output_dir: str):
    """
    Render component planes for ALL columns used in training.
    """
    m, n, dim = weights.shape
    training_columns = list(original_df.columns)
    primary_id_col = config.get('primary_id', 'primary_id')

    try:
        pid_index = training_columns.index(primary_id_col)
    except ValueError:
        pid_index = -1

    # Ensure correct number of columns
    if len(training_columns) != dim:
        print(
            f"WARNING: Number of training columns ({len(training_columns)}) does not match SOM dimension ({dim}). Component planes may have incorrect labels.")
        training_columns = [f"dim_{i}" for i in range(dim)]

    for i, col_name in enumerate(training_columns):
        if i == pid_index:
            continue
        plane_values = weights[:, :, i]

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
        _create_map(de_normalized_values, map_type, f"Component Plane: {col_name}",
                    output_file, cmap='coolwarm', cbar_label=cbar_label_text)


def generate_pie_map(weights: np.ndarray, map_type: str, pie_data: dict,
                     output_file: str, cmap_name: str = 'tab20b'):
    """
    Visualizes categorical data distribution on the SOM grid using pie charts.
    """
    m, n, _ = weights.shape

    fig, ax = plt.subplots(figsize=(n * 1.2, m * 1.2))
    ax.set_aspect('equal')
    ax.axis('off')

    patches = _grid_patches(m, n, map_type)
    bg_collection = plt.matplotlib.collections.PatchCollection(
        patches, facecolor='#f0f0f0', edgecolor='white')
    ax.add_collection(bg_collection)
    ax.autoscale_view()

    categories = pie_data['categories']
    cat_keys = sorted(categories.keys(), key=int)
    num_categories = len(cat_keys)
    cmap = plt.get_cmap(cmap_name)

    coords = np.array([p.get_center() if map_type == 'square' else p.xy
                       for p in patches]).reshape(m, n, 2)
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


def generate_pie_maps(weights: np.ndarray, map_type: str, config: dict,
                      working_dir: str, output_dir: str):
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
            generate_pie_map(weights, map_type, pie_data, output_file)
        else:
            print(f"WARNING: Pie data file not found for column '{col}' at '{json_path}'. Skipping Pie Map.")


def generate_cluster_map(weights: np.ndarray, map_type: str, clusters: dict,
                         output_file: str):
    """Render map of active neurons (clusters)."""
    m, n, _ = weights.shape

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

    _create_map(labels.astype(float), map_type, "Cluster Map", output_file, cmap=cmap)


def generate_distance_map(weights: np.ndarray, map_type: str, normalized_data: np.ndarray,
                          mask: np.ndarray, output_file: str, show_title: bool = True):
    """Per-neuron quantization error map."""
    neuron_error_map = _neuron_qe_map(weights, normalized_data, mask)
    _create_map(neuron_error_map, map_type, "Distance Map (Neuron QE)", output_file,
                cmap='magma', cbar_label="Quantization Error", show_title=show_title)


def generate_dead_neurons_map(weights: np.ndarray, map_type: str,
                              normalized_data: np.ndarray, output_file: str,
                              show_title: bool = True, mask: np.ndarray = None):
    """
    Generates a map showing dead (inactive) neurons.
    Dead neurons are those that have not been assigned any data samples (hit count = 0).
    Uses binary colormap: black for dead neurons, white for active neurons.
    """
    m, n, _ = weights.shape
    hit_counts = np.bincount(_bmu_indices(weights, normalized_data, mask),
                             minlength=m * n).reshape(m, n)
    activity_map = (hit_counts > 0).astype(float)

    _create_map(activity_map, map_type, "Dead Neurons Map", output_file,
                cmap='binary', cbar_label="Neuron activity (0=dead, 1=active)",
                show_title=show_title)


# ─── Orchestrators ────────────────────────────────────────────────────────────

def generate_individual_maps(weights: np.ndarray, map_type: str,
                             normalized_data: np.ndarray, mask: np.ndarray,
                             output_dir: str):
    """
    Generate individual maps (U-Matrix, Distance Map, Dead Neurons Map) for EA runs.
    Maps are generated WITHOUT titles (legacy: was required for CNN training data).
    """
    maps_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(maps_dir, exist_ok=True)

    generate_u_matrix(weights, map_type, os.path.join(maps_dir, "u_matrix.png"),
                      show_title=False)
    generate_distance_map(weights, map_type, normalized_data, mask,
                          os.path.join(maps_dir, "distance_map.png"), show_title=False)
    generate_dead_neurons_map(weights, map_type, normalized_data,
                              os.path.join(maps_dir, "dead_neurons_map.png"),
                              show_title=False, mask=mask)


def generate_all_maps(weights: np.ndarray, map_type: str, original_df: pd.DataFrame,
                      normalized_data: np.ndarray, config: dict, mask: np.ndarray,
                      output_dir: str):
    """Main orchestrator for generating all maps."""
    maps_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(maps_dir, exist_ok=True)

    print("INFO: Generating visualizations...")

    generate_individual_maps(weights, map_type, normalized_data, mask, output_dir)

    generate_hit_map(weights, map_type, normalized_data,
                     os.path.join(maps_dir, "hit_map.png"), mask=mask)
    generate_component_planes(weights, map_type, original_df, config, maps_dir)

    clusters_path = os.path.join(output_dir, "json", "clusters.json")
    if os.path.exists(clusters_path):
        with open(clusters_path, 'r', encoding='utf-8') as f:
            clusters = json.load(f)
        generate_cluster_map(weights, map_type, clusters,
                             os.path.join(maps_dir, "cluster_map.png"))

    generate_pie_maps(weights, map_type, config, output_dir, maps_dir)

    print("INFO: Map generation completed.")


def render_results_dir(results_dir: str) -> str:
    """
    Re-render all maps for a stored run purely from its saved artifacts —
    no live SOM object, no re-training. Entry point for the UI and for
    ablation-study comparison tooling.

    Reads: csv/weights.npy (required), run_metrics.json (map topology),
    csv/training_data.npy, csv/ignore_mask.csv, csv/original_input.csv,
    json/preprocessing_info.json. Returns the visualizations directory path.
    """
    csv_dir = os.path.join(results_dir, 'csv')
    json_dir = os.path.join(results_dir, 'json')

    weights_path = os.path.join(csv_dir, 'weights.npy')
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"weights.npy not found in '{csv_dir}'")
    weights = np.load(weights_path)

    run_metrics = {}
    rm_path = os.path.join(results_dir, 'run_metrics.json')
    if os.path.isfile(rm_path):
        with open(rm_path, encoding='utf-8') as f:
            run_metrics = json.load(f)
    map_type = run_metrics.get('map_topology', 'hex')

    normalized_data = np.load(os.path.join(csv_dir, 'training_data.npy'))
    original_df = pd.read_csv(os.path.join(csv_dir, 'original_input.csv'))

    mask = None
    mask_path = os.path.join(csv_dir, 'ignore_mask.csv')
    if os.path.isfile(mask_path):
        mask = pd.read_csv(mask_path, header=None).values.astype(bool)

    # Reconstruct column classification from preprocessing_info
    preprocessing_info = {}
    pi_path = os.path.join(json_dir, 'preprocessing_info.json')
    if os.path.isfile(pi_path):
        with open(pi_path, encoding='utf-8') as f:
            preprocessing_info = json.load(f)

    primary_id = None
    numerical, categorical = [], []
    for col, info in preprocessing_info.items():
        if info.get('nunique_ratio', 0.0) >= 0.99 and primary_id is None:
            primary_id = col
            continue
        if info.get('status') == 'ignored':
            continue
        if info.get('is_categorical', False):
            categorical.append(col)
        elif info.get('base_type') == 'numeric':
            numerical.append(col)

    config = {
        'primary_id': primary_id or 'primary_id',
        'numerical_column': numerical,
        'categorical_column': categorical,
    }

    generate_all_maps(weights, map_type, original_df, normalized_data, config,
                      mask, results_dir)
    return os.path.join(results_dir, 'visualizations')
