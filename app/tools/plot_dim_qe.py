"""
plot_dim_qe.py — Per-dimension Quantization Error heatmaps on the SOM map.

Variant A: one heatmap per dimension (all dims in a subplot grid, sorted by mean QE)
Variant B: dominant-dimension map (which dim has highest QE per neuron,
           intensity proportional to magnitude)

Usage:
  python app/tools/plot_dim_qe.py <results_dir>
  python app/tools/plot_dim_qe.py <results_dir> --no-a
  python app/tools/plot_dim_qe.py <results_dir> --no-b
  python app/tools/plot_dim_qe.py <results_dir> --output-dir /path/to/output
"""

import argparse
import json
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.collections
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle, RegularPolygon, Patch
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load(results_dir: str):
    csv_dir = os.path.join(results_dir, 'csv')

    def _j(path):
        if not os.path.isfile(path):
            return {}
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    weights_path = os.path.join(csv_dir, 'weights.npy')
    if not os.path.isfile(weights_path):
        sys.exit(f"ERROR: weights.npy not found in {csv_dir}")
    weights = np.load(weights_path)

    assign_path = os.path.join(csv_dir, 'sample_assignments.csv')
    if not os.path.isfile(assign_path):
        sys.exit(f"ERROR: sample_assignments.csv not found in {csv_dir}")
    assignments = pd.read_csv(assign_path)

    run_metrics = _j(os.path.join(results_dir, 'run_metrics.json'))
    hex_topology = run_metrics.get('map_topology', 'rect') == 'hex'

    # Dimension names from qe_dim_* columns (primary ID already excluded there)
    dim_cols = [c for c in assignments.columns if c.startswith('qe_dim_')]
    dim_names = [c[len('qe_dim_'):] for c in dim_cols]

    if not dim_cols:
        sys.exit("ERROR: no qe_dim_* columns in sample_assignments.csv. "
                 "Re-run SOM analysis to generate per-dim QE.")

    return weights, assignments, dim_cols, dim_names, hex_topology


# ─── Aggregation ──────────────────────────────────────────────────────────────

def _compute_neuron_dim_qe(assignments: pd.DataFrame, m: int, n: int,
                            dim_cols: list) -> np.ndarray:
    """
    Aggregate per-sample qe_dim_* values by BMU neuron.
    Returns (m, n, n_dims) array of mean QE; NaN for neurons with no samples.
    """
    n_dims = len(dim_cols)
    flat = np.full((m * n, n_dims), np.nan)

    df = assignments.copy()
    df['_nidx'] = df['bmu_i'].astype(int) * n + df['bmu_j'].astype(int)

    for col_idx, col in enumerate(dim_cols):
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors='coerce')
        grouped = numeric.groupby(df['_nidx']).mean()
        valid_idx = grouped.index[(grouped.index >= 0) & (grouped.index < m * n)]
        flat[valid_idx.values, col_idx] = grouped[valid_idx].values

    return flat.reshape(m, n, n_dims)


# ─── Rendering helpers ────────────────────────────────────────────────────────

def _make_patches(m: int, n: int, hex_topology: bool):
    patches = []
    if hex_topology:
        side_len = 0.5
        radius = side_len / np.cos(np.pi / 6)
        dy = 2 * radius * (3 / 4)
        for i in range(m):
            for j in range(n):
                x = j * (2 * side_len) + (i % 2) * side_len
                y = i * dy
                patches.append(RegularPolygon((x, y), numVertices=6, radius=radius))
    else:
        side_len = 1.0
        for i in range(m):
            for j in range(n):
                patches.append(Rectangle((j - 0.5, i - 0.5), side_len, side_len))
    return patches


def _draw_single(ax, values_flat: np.ndarray, m: int, n: int, hex_topology: bool,
                 cmap: str, vmin: float, vmax: float):
    """Draw one SOM heatmap onto ax. NaN neurons shown in white on #ececec background."""
    ax.set_facecolor('#ececec')
    patches = _make_patches(m, n, hex_topology)
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap(cmap)(norm(np.nan_to_num(values_flat, nan=vmin)))
    # White for neurons with no assigned samples — stands out against #ececec background
    rgba[np.isnan(values_flat)] = (1.0, 1.0, 1.0, 1.0)

    col = matplotlib.collections.PatchCollection(patches, match_original=False)
    col.set_facecolor(rgba)
    col.set_edgecolor('#ececec')
    col.set_linewidth(0.4)
    ax.add_collection(col)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.axis('off')


# ─── Variant A — one file per dimension ──────────────────────────────────────

def _sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in '-_' else '_' for c in name)


def plot_all_dims(qe_tensor: np.ndarray, dim_names: list, m: int, n: int,
                  hex_topology: bool, output_dir: str):
    n_dims = len(dim_names)
    global_means = np.nanmean(qe_tensor.reshape(-1, n_dims), axis=0)
    order = np.argsort(global_means)[::-1]  # highest mean QE first

    fig_w = max(5.0, n * 0.7)
    fig_h = max(4.0, m * (0.65 if hex_topology else 0.6)) + 1.0

    saved = []
    for rank, dim_idx in enumerate(order, start=1):
        dim_name = dim_names[dim_idx]
        values_flat = qe_tensor[:, :, dim_idx].flatten()
        valid = values_flat[~np.isnan(values_flat)]
        vmax = float(valid.max()) if len(valid) > 0 else 1.0

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor('white')

        _draw_single(ax, values_flat, m, n, hex_topology,
                     cmap='YlOrRd', vmin=0.0, vmax=vmax)

        sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=Normalize(vmin=0.0, vmax=vmax))
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label('Mean QE (normalized)', fontsize=9)

        ax.set_title(
            f"{dim_name}  —  mean QE: {global_means[dim_idx]:.4f}  (rank #{rank}/{n_dims})\n"
            f"yellow = low error · red = high error · white = no samples",
            fontsize=10,
        )

        fname = f"dim_qe_{rank:02d}_{_sanitize(dim_name)}.png"
        out = os.path.join(output_dir, fname)
        plt.tight_layout()
        plt.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        saved.append(out)

    print(f"Saved {len(saved)} dimension maps to {output_dir}")
    return saved


# ─── Variant B — dominant-dimension map ───────────────────────────────────────

def plot_dominant_dim(qe_tensor: np.ndarray, dim_names: list, m: int, n: int,
                      hex_topology: bool, output_path: str):
    n_dims = len(dim_names)
    # hsv gives vivid full-spectrum colors evenly distributed across dimensions
    dim_colors = [matplotlib.colormaps['Dark2'](i / n_dims) for i in range(n_dims)]

    dominant = np.full(m * n, -1, dtype=int)
    intensity = np.zeros(m * n)

    flat = qe_tensor.reshape(-1, n_dims)
    for k in range(m * n):
        row = flat[k]
        if np.all(np.isnan(row)):
            continue
        dominant[k] = int(np.nanargmax(row))
        intensity[k] = float(np.nanmax(row))

    max_i = intensity.max()
    norm_i = intensity / max_i if max_i > 0 else intensity

    patches = _make_patches(m, n, hex_topology)
    facecolors = []
    for k in range(m * n):
        if dominant[k] == -1:
            facecolors.append((0.88, 0.88, 0.88, 1.0))  # gray = no samples
        else:
            r, g, b, _ = dim_colors[dominant[k]]
            alpha = 0.30 + 0.70 * norm_i[k]
            facecolors.append((r, g, b, alpha))

    fig_w = max(9, n * 1.2)
    fig_h = max(6, m * (1.1 if hex_topology else 1.0))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor('#ececec')
    ax.set_facecolor('#ececec')

    col = matplotlib.collections.PatchCollection(patches, match_original=False)
    col.set_facecolor(facecolors)
    col.set_edgecolor('#ececec')
    col.set_linewidth(0.4)
    ax.add_collection(col)
    ax.autoscale_view()
    ax.set_aspect('equal')
    ax.axis('off')

    # Legend: one entry per dimension + gray for empty neurons
    legend_handles = [
        Patch(facecolor=dim_colors[d], label=f"{d+1}. {dim_names[d]}")
        for d in range(n_dims)
    ]
    legend_handles.append(Patch(facecolor=(0.88, 0.88, 0.88), label='No samples'))

    ax.legend(handles=legend_handles,
              loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=8, title='Dominant dimension', title_fontsize=8,
              framealpha=0.9)

    ax.set_title(
        "Dominant QE dimension per neuron\n"
        "color = dimension with highest mean QE · intensity ∝ magnitude",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-dimension QE heatmaps for a SOM results directory.")
    parser.add_argument("results_dir", help="Path to SOM results directory")
    parser.add_argument("--no-a", action="store_true",
                        help="Skip Variant A (all-dims grid)")
    parser.add_argument("--no-b", action="store_true",
                        help="Skip Variant B (dominant-dim map)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: results_dir/maps_dataset)")
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        sys.exit(f"ERROR: directory not found: {args.results_dir}")

    weights, assignments, dim_cols, dim_names, hex_topology = _load(args.results_dir)
    m, n, _ = weights.shape

    print(f"Map: {m}×{n} {'hex' if hex_topology else 'square'} | "
          f"Dimensions ({len(dim_names)}): {dim_names}")

    qe_tensor = _compute_neuron_dim_qe(assignments, m, n, dim_cols)

    out_dir = args.output_dir or os.path.join(args.results_dir, 'maps_dataset')
    os.makedirs(out_dir, exist_ok=True)

    if not args.no_a:
        plot_all_dims(qe_tensor, dim_names, m, n, hex_topology, out_dir)

    if not args.no_b:
        plot_dominant_dim(qe_tensor, dim_names, m, n, hex_topology,
                          os.path.join(out_dir, 'dim_qe_dominant.png'))


if __name__ == '__main__':
    main()
