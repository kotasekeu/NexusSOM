"""
plot_som_topology.py — SOM weight vectors + topological grid in projected space.

Projection algorithms:
  --projection pca    Linear, fast. Fails on non-linear manifolds (Swiss Roll).
  --projection umap   Non-linear, unfolds the SOM manifold. Best for large maps.
  --projection tsne   Non-linear, fits data+weights jointly. Slower than UMAP.
  --projection isomap Geodesic distance preserving. Best for manifold benchmarks.

Data layers:
  default         Scatter of training samples (faint background)
  --density       Hexbin density map instead of scatter
  --grid-only     No training samples — pure grid geometry

Output formats:
  default         topology_2d_{method}.png (static)
  --3d            topology_3d_{method}.png (static, only for pca/umap/isomap)
  --html          topology_interactive_{method}.html (Plotly, zoomable, hover info)
  --compare       2×2 grid: (data | weights) × (PCA | ISOMAP) — ablation / Swiss Roll

Usage:
  python app/tools/plot_som_topology.py <results_dir>
  python app/tools/plot_som_topology.py <results_dir> --projection isomap
  python app/tools/plot_som_topology.py <results_dir> --compare
  python app/tools/plot_som_topology.py <results_dir> --projection umap --3d --html
"""

import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load(results_dir: str):
    def _j(path):
        if not os.path.isfile(path):
            return None
        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def _npy(path):
        if not os.path.isfile(path):
            return None
        return np.load(path)

    def _csv(path):
        if not os.path.isfile(path):
            return None
        return pd.read_csv(path)

    csv_dir = os.path.join(results_dir, 'csv')

    run_metrics = _j(os.path.join(results_dir, 'run_metrics.json')) or {}
    meta        = _j(os.path.join(results_dir, 'dataset_meta.json')) or {}
    if 'hex_topology' not in meta:
        meta['hex_topology'] = run_metrics.get('map_topology', 'rect') == 'hex'

    # Load ignore mask and store always-masked column indices in meta.
    # Always-masked columns (e.g. primary ID) were never updated during SOM training;
    # including them in projection distorts PCA/UMAP results.
    mask_path = os.path.join(csv_dir, 'ignore_mask.csv')
    if os.path.isfile(mask_path):
        mask_np = pd.read_csv(mask_path, header=None).values.astype(bool)
        meta['_always_masked_cols'] = np.where(mask_np.all(axis=0))[0].tolist()
        meta['_ignore_mask'] = mask_np
    else:
        meta['_always_masked_cols'] = []
        meta['_ignore_mask'] = None

    return (
        _npy(os.path.join(csv_dir, 'weights.npy')),
        _npy(os.path.join(csv_dir, 'training_data.npy')),
        _csv(os.path.join(csv_dir, 'sample_assignments.csv')),
        meta,
    )


# ─── Mask application ─────────────────────────────────────────────────────────

def _strip_masked(training_data: np.ndarray, weights_flat: np.ndarray,
                  meta: dict) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """
    Remove always-masked columns and zero out per-sample NaN positions.

    Always-masked columns (e.g. primary ID) were never updated during SOM
    training. Including them in PCA/UMAP distorts the projection because:
      - training_data ID column: monotonic [0..1] — structured uninformative variance
      - weights ID column: stuck at random init values — random noise
    Stripping them makes the projected geometry match the actual SOM training space.

    Returns cleaned (training_data, weights_flat, kept_col_indices).
    """
    n_dims = weights_flat.shape[1]
    always_masked = meta.get('_always_masked_cols', [])
    keep = [d for d in range(n_dims) if d not in always_masked]

    wf_clean = weights_flat[:, keep].copy()

    if training_data is None:
        if always_masked:
            print(f'Projection: stripped always-masked dims {always_masked} '
                  f'({n_dims}D → {len(keep)}D)  [weights-only mode]')
        return None, wf_clean, keep

    td_clean = training_data[:, keep].copy()

    # Zero out per-sample NaN positions so they don't skew the projection
    mask = meta.get('_ignore_mask')
    if mask is not None:
        mask_keep = mask[:, keep]          # (N, kept_dims)
        td_clean[mask_keep] = 0.0

    if always_masked:
        print(f'Projection: stripped always-masked dims {always_masked} '
              f'({n_dims}D → {len(keep)}D)')

    return td_clean, wf_clean, keep


# ─── Projection ───────────────────────────────────────────────────────────────

def _project(training_data: np.ndarray, weights_flat: np.ndarray,
             method: str, n_components: int = 2):
    """
    Project training_data and weights_flat into n_components dimensions.
    Returns (data_proj, weights_proj, explained_variance_ratio_or_None).
    If training_data is None, projects weights only (data_proj = None).
    """
    weights_only = training_data is None
    fit_data = weights_flat if weights_only else training_data

    if method == 'pca':
        pca = PCA(n_components=min(n_components, weights_flat.shape[1]))
        pca.fit(fit_data)
        data_proj = None if weights_only else pca.transform(training_data)
        return data_proj, pca.transform(weights_flat), pca.explained_variance_ratio_

    if method == 'umap':
        try:
            from umap import UMAP
        except ImportError:
            sys.exit('ERROR: umap-learn not installed. Run: pip install umap-learn')
        reducer = UMAP(n_components=n_components, random_state=42, verbose=False)
        if weights_only:
            return None, reducer.fit_transform(weights_flat), None
        data_proj = reducer.fit_transform(training_data)
        return data_proj, reducer.transform(weights_flat), None

    if method == 'tsne':
        from sklearn.manifold import TSNE
        # t-SNE has no transform() — fit jointly, then split
        if weights_only:
            proj = TSNE(n_components=n_components, random_state=42,
                        perplexity=min(30, len(weights_flat) - 1),
                        max_iter=1000, verbose=0).fit_transform(weights_flat)
            return None, proj, None
        n_data = len(training_data)
        combined = np.vstack([training_data, weights_flat])
        perplexity = min(30, len(combined) - 1)
        tsne = TSNE(n_components=n_components, random_state=42,
                    perplexity=perplexity, max_iter=1000, verbose=0)
        proj = tsne.fit_transform(combined)
        return proj[:n_data], proj[n_data:], None

    if method == 'isomap':
        import warnings
        from sklearn.manifold import Isomap
        if weights_only:
            n_neighbors = min(15, len(weights_flat) - 1)
            iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                return None, iso.fit_transform(weights_flat), None
        # Joint fit on data + weights so the weight vectors become part of the
        # neighbour graph. Using .transform() after fitting only on data causes
        # boundary neurons to be mapped via Nyström approximation to wrong
        # locations, producing long crossing lines in the weight grid.
        n_neighbors = min(15, len(training_data) - 1)
        iso = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        n_data = len(training_data)
        combined = np.vstack([training_data, weights_flat])
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            combined_proj = iso.fit_transform(combined)
        return combined_proj[:n_data], combined_proj[n_data:], None

    sys.exit(f'ERROR: unknown projection method "{method}". Use pca, umap, tsne, or isomap.')


# ─── Grid connectivity ────────────────────────────────────────────────────────

def _grid_edges(m: int, n: int, hex_topology: bool = False):
    if not hex_topology:
        edges = []
        for i in range(m):
            for j in range(n):
                if j + 1 < n:
                    edges.append(((i, j), (i, j + 1)))
                if i + 1 < m:
                    edges.append(((i, j), (i + 1, j)))
        return edges

    # Hex: cube coordinates — same convention as KohonenSOM.__init__:
    #   x = j - i//2,  z = i,  y = -x - z
    # Two neurons are true hex neighbors iff cube distance == 1.
    # Enumerate all 6 cube directions; convert back to offset to check bounds.
    HEX_DIRS = ((1, -1, 0), (-1, 1, 0), (1, 0, -1), (-1, 0, 1), (0, -1, 1), (0, 1, -1))
    seen: set = set()
    edges = []
    for i in range(m):
        for j in range(n):
            cx = j - i // 2
            cz = i
            for dx, _, dz in HEX_DIRS:
                nx, nz = cx + dx, cz + dz
                ni = nz
                nj = nx + ni // 2
                if 0 <= ni < m and 0 <= nj < n:
                    key = (min(i * n + j, ni * n + nj), max(i * n + j, ni * n + nj))
                    if key not in seen:
                        seen.add(key)
                        edges.append(((i, j), (ni, nj)))
    return edges


# ─── Per-neuron mean QE ───────────────────────────────────────────────────────

def _neuron_qe(assignments: pd.DataFrame, m: int, n: int) -> np.ndarray | None:
    if assignments is None:
        return None
    if not {'bmu_i', 'bmu_j', 'qe'}.issubset(assignments.columns):
        return None
    grid = np.full((m, n), np.nan)
    for (bi, bj), grp in assignments.groupby(['bmu_i', 'bmu_j']):
        if 0 <= int(bi) < m and 0 <= int(bj) < n:
            grid[int(bi), int(bj)] = grp['qe'].mean()
    grid = np.where(np.isnan(grid), np.nanmean(grid), grid)
    return grid


def _edge_colors(w_proj: np.ndarray, edges: list, threshold_pct: float = 85):
    """
    Return per-edge color and a stretched-edge mask.

    An edge is "stretched" if its projected length exceeds `threshold_pct`-th percentile.
    Stretched edges are drawn in orange — they are either true topological crossings
    or projection artefacts (UMAP/t-SNE pulling distant clusters apart).
    Normal edges are drawn in the standard dark slate colour.
    """
    lengths = np.array([
        np.linalg.norm(w_proj[i1, j1] - w_proj[i2, j2])
        for (i1, j1), (i2, j2) in edges
    ])
    threshold = np.percentile(lengths, threshold_pct)
    stretched = lengths > threshold
    colors = np.where(stretched, '#e67e22', '#2c3e50')
    return colors, stretched, threshold


def _auto_scale(n_neurons: int) -> tuple[float, float]:
    """Return (marker_size, line_width) scaled to neuron count."""
    if n_neurons <= 100:
        return 55.0, 0.9
    if n_neurons <= 225:
        return 30.0, 0.6
    if n_neurons <= 400:
        return 18.0, 0.45
    return 10.0, 0.3


# ─── 2D static plot ───────────────────────────────────────────────────────────

def plot_topology_2d(weights: np.ndarray, training_data: np.ndarray,
                     assignments: pd.DataFrame, hex_topology: bool,
                     results_dir: str, output_path: str = None,
                     grid_only: bool = False, density: bool = False,
                     method: str = 'pca', meta: dict = None):
    m, n, dim = weights.shape
    flat_w = weights.reshape(m * n, dim)

    td_proj, wf_proj, _ = _strip_masked(training_data, flat_w, meta or {})
    data_2d, w_2d_flat, var_exp = _project(td_proj, wf_proj, method, 2)
    w_2d = w_2d_flat.reshape(m, n, 2)

    n_neurons  = m * n
    dot_size, lw = _auto_scale(n_neurons)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor('#f8f8f8')

    # Layer 1: data (skip if not available)
    if not grid_only and data_2d is not None:
        if density:
            ax.hexbin(data_2d[:, 0], data_2d[:, 1],
                      gridsize=40, cmap='Blues', mincnt=1,
                      alpha=0.6, zorder=1, linewidths=0.2)
        else:
            ax.scatter(data_2d[:, 0], data_2d[:, 1],
                       color='#2980b9', alpha=0.18, s=4, zorder=1, linewidths=0)

    # Layer 3: grid lines — colour by projected length
    edges = _grid_edges(m, n, hex_topology)
    edge_colors, stretched, _ = _edge_colors(w_2d, edges)
    for (e_idx, ((i1, j1), (i2, j2))) in enumerate(edges):
        ax.plot(
            [w_2d[i1, j1, 0], w_2d[i2, j2, 0]],
            [w_2d[i1, j1, 1], w_2d[i2, j2, 1]],
            color=edge_colors[e_idx], lw=lw,
            alpha=0.55 if stretched[e_idx] else 0.75,
            zorder=3,
        )

    # Layer 2: neurons coloured by mean QE
    nqe      = _neuron_qe(assignments, m, n)
    nqe_flat = nqe.ravel() if nqe is not None else None

    if nqe_flat is not None:
        sc = ax.scatter(
            w_2d[:, :, 0].ravel(), w_2d[:, :, 1].ravel(),
            c=nqe_flat, cmap='plasma', s=dot_size, zorder=5,
            edgecolors='white', linewidths=0.4,
            vmin=np.nanpercentile(nqe_flat, 5),
            vmax=np.nanpercentile(nqe_flat, 95),
        )
        plt.colorbar(sc, ax=ax, label='Mean QE per neuron', shrink=0.75)
    else:
        ax.scatter(w_2d[:, :, 0].ravel(), w_2d[:, :, 1].ravel(),
                   c='crimson', s=dot_size, zorder=5,
                   edgecolors='white', linewidths=0.4)

    # Zoom to neuron bounding box — prevents outlier data from shrinking the grid
    pad = (w_2d[:, :, 0].max() - w_2d[:, :, 0].min()) * 0.08 + 0.05
    ax.set_xlim(w_2d[:, :, 0].min() - pad, w_2d[:, :, 0].max() + pad)
    pad = (w_2d[:, :, 1].max() - w_2d[:, :, 1].min()) * 0.08 + 0.05
    ax.set_ylim(w_2d[:, :, 1].min() - pad, w_2d[:, :, 1].max() + pad)

    var_str = (f'PC1 {var_exp[0]:.1%} / PC2 {var_exp[1]:.1%}'
               if var_exp is not None else method.upper())
    mode = 'grid only' if grid_only else ('density' if density else 'scatter')
    ax.set_xlabel(f'Dim 1  ({var_str})', fontsize=10)
    ax.set_ylabel('Dim 2', fontsize=10)
    ax.set_title(
        f'SOM Topology [{method.upper()}] — {os.path.basename(os.path.normpath(results_dir))}\n'
        f'{m}×{n} grid  |  {"hex" if hex_topology else "rect"}  |  {mode}'
        f'  —  crossing lines = topological errors',
        fontsize=11,
    )

    n_stretched = int(stretched.sum())
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson',
               markersize=8, label='Neuron (mean QE colour)'),
        Line2D([0], [0], color='#2c3e50', lw=1.2, label='Grid edge (normal)'),
        Line2D([0], [0], color='#e67e22', lw=1.2,
               label=f'Grid edge (stretched top 15%, n={n_stretched}) — topology error or projection artefact'),
    ]
    if not grid_only and not density:
        legend.append(Line2D([0], [0], marker='o', color='w',
                             markerfacecolor='#2980b9', markersize=6,
                             alpha=0.7, label='Training sample'))
    ax.legend(handles=legend, fontsize=8, loc='best')

    if method == 'pca':
        ax.text(0.01, 0.01,
                'PCA note: low explained variance may compress the grid — try --projection umap.',
                transform=ax.transAxes, fontsize=7, color='gray', style='italic')

    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(results_dir, f'topology_2d_{method}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


# ─── 3D static plot ───────────────────────────────────────────────────────────

def plot_topology_3d(weights: np.ndarray, training_data: np.ndarray,
                     assignments: pd.DataFrame, hex_topology: bool,
                     results_dir: str, output_path: str = None,
                     elev: float = 45, azim: float = 45,
                     grid_only: bool = False, method: str = 'pca',
                     meta: dict = None):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    m, n, dim = weights.shape
    flat_w = weights.reshape(m * n, dim)
    td_proj, wf_proj, _ = _strip_masked(training_data, flat_w, meta or {})

    active_dims = wf_proj.shape[1]
    if active_dims < 3:
        print('WARNING: fewer than 3 active dimensions after masking — skipping 3D plot.')
        return
    if method == 'tsne':
        print('WARNING: t-SNE 3D is very slow on large datasets — consider umap.')

    data_3d, w_3d_flat, var_exp = _project(td_proj, wf_proj, method, 3)
    w_3d = w_3d_flat.reshape(m, n, 3)

    n_neurons = m * n
    dot_size, lw = _auto_scale(n_neurons)

    fig = plt.figure(figsize=(12, 10))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_proj_type('ortho')
    ax.view_init(elev=elev, azim=azim)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor((0.7, 0.7, 0.7, 0.25))
    ax.grid(False)

    if not grid_only and data_3d is not None:
        ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2],
                   color='#2980b9', alpha=0.12, s=4, linewidths=0)

    edges = _grid_edges(m, n, hex_topology)
    edge_colors, stretched, _ = _edge_colors(w_3d, edges)
    for e_idx, ((i1, j1), (i2, j2)) in enumerate(edges):
        ax.plot(
            [w_3d[i1, j1, 0], w_3d[i2, j2, 0]],
            [w_3d[i1, j1, 1], w_3d[i2, j2, 1]],
            [w_3d[i1, j1, 2], w_3d[i2, j2, 2]],
            color=edge_colors[e_idx], lw=lw,
            alpha=0.5 if stretched[e_idx] else 0.8,
        )

    nqe      = _neuron_qe(assignments, m, n)
    nqe_flat = nqe.ravel() if nqe is not None else None

    if nqe_flat is not None:
        sc = ax.scatter(
            w_3d[:, :, 0].ravel(), w_3d[:, :, 1].ravel(), w_3d[:, :, 2].ravel(),
            c=nqe_flat, cmap='plasma', s=dot_size, zorder=5,
            edgecolors='white', linewidths=0.3,
            vmin=np.nanpercentile(nqe_flat, 5),
            vmax=np.nanpercentile(nqe_flat, 95),
        )
        fig.colorbar(sc, ax=ax, label='Mean QE per neuron', shrink=0.55, pad=0.1)
    else:
        ax.scatter(w_3d[:, :, 0].ravel(), w_3d[:, :, 1].ravel(), w_3d[:, :, 2].ravel(),
                   c='crimson', s=dot_size, zorder=5,
                   edgecolors='white', linewidths=0.3)

    var_str = (f'PC1 {var_exp[0]:.1%}' if var_exp is not None else 'Dim1')
    ax.set_xlabel(var_str, fontsize=9, labelpad=8)
    ax.set_ylabel('Dim 2', fontsize=9, labelpad=8)
    ax.set_zlabel('Dim 3', fontsize=9, labelpad=8)
    ax.set_title(
        f'SOM Topology 3D [{method.upper()}] — {os.path.basename(os.path.normpath(results_dir))}\n'
        f'{m}×{n} | {"hex" if hex_topology else "rect"} | crossing lines = topological errors',
        fontsize=11, pad=14,
    )

    plt.tight_layout()
    if output_path is None:
        output_path = os.path.join(results_dir, f'topology_3d_{method}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {output_path}')
    plt.close()


# ─── Interactive HTML (Plotly) ─────────────────────────────────────────────────

def plot_topology_html(weights: np.ndarray, training_data: np.ndarray,
                       assignments: pd.DataFrame, hex_topology: bool,
                       results_dir: str, output_path: str = None,
                       grid_only: bool = False, method: str = 'pca',
                       meta: dict = None):
    try:
        import plotly.graph_objects as go
    except ImportError:
        sys.exit('ERROR: plotly not installed. Run: pip install plotly')

    m, n, dim = weights.shape
    flat_w = weights.reshape(m * n, dim)

    td_proj, wf_proj, _ = _strip_masked(training_data, flat_w, meta or {})
    data_2d, w_2d_flat, var_exp = _project(td_proj, wf_proj, method, 2)
    w_2d = w_2d_flat.reshape(m, n, 2)

    nqe      = _neuron_qe(assignments, m, n)
    nqe_flat = nqe.ravel() if nqe is not None else np.zeros(m * n)

    fig = go.Figure()

    # Layer 1: training data (skip if not available)
    if not grid_only and data_2d is not None:
        fig.add_trace(go.Scatter(
            x=data_2d[:, 0], y=data_2d[:, 1],
            mode='markers',
            marker=dict(color='#2980b9', size=3, opacity=0.30),
            name='Training data',
            hoverinfo='skip',
            legendgroup='data',
        ))

    # Layer 3: grid edges — normal + stretched as separate traces for clear legend
    edges = _grid_edges(m, n, hex_topology)
    _, stretched, _ = _edge_colors(w_2d, edges)

    ex_n, ey_n, ex_s, ey_s = [], [], [], []
    for e_idx, ((i1, j1), (i2, j2)) in enumerate(edges):
        seg_x = [w_2d[i1, j1, 0], w_2d[i2, j2, 0], None]
        seg_y = [w_2d[i1, j1, 1], w_2d[i2, j2, 1], None]
        if stretched[e_idx]:
            ex_s += seg_x; ey_s += seg_y
        else:
            ex_n += seg_x; ey_n += seg_y

    fig.add_trace(go.Scatter(
        x=ex_n, y=ey_n, mode='lines',
        line=dict(color='#2c3e50', width=0.8),
        name='Grid edge (normal)', hoverinfo='skip', opacity=0.75,
    ))
    if ex_s:
        fig.add_trace(go.Scatter(
            x=ex_s, y=ey_s, mode='lines',
            line=dict(color='#e67e22', width=0.8, dash='dot'),
            name='Grid edge (stretched — topology error or projection artefact)',
            hoverinfo='skip', opacity=0.6,
        ))

    # Layer 2: neurons with hover info
    hover = [
        f'Neuron [{i},{j}]<br>Mean QE: {nqe[i,j]:.4f}'
        for i in range(m) for j in range(n)
    ]
    fig.add_trace(go.Scatter(
        x=w_2d[:, :, 0].ravel(),
        y=w_2d[:, :, 1].ravel(),
        mode='markers',
        marker=dict(
            color=nqe_flat,
            colorscale='Plasma',
            size=8,
            colorbar=dict(title='Mean QE', thickness=14),
            line=dict(color='white', width=0.5),
        ),
        text=hover,
        hovertemplate='%{text}<extra></extra>',
        name='Neurons',
    ))

    var_str = (f'PC1 {var_exp[0]:.1%} / PC2 {var_exp[1]:.1%}'
               if var_exp is not None else method.upper())
    fig.update_layout(
        title=dict(
            text=(f'SOM Topology [{method.upper()}] — '
                  f'{os.path.basename(os.path.normpath(results_dir))}<br>'
                  f'<sub>{m}×{n} grid | {"hex" if hex_topology else "rect"} | '
                  f'{var_str} | crossing lines = topological errors</sub>'),
            font_size=14,
        ),
        xaxis_title='Dim 1',
        yaxis_title='Dim 2',
        yaxis_scaleanchor='x',
        plot_bgcolor='#f8f8f8',
        hovermode='closest',
        autosize=True,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    if output_path is None:
        output_path = os.path.join(results_dir, f'topology_interactive_{method}.html')

    # Full-viewport HTML — plot fills the entire browser window
    html = fig.to_html(full_html=True, include_plotlyjs='cdn', config={'responsive': True})
    html = html.replace(
        '<body>',
        '<body style="margin:0;padding:0;overflow:hidden;">'
    ).replace(
        '<div id="',
        '<div style="width:100vw;height:100vh;" id="'
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'Saved: {output_path}')


# ─── Compare plot (2×2: data/weights × PCA/ISOMAP) ───────────────────────────

def plot_topology_compare(weights: np.ndarray, training_data: np.ndarray,
                          hex_topology: bool, results_dir: str,
                          output_path: str = None, meta: dict = None,
                          assignments=None):
    """
    2×2 ablation grid:
      col 0 = PCA (linear)   col 1 = ISOMAP (geodesic)
      row 0 = training data  row 1 = SOM weight vectors

    Designed for Swiss Roll: PCA collapses the spiral, ISOMAP unfolds it.
    Points coloured by row index (≈ unrolling parameter t for Swiss Roll).
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    flat_w = weights.reshape(-1, weights.shape[2])
    td_clean, wf_clean, _ = _strip_masked(training_data, flat_w, meta or {})
    m, n = weights.shape[0], weights.shape[1]
    n_data    = len(td_clean)
    n_neurons = len(wf_clean)

    methods = [('PCA', 'pca'), ('ISOMAP', 'isomap')]
    rows    = [('Trénovací data', td_clean),
               (f'Váhy SOM ({m}×{n})', wf_clean)]

    # Data colour: row index 0..N-1 (for Swiss Roll this ≈ spiral parameter t)
    data_color = np.arange(n_data, dtype=float)

    # Weight colour: for each neuron, mean row index of its assigned samples.
    # This puts neurons on the same colour scale as the data they represent.
    # Using assignments CSV (bmu_i, bmu_j) avoids recomputing BMUs.
    weight_color = np.zeros(n_neurons, dtype=float)
    if assignments is not None:
        try:
            for ni in range(n_neurons):
                bi, bj = divmod(ni, n)
                mask = (assignments['bmu_i'].astype(int) == bi) & \
                       (assignments['bmu_j'].astype(int) == bj)
                # sample_id is 1-based; row index = sample_id - 1
                sids = assignments.loc[mask, 'sample_id'].astype(int).values - 1
                valid = sids[(sids >= 0) & (sids < n_data)]
                if len(valid) > 0:
                    weight_color[ni] = float(np.mean(valid))
        except Exception:
            weight_color = np.linspace(0, n_data - 1, n_neurons)
    else:
        # Fallback: linear interpolation across neurons
        weight_color = np.linspace(0, n_data - 1, n_neurons)

    edges = _grid_edges(m, n, hex_topology)

    fig = plt.figure(figsize=(14, 11))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28,
                            top=0.92, bottom=0.06, left=0.06, right=0.97)

    for col, (method_label, method_key) in enumerate(methods):
        print(f'  projecting {method_key.upper()} …', end=' ', flush=True)
        try:
            data_proj, w_proj, _ = _project(td_clean, wf_clean, method_key)
        except Exception as exc:
            print(f'FAILED ({exc})')
            for row in range(2):
                ax = fig.add_subplot(gs[row, col])
                ax.text(0.5, 0.5, f'{method_key.upper()}\nfailed:\n{exc}',
                        ha='center', va='center', transform=ax.transAxes, fontsize=8)
                ax.axis('off')
            continue
        print('ok')

        for row, (row_label, _) in enumerate(rows):
            ax = fig.add_subplot(gs[row, col])

            clim = dict(vmin=0, vmax=n_data - 1)
            if row == 0:
                ax.scatter(data_proj[:, 0], data_proj[:, 1],
                           c=data_color, cmap='plasma', s=6, alpha=0.55,
                           linewidths=0, rasterized=True, **clim)
            else:
                for (i0, j0), (i1, j1) in edges:
                    x0, y0 = w_proj[i0 * n + j0]
                    x1, y1 = w_proj[i1 * n + j1]
                    ax.plot([x0, x1], [y0, y1], color='#555', lw=0.7,
                            alpha=0.7, zorder=1)
                ax.scatter(w_proj[:, 0], w_proj[:, 1],
                           c=weight_color, cmap='plasma', s=22,
                           edgecolors='white', linewidths=0.3,
                           zorder=2, **clim)

            ax.set_title(f'{method_label} — {row_label}', fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

    cbar_ax = fig.add_axes([0.15, 0.02, 0.70, 0.018])
    sm = plt.cm.ScalarMappable(cmap='plasma',
                               norm=plt.Normalize(0, n_data - 1))
    sm.set_array([])
    plt.colorbar(sm, cax=cbar_ax, orientation='horizontal',
                 label='Row index (≈ unrolling parameter t for Swiss Roll)')

    fig.suptitle(
        'Topology comparison: PCA (linear) vs ISOMAP (geodesic)\n'
        'Correct SOM training → ISOMAP weight grid unfolds to a flat rectangle',
        fontsize=12,
    )

    if output_path is None:
        output_path = os.path.join(results_dir, 'topology_compare_pca_isomap.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {output_path}')


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Visualize SOM topology (weight grid) in projected space.')
    parser.add_argument('results_dir', help='Path to SOM results directory')
    parser.add_argument('--projection', default='pca',
                        choices=['pca', 'umap', 'tsne', 'isomap'],
                        help='Dimensionality reduction algorithm (default: pca)')
    parser.add_argument('--compare', action='store_true',
                        help='Generate 2×2 PCA vs ISOMAP comparison grid '
                             '(ablation / Swiss Roll validation)')
    parser.add_argument('--3d', dest='do_3d', action='store_true',
                        help='Also generate 3D topology plot (not supported for tsne)')
    parser.add_argument('--only3d', action='store_true',
                        help='Generate only the 3D plot (skip 2D)')
    parser.add_argument('--html', action='store_true',
                        help='Generate interactive Plotly HTML (zoomable, hover info)')
    parser.add_argument('--hex', action='store_true',
                        help='Use hexagonal grid connectivity')
    parser.add_argument('--grid-only', action='store_true',
                        help='Hide training samples — pure grid geometry')
    parser.add_argument('--density', action='store_true',
                        help='Replace scatter with hexbin density map (2D only)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output path for 2D PNG')
    parser.add_argument('--output3d', default=None,
                        help='Output path for 3D PNG')
    parser.add_argument('--output-html', default=None,
                        help='Output path for interactive HTML')
    parser.add_argument('--elev', type=float, default=45,
                        help='3D elevation angle (default: 45)')
    parser.add_argument('--azim', type=float, default=45,
                        help='3D azimuth angle (default: 45)')
    args = parser.parse_args()

    if not os.path.isdir(args.results_dir):
        sys.exit(f'ERROR: directory not found: {args.results_dir}')

    weights, training_data, assignments, meta = _load(args.results_dir)

    if weights is None:
        sys.exit('ERROR: csv/weights.npy not found')
    if training_data is None:
        print('INFO: csv/training_data.npy not found — weights-only mode (grid rendered without data scatter)')

    m, n, dim = weights.shape
    n_neurons = m * n
    always_masked = meta.get('_always_masked_cols', [])
    active_dims = dim - len(always_masked)
    td_shape = training_data.shape if training_data is not None else 'N/A'
    print(f'Weights: {weights.shape}  Training data: {td_shape}  Neurons: {n_neurons}')
    if always_masked:
        print(f'Masked dims (excluded from projection): {always_masked}  '
              f'Active dims for projection: {active_dims}')

    hex_topology = args.hex or bool(meta.get('hex_topology', False))
    print(f'Grid: {m}×{n}  Topology: {"hex" if hex_topology else "rect"}  '
          f'Projection: {args.projection.upper()}')

    if n_neurons > 200 and args.projection == 'pca':
        print('HINT: large map detected — consider --projection umap for better unfolding.')

    if args.compare:
        print('Generating PCA vs ISOMAP comparison grid …')
        plot_topology_compare(weights, training_data, hex_topology,
                              args.results_dir, meta=meta,
                              assignments=assignments)

    if not args.only3d and not args.compare:
        plot_topology_2d(weights, training_data, assignments, hex_topology,
                         args.results_dir, args.output,
                         grid_only=args.grid_only, density=args.density,
                         method=args.projection, meta=meta)

    if (args.do_3d or args.only3d) and not args.compare:
        plot_topology_3d(weights, training_data, assignments, hex_topology,
                         args.results_dir, args.output3d,
                         elev=args.elev, azim=args.azim,
                         grid_only=args.grid_only, method=args.projection, meta=meta)

    if args.html and not args.compare:
        plot_topology_html(weights, training_data, assignments, hex_topology,
                           args.results_dir, args.output_html,
                           grid_only=args.grid_only, method=args.projection, meta=meta)


if __name__ == '__main__':
    main()
