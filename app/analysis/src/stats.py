"""
Statistical computations on SOM results.
All functions are pure (no IO) and receive the data dict from loader.py.
"""
import numpy as np
import pandas as pd


# ─── Global statistics ────────────────────────────────────────────────────────

def compute_global_stats(data: dict) -> dict:
    """
    A1-A5: Global statistics across the full dataset.
    Returns {dimension_stats, category_distributions}.
    """
    df = data['original_df']
    columns = data['columns']
    primary_id = columns['primary_id']

    dimension_stats = _global_numeric_stats(df, columns['numeric'], primary_id)
    category_distributions = _global_category_distributions(df, columns['categorical'])

    return {
        'dimension_stats': dimension_stats,
        'category_distributions': category_distributions,
    }


def _global_numeric_stats(df: pd.DataFrame | None, numeric_cols: list,
                           primary_id: str | None) -> dict:
    if df is None or not numeric_cols:
        return {}
    stats = {}
    for col in numeric_cols:
        if col not in df.columns or col == primary_id:
            continue
        s = df[col].dropna()
        if s.empty:
            continue
        pcts = np.percentile(s, [25, 50, 75, 90, 95])
        stats[col] = {
            'min':    float(s.min()),
            'max':    float(s.max()),
            'mean':   round(float(s.mean()), 4),
            'std':    round(float(s.std()), 4),
            'median': round(float(pcts[1]), 4),
            'p25':    round(float(pcts[0]), 4),
            'p75':    round(float(pcts[2]), 4),
            'p90':    round(float(pcts[3]), 4),
            'p95':    round(float(pcts[4]), 4),
        }
    return stats


def _global_category_distributions(df: pd.DataFrame | None,
                                    categorical_cols: list) -> dict:
    if df is None or not categorical_cols:
        return {}
    distributions = {}
    for col in categorical_cols:
        if col not in df.columns:
            continue
        counts = df[col].value_counts(normalize=True)
        distributions[col] = {str(k): round(float(v), 4) for k, v in counts.items()}
    return distributions


# ─── Per-cluster statistics ───────────────────────────────────────────────────

def compute_cluster_stats(data: dict, global_stats: dict) -> list:
    """
    B1-B8: Statistics per neuron cluster.
    Returns list of cluster dicts sorted by sample_count descending.
    """
    clusters    = data['clusters']
    df          = data['original_df']
    columns     = data['columns']
    pie_data    = data['pie_data']
    neuron_qe   = data['qe_data'].get('neuron_quantization_errors', {})
    primary_id  = columns['primary_id']
    dim_stats   = global_stats.get('dimension_stats', {})

    result = []
    for neuron_key, sample_ids in clusters.items():
        cluster = {'neuron': neuron_key, 'sample_count': len(sample_ids),
                   'sample_ids': sample_ids,
                   'quantization_error': neuron_qe.get(neuron_key, 0.0)}

        # B2: dominant category + purity from pie_data
        dominant, purity = _pie_dominant_purity(neuron_key, pie_data)
        if dominant:
            cluster['dominant_category'] = dominant
        if purity:
            cluster['purity'] = purity

        # B7: full category counts per column
        cat_counts = _pie_category_counts(neuron_key, pie_data)
        if cat_counts:
            cluster['category_counts'] = cat_counts

        # B3-B5: numeric stats per cluster (mean, median, std, min, max)
        if df is not None and columns['numeric'] and primary_id in (df.columns if df is not None else []):
            subset = df[df[primary_id].isin(sample_ids)]
            if not subset.empty:
                num_stats = _cluster_numeric_stats(subset, columns['numeric'], primary_id)
                if num_stats:
                    cluster['dimension_stats'] = num_stats

                # B8: Z-score deviation from global mean
                deviation = _cluster_global_deviation(
                    {col: v['mean'] for col, v in num_stats.items()}, dim_stats)
                if deviation:
                    cluster['global_deviation'] = deviation

        result.append(cluster)

    result.sort(key=lambda c: c['sample_count'], reverse=True)
    return result


def _pie_dominant_purity(neuron_key: str, pie_data: dict) -> tuple[dict, dict]:
    dominant, purity = {}, {}
    for col, col_pie in pie_data.items():
        counts = col_pie.get('counts', {}).get(neuron_key, {})
        categories = col_pie.get('categories', {})
        if not counts:
            continue
        total = sum(counts.values())
        if total == 0:
            continue
        dom_code = max(counts, key=lambda k: counts[k])
        dominant[col] = categories.get(dom_code, dom_code)
        purity[col] = round(counts[dom_code] / total, 4)
    return dominant, purity


def _pie_category_counts(neuron_key: str, pie_data: dict) -> dict:
    """B7: per-cluster counts per categorical value (label → count)."""
    result = {}
    for col, col_pie in pie_data.items():
        counts = col_pie.get('counts', {}).get(neuron_key, {})
        categories = col_pie.get('categories', {})
        if not counts:
            continue
        result[col] = {categories.get(code, code): cnt for code, cnt in counts.items()}
    return result


def _cluster_numeric_stats(subset: pd.DataFrame, numeric_cols: list,
                            primary_id: str | None) -> dict:
    """B3-B5: mean, median, std, min, max for numeric cols in this cluster."""
    stats = {}
    for col in numeric_cols:
        if col not in subset.columns or col == primary_id:
            continue
        s = subset[col].dropna()
        if s.empty:
            continue
        stats[col] = {
            'mean':   round(float(s.mean()), 4),
            'median': round(float(s.median()), 4),
            'std':    round(float(s.std()), 4) if len(s) > 1 else 0.0,
            'min':    float(s.min()),
            'max':    float(s.max()),
        }
    return stats


def _cluster_global_deviation(cluster_means: dict, dim_stats: dict) -> dict:
    """B8: Z-score of cluster mean vs global mean per numeric column."""
    deviation = {}
    for col, c_mean in cluster_means.items():
        g = dim_stats.get(col, {})
        if g.get('std', 0) > 0:
            deviation[col] = round((c_mean - g['mean']) / g['std'], 3)
    return deviation


# ─── Trustworthiness & Continuity ────────────────────────────────────────────

def compute_tc(data: dict, k_values: tuple = (5, 10, 20)) -> dict:
    """
    Trustworthiness & Continuity (Venna & Kaski, 2001).
    T(k): are map-space neighbors true data-space neighbors?
    C(k): are data-space neighbors preserved in map space?
    Both in [0, 1], 1 = perfect.

    Returns {k: {trustworthiness, continuity}} for each k,
    or {} if data is unavailable or too small.
    """
    X = data.get('training_data')
    if X is None or X.ndim != 2:
        return {}

    n = X.shape[0]
    if n < 10:
        return {}

    # Build row_idx → (bmu_i, bmu_j) using id_to_row + clusters
    clusters  = data.get('clusters', {})
    id_to_row = data.get('id_to_row', {})

    bmu_coords = np.zeros((n, 2), dtype=np.float32)
    found = 0
    for neuron_key, sample_ids in clusters.items():
        parts = neuron_key.split('_')
        ni, nj = int(parts[0]), int(parts[1])
        for sid in sample_ids:
            row_idx = id_to_row.get(sid, id_to_row.get(str(sid)))
            if row_idx is not None and 0 <= row_idx < n:
                bmu_coords[row_idx] = [ni, nj]
                found += 1

    if found < n * 0.9:
        return {}

    # Cap at 2000 samples for O(N²) feasibility
    if n > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, 2000, replace=False)
        X          = X[idx]
        bmu_coords = bmu_coords[idx]
        n = 2000

    from scipy.spatial.distance import cdist as _cdist
    data_dist = _cdist(X,          X,          'euclidean')  # (n, n)
    map_dist  = _cdist(bmu_coords, bmu_coords, 'euclidean')  # (n, n)

    # Rank matrices: rank 0 = self, rank 1 = nearest neighbor
    data_rank = np.argsort(np.argsort(data_dist, axis=1), axis=1)
    map_rank  = np.argsort(np.argsort(map_dist,  axis=1), axis=1)

    results = {}
    for k in k_values:
        if k >= n:
            continue
        norm = 2.0 / (n * k * (2 * n - 3 * k - 1))

        # T(k): map-k neighbors that are NOT data-k neighbors
        in_map_k      = (map_rank  > 0) & (map_rank  <= k)
        not_in_data_k = data_rank > k
        t_sum = float(np.sum((data_rank - k) * (in_map_k & not_in_data_k)))

        # C(k): data-k neighbors that are NOT map-k neighbors
        in_data_k    = (data_rank > 0) & (data_rank <= k)
        not_in_map_k = map_rank > k
        c_sum = float(np.sum((map_rank - k) * (in_data_k & not_in_map_k)))

        results[k] = {
            'trustworthiness': round(1.0 - norm * t_sum, 4),
            'continuity':      round(1.0 - norm * c_sum, 4),
        }

    return results


# ─── Silhouette score ─────────────────────────────────────────────────────────

def compute_silhouette(data: dict, n_max: int = 2000) -> dict:
    """
    Per-neuron and global mean silhouette score.
    s(i) = (b - a) / max(a, b), range [-1, 1], 1 = perfectly separated.
    Singleton neurons (1 sample) are excluded from global mean because
    sklearn sets a(i)=0 for singletons, which inflates their score to 1.0.
    Returns {'global': float, 'per_neuron': {neuron_key: float}} or {}.
    """
    X = data.get('training_data')
    if X is None or X.ndim != 2:
        return {}

    clusters  = data.get('clusters', {})
    id_to_row = data.get('id_to_row', {})
    n         = X.shape[0]

    neuron_keys = list(clusters.keys())
    if len(neuron_keys) < 2:
        return {}

    key_to_label = {k: i for i, k in enumerate(neuron_keys)}

    labels = np.full(n, -1, dtype=np.int32)
    for neuron_key, sample_ids in clusters.items():
        label = key_to_label[neuron_key]
        for sid in sample_ids:
            row_idx = id_to_row.get(sid, id_to_row.get(str(sid)))
            if row_idx is not None and 0 <= row_idx < n:
                labels[row_idx] = label

    valid_mask = labels >= 0
    X_v      = X[valid_mask]
    labels_v = labels[valid_mask]

    if len(X_v) < 4 or len(np.unique(labels_v)) < 2:
        return {}

    if len(X_v) > n_max:
        rng = np.random.default_rng(42)
        idx  = rng.choice(len(X_v), n_max, replace=False)
        X_v      = X_v[idx]
        labels_v = labels_v[idx]

    try:
        from sklearn.metrics import silhouette_samples
        scores = silhouette_samples(X_v, labels_v)
    except Exception:
        return {}

    # Labels with exactly 1 sample (after possible subsampling)
    uniq, counts = np.unique(labels_v, return_counts=True)
    singleton_labels = set(uniq[counts == 1].tolist())

    per_neuron: dict = {}
    non_singleton_scores: list = []
    for neuron_key, label in key_to_label.items():
        mask = labels_v == label
        if not mask.any():
            continue
        neuron_scores = scores[mask]
        per_neuron[neuron_key] = round(float(neuron_scores.mean()), 4)
        if label not in singleton_labels:
            non_singleton_scores.extend(neuron_scores.tolist())

    global_mean = (round(float(np.mean(non_singleton_scores)), 4)
                   if non_singleton_scores else None)
    return {'global': global_mean, 'per_neuron': per_neuron}


# ─── Spatial map analysis ─────────────────────────────────────────────────────
# Mathematical replacement of the closed CNN track (docs/cnn/CNN_REQUIREMENTS.md):
# fixed operators on the raw weight matrix instead of learned visual assessment.

def compute_spatial_stats(data: dict) -> dict:
    """
    D4: Spatial quality of the organized map, computed directly on
    weights[m, n, dim] — no images, no training.

    Per feature: spatial gradients, Moran's I (spatial autocorrelation),
    local extrema counts. Per categorical column: dominant-category region
    structure (see compute_regions).

    spatial_quality_score (0-1, higher = better organized) is the mean of:
      - organization:     (mean Moran's I + 1) / 2 — smooth maps autocorrelate
      - smoothness:       1 / (1 + median gradient roughness)
      - region_coherence: 1 - mean dominant-category boundary ratio
        (only when categorical pie data exists)

    Note: adjacency is rook (4-neighbor grid). For hex maps this is an
    approximation, but consistent across runs and therefore comparable.
    """
    weights = data.get('weights')
    if weights is None or getattr(weights, 'ndim', 0) != 3:
        return {}

    feature_names = _weight_feature_names(data, weights.shape[2])
    skip = {data['columns'].get('primary_id')} | set(data['columns'].get('noise', []))

    per_feature: dict = {}
    morans: list = []
    roughness_vals: list = []
    for d, name in enumerate(feature_names):
        if name in skip:
            continue
        plane = weights[:, :, d].astype(float)
        gy, gx = np.gradient(plane)
        grad_mag = np.hypot(gy, gx)
        plane_std = float(plane.std())
        roughness = float(grad_mag.mean() / plane_std) if plane_std > 0 else None
        mi = _morans_i(plane)
        n_max, n_min = _count_local_extrema(plane)
        entry = {
            'gradient_mean': round(float(grad_mag.mean()), 4),
            'gradient_max':  round(float(grad_mag.max()), 4),
            'n_local_maxima': n_max,
            'n_local_minima': n_min,
        }
        if roughness is not None:
            entry['roughness'] = round(roughness, 4)
            roughness_vals.append(roughness)
        if mi is not None:
            entry['morans_i'] = round(mi, 4)
            morans.append(mi)
        per_feature[name] = entry

    regions = compute_regions(data)

    components: list = []
    result: dict = {'per_feature': per_feature}
    if morans:
        morans_mean = float(np.mean(morans))
        result['morans_i_mean'] = round(morans_mean, 4)
        components.append((morans_mean + 1.0) / 2.0)
    if roughness_vals:
        roughness_median = float(np.median(roughness_vals))
        result['gradient_roughness_median'] = round(roughness_median, 4)
        components.append(1.0 / (1.0 + roughness_median))
    if regions:
        result['regions'] = regions
        boundary_ratios = [r['boundary_ratio'] for r in regions.values()
                           if r.get('boundary_ratio') is not None]
        if boundary_ratios:
            components.append(1.0 - float(np.mean(boundary_ratios)))

    if components:
        result['spatial_quality_score'] = round(float(np.mean(components)), 4)
    return result


def compute_regions(data: dict) -> dict:
    """
    D5: Regional structure of dominant categories on the map.
    For each categorical column, flood-fills (4-connectivity) contiguous
    neurons sharing the same dominant category.

    Returns {col: {n_regions, boundary_ratio, regions: [{category, size,
    center: [row, col]}, ...largest 10]}}.
    """
    weights = data.get('weights')
    map_shape = (data.get('run_metrics') or {}).get('map_size') or []
    if len(map_shape) != 2 and weights is not None and getattr(weights, 'ndim', 0) == 3:
        map_shape = [weights.shape[0], weights.shape[1]]
    if len(map_shape) != 2:
        return {}

    try:
        from scipy import ndimage
    except ImportError:
        return {}

    result: dict = {}
    for col, col_pie in (data.get('pie_data') or {}).items():
        mat, code_to_label = _dominant_matrix(col_pie, tuple(map_shape))
        if mat is None:
            continue
        regions: list = []
        for code in np.unique(mat[mat >= 0]):
            labeled, n_blobs = ndimage.label(mat == code)
            for blob_id in range(1, n_blobs + 1):
                cells = np.argwhere(labeled == blob_id)
                center = cells.mean(axis=0)
                regions.append({
                    'category': code_to_label.get(int(code), str(code)),
                    'size':     int(len(cells)),
                    'center':   [round(float(center[0]), 1), round(float(center[1]), 1)],
                })
        regions.sort(key=lambda r: -r['size'])
        result[col] = {
            'n_regions':      len(regions),
            'boundary_ratio': _boundary_ratio(mat),
            'regions':        regions[:10],
        }
    return result


def _weight_feature_names(data: dict, dim: int) -> list:
    """Weight dims follow the training input column order (see
    som/visualization.py generate_component_planes); fall back to dim_i."""
    df = data.get('original_df')
    if df is not None and len(df.columns) == dim:
        return list(df.columns)
    return [f'dim_{i}' for i in range(dim)]


def _morans_i(plane: np.ndarray) -> float | None:
    """Moran's I with rook adjacency. None for constant planes."""
    z = plane - plane.mean()
    denom = float((z ** 2).sum())
    if denom == 0:
        return None
    num = 2.0 * float((z[:, :-1] * z[:, 1:]).sum() + (z[:-1, :] * z[1:, :]).sum())
    w_sum = 2 * (z[:, :-1].size + z[:-1, :].size)
    return (z.size / w_sum) * (num / denom)


def _count_local_extrema(plane: np.ndarray) -> tuple:
    """Strict local maxima/minima in a 3x3 neighborhood (plateaus excluded)."""
    try:
        from scipy.ndimage import maximum_filter, minimum_filter
    except ImportError:
        return 0, 0
    mx = maximum_filter(plane, size=3)
    mn = minimum_filter(plane, size=3)
    not_flat = mx > mn
    return (int(((plane == mx) & not_flat).sum()),
            int(((plane == mn) & not_flat).sum()))


def _dominant_matrix(col_pie: dict, shape: tuple):
    """(m, n) int matrix of dominant category codes per neuron, -1 = no samples.
    Returns (matrix, code → label map); (None, {}) if pie data is unusable."""
    counts = col_pie.get('counts', {})
    categories = col_pie.get('categories', {})
    if not counts:
        return None, {}
    mat = np.full(shape, -1, dtype=int)
    raw_to_code: dict = {}
    code_to_label: dict = {}
    for neuron_key, cnts in counts.items():
        if not cnts or sum(cnts.values()) == 0:
            continue
        try:
            r, c = (int(x) for x in str(neuron_key).split('_'))
        except ValueError:
            continue
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            continue
        dom_raw = max(cnts, key=lambda k: cnts[k])
        code = raw_to_code.setdefault(dom_raw, len(raw_to_code))
        code_to_label[code] = str(categories.get(dom_raw, dom_raw))
        mat[r, c] = code
    if (mat >= 0).sum() == 0:
        return None, {}
    return mat, code_to_label


def _boundary_ratio(mat: np.ndarray) -> float | None:
    """Fraction of adjacent (rook) neuron pairs, both active, whose dominant
    categories differ. 0 = one homogeneous block, 1 = checkerboard."""
    pairs = 0
    mismatches = 0
    h_valid = (mat[:, :-1] >= 0) & (mat[:, 1:] >= 0)
    pairs += int(h_valid.sum())
    mismatches += int((h_valid & (mat[:, :-1] != mat[:, 1:])).sum())
    v_valid = (mat[:-1, :] >= 0) & (mat[1:, :] >= 0)
    pairs += int(v_valid.sum())
    mismatches += int((v_valid & (mat[:-1, :] != mat[1:, :])).sum())
    if pairs == 0:
        return None
    return round(mismatches / pairs, 4)


# ─── Map topology ─────────────────────────────────────────────────────────────

def compute_topology(data: dict) -> dict:
    """
    D1-D2: Map-level topology metrics.
    """
    clusters    = data['clusters']
    run_metrics = data['run_metrics']
    weights     = data['weights']

    map_shape = run_metrics.get('map_size') or []
    if not map_shape and weights is not None and weights.ndim == 3:
        map_shape = [weights.shape[0], weights.shape[1]]

    total_neurons  = (map_shape[0] * map_shape[1]) if len(map_shape) == 2 else (
        len(data['qe_data'].get('neuron_quantization_errors', {})) or len(clusters))
    active_neurons = len(clusters)
    dead_neurons   = max(0, total_neurons - active_neurons)

    sizes = [len(v) for v in clusters.values()] if clusters else [0]
    sizes_arr = np.array(sizes)

    # Gini coefficient — 0 = perfectly even, 1 = everything in one neuron
    sorted_s = np.sort(sizes_arr)
    n = len(sorted_s)
    gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_s) / (n * sorted_s.sum()) - (n + 1) / n
            ) if sorted_s.sum() > 0 else 0.0

    return {
        'size':               map_shape,
        'topology':           run_metrics.get('map_topology', 'hex'),
        'total_neurons':      total_neurons,
        'active_neurons':     active_neurons,
        'dead_neurons':       dead_neurons,
        'dead_ratio':         round(dead_neurons / total_neurons, 4) if total_neurons > 0 else 0.0,
        'coverage_ratio':     round(active_neurons / total_neurons, 4) if total_neurons > 0 else 0.0,
        'density_gini':       round(float(gini), 4),
        'max_cluster_size':   int(sizes_arr.max()),
        'min_cluster_size':   int(sizes_arr.min()),
        'median_cluster_size': round(float(np.median(sizes_arr)), 1),
        'mqe':                data['qe_data'].get('total_quantization_error'),
        'topographic_error':  run_metrics.get('topographic_error'),
        'total_samples':      data['dataset_meta'].get('ds_n_samples',
                                                        sum(len(v) for v in clusters.values())),
    }
