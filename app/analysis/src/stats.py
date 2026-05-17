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
