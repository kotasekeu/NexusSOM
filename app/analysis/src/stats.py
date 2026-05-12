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
