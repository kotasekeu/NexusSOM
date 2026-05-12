"""
Anomaly detection on SOM results.
All functions are pure (no IO) and receive the data dict from loader.py.
"""
import numpy as np
import pandas as pd


# ─── Public entry point ───────────────────────────────────────────────────────

def detect_anomalies(data: dict, global_stats: dict, cluster_stats: list) -> dict:
    """
    C4-C8: Anomaly detection across clusters and globally.
    Returns {global_outlier_count, local_outlier_count, top_anomalies,
             cluster_local_outliers}.
    """
    df          = data['original_df']
    columns     = data['columns']
    pie_data    = data['pie_data']
    primary_id  = columns['primary_id']
    numeric_cols = columns['numeric']

    all_local: list  = []
    cluster_summaries: dict = {}

    for cluster in cluster_stats:
        neuron_key  = cluster['neuron']
        sample_ids  = cluster['sample_ids']
        n           = len(sample_ids)

        # Need at least 3 samples to detect local outliers
        if n < 3:
            continue

        subset = None
        if df is not None and primary_id and primary_id in df.columns:
            subset = df[df[primary_id].isin(sample_ids)]

        outliers: list = []

        # C4/C7: numeric outliers (σ-based + multi-dimensional)
        if subset is not None and not subset.empty and numeric_cols:
            outliers += _detect_numeric_outliers(
                subset, primary_id, numeric_cols, neuron_key, threshold=2.5)

        # Categorical minority in high-purity cluster
        cat_outliers = _detect_categorical_outliers(
            sample_ids, neuron_key, pie_data, purity_threshold=0.85)
        outliers += cat_outliers

        if outliers:
            cluster_summaries[neuron_key] = outliers
            all_local.extend(outliers)

        # C6: "1 of N" — one sample significantly farther from cluster centroid
        one_of_n = None
        if subset is not None and not subset.empty and numeric_cols and n >= 5:
            one_of_n = _detect_one_of_n(
                subset, primary_id, numeric_cols, ratio_threshold=3.0)
        if one_of_n:
            one_of_n['neuron'] = neuron_key
            # Merge into existing cluster outlier entry if present
            existing = next(
                (o for o in all_local if o.get('sample_id') == one_of_n['sample_id']),
                None)
            if existing:
                existing['one_of_n'] = True
                existing['distance_ratio'] = one_of_n['distance_ratio']
            else:
                entry = {
                    'sample_id':      one_of_n['sample_id'],
                    'neuron':         neuron_key,
                    'type':           'one_of_n',
                    'distance_ratio': one_of_n['distance_ratio'],
                    'reasons':        [
                        f"Isolated outlier in neuron {neuron_key}: distance ratio "
                        f"{one_of_n['distance_ratio']:.2f}× cluster median "
                        f"(cluster size {n})"
                    ],
                }
                all_local.append(entry)
                cluster_summaries.setdefault(neuron_key, []).append(entry)

    # Enrich with global extremes from extremes.json
    global_outliers = _enrich_global_extremes(data)

    # Build ranked top_anomalies list
    top_anomalies = _rank_anomalies(all_local, global_outliers, limit=20)

    local_ids  = {o['sample_id'] for o in all_local}
    global_ids = {o['sample_id'] for o in global_outliers}

    return {
        'global_outlier_count':   len(global_ids),
        'local_outlier_count':    len(local_ids),
        'top_anomalies':          top_anomalies,
        'cluster_local_outliers': cluster_summaries,
    }


# ─── Numeric outlier detection ────────────────────────────────────────────────

def _detect_numeric_outliers(subset: pd.DataFrame, primary_id: str | None,
                              numeric_cols: list, neuron_key: str,
                              threshold: float = 2.5) -> list:
    """C4: per-sample z-score vs cluster mean/std. C7: flag multi-dim outliers."""
    col_stats = {}
    for col in numeric_cols:
        if col not in subset.columns or col == primary_id:
            continue
        s = subset[col].dropna()
        if len(s) < 3:
            continue
        std = float(s.std())
        if std == 0:
            continue
        col_stats[col] = {'mean': float(s.mean()), 'std': std}

    if not col_stats:
        return []

    outlier_map: dict = {}  # sample_id → {col: z_score, ...}

    id_col = primary_id if primary_id and primary_id in subset.columns else None

    for _, row in subset.iterrows():
        sid = row[id_col] if id_col else _row_index(row)
        for col, cs in col_stats.items():
            val = row.get(col)
            if pd.isna(val):
                continue
            z = abs((float(val) - cs['mean']) / cs['std'])
            if z > threshold:
                outlier_map.setdefault(sid, {})[col] = round(z, 3)

    result = []
    for sid, cols in outlier_map.items():
        n_dims = len(cols)
        reasons = []
        for col, z in cols.items():
            reasons.append(
                f"{'Multi-dim outlier' if n_dims > 1 else 'Outlier'} in {col}: "
                f"z={z:.2f} vs cluster mean (neuron {neuron_key})"
            )
        result.append({
            'sample_id':  sid,
            'neuron':     neuron_key,
            'type':       'multi_dim' if n_dims > 1 else 'numeric',
            'n_outlier_dims': n_dims,
            'outlier_cols': cols,
            'reasons':    reasons,
        })
    return result


# ─── Categorical minority detection ──────────────────────────────────────────

def _detect_categorical_outliers(sample_ids: list, neuron_key: str,
                                  pie_data: dict,
                                  purity_threshold: float = 0.85) -> list:
    """Flag samples with minority categorical value in a high-purity cluster."""
    result = []
    for col, col_pie in pie_data.items():
        counts = col_pie.get('counts', {}).get(neuron_key, {})
        categories = col_pie.get('categories', {})
        if not counts:
            continue
        total = sum(counts.values())
        if total == 0:
            continue
        dom_code = max(counts, key=lambda k: counts[k])
        purity = counts[dom_code] / total
        if purity < purity_threshold:
            continue

        # Identify which sample IDs hold minority values
        # pie_data doesn't have per-sample assignments, so we report the count
        minority_count = total - counts[dom_code]
        if minority_count == 0:
            continue

        dom_label = categories.get(dom_code, dom_code)
        minority_codes = [c for c in counts if c != dom_code]
        minority_labels = [categories.get(c, c) for c in minority_codes]

        result.append({
            'sample_id':  f'cluster:{neuron_key}',
            'neuron':     neuron_key,
            'type':       'categorical_minority',
            'column':     col,
            'minority_count': minority_count,
            'minority_labels': minority_labels,
            'dominant_label':  dom_label,
            'purity':     round(purity, 4),
            'reasons':    [
                f"{minority_count} sample(s) in neuron {neuron_key} have "
                f"{col}={'/'.join(minority_labels)} while {purity:.0%} show "
                f"{col}={dom_label}"
            ],
        })
    return result


# ─── "1 of N" pattern ─────────────────────────────────────────────────────────

def _detect_one_of_n(subset: pd.DataFrame, primary_id: str | None,
                     numeric_cols: list,
                     ratio_threshold: float = 3.0) -> dict | None:
    """
    C6: find the one sample whose Euclidean distance from the cluster centroid
    is much larger than the others (ratio > ratio_threshold × median distance).
    Returns {sample_id, outlier_distance, median_distance, distance_ratio} or None.
    """
    usable = [c for c in numeric_cols
              if c in subset.columns and c != primary_id and
              subset[c].notna().all()]
    if not usable:
        return None

    mat = subset[usable].values.astype(float)
    centroid = mat.mean(axis=0)
    dists = np.linalg.norm(mat - centroid, axis=1)

    median_dist = float(np.median(dists))
    if median_dist == 0:
        return None

    max_idx = int(np.argmax(dists))
    max_dist = float(dists[max_idx])
    ratio = max_dist / median_dist

    if ratio < ratio_threshold:
        return None

    id_col = primary_id if primary_id and primary_id in subset.columns else None
    sid = subset.iloc[max_idx][id_col] if id_col else int(subset.index[max_idx])

    return {
        'sample_id':       sid,
        'outlier_distance': round(max_dist, 4),
        'median_distance':  round(median_dist, 4),
        'distance_ratio':   round(ratio, 3),
    }


# ─── Global extremes enrichment ───────────────────────────────────────────────

def _enrich_global_extremes(data: dict) -> list:
    """
    C8: Wrap extremes.json entries as anomaly dicts. Skip primary_id column.
    extremes.json format: {sample_id: [reason_string, ...]}
    Reason strings look like: "Value 'COL' (1.23) is the global minimum."
    Returns list of {sample_id, neuron, type, reasons}.
    """
    extremes     = data.get('extremes', {})
    id_to_neuron = data.get('id_to_neuron', {})
    primary_id   = data['columns']['primary_id']

    result = []
    for raw_sid, reasons in extremes.items():
        if not isinstance(reasons, list) or not reasons:
            continue
        # Skip entries that only reference the primary_id column
        non_id_reasons = [
            r for r in reasons
            if primary_id is None or f"'{primary_id}'" not in r
        ]
        if not non_id_reasons:
            continue
        try:
            sid = int(raw_sid)
        except (ValueError, TypeError):
            sid = raw_sid
        neuron = id_to_neuron.get(sid) or id_to_neuron.get(str(sid))
        result.append({
            'sample_id': sid,
            'neuron':    neuron,
            'type':      'global_extreme',
            'reasons':   non_id_reasons,
        })
    return result


# ─── Ranking ──────────────────────────────────────────────────────────────────

def _rank_anomalies(local: list, global_ex: list, limit: int = 20) -> list:
    """
    Merge local and global anomalies into a single ranked list.
    Priority: multi_dim > one_of_n > numeric > global_extreme > categorical_minority.
    """
    type_priority = {
        'multi_dim':            0,
        'one_of_n':             1,
        'numeric':              2,
        'global_extreme':       3,
        'categorical_minority': 4,
    }

    # Merge by sample_id — global extreme enriches existing entries
    merged: dict = {}
    for item in local:
        sid = item['sample_id']
        merged[sid] = dict(item)

    for item in global_ex:
        sid = item['sample_id']
        if sid in merged:
            merged[sid].setdefault('reasons', []).extend(item['reasons'])
            merged[sid]['type'] = 'multi_dim' if merged[sid].get('n_outlier_dims', 0) > 1 else merged[sid]['type']
        else:
            merged[sid] = dict(item)

    def sort_key(a):
        priority = type_priority.get(a.get('type', ''), 99)
        n_dims = a.get('n_outlier_dims', 0)
        ratio  = a.get('distance_ratio', 0.0)
        return (priority, -n_dims, -ratio)

    ranked = sorted(merged.values(), key=sort_key)
    return ranked[:limit]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _row_index(row) -> int:
    return int(row.name)
