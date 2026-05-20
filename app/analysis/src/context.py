"""
Assembles final llm_context.json from loader, stats, and anomalies outputs.
Single public function: build_llm_context(results_dir) -> dict
"""
import json
import os

import pandas as pd

from .loader import load_results
from .stats import (compute_global_stats, compute_cluster_stats, compute_topology,
                    compute_tc, compute_silhouette)
from .anomalies import detect_anomalies, compute_sample_qe


def build_llm_context(results_dir: str) -> dict:
    """
    Load SOM results and compute all statistics.
    Returns the llm_context dict (same schema as llm_context.json).
    """
    data         = load_results(results_dir)
    global_stats = compute_global_stats(data)
    topology     = compute_topology(data)
    tc           = compute_tc(data)
    if tc:
        topology['trustworthiness_continuity'] = tc
    silhouette   = compute_silhouette(data)
    if silhouette.get('global') is not None:
        topology['silhouette'] = silhouette['global']
    clusters     = compute_cluster_stats(data, global_stats)
    sil_per_n    = silhouette.get('per_neuron', {})
    for c in clusters:
        if c['neuron'] in sil_per_n:
            c['silhouette'] = sil_per_n[c['neuron']]
    anomalies    = detect_anomalies(data, global_stats, clusters)
    sample_qe    = compute_sample_qe(data)
    anomaly_records = _build_anomaly_records(
        data, anomalies.get('top_anomalies', []), global_stats, sample_qe)

    return {
        'map':              topology,
        'clusters':         _serialize_clusters(clusters),
        'anomalies':        _serialize_anomalies(anomalies),
        'anomaly_records':  anomaly_records,
        'dimension_stats':  global_stats.get('dimension_stats', {}),
        'category_distributions': global_stats.get('category_distributions', {}),
    }


def save_llm_context(results_dir: str) -> str:
    """
    Build context and write to <results_dir>/json/llm_context.json.
    Returns the output path.
    """
    context = build_llm_context(results_dir)
    out_path = os.path.join(results_dir, 'json', 'llm_context.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2, default=_json_default)
    return out_path


# ─── Anomaly records — full row with delta annotations ────────────────────────

def _build_anomaly_records(data: dict, top_anomalies: list,
                           global_stats: dict, sample_qe: dict | None = None) -> list:
    """
    For each top anomaly (real sample_id only), load the full row from
    original_df and annotate each numeric column with:
      value, delta (vs global mean), delta_pct, flag (global_min/global_max/
      high_deviation/moderate_deviation).
    Categorical columns get value + whether it matches cluster dominant.
    """
    df          = data['original_df']
    primary_id  = data['columns']['primary_id']
    numeric_cols = data['columns']['numeric']
    cat_cols    = data['columns']['categorical']
    dim_stats   = global_stats.get('dimension_stats', {})
    id_to_neuron = data['id_to_neuron']

    if df is None or not primary_id or primary_id not in df.columns:
        return []

    # Build neuron → dominant categories lookup from pie_data
    pie_data    = data['pie_data']
    neuron_dom  = _build_neuron_dominant(pie_data)

    records = []
    for anomaly in top_anomalies:
        sid = anomaly.get('sample_id')
        if sid is None or str(sid).startswith('cluster:'):
            continue

        # Find row
        mask = df[primary_id] == sid
        if not mask.any():
            mask = df[primary_id] == str(sid)
        if not mask.any():
            continue
        row = df[mask].iloc[0]

        neuron = anomaly.get('neuron') or id_to_neuron.get(sid) or id_to_neuron.get(str(sid))
        dom    = neuron_dom.get(neuron, {})

        cols: dict = {}

        # Numeric columns
        for col in numeric_cols:
            if col not in row.index or col == primary_id:
                continue
            raw = row[col]
            if pd.isna(raw):
                continue
            val = float(raw)
            g   = dim_stats.get(col, {})
            entry: dict = {'value': round(val, 4)}
            if g:
                mean = g.get('mean', 0)
                delta = val - mean
                entry['mean']      = mean
                entry['delta']     = round(delta, 3)
                entry['delta_pct'] = round(delta / abs(mean) * 100, 1) if mean != 0 else 0.0
                if val == g.get('min'):
                    entry['flag'] = 'global_min'
                elif val == g.get('max'):
                    entry['flag'] = 'global_max'
                elif abs(entry['delta_pct']) >= 50:
                    entry['flag'] = 'high_deviation'
                elif abs(entry['delta_pct']) >= 20:
                    entry['flag'] = 'moderate_deviation'
            cols[col] = entry

        # Categorical columns
        for col in cat_cols:
            if col not in row.index:
                continue
            val = str(row[col])
            entry = {'value': val}
            dom_val = dom.get(col)
            if dom_val and val != str(dom_val):
                entry['differs_from_cluster_dominant'] = str(dom_val)
            cols[col] = entry

        record: dict = {
            'sample_id':      sid,
            'neuron':         neuron,
            'type':           anomaly.get('type'),
            'distance_ratio': anomaly.get('distance_ratio'),
            'columns':        cols,
        }

        if sample_qe and sid in sample_qe:
            sq = sample_qe[sid]
            record['qe'] = sq['qe']
            qe_dims = {k: v for k, v in sq.get('qe_dims', {}).items()
                       if k != primary_id and k != 'id'}
            if qe_dims:
                primary_id_col = data['columns']['primary_id']
                qe_dims = {k: v for k, v in qe_dims.items() if k != primary_id_col}
                record['top_qe_dim'] = max(qe_dims, key=qe_dims.get)
                record['qe_dims']    = dict(
                    sorted(qe_dims.items(), key=lambda x: -x[1])[:6])

        records.append(record)

    return records


def _build_neuron_dominant(pie_data: dict) -> dict:
    """neuron_key → {col: dominant_label}"""
    result: dict = {}
    for col, col_pie in pie_data.items():
        counts     = col_pie.get('counts', {})
        categories = col_pie.get('categories', {})
        for neuron, cnts in counts.items():
            if not cnts:
                continue
            total = sum(cnts.values())
            if total == 0:
                continue
            dom_code = max(cnts, key=lambda k: cnts[k])
            result.setdefault(neuron, {})[col] = categories.get(dom_code, dom_code)
    return result


# ─── Cluster serialization ────────────────────────────────────────────────────

def _serialize_clusters(clusters: list) -> list:
    """
    Compact cluster representation for llm_context.json.
    Drops category_counts (large, rarely needed by LLM) and dimension_means
    alias. Keeps dominant_category, purity, dimension_stats, global_deviation.
    """
    result = []
    for c in clusters:
        entry: dict = {
            'neuron':             c['neuron'],
            'sample_count':       c['sample_count'],
            'quantization_error': round(c.get('quantization_error', 0.0), 4),
        }
        if 'dominant_category' in c:
            entry['dominant_category'] = c['dominant_category']
        if 'purity' in c:
            entry['purity'] = c['purity']
        if 'silhouette' in c:
            entry['silhouette'] = c['silhouette']
        if 'dimension_stats' in c:
            entry['dimension_stats'] = c['dimension_stats']
            # Flat means alias for context_builder.py compatibility
            entry['dimension_means'] = {
                col: v['mean'] for col, v in c['dimension_stats'].items()
            }
        if 'global_deviation' in c:
            entry['global_deviation'] = c['global_deviation']
            sorted_feats = sorted(c['global_deviation'].items(),
                                  key=lambda x: abs(x[1]), reverse=True)
            entry['top_features'] = [
                {'feature': col, 'z_score': z,
                 'direction': 'high' if z > 0 else 'low'}
                for col, z in sorted_feats[:5]
            ]
        result.append(entry)
    return result


# ─── Anomaly serialization ────────────────────────────────────────────────────

def _serialize_anomalies(anomalies: dict) -> dict:
    """
    Keep only top_anomalies (summary) and drop cluster_local_outliers —
    the detailed per-sample data lives in anomaly_records instead.
    """
    top = [
        a for a in anomalies.get('top_anomalies', [])
        if not str(a.get('sample_id', '')).startswith('cluster:')
    ]
    return {
        'global_outlier_count': anomalies.get('global_outlier_count', 0),
        'local_outlier_count':  anomalies.get('local_outlier_count', 0),
        'top_anomalies':        top,
    }


# ─── JSON serialization ───────────────────────────────────────────────────────

def _json_default(obj):
    if hasattr(obj, 'item'):    # numpy scalar
        return obj.item()
    if hasattr(obj, 'tolist'):  # numpy array
        return obj.tolist()
    return str(obj)
