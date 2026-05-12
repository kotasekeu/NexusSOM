"""
Assembles final llm_context.json from loader, stats, and anomalies outputs.
Single public function: build_llm_context(results_dir) -> dict
"""
import json
import os

from .loader import load_results
from .stats import compute_global_stats, compute_cluster_stats, compute_topology
from .anomalies import detect_anomalies


def build_llm_context(results_dir: str) -> dict:
    """
    Load SOM results and compute all statistics.
    Returns the llm_context dict (same schema as llm_context.json).
    """
    data         = load_results(results_dir)
    global_stats = compute_global_stats(data)
    topology     = compute_topology(data)
    clusters     = compute_cluster_stats(data, global_stats)
    anomalies    = detect_anomalies(data, global_stats, clusters)

    return {
        'map':             topology,
        'clusters':        _serialize_clusters(clusters),
        'anomalies':       _serialize_anomalies(anomalies),
        'dimension_stats': global_stats.get('dimension_stats', {}),
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


# ─── Serialization helpers ────────────────────────────────────────────────────

def _serialize_clusters(clusters: list) -> list:
    """Convert cluster list to llm_context.json cluster schema."""
    result = []
    for c in clusters:
        entry = {
            'neuron':              c['neuron'],
            'sample_count':        c['sample_count'],
            'quantization_error':  c.get('quantization_error', 0.0),
        }
        if 'dominant_category' in c:
            entry['dominant_category'] = c['dominant_category']
        if 'purity' in c:
            entry['purity'] = c['purity']
        if 'category_counts' in c:
            entry['category_counts'] = c['category_counts']
        if 'dimension_stats' in c:
            entry['dimension_stats'] = c['dimension_stats']
            # Flat means dict for context_builder.py legacy key
            entry['dimension_means'] = {
                col: v['mean'] for col, v in c['dimension_stats'].items()
            }
        if 'global_deviation' in c:
            entry['global_deviation'] = c['global_deviation']
        # Omit raw sample_ids to keep JSON compact
        result.append(entry)
    return result


def _serialize_anomalies(anomalies: dict) -> dict:
    """
    Filter out cluster-level categorical_minority pseudo-entries
    (those have sample_id = 'cluster:X') from top_anomalies for the LLM.
    Keep them in cluster_local_outliers for detailed inspection.
    """
    top = [
        a for a in anomalies.get('top_anomalies', [])
        if not str(a.get('sample_id', '')).startswith('cluster:')
    ]
    return {
        'global_outlier_count': anomalies.get('global_outlier_count', 0),
        'local_outlier_count':  anomalies.get('local_outlier_count', 0),
        'top_anomalies':        top,
        'cluster_local_outliers': {
            k: [
                {kk: vv for kk, vv in o.items() if kk != 'sample_id' or
                 not str(vv).startswith('cluster:')}
                for o in v
            ]
            for k, v in anomalies.get('cluster_local_outliers', {}).items()
        },
    }


def _json_default(obj):
    if hasattr(obj, 'item'):   # numpy scalar
        return obj.item()
    if hasattr(obj, 'tolist'): # numpy array
        return obj.tolist()
    return str(obj)
