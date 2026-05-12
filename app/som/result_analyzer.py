"""
Post-run SOM result analyzer.

Reads saved outputs from a results directory (clusters.json, preprocessing_info.json,
pie_data_*.json, quantization_errors.json, run_metrics.json, original_input.csv)
and builds a rich llm_context.json for LLM-based analysis.

Usage as standalone:
    python3 app/som/result_analyzer.py <results_dir>
"""
import json
import os
import argparse

import numpy as np
import pandas as pd


def _load_json(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _infer_primary_id(preprocessing_info: dict, dataset_meta: dict) -> str | None:
    """Return primary_id column name: from dataset_meta if present, else infer from nunique_ratio."""
    if "ds_primary_id_col" in dataset_meta:
        return dataset_meta["ds_primary_id_col"]
    # Column whose every value is unique is almost certainly the record identifier
    for col, info in preprocessing_info.items():
        if info.get("nunique_ratio", 0.0) >= 0.99:
            return col
    return None


def _classify_columns(preprocessing_info: dict, primary_id_col: str | None) -> dict:
    """Classify columns into numeric, categorical, noise from preprocessing_info."""
    numeric = []
    categorical = []
    noise = []
    for col, info in preprocessing_info.items():
        if col == primary_id_col:
            continue
        if info.get('status') == 'ignored':
            noise.append(col)
        elif info.get('is_categorical', False):
            categorical.append(col)
        elif info.get('base_type') == 'numeric':
            numeric.append(col)
    return {
        "numeric": numeric,
        "categorical": categorical,
        "noise": noise,
        "primary_id": primary_id_col,
    }


def _compute_dimension_stats(df: pd.DataFrame | None, numeric_cols: list, primary_id_col: str | None) -> dict:
    if df is None or not numeric_cols:
        return {}
    stats = {}
    for col in numeric_cols:
        if col not in df.columns or col == primary_id_col:
            continue
        series = df[col].dropna()
        if series.empty:
            continue
        stats[col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
        }
    return stats


def _build_cluster_stat(
    neuron_key: str,
    sample_ids: list,
    neuron_qe: dict,
    pie_data: dict,
    columns: dict,
    df: pd.DataFrame | None,
    primary_id_col: str | None,
) -> dict:
    cluster = {
        "neuron": neuron_key,
        "sample_count": len(sample_ids),
        "sample_ids": sample_ids,
        "quantization_error": neuron_qe.get(neuron_key, 0.0),
    }

    # Per-categorical: dominant value and purity from pie_data
    dominant = {}
    purity = {}
    for col, col_pie in pie_data.items():
        counts = col_pie.get("counts", {}).get(neuron_key, {})
        categories = col_pie.get("categories", {})
        if not counts:
            continue
        total = sum(counts.values())
        if total == 0:
            continue
        dominant_code = max(counts, key=lambda k: counts[k])
        dominant[col] = categories.get(dominant_code, dominant_code)
        purity[col] = round(counts[dominant_code] / total, 4)

    if dominant:
        cluster["dominant_category"] = dominant
    if purity:
        cluster["purity"] = purity

    # Per-numeric: cluster-level mean from original data
    if df is not None and columns["numeric"] and primary_id_col and primary_id_col in df.columns:
        subset = df[df[primary_id_col].isin(sample_ids)]
        if not subset.empty:
            means = {}
            for col in columns["numeric"]:
                if col in subset.columns:
                    means[col] = round(float(subset[col].mean()), 4)
            if means:
                cluster["dimension_means"] = means

    return cluster


def _build_anomalies(extremes: dict, id_to_neuron: dict, primary_id_col: str | None) -> dict:
    if not extremes:
        return {"global_outlier_count": 0, "local_outlier_count": 0, "top_anomalies": []}

    pid_prefix = f"Value '{primary_id_col}'" if primary_id_col else None

    top_anomalies = []
    global_count = 0
    local_count = 0

    for sample_id_raw, reasons in extremes.items():
        # Filter reasons that refer to the primary_id column — not analytically meaningful
        if pid_prefix:
            reasons = [r for r in reasons if pid_prefix not in r]
        if not reasons:
            continue

        try:
            sample_id = int(sample_id_raw)
        except (ValueError, TypeError):
            sample_id = sample_id_raw

        neuron = id_to_neuron.get(sample_id, id_to_neuron.get(sample_id_raw, "unknown"))

        has_global = any("global" in r.lower() for r in reasons)
        has_local = any("neuron" in r.lower() for r in reasons)
        if has_global:
            global_count += 1
        if has_local:
            local_count += 1

        top_anomalies.append({
            "sample_id": sample_id,
            "neuron": neuron,
            "reasons": reasons,
        })

    top_anomalies.sort(key=lambda x: len(x["reasons"]), reverse=True)
    return {
        "global_outlier_count": global_count,
        "local_outlier_count": local_count,
        "top_anomalies": top_anomalies,
    }


def _map_shape_from_weights(results_dir: str) -> tuple[int, int] | None:
    """Infer (m, n) map dimensions from weights.npy shape."""
    weights_path = os.path.join(results_dir, "csv", "weights.npy")
    if not os.path.isfile(weights_path):
        return None
    try:
        w = np.load(weights_path)
        if w.ndim == 3:
            return w.shape[0], w.shape[1]
    except Exception:
        pass
    return None


def analyze_results(results_dir: str) -> dict:
    """
    Read all saved SOM outputs from results_dir and return a rich context dict
    matching the format expected by llm/src/context_builder.py.
    """
    json_dir = os.path.join(results_dir, "json")
    csv_dir = os.path.join(results_dir, "csv")

    dataset_meta = _load_json(os.path.join(results_dir, "dataset_meta.json")) or {}
    preprocessing_info = _load_json(os.path.join(json_dir, "preprocessing_info.json")) or {}
    clusters = _load_json(os.path.join(json_dir, "clusters.json")) or {}
    qe_data = _load_json(os.path.join(json_dir, "quantization_errors.json")) or {}
    extremes = _load_json(os.path.join(json_dir, "extremes.json")) or {}
    run_metrics = _load_json(os.path.join(results_dir, "run_metrics.json")) or {}

    primary_id_col = _infer_primary_id(preprocessing_info, dataset_meta)
    columns = _classify_columns(preprocessing_info, primary_id_col)

    # Load original data for per-cluster numeric means (can be None if file missing)
    original_df = _load_csv(os.path.join(csv_dir, "original_input.csv"))

    # Build sample_id → neuron lookup (int and str keys for robustness)
    id_to_neuron: dict = {}
    for neuron_key, ids in clusters.items():
        for sid in ids:
            id_to_neuron[sid] = neuron_key
            id_to_neuron[str(sid)] = neuron_key

    # --- Map overview ---
    map_shape = run_metrics.get("map_size") or None
    if not map_shape:
        inferred = _map_shape_from_weights(results_dir)
        map_shape = list(inferred) if inferred else []

    total_neurons = (
        (map_shape[0] * map_shape[1]) if len(map_shape) == 2 else None
        or len(qe_data.get("neuron_quantization_errors", {}))
        or len(clusters)
    )
    active_neurons = len(clusters)
    dead_neurons = max(0, total_neurons - active_neurons)

    map_section = {
        "size": map_shape,
        "topology": run_metrics.get("map_topology", "hex"),
        "total_samples": dataset_meta.get("ds_n_samples", sum(len(v) for v in clusters.values())),
        "total_neurons": total_neurons,
        "active_neurons": active_neurons,
        "dead_neurons": dead_neurons,
        "dead_ratio": round(dead_neurons / total_neurons, 4) if total_neurons > 0 else 0.0,
        "mqe": qe_data.get("total_quantization_error"),
        "topographic_error": run_metrics.get("topographic_error"),
    }

    # --- Dimension statistics (global, numeric cols only) ---
    dimension_stats = _compute_dimension_stats(original_df, columns["numeric"], primary_id_col)

    # --- Load per-categorical pie data ---
    pie_data: dict = {}
    for cat_col in columns["categorical"]:
        pie_path = os.path.join(json_dir, f"pie_data_{cat_col}.json")
        d = _load_json(pie_path)
        if d:
            pie_data[cat_col] = d

    neuron_qe = qe_data.get("neuron_quantization_errors", {})

    # --- Per-cluster stats ---
    cluster_list = []
    for neuron_key, sample_ids in clusters.items():
        cluster = _build_cluster_stat(
            neuron_key, sample_ids, neuron_qe,
            pie_data, columns, original_df, primary_id_col,
        )
        cluster_list.append(cluster)

    cluster_list.sort(key=lambda c: c["sample_count"], reverse=True)

    # --- Anomalies ---
    anomalies = _build_anomalies(extremes, id_to_neuron, primary_id_col)

    return {
        "map": map_section,
        "columns": columns,
        "clusters": cluster_list,
        "anomalies": anomalies,
        "dimension_stats": dimension_stats,
    }


def save_llm_context(results_dir: str) -> str:
    """Analyze results and write llm_context.json. Returns the output path."""
    context = analyze_results(results_dir)
    json_dir = os.path.join(results_dir, "json")
    os.makedirs(json_dir, exist_ok=True)
    output_path = os.path.join(json_dir, "llm_context.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(context, f, indent=2, ensure_ascii=False)
    print(f"INFO: LLM context saved to '{output_path}'")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build llm_context.json from SOM results directory")
    parser.add_argument("results_dir", help="Path to SOM results directory")
    args = parser.parse_args()
    save_llm_context(args.results_dir)
