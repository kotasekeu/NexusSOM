"""
Load all SOM result files from a results directory into a single data dict.
No computation happens here — only IO and type normalization.
"""
import json
import os

import numpy as np
import pandas as pd


def _load_json(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def _load_csv(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_npy(path: str) -> np.ndarray | None:
    if not os.path.isfile(path):
        return None
    try:
        return np.load(path)
    except Exception:
        return None


def _infer_primary_id(preprocessing_info: dict, dataset_meta: dict) -> str | None:
    if dataset_meta.get('ds_primary_id_col'):
        return dataset_meta['ds_primary_id_col']
    for col, info in preprocessing_info.items():
        if info.get('nunique_ratio', 0.0) >= 0.99:
            return col
    return None


def _classify_columns(preprocessing_info: dict, primary_id_col: str | None) -> dict:
    numeric, categorical, noise = [], [], []
    for col, info in preprocessing_info.items():
        if col == primary_id_col:
            continue
        if info.get('status') == 'ignored':
            noise.append(col)
        elif info.get('is_categorical', False):
            categorical.append(col)
        elif info.get('base_type') == 'numeric':
            numeric.append(col)
    return {'numeric': numeric, 'categorical': categorical,
            'noise': noise, 'primary_id': primary_id_col}


def _load_pie_data(json_dir: str, categorical_cols: list) -> dict:
    pie_data = {}
    for col in categorical_cols:
        d = _load_json(os.path.join(json_dir, f'pie_data_{col}.json'))
        if d:
            pie_data[col] = d
    return pie_data


def _build_id_to_row(original_df: pd.DataFrame | None, primary_id_col: str | None) -> dict:
    """Map sample_id → row index in original_df / training_data.npy (same order)."""
    if original_df is None or not primary_id_col or primary_id_col not in original_df.columns:
        return {}
    return dict(zip(original_df[primary_id_col], original_df.index))


def load_results(results_dir: str) -> dict:
    """
    Load all SOM result artefacts from results_dir.
    Returns a dict with clean, typed data — no computation.
    """
    json_dir = os.path.join(results_dir, 'json')
    csv_dir = os.path.join(results_dir, 'csv')

    dataset_meta     = _load_json(os.path.join(results_dir, 'dataset_meta.json')) or {}
    run_metrics      = _load_json(os.path.join(results_dir, 'run_metrics.json')) or {}
    preprocessing    = _load_json(os.path.join(json_dir, 'preprocessing_info.json')) or {}
    clusters         = _load_json(os.path.join(json_dir, 'clusters.json')) or {}
    qe_data          = _load_json(os.path.join(json_dir, 'quantization_errors.json')) or {}
    extremes         = _load_json(os.path.join(json_dir, 'extremes.json')) or {}

    primary_id_col   = _infer_primary_id(preprocessing, dataset_meta)
    columns          = _classify_columns(preprocessing, primary_id_col)
    pie_data         = _load_pie_data(json_dir, columns['categorical'])

    original_df      = _load_csv(os.path.join(csv_dir, 'original_input.csv'))
    weights          = _load_npy(os.path.join(csv_dir, 'weights.npy'))
    training_data    = _load_npy(os.path.join(csv_dir, 'training_data.npy'))

    id_to_row        = _build_id_to_row(original_df, primary_id_col)

    # Build reverse: sample_id → neuron_key
    id_to_neuron: dict = {}
    for neuron_key, ids in clusters.items():
        for sid in ids:
            id_to_neuron[sid] = neuron_key
            id_to_neuron[str(sid)] = neuron_key

    return {
        'results_dir':    results_dir,
        'dataset_meta':   dataset_meta,
        'run_metrics':    run_metrics,
        'preprocessing':  preprocessing,
        'columns':        columns,
        'clusters':       clusters,       # {neuron_key: [sample_ids]}
        'qe_data':        qe_data,
        'extremes':       extremes,
        'pie_data':       pie_data,       # {col: {categories:{}, counts:{}}}
        'original_df':    original_df,
        'weights':        weights,        # (m, n, dim) or None
        'training_data':  training_data,  # (N, dim) or None
        'id_to_row':      id_to_row,      # sample_id → df row index
        'id_to_neuron':   id_to_neuron,
    }
