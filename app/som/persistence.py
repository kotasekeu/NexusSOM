"""
Persistence layer for the SOM module.

All disk writes for a results directory live here. The core algorithm
(som.py) and the preprocessing stage (preprocess.py) are pure — they return
data structures, and the orchestrator (run.py, EA, tools) decides what to
save via these functions.

Results directory layout (verified by tests/integration/test_som_pipeline.py):

    <working_dir>/
    ├── log.txt, run_metrics.json, dataset_meta.json
    ├── csv/    original_input.csv, training_data.npy, training_data_readable.csv,
    │           ignore_mask.csv, weights.npy, weights_readable.csv,
    │           training_checkpoints.json, sample_coverage.json, sample_assignments.csv
    ├── json/   preprocessing_info.json, clusters.json, quantization_errors.json,
    │           extremes.json, pie_data_*.json, llm_context.json
    └── visualizations/ *.png
"""
import json
import os

import numpy as np
import pandas as pd


def _csv_dir(working_dir: str) -> str:
    path = os.path.join(working_dir, 'csv')
    os.makedirs(path, exist_ok=True)
    return path


def _json_dir(working_dir: str) -> str:
    path = os.path.join(working_dir, 'json')
    os.makedirs(path, exist_ok=True)
    return path


def save_weights(weights: np.ndarray, working_dir: str) -> str:
    """Save final weights as binary .npy plus a human-readable CSV.
    Returns the .npy path."""
    csv_dir = _csv_dir(working_dir)
    m, n, dim = weights.shape

    npy_path = os.path.join(csv_dir, 'weights.npy')
    np.save(npy_path, weights)

    coords = np.indices((m, n)).transpose(1, 2, 0).reshape(-1, 2)
    header = ['neuron_i', 'neuron_j'] + [f'dim_{d}' for d in range(dim)]
    csv_data = np.hstack([coords, weights.reshape(-1, dim)])
    np.savetxt(os.path.join(csv_dir, 'weights_readable.csv'), csv_data,
               delimiter=',', header=','.join(header), comments='')
    return npy_path


def save_training_checkpoints(checkpoints: list, working_dir: str) -> str | None:
    """Save training checkpoints (LSTM training data). No-op for empty list."""
    if not checkpoints:
        return None
    path = os.path.join(_csv_dir(working_dir), 'training_checkpoints.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(checkpoints, f, indent=2)
    return path


def save_sample_coverage(coverage: dict | None, working_dir: str) -> str | None:
    """Save sample coverage statistics incl. per-sample counts. No-op for None."""
    if not coverage:
        return None
    path = os.path.join(_csv_dir(working_dir), 'sample_coverage.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(coverage, f)
    return path


def save_run_metrics(metrics: dict, working_dir: str) -> str:
    path = os.path.join(working_dir, 'run_metrics.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)
    return path


def save_preprocess_artifacts(result, original_df: pd.DataFrame, working_dir: str) -> str:
    """
    Save all preprocessing outputs (original input, normalized training data,
    ignore mask, preprocessing info, dataset stats).

    `result` is a PreprocessResult from som.preprocess.preprocess_data.
    Returns the training_data.npy path.
    """
    csv_dir = _csv_dir(working_dir)
    json_dir = _json_dir(working_dir)

    original_df.to_csv(os.path.join(csv_dir, 'original_input.csv'), index=False)

    npy_path = os.path.join(csv_dir, 'training_data.npy')
    np.save(npy_path, result.training_data)
    pd.DataFrame(result.training_data).to_csv(
        os.path.join(csv_dir, 'training_data_readable.csv'), index=False, header=False)

    pd.DataFrame(result.ignore_mask).to_csv(
        os.path.join(csv_dir, 'ignore_mask.csv'), index=False, header=False)

    with open(os.path.join(json_dir, 'preprocessing_info.json'), 'w', encoding='utf-8') as f:
        json.dump(result.preprocessing_info, f, indent=2, ensure_ascii=False, default=str)

    with open(os.path.join(working_dir, 'dataset_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(result.dataset_stats, f, indent=2)

    return npy_path
