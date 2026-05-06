#!/usr/bin/env python3
"""
Prepare MLP Training Dataset from EA Results

Scans data/results/ across all datasets and seeds, merges results.csv files,
filters penalized individuals, applies a fixed one-hot schema, and saves a
combined dataset ready for MLP training.

Usage:
    python3 prepare_dataset.py --results_root ../../data/results
    python3 prepare_dataset.py --results_root ../../data/results --include_penalized
    python3 prepare_dataset.py --results_root ../../data/results --output data/custom.csv
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

# --- Schema ---

NUMERIC_FEATURES = [
    'start_learning_rate',
    'end_learning_rate',
    'start_radius_init_ratio',
    'start_batch_percent',
    'end_batch_percent',
    'epoch_multiplier',
    'growth_g',
    'num_batches',
    'map_m',
    'map_n',
]

DATASET_FEATURES = [
    'ds_n_samples',
    'ds_n_active_dimensions',
    'ds_n_numeric',
    'ds_n_categorical',
    'ds_missing_ratio',
]

# Fixed schema — must stay stable across all datasets and training runs.
# log-growth removed from batch_growth_type; keep the column list explicit so
# old data with log-growth doesn't silently expand the feature space.
CATEGORICAL_SCHEMA = {
    'lr_decay_type':     ['exp-drop', 'linear-drop', 'log-drop', 'step-down'],
    'radius_decay_type': ['exp-drop', 'linear-drop', 'log-drop', 'step-down'],
    'batch_growth_type': ['exp-growth', 'linear-growth'],
}

TARGET_COLS = [
    'raw_mqe_improvement_ratio',  # primary — dataset-independent (higher = better)
    'raw_topographic_error',
    'dead_neuron_ratio',
]


def discover_result_files(results_root: str) -> list[str]:
    pattern = os.path.join(results_root, '*/results/*/seed_*/results.csv')
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No results.csv files found under '{results_root}'.\n"
            f"Expected pattern: {pattern}"
        )
    return files


def load_and_merge(result_files: list[str]) -> pd.DataFrame:
    frames = []
    for path in result_files:
        df = pd.read_csv(path)
        # Attach source path for debugging
        df['_source'] = path
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(result_files)} files → {len(merged)} rows total")
    return merged


def filter_individuals(df: pd.DataFrame, include_penalized: bool,
                       include_legacy_types: bool) -> pd.DataFrame:
    before = len(df)
    if not include_penalized and 'is_penalized' in df.columns:
        df = df[~df['is_penalized'].astype(bool)].copy()
        print(f"Excluded penalized: {before - len(df)} rows → {len(df)} remaining")

    if not include_legacy_types and 'batch_growth_type' in df.columns:
        # log-growth is no longer in the search space; its all-zeros encoding is ambiguous
        before2 = len(df)
        removed = set(df['batch_growth_type'].unique()) - set(CATEGORICAL_SCHEMA['batch_growth_type'])
        if removed:
            df = df[df['batch_growth_type'].isin(CATEGORICAL_SCHEMA['batch_growth_type'])].copy()
            print(f"Excluded legacy batch_growth_type {removed}: "
                  f"{before2 - len(df)} rows → {len(df)} remaining")

    print(f"Total after filtering: {len(df)} / {before} ({100*len(df)/before:.1f}%)")
    return df


def one_hot_fixed(df: pd.DataFrame) -> pd.DataFrame:
    for col, values in CATEGORICAL_SCHEMA.items():
        if col not in df.columns:
            print(f"  WARNING: '{col}' not found in data — filling with zeros")
            for v in values:
                df[f'{col}_{v}'] = 0
            continue
        unknown = set(df[col].dropna().unique()) - set(values)
        if unknown:
            print(f"  WARNING: '{col}' has unknown values {unknown} — will be encoded as all-zeros")
        for v in values:
            df[f'{col}_{v}'] = (df[col] == v).astype(int)
        df = df.drop(columns=[col])
    return df


def build_feature_columns() -> list[str]:
    features = list(NUMERIC_FEATURES) + list(DATASET_FEATURES)
    for col, values in CATEGORICAL_SCHEMA.items():
        features += [f'{col}_{v}' for v in values]
    return features


def prepare_dataset(results_root: str, output_path: str, include_penalized: bool,
                    include_legacy_types: bool = False) -> None:
    print(f"\n{'='*70}")
    print("MLP DATASET PREPARATION")
    print(f"{'='*70}\n")

    result_files = discover_result_files(results_root)
    print(f"Found {len(result_files)} seed result files across "
          f"{len(set(f.split('/results/')[0] for f in result_files))} datasets\n")

    df = load_and_merge(result_files)

    # Summary before filtering
    if 'dataset_name' in df.columns:
        print("\nSamples per dataset (before filter):")
        for ds, count in df['dataset_name'].value_counts().sort_index().items():
            print(f"  {ds}: {count}")

    df = filter_individuals(df, include_penalized, include_legacy_types)

    # One-hot encode categoricals with fixed schema
    print("\nApplying fixed one-hot schema...")
    df = one_hot_fixed(df)

    feature_cols = build_feature_columns()
    missing_features = [c for c in feature_cols if c not in df.columns]
    if missing_features:
        print(f"\nWARNING: Missing feature columns: {missing_features}")
        feature_cols = [c for c in feature_cols if c in df.columns]

    missing_targets = [c for c in TARGET_COLS if c not in df.columns]
    if missing_targets:
        raise ValueError(f"Missing target columns: {missing_targets}")

    X = df[feature_cols].copy()
    y = df[TARGET_COLS].copy()
    ds_name = df['dataset_name'].copy() if 'dataset_name' in df.columns else pd.Series(['unknown'] * len(df))

    # Drop rows with NaN in features or targets
    valid = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
    dropped = (~valid).sum()
    if dropped:
        print(f"\nDropped {dropped} rows with NaN values")
    X = X[valid].reset_index(drop=True)
    y = y[valid].reset_index(drop=True)
    ds_name = ds_name[valid].reset_index(drop=True)

    # dataset_name is first column — used for stratified split, not a feature
    dataset = pd.concat([ds_name, X, y], axis=1)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    dataset.to_csv(output_path, index=False)

    metadata = {
        'num_samples': len(dataset),
        'num_features': len(feature_cols),
        'num_targets': len(TARGET_COLS),
        'feature_columns': feature_cols,
        'target_columns': TARGET_COLS,
        'categorical_schema': CATEGORICAL_SCHEMA,
        'include_penalized': include_penalized,
        'source_files': result_files,
        'dropped_rows': int(dropped),
    }
    metadata_path = output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print(f"Samples:  {len(dataset)}")
    print(f"Features: {len(feature_cols)}")
    print(f"Targets:  {TARGET_COLS}")
    print(f"\nTarget statistics:")
    print(y.describe().to_string())
    print(f"\nSaved: {output_path}")
    print(f"Meta:  {metadata_path}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MLP training dataset from EA results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 prepare_dataset.py --results_root ../../data/results
  python3 prepare_dataset.py --results_root ../../data/results --include_penalized
  python3 prepare_dataset.py --results_root ../../data/results --output data/my_dataset.csv
        """
    )
    parser.add_argument('--results_root', required=True,
                        help='Root of data/results/ directory')
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: data/all_combined_mlp.csv next to this script)')
    parser.add_argument('--include_penalized', action='store_true',
                        help='Include penalized individuals (default: exclude)')
    parser.add_argument('--include_legacy_types', action='store_true',
                        help='Include configs with removed batch_growth_type values '
                             'e.g. log-growth (default: exclude)')
    args = parser.parse_args()

    if not os.path.isdir(args.results_root):
        print(f"Error: results_root not found: {args.results_root}")
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = args.output or os.path.join(script_dir, 'data', 'all_combined_mlp.csv')

    try:
        prepare_dataset(args.results_root, output_path, args.include_penalized,
                        args.include_legacy_types)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
