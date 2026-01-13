#!/usr/bin/env python3
"""
Prepare MLP Training Dataset from EA Results

This script extracts hyperparameters and quality metrics from EA results.csv
and prepares them for MLP training.

Usage:
    python3 prepare_dataset.py --results_dir ./test/results/20260112_140511
    python3 prepare_dataset.py --results_dir ./test/results/20260112_140511 --output ./mlp/data/dataset.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json


# Hyperparameter columns to extract
HYPERPARAMETER_COLS = [
    'map_size',
    'start_learning_rate',
    'end_learning_rate',
    'lr_decay_type',
    'start_radius',
    'end_radius',
    'radius_decay_type',
    'start_batch_percent',
    'end_batch_percent',
    'batch_growth_type',
    'epoch_multiplier',
    'normalize_weights_flag',
    'growth_g',
    'num_batches',
    'max_epochs_without_improvement'
]

# Target columns (quality metrics to predict)
TARGET_COLS = [
    'best_mqe',
    'topographic_error',
    'dead_neuron_ratio'
]


def parse_map_size(map_size_str):
    """Parse map_size string '[15, 15]' to two separate features"""
    try:
        # Remove brackets and split
        map_size_str = str(map_size_str).strip('[]')
        parts = map_size_str.split(',')
        rows = int(parts[0].strip())
        cols = int(parts[1].strip())
        return rows, cols
    except:
        return None, None


def encode_categorical_features(df, categorical_cols):
    """
    One-hot encode categorical features.

    Args:
        df: DataFrame
        categorical_cols: List of column names to encode

    Returns:
        Tuple of (encoded_df, encoders_dict)
    """
    df_encoded = df.copy()
    encoders = {}

    for col in categorical_cols:
        if col in df_encoded.columns:
            # Create dummy variables
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=False)
            # Store encoder info
            encoders[col] = list(dummies.columns)
            # Drop original column and add dummies
            df_encoded = df_encoded.drop(columns=[col])
            df_encoded = pd.concat([df_encoded, dummies], axis=1)

    return df_encoded, encoders


def prepare_dataset(results_dir, output_path=None):
    """
    Prepare MLP training dataset from EA results.

    Args:
        results_dir: Path to EA results directory
        output_path: Path to save dataset CSV (default: results_dir/mlp_dataset.csv)

    Returns:
        Tuple of (features_df, targets_df, metadata)
    """
    print(f"\n{'='*80}")
    print("PREPARING MLP TRAINING DATASET")
    print(f"{'='*80}\n")

    # Load results.csv
    results_file = os.path.join(results_dir, "results.csv")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    print(f"Loading results from: {results_file}")
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} configurations")

    # Check for required columns
    missing_hyperparam_cols = [col for col in HYPERPARAMETER_COLS if col not in df.columns]
    missing_target_cols = [col for col in TARGET_COLS if col not in df.columns]

    if missing_hyperparam_cols:
        print(f"\n⚠ Warning: Missing hyperparameter columns: {missing_hyperparam_cols}")

    if missing_target_cols:
        print(f"\n⚠ Warning: Missing target columns: {missing_target_cols}")
        if len(missing_target_cols) == len(TARGET_COLS):
            raise ValueError("No target columns found in results.csv")

    # Extract available columns
    available_hyperparam_cols = [col for col in HYPERPARAMETER_COLS if col in df.columns]
    available_target_cols = [col for col in TARGET_COLS if col in df.columns]

    print(f"\nUsing {len(available_hyperparam_cols)} hyperparameter features")
    print(f"Predicting {len(available_target_cols)} target metrics")

    # Parse map_size into separate features
    if 'map_size' in df.columns:
        print("\nParsing map_size...")
        df['map_rows'], df['map_cols'] = zip(*df['map_size'].apply(parse_map_size))
        available_hyperparam_cols.remove('map_size')
        available_hyperparam_cols.extend(['map_rows', 'map_cols'])

    # Extract features and targets
    features_df = df[available_hyperparam_cols].copy()
    targets_df = df[available_target_cols].copy()

    # Identify categorical columns for encoding
    categorical_cols = [col for col in ['lr_decay_type', 'radius_decay_type', 'batch_growth_type']
                       if col in features_df.columns]

    print(f"\nEncoding {len(categorical_cols)} categorical features...")
    features_encoded, encoders = encode_categorical_features(features_df, categorical_cols)

    # Convert boolean to int
    if 'normalize_weights_flag' in features_encoded.columns:
        features_encoded['normalize_weights_flag'] = features_encoded['normalize_weights_flag'].astype(int)

    # Remove any rows with NaN
    valid_mask = ~(features_encoded.isna().any(axis=1) | targets_df.isna().any(axis=1))
    features_clean = features_encoded[valid_mask].reset_index(drop=True)
    targets_clean = targets_df[valid_mask].reset_index(drop=True)

    removed_rows = len(df) - len(features_clean)
    if removed_rows > 0:
        print(f"\n⚠ Removed {removed_rows} rows with missing values")

    # Print dataset statistics
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}\n")

    print(f"Total samples: {len(features_clean)}")
    print(f"Feature dimensions: {features_clean.shape[1]}")
    print(f"Target dimensions: {targets_clean.shape[1]}")

    print(f"\nFeature columns ({len(features_clean.columns)}):")
    for col in features_clean.columns:
        print(f"  - {col}")

    print(f"\nTarget statistics:")
    print(targets_clean.describe())

    # Save dataset
    if output_path is None:
        output_path = os.path.join(results_dir, "mlp_dataset.csv")

    # Combine features and targets
    dataset = pd.concat([features_clean, targets_clean], axis=1)
    dataset.to_csv(output_path, index=False)

    print(f"\n✓ Dataset saved to: {output_path}")

    # Save metadata (feature names, encoders, etc.)
    metadata = {
        'num_samples': len(dataset),
        'num_features': features_clean.shape[1],
        'num_targets': targets_clean.shape[1],
        'feature_columns': list(features_clean.columns),
        'target_columns': list(targets_clean.columns),
        'categorical_encoders': encoders,
        'original_hyperparameter_cols': available_hyperparam_cols,
        'removed_rows': removed_rows
    }

    metadata_path = output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Metadata saved to: {metadata_path}")

    print(f"\n{'='*80}")
    print(f"Dataset saved to: {output_path}")
    print(f"{'='*80}\n")

    return features_clean, targets_clean, metadata


def main():
    parser = argparse.ArgumentParser(
        description='Prepare MLP Training Dataset from EA Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare dataset from EA run
  python3 prepare_dataset.py --results_dir ./test/results/20260112_140511

  # Save to custom location
  python3 prepare_dataset.py --results_dir ./test/results/20260112_140511 \\
      --output ./mlp/data/dataset.csv

  # Combine multiple EA runs
  python3 prepare_dataset.py --results_dir ./test/results/20260112_140511,./test/results/20260112_143240
        """
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to EA results directory (comma-separated for multiple runs)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output path for dataset CSV (default: results_dir/mlp_dataset.csv)'
    )

    args = parser.parse_args()

    # Handle multiple results directories
    results_dirs = [d.strip() for d in args.results_dir.split(',')]

    # Validate directories
    for results_dir in results_dirs:
        if not os.path.exists(results_dir):
            print(f"Error: Results directory not found: {results_dir}")
            sys.exit(1)

    try:
        if len(results_dirs) == 1:
            # Single directory
            features, targets, metadata = prepare_dataset(
                results_dir=results_dirs[0],
                output_path=args.output
            )
        else:
            # Multiple directories - combine datasets
            print(f"Combining {len(results_dirs)} EA runs...\n")

            all_features = []
            all_targets = []

            for i, results_dir in enumerate(results_dirs):
                print(f"\nProcessing run {i+1}/{len(results_dirs)}: {results_dir}")
                features, targets, _ = prepare_dataset(results_dir, output_path=None)
                all_features.append(features)
                all_targets.append(targets)

            # Combine all
            combined_features = pd.concat(all_features, ignore_index=True)
            combined_targets = pd.concat(all_targets, ignore_index=True)

            # Save combined dataset
            output_path = args.output or './mlp/data/combined_dataset.csv'
            dataset = pd.concat([combined_features, combined_targets], axis=1)
            dataset.to_csv(output_path, index=False)

            print(f"\n{'='*80}")
            print(f"COMBINED DATASET: {len(dataset)} total samples")
            print(f"Saved to: {output_path}")
            print(f"{'='*80}\n")

        print("✓ Dataset preparation completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Review the dataset")
        print(f"  2. Train the MLP:")
        print(f"     cd mlp")
        print(f"     python3 src/train.py --dataset {args.output or 'data/dataset.csv'}")

    except Exception as e:
        print(f"\n✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
