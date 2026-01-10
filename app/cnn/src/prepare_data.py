"""
Data Preparation Script for SOM Quality Analysis

This script processes EA run directories and generates a dataset with quality scores
for training the CNN model on RGB SOM map images.

Usage:
    python src/prepare_data.py --runs-dir ../test/results
    python src/prepare_data.py --runs-dir ../test/results --output data/processed/dataset.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import argparse


def calculate_quality_score(df):
    """
    Calculate quality scores based on normalized SOM metrics.

    Args:
        df: DataFrame with columns 'best_mqe', 'topographic_error', 'dead_neuron_ratio'

    Returns:
        DataFrame with added 'quality_score' column
    """
    # Create a copy to avoid modifying the original
    data = df.copy()

    # Initialize scaler for normalization to [0, 1] range
    scaler = MinMaxScaler()

    # Normalize each metric to 0-1 range
    # Lower values are better for these metrics, so we'll invert them later
    metrics_to_normalize = ['best_mqe', 'topographic_error', 'dead_neuron_ratio']

    # Check if all required columns exist
    missing_cols = [col for col in metrics_to_normalize if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Handle missing values by filling with the median
    for col in metrics_to_normalize:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
            print(f"   Warning: Filled {data[col].isnull().sum()} missing values in {col} with median: {median_val}")

    # Normalize metrics
    data[['norm_mqe', 'norm_te', 'norm_dead']] = scaler.fit_transform(
        data[metrics_to_normalize]
    )

    # Calculate quality score
    # Formula: weighted combination of inverted normalized metrics
    # Better maps have lower error values, so we use (1 - normalized_value)
    # Weights: 50% MQE, 30% Topographic Error, 20% Dead Neuron Ratio
    data['quality_score'] = (
        0.5 * (1 - data['norm_mqe']) +
        0.3 * (1 - data['norm_te']) +
        0.2 * (1 - data['norm_dead'])
    )

    # Ensure scores are in [0, 1] range
    data['quality_score'] = data['quality_score'].clip(0, 1)

    return data


def collect_ea_runs(runs_dir):
    """
    Scan EA runs directory and collect all results.csv files with their RGB maps.

    Args:
        runs_dir: Path to directory containing EA run subdirectories

    Returns:
        List of tuples (run_dir, results_csv_path, rgb_maps_dir)
    """
    runs_dir = Path(runs_dir)
    ea_runs = []

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    # Iterate through subdirectories (each is an EA run)
    for run_subdir in sorted(runs_dir.iterdir()):
        if not run_subdir.is_dir():
            continue

        # Check for results.csv
        results_csv = run_subdir / "results.csv"
        if not results_csv.exists():
            continue

        # Check for RGB maps directory
        rgb_maps_dir = run_subdir / "maps_dataset" / "rgb"
        if not rgb_maps_dir.exists():
            print(f"   Warning: RGB maps not found for run {run_subdir.name}, skipping...")
            continue

        ea_runs.append((str(run_subdir), str(results_csv), str(rgb_maps_dir)))

    return ea_runs


def prepare_dataset_from_runs(runs_dir, output_csv_path):
    """
    Main function to prepare the dataset from EA runs.

    Args:
        runs_dir: Path to directory containing EA run subdirectories
        output_csv_path: Path where the prepared dataset.csv will be saved
    """
    print("=" * 60)
    print("SOM Quality Dataset Preparation (RGB Multi-Channel)")
    print("=" * 60)

    # Collect EA runs
    print(f"\n1. Scanning for EA runs in: {runs_dir}")
    ea_runs = collect_ea_runs(runs_dir)
    print(f"   Found {len(ea_runs)} valid EA runs with RGB maps")

    if len(ea_runs) == 0:
        raise ValueError("No valid EA runs found! Make sure each run has results.csv and maps_dataset/rgb/")

    # Load and combine all results
    print("\n2. Loading results from all runs...")
    all_data = []
    total_samples = 0

    for run_dir, results_csv, rgb_maps_dir in ea_runs:
        run_name = Path(run_dir).name
        df = pd.read_csv(results_csv)

        # Add run_dir and rgb_dir to each row
        df['run_dir'] = run_dir
        df['rgb_dir'] = rgb_maps_dir

        print(f"   - Run {run_name}: {len(df)} samples")
        total_samples += len(df)
        all_data.append(df)

    # Combine all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"   Total samples across all runs: {total_samples}")
    print(f"   Columns: {list(combined_df.columns[:10])}... (showing first 10)")

    # Verify required columns
    required_cols = ['uid', 'best_mqe', 'topographic_error', 'dead_neuron_ratio']
    missing = [col for col in required_cols if col not in combined_df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results.csv: {missing}")

    # Calculate quality scores
    print("\n3. Calculating quality scores...")
    combined_df = calculate_quality_score(combined_df)
    print(f"   Quality score range: [{combined_df['quality_score'].min():.4f}, {combined_df['quality_score'].max():.4f}]")
    print(f"   Quality score mean: {combined_df['quality_score'].mean():.4f}")
    print(f"   Quality score std: {combined_df['quality_score'].std():.4f}")

    # Verify RGB image files exist and create filepath
    print("\n4. Verifying RGB image files...")
    valid_rows = []
    missing_count = 0

    for idx, row in combined_df.iterrows():
        uid = row['uid']
        rgb_dir = Path(row['rgb_dir'])
        rgb_image_path = rgb_dir / f"{uid}_rgb.png"

        if rgb_image_path.exists():
            row['filepath'] = str(rgb_image_path)
            valid_rows.append(row)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"   Warning: {missing_count} RGB images not found and will be excluded")

    valid_df = pd.DataFrame(valid_rows)
    print(f"   Valid records with RGB images: {len(valid_df)}")

    if len(valid_df) == 0:
        raise ValueError("No valid records found with corresponding RGB image files!")

    # Create final dataset
    print("\n5. Creating final dataset...")
    dataset = pd.DataFrame({
        'filepath': valid_df['filepath'],
        'quality_score': valid_df['quality_score'],
        'uid': valid_df['uid'],
        'run_dir': valid_df['run_dir']
    })

    # Save dataset
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset.to_csv(output_csv_path, index=False)
    print(f"   Saved dataset to: {output_csv_path}")
    print(f"   Total samples: {len(dataset)}")

    # Display statistics
    print("\n6. Dataset Statistics:")
    print(f"   {'Metric':<30} {'Value':<15}")
    print(f"   {'-'*45}")
    print(f"   {'Total samples':<30} {len(dataset):<15}")
    print(f"   {'Unique EA runs':<30} {dataset['run_dir'].nunique():<15}")
    print(f"   {'Min quality score':<30} {dataset['quality_score'].min():<15.4f}")
    print(f"   {'Max quality score':<30} {dataset['quality_score'].max():<15.4f}")
    print(f"   {'Mean quality score':<30} {dataset['quality_score'].mean():<15.4f}")
    print(f"   {'Median quality score':<30} {dataset['quality_score'].median():<15.4f}")
    print(f"   {'Std quality score':<30} {dataset['quality_score'].std():<15.4f}")

    # Display sample rows
    print("\n7. Sample rows from dataset:")
    print(dataset.head(5).to_string(index=False))

    print("\n" + "=" * 60)
    print("Dataset preparation completed successfully!")
    print("=" * 60)

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare CNN dataset from EA runs')
    parser.add_argument('--runs-dir', type=str, default='../test/results',
                        help='Directory containing EA run subdirectories')
    parser.add_argument('--output', type=str, default='data/processed/dataset.csv',
                        help='Output path for the prepared dataset CSV')
    args = parser.parse_args()

    # Prepare dataset
    try:
        dataset = prepare_dataset_from_runs(args.runs_dir, args.output)
        print(f"\nDataset ready for training!")
        print(f"Next step: Run 'python src/train.py' to train the model")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
