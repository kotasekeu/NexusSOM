"""
Data Preparation Script for SOM Quality Analysis

This script processes the results.csv file containing SOM metrics and generates
a dataset with quality scores for training the CNN model.

Usage:
    python src/prepare_data.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def calculate_quality_score(df):
    """
    Calculate quality scores based on normalized SOM metrics.

    Args:
        df: DataFrame with columns 'best_mqe', 'topographic_error', 'inactive_neuron_ratio'

    Returns:
        DataFrame with added 'quality_score' column
    """
    # Create a copy to avoid modifying the original
    data = df.copy()

    # Initialize scaler for normalization to [0, 1] range
    scaler = MinMaxScaler()

    # Normalize each metric to 0-1 range
    # Lower values are better for these metrics, so we'll invert them later
    metrics_to_normalize = ['best_mqe', 'topographic_error', 'inactive_neuron_ratio']

    # Check if all required columns exist
    missing_cols = [col for col in metrics_to_normalize if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Handle missing values by filling with the median
    for col in metrics_to_normalize:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col].fillna(median_val, inplace=True)
            print(f"Warning: Filled {data[col].isnull().sum()} missing values in {col} with median: {median_val}")

    # Normalize metrics
    data[['norm_mqe', 'norm_te', 'norm_inactive']] = scaler.fit_transform(
        data[metrics_to_normalize]
    )

    # Calculate quality score
    # Formula: weighted combination of inverted normalized metrics
    # Better maps have lower error values, so we use (1 - normalized_value)
    # Weights: 50% MQE, 30% Topographic Error, 20% Inactive Neuron Ratio
    data['quality_score'] = (
        0.5 * (1 - data['norm_mqe']) +
        0.3 * (1 - data['norm_te']) +
        0.2 * (1 - data['norm_inactive'])
    )

    # Ensure scores are in [0, 1] range
    data['quality_score'] = data['quality_score'].clip(0, 1)

    return data


def verify_image_files(df, image_dir):
    """
    Verify that image files exist for all UIDs in the dataset.

    Args:
        df: DataFrame with 'uid' column
        image_dir: Path to directory containing image files

    Returns:
        DataFrame with only rows that have corresponding image files
    """
    image_dir = Path(image_dir)
    valid_rows = []
    missing_count = 0

    for idx, row in df.iterrows():
        uid = row['uid']
        image_path = image_dir / f"{uid}.png"

        if image_path.exists():
            valid_rows.append(row)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} image files not found and will be excluded")

    return pd.DataFrame(valid_rows)


def prepare_dataset(results_csv_path, raw_maps_dir, output_csv_path):
    """
    Main function to prepare the dataset for training.

    Args:
        results_csv_path: Path to the results.csv file
        raw_maps_dir: Directory containing the SOM map images
        output_csv_path: Path where the prepared dataset.csv will be saved
    """
    print("=" * 60)
    print("SOM Quality Dataset Preparation")
    print("=" * 60)

    # Load results CSV
    print(f"\n1. Loading results from: {results_csv_path}")
    if not os.path.exists(results_csv_path):
        raise FileNotFoundError(f"Results file not found: {results_csv_path}")

    df = pd.read_csv(results_csv_path)
    print(f"   Loaded {len(df)} records")
    print(f"   Columns: {list(df.columns)}")

    # Verify required columns
    required_cols = ['uid', 'best_mqe', 'topographic_error', 'inactive_neuron_ratio']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in results.csv: {missing}")

    # Calculate quality scores
    print("\n2. Calculating quality scores...")
    df = calculate_quality_score(df)
    print(f"   Quality score range: [{df['quality_score'].min():.4f}, {df['quality_score'].max():.4f}]")
    print(f"   Quality score mean: {df['quality_score'].mean():.4f}")
    print(f"   Quality score std: {df['quality_score'].std():.4f}")

    # Verify image files exist
    print(f"\n3. Verifying image files in: {raw_maps_dir}")
    df = verify_image_files(df, raw_maps_dir)
    print(f"   Valid records with images: {len(df)}")

    if len(df) == 0:
        raise ValueError("No valid records found with corresponding image files!")

    # Create final dataset with filepath and quality_score
    print("\n4. Creating final dataset...")
    dataset = pd.DataFrame({
        'filepath': df['uid'].apply(lambda x: f"data/raw_maps/{x}.png"),
        'quality_score': df['quality_score']
    })

    # Save dataset
    output_dir = os.path.dirname(output_csv_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset.to_csv(output_csv_path, index=False)
    print(f"   Saved dataset to: {output_csv_path}")
    print(f"   Total samples: {len(dataset)}")

    # Display statistics
    print("\n5. Dataset Statistics:")
    print(f"   {'Metric':<25} {'Value':<15}")
    print(f"   {'-'*40}")
    print(f"   {'Total samples':<25} {len(dataset):<15}")
    print(f"   {'Min quality score':<25} {dataset['quality_score'].min():<15.4f}")
    print(f"   {'Max quality score':<25} {dataset['quality_score'].max():<15.4f}")
    print(f"   {'Mean quality score':<25} {dataset['quality_score'].mean():<15.4f}")
    print(f"   {'Median quality score':<25} {dataset['quality_score'].median():<15.4f}")
    print(f"   {'Std quality score':<25} {dataset['quality_score'].std():<15.4f}")

    # Display sample rows
    print("\n6. Sample rows from dataset:")
    print(dataset.head(10).to_string(index=False))

    print("\n" + "=" * 60)
    print("Dataset preparation completed successfully!")
    print("=" * 60)

    return dataset


if __name__ == "__main__":
    # Define paths
    RESULTS_CSV = "data/results.csv"
    RAW_MAPS_DIR = "data/raw_maps"
    OUTPUT_CSV = "data/processed/dataset.csv"

    # Prepare dataset
    try:
        dataset = prepare_dataset(RESULTS_CSV, RAW_MAPS_DIR, OUTPUT_CSV)
        print(f"\nDataset ready for training!")
        print(f"Next step: Run 'python src/train.py' to train the model")
    except Exception as e:
        print(f"\nError: {str(e)}")
        raise
