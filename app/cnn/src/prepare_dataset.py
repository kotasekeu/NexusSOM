#!/usr/bin/env python3
"""
Prepare CNN Training Dataset from EA Results

This script combines RGB maps, labels, and metrics from EA results
into a single dataset CSV file ready for CNN training.

Usage:
    python3 prepare_dataset.py --results_dir ./test/results/20260112_140511
    python3 prepare_dataset.py --results_dir ./test/results/20260112_140511 --output ./cnn/data/dataset.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path


def load_labels(results_dir):
    """Load labels from labels.csv"""
    labels_file = os.path.join(results_dir, "labels.csv")

    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")

    labels_df = pd.read_csv(labels_file)
    print(f"Loaded {len(labels_df)} labels")

    # Convert labels to binary: good=1, bad=0, bad_auto=0
    labels_df['quality_score'] = labels_df['label'].apply(
        lambda x: 1.0 if x == 'good' else 0.0
    )

    return labels_df


def load_results(results_dir):
    """Load results from results.csv"""
    results_file = os.path.join(results_dir, "results.csv")

    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    results_df = pd.read_csv(results_file)
    print(f"Loaded {len(results_df)} results")

    return results_df


def find_rgb_images(results_dir):
    """Find all RGB images in maps_dataset/rgb directory"""
    rgb_dir = os.path.join(results_dir, "maps_dataset", "rgb")

    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"RGB directory not found: {rgb_dir}")

    rgb_images = []
    for filename in os.listdir(rgb_dir):
        if filename.endswith('_rgb.png'):
            uid = filename.replace('_rgb.png', '')
            filepath = os.path.join(rgb_dir, filename)
            rgb_images.append({'uid': uid, 'filepath': filepath})

    rgb_df = pd.DataFrame(rgb_images)
    print(f"Found {len(rgb_df)} RGB images")

    return rgb_df


def create_dataset(results_dir, output_path=None, include_metrics=True):
    """
    Create dataset CSV file by combining labels, RGB images, and metrics.

    Args:
        results_dir: Path to EA results directory
        output_path: Path to save dataset CSV (default: results_dir/dataset.csv)
        include_metrics: Whether to include additional metrics from results.csv

    Returns:
        DataFrame with the complete dataset
    """
    print(f"\n{'='*80}")
    print("PREPARING CNN TRAINING DATASET")
    print(f"{'='*80}\n")

    # Load components
    print("Loading data...")
    labels_df = load_labels(results_dir)
    rgb_df = find_rgb_images(results_dir)
    results_df = load_results(results_dir)

    # Merge RGB images with labels
    print("\nMerging data...")
    dataset = rgb_df.merge(labels_df[['uid', 'quality_score', 'label']], on='uid', how='inner')
    print(f"After merging with labels: {len(dataset)} samples")

    # Add metrics from results.csv if requested
    if include_metrics:
        metric_cols = ['uid', 'best_mqe', 'topographic_error', 'dead_neuron_ratio',
                       'u_matrix_mean', 'u_matrix_std']

        # Add new columns if they exist
        if 'u_matrix_max' in results_df.columns:
            metric_cols.append('u_matrix_max')
        if 'distance_map_max' in results_df.columns:
            metric_cols.append('distance_map_max')

        # Filter to only include columns that exist
        available_cols = [col for col in metric_cols if col in results_df.columns]

        dataset = dataset.merge(results_df[available_cols], on='uid', how='left')
        print(f"Added {len(available_cols)-1} metric columns")

    # Print label distribution
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}\n")

    print(f"Total samples: {len(dataset)}")
    print(f"\nLabel distribution:")
    label_counts = dataset['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(dataset)) * 100
        print(f"  {label}: {count} ({percentage:.1f}%)")

    print(f"\nQuality score distribution:")
    print(f"  Good (1.0): {(dataset['quality_score'] == 1.0).sum()}")
    print(f"  Bad (0.0): {(dataset['quality_score'] == 0.0).sum()}")

    if include_metrics:
        print(f"\nMetric statistics:")
        metric_cols_to_show = ['best_mqe', 'topographic_error', 'dead_neuron_ratio']
        if 'u_matrix_max' in dataset.columns:
            metric_cols_to_show.append('u_matrix_max')
        if 'distance_map_max' in dataset.columns:
            metric_cols_to_show.append('distance_map_max')

        for col in metric_cols_to_show:
            if col in dataset.columns:
                print(f"  {col}:")
                print(f"    min: {dataset[col].min():.6f}")
                print(f"    max: {dataset[col].max():.6f}")
                print(f"    mean: {dataset[col].mean():.6f}")

    # Save dataset
    if output_path is None:
        output_path = os.path.join(results_dir, "dataset.csv")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset.to_csv(output_path, index=False)
    print(f"\n{'='*80}")
    print(f"Dataset saved to: {output_path}")
    print(f"{'='*80}\n")

    # Check for class imbalance
    good_ratio = (dataset['quality_score'] == 1.0).sum() / len(dataset)
    if good_ratio < 0.1 or good_ratio > 0.9:
        print("⚠️  WARNING: Severe class imbalance detected!")
        print(f"   Good samples: {good_ratio*100:.1f}%")
        print("   Consider:")
        print("   1. Generating more good quality maps with EA")
        print("   2. Using class weights in training")
        print("   3. Using data augmentation for minority class")
        print()

    return dataset


def main():
    parser = argparse.ArgumentParser(
        description='Prepare CNN Training Dataset from EA Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare dataset from EA run
  python3 prepare_dataset.py --results_dir ./test/results/20260112_140511

  # Save to custom location
  python3 prepare_dataset.py --results_dir ./test/results/20260112_140511 \\
      --output ./cnn/data/processed/dataset.csv

  # Without metrics (only images and labels)
  python3 prepare_dataset.py --results_dir ./test/results/20260112_140511 \\
      --no-metrics
        """
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to EA results directory (e.g., ./test/results/20260112_140511)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output path for dataset CSV (default: results_dir/dataset.csv)'
    )

    parser.add_argument(
        '--no-metrics',
        action='store_true',
        help='Do not include metrics from results.csv (only images and labels)'
    )

    args = parser.parse_args()

    # Validate results directory
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    try:
        dataset = create_dataset(
            results_dir=args.results_dir,
            output_path=args.output,
            include_metrics=not args.no_metrics
        )

        print("✓ Dataset preparation completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Review the dataset at: {args.output or os.path.join(args.results_dir, 'dataset.csv')}")
        print(f"  2. Train the CNN:")
        print(f"     cd cnn")
        print(f"     python3 src/train.py --dataset {args.output or os.path.join(args.results_dir, 'dataset.csv')}")

    except Exception as e:
        print(f"\nError creating dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
