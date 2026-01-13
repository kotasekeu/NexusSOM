#!/usr/bin/env python3
"""
Collect LSTM Training Data - Time-Series SOM Training Progress

This script extracts time-series training metrics from SOM training logs
to prepare data for LSTM training.

NOTE: This requires SOM training code to log intermediate metrics.
For proof-of-concept, we'll simulate this data from results.csv

Usage:
    python3 collect_training_data.py --results_dir ./test/results/20260112_140511
    python3 collect_training_data.py --results_dir ./test/results/20260112_140511 --output ./lstm/data/dataset.csv
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import json


def simulate_training_history(final_mqe, final_topo_error, final_dead_ratio, num_checkpoints=10):
    """
    Simulate time-series training data for proof-of-concept.

    In production, this would be replaced with actual logged data during SOM training.

    Args:
        final_mqe: Final MQE value
        final_topo_error: Final topographic error
        final_dead_ratio: Final dead neuron ratio
        num_checkpoints: Number of training checkpoints to simulate

    Returns:
        Dictionary with time-series data
    """
    # Simulate exponential decay to final values
    epochs = np.linspace(10, 1000, num_checkpoints)

    # MQE typically starts high and decreases
    mqe_start = final_mqe * np.random.uniform(3.0, 5.0)
    mqe_history = mqe_start * np.exp(-np.linspace(0, 3, num_checkpoints)) + final_mqe

    # Topographic error decreases but can fluctuate
    topo_start = min(0.5, final_topo_error * np.random.uniform(2.0, 4.0))
    topo_history = topo_start * np.exp(-np.linspace(0, 2, num_checkpoints)) + final_topo_error
    topo_history += np.random.normal(0, final_topo_error * 0.1, num_checkpoints)  # Add noise
    topo_history = np.clip(topo_history, 0, 1)

    # Dead neuron ratio can increase or decrease
    dead_start = np.random.uniform(0.1, 0.3)
    dead_history = dead_start + (final_dead_ratio - dead_start) * (epochs / epochs[-1])
    dead_history += np.random.normal(0, 0.02, num_checkpoints)  # Add noise
    dead_history = np.clip(dead_history, 0, 1)

    return {
        'epochs': epochs.tolist(),
        'mqe': mqe_history.tolist(),
        'topographic_error': topo_history.tolist(),
        'dead_neuron_ratio': dead_history.tolist(),
        'num_checkpoints': num_checkpoints
    }


def collect_training_sequences(results_dir, num_checkpoints=10):
    """
    Collect training sequences from EA results.

    Args:
        results_dir: Path to EA results directory
        num_checkpoints: Number of checkpoints per training sequence

    Returns:
        DataFrame with training sequences
    """
    print(f"\n{'='*80}")
    print("COLLECTING LSTM TRAINING DATA")
    print(f"{'='*80}\n")
    print("NOTE: Using simulated training histories for proof-of-concept")
    print("In production, replace with actual logged metrics from SOM training\n")

    # Load results.csv
    results_file = os.path.join(results_dir, "results.csv")
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    print(f"Loading results from: {results_file}")
    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} configurations")

    # Generate training sequences for each configuration
    sequences = []

    for idx, row in df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Progress: {idx+1}/{len(df)}")

        # Simulate training history
        history = simulate_training_history(
            final_mqe=row['best_mqe'],
            final_topo_error=row['topographic_error'],
            final_dead_ratio=row['dead_neuron_ratio'],
            num_checkpoints=num_checkpoints
        )

        # Create sequence entry
        sequence = {
            'uid': row['uid'],
            'final_mqe': row['best_mqe'],
            'final_topographic_error': row['topographic_error'],
            'final_dead_neuron_ratio': row['dead_neuron_ratio'],
            'training_history': json.dumps(history)
        }

        sequences.append(sequence)

    sequences_df = pd.DataFrame(sequences)

    print(f"\n✓ Generated {len(sequences_df)} training sequences")
    print(f"  Checkpoints per sequence: {num_checkpoints}")

    return sequences_df


def prepare_lstm_dataset(results_dir, output_path=None, num_checkpoints=10):
    """
    Prepare LSTM training dataset.

    Args:
        results_dir: Path to EA results directory
        output_path: Output path for dataset CSV
        num_checkpoints: Number of checkpoints per sequence

    Returns:
        DataFrame with sequences
    """
    # Collect sequences
    sequences_df = collect_training_sequences(results_dir, num_checkpoints)

    # Print statistics
    print(f"\n{'='*80}")
    print("DATASET STATISTICS")
    print(f"{'='*80}\n")

    print(f"Total sequences: {len(sequences_df)}")
    print(f"Sequence length: {num_checkpoints} checkpoints")

    print(f"\nFinal quality distribution:")
    print(f"  MQE: mean={sequences_df['final_mqe'].mean():.6f}, "
          f"std={sequences_df['final_mqe'].std():.6f}")
    print(f"  Topo Error: mean={sequences_df['final_topographic_error'].mean():.6f}, "
          f"std={sequences_df['final_topographic_error'].std():.6f}")
    print(f"  Dead Ratio: mean={sequences_df['final_dead_neuron_ratio'].mean():.6f}, "
          f"std={sequences_df['final_dead_neuron_ratio'].std():.6f}")

    # Save dataset
    if output_path is None:
        output_path = os.path.join(results_dir, "lstm_dataset.csv")

    sequences_df.to_csv(output_path, index=False)
    print(f"\n✓ Dataset saved to: {output_path}")

    # Save metadata
    metadata = {
        'num_sequences': len(sequences_df),
        'num_checkpoints': num_checkpoints,
        'features': ['mqe', 'topographic_error', 'dead_neuron_ratio'],
        'targets': ['final_mqe', 'final_topographic_error', 'final_dead_neuron_ratio'],
        'data_type': 'simulated',  # Change to 'real' when using actual logs
        'note': 'Simulated training histories for proof-of-concept. Replace with actual SOM training logs in production.'
    }

    metadata_path = output_path.replace('.csv', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_path}")

    print(f"\n{'='*80}")
    print(f"Dataset saved to: {output_path}")
    print(f"{'='*80}\n")

    return sequences_df


def main():
    parser = argparse.ArgumentParser(
        description='Collect LSTM Training Data from EA Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect training sequences from EA run
  python3 collect_training_data.py --results_dir ./test/results/20260112_140511

  # Custom output path and number of checkpoints
  python3 collect_training_data.py --results_dir ./test/results/20260112_140511 \\
      --output ./lstm/data/dataset.csv --checkpoints 20

NOTE: This proof-of-concept uses simulated training histories.
For production, modify SOM training code to log intermediate metrics at checkpoints.
        """
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to EA results directory'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output path for dataset CSV (default: results_dir/lstm_dataset.csv)'
    )

    parser.add_argument(
        '--checkpoints',
        type=int,
        default=10,
        help='Number of training checkpoints to collect (default: 10)'
    )

    args = parser.parse_args()

    # Validate directory
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    try:
        sequences_df = prepare_lstm_dataset(
            results_dir=args.results_dir,
            output_path=args.output,
            num_checkpoints=args.checkpoints
        )

        print("✓ Data collection completed successfully!")
        print(f"\nNext steps:")
        print(f"  1. Review the dataset")
        print(f"  2. Train the LSTM:")
        print(f"     cd lstm")
        print(f"     python3 src/train.py --dataset {args.output or 'data/dataset.csv'}")
        print(f"\nFor production:")
        print(f"  1. Modify SOM training code to log intermediate metrics")
        print(f"  2. Re-run EA to collect real training histories")
        print(f"  3. Replace simulated data with actual logged data")

    except Exception as e:
        print(f"\n✗ Error collecting data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
