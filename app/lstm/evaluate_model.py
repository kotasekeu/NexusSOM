#!/usr/bin/env python3
"""
Evaluate Trained LSTM Model

This script evaluates a trained LSTM model on test sequences or makes
early predictions during SOM training.

Usage:
    # Evaluate on test set
    python3 evaluate_model.py --model models/lstm_oracle_standard_20260112_220000_best.keras \\
        --dataset data/dataset.csv

    # Predict from partial sequence (early stopping)
    python3 evaluate_model.py --model models/lstm_oracle_standard_20260112_220000_best.keras \\
        --sequence path/to/partial_sequence.json
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras


def load_model(model_path):
    """Load trained LSTM model"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")
    return model


def evaluate_on_dataset(model, dataset_path):
    """
    Evaluate model on dataset.

    Args:
        model: Trained Keras model
        dataset_path: Path to dataset CSV

    Returns:
        Evaluation metrics
    """
    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset size: {len(df)} sequences")

    # Parse sequences
    sequences = []
    targets = []

    for idx, row in df.iterrows():
        history = json.loads(row['training_history'])

        sequence = np.array([
            history['mqe'],
            history['topographic_error'],
            history['dead_neuron_ratio']
        ]).T

        sequences.append(sequence)

        target = np.array([
            row['final_mqe'],
            row['final_topographic_error'],
            row['final_dead_neuron_ratio']
        ])
        targets.append(target)

    X = np.array(sequences)
    y = np.array(targets)

    # Predict
    print("\nMaking predictions...")
    y_pred = model.predict(X, verbose=1)

    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y), axis=0)
    mse = np.mean((y_pred - y) ** 2, axis=0)
    rmse = np.sqrt(mse)

    overall_mae = np.mean(mae)
    overall_rmse = np.mean(rmse)

    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")

    print("Overall Metrics:")
    print(f"  Mean Absolute Error (MAE): {overall_mae:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {overall_rmse:.6f}")

    target_cols = ['mqe', 'topographic_error', 'dead_neuron_ratio']
    print(f"\nPer-Target Metrics:")
    for i, target in enumerate(target_cols):
        print(f"\n  {target}:")
        print(f"    MAE:  {mae[i]:.6f}")
        print(f"    RMSE: {rmse[i]:.6f}")

    # Sample predictions
    print(f"\n{'='*80}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*80}\n")

    num_samples = min(10, len(y))
    print(f"{'Target':<25} | {'Actual':<15} | {'Predicted':<15} | {'Error':<15}")
    print("-" * 80)

    for i in range(num_samples):
        for j, target in enumerate(target_cols):
            actual = y[i, j]
            predicted = y_pred[i, j]
            error = abs(actual - predicted)
            print(f"{target:<25} | {actual:<15.6f} | {predicted:<15.6f} | {error:<15.6f}")
        if i < num_samples - 1:
            print("-" * 80)

    return {
        'overall_mae': overall_mae,
        'overall_rmse': overall_rmse,
        'per_target_mae': dict(zip(target_cols, mae)),
        'per_target_rmse': dict(zip(target_cols, rmse))
    }


def predict_from_partial_sequence(model, sequence_path):
    """
    Predict final quality from partial training sequence.

    Args:
        model: Trained Keras model
        sequence_path: Path to JSON file with partial training history

    Returns:
        Predicted final quality metrics
    """
    print(f"\nLoading partial sequence from: {sequence_path}")
    with open(sequence_path, 'r') as f:
        history = json.load(f)

    # Convert to numpy array
    sequence = np.array([
        history['mqe'],
        history['topographic_error'],
        history['dead_neuron_ratio']
    ]).T

    # Expand dims for batch
    sequence_batch = np.expand_dims(sequence, axis=0)

    print(f"Sequence shape: {sequence.shape}")
    print(f"Checkpoints available: {len(history['mqe'])}")

    # Predict
    print("\nMaking prediction...")
    prediction = model.predict(sequence_batch, verbose=0)[0]

    # Print results
    print(f"\n{'='*80}")
    print("EARLY PREDICTION")
    print(f"{'='*80}\n")

    target_cols = ['mqe', 'topographic_error', 'dead_neuron_ratio']
    print("Predicted final quality metrics:")
    for target, value in zip(target_cols, prediction):
        print(f"  {target}: {value:.6f}")

    print(f"\n{'='*80}")
    print("EARLY STOPPING RECOMMENDATION")
    print(f"{'='*80}\n")

    # Quality score (lower is better)
    quality_score = prediction[0] * 1.0 + prediction[1] * 1.0 + prediction[2] * 0.5

    if quality_score > 1.0:
        print("⚠ RECOMMENDATION: TERMINATE TRAINING")
        print(f"  Predicted quality score: {quality_score:.4f} (threshold: 1.0)")
        print("  This configuration is unlikely to produce a good SOM.")
    elif quality_score > 0.5:
        print("⚙ RECOMMENDATION: CONTINUE WITH CAUTION")
        print(f"  Predicted quality score: {quality_score:.4f}")
        print("  Quality may be marginal. Monitor progress closely.")
    else:
        print("✓ RECOMMENDATION: CONTINUE TRAINING")
        print(f"  Predicted quality score: {quality_score:.4f}")
        print("  This configuration shows promise.")

    return prediction


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Trained LSTM Model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.keras)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset CSV file (for evaluation)'
    )

    parser.add_argument(
        '--sequence',
        type=str,
        help='Path to partial training sequence JSON (for early prediction)'
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)

    # Evaluate or predict
    if args.dataset:
        metrics = evaluate_on_dataset(model, args.dataset)
    elif args.sequence:
        prediction = predict_from_partial_sequence(model, args.sequence)
    else:
        print("\nError: Please specify either --dataset or --sequence")
        sys.exit(1)


if __name__ == "__main__":
    main()
