#!/usr/bin/env python3
"""
Evaluate Trained MLP Model

This script evaluates a trained MLP model on test data or predicts quality
for new hyperparameter configurations.

Usage:
    # Evaluate on test set
    python3 evaluate_model.py --model models/mlp_prophet_standard_20260112_220000_best.keras \\
        --scaler models/mlp_prophet_standard_20260112_220000_scaler.pkl \\
        --dataset data/dataset.csv

    # Predict for new configuration
    python3 evaluate_model.py --model models/mlp_prophet_standard_20260112_220000_best.keras \\
        --scaler models/mlp_prophet_standard_20260112_220000_scaler.pkl \\
        --config path/to/config.json
"""

import argparse
import os
import sys
import json
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
import joblib


def load_model_and_scaler(model_path, scaler_path):
    """Load trained model and scaler"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("✓ Model loaded successfully")

    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)
    print("✓ Scaler loaded successfully")

    return model, scaler


def predict_from_config(model, scaler, config_path, feature_columns):
    """
    Predict quality metrics from a configuration file.

    Args:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        config_path: Path to configuration JSON file
        feature_columns: List of feature column names (in correct order)

    Returns:
        Predicted quality metrics
    """
    print(f"\nLoading configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # TODO: Encode configuration into feature vector matching training format
    # This needs to match the encoding done in prepare_dataset.py
    print("\n⚠ Configuration prediction not yet implemented")
    print("This requires encoding the config JSON to match the training feature format")

    return None


def evaluate_on_dataset(model, scaler, dataset_path):
    """
    Evaluate model on a dataset.

    Args:
        model: Trained Keras model
        scaler: Fitted StandardScaler
        dataset_path: Path to dataset CSV

    Returns:
        Evaluation metrics
    """
    print(f"\nLoading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Dataset size: {len(df)} samples")

    # Load metadata to get feature and target columns
    metadata_path = dataset_path.replace('.csv', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        feature_cols = metadata['feature_columns']
        target_cols = metadata['target_columns']
    else:
        # Fallback
        target_cols = ['best_mqe', 'topographic_error', 'dead_neuron_ratio']
        feature_cols = [col for col in df.columns if col not in target_cols]

    # Extract features and targets
    X = df[feature_cols].values
    y = df[target_cols].values

    # Scale features
    X_scaled = scaler.transform(X)

    # Predict
    print("\nMaking predictions...")
    y_pred = model.predict(X_scaled, verbose=1)

    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y), axis=0)
    mse = np.mean((y_pred - y) ** 2, axis=0)
    rmse = np.sqrt(mse)

    # Overall metrics
    overall_mae = np.mean(mae)
    overall_rmse = np.mean(rmse)

    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")

    print("Overall Metrics:")
    print(f"  Mean Absolute Error (MAE): {overall_mae:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {overall_rmse:.6f}")

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


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Trained MLP Model',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.keras)'
    )

    parser.add_argument(
        '--scaler',
        type=str,
        required=True,
        help='Path to scaler file (.pkl)'
    )

    parser.add_argument(
        '--dataset',
        type=str,
        help='Path to dataset CSV file (for evaluation)'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration JSON file (for prediction)'
    )

    args = parser.parse_args()

    # Load model and scaler
    model, scaler = load_model_and_scaler(args.model, args.scaler)

    # Evaluate or predict
    if args.dataset:
        metrics = evaluate_on_dataset(model, scaler, args.dataset)
    elif args.config:
        # Load metadata to get feature columns
        model_name = os.path.basename(args.model).replace('_best.keras', '').replace('_final.keras', '')
        metadata_path = os.path.join('models', f'{model_name}_metadata.json')

        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            feature_cols = metadata['feature_columns']
        else:
            print(f"⚠ Warning: Metadata file not found: {metadata_path}")
            feature_cols = None

        prediction = predict_from_config(model, scaler, args.config, feature_cols)
    else:
        print("\nError: Please specify either --dataset or --config")
        sys.exit(1)


if __name__ == "__main__":
    main()
