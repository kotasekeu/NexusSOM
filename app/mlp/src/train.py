"""
Training Script for MLP Hyperparameter Quality Predictor - "The Prophet"

This script trains an MLP to predict SOM quality metrics from hyperparameters.

Usage:
    python src/train.py --dataset data/dataset.csv
    python src/train.py --dataset data/dataset.csv --epochs 200 --batch-size 64
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json

# Import model definitions
from model import create_mlp_model, create_lightweight_mlp, print_model_summary


def load_dataset(dataset_path, test_size=0.15, val_size=0.15, random_state=42):
    """
    Load and split dataset into train, validation, and test sets.

    Args:
        dataset_path: Path to the dataset CSV file
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, target_cols, scaler)
    """
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Total samples: {len(df)}")

    # Load metadata to get feature and target columns
    metadata_path = dataset_path.replace('.csv', '_metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        feature_cols = metadata['feature_columns']
        target_cols = metadata['target_columns']
        print(f"✓ Loaded metadata: {len(feature_cols)} features, {len(target_cols)} targets")
    else:
        # Fallback: assume last 3 columns are targets
        target_cols = ['best_mqe', 'topographic_error', 'dead_neuron_ratio']
        feature_cols = [col for col in df.columns if col not in target_cols]
        print(f"⚠ No metadata found, using default target columns: {target_cols}")

    # Extract features and targets
    X = df[feature_cols].values
    y = df[target_cols].values

    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")

    return (X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test,
            feature_cols, target_cols, scaler)


def create_callbacks(model_name, log_dir):
    """
    Create training callbacks.

    Args:
        model_name: Name for saving the model
        log_dir: Directory for logs

    Returns:
        List of Keras callbacks
    """
    os.makedirs('models', exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        # Save best model based on validation loss
        ModelCheckpoint(
            filepath=f'models/{model_name}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            mode='min',
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            mode='min',
            verbose=1
        ),
        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        # CSV logger for training history
        CSVLogger(
            filename=os.path.join(log_dir, 'training_log.csv'),
            separator=',',
            append=False
        )
    ]

    return callbacks


def train_model(dataset_path, model_type='standard', epochs=100, batch_size=32, learning_rate=0.001):
    """
    Main training function.

    Args:
        dataset_path: Path to the dataset CSV file
        model_type: Type of model ('standard' or 'lite')
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    """
    print("=" * 80)
    print("MLP HYPERPARAMETER QUALITY PREDICTOR - TRAINING")
    print("=" * 80)
    print(f"Model Type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 80)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load and split dataset
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_cols, target_cols, scaler) = load_dataset(dataset_path)

    # Create model
    print("\nCreating model...")
    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    if model_type == 'lite':
        model = create_lightweight_mlp(input_dim=input_dim, output_dim=output_dim, learning_rate=learning_rate)
    else:
        model = create_mlp_model(input_dim=input_dim, output_dim=output_dim, learning_rate=learning_rate)

    print_model_summary(model)

    # Create callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"mlp_prophet_{model_type}_{timestamp}"
    log_dir = f"logs/{model_name}"
    callbacks = create_callbacks(model_name, log_dir)

    # Train model
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    test_results = model.evaluate(X_test, y_test, verbose=1)

    print("\nTest Results:")
    for metric_name, metric_value in zip(model.metrics_names, test_results):
        print(f"  {metric_name}: {metric_value:.6f}")

    # Make predictions on test set and show sample results
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80 + "\n")

    y_pred = model.predict(X_test[:10], verbose=0)

    print(f"{'Actual':<40} | {'Predicted':<40}")
    print("-" * 80)
    for i in range(min(10, len(y_test))):
        actual = f"[{', '.join([f'{v:.6f}' for v in y_test[i]])}]"
        predicted = f"[{', '.join([f'{v:.6f}' for v in y_pred[i]])}]"
        print(f"{actual:<40} | {predicted:<40}")

    # Save final model
    final_model_path = f'models/{model_name}_final.keras'
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Save scaler
    import joblib
    scaler_path = f'models/{model_name}_scaler.pkl'
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")

    # Save model metadata
    metadata = {
        'model_name': model_name,
        'model_type': model_type,
        'timestamp': timestamp,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'feature_columns': feature_cols,
        'target_columns': target_cols,
        'num_train_samples': len(X_train),
        'num_val_samples': len(X_val),
        'num_test_samples': len(X_test),
        'test_loss': float(test_results[0]),
        'test_mae': float(test_results[1]),
        'test_mse': float(test_results[2])
    }

    metadata_path = f'models/{model_name}_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nBest model: models/{model_name}_best.keras")
    print(f"Training logs: {log_dir}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir={log_dir}")

    return model, history, metadata


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train MLP Hyperparameter Quality Predictor')

    parser.add_argument(
        '--dataset',
        type=str,
        default='data/dataset.csv',
        help='Path to dataset CSV file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='standard',
        choices=['standard', 'lite'],
        help='Model architecture to use'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for training'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Train model
    try:
        model, history, metadata = train_model(
            dataset_path=args.dataset,
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
