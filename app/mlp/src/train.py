"""
MLP Training Script — SOM Hyperparameter Quality Predictor

Trains an MLP to predict SOM quality from hyperparameters without running SOM training.
Saves model and scaler to stable paths so EA config can reference them directly.

Usage:
    python src/train.py
    python src/train.py --dataset data/all_combined_mlp.csv --epochs 300
    python src/train.py --model lite --batch-size 64
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau)

sys.path.insert(0, os.path.dirname(__file__))
from model import create_mlp_model, create_lightweight_mlp, print_model_summary

STABLE_MODEL_PATH = 'models/mlp_latest.keras'
STABLE_SCALER_PATH = 'models/mlp_scaler_latest.pkl'
STABLE_META_PATH  = 'models/mlp_latest_metadata.json'


def load_dataset(dataset_path: str, test_size=0.15, val_size=0.15, random_state=42):
    print(f"Loading dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Total samples: {len(df)}")

    metadata_path = dataset_path.replace('.csv', '_metadata.json')
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)

    feature_cols = metadata['feature_columns']
    target_cols  = metadata['target_columns']
    print(f"Features: {len(feature_cols)}  |  Targets: {target_cols}")

    # dataset_name column is present for stratification but is not a feature
    stratify_col = df['dataset_name'] if 'dataset_name' in df.columns else None

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)

    # Stratified split keeps each dataset proportionally represented in all splits
    if stratify_col is not None:
        X_tv, X_test, y_tv, y_test, s_tv, _ = train_test_split(
            X, y, stratify_col, test_size=test_size, random_state=random_state,
            stratify=stratify_col)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=val_size, random_state=random_state, stratify=s_tv)
    else:
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=val_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}  |  Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols, target_cols, scaler


def train_model(dataset_path: str, model_type='standard', epochs=300,
                batch_size=32, learning_rate=0.001):
    print("=" * 70)
    print("MLP TRAINING")
    print("=" * 70)

    np.random.seed(42)
    tf.random.set_seed(42)

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     feature_cols, target_cols, scaler) = load_dataset(dataset_path)

    input_dim  = X_train.shape[1]
    output_dim = y_train.shape[1]

    if model_type == 'lite':
        model = create_lightweight_mlp(input_dim, output_dim, learning_rate)
    else:
        model = create_mlp_model(input_dim, output_dim, learning_rate)
    print_model_summary(model)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name  = f'mlp_{model_type}_{timestamp}'
    log_dir   = f'logs/{run_name}'
    os.makedirs('models', exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=f'models/{run_name}_best.keras',
            monitor='val_loss', save_best_only=True, mode='min', verbose=0),
        EarlyStopping(
            monitor='val_loss', patience=30,
            restore_best_weights=True, mode='min', verbose=1),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=15,
            min_lr=1e-7, mode='min', verbose=1),
        CSVLogger(os.path.join(log_dir, 'history.csv')),
    ]

    print(f"\nTraining ({epochs} epochs max, early stopping patience=30)...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n" + "=" * 70)
    print("TEST SET EVALUATION")
    print("=" * 70)
    test_results = model.evaluate(X_test, y_test, verbose=0)
    for name, val in zip(model.metrics_names, test_results):
        print(f"  {name}: {val:.6f}")

    print("\nSample predictions (first 8 test rows):")
    y_pred = model.predict(X_test[:8], verbose=0)
    target_names = [c.replace('raw_', '') for c in target_cols]
    header = '  '.join(f'{n:>22}' for n in target_names)
    print(f"  {'':8}  {header}")
    for i in range(len(y_pred)):
        actual    = '  '.join(f'{v:22.4f}' for v in y_test[i])
        predicted = '  '.join(f'{v:22.4f}' for v in y_pred[i])
        print(f"  actual:    {actual}")
        print(f"  predicted: {predicted}")
        print()

    # Save timestamped copy
    model.save(f'models/{run_name}_final.keras')
    joblib.dump(scaler, f'models/{run_name}_scaler.pkl')

    # Save stable "latest" paths — these are what EA config references
    model.save(STABLE_MODEL_PATH)
    joblib.dump(scaler, STABLE_SCALER_PATH)

    # Keras 3 returns ['loss', 'compile_metrics'] — extract real names from history instead
    history_keys = [k for k in history.history if not k.startswith('val_')]
    metric_names = history_keys  # e.g. ['loss', 'mae']
    meta = {
        'run_name':         run_name,
        'model_type':       model_type,
        'timestamp':        timestamp,
        'input_dim':        input_dim,
        'output_dim':       output_dim,
        'feature_columns':  feature_cols,
        'target_columns':   target_cols,
        'train_samples':    len(X_train),
        'val_samples':      len(X_val),
        'test_samples':     len(X_test),
        'epochs_trained':   len(history.history['loss']),
        'test_metrics':     dict(zip(metric_names, [float(v) for v in test_results])),
        'stable_model_path':  STABLE_MODEL_PATH,
        'stable_scaler_path': STABLE_SCALER_PATH,
    }
    with open(STABLE_META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print("=" * 70)
    print(f"Best model (timestamped): models/{run_name}_best.keras")
    print(f"Stable path (EA config):  {STABLE_MODEL_PATH}")
    print(f"Scaler (EA config):       {STABLE_SCALER_PATH}")
    print("=" * 70)
    print("\nUpdate config-ea.json NEURAL_NETWORKS section:")
    print(f'  "mlp_model_path":  "{STABLE_MODEL_PATH}"')
    print(f'  "mlp_scaler_path": "{STABLE_SCALER_PATH}"')

    return model, history, meta


def main():
    parser = argparse.ArgumentParser(description='Train MLP quality predictor')
    parser.add_argument('--dataset', default='data/all_combined_mlp.csv')
    parser.add_argument('--model', default='standard', choices=['standard', 'lite'])
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: dataset not found: {args.dataset}")
        sys.exit(1)

    try:
        train_model(args.dataset, args.model, args.epochs,
                    args.batch_size, args.learning_rate)
    except Exception as e:
        import traceback
        print(f"\nTraining failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
