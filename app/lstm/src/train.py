"""
LSTM Training Script — Phase 2 Early Stopping Predictor

Loads pre-split numpy arrays from prepare_dataset.py, pads variable-length
sequences, trains the hybrid LSTM+context model, and saves to stable paths.

Usage:
    cd app/lstm
    python3 src/train.py
    python3 src/train.py --model lite --epochs 200
"""

import argparse
import json
import os
import sys
from datetime import datetime

import joblib
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau)

sys.path.insert(0, os.path.dirname(__file__))
from model import create_hybrid_lstm, create_lightweight_hybrid_lstm, print_model_summary

DATA_DIR          = os.path.join(os.path.dirname(__file__), '..', 'data')
STABLE_MODEL_PATH = 'models/lstm_latest.keras'
STABLE_SCALER_PATH = 'models/lstm_scaler_latest.pkl'
STABLE_META_PATH  = 'models/lstm_latest_metadata.json'


# ── data loading ──────────────────────────────────────────────────────────────

def pad_sequences(X_ragged: np.ndarray) -> np.ndarray:
    """
    Pad a ragged object-array of (seq_len, 6) arrays to a dense
    (N, max_len, 6) float32 array.  Short sequences are zero-padded on the right;
    the LSTM Masking layer will ignore those positions.
    """
    max_len = max(x.shape[0] for x in X_ragged)
    N = len(X_ragged)
    n_feat = X_ragged[0].shape[1]
    out = np.zeros((N, max_len, n_feat), dtype=np.float32)
    for i, seq in enumerate(X_ragged):
        out[i, :seq.shape[0], :] = seq
    return out


def load_splits(data_dir: str):
    print(f'Loading data from: {data_dir}')

    X_train_raw = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
    y_train     = np.load(os.path.join(data_dir, 'y_train.npy'))
    ctx_train   = np.load(os.path.join(data_dir, 'ctx_train.npy'))

    X_val_raw   = np.load(os.path.join(data_dir, 'X_val.npy'),   allow_pickle=True)
    y_val       = np.load(os.path.join(data_dir, 'y_val.npy'))
    ctx_val     = np.load(os.path.join(data_dir, 'ctx_val.npy'))

    X_test_raw  = np.load(os.path.join(data_dir, 'X_test.npy'),  allow_pickle=True)
    y_test      = np.load(os.path.join(data_dir, 'y_test.npy'))
    ctx_test    = np.load(os.path.join(data_dir, 'ctx_test.npy'))

    with open(os.path.join(data_dir, 'metadata.json')) as f:
        meta = json.load(f)

    print(f'  Train: {len(X_train_raw)} windows | Val: {len(X_val_raw)} | Test: {len(X_test_raw)}')

    # Pad sequences
    X_train = pad_sequences(X_train_raw)
    X_val   = pad_sequences(X_val_raw)
    X_test  = pad_sequences(X_test_raw)

    # Normalize context features
    scaler = StandardScaler()
    ctx_train = scaler.fit_transform(ctx_train)
    ctx_val   = scaler.transform(ctx_val)
    ctx_test  = scaler.transform(ctx_test)

    print(f'  Sequence shape: {X_train.shape}  (padded to max len)')
    print(f'  Context shape : {ctx_train.shape}')
    print(f'  Target shape  : {y_train.shape}')

    return (X_train, ctx_train, y_train,
            X_val,   ctx_val,   y_val,
            X_test,  ctx_test,  y_test,
            scaler, meta)


# ── training ──────────────────────────────────────────────────────────────────

def train(data_dir: str, model_type: str, epochs: int,
          batch_size: int, learning_rate: float):
    print('=' * 70)
    print('LSTM PHASE 2 TRAINING')
    print('=' * 70)

    np.random.seed(42)
    tf.random.set_seed(42)

    (X_train, ctx_train, y_train,
     X_val,   ctx_val,   y_val,
     X_test,  ctx_test,  y_test,
     scaler, meta) = load_splits(data_dir)

    context_dim = ctx_train.shape[1]
    output_dim  = y_train.shape[1]

    if model_type == 'lite':
        model = create_lightweight_hybrid_lstm(context_dim, output_dim, learning_rate)
    else:
        model = create_hybrid_lstm(context_dim, output_dim, learning_rate)
    print_model_summary(model)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name  = f'lstm_{model_type}_{timestamp}'
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

    print(f'\nTraining ({epochs} epochs max, patience=30)...')
    history = model.fit(
        {'sequence': X_train, 'context': ctx_train}, y_train,
        validation_data=({'sequence': X_val, 'context': ctx_val}, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print('\n' + '=' * 70)
    print('TEST SET EVALUATION')
    print('=' * 70)
    test_results = model.evaluate(
        {'sequence': X_test, 'context': ctx_test}, y_test, verbose=0)
    history_keys = [k for k in history.history if not k.startswith('val_')]
    for name, val in zip(history_keys, test_results):
        print(f'  {name}: {val:.6f}')

    print('\nSample predictions (first 8 test windows):')
    y_pred = model.predict(
        {'sequence': X_test[:8], 'context': ctx_test[:8]}, verbose=0)
    target_names = [c.replace('raw_', '') for c in meta['target_names']]
    for i in range(len(y_pred)):
        actual    = '  '.join(f'{v:10.4f}' for v in y_test[i])
        predicted = '  '.join(f'{v:10.4f}' for v in y_pred[i])
        print(f'  actual    [{actual}]')
        print(f'  predicted [{predicted}]')
        print()

    # Save timestamped copy
    model.save(f'models/{run_name}_final.keras')
    joblib.dump(scaler, f'models/{run_name}_scaler.pkl')

    # Save stable paths
    model.save(STABLE_MODEL_PATH)
    joblib.dump(scaler, STABLE_SCALER_PATH)

    train_meta = {
        'run_name':        run_name,
        'model_type':      model_type,
        'timestamp':       timestamp,
        'epochs_trained':  len(history.history['loss']),
        'sequence_shape':  list(X_train.shape),
        'context_dim':     context_dim,
        'output_dim':      output_dim,
        'feature_names':   meta['feature_names'],
        'context_names':   meta['context_names'],
        'target_names':    meta['target_names'],
        'train_windows':   len(X_train),
        'val_windows':     len(X_val),
        'test_windows':    len(X_test),
        'test_metrics':    dict(zip(history_keys, [float(v) for v in test_results])),
        'dataset_meta':    meta,
        'stable_model_path':  STABLE_MODEL_PATH,
        'stable_scaler_path': STABLE_SCALER_PATH,
    }
    with open(STABLE_META_PATH, 'w') as f:
        json.dump(train_meta, f, indent=2)

    print('=' * 70)
    print(f'Best model (timestamped): models/{run_name}_best.keras')
    print(f'Stable path (EA config):  {STABLE_MODEL_PATH}')
    print(f'Context scaler:           {STABLE_SCALER_PATH}')
    print('=' * 70)

    return model, history, train_meta


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Phase 2 predictor')
    parser.add_argument('--data',          default=DATA_DIR)
    parser.add_argument('--model',         default='standard',
                        choices=['standard', 'lite'])
    parser.add_argument('--epochs',        type=int,   default=300)
    parser.add_argument('--batch-size',    type=int,   default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data)
    if not os.path.exists(os.path.join(data_dir, 'X_train.npy')):
        print(f'Error: dataset not found in {data_dir}')
        print('Run prepare_dataset.py first.')
        sys.exit(1)

    try:
        train(data_dir, args.model, args.epochs,
              args.batch_size, args.learning_rate)
    except Exception as e:
        import traceback
        print(f'\nTraining failed: {e}')
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
