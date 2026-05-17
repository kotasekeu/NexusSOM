"""
LSTM Training — Phase 3 Dynamic Schedule Controller

Advantage-weighted behavioral cloning:
  - Input:  checkpoint sequence (T, 6) + dataset context (4,)
  - Output: (lr_factor, radius_factor) at each checkpoint (T, 2)
  - Loss:   MSE weighted by advantage (improvement over baseline)

Usage:
    cd app/lstm
    python3 src/train_phase3.py
    python3 src/train_phase3.py --epochs 200 --data data/phase3
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
from model_controller import create_controller_trainable

DATA_DIR           = os.path.join(os.path.dirname(__file__), '..', 'data', 'phase3')
STABLE_MODEL_PATH  = 'models/lstm_controller_latest.keras'
STABLE_SCALER_PATH = 'models/lstm_controller_scaler_latest.pkl'
STABLE_META_PATH   = 'models/lstm_controller_latest_metadata.json'


# ── data loading ──────────────────────────────────────────────────────────────

def load_splits(data_dir: str):
    print(f'Loading Phase 3 data from: {data_dir}')

    def _load(split):
        X   = np.load(os.path.join(data_dir, f'X_{split}.npy'))
        y   = np.load(os.path.join(data_dir, f'y_{split}.npy'))
        ctx = np.load(os.path.join(data_dir, f'ctx_{split}.npy'))
        adv = np.load(os.path.join(data_dir, f'adv_{split}.npy'))
        msk_path = os.path.join(data_dir, f'msk_{split}.npy')
        msk = np.load(msk_path) if os.path.exists(msk_path) else np.ones((len(X), X.shape[1]), np.float32)
        return X, y, ctx, adv, msk

    X_train, y_train, ctx_train, adv_train, msk_train = _load('train')
    X_val,   y_val,   ctx_val,   adv_val,   msk_val   = _load('val')
    X_test,  y_test,  ctx_test,  adv_test,  msk_test  = _load('test')

    with open(os.path.join(data_dir, 'metadata_p3.json')) as f:
        meta = json.load(f)

    pct_active = msk_train.mean() * 100
    print(f'  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}')
    print(f'  Sequence shape (train): {X_train.shape}')
    print(f'  Active timesteps (train): {pct_active:.0f}%  '
          f'(non-perturbed timesteps get zero loss weight)')
    print(f'  Advantage mean (train): {adv_train.mean():.3f}  '
          f'nonzero: {(adv_train > 0).sum()}/{len(adv_train)}')

    # Normalize context
    scaler = StandardScaler()
    ctx_train = scaler.fit_transform(ctx_train)
    ctx_val   = scaler.transform(ctx_val)
    ctx_test  = scaler.transform(ctx_test)

    return (X_train, ctx_train, y_train, adv_train,
            X_val,   ctx_val,   y_val,   adv_val,
            X_test,  ctx_test,  y_test,  adv_test,
            scaler, meta)


# ── training ──────────────────────────────────────────────────────────────────

def train(data_dir: str, epochs: int, batch_size: int, learning_rate: float):
    print('=' * 70)
    print('LSTM PHASE 3 TRAINING — Dynamic Schedule Controller')
    print('=' * 70)

    np.random.seed(42)
    tf.random.set_seed(42)

    (X_train, ctx_train, y_train, adv_train,
     X_val,   ctx_val,   y_val,   adv_val,
     X_test,  ctx_test,  y_test,  adv_test,
     scaler, meta) = load_splits(data_dir)

    context_dim = ctx_train.shape[1]
    model = create_controller_trainable(context_dim, learning_rate)

    # Combined timestep weight: advantage (per-trajectory) × perturb_mask (per-timestep).
    # Non-perturbed timesteps (target=1.0, ~60% of data) get zero weight so
    # the model cannot collapse to predicting 1.0 for everything.
    def _combined_weight(adv, msk):
        return (adv[:, np.newaxis] * msk).astype(np.float32)  # (N, T)

    adv_train_tiled = _combined_weight(adv_train, msk_train)
    adv_val_tiled   = _combined_weight(adv_val,   msk_val)
    adv_test_tiled  = _combined_weight(adv_test,  msk_test)
    model.summary()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name  = f'lstm_controller_{timestamp}'
    log_dir   = f'logs/{run_name}'
    os.makedirs('models', exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            filepath=f'models/{run_name}_best.keras',
            monitor='val_loss', save_best_only=True, mode='min', verbose=0),
        EarlyStopping(
            monitor='val_loss', patience=40,
            restore_best_weights=True, mode='min', verbose=1),
        ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=20,
            min_lr=1e-7, mode='min', verbose=1),
        CSVLogger(os.path.join(log_dir, 'history.csv')),
    ]

    print(f'\nTraining ({epochs} epochs max, advantage-weighted loss)...')
    history = model.fit(
        {'sequence': X_train, 'context': ctx_train}, y_train,
        sample_weight=adv_train_tiled,
        validation_data=(
            {'sequence': X_val, 'context': ctx_val}, y_val, adv_val_tiled),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    print('\n' + '=' * 70)
    print('TEST SET EVALUATION')
    print('=' * 70)
    test_results = model.evaluate(
        {'sequence': X_test, 'context': ctx_test}, y_test,
        sample_weight=adv_test_tiled, verbose=0)
    history_keys = [k for k in history.history if not k.startswith('val_')]
    for name, val in zip(history_keys, test_results):
        print(f'  {name}: {val:.6f}')

    print('\nSample predictions (first 5 test sequences, last checkpoint):')
    y_pred = model.predict(
        {'sequence': X_test[:5], 'context': ctx_test[:5]}, verbose=0)
    for i in range(min(5, len(y_pred))):
        last = -1  # last valid checkpoint
        act = y_pred[i, last]
        tgt = y_test[i, last]
        print(f'  [{i}] predicted (lr_f={act[0]:.3f}, rad_f={act[1]:.3f})  '
              f'target ({tgt[0]:.3f}, {tgt[1]:.3f})  '
              f'adv={adv_test[i]:.3f}')

    model.save(f'models/{run_name}_final.keras')
    joblib.dump(scaler, f'models/{run_name}_scaler.pkl')

    model.save(STABLE_MODEL_PATH)
    joblib.dump(scaler, STABLE_SCALER_PATH)

    train_meta = {
        'run_name':       run_name,
        'timestamp':      timestamp,
        'phase':          3,
        'epochs_trained': len(history.history['loss']),
        'train_size':     len(X_train),
        'val_size':       len(X_val),
        'test_size':      len(X_test),
        'sequence_shape': list(X_train.shape),
        'context_dim':    context_dim,
        'output_dim':     2,
        'feature_names':  meta['feature_names'],
        'action_names':   meta['action_names'],
        'test_metrics':   dict(zip(history_keys, [float(v) for v in test_results])),
        'stable_model_path':  STABLE_MODEL_PATH,
        'stable_scaler_path': STABLE_SCALER_PATH,
    }
    with open(STABLE_META_PATH, 'w') as f:
        json.dump(train_meta, f, indent=2)

    print('=' * 70)
    print(f'Best model: models/{run_name}_best.keras')
    print(f'Stable:     {STABLE_MODEL_PATH}')
    print('=' * 70)

    return model, history, train_meta


def main():
    parser = argparse.ArgumentParser(description='Train LSTM Phase 3 controller')
    parser.add_argument('--data',          default=DATA_DIR)
    parser.add_argument('--epochs',        type=int,   default=300)
    parser.add_argument('--batch-size',    type=int,   default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data)
    if not os.path.exists(os.path.join(data_dir, 'X_train.npy')):
        print(f'Dataset not found in {data_dir}')
        print('Run prepare_phase3_dataset.py first.')
        sys.exit(1)

    try:
        train(data_dir, args.epochs, args.batch_size, args.learning_rate)
    except Exception as e:
        import traceback
        print(f'\nTraining failed: {e}')
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
