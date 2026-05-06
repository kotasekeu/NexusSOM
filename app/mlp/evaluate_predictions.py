#!/usr/bin/env python3
"""Quick prediction evaluation on test set."""

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

MODEL_PATH  = 'models/mlp_latest.keras'
SCALER_PATH = 'models/mlp_scaler_latest.pkl'
DATASET     = 'data/all_combined_mlp.csv'
META        = 'data/all_combined_mlp_metadata.json'


def main():
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(META) as f:
        meta = json.load(f)

    df = pd.read_csv(DATASET)
    X  = df[meta['feature_columns']].values.astype(np.float32)
    y  = df[meta['target_columns']].values.astype(np.float32)
    ds = df['dataset_name']

    _, X_test, _, y_test, _, ds_test = train_test_split(
        X, y, ds, test_size=0.15, random_state=42, stratify=ds)
    X_test_s = scaler.transform(X_test)

    y_pred = model.predict(X_test_s, verbose=0)

    targets = [c.replace('raw_', '') for c in meta['target_columns']]
    mae = np.abs(y_test - y_pred).mean(axis=0)
    rmse = np.sqrt(((y_test - y_pred) ** 2).mean(axis=0))

    print("\nPer-target accuracy on test set:")
    print(f"  {'Target':<30}  {'MAE':>8}  {'RMSE':>8}")
    for t, m, r in zip(targets, mae, rmse):
        print(f"  {t:<30}  {m:8.4f}  {r:8.4f}")

    print(f"\nSample predictions (12 rows):")
    hdr = '  '.join(f'{t[:12]:>14}' for t in targets)
    print(f"  {'':6}  {'dataset':<30}  {hdr}")
    print(f"  {'':6}  {'':30}  {'  '.join(['actual / pred  '] * len(targets))}")
    print("  " + "-" * 100)
    for i in range(12):
        pairs = '  '.join(f'{y_test[i,j]:6.3f}/{y_pred[i,j]:6.3f}' for j in range(len(targets)))
        print(f"  {i:3d}    {ds_test.iloc[i]:<30}  {pairs}")

    print(f"\nmodel metrics_names: {model.metrics_names}")
    print(f"Test samples: {len(X_test)}")


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()
