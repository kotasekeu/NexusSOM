#!/usr/bin/env python3
"""
MLP Model Visualization

Generates diagnostic plots for a trained MLP model:
  - Scatter actual vs. predicted per target
  - Residual (error) distribution per target
  - MAE breakdown per dataset
  - Feature importance (permutation-based)

Saves plots to visualizations/<run_label>/.
Optionally compares two models side-by-side.

Usage:
    python3 visualize_model.py
    python3 visualize_model.py --model models/mlp_latest.keras --label v1
    python3 visualize_model.py --compare models/mlp_standard_20260330.keras --label_b v0
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_MODEL  = 'models/mlp_latest.keras'
DEFAULT_SCALER = 'models/mlp_scaler_latest.pkl'
DATASET        = 'data/all_combined_mlp.csv'
META           = 'data/all_combined_mlp_metadata.json'

TARGET_LABELS = {
    'raw_mqe_improvement_ratio': 'MQE improvement ratio',
    'raw_topographic_error':     'Topographic error',
    'dead_neuron_ratio':         'Dead neuron ratio',
}

COLORS = ['#2196F3', '#FF5722', '#4CAF50']


# ── data loading ─────────────────────────────────────────────────────────────

def load_test_split(dataset=DATASET, meta_path=META):
    import tensorflow as tf
    with open(meta_path) as f:
        meta = json.load(f)
    df = pd.read_csv(dataset)
    X  = df[meta['feature_columns']].values.astype(np.float32)
    y  = df[meta['target_columns']].values.astype(np.float32)
    ds = df['dataset_name'] if 'dataset_name' in df.columns else pd.Series(['?'] * len(df))
    _, X_test, _, y_test, _, ds_test = train_test_split(
        X, y, ds, test_size=0.15, random_state=42, stratify=ds)
    return X_test, y_test, ds_test, meta


def load_model_and_scaler(model_path, scaler_path):
    import tensorflow as tf
    model  = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def predict(model, scaler, X_test):
    return model.predict(scaler.transform(X_test), verbose=0)


# ── individual plots ──────────────────────────────────────────────────────────

def plot_scatter(y_test, y_pred, target_cols, out_dir, label):
    n = len(target_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i, (col, ax) in enumerate(zip(target_cols, axes)):
        t_label = TARGET_LABELS.get(col, col)
        lo = min(y_test[:, i].min(), y_pred[:, i].min())
        hi = max(y_test[:, i].max(), y_pred[:, i].max())
        margin = (hi - lo) * 0.05

        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.35, s=12, color=COLORS[i])
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', linewidth=1, alpha=0.5)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        mae  = np.abs(y_test[:, i] - y_pred[:, i]).mean()
        rmse = np.sqrt(((y_test[:, i] - y_pred[:, i]) ** 2).mean())
        ax.set_title(f'{t_label}\nMAE={mae:.4f}  RMSE={rmse:.4f}')
        ax.set_aspect('equal')

    fig.suptitle(f'Actual vs. Predicted — {label}', fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'scatter.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_residuals(y_test, y_pred, target_cols, out_dir, label):
    n = len(target_cols)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for i, (col, ax) in enumerate(zip(target_cols, axes)):
        t_label = TARGET_LABELS.get(col, col)
        residuals = y_pred[:, i] - y_test[:, i]
        ax.hist(residuals, bins=40, color=COLORS[i], alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.axvline(residuals.mean(), color='red', linewidth=1,
                   label=f'mean={residuals.mean():.4f}')
        ax.set_xlabel('Residual (pred − actual)')
        ax.set_ylabel('Count')
        ax.set_title(f'{t_label}\nstd={residuals.std():.4f}')
        ax.legend(fontsize=8)

    fig.suptitle(f'Residual Distribution — {label}', fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'residuals.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_per_dataset(y_test, y_pred, target_cols, ds_test, out_dir, label):
    datasets = sorted(ds_test.unique())
    n_ds  = len(datasets)
    n_tgt = len(target_cols)
    x = np.arange(n_ds)
    width = 0.8 / n_tgt

    fig, axes = plt.subplots(1, n_tgt, figsize=(5 * n_tgt, 4), sharey=False)
    if n_tgt == 1:
        axes = [axes]

    for i, (col, ax) in enumerate(zip(target_cols, axes)):
        t_label = TARGET_LABELS.get(col, col)
        maes = []
        for ds in datasets:
            mask = ds_test.values == ds
            mae = np.abs(y_test[mask, i] - y_pred[mask, i]).mean()
            maes.append(mae)
        bars = ax.bar(x, maes, width=0.6, color=COLORS[i], alpha=0.8, edgecolor='white')
        ax.bar_label(bars, fmt='%.4f', padding=2, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([d[:20] for d in datasets], rotation=20, ha='right', fontsize=8)
        ax.set_ylabel('MAE')
        ax.set_title(t_label)

    fig.suptitle(f'MAE per Dataset — {label}', fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'per_dataset.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_feature_importance(model, scaler, X_test, y_test, feature_cols,
                            out_dir, label, n_repeats=3):
    """Permutation importance: shuffle each feature, measure MSE increase."""
    X_scaled = scaler.transform(X_test)
    base_pred = model.predict(X_scaled, verbose=0)
    base_mse  = ((base_pred - y_test) ** 2).mean()

    importances = np.zeros(len(feature_cols))
    rng = np.random.default_rng(42)

    for j, feat in enumerate(feature_cols):
        delta = 0.0
        for _ in range(n_repeats):
            X_perm = X_scaled.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_pred = model.predict(X_perm, verbose=0)
            perm_mse  = ((perm_pred - y_test) ** 2).mean()
            delta += perm_mse - base_mse
        importances[j] = delta / n_repeats

    order = np.argsort(importances)[::-1]
    sorted_feats = [feature_cols[k] for k in order]
    sorted_imp   = importances[order]

    # Shorten feature names for display
    def shorten(name):
        replacements = {
            'start_': 's_', 'end_': 'e_', 'lr_decay_type_': 'lr_',
            'radius_decay_type_': 'rad_', 'batch_growth_type_': 'batch_',
            'ds_n_': 'ds_', '_ratio': '_r',
        }
        for k, v in replacements.items():
            name = name.replace(k, v)
        return name

    display_names = [shorten(f) for f in sorted_feats]

    fig, ax = plt.subplots(figsize=(10, max(5, len(feature_cols) * 0.35)))
    colors_bar = ['#E53935' if v > 0 else '#78909C' for v in sorted_imp]
    bars = ax.barh(range(len(sorted_feats)), sorted_imp, color=colors_bar, alpha=0.85)
    ax.set_yticks(range(len(sorted_feats)))
    ax.set_yticklabels(display_names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('MSE increase when feature is permuted')
    ax.set_title(f'Feature Importance (permutation) — {label}')
    fig.tight_layout()
    path = os.path.join(out_dir, 'feature_importance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

    return sorted_feats, sorted_imp


# ── comparison ────────────────────────────────────────────────────────────────

def plot_comparison(results_a, results_b, target_cols, label_a, label_b, out_dir):
    """Side-by-side MAE comparison between two models."""
    y_test = results_a['y_test']
    mae_a = np.abs(results_a['y_pred'] - y_test).mean(axis=0)
    mae_b = np.abs(results_b['y_pred'] - y_test).mean(axis=0)

    n = len(target_cols)
    x = np.arange(n)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(7, n * 2.5), 4))
    bars_a = ax.bar(x - width / 2, mae_a, width, label=label_a, color='#2196F3', alpha=0.8)
    bars_b = ax.bar(x + width / 2, mae_b, width, label=label_b, color='#FF5722', alpha=0.8)
    ax.bar_label(bars_a, fmt='%.4f', padding=2, fontsize=8)
    ax.bar_label(bars_b, fmt='%.4f', padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS.get(c, c) for c in target_cols], fontsize=9)
    ax.set_ylabel('MAE (test set)')
    ax.set_title(f'Model comparison: {label_a}  vs  {label_b}')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, 'comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

    print(f'\n  {"Target":<35}  {label_a:>10}  {label_b:>10}  {"Δ (b-a)":>10}')
    print(f'  {"-"*70}')
    for i, col in enumerate(target_cols):
        delta = mae_b[i] - mae_a[i]
        sign  = '▼' if delta < 0 else '▲' if delta > 0 else '='
        print(f'  {TARGET_LABELS.get(col, col):<35}  {mae_a[i]:>10.4f}  {mae_b[i]:>10.4f}  {sign}{abs(delta):>9.4f}')


# ── main ──────────────────────────────────────────────────────────────────────

def run_single(model_path, scaler_path, label, out_dir):
    print(f'\nLoading model: {model_path}')
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    X_test, y_test, ds_test, meta = load_test_split()
    y_pred = predict(model, scaler, X_test)
    target_cols = meta['target_columns']
    feature_cols = meta['feature_columns']

    os.makedirs(out_dir, exist_ok=True)
    print(f'Generating plots → {out_dir}/')

    plot_scatter(y_test, y_pred, target_cols, out_dir, label)
    plot_residuals(y_test, y_pred, target_cols, out_dir, label)
    plot_per_dataset(y_test, y_pred, target_cols, ds_test, out_dir, label)
    plot_feature_importance(model, scaler, X_test, y_test, feature_cols, out_dir, label)

    return {'y_test': y_test, 'y_pred': y_pred, 'target_cols': target_cols}


def main():
    parser = argparse.ArgumentParser(description='Visualize trained MLP model')
    parser.add_argument('--model',   default=DEFAULT_MODEL,  help='Model A .keras path')
    parser.add_argument('--scaler',  default=DEFAULT_SCALER, help='Scaler A .pkl path')
    parser.add_argument('--label',   default='latest',       help='Label for model A')
    parser.add_argument('--compare', default=None,           help='Model B .keras path for comparison')
    parser.add_argument('--scaler_b',default=None,           help='Scaler B .pkl path')
    parser.add_argument('--label_b', default='prev',         help='Label for model B')
    args = parser.parse_args()

    out_a = os.path.join('visualizations', args.label)
    results_a = run_single(args.model, args.scaler, args.label, out_a)

    if args.compare:
        scaler_b = args.scaler_b or args.compare.replace('.keras', '_scaler.pkl')
        out_b = os.path.join('visualizations', args.label_b)
        results_b = run_single(args.compare, scaler_b, args.label_b, out_b)

        out_cmp = os.path.join('visualizations', f'{args.label}_vs_{args.label_b}')
        os.makedirs(out_cmp, exist_ok=True)
        print(f'\nComparison → {out_cmp}/')
        plot_comparison(results_a, results_b, results_a['target_cols'],
                        args.label, args.label_b, out_cmp)

    print('\nDone.')


if __name__ == '__main__':
    main()
