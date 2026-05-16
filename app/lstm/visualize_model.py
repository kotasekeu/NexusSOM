#!/usr/bin/env python3
"""
LSTM Model Visualization — Phase 2 (early stopping) or Phase 3 (controller)

Phase 2 plots:
  - scatter.png         Actual vs. predicted per target
  - residuals.png       Residual error distribution per target
  - prefix_accuracy.png MAE at each prefix length K (20–70%)
  - early_stopping.png  Quality score distribution + threshold confusion matrix

Phase 3 plots:
  - scatter_p3.png      Predicted vs. actual lr_factor and radius_factor
  - residuals_p3.png    Residual error distribution for each action
  - advantage_p3.png    MAE broken down by advantage quartile

Usage:
    cd app/lstm
    python3 visualize_model.py                    # Phase 2
    python3 visualize_model.py --phase 3          # Phase 3 controller
    python3 visualize_model.py --phase 2 --label v2 --compare models/lstm_standard.keras
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Phase 2 defaults
DEFAULT_MODEL_P2  = 'models/lstm_latest.keras'
DEFAULT_SCALER_P2 = 'models/lstm_scaler_latest.pkl'
DATA_DIR_P2       = 'data'

# Phase 3 defaults
DEFAULT_MODEL_P3  = 'models/lstm_controller_latest.keras'
DEFAULT_SCALER_P3 = 'models/lstm_controller_scaler_latest.pkl'
DATA_DIR_P3       = 'data/phase3'

# Keep backward-compat names
DEFAULT_MODEL  = DEFAULT_MODEL_P2
DEFAULT_SCALER = DEFAULT_SCALER_P2
DATA_DIR       = DATA_DIR_P2

TARGET_LABELS = {
    'raw_mqe_improvement_ratio': 'MQE improvement ratio',
    'raw_topographic_error':     'Topographic error',
    'dead_neuron_ratio':         'Dead neuron ratio',
}
TARGET_COLS = ['raw_mqe_improvement_ratio', 'raw_topographic_error', 'dead_neuron_ratio']

COLORS = ['#2196F3', '#FF5722', '#4CAF50']
QUALITY_THRESHOLD = 0.75


# ── data loading ──────────────────────────────────────────────────────────────

def load_test_data(data_dir=DATA_DIR):
    X_test   = np.load(os.path.join(data_dir, 'X_test.npy'),   allow_pickle=True)
    y_test   = np.load(os.path.join(data_dir, 'y_test.npy'),   allow_pickle=True)
    ctx_test = np.load(os.path.join(data_dir, 'ctx_test.npy'), allow_pickle=True)
    with open(os.path.join(data_dir, 'metadata.json')) as f:
        meta = json.load(f)
    return X_test, y_test, ctx_test, meta


def load_model_and_scaler(model_path, scaler_path):
    import tensorflow as tf
    # Phase 3 controller uses custom layers — must provide custom_objects
    if 'controller' in model_path:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from model_controller import _TileContext, _ScaleSigmoid
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={'_TileContext': _TileContext, '_ScaleSigmoid': _ScaleSigmoid}
        )
    else:
        model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def run_predictions(model, scaler, X_test, ctx_test):
    """Run inference sample-by-sample (variable-length sequences)."""
    preds = []
    for i in range(len(X_test)):
        seq = np.expand_dims(X_test[i].astype(np.float32), 0)   # (1, K, 6)
        ctx = scaler.transform(ctx_test[i:i+1])                  # (1, 4)
        pred = model.predict({'sequence': seq, 'context': ctx}, verbose=0)[0]
        preds.append(pred)
    return np.array(preds)   # (N, 3)


def quality_score(y):
    """quality = (1 - mqe_ratio) + topo_error + dead_ratio*0.5  (lower = better)"""
    return (1.0 - y[:, 0]) + y[:, 1] + y[:, 2] * 0.5


def prefix_lengths(X_test):
    return np.array([x.shape[0] for x in X_test])


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_scatter(y_test, y_pred, out_dir, label):
    n = len(TARGET_COLS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    for i, (col, ax) in enumerate(zip(TARGET_COLS, axes)):
        t_label = TARGET_LABELS[col]
        lo = min(y_test[:, i].min(), y_pred[:, i].min())
        hi = max(y_test[:, i].max(), y_pred[:, i].max())
        margin = (hi - lo) * 0.05

        ax.scatter(y_test[:, i], y_pred[:, i], alpha=0.45, s=20, color=COLORS[i])
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                'k--', linewidth=1, alpha=0.5)
        ax.set_xlim(lo - margin, hi + margin)
        ax.set_ylim(lo - margin, hi + margin)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        mae  = np.abs(y_test[:, i] - y_pred[:, i]).mean()
        rmse = np.sqrt(((y_test[:, i] - y_pred[:, i]) ** 2).mean())
        r    = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]
        ax.set_title(f'{t_label}\nMAE={mae:.4f}  RMSE={rmse:.4f}  r={r:.3f}')
        ax.set_aspect('equal')

    fig.suptitle(f'Actual vs. Predicted — {label}', fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'scatter.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_residuals(y_test, y_pred, out_dir, label):
    n = len(TARGET_COLS)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))

    for i, (col, ax) in enumerate(zip(TARGET_COLS, axes)):
        t_label = TARGET_LABELS[col]
        res = y_pred[:, i] - y_test[:, i]
        ax.hist(res, bins=30, color=COLORS[i], alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.axvline(res.mean(), color='red', linewidth=1,
                   label=f'mean={res.mean():.4f}')
        ax.set_xlabel('Residual (pred − actual)')
        ax.set_ylabel('Count')
        ax.set_title(f'{t_label}\nstd={res.std():.4f}')
        ax.legend(fontsize=8)

    fig.suptitle(f'Residual Distribution — {label}', fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'residuals.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_prefix_accuracy(y_test, y_pred, X_test, out_dir, label):
    """
    Show MAE per target and quality-score MAE broken down by prefix length K.
    Each test window was cut at exactly one K-fraction — groups are exact.
    """
    lengths = prefix_lengths(X_test)
    unique_lens = sorted(set(lengths))

    # Map lengths to K-fraction labels (resample_len=200)
    def to_pct(l):
        return f'{round(l / 200 * 100):d}%'

    k_labels = [to_pct(l) for l in unique_lens]

    qs_true = quality_score(y_test)
    qs_pred = quality_score(y_pred)

    # MAE per target per K
    target_maes = {col: [] for col in TARGET_COLS}
    qs_maes = []
    counts  = []
    for l in unique_lens:
        mask = lengths == l
        counts.append(mask.sum())
        for j, col in enumerate(TARGET_COLS):
            target_maes[col].append(np.abs(y_test[mask, j] - y_pred[mask, j]).mean())
        qs_maes.append(np.abs(qs_true[mask] - qs_pred[mask]).mean())

    x = np.arange(len(unique_lens))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: per-target MAE
    ax = axes[0]
    width = 0.25
    for j, col in enumerate(TARGET_COLS):
        offset = (j - 1) * width
        bars = ax.bar(x + offset, target_maes[col], width,
                      label=TARGET_LABELS[col], color=COLORS[j], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}\n(n={c})' for k, c in zip(k_labels, counts)], fontsize=9)
    ax.set_xlabel('Prefix length K')
    ax.set_ylabel('MAE')
    ax.set_title('Per-target MAE by prefix length')
    ax.legend(fontsize=8)

    # Right: quality-score MAE
    ax = axes[1]
    ax.bar(x, qs_maes, color='#9C27B0', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{k}\n(n={c})' for k, c in zip(k_labels, counts)], fontsize=9)
    ax.set_xlabel('Prefix length K')
    ax.set_ylabel('Quality score MAE')
    ax.set_title('Quality score MAE by prefix length')
    for xi, v in zip(x, qs_maes):
        ax.text(xi, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    fig.suptitle(f'Accuracy by Prefix Length — {label}', fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'prefix_accuracy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_early_stopping(y_test, y_pred, out_dir, label, threshold=QUALITY_THRESHOLD):
    """
    Show quality score distribution (predicted vs actual) and
    confusion matrix for the early-stopping decision at the given threshold.
    """
    qs_true = quality_score(y_test)
    qs_pred = quality_score(y_pred)

    stop_true = qs_true > threshold
    stop_pred = qs_pred > threshold

    tp = (stop_pred &  stop_true).sum()
    tn = (~stop_pred & ~stop_true).sum()
    fp = (stop_pred & ~stop_true).sum()
    fn = (~stop_pred &  stop_true).sum()
    n  = len(qs_true)

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[2, 2, 1.2])

    # ── left: QS distribution (actual) ────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    bins = np.linspace(min(qs_true.min(), qs_pred.min()) - 0.02,
                       max(qs_true.max(), qs_pred.max()) + 0.02, 30)
    ax0.hist(qs_true, bins=bins, color='#2196F3', alpha=0.6, label='Actual')
    ax0.hist(qs_pred, bins=bins, color='#FF5722', alpha=0.6, label='Predicted')
    ax0.axvline(threshold, color='black', linewidth=1.5, linestyle='--',
                label=f'threshold={threshold}')
    ax0.set_xlabel('Quality score')
    ax0.set_ylabel('Count')
    ax0.set_title('Quality score distribution\n(actual vs. predicted)')
    ax0.legend(fontsize=8)

    # ── middle: scatter QS actual vs predicted ─────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    colors_pt = np.where(stop_true, '#E53935', '#43A047')
    ax1.scatter(qs_true, qs_pred, c=colors_pt, alpha=0.6, s=25)
    lo = min(qs_true.min(), qs_pred.min()) - 0.02
    hi = max(qs_true.max(), qs_pred.max()) + 0.02
    ax1.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, alpha=0.5)
    ax1.axhline(threshold, color='red',  linewidth=0.8, alpha=0.4)
    ax1.axvline(threshold, color='blue', linewidth=0.8, alpha=0.4)
    ax1.set_xlabel('Actual quality score')
    ax1.set_ylabel('Predicted quality score')
    mae_qs = np.abs(qs_true - qs_pred).mean()
    ax1.set_title(f'QS scatter  MAE={mae_qs:.4f}\n'
                  f'red = should stop, green = should continue')
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)

    # ── right: confusion matrix ────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    cm = np.array([[tn, fp], [fn, tp]])
    im = ax2.imshow(cm, cmap='Blues', aspect='auto')
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Pred: OK', 'Pred: STOP'])
    ax2.set_yticklabels(['True: OK', 'True: STOP'])
    for (r, c), val in np.ndenumerate(cm):
        ax2.text(c, r, str(val), ha='center', va='center',
                 fontsize=14, fontweight='bold',
                 color='white' if val > cm.max() / 2 else 'black')
    acc  = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    ax2.set_title(f'Confusion (thresh={threshold})\n'
                  f'Acc={acc:.2f}  Prec={prec:.2f}  Rec={rec:.2f}')

    fig.suptitle(f'Early Stopping Analysis — {label}', fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'early_stopping.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


# ── comparison ─────────────────────────────────────────────────────────────────

def plot_comparison(results_a, results_b, label_a, label_b, out_dir):
    """Side-by-side MAE comparison between two models."""
    y_test = results_a['y_test']
    mae_a  = np.abs(results_a['y_pred'] - y_test).mean(axis=0)
    mae_b  = np.abs(results_b['y_pred'] - y_test).mean(axis=0)

    x     = np.arange(len(TARGET_COLS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 4))
    bars_a = ax.bar(x - width / 2, mae_a, width, label=label_a, color='#2196F3', alpha=0.8)
    bars_b = ax.bar(x + width / 2, mae_b, width, label=label_b, color='#FF5722', alpha=0.8)
    ax.bar_label(bars_a, fmt='%.4f', padding=2, fontsize=8)
    ax.bar_label(bars_b, fmt='%.4f', padding=2, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS[c] for c in TARGET_COLS], fontsize=9)
    ax.set_ylabel('MAE (test set)')
    ax.set_title(f'Model comparison: {label_a}  vs  {label_b}')
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, 'comparison.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')

    print(f'\n  {"Target":<35}  {label_a:>10}  {label_b:>10}  {"Δ (b-a)":>10}')
    print(f'  {"-" * 70}')
    for i, col in enumerate(TARGET_COLS):
        delta = mae_b[i] - mae_a[i]
        sign  = '▼' if delta < 0 else '▲' if delta > 0 else '='
        print(f'  {TARGET_LABELS[col]:<35}  {mae_a[i]:>10.4f}  {mae_b[i]:>10.4f}  '
              f'{sign}{abs(delta):>9.4f}')

    # Quality score comparison
    qs_mae_a = np.abs(quality_score(y_test) - quality_score(results_a['y_pred'])).mean()
    qs_mae_b = np.abs(quality_score(y_test) - quality_score(results_b['y_pred'])).mean()
    delta_qs = qs_mae_b - qs_mae_a
    sign = '▼' if delta_qs < 0 else '▲'
    print(f'  {"Quality score (composite)":<35}  {qs_mae_a:>10.4f}  {qs_mae_b:>10.4f}  '
          f'{sign}{abs(delta_qs):>9.4f}')


# ── Phase 3 data + plots ──────────────────────────────────────────────────────

def load_test_data_p3(data_dir=DATA_DIR_P3):
    X_test   = np.load(os.path.join(data_dir, 'X_test.npy'),   allow_pickle=True)
    y_test   = np.load(os.path.join(data_dir, 'y_test.npy'),   allow_pickle=True)
    ctx_test = np.load(os.path.join(data_dir, 'ctx_test.npy'), allow_pickle=True)
    adv_test = np.load(os.path.join(data_dir, 'adv_test.npy'), allow_pickle=True)
    with open(os.path.join(data_dir, 'metadata_p3.json'), encoding='utf-8') as f:
        meta = json.load(f)
    return X_test, y_test, ctx_test, adv_test, meta


def run_predictions_p3(model, scaler, X_test, ctx_test):
    """Run inference for Phase 3 controller (seq2seq: output shape = (T, 2) per sample)."""
    preds = []
    for i in range(len(X_test)):
        seq = np.expand_dims(X_test[i].astype(np.float32), 0)   # (1, T, 6)
        ctx = scaler.transform(ctx_test[i:i+1])                  # (1, 4)
        pred = model.predict({'sequence': seq, 'context': ctx}, verbose=0)[0]  # (T, 2)
        preds.append(pred)
    return preds   # list of (T, 2)


def plot_scatter_p3(y_test, y_pred_list, out_dir, label):
    """Scatter: predicted vs actual lr_factor and radius_factor (flattened across all timesteps)."""
    action_names = ['lr_factor', 'radius_factor']
    colors_p3    = ['#2196F3', '#FF5722']

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, (name, color) in enumerate(zip(action_names, colors_p3)):
        actual = np.concatenate([y_test[j][:, i] for j in range(len(y_test))])
        pred   = np.concatenate([y_pred_list[j][:, i] for j in range(len(y_pred_list))])
        ax = axes[i]
        lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        m = (hi - lo) * 0.05
        ax.scatter(actual, pred, alpha=0.3, s=8, color=color)
        ax.plot([lo - m, hi + m], [lo - m, hi + m], 'k--', linewidth=1, alpha=0.5)
        mae  = np.abs(actual - pred).mean()
        rmse = np.sqrt(((actual - pred) ** 2).mean())
        r    = np.corrcoef(actual, pred)[0, 1] if len(actual) > 1 else float('nan')
        ax.set_title(f'{name}\nMAE={mae:.4f}  RMSE={rmse:.4f}  r={r:.3f}')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_xlim(lo - m, hi + m)
        ax.set_ylim(lo - m, hi + m)
        ax.set_aspect('equal')

    fig.suptitle(f'Controller Predictions — {label}', fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'scatter_p3.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_residuals_p3(y_test, y_pred_list, out_dir, label):
    action_names = ['lr_factor', 'radius_factor']
    colors_p3    = ['#2196F3', '#FF5722']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, (name, color) in enumerate(zip(action_names, colors_p3)):
        actual = np.concatenate([y_test[j][:, i] for j in range(len(y_test))])
        pred   = np.concatenate([y_pred_list[j][:, i] for j in range(len(y_pred_list))])
        res    = pred - actual
        ax = axes[i]
        ax.hist(res, bins=40, color=color, alpha=0.7, edgecolor='white')
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        ax.axvline(res.mean(), color='red', linewidth=1, label=f'mean={res.mean():.4f}')
        ax.set_xlabel('Residual (pred − actual)')
        ax.set_ylabel('Count')
        ax.set_title(f'{name}\nstd={res.std():.4f}')
        ax.legend(fontsize=8)

    fig.suptitle(f'Residual Distribution — {label}', fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'residuals_p3.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def plot_advantage_p3(y_test, y_pred_list, adv_test, out_dir, label):
    """MAE per action broken down by advantage quartile."""
    action_names = ['lr_factor', 'radius_factor']
    colors_p3    = ['#2196F3', '#FF5722']

    # Compute per-trajectory MAE
    traj_mae = np.array([
        np.abs(y_test[j] - y_pred_list[j]).mean(axis=0)   # (2,)
        for j in range(len(y_test))
    ])   # (N, 2)

    quartiles   = np.percentile(adv_test, [0, 25, 50, 75, 100])
    q_labels    = ['Q1\n(0–25%)', 'Q2\n(25–50%)', 'Q3\n(50–75%)', 'Q4\n(75–100%)']
    q_masks     = [
        (adv_test >= quartiles[0]) & (adv_test < quartiles[1]),
        (adv_test >= quartiles[1]) & (adv_test < quartiles[2]),
        (adv_test >= quartiles[2]) & (adv_test < quartiles[3]),
        (adv_test >= quartiles[3]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x     = np.arange(len(q_labels))
    width = 0.35

    for ai, (name, color) in enumerate(zip(action_names, colors_p3)):
        ax = axes[ai]
        means  = [traj_mae[m, ai].mean() if m.sum() > 0 else 0 for m in q_masks]
        counts = [m.sum() for m in q_masks]
        bars   = ax.bar(x, means, width=0.5, color=color, alpha=0.8)
        ax.bar_label(bars, fmt='%.4f', padding=2, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{q}\n(n={c})' for q, c in zip(q_labels, counts)], fontsize=9)
        ax.set_xlabel('Advantage quartile')
        ax.set_ylabel('MAE')
        ax.set_title(f'{name}\nMAE by advantage quartile')

    fig.suptitle(f'Controller MAE vs Advantage — {label}', fontsize=13, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, 'advantage_p3.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')


def run_single_p3(model_path, scaler_path, label, out_dir):
    print(f'\nLoading Phase 3 controller: {model_path}')
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    X_test, y_test, ctx_test, adv_test, meta = load_test_data_p3()

    print(f'Running predictions on {len(X_test)} test trajectories...')
    y_pred_list = run_predictions_p3(model, scaler, X_test, ctx_test)

    os.makedirs(out_dir, exist_ok=True)
    print(f'Generating plots → {out_dir}/')

    plot_scatter_p3(y_test, y_pred_list, out_dir, label)
    plot_residuals_p3(y_test, y_pred_list, out_dir, label)
    plot_advantage_p3(y_test, y_pred_list, adv_test, out_dir, label)


# ── main ──────────────────────────────────────────────────────────────────────

def run_single(model_path, scaler_path, label, out_dir):
    print(f'\nLoading model: {model_path}')
    model, scaler = load_model_and_scaler(model_path, scaler_path)
    X_test, y_test, ctx_test, meta = load_test_data()

    print(f'Running predictions on {len(X_test)} test windows...')
    y_pred = run_predictions(model, scaler, X_test, ctx_test)

    os.makedirs(out_dir, exist_ok=True)
    print(f'Generating plots → {out_dir}/')

    plot_scatter(y_test, y_pred, out_dir, label)
    plot_residuals(y_test, y_pred, out_dir, label)
    plot_prefix_accuracy(y_test, y_pred, X_test, out_dir, label)
    plot_early_stopping(y_test, y_pred, out_dir, label)

    return {'y_test': y_test, 'y_pred': y_pred, 'X_test': X_test}


def main():
    parser = argparse.ArgumentParser(description='Visualize trained LSTM model')
    parser.add_argument('--phase',   type=int, default=2, choices=[2, 3],
                        help='Which model to visualize: 2=early stopping, 3=controller (default: 2)')
    parser.add_argument('--model',   default=None,  help='Model .keras path (overrides phase default)')
    parser.add_argument('--scaler',  default=None,  help='Scaler .pkl path (overrides phase default)')
    parser.add_argument('--label',   default=None,  help='Label for the run (default: phase2/phase3)')
    parser.add_argument('--compare', default=None,  help='Model B .keras path (Phase 2 only)')
    parser.add_argument('--scaler_b',default=None,  help='Scaler B .pkl path')
    parser.add_argument('--label_b', default='prev',help='Label for model B')
    args = parser.parse_args()

    if args.phase == 3:
        model_path  = args.model  or DEFAULT_MODEL_P3
        scaler_path = args.scaler or DEFAULT_SCALER_P3
        label       = args.label  or 'phase3'
        out_dir     = os.path.join('visualizations', label)
        run_single_p3(model_path, scaler_path, label, out_dir)
    else:
        model_path  = args.model  or DEFAULT_MODEL_P2
        scaler_path = args.scaler or DEFAULT_SCALER_P2
        label       = args.label  or 'phase2'
        out_a       = os.path.join('visualizations', label)
        results_a   = run_single(model_path, scaler_path, label, out_a)

        if args.compare:
            scaler_b  = args.scaler_b or args.compare.replace('.keras', '_scaler.pkl')
            out_b     = os.path.join('visualizations', args.label_b)
            results_b = run_single(args.compare, scaler_b, args.label_b, out_b)
            out_cmp   = os.path.join('visualizations', f'{label}_vs_{args.label_b}')
            os.makedirs(out_cmp, exist_ok=True)
            print(f'\nComparison → {out_cmp}/')
            plot_comparison(results_a, results_b, label, args.label_b, out_cmp)

    print('\nDone.')


if __name__ == '__main__':
    main()
