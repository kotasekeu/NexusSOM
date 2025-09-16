# graphs.py
import os
import matplotlib.pyplot as plt
import numpy as np


def _setup_plot(title: str, xlabel: str, ylabel: str, figsize: tuple = (10, 6)):
    """
    Set up the plot with title, axis labels, and grid.
    """
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    return plt.gca()


def _plot_history(ax, history_data: list, label: str, color: str, drawstyle: str = 'default'):
    """
    Plot training history data on the provided axis.
    """
    if not history_data:
        return
    iterations, values = zip(*history_data)
    ax.plot(iterations, values, label=label, color=color, drawstyle=drawstyle)
    ax.legend()
    plt.tight_layout()


def generate_training_plots(training_results: dict, output_dir: str):
    """
    Generate and save training progress plots (MQE, learning rate, radius, batch size).
    """
    plots_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(plots_dir, exist_ok=True)
    # print("INFO: Generating training progress plots...")

    history = training_results.get('history', {})
    if not history:
        print("WARNING: Training history data missing in results. Plots will not be generated.")
        return

    mqe_history = history.get('mqe', [])
    if mqe_history:
        ax = _setup_plot("MQE Evolution", "Iteration", "MQE")
        _plot_history(ax, mqe_history, "MQE", 'royalblue')

        best_mqe_val = training_results.get('best MQE')
        if best_mqe_val:
            iterations, values = zip(*mqe_history)
            best_mqe_iter = iterations[np.argmin(values)]
            ax.scatter(best_mqe_iter, best_mqe_val, color='red', zorder=5, label=f"Best MQE: {best_mqe_val:.4f}")
            ax.legend()

        plt.savefig(os.path.join(plots_dir, "mqe_evolution.png"), dpi=150)
        plt.close()

    lr_history = history.get('learning_rate', [])
    if lr_history:
        ax = _setup_plot("Learning Rate Evolution", "Iteration", "Value")
        _plot_history(ax, lr_history, "Learning Rate", 'green')
        plt.savefig(os.path.join(plots_dir, "learning_rate_decay.png"), dpi=150)
        plt.close()

    radius_history = history.get('radius', [])
    if radius_history:
        ax = _setup_plot("Radius Evolution", "Iteration", "Value")
        _plot_history(ax, radius_history, "Radius", 'purple')
        plt.savefig(os.path.join(plots_dir, "radius_decay.png"), dpi=150)
        plt.close()

    batch_size_history = history.get('batch_size', [])
    if batch_size_history:
        ax = _setup_plot("Processed Samples Evolution", "Iteration", "Sample Count")
        _plot_history(ax, batch_size_history, "Sample Count", 'orange', drawstyle='steps-mid')
        plt.savefig(os.path.join(plots_dir, "batch_size_growth.png"), dpi=150)
        plt.close()

    # print("INFO: Training plots successfully generated.")
