# graphs.py
import os
import matplotlib.pyplot as plt
import numpy as np


def _setup_plot(title: str, xlabel: str, ylabel: str, figsize: tuple = (10, 6)):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)
    return plt.gca()


def _plot_history(ax, history_data: list, label: str, color: str, drawstyle: str = 'default'):
    if not history_data:
        return
    iterations, values = zip(*history_data)
    ax.plot(iterations, values, label=label, color=color, drawstyle=drawstyle)
    ax.legend()
    plt.tight_layout()


def generate_training_plots(training_results: dict, working_dir: str):
    plots_dir = os.path.join(working_dir, "visualizations")
    os.makedirs(plots_dir, exist_ok=True)
    print("INFO: Generuji grafy z průběhu tréninku...")

    history = training_results.get('history', {})
    if not history:
        print("WARNING: V výsledcích tréninku chybí data o historii. Grafy nebudou vygenerovány.")
        return

    mqe_history = history.get('mqe', [])
    if mqe_history:
        ax = _setup_plot("Vývoj kvantizační chyby (MQE)", "Iterace", "MQE")
        _plot_history(ax, mqe_history, "MQE", 'royalblue')

        best_mqe_val = training_results.get('best MQE')
        if best_mqe_val:
            iterations, values = zip(*mqe_history)
            best_mqe_iter = iterations[np.argmin(values)]
            ax.scatter(best_mqe_iter, best_mqe_val, color='red', zorder=5, label=f"Nejlepší MQE: {best_mqe_val:.4f}")
            ax.legend()

        plt.savefig(os.path.join(plots_dir, "mqe_evolution.png"), dpi=150)
        plt.close()

    lr_history = history.get('learning_rate', [])
    if lr_history:
        ax = _setup_plot("Vývoj rychlosti učení (Learning Rate)", "Iterace", "Hodnota")
        _plot_history(ax, lr_history, "Learning Rate", 'green')
        plt.savefig(os.path.join(plots_dir, "learning_rate_decay.png"), dpi=150)
        plt.close()

    radius_history = history.get('radius', [])
    if radius_history:
        ax = _setup_plot("Vývoj poloměru sousedství (Radius)", "Iterace", "Hodnota")
        _plot_history(ax, radius_history, "Radius", 'purple')
        plt.savefig(os.path.join(plots_dir, "radius_decay.png"), dpi=150)
        plt.close()

    batch_size_history = history.get('batch_size', [])
    if batch_size_history:
        ax = _setup_plot("Vývoj počtu zpracovaných vzorků", "Iterace", "Počet vzorků")
        _plot_history(ax, batch_size_history, "Počet vzorků", 'orange', drawstyle='steps-mid')
        plt.savefig(os.path.join(plots_dir, "batch_size_growth.png"), dpi=150)
        plt.close()

    print("INFO: Grafy z tréninku úspěšně vygenerovány.")