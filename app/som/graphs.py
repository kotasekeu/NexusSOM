# graphs.py
import os
import matplotlib.pyplot as plt
import numpy as np
from som import KohonenSOM


def _setup_plot(title: str, xlabel: str, ylabel: str, figsize: tuple = (10, 6)):
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, linestyle='--', alpha=0.6)


def generate_training_plots(som: KohonenSOM, training_results: dict, config: dict, working_dir: str):
    plots_dir = os.path.join(working_dir, "visualizations")
    os.makedirs(plots_dir, exist_ok=True)

    print("INFO: Generuji grafy z průběhu tréninku...")

    mqe_history = training_results.get('mqe_history', [])
    if mqe_history:
        epochs_ran = len(mqe_history)
        iterations = np.arange(epochs_ran)

        _setup_plot(
            title="Vývoj kvantizační chyby (MQE) v průběhu tréninku",
            xlabel="Iterace",
            ylabel="MQE"
        )
        plt.plot(iterations, mqe_history, label="MQE", color='royalblue')

        best_mqe_val = training_results.get('best MQE')
        if best_mqe_val is not None:
            best_mqe_iter = np.argmin(mqe_history)
            plt.scatter(best_mqe_iter, best_mqe_val, color='red', zorder=5, label=f"Nejlepší MQE: {best_mqe_val:.4f}")

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "mqe_evolution.png"), dpi=150)
        plt.close()

    total_iterations = config.get("epoch_multiplier", 1) * config.get("num_samples", 1000)  # Aproximace
    if 'epochs_ran' in training_results:
        total_iterations = training_results['epochs_ran']

    iterations_axis = np.arange(total_iterations)

    # Learning Rate
    _setup_plot("Vývoj rychlosti učení (Learning Rate)", "Iterace", "Hodnota")
    lr_values = [
        som.get_decay_value(i, total_iterations, som.start_learning_rate, som.end_learning_rate, som.lr_decay_type) for
        i in iterations_axis]
    plt.plot(iterations_axis, lr_values, color='green', label=f"Typ: {som.lr_decay_type}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "learning_rate_decay.png"), dpi=150)
    plt.close()

    # Radius
    # TODO - maybe show zone of gaussian influence on the plot
    _setup_plot("Vývoj poloměru sousedství (Radius)", "Iterace", "Hodnota")
    radius_values = [som.get_decay_value(i, total_iterations, som.start_radius, som.end_radius, som.radius_decay_type)
                     for i in iterations_axis]
    plt.plot(iterations_axis, radius_values, color='purple', label=f"Typ: {som.radius_decay_type}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "radius_decay.png"), dpi=150)
    plt.close()

    # Batch Percent
    batch_size_history = training_results.get('batch_size_history', [])
    if batch_size_history:
        _setup_plot("Vývoj počtu zpracovaných vzorků", "Iterace", "Počet vzorků")

        iterations_axis = np.arange(len(batch_size_history))

        plt.plot(iterations_axis, batch_size_history, color='orange', drawstyle='steps-mid',
                 label=f"Skutečný počet (typ: {som.batch_growth_type if som.processing_type == 'hybrid' else som.processing_type})")

        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "batch_size_growth.png"), dpi=150)

        plt.close()

    print("INFO: Grafy z tréninku úspěšně vygenerovány.")