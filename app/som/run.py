import argparse
import os
import sys

from som.analysis import perform_analysis
from som.graphs import generate_training_plots
from som.visualization import generate_all_maps
from som.preprocess import validate_input_data, preprocess_data
from som.persistence import (save_preprocess_artifacts, save_weights,
                             save_training_checkpoints, save_sample_coverage,
                             save_run_metrics)
from analysis.src.context import save_llm_context
from som.utils import load_configuration, log_message, \
    get_working_directory
from som.som import KohonenSOM


def _build_lstm_controller_fn(nn_cfg: dict, dataset_stats: dict):
    """
    Load LSTM Phase 3 controller and return a dynamic_schedule_fn for som.train(), or None.
    nn_cfg must have use_lstm_controller=true and lstm_controller_model_path set.
    """
    if not nn_cfg.get('use_lstm_controller', False):
        return None
    try:
        import sys as _sys
        _app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _app_dir not in _sys.path:
            _sys.path.insert(0, _app_dir)
        from ea.nn_integration import NeuralNetworkIntegration
        nn = NeuralNetworkIntegration(
            use_mlp=False,
            use_lstm=False,
            use_cnn=False,
            use_lstm_controller=True,
            lstm_controller_model_path=nn_cfg.get('lstm_controller_model_path'),
            lstm_controller_scaler_path=nn_cfg.get('lstm_controller_scaler_path'),
        )
        if not nn.can_use_dynamic_schedule():
            print("WARNING: LSTM controller could not be loaded — running without dynamic control")
            return None
        ds_context = [
            dataset_stats.get('ds_n_samples', 0),
            dataset_stats.get('ds_n_active_dimensions', 0),
            dataset_stats.get('ds_n_numeric', 0),
            dataset_stats.get('ds_n_categorical', 0),
        ]
        print("INFO: LSTM Phase 3 controller enabled (dynamic LR + radius adjustment)")
        return nn.get_dynamic_schedule_fn(dataset_context=ds_context)
    except Exception as e:
        print(f"WARNING: LSTM controller init failed ({e}) — running without dynamic control")
        return None


def _build_lstm_early_stop_fn(nn_cfg: dict, dataset_stats: dict):
    """
    Load LSTM model and return an early-stop callback for som.train(), or None.
    nn_cfg must have use_lstm=true and lstm_model_path / lstm_scaler_path set.
    """
    if not nn_cfg.get('use_lstm', False):
        return None
    try:
        import sys as _sys
        _app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _app_dir not in _sys.path:
            _sys.path.insert(0, _app_dir)
        from ea.nn_integration import NeuralNetworkIntegration
        nn = NeuralNetworkIntegration(
            use_mlp=False,
            use_lstm=True,
            use_cnn=False,
            mlp_model_path=None,
            mlp_scaler_path=None,
            lstm_model_path=nn_cfg.get('lstm_model_path'),
            lstm_scaler_path=nn_cfg.get('lstm_scaler_path'),
        )
        if not nn.can_check_early_stopping():
            print("WARNING: LSTM model could not be loaded — running without early stopping")
            return None
        threshold = nn_cfg.get('lstm_quality_threshold', 0.75)
        ds_context = [
            dataset_stats.get('ds_n_samples', 0),
            dataset_stats.get('ds_n_active_dimensions', 0),
            dataset_stats.get('ds_n_numeric', 0),
            dataset_stats.get('ds_n_categorical', 0),
        ]
        print(f"INFO: LSTM early stopping enabled (threshold={threshold})")

        def lstm_early_stop_fn(checkpoints):
            history = {
                'progress':          [c['progress'] for c in checkpoints],
                'mqe':               [c['mqe'] for c in checkpoints],
                'topographic_error': [c['topographic_error'] for c in checkpoints],
                'dead_neuron_ratio': [c['dead_neuron_ratio'] for c in checkpoints],
                'learning_rate':     [c['learning_rate'] for c in checkpoints],
                'radius':            [c['radius'] for c in checkpoints],
            }
            return nn.should_stop_early(history, threshold, dataset_context=ds_context)

        return lstm_early_stop_fn
    except Exception as e:
        print(f"WARNING: LSTM init failed ({e}) — running without early stopping")
        return None


def run_pipeline(input_path: str, config: dict | str, output_dir: str = None,
                 seed: int = None) -> str:
    """
    Programmatic entry point for one full SOM run:
    validate → preprocess → train → analyze → llm context → plots → maps.

    This is the API the multi-seed tool, ablation tooling, and the UI call —
    same pipeline as the CLI, no subprocess needed.

    Args:
        input_path: path to the input CSV file.
        config: configuration dict, or path to a JSON config file.
        output_dir: results directory; default is a timestamped folder
            next to the input CSV (results/<timestamp>).
        seed: overrides config['random_seed'] when given (multi-seed runs).

    Returns the results directory path. Raises on failure.
    """
    if isinstance(config, str):
        config = load_configuration(config)
    config = dict(config)  # local copy — callers' dict stays untouched
    if seed is not None:
        config['random_seed'] = seed

    if output_dir:
        working_dir = output_dir
        os.makedirs(working_dir, exist_ok=True)
        log_message(working_dir, "SYSTEM", f"Using specified output directory: '{working_dir}'")
    else:
        working_dir = get_working_directory(input_path)
        log_message(working_dir, "SYSTEM", f"Created default output directory: '{working_dir}'")

    log_message(working_dir, "SYSTEM",
                f"Pipeline started with input: '{input_path}', seed: {config.get('random_seed')}")

    # Data validation and preprocessing
    log_message(working_dir, "SYSTEM", "Starting data validation and preprocessing...")

    input_data_df = validate_input_data(input_path, working_dir, config)
    log_message(working_dir, "SYSTEM", "Input data loaded for validation.")

    config['num_samples'] = len(input_data_df)

    log_message(working_dir, "SYSTEM",
                f"Input data '{input_path}' validated successfully. Shape: {input_data_df.shape}")

    # Preprocess and normalize data (pure), then persist the artifacts
    log_fn = lambda message: log_message(working_dir, "SYSTEM", message)  # noqa: E731
    pre = preprocess_data(input_data_df, config, log_fn=log_fn)
    save_preprocess_artifacts(pre, input_data_df, working_dir)
    log_message(working_dir, "SYSTEM", "Data preprocessing completed and saved.")

    training_data = pre.training_data
    ignore_mask = pre.ignore_mask
    dataset_stats = pre.dataset_stats

    # Column classification feeds analysis and visualization below.
    # Kept out of the original config object — preprocess no longer mutates it.
    runtime_config = {
        **config,
        'numerical_column': pre.numerical_column,
        'categorical_column': pre.categorical_column,
        'preprocessing_info': pre.preprocessing_info,
    }

    # NN models — loaded from NEURAL_NETWORKS section if present
    nn_cfg = config.get('NEURAL_NETWORKS', {})
    lstm_early_stop_fn  = _build_lstm_early_stop_fn(nn_cfg, dataset_stats)
    dynamic_schedule_fn = _build_lstm_controller_fn(nn_cfg, dataset_stats)

    # SOM initialization and training
    log_message(working_dir, "SYSTEM", "Initializing and training SOM...")

    som_params = {**config, 'dim': training_data.shape[1]}
    som_params.pop('NEURAL_NETWORKS', None)
    som_params.pop('PREPROCES_DATA', None)

    som = KohonenSOM(**som_params)

    # Train SOM (pure compute), then persist training artifacts
    log_message(working_dir, "SYSTEM", "Starting SOM training...")
    training_results = som.train(training_data, ignore_mask=ignore_mask,
                                 lstm_early_stop_fn=lstm_early_stop_fn,
                                 dynamic_schedule_fn=dynamic_schedule_fn,
                                 log_fn=log_fn)
    save_weights(som.weights, working_dir)
    save_training_checkpoints(training_results.get('checkpoints', []), working_dir)
    save_sample_coverage(training_results.get('sample_coverage'), working_dir)

    lstm_stopped = training_results.get('lstm_stopped', False)
    log_message(working_dir, "SYSTEM",
                f"SOM training completed. Best MQE: {training_results['best_mqe']:.6f}"
                + (f" [LSTM early stop at progress={training_results.get('lstm_stop_progress', '?')}]"
                   if lstm_stopped else ""))

    # Save run metrics (topographic_error, map geometry) for downstream analysis
    final_te = som.calculate_topographic_error(training_data, mask=ignore_mask)
    run_metrics = {
        "map_size": [som.m, som.n],
        "map_topology": config.get('map_type', 'hex'),
        "best_mqe": training_results['best_mqe'],
        "topographic_error": final_te,
        "duration": training_results.get('duration'),
        "lstm_stopped": lstm_stopped,
        "lstm_stop_progress": training_results.get('lstm_stop_progress'),
    }
    save_run_metrics(run_metrics, working_dir)

    # Analysis phase
    log_message(working_dir, "SYSTEM", "Starting analysis phase...")
    perform_analysis(som, input_data_df, training_data, runtime_config, working_dir,
                     ignore_mask=ignore_mask)
    log_message(working_dir, "SYSTEM", "Analysis phase completed.")

    # Build LLM context from analysis outputs
    log_message(working_dir, "SYSTEM", "Building LLM context...")
    save_llm_context(working_dir)
    log_message(working_dir, "SYSTEM", "LLM context ready.")

    # Generate training plots (optional — batch contexts may skip them)
    if config.get('save_training_plots', True):
        log_message(working_dir, "SYSTEM", "Generating training plots...")
        generate_training_plots(training_results, working_dir)
        log_message(working_dir, "SYSTEM", "Training plots generated.")

    # Generate all SOM visualizations (optional)
    if config.get('save_visualizations', True):
        log_message(working_dir, "SYSTEM", "Generating SOM visualizations...")
        generate_all_maps(som.weights, som.map_type, input_data_df, training_data,
                          runtime_config, ignore_mask, working_dir)
        log_message(working_dir, "SYSTEM", "SOM visualizations generated.")

    log_message(working_dir, "SYSTEM", "SOM analysis finished successfully.")
    return working_dir


def main():
    # Parse command-line arguments for input, config, and output paths
    parser = argparse.ArgumentParser(description='SOM algorithm')
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file')
    parser.add_argument('-c', '--config', required=True, help='Path to a custom configuration file (JSON)')
    parser.add_argument('-o', '--output',
                        help='Optional: Path to the output directory. If not provided, a timestamped folder will be created next to the input CSV file.')
    parser.add_argument('-s', '--seed', type=int,
                        help='Optional: Override random_seed from the config.')
    args = parser.parse_args()

    try:
        working_dir = run_pipeline(args.input, args.config,
                                   output_dir=args.output, seed=args.seed)
        print(f"INFO: Results saved to '{working_dir}'")
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred: {e}")
        sys.exit(1)


# Entry point for script execution
if __name__ == "__main__":
    main()
