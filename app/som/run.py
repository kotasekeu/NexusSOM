import argparse
import json
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

from som.analysis import perform_analysis
from som.graphs import generate_training_plots
from som.visualization import generate_all_maps
from som.preprocess import validate_input_data, preprocess_data
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


def main():
    # Parse command-line arguments for input, config, and output paths
    parser = argparse.ArgumentParser(description='SOM algorithm')
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file')
    parser.add_argument('-c', '--config', required=True, help='Path to a custom configuration file (JSON)')
    parser.add_argument('-o', '--output',
                        help='Optional: Path to the output directory. If not provided, a timestamped folder will be created next to the input CSV file.')
    args = parser.parse_args()

    working_dir = None
    try:
        # Determine output directory, create if necessary
        if args.output:
            working_dir = args.output
            os.makedirs(working_dir, exist_ok=True)
            print(f"INFO: Using specified output directory: '{working_dir}'")
            log_message(working_dir, "SYSTEM", f"Using specified output directory: '{working_dir}'")
        else:
            input_base_dir = os.path.dirname(os.path.abspath(args.input))
            working_dir = get_working_directory(args.input)
            print(f"INFO: Created default output directory: '{working_dir}'")
            log_message(working_dir, "SYSTEM", f"Created default output directory: '{working_dir}'")

        # Log application start
        log_message(working_dir, "SYSTEM", f"Application started with input: '{args.input}', config: '{args.config}'")
        log_message(working_dir, "SYSTEM", f"Output directory: '{working_dir}'")

        # Load configuration file
        config = load_configuration(args.config)
        log_message(working_dir, "SYSTEM", f"Configuration file '{args.config}' loaded successfully.")

        # Data validation and preprocessing
        log_message(working_dir, "SYSTEM", "Starting data validation and preprocessing...")

        input_data_df = validate_input_data(args.input, working_dir, config) # TODO project settings like delimiter, selected columns etc. update config
        log_message(working_dir, "SYSTEM", "Input data loaded for validation.")

        config['num_samples'] = len(input_data_df)

        log_message(working_dir, "SYSTEM",
                    f"Input data '{args.input}' validated successfully. Shape: {input_data_df.shape}")

        # Preprocess and normalize data
        training_data_path, _, ignore_mask, dataset_stats = preprocess_data(input_data_df, config, working_dir)
        log_message(working_dir, "SYSTEM", "Data preprocessing completed.")

        training_data = np.load(training_data_path)
        log_message(working_dir, "SYSTEM",
                    f"Training data loaded from '{training_data_path}'. Shape: {training_data.shape}")

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

        # Train SOM
        log_message(working_dir, "SYSTEM", "Starting SOM training...")
        training_results = som.train(training_data, ignore_mask=ignore_mask, working_dir=working_dir,
                                     lstm_early_stop_fn=lstm_early_stop_fn,
                                     dynamic_schedule_fn=dynamic_schedule_fn)

        lstm_stopped = training_results.get('lstm_stopped', False)
        log_message(working_dir, "SYSTEM",
                    f"SOM training completed. Best MQE: {training_results['best_mqe']:.6f}"
                    + (f" [LSTM early stop at progress={training_results.get('lstm_stop_progress', '?')}]"
                       if lstm_stopped else ""))

        # Save run metrics for result_analyzer (topographic_error, map geometry)
        checkpoints = training_results.get('checkpoints', [])
        final_te = checkpoints[-1]['topographic_error'] if checkpoints else None
        run_metrics = {
            "map_size": [som.m, som.n],
            "map_topology": config.get('map_type', 'hex'),
            "best_mqe": training_results['best_mqe'],
            "topographic_error": final_te,
            "duration": training_results.get('duration'),
            "lstm_stopped": lstm_stopped,
            "lstm_stop_progress": training_results.get('lstm_stop_progress'),
        }
        with open(os.path.join(working_dir, "run_metrics.json"), 'w') as f:
            json.dump(run_metrics, f, indent=2)

        # Analysis phase
        log_message(working_dir, "SYSTEM", "Starting analysis phase...")
        perform_analysis(som, input_data_df, training_data, config, working_dir)
        log_message(working_dir, "SYSTEM", "Analysis phase completed.")

        # Build LLM context from analysis outputs
        log_message(working_dir, "SYSTEM", "Building LLM context...")
        save_llm_context(working_dir)
        log_message(working_dir, "SYSTEM", "LLM context ready.")

        # Generate training plots
        log_message(working_dir, "SYSTEM", "Generating training plots...")
        generate_training_plots(training_results, working_dir)
        log_message(working_dir, "SYSTEM", "Training plots generated.")

        # Generate all SOM visualizations
        log_message(working_dir, "SYSTEM", "Generating SOM visualizations...")
        generate_all_maps(som, input_data_df, training_data, config, ignore_mask, working_dir)
        log_message(working_dir, "SYSTEM", "SOM visualizations generated.")

        log_message(working_dir, "SYSTEM", "SOM analysis finished successfully.")

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        if working_dir:
            log_message(working_dir, "ERROR", str(e))
        sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        if working_dir:
            log_message(working_dir, "ERROR", str(e))
        sys.exit(1)
    except Exception as e:
        print(f"FATAL ERROR: An unexpected error occurred: {e}")
        if working_dir:
            log_message(working_dir, "FATAL ERROR", str(e))
        sys.exit(1)

# Entry point for script execution
if __name__ == "__main__":
    main()
