import argparse
import os
import sys
from datetime import datetime
import pandas as pd

from preprocess import validate_input_data, preprocess_data
from utils import load_configuration, log_message, \
    get_working_directory
from som import KohonenSOM

def main():
    parser = argparse.ArgumentParser(description='SOM algorithm')
    parser.add_argument('-i', '--input', required=True, help='Path to the input CSV file')
    parser.add_argument('-c', '--config', required=True, help='Path to a custom configuration file (JSON)')
    parser.add_argument('-o', '--output',
                        help='Optional: Path to the output directory. If not provided, a timestamped folder will be created next to the input CSV file.')
    args = parser.parse_args()

    working_dir = None
    try:
        if args.output:
            working_dir = args.output
            os.makedirs(working_dir, exist_ok=True)
            print(f"INFO: Using specified output directory: '{working_dir}'")
        else:
            input_base_dir = os.path.dirname(os.path.abspath(args.input))
            working_dir = get_working_directory(args.input)
            print(f"INFO: Created default output directory: '{working_dir}'")

        log_message(working_dir, "SYSTEM", f"Application started with input: '{args.input}', config: '{args.config}'")
        log_message(working_dir, "SYSTEM", f"Output directory: '{working_dir}'")

        config = load_configuration(args.config)
        log_message(working_dir, "SYSTEM", f"Configuration file '{args.config}' loaded successfully.")

        config['input_file_name'] = os.path.basename(args.input)

        log_message(working_dir, "SYSTEM", "Starting data validation and preprocessing...")

        input_data_df = validate_input_data(args.input, working_dir, config) # TODO project settings like delimiter, selected columns etc. update config

        log_message(working_dir, "SYSTEM",
                    f"Input data '{args.input}' validated successfully. Shape: {input_data_df.shape}")

        normalized_output_path, normalized_df, categorical_info = \
            preprocess_data(input_data_df, config, working_dir)

        log_message(working_dir, "SYSTEM", f"Data preprocessed and normalized. Saved to '{normalized_output_path}'.")
        log_message(working_dir, "SYSTEM", f"Normalized data shape: {normalized_df.shape}")

        log_message(working_dir, "SYSTEM", "Initializing and training SOM...")

        som_params = {**config}

        if 'map_size' in som_params and isinstance(som_params['map_size'], list):
            map_width, map_height = som_params['map_size'][0] if isinstance(som_params['map_size'][0], list) else \
            som_params['map_size']
        else:
            map_width, map_height = (10, 10) # TODO change to auto calculation based on data size

        som_params['m'] = map_width
        som_params['n'] = map_height
        som_params['dim'] = normalized_df.shape[1]

        som_params.pop('map_size', None)
        som_params.pop('input_file_name', None)
        som_params.pop('delimiter', None)
        som_params.pop('categorical_threshold_numeric', None)

        print(som_params)
        # {'start_learning_rate': 0.9, 'end_learning_rate': 0.1, 'lr_decay_type': 'linear-drop',
        #  'start_radius_init_ratio': None, 'end_radius': 1, 'radius_decay_type': 'linear-drop',
        #  'start_batch_percent': 0.1, 'end_batch_percent': 5.0, 'batch_growth_type': 'exp-growth',
        #  'epoch_multiplier': 50.0, 'normalize_weights_flag': False, 'growth_g': 15.0, 'random_seed': 42,
        #  'map_type': 'hex', 'num_batches': 10, 'max_epochs_without_improvement': 500, 'm': 10, 'n': 10, 'dim': 6}

        som = KohonenSOM(**som_params)

        training_results = som.train(normalized_df.values)

        log_message(working_dir, "SYSTEM", f"SOM training completed. Best MQE: {training_results['final_mqe']:.6f}")

        log_message(working_dir, "SYSTEM", "Generating visualizations... (TODO)")
        # vizualization.generate_map_plots(som, original_data, normalized_data, working_dir, config['PREPROCESS_INFO'])

        # log_final_best(uid, som_config, training_results['final_mqe'], training_results['training_duration'])
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


if __name__ == "__main__":
    main()