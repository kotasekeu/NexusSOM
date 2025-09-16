import argparse
import os
import sys
from datetime import datetime
import pandas as pd
from analysis import perform_analysis
from graphs import generate_training_plots
from visualization import generate_all_maps
import numpy as np

from preprocess import validate_input_data, preprocess_data
from utils import load_configuration, log_message, \
    get_working_directory
from som import KohonenSOM

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
        training_data_path, _, ignore_mask = preprocess_data(input_data_df, config, working_dir)
        log_message(working_dir, "SYSTEM", "Data preprocessing completed.")

        training_data = np.load(training_data_path)
        log_message(working_dir, "SYSTEM",
                    f"Training data loaded from '{training_data_path}'. Shape: {training_data.shape}")

        # SOM initialization and training
        log_message(working_dir, "SYSTEM", "Initializing and training SOM...")
        som_params = {**config, 'dim': training_data.shape[1]}

        som = KohonenSOM(**som_params)

        # Train SOM
        log_message(working_dir, "SYSTEM", "Starting SOM training...")
        training_results = som.train(training_data, ignore_mask=ignore_mask, working_dir=working_dir)

        log_message(working_dir, "SYSTEM", f"SOM training completed. Best MQE: {training_results['best_mqe']:.6f}")

        # Analysis phase
        log_message(working_dir, "SYSTEM", "Starting analysis phase...")
        perform_analysis(som, input_data_df, training_data, config, working_dir)
        log_message(working_dir, "SYSTEM", "Analysis phase completed.")

        # Generate training plots
        log_message(working_dir, "SYSTEM", "Generating training plots...")
        generate_training_plots(training_results, working_dir)
        log_message(working_dir, "SYSTEM", "Training plots generated.")

        # Generate all SOM visualizations
        log_message(working_dir, "SYSTEM", "Generating SOM visualizations...")
        generate_all_maps(som, input_data_df, training_data, config, working_dir)
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
