import argparse
import os
import sys
from datetime import datetime
import pandas as pd

from preprocess import validate_input_data, preprocess_data
from utils import load_configuration, log_message, \
    get_working_directory
# from kohonen import KohonenSOM

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


if __name__ == "__main__":
    main()