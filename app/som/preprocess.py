import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import numpy as np
import json
from utils import log_message

def _clean_dataframe_boundaries(df: pd.DataFrame, working_dir: str) -> pd.DataFrame:
    # Remove completely empty rows from the dataframe
    initial_rows = df.shape[0]
    df_cleaned = df.dropna(how='all')
    if df_cleaned.shape[0] < initial_rows:
        log_message(working_dir, "SYSTEM", f"Removed {initial_rows - df_cleaned.shape[0]} empty rows from input data.")
    return df_cleaned

def _read_csv_robust(input_path: str, working_dir: str, delimiter: str = ',', **kwargs) -> pd.DataFrame:
    # Robust CSV reading with error handling and logging
    try:
        df = pd.read_csv(input_path, delimiter=delimiter, skipinitialspace=True, skip_blank_lines=True, **kwargs)
        log_message(working_dir, "SYSTEM", f"CSV file '{input_path}' loaded successfully.")
        df = _clean_dataframe_boundaries(df, working_dir)
        if df.empty:
            raise ValueError("CSV file is empty or contains only headers after cleaning.")
        return df
    except pd.errors.EmptyDataError:
        log_message(working_dir, "ERROR", f"CSV file '{input_path}' is empty.")
        raise ValueError("CSV file is empty.")
    except pd.errors.ParserError as e:
        log_message(working_dir, "ERROR", f"CSV parsing error: {e}. Check delimiter and format.")
        raise ValueError(f"Error parsing CSV file: {e}. Check delimiter and format.")
    except Exception as e:
        log_message(working_dir, "ERROR", f"General error while reading CSV: {e}")
        raise ValueError(f"General error while reading CSV: {e}")

def validate_input_data(input_path: str, working_dir: str, config_settings: dict) -> pd.DataFrame:
    # Validate existence and structure of input data
    if not os.path.exists(input_path):
        log_message(working_dir, "ERROR", f"Input file '{input_path}' not found.")
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

    try:
        df = _read_csv_robust(input_path, working_dir, delimiter=config_settings.get("delimiter", ","))
    except ValueError as e:
        log_message(working_dir, "ERROR", f"Validation error for file '{input_path}': {e}")
        raise ValueError(f"Validation error for file '{input_path}': {e}")

    if df.empty:
        log_message(working_dir, "ERROR", f"Input file '{input_path}' is empty or contains no valid data.")
        raise ValueError(f"Input file '{input_path}' is empty or contains no valid data.")

    selected_columns = config_settings.get("selected_columns")
    if selected_columns:
        missing_columns = [col for col in selected_columns if col not in df.columns]
        if missing_columns:
            log_message(working_dir, "ERROR", f"Input file '{input_path}' is missing required columns: {', '.join(missing_columns)}")
            raise ValueError(f"Input file '{input_path}' is missing required columns: {', '.join(missing_columns)}")
        df = df[selected_columns].copy()
        log_message(working_dir, "SYSTEM", f"Selected columns for processing: {selected_columns}")

    log_message(working_dir, "SYSTEM", f"Input data validation completed for '{input_path}'.")
    return df

def preprocess_data(df: pd.DataFrame, config_settings: dict, working_dir: str) -> tuple[str, pd.DataFrame, dict]:
    # Preprocess and normalize input data for SOM training
    data = df.copy()
    log_message(working_dir, "SYSTEM", "Starting preprocessing and normalization of input data.")

    # Fill missing values in numerical columns with median
    for col in data.select_dtypes(include=np.number).columns:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            log_message(working_dir, "SYSTEM", f"Filled NaN in numerical column '{col}' with median ({median_val}).")

    # Fill missing values in categorical/text columns with empty string
    for col in data.select_dtypes(include=['object']).columns:
        if data[col].isnull().any():
            data[col] = data[col].fillna("")
            log_message(working_dir, "SYSTEM", f"Filled NaN in text/categorical column '{col}' with empty string.")

    processed = pd.DataFrame()
    categorical_info = {'categorical_column': [], 'numerical_column': [], 'string_column': [], 'categorical_groups': {}}

    # Encode columns and categorize them for SOM
    for col in data.columns:
        series = data[col]

        # Special case for ID column
        if col.startswith('id_') or col.endswith('_id') or col == config_settings.get('primary_id'):
            processed[col], _ = pd.factorize(series.fillna(""), sort=True)
            categorical_info['categorical_column'].append(col)
            log_message(working_dir, "SYSTEM", f"Encoded primary ID column '{col}' as categorical.")
            continue

        n_uniques = series.nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(series):
            # Treat numeric columns with few unique values as categorical
            if n_uniques <= config_settings.get("categorical_threshold_numeric", 30):
                processed[col], _ = pd.factorize(series, sort=True)
                categorical_info['categorical_column'].append(col)
                log_message(working_dir, "SYSTEM", f"Encoded numeric column '{col}' as categorical (unique values: {n_uniques}).")
            else:
                processed[col] = series
                categorical_info['numerical_column'].append(col)
                log_message(working_dir, "SYSTEM", f"Numeric column '{col}' kept as continuous (unique values: {n_uniques}).")
        else:
            # Treat text/object columns with few unique values as categorical
            if n_uniques <= config_settings.get("categorical_threshold_text", 30):
                processed[col], _ = pd.factorize(series, sort=True)
                categorical_info['categorical_column'].append(col)
                log_message(working_dir, "SYSTEM", f"Encoded text column '{col}' as categorical (unique values: {n_uniques}).")
                # ... (grouping code if needed) ...
            else:
                processed[col], _ = pd.factorize(series, sort=True)
                categorical_info['string_column'].append(col)
                log_message(working_dir, "SYSTEM", f"Encoded text column '{col}' as string (unique values: {n_uniques}).")

    # Remove primary_id from numerical and categorical lists
    primary_id = config_settings.get('primary_id')
    if primary_id:
        categorical_info['numerical_column'] = [c for c in categorical_info['numerical_column'] if c != primary_id]
        categorical_info['categorical_column'] = [c for c in categorical_info['categorical_column'] if c != primary_id]
        log_message(working_dir, "SYSTEM", f"Primary ID column '{primary_id}' excluded from scaling columns.")

    # Scale all columns (except primary_id) to [0,1] range
    cols_for_scaling = [c for c in processed.columns if c != primary_id]

    if not cols_for_scaling:
        log_message(working_dir, "ERROR", "No columns available for scaling after removing primary_id.")
        raise ValueError("No columns available for scaling after removing primary_id.")

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(processed[cols_for_scaling])
    log_message(working_dir, "SYSTEM", f"Scaled columns: {cols_for_scaling}")

    # Create final normalized DataFrame
    normalized_df = pd.DataFrame(scaled_values, columns=cols_for_scaling)
    # Add primary_id back (unscaled) if present
    if primary_id and primary_id in processed.columns:
        normalized_df.insert(0, primary_id, processed[primary_id])
        log_message(working_dir, "SYSTEM", f"Primary ID column '{primary_id}' added back to normalized data.")

    # Save normalized data to output file
    output_filename = "normalized_" + os.path.basename(config_settings.get("input_file_name", "data.csv"))
    output_path = os.path.join(working_dir, output_filename)
    normalized_df.to_csv(output_path, index=False, sep=',')
    log_message(working_dir, "SYSTEM", f"Normalized data saved to '{output_path}'.")

    config_settings.update(categorical_info)
    log_message(working_dir, "SYSTEM", "Preprocessing and normalization completed successfully.")

    return output_path, normalized_df, categorical_info