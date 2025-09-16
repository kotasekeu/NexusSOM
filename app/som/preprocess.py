import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
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
    # Read CSV file with error handling and logging
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

def preprocess_data(df: pd.DataFrame, config: dict, working_dir: str) -> tuple[str, pd.DataFrame, np.ndarray]:
    """
    Main function for data preprocessing.
    """
    log_message(working_dir, "SYSTEM", "--- Starting Data Preprocessing ---")

    # Prepare output directories and save original input
    csv_dir = os.path.join(working_dir, "csv")
    json_dir = os.path.join(working_dir, "json")
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    df.to_csv(os.path.join(csv_dir, "original_input.csv"), index=False)

    analysis_df = df.copy()
    total_rows = len(analysis_df)
    primary_id_col = config.get('primary_id', 'primary_id')

    # Analyze columns and generate metadata
    preprocessing_info = {}
    cols_to_ignore = []

    for col in analysis_df.columns:
        series = analysis_df[col]
        nunique = series.nunique()
        nunique_ratio = nunique / total_rows if total_rows > 0 else 0

        col_info = {'status': 'used', 'reason': ''}

        col_info.update({'type': str(series.dtype), 'nunique': nunique, 'nunique_ratio': nunique_ratio})

        if pd.api.types.is_numeric_dtype(series):
            col_info['base_type'] = 'numeric'
            col_info['is_categorical'] = nunique <= config.get('categorical_threshold_numeric', 30)
        else:
            col_info['base_type'] = 'text'
            if nunique_ratio > config.get('noise_threshold_ratio', 0.2):
                col_info.update({'status': 'ignored', 'reason': 'High cardinality / noise', 'is_noise': True})
                cols_to_ignore.append(col)
            else:
                col_info['is_categorical'] = nunique <= config.get('categorical_threshold_text', 30)

        preprocessing_info[col] = col_info

    log_message(working_dir, "SYSTEM", f"Columns ignored as noise: {cols_to_ignore}")

    info_path = os.path.join(json_dir, "preprocessing_info.json")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(preprocessing_info, f, indent=2, ensure_ascii=False, default=str)
    log_message(working_dir, "SYSTEM", f"Preprocessing analysis saved to '{info_path}'")

    # Create training dataframe by excluding ignored columns
    cols_for_training = [col for col in analysis_df.columns if col not in cols_to_ignore]
    training_df = analysis_df[cols_for_training].copy()

    # Create and save ignore mask for NaN values and primary ID column
    ignore_mask = training_df.isnull()

    if primary_id_col and primary_id_col in training_df.columns:
        ignore_mask[primary_id_col] = True
        log_message(working_dir, "SYSTEM", f"Primary ID column '{primary_id_col}' marked to be ignored in the mask.")

    ignore_mask_np = ignore_mask.values

    mask_path = os.path.join(csv_dir, "ignore_mask.csv")
    pd.DataFrame(ignore_mask_np).to_csv(mask_path, index=False, header=False)
    log_message(working_dir, "SYSTEM",
                f"Created and saved ignore mask to '{mask_path}' ({ignore_mask_np.sum()} marked values).")

    # Fill missing values in training dataframe
    fill_values = {}
    for col in training_df.columns:
        if training_df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(training_df[col]):
                fill_values[col] = training_df[col].median()
            else:
                fill_values[col] = ""
    if fill_values:
        training_df.fillna(value=fill_values, inplace=True)

    # Encode categorical columns to numeric values
    encoded_df = pd.DataFrame()
    for col in training_df.columns:
        if pd.api.types.is_numeric_dtype(training_df[col]):
            encoded_df[col] = training_df[col]
        else:
            encoded_df[col], _ = pd.factorize(training_df[col], sort=True)

    # Normalize encoded data and save results
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(encoded_df)

    npy_path = os.path.join(csv_dir, "training_data.npy")
    np.save(npy_path, scaled_values)
    log_message(working_dir, "SYSTEM", f"Normalized training data saved to '{npy_path}'")

    readable_csv_path = os.path.join(csv_dir, "training_data_readable.csv")
    pd.DataFrame(scaled_values).to_csv(readable_csv_path, index=False, header=False)
    log_message(working_dir, "SYSTEM", f"Readable training data saved to '{readable_csv_path}'")

    config.update({'preprocessing_info': preprocessing_info})

    log_message(working_dir, "SYSTEM", "--- Data Preprocessing Finished ---")

    # Return path to .npy, original dataframe, and final ignore mask
    return npy_path, df, ignore_mask_np