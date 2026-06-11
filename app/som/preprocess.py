"""
Input preprocessing stage for the SOM pipeline.

`preprocess_data` is pure: it takes a dataframe + config and returns a
PreprocessResult (arrays + metadata). It writes nothing to disk and does not
mutate the config — persistence of artifacts is handled by
som.persistence.save_preprocess_artifacts, orchestration by som/run.py.
"""
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from som.utils import log_message


@dataclass
class PreprocessResult:
    """Output contract of the preprocessing stage."""
    training_data: np.ndarray      # normalized (N, dim) matrix for SOM training
    ignore_mask: np.ndarray        # boolean (N, dim); True = dimension invisible
    preprocessing_info: dict       # per-column classification and reasons
    dataset_stats: dict            # ds_* statistics (NN training features)
    numerical_column: list         # analyzable numeric feature names (no primary ID)
    categorical_column: list       # analyzable categorical feature names (no primary ID)

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

def _compute_dataset_stats(
    analysis_df: pd.DataFrame,
    numerical_column: list,
    categorical_column: list,
    cols_to_ignore: list,
    cols_for_training: list,
    training_df: pd.DataFrame,
    primary_id_col: str,
    scaled_values: np.ndarray,
) -> dict:
    n_missing = int(training_df.isnull().sum().sum())
    total_cells = training_df.shape[0] * training_df.shape[1]
    has_primary_id = int(primary_id_col in analysis_df.columns)
    n_active_dims = scaled_values.shape[1] - has_primary_id

    return {
        "ds_n_samples": int(analysis_df.shape[0]),
        "ds_n_original_cols": int(analysis_df.shape[1]),
        "ds_n_training_cols": int(len(cols_for_training)),
        "ds_n_dimensions": int(scaled_values.shape[1]),
        "ds_n_active_dimensions": int(n_active_dims),
        "ds_n_numeric": int(len(numerical_column)),
        "ds_n_categorical": int(len(categorical_column)),
        "ds_n_ignored": int(len(cols_to_ignore)),
        "ds_n_missing_values": n_missing,
        "ds_missing_ratio": round(n_missing / total_cells, 6) if total_cells > 0 else 0.0,
        "ds_has_primary_id": has_primary_id,
    }


PREPROCESS_STRATEGIES = ('nexus', 'scale-only', 'none')


def preprocess_data(df: pd.DataFrame, config: dict, log_fn=None) -> PreprocessResult:
    """
    Analyze, clean, encode, and normalize the input dataframe.

    Pure function: no disk writes, no config mutation. Save artifacts with
    som.persistence.save_preprocess_artifacts(result, df, working_dir).
    log_fn is an optional callback(message) for progress logging.

    The strategy (config key `preprocess_strategy`) controls how much of the
    pipeline is applied — the ablation ladder for the preprocessing component:

      - 'nexus' (default): full pipeline — noise-column exclusion, NaN/primary-ID
        ignore mask, median fill, categorical encoding, MinMax normalization.
      - 'scale-only': minimal sane preparation — keep all columns, no ignore
        mask, median fill, categorical encoding, MinMax normalization.
        Isolates the contribution of the mask + noise exclusion.
      - 'none': raw values — encoding only (SOM needs numbers), no mask,
        no normalization. Isolates the contribution of normalization
        (expected: organization collapses).
    """
    _log = log_fn if log_fn is not None else (lambda message: None)

    strategy = config.get('preprocess_strategy', 'nexus')
    if strategy not in PREPROCESS_STRATEGIES:
        raise ValueError(f"Unknown preprocess_strategy '{strategy}' "
                         f"(expected one of {PREPROCESS_STRATEGIES})")
    exclude_noise = strategy == 'nexus'
    build_mask = strategy == 'nexus'
    normalize = strategy in ('nexus', 'scale-only')

    _log(f"--- Starting Data Preprocessing (strategy: {strategy}) ---")

    analysis_df = df.copy()
    total_rows = len(analysis_df)
    primary_id_col = config.get('primary_id', 'primary_id')

    # Analyze columns and generate metadata
    preprocessing_info = {}
    cols_to_ignore = []

    numerical_column = []
    categorical_column = []

    for col in analysis_df.columns:
        series = analysis_df[col]
        nunique = series.nunique()
        nunique_ratio = nunique / total_rows if total_rows > 0 else 0

        col_info = {'status': 'used', 'reason': ''}

        col_info.update({'type': str(series.dtype), 'nunique': nunique, 'nunique_ratio': nunique_ratio})

        # The primary ID column identifies records — it must not be treated as an
        # analyzable feature (it is also masked for training further below).
        is_primary_id = col == primary_id_col

        if pd.api.types.is_numeric_dtype(series):
            col_info['base_type'] = 'numeric'
            if nunique <= config.get('categorical_threshold_numeric', 30):
                col_info['is_categorical'] = True
                if col_info.get('status') == 'used' and not is_primary_id:
                    categorical_column.append(col)
            else:
                col_info['is_categorical'] = False
                if col_info.get('status') == 'used' and not is_primary_id:
                    numerical_column.append(col)
        else:
            col_info['base_type'] = 'text'
            if exclude_noise and nunique_ratio > config.get('noise_threshold_ratio', 0.2):
                col_info.update({'status': 'ignored', 'reason': 'High cardinality / noise', 'is_noise': True})
                cols_to_ignore.append(col)

            elif nunique <= config.get('categorical_threshold_text', 30):
                col_info['is_categorical'] = True
                if col_info.get('status') == 'used' and not is_primary_id:
                    categorical_column.append(col)
            else:
                col_info['is_categorical'] = False

        preprocessing_info[col] = col_info

    _log(f"Columns ignored as noise: {cols_to_ignore}")

    # Create training dataframe by excluding ignored columns
    cols_for_training = [col for col in analysis_df.columns if col not in cols_to_ignore]
    training_df = analysis_df[cols_for_training].copy()

    # Create ignore mask for NaN values and primary ID column
    if build_mask:
        ignore_mask = training_df.isnull()

        if primary_id_col and primary_id_col in training_df.columns:
            ignore_mask[primary_id_col] = True
            _log(f"Primary ID column '{primary_id_col}' marked to be ignored in the mask.")

        ignore_mask_np = ignore_mask.values
        _log(f"Created ignore mask ({ignore_mask_np.sum()} marked values).")
    else:
        ignore_mask_np = np.zeros(training_df.shape, dtype=bool)
        _log(f"Ignore mask disabled (strategy: {strategy}).")

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

    # Normalize encoded data (skipped entirely for the 'none' strategy)
    if normalize:
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(encoded_df)
    else:
        scaled_values = encoded_df.values.astype(float)
        _log(f"Normalization disabled (strategy: {strategy}).")

    # Fully-masked columns (primary ID, all-NaN columns) never train — zero
    # them in the data so they contribute exactly nothing even to computations
    # that run without the mask. Zero here is NOT a value: the column is inert
    # on both sides (weights of fully-masked dims are zeroed at train start).
    # Linking back to the original dataset is done by row order / the ID in
    # original_input.csv, never through this matrix.
    if build_mask:
        fully_masked_cols = ignore_mask_np.all(axis=0)
        if fully_masked_cols.any():
            scaled_values[:, fully_masked_cols] = 0.0
            _log(f"Zeroed {int(fully_masked_cols.sum())} fully-masked column(s) "
                 f"in the training matrix.")

    # Compute dataset statistics for ML model training
    dataset_stats = _compute_dataset_stats(
        analysis_df=analysis_df,
        numerical_column=numerical_column,
        categorical_column=categorical_column,
        cols_to_ignore=cols_to_ignore,
        cols_for_training=cols_for_training,
        training_df=training_df,
        primary_id_col=primary_id_col,
        scaled_values=scaled_values,
    )
    dataset_stats['ds_preprocess_strategy'] = strategy

    _log("--- Data Preprocessing Finished ---")

    return PreprocessResult(
        training_data=scaled_values,
        ignore_mask=ignore_mask_np,
        preprocessing_info=preprocessing_info,
        dataset_stats=dataset_stats,
        numerical_column=numerical_column,
        categorical_column=categorical_column,
    )