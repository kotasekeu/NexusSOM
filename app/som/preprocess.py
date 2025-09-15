import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import json
from utils import log_message

def _clean_dataframe_boundaries(df: pd.DataFrame, working_dir: str) -> pd.DataFrame:
    initial_rows = df.shape[0]
    df_cleaned = df.dropna(how='all')
    if df_cleaned.shape[0] < initial_rows:
        log_message(working_dir,"SYSTEM", f"Removed {initial_rows - df_cleaned.shape[0]} empty rows from data.")
        pass
    return df_cleaned


def _read_csv_robust(input_path: str, working_dir: str, delimiter: str = ',', **kwargs) -> pd.DataFrame:
    try:
        df = pd.read_csv(input_path, delimiter=delimiter, skipinitialspace=True, skip_blank_lines=True, **kwargs)

        df = _clean_dataframe_boundaries(df, working_dir)

        if df.empty:
            raise ValueError("CSV soubor je prázdný nebo obsahuje pouze hlavičky po čištění.")

        return df
    except pd.errors.EmptyDataError:
        raise ValueError("CSV soubor je prázdný.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Chyba při parsování CSV souboru: {e}. Zkontrolujte oddělovač a formát.")
    except Exception as e:
        raise ValueError(f"Obecná chyba při čtení CSV: {e}")


def validate_input_data(input_path: str, working_dir: str, config_settings: dict) -> pd.DataFrame:

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Vstupní soubor '{input_path}' nebyl nalezen.")

    try:
        df = _read_csv_robust(input_path, working_dir, delimiter=config_settings.get("delimiter", ","))
    except ValueError as e:
        raise ValueError(f"Chyba při validaci souboru '{input_path}': {e}")

    if df.empty:
        raise ValueError(f"Vstupní soubor '{input_path}' je prázdný nebo neobsahuje žádná platná data.")

    selected_columns = config_settings.get("selected_columns")
    if selected_columns:
        missing_columns = [col for col in selected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Vstupní soubor '{input_path}' postrádá požadované sloupce: {', '.join(missing_columns)}")
        df = df[selected_columns].copy()

    return df

def preprocess_data(df: pd.DataFrame, config_settings: dict, working_dir: str) -> str:
    data = df.copy()

    # TODO - implement replacement for empty values
    processed = pd.DataFrame()
    categorical_info = {'categorical_column': [], 'numerical_column': [], 'string_column': [], 'categorical_groups': {}}

    for col in data.columns:
        series = data[col]

        if col.startswith('id_') or col.endswith('_id') or col == config_settings.get('primary_id'):
            processed[col] = pd.factorize(series.fillna(""), sort=True)[0]
            categorical_info['categorical_column'].append(col)
            log_message(working_dir,"SYSTEM", f"Column '{col}': identified as categorical (ID column).")
            continue

        n_uniques = series.nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(series):
            if n_uniques <= config_settings.get("categorical_threshold_numeric", 30):
                processed[col] = pd.factorize(series.fillna(""), sort=True)[0]
                categorical_info['categorical_column'].append(col)
                log_message(working_dir,"SYSTEM", f"Column '{col}': identified as categorical (numerical with {n_uniques} unique values).")
            else:
                num = pd.to_numeric(series, errors="coerce")
                processed[col] = num
                categorical_info['numerical_column'].append(col)
                log_message(working_dir,"SYSTEM", f"Column '{col}': identified as numerical ({n_uniques} unique values).")
        else:
            if n_uniques <= config_settings.get("categorical_threshold_text", 30):
                processed[col] = pd.factorize(series.fillna(""), sort=True)[0]
                categorical_info['categorical_column'].append(col)
                log_message(working_dir,"SYSTEM", f"Column '{col}': identified as categorical (text with {n_uniques} unique values).")

                parts = col.split('_')
                if len(parts) > 1:
                    prefix = parts[0]
                    if prefix not in categorical_info['categorical_groups']:
                        categorical_info['categorical_groups'][prefix] = []
                    categorical_info['categorical_groups'][prefix].append(col)
            else:
                processed[col] = pd.factorize(series.fillna(""), sort=True)[0]
                categorical_info['string_column'].append(col)
                log_message(working_dir,"SYSTEM", f"Column '{col}': identified as string/high-cardinality categorical ({n_uniques} unique values).")

    primary_id_col = config_settings.get('primary_id')
    if primary_id_col in processed.columns:
        id_data = processed[[primary_id_col]].copy()
        processed = processed.drop(columns=[primary_id_col])

        if primary_id_col in categorical_info['categorical_column']:
            categorical_info['categorical_column'].remove(primary_id_col)
        elif primary_id_col in categorical_info['numerical_column']:
            categorical_info['numerical_column'].remove(primary_id_col)
        elif primary_id_col in categorical_info['string_column']:
            categorical_info['string_column'].remove(primary_id_col)
    else:
        id_data = None

    if not processed.empty:
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(processed.values)
        normalized_df = pd.DataFrame(scaled_values, columns=processed.columns)
        log_message(working_dir,"SYSTEM", f"Data normalized using MinMaxScaler.")
    else:
        normalized_df = pd.DataFrame()
        log_message(working_dir,"SYSTEM", "No numerical or categorical data to normalize.")

    if id_data is not None:
        # TODO - check if primary_id is not calculated in SOM analysis
        normalized_df = pd.concat([id_data, normalized_df], axis=1)
        log_message(working_dir,"SYSTEM", f"Primary ID '{primary_id_col}' re-attached to normalized data.")


    output_filename = "normalized_" + os.path.basename(config_settings.get("input_file_name", "data.csv"))
    output_path = os.path.join(working_dir, output_filename)
    normalized_df.to_csv(output_path, index=False, sep=',')
    log_message(working_dir,"SYSTEM", f"Normalized data saved to '{output_path}'.")

    categorical_info_path = os.path.join(working_dir, "column_info.json")
    with open(categorical_info_path, "w", encoding="utf-8") as f:
        json.dump(categorical_info, f, ensure_ascii=False, indent=2)

    return output_path, normalized_df, categorical_info