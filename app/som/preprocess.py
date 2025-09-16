import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import numpy as np
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


def preprocess_data(df: pd.DataFrame, config_settings: dict, working_dir: str) -> tuple[str, pd.DataFrame, dict]:
    data = df.copy()

    for col in data.select_dtypes(include=np.number).columns:
        if data[col].isnull().any():
            median_val = data[col].median()
            data[col] = data[col].fillna(median_val)
            log_message(working_dir, "SYSTEM", f"Filled NaN in numerical column '{col}' with median ({median_val}).")

    for col in data.select_dtypes(include=['object']).columns:
        if data[col].isnull().any():
            data[col] = data[col].fillna("")
            log_message(working_dir, "SYSTEM", f"Filled NaN in text/categorical column '{col}' with empty string.")

    processed = pd.DataFrame()
    categorical_info = {'categorical_column': [], 'numerical_column': [], 'string_column': [], 'categorical_groups': {}}

    for col in data.columns:
        series = data[col]

        # Speciální případ pro ID sloupec
        if col.startswith('id_') or col.endswith('_id') or col == config_settings.get('primary_id'):
            # Zde už by neměly být NaN, ale fillna("") pro jistotu zůstává
            processed[col], _ = pd.factorize(series.fillna(""), sort=True)
            categorical_info['categorical_column'].append(col)
            continue

        n_uniques = series.nunique(dropna=True)

        if pd.api.types.is_numeric_dtype(series):
            # Číselné sloupce s malým počtem unikátních hodnot jsou kategorické
            if n_uniques <= config_settings.get("categorical_threshold_numeric", 30):
                processed[col], _ = pd.factorize(series, sort=True)
                categorical_info['categorical_column'].append(col)
            else:
                # Číselné sloupce jsou již čisté (bez NaN)
                processed[col] = series
                categorical_info['numerical_column'].append(col)
        else:  # Textové/objektové sloupce
            # Zde už by neměly být NaN
            if n_uniques <= config_settings.get("categorical_threshold_text", 30):
                processed[col], _ = pd.factorize(series, sort=True)
                categorical_info['categorical_column'].append(col)
                # ... (kód pro seskupení) ...
            else:
                processed[col], _ = pd.factorize(series, sort=True)
                categorical_info['string_column'].append(col)

    # Odstranění primary_id z numerických a kategorických sloupců
    primary_id = config_settings.get('primary_id')
    if primary_id:
        categorical_info['numerical_column'] = [c for c in categorical_info['numerical_column'] if c != primary_id]
        categorical_info['categorical_column'] = [c for c in categorical_info['categorical_column'] if c != primary_id]

    # Škálování všech sloupců do rozsahu [0,1]
    # Ujistíme se, že pracujeme jen se sloupci, které mají být v SOM
    cols_for_scaling = [c for c in processed.columns if c != primary_id]

    if not cols_for_scaling:
        raise ValueError("No columns available for scaling after removing primary_id.")

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(processed[cols_for_scaling])

    # Vytvoření finálního DataFrame
    normalized_df = pd.DataFrame(scaled_values, columns=cols_for_scaling)
    # Pokud existuje primary_id, přidáme ho zpět (neškálovaný)
    if primary_id and primary_id in processed.columns:
        normalized_df.insert(0, primary_id, processed[primary_id])

    # Uložení normalizovaných dat
    output_filename = "normalized_" + os.path.basename(config_settings.get("input_file_name", "data.csv"))
    output_path = os.path.join(working_dir, output_filename)
    normalized_df.to_csv(output_path, index=False, sep=',')

    config_settings.update(categorical_info)

    return output_path, normalized_df, categorical_info