# analysis.py
import sys
import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
from som import KohonenSOM
from scipy.spatial.distance import cdist


def _prepare_pie_map_data(df_assigned: pd.DataFrame, categorical_cols: list) -> dict:
    pie_data = {}
    for col in categorical_cols:
        # Převedeme kategorické hodnoty na standardní stringy hned na začátku
        df_assigned[col] = df_assigned[col].astype(str)

        counts_df = df_assigned.groupby(['bmu_key', col]).size().unstack(fill_value=0)

        all_categories = sorted(df_assigned[col].dropna().unique())
        # Hodnoty v category_map jsou teď zaručeně stringy
        category_map = {str(i + 1): cat for i, cat in enumerate(all_categories)}

        counts_dict = {}
        # Explicitně převedeme index (bmu_key) na string
        for bmu_key_np, row in counts_df.iterrows():
            bmu_key = str(bmu_key_np)

            counts_dict[bmu_key] = {
                # Ujistíme se, že i klíče i hodnoty jsou standardní typy
                str(next(k for k, v in category_map.items() if v == cat_name)): int(count)
                for cat_name, count in row.items()
            }

        pie_data[col] = {
            "categories": category_map,
            "counts": counts_dict
        }
    return pie_data

def _save_quantization_errors(som: KohonenSOM, normalized_data: np.ndarray, working_dir: str):
    neuron_error_map, total_qe = som.compute_quantization_error(normalized_data)

    neuron_errors_dict = {}
    if neuron_error_map is not None:
        for i in range(som.m):
            for j in range(som.n):
                key = f"{i}_{j}"
                neuron_errors_dict[key] = float(neuron_error_map[i, j])

    output_data = {
        "total_quantization_error": float(total_qe),
        "neuron_quantization_errors": neuron_errors_dict
    }

    output_path = os.path.join(working_dir, "quantization_errors.json")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"INFO: Kvantizační chyby uloženy do '{output_path}'")
    except Exception as e:
        print(f"ERROR: Nepodařilo se uložit kvantizační chyby: {e}")

def _get_bmu_assignments(som: KohonenSOM, normalized_data: np.ndarray, original_df: pd.DataFrame,
                         primary_id_col: str) -> pd.DataFrame:

    num_neurons = som.m * som.n
    flat_weights = som.weights.reshape(num_neurons, som.dim)

    dists = cdist(normalized_data, flat_weights, 'euclidean')
    bmu_flat_indices = np.argmin(dists, axis=1)

    bmu_coords_i, bmu_coords_j = np.unravel_index(bmu_flat_indices, (som.m, som.n))

    df_assigned = original_df.copy()
    df_assigned['bmu_i'] = bmu_coords_i
    df_assigned['bmu_j'] = bmu_coords_j
    df_assigned['bmu_key'] = [f"{i}_{j}" for i, j in zip(bmu_coords_i, bmu_coords_j)]

    clusters = df_assigned.groupby('bmu_key')[primary_id_col].apply(list).to_dict()

    return df_assigned, clusters


def _detect_extremes(df_assigned: pd.DataFrame, numerical_cols: list, primary_id_col: str,
                     std_threshold: float) -> dict:
    extremes = defaultdict(list)

    if df_assigned.empty or not numerical_cols:
        return {}

    global_stats = df_assigned[numerical_cols].agg(['mean', 'std', 'min', 'max'])

    for col in numerical_cols:
        mean, std = global_stats[col]['mean'], global_stats[col]['std']
        min_val, max_val = global_stats[col]['min'], global_stats[col]['max']

        if pd.notna(std) and std > 0:
            lower_bound, upper_bound = mean - std_threshold * std, mean + std_threshold * std
            outliers = df_assigned[(df_assigned[col] < lower_bound) | (df_assigned[col] > upper_bound)]
            for _, row in outliers.iterrows():
                reason = f"Hodnota '{col}' ({row[col]:.2f}) je o více než {std_threshold} std. odchylek od globálního průměru ({mean:.2f})."
                extremes[row[primary_id_col]].append(reason)

        min_rows = df_assigned[df_assigned[col] == min_val]
        for _, row in min_rows.iterrows():
            extremes[row[primary_id_col]].append(f"Hodnota '{col}' ({min_val:.2f}) je globální minimum.")

        max_rows = df_assigned[df_assigned[col] == max_val]
        for _, row in max_rows.iterrows():
            extremes[row[primary_id_col]].append(f"Hodnota '{col}' ({max_val:.2f}) je globální maximum.")

    grouped = df_assigned.groupby('bmu_key')[numerical_cols]

    counts = grouped.size()

    neurons_for_local_analysis = counts[counts > 1].index

    if len(neurons_for_local_analysis) > 0:
        local_stats = df_assigned[df_assigned['bmu_key'].isin(neurons_for_local_analysis)] \
            .groupby('bmu_key')[numerical_cols].agg(['mean', 'std'])

        local_stats.columns = ['_'.join(col).strip() for col in local_stats.columns.values]

        df_with_local_stats = df_assigned.join(local_stats, on='bmu_key')

        for col in numerical_cols:
            mean_col_local = f'{col}_mean'
            std_col_local = f'{col}_std'

            mask = (
                    pd.notna(df_with_local_stats[mean_col_local]) &
                    pd.notna(df_with_local_stats[std_col_local]) &
                    (df_with_local_stats[std_col_local] > 0)
            )

            mask_outliers = mask & (
                        np.abs(df_with_local_stats[col] - df_with_local_stats[mean_col_local]) > std_threshold *
                        df_with_local_stats[std_col_local])

            local_outliers = df_with_local_stats[mask_outliers]

            for _, row in local_outliers.iterrows():
                reason = f"Hodnota '{col}' ({row[col]:.2f}) se o více než {std_threshold} std. odchylek liší od průměru svého neuronu ({row[mean_col_local]:.2f})."
                extremes[row[primary_id_col]].append(reason)

    return dict(extremes)


def perform_analysis(som: KohonenSOM, original_df: pd.DataFrame, normalized_data: np.ndarray, config: dict,
                     working_dir: str):
    print("INFO: Spouštím analýzu organizovaných dat...")

    primary_id_col = config.get('primary_id')
    if primary_id_col is None and 'primary_id' in original_df.columns:
        primary_id_col = 'primary_id'
    numerical_cols = config.get('numerical_column', [])
    std_threshold = config.get('std_threshold', 2.5)

    if not primary_id_col or not numerical_cols:
        print("ERROR: V konfiguraci chybí 'primary_id' nebo 'numerical_column' pro analýzu.")
        return

    df_assigned, clusters = _get_bmu_assignments(som, normalized_data, original_df, primary_id_col)

    clusters_path = os.path.join(working_dir, "clusters.json")
    with open(clusters_path, 'w', encoding='utf-8') as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)
    print(f"INFO: Mapování shluků uloženo do '{clusters_path}'")

    _save_quantization_errors(som, normalized_data, working_dir)


    extremes_data = _detect_extremes(df_assigned, numerical_cols, primary_id_col, std_threshold)

    extremes_path = os.path.join(working_dir, "extremes.json")
    with open(extremes_path, 'w', encoding='utf-8') as f:
        json.dump(extremes_data, f, indent=2, ensure_ascii=False)
    print(f"INFO: Analýza extrémů uložena do '{extremes_path}'")

    categorical_cols = config.get('categorical_column', [])
    if categorical_cols:
        pie_map_data = _prepare_pie_map_data(df_assigned, categorical_cols)
        for col, data in pie_map_data.items():
            pie_data_path = os.path.join(working_dir, f"pie_data_{col}.json")
            with open(pie_data_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"INFO: Data pro koláčové mapy uložena.")

    print("INFO: Analýza dat dokončena.")