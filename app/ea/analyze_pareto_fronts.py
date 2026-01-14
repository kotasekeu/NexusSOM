#!/usr/bin/env python3
"""
Analyze Pareto Fronts from Multiple EA Runs

This script compares Pareto fronts across multiple EA runs to identify
the best configurations and understand evolutionary trends.

Usage:
    python3 analyze_pareto_fronts.py --results_base ./test/results

Features:
- Loads all Pareto fronts from results directories
- Compares quality metrics across runs
- Identifies best overall configurations
- Generates comparison visualizations
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def find_result_directories(base_path: str) -> list:
    """Find all EA result directories (YYYYMMDD_HHMMSS format)"""
    result_dirs = []

    if not os.path.exists(base_path):
        print(f"Error: Base path does not exist: {base_path}")
        return result_dirs

    for entry in os.listdir(base_path):
        full_path = os.path.join(base_path, entry)
        if os.path.isdir(full_path):
            # Check if it matches YYYYMMDD_HHMMSS pattern
            if len(entry) == 15 and entry[8] == '_':
                # Check if pareto_front.csv exists
                pareto_file = os.path.join(full_path, "pareto_front.csv")
                if os.path.exists(pareto_file):
                    result_dirs.append(full_path)

    return sorted(result_dirs)


def load_pareto_front(results_dir: str) -> pd.DataFrame:
    """Load Pareto front from a results directory"""
    pareto_file = os.path.join(results_dir, "pareto_front.csv")

    if not os.path.exists(pareto_file):
        return None

    df = pd.read_csv(pareto_file)

    # Add source directory info
    run_name = os.path.basename(results_dir)
    df['run_id'] = run_name

    return df


def analyze_single_front(df: pd.DataFrame, run_id: str) -> dict:
    """Analyze a single Pareto front"""
    stats = {
        'run_id': run_id,
        'pareto_size': len(df),
        'mqe_min': df['best_mqe'].min(),
        'mqe_max': df['best_mqe'].max(),
        'mqe_mean': df['best_mqe'].mean(),
        'topo_error_min': df['topographic_error'].min(),
        'topo_error_max': df['topographic_error'].max(),
        'topo_error_mean': df['topographic_error'].mean(),
        'dead_ratio_min': df['dead_neuron_ratio'].min(),
        'dead_ratio_max': df['dead_neuron_ratio'].max(),
        'dead_ratio_mean': df['dead_neuron_ratio'].mean(),
        'duration_min': df['duration'].min(),
        'duration_max': df['duration'].max(),
        'duration_mean': df['duration'].mean(),
    }

    # Add new metrics if they exist
    if 'u_matrix_max' in df.columns:
        stats['u_matrix_max_min'] = df['u_matrix_max'].min()
        stats['u_matrix_max_max'] = df['u_matrix_max'].max()
        stats['u_matrix_max_mean'] = df['u_matrix_max'].mean()

    if 'distance_map_max' in df.columns:
        stats['distance_map_max_min'] = df['distance_map_max'].min()
        stats['distance_map_max_max'] = df['distance_map_max'].max()
        stats['distance_map_max_mean'] = df['distance_map_max'].mean()

    return stats


def find_best_configurations(all_fronts: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Find the best configurations across all Pareto fronts"""

    # Define quality score: lower is better
    # Weighted combination of metrics
    all_fronts['quality_score'] = (
        all_fronts['best_mqe'] * 1.0 +
        all_fronts['topographic_error'] * 1.0 +
        all_fronts['dead_neuron_ratio'] * 0.5 +
        all_fronts['duration'] * 0.001  # Small weight for speed
    )

    # Add organization penalty if available
    if 'u_matrix_max' in all_fronts.columns and 'distance_map_max' in all_fronts.columns:
        org_penalty = all_fronts[['u_matrix_max', 'distance_map_max']].max(axis=1)
        org_penalty = org_penalty.apply(lambda x: max(0, x - 1.0) * 10.0)  # Penalty for > 1.0
        all_fronts['quality_score'] += org_penalty

    # Sort by quality score
    best_configs = all_fronts.sort_values('quality_score').head(top_n)

    return best_configs


def plot_pareto_comparison(all_fronts: pd.DataFrame, output_dir: str):
    """Generate comparison plots for Pareto fronts"""

    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: MQE vs Topographic Error (colored by run)
    plt.figure(figsize=(12, 8))

    runs = all_fronts['run_id'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for i, run in enumerate(runs):
        run_data = all_fronts[all_fronts['run_id'] == run]
        plt.scatter(run_data['best_mqe'], run_data['topographic_error'],
                   label=run, alpha=0.6, s=50, c=[colors[i]])

    plt.xlabel('Mean Quantization Error (MQE)', fontsize=12)
    plt.ylabel('Topographic Error', fontsize=12)
    plt.title('Pareto Fronts: MQE vs Topographic Error', fontsize=14, weight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mqe_vs_topo_error.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2: Dead Neuron Ratio Distribution
    plt.figure(figsize=(12, 8))

    for i, run in enumerate(runs):
        run_data = all_fronts[all_fronts['run_id'] == run]
        plt.scatter(run_data['dead_neuron_ratio'], run_data['best_mqe'],
                   label=run, alpha=0.6, s=50, c=[colors[i]])

    plt.xlabel('Dead Neuron Ratio', fontsize=12)
    plt.ylabel('Mean Quantization Error (MQE)', fontsize=12)
    plt.title('Pareto Fronts: Dead Neurons vs MQE', fontsize=14, weight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dead_neurons_vs_mqe.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 3: Duration vs MQE (Speed vs Quality tradeoff)
    plt.figure(figsize=(12, 8))

    for i, run in enumerate(runs):
        run_data = all_fronts[all_fronts['run_id'] == run]
        plt.scatter(run_data['duration'], run_data['best_mqe'],
                   label=run, alpha=0.6, s=50, c=[colors[i]])

    plt.xlabel('Training Duration (seconds)', fontsize=12)
    plt.ylabel('Mean Quantization Error (MQE)', fontsize=12)
    plt.title('Pareto Fronts: Speed vs Quality', fontsize=14, weight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_vs_mqe.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4: Organization quality (if available)
    if 'u_matrix_max' in all_fronts.columns and 'distance_map_max' in all_fronts.columns:
        plt.figure(figsize=(12, 8))

        for i, run in enumerate(runs):
            run_data = all_fronts[all_fronts['run_id'] == run]
            org_max = run_data[['u_matrix_max', 'distance_map_max']].max(axis=1)
            plt.scatter(org_max, run_data['best_mqe'],
                       label=run, alpha=0.6, s=50, c=[colors[i]])

        plt.axvline(x=1.0, color='red', linestyle='--', linewidth=2, label='Threshold (1.0)')
        plt.xlabel('Organization Max (max of U-Matrix and Distance Map)', fontsize=12)
        plt.ylabel('Mean Quantization Error (MQE)', fontsize=12)
        plt.title('Pareto Fronts: Organization Quality vs MQE', fontsize=14, weight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'organization_vs_mqe.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"✓ Plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze Pareto Fronts from Multiple EA Runs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all EA runs in results directory
  python3 analyze_pareto_fronts.py --results_base ./test/results

  # Analyze and show top 20 best configurations
  python3 analyze_pareto_fronts.py --results_base ./test/results --top_n 20

  # Export combined Pareto front to CSV
  python3 analyze_pareto_fronts.py --results_base ./test/results --export combined_pareto.csv
        """
    )

    parser.add_argument(
        '--results_base',
        type=str,
        default='./test/results',
        help='Base directory containing EA result folders (default: ./test/results)'
    )

    parser.add_argument(
        '--top_n',
        type=int,
        default=10,
        help='Number of top configurations to display (default: 10)'
    )

    parser.add_argument(
        '--export',
        type=str,
        help='Export combined Pareto front to CSV file'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate comparison plots'
    )

    parser.add_argument(
        '--plot_dir',
        type=str,
        default='./pareto_analysis',
        help='Directory to save plots (default: ./pareto_analysis)'
    )

    args = parser.parse_args()

    # Find all result directories
    print(f"Searching for EA results in: {args.results_base}")
    result_dirs = find_result_directories(args.results_base)

    if len(result_dirs) == 0:
        print("No EA result directories found.")
        sys.exit(1)

    print(f"Found {len(result_dirs)} EA runs:\n")

    # Load all Pareto fronts
    all_fronts = []
    run_stats = []

    for result_dir in result_dirs:
        run_id = os.path.basename(result_dir)
        print(f"Loading {run_id}...")

        df = load_pareto_front(result_dir)
        if df is not None:
            all_fronts.append(df)
            stats = analyze_single_front(df, run_id)
            run_stats.append(stats)

    if len(all_fronts) == 0:
        print("No Pareto fronts loaded.")
        sys.exit(1)

    # Combine all fronts
    combined_fronts = pd.concat(all_fronts, ignore_index=True)

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    # Print run-by-run statistics
    stats_df = pd.DataFrame(run_stats)

    print("Run-by-Run Statistics:\n")
    print(stats_df.to_string(index=False))

    print(f"\n{'='*80}")
    print(f"BEST CONFIGURATIONS (Top {args.top_n})")
    print(f"{'='*80}\n")

    # Find best configurations
    best_configs = find_best_configurations(combined_fronts, top_n=args.top_n)

    # Select columns to display
    display_cols = ['run_id', 'uid', 'best_mqe', 'topographic_error', 'dead_neuron_ratio', 'duration']
    if 'u_matrix_max' in best_configs.columns:
        display_cols.append('u_matrix_max')
    if 'distance_map_max' in best_configs.columns:
        display_cols.append('distance_map_max')
    display_cols.append('quality_score')

    print(best_configs[display_cols].to_string(index=False))

    # Export combined front if requested
    if args.export:
        combined_fronts.to_csv(args.export, index=False)
        print(f"\n✓ Combined Pareto front exported to: {args.export}")

    # Generate plots if requested
    if args.plot:
        print(f"\nGenerating comparison plots...")
        plot_pareto_comparison(combined_fronts, args.plot_dir)

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
