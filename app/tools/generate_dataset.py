#!/usr/bin/env python3
"""
Virtual Dataset Generator

Generates synthetic datasets based on preprocessing_info.json schema.
Preserves column types, distributions, and categorical properties.

Usage:
    python3 generate_dataset.py --schema path/to/preprocessing_info.json --output output.csv --rows 1000
    python3 generate_dataset.py --schema path/to/preprocessing_info.json --output output.csv --rows 500 --seed 42
"""

import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path


def load_schema(schema_path: str) -> dict:
    """Load preprocessing_info.json schema."""
    with open(schema_path, 'r') as f:
        return json.load(f)


def generate_numeric_column(name: str, info: dict, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Generate numeric column based on schema info."""
    dtype = info.get('type', 'float64')
    nunique = info.get('nunique', n_rows)
    nunique_ratio = info.get('nunique_ratio', 1.0)

    # Determine if integer or float
    is_integer = 'int' in dtype

    if is_integer:
        # For integers, generate values based on nunique
        if nunique_ratio == 1.0:
            # Unique IDs - sequential or random unique
            values = rng.permutation(np.arange(1, n_rows + 1))
        else:
            # Repeated integers
            unique_vals = rng.integers(0, max(nunique * 2, 100), size=min(nunique, n_rows))
            values = rng.choice(unique_vals, size=n_rows)
    else:
        # For floats, generate based on distribution
        if nunique_ratio > 0.8:
            # High uniqueness - continuous distribution
            values = rng.normal(loc=50, scale=20, size=n_rows)
            values = np.abs(values)  # Ensure positive for most cases
        else:
            # Lower uniqueness - some repeated values
            unique_vals = rng.normal(loc=50, scale=20, size=min(nunique, n_rows))
            values = rng.choice(unique_vals, size=n_rows)

    return values


def generate_categorical_column(name: str, info: dict, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Generate categorical column based on schema info."""
    nunique = info.get('nunique', 2)
    base_type = info.get('base_type', 'text')

    if base_type == 'text':
        # Generate text categories
        if nunique == 2:
            # Binary - common case like diagnosis
            categories = [f'{name}_A', f'{name}_B']
        else:
            categories = [f'{name}_cat_{i}' for i in range(nunique)]
    else:
        # Numeric categories
        categories = list(range(nunique))

    # Random distribution with slight imbalance
    weights = rng.dirichlet(np.ones(nunique) * 2)
    values = rng.choice(categories, size=n_rows, p=weights)

    return values


def generate_column(name: str, info: dict, n_rows: int, rng: np.random.Generator) -> np.ndarray:
    """Generate a single column based on its schema info."""
    status = info.get('status', 'used')

    # Skip removed columns
    if status == 'removed':
        return None

    is_categorical = info.get('is_categorical', False)
    base_type = info.get('base_type', 'numeric')

    if is_categorical or base_type == 'text':
        return generate_categorical_column(name, info, n_rows, rng)
    else:
        return generate_numeric_column(name, info, n_rows, rng)


def generate_dataset(schema: dict, n_rows: int, seed: int = None) -> pd.DataFrame:
    """Generate complete dataset based on schema."""
    rng = np.random.default_rng(seed)

    data = {}
    for col_name, col_info in schema.items():
        values = generate_column(col_name, col_info, n_rows, rng)
        if values is not None:
            data[col_name] = values

    df = pd.DataFrame(data)
    return df


def print_schema_summary(schema: dict):
    """Print summary of the schema."""
    total = len(schema)
    used = sum(1 for v in schema.values() if v.get('status') == 'used')
    removed = total - used
    categorical = sum(1 for v in schema.values() if v.get('is_categorical', False))
    numeric = sum(1 for v in schema.values() if v.get('base_type') == 'numeric' and not v.get('is_categorical', False))
    text = sum(1 for v in schema.values() if v.get('base_type') == 'text')

    print(f"\nSchema Summary:")
    print(f"  Total columns: {total}")
    print(f"  Used: {used}, Removed: {removed}")
    print(f"  Numeric: {numeric}, Categorical: {categorical}, Text: {text}")


def print_dataset_summary(df: pd.DataFrame):
    """Print summary of generated dataset."""
    print(f"\nGenerated Dataset Summary:")
    print(f"  Rows: {len(df)}")
    print(f"  Columns: {len(df.columns)}")
    print(f"\nColumn Types:")
    for col in df.columns:
        dtype = df[col].dtype
        nunique = df[col].nunique()
        print(f"  {col}: {dtype} (unique: {nunique})")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset from preprocessing_info.json schema"
    )
    parser.add_argument(
        '--schema', '-s',
        type=str,
        required=True,
        help='Path to preprocessing_info.json'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--rows', '-r',
        type=int,
        default=1000,
        help='Number of rows to generate (default: 1000)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print detailed output'
    )

    args = parser.parse_args()

    # Check schema file exists
    if not os.path.exists(args.schema):
        print(f"Error: Schema file not found: {args.schema}")
        sys.exit(1)

    # Load schema
    print(f"Loading schema from: {args.schema}")
    schema = load_schema(args.schema)

    if args.verbose:
        print_schema_summary(schema)

    # Generate dataset
    print(f"Generating {args.rows} rows...")
    df = generate_dataset(schema, args.rows, args.seed)

    if args.verbose:
        print_dataset_summary(df)

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save dataset
    df.to_csv(args.output, index=False)
    print(f"\nDataset saved to: {args.output}")
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}")


if __name__ == "__main__":
    main()
