#!/usr/bin/env python3
"""
CNN Dataset Preparation with Pseudo-Labeling

This script prepares training data for the CNN quality predictor:
1. Scans all EA results from data/datasets/*/results/EA/results.csv
2. Auto-labels extreme cases (clearly good/bad maps)
3. Generates images from SOM weights (organized by size)
4. Creates dataset CSV for training

Usage (from project root):
    python app/cnn/src/prepare_dataset.py --output data/cnn/datasets/dataset_v1.csv
    python app/cnn/src/prepare_dataset.py --pseudo-label --model app/cnn/models/best.keras
"""

import os
import sys
import json
import argparse
import ast
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image


# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from som_converter import SOMToImageConverter


class AutoLabeler:
    """
    Auto-labels SOM maps based on quality metrics.

    Labels extreme cases with high confidence:
    - Clearly BAD: high dead neuron ratio or high topographic error
    - Clearly GOOD: low dead ratio AND low topographic error
    - Uncertain: everything else (requires manual or pseudo-labeling)
    """

    def __init__(self, config: dict = None):
        """
        Initialize with labeling thresholds.

        Args:
            config: Dict with threshold configuration
        """
        default_config = {
            # BAD thresholds (if ANY exceeded -> BAD)
            'bad_dead_ratio': 0.30,      # >30% dead neurons -> BAD
            'bad_topo_error': 0.50,      # >50% topo error -> BAD

            # GOOD thresholds (ALL must be satisfied -> GOOD)
            'good_dead_ratio': 0.05,     # <5% dead neurons
            'good_topo_error': 0.10,     # <10% topo error

            # Label values
            'bad_label': 0.1,            # Quality score for BAD maps
            'good_label': 0.9,           # Quality score for GOOD maps
        }

        self.config = {**default_config, **(config or {})}

    def label(self, row: pd.Series) -> Tuple[Optional[float], str]:
        """
        Auto-label a single sample based on metrics.

        Args:
            row: Series with 'dead_neuron_ratio' and 'topographic_error'

        Returns:
            Tuple of (quality_score, label_source)
            quality_score is None if uncertain
        """
        dead = row.get('dead_neuron_ratio', 0)
        topo = row.get('topographic_error', 0)

        # Check for BAD conditions (any condition triggers BAD)
        if dead > self.config['bad_dead_ratio']:
            return self.config['bad_label'], 'auto_dead'

        if topo > self.config['bad_topo_error']:
            return self.config['bad_label'], 'auto_topo'

        # Check for GOOD conditions (all conditions must be met)
        if dead < self.config['good_dead_ratio'] and topo < self.config['good_topo_error']:
            return self.config['good_label'], 'auto_good'

        # Uncertain - needs manual or pseudo-labeling
        return None, 'unlabeled'

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Auto-label entire dataframe.

        Args:
            df: DataFrame with quality metrics

        Returns:
            DataFrame with added 'quality_score' and 'label_source' columns
        """
        results = df.apply(self.label, axis=1, result_type='expand')
        df = df.copy()
        df['quality_score'] = results[0]
        df['label_source'] = results[1]
        return df


class DatasetPreparer:
    """
    Prepares CNN training dataset from EA results.
    """

    def __init__(self, project_root: str = None):
        """
        Initialize preparer.

        Args:
            project_root: Path to NexusSom project root
        """
        if project_root is None:
            # Assume running from project root
            project_root = Path.cwd()
        self.project_root = Path(project_root)
        self.datasets_dir = self.project_root / 'data' / 'datasets'
        self.cnn_data_dir = self.project_root / 'data' / 'cnn'
        self.images_dir = self.cnn_data_dir / 'images'
        self.labels_dir = self.cnn_data_dir / 'labels'
        self.output_dir = self.cnn_data_dir / 'datasets'

        # Create directories
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.auto_labeler = AutoLabeler()
        self.converter = SOMToImageConverter(target_size=None, method='rgb')

    def find_ea_results(self) -> List[Tuple[str, Path]]:
        """
        Find all EA results.csv files.

        Returns:
            List of (dataset_name, results_path) tuples
        """
        results = []

        for dataset_dir in self.datasets_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            results_csv = dataset_dir / 'results' / 'EA' / 'results.csv'
            if results_csv.exists():
                results.append((dataset_dir.name, results_csv))

        return results

    def parse_map_size(self, map_size_str: str) -> Tuple[int, int]:
        """
        Parse map_size from string like "[10, 10]" to tuple (10, 10).
        """
        try:
            size = ast.literal_eval(map_size_str)
            if isinstance(size, list) and len(size) == 2:
                return tuple(size)
        except (ValueError, SyntaxError):
            pass
        return (0, 0)

    def load_all_results(self) -> pd.DataFrame:
        """
        Load and combine all EA results.

        Returns:
            Combined DataFrame with all results
        """
        all_results = []

        for dataset_name, results_path in self.find_ea_results():
            print(f"Loading {dataset_name}: {results_path}")

            df = pd.read_csv(results_path)
            df['dataset_name'] = dataset_name
            df['results_path'] = str(results_path.parent)

            # Parse map_size
            if 'map_size' in df.columns:
                df['map_size_tuple'] = df['map_size'].apply(self.parse_map_size)
                df['map_width'] = df['map_size_tuple'].apply(lambda x: x[0])
                df['map_height'] = df['map_size_tuple'].apply(lambda x: x[1])

            all_results.append(df)

        if not all_results:
            return pd.DataFrame()

        combined = pd.concat(all_results, ignore_index=True)
        print(f"\nTotal samples: {len(combined)}")
        return combined

    def find_or_create_image(self, row: pd.Series) -> Optional[str]:
        """
        Find existing image or create from weights.

        Args:
            row: Series with uid, map_width, map_height, results_path

        Returns:
            Path to image (relative to project root) or None if not found
        """
        uid = row['uid']
        width = row['map_width']
        height = row['map_height']
        results_path = Path(row['results_path'])

        # Target directory based on size
        size_dir = self.images_dir / f"{width}x{height}"
        size_dir.mkdir(exist_ok=True)

        target_image = size_dir / f"{uid}.png"

        # If image already exists, return it
        if target_image.exists():
            return str(target_image.relative_to(self.project_root))

        # Try to find existing RGB image in EA results
        possible_sources = [
            results_path / 'maps_dataset' / 'rgb' / f"{uid}_rgb.png",
            results_path / 'visualizations' / f"{uid}_rgb.png",
            results_path / f"{uid}_rgb.png",
        ]

        for source in possible_sources:
            if source.exists():
                # Copy to our structure
                img = Image.open(source)
                img.save(target_image)
                return str(target_image.relative_to(self.project_root))

        # Try to load weights and generate image
        weights_paths = [
            results_path / 'weights' / f"{uid}_weights.npy",
            results_path / f"{uid}_weights.npy",
        ]

        for weights_path in weights_paths:
            if weights_path.exists():
                weights = np.load(weights_path)
                self.converter.save_image(weights, str(target_image))
                return str(target_image.relative_to(self.project_root))

        # Image not found and cannot be created
        return None

    def prepare_dataset(self, output_path: str = None, include_unlabeled: bool = False) -> pd.DataFrame:
        """
        Prepare full dataset with auto-labeling.

        Args:
            output_path: Path to save dataset CSV
            include_unlabeled: Whether to include unlabeled samples

        Returns:
            Prepared DataFrame
        """
        print("=" * 60)
        print("CNN DATASET PREPARATION")
        print("=" * 60)

        # Load all results
        df = self.load_all_results()
        if df.empty:
            print("No EA results found!")
            return df

        # Auto-label
        print("\nAuto-labeling...")
        df = self.auto_labeler.label_dataframe(df)

        # Print labeling statistics
        label_counts = df['label_source'].value_counts()
        print("\nLabel distribution:")
        for label, count in label_counts.items():
            pct = count / len(df) * 100
            print(f"  {label}: {count} ({pct:.1f}%)")

        # Find or create images
        print("\nProcessing images...")
        df['filepath'] = df.apply(self.find_or_create_image, axis=1)

        # Count images found
        images_found = df['filepath'].notna().sum()
        print(f"Images found/created: {images_found}/{len(df)}")

        # Filter to samples with images
        df = df[df['filepath'].notna()].copy()

        # Filter unlabeled if requested
        if not include_unlabeled:
            df = df[df['quality_score'].notna()].copy()
            print(f"Labeled samples with images: {len(df)}")

        # Select columns for output
        output_columns = [
            'filepath',
            'quality_score',
            'map_width',
            'uid',
            'label_source',
            'dataset_name',
            'dead_neuron_ratio',
            'topographic_error',
            'best_mqe',
        ]

        # Only include columns that exist
        output_columns = [c for c in output_columns if c in df.columns]
        output_df = df[output_columns].copy()

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_path, index=False)
            print(f"\nDataset saved to: {output_path}")

        # Save labels separately
        labels_path = self.labels_dir / 'auto_labels.csv'
        labels_df = df[['uid', 'quality_score', 'label_source', 'dataset_name']].copy()
        labels_df.to_csv(labels_path, index=False)
        print(f"Labels saved to: {labels_path}")

        print("\n" + "=" * 60)
        print("PREPARATION COMPLETE")
        print("=" * 60)

        return output_df


class PseudoLabeler:
    """
    Pseudo-labeling using trained CNN model.
    """

    def __init__(self, model_path: str):
        """
        Initialize with trained model.

        Args:
            model_path: Path to trained .keras model
        """
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, image_path: str) -> float:
        """
        Predict quality score for image.

        Args:
            image_path: Path to image

        Returns:
            Quality score (0-1)
        """
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = self.model.predict(img_array, verbose=0)
        return float(prediction[0][0])

    def pseudo_label(self, df: pd.DataFrame, confidence_threshold: float = 0.15) -> pd.DataFrame:
        """
        Add pseudo-labels to unlabeled samples.

        Args:
            df: DataFrame with 'filepath' and 'quality_score' columns
            confidence_threshold: Only label if prediction is this far from 0.5

        Returns:
            DataFrame with pseudo-labels added
        """
        df = df.copy()

        # Find unlabeled samples
        unlabeled_mask = df['quality_score'].isna()
        unlabeled_count = unlabeled_mask.sum()

        if unlabeled_count == 0:
            print("No unlabeled samples to process")
            return df

        print(f"Pseudo-labeling {unlabeled_count} samples...")

        pseudo_labels = []
        pseudo_sources = []

        for idx, row in df[unlabeled_mask].iterrows():
            filepath = row['filepath']
            try:
                pred = self.predict(filepath)

                # Check confidence (distance from 0.5)
                confidence = abs(pred - 0.5)

                if confidence >= confidence_threshold:
                    pseudo_labels.append(pred)
                    pseudo_sources.append('pseudo')
                else:
                    pseudo_labels.append(None)
                    pseudo_sources.append('unlabeled')

            except Exception as e:
                print(f"Error processing {filepath}: {e}")
                pseudo_labels.append(None)
                pseudo_sources.append('error')

        # Update dataframe
        df.loc[unlabeled_mask, 'quality_score'] = pseudo_labels
        df.loc[unlabeled_mask, 'label_source'] = pseudo_sources

        # Statistics
        labeled_count = sum(1 for l in pseudo_labels if l is not None)
        print(f"Pseudo-labeled: {labeled_count}/{unlabeled_count}")

        return df


def parse_args():
    parser = argparse.ArgumentParser(
        description='Prepare CNN training dataset from EA results'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/cnn/datasets/dataset_v1.csv',
        help='Output dataset CSV path'
    )

    parser.add_argument(
        '--include-unlabeled',
        action='store_true',
        help='Include unlabeled samples in output'
    )

    parser.add_argument(
        '--pseudo-label',
        action='store_true',
        help='Apply pseudo-labeling using trained model'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Path to trained model for pseudo-labeling'
    )

    parser.add_argument(
        '--confidence',
        type=float,
        default=0.15,
        help='Confidence threshold for pseudo-labeling'
    )

    parser.add_argument(
        '--project-root',
        type=str,
        default=None,
        help='Project root directory'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Prepare dataset
    preparer = DatasetPreparer(project_root=args.project_root)
    df = preparer.prepare_dataset(
        output_path=args.output,
        include_unlabeled=args.include_unlabeled or args.pseudo_label
    )

    # Apply pseudo-labeling if requested
    if args.pseudo_label:
        if not args.model:
            print("Error: --model required for pseudo-labeling")
            sys.exit(1)

        labeler = PseudoLabeler(args.model)
        df = labeler.pseudo_label(df, confidence_threshold=args.confidence)

        # Save updated dataset
        df.to_csv(args.output, index=False)
        print(f"Updated dataset saved to: {args.output}")


if __name__ == '__main__':
    main()
