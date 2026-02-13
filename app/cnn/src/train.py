"""
Training Script for SOM Quality Prediction CNN

This script loads the prepared dataset, creates data generators,
builds the model, and trains it to predict quality scores.

Usage:
    python src/train.py
    python src/train.py --epochs 100 --batch-size 32 --model lite
"""

import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
try:
    # Try Keras 2.x style import
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
except (ImportError, ModuleNotFoundError):
    # Keras 3.x style import
    from keras.preprocessing.image import ImageDataGenerator
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger

from PIL import Image
from sklearn.model_selection import train_test_split

# Import model definitions
from model import create_cnn_model, create_lightweight_model, print_model_summary


class GroupedSizeDataLoader:
    """
    Data loader that groups images by size for efficient batching.

    Groups all images of the same size together, allowing batch_size > 1
    while maintaining native resolution (1 neuron = 1 pixel).

    This prevents CNN from learning size-based patterns instead of quality patterns.
    """

    def __init__(self, dataframe, batch_size=32, shuffle=True, augment=False):
        """
        Initialize the grouped data loader.

        Args:
            dataframe: DataFrame with 'filepath' and 'quality_score' columns
            batch_size: Number of samples per batch (same-size images)
            shuffle: Whether to shuffle data within groups
            augment: Whether to apply data augmentation
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment

        # Data augmentation - only flips for small maps
        if self.augment:
            self.aug_gen = ImageDataGenerator(
                horizontal_flip=True,
                vertical_flip=True,
                fill_mode='nearest'
            )

        # Group dataframe by image size
        self.size_groups = self._group_by_size(dataframe)
        self.n_samples = len(dataframe)

        # Calculate total batches
        self._calculate_batches()

    def _get_image_size(self, filepath):
        """Get image dimensions without loading full image."""
        try:
            with Image.open(filepath) as img:
                return img.size  # (width, height)
        except Exception as e:
            print(f"Error reading size of {filepath}: {e}")
            return (0, 0)

    def _group_by_size(self, dataframe):
        """Group dataframe rows by image size."""
        df = dataframe.copy().reset_index(drop=True)

        # Add size column
        df['_size'] = df['filepath'].apply(self._get_image_size)

        # Group by size
        groups = {}
        for size, group_df in df.groupby('_size'):
            if size != (0, 0):  # Skip invalid images
                groups[size] = group_df.drop(columns=['_size']).reset_index(drop=True)

        print(f"  Grouped into {len(groups)} size categories:")
        for size, group_df in sorted(groups.items()):
            print(f"    {size[0]}x{size[1]}: {len(group_df)} images")

        return groups

    def _calculate_batches(self):
        """Calculate total number of batches across all groups."""
        self.total_batches = 0
        self.group_batches = {}

        for size, group_df in self.size_groups.items():
            n_batches = int(np.ceil(len(group_df) / self.batch_size))
            self.group_batches[size] = n_batches
            self.total_batches += n_batches

    def __len__(self):
        """Return total number of batches per epoch."""
        return self.total_batches

    def load_and_preprocess_image(self, filepath):
        """Load image in native size."""
        try:
            img = Image.open(filepath).convert('RGB')
            img_array = np.array(img, dtype=np.float32) / 255.0
            return img_array
        except Exception as e:
            print(f"Error loading image {filepath}: {e}")
            return None

    def __call__(self):
        """Generator function that yields batches of same-sized images."""
        while True:
            # Get list of sizes and optionally shuffle
            sizes = list(self.size_groups.keys())
            if self.shuffle:
                np.random.shuffle(sizes)

            for size in sizes:
                group_df = self.size_groups[size]
                indices = np.arange(len(group_df))

                if self.shuffle:
                    np.random.shuffle(indices)

                # Create batches from this size group
                for start_idx in range(0, len(indices), self.batch_size):
                    end_idx = min(start_idx + self.batch_size, len(indices))
                    batch_indices = indices[start_idx:end_idx]

                    batch_images = []
                    batch_scores = []

                    for idx in batch_indices:
                        row = group_df.iloc[idx]
                        img = self.load_and_preprocess_image(row['filepath'])
                        if img is not None:
                            batch_images.append(img)
                            batch_scores.append(row['quality_score'])

                    if not batch_images:
                        continue

                    batch_images = np.array(batch_images)
                    batch_scores = np.array(batch_scores, dtype=np.float32)

                    # Apply augmentation if enabled
                    if self.augment:
                        augmented_images = []
                        for img in batch_images:
                            img_aug = self.aug_gen.random_transform(img)
                            augmented_images.append(img_aug)
                        batch_images = np.array(augmented_images)

                    yield batch_images, batch_scores


# Keep old loader for backward compatibility
ImageDataLoader = GroupedSizeDataLoader


def load_dataset(dataset_path, test_size=0.15, val_size=0.15, random_state=42):
    """
    Load and split the dataset into train, validation, and test sets.

    Args:
        dataset_path: Path to the dataset CSV file
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"Total samples: {len(df)}")

    # Split into train+val and test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state
    )

    # Split train+val into train and val
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size, random_state=random_state
    )

    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")

    return train_df, val_df, test_df


def create_callbacks(model_name, log_dir):
    """
    Create training callbacks for model checkpointing, early stopping, etc.

    Args:
        model_name: Name for saving the model
        log_dir: Directory for logs

    Returns:
        List of Keras callbacks
    """
    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        # Save best model based on validation loss (native Keras format)
        ModelCheckpoint(
            filepath=f'models/{model_name}_best.keras',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        ),
        # Save checkpoints every epoch (native Keras format)
        ModelCheckpoint(
            filepath=f'models/{model_name}_epoch_{{epoch:02d}}.keras',
            save_freq='epoch',
            save_weights_only=False,
            verbose=0
        ),
        # Early stopping to prevent overfitting
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            mode='min',
            verbose=1
        ),
        # Reduce learning rate when validation loss plateaus
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            mode='min',
            verbose=1
        ),
        # TensorBoard logging
        TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            write_images=False,
            update_freq='epoch'
        ),
        # CSV logger for training history
        CSVLogger(
            filename=os.path.join(log_dir, 'training_log.csv'),
            separator=',',
            append=False
        )
    ]

    return callbacks


def train_model(dataset_path, model_type='standard', epochs=50,
                batch_size=32, learning_rate=0.001):
    """
    Main training function with grouped-by-size batching.

    Uses GroupedSizeDataLoader to batch images of the same size together,
    allowing efficient training while maintaining native resolution.

    Args:
        dataset_path: Path to the dataset CSV file
        model_type: Type of model ('standard' or 'lite')
        epochs: Number of training epochs
        batch_size: Number of samples per batch (within same-size groups)
        learning_rate: Learning rate for optimizer
    """
    print("=" * 80)
    print("SOM QUALITY PREDICTION - TRAINING (GAP + Grouped Batching)")
    print("=" * 80)
    print(f"Model Type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Batch Size: {batch_size} (grouped by image size)")
    print(f"Input Size: Variable (native SOM map size, no interpolation)")
    print(f"Learning Rate: {learning_rate}")
    print("=" * 80)

    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load and split dataset
    train_df, val_df, test_df = load_dataset(dataset_path)

    # Create data generators with grouped batching
    print("\nCreating grouped data generators...")
    print("Training set:")
    train_gen = GroupedSizeDataLoader(
        train_df, batch_size=batch_size,
        shuffle=True, augment=True
    )
    print("Validation set:")
    val_gen = GroupedSizeDataLoader(
        val_df, batch_size=batch_size,
        shuffle=False, augment=False
    )
    print("Test set:")
    test_gen = GroupedSizeDataLoader(
        test_df, batch_size=batch_size,
        shuffle=False, augment=False
    )

    # Create model with variable input shape
    print("\nCreating model with variable input support...")
    input_shape = (None, None, 3)  # Variable height/width
    if model_type == 'lite':
        model = create_lightweight_model(input_shape=input_shape, learning_rate=learning_rate)
    else:
        model = create_cnn_model(input_shape=input_shape, learning_rate=learning_rate)

    print_model_summary(model)

    # Create callbacks
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"som_quality_{model_type}_{timestamp}"
    log_dir = f"logs/{model_name}"
    callbacks = create_callbacks(model_name, log_dir)

    # Train model
    print("\nStarting training...")
    print(f"Training batches per epoch: {len(train_gen)}")
    print(f"Validation batches per epoch: {len(val_gen)}")

    history = model.fit(
        train_gen(),
        steps_per_epoch=len(train_gen),
        validation_data=val_gen(),
        validation_steps=len(val_gen),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "=" * 80)
    print("EVALUATING ON TEST SET")
    print("=" * 80)
    test_results = model.evaluate(
        test_gen(),
        steps=len(test_gen),
        verbose=1
    )

    print("\nTest Results:")
    for metric_name, metric_value in zip(model.metrics_names, test_results):
        print(f"  {metric_name}: {metric_value:.6f}")

    # Save final model
    final_model_path = f'models/{model_name}_final.keras'
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    # Save test set for later evaluation
    test_df.to_csv(f'models/{model_name}_test_set.csv', index=False)
    print(f"Test set saved to: models/{model_name}_test_set.csv")

    print("\n" + "=" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nBest model: models/{model_name}_best.keras")
    print(f"Training logs: {log_dir}")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir={log_dir}")

    return model, history


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SOM Quality Prediction CNN (GAP + Grouped Batching)')

    parser.add_argument(
        '--dataset',
        type=str,
        default='data/processed/dataset.csv',
        help='Path to dataset CSV file'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='standard',
        choices=['standard', 'lite'],
        help='Model architecture to use'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size (images grouped by size)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer'
    )

    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_arguments()

    # Train model with grouped batching
    try:
        model, history = train_model(
            dataset_path=args.dataset,
            model_type=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        raise
