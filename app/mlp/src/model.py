"""
MLP Model Architecture - "The Prophet"

Multi-Layer Perceptron for predicting SOM quality from hyperparameters.
Predicts quality metrics WITHOUT running SOM training.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_mlp_model(input_dim, output_dim=3, learning_rate=0.001):
    """
    Create MLP model for hyperparameter -> quality prediction.

    Args:
        input_dim: Number of input features (encoded hyperparameters)
        output_dim: Number of outputs (default: 3 for mqe, topo_error, dead_ratio)
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(input_dim,), name='hyperparameters'),

        # Dense layers with dropout for regularization
        layers.Dense(256, activation='relu', name='dense1'),
        layers.BatchNormalization(name='bn1'),
        layers.Dropout(0.3, name='dropout1'),

        layers.Dense(128, activation='relu', name='dense2'),
        layers.BatchNormalization(name='bn2'),
        layers.Dropout(0.3, name='dropout2'),

        layers.Dense(64, activation='relu', name='dense3'),
        layers.BatchNormalization(name='bn3'),
        layers.Dropout(0.2, name='dropout3'),

        layers.Dense(32, activation='relu', name='dense4'),
        layers.Dropout(0.2, name='dropout4'),

        # Output layer - no activation for regression
        layers.Dense(output_dim, activation='linear', name='output')
    ], name='mlp_prophet')

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def create_lightweight_mlp(input_dim, output_dim=3, learning_rate=0.001):
    """
    Create lightweight MLP for faster inference.

    Args:
        input_dim: Number of input features
        output_dim: Number of outputs
        learning_rate: Learning rate

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,), name='hyperparameters'),

        layers.Dense(128, activation='relu', name='dense1'),
        layers.Dropout(0.2, name='dropout1'),

        layers.Dense(64, activation='relu', name='dense2'),
        layers.Dropout(0.2, name='dropout2'),

        layers.Dense(32, activation='relu', name='dense3'),

        layers.Dense(output_dim, activation='linear', name='output')
    ], name='mlp_prophet_lite')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def print_model_summary(model):
    """Print model architecture summary"""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80 + "\n")

    model.summary()

    print("\n" + "="*80)
    print("MODEL DETAILS")
    print("="*80)
    print(f"Total parameters: {model.count_params():,}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test model creation
    print("Testing MLP model creation...\n")

    # Standard model
    model = create_mlp_model(input_dim=20, output_dim=3)
    print_model_summary(model)

    # Lightweight model
    print("\n" + "="*80)
    print("LIGHTWEIGHT MODEL")
    print("="*80 + "\n")
    model_lite = create_lightweight_mlp(input_dim=20, output_dim=3)
    print_model_summary(model_lite)

    print("✓ Model creation test successful!")
