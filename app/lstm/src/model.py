"""
LSTM Model Architecture - "The Oracle"

LSTM network for predicting final SOM quality from early training progress.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_lstm_model(sequence_length, num_features, output_dim=3, learning_rate=0.001):
    """
    Create LSTM model for training progress -> final quality prediction.

    Args:
        sequence_length: Number of time steps in each sequence
        num_features: Number of features per time step
        output_dim: Number of outputs (default: 3 for mqe, topo_error, dead_ratio)
        learning_rate: Learning rate for optimizer

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=(sequence_length, num_features), name='training_sequence'),

        # LSTM layers with dropout
        layers.LSTM(128, return_sequences=True, name='lstm1'),
        layers.Dropout(0.3, name='dropout1'),

        layers.LSTM(64, return_sequences=True, name='lstm2'),
        layers.Dropout(0.3, name='dropout2'),

        layers.LSTM(32, return_sequences=False, name='lstm3'),
        layers.Dropout(0.2, name='dropout3'),

        # Dense layers
        layers.Dense(32, activation='relu', name='dense1'),
        layers.Dropout(0.2, name='dropout4'),

        layers.Dense(16, activation='relu', name='dense2'),

        # Output layer
        layers.Dense(output_dim, activation='linear', name='output')
    ], name='lstm_oracle')

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def create_bidirectional_lstm(sequence_length, num_features, output_dim=3, learning_rate=0.001):
    """
    Create bidirectional LSTM model (processes sequences forward and backward).

    Args:
        sequence_length: Number of time steps
        num_features: Number of features per time step
        output_dim: Number of outputs
        learning_rate: Learning rate

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, num_features), name='training_sequence'),

        # Bidirectional LSTM layers
        layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='bilstm1'),
        layers.Dropout(0.3, name='dropout1'),

        layers.Bidirectional(layers.LSTM(32, return_sequences=False), name='bilstm2'),
        layers.Dropout(0.2, name='dropout2'),

        # Dense layers
        layers.Dense(32, activation='relu', name='dense1'),
        layers.Dropout(0.2, name='dropout3'),

        layers.Dense(output_dim, activation='linear', name='output')
    ], name='bilstm_oracle')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mse']
    )

    return model


def create_lightweight_lstm(sequence_length, num_features, output_dim=3, learning_rate=0.001):
    """
    Create lightweight LSTM for faster inference.

    Args:
        sequence_length: Number of time steps
        num_features: Number of features per time step
        output_dim: Number of outputs
        learning_rate: Learning rate

    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, num_features), name='training_sequence'),

        layers.LSTM(64, return_sequences=True, name='lstm1'),
        layers.Dropout(0.2, name='dropout1'),

        layers.LSTM(32, return_sequences=False, name='lstm2'),

        layers.Dense(16, activation='relu', name='dense1'),

        layers.Dense(output_dim, activation='linear', name='output')
    ], name='lstm_oracle_lite')

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
    print("Testing LSTM model creation...\n")

    # Standard LSTM
    model = create_lstm_model(sequence_length=10, num_features=3, output_dim=3)
    print_model_summary(model)

    # Bidirectional LSTM
    print("\n" + "="*80)
    print("BIDIRECTIONAL LSTM MODEL")
    print("="*80 + "\n")
    model_bi = create_bidirectional_lstm(sequence_length=10, num_features=3, output_dim=3)
    print_model_summary(model_bi)

    # Lightweight LSTM
    print("\n" + "="*80)
    print("LIGHTWEIGHT LSTM MODEL")
    print("="*80 + "\n")
    model_lite = create_lightweight_lstm(sequence_length=10, num_features=3, output_dim=3)
    print_model_summary(model_lite)

    print("✓ Model creation test successful!")
