"""
CNN Model Definition for SOM Quality Prediction

This module defines a Convolutional Neural Network architecture
for predicting quality scores from SOM visualization images.

The model performs regression to output a single quality score (0-1).

Architecture uses Global Average Pooling (GAP) to support variable input sizes:
- SOM maps can be 5x5 to 30x30 neurons
- Each neuron = 1 pixel (no upscaling needed)
- GAP collapses spatial dimensions to fixed-size feature vector
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers


def create_cnn_model(input_shape=None, learning_rate=0.001):
    """
    Create a CNN model for SOM quality prediction with variable input size support.

    Architecture:
        - 4 Convolutional blocks with increasing filters (padding='same')
        - Batch normalization and dropout for regularization
        - Global Average Pooling to handle any input size
        - Dense layers for final regression output

    Args:
        input_shape: Tuple of (height, width, channels) or None for dynamic size.
                     Use (None, None, 3) for variable size input.
        learning_rate: Learning rate for the Adam optimizer

    Returns:
        Compiled Keras model
    """
    # Input layer - supports dynamic size with (None, None, 3)
    # For SOM maps: 5x5 to 30x30 pixels (1 neuron = 1 pixel)
    if input_shape is None:
        input_shape = (None, None, 3)
    inputs = keras.Input(shape=input_shape, name='image_input')

    # Architecture designed for small images (5x5 to 30x30)
    # NO MaxPooling to preserve spatial information on small maps
    # All convolutions use padding='same' to maintain dimensions

    # Convolutional Block 1
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1_1')(inputs)
    x = layers.BatchNormalization(name='bn1_1')(x)
    x = layers.Activation('relu', name='relu1_1')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', name='conv1_2')(x)
    x = layers.BatchNormalization(name='bn1_2')(x)
    x = layers.Activation('relu', name='relu1_2')(x)
    x = layers.Dropout(0.25, name='dropout1')(x)

    # Convolutional Block 2
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2_1')(x)
    x = layers.BatchNormalization(name='bn2_1')(x)
    x = layers.Activation('relu', name='relu2_1')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', name='conv2_2')(x)
    x = layers.BatchNormalization(name='bn2_2')(x)
    x = layers.Activation('relu', name='relu2_2')(x)
    x = layers.Dropout(0.25, name='dropout2')(x)

    # Convolutional Block 3
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3_1')(x)
    x = layers.BatchNormalization(name='bn3_1')(x)
    x = layers.Activation('relu', name='relu3_1')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', name='conv3_2')(x)
    x = layers.BatchNormalization(name='bn3_2')(x)
    x = layers.Activation('relu', name='relu3_2')(x)
    x = layers.Dropout(0.3, name='dropout3')(x)

    # Convolutional Block 4
    x = layers.Conv2D(256, (3, 3), padding='same', name='conv4_1')(x)
    x = layers.BatchNormalization(name='bn4_1')(x)
    x = layers.Activation('relu', name='relu4_1')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', name='conv4_2')(x)
    x = layers.BatchNormalization(name='bn4_2')(x)
    x = layers.Activation('relu', name='relu4_2')(x)
    x = layers.Dropout(0.3, name='dropout4')(x)

    # Global Average Pooling to reduce parameters
    x = layers.GlobalAveragePooling2D(name='global_avg_pool')(x)

    # Dense layers for regression
    x = layers.Dense(256, kernel_regularizer=regularizers.l2(0.001), name='dense1')(x)
    x = layers.BatchNormalization(name='bn_dense1')(x)
    x = layers.Activation('relu', name='relu_dense1')(x)
    x = layers.Dropout(0.5, name='dropout_dense1')(x)

    x = layers.Dense(128, kernel_regularizer=regularizers.l2(0.001), name='dense2')(x)
    x = layers.BatchNormalization(name='bn_dense2')(x)
    x = layers.Activation('relu', name='relu_dense2')(x)
    x = layers.Dropout(0.5, name='dropout_dense2')(x)

    # Output layer: Single neuron with sigmoid activation for [0, 1] range
    outputs = layers.Dense(1, activation='sigmoid', name='quality_output')(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='SOM_Quality_CNN')

    # Compile model with Adam optimizer and MSE loss
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=[
            'mae',  # Mean Absolute Error
            'mse',  # Mean Squared Error
            keras.metrics.RootMeanSquaredError(name='rmse')
        ]
    )

    return model


def create_lightweight_model(input_shape=None, learning_rate=0.001):
    """
    Create a lightweight CNN model for faster training (alternative architecture).

    This is a simpler model with fewer parameters, useful for:
    - Quick prototyping
    - Limited computational resources
    - Smaller datasets

    Supports variable input sizes (5x5 to 30x30 SOM maps).

    Args:
        input_shape: Tuple of (height, width, channels) or None for dynamic size.
        learning_rate: Learning rate for the Adam optimizer

    Returns:
        Compiled Keras model
    """
    if input_shape is None:
        input_shape = (None, None, 3)
    inputs = keras.Input(shape=input_shape, name='image_input')

    # Block 1 - no pooling for small maps
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Dropout(0.3)(x)

    # Global pooling and output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='SOM_Quality_CNN_Lite')

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mean_squared_error',
        metrics=['mae', 'mse', keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    return model


def print_model_summary(model):
    """
    Print a detailed summary of the model architecture.

    Args:
        model: Keras model
    """
    print("=" * 80)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 80)
    model.summary()
    print("=" * 80)
    print(f"Total parameters: {model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")
    print("=" * 80)


if __name__ == "__main__":
    # Test model creation with variable input size
    print("\n--- Creating Standard CNN Model (variable input) ---")
    model = create_cnn_model(input_shape=(None, None, 3))
    print_model_summary(model)

    print("\n--- Creating Lightweight CNN Model (variable input) ---")
    lite_model = create_lightweight_model(input_shape=(None, None, 3))
    print_model_summary(lite_model)

    # Test with different input sizes
    print("\n--- Testing variable input sizes ---")
    import numpy as np
    for size in [(5, 5), (10, 10), (20, 20), (30, 30)]:
        test_input = np.random.rand(1, size[0], size[1], 3).astype(np.float32)
        output = model.predict(test_input, verbose=0)
        print(f"Input {size[0]}x{size[1]} -> Output: {output[0][0]:.4f}")

    print("\nModels support variable SOM map sizes (5x5 to 30x30)!")
