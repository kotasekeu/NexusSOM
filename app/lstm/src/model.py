"""
LSTM Model Architecture — Phase 2 Early Stopping Predictor

Hybrid input: variable-length sequence (LSTM) + static context (Dense).
Predicts final SOM quality from a K%-prefix of the training trajectory.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_hybrid_lstm(context_dim=4, output_dim=3, learning_rate=0.001):
    """
    Hybrid LSTM + static context model.

    Sequence input: (batch, time, 6) — variable time dimension, masked zeros ignored
    Context input:  (batch, 4)       — dataset properties

    Returns compiled Keras model.
    """
    # Sequence branch
    seq_input = keras.Input(shape=(None, 6), name='sequence')
    x = layers.Masking(mask_value=0.0)(seq_input)
    x = layers.LSTM(64, return_sequences=True, name='lstm1')(x)
    x = layers.Dropout(0.2, name='drop1')(x)
    lstm_out = layers.LSTM(32, name='lstm2')(x)

    # Static context branch
    ctx_input = keras.Input(shape=(context_dim,), name='context')
    ctx_out = layers.Dense(16, activation='relu', name='ctx_dense')(ctx_input)

    # Combine
    combined = layers.Concatenate(name='concat')([lstm_out, ctx_out])
    x = layers.Dense(32, activation='relu', name='dense1')(combined)
    x = layers.Dropout(0.2, name='drop2')(x)
    output = layers.Dense(output_dim, activation='linear', name='output')(x)

    model = keras.Model(inputs=[seq_input, ctx_input], outputs=output,
                        name='lstm_predictor')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model


def create_lightweight_hybrid_lstm(context_dim=4, output_dim=3, learning_rate=0.001):
    """Smaller variant for faster iteration."""
    seq_input = keras.Input(shape=(None, 6), name='sequence')
    x = layers.Masking(mask_value=0.0)(seq_input)
    lstm_out = layers.LSTM(32, name='lstm1')(x)

    ctx_input = keras.Input(shape=(context_dim,), name='context')
    ctx_out = layers.Dense(8, activation='relu', name='ctx_dense')(ctx_input)

    combined = layers.Concatenate(name='concat')([lstm_out, ctx_out])
    x = layers.Dense(16, activation='relu', name='dense1')(combined)
    output = layers.Dense(output_dim, activation='linear', name='output')(x)

    model = keras.Model(inputs=[seq_input, ctx_input], outputs=output,
                        name='lstm_predictor_lite')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model


def print_model_summary(model):
    print('\n' + '=' * 70)
    print('MODEL ARCHITECTURE')
    print('=' * 70)
    model.summary()
    print(f'Total parameters: {model.count_params():,}')
    print('=' * 70 + '\n')
