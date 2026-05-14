"""
LSTM Controller — Phase 3 Dynamic Schedule

Stateful LSTM that processes one checkpoint at a time during SOM training
and outputs (lr_factor, radius_factor) to adjust the training schedule.

Input per step:  (progress, mqe_rel, topographic_error, dead_neuron_ratio,
                  lr_rel, radius_rel)  — 6 features, same as Phase 2
Output:          (lr_factor, radius_factor)  — multipliers in [0.5, 1.5]

The model is stateful: hidden state is preserved across checkpoints within
one SOM training run and reset between runs.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class _TileContext(layers.Layer):
    """Tile a (batch, D) context vector to (batch, T, D) matching a sequence."""
    def call(self, inputs):
        ctx, seq = inputs
        T = tf.shape(seq)[1]
        return tf.repeat(tf.expand_dims(ctx, 1), T, axis=1)

    def compute_output_shape(self, input_shapes):
        ctx_shape, seq_shape = input_shapes
        return (ctx_shape[0], seq_shape[1], ctx_shape[-1])

    def get_config(self):
        return super().get_config()


class _ScaleSigmoid(layers.Layer):
    """Map sigmoid output [0,1] to [0.5, 1.5]: y = x * 1.0 + 0.5."""
    def call(self, x):
        return x * 1.0 + 0.5

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return super().get_config()


def create_controller(context_dim=4, learning_rate=0.001):
    """
    Stateful LSTM controller for inference — processes one checkpoint per call.

    sequence input: (batch=1, time=1, 6)
    context input:  (batch=1, context_dim)
    output:         (batch=1, 2) — (lr_factor, radius_factor) in [0.5, 1.5]
    """
    seq_input = keras.Input(batch_shape=(1, 1, 6), name='sequence')
    x = layers.LSTM(64, stateful=True, return_sequences=False, name='lstm1')(seq_input)
    x = layers.Dense(32, activation='relu', name='dense1')(x)

    ctx_input = keras.Input(shape=(context_dim,), name='context')
    ctx_out   = layers.Dense(16, activation='relu', name='ctx_dense')(ctx_input)

    combined = layers.Concatenate(name='concat')([x, ctx_out])
    x = layers.Dense(16, activation='relu', name='dense2')(combined)
    raw    = layers.Dense(2, activation='sigmoid', name='output_raw')(x)
    output = _ScaleSigmoid(name='output')(raw)

    model = keras.Model(inputs=[seq_input, ctx_input], outputs=output,
                        name='lstm_controller')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model


def create_controller_trainable(context_dim=4, learning_rate=0.001):
    """
    Non-stateful training version — variable-length sequences, return_sequences=True.

    sequence input: (batch, T, 6)
    context input:  (batch, context_dim)
    output:         (batch, T, 2) — (lr_factor, radius_factor) in [0.5, 1.5]
    """
    seq_input = keras.Input(shape=(None, 6), name='sequence')
    x = layers.Masking(mask_value=0.0)(seq_input)
    x = layers.LSTM(64, return_sequences=True, name='lstm1')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=True, name='lstm2')(x)
    x = layers.Dense(32, activation='relu', name='dense1')(x)

    ctx_input = keras.Input(shape=(context_dim,), name='context')
    ctx_out   = layers.Dense(16, activation='relu', name='ctx_dense')(ctx_input)

    ctx_tiled = _TileContext(name='ctx_tile')([ctx_out, x])

    combined = layers.Concatenate(axis=-1, name='concat')([x, ctx_tiled])
    combined = layers.Dense(16, activation='relu', name='dense2')(combined)
    raw    = layers.Dense(2, activation='sigmoid', name='output_raw')(combined)
    output = _ScaleSigmoid(name='output')(raw)

    model = keras.Model(inputs=[seq_input, ctx_input], outputs=output,
                        name='lstm_controller_train')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae'],
    )
    return model
