"""
Neural Network Integration for EA

This module provides optional integration of neural networks for:
1. MLP "The Prophet" - Fast fitness estimation from hyperparameters
2. LSTM "The Oracle" - Early stopping predictions during training
3. CNN "The Eye" - Post-training visual quality assessment

All neural network features are OPTIONAL and the EA works without them.
"""

import os
import sys
import numpy as np
import warnings
from typing import Optional, Dict, Tuple, Any

# Lazy imports - only load TensorFlow when NN features are enabled
_tf_loaded = False
_keras = None
_joblib = None


def _ensure_tensorflow():
    """Lazy load TensorFlow and dependencies"""
    global _tf_loaded, _keras, _joblib

    if _tf_loaded:
        return True

    try:
        import tensorflow as tf
        from tensorflow import keras
        import joblib

        # Suppress TensorFlow warnings
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        tf.get_logger().setLevel('ERROR')

        _keras = keras
        _joblib = joblib
        _tf_loaded = True
        return True
    except ImportError as e:
        warnings.warn(f"TensorFlow/Keras not available: {e}. Neural network features disabled.")
        return False


class NeuralNetworkIntegration:
    """
    Integration layer for neural networks in EA.

    Usage:
        # Create integration (with or without NN)
        nn = NeuralNetworkIntegration(use_mlp=True, use_lstm=False, use_cnn=True)

        # Fast fitness estimation (MLP)
        if nn.can_predict_fitness():
            predicted_quality = nn.predict_fitness(config)

        # Early stopping check (LSTM)
        if nn.can_check_early_stopping():
            should_stop = nn.should_stop_early(training_history)

        # Visual quality assessment (CNN)
        if nn.can_assess_visual_quality():
            quality_score = nn.assess_visual_quality(rgb_image_path)
    """

    def __init__(self,
                 use_mlp: bool = False,
                 use_lstm: bool = False,
                 use_cnn: bool = False,
                 mlp_model_path: Optional[str] = None,
                 mlp_scaler_path: Optional[str] = None,
                 lstm_model_path: Optional[str] = None,
                 cnn_model_path: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize neural network integration.

        Args:
            use_mlp: Enable MLP for fitness prediction
            use_lstm: Enable LSTM for early stopping
            use_cnn: Enable CNN for visual quality assessment
            mlp_model_path: Path to MLP model (.keras)
            mlp_scaler_path: Path to MLP scaler (.pkl)
            lstm_model_path: Path to LSTM model (.keras)
            cnn_model_path: Path to CNN model (.keras)
            verbose: Print status messages
        """
        self.use_mlp = use_mlp
        self.use_lstm = use_lstm
        self.use_cnn = use_cnn
        self.verbose = verbose

        self.mlp_model = None
        self.mlp_scaler = None
        self.mlp_metadata = None

        self.lstm_model = None
        self.lstm_metadata = None

        self.cnn_model = None
        self.cnn_metadata = None

        # Only load TensorFlow if any NN feature is enabled
        if self.use_mlp or self.use_lstm or self.use_cnn:
            if not _ensure_tensorflow():
                self._print("⚠ TensorFlow not available. All NN features disabled.")
                self.use_mlp = False
                self.use_lstm = False
                self.use_cnn = False
                return

        # Load models
        if self.use_mlp:
            self._load_mlp(mlp_model_path, mlp_scaler_path)

        if self.use_lstm:
            self._load_lstm(lstm_model_path)

        if self.use_cnn:
            self._load_cnn(cnn_model_path)

    def _print(self, message: str):
        """Print message if verbose"""
        if self.verbose:
            print(message)

    def _load_mlp(self, model_path: Optional[str], scaler_path: Optional[str]):
        """Load MLP model and scaler"""
        try:
            if model_path is None:
                # Auto-detect latest model
                model_dir = os.path.join(os.path.dirname(__file__), '..', 'mlp', 'models')
                if os.path.exists(model_dir):
                    models = [f for f in os.listdir(model_dir) if f.endswith('_best.keras')]
                    if models:
                        models.sort(reverse=True)  # Get latest
                        model_path = os.path.join(model_dir, models[0])
                        scaler_path = model_path.replace('_best.keras', '_scaler.pkl')

            if model_path and os.path.exists(model_path):
                self._print(f"Loading MLP model: {model_path}")
                self.mlp_model = _keras.models.load_model(model_path)

                if scaler_path and os.path.exists(scaler_path):
                    self.mlp_scaler = _joblib.load(scaler_path)
                    self._print("✓ MLP model and scaler loaded")

                    # Load metadata
                    metadata_path = model_path.replace('_best.keras', '_metadata.json')
                    if os.path.exists(metadata_path):
                        import json
                        with open(metadata_path, 'r') as f:
                            self.mlp_metadata = json.load(f)
                else:
                    self._print("⚠ MLP scaler not found. MLP disabled.")
                    self.mlp_model = None
                    self.use_mlp = False
            else:
                self._print("⚠ MLP model not found. MLP features disabled.")
                self.use_mlp = False
        except Exception as e:
            self._print(f"⚠ Failed to load MLP: {e}. MLP features disabled.")
            self.mlp_model = None
            self.use_mlp = False

    def _load_lstm(self, model_path: Optional[str]):
        """Load LSTM model"""
        try:
            if model_path is None:
                # Auto-detect latest model
                model_dir = os.path.join(os.path.dirname(__file__), '..', 'lstm', 'models')
                if os.path.exists(model_dir):
                    models = [f for f in os.listdir(model_dir) if f.endswith('_best.keras')]
                    if models:
                        models.sort(reverse=True)
                        model_path = os.path.join(model_dir, models[0])

            if model_path and os.path.exists(model_path):
                self._print(f"Loading LSTM model: {model_path}")
                self.lstm_model = _keras.models.load_model(model_path)
                self._print("✓ LSTM model loaded")

                # Load metadata
                metadata_path = model_path.replace('_best.keras', '_metadata.json')
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        self.lstm_metadata = json.load(f)
            else:
                self._print("⚠ LSTM model not found. LSTM features disabled.")
                self.use_lstm = False
        except Exception as e:
            self._print(f"⚠ Failed to load LSTM: {e}. LSTM features disabled.")
            self.lstm_model = None
            self.use_lstm = False

    def _load_cnn(self, model_path: Optional[str]):
        """Load CNN model"""
        try:
            if model_path is None:
                # Auto-detect latest model
                model_dir = os.path.join(os.path.dirname(__file__), '..', 'cnn', 'models')
                if os.path.exists(model_dir):
                    models = [f for f in os.listdir(model_dir) if f.endswith('_best.keras')]
                    if models:
                        models.sort(reverse=True)
                        model_path = os.path.join(model_dir, models[0])

            if model_path and os.path.exists(model_path):
                self._print(f"Loading CNN model: {model_path}")
                self.cnn_model = _keras.models.load_model(model_path)
                self._print("✓ CNN model loaded")

                # Load metadata
                metadata_path = model_path.replace('_best.keras', '_metadata.json')
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        self.cnn_metadata = json.load(f)
            else:
                self._print("⚠ CNN model not found. CNN features disabled.")
                self.use_cnn = False
        except Exception as e:
            self._print(f"⚠ Failed to load CNN: {e}. CNN features disabled.")
            self.cnn_model = None
            self.use_cnn = False

    # =========================================================================
    # MLP - "The Prophet" - Fast Fitness Estimation
    # =========================================================================

    def can_predict_fitness(self) -> bool:
        """Check if MLP fitness prediction is available"""
        return self.use_mlp and self.mlp_model is not None and self.mlp_scaler is not None

    def encode_config_for_mlp(self, config: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Encode configuration into feature vector for MLP.

        Args:
            config: Configuration dictionary

        Returns:
            Encoded feature vector or None if encoding fails
        """
        if not self.mlp_metadata:
            return None

        try:
            feature_cols = self.mlp_metadata['feature_columns']
            features = []

            for col in feature_cols:
                if col in ['map_rows', 'map_cols']:
                    # Parse from map_size if needed
                    if col == 'map_rows' and 'map_rows' in config:
                        features.append(config['map_rows'])
                    elif col == 'map_cols' and 'map_cols' in config:
                        features.append(config['map_cols'])
                    elif 'map_size' in config:
                        map_size = config['map_size']
                        if isinstance(map_size, (list, tuple)) and len(map_size) == 2:
                            features.append(map_size[0] if col == 'map_rows' else map_size[1])
                        else:
                            features.append(0)
                    else:
                        features.append(0)
                elif col.startswith(('lr_decay_type_', 'radius_decay_type_', 'batch_growth_type_')):
                    # One-hot encoded categorical
                    base_feature = col.rsplit('_', 1)[0]  # e.g., 'lr_decay_type'
                    value = col.split('_')[-1]  # e.g., 'exp-drop'
                    features.append(1 if config.get(base_feature) == value else 0)
                else:
                    # Numeric feature
                    features.append(config.get(col, 0))

            return np.array(features).reshape(1, -1)

        except Exception as e:
            self._print(f"⚠ Failed to encode config: {e}")
            return None

    def predict_fitness(self, config: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        """
        Predict SOM quality metrics from configuration.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (mqe, topographic_error, dead_neuron_ratio) or None
        """
        if not self.can_predict_fitness():
            return None

        try:
            # Encode configuration
            features = self.encode_config_for_mlp(config)
            if features is None:
                return None

            # Scale features
            features_scaled = self.mlp_scaler.transform(features)

            # Predict
            prediction = self.mlp_model.predict(features_scaled, verbose=0)[0]

            return tuple(prediction)  # (mqe, topo_error, dead_ratio)

        except Exception as e:
            self._print(f"⚠ MLP prediction failed: {e}")
            return None

    # =========================================================================
    # LSTM - "The Oracle" - Early Stopping
    # =========================================================================

    def can_check_early_stopping(self) -> bool:
        """Check if LSTM early stopping is available"""
        return self.use_lstm and self.lstm_model is not None

    def should_stop_early(self, training_history: Dict[str, list],
                         quality_threshold: float = 1.0) -> Tuple[bool, Optional[float]]:
        """
        Check if training should be stopped early based on predicted final quality.

        Args:
            training_history: Dict with lists of 'mqe', 'topographic_error', 'dead_neuron_ratio'
            quality_threshold: Threshold for stopping (higher = more likely to stop)

        Returns:
            Tuple of (should_stop, predicted_quality_score)
        """
        if not self.can_check_early_stopping():
            return False, None

        try:
            # Prepare sequence
            sequence = np.array([
                training_history['mqe'],
                training_history['topographic_error'],
                training_history['dead_neuron_ratio']
            ]).T  # Shape: (num_checkpoints, 3)

            # Expand dims for batch
            sequence_batch = np.expand_dims(sequence, axis=0)

            # Predict final quality
            prediction = self.lstm_model.predict(sequence_batch, verbose=0)[0]
            final_mqe, final_topo, final_dead = prediction

            # Calculate quality score (lower is better)
            quality_score = final_mqe * 1.0 + final_topo * 1.0 + final_dead * 0.5

            should_stop = quality_score > quality_threshold

            return should_stop, float(quality_score)

        except Exception as e:
            self._print(f"⚠ LSTM prediction failed: {e}")
            return False, None

    # =========================================================================
    # CNN - "The Eye" - Visual Quality Assessment
    # =========================================================================

    def can_assess_visual_quality(self) -> bool:
        """Check if CNN visual assessment is available"""
        return self.use_cnn and self.cnn_model is not None

    def assess_visual_quality(self, rgb_image_path: str, threshold: float = 0.5) -> Optional[Tuple[float, str]]:
        """
        Assess SOM quality from RGB visualization.

        Args:
            rgb_image_path: Path to RGB SOM image
            threshold: Threshold for good/bad classification

        Returns:
            Tuple of (quality_score, quality_label) or None
        """
        if not self.can_assess_visual_quality():
            return None

        try:
            from PIL import Image

            # Load and preprocess image
            img = Image.open(rgb_image_path).convert('RGB')
            img = img.resize((224, 224), Image.LANCZOS)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_batch = np.expand_dims(img_array, axis=0)

            # Predict
            prediction = self.cnn_model.predict(img_batch, verbose=0)[0][0]
            quality_label = "GOOD" if prediction >= threshold else "BAD"

            return float(prediction), quality_label

        except Exception as e:
            self._print(f"⚠ CNN prediction failed: {e}")
            return None

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of NN integration status"""
        return {
            'mlp_enabled': self.use_mlp,
            'mlp_available': self.can_predict_fitness(),
            'lstm_enabled': self.use_lstm,
            'lstm_available': self.can_check_early_stopping(),
            'cnn_enabled': self.use_cnn,
            'cnn_available': self.can_assess_visual_quality(),
            'tensorflow_loaded': _tf_loaded
        }


# ============================================================================
# Convenience Functions
# ============================================================================

def create_nn_integration(use_mlp: bool = False,
                         use_lstm: bool = False,
                         use_cnn: bool = False,
                         **kwargs) -> NeuralNetworkIntegration:
    """
    Create neural network integration with sensible defaults.

    Args:
        use_mlp: Enable MLP (fitness prediction)
        use_lstm: Enable LSTM (early stopping)
        use_cnn: Enable CNN (visual quality)
        **kwargs: Additional arguments for NeuralNetworkIntegration

    Returns:
        NeuralNetworkIntegration instance
    """
    return NeuralNetworkIntegration(
        use_mlp=use_mlp,
        use_lstm=use_lstm,
        use_cnn=use_cnn,
        **kwargs
    )


if __name__ == "__main__":
    # Test NN integration
    print("Testing Neural Network Integration...\n")

    # Test without NN (should work fine)
    print("1. Testing without NN:")
    nn_disabled = NeuralNetworkIntegration(use_mlp=False, use_lstm=False, use_cnn=False)
    print(f"   Status: {nn_disabled.get_status_summary()}\n")

    # Test with NN (if available)
    print("2. Testing with all NN enabled:")
    nn_enabled = NeuralNetworkIntegration(use_mlp=True, use_lstm=True, use_cnn=True)
    status = nn_enabled.get_status_summary()
    print(f"   Status: {status}")

    if nn_enabled.can_predict_fitness():
        print("\n3. Testing MLP prediction:")
        test_config = {
            'map_size': [15, 15],
            'start_learning_rate': 0.5,
            'end_learning_rate': 0.01,
            'lr_decay_type': 'exp-drop',
            'start_radius': 5.0,
            'end_radius': 1.0,
            'radius_decay_type': 'exp-drop',
            'start_batch_percent': 0.1,
            'end_batch_percent': 1.0,
            'batch_growth_type': 'linear-growth',
            'epoch_multiplier': 10.0,
            'normalize_weights_flag': True,
            'growth_g': 2.0,
            'num_batches': 10
        }
        prediction = nn_enabled.predict_fitness(test_config)
        print(f"   Predicted quality: {prediction}")

    print("\n✓ Neural Network Integration test complete!")
