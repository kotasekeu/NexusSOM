#!/usr/bin/env python3
"""
Export Keras Model for Google Colab Compatibility

This script converts a .keras model to SavedModel format or
exports just the weights for better cross-version compatibility.

Usage:
    python3 export_model_for_colab.py --model models/som_quality_standard_20260112_210424_best.keras
    python3 export_model_for_colab.py --model models/som_quality_standard_20260112_210424_best.keras --format savedmodel
    python3 export_model_for_colab.py --model models/som_quality_standard_20260112_210424_best.keras --format h5
"""

import argparse
import os
import sys
import tensorflow as tf
from tensorflow import keras
import json

def export_to_savedmodel(model_path, output_dir):
    """Export model to TensorFlow SavedModel format"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    print(f"Exporting to SavedModel format: {output_dir}")
    # Keras 3 uses export() instead of save(save_format='tf')
    model.export(output_dir)
    print(f"✓ Model exported to: {output_dir}")

    # Save model metadata
    metadata = {
        'input_shape': list(model.input_shape),
        'output_shape': list(model.output_shape),
        'num_layers': len(model.layers),
        'total_params': model.count_params(),
        'framework': 'tensorflow',
        'format': 'savedmodel'
    }

    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_file}")

    return output_dir


def export_to_h5(model_path, output_path):
    """Export model to legacy H5 format"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    # Ensure output path has .h5 extension
    if not output_path.endswith('.h5'):
        output_path = output_path + '.h5'

    print(f"Exporting to H5 format: {output_path}")
    # Keras 3 auto-detects format from extension
    model.save(output_path)
    print(f"✓ Model exported to: {output_path}")

    return output_path


def export_weights_only(model_path, output_dir):
    """Export model architecture (JSON) and weights separately"""
    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)

    os.makedirs(output_dir, exist_ok=True)

    # Export architecture as JSON
    arch_file = os.path.join(output_dir, 'architecture.json')
    with open(arch_file, 'w') as f:
        f.write(model.to_json())
    print(f"✓ Architecture saved to: {arch_file}")

    # Export weights
    weights_file = os.path.join(output_dir, 'weights.h5')
    model.save_weights(weights_file)
    print(f"✓ Weights saved to: {weights_file}")

    # Save model metadata
    metadata = {
        'input_shape': list(model.input_shape),
        'output_shape': list(model.output_shape),
        'num_layers': len(model.layers),
        'total_params': model.count_params(),
        'optimizer': model.optimizer.get_config() if model.optimizer else None,
        'loss': model.loss if isinstance(model.loss, str) else str(model.loss),
        'format': 'weights_only'
    }

    metadata_file = os.path.join(output_dir, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved to: {metadata_file}")

    # Create loader script
    loader_script = os.path.join(output_dir, 'load_model.py')
    with open(loader_script, 'w') as f:
        f.write("""
# Load model from architecture + weights
from tensorflow import keras
import json

# Load architecture
with open('architecture.json', 'r') as f:
    model = keras.models.model_from_json(f.read())

# Load weights
model.load_weights('weights.h5')

# Compile model (adjust optimizer/loss as needed)
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

print("✓ Model loaded successfully!")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
""")
    print(f"✓ Loader script saved to: {loader_script}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description='Export Keras Model for Google Colab Compatibility',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to SavedModel format (recommended)
  python3 export_model_for_colab.py --model models/best_model.keras --format savedmodel

  # Export to H5 format
  python3 export_model_for_colab.py --model models/best_model.keras --format h5

  # Export architecture + weights separately
  python3 export_model_for_colab.py --model models/best_model.keras --format weights
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the .keras model file'
    )

    parser.add_argument(
        '--format',
        type=str,
        choices=['savedmodel', 'h5', 'weights'],
        default='savedmodel',
        help='Export format (default: savedmodel)'
    )

    parser.add_argument(
        '--output',
        type=str,
        help='Output path (auto-generated if not specified)'
    )

    args = parser.parse_args()

    # Validate model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)

    # Generate output path if not specified
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.model))[0]
        if args.format == 'savedmodel':
            args.output = f"models/{base_name}_savedmodel"
        elif args.format == 'h5':
            args.output = f"models/{base_name}.h5"
        else:  # weights
            args.output = f"models/{base_name}_weights"

    print(f"\n{'='*80}")
    print("EXPORTING MODEL FOR GOOGLE COLAB")
    print(f"{'='*80}\n")
    print(f"Input: {args.model}")
    print(f"Format: {args.format}")
    print(f"Output: {args.output}\n")

    try:
        if args.format == 'savedmodel':
            export_to_savedmodel(args.model, args.output)
        elif args.format == 'h5':
            export_to_h5(args.model, args.output)
        else:  # weights
            export_weights_only(args.model, args.output)

        print(f"\n{'='*80}")
        print("EXPORT COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")

        if args.format == 'savedmodel':
            print(f"\nTo use in Colab:")
            print(f"  model = keras.models.load_model('{os.path.basename(args.output)}')")
        elif args.format == 'h5':
            print(f"\nTo use in Colab:")
            print(f"  model = keras.models.load_model('{os.path.basename(args.output)}')")
        else:  # weights
            print(f"\nTo use in Colab:")
            print(f"  1. Upload the entire '{os.path.basename(args.output)}' directory")
            print(f"  2. Run the load_model.py script in that directory")

        print()

    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
