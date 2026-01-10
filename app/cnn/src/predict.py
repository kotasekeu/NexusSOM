"""
Prediction Script for SOM Quality Evaluation

This script loads a trained model and predicts the quality score
for new, unseen SOM visualization images.

Usage:
    # Single image prediction
    python src/predict.py --model models/som_quality_best.keras --image data/raw_maps/example.png

    # Batch prediction on a directory
    python src/predict.py --model models/som_quality_best.keras --image-dir data/raw_maps/ --output predictions.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import tensorflow as tf
from tensorflow import keras


def load_trained_model(model_path):
    """
    Load a trained Keras model.

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded Keras model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    return model


def preprocess_image(image_path, target_size=(224, 224)):
    """
    Load and preprocess a single image for prediction.

    Args:
        image_path: Path to the image file
        target_size: Target size (height, width) for the image

    Returns:
        Preprocessed image array ready for model input
    """
    try:
        # Load image
        img = Image.open(image_path).convert('RGB')

        # Resize to target size
        img = img.resize(target_size, Image.LANCZOS)

        # Convert to array and normalize to [0, 1]
        img_array = np.array(img, dtype=np.float32) / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    except Exception as e:
        raise ValueError(f"Error processing image {image_path}: {e}")


def predict_single_image(model, image_path, target_size=(224, 224)):
    """
    Predict quality score for a single image.

    Args:
        model: Trained Keras model
        image_path: Path to the image file
        target_size: Target size for image preprocessing

    Returns:
        Predicted quality score (float)
    """
    # Preprocess image
    img_array = preprocess_image(image_path, target_size)

    # Make prediction
    prediction = model.predict(img_array, verbose=0)

    # Extract scalar value
    quality_score = float(prediction[0][0])

    return quality_score


def predict_batch(model, image_paths, target_size=(224, 224), batch_size=32):
    """
    Predict quality scores for multiple images in batches.

    Args:
        model: Trained Keras model
        image_paths: List of paths to image files
        target_size: Target size for image preprocessing
        batch_size: Number of images to process at once

    Returns:
        List of predicted quality scores
    """
    predictions = []
    num_images = len(image_paths)

    print(f"Predicting quality scores for {num_images} images...")

    for i in range(0, num_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        # Load and preprocess batch
        for img_path in batch_paths:
            try:
                img_array = preprocess_image(img_path, target_size)
                batch_images.append(img_array[0])  # Remove batch dimension
            except Exception as e:
                print(f"Warning: Skipping {img_path} - {e}")
                batch_images.append(np.zeros((*target_size, 3), dtype=np.float32))

        # Stack into batch
        batch_images = np.array(batch_images)

        # Predict
        batch_predictions = model.predict(batch_images, verbose=0)

        # Extract scores
        for pred in batch_predictions:
            predictions.append(float(pred[0]))

        # Progress update
        processed = min(i + batch_size, num_images)
        print(f"  Processed {processed}/{num_images} images...")

    return predictions


def predict_directory(model, image_dir, output_csv=None, target_size=(224, 224)):
    """
    Predict quality scores for all images in a directory.

    Args:
        model: Trained Keras model
        image_dir: Directory containing images
        output_csv: Optional path to save results as CSV
        target_size: Target size for image preprocessing

    Returns:
        DataFrame with filenames and predicted scores
    """
    # Get all image files
    image_dir = Path(image_dir)
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    image_paths = [
        p for p in image_dir.glob('*')
        if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        raise ValueError(f"No images found in directory: {image_dir}")

    print(f"Found {len(image_paths)} images in {image_dir}")

    # Predict in batches
    predictions = predict_batch(model, image_paths, target_size)

    # Create results DataFrame
    results = pd.DataFrame({
        'filename': [p.name for p in image_paths],
        'filepath': [str(p) for p in image_paths],
        'predicted_quality_score': predictions
    })

    # Sort by score (descending)
    results = results.sort_values('predicted_quality_score', ascending=False)

    # Save to CSV if requested
    if output_csv:
        results.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

    return results


def print_prediction_summary(results_df):
    """
    Print a summary of prediction results.

    Args:
        results_df: DataFrame with prediction results
    """
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY")
    print("=" * 80)
    print(f"Total images: {len(results_df)}")
    print(f"Average quality score: {results_df['predicted_quality_score'].mean():.4f}")
    print(f"Median quality score: {results_df['predicted_quality_score'].median():.4f}")
    print(f"Min quality score: {results_df['predicted_quality_score'].min():.4f}")
    print(f"Max quality score: {results_df['predicted_quality_score'].max():.4f}")
    print(f"Std quality score: {results_df['predicted_quality_score'].std():.4f}")

    print("\n" + "-" * 80)
    print("TOP 10 HIGHEST QUALITY MAPS:")
    print("-" * 80)
    top_10 = results_df.head(10)
    for idx, row in top_10.iterrows():
        print(f"  {row['filename']:<50} Score: {row['predicted_quality_score']:.4f}")

    print("\n" + "-" * 80)
    print("TOP 10 LOWEST QUALITY MAPS:")
    print("-" * 80)
    bottom_10 = results_df.tail(10)
    for idx, row in bottom_10.iterrows():
        print(f"  {row['filename']:<50} Score: {row['predicted_quality_score']:.4f}")

    print("=" * 80)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Predict SOM quality scores using trained CNN model'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.keras or .h5)'
    )
    parser.add_argument(
        '--image',
        type=str,
        help='Path to a single image file for prediction'
    )
    parser.add_argument(
        '--image-dir',
        type=str,
        help='Directory containing images for batch prediction'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output CSV file for batch predictions'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size for preprocessing (default: 224)'
    )

    return parser.parse_args()


def main():
    """Main function for prediction script."""
    args = parse_arguments()

    # Validate arguments
    if not args.image and not args.image_dir:
        raise ValueError("Either --image or --image-dir must be specified")

    target_size = (args.image_size, args.image_size)

    # Load model
    print("=" * 80)
    print("SOM QUALITY PREDICTION")
    print("=" * 80)
    model = load_trained_model(args.model)

    # Single image prediction
    if args.image:
        print(f"\nPredicting quality score for: {args.image}")
        quality_score = predict_single_image(model, args.image, target_size)
        print("\n" + "=" * 80)
        print(f"PREDICTED QUALITY SCORE: {quality_score:.6f}")
        print("=" * 80)

        # Interpret the score
        if quality_score >= 0.8:
            quality_level = "Excellent"
        elif quality_score >= 0.6:
            quality_level = "Good"
        elif quality_score >= 0.4:
            quality_level = "Fair"
        else:
            quality_level = "Poor"

        print(f"Quality Level: {quality_level}")
        print("=" * 80)

    # Batch prediction
    if args.image_dir:
        print(f"\nPredicting quality scores for images in: {args.image_dir}")
        results = predict_directory(
            model,
            args.image_dir,
            output_csv=args.output,
            target_size=target_size
        )
        print_prediction_summary(results)

        if not args.output:
            print("\nNote: Use --output to save results to CSV file")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        raise
