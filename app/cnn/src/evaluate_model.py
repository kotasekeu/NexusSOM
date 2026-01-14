#!/usr/bin/env python3
"""
Evaluate Trained CNN Model on SOM Maps

This script loads a trained model and evaluates it on test data or new SOM maps.

Usage:
    # Evaluate on test set
    python3 evaluate_model.py --model models/som_quality_standard_20260112_210424_best.keras \\
        --test_set models/som_quality_standard_20260112_210424_test_set.csv

    # Predict quality for new maps
    python3 evaluate_model.py --model models/som_quality_standard_20260112_210424_best.keras \\
        --predict ./test/results/20260112_133705/maps_dataset/rgb
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

import tensorflow as tf
from tensorflow import keras


def load_model(model_path):
    """Load trained Keras model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    model = keras.models.load_model(model_path)
    print(f"✓ Model loaded successfully")
    return model


def load_and_preprocess_image(filepath, image_size=(224, 224)):
    """Load and preprocess a single image"""
    try:
        img = Image.open(filepath).convert('RGB')
        img = img.resize(image_size, Image.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return img_array
    except Exception as e:
        print(f"Error loading image {filepath}: {e}")
        return None


def evaluate_on_test_set(model, test_set_path, image_size=(224, 224)):
    """Evaluate model on the saved test set"""
    print(f"\n{'='*80}")
    print("EVALUATING ON TEST SET")
    print(f"{'='*80}\n")

    # Load test set
    test_df = pd.read_csv(test_set_path)
    print(f"Test set size: {len(test_df)} samples")

    # Load images and labels
    images = []
    labels = []

    for _, row in test_df.iterrows():
        img = load_and_preprocess_image(row['filepath'], image_size)
        if img is not None:
            images.append(img)
            labels.append(row['quality_score'])

    images = np.array(images)
    labels = np.array(labels)

    print(f"Loaded {len(images)} images")

    # Predict
    predictions = model.predict(images, verbose=1)
    predictions = predictions.flatten()

    # Calculate metrics
    errors = np.abs(predictions - labels)
    mae = np.mean(errors)
    rmse = np.sqrt(np.mean((predictions - labels) ** 2))

    # Classification metrics (threshold at 0.5)
    pred_classes = (predictions >= 0.5).astype(int)
    true_classes = (labels >= 0.5).astype(int)

    accuracy = np.mean(pred_classes == true_classes)
    tp = np.sum((pred_classes == 1) & (true_classes == 1))
    fp = np.sum((pred_classes == 1) & (true_classes == 0))
    fn = np.sum((pred_classes == 0) & (true_classes == 1))
    tn = np.sum((pred_classes == 0) & (true_classes == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Print results
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}\n")

    print("Regression Metrics:")
    print(f"  Mean Absolute Error (MAE): {mae:.6f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.6f}")

    print(f"\nClassification Metrics (threshold=0.5):")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  True Negatives: {tn}")
    print(f"  False Negatives: {fn}")

    # Show some predictions
    print(f"\n{'='*80}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*80}\n")

    for i in range(min(10, len(predictions))):
        pred_label = "GOOD" if predictions[i] >= 0.5 else "BAD"
        true_label = "GOOD" if labels[i] >= 0.5 else "BAD"
        match = "✓" if pred_label == true_label else "✗"
        print(f"{match} Sample {i+1}: Predicted={predictions[i]:.4f} ({pred_label}), True={labels[i]:.1f} ({true_label})")

    return predictions, labels


def predict_on_directory(model, rgb_dir, image_size=(224, 224), threshold=0.5):
    """Predict quality scores for all RGB images in a directory"""
    print(f"\n{'='*80}")
    print("PREDICTING ON NEW IMAGES")
    print(f"{'='*80}\n")

    if not os.path.exists(rgb_dir):
        raise FileNotFoundError(f"Directory not found: {rgb_dir}")

    # Find all RGB images
    rgb_files = []
    for filename in os.listdir(rgb_dir):
        if filename.endswith('_rgb.png'):
            uid = filename.replace('_rgb.png', '')
            filepath = os.path.join(rgb_dir, filename)
            rgb_files.append({'uid': uid, 'filepath': filepath})

    if len(rgb_files) == 0:
        print("No RGB images found in directory")
        return

    print(f"Found {len(rgb_files)} RGB images")

    # Load images
    results = []
    for item in rgb_files:
        img = load_and_preprocess_image(item['filepath'], image_size)
        if img is not None:
            img_batch = np.expand_dims(img, axis=0)
            prediction = model.predict(img_batch, verbose=0)[0][0]

            quality_label = "GOOD" if prediction >= threshold else "BAD"
            confidence = prediction if prediction >= 0.5 else (1 - prediction)

            results.append({
                'uid': item['uid'],
                'quality_score': prediction,
                'quality_label': quality_label,
                'confidence': confidence
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Sort by quality score (descending)
    results_df = results_df.sort_values('quality_score', ascending=False)

    # Print summary
    print(f"\n{'='*80}")
    print("PREDICTION SUMMARY")
    print(f"{'='*80}\n")

    good_count = (results_df['quality_score'] >= threshold).sum()
    bad_count = (results_df['quality_score'] < threshold).sum()

    print(f"Total predictions: {len(results_df)}")
    print(f"  Good (score >= {threshold}): {good_count}")
    print(f"  Bad (score < {threshold}): {bad_count}")

    print(f"\nTop 10 Best Quality Maps:")
    print(results_df.head(10)[['uid', 'quality_score', 'quality_label', 'confidence']].to_string(index=False))

    print(f"\nTop 10 Worst Quality Maps:")
    print(results_df.tail(10)[['uid', 'quality_score', 'quality_label', 'confidence']].to_string(index=False))

    # Save results
    output_file = os.path.join(rgb_dir, "cnn_predictions.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Predictions saved to: {output_file}")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Trained CNN Model on SOM Maps',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file (.keras)'
    )

    parser.add_argument(
        '--test_set',
        type=str,
        help='Path to test set CSV file (for evaluation)'
    )

    parser.add_argument(
        '--predict',
        type=str,
        help='Directory containing RGB images to predict (for inference)'
    )

    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size for model input (default: 224)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Threshold for good/bad classification (default: 0.5)'
    )

    args = parser.parse_args()

    # Load model
    model = load_model(args.model)
    image_size = (args.image_size, args.image_size)

    # Evaluate or predict
    if args.test_set:
        evaluate_on_test_set(model, args.test_set, image_size)

    if args.predict:
        predict_on_directory(model, args.predict, image_size, args.threshold)

    if not args.test_set and not args.predict:
        print("\nError: Please specify either --test_set or --predict")
        sys.exit(1)


if __name__ == "__main__":
    main()
