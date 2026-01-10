"""
Model Evaluation Script

This script evaluates a trained model on the test set and generates
visualizations and detailed metrics.

Usage:
    python src/evaluate.py --model models/som_quality_best.keras --test-csv models/test_set.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from predict import preprocess_image


def load_test_data(test_csv_path):
    """
    Load test dataset from CSV file.

    Args:
        test_csv_path: Path to test set CSV file

    Returns:
        DataFrame with test data
    """
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test set file not found: {test_csv_path}")

    df = pd.read_csv(test_csv_path)
    print(f"Loaded {len(df)} test samples")
    return df


def evaluate_model(model, test_df, target_size=(224, 224)):
    """
    Evaluate model on test set.

    Args:
        model: Trained Keras model
        test_df: DataFrame with 'filepath' and 'quality_score' columns
        target_size: Image size for preprocessing

    Returns:
        Dictionary with predictions and ground truth
    """
    print("\nEvaluating model on test set...")

    predictions = []
    ground_truth = []
    filepaths = []

    for idx, row in test_df.iterrows():
        try:
            # Load and preprocess image
            img_array = preprocess_image(row['filepath'], target_size)

            # Predict
            pred = model.predict(img_array, verbose=0)
            pred_score = float(pred[0][0])

            predictions.append(pred_score)
            ground_truth.append(row['quality_score'])
            filepaths.append(row['filepath'])

            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(test_df)} samples...")

        except Exception as e:
            print(f"Warning: Failed to process {row['filepath']}: {e}")

    return {
        'predictions': np.array(predictions),
        'ground_truth': np.array(ground_truth),
        'filepaths': filepaths
    }


def calculate_metrics(predictions, ground_truth):
    """
    Calculate evaluation metrics.

    Args:
        predictions: Array of predicted values
        ground_truth: Array of true values

    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(ground_truth, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ground_truth, predictions)
    r2 = r2_score(ground_truth, predictions)

    # Calculate percentage errors
    errors = np.abs(predictions - ground_truth)
    mean_error_pct = np.mean(errors) * 100
    median_error_pct = np.median(errors) * 100

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Mean Error %': mean_error_pct,
        'Median Error %': median_error_pct
    }

    return metrics


def plot_predictions(predictions, ground_truth, output_path):
    """
    Create scatter plot of predictions vs ground truth.

    Args:
        predictions: Array of predicted values
        ground_truth: Array of true values
        output_path: Path to save the plot
    """
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(ground_truth, predictions, alpha=0.5, s=50)

    # Perfect prediction line
    min_val = min(ground_truth.min(), predictions.min())
    max_val = max(ground_truth.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

    plt.xlabel('Ground Truth Quality Score', fontsize=12)
    plt.ylabel('Predicted Quality Score', fontsize=12)
    plt.title('Model Predictions vs Ground Truth', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved prediction plot to: {output_path}")
    plt.close()


def plot_error_distribution(predictions, ground_truth, output_path):
    """
    Create histogram of prediction errors.

    Args:
        predictions: Array of predicted values
        ground_truth: Array of true values
        output_path: Path to save the plot
    """
    errors = predictions - ground_truth

    plt.figure(figsize=(10, 6))

    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    plt.axvline(x=np.mean(errors), color='g', linestyle='--', linewidth=2, label=f'Mean Error: {np.mean(errors):.4f}')

    plt.xlabel('Prediction Error (Predicted - Ground Truth)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved error distribution plot to: {output_path}")
    plt.close()


def plot_residuals(predictions, ground_truth, output_path):
    """
    Create residual plot.

    Args:
        predictions: Array of predicted values
        ground_truth: Array of true values
        output_path: Path to save the plot
    """
    residuals = predictions - ground_truth

    plt.figure(figsize=(10, 6))

    plt.scatter(predictions, residuals, alpha=0.5, s=50)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)

    plt.xlabel('Predicted Quality Score', fontsize=12)
    plt.ylabel('Residuals (Predicted - Ground Truth)', fontsize=12)
    plt.title('Residual Plot', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved residual plot to: {output_path}")
    plt.close()


def create_evaluation_report(metrics, predictions, ground_truth, output_dir):
    """
    Create a comprehensive evaluation report.

    Args:
        metrics: Dictionary of evaluation metrics
        predictions: Array of predicted values
        ground_truth: Array of true values
        output_dir: Directory to save report and plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Print metrics
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name:<20}: {metric_value:.6f}")
    print("=" * 80)

    # Save metrics to file
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("SOM Quality Analyzer - Evaluation Metrics\n")
        f.write("=" * 60 + "\n\n")
        for metric_name, metric_value in metrics.items():
            f.write(f"{metric_name:<20}: {metric_value:.6f}\n")
        f.write("\n" + "=" * 60 + "\n")
    print(f"\nSaved metrics to: {metrics_path}")

    # Create plots
    print("\nGenerating evaluation plots...")
    plot_predictions(
        predictions, ground_truth,
        os.path.join(output_dir, 'predictions_vs_truth.png')
    )
    plot_error_distribution(
        predictions, ground_truth,
        os.path.join(output_dir, 'error_distribution.png')
    )
    plot_residuals(
        predictions, ground_truth,
        os.path.join(output_dir, 'residual_plot.png')
    )

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate trained SOM quality prediction model'
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model file'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        required=True,
        help='Path to test set CSV file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='evaluation_results',
        help='Directory to save evaluation results'
    )
    parser.add_argument(
        '--image-size',
        type=int,
        default=224,
        help='Image size for preprocessing'
    )

    return parser.parse_args()


def main():
    """Main evaluation function."""
    args = parse_arguments()

    print("=" * 80)
    print("SOM QUALITY ANALYZER - MODEL EVALUATION")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from: {args.model}")
    model = keras.models.load_model(args.model)
    print("Model loaded successfully!")

    # Load test data
    print(f"\nLoading test data from: {args.test_csv}")
    test_df = load_test_data(args.test_csv)

    # Evaluate
    target_size = (args.image_size, args.image_size)
    results = evaluate_model(model, test_df, target_size)

    # Calculate metrics
    print("\nCalculating evaluation metrics...")
    metrics = calculate_metrics(results['predictions'], results['ground_truth'])

    # Create report
    create_evaluation_report(
        metrics,
        results['predictions'],
        results['ground_truth'],
        args.output_dir
    )

    # Save predictions
    results_df = pd.DataFrame({
        'filepath': results['filepaths'],
        'ground_truth': results['ground_truth'],
        'prediction': results['predictions'],
        'error': results['predictions'] - results['ground_truth'],
        'absolute_error': np.abs(results['predictions'] - results['ground_truth'])
    })
    results_df = results_df.sort_values('absolute_error', ascending=False)

    results_csv = os.path.join(args.output_dir, 'detailed_predictions.csv')
    results_df.to_csv(results_csv, index=False)
    print(f"\nSaved detailed predictions to: {results_csv}")

    # Show worst predictions
    print("\n" + "=" * 80)
    print("TOP 10 WORST PREDICTIONS (Highest Absolute Error)")
    print("=" * 80)
    for idx, row in results_df.head(10).iterrows():
        print(f"File: {Path(row['filepath']).name}")
        print(f"  Ground Truth: {row['ground_truth']:.4f}")
        print(f"  Prediction:   {row['prediction']:.4f}")
        print(f"  Error:        {row['error']:.4f}")
        print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nEvaluation failed: {e}")
        raise
