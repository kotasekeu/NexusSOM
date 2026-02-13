#!/usr/bin/env python3
"""
Test SOM Map to Pixel Conversion

Demonstrates how SOM weight maps are converted to images for CNN input.
Shows that the CNN sees structural patterns, not just size.

Key insight: We need to ensure CNN learns QUALITY patterns, not SIZE patterns.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_mock_som_weights(grid_size: tuple, n_features: int, pattern: str = 'good') -> np.ndarray:
    """
    Create mock SOM weight matrix simulating different quality patterns.

    Args:
        grid_size: (rows, cols) of SOM grid
        n_features: Number of input features (dimensions)
        pattern: 'good', 'bad', or 'medium' quality

    Returns:
        Weight matrix of shape (rows, cols, n_features)
    """
    rows, cols = grid_size
    weights = np.zeros((rows, cols, n_features))

    if pattern == 'good':
        # Good map: smooth gradients, all neurons active, organized topology
        for i in range(rows):
            for j in range(cols):
                # Smooth gradient based on position
                weights[i, j, :] = np.linspace(
                    (i + j) / (rows + cols),
                    1 - (i + j) / (rows + cols),
                    n_features
                )
                # Add small noise for realism
                weights[i, j, :] += np.random.normal(0, 0.05, n_features)

    elif pattern == 'bad':
        # Bad map: many dead neurons, random patches, poor topology
        for i in range(rows):
            for j in range(cols):
                if np.random.random() < 0.3:  # 30% dead neurons
                    weights[i, j, :] = 0
                else:
                    # Random values, no organization
                    weights[i, j, :] = np.random.random(n_features)

    else:  # medium
        # Medium: some organization but with issues
        for i in range(rows):
            for j in range(cols):
                base = np.linspace(i / rows, j / cols, n_features)
                noise = np.random.normal(0, 0.2, n_features)
                weights[i, j, :] = np.clip(base + noise, 0, 1)
                if np.random.random() < 0.1:  # 10% dead neurons
                    weights[i, j, :] = 0

    return np.clip(weights, 0, 1)


def weights_to_rgb_image(weights: np.ndarray) -> np.ndarray:
    """
    Convert SOM weight matrix to RGB image.

    Current approach: Use first 3 dimensions as RGB.
    Alternative approaches could be:
    - PCA to reduce to 3 components
    - U-matrix visualization
    - Component planes

    Args:
        weights: Shape (rows, cols, n_features)

    Returns:
        RGB image of shape (rows, cols, 3) normalized to [0, 1]
    """
    rows, cols, n_features = weights.shape

    if n_features >= 3:
        # Use first 3 features as RGB
        rgb = weights[:, :, :3].copy()
    elif n_features == 2:
        # Use 2 features + their mean
        rgb = np.stack([
            weights[:, :, 0],
            weights[:, :, 1],
            (weights[:, :, 0] + weights[:, :, 1]) / 2
        ], axis=-1)
    else:
        # Single feature - grayscale to RGB
        rgb = np.stack([weights[:, :, 0]] * 3, axis=-1)

    # Normalize to [0, 1]
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

    return rgb.astype(np.float32)


def weights_to_umatrix_image(weights: np.ndarray) -> np.ndarray:
    """
    Convert SOM weights to U-matrix visualization.

    U-matrix shows distances between neighboring neurons.
    Dark = similar neighbors (good clustering)
    Light = different neighbors (cluster boundaries)

    Args:
        weights: Shape (rows, cols, n_features)

    Returns:
        RGB image of shape (rows, cols, 3)
    """
    rows, cols, n_features = weights.shape
    umatrix = np.zeros((rows, cols))

    for i in range(rows):
        for j in range(cols):
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    dist = np.linalg.norm(weights[i, j] - weights[ni, nj])
                    neighbors.append(dist)
            umatrix[i, j] = np.mean(neighbors) if neighbors else 0

    # Normalize
    umatrix = (umatrix - umatrix.min()) / (umatrix.max() - umatrix.min() + 1e-8)

    # Convert to RGB (grayscale)
    rgb = np.stack([umatrix] * 3, axis=-1)

    return rgb.astype(np.float32)


def calculate_quality_metrics(weights: np.ndarray) -> dict:
    """
    Calculate quality metrics from SOM weights.

    These metrics are what CNN should learn to predict.
    """
    rows, cols, n_features = weights.shape

    # Dead neuron ratio (neurons with zero or near-zero weights)
    neuron_magnitudes = np.linalg.norm(weights, axis=2)
    dead_threshold = 0.01
    dead_ratio = np.mean(neuron_magnitudes < dead_threshold)

    # Topographic organization (smoothness of weight changes)
    topo_errors = []
    for i in range(rows):
        for j in range(cols):
            for di, dj in [(0, 1), (1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < rows and 0 <= nj < cols:
                    diff = np.linalg.norm(weights[i, j] - weights[ni, nj])
                    topo_errors.append(diff)

    avg_neighbor_dist = np.mean(topo_errors) if topo_errors else 0

    # Weight variance (diversity of neurons)
    weight_variance = np.var(weights)

    return {
        'dead_ratio': dead_ratio,
        'avg_neighbor_distance': avg_neighbor_dist,
        'weight_variance': weight_variance,
        'quality_estimate': 1.0 - dead_ratio - (avg_neighbor_dist * 0.5)
    }


def test_conversion():
    """
    Test and visualize the conversion process for different map sizes.
    """
    print("=" * 80)
    print("SOM MAP TO CNN IMAGE CONVERSION TEST")
    print("=" * 80)

    # Test configurations
    configs = [
        {'size': (5, 5), 'features': 10, 'pattern': 'good'},
        {'size': (5, 5), 'features': 10, 'pattern': 'bad'},
        {'size': (20, 20), 'features': 10, 'pattern': 'good'},
        {'size': (20, 20), 'features': 10, 'pattern': 'bad'},
    ]

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    fig.suptitle('SOM Map Conversion: Size vs Quality Patterns', fontsize=14)

    for idx, config in enumerate(configs):
        size = config['size']
        pattern = config['pattern']
        n_features = config['features']

        # Create mock weights
        weights = create_mock_som_weights(size, n_features, pattern)

        # Convert to images
        rgb_img = weights_to_rgb_image(weights)
        umatrix_img = weights_to_umatrix_image(weights)

        # Calculate metrics
        metrics = calculate_quality_metrics(weights)

        # Display
        row = idx

        # RGB visualization
        axes[row, 0].imshow(rgb_img, interpolation='nearest')
        axes[row, 0].set_title(f'{size[0]}x{size[1]} {pattern.upper()}\nRGB (first 3 dims)')
        axes[row, 0].axis('off')

        # U-matrix
        axes[row, 1].imshow(umatrix_img, interpolation='nearest', cmap='viridis')
        axes[row, 1].set_title(f'U-Matrix\n(neighbor distances)')
        axes[row, 1].axis('off')

        # Metrics text
        axes[row, 2].axis('off')
        metrics_text = (
            f"Size: {size[0]}x{size[1]}\n"
            f"Pattern: {pattern}\n"
            f"Dead ratio: {metrics['dead_ratio']:.2%}\n"
            f"Neighbor dist: {metrics['avg_neighbor_distance']:.3f}\n"
            f"Variance: {metrics['weight_variance']:.3f}\n"
            f"Quality est: {metrics['quality_estimate']:.2f}"
        )
        axes[row, 2].text(0.1, 0.5, metrics_text, fontsize=12,
                          verticalalignment='center', family='monospace')
        axes[row, 2].set_title('Quality Metrics')

        print(f"\n{size[0]}x{size[1]} {pattern.upper()} map:")
        print(f"  Image shape: {rgb_img.shape}")
        print(f"  Dead ratio: {metrics['dead_ratio']:.2%}")
        print(f"  Quality estimate: {metrics['quality_estimate']:.2f}")

    plt.tight_layout()

    # Save figure
    output_dir = Path(__file__).parent.parent / 'tests'
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / 'map_conversion_test.png'
    plt.savefig(output_path, dpi=150)
    print(f"\n\nVisualization saved to: {output_path}")
    plt.show()


def demonstrate_cnn_input():
    """
    Show exactly what CNN receives as input.
    """
    print("\n" + "=" * 80)
    print("CNN INPUT DEMONSTRATION")
    print("=" * 80)

    # Small good map
    small_good = create_mock_som_weights((5, 5), 10, 'good')
    small_good_img = weights_to_rgb_image(small_good)

    # Large bad map
    large_bad = create_mock_som_weights((20, 20), 10, 'bad')
    large_bad_img = weights_to_rgb_image(large_bad)

    print("\nSmall (5x5) GOOD map:")
    print(f"  Weights shape: {small_good.shape}")
    print(f"  CNN input shape: {small_good_img.shape}")
    print(f"  CNN input (batch): (1, 5, 5, 3)")

    print("\nLarge (20x20) BAD map:")
    print(f"  Weights shape: {large_bad.shape}")
    print(f"  CNN input shape: {large_bad_img.shape}")
    print(f"  CNN input (batch): (1, 20, 20, 3)")

    print("\n" + "-" * 40)
    print("KEY POINT:")
    print("-" * 40)
    print("""
CNN with GAP architecture:
- 5x5 input  -> Conv layers -> 5x5x256 features  -> GAP -> 256 values -> prediction
- 20x20 input -> Conv layers -> 20x20x256 features -> GAP -> 256 values -> prediction

The GAP (Global Average Pooling) averages ALL spatial positions.
This means the CNN learns PATTERNS in the feature maps, not SIZE.

HOWEVER, there's still a risk:
- Small maps have fewer pixels to average -> different statistics
- The CNN might still learn size-correlated features

SOLUTION OPTIONS:
1. Resize all maps to same size (loses "1 neuron = 1 pixel" benefit)
2. Add map size as explicit input to Dense layers
3. Use data augmentation with size variations
4. Normalize by size in the visualization
""")


if __name__ == '__main__':
    test_conversion()
    demonstrate_cnn_input()
