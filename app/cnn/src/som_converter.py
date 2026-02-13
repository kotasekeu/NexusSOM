#!/usr/bin/env python3
"""
SOM to Image Converter

Converts SOM weight matrices to images for CNN input.
Used both during training (from saved weights) and inference (from live SOM).

The converter ensures consistent transformation regardless of map size.
"""

import numpy as np
from typing import Tuple, Optional
from PIL import Image


class SOMToImageConverter:
    """
    Converts SOM weight matrices to RGB images for CNN processing.

    Supports multiple visualization methods:
    - RGB: First 3 weight dimensions as color channels
    - U-matrix: Neighbor distance visualization
    - Component planes: Individual dimension heatmaps

    For CNN training, images can be:
    - Native size (1 neuron = 1 pixel) - requires batch_size=1
    - Resized to fixed size - allows batching but loses precision
    """

    def __init__(self, target_size: Optional[Tuple[int, int]] = None,
                 method: str = 'rgb',
                 normalize: bool = True):
        """
        Initialize converter.

        Args:
            target_size: If set, resize all outputs to this (height, width).
                         If None, use native size (1 neuron = 1 pixel).
            method: Visualization method ('rgb', 'umatrix', 'combined')
            normalize: Whether to normalize output to [0, 1]
        """
        self.target_size = target_size
        self.method = method
        self.normalize = normalize

    def convert(self, weights: np.ndarray) -> np.ndarray:
        """
        Convert SOM weights to RGB image.

        Args:
            weights: SOM weight matrix of shape (rows, cols, n_features)

        Returns:
            RGB image of shape (H, W, 3) where:
            - H, W = rows, cols if target_size is None
            - H, W = target_size otherwise
        """
        if self.method == 'rgb':
            img = self._weights_to_rgb(weights)
        elif self.method == 'umatrix':
            img = self._weights_to_umatrix(weights)
        elif self.method == 'combined':
            img = self._weights_to_combined(weights)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Resize if target size is set
        if self.target_size is not None:
            img = self._resize(img, self.target_size)

        return img

    def _weights_to_rgb(self, weights: np.ndarray) -> np.ndarray:
        """
        Convert weights to RGB using first 3 dimensions.

        For n_features < 3, duplicates channels as needed.
        """
        rows, cols, n_features = weights.shape

        if n_features >= 3:
            rgb = weights[:, :, :3].copy()
        elif n_features == 2:
            rgb = np.stack([
                weights[:, :, 0],
                weights[:, :, 1],
                (weights[:, :, 0] + weights[:, :, 1]) / 2
            ], axis=-1)
        else:
            rgb = np.stack([weights[:, :, 0]] * 3, axis=-1)

        if self.normalize:
            rgb = self._normalize(rgb)

        return rgb.astype(np.float32)

    def _weights_to_umatrix(self, weights: np.ndarray) -> np.ndarray:
        """
        Convert weights to U-matrix visualization.

        U-matrix shows distances between neighboring neurons.
        Good for visualizing cluster boundaries.
        """
        rows, cols, _ = weights.shape
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

        if self.normalize:
            umatrix = self._normalize(umatrix)

        # Convert to RGB (grayscale)
        rgb = np.stack([umatrix] * 3, axis=-1)
        return rgb.astype(np.float32)

    def _weights_to_combined(self, weights: np.ndarray) -> np.ndarray:
        """
        Combine RGB and U-matrix into single visualization.

        R, G = first 2 weight dimensions
        B = U-matrix (neighbor distances)
        """
        rows, cols, n_features = weights.shape

        # RGB channels from weights
        if n_features >= 2:
            r = weights[:, :, 0]
            g = weights[:, :, 1]
        else:
            r = g = weights[:, :, 0]

        # Blue channel from U-matrix
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

        rgb = np.stack([r, g, umatrix], axis=-1)

        if self.normalize:
            rgb = self._normalize(rgb)

        return rgb.astype(np.float32)

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """Normalize array to [0, 1] range."""
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max - arr_min > 1e-8:
            return (arr - arr_min) / (arr_max - arr_min)
        return arr - arr_min  # All same value -> zeros

    def _resize(self, img: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Resize image to target size.

        Uses NEAREST interpolation to preserve discrete neuron values.
        """
        # Convert to PIL, resize, convert back
        # Scale to 0-255 for PIL
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_resized = pil_img.resize((target_size[1], target_size[0]), Image.NEAREST)
        return np.array(pil_resized, dtype=np.float32) / 255.0

    def save_image(self, weights: np.ndarray, path: str):
        """
        Convert weights to image and save to file.

        Args:
            weights: SOM weight matrix
            path: Output file path (png, jpg, etc.)
        """
        img = self.convert(weights)
        img_uint8 = (img * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_uint8)
        pil_img.save(path)


def load_som_weights(weights_path: str) -> np.ndarray:
    """
    Load SOM weights from file.

    Supports:
    - .npy files (numpy array)
    - .npz files (numpy compressed)
    - .json files (serialized weights)

    Args:
        weights_path: Path to weights file

    Returns:
        Weight matrix of shape (rows, cols, n_features)
    """
    if weights_path.endswith('.npy'):
        return np.load(weights_path)
    elif weights_path.endswith('.npz'):
        data = np.load(weights_path)
        # Assume weights are stored under 'weights' key
        return data['weights'] if 'weights' in data else data[data.files[0]]
    elif weights_path.endswith('.json'):
        import json
        with open(weights_path, 'r') as f:
            data = json.load(f)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported file format: {weights_path}")


# Convenience functions for common use cases

def som_to_image_native(weights: np.ndarray, method: str = 'rgb') -> np.ndarray:
    """
    Convert SOM to image at native resolution (1 neuron = 1 pixel).

    Args:
        weights: SOM weight matrix (rows, cols, features)
        method: 'rgb', 'umatrix', or 'combined'

    Returns:
        RGB image (rows, cols, 3)
    """
    converter = SOMToImageConverter(target_size=None, method=method)
    return converter.convert(weights)


def som_to_image_fixed(weights: np.ndarray, size: int = 32, method: str = 'rgb') -> np.ndarray:
    """
    Convert SOM to fixed-size image.

    Args:
        weights: SOM weight matrix (rows, cols, features)
        size: Target size (size x size)
        method: 'rgb', 'umatrix', or 'combined'

    Returns:
        RGB image (size, size, 3)
    """
    converter = SOMToImageConverter(target_size=(size, size), method=method)
    return converter.convert(weights)


if __name__ == '__main__':
    # Test the converter
    print("Testing SOM to Image Converter")
    print("=" * 50)

    # Create test weights
    np.random.seed(42)

    for size in [(5, 5), (10, 10), (20, 20)]:
        weights = np.random.rand(size[0], size[1], 10)

        # Native size
        img_native = som_to_image_native(weights)
        print(f"\n{size[0]}x{size[1]} map -> native: {img_native.shape}")

        # Fixed size
        img_fixed = som_to_image_fixed(weights, size=32)
        print(f"{size[0]}x{size[1]} map -> fixed 32x32: {img_fixed.shape}")

        # U-matrix
        img_umatrix = som_to_image_native(weights, method='umatrix')
        print(f"{size[0]}x{size[1]} map -> U-matrix: {img_umatrix.shape}")

    print("\n" + "=" * 50)
    print("Converter ready for use in training and inference.")
