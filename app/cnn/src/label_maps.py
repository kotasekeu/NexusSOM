#!/usr/bin/env python3
"""
Interactive U-Matrix Map Labeling Tool

This tool allows human experts to label U-Matrix visualizations as good/bad
for training the CNN quality predictor (The Eye).

Usage:
    python3 label_maps.py --results_dir /path/to/results/YYYYMMDD_HHMMSS

Features:
- Shows U-Matrix map (not RGB)
- Displays key SOM metrics (MQE, topographic error, dead neurons)
- Keyboard shortcuts: G (good), B (bad), S (save figure), Q (quit)
- Saves labels to labels.csv
- Supports resume (skips already labeled maps)
- Progress tracking
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import csv

class MapLabeler:
    def __init__(self, results_dir: str, auto_label_bad_threshold: float = 0.5):
        self.results_dir = results_dir
        self.maps_dir = os.path.join(results_dir, "maps_dataset")
        self.labels_file = os.path.join(results_dir, "labels.csv")
        self.results_csv = os.path.join(results_dir, "results.csv")
        self.auto_label_bad_threshold = auto_label_bad_threshold

        # Load results for metrics
        if not os.path.exists(self.results_csv):
            raise FileNotFoundError(f"Results file not found: {self.results_csv}")

        self.results_df = pd.read_csv(self.results_csv)
        print(f"Loaded {len(self.results_df)} configurations from results.csv")

        # Load existing labels if any
        self.labels = self._load_existing_labels()

        # Auto-label maps with excessive dead neurons (>50% by default)
        auto_labeled_count = self._auto_label_bad_maps()

        # Find all U-Matrix maps
        self.u_matrix_files = self._find_u_matrix_maps()

        # Filter out already labeled
        self.unlabeled_files = [f for f in self.u_matrix_files
                                if self._get_uid_from_filename(f) not in self.labels]

        print(f"Found {len(self.u_matrix_files)} total U-Matrix maps")
        print(f"Already labeled: {len(self.labels)} (including {auto_labeled_count} auto-labeled as 'bad')")
        print(f"Remaining to label: {len(self.unlabeled_files)}")

        self.current_index = 0
        self.current_label = None
        self.current_fig = None
        self.current_uid = None

    def _load_existing_labels(self):
        """Load existing labels from CSV"""
        labels = {}
        if os.path.exists(self.labels_file):
            try:
                with open(self.labels_file, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        labels[row['uid']] = row['label']
            except Exception as e:
                print(f"Warning: Could not load existing labels: {e}")
        return labels

    def _auto_label_bad_maps(self) -> int:
        """Automatically label maps with excessive dead neurons as 'bad'"""
        auto_labeled = 0

        for _, row in self.results_df.iterrows():
            uid = row['uid']

            # Skip if already labeled
            if uid in self.labels:
                continue

            # Check dead neuron ratio
            dead_ratio = row.get('dead_neuron_ratio', 0)

            if isinstance(dead_ratio, (int, float)) and dead_ratio > self.auto_label_bad_threshold:
                # Auto-label as bad
                self._save_label(uid, 'bad_auto')
                auto_labeled += 1

        if auto_labeled > 0:
            print(f"\n✓ Auto-labeled {auto_labeled} maps as 'bad' (dead_neuron_ratio > {self.auto_label_bad_threshold:.0%})")

        return auto_labeled

    def _find_u_matrix_maps(self):
        """Find all U-Matrix PNG files"""
        u_matrix_files = []
        if os.path.exists(self.maps_dir):
            for f in os.listdir(self.maps_dir):
                if f.endswith('_u_matrix.png'):
                    u_matrix_files.append(f)

        # Sort by UID for consistency
        u_matrix_files.sort()
        return u_matrix_files

    def _get_uid_from_filename(self, filename: str) -> str:
        """Extract UID from filename"""
        return filename.replace('_u_matrix.png', '')

    def _get_metrics_for_uid(self, uid: str) -> dict:
        """Get SOM metrics for a given UID"""
        row = self.results_df[self.results_df['uid'] == uid]
        if len(row) == 0:
            return None

        row = row.iloc[0]

        # Extract U-Matrix statistics from results.csv
        u_matrix_stats = None
        if 'u_matrix_mean' in row and 'u_matrix_std' in row:
            mean_val = row.get('u_matrix_mean')
            std_val = row.get('u_matrix_std')

            if isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)):
                # Estimate min/max from mean ± 2*std (covers ~95% of values)
                estimated_min = max(0, mean_val - 2 * std_val)
                estimated_max = mean_val + 2 * std_val

                u_matrix_stats = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': estimated_min,
                    'max': estimated_max
                }

        return {
            'mqe': row.get('best_mqe', 'N/A'),
            'topographic_error': row.get('topographic_error', 'N/A'),
            'dead_neuron_ratio': row.get('dead_neuron_ratio', 'N/A'),
            'dead_neuron_count': row.get('dead_neuron_count', 'N/A'),
            'map_size': f"{row.get('map_size', 'N/A')}",
            'duration': row.get('duration', 'N/A'),
            'u_matrix_stats': u_matrix_stats
        }

    def _save_label(self, uid: str, label: str):
        """Save a single label to CSV"""
        # Update in-memory labels
        self.labels[uid] = label

        # Write to CSV (append mode)
        file_exists = os.path.exists(self.labels_file)

        with open(self.labels_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                # Write header
                writer.writerow(['uid', 'label', 'timestamp'])

            # Write label with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([uid, label, timestamp])

        print(f"✓ Saved label: {uid} -> {label}")

    def _on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'g':
            self.current_label = 'good'
            plt.close()
        elif event.key == 'b':
            self.current_label = 'bad'
            plt.close()
        elif event.key == 's':
            # Save figure
            if self.current_fig and self.current_uid:
                save_path = os.path.join(self.results_dir, f"{self.current_uid[:16]}_labeled.png")
                self.current_fig.savefig(save_path, dpi=150, bbox_inches='tight')
                print(f"\n✓ Saved figure to: {save_path}")
            # Don't close - continue showing
        elif event.key == 'q':
            self.current_label = 'quit'
            plt.close()

    def label_map(self, filename: str) -> str:
        """Show map and get label from user"""
        uid = self._get_uid_from_filename(filename)
        self.current_uid = uid

        # Build paths to all three maps
        u_matrix_path = os.path.join(self.maps_dir, f"{uid}_u_matrix.png")
        distance_path = os.path.join(self.maps_dir, f"{uid}_distance_map.png")
        dead_neurons_path = os.path.join(self.maps_dir, f"{uid}_dead_neurons_map.png")

        if not os.path.exists(u_matrix_path):
            print(f"Warning: U-Matrix file not found: {u_matrix_path}")
            return 'skip'

        # Get metrics
        metrics = self._get_metrics_for_uid(uid)

        # Create figure with 3 subplots (one for each map type)
        fig = plt.figure(figsize=(18, 7))
        self.current_fig = fig

        # Load and display all three maps
        try:
            # U-Matrix (Red channel in RGB)
            ax1 = fig.add_subplot(1, 3, 1)
            img_u = mpimg.imread(u_matrix_path)
            ax1.imshow(img_u)

            # Title - focus on visual interpretation, not absolute values
            ax1.set_title('U-Matrix (Red channel in RGB)\nDark = similar | Yellow = boundaries',
                         fontsize=11, weight='bold', color='red')
            ax1.axis('off')

            # Distance Map (Green channel in RGB)
            if os.path.exists(distance_path):
                ax2 = fig.add_subplot(1, 3, 2)
                img_dist = mpimg.imread(distance_path)
                ax2.imshow(img_dist)
                ax2.set_title('Distance Map\n(Green channel in RGB)', fontsize=12, weight='bold', color='green')
                ax2.axis('off')

            # Dead Neurons Map (Blue channel in RGB)
            if os.path.exists(dead_neurons_path):
                ax3 = fig.add_subplot(1, 3, 3)
                img_dead = mpimg.imread(dead_neurons_path)
                ax3.imshow(img_dead)
                ax3.set_title('Dead Neurons Map\n(Blue channel in RGB)', fontsize=12, weight='bold', color='blue')
                ax3.axis('off')

            # Add metrics text at the top
            if metrics:
                # Line 1: UID
                metrics_line1 = f"UID: {uid[:32]}"

                # Line 2: Quality metrics
                metrics_line2 = ""
                if isinstance(metrics['mqe'], float):
                    metrics_line2 += f"MQE: {metrics['mqe']:.6f}"
                else:
                    metrics_line2 += f"MQE: {metrics['mqe']}"

                if isinstance(metrics['topographic_error'], float):
                    metrics_line2 += f"  |  Topo Error: {metrics['topographic_error']:.4f}"
                else:
                    metrics_line2 += f"  |  Topo Error: {metrics['topographic_error']}"

                if isinstance(metrics['dead_neuron_ratio'], float):
                    metrics_line2 += f"  |  Dead: {metrics['dead_neuron_ratio']:.2%}"
                else:
                    metrics_line2 += f"  |  Dead: {metrics['dead_neuron_ratio']}"

                metrics_line2 += f"  |  Map Size: {metrics['map_size']}"

                if isinstance(metrics['duration'], float):
                    metrics_line2 += f"  |  Duration: {metrics['duration']:.2f}s"

                # Line 3: U-Matrix color scale (CRITICAL for interpretation!)
                metrics_line3 = ""
                if metrics.get('u_matrix_stats'):
                    stats = metrics['u_matrix_stats']
                    metrics_line3 = f"U-Matrix Scale: min={stats['min']:.6f}, max={stats['max']:.6f}, mean={stats['mean']:.6f} (±{stats['std']:.6f})"

                # Combine lines
                metrics_text = f"{metrics_line1}\n{metrics_line2}"
                if metrics_line3:
                    metrics_text += f"\n{metrics_line3}"

                # Add metrics text box at top
                fig.text(0.5, 0.95, metrics_text, ha='center', fontsize=9,
                        family='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=0.7))

            # Add instruction text at bottom
            fig.text(0.5, 0.02,
                    'Press: [G] Good | [B] Bad | [S] Save Figure | [Q] Quit',
                    ha='center', fontsize=12, weight='bold',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9, pad=0.6))

            # Add progress text at very top
            progress_text = f"Progress: {len(self.labels)}/{len(self.u_matrix_files)} labeled  " \
                          f"(Remaining: {len(self.unlabeled_files) - self.current_index})"
            fig.text(0.5, 0.99, progress_text, ha='center', fontsize=10, weight='bold')

            # Connect keyboard event
            self.current_label = None
            fig.canvas.mpl_connect('key_press_event', self._on_key)

            plt.tight_layout(rect=[0, 0.04, 1, 0.94])  # Leave space for top/bottom text
            plt.show()

            return self.current_label or 'skip'

        except Exception as e:
            print(f"Error loading maps for {uid}: {e}")
            import traceback
            traceback.print_exc()
            return 'skip'

    def run(self):
        """Run interactive labeling session"""
        if len(self.unlabeled_files) == 0:
            print("✓ All maps have been labeled!")
            return

        print("\n" + "="*70)
        print("Interactive U-Matrix Labeling Tool")
        print("="*70)
        print("\nPreprocessing:")
        print(f"  ✓ Maps with dead_neuron_ratio > {self.auto_label_bad_threshold:.0%} are auto-labeled as 'bad'")
        print("\nInstructions:")
        print("  - Review each U-Matrix visualization")
        print("  - Press 'G' for Good quality maps")
        print("  - Press 'B' for Bad quality maps")
        print("  - Press 'S' to Save figure (keeps window open)")
        print("  - Press 'Q' to Quit (saves progress)")
        print("\nCriteria for 'Good' maps:")
        print("  ✓ Clear cluster boundaries")
        print("  ✓ Low MQE (good quantization)")
        print("  ✓ Low topographic error (preserves topology)")
        print("  ✓ Few dead neurons (<50%)")
        print("  ✓ Distinct patterns visible")
        print("\n" + "="*70 + "\n")

        try:
            for i, filename in enumerate(self.unlabeled_files):
                self.current_index = i
                uid = self._get_uid_from_filename(filename)

                print(f"\n[{i+1}/{len(self.unlabeled_files)}] Showing: {uid[:32]}...")

                label = self.label_map(filename)

                if label == 'quit':
                    print("\n✓ Quitting and saving progress...")
                    break
                elif label == 'skip':
                    print("  → Skipped")
                    continue
                elif label in ['good', 'bad']:
                    self._save_label(uid, label)
                else:
                    print(f"  → Unknown label: {label}, skipping")

            # Print summary
            print("\n" + "="*70)
            print("Labeling Session Summary")
            print("="*70)

            good_count = sum(1 for l in self.labels.values() if l == 'good')
            bad_manual_count = sum(1 for l in self.labels.values() if l == 'bad')
            bad_auto_count = sum(1 for l in self.labels.values() if l == 'bad_auto')
            total_bad = bad_manual_count + bad_auto_count

            print(f"Total labeled: {len(self.labels)}/{len(self.u_matrix_files)}")
            print(f"  - Good: {good_count}")
            print(f"  - Bad: {total_bad}")
            print(f"    • Manual: {bad_manual_count}")
            print(f"    • Auto (dead_neuron_ratio > {self.auto_label_bad_threshold:.0%}): {bad_auto_count}")
            print(f"\nLabels saved to: {self.labels_file}")
            print("="*70 + "\n")

        except KeyboardInterrupt:
            print("\n\n✓ Interrupted by user. Progress saved.")
        except Exception as e:
            print(f"\n✗ Error during labeling: {e}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description='Interactive U-Matrix Map Labeling Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Label maps from a specific EA run (auto-labels maps with >50% dead neurons as bad)
  python3 label_maps.py --results_dir ./test/results/20260112_074649

  # Use custom threshold for auto-labeling (e.g., 70%)
  python3 label_maps.py --results_dir ./test/results/20260112_074649 --auto_bad_threshold 0.7

  # Resume labeling (automatically skips already labeled maps)
  python3 label_maps.py --results_dir ./test/results/20260112_074649
        """
    )

    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Path to EA results directory (e.g., test/results/20260112_074649)'
    )

    parser.add_argument(
        '--auto_bad_threshold',
        type=float,
        default=0.5,
        help='Auto-label maps with dead_neuron_ratio above this threshold as bad (default: 0.5 = 50%%)'
    )

    args = parser.parse_args()

    # Validate results directory
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)

    # Run labeler
    labeler = MapLabeler(args.results_dir, auto_label_bad_threshold=args.auto_bad_threshold)
    labeler.run()


if __name__ == "__main__":
    main()
