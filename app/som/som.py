# som.py
import sys
import numpy as np
import time
import math
from datetime import datetime
from tqdm import tqdm
import os
from som.utils import log_message
from collections import deque

class KohonenSOM:
    def __init__(self, **kwargs):
        # Initialize SOM hyperparameters from kwargs
        self.start_learning_rate = kwargs.get('start_learning_rate')
        self.end_learning_rate = kwargs.get('end_learning_rate')
        self.lr_decay_type = kwargs.get('lr_decay_type')

        self.start_radius_init_ratio = kwargs.get('start_radius_init_ratio')
        self.start_radius = 1
        self.end_radius = kwargs.get('end_radius')
        self.radius_decay_type = kwargs.get('radius_decay_type')

        self.start_batch_percent = kwargs.get('start_batch_percent')
        self.end_batch_percent = kwargs.get('end_batch_percent')
        self.batch_growth_type = kwargs.get('batch_growth_type')

        self.epoch_multiplier = kwargs.get('epoch_multiplier')
        self.normalize_weights_flag = kwargs.get('normalize_weights_flag')
        self.growth_g = kwargs.get('growth_g')
        self.random_seed = kwargs.get('random_seed')
        self.map_type = kwargs.get('map_type')
        self.num_batches = kwargs.get('num_batches')
        self.max_epochs_without_improvement = kwargs.get('max_epochs_without_improvement')

        # Determine SOM map size
        if 'map_size' in kwargs and isinstance(kwargs['map_size'], list):
            map_width, map_height = kwargs['map_size'][0] if isinstance(kwargs['map_size'][0], list) else \
                kwargs['map_size']
        else:
            map_width, map_height = (10, 10)  # TODO: Replace with automatic calculation based on data size

        self.m = map_width
        self.n = map_height

        self.dim = kwargs.get('dim', 3)

        self.processing_type = kwargs.get('processing_type', 'hybrid')  # 'stochastic', 'deterministic', 'hybrid'
        self.total_weight_updates = 0

        # History tracking for training metrics
        self.history = {
            'mqe': [],
            'learning_rate': [],
            'radius': [],
            'batch_size': []
        }
        self.best_mqe = float('inf')

        self.set_radius()
        self.weights = np.random.rand(self.m, self.n, self.dim)

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Precompute neuron coordinates for distance calculations
        self.neuron_coords = np.indices((self.m, self.n)).transpose(1, 2, 0)

        self.early_stopping_window = kwargs.get('early_stopping_window', 50000)  # FIXME For now is this feature disabled
        self.early_stopping_patience = kwargs.get('max_epochs_without_improvement', 50000) # FIXME For now is this feature disabled
        self.mqe_evaluations_per_run = kwargs.get('mqe_evaluations_per_run', 20)

        if self.map_type == 'hex':
            # Calculate cube coordinates for hexagonal grid
            q_coords = self.neuron_coords[:, :, 1] - np.floor(self.neuron_coords[:, :, 0] / 2)
            r_coords = self.neuron_coords[:, :, 0]
            x = q_coords
            z = r_coords
            y = -x - z
            self.cube_coords = np.stack([x, y, z], axis=-1)

    def _get_neighbors(self, i: int, j: int) -> list[tuple[int, int]]:
        neighbors = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < self.m and 0 <= nj < self.n:
                    neighbors.append((ni, nj))
        return neighbors

    def calculate_dead_neurons(self, data: np.ndarray) -> tuple[int, float]:
        """
        Calculates the number and percentage of dead neurons (neurons that never won).
        """
        num_neurons = self.m * self.n
        if data.shape[0] == 0:
            return num_neurons, 1.0  # If no data, all neurons are dead

        flat_weights = self.weights.reshape(num_neurons, self.dim)

        dists = np.linalg.norm(data[:, np.newaxis, :] - flat_weights[np.newaxis, :, :], axis=2)
        bmu_indices = np.argmin(dists, axis=1)

        hit_counts = np.bincount(bmu_indices, minlength=num_neurons)

        dead_neuron_count = np.count_nonzero(hit_counts == 0)
        dead_neuron_ratio = dead_neuron_count / num_neurons

        return dead_neuron_count, dead_neuron_ratio

    def calculate_topographic_error(self, data: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Calculates the topographic error.
        An error occurs if the BMU and the second-best neuron are not neighbors.
        """
        error_count = 0
        flat_weights = self.weights.reshape(-1, self.dim)

        for i in range(data.shape[0]):
            sample = data[i]
            sample_mask = mask[i] if mask is not None else None

            diffs = flat_weights - sample
            if sample_mask is not None:
                diffs *= ~sample_mask

            dists = np.linalg.norm(diffs, axis=1)

            best_two_indices = np.argsort(dists)[:2]

            bmu1_i, bmu1_j = divmod(best_two_indices[0], self.n)
            bmu2_i, bmu2_j = divmod(best_two_indices[1], self.n)

            if (bmu2_i, bmu2_j) not in self._get_neighbors(bmu1_i, bmu1_j):
                error_count += 1

        return error_count / data.shape[0]

    def calculate_u_matrix_metrics(self) -> dict:
        m, n, weights = self.m, self.n, self.weights
        u_matrix = np.zeros((m, n))

        diffs_v = np.linalg.norm(weights[1:, :, :] - weights[:-1, :, :], axis=2)
        u_matrix[1:, :] += diffs_v
        u_matrix[:-1, :] += diffs_v
        diffs_h = np.linalg.norm(weights[:, 1:, :] - weights[:, :-1, :], axis=2)
        u_matrix[:, 1:] += diffs_h
        u_matrix[:, :-1] += diffs_h

        counts = np.full((m, n), 4.0)
        counts[[0, -1], :] -= 1
        counts[:, [0, -1]] -= 1
        u_matrix /= counts

        return {
            'u_matrix_mean': np.mean(u_matrix),
            'u_matrix_std': np.std(u_matrix)
        }

    def normalize_weights(self) -> None:
        # Normalize weights for each neuron to unit norm
        norms = np.linalg.norm(self.weights, axis=2, keepdims=True)
        norms[norms == 0] = 1
        self.weights /= norms

    def get_decay_value(self, t: int, N: int, start: float, end: float, decay_type: str) -> float:
        # Compute decayed value for learning rate, radius, or batch percent
        if N <= 1:
            return start
        if decay_type == 'static':
            return start
        elif decay_type == 'linear-drop':
            return start - (t / (N - 1)) * (start - end)
        elif decay_type == 'linear-growth':
            return start + (t / (N - 1)) * (end - start)
        elif decay_type == 'exp-drop':
            norm = (1 - np.exp(-self.growth_g * t / N)) / (1 - np.exp(-self.growth_g))
            return start - norm * (start - end)
        elif decay_type == 'exp-growth':
            return start + (end - start) * (np.exp(self.growth_g * t / N) - 1) / (np.exp(self.growth_g) - 1)
        elif decay_type == 'log-drop':
            norm = np.log(self.growth_g * t + 1) / np.log(self.growth_g * N + 1)
            return start - norm * (start - end)
        elif decay_type == 'log-growth':
            return start + (end - start) * (np.log(self.growth_g * t + 1) / np.log(self.growth_g * N + 1))
        elif decay_type == 'step-down':
            step_count = 10
            step_size = N // step_count
            current_step = min(t // step_size, step_count - 1)
            factor = 0.7 ** current_step
            return max(end, start * factor)
        else:
            raise ValueError(f"Unknown decay_type: {decay_type}")

    def get_batch_percent(self, t: int, N: int) -> float:
        # Get current batch percent for hybrid processing
        return self.get_decay_value(t, N, self.start_batch_percent, self.end_batch_percent, self.batch_growth_type)

    def find_bmu(self, sample: np.ndarray, mask: np.ndarray = None) -> tuple[int, int]:
        # Find Best Matching Unit (BMU) for a given sample
        flat_weights = self.weights.reshape(-1, self.dim)
        diffs = flat_weights - sample

        if mask is not None:
            valid_dims_mask = ~mask
            diffs *= valid_dims_mask

        dists = np.linalg.norm(diffs, axis=1)
        bmu_flat_idx = np.argmin(dists)
        return divmod(bmu_flat_idx, self.n)

    def update_weights(self, sample: np.ndarray, bmu_idx: tuple[int, int],
                       current_learning_rate: float, radius: float,
                       mask: np.ndarray = None) -> None:
        # Update weights of neurons based on BMU and neighborhood function
        bmu_i, bmu_j = bmu_idx

        if self.map_type == 'square':
            distances = np.linalg.norm(self.neuron_coords - np.array([bmu_i, bmu_j]), axis=2)
        else:  # hex
            bmu_cube_coords = self.cube_coords[bmu_i, bmu_j]
            distances = np.sum(np.abs(self.cube_coords - bmu_cube_coords), axis=2) / 2

        radius_squared = radius ** 2
        influence = np.exp(-distances ** 2 / (2 * radius_squared + 1e-8))

        update_term = current_learning_rate * (sample - self.weights)

        if mask is not None:
            valid_dims_mask = ~mask
            update_term *= valid_dims_mask

        self.weights += influence[:, :, np.newaxis] * update_term

    def set_radius(self) -> None:
        # Set initial neighborhood radius based on ratio and map size
        ratio = self.start_radius_init_ratio
        if ratio is None:
            ratio = 1.0
        self.start_radius = ratio * max(self.m, self.n)

    def grid_distance(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        # Compute grid distance between two neurons (square or hex)
        i1, j1 = a
        i2, j2 = b
        if self.map_type == 'square':
            return math.hypot(i1 - i2, j1 - j2)
        q1, r1 = j1, i1
        q2, r2 = j2, i2
        x1, z1 = q1, r1
        y1 = -x1 - z1
        x2, z2 = q2, r2
        y2 = -x2 - z2
        return (abs(x1 - x2) + abs(y1 - y2) + abs(z1 - z2)) / 2

    def compute_quantization_error(self, data: np.ndarray,
                                   mask: np.ndarray = None) -> tuple[np.ndarray | None, float]:
        # Compute quantization error for all samples and per neuron
        num_neurons = self.m * self.n
        flat_weights = self.weights.reshape(num_neurons, self.dim)

        dists = np.linalg.norm(data[:, np.newaxis, :] - flat_weights[np.newaxis, :, :], axis=2)
        bmu_indexes = np.argmin(dists, axis=1)

        winning_weights = flat_weights[bmu_indexes]
        diffs = data - winning_weights

        if mask is not None:
            valid_dims_mask = ~mask
            diffs *= valid_dims_mask

        errors_per_sample = np.linalg.norm(diffs, axis=1)
        total_qe = errors_per_sample.mean()

        sum_errors = np.bincount(bmu_indexes, weights=errors_per_sample, minlength=num_neurons)
        counts = np.bincount(bmu_indexes, minlength=num_neurons)

        neuron_errors = np.divide(sum_errors, counts, out=np.zeros_like(sum_errors), where=counts != 0)
        neuron_error_map = neuron_errors.reshape(self.m, self.n)

        return neuron_error_map, total_qe

    def train(self, data: np.ndarray, ignore_mask: np.ndarray = None, working_dir: str = '.') -> dict:
        # Main training loop for SOM
        start_time = time.monotonic()

        self.mqe_history = []
        self.best_mqe = float('inf')
        converged = False
        self.total_weight_updates = 0

        total_samples = data.shape[0]
        total_iterations = int(total_samples * self.epoch_multiplier)

        best_moving_avg = float('inf')
        epochs_without_improvement = 0
        recent_mqe_history = deque(maxlen=self.early_stopping_window)

        # Prepare batch indices for hybrid processing
        if self.processing_type == 'hybrid':
            shuffled_indices = np.random.permutation(total_samples)
            section_indices = np.array_split(shuffled_indices, self.num_batches)

        if self.mqe_evaluations_per_run > 0:
            mqe_compute_interval = max(1, total_iterations // self.mqe_evaluations_per_run)
        else:
            mqe_compute_interval = total_iterations

        pbar = tqdm(range(total_iterations), desc="SOM Training", unit="iter")
        for iteration in pbar:
            current_lr = self.get_decay_value(iteration, total_iterations, self.start_learning_rate,
                                              self.end_learning_rate, self.lr_decay_type)
            current_radius = self.get_decay_value(iteration, total_iterations, self.start_radius, self.end_radius,
                                                  self.radius_decay_type)

            self.history['learning_rate'].append((iteration, current_lr))
            self.history['radius'].append((iteration, current_radius))

            # Unified logic for sample selection
            indices_to_process = []
            if self.processing_type == 'stochastic':
                idx = np.random.randint(0, total_samples)
                indices_to_process = [idx]

            elif self.processing_type == 'deterministic':
                indices_to_process = np.arange(total_samples)

            elif self.processing_type == 'hybrid':
                batch_percent = self.get_batch_percent(iteration, total_iterations)
                samples_per_section_float = (total_samples * batch_percent / 100.0)
                samples_per_section = max(1, math.ceil(samples_per_section_float))

                selected_indices = []
                for section in section_indices:
                    num_to_take = min(samples_per_section, len(section))
                    chosen = np.random.choice(section, num_to_take, replace=False)
                    selected_indices.extend(chosen)
                indices_to_process = selected_indices

            self.history['batch_size'].append((iteration, len(indices_to_process)))

            if np.size(indices_to_process) == 0:
                continue

            samples_to_process = data[indices_to_process]
            sample_masks = ignore_mask[indices_to_process] if ignore_mask is not None else None

            # Update weights for each selected sample
            for i, sample in enumerate(samples_to_process):
                sample_mask = sample_masks[i] if sample_masks is not None else None

                bmu_idx = self.find_bmu(sample, mask=sample_mask)
                self.update_weights(sample, bmu_idx, current_lr, current_radius, mask=sample_mask)

            self.total_weight_updates += len(samples_to_process)

            if self.normalize_weights_flag:
                self.normalize_weights()

                # Compute MQE and check stopping criteria
            if iteration % mqe_compute_interval == 0 or iteration == total_iterations - 1:
                _, current_mqe = self.compute_quantization_error(data, mask=ignore_mask)
                self.history['mqe'].append((iteration, current_mqe))

                if current_mqe < self.best_mqe:
                    self.best_mqe = current_mqe

                recent_mqe_history.append(current_mqe)

                if len(recent_mqe_history) == self.early_stopping_window:
                    current_moving_avg = np.mean(recent_mqe_history)

                    if current_moving_avg < best_moving_avg:
                        best_moving_avg = current_moving_avg
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                    pbar.set_postfix(
                        best_mqe=f"{self.best_mqe:.6f}",
                        moving_avg=f"{best_moving_avg:.6f}",
                        checks_no_imp=epochs_without_improvement
                    )

                    if epochs_without_improvement >= self.early_stopping_patience:
                        print(
                            f"\nStopping training: The moving average did not improve after {self.early_stopping_patience} checks.")
                        converged = True
                        break

        pbar.close()

        duration = time.monotonic() - start_time
        csv_dir = os.path.join(working_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)

        # Save final weights as binary .npy file
        npy_path = os.path.join(csv_dir, "weights.npy")
        np.save(npy_path, self.weights)
        log_message(working_dir, "SYSTEM", f"Final weights saved to '{npy_path}'")

        # Save final weights as readable .csv file for inspection
        weights_reshaped = self.weights.reshape(-1, self.dim)
        coords = np.indices((self.m, self.n)).transpose(1, 2, 0).reshape(-1, 2)
        header = ['neuron_i', 'neuron_j'] + [f'dim_{d}' for d in range(self.dim)]
        csv_data = np.hstack([coords, weights_reshaped])

        csv_path = os.path.join(csv_dir, "weights_readable.csv")
        np.savetxt(csv_path, csv_data, delimiter=',', header=','.join(header), comments='')
        log_message(working_dir, "SYSTEM", f"Readable final weights saved to '{csv_path}'")

        return {
            'best_mqe': self.best_mqe,
            'duration': duration,
            'total_weight_updates': self.total_weight_updates,
            "epochs_ran": iteration + 1,
            "converged": converged,
            "history": self.history
        }
