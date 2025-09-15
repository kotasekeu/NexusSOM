# som.py
import sys
import numpy as np
import time
import math
from datetime import datetime
from tqdm import tqdm

class KohonenSOM:
    def __init__(self, **kwargs):

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
        self.m = kwargs.get('m', 10)
        self.n = kwargs.get('n', 10)
        self.dim = kwargs.get('dim', 3)

        self.processing_type = kwargs.get('processing_type', 'hybrid')  # 'stochastic', 'deterministic', 'hybrid'
        self.total_weight_updates = 0

        self.mqe_history = []
        self.best_mqe = float('inf')

        self.set_radius()
        self.weights = np.random.rand(self.m, self.n, self.dim)

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        self.neuron_coords = np.indices((self.m, self.n)).transpose(1, 2, 0)

        if self.map_type == 'hex':
            q_coords = self.neuron_coords[:, :, 1] - np.floor(self.neuron_coords[:, :, 0] / 2)
            r_coords = self.neuron_coords[:, :, 0]
            x = q_coords
            z = r_coords
            y = -x - z
            self.cube_coords = np.stack([x, y, z], axis=-1)

    def normalize_weights(self) -> None:
        norms = np.linalg.norm(self.weights, axis=2, keepdims=True)
        norms[norms == 0] = 1

        self.weights /= norms

    def get_decay_value(self, t: int, N: int, start: float, end: float, decay_type: str) -> float:
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
        return self.get_decay_value(t, N, self.start_batch_percent, self.end_batch_percent, self.batch_growth_type)

    def find_bmu(self, sample: np.ndarray) -> tuple[int, int]:
        flat = self.weights.reshape(-1, self.dim)
        diffs = flat - sample
        dists = np.linalg.norm(diffs, axis=1)
        idx = np.argmin(dists)
        return divmod(idx, self.n)

    def update_weights(self, sample: np.ndarray, bmu_idx: tuple[int, int], current_learning_rate: float,
                       radius: float) -> None:
        bmu_i, bmu_j = bmu_idx

        if self.map_type == 'square':
            distances = np.linalg.norm(self.neuron_coords - np.array([bmu_i, bmu_j]), axis=2)
        else:  # hex
            bmu_cube_coords = self.cube_coords[bmu_i, bmu_j]
            distances = np.sum(np.abs(self.cube_coords - bmu_cube_coords), axis=2) / 2

        radius_squared = radius ** 2
        influence = np.exp(-distances ** 2 / (2 * radius_squared + 1e-8))

        self.weights += influence[:, :, np.newaxis] * current_learning_rate * (sample - self.weights)

    def set_radius(self) -> None:
        ratio = self.start_radius_init_ratio
        if ratio is None:
            ratio = 1.0
        self.start_radius = ratio * max(self.m, self.n)

    def grid_distance(self, a: tuple[int, int], b: tuple[int, int]) -> float:
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

    def compute_quantization_error(self, data: np.ndarray) -> tuple[np.ndarray | None, float]:
        num_neurons = self.m * self.n
        flat_weights = self.weights.reshape(num_neurons, self.dim)

        dists = np.linalg.norm(data[:, np.newaxis, :] - flat_weights[np.newaxis, :, :], axis=2)
        bmu_indexes = np.argmin(dists, axis=1)

        winning_weights = flat_weights[bmu_indexes]
        errors_per_sample = np.linalg.norm(data - winning_weights, axis=1)
        total_qe = errors_per_sample.mean()

        sum_errors = np.bincount(bmu_indexes, weights=errors_per_sample, minlength=num_neurons)
        counts = np.bincount(bmu_indexes, minlength=num_neurons)

        neuron_errors = np.divide(sum_errors, counts, out=np.zeros_like(sum_errors), where=counts != 0)
        neuron_error_map = neuron_errors.reshape(self.m, self.n)

        return neuron_error_map, total_qe

    def train(self, data: np.ndarray) -> dict:
        start_time = time.monotonic()

        self.mqe_history = []
        self.best_mqe = float('inf')
        epochs_without_improvement = 0
        converged = False
        self.total_weight_updates = 0

        total_samples = data.shape[0]
        total_iterations = int(total_samples * self.epoch_multiplier)

        if self.processing_type == 'hybrid':
            shuffled_indices = np.random.permutation(total_samples)
            section_indices = np.array_split(shuffled_indices, self.num_batches)

        pbar = tqdm(range(total_iterations), desc="Trénink SOM", unit="iter")
        for iteration in pbar:
            current_lr = self.get_decay_value(iteration, total_iterations, self.start_learning_rate,
                                              self.end_learning_rate, self.lr_decay_type)
            current_radius = self.get_decay_value(iteration, total_iterations, self.start_radius, self.end_radius,
                                                  self.radius_decay_type)

            samples_to_process = []

            if self.processing_type == 'stochastic':
                idx = np.random.randint(0, total_samples)
                samples_to_process = data[idx:idx + 1]

            elif self.processing_type == 'deterministic':
                samples_to_process = data

            elif self.processing_type == 'hybrid':
                batch_percent = self.get_batch_percent(iteration, total_iterations)
                samples_per_section_float = (total_samples * batch_percent / 100.0) / self.num_batches
                samples_per_section = max(1, math.ceil(samples_per_section_float))

                selected_indices = []
                for section in section_indices:
                    num_to_take = min(samples_per_section, len(section))
                    chosen = np.random.choice(section, num_to_take, replace=False)
                    selected_indices.extend(chosen)

                samples_to_process = data[selected_indices]

            if len(samples_to_process) == 0:
                continue

            for sample in samples_to_process:
                bmu_idx = self.find_bmu(sample)
                self.update_weights(sample, bmu_idx, current_lr, current_radius)

            self.total_weight_updates += len(samples_to_process)

            if self.normalize_weights_flag:
                self.normalize_weights()

            _, current_mqe = self.compute_quantization_error(data)
            self.mqe_history.append(current_mqe)

            if current_mqe < self.best_mqe:
                self.best_mqe = current_mqe
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            pbar.set_postfix(best_mqe=f"{self.best_mqe:.6f}", epochs_no_imp=epochs_without_improvement)

            if epochs_without_improvement >= self.max_epochs_without_improvement:
                print(f"\nUkončuji trénink: Žádné zlepšení po {self.max_epochs_without_improvement} iteracích.")
                converged = True
                break

        pbar.close()

        duration = time.monotonic() - start_time

        return {
            'best_mqe': self.best_mqe,
            'duration': duration,
            'total_weight_updates': self.total_weight_updates,
            "mqe_history": self.mqe_history,
            "epochs_ran": iteration + 1,
            "converged": converged
        }
