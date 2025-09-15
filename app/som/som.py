# som.py
import sys
import numpy as np

class KohonenSOM:
    def __init__(self, **kwargs):

        self.start_learning_rate = kwargs.get('start_learning_rate')
        self.end_learning_rate = kwargs.get('end_learning_rate')
        self.lr_decay_type = kwargs.get('lr_decay_type')

        self.start_radius_init_ratio = kwargs.get('start_radius_init_ratio')
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

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

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

    def set_radius(self, radius: float | None) -> None:
        if radius is None:
            self.start_radius = max(self.m, self.n) / 2
        elif radius < self.end_radius:
            self.start_radius = self.end_radius
        else:
            self.start_radius = radius

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
        bmu_indexes = np.argmin(dists, axis=1)  # Plochý index BMU pro každý vzorek

        winning_weights = flat_weights[bmu_indexes]
        errors_per_sample = np.linalg.norm(data - winning_weights, axis=1)
        total_qe = errors_per_sample.mean()

        sum_errors = np.bincount(bmu_indexes, weights=errors_per_sample, minlength=num_neurons)
        counts = np.bincount(bmu_indexes, minlength=num_neurons)

        neuron_errors = np.divide(sum_errors, counts, out=np.zeros_like(sum_errors), where=counts != 0)
        neuron_error_map = neuron_errors.reshape(self.m, self.n)

        return neuron_error_map, total_qe

    def train(self, data: np.ndarray) -> None:
        start_time = time.monotonic()
        self.mqe_history = []
        self.best_mqe = float('inf')
        epochs_without_improvement = 0
        converged = False
        start = datetime.now()
        log_message(f"Začátek trénování: {start.strftime('%Y-%m-%d %H:%M:%S')}")
        total_samples = data.shape[0]
        total_epochs = int(total_samples * self.epoch_multiplier)
        no_improvement_count = 0
        self.epochs_run = 0
        self.mqe_history = []
        self.epochs_history = []
        pbar = tqdm(total=total_epochs, desc=f"Epochy (začátek: {start.strftime('%H:%M:%S')})",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        for epoch in range(total_epochs):
            samples_per_batch = 1;
            current_lr = self.get_decay_value(epoch, total_epochs, self.start_learning_rate, self.end_learning_rate, self.lr_decay_type)
            current_radius = self.get_decay_value(epoch, total_epochs, self.start_radius, self.end_radius, self.radius_decay_type)
            if self.processing_type == 'hybrid':
                batch_percent = self.get_batch_percent(epoch, total_epochs)
                samples_per_batch = math.ceil(total_samples * batch_percent / 100)
            if self.processing_type == 'stochastic':
                if self.base_seed is not None:
                    np.random.seed(self.base_seed + epoch)
                idx = np.random.randint(0, total_samples)
                sample = data[idx]
                bmu_idx = self.find_bmu(sample)
                self.update_weights(sample, bmu_idx, current_lr, current_radius)
            elif self.processing_type == 'deterministic':
                for sample in data:
                    bmu_idx = self.find_bmu(sample)
                    self.update_weights(sample, bmu_idx, current_lr, current_radius)
            else:
                if self.base_seed is not None:
                    np.random.seed(self.base_seed + epoch)
                indices = np.random.permutation(total_samples)
                for batch_idx in range(self.num_batches):
                    start_idx = batch_idx * (total_samples // self.num_batches)
                    end_idx = min((batch_idx + 1) * (total_samples // self.num_batches), total_samples)
                    batch_indices = indices[start_idx:end_idx]
                    if samples_per_batch < len(batch_indices):
                        batch_indices = batch_indices[:samples_per_batch]
                    for idx in batch_indices:
                        sample = data[idx]
                        bmu_idx = self.find_bmu(sample)
                        self.update_weights(sample, bmu_idx, current_lr, current_radius)
            if self.processing_type == 'stochastic':
                self.total_weight_updates += 1
            else:
                self.total_weight_updates += total_samples
            if self.normalize_weights_flag:
                self.normalize_weights()
            should_compute_mqe = False
            total_qe = None
            if self.processing_type == 'deterministic':
                should_compute_mqe = True
            else:
                interval = max(1, total_epochs // 500)
                if epoch % interval == 0:
                    should_compute_mqe = True
            if should_compute_mqe:
                codebook_vectors = self.weights.reshape(-1, self.dim)
                bmu_indexes = np.array([self.find_bmu(x)[0] * self.n + self.find_bmu(x)[1] for x in data])
                _, total_qe = self.compute_quantization_error(data, codebook_vectors, bmu_indexes, (self.m, self.n), compute_neuron_map=False)
                self.mqe_history.append(total_qe)
                self.epochs_history.append(epoch)
                self.learning_rate_history.append(current_lr)
                self.radius_history.append(current_radius)
                if self.processing_type == 'hybrid':
                    self.batch_size_history.append(samples_per_batch)
                if self.min_q_error is not None and total_qe <= self.min_q_error:
                    log_message(f"Dosažena limitní MQE {self.min_q_error}. Ukončuji trénování.")
                    break
                if total_qe < self.best_mqe:
                    self.best_mqe = total_qe
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if self.max_epochs_without_improvement is not None and no_improvement_count >= self.max_epochs_without_improvement:
                        log_message(f"Žádné zlepšení po {self.max_epochs_without_improvement} epochách. Ukončuji trénování.")
                        break
            current_mqe = np.random.rand()
            self.mqe_history.append(current_mqe)
            if current_mqe < self.best_mqe:
                self.best_mqe = current_mqe
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if self.max_epochs_without_improvement is not None and \
                    epochs_without_improvement >= self.max_epochs_without_improvement:
                converged = True
                break
            if epoch % 100 == 0:
                elapsed_time = time.monotonic() - start_time
                if total_qe is not None:
                    log_message(f"{epoch}|{total_samples}|{samples_per_batch}|{current_radius:.4f}|{current_lr:.6f}|{total_qe:.6f} (čas: {str(elapsed_time).split('.')[0]})")
                else:
                    log_message(f"{epoch}|{total_samples}|{samples_per_batch}|{current_radius:.4f}|{current_lr:.6f}|N/A (čas: {str(elapsed_time).split('.')[0]})")
            pbar.update(1)
            self.epochs_run = epoch + 1
        pbar.close()
        training_duration = time.monotonic() - start_time
        return {
            "final_mqe": self.best_mqe,
            "mqe_history": self.mqe_history,
            "total_weight_updates": self.total_weight_updates,
            "training_duration": training_duration,
            "epochs_ran": epoch + 1,
            "converged": converged
        }

    @property
    def best_mqe(self):
        return 0.05