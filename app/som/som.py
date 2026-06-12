# som.py — core SOM algorithm.
# Pure compute over numpy arrays: no disk IO, no logging side effects.
# Persistence of results lives in som/persistence.py; orchestration in som/run.py.
import math
import time
from collections import deque

import numpy as np
from tqdm import tqdm

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
        # 'reshuffle' — default: without-replacement epoch shuffling —
        #               guaranteed equal hit counts, no quality cost,
        #               faster (docs/ea/SEARCH_SPACE.md step 1, exp A–C)
        # 'random'    — legacy: np.random.choice per iteration (with
        #               replacement across iterations, coverage ~ Poisson);
        #               pre-2026-06-12 runs used this — their configs must
        #               say so explicitly to stay replayable
        sampling_method = kwargs.get('sampling_method', 'reshuffle')
        if sampling_method == 'cycle':  # deprecated alias
            sampling_method = 'reshuffle'
        if sampling_method not in ('random', 'reshuffle'):
            raise ValueError(f"Unknown sampling_method: {sampling_method}")
        self.sampling_method = sampling_method
        self.max_epochs_without_improvement = kwargs.get('max_epochs_without_improvement')

        # Determine SOM map size
        if 'map_size' in kwargs and isinstance(kwargs['map_size'], list):
            map_width, map_height = kwargs['map_size'][0] if isinstance(kwargs['map_size'][0], list) else \
                kwargs['map_size']
        else:
            map_width, map_height = (10, 10)  # TODO: Replace with automatic calculation based on data size

        self.m = map_width
        self.n = map_height
        self.set_radius()

        self.dim = kwargs.get('dim', 3)

        self.total_weight_updates = 0

        # History tracking for training metrics
        self.history = {
            'mqe': [],
            'learning_rate': [],
            'radius': [],
            'batch_size': []
        }
        self.best_mqe = float('inf')

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        self.weights = np.random.rand(self.m, self.n, self.dim)

        # Precompute neuron coordinates for distance calculations
        self.neuron_coords = np.indices((self.m, self.n)).transpose(1, 2, 0)

        self.early_stopping_window = kwargs.get('early_stopping_window', 50000)  # FIXME For now is this feature disabled
        self.early_stopping_patience = kwargs.get('max_epochs_without_improvement', 50000) # FIXME For now is this feature disabled
        self.mqe_evaluations_per_run = kwargs.get('mqe_evaluations_per_run', 20)

        # Checkpointing for LSTM training data collection
        self.save_checkpoints = kwargs.get('save_checkpoints', False)
        self.checkpoint_count = kwargs.get('checkpoint_count', 10)
        self.track_sample_coverage = kwargs.get('track_sample_coverage', False)
        # If True, save a checkpoint at every MQE evaluation (ignores checkpoint_count)
        self.checkpoint_every_mqe = kwargs.get('checkpoint_every_mqe', False)
        # Progress bar — disable for batch contexts (EA runs hundreds of trainings)
        self.show_progress = kwargs.get('show_progress', True)

        if self.map_type == 'hex':
            # Calculate cube coordinates for hexagonal grid
            q_coords = self.neuron_coords[:, :, 1] - np.floor(self.neuron_coords[:, :, 0] / 2)
            r_coords = self.neuron_coords[:, :, 0]
            x = q_coords
            z = r_coords
            y = -x - z
            self.cube_coords = np.stack([x, y, z], axis=-1)

    def calculate_dead_neurons(self, data: np.ndarray,
                               mask: np.ndarray = None) -> tuple[int, float]:
        """
        Calculates the number and percentage of dead neurons (neurons that never won).
        Masked dimensions are excluded so BMU selection matches training.
        """
        num_neurons = self.m * self.n
        if data.shape[0] == 0:
            return num_neurons, 1.0  # If no data, all neurons are dead

        flat_weights = self.weights.reshape(num_neurons, self.dim)

        diffs = data[:, np.newaxis, :] - flat_weights[np.newaxis, :, :]
        if mask is not None:
            diffs *= (~mask)[:, np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=2)
        bmu_indices = np.argmin(dists, axis=1)

        hit_counts = np.bincount(bmu_indices, minlength=num_neurons)

        dead_neuron_count = np.count_nonzero(hit_counts == 0)
        dead_neuron_ratio = dead_neuron_count / num_neurons

        return dead_neuron_count, dead_neuron_ratio

    def calculate_topographic_error(self, data: np.ndarray, mask: np.ndarray = None) -> float:
        """
        Calculates the topographic error (vectorized).
        An error occurs if the BMU and the second-best neuron are not Moore neighbors.
        """
        flat_weights = self.weights.reshape(-1, self.dim)

        # (N, M, dim) distance matrix for all samples and neurons at once
        diffs = data[:, np.newaxis, :] - flat_weights[np.newaxis, :, :]
        if mask is not None:
            diffs *= (~mask)[:, np.newaxis, :]
        dists = np.linalg.norm(diffs, axis=2)  # (N, M)

        best_two = np.argsort(dists, axis=1)[:, :2]  # (N, 2)
        bmu1_i, bmu1_j = np.divmod(best_two[:, 0], self.n)
        bmu2_i, bmu2_j = np.divmod(best_two[:, 1], self.n)

        if self.map_type == 'hex':
            # Hex has 6 neighbors — use cube coordinate distance (== 1 iff true neighbor).
            # cube_coords are precomputed correctly in __init__ with floor(i/2) offset.
            cube_flat = self.cube_coords.reshape(-1, 3)
            bmu1_cube = cube_flat[best_two[:, 0]]  # (N, 3)
            bmu2_cube = cube_flat[best_two[:, 1]]  # (N, 3)
            not_neighbors = (np.sum(np.abs(bmu1_cube - bmu2_cube), axis=1) / 2) > 1
        else:
            # Square: Moore neighborhood — max(|Δi|, |Δj|) <= 1
            not_neighbors = (np.abs(bmu1_i - bmu2_i) > 1) | (np.abs(bmu1_j - bmu2_j) > 1)
        return not_neighbors.mean()

    def calculate_topological_correlation(self) -> float:
        """
        Spearman rank correlation between pairwise distances of weight vectors
        (data space) and pairwise distances of neurons (grid space).

        High ρ → map unfolds the data manifold: neurons far apart in the grid
                  are also far apart in weight space.
        Low ρ  → map is crumpled: grid distances and weight distances diverge.

        Uses only weight vectors — independent of dataset size.
        Returns ρ ∈ [-1, 1]. Higher is better (EA objective: minimise 1 − ρ).
        """
        from scipy.spatial.distance import pdist
        from scipy.stats import spearmanr

        flat_w = self.weights.reshape(-1, self.dim)

        data_dists = pdist(flat_w, metric='euclidean')

        coord_flat = self.neuron_coords.reshape(-1, 2)
        i_vals = coord_flat[:, 0].astype(float)
        j_vals = coord_flat[:, 1].astype(float)
        if self.map_type == 'hex':
            x_phys = j_vals + 0.5 * (i_vals % 2)
            y_phys = i_vals * (np.sqrt(3) / 2)
            grid_dists = pdist(np.stack([x_phys, y_phys], axis=1), metric='euclidean')
        else:
            grid_dists = pdist(np.stack([i_vals, j_vals], axis=1), metric='euclidean')

        if data_dists.std() < 1e-10 or grid_dists.std() < 1e-10:
            return 0.0

        rho, _ = spearmanr(data_dists, grid_dists)
        return float(rho) if not np.isnan(rho) else 0.0

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
            'u_matrix_std': np.std(u_matrix),
            'u_matrix_max': np.max(u_matrix)
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

        if self.map_type == 'hex':
            bmu_cube_coords = self.cube_coords[bmu_i, bmu_j]
            distances = np.sum(np.abs(self.cube_coords - bmu_cube_coords), axis=2) / 2
        else:  # square / rect — everything non-hex, same convention as all other branches
            distances = np.linalg.norm(self.neuron_coords - np.array([bmu_i, bmu_j]), axis=2)

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

    def train(self, data: np.ndarray, ignore_mask: np.ndarray = None,
              lstm_early_stop_fn=None, dynamic_schedule_fn=None, log_fn=None) -> dict:
        """
        Main training loop. Pure compute: writes nothing to disk.

        Args:
            data: normalized training data (N, dim).
            ignore_mask: per-sample boolean mask of dimensions to ignore.
            lstm_early_stop_fn: optional callback(checkpoints) -> (stop, score).
            dynamic_schedule_fn: optional callback(checkpoint) -> (lr_factor, radius_factor).
            log_fn: optional callback(message) for progress/intervention logging.

        Returns dict with best_mqe, duration, history, checkpoints, sample_coverage
        (incl. per-sample counts), convergence flags. Final weights stay on
        self.weights — persist via som.persistence.save_weights().
        """
        _log = log_fn if log_fn is not None else (lambda message: None)
        start_time = time.monotonic()

        if ignore_mask is not None:
            # Dimensions masked for every sample (e.g. the primary ID column)
            # can never receive a weight update — their weights would stay
            # random initialization noise. Zero them so they contribute exactly
            # nothing to computations that run without the mask (U-matrix
            # metrics, topological correlation, stored-artifact rendering).
            fully_masked_dims = ignore_mask.all(axis=0)
            if fully_masked_dims.any():
                self.weights[:, :, fully_masked_dims] = 0.0

        self.mqe_history = []
        self.best_mqe = float('inf')
        converged = False
        self.total_weight_updates = 0

        total_samples = data.shape[0]
        total_iterations = int(total_samples * self.epoch_multiplier)

        best_moving_avg = float('inf')
        epochs_without_improvement = 0
        recent_mqe_history = deque(maxlen=self.early_stopping_window)

        # Prepare batch indices for processing.
        # The sampling scheme (permutation -> array_split -> per-iteration
        # ceil + choice/pointer) is replicated in app/tools/coverage_sim.py —
        # keep both in sync (verified via its `verify` subcommand).
        shuffled_indices = np.random.permutation(total_samples)
        section_indices = np.array_split(shuffled_indices, self.num_batches)

        if self.sampling_method == 'reshuffle':
            _sec_orders = [np.random.permutation(sec) for sec in section_indices]
            _sec_positions = [0] * self.num_batches

        if self.mqe_evaluations_per_run > 0:
            mqe_compute_interval = max(1, total_iterations // self.mqe_evaluations_per_run)
        else:
            mqe_compute_interval = total_iterations

        # Checkpointing setup for LSTM training data
        checkpoints = []          # written to disk when save_checkpoints=True
        lstm_checkpoints = []     # in-memory only, for LSTM early stopping
        if self.save_checkpoints and self.checkpoint_every_mqe:
            checkpoint_interval = mqe_compute_interval  # checkpoint at every MQE evaluation
        elif self.save_checkpoints and self.checkpoint_count > 0:
            checkpoint_interval = max(1, total_iterations // self.checkpoint_count)
        else:
            checkpoint_interval = total_iterations + 1  # Never save checkpoints

        # LSTM fires after this many checkpoints (≥ 20 % of training = minimum K the model saw)
        lstm_min_checkpoints = max(2, self.mqe_evaluations_per_run // 5)
        lstm_stopped = False
        lstm_stop_progress = None

        # Phase 3 dynamic controller: cumulative multipliers applied on top of static schedule
        _cum_lr_factor = 1.0
        _cum_radius_factor = 1.0
        _ctrl_cp_count = 0
        _ctrl_log_every = max(1, self.mqe_evaluations_per_run // 10)

        _coverage = np.zeros(total_samples, dtype=np.int64) if self.track_sample_coverage else None

        pbar = tqdm(range(total_iterations), desc="SOM Training", unit="iter",
                    disable=not self.show_progress)
        iteration = 0  # Initialize iteration variable to avoid scope issues
        for iteration in pbar:
            current_lr = self.get_decay_value(iteration, total_iterations, self.start_learning_rate,
                                              self.end_learning_rate, self.lr_decay_type) * _cum_lr_factor
            current_radius = self.get_decay_value(iteration, total_iterations, self.start_radius, self.end_radius,
                                                  self.radius_decay_type) * _cum_radius_factor

            self.history['learning_rate'].append((iteration, current_lr))
            self.history['radius'].append((iteration, current_radius))

            # Sample selection: dynamic batch from divided sections
            batch_percent = self.get_batch_percent(iteration, total_iterations)
            samples_per_section_float = (total_samples * batch_percent / 100.0)
            samples_per_section = max(1, math.ceil(samples_per_section_float))

            selected_indices = []
            if self.sampling_method == 'reshuffle':
                for s, section in enumerate(section_indices):
                    sec_len = len(section)
                    if sec_len == 0:
                        continue
                    remaining = min(samples_per_section, sec_len)
                    while remaining > 0:
                        take_now = min(remaining,
                                       sec_len - _sec_positions[s])
                        selected_indices.extend(
                            _sec_orders[s][_sec_positions[s]:
                                           _sec_positions[s] + take_now])
                        _sec_positions[s] += take_now
                        remaining -= take_now
                        if _sec_positions[s] >= sec_len:
                            _sec_orders[s] = np.random.permutation(section)
                            _sec_positions[s] = 0
            else:
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

            if _coverage is not None:
                # reshuffle can wrap an epoch boundary within one iteration,
                # producing duplicate indices — np.add.at counts them all
                np.add.at(_coverage, indices_to_process, 1)

            if self.normalize_weights_flag:
                self.normalize_weights()

            # Compute MQE and check stopping criteria
            if iteration % mqe_compute_interval == 0 or iteration == total_iterations - 1:
                _, current_mqe = self.compute_quantization_error(data, mask=ignore_mask)
                self.history['mqe'].append((iteration, current_mqe))

                # Build checkpoint record (needed for both file saving and LSTM)
                if (self.save_checkpoints and iteration % checkpoint_interval == 0) \
                        or lstm_early_stop_fn is not None:
                    topo_error = self.calculate_topographic_error(data, mask=ignore_mask)
                    _, dead_ratio = self.calculate_dead_neurons(data, mask=ignore_mask)
                    progress = iteration / total_iterations
                    cp = {
                        'iteration': iteration,
                        'progress': progress,
                        'mqe': current_mqe,
                        'topographic_error': topo_error,
                        'dead_neuron_ratio': dead_ratio,
                        'learning_rate': current_lr,
                        'radius': current_radius,
                        'lr_factor': 1.0,
                        'radius_factor': 1.0,
                    }

                    # Phase 3 dynamic controller: apply factor, update cumulative multipliers
                    if dynamic_schedule_fn is not None:
                        lr_f, rad_f = dynamic_schedule_fn(cp)
                        lr_f = float(np.clip(lr_f, 0.5, 2.0))
                        rad_f = float(np.clip(rad_f, 0.5, 2.0))
                        _cum_lr_factor = float(np.clip(_cum_lr_factor * lr_f, 0.05, 5.0))
                        _cum_radius_factor = float(np.clip(_cum_radius_factor * rad_f, 0.05, 5.0))
                        cp['lr_factor'] = lr_f
                        cp['radius_factor'] = rad_f
                        _ctrl_cp_count += 1
                        # Periodic milestone log (every ~10% of training)
                        if _ctrl_cp_count % _ctrl_log_every == 0:
                            _log(f"LSTM ctrl @ {progress:.0%}: "
                                 f"step lr_f={lr_f:.4f} rad_f={rad_f:.4f} | "
                                 f"cum_lr={_cum_lr_factor:.4f} cum_rad={_cum_radius_factor:.4f}")
                        # Intervention log: fired when controller deviates >1% from neutral
                        _lr_dev = abs(lr_f - 1.0)
                        _rad_dev = abs(rad_f - 1.0)
                        if _lr_dev > 0.01 or _rad_dev > 0.01:
                            _log(f"LSTM ctrl INTERVENTION @ {progress:.1%}: "
                                 f"lr_f={lr_f:.4f} (Δ{lr_f - 1.0:+.4f}) "
                                 f"rad_f={rad_f:.4f} (Δ{rad_f - 1.0:+.4f}) | "
                                 f"effective lr={current_lr:.5f} R={current_radius:.4f} | "
                                 f"cum_lr={_cum_lr_factor:.4f} cum_rad={_cum_radius_factor:.4f}")

                    if self.save_checkpoints and iteration % checkpoint_interval == 0:
                        checkpoints.append(cp)
                    if lstm_early_stop_fn is not None:
                        lstm_checkpoints.append(cp)

                    # LSTM early stopping — only after enough progress (≥ 20 % of run)
                    if lstm_early_stop_fn is not None \
                            and len(lstm_checkpoints) >= lstm_min_checkpoints:
                        should_stop, lstm_score = lstm_early_stop_fn(lstm_checkpoints)
                        if should_stop:
                            _log(f"LSTM early stopping triggered at iteration {iteration} "
                                 f"(progress={progress:.1%}, score={lstm_score:.3f})")
                            converged = True
                            lstm_stopped = True
                            lstm_stop_progress = progress
                            break

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

        # Sample coverage stats if tracking was enabled (persisted by the caller)
        coverage_stats = None
        if _coverage is not None:
            never_processed = int(np.sum(_coverage == 0))
            coverage_stats = {
                'min': int(_coverage.min()),
                'max': int(_coverage.max()),
                'mean': round(float(_coverage.mean()), 2),
                'std': round(float(_coverage.std()), 2),
                'never_processed': never_processed,
                'never_processed_ratio': round(never_processed / total_samples, 4),
                'total_samples': total_samples,
                'counts': _coverage.tolist(),
            }
            _log(f"Sample coverage: min={coverage_stats['min']} "
                 f"max={coverage_stats['max']} "
                 f"mean={coverage_stats['mean']} "
                 f"never_processed={never_processed} "
                 f"({coverage_stats['never_processed_ratio']*100:.1f}%)")

        return {
            'best_mqe': self.best_mqe,
            'duration': duration,
            'total_weight_updates': self.total_weight_updates,
            "epochs_ran": iteration + 1,
            "converged": converged,
            "lstm_stopped": lstm_stopped,
            "lstm_stop_progress": lstm_stop_progress,
            "history": self.history,
            "checkpoints": checkpoints if self.save_checkpoints else [],
            "sample_coverage": coverage_stats,
        }
