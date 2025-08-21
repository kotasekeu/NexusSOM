CONFIG = {
    "EA_SETTINGS": {
        "population_size": 10,
        "generations": 5,
        "sample_size": 1000,
        "input_dim": 10,
    },

    "SEARCH_SPACE": {
        "map_size": [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30)],
        "start_learning_rate": [0.99, 0.9, 0.8, 0.7, 0.6, 0.5],
        "end_learning_rate": [0.5, 0.4, 0.3, 0.2, 0.1, 0.075, 0.05, 0.01],
        "lr_decay_type": ["linear-drop", "exp-drop"],

        "start_radius_init_ratio": [1, 0.5, 0.25],
        "end_radius": [2, 1, 0.5],
        "radius_decay_type": ["linear-drop", "exp-drop"],

        "start_batch_percent": [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0],
        "end_batch_percent": [1.0, 2.0, 5.0, 7.5, 10.0, 15.0],
        "batch_growth_type": ["exp-growth", "linear-growth"],

        "epoch_multiplier": [0.5, 1.0, 2.0, 5.0],
        "normalize_weights_flag": [false, true],
        "growth_g": [5.0, 15.0, 30.0, 50.0]
    },

    "FIXED_PARAMS": {
        "random_seed": 42,
        "map_type": "square",
        "num_batches": 10,
        "max_epochs_without_improvement": 25
    }
}
