CONFIG = {
    "EA_SETTINGS": {
        "population_size": 20,
        "generations": 30
    },
    "SEARCH_SPACE": {
        "map_size": [
            (8, 8),
            (10, 10),
            (15, 15)
        ],
        "processing_type": ["stochastic", "deterministic", "hybrid"],
        "start_learning_rate": [0.9, 0.8, 0.7, 0.6, 0.5],
        "end_learning_rate": [0.2, 0.1, 0.05, 0.01],
        "lr_decay_type": ["linear-drop", "exp-drop", "log-drop", "step-down"],
        "start_radius_init_ratio": [1.0, 0.75, 0.5, 0.25, 0.1],
        "radius_decay_type": ["linear-drop", "exp-drop", "log-drop", "step-down"],
        "start_batch_percent": [0.025, 0.5, 1.0, 5.0, 10.0],
        "end_batch_percent": [3.0, 5.0, 7.5, 10.0, 15.0],
        "batch_growth_type": ["linear-growth", "exp-growth", "log-growth"],
        "epoch_multiplier": [5.0, 10.0, 15.0],
        "normalize_weights_flag": [False, True],
        "growth_g": [1.0, 5.0, 15.0, 25.0, 35.0],
        "num_batches": [1, 3, 5, 10, 20],
        "map_type": ["hex", "square"]
    },
    "FIXED_PARAMS": {
        "end_radius": 1.0,
        "random_seed": 42,
        "mqe_evaluations_per_run": 500,
        "max_epochs_without_improvement": 50,
        "early_stopping_window": 5
    },
    "PREPROCES_DATA": {
        "delimiter": ",",
        "categorical_threshold_numeric": 30,
        "categorical_threshold_text": 30,
        "noise_threshold_ratio": 0.2,
        "primary_id": "primary_id"
    },
    "DATA_PARAMS": {
        "sample_size": 1000,
        "niput_dim": 10
    },
    "NEURAL_NETWORKS": {
        "use_mlp": False,  # Enable MLP "The Prophet" for fast fitness estimation
        "use_lstm": False,  # Enable LSTM "The Oracle" for early stopping
        "use_cnn": False,  # Enable CNN "The Eye" for visual quality assessment
        "mlp_model_path": None,  # Auto-detect if None
        "mlp_scaler_path": None,  # Auto-detect if None
        "lstm_model_path": None,  # Auto-detect if None
        "cnn_model_path": None,  # Auto-detect if None
        "lstm_quality_threshold": 1.0,  # Quality threshold for early stopping
        "mlp_filter_bad_configs": False,  # Filter configs with bad predicted quality
        "mlp_bad_quality_threshold": 0.5,  # MQE threshold for filtering
        "verbose": True  # Print NN status messages
    }
}