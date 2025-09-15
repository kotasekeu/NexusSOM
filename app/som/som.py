# som.py
class KohonenSOM:
    def __init__(self, dim: int, **kwargs):
        self.dim = dim
        self.m = kwargs.get('m', 10)
        self.n = kwargs.get('n', 10)

        print(f"INFO: KohonenSOM initialized with dim={self.dim}, map_size={self.m}x{self.n}")

    def train(self, data) -> dict:
        print(f"INFO: Training SOM with data of shape {data.shape}")
        # Simulace výsledků
        return {
            'final_mqe': 0.05,
            'training_duration': 10.5,
            'total_weight_updates': 1000
        }

    @property
    def best_mqe(self):
        return 0.05