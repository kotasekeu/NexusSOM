# Neural Networks — Quick Start

> **Tento soubor je zastaralý.** Viz aktuální dokumentaci: [NN_INTEGRATION.md](NN_INTEGRATION.md)

---

## TL;DR

```bash
# Spuštění EA s MLP + LSTM
python3 app/run_ea.py --config data/datasets/LungCancerDataset/config-ea.json
```

Modely jsou uloženy v:
- `app/mlp/models/mlp_latest.keras`
- `app/lstm/models/lstm_latest.keras`

Config sekce `NEURAL_NETWORKS` v `config-ea.json` ovládá zapnutí/vypnutí.

Detaily architektury, konfigurace a řešení problémů: [NN_INTEGRATION.md](NN_INTEGRATION.md)
