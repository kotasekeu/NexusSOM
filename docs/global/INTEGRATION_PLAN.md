# NexusSOM — Integrační plán

**Verze**: 2.0  
**Datum**: 2026-05-15

---

## Architektura systému

```
Dataset CSV
    │
    ▼
┌─────────────────────────────────────────────┐
│  EA (NSGA-II)                               │
│                                             │
│  ┌──────────┐   config   ┌───────────────┐  │
│  │  MLP     │──────────▶│  SOM trénink  │  │
│  │ pre-screen│           │               │  │
│  └──────────┘           │  LSTM early   │  │
│                          │  stopping     │  │
│  Pareto fronta           │               │  │
│  HV + Spacing + Spread   │  LSTM         │  │
│  pareto_metrics.csv      │  controller   │  │
└──────────────────────────┴───────────────┘──┘
    │
    ▼
Nejlepší SOM konfigurace
    │
    ▼
Analytický modul → llm_context.json → LLM report
```

---

## Role jednotlivých komponent

| Komponenta | Kdy se aktivuje | Co dělá |
|---|---|---|
| **Kalibrační sonda** | Před G0 | N rychlých SOM běhů → `org_threshold` (70. percentil) |
| **Vesanto + dynamic search space** | Před G0 | Nastaví rozsah `map_size` a `epoch_multiplier` z `n_samples` |
| **MLP pre-screen** | Per evaluace | Predikuje kvalitu konfigurace → přeskočí slabé (`pred_ratio < threshold`) |
| **SOM trénink** | Per evaluace | Trénuje SOM, generuje checkpointy |
| **LSTM early stopping** | Per evaluace (po 20 % tréninku) | Predikuje finální kvalitu → zastaví špatné běhy |
| **LSTM controller** | Per checkpoint | Multiplicativně upravuje `lr_factor` a `radius_factor` |
| **NSGA-II + Pareto** | Per generace | Selekce, crowding distance, archiv rank==0, HV+Spacing+Spread do CSV |

---

## Trénovací plán (7 fází)

Detaily v [docs/TRAINING_PLAN.md](../TRAINING_PLAN.md). Zkrácený přehled:

### Fáze 1 — 1 dataset, 1 seed → pipeline ověření

**Stav**: ✅ BreastCancer seed_42 dokončen (149 individuálů, 87 feasible)

```
6 seedů × 5 generací × 30 populace = 900 evaluací
use_lstm_controller: false
use_lstm: false
use_mlp: false
```

Po EA: perturbace Pareto konfigurací (`generate_phase3_data.py`) → trénink MLP, LSTM Phase 2, LSTM Phase 3.

### Fáze 2–3 — 5 → 10 reálných datasetů

Postupné rozšiřování, kumulativní přetrénování modelů. Cíl: křivka učení (1 DS → 5 DS → 10 DS) a generalizace přes datasety.

### Fáze 4 — Generované datasety *(odloženo)*

Nástroj existuje, fáze se řeší až "máme spoustu času".

### Fáze 5 — EA s dynamickým řízením

Srovnání: statický schedule (use_lstm_controller: false) vs dynamický (true).  
Metriky: finální HV, HV křivka, Spacing, Spread, počet LSTM zásahů.

### Fáze 6 — Srovnání 4 SOM módů

Stochastický / deterministický / MLP+statický / MLP+LSTM controller — per dataset.

### Fáze 7 — Anomaly detection test

Záměrně zanesená chyba v datasetu → SOM analýza → LLM signal.

---

## Konfigurace EA běhů

### Fáze 1 (sběr dat, modely vypnuty)
```json
{
  "NN_CONFIG": {
    "use_mlp": false,
    "use_lstm": false,
    "use_lstm_controller": false,
    "use_cnn": false
  },
  "EA_SETTINGS": {
    "seeds": [42, 1337, 7, 101, 2026, 999],
    "num_generations": 5,
    "population_size": 30
  }
}
```

### Fáze 5 (dynamické řízení zapnuto)
```json
{
  "NN_CONFIG": {
    "use_mlp": true,
    "use_lstm": true,
    "use_lstm_controller": true
  }
}
```

---

## Výstupní soubory per seed

```
results/<TIMESTAMP>/seed_<N>/
├── results.csv               ← všichni jedinci (hyperparametry + metriky)
├── pareto_front.csv          ← Pareto archiv per generace (per řešení)
├── pareto_metrics.csv        ← HV, Spacing, Spread per generace
├── status.csv                ← stav evaluací
├── log.txt                   ← textový log
└── individuals/<uid>/
    ├── csv/
    │   ├── training_checkpoints.json   ← pro LSTM trénink
    │   ├── weights.npy                 ← váhy SOM
    │   └── weights_readable.csv
    └── visualizations/
        ├── u_matrix.png
        ├── distance_map.png
        └── dead_neurons_map.png
```

---

## Trénink NN modelů

### MLP
```bash
.venv/bin/python3 app/mlp/prepare_dataset.py --results_root data/datasets/<DS>/results
cd app/mlp && ../.venv/bin/python3 src/train.py
../.venv/bin/python3 visualize_model.py
```

### LSTM early stopping
```bash
.venv/bin/python3 app/lstm/prepare_dataset.py --results_root data/datasets/<DS>/results
cd app/lstm && ../.venv/bin/python3 src/train.py
../.venv/bin/python3 visualize_model.py
```

### LSTM controller (Phase 3)
```bash
# 1. Perturbační data z Pareto konfigurací
.venv/bin/python3 app/lstm/generate_phase3_data.py \
    --seed_dir data/datasets/<DS>/results/<TS>/seed_42 \
    --dataset  data/datasets/<DS>/dataset.csv \
    --n_pareto 5 --n_variants 8

# 2. Příprava datasetu
.venv/bin/python3 app/lstm/prepare_phase3_dataset.py

# 3. Trénink
cd app/lstm && ../.venv/bin/python3 src/train_phase3.py
```

---

## Klíčové soubory

| Soubor | Role |
|---|---|
| [app/ea/ea.py](../../app/ea/ea.py) | EA hlavní logika: NSGA-II, kalibrace, HV/Spacing, NN integrace |
| [app/ea/nn_integration.py](../../app/ea/nn_integration.py) | Načítání MLP, LSTM, LSTM controller |
| [app/som/som.py](../../app/som/som.py) | SOM trénink, LSTM controller callback, intervention log |
| [app/mlp/src/train.py](../../app/mlp/src/train.py) | MLP trénink |
| [app/lstm/src/train.py](../../app/lstm/src/train.py) | LSTM early stopping trénink |
| [app/lstm/src/train_phase3.py](../../app/lstm/src/train_phase3.py) | LSTM controller trénink |
| [app/lstm/generate_phase3_data.py](../../app/lstm/generate_phase3_data.py) | Perturbační trajektorie |
| [docs/TRAINING_PLAN.md](../TRAINING_PLAN.md) | Kompletní 7-fázový plán |
| [docs/ea/VERIFICATION.md](../ea/VERIFICATION.md) | HV, Spacing, Spread — interpretace a nástroje |
| [docs/ea/issues.md](../ea/issues.md) | Všechny problémy a rozhodnutí EA |
| [docs/lstm/LSTM_DYNAMIC_CONTROL.md](../lstm/LSTM_DYNAMIC_CONTROL.md) | Phase 3 controller dokumentace |

---

## Aktuální stav

| Fáze | Stav | Poznámka |
|---|---|---|
| Fáze 1 — sběr dat | 🔄 Probíhá | BreastCancer seed_42 ✅; zbývá 5 seedů |
| Fáze 1 — trénink modelů | ⏳ Čeká | Spustit po dokončení všech seedů |
| Fáze 2–3 | ⏳ Čeká | Po dokončení Fáze 1 |
| Fáze 4 (generované DS) | ⏸ Odloženo | — |
| Fáze 5–7 | ⏳ Čeká | — |
