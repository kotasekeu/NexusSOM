# NN Integrace do EA — Přehled a návod k použití

Popis architektury MLP + LSTM pre-screen a early stopping v EA pipeline.

---

## Přehled

EA nyní obsahuje tři volitelné NN moduly:

| Modul | Role | Stav |
|---|---|---|
| **MLP** | Pre-screen — přeskočí konfiguraci s předpovídanou nízkou kvalitou | ✅ Phase 2 hotovo |
| **LSTM** | Early stopping — ukončí SOM trénink pokud predikce z prefixu ukáže špatný výsledek | ✅ Phase 2 hotovo |
| **LSTM Controller** | Dynamické řízení — upravuje lr/radius faktory v každém MQE checkpointu | ✅ Phase 3 Část 1 hotovo |
| **CNN** | Klasifikace mapy z U-Matrix vizualizace | ❌ Uzavřeno (viz CNN_REQUIREMENTS.md) |

---

## Architektura MLP (pre-screen filtr)

MLP předpoví výslednou kvalitu SOM konfigurace **bez spuštění trénování**. EA volá MLP jako první krok při hodnocení každého jedince.

**Vstupy (25 dimenzí):**
- 10 numerických hyperparametrů SOM: `start_learning_rate`, `end_learning_rate`, `start_radius_init_ratio`, `start_batch_percent`, `end_batch_percent`, `epoch_multiplier`, `growth_g`, `num_batches`, `map_m`, `map_n`
- 5 statistik datasetu: `ds_n_samples`, `ds_n_active_dimensions`, `ds_n_numeric`, `ds_n_categorical`, `ds_missing_ratio`
- 10 one-hot kategorických: `lr_decay_type_{exp-drop,linear-drop,log-drop,step-down}`, `radius_decay_type_{exp-drop,linear-drop,log-drop,step-down}`, `batch_growth_type_{exp-growth,linear-growth}`

**Výstupy (3 targety):**
`raw_mqe_improvement_ratio`, `raw_topographic_error`, `dead_neuron_ratio`

**Filtrování:**
```python
if pred_mqe_improvement_ratio < mlp_bad_quality_threshold:
    skip  # nízké predikované zlepšení = špatná konfigurace
```
Výchozí práh: `0.5` — přeskočí konfigurace kde MLP předpovídá < 50% MQE improvement ratio.

**Orientační přesnost (4 datasety, ~3 500 vzorků):**
- `mqe_improvement_ratio` MAE ≈ 0.05
- `topographic_error` MAE ≈ 0.019
- `dead_neuron_ratio` MAE ≈ 0.018

---

## Architektura LSTM (early stopping)

LSTM dostane K-prefix sekvence SOM checkpointů a předpoví finální kvalitu. EA ukončí trénování pokud predikce z dostatečně dlouhého prefixu ukazuje na slabý výsledek.

**Hybridní architektura (dva vstupy):**
```
sequence input (K, 6):  mqe_ratio, topographic_error, dead_ratio, progress, map_m, map_n
    → Masking → LSTM(64) → Dropout(0.3) → LSTM(32) → concat ↘
context input (4,):     n_samples, n_active_dim, n_numeric, n_categorical              → Dense(16) ↗
    concat → Dense(32) → Dropout(0.2) → Dense(3)
```

**Výstupy:** `raw_mqe_improvement_ratio`, `raw_topographic_error`, `dead_neuron_ratio`

**Quality score:**
```python
quality_score = (1.0 - pred_mqe_improvement_ratio) + pred_topographic_error + pred_dead_neuron_ratio * 0.5
# nižší = lepší; zastavit pokud quality_score > lstm_quality_threshold
```
Výchozí práh: `0.75` (kalibrováno na p75 distribuce quality_score z testovacích dat).

**Minimální prefix:** LSTM se aktivuje až po `max(2, mqe_evaluations_per_run // 5)` checkpointech = 20% trénování (pro 300 evaluací = 60 checkpointů). Model byl natrénován na K ∈ {20,30,40,50,60,70}% délky sekvence.

---

## Konfigurace v `config-ea.json`

```json
"NEURAL_NETWORKS": {
  "use_mlp": true,
  "use_lstm": true,
  "use_cnn": false,
  "use_lstm_controller": false,
  "mlp_model_path": "app/mlp/models/mlp_latest.keras",
  "mlp_scaler_path": "app/mlp/models/mlp_scaler_latest.pkl",
  "lstm_model_path": "app/lstm/models/lstm_latest.keras",
  "lstm_scaler_path": "app/lstm/models/lstm_scaler_latest.pkl",
  "lstm_controller_model_path": "app/lstm/models/lstm_controller_latest.keras",
  "lstm_controller_scaler_path": "app/lstm/models/lstm_controller_scaler_latest.pkl",
  "mlp_filter_bad_configs": true,
  "mlp_bad_quality_threshold": 0.5,
  "lstm_quality_threshold": 0.75,
  "verbose": true
}
```

**`use_lstm_controller`**: aktivuje Phase 3 dynamický kontroler. Model se musí nejdřív natrénovat
(`app/lstm/src/train_phase3.py`). Pro Část 1 (24 trajektorií) je model konzervativní — predikuje
faktory blízko 1.0. Smysluplné výsledky přijdou po Části 2 (~200+ trajektorií).

**Cesty** jsou relativní k pracovnímu adresáři při spuštění `run_ea.py` (= kořen projektu).

---

## Jak NN vstupují do trénování EA — tok

```
EA: generate individual
        ↓
MLP pre-screen: encode_config_for_mlp(individual, dataset_meta)
    pred_mqe = mlp.predict(feature_vector)
    if pred_mqe < 0.5 → log "MLP pre-screen: skipping" → skip SOM → penalty
        ↓ (prošlo)
LSTM Controller (use_lstm_controller=true):
    dynamic_schedule_fn = nn.get_dynamic_schedule_fn(dataset_context)
        ↓
SOM training: trénování s checkpointy
    při každém MQE checkpointu:
      → dynamic_schedule_fn(cp) → (lr_factor, radius_factor)
         current_lr *= cum_lr_factor, current_radius *= cum_radius_factor
      → lstm_checkpoints[] se plní (LSTM early stopping)
    po >= 60 checkpointech (use_lstm=true):
        quality_score = (1-pred_mqe) + te + dead*0.5
        if quality_score > 0.75 → log "LSTM early stop" → přerušit trénink
        ↓ (dokončeno nebo ukončeno)
Evaluate individual → results.csv → Pareto
```

---

## Spuštění testu

```bash
# Z kořene projektu
python3 app/run_ea.py --config data/datasets/LungCancerDataset/config-ea.json
```

**Očekávané log zprávy:**

Při startu:
```
INFO: NN integration enabled — MLP=True, LSTM=True, CNN=False
✓ MLP model and scaler loaded
✓ LSTM model and scaler loaded
```

Při MLP filtrování:
```
MLP pre-screen: predicted MQE=0.31 < threshold=0.50 — skipping SOM training
```

Při LSTM early stopping:
```
LSTM early stop: quality_score=0.82 > threshold=0.75 at progress=0.43
```

---

## Ověření výsledků po testu

Po doběhnutí EA runu zkontroluj:

1. **MLP přeskočení** — kolik jedinců bylo filtrováno MLP před SOM tréninkem:
   ```bash
   grep "mlp_skip" data/results/LungCancerDataset/results/seed_*/results.csv | wc -l
   ```

2. **LSTM ukončení** — kolik SOM tréninků bylo zkráceno:
   ```bash
   grep "lstm_early_stop" data/results/LungCancerDataset/results/seed_*/results.csv | wc -l
   ```

3. **Kvalita Pareto fronty** — zkontroluj `raw_mqe_improvement_ratio` archivu; neměla by být systematicky nižší než baseline bez NN.

4. **Penalty check** — `is_penalized` u LSTM-ukončených běhů by mělo být `False` pokud LSTM pracuje správně (zastavilo skutečně špatné, ne dobré konfigurace).

---

## Přetrénování modelů

Po každém novém setu EA výsledků:

```bash
# MLP
python3 app/mlp/prepare_dataset.py --results_root data/results
cd app/mlp && python3 src/train.py

# LSTM Phase 2 (early stopping)
python3 app/lstm/prepare_dataset.py --results_root data/results
cd app/lstm && python3 src/train.py

# LSTM Phase 3 (dynamic controller) — po generování nových perturbovaných dat
python3 app/lstm/generate_phase3_data.py \
    --seed_dir data/datasets/<DS>/results/<RUN>/seed_42 \
    --dataset  data/datasets/<DS>/dataset.csv \
    --n_pareto 5 --n_variants 8
python3 app/lstm/prepare_phase3_dataset.py \
    --trajectories app/lstm/data/phase3/trajectories.json \
    --seed_dir data/datasets/<DS>/results/<RUN>/seed_42
cd app/lstm && python3 src/train_phase3.py
```

Stabilní cesty (`*_latest.keras`) se přepíší automaticky. Config-ea.json není potřeba měnit.

---

## Opravené chyby při implementaci Phase 2

Viz `docs/ea/ISSUES.md` položky #55–#59. Souhrn:

| # | Chyba | Efekt | Oprava |
|---|---|---|---|
| 55 | LSTM gated za `save_checkpoints` | LSTM nikdy nespustilo | Samostatný `lstm_checkpoints[]` |
| 56 | LSTM po 2 checkpointech (< 20%) | Předčasné predikce ze 2% dat | `lstm_min_checkpoints = run // 5` |
| 57 | `quality_score` invertovaný | Zastavovalo dobré běhy | `(1 - mqe_ratio)` místo `mqe_ratio` |
| 58 | MLP filtr `> threshold` | Přeskakoval dobré konfigurace | Opraveno na `< threshold` |
| 59 | MLP metadata cesta pro stable path | JSON decode error, MLP disabled | Kandidátní seznam cest |

---

## Stav Phase 3 (dynamické řízení)

**Část 1 hotova (2026-05-13):** pipeline end-to-end funkční, model natrénován.

| Co | Kde | Stav |
|---|---|---|
| `dynamic_schedule_fn` callback | `som.py` | ✅ |
| Data generování | `app/lstm/generate_phase3_data.py` | ✅ 45 trajektorií |
| Model architektura | `app/lstm/src/model_controller.py` | ✅ |
| Dataset příprava | `app/lstm/prepare_phase3_dataset.py` | ✅ |
| Trénink | `app/lstm/src/train_phase3.py` | ✅ MAE=0.041 |
| EA integrace | `nn_integration.py` + `ea.py` | ✅ |

**Část 2 — čeká:** rozšíření na 4 datasety, 200–400 SOM běhů, elbow features.
Viz `docs/lstm/LSTM_DYNAMIC_CONTROL.md` pro detailní plán.

**Pro aktivaci Phase 3 v EA:** nastav `"use_lstm_controller": true` v config.
Současný model je konzervativní (trénován na 24 trajektoriích) — výsledky se zlepší po Části 2.

FR-LSTM-2.6 (ověření LSTM early stopping v EA běhu) — otevřeno, čeká na test run.
