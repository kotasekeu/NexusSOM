# LSTM — Early Stopping + Dynamic Schedule Controller

Dva oddělené LSTM modely pro optimalizaci SOM tréninku:

| Model | Fáze | Účel | Soubor |
|---|---|---|---|
| **Early stopping** | Phase 2 | Z prefixu sekvence predikuje finální kvalitu SOM → zastaví špatné běhy | `lstm_latest.keras` |
| **Controller** | Phase 3 | Per-checkpoint predikuje `(lr_factor, radius_factor)` → dynamicky ohýbá decay křivky | `lstm_controller_latest.keras` |

Detailní teorie: [docs/lstm/LSTM_DYNAMIC_CONTROL.md](../../docs/lstm/LSTM_DYNAMIC_CONTROL.md)  
Analýza vizualizací a známé problémy: [docs/lstm/LSTM_MODEL_ANALYSIS.md](../../docs/lstm/LSTM_MODEL_ANALYSIS.md)  
Phase 3 data generation: [docs/lstm/PHASE3_DATA_GENERATION.md](../../docs/lstm/PHASE3_DATA_GENERATION.md)

---

## Adresářová struktura

```
app/lstm/
├── README.md                       # tento soubor
├── prepare_dataset.py              # Phase 2: příprava trénovacích dat z EA výsledků
├── prepare_phase3_dataset.py       # Phase 3: příprava dat z perturbačních trajektorií
├── generate_phase3_data.py         # Phase 3: generování perturbačních běhů SOM
├── visualize_model.py              # vizualizace obou modelů (--phase 2 / --phase 3)
├── evaluate_model.py               # detailní evaluace Phase 2
├── collect_training_data.py        # legacy — nepoužívat
├── src/
│   ├── model.py                    # Phase 2 architektura (hybrid LSTM + context)
│   ├── model_controller.py         # Phase 3 architektura (_TileContext, _ScaleSigmoid)
│   ├── train.py                    # Phase 2 trénink
│   └── train_phase3.py             # Phase 3 trénink
├── data/
│   ├── X_train.npy  y_train.npy    # Phase 2 trénovací data
│   ├── ctx_train.npy               # Phase 2 dataset kontext
│   ├── metadata.json               # Phase 2 metadata
│   └── phase3/
│       ├── trajectories.json       # surové perturbační trajektorie
│       ├── X_train.npy  y_train.npy
│       ├── ctx_train.npy  adv_train.npy
│       └── metadata_p3.json
└── models/
    ├── lstm_latest.keras           # Phase 2 model
    ├── lstm_scaler_latest.pkl      # Phase 2 scaler
    ├── lstm_controller_latest.keras
    └── lstm_controller_scaler_latest.pkl
```

---

## Phase 2 — Early Stopping Predictor

### Co dělá

Dostane prvních K % MQE checkpointů SOM tréninku (K ∈ 20–70 %) a predikuje finální hodnoty:
- `raw_mqe_improvement_ratio` — jak moc SOM zlepší MQE oproti počátku
- `raw_topographic_error` — výsledná topologická chyba
- `dead_neuron_ratio` — podíl neaktivních neuronů

Z predikcí se spočítá `quality_score = (1 − mqe_ratio) + te + dead×0.5`. Pokud překročí threshold → trénink se zastaví.

### Vstup / výstup modelu

```
Vstup:
  sequence:  (K, 6)   — progress, mqe_rel, topo_error, dead_ratio, lr_rel, radius_rel
  context:   (1, 4)   — ds_n_samples, ds_n_active_dimensions, ds_n_numeric, ds_n_categorical

Výstup:
  (3,)  — predikce [mqe_improvement_ratio, topographic_error, dead_neuron_ratio]
```

### Pipeline přípravy dat

```bash
# 1. Připrav trénovací data ze všech EA seedů a datasetů
.venv/bin/python3 app/lstm/prepare_dataset.py \
    --results_root data/datasets \
    --output app/lstm/data

# Výstup: data/X_train.npy, y_train.npy, ctx_train.npy (+ val, test), metadata.json
```

**Co skript dělá:**
- Prochází `data/datasets/<DS>/results/<TS>/seed_*/` a sbírá `training_checkpoints.json`
- Z každého běhu vytvoří K-prefix okna (K ∈ 20, 30, 40, 50, 60, 70 % délky sekvence)
- Normalizuje sekvence (MQE/initial MQE, LR/initial LR, radius/initial radius)
- Split 70/15/15 na úrovni UID (žádný leak mezi variantami stejného jedince)

### Trénink

```bash
cd app/lstm
../.venv/bin/python3 src/train.py
# Model se uloží jako: models/lstm_latest.keras + models/lstm_scaler_latest.pkl
```

### Vizualizace

```bash
cd app/lstm
../.venv/bin/python3 visualize_model.py --phase 2
# Výstup: visualizations/phase2/{scatter,residuals,prefix_accuracy,early_stopping}.png
```

**Interpretace grafů:**
- `scatter.png` — `r ≈ 0.97` pro MQE (dobrý), `r ≈ 0.50` pro TE (strukturálně slabší)
- `early_stopping.png` — confusion matrix; pokud jsou všechny body jen v jedné třídě, threshold `lstm_quality_threshold` potřebuje kalibraci

### Konfigurace a aktivace

V `config-ea.json` (NEURAL_NETWORKS sekce):
```json
{
  "use_lstm": true,
  "lstm_model_path": "app/lstm/models/lstm_latest.keras",
  "lstm_scaler_path": "app/lstm/models/lstm_scaler_latest.pkl",
  "lstm_quality_threshold": 0.75
}
```

> **Poznámka k thresholdu**: hodnota 0.75 není kalibrovaná na skutečnou distribuci quality score.
> Doporučeno: spočítat 60.–70. percentil quality score na trénovací sadě a použít ho jako threshold.
> Viz issue #68 v [docs/ea/ISSUES.md](../../docs/ea/ISSUES.md).

### Kdy se early stopping aktivuje

- Nejdříve po `lstm_min_checkpoints = max(2, mqe_evaluations_per_run // 5)` checkpointech (= 20 % tréninku)
- Volá se každý checkpoint od tohoto prahu
- Pokud `quality_score > lstm_quality_threshold` → SOM trénink se zastaví

---

## Phase 3 — Dynamic Schedule Controller

### Co dělá

Při každém MQE checkpointu vrátí `(lr_factor, radius_factor)` ∈ [0.5, 1.5]. Faktory se aplikují multiplikativně na statický schedule:

```
actual_lr     = static_schedule_lr(t)     × cumulative_lr_factor
actual_radius = static_schedule_radius(t) × cumulative_radius_factor
```

Controller nenahrazuje decay křivku — ohýbá ji. Statický schedule zůstává jako fallback.

### Vstup / výstup modelu

```
Vstup (per checkpoint):
  sequence:  (1, 1, 6) — progress, mqe_rel, topo_error, dead_ratio, lr_rel, radius_rel
  context:   (1, 4)    — ds_n_samples, ds_n_active_dimensions, ds_n_numeric, ds_n_categorical

Výstup:
  (1, 2)  — (lr_factor, radius_factor), oba ∈ [0.5, 1.5]
```

### Pipeline přípravy dat — Phase 3

#### Krok 1: Generování perturbačních trajektorií

```bash
# Všechny datasety + všechny seedy (doporučeno)
.venv/bin/python3 app/lstm/generate_phase3_data.py \
    --results_root data/datasets \
    --n_pareto 5 --n_variants 8 \
    --n_workers 0 \
    --output app/lstm/data/phase3

# Jeden seed (debug)
.venv/bin/python3 app/lstm/generate_phase3_data.py \
    --seed_dir data/datasets/LungCancerDataset/results/20260515_111812/seed_42 \
    --dataset  data/datasets/LungCancerDataset/dataset.csv \
    --n_pareto 5 --n_variants 8
```

**Co skript dělá:**
- Vezme top N Pareto individuí z každého EA seedu
- Každé individuum spustí 1× jako baseline + 8× s náhodnou perturbací
- Perturbace: `lr_factor, radius_factor = U(0.75, 1.25)` s `prob=0.4` per checkpoint
- **Fyzikální meze** jsou vynuceny: `lr ∈ [1e-4, start_lr]`, `radius ∈ [1.0, max(m,n)]`
- Výstup: `advantage = (Δmqe/σ_mqe) + (Δte/σ_te) + (Δdead/σ_dead)` (multi-objektová)
- Paralelizace přes `Pool(cpu_count - 1)` — výstup je jeden `trajectories.json`

**Typy perturbace** (střídají se cyklicky):

| Typ | Co se mění |
|---|---|
| `lr+radius` | oba parametry současně |
| `lr_only` | pouze learning rate |
| `radius_only` | pouze radius |

#### Krok 2: Příprava trénovacích dat

```bash
# Auto-detekce seed dirs ze všech datasetů (doporučeno)
.venv/bin/python3 app/lstm/prepare_phase3_dataset.py

# Explicitně
.venv/bin/python3 app/lstm/prepare_phase3_dataset.py \
    --results_root data/datasets \
    --trajectories app/lstm/data/phase3/trajectories.json \
    --output app/lstm/data/phase3
```

**Co skript dělá:**
- Načte `trajectories.json`, přeskočí baseline varianty (ty nemají perturbační signál)
- Normalizuje sekvence checkpointů (stejný postup jako Phase 2)
- Advantage-weighted training: `adv = max(0, advantage) / max_advantage` → [0, 1]
- Z-score normalizace advantage: každá složka Δ se dělí svým globálním σ přes všechny záznamy — zabraňuje dominanci jedné metriky napříč datasety různých měřítek
- Výstup: `X, y, ctx, adv` pro train/val/test, `metadata_p3.json`

> **Důležité**: Před tréninkem filtrovat timestepy kde `|lr_factor − 1.0| < ε AND |radius_factor − 1.0| < ε`
> (60 % timestepů nemá perturbaci → model kolabuje na predikci 1.0).
> Viz issue #71 v [docs/ea/ISSUES.md](../../docs/ea/ISSUES.md).

#### Krok 3: Trénink

```bash
cd app/lstm
../.venv/bin/python3 src/train_phase3.py
# Model: models/lstm_controller_latest.keras + models/lstm_controller_scaler_latest.pkl
```

### Vizualizace

```bash
cd app/lstm
../.venv/bin/python3 visualize_model.py --phase 3
# Výstup: visualizations/phase3/{scatter_p3,residuals_p3,advantage_p3}.png
```

**Interpretace grafů:**
- `scatter_p3.png` — pokud jsou predikce clustered na 0.5 nebo 1.0, model kolaboval (issue #71)
- `residuals_p3.png` — spike na +0.5 = padding artefakt (issue #72)
- `advantage_p3.png` — Q1 n=0 = bug v quartile split při many-tie hodnotách (issue #73)

### Konfigurace a aktivace

V `config-ea.json` nebo `config-som-mlp-lstm.json`:
```json
{
  "use_lstm_controller": true,
  "lstm_controller_model_path": "app/lstm/models/lstm_controller_latest.keras",
  "lstm_controller_scaler_path": "app/lstm/models/lstm_controller_scaler_latest.pkl"
}
```

### Logování zásahů v SOM

```
LSTM ctrl @ 20%: step lr_f=0.998 rad_f=1.002 | cum_lr=0.941 cum_rad=0.970
LSTM ctrl INTERVENTION @ 43%: lr_f=1.052 (Δ+0.052) rad_f=0.931 (Δ-0.069) | effective lr=0.003 R=3.18 | cum_lr=1.018 cum_rad=0.894
```

Milestone log každých ~10 % tréninku. Intervention log pouze při zásahu > 1 %.

---

## Kompletní pipeline (od nuly)

```bash
# ── PHASE 2 ──────────────────────────────────────────────────────────
# Předpoklad: EA běhy dokončeny v data/datasets/

# 1. Připrav data
.venv/bin/python3 app/lstm/prepare_dataset.py --results_root data/datasets

# 2. Natrénuj
cd app/lstm && ../.venv/bin/python3 src/train.py

# 3. Vizualizuj
../.venv/bin/python3 visualize_model.py --phase 2

# ── PHASE 3 ──────────────────────────────────────────────────────────
# Předpoklad: Phase 2 model natrénován, EA výsledky k dispozici

# 1. Generuj perturbační data (paralelně, ~30–90 min dle počtu seedů)
cd .. && .venv/bin/python3 app/lstm/generate_phase3_data.py \
    --results_root data/datasets --n_pareto 5 --n_variants 8

# 2. Připrav trénovací dataset
.venv/bin/python3 app/lstm/prepare_phase3_dataset.py

# 3. Natrénuj controller
cd app/lstm && ../.venv/bin/python3 src/train_phase3.py

# 4. Vizualizuj
../.venv/bin/python3 visualize_model.py --phase 3
```

---

## Aktuální stav modelů

| Model | Trénovací data | Test MAE | Stav |
|---|---|---|---|
| Phase 2 (early stopping) | 4 datasety × 5 seedů × 5 gen × 30 pop | 0.041 | ✅ Nasazeno, threshold potřebuje kalibraci |
| Phase 3 (controller) | Perturbace Pareto individuí, 4 DS | 0.10–0.12 | ⚠️ Fyzikální oprava dat nutná — regenerovat |

**Prioritní opravy před dalším nasazením Phase 3:**
1. Regenerovat `trajectories.json` se `make_constrained_perturb_fn` (již implementováno)
2. Filtrovat nepřeturbované timestepy v `prepare_phase3_dataset.py`
3. Opravit padding masking při evaluaci

Viz [docs/lstm/LSTM_MODEL_ANALYSIS.md](../../docs/lstm/LSTM_MODEL_ANALYSIS.md) pro detaily.
