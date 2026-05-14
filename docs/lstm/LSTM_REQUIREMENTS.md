# LSTM — Požadavky a roadmapa

**Verze**: 3.4  
**Aktualizováno**: 2026-05-13  
**Komponenta**: LSTM

---

## Stav implementace

### Hotovo ✅

**Data:**
- ✅ `training_checkpoints.json` — 5 853 individuí, 300–330 checkpointů každý
- ✅ Checkpoint formát: `{iteration, progress, mqe, topographic_error, dead_neuron_ratio, learning_rate, radius}`
- ✅ `results.csv` s targety, `pareto_front.csv` s identifikací nejlepších individuí

**Integrace v EA:**
- ✅ `lstm_early_stop_fn` callback — SOM ho volá při každém MQE checkpointu (`som.py:382`)
- ✅ `nn_integration.py::should_stop_early()` — interface pro predikci
- ✅ `ea.py:1161` — callback sestavení a předání do `som.train()`

### Hotovo (Phase 2) ✅

- ✅ `app/lstm/prepare_dataset.py` — 5 853 individuí → 18 282 tréninkových oken (K ∈ 20–70 %)
- ✅ `app/lstm/src/model.py` — hybridní LSTM+context architektura
- ✅ `app/lstm/src/train.py` — trénink, stable cesty, Keras 3 kompatibilní
- ✅ `app/lstm/models/lstm_latest.keras` + `lstm_scaler_latest.pkl` — natrénovaný model (test MAE≈0.023)
- ✅ Rozšíření `lstm_early_stop_fn` o progress/lr/radius (`ea.py:1163`)
- ✅ Rozšíření `should_stop_early` na 6-dim vstup + hybrid model interface (`nn_integration.py`)

### Hotovo (Phase 3, Část 1) ✅

- ✅ FR-LSTM-3.1: `dynamic_schedule_fn` callback v `som.py` (faktory, kumulativní drift, clipping)
- ✅ FR-LSTM-3.2: `app/lstm/generate_phase3_data.py` — 45 trajektorií, 14 013 checkpointů (LungCancer)
- ✅ FR-LSTM-3.3: `app/lstm/src/model_controller.py` — stateful (inference) + trainable verze
- ✅ FR-LSTM-3.4: `app/lstm/prepare_phase3_dataset.py` — advantage-weighted tréninkový dataset
- ✅ FR-LSTM-3.5: `app/lstm/src/train_phase3.py` — trénink, stable cesty, test MAE=0.041
- ✅ EA integrace: `nn_integration.py::get_dynamic_schedule_fn()` + `ea.py` wiring
- ✅ Config: `use_lstm_controller` flag v `config-ea.json` (default: false)

### Chybí ❌

- ❌ FR-LSTM-2.6: Ověření early stopping v EA běhu
- ❌ FR-LSTM-3.6: Rozšíření dat na 4 datasety, 200–400 SOM běhů (Část 2)
- ❌ FR-LSTM-3.7: Elbow features (d1, d2 MQE) jako extra dimenze sekvence
- ❌ FR-LSTM-3.8: Srovnávací EA validace: statický schedule vs. LSTM kontroler

---

## Přehled dvou fází a jejich přínos

```
Phase 2 ✅ hotovo                Phase 3 Část 1 ✅ hotovo
──────────────────────────────   ──────────────────────────────────
Vstup: checkpoints[0..K]         Vstup: checkpoint[t] (streamově)
Výstup: quality_score            Výstup: lr_factor, radius_factor
Účel: early stopping             Účel: dynamická úprava decay curves
Trénink: supervised              Trénink: advantage-weighted imitation
Data: existují ✅                 Data: 45 trajektorií ✅ (Část 2: ❌)
Test MAE ≈ 0.023                 Test MAE = 0.041
```

### Kde každá fáze přináší hodnotu

**Phase 2 — přínos je v EA, ne ve standalone SOM:**
Early stopping je relevantní jen pro EA, který evaluuje stovky konfigurací za běh.
LSTM zastaví slabé konfigurace na 30–50 % tréninku → EA zvládne více generací
za stejný výpočetní budget → lepší Pareto fronta.

Pro standalone SOM (jeden záměrně dobrý config) nemá early stop co dělat —
konfigurace je předem dobrá a LSTM ji nezastaví. Overhead monitoringu (~+22 %)
převáží přínos. Proto `checkpoint_every_mqe` v `config-som.json` bez LSTM = false.

**Phase 3 — přínos je v standalone SOM:**
Dynamický controller nahradí statické decay křivky (linear-drop, step-down…).
LSTM v každém checkpointu rozhodne na základě aktuálního konvergenčního tempa:
"zpomal pokles LR", "zmenši radius rychleji", "přepni do fine-tuning fáze".
Výsledkem je lepší výsledná kvalita SOM mapy — to je skutečný přínos pro aplikaci.

---

## Phase 2 — Early stopping prediktor

### Vstupní features — vše co checkpoint má

Dát modelu všechny dostupné informace z checkpointu dává smysl — LR a radius nesou informaci o tom, jak daleko je decay, v jaké fázi tréninku se SOM nachází a jak rychle konverguje.

**Sekvence (time-varying, N kroků × 6 dimenzí):**

| Feature | Normalizace | Zdůvodnění |
|---|---|---|
| `progress` | 0–1, hotovo | Kde v tréninku jsme — kritické pro kontext |
| `mqe / mqe[0]` | relativní pokles (≤1) | Jak rychle klesá error |
| `topographic_error` | 0–1 | Topologická kvalita průběžně |
| `dead_neuron_ratio` | 0–1 | Stav mrtvých neuronů průběžně |
| `learning_rate / lr[0]` | relativní pokles | Kde jsme na decay křivce LR |
| `radius / radius[0]` | relativní pokles | Kde jsme na decay křivce sousedství |

**Statický kontext (constant, 4 dimenze — přidány jako dense vstup vedle LSTM):**

| Feature | Popis |
|---|---|
| `ds_n_samples` | Velikost datasetu |
| `ds_n_active_dimensions` | Aktivní dimenze (nejdůležitější dle MLP feature importance) |
| `ds_n_numeric` / `ds_n_categorical` | Typy dimenzí |

Hybridní architektura: `LSTM(sequence) + Dense(context) → concat → Dense → output`.

---

### FR-LSTM-2.1 — Změna v `ea.py` ❌ *(1 řádek)*

Aktuální stav (`ea.py:1164`):
```python
history = {
    'mqe':               [c['mqe'] for c in checkpoints],
    'topographic_error': [c['topographic_error'] for c in checkpoints],
    'dead_neuron_ratio': [c['dead_neuron_ratio'] for c in checkpoints],
}
```

Potřebný stav — přidat zbývající 3 fields:
```python
history = {
    'progress':          [c['progress'] for c in checkpoints],
    'mqe':               [c['mqe'] for c in checkpoints],
    'topographic_error': [c['topographic_error'] for c in checkpoints],
    'dead_neuron_ratio': [c['dead_neuron_ratio'] for c in checkpoints],
    'learning_rate':     [c['learning_rate'] for c in checkpoints],
    'radius':            [c['radius'] for c in checkpoints],
}
```

---

### FR-LSTM-2.2 — Změna v `nn_integration.py` ❌ *(~5 řádků)*

`should_stop_early` aktuálně sestavuje `(N, 3)` matici. Rozšířit na `(N, 6)`:
```python
sequence = np.stack([
    training_history['progress'],
    [m / training_history['mqe'][0] for m in training_history['mqe']],
    training_history['topographic_error'],
    training_history['dead_neuron_ratio'],
    [lr / training_history['learning_rate'][0] for lr in training_history['learning_rate']],
    [r / training_history['radius'][0] for r in training_history['radius']],
], axis=1)  # (N, 6)
```

---

### FR-LSTM-2.3 — Příprava datasetu ❌

Skript `app/lstm/prepare_dataset.py`:

1. Načte checkpoints ze všech 5 853 individuí
2. Normalizuje sekvenci na 6 features (viz výše)
3. Resample na 200 bodů přes `progress` osu (zvládne i sekvence zkrácené early stoppingem)
4. Pro každého jedince vygeneruje K%-okna: `K ∈ {20, 30, 40, 50, 60, 70}`
5. Target z `results.csv`: `raw_mqe_improvement_ratio`, `raw_topographic_error`, `dead_neuron_ratio`
6. Pareto individuí 2× oversampled
7. Split 70/15/15 na úrovni jedince (stratifikovaný podle datasetu)

Výstup:
- `app/lstm/data/sequences_X.npy` — shape `(N_samples, seq_len, 6)`, seq_len proměnná (40–200)
- `app/lstm/data/sequences_y.npy` — shape `(N_samples, 3)`
- `app/lstm/data/sequences_context.npy` — shape `(N_samples, 4)` statický kontext
- `app/lstm/data/metadata.json`

---

### FR-LSTM-2.4 — Architektura modelu ❌

Hybridní vstup (Keras functional API):

```python
# Sekvenční vstup
seq_input = keras.Input(shape=(None, 6), name='sequence')
x = layers.LSTM(64, return_sequences=True)(seq_input)
x = layers.Dropout(0.2)(x)
lstm_out = layers.LSTM(32)(x)

# Statický kontext
ctx_input = keras.Input(shape=(4,), name='context')
ctx_out = layers.Dense(16, activation='relu')(ctx_input)

# Kombinace
combined = layers.Concatenate()([lstm_out, ctx_out])
x = layers.Dense(32, activation='relu')(combined)
x = layers.Dropout(0.2)(x)
output = layers.Dense(3, activation='linear')(x)  # mqe_ratio, te, dead

model = keras.Model(inputs=[seq_input, ctx_input], outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

Stabilní výstupní cesty (konzistentní s MLP):
- `app/lstm/models/lstm_latest.keras`
- `app/lstm/models/lstm_scaler_latest.pkl`

---

## Phase 3 — Dynamický kontroler

### Co Phase 3 dělá jinak

Phase 2 čte **celý prefix** (batch), Phase 3 musí fungovat **online** — dostává jeden checkpoint za druhým a udržuje skrytý stav (paměť). Výstupem není quality score ale **akce**: faktory pro úpravu LR a radius.

```
Checkpoint t  →  LSTM(hidden_t-1, stateful)  →  hidden_t + (lr_factor, radius_factor)
                                                         ↓
                                         SOM aplikuje: lr     *= lr_factor
                                                       radius *= radius_factor
```

Klíčový rozdíl od Phase 2: **`stateful=True`** — model si pamatuje celý průběh tréninku.

---

### Datová mezera — proč jsou potřeba nová data

Všechny existující checkpointy (5 853 individuí) mají `lr_factor=1.0, radius_factor=1.0`
vždy. LSTM kontroler se nemá z čeho naučit vztah "v tomto stavu konvergence → tato
úprava → tento výsledek". Bez perturbovaných trajektorií nelze Phase 3 natrénovat.

**Zvolen přístup: advantage-weighted imitation learning** (behavioral cloning s quality signálem)

```
Pro každý perturbed run R:
  advantage = mqe_baseline − mqe_R    # kladné = R je lepší než bez perturbace

Pro každý checkpoint t v R:
  X[t] = stav konvergence (6 features)
  y[t] = (lr_factor_t, radius_factor_t)

loss = advantage × MSE(predicted, actual)
```

Kladný advantage → tlak k replikaci těchto faktorů.
Záporný advantage → tlak pryč od těchto faktorů.
Nepotřebuje RL infrastrukturu — čistý supervised trénink.

---

### Implementační plán — Část 1 (smoke test)

~15 SOM běhů, 1 dataset (LungCancer), 3 Pareto individua × 5 variant.

| Soubor | Co dělá | Stav |
|---|---|---|
| `som.py` | `dynamic_schedule_fn` callback hook, kumulativní faktory, clipping | ✅ |
| `app/lstm/generate_phase3_data.py` | 45 SOM běhů (5 Pareto × 9), 14 013 checkpointů | ✅ |
| `app/lstm/src/model_controller.py` | Stateful (inference) + trainable LSTM architektura | ✅ |
| `app/lstm/prepare_phase3_dataset.py` | Advantage-weighted dataset, split na uid úrovni | ✅ |
| `app/lstm/src/train_phase3.py` | Trénink, stable cesty, test MAE=0.041 | ✅ |
| `app/ea/nn_integration.py` | `get_dynamic_schedule_fn()`, controller load | ✅ |
| `app/ea/ea.py` | Wiring do `som.train(dynamic_schedule_fn=...)` | ✅ |
| `config-ea.json` | `use_lstm_controller` flag (default: false) | ✅ |

Ověření: ztráta klesla (MAE 0.048→0.041), predikce rozlišují trajektorie ✅

### Implementační plán — Část 2 (pořádný trénink)

~200–400 SOM běhů, 4 datasety, 5–10 Pareto individuí × 10 variant.

| Rozšíření | Popis |
|---|---|
| Všechny 4 datasety | BreastCancer, IndianLiver, LungCancer, Pima |
| Elbow features | Přidat d1, d2 MQE jako extra dimenze sekvence |
| Více variant | 10 perturbačních semen na individuum |
| Validace | Srovnávací EA běh: statický schedule vs. LSTM kontroler |

---

### Závislosti Phase 3

| Co je potřeba | Kde | Složitost |
|---|---|---|
| `dynamic_schedule_fn` callback | `som.py` | Nízká |
| Data generation script | `generate_phase3_data.py` | Nízká |
| Stateful LSTM model | `model_controller.py` | Střední |
| Training pipeline | `prepare_phase3_dataset.py` + `train_phase3.py` | Střední |
| Rozšíření na všechny datasety (Část 2) | `generate_phase3_data.py` + čas | Střední (čas) |
| RL trénink (alternativa, pokud imitation selže) | `train_rl.py` | Vysoká |

---

## Matice požadavků

| ID | Popis | Fáze | Stav |
|---|---|---|---|
| FR-LSTM-2.1 | Přidat lr/radius/progress do `lstm_early_stop_fn` v `ea.py` | 2 | ✅ |
| FR-LSTM-2.2 | Rozšíření `should_stop_early` na 6-dim vstup | 2 | ✅ |
| FR-LSTM-2.3 | `prepare_dataset.py` — sekvence + kontext + K%-okna | 2 | ✅ |
| FR-LSTM-2.4 | Hybridní LSTM model (sekvenční + statický vstup) | 2 | ✅ |
| FR-LSTM-2.5 | `train.py` — trénink, stabilní výstupní cesty | 2 | ✅ |
| FR-LSTM-2.6 | Ověření early stopping v EA běhu | 2 | ❌ |
| FR-LSTM-3.1 | `dynamic_schedule_fn` callback v `som.py` | 3 | ✅ |
| FR-LSTM-3.2 | `generate_phase3_data.py` — 45 perturbovaných SOM běhů (LungCancer) | 3 | ✅ |
| FR-LSTM-3.3 | `model_controller.py` — stateful + trainable LSTM, výstup `(lr_f, radius_f)` | 3 | ✅ |
| FR-LSTM-3.4 | `prepare_phase3_dataset.py` — advantage-weighted tréninkový dataset | 3 | ✅ |
| FR-LSTM-3.5 | `train_phase3.py` — trénink, test MAE=0.041, stable cesty | 3 | ✅ |
| FR-LSTM-3.6 | Rozšíření dat na 4 datasety, 200–400 běhů (Část 2) | 3 | ❌ |
| FR-LSTM-3.7 | Elbow features (d1, d2 MQE) jako extra dimenze sekvence | 3 | ❌ |
| FR-LSTM-3.8 | Srovnávací EA validace: statický schedule vs. LSTM kontroler | 3 | ❌ |

**Předpoklady Phase 3:** FR-LSTM-2.1–2.5 jsou splněny. FR-LSTM-2.6 (ověření v EA)
je nezávislé — Phase 3 implementace může probíhat paralelně.
