# LSTM — Požadavky a roadmapa

**Verze**: 3.1  
**Aktualizováno**: 2026-05-06  
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

### Chybí ❌

- ❌ FR-LSTM-2.6: Ověření early stopping v EA běhu
- ❌ `lstm_scaler_path` expozice v `config-ea.json` (volitelné — auto-detect funguje)

---

## Přehled dvou fází

```
Phase 2 (nyní)          Phase 3 (budoucnost)
──────────────────────  ──────────────────────────────────
Vstup: checkpoints[0..K]  Vstup: checkpoint[t] (streamově)
Výstup: quality_score     Výstup: lr_factor, radius_factor, stop
Účel: early stopping      Účel: dynamická úprava decay curves
Trénink: supervised       Trénink: RL nebo behavioral cloning
Data: existují ✅          Data: CHYBÍ ❌ (viz sekce níže)
```

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

Phase 2 čte **celý prefix** (batch), Phase 3 musí fungovat **online** — dostává jeden checkpoint za druhým a udržuje skrytý stav (paměť). Výstupem není quality score ale **akce**: faktory pro úpravu LR a radius pro příštích ~N/300 iterací.

```
Checkpoint t  →  LSTM(hidden_t-1)  →  hidden_t + (lr_factor, radius_factor, stop)
                                          ↓
                              SOM aplikuje: lr = schedule_lr × lr_factor
                                            radius = schedule_radius × radius_factor
```

---

### Změny v SOM pro Phase 3

**`som.py` — training loop** (aktuálně `som.py:362`):

```python
# Aktuální (Phase 2):
if lstm_early_stop_fn is not None and len(checkpoints) >= 2:
    should_stop, lstm_score = lstm_early_stop_fn(checkpoints)

# Phase 3 — rozšíření callback interface:
if lstm_controller_fn is not None and len(checkpoints) >= 2:
    should_stop, lstm_score, lr_factor, radius_factor = lstm_controller_fn(checkpoints[-1])
    current_lr    *= lr_factor     # modifikace pro příštích mqe_compute_interval iterací
    current_radius *= radius_factor
```

Zpětná kompatibilita: pokud `lstm_controller_fn is None`, chování beze změny.

**Nový parametr `som.train()`:**
```python
def train(self, data, ignore_mask=None, working_dir='.', 
          lstm_early_stop_fn=None,      # Phase 2 — zůstává
          lstm_controller_fn=None):     # Phase 3 — nový
```

---

### Změny v `ea.py` pro Phase 3

```python
# Phase 2 callback (zůstává):
lstm_early_stop_fn = None
if nn.can_check_early_stopping():
    def lstm_early_stop_fn(checkpoints): ...

# Phase 3 callback (nový):
lstm_controller_fn = None
if nn.can_control_training():  # nová metoda v nn_integration.py
    def lstm_controller_fn(checkpoint): ...
```

---

### Fundamentální datová mezera Phase 3

**Problém**: Všechny existující běhy EA používají statický decay schedule. LR a radius se mění deterministicky podle konfigurace — nikdy nebyly adjustovány mid-training. Neexistují tedy trénovací příklady "v bodě X jsem upravil LR, výsledek byl lepší/horší."

Pro trénink Phase 3 kontroleru jsou potřeba nová data. Dvě schůdné cesty:

#### Cesta A — Behavioral cloning z variabilních schedule běhů

Spustit novou sadu EA běhů kde se Pareto individuím záměrně variují decay křivky (různé growth_g, různé decay typy). LSTM se pak naučí: "při tomto stavu konvergence dosáhl rychlejší/pomalejší decay takového výsledku."

Výhoda: jednoduchý supervised training, nepotřebuje RL infrastrukturu.  
Nevýhoda: vyžaduje ~500–1000 extra EA runs s cílenou variací.

#### Cesta B — Reinforcement Learning (online)

RL trénink kde agent (LSTM) přímo spouští SOM epizody, dostává reward = `raw_mqe_improvement_ratio` po skončení tréninku, a učí se politiku která maximalizuje reward.

```
Agent → akce (lr_factor, radius_factor) → SOM trénuje N iterací → reward
```

Výhoda: nepotřebuje žádná extra data, naučí se obecnou politiku.  
Nevýhoda: každá epizoda = jeden SOM trénink (~minuty), trénink agenta vyžaduje tisíce epizod → dny výpočtu.

#### Doporučení

Cesta A — behavioral cloning — je realistická jako první krok a staví na existující EA infrastruktuře. Stačí přidat `growth_g` a `lr_decay_type` variace do nové EA kampaně na 2–3 datasetech a sesbírat ~2 000 individuí s variabilními křivkami.

---

### Souhrn závislostí Phase 3

| Co je potřeba | Kde | Složitost |
|---|---|---|
| Rozšíření callback rozhraní `(stop, score, lr_f, radius_f)` | `som.py`, `ea.py`, `nn_integration.py` | Nízká |
| Stateful LSTM model (online, jeden checkpoint za druhým) | `app/lstm/src/model_controller.py` | Střední |
| Nová trénovací data s variabilními schedules | EA kampaň | Střední (čas) |
| RL trénink loop | `app/lstm/src/train_rl.py` | Vysoká |

Phase 3 je realisticky 2–3 iterace EA kampaní po Phase 2 — až bude prediktor funkční a model prokáže, že rozumí dynamice konvergence.

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
| FR-LSTM-3.1 | Rozšíření callback na `(stop, score, lr_f, radius_f)` | 3 | ❌ |
| FR-LSTM-3.2 | `lstm_controller_fn` v `ea.py` a `som.py` | 3 | ❌ |
| FR-LSTM-3.3 | Stateful LSTM kontroler (online inference) | 3 | ❌ |
| FR-LSTM-3.4 | Trénovací data s variabilními schedules (EA kampaň) | 3 | ❌ |
| FR-LSTM-3.5 | Trénink kontroleru (behavioral cloning nebo RL) | 3 | ❌ |

**Blokery Phase 3 odstraněny po Phase 2** — `should_stop_early` rozhraní + funkční prediktor jsou předpokladem pro kontroler.
