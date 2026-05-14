# NexusSOM — Plán trénování a testování

**Verze**: 1.0  
**Datum**: 2026-05-14  
**Stav**: Aktivní plán

---

## Architektura systému — co kam patří

### Role jednotlivých komponent

| Komponenta | Role | Vstupy | Výstupy |
|---|---|---|---|
| **Vesanto heuristika** | Určuje `map_size` z dat | `n_samples` | `map_m`, `map_n` |
| **Dynamic search space** | Určuje `epoch_multiplier` z dat | `n_samples` | rozsah EM |
| **MLP** | Doporučí počáteční hyperparametry SOM | statistiky datasetu | `start_lr`, `start_radius`, (batch params) |
| **LSTM early stopping** | Zastaví špatný SOM běh po K% tréninku | prefix sekvence checkpointů | stop / continue |
| **LSTM controller (Phase 3)** | Dynamicky řídí LR a R v průběhu tréninku | checkpoint sekvence + kontext | `lr_factor`, `radius_factor` |
| **Analýza mapy** | Kvantitativní popis naučené mapy pro LLM | naučená SOM | U-matrix, QE, component planes, dead neurons |

### Pevně fixované parametry (výsledek analýzy permutační důležitosti)

| Parametr | Hodnota | Odůvodnění |
|---|---|---|
| `num_batches` | fixní (8–10) | Ochrana před stochastickým výběrem stejných vzorků — bez dostatku batchů může SOM trénovat opakovaně na jednom vstupním vektoru; rozdělení datasetu zajišťuje rovnoměrné pokrytí |
| `lr_decay_type` | dle výsledků EA | Permutační důležitost ≈ 0; typ tvaru křivky nemá vliv na kvalitu; fixujeme po analýze výsledků |
| `radius_decay_type` | dle výsledků EA | Stejná logika jako LR decay |
| `growth_g` | dle výsledků EA | Součást decay křivky; fixujeme spolu s typem |
| `end_learning_rate` | dle výsledků EA | Konečná hodnota — LSTM controller adaptuje průběh |
| `normalize_weights_flag` | `false` | 100 % penalizovaných běhů s `true` |

**Poznámka k decay**: Typ decay křivky (linear/exp/log/step-down) se fixuje až po analýze výsledků EA tréninkových běhů — empiricky, ne intuitivně. Pro tréninkové EA běhy zůstává v search space.

### Zbývající EA search space (po fixaci)
- `map_size` — Vesanto (dynamicky)
- `epoch_multiplier` — dynamicky z `n_samples`
- `start_learning_rate`
- `start_radius_init_ratio`
- `start_batch_percent`, `end_batch_percent`, `batch_growth_type`

---

## Fáze 1 — Sběr dat: malý dataset

**Cíl**: Ověřit pipeline end-to-end, získat první trénovací data pro MLP a LSTM.

### Konfigurace EA běhů

```
6 seedů × 5 generací × 30 populace = 900 evaluací
use_lstm_controller: false  (controller ještě není natrénován)
use_lstm: false             (early stop vypnutý — potřebujeme plné trajektorie)
use_mlp: false              (MLP ještě není natrénován na nových datech)
```

Dataset: jeden malý reálný dataset (< 2 000 vzorků).

### Po EA běhu — příprava Phase 3 dat

```bash
# Perturbační varianty z Pareto konfigurací
python3 app/lstm/generate_phase3_data.py \
    --seed_dir data/results/<DS>/seed_42 \
    --dataset  data/datasets/<DS>/dataset.csv \
    --n_pareto 5 --n_variants 8

# Dataset příprava
python3 app/lstm/prepare_phase3_dataset.py ...
```

Perturbace: náhodné změny `lr_factor` a `radius_factor` v MQE checkpointech (PERTURB_PROB=0.4, ±25 %). Tím vznikají trénovací příklady: "v tomto stavu konvergence jsem změnil LR/R o X — výsledek byl Y".

### Trénování modelů

```bash
# MLP (predikce počátečních hyperparametrů z dataset stats)
python3 app/mlp/prepare_dataset.py --results_root data/results
cd app/mlp && python3 src/train.py

# LSTM early stopping (Phase 2)
python3 app/lstm/prepare_dataset.py --results_root data/results
cd app/lstm && python3 src/train.py

# LSTM controller (Phase 3)
python3 app/lstm/prepare_phase3_dataset.py ...
cd app/lstm && python3 src/train_phase3.py
```

### Výstupy Fáze 1

- Natrénované modely: `mlp_latest.keras`, `lstm_latest.keras`, `lstm_controller_latest.keras`
- Statistiky modelů (MAE, RMSE, Pearson r, confusion matrix early stop)
- Vizualizace: `app/mlp/visualize_model.py`, `app/lstm/visualize_model.py`
- **Baseline reference** pro srovnání s pozdějšími fázemi

---

## Fáze 2 — Rozšíření na 5 reálných datasetů

**Cíl**: Ověřit zda modely generalizují přes různé datasety; vizualizovat křivku učení.

### Konfigurace

```
5 datasetů × 6 seedů × 5 generací × 30 populace = 4 500 evaluací
Stejná EA konfigurace jako Fáze 1
```

### Trénování

Přetrénovat všechny tři modely na **kumulativních datech** (Fáze 1 + Fáze 2 dohromady).

### Srovnání

| Metrika | Model Fáze 1 (1 DS) | Model Fáze 2 (5 DS) | Δ |
|---|---|---|---|
| MLP MAE (mqe) | ? | ? | ? |
| LSTM MAE (mqe) | ? | ? | ? |
| Controller MAE | ? | ? | ? |
| Early stop Acc | ? | ? | ? |

Cíl: ukázat že s více daty modely generalizují lépe — viditelná křivka učení.

---

## Fáze 3 — Rozšíření na 10 reálných datasetů

**Cíl**: Finální reálná sada dat; konvergence modelu.

### Konfigurace

```
5 nových datasetů × 6 seedů × 5 generací × 30 populace = 4 500 evaluací
Celkem s Fází 1+2: ~9 900 evaluací, 10 datasetů
```

### Trénování a srovnání

Stejný postup jako Fáze 2. Srovnávací tabulka: 1 DS → 5 DS → 10 DS.

---

## Fáze 4 — Generované datasety *(odloženo)*

> **Stav**: Nástroj na generování virtuálních datasetů existuje, ale tato fáze se řeší až "máme spoustu času". Fáze 5–7 pokračují přímo z reálných datasetů (Fáze 3).

---

## Fáze 5 — EA s dynamickým řízením

**Cíl**: Ověřit zda aktivní LSTM controller zlepšuje kvalitu výsledků EA.

### Konfigurace

```
use_lstm_controller: true   (model natrénovaný v Fázi 1–3)
use_lstm: true              (early stopping zapnutý)
use_mlp: true               (pre-screen filtr)
```

Srovnávací běh na stejných datasetech jako Fáze 3:
- **Větev A**: statický schedule (use_lstm_controller: false)
- **Větev B**: dynamicky řízený schedule (use_lstm_controller: true)

### Metriky srovnání

| Metrika | Větev A (statická) | Větev B (dynamická) | Δ |
|---|---|---|---|
| Pareto HV (finální gen) | ? | ? | ? |
| Pareto Spacing (finální gen) | ? | ? | ? |
| HV křivka (per-gen) | ? | ? | Δ trend |
| Průměrný `raw_mqe_improvement_ratio` | ? | ? | ? |
| Průměrný `topographic_error` | ? | ? | ? |
| Počet feasible řešení | ? | ? | ? |
| Průměrná doba evaluace | ? | ? | ? |
| Počet MLP skipped | ? | ? | ? |
| Počet LSTM early stop | ? | ? | ? |

### Přetrénování na dynamicky řízených datech?

Pokud větev B produkuje dostatečně odlišné trajektorie (controller skutečně zasahuje), stojí za to přetrénovat modely na těchto datech a srovnat:
- Generalizace: zlepšila se přesnost predikce?
- Distribuce shift: jsou nové trajektorie sémanticky jiné?

Přetrénování provedeme ale výsledky bereme s rezervou — controller mění distribuci dat, na které se trénuje.

---

## Fáze 6 — Finální srovnání SOM módů

**Cíl**: Pro každý dataset porovnat 4 způsoby trénování SOM.

### Čtyři módů

| Mód | Popis | Konfigurace |
|---|---|---|
| **Stochastický** | Čistě náhodný výběr parametrů, bez opory o data | žádné NN, náhodný config |
| **Deterministický (statický)** | Pevné decay křivky, žádné dynamické řízení | fixní config, bez NN |
| **Hybridní — dynamicky řízený** | MLP doporučí start params, LSTM controller řídí průběh | use_mlp + use_lstm_controller |
| **Statický podle křivek** | Optimální start params z MLP, ale statický schedule | use_mlp, bez controlleru |

### Co porovnáváme

Pro každý dataset a každý mód:
- Finální MQE improvement ratio
- Topographic error
- Dead neuron ratio
- Qualita organizace mapy (U-matrix, component planes)
- Výstup analýzy pro LLM (jak bohatý kontext)
- Doba tréninku

Vizualizace: side-by-side U-matrix, component planes, hustota BMU mapování.

---

## Fáze 7 — Anomaly detection test

**Cíl**: Ověřit že SOM analýza detekuje záměrně zanesenou anomálii v datasetu.

### Typy záměrných chyb

| Typ chyby | Jak se projeví v SOM analýze |
|---|---|
| **Outlier cluster** — skupina vzorků vzdálená od distribuce | Izolovaný region s vysokou U-matrix hodnotou; neurony s vysokou QE |
| **Šum v jednom feature** — náhodné hodnoty v jednom sloupci | Component plane pro daný feature bude bez struktury, ostatní zachovají vzory |
| **Systematická bias** — posunutá distribuce jedné skupiny | Asymetrická hustota BMU mapování; nerovnoměrné pokrytí mapy |
| **Chybějící hodnoty / label corruption** — záměrně chybné labely | Neurony s neočekávanou kombinací features; vyšší dead ratio v postižené oblasti |

### Postup testu

1. Vezmi existující dataset s dobře naučenou mapou (výsledek Fáze 3)
2. Zanes konkrétní typ chyby do kopie datasetu
3. Natrénuj SOM stejnou konfigurací na poškozených datech
4. Porovnej analýzu:
   - Čistá data → LLM kontext
   - Poškozená data → LLM kontext
5. Ověř zda LLM (nebo přímo analýza) signalizuje anomálii

### Implementační poznámka

Detekce není nová feature — analýza mapy již počítá:
- Quantization error per neuron (lokální anomálie)
- U-matrix hodnoty (globální hranice a izolované clustery)
- Component planes (feature-level distribuce)
- Dead neurons a pokrytí mapy

Záměrná chyba by měla být viditelná v těchto existujících metrikách bez nutnosti nové implementace. Test ověří, zda je signál dostatečně silný.

---

## Souhrn pořadí fází

```
Fáze 1  Malý dataset (1)     → EA sběr + perturbace + train MLP/LSTM/Controller
           ↓
Fáze 2  5 reálných DS        → EA sběr + přetrénování → srovnání 1 vs 5 DS
           ↓
Fáze 3  10 reálných DS       → EA sběr + přetrénování → srovnání 1→5→10 DS
           ↓
Fáze 4  Generované DS        → ODLOŽENO (nástroj existuje, fáze "máme čas")
           ↓
Fáze 5  EA s controllerem    → srovnání statický vs dynamický schedule
           ↓
Fáze 6  SOM módů srovnání    → stochastický / statický / dynamický / hybridní
           ↓
Fáze 7  Anomaly detection    → dataset se zanesenou chybou
```

---

## Otevřené otázky

| # | Otázka | Stav |
|---|---|---|
| 1 | Který decay type fixovat pro produkci? | ❓ Čeká na analýzu výsledků EA Fáze 1 |
| 2 | Generované datasety — nástroj existuje, fáze odložena | ⏸ Odloženo |
| 3 | Hypervolume výpočet pro Pareto srovnání — implementován? | ✅ Implementováno: `_compute_pareto_metrics()` v `ea.py` — HV (pymoo, ref=[1.1,1.1,1.1]) + Spacing per-gen, výstup do `pareto_metrics.csv` |
| 4 | LLM integrace — jak bohatý kontext pro anomaly test? | ❓ Závisí na analýza implementaci |
| 5 | Přetrénování na dynamicky řízených datech (Fáze 5) — jak hodnotit distribuci shift? | ❓ KL divergence trajektorií |
