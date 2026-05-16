# NexusSOM — Komponenty systému

**Verze**: 2.0  
**Datum**: 2026-05-15

---

## Vrstva 1 — Jádro (implementováno)

### 1. Self-Organizing Map (SOM)

**Role**: Primární analytický engine pro exploraci a vizualizaci dat.

#### Statický režim ✅
- Hybridní trénink: progresivně větší batche z různých sekcí datasetu
- Decay křivky: linear, exp, log, step-down pro LR i radius
- Metriky: MQE, Topographic Error, Dead Neuron Ratio
- Checkpointy každý MQE výpočet (pro LSTM trénink)
- Výstupy: váhy `.npy`, vizualizace (U-Matrix, Distance Map, Dead Neurons Map)
- Reprodukovatelnost: fixní random seed

#### Dynamický režim (LSTM controller) ✅
- Multiplicativní kumulativní řízení: `current_lr = static_schedule(t) × cum_lr_factor`
- Controller zasahuje každý checkpoint, faktory v rozsahu [0.5, 1.5]
- Logování: milníky každých 10 % + intervence při odchylce > 1 %
- Batch parametry zůstávají statické (řízeno EA)

---

### 2. Evoluční algoritmus (EA)

**Role**: Autonomní optimalizace hyperparametrů SOM, generování trénovacích dat.

#### Algoritmus ✅
- **NSGA-II** s constrained dominance (Deb 2002)
- **Objectives** (minimalizace): `[raw_mqe_ratio, topo_error, dead_ratio]`
  - `raw_mqe_ratio = final_mqe / initial_mqe` — čistá konvergence bez penalizace
- **Genetic operators**: SBX crossover + polynomiální mutace, log-scale pro LR/radius
- **Archiv**: rank==0 jedinci, pruning přes crowding distance při překročení `max_archive_size`
- **Constraint violation**: graduated pásma 1.5/2.5/5.0 — infeasible nekazí raw objectives

#### Inicializace a kalibrace ✅
- **Kalibrační sonda**: N rychlých SOM běhů před G0 → 70. percentil `max(u_matrix_max, dist_map_max)` jako `org_threshold`
- **Vesanto heuristic**: `optimal_side = sqrt(5 × sqrt(n_samples))` → dynamický search space pro `map_size`
- **Dynamický `epoch_multiplier`**: cílový počet iterací [3 000, 20 000] / n_samples
- **Multi-seed strategie**: každý seed exploruje prostor nezávisle, resetuje ARCHIVE a running stats

#### Pareto metriky ✅
- **HV** (pymoo, ref=[1.1,1.1,1.1] v normalizovaném prostoru)
- **Spacing** (nearest-neighbor v 3D)
- **Maximum Spread** per dimenzi
- Normalizace: globální running min/max per seed
- Výstup: `pareto_metrics.csv` (1 řádek/generace) + `pareto_front.csv` (1 řádek/řešení)

#### NN integrace ✅
- `use_mlp: true` — MLP pre-screen filtr (přeskočí predikované slabé konfigurace)
- `use_lstm: true` — LSTM early stopping (zastaví špatný běh po K% tréninku)
- `use_lstm_controller: true` — Phase 3 dynamické řízení LR a R

---

## Vrstva 2 — AI modely (implementovány, trénink probíhá)

### 3. MLP "The Oracle" 🔮 ✅

**Role**: Pre-screen filtr + doporučení počátečních hyperparametrů z charakteristik datasetu.

**Vstupy**: dataset statistiky (`ds_n_samples`, `ds_n_features`, `ds_n_categorical`, ...)  
**Výstupy**: predikce `raw_mqe_ratio`, `topo_error`, `dead_ratio` pro danou konfiguraci

**Architektura**: Dense MLP, vstup = dataset stats + hyperparametry, výstup = 3 metriky kvality

**Dva režimy použití v EA**:
1. **Pre-screen**: `pred_mqe_ratio < threshold` → přeskočit evaluaci (ušetřit čas)
2. **Doporučení**: injekce dobrého startovního bodu do počáteční populace

**Tréninkový pipeline**:
```
app/mlp/prepare_dataset.py --results_root <path>
app/mlp/src/train.py
app/mlp/visualize_model.py
```

**Stav**: Pipeline implementována, první trénovací data k dispozici (BreastCancer seed_42, 87 feasible).

---

### 4. LSTM Early Stopping (Phase 2) 🧠 ✅

**Role**: Předpověď finální kvality SOM z prefixu trénovací sekvence — zastaví špatné běhy.

**Vstup**: normalizovaná sekvence checkpointů (20–70 % délky), 200-bodový resample  
**Výstupy**: predikce `final_mqe_ratio`, `final_topo_error`, `final_dead_ratio`

**Architektura**: LSTM(64) + LSTM(32) + context Dense(16) → 3 výstupy  
**Trénování**: K-prefix windowing při K ∈ {20,30,...,70}%, 18k oken z 149 běhů

**Stopping logika**: `quality_score = (1 - mqe_ratio) + te + dead×0.5`; stop pokud `score < threshold (0.75)`  
**Práh aktivace**: min. 20 % tréninku (= `mqe_evaluations_per_run / 5` checkpointů)

**Stav**: Architektura implementována, model natrénován na LungCancer datech (MAE=0.023).

---

### 5. LSTM Controller (Phase 3) 🧠 ✅

**Role**: Dynamické řízení LR a radius faktoru v průběhu SOM tréninku.

**Vstup**: sekvence checkpointů (progress, mqe, topo, dead, lr, radius) + kontext datasetu  
**Výstupy**: `lr_factor`, `radius_factor` ∈ [0.5, 1.5] per checkpoint

**Architektura**: LSTM + `_TileContext` (tiling kontextu) + `_ScaleSigmoid` (škálování výstupu)  
**Trénování**: perturbační trajektorie z Pareto konfigurací (PERTURB_PROB=0.4, ±25 %)

**Kumulativní mechanismus**: `cum_factor *= step_factor` — malé kroky se násobí (0.998^300 ≈ 0.55)

**Stav**: Architektura implementována, základní model natrénován. Pro Phase 1 se sbírají trénovací data.

---

### 6. CNN "The Eye" 👁️ ⏸ Uzavřeno

**Role**: Vizuální hodnocení kvality SOM map.

**Stav**: Implementace existuje (`app/cnn/`), ale CNN jako primární fitness signál byl nahrazen přístupem MLP+LSTM na raw metrikách. CNN objective je v EA kódu volitelné (`use_cnn: true`), ale není součástí aktivního tréninkového plánu.

Infrastruktura (RGB kombinování map, `maps_dataset/rgb/`) zůstává funkční.

---

## Vrstva 3 — Aplikační vrstva (implementováno)

### 7. Analytický modul 🔍 ✅

**Role**: Automatické statistiky a detekce anomálií na naučené SOM mapě.

**Výstupy**:
- Globální statistiky per feature (min/max/mean/std/percentily)
- Distribuce kategorií per neuron
- Z-score odchylky clusterů od globálního průměru
- Lokální outlieři (Z-score > 2.5σ od centroidu clusteru)
- U-matrix, component planes, pokrytí mapy

```
app/analysis/src/
├── loader.py    ← IO
├── stats.py     ← čisté statistiky
├── anomalies.py ← detekce anomálií
└── context.py   ← sestavení llm_context.json
```

---

### 8. LLM "The Voice" 🗣️ ✅

**Role**: Překlad technických výstupů SOM do lidského jazyka.

**Módy**: `report` (jednorázový full report) + `chat` (interaktivní Q&A)  
**Vstup**: `llm_context.json` + `dataset_context.txt`  
**Výstup**: `llm/report.md` + `prompt_log.json`

---

## Vrstva 4 — Budoucí vývoj

| Komponenta | Stav | Poznámka |
|---|---|---|
| **UI** | 🔜 | React/D3.js vizualizace, upload dat |
| **Databáze** | 🔜 | PostgreSQL metadata + S3 artefakty |
| **Growing SOM** | 🔜 | Dynamická topologie, vkládání/mazání neuronů |

---

## Přehled stavu komponent

| Komponenta | Stav | Priorita |
|---|---|---|
| SOM (statický) | ✅ Hotovo | P1 |
| SOM (dynamický — LSTM controller) | ✅ Hotovo | P1 |
| EA (NSGA-II + kalibrace + HV) | ✅ Hotovo | P1 |
| MLP (Oracle) | ✅ Pipeline hotova, trénink čeká na data | P1 |
| LSTM Early Stopping (Phase 2) | ✅ Hotovo | P1 |
| LSTM Controller (Phase 3) | ✅ Hotovo, rozšiřuje se dataset | P1 |
| Analytický modul | ✅ Hotovo | P2 |
| LLM (Voice) | ✅ Hotovo | P2 |
| CNN (Eye) | ⏸ Uzavřeno | — |
| UI | 🔜 Plánováno | P3 |
| Databáze | 🔜 Plánováno | P3 |
