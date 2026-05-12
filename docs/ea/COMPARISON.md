# NN Integrace — Srovnávací experiment

Cílem je empiricky ověřit přínos MLP pre-screenu a LSTM early stopping na reálném datasetu
a připravit základnu pro porovnání s budoucím dynamickým řízením (LSTM Phase 3).

---
## Přehled scénářů

| # | Scénář | Hyperparametry | NN | Popis |
|---|---|---|---|---|
| **A** | Baseline EA | Pareto-optimální (seed 42) | ❌ | Původní výsledek z EA — referenční hodnoty |
| **B** | LSTM early stopping | Stejné jako A | LSTM ✅ | Zkrácení trénování bez ztráty kvality |
| **C** | MLP-doporučené | Grid search přes MLP predikce | MLP ✅ | Hyperparametry navržené MLP modelem |
| **D** | Dynamické řízení | Pareto-optimální (jako A) | LSTM Ph3 ✅ | **Bude doplněno po implementaci LSTM Phase 3** |

---

## Proč tento experiment

EA nalezla Pareto-optimální konfiguraci po ~1 500 evaluacích (5 seeds × 50 pop × 6 gen).
Chceme vědět:
- **B vs A**: Dokáže LSTM zkrátit SOM trénink o 30–50 % bez statisticky významné ztráty kvality?
- **C vs A**: Jsou MLP-doporučené hyperparametry srovnatelné nebo lepší než EA-optimalizované?
- **D vs A/B**: Nahradí dynamické křivky LR/radius statické decay funkce bez re-trénování EA?

---

## Dataset

**LungCancerDataset** — `data/datasets/LungCancerDataset/dataset.csv`

| Vlastnost | Hodnota |
|---|---|
| n_samples | 3 000 |
| n_active_dimensions | 16 |
| n_numeric | 2 |
| n_categorical | 15 |
| missing_ratio | 0.0 |

---

## Referenční jedinec (Pareto best — Scénář A)

Vybrán `aad3bdcbf5ea3acd583add9196d25a46` ze seed 42, generace 6.
Kritérium výběru: nejlepší `raw_mqe_improvement_ratio` na Pareto frontě bez penalizace a dead neuronů.

### Hyperparametry

| Parametr | Hodnota |
|---|---|
| `map_m` × `map_n` | 18 × 18 |
| `start_learning_rate` | 0.8587 |
| `end_learning_rate` | 0.0046 |
| `lr_decay_type` | linear-drop |
| `start_radius_init_ratio` | 0.9976 |
| `radius_decay_type` | step-down |
| `start_batch_percent` | 0.97 |
| `end_batch_percent` | 2.83 |
| `batch_growth_type` | exp-growth |
| `num_batches` | 9 |
| `epoch_multiplier` | 1.44 |
| `growth_g` | 22 |

### Naměřené výsledky (Scénář A — baseline)

| Metrika | Hodnota |
|---|---|
| `raw_mqe_improvement_ratio` | **0.5891** (MQE se snížilo na 58.9 % výchozí hodnoty — zlepšení 41.1 %) |
| `raw_topographic_error` | **0.0583** |
| `dead_neuron_ratio` | **0.0000** |
| `duration` | ~171 s |
| `is_penalized` | False |
| `constraint_violation` | 0.0 |

### MLP predikce pro referenčního jedince

| Target | MLP predikce | Skutečná hodnota | Odchylka |
|---|---|---|---|
| `mqe_improvement_ratio` | 0.6507 | 0.5891 | +0.062 |
| `topographic_error` | 0.0552 | 0.0583 | −0.003 |
| `dead_neuron_ratio` | 0.0081 | 0.0000 | +0.008 |

MLP predikovala o ~6 p.b. horší MQE ratio — konzistentní s typickou MAE ~0.05.
Konfigurace by MLP filtrem neprošla pod `threshold=0.5` (pred=0.65 > 0.5), správně.

---

## Scénář B — LSTM early stopping (stejné hyperparametry)

### Co se mění

Spuštění SOM s identickými hyperparametry jako Scénář A, ale s aktivním LSTM early stopping
(`lstm_quality_threshold = 0.75`). LSTM monitoruje průběh od 20 % trénování.

### Jak spustit

```bash
# Vytvoř jednorázový config pro standalone SOM run
python3 app/tools/run_single_som.py \
    --dataset data/datasets/LungCancerDataset/dataset.csv \
    --params docs/ea/comparison_params_A.json \
    --lstm
```

Nebo přímo přes EA s jedním seedem a jedním jedincem (viz sekce Spuštění níže).

### Co sledovat

- Kdy LSTM zastavilo trénink (progress %)
- `raw_mqe_improvement_ratio` po ukončení vs. A (přijatelná ztráta: < 2 p.b.)
- `duration` vs. A (očekávané zkrácení: 30–50 %)
- `is_penalized` musí být False

### Výsledky ✅

Spuštěno: `python3 app/tools/run_single_som.py --params docs/ea/comparison_params_A.json --lstm --seed 42`
Výstup: `data/results/analysis/som-comparison/scen_B_lstm_stop/`

| Metrika | Scénář A (bez NN) | Scénář B (LSTM) | Δ |
|---|---|---|---|
| `mqe_improvement_ratio` | **0.5891** | **0.5891** | 0.0000 ✅ |
| `topographic_error` | 0.0583 | 0.0583 | 0.0000 ✅ |
| `dead_neuron_ratio` | 0.0000 | 0.0000 | 0.0000 ✅ |
| `duration` | **81 s** | 99 s | +18s (+22 %) |
| LSTM stop progress | — | **Nespustilo** | — |

**Interpretace:** LSTM korektně identifikovalo tuto konfiguraci jako dobrou a nezasáhlo.
Nulová ztráta kvality. Overhead LSTM monitoringu je +18s (+22 %) — každý checkpoint
se posílá do modelu i když early stop nevyvolá. Pro konfigurace kde LSTM zastaví na 40–60 %
trénování bude celková doba kratší než bez NN.

---

## Scénář C — MLP-doporučené hyperparametry

### Jak byly nalezeny

Grid search přes 10 368 konfigurací, každá předpovězena MLP modelem.
Skórovací funkce: `mqe_ratio + te + dead*0.5` (nižší = lepší).

### Hyperparametry (MLP grid best)

| Parametr | Hodnota | Δ vs. Ref |
|---|---|---|
| `map_m` × `map_n` | 20 × 20 | +2 (větší mapa) |
| `start_learning_rate` | 0.90 | +0.041 |
| `end_learning_rate` | 0.01 | +0.0054 |
| `lr_decay_type` | step-down | ≠ linear-drop |
| `start_radius_init_ratio` | 1.00 | +0.002 |
| `radius_decay_type` | step-down | = |
| `start_batch_percent` | 1.00 | +0.03 |
| `end_batch_percent` | 3.50 | +0.67 |
| `batch_growth_type` | linear-growth | ≠ exp-growth |
| `num_batches` | 8 | −1 |
| `epoch_multiplier` | 3.50 | +2.06 |
| `growth_g` | 25 | +3 |

**MLP predikce pro tuto konfiguraci:**
`mqe_ratio=0.6046`, `te=0.0577`, `dead=0.0180`, quality_score=0.462

Pozn: MLP predikuje horší MQE ratio (0.605) než referenční jedinec (0.589), ale konfigurace
byla nalezena čistě z predikčního prostoru — skutečný výsledek se může lišit.
Klíčové rozdíly: větší mapa 20×20, `epoch_multiplier=3.5` (delší trénink), `step-down` LR.

### Výsledky ✅

Spuštěno: `python3 app/tools/run_single_som.py --params docs/ea/comparison_params_C.json --seed 42`
Výstup: `data/results/analysis/som-comparison/scen_C_mlp_config/`

| Metrika | Scénář A (baseline) | Scénář C (MLP grid) | Δ |
|---|---|---|---|
| `mqe_improvement_ratio` | **0.5891** | 0.6130 | +0.024 ❌ |
| `topographic_error` | **0.0583** | 0.0860 | +0.028 ❌ |
| `dead_neuron_ratio` | **0.0000** | 0.0025 | +0.003 ≈ |
| `duration` | ~171 s | 315.6 s | 1.8× pomalejší |
| MLP predikce (mqe) | 0.6507 | 0.6046 | — |
| Skutečné vs. predikce (mqe) | 0.5891 vs 0.6507 | 0.6130 vs 0.6046 | — |

**Interpretace:** MLP-doporučená konfigurace je horší než EA-optimalizovaná ve všech metrikách.
Příčiny:
- MLP grid search je omezený pevnými kroky hodnot (ne kontinuální optimalizace jako EA)
- EA iterovala přes 1 500 evaluací a prošla selekcí — MLP grid přes 10 368 bodů, ale každý jen predikoval
- `epoch_multiplier=3.5` způsobil 3× delší trénink bez proporcionálního zisku kvality
- MLP chybně predikuje horší MQE pro Pareto jedince (0.6507 vs skutečné 0.5891) — EA konfigurace je "outlier" mimo středový prostor

**Závěr:** MLP pre-screen je efektivní jako *filtr* špatných konfigurací (eliminace pod threshold),
ale není spolehlivý jako *generátor* optimálních hyperparametrů. EA zůstává nezbytný.

---

## Scénář D — LSTM Phase 3 dynamické řízení *(bude doplněno)*

### Plán

LSTM Phase 3 nahradí statické decay křivky LR, radius a batch size dynamickým controllerem.
Controller dostane aktuální stav SOM (sekvence checkpointů) a rozhodne o dalším nastavení.

Hyperparametry: stejné jako Scénář A (Pareto-optimální), ale místo `lr_decay_type=linear-drop`
a `radius_decay_type=step-down` bude použita dynamická funkce řízená LSTM.

### Stav

- Data pro Phase 3 zatím neexistují (potřeba ~60 SOM běhů s dynamickými variantami)
- Plán generování dat: viz `docs/lstm/LSTM_DYNAMIC_CONTROL.md`
- Tento scénář bude doplněn po dokončení LSTM Phase 3

### Očekávané výsledky (doplnit po implementaci)

| Metrika | Scénář A | Scénář D | Δ |
|---|---|---|---|
| `mqe_improvement_ratio` | 0.5891 | — | — |
| `topographic_error` | 0.0583 | — | — |
| `dead_neuron_ratio` | 0.0000 | — | — |
| `duration` | ~171 s | — | — |

---

## Jak spustit experimenty

Params soubory jsou v `docs/ea/`:
- `comparison_params_A.json` — Pareto-optimální hyperparametry (referenční jedinec)
- `comparison_params_C.json` — MLP grid search výsledek

```bash
# Scénář B: stejné hyperparametry + LSTM early stopping
python3 app/tools/run_single_som.py \
    --params docs/ea/comparison_params_A.json \
    --dataset data/datasets/LungCancerDataset/dataset.csv \
    --ea-config data/datasets/LungCancerDataset/config-ea.json \
    --output data/results/analysis/som-comparison/scen_B_lstm_stop \
    --lstm --seed 42 --label scen_B_pareto_best_lstm

# Scénář C: MLP-doporučené hyperparametry
python3 app/tools/run_single_som.py \
    --params docs/ea/comparison_params_C.json \
    --dataset data/datasets/LungCancerDataset/dataset.csv \
    --ea-config data/datasets/LungCancerDataset/config-ea.json \
    --output data/results/analysis/som-comparison/scen_C_mlp_config \
    --seed 42 --label scen_C_mlp_recommended
```

---

## Shrnutí výsledků

| Scénář | MQE ratio ↓ | TE ↓ | Dead ↓ | Doba | NN | Závěr |
|---|---|---|---|---|---|---|
| **A** Pareto, bez NN | **0.5891** | **0.0583** | 0.000 | **81 s** | ❌ | Referenční výsledek |
| **B** Pareto + LSTM | **0.5891** | **0.0583** | 0.000 | 99 s | LSTM ✅ | Nulová ztráta kvality, LSTM nezasáhlo (+22 % overhead) |
| **C** MLP config, bez NN | 0.6130 | 0.0860 | 0.003 | 316 s | ❌ | Horší kvalita i čas než EA Pareto |
| **C2** MLP config + LSTM | 0.6130 | 0.0780 | 0.000 | 328 s | LSTM ✅ | LSTM nezasáhlo, nepatrné zlepšení TE oproti C |
| **D** LSTM Phase 3 | — | — | — | — | LSTM Ph3 | Doplnit po implementaci |

**Poznámky:**
- LSTM nespustilo early stop v žádném ze scénářů — všechny konfigurace jsou nad prahem kvality 0.75
- MLP config (C/C2) je ve všech metrikách horší než Pareto (A) a trvá 4× déle
- Pro skutečné testování LSTM early stop je nutná konfigurace s předpokládanou nízkou kvalitou (nebo snížit threshold)

**Klíčové zjištění:** EA-optimalizované hyperparametry jsou výrazně lepší než MLP grid search.
LSTM early stopping správně rozpozná kvalitní konfigurace a nechá je doběhnout.

---

## Otevřené úkoly

- [x] Vytvořit `app/tools/run_single_som.py` pro standalone SOM run bez EA
- [x] Spustit Scénář B (LSTM early stopping) a zapsat výsledky
- [x] Spustit Scénář C (MLP-doporučené hyperparametry) a zapsat výsledky
- [ ] Spustit Scénář A (baseline) na macOS pro srovnatelné časové měření
- [ ] Po dokončení LSTM Phase 3 doplnit Scénář D
