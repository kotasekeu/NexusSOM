# MLP — Vizualizace natrénovaného modelu

Popis generovaných grafů, jejich interpretace a příkazy pro spuštění.

---

## Generování grafů

Z adresáře `app/mlp/`:

```bash
# Aktuální model (mlp_latest.keras)
python3 visualize_model.py

# Konkrétní model s vlastním labelem
python3 visualize_model.py --model models/mlp_v2.keras --label v2

# Porovnání dvou modelů
python3 visualize_model.py --label v2 --compare models/mlp_v1.keras --label_b v1
```

**Výstup:** `app/mlp/visualizations/{label}/` — čtyři PNG soubory.

---

## Grafy

### 1. `scatter.png` — Actual vs. Predicted

Bodový graf skutečné vs. predikované hodnoty pro každý ze tří targetů. Přerušovaná čára = ideální predikce (y = x).

| Target | Co říká dobrý scatter |
|---|---|
| MQE improvement ratio | Body blízko diagonály, rovnoměrně rozložené po celém rozsahu 0.2–0.8 |
| Topographic error | Body soustředěné blízko 0 — TE je přirozeně nízká u většiny konfigurací |
| Dead neuron ratio | Bimodální — buď skoro 0 (dobrá mapa) nebo výrazně vyšší (mrtvé neurony) |

**Aktuální model (v1, 4 datasety):**

- MQE: MAE=0.048, RMSE=0.063 — dobrá predikce středních hodnot, větší rozptyl u extrémů
- TE: MAE=0.019 — velmi přesné
- Dead: MAE=0.018 — velmi přesné

---

### 2. `residuals.png` — Distribuce chyb

Histogram residuálů `(predikce − skutečná hodnota)` pro každý target. Vertikální čáry: černá = 0, červená = průměr residuálu.

**Co sledovat:**
- **Centrace kolem nuly** — model není systematicky nad nebo pod. Aktuální modely: mean ≈ 0 u všech tří targetů ✓
- **Symetrie** — asymetrický histogram = model přeceňuje nebo podceňuje část rozsahu
- **Šířka** — std residuálů. MQE: std=0.062, TE: std=0.030, Dead: std=0.023

**Aktuální model:**
MQE má mírně asymetrický ocas doleva (model trochu přeceňuje nízké hodnoty). TE a Dead jsou symetrické a úzké.

---

### 3. `per_dataset.png` — MAE per dataset

Sloupcový graf průměrné absolutní chyby zvlášť pro každý dataset a každý target.

**Aktuální výsledky:**

| Dataset | MQE MAE | TE MAE | Dead MAE |
|---|---|---|---|
| BreastCancer | 0.061 | 0.013 | 0.021 |
| IndianLiverPatientRecords | 0.068 | 0.014 | 0.029 |
| LungCancerDataset | **0.023** | 0.029 | **0.008** |
| PimaIndiansDiabetes | 0.063 | 0.013 | 0.021 |

**Interpretace:**
LungCancer (n=3000) je predikován nejlépe pro MQE a Dead — dataset s větším počtem vzorků má pravděpodobně stabilnější výsledky EA a model se ho snáze naučí. IndianLiverPatientRecords je nejhorší pro MQE — pravděpodobně specifická kombinace dimenzí a velikosti, která se na ostatní datasety nezobecňuje dobře. S přidáním více datasetů se tyto rozdíly vyrovnají.

---

### 4. `feature_importance.png` — Důležitost příznaků (permutační)

Horizontální sloupcový graf. Každý příznak je permutován (promíchán) a měří se nárůst MSE — čím větší nárůst, tím důležitější příznak.

**Aktuální výsledky a interpretace:**

#### Skupina 1 — Dataset vlastnosti (dominující)

```
ds_active_dimensions  ████████████████████████  0.0093
ds_numeric            ████████████              0.0049
ds_n_samples          ████████████              0.0048
ds_categorical        ██████████                0.0038
```

Daleko největší vliv. Model se primárně naučil odpovídat na otázku *"jaký typ datasetu to je?"* — počet dimenzí a jejich typy determinují jaká konfigurace SOM bude fungovat. To dává smysl: větší mapa potřebuje více iterací, více kategorických dimenzí mění distribuci vzdáleností, čímž se mění optimální learning rate a radius.

#### Skupina 2 — Klíčové hyperparametry mapy

```
start_radius_init_ratio  ██              0.0008
epoch_multiplier         █               0.0004
map_m / map_n            █               0.0003
start_learning_rate      █               0.0002
```

Po dataset vlastnostech jsou to parametry, které určují *jak je mapa organizována* — jak daleko sousedství sahá na začátku (`start_radius_init_ratio`) a kolik průchodů dat SOM dostane (`epoch_multiplier`). Tvar mapy (`map_m`, `map_n`) je důležitý ale méně než radius — Vesanto heuristika ho dobře omezuje, takže zbývá jen ladění v rámci rozsahu.

#### Skupina 3 — Zanedbatelný vliv

```
growth_g, num_batches, batch_growth_type,
lr_decay_type, radius_decay_type,
end_learning_rate, end_batch_percent  ≈ 0
```

Tyto parametry mají vliv na průběh tréninku (tvar křivky), ale výsledná kvalita mapy na nich závisí minimálně — pokud jsou základní parametry (radius, epoch_multiplier, map_size) správně nastavené, tvar poklesu learning rate nebo počet dávek výsledek zásadně nemění.

**Praktický závěr:** Po více bězích EA budou tato data základem pro rozhodnutí co zafixovat nebo zúžit v search space — viz diskuzi v [RESULTS_PREPROCESS.MD](../ea/RESULTS_PREPROCESS.MD).

---

## Srovnání dvou modelů

Po přetrénování na více datech:

```bash
# Záloha před přetrénováním
cp models/mlp_latest.keras models/mlp_v1.keras
cp models/mlp_scaler_latest.pkl models/mlp_scaler_v1.pkl

# Přetrénování (přepíše mlp_latest)
python3 src/train.py

# Srovnání
python3 visualize_model.py --label v2 --compare models/mlp_v1.keras --label_b v1
```

Výstup `visualizations/v2_vs_v1/comparison.png` ukáže sloupcový graf MAE pro oba modely side-by-side + tabulku delta:

```
Target                               v2          v1       Δ (v1-v2)
----------------------------------------------------------------------
mqe_improvement_ratio              0.0312      0.0484     ▼ 0.0172
topographic_error                  0.0165      0.0189     ▼ 0.0024
dead_neuron_ratio                  0.0141      0.0176     ▼ 0.0035
```

▼ = zlepšení, ▲ = zhoršení oproti předchozí verzi.
