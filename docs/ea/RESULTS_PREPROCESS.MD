# Předzpracování výsledků EA pro trénink neuronových sítí

Postup přípravy trénovacích dat pro MLP, LSTM a CNN ze surových výsledků EA.
Vstupní data: `data/results/` — viz [RESULTS.md](./RESULTS.md).

---

## Přehled cílů jednotlivých sítí

| Síť | Vstup | Výstup | Zdroj dat |
|---|---|---|---|
| **MLP** | Hyperparametry konfigurace | Predikce kvality (MQE, TE) před spuštěním SOM | `results.csv` |
| **LSTM** | Časová řada prvních N% tréninku | Predikce výsledné kvality / konvergence | `training_checkpoints.json` |
| **CNN** | Váhová matice natrénované SOM | Skóre kvality topologické organizace mapy | `weights.npy` |

---

## 1. MLP — Prediktor kvality z hyperparametrů

**Cíl:** Rychlá predikce kvality konfigurace před jejím skutečným spuštěním. EA použije MLP k filtraci pravděpodobně špatných konfigurací a ušetří výpočetní čas.

### Krok 1 — Sběr dat

- Načíst `results.csv` ze všech datasetů a seedů
- Sloučit do jednoho DataFrame
- Přidat sloupec `source_dataset` a metadata datasetu (z `dataset_meta.json`)

### Krok 2 — Filtrování

- Vyloučit jedince kde `status != completed` (načíst ze `status.csv`)
- Zvážit strategii pro penalizované jedince:
  - Volba A: vyloučit (`is_penalized == True`) — čistší data, méně vzorků
  - Volba B: ponechat s přidanou vahou penalizace jako feature — MLP se naučí penalizované oblasti

### Krok 3 — Normalizace MQE napříč datasety

MQE hodnoty jsou specifické pro dataset (BreastCancer MQE ≠ LungCancer MQE). Nutná normalizace:
- **Relativní MQE** = `best_mqe / initial_mqe` = `1 - mqe_improvement_ratio` — bezrozměrné, srovnatelné napříč datasety
- Alternativa: z-score normalizace MQE v rámci každého datasetu zvlášť

### Krok 4 — Příprava features

**Numerické features (přímé):**
`start_learning_rate`, `end_learning_rate`, `start_batch_percent`, `end_batch_percent`, `epoch_multiplier`, `growth_g`, `num_batches`, `map_m`, `map_n`

**Dataset-level features (kontext):**
`ds_n_samples`, `ds_n_active_dimensions`, `ds_n_numeric`, `ds_n_categorical`, `ds_missing_ratio`

**Kategorické features (one-hot encoding):**
- `lr_decay_type`: `exp-drop`, `linear-drop`, `log-drop`, `step-down`
- `radius_decay_type`: `exp-drop`, `linear-drop`, `log-drop`, `step-down`
- `batch_growth_type`: `exp-growth`, `linear-growth` *(pozor: `log-growth` odstraněno, starší data ho mohou obsahovat — sjednotit schéma)*

**Upozornění:** Stávající `bc_dataset.csv` v `app/mlp/data/` byl trénován na starých datech s `log-growth`. Při přetrénování sjednotit one-hot schéma.

### Krok 5 — Targets

Primární target: `raw_best_mqe` (nebo normalizovaná verze z kroku 3)
Sekundární targety: `raw_topographic_error`, `dead_neuron_ratio`

Zvážit multi-output regrese vs. samostatné modely.

### Krok 6 — Rozdělení a export

- Train/val/test split: stratifikovat podle datasetu (aby každý dataset byl zastoupen ve všech splitech)
- Uložit scaler (StandardScaler nebo MinMaxScaler) společně s modelem — nutné pro inference
- Výstup: `app/mlp/data/all_combined_mlp.csv` + aktualizovaný `metadata.json`

---

## 2. LSTM — Prediktor konvergence z průběhu tréninku

**Cíl:** Z prvních ~30 % trénovací křivky predikovat výslednou kvalitu. EA nebo uživatel může přerušit slibné/beznadějné konfigurace dříve.

### Krok 1 — Sběr sekvencí

- Pro každého jedince načíst `training_checkpoints.json`
- Základní features sekvence: `[progress, mqe, topographic_error, dead_neuron_ratio, learning_rate, radius]`
- Délka sekvence: 500 bodů (dle `mqe_evaluations_per_run`)

### Krok 2 — Normalizace sekvencí

- `progress` (0–1): již normalizováno
- `mqe`: podělit počáteční MQE (`checkpoints[0].mqe`) → relativní pokles
- `learning_rate`, `radius`: podělit počáteční hodnotou → relativní pokles
- `dead_neuron_ratio`, `topographic_error`: (0–1), ponechat

### Krok 3 — Definice predikčního okna

Rozhodnout jakou délku vstupu LSTM dostane:
- **Vstup:** prvních `K%` sekvence (např. K=20, 30, 50 — experiment)
- **Target:** finální `raw_best_mqe` (normalizované)

Připravit více variant datasetu pro různá K.

### Krok 4 — Zarovnání délek

Sekvence mohou mít různou délku (early stopping). Možnosti:
- Ořezat na minimum z dostupných délek
- Pad + mask pro kratší sekvence

### Krok 5 — Přidání kontextových features

Statické hyperparametry (map_size, epoch_multiplier, decay typy) jako doplňkový vstup vedle časové řady (LSTM + dense branch).

### Krok 6 — Export

Výstup: numpy array `(n_samples, seq_len, n_features)` + target array `(n_samples,)` uložené do `app/lstm/data/`.

---

## 3. CNN — Prediktor kvality z váhové matice SOM

**Cíl:** Ohodnotit kvalitu topologické organizace natrénované SOM na základě struktury váhové matice. Alternativa k výpočtu U-matice v reálném čase.

### Krok 1 — Výběr vzorků

- CNN trénovat primárně na **Pareto-frontě** (archiv) — nejkvalitnější mapy
- Doplnit negativními vzorky (penalizovaní jedinci) pro kontrast
- Cílem není pokrýt celý prostor konfigurací, ale naučit se rozlišit organizovaná vs. neorganizovaná zobrazení

### Krok 2 — Sběr vah

- Načíst `weights.npy` pro každého vybraného jedince (tvar `m × n × d`)
- Problém: různé datasety mají různé `d` (dimenze)

### Krok 3 — Sjednocení vstupu

Možnosti jak sjednotit různé tvary:
- **Varianta A:** Trénovat samostatné CNN modely per-dataset (jednodušší, nescaluje)
- **Varianta B:** Promítnout váhy do fixní reprezentace (PCA na d dimenzí → konstantní d=10)
- **Varianta C:** Použít vizualizace místo vah — `u_matrix.png` a `distance_map.png` jsou vždy RGB obrázky. Resize na 64×64 → unified input.

Doporučení: začít s Variantou C (obrázky), je nejjednodušší na sjednocení.

### Krok 4 — Targets

- `raw_best_mqe` (normalizované)
- Nebo binární klasifikace: `is_pareto` (organizováno = 1, penalizováno = 0)

### Krok 5 — Data augmentation

- Rotace / flip map (hexagonální topologie je symetrická)
- Pomůže s malým počtem trénovacích vzorků (Pareto front je malý)

### Krok 6 — Export

Výstup: uložit jako numpy array obrázků `(n, 64, 64, 3)` nebo `(n, 64, 64, 1)` + target do `app/cnn/data/`.

---

## Sdílené kroky — všechny sítě

### Distribuce dat napříč datasety

| Dataset | n_samples | Jedinců (~) | Checkpointů (~) |
|---|---|---|---|
| 01-BreastCancer | 569 | 1 468 | 734 000 |
| 02-IndianLiverPatientRecords | ? | ~1 460 | ~730 000 |
| 03-LungCancerDataset | 3 000 | ~1 465 | ~732 500 |
| 04-PimaIndiansDiabetes | ? | ~1 460 | ~730 000 |

### Konzistentní UID napříč soubory

`uid` v `results.csv` = název složky v `individuals/` = hash konfigurace. Slouží jako primární klíč pro spojování dat ze `results.csv`, `pareto_front.csv` a `individuals/`.

### Verze schématu features

Při přidávání/odebírání parametrů (např. odstranění `log-growth`) zaznamenat verzi schématu do metadata souboru. Stávající `bc_dataset_metadata.json` obsahuje `log-growth` — je třeba aktualizovat.
