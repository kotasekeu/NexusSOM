# SOM Anomaly Validation — Trips Dataset

Dokument popisuje celý proces validace detekce anomálií v NexusSOM: generování syntetického datasetu s úmyslně zanesými chybami, spuštění SOM, porovnání výsledků s ground truth a vývoj per-dimenzního QE jako nové detekční metriky.

---

## 1. Motivace

Cíl: ověřit, zda SOM dokáže najít záměrně zanesené chyby v datasetech podobných reálným podnikovým datům. Testovány dva typy anomálií:

- **Bodový outlier** — měřicí chyba, extrémní hodnota jednoho vzorku
- **Skrytá skupina (subgroup)** — vzorky se správně vypadající kategorií, ale špatným *poměrem* příznaků (cesta na Slovensko za cenu cesty do Japonska)

Druhý typ je záměrně těžší — SOM ho nechytí, pokud nemá příznaky, které ratio přímo kódují.

---

## 2. Dataset — Pracovní Cesty

### Generátor

Skript: `app/tools/generate_trips_dataset.py`

```bash
.venv/bin/python3 app/tools/generate_trips_dataset.py \
    --rows 800 --inject both --seed 42 \
    --output data/datasets/Trips/dataset
```

Generuje čtyři soubory:

| Soubor | Obsah |
|--------|-------|
| `dataset.csv` | Čistý vstup pro SOM (bez labelu) |
| `dataset_labeled.csv` | Stejná data + sloupec `_anomaly_label` (0/1/2) |
| `dataset_groundtruth.json` | ID anomálních vzorků + metadata |
| `dataset_config.json` | Konfigurační šablona pro SOM |

### Struktura dat

| Sloupec | Typ | Popis |
|---------|-----|-------|
| `id` | int | Primární klíč (1–N) |
| `destination_category` | kategorie | domestic / nearby_eu / europe / intercontinental |
| `purpose` | kategorie | conference / client_visit / training / other |
| `distance_km` | float | Vzdálenost v km |
| `duration_days` | int | Délka cesty ve dnech |
| `transport_cost` | int | Náklady na dopravu v CZK |
| `accommodation_cost` | int | Náklady na ubytování v CZK |
| `total_cost` | int | Celkové náklady |
| `cost_per_km` | float | **Odvozený příznak** — celkové náklady / km |

### Kategorie destinací

| Kategorie | Průměrná vzdálenost | Sazba dopravy | Průměrná délka |
|-----------|--------------------:|---------------|----------------|
| domestic | 280 km | 10 CZK/km | 1,8 dne |
| nearby_eu | 580 km | 16 CZK/km | 2,8 dne |
| europe | 1 600 km | 28 CZK/km | 4,8 dne |
| intercontinental | 8 200 km | 13 CZK/km | 9,0 dne |

Náklady jsou generovány jako `distance × rate × lognormal(0, 0.28)` + `duration × daily_rate × lognormal(0, 0.38)`.

---

## 3. Typy Anomálií

### 3.1 Outlier — Chyba Měření (label = 1)

**Co:** Náhodně vybrané řádky dostanou `transport_cost` vynásobený faktorem `uniform(8, 15)`.

**Proč:** Simuluje duplicitní účtování, chybu v měně, podvodné transakce.

**Jak SOM detekuje:** Vzorek je z-score outlier v rámci svého clusteru pro `transport_cost` a `total_cost`. Také `cost_per_km` je extrémní.

**Parametry (seed=42, N=800):**
- Počet injektovaných: **24** (3 % z 800)
- `cost_per_km` dosahuje stovek až tisíců CZK/km (vs. normál 20–50)

### 3.2 Subgroup — Ratio Anomálie (label = 2)

**Co:** Vzorky z kategorie `domestic` nebo `nearby_eu` (krátká vzdálenost ~200–960 km) dostanou náklady a délku na úrovni `intercontinental` cest.

**Proč:** Simuluje "Slovensko za cenu Japonska" — vzorky vypadají normálně podle kategorie, ale poměr cena/vzdálenost/délka je zcela mimo.

**Jak vypadá:**
- `destination_category`: nearby_eu (vzdálenost 200–960 km)
- `duration_days`: 5–21 dní (intercontinental rozsah)
- `transport_cost`: ~48 000 CZK (intercontinental let)
- `cost_per_km`: **124–1 442 CZK/km** vs. normál nearby_eu median **33 CZK/km**

**Parametry (seed=42, N=800):**
- Počet injektovaných: **32** (4 % z 800)

### Proč je Subgroup těžší na detekci

Bez odvozeného příznaku `cost_per_km`:
- Vzorky se seřadí do clusteru podle *absolutních nákladů* → přiřadí se k intercontinental clusteru
- V tom clusteru není jejich vzdálenost anomální z pohledu z-score, protože intercontinental cluster má přirozeně vysoké náklady
- SOM nevidí, že poměr cena/km je zcela jiný

S `cost_per_km`:
- Subgroup vzorky mají extrémně vysoký `cost_per_km` (200–1 400 CZK/km) vs. intercontinental (21 CZK/km median)
- Tyto vzorky sedí daleko od *každého* neuronu → vysoká Quantization Error (QE)
- Detekce přes `high_qe` typ funguje

---

## 4. Odvozený Příznak `cost_per_km`

Klíčová úprava generátoru — přidána do `_generate_row()`, `inject_outliers()` a `inject_subgroup()`:

```python
cost_per_km = round(total_cost / distance_km, 1)
```

Tento příznak *explicitně kóduje ratio*, který byl předtím pro SOM neviditelný.

| Kategorie | Median cost_per_km | Anomálie |
|-----------|-------------------:|----------|
| domestic (čistý) | 24,4 CZK/km | — |
| nearby_eu (čistý) | 33,3 CZK/km | — |
| intercontinental (čistý) | 21,1 CZK/km | — |
| **subgroup** | **315 CZK/km** | ×9,5 nad nearby_eu |
| **outlier** | stovky–tisíce | ×8–15 transport_cost |

---

## 5. Spuštění Validace

### 5.1 Generování datasetu

```bash
.venv/bin/python3 app/tools/generate_trips_dataset.py \
    --rows 800 --inject both --seed 42
```

### 5.2 Spuštění SOM

```bash
PYTHONPATH=app .venv/bin/python3 app/run_som.py \
    -i data/datasets/Trips/dataset.csv \
    -c data/datasets/Trips/dataset_config.json
```

Výsledky se uloží do `data/datasets/Trips/results/<timestamp>/`.

### 5.3 Porovnání s Ground Truth

```python
import json
import pandas as pd

with open('data/datasets/Trips/dataset_groundtruth.json') as f:
    gt = json.load(f)

with open('data/datasets/Trips/results/<timestamp>/json/llm_context.json') as f:
    ctx = json.load(f)

outlier_ids  = set(gt['anomalies']['outlier']['ids'])
subgroup_ids = set(gt['anomalies']['subgroup']['ids'])

top = ctx['anomalies']['top_anomalies']
detected = {a['sample_id'] for a in top}

tp_out = len(detected & outlier_ids)
tp_sub = len(detected & subgroup_ids)
precision = (tp_out + tp_sub) / len(top)
```

---

## 6. Per-Dimenzní QE — Jak Funguje

### 6.1 Motivace

Původní `sample_assignments.csv` měl jen skalární `qe` (celková vzdálenost vzorku od BMU neuronu). To říká *že* je vzorek anomální, ale ne *proč*.

Per-dimenzní QE rozkládá vzdálenost: pro každý příznak zvlášť, kolik přispívá k celkové vzdálenosti od neuronu. Vzorec v normalizovaném prostoru:

```
qe_dim[feature] = |normalized_value[feature] - bmu_weight[feature]|
```

### 6.2 Kde Vzniká

**`app/som/analysis.py`** — funkce `_get_bmu_assignments()`:
```python
winning_weights = flat_weights[bmu_flat_indices]
diffs = np.abs(normalized_data - winning_weights)
for k, col in enumerate(training_cols):
    if col != primary_id_col and k < diffs.shape[1]:
        df_assigned[f'qe_dim_{col}'] = diffs[:, k]
```

Sloupce `qe_dim_<feature>` jsou uloženy v `sample_assignments.csv`.

**`app/analysis/src/anomalies.py`** — funkce `compute_sample_qe()`:
```python
# Pro každý vzorek v každém clusteru:
bmu_weight = flat_weights[bi * n + bj]
diffs = np.abs(training_data[row_idx] - bmu_weight)
qe_dims = {col: abs_diff for col, abs_diff in zip(training_cols, diffs)}
```

Výsledek se přidá do `top_anomalies` a `anomaly_records` v `llm_context.json`.

### 6.3 Kde Se Vyskytuje

**`sample_assignments.csv`** — per-řádek analýza:
```
sample_id, bmu_i, bmu_j, bmu_key, qe,
qe_dim_destination_category, qe_dim_distance_km, qe_dim_duration_days,
qe_dim_transport_cost, qe_dim_accommodation_cost, qe_dim_total_cost,
qe_dim_cost_per_km, is_outlier
```

**`llm_context.json`** — v každém záznamu v `top_anomalies` a `anomaly_records`:
```json
{
  "sample_id": 599,
  "type": "high_qe",
  "qe": 1.0186,
  "top_qe_dim": "cost_per_km",
  "qe_dims": {
    "cost_per_km": 0.768,
    "destination_category": 0.165,
    "accommodation_cost": 0.111,
    "distance_km": 0.104,
    "duration_days": 0.045,
    "transport_cost": 0.015
  }
}
```

LLM dostane přesný důvod anomálie: *"tento vzorek je daleko od neuronu hlavně kvůli cost_per_km"*.

---

## 7. Typy Anomálií v Detekčním Pipeline

| Typ | Priorita | Detekuje | Kdy |
|-----|----------|---------|-----|
| `multi_dim` | 0 (nejvyšší) | z-score >2.5σ ve 2+ dimenzích v rámci clusteru | Silné outliers uvnitř clusteru |
| `one_of_n` | 1 | 1 vzorek >3× vzdálenější od centroidu clusteru než ostatní | Osamocený vzorek v clusteru |
| `high_qe` | 1 | QE > 2.5× medián datasetu, nezachycen výše | Vzorky daleko od každého neuronu |
| `numeric` | 2 | z-score >2.5σ v 1 dimenzi v rámci clusteru | Slabší outliers |
| `global_extreme` | 3 | Globální min/max nebo >2.5σ od globálního průměru | Absolutní extrémy |
| `categorical_minority` | 4 | Minoritní kategorie ve vysoké čistotě clusteru | Kategorické anomálie |

**Sekundární řazení:** počet outlier dimenzí → QE → distance ratio.

### Proč `high_qe` Funguje pro Subgroup

Subgroup vzorky:
1. Přiřadí se do clusteru poblíž intercontinental vzorků (podobné absolutní náklady)
2. Uvnitř toho clusteru *nevynikají* z-score — cluster je heterogenní
3. Ale jsou daleko od každého neuronu, protože jejich kombinace příznaků (blízká vzdálenost + vysoká cena/km) nikde v trénovacích datech nepřevažuje
4. → vysoké QE → `high_qe` detekce

---

## 8. Výsledky Detekce

Testováno na Trips datasetu, N=800, seed=42, 56 anomálií (24 outlier + 32 subgroup).

### 8.1 Evoluce Detekce

| Varianta | Precision (top 20) | Outlier Recall | Subgroup Recall |
|----------|-------------------:|---------------:|----------------:|
| Původní (bez `cost_per_km`, cluster z-score) | 65 % | 46 % | 6 % |
| S `cost_per_km`, pouze cluster z-score | 40 % | 29 % | 3 % |
| S `cost_per_km` + `high_qe` detekce | **85 %** | **38 %** | **25 %** |

### 8.2 Proč Přidání `cost_per_km` Zhoršilo Cluster Z-Score

Přidáním `cost_per_km` se outlier vzorky přesunuly do jiných clusterů (příznak cost_per_km je teď součástí vzdálenosti). Uvnitř nových clusterů mají menší z-score → méně výsledků přes z-score detekci. Ale v QE-based detekci je výsledek lepší, protože `cost_per_km` zvýšilo QE pro subgroup vzorky.

### 8.3 Proč Subgroup Recall Není 100 %

- Vzorky se stejnou klusterovou "identitou" (např. 5 subgroup vzorků na neuronu 10_9) si navzájem snižují z-score → v rámci clusteru vypadají normálně
- QE-based detekce je omezena prahem 2.5× medián — vzorky s nižším QE (<0.28) nejsou zachyceny
- Typicky 12–15 z 32 subgroup vzorků má QE nad prahem (záleží na konkrétním trénování SOM)

### 8.4 Kde Selhává Detekce

**Falešně pozitivní (FP)** v top 20 jsou převážně čisté vzorky s:
- Neobvyklou kombinací kategorií (purpose × destination)  
- Vyšší přirozenou variabilitou v accommodation_cost (lognormální šum)

Tyto FP mají QE 0.3–0.65 — překrývají se s dolní částí subgroup distribuce. Zvýšení prahu na 3× medián by snížilo FP za cenu nižšího recall.

---

## 9. Klíčové Závěry

### Detekce Ratio Anomálií Vyžaduje Explicitní Příznak

SOM vidí jen to, co je v datech. Ratio anomálie (špatný poměr cena/vzdálenost) musí být zakódována jako příznak — `cost_per_km` = odvozený sloupec. Bez něj SOM seřadí vzorky podle absolutních nákladů a ratio je neviditelné.

### LLM Je Omezen na `llm_context.json`

LLM chat neskenuje surová data — dostane jen `llm_context.json` (extremes, clusters, anomaly_records). Kvalita detekce LLM je tedy přesně taková, jakou mu dá SOM analýza. Bottleneck není LLM, ale ranking anomálií.

### QE Jako Detekční Metrika

Per-sample QE (vzdálenost od BMU) je silnější signál pro subgroup anomálie než z-score v rámci clusteru, protože:
- Funguje i když se více anomálních vzorků shlukuje na jednom neuronu
- Nezávisí na velikosti clusteru
- Říká nejen *že* je vzorek anomální, ale přes `qe_dims` i *proč*

---

## 10. Soubory Projektu

| Soubor | Role |
|--------|------|
| `app/tools/generate_trips_dataset.py` | Generátor datasetu s injekcí anomálií |
| `data/datasets/Trips/dataset.csv` | Vstup pro SOM (bez labelů) |
| `data/datasets/Trips/dataset_labeled.csv` | Data + `_anomaly_label` (0/1/2) |
| `data/datasets/Trips/dataset_groundtruth.json` | Ground truth (IDs anomálií) |
| `data/datasets/Trips/dataset_config.json` | Konfigurace SOM pro tento dataset |
| `app/som/analysis.py` | In-run analýza: BMU assignment + per-dim QE výpočet |
| `app/analysis/src/anomalies.py` | Detekce anomálií: z-score, 1-of-N, high_qe, ranking |
| `app/analysis/src/loader.py` | Načítání výsledků SOM (včetně sample_assignments) |
| `app/analysis/src/context.py` | Sestavení `llm_context.json` + anomaly_records s QE |

---

## 11. Konfigurace SOM pro Validaci

Klíčové parametry v `dataset_config.json`:

```json
{
  "map_size": [15, 15],
  "map_type": "hex",
  "primary_id": "id",
  "random_seed": 42,
  "mqe_evaluations_per_run": 300,
  "max_epochs_without_improvement": 300,
  "categorical_threshold_numeric": 30
}
```

Poznámka: `duration_days` (hodnoty 1–21, max 21 ≤ 30) je automaticky preprocessorem označen jako *kategorický* příznak. To je správné chování — jde o diskrétní ordinální hodnotu, ne spojitou.

---

## 12. Rozšíření Validace

Doporučená rozšíření pro robustnější testování:

1. **Více seedů** — spustit SOM s 5–10 různými `random_seed` hodnotami a průměrovat precision/recall
2. **Jiné datasety** — otestovat stejný typ subgroup anomálie na BreastCancer nebo LungCancer datech
3. **Threshold tuning** — experimentovat s prahem pro `high_qe` (momentálně 2.5× medián)
4. **Větší mapa** — 20×20 nebo 25×25 SOM může lépe separovat subgroup vzorky do izolovaných neuronů
5. **Epoch multiplier** — vyšší hodnota = lepší trénink = přesnější QE distribuce
