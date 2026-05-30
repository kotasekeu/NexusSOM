# generate_trips_dataset.py

Generátor syntetického datasetu pracovních cest s injekcí anomálií se známými ID.

Kód: [`app/tools/generate_trips_dataset.py`](../../app/tools/generate_trips_dataset.py)

---

## Účel

Generuje realistická data pracovních cest (destinace, vzdálenost, délka, náklady) se záměrně vloženými anomáliemi. Protože víme přesně která ID jsou anomálie, lze SOM detekcí změřit **precision, recall, F1** — objektivní ověření funkčnosti outlier detekce.

---

## Typy anomálií

| Typ | Popis | SOM signál |
|-----|-------|-----------|
| `outlier` | Náhodné řádky s extrémními náklady (8–15× normál) — chyba měření | `global_extreme`, numerický outlier uvnitř clusteru |
| `subgroup` | Blízké cesty (~400 km) s mezikontinentálními náklady a délkou — "Slovensko za ceny Japonska" | `multi-dim outlier` (nízká vzdálenost v drahém clusteru), kategorická menšina |

---

## Použití

```bash
# Základní — oba typy anomálií
python app/tools/generate_trips_dataset.py --rows 800 --output data/datasets/Trips/dataset

# Vlastní frakce anomálií
python app/tools/generate_trips_dataset.py \
  --rows 800 \
  --inject both \
  --outlier-fraction 0.03 \
  --subgroup-fraction 0.04 \
  --seed 42

# Pouze jeden typ
python app/tools/generate_trips_dataset.py --rows 800 --inject outlier --seed 42
python app/tools/generate_trips_dataset.py --rows 800 --inject subgroup --seed 42
```

### Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--rows` | 800 | Počet řádků |
| `--output` | `data/datasets/Trips/dataset` | Výstupní prefix (bez přípony) |
| `--inject` | `both` | Typ anomálií: `outlier`, `subgroup`, `both`, `none` |
| `--outlier-fraction` | 0.03 | Podíl outlier anomálií (0–1) |
| `--subgroup-fraction` | 0.04 | Podíl subgroup anomálií (0–1) |
| `--seed` | 42 | Random seed |
| `--verbose` | False | Výpis detailů generování |

---

## Výstupní soubory

| Soubor | Obsah |
|--------|-------|
| `{base}.csv` | Čistý CSV pro SOM (bez label sloupce) |
| `{base}_labeled.csv` | Stejná data + sloupec `_anomaly_label` (0=čistý, 1=outlier, 2=subgroup) |
| `{base}_groundtruth.json` | ID anomálií a metadata pro evaluaci |
| `{base}_config.json` | Šablona config-som.json pro tento dataset |

---

## Evaluace detekce

Po spuštění SOM analýzy porovnat detekované anomálie s `groundtruth.json`:

```python
detected = set(sample_assignments[sample_assignments['is_outlier']]['sample_id'])
true_anomalies = set(groundtruth['outlier_ids'] + groundtruth['subgroup_ids'])

precision = len(detected & true_anomalies) / len(detected)
recall    = len(detected & true_anomalies) / len(true_anomalies)
f1        = 2 * precision * recall / (precision + recall)
```
