# Analysis Module — dokumentace

**Verze**: 2.0  
**Aktualizováno**: 2026-05-12  
**Komponenta**: `app/analysis/`

---

## Přehled

`app/analysis/` je samostatný modul pro post-processing SOM výsledků. Přečte výstupy
SOM tréninku, vypočítá statistiky a anomálie, a sestaví `llm_context.json` — strukturovaný
kontext připravený pro LLM analýzu.

Modul je **nezávislý na `app/som/`** — pracuje pouze se soubory v adresáři výsledků.

---

## Struktura modulu

```
app/analysis/
├── __init__.py
└── src/
    ├── __init__.py
    ├── loader.py       ← IO pouze, žádné výpočty
    ├── stats.py        ← globální statistiky, cluster statistiky, topologie
    ├── anomalies.py    ← detekce anomálií (lokální, globální, 1-z-N)
    └── context.py      ← assembly finálního llm_context.json
```

---

## Místo v pipeline

```
run_som.py
  ↓
preprocess_data()     → original_input.csv, preprocessing_info.json, dataset_meta.json
  ↓
som.train()           → weights.npy, run_metrics.json
  ↓
perform_analysis()    → clusters.json, quantization_errors.json, extremes.json,
                        pie_data_*.json
  ↓
save_llm_context()    → json/llm_context.json   ← app/analysis/src/context.py
  ↓
run_llm.py            → čte llm_context.json → LLM prompt → report.md
```

---

## Vstupní soubory

Všechny soubory jsou čteny z `<results_dir>/`:

| Soubor | Obsah | Odkud pochází |
|---|---|---|
| `dataset_meta.json` | Statistiky datasetu (n_samples, primary_id_col, atd.) | `preprocess_data()` |
| `run_metrics.json` | map_size, topology, topographic_error, duration | `run.py` (od verze 1.0) |
| `json/preprocessing_info.json` | Typ každého sloupce, nunique_ratio, status | `preprocess_data()` |
| `json/clusters.json` | `{neuron_key: [id1, id2, ...]}` — přiřazení vzorků | `perform_analysis()` |
| `json/quantization_errors.json` | Celková MQE + per-neuron QE | `perform_analysis()` |
| `json/extremes.json` | `{sample_id: [reason_string, ...]}` — globální extrémy | `perform_analysis()` |
| `json/pie_data_COL.json` | `{categories: {}, counts: {neuron_key: {code: count}}}` | `perform_analysis()` |
| `csv/original_input.csv` | Původní nenormalizovaná data | `preprocess_data()` |
| `csv/weights.npy` | Váhy SOM mapy `(m, n, dim)` — pro inferenci map_size | `som.train()` |
| `csv/training_data.npy` | Normalizovaná data `(N, dim)` — pro centroid vzdálenosti | `som.train()` |

---

## Jak se inferuje primary_id

Primary ID sloupec se identifikuje automaticky ze dvou zdrojů (v tomto pořadí):

1. `dataset_meta.json["ds_primary_id_col"]`
2. Sloupec s `nunique_ratio >= 0.99` v `preprocessing_info.json`

Primary ID je vyřazeno z:
- numerických statistik
- průměrů clusterů
- anomálií (záznamy z `extremes.json` odkazující na primary_id sloupec jsou filtrovány)

---

## Výstupní soubor: `json/llm_context.json`

### Schéma

```json
{
  "map": {
    "size": [18, 18],
    "topology": "hex",
    "total_samples": 3000,
    "total_neurons": 324,
    "active_neurons": 324,
    "dead_neurons": 0,
    "dead_ratio": 0.0,
    "coverage_ratio": 1.0,
    "density_gini": 0.21,
    "max_cluster_size": 28,
    "min_cluster_size": 1,
    "median_cluster_size": 9.0,
    "mqe": 1.4657,
    "topographic_error": 0.058
  },

  "clusters": [
    {
      "neuron": "0_0",
      "sample_count": 28,
      "quantization_error": 1.5221,
      "dominant_category": {
        "LUNG_CANCER": "NO",
        "COUGHING": "1"
      },
      "purity": {
        "LUNG_CANCER": 0.8571,
        "COUGHING": 1.0
      },
      "category_counts": {
        "LUNG_CANCER": { "NO": 24, "YES": 4 }
      },
      "dimension_stats": {
        "AGE": { "mean": 55.75, "median": 56.0, "std": 12.3, "min": 30.0, "max": 79.0 }
      },
      "dimension_means": { "AGE": 55.75 },
      "global_deviation": { "AGE": 0.04 }
    }
  ],

  "anomalies": {
    "global_outlier_count": 99,
    "local_outlier_count": 268,
    "top_anomalies": [
      {
        "sample_id": 1795,
        "neuron": "12_5",
        "type": "one_of_n",
        "distance_ratio": 8.5,
        "reasons": [
          "Isolated outlier in neuron 12_5: distance ratio 8.50× cluster median (cluster size 5)"
        ]
      }
    ],
    "cluster_local_outliers": {
      "12_5": [ ... ]
    }
  },

  "dimension_stats": {
    "AGE": {
      "min": 30.0, "max": 80.0, "mean": 55.169, "std": 14.7237,
      "median": 56.0, "p25": 42.0, "p75": 68.0, "p90": 72.0, "p95": 75.0
    }
  },

  "category_distributions": {
    "LUNG_CANCER": { "NO": 0.613, "YES": 0.387 },
    "SMOKING":     { "1": 0.551, "2": 0.449 }
  }
}
```

### Popis polí

**`map`**
- `size` — rozměry mapy `[m, n]`, z `run_metrics.json` nebo inferováno z `weights.npy`
- `density_gini` — Gini koeficient distribuce vzorků (0=rovnoměrné, 1=vše v jednom neuronu)
- `topographic_error` — null pro starší běhy bez `run_metrics.json`

**`clusters`** — seřazeny sestupně dle `sample_count`
- `dominant_category` — hodnota s nejvyšším počtem vzorků per kategorický sloupec
- `purity` — podíl dominantní kategorie (0.0–1.0)
- `category_counts` — počty vzorků per hodnota per kategorický sloupec
- `dimension_stats` — mean, median, std, min, max pro každý numerický sloupec
- `dimension_means` — aliasový slovník mean hodnot (kompatibilita s context_builder)
- `global_deviation` — Z-score odchylky cluster mean od globálního průměru

**`anomalies`**
- `one_of_n` — vzorek jehož vzdálenost od centroidu clusteru je ≥ 3× medián vzdáleností ostatních
- `numeric` — vzorek s Z-score > 2.5 od průměru svého neuronu
- `multi_dim` — numeric outlier na ≥2 dimenzích současně
- `global_extreme` — vzorek na globálním min/max (z `extremes.json`)

**`dimension_stats`** — globální statistiky, percentily p25/p75/p90/p95

**`category_distributions`** — globální četnosti kategorií (třídní vyváženost)

---

## Spuštění

### Automaticky po SOM tréninku

`run.py` volá `save_llm_context()` automaticky po `perform_analysis()`:

```bash
python3 app/run_som.py -i data/datasets/LungCancerDataset/dataset.csv \
                       -c data/datasets/LungCancerDataset/config-som.json
# → automaticky uloží llm_context.json do results/<timestamp>/json/
```

### Standalone — zpětné generování kontextu

Pro existující výsledky bez `llm_context.json` nebo po aktualizaci analýzy:

```bash
python3 app/run_analysis.py -i data/datasets/LungCancerDataset/results/20260511_172536
# → data/datasets/LungCancerDataset/results/20260511_172536/json/llm_context.json
```

---

## Integrace s context_builder.py

`app/llm/src/context_builder.py::build_context()` preferuje předem sestavený `llm_context.json`:

```
build_context(dataset_path)
  ↓
llm_context.json existuje?  → načti přímo
                     ↓ ne
           _build_context_from_raw()
                     ↓
           analysis.src.context.build_llm_context()
                     ↓ (fallback pokud import selže)
           minimální načtení surových souborů
```

---

## run_metrics.json

Soubor ukládaný z `run.py` přímo po dokončení `som.train()`:

```json
{
  "map_size": [18, 18],
  "map_topology": "hex",
  "best_mqe": 1.440918,
  "topographic_error": 0.0583,
  "duration": 77.4,
  "lstm_stopped": false,
  "lstm_stop_progress": null
}
```

Pro starší běhy bez `run_metrics.json` je `topographic_error` v `llm_context.json` `null`
a `map_size` se inferuje z `weights.npy` tvaru `(m, n, dim)`.

---

## Omezení

| # | Problém | Dopad | Řešení |
|---|---|---|---|
| 1 | Binární sloupce zobrazují kódy "1"/"2" místo doménových labelů | LLM vidí "SMOKING=1" ne "Symptom absent" | Přidat `dataset_context.txt` |
| 2 | `topographic_error = null` pro staré běhy | Chybí jedno pole v map overview | Spustit `run_analysis.py` znovu po aktualizaci |
| 3 | `global_outlier_count` zahrnuje všechny instance globálního min/max | Vysoký počet pro sdílené krajní hodnoty (AGE=30 = 99 vzorků) | Normální — viz `dataset_context.txt` pro interpretaci |
| 4 | Kategorické minority flagovány na úrovni clusteru, ne per-vzorek | LLM nedostane konkrétní sample_id | `pie_data` neobsahuje per-vzorek přiřazení |
