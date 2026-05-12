# Požadavky na analýzu výsledků SOM

**Verze**: 2.0  
**Aktualizováno**: 2026-05-12  
**Komponenta**: `app/analysis/` (`loader.py`, `stats.py`, `anomalies.py`, `context.py`)

Tento dokument popisuje všechny analytické výpočty, které je možné provést na výstupech
SOM tréninku (clusterovaný dataset). Cílem je sestavit co nejbohatší `llm_context.json`
tak, aby LLM dokázal podat smysluplnou, statisticky podloženou analýzu dat.

---

## Přehled stavu implementace

| Oblast | ID | Popis | Stav |
|---|---|---|---|
| Globální statistiky | A1 | Min, max, mean, std per numerický sloupec | ✅ |
| Globální statistiky | A2 | Medián, percentily (p25, p75, p90, p95) | ✅ |
| Globální statistiky | A3 | Skewness, kurtosis — tvar distribuce | ❌ |
| Globální statistiky | A4 | Korelace mezi numerickými sloupci | ❌ |
| Globální statistiky | A5 | Globální distribuce kategorií (četnost per hodnota) | ✅ |
| Cluster — základní | B1 | Počet vzorků, QE per neuron | ✅ |
| Cluster — základní | B2 | Dominantní kategorie + purity per neuron | ✅ |
| Cluster — základní | B3 | Mean numerických sloupců per neuron | ✅ |
| Cluster — rozšířené | B4 | Medián + std numerických sloupců per neuron | ✅ |
| Cluster — rozšířené | B5 | Min + max numerických sloupců per neuron | ✅ |
| Cluster — rozšířené | B6 | Entropie kategorií per neuron (homogenita) | ❌ |
| Cluster — rozšířené | B7 | Počet vzorků s každou hodnotou per kategorický sloupec | ✅ |
| Cluster — rozšířené | B8 | Odchylka clusteru od globálního průměru (Z-score) | ✅ |
| Anomálie — globální | C1 | Globální min/max (všechny výskyty) | ✅ |
| Anomálie — globální | C2 | Globální outliers >k·σ od průměru | ✅ |
| Anomálie — globální | C3 | IQR-based outliers (robustní vůči extrémům) | ❌ |
| Anomálie — lokální | C4 | Lokální outlier: vzorek >k·σ od průměru svého neuronu | ✅ |
| Anomálie — lokální | C5 | Vzdálenost vzorku od centroidu neuronu (normalizovaná) | ❌ |
| Anomálie — lokální | C6 | Cluster s 1 extrémem mezi N homogenními vzorky | ✅ |
| Anomálie — lokální | C7 | Vícerozměrný outlier — odchylka na ≥k dimenzích najednou | ✅ |
| Anomálie — hraniční | C8 | Hraniční vzorky — skoro stejná vzdálenost ke dvěma neuronům | ❌ |
| Mapa — topologie | D1 | Aktivní vs. mrtvé neurony | ✅ |
| Mapa — topologie | D2 | Hustota obsazení — Gini koeficient, coverage ratio | ✅ |
| Mapa — topologie | D3 | Konzistence sousedů — jak podobné jsou sousední neurony | ❌ |
| Mapa — topologie | D4 | Prostorové gradienty — jak se hodnoty mění přes mapu | ❌ |
| Mapa — topologie | D5 | Regionální shrnutí — oblasti mapy s podobným profilem | ❌ |
| Cluster — zdraví | E1 | Kompaktnost clusteru (intra-cluster variance) | ❌ |
| Cluster — zdraví | E2 | Separace clusterů (inter-cluster vzdálenosti) | ❌ |
| Cluster — zdraví | E3 | Cluster health score — kombinovaná metrika | ❌ |

---

## A — Globální statistiky datasetu

### A1 ✅ Min, max, mean, std per numerický sloupec
`stats._global_numeric_stats`

### A2 ✅ Medián a percentily
`stats._global_numeric_stats` — `np.percentile(series, [25, 50, 75, 90, 95])`

**Výstup:**
```json
"dimension_stats": {
  "AGE": {
    "min": 30, "max": 80, "mean": 55.2, "std": 14.7,
    "median": 56.0, "p25": 44.0, "p75": 66.0, "p90": 73.0, "p95": 76.0
  }
}
```

### A3 ❌ Skewness a kurtosis

**Co detekuje:** Asymetrie distribuce a ostrost vrcholu (outlier-prone distribuce).

**Implementace:** `scipy.stats.skew`, `scipy.stats.kurtosis`

**Výstup:**
```json
"AGE": { "skewness": -0.12, "kurtosis": 2.1 }
```

---

### A4 ❌ Korelace mezi numerickými sloupci

**Co detekuje:** Lineární závislosti mezi dimenzemi (|r| > 0.7 = redundantní nebo kauzální vztah).

**Výstup:**
```json
"correlations": { "AGE_vs_SMOKING": 0.23 }
```

---

### A5 ✅ Globální distribuce kategorií
`stats._global_category_distributions` — `df[col].value_counts(normalize=True)`

**Výstup:**
```json
"category_distributions": {
  "LUNG_CANCER": { "NO": 0.613, "YES": 0.387 }
}
```

---

## B — Statistiky per cluster (neuron)

### B1 ✅ Počet vzorků, QE per neuron
`stats.compute_cluster_stats` — z `clusters.json` a `quantization_errors.json`

### B2 ✅ Dominantní kategorie + purity
`stats._pie_dominant_purity` — z `pie_data_*.json`

### B3–B5 ✅ Mean, median, std, min, max per cluster
`stats._cluster_numeric_stats`

```json
"dimension_stats": {
  "AGE": { "mean": 62.6, "median": 63.0, "std": 8.2, "min": 45, "max": 79 }
}
```

### B6 ❌ Entropie kategorie per cluster

**Výpočet:** `H = -Σ p_i * log2(p_i)` kde `p_i` je podíl kategorie i.

```json
{ "entropy": { "LUNG_CANCER": 0.0, "SMOKING": 0.97 } }
```

---

### B7 ✅ Počty hodnot per kategorický sloupec per neuron
`stats._pie_category_counts`

```json
{ "category_counts": { "LUNG_CANCER": { "YES": 25, "NO": 2 } } }
```

### B8 ✅ Odchylka clusteru od globálního průměru (Z-score)
`stats._cluster_global_deviation`

```json
{ "global_deviation": { "AGE": 0.51 } }
```

`Z = (cluster_mean − global_mean) / global_std`

---

## C — Detekce anomálií

### C1 ✅ Globální extrémy
`anomalies._enrich_global_extremes` — z `extremes.json`

### C2 ✅ Lokální outliers > k·σ
`anomalies._detect_numeric_outliers` — threshold=2.5σ

### C3 ❌ IQR-based globální outliers

**Co detekuje:** Vzorky mimo `[p25 − 1.5·IQR, p75 + 1.5·IQR]` — Tukeyho metoda, robustní
pro zešikmené distribuce.

---

### C4 ✅ Lokální outlier >k·σ od průměru neuronu
`anomalies._detect_numeric_outliers` — per-vzorek Z-score vs cluster mean/std.

### C5 ❌ Vzdálenost vzorku od centroidu neuronu

**Co detekuje:** Euklidovská vzdálenost vzorku od váhového vektoru neuronu v normalizovaném prostoru.

```python
dist = np.linalg.norm(training_data[idx] - weights[i, j])
```

---

### C6 ✅ Cluster s lokálním extrémem — "1 z N"
`anomalies._detect_one_of_n`

**Algoritmus:**
1. Pro každý cluster s ≥ 5 vzorky
2. Spočítat per-vzorek euklidovskou vzdálenost od centroidu (průměr numerických sloupců)
3. Pokud `max_distance > 3.0 × median_distance`: outlier

```json
{
  "sample_id": 1795,
  "type": "one_of_n",
  "distance_ratio": 8.5,
  "reasons": ["Isolated outlier in neuron 12_5: distance ratio 8.50× cluster median (cluster size 5)"]
}
```

**LLM interpretace:** "V neuronu 12_5 je 4 podobných vzorků a jeden (ID 1795), který se od ostatních liší 8.5×."

---

### C7 ✅ Vícerozměrný outlier
`anomalies._detect_numeric_outliers` — `type: multi_dim` pro vzorky s outlier na ≥2 dimenzích.

```json
{
  "sample_id": 245,
  "type": "multi_dim",
  "n_outlier_dims": 3,
  "outlier_cols": { "AGE": 2.8, "SMOKING": 3.1, "COUGHING": 2.6 }
}
```

---

### C8 ❌ Hraniční vzorky (boundary samples)

**Co detekuje:** Vzorky s téměř stejnou vzdáleností k BMU a druhému nejlepšímu neuronu.

```python
boundary_score = d1 / d2  # blízko 1.0 = hraniční
```

**Potřeba:** `training_data.npy` + `weights.npy` pro výpočet vzdáleností ke všem neuronům.

---

## D — Topologie mapy

### D1 ✅ Aktivní vs. mrtvé neurony
`stats.compute_topology` — počty aktivních, mrtvých neuronů.

### D2 ✅ Hustota obsazení
`stats.compute_topology`:
- `coverage_ratio` — podíl aktivních neuronů
- `density_gini` — Gini koeficient distribuce vzorků po neuronech
- `max_cluster_size`, `min_cluster_size`, `median_cluster_size`

### D3 ❌ Konzistence sousedů

**Co detekuje:** Průměrná vzdálenost váhových vektorů sousedních neuronů (základ U-matrix).

### D4 ❌ Prostorové gradienty hodnot

**Co detekuje:** Korelace souřadnic neuronů s průměrnými hodnotami per numerický sloupec.

### D5 ❌ Regionální shrnutí mapy

**Algoritmus:** Flood-fill nebo DBSCAN na sousedních neuronech se stejnou dominantní kategorií.

---

## E — Zdraví clusterů

### E1 ❌ Kompaktnost clusteru

Průměrná vzdálenost vzorků od centroidu neuronu v původním prostoru. Odpovídá QE,
ale v nenormalizovaném prostoru.

### E2 ❌ Separace clusterů

Průměrná vzdálenost váhových vektorů sousedních neuronů.

### E3 ❌ Cluster health score

`health = compactness × (1 − outlier_ratio)`

---

## Prioritizace budoucí implementace

### Priorita 1 — klíčové pro lepší LLM report

| ID | Co přidat | Modul |
|---|---|---|
| **C8** | Hraniční vzorky (boundary samples) | `anomalies.py` |
| **A3** | Skewness + kurtosis | `stats.py` |
| **B6** | Entropie kategorií per cluster | `stats.py` |
| **C3** | IQR-based outliers (Tukey) | `anomalies.py` |

### Priorita 2 — pokročilá topologická analýza

| ID | Co přidat | Modul |
|---|---|---|
| **D3** | Konzistence sousedů (U-matrix numericky) | `stats.py` |
| **D5** | Regionální shrnutí mapy | nový `regions.py` |
| **E1–E3** | Cluster health score | `stats.py` |

### Priorita 3 — pokročilé vztahy

| ID | Co přidat | Modul |
|---|---|---|
| **A4** | Korelace numerických sloupců | `stats.py` |
| **C5** | Centroid vzdálenost z normalizovaných dat | `anomalies.py` |
| **D4** | Prostorové gradienty hodnot | nový `regions.py` |
