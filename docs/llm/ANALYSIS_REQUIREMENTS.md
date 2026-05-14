# Požadavky na analýzu výsledků SOM

**Verze**: 2.1  
**Aktualizováno**: 2026-05-13  
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
| Mapa — topologie | D4 | Prostorové gradienty — jak se hodnoty mění přes mapu | 🔄 plánováno |
| Mapa — topologie | D5 | Regionální shrnutí — flood-fill oblastí se stejnou dominant_category | 🔄 plánováno |
| Mapa — prostorová | D6 | Moran's I per feature — prostorová autokorelace | 🔄 plánováno |
| Mapa — prostorová | D7 | Lokální extrémy per feature — vrcholy a údolí váhové matice | 🔄 plánováno |
| Mapa — prostorová | D8 | Hranice clusterů — Sobel na dominant_category matici | 🔄 plánováno |
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

### D4 🔄 Prostorové gradienty per feature

**Co detekuje:** Jak rychle se průměrná hodnota dimenze mění při pohybu přes mapu.
Vysoký gradient = ostrá hranice (separace skupin). Nízký gradient = homogenní distribuce.

**Implementace:** `np.gradient(weights[:,:,d])` — vrátí `(dy, dx)` per každou buňku.

```python
dy, dx = np.gradient(weights[:, :, d])
magnitude = np.sqrt(dx**2 + dy**2)      # (m, n) — strmost per neuron

# Výstup per dimenzi:
gradient_mean[d]     # průměrná strmost přes celou mapu
gradient_max[d]      # nejostřejší přechod — lokace hranice
gradient_map[d]      # celá (m, n) matice pro vizualizaci
```

**LLM výstup:** "Dimenze AGE vytváří ostrý přechod (gradient 0.74) v oblasti neuronu 8_12.
GENDER je distribuován rovnoměrně (gradient 0.12) — nepřispívá k prostorové separaci."

**Přínos pro EA/NN:** `gradient_mean` je složkou `spatial_quality_score` → EA cílí na
mapy s ostrými prostorově smysluplnými hranicemi.

---

### D5 🔄 Regionální shrnutí — flood-fill

**Co detekuje:** Spojité oblasti mapy kde sousední neurony sdílí stejnou dominantní kategorii.
Výsledek: "levá třetina mapy = LUNG_CANCER=NO, pravá dvě třetiny = YES."

**Implementace:** Flood-fill na `dominant_category_map[m, n]` (matice labelů).

```python
# Vstup: dominant_category per neuron jako 2D matice labelů
# Flood-fill: 4-sousedé se stejným labelem → region

# Výstup:
regions = [
    {"id": 0, "label": "NO", "neurons": [(0,0), (0,1), ...], "area": 42,
     "purity": 0.87, "bbox": (0, 0, 8, 12)},
    {"id": 1, "label": "YES", "neurons": [...], "area": 120, "purity": 0.91, ...}
]
region_count        # počet spojitých regionů (méně = čistší separace)
largest_region_ratio # podíl největšího regionu (blízko 1.0 = jeden dominantní region)
```

**LLM výstup:** "Mapa obsahuje 2 hlavní regiony: NO-cancer (42 neuronů, purity 87 %)
v levé části a YES-cancer (120 neuronů, purity 91 %) v pravé části."

**Přínos pro EA/NN:** `region_count` a `largest_region_ratio` jsou klíčové složky
`spatial_quality_score` — málo velkých čistých regionů = dobře organizovaná mapa.

---

### D6 🔄 Moran's I per feature

**Co detekuje:** Míra prostorové autokorelace — nakolik jsou podobné hodnoty prostorově seskupeny.

```
I ≈ +1  → hodnoty jsou prostorově shluknuty (ideální pro SOM)
I ≈  0  → náhodné rozložení (špatná topologická organizace)
I <  0  → šachovnicový vzor (patologická mapa)
```

**Implementace:** Vážená suma `Σᵢ Σⱼ wᵢⱼ (xᵢ - x̄)(xⱼ - x̄)` kde `wᵢⱼ = 1` pro sousední neurony.

**Přínos:** Přímá numerická validace topologické kvality — lepší než `topographic_error`
pro zachycení, zda mapa vytváří smysluplné prostorové shluky.
Lze zahrnout jako složku EA Pareto cíle i LSTM Phase 3 reward.

---

### D7 🔄 Lokální extrémy per feature

**Co detekuje:** Neurony které jsou lokálním maximem nebo minimem pro danou dimenzi v (3×3) okolí.
Centrum věkové skupiny 65+, centrum mladých pacientů, centrum kuřáků atd.

```python
from scipy.ndimage import maximum_filter, minimum_filter
local_max = (feature_map == maximum_filter(feature_map, size=3))
# → seznam (row, col) neuronů které jsou "vrcholem" pro dimenzi d
```

**LLM výstup:** "Neuron 14_3 je lokální maximum pro AGE (průměr 71.2) a SMOKING (purity 94 %).
Reprezentuje profil starší kuřácký cluster s nejvyšším rizikem."

---

### D8 🔄 Hranice clusterů — Sobel

**Co detekuje:** Kde se mění dominantní kategorie — ostrost hranic mezi clustery.
Doplňuje U-matrix (ta zachycuje obecnou topologickou vzdálenost, Sobel zachycuje
konkrétní kategorické přechody).

```python
from scipy.ndimage import convolve
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
# Aplikovat na dominant_category matici zakódovanou jako int
boundary_map = np.abs(convolve(cat_map_int, sobel_x)) + np.abs(convolve(cat_map_int, sobel_x.T))
# → (m, n) matice — vysoké hodnoty = hranice kategorie
```

**Přínos:** Vizualizace hranic bez renderování PNG. Číselná složka `spatial_quality_score`.

---

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

### Priorita 1 — prostorové metriky (nová oblast, přínos pro EA + LLM + LSTM)

Tyto metriky nahrazují CNN a vstupují do tří modulů najednou: `app/analysis/` (LLM),
`ea.py` (Pareto cíl), `app/lstm/` (Phase 3 reward).

| ID | Co přidat | Modul | Přínos |
|---|---|---|---|
| **D4** | Prostorové gradienty per feature (`np.gradient`) | `stats.py` | EA cíl + LLM |
| **D5** | Regionální shrnutí flood-fill | `stats.py` | EA cíl + LLM (nejvyšší) |
| **D6** | Moran's I per feature | `stats.py` | EA cíl + LSTM reward |
| **D7** | Lokální extrémy per feature | `stats.py` | LLM interpretace |
| **D8** | Hranice clusterů (Sobel) | `stats.py` | EA cíl |

`spatial_quality_score = f(D4, D5, D6, D8)` — kompozitní skóre pro EA a LSTM.

### Priorita 2 — klíčové pro lepší LLM report

| ID | Co přidat | Modul |
|---|---|---|
| **C8** | Hraniční vzorky (boundary samples) | `anomalies.py` |
| **A3** | Skewness + kurtosis | `stats.py` |
| **B6** | Entropie kategorií per cluster | `stats.py` |
| **C3** | IQR-based outliers (Tukey) | `anomalies.py` |

### Priorita 3 — pokročilá topologická analýza

| ID | Co přidat | Modul |
|---|---|---|
| **D3** | Konzistence sousedů (U-matrix numericky) | `stats.py` |
| **E1–E3** | Cluster health score | `stats.py` |

### Priorita 4 — pokročilé vztahy

| ID | Co přidat | Modul |
|---|---|---|
| **A4** | Korelace numerických sloupců | `stats.py` |
| **C5** | Centroid vzdálenost z normalizovaných dat | `anomalies.py` |
