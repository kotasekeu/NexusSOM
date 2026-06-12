# plot_som_topology.py

Vizualizace topologie SOM — projekce vah a trénovacích dat do 2D/3D/HTML.

Kód: [`app/tools/plot_som_topology.py`](../../app/tools/plot_som_topology.py)

---

## Účel

Zobrazuje prostorovou strukturu natrénované SOM — jak jsou neurony rozmístěny v datovém prostoru a jak jsou propojeny hranami dle topologie mřížky. Umožňuje vizuálně odhalit zmačkané oblasti, mrtvé neurony, nebo projekční artefakty.

---

## Výstupy

| Soubor | Obsah |
|--------|-------|
| `topology_2d_{proj}.png` | 2D projekce vah + trénovací data + hrany mřížky |
| `topology_3d_{proj}.png` | 3D projekce (rotovatelná pohledem) |
| `topology_interactive_{proj}.html` | Interaktivní Plotly (zoom, rotace, hover) |

`{proj}` = použitá projekční metoda (`pca`, `umap`, `tsne`).

---

## Použití

```bash
# Základní — 2D PCA
python app/tools/plot_som_topology.py data/datasets/Iris/results/<run>/

# Jiná projekce
python app/tools/plot_som_topology.py <results_dir> --projection umap
python app/tools/plot_som_topology.py <results_dir> --projection tsne

# 3D graf
python app/tools/plot_som_topology.py <results_dir> --3d

# HTML interaktivní
python app/tools/plot_som_topology.py <results_dir> --html

# Všechny formáty najednou
python app/tools/plot_som_topology.py <results_dir> --3d --html

# Jen síť neuronů bez trénovacích dat
python app/tools/plot_som_topology.py <results_dir> --grid-only

# Hustotní overlay
python app/tools/plot_som_topology.py <results_dir> --density

# Manuální override hex topologie
python app/tools/plot_som_topology.py <results_dir> --hex
```

### Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `results_dir` | — | Složka SOM běhu |
| `--projection` | `pca` | Projekční metoda: `pca`, `umap`, `tsne`, `isomap` |
| `--compare` | False | 2×2 grid: PCA vs ISOMAP × data vs váhy (ablation) |
| `--3d` | False | Přidat 3D graf |
| `--only3d` | False | Jen 3D, přeskočit 2D |
| `--html` | False | Vygenerovat interaktivní HTML |
| `--hex` | False | Manuální override — vykreslit jako hex topologii |
| `--grid-only` | False | Jen hrany mřížky, bez trénovacích dat |
| `--density` | False | Overlay hustoty trénovacích dat |
| `--elev` / `--azim` | 45 / 45 | Pohled 3D grafu |
| `--output` / `--output3d` / `--output-html` | auto | Vlastní cesty výstupu |

---

## ISOMAP projekce a --compare mode

### Kdy použít ISOMAP

ISOMAP (Isometric Feature Mapping, Tenenbaum et al. 2000) měří vzdálenosti podél povrchu manifoldu (geodetické vzdálenosti), ne přímou euklidovskou cestou. Pro Swiss Roll:

- **PCA**: vidí největší rozptyl napříč závity spirály → sesype je přes sebe → "chaos"
- **ISOMAP**: sleduje povrch spirály → rozbalí ji do 2D obdélníku

Pro produkční datasety (Iris, BreastCancer) ISOMAP oproti UMAP nepřináší výrazně lepší výsledky a může být pomalejší.

### --compare mode

Generuje `topology_compare_pca_isomap.png` — 2×2 grid:

```
                  PCA (lineární)          ISOMAP (geodetická)
Trénovací data  │ spirála sesypaná        │ rozbalený 2D obdélník
                │ do elipsy               │ vzorků
SOM váhy        │ mřížka kopírující       │ vyžehlená 2D mřížka
(m×n neuronů)   │ závity spirály          │ neuronů
```

Barva = index řádku (≈ parametr `t` pro Swiss Roll). Pokud je výsledek SOM správný:
- ISOMAP sloupec ukáže obdélník s plynulým přechodem barev (spirála rozvinuta)
- Váhová mřížka bude rovnoměrná bez křížení

```bash
python app/tools/plot_som_topology.py <results_dir> --compare
```

---

## Jak funguje

1. Načte `weights.npy`, `training_data.npy`, `sample_assignments.csv`
2. Odstraní always-masked dimenze (`ignore_mask.csv`) — viz sekce níže
3. Projektuje váhy i trénovací data do 2D/3D pomocí zvolené metody
4. Vykreslí hrany mřížky dle topologie (hex cube-coordinate algoritmus nebo rect)
5. Detekuje "natažené hrany" (top 15 % nejdelších) — oranžová = možný problém topologie

### Automatická detekce topologie

Typ mřížky (hex/rect) se čte z `run_metrics.json` (klíč `map_topology`). Manuální `--hex` flag přepíše.

### Maskování dimenzí

Always-masked sloupce (typicky primární ID) jsou odstraněny před projekcí. Bez tohoto kroku by monotónní ID sloupec (0→1) zkreslil PCA/UMAP komponenty. Viz [issues.md #84](../ea/issues.md).

### Hex hrany

Hrany pro hex topologii se počítají v cube koordinátech (6 směrů: `(±1,∓1,0), (±1,0,∓1), (0,±1,∓1)`). Parity-based algoritmus byl buggy a způsoboval "guláš" v grafech — viz [issues.md #82](../ea/issues.md).

---

## Barevné kódování

- **Modré body** = trénovací data
- **Černé body** = neurony (váhy)
- **Šedé hrany** = propojení sousedních neuronů dle topologie mřížky
- **Oranžové hrany** = top 15 % nejdelších hran (indikátor natažení topologie)

