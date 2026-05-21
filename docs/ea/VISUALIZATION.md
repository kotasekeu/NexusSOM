# Vizualizační nástroje EA a SOM

Dokumentace ke skriptům `plot_pareto_evolution.py` a `plot_som_topology.py`.

---

## plot_pareto_evolution.py

Vizualizace evoluce Pareto fronty z výsledků EA běhu.

### Vstup

```
<results_dir>/pareto_front.csv
```

Sloupce: `generation`, `uid`, `raw_mqe_ratio`, `raw_te`, `dead_ratio`,
`is_penalized`, `map_m`, `map_n`, `duration`, `constraint_violation`.

### Výstup

| Soubor | Obsah |
|--------|-------|
| `pareto_evolution.png` | 5-panelový přehled (archiv + MQE evoluce + 3× 2D projekce) |
| `pareto_3d.png` | 3D scatter se shadow projekcemi na stěny |
| `pareto_evolution_stats.csv` | Statistiky per-generace (volitelné `--csv`) |

### Orientace os — konvence

Všechny osy jsou **invertované**: nízká hodnota (dobrá) = vpravo nahoře.
Konvergence generací je vždy vizuálně směrem doprava-nahoru.

| Osa | Metrika | Směr |
|-----|---------|------|
| X | `raw_mqe_ratio` | → lepší |
| Y | `raw_te` / `dead_ratio` | ↑ lepší |
| Z (3D) | `dead_ratio` | ↑ lepší |

### Parametry

```bash
python app/tools/plot_pareto_evolution.py <results_dir> [options]

--output PATH         výstupní cesta pro 2D graf
--output3d PATH       výstupní cesta pro 3D graf
--no3d                přeskočit 3D graf
--fixed-range         osy fixovány na [0, 1] — srovnatelné grafy napříč běhy
--guide-lines         vodicí čáry z každého bodu na všechny tři stěny (3D)
--space-grid N        N dělení na každé ose (3D)
--lattice             mřížka dělení v 3D (vyžaduje --space-grid)
--elev FLOAT          elevace pohledu 3D (default: 22)
--azim FLOAT          azimut pohledu 3D (default: 225)
--csv [PATH]          exportovat statistiky do CSV
```

### --fixed-range

Nastavuje všechny osy na rozsah `[0, 1]`. Použít pro porovnání více EA běhů
vedle sebe — mřížka i body mají vždy stejné měřítko.

```bash
# Srovnání dvou běhů
python app/tools/plot_pareto_evolution.py results/run_A --fixed-range --output run_A.png
python app/tools/plot_pareto_evolution.py results/run_B --fixed-range --output run_B.png
```

---

## plot_som_topology.py

Vizualizace váhových vektorů SOM neuronů a jejich topologické mřížky v prostoru
redukovaném z D-dimenzionálního prostoru dat.

**Hlavní účel:** Křížení čar mřížky = topologická chyba. Graf přímo vizualizuje
to, co TE metrika vyjadřuje číslem.

### Vstup

```
<results_dir>/csv/weights.npy          (m, n, dim)
<results_dir>/csv/training_data.npy    (N, dim)
<results_dir>/csv/sample_assignments.csv  (pro obarvení neuronů QE)
<results_dir>/dataset_meta.json        (detekce hex topologie)
```

### Výstup

| Soubor | Obsah |
|--------|-------|
| `topology_2d_<method>.png` | Statický 2D graf |
| `topology_3d_<method>.png` | Statický 3D graf (volitelné) |
| `topology_interactive_<method>.html` | Interaktivní Plotly — zoom, hover |

### Vrstvy grafu

| Vrstva | Co zobrazuje | Výchozí stav |
|--------|-------------|--------------|
| 1 | Trénovací vzorky (scatter nebo density) | zapnuto |
| 2 | Neurony obarvené průměrným QE (plasma colormap) | zapnuto |
| 3 | Čáry topologické mřížky (sousední neurony) | vždy |

### Projekční algoritmy

| Algoritmus | Použití | Omezení |
|-----------|---------|---------|
| `pca` | Default. Rychlé, lineární. | Selhává na velkých mapách — nízké % vysvětlené variance stlačí mřížku do středu |
| `umap` | **Doporučeno pro mapy 15×15+.** Rozbalí nelineární manifold. | `pip install umap-learn` |
| `tsne` | Pomalé, fituje data a váhy společně. | Bez `transform()` — nelze separátně |

Pro mapy > 200 neuronů skript automaticky doporučí `--projection umap`.

### Čitelnost podle velikosti mapy

Skript automaticky škáluje velikost prvků podle počtu neuronů:

| Neurony | Marker size | Šířka čar |
|---------|------------|-----------|
| ≤ 100 | 55 | 0.9 |
| ≤ 225 | 30 | 0.6 |
| ≤ 400 | 18 | 0.45 |
| > 400 | 10 | 0.3 |

### Parametry

```bash
python app/tools/plot_som_topology.py <results_dir> [options]

--projection {pca,umap,tsne}   algoritmus projekce (default: pca)
--grid-only                    skrýt trénovací vzorky — čistá geometrie mřížky
--density                      hexbin hustotní mapa místo scatter (2D)
--hex                          hexagonální topologie (auto z dataset_meta.json)
--3d                           také generovat 3D graf
--only3d                       jen 3D, přeskočit 2D
--html                         interaktivní HTML (Plotly, fullscreen)
--output PATH                  výstup 2D PNG
--output3d PATH                výstup 3D PNG
--output-html PATH             výstup HTML
--elev FLOAT                   elevace 3D pohledu (default: 45)
--azim FLOAT                   azimut 3D pohledu (default: 45)
```

### Typické použití

```bash
# Základní 2D, PCA — rychlý přehled
python app/tools/plot_som_topology.py data/datasets/Wine/results/

# Velká mapa — UMAP pro čitelnou mřížku
python app/tools/plot_som_topology.py data/datasets/Trips/results/ \
    --projection umap --grid-only

# Interaktivní HTML pro detailní inspekci (fullscreen, hover info)
python app/tools/plot_som_topology.py data/datasets/Trips/results/ \
    --projection umap --html

# Hustotní pozadí + 3D
python app/tools/plot_som_topology.py data/datasets/Wine/results/ \
    --projection umap --density --3d --elev 60 --azim -30
```

### Interaktivní HTML

`--html` generuje Plotly vizualizaci přes celou obrazovku prohlížeče.

Funkce:
- Zoom, pan, zoom-to-selection
- Hover na každém neuronu zobrazí: index `[i, j]` + průměrné QE
- `responsive: true` — přizpůsobí se resize okna

### Interpretace grafu

**Nízká topografická chyba (zdravá mapa):**  
Mřížka se hladce roztáhne přes cloud dat. Čáry netvoří uzly ani se nekříží.
Neurony pokrývají datový prostor rovnoměrně.

**Vysoká topografická chyba (kolaps topologie):**  
Čáry mřížky se kříží, tvoří „uzly" nebo se část mřížky prudce láme a skládá
sama do sebe. Prostorově blízké neurony reprezentují vzdálená data.

> **Poznámka:** Křížení mohou částečně způsobit i omezení PCA projekce
> (nízký % vysvětlené variance). Při pochybnostech ověřit s `--projection umap`.

### Vztah k TE metrice

Topographic Error (TE) = procento vzorků, kde BMU a second-BMU nejsou sousední
neurony. Tento graf je vizuální ekvivalent: každé křížení čar odpovídá oblasti,
kde SOM topologii porušila. Graf umožňuje lokalizovat problematické oblasti mapy,
TE číslo dává globální souhrn.
