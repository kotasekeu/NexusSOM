# plot_dim_qe.py

Heatmapy per-dimenzní kvantizační chyby (QE) na SOM mapě.

Kód: [`app/tools/plot_dim_qe.py`](../../app/tools/plot_dim_qe.py)

---

## Účel

Zobrazuje kde na mapě má SOM problém s konkrétní dimenzí (featurou). Doplňuje celkové QE o pohled na jednotlivé dimenze — odpovídá na otázku "proč je tento vzorek anomální" v prostorové perspektivě celé mapy.

---

## Varianty výstupu

### Varianta A — individuální heatmapy (`dim_qe_NN_<název>.png`)

Jeden soubor na každou dimenzi, seřazeny od nejvyšší průměrné QE (rank #1 = nejproblematičtější dimenze). Každý neuron obarven intenzitou průměrné `|x_d − w_d|` pro vzorky přiřazené k tomuto neuronu.

- **žlutá** = nízká chyba
- **červená** = vysoká chyba
- **bílá** = neuron bez přiřazených vzorků

### Varianta B — dominantní dimenze (`dim_qe_dominant.png`)

Jeden graf. Každý neuron obarven dle dimenze s **nejvyšší** průměrnou QE; intenzita barvy ∝ velikost té chyby. Ukazuje prostorovou strukturu: "v tomto rohu mapy dominuje feature X, tam feature Y".

- **šedé pozadí** + **šedé neurony bez vzorků** = splývají záměrně (vizuálně "prázdná místa")
- Paleta `hsv` — plné spektrum barev pro maximální rozlišitelnost dimenzí

---

## Použití

```bash
# Obě varianty (default)
python app/tools/plot_dim_qe.py data/datasets/Iris/results/<run>/

# Jen dominantní mapa
python app/tools/plot_dim_qe.py <results_dir> --no-a

# Jen individuální heatmapy
python app/tools/plot_dim_qe.py <results_dir> --no-b

# Vlastní výstupní složka
python app/tools/plot_dim_qe.py <results_dir> --output-dir /tmp/qe_maps
```

### Parametry

| Parametr | Popis |
|----------|-------|
| `results_dir` | Cesta k výsledkům SOM běhu (složka s `csv/`) |
| `--no-a` | Přeskočit Variantu A (individuální heatmapy) |
| `--no-b` | Přeskočit Variantu B (dominantní dimenze) |
| `--output-dir` | Výstupní složka (default: `results_dir/maps_dataset/`) |

---

## Vstupy (načítá automaticky)

| Soubor | Odkud |
|--------|-------|
| `csv/weights.npy` | Váhy SOM `(m, n, dim)` |
| `csv/sample_assignments.csv` | Přiřazení vzorků + `qe_dim_*` sloupce |
| `run_metrics.json` | Typ topologie (hex / rect) |

Dimenze se čtou přímo z hlavičky `sample_assignments.csv` — sloupce `qe_dim_<název>`. Primární ID sloupec (`id`) je automaticky vyloučen (není v `qe_dim_*`).

---

## Výstupní soubory

```
maps_dataset/
  dim_qe_01_<nejhorší_dim>.png
  dim_qe_02_<druhá_nejhorší>.png
  ...
  dim_qe_dominant.png
```

Číslo v názvu = pořadí dle globální průměrné QE (1 = největší problém).

---

## Poznámky

- Funguje pro hex i čtvercovou topologii
- Škáluje na libovolný počet dimenzí (30 dimenzí = 30 souborů)
- Neurony bez přiřazených vzorků (dead neurons) jsou vizuálně odlišeny: bílé v Var. A, šedé v Var. B
