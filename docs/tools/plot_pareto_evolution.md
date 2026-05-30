# plot_pareto_evolution.py

Vizualizace vývoje Pareto fronty přes generace EA.

Kód: [`app/tools/plot_pareto_evolution.py`](../../app/tools/plot_pareto_evolution.py)

---

## Účel

Zobrazuje jak se Pareto archiv EA zlepšoval generaci po generaci. Vstupem je `pareto_front.csv` z EA běhu.

---

## Výstupy

### `pareto_evolution.png` — 2D přehled (5 panelů)

| Panel | Obsah |
|-------|-------|
| Archive size | Počet řešení v archivu per generace, barevně feasible/infeasible |
| MQE quality | Vývoj best/median/worst `raw_mqe_ratio` přes generace |
| MQE vs TE | Scatter 2D projekce Pareto fronty |
| MQE vs (1−ρ) | Scatter — MQE vs topologická korelace |
| TE vs (1−ρ) | Scatter — TE vs topologická korelace |

### `pareto_3d.png` — 3D scatter

Všechna tři Pareto cíle najednou: `raw_mqe_ratio × raw_te × (1−ρ)`. Shadow projekce na třech stěnách (MQE×TE, MQE×(1−ρ), TE×(1−ρ)). Orthografická projekce.

---

## Použití

```bash
# Základní
python app/tools/plot_pareto_evolution.py data/datasets/Iris/results/<run>/

# Jen 2D graf
python app/tools/plot_pareto_evolution.py <results_dir> --no3d

# Vlastní výstupní soubory
python app/tools/plot_pareto_evolution.py <results_dir> \
  --output my_2d.png \
  --output3d my_3d.png

# Export statistik do CSV
python app/tools/plot_pareto_evolution.py <results_dir> --csv

# Fixní osy [0,1] pro srovnání více běhů
python app/tools/plot_pareto_evolution.py <results_dir> --fixed-range
```

### Parametry

| Parametr | Popis |
|----------|-------|
| `results_dir` | Složka EA běhu (obsahuje `pareto_front.csv`) |
| `--output` / `-o` | Cesta pro 2D PNG (default: `results_dir/pareto_evolution.png`) |
| `--output3d` | Cesta pro 3D PNG (default: `results_dir/pareto_3d.png`) |
| `--no3d` | Přeskočit 3D graf |
| `--guide-lines` | Čáry z každého bodu ke stěnám v 3D |
| `--space-grid N` | N dělení na každé ose 3D |
| `--lattice` | Mřížka v 3D prostoru |
| `--elev` | Elevace pohledu 3D (default: 22°) |
| `--azim` | Azimut pohledu 3D (default: 225°) |
| `--fixed-range` | Fixní osy [0, 1] pro cross-run srovnání |
| `--csv` / `-c` | Export per-generace statistik do CSV |

---

## Barevné kódování

- **Barva bodu** = generace (viridis: tmavá = raná generace, světlá = pozdní)
- **Kruh** = feasible řešení
- **×** = infeasible řešení (porušuje constraint)
- **Červená hvězda** = finální archiv (poslední generace)

---

## Vstupy

`pareto_front.csv` — generováno automaticky EA při každém běhu.  
Vyžaduje sloupce: `generation`, `uid`, `raw_mqe_ratio`, `raw_te`, `raw_topo_corr`, `constraint_violation`, `is_penalized`, `map_m`, `map_n`.

---

## Zpětná kompatibilita

Pro starší běhy bez `raw_topo_corr` (před issue #85) se automaticky použije `dead_ratio` jako fallback pro třetí dimenzi.
