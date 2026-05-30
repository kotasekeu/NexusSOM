# generate_benchmark.py

Generátor syntetických benchmarkových datasetů pro ověření správnosti SOM.

Kód: [`app/tools/generate_benchmark.py`](../../app/tools/generate_benchmark.py)

---

## Účel

Standardní benchmarky z teorie neuronových map — mají **známý správný výsledek**, takže slouží jako validační nástroj implementace, ne jen jako vizualizace.

---

## Benchmarky

### Swiss Roll

3D data ležící na 2D manifoldu (spirálová plocha). Vnitřní dimenzionalita dat je 2D, ale body jsou navinuty do 3D prostoru jako piškotová roláda.

**Správně natrénovaná SOM** mapu rozbalí — neurony sledují závity spirály. Promítnuté do 2D dá rovný vyžehlený obdélník.

**Metriky správného výsledku:**
- Spearman ρ → 1.0 (globální topologie zachována)
- TE ≈ 0 (lokální sousedství neporušeno)

**Sloupce výstupu:** `id, x, y, z, t`  
`t` = unrolling parameter (ground truth pořadí podél spirály) — SOM ho nepoužívá k trénování, ale lze ho zobrazit jako barvu v topology grafech pro vizuální ověření rozbalení.

**Konfigurace:** 15×15 hex mapa (Vesantovo pravidlo: 5×√2000 ≈ 224 neuronů)

---

### Space-filling (Prostorová výplň)

Rovnoměrně rozmístěné 2D body v jednotkovém čtverci [0,1]². SOM mapa je 1D řetěz neuronů (1×N).

**Správně natrénovaná SOM** prochází čtvercem jako had bez křížení — aproximuje prostorově výplňovou křivku (podobnou Hilbertově nebo Peanově křivce).

**Metriky správného výsledku:**
- TE ≈ 0 (sousední neurony v řetězu mapují sousední oblasti prostoru)
- Vizuálně: žádná křížení v topology grafu

**Sloupce výstupu:** `id, x, y`

**Konfigurace:** 1×100 rect mapa (hex nemá smysl pro jeden řádek)

---

## Použití

```bash
# Oba benchmarky najednou (doporučeno)
python app/tools/generate_benchmark.py all

# Pouze Swiss Roll
python app/tools/generate_benchmark.py swiss-roll

# Pouze Space-filling
python app/tools/generate_benchmark.py space-filling
```

### Parametry

| Parametr | Default | Popis |
|----------|---------|-------|
| `--samples N` | 2000 / 1000 | Počet datových bodů |
| `--noise F` | 0.1 | Šum Swiss Roll (std Gaussiánu) |
| `--chain N` | auto | Délka řetězu Space-filling (default: samples/10, max 200) |
| `--seed N` | 42 | Random seed |

```bash
# Větší dataset s vyšším šumem
python app/tools/generate_benchmark.py swiss-roll --samples 5000 --noise 0.3

# Kratší řetěz
python app/tools/generate_benchmark.py space-filling --samples 2000 --chain 50
```

---

## Výstup

```
data/datasets/
  SwissRoll/
    swiss_roll.csv
    config-som.json     ← map_size automaticky dle Vesantova pravidla
  SpaceFilling/
    space_filling.csv
    config-som.json     ← map_size: [1, chain_len], map_type: rect
```

---

## Závislosti

- `sklearn.datasets.make_swiss_roll` (scikit-learn)
- `numpy`, `pandas`
