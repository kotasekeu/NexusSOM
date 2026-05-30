# Running the SOM Algorithm

This document describes how to execute a single run of the SOM algorithm using the `run.py` script. Both an input data file and a configuration file are required for execution.

## Basic Usage

To run the algorithm, you must provide paths to your input data and a JSON configuration file. Navigate to the project's root directory and use the following command structure:

```bash
python3 run_som.py -i /path/to/your/data.csv -c /path/to/your/config.json
```

## Command-Line Arguments

The script requires the following arguments:

*   `-i`, `--input`: **(Required)** The path to the input CSV data file.
*   `-c`, `--config`: **(Required)** The path to the JSON configuration file that defines all hyperparameters for the run.
*   `-o`, `--output`: **(Optional)** The path to the directory where all results will be saved. If omitted, a new timestamped directory will be created automatically.

### Argument Summary

| Argument | Full Name | Required? | Description |
| :--- | :--- | :--- | :--- |
| `-i` | `--input` | **Yes** | Path to the input CSV data file. |
| `-c` | `--config` | **Yes** | Path to the JSON configuration file. |
| `-o` | `--output` | No | Path to the output directory. |

## Example Scenarios

- **Standard Run (Automatic Output Directory):**
  This command will process `wine.csv` using settings from `wine-config.json` and save the results into a new, timestamped directory (e.g., `results/20240918_103000/`).

  ```bash
  python3 run_som.py -i data/wine.csv -c config/wine-config.json
  ```

- **Run with a Specified Output Directory:**
  This command does the same as above, but saves all results into the specific folder `my-wine-analysis/`.

  ```bash
  python3 run_som.py -i data/wine.csv -c config/wine-config.json -o results/my-wine-analysis
  ```

---

## Output Structure

All results, logs, generated files, and visualizations from each run are stored in a dedicated output directory. This ensures that all artifacts from a single experiment are self-contained and reproducible.

For a detailed description of the generated files and the directory structure, please refer to the **[Results Guide (RESULTS.md)](../RESULTS.md)**.

For details on how to format the configuration file, see the **[Configuration Guide (CONFIG.md)](./CONFIG.md)**.

---

## Kompletní pipeline — od dat po report

Všechny příkazy se spouštějí z **root adresáře projektu** (`/NexusSom/`).  
Python interpreter: `.venv/bin/python3`

---

### Krok 0 — Generování datasetu (pokud nemáme reálná data)

```bash
# Benchmarkové datasety (Swiss Roll + Space-filling)
.venv/bin/python3 app/tools/generate_benchmark.py all

# Pracovní cesty s injektovanými anomáliemi
.venv/bin/python3 app/tools/generate_trips_dataset.py \
  --rows 800 --inject both --seed 42 \
  --output data/datasets/Trips/dataset
```

Výstup: `data/datasets/SwissRoll/`, `data/datasets/SpaceFilling/`, `data/datasets/Trips/`  
Každá složka obsahuje CSV + `config-som.json`.

---

### Krok 1 — Trénování SOM

```bash
.venv/bin/python3 app/run_som.py \
  -i data/datasets/SwissRoll/swiss_roll.csv \
  -c data/datasets/SwissRoll/config-som.json
```

Výstup: `data/datasets/SwissRoll/results/<timestamp>/`

```
results/<timestamp>/
  csv/
    weights.npy               ← natrénované váhy (m, n, dim)
    training_data.npy         ← normalizovaná vstupní data
    sample_assignments.csv    ← přiřazení vzorků k neuronům + per-dim QE
    ignore_mask.csv           ← maska maskovaných dimenzí
    training_data_readable.csv
    weights_readable.csv
  maps_dataset/               ← U-matrix, hitmap, distance map, ...
  run_metrics.json            ← MQE, TE, duration, map_topology
  dataset_meta.json           ← metadata datasetu
  log.txt
```

---

### Krok 2 — Analýza výsledků

Generuje `llm_context.json` (vstupy pro LLM report) a detekuje anomálie.

```bash
.venv/bin/python3 app/run_analysis.py \
  -i data/datasets/SwissRoll/results/<timestamp>/
```

Výstup (přidá do výsledkové složky):
- `llm_context.json` — strukturovaný kontext pro LLM
- `anomalies.json` — detekované anomálie
- `quantization_errors.json` — QE per neuron

---

### Krok 3 — Topologické grafy

```bash
# 2D PCA (default)
.venv/bin/python3 app/tools/plot_som_topology.py \
  data/datasets/SwissRoll/results/<timestamp>/

# 2D + 3D + interaktivní HTML
.venv/bin/python3 app/tools/plot_som_topology.py \
  data/datasets/SwissRoll/results/<timestamp>/ \
  --3d --html

# UMAP projekce
.venv/bin/python3 app/tools/plot_som_topology.py \
  data/datasets/SwissRoll/results/<timestamp>/ \
  --projection umap --3d --html
```

Výstup (do `maps_dataset/`):
- `topology_2d_pca.png`
- `topology_3d_pca.png`
- `topology_interactive_pca.html`

---

### Krok 4 — Per-dimenzní QE heatmapy

```bash
.venv/bin/python3 app/tools/plot_dim_qe.py \
  data/datasets/SwissRoll/results/<timestamp>/
```

Výstup (do `maps_dataset/`):
- `dim_qe_01_<nejhorší_dimenze>.png` ... `dim_qe_N_<dimenze>.png`
- `dim_qe_dominant.png`

---

### Krok 5 — LLM report

Vyžaduje běžící Ollama server (`http://localhost:11434`).

```bash
# Textový report
.venv/bin/python3 app/run_llm.py \
  -i data/datasets/SwissRoll/results/<timestamp>/ \
  -m report

# PDF report
.venv/bin/python3 app/run_llm.py \
  -i data/datasets/SwissRoll/results/<timestamp>/ \
  -m pdf

# Interaktivní chat o výsledcích
.venv/bin/python3 app/run_llm.py \
  -i data/datasets/SwissRoll/results/<timestamp>/ \
  -m chat

# Vlastní model
.venv/bin/python3 app/run_llm.py \
  -i data/datasets/SwissRoll/results/<timestamp>/ \
  -m report \
  --model llama3.1:8b \
  --url http://localhost:11434
```

---

### Kompletní pipeline — jeden blok (copy-paste)

```bash
DS=SwissRoll
CSV=data/datasets/$DS/swiss_roll.csv
CFG=data/datasets/$DS/config-som.json

# 0. Generovat data (pokud neexistuje)
[ ! -f "$CSV" ] && .venv/bin/python3 app/tools/generate_benchmark.py swiss-roll

# 1. Trénink SOM
.venv/bin/python3 app/run_som.py -i $CSV -c $CFG

# Zjistit cestu k výsledkům
RUN=$(ls -td data/datasets/$DS/results/*/ | head -1)
echo "Results: $RUN"

# 2. Analýza
.venv/bin/python3 app/run_analysis.py -i $RUN

# 3. Topologie
.venv/bin/python3 app/tools/plot_som_topology.py $RUN --3d --html

# 4. Per-dim QE
.venv/bin/python3 app/tools/plot_dim_qe.py $RUN

# 5. LLM report
.venv/bin/python3 app/run_llm.py -i $RUN -m report
```

---

### EA optimalizace hyperparametrů (volitelné)

```bash
.venv/bin/python3 app/run_ea.py \
  -i data/datasets/SwissRoll/swiss_roll.csv \
  -c data/datasets/SwissRoll/config-ea.json
```

Po dokončení EA vizualizovat Pareto frontu:

```bash
EA_RUN=data/datasets/SwissRoll/results/<ea_timestamp>/seed_42/
.venv/bin/python3 app/tools/plot_pareto_evolution.py $EA_RUN
```

Výstup: `pareto_evolution.png`, `pareto_3d.png`