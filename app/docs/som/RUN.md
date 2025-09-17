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