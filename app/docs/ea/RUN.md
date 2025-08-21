# Running the Evolutionary Algorithm

This document describes how to execute the evolutionary algorithm for SOM optimization using the `ea.py` script. You can specify an input data file and/or a configuration file, or rely on default settings.

## Basic Usage

Navigate to the project directory and run:

```bash
python3 ea.py
```

This will use the default configuration from `ea_config.py` and generate synthetic data according to the configuration parameters.

## Using a Custom Input Data File

To use your own CSV data file as input, specify the path with the `-i` or `--input` argument:

```bash
python3 ea.py -i /path/to/your/input.csv
```

If no input file is provided, the algorithm will generate synthetic data based on the parameters in the configuration.

## Using a Custom Configuration File

You can provide a custom configuration in JSON format with the `-c` or `--config` argument:

```bash
python3 ea.py -c /path/to/your/config.json
```

If no configuration file is specified, the script will attempt to use the default `ea_config.py` file.

## Combining Input and Configuration Files

You may specify both an input data file and a custom configuration file:

```bash
python3 ea.py -i /path/to/your/input.csv -c /path/to/your/config.json
```

## Argument Summary

- `-i`, `--input` &nbsp;&nbsp;&nbsp;&nbsp; Path to the input CSV data file (optional)
- `-c`, `--config` &nbsp;&nbsp; Path to the custom configuration JSON file (optional)

If neither argument is provided, the script will use default settings and generate synthetic data.

## Output Directory Structure

All results, logs, and generated files from each run are stored in a dedicated subdirectory under `results/`, named according to the current timestamp (e.g., `results/20240611_153045`). This ensures that outputs from different runs are separated and easily identifiable.

## Example Scenarios

- **Default run (synthetic data, default config):**
  ```bash
  python3 ea.py
  ```
- **Custom data, default config:**
  ```bash
  python3 ea.py -i data/mydata.csv
  ```
- **Synthetic data, custom config:**
  ```bash
  python3 ea.py -c config/myconfig.json
  ```
- **Custom data and custom config:**
  ```bash
  python3 ea.py -i data/mydata.csv -c config/myconfig.json
  ```

Refer to the documentation for details on configuration parameters and expected input data format.
