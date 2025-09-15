import os
from datetime import datetime
import json

def log_message(working_dir: str, level: str, message: str) -> None:
    """
    Log a message to the log file in the specified working directory.
    """
    log_path = os.path.join(working_dir, "log.txt")
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{now}] [{level}] {message}\n")

def load_configuration(json_path: str) -> dict:
    """
    Load configuration from a JSON file.
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Configuration file '{json_path}' not found.")
    try:
        with open(json_path, 'r', encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from '{json_path}': {e}")

def get_working_directory(input_file: str = None) -> str:
    """
    Create and return a working directory for results, named by timestamp.
    This function is primarily for run.py's internal logic,
    but can be here if needed elsewhere.
    """
    if input_file:
        base_dir = os.path.dirname(os.path.abspath(input_file))
    else:
        base_dir = os.getcwd()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = os.path.join(base_dir, "results", f"{timestamp}")

    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir