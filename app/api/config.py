from pathlib import Path
import os

DB_PATH = Path(os.environ.get("NEXUSOM_DB", "nexusom.db"))
DATA_ROOT = Path(os.environ.get("NEXUSOM_DATA", "data/datasets"))
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("API_PORT", "8000"))
