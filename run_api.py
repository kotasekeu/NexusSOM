#!/usr/bin/env python3
"""Start the NexusSom FastAPI server.

Usage:
  .venv/bin/python3 run_api.py
  .venv/bin/python3 run_api.py --port 8001
  .venv/bin/python3 run_api.py --no-reload
"""
import argparse
import subprocess
import sys
from pathlib import Path

root = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=8000)
parser.add_argument("--no-reload", action="store_true")
args = parser.parse_args()

cmd = [
    str(root / ".venv" / "bin" / "uvicorn"),
    "app.api.main:app",
    "--host", args.host,
    "--port", str(args.port),
]
if not args.no_reload:
    cmd.append("--reload")

print(f"API:     http://{args.host}:{args.port}")
print(f"Swagger: http://{args.host}:{args.port}/docs")
print(f"ReDoc:   http://{args.host}:{args.port}/redoc")

subprocess.run(cmd, cwd=str(root))
