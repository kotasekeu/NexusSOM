#!/usr/bin/env python3
"""Start API + Streamlit UI together.

Usage:
  .venv/bin/python3 run_all.py
  .venv/bin/python3 run_all.py --api-port 8001 --ui-port 8502

Ctrl+C stops both processes.
"""
import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

root = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--api-host", default="127.0.0.1")
parser.add_argument("--api-port", type=int, default=8000)
parser.add_argument("--ui-port",  type=int, default=8501)
parser.add_argument("--no-reload", action="store_true", help="Disable uvicorn auto-reload")
args = parser.parse_args()

api_cmd = [
    str(root / ".venv" / "bin" / "uvicorn"),
    "app.api.main:app",
    "--host", args.api_host,
    "--port", str(args.api_port),
]
if not args.no_reload:
    api_cmd.append("--reload")

ui_cmd = [
    str(root / ".venv" / "bin" / "streamlit"),
    "run", str(root / "app" / "ui" / "app.py"),
    "--server.port", str(args.ui_port),
    "--server.headless", "false",
]

print("=" * 50)
print(f"  API:     http://{args.api_host}:{args.api_port}")
print(f"  Swagger: http://{args.api_host}:{args.api_port}/docs")
print(f"  UI:      http://localhost:{args.ui_port}")
print("  Ctrl+C to stop both")
print("=" * 50)

procs = []
try:
    procs.append(subprocess.Popen(api_cmd, cwd=str(root)))
    time.sleep(1)   # give API a moment to bind port before UI starts
    procs.append(subprocess.Popen(ui_cmd,  cwd=str(root)))

    for p in procs:
        p.wait()

except KeyboardInterrupt:
    print("\nShutting down...")
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    print("Done.")
