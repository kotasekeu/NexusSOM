#!/usr/bin/env python3
"""Shortcut: .venv/bin/python3 run_ui.py"""
import subprocess, sys, os
from pathlib import Path

root = Path(__file__).parent
app  = root / 'app' / 'ui' / 'app.py'

subprocess.run([
    str(root / '.venv' / 'bin' / 'streamlit'),
    'run', str(app),
    '--server.headless', 'false',
], cwd=str(root))
