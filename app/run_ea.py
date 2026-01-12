#!/usr/bin/env python3
"""
Wrapper script to run EA.
Suppresses resource tracker warnings from Python 3.14+
"""
import sys
import os

# Suppress resource tracker warnings via environment variable
# (These are benign warnings from Python 3.14's stricter resource tracking)
os.environ.setdefault('PYTHONWARNINGS', 'ignore::UserWarning:multiprocessing.resource_tracker')

from ea import ea

if __name__ == '__main__':
    sys.exit(ea.main())