#!/usr/bin/env python3
"""
CLI entry point for the analysis module.

Usage:
    python3 app/run_analysis.py -i data/datasets/LungCancerDataset/results/20260511_172536
"""
import argparse
import sys
import os

# Allow imports from app/ directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analysis.src.context import save_llm_context


def main():
    parser = argparse.ArgumentParser(description='Analyse SOM results and write llm_context.json')
    parser.add_argument('-i', '--input', required=True,
                        help='Path to SOM results directory (timestamped run dir)')
    args = parser.parse_args()

    results_dir = os.path.abspath(args.input)
    if not os.path.isdir(results_dir):
        print(f"Error: '{results_dir}' is not a directory", file=sys.stderr)
        return 1

    print(f"Analysing: {results_dir}")
    try:
        out_path = save_llm_context(results_dir)
        print(f"Written: {out_path}")
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
