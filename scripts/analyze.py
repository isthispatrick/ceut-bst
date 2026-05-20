from __future__ import annotations

import argparse

from cuet_bst.analyze_core import run_analysis
from cuet_bst.settings import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CUET Business Studies frequency, repetition, and report outputs.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("analyze", args.verbose)
    outputs = run_analysis(logger)
    for name, path in outputs.items():
        logger.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
