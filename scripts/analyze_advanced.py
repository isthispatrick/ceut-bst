from __future__ import annotations

import argparse

from cuet_bst.advanced_analysis import run_advanced_analysis
from cuet_bst.settings import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate advanced CUET BST exam-intelligence outputs.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("analyze_advanced", args.verbose)
    outputs = run_advanced_analysis(logger)
    for name, path in outputs.items():
        logger.info("%s -> %s", name, path)


if __name__ == "__main__":
    main()
