from __future__ import annotations

import argparse

from cuet_bst.ensemble_core import run_ensemble_classification
from cuet_bst.settings import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ensemble CUET BST NCERT/concept classification.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("classify_ensemble", args.verbose)
    run_ensemble_classification(logger)


if __name__ == "__main__":
    main()
