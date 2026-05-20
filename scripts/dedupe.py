from __future__ import annotations

import argparse

from cuet_bst.dedupe_core import run_deduplication
from cuet_bst.settings import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Create canonical CUET BST question IDs and duplicate reports.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("dedupe", args.verbose)
    summary = run_deduplication(logger)
    logger.info("raw_rows=%s unique_questions=%s duplicate_rows=%s", summary["raw_rows"], summary["unique_questions"], summary["duplicate_rows"])


if __name__ == "__main__":
    main()
