from __future__ import annotations

import argparse

from cuet_bst.ai_golden import generate_ai_silver_labels
from cuet_bst.settings import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI-assisted silver labels for the CUET BST human benchmark queue.")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logger = setup_logging("generate_ai_silver_labels", args.verbose)
    labels = generate_ai_silver_labels(limit=args.limit, batch_size=args.batch_size, overwrite=args.overwrite, logger=logger)
    logger.info("Wrote %s AI silver labels", len(labels))


if __name__ == "__main__":
    main()
