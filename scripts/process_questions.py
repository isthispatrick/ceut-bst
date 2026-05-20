from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from cuet_bst.classify import classify_questions
from cuet_bst.extract import extract_all
from cuet_bst.parser import empty_questions, normalize_question_frame, parse_extracted_texts
from cuet_bst.settings import MANUAL_IMPORT_DIR, PROCESSED_DIR, QUESTION_COLUMNS, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract, parse, classify, and export CUET Business Studies questions.")
    parser.add_argument("--skip-extract", action="store_true", help="Use existing data/processed/extraction_manifest.csv.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_logging("process_questions", args.verbose)
    if not args.skip_extract:
        extract_all(logger)
    parsed = parse_extracted_texts(logger)
    manual = load_manual_question_csvs(logger)
    if not manual.empty:
        parsed = pd.concat([parsed, manual], ignore_index=True)
        parsed = normalize_question_frame(parsed)
    if parsed.empty:
        parsed = empty_questions()
    questions = classify_questions(parsed, logger)
    logger.info("Wrote %s classified questions to %s", len(questions), PROCESSED_DIR / "questions.csv")


def load_manual_question_csvs(logger) -> pd.DataFrame:
    frames = []
    for path in sorted(MANUAL_IMPORT_DIR.glob("*.csv")):
        try:
            df = pd.read_csv(path).fillna("")
        except Exception as exc:
            logger.warning("Could not read manual CSV %s: %s", path, exc)
            continue
        if "question_text" not in df.columns:
            continue
        for column in QUESTION_COLUMNS:
            if column not in df.columns:
                df[column] = ""
        df["source"] = df["source"].replace("", f"manual:{path.name}") if "source" in df.columns else f"manual:{path.name}"
        df["raw_file_path"] = df["raw_file_path"].replace("", str(path)) if "raw_file_path" in df.columns else str(path)
        frames.append(df[QUESTION_COLUMNS])
    return pd.concat(frames, ignore_index=True) if frames else empty_questions()


if __name__ == "__main__":
    main()
