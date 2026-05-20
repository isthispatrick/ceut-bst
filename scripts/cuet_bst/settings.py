from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_IMPORT_DIR = DATA_DIR / "manual_imports"
MANUAL_OFFICIAL_DIR = DATA_DIR / "manual_official_papers"
VERIFIED_DIR = DATA_DIR / "verified"
REPORTS_DIR = ROOT / "reports"
LOG_DIR = DATA_DIR / "logs"
DB_PATH = PROCESSED_DIR / "cuet_bst.duckdb"


def load_env_file(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs without overriding already-set variables."""
    env_path = path or (ROOT / ".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file()

QUESTION_COLUMNS = [
    "source",
    "year",
    "date",
    "shift",
    "set_name",
    "question_id",
    "question_number",
    "question_text",
    "passage_text",
    "case_text",
    "option_a",
    "option_b",
    "option_c",
    "option_d",
    "correct_option",
    "answer_source",
    "question_type",
    "difficulty_estimate",
    "unit",
    "chapter",
    "subtopic",
    "ncert_keyword_match",
    "repeated_concept_cluster",
    "confidence_score",
    "source_url",
    "raw_file_path",
    "needs_review",
]

ADVANCED_COLUMNS = QUESTION_COLUMNS + [
    "canonical_question_id",
    "duplicate_group_size",
    "duplicate_source_urls",
    "source_tier",
    "source_weight",
    "year_weight",
    "weighted_frequency_score",
    "recency_weighted_score",
    "ncert_heading",
    "ncert_page",
    "ncert_paragraph",
    "ncert_concept",
    "concept_type",
    "question_pattern",
    "micro_concept",
    "micro_concept_confidence",
    "micro_concept_note",
    "rule_label",
    "embedding_label",
    "bm25_label",
    "llm_label",
    "final_confidence",
    "review_reason",
]


def ensure_dirs() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MANUAL_IMPORT_DIR, MANUAL_OFFICIAL_DIR, VERIFIED_DIR, REPORTS_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_logging(name: str = "cuet_bst", verbose: bool = False) -> logging.Logger:
    ensure_dirs()
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    stream.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(LOG_DIR / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    return logger


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
