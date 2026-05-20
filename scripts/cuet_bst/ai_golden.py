from __future__ import annotations

import json
import logging
import os
from typing import Any

import pandas as pd

from .llm_client import chat_completion, configured_model, parse_json_array
from .settings import CONFIG_DIR, PROCESSED_DIR, VERIFIED_DIR, ensure_dirs, load_json


HUMAN_GOLD_QUEUE = PROCESSED_DIR / "human_gold_review_queue.csv"
SILVER_LABELS = VERIFIED_DIR / "silver_labels_ai.csv"
LEGACY_AI_GOLDEN = VERIFIED_DIR / "golden_labels_ai.csv"
GOLDEN_LABELS = VERIFIED_DIR / "golden_labels.csv"


def generate_ai_silver_labels(
    *,
    limit: int = 300,
    batch_size: int = 5,
    overwrite: bool = False,
    logger: logging.Logger | None = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    ensure_dirs()
    if not HUMAN_GOLD_QUEUE.exists():
        raise FileNotFoundError("Missing data/processed/human_gold_review_queue.csv. Run python scripts/analyze_advanced.py first.")
    queue = pd.read_csv(HUMAN_GOLD_QUEUE).fillna("").head(limit)
    existing = _existing_labels()
    existing_ids = set(existing["canonical_question_id"].astype(str)) if not existing.empty and not overwrite else set()
    rows: list[dict[str, Any]] = existing.to_dict("records") if not overwrite else []
    todo = queue[~queue["canonical_question_id"].astype(str).isin(existing_ids)].copy()
    taxonomy = _compact_taxonomy()
    logger.info("Generating AI silver labels for %s questions with %s", len(todo), configured_model())

    for start in range(0, len(todo), batch_size):
        batch = todo.iloc[start : start + batch_size]
        try:
            labels = _label_batch(batch, taxonomy)
        except Exception as exc:
            logger.warning("AI batch failed at rows %s-%s: %s; using current predictions", start + 1, start + len(batch), exc)
            labels = [_fallback_label(row) for _, row in batch.iterrows()]
        rows.extend(labels)
        _write_outputs(pd.DataFrame(rows))
        logger.info("AI silver labels saved: %s/%s", min(start + batch_size, len(todo)), len(todo))

    out = pd.DataFrame(rows).drop_duplicates("canonical_question_id", keep="last")
    _write_outputs(out)
    return out


def _label_batch(batch: pd.DataFrame, taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    payload = []
    for _, row in batch.iterrows():
        payload.append(
            {
                "canonical_question_id": row.get("canonical_question_id", ""),
                "question_text": str(row.get("question_text", ""))[:1800],
                "options": {
                    "A": row.get("option_a", ""),
                    "B": row.get("option_b", ""),
                    "C": row.get("option_c", ""),
                    "D": row.get("option_d", ""),
                },
                "current_prediction": {
                    "chapter": row.get("chapter", ""),
                    "subtopic": row.get("subtopic", ""),
                    "micro_concept": row.get("micro_concept", ""),
                    "question_type": row.get("question_type", "") or row.get("question_pattern", ""),
                    "difficulty": row.get("difficulty_estimate", ""),
                    "correct_option": row.get("correct_option", ""),
                },
            }
        )
    prompt = {
        "task": "Create AI-assisted silver audit labels for CUET UG Business Studies questions. Use official CUET BST/NCERT-style taxonomy. Return JSON array only.",
        "rules": [
            "Do not invent a correct option if the question/options are insufficient; keep the current correct_option or blank.",
            "Use one of the provided chapter/subtopic names where possible.",
            "question_type should be one of definition-based, feature-identification, principle-identification, case-study diagnosis, match-the-following, assertion-reason, statement true/false, chronology/process order, concept-to-example, example-to-concept.",
            "difficulty should be easy, medium, or hard.",
        ],
        "taxonomy": taxonomy,
        "questions": payload,
        "schema": [
            {
                "canonical_question_id": "string",
                "chapter": "string",
                "subtopic": "string",
                "micro_concept": "string",
                "question_type": "string",
                "difficulty_estimate": "easy|medium|hard",
                "correct_option": "A|B|C|D|blank",
                "ai_confidence": "0-1",
                "verification_source": "ai_silver",
                "notes": "short reason",
            }
        ],
    }
    content = chat_completion(
        [
            {"role": "system", "content": "You are an NCERT Business Studies PYQ auditor. Return strict JSON only."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
        max_tokens=max(900, int(os.getenv("CUET_GOLDEN_MAX_TOKENS", "1800"))),
        timeout=60,
    )
    parsed = parse_json_array(content)
    by_id = {str(item.get("canonical_question_id", "")): item for item in parsed}
    rows = []
    for _, original in batch.iterrows():
        cid = str(original.get("canonical_question_id", ""))
        item = by_id.get(cid) or _fallback_label(original)
        rows.append(_clean_label(item, original))
    return rows


def _clean_label(item: dict[str, Any], original: pd.Series) -> dict[str, Any]:
    return {
        "canonical_question_id": str(item.get("canonical_question_id", original.get("canonical_question_id", ""))),
        "chapter": str(item.get("chapter", original.get("chapter", ""))),
        "subtopic": str(item.get("subtopic", original.get("subtopic", ""))),
        "micro_concept": str(item.get("micro_concept", original.get("micro_concept", ""))),
        "question_type": str(item.get("question_type", original.get("question_type", "") or original.get("question_pattern", ""))),
        "difficulty_estimate": str(item.get("difficulty_estimate", original.get("difficulty_estimate", ""))).lower(),
        "correct_option": str(item.get("correct_option", original.get("correct_option", ""))).upper().strip(),
        "ai_confidence": str(item.get("ai_confidence", item.get("confidence", ""))),
        "verification_source": str(item.get("verification_source", "ai_silver")) or "ai_silver",
        "notes": str(item.get("notes", ""))[:300],
    }


def _fallback_label(row: pd.Series) -> dict[str, Any]:
    return {
        "canonical_question_id": row.get("canonical_question_id", ""),
        "chapter": row.get("chapter", ""),
        "subtopic": row.get("subtopic", ""),
        "micro_concept": row.get("micro_concept", ""),
        "question_type": row.get("question_type", "") or row.get("question_pattern", ""),
        "difficulty_estimate": row.get("difficulty_estimate", ""),
        "correct_option": row.get("correct_option", ""),
        "ai_confidence": row.get("final_confidence", ""),
        "verification_source": "ai_silver_fallback_current_prediction",
        "notes": "Fallback to current pipeline label because AI audit call failed.",
    }


def _existing_labels() -> pd.DataFrame:
    if SILVER_LABELS.exists():
        return pd.read_csv(SILVER_LABELS).fillna("")
    if LEGACY_AI_GOLDEN.exists():
        return pd.read_csv(LEGACY_AI_GOLDEN).fillna("")
    return pd.DataFrame()


def _write_outputs(labels: pd.DataFrame) -> None:
    VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
    labels.to_csv(SILVER_LABELS, index=False)


def _compact_taxonomy() -> dict[str, list[str]]:
    taxonomy = load_json(CONFIG_DIR / "cuet_topics.json")
    return {unit["chapter"]: list(unit["subtopics"].keys()) for unit in taxonomy.get("units", [])}


def generate_ai_golden_labels(**kwargs) -> pd.DataFrame:
    return generate_ai_silver_labels(**kwargs)
