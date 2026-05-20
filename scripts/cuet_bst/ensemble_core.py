from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .intelligence import (
    apply_manual_labels,
    build_ncert_index,
    classify_question_pattern,
    combine_labels,
    embedding_best,
    identify_micro_concept,
    identify_micro_concept_with_confidence,
    infer_concept_type,
    llm_label,
    make_question_text,
    rule_label,
    tfidf_best,
)
from .settings import ADVANCED_COLUMNS, PROCESSED_DIR


LLM_CACHE_PATH = PROCESSED_DIR / "llm_classification_cache.jsonl"


def run_ensemble_classification(logger: logging.Logger | None = None) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    input_path = PROCESSED_DIR / "questions_canonical.csv"
    if not input_path.exists():
        raise FileNotFoundError("Missing data/processed/questions_canonical.csv. Run python scripts/dedupe.py first.")
    df = pd.read_csv(input_path).fillna("")
    if df.empty:
        return _write_empty()

    ncert = build_ncert_index()
    question_texts = [make_question_text(row) for _, row in df.iterrows()]
    bm25_labels = tfidf_best(question_texts, ncert)
    embedding_labels = embedding_best(question_texts, ncert)
    llm_cache = _read_llm_cache()
    llm_max_calls = int(os.getenv("CUET_LLM_MAX_CALLS", "100"))
    llm_calls = 0
    rows = []

    for idx, (_, row) in enumerate(df.iterrows()):
        record = row.to_dict()
        text = question_texts[idx]
        rule = rule_label(text)
        bm25 = bm25_labels[idx]
        embedding = embedding_labels[idx]
        cache_key = str(record.get("canonical_question_id", ""))
        if cache_key in llm_cache:
            llm = llm_cache[cache_key]
        elif _should_call_llm(rule, embedding, bm25) and llm_calls < llm_max_calls:
            llm = llm_label(text, [rule, embedding, bm25])
            llm_calls += 1
            if llm.score > 0:
                llm_cache[cache_key] = llm
                _append_llm_cache(cache_key, llm)
        else:
            best = max([rule, embedding, bm25], key=lambda item: item.score)
            llm = type(best)(best.label, best.chapter, best.subtopic, 0.0, best.heading, best.concept)

        final, confidence, review_reason = combine_labels(rule, embedding, bm25, llm)
        heading, paragraph, concept, concept_type = _best_ncert_fields(final, bm25, embedding, ncert)
        pattern = classify_question_pattern(text)
        difficulty = str(record.get("difficulty_estimate", "")).strip() or estimate_fallback_difficulty(record, pattern)
        micro_concept, micro_confidence, micro_note = identify_micro_concept_with_confidence(text, final.chapter, final.subtopic)

        record.update(
            {
                "unit": _unit_for(final.chapter, ncert),
                "chapter": final.chapter or record.get("chapter", ""),
                "subtopic": final.subtopic or record.get("subtopic", ""),
                "ncert_heading": heading,
                "ncert_page": "",
                "ncert_paragraph": paragraph,
                "ncert_concept": concept,
                "concept_type": concept_type,
                "question_type": pattern,
                "question_pattern": pattern,
                "difficulty_estimate": difficulty,
                "micro_concept": micro_concept,
                "micro_concept_confidence": f"{micro_confidence:.2f}",
                "micro_concept_note": micro_note,
                "rule_label": rule.label,
                "embedding_label": embedding.label,
                "bm25_label": bm25.label,
                "llm_label": llm.label if llm.score else "skipped_or_unavailable",
                "confidence_score": f"{confidence:.2f}",
                "final_confidence": f"{confidence:.2f}",
                "needs_review": "true" if review_reason else "false",
                "review_reason": review_reason,
                "repeated_concept_cluster": f"{final.chapter} :: {final.subtopic} :: {micro_concept}",
            }
        )
        rows.append(record)

    out = pd.DataFrame(rows)
    for column in ADVANCED_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    out = out[ADVANCED_COLUMNS + [column for column in out.columns if column not in ADVANCED_COLUMNS]]
    out = apply_manual_labels(out)
    out.to_csv(PROCESSED_DIR / "questions_advanced.csv", index=False)
    out.to_json(PROCESSED_DIR / "questions_advanced.json", orient="records", indent=2, force_ascii=False)
    _write_ncert_map(out)
    _write_patterns(out)
    _write_ai_review_suggestions(out)
    logger.info("Wrote %s ensemble-classified canonical questions; LLM calls this run=%s", len(out), llm_calls)
    return out


def _should_call_llm(rule, embedding, bm25) -> bool:
    mode = os.getenv("CUET_LLM_MODE", "ambiguous").strip().lower()
    if mode in {"off", "none", "false", "0"}:
        return False
    if mode in {"heavy", "all"}:
        return True
    labels = {(item.chapter, item.subtopic) for item in [rule, embedding, bm25] if item.chapter}
    return len(labels) > 1 or max(rule.score, embedding.score, bm25.score) < 0.75


def estimate_fallback_difficulty(record: dict, pattern: str) -> str:
    from .classify import estimate_difficulty

    return estimate_difficulty(make_question_text(record), pattern)


def _best_ncert_fields(final, bm25, embedding, ncert: pd.DataFrame) -> tuple[str, str, str, str]:
    candidates = [bm25, embedding]
    chosen = next((item for item in candidates if item.chapter == final.chapter and item.subtopic == final.subtopic), None)
    if chosen and chosen.heading:
        return chosen.heading, "", chosen.concept, infer_concept_type(f"{chosen.heading} {chosen.concept}")
    matches = ncert[(ncert["chapter"] == final.chapter) & (ncert["subtopic"] == final.subtopic)]
    if matches.empty:
        return final.subtopic.title(), "", "", infer_concept_type(final.subtopic)
    row = matches.iloc[0]
    return str(row["ncert_heading"]), str(row.get("ncert_paragraph", "")), str(row["ncert_concept"]), str(row["concept_type"])


def _unit_for(chapter: str, ncert: pd.DataFrame) -> str:
    matches = ncert[ncert["chapter"] == chapter]
    return str(matches.iloc[0]["unit"]) if not matches.empty else ""


def _write_ncert_map(df: pd.DataFrame) -> None:
    columns = [
        "canonical_question_id",
        "chapter",
        "subtopic",
        "ncert_heading",
        "ncert_page",
        "ncert_paragraph",
        "ncert_concept",
        "concept_type",
        "question_pattern",
        "micro_concept",
        "micro_concept_confidence",
        "micro_concept_note",
        "final_confidence",
        "needs_review",
    ]
    df[columns].to_csv(PROCESSED_DIR / "question_ncert_map.csv", index=False)


def _write_patterns(df: pd.DataFrame) -> None:
    df[
        [
            "canonical_question_id",
            "chapter",
            "subtopic",
            "question_pattern",
            "micro_concept",
            "micro_concept_confidence",
            "micro_concept_note",
            "difficulty_estimate",
            "final_confidence",
        ]
    ].to_csv(PROCESSED_DIR / "question_patterns.csv", index=False)
    pattern_by_topic = (
        df.groupby(["chapter", "subtopic", "question_pattern"], dropna=False)
        .size()
        .reset_index(name="question_count")
        .sort_values("question_count", ascending=False)
    )
    pattern_by_topic.to_csv(PROCESSED_DIR / "pattern_by_topic.csv", index=False)
    clusters = (
        df.groupby(["chapter", "subtopic", "micro_concept"], dropna=False)
        .agg(
            question_count=("canonical_question_id", "count"),
            weighted_score=("weighted_frequency_score", "sum"),
            recency_score=("recency_weighted_score", "sum"),
            avg_micro_concept_confidence=("micro_concept_confidence", lambda values: pd.to_numeric(values, errors="coerce").mean()),
            example_question=("question_text", "first"),
        )
        .reset_index()
        .sort_values(["weighted_score", "question_count"], ascending=False)
    )
    clusters.to_csv(PROCESSED_DIR / "micro_concept_clusters.csv", index=False)


def _write_ai_review_suggestions(df: pd.DataFrame) -> None:
    if "needs_review" not in df.columns:
        pd.DataFrame().to_csv(PROCESSED_DIR / "ai_manual_review_suggestions.csv", index=False)
        return
    review = df[df["needs_review"].astype(str).str.lower().isin(["true", "yes"])].copy()
    if review.empty:
        pd.DataFrame(
            columns=[
                "canonical_question_id",
                "ai_suggested_chapter",
                "ai_suggested_subtopic",
                "ai_suggested_question_pattern",
                "ai_suggested_difficulty",
                "ai_suggested_ncert_heading",
                "ai_suggested_micro_concept",
                "llm_label",
                "final_confidence",
                "review_reason",
                "question_text",
            ]
        ).to_csv(PROCESSED_DIR / "ai_manual_review_suggestions.csv", index=False)
        return
    suggestions = pd.DataFrame(
        {
            "canonical_question_id": review["canonical_question_id"],
            "ai_suggested_chapter": review["chapter"],
            "ai_suggested_subtopic": review["subtopic"],
            "ai_suggested_question_pattern": review["question_pattern"],
            "ai_suggested_difficulty": review["difficulty_estimate"],
            "ai_suggested_ncert_heading": review["ncert_heading"],
            "ai_suggested_micro_concept": review["micro_concept"],
            "micro_concept_confidence": review.get("micro_concept_confidence", ""),
            "micro_concept_note": review.get("micro_concept_note", ""),
            "llm_label": review["llm_label"],
            "rule_label": review["rule_label"],
            "embedding_label": review["embedding_label"],
            "bm25_label": review["bm25_label"],
            "final_confidence": review["final_confidence"],
            "review_reason": review["review_reason"],
            "question_text": review["question_text"],
        }
    )
    suggestions.to_csv(PROCESSED_DIR / "ai_manual_review_suggestions.csv", index=False)


def _write_empty() -> pd.DataFrame:
    df = pd.DataFrame(columns=ADVANCED_COLUMNS)
    df.to_csv(PROCESSED_DIR / "questions_advanced.csv", index=False)
    df.to_json(PROCESSED_DIR / "questions_advanced.json", orient="records", indent=2)
    pd.DataFrame().to_csv(PROCESSED_DIR / "question_ncert_map.csv", index=False)
    pd.DataFrame().to_csv(PROCESSED_DIR / "question_patterns.csv", index=False)
    pd.DataFrame().to_csv(PROCESSED_DIR / "pattern_by_topic.csv", index=False)
    pd.DataFrame().to_csv(PROCESSED_DIR / "micro_concept_clusters.csv", index=False)
    pd.DataFrame().to_csv(PROCESSED_DIR / "ai_manual_review_suggestions.csv", index=False)
    return df


def _read_llm_cache() -> dict[str, Any]:
    if not LLM_CACHE_PATH.exists():
        return {}
    from .intelligence import LabelScore

    cache = {}
    context = _llm_cache_context()
    with LLM_CACHE_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("cache_context", "") != context:
                continue
            cache[row["canonical_question_id"]] = LabelScore(
                row.get("label", ""),
                row.get("chapter", ""),
                row.get("subtopic", ""),
                float(row.get("score", 0.0)),
            )
    return cache


def _append_llm_cache(canonical_id: str, label) -> None:
    LLM_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LLM_CACHE_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "canonical_question_id": canonical_id,
                    "label": label.label,
                    "chapter": label.chapter,
                    "subtopic": label.subtopic,
                    "score": label.score,
                    "cache_context": _llm_cache_context(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _llm_cache_context() -> str:
    base_url = os.getenv("CUET_LLM_BASE_URL", "")
    model = os.getenv("CUET_LLM_MODEL", "")
    mode = os.getenv("CUET_LLM_MODE", "")
    return f"{base_url}|{model}|{mode}"
