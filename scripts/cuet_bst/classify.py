from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from .settings import CONFIG_DIR, PROCESSED_DIR, QUESTION_COLUMNS, load_json


@dataclass
class TopicMatch:
    unit: str = ""
    chapter: str = ""
    subtopic: str = ""
    keywords: list[str] | None = None
    confidence: float = 0.0


def classify_questions(df: pd.DataFrame, logger: logging.Logger | None = None) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    taxonomy = load_taxonomy()
    rows: list[dict[str, Any]] = []
    for _, row in df.fillna("").iterrows():
        record = row.to_dict()
        text = " ".join(
            [
                str(record.get("question_text", "")),
                str(record.get("passage_text", "")),
                str(record.get("option_a", "")),
                str(record.get("option_b", "")),
                str(record.get("option_c", "")),
                str(record.get("option_d", "")),
            ]
        )
        match = match_topic(text, taxonomy)
        record["question_type"] = detect_question_type(text)
        record["difficulty_estimate"] = estimate_difficulty(text, record["question_type"])
        record["unit"] = match.unit
        record["chapter"] = match.chapter
        record["subtopic"] = match.subtopic
        record["ncert_keyword_match"] = ", ".join(match.keywords or [])
        record["confidence_score"] = f"{match.confidence:.2f}"
        record["needs_review"] = "yes" if match.confidence < 0.30 else "no"
        record["repeated_concept_cluster"] = make_cluster_label(record, text)
        rows.append(record)
    out = pd.DataFrame(rows, columns=QUESTION_COLUMNS)
    out.to_csv(PROCESSED_DIR / "questions.csv", index=False)
    out.to_json(PROCESSED_DIR / "questions.json", orient="records", indent=2, force_ascii=False)
    return out


def load_taxonomy() -> list[dict[str, Any]]:
    return load_json(CONFIG_DIR / "cuet_topics.json")["units"]


def match_topic(text: str, taxonomy: list[dict[str, Any]]) -> TopicMatch:
    lower = normalize_for_match(text)
    best = TopicMatch(keywords=[])
    best_score = 0.0
    for unit in taxonomy:
        chapter = unit["chapter"]
        chapter_terms = chapter.lower().split()
        for subtopic, keywords in unit["subtopics"].items():
            score = 0.0
            hits: list[str] = []
            for keyword in keywords:
                normalized = normalize_for_match(keyword)
                if not normalized:
                    continue
                if normalized in lower:
                    hits.append(keyword)
                    score += 2.5 + min(2.0, len(normalized.split()) * 0.4)
                else:
                    token_hits = sum(1 for token in normalized.split() if re.search(rf"\b{re.escape(token)}\b", lower))
                    if token_hits and len(normalized.split()) > 1:
                        score += token_hits / len(normalized.split())
            for term in chapter_terms:
                if len(term) > 4 and re.search(rf"\b{re.escape(term)}\b", lower):
                    score += 0.5
            if score > best_score:
                confidence = 1 - math.exp(-score / 7)
                best = TopicMatch(
                    unit=unit["unit"],
                    chapter=chapter,
                    subtopic=subtopic,
                    keywords=hits[:8],
                    confidence=min(confidence, 0.98),
                )
                best_score = score
    return best


def detect_question_type(text: str) -> str:
    lower = text.lower()
    if "assertion" in lower and "reason" in lower:
        return "assertion-reason"
    if "match" in lower and ("column" in lower or "list" in lower):
        return "match-the-following"
    if "case" in lower or "read the following" in lower or len(text) > 1100:
        return "case-based"
    if "statement" in lower or "statements" in lower:
        return "statement-based"
    if any(term in lower for term in ["chronological", "sequence", "arrange", "order"]):
        return "chronology"
    if any(term in lower for term in ["meaning of", "defined as", "refers to", "definition"]):
        return "definition"
    if any(term in lower for term in ["example", "case", "situation", "which function", "which principle is violated"]):
        return "application"
    return "direct"


def estimate_difficulty(text: str, question_type: str) -> str:
    lower = text.lower()
    score = 0
    score += 2 if question_type in {"case-based", "assertion-reason", "match-the-following"} else 0
    score += 1 if question_type in {"statement-based", "chronology"} else 0
    score += 1 if len(text) > 900 else 0
    score += 1 if len(re.findall(r"\b(?:not|incorrect|except|false)\b", lower)) else 0
    score += 1 if len(re.findall(r"\b[ivx]+\b", lower)) >= 3 else 0
    if score >= 4:
        return "hard"
    if score >= 2:
        return "medium"
    return "easy"


def make_cluster_label(record: dict[str, Any], text: str) -> str:
    if record.get("chapter") and record.get("subtopic"):
        return f"{record['chapter']} :: {record['subtopic']}"
    tokens = [token for token in re.findall(r"[a-zA-Z]{5,}", text.lower()) if token not in STOPWORDS]
    return "keyword :: " + " ".join(tokens[:4]) if tokens else "unclassified"


def normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text.lower())).strip()


STOPWORDS = {
    "which",
    "following",
    "statement",
    "statements",
    "correct",
    "incorrect",
    "business",
    "studies",
    "question",
    "answer",
    "option",
}
