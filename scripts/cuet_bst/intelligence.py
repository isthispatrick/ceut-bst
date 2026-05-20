from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .classify import detect_question_type, estimate_difficulty, load_taxonomy, match_topic
from .llm_client import chat_completion, configured_api_key, parse_json_object
from .settings import CONFIG_DIR, MANUAL_IMPORT_DIR, PROCESSED_DIR, load_json
from .util import stable_hash


SOURCE_WEIGHTS = {
    "official_nta": 1.00,
    "official_manual": 1.00,
    "official_syllabus_ncert": 0.90,
    "structured_pyq_site": 0.65,
    "blog_article": 0.45,
    "reddit_community": 0.15,
    "manual_import": 0.65,
    "unknown": 0.35,
}

YEAR_WEIGHTS = {"2025": 1.00, "2024": 0.80, "2023": 0.60, "2022": 0.45}

QUESTION_PATTERNS = [
    "definition-based",
    "feature-identification",
    "principle-identification",
    "case-study diagnosis",
    "match-the-following",
    "assertion-reason",
    "statement true/false",
    "chronology/process order",
    "concept-to-example",
    "example-to-concept",
]

CONCEPT_TYPES = ["definition", "feature", "process", "principle", "case-application", "difference", "factor", "example"]

MICRO_CONCEPTS = {
    "scalar chain": ["scalar chain", "gang plank"],
    "unity of command": ["unity of command", "one boss"],
    "unity of direction": ["unity of direction", "one head one plan"],
    "division of work": ["division of work", "specialisation"],
    "equity": ["equity", "kindliness", "justice"],
    "discipline": ["discipline", "obedience"],
    "centralisation/decentralisation": ["centralisation", "decentralisation"],
    "scientific management": ["scientific management", "science not rule of thumb", "mental revolution"],
    "method study": ["method study"],
    "motion study": ["motion study"],
    "time study": ["time study"],
    "fatigue study": ["fatigue study"],
    "differential piece wage": ["differential piece wage"],
    "functional foremanship": ["functional foremanship"],
    "types of plans": ["policy", "procedure", "method", "rule", "programme", "budget", "strategy", "objective"],
    "delegation": ["delegation", "authority", "responsibility", "accountability"],
    "formal/informal organisation": ["formal organisation", "informal organisation", "grapevine"],
    "recruitment": ["recruitment", "internal sources", "external sources", "campus recruitment", "advertisement"],
    "selection": ["selection", "employment test", "interview", "medical examination"],
    "training": ["training", "apprenticeship", "vestibule", "internship", "job rotation"],
    "maslow hierarchy": ["maslow", "physiological", "safety", "esteem", "self actualisation", "belongingness"],
    "leadership style": ["autocratic", "democratic", "laissez faire", "leadership"],
    "communication barriers": ["semantic", "psychological", "organisational", "personal barrier", "barriers"],
    "financial risk": ["financial risk", "debt", "equity", "interest"],
    "capital structure": ["capital structure", "trading on equity", "debt equity"],
    "fixed capital": ["fixed capital", "technology upgradation", "scale of operations"],
    "working capital": ["working capital", "operating cycle", "inventory", "receivables"],
    "marketing mix": ["marketing mix", "four ps", "4ps"],
    "product mix": ["product mix", "branding", "labelling", "packaging", "trademark", "brand mark"],
    "price mix": ["price mix", "pricing", "price"],
    "promotion mix": ["promotion mix", "advertising", "sales promotion", "personal selling", "public relations"],
    "place/channel": ["place mix", "channel", "distribution", "wholesaler", "retailer"],
    "consumer rights": ["right to safety", "right to be informed", "right to choose", "right to be heard", "right to seek redressal", "right to consumer education"],
    "consumer redressal": ["district commission", "state commission", "national commission", "consumer protection act"],
    "entrepreneur characteristics": ["initiative", "creativity", "confidence", "perseverance", "competencies"],
}


@dataclass
class LabelScore:
    label: str
    chapter: str
    subtopic: str
    score: float
    heading: str = ""
    concept: str = ""


def normalize_question_identity(row: pd.Series | dict[str, Any]) -> str:
    parts = [
        row.get("question_text", ""),
        row.get("option_a", ""),
        row.get("option_b", ""),
        row.get("option_c", ""),
        row.get("option_d", ""),
    ]
    text = " ".join(str(part) for part in parts)
    text = text.lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\b(option|answer|correct|business|studies)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def question_hash(row: pd.Series | dict[str, Any]) -> str:
    return stable_hash(normalize_question_identity(row), 24)


def source_tier(source: str, source_url: str = "") -> str:
    lower = f"{source} {source_url}".lower()
    if "official_manual" in lower or "manual-official" in lower or "manual_official_papers" in lower:
        return "official_manual"
    if "nta.ac.in" in lower and any(term in lower for term in ["question", "answer", "exampaper", "paper"]):
        return "official_nta"
    if "cuet.nta.nic.in/syllabus" in lower or "ncert" in lower or "syllabus" in lower:
        return "official_syllabus_ncert"
    if any(domain in lower for domain in ["afterboards", "dubuddy"]):
        return "structured_pyq_site"
    if any(domain in lower for domain in ["pw.live", "physicswallah", "testcoach", "blog"]):
        return "blog_article"
    if "reddit" in lower:
        return "reddit_community"
    if lower.startswith("manual") or "manual://" in lower:
        return "manual_import"
    return "unknown"


def source_weight(source: str, source_url: str = "") -> float:
    return SOURCE_WEIGHTS[source_tier(source, source_url)]


def year_weight(year: Any) -> float:
    text = str(year).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return YEAR_WEIGHTS.get(text, 0.35)


def build_ncert_index() -> pd.DataFrame:
    taxonomy = load_json(CONFIG_DIR / "cuet_topics.json")
    rows: list[dict[str, Any]] = []
    for unit in taxonomy["units"]:
        chapter = unit["chapter"]
        for subtopic, keywords in unit["subtopics"].items():
            heading = subtopic.title()
            concept_text = "; ".join(keywords)
            rows.append(
                {
                    "unit": unit["unit"],
                    "chapter": chapter,
                    "subtopic": subtopic,
                    "ncert_heading": heading,
                    "ncert_page": "",
                    "ncert_paragraph": "",
                    "ncert_concept": concept_text,
                    "concept_type": infer_concept_type(f"{subtopic} {concept_text}"),
                    "index_text": f"{chapter} {subtopic} {heading} {concept_text}",
                    "source": "syllabus_taxonomy_fallback",
                }
            )
    out = PROCESSED_DIR / "ncert_reverse_index.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


def infer_concept_type(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ["process", "steps", "procedure"]):
        return "process"
    if any(term in lower for term in ["principle", "fayol", "taylor", "scalar", "unity"]):
        return "principle"
    if any(term in lower for term in ["features", "characteristics", "nature"]):
        return "feature"
    if any(term in lower for term in ["factor", "factors", "affecting"]):
        return "factor"
    if any(term in lower for term in ["difference", " vs ", "versus"]):
        return "difference"
    if any(term in lower for term in ["example", "case"]):
        return "example"
    if any(term in lower for term in ["meaning", "definition", "refers"]):
        return "definition"
    return "case-application" if len(text) > 500 else "definition"


def classify_question_pattern(text: str) -> str:
    lower = text.lower()
    if "assertion" in lower and "reason" in lower:
        return "assertion-reason"
    if "match" in lower and ("list" in lower or "column" in lower):
        return "match-the-following"
    if any(term in lower for term in ["arrange", "sequence", "chronological", "order of", "steps"]):
        return "chronology/process order"
    if any(term in lower for term in ["read the following", "case", "situation", "which principle is", "which function is being"]):
        return "case-study diagnosis"
    if any(term in lower for term in ["statement", "true", "false", "correct statements", "incorrect statements"]):
        return "statement true/false"
    if any(term in lower for term in ["example of", "is an example", "identify the concept"]):
        return "example-to-concept"
    if any(term in lower for term in ["which of the following is not", "feature", "characteristic"]):
        return "feature-identification"
    if any(term in lower for term in ["principle", "scalar chain", "unity of", "fayol", "taylor"]):
        return "principle-identification"
    if any(term in lower for term in ["meaning", "defined", "refers to", "definition"]):
        return "definition-based"
    return "concept-to-example" if any(term in lower for term in ["which one", "which option"]) else "definition-based"


def identify_micro_concept(text: str, chapter: str = "", subtopic: str = "") -> str:
    return identify_micro_concept_with_confidence(text, chapter, subtopic)[0]


def identify_micro_concept_with_confidence(text: str, chapter: str = "", subtopic: str = "") -> tuple[str, float, str]:
    lower = f"{chapter} {subtopic} {text}".lower()
    best_label = ""
    best_hits = 0
    for label, keywords in MICRO_CONCEPTS.items():
        hits = sum(1 for keyword in keywords if keyword in lower)
        if hits > best_hits:
            best_hits = hits
            best_label = label
    if best_label:
        confidence = min(0.98, 0.62 + best_hits * 0.18)
        note = "" if confidence >= 0.70 else "Micro-concept uncertain; trust chapter/subtopic more."
        return best_label, confidence, note
    tokens = [token for token in re.findall(r"[a-z]{5,}", lower) if token not in STOPWORDS]
    if not tokens:
        fallback = str(subtopic or "unclassified")
        return fallback, 0.35, "Micro-concept uncertain; trust chapter/subtopic more."
    token = Counter(tokens).most_common(1)[0][0]
    fallback = str(subtopic or token or "unclassified")
    return fallback, 0.45, "Micro-concept uncertain; trust chapter/subtopic more."


def make_question_text(row: pd.Series | dict[str, Any]) -> str:
    return " ".join(
        str(row.get(column, ""))
        for column in ["question_text", "passage_text", "option_a", "option_b", "option_c", "option_d"]
    ).strip()


def tfidf_best(question_texts: list[str], labels: pd.DataFrame) -> list[LabelScore]:
    corpus = labels["index_text"].fillna("").astype(str).tolist()
    if not corpus:
        return [LabelScore("", "", "", 0.0) for _ in question_texts]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1)
    label_matrix = vectorizer.fit_transform(corpus)
    q_matrix = vectorizer.transform(question_texts)
    sims = cosine_similarity(q_matrix, label_matrix)
    results: list[LabelScore] = []
    for row in sims:
        idx = int(np.argmax(row)) if len(row) else 0
        score = float(row[idx]) if len(row) else 0.0
        label_row = labels.iloc[idx]
        results.append(
            LabelScore(
                label=f"{label_row['chapter']} :: {label_row['subtopic']}",
                chapter=str(label_row["chapter"]),
                subtopic=str(label_row["subtopic"]),
                score=min(0.98, score * 2.2),
                heading=str(label_row["ncert_heading"]),
                concept=str(label_row["ncert_concept"]),
            )
        )
    return results


def embedding_best(question_texts: list[str], labels: pd.DataFrame) -> list[LabelScore]:
    if os.getenv("CUET_USE_SENTENCE_TRANSFORMERS", "").lower() not in {"1", "true", "yes"}:
        return tfidf_best(question_texts, labels)
    try:
        from sentence_transformers import SentenceTransformer

        model_name = os.getenv("CUET_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        label_vectors = model.encode(labels["index_text"].astype(str).tolist(), normalize_embeddings=True)
        q_vectors = model.encode(question_texts, normalize_embeddings=True)
        sims = np.matmul(q_vectors, label_vectors.T)
    except Exception:
        return tfidf_best(question_texts, labels)
    results = []
    for row in sims:
        idx = int(np.argmax(row))
        score = float(row[idx])
        label_row = labels.iloc[idx]
        results.append(
            LabelScore(
                label=f"{label_row['chapter']} :: {label_row['subtopic']}",
                chapter=str(label_row["chapter"]),
                subtopic=str(label_row["subtopic"]),
                score=max(0.0, min(0.98, score)),
                heading=str(label_row["ncert_heading"]),
                concept=str(label_row["ncert_concept"]),
            )
        )
    return results


def rule_label(text: str) -> LabelScore:
    match = match_topic(text, load_taxonomy())
    return LabelScore(
        label=f"{match.chapter} :: {match.subtopic}" if match.chapter else "",
        chapter=match.chapter,
        subtopic=match.subtopic,
        score=match.confidence,
    )


def llm_label(text: str, candidates: list[LabelScore]) -> LabelScore:
    if not configured_api_key():
        best = max(candidates, key=lambda item: item.score)
        return LabelScore(best.label, best.chapter, best.subtopic, 0.0, best.heading, best.concept)
    max_tokens = int(os.getenv("CUET_LLM_MAX_TOKENS", "180"))
    candidate_payload = [
        {"chapter": item.chapter, "subtopic": item.subtopic, "score": round(item.score, 3)}
        for item in candidates
        if item.chapter
    ][:5]
    prompt = {
        "task": "Classify this CUET UG Business Studies question into one candidate chapter/subtopic and return compact JSON.",
        "question": text[:2200],
        "candidates": candidate_payload,
        "schema": {"chapter": "string", "subtopic": "string", "confidence": "0-1", "reason": "short"},
    }
    try:
        content = chat_completion(
            [
                {"role": "system", "content": "You classify CUET Business Studies PYQs using NCERT terminology. Return JSON only."},
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
            max_tokens=max_tokens,
            timeout=25,
        )
        parsed = parse_json_object(content)
        chapter = str(parsed.get("chapter", ""))
        subtopic = str(parsed.get("subtopic", ""))
        confidence = float(parsed.get("confidence", 0.0))
        return LabelScore(f"{chapter} :: {subtopic}", chapter, subtopic, max(0.0, min(1.0, confidence)))
    except Exception:
        best = max(candidates, key=lambda item: item.score)
        return LabelScore(best.label, best.chapter, best.subtopic, 0.0, best.heading, best.concept)
def combine_labels(rule: LabelScore, embedding: LabelScore, bm25: LabelScore, llm: LabelScore) -> tuple[LabelScore, float, str]:
    weighted: dict[tuple[str, str], float] = {}
    for item, weight in [(rule, 0.25), (embedding, 0.25), (bm25, 0.20), (llm, 0.30)]:
        key = (item.chapter, item.subtopic)
        if not item.chapter:
            continue
        weighted[key] = weighted.get(key, 0.0) + weight * item.score
    if not weighted:
        return LabelScore("", "", "", 0.0), 0.0, "no classifier produced a label"
    best_key, best_score = max(weighted.items(), key=lambda pair: pair[1])
    labels = [rule, embedding, bm25, llm]
    agreement = sum(1 for item in labels if (item.chapter, item.subtopic) == best_key)
    agreement_bonus = agreement / len(labels) * 0.25
    final_confidence = min(0.99, best_score + agreement_bonus)
    reason = "" if final_confidence >= 0.75 and agreement >= 2 else f"confidence={final_confidence:.2f}; agreement={agreement}/4"
    template = next((item for item in labels if (item.chapter, item.subtopic) == best_key), rule)
    return LabelScore(f"{best_key[0]} :: {best_key[1]}", best_key[0], best_key[1], final_confidence, template.heading, template.concept), final_confidence, reason


def apply_manual_labels(df: pd.DataFrame, labels: pd.DataFrame | None = None) -> pd.DataFrame:
    if labels is not None:
        return _apply_manual_label_frame(df, labels)
    labels_path = MANUAL_IMPORT_DIR.parent / "manual_labels.csv"
    if not labels_path.exists():
        labels_path = MANUAL_IMPORT_DIR / "manual_labels.csv"
    if not labels_path.exists():
        return df
    labels = pd.read_csv(labels_path).fillna("")
    return _apply_manual_label_frame(df, labels)


def _apply_manual_label_frame(df: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    if "canonical_question_id" not in labels.columns:
        return df
    editable = ["chapter", "subtopic", "question_pattern", "difficulty_estimate", "ncert_heading", "micro_concept"]
    df = df.copy()
    label_map = labels.drop_duplicates("canonical_question_id", keep="last").set_index("canonical_question_id")
    for idx, row in df.iterrows():
        cid = row.get("canonical_question_id", "")
        if cid not in label_map.index:
            continue
        for column in editable:
            value = label_map.loc[cid].get(column, "")
            if str(value).strip():
                df.at[idx, column] = value
        df.at[idx, "needs_review"] = "false"
        df.at[idx, "review_reason"] = "manual label applied"
    return df


STOPWORDS = {
    "which",
    "following",
    "statement",
    "correct",
    "incorrect",
    "answer",
    "question",
    "option",
    "business",
    "studies",
    "given",
    "below",
}
