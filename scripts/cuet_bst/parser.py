from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .settings import PROCESSED_DIR, QUESTION_COLUMNS
from .util import infer_shift, infer_year, normalize_text, stable_hash


OPTION_PATTERNS = [
    re.compile(r"(?:^|\n)\s*(?:\(?A\)?|A\.|1\)|\(1\))\s+(.+?)(?=(?:\n\s*(?:\(?B\)?|B\.|2\)|\(2\))\s+)|$)", re.I | re.S),
    re.compile(r"(?:^|\n)\s*(?:\(?B\)?|B\.|2\)|\(2\))\s+(.+?)(?=(?:\n\s*(?:\(?C\)?|C\.|3\)|\(3\))\s+)|$)", re.I | re.S),
    re.compile(r"(?:^|\n)\s*(?:\(?C\)?|C\.|3\)|\(3\))\s+(.+?)(?=(?:\n\s*(?:\(?D\)?|D\.|4\)|\(4\))\s+)|$)", re.I | re.S),
    re.compile(r"(?:^|\n)\s*(?:\(?D\)?|D\.|4\)|\(4\))\s+(.+?)(?=(?:\n\s*(?:Answer|Correct|Question ID|Q\.?\s*\d+|\d+\.)\b)|$)", re.I | re.S),
]

QUESTION_START = re.compile(
    r"(?=(?:^|\n)\s*(?:(?:Question\s*ID\s*[:\-]?\s*\d+)|(?:Q(?:uestion)?\.?\s*\d+)|(?:\d{1,3}\s*[\).]))\s*)",
    re.I,
)


def parse_extracted_texts(logger: logging.Logger | None = None) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    manifest_path = PROCESSED_DIR / "extraction_manifest.csv"
    if not manifest_path.exists():
        logger.warning("No extraction manifest found at %s", manifest_path)
        return empty_questions()
    manifest = pd.read_csv(manifest_path).fillna("")
    rows: list[dict[str, Any]] = []
    for _, item in manifest.iterrows():
        text_path = Path(item["text_file_path"])
        if not text_path.exists():
            continue
        parsed = parse_structured_html_questions(
            Path(item.get("raw_file_path", "")),
            source=item.get("source", ""),
            source_url=item.get("source_url", ""),
        )
        if not parsed:
            raw_path = Path(item.get("raw_file_path", ""))
            if raw_path.suffix.lower() in {".html", ".htm"} and str(item.get("source_url", "")).startswith("http"):
                continue
            text = text_path.read_text(encoding="utf-8", errors="ignore")
            parsed = parse_document(
                text,
                source=item.get("source", ""),
                source_url=item.get("source_url", ""),
                raw_file_path=item.get("raw_file_path", ""),
            )
        rows.extend(parsed)
    df = pd.DataFrame(rows)
    if df.empty:
        df = empty_questions()
    df = normalize_question_frame(df)
    df = dedupe_questions(df)
    df.to_csv(PROCESSED_DIR / "parsed_questions.csv", index=False)
    return df


def parse_document(text: str, *, source: str, source_url: str, raw_file_path: str) -> list[dict[str, Any]]:
    text = normalize_text(text)
    chunks = [chunk.strip() for chunk in QUESTION_START.split(text) if chunk.strip()]
    rows: list[dict[str, Any]] = []
    if len(chunks) < 2:
        chunks = _fallback_chunks(text)
    for index, chunk in enumerate(chunks, start=1):
        if not _looks_like_question(chunk):
            continue
        rows.append(parse_question_block(chunk, index, source, source_url, raw_file_path))
    return rows


def parse_structured_html_questions(path: Path, *, source: str, source_url: str) -> list[dict[str, Any]]:
    if not path.exists() or path.suffix.lower() not in {".html", ".htm"}:
        return []
    if "afterboards.in" not in source_url.lower() and "dubuddy.in" not in source_url.lower():
        return []
    html = path.read_text(encoding="utf-8", errors="ignore")
    rows: list[dict[str, Any]] = []
    for match in re.finditer(r'\{\\+"_id\\+":.*?\\+"mockUrl\\+":\\+".*?\\+"\}', html):
        raw = match.group(0)
        try:
            obj = json.loads(bytes(raw, "utf-8").decode("unicode_escape"))
        except Exception:
            continue
        if not obj.get("question") or not obj.get("_id"):
            continue
        meta = " ".join(
            [
                source,
                source_url,
                str(obj.get("customMockName", "")),
                str(obj.get("examDisplayName", "")),
                str(obj.get("examID", "")),
            ]
        )
        rows.append(
            {
                "source": source,
                "year": infer_year(meta),
                "date": _first_match(meta, r"\b(\d{1,2}\s+[A-Za-z]{3,9})\b"),
                "shift": infer_shift(meta),
                "set_name": str(obj.get("customMockName", "") or obj.get("examDisplayName", "")),
                "question_id": str(obj.get("_id", "")),
                "question_number": str(obj.get("questionNumber", "")),
                "question_text": normalize_text(str(obj.get("question", ""))),
                "passage_text": normalize_text(str(obj.get("comprehension", ""))),
                "case_text": normalize_text(str(obj.get("comprehension", ""))),
                "option_a": str(obj.get("option1", "")),
                "option_b": str(obj.get("option2", "")),
                "option_c": str(obj.get("option3", "")),
                "option_d": str(obj.get("option4", "")),
                "correct_option": _normalize_option(str(obj.get("correctAnswer", ""))),
                "answer_source": source,
                "question_type": "",
                "difficulty_estimate": str(obj.get("difficulty", "")).lower(),
                "unit": "",
                "chapter": "",
                "subtopic": "",
                "ncert_keyword_match": str(obj.get("subTopic", "")),
                "repeated_concept_cluster": "",
                "confidence_score": "",
                "source_url": source_url,
                "raw_file_path": str(path),
                "needs_review": "",
            }
        )
    return rows


def parse_question_block(block: str, fallback_number: int, source: str, source_url: str, raw_file_path: str) -> dict[str, Any]:
    question_id = _first_match(block, r"Question\s*ID\s*[:\-]?\s*(\d+)")
    question_number = _first_match(block, r"(?:^|\n)\s*(?:Q(?:uestion)?\.?\s*|)(\d{1,3})\s*[\).]") or str(fallback_number)
    answer = _first_match(block, r"(?:Answer|Correct\s*Option|Correct\s*Answer|Ans)\s*[:\-]?\s*([A-D1-4])")
    answer = _normalize_option(answer)
    options = [_clean_option(match.search(block).group(1)) if match.search(block) else "" for match in OPTION_PATTERNS]
    question_text = _question_without_options(block)
    passage = _first_match(block, r"(?:Read the following|Case Study|Case based|Passage)[:\-\s]+(.+?)(?=\n\s*(?:Question ID|Q\.?\s*\d+|\d+\s*[\).]))")
    meta_text = f"{source} {source_url} {raw_file_path} {block[:300]}"
    return {
        "source": source,
        "year": infer_year(meta_text),
        "date": _first_match(meta_text, r"\b(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})\b"),
        "shift": infer_shift(meta_text),
        "set_name": _first_match(meta_text, r"\bset\s*[-:]?\s*([A-Za-z0-9]+)\b"),
        "question_id": question_id or stable_hash(question_text + source_url, 18),
        "question_number": question_number,
        "question_text": question_text,
        "passage_text": passage or "",
        "case_text": passage or "",
        "option_a": options[0],
        "option_b": options[1],
        "option_c": options[2],
        "option_d": options[3],
        "correct_option": answer,
        "answer_source": source if answer else "",
        "source_url": source_url,
        "raw_file_path": raw_file_path,
    }


def normalize_question_frame(df: pd.DataFrame) -> pd.DataFrame:
    for column in QUESTION_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df = df[QUESTION_COLUMNS]
    for column in df.columns:
        df[column] = df[column].fillna("").astype(str).map(lambda value: normalize_text(value))
    return df


def dedupe_questions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["_dedupe_key"] = df.apply(_dedupe_key, axis=1)
    df = df.sort_values(["question_id", "source_url"]).drop_duplicates("_dedupe_key", keep="first")
    return df.drop(columns=["_dedupe_key"]).reset_index(drop=True)


def empty_questions() -> pd.DataFrame:
    return pd.DataFrame(columns=QUESTION_COLUMNS)


def _dedupe_key(row: pd.Series) -> str:
    if row.get("question_id") and len(str(row.get("question_id"))) > 6:
        return str(row.get("question_id"))
    text = re.sub(r"\W+", " ", str(row.get("question_text", "")).lower()).strip()
    return hashlib.sha1(text[:500].encode("utf-8", errors="ignore")).hexdigest()


def _looks_like_question(chunk: str) -> bool:
    lower = chunk.lower()
    has_prompt = "?" in chunk or any(word in lower for word in ["which", "what", "identify", "match", "assertion", "statement", "following"])
    has_options = sum(1 for pattern in OPTION_PATTERNS if pattern.search(chunk)) >= 2
    return has_prompt or has_options


def _fallback_chunks(text: str) -> list[str]:
    lines = text.splitlines()
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        if re.match(r"^\s*\d{1,3}\s*[\).]\s+", line) and current:
            chunks.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append("\n".join(current))
    return chunks


def _question_without_options(block: str) -> str:
    cut = re.split(r"\n\s*(?:\(?A\)?|A\.|1\)|\(1\))\s+", block, maxsplit=1, flags=re.I)[0]
    cut = re.sub(r"Question\s*ID\s*[:\-]?\s*\d+", "", cut, flags=re.I)
    cut = re.sub(r"^\s*(?:Q(?:uestion)?\.?\s*|)\d{1,3}\s*[\).]\s*", "", cut, flags=re.I)
    cut = re.sub(r"(?:Answer|Correct\s*Option|Correct\s*Answer|Ans)\s*[:\-]?\s*[A-D1-4].*$", "", cut, flags=re.I | re.S)
    return normalize_text(cut)


def _first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.I | re.S)
    return normalize_text(match.group(1)) if match else ""


def _normalize_option(value: str) -> str:
    mapping = {"1": "A", "2": "B", "3": "C", "4": "D"}
    values = re.findall(r"[A-D1-4]", value.strip().upper())
    if not values:
        return ""
    return ",".join(mapping.get(item, item) for item in values)


def _clean_option(value: str) -> str:
    return normalize_text(re.sub(r"\s+", " ", value))
