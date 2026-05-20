from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def stable_hash(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def safe_filename(value: str, max_len: int = 120) -> str:
    value = re.sub(r"https?://", "", value)
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
    return value[:max_len] or "download"


def url_extension(url: str, content_type: str = "") -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".pdf", ".html", ".htm", ".csv", ".json", ".txt"}:
        return suffix
    if "pdf" in content_type:
        return ".pdf"
    if "json" in content_type:
        return ".json"
    if "csv" in content_type:
        return ".csv"
    if "html" in content_type or "text" in content_type:
        return ".html"
    return ".bin"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def infer_year(text: str) -> str:
    match = re.search(r"\b(20(?:2[2-6]|1[9]))\b", text)
    return match.group(1) if match else ""


def infer_shift(text: str) -> str:
    match = re.search(r"\bshift\s*[-:]?\s*([0-9A-Za-z]+)\b", text, flags=re.I)
    return match.group(1) if match else ""
