from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from .downloader import MANIFEST
from .settings import MANUAL_OFFICIAL_DIR, PROCESSED_DIR, env_bool, ensure_dirs
from .util import normalize_text, read_jsonl


EXTRACTED_DIR = PROCESSED_DIR / "extracted_text"


def extract_all(logger: logging.Logger | None = None) -> pd.DataFrame:
    ensure_dirs()
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    logger = logger or logging.getLogger(__name__)
    manifest = [row for row in read_jsonl(MANIFEST) if row.get("status") == "downloaded"]
    manifest.extend(_manual_official_items())
    rows: list[dict[str, Any]] = []
    for item in manifest:
        if not _is_relevant_source(item):
            continue
        raw_path = Path(item.get("raw_file_path", ""))
        if not raw_path.exists():
            continue
        text = extract_file(raw_path, logger=logger)
        out_path = EXTRACTED_DIR / f"{raw_path.stem}.txt"
        out_path.write_text(text, encoding="utf-8", errors="ignore")
        rows.append(
            {
                "source": item.get("source_name", ""),
                "source_url": item.get("source_url", ""),
                "raw_file_path": str(raw_path),
                "text_file_path": str(out_path),
                "text_chars": len(text),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED_DIR / "extraction_manifest.csv", index=False)
    return df


def _is_relevant_source(item: dict[str, Any]) -> bool:
    url = f"{item.get('source_url', '')} {item.get('final_url', '')} {item.get('source_name', '')}".lower()
    relevant_terms = [
        "business-studies",
        "business studies",
        "subject 305",
        "cuet-business-studies",
        "2025030154-1.pdf",
        "cuet.nta.nic.in/syllabus",
        "manual://",
        "manual-official://",
        "official_manual",
    ]
    if any(term in url for term in relevant_terms):
        return True
    if "nta.ac.in/download/exampaper" in url and any(term in url for term in ["business", "305", "bst"]):
        return True
    return False


def _manual_official_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not MANUAL_OFFICIAL_DIR.exists():
        return items
    for path in sorted(MANUAL_OFFICIAL_DIR.glob("*")):
        if path.suffix.lower() not in {".pdf", ".html", ".htm", ".txt"}:
            continue
        items.append(
            {
                "source_name": "official_manual",
                "source_url": f"manual-official://{path.name}",
                "final_url": f"manual-official://{path.name}",
                "raw_file_path": str(path),
                "status": "downloaded",
            }
        )
    return items


def extract_file(path: Path, logger: logging.Logger | None = None) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path, logger=logger)
    if suffix in {".html", ".htm"}:
        return extract_html(path)
    if suffix in {".txt", ".csv", ".json"}:
        return normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    return ""


def extract_pdf(path: Path, logger: logging.Logger | None = None) -> str:
    logger = logger or logging.getLogger(__name__)
    parts: list[str] = []
    try:
        import fitz

        with fitz.open(path) as doc:
            for page_number, page in enumerate(doc, start=1):
                text = page.get_text("text") or ""
                if text.strip():
                    parts.append(f"\n\n--- Page {page_number} ---\n{text}")
    except Exception as exc:
        logger.debug("PyMuPDF failed for %s: %s", path, exc)

    text = normalize_text("\n".join(parts))
    if len(text) > 300:
        return text

    try:
        import pdfplumber

        parts = []
        with pdfplumber.open(path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    parts.append(f"\n\n--- Page {page_number} ---\n{page_text}")
    except Exception as exc:
        logger.debug("pdfplumber failed for %s: %s", path, exc)

    text = normalize_text("\n".join(parts))
    if len(text) > 300 or not env_bool("CUET_ENABLE_OCR", False):
        return text
    return _ocr_pdf(path, logger)


def _ocr_pdf(path: Path, logger: logging.Logger) -> str:
    try:
        import fitz
        import pytesseract
        from PIL import Image
    except Exception as exc:
        logger.warning("OCR requested but dependencies are unavailable: %s", exc)
        return ""

    parts: list[str] = []
    try:
        with fitz.open(path) as doc:
            for page_number, page in enumerate(doc, start=1):
                pix = page.get_pixmap(dpi=200)
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(image)
                parts.append(f"\n\n--- OCR Page {page_number} ---\n{text}")
    except Exception as exc:
        logger.warning("OCR failed for %s: %s", path, exc)
    return normalize_text("\n".join(parts))


def extract_html(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    table_texts = []
    for table in soup.find_all("table"):
        table_texts.append(table.get_text(" | ", strip=True))
    text = soup.get_text("\n", strip=True)
    if table_texts:
        text += "\n\n--- Tables ---\n" + "\n".join(table_texts)
    return normalize_text(text)
