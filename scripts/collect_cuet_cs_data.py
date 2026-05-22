from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib import robotparser
from urllib.parse import urlparse

import pandas as pd
import requests

from cuet_bst.extract import extract_file
from cuet_bst.util import normalize_text
from init_cuet_cs import PROCESSED_DIR, RAW_DIR, build_quality_summary, build_sources, build_strategy, build_taxonomy, build_topic_frequency, write_csv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "cuet_cs_topics.json"
SOURCES = ROOT / "config" / "cuet_cs_sources.json"
PUBLIC_RAW = RAW_DIR / "public"
EXTRACTED = PROCESSED_DIR / "extracted_text"
USER_AGENT = "cuet-cs-analysis/0.2 (+local research; respects robots.txt)"


@dataclass(frozen=True)
class PublicFile:
    name: str
    year: int
    kind: str
    source_tier: str
    source_weight: float
    url: str


PUBLIC_FILES = [
    PublicFile(
        "aglasem_cuet_cs_2025.pdf",
        2025,
        "question_paper",
        "structured_pyq_site",
        0.65,
        "https://docs.aglasem.com/product/single-doc-download/8d9c2d9a-1277-11f1-bcb2-0a5e36bc6706?url=https%3A%2F%2Fdocs.aglasem.com%2Fview%2F8d9c2d9a-1277-11f1-bcb2-0a5e36bc6706",
    ),
    PublicFile(
        "aglasem_cuet_cs_2024.pdf",
        2024,
        "question_paper",
        "structured_pyq_site",
        0.65,
        "https://docs.aglasem.com/product/single-doc-download/3c7cf9ee-1277-11f1-8401-0a5e36bc6706?url=https%3A%2F%2Fdocs.aglasem.com%2Fview%2F3c7cf9ee-1277-11f1-8401-0a5e36bc6706",
    ),
    PublicFile(
        "aglasem_cuet_cs_2023.pdf",
        2023,
        "question_paper",
        "structured_pyq_site",
        0.65,
        "https://docs.aglasem.com/product/single-doc-download/d24ba084-390c-11ee-a5c4-0a5e36bc6706?url=https%3A%2F%2Fdocs.aglasem.com%2Fview%2Fd24ba084-390c-11ee-a5c4-0a5e36bc6706",
    ),
    PublicFile(
        "aglasem_cuet_cs_2022.pdf",
        2022,
        "question_paper",
        "structured_pyq_site",
        0.65,
        "https://docs.aglasem.com/product/single-doc-download/f94cd7e8-cecc-11ed-b3d3-0a5e36bc6706?url=https%3A%2F%2Fdocs.aglasem.com%2Fview%2Ff94cd7e8-cecc-11ed-b3d3-0a5e36bc6706",
    ),
    PublicFile(
        "nta_final_answer_key_2024.pdf",
        2024,
        "answer_key",
        "official_answer_key_mirror",
        0.8,
        "https://studyfordreams.in/wp-content/uploads/2024/07/CUET-UG-CBT-Answer-Key-2024.pdf",
    ),
    PublicFile(
        "nta_final_answer_key_2025_mirror.pdf",
        2025,
        "answer_key",
        "official_answer_key_mirror",
        0.8,
        "https://cdn-images.prepp.in/public/image/CUET_UG_Final_Answer_Key_2025_75f2d60e730257946fde4bb24156c77f.pdf",
    ),
    PublicFile(
        "collegedekho_cuet_cs_question_paper.pdf",
        2024,
        "question_paper",
        "structured_pyq_site",
        0.65,
        "https://static.collegedekho.com/media/uploads/2024/05/07/paper_20240419140753.pdf",
    ),
]


def main() -> None:
    PUBLIC_RAW.mkdir(parents=True, exist_ok=True)
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": USER_AGENT})

    manifest = []
    for item in PUBLIC_FILES:
        result = download_public_file(session, item)
        manifest.append(result)
        time.sleep(1.0)

    manifest_df = pd.DataFrame(manifest)
    manifest_df.to_csv(PROCESSED_DIR / "internet_import_manifest.csv", index=False)

    extracted_rows = []
    question_rows = []
    answer_rows = []
    for row in manifest:
        if row["status"] not in {"downloaded", "cached"}:
            continue
        path = Path(row["raw_file_path"])
        text = extract_file(path)
        text_path = EXTRACTED / f"{path.stem}.txt"
        text_path.write_text(text, encoding="utf-8", errors="ignore")
        extracted_rows.append({**row, "text_file_path": str(text_path), "text_chars": len(text)})
        if row["kind"] == "question_paper":
            question_rows.extend(parse_question_metadata(text, row))
        elif row["kind"] == "answer_key":
            answer_rows.extend(parse_answer_key(text, row))

    pd.DataFrame(extracted_rows).to_csv(PROCESSED_DIR / "cs_extraction_manifest.csv", index=False)
    questions = pd.DataFrame(question_rows)
    answers = pd.DataFrame(answer_rows)
    if not answers.empty:
        answers = answers.drop_duplicates(["year", "question_id", "language"], keep="first")
    if not questions.empty and not answers.empty:
        answer_map = answers.sort_values("source_weight", ascending=False).drop_duplicates("question_id").set_index("question_id")["correct_option"].to_dict()
        questions["correct_option"] = questions["question_id"].map(answer_map).fillna(questions["correct_option"])
    questions = normalize_questions(questions)
    answers.to_csv(PROCESSED_DIR / "answer_key_entries.csv", index=False)
    questions.to_csv(PROCESSED_DIR / "questions_advanced.csv", index=False)

    regenerate_analysis(questions, answers, manifest_df)
    print(f"Imported {len(questions)} CS question metadata rows and {len(answers)} answer-key entries.")


def download_public_file(session: requests.Session, item: PublicFile) -> dict:
    out = PUBLIC_RAW / item.name
    result = {
        "name": item.name,
        "year": item.year,
        "kind": item.kind,
        "source_tier": item.source_tier,
        "source_weight": item.source_weight,
        "source_url": item.url,
        "raw_file_path": str(out),
        "status": "",
        "http_status": "",
        "bytes": 0,
        "note": "",
    }
    if out.exists() and out.stat().st_size > 1000:
        result.update({"status": "cached", "bytes": out.stat().st_size, "note": "Used cached file."})
        return result
    if not can_fetch(session, item.url):
        result.update({"status": "skipped", "note": "Blocked by robots.txt."})
        return result
    try:
        response = session.get(item.url, timeout=90, allow_redirects=True)
        result["http_status"] = response.status_code
        content_type = response.headers.get("content-type", "")
        if response.status_code >= 400:
            result.update({"status": "failed", "note": f"HTTP {response.status_code}"})
            return result
        if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
            result.update({"status": "failed", "note": f"Unexpected content type: {content_type}"})
            return result
        out.write_bytes(response.content)
        result.update({"status": "downloaded", "bytes": out.stat().st_size, "note": "Downloaded public PDF."})
    except requests.RequestException as exc:
        result.update({"status": "failed", "note": str(exc)[:250]})
    return result


def can_fetch(session: requests.Session, url: str) -> bool:
    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return False
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    parser = robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        response = session.get(robots_url, timeout=20)
        parser.parse(response.text.splitlines() if response.status_code < 400 else [])
        return parser.can_fetch(USER_AGENT, url)
    except requests.RequestException:
        return True


def parse_question_metadata(text: str, meta: dict) -> list[dict]:
    rows = []
    clean = normalize_text(text)
    subject = "Computer Science"
    shift = first_match(clean, r"Shift\s*(\d+)")
    paper_name = first_match(clean, r"Question Paper Name\s*:\s*(.+?)(?:Subject Name|Creation Date|Duration)")
    sections = section_ranges(clean)
    for match in re.finditer(r"Question Number\s*:\s*(\d+)\s+Question Id\s*:\s*(\d+).*?Question Type\s*:\s*([A-Z]+)", clean, flags=re.I | re.S):
        qn = int(match.group(1))
        qid = match.group(2)
        section = infer_section(qn, sections)
        rows.append(
            {
                "source": meta["name"],
                "year": meta["year"],
                "date": first_match(clean, r"Creation Date\s*:\s*(\d{4}-\d{2}-\d{2})"),
                "shift": shift,
                "set_name": paper_name,
                "question_id": qid,
                "question_number": qn,
                "question_text": "",
                "option_a": "",
                "option_b": "",
                "option_c": "",
                "option_d": "",
                "correct_option": "",
                "answer_source": "",
                "question_type": match.group(3).upper(),
                "difficulty_estimate": "unknown",
                "unit": section,
                "chapter": "Unclassified - metadata only",
                "subtopic": "Question text not machine-readable",
                "canonical_question_id": stable_id(qid or f"{meta['name']}-{qn}"),
                "source_tier": meta["source_tier"],
                "source_weight": meta["source_weight"],
                "year_weight": year_weight(int(meta["year"])),
                "question_pattern": "metadata-only MCQ",
                "micro_concept": "needs OCR/manual import",
                "final_confidence": 0.15,
                "needs_review": True,
                "source_url": meta["source_url"],
                "raw_file_path": meta["raw_file_path"],
                "evidence_status": "metadata_only_question_id",
                "section_scope": subject if section == "Section B1 Computer Science" else section,
            }
        )
    for match in re.finditer(r"Section\s*:\s*(COMPULSORY|COMPUTER SCIENCE).*?Item No\s*:\s*(\d+)\s+Question ID\s*:\s*(\d+)\s+Question Type\s*:\s*([A-Z]+)", clean, flags=re.I | re.S):
        raw_section = normalize_text(match.group(1)).upper()
        qn = int(match.group(2))
        qid = match.group(3)
        section = "Section B1 Computer Science" if raw_section == "COMPUTER SCIENCE" else "Section A Common Core"
        rows.append(
            {
                "source": meta["name"],
                "year": meta["year"],
                "date": first_match(clean, r"Exam Date\s*:\s*([0-9 A-Za-z./-]+)"),
                "shift": first_match(clean, r"Exam Shift\s*:\s*(\d+)"),
                "set_name": first_match(clean, r"Set Name\s*:\s*(.+?)(?:Exam Date|Langauge|Language|Section)"),
                "question_id": qid,
                "question_number": qn,
                "question_text": "",
                "option_a": "",
                "option_b": "",
                "option_c": "",
                "option_d": "",
                "correct_option": "",
                "answer_source": "",
                "question_type": match.group(4).upper(),
                "difficulty_estimate": "unknown",
                "unit": section,
                "chapter": "Unclassified - metadata only",
                "subtopic": "Question text not machine-readable",
                "canonical_question_id": stable_id(qid or f"{meta['name']}-{qn}"),
                "source_tier": meta["source_tier"],
                "source_weight": meta["source_weight"],
                "year_weight": year_weight(int(meta["year"])),
                "question_pattern": "metadata-only MCQ",
                "micro_concept": "needs OCR/manual import",
                "final_confidence": 0.15,
                "needs_review": True,
                "source_url": meta["source_url"],
                "raw_file_path": meta["raw_file_path"],
                "evidence_status": "metadata_only_question_id",
                "section_scope": "Computer Science" if section == "Section B1 Computer Science" else section,
            }
        )
    return rows


def parse_answer_key(text: str, meta: dict) -> list[dict]:
    rows = []
    clean = normalize_text(text)
    blocks = re.split(r"CUET\s*\(UG\)\s*\d{4}\s*:\s*Final Answer Keys", clean, flags=re.I)
    for block in blocks:
        if "Subject :308" not in block and "Subject : 308" not in block and "Computer" not in block[:300]:
            continue
        header = block[:600]
        if "Computer" not in header and "308" not in header:
            continue
        date = first_match(header, r"Exam Date\s*:\s*([0-9./-]+)")
        language = first_match(block, r"Question Id\.\s*Key\s*([A-Z][A-Za-z ]+?)\s+\d")
        for qid, key in re.findall(r"\b(\d{7,12})\s+(Drop|[1-4])\b", block):
            rows.append(
                {
                    "year": meta["year"],
                    "date": date,
                    "question_id": qid,
                    "correct_option": normalize_key(key),
                    "language": normalize_text(language) or "Unknown",
                    "source": meta["name"],
                    "source_tier": meta["source_tier"],
                    "source_weight": meta["source_weight"],
                    "source_url": meta["source_url"],
                    "raw_file_path": meta["raw_file_path"],
                }
            )
    return rows


def normalize_questions(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source",
        "year",
        "date",
        "shift",
        "set_name",
        "question_id",
        "question_number",
        "question_text",
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
        "canonical_question_id",
        "source_tier",
        "source_weight",
        "year_weight",
        "question_pattern",
        "micro_concept",
        "final_confidence",
        "needs_review",
        "source_url",
        "raw_file_path",
        "evidence_status",
        "section_scope",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in df.columns:
            df[column] = ""
    df = df[columns].copy()
    df = df.drop_duplicates(["question_id", "source"], keep="first")
    return df


def regenerate_analysis(questions: pd.DataFrame, answers: pd.DataFrame, manifest: pd.DataFrame) -> None:
    topics = json.loads(CONFIG.read_text(encoding="utf-8"))
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    taxonomy = build_taxonomy(topics)
    source_rows = build_sources(sources)
    priority = pd.read_csv(PROCESSED_DIR / "study_priority.csv") if (PROCESSED_DIR / "study_priority.csv").exists() else pd.DataFrame()
    if priority.empty:
        from init_cuet_cs import build_priority

        priority = build_priority(taxonomy)
    priority = priority.copy()
    question_count = len(questions)
    answer_count = len(answers)
    cs_section_count = int(questions["section_scope"].astype(str).eq("Computer Science").sum()) if not questions.empty else 0
    priority["pyq_evidence_questions"] = 0
    priority["answer_key_evidence_count"] = answer_count
    priority["metadata_question_rows"] = question_count
    priority["cs_section_question_rows"] = cs_section_count
    priority["evidence_status"] = "paper_metadata_imported_topic_needs_ocr_or_manual_text"
    priority["combined_score"] = priority["raw_score"].astype(float) + (0.15 if question_count else 0)
    priority.to_csv(PROCESSED_DIR / "study_priority.csv", index=False)
    topic_frequency = build_topic_frequency(taxonomy)
    topic_frequency["metadata_question_rows"] = question_count
    topic_frequency["answer_key_evidence_count"] = answer_count
    topic_frequency["evidence_status"] = "metadata-only imported; topic counts remain syllabus prior"
    topic_frequency.to_csv(PROCESSED_DIR / "topic_frequency.csv", index=False)
    strategy = build_strategy(taxonomy)
    strategy["evidence_status"] = "public papers imported, but question text is not reliably extractable"
    strategy.to_csv(PROCESSED_DIR / "question_format_strategy.csv", index=False)
    quality = build_quality_summary(taxonomy, source_rows)
    updates = {
        "total_parsed_rows": int(question_count),
        "unique_questions_after_dedupe": int(questions["question_id"].nunique()) if not questions.empty else 0,
        "official_source_rows": int(answer_count),
        "third_party_source_rows": int(question_count),
        "downloaded_public_files": int(manifest["status"].isin(["downloaded", "cached"]).sum()) if not manifest.empty else 0,
        "answer_key_entries": int(answer_count),
        "machine_readable_question_text_rows": int((questions["question_text"].astype(str).str.len() > 20).sum()) if not questions.empty else 0,
        "data_status": "Imported public CS papers/answer keys. Current papers expose question IDs/sections; most question text needs OCR/manual import before true topic frequency.",
    }
    for metric, value in updates.items():
        quality.loc[quality["metric"].eq(metric), "value"] = value
        if metric not in set(quality["metric"]):
            quality = pd.concat([quality, pd.DataFrame([{"metric": metric, "value": value}])], ignore_index=True)
    quality.to_csv(PROCESSED_DIR / "data_quality_summary.csv", index=False)
    summary = pd.DataFrame(
        [
            {"metric": "public_question_papers_imported", "value": int((manifest["kind"].eq("question_paper") & manifest["status"].isin(["downloaded", "cached"])).sum()) if not manifest.empty else 0},
            {"metric": "question_metadata_rows", "value": question_count},
            {"metric": "answer_key_entries", "value": answer_count},
            {"metric": "cs_section_question_rows", "value": cs_section_count},
            {"metric": "machine_readable_question_text_rows", "value": updates["machine_readable_question_text_rows"]},
            {"metric": "topic_frequency_validity", "value": "Not valid yet without OCR/manual question text."},
        ]
    )
    summary.to_csv(PROCESSED_DIR / "internet_evidence_summary.csv", index=False)


def section_ranges(text: str) -> list[tuple[str, int, int]]:
    ranges = []
    for match in re.finditer(r"(Compulsory|Computer Science)\s+Section Id.*?Number of Questions\s*:\s*(\d+)", text, flags=re.I | re.S):
        name = "Section A Common Core" if match.group(1).lower() == "compulsory" else "Section B1 Computer Science"
        count = int(match.group(2))
        ranges.append((name, count, match.start()))
    return ranges


def infer_section(question_number: int, ranges: list[tuple[str, int, int]]) -> str:
    if question_number <= 15:
        return "Section A Common Core"
    return "Section B1 Computer Science"


def first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.I | re.S)
    return normalize_text(match.group(1)) if match else ""


def year_weight(year: int) -> float:
    return {2025: 1.0, 2024: 0.8, 2023: 0.6, 2022: 0.45}.get(year, 0.5)


def normalize_key(key: str) -> str:
    if key.lower() == "drop":
        return "DROP"
    return {"1": "A", "2": "B", "3": "C", "4": "D"}.get(key, key)


def stable_id(value: str) -> str:
    return "cs_" + hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:16]


if __name__ == "__main__":
    main()
