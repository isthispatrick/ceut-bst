from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "cuet_cs_topics.json"
SOURCES = ROOT / "config" / "cuet_cs_sources.json"
DATA_DIR = ROOT / "data" / "cuet_cs"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_IMPORT_DIR = DATA_DIR / "manual_imports"
MANUAL_OFFICIAL_DIR = DATA_DIR / "manual_official_papers"
VERIFIED_DIR = DATA_DIR / "verified"
REPORTS_DIR = ROOT / "reports" / "cuet_cs"


def main() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, MANUAL_IMPORT_DIR, MANUAL_OFFICIAL_DIR, VERIFIED_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    topics = json.loads(CONFIG.read_text(encoding="utf-8"))
    sources = json.loads(SOURCES.read_text(encoding="utf-8"))
    taxonomy = build_taxonomy(topics)
    source_rows = build_sources(sources)

    write_csv("syllabus_taxonomy.csv", taxonomy)
    write_csv("ncert_reverse_index.csv", taxonomy)
    write_csv("source_discovery.csv", source_rows)
    write_csv("study_priority.csv", build_priority(taxonomy))
    write_csv("topic_frequency.csv", build_topic_frequency(taxonomy))
    write_csv("repeated_concepts.csv", build_repeated_concepts(taxonomy))
    write_csv("question_format_strategy.csv", build_strategy(taxonomy))
    write_csv("data_quality_summary.csv", build_quality_summary(taxonomy, source_rows))
    write_csv("questions_advanced.csv", empty_questions())
    write_csv("suspicious_classifications.csv", pd.DataFrame(columns=["canonical_question_id", "chapter", "subtopic", "review_reason"]))
    write_csv("accuracy_summary.csv", pd.DataFrame([{"metric": "status", "value": "No parsed CUET CS PYQ labels yet."}]))

    (MANUAL_IMPORT_DIR / "README.md").write_text(
        "Put CUET Computer Science / Informatics Practices PDFs, HTML files, or CSV exports here.\n"
        "Then extend the parser or adapt the BST pipeline to create data/cuet_cs/processed/questions_advanced.csv.\n",
        encoding="utf-8",
    )
    (MANUAL_OFFICIAL_DIR / "README.md").write_text(
        "Place manually downloaded official NTA CUET Computer Science / Informatics Practices PDFs here.\n"
        "Do not bypass NTA blocking, login, paywalls, CAPTCHA, or anti-bot restrictions.\n",
        encoding="utf-8",
    )
    print(f"Initialized CUET CS dashboard data in {PROCESSED_DIR}")


def build_taxonomy(config: dict) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    section_counts: dict[str, int] = {}
    chapter_counts: dict[str, int] = {}
    for section in config["sections"]:
        for chapter in section["chapters"]:
            chapter_counts[chapter["chapter"]] = chapter_counts.get(chapter["chapter"], 0) + 1
            for subtopic in chapter["subtopics"]:
                section_counts[subtopic] = section_counts.get(subtopic, 0) + 1
                rows.append(
                    {
                        "subject_code": config["subject_code"],
                        "subject_name": config["subject_name"],
                        "section": section["section"],
                        "unit": section["unit"],
                        "chapter": chapter["chapter"],
                        "subtopic": subtopic,
                        "ncert_heading": chapter["chapter"],
                        "ncert_concept": subtopic,
                        "concept_type": infer_concept_type(chapter["chapter"], subtopic),
                        "official_syllabus_url": config["official_syllabus_url"],
                    }
                )
    df = pd.DataFrame(rows)
    df["syllabus_overlap_score"] = df["subtopic"].map(section_counts).fillna(1).astype(float)
    df["chapter_overlap_score"] = df["chapter"].map(chapter_counts).fillna(1).astype(float)
    return df


def build_sources(config: dict) -> pd.DataFrame:
    return pd.DataFrame(config["sources"])


def build_priority(taxonomy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = taxonomy.groupby(["chapter", "subtopic"], dropna=False).agg(
        section_count=("section", "nunique"),
        section_list=("section", lambda values: ", ".join(sorted(set(map(str, values))))),
        unit_list=("unit", lambda values: ", ".join(sorted(set(map(str, values))))),
        syllabus_overlap_score=("syllabus_overlap_score", "max"),
        chapter_overlap_score=("chapter_overlap_score", "max"),
        concept_type=("concept_type", "first"),
    )
    for (chapter, subtopic), row in grouped.reset_index().set_index(["chapter", "subtopic"]).iterrows():
        overlap = float(row["syllabus_overlap_score"]) + float(row["chapter_overlap_score"]) * 0.25
        if overlap >= 2.5:
            tier = "Syllabus Tier 1 = high overlap"
        elif overlap >= 1.5:
            tier = "Syllabus Tier 2 = medium overlap"
        else:
            tier = "Needs PYQ data"
        rows.append(
            {
                "chapter": chapter,
                "subtopic": subtopic,
                "priority_tier": tier,
                "raw_score": round(overlap, 2),
                "source_weighted_frequency": 0,
                "recency_score": 0,
                "question_count": 0,
                "roi": 0,
                "top_pattern": suggested_pattern(str(chapter), str(subtopic)),
                "top_micro_concept": subtopic,
                "difficulty": suggested_difficulty(str(chapter), str(subtopic)),
                "study_hours_required": suggested_hours(str(chapter), str(subtopic)),
                "recommended_action": recommended_action(str(chapter), str(subtopic), tier),
                "data_status": "syllabus_only_no_pyq_frequency_yet",
                "section_list": row["section_list"],
                "unit_list": row["unit_list"],
                "concept_type": row["concept_type"],
            }
        )
    return pd.DataFrame(rows).sort_values(["raw_score", "chapter", "subtopic"], ascending=[False, True, True])


def build_topic_frequency(taxonomy: pd.DataFrame) -> pd.DataFrame:
    priority = build_priority(taxonomy)
    return priority[["chapter", "subtopic", "question_count", "source_weighted_frequency", "raw_score", "data_status"]].copy()


def build_repeated_concepts(taxonomy: pd.DataFrame) -> pd.DataFrame:
    priority = build_priority(taxonomy)
    return priority.rename(columns={"top_micro_concept": "micro_concept"})[
        ["chapter", "subtopic", "micro_concept", "question_count", "source_weighted_frequency", "raw_score", "data_status"]
    ]


def build_strategy(taxonomy: pd.DataFrame) -> pd.DataFrame:
    priority = build_priority(taxonomy)
    priority["dominant_question_pattern"] = priority["top_pattern"]
    priority["how_to_study_it"] = priority.apply(lambda row: how_to_study(str(row["chapter"]), str(row["subtopic"])), axis=1)
    priority["common_traps"] = priority.apply(lambda row: common_traps(str(row["chapter"]), str(row["subtopic"])), axis=1)
    priority["ncert_heading_to_revise"] = priority["chapter"]
    return priority[
        [
            "chapter",
            "subtopic",
            "priority_tier",
            "dominant_question_pattern",
            "how_to_study_it",
            "common_traps",
            "ncert_heading_to_revise",
            "data_status",
        ]
    ]


def build_quality_summary(taxonomy: pd.DataFrame, sources: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("subject_code", "308"),
        ("subject_name", "Computer Science / Information Practices"),
        ("total_parsed_rows", 0),
        ("unique_questions_after_dedupe", 0),
        ("duplicate_rate_percent", 0),
        ("official_source_rows", 0),
        ("third_party_source_rows", 0),
        ("syllabus_topics", len(taxonomy)),
        ("chapters", taxonomy["chapter"].nunique()),
        ("configured_sources", len(sources)),
        ("data_status", "Initialized from official syllabus taxonomy; PYQ collection not run yet."),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def empty_questions() -> pd.DataFrame:
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
    ]
    return pd.DataFrame(columns=columns)


def infer_concept_type(chapter: str, subtopic: str) -> str:
    text = f"{chapter} {subtopic}".lower()
    if any(term in text for term in ["sql", "select", "join", "group", "function", "mysql"]):
        return "query-practice"
    if any(term in text for term in ["stack", "queue", "search", "sort", "hash", "python"]):
        return "algorithm-dry-run"
    if any(term in text for term in ["network", "protocol", "security", "cyber", "firewall"]):
        return "concept-application"
    if any(term in text for term in ["pandas", "matplotlib", "dataframe", "series", "plot"]):
        return "library-output"
    return "definition-feature"


def suggested_pattern(chapter: str, subtopic: str) -> str:
    concept_type = infer_concept_type(chapter, subtopic)
    return {
        "query-practice": "SQL output / query identification",
        "algorithm-dry-run": "dry-run / output tracing",
        "concept-application": "definition + application",
        "library-output": "code/output interpretation",
        "definition-feature": "definition-based",
    }[concept_type]


def suggested_difficulty(chapter: str, subtopic: str) -> str:
    text = f"{chapter} {subtopic}".lower()
    if any(term in text for term in ["join", "having", "postfix", "binary search", "hash", "pivot", "missing values"]):
        return "medium"
    return "easy"


def suggested_hours(chapter: str, subtopic: str) -> float:
    return 2.5 if suggested_difficulty(chapter, subtopic) == "medium" else 1.5


def recommended_action(chapter: str, subtopic: str, tier: str) -> str:
    if "SQL" in chapter or "sql" in subtopic.lower():
        return "Practice query outputs, clauses, joins, functions, and error traps."
    if any(term in chapter.lower() for term in ["stack", "queue", "search", "sort"]):
        return "Do dry runs by hand and memorize operation order and edge cases."
    if "pandas" in chapter.lower() or "matplotlib" in chapter.lower():
        return "Practice small code snippets and identify expected DataFrame or plot output."
    if "network" in chapter.lower() or "security" in chapter.lower():
        return "Revise definitions, devices, protocols, threats, and one-line differences."
    return "Revise from syllabus and add PYQs before trusting frequency."


def how_to_study(chapter: str, subtopic: str) -> str:
    if "SQL" in chapter or "sql" in subtopic.lower():
        return "Write the clause order, then solve 15-20 SELECT/function/grouping questions."
    if any(term in chapter.lower() for term in ["stack", "queue", "search", "sort"]):
        return "Trace operations in a table: input, pointer/index, stack/queue state, output."
    if "pandas" in chapter.lower() or "matplotlib" in chapter.lower():
        return "Run or mentally trace short snippets; focus on Series/DataFrame shape and labels."
    if "network" in chapter.lower() or "security" in chapter.lower():
        return "Make a compare table for devices, topologies, protocols, threats, and prevention."
    return "Make concise notes, then attach PYQ examples after import."


def common_traps(chapter: str, subtopic: str) -> str:
    text = f"{chapter} {subtopic}".lower()
    if "join" in text or "group" in text or "having" in text:
        return "Confusing WHERE vs HAVING, aggregate vs row-level filters, and join output size."
    if "key" in text:
        return "Mixing candidate, primary, alternate, and foreign keys."
    if "stack" in text or "queue" in text:
        return "Reversing LIFO/FIFO or losing one operation in a dry run."
    if "sort" in text or "search" in text:
        return "Wrong best/worst case and off-by-one index changes."
    if "pandas" in text:
        return "Confusing label indexing with positional indexing and Series vs DataFrame output."
    if "network" in text:
        return "Mixing devices, topologies, and internet/web terminology."
    return "Overtrusting syllabus-only priority before PYQ import."


def write_csv(name: str, data: pd.DataFrame) -> None:
    data.to_csv(PROCESSED_DIR / name, index=False)


if __name__ == "__main__":
    main()
