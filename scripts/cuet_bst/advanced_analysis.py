from __future__ import annotations

import logging
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .settings import PROCESSED_DIR, REPORTS_DIR, ensure_dirs
from .validation import apply_taxonomy_validation, run_validation_outputs


def run_advanced_analysis(logger: logging.Logger | None = None) -> dict[str, Path]:
    ensure_dirs()
    logger = logger or logging.getLogger(__name__)
    input_path = PROCESSED_DIR / "questions_advanced.csv"
    if not input_path.exists():
        raise FileNotFoundError("Missing data/processed/questions_advanced.csv. Run python scripts/classify_ensemble.py first.")
    df = pd.read_csv(input_path).fillna("")
    df = apply_taxonomy_validation(df)
    df.to_csv(input_path, index=False)
    df.to_json(PROCESSED_DIR / "questions_advanced.json", orient="records", indent=2, force_ascii=False)
    outputs: dict[str, Path] = {}
    outputs["data_quality"] = _write_advanced_data_quality(df)
    outputs["study_priority"] = _write_study_priority(df)
    outputs["high_roi_topics"] = _write_high_roi()
    outputs["low_frequency_topics"] = _write_low_frequency(df)
    outputs["reddit_insights"] = _write_reddit_insights(df)
    outputs["hidden_gems"] = _write_hidden_and_overhyped("hidden_gems")
    outputs["overhyped_topics"] = _write_hidden_and_overhyped("overhyped_topics")
    outputs["study_plans"] = _write_study_plans()
    run_validation_outputs(df)
    outputs["human_gold_review_queue"] = PROCESSED_DIR / "human_gold_review_queue.csv"
    outputs["accuracy_summary"] = PROCESSED_DIR / "accuracy_summary.csv"
    outputs["suspicious_classifications"] = PROCESSED_DIR / "suspicious_classifications.csv"
    outputs["question_format_strategy"] = PROCESSED_DIR / "question_format_strategy.csv"
    outputs["final_study_pack"] = REPORTS_DIR / "cuet_bst_final_study_pack.pdf"
    outputs["advanced_report"] = _write_pdf_report(df)
    outputs["advanced_database"] = _write_database(df, logger)
    logger.info("Advanced analysis complete for %s canonical questions", len(df))
    return outputs


def _write_advanced_data_quality(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "data_quality_summary.csv"
    raw_path = PROCESSED_DIR / "questions_with_canonical_ids.csv"
    raw = pd.read_csv(raw_path).fillna("") if raw_path.exists() else df
    raw_rows = len(raw)
    unique_questions = len(df)
    duplicate_rate = 0 if raw_rows == 0 else round((raw_rows - unique_questions) / raw_rows * 100, 2)
    review_count = int(df.get("needs_review", pd.Series(dtype=str)).astype(str).str.lower().isin(["true", "yes"]).sum())
    official = int(raw.get("source_tier", pd.Series(dtype=str)).isin(["official_nta", "official_manual", "official_syllabus_ncert"]).sum())
    third_party = raw_rows - official
    rows = [
        {"metric": "total_parsed_rows", "value": raw_rows},
        {"metric": "unique_questions_after_dedupe", "value": unique_questions},
        {"metric": "duplicate_rate_percent", "value": duplicate_rate},
        {"metric": "official_source_rows", "value": official},
        {"metric": "third_party_source_rows", "value": third_party},
        {"metric": "low_confidence_classifications_lt_075", "value": review_count},
        {"metric": "manual_review_queue_count", "value": review_count},
    ]
    if "source_tier" in raw.columns:
        for tier, count in raw["source_tier"].value_counts().items():
            rows.append({"metric": f"source_tier_{tier}", "value": int(count)})
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _write_study_priority(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "study_priority.csv"
    if df.empty:
        pd.DataFrame().to_csv(out, index=False)
        return out
    rows = []
    for (chapter, subtopic), group in df.groupby(["chapter", "subtopic"], dropna=False):
        source_weighted_frequency = float(pd.to_numeric(group["weighted_frequency_score"], errors="coerce").fillna(0).sum())
        recency_score = float(pd.to_numeric(group["recency_weighted_score"], errors="coerce").fillna(0).sum())
        years = {str(year).replace(".0", "") for year in group["year"] if str(year).strip()}
        shifts = {str(shift) for shift in group["shift"] if str(shift).strip()}
        patterns = {str(pattern) for pattern in group["question_pattern"] if str(pattern).strip()}
        cross_year_consistency = min(len(years) / 4, 1.0)
        cross_shift_consistency = min(len(shifts) / 8, 1.0)
        question_type_diversity = min(len(patterns) / 5, 1.0)
        ncert_directness = _ncert_directness(group)
        student_confusion_score = _student_confusion(str(subtopic))
        raw_score = (
            0.30 * _scale(source_weighted_frequency, 12)
            + 0.25 * _scale(recency_score, 10)
            + 0.15 * cross_year_consistency
            + 0.10 * cross_shift_consistency
            + 0.10 * question_type_diversity
            + 0.05 * ncert_directness
            + 0.05 * student_confusion_score
        )
        difficulty = _dominant(group["difficulty_estimate"])
        hours = _study_hours_required(group)
        expected_marks = round(raw_score * 100, 2)
        roi = round(expected_marks / max(hours, 0.5), 2)
        rows.append(
            {
                "chapter": chapter,
                "subtopic": subtopic,
                "question_count": len(group),
                "source_weighted_frequency": round(source_weighted_frequency, 3),
                "recency_score": round(recency_score, 3),
                "cross_year_consistency": round(cross_year_consistency, 3),
                "cross_shift_consistency": round(cross_shift_consistency, 3),
                "question_type_diversity": round(question_type_diversity, 3),
                "ncert_directness": round(ncert_directness, 3),
                "student_confusion_score": round(student_confusion_score, 3),
                "raw_score": round(raw_score, 4),
                "study_priority_score": round(raw_score * 100, 2),
                "difficulty": difficulty,
                "study_hours_required": hours,
                "expected_marks_contribution": expected_marks,
                "roi": roi,
                "top_pattern": _dominant(group["question_pattern"]),
                "top_micro_concept": _dominant(group["micro_concept"]),
                "recommended_action": _recommended_action(raw_score, difficulty, _dominant(group["question_pattern"])),
            }
        )
    priority = pd.DataFrame(rows).sort_values(["raw_score", "roi"], ascending=False)
    if not priority.empty:
        priority["percentile_rank"] = priority["raw_score"].rank(pct=True, method="max").round(4)
        priority["priority_tier"] = priority["percentile_rank"].map(_priority_tier)
    priority.to_csv(out, index=False)
    return out


def _write_high_roi() -> Path:
    priority = _priority()
    out = PROCESSED_DIR / "high_roi_topics.csv"
    if priority.empty:
        priority.to_csv(out, index=False)
        return out
    high = priority[(priority["roi"] >= priority["roi"].quantile(0.65)) & (priority["study_priority_score"] >= priority["study_priority_score"].median())]
    high.sort_values(["roi", "study_priority_score"], ascending=False).to_csv(out, index=False)
    return out


def _write_low_frequency(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "low_frequency_topics.csv"
    if df.empty:
        pd.DataFrame().to_csv(out, index=False)
        return out
    rows = []
    for (chapter, subtopic), group in df.groupby(["chapter", "subtopic"], dropna=False):
        years = pd.to_numeric(group["year"], errors="coerce").dropna()
        count = len(group)
        if count > 3:
            continue
        difficulty = _dominant(group["difficulty_estimate"])
        last_seen = int(years.max()) if len(years) else ""
        if count == 0:
            action = "Skip for now"
        elif difficulty == "easy":
            action = "Do NCERT summary and one PYQ pass"
        elif last_seen and int(last_seen) >= 2025:
            action = "Study after high-priority topics"
        else:
            action = "Revise one day before exam"
        rows.append(
            {
                "chapter": chapter,
                "subtopic": subtopic,
                "question_count": count,
                "last_seen_year": last_seen,
                "difficulty": difficulty,
                "study_action": action,
            }
        )
    pd.DataFrame(rows).sort_values(["question_count", "last_seen_year"], ascending=[True, False]).to_csv(out, index=False)
    return out


def _write_reddit_insights(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "reddit_insights.csv"
    if df.empty:
        pd.DataFrame().to_csv(out, index=False)
        return out
    reddit_text = _reddit_text()
    actual = (
        df.groupby(["chapter", "subtopic"], dropna=False)
        .agg(actual_weighted_frequency=("weighted_frequency_score", "sum"), actual_count=("canonical_question_id", "count"))
        .reset_index()
    )
    rows = []
    for _, row in actual.iterrows():
        topic = f"{row['chapter']} {row['subtopic']}".lower()
        terms = [term for term in topic.replace("and", " ").split() if len(term) > 4]
        hype = sum(reddit_text.count(term) for term in terms)
        freq = float(row["actual_weighted_frequency"])
        if hype > 0 and freq >= actual["actual_weighted_frequency"].quantile(0.60):
            mismatch = "validated_by_data"
        elif hype > 2 and freq < actual["actual_weighted_frequency"].median():
            mismatch = "overhyped"
        elif hype <= 1 and freq >= actual["actual_weighted_frequency"].quantile(0.70):
            mismatch = "underhyped_hidden_gem"
        else:
            mismatch = "insufficient_data" if hype == 0 else "mixed_signal"
        rows.append(
            {
                "topic": str(row["subtopic"]),
                "chapter": str(row["chapter"]),
                "reddit_hype_score": hype,
                "actual_weighted_frequency": round(freq, 3),
                "actual_pyq_frequency": int(row["actual_count"]),
                "mismatch_type": mismatch,
            }
        )
    pd.DataFrame(rows).sort_values(["actual_weighted_frequency", "reddit_hype_score"], ascending=False).to_csv(out, index=False)
    return out


def _write_hidden_and_overhyped(kind: str) -> Path:
    reddit = _read_csv("reddit_insights.csv")
    out = PROCESSED_DIR / f"{kind}.csv"
    if reddit.empty:
        reddit.to_csv(out, index=False)
        return out
    key = "underhyped_hidden_gem" if kind == "hidden_gems" else "overhyped"
    reddit[reddit["mismatch_type"] == key].sort_values("actual_weighted_frequency", ascending=False).to_csv(out, index=False)
    return out


def _write_study_plans() -> Path:
    priority = _priority()
    out = PROCESSED_DIR / "study_plans.csv"
    if priority.empty:
        priority.to_csv(out, index=False)
        return out
    rows = []
    for days in [3, 5, 7, 14]:
        remaining = 4.0
        day = 1
        for _, topic in priority.sort_values(["roi", "study_priority_score"], ascending=False).iterrows():
            hours = float(topic.get("study_hours_required", 1.0))
            while hours > 0 and day <= days:
                allocation = min(hours, remaining)
                rows.append(
                    {
                        "plan_days": days,
                        "day": day,
                        "allocated_hours": round(allocation, 2),
                        "chapter": topic["chapter"],
                        "subtopic": topic["subtopic"],
                        "expected_marks_contribution": topic["expected_marks_contribution"],
                        "roi": topic["roi"],
                        "recommended_action": topic["recommended_action"],
                    }
                )
                hours -= allocation
                remaining -= allocation
                if remaining <= 0.05:
                    day += 1
                    remaining = 4.0
                if day > days:
                    break
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _write_pdf_report(df: pd.DataFrame) -> Path:
    out = REPORTS_DIR / "cuet_bst_analysis_report.pdf"
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        styles = getSampleStyleSheet()
        review_count = int(df.get("needs_review", pd.Series(dtype=str)).astype(str).str.lower().isin(["true", "yes"]).sum())
        llm_values = df.get("llm_label", pd.Series(dtype=str)).astype(str)
        usable_llm = int((~llm_values.isin(["", "skipped_or_unavailable"])).sum())
        story = [
            Paragraph("CUET BST Exam-Intelligence Report", styles["Title"]),
            Paragraph("Historical PYQ analysis only. This is not a guaranteed prediction of future CUET papers.", styles["BodyText"]),
            Spacer(1, 10),
            Paragraph("Method Used", styles["Heading2"]),
            Paragraph(
                "The system first collects public PYQ rows, preserves raw files, parses questions/options, then deduplicates copied questions into canonical question IDs. "
                "Each canonical question is classified with a hybrid ensemble: syllabus keyword rules, TF-IDF/BM25 retrieval against the CUET BST taxonomy, embedding-style similarity, and optional OpenAI-compatible LLM classification. "
                "Final study rankings use canonical unique questions, source reliability weights, recency weights, cross-year consistency, cross-shift consistency, question-pattern diversity, NCERT directness, and weak Reddit/community sentiment.",
                styles["BodyText"],
            ),
            Spacer(1, 8),
            Paragraph("How To Use This Report", styles["Heading2"]),
            Paragraph(
                "Start with high-ROI and high-priority topics, then practice the dominant question pattern for each topic. "
                "For example, if a concept mostly appears as case diagnosis, learn examples and identification cues, not just definitions. "
                "Rows still marked for review should be treated as useful AI suggestions rather than final truth until manually checked.",
                styles["BodyText"],
            ),
            Spacer(1, 8),
        ]
        quality = _read_csv("data_quality_summary.csv")
        priority = _priority()
        clusters = _read_csv("micro_concept_clusters.csv")
        hidden = _read_csv("hidden_gems.csv")
        overhyped = _read_csv("overhyped_topics.csv")
        plans = _read_csv("study_plans.csv")
        review_summary = pd.DataFrame(
            [
                {"metric": "canonical_questions", "value": len(df)},
                {"metric": "usable_llm_labels", "value": usable_llm},
                {"metric": "ai_review_suggestions_remaining", "value": review_count},
                {"metric": "report_generated_utc", "value": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")},
            ]
        )
        chapter_weighted = (
            df.assign(weighted_frequency_score=pd.to_numeric(df.get("weighted_frequency_score", 0), errors="coerce").fillna(0))
            .groupby("chapter", dropna=False)
            .agg(unique_questions=("canonical_question_id", "count"), weighted_score=("weighted_frequency_score", "sum"))
            .reset_index()
            .sort_values("weighted_score", ascending=False)
            .head(12)
        )
        for title, table in [
            ("Data Quality", quality),
            ("AI Review Status", review_summary),
            ("Top Chapters by Weighted Score", chapter_weighted),
            ("Top Topics by Weighted Study Priority", priority.head(15)),
            ("Most Repeated Micro-Concepts", clusters.head(15)),
            ("Hidden Gems", hidden.head(10)),
            ("Overhyped Topics", overhyped.head(10)),
            ("5-Day Study Sequence", plans[plans.get("plan_days", pd.Series(dtype=str)).astype(str) == "5"].head(12) if not plans.empty else plans),
        ]:
            story.append(Paragraph(title, styles["Heading2"]))
            story.append(_report_table(table))
            story.append(Spacer(1, 8))
        doc = SimpleDocTemplate(str(out), pagesize=A4, title="CUET BST Exam Intelligence")
        doc.build(story)
    except Exception:
        out.write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")
    return out


def _write_database(df: pd.DataFrame, logger: logging.Logger) -> Path:
    out = PROCESSED_DIR / "cuet_bst_advanced.duckdb"
    try:
        import duckdb

        try:
            con = duckdb.connect(str(out))
        except Exception:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            out = PROCESSED_DIR / f"cuet_bst_advanced_{stamp}.duckdb"
            con = duckdb.connect(str(out))
        con.execute("CREATE OR REPLACE TABLE questions_advanced AS SELECT * FROM df")
        for name in ["study_priority", "micro_concept_clusters", "reddit_insights", "study_plans"]:
            path = PROCESSED_DIR / f"{name}.csv"
            if path.exists():
                con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_csv_auto('{path.as_posix()}')")
        con.close()
    except Exception as exc:
        logger.warning("Could not write advanced DuckDB file: %s", exc)
        return PROCESSED_DIR / "questions_advanced.csv"
    return out


def _ncert_directness(group: pd.DataFrame) -> float:
    direct_types = {"definition", "feature", "principle", "process"}
    values = [1.0 if str(value) in direct_types else 0.65 for value in group["concept_type"]]
    return float(np.mean(values)) if values else 0.0


def _student_confusion(topic: str) -> float:
    text = _reddit_text()
    if not text:
        return 0.0
    terms = [term for term in topic.lower().split() if len(term) > 4]
    if not terms:
        return 0.0
    return min(sum(text.count(term) for term in terms) / 10, 1.0)


def _study_hours_required(group: pd.DataFrame) -> float:
    difficulty = _dominant(group["difficulty_estimate"])
    base = {"easy": 0.75, "medium": 1.25, "hard": 1.75}.get(difficulty, 1.25)
    diversity = min(group["question_pattern"].nunique(), 5) * 0.2
    return round(base + diversity, 2)


def _recommended_action(score: float, difficulty: str, pattern: str) -> str:
    if score >= 0.65 and "case" in pattern:
        return "Learn NCERT line plus practice case diagnosis examples"
    if score >= 0.55:
        return "Study deeply and solve all available PYQs"
    if difficulty == "easy":
        return "Memorize NCERT summary and common examples"
    return "Study after high-priority topics"


def _dominant(series: pd.Series) -> str:
    values = series.astype(str).replace("", pd.NA).dropna()
    return str(values.mode().iloc[0]) if not values.empty else ""


def _scale(value: float, cap: float) -> float:
    return max(0.0, min(value / cap, 1.0))


def _priority_tier(percentile: float) -> str:
    if percentile >= 0.85:
        return "Tier 1 = Must do"
    if percentile >= 0.65:
        return "Tier 2 = High ROI"
    if percentile >= 0.35:
        return "Tier 3 = Medium priority"
    return "Tier 4 = Low priority"


def _priority() -> pd.DataFrame:
    return _read_csv("study_priority.csv")


def _read_csv(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def _reddit_text() -> str:
    path = PROCESSED_DIR / "reddit_discussions.csv"
    if not path.exists():
        return ""
    reddit = pd.read_csv(path).fillna("")
    return " ".join(reddit.astype(str).agg(" ".join, axis=1).tolist()).lower()


def _report_table(df: pd.DataFrame):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    if df is None or df.empty:
        df = pd.DataFrame([{"status": "No data yet"}])
    data = [list(df.columns)] + df.astype(str).head(18).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("FONTSIZE", (0, 0), (-1, -1), 6),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table
