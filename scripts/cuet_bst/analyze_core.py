from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from datetime import datetime, timezone

import pandas as pd
from pandas.errors import EmptyDataError

from .settings import DB_PATH, PROCESSED_DIR, QUESTION_COLUMNS, REPORTS_DIR, ensure_dirs


def run_analysis(logger: logging.Logger | None = None) -> dict[str, Path]:
    ensure_dirs()
    logger = logger or logging.getLogger(__name__)
    questions_path = PROCESSED_DIR / "questions.csv"
    if not questions_path.exists():
        raise FileNotFoundError(f"Run scripts/process_questions.py first. Missing: {questions_path}")
    try:
        df = pd.read_csv(questions_path).fillna("")
    except EmptyDataError:
        df = pd.DataFrame(columns=QUESTION_COLUMNS)
    outputs: dict[str, Path] = {}

    outputs["topic_frequency"] = _write_topic_frequency(df)
    outputs["repeated_concepts"] = _write_repeated_concepts(df)
    outputs["chapter_year_heatmap"] = _write_pivot(df, "chapter", "year", "chapter_year_heatmap.csv")
    outputs["subtopic_year_heatmap"] = _write_pivot(df, "subtopic", "year", "subtopic_year_heatmap.csv")
    outputs["question_type_by_chapter"] = _write_pivot(df, "chapter", "question_type", "question_type_by_chapter.csv")
    outputs["chapter_correlation"] = _write_chapter_correlation(df)
    outputs["study_priority"] = _write_study_priority(df)
    outputs["reddit_insights"] = _write_reddit_insights(df)
    outputs["database"] = _write_database(df, logger)
    outputs["report"] = generate_pdf_report(df, outputs)
    return outputs


def _write_topic_frequency(df: pd.DataFrame) -> Path:
    group_cols = ["unit", "chapter", "subtopic"]
    freq = (
        df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="question_count")
        .sort_values("question_count", ascending=False)
    )
    if "year" in df.columns:
        years = df[df["year"].astype(str) != ""].groupby(group_cols)["year"].nunique().reset_index(name="years_seen")
        freq = freq.merge(years, on=group_cols, how="left").fillna({"years_seen": 0})
    out = PROCESSED_DIR / "topic_frequency.csv"
    freq.to_csv(out, index=False)
    return out


def _write_repeated_concepts(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "repeated_concepts.csv"
    columns = ["repeated_concept_cluster", "question_count", "years_seen", "year_count", "chapters", "example_question"]
    if df.empty:
        pd.DataFrame(columns=columns).to_csv(out, index=False)
        return out
    rows = []
    for cluster, group in df.groupby("repeated_concept_cluster", dropna=False):
        if not str(cluster).strip():
            continue
        years = sorted({str(year) for year in group["year"] if str(year).strip()})
        rows.append(
            {
                "repeated_concept_cluster": cluster,
                "question_count": len(group),
                "years_seen": ", ".join(years),
                "year_count": len(years),
                "chapters": ", ".join(sorted({str(value) for value in group["chapter"] if str(value).strip()})),
                "example_question": str(group.iloc[0].get("question_text", ""))[:500],
            }
        )
    pd.DataFrame(rows, columns=columns).sort_values(["question_count", "year_count"], ascending=False).to_csv(out, index=False)
    return out


def _write_pivot(df: pd.DataFrame, index: str, columns: str, filename: str) -> Path:
    if df.empty or index not in df.columns or columns not in df.columns:
        pivot = pd.DataFrame()
    else:
        pivot = pd.crosstab(df[index].replace("", "Unknown"), df[columns].replace("", "Unknown"))
    out = PROCESSED_DIR / filename
    pivot.to_csv(out)
    return out


def _write_chapter_correlation(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "chapter_correlation.csv"
    if df.empty or "year" not in df.columns or "chapter" not in df.columns:
        pd.DataFrame().to_csv(out)
        return out
    usable = df[(df["year"].astype(str) != "") & (df["chapter"].astype(str) != "")]
    matrix = pd.crosstab(usable["year"], usable["chapter"])
    corr = matrix.corr().fillna(0)
    corr.to_csv(out)
    return out


def _write_study_priority(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "study_priority.csv"
    if df.empty:
        pd.DataFrame().to_csv(out, index=False)
        return out
    max_year = pd.to_numeric(df["year"], errors="coerce").max()
    if pd.isna(max_year):
        max_year = 0
    rows: list[dict[str, Any]] = []
    for (chapter, subtopic), group in df.groupby(["chapter", "subtopic"], dropna=False):
        years = pd.to_numeric(group["year"], errors="coerce").dropna()
        frequency = len(group)
        recency = float(((years - max_year + 1).clip(lower=0) + 1).sum()) if len(years) else 0.0
        type_bonus = 1.0 + 0.15 * group["question_type"].isin(["case-based", "assertion-reason", "match-the-following"]).mean()
        priority_score = round((frequency * 1.0 + recency * 0.7) * type_bonus, 2)
        rows.append(
            {
                "chapter": chapter or "Unknown",
                "subtopic": subtopic or "Unknown",
                "question_count": frequency,
                "recent_weight": round(recency, 2),
                "priority_score": priority_score,
                "probability_estimate": round(frequency / max(len(df), 1), 4),
                "suggested_rank_reason": "High frequency + recent appearance" if priority_score >= 3 else "Lower historical signal",
            }
        )
    pd.DataFrame(rows).sort_values("priority_score", ascending=False).to_csv(out, index=False)
    return out


def _write_reddit_insights(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "reddit_insights.csv"
    topic_path = PROCESSED_DIR / "reddit_discussions.csv"
    if topic_path.exists():
        reddit = pd.read_csv(topic_path).fillna("")
        text = " ".join(reddit.astype(str).agg(" ".join, axis=1).tolist()).lower()
    else:
        text = ""
    actual = df.groupby("subtopic").size().to_dict() if not df.empty else {}
    rows = []
    for topic, count in sorted(actual.items(), key=lambda item: item[1], reverse=True):
        topic_text = str(topic).lower()
        terms = [term for term in topic_text.replace("and", " ").split() if len(term) > 3]
        hype = sum(text.count(term) for term in terms)
        validation = "validated" if hype and count else "actual-only"
        if hype > count * 2:
            validation = "possibly overhyped"
        elif count > hype * 2 and count >= 2:
            validation = "under-discussed"
        rows.append(
            {
                "topic": topic,
                "reddit_hype_score": hype,
                "actual_pyq_frequency": count,
                "mismatch_validation": validation,
                "evidence_terms": ", ".join(terms[:6]),
            }
        )
    if not rows:
        rows.append(
            {
                "topic": "No Reddit import yet",
                "reddit_hype_score": 0,
                "actual_pyq_frequency": 0,
                "mismatch_validation": "manual Reddit/search import needed",
                "evidence_terms": "",
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _write_database(df: pd.DataFrame, logger: logging.Logger) -> Path:
    try:
        import duckdb

        try:
            con = duckdb.connect(str(DB_PATH))
            con.execute("CREATE OR REPLACE TABLE questions AS SELECT * FROM df")
            con.close()
        except Exception as primary_exc:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
            fallback_db = PROCESSED_DIR / f"cuet_bst_{stamp}.duckdb"
            logger.warning("Primary DuckDB file unavailable, writing fallback DuckDB: %s", primary_exc)
            con = duckdb.connect(str(fallback_db))
            con.execute("CREATE OR REPLACE TABLE questions AS SELECT * FROM df")
            con.close()
            return fallback_db
    except Exception as exc:
        sqlite_path = PROCESSED_DIR / "cuet_bst.sqlite"
        logger.warning("DuckDB unavailable, writing SQLite fallback: %s", exc)
        import sqlite3

        try:
            with sqlite3.connect(sqlite_path) as con:
                df.to_sql("questions", con, if_exists="replace", index=False)
            return sqlite_path
        except Exception as sqlite_exc:
            logger.warning("SQLite fallback failed; CSV outputs remain authoritative: %s", sqlite_exc)
            return PROCESSED_DIR / "questions.csv"
    return DB_PATH


def generate_pdf_report(df: pd.DataFrame, outputs: dict[str, Path]) -> Path:
    out = REPORTS_DIR / "cuet_bst_analysis_report.pdf"
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        doc = SimpleDocTemplate(str(out), pagesize=A4, title="CUET Business Studies PYQ Analysis")
        styles = getSampleStyleSheet()
        story = [
            Paragraph("CUET UG Business Studies PYQ Analysis", styles["Title"]),
            Paragraph("Subject code 305. Correlation and frequency are historical signals, not guaranteed predictions.", styles["BodyText"]),
            Spacer(1, 12),
        ]
        story.append(Paragraph("Top 10 Repeated Chapters", styles["Heading2"]))
        chapter = df.groupby("chapter").size().sort_values(ascending=False).head(10).reset_index(name="questions") if not df.empty else pd.DataFrame(columns=["chapter", "questions"])
        story.append(_report_table(chapter))
        story.append(Spacer(1, 10))

        story.append(Paragraph("Top 30 Repeated Subtopics", styles["Heading2"]))
        subtopic = df.groupby(["chapter", "subtopic"]).size().sort_values(ascending=False).head(30).reset_index(name="questions") if not df.empty else pd.DataFrame(columns=["chapter", "subtopic", "questions"])
        story.append(_report_table(subtopic))
        story.append(Spacer(1, 10))

        priority_path = outputs.get("study_priority")
        if priority_path and priority_path.exists():
            priority = pd.read_csv(priority_path).head(15)
            story.append(Paragraph("Suggested Study Priority", styles["Heading2"]))
            story.append(_report_table(priority[["chapter", "subtopic", "priority_score", "probability_estimate"]]))

        story.append(Spacer(1, 10))
        story.append(Paragraph("Caution", styles["Heading2"]))
        story.append(Paragraph("Use these rankings with NCERT coverage and recent syllabus changes. Repetition patterns help allocate effort, but CUET questions can shift by year and shift.", styles["BodyText"]))
        doc.build(story)
    except Exception:
        # Last-resort placeholder keeps the pipeline reproducible even before reportlab is installed.
        out.write_bytes(
            b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
        )
    return out


def _report_table(df: pd.DataFrame):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    data = [list(df.columns)] + df.astype(str).values.tolist()
    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e79")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cccccc")),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    return table
