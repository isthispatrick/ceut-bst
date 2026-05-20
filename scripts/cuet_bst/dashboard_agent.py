from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .llm_client import chat_completion
from .settings import PROCESSED_DIR


APPROVED_FILES = {
    "study_priority.csv": PROCESSED_DIR / "study_priority.csv",
    "repeated_concepts.csv": PROCESSED_DIR / "repeated_concepts.csv",
    "question_format_strategy.csv": PROCESSED_DIR / "question_format_strategy.csv",
    "suspicious_classifications.csv": PROCESSED_DIR / "suspicious_classifications.csv",
    "accuracy_summary.csv": PROCESSED_DIR / "accuracy_summary.csv",
    "topic_frequency.csv": PROCESSED_DIR / "topic_frequency.csv",
    "ai_manual_review_suggestions.csv": PROCESSED_DIR / "ai_manual_review_suggestions.csv",
}


@dataclass
class ChartSpec:
    chart_type: str
    title: str
    data: pd.DataFrame
    x: str = ""
    y: str = ""
    color: str = ""


@dataclass
class AnalysisEvidence:
    files_used: list[str]
    top_rows_used: dict[str, list[dict[str, Any]]]
    numbers_used: dict[str, Any]
    confidence_level: str
    caveats: list[str]
    context_text: str
    chart: ChartSpec | None = None


def load_analysis_tables() -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for name, path in APPROVED_FILES.items():
        if path.exists():
            tables[name] = pd.read_csv(path).fillna("")
        else:
            tables[name] = pd.DataFrame()
    return tables


def build_evidence(question: str, tables: dict[str, pd.DataFrame]) -> AnalysisEvidence:
    q = question.lower()
    files_used: list[str] = []
    top_rows: dict[str, list[dict[str, Any]]] = {}
    numbers: dict[str, Any] = {}
    caveats = ["All results are historical PYQ analysis, not prediction certainty.", "Do not treat AI/silver labels as human ground truth."]

    priority = tables.get("study_priority.csv", pd.DataFrame())
    strategy = tables.get("question_format_strategy.csv", pd.DataFrame())
    suspicious = tables.get("suspicious_classifications.csv", pd.DataFrame())
    accuracy = tables.get("accuracy_summary.csv", pd.DataFrame())
    topic_frequency = tables.get("topic_frequency.csv", pd.DataFrame())
    review = tables.get("ai_manual_review_suggestions.csv", pd.DataFrame())

    if not priority.empty:
        files_used.append("study_priority.csv")
        top_priority = priority.head(12)
        top_rows["study_priority.csv"] = _records(top_priority, ["chapter", "subtopic", "priority_tier", "source_weighted_frequency", "raw_score", "roi", "top_pattern", "top_micro_concept"])
        numbers["tier_1_topics"] = int(priority.get("priority_tier", pd.Series(dtype=str)).astype(str).eq("Tier 1 = Must do").sum())
        numbers["tier_2_topics"] = int(priority.get("priority_tier", pd.Series(dtype=str)).astype(str).eq("Tier 2 = High ROI").sum())

    if not strategy.empty and any(term in q for term in ["how", "study", "pattern", "trap", "format", "ncert"]):
        files_used.append("question_format_strategy.csv")
        top_rows["question_format_strategy.csv"] = _records(strategy.head(10), ["chapter", "subtopic", "dominant_question_pattern", "how_to_study_it", "common_traps", "ncert_heading_to_revise"])

    if not suspicious.empty and any(term in q for term in ["suspicious", "review", "uncertain", "mismatch", "weak"]):
        files_used.append("suspicious_classifications.csv")
        top_rows["suspicious_classifications.csv"] = _records(suspicious.head(10), ["chapter", "subtopic", "micro_concept", "taxonomy_mismatch_reason"])
        numbers["suspicious_classifications"] = int(len(suspicious))

    if not accuracy.empty and any(term in q for term in ["accuracy", "benchmark", "silver", "gold", "confusion"]):
        files_used.append("accuracy_summary.csv")
        top_rows["accuracy_summary.csv"] = accuracy.to_dict("records")

    if not topic_frequency.empty and any(term in q for term in ["frequency", "chapter", "topic"]):
        files_used.append("topic_frequency.csv")
        top_rows["topic_frequency.csv"] = _records(topic_frequency.head(12), list(topic_frequency.columns[:8]))

    if not review.empty and any(term in q for term in ["manual", "review", "uncertain", "weak"]):
        files_used.append("ai_manual_review_suggestions.csv")
        numbers["manual_review_suggestions"] = int(len(review))

    chart = infer_chart(question, tables)
    if chart:
        for name in _chart_files(chart, tables):
            if name not in files_used:
                files_used.append(name)
        top_rows[f"chart:{chart.title}"] = chart.data.head(10).to_dict("records")

    if not files_used:
        files_used = ["study_priority.csv", "accuracy_summary.csv"]

    official_rows = _metric_from_quality()
    if official_rows == 0:
        caveats.append("Official NTA question rows are still 0, so source reliability is limited by third-party PYQ data.")
    numbers["official_source_rows"] = official_rows

    confidence = "medium"
    if official_rows == 0 or (not suspicious.empty and len(suspicious) > 0):
        confidence = "medium-low for fine-grained micro-concepts; higher for chapter/subtopic trends"
    if not accuracy.empty:
        numbers.update(_accuracy_numbers(accuracy))

    context = _context_text(files_used, top_rows, numbers, caveats)
    return AnalysisEvidence(sorted(set(files_used)), top_rows, numbers, confidence, caveats, context, chart)


def answer_question(question: str, evidence: AnalysisEvidence) -> str:
    system = (
        "You are a read-only CUET UG Business Studies data-analysis assistant. "
        "Use only the provided evidence. Do not invent data. Do not claim prediction certainty. "
        "Never mention or expose API keys, environment variables, or .env content. "
        "Always include sections: Answer, Evidence, Confidence, Caveats."
    )
    content = chat_completion(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"EVIDENCE:\n{evidence.context_text}\n\nQUESTION:\n{question}"},
        ],
        max_tokens=900,
        timeout=60,
    )
    return content


def infer_chart(question: str, tables: dict[str, pd.DataFrame]) -> ChartSpec | None:
    q = question.lower()
    priority = tables.get("study_priority.csv", pd.DataFrame())
    suspicious = tables.get("suspicious_classifications.csv", pd.DataFrame())
    topic_frequency = tables.get("topic_frequency.csv", pd.DataFrame())

    if any(term in q for term in ["show top", "top 10", "top topics", "weighted score"]) and not priority.empty:
        score_col = "source_weighted_frequency" if "source_weighted_frequency" in priority.columns else "study_priority_score"
        data = priority.sort_values(score_col, ascending=False).head(10).copy()
        data["topic"] = data["chapter"].astype(str) + " -> " + data["subtopic"].astype(str)
        return ChartSpec("bar", "Top 10 Topics By Weighted Score", data, x=score_col, y="topic", color="chapter")

    if "compare" in q and "marketing" in q and "business finance" in q and not priority.empty:
        data = priority[priority["chapter"].astype(str).isin(["Marketing", "Business Finance"])].copy()
        grouped = data.groupby("chapter", dropna=False).agg(
            weighted_score=("source_weighted_frequency", "sum"),
            avg_roi=("roi", "mean"),
            topics=("subtopic", "count"),
        ).reset_index()
        return ChartSpec("bar", "Marketing vs Business Finance", grouped, x="weighted_score", y="chapter", color="chapter")

    if "chapter frequency" in q:
        if not topic_frequency.empty and "chapter" in topic_frequency.columns:
            count_col = "question_count" if "question_count" in topic_frequency.columns else topic_frequency.columns[-1]
            data = topic_frequency.groupby("chapter", dropna=False)[count_col].sum().reset_index(name="question_count")
        elif not priority.empty:
            data = priority.groupby("chapter", dropna=False)["question_count"].sum().reset_index(name="question_count")
        else:
            return None
        return ChartSpec("bar", "Chapter Frequency", data.sort_values("question_count", ascending=False), x="question_count", y="chapter", color="chapter")

    if "suspicious" in q and "chapter" in q and not suspicious.empty:
        data = suspicious["chapter"].replace("", "Unknown").value_counts().rename_axis("chapter").reset_index(name="suspicious_count")
        return ChartSpec("bar", "Suspicious Classifications By Chapter", data, x="suspicious_count", y="chapter")

    if "pie" in q and "tier" in q and not priority.empty:
        data = priority["priority_tier"].replace("", "Unknown").value_counts().rename_axis("priority_tier").reset_index(name="topics")
        return ChartSpec("pie", "Priority Tier Distribution", data, x="priority_tier", y="topics")

    if any(term in q for term in ["line", "trend"]) and not topic_frequency.empty and "years_seen" in topic_frequency.columns:
        rows = []
        for _, row in topic_frequency.iterrows():
            for year in str(row.get("years_seen", "")).replace(";", ",").split(","):
                year = year.strip()
                if year:
                    rows.append({"year": year, "question_count": float(row.get("question_count", 1) or 1)})
        data = pd.DataFrame(rows)
        if not data.empty:
            data = data.groupby("year", dropna=False)["question_count"].sum().reset_index()
            return ChartSpec("line", "Year Trend From Topics Seen", data, x="year", y="question_count")

    if "heatmap" in q and not topic_frequency.empty and {"chapter", "subtopic", "question_count"}.issubset(topic_frequency.columns):
        pivot = topic_frequency.pivot_table(index="chapter", columns="subtopic", values="question_count", aggfunc="sum", fill_value=0)
        return ChartSpec("heatmap", "Chapter vs Subtopic Heatmap", pivot)

    if "table" in q and not priority.empty:
        return ChartSpec("table", "Top Study Priority Table", priority.head(20))

    return None


def _chart_files(chart: ChartSpec, tables: dict[str, pd.DataFrame]) -> list[str]:
    if "Suspicious" in chart.title:
        return ["suspicious_classifications.csv"]
    if "Frequency" in chart.title:
        return ["topic_frequency.csv", "study_priority.csv"]
    return ["study_priority.csv"]


def _records(df: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    cols = [col for col in columns if col in df.columns]
    return df[cols].head(12).to_dict("records") if cols else df.head(12).to_dict("records")


def _metric_from_quality() -> int:
    path = PROCESSED_DIR / "data_quality_summary.csv"
    if not path.exists():
        return 0
    quality = pd.read_csv(path).fillna("")
    matches = quality[quality["metric"].astype(str).eq("official_source_rows")]
    if matches.empty:
        return 0
    try:
        return int(float(matches.iloc[0]["value"]))
    except Exception:
        return 0


def _accuracy_numbers(accuracy: pd.DataFrame) -> dict[str, Any]:
    result = {}
    for _, row in accuracy.iterrows():
        metric = str(row.get("metric", ""))
        value = row.get("value", "")
        if "accuracy" in metric or "status" in metric:
            result[metric] = value
    return result


def _context_text(files_used: list[str], top_rows: dict[str, list[dict[str, Any]]], numbers: dict[str, Any], caveats: list[str]) -> str:
    return (
        f"FILES USED: {', '.join(sorted(set(files_used)))}\n"
        f"NUMBERS USED: {numbers}\n"
        f"TOP ROWS USED: {top_rows}\n"
        f"CAVEATS: {caveats}\n"
    )
