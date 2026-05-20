from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .settings import CONFIG_DIR, PROCESSED_DIR, REPORTS_DIR, VERIFIED_DIR, ensure_dirs, load_json


GOLDEN_LABELS_PATH = VERIFIED_DIR / "golden_labels.csv"
SILVER_LABELS_PATH = VERIFIED_DIR / "silver_labels_ai.csv"
LEGACY_AI_GOLDEN_PATH = VERIFIED_DIR / "golden_labels_ai.csv"


def run_validation_outputs(df: pd.DataFrame) -> pd.DataFrame:
    ensure_dirs()
    df = apply_taxonomy_validation(df)
    write_human_gold_review_queue(df)
    write_accuracy_evaluation(df)
    write_ai_review_suggestions(df)
    write_question_format_strategy(df)
    write_final_study_pack(df)
    return df


def apply_taxonomy_validation(df: pd.DataFrame) -> pd.DataFrame:
    config_path = CONFIG_DIR / "chapter_allowed_concepts.json"
    allowed = load_json(config_path) if config_path.exists() else {}
    out = df.copy()
    if "micro_concept_confidence" not in out.columns:
        out["micro_concept_confidence"] = ""
    if "micro_concept_note" not in out.columns:
        out["micro_concept_note"] = ""
    mismatch_flags: list[bool] = []
    reasons: list[str] = []
    for _, row in out.iterrows():
        chapter = str(row.get("chapter", "")).strip()
        subtopic = str(row.get("subtopic", "")).strip()
        micro = str(row.get("micro_concept", "")).strip()
        chapter_allowed = allowed.get(chapter)
        row_reasons: list[str] = []
        if not chapter_allowed:
            row_reasons.append("unknown chapter")
        else:
            valid_subtopics = {str(value).strip().lower() for value in chapter_allowed.get("allowed_subtopics", [])}
            valid_micro = {str(value).strip().lower() for value in chapter_allowed.get("allowed_micro_concepts", [])}
            if subtopic and subtopic.lower() not in valid_subtopics:
                row_reasons.append(f"subtopic not allowed for chapter: {subtopic}")
            if micro and not _micro_allowed(micro.lower(), valid_micro, valid_subtopics):
                row_reasons.append(f"micro_concept not allowed for chapter: {micro}")
        mismatch_flags.append(bool(row_reasons))
        reasons.append("; ".join(row_reasons))
    out["taxonomy_mismatch"] = ["true" if flag else "false" for flag in mismatch_flags]
    out["taxonomy_mismatch_reason"] = reasons
    out = _apply_micro_confidence(out)
    mismatch_mask = out["taxonomy_mismatch"].astype(str).str.lower().eq("true")
    if "needs_review" not in out.columns:
        out["needs_review"] = "false"
    if "review_reason" not in out.columns:
        out["review_reason"] = ""
    out["needs_review"] = out["needs_review"].astype(str)
    out["review_reason"] = out["review_reason"].astype(str)
    out.loc[mismatch_mask, "needs_review"] = "true"
    out.loc[mismatch_mask, "review_reason"] = out.loc[mismatch_mask].apply(_merge_review_reason, axis=1)
    low_micro_mask = pd.to_numeric(out["micro_concept_confidence"], errors="coerce").fillna(1) < 0.70
    out.loc[low_micro_mask, "needs_review"] = "true"
    out.loc[low_micro_mask & out["review_reason"].astype(str).eq(""), "review_reason"] = "micro-concept confidence below 0.70"
    suspicious_cols = [
        "canonical_question_id",
        "chapter",
        "subtopic",
        "micro_concept",
        "question_pattern",
        "final_confidence",
        "taxonomy_mismatch_reason",
        "question_text",
        "source_url",
    ]
    out.loc[mismatch_mask, [col for col in suspicious_cols if col in out.columns]].to_csv(
        PROCESSED_DIR / "suspicious_classifications.csv", index=False
    )
    return out


def write_human_gold_review_queue(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "human_gold_review_queue.csv"
    if df.empty:
        pd.DataFrame().to_csv(out, index=False)
        return out
    temp = df.copy()
    temp["weighted_frequency_score"] = pd.to_numeric(temp.get("weighted_frequency_score", 0), errors="coerce").fillna(0)
    temp["recency_weighted_score"] = pd.to_numeric(temp.get("recency_weighted_score", 0), errors="coerce").fillna(0)
    temp["final_confidence"] = pd.to_numeric(temp.get("final_confidence", 0), errors="coerce").fillna(0)
    temp["micro_concept_confidence"] = pd.to_numeric(temp.get("micro_concept_confidence", 0), errors="coerce").fillna(0)
    temp["importance_for_review"] = (
        temp["weighted_frequency_score"] * 0.55 + temp["recency_weighted_score"] * 0.35 + (1 - temp["final_confidence"]) * 0.10
    )
    cols = [
        "canonical_question_id",
        "importance_for_review",
        "year",
        "source",
        "chapter",
        "subtopic",
        "micro_concept",
        "question_type",
        "question_pattern",
        "difficulty_estimate",
        "correct_option",
        "final_confidence",
        "micro_concept_confidence",
        "micro_concept_note",
        "taxonomy_mismatch",
        "question_text",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "source_url",
    ]
    selected: list[pd.DataFrame] = []
    selected.append(temp.sort_values("importance_for_review", ascending=False).head(50))
    suspicious = temp[temp.get("taxonomy_mismatch", "").astype(str).str.lower().eq("true")]
    selected.append(suspicious.sort_values("importance_for_review", ascending=False).head(25))
    priority = _read_csv("study_priority.csv")
    if not priority.empty and {"chapter", "subtopic"}.issubset(priority.columns):
        high = priority[priority.get("priority_tier", "").astype(str).isin(["Tier 1 = Must do", "Tier 2 = High ROI"])]
        keys = {(str(row["chapter"]), str(row["subtopic"])) for _, row in high.iterrows()}
        micro_rows = temp[temp.apply(lambda row: (str(row.get("chapter", "")), str(row.get("subtopic", ""))) in keys, axis=1)]
        selected.append(micro_rows.sort_values(["weighted_frequency_score", "micro_concept_confidence"], ascending=[False, True]).head(25))
    queue = pd.concat(selected, ignore_index=True).drop_duplicates("canonical_question_id", keep="first")
    if len(queue) < 100:
        existing_ids = set(queue["canonical_question_id"].astype(str))
        filler = temp[~temp["canonical_question_id"].astype(str).isin(existing_ids)].sort_values("importance_for_review", ascending=False)
        queue = pd.concat([queue, filler.head(100 - len(queue))], ignore_index=True)
    queue = queue.drop_duplicates("canonical_question_id", keep="first").head(100)
    queue[[c for c in cols if c in queue.columns]].to_csv(out, index=False)
    # Compatibility name for older dashboard links.
    queue[[c for c in cols if c in queue.columns]].to_csv(PROCESSED_DIR / "golden_review_queue.csv", index=False)
    return out


def write_accuracy_evaluation(df: pd.DataFrame) -> dict[str, Path]:
    summary_path = PROCESSED_DIR / "accuracy_summary.csv"
    silver = _read_labels(SILVER_LABELS_PATH, fallback=LEGACY_AI_GOLDEN_PATH)
    human = _read_human_gold_labels()
    rows: list[dict[str, Any]] = []
    rows.extend(_accuracy_rows(df, silver, "pipeline_vs_silver"))
    if human.empty:
        rows.append({"metric": "pipeline_vs_human_gold_status", "value": "Human-verified accuracy unavailable yet."})
    else:
        rows.extend(_accuracy_rows(df, human, "pipeline_vs_human_gold"))
    pd.DataFrame(rows).to_csv(summary_path, index=False)

    outputs = {"accuracy_summary": summary_path}
    for name, labels in [("silver", silver), ("human_gold", human)]:
        confusion_path = PROCESSED_DIR / f"{name}_chapter_confusion_matrix.csv"
        common_path = PROCESSED_DIR / f"{name}_classification_confusions.csv"
        _write_confusion_outputs(df, labels, confusion_path, common_path)
        outputs[f"{name}_chapter_confusion_matrix"] = confusion_path
        outputs[f"{name}_classification_confusions"] = common_path
    # Compatibility copies for existing dashboard components.
    _copy_if_exists(PROCESSED_DIR / "silver_chapter_confusion_matrix.csv", PROCESSED_DIR / "chapter_confusion_matrix.csv")
    _copy_if_exists(PROCESSED_DIR / "silver_classification_confusions.csv", PROCESSED_DIR / "classification_confusions.csv")
    return outputs


def _accuracy_rows(df: pd.DataFrame, labels: pd.DataFrame, prefix: str) -> list[dict[str, Any]]:
    if labels.empty:
        return [{"metric": f"{prefix}_rows", "value": 0}]
    if "canonical_question_id" not in labels.columns:
        return [{"metric": f"{prefix}_status", "value": "labels missing canonical_question_id"}]
    merged = df.merge(labels, on="canonical_question_id", how="inner", suffixes=("_pred", "_gold"))
    fields = {
        "chapter": ("chapter_pred", "chapter_gold"),
        "subtopic": ("subtopic_pred", "subtopic_gold"),
        "micro_concept": ("micro_concept_pred", "micro_concept_gold"),
    }
    fields["question_type"] = ("question_type_pred", "question_type_gold") if "question_type_pred" in merged.columns else ("question_pattern_pred", "question_type_gold")
    rows = [{"metric": f"{prefix}_rows", "value": len(labels)}, {"metric": f"{prefix}_matched_rows", "value": len(merged)}]
    for label, (pred, gold) in fields.items():
        if pred not in merged.columns or gold not in merged.columns:
            rows.append({"metric": f"{prefix}_{label}_accuracy", "value": ""})
            continue
        rows.append({"metric": f"{prefix}_{label}_accuracy", "value": round(_accuracy(merged[pred], merged[gold]), 4)})
    return rows


def _write_confusion_outputs(df: pd.DataFrame, labels: pd.DataFrame, confusion_path: Path, common_path: Path) -> None:
    if labels.empty or "canonical_question_id" not in labels.columns:
        pd.DataFrame().to_csv(confusion_path, index=False)
        pd.DataFrame().to_csv(common_path, index=False)
        return
    merged = df.merge(labels, on="canonical_question_id", how="inner", suffixes=("_pred", "_gold"))
    if not merged.empty and "chapter_pred" in merged.columns and "chapter_gold" in merged.columns:
        pd.crosstab(merged["chapter_gold"], merged["chapter_pred"]).to_csv(confusion_path)
        _common_confusions(merged).to_csv(common_path, index=False)
    else:
        pd.DataFrame().to_csv(confusion_path, index=False)
        pd.DataFrame().to_csv(common_path, index=False)


def _read_labels(path: Path, fallback: Path | None = None) -> pd.DataFrame:
    source = path if path.exists() else fallback
    if source is None or not source.exists():
        return pd.DataFrame()
    labels = pd.read_csv(source).fillna("")
    if "verification_source" in labels.columns:
        labels["verification_source"] = labels["verification_source"].astype(str).replace({"ai_verified": "ai_silver"})
    return labels


def _read_human_gold_labels() -> pd.DataFrame:
    if not GOLDEN_LABELS_PATH.exists():
        return pd.DataFrame()
    labels = pd.read_csv(GOLDEN_LABELS_PATH).fillna("")
    if "verification_source" not in labels.columns:
        return labels
    mask = ~labels["verification_source"].astype(str).str.lower().str.contains("ai_")
    return labels[mask].copy()


def _copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.write_bytes(source.read_bytes())


def _apply_micro_confidence(out: pd.DataFrame) -> pd.DataFrame:
    confidence = pd.to_numeric(out["micro_concept_confidence"], errors="coerce")
    fallback_conf = out["taxonomy_mismatch"].astype(str).str.lower().eq("true").map({True: 0.45, False: 0.82})
    out["micro_concept_confidence"] = confidence.fillna(fallback_conf).clip(lower=0, upper=1).round(2)
    low_mask = out["micro_concept_confidence"] < 0.70
    out.loc[low_mask, "micro_concept_note"] = "Micro-concept uncertain; trust chapter/subtopic more."
    out.loc[low_mask & out["subtopic"].astype(str).ne(""), "micro_concept"] = out.loc[low_mask & out["subtopic"].astype(str).ne(""), "subtopic"]
    return out


def write_ai_review_suggestions(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "ai_manual_review_suggestions.csv"
    if df.empty or "needs_review" not in df.columns:
        pd.DataFrame().to_csv(out, index=False)
        return out
    review = df[df["needs_review"].astype(str).str.lower().isin(["true", "yes"])].copy()
    columns = {
        "canonical_question_id": review.get("canonical_question_id", pd.Series(dtype=str)),
        "ai_suggested_chapter": review.get("chapter", pd.Series(dtype=str)),
        "ai_suggested_subtopic": review.get("subtopic", pd.Series(dtype=str)),
        "ai_suggested_question_pattern": review.get("question_pattern", pd.Series(dtype=str)),
        "ai_suggested_difficulty": review.get("difficulty_estimate", pd.Series(dtype=str)),
        "ai_suggested_ncert_heading": review.get("ncert_heading", pd.Series(dtype=str)),
        "ai_suggested_micro_concept": review.get("micro_concept", pd.Series(dtype=str)),
        "micro_concept_confidence": review.get("micro_concept_confidence", pd.Series(dtype=str)),
        "micro_concept_note": review.get("micro_concept_note", pd.Series(dtype=str)),
        "llm_label": review.get("llm_label", pd.Series(dtype=str)),
        "rule_label": review.get("rule_label", pd.Series(dtype=str)),
        "embedding_label": review.get("embedding_label", pd.Series(dtype=str)),
        "bm25_label": review.get("bm25_label", pd.Series(dtype=str)),
        "final_confidence": review.get("final_confidence", pd.Series(dtype=str)),
        "review_reason": review.get("review_reason", pd.Series(dtype=str)),
        "taxonomy_mismatch_reason": review.get("taxonomy_mismatch_reason", pd.Series(dtype=str)),
        "question_text": review.get("question_text", pd.Series(dtype=str)),
    }
    pd.DataFrame(columns).to_csv(out, index=False)
    return out


def write_question_format_strategy(df: pd.DataFrame) -> Path:
    out = PROCESSED_DIR / "question_format_strategy.csv"
    priority_path = PROCESSED_DIR / "study_priority.csv"
    if df.empty or not priority_path.exists():
        pd.DataFrame().to_csv(out, index=False)
        return out
    priority = pd.read_csv(priority_path).fillna("")
    high = priority[priority.get("priority_tier", "").astype(str).isin(["Tier 1 = Must do", "Tier 2 = High ROI"])]
    if high.empty:
        high = priority.head(25)
    rows: list[dict[str, Any]] = []
    for _, topic in high.iterrows():
        chapter = str(topic.get("chapter", ""))
        subtopic = str(topic.get("subtopic", ""))
        group = df[(df["chapter"].astype(str) == chapter) & (df["subtopic"].astype(str) == subtopic)].copy()
        if group.empty:
            continue
        pattern = _dominant(group.get("question_pattern", pd.Series(dtype=str)))
        heading = _dominant(group.get("ncert_heading", pd.Series(dtype=str)))
        reps = _representative_questions(group, 5)
        rows.append(
            {
                "chapter": chapter,
                "subtopic": subtopic,
                "priority_tier": topic.get("priority_tier", ""),
                "dominant_question_pattern": pattern,
                "how_to_study_it": _study_strategy(pattern, subtopic),
                "representative_pyqs": reps,
                "common_traps": _common_traps(pattern, subtopic, chapter),
                "ncert_heading_to_revise": heading,
            }
        )
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_final_study_pack(df: pd.DataFrame) -> Path:
    out = REPORTS_DIR / "cuet_bst_final_study_pack.pdf"
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

        styles = getSampleStyleSheet()
        priority = _read_csv("study_priority.csv")
        clusters = _read_csv("micro_concept_clusters.csv")
        strategy = _read_csv("question_format_strategy.csv")
        plans = _read_csv("study_plans.csv")
        low = _read_csv("low_frequency_topics.csv")
        patterns = _chapter_pattern_table(df)
        story = [
            Paragraph("CUET BST Final Study Pack", styles["Title"]),
            Paragraph("Use this as an exam-decision pack: what to study first, how questions appear, and what to skip when time is low.", styles["BodyText"]),
            Spacer(1, 10),
            Paragraph("Tier 1 Topics: Must Do", styles["Heading2"]),
            _report_table(priority[priority.get("priority_tier", pd.Series(dtype=str)).astype(str).eq("Tier 1 = Must do")].head(25)),
            Spacer(1, 8),
            Paragraph("Tier 2 Topics: High ROI", styles["Heading2"]),
            _report_table(priority[priority.get("priority_tier", pd.Series(dtype=str)).astype(str).eq("Tier 2 = High ROI")].head(25)),
            Spacer(1, 8),
            Paragraph("Most Repeated Micro-Concepts", styles["Heading2"]),
            _report_table(clusters.head(25)),
            Spacer(1, 8),
            Paragraph("Chapter-Wise Question Formats", styles["Heading2"]),
            _report_table(patterns.head(30)),
            PageBreak(),
            Paragraph("Question Format Strategy", styles["Heading2"]),
            _report_table(strategy.head(25)),
            Spacer(1, 8),
            Paragraph("5-Day Study Plan", styles["Heading2"]),
            _report_table(plans[plans.get("plan_days", pd.Series(dtype=str)).astype(str).eq("5")].head(30) if not plans.empty else plans),
            Spacer(1, 8),
            Paragraph("3-Day Emergency Plan", styles["Heading2"]),
            _report_table(plans[plans.get("plan_days", pd.Series(dtype=str)).astype(str).eq("3")].head(24) if not plans.empty else plans),
            Spacer(1, 8),
            Paragraph("Topics To Skip Or Only Skim If Time Is Low", styles["Heading2"]),
            _report_table(low.head(25)),
            Spacer(1, 8),
            Paragraph("Caution", styles["Heading2"]),
            Paragraph(
                "This pack is based on historical public PYQ data and model-assisted classification. It improves study prioritisation, but it cannot guarantee future CUET paper composition.",
                styles["BodyText"],
            ),
        ]
        doc = SimpleDocTemplate(str(out), pagesize=A4, title="CUET BST Final Study Pack")
        doc.build(story)
    except Exception:
        out.write_bytes(b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n")
    return out


def _micro_allowed(micro: str, valid_micro: set[str], valid_subtopics: set[str]) -> bool:
    if micro in valid_micro or micro in valid_subtopics:
        return True
    if not valid_micro:
        return True
    return any(micro in allowed or allowed in micro for allowed in valid_micro)


def _merge_review_reason(row: pd.Series) -> str:
    existing = str(row.get("review_reason", "")).strip()
    mismatch = str(row.get("taxonomy_mismatch_reason", "")).strip()
    if existing and mismatch:
        return f"{existing}; taxonomy mismatch: {mismatch}"
    return f"taxonomy mismatch: {mismatch}" if mismatch else existing


def _accuracy(predicted: pd.Series, golden: pd.Series) -> float:
    pred = predicted.astype(str).str.strip().str.lower()
    gold = golden.astype(str).str.strip().str.lower()
    valid = gold.ne("")
    if not valid.any():
        return 0.0
    return float(pred[valid].eq(gold[valid]).mean())


def _common_confusions(merged: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    comparisons = [
        ("chapter", "chapter_pred", "chapter_gold"),
        ("subtopic", "subtopic_pred", "subtopic_gold"),
    ]
    for field, pred, gold in comparisons:
        if pred not in merged.columns or gold not in merged.columns:
            continue
        wrong = merged[merged[pred].astype(str).str.lower() != merged[gold].astype(str).str.lower()]
        if wrong.empty:
            continue
        grouped = wrong.groupby([gold, pred], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
        for _, row in grouped.head(25).iterrows():
            rows.append({"field": field, "gold_label": row[gold], "predicted_label": row[pred], "count": int(row["count"])})
    return pd.DataFrame(rows)


def _representative_questions(group: pd.DataFrame, limit: int) -> str:
    temp = group.copy()
    temp["weighted_frequency_score"] = pd.to_numeric(temp.get("weighted_frequency_score", 0), errors="coerce").fillna(0)
    questions = []
    for _, row in temp.sort_values(["weighted_frequency_score", "final_confidence"], ascending=False).head(limit).iterrows():
        text = str(row.get("question_text", "")).replace("\n", " ").strip()
        questions.append(text[:220])
    return " || ".join(questions)


def _study_strategy(pattern: str, subtopic: str) -> str:
    lower = f"{pattern} {subtopic}".lower()
    if "case" in lower:
        return "Practise diagnosing the concept from short business situations; make example cues for each NCERT term."
    if "chronology" in lower or "process" in lower:
        return "Memorise the NCERT sequence exactly, then practise shuffled-order PYQs."
    if "match" in lower:
        return "Create two-column tables for terms, features, examples, and legal bodies."
    if "principle" in lower:
        return "Learn the exact principle names, keywords, and one business example for each."
    if "feature" in lower:
        return "Revise feature lists and practise eliminating options that are true but from another chapter."
    return "Revise NCERT definitions, headings, and common examples; then solve direct PYQs."


def _common_traps(pattern: str, subtopic: str, chapter: str) -> str:
    lower = f"{pattern} {subtopic} {chapter}".lower()
    traps = []
    if "marketing" in lower:
        traps.append("confusing product, promotion, price, and place cues")
    if "finance" in lower or "capital" in lower:
        traps.append("mixing financing decision with investment or dividend decision")
    if "fayol" in lower or "principle" in lower:
        traps.append("mixing unity of command with unity of direction")
    if "planning" in lower:
        traps.append("mixing policy, procedure, method, rule, programme, and budget")
    if "staffing" in lower:
        traps.append("mixing recruitment, selection, training, and development stages")
    if "consumer" in lower:
        traps.append("mixing consumer rights with responsibilities and redressal bodies")
    if "case" in lower:
        traps.append("over-reading story details instead of identifying NCERT cue words")
    return "; ".join(traps) if traps else "watch for options from nearby chapters that sound correct but do not match the NCERT heading"


def _chapter_pattern_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "chapter" not in df.columns:
        return pd.DataFrame()
    table = (
        df.groupby(["chapter", "question_pattern"], dropna=False)
        .size()
        .reset_index(name="question_count")
        .sort_values(["chapter", "question_count"], ascending=[True, False])
    )
    return table


def _dominant(series: pd.Series) -> str:
    values = series.astype(str).replace("", pd.NA).dropna()
    return str(values.mode().iloc[0]) if not values.empty else ""


def _read_csv(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def _report_table(df: pd.DataFrame):
    from reportlab.lib import colors
    from reportlab.platypus import Table, TableStyle

    if df is None or df.empty:
        df = pd.DataFrame([{"status": "No data yet"}])
    compact = df.copy()
    keep = [col for col in compact.columns if col not in {"question_text", "source_url", "duplicate_source_urls", "example_question"}]
    compact = compact[keep[:8]]
    data = [list(compact.columns)] + compact.astype(str).head(30).values.tolist()
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
