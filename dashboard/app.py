from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
PROCESSED = ROOT / "data" / "processed"
DATA_DIR = ROOT / "data"
REPORT = ROOT / "reports" / "cuet_bst_analysis_report.pdf"
STUDY_PACK = ROOT / "reports" / "cuet_bst_final_study_pack.pdf"
MANUAL_LABELS = DATA_DIR / "manual_labels.csv"
VERIFIED_DIR = DATA_DIR / "verified"
GOLDEN_LABELS = VERIFIED_DIR / "golden_labels.csv"
SILVER_LABELS = VERIFIED_DIR / "silver_labels_ai.csv"


st.set_page_config(page_title="CUET BST PYQ Analysis", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stToolbar"] { visibility: hidden; height: 0; position: fixed; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stStatusWidget"] { visibility: hidden; height: 0; position: fixed; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; height: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("CUET UG Business Studies PYQ Analysis")


@st.cache_data(show_spinner=False)
def load_questions() -> pd.DataFrame:
    advanced = PROCESSED / "questions_advanced.csv"
    path = advanced if advanced.exists() else PROCESSED / "questions.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def apply_filters(data: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")
    result = data.copy()
    for column in ["year", "unit", "chapter", "subtopic", "question_pattern", "difficulty_estimate", "source_tier", "source"]:
        if column not in result.columns:
            continue
        values = sorted([str(value) for value in result[column].unique() if str(value).strip()])
        selected = st.sidebar.multiselect(column.replace("_", " ").title(), values)
        if selected:
            result = result[result[column].astype(str).isin(selected)]
    query = st.sidebar.text_input("Search questions")
    if query:
        mask = result.astype(str).agg(" ".join, axis=1).str.contains(query, case=False, regex=False)
        result = result[mask]
    export_controls(result)
    return result


def export_controls(data: pd.DataFrame) -> None:
    st.sidebar.download_button("CSV", data.to_csv(index=False), file_name="cuet_bst_questions_filtered.csv")
    st.sidebar.download_button("JSON", data.to_json(orient="records", indent=2), file_name="cuet_bst_questions_filtered.json")
    try:
        import io

        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            data.to_excel(writer, index=False, sheet_name="questions")
        st.sidebar.download_button("Excel", buffer.getvalue(), file_name="cuet_bst_questions_filtered.xlsx")
    except Exception:
        pass
    if REPORT.exists():
        st.sidebar.download_button("PDF report", REPORT.read_bytes(), file_name="cuet_bst_analysis_report.pdf")
    if STUDY_PACK.exists():
        st.sidebar.download_button("Final study pack", STUDY_PACK.read_bytes(), file_name="cuet_bst_final_study_pack.pdf")


def overview(data: pd.DataFrame) -> None:
    quality = load_csv("data_quality_summary.csv")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    raw_rows = metric_value(quality, "total_parsed_rows", len(data))
    unique_rows = metric_value(quality, "unique_questions_after_dedupe", len(data))
    duplicate_rate = metric_value(quality, "duplicate_rate_percent", 0)
    c1.metric("Raw rows", raw_rows)
    c2.metric("Unique questions", unique_rows)
    c3.metric("Duplicate rate", f"{duplicate_rate}%")
    c4.metric("Chapters", data["chapter"].replace("", pd.NA).dropna().nunique())
    c5.metric("Subtopics", data["subtopic"].replace("", pd.NA).dropna().nunique())
    c6.metric("Needs review", int(data.get("needs_review", "").astype(str).str.lower().isin(["yes", "true"]).sum()))
    st.plotly_chart(weighted_bar(data, "chapter", "Chapter Weighted Frequency"), use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pie_count(data, "question_pattern" if "question_pattern" in data.columns else "question_type", "Question Pattern Distribution"), use_container_width=True)
    with col2:
        st.plotly_chart(heatmap(data, "chapter", "year", "Chapter vs Year"), use_container_width=True)
    st.plotly_chart(topic_network(), use_container_width=True)


def chapter_frequency(data: pd.DataFrame) -> None:
    st.plotly_chart(weighted_bar(data, "chapter", "Chapter Weighted Frequency"), use_container_width=True)
    st.dataframe(weighted_table(data, "chapter"), use_container_width=True)


def subtopic_frequency(data: pd.DataFrame) -> None:
    table = weighted_table(data, "subtopic").head(30)
    st.plotly_chart(px.bar(table, x="weighted_score", y="subtopic", orientation="h", title="Top Repeated Subtopics"), use_container_width=True)
    st.dataframe(table, use_container_width=True)


def year_trend(data: pd.DataFrame) -> None:
    trend = data[data["year"].astype(str) != ""].groupby(["year", "chapter"]).size().reset_index(name="questions")
    if trend.empty:
        st.info("No year metadata available yet.")
        return
    st.plotly_chart(px.line(trend, x="year", y="questions", color="chapter", markers=True), use_container_width=True)
    st.plotly_chart(heatmap(data, "chapter", "year", "Chapter vs Year"), use_container_width=True)


def question_type_analysis(data: pd.DataFrame) -> None:
    column = "question_pattern" if "question_pattern" in data.columns else "question_type"
    st.plotly_chart(pie_count(data, column, "Question Pattern Distribution"), use_container_width=True)
    pivot = pd.crosstab(data["chapter"].replace("", "Unknown"), data[column].replace("", "Unknown"))
    st.dataframe(pivot, use_container_width=True)


def repeated_concepts() -> None:
    concepts = load_csv("repeated_concepts.csv")
    if concepts.empty:
        st.info("Run `python scripts/analyze.py` to generate repeated concept clusters.")
        return
    st.dataframe(concepts, use_container_width=True)


def reddit_vs_actual() -> None:
    reddit = load_csv("reddit_insights.csv")
    if reddit.empty:
        st.info("Run `python scripts/analyze.py` after importing Reddit summaries.")
        return
    st.plotly_chart(
        px.scatter(
            reddit,
            x="reddit_hype_score",
            y="actual_weighted_frequency" if "actual_weighted_frequency" in reddit.columns else "actual_pyq_frequency",
            hover_name="topic",
            color="mismatch_type" if "mismatch_type" in reddit.columns else "mismatch_validation",
            title="Reddit Hype vs Actual PYQ Frequency",
        ),
        use_container_width=True,
    )
    st.dataframe(reddit, use_container_width=True)


def question_explorer(data: pd.DataFrame) -> None:
    columns = [
        "year",
        "chapter",
        "subtopic",
        "question_type",
        "question_pattern",
        "micro_concept",
        "micro_concept_confidence",
        "micro_concept_note",
        "ncert_heading",
        "difficulty_estimate",
        "question_text",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
        "correct_option",
        "confidence_score",
        "final_confidence",
        "needs_review",
        "source_url",
    ]
    st.dataframe(data[[column for column in columns if column in data.columns]], use_container_width=True, height=650)


def study_priority() -> None:
    priority = load_csv("study_priority.csv")
    if priority.empty:
        st.info("Run `python scripts/analyze.py` to generate the study priority list.")
        return
    st.dataframe(priority, use_container_width=True)
    score_col = "study_priority_score" if "study_priority_score" in priority.columns else "priority_score"
    st.plotly_chart(px.bar(priority.head(25), x=score_col, y="subtopic", color="chapter", orientation="h"), use_container_width=True)


def data_quality_page() -> None:
    quality = load_csv("data_quality_summary.csv")
    if quality.empty:
        st.info("Run `python scripts/dedupe.py` to generate data quality metrics.")
        return
    st.dataframe(quality, use_container_width=True)
    source_counts = quality[quality["metric"].astype(str).str.startswith("source_tier_")].copy()
    if not source_counts.empty:
        source_counts["source_tier"] = source_counts["metric"].str.replace("source_tier_", "", regex=False)
        st.plotly_chart(px.bar(source_counts, x="source_tier", y="value", title="Rows by Source Tier"), use_container_width=True)


def source_reliability_page(data: pd.DataFrame) -> None:
    if "source_tier" not in data.columns:
        st.info("Run the advanced pipeline to calculate source tiers.")
        return
    table = (
        data.groupby(["source_tier", "source"], dropna=False)
        .agg(
            unique_questions=("canonical_question_id", "count"),
            weighted_score=("weighted_frequency_score", "sum"),
            avg_confidence=("final_confidence", lambda values: pd.to_numeric(values, errors="coerce").mean()),
        )
        .reset_index()
        .sort_values("weighted_score", ascending=False)
    )
    st.dataframe(table, use_container_width=True)
    st.plotly_chart(px.bar(table, x="source_tier", y="weighted_score", color="source", title="Source Reliability Weight"), use_container_width=True)


def dedupe_report_page() -> None:
    groups = load_csv("duplicate_groups.csv")
    if groups.empty:
        st.info("Run `python scripts/dedupe.py` to generate duplicate groups.")
        return
    st.metric("Duplicate groups", int((pd.to_numeric(groups["duplicate_group_size"], errors="coerce") > 1).sum()))
    st.dataframe(groups, use_container_width=True, height=650)


def ncert_reverse_index_page(data: pd.DataFrame) -> None:
    reverse = load_csv("ncert_reverse_index.csv")
    mapping = load_csv("question_ncert_map.csv")
    if reverse.empty:
        st.info("Run `python scripts/classify_ensemble.py` to generate the NCERT reverse index.")
        return
    st.subheader("NCERT / Syllabus Index")
    st.dataframe(reverse, use_container_width=True, height=280)
    if not mapping.empty:
        count = mapping.groupby(["chapter", "ncert_heading", "concept_type"], dropna=False).size().reset_index(name="pyq_count")
        st.subheader("PYQ-to-NCERT Reverse Index")
        st.dataframe(count.sort_values("pyq_count", ascending=False), use_container_width=True)


def pattern_analysis_page() -> None:
    table = load_csv("pattern_by_topic.csv")
    if table.empty:
        st.info("Run `python scripts/classify_ensemble.py` to generate question patterns.")
        return
    st.plotly_chart(px.bar(table.head(40), x="question_count", y="subtopic", color="question_pattern", orientation="h"), use_container_width=True)
    st.dataframe(table, use_container_width=True)


def micro_clusters_page() -> None:
    clusters = load_csv("micro_concept_clusters.csv")
    if clusters.empty:
        st.info("Run `python scripts/classify_ensemble.py` to generate micro-concept clusters.")
        return
    st.plotly_chart(px.bar(clusters.head(30), x="weighted_score", y="micro_concept", color="chapter", orientation="h"), use_container_width=True)
    st.dataframe(clusters, use_container_width=True, height=650)


def high_roi_plan_page() -> None:
    high_roi = load_csv("high_roi_topics.csv")
    plans = load_csv("study_plans.csv")
    if high_roi.empty and plans.empty:
        st.info("Run `python scripts/analyze_advanced.py` to generate high ROI topics and study plans.")
        return
    days = st.segmented_control("Plan length", [3, 5, 7, 14], default=5)
    if not plans.empty:
        st.dataframe(plans[plans["plan_days"].astype(str) == str(days)], use_container_width=True, height=360)
    if not high_roi.empty:
        st.plotly_chart(px.bar(high_roi.head(25), x="roi", y="subtopic", color="chapter", orientation="h", title="High ROI Topics"), use_container_width=True)
        st.dataframe(high_roi, use_container_width=True)


def golden_dataset_review_page() -> None:
    queue = load_csv("human_gold_review_queue.csv")
    if queue.empty:
        queue = load_csv("golden_review_queue.csv")
    if queue.empty:
        st.info("Run `python scripts/analyze_advanced.py` to generate the 100-question human benchmark queue.")
        return
    silver_labels = pd.read_csv(SILVER_LABELS).fillna("") if SILVER_LABELS.exists() else pd.DataFrame()
    current_golden = pd.read_csv(GOLDEN_LABELS).fillna("") if GOLDEN_LABELS.exists() else pd.DataFrame()
    if "verification_source" in current_golden.columns:
        current_golden = current_golden[~current_golden["verification_source"].astype(str).str.lower().str.contains("ai_")]
    c1, c2, c3 = st.columns(3)
    c1.metric("Human benchmark queue", len(queue))
    c2.metric("AI silver labels", len(silver_labels))
    c3.metric("Human gold labels", len(current_golden))
    st.caption("Review this 100-question benchmark queue. Human-saved labels go to data/verified/golden_labels.csv.")
    with st.expander("AI silver-label controls"):
        st.write("This uses the configured LLM to create AI silver labels. They are useful as a bootstrap, but they are not human gold labels.")
        limit = st.number_input("AI silver label limit", min_value=10, max_value=100, value=100, step=10)
        batch_size = st.number_input("Batch size", min_value=1, max_value=10, value=5, step=1)
        overwrite = st.checkbox("Regenerate existing AI silver labels", value=False)
        if st.button("Generate AI silver labels now"):
            with st.spinner("Generating AI silver labels. This may take a few minutes."):
                try:
                    from cuet_bst.ai_golden import generate_ai_silver_labels

                    labels = generate_ai_silver_labels(limit=int(limit), batch_size=int(batch_size), overwrite=overwrite)
                    st.success(f"Generated {len(labels)} AI silver labels. Rerun analysis to refresh accuracy metrics.")
                    st.cache_data.clear()
                except Exception as exc:
                    st.error(f"AI silver labeling failed: {exc}")
        if not silver_labels.empty:
            st.dataframe(silver_labels.head(50), use_container_width=True, height=260)
    editable_cols = [
        "canonical_question_id",
        "chapter",
        "subtopic",
        "micro_concept",
        "question_type",
        "question_pattern",
        "difficulty_estimate",
        "correct_option",
        "question_text",
        "option_a",
        "option_b",
        "option_c",
        "option_d",
    ]
    editable = queue[[column for column in editable_cols if column in queue.columns]].copy()
    if "question_type" not in editable.columns and "question_pattern" in editable.columns:
        editable["question_type"] = editable["question_pattern"]
    edited = st.data_editor(editable, use_container_width=True, height=650, num_rows="fixed")
    if st.button("Save golden labels"):
        VERIFIED_DIR.mkdir(parents=True, exist_ok=True)
        save_cols = ["canonical_question_id", "chapter", "subtopic", "micro_concept", "question_type", "difficulty_estimate", "correct_option"]
        labels = edited[[column for column in save_cols if column in edited.columns]].copy()
        labels["verified_at"] = pd.Timestamp.utcnow().isoformat()
        labels["verification_source"] = "human_verified"
        if GOLDEN_LABELS.exists():
            old = pd.read_csv(GOLDEN_LABELS).fillna("")
            if "verification_source" in old.columns:
                old = old[~old["verification_source"].astype(str).str.lower().str.contains("ai_")]
            combined = pd.concat([old, labels], ignore_index=True).drop_duplicates("canonical_question_id", keep="last")
        else:
            combined = labels
        combined.to_csv(GOLDEN_LABELS, index=False)
        st.success(f"Saved {len(labels)} golden labels to {GOLDEN_LABELS}")


def accuracy_evaluation_page() -> None:
    summary = load_csv("accuracy_summary.csv")
    silver_confusion = load_csv("silver_chapter_confusion_matrix.csv")
    human_confusion = load_csv("human_gold_chapter_confusion_matrix.csv")
    silver_common = load_csv("silver_classification_confusions.csv")
    human_common = load_csv("human_gold_classification_confusions.csv")
    if summary.empty:
        st.info("Run `python scripts/analyze_advanced.py` after saving golden labels.")
        return
    st.subheader("Accuracy Summary")
    st.dataframe(summary, use_container_width=True)
    for title, confusion, common in [
        ("Pipeline vs Silver AI Labels", silver_confusion, silver_common),
        ("Pipeline vs Human Gold Labels", human_confusion, human_common),
    ]:
        st.subheader(title)
        if confusion.empty:
            st.info("No confusion matrix available yet.")
            continue
        first_col = confusion.columns[0]
        matrix = confusion.rename(columns={first_col: "gold_chapter"}).set_index("gold_chapter")
        st.plotly_chart(px.imshow(matrix, text_auto=True, aspect="auto", title=f"{title}: Chapter Confusion Matrix"), use_container_width=True)
        if not common.empty:
            st.dataframe(common, use_container_width=True)


def suspicious_classifications_page() -> None:
    suspicious = load_csv("suspicious_classifications.csv")
    if suspicious.empty:
        st.success("No taxonomy mismatches currently detected.")
        return
    st.metric("Suspicious rows", len(suspicious))
    st.dataframe(suspicious, use_container_width=True, height=650)


def question_format_strategy_page() -> None:
    strategy = load_csv("question_format_strategy.csv")
    if strategy.empty:
        st.info("Run `python scripts/analyze_advanced.py` to generate question format strategy.")
        return
    st.dataframe(strategy, use_container_width=True, height=650)
    pattern_counts = strategy["dominant_question_pattern"].value_counts().rename_axis("pattern").reset_index(name="topics")
    st.plotly_chart(px.bar(pattern_counts, x="topics", y="pattern", orientation="h", title="Dominant Formats in High-Priority Topics"), use_container_width=True)


def ask_ai_page(data: pd.DataFrame) -> None:
    st.caption("Ask questions about the processed CUET BST analysis. The assistant uses approved CSVs and read-only summaries.")
    if "ai_chat_messages" not in st.session_state:
        st.session_state.ai_chat_messages = [
            {
                "role": "assistant",
                "content": "Ask me about priorities, charts, suspicious classifications, question formats, or what to study next.",
            }
        ]
    for message in st.session_state.ai_chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask about this CUET BST analysis")
    if not prompt:
        return
    st.session_state.ai_chat_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking over the analysis..."):
            try:
                from cuet_bst.dashboard_agent import answer_question, build_evidence, load_analysis_tables

                evidence = build_evidence(prompt, load_analysis_tables())
                answer = answer_question(prompt, evidence)
            except Exception as exc:
                answer = f"AI assistant failed: {exc}"
            st.markdown(answer)
            if "evidence" in locals() and evidence.chart is not None:
                render_chart_spec(evidence.chart)
            if "evidence" in locals():
                render_evidence(evidence)
            st.session_state.ai_chat_messages.append({"role": "assistant", "content": answer})


def manual_review_queue(data: pd.DataFrame) -> None:
    if "canonical_question_id" not in data.columns:
        st.info("Run the advanced pipeline to create canonical question IDs.")
        return
    mask = data.get("needs_review", "").astype(str).str.lower().isin(["true", "yes"])
    review = data[mask].copy()
    st.metric("Rows needing review", len(review))
    if review.empty:
        st.success("No low-confidence rows currently need review.")
        return
    suggestions = load_csv("ai_manual_review_suggestions.csv")
    if not suggestions.empty:
        st.subheader("AI review suggestions")
        st.dataframe(
            suggestions[
                [
                    column
                    for column in [
                        "canonical_question_id",
                        "ai_suggested_chapter",
                        "ai_suggested_subtopic",
                        "ai_suggested_question_pattern",
                        "ai_suggested_difficulty",
                        "ai_suggested_ncert_heading",
                        "ai_suggested_micro_concept",
                        "micro_concept_confidence",
                        "micro_concept_note",
                        "llm_label",
                        "final_confidence",
                        "review_reason",
                    ]
                    if column in suggestions.columns
                ]
            ],
            use_container_width=True,
            height=260,
        )
    columns = [
        "canonical_question_id",
        "chapter",
        "subtopic",
        "question_pattern",
        "difficulty_estimate",
        "ncert_heading",
        "micro_concept",
        "final_confidence",
        "review_reason",
        "question_text",
    ]
    edited = st.data_editor(review[[column for column in columns if column in review.columns]], use_container_width=True, height=560, num_rows="fixed")
    if st.button("Save manual labels"):
        save_cols = ["canonical_question_id", "chapter", "subtopic", "question_pattern", "difficulty_estimate", "ncert_heading", "micro_concept"]
        new_labels = edited[[column for column in save_cols if column in edited.columns]].copy()
        if MANUAL_LABELS.exists():
            old = pd.read_csv(MANUAL_LABELS).fillna("")
            combined = pd.concat([old, new_labels], ignore_index=True)
            combined = combined.drop_duplicates("canonical_question_id", keep="last")
        else:
            combined = new_labels
        combined.to_csv(MANUAL_LABELS, index=False)
        st.success(f"Saved {len(new_labels)} reviewed labels to {MANUAL_LABELS}")


def study_command_center_page(data: pd.DataFrame) -> None:
    priority = load_csv("study_priority.csv")
    high_roi = load_csv("high_roi_topics.csv")
    plans = load_csv("study_plans.csv")
    suspicious = load_csv("suspicious_classifications.csv")
    review = load_csv("ai_manual_review_suggestions.csv")
    questions = data.copy()

    st.subheader("What To Study Today")
    if not priority.empty:
        today = priority.sort_values(["priority_tier", "roi"], ascending=[True, False]).head(8)
        st.dataframe(today[[col for col in ["chapter", "subtopic", "priority_tier", "roi", "top_pattern", "top_micro_concept", "recommended_action"] if col in today.columns]], use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top Tier 1 Topics")
        tier1 = priority[priority.get("priority_tier", "").astype(str).eq("Tier 1 = Must do")] if not priority.empty else pd.DataFrame()
        st.dataframe(tier1.head(12), use_container_width=True, height=280)
    with c2:
        st.subheader("High ROI Topics")
        st.dataframe(high_roi.head(12), use_container_width=True, height=280)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Weak / Uncertain Classifications")
        st.metric("Suspicious", len(suspicious))
        st.metric("Manual review suggestions", len(review))
        if not suspicious.empty:
            st.dataframe(suspicious.head(8), use_container_width=True, height=260)
    with c4:
        st.subheader("PYQ Practice List")
        if not questions.empty:
            questions["weighted_frequency_score"] = pd.to_numeric(questions.get("weighted_frequency_score", 0), errors="coerce").fillna(0)
            pyq_cols = [col for col in ["chapter", "subtopic", "question_pattern", "micro_concept", "micro_concept_note", "question_text"] if col in questions.columns]
            st.dataframe(questions.sort_values("weighted_frequency_score", ascending=False)[pyq_cols].head(15), use_container_width=True, height=330)

    st.subheader("Emergency Plans")
    p1, p2 = st.columns(2)
    with p1:
        st.caption("3-day emergency plan")
        if not plans.empty:
            st.dataframe(plans[plans["plan_days"].astype(str).eq("3")].head(20), use_container_width=True, height=300)
    with p2:
        st.caption("5-day plan")
        if not plans.empty:
            st.dataframe(plans[plans["plan_days"].astype(str).eq("5")].head(24), use_container_width=True, height=300)

    st.subheader("Ask AI")
    question = st.text_input("Ask the command center", key="command_center_ai_input", placeholder="e.g. What should I study tonight and why?")
    if st.button("Ask Command Center AI") and question:
        try:
            from cuet_bst.dashboard_agent import answer_question, build_evidence, load_analysis_tables

            evidence = build_evidence(question, load_analysis_tables())
            st.markdown(answer_question(question, evidence))
            if evidence.chart is not None:
                render_chart_spec(evidence.chart)
            render_evidence(evidence)
        except Exception as exc:
            st.error(f"AI assistant failed: {exc}")


def render_chart_spec(chart) -> None:
    st.subheader(chart.title)
    if chart.chart_type == "bar":
        st.plotly_chart(px.bar(chart.data, x=chart.x, y=chart.y, color=chart.color or None, orientation="h"), use_container_width=True)
    elif chart.chart_type == "line":
        st.plotly_chart(px.line(chart.data, x=chart.x, y=chart.y, color=chart.color or None, markers=True), use_container_width=True)
    elif chart.chart_type == "heatmap":
        st.plotly_chart(px.imshow(chart.data, text_auto=True, aspect="auto"), use_container_width=True)
    elif chart.chart_type == "pie":
        st.plotly_chart(px.pie(chart.data, names=chart.x, values=chart.y), use_container_width=True)
    else:
        st.dataframe(chart.data, use_container_width=True)


def render_evidence(evidence) -> None:
    with st.expander("Answer evidence", expanded=True):
        st.write("Files used:", evidence.files_used)
        st.write("Numbers used:", evidence.numbers_used)
        st.write("Confidence level:", evidence.confidence_level)
        st.write("Caveats:", evidence.caveats)
        st.write("Top rows used:")
        st.json(evidence.top_rows_used)


def topic_network():
    corr = load_csv("chapter_correlation.csv")
    if corr.empty:
        return go.Figure(layout={"title": "Topic Co-occurrence Network"})
    first_col = corr.columns[0]
    corr = corr.rename(columns={first_col: "chapter"}).set_index("chapter")
    nodes = list(corr.index)
    if not nodes:
        return go.Figure(layout={"title": "Topic Co-occurrence Network"})
    import math

    positions = {
        node: (math.cos(2 * math.pi * idx / len(nodes)), math.sin(2 * math.pi * idx / len(nodes)))
        for idx, node in enumerate(nodes)
    }
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for source in nodes:
        for target in nodes:
            if source >= target:
                continue
            value = float(corr.loc[source, target]) if target in corr.columns else 0
            if value >= 0.35:
                x0, y0 = positions[source]
                x1, y1 = positions[target]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
    node_x = [positions[node][0] for node in nodes]
    node_y = [positions[node][1] for node in nodes]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line={"width": 1, "color": "#9aa4b2"}, hoverinfo="none"))
    fig.add_trace(
        go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            text=nodes,
            textposition="top center",
            marker={"size": 14, "color": "#2563eb"},
        )
    )
    fig.update_layout(title="Chapter Co-occurrence / Correlation Network", showlegend=False, xaxis_visible=False, yaxis_visible=False)
    return fig


def count_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    return data[column].replace("", "Unknown").value_counts().rename_axis(column).reset_index(name="count")


def weighted_table(data: pd.DataFrame, column: str) -> pd.DataFrame:
    if "weighted_frequency_score" not in data.columns:
        table = count_table(data, column)
        table["weighted_score"] = table["count"]
        return table
    temp = data.copy()
    temp[column] = temp[column].replace("", "Unknown")
    temp["weighted_frequency_score"] = pd.to_numeric(temp["weighted_frequency_score"], errors="coerce").fillna(1)
    return (
        temp.groupby(column)
        .agg(count=(column, "size"), weighted_score=("weighted_frequency_score", "sum"))
        .reset_index()
        .sort_values("weighted_score", ascending=False)
    )


def bar_count(data: pd.DataFrame, column: str, title: str):
    table = count_table(data, column).head(30)
    return px.bar(table, x="count", y=column, orientation="h", title=title)


def weighted_bar(data: pd.DataFrame, column: str, title: str):
    table = weighted_table(data, column).head(30)
    return px.bar(table, x="weighted_score", y=column, orientation="h", title=title)


def pie_count(data: pd.DataFrame, column: str, title: str):
    table = count_table(data, column)
    return px.pie(table, names=column, values="count", title=title)


def heatmap(data: pd.DataFrame, index: str, columns: str, title: str):
    pivot = pd.crosstab(data[index].replace("", "Unknown"), data[columns].replace("", "Unknown"))
    if pivot.empty:
        return go.Figure()
    return px.imshow(pivot, text_auto=True, aspect="auto", title=title)


def metric_value(quality: pd.DataFrame, metric: str, default):
    if quality.empty or "metric" not in quality.columns:
        return default
    matches = quality[quality["metric"] == metric]
    if matches.empty:
        return default
    value = matches.iloc[0]["value"]
    try:
        if float(value).is_integer():
            return int(float(value))
    except Exception:
        pass
    return value


def dashboard_ai_context(data: pd.DataFrame) -> str:
    pieces: list[str] = []
    quality = load_csv("data_quality_summary.csv")
    priority = load_csv("study_priority.csv")
    clusters = load_csv("micro_concept_clusters.csv")
    strategy = load_csv("question_format_strategy.csv")
    suspicious = load_csv("suspicious_classifications.csv")
    accuracy = load_csv("accuracy_summary.csv")
    plans = load_csv("study_plans.csv")
    if not quality.empty:
        pieces.append("DATA QUALITY\n" + quality.to_string(index=False))
    if not priority.empty:
        cols = [col for col in ["chapter", "subtopic", "raw_score", "percentile_rank", "priority_tier", "roi", "top_pattern", "top_micro_concept"] if col in priority.columns]
        pieces.append("TOP STUDY PRIORITIES\n" + priority[cols].head(30).to_string(index=False))
    if not clusters.empty:
        cols = [col for col in ["chapter", "subtopic", "micro_concept", "question_count", "weighted_score", "recency_score"] if col in clusters.columns]
        pieces.append("MICRO CONCEPT CLUSTERS\n" + clusters[cols].head(25).to_string(index=False))
    if not strategy.empty:
        cols = [col for col in ["chapter", "subtopic", "priority_tier", "dominant_question_pattern", "how_to_study_it", "common_traps", "ncert_heading_to_revise"] if col in strategy.columns]
        pieces.append("QUESTION FORMAT STRATEGY\n" + strategy[cols].head(20).to_string(index=False))
    if not suspicious.empty:
        pieces.append(f"SUSPICIOUS CLASSIFICATIONS COUNT\n{suspicious.shape[0]}")
    if not accuracy.empty:
        pieces.append("ACCURACY SUMMARY\n" + accuracy.to_string(index=False))
    if not plans.empty:
        cols = [col for col in ["plan_days", "day", "allocated_hours", "chapter", "subtopic", "roi", "recommended_action"] if col in plans.columns]
        pieces.append("3 AND 5 DAY STUDY PLANS\n" + plans[plans["plan_days"].astype(str).isin(["3", "5"])][cols].head(40).to_string(index=False))
    pieces.append(
        "CURRENT FILTERED VIEW\n"
        + json.dumps(
            {
                "rows": len(data),
                "chapters": int(data["chapter"].replace("", pd.NA).dropna().nunique()) if "chapter" in data.columns else 0,
                "needs_review": int(data.get("needs_review", "").astype(str).str.lower().isin(["true", "yes"]).sum()) if "needs_review" in data.columns else 0,
            },
            indent=2,
        )
    )
    return "\n\n".join(pieces)


df = load_questions()
if df.empty:
    st.warning("No processed questions yet. Run `python scripts/collect_data.py`, `python scripts/process_questions.py`, and `python scripts/analyze.py`.")
    st.stop()

page = st.sidebar.radio(
    "Page",
    [
        "Overview",
        "Data Quality",
        "Source Reliability",
        "Deduplication Report",
        "NCERT Reverse Index",
        "Chapter frequency",
        "Subtopic frequency",
        "Year-wise trend",
        "Question Pattern Analysis",
        "Micro-concept Clusters",
        "Reddit vs actual PYQ",
        "Raw question explorer",
        "High ROI Study Plan",
        "Human Benchmark Review",
        "Accuracy Evaluation",
        "Suspicious Classifications",
        "Question Format Strategy",
        "Study Command Center",
        "Ask AI",
        "Manual Review Queue",
    ],
)

filtered = apply_filters(df)

if page == "Overview":
    overview(filtered)
elif page == "Data Quality":
    data_quality_page()
elif page == "Source Reliability":
    source_reliability_page(filtered)
elif page == "Deduplication Report":
    dedupe_report_page()
elif page == "NCERT Reverse Index":
    ncert_reverse_index_page(filtered)
elif page == "Chapter frequency":
    chapter_frequency(filtered)
elif page == "Subtopic frequency":
    subtopic_frequency(filtered)
elif page == "Year-wise trend":
    year_trend(filtered)
elif page == "Question Pattern Analysis":
    question_type_analysis(filtered)
elif page == "Micro-concept Clusters":
    micro_clusters_page()
elif page == "Reddit vs actual PYQ":
    reddit_vs_actual()
elif page == "Raw question explorer":
    question_explorer(filtered)
elif page == "High ROI Study Plan":
    high_roi_plan_page()
elif page == "Human Benchmark Review":
    golden_dataset_review_page()
elif page == "Accuracy Evaluation":
    accuracy_evaluation_page()
elif page == "Suspicious Classifications":
    suspicious_classifications_page()
elif page == "Question Format Strategy":
    question_format_strategy_page()
elif page == "Study Command Center":
    study_command_center_page(filtered)
elif page == "Ask AI":
    ask_ai_page(filtered)
elif page == "Manual Review Queue":
    manual_review_queue(filtered)
