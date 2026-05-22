from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DATA_DIR = ROOT / "data" / "cuet_cs"
PROCESSED = DATA_DIR / "processed"
REPORTS = ROOT / "reports" / "cuet_cs"


st.set_page_config(page_title="CUET Computer Science Dashboard", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stToolbar"] { visibility: hidden; height: 0; position: fixed; }
    [data-testid="stDecoration"] { display: none; }
    [data-testid="stStatusWidget"] { visibility: hidden; height: 0; position: fixed; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; height: 0; }
    .cs-warning { padding: 0.85rem 1rem; border: 1px solid #f3c26b; background: #fff8ea; border-radius: 0.5rem; color: #5f4100; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = PROCESSED / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path).fillna("")


def metric_value(quality: pd.DataFrame, metric: str, fallback: object = 0) -> object:
    if quality.empty or "metric" not in quality.columns:
        return fallback
    match = quality[quality["metric"].astype(str).eq(metric)]
    if match.empty:
        return fallback
    return match.iloc[0].get("value", fallback)


def filtered(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    st.sidebar.header("Filters")
    for column in ["section", "unit", "chapter", "priority_tier", "concept_type", "difficulty"]:
        if column not in result.columns:
            continue
        values = sorted([str(value) for value in result[column].unique() if str(value).strip()])
        selected = st.sidebar.multiselect(column.replace("_", " ").title(), values)
        if selected:
            result = result[result[column].astype(str).isin(selected)]
    query = st.sidebar.text_input("Search")
    if query and not result.empty:
        result = result[result.astype(str).agg(" ".join, axis=1).str.contains(query, case=False, regex=False)]
    if not result.empty:
        st.sidebar.download_button("Export CSV", result.to_csv(index=False), "cuet_cs_filtered.csv")
    return result


def show_data_status() -> None:
    quality = load_csv("data_quality_summary.csv")
    status = metric_value(quality, "data_status", "No data status found. Run python scripts/init_cuet_cs.py.")
    st.markdown(
        f"<div class='cs-warning'><b>Data status:</b> {status} This dashboard is CS-only: Section A + Section B1. Frequency and prediction-style scores stay conservative until CS PYQs are imported and parsed.</div>",
        unsafe_allow_html=True,
    )


def overview() -> None:
    quality = load_csv("data_quality_summary.csv")
    priority = load_csv("study_priority.csv")
    taxonomy = load_csv("syllabus_taxonomy.csv")
    show_data_status()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Subject code", metric_value(quality, "subject_code", "308"))
    c2.metric("Parsed PYQ rows", metric_value(quality, "total_parsed_rows", 0))
    c3.metric("Syllabus topics", metric_value(quality, "syllabus_topics", len(taxonomy)))
    c4.metric("Chapters", metric_value(quality, "chapters", taxonomy.get("chapter", pd.Series(dtype=str)).nunique()))
    c5.metric("Sources configured", metric_value(quality, "configured_sources", 0))

    if not priority.empty:
        st.plotly_chart(px.bar(priority.head(25), x="raw_score", y="subtopic", color="chapter", orientation="h", title="Syllabus-Overlap Priority Until PYQs Are Parsed"), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(px.histogram(priority, y="priority_tier", title="Syllabus Priority Tiers"), use_container_width=True)
        with col2:
            st.plotly_chart(px.histogram(priority, y="top_pattern", title="Expected Question Pattern Mix"), use_container_width=True)


def syllabus_page() -> None:
    taxonomy = filtered(load_csv("syllabus_taxonomy.csv"))
    st.subheader("Official Syllabus Taxonomy")
    if taxonomy.empty:
        st.info("Run `python scripts/init_cuet_cs.py` first.")
        return
    st.dataframe(taxonomy, use_container_width=True, height=650)


def priority_page() -> None:
    priority = filtered(load_csv("study_priority.csv"))
    st.subheader("Study Priority")
    show_data_status()
    if priority.empty:
        return
    st.dataframe(priority, use_container_width=True, height=650)
    st.plotly_chart(px.bar(priority.head(30), x="raw_score", y="subtopic", color="chapter", orientation="h", title="Top Syllabus-Overlap Topics"), use_container_width=True)


def strategy_page() -> None:
    strategy = filtered(load_csv("question_format_strategy.csv"))
    st.subheader("Question Format Strategy")
    if strategy.empty:
        return
    for _, row in strategy.head(24).iterrows():
        with st.expander(f"{row.get('chapter', '')} -> {row.get('subtopic', '')}"):
            st.write(f"**Expected pattern:** {row.get('dominant_question_pattern', '')}")
            st.write(f"**How to study:** {row.get('how_to_study_it', '')}")
            st.write(f"**Common traps:** {row.get('common_traps', '')}")
            st.write(f"**Revise:** {row.get('ncert_heading_to_revise', '')}")
    st.dataframe(strategy, use_container_width=True, height=450)


def source_page() -> None:
    sources = load_csv("source_discovery.csv")
    st.subheader("Public Source Discovery")
    st.caption("Use only public pages/PDFs. Do not bypass login, paywalls, CAPTCHA, anti-bot blocks, or NTA 403 restrictions.")
    if sources.empty:
        return
    st.dataframe(sources, use_container_width=True, height=520, column_config={"url": st.column_config.LinkColumn("url")})
    st.markdown(
        """
        Manual import folders:

        - `data/cuet_cs/manual_imports/` for public PDFs, HTML, or CSV exports
        - `data/cuet_cs/manual_official_papers/` for manually downloaded official NTA PDFs
        """
    )


def raw_questions_page() -> None:
    questions = load_csv("questions_advanced.csv")
    st.subheader("Raw Question Explorer")
    if questions.empty:
        st.info("No parsed CUET Computer Science questions yet. Add public PDFs/CSVs to `data/cuet_cs/manual_imports/` and wire the parser when ready.")
        return
    st.dataframe(filtered(questions), use_container_width=True, height=650)


def ask_ai_page() -> None:
    st.subheader("Ask AI")
    st.caption("The assistant uses only CUET Computer Science processed CSV summaries. It will say when PYQ data is missing.")
    question = st.chat_input("Ask about CUET CS study priority, syllabus, SQL, Python, networks, or data import")
    if "cs_ai_messages" not in st.session_state:
        st.session_state.cs_ai_messages = [
            {"role": "assistant", "content": "Ask me what to study first for CUET Computer Science."}
        ]
    for message in st.session_state.cs_ai_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if not question:
        return
    st.session_state.cs_ai_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        answer = answer_ai(question)
        st.markdown(answer)
        st.session_state.cs_ai_messages.append({"role": "assistant", "content": answer})


def answer_ai(question: str) -> str:
    priority = load_csv("study_priority.csv")
    quality = load_csv("data_quality_summary.csv")
    strategy = load_csv("question_format_strategy.csv")
    sources = load_csv("source_discovery.csv")
    evidence = {
        "data_quality": quality.to_dict("records"),
        "top_priority_rows": priority.head(15).to_dict("records"),
        "strategy_rows": strategy.head(10).to_dict("records"),
        "source_rows": sources.head(10).to_dict("records"),
        "caveats": [
            "This is CUET Computer Science subject code 308, using Section A plus Section B1 only.",
            "Current CS dashboard is syllabus-initialized; parsed PYQ frequency is not available until CS papers are imported.",
            "Do not claim prediction certainty.",
        ],
    }
    try:
        from cuet_bst.llm_client import chat_completion

        return chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a CUET Computer Science study analyst. "
                        "Use only the evidence. If parsed PYQ data is missing, say that clearly. "
                        "Give concrete study actions but do not invent historical frequency."
                    ),
                },
                {"role": "user", "content": f"EVIDENCE:\n{evidence}\n\nQUESTION:\n{question}"},
            ],
            max_tokens=800,
            timeout=60,
        )
    except Exception as exc:
        return (
            f"AI assistant failed: {exc}\n\n"
            "Fallback: Start with SQL, database concepts, Python exception/file handling, stack, queue, searching, sorting, data handling basics, and networks/security. "
            "This recommendation is syllabus-overlap based, not PYQ-frequency based yet."
        )


def study_plan_page() -> None:
    priority = load_csv("study_priority.csv")
    st.subheader("Starter Study Plan")
    show_data_status()
    if priority.empty:
        return
    tiered = priority.sort_values(["raw_score", "difficulty"], ascending=[False, True]).head(18)
    days = {
        "3-day emergency": tiered.head(9),
        "5-day balanced": tiered.head(15),
        "7-day deeper": tiered.head(21),
    }
    for name, rows in days.items():
        with st.expander(name, expanded=name == "3-day emergency"):
            for day, (_, row) in enumerate(rows.iterrows(), start=1):
                st.write(f"**Block {day}: {row['chapter']} -> {row['subtopic']}**")
                st.caption(row["recommended_action"])


def main() -> None:
    st.title("CUET UG Computer Science Dashboard")
    st.caption("Subject code 308. CS-only scope: Section A + Section B1 from the official syllabus.")
    pages = {
        "Overview": overview,
        "Syllabus Taxonomy": syllabus_page,
        "Study Priority": priority_page,
        "Question Strategy": strategy_page,
        "Source Discovery": source_page,
        "Raw Question Explorer": raw_questions_page,
        "Study Plan": study_plan_page,
        "Ask AI": ask_ai_page,
    }
    choice = st.sidebar.radio("Page", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()
