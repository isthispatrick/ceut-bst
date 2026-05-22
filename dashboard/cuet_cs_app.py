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
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Subject code", metric_value(quality, "subject_code", "308"))
    c2.metric("Question metadata rows", metric_value(quality, "total_parsed_rows", 0))
    c3.metric("Answer-key entries", metric_value(quality, "answer_key_entries", 0))
    c4.metric("Readable question text", metric_value(quality, "machine_readable_question_text_rows", 0))
    c5.metric("Syllabus topics", metric_value(quality, "syllabus_topics", len(taxonomy)))
    c6.metric("Chapters", metric_value(quality, "chapters", taxonomy.get("chapter", pd.Series(dtype=str)).nunique()))

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
    manifest = load_csv("internet_import_manifest.csv")
    st.subheader("Public Source Discovery")
    st.caption("Use only public pages/PDFs. Do not bypass login, paywalls, CAPTCHA, anti-bot blocks, or NTA 403 restrictions.")
    if sources.empty:
        return
    st.dataframe(sources, use_container_width=True, height=520, column_config={"url": st.column_config.LinkColumn("url")})
    if not manifest.empty:
        st.subheader("Imported Public Files")
        st.dataframe(manifest, use_container_width=True, height=320, column_config={"source_url": st.column_config.LinkColumn("source_url")})
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


def mock_practice_page() -> None:
    st.subheader("Interactive CUET CS Mock Practice")
    bank = load_csv("practice_question_bank.csv")
    blueprint = load_csv("mock_blueprint.csv")
    if bank.empty:
        st.info("Run `python scripts/generate_cuet_cs_mocks.py` to create the practice bank.")
        return
    st.warning(
        "These are CUET-style practice questions generated from the CS-only syllabus priority model and imported paper structure. "
        "They are not exact PYQs because the public PDFs do not expose machine-readable question text yet."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Practice questions", len(bank))
    c2.metric("Tier 1/Core", int(bank["priority_tier"].astype(str).str.contains("Tier 1", case=False, na=False).sum()))
    c3.metric("Chapters", bank["chapter"].nunique())
    c4.metric("Question types", bank["question_type"].nunique())

    chapters = sorted(bank["chapter"].dropna().astype(str).unique())
    selected_chapters = st.multiselect("Focus chapters", chapters, default=chapters)
    mode = st.radio("Mock type", ["quick", "focused", "full_cuet_style"], horizontal=True)
    seed = st.number_input("Shuffle seed", min_value=1, max_value=9999, value=305)
    filtered_bank = bank[bank["chapter"].astype(str).isin(selected_chapters)].copy()
    if filtered_bank.empty:
        st.info("Select at least one chapter.")
        return
    mock_questions = build_mock(filtered_bank, blueprint, mode, int(seed))
    st.caption(f"Loaded {len(mock_questions)} questions. Full CUET-style mode targets 15 Section A + 25 Section B1 when enough questions are available.")

    with st.form("cuet_cs_mock_form"):
        answers = {}
        for index, row in mock_questions.reset_index(drop=True).iterrows():
            st.markdown(f"**Q{index + 1}. [{row['chapter']} -> {row['subtopic']}]**")
            st.write(row["question_text"])
            options = {
                "A": row["option_a"],
                "B": row["option_b"],
                "C": row["option_c"],
                "D": row["option_d"],
            }
            answers[row["practice_id"]] = st.radio(
                "Choose one",
                ["Not answered", "A", "B", "C", "D"],
                format_func=lambda value, opts=options: value if value == "Not answered" else f"{value}. {opts[value]}",
                key=f"mock_{row['practice_id']}",
            )
            st.divider()
        submitted = st.form_submit_button("Submit mock")

    if submitted:
        review_rows = []
        correct = 0
        attempted = 0
        for _, row in mock_questions.iterrows():
            chosen = answers.get(row["practice_id"], "Not answered")
            is_attempted = chosen != "Not answered"
            is_correct = chosen == row["correct_option"]
            attempted += int(is_attempted)
            correct += int(is_correct)
            review_rows.append(
                {
                    "chapter": row["chapter"],
                    "subtopic": row["subtopic"],
                    "chosen": chosen,
                    "correct": row["correct_option"],
                    "result": "correct" if is_correct else ("skipped" if not is_attempted else "wrong"),
                    "explanation": row["explanation"],
                    "priority_tier": row["priority_tier"],
                }
            )
        score = correct * 5 - (attempted - correct) * 1
        max_score = len(mock_questions) * 5
        st.success(f"Score: {score} / {max_score} | Correct: {correct} | Attempted: {attempted} | Accuracy: {(correct / attempted * 100) if attempted else 0:.1f}%")
        review = pd.DataFrame(review_rows)
        st.plotly_chart(px.histogram(review, y="chapter", color="result", title="Mock Result By Chapter"), use_container_width=True)
        weak = review[review["result"].isin(["wrong", "skipped"])]
        if not weak.empty:
            st.subheader("What To Practice Next")
            st.dataframe(weak[["chapter", "subtopic", "result", "explanation", "priority_tier"]], use_container_width=True, height=360)
        st.subheader("Full Review")
        st.dataframe(review, use_container_width=True, height=520)


def build_mock(bank: pd.DataFrame, blueprint: pd.DataFrame, mode: str, seed: int) -> pd.DataFrame:
    row = blueprint[blueprint["mock_type"].astype(str).eq(mode)]
    if row.empty:
        total, section_a_count, section_b_count = 10, 4, 6
    else:
        total = int(row.iloc[0]["questions"])
        section_a_count = int(row.iloc[0]["section_a"])
        section_b_count = int(row.iloc[0]["section_b1"])
    rng = random_state(seed)
    section_a = bank[bank["section"].astype(str).eq("Section A Common Core")]
    section_b = bank[bank["section"].astype(str).eq("Section B1 Computer Science")]
    picked = pd.concat(
        [
            weighted_sample(section_a, min(section_a_count, len(section_a)), rng),
            weighted_sample(section_b, min(section_b_count, len(section_b)), rng),
        ],
        ignore_index=True,
    )
    if len(picked) < total:
        remaining = bank[~bank["practice_id"].isin(set(picked["practice_id"]))]
        picked = pd.concat([picked, weighted_sample(remaining, min(total - len(picked), len(remaining)), rng)], ignore_index=True)
    return picked.sample(frac=1, random_state=seed).reset_index(drop=True)


def weighted_sample(data: pd.DataFrame, count: int, rng) -> pd.DataFrame:
    if data.empty or count <= 0:
        return data.head(0)
    weights = pd.to_numeric(data.get("raw_score", 1), errors="coerce").fillna(1).clip(lower=0.1)
    return data.sample(n=min(count, len(data)), replace=False, weights=weights, random_state=rng.randint(1, 1_000_000))


def random_state(seed: int):
    import random

    return random.Random(seed)


def internet_evidence_page() -> None:
    st.subheader("Internet Evidence")
    summary = load_csv("internet_evidence_summary.csv")
    questions = load_csv("questions_advanced.csv")
    answers = load_csv("answer_key_entries.csv")
    manifest = load_csv("internet_import_manifest.csv")
    if summary.empty:
        st.info("Run `python scripts/collect_cuet_cs_data.py` to import public CS evidence.")
        return
    cols = st.columns(6)
    summary_map = dict(zip(summary["metric"], summary["value"]))
    cols[0].metric("Public papers", summary_map.get("public_question_papers_imported", 0))
    cols[1].metric("Question metadata", summary_map.get("question_metadata_rows", 0))
    cols[2].metric("CS-section rows", summary_map.get("cs_section_question_rows", 0))
    cols[3].metric("Answer keys", summary_map.get("answer_key_entries", 0))
    cols[4].metric("Readable text rows", summary_map.get("machine_readable_question_text_rows", 0))
    cols[5].metric("Topic validity", "blocked")
    st.warning(str(summary_map.get("topic_frequency_validity", "Topic frequency needs readable question text.")))
    if not questions.empty:
        st.plotly_chart(px.histogram(questions, x="year", color="section_scope", title="Imported Question Metadata By Year And Section"), use_container_width=True)
    if not answers.empty:
        st.plotly_chart(px.histogram(answers, x="year", color="correct_option", title="Answer-Key Entries By Year And Option"), use_container_width=True)
    st.dataframe(summary, use_container_width=True)
    with st.expander("Imported file manifest", expanded=False):
        st.dataframe(manifest, use_container_width=True, column_config={"source_url": st.column_config.LinkColumn("source_url")})


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
    internet = load_csv("internet_evidence_summary.csv")
    questions = load_csv("questions_advanced.csv")
    answers = load_csv("answer_key_entries.csv")
    evidence = {
        "data_quality": quality.to_dict("records"),
        "internet_evidence_summary": internet.to_dict("records"),
        "top_priority_rows": priority.head(15).to_dict("records"),
        "strategy_rows": strategy.head(10).to_dict("records"),
        "source_rows": sources.head(10).to_dict("records"),
        "question_metadata_sample": questions.head(10).to_dict("records"),
        "answer_key_sample": answers.head(10).to_dict("records"),
        "caveats": [
            "This is CUET Computer Science subject code 308, using Section A plus Section B1 only.",
            "Public papers and answer keys are imported, but most question bodies are not machine-readable yet.",
            "Topic frequency is not valid until OCR/manual question text is available.",
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
        "Internet Evidence": internet_evidence_page,
        "Raw Question Explorer": raw_questions_page,
        "Mock Practice": mock_practice_page,
        "Study Plan": study_plan_page,
        "Ask AI": ask_ai_page,
    }
    choice = st.sidebar.radio("Page", list(pages.keys()))
    pages[choice]()


if __name__ == "__main__":
    main()
