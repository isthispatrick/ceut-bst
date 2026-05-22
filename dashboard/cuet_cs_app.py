from __future__ import annotations

from datetime import datetime, timezone
import sys
import uuid
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DATA_DIR = ROOT / "data" / "cuet_cs"
PROCESSED = DATA_DIR / "processed"
REPORTS = ROOT / "reports" / "cuet_cs"
ATTEMPTS_PATH = PROCESSED / "mock_attempts.csv"
GENERATED_PAPER_PATH = REPORTS / "cuet_cs_adaptive_mock_2026.csv"


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
    attempts = load_attempts()
    profile = build_weak_profile(attempts)
    if not attempts.empty:
        last_session = attempts.sort_values("attempted_at").iloc[-1]["session_id"]
        last = attempts[attempts["session_id"].eq(last_session)]
        attempted = int(last["attempted"].sum())
        correct = int(last["is_correct"].sum())
        score = int(last["score_delta"].sum())
        st.info(f"Remembered {attempts['session_id'].nunique()} mock attempts. Last score: {score}/{len(last) * 5}, accuracy: {(correct / attempted * 100) if attempted else 0:.1f}%.")

    paper_source = st.radio(
        "Paper",
        ["fresh_adaptive", "generated_adaptive_2026", "build_new_mock"],
        format_func=lambda value: {
            "fresh_adaptive": "Fresh adaptive mock",
            "generated_adaptive_2026": "Use exported PDF paper",
            "build_new_mock": "Build a custom mock from bank",
        }[value],
        horizontal=True,
    )
    if "cs_mock_seed" not in st.session_state:
        st.session_state.cs_mock_seed = uuid.uuid4().int % 999_999
    if st.button("Generate different paper", type="primary"):
        st.session_state.cs_mock_seed = uuid.uuid4().int % 999_999
        st.session_state.cs_exam_key = ""
        st.rerun()

    mode = "adaptive"
    seed = int(st.session_state.cs_mock_seed)
    if paper_source == "fresh_adaptive":
        target_questions = estimate_mock_size(blueprint, "full_cuet_style")
        candidate_bank = avoid_recent_questions(bank, attempts, target_questions)
        mock_questions = build_mock(candidate_bank, blueprint, "adaptive", seed, profile)
        st.caption("Generated a fresh adaptive full mock. It prioritizes high-value CS topics, avoids your recently attempted questions when possible, and changes when you click Generate different paper.")
    elif paper_source == "generated_adaptive_2026":
        mode = "generated_adaptive_2026"
        mock_questions = load_generated_paper()
        if mock_questions.empty:
            st.error("Generated paper CSV is missing. Recreate it from the reports/cuet_cs export.")
            return
        st.caption("Loaded the exported PDF paper as an interactive exam. Use Fresh adaptive mock for a different paper each practice.")
    else:
        chapters = sorted(bank["chapter"].dropna().astype(str).unique())
        default_chapters = sorted(profile["chapter"].head(4).tolist()) if not profile.empty else chapters
        selected_chapters = st.multiselect("Focus chapters", chapters, default=default_chapters or chapters)
        mode = st.radio("Mock type", ["adaptive", "quick", "focused", "full_cuet_style"], horizontal=True)
        seed = st.number_input("Shuffle seed", min_value=1, max_value=9999, value=305)
        filtered_bank = bank[bank["chapter"].astype(str).isin(selected_chapters)].copy()
        if filtered_bank.empty:
            st.info("Select at least one chapter.")
            return
        mock_questions = build_mock(filtered_bank, blueprint, mode, int(seed), profile)
        st.caption(f"Loaded {len(mock_questions)} questions. Full CUET-style mode targets 15 Section A + 25 Section B1 when enough questions are available.")
        if mode == "adaptive":
            st.caption("Adaptive mode weights high-value CS topics plus your wrong/skipped topics from saved attempts.")
    st.download_button(
        "Download this paper as CSV",
        mock_questions.to_csv(index=False),
        file_name=f"cuet_cs_{mode}_mock.csv",
        mime="text/csv",
    )
    render_cuet_exam(mock_questions, mode, bank)
    personal_coach(load_attempts(), bank)


def load_generated_paper() -> pd.DataFrame:
    if not GENERATED_PAPER_PATH.exists():
        return pd.DataFrame()
    paper = pd.read_csv(GENERATED_PAPER_PATH).fillna("")
    if "practice_id" not in paper.columns:
        paper["practice_id"] = [f"generated_{index + 1}" for index in range(len(paper))]
    return paper


def estimate_mock_size(blueprint: pd.DataFrame, mode: str) -> int:
    row = blueprint[blueprint["mock_type"].astype(str).eq(mode)]
    if row.empty:
        return 40
    return int(row.iloc[0]["questions"])


def avoid_recent_questions(bank: pd.DataFrame, attempts: pd.DataFrame, target_questions: int) -> pd.DataFrame:
    if attempts.empty or "practice_id" not in attempts.columns:
        return bank
    recent = attempts.sort_values("attempted_at", ascending=False).head(target_questions * 3)
    recent_ids = set(recent["practice_id"].astype(str))
    fresh = bank[~bank["practice_id"].astype(str).isin(recent_ids)].copy()
    return fresh if len(fresh) >= target_questions else bank


def render_cuet_exam(mock_questions: pd.DataFrame, mode: str, bank: pd.DataFrame) -> None:
    questions = mock_questions.reset_index(drop=True).copy()
    exam_key = f"cs_exam_{mode}_{'-'.join(questions['practice_id'].astype(str).head(8).tolist())}_{len(questions)}"
    if st.session_state.get("cs_exam_key") != exam_key:
        st.session_state.cs_exam_key = exam_key
        st.session_state.cs_exam_answers = {}
        st.session_state.cs_exam_current = 0
        st.session_state.cs_exam_submitted = False
        st.session_state.cs_exam_review = pd.DataFrame()
    answers = st.session_state.cs_exam_answers
    current = int(st.session_state.cs_exam_current)
    current = max(0, min(current, len(questions) - 1))

    answered = sum(1 for value in answers.values() if value != "Not answered")
    remaining = len(questions) - answered
    m1, m2, m3 = st.columns(3)
    m1.metric("Questions", len(questions))
    m2.metric("Answered", answered)
    m3.metric("Not answered", remaining)

    st.markdown("#### Question Palette")
    palette_cols = st.columns(10)
    for index, row in questions.iterrows():
        value = answers.get(row["practice_id"], "Not answered")
        label = f"{index + 1}" if value == "Not answered" else f"{index + 1}*"
        if palette_cols[index % 10].button(label, key=f"palette_{exam_key}_{index}", help="* means answered"):
            st.session_state.cs_exam_current = index
            st.rerun()

    row = questions.iloc[current]
    st.markdown("---")
    st.markdown(f"### Q{current + 1}. [{row['chapter']} -> {row['subtopic']}]")
    st.write(row["question_text"])
    options = {
        "A": row["option_a"],
        "B": row["option_b"],
        "C": row["option_c"],
        "D": row["option_d"],
    }
    previous = answers.get(row["practice_id"], "Not answered")
    choice = st.radio(
        "Choose one",
        ["Not answered", "A", "B", "C", "D"],
        index=["Not answered", "A", "B", "C", "D"].index(previous) if previous in ["Not answered", "A", "B", "C", "D"] else 0,
        format_func=lambda value: value if value == "Not answered" else f"{value}. {options[value]}",
        key=f"choice_{exam_key}_{row['practice_id']}",
    )
    answers[row["practice_id"]] = choice
    st.session_state.cs_exam_answers = answers

    n1, n2, n3, n4 = st.columns([1, 1, 1, 2])
    if n1.button("Previous", disabled=current == 0):
        st.session_state.cs_exam_current = current - 1
        st.rerun()
    if n2.button("Next", disabled=current == len(questions) - 1):
        st.session_state.cs_exam_current = current + 1
        st.rerun()
    if n3.button("Clear"):
        answers[row["practice_id"]] = "Not answered"
        st.session_state.cs_exam_answers = answers
        st.rerun()
    finish = n4.button("Finish And Analyze", type="primary")
    if finish:
        review, attempt_rows = score_mock_attempt(questions, answers, mode)
        save_attempts(attempt_rows)
        st.session_state.cs_exam_submitted = True
        st.session_state.cs_exam_review = review
        st.rerun()

    if st.session_state.cs_exam_submitted and not st.session_state.cs_exam_review.empty:
        show_mock_analysis(st.session_state.cs_exam_review, bank)


def score_mock_attempt(questions: pd.DataFrame, answers: dict[str, str], mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    review_rows = []
    attempt_rows = []
    session_id = str(uuid.uuid4())
    attempted_at = datetime.now(timezone.utc).isoformat()
    for _, row in questions.iterrows():
        chosen = answers.get(row["practice_id"], "Not answered")
        is_attempted = chosen != "Not answered"
        is_correct = chosen == row["correct_option"]
        result = "correct" if is_correct else ("skipped" if not is_attempted else "wrong")
        review_rows.append(
            {
                "chapter": row["chapter"],
                "subtopic": row["subtopic"],
                "chosen": chosen,
                "correct": row["correct_option"],
                "result": result,
                "explanation": row["explanation"],
                "priority_tier": row["priority_tier"],
                "question_type": row["question_type"],
                "difficulty": row["difficulty"],
            }
        )
        attempt_rows.append(
            {
                "session_id": session_id,
                "attempted_at": attempted_at,
                "mock_type": mode,
                "practice_id": row["practice_id"],
                "section": row["section"],
                "chapter": row["chapter"],
                "subtopic": row["subtopic"],
                "question_type": row["question_type"],
                "difficulty": row["difficulty"],
                "priority_tier": row["priority_tier"],
                "raw_score": row["raw_score"],
                "chosen": chosen,
                "correct_option": row["correct_option"],
                "attempted": int(is_attempted),
                "is_correct": int(is_correct),
                "score_delta": 5 if is_correct else (-1 if is_attempted else 0),
                "result": result,
            }
        )
    return pd.DataFrame(review_rows), pd.DataFrame(attempt_rows)


def show_mock_analysis(review: pd.DataFrame, bank: pd.DataFrame) -> None:
    attempted = int(review["chosen"].ne("Not answered").sum())
    correct = int(review["result"].eq("correct").sum())
    score = correct * 5 - (attempted - correct) * 1
    max_score = len(review) * 5
    scorecard = scorecard_summary(score, max_score, correct, attempted, len(review))
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Marks", f"{score}/{max_score}", f"{scorecard['score_percent']:.1f}%")
    s2.metric("Accuracy", f"{scorecard['accuracy']:.1f}%")
    s3.metric("Attempted", f"{attempted}/{len(review)}")
    s4.metric("Verdict", scorecard["label"])
    st.info(scorecard["message"])
    st.plotly_chart(px.histogram(review, y="chapter", color="result", title="Mock Result By Chapter"), use_container_width=True)
    weak = review[review["result"].isin(["wrong", "skipped"])]
    if not weak.empty:
        st.subheader("What To Practice Next")
        st.dataframe(weak[["chapter", "subtopic", "question_type", "difficulty", "result", "explanation", "priority_tier"]], use_container_width=True, height=360)
    st.subheader("Full Review")
    st.dataframe(review, use_container_width=True, height=520)
    st.caption("This attempt has been saved locally and will influence the next adaptive mock.")
    result_question = st.text_area(
        "Ask AI about this result",
        value="Analyze this mock result and tell me exactly what I should study before the next mock.",
        key="mock_result_ai_question",
    )
    if st.button("Ask AI Coach About This Result"):
        attempts = load_attempts()
        profile = build_weak_profile(attempts).head(10)
        with st.spinner("Asking the model to analyze this result..."):
            st.markdown(ai_attempt_analysis(profile, attempts, bank, user_question=result_question, latest_review=review, scorecard=scorecard))


def scorecard_summary(score: int, max_score: int, correct: int, attempted: int, total: int) -> dict[str, float | str]:
    accuracy = (correct / attempted * 100) if attempted else 0.0
    score_percent = (score / max_score * 100) if max_score else 0.0
    attempted_percent = (attempted / total * 100) if total else 0.0
    if score_percent >= 80 and accuracy >= 85:
        label = "Excellent"
        message = "Excellent mock. Keep speed steady, revise only the wrong/skipped rows, then take a harder adaptive paper."
    elif score_percent >= 65 and accuracy >= 75:
        label = "Good"
        message = "Good score. You are in a strong zone, but skipped or weak chapters can still cost marks. Fix the red rows before the next mock."
    elif score_percent >= 45:
        label = "Average"
        message = "Average attempt. This is workable, but you need targeted revision before it becomes exam-safe."
    else:
        label = "Needs Work"
        message = "Not good enough yet, but useful data. Do not panic: revise the weak-topic table, then retake a fresh adaptive mock."
    if attempted_percent < 60:
        message += " Your attempt rate is low, so focus on confidence and elimination practice too."
    return {
        "score_percent": score_percent,
        "accuracy": accuracy,
        "attempted_percent": attempted_percent,
        "label": label,
        "message": message,
    }


def build_mock(bank: pd.DataFrame, blueprint: pd.DataFrame, mode: str, seed: int, profile: pd.DataFrame | None = None) -> pd.DataFrame:
    if mode == "adaptive":
        mode = "full_cuet_style"
        bank = apply_adaptive_weights(bank, profile)
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
    weight_col = "adaptive_weight" if "adaptive_weight" in data.columns else "raw_score"
    weights = pd.to_numeric(data.get(weight_col, 1), errors="coerce").fillna(1).clip(lower=0.1)
    return data.sample(n=min(count, len(data)), replace=False, weights=weights, random_state=rng.randint(1, 1_000_000))


def random_state(seed: int):
    import random

    return random.Random(seed)


def load_attempts() -> pd.DataFrame:
    if not ATTEMPTS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(ATTEMPTS_PATH).fillna("")


def save_attempts(new_rows: pd.DataFrame) -> None:
    if new_rows.empty:
        return
    old = load_attempts()
    combined = pd.concat([old, new_rows], ignore_index=True) if not old.empty else new_rows
    combined.to_csv(ATTEMPTS_PATH, index=False)


def build_weak_profile(attempts: pd.DataFrame) -> pd.DataFrame:
    if attempts.empty:
        return pd.DataFrame(columns=["chapter", "subtopic", "wrong", "skipped", "attempted", "accuracy", "weakness_score"])
    data = attempts.copy()
    for column in ["attempted", "is_correct"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["wrong"] = ((data["attempted"].eq(1)) & (data["is_correct"].eq(0))).astype(int)
    data["skipped"] = data["attempted"].eq(0).astype(int)
    grouped = data.groupby(["chapter", "subtopic"], dropna=False).agg(
        wrong=("wrong", "sum"),
        skipped=("skipped", "sum"),
        attempted=("attempted", "sum"),
        correct=("is_correct", "sum"),
    ).reset_index()
    grouped["accuracy"] = grouped.apply(lambda row: row["correct"] / row["attempted"] if row["attempted"] else 0, axis=1)
    grouped["weakness_score"] = grouped["wrong"] * 2.0 + grouped["skipped"] * 0.8 + (1 - grouped["accuracy"])
    return grouped.sort_values("weakness_score", ascending=False)


def apply_adaptive_weights(bank: pd.DataFrame, profile: pd.DataFrame | None) -> pd.DataFrame:
    result = bank.copy()
    result["adaptive_weight"] = pd.to_numeric(result.get("raw_score", 1), errors="coerce").fillna(1)
    if profile is None or profile.empty:
        return result
    weak = profile[["chapter", "subtopic", "weakness_score"]]
    result = result.merge(weak, on=["chapter", "subtopic"], how="left")
    result["weakness_score"] = pd.to_numeric(result["weakness_score"], errors="coerce").fillna(0)
    result["adaptive_weight"] = result["adaptive_weight"] + result["weakness_score"] * 2.5
    return result


def personal_coach(attempts: pd.DataFrame, bank: pd.DataFrame) -> None:
    st.subheader("Personal Coach")
    if attempts.empty:
        st.write("Take one mock first. I will remember your mistakes and generate the next paper around your weak topics.")
        return
    profile = build_weak_profile(load_attempts()).head(10)
    if profile.empty:
        return
    st.dataframe(profile, use_container_width=True, height=260)
    coach_question = st.text_input(
        "Ask AI about your saved mock history",
        value="What are my weakest topics and what should I study today?",
        key="saved_result_ai_question",
    )
    if st.button("Ask AI Coach About Saved Results"):
        with st.spinner("Asking the model to analyze your weak spots..."):
            st.markdown(ai_attempt_analysis(profile, attempts, bank, user_question=coach_question))


def ai_attempt_analysis(
    profile: pd.DataFrame,
    attempts: pd.DataFrame,
    bank: pd.DataFrame,
    user_question: str = "",
    latest_review: pd.DataFrame | None = None,
    scorecard: dict[str, float | str] | None = None,
) -> str:
    evidence = {
        "weak_topics": profile.to_dict("records"),
        "attempt_summary": {
            "attempts": int(attempts["session_id"].nunique()),
            "rows": int(len(attempts)),
            "overall_accuracy": float(pd.to_numeric(attempts["is_correct"], errors="coerce").fillna(0).sum() / max(pd.to_numeric(attempts["attempted"], errors="coerce").fillna(0).sum(), 1)),
        },
        "latest_mock_scorecard": scorecard or {},
        "latest_mock_review_rows": latest_review.head(40).to_dict("records") if latest_review is not None and not latest_review.empty else [],
        "available_high_priority_questions": bank.sort_values("raw_score", ascending=False).head(12)[["chapter", "subtopic", "raw_score", "question_type"]].to_dict("records"),
        "caveat": "CUET CS 2026 exact paper cannot be predicted. Use this for adaptive practice based on saved mock mistakes plus CS syllabus priority.",
    }
    try:
        from cuet_bst.llm_client import chat_completion, configured_model

        return chat_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a CUET Computer Science coach. Use only the evidence. "
                        "Do not claim certainty about future questions. Give a precise study plan, weak-topic diagnosis, "
                        "and what the next adaptive mock should emphasize."
                    ),
                },
                {"role": "user", "content": f"EVIDENCE:\n{evidence}\n\nQUESTION:\n{user_question or 'Analyze my results and tell me what to study next.'}\nConfigured model: {configured_model()}"},
            ],
            max_tokens=900,
            timeout=60,
        )
    except Exception as exc:
        return f"AI analysis failed: {exc}\n\nFocus first on the weak topics in the table above, then retake adaptive mode."


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
    try:
        from cuet_bst.llm_client import configured_model

        st.caption(f"Model route: `{configured_model()}`")
    except Exception:
        pass
    typed_question = st.text_input(
        "Ask a question",
        placeholder="Example: Based on my mocks, what should I study today?",
        key="cs_ai_regular_input",
    )
    ask_clicked = st.button("Ask AI", type="primary")
    chat_question = st.chat_input("Ask about CUET CS study priority, syllabus, SQL, Python, networks, or data import")
    question = typed_question if ask_clicked and typed_question.strip() else chat_question
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
    attempts = load_attempts()
    weak_profile = build_weak_profile(attempts).head(12)
    evidence = {
        "data_quality": quality.to_dict("records"),
        "internet_evidence_summary": internet.to_dict("records"),
        "top_priority_rows": priority.head(15).to_dict("records"),
        "strategy_rows": strategy.head(10).to_dict("records"),
        "source_rows": sources.head(10).to_dict("records"),
        "question_metadata_sample": questions.head(10).to_dict("records"),
        "answer_key_sample": answers.head(10).to_dict("records"),
        "saved_mock_attempt_count": int(attempts["session_id"].nunique()) if not attempts.empty and "session_id" in attempts.columns else 0,
        "saved_mock_weak_topics": weak_profile.to_dict("records"),
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
