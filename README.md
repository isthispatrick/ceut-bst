# CUET Business Studies PYQ Analysis System

This project collects public CUET UG Business Studies (subject code 305) previous-year question material, extracts questions, deduplicates repeated copies, classifies them by syllabus/NCERT concepts, analyzes repetition patterns, and serves an interactive Streamlit dashboard.

The collector is conservative by design:

- checks `robots.txt` before automated fetches
- uses a clear user agent and a per-host delay
- caches downloads in `data/raw`
- does not bypass login, paywalls, CAPTCHA, or anti-bot restrictions
- writes skipped/failed downloads to `data/raw/manifest.jsonl`
- supports manual import when a site blocks scraping or uses protected/dynamic content

## Project Structure

```text
config/
  cuet_sources.json          Public source list and discovery queries
  cuet_topics.json           Official-syllabus taxonomy and keyword dictionary
  chapter_allowed_concepts.json Taxonomy guardrails for suspicious labels
dashboard/
  app.py                     Streamlit exam-intelligence dashboard
data/
  raw/                       Cached public downloads and imported files
  manual_imports/            Manual CSVs and discovered URLs
  manual_official_papers/    Place manually downloaded official NTA PDFs here
  verified/                  Human gold labels and AI silver labels
  processed/                 Cleaned datasets and analysis tables
reports/
  cuet_bst_analysis_report.pdf
  cuet_bst_final_study_pack.pdf
scripts/
  collect_data.py            Download public pages/PDFs respectfully
  import_manual.py           Register manually downloaded files
  process_questions.py       Extract, parse, and run baseline classification
  dedupe.py                  Canonical question IDs and duplicate reports
  classify_ensemble.py       Rule + TF-IDF/BM25 + embedding/LLM classification
  generate_ai_silver_labels.py AI-assisted silver labels for benchmark questions
  generate_ai_golden_labels.py Compatibility wrapper for silver labels
  analyze_advanced.py        Weighted scoring, study plans, PDF report
  analyze.py                 Legacy frequency report
  cuet_bst/                  Reusable pipeline modules
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

OCR is optional. If PDFs are scanned images, install Tesseract locally and set `CUET_ENABLE_OCR=true` in `.env`.

LLM classification is optional. If `HACKCLUB_AI_API_KEY`, `CUET_LLM_API_KEY`, or `OPENAI_API_KEY` is present, `scripts/classify_ensemble.py` uses the configured OpenAI-compatible `CUET_LLM_BASE_URL` and `CUET_LLM_MODEL`; otherwise it degrades to local rule + TF-IDF/BM25 + embedding-style similarity. The default Hack Club model is `ibm-granite/granite-4.1-8b`, chosen for cheap classification. `CUET_LLM_MAX_CALLS` and `CUET_LLM_MAX_TOKENS` cap spend per run.

The classifier also writes `data/processed/ai_manual_review_suggestions.csv` for low-confidence rows, so the Streamlit Manual Review Queue can show AI-proposed corrections before you save final labels to `data/manual_labels.csv`.

## Run The Pipeline

```powershell
python scripts/collect_data.py
python scripts/process_questions.py
python scripts/dedupe.py
python scripts/classify_ensemble.py
python scripts/analyze_advanced.py
python scripts/generate_ai_silver_labels.py --limit 100 --batch-size 5
python scripts/analyze_advanced.py
streamlit run dashboard/app.py
```

## CUET Computer Science Dashboard

This repo also includes a parallel Streamlit dashboard for CUET UG Computer Science, subject code `308`.
It is scoped to Computer Science only: Section A plus Section B1 from the official subject 308 syllabus. It keeps PYQ-frequency fields conservative until CS papers are imported and parsed.

```powershell
python scripts/init_cuet_cs.py
python scripts/collect_cuet_cs_data.py
python scripts/generate_cuet_cs_mocks.py
streamlit run dashboard/cuet_cs_app.py --server.port 8503
```

The CS dashboard reads and writes under:

```text
data/cuet_cs/raw/
data/cuet_cs/manual_imports/
data/cuet_cs/manual_official_papers/
data/cuet_cs/processed/
reports/cuet_cs/
```

Official source used for the initial taxonomy:

- NTA syllabus page: https://cuet.nta.nic.in/syllabus/
- Official subject 308 PDF: https://cdnbbsr.s3waas.gov.in/s3d1a21da7bca4abff8b0b61b87597de73/uploads/2025/03/2025030172.pdf
  - Dashboard uses Section A and Section B1 only.

The CS dashboard currently provides syllabus-overlap study priority, public PDF/answer-key import, source discovery, manual-import folders, question-format strategy, and an Ask AI page. It does not invent historical topic frequency when public PDFs expose only question IDs/answer keys but not machine-readable question text.

CS evidence importer outputs:

- `data/cuet_cs/processed/internet_import_manifest.csv`
- `data/cuet_cs/processed/cs_extraction_manifest.csv`
- `data/cuet_cs/processed/internet_evidence_summary.csv`
- `data/cuet_cs/processed/answer_key_entries.csv`
- `data/cuet_cs/processed/questions_advanced.csv`

Mock practice outputs:

- `data/cuet_cs/processed/practice_question_bank.csv`
- `data/cuet_cs/processed/mock_blueprint.csv`

The Streamlit mock page saves your personal attempts locally to `data/cuet_cs/processed/mock_attempts.csv`. That file is ignored by git so your weak-topic history stays on your machine.

The dashboard reads:

- `data/processed/questions_advanced.csv`
- `data/processed/questions_canonical.csv`
- `data/processed/data_quality_summary.csv`
- `data/processed/study_priority.csv`
- `data/processed/micro_concept_clusters.csv`
- `data/processed/reddit_insights.csv`
- `reports/cuet_bst_analysis_report.pdf`
- `reports/cuet_bst_final_study_pack.pdf`

## Manual Import Fallback

Use this whenever a site blocks scraping, requires JavaScript interaction, or exposes a download only after a normal browser click.

```powershell
python scripts/import_manual.py "C:\path\to\paper.pdf" --source-url "https://public-source.example/file.pdf"
python scripts/import_manual.py "C:\path\to\questions.csv" --source-name "manual-csv"
```

Manual CSVs should include any subset of the question columns. At minimum:

```csv
question_text,option_a,option_b,option_c,option_d,correct_option,year,source_url
```

To add search-discovered mirrors safely, paste public URLs into:

```text
data/manual_imports/discovered_urls.txt
```

Then rerun:

```powershell
python scripts/collect_data.py
```

Search queries are written to `data/manual_imports/search_queries.md`.

## Official Manual Import

If NTA blocks automated access, do not bypass it. Download the official CUET Business Studies PDF normally in your browser and place it in:

```text
data/manual_official_papers/
```

Then rerun:

```powershell
python scripts/process_questions.py
python scripts/dedupe.py
python scripts/classify_ensemble.py
python scripts/analyze_advanced.py
```

Rows parsed from this folder are marked `source_tier=official_manual` with `source_weight=1.0`.

## Outputs

Primary cleaned outputs:

- `data/processed/questions.csv`
- `data/processed/questions.json`
- `data/processed/questions_canonical.csv`
- `data/processed/questions_advanced.csv`
- `data/processed/data_quality_summary.csv`
- `data/processed/duplicate_groups.csv`
- `data/processed/ncert_reverse_index.csv`
- `data/processed/question_ncert_map.csv`
- `data/processed/question_patterns.csv`
- `data/processed/pattern_by_topic.csv`
- `data/processed/micro_concept_clusters.csv`
- `data/processed/topic_frequency.csv`
- `data/processed/repeated_concepts.csv`
- `data/processed/reddit_insights.csv`
- `data/processed/study_priority.csv`
- `data/processed/high_roi_topics.csv`
- `data/processed/low_frequency_topics.csv`
- `data/processed/hidden_gems.csv`
- `data/processed/overhyped_topics.csv`
- `data/processed/study_plans.csv`
- `data/processed/human_gold_review_queue.csv`
- `data/processed/accuracy_summary.csv`
- `data/processed/silver_chapter_confusion_matrix.csv`
- `data/processed/silver_classification_confusions.csv`
- `data/processed/human_gold_chapter_confusion_matrix.csv`
- `data/processed/human_gold_classification_confusions.csv`
- `data/processed/suspicious_classifications.csv`
- `data/processed/question_format_strategy.csv`
- `data/processed/chapter_year_heatmap.csv`
- `data/processed/chapter_correlation.csv`
- `reports/cuet_bst_analysis_report.pdf`
- `reports/cuet_bst_final_study_pack.pdf`

## Review Mode

Low-confidence or classifier-disagreement rows are marked with:

```text
needs_review = true
final_confidence < 0.75 or classifier disagreement
```

Open the dashboard's Manual Review Queue to edit chapter, subtopic, question pattern, difficulty, NCERT heading, and micro-concept. Corrections are saved to:

```text
data/manual_labels.csv
```

Future `python scripts/classify_ensemble.py` runs apply those manual labels first.

## Audit And Validation

The dashboard includes a Human Benchmark Review page for a 100-question benchmark queue:

- top 50 highest-weighted questions
- top 25 suspicious taxonomy classifications
- top 25 high-priority micro-concept rows

Human-reviewed labels are saved to:

```text
data/verified/golden_labels.csv
```

You can also generate AI-assisted silver labels:

```powershell
python scripts/generate_ai_silver_labels.py --limit 100 --batch-size 5
python scripts/analyze_advanced.py
```

AI-generated labels are saved to `data/verified/silver_labels_ai.csv`. They are **not** copied into `data/verified/golden_labels.csv`; that file is reserved for human-verified labels only.

After rerunning `python scripts/analyze_advanced.py`, the system writes separate `pipeline_vs_silver_*` and `pipeline_vs_human_gold_*` accuracy metrics. If no human labels exist, the dashboard shows “Human-verified accuracy unavailable yet.” The Suspicious Classifications page uses `config/chapter_allowed_concepts.json` to flag subtopics or micro-concepts that do not belong under the predicted chapter.

The dashboard also includes Ask AI and Study Command Center pages. Ask AI uses a read-only dataframe analysis layer over approved CSVs, appends evidence to answers, and can create safe predefined Plotly charts from natural language.

- Which Tier 1 topics should I study first?
- Why is this chapter high priority?
- Which topics are high ROI but underhyped?
- What should I revise in the last 3 days?

## Notes

The report and dashboard include raw-vs-unique data quality, source reliability, deduplication groups, weighted frequency, NCERT reverse index, pattern distribution, micro-concept clusters, Reddit/community-vs-actual comparison, high-ROI topics, low-frequency topics, and 3/5/7/14-day study plans.

Historical correlation is not a guarantee of future CUET question selection. Use the rankings as an allocation guide alongside NCERT and the latest official syllabus.
