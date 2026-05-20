import React, { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  AlertTriangle,
  BarChart3,
  BookOpen,
  Brain,
  CheckCircle2,
  Database,
  Download,
  FileQuestion,
  FileText,
  Layers,
  ListChecks,
  Search,
  ShieldAlert,
  Target,
  TrendingUp,
  Zap
} from 'lucide-react';
import studyPriorityCsv from '../data/processed/study_priority.csv?raw';
import studyPriorityUrl from '../data/processed/study_priority.csv?url';
import dataQualityCsv from '../data/processed/data_quality_summary.csv?raw';
import strategyCsv from '../data/processed/question_format_strategy.csv?raw';
import strategyUrl from '../data/processed/question_format_strategy.csv?url';
import accuracyCsv from '../data/processed/accuracy_summary.csv?raw';
import clustersCsv from '../data/processed/micro_concept_clusters.csv?raw';
import suspiciousCsv from '../data/processed/suspicious_classifications.csv?raw';
import plansCsv from '../data/processed/study_plans.csv?raw';
import topicFrequencyCsv from '../data/processed/topic_frequency.csv?raw';
import repeatedConceptsCsv from '../data/processed/repeated_concepts.csv?raw';
import reviewSuggestionsCsv from '../data/processed/ai_manual_review_suggestions.csv?raw';
import questionsAdvancedCsv from '../data/processed/questions_advanced.csv?raw';
import questionsAdvancedUrl from '../data/processed/questions_advanced.csv?url';
import redditInsightsCsv from '../data/processed/reddit_insights.csv?raw';
import ncertIndexCsv from '../data/processed/ncert_reverse_index.csv?raw';
import studyPackUrl from '../reports/cuet_bst_final_study_pack.pdf?url';
import analysisReportUrl from '../reports/cuet_bst_analysis_report.pdf?url';
import './styles.css';

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === ',' && !quoted) {
      row.push(cell);
      cell = '';
    } else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && next === '\n') index += 1;
      row.push(cell);
      if (row.some((value) => value.trim() !== '')) rows.push(row);
      row = [];
      cell = '';
    } else {
      cell += char;
    }
  }
  if (cell || row.length) {
    row.push(cell);
    rows.push(row);
  }
  const [headers = [], ...data] = rows;
  return data.map((values) =>
    Object.fromEntries(headers.map((header, index) => [header.trim(), (values[index] || '').trim()]))
  );
}

const tables = {
  studyPriority: parseCsv(studyPriorityCsv),
  dataQuality: parseCsv(dataQualityCsv),
  strategy: parseCsv(strategyCsv),
  accuracy: parseCsv(accuracyCsv),
  clusters: parseCsv(clustersCsv),
  suspicious: parseCsv(suspiciousCsv),
  plans: parseCsv(plansCsv),
  topicFrequency: parseCsv(topicFrequencyCsv),
  repeatedConcepts: parseCsv(repeatedConceptsCsv),
  reviewSuggestions: parseCsv(reviewSuggestionsCsv),
  questionsAdvanced: parseCsv(questionsAdvancedCsv),
  redditInsights: parseCsv(redditInsightsCsv),
  ncertIndex: parseCsv(ncertIndexCsv)
};

function number(value) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function topRows(rows, count, scoreKey) {
  return [...rows].sort((a, b) => number(b[scoreKey]) - number(a[scoreKey])).slice(0, count);
}

function qualityMetric(metricName, fallback = '0') {
  return tables.dataQuality.find((row) => row.metric === metricName)?.value || fallback;
}

function groupSum(rows, groupKey, valueKey = null) {
  const groups = {};
  rows.forEach((row) => {
    const key = row[groupKey] || 'Unknown';
    groups[key] ||= { label: key, count: 0, value: 0 };
    groups[key].count += 1;
    groups[key].value += valueKey ? number(row[valueKey]) : 1;
  });
  return Object.values(groups).sort((a, b) => b.value - a.value || b.count - a.count);
}

function unique(values) {
  return [...new Set(values.filter(Boolean))].sort();
}

function formatPercent(value) {
  const parsed = number(value);
  if (parsed <= 1 && parsed > 0) return `${(parsed * 100).toFixed(1)}%`;
  return `${parsed.toFixed(1)}%`;
}

const pages = [
  ['command', 'Study Command Center', Target],
  ['overview', 'Overview', BarChart3],
  ['quality', 'Data Quality', Database],
  ['sources', 'Source Reliability', ShieldAlert],
  ['priority', 'Study Priority', TrendingUp],
  ['patterns', 'Question Formats', Layers],
  ['concepts', 'Micro-Concepts', Brain],
  ['suspicious', 'Suspicious Classifications', AlertTriangle],
  ['accuracy', 'Accuracy Benchmarks', CheckCircle2],
  ['ncert', 'NCERT Reverse Index', BookOpen],
  ['reddit', 'Reddit vs Actual', Zap],
  ['explorer', 'Raw PYQ Explorer', FileQuestion],
  ['ask', 'Ask AI', Brain],
  ['reports', 'Reports & Exports', Download]
];

function App() {
  const [page, setPage] = useState('command');
  const [query, setQuery] = useState('');
  const [chapterFilter, setChapterFilter] = useState('All');
  const [aiQuestion, setAiQuestion] = useState('What should I study first and why?');
  const [aiResult, setAiResult] = useState(null);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState('');

  const chapters = useMemo(
    () => unique([...tables.studyPriority.map((row) => row.chapter), ...tables.questionsAdvanced.map((row) => row.chapter)]),
    []
  );

  const filteredPriority = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return tables.studyPriority.filter((row) => {
      const chapterMatch = chapterFilter === 'All' || row.chapter === chapterFilter;
      const queryMatch = !needle || `${row.chapter} ${row.subtopic} ${row.priority_tier} ${row.top_micro_concept}`.toLowerCase().includes(needle);
      return chapterMatch && queryMatch;
    });
  }, [chapterFilter, query]);

  const filteredQuestions = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return tables.questionsAdvanced.filter((row) => {
      const chapterMatch = chapterFilter === 'All' || row.chapter === chapterFilter;
      const haystack = `${row.question_text} ${row.chapter} ${row.subtopic} ${row.micro_concept} ${row.question_pattern}`.toLowerCase();
      return chapterMatch && (!needle || haystack.includes(needle));
    });
  }, [chapterFilter, query]);

  const sourceBreakdown = useMemo(() => {
    const groups = {};
    tables.questionsAdvanced.forEach((row) => {
      const tier = row.source_tier || 'unknown';
      groups[tier] ||= { source_tier: tier, rows: 0, unique_questions: new Set(), weighted_score: 0, avg_confidence: 0 };
      groups[tier].rows += 1;
      groups[tier].unique_questions.add(row.canonical_question_id || row.question_id || row.question_text);
      groups[tier].weighted_score += number(row.source_weight);
      groups[tier].avg_confidence += number(row.final_confidence || row.confidence_score);
    });
    return Object.values(groups)
      .map((row) => ({
        ...row,
        unique_questions: row.unique_questions.size,
        avg_confidence: row.rows ? (row.avg_confidence / row.rows).toFixed(2) : '0.00',
        weighted_score: row.weighted_score.toFixed(1)
      }))
      .sort((a, b) => b.rows - a.rows);
  }, []);

  async function askAi(event) {
    event?.preventDefault();
    if (!aiQuestion.trim()) return;
    setAiLoading(true);
    setAiError('');
    setAiResult(null);
    try {
      const response = await fetch('/api/ask-ai', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: aiQuestion })
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Ask AI failed');
      setAiResult(payload);
    } catch (error) {
      setAiError(error.message);
    } finally {
      setAiLoading(false);
    }
  }

  const context = {
    page,
    setPage,
    query,
    setQuery,
    chapterFilter,
    setChapterFilter,
    chapters,
    filteredPriority,
    filteredQuestions,
    sourceBreakdown,
    aiQuestion,
    setAiQuestion,
    aiResult,
    aiLoading,
    aiError,
    askAi
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span>BST</span>
          <div>
            <strong>CUET Intelligence</strong>
            <small>Subject code 305</small>
          </div>
        </div>
        <nav>
          {pages.map(([id, label, Icon]) => (
            <button key={id} className={page === id ? 'active' : ''} type="button" onClick={() => setPage(id)}>
              <Icon size={17} />
              {label}
            </button>
          ))}
        </nav>
      </aside>
      <main className="workspace">
        <TopBar {...context} />
        {page === 'command' && <CommandCenter {...context} />}
        {page === 'overview' && <Overview />}
        {page === 'quality' && <DataQuality />}
        {page === 'sources' && <SourceReliability sourceBreakdown={sourceBreakdown} />}
        {page === 'priority' && <StudyPriority {...context} />}
        {page === 'patterns' && <QuestionFormats />}
        {page === 'concepts' && <MicroConcepts />}
        {page === 'suspicious' && <Suspicious />}
        {page === 'accuracy' && <Accuracy />}
        {page === 'ncert' && <NcertIndex />}
        {page === 'reddit' && <RedditVsActual />}
        {page === 'explorer' && <RawExplorer {...context} />}
        {page === 'ask' && <AskAi {...context} />}
        {page === 'reports' && <Reports />}
      </main>
    </div>
  );
}

function TopBar({ page, query, setQuery, chapterFilter, setChapterFilter, chapters }) {
  const label = pages.find(([id]) => id === page)?.[1] || 'Dashboard';
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">Deployed Vercel dashboard</p>
        <h1>{label}</h1>
      </div>
      <div className="toolbar">
        <label className="input-wrap">
          <Search size={16} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search topics or PYQs" />
        </label>
        <select value={chapterFilter} onChange={(event) => setChapterFilter(event.target.value)}>
          <option>All</option>
          {chapters.map((chapter) => (
            <option key={chapter}>{chapter}</option>
          ))}
        </select>
      </div>
    </header>
  );
}

function CommandCenter(props) {
  const tier1 = tables.studyPriority.filter((row) => row.priority_tier === 'Tier 1 = Must do');
  const highRoi = topRows(tables.studyPriority, 8, 'roi');
  const threeDay = tables.plans.filter((row) => row.plan_days === '3');
  const fiveDay = tables.plans.filter((row) => row.plan_days === '5');
  const practice = topRows(tables.questionsAdvanced, 8, 'weighted_frequency_score');
  return (
    <div className="stack">
      <AlertBanner />
      <section className="metric-grid">
        <Metric label="Parsed rows" value={qualityMetric('total_parsed_rows')} />
        <Metric label="Canonical questions" value={qualityMetric('unique_questions_after_dedupe')} />
        <Metric label="Duplicate rate" value={`${qualityMetric('duplicate_rate_percent')}%`} />
        <Metric label="Manual review rows" value={qualityMetric('low_confidence_classifications')} tone="warn" />
        <Metric label="Official rows" value={qualityMetric('official_source_rows')} tone="warn" />
      </section>
      <section className="grid two">
        <Panel title="What To Study Today" icon={<Target size={19} />}>
          <TopicCards rows={tier1.slice(0, 8)} />
        </Panel>
        <Panel title="Fast Marks: High ROI" icon={<TrendingUp size={19} />}>
          <TopicCards rows={highRoi} scoreKey="roi" />
        </Panel>
      </section>
      <section className="grid two">
        <Panel title="3-Day Emergency Plan" icon={<Zap size={19} />}>
          <StudyPlan rows={threeDay} />
        </Panel>
        <Panel title="5-Day Score Plan" icon={<ListChecks size={19} />}>
          <StudyPlan rows={fiveDay} />
        </Panel>
      </section>
      <section className="grid two">
        <Panel title="PYQ Practice List" icon={<FileQuestion size={19} />}>
          <QuestionList rows={practice} />
        </Panel>
        <Panel title="Ask AI From The Command Center" icon={<Brain size={19} />}>
          <AskAi compact {...props} />
        </Panel>
      </section>
    </div>
  );
}

function Overview() {
  const chapterFrequency = groupSum(tables.topicFrequency, 'chapter', 'question_count').slice(0, 12);
  const tierDistribution = groupSum(tables.studyPriority, 'priority_tier');
  const topRepeated = topRows(tables.repeatedConcepts, 10, 'weighted_score');
  return (
    <div className="stack">
      <AlertBanner />
      <section className="metric-grid">
        <Metric label="Unique questions" value={qualityMetric('unique_questions_after_dedupe')} />
        <Metric label="Question rows" value={qualityMetric('total_parsed_rows')} />
        <Metric label="Third-party rows" value={qualityMetric('third_party_source_rows')} />
        <Metric label="Suspicious classifications" value={tables.suspicious.length} tone="warn" />
      </section>
      <section className="grid two">
        <Panel title="Chapter Frequency" icon={<BarChart3 size={19} />}>
          <BarList rows={chapterFrequency} labelKey="label" valueKey="value" />
        </Panel>
        <Panel title="Priority Tier Distribution" icon={<Layers size={19} />}>
          <BarList rows={tierDistribution} labelKey="label" valueKey="count" />
        </Panel>
      </section>
      <Panel title="Most Repeated Concepts" icon={<Brain size={19} />}>
        <DataTable rows={topRepeated} columns={['chapter', 'subtopic', 'micro_concept', 'weighted_score', 'question_count']} />
      </Panel>
    </div>
  );
}

function DataQuality() {
  const rows = tables.dataQuality.map((row) => ({ metric: row.metric.replaceAll('_', ' '), value: row.value }));
  const review = tables.reviewSuggestions.slice(0, 8);
  return (
    <div className="stack">
      <AlertBanner />
      <section className="grid two">
        <Panel title="Quality Summary" icon={<Database size={19} />}>
          <DataTable rows={rows} columns={['metric', 'value']} />
        </Panel>
        <Panel title="Review Pressure" icon={<AlertTriangle size={19} />}>
          <Metric label="AI review suggestions" value={tables.reviewSuggestions.length} tone="warn" />
          <Metric label="Suspicious taxonomy rows" value={tables.suspicious.length} tone="warn" />
          <Metric label="Human gold labels" value={qualityMetric('human_verified_labels', '0')} />
        </Panel>
      </section>
      <Panel title="Top AI Manual Review Suggestions" icon={<FileQuestion size={19} />}>
        <DataTable rows={review} columns={['canonical_question_id', 'chapter', 'subtopic', 'micro_concept', 'review_reason']} />
      </Panel>
    </div>
  );
}

function SourceReliability({ sourceBreakdown }) {
  return (
    <div className="stack">
      <AlertBanner />
      <Panel title="Source Reliability Breakdown" icon={<ShieldAlert size={19} />}>
        <DataTable rows={sourceBreakdown} columns={['source_tier', 'rows', 'unique_questions', 'weighted_score', 'avg_confidence']} />
      </Panel>
      <Panel title="Why This Matters" icon={<BookOpen size={19} />}>
        <p className="body-copy">
          The system weights official NTA rows highest, structured PYQ sites below that, blogs lower, and community sentiment
          as weak signal. Current official-source rows are {qualityMetric('official_source_rows')}, so the deployed rankings
          are useful for study allocation but should still be validated with manually imported official papers.
        </p>
      </Panel>
    </div>
  );
}

function StudyPriority({ filteredPriority }) {
  return (
    <div className="stack">
      <Panel title="Priority Table" icon={<TrendingUp size={19} />}>
        <DataTable
          rows={filteredPriority}
          columns={[
            'chapter',
            'subtopic',
            'priority_tier',
            'percentile_rank',
            'raw_score',
            'source_weighted_frequency',
            'recency_score',
            'roi',
            'top_pattern',
            'top_micro_concept'
          ]}
          limit={80}
        />
      </Panel>
    </div>
  );
}

function QuestionFormats() {
  return (
    <div className="stack">
      <Panel title="High-Priority Format Strategy" icon={<Layers size={19} />}>
        <StrategyGrid rows={tables.strategy.slice(0, 18)} />
      </Panel>
      <Panel title="Detailed Strategy Table" icon={<FileText size={19} />}>
        <DataTable
          rows={tables.strategy}
          columns={['chapter', 'subtopic', 'dominant_question_pattern', 'how_to_study_it', 'common_traps', 'ncert_heading_to_revise']}
          limit={60}
        />
      </Panel>
    </div>
  );
}

function MicroConcepts() {
  const topConcepts = topRows(tables.clusters, 18, 'weighted_score');
  return (
    <div className="stack">
      <Panel title="Repeated Micro-Concept Clusters" icon={<Brain size={19} />}>
        <BarList rows={topConcepts} labelKey="micro_concept" valueKey="weighted_score" metaKey="chapter" />
      </Panel>
      <Panel title="Micro-Concept Details" icon={<Database size={19} />}>
        <DataTable rows={tables.clusters} columns={['chapter', 'subtopic', 'micro_concept', 'question_count', 'weighted_score', 'years_seen']} limit={80} />
      </Panel>
    </div>
  );
}

function Suspicious() {
  const byChapter = groupSum(tables.suspicious, 'chapter').slice(0, 12);
  return (
    <div className="stack">
      <Panel title="Suspicious Classifications By Chapter" icon={<AlertTriangle size={19} />}>
        <BarList rows={byChapter} labelKey="label" valueKey="count" />
      </Panel>
      <Panel title="Rows To Review" icon={<FileQuestion size={19} />}>
        <DataTable rows={tables.suspicious} columns={['canonical_question_id', 'chapter', 'subtopic', 'micro_concept', 'review_reason']} limit={80} />
      </Panel>
    </div>
  );
}

function Accuracy() {
  const silverRows = tables.accuracy.filter((row) => row.metric.startsWith('pipeline_vs_silver'));
  const humanStatus = tables.accuracy.find((row) => row.metric === 'pipeline_vs_human_gold_status')?.value;
  const chapterAccuracy = tables.accuracy.find((row) => row.metric === 'pipeline_vs_silver_chapter_accuracy')?.value;
  const subtopicAccuracy = tables.accuracy.find((row) => row.metric === 'pipeline_vs_silver_subtopic_accuracy')?.value;
  const microAccuracy = tables.accuracy.find((row) => row.metric === 'pipeline_vs_silver_micro_concept_accuracy')?.value;
  const typeAccuracy = tables.accuracy.find((row) => row.metric === 'pipeline_vs_silver_question_type_accuracy')?.value;
  return (
    <div className="stack">
      <section className="metric-grid">
        <Metric label="Chapter accuracy vs silver" value={formatPercent(chapterAccuracy)} />
        <Metric label="Subtopic accuracy vs silver" value={formatPercent(subtopicAccuracy)} />
        <Metric label="Micro-concept vs silver" value={formatPercent(microAccuracy)} tone="warn" />
        <Metric label="Question type vs silver" value={formatPercent(typeAccuracy)} />
      </section>
      <Panel title="Accuracy Summary" icon={<CheckCircle2 size={19} />}>
        <DataTable rows={silverRows} columns={['metric', 'value']} />
      </Panel>
      <Panel title="Human Gold Benchmark" icon={<AlertTriangle size={19} />}>
        <p className="body-copy">{humanStatus || 'Human-verified accuracy unavailable yet.'}</p>
        <p className="body-copy">
          Silver labels are AI-seeded labels, not ground truth. Use the local review queue to create `data/verified/golden_labels.csv`.
        </p>
      </Panel>
    </div>
  );
}

function NcertIndex() {
  return (
    <div className="stack">
      <Panel title="NCERT / Syllabus Reverse Index" icon={<BookOpen size={19} />}>
        <DataTable rows={tables.ncertIndex} columns={['chapter', 'subtopic', 'ncert_heading', 'ncert_concept', 'concept_type']} limit={100} />
      </Panel>
    </div>
  );
}

function RedditVsActual() {
  return (
    <div className="stack">
      <AlertBanner />
      <Panel title="Community Signal vs PYQ Data" icon={<Zap size={19} />}>
        <DataTable rows={tables.redditInsights} columns={['topic', 'reddit_hype_score', 'actual_weighted_frequency', 'mismatch_type', 'summary']} limit={80} />
      </Panel>
    </div>
  );
}

function RawExplorer({ filteredQuestions }) {
  return (
    <div className="stack">
      <Panel title="Raw Question Explorer" icon={<FileQuestion size={19} />}>
        <DataTable
          rows={filteredQuestions}
          columns={[
            'year',
            'chapter',
            'subtopic',
            'micro_concept',
            'question_pattern',
            'difficulty_estimate',
            'question_text',
            'correct_option',
            'source_tier'
          ]}
          limit={100}
        />
      </Panel>
    </div>
  );
}

function AskAi({ aiQuestion, setAiQuestion, aiResult, aiLoading, aiError, askAi, compact = false }) {
  return (
    <section className={compact ? 'ask-card compact' : 'panel ask-card'}>
      {!compact && (
        <div className="panel-title">
          <Brain size={19} />
          <h2>Ask AI Data Analyst</h2>
        </div>
      )}
      <p className="muted">
        The assistant reads approved processed CSV summaries through a Vercel serverless function. It cannot execute arbitrary
        code and will include evidence, numbers, caveats, and source files.
      </p>
      <form className="ask-form" onSubmit={askAi}>
        <input
          value={aiQuestion}
          onChange={(event) => setAiQuestion(event.target.value)}
          placeholder="Try: compare Marketing and Business Finance"
        />
        <button type="submit" disabled={aiLoading}>
          {aiLoading ? 'Thinking...' : 'Ask'}
        </button>
      </form>
      <div className="quick-prompts">
        {[
          'show top 10 topics by weighted score',
          'compare Marketing and Business Finance',
          'show chapter frequency',
          'show suspicious classifications by chapter',
          'what should I study in 3 days?'
        ].map((prompt) => (
          <button key={prompt} type="button" onClick={() => setAiQuestion(prompt)}>
            {prompt}
          </button>
        ))}
      </div>
      {aiError && <p className="error">{aiError}</p>}
      {aiResult && <AiResult result={aiResult} />}
    </section>
  );
}

function Reports() {
  const exports = [
    ['Final study pack PDF', studyPackUrl, 'Tier 1 and Tier 2 topics, PYQ examples, traps, 3-day and 5-day plans.'],
    ['Analysis report PDF', analysisReportUrl, 'Full pipeline report with data quality, source caveats, and ranking logic.'],
    ['Study priority CSV', studyPriorityUrl, 'Weighted topic scores and priority tiers.'],
    ['Questions advanced CSV', questionsAdvancedUrl, 'Canonical question-level dataset.'],
    ['Question strategy CSV', strategyUrl, 'How to study each high-priority topic.']
  ];
  return (
    <div className="stack">
      <Panel title="Download Reports And Data" icon={<Download size={19} />}>
        <div className="download-grid">
          {exports.map(([label, href, description]) => (
            <a key={href} href={href} className="download-card">
              <Download size={19} />
              <strong>{label}</strong>
              <span>{description}</span>
            </a>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function AlertBanner() {
  return (
    <section className="notice">
      <AlertTriangle size={19} />
      <span>
        Historical PYQ analysis only. Official NTA rows are currently {qualityMetric('official_source_rows')}; use this as
        a study decision engine, not a guaranteed paper prediction.
      </span>
    </section>
  );
}

function Metric({ label, value, tone = 'default' }) {
  return (
    <article className={`metric ${tone}`}>
      <span>{label}</span>
      <strong>{value || '0'}</strong>
    </article>
  );
}

function Panel({ title, icon, children }) {
  return (
    <section className="panel">
      <div className="panel-title">
        {icon}
        <h2>{title}</h2>
      </div>
      {children}
    </section>
  );
}

function TopicCards({ rows, scoreKey = 'raw_score' }) {
  return (
    <div className="topic-list">
      {rows.map((row) => (
        <article key={`${row.chapter}-${row.subtopic}`}>
          <div>
            <strong>{row.subtopic}</strong>
            <span>{row.chapter}</span>
            <small>{row.top_pattern || row.top_micro_concept}</small>
          </div>
          <b>{row[scoreKey]}</b>
        </article>
      ))}
    </div>
  );
}

function StudyPlan({ rows }) {
  if (!rows.length) return <p className="muted">No study-plan rows found.</p>;
  return (
    <div className="plan-list">
      {rows.slice(0, 8).map((row) => (
        <article key={`${row.plan_days}-${row.day}-${row.chapter}-${row.subtopic}`}>
          <b>Day {row.day}</b>
          <div>
            <strong>{row.chapter}</strong>
            <span>{row.subtopic}</span>
            <small>{row.action || row.study_action}</small>
          </div>
        </article>
      ))}
    </div>
  );
}

function QuestionList({ rows }) {
  return (
    <div className="question-list">
      {rows.map((row) => (
        <article key={row.canonical_question_id || row.question_text}>
          <strong>{row.chapter}{' -> '}{row.subtopic}</strong>
          <p>{row.question_text}</p>
          <span>{row.question_pattern || row.question_type} - {row.difficulty_estimate}</span>
        </article>
      ))}
    </div>
  );
}

function StrategyGrid({ rows }) {
  return (
    <div className="strategy-grid">
      {rows.map((row) => (
        <article key={`${row.chapter}-${row.subtopic}`}>
          <strong>{row.chapter}{' -> '}{row.subtopic}</strong>
          <span>{row.dominant_question_pattern}</span>
          <p>{row.how_to_study_it}</p>
          <small>{row.common_traps}</small>
        </article>
      ))}
    </div>
  );
}

function BarList({ rows, labelKey, valueKey, metaKey }) {
  const max = Math.max(...rows.map((row) => number(row[valueKey])), 1);
  return (
    <div className="bar-list">
      {rows.map((row) => {
        const value = number(row[valueKey]);
        return (
          <article key={`${row[labelKey]}-${row[metaKey] || value}`}>
            <div className="bar-label">
              <strong>{row[labelKey]}</strong>
              {metaKey && <span>{row[metaKey]}</span>}
              <b>{value.toFixed(value > 20 ? 0 : 1)}</b>
            </div>
            <div className="bar-track">
              <span style={{ width: `${Math.max(4, (value / max) * 100)}%` }} />
            </div>
          </article>
        );
      })}
    </div>
  );
}

function DataTable({ rows, columns, limit = 40 }) {
  const visibleRows = rows.slice(0, limit);
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column}>{column.replaceAll('_', ' ')}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {visibleRows.map((row, index) => (
            <tr key={`${index}-${columns.map((column) => row[column]).join('-')}`}>
              {columns.map((column) => (
                <td key={column}>{row[column] || ''}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      {rows.length > limit && <p className="table-note">Showing {limit} of {rows.length} rows. Use search or chapter filter to narrow the table.</p>}
    </div>
  );
}

function AiResult({ result }) {
  const { answer, evidence } = result;
  return (
    <div className="ai-result">
      <article>
        <h3>Answer</h3>
        <div className="ai-answer">
          <MarkdownLite text={answer} />
        </div>
      </article>
      {evidence?.chart && <ChartRenderer chart={evidence.chart} />}
      {evidence && (
        <article>
          <h3>Evidence</h3>
          <dl className="evidence">
            <div>
              <dt>Files used</dt>
              <dd>{(evidence.filesUsed || []).join(', ')}</dd>
            </div>
            <div>
              <dt>Confidence</dt>
              <dd>{evidence.confidenceLevel}</dd>
            </div>
            <div>
              <dt>Numbers</dt>
              <dd>
                <code>{JSON.stringify(evidence.numbersUsed || {})}</code>
              </dd>
            </div>
            <div>
              <dt>Caveats</dt>
              <dd>{(evidence.caveats || []).join(' ')}</dd>
            </div>
          </dl>
        </article>
      )}
    </div>
  );
}

function MarkdownLite({ text }) {
  return String(text || '')
    .split(/\n{2,}/)
    .filter((block) => block.trim())
    .map((block, index) => {
      const trimmed = block.trim();
      const heading = trimmed.match(/^\*\*(.+):?\*\*$/);
      if (heading) return <h4 key={index}>{heading[1].replace(/:$/, '')}</h4>;

      const lines = trimmed.split('\n').filter(Boolean);
      const isList = lines.every((line) => /^(\d+\.|-)\s+/.test(line.trim()));
      if (isList) {
        return (
          <ul key={index}>
            {lines.map((line, lineIndex) => (
              <li key={lineIndex}>
                <InlineMarkdown text={line.replace(/^(\d+\.|-)\s+/, '')} />
              </li>
            ))}
          </ul>
        );
      }

      return (
        <p key={index}>
          <InlineMarkdown text={trimmed} />
        </p>
      );
    });
}

function InlineMarkdown({ text }) {
  const parts = String(text || '').split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={index}>{part.slice(2, -2)}</strong>;
    }
    return <React.Fragment key={index}>{part}</React.Fragment>;
  });
}

function ChartRenderer({ chart }) {
  if (chart.chartType === 'table') {
    return (
      <article>
        <h3>{chart.title}</h3>
        <DataTable rows={chart.data || []} columns={Object.keys(chart.data?.[0] || {}).slice(0, 6)} />
      </article>
    );
  }

  if (chart.chartType === 'pie') {
    const total = (chart.data || []).reduce((sum, row) => sum + number(row[chart.y]), 0) || 1;
    return (
      <article>
        <h3>{chart.title}</h3>
        <div className="pie-list">
          {(chart.data || []).map((row) => (
            <div key={row[chart.x]}>
              <span>{row[chart.x]}</span>
              <b>{row[chart.y]}</b>
              <i style={{ width: `${(number(row[chart.y]) / total) * 100}%` }} />
            </div>
          ))}
        </div>
      </article>
    );
  }

  return (
    <article>
      <h3>{chart.title}</h3>
      <BarList rows={chart.data || []} labelKey={chart.y} valueKey={chart.x} metaKey={chart.color} />
    </article>
  );
}

createRoot(document.getElementById('root')).render(<App />);
