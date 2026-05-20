import React, { useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { AlertTriangle, BarChart3, BookOpen, Download, FileText, Target, TrendingUp } from 'lucide-react';
import studyPriorityCsv from '../data/processed/study_priority.csv?raw';
import dataQualityCsv from '../data/processed/data_quality_summary.csv?raw';
import strategyCsv from '../data/processed/question_format_strategy.csv?raw';
import accuracyCsv from '../data/processed/accuracy_summary.csv?raw';
import clustersCsv from '../data/processed/micro_concept_clusters.csv?raw';
import studyPackUrl from '../reports/cuet_bst_final_study_pack.pdf?url';
import analysisReportUrl from '../reports/cuet_bst_analysis_report.pdf?url';
import './styles.css';

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    const next = text[i + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      i += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === ',' && !quoted) {
      row.push(cell);
      cell = '';
    } else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && next === '\n') i += 1;
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

const studyPriority = parseCsv(studyPriorityCsv);
const dataQuality = parseCsv(dataQualityCsv);
const strategy = parseCsv(strategyCsv);
const accuracy = parseCsv(accuracyCsv);
const clusters = parseCsv(clustersCsv);

function numeric(value) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function metric(name, fallback = '0') {
  return dataQuality.find((row) => row.metric === name)?.value || fallback;
}

function topRows(rows, count, scoreKey) {
  return [...rows].sort((a, b) => numeric(b[scoreKey]) - numeric(a[scoreKey])).slice(0, count);
}

function App() {
  const [query, setQuery] = useState('');
  const filteredPriority = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return studyPriority;
    return studyPriority.filter((row) => `${row.chapter} ${row.subtopic} ${row.priority_tier}`.toLowerCase().includes(needle));
  }, [query]);

  const tier1 = studyPriority.filter((row) => row.priority_tier === 'Tier 1 = Must do');
  const highRoi = topRows(studyPriority, 10, 'roi');
  const topClusters = topRows(clusters, 12, 'weighted_score');
  const silverStatus = accuracy.filter((row) => row.metric.startsWith('pipeline_vs_silver'));
  const humanStatus = accuracy.find((row) => row.metric === 'pipeline_vs_human_gold_status')?.value;

  return (
    <main className="shell">
      <section className="hero">
        <div>
          <p className="eyebrow">CUET UG Business Studies 305</p>
          <h1>Exam Intelligence Dashboard</h1>
          <p className="lede">
            Static Vercel view of the latest PYQ analysis. The full Streamlit command center remains available locally for
            AI chat, review workflows, and pipeline reruns.
          </p>
          <div className="actions">
            <a href={studyPackUrl} className="button primary">
              <Download size={18} /> Final study pack
            </a>
            <a href={analysisReportUrl} className="button">
              <FileText size={18} /> Analysis report
            </a>
          </div>
        </div>
        <div className="hero-metrics">
          <Metric label="Parsed rows" value={metric('total_parsed_rows')} />
          <Metric label="Unique questions" value={metric('unique_questions_after_dedupe')} />
          <Metric label="Duplicate rate" value={`${metric('duplicate_rate_percent')}%`} />
          <Metric label="Official rows" value={metric('official_source_rows')} warning />
        </div>
      </section>

      <section className="notice">
        <AlertTriangle size={20} />
        <span>
          Historical PYQ analysis only. Official NTA question rows are currently {metric('official_source_rows')}; rankings
          should guide study allocation, not be treated as guaranteed prediction.
        </span>
      </section>

      <section className="grid two">
        <Panel title="Tier 1: Must Do" icon={<Target size={20} />}>
          <TopicTable rows={tier1.slice(0, 12)} />
        </Panel>
        <Panel title="Highest ROI Topics" icon={<TrendingUp size={20} />}>
          <TopicTable rows={highRoi} score="roi" />
        </Panel>
      </section>

      <section className="grid two">
        <Panel title="Most Repeated Micro-Concepts" icon={<BarChart3 size={20} />}>
          <BarList rows={topClusters} labelKey="micro_concept" valueKey="weighted_score" metaKey="chapter" />
        </Panel>
        <Panel title="How To Study High-Priority Formats" icon={<BookOpen size={20} />}>
          <div className="strategy-list">
            {strategy.slice(0, 8).map((row) => (
              <article key={`${row.chapter}-${row.subtopic}`}>
                <strong>{row.chapter}{' -> '}{row.subtopic}</strong>
                <span>{row.dominant_question_pattern}</span>
                <p>{row.how_to_study_it}</p>
              </article>
            ))}
          </div>
        </Panel>
      </section>

      <section className="panel">
        <div className="panel-title">
          <FileText size={20} />
          <h2>Search Study Priority</h2>
        </div>
        <input
          className="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search chapter, subtopic, or tier"
        />
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Chapter</th>
                <th>Subtopic</th>
                <th>Tier</th>
                <th>Raw</th>
                <th>ROI</th>
                <th>Pattern</th>
              </tr>
            </thead>
            <tbody>
              {filteredPriority.slice(0, 40).map((row) => (
                <tr key={`${row.chapter}-${row.subtopic}`}>
                  <td>{row.chapter}</td>
                  <td>{row.subtopic}</td>
                  <td>{row.priority_tier}</td>
                  <td>{row.raw_score}</td>
                  <td>{row.roi}</td>
                  <td>{row.top_pattern}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="grid two">
        <Panel title="Silver Benchmark Accuracy" icon={<BarChart3 size={20} />}>
          <dl className="accuracy">
            {silverStatus.map((row) => (
              <div key={row.metric}>
                <dt>{row.metric.replace('pipeline_vs_silver_', '').replaceAll('_', ' ')}</dt>
                <dd>{row.value}</dd>
              </div>
            ))}
          </dl>
        </Panel>
        <Panel title="Human Gold Benchmark" icon={<AlertTriangle size={20} />}>
          <p className="muted">{humanStatus || 'Human-verified accuracy unavailable yet.'}</p>
          <p className="muted">
            Use the local Streamlit dashboard to review the 100-question human benchmark queue and save true gold labels.
          </p>
        </Panel>
      </section>
    </main>
  );
}

function Metric({ label, value, warning = false }) {
  return (
    <div className={warning ? 'metric warning' : 'metric'}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
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

function TopicTable({ rows, score = 'raw_score' }) {
  return (
    <div className="topic-list">
      {rows.map((row) => (
        <article key={`${row.chapter}-${row.subtopic}`}>
          <div>
            <strong>{row.subtopic}</strong>
            <span>{row.chapter}</span>
          </div>
          <b>{row[score]}</b>
        </article>
      ))}
    </div>
  );
}

function BarList({ rows, labelKey, valueKey, metaKey }) {
  const max = Math.max(...rows.map((row) => numeric(row[valueKey])), 1);
  return (
    <div className="bar-list">
      {rows.map((row) => {
        const value = numeric(row[valueKey]);
        return (
          <article key={`${row[labelKey]}-${row[metaKey]}`}>
            <div className="bar-label">
              <strong>{row[labelKey]}</strong>
              <span>{row[metaKey]}</span>
              <b>{value.toFixed(1)}</b>
            </div>
            <div className="bar-track">
              <span style={{ width: `${Math.max(6, (value / max) * 100)}%` }} />
            </div>
          </article>
        );
      })}
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App />);
