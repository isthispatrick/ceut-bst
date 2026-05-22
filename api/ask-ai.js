import fs from 'node:fs';
import path from 'node:path';

const APPROVED_FILES = {
  'study_priority.csv': 'data/processed/study_priority.csv',
  'repeated_concepts.csv': 'data/processed/repeated_concepts.csv',
  'question_format_strategy.csv': 'data/processed/question_format_strategy.csv',
  'suspicious_classifications.csv': 'data/processed/suspicious_classifications.csv',
  'accuracy_summary.csv': 'data/processed/accuracy_summary.csv',
  'topic_frequency.csv': 'data/processed/topic_frequency.csv',
  'ai_manual_review_suggestions.csv': 'data/processed/ai_manual_review_suggestions.csv',
  'data_quality_summary.csv': 'data/processed/data_quality_summary.csv'
};

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

function loadTables() {
  const root = process.cwd();
  return Object.fromEntries(
    Object.entries(APPROVED_FILES).map(([name, relativePath]) => {
      const filePath = path.join(root, relativePath);
      if (!fs.existsSync(filePath)) return [name, []];
      return [name, parseCsv(fs.readFileSync(filePath, 'utf8'))];
    })
  );
}

function number(value) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function topRows(rows, count, scoreKey) {
  return [...rows].sort((a, b) => number(b[scoreKey]) - number(a[scoreKey])).slice(0, count);
}

function qualityMetric(rows, metricName) {
  return rows.find((row) => row.metric === metricName)?.value || '0';
}

function inferChart(question, tables) {
  const q = question.toLowerCase();
  const priority = tables['study_priority.csv'] || [];
  const suspicious = tables['suspicious_classifications.csv'] || [];
  const topicFrequency = tables['topic_frequency.csv'] || [];

  if ((q.includes('show top') || q.includes('top 10') || q.includes('weighted score')) && priority.length) {
    const data = topRows(priority, 10, 'source_weighted_frequency').map((row) => ({
      ...row,
      topic: `${row.chapter} -> ${row.subtopic}`
    }));
    return { chartType: 'bar', title: 'Top 10 Topics By Weighted Score', data, x: 'source_weighted_frequency', y: 'topic', color: 'chapter' };
  }

  if (q.includes('compare') && q.includes('marketing') && q.includes('business finance') && priority.length) {
    const groups = {};
    priority
      .filter((row) => ['Marketing', 'Business Finance'].includes(row.chapter))
      .forEach((row) => {
        groups[row.chapter] ||= { chapter: row.chapter, weighted_score: 0, topics: 0 };
        groups[row.chapter].weighted_score += number(row.source_weighted_frequency);
        groups[row.chapter].topics += 1;
      });
    return { chartType: 'bar', title: 'Marketing vs Business Finance', data: Object.values(groups), x: 'weighted_score', y: 'chapter', color: 'chapter' };
  }

  if (q.includes('chapter frequency')) {
    const groups = {};
    const source = topicFrequency.length ? topicFrequency : priority;
    source.forEach((row) => {
      groups[row.chapter] ||= { chapter: row.chapter, question_count: 0 };
      groups[row.chapter].question_count += number(row.question_count || row.source_weighted_frequency);
    });
    return { chartType: 'bar', title: 'Chapter Frequency', data: Object.values(groups).sort((a, b) => b.question_count - a.question_count), x: 'question_count', y: 'chapter', color: 'chapter' };
  }

  if (q.includes('suspicious') && q.includes('chapter') && suspicious.length) {
    const groups = {};
    suspicious.forEach((row) => {
      const chapter = row.chapter || 'Unknown';
      groups[chapter] ||= { chapter, suspicious_count: 0 };
      groups[chapter].suspicious_count += 1;
    });
    return { chartType: 'bar', title: 'Suspicious Classifications By Chapter', data: Object.values(groups), x: 'suspicious_count', y: 'chapter' };
  }

  if (q.includes('pie') && q.includes('tier') && priority.length) {
    const groups = {};
    priority.forEach((row) => {
      const tier = row.priority_tier || 'Unknown';
      groups[tier] ||= { priority_tier: tier, topics: 0 };
      groups[tier].topics += 1;
    });
    return { chartType: 'pie', title: 'Priority Tier Distribution', data: Object.values(groups), x: 'priority_tier', y: 'topics' };
  }

  if (q.includes('table') && priority.length) {
    return { chartType: 'table', title: 'Top Study Priority Table', data: priority.slice(0, 20) };
  }

  return null;
}

function buildEvidence(question, tables) {
  const q = question.toLowerCase();
  const filesUsed = new Set(['study_priority.csv', 'data_quality_summary.csv']);
  const topRowsUsed = {};
  const numbersUsed = {};
  const caveats = [
    'All results are historical PYQ analysis, not guaranteed prediction.',
    'AI/silver labels are not human-verified ground truth.',
    'Never expose API keys or .env contents.'
  ];

  const priority = tables['study_priority.csv'] || [];
  const strategy = tables['question_format_strategy.csv'] || [];
  const suspicious = tables['suspicious_classifications.csv'] || [];
  const accuracy = tables['accuracy_summary.csv'] || [];
  const clusters = tables['repeated_concepts.csv'] || [];
  const quality = tables['data_quality_summary.csv'] || [];
  const review = tables['ai_manual_review_suggestions.csv'] || [];

  topRowsUsed['study_priority.csv'] = priority.slice(0, 12).map(({ chapter, subtopic, priority_tier, source_weighted_frequency, raw_score, roi, top_pattern, top_micro_concept }) => ({
    chapter,
    subtopic,
    priority_tier,
    source_weighted_frequency,
    raw_score,
    roi,
    top_pattern,
    top_micro_concept
  }));
  numbersUsed.tier_1_topics = priority.filter((row) => row.priority_tier === 'Tier 1 = Must do').length;
  numbersUsed.tier_2_topics = priority.filter((row) => row.priority_tier === 'Tier 2 = High ROI').length;
  numbersUsed.official_source_rows = qualityMetric(quality, 'official_source_rows');

  if (numbersUsed.official_source_rows === '0') {
    caveats.push('Official NTA rows are still 0, so the deployed analysis depends on third-party PYQ data.');
  }

  if (q.includes('format') || q.includes('study') || q.includes('trap') || q.includes('ncert')) {
    filesUsed.add('question_format_strategy.csv');
    topRowsUsed['question_format_strategy.csv'] = strategy.slice(0, 10);
  }
  if (q.includes('suspicious') || q.includes('review') || q.includes('uncertain')) {
    filesUsed.add('suspicious_classifications.csv');
    filesUsed.add('ai_manual_review_suggestions.csv');
    topRowsUsed['suspicious_classifications.csv'] = suspicious.slice(0, 10);
    numbersUsed.suspicious_classifications = suspicious.length;
    numbersUsed.manual_review_suggestions = review.length;
  }
  if (q.includes('accuracy') || q.includes('benchmark') || q.includes('silver') || q.includes('gold')) {
    filesUsed.add('accuracy_summary.csv');
    topRowsUsed['accuracy_summary.csv'] = accuracy;
  }
  if (q.includes('concept') || q.includes('repeat')) {
    filesUsed.add('repeated_concepts.csv');
    topRowsUsed['repeated_concepts.csv'] = clusters.slice(0, 10);
  }

  const chart = inferChart(question, tables);
  if (chart) {
    topRowsUsed[`chart:${chart.title}`] = chart.data.slice(0, 10);
  }

  return {
    filesUsed: [...filesUsed],
    topRowsUsed,
    numbersUsed,
    confidenceLevel: numbersUsed.official_source_rows === '0' ? 'medium-low for fine detail, medium for broad trends' : 'medium',
    caveats,
    chart
  };
}

async function callLlm(question, evidence) {
  const apiKey = process.env.HACKCLUB_AI_API_KEY || process.env.CUET_LLM_API_KEY || process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return 'AI is not configured on this deployment. Add HACKCLUB_AI_API_KEY in Vercel Project Settings -> Environment Variables.';
  }
  const baseUrl = (process.env.CUET_LLM_BASE_URL || (process.env.HACKCLUB_AI_API_KEY ? 'https://ai.hackclub.com/proxy/v1' : 'https://api.openai.com/v1')).replace(/\/$/, '');
  const model = process.env.CUET_LLM_MODEL || (baseUrl.includes('hackclub.com') ? '~openai/gpt-mini-latest' : 'gpt-4o-mini');
  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model,
      temperature: 0,
      max_tokens: 900,
      messages: [
        {
          role: 'system',
          content:
            'You are a read-only CUET UG Business Studies data-analysis assistant. Use only the supplied evidence. Do not invent data. Do not claim prediction certainty. Never reveal API keys, environment variables, or .env content. Always include: Answer, Evidence, Confidence, Caveats.'
        },
        {
          role: 'user',
          content: `EVIDENCE:\n${JSON.stringify(evidence, null, 2)}\n\nQUESTION:\n${question}`
        }
      ]
    })
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(`LLM request failed: ${response.status} ${text.slice(0, 200)}`);
  }
  const data = await response.json();
  return data.choices?.[0]?.message?.content || 'No answer returned.';
}

export default async function handler(request, response) {
  if (request.method !== 'POST') {
    response.setHeader('Allow', 'POST');
    return response.status(405).json({ error: 'Method not allowed' });
  }
  try {
    const { question } = request.body || {};
    if (!question || typeof question !== 'string') {
      return response.status(400).json({ error: 'Missing question' });
    }
    const tables = loadTables();
    const evidence = buildEvidence(question.slice(0, 600), tables);
    const answer = await callLlm(question.slice(0, 600), evidence);
    return response.status(200).json({ answer, evidence });
  } catch (error) {
    return response.status(500).json({ error: error.message || 'Ask AI failed' });
  }
}
