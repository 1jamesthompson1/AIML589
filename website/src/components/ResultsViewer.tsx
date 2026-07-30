import { useState, useMemo } from 'react';

interface EvalResult {
  question_id: string; question: string; sub_question: string; column_name: string;
  question_format: string; system_prompt_id: string; subpopulation: string;
  model_answer: string; model_reasoning: string; categories: string;
  model_distribution: string; true_distribution: string;
  kl_divergence: string; cross_entropy: string; expected_text: string;
}

interface ModelRun { config: { target: string; dataset: string; run_name: string }; results: EvalResult[]; }

export interface EvalData { models: Record<string, ModelRun[]>; }

interface Props { data: EvalData; }

function DistChart({ dist, labels, color, title }: { dist: number[]; labels: string[]; color: string; title: string }) {
  const maxVal = Math.max(...dist, 0.01);
  const containerH = 3.5;
  return (
    <div style={{ marginBottom: '0.75rem' }}>
      <p style={{ fontSize: '0.7rem', fontWeight: 600, color, marginBottom: '0.3rem' }}>{title}</p>
      <div style={{ display: 'flex', height: `${containerH}rem`, gap: '3px', alignItems: 'flex-end' }}>
        {dist.map((p, i) => {
          const h = Math.max((p / maxVal) * (containerH - 1.2), 0.15);
          return (
            <div key={i} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-end', height: '100%' }}>
              <span style={{ fontSize: '0.5rem', color: 'var(--color-text)', lineHeight: '0.7rem' }}>{(p * 100).toFixed(0)}%</span>
              <div style={{ width: '100%', height: `${h}rem`, background: color, borderRadius: '0.15rem 0.15rem 0 0' }} />
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', marginTop: '0.25rem' }}>
        {labels.map((l, i) => <span key={i} style={{ flex: 1, fontSize: '0.5rem', color: 'var(--color-muted)', textAlign: 'center', lineHeight: 1.15, wordBreak: 'break-word', padding: '0 2px' }}>{l}</span>)}
      </div>
    </div>
  );
}

function tryParseJSON(s: string): any { try { return JSON.parse(s); } catch { return null; } }

function findMatchingFT(ftModels: string[], baseName: string): string {
  return ftModels.filter((m) => m.includes(baseName))[0] || '';
}

function parseDist(s: string): number[] | null {
  const d = tryParseJSON(s);
  return Array.isArray(d) ? d : null;
}

function avgDist(rows: EvalResult[]): number[] | null {
  const parsed: number[][] = [];
  for (const r of rows) {
    const d = parseDist(r.model_distribution);
    if (d) parsed.push(d);
  }
  if (!parsed.length) return null;
  const avg = Array(parsed[0].length).fill(0);
  for (const d of parsed) for (let i = 0; i < d.length; i++) avg[i] += d[i];
  return avg.map((v) => v / parsed.length);
}

export default function ResultsViewer({ data }: Props) {
  const modelEntries = useMemo(() => Object.keys(data.models || {}), [data]);
  const baseModels = useMemo(() => modelEntries.filter((m) => !m.includes('-nz-wvs-')).sort((a, b) => {
    const ga = parseFloat(a.match(/[\d.]+/)?.[0] || '0');
    const gb = parseFloat(b.match(/[\d.]+/)?.[0] || '0');
    return gb - ga;
  }), [modelEntries]);
  const ftModels = useMemo(() => modelEntries.filter((m) => m.includes('-nz-wvs-')), [modelEntries]);

  const [selectedBase, setSelectedBase] = useState(baseModels[0] || '');
  const [selectedFT, setSelectedFT] = useState(() => findMatchingFT(ftModels, baseModels[0] || ''));
  const [selectedQId, setSelectedQId] = useState('');
  const [promptView, setPromptView] = useState('avg');
  const [search, setSearch] = useState('');

  const baseRun = useMemo(() => {
    const runs = data.models[selectedBase];
    return runs && runs.length > 0 ? runs[0] : null;
  }, [data, selectedBase]);

  const ftRun = useMemo(() => {
    if (!selectedFT) return null;
    const runs = data.models[selectedFT];
    return runs && runs.length > 0 ? runs[0] : null;
  }, [data, selectedFT]);

  const baseHfPath = baseRun?.config?.target || '';
  const ftHfPath = ftRun?.config?.target
    ? ftRun.config.target.includes('/') ? ftRun.config.target : `1jamesthompson1/${ftRun.config.target}`
    : '';

  const questionIds = useMemo(() => {
    if (!baseRun) return [];
    const seen = new Set<string>();
    const out: { qid: string; label: string }[] = [];
    for (const r of baseRun.results) {
      if (!seen.has(r.question_id)) {
        seen.add(r.question_id);
        const sub = r.sub_question || r.question.slice(0, 50);
        out.push({ qid: r.question_id, label: `Q${r.question_id}: ${sub}` });
      }
    }
    return out.sort((a, b) => Number(a.qid) - Number(b.qid));
  }, [baseRun]);

  const currentQText = useMemo(() => {
    if (!selectedQId || !baseRun) return '';
    const r = baseRun.results.find((x) => x.question_id === selectedQId);
    if (!r) return '';
    const sub = r.sub_question ? ` — ${r.sub_question}` : '';
    return `Q${selectedQId}: ${r.question}${sub}`;
  }, [selectedQId, baseRun]);

  const subpops = ['overall', 'cluster_0', 'cluster_1'];
  const subpopLabels: Record<string, string> = { overall: 'Overall', cluster_0: 'Cluster 0', cluster_1: 'Cluster 1' };
  const subpopColors: Record<string, string> = { overall: '#0f3460', cluster_0: '#0f3460', cluster_1: '#e94560' };

  const systemPrompts = useMemo(() => {
    if (!selectedQId || !baseRun) return ['avg'];
    const prompts = new Set<string>();
    for (const r of baseRun.results) {
      if (r.question_id === selectedQId) prompts.add(r.system_prompt_id);
    }
    return ['avg', ...Array.from(prompts).sort()];
  }, [selectedQId, baseRun]);

  const systemPromptLabels: Record<string, string> = { avg: 'Average across prompts' };
  for (const p of systemPrompts) {
    if (p !== 'avg') systemPromptLabels[p] = p;
  }

  function getDistributions(subpop: string) {
    if (!selectedQId || !baseRun) return null;
    const baseRows = baseRun.results.filter((r) => r.question_id === selectedQId && r.subpopulation === subpop);

    let ftRows: EvalResult[] = [];
    if (ftRun) {
      ftRows = ftRun.results.filter((r) => r.question_id === selectedQId && r.subpopulation === subpop);
    }

    let labels: string[] = [];
    const sample = baseRows[0] || ftRows[0];
    if (sample && sample.categories) {
      const cats = tryParseJSON(sample.categories);
      if (Array.isArray(cats) && cats.length > 0) labels = cats;
    }
    // Fallback labels from distribution length
    if (labels.length === 0 && sample && sample.model_distribution) {
      const d = parseDist(sample.model_distribution);
      if (d) labels = d.map((_, i) => `Opt ${i + 1}`);
    }

    let baseDist: number[] | null = null;
    let ftDist: number[] | null = null;
    let trueDist: number[] | null = null;

    if (promptView === 'avg') {
      baseDist = avgDist(baseRows);
      ftDist = avgDist(ftRows);
    } else {
      const b = baseRows.find((r) => r.system_prompt_id === promptView);
      const f = ftRows.find((r) => r.system_prompt_id === promptView);
      if (b) baseDist = parseDist(b.model_distribution);
      if (f) ftDist = parseDist(f.model_distribution);
    }

    const trueSrc = baseRows[0] || ftRows[0];
    if (trueSrc) trueDist = parseDist(trueSrc.true_distribution);

    return { baseDist, ftDist, trueDist, labels };
  }

  if (!modelEntries.length) {
    return <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--color-muted)' }}><p>No evaluation data found.</p></div>;
  }

  return (
    <div style={{ background: 'var(--color-surface)', borderTop: '1px solid var(--color-border)' }}>
      <div class="container" style={{ padding: '2rem 0' }}>
        <p style={{ fontSize: '0.8rem', color: 'var(--color-muted)', marginBottom: '1.5rem', lineHeight: 1.6 }}>
          Questions from the <a href="https://huggingface.co/datasets/1jamesthompson1/wvs-nz-value-alignment" target="_blank" rel="noopener noreferrer">WVS-NZ Value Alignment dataset ↗</a>.
          Base model: <a href={`https://huggingface.co/${baseHfPath}`} target="_blank" rel="noopener noreferrer">HF ↗</a>
          {ftHfPath && <> · Fine-tuned: <a href={`https://huggingface.co/${ftHfPath}`} target="_blank" rel="noopener noreferrer">HF ↗</a></>}
        </p>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', marginBottom: '2rem' }}>
          <div>
            <label style={label}>Base model</label>
            <select value={selectedBase} onChange={(e) => {
              const newBase = e.target.value;
              setSelectedBase(newBase);
              setSelectedFT(findMatchingFT(ftModels, newBase));
              setSelectedQId('');
            }} style={select}>
              {baseModels.map((m) => <option key={m}>{m}</option>)}
            </select>
          </div>
          <div>
            <label style={label}>Fine-tuned model</label>
            <select value={selectedFT} onChange={(e) => setSelectedFT(e.target.value)} style={select}>
              <option value="">(none)</option>
              {ftModels.map((m) => <option key={m}>{m}</option>)}
            </select>
          </div>
          <div>
            <label style={label}>Response view</label>
            <select value={promptView} onChange={(e) => setPromptView(e.target.value)} style={select}>
              {systemPrompts.map((p) => <option key={p} value={p}>{systemPromptLabels[p]}</option>)}
            </select>
          </div>
        </div>

        {questionIds.length > 0 ? (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 3fr', gap: '2rem' }}>
            <div>
              <label style={label}>Question</label>
              <input type="text" placeholder="Search questions..." value={search} onChange={(e) => setSearch(e.target.value)} style={{
                ...select, marginBottom: '0.5rem', width: '100%', boxSizing: 'border-box',
              }} />
              <div style={{ maxHeight: '55vh', overflowY: 'auto', border: '1px solid var(--color-border)', borderRadius: '0.5rem' }}>
                {questionIds.filter(({ label }) => !search || label.toLowerCase().includes(search.toLowerCase())).map(({ qid, label }) => (
                  <button key={qid} onClick={() => setSelectedQId(qid)} style={{
                    display: 'block', width: '100%', textAlign: 'left', padding: '0.75rem', border: 'none',
                    borderBottom: '1px solid var(--color-border)',
                    background: qid === selectedQId ? 'var(--color-primary)' : 'transparent',
                    color: qid === selectedQId ? 'white' : 'var(--color-text)', cursor: 'pointer', fontSize: '0.8rem',
                  }}>
                    <strong>{label}</strong>
                  </button>
                ))}
              </div>
            </div>

            <div>
              {currentQText && <h3 style={{ marginBottom: '1rem', wordBreak: 'break-word' }}>{currentQText}</h3>}

              {subpops.map((subpop) => {
                const d = getDistributions(subpop);
                if (!d) return null;
                const hasAny = d.baseDist || d.ftDist || d.trueDist;
                if (!hasAny) return <div key={subpop} style={{ marginBottom: '1rem', padding: '1rem', background: 'var(--color-bg)', borderRadius: '0.75rem', border: '1px solid var(--color-border)' }}>
                  <p style={{ fontSize: '0.8rem', fontWeight: 700, color: subpopColors[subpop] }}>{subpopLabels[subpop]}</p>
                  <p style={{ fontSize: '0.75rem', color: 'var(--color-muted)', marginTop: '0.5rem' }}>No data for this subpopulation.</p>
                </div>;

                return (
                  <div key={subpop} style={{ marginBottom: '2rem', padding: '1rem', background: 'var(--color-bg)', borderRadius: '0.75rem', border: '1px solid var(--color-border)' }}>
                    <p style={{ fontSize: '0.8rem', fontWeight: 700, color: subpopColors[subpop], marginBottom: '0.75rem' }}>{subpopLabels[subpop]}</p>
                    {d.baseDist && d.labels.length > 0 && <DistChart dist={d.baseDist} labels={d.labels} color="#6b7280" title="Base model" />}
                    {d.ftDist && d.labels.length > 0 && <DistChart dist={d.ftDist} labels={d.labels} color="#e94560" title="Fine-tuned model" />}
                    {d.trueDist && d.labels.length > 0 && <DistChart dist={d.trueDist} labels={d.labels} color="#16a34a" title="Ground truth (cluster)" />}
                    {d.labels.length === 0 && <p style={{ fontSize: '0.75rem', color: 'var(--color-muted)' }}>No label data.</p>}
                  </div>
                );
              })}
            </div>
          </div>
        ) : (
          <p style={{ color: 'var(--color-muted)' }}>Select a model to view results.</p>
        )}
      </div>
    </div>
  );
}

const label: React.CSSProperties = { display: 'block', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-muted)', marginBottom: '0.25rem' };
const select: React.CSSProperties = { padding: '0.5rem 0.75rem', borderRadius: '0.5rem', border: '1px solid var(--color-border)', background: 'white', fontSize: '0.875rem', minWidth: '180px', fontFamily: 'inherit', color: 'var(--color-text)' };
