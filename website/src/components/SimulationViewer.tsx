import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { cachedFetchJson, cachedFetchText } from '../lib/cachedFetch';

/* Behavioural simulation runs viewer.
 *
 * Data lives in the project's public HF storage bucket (see
 * code/behavioural-simulations/export_results.py): a manifest (index.json)
 * lists every run; each run directory holds transcript.json, judge.json and
 * self_review.txt. This component fetches the manifest from
 * the bucket and lets visitors compare two runs side by side - across two
 * different models, or across different runs (versions) of the same model
 * and scenario.
 */

const DEFAULT_BUCKET = '1jamesthompson1/wvs-nz-value-alignment-evals';
const DRY_RUN_MODEL = 'mockllm/dry-run';

const urlParams = typeof window !== 'undefined'
  ? new URLSearchParams(window.location.search)
  : new URLSearchParams();

// Data source default: the dev server (npm run dev) reads the **local**
// artifacts mirror; production builds always read the HF bucket. URL param
// overrides the default either way: `?local=1` forces local (needs
// website/public/bs -> artifacts/bs from `make website-local-data`),
// `?local=0` forces the bucket.
const IS_DEV = import.meta.env.DEV;
const localParam = urlParams.get('local');
const LOCAL_MODE = localParam === '1' || (localParam !== '0' && IS_DEV);
const LOCAL_BS_BASE = '/bs/runs/';

const MANIFEST_URL = urlParams.get('manifest')
  ?? (LOCAL_MODE
    ? `${LOCAL_BS_BASE}index.json`
    : `https://huggingface.co/buckets/${DEFAULT_BUCKET}/resolve/bs/runs/index.json`);

interface RunInfo {
  run_id?: string;
  model: string;
  profile_id: string;
  situation_id: string;
  type: string;
  status: string;
  version: number;
  created: string;
  run_dir: string;
  judge_score: number | null;
  judge_verdict: string | null;
}

interface SituationInfo {
  profile_id: string;
  profile_name: string;
  situation_id: string;
  name: string;
  type: string;
}

interface Manifest {
  schema: string;
  base_url: string;
  bucket: string;
  profiles: Record<string, { name: string }>;
  situations: SituationInfo[];
  models: string[];
  runs: RunInfo[];
}

interface ToolCall { id?: string; function: string; arguments: unknown; }
interface Msg {
  index: number;
  role: string;
  content?: string;
  reasoning?: string;
  tool_calls?: ToolCall[];
  function?: string;
  tool_call_id?: string;
}
interface Transcript { schema: string; run: { model: string; created: string; situation_name?: string }; messages: Msg[]; }
interface JudgeDoc { score: number | null; explanation: string; structured?: any; }

interface RunData {
  transcript: Transcript | null;
  judge: JudgeDoc | null;
  selfReview: string;
  error?: string;
}

async function loadRun(
  baseUrl: string,
  run: RunInfo,
  onTranscript?: (partial: RunData) => void,
): Promise<RunData> {
  const base = `${baseUrl}${run.run_dir}/`;
  // Fetch the transcript first and surface it straight away (each bucket
  // fetch has ~1s of redirect/CDN overhead, so the review files load in the
  // background rather than blocking the transcript).
  const transcript = await cachedFetchJson<Transcript>(`${base}transcript.json`);
  onTranscript?.({ transcript, judge: null, selfReview: '' });
  const [judge, selfReview] = await Promise.all([
    cachedFetchJson<JudgeDoc>(`${base}judge.json`).catch(() => null),
    cachedFetchText(`${base}self_review.txt`).catch(() => ''),
  ]);
  return { transcript, judge, selfReview };
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toUTCString().replace(/ GMT$/, '');
}

function shortModel(model: string): string {
  return model.replace(/^openrouter\//, '');
}

const label: React.CSSProperties = { display: 'block', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-muted)', marginBottom: '0.25rem' };
const select: React.CSSProperties = { padding: '0.5rem 0.75rem', borderRadius: '0.5rem', border: '1px solid var(--color-border)', background: 'white', fontSize: '0.875rem', minWidth: '200px', fontFamily: 'inherit', color: 'var(--color-text)' };
const card: React.CSSProperties = { background: 'var(--color-bg)', border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '0.75rem 1rem' };

/* Render a block of summary text, splitting out any <thinking>…</thinking>
 * segments behind a collapsed "Reasoning" disclosure (hidden by default) so
 * the model's private reasoning does not clutter the visible summary. Text
 * outside <thinking> tags is shown inline as normal. */
function ThinkingAwareText({ text, mono }: { text: string; mono?: boolean }) {
  const parts = text.split(/<thinking>|<\/thinking>/i);
  if (parts.length <= 1) {
    return <pre style={{ whiteSpace: 'pre-wrap', fontFamily: mono ? 'var(--font-mono)' : 'inherit', fontSize: '0.85rem', lineHeight: 1.55, margin: 0 }}>{text}</pre>;
  }
  // Interspersed <thinking> blocks: the first element is always a "normal" part
  // (possibly empty), then each subsequent odd index is a thinking segment.
  const divs = parts.map((chunk, i) => {
    const trimmed = chunk.trim();
    if (i % 2 === 1) {
      return (
        <details key={i} style={{ margin: '0.25rem 0' }}>
          <summary style={{ fontSize: '0.72rem', color: 'var(--color-muted)', cursor: 'pointer', fontStyle: 'italic' }}>Reasoning expansion</summary>
          <pre style={{ whiteSpace: 'pre-wrap', fontFamily: mono ? 'var(--font-mono)' : 'inherit', fontSize: '0.8rem', lineHeight: 1.5, marginTop: '0.3rem', fontStyle: 'italic', color: 'var(--color-muted)' }}>{trimmed || chunk}</pre>
        </details>
      );
    }
    return trimmed ? <pre key={i} style={{ whiteSpace: 'pre-wrap', fontFamily: mono ? 'var(--font-mono)' : 'inherit', fontSize: '0.85rem', lineHeight: 1.55, margin: 0 }}>{trimmed}</pre> : null;
  });
  return <div>{divs}</div>;
}

function scoreColor(score: number | null | undefined): string {
  if (score == null) return 'var(--color-muted)';
  if (score >= 4) return '#16a34a';
  if (score >= 3) return '#ca8a04';
  return '#dc2626';
}

function ScoreBadge({ run }: { run: RunInfo }) {
  const color = scoreColor(run.judge_score);
  return (
    <span style={{ fontSize: '0.8rem', fontWeight: 700, color, background: `${color}1a`, padding: '0.15rem 0.5rem', borderRadius: '0.5rem', whiteSpace: 'nowrap' }}>
      {run.judge_score != null ? `${run.judge_score}/5` : 'n/a'}
      {run.judge_verdict ? ` · ${run.judge_verdict}` : ''}
    </span>
  );
}

function MessageView({ msg, highlighted, onSelect, showReasoning }: {
  msg: Msg;
  highlighted: boolean;
  onSelect: (index: number) => void;
  showReasoning: boolean;
}) {
  const isClientMsg = msg.role === 'tool' && msg.function === 'send_client_message';
  const clientText = isClientMsg && msg.content ? msg.content.replace(/^Reply:\s*/, '') : null;

  let style: React.CSSProperties = { marginBottom: '0.6rem', fontSize: '0.85rem', lineHeight: 1.55, cursor: 'pointer', transition: 'box-shadow 0.15s' };
  if (highlighted) style = { ...style, boxShadow: '0 0 0 3px var(--color-primary)' };

  const chip = (text: string, bg: string, fg: string): React.CSSProperties => ({
    display: 'inline-block', fontSize: '0.65rem', fontWeight: 600, textTransform: 'uppercase',
    letterSpacing: '0.04em', background: bg, color: fg, padding: '0.1rem 0.45rem', borderRadius: '0.35rem', marginRight: '0.4rem',
  });

  return (
    <div id={`msg-${msg.index}`} onClick={() => onSelect(msg.index)} style={style}>
      {msg.role === 'system' && (
        <div style={card}>
          <details>
            <summary style={{ cursor: 'pointer', fontSize: '0.72rem', color: 'var(--color-muted)', fontStyle: 'italic' }}>
              <span style={chip('system', '#e2e8f0', '#475569')}>System</span> deployment system prompt <span style={{ textDecoration: 'underline' }}>(expand)</span>
            </summary>
            <pre style={{ whiteSpace: 'pre-wrap', margin: '0.5rem 0 0', fontSize: '0.75rem', color: 'var(--color-muted)' }}>{msg.content}</pre>
          </details>
        </div>
      )}

      {msg.role === 'user' && (
        <div style={{ ...card, background: '#eef2ff', borderColor: '#c7d2fe' }}>
          <span style={chip('work item', '#e0e7ff', '#3730a3')}>Work item</span>
          <div style={{ marginTop: '0.3rem' }}>{msg.content}</div>
        </div>
      )}

      {msg.role === 'assistant' && (
        <div>
          {msg.reasoning && showReasoning && (
            <details style={{ marginBottom: '0.3rem' }}>
              <summary style={{ fontSize: '0.7rem', color: 'var(--color-muted)', cursor: 'pointer' }}>Reasoning</summary>
              <pre style={{ whiteSpace: 'pre-wrap', background: 'var(--color-bg)', border: '1px solid var(--color-border)', borderRadius: '0.4rem', padding: '0.5rem', fontSize: '0.75rem', marginTop: '0.25rem', fontStyle: 'italic', color: 'var(--color-muted)' }}>{msg.reasoning}</pre>
            </details>
          )}
          {msg.content && <div style={card}>{msg.content}</div>}
          {(msg.tool_calls || []).map((tc, i) => (
            <details key={i} style={{ marginTop: '0.3rem' }}>
              <summary style={{ fontSize: '0.78rem', cursor: 'pointer', fontFamily: 'monospace', color: 'var(--color-primary)' }}>
                ⚙ {tc.function}({JSON.stringify(tc.arguments)?.slice(0, 120)}{JSON.stringify(tc.arguments)?.length > 120 ? '…' : ''})
              </summary>
              <pre style={{ whiteSpace: 'pre-wrap', background: '#0b1220', color: '#e2e8f0', borderRadius: '0.4rem', padding: '0.5rem', fontSize: '0.75rem', marginTop: '0.25rem' }}>
                {JSON.stringify(tc.arguments, null, 2)}
              </pre>
            </details>
          ))}
        </div>
      )}

      {msg.role === 'tool' && (
        isClientMsg && clientText ? (
          <div style={{ ...card, background: '#f0fdf4', borderColor: '#bbf7d0', marginLeft: '1rem' }}>
            <span style={chip('client', '#dcfce7', '#15803d')}>Client</span>
            <div style={{ marginTop: '0.3rem', whiteSpace: 'pre-wrap' }}>{clientText}</div>
          </div>
        ) : (
          <details style={{ marginLeft: '1rem' }}>
            <summary style={{ fontSize: '0.72rem', cursor: 'pointer', fontFamily: 'monospace', color: 'var(--color-muted)' }}>
              ↩ {msg.function}
            </summary>
            <pre style={{ whiteSpace: 'pre-wrap', background: '#f8fafc', border: '1px solid var(--color-border)', borderRadius: '0.4rem', padding: '0.5rem', fontSize: '0.75rem', marginTop: '0.25rem', maxHeight: '18rem', overflow: 'auto' }}>
              {msg.content}
            </pre>
          </details>
        )
      )}
    </div>
  );
}

function RunPanel({ title, run, data, highlightIndex, onSelect, showReasoning, active, onRetry }: {
  title: string;
  run: RunInfo | null;
  data: RunData | null;
  highlightIndex: number | null;
  onSelect: (index: number) => void;
  showReasoning: boolean;
  active: boolean;
  onRetry: () => void;
}) {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (active && highlightIndex != null && ref.current) {
      const el = ref.current.querySelector(`#msg-${highlightIndex}`);
      el?.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }, [highlightIndex, active, data]);

  return (
    <div ref={ref} style={{ minWidth: 0 }}>
      <div style={{ ...card, marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <div>
            <p style={{ fontSize: '0.7rem', color: 'var(--color-muted)', marginBottom: '0.15rem' }}>{title}</p>
            <p style={{ fontSize: '0.95rem', fontWeight: 700 }}>{run ? shortModel(run.model) : '—'}</p>
          </div>
          {run && <ScoreBadge run={run} />}
        </div>
        {run && data?.transcript && (
          <p style={{ fontSize: '0.72rem', color: 'var(--color-muted)', marginTop: '0.4rem' }}>
            {data.transcript.run.situation_name ?? run.situation_id} · run {run.version} · {fmtDate(data.transcript.run.created)}
            {run.run_id ? ` · ${run.run_id}` : ''}
          </p>
        )}
      </div>

      {data?.error && (
        <div style={{ ...card, borderColor: '#fca5a5', color: '#b91c1c', marginBottom: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '1rem', flexWrap: 'wrap' }}>
          <span>Failed to load run: {data.error}</span>
          <button onClick={onRetry} style={{ padding: '0.4rem 0.9rem', borderRadius: '0.5rem', border: '1px solid #fca5a5', background: 'white', cursor: 'pointer', fontSize: '0.8rem', fontFamily: 'inherit', color: '#b91c1c' }}>
            Retry
          </button>
        </div>
      )}

      {!data && <div style={{ color: 'var(--color-muted)', fontSize: '0.85rem', padding: '1rem 0' }}>Loading run…</div>}
      {data && (
        <>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.9rem', marginBottom: '1.25rem' }}>
            {data.judge && (() => {
              const s = data.judge.structured;
              const items: { label: string; reason: string; answer: string }[] = [];
              for (const [key, v] of Object.entries(s?.profile_assessment ?? {})) {
                const vv = v as any;
                items.push({ label: vv?.label ?? key, reason: vv?.reason ?? '', answer: String(vv?.answer ?? vv?.choice ?? vv?.score ?? '') });
              }
              const decisions: { question: string; comment: string; answer: string }[] = [];
              for (const [key, v] of Object.entries(s?.key_decisions ?? {})) {
                const vv = v as any;
                decisions.push({ question: vv?.question ?? key, comment: vv?.comment ?? '', answer: String(vv?.decision ?? vv?.score ?? '') });
              }
              return (
                <div style={{ background: '#f0f4ff', border: '2px solid #6366f1', borderRadius: '0.75rem', padding: '0.9rem 1rem', boxShadow: '0 2px 8px rgba(99,102,241,0.12)' }}>
                  <p style={{ fontSize: '0.7rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#4f46e5', marginBottom: '0.5rem' }}>⚖ Judge review</p>
                  <p style={{ fontSize: '0.88rem', lineHeight: 1.55 }}>{s?.overall?.summary}</p>
                  {items.map((it, i) => (
                    <p key={i} style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
                      <strong>{it.label}:</strong> {it.answer}
                      {it.reason ? <span style={{ color: 'var(--color-muted)' }}> — {it.reason}</span> : null}
                    </p>
                  ))}
                  {decisions.length > 0 && (
                    <div style={{ borderTop: '1px solid #c7d2fe', marginTop: '0.6rem', paddingTop: '0.5rem' }}>
                      {decisions.map((d, i) => (
                        <p key={i} style={{ fontSize: '0.8rem', marginTop: i ? '0.4rem' : 0 }}>
                          <strong>{d.question}</strong> → {d.answer}
                          {d.comment ? <span style={{ color: 'var(--color-muted)' }}> — {d.comment}</span> : null}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              );
            })()}

            {data.selfReview && (
              <div style={{ background: '#fffbeb', border: '2px solid #f59e0b', borderRadius: '0.75rem', padding: '0.9rem 1rem', boxShadow: '0 2px 8px rgba(245,158,11,0.12)' }}>
                <p style={{ fontSize: '0.7rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#b45309', marginBottom: '0.75rem' }}>🧭 Model summary</p>
                <div><ThinkingAwareText text={data.selfReview} /></div>
              </div>
            )}
          </div>

          <div style={{ borderTop: '2px solid var(--color-border)', paddingTop: '0.75rem' }}>
            <p style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-muted)', marginBottom: '0.6rem' }}>
              💬 Transcript <span style={{ fontWeight: 400, textTransform: 'none' }}>(full conversation)</span>
            </p>
            {(data.transcript?.messages ?? []).map((m) => (
              <MessageView key={m.index} msg={m} highlighted={highlightIndex === m.index} onSelect={onSelect} showReasoning={showReasoning} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

/* ---------------------------------------------------------------------------
 * Comparisons (build_comparisons.py output): one JSON file per trajectory
 * pair, indexed by `bs/comparisons/index.csv`
 * (comparison_id -> agent run ids + models).
 * ------------------------------------------------------------------------ */

interface ComparisonRow {
  comparison_id: string;
  scenario: string;
  model_pair: string;
  agent1_run_id: string;
  agent1_model: string;
  agent2_run_id: string;
  agent2_model: string;
}
interface ComparisonAudit { agent: string; model: string; raw: string; }
interface ComparisonDoc {
  schema: string;
  comparison_id: string;
  scenario: {
    profile_id: string;
    situation_id: string;
    profile_summary?: string;
    situation_summary?: string;
  };
  summary: { model: string; reasoning_effort?: string; text: string };
  audits: Record<string, ComparisonAudit>;
}

function splitCsvLine(line: string): string[] {
  const fields: string[] = [];
  let cur = '';
  let quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (quoted) {
      if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') quoted = true;
    } else if (ch === '"') quoted = true;
    else if (ch === ',') { fields.push(cur); cur = ''; }
    else cur += ch;
  }
  fields.push(cur);
  return fields;
}

function parseComparisonCsv(text: string): ComparisonRow[] {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).map((line) => {
    const cells = splitCsvLine(line);
    const row = {} as ComparisonRow;
    headers.forEach((h, i) => { (row as unknown as Record<string, string>)[h] = cells[i] ?? ''; });
    return row;
  });
}

function comparisonsBase(): string {
  if (LOCAL_MODE) return '/bs/comparisons/';
  // The manifest sits in `bs/runs/index.json`; comparisons live next to it.
  return MANIFEST_URL.replace(/bs\/runs\/?.*$/, 'bs/comparisons/');
}

const VIEW_MODES = [
  { id: 'single' as const, label: 'Single run' },
  { id: 'compare' as const, label: 'Comparison' },
];

const modeButton = (active: boolean): React.CSSProperties => ({
  padding: '0.45rem 1rem', borderRadius: '0.5rem',
  border: `1px solid ${active ? 'var(--color-primary)' : 'var(--color-border)'}`,
  background: active ? 'var(--color-primary)' : 'white',
  color: active ? 'white' : 'var(--color-text)',
  cursor: 'pointer', fontSize: '0.85rem', fontFamily: 'inherit', fontWeight: active ? 700 : 400,
});

function findComparison(rows: ComparisonRow[], a: string, b: string): ComparisonRow | undefined {
  return rows.find(
    (r) =>
      (r.agent1_run_id === a && r.agent2_run_id === b) ||
      (r.agent1_run_id === b && r.agent2_run_id === a),
  );
}

/* The round-by-round difference summary written by the summary model, with
 * each generating model's audit of that summary, shown above the two
 * side-by-side trajectories. */
function ComparisonPanel({ comparison, row }: { comparison: ComparisonDoc | null; row: ComparisonRow | null }) {
  if (!row) return null;
  return (
    <div style={{ ...card, marginBottom: '1.5rem', background: 'var(--color-surface)' }}>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.75rem', alignItems: 'center', marginBottom: '0.9rem' }}>
        <p style={{ fontSize: '0.7rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-muted)', margin: 0 }}>
          ⚖ Cross-model comparison
        </p>
        {comparison && (
          <span style={{ fontSize: '0.75rem', color: 'var(--color-muted)' }}>
            summary by {comparison.summary.model.replace(/^openrouter\//, '')}
          </span>
        )}
        {comparison?.scenario?.situation_summary && (
          <p style={{ fontSize: '0.8rem', color: 'var(--color-muted)', margin: 0, flexBasis: '100%' }}>
            {comparison.scenario.situation_summary}
          </p>
        )}
      </div>
      {!comparison && <div style={{ color: 'var(--color-muted)', fontSize: '0.85rem' }}>Loading comparison…</div>}
      {comparison && (
        <>
          <div style={{ ...card, background: '#eef2ff', borderColor: '#c7d2fe', marginBottom: '0.9rem' }}>
            <ThinkingAwareText text={comparison.summary.text} />
          </div>
          {Object.entries(comparison.audits ?? {}).map(([runId, audit]) => (
            audit.raw && (
              <div key={runId} style={{ ...card, marginBottom: '0.6rem' }}>
                <p style={{ fontSize: '0.7rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--color-muted)', marginBottom: '0.4rem' }}>
                  🔎 Audit ({audit.agent} · {audit.model.replace(/^openrouter\//, '')})
                </p>
                <ThinkingAwareText text={audit.raw} />
              </div>
            )
          ))}
        </>
      )}
    </div>
  );
}

export default function SimulationViewer() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [compRows, setCompRows] = useState<ComparisonRow[] | null>(null);
  const [loadError, setLoadError] = useState<string>('');
  const [situationKey, setSituationKey] = useState<string>('');
  const [mode, setMode] = useState<'single' | 'compare'>('single');

  // Single mode: the run to watch (by globally unique run_id).
  const [singleRunId, setSingleRunId] = useState<string | null>(null);
  // Compare mode: the comparison row resolved from deep-link params.
  const [deepLinkPair, setDeepLinkPair] = useState<{ a: string | null; b: string | null }>({ a: null, b: null });
  // Explicit comparison pick (the selection dropdown above the panel).
  const [comparisonPick, setComparisonPick] = useState<string | null>(null);

  // Loaded documents.
  const [dataSingle, setDataSingle] = useState<RunData | null>(null);
  const [dataA, setDataA] = useState<RunData | null>(null);
  const [dataB, setDataB] = useState<RunData | null>(null);
  const [comparison, setComparison] = useState<ComparisonDoc | null>(null);

  const [highlightIndex, setHighlightIndex] = useState<number | null>(null);
  const [highlightOrigin, setHighlightOrigin] = useState<'A' | 'B' | null>(null);
  const [showReasoning, setShowReasoning] = useState(true);
  const [showDryRun, setShowDryRun] = useState(false);

  const cache = useRef<Record<string, RunData>>({});
  const compCache = useRef<Record<string, ComparisonDoc>>({});
  const retryTickRef = useRef(0);
  const [, forceRerender] = useState(0);
  const bumpRetry = useCallback(() => {
    retryTickRef.current += 1;
    forceRerender(retryTickRef.current);
  }, []);

  /* ------------------------------- data ------------------------------- */

  useEffect(() => {
    console.info(`[SimulationViewer] fetching manifest: ${MANIFEST_URL}`);
    // The manifest is the gatekeeper for new runs, so always re-fetch it
    // (0 TTL) rather than serving a stale copy from the localStorage cache.
    cachedFetchJson<Manifest>(MANIFEST_URL, 0)
      .then((m) => {
        console.info(`[SimulationViewer] manifest loaded: ${m.runs?.length ?? 0} runs, ${m.situations?.length ?? 0} situations`);
        setManifest(m);
      })
      .catch((e) => {
        console.error('[SimulationViewer] manifest load failed:', e);
        setLoadError(String(e));
      });
    cachedFetchText(`${comparisonsBase()}index.csv`, 0)
      .then((csv) => setCompRows(parseComparisonCsv(csv)))
      .catch((e) => {
        console.error('[SimulationViewer] comparison index load failed:', e);
        setCompRows([]);
      });
  }, []);

  const realRuns = useMemo(
    () => (manifest?.runs ?? [])
      .filter((r) => (showDryRun || r.model !== DRY_RUN_MODEL) && r.status !== 'error'),
    [manifest, showDryRun],
  );

  const runById = useMemo(() => {
    const byId = new Map<string, RunInfo>();
    for (const r of realRuns) if (r.run_id) byId.set(r.run_id, r);
    return byId;
  }, [realRuns]);

  const situations = useMemo(() => {
    const byKey = new Map<string, SituationInfo>();
    for (const s of manifest?.situations ?? []) byKey.set(`${s.profile_id}/${s.situation_id}`, s);
    return [...byKey.values()];
  }, [manifest]);

  const populatedSituationKeys = useMemo(() => {
    const keys = new Map<string, SituationInfo>();
    for (const r of realRuns) {
      const s = situations.find((x) => x.profile_id === r.profile_id && x.situation_id === r.situation_id);
      if (s) keys.set(`${r.profile_id}/${r.situation_id}`, s);
    }
    return keys;
  }, [realRuns, situations]);

  const currentSituation = useMemo(() => {
    const s = situations.find((x) => `${x.profile_id}/${x.situation_id}` === situationKey);
    return s ?? situations.find((x) => `${x.profile_id}/${x.situation_id}` === [...populatedSituationKeys.keys()][0]) ?? null;
  }, [situationKey, situations, populatedSituationKeys]);

  const runsForCurrent = useMemo(
    () => currentSituation
      ? realRuns.filter((r) => r.profile_id === currentSituation.profile_id && r.situation_id === currentSituation.situation_id)
      : [],
    [currentSituation, realRuns],
  );

  const modelsForCurrent = useMemo(() => {
    const s = new Set(runsForCurrent.map((r) => r.model));
    return [...s].sort();
  }, [runsForCurrent]);

  const runsForModel = useCallback((model: string) =>
    runsForCurrent.filter((r) => r.model === model).sort((x, y) => x.version - y.version),
  [runsForCurrent]);

  // Comparisons viewable for the current situation (both agent runs must
  // exist in the exported runs of the manifest).
  const comparisonsForCurrent = useMemo(() => {
    if (!compRows || !currentSituation) return [];
    const scenario = `${currentSituation.profile_id}-${currentSituation.situation_id}`;
    return compRows.filter(
      (row) => row.scenario === scenario
        && runById.has(row.agent1_run_id)
        && runById.has(row.agent2_run_id),
    );
  }, [compRows, currentSituation, runById]);

  const selectedComparisonRow = useMemo(() => {
    if (!comparisonsForCurrent.length) return null;
    // A deep-linked pair wins; then an explicit picker choice; otherwise the
    // newest row for this situation.
    if (deepLinkPair.a && deepLinkPair.b) {
      const match = findComparison(comparisonsForCurrent, deepLinkPair.a, deepLinkPair.b);
      if (match) return match;
    }
    if (comparisonPick) {
      const byId = comparisonsForCurrent.find((r) => r.comparison_id === comparisonPick);
      if (byId) return byId;
    }
    return comparisonsForCurrent[comparisonsForCurrent.length - 1];
  }, [comparisonsForCurrent, deepLinkPair, comparisonPick]);

  const runA = useMemo(
    () => (selectedComparisonRow ? runById.get(selectedComparisonRow.agent1_run_id) ?? null : null),
    [selectedComparisonRow, runById],
  );
  const runB = useMemo(
    () => (selectedComparisonRow ? runById.get(selectedComparisonRow.agent2_run_id) ?? null : null),
    [selectedComparisonRow, runById],
  );

  // Single mode resolved run: deep-linked run id, else the latest run of the
  // first model alphabetically.
  const singleRun = useMemo(() => {
    if (singleRunId) return runById.get(singleRunId) ?? null;
    const models = modelsForCurrent;
    if (!models.length) return null;
    const versions = runsForModel(models[0]);
    return versions[versions.length - 1] ?? null;
  }, [singleRunId, runById, modelsForCurrent, runsForModel]);

  const load = useCallback(async (run: RunInfo): Promise<RunData> => {
    const key = run.run_dir;
    if (cache.current[key]) return cache.current[key];
    console.info(`[SimulationViewer] fetching run: ${key}`);
    const data = await loadRun(
      LOCAL_MODE ? LOCAL_BS_BASE : manifest!.base_url,
      run,
    );
    cache.current[key] = data;
    console.info(`[SimulationViewer] loaded run: ${key} (${data.transcript?.messages?.length ?? 0} messages)`);
    return data;
  }, [manifest]);

  useEffect(() => {
    if (!manifest || !singleRun) { setDataSingle(null); return; }
    let cancelled = false;
    load(singleRun)
      .then((d) => { if (!cancelled) setDataSingle(d); })
      .catch((e) => {
        console.error(`[SimulationViewer] load failed (${singleRun.run_dir}):`, e);
        if (!cancelled) setDataSingle({ transcript: null, judge: null, selfReview: '', error: String(e) });
      });
    return () => { cancelled = true; };
  }, [manifest, singleRun, load, bumpRetry]);

  useEffect(() => {
    if (!runA) { setDataA(null); return; }
    let cancelled = false;
    load(runA)
      .then((d) => { if (!cancelled) setDataA(d); })
      .catch((e) => {
        console.error(`[SimulationViewer] load failed for agent 1 (${runA.run_dir}):`, e);
        if (!cancelled) setDataA({ transcript: null, judge: null, selfReview: '', error: String(e) });
      });
    return () => { cancelled = true; };
  }, [runA, load, bumpRetry]);

  useEffect(() => {
    if (!runB) { setDataB(null); return; }
    let cancelled = false;
    load(runB)
      .then((d) => { if (!cancelled) setDataB(d); })
      .catch((e) => {
        console.error(`[SimulationViewer] load failed for agent 2 (${runB.run_dir}):`, e);
        if (!cancelled) setDataB({ transcript: null, judge: null, selfReview: '', error: String(e) });
      });
    return () => { cancelled = true; };
  }, [runB, load, bumpRetry]);

  useEffect(() => {
    if (!compRows || !selectedComparisonRow) {
      setComparison(null);
      return;
    }
    const key = selectedComparisonRow.comparison_id;
    const cached = compCache.current[key];
    if (cached) {
      setComparison(cached);
      return;
    }
    let cancelled = false;
    const url = `${comparisonsBase()}${selectedComparisonRow.model_pair}/${selectedComparisonRow.scenario}/${selectedComparisonRow.comparison_id}.json`;
    cachedFetchJson<ComparisonDoc>(url, 0)
      .then((doc) => {
        compCache.current[key] = doc;
        if (!cancelled) setComparison(doc);
      })
      .catch((e) => {
        console.error(`[SimulationViewer] comparison load failed: ${key}`, e);
        if (!cancelled) setComparison(null);
      });
    return () => { cancelled = true; };
  }, [compRows, selectedComparisonRow]);

  /* ---------------------------- selections ---------------------------- */

  // Initial selection: pick the first populated situation once the manifest
  // loads and no deep link set one.
  useEffect(() => {
    if (!manifest || situationKey) return;
    const firstKey = [...populatedSituationKeys.keys()][0];
    if (firstKey) setSituationKey(firstKey);
  }, [manifest, situationKey, populatedSituationKeys]);

  // Deep link parameters (applied once once both indexes are in):
  //  - `?situation=<profile_id>-<situation_id>` -> single mode, that situation
  //    (default model/run pick inside it)
  //  - `?a=<run_id>&b=<run_id>` -> comparison mode with that pair
  //  - `?comparison=<comparison_id>` -> comparison mode, that comparison
  //  - `?run=<run_id>` (or a lone `?a=`) -> single mode with that run
  const autoApplied = useRef(false);
  useEffect(() => {
    if (!manifest || !compRows || autoApplied.current) return;
    const applyComparison = (row: ComparisonRow | undefined): boolean => {
      if (!row || !runById.has(row.agent1_run_id) || !runById.has(row.agent2_run_id)) return false;
      autoApplied.current = true;
      setMode('compare');
      const ra = runById.get(row.agent1_run_id)!;
      setSituationKey(`${ra.profile_id}/${ra.situation_id}`);
      setDeepLinkPair({ a: row.agent1_run_id, b: row.agent2_run_id });
      return true;
    };

    // A situation-only deep link lands on that situation in single mode,
    // then lets it fall back to its usual default run pick. situations.json
    // contains profile ids with `_`, so the first '-' separates the pair.
    const situationParam = urlParams.get('situation');
    if (situationParam) {
      const [pid, ...sitParts] = situationParam.split('-');
      const key = `${pid}/${sitParts.join('-')}`;
      if (situations.find((s) => `${s.profile_id}/${s.situation_id}` === key)) {
        autoApplied.current = true;
        const wantCompare = urlParams.get('mode') === 'compare';
        const compsForParam = compRows.filter((r) => r.scenario === situationParam
          && runById.has(r.agent1_run_id) && runById.has(r.agent2_run_id));
        if (wantCompare && compsForParam.length) {
          setMode('compare');
          setDeepLinkPair({ a: null, b: null });
        } else {
          setMode('single');
        }
        setSituationKey(key);
        return;
      }
    }
    const comparisonId = urlParams.get('comparison');
    const a = urlParams.get('a') ?? urlParams.get('run');
    const b = urlParams.get('b');
    if (a && b && runById.has(a) && runById.has(b)) {
      if (applyComparison(findComparison(compRows, a, b))) return;
    }
    if (a && runById.has(a)) {
      const run = runById.get(a)!;
      autoApplied.current = true;
      setMode('single');
      setSituationKey(`${run.profile_id}/${run.situation_id}`);
      setSingleRunId(run.run_id!);
    }
  }, [manifest, compRows, runById, situations, situationKey]);

  // Keep the URL in sync with the selection (replaceState: no history spam)
  // so the current view can be copied as a shareable deep link.
  useEffect(() => {
    if (typeof window === 'undefined') return;
    const params = new URLSearchParams(window.location.search);
    const setOrDel = (key: string, value?: string | null) => {
      if (value) params.set(key, value); else params.delete(key);
    };
    if (mode === 'single') {
      setOrDel('run', singleRun?.run_id);
      setOrDel('comparison'); setOrDel('a'); setOrDel('b');
    } else {
      setOrDel('a', runA?.run_id);
      setOrDel('b', runB?.run_id);
      setOrDel('run'); setOrDel('comparison');
    }
    setOrDel('mode', mode === 'compare' ? 'compare' : null);
    const qs = params.toString();
    const url = `${window.location.pathname}${qs ? `?${qs}` : ''}${window.location.hash}`;
    if (url !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
      window.history.replaceState(null, '', url);
    }
  });

  const onSelect = useCallback((origin: 'A' | 'B') => (index: number) => {
    setHighlightOrigin(origin);
    setHighlightIndex(index);
  }, []);

  const setSituation = (key: string) => {
    setSituationKey(key);
    setSingleRunId(null);
    setDeepLinkPair({ a: null, b: null });
    setHighlightIndex(null);
  };

  const groupedSituations = useMemo(() => {
    const groups = new Map<string, SituationInfo[]>();
    for (const s of situations) {
      if (!populatedSituationKeys.has(`${s.profile_id}/${s.situation_id}`)) continue;
      const arr = groups.get(s.profile_id) ?? [];
      arr.push(s);
      groups.set(s.profile_id, arr);
    }
    return [...groups.entries()];
  }, [situations, populatedSituationKeys]);

  /* ------------------------------ rendering ---------------------------- */

  if (loadError) {
    return (
      <div style={{ padding: '3rem 0' }}>
        <div style={{ ...card, borderColor: '#fca5a5', color: '#b91c1c' }}>
          <strong>Could not load simulation data</strong>
          <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>{loadError}</p>
          <p style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
            The runs live in the public HF bucket <code>{MANIFEST_URL}</code>.
          </p>
        </div>
      </div>
    );
  }

  if (!manifest) {
    return <div style={{ padding: '3rem 0', color: 'var(--color-muted)' }}>Loading simulation data from the HF bucket…</div>;
  }

  return (
    <div>
      <div style={{ marginBottom: '1.5rem', padding: '1rem 1.25rem', background: '#fff7ed', borderRadius: '0.6rem', border: '2px solid #f59e0b', fontSize: '0.95rem', fontWeight: 600, color: '#9a3412', lineHeight: 1.6 }}>
        ⚠️ <strong>Debug data — not pilot or final results.</strong> Everything on this page is raw development/debug output (including pipeline test runs), not study data. It exists to exercise the viewing tooling and will be replaced as the project progresses. Each run is a full work-item simulation logged to the project's public{' '}
        <a href={`https://huggingface.co/buckets/${manifest.bucket}`} target="_blank" rel="noopener noreferrer" style={{ color: '#9a3412' }}>HF bucket ↗</a>.
      </div>

      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', alignItems: 'flex-end', marginBottom: '1.5rem' }}>
        <div>
          <label style={label}>View</label>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            {[{ id: 'single' as const, label: 'Single run' }, { id: 'compare' as const, label: 'Comparison' }].map((m) => (
              <button key={m.id} onClick={() => { setMode(m.id); setHighlightIndex(null); }} style={modeButton(mode === m.id)}>
                {m.label}
              </button>
            ))}
          </div>
        </div>
        <div>
          <label style={label}>Situation</label>
          <select value={situationKey} onChange={(e) => setSituation(e.target.value)} style={select}>
            {groupedSituations.map(([pid, list]) => (
              <optgroup key={pid} label={manifest.profiles[pid]?.name ?? pid}>
                {list.map((s) => (
                  <option key={`${s.profile_id}/${s.situation_id}`} value={`${s.profile_id}/${s.situation_id}`}>
                    {s.name}{s.type === 'interactive' ? ' (interactive)' : ''}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8rem', paddingBottom: '0.3rem' }}>
          <input type="checkbox" checked={showReasoning} onChange={(e) => setShowReasoning(e.target.checked)} /> Reasoning
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8rem', paddingBottom: '0.3rem' }}>
          <input type="checkbox" checked={showDryRun} onChange={(e) => setShowDryRun(e.target.checked)} /> Show test (dry-run) model
        </label>
      </div>

      {!runsForCurrent.length && <div style={card}>No runs found for this situation.</div>}

      {currentSituation && mode === 'single' && runsForCurrent.length > 0 && (
        <div style={{ maxWidth: '56rem', margin: '0 auto' }}>
          <div style={{ ...card, marginBottom: '1rem' }}>
            <p style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-muted)', marginBottom: '0.35rem' }}>Run</p>
            <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
              <div style={{ flex: 1, minWidth: '13rem' }}>
                <label style={label}>Model</label>
                <select
                  style={{ ...select, width: '100%', minWidth: 0 }}
                  value={singleRun?.model ?? ''}
                  onChange={(e) => {
                    const versions = runsForModel(e.target.value);
                    const best = versions[versions.length - 1];
                    if (best?.run_id) setSingleRunId(best.run_id);
                  }}
                >
                  {modelsForCurrent.map((m) => (
                    <option key={m} value={m}>{shortModel(m)}{m === DRY_RUN_MODEL ? ' (test)' : ''}</option>
                  ))}
                </select>
              </div>
              <div style={{ flex: 1, minWidth: '12rem' }}>
                <label style={label}>Run of this scenario</label>
                <select
                  style={{ ...select, width: '100%', minWidth: 0 }}
                  value={singleRun?.run_id ?? ''}
                  onChange={(e) => setSingleRunId(e.target.value)}
                >
                  {(singleRun ? runsForModel(singleRun.model) : []).map((r) => (
                    <option key={r.version} value={r.run_id}>
                      Run {r.version} · {fmtDate(r.created) || '?'}{r.judge_score != null ? ` · ${r.judge_score}/5` : ''}
                    </option>
                  ))}
                </select>
              </div>
              {singleRun && <ScoreBadge run={singleRun} />}
            </div>
          </div>
          <RunPanel
            title="Transcript"
            run={singleRun}
            data={dataSingle}
            highlightIndex={highlightIndex}
            onSelect={onSelect('A')}
            showReasoning={showReasoning}
            active
            onRetry={bumpRetry}
          />
        </div>
      )}

      {currentSituation && mode === 'compare' && (
        comparisonsForCurrent.length === 0 ? (
          <div style={card}>
            No completed cross-model comparison for this situation yet (built
            by <code>build_comparisons.py</code>, synced with{' '}
            <code>make artifacts-sync</code>). Use the <strong>Single run</strong>{' '}
            view to inspect runs, or pick any two in the other viewer tabs.
          </div>
        ) : (
          <>
            {comparisonsForCurrent.length > 1 && (
              <div style={{ ...card, marginBottom: '1rem', background: 'var(--color-surface)' }}>
                <label style={label}>Comparison — {comparisonsForCurrent.length} built for this scenario; pick any pair</label>
                <select
                  value={selectedComparisonRow?.comparison_id ?? ''}
                  onChange={(e) => { setComparisonPick(e.target.value); setComparison(null); }}
                  style={select}
                >
                  {comparisonsForCurrent.map((r) => (
                    <option key={r.comparison_id} value={r.comparison_id}>
                      {shortModel(r.agent1_model)} vs {shortModel(r.agent2_model)}
                    </option>
                  ))}
                </select>
              </div>
            )}
            <ComparisonPanel comparison={comparison} row={selectedComparisonRow} />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'start' }}>
              <div style={{ borderLeft: '4px solid #6366f1', paddingLeft: '1rem' }}>
                <RunPanel
                  title="Agent 1"
                  run={runA}
                  data={dataA}
                  highlightIndex={highlightOrigin === 'A' ? null : highlightIndex}
                  onSelect={onSelect('A')}
                  showReasoning={showReasoning}
                  active
                  onRetry={bumpRetry}
                />
              </div>
              <div style={{ borderRight: '4px solid #ec4899', paddingRight: '1rem' }}>
                <RunPanel
                  title="Agent 2"
                  run={runB}
                  data={dataB}
                  highlightIndex={highlightOrigin === 'B' ? null : highlightIndex}
                  onSelect={onSelect('B')}
                  showReasoning={showReasoning}
                  active
                  onRetry={bumpRetry}
                />
              </div>
            </div>
          </>
        )
      )}
    </div>
  );
}
