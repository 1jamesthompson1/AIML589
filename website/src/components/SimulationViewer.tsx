import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { cachedFetchJson, cachedFetchText } from '../lib/cachedFetch';

/* Behavioural simulation runs viewer.
 *
 * Data lives in the project's public HF storage bucket (see
 * code/behavioural-simulations/export_results.py): a manifest (index.json)
 * lists every run; each run directory holds transcript.json, judge.json,
 * self_review.txt and audit.txt. This component fetches the manifest from
 * the bucket and lets visitors compare two runs side by side - across two
 * different models, or across different runs (versions) of the same model
 * and scenario.
 */

const DEFAULT_BUCKET = '1jamesthompson1/wvs-nz-value-alignment-evals';
const DRY_RUN_MODEL = 'mockllm/dry-run';

const urlParams = typeof window !== 'undefined'
  ? new URLSearchParams(window.location.search)
  : new URLSearchParams();
const MANIFEST_URL = urlParams.get('manifest')
  ?? `https://huggingface.co/buckets/${DEFAULT_BUCKET}/resolve/bs/runs/index.json`;

interface RunInfo {
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
  audit: string;
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
  onTranscript?.({ transcript, judge: null, selfReview: '', audit: '' });
  const [judge, selfReview, audit] = await Promise.all([
    cachedFetchJson<JudgeDoc>(`${base}judge.json`).catch(() => null),
    cachedFetchText(`${base}self_review.txt`).catch(() => ''),
    cachedFetchText(`${base}audit.txt`).catch(() => ''),
  ]);
  return { transcript, judge, selfReview, audit };
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

            {(data.selfReview || data.audit) && (
              <div style={{ background: '#fffbeb', border: '2px solid #f59e0b', borderRadius: '0.75rem', padding: '0.9rem 1rem', boxShadow: '0 2px 8px rgba(245,158,11,0.12)' }}>
                <p style={{ fontSize: '0.7rem', fontWeight: 800, textTransform: 'uppercase', letterSpacing: '0.06em', color: '#b45309', marginBottom: '0.5rem' }}>🧭 Model summary & audit</p>
                {data.selfReview && (
                  <>
                    <p style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em', color: '#b45309', marginBottom: '0.25rem' }}>Self-review</p>
                    <div style={{ margin: '0 0 0.75rem' }}><ThinkingAwareText text={data.selfReview} /></div>
                  </>
                )}
                {data.audit && (
                  <>
                    <p style={{ fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.04em', color: '#b45309', marginBottom: '0.25rem' }}>Audit</p>
                    <div><ThinkingAwareText text={data.audit} /></div>
                  </>
                )}
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

interface SideConfig { model: string; version: number; }

export default function SimulationViewer() {
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [loadError, setLoadError] = useState<string>('');
  const [situationKey, setSituationKey] = useState<string>('');
  const [sideA, setSideA] = useState<SideConfig | null>(null);
  const [sideB, setSideB] = useState<SideConfig | null>(null);
  const [dataA, setDataA] = useState<RunData | null>(null);
  const [dataB, setDataB] = useState<RunData | null>(null);
  const [highlightIndex, setHighlightIndex] = useState<number | null>(null);
  const [highlightOrigin, setHighlightOrigin] = useState<'A' | 'B' | null>(null);
  const [showReasoning, setShowReasoning] = useState(true);
  const [showDryRun, setShowDryRun] = useState(false);
  const cache = useRef<Record<string, RunData>>({});

  useEffect(() => {
    console.info(`[SimulationViewer] fetching manifest: ${MANIFEST_URL}`);
    // The manifest is the gatekeeper for new runs, so always re-fetch it (0 TTL)
    // rather than serving a stale copy from the localStorage cache.
    cachedFetchJson<Manifest>(MANIFEST_URL, 0)
      .then((m) => {
        console.info(`[SimulationViewer] manifest loaded: ${m.runs?.length ?? 0} runs, ${m.situations?.length ?? 0} situations`);
        setManifest(m);
      })
      .catch((e) => {
        console.error('[SimulationViewer] manifest load failed:', e);
        setLoadError(String(e));
      });
  }, []);

  const realRuns = useMemo(
    () => (manifest?.runs ?? []).filter((r) => (showDryRun || r.model !== DRY_RUN_MODEL) && r.status !== 'error'),
    [manifest, showDryRun],
  );

  const situations = useMemo(() => {
    const byKey = new Map<string, SituationInfo>();
    for (const s of manifest?.situations ?? []) byKey.set(`${s.profile_id}/${s.situation_id}`, s);
    return [...byKey.values()];
  }, [manifest]);

  const runsForSituation = useCallback((profileId: string, situationId: string) =>
    realRuns.filter((r) => r.profile_id === profileId && r.situation_id === situationId),
  [realRuns]);

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

  const [curProfileId, curSituationId] = currentSituation
    ? [currentSituation.profile_id, currentSituation.situation_id]
    : [null, null];

  const runsForCurrent = useMemo(
    () => curProfileId && curSituationId ? runsForSituation(curProfileId, curSituationId) : [],
    [curProfileId, curSituationId, runsForSituation],
  );

  const modelsForCurrent = useMemo(() => {
    const s = new Set(runsForCurrent.map((r) => r.model));
    return [...s].sort();
  }, [runsForCurrent]);

  const versionsFor = useCallback((model: string) =>
    runsForCurrent.filter((r) => r.model === model).sort((a, b) => a.version - b.version),
  [runsForCurrent]);

  const resolveSide = useCallback((side: SideConfig | null): RunInfo | null => {
    if (!side || !modelsForCurrent.includes(side.model)) return null;
    const versions = versionsFor(side.model);
    return versions.find((r) => r.version === side.version) ?? versions[versions.length - 1] ?? null;
  }, [modelsForCurrent, versionsFor]);

  const runA = resolveSide(sideA);
  const runB = resolveSide(sideB);

  // Assurance: the two sides must never resolve to the exact same run. If a
  // selection lands both sides on one run (defaults, swap, or a pick that
  // matches the other side), move side B onto a different run - another
  // model if one exists, otherwise another version of the same model.
  useEffect(() => {
    if (!runA || !runB || runA.run_dir !== runB.run_dir) return;
    const others = runsForCurrent.filter((r) => r.run_dir !== runA.run_dir);
    if (!others.length) return;
    const otherModel = others.find((r) => r.model !== runB.model);
    const pick = otherModel ?? others[0];
    setSideB({ model: pick.model, version: pick.version });
  }, [runA, runB, runsForCurrent]);

  // Initial selection: pick the first populated situation once the manifest loads.
  useEffect(() => {
    if (!manifest || situationKey) return;
    const firstKey = [...populatedSituationKeys.keys()][0];
    if (firstKey) setSituationKey(firstKey);
  }, [manifest, situationKey, populatedSituationKeys]);

  useEffect(() => {
    if (!currentSituation) return;
    const runs = runsForCurrent;
    if (!runs.length) return;
    const models = [...new Set(runs.map((r) => r.model))].sort();

    const mkSide = (model: string): SideConfig => {
      const versions = versionsFor(model);
      return { model, version: versions[versions.length - 1].version };
    };
    if (!sideA) setSideA(mkSide(models[0]));
    if (!sideB) {
      if (models.length > 1) setSideB(mkSide(models[1]));
      else {
        const versions = versionsFor(models[0]);
        if (versions.length > 1) setSideB({ model: models[0], version: versions[0].version });
      }
    }
  }, [currentSituation, runsForCurrent, sideA, sideB, versionsFor]);

  const load = useCallback(async (run: RunInfo, onTranscript?: (partial: RunData) => void): Promise<RunData> => {
    const key = run.run_dir;
    if (cache.current[key]) {
      console.info(`[SimulationViewer] run served from cache: ${key}`);
      return cache.current[key];
    }
    console.info(`[SimulationViewer] fetching run: ${key}`);
    const data = await loadRun(manifest!.base_url, run, onTranscript);
    cache.current[key] = data;
    console.info(`[SimulationViewer] loaded run: ${key} (${data.transcript?.messages?.length ?? 0} messages)`);
    return data;
  }, [manifest]);

  const [retryTick, setRetryTick] = useState(0);

  useEffect(() => {
    if (!runA) { setDataA(null); return; }
    let cancelled = false;
    load(runA, setDataA).then((d) => { if (!cancelled) setDataA(d); }).catch((e) => { console.error(`[SimulationViewer] load failed for side A (${runA.run_dir}):`, e); if (!cancelled) setDataA({ transcript: null, judge: null, selfReview: '', audit: '', error: String(e) }); });
    return () => { cancelled = true; };
  }, [runA, load, retryTick]);

  useEffect(() => {
    if (!runB) { setDataB(null); return; }
    let cancelled = false;
    load(runB, setDataB).then((d) => { if (!cancelled) setDataB(d); }).catch((e) => { console.error(`[SimulationViewer] load failed for side B (${runB.run_dir}):`, e); if (!cancelled) setDataB({ transcript: null, judge: null, selfReview: '', audit: '', error: String(e) }); });
    return () => { cancelled = true; };
  }, [runB, load, retryTick]);

  const onSelect = useCallback((origin: 'A' | 'B') => (index: number) => {
    setHighlightOrigin(origin);
    setHighlightIndex(index);
  }, []);

  const swap = () => { setSideA(sideB); setSideB(sideA); };

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
          <label style={label}>Situation</label>
          <select value={situationKey} onChange={(e) => { setSituationKey(e.target.value); setSideA(null); setSideB(null); setHighlightIndex(null); }} style={select}>
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
      </div>

      {!runsForCurrent.length && (
        <div style={card}>No runs found for this situation.</div>
      )}

      {runsForCurrent.length > 0 && currentSituation && (
        <>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginBottom: '1.25rem' }}>
            {([['A', sideA, setSideA, dataA, runB], ['B', sideB, setSideB, dataB, runA]] as const).map(([letter, side, setSide, data, otherRun]) => {
              const chosenRun = letter === 'A' ? runA : runB;
              // Runs available to pick on this side: never the run currently
              // shown on the other side.
              const versions = side
                ? versionsFor(side.model).filter((r) => r.run_dir !== otherRun?.run_dir)
                : [];
              return (
                <div key={letter}>
                  <div style={{ ...card, marginBottom: '0.5rem', borderColor: letter === 'A' ? '#c7d2fe' : '#fecdd3' }}>
                    <p style={{ fontSize: '0.7rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-muted)', marginBottom: '0.35rem' }}>Side {letter}{letter === 'A' ? ' (left)' : ' (right)'}</p>
                    <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
                      <div style={{ flex: 1, minWidth: '13rem' }}>
                        <label style={label}>Model</label>
                        <select style={{ ...select, width: '100%', minWidth: 0 }} value={side?.model ?? ''} onChange={(e) => {
                          const model = e.target.value;
                          const versions = versionsFor(model);
                          setSide({ model, version: versions[versions.length - 1].version });
                        }}>
                          {modelsForCurrent.map((m) => <option key={m} value={m}>{shortModel(m)}{m === DRY_RUN_MODEL ? ' (test)' : ''}</option>)}
                        </select>
                      </div>
                      <div style={{ flex: 1, minWidth: '12rem' }}>
                        <label style={label}>Run of this scenario</label>
                        <select style={{ ...select, width: '100%', minWidth: 0 }} value={versions.some((r) => r.version === side?.version) ? side?.version : versions[versions.length - 1]?.version ?? ''} onChange={(e) => setSide({ model: side!.model, version: Number(e.target.value) })}>
                          {versions.map((r) => (
                            <option key={r.version} value={r.version}>
                              Run {r.version} · {fmtDate(r.created) || '?'}{r.judge_score != null ? ` · ${r.judge_score}/5` : ''}
                            </option>
                          ))}
                        </select>
                      </div>
                    </div>
                    <p style={{ fontSize: '0.72rem', color: 'var(--color-muted)', marginTop: '0.35rem' }}>
                      {versions.length} run{versions.length === 1 ? '' : 's'} of this scenario for this model.
                    </p>
                  </div>
                  {chosenRun && <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.8rem', color: 'var(--color-muted)' }}>Comparing:</span>
                    <ScoreBadge run={chosenRun} />
                  </div>}
                </div>
              );
            })}
          </div>

          <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'center', marginBottom: '1.5rem' }}>
            <button onClick={swap} style={{ padding: '0.5rem 1rem', borderRadius: '0.5rem', border: '1px solid var(--color-border)', background: 'white', cursor: 'pointer', fontSize: '0.85rem', fontFamily: 'inherit' }}>⇄ Swap sides</button>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8rem' }}>
              <input type="checkbox" checked={showReasoning} onChange={(e) => setShowReasoning(e.target.checked)} /> Reasoning
            </label>
            <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8rem' }}>
              <input type="checkbox" checked={showDryRun} onChange={(e) => setShowDryRun(e.target.checked)} /> Show test (dry-run) model
            </label>
            <span style={{ fontSize: '0.75rem', color: 'var(--color-muted)' }}>Click a message to align it with the same step on the other side.</span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', alignItems: 'start' }}>
            <div style={{ borderLeft: '4px solid #6366f1', paddingLeft: '1rem' }}>
              <RunPanel title="Side A" run={runA} data={dataA} highlightIndex={highlightOrigin === 'A' ? null : highlightIndex} onSelect={onSelect('A')} showReasoning={showReasoning} active onRetry={() => setRetryTick((t) => t + 1)} />
            </div>
            <div style={{ borderRight: '4px solid #ec4899', paddingRight: '1rem' }}>
              <RunPanel title="Side B" run={runB} data={dataB} highlightIndex={highlightOrigin === 'B' ? null : highlightIndex} onSelect={onSelect('B')} showReasoning={showReasoning} active onRetry={() => setRetryTick((t) => t + 1)} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}