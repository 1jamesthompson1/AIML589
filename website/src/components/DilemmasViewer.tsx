import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { cachedFetchJson, cachedFetchText } from '../lib/cachedFetch';
import DataSourceNotice from './DataSourceNotice';
import {
  DEFAULT_BUCKET,
  LOCAL_DILEMMAS_BASE,
  isLocalSourceUrl,
  manifestUrl,
  urlParams,
} from '../lib/dataSource';
import { SURVEY_ROLE_LEAD } from '../lib/survey';

/* The survey block's framing (survey/comparison-question.md) with the role,
 * situation and question leads kept as the reader sees them in the public
 * consultation. A survey block always shows exactly two answers, so its
 * wording is singular; the viewer can show every model with runs, hence the
 * plural-neutral variants for the situation and question leads. */
const VIEWER_SITUATION_LEAD =
  'Each model below was given this situation, independently of the others:';
const VIEWER_QUESTION_LEAD = 'They were all asked:';

/* Dilemma runs viewer.
 *
 * Data lives in the project's public HF storage bucket (or the local
 * artifacts mirror in dev):
 *   dilemmas/runs/index.json         - profiles (with system prompts),
 *                                      dilemmas and every run
 *   dilemmas/runs/<path>/response.txt
 *   dilemmas/runs/<path>/judge.json
 *
 * Every model's response to one dilemma side by side, with a per-model run
 * selector. The selected profile's system prompt is shown collapsed under
 * the dilemma stimulus.
 */

const RUNS_MANIFEST_URL = manifestUrl(LOCAL_DILEMMAS_BASE, 'dilemmas/runs/index.json');
const MANIFEST_IS_LOCAL = isLocalSourceUrl(RUNS_MANIFEST_URL);
const RUNS_BASE = MANIFEST_IS_LOCAL
  ? LOCAL_DILEMMAS_BASE
  : RUNS_MANIFEST_URL.replace(/index\.json(?:\?.*)?$/, '');

interface DilemmaInfo {
  profile_id: string;
  situation_id: string;
  tension?: number | null;
  tension_name?: string;
  situation?: string;
  question?: string;
  answer_options?: string[];
  value_options?: string[];
  /** model -> judge choice -> exported runs (runs index v2, additive field) */
  choice_counts?: Record<string, Record<string, number>>;
}

interface RunRow {
  run_id: string;
  model: string;
  profile_id: string;
  situation_id: string;
  version: number;
  created: string;
  path: string;
}

interface RunsManifest {
  schema: string;
  generated_at?: string;
  bucket?: string;
  base_url?: string;
  profiles?: Record<
    string,
    { name?: string; role_summary?: string; system_prompt?: string }
  >;
  dilemmas?: DilemmaInfo[];
  models?: string[];
  runs: RunRow[];
}

interface JudgeDoc {
  value?: number | null;
  answer?: string;
  explanation?: string;
  metadata?: {
    choice?: string;
    value?: string;
    reason?: string;
    dilemma_id?: string;
    profile_id?: string;
  };
}

interface RunData {
  status: 'loading' | 'ready' | 'error';
  response?: string;
  judge?: JudgeDoc | null;
  error?: string;
}

const label: React.CSSProperties = { display: 'block', fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--color-muted)', marginBottom: '0.25rem' };
const select: React.CSSProperties = { padding: '0.5rem 0.75rem', borderRadius: '0.5rem', border: '1px solid var(--color-border)', background: 'white', fontSize: '0.875rem', minWidth: '200px', fontFamily: 'inherit', color: 'var(--color-text)' };
const card: React.CSSProperties = { background: 'var(--color-bg)', border: '1px solid var(--color-border)', borderRadius: '0.75rem', padding: '0.75rem 1rem' };

function shortModel(model: string): string {
  return model.replace(/^openrouter\//, '');
}

function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '';
  const d = new Date(iso);
  return d.toUTCString().replace(/ GMT$/, '');
}

function splitKey(key: string): [string, string] {
  const slash = key.indexOf('/');
  return [key.slice(0, slash), key.slice(slash + 1)];
}

function profileLabel(manifest: RunsManifest | null, id: string): string {
  return manifest?.profiles?.[id]?.name
    || id.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

function dilemmaLabel(dilemma: DilemmaInfo): string {
  const text = dilemma.question || dilemma.situation || dilemma.situation_id;
  return text.length > 90 ? `${text.slice(0, 89)}…` : text;
}

/* Outcome distribution for one dilemma: the share of each model's exported
 * runs that the judge placed on each answer option. The series stay curated -
 * the Qwen 3.8 family (base and fine-tunes) plus the two frontier comparison
 * models - so spot-test extras (haiku, gpt-5.6-luna) and one-off adapters do
 * not clutter the study view. */
const OUTCOME_COLOURS = ['#0f3460', '#e94560', '#16a34a', '#7c3aed', '#f59e0b'];

const QWEN_RE = /qwen3\.8-27b/i;
const FRONTIER_RE = /deepseek-v4\.1-flash|glm-5\.3-flash/i;

function outcomeSeries(dilemma: DilemmaInfo | null): string[] {
  const counts = dilemma?.choice_counts;
  if (!counts) return [];
  const models = Object.keys(counts).filter((model) => Object.keys(counts[model]).length);
  const qwen = models.filter((model) => QWEN_RE.test(model)).sort();
  const frontier = models.filter((model) => FRONTIER_RE.test(model)).sort();
  if (qwen.length) return [...qwen, ...frontier];
  if (frontier.length) return frontier;
  return models.sort();
}

function outcomeLabel(model: string, fineTunedCount: number): string {
  if (/Qwen3\.8-27B-nz-wvs/i.test(model)) {
    if (fineTunedCount < 2) return 'Qwen3.8-27B fine-tuned';
    const variant = model.split(/nz-wvs-/i)[1] ?? model;
    return `Qwen3.8-27B FT ${variant.replace(/[-_]/g, ' ')}`;
  }
  if (/Qwen3\.8-27B$/i.test(model)) {
    return model.startsWith('openai/')
      ? 'Qwen3.8-27B base (local)'
      : 'Qwen3.8-27B base (OpenRouter)';
  }
  return shortModel(model);
}

function OutcomeChart({ dilemma }: { dilemma: DilemmaInfo | null }) {
  const counts = dilemma?.choice_counts;
  const options = dilemma?.answer_options ?? [];
  const series = outcomeSeries(dilemma);
  if (!counts || !series.length || !options.length) return null;

  const totals = Object.fromEntries(
    series.map((model) => [model, Object.values(counts[model]).reduce((a, b) => a + b, 0)])
  );
  const fineTunedCount = series.filter((model) => /Qwen3\.8-27B-nz-wvs/i.test(model)).length;

  return (
    <section className="dilemma-outcomes" aria-label="Outcome distribution">
      <p className="dilemma-outcomes__title">
        Outcome distribution{' '}
        <span className="muted-text">
          — the share of each model's runs the judge placed on each answer option
        </span>
      </p>
      <div className="dilemma-outcomes__legend">
        {series.map((model, index) => (
          <span key={model} className="dilemma-outcomes__legend-item">
            <span
              className="dilemma-outcomes__swatch"
              style={{ background: OUTCOME_COLOURS[index % OUTCOME_COLOURS.length] }}
            />
            {outcomeLabel(model, fineTunedCount)} · {totals[model]} runs
          </span>
        ))}
      </div>
      <div className="dilemma-outcomes__chart">
        {options.map((option, optionIndex) => (
          <div key={option} className="dilemma-outcomes__group" title={`${optionIndex + 1}. ${option}`}>
            <div className="dilemma-outcomes__bars">
              {series.map((model, index) => {
                const n = counts[model][option] ?? 0;
                const share = totals[model] ? n / totals[model] : 0;
                return (
                  <div
                    key={model}
                    className="dilemma-outcomes__bar-wrap"
                    title={`${outcomeLabel(model, fineTunedCount)}: ${n}/${totals[model]} runs (${Math.round(share * 100)}%)`}
                  >
                    <span className="dilemma-outcomes__bar-value">
                      {n ? `${Math.round(share * 100)}%` : ''}
                    </span>
                    <div
                      className="dilemma-outcomes__bar"
                      style={{
                        height: `${share * 100}%`,
                        background: OUTCOME_COLOURS[index % OUTCOME_COLOURS.length],
                      }}
                    />
                  </div>
                );
              })}
            </div>
            <span className="dilemma-outcomes__group-label">{optionIndex + 1}</span>
          </div>
        ))}
      </div>
      <ol className="dilemma-outcomes__options">
        {options.map((option) => <li key={option}>{option}</li>)}
      </ol>
    </section>
  );
}

/* The judge file written by export_results.py holds the option the judge
 * matched to the response, the value it says the response prioritised, and
 * its one-line reason. Older files may carry only answer/explanation. */
function JudgeNote({ judge }: { judge: JudgeDoc | null | undefined }) {
  if (!judge) return <p className="muted-text dilemma-judge__empty">No judge result is stored for this run.</p>;
  const choice = judge.metadata?.choice ?? judge.answer ?? '—';
  const value = judge.metadata?.value;
  const reason = judge.metadata?.reason ?? judge.explanation;
  return (
    <div className="dilemma-judge">
      <p className="dilemma-judge__title">⚖ Judge classification</p>
      <p className="dilemma-judge__line"><strong>Decision:</strong> {choice}</p>
      {value && <p className="dilemma-judge__line"><strong>Value prioritised:</strong> {value}</p>}
      {reason && <p className="dilemma-judge__reason">{reason}</p>}
    </div>
  );
}

function DilemmaStimulus({ dilemma, fallbackId, profileName, roleSummary, systemPrompt }: {
  dilemma: DilemmaInfo | null;
  fallbackId: string;
  profileName: string;
  roleSummary?: string;
  systemPrompt?: string;
}) {
  return (
    <section className="dilemma-stimulus" aria-label="Dilemma">
      <p className="dilemma-stimulus__meta">
        <span>{profileName}</span>
        {dilemma?.tension_name && <span>Value tension: {dilemma.tension_name}</span>}
        <span className="muted-text">{dilemma?.situation_id ?? fallbackId}</span>
      </p>
      <p className="dilemma-stimulus__lead">{SURVEY_ROLE_LEAD}</p>
      {roleSummary && <p className="dilemma-stimulus__role">{roleSummary}</p>}
      <p className="dilemma-stimulus__lead">{VIEWER_SITUATION_LEAD}</p>
      {dilemma?.situation
        ? <p className="dilemma-stimulus__situation">{dilemma.situation}</p>
        : <p className="muted-text">Loading the dilemma text…</p>}
      {dilemma?.question && (
        <>
          <p className="dilemma-stimulus__lead">{VIEWER_QUESTION_LEAD}</p>
          <p className="dilemma-stimulus__question">{dilemma.question}</p>
        </>
      )}
      {systemPrompt && (
        <details className="dilemma-prompt">
          <summary>System prompt given to the model ({profileName})</summary>
          <pre className="simulation-preserve-text dilemma-prompt__text">{systemPrompt}</pre>
        </details>
      )}
      {(dilemma?.answer_options?.length ?? 0) > 0 && (
        <details className="dilemma-options">
          <summary>Answer options the judge classifies against ({dilemma!.answer_options!.length})</summary>
          <ul>
            {dilemma!.answer_options!.map((option) => <li key={option}>{option}</li>)}
          </ul>
        </details>
      )}
    </section>
  );
}

function RunResponse({ data, onRetry, run }: { data: RunData | null; onRetry: () => void; run: RunRow | null }) {
  if (!data || data.status === 'loading') return <p className="muted-text">Loading response…</p>;
  if (data.status === 'error') {
    return (
      <div className="comparison-error">
        <span>Could not load this response: {data.error}</span>
        <button type="button" onClick={onRetry}>Retry</button>
      </div>
    );
  }
  return (
    <>
      {data.response
        ? <pre className="simulation-preserve-text dilemma-response-text">{data.response}</pre>
        : <p className="muted-text"><em>No response text stored for this run.</em></p>}
      <JudgeNote judge={data.judge} />
      {run && <p className="dilemma-provenance muted-text">Run {run.version} · {fmtDate(run.created)} · {run.run_id}</p>}
    </>
  );
}

export default function DilemmasViewer() {
  const [manifest, setManifest] = useState<RunsManifest | null>(null);
  const [loadError, setLoadError] = useState('');
  const [dilemmaKey, setDilemmaKey] = useState('');
  const [pickedRunIds, setPickedRunIds] = useState<Record<string, string>>({});
  const [runDocs, setRunDocs] = useState<Record<string, RunData>>({});
  const [fallbackDilemma, setFallbackDilemma] = useState<DilemmaInfo | null>(null);
  const [retryTick, setRetryTick] = useState(0);

  const runCache = useRef<Record<string, RunData>>({});
  const inFlight = useRef<Set<string>>(new Set());
  const autoApplied = useRef(false);

  const bumpRetry = useCallback(() => setRetryTick((tick) => tick + 1), []);

  /* ------------------------------- data ------------------------------- */

  useEffect(() => {
    console.info(`[DilemmasViewer] fetching manifest: ${RUNS_MANIFEST_URL}`);
    // The manifest is the gatekeeper for new runs, so always re-fetch it
    // (0 TTL) rather than serving a stale copy from the localStorage cache.
    cachedFetchJson<RunsManifest>(RUNS_MANIFEST_URL, 0)
      .then((data) => {
        console.info(`[DilemmasViewer] manifest loaded: ${data.runs?.length ?? 0} runs`);
        setManifest(data);
      })
      .catch((error) => {
        console.error('[DilemmasViewer] manifest load failed:', error);
        setLoadError(String(error));
      });
  }, []);

  const runs = useMemo(() => manifest?.runs ?? [], [manifest]);

  const dilemmasList = useMemo<DilemmaInfo[]>(() => {
    if (manifest?.dilemmas?.length) return manifest.dilemmas;
    if (!manifest) return [];
    // Older index without the dilemma inventory: list the keys that have
    // runs; the text is fetched from a run's config once one is selected.
    const byKey = new Map<string, DilemmaInfo>();
    for (const run of manifest.runs) {
      const key = `${run.profile_id}/${run.situation_id}`;
      if (!byKey.has(key)) {
        byKey.set(key, { profile_id: run.profile_id, situation_id: run.situation_id });
      }
    }
    return [...byKey.values()].sort((a, b) =>
      a.profile_id.localeCompare(b.profile_id) || a.situation_id.localeCompare(b.situation_id));
  }, [manifest]);

  const groupedDilemmas = useMemo(() => {
    const groups = new Map<string, DilemmaInfo[]>();
    for (const dilemma of dilemmasList) {
      const list = groups.get(dilemma.profile_id) ?? [];
      list.push(dilemma);
      groups.set(dilemma.profile_id, list);
    }
    return [...groups.entries()];
  }, [dilemmasList]);

  const [profileId, situationId] = dilemmaKey ? splitKey(dilemmaKey) : ['', ''];

  const runsForDilemma = useMemo(
    () => runs
      .filter((run) => run.profile_id === profileId && run.situation_id === situationId)
      .sort((a, b) => a.model.localeCompare(b.model) || a.version - b.version),
    [runs, profileId, situationId],
  );

  const modelsForDilemma = useMemo(
    () => [...new Set(runsForDilemma.map((run) => run.model))].sort(),
    [runsForDilemma],
  );

  const runsForModel = useCallback(
    (model: string) => runsForDilemma.filter((run) => run.model === model),
    [runsForDilemma],
  );

  const currentDilemma = useMemo(
    () => dilemmasList.find((d) => `${d.profile_id}/${d.situation_id}` === dilemmaKey) ?? null,
    [dilemmasList, dilemmaKey],
  );

  const stimulus = useMemo(() => {
    if (currentDilemma?.situation) return currentDilemma;
    if (fallbackDilemma && `${fallbackDilemma.profile_id}/${fallbackDilemma.situation_id}` === dilemmaKey) {
      return fallbackDilemma;
    }
    return currentDilemma;
  }, [currentDilemma, fallbackDilemma, dilemmaKey]);

  // Selected run per model: the explicit pick, else the model's newest run.
  const selectedRuns = useMemo(
    () => modelsForDilemma
      .map((model) => {
        const list = runsForModel(model);
        return list.find((run) => run.run_id === pickedRunIds[model]) ?? list[list.length - 1] ?? null;
      })
      .filter((run): run is RunRow => run != null),
    [modelsForDilemma, runsForModel, pickedRunIds],
  );

  const ensureRun = useCallback(async (run: RunRow) => {
    const cached = runCache.current[run.run_id];
    if (cached && cached.status !== 'error') return;
    if (inFlight.current.has(run.run_id)) return;
    inFlight.current.add(run.run_id);
    const loading: RunData = { status: 'loading' };
    runCache.current[run.run_id] = loading;
    setRunDocs((previous) => ({ ...previous, [run.run_id]: loading }));
    try {
      const base = `${RUNS_BASE}${run.path}/`;
      const [response, judge] = await Promise.all([
        cachedFetchText(`${base}response.txt`),
        cachedFetchJson<JudgeDoc>(`${base}judge.json`).catch(() => null),
      ]);
      // An empty judge file ({} in runs with no score) means "no result",
      // not a judge panel with missing fields.
      const judged = judge && (judge.answer || judge.explanation || judge.metadata) ? judge : null;
      runCache.current[run.run_id] = { status: 'ready', response: response.trim(), judge: judged };
    } catch (error) {
      console.error(`[DilemmasViewer] run load failed (${run.path}):`, error);
      runCache.current[run.run_id] = { status: 'error', error: String(error) };
    } finally {
      inFlight.current.delete(run.run_id);
      setRunDocs((previous) => ({ ...previous, [run.run_id]: runCache.current[run.run_id] }));
    }
  }, []);

  const runData = useCallback(
    (runId: string | undefined): RunData | null => (runId ? runDocs[runId] ?? runCache.current[runId] ?? null : null),
    [runDocs],
  );

  /* ---------------------------- selections ---------------------------- */

  // Initial selection: first dilemma once the manifest loads and no deep
  // link set one.
  useEffect(() => {
    if (!manifest || dilemmaKey || !dilemmasList.length) return;
    const first = dilemmasList.find((d) =>
      runs.some((run) => run.profile_id === d.profile_id && run.situation_id === d.situation_id));
    setDilemmaKey(`${(first ?? dilemmasList[0]).profile_id}/${(first ?? dilemmasList[0]).situation_id}`);
  }, [manifest, dilemmaKey, dilemmasList, runs]);

  // Default each model to its newest exported run of the selected dilemma.
  useEffect(() => {
    if (!modelsForDilemma.length) return;
    setPickedRunIds((previous) => {
      let changed = false;
      const next = { ...previous };
      for (const model of modelsForDilemma) {
        const list = runsForModel(model);
        const valid = next[model] && list.some((run) => run.run_id === next[model]);
        if (!valid && list.length) {
          next[model] = list[list.length - 1].run_id;
          changed = true;
        }
      }
      return changed ? next : previous;
    });
  }, [modelsForDilemma, runsForModel]);

  // Load every selected per-model response (guarded by the cache so this can
  // run on each render).
  useEffect(() => {
    for (const run of selectedRuns) void ensureRun(run);
  }, [selectedRuns, ensureRun, retryTick]);

  // Older manifests have no dilemma inventory; take the text from the first
  // run's config so the stimulus can still be rendered.
  useEffect(() => {
    if (!manifest || manifest.dilemmas?.length || !runsForDilemma.length) {
      setFallbackDilemma(null);
      return;
    }
    let cancelled = false;
    cachedFetchJson<{ dilemma?: DilemmaInfo }>(`${RUNS_BASE}${runsForDilemma[0].path}/config.json`, 0)
      .then((config) => { if (!cancelled && config.dilemma) setFallbackDilemma(config.dilemma); })
      .catch((error) => console.info('[DilemmasViewer] dilemma fallback load failed:', String(error)));
    return () => { cancelled = true; };
  }, [manifest, runsForDilemma]);

  // Deep link: ?dilemma=<profile_id>-<situation_id>
  useEffect(() => {
    if (!manifest || autoApplied.current) return;
    autoApplied.current = true;
    const dilemmaParam = urlParams.get('dilemma');
    if (dilemmaParam) {
      const dash = dilemmaParam.indexOf('-');
      if (dash > 0) {
        const key = `${dilemmaParam.slice(0, dash)}/${dilemmaParam.slice(dash + 1)}`;
        if (dilemmasList.some((d) => `${d.profile_id}/${d.situation_id}` === key)) {
          setDilemmaKey(key);
        }
      }
    }
  }, [manifest, dilemmasList]);

  // Keep the URL shareable (replaceState: no history spam).
  useEffect(() => {
    if (typeof window === 'undefined' || !manifest) return;
    const params = new URLSearchParams(window.location.search);
    if (dilemmaKey) params.set('dilemma', dilemmaKey.replace('/', '-'));
    else params.delete('dilemma');
    const query = params.toString();
    const url = `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`;
    if (url !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
      window.history.replaceState(null, '', url);
    }
  });

  const selectDilemma = (key: string) => {
    setDilemmaKey(key);
  };

  /* ------------------------------ rendering ---------------------------- */

  if (loadError) {
    return (
      <div style={{ padding: '3rem 0' }}>
        <div style={{ ...card, borderColor: '#fca5a5', color: '#b91c1c' }}>
          <strong>Could not load dilemma data</strong>
          <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>{loadError}</p>
          <p style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
            The selected source is <code>{RUNS_MANIFEST_URL}</code>. {MANIFEST_IS_LOCAL
              ? 'No local dilemma manifest was found. Run uv run python code/dilemmas/export_results.py to populate artifacts/dilemmas/runs/index.json.'
              : 'Check the public bucket link in the data-source notice below.'}
          </p>
        </div>
        <DataSourceNotice manifestUrl={RUNS_MANIFEST_URL} kind="dilemma" />
      </div>
    );
  }

  if (!manifest) {
    return <div style={{ padding: '3rem 0', color: 'var(--color-muted)' }}>Loading dilemma data from the selected source…</div>;
  }

  return (
    <div>
      <DataSourceNotice bucket={manifest.bucket || DEFAULT_BUCKET} manifestUrl={RUNS_MANIFEST_URL} kind="dilemma" />
      {manifest.generated_at && <p className="manifest-generated muted-text">Manifest generated {new Date(manifest.generated_at).toLocaleString()}</p>}
      <div style={{ marginBottom: '1.5rem', padding: '1rem 1.25rem', background: '#fff7ed', borderRadius: '0.6rem', border: '2px solid #f59e0b', fontSize: '0.9rem', fontWeight: 600, color: '#9a3412', lineHeight: 1.6 }}>
        ⚠️ <strong>Development results.</strong> This viewer exposes the raw dilemma artifacts currently present in the selected local or bucket manifest. Treat them as pipeline results, not as pilot/final study conclusions.
      </div>

      <div className="results-controls">
        <div>
          <label style={label}>Dilemma</label>
          <select value={dilemmaKey} onChange={(e) => selectDilemma(e.target.value)} style={select}>
            {groupedDilemmas.map(([id, list]) => (
              <optgroup key={id} label={profileLabel(manifest, id)}>
                {list.map((dilemma) => (
                  <option key={`${dilemma.profile_id}/${dilemma.situation_id}`} value={`${dilemma.profile_id}/${dilemma.situation_id}`}>
                    {dilemmaLabel(dilemma)}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </div>
        <span className="muted-text" style={{ paddingBottom: '0.4rem', fontSize: '0.8rem' }}>
          {modelsForDilemma.length} model{modelsForDilemma.length === 1 ? '' : 's'} · {runsForDilemma.length} response{runsForDilemma.length === 1 ? '' : 's'}
        </span>
      </div>

      <DilemmaStimulus
        dilemma={stimulus}
        fallbackId={dilemmaKey}
        profileName={stimulus ? profileLabel(manifest, stimulus.profile_id) : profileLabel(manifest, profileId)}
        roleSummary={manifest.profiles?.[stimulus?.profile_id ?? profileId]?.role_summary}
        systemPrompt={manifest.profiles?.[stimulus?.profile_id ?? profileId]?.system_prompt}
      />

      <OutcomeChart dilemma={stimulus} />

      {modelsForDilemma.length ? (
        <div className="dilemma-response-grid">
          {modelsForDilemma.map((model) => {
            const list = runsForModel(model);
            const selected = selectedRuns.find((run) => run.model === model) ?? null;
            return (
              <article key={model} className="dilemma-panel">
                <div className="dilemma-panel__head">
                  <div>
                    <p className="dilemma-panel__kicker">Model</p>
                    <p className="dilemma-panel__model">{shortModel(model)}</p>
                  </div>
                  <div>
                    <label style={label}>Response to use</label>
                    <select
                      style={{ ...select, minWidth: '12rem' }}
                      value={selected?.run_id ?? ''}
                      onChange={(e) => setPickedRunIds((previous) => ({ ...previous, [model]: e.target.value }))}
                    >
                      {list.map((run) => (
                        <option key={run.run_id} value={run.run_id}>
                          Run {run.version} · {fmtDate(run.created) || '?'}
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
                <RunResponse data={runData(selected?.run_id)} onRetry={bumpRetry} run={selected} />
              </article>
            );
          })}
        </div>
      ) : (
        <div style={card}>No exported responses were found for this dilemma.</div>
      )}
    </div>
  );
}
