import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { cachedFetchJson, cachedFetchText } from '../lib/cachedFetch';
import DataSourceNotice from './DataSourceNotice';
import {
  DEFAULT_BUCKET,
  LOCAL_DILEMMAS_BASE,
  dilemmaComparisonsBaseUrl,
  isLocalSourceUrl,
  manifestUrl,
  urlParams,
} from '../lib/dataSource';
import {
  SURVEY_AI_USE_QUESTION,
  SURVEY_AI_USE_SCALE,
  SURVEY_PREFERENCE_QUESTION,
  SURVEY_PREFERENCE_SCALE,
} from '../lib/survey';

/* Dilemma runs viewer.
 *
 * Data lives in the project's public HF storage bucket (or the local
 * artifacts mirror in dev):
 *   dilemmas/runs/index.json         - profiles, dilemmas and every run
 *   dilemmas/runs/<path>/response.txt
 *   dilemmas/runs/<path>/judge.json
 *   dilemmas/comparisons/index.json  - the stored pairwise comparisons
 *   dilemmas/comparisons/<pair>/<scenario>/<id>.json
 *
 * Two modes: every model's response to one dilemma side by side (with a
 * per-model run selector), and a survey-style read of one stored comparison
 * summary with its two source responses.
 */

const RUNS_MANIFEST_URL = manifestUrl(LOCAL_DILEMMAS_BASE, 'dilemmas/runs/index.json');
const MANIFEST_IS_LOCAL = isLocalSourceUrl(RUNS_MANIFEST_URL);
const RUNS_BASE = MANIFEST_IS_LOCAL
  ? LOCAL_DILEMMAS_BASE
  : RUNS_MANIFEST_URL.replace(/index\.json(?:\?.*)?$/, '');
const COMPARISONS_BASE = dilemmaComparisonsBaseUrl(RUNS_MANIFEST_URL);

interface DilemmaInfo {
  profile_id: string;
  situation_id: string;
  tension?: number | null;
  tension_name?: string;
  situation?: string;
  question?: string;
  answer_options?: string[];
  value_options?: string[];
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
  profiles?: Record<string, { name?: string }>;
  dilemmas?: DilemmaInfo[];
  models?: string[];
  runs: RunRow[];
}

interface ComparisonRow {
  comparison_id: string;
  profile_id: string;
  situation_id: string;
  model_pair: string;
  scenario: string;
  agent1_run_id: string;
  agent1_model: string;
  agent2_run_id: string;
  agent2_model: string;
  summary_model?: string;
}

interface ComparisonsIndex {
  schema: string;
  comparisons: ComparisonRow[];
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

interface ComparisonAgent {
  run_id: string;
  model: string;
  response: string;
}

interface ComparisonDoc {
  schema: string;
  comparison_id: string;
  scenario: {
    profile_id: string;
    situation_id: string;
    dilemma?: DilemmaInfo;
    tension?: { name?: string; guidance?: string };
  };
  agents: Record<string, ComparisonAgent>;
  summary: { model: string; text: string };
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

const modeButton = (active: boolean): React.CSSProperties => ({
  padding: '0.45rem 1rem', borderRadius: '0.5rem',
  border: `1px solid ${active ? 'var(--color-primary)' : 'var(--color-border)'}`,
  background: active ? 'var(--color-primary)' : 'white',
  color: active ? 'white' : 'var(--color-text)',
  cursor: 'pointer', fontSize: '0.85rem', fontFamily: 'inherit', fontWeight: active ? 700 : 400,
});

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

function DilemmaStimulus({ dilemma, fallbackId, profileName }: {
  dilemma: DilemmaInfo | null;
  fallbackId: string;
  profileName: string;
}) {
  return (
    <section className="dilemma-stimulus" aria-label="Dilemma">
      <p className="dilemma-stimulus__meta">
        <span>{profileName}</span>
        {dilemma?.tension_name && <span>Value tension: {dilemma.tension_name}</span>}
        <span className="muted-text">{dilemma?.situation_id ?? fallbackId}</span>
      </p>
      {dilemma?.situation
        ? <p className="dilemma-stimulus__situation">{dilemma.situation}</p>
        : <p className="muted-text">Loading the dilemma text…</p>}
      {dilemma?.question && <p className="dilemma-stimulus__question">{dilemma.question}</p>}
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

/* The comparison panel is the survey-style read of one stored comparison:
 * the dilemma stimulus, the summary paragraph the survey-style block would
 * show ("this is what happened"), the two preference questions, and the two
 * source responses with their judge classifications. */
function DilemmaComparisonPanel({ row, doc, error, onRetry, runData, runById }: {
  row: ComparisonRow;
  doc: ComparisonDoc | null;
  error: string;
  onRetry: () => void;
  runData: (runId: string | undefined) => RunData | null;
  runById: Map<string, RunRow>;
}) {
  const dilemma = doc?.scenario?.dilemma;
  const agents: Array<[string, ComparisonAgent | undefined]> = [
    ['Agent 1', doc?.agents?.['Agent 1']],
    ['Agent 2', doc?.agents?.['Agent 2']],
  ];
  return (
    <div style={{ ...card, marginBottom: '1.5rem', background: 'var(--color-surface)' }}>
      <div className="dilemma-comparison__header">
        <p className="dilemma-comparison__kicker">⚖ Cross-model comparison</p>
        {doc && <span className="muted-text">summary by {shortModel(doc.summary.model)}</span>}
        <span className="muted-text">
          Agent 1 = {shortModel(row.agent1_model)} · Agent 2 = {shortModel(row.agent2_model)}
        </span>
      </div>
      {!doc && error && (
        <div className="comparison-error">
          <span>Could not load the stored comparison: {error}</span>
          <button type="button" onClick={onRetry}>Retry</button>
        </div>
      )}
      {!doc && !error && <p className="muted-text">Loading comparison…</p>}
      {doc && (
        <>
          <section className="survey-preview" aria-label="Comparison block as respondents would see it">
            <p className="survey-preview__kicker">Survey block — as respondents would see it</p>
            {dilemma?.situation && (
              <p className="survey-preview__text"><strong>Situation:</strong> {dilemma.situation}</p>
            )}
            {dilemma?.question && (
              <p className="survey-preview__text"><strong>Question:</strong> {dilemma.question}</p>
            )}
            <p className="survey-preview__text"><strong>This is what happened:</strong></p>
            <div className="survey-preview__summary">
              <pre className="simulation-preserve-text dilemma-summary-text">{doc.summary.text || ''}</pre>
            </div>
            <div className="survey-preview__question-block">
              <p className="survey-preview__question">{SURVEY_PREFERENCE_QUESTION}</p>
              <ul className="survey-preview__scale">
                {SURVEY_PREFERENCE_SCALE.map((option) => <li key={option}>{option}</li>)}
              </ul>
            </div>
            <div className="survey-preview__question-block">
              <p className="survey-preview__question">{SURVEY_AI_USE_QUESTION}</p>
              <ul className="survey-preview__scale">
                {SURVEY_AI_USE_SCALE.map((option) => <li key={option}>{option}</li>)}
              </ul>
            </div>
          </section>
          <p className="survey-preview-note muted-text">
            The survey shows both agents anonymously (Agent 1 / Agent 2). The paragraph above is
            stored in this comparison file; the responses and judge results below load from the
            currently exported runs.
          </p>
          <div className="dilemma-agent-grid">
            {agents.map(([name, agent], index) => {
              const judgeData = agent?.run_id ? runData(agent.run_id) : null;
              const runKnown = agent?.run_id ? runById.has(agent.run_id) : false;
              return (
                <article key={name} className={`dilemma-agent dilemma-agent--${index === 0 ? 'a' : 'b'}`}>
                  <p className="dilemma-agent__kicker">{name}</p>
                  <p className="dilemma-agent__model">{agent ? shortModel(agent.model) : shortModel(index === 0 ? row.agent1_model : row.agent2_model)}</p>
                  {!agent && <p className="muted-text">This agent's response is not stored in the comparison file.</p>}
                  {agent?.response && (
                    <pre className="simulation-preserve-text dilemma-response-text">{agent.response}</pre>
                  )}
                  {agent?.run_id && (!runKnown
                    ? <p className="muted-text dilemma-judge__empty">This run is not in the current runs export, so the judge result is unavailable.</p>
                    : judgeData?.status === 'ready'
                      ? <JudgeNote judge={judgeData.judge} />
                      : judgeData?.status === 'error'
                        ? <p className="muted-text dilemma-judge__empty">Judge result unavailable for this run in the current export.</p>
                        : <p className="muted-text dilemma-judge__empty">Loading judge result…</p>)}
                </article>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}

export default function DilemmasViewer() {
  const [manifest, setManifest] = useState<RunsManifest | null>(null);
  const [compRows, setCompRows] = useState<ComparisonRow[] | null>(null);
  const [loadError, setLoadError] = useState('');
  const [mode, setMode] = useState<'inline' | 'compare'>('inline');
  const [dilemmaKey, setDilemmaKey] = useState('');
  const [pickedRunIds, setPickedRunIds] = useState<Record<string, string>>({});
  const [runDocs, setRunDocs] = useState<Record<string, RunData>>({});
  const [fallbackDilemma, setFallbackDilemma] = useState<DilemmaInfo | null>(null);
  const [comparisonPick, setComparisonPick] = useState<string | null>(null);
  const [comparisonDoc, setComparisonDoc] = useState<ComparisonDoc | null>(null);
  const [comparisonError, setComparisonError] = useState('');
  const [retryTick, setRetryTick] = useState(0);

  const runCache = useRef<Record<string, RunData>>({});
  const inFlight = useRef<Set<string>>(new Set());
  const comparisonCache = useRef<Record<string, ComparisonDoc>>({});
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
    // The comparison index is optional: no comparisons exist until
    // compare.py has been run against the exported responses.
    cachedFetchJson<ComparisonsIndex>(`${COMPARISONS_BASE}index.json`, 0)
      .then((index) => setCompRows(index.comparisons ?? []))
      .catch((error) => {
        console.info('[DilemmasViewer] no comparison index:', String(error));
        setCompRows([]);
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

  const runById = useMemo(() => {
    const byId = new Map<string, RunRow>();
    for (const run of runs) if (run.run_id) byId.set(run.run_id, run);
    return byId;
  }, [runs]);

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

  const comparisonsForDilemma = useMemo(() => {
    if (!compRows || !dilemmaKey) return [];
    return compRows
      .filter((row) => row.profile_id === profileId && row.situation_id === situationId && row.comparison_id)
      .sort((a, b) => a.comparison_id.localeCompare(b.comparison_id));
  }, [compRows, dilemmaKey, profileId, situationId]);

  const selectedComparison = useMemo(() => {
    if (!comparisonsForDilemma.length) return null;
    if (comparisonPick) {
      const byId = comparisonsForDilemma.find((row) => row.comparison_id === comparisonPick);
      if (byId) return byId;
    }
    return comparisonsForDilemma[0];
  }, [comparisonsForDilemma, comparisonPick]);

  useEffect(() => {
    if (mode !== 'compare' || !selectedComparison) {
      setComparisonDoc(null);
      setComparisonError('');
      return;
    }
    const key = selectedComparison.comparison_id;
    const cached = comparisonCache.current[key];
    if (cached) {
      setComparisonDoc(cached);
      setComparisonError('');
      return;
    }
    setComparisonDoc(null);
    setComparisonError('');
    let cancelled = false;
    const url = `${COMPARISONS_BASE}${selectedComparison.model_pair}/${selectedComparison.scenario}/${key}.json`;
    cachedFetchJson<ComparisonDoc>(url, 0)
      .then((doc) => {
        comparisonCache.current[key] = doc;
        if (!cancelled) setComparisonDoc(doc);
      })
      .catch((error) => {
        console.error(`[DilemmasViewer] comparison load failed: ${key}`, error);
        if (!cancelled) setComparisonError(String(error));
      });
    return () => { cancelled = true; };
  }, [mode, selectedComparison, retryTick]);

  // Load the two compared runs so their judge results can be shown next to
  // the stored responses.
  useEffect(() => {
    if (mode !== 'compare' || !selectedComparison) return;
    for (const runId of [selectedComparison.agent1_run_id, selectedComparison.agent2_run_id]) {
      const run = runId ? runById.get(runId) : undefined;
      if (run) void ensureRun(run);
    }
  }, [mode, selectedComparison, runById, ensureRun, retryTick]);

  // Deep links (applied once both indexes are in):
  //   ?dilemma=<profile_id>-<situation_id>
  //   ?comparison=<comparison_id>          -> comparison mode, that pair
  //   ?mode=compare
  useEffect(() => {
    if (!manifest || compRows === null || autoApplied.current) return;
    autoApplied.current = true;
    const comparisonParam = urlParams.get('comparison');
    if (comparisonParam) {
      const row = compRows.find((item) => item.comparison_id === comparisonParam);
      if (row?.profile_id && row?.situation_id) {
        setDilemmaKey(`${row.profile_id}/${row.situation_id}`);
        setMode('compare');
        setComparisonPick(row.comparison_id);
        return;
      }
    }
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
    if (urlParams.get('mode') === 'compare') setMode('compare');
  }, [manifest, compRows, dilemmasList]);

  // Keep the URL shareable (replaceState: no history spam).
  useEffect(() => {
    if (typeof window === 'undefined' || !manifest) return;
    const params = new URLSearchParams(window.location.search);
    const setOrDel = (key: string, value?: string | null) => {
      if (value) params.set(key, value); else params.delete(key);
    };
    setOrDel('dilemma', dilemmaKey ? dilemmaKey.replace('/', '-') : null);
    setOrDel('mode', mode === 'compare' ? 'compare' : null);
    setOrDel('comparison', mode === 'compare' ? selectedComparison?.comparison_id ?? null : null);
    const query = params.toString();
    const url = `${window.location.pathname}${query ? `?${query}` : ''}${window.location.hash}`;
    if (url !== `${window.location.pathname}${window.location.search}${window.location.hash}`) {
      window.history.replaceState(null, '', url);
    }
  });

  const selectDilemma = (key: string) => {
    setDilemmaKey(key);
    setComparisonPick(null);
    setComparisonDoc(null);
    setComparisonError('');
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
          <label style={label}>View</label>
          <div style={{ display: 'flex', gap: '0.4rem' }}>
            {[{ id: 'inline' as const, label: 'Inline responses' }, { id: 'compare' as const, label: 'Comparison' }].map((item) => (
              <button key={item.id} onClick={() => setMode(item.id)} style={modeButton(mode === item.id)}>
                {item.label}
              </button>
            ))}
          </div>
        </div>
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
      />

      {mode === 'inline' && (
        modelsForDilemma.length ? (
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
        )
      )}

      {mode === 'compare' && (
        compRows === null ? (
          <p className="muted-text">Loading comparisons…</p>
        ) : comparisonsForDilemma.length === 0 ? (
          <div style={card}>
            No stored comparisons exist for this dilemma in the selected source. Generate them
            with <code>uv run python code/dilemmas/compare.py</code>, then sync or refresh — the
            viewer lists every cross-model pair here.
          </div>
        ) : (
          <>
            <div style={{ ...card, marginBottom: '1rem', background: 'var(--color-surface)' }}>
              <label style={label}>
                Comparison — {comparisonsForDilemma.length} stored pair{comparisonsForDilemma.length === 1 ? '' : 's'} for this dilemma
              </label>
              <select
                value={selectedComparison?.comparison_id ?? ''}
                onChange={(e) => setComparisonPick(e.target.value)}
                style={select}
              >
                {comparisonsForDilemma.map((row) => {
                  const versionA = runById.get(row.agent1_run_id)?.version;
                  const versionB = runById.get(row.agent2_run_id)?.version;
                  return (
                    <option key={row.comparison_id} value={row.comparison_id}>
                      {shortModel(row.agent1_model)} (run {versionA ?? '?'}) vs {shortModel(row.agent2_model)} (run {versionB ?? '?'})
                    </option>
                  );
                })}
              </select>
            </div>
            {selectedComparison && (
              <DilemmaComparisonPanel
                row={selectedComparison}
                doc={comparisonDoc}
                error={comparisonError}
                onRetry={bumpRetry}
                runData={runData}
                runById={runById}
              />
            )}
          </>
        )
      )}
    </div>
  );
}
