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

/* The survey block's framing (survey/comparison-question.md): role lead,
 * the profile's dilemma intro, the situation and the question, then the
 * answers. A survey block always shows exactly two answers, so its stored
 * outro ("Two different agents answered this question") is not shown here -
 * the viewer can show every model with runs. The question lead is the
 * plural-neutral fallback for profiles without an intro. */
const VIEWER_QUESTION_LEAD = 'They were all asked:';

/* Dilemma runs viewer.
 *
 * Data lives in the project's public HF storage bucket (or the local
 * artifacts mirror in dev), one manifest per exported full-run batch:
 *   dilemmas/runs/batches.json             - every exported batch
 *   dilemmas/runs/full/<run-id>/index.json - that batch's profiles,
 *                                            dilemmas and runs
 *   dilemmas/runs/full/<run-id>/<path>/response.txt
 *   dilemmas/runs/full/<run-id>/<path>/judge.json
 *
 * There is no merged/live view: the viewer defaults to the newest batch and
 * the Batch selector loads one batch's own manifest at a time. Every model's
 * response to one dilemma side by side, with a per-model run selector. The
 * selected profile's system prompt is shown collapsed under the dilemma
 * stimulus.
 */

const RUNS_MANIFEST_URL = manifestUrl(LOCAL_DILEMMAS_BASE, 'dilemmas/runs/index.json');
const MANIFEST_IS_LOCAL = isLocalSourceUrl(RUNS_MANIFEST_URL);
const RUNS_BASE = MANIFEST_IS_LOCAL
  ? LOCAL_DILEMMAS_BASE
  : RUNS_MANIFEST_URL.replace(/index\.json(?:\?.*)?$/, '');

interface DilemmaInfo {
  profile_id: string;
  situation_id: string;
  situation?: string;
  question?: string;
  answer_options?: string[];
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
  /** Per-batch exports only: the bundle this manifest is the export of. */
  run_id?: string;
  profiles?: Record<
    string,
    {
      name?: string;
      role_summary?: string;
      dilemma_intro?: string;
      dilemma_outro?: string;
      system_prompt?: string;
    }
  >;
  dilemmas?: DilemmaInfo[];
  models?: string[];
  runs: RunRow[];
}

/** One exported full-run bundle (see code/dilemmas/full_run.py). */
interface BatchInfo {
  run_id: string;
  generated_at?: string;
  models?: string[];
  runs?: number;
  /** Set while the batch is still running (exported with --partial). */
  partial?: boolean;
}

interface BatchesDoc {
  schema?: string;
  generated_at?: string;
  batches: BatchInfo[];
}

interface JudgeDoc {
  value?: number | null;
  answer?: string;
  explanation?: string;
  metadata?: {
    choice?: string;
    value?: string;
    reason?: string;
    quote?: string;
    reasoning_source?: string;
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
 * models - so spot-test extras (haiku, gpt-6-luna) and one-off adapters do
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

/* Dilemma-level statistics for the all-dilemmas table, computed from the
 * same choice_counts the outcome chart plots, over the same curated series,
 * with the definitions the report analysis uses (code/dilemmas/analyze.py,
 * spot_test.py):
 *   Balance         H / ln(k) over the pooled shares of every run (1.00 =
 *                   runs spread evenly across the options, 0.00 = every run
 *                   on one option).
 *   Balance (not
 *   other)          the same H / ln(k) pooled over the decided options only:
 *                   the "Other or unclear" catch-all is dropped and the
 *                   remaining shares renormalised, so a dilemma that looks
 *                   balanced only because runs fall on the catch-all reads as
 *                   what it is. Null when there are fewer than two other
 *                   options or every run was "Other or unclear".
 *   Avg divergence  mean pairwise total variation distance between the
 *                   models' outcome distributions (0.00 = identical,
 *                   1.00 = no shared option).
 *   Avg stability   mean over models of 1 - H / ln(k) within that model's
 *                   own runs (1.00 = same option every time).
 * A metric is null when it cannot be computed: no judged runs, a single
 * answer option (ln k = 0) or one model only (no pairs to average). */

function normalisedEntropy(shares: number[]): number {
  const entropy = -shares.reduce((sum, p) => sum + (p > 0 ? p * Math.log(p) : 0), 0);
  return shares.length > 1 ? entropy / Math.log(shares.length) : 0;
}

/* The "Other or unclear" catch-all ("not other" metrics drop it). Matched on
 * the option text so it also holds for older manifests with other labels. */
function otherOptionIndex(options: string[]): number {
  return options.findIndex((option) => /^other\b/i.test(option.trim()));
}

interface DilemmaStats {
  runs: number;
  balance: number | null;
  balanceNoOther: number | null;
  divergence: number | null;
  stability: number | null;
}

function dilemmaStats(dilemma: DilemmaInfo): DilemmaStats {
  const options = dilemma.answer_options ?? [];
  const counts = dilemma.choice_counts ?? {};
  const perModel = outcomeSeries(dilemma)
    .map((model) => options.map((option) => counts[model]?.[option] ?? 0))
    .filter((row) => row.some((count) => count > 0));
  const rowTotal = (row: number[]) => row.reduce((sum, count) => sum + count, 0);
  const runs = perModel.reduce((sum, row) => sum + rowTotal(row), 0);
  if (!runs || options.length < 2) {
    return { runs, balance: null, balanceNoOther: null, divergence: null, stability: null };
  }
  const shares = perModel.map((row) => row.map((count) => count / rowTotal(row)));
  const pooled = options.map((_, index) => perModel.reduce((sum, row) => sum + row[index], 0) / runs);
  // Drop the catch-all and renormalise: entropy is not scale-invariant, so
  // the remaining shares have to sum to one before scoring.
  const otherIndex = otherOptionIndex(options);
  const decided = pooled.filter((_, index) => index !== otherIndex);
  const decidedTotal = decided.reduce((sum, share) => sum + share, 0);
  const balanceNoOther = otherOptionIndex(options) >= 0 && decided.length > 1 && decidedTotal > 0
    ? normalisedEntropy(decided.map((share) => share / decidedTotal))
    : null;
  const pairTvds = shares.flatMap((left, position) =>
    shares.slice(position + 1).map((right) =>
      0.5 * left.reduce((sum, share, index) => sum + Math.abs(share - right[index]), 0)));
  return {
    runs,
    balance: normalisedEntropy(pooled),
    balanceNoOther,
    divergence: pairTvds.length ? pairTvds.reduce((sum, value) => sum + value, 0) / pairTvds.length : null,
    stability: shares.reduce((sum, row) => sum + 1 - normalisedEntropy(row), 0) / shares.length,
  };
}

/* All-dilemma statistics table: one row per dilemma in the manifest. The
 * columns are sortable (click a heading; the arrow shows the active column
 * and direction), defaulting to average divergence descending so the
 * dilemmas where the models split most sit at the top. The table doubles as
 * navigation: clicking a row selects that dilemma for the rest of the viewer
 * and collapses the table so the newly selected dilemma's view is in place.
 * The row for the selected dilemma is highlighted. */

type StatsSortKey = 'profile' | 'dilemma' | 'runs' | 'balance' | 'balanceNoOther' | 'divergence' | 'stability';

/* Numeric columns start largest-first (the old fixed divergence order), text
 * columns A→Z; clicking the active column flips the direction. */
const STATS_SORT_COLUMNS: Array<{ key: StatsSortKey; label: string; numeric: boolean }> = [
  { key: 'profile', label: 'Profile', numeric: false },
  { key: 'dilemma', label: 'Dilemma', numeric: false },
  { key: 'runs', label: 'Runs', numeric: true },
  { key: 'balance', label: 'Balance', numeric: true },
  { key: 'balanceNoOther', label: 'Balance (not other)', numeric: true },
  { key: 'divergence', label: 'Avg divergence', numeric: true },
  { key: 'stability', label: 'Avg stability', numeric: true },
];

function DilemmaStatsTable({ dilemmas, manifest, currentKey, onSelect }: {
  dilemmas: DilemmaInfo[];
  manifest: RunsManifest;
  currentKey: string;
  onSelect: (key: string) => void;
}) {
  const [sort, setSort] = useState<{ key: StatsSortKey; desc: boolean }>({ key: 'divergence', desc: true });
  const toggleSort = (key: StatsSortKey) => {
    setSort((previous) => (
      previous.key === key
        ? { key, desc: !previous.desc }
        : { key, desc: STATS_SORT_COLUMNS.find((column) => column.key === key)!.numeric }
    ));
  };
  const rows = useMemo(() => {
    const value = (row: { dilemma: DilemmaInfo; stats: DilemmaStats }): string | number | null => {
      switch (sort.key) {
        case 'profile': return profileLabel(manifest, row.dilemma.profile_id);
        // Sort by the full dilemma text, not the 90-character table label.
        case 'dilemma': return row.dilemma.question || row.dilemma.situation || row.dilemma.situation_id;
        case 'runs': return row.stats.runs || null;
        default: return row.stats[sort.key];
      }
    };
    const factor = sort.desc ? -1 : 1;
    return dilemmas
      .map((dilemma) => ({ dilemma, stats: dilemmaStats(dilemma) }))
      .sort((a, b) => {
        const left = value(a);
        const right = value(b);
        // A null metric (uncomputable, or no runs) always sorts last.
        if (left == null || right == null) {
          return (left == null ? 1 : 0) - (right == null ? 1 : 0);
        }
        const primary = typeof left === 'string' || typeof right === 'string'
          ? String(left).localeCompare(String(right), undefined, { sensitivity: 'base' })
          : left - right;
        return factor * primary
          || a.dilemma.profile_id.localeCompare(b.dilemma.profile_id)
          || a.dilemma.situation_id.localeCompare(b.dilemma.situation_id);
      });
  }, [dilemmas, manifest, sort]);
  // Controlled so that picking a dilemma can collapse the table and leave
  // the reader on that dilemma's stimulus, chart and responses.
  const [open, setOpen] = useState(false);
  // Older manifests without the inventory have no answer options or counts,
  // so no row can be scored; hide the table rather than showing dashes.
  if (!rows.some((row) => row.stats.runs > 0)) return null;
  const fixed = (value: number | null) => (value == null ? '—' : value.toFixed(2));
  const pick = (key: string) => {
    onSelect(key);
    setOpen(false);
  };
  return (
    <details
      className="dilemma-stats"
      open={open}
      onToggle={(event) => setOpen(event.currentTarget.open)}
    >
      <summary>
        Dilemma statistics
        <span>{rows.length} dilemmas · balance, balance (not other), divergence, stability · click a column to sort, a row to open that dilemma</span>
      </summary>
      <p className="muted-text dilemma-stats__note">
        Over the runs the outcome chart plots: <strong>Balance</strong> is H / ln(k) across every
        run's option (1.00 = runs spread evenly over the options, 0.00 = every run on one);{' '}
        <strong>Balance (not other)</strong> is the same H / ln(k) over the decided options only
        ("Other or unclear" dropped and the shares renormalised), so balance earned by the
        catch-all is not counted as the models splitting;{' '}
        <strong>avg divergence</strong> is the mean pairwise total variation distance between the
        models' outcome distributions (0.00 = identical, 1.00 = no shared option);{' '}
        <strong>avg stability</strong> is the mean over models of 1 - H / ln(k) within that
        model's own runs (1.00 = same option every time).
      </p>
      <div className="dilemma-stats__scroll">
        <table className="dilemma-stats__table">
          <thead>
            <tr>
              {STATS_SORT_COLUMNS.map((column) => {
                const active = sort.key === column.key;
                return (
                  <th
                    key={column.key}
                    className={column.numeric ? 'dilemma-stats__number' : undefined}
                    aria-sort={active ? (sort.desc ? 'descending' : 'ascending') : undefined}
                  >
                    <button
                      type="button"
                      className={active ? 'dilemma-stats__sort dilemma-stats__sort--active' : 'dilemma-stats__sort'}
                      onClick={() => toggleSort(column.key)}
                      title={`Sort by ${column.label.toLowerCase()}`}
                    >
                      {column.label}
                      <span className="dilemma-stats__sort-arrow" aria-hidden="true">
                        {active ? (sort.desc ? '▼' : '▲') : '↕'}
                      </span>
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {rows.map(({ dilemma, stats }) => {
              const key = `${dilemma.profile_id}/${dilemma.situation_id}`;
              const fullText = dilemma.question || dilemma.situation || dilemma.situation_id;
              return (
                <tr
                  key={key}
                  className={key === currentKey ? 'dilemma-stats__row--current' : undefined}
                  onClick={() => pick(key)}
                >
                  <td>{profileLabel(manifest, dilemma.profile_id)}</td>
                  <td>
                    <button type="button" className="dilemma-stats__link" title={fullText}>
                      {dilemmaLabel(dilemma)}
                    </button>
                  </td>
                  <td className="dilemma-stats__number">{stats.runs || '—'}</td>
                  <td className="dilemma-stats__number">{fixed(stats.balance)}</td>
                  <td className="dilemma-stats__number">{fixed(stats.balanceNoOther)}</td>
                  <td className="dilemma-stats__number">{fixed(stats.divergence)}</td>
                  <td className="dilemma-stats__number">{fixed(stats.stability)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </details>
  );
}

/* The judge file written by export_results.py holds the option the judge
 * matched to the response, its one-line reason and a verbatim quote showing
 * what the assistant valued. Older files may carry only answer/explanation. */
function JudgeNote({ judge }: { judge: JudgeDoc | null | undefined }) {
  if (!judge) return <p className="muted-text dilemma-judge__empty">No judge result is stored for this run.</p>;
  const choice = judge.metadata?.choice ?? judge.answer ?? '—';
  const reason = judge.metadata?.reason ?? judge.explanation;
  const quote = judge.metadata?.quote;
  return (
    <div className="dilemma-judge">
      <p className="dilemma-judge__title">⚖ Judge classification</p>
      <p className="dilemma-judge__line"><strong>Decision:</strong> {choice}</p>
      {reason && <p className="dilemma-judge__reason">{reason}</p>}
      {quote && <blockquote className="dilemma-judge__quote">“{quote}”</blockquote>}
    </div>
  );
}

function DilemmaStimulus({ dilemma, fallbackId, profileName, roleSummary, dilemmaIntro, systemPrompt }: {
  dilemma: DilemmaInfo | null;
  fallbackId: string;
  profileName: string;
  roleSummary?: string;
  dilemmaIntro?: string;
  systemPrompt?: string;
}) {
  return (
    <section className="dilemma-stimulus" aria-label="Dilemma">
      <p className="dilemma-stimulus__meta">
        <span>{profileName}</span>
        <span className="muted-text">{dilemma?.situation_id ?? fallbackId}</span>
      </p>
      <p className="dilemma-stimulus__lead">{SURVEY_ROLE_LEAD}</p>
      {roleSummary && <p className="dilemma-stimulus__role">{roleSummary}</p>}
      {dilemmaIntro && <p className="dilemma-stimulus__intro">{dilemmaIntro}</p>}
      {dilemma?.situation && (
        <p className="dilemma-stimulus__situation">{dilemma.situation}</p>
      )}
      {dilemma?.question ? (
        <>
          {!dilemmaIntro && <p className="dilemma-stimulus__lead">{VIEWER_QUESTION_LEAD}</p>}
          <p className="dilemma-stimulus__question">{dilemma.question}</p>
        </>
      ) : (
        <p className="muted-text">Loading the dilemma text…</p>
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
  const [batches, setBatches] = useState<BatchesDoc | null>(null);
  const [batchesReady, setBatchesReady] = useState(false);
  const [selectedBatch, setSelectedBatch] = useState('');
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

  /* One batch at a time: the selected batch's own manifest. There is no
   * merged/live manifest. */
  const manifestUrl = selectedBatch ? `${RUNS_BASE}full/${selectedBatch}/index.json` : '';
  const runsBase = selectedBatch ? `${RUNS_BASE}full/${selectedBatch}/` : `${RUNS_BASE}full/`;

  useEffect(() => {
    console.info(`[DilemmasViewer] fetching batches: ${RUNS_BASE}batches.json`);
    cachedFetchJson<BatchesDoc>(`${RUNS_BASE}batches.json`, 0)
      .then((data) => {
        console.info(`[DilemmasViewer] batches loaded: ${data.batches?.length ?? 0}`);
        setBatches(data);
        // Default to the newest batch (batches.json is newest-first).
        setSelectedBatch((current) => current || data.batches?.find((batch) => batch.run_id)?.run_id || '');
      })
      .catch(() => setBatches(null)) // no batches.json: show the empty state
      .finally(() => setBatchesReady(true));
  }, []);

  useEffect(() => {
    if (!batchesReady || !manifestUrl) return;
    console.info(`[DilemmasViewer] fetching manifest: ${manifestUrl}`);
    setManifest(null);
    setLoadError('');
    // The manifest is the gatekeeper for new runs, so always re-fetch it
    // (0 TTL) rather than serving a stale copy from the localStorage cache.
    cachedFetchJson<RunsManifest>(manifestUrl, 0)
      .then((data) => {
        console.info(`[DilemmasViewer] manifest loaded: ${data.runs?.length ?? 0} runs`);
        setManifest(data);
      })
      .catch((error) => {
        console.error('[DilemmasViewer] manifest load failed:', error);
        setLoadError(String(error));
      });
  }, [manifestUrl, batchesReady]);

  // A batch switch resets the per-model picks and the run caches (run ids
  // are batch-local: eval id + sample + epoch).
  useEffect(() => {
    setPickedRunIds({});
    setRunDocs({});
    runCache.current = {};
    inFlight.current.clear();
    setFallbackDilemma(null);
  }, [selectedBatch]);

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
      const base = `${runsBase}${run.path}/`;
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
  }, [runsBase]);

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
    cachedFetchJson<{ dilemma?: DilemmaInfo }>(`${runsBase}${runsForDilemma[0].path}/config.json`, 0)
      .then((config) => { if (!cancelled && config.dilemma) setFallbackDilemma(config.dilemma); })
      .catch((error) => console.info('[DilemmasViewer] dilemma fallback load failed:', String(error)));
    return () => { cancelled = true; };
  }, [manifest, runsForDilemma, runsBase]);

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

  if (batchesReady && !manifestUrl) {
    return (
      <div style={{ padding: '3rem 0' }}>
        <div style={card}>
          <strong>No dilemma batches found</strong>
          <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>
            <code>{RUNS_BASE}batches.json</code> is missing or lists no exported
            batches. Export one with{' '}
            <code>uv run python code/dilemmas/export_results.py</code> and reload.
          </p>
        </div>
        <DataSourceNotice manifestUrl={`${RUNS_BASE}batches.json`} kind="dilemma" />
      </div>
    );
  }

  if (loadError) {
    return (
      <div style={{ padding: '3rem 0' }}>
        <div style={{ ...card, borderColor: '#fca5a5', color: '#b91c1c' }}>
          <strong>Could not load dilemma data</strong>
          <p style={{ marginTop: '0.5rem', fontSize: '0.85rem' }}>{loadError}</p>
          <p style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
            The selected manifest is <code>{manifestUrl}</code>. {MANIFEST_IS_LOCAL
              ? 'No local batch manifest was found. Check the batch exists under artifacts/dilemmas/runs/full/.'
              : 'Check the public bucket link in the data-source notice below.'}
          </p>
        </div>
        <DataSourceNotice manifestUrl={manifestUrl} kind="dilemma" />
      </div>
    );
  }

  if (!manifest) {
    return <div style={{ padding: '3rem 0', color: 'var(--color-muted)' }}>Loading dilemma data from the selected source…</div>;
  }

  return (
    <div>
      <DataSourceNotice bucket={manifest.bucket || DEFAULT_BUCKET} manifestUrl={manifestUrl} kind="dilemma" />
      {manifest.generated_at && <p className="manifest-generated muted-text">Manifest generated {new Date(manifest.generated_at).toLocaleString()}</p>}
      <div style={{ marginBottom: '1.5rem', padding: '1rem 1.25rem', background: '#fff7ed', borderRadius: '0.6rem', border: '2px solid #f59e0b', fontSize: '0.9rem', fontWeight: 600, color: '#9a3412', lineHeight: 1.6 }}>
        ⚠️ <strong>Development results.</strong> This viewer exposes the raw dilemma artifacts currently present in the selected local or bucket manifest. Treat them as pipeline results, not as pilot/final study conclusions.
      </div>

      <DilemmaStatsTable
        dilemmas={dilemmasList}
        manifest={manifest}
        currentKey={dilemmaKey}
        onSelect={selectDilemma}
      />

      <div className="results-controls">
        {batches && batches.batches.length > 0 && (
          <div>
            <label style={label}>Batch</label>
            <select
              value={selectedBatch}
              onChange={(e) => setSelectedBatch(e.target.value)}
              style={select}
            >
              {/* One batch at a time: each option loads that batch's own
                  manifest (runs/full/<run-id>/index.json). */}
              {batches.batches
                .filter((batch) => batch.run_id)
                .map((batch) => (
                  <option key={batch.run_id} value={batch.run_id}>
                    {batch.run_id} · {batch.models?.length ?? '?'} models · {batch.runs ?? '?'} runs{batch.partial ? ' · partial' : ''}
                  </option>
                ))}
            </select>
          </div>
        )}
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
        roleSummary={
          manifest.profiles?.[stimulus?.profile_id ?? profileId]?.role_summary
          || manifest.profiles?.[stimulus?.profile_id ?? profileId]?.name
        }
        dilemmaIntro={manifest.profiles?.[stimulus?.profile_id ?? profileId]?.dilemma_intro}
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
