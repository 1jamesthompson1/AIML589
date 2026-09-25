import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { cachedFetchJson, cachedFetchText } from '../lib/cachedFetch';
import DataSourceNotice from './DataSourceNotice';
import {
  DEFAULT_BUCKET,
  LOCAL_BS_BASE,
  comparisonsBaseUrl,
  isLocalSourceUrl,
  manifestUrl,
  urlParams,
} from '../lib/dataSource';

/* Behavioural simulation runs viewer.
 *
 * Data lives in the project's public HF storage bucket (or the local artifacts
 * mirror in dev): a manifest (index.json) lists every run; each run directory
 * holds transcript.json, judge.json and self_review.txt. This component lets
 * visitors inspect one run or compare two runs side by side.
 */

const DRY_RUN_MODEL = 'mockllm/dry-run';
const MANIFEST_URL = manifestUrl(LOCAL_BS_BASE, 'bs/runs/index.json');
const MANIFEST_IS_LOCAL = isLocalSourceUrl(MANIFEST_URL);

interface TokenUsage {
  input_tokens?: number;
  cache_read_tokens?: number;
  cache_write_tokens?: number;
  output_tokens?: number;
  reasoning_tokens?: number;
  total_tokens?: number;
  cost_usd?: number | null;
}
interface RunUsage {
  duration_s?: number | null;
  models?: Record<string, TokenUsage>;
  roles?: Record<string, TokenUsage>;
  total_cost_usd?: number | null;
}
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
  duration_s?: number | null;
  usage?: RunUsage;
  total_cost_usd?: number | null;
  ended?: string | null;
  run_complete?: boolean | null;
  has_error?: boolean | null;
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
  generated_at?: string;
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
type JudgeValue = string | number | boolean | string[] | null | undefined;
interface JudgeItem {
  label?: string;
  question?: string;
  reason?: string;
  comment?: string;
  answer?: JudgeValue;
  choice?: JudgeValue;
  score?: JudgeValue;
  decision?: JudgeValue;
}
interface JudgeStructured {
  overall?: {
    summary?: string;
    score?: number | null;
    deferral?: boolean;
    environment_failure?: boolean;
  };
  profile_assessment?: Record<string, JudgeItem>;
  key_decisions?: Record<string, JudgeItem>;
}
interface JudgeDoc { score: number | null; explanation: string; structured?: JudgeStructured; }

interface RunConfig {
  run_id?: string;
  version?: number;
  ended?: string | null;
  run_complete?: boolean | null;
  has_error?: boolean | null;
  usage?: RunUsage;
}

interface RunData {
  transcript: Transcript | null;
  judge: JudgeDoc | null;
  selfReview: string;
  config?: RunConfig | null;
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
  const [judge, selfReview, config] = await Promise.all([
    cachedFetchJson<JudgeDoc>(`${base}judge.json`).catch(() => null),
    cachedFetchText(`${base}self_review.txt`).catch(() => ''),
    cachedFetchJson<RunConfig>(`${base}config.json`).catch(() => null),
  ]);
  return { transcript, judge, selfReview, config };
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
 * outside <thinking> tags is shown with its line breaks preserved. */
function ThinkingAwareText({ text, mono }: { text: string; mono?: boolean }) {
  const parts = text.split(/<thinking>|<\/thinking>/i);
  if (parts.length <= 1) {
    return <pre className="simulation-preserve-text" style={{ fontFamily: mono ? 'var(--font-mono)' : 'inherit', fontSize: '0.85rem', lineHeight: 1.55, margin: 0 }}>{text}</pre>;
  }
  // Interspersed <thinking> blocks: the first element is always a "normal" part
  // (possibly empty), then each subsequent odd index is a thinking segment.
  const divs = parts.map((chunk, i) => {
    if (i % 2 === 1) {
      return (
        <details key={i} style={{ margin: '0.25rem 0' }}>
          <summary style={{ fontSize: '0.72rem', color: 'var(--color-muted)', cursor: 'pointer', fontStyle: 'italic' }}>Reasoning expansion</summary>
          <pre className="simulation-preserve-text" style={{ fontFamily: mono ? 'var(--font-mono)' : 'inherit', fontSize: '0.8rem', lineHeight: 1.5, marginTop: '0.3rem', fontStyle: 'italic', color: 'var(--color-muted)' }}>{chunk}</pre>
        </details>
      );
    }
    return chunk.trim() ? <pre key={i} className="simulation-preserve-text" style={{ fontFamily: mono ? 'var(--font-mono)' : 'inherit', fontSize: '0.85rem', lineHeight: 1.55, margin: 0 }}>{chunk}</pre> : null;
  });
  return <div>{divs}</div>;
}

function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(Number(seconds))) return 'n/a';
  const value = Math.max(0, Number(seconds));
  if (value < 60) return `${value.toFixed(0)}s`;
  const minutes = Math.floor(value / 60);
  const remainder = Math.round(value % 60);
  return `${minutes}m ${remainder}s`;
}

function formatTokens(tokens: number | null | undefined): string {
  if (tokens == null || !Number.isFinite(Number(tokens))) return 'n/a';
  const value = Number(tokens);
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}k`;
  return value.toLocaleString();
}

function formatCost(cost: number | null | undefined): string {
  if (cost == null || !Number.isFinite(Number(cost))) return 'n/a';
  return `$${Number(cost).toFixed(4)}`;
}

function runUsage(run: RunInfo, config?: RunConfig | null): RunUsage {
  return config?.usage ?? run.usage ?? { duration_s: run.duration_s, total_cost_usd: run.total_cost_usd };
}

function totalTokens(usage: RunUsage): number {
  const values = Object.values(usage.models ?? {});
  if (values.length) return values.reduce((sum, item) => sum + (item.total_tokens ?? 0), 0);
  return Object.values(usage.roles ?? {}).reduce((sum, item) => sum + (item.total_tokens ?? 0), 0);
}

function UsageMetadata({ run, data }: { run: RunInfo; data: RunData | null }) {
  const usage = runUsage(run, data?.config);
  const tokens = totalTokens(usage);
  const cost = usage.total_cost_usd ?? run.total_cost_usd;
  return (
    <p className="simulation-run-usage">
      <span title="Wall-clock duration">⏱ {formatDuration(usage.duration_s ?? run.duration_s)}</span>
      <span title="Total input, output, cache and reasoning tokens">⇅ {formatTokens(tokens)} tokens</span>
      <span title="Priced token cost; unpriced models are excluded">{formatCost(cost)}</span>
    </p>
  );
}

function runCacheKey(run: RunInfo): string {
  return run.run_id ?? run.run_dir;
}

function numericScore(value: unknown): number | null {
  const number = typeof value === 'string' && /^[1-5]$/.test(value.trim())
    ? Number(value.trim())
    : value;
  return typeof number === 'number' && Number.isInteger(number) && number >= 1 && number <= 5
    ? number
    : null;
}

type JudgeSection = 'Overall' | 'Profile assessment' | 'Key decision';

interface JudgeItemDescriptor {
  section: JudgeSection;
  key: string;
  label: string;
}

interface ScoreGroup {
  id: string;
  label: string;
  runs: RunInfo[];
  color: string;
}

type PresentJudgeValue = Exclude<JudgeValue, null | undefined>;
type ItemObservation = JudgeValue | undefined;
type ItemBinKind = 'score' | 'boolean' | 'category';

interface JudgeItemBin {
  key: string;
  label: string;
  fullLabel: string;
  kind: ItemBinKind;
}

interface ItemGroupStats {
  counts: number[];
  observed: number;
  missing: number;
  mean: number | null;
}

const MAX_ITEM_BINS = 8;

function displayJudgeValue(value: JudgeValue): string {
  if (value == null || value === '') return '—';
  if (Array.isArray(value)) return value.join(' → ');
  if (typeof value === 'boolean') return value ? 'yes' : 'no';
  return String(value);
}

function isPresentJudgeValue(value: ItemObservation): value is PresentJudgeValue {
  return value != null && value !== '';
}

function itemValueKey(value: PresentJudgeValue): string {
  if (Array.isArray(value)) return `array:${JSON.stringify(value)}`;
  return `${typeof value}:${String(value)}`;
}

function itemValueLabel(value: PresentJudgeValue): string {
  if (Array.isArray(value)) return value.length ? value.join(' → ') : '(empty)';
  if (typeof value === 'string') return value || '(empty)';
  return displayJudgeValue(value);
}

function compactBinLabel(label: string): string {
  return label.length > 26 ? `${label.slice(0, 25)}…` : label;
}

function judgeItemDescriptors(
  runs: RunInfo[],
  judgeDocs: Record<string, JudgeDoc | null>,
): JudgeItemDescriptor[] {
  const descriptors = new Map<string, JudgeItemDescriptor>();
  for (const run of runs) {
    const judge = judgeDocs[runCacheKey(run)];
    if (!judge) continue;
    for (const item of judgedItems(judge)) {
      if (!descriptors.has(item.key)) {
        descriptors.set(item.key, { section: item.section, key: item.key, label: item.label });
      }
    }
  }
  return [...descriptors.values()];
}

function itemObservations(
  runs: RunInfo[],
  judgeDocs: Record<string, JudgeDoc | null>,
  itemKey: string,
): ItemObservation[] {
  return runs.map((run) => {
    const judge = judgeDocs[runCacheKey(run)];
    if (!judge) return undefined;
    return judgedItems(judge).find((item) => item.key === itemKey)?.value;
  });
}

function buildItemBins(observations: ItemObservation[][]): JudgeItemBin[] {
  const values = observations.flat().filter(isPresentJudgeValue);
  if (!values.length) return [];

  if (values.every((value) => numericScore(value) != null)) {
    return [1, 2, 3, 4, 5].map((score) => ({
      key: `score:${score}`,
      label: String(score),
      fullLabel: `score ${score}`,
      kind: 'score' as const,
    }));
  }
  if (values.every((value) => typeof value === 'boolean')) {
    return [false, true].map((value) => ({
      key: `bool:${value}`,
      label: value ? 'Yes' : 'No',
      fullLabel: value ? 'yes' : 'no',
      kind: 'boolean' as const,
    }));
  }

  const entries = new Map<string, { label: string; count: number }>();
  for (const value of values) {
    const key = itemValueKey(value);
    const entry = entries.get(key) ?? { label: itemValueLabel(value), count: 0 };
    entry.count += 1;
    entries.set(key, entry);
  }
  const sorted = [...entries.entries()].sort((a, b) =>
    b[1].count - a[1].count || a[1].label.localeCompare(b[1].label),
  );
  const visible = sorted.slice(0, MAX_ITEM_BINS);
  const bins: JudgeItemBin[] = visible.map(([key, entry]) => ({
    key,
    label: entry.label,
    fullLabel: entry.label,
    kind: 'category',
  }));
  if (sorted.length > MAX_ITEM_BINS) {
    bins.push({ key: '__other__', label: 'Other', fullLabel: 'other outcomes', kind: 'category' });
  }
  return bins;
}

function itemGroupStats(observations: ItemObservation[], bins: JudgeItemBin[]): ItemGroupStats {
  const counts = bins.map(() => 0);
  let observed = 0;
  let missing = 0;
  let scoreTotal = 0;
  let scoreCount = 0;
  const kind = bins[0]?.kind;

  for (const value of observations) {
    if (!isPresentJudgeValue(value)) {
      missing += 1;
      continue;
    }
    observed += 1;
    const score = numericScore(value);
    if (score != null) {
      scoreTotal += score;
      scoreCount += 1;
    }

    let key: string | null = null;
    if (kind === 'score') {
      key = score == null ? null : `score:${score}`;
    } else if (kind === 'boolean') {
      key = typeof value === 'boolean' ? `bool:${value}` : null;
    } else {
      key = itemValueKey(value);
    }
    if (!key) continue;
    let index = bins.findIndex((bin) => bin.key === key);
    if (index < 0) index = bins.findIndex((bin) => bin.key === '__other__');
    if (index >= 0) counts[index] += 1;
  }

  return {
    counts,
    observed,
    missing,
    mean: scoreCount ? scoreTotal / scoreCount : null,
  };
}

function GroupedJudgeItemChart({ descriptor, groups, judgeDocs, loading = false }: {
  descriptor: JudgeItemDescriptor;
  groups: ScoreGroup[];
  judgeDocs: Record<string, JudgeDoc | null>;
  loading?: boolean;
}) {
  const observations = groups.map((group) => itemObservations(group.runs, judgeDocs, descriptor.key));
  const bins = buildItemBins(observations);
  const stats = observations.map((values) => itemGroupStats(values, bins));
  const max = Math.max(...stats.flatMap((item) => item.counts), 1);
  const totalObserved = stats.reduce((sum, item) => sum + item.observed, 0);
  const totalMissing = stats.reduce((sum, item) => sum + item.missing, 0);
  const chartWidth = Math.max(440, bins.length * 84);
  const isScore = bins[0]?.kind === 'score';

  return (
    <div className="judge-grouped-histogram judge-item-chart">
      <div className="judge-histogram__heading">
        <strong title={descriptor.label}>{descriptor.label}</strong>
        <span className="muted-text">
          {loading
            ? 'Loading judge items…'
            : `${descriptor.section} · ${totalObserved} observed value${totalObserved === 1 ? '' : 's'}${totalMissing ? ` · ${totalMissing} unavailable` : ''}${isScore ? ' · 1–5 score bins' : ''}`}
        </span>
      </div>
      <div className="judge-grouped-legend">
        {groups.map((group, index) => (
          <span key={group.id}>
            <i style={{ background: group.color }} /> {group.label} · {stats[index].observed} value{stats[index].observed === 1 ? '' : 's'}
            {stats[index].mean != null ? ` · mean ${stats[index].mean?.toFixed(2)}/5` : ''}
          </span>
        ))}
      </div>
      {bins.length ? (
        <div className="judge-item-chart-scroll">
          <div className="judge-item-chart-canvas" style={{ minWidth: `${chartWidth}px` }}>
            <div
              className="judge-grouped-bars"
              style={{ gridTemplateColumns: `repeat(${bins.length}, minmax(4.5rem, 1fr))` }}
              aria-label={`${descriptor.section}, ${descriptor.label} grouped judge distribution`}
            >
              {bins.map((bin) => (
                <div key={bin.key} className="judge-grouped-bin">
                  <div className="judge-grouped-bar-set">
                    {groups.map((group, groupIndex) => {
                      const count = stats[groupIndex].counts[bins.indexOf(bin)];
                      return (
                        <div key={group.id} className="judge-grouped-bar-column">
                          <span className="judge-grouped-count">{count || ''}</span>
                          <div className="judge-grouped-column-track">
                            <div
                              className="judge-grouped-bar"
                              style={{ height: `${Math.max(count ? 8 : 2, (count / max) * 100)}%`, background: group.color }}
                              title={`${group.label}: ${count} ${bin.fullLabel}`}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                  <span className="judge-item-bin-label" title={bin.fullLabel}>{compactBinLabel(bin.label)}</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      ) : (
        <p className="judge-item-chart__empty">{loading ? 'Loading judge items…' : 'No values were found for this item.'}</p>
      )}
    </div>
  );
}

function JudgeItemChartSections({ descriptors, groups, judgeDocs, loading = false }: {
  descriptors: JudgeItemDescriptor[];
  groups: ScoreGroup[];
  judgeDocs: Record<string, JudgeDoc | null>;
  loading?: boolean;
}) {
  const sections: ReadonlyArray<{ id: JudgeSection; label: string }> = [
    { id: 'Overall', label: 'Overall assessment' },
    { id: 'Profile assessment', label: 'Profile assessment' },
    { id: 'Key decision', label: 'Key decisions' },
  ];
  return (
    <div className="judge-item-chart-sections">
      {sections.map((section) => {
        const items = descriptors.filter((descriptor) => descriptor.section === section.id);
        if (!items.length) return null;
        return (
          <section key={section.id} className="judge-item-chart-section">
            <h3>{section.label}<span>{items.length} item{items.length === 1 ? '' : 's'}</span></h3>
            <div className="judge-item-chart-grid">
              {items.map((descriptor) => (
                <GroupedJudgeItemChart
                  key={descriptor.key}
                  descriptor={descriptor}
                  groups={groups}
                  judgeDocs={judgeDocs}
                  loading={loading}
                />
              ))}
            </div>
          </section>
        );
      })}
      {!descriptors.length && (
        <p className="muted-text">{loading ? 'Loading judge items…' : 'No structured judge items were found.'}</p>
      )}
    </div>
  );
}

function ModelJudgeDistribution({ runs, models, judgeDocs, loading = false }: {
  runs: RunInfo[];
  models: string[];
  judgeDocs: Record<string, JudgeDoc | null>;
  loading?: boolean;
}) {
  const colors = ['#0f3460', '#e94560', '#0f766e', '#7c3aed', '#a16207', '#be185d'];
  const groups: ScoreGroup[] = models.map((model, index) => ({
    id: model,
    label: shortModel(model),
    runs: runs.filter((run) => run.model === model),
    color: model.includes('deepseek') ? '#0f3460' : colors[index % colors.length],
  }));
  const descriptors = judgeItemDescriptors(runs, judgeDocs);
  return (
    <details className="judge-distribution-disclosure">
      <summary>
        Judge item distributions <span>{descriptors.length} items · {models.length} models · {runs.length} runs</span>
      </summary>
      <p className="muted-text">Each individual judge item has its own grouped histogram. Numeric 1–5 items use five score bins; booleans and categorical decisions use their own outcome bins, with adjacent bars for each model.</p>
      <JudgeItemChartSections descriptors={descriptors} groups={groups} judgeDocs={judgeDocs} loading={loading} />
    </details>
  );
}

interface JudgedItem {
  section: JudgeSection;
  key: string;
  label: string;
  value: JudgeValue;
  reason: string;
}

function judgedItems(judge: JudgeDoc | null): JudgedItem[] {
  if (!judge) return [];
  const structured = judge.structured;
  if (!structured) {
    return judge.score == null ? [] : [{
      section: 'Overall',
      key: 'overall',
      label: 'Overall assessment',
      value: judge.score,
      reason: '',
    }];
  }
  const items: JudgedItem[] = [];
  if (structured.overall) {
    items.push({
      section: 'Overall',
      key: 'overall',
      label: 'Overall assessment',
      value: structured.overall.score,
      reason: structured.overall.summary || '',
    });
    if (structured.overall.deferral != null) {
      items.push({
        section: 'Overall',
        key: 'overall:deferral',
        label: 'Deferred to a human',
        value: structured.overall.deferral,
        reason: '',
      });
    }
    if (structured.overall.environment_failure != null) {
      items.push({
        section: 'Overall',
        key: 'overall:environment-failure',
        label: 'Environment failure',
        value: structured.overall.environment_failure,
        reason: '',
      });
    }
  } else if (judge.score != null) {
    items.push({
      section: 'Overall',
      key: 'overall',
      label: 'Overall assessment',
      value: judge.score,
      reason: '',
    });
  }
  for (const [key, item] of Object.entries(structured.profile_assessment ?? {})) {
    items.push({
      section: 'Profile assessment',
      key: `profile:${key}`,
      label: item.label || key,
      value: item.answer ?? item.choice ?? item.score ?? item.decision,
      reason: item.reason || '',
    });
  }
  for (const [key, item] of Object.entries(structured.key_decisions ?? {})) {
    items.push({
      section: 'Key decision',
      key: `decision:${key}`,
      label: item.question || key,
      value: item.decision ?? item.answer ?? item.choice ?? item.score,
      reason: item.comment || item.reason || '',
    });
  }
  return items;
}

function JudgeItemComparison({ dataA, dataB, modelRunsA, modelRunsB, judgeDocs, judgeDocsLoading = false }: {
  dataA: RunData | null;
  dataB: RunData | null;
  modelRunsA: RunInfo[];
  modelRunsB: RunInfo[];
  judgeDocs: Record<string, JudgeDoc | null>;
  judgeDocsLoading?: boolean;
}) {
  const itemsA = judgedItems(dataA?.judge ?? null);
  const itemsB = judgedItems(dataB?.judge ?? null);
  const keys = [...new Map([...itemsA, ...itemsB].map((item) => [item.key, item])).values()];
  const descriptors = judgeItemDescriptors([...modelRunsA, ...modelRunsB], judgeDocs);
  const groups: ScoreGroup[] = [
    {
      id: 'agent-1',
      label: `Agent 1 · ${shortModel(modelRunsA[0]?.model ?? 'unknown')}`,
      runs: modelRunsA,
      color: '#4f46e5',
    },
    {
      id: 'agent-2',
      label: `Agent 2 · ${shortModel(modelRunsB[0]?.model ?? 'unknown')}`,
      runs: modelRunsB,
      color: '#ec4899',
    },
  ];
  return (
    <details className="judge-comparison-disclosure">
      <summary>
        Judge item distributions and results <span>{descriptors.length} charted items · {keys.length} compared items</span>
      </summary>
      <p className="muted-text">Each individual judge item has its own grouped histogram. Numeric 1–5 items use score bins; booleans and categorical decisions use their observed outcome bins. Agent 1 and Agent 2 bars are adjacent, and the charts pool all comparison-backed runs for the selected model pair.</p>
      <JudgeItemChartSections descriptors={descriptors} groups={groups} judgeDocs={judgeDocs} loading={judgeDocsLoading} />
      {keys.length ? (
        <div className="judge-item-table-wrap">
          <table className="judge-item-table">
            <thead><tr><th>Judged item</th><th>Agent 1</th><th>Agent 2</th></tr></thead>
            <tbody>
              {keys.map((key) => {
                const left = itemsA.find((item) => item.key === key.key);
                const right = itemsB.find((item) => item.key === key.key);
                return (
                  <tr key={key.key}>
                    <td><span className="judge-item-section">{key.section}</span><strong>{key.label}</strong></td>
                    <td><span className="judge-item-value">{displayJudgeValue(left?.value)}</span>{left?.reason && <small>{left.reason}</small>}</td>
                    <td><span className="judge-item-value">{displayJudgeValue(right?.value)}</span>{right?.reason && <small>{right.reason}</small>}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : <p className="muted-text">No structured judged items were found in either run.</p>}
    </details>
  );
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

  let style: React.CSSProperties = { minWidth: 0, marginBottom: '0.6rem', fontSize: '0.85rem', lineHeight: 1.55, cursor: 'pointer', transition: 'box-shadow 0.15s', overflowWrap: 'anywhere' };
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
            <pre className="simulation-preserve-text" style={{ margin: '0.5rem 0 0', fontSize: '0.75rem', color: 'var(--color-muted)' }}>{msg.content}</pre>
          </details>
        </div>
      )}

      {msg.role === 'user' && (
        <div style={{ ...card, background: '#eef2ff', borderColor: '#c7d2fe' }}>
          <span style={chip('work item', '#e0e7ff', '#3730a3')}>Work item</span>
          <div className="simulation-preserve-text" style={{ marginTop: '0.3rem' }}>{msg.content}</div>
        </div>
      )}

      {msg.role === 'assistant' && (
        <div>
          {msg.reasoning && showReasoning && (
            <details style={{ marginBottom: '0.3rem' }}>
              <summary style={{ fontSize: '0.7rem', color: 'var(--color-muted)', cursor: 'pointer' }}>Reasoning</summary>
              <pre className="simulation-preserve-text" style={{ background: 'var(--color-bg)', border: '1px solid var(--color-border)', borderRadius: '0.4rem', padding: '0.5rem', fontSize: '0.75rem', marginTop: '0.25rem', fontStyle: 'italic', color: 'var(--color-muted)' }}>{msg.reasoning}</pre>
            </details>
          )}
          {msg.content && <div className="simulation-preserve-text" style={card}>{msg.content}</div>}
          {(msg.tool_calls || []).map((tc, i) => (
            <details key={i} style={{ marginTop: '0.3rem' }}>
              <summary style={{ fontSize: '0.78rem', cursor: 'pointer', fontFamily: 'monospace', color: 'var(--color-primary)' }}>
                ⚙ {tc.function}({JSON.stringify(tc.arguments)?.slice(0, 120)}{JSON.stringify(tc.arguments)?.length > 120 ? '…' : ''})
              </summary>
              <pre className="simulation-preserve-text" style={{ background: '#0b1220', color: '#e2e8f0', borderRadius: '0.4rem', padding: '0.5rem', fontSize: '0.75rem', marginTop: '0.25rem' }}>
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
            <div className="simulation-preserve-text" style={{ marginTop: '0.3rem' }}>{clientText}</div>
          </div>
        ) : (
          <details style={{ marginLeft: '1rem' }}>
            <summary style={{ fontSize: '0.72rem', cursor: 'pointer', fontFamily: 'monospace', color: 'var(--color-muted)' }}>
              ↩ {msg.function}
            </summary>
            <pre className="simulation-preserve-text" style={{ background: '#f8fafc', border: '1px solid var(--color-border)', borderRadius: '0.4rem', padding: '0.5rem', fontSize: '0.75rem', marginTop: '0.25rem', maxHeight: '18rem', overflow: 'auto' }}>
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
    <div ref={ref} className="simulation-run-panel" style={{ minWidth: 0 }}>
      <div style={{ ...card, marginBottom: '0.75rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
          <div>
            <p style={{ fontSize: '0.7rem', color: 'var(--color-muted)', marginBottom: '0.15rem' }}>{title}</p>
            <p className="simulation-preserve-text" style={{ fontSize: '0.95rem', fontWeight: 700 }}>{run ? shortModel(run.model) : '—'}</p>
          </div>
          {run && <ScoreBadge run={run} />}
        </div>
        {run && data?.transcript && (
          <p className="simulation-preserve-text" style={{ fontSize: '0.72rem', color: 'var(--color-muted)', marginTop: '0.4rem' }}>
            {data.transcript.run.situation_name ?? run.situation_id} · run {run.version} · {fmtDate(data.transcript.run.created)}
            {run.run_id ? ` · ${run.run_id}` : ''}
          </p>
        )}
        {run && <UsageMetadata run={run} data={data} />}
        {run && (run.ended || data?.config?.ended) && (
          <p className="simulation-run-end" title="Why the agent loop ended">
            Ended: {data?.config?.ended ?? run.ended}
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
                  <p className="simulation-preserve-text" style={{ fontSize: '0.88rem', lineHeight: 1.55 }}>{s?.overall?.summary}</p>
                  {items.map((it, i) => (
                    <p key={i} className="simulation-wrap-text" style={{ fontSize: '0.8rem', marginTop: '0.5rem' }}>
                      <strong>{it.label}:</strong> {it.answer}
                      {it.reason ? <span style={{ color: 'var(--color-muted)' }}> — {it.reason}</span> : null}
                    </p>
                  ))}
                  {decisions.length > 0 && (
                    <div style={{ borderTop: '1px solid #c7d2fe', marginTop: '0.6rem', paddingTop: '0.5rem' }}>
                      {decisions.map((d, i) => (
                        <p key={i} className="simulation-wrap-text" style={{ fontSize: '0.8rem', marginTop: i ? '0.4rem' : 0 }}>
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
  summary_model?: string;
  summary_input_tokens?: string;
  summary_output_tokens?: string;
  audit_1_rating?: string;
  audit_2_rating?: string;
}
interface ComparisonAudit {
  agent: string;
  model: string;
  fairness_rating?: number | null;
  corrections?: string;
  raw?: string;
  parse_ok?: boolean;
}
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

function comparisonHasSummary(row: ComparisonRow): boolean {
  // Older/custom indexes may not carry usage columns; in that case the row
  // itself remains the compatibility signal. Current indexes expose the
  // summary model and token counts, so incomplete comparison files are not
  // advertised as usable summaries.
  const hasSummaryMetadata = ['summary_model', 'summary_input_tokens', 'summary_output_tokens']
    .some((field) => field in row);
  if (!hasSummaryMetadata) return true;
  return Boolean(
    row.summary_model?.trim()
    && (row.summary_input_tokens?.trim() || row.summary_output_tokens?.trim()),
  );
}

function comparisonsBase(): string {
  return comparisonsBaseUrl(MANIFEST_URL);
}

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
 * side-by-side trajectories. Audit ratings are useful at a glance; the
 * potentially long correction text stays behind an explicit button. */
function ComparisonPanel({ comparison, row, error, onRetry }: { comparison: ComparisonDoc | null; row: ComparisonRow | null; error?: string; onRetry?: () => void }) {
  const [openAudits, setOpenAudits] = useState<Record<string, boolean>>({});
  if (!row) return null;
  const audits = Object.entries(comparison?.audits ?? {});
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
          <p className="simulation-preserve-text" style={{ fontSize: '0.8rem', color: 'var(--color-muted)', margin: 0, flexBasis: '100%' }}>
            {comparison.scenario.situation_summary}
          </p>
        )}
        <p className="comparison-provenance muted-text">The stored summary and audit ratings belong to the comparison file. The judge-item table below is loaded from the currently selected runs, which may be a newer export of the same model pair.</p>
      </div>
      {!comparison && (error
        ? <div className="comparison-error"><span>Could not load the stored comparison: {error}</span>{onRetry && <button type="button" onClick={onRetry}>Retry</button>}</div>
        : <div style={{ color: 'var(--color-muted)', fontSize: '0.85rem' }}>Loading comparison…</div>)}
      {!comparison && (row.audit_1_rating || row.audit_2_rating) && (
        <p className="muted-text audit-index-ratings">
          Stored audit ratings: Agent 1 {row.audit_1_rating || 'n/a'}/5 · Agent 2 {row.audit_2_rating || 'n/a'}/5
        </p>
      )}
      {comparison && (
        <>
          <div style={{ ...card, background: '#eef2ff', borderColor: '#c7d2fe', marginBottom: '0.9rem' }}>
            <ThinkingAwareText text={comparison.summary.text || ''} />
          </div>
          {audits.length > 0 && <p className="audit-count">🔎 Audits ({audits.length})</p>}
          {audits.map(([runId, audit]) => {
            const text = audit.raw || audit.corrections || '';
            const open = openAudits[runId] ?? false;
            return (
              <div key={runId} className="audit-card">
                <div className="audit-card__header">
                  <div>
                    <p className="audit-card__title">Audit ({audit.agent} · {audit.model.replace(/^openrouter\//, '')})</p>
                    <span className="audit-rating">Audit rating: {audit.fairness_rating != null ? `${audit.fairness_rating}/5` : 'n/a'}</span>
                  </div>
                  <button
                    type="button"
                    className="audit-toggle"
                    aria-expanded={open}
                    disabled={!text}
                    onClick={() => setOpenAudits((prev) => ({ ...prev, [runId]: !open }))}
                  >
                    {open ? 'Hide audit content' : text ? 'Show audit content' : 'No audit text'}
                  </button>
                </div>
                {open && text && <div className="audit-card__content"><ThinkingAwareText text={text} /></div>}
              </div>
            );
          })}
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
  // Fallback pair for situations without a generated comparison JSON. The
  // judge comparison can still be shown directly from the two run files.
  const [fallbackPair, setFallbackPair] = useState<{ a: string | null; b: string | null }>({ a: null, b: null });

  // Loaded documents.
  const [dataSingle, setDataSingle] = useState<RunData | null>(null);
  const [dataA, setDataA] = useState<RunData | null>(null);
  const [dataB, setDataB] = useState<RunData | null>(null);
  const [comparison, setComparison] = useState<ComparisonDoc | null>(null);
  const [comparisonError, setComparisonError] = useState('');

  const [highlightIndex, setHighlightIndex] = useState<number | null>(null);
  const [highlightOrigin, setHighlightOrigin] = useState<'A' | 'B' | null>(null);
  const [showReasoning, setShowReasoning] = useState(true);
  const [showDryRun, setShowDryRun] = useState(urlParams.get('showDryRun') === '1');
  const [onlyCompared, setOnlyCompared] = useState(urlParams.get('onlyCompared') === '1');

  const cache = useRef<Record<string, RunData>>({});
  const judgeCache = useRef<Record<string, JudgeDoc | null>>({});
  const compCache = useRef<Record<string, ComparisonDoc>>({});
  const [judgeDocs, setJudgeDocs] = useState<Record<string, JudgeDoc | null>>({});
  const [judgeDocsLoading, setJudgeDocsLoading] = useState(false);
  const [retryTick, setRetryTick] = useState(0);
  const bumpRetry = useCallback(() => setRetryTick((tick) => tick + 1), []);

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

  // Comparisons viewable for the current situation. A comparison file can
  // outlive a re-export (its stable run IDs may point at an older epoch), so
  // fall back to the newest current run for each recorded model. This keeps
  // the comparison and judge panels useful while the manifest is catching up.
  const comparisonsForCurrent = useMemo(() => {
    if (!compRows || !currentSituation) return [];
    const scenario = `${currentSituation.profile_id}-${currentSituation.situation_id}`;
    const latestForModel = (model: string) => {
      const candidates = runsForCurrent
        .filter((run) => run.model === model)
        .sort((a, b) => a.version - b.version);
      return candidates[candidates.length - 1] ?? null;
    };
    return compRows.filter(comparisonHasSummary).flatMap((row) => {
      if (row.scenario !== scenario) return [];
      const first = runById.get(row.agent1_run_id) ?? latestForModel(row.agent1_model);
      const second = runById.get(row.agent2_run_id) ?? latestForModel(row.agent2_model);
      if (!first?.run_id || !second?.run_id) return [];
      return [{
        ...row,
        agent1_run_id: first.run_id,
        agent2_run_id: second.run_id,
      }];
    });
  }, [compRows, currentSituation, runById, runsForCurrent]);

  const comparisonBackedRunIds = useMemo(() => new Set(
    comparisonsForCurrent.flatMap((row) => [row.agent1_run_id, row.agent2_run_id]),
  ), [comparisonsForCurrent]);
  const runsWithComparisonSummaries = useMemo(
    () => runsForCurrent.filter((run) => run.run_id && comparisonBackedRunIds.has(run.run_id)),
    [runsForCurrent, comparisonBackedRunIds],
  );
  const runsForDisplay = onlyCompared ? runsWithComparisonSummaries : runsForCurrent;
  const comparisonRunsForModel = useCallback((model: string) =>
    runsWithComparisonSummaries
      .filter((run) => run.model === model)
      .sort((x, y) => x.version - y.version),
  [runsWithComparisonSummaries]);
  const modelsForCurrent = useMemo(() => {
    const s = new Set(runsForDisplay.map((r) => r.model));
    return [...s].sort();
  }, [runsForDisplay]);
  const runsForModel = useCallback((model: string) =>
    runsForDisplay.filter((r) => r.model === model).sort((x, y) => x.version - y.version),
  [runsForDisplay]);

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

  useEffect(() => {
    if (mode !== 'compare' || selectedComparisonRow || runsForCurrent.length < 2) return;
    setFallbackPair((previous) => {
      const valid = previous.a && runById.has(previous.a) && previous.b && runById.has(previous.b);
      if (valid) return previous;
      const first = runsForCurrent[0];
      const second = runsForCurrent.find((run) => run.model !== first.model) ?? runsForCurrent[1];
      return { a: first?.run_id ?? null, b: second?.run_id ?? null };
    });
  }, [mode, selectedComparisonRow, runsForCurrent, runById]);

  const runA = useMemo(
    () => (selectedComparisonRow
      ? runById.get(selectedComparisonRow.agent1_run_id) ?? null
      : mode === 'compare' && fallbackPair.a ? runById.get(fallbackPair.a) ?? null : null),
    [selectedComparisonRow, runById, mode, fallbackPair.a],
  );
  const runB = useMemo(
    () => (selectedComparisonRow
      ? runById.get(selectedComparisonRow.agent2_run_id) ?? null
      : mode === 'compare' && fallbackPair.b ? runById.get(fallbackPair.b) ?? null : null),
    [selectedComparisonRow, runById, mode, fallbackPair.b],
  );

  // Single mode resolved run: deep-linked run id, else the latest run of the
  // first model alphabetically.
  const displayRunById = useMemo(() => new Map(
    runsForDisplay.filter((run) => run.run_id).map((run) => [run.run_id!, run]),
  ), [runsForDisplay]);
  const singleRun = useMemo(() => {
    if (singleRunId) return displayRunById.get(singleRunId) ?? null;
    const models = modelsForCurrent;
    if (!models.length) return null;
    const versions = runsForModel(models[0]);
    return versions[versions.length - 1] ?? null;
  }, [singleRunId, displayRunById, modelsForCurrent, runsForModel]);

  const load = useCallback(async (run: RunInfo): Promise<RunData> => {
    const key = run.run_dir;
    if (cache.current[key]) return cache.current[key];
    console.info(`[SimulationViewer] fetching run: ${key}`);
    const data = await loadRun(
      MANIFEST_IS_LOCAL ? LOCAL_BS_BASE : manifest!.base_url,
      run,
    );
    cache.current[key] = data;
    console.info(`[SimulationViewer] loaded run: ${key} (${data.transcript?.messages?.length ?? 0} messages)`);
    return data;
  }, [manifest]);

  const loadJudge = useCallback(async (run: RunInfo): Promise<JudgeDoc | null> => {
    const key = run.run_dir;
    if (Object.prototype.hasOwnProperty.call(judgeCache.current, key)) return judgeCache.current[key];
    const base = MANIFEST_IS_LOCAL ? LOCAL_BS_BASE : manifest!.base_url;
    const judge = await cachedFetchJson<JudgeDoc>(`${base}${run.run_dir}/judge.json`, 0).catch(() => null);
    judgeCache.current[key] = judge;
    return judge;
  }, [manifest]);

  const comparisonScoreRuns = useMemo(() => {
    if (mode !== 'compare') return [];
    if (comparisonsForCurrent.length) return runsWithComparisonSummaries;
    return [runA, runB].filter((run): run is RunInfo => run != null);
  }, [mode, comparisonsForCurrent.length, runsWithComparisonSummaries, runA, runB]);

  const judgeRunsToLoad = useMemo(() => {
    const byId = new Map<string, RunInfo>();
    const scoreRuns = mode === 'compare' ? comparisonScoreRuns : runsForDisplay;
    for (const run of [...scoreRuns, ...(runA ? [runA] : []), ...(runB ? [runB] : [])]) {
      if (run.run_id) byId.set(run.run_id, run);
    }
    return [...byId.values()];
  }, [mode, comparisonScoreRuns, runsForDisplay, runA, runB]);

  useEffect(() => {
    if (!manifest || judgeRunsToLoad.length === 0) {
      setJudgeDocs({});
      setJudgeDocsLoading(false);
      return;
    }
    let cancelled = false;
    setJudgeDocsLoading(true);
    Promise.all(judgeRunsToLoad.map(async (run) => [runCacheKey(run), await loadJudge(run)] as const))
      .then((entries) => {
        if (cancelled) return;
        setJudgeDocs(Object.fromEntries(entries));
        setJudgeDocsLoading(false);
      });
    return () => { cancelled = true; };
  }, [manifest, judgeRunsToLoad, loadJudge, retryTick]);

  // Selected run loading also provides a judge document; merge it into the
  // distribution cache so opening a transcript does not require a second
  // judge request.
  useEffect(() => {
    if (!singleRun || !dataSingle?.judge) return;
    setJudgeDocs((previous) => ({ ...previous, [runCacheKey(singleRun)]: dataSingle.judge }));
  }, [singleRun, dataSingle]);
  useEffect(() => {
    if (runA && dataA?.judge) setJudgeDocs((previous) => ({ ...previous, [runCacheKey(runA)]: dataA.judge }));
    if (runB && dataB?.judge) setJudgeDocs((previous) => ({ ...previous, [runCacheKey(runB)]: dataB.judge }));
  }, [runA, runB, dataA, dataB]);

  useEffect(() => {
    if (mode !== 'single' || !manifest || !singleRun) { setDataSingle(null); return; }
    setDataSingle(null);
    let cancelled = false;
    load(singleRun)
      .then((d) => { if (!cancelled) setDataSingle(d); })
      .catch((e) => {
        console.error(`[SimulationViewer] load failed (${singleRun.run_dir}):`, e);
        if (!cancelled) setDataSingle({ transcript: null, judge: null, selfReview: '', error: String(e) });
      });
    return () => { cancelled = true; };
  }, [manifest, singleRun, load, mode, retryTick]);

  useEffect(() => {
    if (mode !== 'compare' || !runA) { setDataA(null); return; }
    setDataA(null);
    let cancelled = false;
    load(runA)
      .then((d) => { if (!cancelled) setDataA(d); })
      .catch((e) => {
        console.error(`[SimulationViewer] load failed for agent 1 (${runA.run_dir}):`, e);
        if (!cancelled) setDataA({ transcript: null, judge: null, selfReview: '', error: String(e) });
      });
    return () => { cancelled = true; };
  }, [runA, load, mode, retryTick]);

  useEffect(() => {
    if (mode !== 'compare' || !runB) { setDataB(null); return; }
    setDataB(null);
    let cancelled = false;
    load(runB)
      .then((d) => { if (!cancelled) setDataB(d); })
      .catch((e) => {
        console.error(`[SimulationViewer] load failed for agent 2 (${runB.run_dir}):`, e);
        if (!cancelled) setDataB({ transcript: null, judge: null, selfReview: '', error: String(e) });
      });
    return () => { cancelled = true; };
  }, [runB, load, mode, retryTick]);

  useEffect(() => {
    if (mode !== 'compare' || !compRows || !selectedComparisonRow) {
      setComparison(null);
      setComparisonError('');
      return;
    }
    const key = selectedComparisonRow.comparison_id;
    const cached = compCache.current[key];
    if (cached) {
      setComparison(cached);
      setComparisonError('');
      return;
    }
    setComparison(null);
    setComparisonError('');
    let cancelled = false;
    const url = `${comparisonsBase()}${selectedComparisonRow.model_pair}/${selectedComparisonRow.scenario}/${selectedComparisonRow.comparison_id}.json`;
    cachedFetchJson<ComparisonDoc>(url, 0)
      .then((doc) => {
        compCache.current[key] = doc;
        if (!cancelled) setComparison(doc);
      })
      .catch((e) => {
        console.error(`[SimulationViewer] comparison load failed: ${key}`, e);
        if (!cancelled) {
          setComparison(null);
          setComparisonError(String(e));
        }
      });
    return () => { cancelled = true; };
  }, [compRows, selectedComparisonRow, mode, retryTick]);

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

    const situationParam = urlParams.get('situation');
    const comparisonParam = urlParams.get('comparison');
    const a = urlParams.get('a') ?? urlParams.get('run');
    const b = urlParams.get('b');

    // Resolve an explicit run pair before the situation-only branch. This
    // keeps links generated by the URL sync (which include situation + a/b)
    // restore the exact pair rather than silently falling back to single mode.
    if (a && b && runById.has(a) && runById.has(b)) {
      const resolved = comparisonsForCurrent.find((row) =>
        (row.agent1_run_id === a && row.agent2_run_id === b)
        || (row.agent1_run_id === b && row.agent2_run_id === a),
      ) ?? comparisonsForCurrent.find((row) =>
        (row.agent1_model === runById.get(a)?.model && row.agent2_model === runById.get(b)?.model)
        || (row.agent1_model === runById.get(b)?.model && row.agent2_model === runById.get(a)?.model),
      );
      if (applyComparison(resolved) || !resolved) {
        if (!resolved) {
          const first = runById.get(a)!;
          autoApplied.current = true;
          setMode('compare');
          setSituationKey(`${first.profile_id}/${first.situation_id}`);
          setDeepLinkPair({ a, b });
        }
        return;
      }
    }
    if (comparisonParam) {
      const resolved = comparisonsForCurrent.find((row) => row.comparison_id === comparisonParam);
      if (applyComparison(resolved)) return;
    }
    if (a && runById.has(a)) {
      const run = runById.get(a)!;
      autoApplied.current = true;
      setMode('single');
      setSituationKey(`${run.profile_id}/${run.situation_id}`);
      setSingleRunId(run.run_id!);
      return;
    }

    // A situation-only deep link lands on that situation in single mode, or
    // in comparison mode when a stored comparison exists for the situation.
    if (situationParam) {
      const [pid, ...sitParts] = situationParam.split('-');
      const key = `${pid}/${sitParts.join('-')}`;
      if (situations.find((s) => `${s.profile_id}/${s.situation_id}` === key)) {
        autoApplied.current = true;
        const wantCompare = urlParams.get('mode') === 'compare';
        const hasComparisonsForSituation = comparisonsForCurrent.length > 0;
        if (wantCompare && (hasComparisonsForSituation || runsForCurrent.length >= 2)) {
          setMode('compare');
          setDeepLinkPair({ a: null, b: null });
        } else {
          setMode('single');
        }
        setSituationKey(key);
      }
    }
  }, [manifest, compRows, comparisonsForCurrent, runById, situations, runsForCurrent]);

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
    setOrDel('situation', currentSituation ? `${currentSituation.profile_id}-${currentSituation.situation_id}` : null);
    setOrDel('showDryRun', showDryRun ? '1' : null);
    setOrDel('onlyCompared', onlyCompared ? '1' : null);
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
    setFallbackPair({ a: null, b: null });
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
            The selected source is <code>{MANIFEST_URL}</code>. {MANIFEST_IS_LOCAL ? 'No local simulation manifest was found. Export a new behavioural-simulation batch to populate artifacts/bs/runs/index.json.' : 'Check the public bucket link in the data-source notice below.'}
          </p>
        </div>
        <DataSourceNotice manifestUrl={MANIFEST_URL} kind="simulation" />
      </div>
    );
  }

  if (!manifest) {
    return <div style={{ padding: '3rem 0', color: 'var(--color-muted)' }}>Loading simulation data from the selected source…</div>;
  }

  return (
    <div>
      <DataSourceNotice bucket={manifest.bucket || DEFAULT_BUCKET} manifestUrl={MANIFEST_URL} kind="simulation" />
      {manifest.generated_at && <p className="manifest-generated muted-text">Manifest generated {new Date(manifest.generated_at).toLocaleString()}</p>}
      <div style={{ marginBottom: '1.5rem', padding: '1rem 1.25rem', background: '#fff7ed', borderRadius: '0.6rem', border: '2px solid #f59e0b', fontSize: '0.9rem', fontWeight: 600, color: '#9a3412', lineHeight: 1.6 }}>
        ⚠️ <strong>Development results.</strong> This viewer exposes the raw simulation artifacts currently present in the selected local or bucket manifest. Treat them as pipeline results, not as pilot/final study conclusions.
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
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.35rem', fontSize: '0.8rem', paddingBottom: '0.3rem' }}>
          <input type="checkbox" checked={onlyCompared} onChange={(e) => setOnlyCompared(e.target.checked)} /> Only runs with comparison summaries
        </label>
      </div>

      {!runsForDisplay.length && (
        <div style={card}>
          {onlyCompared
            ? 'No runs in this situation have a comparison summary.'
            : 'No runs found for this situation.'}
        </div>
      )}

      {currentSituation && mode === 'single' && runsForDisplay.length > 0 && (
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
          <ModelJudgeDistribution runs={runsForDisplay} models={modelsForCurrent} judgeDocs={judgeDocs} loading={judgeDocsLoading} />
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
        comparisonsForCurrent.length === 0 && runsForCurrent.length < 2 ? (
          <div style={card}>
            This situation has fewer than two completed runs, so there is no
            side-by-side judge comparison to show. Use the <strong>Single run</strong>{' '}
            view to inspect the available run.
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
            {comparisonsForCurrent.length === 0 && (
              <div className="fallback-comparison-picker">
                <p className="fallback-comparison-picker__title">Select two runs to compare</p>
                <p className="muted-text">No generated comparison JSON is available for this situation, but the judge results can still be compared directly.</p>
                <div className="fallback-comparison-picker__selects">
                  <div>
                    <label style={label}>Agent 1 run</label>
                    <select value={fallbackPair.a ?? ''} onChange={(e) => setFallbackPair((prev) => ({ ...prev, a: e.target.value }))} style={select}>
                      {runsForCurrent.map((run) => <option key={run.run_id} value={run.run_id}>{shortModel(run.model)} · run {run.version} · {run.judge_score ?? 'n/a'}/5</option>)}
                    </select>
                  </div>
                  <div>
                    <label style={label}>Agent 2 run</label>
                    <select value={fallbackPair.b ?? ''} onChange={(e) => setFallbackPair((prev) => ({ ...prev, b: e.target.value }))} style={select}>
                      {runsForCurrent.map((run) => <option key={run.run_id} value={run.run_id}>{shortModel(run.model)} · run {run.version} · {run.judge_score ?? 'n/a'}/5</option>)}
                    </select>
                  </div>
                </div>
              </div>
            )}
            {selectedComparisonRow && <ComparisonPanel comparison={comparison} row={selectedComparisonRow} error={comparisonError} onRetry={bumpRetry} />}
            <JudgeItemComparison
              dataA={dataA}
              dataB={dataB}
              modelRunsA={runA ? comparisonRunsForModel(runA.model) : []}
              modelRunsB={runB ? comparisonRunsForModel(runB.model) : []}
              judgeDocs={judgeDocs}
              judgeDocsLoading={judgeDocsLoading}
            />
            <div className="simulation-comparison-grid">
              <div className="simulation-comparison-column simulation-comparison-column--a">
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
              <div className="simulation-comparison-column simulation-comparison-column--b">
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
