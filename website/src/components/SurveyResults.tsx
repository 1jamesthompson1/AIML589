import { useEffect, useMemo, useRef, useState } from 'react';
import { cachedFetchJson } from '../lib/cachedFetch';
import DataSourceNotice from './DataSourceNotice';
import { LOCAL_SURVEY_BASE, manifestUrl } from '../lib/dataSource';

/* Survey results viewer.
 *
 * The aggregate is written by survey/analyze.py to
 * dilemmas/survey-results/index.json (local artifacts mirror in dev, public
 * HF bucket in production). It holds only shares and counts - no individual
 * responses - so the page can be public.
 *
 * Two different results are shown side by side:
 * - the direct head-to-head vote (the H2 test): which response respondents
 *   preferred, measured on the 5-point scale and as a Bradley-Terry win rate;
 * - the answer-option distributions: how close each model's judged option
 *   shares are to the options the public endorsed (TVD).
 * They need not agree: a model can win more votes while the other model's
 * overall option distribution is closer to the public's.
 */

const SURVEY_MANIFEST_URL = manifestUrl(
  LOCAL_SURVEY_BASE,
  'dilemmas/survey-results/index.json',
);

const PUBLIC_COLOUR = '#0f3460';
const BASE_COLOUR = '#e94560';
const FINE_TUNE_COLOUR = '#16a34a';

type Closer = 'base' | 'fine_tune' | 'tie';
type SortMode = 'fine_tune' | 'base' | 'alpha';

interface PreferenceStats {
  ratings: number;
  mean: number;
  ci_low: number;
  ci_high: number;
  base_wins: number;
  fine_tune_wins: number;
  ties: number;
  decided: number;
  base_share: number;
  fine_tune_share: number;
  win_rate: number;
}

interface PreferenceTest {
  estimate: number;
  ci: [number, number];
  p: number;
  n: number;
  respondents: number;
}

interface SurveyDilemma {
  dilemma_id: string;
  profile_id: string;
  situation_id: string;
  question: string;
  situation: string;
  answer_options: string[];
  ratings: number;
  tie_share: number;
  preference: PreferenceStats;
  shares: {
    public: number[];
    base: number[];
    fine_tune: number[];
  };
  tvd: {
    base: number;
    fine_tune: number;
  };
  closer: Closer;
  top: {
    public: number;
    base: number;
    fine_tune: number;
  };
}

interface SurveyPayload {
  schema: string;
  generated_at: string;
  run_id: string;
  subset: string;
  models: {
    base: string;
    fine_tune: string;
    base_label: string;
    fine_tune_label: string;
  };
  summary: {
    ratings: number;
    respondents: number;
    dilemmas: number;
    tie_share: number;
    mean_tvd_base: number;
    mean_tvd_fine_tune: number;
    base_closer: number;
    fine_tune_closer: number;
    ties: number;
  };
  preference: PreferenceStats & {
    lmm: PreferenceTest;
    bradley_terry: PreferenceTest;
    h2_supported: boolean;
  };
  method: string;
  model_share_source: string;
  selection: {
    rule: string;
    per_dilemma: number | null;
    p_threshold: number | null;
  };
  dilemmas: SurveyDilemma[];
}

const SORT_OPTIONS: Array<{ value: SortMode; label: string }> = [
  { value: 'fine_tune', label: 'Fine-tune advantage' },
  { value: 'base', label: 'Base advantage' },
  { value: 'alpha', label: 'Dilemma name' },
];

function pct(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function score(value: number): string {
  return value.toFixed(2);
}

function signed(value: number, digits = 2): string {
  return `${value >= 0 ? '+' : '−'}${Math.abs(value).toFixed(digits)}`;
}

function fmtP(value: number): string {
  return value < 0.001 ? '<0.001' : value.toFixed(3);
}

function prettyId(value: string): string {
  return value.replaceAll('_', ' ');
}

function closerLabel(closer: Closer): string {
  if (closer === 'fine_tune') return 'Fine-tune closer';
  if (closer === 'base') return 'Base closer';
  return 'Tied';
}

function SummaryCards({ payload }: { payload: SurveyPayload }) {
  const summary = payload.summary;
  const preference = payload.preference;
  const bt = preference.bradley_terry;
  const lmm = preference.lmm;

  const baseVote = bt.ci[1] < 0.5;
  const fineTuneVote = bt.ci[0] > 0.5;
  const voteWinner = baseVote
    ? 'Base model'
    : fineTuneVote
      ? 'Fine-tuned model'
      : 'No clear winner';
  const voteClass = baseVote ? 'base' : fineTuneVote ? 'fine_tune' : 'tie';

  const lmmWinner =
    lmm.ci[1] < 0
      ? 'Base model'
      : lmm.ci[0] > 0
        ? 'Fine-tuned model'
        : 'No significant difference';

  const fineTuneClosest = summary.mean_tvd_fine_tune < summary.mean_tvd_base;
  const closest = fineTuneClosest
    ? payload.models.fine_tune_label
    : payload.models.base_label;
  const closestScore = fineTuneClosest
    ? summary.mean_tvd_fine_tune
    : summary.mean_tvd_base;
  const otherScore = fineTuneClosest
    ? summary.mean_tvd_base
    : summary.mean_tvd_fine_tune;
  const tvdLeader =
    summary.fine_tune_closer === summary.base_closer
      ? 'tied'
      : summary.fine_tune_closer > summary.base_closer
        ? 'fine-tune'
        : 'base';
  const leaderCount = Math.max(summary.fine_tune_closer, summary.base_closer);
  const otherCount = Math.min(summary.fine_tune_closer, summary.base_closer);

  return (
    <div className="survey-results__cards">
      <article className={`survey-results__card survey-results__card--headline survey-results__card--${voteClass}`}>
        <span className="survey-results__label">
          Direct vote (<a className="survey-results__rq-link" href="/#rq2">RQ2</a>)
        </span>
        <strong className="survey-results__value">{voteWinner}</strong>
        <span className="survey-results__note">
          Base {pct(preference.base_share)} vs fine-tune {pct(preference.fine_tune_share)} of decided votes
          {' · '}fine-tune win rate {bt.estimate.toFixed(3)} [{bt.ci[0].toFixed(3)}, {bt.ci[1].toFixed(3)}]
          {' · '}p = {fmtP(bt.p)}
        </span>
      </article>
      <article className="survey-results__card">
        <span className="survey-results__label">Likert preference (LMM)</span>
        <strong className="survey-results__value">{lmmWinner}</strong>
        <span className="survey-results__note">
          {signed(lmm.estimate)} points [{lmm.ci[0].toFixed(3)}, {lmm.ci[1].toFixed(3)}]
          {' · '}p = {fmtP(lmm.p)}
        </span>
      </article>
      <article className="survey-results__card">
        <span className="survey-results__label">Option shares closest</span>
        <strong className="survey-results__value">{closest}</strong>
        <span className="survey-results__note">
          Mean TVD {score(closestScore)} vs {score(otherScore)}; {tvdLeader} closer on{' '}
          {leaderCount} of {summary.dilemmas}
          {otherCount !== leaderCount ? ` (other on ${otherCount})` : ''}
        </span>
      </article>
      <article className="survey-results__card">
        <span className="survey-results__label">Public sample</span>
        <strong className="survey-results__value">
          {summary.ratings.toLocaleString('en-NZ')}
        </strong>
        <span className="survey-results__note">
          ratings from {summary.respondents.toLocaleString('en-NZ')} respondents
        </span>
      </article>
    </div>
  );
}

function Rq2Summary({ payload }: { payload: SurveyPayload }) {
  const preference = payload.preference;
  const bt = preference.bradley_terry;
  const lmm = preference.lmm;
  const baseVote = bt.ci[1] < 0.5;
  const fineTuneVote = bt.ci[0] > 0.5;
  const direction = baseVote
    ? 'The Bradley–Terry interval is below 0.5, so base was voted for more often.'
    : fineTuneVote
      ? 'The Bradley–Terry interval is above 0.5, so the fine-tune was voted for more often.'
      : 'The Bradley–Terry interval includes 0.5, so the direct vote is inconclusive.';

  return (
    <div className="survey-preference__h2">
      <div className="survey-preference__stat">
        <span className="survey-results__label">Bradley–Terry · fine-tune win rate</span>
        <strong>{bt.estimate.toFixed(3)}</strong>
        <span>
          95% CI [{bt.ci[0].toFixed(3)}, {bt.ci[1].toFixed(3)}] · p = {fmtP(bt.p)}
        </span>
      </div>
      <div className="survey-preference__stat">
        <span className="survey-results__label">LMM · mean preference</span>
        <strong>{signed(lmm.estimate)} points</strong>
        <span>
          95% CI [{lmm.ci[0].toFixed(3)}, {lmm.ci[1].toFixed(3)}] · p = {fmtP(lmm.p)}
        </span>
      </div>
      <div className="survey-preference__stat">
        <span className="survey-results__label">Direct votes</span>
        <strong>
          {pct(preference.base_share)} base · {pct(preference.fine_tune_share)} fine-tune
        </strong>
        <span>
          {preference.base_wins} base · {preference.fine_tune_wins} fine-tune ·{' '}
          {preference.ties} tied
        </span>
      </div>
      <p className="survey-preference__verdict">
        The pre-registered H2 test for RQ2 is{' '}
        <strong>{payload.preference.h2_supported ? 'supported' : 'not supported'}</strong>.
        {' '}{direction}
      </p>
    </div>
  );
}

function PreferenceChart({
  dilemmas,
  selectedId,
  onSelect,
}: {
  dilemmas: SurveyDilemma[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const ordered = useMemo(
    () => [...dilemmas].sort((a, b) => b.preference.mean - a.preference.mean),
    [dilemmas],
  );
  const ticks = [-2, -1, 0, 1, 2];

  return (
    <div className="survey-preference__chart">
      <div className="survey-preference__legend">
        <span className="survey-preference__legend-item">
          <span className="survey-preference__swatch survey-preference__swatch--base" />
          Base preferred
        </span>
        <span className="survey-preference__legend-item">
          <span className="survey-preference__swatch survey-preference__swatch--ft" />
          Fine-tune preferred
        </span>
        <span className="muted-text">
          bar = mean preference · line = 95% CI · sorted by mean
        </span>
      </div>

      <div className="survey-preference__axis" aria-hidden="true">
        <span />
        <span className="survey-preference__axis-track">
          {ticks.map((tick) => (
            <span
              key={tick}
              className="survey-preference__tick"
              style={{
                left: `${((tick + 2) / 4) * 100}%`,
                transform: 'translateX(-50%)',
              }}
            >
              {tick > 0 ? `+${tick}` : tick}
            </span>
          ))}
        </span>
        <span />
      </div>

      <div className="survey-preference__rows">
        {ordered.map((dilemma) => {
          const pref = dilemma.preference;
          const mean = Math.max(-2, Math.min(2, pref.mean));
          const zero = 50;
          const half = (Math.abs(mean) / 2) * 50;
          const left = mean < 0 ? zero - half : zero;
          const colour = mean < 0 ? BASE_COLOUR : FINE_TUNE_COLOUR;
          const ciLow = ((Math.max(-2, pref.ci_low) + 2) / 4) * 100;
          const ciHigh = ((Math.min(2, pref.ci_high) + 2) / 4) * 100;
          const voteClass = pref.base_share > pref.fine_tune_share
            ? 'base'
            : pref.fine_tune_share > pref.base_share
              ? 'fine_tune'
              : 'tie';
          const voteLabel = voteClass === 'base'
            ? 'Base'
            : voteClass === 'fine_tune'
              ? 'Fine-tune'
              : 'Tied';
          return (
            <button
              key={dilemma.dilemma_id}
              type="button"
              className={`survey-preference__row${dilemma.dilemma_id === selectedId ? ' is-selected' : ''}`}
              onClick={() => onSelect(dilemma.dilemma_id)}
              aria-pressed={dilemma.dilemma_id === selectedId}
              aria-label={`${prettyId(dilemma.situation_id)}: mean preference ${signed(pref.mean)}, 95% CI ${signed(pref.ci_low)} to ${signed(pref.ci_high)}. Base wins ${pref.base_wins}, fine-tune wins ${pref.fine_tune_wins}, tied ${pref.ties}. ${voteLabel} preferred.`}
            >
              <span className="survey-preference__label" title={prettyId(dilemma.situation_id)}>
                {prettyId(dilemma.situation_id)}
              </span>
              <span className="survey-preference__track" aria-hidden="true">
                <span className="survey-preference__zero" />
                <span
                  className={`survey-preference__ci survey-preference__ci--${voteClass}`}
                  style={{ left: `${ciLow}%`, width: `${ciHigh - ciLow}%` }}
                />
                <span
                  className="survey-preference__bar"
                  style={{ left: `${left}%`, width: `${half}%`, background: colour }}
                />
                <span
                  className="survey-preference__dot"
                  style={{ left: `${((mean + 2) / 4) * 100}%`, background: colour }}
                />
              </span>
              <span className={`survey-preference__badge survey-preference__badge--${voteClass}`}>
                {voteLabel} {signed(pref.mean)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function TvdChart({
  dilemmas,
  selectedId,
  onSelect,
}: {
  dilemmas: SurveyDilemma[];
  selectedId: string;
  onSelect: (id: string) => void;
}) {
  const [sort, setSort] = useState<SortMode>('fine_tune');

  const ordered = useMemo(() => {
    const rows = [...dilemmas];
    const gap = (dilemma: SurveyDilemma) =>
      dilemma.tvd.base - dilemma.tvd.fine_tune;
    if (sort === 'alpha') {
      rows.sort((a, b) => prettyId(a.situation_id).localeCompare(prettyId(b.situation_id)));
    } else if (sort === 'base') {
      // Negative gap = base is closer; start with the base's best dilemmas.
      rows.sort((a, b) => gap(a) - gap(b));
    } else {
      rows.sort((a, b) => gap(b) - gap(a));
    }
    return rows;
  }, [dilemmas, sort]);

  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <section className="survey-tvd" aria-label="Model distance to the public by dilemma">
      <div className="survey-tvd__head">
        <div>
          <h2>How close each model is to the public</h2>
          <p className="muted-text">
            Each row is a dilemma. Dots show the total variation distance
            between the public&rsquo;s answer-option shares and the model&rsquo;s.
            Lower is closer. Click a dilemma to see its three distributions.
          </p>
        </div>
        <label className="survey-tvd__sort">
          <span>Sort</span>
          <select value={sort} onChange={(event) => setSort(event.target.value as SortMode)}>
            {SORT_OPTIONS.map((option) => (
              <option key={option.value} value={option.value}>{option.label}</option>
            ))}
          </select>
        </label>
      </div>

      <div className="survey-tvd__legend">
        <span className="survey-tvd__legend-item">
          <span className="survey-tvd__swatch survey-tvd__swatch--base" />
          Base model
        </span>
        <span className="survey-tvd__legend-item">
          <span className="survey-tvd__swatch survey-tvd__swatch--ft" />
          Fine-tuned model
        </span>
        <span className="muted-text">0 = identical · 1 = no shared option</span>
      </div>

      <div className="survey-tvd__axis" aria-hidden="true">
        <span />
        <span className="survey-tvd__axis-track">
          {ticks.map((tick) => (
            <span
              key={tick}
              className="survey-tvd__tick"
              style={
                tick === 0
                  ? { left: 0 }
                  : tick === 1
                    ? { right: 0 }
                    : { left: `${tick * 100}%`, transform: 'translateX(-50%)' }
              }
            >
              {tick}
            </span>
          ))}
        </span>
        <span />
      </div>

      <div className="survey-tvd__rows">
        {ordered.map((dilemma) => {
          const baseCloser = dilemma.tvd.base < dilemma.tvd.fine_tune;
          const fineTuneCloser = dilemma.tvd.fine_tune < dilemma.tvd.base;
          const left = Math.min(dilemma.tvd.base, dilemma.tvd.fine_tune) * 100;
          const width = Math.abs(dilemma.tvd.base - dilemma.tvd.fine_tune) * 100;
          return (
            <button
              key={dilemma.dilemma_id}
              type="button"
              className={`survey-tvd__row${dilemma.dilemma_id === selectedId ? ' is-selected' : ''}`}
              onClick={() => onSelect(dilemma.dilemma_id)}
              aria-pressed={dilemma.dilemma_id === selectedId}
              aria-label={`${prettyId(dilemma.situation_id)}: base TVD ${score(dilemma.tvd.base)}, fine-tuned TVD ${score(dilemma.tvd.fine_tune)}. ${closerLabel(dilemma.closer)}.`}
            >
              <span className="survey-tvd__label" title={prettyId(dilemma.situation_id)}>
                {prettyId(dilemma.situation_id)}
              </span>
              <span className="survey-tvd__track" aria-hidden="true">
                <span
                  className="survey-tvd__line"
                  style={{ left: `${left}%`, width: `${width}%` }}
                />
                <span
                  className={`survey-tvd__dot survey-tvd__dot--base${baseCloser ? ' is-closer' : ''}`}
                  style={{ left: `${dilemma.tvd.base * 100}%` }}
                />
                <span
                  className={`survey-tvd__dot survey-tvd__dot--ft${fineTuneCloser ? ' is-closer' : ''}`}
                  style={{ left: `${dilemma.tvd.fine_tune * 100}%` }}
                />
              </span>
              <span className={`survey-tvd__badge survey-tvd__badge--${dilemma.closer}`}>
                {closerLabel(dilemma.closer)}
              </span>
            </button>
          );
        })}
      </div>
    </section>
  );
}

function OptionChart({ dilemma, payload }: { dilemma: SurveyDilemma; payload: SurveyPayload }) {
  const series = [
    { key: 'public', label: 'Public', colour: PUBLIC_COLOUR, values: dilemma.shares.public },
    { key: 'base', label: payload.models.base_label, colour: BASE_COLOUR, values: dilemma.shares.base },
    {
      key: 'fine_tune',
      label: payload.models.fine_tune_label,
      colour: FINE_TUNE_COLOUR,
      values: dilemma.shares.fine_tune,
    },
  ];
  const width = 420;
  const height = 280;
  const margin = { top: 26, right: 10, bottom: 28, left: 36 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const groupWidth = plotWidth / Math.max(dilemma.answer_options.length, 1);
  const barWidth = Math.min(28, groupWidth / 4.2);
  const barGap = 5;
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  return (
    <div className="survey-option-chart">
      <div className="survey-option-chart__legend">
        {series.map((entry) => (
          <span key={entry.key} className="survey-option-chart__legend-item">
            <span className="survey-option-chart__swatch" style={{ background: entry.colour }} />
            {entry.label}
          </span>
        ))}
      </div>
      <svg
        className="survey-option-chart__svg"
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={`Answer-option shares for ${prettyId(dilemma.situation_id)}`}
      >
        {ticks.map((tick) => {
          const y = margin.top + plotHeight * (1 - tick);
          return (
            <g key={tick}>
              <line
                x1={margin.left}
                x2={width - margin.right}
                y1={y}
                y2={y}
                stroke="var(--color-border)"
                strokeWidth={1}
              />
              <text
                x={margin.left - 8}
                y={y + 3.5}
                textAnchor="end"
                fontSize={10}
                fill="var(--color-muted)"
              >
                {Math.round(tick * 100)}%
              </text>
            </g>
          );
        })}
        {dilemma.answer_options.map((option, optionIndex) => {
          const center = margin.left + groupWidth * (optionIndex + 0.5);
          return (
            <g key={option}>
              {series.map((entry, seriesIndex) => {
                const value = entry.values[optionIndex] ?? 0;
                const barHeight = plotHeight * value;
                const x = center + (seriesIndex - 1) * (barWidth + barGap) - barWidth / 2;
                const y = margin.top + plotHeight - barHeight;
                return (
                  <g key={entry.key}>
                    <rect
                      x={x}
                      y={y}
                      width={barWidth}
                      height={Math.max(barHeight, value > 0 ? 2 : 0)}
                      rx={3}
                      fill={entry.colour}
                    >
                      <title>{`${entry.label}: ${pct(value)}`}</title>
                    </rect>
                    <text
                      x={x + barWidth / 2}
                      y={y - 6}
                      textAnchor="middle"
                      fontSize={9.5}
                      fill="var(--color-muted)"
                    >
                      {value > 0 ? pct(value) : ''}
                    </text>
                  </g>
                );
              })}
              <text
                x={center}
                y={height - 10}
                textAnchor="middle"
                fontSize={11}
                fontWeight={700}
                fill="var(--color-muted)"
              >
                {optionIndex + 1}
              </text>
            </g>
          );
        })}
      </svg>
      <ol className="survey-option-chart__options">
        {dilemma.answer_options.map((option, optionIndex) => (
          <li
            key={option}
            className={optionIndex === dilemma.top.public ? 'is-public-top' : undefined}
          >
            <span className="survey-option-chart__index">{optionIndex + 1}</span>
            <span>{option}</span>
            {optionIndex === dilemma.top.public && (
              <span className="survey-option-chart__tag">public top</span>
            )}
          </li>
        ))}
      </ol>
    </div>
  );
}

function DilemmaDetail({ dilemma, payload }: { dilemma: SurveyDilemma; payload: SurveyPayload }) {
  const publicTop = dilemma.answer_options[dilemma.top.public] ?? '';
  const baseTop = dilemma.answer_options[dilemma.top.base] ?? '';
  const fineTuneTop = dilemma.answer_options[dilemma.top.fine_tune] ?? '';
  const publicTopShare = dilemma.shares.public[dilemma.top.public] ?? 0;
  const baseTopShare = dilemma.shares.base[dilemma.top.base] ?? 0;
  const fineTuneTopShare = dilemma.shares.fine_tune[dilemma.top.fine_tune] ?? 0;
  const preference = dilemma.preference;
  const voteClass = preference.base_share > preference.fine_tune_share
    ? 'base'
    : preference.fine_tune_share > preference.base_share
      ? 'fine_tune'
      : 'tie';
  const voteText = voteClass === 'base'
    ? 'Base preferred on this dilemma'
    : voteClass === 'fine_tune'
      ? 'Fine-tuned model preferred on this dilemma'
      : 'No preference on this dilemma';
  const verdict =
    dilemma.closer === 'fine_tune'
      ? 'Option shares: fine-tuned model closer'
      : dilemma.closer === 'base'
        ? 'Option shares: base model closer'
        : 'Option shares: equally close';

  return (
    <article className="survey-detail">
      <div className="survey-detail__head">
        <span className="survey-detail__eyebrow">{prettyId(dilemma.profile_id)}</span>
        <h2>{dilemma.question}</h2>
        <details className="survey-detail__story">
          <summary>Read the situation</summary>
          <p>{dilemma.situation}</p>
        </details>
      </div>

      <div className="survey-detail__metrics">
        <span><strong>{dilemma.ratings}</strong> ratings</span>
        <span><strong>{pct(dilemma.tie_share)}</strong> tied</span>
        <span>Base TVD <strong>{score(dilemma.tvd.base)}</strong></span>
        <span>Fine-tune TVD <strong>{score(dilemma.tvd.fine_tune)}</strong></span>
      </div>

      <div className={`survey-detail__vote survey-detail__vote--${voteClass}`}>
        <strong>{voteText}</strong>
        <span>
          Base {pct(preference.base_share)} · fine-tune {pct(preference.fine_tune_share)} of{' '}
          {preference.decided} decided votes
          {' '}({preference.base_wins}–{preference.fine_tune_wins}; {preference.ties} tied).
          {' '}Mean preference {signed(preference.mean)} points
          {' '}(positive = fine-tune).
        </span>
      </div>

      <OptionChart dilemma={dilemma} payload={payload} />

      <div
        className={`survey-detail__verdict survey-detail__verdict--${dilemma.closer}`}
        role="status"
      >
        {verdict}
      </div>

      <div className="survey-detail__reading">
        <p>
          <strong>Public:</strong> {publicTop} ({pct(publicTopShare)} of votes)
        </p>
        <p>
          <strong>Base model:</strong> {baseTop} ({pct(baseTopShare)} of runs)
        </p>
        <p>
          <strong>Fine-tuned model:</strong> {fineTuneTop} ({pct(fineTuneTopShare)} of runs)
        </p>
      </div>
    </article>
  );
}

export default function SurveyResults() {
  const [payload, setPayload] = useState<SurveyPayload | null>(null);
  const [selectedId, setSelectedId] = useState('');
  const [error, setError] = useState('');
  const detailRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    console.info(`[SurveyResults] fetching ${SURVEY_MANIFEST_URL}`);
    cachedFetchJson<SurveyPayload>(SURVEY_MANIFEST_URL, 0)
      .then((data) => setPayload(data))
      .catch((loadError) => {
        console.error('[SurveyResults] load failed:', loadError);
        setError(String(loadError));
      });
  }, []);

  useEffect(() => {
    if (!payload?.dilemmas.length || selectedId) return;
    const requested = new URLSearchParams(window.location.search).get('dilemma');
    const match = payload.dilemmas.find((dilemma) => dilemma.dilemma_id === requested);
    setSelectedId((match ?? payload.dilemmas[0]).dilemma_id);
  }, [payload, selectedId]);

  const selected = useMemo(() => {
    if (!payload?.dilemmas.length) return null;
    return (
      payload.dilemmas.find((dilemma) => dilemma.dilemma_id === selectedId) ??
      payload.dilemmas[0]
    );
  }, [payload, selectedId]);

  const selectDilemma = (dilemmaId: string) => {
    setSelectedId(dilemmaId);
    if (typeof window !== 'undefined') {
      // Keep the selection in the URL so a dilemma can be linked to and
      // survives a refresh (the viewer's tab already uses ?tab=survey).
      const params = new URLSearchParams(window.location.search);
      params.set('tab', 'survey');
      params.set('dilemma', dilemmaId);
      window.history.replaceState(
        null,
        '',
        `${window.location.pathname}?${params.toString()}${window.location.hash}`,
      );
    }
    // On a narrow screen the detail panel sits below the long ranked chart,
    // so a row click would otherwise update something off-screen.
    if (
      typeof window !== 'undefined' &&
      window.matchMedia('(max-width: 980px)').matches
    ) {
      window.requestAnimationFrame(() => {
        detailRef.current?.scrollIntoView({
          behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches
            ? 'auto'
            : 'smooth',
          block: 'start',
        });
      });
    }
  };

  if (error) {
    return (
      <div className="survey-results">
        <DataSourceNotice manifestUrl={SURVEY_MANIFEST_URL} kind="survey results" />
        <div className="survey-results__state">
          <strong>Survey results are not available.</strong>
          <p className="muted-text">
            The aggregate could not be loaded from {SURVEY_MANIFEST_URL}. Run
            <code> survey/analyze.py</code> and sync the artifacts before viewing this tab.
          </p>
        </div>
      </div>
    );
  }

  if (!payload) {
    return (
      <div className="survey-results">
        <DataSourceNotice manifestUrl={SURVEY_MANIFEST_URL} kind="survey results" />
        <p className="muted-text">Loading survey results…</p>
      </div>
    );
  }

  if (!payload.dilemmas.length) {
    return (
      <div className="survey-results">
        <DataSourceNotice manifestUrl={SURVEY_MANIFEST_URL} kind="survey results" />
        <div className="survey-results__state">
          <strong>No survey dilemmas are published yet.</strong>
          <p className="muted-text">
            The aggregate loaded, but it has no per-dilemma results.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="survey-results">
      <DataSourceNotice manifestUrl={SURVEY_MANIFEST_URL} kind="survey results" />

      <div className="survey-results__intro">
        <p className="section-label">Public consultation</p>
        <h2>What the public preferred</h2>
        <p className="survey-results__lede">
          Respondents compared a base-model answer with a fine-tuned answer to the
          same dilemma. The <strong>direct vote</strong> asks which response they
          preferred; the <strong>option-share comparison</strong> asks a different
          question — how close each model&rsquo;s distribution of answer options is
          to the options the public endorsed. The two results need not agree, and
          on this survey they do not.
        </p>
      </div>

      <SummaryCards payload={payload} />

      <div className="survey-results__grid">
        <div className="survey-results__charts">
          <section className="survey-preference" aria-label="Head-to-head preference by dilemma">
            <div className="survey-tvd__head">
              <div>
                <h2>Head-to-head vote</h2>
                <p className="muted-text">
                  <a href="/#rq2">RQ2</a> asks whether users prefer models aligned
                  to their values over the unaligned base model. The pre-registered
                  H2 test uses the 5-point preference scale; positive = fine-tune,
                  negative = base. Bars show each dilemma&rsquo;s mean; the line is
                  its 95% CI. Click a dilemma to see both results for it.
                </p>
              </div>
            </div>
            <Rq2Summary payload={payload} />
            <PreferenceChart
              dilemmas={payload.dilemmas}
              selectedId={selected?.dilemma_id ?? ''}
              onSelect={selectDilemma}
            />
          </section>

          <TvdChart
            dilemmas={payload.dilemmas}
            selectedId={selected?.dilemma_id ?? ''}
            onSelect={selectDilemma}
          />
        </div>
        <div ref={detailRef} className="survey-results__detail-anchor">
          {selected && <DilemmaDetail dilemma={selected} payload={payload} />}
        </div>
      </div>

      <details className="survey-results__method">
        <summary>How to read this</summary>
        <p>{payload.method}</p>
        <p>
          <strong>Direct vote (RQ2):</strong> the 5-point preference rating
          ({payload.preference.base_wins.toLocaleString('en-NZ')} base ·{' '}
          {payload.preference.fine_tune_wins.toLocaleString('en-NZ')} fine-tune ·{' '}
          {payload.preference.ties.toLocaleString('en-NZ')} tied). The Bradley–Terry
          win rate is for the fine-tune (0.5 = no preference); the LMM is the mean
          preference in points (positive = fine-tune). The pre-registered H2 test was{' '}
          {payload.preference.h2_supported ? 'supported' : 'not supported'} under
          the pre-registered rule.
        </p>
        <p>
          <strong>Option shares:</strong> public shares cover{' '}
          {payload.summary.ratings.toLocaleString('en-NZ')} ratings from{' '}
          {payload.summary.respondents.toLocaleString('en-NZ')} respondents across{' '}
          {payload.summary.dilemmas} dilemmas. The survey only fielded dilemmas where
          the two models differed significantly, and each shown dilemma contributed{' '}
          {payload.selection.per_dilemma ?? 17} selected pairs; the public shares
          therefore describe those comparisons rather than every possible run. Model
          shares use {payload.model_share_source}.
        </p>
        <p>
          Data: <a href={SURVEY_MANIFEST_URL}>aggregate JSON</a> · run{' '}
          <code>{payload.run_id}</code> · subset <code>{payload.subset}</code> ·
          generated {new Date(payload.generated_at).toLocaleString('en-NZ')}.
        </p>
      </details>
    </div>
  );
}
