# Project website

Static site for the NZ Value Alignment research project, hosted at [nz-llm.sjhl.nz](https://nz-llm.sjhl.nz).

Built with [Astro](https://astro.build) + React for interactive components.

## Source layout

```
src/
├── pages/          # Routes — each file maps to a URL path
│   ├── index.astro            # / — landing (hero, methodology pipeline, call to action)
│   ├── join-survey.astro      # /join-survey
│   ├── results-viewer.astro   # /results-viewer — eval data browser
│   └── about.astro            # /about — project background and team
├── components/     # Reusable UI
│   ├── Nav.astro                # Top navigation bar
│   ├── PipelineAnimation.tsx    # Interactive pipeline graphic (React island)
│   ├── ResultsViewer.tsx        # Eval results browser (React island)
│   ├── SimulationViewer.tsx     # Behavioural-simulation runs browser (React island)
│   ├── DilemmasViewer.tsx       # Dilemma runs browser (React island)
│   ├── DataSourceNotice.tsx     # Local/bucket source indicator and switch link
│   └── TabbedResults.tsx        # Results-viewer tabs (evals / simulations / dilemmas / interact)
├── layouts/        # Page shell components
│   └── BaseLayout.astro        # Shared header/footer wrapper
├── lib/            # Runtime data-source and fetch helpers
└── styles/
    └── global.css              # Design tokens and base styles
```

## Commands

```bash
npm run dev       # Dev server with hot reload
npm run build     # Astro build → dist/
npm run preview   # Serve built site locally
```

Stop a running dev server before `npm run build`: both commands share the
`node_modules/.vite` dependency cache, and a production build can overwrite
it with production React, breaking the dev server with `_jsxDEV is not a
function`. If that happens, stop the dev server (`astro dev stop`), delete
`node_modules/.vite`, and start it again.

## Data flow

All data viewers fetch **at runtime**; heavy artifacts are not bundled by a
production build. The default source depends on the environment:

- `npm run dev` reads `artifacts/ft`, `artifacts/bs` and `artifacts/dilemmas`
  directly from the repository root through a dev-server middleware. It does
  not require a symlink, a download, or an HF sync.
- Production builds and `npm run preview` read the public HF storage bucket
  (`1jamesthompson1/wvs-nz-value-alignment-evals`).
- `?local=0` forces the bucket in development; `?local=1` forces local files
  where they are served. `?manifest=<url>` remains an explicit manifest
  override for testing.
- The source indicator above each viewer shows which source is active and
  links to the other one.

Local run files deliberately bypass the browser cache, so replacing a CSV,
transcript, judge file, or manifest in `artifacts/` is visible on refresh. If a
local manifest is absent (for example, before the next simulation export),
the viewer reports that clearly and does not silently fall back to the bucket.
Do not use the local-artifact mode for a production build.

## Inspecting local and remote artifacts

The Makefile only handles the project workflows. To inspect the public bucket
without changing it, use the HF CLI directly:

```bash
hf buckets list 1jamesthompson1/wvs-nz-value-alignment-evals -R --tree
hf buckets info 1jamesthompson1/wvs-nz-value-alignment-evals
```

`make artifacts-sync` is intentionally non-deleting: it uploads/changes local
artifacts but leaves historical remote objects in place. There is deliberately
no whole-bucket delete target: the local mirror may be incomplete while a new
simulation batch is being prepared. The current project has remote historical
logs and older comparison directories that are not present in the local
mirror; review a prefix-specific plan and replace only the relevant remote
prefix after the new local batch has been validated.

`make artifacts-pull` is only for intentionally downloading the remote mirror
into `artifacts/`; the website dev server does not call it.

### Fine-tuning evals

`code/fine-tuning/analyze.py` writes the webapp manifest
`artifacts/ft/evals/index.json` (models → runs/configs plus the question
inventory). Each selected run's `per_question_results.csv` is fetched lazily
and parsed in the browser. The viewer exposes all manifest models, selected
runs, prompt variants, subpopulations, question-level distributions, and
aggregate row/accuracy/KL/CE/TVD summaries. The question search is fuzzy and
matches the full question text, sub-item, column name, and answer options, so
typing `honest` finds items whose question or response scale contains it. Run
metadata includes elapsed runtime when it is present in `config.json`;
fine-tuning evaluations do not currently record provider token usage or cost.

### Behavioural simulation runs

`code/behavioural-simulations/export_results.py` writes
`artifacts/bs/runs/index.json` and one directory per run containing
`transcript.json`, `judge.json`, `self_review.txt`, and `config.json`.
The manifest includes judge score/status and, after re-export, duration,
token usage, and priced cost. The viewer also fetches `config.json` as a
fallback for older manifests, so runtime/token/cost metadata is displayed for
both old and newly exported local runs.

The simulation viewer provides:

- a respondent-view preview at the top of the comparison panel: the survey
  block's work-profile and situation descriptions, the "this is what happened"
  summary, and the two questions with their answer scales, exactly as
  respondents see them (agent model identities stay in the researcher-facing
  header);
- a collapsed judge-distribution panel with one grouped bar chart per
  individual item in the structured judging object (overall fields, each
  profile-assessment field, and each key decision); numeric 1–5 items use
  score bins, while booleans and categorical decisions use observed outcome
  bins, with adjacent bars for the compared agents (or models in the
  single-run view);
- a checkbox that limits the model/run list to runs participating in an
  indexed comparison summary;
- paired judge-item tables and comparison-backed item charts in the
  comparison view; if an old comparison file points at an older run ID, the
  viewer pairs the current runs and labels that provenance explicitly;
- audit fairness ratings (for example, `4/5`) always visible, while the long
  audit correction/raw text is hidden until its **Show audit content** button is
  pressed.

There is no per-run `audit.txt`. Cross-model audits are embedded under
`audits` in each `artifacts/bs/comparisons/...json` document.

### Dilemma runs

`code/dilemmas/export_results.py` writes one manifest per exported full-run
batch (`artifacts/dilemmas/runs/full/<run-id>/index.json`, schema
`wvs-dilemma-runs-index/v2`: profiles with their one-sentence role summary and
system prompts, the dilemma inventory, models, and every run) plus one
directory per run containing `response.txt`, `judge.json`, `config.json`, and
the transcript. `artifacts/dilemmas/runs/batches.json` lists every exported
batch for the **Batch** selector.

The dilemma viewer shows one card per model for the selected dilemma, each
with a run selector so a specific response (run 1–N) can be chosen, the
response text, and the judge's classification (decision, reason, and — when
the run's `judge.json` carries it — the verbatim quote the judge picked to
show what the model valued). The stimulus keeps the survey block's framing
(role lead → the profile's dilemma intro → situation → outro → question, with
the role lead and outro mirrored from `survey/comparison-question.md` and the
profile JSON), shows
the profile's role summary and the situation the model was given, and keeps
the selected profile's system prompt collapsed under it (the prompt actually
used by the exported runs; older manifests without them simply omit them).

At the top of the viewer, above the dilemma selector, a collapsed **Dilemma
statistics** table lists every dilemma in the manifest with the same summary
metrics the report's `code/dilemmas/analyze.py` tables use, computed over the
runs the chart plots: **balance** of the pooled outcomes (H / ln k over all
runs, 1.00 = even spread), **balance (not other)** (the same H / ln k over the
decided options only — "Other or unclear" dropped and the shares
renormalised, so balance the catch-all earned is not mistaken for the models
splitting), **avg divergence** between the models (mean pairwise total
variation distance between their outcome distributions) and **avg stability**
(the mean over models of 1 - H / ln k within each model's own runs). Columns
are sortable by clicking a heading (numeric columns start largest-first, text
columns A→Z; default is avg divergence descending) and the selected dilemma's
row is highlighted; clicking a row opens that dilemma in the viewer and
collapses the table.

Under the stimulus, an **outcome distribution** chart shows, per answer
option, the share of each model's exported runs the judge placed on it
(grouped bars; the answer options are listed underneath). The series are
curated from the dilemma's `choice_counts`: the Qwen 3.8 family (base and
fine-tunes) plus deepseek-v4.1-flash and glm-5.3-flash when they have runs for
the dilemma — spot-test extras (haiku-5.5, gpt-6-luna) never appear, and
when no Qwen runs exist the chart falls back to the frontier pair.

A **Batch** selector above the dilemma picker lists every exported full run
(`runs/batches.json`) and loads the selected batch's own manifest
(`runs/full/<run-id>/index.json`); it defaults to the newest batch. There is
no merged/live view, and spot runs are not shown in it.

An older runs index without the dilemma inventory still works: the viewer
falls back to a selected run's `config.json` for the dilemma text.
## Deployment

Pushing to `main` triggers a GitHub Action that builds and deploys to GitHub
Pages. DNS for `nz-llm.sjhl.nz` is a `CNAME` pointing to
`1jamesthompson1.github.io`.

## Adding eval runs

1. Add a run directory under `code/fine-tuning/output/evals/<model>/<run>/`
   with `config.json` and `per_question_results.csv`.
2. Run `uv run python code/fine-tuning/analyze.py` to regenerate
   `artifacts/ft/evals/index.json`.
3. For simulations, run the export workflow in
   `code/behavioural-simulations/README.md` to regenerate the run/comparison
   manifests.
4. For dilemmas, run `uv run python code/dilemmas/export_results.py` to
   regenerate the runs manifest as described in `code/dilemmas/README.md`.
5. Use the normal repository artifact workflow when the results should be
   published; the website itself never syncs or downloads artifacts.

The site picks up a new manifest on refresh. It does not need a rebuild for a
runtime-only data update.
