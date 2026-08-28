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
│   └── TabbedResults.tsx        # Results-viewer tabs (evals / simulations / interact)
├── layouts/        # Page shell components
│   └── BaseLayout.astro        # Shared header/footer wrapper
├── styles/
│   └── global.css              # Design tokens and base styles
```

## Commands

```bash
npm run dev       # Dev server with hot reload
npm run build     # Astro build → dist/
npm run preview   # Serve built site locally
```

## Data flow

Both data viewers fetch **at runtime** from the project's public HF storage
bucket (`1jamesthompson1/wvs-nz-value-alignment-evals`) — nothing is bundled
at build time, so visitors only download the runs they actually select.

### Fine-tuning evals

1. `code/fine-tuning/export_evals_manifest.py` writes the small manifest
   `ft/evals/index.json` (models → runs → configs, no results data); the
   per-run `per_question_results.csv` files stay in the bucket.
2. `make artifacts-sync` pushes them to the bucket (see the repo root
   README; `code/fine-tuning/output` symlinks to `artifacts/ft`).
3. `ResultsViewer.tsx` fetches the manifest, then — only when a model is
   selected — fetches that run's `config.json` + `per_question_results.csv`
   and parses them in the browser (papaparse).

### Behavioural-simulation runs (live from the HF bucket)

`SimulationViewer.tsx` fetches run data **at runtime** from the project's
public HF storage bucket — no build-time bundling. It loads the webapp
manifest (`bs/runs/index.json`, written by
`code/behavioural-simulations/export_results.py`) and, on demand, each run's
`transcript.json` / `judge.json` / `self_review.txt` / `audit.txt` via
`https://huggingface.co/buckets/<bucket>/resolve/bs/runs/<run_dir>/...`. The
bucket is kept up to date by `make artifacts-sync` (see the repo root
README; `code/behavioural-simulations/output` symlinks to `artifacts/bs`).
Override the manifest URL for testing with `?manifest=<url>`.

Client-side caching (`src/lib/cachedFetch.ts`) stores fetched files in
localStorage with a 1-hour TTL so repeat views are instant, and uses
`cache: 'no-store'` on the fetch so the browser never reuses an expired
signed CDN redirect. The **manifest is always re-fetched (0 TTL)**: it is
the small gatekeeper that lists which runs exist, so a freshly exported
batch shows up on the next visit rather than serving a stale copy. If you
re-exported and still see old data, hard-reload and clear the site's
`hf-bucket:*` localStorage keys (DevTools → Application → Local Storage).

The viewer lets visitors pick a situation, then select two runs to compare
side by side — different models, or different runs (versions) of the same
model and scenario. Clicking a message highlights the matching step on the
other side.

## Deployment

Pushing to `main` triggers a GitHub Action that builds and deploys to GitHub Pages. DNS for `nz-llm.sjhl.nz` is a `CNAME` pointing to `1jamesthompson1.github.io`.

## Adding eval runs

Drop a new run directory under `code/fine-tuning/output/evals/<model>/<run>/` with `config.json` and `per_question_results.csv`, run `uv run export_evals_manifest.py` in `code/fine-tuning/`, then `make artifacts-sync` (or commit — the pre-commit hook syncs). The site picks it up automatically; no rebuild needed.
