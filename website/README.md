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
│   └── ResultsViewer.tsx        # Eval results browser (React island)
├── layouts/        # Page shell components
│   └── BaseLayout.astro        # Shared header/footer wrapper
├── styles/
│   └── global.css              # Design tokens and base styles
├── scripts/
│   └── build-data.mjs          # Reads eval CSVs → writes src/data/evals.json
└── data/
    └── evals.json              # Generated eval data (do not edit by hand)
```

## Commands

```bash
npm run dev       # Dev server with hot reload
npm run data      # Refresh eval data from code/fine-tuning/output/evals/
npm run build     # data + Astro build → dist/
npm run preview   # Serve built site locally
```

## Data flow

1. Eval CSVs live in `../code/fine-tuning/output/evals/<model>/<run>/`
2. `npm run data` runs `scripts/build-data.mjs` → writes `src/data/evals.json`
3. Astro imports the JSON at build time; site is fully static

## Deployment

Pushing to `main` triggers a GitHub Action that builds and deploys to GitHub Pages. DNS for `nz-llm.sjhl.nz` is a `CNAME` pointing to `1jamesthompson1.github.io`.

## Adding eval runs

Drop a new run directory under `code/fine-tuning/output/evals/<model>/<run>/` with `config.json` and `per_question_results.csv`, run `npm run data`, then rebuild.
