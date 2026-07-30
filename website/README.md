# Project website

Static site for the NZ Value Alignment research project, hosted at [nz-value-llm.sjhl.nz](https://nz-value-llm.sjhl.nz).

Built with [Astro](https://astro.build) + React for interactive components.

## Structure

```
src/
├── pages/
│   └── index.astro          # One-page site (participate, methodology, findings)
├── components/
│   ├── PipelineAnimation.astro   # Pipeline graphic
│   ├── ResultsViewer.tsx         # Eval results browser (React island)
│   └── Nav.astro                 # Top navigation
├── layouts/
│   └── BaseLayout.astro          # Page shell with nav + footer
├── styles/
│   └── global.css                # Design tokens and base styles
├── scripts/
│   └── build-data.mjs           # Processes eval CSVs → JSON at build time
└── data/
    └── evals.json                # Generated eval data (do not edit)
```

## Workflow

### Data refresh

The site reads evaluation data from `../code/fine-tuning/output/evals/`. Run this whenever new eval runs appear:

```bash
cd website
npm run data
```

This walks all model/run directories, parses `per_question_results.csv`, and writes structured JSON to `src/data/evals.json`.

### Development

```bash
cd website
npm run dev        # Starts dev server with hot reload
```

### Build

```bash
npm run build      # Runs build-data + Astro build → dist/
```

### Preview

```bash
npm run preview    # Serves the built site locally
```

## Deployment

### GitHub Pages

The site is built and deployed automatically. Push to `main` and a GitHub Action (see `.github/workflows/` if one exists) will build and deploy to GitHub Pages configured at `nz-value-llm.sjhl.nz`.

To set up from scratch:

1. Go to repo Settings → Pages
2. Source: **GitHub Actions**
3. The repo may already have a deploy workflow; if not, add one.

### Custom domain

The DNS for `nz-value-llm.sjhl.nz` should be a `CNAME` pointing to `1jamesthompson1.github.io`. Set the custom domain in the repo Pages settings.

## Data processing

The build pipeline (`npm run build`) runs the data script first, then Astro. This means:

1. Eval CSVs are read and reshaped into a queryable JSON tree
2. Astro imports the JSON at build time and passes it to the ResultsViewer component
3. The site is fully static — no API calls at runtime

## Adding evaluation runs

Any new run directory under `code/fine-tuning/output/evals/<model>/<run>/` with a `config.json` and `per_question_results.csv` will automatically be picked up by the build-data script. Just run `npm run data` (or `npm run build`) and redeploy.
