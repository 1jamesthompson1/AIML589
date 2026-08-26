# Behavioural simulations

Behavioural simulations of the (base and fine-tuned) LLMs: each model is
placed in an agentic ReAct harness, given a **profile** (a realistic,
production-style agentic setup — persona, tools, knowledge base) and a
**situation** (a work item to handle), and its behaviour is evaluated. Built
on [inspect-ai](https://inspect.aisi.org.uk) as a proper inspect-ai **task
file**, so you get the standard inspect workflow: `inspect eval` CLI, model
grids with `--model-spec`, model roles, per-sample sandboxes, and eval logs.

The setups deliberately mirror real production tooling and working language
(a deployed MSD work assistant, a recruitment screening assistant). There is
no test framing anywhere in the agent's prompt: cases are presented as live
work items, and agents are expected to work them on the evidence via their
tools. The point is to prevent "obvious test-time behaviour" — models
acting as if they are being evaluated rather than doing the job.

For each run the pipeline produces three things:

1. **Trajectory** — the full run logged by inspect-ai: agent reasoning, tool
   calls and observations, and (for interactive situations) the dialogue with
   a simulated client/candidate.
2. **Model summary** — after the run, the model reviews its own trajectory
   (`self_review_summary` solver) and summarises what it did and the key
   decisions it made.
3. **Judge structured summary** — a judge model (`scenario_judge` scorer) is
   asked for a structured JSON evaluation using a rubric that is specific to
   **each profile** (judge fields) and **each situation** (key decisions).
   Every grading item declares a **type** — `likert` (default: a 1–5 score),
   `multichoice` (pick one of the item's `options`), `bool` (true/false) or
   `number` (a plain value, optionally with `unit`/`min`/`max`). The exemplar
   and the judge's grading scheme are generated from these types, so the
   judge is always told exactly what shape each answer must take. The `overall`
   verdict is always a 1–5 likert score regardless of the item types.

## Layout

| Path | Purpose |
|---|---|
| `run_simulations.py` | The inspect-ai **task file**: one `@task` per profile (plus a `scenario` task derived via `task_with()`), and the harness: per-sample setup, a production-style ReAct agent (`react`, no submit tool), self-review solver, judge scorer, dry-run dummy model. |
| `profiles/__init__.py` | Small shared toolkit (JSON loading, profile registry, case-note writer, the "talk to a simulated person" implementation). |
| `profiles/<id>/` | One **subdirectory per profile**: the profile module (`__init__.py`: deployment framing, judge fields, `SYSTEMS` groupings, its constant `tools()`), its **environment data** (`data/*.json` — see below) and its own `situations.json` (work items + rubrics + personas for interactive contacts). |
| `export_results.py` | Reads eval logs and writes per-run `trajectory.*`, `model_summary.json`, `judge_summary.json`, `config.json` to `output/runs/<model>/<profile>-<situation>/` plus an `index.csv`. |
| `analyze.py` | Marimo notebook that renders the profiles/situations summary LaTeX table to `code/figures/behavioural-sim-profiles.tex` (for the report). |
| `output/` | Eval logs (`logs/`, gitignored) and exported runs. |

### Environment data (`profiles/<id>/data/`)

Each profile's simulated world — everything its tools read from — lives in
its `data/` directory as JSON datasets (loaded by `knowledge_base()` in
`profiles/__init__.py`, keyed by file stem). This is the environment data:
client records, income/bank data, policy corpora. The tools are the only way
agents can access it; nothing is put in the prompt.

For the **welfare** profile, the policy library is real: built by
`profiles/welfare/build_welfare_policy_db.py` from public MSD documents —

- the **entire Social Security Act 2018**
  ([legislation.govt.nz](https://www.legislation.govt.nz)), parsed from the
  whole-act HTML into one chunked entry per section (~1,100 chunks), and
- **Work and Income public guidance pages**
  ([workandincome.govt.nz](https://www.workandincome.govt.nz)): debt
  repayment, obligations, income reporting, Jobseeker Support, Special
  Needs Grants, Emergency Benefit, Recoverable Assistance Payment,
  Temporary Additional Support.

Both sources are Crown copyright. The build also writes a simple vector
index (`policy_vectors.npz`: one normalised embedding per chunk,
`sentence-transformers/all-MiniLM-L6-v2`, a project dependency), which
`lookup_msd_policy` uses for semantic RAG-style retrieval at eval time; a
lexical scorer is used as a fallback if the index or model is unavailable.
To rebuild after editing sources:

```bash
uv run profiles/welfare/build_welfare_policy_db.py            # fetch + rebuild data/
uv run profiles/welfare/build_welfare_policy_db.py --dry-run  # stats only
```

Client/income datasets (`clients.json`, `income_data.json`) remain
synthetic (no real client data exists or should exist).

For the **recruitment_screener** profile, the environment is built for
production realism:

- **Candidate CVs** (`data/cvs/*.pdf`): one full uploaded document per
  candidate, each in a different style/format (plain text CV, formal prose
  CV, designed two-column template, corporate template, internal application
  form, skills-forward layout, metrics/certs-heavy CV, academic CV) — as a
  real ATS would hold them. The agent reads documents via
  `read_candidate_cv`; ATS records (`candidates.json`) hold only short
  database summaries. Regenerate with:
  `uv run profiles/recruitment_screener/build_recruitment_assets.py`.
- **Hiring policies** (`data/policy.json`): six long-form fictional
  Aotearoa Technology policies (recruitment & selection, EEO,
  right-to-work, hiring instructions/escalation, pay transparency, privacy &
  communication). Each is adapted from openly licensed NZ government guidance
  (CC BY 4.0: Public Service Commission, Employment New Zealand, Privacy
  Commissioner, Immigration NZ) with the source noted in each entry — no
  copyrighted corporate policy text is used. Maintained directly as JSON;
  `lookup_company_policy` ranks and returns whole documents.
- **Sandboxed web** (`data/web_cache.json`): the corpus served by the
  `web_search` / `web_read` tools ("approved web research gateway"). Pages
  simulate what an agent might find researching screening questions
  (employment law pages, HR practice articles, candidate-perspective forum
  threads), adapted from the same open sources plus clearly fictional pages.
  The gateway is offline by design — no live network at eval time — so runs
  stay deterministic and reproducible. Maintained directly as JSON.

Current placeholder profiles (2 profiles × 5 situations):

- **`welfare`** (Ministry of Social Development benefits case officer):
  initial application with an income inconsistency (interactive); overpayment
  recovery for a sole parent; anomaly-flagged interview (interactive);
  discretionary emergency grant; work-capacity medical certificate review.
- **`recruitment_screener`** (recruiting screening assistant in an ATS):
  shortlist ranking of 8 CVs (career-gap, older, self-taught and
  overseas-qualified candidates embedded); screening chat (interactive);
  hiring-manager right-to-work rule conflict; rejection messages;
  pay-transparency negotiation (interactive).

## Workflows


### Run the simulations

You need a served model, e.g. the vLLM server from
`code/fine-tuning/serve.py` (an OpenAI-compatible endpoint; models/adapters
are addressed by their served name):

```bash
# everything (all profiles x situations, one task per profile)
uv run inspect eval run_simulations.py \
    --model openai/Qwen/Qwen3.6-27B --model-base-url http://localhost:8000/v1

# one profile, or one scenario (derived with task_with())
uv run inspect eval run_simulations.py@welfare --model openai/Qwen/Qwen3.6-27B
uv run inspect eval run_simulations.py@scenario \
    -T profile=welfare -T situation=initial_benefit_application \
    --model openai/Qwen/Qwen3.6-27B \
    --log-dir output/logs

# base vs fine-tuned grid, with the base model playing the simulated
# client/candidate and the judge (model roles) — one command, one log per model
uv run inspect eval run_simulations.py@scenario \
    -T profile=recruitment_screener -T situation=screening_chat \
    --model-spec '{model: openai/Qwen/Qwen3.6-27B, temperature: 0}' \
    --model-spec '{model: openai/Qwen/Qwen3.6-27B-nz-wvs-modal_response-cluster_0, temperature: 0}' \
    --model-role user=openai/Qwen/Qwen3.6-27B \
    --model-role judge=openai/Qwen/Qwen3.6-27B \
    --log-dir output/logs
```

Every sample runs in its own sandbox (`sandbox="local"`: a fresh isolated
working directory per sample; file-writing tools only touch the sandbox and
the agent can never execute arbitrary code).

Eval logs are written to `output/logs/` by default (`INSPECT_LOG_DIR`
in this directory's `.env`; override per-run with `--log-dir`).

### Smoke test without a GPU

```bash
uv run run_simulations.py --dry-run   # programmatic eval with the scripted dummy model
uv run export_results.py
```

Exercises the whole pipeline (sandbox, trajectory, self-summary, judge) for
all profiles × situations offline, so the plumbing can be verified before
pointing it at real models.

### Export and inspect results

```bash
uv run export_results.py
```

Writes per-run files as described above; `output/runs/index.csv` summarises
the grid (judge score and verdict per run).

### Profiles / situations summary table (for the report)

```bash
uv run marimo edit analyze.py   # then run all cells
```

`analyze.py` renders a booktabs LaTeX table of every profile (as the table
index) with its situations and whether each is interactive, and writes it to
`code/figures/behavioural-sim-profiles.tex` — a bare tabular ready for the
report via `\ctable{behavioural-sim-profiles.tex}`. The table is data-driven
off `profiles/`, so it picks up new profiles or situations automatically.

## Notes

- The agent terminates production-style: `react(submit=False)` ends the run
  the moment the model produces a message without calling a tool — no synthetic
  `submit()` tool, no evaluation language. The agent's closing message *is* the
  outcome (a deployed assistant's final response), which is what the summary
  and judge see.
- Judge calls are logged as ordinary model events in the eval log, so the
  whole evaluation is auditable. Judge `value` is the 1–5 overall score; the
  structured JSON is in the score's `explanation`/`metadata`.