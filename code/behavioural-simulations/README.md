# Behavioural simulations

Behavioural simulations of the (base and fine-tuned) LLMs: each model is
placed in an agentic ReAct harness, given a **profile** (a realistic,
production-style agentic setup - persona, tools, environment data) and a
**situation** (a work item to handle), and its behaviour is evaluated. Built
on [inspect-ai](https://inspect.aisi.org.uk) as a proper inspect-ai **task
file**, so you get the standard inspect workflow: `inspect eval` CLI, model
grids with `--model-spec`, model roles, per-sample sandboxes, and eval logs.

The setups deliberately mirror real production tooling and working language
(a deployed MSD work assistant, a recruitment screening assistant). There is
no test framing anywhere in the agent's prompt: cases are presented as live
work items, and agents are expected to work them on the evidence via their
tools. The point is to prevent "obvious test-time behaviour" - models
acting as if they are being evaluated rather than doing the job.

For each run the pipeline produces three things:

1. **Trajectory** - the full run logged by inspect-ai: agent reasoning, tool
   calls and observations, and (for interactive situations) the dialogue with
   a simulated client/candidate.
2. **Model summary** - once the run ends (at its terminal event, below),
   the model reviews its own trajectory (`self_review_summary` solver) and
   summarises what it did and the key decisions it made.
3. **Judge structured summary** - a judge model (`scenario_judge` scorer) is
   asked for a structured JSON evaluation using a rubric that is specific to
   **each profile** (judge fields) and **each situation** (key decisions).
   Every grading item declares a **type** - `likert` (default: a 1–5 score),
   `multichoice` (pick one of the item's `options`), `bool` (true/false) or
   `number` (a plain value, optionally with `unit`/`min`/`max`). The exemplar
   and the judge's grading scheme are generated from these types, so the
   judge is always told exactly what shape each answer must take. The `overall`
   verdict is always a 1–5 likert score regardless of the item types.

## Layout

| Path | Purpose |
|---|---|
| `run_simulations.py` | The inspect-ai **task file**: one `@task` per profile (plus a `scenario` task derived via `task_with()`), and the harness: per-sample setup, a production-style agent loop that ends at each situation's natural **terminal event** (`terminus_agent` + `terminate` in `situations.json`), self-review solver, judge scorer, dry-run dummy model. |
| `profiles/__init__.py` | Small shared toolkit (situations loading, profile registry, case-note writer, the "talk to a simulated person" implementation). |
| `profiles/<id>/` | One **subdirectory per profile**: the profile module (`__init__.py`: deployment framing, static system prompt, judge fields, and `tools()` which loads the profile's **own** environment data), its **environment data** (`data/*.json` - see below) and its own `situations.json` (work items + rubrics + personas for interactive contacts). |
| `export_results.py` | Reads eval logs and writes per-run outputs to `output/runs/<model>-<timestamp>/<profile>-<situation>/` plus an `index.csv`. |
| `analyze.py` | Marimo notebook that renders the profiles/situations summary LaTeX table to `code/figures/behavioural-sim-profiles.tex` (for the report). |
| `output/` | Eval logs (`logs/`, gitignored) and exported runs. |

### Environment data (`profiles/<id>/data/`)

Each profile's simulated world - everything its tools read from - lives in
its `data/` directory as JSON datasets. **Each profile module parses its own
data files itself** (`tools()` loads what it needs, in whatever layout suits
that profile): there is no shared "knowledge base" loader with a fixed
file-keyed shape. Files are laid out per profile: client records,
income/bank data, policy corpora, candidate pools, web cache... The tools
are the only way agents can access it; nothing is put in the prompt.

Current profiles (5 profiles × 5 situations):

| Profile | Directory |
|---|---|
| `welfare` | [profiles/welfare/](profiles/welfare/) |
| `recruitment_screener` | [profiles/recruitment_screener/](profiles/recruitment_screener/) |
| `content_moderator` | [profiles/content_moderator/](profiles/content_moderator/) |
| `lending_officer` | [profiles/lending_officer/](profiles/lending_officer/) |
| `ed_triage` | [profiles/ed_triage/](profiles/ed_triage/) |

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
# client/candidate and the judge (model roles) - one command, one log per model
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

Writes one directory per run (model x profile x situation) to
`output/runs/<model>-<timestamp>/<profile>-<situation>/` containing:

- `transcript.json` - the run as a structured JSON document
  (schema tag `wvs-run-transcript/v1`): a `run` header (model, profile,
  situation, type/interactivity) followed by the flat `messages` stream -
  system prompt, work item, agent turns with optional `tool_calls`,
  matching `tool` results and the closing outcome message. Designed to be
  loaded by the public webapp and reusable as training data.
- `transcript.md` - the same run as readable text (for the report),
- `self_review.txt` - the model's own summary of its work (plain text),
- `judge.json` - the judge's structured, rubric-based evaluation plus its
  overall 1–5 score,
- `audit.txt` - the audit judge's fact-check statement on the self-review
  (plain text),
- `config.json` - provenance for the run (model, eval log, ids).

`output/runs/index.csv` summarises the grid (judge score and verdict per
run) and `output/runs/index.json` is the webapp manifest (situations, models,
and every run with its version/date/score) that
`website/src/components/SimulationViewer.tsx` fetches. Every sample is
exported: repeated runs of the same scenario by the same model are kept as
numbered versions (v1 = oldest; the most recent version keeps the plain
directory name, older ones get a `-v<N>` suffix). The parent directory is
the model name plus the export session's timestamp; on each export any
earlier session for that model is removed first, so re-exports never
accumulate duplicate sessions. Use
`--latest-only` to export just the newest version per (model,
profile-situation) and `--include-errors` to also export samples from
interrupted runs.

The runs are published to the public HF bucket by the normal artifact flow:
`code/behavioural-simulations/output` symlinks to `artifacts/bs`, so
`make artifacts-sync` (also run by the pre-commit hook) pushes whatever
`export_results.py` wrote to the bucket, under `bs/runs/`.

### Profiles / situations summary table (for the report)

## Terminal events (`terminate` in `situations.json`)

The end of a run is a **domain decision**, declared per situation. An
optional `terminate` block in each situation defines the natural moment the
work is done; the agent loop (`terminus_agent`) stops the run at that
moment, then the self-review is triggered. Omitting `terminate` keeps the
classic rule: the run ends when the model produces a message without a tool
call.

| `mode` | Stops when | Example |
|---|---|---|
| `tool_called` (with `tools: [...]`) | The last action was one of the listed decision tools | Application decision made (`approve_benefit`) — welfare `initial_benefit_application` |
| `interlocutor_closed` | The simulated person ends the conversation once satisfied | Email chain closed — recruitment `pay_transparency_negotiation` |

The reason every run ended is recorded in `metadata["ended"]`:
`tool_called:<tool>`, `interlocutor_closed`, `closing_message` (no tool call
and no terminus), `message_limit`, or `model_length`.

For `interlocutor_closed`, the simulated person is told (in its persona's
prompt, invisible to the target model) to wind the conversation down and
mark its reply with the hidden line `[CONVERSATION OVER]`; the messaging
tool strips that line, and reports "The person has closed the conversation"
as the tool result - the person, not the harness, decides the exchange is
over. (The same mechanism will carry the phone/`[CALL ENDED]` case later.)

## Notes

- Profiles with no terminal-event machinery yet (content_moderator,
  ed_triage, lending_officer - their toolsets are not built) simply end on
  the closing message; `terminate` blocks can be added as their tools land.
- Judge calls are logged as ordinary model events in the eval log, so the
  whole evaluation is auditable. Judge `value` is the 1–5 overall score; the
  structured JSON is in the score's `explanation`/`metadata`.