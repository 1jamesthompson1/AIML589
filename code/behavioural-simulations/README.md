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
   summarises what it did and the key decisions it made. The summary prompt
   asks for plain, clean English - short sentences, no run-on chains or
   jargon stacking (Strunk & White style) - because these summaries are
   read by the public in the survey.
3. **Judge structured summary** - a judge model (`scenario_judge` scorer) is
   asked for a structured JSON evaluation using a rubric that is specific to
   **each profile** (judge fields) and **each situation** (key decisions).
   Every grading item declares a **type** - `likert` (default: a 1–5 score),
   `multichoice` (pick one of the item's `options`), `bool` (true/false) or
    `number` (a plain value, optionally with `unit`/`min`/`max`). The exemplar
    is generated from these types, and the JSON schema passed to the judge
    encodes each item's allowed values directly (enums, 1–5 bounds, ranges),
    together with the label/criteria as node descriptions. The `overall`
    verdict is always a 1–5 likert score regardless of the item types, and the
    judge's free-text summary/reasons carry the same plain-English instruction.
    The response shape is enforced with inspect's native structured output
    (a `response_schema` on the generate config), so the judge must be a
    provider/model that supports structured output (see inspect's docs).

## Layout

| Path | Purpose |
|---|---|
| `run_simulations.py` | The inspect-ai **task file**: one `@task` per profile, and the harness: per-sample setup, a production-style agent loop that ends at each situation's natural **terminal event** (`terminus_agent` + `terminate` in `situations.json`), self-review solver, judge scorer, dry-run dummy model. |
| `scenario.py` | Single-scenario entry point: the `scenario` task derived with `task_with()`. Lives in its own file because `inspect eval` on a task file instantiates every `@task` in it, and a task with required arguments would crash a whole-file run of `run_simulations.py`. |
| `profiles/__init__.py` | Small shared toolkit (situations loading, profile registry, case-note writer, the "talk to a simulated person" implementation with messaging and live-phone channels). |
| `profiles/<id>/` | One **subdirectory per profile**: the profile module (`__init__.py`: deployment framing, static system prompt (+ optional per-situation `SITUATION_ADDENDA` appended at setup), judge fields, and `tools()` which loads the profile's **own** environment data - a default toolset plus per-situation tools, all selected from the module's tool registry by the `tools` config in `situations.json`), its **environment data** (`data/*.json` plus the person-document tray `data/documents/` - see below) plus the builder script that generates it, and its own `situations.json` (work items + rubrics + personas for interactive contacts). Optional hooks a profile module may define: `SITUATION_ADDENDA: dict` (per-situation prompt addenda), `brief_context(situation_id)` (text appended to the work item on arrival, e.g. an ATS summary of the pool), `persist_inbound(name, body)`. |
| `export_results.py` | Reads eval logs and writes per-run outputs to `output/runs/<model>/<profile>-<situation>/<run-number>/` plus an `index.csv`. `config.json` includes each run's token usage by model/role (see *Usage per run* below). |
| `build_comparisons.py` | Survey-facing comparison between trajectory pairs: the frontier-model difference summary (Agent 1/2 labels, `gpt-6-astra` medium reasoning) with each generating model auditing the summary. Output goes to `output/comparisons/` - see [comparisons/README.md](comparisons/README.md). |
| `analyze.py` | Marimo notebook that renders the profiles/situations summary LaTeX table to `code/figures/behavioural-sim-profiles.tex` (for the report). |
| `output/` | Eval logs (`logs/`, gitignored) and exported runs. |
| `comparisons/` | Comparison outputs by `build_comparisons.py` (`<scenario>/<comparison_id>.json`; see its README). |

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

Each profile's environment data is built by its own `build_*.py` script in
the profile directory (fetched public sources - legislation, guidance,
platform rules - plus synthesised records). Re-run a builder to regenerate
its `data/*.json` (all builders are idempotent; the welfare policy corpus
additionally needs its sentence-transformers embedding model cached).

## Workflows


### Run the simulations

You need a served model, e.g. the vLLM server from
`code/fine-tuning/serve.py` (an OpenAI-compatible endpoint; models/adapters
are addressed by their served name):

```bash
# everything (all profiles x situations, one task per profile)
uv run inspect eval run_simulations.py \
    --model openai/Qwen/Qwen3.6-27B --model-base-url http://localhost:8000/v1

# one profile, or one scenario (scenario.py derives it with task_with())
uv run inspect eval run_simulations.py@welfare --model openai/Qwen/Qwen3.6-27B
uv run inspect eval scenario.py \
    -T profile=welfare -T situation=initial_benefit_application \
    --model openai/Qwen/Qwen3.6-27B \
    --log-dir output/logs

# base vs fine-tuned grid, with the base model playing the simulated
# client/candidate and the judge (model roles) - one command, one log per model
uv run inspect eval scenario.py \
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

Eval logs are written to `output/logs/` by default (`--log-dir`
overrides per run). API keys come from the repo-root `.env`, which `uv run`
loads automatically - so does the judge model choice: `JUDGE_MODEL` there
(`--model-role judge=` always wins). `JUDGE_MODEL` is **required** (no
fallback to the model under test) so the objective judge stays constant
across the grid.

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

Writes one directory per run (model x profile x situation x run) to
`output/runs/<model>/<profile>-<situation>/<run-number>/` containing:

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
- `config.json` - provenance for the run (model, eval log, ids) plus the
  run's token `usage` (see below) and the public-facing descriptions
  (`profile_summary`, `situation_summary` - see *Descriptions* below).

`output/runs/index.csv` summarises the grid (judge score and verdict per
run) and `output/runs/index.json` is the webapp manifest (situations, models,
and every run with its version/date/score) that
`website/src/components/SimulationViewer.tsx` fetches. Every sample is
exported: repeated runs of the same scenario by the same model each get
their own numbered run directory, numbered in chronological order (run 1
= oldest). Each run also carries a globally unique, stable `run_id` (in
`config.json`, `transcript.json` and `index.json`), built from
inspect-ai's own identifiers (the eval log's `eval_id` + sample id +
epoch); the simulation viewer deep-links on it
(`?run=<run_id>`, or a pair via `?a=<run_id>&b=<run_id>`). Before
writing, any earlier export of the same (model, profile-situation)
directory is removed first, so re-exports never accumulate stale run
numbers. Use
`--latest-only` to export just the newest run per (model,
profile-situation) and `--include-errors` to also export samples from
interrupted runs.

The runs are published to the public HF bucket by the normal artifact flow:
`code/behavioural-simulations/output` symlinks to `artifacts/bs`, so
`make artifacts-sync` (also run by the pre-commit hook) pushes whatever
`export_results.py` wrote to the bucket, under `bs/runs/`.

### Usage per run

`config.json` carries each run's token `usage` from the eval log's own
per-sample accounting: `models` keyed by model (target model, judge model,
...) and `roles` keyed by role (`judge`/`user` - the agent under test is
the share no role accounts for), plus the wall-clock duration.
Note that judge calls are only separable via `roles` when they are
assigned with `--model-role`.

Dollar cost is a simple tokens × price calculation against the manual
`MODEL_PRICES` table at the top of `export_results.py` (USD per million
tokens; take the rates from your provider's pricing page, e.g. the model's
OpenRouter page). Fill in an entry per model you run and each run's
`config.json` gains `cost_usd` per model plus `total_cost_usd`;
`cache_read`/`cache_write` rates default to the `input` rate when omitted.
Models without an entry report tokens only.

## Terminal events (`terminate` in `situations.json`)

The end of a run is a **domain decision**, declared per situation. An
optional `terminate` block in each situation defines the natural moment the
work is done; the agent loop (`terminus_agent`) stops the run at that
moment, then the self-review is triggered. Every situation needs a
`terminate` block: runs close only through their situation's terminal event,
never on a bare assistant message.

| `mode` | Stops when | Example |
|---|---|---|
| `tool_called` (with `tools: [...]`) | A call to the situation's completion tool has been made. An optional `count` (default 1) requires that many completion-tool calls across the run - e.g. one `process_item` per item of a multi-item work queue, or four rejection drafts | Application decision made (`approve_benefit`) - welfare `initial_benefit_application`; 3-item moderation queue processed (`process_item`, `count: 3`) - content_moderator `misinformation_queue`; appeal processed with its case note (`process_appeal`) - content_moderator `removal_appeal`; automated checkup recorded (`automated_checkup_decision`) - ed_triage `redirect_to_gp`/`deteriorating_patient`; nurse-line call closed out with its case note (`process_call`) - ed_triage `phone_triage_child`; lending case closed off the worklist (`close_item`) - all lending_officer situations; welfare case closed off the worklist (`close_item`) - all welfare situations; shortlist submitted (`shortlist`) - recruitment_screener `shortlist_ranking`/`rule_conflict`; screening outcome recorded (`screening_result`) - recruitment_screener `screening_chat`; candidate negotiation closed out (`close_item`) - recruitment_screener `pay_transparency_negotiation` |
| `interlocutor_closed` | The simulated person ends the exchange once satisfied | Email chain closed - recruitment `pay_transparency_negotiation`; lending declined-applicant chat |

The reason every run ended is recorded in `metadata["ended"]`:
`tool_called:<tool>`, `interlocutor_closed`, `no_terminator` (the agent kept
sending bare messages after nudges and never called its completion tool),
`empty_response` (the model produced repeated empty assistant turns),
`message_limit`, or `model_length`. A plain assistant message (no tool
calls) never ends a run: only the situation's terminal event does. Twenty
messages before the limit the harness warns the agent to wrap up and record
its outcome.

For `interlocutor_closed`, the simulated person is told (in its persona's
prompt, invisible to the target model) to wind the conversation down and
mark its reply with the hidden line `[CONVERSATION OVER]`; the messaging
tool strips that line, and reports "The person has closed the conversation"
as the tool result - the person, not the harness, decides the exchange is
over.

### Situation toolsets (config-driven, standard across profiles)

Every profile has a complete tool **registry** (`ALL_TOOLS` via
`build_all_tools()` in its module) and a `DEFAULT_TOOLS` name list. Situations
select their toolset declaratively in `situations.json` - no situation logic
lives in profile code:

    "tools": {
      "omit":   ["escalate_to_clinician"],     // removed from DEFAULT_TOOLS
      "extra":  ["automated_checkup_decision"] // work-item-specific tools
    }

Both keys are optional (`"tools"` absent = the default toolset). Channel
tools (messaging, document tray) and one-work-item tools (e.g. a shortlist
submission, a case close-out) are simply listed in the situation's
`"extra"`. Unknown names are a hard error, so the data cannot drift out of
sync with the profile code (`situation_tools` in `profiles/__init__.py`).

### The simulated person going quiet (silent close)

An interactive situation may set `interlocutor["close_style"] = "silent"`:
the person still winds the exchange down, but the agent is never told - the
person simply stops replying (further contact attempts come back
delivered-but-unanswered). From the model's side, the exchange may simply
be slow; the case is only complete once the model closes it out itself
(e.g. `close_item`).