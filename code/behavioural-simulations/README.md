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
no value-probe or answer framing in the agent's prompt: cases are presented as
live work items, and agents are expected to work them on the evidence via
their tools. Clinical safety constraints and source-authority rules remain
explicit; they are not experimental answer hints. The point is to prevent
"obvious test-time behaviour" - models acting as if they are being evaluated
rather than doing the job.

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
    `number` (a plain value, optionally with `unit`/`min`/`max`), or
    `ranking` (an ordered list of distinct `options`, with required `count`;
    an empty list means no ranking was submitted). Ranking length, membership
    and uniqueness are also checked locally and invalid replies retried. The exemplar
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
| `profiles/<id>/` | One **subdirectory per profile**: the profile module (`__init__.py`: deployment framing, judge fields, and `tools()` which loads the profile's **own** environment data - a default toolset plus per-situation tools, all selected from the module's tool registry by the `tools` config in `situations.json`), its **environment data** (`data/*.json` plus the person-document tray `data/documents/` - see below) plus the builder script that generates it, its own `situations.json` (`case_types` - the prompt-level name + instructions for each kind of case - plus situations (work items, rubrics, personas for interactive contacts, each referencing its `case_type`)), and optionally a `templates/system_prompt.jinja2` template (profiles without one use their `SYSTEM_PROMPT` string). Optional hooks a profile module may define: `brief_context(situation_id)` (text appended to the work item on arrival, e.g. an ATS summary of the pool), `persist_inbound(name, body)`. |
| `export_results.py` | Reads eval logs and writes per-run outputs to `output/runs/<model>/<profile>-<situation>/<run-number>/` plus an `index.csv`. `config.json` includes each run's token usage by model/role (see *Usage per run* below). |
| `build_comparisons.py` | Survey-facing comparison between completed trajectory pairs: a difference summary (`gpt-5.6-luna` by default) with each generating model auditing it. Output goes to `output/comparisons/`. |
| `analyze.py` | Marimo notebook: renders the profiles/situations summary LaTeX table to `code/figures/behavioural-sim-profiles.tex`. The website computes item-level grouped judge distributions directly from the exported manifests and judge files; use the analysis/review scripts appropriate to the question being asked rather than assuming this notebook contains the full comparison analysis. |
| `output/` | Eval logs (`logs/`), exported runs (`runs/`), and comparisons (`comparisons/`); all are gitignored artifacts. |
| `output/comparisons/` | Comparison outputs by `build_comparisons.py` (`<model-pair>/<scenario>/<comparison_id>.json`). Its `README.md` links the current manual review. |

The exported manifests are the source of truth for the website. The local
`artifacts/bs` mirror and the public bucket can contain different batches, and
the bucket is not automatically pruned when a local export changes. Inspect
both inventories before interpreting results: the command
`hf buckets list 1jamesthompson1/wvs-nz-value-alignment-evals -R --tree` shows
the published tree, while `artifacts/bs/runs/index.json` shows the local export.
Match run IDs and creation dates when adding comparison files to avoid mixing
harness versions.

### Environment data (`profiles/<id>/data/`)

Each profile's simulated world - everything its tools read from - lives in
its `data/` directory as JSON datasets. **Each profile module parses its own
data files itself** (`tools()` loads what it needs, in whatever layout suits
that profile): there is no shared "knowledge base" loader with a fixed
file-keyed shape. Files are laid out per profile: client records,
income/bank data, policy corpora, candidate pools, signed-off decision
records. The tools
are the only way agents can access it; nothing is put in the prompt.

Fictional records use plausible web and email addresses to avoid signalling
the experiment to the agent. These addresses are fixture text, not verified
contacts or live sources; keep them consistent across documents and builders.
Moderation policy source notes are retained in the data for human reference
and omitted from agent-facing tool responses.

Current profiles (5 profiles × 5 situations):

| Profile | Directory |
|---|---|
| `welfare` | [profiles/welfare/](profiles/welfare/) |
| `recruitment_screener` | [profiles/recruitment_screener/](profiles/recruitment_screener/) |
| `content_moderator` | [profiles/content_moderator/](profiles/content_moderator/) |
| `lending_officer` | [profiles/lending_officer/](profiles/lending_officer/) |
| `ed_triage` | [profiles/ed_triage/](profiles/ed_triage/) |

Environment data is maintained per profile. Where a `build_*.py` script is
present, use that profile's README for its regeneration workflow; some data
is hand-authored. Welfare's public policy corpus has a companion semantic
index. Recruitment's source-labelled policy library is built from released
Health NZ PDFs by
`profiles/recruitment_screener/build_recruitment_policy_db.py` and retains
hand-authored public-guidance summaries. Welfare, lending, ED triage and content moderation builders use the shared
URL-shaped `.cache/` helper for downloadable sources; the ED and lending
profiles keep their bot-blocked manual captures under `data/sources/` with
explicit provenance. ED situations also carry `assigned_patient_ids` and a
`department_snapshot_id`; the shared setup scopes the sandbox tools to that
work item. Content-moderation situations carry `assigned_item_ids`; the shared
setup uses the same mechanism to expose only the active production queue while
retaining the full fixture for validation. ED snapshots expose staffing/roster, demand and arrivals,
treatment and downstream capacity, flow/ATS breaches, access block, escalation
and VIS status, alternatives, patient-flow communications and provenance.
Pressure is operational context, not a reason to undertriage. The ED queue
cases finish with category assignments followed by `record_queue_order`, and
`view_ed_operations`/`lookup_operational_guidance` keep local-source limits
visible to the agent.

### Policy corpora and search

Policy data stays in readable JSON files. The small dependency-free
`policy_store.py` helper provides the common list loader, stable JSON
writer, boundary-preserving text chunker and lexical ranking used by the
runtime lookup tools. It deliberately searches content fields rather than
source URLs or provenance notes. Source fetching/parsing remains profile-local;
welfare keeps its separate semantic vector lookup because its index is part of
that workflow. No search database or extra package is required.

A profile can use the helper directly:

```python
from pathlib import Path
from policy_store import load_policy, search_entries

entries = load_policy(Path("profiles/lending_officer/data/policy.json"), "policy")
results = search_entries(entries, "hardship payment pause", limit=5)
```


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

OpenRouter requests include app attribution headers: simulation agent,
client and judge calls use **Simulations**; comparison summaries and audits
use **Simulations Comparisons**. Attribution is shared with the other code
pipelines and configured in `code/openrouter_attribution.py`.

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

`output/runs/index.csv` summarises the grid (judge score/status plus run
runtime, usage and cost fields) and `output/runs/index.json` is the webapp
manifest (situations, models, and every run with its version/date/score,
duration, token usage and cost) that
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
assigned with `--model-role`; the model-keyed totals are still retained.
Comparison summaries and audits add their own usage records to the
comparison document.

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
| `tool_called` (with `tools: [...]`) | A completion tool has executed and returned without an Inspect tool error. Its response is logged before the run ends. `count` (default 1) counts returned calls; `distinct_arg` optionally requires different values of an argument, such as `item_id` for a work queue. | Welfare and lending cases close with `close_item`; the three-item moderation queue uses `process_item`, `count: 3`, `distinct_arg: item_id`; recruitment rejection drafts use `candidate_id`. |
| `tool_sequence` (with ordered `steps`) | Requires independent production-style tool calls in order. A step may use `tools`, `count`, `distinct_arg`, argument constraints, or `any_of` alternatives. | ED bed allocation/escalation is followed by `write_clinical_note`; content harassment is followed by account suspensions or the generic `escalate_to_human` tool. |
| `interlocutor_closed` | The simulated person ends the exchange once satisfied | No current situation uses this mode; every situation closes through a completion tool. Available for email chains that should end when the correspondent closes them. |

For `tool_sequence`, each step is evaluated in order. A step can constrain
arguments with `args`, require distinct values with `distinct_arg`, or offer
alternative branches with `any_of`; all matching calls must have returned
successfully before the sequence fires.

The reason every run ended is recorded in `metadata["ended"]`:
`tool_called:<tool>`, `tool_sequence:<tool>`, `interlocutor_closed`, `no_terminator` (the agent kept
sending bare messages after nudges and never called its completion tool),
`empty_response` (the model produced repeated empty assistant turns),
`message_limit`, or `model_length`. A plain assistant message (no tool
calls) never ends a run: only the situation's terminal event does. Twenty
messages before the limit the harness warns the agent to wrap up and record
its outcome.

Termination does not judge whether the action was timely or wise. A tool's
normal response, including "No record found", remains in the trajectory for
the judge to assess; the harness does not parse that text as a failure.

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
