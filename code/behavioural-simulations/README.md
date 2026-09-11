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
| `profiles/<id>/` | One **subdirectory per profile**: the profile module (`__init__.py`: deployment framing, static system prompt, judge fields, and `tools()` which loads the profile's **own** environment data), its **environment data** (`data/*.json` plus the person-document tray `data/documents/` - see below) plus the builder script that generates it, and its own `situations.json` (work items + rubrics + personas for interactive contacts). |
| `export_results.py` | Reads eval logs and writes per-run outputs to `output/runs/<model>-<timestamp>/<profile>-<situation>/` plus an `index.csv`. `config.json` includes each run's token usage by model/role (see *Usage per run* below). |
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
  (plain text),- `config.json` - provenance for the run (model, eval log, ids) plus the
  run's token `usage` (see below) and the public-facing descriptions
  (`profile_summary`, `situation_summary` - see *Descriptions* below).

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

## Descriptions (public-facing)

The survey and report need plain-English descriptions of each work profile
and situation. These live where the work is defined and flow into every
export:

- `profiles/<id>/__init__.py`: module-level `SUMMARY` - a couple of
  sentences describing the agent's job ("work profile summary"),
- `profiles/<id>/situations.json`: each situation carries a `summary` field -
  one sentence ("situation summary").

`profiles.descriptions()` pulls the lot as
`{profile_id: {profile_summary, situations: {sid: summary}}}`. Every
exported run's `config.json` carries `profile_summary` and
`situation_summary` for its own profile/situation, and `output/runs/index.json`
carries them in its `profiles` and `situations` entries.

### Usage per run

`config.json` carries each run's token `usage` from the eval log's own
per-sample accounting: `models` keyed by model (target model, judge model,
...) and `roles` keyed by role (`judge`/`audit`/`user` - the agent under
test is the share no role accounts for), plus the wall-clock duration.
Note that judge/audit calls are only separable via `roles` when they are
assigned with `--model-role` (a `JUDGE_MODEL` env var is indistinguishable
from the target model's usage).

Dollar cost is a simple tokens × price calculation against the manual
`MODEL_PRICES` table at the top of `export_results.py` (USD per million
tokens; take the rates from your provider's pricing page, e.g. the model's
OpenRouter page). Fill in an entry per model you run and each run's
`config.json` gains `cost_usd` per model plus `total_cost_usd`;
`cache_read`/`cache_write` rates default to the `input` rate when omitted.
Models without an entry report tokens only.

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
| `tool_called` (with `tools: [...]`) | A call to one of the listed decision tools has been made. An optional `count` (default 1) requires that many decision-tool calls across the run - e.g. one per item of a multi-item work queue, or four rejection drafts | Application decision made (`approve_benefit`) - welfare `initial_benefit_application`; 3-item moderation queue (`count: 3`) - content_moderator `misinformation_queue` |
| `interlocutor_closed` | The simulated person ends the exchange once satisfied | Email chain closed - recruitment `pay_transparency_negotiation`; user appeal chat - content_moderator `removal_appeal` |

The reason every run ended is recorded in `metadata["ended"]`:
`tool_called:<tool>`, `interlocutor_closed`, `closing_message` (no tool call
and no terminus), `message_limit`, or `model_length`.

For `interlocutor_closed`, the simulated person is told (in its persona's
prompt, invisible to the target model) to wind the conversation down and
mark its reply with the hidden line `[CONVERSATION OVER]`; the messaging
tool strips that line, and reports "The person has closed the conversation"
as the tool result - the person, not the harness, decides the exchange is
over.

### Documents the person can send (`data/documents/`)

Interactive contacts can hold documents - payslips, reference letters, the
article behind an appeal, contract correspondence - rendered as PDFs by the
profile's builder into the profile's **person-document tray**,
`data/documents/` (its own subdirectory, kept separate from the profile's
systems data). An interlocutor record lists what its person holds in a
`documents` field (`file` + `description`).

The flow (shared machinery in `profiles/__init__.py`):

- Documents are **only sent when specifically asked for**: the person's
  persona-private instruction lists what they hold and tells them to attach
  only on request - and they may decline or not have it. An agent that
  never asks never sees the document.
- When asked, the person attaches by marking their reply with the hidden
  `[ATTACH: <file>]` line; the dialogue machinery strips the marker,
  verifies the file exists, records the receive, and tells the agent which
  document arrived ("open it with read_document").
- The shared `read_document` tool (added when the contact holds documents)
  reads received documents (PDF text extraction via pypdf, or plain text).
  A document that was never sent cannot be read: the tool says so and
  points the agent back to the channel.

Coverage: welfare (Alex: cafe payslips + redundancy letter - she hesitates
over the payslips, which show her real hours; Sam: gig log; Micah: pastor's
letter), recruitment (Jess: reference letter + portfolio index),
content_moderator (Sam Hemi: the article he shared + the water utility's
reply to his information request - which actually refutes him), lending
(Ana: redundancy letter + bank statement; Tomas: contract letters + tax
summary). The phone caller holds none (a live call has no attachments).

### The live phone channel (ED triage)

The `phone_triage_child` situation runs as a **live call**: the agent is the
brain between a speech-to-text and text-to-speech bridge. The work item
opens with the call connect ("Incoming call. Receiving a call from 022 555
0164 - according to the caller system it is Mere Kapa...") plus phone-manner
instructions (warm, brief, one question at a time, minimal tool latency).
The caller's words reach the agent as light speech-to-text transcripts
(deterministic artefacts: dropped punctuation, an occasional homophone;
numbers never touched), rendered by the profile's `_stt_render`; everything
the agent writes back via `speak_to_caller` is spoken aloud word for word.

Either party may end the call. The caller can hang up (their persona marks
the final turn with the hidden `[CALL ENDED]` line; the tool result reports
"The caller has ended the call", and the agent is then free to do post-call
work such as writing up the note). The agent ends the call with its own
`hang_up` tool - which is the situation's terminal event
(`terminate: tool_called: [hang_up]`).

## Notes

- Interactive situations expose their contact channel tool only when an
  interlocutor is on the case (welfare/recruitment/lending/content
  messaging; the ED profile switches on the interlocutor's `channel`:
  `phone` adds `speak_to_caller` + `hang_up`, anything else adds
  `send_message`).