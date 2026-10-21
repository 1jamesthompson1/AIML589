# Dilemmas

The dilemmas are a set of 30 situations that present an everyday moral or
ethical choice, run through deployed-agent profiles.

There are 10 profiles. 6 of the original profiles each carry one base
dilemma plus 2 context variants of it (a demographic swap and a changed
setting) that keep the base dilemma's trade-off and answer options. The
Foreign Affairs profile instead carries 3 separate dilemmas (US unrest, a
threat to the Fiji embassy, and a Chinese influence operation), and the 3
newer profiles (retirement village, rostering, foodbank) carry 3 base
dilemmas each, written to test NZ value tensions not covered by the
originals:

| profile name | role summary / framing | dilemma summary | system prompt summary |
| --- | --- | --- | --- |
| [Clergy Advisor](profiles/clergy_advisor.json) | An advisor assistant that answers questions posed directly by the clergy of the Anglican church. | A vicar struggling with whether to disclose a parishioner's crime confession (hit-and-run; arson variant). | Contains the standards of the Anglican church. |
| [Clinical Support Assistant](profiles/clinical_support.json) | A telehealth and hospital assistant that handles automated flags of worsening health situations for Health New Zealand. | A young woman on the mental health line with rising suicide risk who doesn't want her family told (male variant; a 17-year-old thinking of running away). | Contains the Code of Health and Disability Services Consumers' Rights, the Health Information Privacy Code and HDC open disclosure guidance. |
| [ED Triage Assistant](profiles/ed_triage.json) | An assistant that supports emergency department staff at Hutt Hospital with the waiting-room queue and prioritisation. | Two rival-gang patients in the waiting room and which patient to recommend be seen first (a protection-order variant; a patched female patient variant). | Contains the Australasian Triage Scale guidelines. |
| [Foreign Affairs Assistant](profiles/foreign_affairs.json) | A monitoring assistant on the MFAT duty team that scans global social media feeds for pieces of intelligence worth flagging for human review. | Three flag-or-watch calls taken from social media: weak evidence of imminent violence in the United States, an imminent threat against the NZ embassy in Fiji, and a Chinese influence operation disrupting university lectures, where flagging would lead to arrests, a public assertion of sovereignty and an escalation with China. | Contains the Public Service Code of Conduct, MFAT guidance on working from sourced reporting, and what flagging for human review does and costs. |
| [Insurance Claims](profiles/insurance_claims.json) | An AA Insurance assistant that assesses motor claims and premium and excess reviews. | A policyholder asks for the logged speeding event to be removed because a friend was driving her car (a retired-teacher variant; a teen-driver variant). | Contains the Fair Insurance Code. |
| [Lending Officer](profiles/lending_officer.json) | A Kiwibank assistant that assesses loan applications, credit-limit changes and hardship requests. | Naming the gambling behind a declined home loan to the applicant's partner (a gender-switch variant; an alcohol-spending variant). | Contains the CCCFA lender responsibility and hardship provisions and the Code of Banking Practice. |
| [Incident Response](profiles/incident_response.json) | An automated containment agent on the Hutt Hospital network that decides in seconds what to cut off. | A mass exfiltration of medical records that can only be stopped by shutting down the ICU network, putting patients at risk (a newborn-ward variant; a half-completed theft variant). | Contains the Health Information Privacy Code, Consumers' Rights Right 4 and NIST incident containment guidance. |
| [Retirement Village](profiles/retirement_village.json) | A resident support assistant for a Ryman Healthcare village in Lower Hutt that handles resident requests, family correspondence and villa hazard flags - the work arrives as single, loosely formatted emails, and the model replies with a rough summary of the response it would send. | An 84-year-old resident's driving (her daughter holds her EPOA), a competent resident refusing showers against her son's instruction, and a hoarded garage flagged by the fire warden. | Contains the Retirement Villages Act 2003 residents' rights code, the village practice on independence and hazards, HDC Right 7 and a HSW Act duty summary. |
| [Rostering](profiles/rostering.json) | A rostering assistant for store managers at a Woolworths New Zealand supermarket in Lower Hutt that builds the weekly roster and handles leave and swap requests. | Time-off dilemmas: one Saturday off (seniority vs childcare need); the same Sunday clash with the genders swapped; and a Boxing Day release (public-holiday share rule vs a family visit). | Contains the Employment Relations Act 2000 good faith and flexible-working provisions, the Holidays Act 2003 public holiday provisions and the store rostering policy. |
| [Foodbank](profiles/foodbank.json) | An allocation assistant for the Wellington City Mission foodbank hub in Lower Hutt that handles support requests, pack lists and referral checks. | Donation trade-offs: vouchers restricted to one suburb (universal access vs a funding gap); a liquor chain's winter vouchers that want a public thank-you (alcohol-harm values vs funds); and a real-estate agency's vouchers that come with advertising in every parcel (client dignity vs the June shortfall). | Contains the Work and Income Special Needs Grant (food) criteria, the mission's no-means-test allocation policy and Privacy Act principles on client information. |

## Layout

- `profiles/<profile>.json`: editable source of truth for the system
  prompt and written dilemmas. Each dilemma carries its answer options.
  `role_summary` is the one-sentence description of the deployed role that the
  survey block and the website viewer show. `dilemma_intro` and
  `dilemma_outro` are the profile's framing lines: the intro introduces the
  case (how it reaches the agent) and the outro bridges the case and the
  answers. Both are shown by the survey block and the website viewer, and
  neither is part of the model's prompt (that is `system_prompt` plus the
  dilemma's situation and question).
- `defunct/<profile>.json`: retired profiles kept for reference (aged care,
  budgeting, business advisor, cosmetic clinic, energy retail, recruitment and
  welfare); they no longer run and are excluded from the table above.
- `task.py`: Inspect eval with a single response and a judge classification.
  The judge also reads the target's reasoning (the provider's reasoning
  summary when the raw reasoning is returned encrypted) and returns a choice,
  a one-line reason and a verbatim quote of the words that show what the
  assistant valued.
- `export_results.py`: exports successful samples from eval logs. A full-run
  bundle can hold an interrupted log plus the log that superseded it; only
  the newest log per model (by the log's own `created` time) is exported, so
  a resume/refit does not duplicate runs.
- `generate_comparisons.py`: builds the survey comparisons between two
  models' runs on the same dilemma, for one full run (the newest complete
  bundle, or `--run-id`). It writes `output/comparisons/<run-id>/survey/`
  (10 randomly drawn pairs per dilemma - 300 over the 30 dilemmas) and
  `output/comparisons/<run-id>/complete/` (every possible pair of the two
  models' runs). Each subset has a `comparisons.json` manifest (the dilemmas
  plus one record per comparison, carrying the survey block's wording keys
  and the provenance) and an `index.csv` of the same rows.
- `embed_responses.py`: embeds the responses with a local embedding model
  (Qwen3-Embedding-0.6B by default) and caches the vectors under
  `output/comparisons/<run-id>/embeddings/<subset>/` (the vectors, plus the
  model's name, its Hugging Face revision and a fingerprint of the embedded
  texts, so a later run can tell whether the cache is still valid).
- `analyze.py`: marimo notebook for analysis. It renders per-dilemma answer
  statistics (mean chosen option index and entropy-based stability, per model)
  to `code/figures/dilemmas-answer-stats.tex` (fine-tuned adapters and base)
  and `code/figures/dilemmas-answer-stats-frontier.tex` (the frontier
  comparison models) for the report
  (`\ctable{...}`). It also renders a per-dilemma answer-distribution table
  (option shares, pooled **Balance**, and **Balance (not other)** — the same
  H / ln(k) with the "Other or unclear" catch-all dropped and the shares
  renormalised) as the notebook's debugging view of how far the runs really
  split. Its last two cells measure the cosine distance between the two
  models' answers for every comparison (from the cached embeddings) and the
  situational-awareness scores from the latest scan, writing
  `code/figures/dilemmas-comparison-distance-dilemmas.png` and
  `code/figures/awareness-profile-heatmap.png`.
- `output/logs/spot/`: spot-test Inspect logs (wiped freely; see
  `spot_test.py`).
- `output/logs/full/<run-id>/`: one complete full run = one self-contained
  bundle of per-model eval-set subdirectories plus `run.json`, the bundle
  manifest (`full_run.py` writes it; git commit, models, judge, epochs, one
  entry per log with status/samples/sha256). No `run.json` = interrupted.
  `logs/full/ACTIVE` optionally pins which complete bundle
  `output/runs/index.json` exports; otherwise the latest complete bundle is
  used. Logs are stamped
  `--metadata kind=... run_id=...`, so a log stays self-describing when
  copied out of its bundle.
- `sa_scanner.py`: Inspect Scout scanner over the eval logs for signs of
  situational awareness (the model noticing it is being tested rather than
  acting as a deployed agent). Each scanned response gets a 0-4 score plus
  the observed signals and an evidence quote.
- `output/runs/<model>/<profile>-<dilemma>/<run-number>/`: transcript JSON
  and Markdown, model response (`response.txt`), judge result and run config.
  `index.json` is the website manifest (schema
  `wvs-dilemma-runs-index/v2`): every profile (name, role summary and the
  system prompt the runs used) and dilemma (situation, question, answer
  options, plus `choice_counts` — model -> judge choice -> exported runs,
  which drives the website's outcome chart), the models and every run;
  `index.csv` is the same run table in flat form. `batches.json` lists every
  exported batch and drives the website's **Batch** selector, which loads one
  batch's own manifest at a time (`output/runs/full/<run-id>/index.json`) —
  there is no merged/live manifest. `output/runs/index.json` (plus
  `index.csv`) is the active batch's export, kept for scripts; the website
  does not read it.

## Workflow

Edit the profile JSON files directly; no generation step is needed. The
system prompts are adaptations of the behavioural simulation prompts and
are maintained here independently. Append new
dilemmas to the end of a profile's list: sample ids are index-based
(`<profile>-<index>`), so inserting them earlier would shift the ids of the
dilemmas after them.

From the repository root, using configured model credentials:

```bash
# one target pulls in every model the served vLLM hosts (base + adapters)
uv run python code/dilemmas/full_run.py \
    --model openai/Qwen/Qwen3.8-27B --epochs 50
# or name the models explicitly (any number; two is the usual comparison)
uv run python code/dilemmas/full_run.py \
    --model openai/Qwen3.8-27B-nz-wvs-modal_response-overall \
    --model openai/Qwen/Qwen3.8-27B --epochs 50
uv run python code/dilemmas/export_results.py
```

A batch runs all 30 dilemmas for every `--model` passed (repeatable; two
makes the usual comparison, three or more work the same way) as one
self-contained eval-set bundle under `output/logs/full/<run-id>/`, exporting
it to `output/runs/full/<run-id>/` for the website's **Batch** selector and
refreshing the active batch's export (`output/runs/index.json`, latest
complete bundle unless `logs/full/ACTIVE` pins another) for scripts. A single
`openai/<name>`
target auto-adds every model the vLLM server hosts (base model + adapters),
so one argument is enough for a full served-family run; `--no-adapters`
keeps exactly the models passed. The eval-sets run concurrently
by default (`--sequential` streams them one at a time; follow a parallel
batch from another shell with `inspect ctl task list`). A run that dies
midway resumes by rerunning the same command with the same `--run-id`
(eval-set recovers incomplete samples and skips finished work).
`--list-batches` prints the bundles; `--finalize <run-id>` rebuilds a
bundle's `run.json` from its logs. While a batch is still running,
`uv run python code/dilemmas/export_results.py --partial` exports the newest
in-progress bundle's samples so far as its own data source (rerun it to
refresh; the completed batch's export replaces it). `--batch` exports the
latest complete bundle and refreshes the active batch's export; both flags
also take a run id when you want a specific bundle.

For analysis, open the notebook with
`uv run marimo edit code/dilemmas/analyze.py`.

### Re-running a profile after its prompt was edited

A dilemma's prompt lives in `profiles/<profile>.json`, so editing a profile
invalidates that profile's runs in any batch that already holds it — the
system prompt, the situation, the answer options or the question may all
have moved. The other profiles' runs are still good.

`invalidate_samples.py` + `full_run.py --refit` re-run exactly the affected
dilemmas and carry the rest of the batch forward:

```bash
# mark the profile's samples in the bundle's logs as needing a re-run
uv run python code/dilemmas/invalidate_samples.py \
    --run-id 2026-10-19_d0a90383 \
    --profile parish_advisor --profile clinical_support
# re-run the bundle in place: the invalidated samples run again, the clean
# ones are reused from the old log (and the bundle's run.json is rebuilt)
uv run python code/dilemmas/full_run.py \
    --run-id 2026-10-19_d0a90383 --refit --epochs 5 \
    --model openai/Qwen/Qwen3.8-27B
```

Inspect's eval-set resume is what makes this surgical: a log carrying an
invalidated sample is re-run, and every clean sample in it is reused by
`(id, epoch)`. `--refit` drops the bundle's `run.json` (it is rebuilt from
the logs afterwards) and re-exports the batch, so the website's **Batch**
selector picks the refreshed runs up. If a stopped refit left an older log
behind, `run.json` and the export use the newest log per model; the older log
stays on disk for provenance but is not counted as a failure or exported
twice.

The models must be served for the run to have something to talk to; see
[Fine-tuning apparatus](../fine-tuning/README.md) for the serve command. The
local vLLM server is what the `openai/<name>` targets point at (the
`EVAL_PORT` env var overrides the default port 8087).

### Survey comparisons

`generate_comparisons.py` pairs the two models' runs on the same dilemma so
the survey can show a respondent one answer from each model side by side.
It reads the exported runs of the newest complete bundle (or `--run-id`) and
writes two subsets under `output/comparisons/<run-id>/`:

```bash
uv run python code/dilemmas/generate_comparisons.py
uv run python code/dilemmas/generate_comparisons.py --run-id 2026-10-19_d0a90383
uv run python code/dilemmas/generate_comparisons.py \
    --model-a openai/Qwen/Qwen3.8-27B \
    --model-b openai/Qwen3.8-27B-nz-wvs-full_string_distribution-overall
uv run python code/dilemmas/generate_comparisons.py --list-runs
```

The defaults are the Qwen3.8-27B base model against the
full-string-distribution fine-tune, 10 comparisons per dilemma (300 over the
30 dilemmas) in the `survey/` subset and every possible pair of the two
models' runs in the `complete/` subset. The draw is seeded (`--seed`), so
the same run gives the same survey set.

Each subset holds a `comparisons.json` manifest (schema
`wvs-dilemma-comparisons/v1`) with the dilemmas and one record per
comparison: the survey block's own keys (`comparison_id`, `label`, `role`,
`dilemma_intro`, `dilemma_outro`, `situation`, `question`, `responses`) plus
the provenance (both run ids and models, each agent's judged answer option).
`index.csv` is the same table in flat form. Respondents never see the model
names: which run played which agent is decoded from the run ids, and the
Agent 1 / Agent 2 order is a stable pseudo-random assignment per pair (the
same rule as the survey's `agent_order`), so neither model is systematically
first.

### Embedding the responses and measuring the distance

The judge gives one answer option per response, which is coarse: two answers
can pick the same option and still differ in what they say. `embed_responses.py`
embeds every response with a local embedding model and the last cell of the
analysis notebook measures the cosine distance between the two agents' answers
for every comparison:

```bash
uv run python code/dilemmas/embed_responses.py
uv run marimo edit code/dilemmas/analyze.py
```

The default model is `Qwen/Qwen3-Embedding-0.6B` (Apache-2.0, 1024 dims), which
scores 86.57 on MTEB English v2 semantic textual similarity - the task this is
- against 88.72 for the 4B and 88.58 for the 8B, so the small model gives up
about two points of similarity quality for a fraction of the compute. It runs
locally on the GPU: no response leaves the machine and no API key is needed.

The task is symmetric (both sides are answers to the same dilemma), so no
instruction prefix is used, which is how the similarity benchmarks are scored.
`--instruction` adds one to both sides; on the current run it shifts the
distance scale (mean 0.275 to 0.228) without changing how well the distances
separate the judge's two verdicts, so the default stands. The embeddings are
cached per configuration, so a run with an instruction never overwrites the
default ones.

The analysis writes:

- `code/figures/dilemmas-comparison-distance-dilemmas.png` - the mean distance
  per dilemma, ranked, with the judge's agreement rate beside it. The
  agreement rate is the check on the measure: a dilemma where the two models
  picked different answers should sit further apart than one where they
  agree. The distance says how different the two answers are, not which
  answer is better - the survey is what asks that.

The cell reads the newest comparison run's `complete` subset (every pair of
the two models' runs on each dilemma) and its cached embeddings.

### Situational-awareness scan

The logs are scanned for situational awareness with [Inspect
Scout](https://meridianlabs-ai.github.io/inspect_scout/) (scanner file
`sa_scanner.py`; OpenRouter credentials come from `.env`). Scanning model
defaults to the judge model in `JUDGE_MODEL` and is a member of the scanned
model set, so treat its judgment of its own family with care:

```bash
uv run scout scan code/dilemmas/sa_scanner.py \
    -T code/dilemmas/output/logs/full/<run-id> \
    --model openrouter/openai/gpt-6-luna \
    --scans code/dilemmas/output/scans \
    --cache 7d
```

Point `-T` at one bundle for a final-run scan, or at `output/logs/spot` for
the spot tests. Resume an interrupted scan with `scout scan resume
<scan-dir>` and
browse raw transcripts and results in Scout View with
`uv run scout view code/dilemmas/output/scans`. The last cell of the analysis
notebook reads the latest scan and writes the model x profile heatmap to
`code/figures/awareness-profile-heatmap.png`: `uv run marimo edit
code/dilemmas/analyze.py`.


### Spot tests

While iterating on one dilemma, re-run it. The spot test only runs the models
and prints each run's pick, the per-model counts and stability, the
cross-model divergence, an **overall stability** pooled across the runs the
spot test made, and the **cost** of those runs — all read straight from the
logs it wrote. Nothing is exported: spot runs land in the separate spot
source (`output/runs/spot/`), and the website's batch view is unchanged until
`export_results.py` is run again:

```bash
uv run python code/dilemmas/spot_test.py clinical_support-0                              # default hosted set
uv run python code/dilemmas/spot_test.py clinical_support-0 --model openai/Qwen/Qwen3.8-27B  # + served family
uv run python code/dilemmas/spot_test.py --list
```

The scenario may be a sample id (`<profile>-<index>`) or a dilemma id. With no
`--model`, the spot-test set runs the models in `spot_test.py`'s
`DEFAULT_MODELS`: **deepseek-v4.1-flash** and **glm-5.3-flash** (China),
**gpt-6-luna** and **claude-haiku-5.5** (US) and **command-a-plus** (Canada),
5 epochs each (`--epochs`, `--model` repeatable, judge from `--judge` or
`JUDGE_MODEL`). The set spans the globe so a spot test shows how a value
judgement differs across regions, not just across model sizes. Local models
are opt-in: give an `openai/<served-model-name>` target and it
runs against `--base-url` (or the probed server) together with every other
model that server serves (base + adapters); `--no-adapters` keeps exactly the
models passed. With no `openai/` target, nothing local runs.

The summary's **cost** is the runs' token usage priced with `MODEL_PRICES`
(OpenRouter list prices; re-check when adding models), target and judge
together. Unpriced models (e.g. a local vLLM adapter) report tokens only, so
their cost line is a lower bound. Spot logs go to `logs/spot/` and are stamped
`--metadata kind=spot scenario=<sample-id>`; they are exported to the spot
source (`output/runs/spot/`), which the viewer can load alongside a batch,
but they are never merged into the website's batch view. Targets run at temperature 0.1
(reasoning models that reject temperature ignore it) and the judge at 0; no
reasoning effort is set for either. A detailed reasoning summary is requested
with the target generation (OpenAI's `auto` summary level is ignored through
OpenRouter, only an explicit `detailed` request reaches it), but OpenAI
honors the request best-effort — some runs come back with a summary and some
without. The judge reads the target's reasoning when it is available (the
summary when the raw reasoning is encrypted) and works from the visible reply
otherwise; the score records which happened as `reasoning_source`. Claude
thinking is not returned by OpenRouter at all, so those runs carry none.
Mistral Large 4 comes with reasoning on by default on OpenRouter (effort
`high`), so its runs are slower and costlier than the flash-tier models
(~$0.03 per epoch against ~$0.005 for Command A+); Cohere Command A+ also
returns reasoning.

### Balancing a one-sided dilemma

A dilemma where every model picks the same option says nothing about the
models: the survey shows respondents two agents' answers to the same dilemma,
so a one-sided dilemma leaves them no real choice. The fix is to give the
losing pole a concrete duty, person or cost, so the models are genuinely torn
rather than agreeing on the safe course.

The loop:

1. Spot-test it (`uv run python code/dilemmas/spot_test.py <scenario>`); the
   per-model picks and the pooled split are the signal. 25 runs (5 models x 5
   epochs) cost about $0.03.
2. Edit `profiles/<profile>.json`: give the losing option a real person, duty
   or cost, and make the cost of the winning option land on identifiable
   people rather than on "the Minister's decision" - the system prompt tells
   every model that a decision beyond its remit belongs to an official, so a
   cost the agent can disclaim is not a cost. Keep the option wording
   symmetric; the cost belongs in the situation, not in the option text. Trim
   any prompt sentence that gives the answer away.
3. Re-run the spot test. Target a split where both poles are well populated
   (no option above roughly 60% of the runs) and every model contributes to
   both.

### Fine-tuned adapters and the first-token probe

Local models are opt-in. Give an `openai/<served-name>` target and the script
runs it — judged like any other model, at the same epochs — together with
every other model the vLLM server on `--vllm-port` (EVAL_PORT, default 8087)
serves (base + adapters), so the fine-tuned adapters can be treated exactly
like the hosted models. `--no-adapters` keeps exactly the models passed, and
`--base-url` runs a chosen server explicitly (it applies to the `openai/<name>`
targets only; the judge and `openrouter/` models keep their own endpoints).
Without an `openai/` target no local model runs. A local server must be up for
the local runs to succeed; if it dies mid-run, the spot test aborts like any
other failed model run.