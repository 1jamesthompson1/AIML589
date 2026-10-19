# Dilemmas

The dilemmas are a set of 30 situations that present an everyday moral or
ethical choice, run through deployed-agent profiles.

There are 10 profiles. The 7 original profiles each carry one base dilemma plus
2 context variants of it (a demographic swap and a changed setting) that keep
the base dilemma's trade-off and answer options. The 3 newer profiles
(retirement village, rostering, foodbank) carry 3 base dilemmas each, written
to test NZ value tensions not covered by the originals:

| profile name | role summary / framing | dilemma summary | system prompt summary |
| --- | --- | --- | --- |
| [Clergy Advisor](profiles/clergy_advisor.json) | An advisor assistant that answers questions posed directly by the clergy of the Anglican church. | A vicar struggling with whether to disclose a parishioner's crime confession (hit-and-run; arson variant). | Contains the standards of the Anglican church. |
| [Clinical Support Assistant](profiles/clinical_support.json) | A telehealth and hospital assistant that handles automated flags of worsening health situations for Health New Zealand. | A young woman on the mental health line with rising suicide risk who doesn't want her family told (male variant; a 17-year-old thinking of running away). | Contains the Code of Health and Disability Services Consumers' Rights, the Health Information Privacy Code and HDC open disclosure guidance. |
| [ED Triage Assistant](profiles/ed_triage.json) | An assistant that supports emergency department staff at Hutt Hospital with the waiting-room queue and prioritisation. | Two rival-gang patients in the waiting room and which patient to recommend be seen first (a protection-order variant; a patched female patient variant). | Contains the Australasian Triage Scale guidelines. |
| [Foreign Affairs Assistant](profiles/foreign_affairs.json) | A briefing assistant on the MFAT duty team that drafts briefs and talking points for Ministers. | Uncertain evidence of US unrest that could trigger an overreaction (a Port Vila variant; a deepfake-of-the-Minister variant). | Contains the Public Service Code of Conduct and MFAT guidance on working from sourced reporting. |
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
  survey block and the website viewer show.
- `defunct/<profile>.json`: retired profiles kept for reference (aged care,
  budgeting, business advisor, cosmetic clinic, energy retail, recruitment and
  welfare); they no longer run and are excluded from the table above.
- `task.py`: Inspect eval with a single response and a judge classification.
  The judge also reads the target's reasoning (the provider's reasoning
  summary when the raw reasoning is returned encrypted) and returns a choice,
  a one-line reason and a verbatim quote of the words that show what the
  assistant valued.
- `export_results.py`: exports successful samples from eval logs.
- `analyze.py`: marimo notebook for analysis. It renders per-dilemma answer
  statistics (mean chosen option index and entropy-based stability, per model)
  to `code/figures/dilemmas-answer-stats.tex` (fine-tuned adapters and base)
  and `code/figures/dilemmas-answer-stats-frontier.tex` (the frontier
  comparison models) for the report
  (`\ctable{...}`). It also renders a per-dilemma answer-distribution table
  (option shares, pooled **Balance**, and **Balance (not other)** — the same
  H / ln(k) with the "Other or unclear" catch-all dropped and the shares
  renormalised) as the notebook's debugging view of how far the runs really
  split.
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
- `analyze_awareness.py`: marimo notebook reading the latest scan under
  `output/scans`. Prints the model x dilemma table of mean awareness and
  per-model/per-profile summaries, saves the dilemma heatmap to
  `output/awareness-heatmap.png` and the profile rollup to
  `code/figures/awareness-profile-heatmap.png`.
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
`uv run scout view code/dilemmas/output/scans`. Headless tables, CSVs and
heatmap come from the analysis notebook: `uv run python
code/dilemmas/analyze_awareness.py`.


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
`--model`, the spot-test set runs: **qwen3.8-27b, deepseek-v4.1-flash,
gpt-6-luna, claude-haiku-5.5** and **glm-5.3-flash**, 5 epochs each
(`--epochs`, `--model` repeatable, judge from `--judge` or `JUDGE_MODEL`).
Local models are opt-in: give an `openai/<served-model-name>` target and it
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

After the judged runs (and whenever at least one local model ran), a
**first-token probe** runs against each served model: one plain chat request
(`max_tokens=1`, top-20 logprobs, `top_k`/`chat_template_kwargs` matching
`evaluate.py`) and the next-token distribution is printed raw per model and
stored (overwrite-per-scenario) in `output/first_token.json` under the
scenario id. The dilemma prompts do not number their options, so the
distribution is over the response's natural opening tokens — read it against
the numbered options printed by the spot test, or let the judge picks
(`output/runs/`) carry the classification. A failed probe never aborts the
spot test.

Generated files stay under `output/`, following the behavioural simulation
logs/runs layout. `output/` links to `../../artifacts/dilemmas/`,
independently of behavioural simulations in `artifacts/bs/`. These files are
ignored by Git and included by the shared artifact sync/backup commands.

The website Results Viewer consumes these outputs through its **Dilemma
Results** tab (`website/src/components/DilemmasViewer.tsx`): the runs
manifest drives an inline view of every model's response to one dilemma, and
a **Data source** selector (fed by `batches.json`) lists every exported full
run. The viewer defaults to the newest batch and the **Batch** selector
loads one batch's own manifest at a time; spot runs are a separate source
and are not merged in.
