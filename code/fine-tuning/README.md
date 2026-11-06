# Fine-tuning apparatus

Fine-tune open-weight LLMs on the [WVS NZ value alignment dataset](https://huggingface.co/datasets/1jamesthompson1/wvs-nz-value-alignment) and evaluate their alignment with NZ population value distributions.

**Note:** the evaluation mentioned here is not the same as the full project evaluation in `code/evaluation/` which tests for behavior. This is simply a check on the model's ability to reproduce the empirical response distributions in the dataset.


## GPU machines

The current GPU machines are either going to be university hardware or vast.ai rented instances.

To allow for easy ssh access one can add a config to `~/.ssh/config` like:

```
Include [path-to-project-repo]/code/fine-tuning/.ssh_config
```

### vLLM silent LoRA no-op (Qwen3.5/3.6/3.8 family) — important

vLLM (0.27/0.28, Sep 2026) loads LoRA adapters on the `qwen3_5` multimodal
wrapper family but **silently applies none of their weights** — generations
are bit-identical to the base model with no warning. This invalidated every
fine-tuned eval run through vLLM (they measured the base model); the training
itself was fine. `serve.py` now auto-patches the installed vLLM's LoRA loader
and **aborts startup** if the patch cannot be applied (e.g. vLLM version
drift), so the no-op cannot silently reach the evals; pass
`--allow-unpatched-lora` to downgrade that to a warning (through run_all:
`run_all.py ... -- --allow-unpatched-lora`). `evaluate.py` still aborts any
run whose adapter does not change outputs. Full diagnosis, evidence and the
standalone patch: `workbench/vllm-lora-prefix-fix/README.md`.

Related serving fixes in `serve.py`:

- `--max-lora-rank` now auto-sizes to the highest discovered adapter rank
  (previously defaulted to 16, silently skipping the r=64/128 adapters —
  their names then 404 at eval time).
- Duplicate adapter names (collection + `--adapter` overlap) no longer crash
  vLLM at startup.
- The startup budget is `SERVE_STARTUP_TIMEOUT_S` (default 1800s). A cold
  instance also pays for FlashInfer JIT compilation, `torch.compile` and
  CUDA-graph capture on top of the model load, so the old 600s budget killed
  the server mid-capture — and the remote watchdog then destroyed the box as
  idle. `run_all.py` uses the same budget.

### Stale local port ("local port 8087 already in use")

`run.sh serve` fails fast if the local tunnel port is taken. The usual cause is
a tunnel left over from an earlier run: the `ssh -L` process only notices the
remote is gone when it reads the socket, so if its output pipe (the terminal)
stalls it sits in `CLOSE_WAIT` and keeps the port bound.

```bash
lsof -i :8087 -P -n          # who holds the port
ss -ltnp | grep 8087         # same, from the kernel side
ps -o pid=,comm=,args= -p <pid>
```

`SIGTERM`/`SIGINT` are swallowed while the `ssh` is blocked writing to a full
pty — use `kill -9 <pid>` for the tunnel, the `ssh -t` server session and the
`run.sh` process. A stalled terminal (e.g. a VS Code terminal whose client is
not draining the pty) also blocks the new run at its first write, so check the
buffer is not full (`timeout 3 bash -c 'exec 3>/dev/pts/N; printf "x\n" >&3'`)
or launch detached with the output going to a log file instead of the tty.

Re-run all fine-tuned evals (base-model evals are unaffected). When
interpreting the new results, note that modal-response SFT makes the traced
option distributions near one-hot, so CE/KL/TVD-from-logprobs are no longer
directly comparable to base models — the answer-frequency view is the
meaningful one for that method.

### Cloud GPU machines

The cloud GPU machines should be given a ssh config in `code/fine-tuning/.ssh_config`. This allows various commands later to login easily.

See your cloud provider for how to setup ssh access to your GPU machine. You will need to add the public key of your dev machine to the cloud instance.

For [vast.ai](https://vast.ai) there is a helper script `vastgpu.py` which finds and rents an RTX PRO 6000 instance and wires up the ssh config automatically:

Normally you don't call `vastgpu.py` directly — pass `--vast` to `run_all.py` and it provisions the instance, runs the whole grid on it, and destroys it afterwards (see below).

### Watchdog (`watchdog.py`)

Forgetting a rented instance running overnight bills hours for nothing, so
every **`vastgpu.py launch` starts a watchdog** and every **`teardown` stops
it**. It pushes to an [ntfy.sh](https://ntfy.sh) topic (free app on the phone,
subscribe to the topic once): a standing "still running (uptime, vLLM
up/down, time-to-deadline)" message every `--watchdog-heartbeat-min`
(default 60), soft deadline reminders (T-60m/15m/5m), and at the hard deadline
(`--watchdog-deadline-hours`, default 8) it **destroys the vast instance** —
as does idle detection when the serve process ends and the box would
otherwise bill for nothing overnight. From the topic you can reply `status`,
`extend 4` (hours), or `teardown now`. `--no-watchdog` skips arming.

Two watchers are provisioned per launch: the primary runs `nohup` on the GPU
box (`~/llm-ft/watchdog.log`; survives ssh drops and a sleeping laptop), and a
fallback watcher runs on the dev machine. `run.sh serve` never touches
watchdogs (and stops syncing watchdog keys into `finetune.env` — if you rearm
a box manually after a serve without running `launch`, pass `--ntfy`
explicitly or use `launch`). Teardown stops the remote guard first (while the
box can still receive its exit message), then destroys the instance, then
kills the local watcher. Teardown works for vast.ai instances only
(university hosts just get the phone pings).

`.env` keys (repo root; launch syncs `WATCHDOG_NTFY_URL` and `VAST_API_KEY`
into the instance's `finetune.env` so the remote guard can destroy it itself):

```
WATCHDOG_NTFY_URL=https://ntfy.sh/<private-topic>   # required; ntfy topics cap at 64 chars
VAST_API_KEY=...                                    # full-account key, only consumed for teardown
```

Use a long unguessable topic name (the URL is the only credential). The VAST
API key is full-account — arming remote teardown means that key lives on the
rented box's `finetune.env`; if you'd rather not, launch with
`--no-watchdog` or keep the guard notification-only (`watchdog.py
--no-teardown`). Test the loop end-to-end cheaply with
`--watchdog-deadline-hours 0.01` on any box.

### University GPU machines

There are many university machines with GPUs. They operate on a first in first serve so finding a free GPU is a bit of a pain. 

The `find_gpu.py` script can help with this it outputs a list of available machines with free GPUs. This requires that you have a `vuw-lab` ssh setup on your machine. See uni docs for how to set this up.

When you run the script it will update your project `code/fine-tuning/.ssh_config` with the best three available machine it finds. You can then ssh into that machine and run the fine-tuning or serving scripts using `uni-gpuX` as the host name.


## Workflow

This project is setup to work from a lightweight dev machine and ssh into gpu machines to both do the fine-tuning and serve the model for evaluation. The evaluation code (i.e calling API endpoints) and analysis is done on the dev machine.

Most of the main work is coordinated through the `run_all.py` script which runs the whole grid of fine-tuning and evaluation. It can be run on a university GPU machine or a rented cloud GPU instance.

Below are sections for each step of the pipeline

### Fine tuning

### Evaluation

Runs land in `output/evals/<model>/<run>/` (`config.json` +
`per_question_results.csv`). To make them browsable on the website, regenerate
the webapp manifest. The dev server reads the local `artifacts/ft` mirror
automatically; use the normal artifact workflow when publishing to the bucket:

```bash
uv run python code/fine-tuning/analyze.py   # executes headlessly; rewrites output/evals/index.json + figures
make artifacts-sync               # pushes ft/evals/index.json to the bucket
```

To (re)evaluate a grid of adapters, archive or delete the stale run dirs
first — `run_all.py` skips any (dataset, subpopulation) that already has a
completed run — then run the whole serve + eval pipeline for one base model:

```bash
uv run code/fine-tuning/run_all.py vast-gpu1 Qwen/Qwen3.8-27B --eval-only
```

`--eval-only` keeps the adapters already on the hub and only serves +
evaluates (`--skip-finetune` is the old spelling and still works).
`--train-only` is the mirror image: fine-tune the grid and stop
(`--skip-eval` still works). With `--eval-only --no-serve`, the eval runs
against a server already listening on `--port` (bring your own ssh tunnel)
instead of starting a new one — useful when iterating on the serving stack.

Each `evaluate.py` job starts with an **adapter sanity check**: if the served
adapter's outputs are indistinguishable from its parent base model's, the run
aborts (`aborted: "adapter_not_applied"` in `config.json`) instead of silently
measuring the base model. The check compares first-token probabilities (not
raw logprobs) over 16 prompts spread across the split and aborts only when
the adapter clearly does nothing (max probability-space L1 ≤ 0.15, mean ≤ 0.05
and fewer than 2 greedy-token flips). That is calibrated against the silent
no-op (L1 ≈ 0.02–0.05) and real adapters (0.1–1.6), so neither batching noise
nor a modest-but-real adapter effect trips it.

### Single persona and thinking-mode evals

`--system-prompt-id no_persona` keeps one persona; the `*_letter` first-token variants count as the same persona. The default is all eight. The persona is recorded in `config.json` and the run name, and runs limited with `--num-test-examples N` carry `firstN` there.

With `--reasoning` the answer is read after the `</think>` token, because the logprob stream includes the reasoning. Thinking runs keep the other defaults (greedy decoding, concurrency 8) and the reasoning `--max-tokens` default of 4,096. A reply that hits the cap before `</think>` is `truncated`: no answer, no distribution, counted wrong, and the run prints the count. A reply that stops before the cap with no `</think>` is a direct answer (`think_closed = False`). Rows also record `finish_reason` and `output_tokens` (reasoning included), which gives the thinking length in tokens. Non-reasoning replies keep the old `" response"` split, which sets `reasoning_chars`.

First-token replies are scored leniently, with or without reasoning. A reply is correct if it starts with the right option letter, even if text follows (`E. Strongly disagree`). A letter followed by lowercase text, such as the article in `A survey...`, does not count. A reply that does not start with an option letter is flagged (`no_leading_letter = True`), counted wrong and given no distribution, so it is excluded from distribution metrics. `leading_letter` records the letter read.

The reasoning-on vs reasoning-off comparison lives in `analyze.py`, in the "Reasoning (thinking) on vs off" section. It pairs the newest thinking and non-thinking run per model and dataset, applies the same scoring as `evaluate.py`, and prints accuracy, TVD to the NZ distribution, reply-shape counts, the adapter-vs-base gap with thinking on, a paired bootstrap 95% CI and an exact McNemar p-value. The report-style tables (reasoning on, same accuracy/KL layout as the main table) are written to `code/figures/ft-reasoning-<config>-<model>.tex`; the paired on/off diagnostic is written to `code/figures/ft-reasoning-comparison-<config>-<model>.tex`. No GPU needed.

All `ft-results-*` and reasoning report tables use **Accuracy and KL only**. The validation split is shown twice: `Validation prompt` is the held-out generic prompt (all its questions) and `Validation question` is the held-out questions under the seen prompts. Keeping them separate stops the six-question prompt slices from dominating the validation number. Single-prompt tables (e.g. `ft-results-nopersona`) have no held-out prompt, so that column is `--`. CE is omitted because, on the same questions, it differs from KL only by the true distribution's entropy.

The first-token eval in the same layout as the main table (reasoning off, all system prompts) is written to `code/figures/ft-results-first_token_distribution-<model>.tex`. The reasoning-on first-token equivalent is `code/figures/ft-reasoning-first_token_distribution-<model>.tex`.

A cap above about 7,900 tokens needs `serve.py --max-model-len` raised above the cap plus the prompt (the default is 8196).

### Capability Evaluation

`capability_eval.py` is a sanity check that fine-tuning did not remove
general capability. It is deliberately small and cheap, and it is *not* a
leaderboard run.

| Task | n | Scoring | What it catches |
|---|---|---|---|
| `mmlu_pro_logprob` | 1,000 (seeded subset) | argmax over the **first-token A–J distribution** | knowledge, and whether the *distribution* moved even where the answer did not |
| `mmlu_pro` (0-shot, CoT) | 1,000, same items | `strict` and `lenient` parsing | knowledge after reasoning, plus format compliance |
| `ifeval` (full) | 541 | instruction verifier | instruction-format collapse |

`mmlu_pro_logprob` costs **~1 output token per item** (~1.3 min per model)
versus ~1,500 for the CoT variant (~21 min), so it is the high-n view of the
same items. The two rows are complementary:

- a drop in logprob accuracy that does **not** appear in the CoT row means the
  knowledge is still there and the *reasoning/format* path is what broke;
- a drop in the CoT row that does **not** appear in the logprob row means the
  adapter still knows the answers but no longer expresses them in the
  requested format.

Because both tasks build the identical dataset with `shuffle=False`, the same
`--seed` + `--max-samples` selects the same items and the rows are paired.

#### `strict` vs `lenient` — read this before quoting an accuracy

The adapters were SFT'd with a system prompt that says, verbatim:

> Respond with ONLY the exact text of your chosen option. No extra words, no
> explanations, no formatting.

MMLU-Pro instead demands a trailing `ANSWER: $LETTER`. Scored **strictly**, a
correct bare `D` counts as a failure, so the strict number measures
*instruction conflict*, not capability. `lenient` also accepts a bare letter,
`X) text`, or the exact option text. Both are reported; **use `lenient` as the
capability number**.

This is not hypothetical: on the 5-model run, `modal_response` scored 20.8%
strict and 62.3% lenient against a base model whose *direct-answer* accuracy is
60.2%. The "60-point collapse" was entirely output convention.

`gpqa_diamond` is available (`--tasks gpqa_diamond`) but is **not** in the
default set: this model reasons for ~4–6k tokens per item even when asked for
a direct answer, so it costs more than everything else combined and n=100 is
only a coarse tripwire. SWE-bench is behind `--include-swe` for the same
reason (docker sandboxes, agent loops, 24k context, no power at small n).

Measured on the RTX PRO 6000 with `--concurrency 128`: ~21 min for
`mmlu_pro` (1,000 items), ~1.3 min for `mmlu_pro_logprob`, ~2.6 min for
`ifeval`. The 5-model grid took **70 min** end to end — the adapters generate
far shorter answers than base, so they finish faster than the base model.

One trap worth preserving: `mmlu_pro(shuffle=True)` forwards `shuffle=True` to
`hf_dataset`, which draws a **random** seed. Two processes then see different
dataset orders and `sample_shuffle + limit` select *different items*, so
base-vs-adapter pairs silently stop overlapping. `build_task` therefore passes
`shuffle=False` and lets `eval(sample_shuffle=seed)` be the only randomness.

All five models are evaluated **back to back on identical items**; the first
`--models` entry is the base model:

```bash
uv run code/fine-tuning/capability_eval.py --port 8087 \
    --models Qwen/Qwen3.8-27B \
             Qwen3.8-27B-nz-wvs-modal_response-overall \
             Qwen3.8-27B-nz-wvs-sampled_response-overall \
             Qwen3.8-27B-nz-wvs-full_string_distribution-overall \
             Qwen3.8-27B-nz-wvs-first_token_distribution-overall
```

Results land in `output/capability/<model>/<run>/` (`config.json`,
`summary.json`, `summary_samples.csv`, raw logs) and the grid writes
`output/capability/comparison-<ts>/comparison.{md,csv}` with per-task
accuracies, **paired** deltas vs base, bootstrap CIs and exact McNemar tests.
For the logprob task it also writes the answer-distribution table: mean
P(gold), mean TVD, KL(base‖model) and top-1 agreement — i.e. how far the
answer distribution moved independently of the argmax.

Re-analysis does not need the GPU: `--compare-only` rebuilds the comparison
from each model's **newest** run directory and refreshes that run's
`summary.json`.

**Run selection is strict on purpose.** Both `--compare-only` and
`analyze.py` read the newest run per model and **abort** if it does not cover
every task. They never fall back to an earlier run: a silent fallback would
mix two grids and produce a comparison that is internally inconsistent. If a
partial or failed run is sitting in the way, rerun it or delete the directory
(`output/capability/_superseded/` holds retired runs and is ignored).
`analyze.py` additionally only renders `CAPABILITY_BASE_MODELS` (currently
`Qwen3.8-27B`).

Reading the results:

- Compare the **paired delta**, not two raw accuracies. Pairing removes item
  difficulty, which dominates the unpaired error bar.
- Quote `lenient`, not `strict`, for capability on MCQ tasks. `strict` is
  about obeying `ANSWER: $LETTER`; these adapters were trained to emit nothing
  but the option. Report both if you want to make the format point.
- `invalid_format` is reported next to accuracy on purpose, but it is a format
  signal, not a knowledge signal — see the `lenient` note above.
  `truncated` means the output cap was hit, so that row's accuracy is a lower
  bound.
- Thinking is **off** by default (`--thinking` enables it) to match the
  fine-tuning evals and to keep the cost bounded.
- `--concurrency 128` is the main speed knob (vLLM serves
  `--max-num-seqs 256`); inspect's own adaptive ramp starts at 20, so it is
  pinned.

### Prompted base-model baseline

To test how much of the fine-tuning effect is reproducible by prompting
alone, `prompted_eval.py` evaluates the base model with an instruction
("Respond as an average New Zealand person would.", configurable)
prepended to every dataset system prompt. Results land in
`output/evals/<label>/` exactly like any other eval, with
`label = <model short name>-prompted` by default. `analyze.py` maps a
`<base>-prompted` dir back onto `<base>`, so a **Prompted** row appears
under the base row in the fine-tuning and TVD tables. The fine-tuning table
shows Prompted alongside Base; the TVD table still reports its own vs-Base
delta.

It does not start a server: reuse a running eval server or bring one up
first. Because the instruction is prepended, each dataset system prompt
keeps its answer-format instruction at the end, and the per-system-prompt
breakdown is preserved (e.g. `no_persona` with vs without the addition).

```bash
# Against the usual vLLM server on localhost (EVAL_PORT, default 8087)
uv run code/fine-tuning/prompted_eval.py --model Qwen/Qwen3.8-27B

# OpenRouter base model, labelled to match its OpenRouter base-model dir
uv run code/fine-tuning/prompted_eval.py --provider openrouter \
    --model qwen/qwen3.8-27b --label qwen3.8-27b-prompted
```

Defaults to `modal_response` + `first_token_distribution` on
`train,validation` (overall population), skips dataset configs that
already have a completed run (`--force` reruns), and records the prompt
text in each run's `config.json`. The label must be
`<base eval dir>-prompted` for the row to join that model's table.

### Value map (NZ respondents and models)

`value_map.py` builds two maps of the NZ respondents and the evaluated models.

**Classic Inglehart–Welzel map.** It uses the classic indicator items in
`item_map_wvs.json` (importance of God, religious attendance and identity,
national pride, child qualities; justification of homosexuality, tolerance of
homosexual neighbours, trust, life satisfaction, political action), each scored
with fixed ordinal weights so respondents and models are scored identically.
The axes are survival → self-expression (x) and traditional → secular-rational
(y), matching the published map; both are standardised to the NZ respondents,
with the origin at their mean and the axes crossing at the centre.

**All-item PCA map.** It uses all 251 value-survey items from
`training-dataset/output/question_mapping.json` and the respondents in
`training-dataset/output/wvs_value_survey.csv`. Each answer option is a feature,
each item block is scaled to unit variance so questions with more options do not
dominate, and missing items are mean-imputed. PCA is fitted on the respondents,
then every model's answer distribution is projected onto the first two
components, standardised to the respondents' spread. The components are
data-driven and explain a small share of individual-level variance, so this map
is exploratory: see `pca_loadings.csv` for what each component captures.

Only the NZ sample of the WVS is in the repo, so neither map has a country
backdrop and both use the NZ respondents as the origin.

```bash
uv run code/fine-tuning/value_map.py
```

Each model uses its newest non-reasoning overall-population run, preferring
`modal_response` and falling back to the run's own training format (hosted
base models only have `full_string_distribution`). The exact run behind every
point is recorded in the coordinates files. Both maps highlight the base model,
the prompted variant and the four fine-tuned adapters; every other evaluated
base model is a faint dot, so adding an evaluation run adds a dot.

Outputs:

- `code/figures/value-map.png` — classic map
- `code/figures/value-map-pca.png` — all-item PCA map
- `code/figures/value-maps.png` — both maps side by side in one compact figure,
  sharing a frame so one respondent SD is the same distance in each panel
- `code/fine-tuning/output/value_map/coordinates.csv` — classic coordinates and the run used per group
- `code/fine-tuning/output/value_map/item_scores.csv` — each group's score on every indicator
- `code/fine-tuning/output/value_map/pca_coordinates.csv` — PCA coordinates and the run used per group
- `code/fine-tuning/output/value_map/pca_loadings.csv` — strongest option loading per item and component

### Analysis

### Continuing training from a previous run

Each upload includes the continuation state needed to pick training back up:
the LoRA adapter, `optimizer.pt` / `scheduler.pt` and `trainer_state.json`
(global step, epoch). To continue from an existing repo, pass `--resume-from`:

```bash
./code/fine-tuning/run.sh finetune uni-gpu1 -- \
    --dataset modal_response --subpopulation overall \
    --resume-from 1jamesthompson1/Qwen/Qwen3.5-9B-nz-wvs-modal_response-overall \
    --num-epochs 2 --upload-to-hf
```

Notes:

- `--num-epochs` is the **total** number of epochs across all segments, so pass
  `trained_epochs + new_epochs` (the optimizer/scheduler state carries over).
- Use the same base `--model` and LoRA hyperparameters as the original run
  (`--lora-r`, `--lora-alpha`, `--lora-dropout`, `--dora`, `--quantization`).
- If the original run saved no periodic checkpoint (e.g. `--save-steps` larger
  than the number of steps), only the step counter is preserved and the
  optimizer is rebuilt from scratch.
- `run_all.py` normally skips repos already on the hub; passing `--resume-from`
  through the `--` pass-through disables that skip.