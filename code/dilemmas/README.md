# Dilemmas

Handwritten New Zealand moral dilemmas: one situation and question in, one
short model response out. The current set has 31 dilemmas across sixteen
deployed-agent profiles and eleven value tensions. A profile is the realistic
agent the model is deployed as - a lending officer, telehealth assistant,
aged-care assessor, parish advisory assistant and so on - each with its own
system prompt. Where public policy applies, the prompt embeds short excerpts
of the real source (CCCFA and the Banking Code, the electricity Consumer Care
Obligations, the Fair Insurance Code, the Code of Health and Disability
Services Consumers' Rights, the ERA good-faith and justification tests,
Anglican safeguarding policy) in the style of the behavioural simulations,
with the source named. Each prompt also states the reply mode: the model
either replies directly to the person in front of it, or lays out the first
step of an agentic workflow (the decision and the plan). The everyday-service
and adviser profiles were added alongside the first five; the content
moderator profile was removed because its dilemmas were not working. The
welfare profile keeps its three extra tension 9 variants that share one
paperwork error - a review form completed on a superseded version - and vary
only in how sympathetic the client is.

## Layout

- `profiles/<profile>.json`: editable source of truth for the system
  prompt and written dilemmas. Each dilemma carries its answer options and a
  `tension` id; the tension's name and value options come from `tensions.json`.
  `role_summary` is the one-sentence description of the deployed role that the
  survey block and the website viewer show.
- `tensions.json`: reference definitions of the eleven value tensions (id,
  name, value options, guidance), curated from the ranked table in the
  Schwartz value-map section of `code/fine-tuning/analyze.py`.
- `task.py`: Inspect eval with a single response and a judge classification.
- `export_results.py`: exports successful samples from eval logs.
- `analyze.py`: marimo notebook for analysis. It renders the value-tension
  reference table from `tensions.json` to
  `code/figures/dilemmas-tensions.tex`, and per-dilemma answer statistics
  (mean chosen option index and entropy-based stability, per model) to
  `code/figures/dilemmas-answer-stats.tex`, both for the report
  (`\ctable{...}`).
- `output/logs/`: Inspect logs.
- `output/runs/<model>/<profile>-<dilemma>/<run-number>/`: transcript JSON
  and Markdown, model response (`response.txt`), judge result and run config.
  `index.json` is the website manifest (schema
  `wvs-dilemma-runs-index/v2`): every profile (name, role summary and the
  system prompt the runs used) and dilemma (situation, question, answer
  options, and the tension name and value options resolved from
  `tensions.json`, plus `choice_counts` — model -> judge choice -> exported
  runs, which drives the website's outcome chart), the models and every run;
  `index.csv` is the same run table in flat form.

## Workflow

Edit the profile JSON files directly; no generation step is needed. The
system prompts are adaptations of the behavioural simulation prompts and
are maintained here independently. Dilemmas link to a tension by its
`tension` id; the name and value options live in `tensions.json`. Append new
dilemmas to the end of a profile's list: sample ids are index-based
(`<profile>-<index>`), so inserting them earlier would shift the ids of the
dilemmas after them.

From the repository root, using configured model credentials:

```bash
uv run inspect eval code/dilemmas/task.py \
    --model <provider/model> --model-role judge=<provider/model> \
    --epochs 3 --log-dir code/dilemmas/output/logs
uv run python code/dilemmas/export_results.py
```

For analysis, open the notebook with
`uv run marimo edit code/dilemmas/analyze.py`.

### Spot tests

While iterating on one dilemma, re-run it and replace its exported runs so the
local website copy shows only the new results:

```bash
uv run python code/dilemmas/spot_test.py cosmetic_clinic-0
uv run python code/dilemmas/spot_test.py --list
```

The scenario may be a sample id (`<profile>-<index>`) or a dilemma id. With no
`--model`, the spot-test set runs: **qwen3.8-27b, deepseek-v4.1-flash,
gpt-5.6-luna, claude-haiku-5.5** and **glm-5.3-flash**, 5 epochs each
(`--epochs`, `--model` repeatable, judge from `--judge` or `JUDGE_MODEL`).
The fine-tuned Qwen adapters are served locally and run with `--base-url` and
`--model openai/<served-model-name>`.

The script clears the scenario's exported run directories, re-exports the
scenario from its latest run set only (other dilemmas keep their history), and
prints each run's pick, the per-model counts and stability, the cross-model
divergence, an **overall stability** pooled across the runs the spot test made,
and the **cost** of those runs: exported token usage priced with
`MODEL_PRICES` (OpenRouter list prices; re-check when adding models), target
and judge together. Unpriced models (e.g. a local vLLM adapter) report tokens
only, so their cost line is a lower bound. Runs use the provider's default
reasoning level — no reasoning effort is set for the target or the judge.

Generated files stay under `output/`, following the behavioural simulation
logs/runs layout. `output/` links to `../../artifacts/dilemmas/`,
independently of behavioural simulations in `artifacts/bs/`. These files are
ignored by Git and included by the shared artifact sync/backup commands.

The website Results Viewer consumes these outputs through its **Dilemma
Results** tab (`website/src/components/DilemmasViewer.tsx`): the runs
manifest drives an inline view of every model's response to one dilemma
(with a per-model run selector, the judge classification and the profile's
system prompt). The response is already short, so no separate self-review
generation is needed.
