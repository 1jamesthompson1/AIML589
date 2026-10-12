# Dilemmas

Handwritten New Zealand moral dilemmas: one situation and question in, one
short model response out. The current set has 25 dilemmas across six work
environments and eleven value tensions. The welfare profile carries three
extra tension 9 variants that share one paperwork error - a review form
completed on a superseded version - and vary only in how sympathetic the
client is.

## Layout

- `profiles/<environment>.json`: editable source of truth for the system
  prompt and written dilemmas. Each dilemma carries its answer options and a
  `tension` id; the tension's name and value options come from `tensions.json`.
- `tensions.json`: reference definitions of the eleven value tensions (id,
  name, value options, guidance), curated from the ranked table in the
  Schwartz value-map section of `code/fine-tuning/analyze.py`.
- `task.py`: Inspect eval with a single response and a judge classification.
- `export_results.py`: exports successful samples from eval logs.
- `compare.py`: asks an LLM for one paragraph comparing two model responses,
  with the dilemma and its value tension supplied as context.
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
  `wvs-dilemma-runs-index/v2`): the six profiles, all 25 dilemmas (situation,
  question, answer options, and the tension name and value options resolved
  from `tensions.json`), the models and every run;
  `index.csv` is the same run table in flat form.
- `output/comparisons/<model-pair>/<profile>-<dilemma>/<pair-id>.json`:
  LLM difference paragraph, source responses/run IDs, value tension, prompts
  and comparison model usage. `output/comparisons/index.json` lists every
  stored comparison file for the website.

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
uv run python code/dilemmas/compare.py --dry-run
uv run python code/dilemmas/compare.py
```

For analysis, open the notebook with
`uv run marimo edit code/dilemmas/analyze.py`.

The judge can also be set with `JUDGE_MODEL`. Generated files stay under
`output/`, following the behavioural simulation logs/runs/comparisons
layout. `output/` links to `../../artifacts/dilemmas/`, independently of
behavioural simulations in `artifacts/bs/`. These files are ignored by Git
and included by the shared artifact sync/backup commands.

The website Results Viewer consumes these outputs through its **Dilemma
Results** tab (`website/src/components/DilemmasViewer.tsx`): the runs
manifest drives an inline view of every model's response to one dilemma
(with a per-model run selector and the judge classification), and the
comparisons index drives a survey-style view of one stored comparison with
its two source responses. The response is already short, so no separate
self-review generation is needed.

Comparisons use the same default OpenRouter model as behavioural simulations;
override with `--model <provider/model>`. `--compare <full-model-id-1>
<full-model-id-2>` restricts the generating models. Each comparison takes one
response from each of two different models on the same dilemma; repetitions
produce all cross-model combinations. Empty responses are skipped, and
mismatched dilemma versions are rejected. Agent labels follow sorted run IDs.
Use `--limit N` for a small batch. Rerunning skips existing summaries whose
inputs, prompts and comparison model are unchanged; `--force` regenerates them.
The index is always rebuilt from every stored comparison file, so a partial
batch still leaves a complete index.

The prompt includes the tension name, its two value descriptions and guidance
from `tensions.json`. It asks for supported differences in decisions and
reasons without forcing agents onto opposing sides or judging which is better.
No additional model audits are performed.
