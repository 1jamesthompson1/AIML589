# Dilemmas

Thirty everyday moral or ethical dilemmas, run through ten deployed-agent
profiles (three variants per profile). `profiles/<profile>.json` is the source
of truth: the system prompt, role summary, framing and each dilemma's
situation, question and answer options. Retired profiles live in `defunct/`.

## Workflow

```bash
# Run the served base model and every adapter it hosts (or pass --model
# explicitly for the models you want)
uv run python code/dilemmas/full_run.py \
    --model openai/Qwen/Qwen3.8-27B --epochs 50

# Export the newest complete bundle for the website and analysis
uv run python code/dilemmas/export_results.py

# Analysis: answer stats, H1, comparison pools and the survey subset
uv run marimo edit code/dilemmas/analyze.py
```

`full_run.py --status` shows progress. Every script's module docstring and
`--help` cover its flags and output format.

### Quality judge ensemble

`judge_quality.py` can score a pool with one judge or a high-reasoning
ensemble. The ensemble writes per-judge files to `judges/<slug>/` and a
median/majority result to `ensemble/` (`scores.json`, `index.csv`,
`meta.json`, `summary.md`), including agreement and top-17 overlap with the
previous survey build. A comparison missing any judge's score is dropped
with a warning and listed under `dropped_comparisons` in `meta.json`;
`competence_winner` is reported as `base`/`ft`/`tie`/`both_bad`, mapped
through the randomised A/B order. The cache key includes the reasoning
effort and token ceiling, so high-reasoning rows do not reuse default rows.

```bash
uv run python code/dilemmas/judge_quality.py \
    --subset complete/modal_response-overall \
    --judges "openrouter/anthropic/claude-haiku-5.5,openrouter/openai/gpt-6-luna,openrouter/deepseek/deepseek-v4.1-flash" \
    --reasoning-effort high --max-tokens 16000 --workers 32
```

Use `--estimate` first; `--limit N` scores N pairs per judge.

## Re-running a profile

Editing `profiles/<profile>.json` invalidates that profile's samples in an
existing batch. Mark the samples and refit the same bundle; clean samples are
reused by Inspect:

```bash
uv run python code/dilemmas/invalidate_samples.py \
    --run-id <run-id> --profile <profile>
uv run python code/dilemmas/full_run.py \
    --run-id <run-id> --refit --epochs <n> \
    --model openai/Qwen/Qwen3.8-27B
```

Use the batch's own epoch count for `--epochs`. `--refit` rebuilds `run.json`
and re-exports the batch, so the website and survey pick up the new wording.

## Pipeline

| Script | Purpose |
| --- | --- |
| `task.py` | Inspect task and judge |
| `full_run.py` | run models on all dilemmas into `output/logs/full/<run-id>/` |
| `export_results.py` | export a complete bundle to `output/runs/full/<run-id>/` |
| `generate_comparisons.py` | base vs fine-tune pairs in `output/comparisons/<run-id>/`; `all-models` emits only pairs with one base and one fine-tune |
| `embed_responses.py` | response embeddings for cosine distance |
| `judge_quality.py` | LLM scores: value divergence, survey quality, interest |
| `analyze.py` | figures, H1 test and the survey subset under `output/comparisons/<run-id>/survey/<model>/` |
| `h1_test.py` | fine-tune vs base permutation test |
| `spot_test.py` | quick single-dilemma runs |
| `sa_scanner.py` | situational-awareness scan |

The website reads `output/runs/full/<run-id>/index.json`; the survey reads
the selected subset under `output/comparisons/<run-id>/survey/<model>/` and
is pushed with `survey/qualtrics_sync.py`. `generate_comparisons.py` validates
that `base` is the non-adapter model and `ft` carries the `-nz-wvs-` marker;
it fails rather than writing a pool with the roles reversed. Runs are paired
by inspect `epoch` (the original sample index); older manifests without
`epoch` fall back to the export `version`. `analyze.py` prefers
`quality/index/<model>/ensemble/scores.json` when it exists, so the survey
subset ranks by the high-reasoning ensemble's mean `value_divergence`;
without an ensemble it falls back to the single-judge scores. A dilemma with
fewer than 17 scored pairs aborts the subset build instead of filling slots
with an unranked pair.

## Notes

- Append new dilemmas to the end of a profile's list: sample ids are
  index-based (`<profile>-<index>`), so inserting earlier shifts later ids.
- Spot-test a dilemma while editing it:
  `uv run python code/dilemmas/spot_test.py <profile>-<index>`.
- A one-sided dilemma (every model picks the same option) usually needs the
  losing option to carry a concrete duty, person or cost; keep the option
  wording symmetric.
