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

### Capability Evaluation

### Value map (OpenRouter base models)

`evaluate.py` can evaluate hosted base models through OpenRouter. Eligible
base-model and fine-tuned-adapter runs are included automatically by
`analyze.py` in a PCA map fitted to the actual NZ WVS respondents.

Run one model, then rebuild the analysis (requires `OPENROUTER_API_KEY` in
`.env`):

```bash
uv run code/fine-tuning/evaluate.py \
    --provider openrouter \
    --api-url https://openrouter.ai/api \
    --model deepseek/deepseek-v4.1-flash \
    --dataset full_string_distribution \
    --splits validation \
    --subpopulation overall \
    --temperature 0.0 --max-tokens 2048 \
    --concurrency 8 --max-retries 5

uv run python code/fine-tuning/analyze.py
```

Repeat the evaluation command for each desired model. OpenRouter requests
are attributed through `code/openrouter_attribution.py`.

A run is eligible when it is complete, uses the overall population and
logprobs, and reasoning is successfully disabled. Fine-tuned adapters use
their newest compatible evaluation on the format they were trained on and
must match the base runs' dataset revision.

`analyze.py` aligns answer categories to the canonical dataset, drops items
with incompatible options, and projects each model's prompt-averaged
answers and the NZ population onto two respondent-level PCA components.
PC1 is anchored as traditional/progressive and PC2 as religious/secular.

Outputs:

- `code/figures/value-map.png` — models, respondent cloud, and NZ population
- `code/figures/value-map-loadings.png` — questions defining each axis
- `code/fine-tuning/output/value_map/` — coordinates and item loadings

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