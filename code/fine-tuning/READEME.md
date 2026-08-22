# Fine-tuning apparatus

Fine-tune open-weight LLMs on the [WVS NZ value alignment dataset](https://huggingface.co/datasets/1jamesthompson1/wvs-nz-value-alignment) and evaluate their alignment with NZ population value distributions.

**Note:** the evaluation mentioned here is not the same as the full project evaluation in `code/evaluation/` which tests for behavior. This is simply a check on the model's ability to reproduce the empirical response distributions in the dataset.

## Workflow

This project is setup to work from a lightweight dev machine and ssh into gpu machines to both do the fine-tuning and serve the model for evaluation. The evaluation code (i.e calling API endpoints) and analysis is done on the dev machine.

There is a single `run.sh` script that handles most of GPU machine workflow see [CLI](#cli) below.

## GPU machines

The current GPU machines are either going to be university hardware or vast.ai rented instances.

To allow for easy ssh access one can add a config to `~/.ssh/config` like:

```
Include [path-to-project-repo]/code/fine-tuning/.ssh_config
```

### Cloud GPU machines

The cloud GPU machines should be given a ssh config in `code/fine-tuning/.ssh_config`. This allows various commands later to login easily.

See your cloud provider for how to setup ssh access to your GPU machine. You will need to add the public key of your dev machine to the cloud instance.

### University GPU machines

There are many university machines with GPUs. They operate on a first in first serve so finding a free GPU is a bit of a pain. 

The `find_gpu.sh` script can help with this it outputs a list of available machines with free GPUs. This requires that you have a `vuw-lab` ssh setup on your machine. See uni docs for how to set this up.

When you run the script it will update your project `code/fine-tuning/.ssh_config` with the best three available machine it finds. You can then ssh into that machine and run the fine-tuning or serving scripts using `uni-gpuX` as the host name.

## CLI

All remote operations go through a single `run.sh` script:

```bash
# Fine-tune a model
./code/fine-tuning/run.sh finetune <ssh-host> [-- finetune args...]

# Serve a model for evaluation
./code/fine-tuning/run.sh serve <ssh-host> [--port PORT] [-- serve args...]
```

The `--` separator is optional if you don't need to pass flags that conflict with `run.sh` options.

### Fine-tuning

Fine tuning is done on a GPU machine. It copies the `finetune.py` script to the remote and runs it there.

The finetuning is depending on parameters either a simple SFT or a slightly more complex distributional fine-tuning. In all cases it is LoRA fine-tuning on top of a base model.

```bash
./code/fine-tuning/run.sh finetune uni-gpu1 -- \
    --dataset modal_response --subpopulation cluster_0 --upload-to-hf
```

### Uploading adapters

Each fine-tuning run produces a LoRA adapter (a few MB) which gets uploaded to its own Hugging Face Hub repo:

```
{HF_ORG}/{model_slug}-nz-wvs-{dataset}-{subpopulation}
```

e.g. `1jamesthompson1/Qwen3.6-27B-nz-wvs-modal_response-cluster_0`

To upload, set the `HF_ORG` environment variable in `.env` and pass `--upload-to-hf`:

```bash
./code/fine-tuning/run.sh finetune uni-gpu1 \
    --dataset modal_response --subpopulation cluster_0 \
    --upload-to-hf
```

You can also auto-add each new adapter repo to a HF Collection for easy browsing:

```bash
./code/fine-tuning/run.sh finetune uni-gpu1 \
    --dataset modal_response --subpopulation cluster_0 \
    --upload-to-hf \
    --hf-collection "1jamesthompson1/wvs-nz-lora-adapters"
```

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

### Batch run (full config grid)

`run_all.py` drives the whole grid — all 4 dataset configs (finetuning
methods) x 3 subpopulations — using finetune.py's default hyperparameters
(see `uv run finetune.py --help`):

```bash
uv run ./code/fine-tuning/run_all.py uni-gpu1 Qwen/Qwen3.5-9B              # finetune + evaluate all 12
uv run ./code/fine-tuning/run_all.py uni-gpu1 Qwen/Qwen3.5-9B --skip-finetune   # evaluate only
uv run ./code/fine-tuning/run_all.py uni-gpu1 Qwen/Qwen3.5-9B --skip-eval       # finetune only
uv run ./code/fine-tuning/run_all.py uni-gpu1 Qwen/Qwen3.5-9B --subpop cluster_0
```

`--force` retrains every job even if its adapter repo already exists on the
HF Hub (re-uploading over the old adapters). Use this to redo the grid after
changing the data pipeline or hyperparameters:

```bash
uv run ./code/fine-tuning/run_all.py uni-gpu1 Qwen/Qwen3.5-9B --force
```

To retrain only the stale repos — e.g. anything last updated more than 2
hours ago — use `--force-older-than HOURS` instead:

```bash
uv run ./code/fine-tuning/run_all.py uni-gpu1 Qwen/Qwen3.5-9B --force-older-than 2
```

`--subpop` limits the finetune step to one subpopulation (eval always covers
the full validation split — see below).

It fine-tunes each `(dataset, subpopulation)` pair (uploading the adapter to
HF), then starts a single multi-LoRA vLLM server with all 12 adapters and
runs `batch_eval.py` to evaluate every adapter on the validation split.
Set `EVAL_PORT` or `HF_COLLECTION` env vars to override defaults.

### Run metadata

Every run saves a `finetune_config.json` alongside the adapter with all hyperparameters, timestamp, and git commit hash.

### Main report table

`analyze.py` also renders the main fine-tuning table for the report: for each
base model a table with one row per fine-tuned version (base model at the
top) and 9 columns — accuracy, cross-entropy and KL divergence on the train,
validation and overall splits. Metrics are taken from the modal eval config
(`modal_response`) so accuracy always means exact match against the modal
response and CE/KL are against the empirical response distribution. Every
fine-tuned version is matched to the subpopulation it was trained on; the
base model is scored on the overall population. The LaTeX (booktabs) tables
are written to `code/figures/ft-results-<model>.tex` and included in the
report with `\ctable{ft-results-<model>.tex}{<caption>}`.

Note: only Qwen3.5-9B has eval runs in the current config format
(`modal_response`/`sampled_response`/`first_token_distribution`); it is the
only model that gets a table. The Qwen3.5-2B and Qwen3.6-27B runs use the
older `single_modal`/`single_sample` config naming and predate per-row
split/subpopulation tagging — re-evaluate them before they can appear in the
table.

## Dataset configs for evaluation

Four dataset configs are available (all share the same input prompts; only
the expected answer differs):

| Config | Expected answer | What it measures |
|---|---|---|
| `modal_response` | Mode (most common response) | Accuracy (exact match vs majority) + KL/CE vs true distribution |
| `sampled_response` | Random sample from cluster | Accuracy (exact match vs a typical individual) + KL/CE vs true distribution |
| `full_string_distribution` | Empirical distribution over categories | KL/CE vs true distribution (train: weighted NLL over option strings; eval: served logprobs — not yet implemented) |
| `first_token_distribution` | Empirical distribution; single-letter answers | Accuracy (vs the modal letter) + KL/CE vs true distribution over single-letter answers |

Each eval run is **one inference pass over the train + validation splits**
(~2050 examples total, all subpopulations together by default). Every row is
tagged with its `split` (train/validation) and `subpopulation` in
`per_question_results.csv`, so analysis can re-split the results. Use
`--subpopulation <name>` to evaluate only one subpopulation and `--reasoning`
to enable chain-of-thought (default off).

`evaluate.py` supports all four configs. For `first_token_distribution` the model's per-letter probabilities at the answer position are scored against the empirical distribution (matching the first-token training loss); accuracy is exact match against the modal letter.

### Batch evaluation

`batch_eval.py` runs the missing evaluations in parallel — one full-set
(train+validation) pass per model per dataset config:

| Flag | Default |
|---|---|
| `--datasets` | `modal_response,sampled_response,first_token_distribution` |
| `--subpopulation` | all subpopulations in one pass (forwarded to each eval) |
| `--model` | all models on the server |
| `--concurrency` | auto-detect from server KV cache |

```bash
uv run code/fine-tuning/batch_eval.py --port 8087                                    # all models, 3 configs
uv run code/fine-tuning/batch_eval.py --port 8087 --model Qwen3.5-9B-nz-wvs-modal_response-overall
uv run code/fine-tuning/batch_eval.py --port 8087 --datasets modal_response --dry-run
```

Completed runs are detected from the eval `config.json`; only runs covering
both train and validation splits (unfiltered, non-reasoning) count, so old
validation-only or per-cluster evals don't block the new full-set passes.

## Evaluation

To do evaluation you load the model you want to evaluate on a GPU machine and then run the evaluation script from your dev machine. The evaluation script will send requests to the GPU machine to get the model's responses.

### Serving the model

Simply run `run.sh serve` to start a vLLM server on a GPU machine and tunnel the port back to your dev machine.

```bash
# Base model only (no adapters)
./code/fine-tuning/run.sh serve uni-gpu1 --port 8087 -- \
    --model Qwen/Qwen3.6-27B

# With a LoRA adapter
./code/fine-tuning/run.sh serve uni-gpu1 --port 8087 -- \
    --model Qwen/Qwen3.6-27B \
    --adapter cluster_0=1jamesthompson1/Qwen3.6-27B-nz-wvs-modal_response-cluster_0
```

`--port` sets the port on your laptop (default `8080`). `run.sh` handles random free port allocation on the remote and SSH tunnel setup automatically.
