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

### Batch run (full config grid)

`run_all.py` drives the whole grid — all 4 dataset configs (finetuning
methods) x 3 subpopulations — using finetune.py's default hyperparameters
(see `uv run finetune.py --help`):

```bash
uv run ./code/fine-tuning/run_all.py uni-gpu1 Qwen/Qwen3.5-9B              # finetune + evaluate all 12
uv run ./code/fine-tuning/run_all.py uni-gpu1 Qwen/Qwen3.5-9B --skip-finetune   # evaluate only
uv run ./code/fine-tuning/run_all.py uni-gpu1 Qwen/Qwen3.5-9B --skip-eval       # finetune only
```

It fine-tunes each `(dataset, subpopulation)` pair (uploading the adapter to
HF), then starts a single multi-LoRA vLLM server with all 12 adapters and
runs `batch_eval.py` to evaluate every adapter on the validation split.
Set `EVAL_PORT` or `HF_COLLECTION` env vars to override defaults.

### Run metadata

Every run saves a `finetune_config.json` alongside the adapter with all hyperparameters, timestamp, and git commit hash.

## Dataset configs for evaluation

Four dataset configs are available (all share the same input prompts; only
the expected answer differs):

| Config | Expected answer | What it measures |
|---|---|---|
| `modal_response` | Mode (most common response) | Accuracy (exact match vs majority) + KL/CE vs true distribution |
| `sampled_response` | Random sample from cluster | Accuracy (exact match vs a typical individual) + KL/CE vs true distribution |
| `full_string_distribution` | Empirical distribution over categories | KL/CE vs true distribution (train: weighted NLL over option strings; eval: served logprobs — not yet implemented) |
| `first_token_distribution` | Empirical distribution; single-letter answers | First-token soft CE (Cao et al. 2025) — one forward pass, loss on the K answer-token logits at the answer position |

Each eval run covers **all subpopulations** (cluster_0, cluster_1, overall) in a single pass. The `subpopulation` column in `per_question_results.csv` identifies which subpopulation each row belongs to.

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
