# Fine-tuning apparatus

Fine-tune open-weight LLMs on the [WVS NZ value alignment dataset](https://huggingface.co/datasets/1jamesthompson1/wvs-nz-value-alignment) and evaluate their alignment with NZ population value distributions.

**Note:** the evaluation mentioned here is not the same as the full project evaluation in `code/evaluation/` which tests for behavior. This is simply a check on the model's ability to reproduce the empirical response distributions in the dataset.

## Workflow

This project is setup to work from a lightweight dev machine and ssh into gpu machines to both do the fine-tuning and serve the model for evaluation.

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
    --dataset single_modal --subpopulation cluster_0 --upload-to-hf
```

### Uploading adapters

Each fine-tuning run produces a LoRA adapter (a few MB) which gets uploaded to its own Hugging Face Hub repo:

```
{HF_ORG}/{model_slug}-nz-wvs-{dataset}-{subpopulation}
```

e.g. `1jamesthompson1/Qwen3.6-27B-nz-wvs-single_modal-cluster_0`

To upload, set the `HF_ORG` environment variable in `.env` and pass `--upload-to-hf`:

```bash
./code/fine-tuning/run_finetune.sh uni-gpu1 \
    --dataset single_modal --subpopulation cluster_0 \
    --upload-to-hf
```

You can also auto-add each new adapter repo to a HF Collection for easy browsing:

```bash
./code/fine-tuning/run_finetune.sh uni-gpu1 \
    --dataset single_modal --subpopulation cluster_0 \
    --upload-to-hf \
    --hf-collection "1jamesthompson1/wvs-nz-lora-adapters"
```

### Run metadata

Every run saves a `finetune_config.json` alongside the adapter with all hyperparameters, timestamp, and git commit hash.

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
    --adapter cluster_0=1jamesthompson1/Qwen3.6-27B-nz-wvs-single_modal-cluster_0
```

`--port` sets the port on your laptop (default `8080`). `run.sh` handles random free port allocation on the remote and SSH tunnel setup automatically.

### Legacy scripts

The individual `serve.sh` and `run_finetune.sh` scripts still work as before, but `run.sh` is the recommended entry point.

#### Serving with LoRA adapters

Each fine-tuned adapter lives in its own HF Hub repo. Load one or more adapters on top of a base model:

```bash
# Single adapter
./code/fine-tuning/serve.sh uni-gpu1 --port 8087 \
    --model Qwen/Qwen3.6-27B \
    --adapter cluster_0=1jamesthompson1/Qwen3.6-27B-nz-wvs-single_modal-cluster_0

# Multiple adapters loaded simultaneously (clients select via model param)
./code/fine-tuning/serve.sh uni-gpu1 --port 8087 \
    --model Qwen/Qwen3.6-27B \
    --adapter cluster_0=1jamesthompson1/Qwen3.6-27B-nz-wvs-single_modal-cluster_0 \
    --adapter cluster_1=1jamesthompson1/Qwen3.6-27B-nz-wvs-single_modal-cluster_1 \
    --adapter overall=1jamesthompson1/Qwen3.6-27B-nz-wvs-distributional-overall
```

##### Auto-construct adapter from dataset + subpopulation

`serve.sh` can auto-construct the adapter repo name for you using `--dataset` and `--subpopulation`:

```bash
./code/fine-tuning/serve.sh uni-gpu1 --port 8087 \
    --model Qwen/Qwen3.6-27B \
    --dataset single_modal --subpopulation cluster_0
```

This looks up `HF_ORG` from your `.env`, builds the repo name `{HF_ORG}/Qwen3.6-27B-nz-wvs-single_modal-cluster_0`, and passes `--adapter cluster_0=<repo>` to `serve.py`.

**Client usage:** In your evaluation script, set `--model <adapter_name>` to target a specific fine-tuned version:

```bash
uv run evaluate.py --port 8087 --model cluster_0 \
    --dataset single_modal --subpopulation cluster_0
```

### Running evaluation

Once the model is being served you can run the evaluation script from your dev machine. This will send requests to the model and save the results in `output/eval/<model>_<dataset>_<subpopulation>/` by default.

```bash
uv run code/fine-tuning/evaluate.py \
    --model Qwen/Qwen3.6-27B \
    --port 8087 \
    --dataset distributional \
    --subpopulation overall
```
