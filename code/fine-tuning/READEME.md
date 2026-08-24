# Fine-tuning apparatus

Fine-tune open-weight LLMs on the [WVS NZ value alignment dataset](https://huggingface.co/datasets/1jamesthompson1/wvs-nz-value-alignment) and evaluate their alignment with NZ population value distributions.

**Note:** the evaluation mentioned here is not the same as the full project evaluation in `code/evaluation/` which tests for behavior. This is simply a check on the model's ability to reproduce the empirical response distributions in the dataset.


## GPU machines

The current GPU machines are either going to be university hardware or vast.ai rented instances.

To allow for easy ssh access one can add a config to `~/.ssh/config` like:

```
Include [path-to-project-repo]/code/fine-tuning/.ssh_config
```

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

### Capability Evaluation

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