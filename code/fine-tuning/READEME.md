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

## Fine-tuning

Not yet completed...

## Evaluation

To do evaluation you load the model you want to evaluate on a GPU machine and then run the evaluation script from your dev machine. The evaluation script will send requests to the GPU machine to get the model's responses.

### Serving the model

Sipmly run `serve.sh` on the GPU machine to start a vLLM server. This will make the model available at `localhost:8087` on your dev machine. You can change the port if you want to run multiple servers.

The following command will serve the model `Qwen/Qwen3.6-27B-FP8` with gpu machine `uni-gpu1` on port `8087`:

```bash
./code/fine-tuning/serve.sh uni-gpu1 --model Qwen/Qwen3.6-27B-FP8 --host 0.0.0.0 --port 8087
```

### Running evaluation

Once the model is being served you can run the evaluation script from your dev machine. This will send requests to the model and save the results in `output/eval/<model>_<dataset>_<subpopulation>/` by default (e.g. `output/eval/Qwen_Qwen3.6-27B-FP8_distributional_overall/`).

```bash
uv run code/fine-tuning/evaluate_model.py \
    --model Qwen/Qwen3.6-27B-FP8 \
    --port 8087 \
    --dataset distributional \
    --subpopulation overall
```
