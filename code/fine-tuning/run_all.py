#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "python-dotenv>=1.1.0",
# ]
# ///
"""
run_all.py — fine-tune + evaluate the full WVS config grid.

Runs 4 dataset configs (finetuning methods) x 3 subpopulations = 12 runs:
    Finetune:  modal_response, sampled_response,
               full_string_distribution, first_token_distribution
    Target:    cluster_0, cluster_1, overall

Jobs whose LoRA adapter already exists on the Hugging Face Hub
({HF_ORG}/{model_slug}-nz-wvs-{dataset}-{subpopulation}) are skipped.

Usage:
    uv run run_all.py <ssh-host> <model> [--skip-finetune] [--skip-eval] [--dry-run] [-- extra args...]

Example:
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --skip-eval
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --dry-run

Environment:
    EVAL_PORT      Local port for the eval server tunnel (default: 8087)
    HF_COLLECTION  Collection of adapters for serving
                   (default: $HF_ORG/wvs-nz-lora-adapters)
    HF_TOKEN / HF_ORG  Read from the repo root .env or the environment

Pipeline:
    1. finetune.py (default hyperparameters, see its --help) for each
       (dataset, subpopulation), uploading the LoRA adapter to
       {HF_ORG}/{model_slug}-nz-wvs-{dataset}-{subpopulation}
    2. serve.py once with ALL adapters (multi-LoRA) from the collection
    3. batch_eval.py runs the missing model x dataset evals in parallel
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SH = SCRIPT_DIR / "run.sh"
BATCH_EVAL = SCRIPT_DIR / "batch_eval.py"

DATASETS = [
    "modal_response",
    "sampled_response",
    "full_string_distribution",
    "first_token_distribution",
]
SUBPOPS = ["cluster_0", "cluster_1", "overall"]


def adapter_exists(repo_id: str, token: str | None) -> bool:
    """True if the adapter repo already exists on the HF Hub."""
    url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as e:
        return e.code == 200
    except Exception:
        return False


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fine-tune + evaluate the full WVS config grid (4x3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("host", help="ssh host (e.g. uni-gpu1, from .ssh_config)")
    p.add_argument(
        "model",
        nargs="?",
        default="Qwen/Qwen3.5-9B",
        help="Base model on HF Hub",
    )
    p.add_argument("--skip-finetune", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the finetune plan (with skip decisions) without running anything",
    )
    args, extra = p.parse_known_args(argv)
    args.extra = extra  # anything else is passed through to finetune.py
    return args


def banner(title: str):
    print()
    print("═" * 60)
    print(f"  {title}")
    print("═" * 60)


def run_finetune(host, model, dataset, subpop, extra):
    cmd = [
        str(RUN_SH),
        "finetune",
        host,
        "--",
        "--model",
        model,
        "--dataset",
        dataset,
        "--subpopulation",
        subpop,
        "--upload-to-hf",  # eval step serves adapters from the HF collection
        *extra,
    ]
    banner(f"FINETUNE  {model}  |  {dataset}  |  {subpop}")
    subprocess.run(cmd, check=True)


def server_is_up(url: str, timeout_s: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def serve_adapters(host, model, collection, port, extra):
    banner(f"SERVING all adapters from {collection} on :{port}")
    cmd = [
        str(RUN_SH),
        "serve",
        host,
        "--port",
        str(port),
        "--",
        "--model",
        model,
        "--hf-collection",
        collection,
        *extra,
    ]
    proc = subprocess.Popen(cmd)  # inherit stdio so run.sh logs are visible
    return proc


def stop_server(proc, timeout_s: float = 30.0):
    if proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)  # run.sh trap kills the ssh + tunnel
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_eval(port):
    banner("EVALUATING all served adapters on the validation split")
    cmd = ["uv", "run", str(BATCH_EVAL), "--port", str(port)]
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True)


def main(argv=None):
    args = parse_args(argv)
    model_slug = args.model.split("/")[-1]
    port = int(os.environ.get("EVAL_PORT", "8087"))
    url = f"http://localhost:{port}/v1/models"

    # Load HF_TOKEN / HF_ORG from the repo root .env (same source as run.sh)
    from dotenv import load_dotenv

    load_dotenv(SCRIPT_DIR.parent.parent / ".env")
    hf_token = os.environ.get("HF_TOKEN")
    hf_org = os.environ.get("HF_ORG")
    collection = os.environ.get("HF_COLLECTION")

    if not args.skip_finetune:
        for ds in DATASETS:
            for pop in SUBPOPS:
                repo = f"{hf_org}/{model_slug}-nz-wvs-{ds}-{pop}"
                if hf_org and adapter_exists(repo, hf_token):
                    print(f"[skip] {repo} already on hub")
                    continue
                if args.dry_run:
                    print(f"[dry] would run {ds} / {pop} -> {repo}")
                    continue
                run_finetune(args.host, args.model, ds, pop, args.extra)

    if args.skip_eval:
        return

    proc = serve_adapters(args.host, args.model, collection, port, args.extra)
    try:
        print(f"Waiting for server on localhost:{port}...", flush=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if server_is_up(url):
                break
            time.sleep(2)
        if not server_is_up(url):
            print("[error] server did not start within 600s — aborting")
            sys.exit(1)
        print("Server ready.")

        run_eval(port)
    except KeyboardInterrupt:
        print("\nInterrupted — shutting down server...")
    finally:
        stop_server(proc)

    print()
    print("Done. Results: code/fine-tuning/output/evals/")


if __name__ == "__main__":
    main()
