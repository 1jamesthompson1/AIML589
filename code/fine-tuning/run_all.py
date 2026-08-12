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
Pass --force to retrain and overwrite those repos anyway, or
--force-older-than HOURS to only retrain repos last updated more than
HOURS hours ago.

Usage:
    uv run run_all.py <ssh-host> <model> [--skip-finetune] [--skip-eval] [--dry-run] [-- extra args...]

Example:
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --skip-eval
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --dry-run
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --force
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --force-older-than 2
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --subpop cluster_0

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
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
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

# finetune.py-only flags: extra args are forwarded to serve.py too, but vLLM
# rejects them. Flags whose values could be mistaken for a new flag are not
# in this list (serve.py must never receive them).
FINETUNE_ONLY_FLAGS = {
    "--dataset",
    "--subpopulation",
    "--output-dir",
    "--lora-r",
    "--lora-alpha",
    "--lora-dropout",
    "--dora",
    "--quantization",
    "--lr",
    "--num-epochs",
    "--batch-size",
    "--eval-batch-size",
    "--gradient-accumulation-steps",
    "--max-seq-length",
    "--warmup-ratio",
    "--logging-steps",
    "--save-steps",
    "--eval-steps",
    "--upload-to-hf",
    "--resume-from",
}


def filter_finetune_only(extra: list[str]) -> list[str]:
    """Drop finetune-only flags (and their values) from ``extra``.

    Extra args after ``--`` are passed to both finetune.py and serve.py; vLLM
    aborts on unknown flags like ``--num-epochs``. Flags listed in
    ``FINETUNE_ONLY_FLAGS`` are removed along with their following value
    (``--flag value`` and ``--flag=value`` both handled).
    """
    out = []
    it = iter(extra)
    for a in it:
        flag = a.split("=", 1)[0]
        if flag in FINETUNE_ONLY_FLAGS and "=" not in a:
            nxt = next(it, None)
            if nxt is not None and not nxt.startswith("-"):
                continue  # consumed the flag's value
            if nxt is not None:
                out.append(nxt)  # next token is another flag, keep it
        elif flag not in FINETUNE_ONLY_FLAGS:
            out.append(a)
    return out


def adapter_last_modified(repo_id: str, token: str | None) -> datetime | None:
    """Last commit time of the adapter repo on the HF Hub.

    Returns None if the repo doesn't exist or the API is unreachable.
    """
    url = f"https://huggingface.co/api/models/{repo_id}"
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read())
            last_modified = data.get("lastModified")
            if not last_modified:
                return datetime.now(timezone.utc)  # repo exists, no timestamp
            return datetime.fromisoformat(last_modified.replace("Z", "+00:00"))
    except urllib.error.HTTPError as e:
        return datetime.now(timezone.utc) if e.code == 200 else None
    except Exception:
        return None


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
    p.add_argument(
        "--subpop",
        default=None,
        choices=SUBPOPS,
        help="Run only this subpopulation (skip the others)",
    )
    p.add_argument("--skip-finetune", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    force = p.add_mutually_exclusive_group()
    force.add_argument(
        "--force",
        action="store_true",
        help="Retrain every job and overwrite existing adapter repos on the "
        "HF Hub (default: skip jobs whose repo already exists)",
    )
    force.add_argument(
        "--force-older-than",
        type=float,
        metavar="HOURS",
        help="Retrain jobs whose adapter repo was last updated more than "
        "HOURS hours ago; skip jobs updated more recently",
    )
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
        # finetune-only flags (e.g. --num-epochs) must not reach vLLM
        *filter_finetune_only(extra),
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

    # When resuming from a previous upload, the target repo already exists on
    # the hub — don't let the skip logic swallow the run.
    resuming = any(
        a == "--resume-from" or a.startswith("--resume-from=") for a in args.extra
    )

    if not args.skip_finetune:
        for ds in DATASETS:
            subpops = [args.subpop] if args.subpop else SUBPOPS
            for pop in subpops:
                repo = f"{hf_org}/{model_slug}-nz-wvs-{ds}-{pop}"
                if hf_org and not resuming:
                    last_modified = adapter_last_modified(repo, hf_token)
                    if last_modified is not None:
                        age_h = (
                            datetime.now(timezone.utc) - last_modified
                        ).total_seconds() / 3600
                        if not args.force and args.force_older_than is None:
                            print(f"[skip] {repo} already on hub")
                            continue
                        if args.force_older_than is not None:
                            if age_h < args.force_older_than:
                                print(
                                    f"[skip] {repo} updated {age_h:.1f}h ago "
                                    f"(< {args.force_older_than:g}h)"
                                )
                                continue
                            print(f"[force] {repo} last updated {age_h:.1f}h ago")
                        else:
                            print(f"[force] {repo} already on hub, retraining")
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
