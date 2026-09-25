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
    uv run run_all.py <ssh-host> <model> [--eval-only | --train-only] [--no-serve] [--dry-run] [-- extra args...]
    uv run run_all.py --vast <model> [--keep-vast] [...]     # provision GPUs on vast.ai

Example:
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --train-only   # fine-tune, no serve/eval
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --eval-only    # serve + evaluate existing adapters
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --eval-only --no-serve   # eval an already-running server
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --dry-run
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --force
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --force-older-than 2
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --subpop cluster_0
    uv run run_all.py uni-gpu1 Qwen/Qwen3.5-9B --adapters modal_response-overall,sampled_response-overall
    uv run run_all.py --vast Qwen/Qwen3.5-9B                 # cloud: rent, run, destroy

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
    3. evaluate.py runs the missing model x dataset evals in parallel
       (merges what used to be batch_eval.py)

Cloud mode (--vast):
    Instead of an <ssh-host>, run_all.py calls vastgpu.py (see its
    docstring) to rent the cheapest available RTX_PRO_6000 instance on
    vast.ai, waits for it to boot, writes its ssh config to .ssh_config
    (Host vast-gpu1) and runs the whole pipeline there. The instance is
    destroyed at the end unless --keep-vast is given.
"""

import argparse
import json
import os
import select
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_SH = SCRIPT_DIR / "run.sh"
EVALUATE = SCRIPT_DIR / "evaluate.py"
VAST_GPU = SCRIPT_DIR / "vastgpu.py"

DATASETS = [
    "modal_response",
    "sampled_response",
    "full_string_distribution",
    "first_token_distribution",
]
SUBPOPS = ["cluster_0", "cluster_1", "overall"]

# How long to wait after launching the server before the first health probe
# (model + LoRA loading takes minutes). Keep in sync with serve.py.
STARTUP_GRACE_S = 180

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


# serve.py-only flags: passed to serve.py but must never reach finetune.py
# (which rejects unknown flags). E.g. --max-lora-rank must match the trained
# LoRA rank or vLLM skips the adapters at serve time.
SERVE_ONLY_FLAGS = {
    "--max-lora-rank",
}


def filter_serve_only(extra: list[str]) -> list[str]:
    """Drop serve-only flags (and their values) from ``extra``.

    Mirrors ``filter_finetune_only``: the flag itself and its following value
    (or ``--flag=value``) are removed, other tokens are kept.
    """
    out = []
    it = iter(extra)
    for a in it:
        flag = a.split("=", 1)[0]
        if flag in SERVE_ONLY_FLAGS and "=" not in a:
            nxt = next(it, None)
            if nxt is not None and not nxt.startswith("-"):
                continue  # consumed the flag's value
            if nxt is not None:
                out.append(nxt)  # next token is another flag, keep it
        elif flag not in SERVE_ONLY_FLAGS:
            out.append(a)
    return out


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
    # Everything after a bare `--` is pure pass-through for the remote
    # scripts (finetune.py/serve.py). Split it off BEFORE argparse: otherwise
    # the first option-looking token (e.g. `--lora-r`) is consumed as the
    # still-open `model` positional and breaks host/model detection.
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough: list[str] = []
    if "--" in argv:
        sep = argv.index("--")
        passthrough = argv[sep + 1 :]
        argv = argv[:sep]
    p = argparse.ArgumentParser(
        description="Fine-tune + evaluate the full WVS config grid (4x3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "host",
        nargs="?",
        help="ssh host (e.g. uni-gpu1, from .ssh_config). Not needed with --vast",
    )
    p.add_argument(
        "model",
        nargs="?",
        default=None,
        help="Base model on HF Hub (default: Qwen/Qwen3.5-9B)",
    )
    p.add_argument(
        "--subpop",
        default="overall",
        choices=SUBPOPS,
        help="Run only this subpopulation (skip the others)",
    )
    p.add_argument(
        "--adapters",
        default=None,
        metavar="SUFFIX[,SUFFIX...]",
        help="Only run {dataset}-{population} suffixes listed here, e.g. "
        "'modal_response-overall,sampled_response-cluster_0'. Limits "
        "finetune jobs AND which collection adapters are served/evaluated. "
        "Default: all matching adapters.",
    )
    p.add_argument(
        "--vast",
        action="store_true",
        help="Provision a vast.ai RTX_PRO_6000 instance (via vastgpu.py), "
        "run the pipeline on it, then destroy it. Cannot be combined with "
        "an <ssh-host> positional.",
    )
    p.add_argument(
        "--keep-vast",
        action="store_true",
        help="With --vast: leave the rented instance running after the run "
        "(tear it down later with `uv run vastgpu.py teardown`)",
    )
    p.add_argument(
        "--vast-args",
        default="",
        help="With --vast: extra args forwarded to `vastgpu.py launch`, "
        "e.g. '--max-price 1.5' or '--min-inet 5000'. "
        "Quote the whole string.",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip fine-tuning: serve the existing adapters and evaluate "
        "them (same as --skip-finetune)",
    )
    mode.add_argument(
        "--train-only",
        action="store_true",
        help="Fine-tune only: skip serving and evaluation (same as --skip-eval)",
    )
    p.add_argument("--skip-finetune", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--skip-eval", action="store_true", help=argparse.SUPPRESS)
    p.add_argument(
        "--no-serve",
        action="store_true",
        help="With --eval-only: evaluate against a server already running on "
        "--port (the ssh tunnel must be up) instead of starting one",
    )
    p.add_argument(
        "--eval-concurrency",
        type=int,
        default=DEFAULT_EVAL_CONCURRENCY,
        help="Number of evaluations to run in parallel (multi-LoRA serving "
        "can't batch across adapters, so keep this low; default 2)",
    )
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
    args.extra = extra + passthrough  # anything else is passed through to finetune.py

    # The model positional follows the optional host, so with --vast the
    # single positional lands in `host`. Untangle that here:
    #   run_all.py uni-gpu1 Qwen/Qwen3.5-9B   -> host + model
    #   run_all.py --vast Qwen/Qwen3.5-9B     -> model only
    if args.vast and args.host:
        if args.model is not None:
            p.error("--vast provisions its own instance; don't pass a <ssh-host>")
        args.model = args.host
        args.host = None
    if args.model is None and not args.vast:
        args.model = "Qwen/Qwen3.5-9B"
    if not args.vast and not args.host:
        p.error("missing <ssh-host> (or use --vast to provision one)")
    if args.no_serve and not (args.eval_only or args.skip_finetune):
        p.error("--no-serve only makes sense with --eval-only/--skip-finetune")
    if args.no_serve and args.vast:
        p.error("--no-serve cannot be combined with --vast (nothing would be serving)")
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
        # serve-only flags (e.g. --max-lora-rank) must not reach finetune.py
        *filter_serve_only(extra),
    ]
    banner(f"FINETUNE  {model}  |  {dataset}  |  {subpop}")
    subprocess.run(cmd, check=True)


def model_short_name(model: str) -> str:
    return model.split("/", 1)[-1]


def server_is_up(url: str, timeout_s: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def serve_adapters(host, model, collection, port, extra, adapter_suffixes=None):
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
    if adapter_suffixes:
        cmd.extend(["--adapter-suffixes", adapter_suffixes])
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


# ── evaluation ────────────────────────────────────────────────

EVAL_DATASETS = ["modal_response", "first_token_distribution"]
EVAL_DIR = SCRIPT_DIR / "output" / "evals"
DEFAULT_EVAL_CONCURRENCY = 4


def get_served_models(url: str) -> list[str]:
    """Model ids exposed by the vLLM server (assumes vLLM's default api key)."""
    req = urllib.request.Request(url, headers={"Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    return [m["id"] for m in data.get("data", [])]


def existing_completed_runs(model_short: str) -> set[tuple]:
    """Set of (dataset, subpopulation-or-None) tuples with completed evals.

    ``subpopulation=None`` means a full-set pass (train + validation, all
    subpopulations). Runs only count with both splits covered and no
    reasoning/abort flags — old validation-only passes don't block reruns.
    """
    completed = set()
    model_dir = EVAL_DIR / model_short
    if not model_dir.is_dir():
        return completed
    for run_dir in model_dir.iterdir():
        config_path = run_dir / "config.json"
        results_path = run_dir / "per_question_results.csv"
        if (
            not (config_path.exists() and results_path.exists())
            or results_path.stat().st_size == 0
        ):
            continue
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            continue
        if config.get("reasoning") or config.get("aborted"):
            continue
        splits = config.get("splits") or []
        if not {"train", "validation"}.issubset(splits):
            continue
        completed.add((config.get("dataset"), config.get("subpopulation")))
    return completed


def run_eval_jobs(
    jobs: list[tuple[str, str]], port: int, subpop: str | None, concurrency: int
) -> list[tuple[str, str]]:
    """Run `evaluate.py` for each (model, dataset) job in parallel.

    More than one pass beats the server if it goes down or a query hangs.
    Returns the jobs that failed after one retry attempt.
    """
    api_url = f"http://localhost:{port}"
    pool = ThreadPoolExecutor(max_workers=concurrency)

    def run_one(model: str, ds: str) -> int:
        cmd = [
            "uv",
            "run",
            str(EVALUATE),
            "--api-url",
            api_url,
            "--api-key",
            "EMPTY",
            "--model",
            model,
            "--dataset",
            ds,
        ]
        if subpop:
            cmd += ["--subpopulation", subpop]
        label = f"{model_short_name(model)} {ds}"
        if subpop:
            label += f" subpop={subpop}"
        print(f"  [start] {label}", flush=True)
        t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=SCRIPT_DIR,
        )
        out_lines = []
        while proc.poll() is None:
            r, _, _ = select.select([proc.stdout], [], [], 5)
            if r:
                line = proc.stdout.readline()
                if line:
                    out_lines.append(line)
                    print(f"  [{label}] {line}", end="", flush=True)
        for line in proc.stdout:
            out_lines.append(line)
            print(f"  [{label}] {line}", end="", flush=True)
        proc.stdout.close()
        stderr_text = proc.stderr.read()
        proc.stderr.close()
        ret = proc.returncode
        elapsed = int(time.monotonic() - t0)
        if ret != 0:
            print(f"  [FAIL]  {label}  (elapsed {elapsed}s)")
            for line in stderr_text.strip().splitlines():
                print(f"           {line}")
        else:
            print(f"  [done]  {label}  (elapsed {elapsed}s)")
        return ret

    def run_pool(job_list: list[tuple[str, str]]) -> list[tuple[str, str]]:
        failed = []
        futures = {pool.submit(run_one, *j): j for j in job_list}
        for future in as_completed(futures):
            job = futures[future]
            try:
                ret = future.result()
            except Exception as e:
                print(f"  [exception] {job}: {e}")
                ret = 1
            if ret != 0:
                failed.append(job)
        pool.shutdown(wait=True)
        return failed

    failed = run_pool(jobs)
    if failed:
        print(f"\n  retrying {len(failed)} failed job(s)...\n")
        pool = ThreadPoolExecutor(max_workers=concurrency)
        failed = run_pool(failed)
    return failed


def run_eval(port, subpop=None, concurrency: int = DEFAULT_EVAL_CONCURRENCY):
    banner(
        f"EVALUATING all served adapters on the validation split "
        f"(concurrency={concurrency})"
    )
    url = f"http://localhost:{port}/v1/models"
    try:
        models = get_served_models(url)
    except Exception as e:
        print(f"[error] could not reach server at {url}: {e}")
        sys.exit(1)
    if not models:
        print("[error] no models returned by server")
        sys.exit(1)

    jobs = []
    for model in models:
        short = model_short_name(model)
        completed = existing_completed_runs(short)
        for ds in EVAL_DATASETS:
            if (ds, subpop) in completed:
                print(f"  [skip] {short} {ds}: already evaluated")
                continue
            jobs.append((model, ds))
    if not jobs:
        print("  all evaluations already complete — nothing to run.")
        return
    print(f"  {len(jobs)} evaluation(s) needed across {len(models)} model(s):")
    for model in models:
        n = sum(1 for m, _ in jobs if m == model)
        print(f"    - {model_short_name(model)}: {n}")
    print()

    failed = run_eval_jobs(jobs, port, subpop, concurrency)
    if failed:
        print(
            f"\n[error] {len(failed)} evaluation(s) still failing: "
            + ", ".join(f"{m} {d}" for m, d in failed)
        )
        sys.exit(1)
    print(f"\nAll {len(jobs)} evaluations completed.")


def provision_vast(dry_run: bool, vast_args: str = "") -> str | None:
    """Provision a vast.ai GPU instance via vastgpu.py; return the ssh alias.

    ``vast_args`` (from --vast-args) is forwarded to `vastgpu.py launch`,
    e.g. "--template-hash <hash>" or "--max-price 1.5".

    On failure this exits, because nothing can run without a host.
    """
    if dry_run:
        print("[dry] would provision a vast.ai RTX_PRO_6000 instance (vastgpu.py)")
        return None
    banner("PROVISIONING vast.ai RTX_PRO_6000 instance")
    import shlex

    cmd = ["uv", "run", str(VAST_GPU), "launch", "--yes", *shlex.split(vast_args)]
    try:
        proc = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        # Don't hide the provisioning failure — print what vastgpu.py said.
        print(e.stdout or "", end="")
        print(e.stderr or "", file=sys.stderr, end="")
        sys.exit("[error] vast.ai provisioning failed (see output above)")
    print(proc.stdout, end="")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    alias = lines[-1].strip() if lines else ""
    if not alias:
        sys.exit("[error] vastgpu.py launch did not return an ssh host alias")
    print(f"Host: {alias}")
    return alias


def teardown_vast() -> None:
    """Destroy the vast.ai instance provisioned at the start of the run."""
    banner("TEARING DOWN vast.ai instance")
    subprocess.run(
        ["uv", "run", str(VAST_GPU), "teardown", "--yes"],
        cwd=SCRIPT_DIR,
        check=False,
    )


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

    # In cloud mode, run_all.py provisions the GPU instance itself.
    host = args.host
    vast_alias = None
    if args.vast:
        vast_alias = provision_vast(args.dry_run, args.vast_args)
        host = vast_alias

    # --eval-only/--train-only are the explicit spellings of the old
    # --skip-finetune/--skip-eval flags; both keep working.
    skip_finetune = args.skip_finetune or args.eval_only
    skip_eval = args.skip_eval or args.train_only

    try:
        if not skip_finetune:
            wanted = {s.strip() for s in (args.adapters or "").split(",") if s.strip()}
            for ds in DATASETS:
                subpops = [args.subpop] if args.subpop else SUBPOPS
                for pop in subpops:
                    repo = f"{hf_org}/{model_slug}-nz-wvs-{ds}-{pop}"
                    if wanted and f"{ds}-{pop}" not in wanted:
                        print(f"[skip] {repo}: not in --adapters {args.adapters}")
                        continue
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
                    run_finetune(host, args.model, ds, pop, args.extra)

        # Dry-run prints the plan without running anything (also in cloud mode).
        if skip_eval or args.dry_run:
            if args.dry_run and not skip_eval:
                if args.no_serve:
                    print(
                        f"[dry] would evaluate {args.model} adapters on the "
                        f"server already running on localhost:{port}"
                    )
                else:
                    print(
                        f"[dry] would serve {args.model} adapters from "
                        f"{collection} on {host} and evaluate on "
                        f"localhost:{port}"
                    )
            return

        proc = None
        if args.no_serve:
            print(f"[eval] using the server already running on localhost:{port}")
        else:
            proc = serve_adapters(
                host, args.model, collection, port, args.extra, args.adapters
            )
        try:
            # Give vLLM a head start before polling — early probes just race the
            # model load and show up as connection refused through the tunnel.
            if proc is not None:
                print(
                    f"Giving vLLM {STARTUP_GRACE_S}s to start before polling...",
                    flush=True,
                )
                time.sleep(STARTUP_GRACE_S)
            print(f"Waiting for server on localhost:{port}...", flush=True)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if server_is_up(url):
                    break
                if proc is not None and proc.poll() is not None:
                    print(
                        f"[error] serve process exited with code "
                        f"{proc.returncode} before the server came up — "
                        "aborting (see the serve output above)"
                    )
                    sys.exit(1)
                time.sleep(2)
            if not server_is_up(url):
                print("[error] server did not start within 600s — aborting")
                sys.exit(1)
            print("Server ready.")

            run_eval(port, args.subpop, args.eval_concurrency)
        except KeyboardInterrupt:
            print("\nInterrupted — shutting down server...")
        finally:
            if proc is not None:
                stop_server(proc)
    finally:
        # Never leave a rented instance running (unless the user asked to).
        if vast_alias and not args.keep_vast:
            teardown_vast()

    print()
    print("Done. Results: code/fine-tuning/output/evals/")


if __name__ == "__main__":
    main()
