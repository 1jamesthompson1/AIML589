#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""
Batch-run all missing evaluations for models served on a vLLM server.

Queries the server for available models, checks which evaluation runs
(model × dataset × reasoning) are already complete in
output/evals/, and runs the missing ones in parallel.

Usage:
    uv run code/fine-tuning/batch_eval.py --port 8000
    uv run code/fine-tuning/batch_eval.py --port 8000 --model Qwen/Qwen3.6-27B
    uv run code/fine-tuning/batch_eval.py --api-url http://localhost:8000 --concurrency 3
    uv run code/fine-tuning/batch_eval.py --port 8000 --dry-run
"""

import argparse
import json
import re
import select
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

DATASETS = ["single_modal", "single_sample"]
# Each eval run covers all subpopulations (cluster_0, cluster_1, overall)
# in a single pass — the per_question_results.csv includes a subpopulation
# column for downstream filtering.
REASONING_MODES = [False, True]

SCRIPT_DIR = Path(__file__).resolve().parent
EVALS_DIR = SCRIPT_DIR / "output" / "evals"

# Observed per-eval durations on RTX 6000 Pro Blackwell (Qwen3.6-27B):
#   no_reasoning: ~30s  (mainly prompt processing + API round-trip)
#   reasoning:    ~30m = 1800s  (long CoT generation)
TIME_NO_REASONING = 30
TIME_REASONING = 1800


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Batch-run missing evaluations for models on a vLLM server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", type=int, default=8000, help="vLLM server port")
    p.add_argument("--api-url", default=None, help="Full API URL (overrides --port)")
    p.add_argument("--api-key", default="EMPTY", help="API key for the vLLM server")
    p.add_argument("--model", default=None, help="Only evaluate this specific model")
    p.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of evaluations to run in parallel (default: auto-detect from server)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List missing evaluations without running them",
    )
    p.add_argument("--hf-token", default=None, help="HF token for dataset loading")
    return p.parse_args(argv)


def get_available_models(api_url: str, api_key: str) -> list[dict]:
    url = f"{api_url}/v1/models"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode())
    return data.get("data", [])


def get_server_metrics(api_url: str) -> dict:
    """Fetch vLLM metrics and extract KV cache / LoRA config."""
    url = f"{api_url}/metrics"
    with urllib.request.urlopen(url, timeout=15) as resp:
        text = resp.read().decode()

    info = {}

    m = re.search(
        r'vllm:cache_config_info\{[^}]*?kv_cache_max_concurrency="([^"]+)"',
        text,
    )
    if m:
        info["kv_cache_max_concurrency"] = float(m.group(1))

    m = re.search(
        r'vllm:cache_config_info\{[^}]*?gpu_memory_utilization="([^"]+)"', text
    )
    if m:
        info["gpu_memory_utilization"] = float(m.group(1))

    m = re.search(r'vllm:cache_config_info\{[^}]*?num_gpu_blocks="([^"]+)"', text)
    if m:
        info["num_gpu_blocks"] = int(m.group(1))

    m = re.search(r'vllm:cache_config_info\{[^}]*?block_size="([^"]+)"', text)
    if m:
        info["block_size"] = int(m.group(1))

    # Count distinct LoRA adapters from lora_requests_info
    adapters = set()
    for line in text.splitlines():
        if "vllm:lora_requests_info{" in line and "running_lora_adapters" in line:
            m2 = re.search(r'running_lora_adapters="([^"]*)"', line)
            if m2 and m2.group(1):
                adapters.add(m2.group(1))
    info["lora_adapters"] = sorted(adapters)

    return info


def model_short_name(model: str) -> str:
    return model.split("/", 1)[-1]


def existing_completed_runs(model_short: str) -> set[tuple[str, bool]]:
    """Return set of (dataset, reasoning) tuples for runs that cover all subpops."""
    completed = set()
    model_dir = EVALS_DIR / model_short
    if not model_dir.exists():
        return completed
    for run_dir in model_dir.iterdir():
        if not run_dir.is_dir():
            continue
        config_path = run_dir / "config.json"
        results_path = run_dir / "per_question_results.csv"
        if not config_path.exists() or not results_path.exists():
            continue
        if results_path.stat().st_size == 0:
            continue
        try:
            config = json.loads(config_path.read_text())
        except Exception:
            continue
        completed.add(
            (
                config.get("dataset"),
                config.get("reasoning"),
            )
        )
    return completed


def _fmt_duration(seconds: int) -> str:
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _parallel_estimate(n_reasoning: int, n_no_reasoning: int, concurrency: int) -> int:
    reasoning_batches = (n_reasoning + concurrency - 1) // concurrency
    no_reasoning_batches = (n_no_reasoning + concurrency - 1) // concurrency
    return reasoning_batches * TIME_REASONING + no_reasoning_batches * TIME_NO_REASONING


def main():
    args = parse_args()
    api_url = args.api_url or f"http://localhost:{args.port}"
    print(f"[batch] API URL: {api_url}")

    try:
        models_data = get_available_models(api_url, args.api_key)
    except Exception as e:
        print(f"[error] could not reach server at {api_url}: {e}")
        sys.exit(1)

    if not models_data:
        print("[error] no models returned by server")
        sys.exit(1)

    model_ids = [m["id"] for m in models_data]

    if args.model:
        if args.model not in model_ids:
            print(f"[error] model '{args.model}' not found on server")
            print(f"  available models: {', '.join(model_ids)}")
            sys.exit(1)
        model_ids = [args.model]

    # Fetch server capacity metrics
    try:
        srv = get_server_metrics(api_url)
    except Exception:
        srv = {}

    kv_conc = srv.get("kv_cache_max_concurrency")
    has_loras = len(srv.get("lora_adapters", [])) > 0

    print(f"[batch] models on server ({len(model_ids)}):")
    for m in model_ids:
        print(f"         - {m}")

    evaluate_script = Path(__file__).resolve().parent / "evaluate.py"
    if not evaluate_script.exists():
        print(f"[error] evaluate.py not found at {evaluate_script}")
        sys.exit(1)

    jobs = []
    for model in model_ids:
        short = model_short_name(model)
        completed = existing_completed_runs(short)
        for ds in DATASETS:
            for reasoning in REASONING_MODES:
                if (ds, reasoning) in completed:
                    continue
                jobs.append((model, ds, reasoning))

    if not jobs:
        print("\n[batch] all evaluations complete — nothing to run.")
        return

    print(
        f"\n[batch] {len(jobs)} evaluation(s) needed across {len(model_ids)} model(s):\n"
    )

    for model in model_ids:
        short = model_short_name(model)
        model_jobs = [(ds, r) for (m, ds, r) in jobs if m == model]
        n_reasoning = sum(1 for _, r in model_jobs if r)
        n_no_reasoning = len(model_jobs) - n_reasoning
        total_s = n_reasoning * TIME_REASONING + n_no_reasoning * TIME_NO_REASONING
        print(f"  {short}")
        print(
            f"    missing: {n_reasoning} reasoning + {n_no_reasoning} no_reasoning = {len(model_jobs)} total"
        )
        print(f"    est. sequential time: {_fmt_duration(total_s)}")
        print()

    total_reasoning = sum(1 for _, _, r in jobs if r)
    total_no_reasoning = len(jobs) - total_reasoning

    print(
        f"  Totals: {total_reasoning} reasoning + {total_no_reasoning} no_reasoning = {len(jobs)} jobs"
    )
    print()

    # Server capacity section
    if kv_conc is not None:
        optimal = min(8, int(kv_conc))
        print(f"  Server KV cache capacity: {kv_conc:.1f} concurrent sequences")
        print(f"  Recommended --concurrency: {optimal}")
        print("    (higher concurrency risks queueing on KV cache)")
        print("    (non-reasoning jobs are ~30s so they squeeze through easily)")
        if has_loras:
            print(f"  LoRA adapters: {len(srv.get('lora_adapters', []))}")
            print("    Requests to different LoRA adapters cannot be batched together")
            print(
                "    by vLLM, so concurrency benefits only apply within each adapter."
            )
            for a in srv.get("lora_adapters", []):
                n = sum(1 for m, _, _ in jobs if m == a)
                print(f"      {a}: {n} job(s)")
    else:
        optimal = 2
        print("  (could not query server KV cache capacity, defaulting to 2)")

    print()
    print("  Estimated wall time by concurrency (assumes same-adapter batching):")
    for c in [1, 2, optimal, optimal + 2]:
        if c < 1:
            continue
        est = _fmt_duration(_parallel_estimate(total_reasoning, total_no_reasoning, c))
        marker = " ← recommended" if c == optimal else ""
        print(f"    --concurrency {c}: ~{est}{marker}")
    print()

    if args.dry_run:
        return

    concurrency = args.concurrency if args.concurrency is not None else optimal
    t0 = time.monotonic()
    print(f"\n[batch] running with concurrency={concurrency}...\n")

    def run_eval(model, ds, reasoning):
        cmd = [
            "uv",
            "run",
            str(evaluate_script),
            "--api-url",
            api_url,
            "--api-key",
            args.api_key,
            "--model",
            model,
            "--dataset",
            ds,
            "--no-plots",
        ]
        if reasoning:
            cmd.append("--reasoning")
        if args.hf_token:
            cmd += ["--hf-token", args.hf_token]

        label = f"{model_short_name(model)}  {ds}  {'reasoning' if reasoning else 'no_reasoning'}"
        print(f"  [start] {label}", flush=True)

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=SCRIPT_DIR,
        )

        deadline = time.monotonic() + 10800
        out_lines = []
        while proc.poll() is None:
            r, _, _ = select.select([proc.stdout], [], [], 5)
            if r:
                line = proc.stdout.readline()
                if line:
                    out_lines.append(line)
                    print(f"  [{label}] {line}", end="", flush=True)
            elif time.monotonic() > deadline:
                proc.kill()
                proc.wait()
                print(f"  [TIMEOUT] {label}")
                return 1
        for line in proc.stdout:
            out_lines.append(line)
            print(f"  [{label}] {line}", end="", flush=True)
        proc.stdout.close()
        stderr_text = proc.stderr.read()
        proc.stderr.close()
        ret = proc.returncode

        if ret != 0:
            print(f"  [FAIL]  {label}")
            for line in stderr_text.strip().splitlines():
                print(f"           {line}")
        else:
            print(f"  [done]  {label}")
        return ret

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(run_eval, *j): j for j in jobs}
        for future in as_completed(futures):
            future.result()

    failed = sum(1 for f in futures if f.exception() is not None or f.result() != 0)
    elapsed = time.monotonic() - t0
    if failed:
        print(
            f"\n[batch] completed in {_fmt_duration(int(elapsed))} with {failed} failure(s)"
        )
    else:
        print(
            f"\n[batch] all {len(jobs)} evaluations completed in {_fmt_duration(int(elapsed))}."
        )


if __name__ == "__main__":
    main()
