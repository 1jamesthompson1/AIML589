#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "inspect_ai>=0.3.255",
#   "inspect_evals[swe_bench]",
#   "inspect_harbor",
# ]
# ///
"""
Run capability evals for a single served model against a vLLM server.

Tasks (all from existing packages, nothing hand-rolled):
  mmlu_pro     - inspect_evals.mmlu_pro (official)
  gpqa_diamond - inspect_harbor.gpqa_diamond (Meridian Labs)
  swe_bench    - inspect_evals, defaults to SWE-bench Verified Mini
                 (MariusHobbhahn/swe-bench-verified-mini, 50 public
                 instances, no gated access) with only the first 5.
                 Use --swe-dataset to switch:
                   verified_mini  - MariusHobbhahn/swe-bench-verified-mini
                                    (50, public)
                   verified       - princeton-nlp/SWE-bench_Verified (500, gated)
                   mini           - princeton-nlp/SWE-bench_Mini (300, gated)
                 SWE-bench runs in a docker sandbox (ghcr.io/epoch-research
                 images, ~5-15 GB each, cached after first pull).

Each task runs on a deterministic fixed subset (sample_shuffle=<seed> +
max_samples) so the base model and every adapter see the SAME items.
Results are stored in output/capability/<model>/<run>/ (summary.json,
config.json, plus the raw inspect logs).

Usage (against the vLLM server from serve.py / run.sh serve):
    uv run capability_eval.py --port 8000 --model Qwen/Qwen3.6-27B
    uv run capability_eval.py --port 8000 --model Qwen/Qwen3.6-27B \
        --tasks mmlu_pro,swe_bench --max-samples 10

Resume a failed/interrupted run with --resume (or --resume <run dir>):
this calls eval_retry(), which re-runs only the samples that errored or
were interrupted and reuses the completed ones from the previous log, so
results stay on the same deterministic subset. Retried tasks reuse the
original run's vLLM connection (port/api url is taken from the log); a
task that never ran in the previous run is started fresh using the
current --port/--api-url settings. Note: sandboxed tasks (gpqa_diamond,
swe_bench) cannot be retried from a log in inspect_ai (sandbox providers
lack config_deserialize), so their logs are re-run fresh instead.
    uv run capability_eval.py --model Qwen/Qwen3.6-27B --resume
    uv run capability_eval.py --model Qwen/Qwen3.6-27B \
        --resume output/capability/Qwen3.6-27B/capability-20260813_105838
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai import eval_retry
from inspect_ai.log import read_eval_log
from inspect_evals.mmlu_pro import mmlu_pro
from inspect_harbor import gpqa_diamond

# Default fixed-subset sizes (override with --max-samples).
TASK_DEFAULTS = {
    "mmlu_pro": 300,
    "gpqa_diamond": 100,
    "swe_bench": 5,
}

TASKS = ["mmlu_pro", "gpqa_diamond", "swe_bench"]

SWE_DATASETS = {
    "verified_mini": None,  # swe_bench_verified_mini() (public, 50)
    "mini": {"dataset": "princeton-nlp/SWE-bench_Mini", "revision": None},
    "verified": None,  # swe_bench() default (gated, pinned revision)
}


def build_task(name, swe_dataset):
    """Instantiate one eval task.

    mmlu_pro: task-level shuffle is disabled so eval(sample_shuffle=seed,
    max_samples=N) selects the SAME deterministic subset for every model.
    gpqa_diamond: 'local' sandbox (no docker needed; MCQ only).
    swe_bench: docker sandbox; dataset chosen by --swe-dataset.
    """
    if name == "mmlu_pro":
        return mmlu_pro(shuffle=True, fewshot=0)
    if name == "gpqa_diamond":
        return gpqa_diamond(sandbox_env_name="local")
    if name == "swe_bench":
        from inspect_evals.swe_bench import swe_bench, swe_bench_verified_mini

        if swe_dataset == "verified_mini":
            return swe_bench_verified_mini()
        return swe_bench(**SWE_DATASETS[swe_dataset])
    raise SystemExit(f"unknown task: {name} (choose from {', '.join(TASKS)})")


def latest_run_dir(model_name):
    """Most recently modified run dir for a model under output/capability/."""
    model_dir = Path("output") / "capability" / model_name.split("/", 1)[-1]
    if not model_dir.is_dir():
        raise SystemExit(f"no previous runs under {model_dir}")
    runs = sorted(
        (d for d in model_dir.iterdir() if d.is_dir()),
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        raise SystemExit(f"no previous runs under {model_dir}")
    return runs[0]


def find_task_log(run_dir, task):
    """Most recent inspect log for a task inside a run dir (best match by name).

    Returns None (with a warning) if the task never ran in this run dir.
    """
    logs_dir = run_dir / "inspect-logs"
    if not logs_dir.is_dir():
        print(f"  [warn] no inspect-logs dir in {run_dir}")
        return None
    tag = task.replace("_", "-")  # log file names use e.g. mmlu-pro
    matches = [
        p
        for p in logs_dir.iterdir()
        if p.suffix in (".json", ".eval") and tag in p.name
    ]
    if not matches:
        print(f"  [warn] no log for task {task} in {logs_dir}")
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def metric_value(metric):
    """Metric results are floats, or EvalMetric (dict or pydantic model with a
    'value' field). Values read back from a log file are pydantic EvalMetric
    objects, so unwrap them before rounding.
    """
    if isinstance(metric, dict):
        return metric.get("value")
    value = getattr(metric, "value", None)
    return value if value is not None else metric


def summarise(log):
    """Extract a small summary dict from an inspect EvalLog."""
    task_name = log.eval.task if log.eval else "unknown"
    if not log.results or not log.results.scores:
        return {
            "task": task_name,
            "error": log.error.message if log.error else "no results",
        }
    score = log.results.scores[0]
    metrics = score.metrics or {}
    # inspect_ai >=0.4 renamed EvalScore.sample_results -> EvalScore.samples
    samples = (
        getattr(score, "samples", None) or getattr(score, "sample_results", None) or []
    )
    n = len(samples)
    acc = metric_value(metrics.get("accuracy"))
    stderr = metric_value(metrics.get("stderr"))
    return {
        "task": task_name,
        "n": n,
        "accuracy": round(acc, 4) if isinstance(acc, (int, float)) else acc,
        "stderr": round(stderr, 4) if isinstance(stderr, (int, float)) else stderr,
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Capability evals (mmlu_pro, gpqa_diamond, swe_bench) "
        "against a served model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--api-url",
        default=None,
        help="vLLM server URL (default: http://localhost:<port>)",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="vLLM server port (used when --api-url is not set)",
    )
    p.add_argument("--api-key", default="EMPTY", help="API key for the vLLM server")
    p.add_argument(
        "--model",
        default="Qwen/Qwen3.6-27B",
        help="Model name as registered on the server",
    )
    p.add_argument(
        "--tasks", nargs="+", default=TASKS, choices=TASKS, help="Tasks to run"
    )
    p.add_argument(
        "--swe-dataset",
        default="verified_mini",
        choices=list(SWE_DATASETS),
        help="SWE-bench dataset: verified_mini (MariusHobbhahn 50, public), "
        "verified (SWE-bench_Verified, 500, gated), mini (SWE-bench_Mini, "
        "300, gated)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per task (default: per-task default, "
        f"{', '.join(f'{k}={v}' for k, v in TASK_DEFAULTS.items())})",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Seed for the deterministic subset"
    )
    p.add_argument(
        "--display",
        default="plain",
        choices=["full", "conversation", "rich", "plain", "log", "none"],
        help="inspect_ai display: plain = per-sample progress lines, "
        "rich/full = live progress bars and tables, none = silent",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="inspect_ai console log level (DEBUG is most verbose)",
    )
    p.add_argument(
        "--log-realtime",
        action="store_true",
        help="Stream eval logs to the console in realtime as samples run",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Results dir (default: output/capability/<model>/<run>)",
    )
    p.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        metavar="RUN_DIR",
        help="Resume a previous run: re-run only errored/interrupted samples, "
        "reusing completed ones (eval_retry). Bare --resume picks the most "
        "recent run for this model; or pass a run dir explicitly. Connection "
        "settings come from the original run's log, so --port is ignored.",
    )
    return p.parse_args(argv)


def run_fresh(task, args, output_dir, model, model_args):
    """Run a task from scratch with the deterministic subset settings."""
    limit = args.max_samples or TASK_DEFAULTS[task]
    return inspect_eval(
        tasks=[build_task(task, args.swe_dataset)],
        model=model,
        model_base_url=f"{args.api_url}/v1",
        model_args=model_args,
        limit=limit,
        max_samples=16,
        sample_shuffle=args.seed,
        epochs=1,
        retry_on_error=3,
        max_retries=2,
        display=args.display,
        log_level=args.log_level,
        log_realtime=args.log_realtime,
        log_dir=str(output_dir / "inspect-logs"),
        log_format="json",
    )


def main():
    args = parse_args()

    if args.api_url is None:
        args.api_url = f"http://localhost:{args.port}"
    model = f"openai/{args.model}"

    # responses_api=False: inspect_ai treats any unrecognized model name as an
    # OpenAI "latest" model, which flips on the Responses API (incl. token
    # counting via /responses/input_tokens). vLLM doesn't implement that
    # endpoint, so force the chat-completions path for self-hosted servers.
    model_args = {"api_key": args.api_key, "responses_api": False}

    print("=" * 60)
    print("Capability evaluation configuration")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")
    print(f"  model spec: {model}")
    print("=" * 60)

    resume_dir = None
    if args.resume:
        resume_dir = (
            Path(args.resume) if args.resume != "latest" else latest_run_dir(args.model)
        )
        print(f"[resume] continuing run: {resume_dir}")
        print(
            "[resume] note: retried tasks reuse the original run's connection settings; "
            "tasks that never ran are started fresh with the current flags"
        )
        output_dir = resume_dir
    else:
        run_name = f"capability-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        if args.output_dir is None:
            output_dir = (
                Path("output") / "capability" / args.model.split("/", 1)[-1] / run_name
            )
        else:
            output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "target": args.model,
        "tasks": list(args.tasks),
        "max_samples": args.max_samples,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(),
        "api_url": args.api_url,
        "display": args.display,
        "log_level": args.log_level,
        "log_realtime": args.log_realtime,
        "note": "Fixed deterministic subsets; run base and each adapter with "
        "identical flags for a fair pre/post comparison.",
    }
    config_path = output_dir / "config.json"
    if resume_dir is not None and config_path.exists():
        # keep the original run's config and stamp the resume onto it
        prev = json.loads(config_path.read_text())
        prev["resume"] = {"at": datetime.now().isoformat(), "tasks": list(args.tasks)}
        config = prev
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[save] config -> {config_path}")

    summaries = {}
    for task in args.tasks:
        print(f"\n=== {task} ===")
        try:
            if resume_dir is not None:
                log_path = find_task_log(resume_dir, task)
                if log_path is None:
                    # task never ran in this run dir - run it fresh with the
                    # current connection settings
                    print(f"  {task}: no previous log; running fresh")
                    logs = run_fresh(task, args, output_dir, model, model_args)
                else:
                    log = read_eval_log(log_path)
                    if log.status == "success":
                        print(f"  {task}: already successful, nothing to retry")
                        summaries[task] = summarise(log)
                        continue
                    print(f"  {task}: retrying from {log_path.name}")
                    try:
                        logs = eval_retry(
                            tasks=str(log_path),
                            max_samples=16,
                            display=args.display,
                            log_level=args.log_level,
                            log_realtime=args.log_realtime,
                            log_dir=str(output_dir / "inspect-logs"),
                            log_format="json",
                        )
                    except Exception as e:
                        # inspect_ai cannot reconstruct sandboxed tasks from a
                        # log (sandbox providers lack config_deserialize), so
                        # retries fail for gpqa_diamond ("local" sandbox) and
                        # swe_bench ("docker" sandbox). Fall back to a fresh
                        # run of the task.
                        if "config_deserialize" in str(e):
                            print(
                                f"  [warn] cannot retry {log_path.name} "
                                f"({e}); running fresh"
                            )
                            logs = run_fresh(task, args, output_dir, model, model_args)
                        else:
                            raise
            else:
                logs = run_fresh(task, args, output_dir, model, model_args)
        except Exception as e:
            print(f"[error] {task} failed: {e}")
            summaries[task] = {"task": task, "error": str(e)}
            continue
        summaries[task] = (
            summarise(logs[0]) if logs else {"task": task, "error": "no log"}
        )
        s = summaries[task]
        if "error" not in s:
            acc = s["accuracy"]
            acc_s = f"{acc * 100:.1f}%" if acc is not None else "n/a"
            print(f"  {task}: n={s['n']}, accuracy={acc_s}")
        else:
            print(f"  {task}: ERROR {s['error']}")

    (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"[save] summary -> {output_dir / 'summary.json'}")

    print("\n" + "=" * 60)
    print("Capability summary")
    print("=" * 60)
    for task, s in summaries.items():
        if "error" in s:
            print(f"  {task:<14} ERROR: {s['error']}")
        else:
            acc = s["accuracy"]
            acc_s = f"{acc * 100:.1f}%" if acc is not None else "n/a"
            print(f"  {task:<14} n={s['n']:<5} accuracy={acc_s}")
    print("=" * 60)
    print(f"[done] results saved to {output_dir.resolve()}")
    print(
        "Compare base vs adapters by re-running with --model and diffing summary.json"
    )


if __name__ == "__main__":
    main()
