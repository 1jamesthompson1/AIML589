#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""
prompted_eval.py — evaluate a base model with an added persona instruction.

Runs ``evaluate.py`` with a fixed instruction prepended to every system
prompt, writing the results to ``output/evals/<label>/`` (label defaults to
``<model short name>-prompted``). The prompted run then appears as a
"Prompted" row next to that model's base and fine-tuned rows in the
``analyze.py`` tables, showing how much of the fine-tuning effect can be
reproduced by prompting alone (the necessity-of-fine-tuning baseline).

The instruction is prepended, not appended, so each dataset system prompt
keeps its answer-format instruction (the trailing IMPORTANT block) at the
end. The per-system-prompt breakdown is preserved, so a single prompt
(e.g. ``no_persona``) can be compared with and without the addition.

Requires a running endpoint:
- ``--provider vllm`` (default): a vLLM server serving the base model,
  e.g. the server ``run_all.py`` uses for adapter evals. It does NOT start
  a server — bring one up first (``run.sh serve ...``) or reuse a running
  one.
- ``--provider openrouter``: the hosted API; needs ``OPENROUTER_API_KEY``
  in the repo ``.env`` (see ``.env.example``).

Usage:
    # vLLM server on the usual local port (EVAL_PORT, default 8087)
    uv run prompted_eval.py --model Qwen/Qwen3.8-27B

    # OpenRouter base model (label matches the OpenRouter base-model dir)
    uv run prompted_eval.py --provider openrouter \
        --model qwen/qwen3.8-27b --label qwen3.8-27b-prompted

    # Custom instruction / dataset subset, rerun finished configs
    uv run prompted_eval.py --model Qwen/Qwen3.8-27B \
        --prompt "Answer as a typical New Zealander would." \
        --datasets modal_response --force

Outputs land in ``output/evals/<label>/<dataset>-<subpopulation>-<timestamp>/``
(config.json + per_question_results.csv), exactly like every other eval run.
``--label`` must be ``<base model eval dir>-prompted`` for the row to slot
into that model's analysis tables (e.g. ``Qwen3.8-27B`` -> the existing
``output/evals/Qwen3.8-27B/`` base evals).
"""

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVALUATE = SCRIPT_DIR / "evaluate.py"

DEFAULT_MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_PROMPT = "Respond as an average New Zealand person would."
DEFAULT_DATASETS = ["modal_response", "first_token_distribution"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a base model with a prepended persona instruction "
        "as a prompted baseline for the fine-tuning tables.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Base model id as registered on the server (vLLM) or OpenRouter",
    )
    p.add_argument(
        "--label",
        default=None,
        help="Model-directory name under output/evals/ "
        "(default: <model short name>-prompted). Must be "
        "'<base eval dir>-prompted' to appear in that model's analysis tables.",
    )
    p.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Instruction prepended to every system prompt",
    )
    p.add_argument(
        "--provider",
        default="vllm",
        choices=["vllm", "openrouter"],
        help="Serving stack behind --api-url",
    )
    p.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("EVAL_PORT", "8087")),
        help="vLLM server port (used when --api-url is not set)",
    )
    p.add_argument(
        "--api-url",
        default=None,
        help="Override the endpoint URL (default: http://localhost:<port> "
        "for vLLM, https://openrouter.ai/api for OpenRouter)",
    )
    p.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset configs to evaluate",
    )
    p.add_argument(
        "--splits",
        default="train,validation",
        help="Comma-separated dataset splits to evaluate",
    )
    p.add_argument(
        "--subpopulation",
        default="overall",
        choices=["cluster_0", "cluster_1", "overall"],
        help="Population to evaluate against (the main tables use overall)",
    )
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per response (default: evaluate.py's 1000)",
    )
    p.add_argument("--concurrency", type=int, default=8, help="Requests in flight")
    p.add_argument("--max-retries", type=int, default=5)
    p.add_argument(
        "--force",
        action="store_true",
        help="Rerun dataset configs that already have a completed run",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without running anything",
    )
    return p.parse_args(argv)


def model_short_name(model: str) -> str:
    return model.split("/", 1)[-1]


def completed_runs(label_dir: Path) -> dict[tuple, set[str]]:
    """(dataset, subpopulation) -> splits covered by finished runs.

    Mirrors ``run_all.py``'s skip logic: a run only counts when it has a
    config and a non-empty results CSV, and is neither reasoning nor aborted.
    """
    done: dict[tuple, set[str]] = {}
    if not label_dir.is_dir():
        return done
    for run_dir in label_dir.iterdir():
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
        if config.get("aborted") or config.get("reasoning"):
            continue
        key = (config.get("dataset"), config.get("subpopulation"))
        done.setdefault(key, set()).update(config.get("splits") or [])
    return done


def build_command(args, dataset: str, label: str, api_url: str) -> list[str]:
    cmd = [
        "uv",
        "run",
        str(EVALUATE),
        "--provider",
        args.provider,
        "--api-url",
        api_url,
        "--model",
        args.model,
        "--dataset",
        dataset,
        "--splits",
        args.splits,
        "--subpopulation",
        args.subpopulation,
        "--system-prompt-prefix",
        args.prompt,
        "--output-label",
        label,
        "--temperature",
        str(args.temperature),
        "--concurrency",
        str(args.concurrency),
        "--max-retries",
        str(args.max_retries),
    ]
    if args.max_tokens is not None:
        cmd += ["--max-tokens", str(args.max_tokens)]
    if args.provider == "vllm":
        cmd += ["--api-key", "EMPTY"]
    return cmd


def main(argv=None):
    args = parse_args(argv)

    label = args.label or f"{model_short_name(args.model)}-prompted"
    if "/" in label or "\\" in label:
        raise SystemExit("--label must be a single directory name (no path separators)")

    api_url = args.api_url or (
        "https://openrouter.ai/api"
        if args.provider == "openrouter"
        else f"http://localhost:{args.port}"
    )
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    requested_splits = {s.strip() for s in args.splits.split(",") if s.strip()}
    label_dir = SCRIPT_DIR / "output" / "evals" / label
    done = completed_runs(label_dir)

    print("=" * 60)
    print("Prompted base-model evaluation")
    print("=" * 60)
    print(f"  model:    {args.model}")
    print(f"  provider: {args.provider}  ({api_url})")
    print(f"  label:    {label}  -> {label_dir}")
    print(f"  prompt:   {args.prompt!r}")
    print(f"  datasets: {', '.join(datasets)}")
    print("=" * 60)

    failures = []
    for dataset in datasets:
        key = (dataset, args.subpopulation)
        covered = done.get(key, set())
        if requested_splits and requested_splits.issubset(covered) and not args.force:
            print(
                f"[skip] {dataset}/{args.subpopulation}: already completed "
                f"({','.join(sorted(covered))}; use --force to rerun)"
            )
            continue
        cmd = build_command(args, dataset, label, api_url)
        print(f"\n[run] {dataset}")
        if args.dry_run:
            print("      " + shlex.join(cmd))
            continue
        ret = subprocess.run(cmd, cwd=SCRIPT_DIR).returncode
        if ret != 0:
            print(f"[FAIL] {dataset} (exit {ret})")
            failures.append(dataset)
        else:
            print(f"[done] {dataset}")

    if args.dry_run:
        return
    if failures:
        sys.exit(
            f"[error] {len(failures)} dataset config(s) failed: {', '.join(failures)}"
        )
    print(f"\nAll prompted evals complete under {label_dir}")


if __name__ == "__main__":
    main()
