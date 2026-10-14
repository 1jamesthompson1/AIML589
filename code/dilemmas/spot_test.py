"""Spot-test one dilemma: re-run it, replace its exported runs, print the picks.

The iteration loop for hand-editing a dilemma in ``profiles/``:

    uv run python code/dilemmas/spot_test.py cosmetic_clinic-0
    uv run python code/dilemmas/spot_test.py cosmetic_clinic_t8_filler_requests
    uv run python code/dilemmas/spot_test.py --list

The scenario may be a sample id (``<profile>-<index>``) or a dilemma id. The
script runs a fresh set of epochs (default 5) for each given model and replaces
only those models' exported runs for the scenario (their earlier runs are
cleared; every other model's runs stay in place), so the local website copy
shows the new results alongside the untouched ones. Other dilemmas are
untouched.

Models run in parallel; by default their inspect output is captured to a
temporary log file (path printed, tail shown on failure) so the console stays
quiet. Pass ``--verbose`` to stream it instead.

Only the models passed with ``--model`` (repeatable) are run; with no
``--model`` the default pair is used. The judge comes from ``--judge`` or
``JUDGE_MODEL`` in ``.env``. Pass ``--base-url`` to run against a vLLM server
(e.g. the fine-tuned Qwen3.8 adapters served from code/fine-tuning) with the
model as ``openai/<served-model-name>``; the judge role keeps its own endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

from export_results import export as export_runs
from export_results import slug

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT_DIR = HERE / "output"
LOG_DIR = OUTPUT_DIR / "logs"
RUNS_DIR = OUTPUT_DIR / "runs"

# Fallback only: used when no --model is passed; any --model replaces it.
# The spot-test set: the Qwen base model, two frontier comparison models, a
# fast Anthropic model and GLM. The fine-tuned Qwen adapters are served
# locally and run with --base-url and --model openai/<served-model-name>.
DEFAULT_MODELS = (
    "openrouter/qwen/qwen3.8-27b",
    "openrouter/deepseek/deepseek-v4.1-flash",
    "openrouter/openai/gpt-5.6-luna",
    "openrouter/anthropic/claude-haiku-5.5",
    "openrouter/z-ai/glm-5.3-flash",
)

# OpenRouter list prices, USD per million tokens, fetched from
# https://openrouter.ai/api/v1/models on 2026-10-14 (per-token prices x 1e6).
# Re-check when adding models or after price changes; unpriced models report
# tokens only. Prices change: the behavioural-similarity table in
# code/behavioural-simulations/export_results.py is a 2026-09-08 snapshot and
# GLM has since doubled.
MODEL_PRICES: dict[str, dict[str, float]] = {
    "openrouter/qwen/qwen3.8-27b": {
        "input": 0.04,
        "output": 2.30,
        "cache_read": 0.0198,
    },
    "openrouter/deepseek/deepseek-v4.1-flash": {
        "input": 0.02529,
        "output": 0.891,
        "cache_read": 0.0054,
    },
    "openrouter/openai/gpt-5.6-luna": {
        "input": 0.20,
        "output": 1.20,
        "cache_read": 0.02,
        "cache_write": 0.25,
    },
    "openrouter/anthropic/claude-haiku-5.5": {
        "input": 0.10,
        "output": 0.50,
        "cache_read": 0.01,
        "cache_write": 0.125,
    },
    "openrouter/z-ai/glm-5.3-flash": {
        "input": 0.15,
        "output": 0.50,
        "cache_read": 0.03,
    },
}
PRICE_UNIT = 1_000_000


def _price_for(model: str) -> dict[str, float] | None:
    """The MODEL_PRICES entry for a model (longest matching prefix)."""
    matches = [price for price in MODEL_PRICES if model.startswith(price)]
    return MODEL_PRICES[max(matches, key=len)] if matches else None


def run_cost(usage: dict) -> tuple[float, bool]:
    """USD cost of one run's exported usage block, and False when it holds a
    model with no price entry.

    The block covers the evaluated model and the judge (each run judges its
    own response), so both are billed. Cached input uses the cache rate when
    priced; cache writes default to 0 (most providers do not bill them).
    """
    total = 0.0
    priced = True
    for model, tokens in (usage or {}).items():
        rates = _price_for(model)
        if rates is None:
            priced = False
            continue
        input_rate = rates.get("input", 0.0)
        total += (
            (tokens.get("input_tokens") or 0) * input_rate
            + (tokens.get("input_tokens_cache_read") or 0)
            * rates.get("cache_read", input_rate)
            + (tokens.get("input_tokens_cache_write") or 0)
            * rates.get("cache_write", 0.0)
            + (tokens.get("output_tokens") or 0) * rates.get("output", 0.0)
        ) / PRICE_UNIT
    return total, priced


def scenario_lookup() -> dict[str, tuple[str, int, str, dict]]:
    """Map every sample id and dilemma id to (profile, index, id, dilemma)."""
    lookup: dict[str, tuple[str, int, str, dict]] = {}
    for path in sorted((HERE / "profiles").glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        for index, dilemma in enumerate(profile["dilemmas"]):
            entry = (profile["profile"], index, dilemma["id"], dilemma)
            lookup[f"{profile['profile']}-{index}"] = entry
            lookup[dilemma["id"]] = entry
    return lookup


def default_judge() -> str | None:
    """The judge model from the environment or the repo-root .env file."""
    if os.environ.get("JUDGE_MODEL"):
        return os.environ["JUDGE_MODEL"]
    env_path = REPO / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("JUDGE_MODEL="):
                return line.split("=", 1)[1].strip().strip("'\"")
    return None


def inspect_command() -> list[str]:
    candidate = Path(sys.executable).with_name("inspect")
    if candidate.exists():
        return [str(candidate)]
    if shutil.which("inspect"):
        return ["inspect"]
    return ["uv", "run", "inspect"]


def eval_command(
    model: str, judge: str, epochs: int, sample_id: str, base_url: str | None = None
) -> list[str]:
    """The inspect eval command for one model and sample.

    ``--model-base-url`` applies to the evaluated model only (inspect resolves
    the judge role separately), so a vLLM-served target can still be judged by
    a hosted model."""
    command = inspect_command() + [
        "eval",
        "code/dilemmas/task.py",
        "--model",
        model,
        "--model-role",
        f"judge={judge}",
        "--epochs",
        str(epochs),
        "--log-dir",
        "code/dilemmas/output/logs",
        "--sample-id",
        sample_id,
    ]
    if base_url:
        command += [
            "--model-base-url",
            base_url,
            "-M",
            "api_key=EMPTY",
            "-M",
            "responses_api=False",
        ]
    return command


def run_models(
    models: list[str],
    judge: str,
    epochs: int,
    sample_id: str,
    base_url: str | None = None,
    verbose: bool = False,
) -> None:
    """Run one epoch set per model in parallel, keeping the console quiet.

    Each model's inspect output is captured to a temporary log file unless
    ``verbose`` is set; the log path is printed so it can be tailed, and the
    last lines are shown if a run fails. A failure aborts before the exports
    are touched."""
    procs: list[tuple[str, subprocess.Popen, Path | None]] = []
    for model in models:
        command = eval_command(model, judge, epochs, sample_id, base_url)
        if verbose:
            print(f"\n>>> {' '.join(command)}\n", flush=True)
            procs.append((model, subprocess.Popen(command, cwd=REPO), None))
        else:
            log = tempfile.NamedTemporaryFile(
                prefix="spot-", suffix=".log", delete=False, mode="w"
            )
            path = Path(log.name)
            proc = subprocess.Popen(
                command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT
            )
            log.close()
            procs.append((model, proc, path))
            print(f"  started {model} (log: {path})")

    started = time.monotonic()
    failures = []
    for model, proc, path in procs:
        code = proc.wait()
        short = model.split("/")[-1]
        elapsed = time.monotonic() - started
        print(
            f"  {short:34s} {'ok' if code == 0 else f'FAILED (exit {code})'}"
            f"  ({elapsed:.0f}s)"
        )
        if code != 0:
            failures.append((model, path))

    if failures:
        for model, path in failures:
            print(f"\n--- tail of {path} ---")
            lines = path.read_text(errors="replace").splitlines() if path else []
            print("\n".join(lines[-25:]))
        raise SystemExit(
            "spot test aborted: a model run failed; exports left untouched"
        )


def clear_exported(profile_id: str, dilemma_id: str, models: list[str]) -> list[Path]:
    """Remove the exported run directories for the scenario and these models."""
    directory = slug(f"{profile_id}-{dilemma_id}")
    removed = []
    for model in models:
        path = RUNS_DIR / slug(model) / directory
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(path)
    return removed


def stability(picks: list[int], option_count: int) -> float:
    """1 - H / ln(k) over the picked option indices (1.00 = all the same)."""
    counts = Counter(picks)
    proportions = [n / len(picks) for n in counts.values()]
    entropy = -sum(p * math.log(p) for p in proportions)
    return 1 - entropy / math.log(option_count)


def summarise(dilemma_id: str, options: list[str], rerun_models: set[str]) -> None:
    """Print the picks per run, counts, stability, cost and cross-model divergence.

    Models in ``rerun_models`` were run by this invocation; the rest are
    earlier exported runs for the scenario that were kept. Stability is
    1 - H / ln(k) over the picks (1.00 = every run made the same choice): per
    model, then pooled across the runs this spot test made. Costs are the
    exported token usage priced with MODEL_PRICES, target and judge together.
    """
    manifest = json.loads((RUNS_DIR / "index.json").read_text(encoding="utf-8"))
    rows = [row for row in manifest["runs"] if row["situation_id"] == dilemma_id]
    by_model: dict[str, list[tuple[int, int | None]]] = {}
    costs: dict[str, float] = {}
    unpriced: set[str] = set()
    for row in rows:
        run_dir = RUNS_DIR / row["path"]
        judge = json.loads((run_dir / "judge.json").read_text(encoding="utf-8"))
        choice = (judge.get("metadata") or {}).get("choice") or judge.get("answer")
        index = options.index(choice) + 1 if choice in options else None
        by_model.setdefault(row["model"], []).append((row["version"], index))
        usage = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        cost, priced = run_cost(usage.get("usage") or {})
        costs[row["model"]] = costs.get(row["model"], 0.0) + cost
        if not priced:
            unpriced.add(row["model"])

    print(f"\nResults for {dilemma_id} ({len(rows)} exported runs)")
    distributions: dict[str, Counter] = {}
    for model, runs in sorted(by_model.items()):
        picks = [index for _, index in sorted(runs)]
        ran = model in rerun_models
        name = model.split("/")[-1] + (" (ran)" if ran else " (kept)")
        if any(index is None for index in picks):
            print(f"  {name:30s} picks {picks} (missing judge result)")
            continue
        counts = Counter(picks)
        rendered = ", ".join(f"{index} x{n}" for index, n in sorted(counts.items()))
        cost = f"${costs.get(model, 0.0):.4f}" if ran else "kept (no new spend)"
        print(
            f"  {name:30s} picks {picks}  |  {rendered}"
            f"  |  stability {stability(picks, len(options)):.2f}"
            f"  |  cost {cost}"
        )
        distributions[model] = counts

    pooled = [
        index
        for model in sorted(rerun_models)
        for _, index in sorted(by_model.get(model, []))
        if index is not None
    ]
    if pooled:
        total_cost = sum(costs.get(model, 0.0) for model in rerun_models)
        warning = (
            " (some models are unpriced: tokens only, cost is a lower bound)"
            if unpriced & rerun_models
            else ""
        )
        print(
            f"  overall stability (all {len(pooled)} runs this spot test made): "
            f"{stability(pooled, len(options)):.2f}"
        )
        print(f"  spot-test cost: ${total_cost:.4f}{warning}")

    models = list(distributions)
    if len(models) >= 2:
        tvds = []
        for position, left in enumerate(models):
            for right in models[position + 1 :]:
                left_total = sum(distributions[left].values())
                right_total = sum(distributions[right].values())
                tvds.append(
                    0.5
                    * sum(
                        abs(
                            distributions[left][index] / left_total
                            - distributions[right][index] / right_total
                        )
                        for index in range(1, len(options) + 1)
                    )
                )
        print(f"  cross-model divergence (TVD): {sum(tvds) / len(tvds):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        nargs="?",
        help="Sample id (e.g. cosmetic_clinic-0) or dilemma id",
    )
    parser.add_argument("--model", action="append", dest="models", default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--judge", default=None)
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL of a vLLM server (e.g. http://localhost:8087/v1); pass "
        "served models as openai/<name>. The judge keeps its own endpoint.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream each model's inspect output to the console instead of "
        "capturing it to a temporary log file.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true", help="List every scenario id")
    args = parser.parse_args()

    lookup = scenario_lookup()
    if args.list:
        for key in sorted(k for k in lookup if k.rsplit("-", 1)[-1].isdigit()):
            print(f"{key:34s} {lookup[key][2]}")
        return
    if not args.scenario:
        parser.error("give a scenario id, or use --list")
    if args.scenario not in lookup:
        raise SystemExit(
            f"unknown scenario {args.scenario!r}: use a sample id or a dilemma id "
            "(see --list)"
        )

    profile_id, index, dilemma_id, dilemma = lookup[args.scenario]
    sample_id = f"{profile_id}-{index}"
    judge = args.judge or default_judge()
    if not judge:
        raise SystemExit("no judge model: set JUDGE_MODEL in .env or pass --judge")
    if args.models:
        models = list(args.models)
    else:
        models = list(DEFAULT_MODELS)
        print(f"No --model given; using the default pair: {', '.join(models)}")

    print(f"Scenario: {sample_id} ({dilemma_id})")
    print(f"Question: {dilemma['question']}")
    for number, option in enumerate(dilemma["answer_options"], 1):
        print(f"  {number}. {option}")
    print(f"Models: {', '.join(models)} | epochs: {args.epochs} | judge: {judge}")

    if args.dry_run:
        for model in models:
            print(
                ">>> "
                + " ".join(
                    eval_command(model, judge, args.epochs, sample_id, args.base_url)
                )
            )
        print("dry run: no evaluation, no export")
        return

    run_models(models, judge, args.epochs, sample_id, args.base_url, args.verbose)

    for path in clear_exported(profile_id, dilemma_id, models):
        print(f"removed old export {path.relative_to(REPO)}")
    # Replace only the re-run models' runs; the other models' exported runs
    # for the scenario stay in place. (Pass --reset-model to export_results.py
    # to drop them instead.)
    count = asyncio.run(
        export_runs(LOG_DIR, RUNS_DIR, reset_samples=frozenset({sample_id}))
    )
    print(f"re-exported {count} runs")

    summarise(dilemma_id, dilemma["answer_options"], set(models))


if __name__ == "__main__":
    main()
