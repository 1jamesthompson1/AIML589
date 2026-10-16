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

When no explicitly local model is requested, a vLLM server on
``--vllm-port`` (default 8087, from ``EVAL_PORT``) is probed; if it answers,
every served adapter is auto-added as an ``openai/<name>`` target alongside
the other models. For those adapters an extra **first-token probe** runs
after the epochs: one plain chat request (``max_tokens=1``, top-20
logprobs, ``top_k`` and ``chat_template_kwargs`` matching the fine-tuning
eval) whose next-token distribution is printed raw per model — the
"as a regular model" preference signal (the dilemma prompts do not
number their options, so no option mapping is forced).
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
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from export_results import export as export_runs
from export_results import slug

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT_DIR = HERE / "output"
LOG_DIR = OUTPUT_DIR / "logs"
RUNS_DIR = OUTPUT_DIR / "runs"
FIRST_TOKEN_PATH = OUTPUT_DIR / "first_token.json"

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
    """Map every sample id and dilemma id to (profile, index, id, dilemma).

    The dilemma carries its profile's ``system_prompt`` (as the Inspect task
    attaches it per sample) — the first-token probe reads it from here.
    """
    lookup: dict[str, tuple[str, int, str, dict]] = {}
    for path in sorted((HERE / "profiles").glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        for index, dilemma in enumerate(profile["dilemmas"]):
            dilemma = {**dilemma, "system_prompt": profile["system_prompt"]}
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


# vLLM eval server: EVAL_PORT matches run_all.py's tunnel port default.
VLLM_PORT = int(os.environ.get("EVAL_PORT", "8087"))


def vllm_served_models(port: int) -> list[str]:
    """The model ids a vLLM server serves on the port ([] if it is not up)."""
    try:
        with urllib.request.urlopen(
            f"http://localhost:{port}/v1/models", timeout=2
        ) as resp:
            payload = json.loads(resp.read())
        return [item["id"] for item in payload.get("data", []) if item.get("id")]
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        return []


def first_token_probe(
    model: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
) -> dict | None:
    """One-token chat completion: the model's natural first-token distribution.

    Treats the adapter exactly as a regular model — a plain
    ``[system, user]`` chat request asking for the next token, no option
    list wrangling. Note the dilemma prompts do not number their options,
    so the distribution is over raw first tokens (the prose opening), not
    option indices. ``chat_template_kwargs``/``top_k`` match evaluate.py so
    the request reaches the adapter in its fine-tuned prompt format
    (THINKING_ENABLED=False ⇒ closed empty think block). Returns a small
    summary dict, or None when the probe fails (never fatal).
    """
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 20,
                "logprobs": True,
                "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }
        ).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer EMPTY",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as resp:
            payload = json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError) as error:
        print(f"  first-token probe failed for {model}: {error}")
        return None
    choice = (payload.get("choices") or [{}])[0]
    content = (choice.get("logprobs") or {}).get("content") or []
    raw = (content[0].get("top_logprobs") or [])[:20] if content else []
    if not raw:
        return None
    return {
        "first_token": raw[0].get("token"),
        "top_tokens": [
            {
                "token": entry.get("token"),
                "logprob": entry.get("logprob"),
                "p": round(math.exp(entry.get("logprob") or 0.0), 4),
            }
            for entry in raw
        ],
    }


def run_first_token_probes(
    local_models: list[str], base_url: str, dilemma: dict, sample_id: str
) -> dict[str, dict]:
    """Probe every vLLM-served model once and store the distributions.

    Results go to ``output/first_token.json`` under the scenario key (the
    same ``<profile>-<dilemma>`` id the run exporter uses) so spot tests
    overwrite rather than accumulate. The dilemma prompts do not number
    their options, so the top tokens are printed raw (the human maps them
    onto the options printed above, or the analysed prose prefix).
    """
    results: dict[str, dict] = {}
    prompt = f"{dilemma['situation']}\n\n{dilemma['question']}"
    for model in local_models:
        # The chat API wants the bare served id; ``openai/<name>`` is inspect
        # routing sugar — a 404 otherwise (the provider prefix is not part of
        # the served model name).
        served = model.removeprefix("openai/")
        probe = first_token_probe(served, base_url, dilemma["system_prompt"], prompt)
        if not probe:
            print(f"  {model.split('/')[-1]:34s} first-token probe unavailable")
            continue
        rendered = [
            f"'{entry['token']}' {entry['p']:.2f}" for entry in probe["top_tokens"][:6]
        ]
        print(f"  {model.split('/')[-1]:34s} first token: {', '.join(rendered)}")
        results[model] = probe

    path = FIRST_TOKEN_PATH
    payload = {}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            payload = {}
    payload[sample_id] = {
        "scenario": sample_id,
        "dilemma_id": dilemma["id"],
        "base_url": base_url,
        "models": results,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  first-token probe written to {path.relative_to(REPO)}")
    return results


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
        # Only vLLM-served targets get the base URL; openrouter/ models and
        # the judge keep their own endpoints.
        command_base = base_url if base_url and model.startswith("openai/") else None
        command = eval_command(model, judge, epochs, sample_id, command_base)
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
        help="Base URL of a vLLM server (e.g. http://localhost:8087/v1). It "
        "applies to the openai/<name> target models (the judge and any "
        "openrouter/ models keep their own endpoint). Without --base-url, "
        "any server on --vllm-port is probed and its adapters implicitly "
        "get http://localhost:<port>/v1 as base URL.",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=VLLM_PORT,
        help="Port to probe for an already-running vLLM server (EVAL_PORT "
        "env overrides the default). Served adapters are auto-added as "
        "targets and get a first-token probe.",
    )
    parser.add_argument(
        "--no-adapters",
        action="store_true",
        help="Disable the auto-discovery of served adapters on --vllm-port.",
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

    # ── vLLM adapter auto-discovery + per-model base URL ──────────
    # Local adapter models (``openai/<name>``) talk to the vLLM server;
    # every other model keeps its own provider endpoint. An explicit
    # ``--base-url`` applies to the ``openai/<name>`` models only: the
    # judge and openrouter models always use their own endpoints.
    # Without ``--base-url`` the server on --vllm-port is probed: when it
    # answers, every served adapter is auto-added as a target and the
    # implicit base URL ``http://localhost:<port>/v1`` covers them.
    local_models = [model for model in models if model.startswith("openai/")]
    base_url = args.base_url
    if not base_url and not args.no_adapters:
        served = vllm_served_models(args.vllm_port)
        if served:
            implicit = f"http://localhost:{args.vllm_port}/v1"
            print(
                f"vLLM server on :{args.vllm_port} serves {len(served)} "
                f"model(s): {', '.join(served)}"
            )
            for name in served:
                candidate = f"openai/{name}"
                if candidate not in models:
                    models.append(candidate)
                    print(f"  auto-added {candidate}")
            local_models = [model for model in models if model.startswith("openai/")]
            if local_models and base_url is None:
                base_url = implicit
    if local_models and base_url is None:
        raise SystemExit(
            "openai/<name> models requested but no --base-url given and no "
            f"vLLM server answered on port {args.vllm_port}"
        )

    print(f"Scenario: {sample_id} ({dilemma_id})")
    print(f"Question: {dilemma['question']}")
    for number, option in enumerate(dilemma["answer_options"], 1):
        print(f"  {number}. {option}")
    print(f"Models: {', '.join(models)} | epochs: {args.epochs} | judge: {judge}")

    if args.dry_run:
        for model in models:
            command_base = base_url if model.startswith("openai/") else None
            print(
                ">>> "
                + " ".join(
                    eval_command(model, judge, args.epochs, sample_id, command_base)
                )
            )
        if local_models and base_url:
            print(f"(first-token probes would run against {base_url})")
        print("dry run: no evaluation, no export")
        return

    run_models(models, judge, args.epochs, sample_id, base_url, args.verbose)

    # First-token probe for the vLLM-served adapters ("as a regular model"):
    # one plain next-token request each; failure is never fatal.
    if local_models and base_url:
        print(f"\nFirst-token probes ({base_url}):")
        run_first_token_probes(local_models, base_url, dilemma, sample_id)

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
