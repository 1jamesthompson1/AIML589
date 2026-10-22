"""Spot-test one dilemma: re-run it for the chosen models.

The iteration loop for hand-editing a dilemma in ``profiles/``:

    uv run python code/dilemmas/spot_test.py clinical_support-0
    uv run python code/dilemmas/spot_test.py clinical_support_suicide_risk
    uv run python code/dilemmas/spot_test.py --list

The scenario may be a sample id (``<profile>-<index>``) or a dilemma id. The
script runs a fresh set of epochs (default 5) for each given model, then
prints each model's picks, stability and cost read straight from the ``.eval``
logs it wrote.
"""

from __future__ import annotations

import argparse
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
from datetime import datetime
from pathlib import Path

from inspect_ai.log import read_eval_log

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT_DIR = HERE / "output"
# Spot-test logs live beside (not inside) the full-run bundles, so they can
# be wiped freely and never get mistaken for a complete run.
LOG_DIR = OUTPUT_DIR / "logs" / "spot"
FIRST_TOKEN_PATH = OUTPUT_DIR / "first_token.json"

# Fallback only: used when no --model is passed; any --model replaces it.
# The spot-test set spans the globe: DeepSeek and GLM (China), GPT and Claude
# (US), Mistral Large 4 (France) and Cohere Command A+ (Canada), plus the
# Qwen base model when it is wanted. The fine-tuned Qwen adapters are served
# locally and run with --base-url and --model openai/<served-model-name>.
DEFAULT_MODELS = (
    # "openrouter/qwen/qwen3.8-27b",
    "openrouter/deepseek/deepseek-v4.1-flash",
    "openrouter/openai/gpt-6-luna",
    "openrouter/anthropic/claude-haiku-5.5",
    "openrouter/z-ai/glm-5.3-flash",
    # "openrouter/mistralai/mistral-large-4-0",
    "openrouter/cohere/command-a-plus",
)

# OpenRouter list prices, USD per million tokens, fetched from
# https://openrouter.ai/api/v1/models (per-token prices x 1e6). Whole set
# re-fetched 2026-10-21 (the DeepSeek, GLM and Qwen prices had moved since
# the original 2026-10-14 fetch). Re-check when adding models or after price
# changes; unpriced models report tokens only. Prices change: the
# behavioural-similarity table in
# code/behavioural-simulations/export_results.py is a 2026-09-08 snapshot and
# GLM has since doubled.
MODEL_PRICES: dict[str, dict[str, float]] = {
    "openrouter/qwen/qwen3.8-27b": {
        "input": 0.04,
        "output": 1.35,
        "cache_read": 0.018,
    },
    "openrouter/deepseek/deepseek-v4.1-flash": {
        "input": 0.02832,
        "output": 0.12,
        "cache_read": 0.000566,
    },
    "openrouter/openai/gpt-6-luna": {
        "input": 0.10,
        "output": 0.50,
        "cache_read": 0.01,
        "cache_write": 0.125,
    },
    "openrouter/anthropic/claude-haiku-5.5": {
        "input": 0.10,
        "output": 0.50,
        "cache_read": 0.01,
        "cache_write": 0.125,
    },
    "openrouter/z-ai/glm-5.3-flash": {
        "input": 0.047076,
        "output": 0.12,
        "cache_read": 0.006,
    },
    "openrouter/mistralai/mistral-large-4-0": {
        "input": 1.36,
        "output": 4.18,
        "cache_read": 0.14,
    },
    "openrouter/cohere/command-a-plus": {
        "input": 0.30,
        "output": 1.50,
        "cache_read": 0.15,
    },
}
PRICE_UNIT = 1_000_000

# Fine-tuned adapter id convention (code/fine-tuning/finetune.py):
# {HF_ORG}/{model_slug}-nz-wvs-{dataset}-{subpopulation}. Console displays
# truncate these to "FT <dataset>"; full ids stay in manifests and JSON.
FT_MARKER = "-nz-wvs-"


def short_name(model: str) -> str:
    """Compact console display name: provider prefix dropped; fine-tuned
    adapters (``<base>-nz-wvs-<dataset>-<population>``) compress to
    ``FT <dataset>`` so table columns stay narrow. Full ids remain in the
    manifest, the exported runs and the Inspect logs list."""
    name = model.split("/")[-1]
    if FT_MARKER in name:
        rest = name.partition(FT_MARKER)[2]
        return f"FT {rest.rpartition('-')[0] or rest}"
    return name


def _price_for(model: str) -> dict[str, float] | None:
    """The MODEL_PRICES entry for a model (longest matching prefix)."""
    matches = [price for price in MODEL_PRICES if model.startswith(price)]
    return MODEL_PRICES[max(matches, key=len)] if matches else None


def run_cost(usage: dict) -> tuple[float, bool]:
    """USD cost of one run's usage block, and False when it holds a model
    with no price entry.

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
            print(f"  {short_name(model):34s} first-token probe unavailable")
            continue
        rendered = [
            f"'{entry['token']}' {entry['p']:.2f}" for entry in probe["top_tokens"][:6]
        ]
        print(f"  {short_name(model):34s} first token: {', '.join(rendered)}")
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
    a hosted model. The metadata stamps mark every log as a spot test of this
    scenario, so logs stay self-describing when copied around."""
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
        "code/dilemmas/output/logs/spot",
        "--sample-id",
        sample_id,
        "--metadata",
        "kind=spot",
        "--metadata",
        f"scenario={sample_id}",
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
    ``verbose`` is set; the last lines are shown if a run fails. A failure
    aborts before the exports are touched."""
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

    started = time.monotonic()
    failures = []
    for model, proc, path in procs:
        code = proc.wait()
        short = short_name(model)
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


def fresh_eval_logs(started_at: float) -> dict[str, list[tuple[datetime, Path]]]:
    """The spot logs this test wrote, grouped by their header's model.

    A log counts when its header ``created`` time is after ``started_at`` (a
    cheap mtime pre-filter skips the older logs without parsing). The file
    name carries no model, so the grouping uses the header, where both
    ``openrouter/`` targets and the vLLM ``openai/<name>`` adapters map to
    their model.
    """
    fresh: dict[str, list[tuple[datetime, Path]]] = {}
    for path in sorted(LOG_DIR.glob("*.eval")):
        try:
            if path.stat().st_mtime < started_at:
                continue
            header = read_eval_log(path, header_only=True)
            created = datetime.fromisoformat(header.eval.created)
        except Exception:
            continue  # unreadable/truncated log or bad timestamp: skip it
        if created.timestamp() <= started_at:
            continue
        fresh.setdefault(header.eval.model, []).append((created, path))
    return fresh


def list_new_eval_logs(
    fresh: dict[str, list[tuple[datetime, Path]]], models: list[str]
) -> None:
    """Print the inspect ``.eval`` logs this spot test wrote, one block each."""
    if not fresh:
        print("  (none)")
        return
    remaining = dict(fresh)
    for model in dict.fromkeys(models):
        logs = remaining.pop(model, None)
        if not logs:
            continue
        print(f"  {model}")
        for created, path in sorted(logs):
            print(
                f"    {created.strftime('%H:%M:%S')}  {path.relative_to(REPO).as_posix()}"
            )
    for model, logs in sorted(remaining.items()):  # not in --model: one block anyway
        print(f"  {model}")
        for created, path in sorted(logs):
            print(
                f"    {created.strftime('%H:%M:%S')}  {path.relative_to(REPO).as_posix()}"
            )


def stability(picks: list[int], option_count: int) -> float:
    """1 - H / ln(k) over the picked option indices (1.00 = all the same)."""
    counts = Counter(picks)
    proportions = [n / len(picks) for n in counts.values()]
    entropy = -sum(p * math.log(p) for p in proportions)
    return 1 - entropy / math.log(option_count)


def summarise(
    fresh: dict[str, list[tuple[datetime, Path]]],
    dilemma_id: str,
    options: list[str],
    rerun_models: set[str],
) -> None:
    """Print the re-run models' picks per run, counts, stability and cost.

    Read from the ``.eval`` logs this spot test just wrote — nothing has to
    be exported first. One aligned table line per model (``short_name``
    truncates the fine-tuned adapter prefix). Stability is 1 - H / ln(k) over
    the picks (1.00 = every run made the same choice); costs are the runs'
    token usage priced with MODEL_PRICES, target and judge together.
    """
    by_model: dict[str, list[tuple[datetime, int, int | None]]] = {}
    costs: dict[str, float] = {}
    unpriced: set[str] = set()
    runs = 0
    skipped = 0
    for model in sorted(fresh):
        for created, path in sorted(fresh[model]):
            log = read_eval_log(path)
            for sample in log.samples or []:
                scenario = (sample.metadata or {}).get("dilemma") or {}
                if scenario.get("id") != dilemma_id:
                    continue
                # Match the export: errored or empty runs are not results
                # (e.g. samples left behind by an aborted eval).
                response = sample.output.completion if sample.output else ""
                if sample.error or not (response or "").strip():
                    skipped += 1
                    continue
                runs += 1
                score = (sample.scores or {}).get("dilemma_judge")
                choice = ""
                if score is not None:
                    choice = (
                        (score.metadata or {}).get("choice") or score.answer or ""
                    ).strip()
                index = options.index(choice) + 1 if choice in options else None
                by_model.setdefault(model, []).append((created, sample.epoch, index))
                usage = {
                    name: tokens.model_dump(mode="json")
                    for name, tokens in (sample.model_usage or {}).items()
                }
                cost, priced = run_cost(usage)
                costs[model] = costs.get(model, 0.0) + cost
                if not priced:
                    unpriced.add(model)

    print(f"\nResults for {dilemma_id} ({runs} runs this spot test)")
    if skipped:
        print(f"  ({skipped} sample(s) skipped: errored or empty response)")
    distributions: dict[str, Counter] = {}
    for model, model_runs in sorted(by_model.items()):
        picks = [index for _, _, index in sorted(model_runs)]
        name = short_name(model)
        if any(index is None for index in picks):
            print(f"  {name:30s} picks {picks} (missing judge result)")
            continue
        counts = Counter(picks)
        rendered = ", ".join(f"{index} x{n}" for index, n in sorted(counts.items()))
        print(
            f"  {name:30s} picks {picks}  |  {rendered}"
            f"  |  stability {stability(picks, len(options)):.2f}"
            f"  |  cost ${costs.get(model, 0.0):.4f}"
        )
        distributions[model] = counts

    pooled = [
        index
        for model in sorted(rerun_models)
        for _, _, index in sorted(by_model.get(model, []))
        if index is not None
    ]
    if pooled:
        total_cost = sum(costs.get(model, 0.0) for model in rerun_models)
        warning = (
            " (some models are unpriced: tokens only, cost is a lower bound)"
            if unpriced
            else ""
        )
        print(
            f"\n  overall stability (all {len(pooled)} runs this spot test made): "
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
        help="Sample id (e.g. clinical_support-0) or dilemma id",
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
        help="Keep exactly the models passed: don't auto-add the other "
        "models the vLLM server serves when an openai/<name> target is "
        "given.",
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
        print(f"No --model given; using the default set: {', '.join(models)}")

    # ── vLLM-served models + per-model base URL ──────────────────
    # Local models (``openai/<name>``) talk to the vLLM server; every other
    # model keeps its own provider endpoint. Nothing local runs unless an
    # ``openai/<name>`` target is given: that target runs against
    # --base-url (or the probed server) together with every other model the
    # server serves (base + adapters). --no-adapters keeps exactly the
    # models passed. An explicit --base-url applies to the ``openai/<name>``
    # models only: the judge and openrouter models always use their own
    # endpoints.
    base_url = args.base_url
    local_models = [model for model in models if model.startswith("openai/")]
    if local_models and not args.no_adapters:
        served = vllm_served_models(args.vllm_port)
        if served:
            if base_url is None:
                base_url = f"http://localhost:{args.vllm_port}/v1"
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
        # --no-adapters (or a server that stopped answering): resolve the
        # base URL from the server on the probed port when one answers.
        if vllm_served_models(args.vllm_port):
            base_url = f"http://localhost:{args.vllm_port}/v1"
            print(f"vLLM server answered on :{args.vllm_port}")
        else:
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

    started_at = time.time()  # logs created after this are this spot test's
    run_models(models, judge, args.epochs, sample_id, base_url, args.verbose)

    # First-token probe for the vLLM-served adapters ("as a regular model"):
    # one plain next-token request each; failure is never fatal.
    if local_models and base_url:
        print(f"\nFirst-token probes ({base_url}):")
        run_first_token_probes(local_models, base_url, dilemma, sample_id)

    # The inspect .eval logs this spot test produced, one block per model,
    # then the picks/stability/cost summary read straight from those logs.
    # Nothing is exported — the website keeps the previous export until
    # `uv run python code/dilemmas/export_results.py` refreshes it.
    print("\nInspect logs:")
    fresh = fresh_eval_logs(started_at)
    list_new_eval_logs(fresh, models)
    summarise(fresh, dilemma_id, dilemma["answer_options"], set(models))


if __name__ == "__main__":
    main()
