"""Export inspect-ai eval logs into per-run outputs for the website and analysis.

For every run (one sample = one run of a situation, isolated in its own
sandbox) this writes a directory per run, grouped by model and scenario
with a numbered run directory per run so a model's repeated runs of the
same scenario sit side by side:

    output/runs/<model>/<profile>-<situation>/<run-number>/
        transcript.json  - structured JSON transcript of the run: the full
                           message stream (system prompt, work item, agent
                           reasoning, tool calls, tool results and the final
                           outcome) in a stable schema designed to be loaded
                           by the public webapp and reused as training data,
        transcript.md    - the same run as readable text (for the report),
        self_review.txt  - the model's own summary of its work (plain text),
        judge.json       - the judge's structured, rubric-based evaluation
                           (the structure differs per profile and situation),
        config.json      - provenance (model, eval log, ids) plus the run's
                           token usage by model and role (see below).

Plus an ``index.csv`` summarising all runs and an ``index.json`` webapp
manifest (situations, models, and every run with its version/date/score,
duration, token usage and cost) for ``website/src/components/SimulationViewer.tsx``.

Every sample is exported: repeated runs of the same scenario by the same
model (re-runs, epochs) each get their own numbered run directory,
numbered in chronological order - run 1 is the oldest (e.g.
``.../welfare/initial_benefit_application/3/``). Each run's number is the
same as its ``version`` field in ``config.json``/``index.json``. Before
writing, any earlier export of the same (model, profile-situation)
directory is removed, so re-exports never accumulate stale run numbers.
Use ``--latest-only`` to export just the newest run of each
(model, profile-situation) pair (numbered 1), and ``--include-errors`` to
also export samples from interrupted/failed runs.

Usage tracking: ``config.json`` carries each run's token usage from the
eval log's own per-sample accounting - ``models`` keyed by model (target
model, judge model, ...) and ``roles`` keyed by role (``judge``, ``audit``,
``user``; the agent under test is what no role accounts for), plus the
wall-clock duration. Dollar cost is a simple tokens × price calculation:
fill in the manual ``MODEL_PRICES`` table (USD per million tokens) and each
run gets ``cost_usd`` per model plus a ``total_cost_usd``; unpriced models
report tokens only.

Usage:

    uv run export_results.py                      # uses output/logs
    uv run export_results.py --log-dir logs --output-dir output/runs
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path

import pandas as pd
from inspect_ai.log import EvalSample, read_eval_log
from inspect_ai.model import ModelUsage

from profiles import get_profile, profile_spec, situations
from run_simulations import render_transcript

# The transcript JSON schema tag written into each transcript.json. Bump when
# the layout below changes so the webapp / training code can tell versions
# apart.
TRANSCRIPT_SCHEMA = "wvs-run-transcript/v1"

DEFAULT_BUCKET = "1jamesthompson1/wvs-nz-value-alignment-evals"

# ---------------------------------------------------------------------------
# Prices (manual)
# ---------------------------------------------------------------------------
# Dollar costs are a simple calculation from the token counts in each run's
# ``config.json`` ``usage``: fill this table with USD per million tokens for
# the models you run (from the provider's pricing page, e.g. a model's
# OpenRouter page) and the export fills in ``cost_usd`` per model and
# ``total_cost_usd`` per run. Models without an entry just get tokens, no
# cost. ``cache_read``/``cache_write`` default to ``input`` when omitted.
# Rates below from https://openrouter.ai/api/v1/models on 2026-09-08
# (per-token prices x 1e6); re-check when you add models or after price
# changes.
MODEL_PRICES: dict[str, dict[str, float]] = {
    "openrouter/z-ai/glm-5.3-flash": {
        "input": 0.075,
        "output": 0.25,
        "cache_read": 0.015,
    },
    "openrouter/deepseek/deepseek-v4-flash-0731": {
        "input": 0.14,
        "output": 0.28,
        "cache_read": 0.028,
    },
    "openrouter/deepseek/deepseek-v4-flash": {
        "input": 0.088606,
        "output": 0.177212,
        "cache_read": 0.0177212,
    },
    "openrouter/openai/gpt-5.6-luna": {
        "input": 0.20,
        "output": 1.20,
        "cache_read": 0.02,
        "cache_write": 0.25,
    },
}
PRICE_UNIT = 1_000_000


def _price_for(model: str) -> dict[str, float] | None:
    """The MODEL_PRICES entry for a model (longest matching prefix)."""
    matches = [p for p in MODEL_PRICES if model.startswith(p)]
    return MODEL_PRICES[max(matches, key=len)] if matches else None


def _tokens_cost(model: str, tokens: dict) -> float | None:
    """Cost of one model's token counts (tokens × prices / 1M), or None when
    the model has no price entry. ``cache_read`` defaults to the input rate
    (cached input bills at the prompt rate unless priced separately);
    ``cache_write`` defaults to 0 (most providers do not bill cache
    writes)."""
    rates = _price_for(model)
    if rates is None:
        return None
    input_rate = rates.get("input", 0.0)
    return round(
        (
            tokens["input_tokens"] * input_rate
            + tokens["cache_read_tokens"] * rates.get("cache_read", input_rate)
            + tokens["cache_write_tokens"] * rates.get("cache_write", 0.0)
            + tokens["output_tokens"] * rates.get("output", 0.0)
        )
        / PRICE_UNIT,
        6,
    )


def _model_slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", model)


def _json_safe(obj):
    """Convert inspect-ai message content (which may contain dataclass
    content blocks such as ContentReasoning) into plain JSON types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _json_safe(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(item) for item in obj]
    return str(obj)


def _split_content(content) -> tuple[str | None, str]:
    """Split message content into ``(reasoning, text)``.

    Reasoning blocks are kept separate from the answer text rather than being
    flattened into it, so the webapp can display them distinctly and training
    pipelines can include or exclude them."""
    if isinstance(content, str):
        return None, content
    reasoning_parts, text_parts = [], []
    for block in content or []:
        text = getattr(block, "text", None)
        block_type = getattr(block, "type", "")
        if not text and hasattr(block, "reasoning"):
            text = getattr(block, "reasoning", "")
        if not text:
            continue
        if block_type == "reasoning" or hasattr(block, "signature"):
            reasoning_parts.append(text)
        else:
            text_parts.append(text)
    reasoning = "\n\n".join(reasoning_parts) or None
    text = "\n\n".join(text_parts)
    return reasoning, text


def _usage_dict(usage: ModelUsage) -> dict:
    """One ModelUsage record as JSON-safe totals (cache/reasoning fields are
    optional in the log; missing means zero)."""
    return {
        "input_tokens": usage.input_tokens,
        "cache_read_tokens": usage.input_tokens_cache_read or 0,
        "cache_write_tokens": usage.input_tokens_cache_write or 0,
        "output_tokens": usage.output_tokens,
        "reasoning_tokens": usage.reasoning_tokens or 0,
        "total_tokens": usage.total_tokens,
    }


def usage_summary(sample: EvalSample) -> dict:
    """Token usage (and dollar cost, where MODEL_PRICES has an entry) for one
    run (sample), from the eval log's own per-sample accounting: ``models``
    keyed by model (target model, judge model, ...) and ``roles`` keyed by
    role (``judge``/``audit``/``user`` - the agent under test is the share no
    role accounts for), plus the wall-clock duration."""
    models = {m: _usage_dict(u) for m, u in (sample.model_usage or {}).items()}
    for model, tokens in models.items():
        tokens["cost_usd"] = _tokens_cost(model, tokens)
    priced = [t["cost_usd"] for t in models.values() if t["cost_usd"] is not None]
    return {
        "duration_s": sample.total_time,
        "models": models,
        "roles": {r: _usage_dict(u) for r, u in (sample.role_usage or {}).items()},
        "total_cost_usd": round(sum(priced), 6) if priced else None,
    }


def structured_transcript(log_eval, sample: EvalSample, meta: dict) -> dict:
    """The run as a stable, JSON-safe document for the webapp and reuse as
    training data.

    A header identifies the deployment context (which model, which work
    profile, which situation), followed by the flat ``messages`` stream in
    conversation order: the system prompt, the initial work item, then the
    agent loop - assistant turns (with optional ``tool_calls``, e.g.
    ``search_client_record``), the matching ``tool`` result messages, and the
    closing assistant message that constitutes the outcome. For interactive
    situations, client replies arrive as tool results on the messaging tool.
    """
    messages = []
    for i, msg in enumerate(sample.messages or []):
        record: dict = {"index": i, "role": msg.role}
        reasoning, text = _split_content(msg.content)
        if reasoning:
            record["reasoning"] = reasoning
        if msg.role == "tool":
            record["function"] = getattr(msg, "function", None)
            record["tool_call_id"] = getattr(msg, "tool_call_id", None)
            record["content"] = (
                msg.text if msg.text else json.dumps(_json_safe(msg.content))
            )
        else:
            record["content"] = text
            if getattr(msg, "tool_calls", None):
                record["tool_calls"] = [
                    {
                        "id": call.id,
                        "function": call.function,
                        "arguments": _json_safe(call.arguments),
                    }
                    for call in msg.tool_calls
                ]
        messages.append(record)

    return {
        "schema": TRANSCRIPT_SCHEMA,
        "run": {
            "model": log_eval.model,
            "task": log_eval.task,
            "created": log_eval.created,
            "profile_id": meta.get("profile_id"),
            "situation_id": meta.get("situation_id"),
            "situation_name": meta.get("situation_name"),
            "situation_type": meta.get("scenario_type"),
            "interactive": meta.get("scenario_type") == "interactive",
            "sample_id": sample.id,
            "epoch": sample.epoch,
            "run_id": run_id(log_eval, sample),
        },
        "messages": messages,
    }


def situation_meta(meta: dict) -> dict:
    """Enrich sample metadata with the situation's display name and the
    public-facing descriptions (from its profile module's situations.json
    and the profile SUMMARY), tolerating renamed/removed items."""
    enriched = dict(meta)
    profile_id, situation_id = meta.get("profile_id"), meta.get("situation_id")
    try:
        match = next(s for s in situations(profile_id) if s["id"] == situation_id)
    except Exception:
        return enriched
    enriched["situation_name"] = match.get("name")
    enriched["situation_summary"] = match.get("summary", "")
    try:
        enriched["profile_summary"] = getattr(get_profile(profile_id), "SUMMARY", "")
    except Exception:
        pass
    return enriched


def export_sample(
    log,
    sample: EvalSample,
    run_dir: Path,
    version: int | None = None,
    usage: dict | None = None,
) -> Path:
    """Write one run's outputs into ``run_dir``; return it."""
    run_dir.mkdir(parents=True, exist_ok=True)

    metadata = sample.metadata or {}
    meta = situation_meta(metadata)
    usage = usage or usage_summary(sample)
    transcript = structured_transcript(log.eval, sample, meta)
    (run_dir / "transcript.json").write_text(json.dumps(transcript, indent=2))
    (run_dir / "transcript.md").write_text(render_transcript(sample.messages or []))

    (run_dir / "self_review.txt").write_text(metadata.get("model_summary", ""))

    score = (sample.scores or {}).get("scenario_judge")
    judge = {}
    if score is not None:
        judge = {
            "score": score.value,
            # Did the judge reply follow the generated schema; and was a
            # substantive overall score recovered at all.
            "schema_ok": (score.metadata or {}).get("structured_output"),
            "parse_failed": (score.metadata or {}).get("parse_failed"),
            "explanation": score.explanation,
            "structured": (score.metadata or {}).get("judge_summary"),
        }
    (run_dir / "judge.json").write_text(json.dumps(judge, indent=2))

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "model": log.eval.model,
                "task": log.eval.task,
                "created": log.eval.created,
                "log": str(log.location),
                "profile_id": meta.get("profile_id"),
                "situation_id": meta.get("situation_id"),
                "situation_name": meta.get("situation_name"),
                "situation_type": meta.get("scenario_type"),
                "sample_id": sample.id,
                "epoch": sample.epoch,
                "version": version,
                "run_id": run_id(log.eval, sample),
                # Why the agent loop ended (see run_simulations.py's
                # terminus_agent) and whether the run is usable: runs whose
                # harness gave out (message limit, model length, an
                # unavailable language backbone) are marked incomplete so
                # downstream pairing/analysis can exclude them.
                "ended": metadata.get("ended"),
                "run_complete": metadata.get("ended")
                not in (
                    None,
                    "message_limit",
                    "model_length",
                    "empty_response",
                    "no_terminator",
                ),
                "has_error": bool(sample.error),
                # Public-facing descriptions (survey / report reuse).
                "profile_summary": meta.get("profile_summary", ""),
                "situation_summary": meta.get("situation_summary", ""),
                # Token usage by model (target model, judge model, ...) and
                # by role (judge/audit/user), plus wall-clock duration.
                "usage": usage,
            },
            indent=2,
        )
    )
    return run_dir


def sample_key(log, sample) -> tuple | None:
    """Export key for a sample: (model, profile-situation). None if it lacks
    the identifiers the export needs."""
    metadata = sample.metadata or {}
    profile_id, situation_id = metadata.get("profile_id"), metadata.get("situation_id")
    if not profile_id or not situation_id:
        return None
    return _model_slug(log.eval.model), f"{profile_id}-{situation_id}"


def run_id(eval, sample) -> str:
    """A stable, globally unique id for the run, built from inspect-ai's own
    identifiers: the eval spec's `eval_id` (globally unique per eval) plus
    the sample id and epoch (unique within the eval). Deterministic across
    re-exports, so a run keeps its id even when the export layout changes."""
    return f"{eval.eval_id}_{sample.id}_e{sample.epoch}"


def collect_runs(log_paths: list[Path], include_errors: bool, latest_only: bool):
    """(model_slug, log, sample, version) tuples to export.

    Every non-errored sample is kept unless ``include_errors``. For each
    (model, profile-situation) key the samples are ordered by eval creation
    time and numbered 1..N (run numbers, oldest first; the number doubles
    as the run directory name). ``latest_only`` collapses each key to just
    its newest sample (run number 1).
    """
    candidates: dict[tuple, list] = {}
    for path in sorted(log_paths):  # filenames embed timestamps => oldest first
        log = read_eval_log(path)
        print(f"{path.name}: model={log.eval.model} samples={len(log.samples or [])}")
        for sample in log.samples or []:
            if sample.error and not include_errors:
                continue
            key = sample_key(log, sample)
            if key is None:
                continue
            candidates.setdefault(key, []).append((log, sample))
    pairs = []
    for key, entries in candidates.items():
        # Newest last: order first by eval creation time, then epoch.
        entries.sort(key=lambda pair: (pair[0].eval.created or "", pair[1].epoch))
        entries = entries[-1:] if latest_only else entries
        for version, (log, sample) in enumerate(entries, start=1):
            pairs.append((key[0], log, sample, version))
    return pairs


def build_manifest(rows: list[dict], output_dir: Path) -> dict:
    """The discovery document the website fetches before loading any runs.

    Written as ``index.json`` next to ``index.csv``; the website
    (``website/src/components/SimulationViewer.tsx``) fetches it from the
    bucket and uses ``base_url + run_dir/<file>`` for every run file.
    """
    import os

    bucket_id = os.environ.get("HF_BUCKET", DEFAULT_BUCKET).strip("/")

    profiles_by_id: dict[str, dict] = {}
    situations_list = []
    for profile_id in sorted({row.get("profile") for row in rows}):
        try:
            spec = profile_spec(get_profile(profile_id))
        except Exception:
            continue
        profiles_by_id[profile_id] = {
            "name": spec["name"],
            "summary": spec.get("summary", ""),
        }
        try:
            for s in situations(profile_id):
                situations_list.append(
                    {
                        "profile_id": profile_id,
                        "profile_name": spec["name"],
                        "situation_id": s["id"],
                        "name": s["name"],
                        "type": s.get("type"),
                        "summary": s.get("summary", ""),
                    }
                )
        except Exception:
            pass

    return {
        "schema": "wvs-sim-runs-index/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bucket": bucket_id,
        "base_url": (f"https://huggingface.co/buckets/{bucket_id}/resolve/bs/runs/"),
        "profiles": profiles_by_id,
        "situations": sorted(
            situations_list, key=lambda s: (s["profile_id"], s["situation_id"])
        ),
        "models": sorted({row.get("model") for row in rows}),
        "runs": [
            {
                "model": row.get("model"),
                "profile_id": row.get("profile"),
                "situation_id": row.get("situation"),
                "type": row.get("type"),
                "status": row.get("status"),
                "version": int(row.get("run_number") or 1),
                "created": row.get("created"),
                "run_dir": row.get("run_dir"),
                "run_id": row.get("run_id"),
                "judge_score": row.get("judge_score"),
                "judge_verdict": row.get("judge_verdict"),
                "ended": row.get("ended"),
                "run_complete": row.get("run_complete"),
                "has_error": row.get("has_error"),
                "duration_s": row.get("duration_s"),
                "usage": row.get("usage"),
                "total_cost_usd": row.get("total_cost_usd"),
            }
            for row in rows
        ],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    p.add_argument(
        "--log-dir",
        default=str(Path(__file__).resolve().parent / "output" / "logs"),
        help="Directory containing the inspect-ai eval logs.",
    )
    p.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "output" / "runs"),
        help="Directory to write the per-run outputs into.",
    )
    p.add_argument(
        "--latest-only",
        action="store_true",
        help="Export only the most recent run per (model, profile-situation) "
        "instead of keeping every run as a numbered version.",
    )
    p.add_argument(
        "--include-errors",
        action="store_true",
        help="Also export samples from interrupted/failed runs (no scores).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)

    log_paths = sorted(list(log_dir.glob("*.eval")) + list(log_dir.glob("*.json")))
    if not log_paths:
        raise SystemExit(
            f"no eval logs found in {log_dir} (run run_simulations.py first)"
        )

    pairs = collect_runs(log_paths, args.include_errors, args.latest_only)
    if not pairs:
        raise SystemExit(
            "no exportable runs found "
            "(skipped errored/interrupted samples; try --include-errors)"
        )

    rows = []
    # Runs are grouped under <model>/<profile>-<situation>/<run-number>/.
    # Before writing a scenario's runs, any earlier export of that
    # (model, scenario) directory is removed first so re-exports never
    # accumulate stale run numbers.
    scenario_dirs: dict[tuple, Path] = {}
    for model_slug, log, sample, version in pairs:
        metadata = sample.metadata or {}
        scenario = (
            f"{metadata.get('profile_id', '?')}-{metadata.get('situation_id', '?')}"
        )
        key = (model_slug, scenario)
        if key not in scenario_dirs:
            scenario_dir = output_dir / model_slug / scenario
            if scenario_dir.is_dir():
                shutil.rmtree(scenario_dir)
            scenario_dirs[key] = scenario_dir
        run_dir = scenario_dirs[key] / str(version)
        usage = usage_summary(sample)
        export_sample(log, sample, run_dir, version=version, usage=usage)

        judge: dict = {}
        score = (sample.scores or {}).get("scenario_judge")
        if score is not None:
            judge["score"] = score.value
            judge["structured"] = (score.metadata or {}).get("judge_summary") or {}
        rows.append(
            {
                "model": log.eval.model,
                "profile": metadata.get("profile_id"),
                "situation": metadata.get("situation_id"),
                "type": metadata.get("scenario_type"),
                "status": "error" if sample.error else "complete",
                "run_number": version,
                "run_id": run_id(log.eval, sample),
                "created": log.eval.created,
                "run_dir": str(run_dir.relative_to(output_dir)),
                "judge_score": judge.get("score"),
                "judge_verdict": judge.get("structured", {})
                .get("overall", {})
                .get("verdict"),
                "ended": metadata.get("ended"),
                "run_complete": metadata.get("ended")
                not in (
                    None,
                    "message_limit",
                    "model_length",
                    "empty_response",
                    "no_terminator",
                ),
                "has_error": bool(sample.error),
                "duration_s": usage.get("duration_s"),
                "usage": usage,
                "total_cost_usd": usage.get("total_cost_usd"),
            }
        )

    index_path = output_dir / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(index_path, index=False)

    manifest_path = output_dir / "index.json"
    manifest_path.write_text(
        json.dumps(build_manifest(rows, output_dir), indent=2) + "\n"
    )

    print(f"\nWrote {len(rows)} run(s) to {output_dir}/")
    print(f"Summary index: {index_path}")
    print(f"Webapp manifest: {manifest_path}")


if __name__ == "__main__":
    main()
