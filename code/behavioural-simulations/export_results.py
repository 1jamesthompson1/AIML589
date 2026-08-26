"""Export inspect-ai eval logs into per-run outputs.

For every run (one sample = one run of a scenario, isolated in its own
sandbox) this writes, under ``output/runs/<model>/<profile>-<situation>/``:

- ``trajectory.json`` / ``trajectory.md`` — the full run (reasoning, tool
  calls, observations, and the dialogue for interactive scenarios),
- ``model_summary.json`` — the model's own review of its trajectory,
- ``judge_summary.json`` — the judge's structured, rubric-based evaluation
  (the structure differs per profile and situation),
- ``config.json`` — model / profile / situation identifiers for this run.

Plus an ``index.csv`` summarising all runs.

Usage:

    uv run export_results.py                      # uses output/logs
    uv run export_results.py --log-dir logs --output-dir output/runs
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import is_dataclass, asdict
from enum import Enum
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log

from run_simulations import render_transcript


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


def structured_trajectory(sample) -> list[dict]:
    """Turn the sample's messages into a JSON-safe, readable trajectory."""
    records = []
    for i, msg in enumerate(sample.messages or []):
        record: dict = {"index": i, "role": msg.role}
        if getattr(msg, "content", None):
            record["content"] = _json_safe(msg.content)
        if getattr(msg, "tool_calls", None):
            record["tool_calls"] = [
                {
                    "function": call.function,
                    "arguments": call.arguments,
                    "id": call.id,
                }
                for call in msg.tool_calls
            ]
        if msg.role == "tool":
            record["function"] = getattr(msg, "function", None)
            record["output"] = (
                msg.text if msg.text else json.dumps(_json_safe(msg.content))
            )
        records.append(record)
    return records


def run_dir_for(log, sample, output_dir: Path) -> Path:
    metadata = sample.metadata or {}
    return (
        output_dir
        / _model_slug(log.eval.model)
        / f"{metadata.get('profile_id', '?')}-{metadata.get('situation_id', '?')}"
        / f"{sample.id}"
    )


def export_log(log, output_dir: Path) -> dict:
    """Write per-run outputs for one eval log; return the index row(s)."""
    rows = []
    for sample in log.samples or []:
        run_dir = run_dir_for(log, sample, output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        metadata = sample.metadata or {}
        trajectory = structured_trajectory(sample)

        (run_dir / "trajectory.json").write_text(json.dumps(trajectory, indent=2))
        (run_dir / "trajectory.md").write_text(render_transcript(sample.messages or []))

        model_summary = metadata.get("model_summary", "")
        (run_dir / "model_summary.json").write_text(
            json.dumps({"model": log.eval.model, "summary": model_summary}, indent=2)
        )

        judge: dict = {}
        score = (sample.scores or {}).get("scenario_judge")
        if score is not None:
            judge = {
                "score": score.value,
                "explanation": score.explanation,
                "structured": (score.metadata or {}).get("judge_summary"),
            }
        (run_dir / "judge_summary.json").write_text(json.dumps(judge, indent=2))

        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "model": log.eval.model,
                    "log": str(log.location),
                    "profile_id": metadata.get("profile_id"),
                    "situation_id": metadata.get("situation_id"),
                    "scenario_type": metadata.get("scenario_type"),
                    "sandboxed_sample": True,
                },
                indent=2,
            )
        )

        structured = judge.get("structured") or {}
        rows.append(
            {
                "model": log.eval.model,
                "profile": metadata.get("profile_id"),
                "situation": metadata.get("situation_id"),
                "type": metadata.get("scenario_type"),
                "run_dir": str(run_dir.relative_to(output_dir.parent)),
                "judge_score": judge.get("score"),
                "judge_verdict": structured.get("overall", {}).get("verdict"),
            }
        )
    return rows


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

    all_rows = []
    for path in log_paths:
        log = read_eval_log(path)
        print(f"{path.name}: model={log.eval.model} samples={len(log.samples or [])}")
        all_rows.extend(export_log(log, output_dir))

    index_path = output_dir / "index.csv"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(index_path, index=False)
    print(f"\nWrote {len(all_rows)} run(s) to {output_dir}/")
    print(f"Summary index: {index_path}")


if __name__ == "__main__":
    main()
