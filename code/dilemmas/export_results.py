"""Export successful dilemma samples using the behavioural simulation layout.

Run from the repository root: uv run python code/dilemmas/export_results.py
Existing numbered exports are updated; stable run IDs preserve provenance.

``index.json`` is the website manifest: it lists every exported run plus the
six work profiles and every dilemma (situation, question, answer options,
and the tension name and value options resolved from ``tensions.json``) so
the viewer can render a dilemma without fetching each run's ``config.json``
first. ``index.csv`` keeps the flat run table.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.log import read_eval_log_async

HERE = Path(__file__).resolve().parent
DEFAULT_BUCKET = "1jamesthompson1/wvs-nz-value-alignment-evals"


def slug(value: str) -> str:
    """Make provider/model and scenario identifiers safe as path components."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip(".") or "unknown"


def load_tensions() -> dict[int, dict]:
    """Tension definitions by id. Dilemmas link to these; they do not copy them."""
    tensions = json.loads((HERE / "tensions.json").read_text(encoding="utf-8"))
    return {tension["id"]: tension for tension in tensions}


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


async def export(log_dir: Path, output_dir: Path) -> int:
    """Group repetitions chronologically and export the logged source content."""
    groups = defaultdict(list)
    seen = set()
    for path in sorted(log_dir.glob("*.eval")):
        log = await read_eval_log_async(
            io.BytesIO(path.read_bytes()), format="eval", resolve_attachments=True
        )
        for sample in log.samples or []:
            meta = sample.metadata or {}
            if sample.error or not meta.get("dilemma"):
                continue
            run_id = f"{log.eval.eval_id}_{sample.id}_e{sample.epoch}"
            if run_id in seen:
                continue
            seen.add(run_id)
            key = (log.eval.model, meta["profile_id"], meta["dilemma"]["id"])
            groups[key].append((log.eval.created, run_id, path, log, sample))

    rows = []
    profiles: dict[str, dict] = {}
    dilemmas: dict[tuple[str, str], dict] = {}
    tensions = load_tensions()
    for (model, profile, dilemma_id), runs in sorted(groups.items()):
        for version, (_, run_id, path, log, sample) in enumerate(
            sorted(runs, key=lambda r: (r[0], r[1])), 1
        ):
            relative = (
                Path(slug(model)) / slug(f"{profile}-{dilemma_id}") / str(version)
            )
            run_dir = output_dir / relative
            run_dir.mkdir(parents=True, exist_ok=True)
            messages = sample.messages or []
            response = sample.output.completion if sample.output else ""
            score = (sample.scores or {}).get("dilemma_judge")
            raw_dilemma = sample.metadata["dilemma"]
            tension = tensions[raw_dilemma["tension"]]
            # Profiles carry only the tension id, so resolve the name and value
            # options from tensions.json. Older logs still hold copies of both;
            # drop them so every exported run has the same dilemma shape.
            dilemma = {
                key: value
                for key, value in raw_dilemma.items()
                if key not in ("tension_name", "value_options")
            }
            profiles.setdefault(
                profile, {"name": sample.metadata.get("profile_name") or profile}
            )
            dilemmas[(profile, dilemma_id)] = {
                "profile_id": profile,
                "situation_id": dilemma_id,
                "tension": dilemma.get("tension"),
                "tension_name": tension["name"],
                "situation": dilemma.get("situation", ""),
                "question": dilemma.get("question", ""),
                "answer_options": dilemma.get("answer_options", []),
                "value_options": tension["value_options"],
            }
            config = {
                "run_id": run_id,
                "model": model,
                "profile_id": profile,
                "situation_id": dilemma_id,
                "sample_id": sample.id,
                "epoch": sample.epoch,
                "version": version,
                "created": log.eval.created,
                "eval_log": str(path.resolve()),
                "dilemma": dilemma,
                "usage": {
                    name: usage.model_dump(mode="json")
                    for name, usage in (sample.model_usage or {}).items()
                },
            }
            write_json(run_dir / "config.json", config)
            write_json(
                run_dir / "transcript.json",
                {
                    "schema": "wvs-run-transcript/v1",
                    "run": {
                        key: value
                        for key, value in config.items()
                        if key not in {"dilemma", "usage"}
                    },
                    "messages": [
                        dict(
                            index=i,
                            **message.model_dump(mode="json", exclude_none=True),
                        )
                        for i, message in enumerate(messages)
                    ],
                },
            )
            (run_dir / "transcript.md").write_text(
                "\n\n".join(f"## {m.role}\n\n{m.text}" for m in messages) + "\n"
            )
            (run_dir / "response.txt").write_text((response or "") + "\n")
            write_json(
                run_dir / "judge.json", score.model_dump(mode="json") if score else {}
            )
            rows.append(
                {
                    key: config[key]
                    for key in (
                        "run_id",
                        "model",
                        "profile_id",
                        "situation_id",
                        "version",
                        "created",
                    )
                }
                | {"path": relative.as_posix()}
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket = os.environ.get("HF_BUCKET", DEFAULT_BUCKET).strip("/")
    write_json(
        output_dir / "index.json",
        {
            "schema": "wvs-dilemma-runs-index/v2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bucket": bucket,
            "base_url": (
                f"https://huggingface.co/buckets/{bucket}/resolve/dilemmas/runs/"
            ),
            "profiles": profiles,
            "dilemmas": sorted(
                dilemmas.values(),
                key=lambda d: (d["profile_id"], d["situation_id"]),
            ),
            "models": sorted({row["model"] for row in rows}),
            "runs": rows,
        },
    )
    with (output_dir / "index.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "run_id",
                "model",
                "profile_id",
                "situation_id",
                "version",
                "created",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", type=Path, default=HERE / "output/logs")
    parser.add_argument("--output-dir", type=Path, default=HERE / "output/runs")
    args = parser.parse_args()
    if not args.log_dir.is_dir():
        parser.error(f"log directory does not exist: {args.log_dir}")
    count = asyncio.run(export(args.log_dir, args.output_dir))
    print(f"Exported {count} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
