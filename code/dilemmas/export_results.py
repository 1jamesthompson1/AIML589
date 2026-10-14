"""Export successful dilemma samples using the behavioural simulation layout.

Run from the repository root: uv run python code/dilemmas/export_results.py
Existing numbered exports are updated; stable run IDs preserve provenance.
``--reset-sample <sample-id>`` re-exports a re-run scenario from its latest
run set only; ``--reset-model`` narrows that to the models being re-run, so a
spot test does not leave stale runs of other models in the manifest (used by
``spot_test.py``). Other dilemmas keep their history.
``retired_samples.json`` lists sample ids retired by profile merges; the
exporter skips them so older logs cannot resurrect the removed profiles.

``index.json`` is the website manifest: it lists every exported run plus each
profile (name, the one-sentence role summary from ``profiles/<profile>.json``
and the system prompt the model was given) and every dilemma (situation,
question, answer options, and the tension name and value options resolved from
``tensions.json``) so the viewer and the survey can render a dilemma without
fetching each run's ``config.json`` first. ``index.csv`` keeps the flat run
table.
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


def load_retired_samples() -> frozenset[str]:
    """Sample ids retired by profile merges; skipped even if older logs hold them."""
    path = HERE / "retired_samples.json"
    if not path.exists():
        return frozenset()
    data = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(data.get("retired", []))


def load_tensions() -> dict[int, dict]:
    """Tension definitions by id. Dilemmas link to these; they do not copy them."""
    tensions = json.loads((HERE / "tensions.json").read_text(encoding="utf-8"))
    return {tension["id"]: tension for tension in tensions}


def load_profiles() -> dict[str, dict]:
    """Profile definitions by profile id.

    The profile JSON files stay the source of truth for what the agent is
    (name, role summary, organisation): the exported run only pins the system
    prompt it actually used.
    """
    profiles = {}
    for path in sorted((HERE / "profiles").glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        profiles[profile["profile"]] = profile
    return profiles


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run_system_prompt(sample) -> str:
    """The profile system prompt as the run actually used it.

    Newer logs store it in the sample metadata; fall back to the first system
    message so older logs export with the same manifest shape.
    """
    prompt = (sample.metadata or {}).get("system_prompt")
    if not prompt:
        for message in sample.messages or []:
            if message.role == "system":
                prompt = message.text
                break
    return (prompt or "").strip()


async def export(
    log_dir: Path,
    output_dir: Path,
    reset_samples: frozenset[str] = frozenset(),
    reset_models: frozenset[str] = frozenset(),
) -> int:
    """Group repetitions chronologically and export the logged source content.

    Sample ids in ``reset_samples`` have their (profile, dilemma) groups
    exported from the most recent run set only, so a re-run scenario replaces
    its earlier runs instead of adding another version. When ``reset_models``
    is given, only those models' runs are exported for the reset scenarios - a
    spot test clears the stale runs of the models it did not re-run."""
    groups = defaultdict(list)
    seen = set()
    reset_groups: set[tuple[str, str]] = set()
    retired = load_retired_samples()
    for path in sorted(log_dir.glob("*.eval")):
        log = await read_eval_log_async(
            io.BytesIO(path.read_bytes()), format="eval", resolve_attachments=True
        )
        for sample in log.samples or []:
            meta = sample.metadata or {}
            if sample.error or not meta.get("dilemma"):
                continue
            if sample.id in retired:
                continue
            run_id = f"{log.eval.eval_id}_{sample.id}_e{sample.epoch}"
            if run_id in seen:
                continue
            seen.add(run_id)
            key = (log.eval.model, meta["profile_id"], meta["dilemma"]["id"])
            groups[key].append((log.eval.created, run_id, path, log, sample))
            if sample.id in reset_samples:
                reset_groups.add((meta["profile_id"], meta["dilemma"]["id"]))

    rows = []
    profiles: dict[str, dict] = {}
    dilemmas: dict[tuple[str, str], dict] = {}
    tensions = load_tensions()
    profile_defs = load_profiles()
    # (profile, dilemma) -> model -> judge choice -> runs, for the website's
    # per-dilemma outcome chart. Filled as runs are exported; the live dict is
    # referenced from every dilemma entry, so the final dump is complete.
    choices: dict[tuple[str, str], dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
    for (model, profile, dilemma_id), runs in sorted(groups.items()):
        if (profile, dilemma_id) in reset_groups:
            if reset_models and model not in reset_models:
                continue  # superseded by the re-run models of a spot test
            newest = max(run[0] for run in runs)
            runs = [run for run in runs if run[0] == newest]
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
            if not response or not response.strip():
                continue  # no visible answer: nothing to export or judge
            score = (sample.scores or {}).get("dilemma_judge")
            if score is not None:
                choice = (
                    (score.metadata or {}).get("choice") or score.answer or ""
                ).strip()
                if choice:
                    choices[(profile, dilemma_id)][model][choice] += 1
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
            definition = profile_defs.get(profile, {})
            profiles.setdefault(
                profile,
                {
                    "name": (
                        sample.metadata.get("profile_name")
                        or definition.get("name")
                        or profile
                    ),
                    "role_summary": definition.get("role_summary", ""),
                    "system_prompt": run_system_prompt(sample),
                },
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
                # model -> judge choice -> exported runs, for the viewer's
                # outcome chart (live dict; complete once every run is seen)
                "choice_counts": choices[(profile, dilemma_id)],
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
    parser.add_argument(
        "--reset-sample",
        action="append",
        default=[],
        metavar="SAMPLE_ID",
        help="Export only the latest run set for this sample's dilemma (repeatable).",
    )
    parser.add_argument(
        "--reset-model",
        action="append",
        default=[],
        metavar="MODEL",
        help="With --reset-sample: only export this model's runs for the reset "
        "scenario (repeatable).",
    )
    args = parser.parse_args()
    if not args.log_dir.is_dir():
        parser.error(f"log directory does not exist: {args.log_dir}")
    count = asyncio.run(
        export(
            args.log_dir,
            args.output_dir,
            frozenset(args.reset_sample),
            frozenset(args.reset_model),
        )
    )
    print(f"Exported {count} runs to {args.output_dir}")


if __name__ == "__main__":
    main()
