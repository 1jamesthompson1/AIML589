"""Export successful dilemma samples using the behavioural simulation layout.

Run from the repository root: uv run python code/dilemmas/export_results.py
Existing numbered exports are updated; stable run IDs preserve provenance.
``--partial`` (newest in-progress bundle) and ``--batch`` (latest complete
bundle) both work without a run id; pass one to pick a specific bundle.
``retired_samples.json`` lists sample ids retired by profile merges; the
exporter skips them so older logs cannot resurrect the removed profiles.

Exports mirror the log layout (``full_run.py`` bundles, spot tests):

    output/runs/full/<run-id>/   one complete batch's export: index.json
                                 (+ index.csv) and every run file
    output/runs/spot/            the rolling spot-test export (its own data
                                 source; never merged into a batch view)
    output/runs/index.json       the active batch's export: the batch
                                 (logs/full/ACTIVE pin, else the latest
                                 complete bundle) and nothing else - kept for
                                 scripts; the website loads per-batch
                                 manifests from batches.json
    output/runs/batches.json     every exported batch - the website's
                                 **Batch** selector source

``index.json`` (and each per-batch manifest) lists every exported run plus each
profile (name, the one-sentence role summary, the survey framing - intro and
outro from ``profiles/<profile>.json`` - and the system prompt the model was
given) and every dilemma (situation, question, answer options) so the viewer
and the survey can render a dilemma without fetching each run's
``config.json`` first. ``index.csv`` keeps the flat run table.
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

from inspect_ai.log import read_eval_log, read_eval_log_async

HERE = Path(__file__).resolve().parent
DEFAULT_BUCKET = "1jamesthompson1/wvs-nz-value-alignment-evals"
LOGS_DIR = HERE / "output" / "logs"
BUNDLES_DIR = LOGS_DIR / "full"
ACTIVE_FILE = BUNDLES_DIR / "ACTIVE"
SPOT_LOG_DIR = LOGS_DIR / "spot"
RUNS_DIR = HERE / "output" / "runs"
FULL_RUNS_DIR = RUNS_DIR / "full"
SPOT_RUNS_DIR = RUNS_DIR / "spot"


def slug(value: str) -> str:
    """Make provider/model and scenario identifiers safe as path components."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip(".") or "unknown"


def complete_bundles() -> list[tuple[str, Path]]:
    """(run_id, dir) for every full-run bundle with a run.json, newest first.

    Newest = the bundle manifest's ``finished`` timestamp (full_run.py writes
    it; the backfilled initial bundle gets one at migration time).
    """
    bundles: list[tuple[str, Path, str]] = []
    if BUNDLES_DIR.is_dir():
        for directory in BUNDLES_DIR.iterdir():
            manifest_path = directory / "run.json"
            if not directory.is_dir() or not manifest_path.exists():
                continue
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            bundles.append(
                (
                    data.get("run_id") or directory.name,
                    directory,
                    data.get("finished") or "",
                )
            )
    bundles.sort(key=lambda item: item[2], reverse=True)
    return [(run_id, directory) for run_id, directory, _ in bundles]


def newest_bundle_logs(bundle: Path) -> list[Path]:
    """The authoritative ``.eval`` log per model in a full-run bundle.

    A resumed or refit model directory can hold the interrupted log and the
    log that superseded it. Eval-set resumes from the newest log, so exporting
    both would duplicate every run. Recency is the log's own ``created``
    timestamp (not the file mtime, which an older log can still be appending
    to), with mtime only as a fallback for unreadable headers.
    """

    newest: dict[Path, tuple[str, float, Path]] = {}
    for path in sorted(bundle.rglob("*.eval")):
        # A normal bundle keeps one directory per model, so the directory
        # names the model and its logs can be compared. The backfilled
        # initial bundle keeps its logs flat in the bundle root (one log
        # per model), so there the log itself is the group: grouping by
        # the bundle root would keep only the newest log and drop every
        # other model's runs from the export.
        directory = path.parent if path.parent != bundle else path
        try:
            header = read_eval_log(path, header_only=True)
            created = header.eval.created or ""
        except Exception:
            created = ""
        key = (created, path.stat().st_mtime, path)
        current = newest.get(directory)
        if current is None or key[:2] > current[:2]:
            newest[directory] = key
    return [newest[directory][2] for directory in sorted(newest)]


def in_progress_bundles() -> list[tuple[str, Path]]:
    """(run_id, dir) for bundles without a run.json, most recently written first.

    Recency comes from the bundle's newest ``.eval`` log: the logs sit in
    per-model subdirectories, so the bundle directory's own mtime does not
    move as a running batch appends samples. Falls back to the directory
    mtime for a bundle whose logs are not readable yet.
    """
    bundles: list[tuple[str, Path, float]] = []
    if BUNDLES_DIR.is_dir():
        for directory in BUNDLES_DIR.iterdir():
            if not directory.is_dir() or (directory / "run.json").exists():
                continue
            newest = max(
                (path.stat().st_mtime for path in directory.rglob("*.eval")),
                default=directory.stat().st_mtime,
            )
            bundles.append((directory.name, directory, newest))
    bundles.sort(key=lambda item: item[2], reverse=True)
    return [(run_id, directory) for run_id, directory, _ in bundles]


def latest_run_id(partial: bool = False) -> str | None:
    """The newest in-progress bundle, or the newest complete one."""
    bundles = in_progress_bundles() if partial else complete_bundles()
    return bundles[0][0] if bundles else None


def active_run_id() -> str | None:
    """The bundle ``output/runs/index.json`` exports: the ``logs/full/ACTIVE``
    pin when it names a complete bundle, else the latest complete bundle."""
    pinned = (
        ACTIVE_FILE.read_text(encoding="utf-8").strip()
        if ACTIVE_FILE.exists()
        else None
    )
    bundles = complete_bundles()
    if pinned:
        for run_id, _ in bundles:
            if run_id == pinned:
                return run_id
    return bundles[0][0] if bundles else None


def load_retired_samples() -> frozenset[str]:
    """Sample ids retired by profile merges; skipped even if older logs hold them."""
    path = HERE / "retired_samples.json"
    if not path.exists():
        return frozenset()
    data = json.loads(path.read_text(encoding="utf-8"))
    return frozenset(data.get("retired", []))


def load_profiles() -> dict[str, dict]:
    """Profile definitions by profile id.

    The profile JSON files stay the source of truth for what the agent is
    (name, role summary, survey framing): the exported run only pins the
    system prompt it actually used.
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
    batch_run_id: str | None = None,
    subdir: str = "",
    partial: bool = False,
    log_paths: list[Path] | None = None,
) -> int:
    """Group repetitions chronologically and export the logged source content.

    Reads every ``*.eval`` under ``log_dir`` (bundles nest logs in per-model
    subdirectories, so the walk is recursive) and exports each run. Rows are
    written relative to ``output_dir`` (numbered versions per model and
    dilemma, chronologically grouped). ``log_paths`` limits the read to an
    explicit set (a bundle's newest log per model); the default walks the
    whole directory (spot tests keep their whole history).

    ``batch_run_id`` marks a per-batch export: the manifest's top-level
    ``run_id`` and ``base_url`` point at that batch and an entry is upserted
    into ``batches.json``. ``subdir`` is the batch's location under
    ``output/runs`` (``full/<run-id>``, ``spot``) — it completes ``base_url``
    so bucket URLs resolve against the manifest's own directory. ``partial``
    marks that batches.json entry as an in-progress bundle's export."""
    groups = defaultdict(list)
    seen = set()
    retired = load_retired_samples()
    paths = (
        sorted(log_paths) if log_paths is not None else sorted(log_dir.rglob("*.eval"))
    )
    for path in paths:
        try:
            log = await read_eval_log_async(
                io.BytesIO(path.read_bytes()), format="eval", resolve_attachments=True
            )
        except Exception as error:
            # A bundle still being written can expose a partially flushed
            # log; skip that file rather than losing the whole export.
            print(
                f"[export] skipping unreadable log {path.relative_to(log_dir)}: {error}"
            )
            continue
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

    rows = []
    profiles: dict[str, dict] = {}
    dilemmas: dict[tuple[str, str], dict] = {}
    profile_defs = load_profiles()
    # (profile, dilemma) -> model -> judge choice -> runs, for the website's
    # per-dilemma outcome chart. Filled as runs are exported; the live dict is
    # referenced from every dilemma entry, so the final dump is complete.
    choices: dict[tuple[str, str], dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(int))
    )
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
            # Older logs hold copies of the removed tension fields; drop them
            # so every exported run has the same dilemma shape.
            dilemma = {
                key: value
                for key, value in raw_dilemma.items()
                if key not in ("tension", "tension_name", "value_options")
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
                    # Survey framing (survey/comparison-question.md) shown
                    # around the dilemma; profiles without it export "".
                    "dilemma_intro": definition.get("dilemma_intro", ""),
                    "dilemma_outro": definition.get("dilemma_outro", ""),
                    "system_prompt": run_system_prompt(sample),
                },
            )
            dilemmas[(profile, dilemma_id)] = {
                "profile_id": profile,
                "situation_id": dilemma_id,
                "situation": dilemma.get("situation", ""),
                "question": dilemma.get("question", ""),
                "answer_options": dilemma.get("answer_options", []),
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
                + (f"{subdir}/" if subdir else "")
            ),
            **({"run_id": batch_run_id} if batch_run_id else {}),
            "active_run_id": active_run_id(),
            "profiles": profiles,
            "dilemmas": sorted(
                dilemmas.values(),
                key=lambda d: (d["profile_id"], d["situation_id"]),
            ),
            "models": sorted({row["model"] for row in rows}),
            "runs": rows,
        },
    )
    upsert_batches(output_dir, batch_run_id or active_run_id(), rows, partial=partial)
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


def upsert_batches(
    output_dir: Path, run_id: str | None, rows: list[dict], partial: bool = False
) -> None:
    """Record an exported batch in ``batches.json`` at the runs root.

    The entry (run id, generated-at, models, run count) drives the website's
    **Batch** selector; the active batch's own export (output/runs/index.json)
    references its sub-export. ``partial`` marks an in-progress bundle's export
    — the completed batch's export replaces the entry.
    """
    if not run_id:
        return
    path = RUNS_DIR / "batches.json"
    batches: dict[str, dict] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            batches = {entry["run_id"]: entry for entry in data.get("batches", [])}
        except (ValueError, KeyError):
            batches = {}
    entry: dict = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "models": sorted({row["model"] for row in rows}),
        "runs": len(rows),
    }
    if partial:
        entry["partial"] = True
    batches[run_id] = entry
    write_json(
        path,
        {
            "schema": "wvs-dilemma-batches/v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "active_run_id": active_run_id(),
            "batches": sorted(
                batches.values(), key=lambda entry: entry["generated_at"], reverse=True
            ),
        },
    )


def batch_export(run_id: str, partial: bool = False) -> int:
    """Export one full-run bundle to ``output/runs/full/<run-id>/``.

    Complete bundles only, unless ``partial``: an in-progress bundle (no
    ``run.json`` yet) exports the samples its ``.eval`` logs hold so far.
    Rerun as the batch progresses — the completed batch's export replaces
    it with the same version numbering."""
    bundle = BUNDLES_DIR / run_id
    if not bundle.is_dir():
        raise SystemExit(f"no such full-run bundle: {run_id} (see full_run.py)")
    if not partial and not (bundle / "run.json").exists():
        raise SystemExit(
            f"not a complete bundle: {run_id} (missing run.json — see "
            "full_run.py, or use --partial to export an in-progress batch)"
        )
    if partial and (bundle / "run.json").exists():
        raise SystemExit(
            f"bundle {run_id} is complete: use --batch to export it as a finished batch"
        )
    return asyncio.run(
        export(
            bundle,
            FULL_RUNS_DIR / run_id,
            batch_run_id=run_id,
            subdir=f"full/{run_id}",
            partial=partial,
            log_paths=newest_bundle_logs(bundle),
        )
    )


def spot_export() -> int:
    """(Re)export the spot-test logs to ``output/runs/spot/`` (idempotent).

    Returns early when no spot log is at least as new as the existing
    manifest (mtime check), so repeated exports - plain ``export_results.py``
    runs, batch runs - do not re-parse every spot log for nothing."""
    logs = (
        [path.stat().st_mtime_ns for path in SPOT_LOG_DIR.glob("*.eval")]
        if SPOT_LOG_DIR.is_dir()
        else []
    )
    if not logs:
        return 0
    index = SPOT_RUNS_DIR / "index.json"
    if index.exists() and max(logs) < index.stat().st_mtime_ns:
        return 0
    return asyncio.run(export(SPOT_LOG_DIR, SPOT_RUNS_DIR, subdir="spot"))


def write_active_index(all_bundles: bool = False) -> int:
    """Build ``output/runs/index.json`` — the active batch's export.

    The active full-run batch's sub-export (every complete bundle with
    ``--all``) and nothing else: the file is one batch exactly as exported,
    for scripts. The website does not read it; the viewer loads per-batch
    manifests from ``batches.json``. Spot-test logs stay their own data
    source (``output/runs/spot/``); the spot export is kept in step with
    ``logs/spot`` here, but is never merged in. Assembled from the
    sub-manifests, so no logs are re-read; missing sub-exports are exported
    first. Also refreshes ``batches.json``'s active stamp with the batch's
    own rows."""
    bundles = complete_bundles()
    active = active_run_id()
    rows: list[dict] = []
    profiles: dict[str, dict] = {}
    dilemmas: dict[tuple[str, str], dict] = {}

    for run_id, _directory in bundles:
        if run_id != active and not all_bundles:
            continue
        # The sub-manifest lives in the per-batch export (output/runs/full),
        # not the logs bundle; export it first when missing.
        if not (FULL_RUNS_DIR / run_id / "index.json").exists():
            batch_export(run_id)
        data = json.loads(
            (FULL_RUNS_DIR / run_id / "index.json").read_text(encoding="utf-8")
        )
        prefix = f"full/{run_id}"
        profiles.update(data.get("profiles") or {})
        for dilemma in data.get("dilemmas") or []:
            dilemmas[(dilemma["profile_id"], dilemma["situation_id"])] = dilemma
        for row in data["runs"]:
            row = dict(row)
            row["path"] = f"{prefix}/{row['path']}"
            rows.append(row)

    spot_export()  # cheap: keeps the separate spot source in step with logs/spot

    bucket = os.environ.get("HF_BUCKET", DEFAULT_BUCKET).strip("/")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        RUNS_DIR / "index.json",
        {
            "schema": "wvs-dilemma-runs-index/v2",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bucket": bucket,
            "base_url": f"https://huggingface.co/buckets/{bucket}/resolve/dilemmas/runs/",
            "active_run_id": active,
            "profiles": profiles,
            "dilemmas": sorted(
                dilemmas.values(),
                key=lambda d: (d["profile_id"], d["situation_id"]),
            ),
            "models": sorted({row["model"] for row in rows}),
            "runs": rows,
        },
    )
    with (RUNS_DIR / "index.csv").open("w", newline="") as stream:
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
    upsert_batches(RUNS_DIR, active, rows)  # refreshes the active stamp
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group()
    target.add_argument(
        "--batch",
        nargs="?",
        const="",
        metavar="RUN_ID",
        help="(Re)export one complete full-run bundle to "
        "output/runs/full/<run-id>/ for the website's Batch selector. "
        "Omit RUN_ID for the latest complete bundle.",
    )
    target.add_argument(
        "--partial",
        nargs="?",
        const="",
        metavar="RUN_ID",
        help="Export an in-progress full-run bundle's samples so far (no "
        "run.json yet) as its own viewer data source. Rerun as the batch "
        "progresses; the completed batch's export replaces it. Omit RUN_ID "
        "for the newest in-progress bundle.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Export every complete bundle into index.json (merged) instead of "
        "just the active one.",
    )
    args = parser.parse_args()

    if args.batch is not None:
        run_id = args.batch or latest_run_id()
        if not run_id:
            parser.error("no complete full-run bundle: write one with full_run.py")
        count = batch_export(run_id)
        active = write_active_index()
        print(
            f"Exported {count} runs to output/runs/full/{run_id}/index.json"
            f" (active export now holds {active} runs)"
        )
        return
    if args.partial is not None:
        run_id = args.partial or latest_run_id(partial=True)
        if not run_id:
            parser.error("no in-progress full-run bundle: start one with full_run.py")
        count = batch_export(run_id, partial=True)
        print(
            f"Exported {count} partial runs to "
            f"output/runs/full/{run_id}/index.json — pick it in the "
            "viewer's Data source selector"
        )
        return
    count = write_active_index(all_bundles=args.all)
    if not count:
        parser.error(
            "nothing to view: write a full-run bundle (full_run.py) or "
            "spot-test logs (spot_test.py)"
        )
    print(f"Refreshed active export: output/runs/index.json ({count} runs)")


if __name__ == "__main__":
    main()
