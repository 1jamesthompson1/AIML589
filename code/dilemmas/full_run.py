"""Run a complete dilemma batch as a self-contained eval-set bundle.

A batch runs all 30 dilemmas for every listed `--model` so they can be
compared like for like — two is the usual comparison, but one, three or
more work the same way. A single `openai/<served-name>` target auto-adds
every model the vLLM server hosts (base model + adapters), so one argument
is enough for a full served-family run; `--no-adapters` keeps exactly the
models passed:

    uv run python code/dilemmas/full_run.py \
        --model openai/Qwen3.8-27B-nz-wvs-modal_response-overall \
        --model openai/Qwen/Qwen3.8-27B --epochs 50
    uv run python code/dilemmas/full_run.py --finalize 2026-10-16_initial-full-runs --epochs 5
    uv run python code/dilemmas/full_run.py --list-batches

The eval-sets run concurrently by default, and the batch prints a per-model
progress table (samples done / total) every ``--progress-interval`` seconds.

Follow progress from another shell with ``inspect ctl task list`` (live counts
from the running evals) or ``uv run python code/dilemmas/full_run.py --status``
(same table, and it also works for a bundle that has finished). A batch can be
run in the background with ``nohup`` and watched with either.

Resume by rerunning the same command with the same ``--run-id``: eval-set recovers incomplete samples and skips completed work (``--incomplete-action retry`` is the eval-set default).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from inspect_ai.log import read_eval_log

from export_results import (
    ACTIVE_FILE,
    active_run_id,
    batch_export,
    complete_bundles,
    newest_bundle_logs,
    write_active_index,
)
from export_results import slug
from spot_test import default_judge, inspect_command, short_name, vllm_served_models

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUTPUT_DIR = HERE / "output"
BUNDLES_DIR = OUTPUT_DIR / "logs" / "full"
RUNS_DIR = OUTPUT_DIR / "runs"

# How long to wait for Inspect's control channel before falling back to the
# flushed logs (the control channel can be slow to answer while the evals
# are busy).
CTL_TIMEOUT_S = 45


def make_run_id() -> str:
    """Date + short random suffix, e.g. ``2026-10-19_ab12cd34``."""
    return f"{datetime.now():%Y-%m-%d}_{uuid4().hex[:8]}"


def bundle_dir(run_id: str) -> Path:
    return BUNDLES_DIR / run_id


def eval_set_command(
    model: str,
    judge: str,
    epochs: int,
    run_id: str,
    base_url: str | None = None,
) -> list[str]:
    """The inspect eval-set command for one model of the batch.

    The per-model log directory is the eval set's storage scope; the bundle
    root holds the per-model directories plus the final ``run.json``.
    ``--model-base-url`` applies to the evaluated model only (inspect
    resolves the judge role separately), so a vLLM-served target can still
    be judged by a hosted model.
    """
    command = inspect_command() + [
        "eval-set",
        "code/dilemmas/task.py",
        "--model",
        model,
        "--model-role",
        f"judge={judge}",
        "--epochs",
        str(epochs),
        "--id",
        f"{run_id}--{slug(model)}",
        "--log-dir",
        f"code/dilemmas/output/logs/full/{run_id}/{slug(model)}",
        "--metadata",
        "kind=full",
        "--metadata",
        f"run_id={run_id}",
        "--metadata",
        f"epochs={epochs}",
    ]
    if base_url and model.startswith("openai/"):
        command += [
            "--model-base-url",
            base_url,
            "-M",
            "api_key=EMPTY",
            "-M",
            "responses_api=False",
        ]
    return command


# ── progress ────────────────────────────────────────────────
#
# How far each model of a bundle has got. Two sources: Inspect's control
# channel (live counts from the running eval processes) and the eval-set
# logs (what has been flushed to disk). The control channel wins where it
# has a row for the model; the logs are always there, including for a
# bundle whose run has finished or died.


@dataclass
class ModelProgress:
    """How far one model's eval-set has got."""

    slug: str
    model: str
    completed: int
    total: int | None
    in_flight: int | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    live: bool = False
    log: Path | None = None

    @property
    def elapsed_s(self) -> float | None:
        """Run time so far (the whole run once the model has finished)."""
        if self.started_at is None:
            return None
        end = self.finished_at or datetime.now(UTC)
        return max(0.0, (end - self.started_at).total_seconds())

    @property
    def eta_s(self) -> float | None:
        """Seconds left, from the rate so far (None when not yet knowable).

        Needs a few completed samples: a rate taken from one sample says
        more about that sample than about the run.
        """
        elapsed = self.elapsed_s
        if not elapsed or not self.total:
            return None
        if self.completed >= self.total:
            return 0.0
        if self.completed < 3:
            return None
        return elapsed / self.completed * (self.total - self.completed)


def _parse_created(value: str | None) -> datetime | None:
    """Parse an ISO timestamp, or None (the control channel uses epoch seconds)."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _read_zip_json(zf: zipfile.ZipFile, member: str) -> dict | None:
    """Read one JSON member, or None if it is missing/unreadable.

    Inspect writes ``.eval`` logs with zstd compression, which only Python
    3.14+ can inflate; on an older interpreter the metadata is unavailable
    but the member *count* still is, so progress degrades rather than fails.
    """
    try:
        with zf.open(member) as stream:
            return json.loads(stream.read())
    except (KeyError, NotImplementedError, ValueError, OSError):
        return None


def _model_progress(model_dir: Path) -> ModelProgress | None:
    """Progress for one model's eval-set directory (None if it has no log)."""
    logs = sorted(model_dir.glob("*.eval"), key=lambda p: p.stat().st_mtime)
    if not logs:
        return None
    log = logs[-1]

    try:
        zf = zipfile.ZipFile(log)
    except (zipfile.BadZipFile, OSError):
        # Mid-write snapshot that cannot be opened; the next poll will see it.
        return None

    with zf:
        names = zf.namelist()
        completed = sum(1 for name in names if name.startswith("samples/"))

        model = model_dir.name
        total: int | None = None
        started_at: datetime | None = None
        finished_at: datetime | None = None

        start = _read_zip_json(zf, "_journal/start.json")
        if start:
            spec = start.get("eval") or {}
            model = spec.get("model") or model
            started_at = _parse_created(spec.get("created"))
            dataset = spec.get("dataset") or {}
            epochs = (spec.get("config") or {}).get("epochs") or 1
            samples = dataset.get("samples")
            if samples:
                total = int(samples) * int(epochs)

        header = _read_zip_json(zf, "header.json")
        if header:
            results = header.get("results") or {}
            if results.get("total_samples"):
                total = int(results["total_samples"])
            stats = header.get("stats") or {}
            started_at = _parse_created(stats.get("started_at")) or started_at
            finished_at = _parse_created(stats.get("completed_at"))

    return ModelProgress(
        slug=model_dir.name,
        model=model,
        completed=completed,
        total=total,
        started_at=started_at,
        finished_at=finished_at,
        log=log,
    )


def control_tasks() -> list[dict]:
    """The control channel's view of the running evals (empty if unavailable).

    ``inspect ctl task list --json`` reports each running eval's sample
    counts. It is asked for over the CLI because that is the supported
    interface; the same interpreter's ``inspect`` is used so no second
    environment is created.
    """
    exe = Path(sys.executable).parent / "inspect"
    if not exe.exists():
        return []
    try:
        proc = subprocess.run(
            [str(exe), "ctl", "task", "list", "--json"],
            capture_output=True,
            text=True,
            timeout=CTL_TIMEOUT_S,
            check=False,
        )
        data = json.loads(proc.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    tasks = data.get("tasks")
    return tasks if isinstance(tasks, list) else []


def _live_for(model_dir: Path, tasks: list[dict]) -> ModelProgress | None:
    """Live progress for one model from the control channel, if it is running."""
    # The bundle lives behind a symlink (code/dilemmas/output -> artifacts/),
    # so the control channel's path and ours differ textually: compare the
    # resolved paths.
    root = model_dir.resolve()
    for task in tasks:
        location = task.get("log_location")
        if not location:
            continue
        try:
            Path(location).resolve().relative_to(root)
        except (ValueError, OSError):
            continue  # this eval belongs to another bundle
        samples = task.get("samples") or {}
        total = samples.get("total") or None
        done = (
            (samples.get("completed") or 0)
            + (samples.get("errored") or 0)
            + (samples.get("cancelled") or 0)
        )
        started_at = _parse_created(task.get("started_at"))
        if started_at is None and task.get("started_at"):
            started_at = datetime.fromtimestamp(float(task["started_at"]), tz=UTC)
        return ModelProgress(
            slug=model_dir.name,
            model=task.get("model") or model_dir.name,
            completed=int(done),
            total=int(total) if total else None,
            in_flight=samples.get("in_flight"),
            started_at=started_at,
            live=True,
            log=Path(location),
        )
    return None


def bundle_progress(bundle: Path, live: bool = True) -> list[ModelProgress]:
    """Per-model progress for one bundle, in directory order."""
    if not bundle.is_dir():
        return []
    tasks = control_tasks() if live else []
    rows = []
    for model_dir in sorted(bundle.iterdir()):
        if not model_dir.is_dir():
            continue
        progress = _live_for(model_dir, tasks) or _model_progress(model_dir)
        if progress is not None:
            rows.append(progress)
    return rows


def newest_bundle() -> Path | None:
    """The most recently touched bundle directory (None if there are none)."""
    if not BUNDLES_DIR.is_dir():
        return None
    bundles = [d for d in BUNDLES_DIR.iterdir() if d.is_dir()]
    if not bundles:
        return None
    return max(bundles, key=lambda d: d.stat().st_mtime)


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _short_model(model: str, width: int = 46) -> str:
    """Elide the middle of a long model name (keep provider and adapter tail)."""
    if len(model) <= width:
        return model
    keep = width - 1
    head = (keep * 2) // 3
    tail = keep - head
    return f"{model[:head]}…{model[-tail:]}"


def render_progress(run_id: str, rows: list[ModelProgress]) -> str:
    """A fixed-width table: model, done/total, running, elapsed, ETA."""
    lines = [f"bundle {run_id}", ""]
    header = f"{'model':46s} {'done':>12s} {'running':>7s} {'elapsed':>8s} {'eta':>8s}"
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        total = row.total
        done = f"{row.completed}/{total}" if total else f"{row.completed}/?"
        running = str(row.in_flight) if row.in_flight is not None else "-"
        lines.append(
            f"{_short_model(row.model):46s} {done:>12s} {running:>7s} "
            f"{_duration(row.elapsed_s):>8s} {_duration(row.eta_s):>8s}"
        )
    if not rows:
        lines.append("(no model logs yet)")
    return "\n".join(lines)


def print_progress(run_id: str) -> None:
    """Print the bundle's per-model progress (samples done / total)."""
    rows = bundle_progress(bundle_dir(run_id))
    if not rows:
        return
    print()
    print(render_progress(run_id, rows))
    print(flush=True)


def status(run_id: str | None, watch: bool, interval: int, live: bool) -> None:
    """Print a bundle's progress, optionally refreshing until interrupted."""
    while True:
        bundle = bundle_dir(run_id) if run_id else newest_bundle()
        if bundle is None:
            raise SystemExit(f"no bundles under {BUNDLES_DIR}")
        rows = bundle_progress(bundle, live=live)
        if watch:
            print("\033[2J", end="")  # clear the screen between refreshes
        print(render_progress(bundle.name, rows), flush=True)
        if not watch:
            return
        time.sleep(interval)


def run_batches(
    models: list[str],
    judge: str,
    epochs: int,
    run_id: str,
    base_url: str | None = None,
    sequential: bool = False,
    progress_interval: int = 60,
) -> None:
    """Run each model's eval-set of the batch.

    Concurrent by default, each model's console output captured to a
    ``<model-slug>.console.log`` at the bundle root (part of the bundle,
    outside the eval-set storage scope so a resume is not seen as dirty).
    ``--sequential`` streams one model's eval-set display at a time instead.
    While the batch runs, the per-model progress table is printed every
    ``progress_interval`` seconds (samples done / total, from the running
    evals where Inspect knows them and from the eval-set logs otherwise).
    A failure aborts before the export step; rerun with the same --run-id to
    resume (eval-set recovers incomplete samples and skips finished work).
    """
    started = time.monotonic()
    failures: list[tuple[str, Path | None]] = []
    if not sequential:
        procs: list[tuple[str, subprocess.Popen, Path | None]] = []
        for model in models:
            command_base = (
                base_url if base_url and model.startswith("openai/") else None
            )
            command = eval_set_command(model, judge, epochs, run_id, command_base)
            console = bundle_dir(run_id) / f"{slug(model)}.console.log"
            console.parent.mkdir(parents=True, exist_ok=True)
            with console.open("w") as stream:
                proc = subprocess.Popen(
                    command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT
                )
            procs.append((model, proc, console))

        consoles = {model: console for model, _, console in procs}

        # Poll rather than block on each process in turn: a parallel batch
        # gives no output at all until its first model finishes, which for a
        # 30-dilemma x N-epoch run is most of an hour.
        pending = {model: proc for model, proc, _ in procs}
        while pending:
            for model, proc in list(pending.items()):
                code = proc.poll()
                if code is None:
                    continue
                del pending[model]
                print(
                    f"  {short_name(model):34s} "
                    f"{'ok' if code == 0 else f'FAILED (exit {code})'}"
                    f"  ({time.monotonic() - started:.0f}s)",
                    flush=True,
                )
                if code != 0:
                    failures.append((model, consoles[model]))
            if pending:
                if progress_interval > 0:
                    print_progress(run_id)
                time.sleep(progress_interval or 5)
        for model, console in failures:
            print(f"\n--- tail of {model}: {console.relative_to(REPO)} ---")
            lines = console.read_text(errors="replace").splitlines()
            print("\n".join(lines[-25:]))
    else:
        for model in models:
            command_base = (
                base_url if base_url and model.startswith("openai/") else None
            )
            command = eval_set_command(model, judge, epochs, run_id, command_base)
            print(f"\n>>> {short_name(model)}: {' '.join(command)}\n", flush=True)
            code = subprocess.run(command, cwd=REPO).returncode
            print(
                f"\n  {short_name(model):34s} {'ok' if code == 0 else f'FAILED (exit {code})'}"
                f"  ({time.monotonic() - started:.0f}s)",
                flush=True,
            )
            if code != 0:
                failures.append((model, None))
                break  # resume by rerunning with the same --run-id
            if progress_interval > 0:
                print_progress(run_id)

    if failures:
        raise SystemExit(
            "batch incomplete: rerun with --run-id "
            f"{run_id} to resume (failed: {', '.join(model for model, _ in failures)})"
        )


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _settled_header(path: Path, header, wait_s: float = 10.0):
    """Re-read a log whose header still says "started" but is actually done.

    Inspect finalises a log's header on a background flush, so a log can
    read as "started" for a moment after the eval-set process that wrote
    it has exited. When every sample is present and scored, that is a
    finished run whose header has not landed yet: wait for it rather than
    declaring the bundle incomplete.
    """
    if header.status != "started":
        return header
    results = header.results
    if not results or not results.total_samples:
        return header
    if results.completed_samples != results.total_samples:
        return header
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        time.sleep(1.0)
        try:
            refreshed = read_eval_log(path, header_only=True)
        except Exception:
            continue
        if refreshed.status != "started":
            return refreshed
    return header


def finalize(
    run_id: str, epochs: int | None = None, judge: str | None = None
) -> dict | None:
    """Build the bundle's ``run.json`` from its logs, writing it only when
    every observed log succeeded.

    Only the newest log per model is observed: an interrupted log superseded
    by a resume/refit stays on disk for provenance but must not make the
    bundle look incomplete or add duplicate runs to its manifest. ``epochs``/
    ``judge`` fill in bundles whose logs predate the metadata stamps (e.g. the
    backfilled 2026-10-16 batch); stamped logs carry them in ``eval.metadata``.
    Returns the manifest, or None when the bundle has no logs or any log
    failed. A model whose eval-set never produced a log is simply absent from
    the manifest.
    """
    bundle = bundle_dir(run_id)
    paths = newest_bundle_logs(bundle)
    if not paths:
        print(f"no .eval logs under {bundle.relative_to(REPO)}")
        return None

    entries = []
    stamped_epochs: int | None = None
    stamped_judge: str | None = None
    for path in paths:
        try:
            header = read_eval_log(path, header_only=True)
        except Exception as error:
            print(f"unreadable log {path.relative_to(REPO)}: {error}")
            return None
        header = _settled_header(path, header)
        metadata = header.eval.metadata or {}
        stamped_epochs = stamped_epochs or metadata.get("epochs")
        roles = header.eval.model_roles or {}
        role = roles.get("judge") if isinstance(roles, dict) else None
        role_model = (
            role.get("model")
            if isinstance(role, dict)
            else getattr(role, "model", None)
        )
        stamped_judge = stamped_judge or role_model
        entries.append(
            {
                "file": path.relative_to(bundle).as_posix(),
                "model": header.eval.model,
                "status": header.status,
                "samples": header.results.completed_samples if header.results else None,
                "created": header.eval.created,
                "eval_set_id": header.eval.eval_set_id,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )

    manifest = {
        "schema": "wvs-dilemma-full-run/v1",
        "run_id": run_id,
        "complete": all(entry["status"] == "success" for entry in entries),
        "created": min(entry["created"] for entry in entries if entry["created"]),
        "finished": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "task_file": "code/dilemmas/task.py",
        "epochs": epochs or stamped_epochs,
        "judge": judge or stamped_judge,
        "models": list(dict.fromkeys(entry["model"] for entry in entries)),
        "logs": entries,
    }
    if not manifest["complete"]:
        failed = [entry["file"] for entry in entries if entry["status"] != "success"]
        print(f"bundle incomplete: failed/cancelled logs: {', '.join(failed)}")
        print("rerun with the same --run-id to resume")
        return None

    (bundle / "run.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"run.json written: {len(entries)} log(s), "
        f"{sum(entry['samples'] or 0 for entry in entries)} samples, "
        f"epochs={manifest['epochs']}, commit={manifest['git_commit']}"
    )
    return manifest


def list_batches() -> None:
    """Print every complete bundle, marking the active one."""
    active = active_run_id()
    if not complete_bundles():
        print("(no complete bundles — write one with full_run.py)")
        return
    for run_id, directory in complete_bundles():
        data = json.loads((directory / "run.json").read_text(encoding="utf-8"))
        mark = "*" if run_id == active else " "
        models = ", ".join(short_name(model) for model in data.get("models", []))
        print(
            f"{mark} {run_id}  epochs={data.get('epochs')}  logs={len(data.get('logs', []))}"
            f"\n    {models}"
        )
    print(
        f"\n(* = batch output/runs/index.json exports; pin with echo <run-id> > {ACTIVE_FILE})"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        default=None,
        help="Target model (repeatable: any number runs as one batch, two "
        "makes the usual comparison). A single openai/<served-name> target "
        "auto-adds every model the vLLM server hosts (base + adapters) "
        "unless --no-adapters is given.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--judge", default=None)
    parser.add_argument(
        "--run-id", default=None, help="Bundle id (default: date + random suffix)"
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Base URL of a vLLM server (applies to the openai/<name> targets "
        "only; the judge and openrouter/ models keep their own endpoints). "
        "Without it, a server on --vllm-port is probed when openai/ models "
        "are requested.",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=int(os.environ.get("EVAL_PORT", "8087")),
        help="Port to probe for an already-running vLLM server (EVAL_PORT env "
        "overrides the default).",
    )
    parser.add_argument(
        "--no-adapters",
        action="store_true",
        help="Disable the auto-discovery of served models on --vllm-port "
        "(a single openai/<served-name> target then runs alone).",
    )
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Stream the eval-sets one at a time instead of running them "
        "concurrently with captured output.",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Skip the batch export and active-export refresh after a successful run.",
    )
    parser.add_argument(
        "--refit",
        action="store_true",
        help="Re-run a bundle that is already complete, in place. The bundle's "
        "run.json is dropped and rebuilt from the logs afterwards. Combined "
        "with invalidate_samples.py this re-runs only the invalidated samples "
        "(e.g. a profile whose prompt was edited) and reuses the rest of "
        "every log.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list-batches", action="store_true")
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print the progress of a bundle (samples done / total per model) "
        "and exit without running anything. --run-id picks the bundle "
        "(default: the most recently touched one).",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="With --status: keep refreshing the table until interrupted",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Seconds between refreshes with --status --watch (default: 30)",
    )
    parser.add_argument(
        "--no-live",
        action="store_true",
        help="With --status: read the flushed logs only (skip the control "
        "channel, which can be slow to answer while the evals are busy)",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=60,
        help="Seconds between progress updates while the batch runs "
        "(default: 60; 0 disables them)",
    )
    parser.add_argument(
        "--finalize",
        metavar="RUN_ID",
        help="(Re)write a bundle's run.json from its logs; only completes "
        "when every log succeeded. --epochs/--judge backfill bundles whose "
        "logs predate the metadata stamps.",
    )
    args = parser.parse_args()

    if args.list_batches:
        list_batches()
        return
    if args.status:
        status(args.run_id, args.watch, args.interval, live=not args.no_live)
        return
    if args.finalize:
        manifest = finalize(args.finalize, args.epochs, args.judge)
        if manifest and not args.no_export:
            count = batch_export(args.finalize)
            active = write_active_index()
            print(
                f"exported {count} runs to output/runs/full/{args.finalize}/index.json"
                f" (active export: {active} runs)"
            )
        return

    if not args.models:
        parser.error(
            "a batch compares models: pass --model (repeatable; two makes a comparison)"
        )
    models = list(args.models)
    run_id = args.run_id or make_run_id()
    bundle = bundle_dir(run_id)
    if (bundle / "run.json").exists():
        if not args.refit:
            raise SystemExit(
                f"bundle {run_id} is already complete: pick a new --run-id"
            )
        # A refit rewrites the bundle's logs, so its manifest has to go: it is
        # rebuilt from the logs by finalize() once the run finishes.
        (bundle / "run.json").unlink()
        print(f"refitting bundle {run_id}: dropped its run.json")
    elif args.refit and not args.run_id:
        raise SystemExit("--refit needs an explicit --run-id")
    if any(bundle.glob("*")) and not args.run_id:
        raise SystemExit(f"{bundle.relative_to(REPO)} already holds a partial bundle")

    judge = args.judge or default_judge()
    if not judge:
        raise SystemExit("no judge model: set JUDGE_MODEL in .env or pass --judge")
    epochs = args.epochs or 5

    # Local (vLLM) models. A single openai/<name> target auto-adds the other
    # adapters the served instance hosts, so one name pulls in the whole
    # served family; --no-adapters keeps exactly the models passed. Every
    # openai/ target talks to the server: --base-url, or the --vllm-port probe.
    base_url = args.base_url
    if len(models) == 1 and models[0].startswith("openai/") and not args.no_adapters:
        served = vllm_served_models(args.vllm_port)
        if served:
            base_url = base_url or f"http://localhost:{args.vllm_port}/v1"
            print(f"vLLM server on :{args.vllm_port} serves {len(served)} model(s)")
            for name in served:
                candidate = f"openai/{name}"
                if candidate not in models:
                    models.append(candidate)
                    print(f"  auto-added {candidate}")
    if any(model.startswith("openai/") for model in models) and not base_url:
        if vllm_served_models(args.vllm_port):
            base_url = f"http://localhost:{args.vllm_port}/v1"
            print(f"vLLM server answered on :{args.vllm_port}")
        else:
            raise SystemExit(
                "openai/<name> models requested but no --base-url given and no "
                f"vLLM server answered on port {args.vllm_port}"
            )

    print(f"Run id: {run_id}")
    print(
        f"Batch ({len(models)} model{'s' if len(models) != 1 else ''}): "
        f"{' vs '.join(short_name(model) for model in models)}"
        f" | epochs: {epochs} | judge: {judge}"
    )

    if args.dry_run:
        for model in models:
            command_base = base_url if model.startswith("openai/") else None
            print(
                ">>> "
                + " ".join(eval_set_command(model, judge, epochs, run_id, command_base))
            )
        print("dry run: no evaluation, no export")
        return

    run_batches(
        models,
        judge,
        epochs,
        run_id,
        base_url,
        args.sequential,
        max(0, args.progress_interval),
    )
    manifest = finalize(run_id, args.epochs, judge)
    if manifest is None:
        raise SystemExit("full run did not complete: no run.json written")

    if not args.no_export:
        count = batch_export(run_id)
        active = write_active_index()
        print(
            f"exported {count} runs to output/runs/full/{run_id}/index.json"
            f" (active export: {active} runs)"
        )


if __name__ == "__main__":
    main()
