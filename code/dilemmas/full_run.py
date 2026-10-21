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

Each run is one bundle under ``output/logs/full/<run-id>/``. The log
directory is the eval set's storage scope (``.eval-set-id``,
``eval-set.json``) and ``inspect eval-set`` takes a single ``--model``, so
the bundle holds one eval-set subdirectory per model (named with the model
slug) plus each model's captured console output at the bundle root. Every
log is stamped ``--metadata kind=full --metadata run_id=... --metadata
epochs=...``, so a log stays self-describing when copied out of the bundle.

The eval-sets run concurrently by default (a comparison should not wait
on itself); ``--sequential`` streams them one at a time instead. While a
parallel batch runs, follow progress from another shell with
``inspect ctl task list`` or ``uv run inspect view <bundle>``.

A bundle is complete when its ``run.json`` exists: models, judge, epochs,
git commit, and one entry per log (status, samples, bytes, sha256). No
run.json = interrupted. Resume by rerunning the same command with the same
``--run-id``: eval-set recovers incomplete samples and skips completed work
(``--incomplete-action retry`` is the eval-set default).

On success the batch is exported to ``output/runs/full/<run-id>/`` for the
website's **Batch** selector, and the active batch's export
(``output/runs/index.json``, for scripts) is refreshed when the new batch is
the active one. The active batch defaults to the latest complete
bundle; pin one with ``echo <run-id> > output/logs/full/ACTIVE``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
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


def run_batches(
    models: list[str],
    judge: str,
    epochs: int,
    run_id: str,
    base_url: str | None = None,
    sequential: bool = False,
) -> None:
    """Run each model's eval-set of the batch.

    Concurrent by default, each model's console output captured to a
    ``<model-slug>.console.log`` at the bundle root (part of the bundle,
    outside the eval-set storage scope so a resume is not seen as dirty).
    ``--sequential`` streams one model's eval-set display at a time instead.
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
        for model, proc, console in procs:
            code = proc.wait()
            print(
                f"  {short_name(model):34s} {'ok' if code == 0 else f'FAILED (exit {code})'}"
                f"  ({time.monotonic() - started:.0f}s)"
            )
            if code != 0:
                failures.append((model, console))
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
        "finished": datetime.now(timezone.utc).isoformat(),
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

    run_batches(models, judge, epochs, run_id, base_url, args.sequential)
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
