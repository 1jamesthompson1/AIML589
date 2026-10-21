"""Mark samples in a bundle's eval logs as invalidated so they re-run.

A dilemma's prompt lives in ``profiles/<profile>.json``. When a profile is
edited (new system prompt, reworded situation, changed answer options) the
runs in an existing bundle no longer match the prompt on disk, so that
profile's results are stale while the other profiles' results are still
valid. Re-running the whole bundle would throw away good work; not
re-running leaves the bundle internally inconsistent.

Inspect's eval-set resume already knows how to do the surgical thing: a log
carrying an *invalidated* sample is re-run, and every clean sample in it is
reused by ``(id, epoch)``. This script flips that flag on the samples of the
chosen profiles, so the next ``full_run.py --run-id <same>`` re-runs exactly
those samples and carries the rest of the log forward untouched.

Usage (from the repository root):

    uv run python code/dilemmas/invalidate_samples.py \\
        --run-id 2026-10-19_d0a90383 \\
        --profile parish_advisor --profile clinical_support

    # then re-run the bundle in place (it resumes and fills the gaps)
    uv run python code/dilemmas/full_run.py \\
        --run-id 2026-10-19_d0a90383 --refit --epochs 5

    # rebuild the bundle manifest and its export
    uv run python code/dilemmas/full_run.py --finalize 2026-10-19_d0a90383

The logs are only edited in place after the invalidation is written, so a
failure part-way leaves the bundle with some models invalidated and others
not; running the script again is idempotent (an already-invalidated sample
keeps its first invalidation record).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.log._edit import ProvenanceData, invalidate_samples

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BUNDLES_DIR = HERE / "output" / "logs" / "full"


def bundle_logs(run_id: str) -> list[Path]:
    """The newest ``.eval`` log per model in the bundle.

    A bundle's model directory can hold more than one log (an interrupted
    run's log and the one that superseded it). The eval-set resumes from
    the newest, so that is the only one worth editing.
    """
    bundle = BUNDLES_DIR / run_id
    if not bundle.is_dir():
        raise SystemExit(f"no bundle at {bundle.relative_to(REPO)}")
    newest: dict[Path, Path] = {}
    for path in sorted(bundle.rglob("*.eval")):
        directory = path.parent
        current = newest.get(directory)
        if current is None or path.stat().st_mtime > current.stat().st_mtime:
            newest[directory] = path
    return [newest[directory] for directory in sorted(newest)]


def select(samples, profiles: list[str], sample_ids: list[str]):
    """The samples to invalidate, by profile id and/or exact sample id."""

    def matches(sample) -> bool:
        sample_id = str(sample.id)
        profile_id = str(sample.metadata.get("profile_id", ""))
        by_profile = any(
            profile_id == profile or sample_id.startswith(f"{profile}-")
            for profile in profiles
        )
        by_id = sample_id in sample_ids
        return by_profile or by_id

    return [sample for sample in samples if matches(sample)]


def load_profiles() -> dict[str, dict]:
    """The profile definitions on disk, keyed by profile id."""
    profiles = {}
    for path in sorted((HERE / "profiles").glob("*.json")):
        profile = json.loads(path.read_text(encoding="utf-8"))
        profiles[profile["profile"]] = profile
    return profiles


def is_stale(sample, profiles: dict[str, dict]) -> bool:
    """True when the sample measured an older version of its scenario.

    A sample carries the dilemma text and the system prompt it was run
    with. When either no longer matches the profile file, the run measured
    an earlier version of the scenario and is stale.
    """
    metadata = sample.metadata or {}
    profile = profiles.get(str(metadata.get("profile_id", "")))
    if profile is None:
        return False
    dilemma = metadata.get("dilemma") or {}
    current = next(
        (d for d in profile["dilemmas"] if d["id"] == metadata.get("dilemma_id")),
        None,
    )
    if current is None:
        return True
    if dilemma.get("situation") != current["situation"]:
        return True
    if dilemma.get("answer_options") != current["answer_options"]:
        return True
    return metadata.get("system_prompt") != profile["system_prompt"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-id",
        required=True,
        help="Bundle to edit (the directory under output/logs/full).",
    )
    parser.add_argument(
        "--profile",
        action="append",
        dest="profiles",
        default=[],
        help="Profile id to invalidate (repeatable), e.g. parish_advisor.",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        dest="sample_ids",
        default=[],
        help="Exact sample id to invalidate (repeatable), e.g. clinical_support-0.",
    )
    parser.add_argument(
        "--stale",
        action="store_true",
        help="Invalidate the samples whose stored dilemma text or system prompt "
        "no longer matches the profile file, whatever the profile. Precise, and "
        "the way to finish a refit that was interrupted part-way.",
    )
    parser.add_argument(
        "--reason",
        default="profile edited after the batch was run",
        help="Provenance reason recorded on the invalidation.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.profiles and not args.sample_ids and not args.stale:
        parser.error("pass at least one --profile, --sample-id or --stale")

    logs = bundle_logs(args.run_id)
    if not logs:
        raise SystemExit(f"no .eval logs under {args.run_id}")

    profiles = load_profiles() if args.stale else {}

    provenance = ProvenanceData(
        author="code/dilemmas/invalidate_samples.py",
        reason=args.reason,
        metadata={"run_id": args.run_id},
    )

    total = 0
    for path in logs:
        log = read_eval_log(path)
        targets = select(log.samples or [], args.profiles, args.sample_ids)
        if args.stale:
            # --stale looks at every sample; --profile/--sample-id narrow it.
            chosen = {id(sample) for sample in targets}
            stale = [
                sample
                for sample in log.samples or []
                if is_stale(sample, profiles) and (not chosen or id(sample) in chosen)
            ]
            targets = stale
        if not targets:
            print(f"  {path.parent.name}: no matching samples")
            continue
        uuids = [str(sample.uuid) for sample in targets]
        updated = invalidate_samples(log, uuids, provenance)
        if args.dry_run:
            print(f"  {path.parent.name}: would invalidate {len(targets)} sample(s)")
        else:
            write_eval_log(updated, path)
            print(f"  {path.parent.name}: invalidated {len(targets)} sample(s)")
        total += len(targets)

    if args.dry_run:
        print(f"dry run: {total} sample(s) would be invalidated")
    else:
        print(
            f"invalidated {total} sample(s) across {len(logs)} log(s); "
            f"re-run the bundle with --run-id {args.run_id} --refit to fill them"
        )


if __name__ == "__main__":
    main()
