"""Build survey comparisons between two models' dilemma responses.

Reads the exported runs of one full-run bundle (``output/runs/full/<run-id>/``,
written by ``export_results.py``) and pairs the two models' runs **on the same
dilemma**. Two subsets are written per run:

- ``survey/``   - the survey set: ``--per-dilemma`` (default 10) randomly
  chosen pairs per dilemma, i.e. 300 comparisons over the 30 dilemmas.
- ``complete/`` - every possible pair of the two models' runs on a dilemma
  (the full cross product), for the analysis that needs all of them.

Both subsets land under ``output/comparisons/<run-id>/<subset>/``:

    comparisons.json   the manifest: schema, the run, the two models, the
                       dilemmas (situation, question, answer options, the
                       profile's role summary and framing) and the list of
                       comparisons - the file the survey code reads.
    index.csv         the same comparisons as a flat table, for eyeballing.

Each comparison record carries the keys the survey's block builder already
uses (``comparison_id``, ``label``, ``role``, ``dilemma_intro``,
``dilemma_outro``, ``situation``, ``question``, ``responses``) plus the
provenance the analysis needs (both run ids, both models, each agent's judged
answer option). Respondents never see the model names.

Pairing: for every dilemma the two models' runs are the candidate pool (at
most ``--samples-per-model`` runs each, taken in version order); the survey
set draws ``--per-dilemma`` distinct pairs from that pool with a seeded RNG
(one RNG per dilemma, so the draw does not depend on the dilemmas' order), the
complete set takes every pair of every run of the two models. With the
5-epoch runs that is 25 candidate pairs per dilemma - 10 drawn for the
survey, all 25 in the complete set.

Which run plays Agent 1 and which Agent 2 is a stable pseudo-random
assignment per pair (the same rule as the survey's ``agent_order``: a hash of
the sorted run ids), so neither model is systematically first and repeated
runs keep the same layout.

Usage:

    uv run python code/dilemmas/generate_comparisions.py
    uv run python code/dilemmas/generate_comparisions.py --run-id 2026-10-19_d0a90383
    uv run python code/dilemmas/generate_comparisons.py \\
        --model-a openai/Qwen/Qwen3.8-27B \\
        --model-b openai/Qwen3.8-27B-nz-wvs-full_string_distribution-overall
    uv run python code/dilemmas/generate_comparisons.py --list-runs
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "output"
LOGS_DIR = OUTPUT_DIR / "logs" / "full"
FULL_RUNS_DIR = OUTPUT_DIR / "runs" / "full"
COMPARISONS_DIR = OUTPUT_DIR / "comparisons"
PROFILES_DIR = HERE / "profiles"

# The default pair: the untuned Qwen3.8-27B base model against the
# full-string-distribution fine-tune (the adapter trained on the model's full
# answer distribution over the WVS answers).
DEFAULT_MODEL_A = "openai/Qwen/Qwen3.8-27B"
DEFAULT_MODEL_B = "openai/Qwen3.8-27B-nz-wvs-full_string_distribution-overall"

# The comparison manifest schema tag. Bump when the layout changes so the
# survey / analysis code can tell versions apart.
COMPARISON_SCHEMA = "wvs-dilemma-comparisons/v1"

# One JSON manifest per subset, plus a flat CSV of the same rows.
MANIFEST_NAME = "comparisons.json"
CSV_NAME = "index.csv"


# --- Small helpers -----------------------------------------------------------


def slug(value: str) -> str:
    """Make a model id safe as a path component (the export's own rule)."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip(".") or "unknown"


def write_json(path: Path, value: object) -> None:
    """Write pretty JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def agent_order(agents: list[dict]) -> list[dict]:
    """Stable pseudo-random Agent 1 / Agent 2 assignment.

    A hash of the sorted run ids picks the order, so no model is
    systematically first (sorted-name order would always put the base model
    first) while repeated builds keep the same layout. The survey code uses
    the same rule, so an agent order can be decoded from the run ids alone.
    """
    key = hashlib.md5(
        "__".join(sorted(agent["run_id"] for agent in agents)).encode()
    ).digest()
    return agents if key[0] % 2 == 0 else agents[::-1]


# --- The run to compare ------------------------------------------------------


def complete_bundles(logs_dir: Path = LOGS_DIR) -> list[tuple[str, Path]]:
    """(run_id, dir) for every complete full-run bundle, newest first.

    Newest = the bundle's own ``created`` stamp (the backfilled initial
    bundle's ``finished`` stamp is a migration artefact, so ``created`` is the
    honest ordering key); ``finished`` breaks ties.
    """
    bundles: list[tuple[str, Path, str]] = []
    if logs_dir.is_dir():
        for directory in sorted(logs_dir.iterdir()):
            manifest_path = directory / "run.json"
            if not directory.is_dir() or not manifest_path.exists():
                continue
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except ValueError:
                continue
            stamp = data.get("created") or data.get("finished") or ""
            bundles.append((data.get("run_id") or directory.name, directory, stamp))
    bundles.sort(key=lambda item: item[2], reverse=True)
    return [(run_id, directory) for run_id, directory, _ in bundles]


def export_bundle(run_id: str) -> None:
    """Export a bundle that has no export yet.

    ``export_results`` is imported lazily so the common case (the export
    already exists) stays stdlib-only.
    """
    sys.path.insert(0, str(HERE))
    from export_results import batch_export  # noqa: PLC0415 - lazy, heavy import

    batch_export(run_id)


def resolve_run(run_id: str | None) -> tuple[str, Path]:
    """The bundle to compare and the directory holding its exported runs.

    ``--run-id`` picks a bundle; without it the newest complete bundle is
    used. The bundle is exported first if its export is missing.
    """
    if run_id:
        bundle = LOGS_DIR / run_id
        if not bundle.is_dir():
            raise SystemExit(
                f"no such full-run bundle: {run_id} "
                f"(available: {[rid for rid, _ in complete_bundles()]})"
            )
    else:
        bundles = complete_bundles()
        if not bundles:
            raise SystemExit(
                "no complete full-run bundle found - write one with full_run.py"
            )
        run_id, bundle = bundles[0]
    export_dir = FULL_RUNS_DIR / run_id
    if not (bundle / "run.json").exists():
        print(
            f"WARNING: {run_id} has no run.json - it is an in-progress bundle, "
            "so its export holds the samples so far, not the whole run."
        )
    if not (export_dir / "index.json").exists():
        print(f"Exporting {run_id} (no export found)...")
        export_bundle(run_id)
    return run_id, export_dir


def resolve_model(requested: str, available: list[str]) -> str:
    """Match a requested model name against the run's models.

    Accepts the full model id (``openai/Qwen/Qwen3.8-27B``), the export's
    directory tag (``openai_Qwen_Qwen3.8-27B``) or a unique part of the id
    (``full_string_distribution``).
    """
    if requested in available:
        return requested
    tags = {slug(model): model for model in available}
    if requested in tags:
        return tags[requested]
    bare = {slug(model).removeprefix("openai_"): model for model in available}
    if requested in bare:
        return bare[requested]
    matches = [model for model in available if requested.lower() in model.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise SystemExit(f"--model {requested!r} is ambiguous: {matches}")
    raise SystemExit(
        f"--model {requested!r} is not in this run (models: {sorted(available)})"
    )


# --- Reading the exported runs ----------------------------------------------


def load_profiles() -> dict[str, dict]:
    """Profile JSONs keyed by profile id (the ``profile`` field).

    The live files are the source of truth for the role summary and the
    survey framing; the manifest's exported copy is the fallback.
    """
    out: dict[str, dict] = {}
    if not PROFILES_DIR.is_dir():
        return out
    for path in sorted(PROFILES_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        out[data.get("profile") or path.stem] = data
    return out


def group_runs(manifest: dict) -> dict[str, dict[str, list[dict]]]:
    """Dilemma id -> model -> that model's runs, in version order.

    The dilemma id is ``<profile>-<situation>``, the same shape the spot
    test's sample ids use.
    """
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for run in manifest.get("runs", []):
        key = f"{run['profile_id']}-{run['situation_id']}"
        grouped[key][run["model"]].append(run)
    for by_model in grouped.values():
        for runs in by_model.values():
            runs.sort(key=lambda run: (run.get("version", 0), run["run_id"]))
    return grouped


def run_response(export_dir: Path, run: dict) -> str:
    """The run's exported response text ('' when the file is missing)."""
    path = export_dir / run["path"] / "response.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def run_choice(export_dir: Path, run: dict) -> str:
    """The judge's chosen answer option for the run ('' when unjudged)."""
    path = export_dir / run["path"] / "judge.json"
    if not path.exists():
        return ""
    try:
        judge = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return ""
    return (
        (judge.get("metadata") or {}).get("choice") or judge.get("answer") or ""
    ).strip()


# --- Pairing -----------------------------------------------------------------


def candidate_pairs(
    runs_a: list[dict], runs_b: list[dict], samples_per_model: int | None
) -> list[tuple[dict, dict]]:
    """Every (model A run, model B run) pair available for one dilemma.

    At most ``samples_per_model`` runs of each model are used (taken in
    version order), so a dilemma with many repetitions does not drown the
    others out. ``samples_per_model`` None uses every run - the complete
    subset does that, so it holds all the pairs there are.
    """
    a_runs = runs_a if samples_per_model is None else runs_a[:samples_per_model]
    b_runs = runs_b if samples_per_model is None else runs_b[:samples_per_model]
    return [(a, b) for a in a_runs for b in b_runs]


def draw_pairs(
    pool: list[tuple[dict, dict]], per_dilemma: int | None, rng: random.Random
) -> list[tuple[dict, dict]]:
    """``per_dilemma`` distinct pairs drawn from the pool (all of it when
    ``per_dilemma`` is None)."""
    if per_dilemma is None or len(pool) <= per_dilemma:
        drawn = list(pool)
    else:
        drawn = rng.sample(pool, per_dilemma)
    return sorted(
        drawn,
        key=lambda pair: (
            pair[0].get("version", 0),
            pair[1].get("version", 0),
            pair[0]["run_id"],
            pair[1]["run_id"],
        ),
    )


# --- Records -----------------------------------------------------------------


def dilemma_meta(
    manifest: dict, dilemma_id: str, profile_files: dict[str, dict]
) -> dict:
    """The dilemma's context: the profile's framing plus the situation, the
    question and the answer options."""
    profile_id, _, situation_id = dilemma_id.partition("-")
    dilemmas = {
        f"{d['profile_id']}-{d['situation_id']}": d
        for d in manifest.get("dilemmas", [])
    }
    dilemma = dilemmas.get(dilemma_id, {})
    exported = (manifest.get("profiles") or {}).get(profile_id, {})
    profile = profile_files.get(profile_id, {})
    return {
        "dilemma_id": dilemma_id,
        "profile_id": profile_id,
        "situation_id": situation_id,
        "name": profile.get("name") or exported.get("name") or profile_id,
        "role_summary": profile.get("role_summary") or exported.get("role_summary", ""),
        "dilemma_intro": profile.get("dilemma_intro")
        or exported.get("dilemma_intro", ""),
        "dilemma_outro": profile.get("dilemma_outro")
        or exported.get("dilemma_outro", ""),
        "situation": dilemma.get("situation", ""),
        "question": dilemma.get("question", ""),
        "answer_options": dilemma.get("answer_options", []),
    }


def build_record(
    pair: tuple[dict, dict],
    export_dir: Path,
    dilemma: dict,
) -> dict | None:
    """One comparison: the two runs as Agent 1 and Agent 2.

    The record holds the survey block's keys (``comparison_id``, ``label``,
    ``role``, ``dilemma_intro``, ``dilemma_outro``, ``situation``,
    ``question``, ``responses``) plus the provenance (run ids, models, the
    judged answer options).
    """
    agents = agent_order(list(pair))
    responses = [run_response(export_dir, run) for run in agents]
    if not all(responses):
        return None  # nothing to show a respondent
    choices = [run_choice(export_dir, run) for run in agents]
    versions = [run.get("version", 0) for run in agents]
    if versions[0] == versions[1]:
        label = f"{dilemma['profile_id']} - {dilemma['situation_id']} v{versions[0]}"
    else:
        label = (
            f"{dilemma['profile_id']} - {dilemma['situation_id']} "
            f"v{versions[0]}/{versions[1]}"
        )
    return {
        "comparison_id": f"{agents[0]['run_id']}__{agents[1]['run_id']}",
        "label": label,
        "dilemma_id": dilemma["dilemma_id"],
        "profile_id": dilemma["profile_id"],
        "situation_id": dilemma["situation_id"],
        # The survey block's wording fields.
        "role": dilemma["role_summary"] or dilemma["name"],
        "dilemma_intro": dilemma["dilemma_intro"],
        "dilemma_outro": dilemma["dilemma_outro"],
        "situation": dilemma["situation"],
        "question": dilemma["question"],
        "responses": responses,
        # Provenance: which run (and model) played which agent.
        "agents": {
            f"Agent {index + 1}": {
                "run_id": run["run_id"],
                "model": run["model"],
                "choice": choices[index],
                "response": responses[index],
                "version": versions[index],
                "run_path": run["path"],
            }
            for index, run in enumerate(agents)
        },
        "same_choice": bool(choices[0]) and choices[0] == choices[1],
    }


def build_comparisons(
    manifest: dict,
    export_dir: Path,
    grouped: dict[str, dict[str, list[dict]]],
    model_pairs: list[tuple[str, str]],
    per_dilemma: int | None,
    samples_per_model: int,
    seed: int,
) -> list[dict]:
    """The comparisons for the given model pairs, one RNG per dilemma.

    ``per_dilemma`` None takes every candidate pair (the complete set);
    otherwise that many pairs per dilemma are drawn at random.
    ``samples_per_model`` None uses every run of each model.
    """
    profile_files = load_profiles()
    records: list[dict] = []
    for dilemma_id in sorted(grouped):
        by_model = grouped[dilemma_id]
        meta = dilemma_meta(manifest, dilemma_id, profile_files)
        for model_a, model_b in model_pairs:
            runs_a = by_model.get(model_a) or []
            runs_b = by_model.get(model_b) or []
            if not runs_a or not runs_b:
                continue
            pool = candidate_pairs(runs_a, runs_b, samples_per_model)
            # One RNG per (dilemma, model pair): the draw does not depend on
            # the order the dilemmas are walked in.
            rng = random.Random(f"{seed}:{dilemma_id}:{model_a}|{model_b}")
            for pair in draw_pairs(pool, per_dilemma, rng):
                record = build_record(pair, export_dir, meta)
                if record is not None:
                    records.append(record)
    return records


# --- Output ------------------------------------------------------------------


def write_subset(
    out_dir: Path,
    subset: str,
    records: list[dict],
    dilemmas: list[dict],
    meta: dict,
) -> None:
    """Write one subset's manifest and flat CSV."""
    write_json(
        out_dir / MANIFEST_NAME,
        {
            "schema": COMPARISON_SCHEMA,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "subset": subset,
            **meta,
            "dilemmas": dilemmas,
            "comparisons": records,
        },
    )
    fields = [
        "comparison_id",
        "dilemma_id",
        "profile_id",
        "situation_id",
        "label",
        "agent1_run_id",
        "agent1_model",
        "agent1_choice",
        "agent2_run_id",
        "agent2_model",
        "agent2_choice",
        "same_choice",
    ]
    with (out_dir / CSV_NAME).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            agent1 = record["agents"]["Agent 1"]
            agent2 = record["agents"]["Agent 2"]
            writer.writerow(
                {
                    "comparison_id": record["comparison_id"],
                    "dilemma_id": record["dilemma_id"],
                    "profile_id": record["profile_id"],
                    "situation_id": record["situation_id"],
                    "label": record["label"],
                    "agent1_run_id": agent1["run_id"],
                    "agent1_model": agent1["model"],
                    "agent1_choice": agent1["choice"],
                    "agent2_run_id": agent2["run_id"],
                    "agent2_model": agent2["model"],
                    "agent2_choice": agent2["choice"],
                    "same_choice": record["same_choice"],
                }
            )


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Full-run bundle to compare (default: the newest complete bundle).",
    )
    parser.add_argument(
        "--model-a",
        default=DEFAULT_MODEL_A,
        help="The first model of the pair (default: the Qwen3.8-27B base model).",
    )
    parser.add_argument(
        "--model-b",
        default=DEFAULT_MODEL_B,
        help="The second model of the pair (default: the full-string-distribution "
        "fine-tune).",
    )
    parser.add_argument(
        "--per-dilemma",
        type=int,
        default=10,
        help="Comparisons per dilemma in the survey subset (default: 10, i.e. "
        "300 over the 30 dilemmas).",
    )
    parser.add_argument(
        "--samples-per-model",
        type=int,
        default=10,
        help="Most runs of each model used per dilemma in the survey subset "
        "(the candidate pool). The complete subset always uses every run.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the survey subset's draw."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=COMPARISONS_DIR,
        help="Directory the run's comparison subsets are written into.",
    )
    parser.add_argument(
        "--no-complete",
        action="store_true",
        help="Skip the complete subset (every possible pair).",
    )
    parser.add_argument(
        "--complete-scope",
        choices=["selected", "all-models"],
        default="selected",
        help="Which pairs the complete subset holds: the two selected models "
        "(default) or every model pair in the run.",
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List the full-run bundles and their models, then exit.",
    )
    return parser.parse_args(argv)


def list_runs() -> None:
    """Print every complete bundle with its models and export path."""
    bundles = complete_bundles()
    if not bundles:
        print("no complete full-run bundles")
        return
    for run_id, directory in bundles:
        print(f"\n{run_id}  ({directory})")
        manifest_path = FULL_RUNS_DIR / run_id / "index.json"
        if not manifest_path.exists():
            print("  (not exported yet)")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"  runs: {len(manifest.get('runs', []))}")
        for model in manifest.get("models", []):
            print(f"    {model}")


def main() -> None:
    args = parse_args()
    if args.list_runs:
        list_runs()
        return

    run_id, export_dir = resolve_run(args.run_id)
    manifest = json.loads((export_dir / "index.json").read_text(encoding="utf-8"))
    available = list(manifest.get("models", []))
    if not available:
        raise SystemExit(f"the run {run_id} has no models in its export")

    model_a = resolve_model(args.model_a, available)
    model_b = resolve_model(args.model_b, available)
    if model_a == model_b:
        raise SystemExit("--model-a and --model-b must be two different models")

    grouped = group_runs(manifest)
    profile_files = load_profiles()
    out_root = args.output_dir / run_id

    # The survey subset: a seeded draw of pairs per dilemma.
    survey_records = build_comparisons(
        manifest,
        export_dir,
        grouped,
        [(model_a, model_b)],
        args.per_dilemma,
        args.samples_per_model,
        args.seed,
    )
    meta = {
        "run_id": run_id,
        "models": {"model_a": model_a, "model_b": model_b},
        "seed": args.seed,
        "per_dilemma": args.per_dilemma,
    }
    dilemmas = [
        dilemma_meta(manifest, dilemma_id, profile_files)
        for dilemma_id in sorted(grouped)
    ]
    write_subset(
        out_root / "survey",
        "survey",
        survey_records,
        dilemmas,
        meta,
    )
    per_dilemma_counts = defaultdict(int)
    for record in survey_records:
        per_dilemma_counts[record["dilemma_id"]] += 1
    print(
        f"survey:   {len(survey_records)} comparisons over "
        f"{len(per_dilemma_counts)} dilemma(s) "
        f"({args.per_dilemma} per dilemma requested)"
    )
    short = {
        dilemma_id: count
        for dilemma_id, count in per_dilemma_counts.items()
        if count < args.per_dilemma
    }
    if short:
        print(
            f"          short of {args.per_dilemma} (fewer runs to pair): "
            + ", ".join(f"{key} ({value})" for key, value in sorted(short.items()))
        )

    # The complete subset: every possible pair.
    if not args.no_complete:
        if args.complete_scope == "all-models":
            model_pairs = [
                (first, second)
                for i, first in enumerate(sorted(available))
                for second in sorted(available)[i + 1 :]
            ]
        else:
            model_pairs = [(model_a, model_b)]
        complete_records = build_comparisons(
            manifest,
            export_dir,
            grouped,
            model_pairs,
            None,
            None,  # every run of every model: all the pairs there are
            args.seed,
        )
        write_subset(
            out_root / "complete",
            "complete",
            complete_records,
            dilemmas,
            {
                **meta,
                "per_dilemma": None,
                "models": {
                    "model_a": model_a,
                    "model_b": model_b,
                    "pairs": [list(pair) for pair in model_pairs],
                },
            },
        )
        print(
            f"complete: {len(complete_records)} comparisons over "
            f"{len(model_pairs)} model pair(s)"
        )

    print(f"\nwritten to {out_root}/")


if __name__ == "__main__":
    main()
