"""Build the comparison pools between the models' dilemma responses.

Reads the exported runs of one full-run bundle (``output/runs/full/<run-id>/``,
written by ``export_results.py``) and pairs runs **on the same dilemma**. The
pools are written per fine-tune against the base model:

    output/comparisons/<run-id>/complete/<model>/comparisons.json
                                                  /index.csv

where ``<model>`` is the fine-tune's short label (the part after ``-nz-wvs-``,
e.g. ``modal_response-overall``). Each pool holds every possible pair of the
base model's runs and that fine-tune's runs on a dilemma.

``--complete-scope selected`` keeps the older single-pool layout
(``complete/comparisons.json`` for one ``--model-b``) and ``all-models``
writes one ``complete/<model-a>__<model-b>/`` pool per base/fine-tune pair.

Pairing: for every dilemma, every run of one model is paired with every run
of the other. The pair is stored by role - ``base`` is the non-adapter model
and ``ft`` is the ``-nz-wvs-`` fine-tune - not as Agent 1/Agent 2. The roles
are validated before a record is written; the survey assigns the two agents
to the roles per respondent with a flow randomizer.

Usage:

    uv run python code/dilemmas/generate_comparisons.py
    uv run python code/dilemmas/generate_comparisons.py --run-id 2026-10-19_d0a90383
    uv run python code/dilemmas/generate_comparisons.py \\
        --complete-scope selected \\
        --model-a openai/Qwen/Qwen3.8-27B \\
        --model-b openai/Qwen3.8-27B-nz-wvs-first_token_distribution-overall
    uv run python code/dilemmas/generate_comparisons.py --list-runs
"""

from __future__ import annotations

import argparse
import csv
import json
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

# The default pair: the untuned Qwen3.8-27B base model against the modal
# response fine-tune (the adapter the survey currently uses).
DEFAULT_MODEL_A = "openai/Qwen/Qwen3.8-27B"
DEFAULT_MODEL_B = "openai/Qwen3.8-27B-nz-wvs-modal_response-overall"

# The fine-tune id marker used by the served adapters.
FT_MARKER = "-nz-wvs-"

# The comparison manifest schema tag. Bump when the layout changes so the
# survey / analysis code can tell versions apart.
COMPARISON_SCHEMA = "wvs-dilemma-comparisons/v2"

# One JSON manifest per subset, plus a flat CSV of the same rows.
MANIFEST_NAME = "comparisons.json"
CSV_NAME = "index.csv"


# --- Small helpers -----------------------------------------------------------


def slug(value: str) -> str:
    """Make a model id safe as a path component (the export's own rule)."""
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip(".") or "unknown"


def model_label(model_id: str) -> str:
    """A short directory label for a model.

    An adapter drops the base model and training prefix, so
    ``openai/Qwen3.8-27B-nz-wvs-modal_response-overall`` becomes
    ``modal_response-overall``. The base model keeps its own name, and a
    comparison between two non-base models joins the two labels with ``__``.
    """
    name = model_id.rsplit("/", 1)[-1]
    short = name.split(FT_MARKER, 1)[1] if FT_MARKER in name else name
    return slug(short)


def is_fine_tune_model(model_id: str) -> bool:
    """Whether a served model id is a ``-nz-wvs-`` fine-tune adapter."""
    return FT_MARKER in (model_id or "").rsplit("/", 1)[-1]


def base_ft_pair(first: str, second: str) -> tuple[str, str] | None:
    """Orient a pair as ``(base, fine-tune)``, or None when it is not one.

    The base model is the served model without the adapter marker. A pair
    with two base models or two fine-tunes has no base/fine-tune roles, so it
    must not be written as a survey comparison.
    """
    first_is_base = not is_fine_tune_model(first)
    second_is_base = not is_fine_tune_model(second)
    if first_is_base == second_is_base:
        return None
    return (first, second) if first_is_base else (second, first)


def require_base_ft_pair(first: str, second: str) -> tuple[str, str]:
    """Orient a pair as ``(base, fine-tune)`` or fail with a clear message."""
    pair = base_ft_pair(first, second)
    if pair is None:
        raise SystemExit(
            "base/ft roles are undefined for "
            f"{first!r} and {second!r}: exactly one side must be the "
            f"non-adapter base model (ids with {FT_MARKER!r} are fine-tunes)"
        )
    return pair


def validate_comparison_roles(record: dict) -> None:
    """Fail when a comparison's ``base``/``ft`` roles are not model-correct.

    ``base`` must be the non-adapter model and ``ft`` must be a
    ``-nz-wvs-`` adapter. This catches manifests whose pair was built in the
    wrong order (which would otherwise put an adapter in the base column in
    every downstream table and in the survey).
    """
    comparison_id = record.get("comparison_id", "<unknown>")
    agents = record.get("agents") or {}
    base = agents.get("base") or {}
    ft = agents.get("ft") or {}
    if not base or not ft:
        raise ValueError(f"{comparison_id}: agents must contain base and ft")
    base_model = base.get("model") or ""
    ft_model = ft.get("model") or ""
    if not base_model or not ft_model:
        raise ValueError(f"{comparison_id}: base/ft model id is missing")
    if is_fine_tune_model(base_model):
        raise ValueError(f"{comparison_id}: base side is a fine-tune ({base_model!r})")
    if not is_fine_tune_model(ft_model):
        raise ValueError(f"{comparison_id}: ft side is not a fine-tune ({ft_model!r})")


def annotate_epochs(manifest: dict, export_dir: Path) -> None:
    """Add the run's inspect ``epoch`` to every manifest row when missing.

    ``version`` is the export's chronological run number and is kept for
    display/path compatibility, but pairing uses ``epoch``. Older exports do
    not carry ``epoch`` in ``index.json``, so read it back from each run's
    ``config.json`` once.
    """
    missing = 0
    for run in manifest.get("runs") or []:
        if run.get("epoch") is not None:
            continue
        config_path = export_dir / str(run.get("path", "")) / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            missing += 1
            continue
        if config.get("epoch") is not None:
            run["epoch"] = config["epoch"]
        else:
            missing += 1
    if missing:
        print(
            f"WARNING: {missing} run(s) have no epoch in index.json or "
            "config.json; pairing will fall back to the export version"
        )


def write_json(path: Path, value: object) -> None:
    """Write pretty JSON, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


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
    others out. ``samples_per_model`` None uses every run - the pool does
    that, so it holds all the pairs there are.
    """
    a_runs = runs_a if samples_per_model is None else runs_a[:samples_per_model]
    b_runs = runs_b if samples_per_model is None else runs_b[:samples_per_model]
    return [(a, b) for a in a_runs for b in b_runs]


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
    """One comparison: the pair's runs in fixed base/ft role order.

    ``pair`` is ``(base_run, ft_run)`` - the pool's ``model_a`` run first,
    the fine-tune second - so the record keys the two sides by role. No
    Agent 1 / Agent 2 assignment happens here; the survey draws that per
    respondent.

    The record holds the survey block's keys (``comparison_id``, ``label``,
    ``role``, ``dilemma_intro``, ``dilemma_outro``, ``situation``,
    ``question``, ``responses``) plus the provenance (run ids, models, the
    judged answer options).
    """
    roles = ("base", "ft")
    runs = list(pair)
    responses = [run_response(export_dir, run) for run in runs]
    if not all(responses):
        return None  # nothing to show a respondent
    choices = [run_choice(export_dir, run) for run in runs]
    versions = [run.get("version", 0) for run in runs]
    if versions[0] == versions[1]:
        label = f"{dilemma['profile_id']} - {dilemma['situation_id']} v{versions[0]}"
    else:
        label = (
            f"{dilemma['profile_id']} - {dilemma['situation_id']} "
            f"v{versions[0]}/{versions[1]}"
        )
    record = {
        "comparison_id": f"{runs[0]['run_id']}__{runs[1]['run_id']}",
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
        "responses": {"base": responses[0], "ft": responses[1]},
        # Provenance: which run (and model) is the base / fine-tune side.
        "agents": {
            role: {
                "run_id": run["run_id"],
                "model": run["model"],
                "choice": choices[index],
                "response": responses[index],
                "version": versions[index],
                "epoch": run.get("epoch"),
                "run_path": run["path"],
            }
            for index, (role, run) in enumerate(zip(roles, runs))
        },
        "same_choice": bool(choices[0]) and choices[0] == choices[1],
    }
    validate_comparison_roles(record)
    return record


def build_comparisons(
    manifest: dict,
    export_dir: Path,
    grouped: dict[str, dict[str, list[dict]]],
    model_pairs: list[tuple[str, str]],
    samples_per_model: int | None,
) -> list[dict]:
    """The comparisons for the given model pairs.

    ``samples_per_model`` None uses every run of each model, which is the
    pool's rule: every pair there is.
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
            for pair in sorted(
                pool,
                key=lambda pair: (
                    pair[0].get("version", 0),
                    pair[1].get("version", 0),
                    pair[0]["run_id"],
                    pair[1]["run_id"],
                ),
            ):
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
        "base_run_id",
        "base_model",
        "base_epoch",
        "base_choice",
        "ft_run_id",
        "ft_model",
        "ft_epoch",
        "ft_choice",
        "same_choice",
    ]
    with (out_dir / CSV_NAME).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            base = record["agents"]["base"]
            ft = record["agents"]["ft"]
            writer.writerow(
                {
                    "comparison_id": record["comparison_id"],
                    "dilemma_id": record["dilemma_id"],
                    "profile_id": record["profile_id"],
                    "situation_id": record["situation_id"],
                    "label": record["label"],
                    "base_run_id": base["run_id"],
                    "base_model": base["model"],
                    "base_epoch": base.get("epoch"),
                    "base_choice": base["choice"],
                    "ft_run_id": ft["run_id"],
                    "ft_model": ft["model"],
                    "ft_epoch": ft.get("epoch"),
                    "ft_choice": ft["choice"],
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
        default=None,
        help="The second model of the pair for --complete-scope selected "
        f"(default: {DEFAULT_MODEL_B}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=COMPARISONS_DIR,
        help="Directory the run's comparison pools are written into.",
    )
    parser.add_argument(
        "--complete-scope",
        choices=["per-model", "selected", "all-models"],
        default="per-model",
        help="Which model pairs the pools hold: one pool per fine-tune "
        "against the base model, under complete/<model>/ (default); the two "
        "selected models in the single complete/ pool; or every base-vs-"
        "fine-tune pair, under complete/<model-a>__<model-b>/.",
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
    annotate_epochs(manifest, export_dir)

    model_a = resolve_model(args.model_a, available)
    base_models = [model for model in available if not is_fine_tune_model(model)]
    if model_a not in base_models:
        raise SystemExit(
            f"--model-a {model_a!r} is not the base model; base model(s) in "
            f"this run: {base_models or 'none'}. Fine-tune ids carry "
            f"{FT_MARKER!r}."
        )

    grouped = group_runs(manifest)
    profile_files = load_profiles()
    out_root = args.output_dir / run_id
    dilemmas = [
        dilemma_meta(manifest, dilemma_id, profile_files)
        for dilemma_id in sorted(grouped)
    ]

    # The pools to write: (directory name, the model pairs each holds). The
    # base comparison layout is one pool per fine-tune, so the analysis can
    # walk the models one at a time.
    if args.complete_scope == "per-model":
        others = [
            model
            for model in sorted(available)
            if is_fine_tune_model(model) and model != model_a
        ]
        if not others:
            raise SystemExit(
                f"the run {run_id} has no fine-tune model other than {model_a}"
            )
        pools = [(model_label(model), [(model_a, model)]) for model in others]
        nested = True
    elif args.complete_scope == "all-models":
        ordered = sorted(available)
        pairs = []
        skipped = []
        for index, first in enumerate(ordered):
            for second in ordered[index + 1 :]:
                pair = base_ft_pair(first, second)
                if pair is None:
                    skipped.append((first, second))
                else:
                    pairs.append(pair)
        if not pairs:
            raise SystemExit(
                "all-models found no base-vs-fine-tune pair; "
                "check the model ids and the fine-tune marker"
            )
        if skipped:
            print(
                f"all-models: skipped {len(skipped)} pair(s) without exactly "
                f"one base and one fine-tune (e.g. {skipped[0][0]!r} vs "
                f"{skipped[0][1]!r})"
            )
        pools = [
            (f"{model_label(first)}__{model_label(second)}", [(first, second)])
            for first, second in pairs
        ]
        nested = True
    else:  # selected: the legacy single pool for one model pair
        model_b = resolve_model(args.model_b or DEFAULT_MODEL_B, available)
        if model_a == model_b:
            raise SystemExit("--model-a and --model-b must be two different models")
        model_a, model_b = require_base_ft_pair(model_a, model_b)
        pools = [("complete", [(model_a, model_b)])]
        nested = False

    total = 0
    written: list[str] = []
    for name, model_pairs in pools:
        records = build_comparisons(
            manifest,
            export_dir,
            grouped,
            model_pairs,
            None,  # every run of every model: all the pairs there are
        )
        if not records:
            print(f"WARNING: no comparisons for {name}; pool not written")
            continue
        first_model, second_model = model_pairs[0]
        out_dir = out_root / "complete" / name if nested else out_root / "complete"
        write_subset(
            out_dir,
            name,
            records,
            dilemmas,
            {
                "run_id": run_id,
                "label": name,
                "models": {
                    # Role names the records use (base = first model of
                    # the pool, ft = second), plus the raw pair.
                    "base": first_model,
                    "ft": second_model,
                    "model_a": first_model,
                    "model_b": second_model,
                    "pairs": [list(pair) for pair in model_pairs],
                },
            },
        )
        total += len(records)
        written.append(name)
        location = f"complete/{name}" if nested else "complete"
        print(f"{location}: {len(records)} comparisons")

    if not written:
        raise SystemExit("no comparison pools were written")
    print(
        f"\n{total} comparisons over {len(written)} pool(s), "
        f"{len(grouped)} dilemma(s), written to {out_root}/complete/"
    )
    print(
        "\nNext: embed the pools (embed_responses.py, which walks every "
        "complete/<model>/ by default), then run the analysis notebook "
        "(analyze.py)."
    )


if __name__ == "__main__":
    main()
