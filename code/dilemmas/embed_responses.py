"""Embed the dilemma responses with a local embedding model.

Reads a comparisons manifest written by ``generate_comparisons.py`` and
embeds every distinct response it holds with Qwen3-Embedding-0.6B, caching
the vectors next to the comparisons:

    output/comparisons/<run-id>/embeddings/<subset>.npz   the vectors
    output/comparisons/<run-id>/embeddings/<subset>.json  the metadata

The cached metadata records the model, its Hugging Face revision, the
instruction used and a fingerprint of the embedded texts, so a later run can
tell whether the cache is still valid (it re-embeds when it is not) and so
the analysis can state exactly which embeddings it read.

Why an embedding model rather than the LLM judge: the judge already gives a
categorical answer option per response, which is coarse - two responses can
pick the same option and still differ in what they say, and two responses
can pick different options for reasons the embedding would call "close". The
embedding gives a continuous distance, so the analysis can measure *how far
apart* two answers are, not just *whether they differ*. It is also free: the
model runs locally on the GPU, so no response ever leaves the machine.

The model: ``Qwen/Qwen3-Embedding-0.6B`` (Apache-2.0, 1024 dimensions, 32k
context). On the MTEB English v2 suite it scores 86.57 on semantic textual
similarity - the task this analysis is - against 88.72 for the 4B and 88.58
for the 8B, so the small model gives up about two points of similarity
quality for a fraction of the compute and memory.

The task is symmetric (both sides of a comparison are answers to the same
dilemma), so no instruction prefix is used by default: that is how the STS
benchmarks are scored. ``--instruction`` adds a task instruction to both
sides if a run wants to test the alternative.

Usage:

    uv run python code/dilemmas/embed_responses.py
    uv run python code/dilemmas/embed_responses.py --run-id 2026-10-19_d0a90383 --subset complete
    uv run python code/dilemmas/embed_responses.py --instruction "Represent the answer's meaning" --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMPARISONS_DIR = HERE / "output" / "comparisons"

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_SCHEMA = "wvs-dilemma-embeddings/v1"


# --- Locating the comparisons ------------------------------------------------


def comparison_dirs(comparisons_dir: Path, subset: str) -> list[tuple[str, Path]]:
    """(run_id, manifest path) for every run with the subset, newest first."""
    found = []
    if comparisons_dir.is_dir():
        for directory in sorted(comparisons_dir.iterdir()):
            manifest = directory / subset / "comparisons.json"
            if directory.is_dir() and manifest.exists():
                found.append((directory.name, manifest))
    found.sort(key=lambda item: item[0], reverse=True)
    return found


def resolve_manifest(run_id: str | None, subset: str) -> tuple[str, Path]:
    """The comparisons manifest to embed."""
    if run_id:
        manifest = COMPARISONS_DIR / run_id / subset / "comparisons.json"
        if not manifest.exists():
            raise SystemExit(
                f"no {subset} comparisons for {run_id} at {manifest} - run "
                "generate_comparisons.py first"
            )
        return run_id, manifest
    found = comparison_dirs(COMPARISONS_DIR, subset)
    if not found:
        raise SystemExit(
            f"no {subset} comparisons under {COMPARISONS_DIR} - run "
            "generate_comparisons.py first"
        )
    return found[0]


# --- The texts ---------------------------------------------------------------


def response_texts(manifest: dict, include_question: bool) -> dict[str, str]:
    """run id -> the text to embed.

    The response alone by default: the distance then measures how far apart
    the two *answers* are, not how similar the two texts are because they
    answer the same question. ``include_question`` prefixes the dilemma
    question instead, which measures the answer-plus-context instead.
    """
    questions = {
        f"{d['profile_id']}-{d['situation_id']}": d.get("question", "")
        for d in manifest.get("dilemmas", [])
    }
    texts: dict[str, str] = {}
    for record in manifest.get("comparisons", []):
        question = questions.get(record.get("dilemma_id", ""), "")
        for agent in record.get("agents", {}).values():
            text = (agent.get("response") or "").strip()
            if not text:
                continue
            if include_question and question:
                text = f"{question}\n\n{text}"
            texts[agent["run_id"]] = text
    return texts


def fingerprint(texts: dict[str, str]) -> str:
    """A digest of the embedded texts (the cache key for the content)."""
    digest = hashlib.sha256()
    for run_id in sorted(texts):
        digest.update(run_id.encode())
        digest.update(b"\x00")
        digest.update(texts[run_id].encode())
        digest.update(b"\x00")
    return digest.hexdigest()


def model_revision(model_id: str) -> str:
    """The model's current Hugging Face commit, for reproducibility.

    Best effort: the analysis falls back to 'unknown' when the hub is not
    reachable.
    """
    try:
        with urllib.request.urlopen(
            f"https://huggingface.co/api/models/{model_id}", timeout=10
        ) as response:
            return json.loads(response.read()).get("sha", "unknown")
    except Exception:  # noqa: BLE001 - the revision is a nicety, not a need
        return "unknown"


def cache_tag(instruction: str | None, include_question: bool) -> str:
    """A short tag for a non-default configuration.

    The default configuration (no instruction, response text only) is cached
    under the subset's own name; any other setting gets its own directory,
    so a run with an instruction never overwrites the default embeddings.
    """
    if instruction is None and not include_question:
        return ""
    settings = json.dumps(
        {"instruction": instruction, "include_question": include_question},
        sort_keys=True,
    )
    return "-" + hashlib.sha256(settings.encode()).hexdigest()[:8]


def cache_dir_for(run_id: str, subset: str, tag: str) -> Path:
    """Where a configuration's embeddings are cached."""
    return COMPARISONS_DIR / run_id / "embeddings" / f"{subset}{tag}"


# --- Embedding ---------------------------------------------------------------


def embed_texts(
    texts: list[str],
    model_id: str,
    device: str,
    batch_size: int,
    instruction: str | None = None,
) -> np.ndarray:
    """The L2-normalised embeddings of the texts, as float32.

    With an instruction, each text is prefixed with the model's own
    instruction format (``Instruct: <task>\\nQuery:<text>``). The task is
    symmetric, so both sides of a comparison get the same prefix.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_id, device=device)
    # float32 throughout: the model's own dtype is bfloat16, which rounds
    # the normalised vectors enough to make a vector's dot with itself
    # 1.004 rather than 1.
    model = model.float()
    if instruction:
        texts = [f"Instruct: {instruction}\nQuery:{text}" for text in texts]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def write_cache(
    cache_dir: Path,
    subset: str,
    run_ids: list[str],
    vectors: np.ndarray,
    meta: dict,
) -> None:
    """Write the vectors and their metadata."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_dir / "vectors.npz", vectors=vectors, run_ids=run_ids)
    (cache_dir / "meta.json").write_text(
        json.dumps(
            {
                "schema": EMBEDDING_SCHEMA,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "subset": subset,
                **meta,
                "n_texts": len(run_ids),
                "dimensions": int(vectors.shape[1]),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def read_cache_meta(cache_dir: Path) -> dict | None:
    """The cached metadata, or None when there is no cache."""
    path = cache_dir / "meta.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


# --- CLI ---------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run-id", default=None, help="Comparisons to embed (default: newest)."
    )
    parser.add_argument(
        "--subset",
        default="complete",
        help="Comparison subset to embed (default: complete).",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run the model on (default: cuda when available).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Encoding batch size."
    )
    parser.add_argument(
        "--instruction",
        default=None,
        help="Task instruction prefixed to both sides (default: none, which "
        "is how the semantic-similarity benchmarks are scored).",
    )
    parser.add_argument(
        "--include-question",
        action="store_true",
        help="Prefix the dilemma question to each response before embedding.",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-embed even if the cache is valid."
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    run_id, manifest_path = resolve_manifest(args.run_id, args.subset)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    texts = response_texts(manifest, args.include_question)
    if not texts:
        raise SystemExit(f"no responses to embed in {manifest_path}")

    run_ids = sorted(texts)
    content = fingerprint(texts)
    revision = model_revision(args.model)
    tag = cache_tag(args.instruction, args.include_question)
    cache_dir = cache_dir_for(run_id, args.subset, tag)
    cached = read_cache_meta(cache_dir)
    if (
        cached
        and not args.force
        and cached.get("model") == args.model
        and cached.get("revision") == revision
        and cached.get("instruction") == args.instruction
        and cached.get("include_question") == args.include_question
        and cached.get("fingerprint") == content
    ):
        print(
            f"Cache is current: {cache_dir / 'vectors.npz'} "
            f"({cached.get('n_texts')} texts, {cached.get('dimensions')} dims, "
            f"model {args.model} @ {revision[:8]})"
        )
        return

    print(
        f"Embedding {len(run_ids)} distinct responses from "
        f"{manifest_path} with {args.model} on {device}..."
    )
    vectors = embed_texts(
        [texts[run_id] for run_id in run_ids],
        args.model,
        device,
        args.batch_size,
        args.instruction,
    )
    write_cache(
        cache_dir,
        args.subset,
        run_ids,
        vectors,
        {
            "run_id": run_id,
            "model": args.model,
            "revision": revision,
            "instruction": args.instruction,
            "include_question": args.include_question,
            "fingerprint": content,
            "device": device,
        },
    )
    print(
        f"Wrote {cache_dir / 'vectors.npz'} ({vectors.shape[0]} x {vectors.shape[1]})"
    )


if __name__ == "__main__":
    main()
