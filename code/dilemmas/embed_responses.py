"""Embed the dilemma responses with a local embedding model.

Reads the comparison pools written by ``generate_comparisons.py`` and embeds
**every distinct response they hold** - the whole pool, not a subset - caching
the vectors next to the comparisons:

    output/comparisons/<run-id>/embeddings/complete/<model>/vectors.npz
                                                            /meta.json

The default ``--subset complete`` walks every per-model pool under
``complete/<model>/`` and embeds each into the matching embeddings directory
(``--subset complete/<model>`` embeds just one). The model is loaded once for
the whole run.

Embedding the whole pool is what makes the survey design possible: the
distance between two answers is only known once both have been embedded, and
the analysis notebook filters the pool on that distance afterwards. The cache
holds one vector per distinct response (the runs), not one per comparison (the
pairs), so a 750-comparison base-vs-model pool needs 300 vectors.

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

The model: ``Qwen/Qwen3-Embedding-4B`` (Apache-2.0, 2560 dimensions, 32k
context). On the MTEB English v2 suite it scores 88.72 on semantic textual
similarity - the task this analysis is - against 88.58 for the 8B, so the
extra parameters of the 8B buy nothing here. The weights load in bfloat16 by
default on a GPU (about 8 GB rather than 16 GB in float32, and the published
MTEB scores use the same precision); ``--dtype`` overrides that. The vectors
are re-normalised in float32 after encoding, so every vector is exactly unit
length whichever dtype runs the model. ``--model`` switches to a smaller
model when a run has to fit a smaller machine, and the script warns loudly
rather than quietly falling back when no GPU is available.

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
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMPARISONS_DIR = HERE / "output" / "comparisons"

DEFAULT_MODEL = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_SCHEMA = "wvs-dilemma-embeddings/v1"


# --- Locating the comparisons ------------------------------------------------


def subset_targets(run_dir: Path, subset: str) -> list[tuple[str, Path]]:
    """(subset name, manifest path) for every pool under one run's subset.

    A subset is either one pool (``complete`` or ``complete/<model>``) with a
    ``comparisons.json``, or the parent of per-model pools (``complete``)
    whose subdirectories each hold one. Per-model subdirectories win over a
    leftover top-level ``complete/comparisons.json``, so a single pool left by
    an older version is not embedded a second time.
    """
    root = run_dir / subset
    if not root.is_dir():
        return []
    nested = sorted(
        child
        for child in root.iterdir()
        if child.is_dir() and (child / "comparisons.json").exists()
    )
    if nested:
        return [
            (f"{subset}/{child.name}", child / "comparisons.json") for child in nested
        ]
    if (root / "comparisons.json").exists():
        return [(subset, root / "comparisons.json")]
    return []


def resolve_targets(
    run_id: str | None, subset: str
) -> tuple[str, list[tuple[str, Path]]]:
    """The run to embed and every (subset name, manifest path) in it."""
    if run_id:
        run_dir = COMPARISONS_DIR / run_id
        targets = subset_targets(run_dir, subset)
        if not targets:
            raise SystemExit(
                f"no {subset} comparisons for {run_id} under {run_dir} - run "
                "generate_comparisons.py first"
            )
        return run_id, targets

    if COMPARISONS_DIR.is_dir():
        legacy_fallback = None
        for directory in sorted(COMPARISONS_DIR.iterdir(), reverse=True):
            if not directory.is_dir():
                continue
            targets = subset_targets(directory, subset)
            if not targets:
                continue
            # A run with per-model pools wins over a newer single-pool run,
            # so a scratch single-pool test does not hide the main run.
            if any(name != subset for name, _ in targets):
                return directory.name, targets
            if legacy_fallback is None:
                legacy_fallback = (directory.name, targets)
        if legacy_fallback is not None:
            return legacy_fallback
    raise SystemExit(
        f"no {subset} comparisons under {COMPARISONS_DIR} - run "
        "generate_comparisons.py first"
    )


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


def select_device(requested: str | None) -> str:
    """The device a run would use: the explicit choice, else CUDA when visible."""
    import torch

    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise SystemExit(
                f"--device {requested} was requested, but torch cannot see a "
                "CUDA GPU (torch.cuda.is_available() is False). Check the "
                "driver, CUDA_VISIBLE_DEVICES and that torch was installed "
                "with CUDA support, or pass --device cpu to accept the CPU."
            )
        return requested

    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_dtype(requested: str, device: str) -> str:
    """The dtype to load the model in, low precision by default on a GPU.

    The 4B weights are about 16 GB in float32, which does not fit a shared
    GPU; bfloat16 halves that to about 8 GB. It is also the dtype the
    Qwen3-Embedding checkpoints and their published MTEB scores use, so it is
    the benchmarked configuration rather than a compromise. The vectors are
    re-normalised in float32 after encoding, so the unit-norm requirement
    still holds exactly.
    """
    import torch

    if requested != "auto":
        return requested
    if device.startswith("cuda"):
        return "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    return "float32"


def warn_if_cpu_fallback(device: str, requested: str | None, model_id: str) -> None:
    """A loud banner when the CPU was selected automatically.

    The GPU is what makes this step quick: a full pool takes seconds on a GPU
    against minutes to hours on a CPU, so a fallback is easy to miss and
    expensive. An explicit ``--device`` is taken as the user's choice and
    stays quiet.
    """
    if device != "cpu" or requested:
        return

    print(
        f"""
{"!" * 78}
WARNING: no CUDA GPU found - embedding on the CPU
{"!" * 78}
  model:  {model_id}
  device: cpu (automatic fallback because --device was not given)

  A CPU pass is roughly 10-50x slower than a GPU pass: the 4B model is
  about 16 GB of weights in float32 and the full pool can take minutes to
  hours on the CPU where a GPU takes seconds.

  If this machine should have a GPU, check:
    nvidia-smi                         is a GPU visible to the driver?
    echo $CUDA_VISIBLE_DEVICES         is the GPU masked out?
    uv run python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
  torch.version.cuda is None when torch was installed without CUDA.

  Pass --device cpu to take the CPU deliberately (and silence this warning),
  or --model Qwen/Qwen3-Embedding-0.6B for a model that fits a small machine.
{"!" * 78}
""",
        file=sys.stderr,
    )


def _out_of_memory(model_id: str, dtype: str, batch_size: int | None) -> SystemExit:
    """The error raised when the GPU cannot hold the model or a batch."""
    where = f" at batch size {batch_size}" if batch_size else ""
    return SystemExit(
        "\nERROR: the GPU ran out of memory loading or running "
        f"{model_id} in {dtype}{where}.\n"
        "The 4B model takes about 8 GB in bfloat16 (16 GB in float32) "
        "before activations. Lower --batch-size, use a larger GPU, or "
        "fall back to a smaller model (--model Qwen/Qwen3-Embedding-0.6B)."
    )


def load_model(model_id: str, device: str, dtype: str):
    """Load the sentence-transformer once, in the requested precision."""
    import torch
    from sentence_transformers import SentenceTransformer

    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    try:
        return SentenceTransformer(
            model_id, device=device, model_kwargs={"torch_dtype": torch_dtype}
        )
    except torch.cuda.OutOfMemoryError as error:
        raise _out_of_memory(model_id, dtype, batch_size=None) from error


def embed_texts(
    model,
    texts: list[str],
    model_id: str,
    dtype: str,
    batch_size: int,
    instruction: str | None = None,
) -> np.ndarray:
    """The L2-normalised embeddings of the texts, as float32.

    With an instruction, each text is prefixed with the model's own
    instruction format (``Instruct: <task>\\nQuery:<text>``). The task is
    symmetric, so both sides of a comparison get the same prefix.
    """
    import torch

    if instruction:
        texts = [f"Instruct: {instruction}\nQuery:{text}" for text in texts]
    try:
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
    except torch.cuda.OutOfMemoryError as error:
        raise _out_of_memory(model_id, dtype, batch_size) from error
    # encode() normalised in the model's own dtype, so redo it in float32:
    # bfloat16 rounds the norm enough to make a vector's dot with itself
    # 1.004 rather than 1.
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)


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
        help="The comparison pool to embed: 'complete' walks every per-model "
        "pool under complete/<model>/ (default); a single pool such as "
        "'complete/modal_response-overall' embeds only that one.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Embedding model (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run the model on (default: cuda when available, "
        "otherwise cpu with a loud warning). Pass 'cpu' to accept a CPU run "
        "silently.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "bfloat16", "float16"),
        default="auto",
        help="Model precision (default: auto - bfloat16 on a CUDA GPU, "
        "float32 on the CPU).",
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


def cache_current(
    cached: dict | None,
    *,
    model: str,
    revision: str,
    dtype: str,
    instruction: str | None,
    include_question: bool,
    content: str,
) -> bool:
    """Whether a cached pool was built with exactly this configuration."""
    return bool(
        cached
        and cached.get("model") == model
        and cached.get("revision") == revision
        and cached.get("instruction") == instruction
        and cached.get("include_question") == include_question
        and cached.get("dtype", "float32") == dtype
        and cached.get("fingerprint") == content
    )


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    run_id, targets = resolve_targets(args.run_id, args.subset)
    revision = model_revision(args.model)
    tag = cache_tag(args.instruction, args.include_question)

    jobs = []
    pending = []
    for subset, manifest_path in targets:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        texts = response_texts(manifest, args.include_question)
        if not texts:
            print(f"WARNING: no responses to embed in {manifest_path}; skipped")
            continue
        cache_dir = cache_dir_for(run_id, subset, tag)
        cached = read_cache_meta(cache_dir)
        job = {
            "subset": subset,
            "manifest_path": manifest_path,
            "texts": texts,
            "run_ids": sorted(texts),
            "content": fingerprint(texts),
            "cache_dir": cache_dir,
        }
        jobs.append(job)
        if not args.force and cache_current(
            cached,
            model=args.model,
            revision=revision,
            dtype=dtype,
            instruction=args.instruction,
            include_question=args.include_question,
            content=job["content"],
        ):
            print(
                f"Cache is current: {cache_dir / 'vectors.npz'} "
                f"({cached.get('n_texts')} texts, {cached.get('dimensions')} dims, "
                f"model {args.model} @ {revision[:8]}, {dtype})"
            )
            continue
        if cached and (
            cached.get("model") not in (None, args.model)
            or cached.get("dtype", "float32") != dtype
        ):
            print(
                f"WARNING: {cache_dir / 'vectors.npz'} was built with "
                f"{cached.get('model')} in {cached.get('dtype', 'float32')}; "
                f"re-embedding with {args.model} in {dtype} replaces it. The "
                "metadata records the model and dtype, so this is a clean rerun.",
                file=sys.stderr,
            )
        pending.append(job)

    if not jobs:
        raise SystemExit(f"no responses to embed under {COMPARISONS_DIR / run_id}")
    if not pending:
        return

    # Only now, so cache hits never warn about a GPU they did not need.
    warn_if_cpu_fallback(device, args.device, args.model)
    print(
        f"Embedding {len(pending)} pool(s) from {run_id} with {args.model} "
        f"({dtype}) on {device}..."
    )
    embedder = load_model(args.model, device, dtype)
    for job in pending:
        print(f"  {job['subset']}: {len(job['run_ids'])} distinct responses")
        vectors = embed_texts(
            embedder,
            [job["texts"][run_id_] for run_id_ in job["run_ids"]],
            args.model,
            dtype,
            args.batch_size,
            args.instruction,
        )
        write_cache(
            job["cache_dir"],
            job["subset"],
            job["run_ids"],
            vectors,
            {
                "run_id": run_id,
                "model": args.model,
                "revision": revision,
                "dtype": dtype,
                "instruction": args.instruction,
                "include_question": args.include_question,
                "fingerprint": job["content"],
                "device": device,
            },
        )
        print(
            f"  wrote {job['cache_dir'] / 'vectors.npz'} "
            f"({vectors.shape[0]} x {vectors.shape[1]})"
        )


if __name__ == "__main__":
    main()
