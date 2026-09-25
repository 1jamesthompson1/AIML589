#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "torch>=2.6.0",
#   "transformers>=4.50.0",
#   "datasets>=5.0.0",
#   "accelerate>=1.6.0",
#   "python-dotenv>=1.1.0",
#   "peft>=0.15.0",
#   "huggingface-hub>=0.30.0",
#   "vllm>=0.19.0",
# ]
# ///
"""
Serve a model via vLLM for evaluation, optionally with LoRA adapters.

Each fine-tuned adapter lives in its own HF Hub repo following the
naming convention ``{model_slug}-nz-wvs-{dataset}-{population}``
(e.g. ``Qwen3.6-27B-nz-wvs-modal_response-cluster_0``).

Adapters whose LoRA rank (from their ``adapter_config.json``) exceeds
``--max-lora-rank`` are skipped with a warning — vLLM refuses to load them
anyway, and one bad adapter would abort server startup. This keeps
mixed-rank collections servable.

Usage:
    # Base model only (no adapters)
    uv run serve.py --model Qwen/Qwen3.6-27B

    # Auto-discover all adapters for this model from a HF Collection
    uv run serve.py --model Qwen/Qwen3.6-27B \
        --hf-collection 1jamesthompson1/wvs-nz-lora-adapters

    # Individual adapters
    uv run serve.py --model Qwen/Qwen3.6-27B \
        --adapter cluster_0=1jamesthompson1/Qwen3.6-27B-nz-wvs-modal_response-cluster_0

    # Remote usage
    scp serve.py user@host:~
    ssh user@host ./serve.py --model Qwen/Qwen3.6-27B \
        --hf-collection 1jamesthompson1/wvs-nz-lora-adapters --hf-token "$HF_TOKEN"
"""

import argparse
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)
for noisy in ("huggingface_hub", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

log = logging.getLogger(__name__)

# How long to wait before the first health probe after launching vLLM.
# Model loading takes minutes (longer with LoRA adapters), and early probes
# just race the bind, showing up as "channel 2: open failed: Connection
# refused" through the tunnel. Keep in sync with run_all.py.
STARTUP_GRACE_S = 180


def discover_adapters_from_collection(
    collection_slug: str, model_slug: str, hf_token: str | None = None
) -> list[tuple[str, str]]:
    """Find all adapter repos in a HF Collection that match the base model.

    Looks for repos named ``{model_slug}-nz-wvs-{dataset}-{population}``
    and returns ``(adapter_name, repo_id)`` tuples where the adapter name
    is derived from the dataset-population suffix.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    prefix = f"{model_slug}-nz-wvs-"
    adapters: list[tuple[str, str]] = []

    try:
        collection = api.get_collection(collection_slug)
    except Exception as e:
        log.warning("[adapters] could not fetch collection %s: %s", collection_slug, e)
        return adapters

    log.info(
        "[adapters] collection %s has %d item(s)",
        collection_slug,
        len(collection.items),
    )

    for item in collection.items:
        if item.item_type != "model":
            continue
        repo_id = item.item_id
        # Extract repo name (last part of repo_id)
        repo_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
        if repo_name.startswith(prefix):
            adapter_name = repo_name
            adapters.append((adapter_name, repo_id))

    return adapters


def get_lora_rank(adapter: str, hf_token: str | None = None) -> int | None:
    """Return the LoRA rank (``r``) of an adapter from its adapter_config.json.

    ``adapter`` may be a local directory or an HF repo id. Returns None if
    the config can't be read (unreachable hub, missing file, etc.).
    """
    from huggingface_hub import hf_hub_download

    try:
        if os.path.isdir(adapter):
            config_path = os.path.join(adapter, "adapter_config.json")
        else:
            config_path = hf_hub_download(
                repo_id=adapter,
                filename="adapter_config.json",
                token=hf_token,
            )
        with open(config_path) as f:
            return json.load(f).get("r")
    except Exception as e:
        log.debug("[adapters] could not read LoRA rank for %s: %s", adapter, e)
        return None


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Serve a fine-tuned model via vLLM for evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model", required=True, help="Base model name or path (HF hub or local)"
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (use 0.0.0.0 for remote access)",
    )
    p.add_argument("--port", type=int, default=8000, help="Port for the vLLM server")
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM GPU memory utilization fraction",
    )
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "auto"],
        help="Model dtype",
    )
    p.add_argument(
        "--gpu",
        default="0",
        help="Comma-separated CUDA device(s) for vLLM, e.g. '0' for a single "
        "GPU or '0,1' to spread across two. 'auto' leaves vLLM to use every "
        "visible GPU. Defaults to a single GPU (set via CUDA_VISIBLE_DEVICES "
        "before the vLLM subprocess starts).",
    )
    p.add_argument(
        "--max-model-len", type=int, default=8196, help="Maximum sequence length"
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading model",
    )
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="Maximum number of sequences per batch. Lower this if OOM with Mamba models.",
    )
    p.add_argument(
        "--enable-auto-tool-choice",
        action="store_true",
        default=True,
        help="Enable tool calling in vLLM. Required for agentic evals "
        "(gpqa_diamond gives the model bash/python/submit tools); without it "
        "vLLM rejects requests with tool_choice='auto' (HTTP 400).",
    )
    p.add_argument(
        "--tool-call-parser",
        default="hermes",
        help="Parser for tool calls in model output. 'hermes' is the format "
        "used by the Qwen3 chat template (see vLLM docs for others).",
    )

    # ── LoRA multi-adapter options ──────────────────────────────────
    p.add_argument(
        "--adapter",
        action="append",
        dest="adapters",
        default=None,
        metavar="NAME=PATH",
        help="LoRA adapter as name=path (repeatable). "
        "Path can be local or a HF repo, "
        "e.g. Qwen3.6-27B-nz-wvs-modal_response-cluster_0=my-org/Qwen3.6-27B-nz-wvs-modal_response-cluster_0",
    )
    p.add_argument(
        "--hf-collection",
        default=None,
        help="HF collection slug to auto-discover adapters from "
        "(e.g. '1jamesthompson1/wvs-nz-lora-adapters'). "
        "Finds all repos matching {model_slug}-nz-wvs-*.",
    )
    p.add_argument(
        "--adapter-suffixes",
        default=None,
        metavar="SUFFIX[,SUFFIX...]",
        help="Only load discovered adapters whose {dataset}-{population} "
        "suffix is listed (e.g. 'modal_response-overall,sampled_response-"
        "cluster_0'). Ignores other adapters in the collection. "
        "Default: all matching adapters.",
    )
    p.add_argument(
        "--max-lora-rank",
        type=int,
        default=None,
        help="Maximum LoRA rank (for multi-LoRA serving). Default: auto-size "
        "to the highest rank among the discovered adapters. An adapter whose "
        "rank exceeds this is silently dropped by vLLM, so the auto default "
        "exists to prevent adapters disappearing at serve time (which makes "
        "their evals 404) — pass an explicit value to pin it.",
    )
    p.add_argument(
        "--no-lora-loader-patch",
        action="store_true",
        default=False,
        help="Skip patching vLLM's LoRA loader. By default serve.py patches "
        "the installed vLLM so PEFT tensor names map onto runtime modules for "
        "multimodal wrapper models (Qwen3.5/3.6/3.8 family). Without the "
        "patch those adapters load but are SILENTLY not applied — generations "
        "are bit-identical to the base model (vLLM issue class: silent LoRA "
        "no-op; see workbench/vllm-lora-prefix-fix for the diagnosis).",
    )
    p.add_argument(
        "--allow-unpatched-lora",
        action="store_true",
        default=False,
        help="If the LoRA loader patch cannot be applied (e.g. vLLM version "
        "drift), warn and continue instead of aborting. Only use this when "
        "you are certain the served base model does not need the prefix "
        "mapping; on multimodal-wrapper models the adapters will be silently "
        "not applied (evaluate.py's adapter check still guards the evals).",
    )

    p.add_argument(
        "vllm_args", nargs="*", help="Extra vLLM args (e.g. --max-num-seqs 500)"
    )

    args, remaining = p.parse_known_args(argv)
    args.vllm_args = remaining + args.vllm_args
    return args


# ── vLLM LoRA loader patch ──────────────────────────────────────

# `_load_adapter` in vllm/lora/worker_manager.py fetches the model's
# hf_to_vllm_mapper for LoRA weight loading. The block shape is stable across
# versions but the mapper method was renamed in vLLM 0.30
# (`get_unstacked_mapper()` in 0.27/0.28 -> `get_rename_mapper()` in 0.30), so
# match it loosely and insert the missing prefix rule right after it.
_PATCH_RE = re.compile(
    r"(?P<indent>[ ]*)hf_to_vllm_mapper = getattr\(model, \"hf_to_vllm_mapper\", None\)\n"
    r"(?P=indent)if hf_to_vllm_mapper is not None:\n"
    r"(?P=indent)[ ]+hf_to_vllm_mapper = hf_to_vllm_mapper\.[A-Za-z_]+\(\)\n"
)


def _lora_prefix_patch(inner: str) -> str:
    """Source text inserted after the mapper lookup (``inner`` = its indent)."""
    return (
        f"{inner}# PATCH (serve.py): map the PEFT text-model prefix onto the\n"
        f"{inner}# runtime module path for multimodal wrapper models. Adapter\n"
        f"{inner}# tensors are named model.layers.* but the runtime language\n"
        f"{inner}# model lives at language_model.model.layers.*; without this\n"
        f"{inner}# rule every LoRA weight lookup misses and the adapter is\n"
        f"{inner}# silently not applied (generations identical to base).\n"
        f"{inner}from vllm.model_executor.models.utils import (\n"
        f"{inner}    WeightsMapper as _ServeWeightsMapper,\n"
        f"{inner})\n"
        f"{inner}\n"
        f"{inner}hf_to_vllm_mapper = hf_to_vllm_mapper | _ServeWeightsMapper(\n"
        f'{inner}    orig_to_new_prefix={{"model.": "language_model.model."}}\n'
        f"{inner})\n"
    )


def patch_vllm_lora_loader() -> bool:
    """Patch the installed vLLM's LoRA weight loader in place.

    Qwen3.5/3.6/3.8 (``qwen3_5`` family) and other multimodal-wrapper models
    expose their language model at ``language_model.model.layers.*`` at
    runtime, while PEFT adapters store ``model.layers.*``. vLLM's
    ``hf_to_vllm_mapper`` lacks the ``model.`` -> ``language_model.model.``
    prefix rule for these models, so every LoRA tensor lookup misses and the
    adapter is SILENTLY not applied! Verified empirically
    on vLLM 0.27/0.28; still broken in vLLM 0.30.0 (the Qwen3-VL mapper has
    rules for ``model.visual.``, ``lm_head.`` and ``model.language_model.``
    but not for the bare ``model.`` prefix). The mapper lookup method was
    renamed (``get_unstacked_mapper()`` -> ``get_rename_mapper()``) in 0.30,
    so the anchor is matched with a loose regex rather than a fixed string.
    See workbench/vllm-lora-prefix-fix.

    This issue https://github.com/vllm-project/vllm/issues/48019 and fix simliar to https://github.com/vllm-project/vllm/pull/49525.

    The patch edits the installed ``vllm/lora/worker_manager.py`` inside the
    current (ephemeral uv) environment. It only affects the LoRA *load*
    mapper, not base weight loading, and is a no-op for models whose adapter
    names already match runtime modules, so it is safe to apply for any
    base model (text-only models never even hit the mapper path).

    Returns:
        True if the patch was applied (or was already present).
    """
    import importlib.util

    spec = importlib.util.find_spec("vllm")
    if spec is None:
        log.error("[patch] vllm not importable — cannot patch LoRA loader.")
        return False
    wm_path = Path(spec.origin).parent / "lora" / "worker_manager.py"
    if not wm_path.exists():
        log.error(
            "[patch] %s not found — skipping LoRA loader patch. This may result in LoRA adapter not being applied silently.",
            wm_path,
        )
        return False
    src = wm_path.read_text()
    if "_ServeWeightsMapper" in src:
        log.error(
            "[patch] Patch seems to already be applied, unlikely so please check vllm version and patch_vllm_lora_loader()"
        )
        return True
    match = _PATCH_RE.search(src)
    if match is None:
        log.error(
            "[patch] anchor not found in %s (vLLM version drift?) — NOT "
            "patching. If the served adapters target a multimodal wrapper "
            "model they will silently NOT be applied. The evaluate.py "
            "adapter sanity check will catch this.",
            wm_path,
        )
        return False
    inner = match.group("indent") + "    "
    patched = src[: match.end()] + _lora_prefix_patch(inner) + src[match.end() :]
    backup = wm_path.with_suffix(".py.pre-lora-prefix-patch")
    if not backup.exists():
        backup.write_text(src)
    wm_path.write_text(patched)
    log.info("[patch] patched %s (backup: %s)", wm_path, backup.name)
    return True


def main():
    from dotenv import load_dotenv

    load_dotenv()
    args = parse_args()

    if args.hf_collection is None:
        args.hf_collection = os.environ.get("HF_COLLECTION")
    if args.hf_collection:
        log.info("[adapters] using HF_COLLECTION: %s", args.hf_collection)

    # Resolve HF_ORG for collection slug if needed
    hf_org = os.environ.get("HF_ORG", "")
    if args.hf_collection and "/" not in args.hf_collection and hf_org:
        args.hf_collection = f"{hf_org}/{args.hf_collection}"
        log.info("[adapters] resolved to %s", args.hf_collection)

    log.info("=" * 60)
    log.info("vLLM Server configuration")
    log.info("=" * 60)
    for k, v in sorted(vars(args).items()):
        log.info("  %s: %s", k, v)

    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--trust-remote-code",
        "--max-model-len",
        str(args.max_model_len),
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        args.tool_call_parser,
        "--enable-prefix-caching",
        "--language-model-only",
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]

    # ── Build adapter list ──────────────────────────────────────────
    adapter_modules: list[tuple[str, str]] = []
    hf_token = os.environ.get("HF_TOKEN")

    base_slug = args.model.split("/")[-1]

    if args.hf_collection:
        log.info(
            "[adapters] discovering adapters from collection %s...", args.hf_collection
        )
        discovered = discover_adapters_from_collection(
            args.hf_collection, base_slug, hf_token
        )
        if args.adapter_suffixes:
            wanted = {s.strip() for s in args.adapter_suffixes.split(",") if s.strip()}
            # Repo name is {model_slug}-nz-wvs-{dataset}-{population}, so the
            # suffix is everything after the "-nz-wvs-" marker.
            discovered = [
                (name, repo_id)
                for name, repo_id in discovered
                if name.split("-nz-wvs-", 1)[1] in wanted
            ]
            log.info(
                "[adapters] filtered by --adapter-suffixes, %d adapter(s) remain",
                len(discovered),
            )
        if not discovered:
            log.warning("[adapters] no matching adapters found for %s", base_slug)
        else:
            log.info("[adapters] discovered %d adapter(s):", len(discovered))
            for name, repo_id in discovered:
                log.info("           %s -> %s", name, repo_id)
                adapter_modules.append((name, repo_id))

    if args.adapters:
        for entry in args.adapters:
            if "=" not in entry:
                print(
                    f"[adapters] WARNING: skipping malformed adapter '{entry}' "
                    f"(expected name=path)"
                )
                continue
            name, path = entry.split("=", 1)
            adapter_modules.append((name.strip(), path.strip()))

    # vLLM aborts startup on duplicate LoRA names (e.g. an --adapter entry
    # that is also discovered from the collection), so keep the first
    # occurrence of each name and warn about the dropped duplicates.
    seen: set[str] = set()
    deduped: list[tuple[str, str]] = []
    for name, path in adapter_modules:
        if name in seen:
            log.warning("[adapters] duplicate adapter name %s — keeping first", name)
            continue
        seen.add(name)
        deduped.append((name, path))
    adapter_modules = deduped

    # vLLM refuses adapters whose rank exceeds --max-lora-rank: it aborts
    # server startup at load time, so drop those here. With the default
    # (None) the limit is auto-sized to the highest discovered rank so
    # adapters are never silently dropped; an explicit --max-lora-rank
    # keeps the old skip-with-warning behaviour for mixed-rank collections.
    if adapter_modules:
        ranks = {name: get_lora_rank(path, hf_token) for name, path in adapter_modules}
        if args.max_lora_rank is None:
            known = [r for r in ranks.values() if r is not None]
            args.max_lora_rank = max(known) if known else 16
            log.info(
                "[adapters] --max-lora-rank auto-sized to %d (highest "
                "discovered adapter rank)",
                args.max_lora_rank,
            )
        kept: list[tuple[str, str]] = []
        for name, path in adapter_modules:
            rank = ranks[name]
            if rank is None:
                log.warning(
                    "[adapters] %s: could not determine LoRA rank, loading anyway",
                    name,
                )
                kept.append((name, path))
            elif rank <= args.max_lora_rank:
                kept.append((name, path))
            else:
                log.warning(
                    "[adapters] skipping %s: rank %d > --max-lora-rank %d "
                    "(requests for it will 404 at eval time)",
                    name,
                    rank,
                    args.max_lora_rank,
                )
        adapter_modules = kept
        if not adapter_modules:
            log.warning(
                "[adapters] no usable adapters (all rank > %d)",
                args.max_lora_rank,
            )

    # Patch vLLM's LoRA loader so multimodal-wrapper adapters are actually
    # applied (silent no-op otherwise — see patch_vllm_lora_loader). Safe to
    # run for any base model: it is a no-op when adapter names already match
    # and text-only models never hit the mapper path.
    if adapter_modules and not args.no_lora_loader_patch:
        if not patch_vllm_lora_loader():
            if args.allow_unpatched_lora:
                log.warning(
                    "[patch] continuing WITHOUT the LoRA prefix patch "
                    "(--allow-unpatched-lora): on multimodal-wrapper models "
                    "(Qwen3.5/3.6/3.8) the served adapters will load but be "
                    "silently not applied; evaluate.py's adapter check should "
                    "abort their evals."
                )
            else:
                log.error(
                    "[patch] ABORTING: the LoRA loader patch could not be "
                    "applied while serving adapters. On multimodal-wrapper "
                    "models (Qwen3.5/3.6/3.8) their weights would be silently "
                    "ignored and the evals would measure the base model. Fix "
                    "the patch anchor for this vLLM version (see "
                    "workbench/vllm-lora-prefix-fix), or pass "
                    "--allow-unpatched-lora if this model does not need it."
                )
                sys.exit(2)
    elif adapter_modules and args.no_lora_loader_patch:
        log.warning(
            "[patch] --no-lora-loader-patch given while serving adapters: on "
            "multimodal-wrapper models (Qwen3.5/3.6/3.8) the adapters will be "
            "silently not applied. Only do this if the model does not need "
            "the prefix mapping."
        )

    if adapter_modules:
        cmd.append("--enable-lora")
        cmd.extend(["--max-lora-rank", str(args.max_lora_rank)])
        # Keep every adapter's weights GPU-resident. VRAM is ample
        # (rank-64 adapters are ~1-2GB each), and max_cpu_loras stays as the
        # swap fallback so nothing ever re-reads from disk/network.
        cmd.extend(["--max-loras", str(len(adapter_modules))])
        cmd.extend(["--max-cpu-loras", str(len(adapter_modules))])
        cmd.extend(
            ["--lora-modules"] + [f"{name}={path}" for name, path in adapter_modules]
        )
        print(f"[adapters] total adapters loaded: {len(adapter_modules)}")

    if args.dtype and args.dtype != "auto":
        cmd.extend(["--dtype", args.dtype])
    if args.vllm_args:
        cmd.extend(args.vllm_args)

    # Pin CUDA devices for the vLLM subprocess (it inherits our environment;
    # nothing in this script initialises CUDA, so setting the variable here is
    # safe). Without this, vLLM spreads tensor-parallel workers across every
    # visible GPU.
    if args.gpu != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        print(f"[serve] CUDA_VISIBLE_DEVICES={args.gpu}")

    print("=" * 60)
    print("[serve] vLLM command:")
    print(f"  {' '.join(cmd)}")
    print("=" * 60)
    print(f"[serve] starting vLLM server on port {args.port}...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        bufsize=1,
        text=True,
    )

    # Give vLLM a head start before the health poll below begins — model
    # loading takes a while, and early probes just race the bind (showing up
    # as "channel 2: open failed: Connection refused" through the tunnel).
    time.sleep(STARTUP_GRACE_S)

    print("[serve] vLLM output:")
    print("=" * 60)
    import urllib.request

    url = f"http://localhost:{args.port}/v1/models"

    def _stream_output():
        for line in iter(proc.stdout.readline, ""):
            print(f"  {line}", end="", flush=True)

    t = threading.Thread(target=_stream_output, daemon=True)
    t.start()

    start_time = time.time()
    while time.time() - start_time < 600:
        try:
            with urllib.request.urlopen(url, timeout=2):
                print("=" * 60)
                print(f"[serve] Server ready! ({time.time() - start_time:.1f}s)")
                break
        except (ConnectionError, urllib.error.URLError, OSError):
            time.sleep(2)
    else:
        print("=" * 60)
        print("\n[serve] server did not start in 600s, logs above")
        proc.terminate()
        proc.wait()
        sys.exit(1)

    print(f"[serve] vLLM listening on http://{args.host}:{args.port}")

    shutdown = threading.Event()

    def _handle_sig(*_):
        if shutdown.is_set():
            return
        shutdown.set()
        print("\n[serve] shutting down...")
        if proc.poll() is None:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        if proc.stdout:
            proc.stdout.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    try:
        proc.wait()
    except KeyboardInterrupt:
        _handle_sig()


if __name__ == "__main__":
    main()
