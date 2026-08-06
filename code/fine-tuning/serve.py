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
import logging
import os
import signal
import subprocess
import sys
import threading
import time

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler()],
    force=True,
)
for noisy in ("huggingface_hub", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

log = logging.getLogger(__name__)


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
        "--max-lora-rank",
        type=int,
        default=16,
        help="Maximum LoRA rank (for multi-LoRA serving)",
    )

    p.add_argument(
        "vllm_args", nargs="*", help="Extra vLLM args (e.g. --max-num-seqs 500)"
    )

    args, remaining = p.parse_known_args(argv)
    args.vllm_args = remaining + args.vllm_args
    return args


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
        "--enable-prefix-caching",
        "--language-model-only",
        "--max-num-seqs",
        str(args.max_num_seqs),
    ]

    # ── Build adapter list ──────────────────────────────────────────
    adapter_modules: list[tuple[str, str]] = []

    base_slug = args.model.split("/")[-1]

    if args.hf_collection:
        log.info(
            "[adapters] discovering adapters from collection %s...", args.hf_collection
        )
        hf_token = os.environ.get("HF_TOKEN")
        discovered = discover_adapters_from_collection(
            args.hf_collection, base_slug, hf_token
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

    if adapter_modules:
        cmd.append("--enable-lora")
        cmd.extend(["--max-lora-rank", str(args.max_lora_rank)])
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
    time.sleep(10)

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
