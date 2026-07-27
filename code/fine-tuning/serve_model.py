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
Serve a fine-tuned model via vLLM for evaluation.

Usage:
    scp serve_model.py user@host:~
    ssh user@host ./serve_model.py --model Qwen/Qwen2.5-1.5B-Instruct \
        --hf-token "$HF_TOKEN"

Or from the repo:
    uv run code/fine-tuning/serve_model.py --help
"""

import argparse
import os
import signal
import subprocess
import sys
import threading
import time


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Serve a fine-tuned model via vLLM for evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--model", required=True, help="Model name or path (HF hub or local)"
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
        "--max-model-len", type=int, default=32768, help="Maximum sequence length"
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Trust remote code when loading model",
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

    print("=" * 60)
    print("vLLM Server configuration")
    print("=" * 60)
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

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
    ]
    if args.dtype and args.dtype != "auto":
        cmd.extend(["--dtype", args.dtype])
    if args.vllm_args:
        cmd.extend(args.vllm_args)

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

    def _handle_sig(*_):
        print("\n[serve] shutting down...")
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
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
