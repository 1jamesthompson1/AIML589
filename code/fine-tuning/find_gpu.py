#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# ///
"""
find_gpu.py — Find free GPUs on the university ECS servers.

Usage:
    uv run find_gpu.py [options]

Options:
    --gpu-mem <GB>     Minimum GPU memory in GB (default: 24)
    --min-gpus <N>     Minimum number of free GPUs (default: 1)
    --jump <host>      Run via a jump host (default: uni-entry)
    --update-config    Write top 3 servers to .ssh_config as uni-gpu1/2/3
    --help, -h         Show this help

Examples:
    uv run find_gpu.py
    uv run find_gpu.py --gpu-mem 48 --min-gpus 2
    uv run find_gpu.py --update-config

What it does:
    1. Checks known ECS GPU servers via SSH + nvidia-smi
    2. Filters by GPU memory and free GPU count (GPUs with no running
       processes — memory.used alone is unreliable, idle CUDA GPUs still
       reserve a few MiB)
    3. Sorts candidates (highest mem → lowest avg util → most free GPUs)
    4. Prints results
    5. With --update-config: writes the top 3 to .ssh_config
       (in an auto-generated marker block, same scheme as vastgpu.py)

Dependencies:
    - SSH access to uni-entry and/or the ECS servers (vuw-lab setup)
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SSH_CONFIG = Path(os.environ.get("CONFIG_OUT") or SCRIPT_DIR / ".ssh_config")

BLOCK_BEGIN = "# BEGIN AUTO-GENERATED uni-gpu"
BLOCK_END = "# END AUTO-GENERATED uni-gpu"

SSH_OPTS = ["-o", "ConnectTimeout=3", "-o", "StrictHostKeyChecking=accept-new"]

SERVERS = [
    "cuda-small0",
    "cuda-small1",
    "cuda00",
    "cuda01",
    "cuda02",
    "cuda03",
    "cuda04",
    "cuda05",
    "cuda06",
    "cuda07",
    "cuda08",
    "cuda09",
    "cuda10",
    "cuda11",
    "cuda12",
    "cuda13",
    "cuda14",
    "cuda15",
    "cuda16",
    "cuda17",
    "cuda18",
    "cuda19",
    "cuda20",
    "cuda21",
    "cuda22",
    "cuda23",
    "cuda24",
    "gryphon",
    "red-tomatoes",
    "piccolo",
    "the-villa",
    "bordeaux",
]

GPU_MEM = {
    "cuda-small0": 16,
    "cuda-small1": 20,
    "cuda00": 24,
    "cuda01": 24,
    "cuda02": 24,
    "cuda03": 24,
    "cuda04": 24,
    "cuda05": 24,
    "cuda06": 24,
    "cuda07": 24,
    "cuda08": 24,
    "cuda09": 24,
    "cuda10": 24,
    "cuda11": 24,
    "cuda12": 24,
    "cuda13": 24,
    "cuda14": 48,
    "cuda15": 48,
    "cuda16": 48,
    "cuda17": 24,
    "cuda18": 48,
    "cuda19": 24,
    "cuda20": 48,
    "cuda21": 48,
    "cuda22": 48,
    "cuda23": 96,
    "cuda24": 96,
    "gryphon": 48,
    "red-tomatoes": 48,
    "piccolo": 24,
    "the-villa": 48,
    "bordeaux": 24,
}

USER = "thompsjame1"
JUMP_HOST = "uni-entry"


def ssh(*args: str, timeout: int = 15) -> tuple[int, str]:
    """Run a remote command; returns (returncode, stdout).

    (returncode = 0, "" ) if the host fails to connect or times out.
    """
    cmd = ["ssh", *SSH_OPTS, *args]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout
    except (subprocess.TimeoutExpired, OSError):
        return 1, ""


def gpu_state(server: str) -> tuple[list[tuple], set[str]]:
    """Return (gpu_rows, busy_uuids) for one server.

    gpu_rows: (index, uuid, mem_total, mem_used, util) parsed from
    nvidia-smi. busy_uuids: uuids of GPUs running at least one compute
    process. A GPU counts as free iff its uuid is not busy.
    """
    rc, out = ssh(
        "-o",
        "ConnectTimeout=5",
        server,
        "nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,"
        "utilization.gpu --format=csv,noheader; echo '===PROCS==='; "
        "nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader",
    )
    if rc != 0 or not out:
        return [], set()
    gpu_text, _, proc_text = out.partition("===PROCS===")
    rows = []
    for line in gpu_text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        try:
            rows.append(
                (
                    parts[0],
                    parts[1],
                    int(parts[2].split()[0]),
                    int(parts[3].split()[0]),
                    int(parts[4].split()[0]),
                )
            )
        except ValueError:
            continue
    busy = {line.strip() for line in proc_text.splitlines() if line.strip()}
    return rows, busy


def scan(min_mem: int, min_gpus: int) -> tuple[list, list[str]]:
    """Return (sorted_candidates, unreachable_servers).

    A candidate is (gpu_mem, avg_util, free_count, server); sorted by
    highest mem, then lowest avg utilization, then most free GPUs.
    """
    candidates = []
    unreachable = []
    for server in SERVERS:
        if GPU_MEM.get(server, 0) < min_mem:
            continue
        rc, _ = ssh(server, "exit")
        if rc:
            unreachable.append(server)
            continue
        rows, busy = gpu_state(server)
        if not rows:
            continue
        free = sum(1 for _, uuid, *_ in rows if uuid not in busy)
        if free < min_gpus:
            continue
        avg_util = sum(r[4] for r in rows) // len(rows)
        mem = GPU_MEM[server]
        candidates.append((mem, avg_util, free, server))
    candidates.sort(key=lambda c: (-c[0], c[1], -c[2]))
    return candidates, unreachable


def write_ssh_block(block: str) -> None:
    """Insert/replace the auto-generated uni-gpu block in .ssh_config."""
    lines = SSH_CONFIG.read_text().splitlines() if SSH_CONFIG.exists() else []
    begin = next((i for i, line in enumerate(lines) if line == BLOCK_BEGIN), None)
    end = next((i for i, line in enumerate(lines) if line == BLOCK_END), None)
    if begin is not None and end is not None:
        new_lines = (
            lines[:begin]
            + [BLOCK_BEGIN]
            + block.splitlines()
            + [BLOCK_END]
            + lines[end + 1 :]
        )
    else:
        new_lines = (
            lines
            + ([""] if lines else [])
            + [BLOCK_BEGIN]
            + block.splitlines()
            + [BLOCK_END]
        )
    SSH_CONFIG.write_text("\n".join(new_lines) + "\n")


def host_block(candidates, count: int) -> str:
    block = []
    for i, (_, _, _, server) in enumerate(candidates[:count], start=1):
        block.extend(
            [
                f"Host uni-gpu{i}",
                f"    HostName {server}.ecs.vuw.ac.nz",
                f"    User {USER}",
                f"    ProxyJump {JUMP_HOST}",
                "    IdentityFile ~/.ssh/id_rsa",
                "    ForwardAgent yes",
                "",
            ]
        )
    return "\n".join(block)


def main() -> None:
    p = argparse.ArgumentParser(description="Find free GPUs on ECS servers.")
    p.add_argument("--gpu-mem", type=int, default=48, help="Minimum GPU memory (GB)")
    p.add_argument("--min-gpus", type=int, default=1, help="Minimum free GPUs")
    p.add_argument(
        "--jump", default=JUMP_HOST, help="Jump host (default: uni-entry; '' disables)"
    )
    p.add_argument(
        "--update-config",
        action="store_true",
        help="Write top 3 servers to .ssh_config as uni-gpu1/2/3",
    )
    args = p.parse_args()

    # Self-propagation: when run from the dev machine with a jump host, copy
    # this script to the jump host and execute it there (it recurses with
    # --jump '' so the jump is disabled on the remote). Same pattern as the
    # old find_gpu.sh. With --update-config the local .ssh_config is seeded
    # to the jump host first so the remote marker-replacement preserves the
    # manual entries, and the result is copied back.
    if args.jump and SCRIPT_DIR != Path("/tmp"):
        remote = (
            "cat > /tmp/find_gpu.py && chmod +x /tmp/find_gpu.py && "
            + "python3 /tmp/find_gpu.py "
            + f"--gpu-mem {args.gpu_mem} --min-gpus {args.min_gpus} "
            + "--jump ''"
        )
        if args.update_config:
            subprocess.run(
                [
                    "scp",
                    *SSH_OPTS,
                    str(SSH_CONFIG),
                    f"{args.jump}:/tmp/ssh_config_update",
                ],
                check=True,
            )
            remote = remote.replace(
                "python3 /tmp/find_gpu.py ",
                "CONFIG_OUT=/tmp/ssh_config_update python3 /tmp/find_gpu.py ",
            )
        with open(__file__) as fh:
            proc = subprocess.run(["ssh", *SSH_OPTS, args.jump, remote], stdin=fh)
        # The remote exits 1 when no servers are found (with its message
        # already printed above), so propagate the exit code rather than
        # crashing on a non-zero return.
        if proc.returncode:
            sys.exit(proc.returncode)
        if args.update_config:
            subprocess.run(
                [
                    "scp",
                    *SSH_OPTS,
                    f"{args.jump}:/tmp/ssh_config_update",
                    str(SSH_CONFIG),
                ],
                check=True,
            )
        return

    print(
        f"Scanning GPU servers (min {args.gpu_mem}GB GPU, "
        f"min {args.min_gpus} free GPUs)...\n"
    )

    candidates, unreachable = scan(args.gpu_mem, args.min_gpus)

    for mem, avg_util, free, server in candidates:
        print("========================================")
        print(f"  {server}  ({mem}GB GPUs, avg util {avg_util}%)")
        print(f"  Free GPUs: {free}")
        print()

    if not candidates:
        print(
            f"No servers found with {args.min_gpus}+ free GPU(s) of "
            f"{args.gpu_mem}GB or more."
        )
        if unreachable:
            print(
                f"\nCould not reach {len(unreachable)} server(s): "
                + ", ".join(unreachable)
            )
        sys.exit(1)

    if unreachable:
        print(
            f"(Could not reach {len(unreachable)} server(s): "
            + ", ".join(unreachable)
            + ")"
        )

    if args.update_config:
        write_ssh_block(host_block(candidates, 3))
        print(f"Wrote {min(3, len(candidates))} host(s) to {SSH_CONFIG}")


if __name__ == "__main__":
    main()
