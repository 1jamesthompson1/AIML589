#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "python-dotenv>=1.1.0",
# ]
# ///
"""
vastgpu.py — find & provision RTX PRO 6000 GPUs on vast.ai.

Called by run_all.py --vast to provision a full train/eval-capable GPU
instance in the cloud:

    uv run run_all.py --vast Qwen/Qwen3.8-27B

All vast.ai calls go through the CLI via `uv run vastai` (already a
project dependency). The WVS template hash is used for every launch (see
DEFAULT_TEMPLATE_HASH); disk is always 150 GB. Searches both marketplace
GPU variants, RTX_PRO_6000_S and RTX_PRO_6000_WS.

Commands:
    find      Search vast.ai for suitable offers (creates nothing)
    launch    Rent the cheapest matching offer, wait for it to boot,
              write an ssh config entry (host alias below) and print
              the alias as the LAST line of stdout
    teardown  Destroy the instance started by 'launch', remove the ssh
              config entry and delete the state file

Options:
    --num-gpus N      Minimum GPUs required (default: 1)
    --min-inet X      Skip offers with download speed below X Mb/s (default: 1000)
    --min-disk X      Skip offers advertising less than X GB disk (default: 150)
    --max-price X     Skip offers above $X/hour total (default: none)
    --host-alias NAME ssh alias written to .ssh_config (default: vast-gpu1)
    --ssh-key PATH    Private key path (default: ~/.ssh/vast_ai)
    --top N           Number of offers 'find' lists (default: 10)
    --fresh           launch: ignore an already-running instance, rent a new one

Examples:
    uv run vastgpu.py find
    uv run vastgpu.py find --max-price 0.75 --top 5
    uv run vastgpu.py launch -y
    uv run vastgpu.py teardown -y

Setup:
    - Set VAST_API_KEY in the repo-root .env (loaded via python-dotenv),
      export it, or `uv run vastai set api-key <KEY>`. launch/teardown
      require it.
    - Register your public key in the vast.ai console (Settings -> SSH);
      instances get your registered keys at creation.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
SSH_CONFIG = SCRIPT_DIR / ".ssh_config"
STATE_FILE = SCRIPT_DIR / ".vastgpu.state"

# Markers delimiting the auto-generated block in .ssh_config
BLOCK_BEGIN = "# BEGIN AUTO-GENERATED vast-gpu"
BLOCK_END = "# END AUTO-GENERATED vast-gpu"

VASTAI = ["uv", "run", "-q", "vastai"]

# Marketplace has no plain "RTX_PRO_6000"; these are the two variants.
DEFAULT_GPU_NAMES = ["RTX_PRO_6000_S", "RTX_PRO_6000_WS"]
DEFAULT_MIN_INET = (
    2000  # Mb/s — skip slow hosts (model downloads + serving need bandwidth)
)
DEFAULT_DISK = 150  # GB; template does NOT carry disk, so it is always sent explicitly
# WVS template (wvs-llm-rtx6000): vastai/vllm:v0.27.1-cuda-13.0, ssh+direct, port 8080
# forwarded, ~/.no_auto_tmux pre-created.
DEFAULT_TEMPLATE_HASH = "026113b1ffdbb5456e831e6a5dd8475f"
DEFAULT_SSH_KEY = "~/.ssh/vast_ai"
DEFAULT_ALIAS = "vast-gpu1"

BOOT_TIMEOUT_S = (
    1200  # image pulls can outrun 10 min; interactive runs can wait forever
)
POLL_INTERVAL_S = 5


def _is_vanished_error(msg: str) -> bool:
    """True if `msg` means the *offer* we picked was just rented by someone
    else (retry next candidate); False for account/payment/invalid-arg
    problems where a different offer won't help."""
    m = msg.lower()
    if any(
        k in m
        for k in (
            "credit",
            "billing",
            "permission",
            "forbidden",
            "unauthorized",
            "invalid api key",
            "login",
            "invalid args",
        )
    ):
        return False
    return any(
        k in m
        for k in (
            "no_such_ask",
            "not available",
            "no longer",
            "sold",
            "rented",
            "unavailable",
        )
    )


def run_vast(*args: str) -> str:
    """Run `uv run vastai <args>` and return its stdout.

    NB: the vastai CLI is quirky — API errors are printed as a JSON payload
    ``{"error": true, "status_code": ..., "msg": ...}`` to *stderr* while the
    process still exits 0. Treat any such payload (or a non-zero exit) as a
    failure so callers see the real error instead of parsing empty output.
    """
    proc = subprocess.run([*VASTAI, *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"vastai {' '.join(args)} failed:\n{proc.stderr.strip() or proc.stdout.strip()}"
        )
    if proc.stderr.strip():
        try:
            err = json.loads(proc.stderr)
            if isinstance(err, dict) and err.get("error"):
                raise RuntimeError(
                    f"vastai {' '.join(args)} failed ({err.get('status_code')}): "
                    f"{err.get('msg', '')}"
                )
        except json.JSONDecodeError:
            pass  # benign warning on stderr, ignore
    return proc.stdout


def check_api_key() -> None:
    # The vastai CLI prefers VAST_API_KEY (env) over ~/.config/vastai/vast_api_key
    if os.environ.get("VAST_API_KEY"):
        return
    key_file = Path.home() / ".config" / "vastai" / "vast_api_key"
    if not key_file.is_file() or key_file.stat().st_size == 0:
        sys.exit(
            "No vast.ai API key found.\n"
            "Either add VAST_API_KEY=<key> to the repo-root .env, "
            "export VAST_API_KEY, or run: uv run vastai set api-key <KEY>"
        )


# ── offer search ──────────────────────────────────────────────


@dataclass
class Offer:
    id: int
    price: float
    num_gpus: int
    gpu_name: str
    vram: float
    cuda_max_good: str
    disk_space: float
    reliability: float
    inet_down: float
    duration: float  # seconds this offer stays available (large = on-demand)


def search_offers(
    num_gpus: int, min_inet: int, min_disk: float = DEFAULT_DISK
) -> list[Offer]:
    # Offer disk_space is the volume the VM actually gets: template-hash
    # creation sizes the instance from the offer's advertised disk and
    # IGNORES the requested --disk. So filter offers by their advertised
    # disk, not just pass a disk at create time.
    offers: list[Offer] = []
    for gpu_name in DEFAULT_GPU_NAMES:
        query = (
            f"gpu_name={gpu_name} num_gpus>={num_gpus} "
            "rentable=true verified=true external=false "
            f"inet_down>={min_inet} disk_space>={min_disk}"
        )
        out = run_vast("search", "offers", query, "--raw")
        data = json.loads(out)
        if isinstance(data, dict):
            data = data.get("offers", [])
        for o in data:
            try:
                price = float(o.get("dph_total", 1e9))
            except (TypeError, ValueError):
                continue
            if o.get("rented") or price <= 0:
                continue
            offers.append(
                Offer(
                    id=int(o.get("id")),
                    price=price,
                    num_gpus=int(o.get("num_gpus", 0)),
                    gpu_name=str(o.get("gpu_name", "")),
                    vram=float(o.get("gpu_ram", 0) or 0) / 1024,  # MiB -> GB
                    cuda_max_good=str(o.get("cuda_max_good", "") or ""),
                    disk_space=float(o.get("disk_space", 0) or 0),
                    reliability=float(o.get("reliability2", 0) or 0),
                    inet_down=float(o.get("inet_down", 0) or 0),
                    duration=float(o.get("duration", 0) or 0),
                )
            )
    offers.sort(key=lambda o: o.price)
    return offers


def render_offers(offers: list[Offer], top: int) -> None:
    header = f"{'ID':<10}{'$/h':<8}{'GPUs':<6}{'GPU':<20}{'VRAM':<7}{'CUDA':<12}{'Disk':<7}{'rel':<6}{'inet(Mb/s)':<12}{'uptime-h'}"
    print(header)
    for o in offers[:top]:
        print(
            f"{o.id:<10}{o.price:<8.2f}{o.num_gpus:<6}{o.gpu_name:<20}"
            f"{o.vram:<7.0f}{o.cuda_max_good:<12}{o.disk_space:<7.0f}"
            f"{o.reliability:<6.2f}{o.inet_down:<12.0f}{o.duration / 3600:>8.0f}"
        )


# ── instance lifecycle ────────────────────────────────────────


def create_instance(offer: Offer, template_hash: str = DEFAULT_TEMPLATE_HASH) -> int:
    # The template carries only ~/.no_auto_tmux — the long apt-get onstart
    # string was rejected by the API (400 Invalid args), and `vastai update
    # template` wipes unspecified fields, so build tooling lives in run.sh
    # (which bootstraps gcc for triton JIT on first use).
    out = run_vast(
        "create",
        "instance",
        str(offer.id),
        "--template_hash",
        template_hash,
        "--disk",
        str(DEFAULT_DISK),
        "--ssh",
        "--direct",
        "--label",
        "wvs-vast",
        "--cancel-unavail",
        "--raw",
    )
    data = json.loads(out)
    inst_id = data.get("id") or data.get("new_contract")
    if not inst_id:
        raise RuntimeError(f"Could not parse instance id from:\n{out}")
    return int(inst_id)


def _ssh_ready(host: str, port: str, key: str) -> bool:
    """True when an actual ssh login to the instance succeeds.

    vast marks the instance "running" and publishes ssh_host/ssh_port before
    sshd is actually accepting connections — run.sh then dies with
    "Connection refused" seconds later. Gate reads on a real poke.
    """
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-i",
                os.path.expanduser(key),
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "KexAlgorithms=curve25519-sha256",
                "-p",
                str(port),
                f"root@{host}",
                "true",
            ],
            capture_output=True,
            timeout=20,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _ssh_df(host: str, port: str, key: str) -> tuple[float, float, float] | None:
    """Return (size_gb, used_pct, free_gb) of the root fs via ssh, or None."""
    try:
        proc = subprocess.run(
            [
                "ssh",
                "-i",
                os.path.expanduser(key),
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                "-o",
                "KexAlgorithms=curve25519-sha256",
                "-p",
                str(port),
                f"root@{host}",
                "df -h / | awk 'NR==2{print $2, $5, $4}'",
            ],
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        size, pct, free = proc.stdout.strip().split()

        def to_gb(v: str) -> float:
            if v.endswith("M"):
                return float(v[:-1]) / 1024
            if v.endswith("K"):
                return float(v[:-1]) / 1024 / 1024
            return float(v.rstrip("G"))

        return (to_gb(size), float(pct.rstrip("%")), to_gb(free))
    except (ValueError, AttributeError):
        return None


def wait_for_instance(
    inst_id: int, ssh_key: str = DEFAULT_SSH_KEY
) -> tuple[str, str] | tuple[None, None]:
    """Poll until the instance is running AND accepts ssh; return (host, port).

    Returns (None, None) on timeout (host docker/storage failures leave the
    instance stuck in loading; the caller then cleans up).
    """
    print(
        f"Waiting for instance {inst_id} to boot (image pull can take minutes)...",
        end="",
    )
    deadline = time.monotonic() + BOOT_TIMEOUT_S
    last_status = "unknown"
    while time.monotonic() < deadline:
        data = _show_instance(inst_id)
        if data is not None:
            last_status = str(data.get("actual_status", ""))
            host = str(data.get("ssh_host", "") or "")
            port = str(data.get("ssh_port", "") or "")
            if (
                last_status == "running"
                and host
                and port
                and _ssh_ready(host, port, ssh_key)
            ):
                print()
                return host, port
        print(".", end="", flush=True)
        time.sleep(POLL_INTERVAL_S)
    print()
    print(
        f"  [warn] instance {inst_id} did not become reachable within "
        f"{BOOT_TIMEOUT_S}s (last status: {last_status}).",
        file=sys.stderr,
    )
    return None, None


def _show_instance(inst_id: int) -> dict | None:
    try:
        out = run_vast("show", "instance", str(inst_id), "--raw")
        return json.loads(out)
    except (RuntimeError, json.JSONDecodeError):
        return None


def instance_status(inst_id: int) -> str:
    data = _show_instance(inst_id)
    return str(data.get("actual_status", "")) if data else ""


def destroy_instance(inst_id: int) -> None:
    try:
        # The CLI asks its own interactive confirmation without --yes; it then
        # exits 0 with "Aborted." on stdout (no error payload), which would
        # look like success. vastgpu.py already handles confirmation.
        run_vast("destroy", "instance", str(inst_id), "--yes")
    except RuntimeError as e:
        print(
            f" [warn] destroy call failed (instance may already be gone):\n{e}",
            file=sys.stderr,
        )


def list_instances() -> list[dict]:
    try:
        out = run_vast("show", "instances", "--raw")
        data = json.loads(out)
        return data if isinstance(data, list) else (data.get("instances") or [])
    except (RuntimeError, json.JSONDecodeError):
        return []


# ── ssh config management ────────────────────────────────────


def ssh_block(alias: str, host: str, port: str, key: str) -> str:
    return (
        f"Host {alias}\n"
        f"    HostName {host}\n"
        f"    User root\n"
        f"    Port {port}\n"
        f"    IdentityFile {key}\n"
        "    # Default hybrid PQ kex (sntrup761x25519) stalls on the path to vast.ai\n"
        "    # instances (handshake never completes); standard curve25519 works.\n"
        "    KexAlgorithms curve25519-sha256\n"
        "    SetEnv TERM=xterm-256color\n"
        "\n"
    )


def write_ssh_block(block: str) -> None:
    text = SSH_CONFIG.read_text() if SSH_CONFIG.exists() else ""
    pattern = re.compile(
        rf"^{re.escape(BLOCK_BEGIN)}.*?^{re.escape(BLOCK_END)}.*$", re.S | re.M
    )
    new_block = f"{BLOCK_BEGIN}\n{block}{BLOCK_END}\n"
    if pattern.search(text):
        text = pattern.sub(lambda _: new_block, text)
    else:
        text = text.rstrip() + "\n\n" + new_block
    SSH_CONFIG.write_text(text)


def remove_ssh_block() -> None:
    if not SSH_CONFIG.exists():
        return
    text = SSH_CONFIG.read_text()
    pattern = re.compile(
        rf"^{re.escape(BLOCK_BEGIN)}.*?^{re.escape(BLOCK_END)}.*$", re.S | re.M
    )
    SSH_CONFIG.write_text(pattern.sub("", text))


# ── instance state ───────────────────────────────────────────


def state_write(
    alias: str,
    inst_id: int,
    host: str,
    port: str,
    template_hash: str = DEFAULT_TEMPLATE_HASH,
) -> None:
    STATE_FILE.write_text(
        json.dumps(
            {
                "alias": alias,
                "instance_id": inst_id,
                "host": host,
                "port": port,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "template_hash": template_hash,
            }
        )
    )


def state_read() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


# ── commands ─────────────────────────────────────────────────


def cmd_find(args: argparse.Namespace) -> None:
    print(
        f"Searching vast.ai for {', '.join(DEFAULT_GPU_NAMES)} (>= {args.num_gpus} GPU) offers..."
    )
    if args.max_price:
        print(f"  (filtering out offers above ${args.max_price}/h)")
    offers = search_offers(args.num_gpus, args.min_inet, args.min_disk)
    if args.max_price:
        offers = [o for o in offers if o.price <= args.max_price]
    if not offers:
        sys.exit("No matching offers found. Try again later.")
    render_offers(offers, args.top)


def cmd_launch(args: argparse.Namespace) -> None:
    check_api_key()

    # Reuse an instance we already launched (unless --fresh or it is gone)
    if not args.fresh:
        st = state_read()
        if (
            st.get("instance_id")
            and instance_status(int(st["instance_id"])) == "running"
        ):
            print(
                f"Instance {st['instance_id']} (from {STATE_FILE.name}) "
                "is already running - reusing it."
            )
            print(args.host_alias)
            return

        # No state file but exactly one wvs-vast instance exists — a previous
        # launch was interrupted before writing state. Let run_all.py continue
        # on it instead of renting a second box.
        ours = [
            d
            for d in list_instances()
            if d.get("label") == "wvs-vast"
            and d.get("actual_status") in ("running", "loading")
        ]
        if len(ours) == 1:
            status = ours[0].get("actual_status")
            print(
                f"Found orphaned wvs-vast instance {ours[0]['id']} "
                f"({status}) — picking it up."
            )
            if status == "running":
                host = str(ours[0].get("ssh_host") or "")
                port = str(ours[0].get("ssh_port") or "")
                if host and port:
                    inst_id = int(ours[0]["id"])
                    state_write(args.host_alias, inst_id, host, port)
                    write_ssh_block(
                        ssh_block(args.host_alias, host, port, args.ssh_key)
                    )
                    print(args.host_alias)
                    return
                # running but no ssh info yet — fall through to boot-wait it
            else:
                host, port = wait_for_instance(int(ours[0]["id"]), args.ssh_key)
                if host and port:
                    inst_id = int(ours[0]["id"])
                    state_write(args.host_alias, inst_id, host, port)
                    write_ssh_block(
                        ssh_block(args.host_alias, host, port, args.ssh_key)
                    )
                    print(args.host_alias)
                    return
                destroy_instance(int(ours[0]["id"]))
                print("  (orphan failed to boot — destroyed; renting a fresh one)")

    print(
        f"Searching vast.ai for {', '.join(DEFAULT_GPU_NAMES)} (>= {args.num_gpus} GPU) offers..."
    )
    offers = search_offers(args.num_gpus, args.min_inet, args.min_disk)
    if args.max_price:
        offers = [o for o in offers if o.price <= args.max_price]
    if not offers:
        sys.exit("No matching offers found - nothing to launch. Try again later.")

    # Offers marked "delayed" are rentable but can vanish between search and
    # create (no_such_ask) — including instant ones, which people grab in
    # the split second after the search. The retry loop below walks through
    # the next best offers instead of dying on the first vanished one.
    print(
        "Top offers (uptime-h = how long the offer stays available; rentable "
        "is enforced in the search):"
    )
    render_offers(offers, min(3, len(offers)))

    tried: set[int] = set()
    for attempt in range(3):
        best = next((o for o in offers if o.id not in tried), None)
        if best is None:
            break
        print()
        print(
            f"Best offer: instance {best.id} at ${best.price:.2f}/h "
            f"(~${best.price * 10:.0f} per 10h run)."
        )
        if attempt == 0 and not args.yes:
            answer = input("Rent this instance now? [y/N] ")
            if answer.lower() != "y":
                sys.exit("Aborted.")
        print(
            f"Creating instance {best.id} (template {args.template}, "
            f"disk {DEFAULT_DISK}GB)..."
        )
        try:
            inst_id = create_instance(best, args.template)
            break
        except RuntimeError as e:
            msg = str(e)
            tried.add(best.id)
            if not _is_vanished_error(msg):
                sys.exit(f"[error] instance creation failed:\n       {msg}")
            print(f"  [warn] {best.id} was just rented — trying the next offer...")
    else:
        sys.exit(
            "[error] every candidate offer was grabbed before we could rent it — "
            "try 'launch' again."
        )
    print(f"Instance id: {inst_id}")

    host, port = wait_for_instance(inst_id, args.ssh_key)
    if not host or not port:
        destroy_instance(inst_id)
        sys.exit(
            f"[error] instance {inst_id} failed to boot (host-side docker/"
            "storage issue?). Destroying it — try 'launch' again (a retry "
            "usually lands on a different machine)."
        )

    # Gate on disk state: some hosts hand out volumes that are already ~full
    # (seen: 193G disk born at 100% used — the training run dies within the
    # first model download). Refuse the machine instead of paying for a
    # doomed run.
    df = _ssh_df(host, port, args.ssh_key)
    if df is not None:
        size, used_pct, free = df
        print(
            f"Instance disk: {size:.0f}G total, {used_pct:.0f}% used, {free:.0f}G free"
        )
        if used_pct >= 90:
            destroy_instance(inst_id)
            sys.exit(
                f"[error] instance {inst_id} booted with a nearly-full disk "
                f"({used_pct:.0f}% used) — the host handed out a dirty "
                "volume. Destroyed; try 'launch' again."
            )
    else:
        print("Instance disk: (could not query)")

    state_write(args.host_alias, inst_id, host, port, args.template)
    write_ssh_block(ssh_block(args.host_alias, host, port, args.ssh_key))

    print()
    print(f"Instance ready: ssh root@{host} -p {port}")
    print(f"ssh config written: Host {args.host_alias} in {SSH_CONFIG}")
    print(args.host_alias)  # last line of stdout = the alias for run_all.py


def cmd_teardown(args: argparse.Namespace) -> None:
    check_api_key()
    st = state_read()
    if st.get("instance_id"):
        inst_id = int(st["instance_id"])
        status = instance_status(inst_id)
        if status:
            print(f"Destroying instance {inst_id} (status: {status})...")
            if not args.yes:
                answer = input(f"Irreversible! Destroy instance {inst_id}? [y/N] ")
                if answer.lower() != "y":
                    sys.exit("Aborted.")
            destroy_instance(inst_id)
        else:
            print(f"Instance {inst_id} not found on vast.ai (already destroyed?)")
    else:
        print(f"No instance recorded in {STATE_FILE.name}.")

    # Safety net: catch any wvs-vast instances left untracked by the state
    # file (e.g. a destroy that was aborted while the vm was still billed).
    tracked = int(st["instance_id"]) if st.get("instance_id") else None
    leftovers = [
        d
        for d in list_instances()
        if d.get("label") == "wvs-vast" and d.get("id") != tracked
    ]
    for d in leftovers:
        print(
            f"Found untracked instance {d['id']} ({d.get('actual_status')}) "
            "with label wvs-vast."
        )
        if not args.yes:
            answer = input(f"Destroy untracked instance {d['id']}? [y/N] ")
            if answer.lower() != "y":
                continue
        destroy_instance(int(d["id"]))

    remove_ssh_block()
    STATE_FILE.unlink(missing_ok=True)
    print("Removed ssh config block and state file. Done.")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vastgpu.py",
        description="Find & provision RTX PRO 6000 GPUs on vast.ai.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    for name, help_text, fn in (
        ("find", "Search vast.ai for suitable offers (creates nothing)", cmd_find),
        (
            "launch",
            "Rent the cheapest matching offer and set up ssh access",
            cmd_launch,
        ),
        (
            "teardown",
            "Destroy the instance from 'launch', clean up ssh config",
            cmd_teardown,
        ),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--num-gpus", type=int, default=1, help="Minimum GPUs required")
        sp.add_argument(
            "--min-inet",
            type=int,
            default=DEFAULT_MIN_INET,
            help="Skip offers with download speed below X Mb/s",
        )
        sp.add_argument(
            "--min-disk",
            type=float,
            default=DEFAULT_DISK,
            help="Skip offers advertising less than X GB of disk "
            "(template-hash VMs get the offer's advertised disk, "
            "not the requested one)",
        )
        sp.add_argument(
            "--max-price",
            type=float,
            default=None,
            help="Skip offers above $X/hour total",
        )
        sp.add_argument(
            "--host-alias",
            default=DEFAULT_ALIAS,
            help="ssh alias written to .ssh_config",
        )
        sp.add_argument("--ssh-key", default=DEFAULT_SSH_KEY, help="Private key path")
        sp.add_argument(
            "--top", type=int, default=10, help="Number of offers 'find' lists"
        )
        if name == "launch":
            sp.add_argument(
                "--template",
                default=DEFAULT_TEMPLATE_HASH,
                help="vast.ai template hash (use a CUDA *devel* "
                "image so nvcc exists for vLLM/FlashInfer JIT)",
            )
        sp.set_defaults(func=fn)
        if name in ("launch", "teardown"):
            sp.add_argument(
                "--fresh",
                action="store_true",
                help="launch: rent a new instance even if one is running",
            )
            sp.add_argument(
                "--yes", "-y", action="store_true", help="Don't ask for confirmation"
            )
    return p.parse_args(argv)


def main(argv=None) -> int:
    load_dotenv(SCRIPT_DIR.parent.parent / ".env")
    args = parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
