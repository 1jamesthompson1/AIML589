#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["python-dotenv>=1.1.0"]
# ///
"""
watchdog.py — phone notifications + auto-teardown for long-running GPU work.

Companion for vast.ai rented instances (the kind that burn money all night
when you forget them), provisioned by ``vastgpu.py launch`` and removed by
``vastgpu.py teardown``. It publishes to an ntfy.sh topic so a phone with the
ntfy app subscribed to the topic shows:

  - a periodic "still running" status (uptime, vLLM up/down, time to deadline),
  - soft deadline reminders (default T-60m / T-15m / T-5m),
  - a hard-deadline notification followed by instance destruction,
  - an idle notice + destruction when the served process has ended and the
    instance would otherwise bill for nothing (e.g. after the laptop went to
    sleep mid-serve — the exact accident this exists for),
  - replies to commands sent as plain *messages* into the topic from the
    phone: ``status`` (report now), ``extend 4`` / ``extend 1.5`` (hours),
    and ``teardown now``.

Two sites:

  --site remote   runs ON the GPU box (nohup'd by vastgpu launch). Survives
                  SSH disconnects and a sleeping laptop — the real overnight
                  guard. Teardown is a direct vast.ai REST call
                  (DELETE /api/v0/instances/{id}/) using VAST_API_KEY and the
                  instance id from VAST_CONTAINERLABEL (or --instance-id), so
                  no vastai CLI is needed inside the container.

  --site local    runs on the dev machine (spawned by vastgpu launch). Belt
                  backup: watches the vastgpu state file; if the rented
                  instance is still running with no local serve tunnel and no
                  remote watchdog (pgrep over ssh), it sends pings and — with
                  --teardown — destroys it past its own deadline.

Without VAST_API_KEY / a verifiable instance id the watchdog degrades to
notification-only and exits when the guarded process ends. On university hosts
(no vast.ai) it is purely a phone presence indicator.

.env keys (repo root; vastgpu launch syncs these to the instance):
    WATCHDOG_NTFY_URL=https://ntfy.sh/<private-topic>   # required, or pass --ntfy
    VAST_API_KEY=...          # full-account key; needed only for auto-teardown

ntfy topic privacy: the topic URL is the only credential (anyone who knows it
can read/post). Use a long unguessable name — ntfy topics are capped at 64
characters, so something like ``wvs-wd-<24 hex chars>`` is plenty and valid.

Examples:
    uv run watchdog.py --site remote --ntfy https://ntfy.sh/wvs-x93 \
        --label "vast-gpu1 serve" --port 8523
    uv run watchdog.py --site local --ntfy https://ntfy.sh/wvs-x93 --teardown
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    from dotenv import load_dotenv  # repo-venv; the remote box has none
except ImportError:  # remote box: env arrives via --envfile instead

    def load_dotenv(*_args, **_kwargs):
        return False


SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parents[1]
STATE_FILE = SCRIPT_DIR / ".vastgpu.state"
SSH_CONFIG = SCRIPT_DIR / ".ssh_config"

# Every message this script publishes carries this ntfy title, so the command
# poll can ignore our own output and treat title-less texts as user commands.
WATCHDOG_TITLE = "wvs watchdog"

POLL_EVERY_S = 45
HTTP_TIMEOUT_S = 20

DEFAULT_HEARTBEAT_MIN = 60
DEFAULT_DEADLINE_HOURS = 8.0
DEFAULT_SOFT_REMINDERS = "60m,15m,5m"


# ── tiny HTTP helpers (stdlib only — the remote box has no requests) ──


def http_text(
    url: str,
    *,
    method: str = "GET",
    data: bytes | None = None,
    headers: dict | None = None,
    timeout: int = HTTP_TIMEOUT_S,
) -> tuple[str | int, str]:
    """Return (status, text); status is the HTTP code or the string "error"."""
    request = urllib.request.Request(
        url, data=data, headers=headers or {}, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        try:
            return e.code, e.read().decode("utf-8", "replace")
        except Exception:
            return e.code, ""
    except OSError as e:
        return "error", str(e)


def bad(status: str | int) -> bool:
    return status == "error" or (isinstance(status, int) and status >= 400)


def ntfy_publish(
    url: str,
    text: str,
    *,
    title: str = WATCHDOG_TITLE,
    priority: int = 3,
    click: str | None = None,
    tags: str = "",
) -> bool:
    """POST one notification to the ntfy topic URL."""
    headers = {"Content-Type": "text/plain; charset=utf-8"}
    if title:
        headers["X-Title"] = title
    headers["X-Priority"] = str(priority)
    if click:
        headers["X-Click"] = click
    if tags:
        headers["X-Tags"] = tags
    status, body = http_text(
        url.rstrip("/"), method="POST", data=text.encode("utf-8"), headers=headers
    )
    if bad(status):
        print(
            f"[watchdog] ntfy publish failed ({status}): {body[:200]}",
            file=sys.stderr,
        )
        return False
    return True


def ntfy_poll(url: str, since_minutes: int = 15) -> list[dict]:
    """Fetch recent messages of the topic (NDJSON events at /json).

    Only title-less events are returned: publishes from this script always
    carry a title, so anything plain is a human command from the phone app.
    """
    base = url.rstrip("/")
    status, body = http_text(f"{base}/json?poll=1&since={since_minutes}m")
    if bad(status):
        return []
    events = []
    for line in body.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "message" and event.get("title") is None:
            events.append(event)
    return events


# ── vast.ai REST (the CLI's own contract: Bearer key, /api/v0 paths) ──


def vast_url() -> str:
    return os.environ.get("VAST_URL", "https://console.vast.ai").rstrip("/")


def vast_api_key() -> str:
    key = os.environ.get("VAST_API_KEY")
    if key:
        return key.strip()
    key_file = Path.home() / ".config" / "vastai" / "vast_api_key"
    if key_file.is_file():
        return key_file.read_text().strip()
    return ""


def vast_instance(id_: int, key: str) -> dict | None:
    """One instance record; {} when gone, None when the host/API was unreachable."""
    status, body = http_text(
        f"{vast_url()}/api/v0/instances/{id_}/",
        headers={"Authorization": f"Bearer {key}"},
    )
    if status == "error":
        return None
    if isinstance(status, int) and status == 404:
        return {}
    if isinstance(status, int) and status >= 500:
        return None
    try:
        data = json.loads(body)
    except ValueError:
        return None
    # Response shapes seen from the API/CLI: {"instances": {…record…}} for a
    # single instance, {"data": {…}} elsewhere. Unwrap whichever appears.
    if isinstance(data, dict):
        inner = data.get("instances", data.get("data"))
        if isinstance(inner, list) and inner:
            inner = inner[0]
        if isinstance(inner, dict):
            data = inner
    return data if isinstance(data, dict) else {}


def vast_destroy(id_: int, key: str) -> bool:
    """Destroy one instance; True when the API reports success."""
    status, body = http_text(
        f"{vast_url()}/api/v0/instances/{id_}/",
        method="DELETE",
        data=b"{}",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
    )
    if bad(status):
        print(
            f"[watchdog] destroy call failed ({status}): {body[:300]}",
            file=sys.stderr,
        )
        return False
    try:
        return bool(json.loads(body).get("success"))
    except ValueError:
        return False


# ── misc helpers ──


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "none"
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m" if minutes else "0m"


def parse_soft(spec: str) -> list[float]:
    """'60m,15m,5m' (bare numbers = minutes) -> reminder offsets in seconds."""
    out: list[float] = []
    for token in spec.split(","):
        token = token.strip().lower()
        if not token:
            continue
        multiplier = 60.0
        if token.endswith("m"):
            token, multiplier = token[:-1], 60.0
        elif token.endswith("h"):
            token, multiplier = token[:-1], 3600.0
        elif token.endswith("s"):
            token, multiplier = token[:-1], 1.0
        try:
            out.append(float(token) * multiplier)
        except ValueError:
            continue
    return sorted(set(out), reverse=True)


def serve_process_alive() -> bool | None:
    """pgrep for the serve.py interpreter (None when pgrep itself is unusable)."""
    try:
        proc = subprocess.run(["pgrep", "-f", "[s]erve.py"], capture_output=True)
        return proc.returncode == 0
    except OSError:
        return None


def vllm_status(port: int | None) -> str:
    if not port:
        return "unknown"
    status, _ = http_text(f"http://127.0.0.1:{port}/v1/models", timeout=5)
    return "up" if status == 200 else "down"


# ── remote site: the primary overnight guard, runs on the GPU box ──


def resolve_ntfy(args: argparse.Namespace) -> str | None:
    """The topic URL: explicit --ntfy wins, else WATCHDOG_NTFY_URL env/envfile."""
    if args.ntfy:
        return args.ntfy
    return os.environ.get("WATCHDOG_NTFY_URL", "").strip() or None


def read_envfile(path) -> None:
    """Tiny KEY=VALUE loader for the remote box (no dotenv there).

    Only sets variables that are not already in the environment, so an
    explicit os.environ always wins.
    """
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        print(f"watchdog: envfile {path} unreadable", file=sys.stderr)
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def run_remote(args: argparse.Namespace) -> int:
    if not args.ntfy:
        load_dotenv(REPO / ".env")  # normal invocation on the dev project env too
    if args.envfile:
        read_envfile(args.envfile)
    args.ntfy = resolve_ntfy(args)
    if not args.ntfy:
        print(
            "watchdog: no --ntfy URL and WATCHDOG_NTFY_URL is unset — cannot notify",
            file=sys.stderr,
        )
        return 2
    click = args.click or "https://console.vast.ai/instances/"

    start = time.monotonic()
    deadline: float | None = (
        start + args.deadline_hours * 3600 if args.deadline_hours > 0 else None
    )
    soft = [s for s in parse_soft(args.soft_reminders) if deadline is not None]
    fired: set[float] = set()
    last_heartbeat = -1e9
    seen: set[tuple] = set()
    serve_seen = False
    label = args.label or "serve"

    def note(text: str, *, priority: int = 3, tags: str = "") -> None:
        ntfy_publish(args.ntfy, text, priority=priority, click=click, tags=tags)

    # Instance identity + teardown arming.
    instance_id = args.instance_id
    if instance_id is None:
        raw = os.environ.get("VAST_CONTAINERLABEL", "")
        if raw.strip().isdigit():
            instance_id = int(raw)
    key = vast_api_key()
    can_teardown = bool(instance_id and key) and not args.no_teardown
    if can_teardown and instance_id:
        # Only trust the id after it resolves to a real instance record.
        record = vast_instance(instance_id, key)
        if record is None or not (record.get("actual_status") or record.get("label")):
            note(
                f"teardown disabled: instance id {instance_id} not verifiable "
                "— notifying only",
                tags="warning",
            )
            can_teardown = False
    elif not args.no_teardown:
        note(
            "teardown NOT armed: needs VAST_API_KEY and the instance id in the "
            "container env — notifying only",
            tags="warning",
        )

    def destroy_now(reason: str) -> None:
        """One loud final tell + destroy (or not-armed warning); always exits."""
        nonlocal stopped
        stopped = True
        if can_teardown:
            note(
                f"HARD: {reason} — destroying instance {instance_id} now",
                priority=5,
                tags="skull",
            )
            ok = vast_destroy(instance_id, key)
            record = vast_instance(instance_id, key) if ok else {}
            gone = record is None or not (record.get("actual_status"))
            note(
                f"instance {instance_id}: "
                + (
                    "destroyed — nothing will bill tonight."
                    if ok and gone
                    else f"destroy call sent; double-check status ({record.get('actual_status') or '?'})"
                ),
                priority=4,
            )
        else:
            note(
                f"{reason} — teardown NOT armed, instance keeps billing",
                priority=5,
                tags="skull",
            )

    stopped = False
    destruction_reason: str | None = None

    def handle_command(event: dict) -> None:
        """Act on one plain message from the phone (no title)."""
        nonlocal deadline, destruction_reason
        text = (event.get("message") or "").strip().lower()
        if not text:
            return
        if text == "status":
            heartbeat(forced=True)
            return
        if text in ("teardown now", "destroy now"):
            if destruction_reason is None:
                destruction_reason = "teardown now (command from phone)"
            return
        if text.startswith("extend"):
            parts = text.split()
            try:
                hours = float(parts[1]) if len(parts) > 1 else 2.0
            except (IndexError, ValueError):
                hours = 2.0
            if deadline is None:
                deadline = time.monotonic() + hours * 3600
            else:
                deadline += hours * 3600
            soft[:] = [s for s in parse_soft(args.soft_reminders)]
            fired.clear()
            note(
                f"deadline extended by {hours}h — now in "
                f"{format_duration(deadline - time.monotonic())}",
                tags="checkered_flag",
            )
            return
        # anything else: a normal human message; ignore silently

    def heartbeat(forced: bool = False) -> None:
        """One status publication; forced = an explicit reply to 'status'."""
        uptime = time.monotonic() - start
        remaining = deadline - time.monotonic() if deadline is not None else None
        note(
            f"{label}: up {format_duration(uptime)}; vLLM {vllm_status(args.port)}; "
            f"deadline in {format_duration(remaining)}"
            + ("  [forced]" if forced else ""),
            tags="computer",
        )

    def sigterm(_sig, _frm):
        # Final presence message only if we did NOT just destroy ourselves.
        if not stopped:
            ntfy_publish(
                args.ntfy,
                f"{label}: watchdog stopped by signal — instance may still be running",
                priority=4,
                tags="warning",
                click=click,
            )
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm)
    signal.signal(signal.SIGINT, sigterm)

    note(
        f"watchdog started on {label}: heartbeat every {args.heartbeat_min}m, "
        f"deadline {format_duration(args.deadline_hours * 3600) if args.deadline_hours > 0 else 'none'}",
        tags="white_check_mark",
    )

    while not stopped:
        now_s = time.monotonic()

        # 1. Pending phone commands (plain messages without our title).
        poll = ntfy_poll(args.ntfy, since_minutes=max(10, POLL_EVERY_S // 60))
        if args.debug:
            left = deadline - now_s if deadline is not None else None
            print(
                f"[tick t+{now_s - start:.0f}s] serve={serve_process_alive()} "
                f"seen={serve_seen} deadline_left={left} me={os.getpid()}",
                file=sys.stderr,
            )
        for event in poll:
            marker = (event.get("time"), event.get("id"), event.get("message"))
            if marker not in seen:
                seen.add(marker)
                handle_command(event)
            if stopped:
                break
        if destruction_reason is not None:
            destroy_now(destruction_reason)
            break

        # 2. Idle kill: serve.py ended while the instance would keep billing.
        # The kill only ever fires AFTER the serve process was seen alive at
        # least once — until then the uv env may still be resolving (a cold
        # first run can take minutes), and killing that would nuke a booting
        # machine, not an idle one.
        life = serve_process_alive()
        if life:
            serve_seen = True
        elif serve_seen:
            if not args.no_idle_destroy:
                destruction_reason = (
                    "serve process ended — idle instance would bill for nothing"
                )
        # life unknown (no pgrep) or not seen yet: nothing to do; heartbeats
        # and the deadline keep us honest either way.

        # 3. Soft/hard deadline reminders; the hard kill is dispatched at the
        # top of the next tick (single destructive path).
        if deadline is not None:
            remaining = deadline - now_s
            for offset in soft:
                if offset not in fired and remaining <= offset:
                    fired.add(offset)
                    note(
                        f"soft deadline: {format_duration(remaining)} left "
                        f"(reply 'extend 4' to add 4h)",
                        priority=4,
                        tags="hourglass",
                    )
            if remaining <= 0 and destruction_reason is None:
                destruction_reason = "hard deadline reached"

        # 4. Periodic heartbeat.
        if (now_s - last_heartbeat) >= args.heartbeat_min * 60:
            last_heartbeat = now_s
            heartbeat()
        time.sleep(POLL_EVERY_S)

    return 0


# ── local site: dev-machine fallback belt ──


def run_local(args: argparse.Namespace) -> int:
    load_dotenv(REPO / ".env")
    if not args.ntfy:
        args.ntfy = os.environ.get("WATCHDOG_NTFY_URL", "")
    if not args.ntfy:
        print("watchdog: no ntfy URL; disabled", file=sys.stderr)
        return 2
    state = None
    try:
        state = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else None
    except (OSError, ValueError):
        state = None
    if not state or not state.get("instance_id"):
        print("watchdog local: no vast instance in state file — nothing to guard")
        return 0
    alias = state.get("alias", "vast")
    label = args.label or f"{alias} (fallback)"
    start = time.monotonic()
    deadline = start + args.deadline_hours * 3600 if args.deadline_hours > 0 else None
    last_hb = -1e9
    print(f"watchdog local: guarding {alias} instance {state['instance_id']}")

    while True:
        # Instance still running? (read-only vast.ai call, local machine)
        id_ = int(state["instance_id"])
        try:
            out = subprocess.run(
                ["uv", "run", "-q", "vastai", "show", "instance", str(id_), "--raw"],
                capture_output=True,
                text=True,
                timeout=60,
            )
            data = json.loads(out.stdout) if out.returncode == 0 else {}
        except (OSError, ValueError, subprocess.SubprocessError):
            data = {}
        status = str(data.get("actual_status", ""))
        if status == "exited" or (data and status not in ("running", "loading")):
            ntfy_publish(
                args.ntfy,
                f"{label}: instance no longer running ({status or 'gone'}) — done.",
                tags="white_check_mark",
            )
            return 0

        # Is the remote watchdog alive on the instance?
        try:
            probe = subprocess.run(
                [
                    "ssh",
                    "-F",
                    str(SSH_CONFIG),
                    alias,
                    "pgrep -f '[w]atchdog.py' >/dev/null",
                ],
                capture_output=True,
                timeout=20,
            )
            remote_wd = probe.returncode == 0
        except (OSError, subprocess.SubprocessError):
            remote_wd = None  # unknown (laptop waking, ssh hiccup)

        # A local serve tunnel (run.sh) marks active use.
        try:
            serve_local = (
                subprocess.run(
                    ["pgrep", "-f", "run.sh serve"], capture_output=True
                ).returncode
                == 0
            )
        except OSError:
            serve_local = False

        remaining = deadline - time.monotonic() if deadline is not None else None
        if (time.monotonic() - last_hb) >= args.heartbeat_min * 60:
            last_hb = time.monotonic()
            if remote_wd:
                ntfy_publish(
                    args.ntfy,
                    f"{label}: remote watchdog active, instance running "
                    f"(deadline in {format_duration(remaining)})",
                    tags="computer",
                )
            else:
                ntfy_publish(
                    args.ntfy,
                    f"{label}: FALLBACK — no remote watchdog on the instance "
                    f"and it is still billed "
                    f"({'deadline in ' + format_duration(remaining) if remaining else 'no deadline'})",
                    priority=4,
                    tags="warning",
                )

        # Fallback teardown: no remote watchdog, no active serve, deadline hit.
        # Only act when the remote state was CONFIRMED absent (False) — an
        # unknown ssh result may be a laptop-waking hiccup.
        if (
            remote_wd is False
            and not serve_local
            and remaining is not None
            and remaining <= 0
        ):
            if args.teardown:
                ntfy_publish(
                    args.ntfy,
                    f"{label}: fallback deadline reached — destroying instance",
                    priority=5,
                    tags="skull",
                )
                subprocess.run(
                    [sys.executable, str(SCRIPT_DIR / "vastgpu.py"), "teardown", "-y"],
                    capture_output=True,
                )
            ntfy_publish(
                args.ntfy,
                f"{label}: fallback teardown attempt finished "
                f"({'(destroyed)' if args.teardown else '(notify-only, pass --teardown to arm)'})",
                priority=4,
            )
            return 0
        time.sleep(60)


# ── CLI ──


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="watchdog.py",
        description="Phone notifications + auto-teardown for long GPU runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--site",
        choices=["remote", "local"],
        default="remote",
        help="remote = on the GPU box (primary); local = dev machine fallback",
    )
    p.add_argument(
        "--ntfy", default=None, help="ntfy topic URL (default: WATCHDOG_NTFY_URL env)"
    )
    p.add_argument("--label", default=None, help="Shown in every notification")
    p.add_argument(
        "--heartbeat-min",
        type=float,
        default=DEFAULT_HEARTBEAT_MIN,
        help="Minutes between standing status messages",
    )
    p.add_argument(
        "--deadline-hours",
        type=float,
        default=DEFAULT_DEADLINE_HOURS,
        help="Hard teardown deadline from start; 0 = no deadline",
    )
    p.add_argument(
        "--soft-reminders",
        default=DEFAULT_SOFT_REMINDERS,
        help="Comma list of offsets before the deadline, e.g. 60m,15m,5m",
    )
    p.add_argument(
        "--no-teardown",
        action="store_true",
        help="Even with a key + instance id never destroy anything",
    )
    p.add_argument(
        "--port",
        type=int,
        default=None,
        help="vLLM port to health-check in heartbeats (remote only)",
    )

    remote_only = p.add_argument_group("remote options")
    remote_only.add_argument(
        "--instance-id",
        type=int,
        default=None,
        help="vast instance id (default: VAST_CONTAINERLABEL)",
    )
    remote_only.add_argument(
        "--envfile",
        default=None,
        help="Simple KEY=VALUE file to source (for the remote "
        "box, which has no dotenv/uv env)",
    )
    remote_only.add_argument(
        "--no-idle-destroy",
        action="store_true",
        help="Do not destroy when serve.py ends; notify only",
    )
    remote_only.add_argument(
        "--debug",
        action="store_true",
        help="Print one tick-status line to stderr each cycle",
    )
    remote_only.add_argument(
        "--click",
        default=None,
        help="URL attached to notifications (default: vast console)",
    )

    local_only = p.add_argument_group("local options")
    local_only.add_argument(
        "--teardown",
        action="store_true",
        help="Arm the fallback destroy for the local deadline",
    )
    p.set_defaults(deadline_hours=DEFAULT_DEADLINE_HOURS)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    return run_remote(args) if args.site == "remote" else run_local(args)


if __name__ == "__main__":
    sys.exit(main())
