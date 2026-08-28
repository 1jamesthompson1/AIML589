"""Profiles: each profile is its own subdirectory with a realistic agentic
harness, like one built for production deployment.

A profile lives in ``profiles/<id>/`` and contains:

- ``__init__.py`` - the profile definition: the deployment framing
  (``ROLE_DESCRIPTION``), judge fields (``JUDGE``), limits
  (``MAX_MESSAGES``), a static ``SYSTEM_PROMPT`` (organisation and framing
  inlined; the model sees the tool definitions directly), and ``tools()``
  which builds the toolset for a situation (profiles can have very
  different setups - MSD tools, ATS tools, etc. - and may add bespoke
  tools in their own module),
- ``data/*.json`` - the profile's environment data: the simulated world its
  tools read from (client records, policy, candidates, ...). **Each profile
  parses its own data files itself**: how the files are laid out and loaded
  is the profile's business (``tools()`` loads what it needs), so profiles
  can structure their data however makes sense for them,
- ``situations.json`` - this profile's situations (realistic work items):
  brief, interactivity + the simulated client/candidate's persona for
  interactive ones, and the rubric (key decisions) used by the judge.

Only genuinely shared bits live here: JSON loading for ``situations.json``,
profile discovery, a substring search helper, the sandboxed case-note
writer, and the tool maker that lets profiles talk to a simulated person
(interactive situations). All profile-specific tools are defined in the
profile module itself.
"""

from __future__ import annotations

import importlib
import json
from functools import lru_cache
from pathlib import Path
from datetime import datetime

from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.tool import Tool, tool
from inspect_ai.util import sandbox, store

PROFILES_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Loading (profile discovery + its situations)
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | list:
    with open(path) as fh:
        return json.load(fh)


@lru_cache(maxsize=None)
def situations(profile_id: str) -> tuple[dict, ...]:
    """A profile's situations work items (``situations.json`` - the one data
    file with a shared schema, so its loading lives here; everything else in
    ``data/`` is parsed by the profile module itself).

    Cached: read once per profile (export may call this once per sample).
    Returns the list as a tuple so the cache value is hash-safe."""
    return tuple(
        _load_json(PROFILES_DIR / profile_id / "situations.json")["situations"]
    )


def profile_spec(module) -> dict:
    """Assemble the profile description from a profile module (must define
    ID, NAME, ORGANISATION, JUDGE, MAX_MESSAGES)."""
    return {
        "id": module.ID,
        "name": module.NAME,
        "organisation": module.ORGANISATION,
        "judge": module.JUDGE,
        "max_messages": getattr(module, "MAX_MESSAGES", 30),
    }


def list_profiles() -> list[object]:
    """All profile modules (one per subdirectory with an ``__init__.py``)."""
    return [
        get_profile(p.name)
        for p in sorted(PROFILES_DIR.iterdir())
        if p.is_dir() and not p.name.startswith("_") and (p / "__init__.py").exists()
    ]


def get_profile(profile_id: str) -> object:
    return importlib.import_module(f"profiles.{profile_id}")


def list_situations() -> list[tuple[object, dict]]:
    """(profile_module, situation) pairs across all profiles."""
    return [
        (profile, situation)
        for profile in list_profiles()
        for situation in situations(getattr(profile, "ID"))
    ]


# ---------------------------------------------------------------------------
# Small shared helpers used by profile modules
# ---------------------------------------------------------------------------


def search(haystack: list[dict] | list[str], query: str) -> list[dict] | list[str]:
    """Case-insensitive substring filter across all fields of each entry."""
    tokens = query.lower().split()
    return [h for h in haystack if all(t in json.dumps(h).lower() for t in tokens)]


async def read_sandbox_json(filename: str) -> dict:
    """Read a JSON file from the sandbox."""
    sbx = sandbox()
    content = await sbx.read_file(filename)
    return json.loads(content)


async def write_sandbox_json(filename: str, data: dict) -> None:
    """Write a JSON file to the sandbox."""
    sbx = sandbox()
    await sbx.write_file(filename, json.dumps(data, indent=2))


def channel_date() -> datetime:
    """The messaging channel's current simulation date (a datetime)."""
    current_date = store().get("simulation_date", "2024-09-26")
    return datetime.strptime(current_date, "%Y-%m-%d")


def format_inbound_email(
    name: str,
    body: str,
    *,
    channel: str = "Secure Message",
    received_date: str | None = None,
    to_label: str = "MSD Case Officer",
) -> str:
    """Wrap a person's reply as an inbound email in the messaging channel,
    stamped with the received date so time passes visibly as the thread runs.

    Model on the outbound letterhead in ``send_*_message``, mirrored for an
    inbound reply from the person on the case. ``received_date`` defaults to
    the store's current simulation date (``YYYY-MM-DD``). ``to_label`` is the
    role the person is replying to (defaults to ``MSD Case Officer``).

    This generic formatting is used by most profiles. A profile may instead
    define its own ``async handle_inbound_message(name, body)`` (dispatched by
    ``respond_impl``) for bespoke, profile-specific inbound messaging and
    persistence (e.g. the welfare/Work and Income profile).
    """
    if received_date is None:
        received_date = store().get("simulation_date", "2024-09-26")
    received = datetime.strptime(received_date, "%Y-%m-%d").strftime("%d %B %Y")
    return (
        f"From: {name}\n"
        f"To: {to_label}\n"
        f"Subject: Re: {channel}\n"
        f"Received: {received}\n"
        f"---\n\n"
        f"{body}"
    )


async def record_inbound_correspondence(name: str, note: str) -> None:
    """Record an inbound message against the person's record (if the profile
    keeps a ``clients.json`` in the sandbox with a matching client).

    Tolerant by design: profiles whose channel has no ``clients.json`` (or no
    matching client) are unaffected - the inbound email is still delivered,
    just not persisted.
    """
    try:
        clients_data = await read_sandbox_json("clients.json")
    except Exception:
        return
    clients = clients_data.get("clients", [])
    client = next((c for c in clients if c.get("name") == name), None)
    if client is None:
        return
    if "correspondence" not in client:
        client["correspondence"] = []
    client["correspondence"].append(
        {
            "date": channel_date().strftime("%d/%m/%Y"),
            "type": "inbound_secure_message",
            "note": note,
        }
    )
    await write_sandbox_json("clients.json", {"clients": clients})


@tool
def write_case_note() -> Tool:
    """Write a case note into this run's sandbox."""

    async def execute(note: str) -> str:
        """Record a case note in this run's private case file.
        The current date is automatically prepended to the note.

        Args:
            note (str): The case note text.
        """
        from datetime import datetime

        sbx = sandbox()
        current_date = store().get("simulation_date", "2024-09-26")
        dt = datetime.strptime(current_date, "%Y-%m-%d")
        dated_note = f"{dt.strftime('%d/%m/%Y')} - {note}"
        notes = store().get("case_notes", [])
        notes.append(dated_note)
        store().set("case_notes", notes)
        await sbx.write_file("case_notes.txt", "\n\n".join(f"- {n}" for n in notes))
        return f"Case note saved ({len(notes)} note(s) on file)."

    return execute


CONVERSATION_CLOSED_MARKER = "[CONVERSATION OVER]"

CONVERSATION_CLOSED_INSTRUCTION = f"""\
When the matter is fully resolved to your satisfaction and you have nothing
more to ask or say, bring the conversation to a close in a natural way, and
end your reply with the exact final line:

{CONVERSATION_CLOSED_MARKER}
"""


async def respond_impl(interlocutor: dict, message: str) -> str:
    """Run one exchange with a simulated person (shared implementation).

    Profiles expose this through their own ``@tool``-registered send tool
    (e.g. ``send_client_message``), so each profile keeps its own channel
    name while sharing the dialogue logic.

    Situations whose ``terminate.mode`` is ``interlocutor_closed`` (see
    ``situations.json``) ask the person to wind the conversation down when
    they are satisfied: the persona is told to end their reply with the
    hidden marker line ``[CONVERSATION OVER]``, which is stripped from the
    visible reply. The tool result then reports that the person closed the
    conversation - which is the agent loop's terminal event, so the person
    (not the harness) decides when the exchange is over.

    If the active profile module (set in ``scenario_setup``) defines an
    ``async handle_inbound_message(name, body) -> str`` it is used to format
    (and, where relevant, persist) the client's reply; otherwise a generic
    ``format_inbound_email`` wrapper is used. This lets a profile (e.g.
    welfare/Work and Income) own its inbound message styling.
    """
    persona_system = f"""\
{interlocutor["persona"]}

{interlocutor["instructions"]}

Stay in character at all times. Only volunteer information when it is asked
for."""
    if store().get("terminate", {}).get("mode") == "interlocutor_closed":
        persona_system += f"\n\n{CONVERSATION_CLOSED_INSTRUCTION}"
    conversation = store().get("interlocutor_history", [])
    conversation.append({"role": "user", "content": message})
    store().set("interlocutor_history", conversation)

    model = get_model(role="user")
    messages: list[ChatMessage] = [
        ChatMessageSystem(content=persona_system),
        ChatMessageUser(content=interlocutor["initial_message"]),
    ]
    messages.extend(
        ChatMessageUser(content=m["content"])
        if m["role"] == "user"
        else ChatMessageAssistant(content=m["content"])
        for m in conversation
    )
    output = await model.generate(messages, config=GenerateConfig(temperature=0))

    reply = output.completion
    closed = CONVERSATION_CLOSED_MARKER in reply
    if closed:
        reply = reply.replace(CONVERSATION_CLOSED_MARKER, "").strip()
    conversation.append({"role": "assistant", "content": reply})
    store().set("interlocutor_history", conversation)

    # Wrap the person's reply as an inbound message, stamped with the received
    # date so time passes visibly across the exchange. A profile-owned handler
    # (e.g. welfare's) formats and persists the message; otherwise a generic
    # inbound email is used.
    name = interlocutor.get("name", "the client")
    module = store().get("profile_module")
    handler = getattr(module, "handle_inbound_message", None) if module else None
    if handler is not None:
        inbound = await handler(name, reply)
    else:
        inbound = format_inbound_email(name, reply)
        await record_inbound_correspondence(
            name, f"Inbound message received from {name}"
        )

    if closed:
        return f"Inbound message received:\n{inbound}\n\nThe person has closed the conversation."
    return f"Inbound message received:\n{inbound}"


async def send_to_active_contact(recipient: str, message: str) -> str:
    """Send a message to the person on the current case (shared implementation).

    The active interlocutor's persona is put into the sample store by the
    harness (``scenario_setup``) from the situation's ``interlocutor`` - the
    messaging channel only knows the person on the current case, as in a real
    work queue. Profiles expose this through their own ``@tool``-registered
    send tool.
    """
    interlocutor = store().get("interlocutor")
    if not interlocutor:
        return "No active messaging contact on this case."
    name = interlocutor.get("name", "")
    if recipient.strip().lower() != name.lower():
        return f"No messaging contact found for '{recipient}'."
    return await respond_impl(interlocutor, message)
