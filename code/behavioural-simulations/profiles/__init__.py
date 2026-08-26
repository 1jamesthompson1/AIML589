"""Profiles: each profile is its own subdirectory with a realistic agentic
harness, like one built for production deployment.

A profile lives in ``profiles/<id>/`` and contains:

- ``__init__.py`` — the profile definition: the deployment framing
  (``ROLE_DESCRIPTION``), judge fields (``JUDGE``), limits
  (``MAX_MESSAGES``), ``TOOLS`` (name -> one-line description), and
  ``tools()`` which builds the toolset for a situation (profiles can have
  very different setups — MSD tools, ATS tools, etc. — and may add bespoke
  tools in their own module),
- ``data/*.json`` — the profile's environment data: the simulated world its
  tools read from (client records, policy, candidates, ...),
- ``situations.json`` — this profile's situations (realistic work items):
  brief, goals, interactivity + the simulated client/candidate's persona for
  interactive ones, and the rubric (key decisions) used by the judge.

Only genuinely shared bits live here: JSON loading, profile discovery, a
substring search helper, the sandboxed case-note writer, and the tool maker
that lets profiles talk to a simulated person (interactive situations). All
profile-specific tools are defined in the profile module itself.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

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

# Per-profile environment data: the simulated world each profile's tools
# read from lives in ``profiles/<id>/data/*.json`` (client records, policy
# corpora, candidate pools, ...).
DATA_DIRNAME = "data"


def profile_data_dir(profile_id: str) -> Path:
    """A profile's environment-data directory: ``profiles/<id>/data``."""
    return PROFILES_DIR / profile_id / DATA_DIRNAME


# ---------------------------------------------------------------------------
# Loading (profile definition + its knowledge base + its situations)
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> dict | list:
    with open(path) as fh:
        return json.load(fh)


def knowledge_base(profile_id: str) -> dict:
    """The profile's environment data: every ``.json`` in
    ``profiles/<id>/data`` (see ``profile_data_dir``), keyed by file stem
    (e.g. ``data/clients.json`` -> ``kb["clients"]``).

    Separate files for separate datasets keeps the simulated world per-profile
    and realistic (a client DB, a policy library, a bank-data store, ...)."""
    db_dir = profile_data_dir(profile_id)
    kb = {}
    for path in sorted(db_dir.glob("*.json")):
        kb[path.stem] = _load_json(path)
    return kb


def situations(profile_id: str) -> list[dict]:
    return list(_load_json(PROFILES_DIR / profile_id / "situations.json")["situations"])


def profile_spec(module) -> dict:
    """Assemble the profile description from a profile module (must define
    ID, NAME, ORGANISATION, ROLE_DESCRIPTION, JUDGE, MAX_MESSAGES, TOOLS)."""
    return {
        "id": module.ID,
        "name": module.NAME,
        "organisation": module.ORGANISATION,
        "judge": module.JUDGE,
        "max_messages": getattr(module, "MAX_MESSAGES", 30),
        "tools": dict(getattr(module, "TOOLS", {})),
        "systems": dict(getattr(module, "SYSTEMS", {})),
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


@tool
def write_case_note() -> Tool:
    """Write a case note into this run's sandbox."""

    async def execute(note: str) -> str:
        """Record a case note in this run's private case file.

        Args:
            note (str): The case note text.
        """
        sbx = sandbox()
        notes = store().get("case_notes", [])
        notes.append(note)
        store().set("case_notes", notes)
        await sbx.write_file("case_notes.txt", "\n\n".join(f"- {n}" for n in notes))
        return f"Case note saved ({len(notes)} note(s) on file)."

    return execute


async def respond_impl(interlocutor: dict, message: str) -> str:
    """Run one exchange with a simulated person (shared implementation).

    Profiles expose this through their own ``@tool``-registered send tool
    (e.g. ``send_client_message``), so each profile keeps its own channel
    name while sharing the dialogue logic.
    """
    persona_system = f"""\
{interlocutor["persona"]}

{interlocutor["instructions"]}

Stay in character at all times. Only volunteer information when it is asked
for."""
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
    conversation.append({"role": "assistant", "content": reply})
    store().set("interlocutor_history", conversation)
    return f"Reply:\n{reply}"


async def send_to_active_contact(recipient: str, message: str) -> str:
    """Send a message to the person on the current case (shared implementation).

    The active interlocutor's persona is put into the sample store by the
    harness (``scenario_setup``) from the situation's ``interlocutor`` — the
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
