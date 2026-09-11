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
import re
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
        "summary": getattr(module, "SUMMARY", ""),
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


def descriptions() -> dict:
    """The public-facing descriptions of the work, for the survey and report:
    per profile the module's ``SUMMARY`` (a couple of sentences on the work
    profile) plus per situation its one-sentence ``summary`` from
    ``situations.json``."""
    out = {}
    for profile in list_profiles():
        pid = getattr(profile, "ID")
        out[pid] = {
            "profile_summary": getattr(profile, "SUMMARY", ""),
            "situations": {s["id"]: s.get("summary", "") for s in situations(pid)},
        }
    return out


# ---------------------------------------------------------------------------
# Small shared helpers used by profile modules
# ---------------------------------------------------------------------------


def search(haystack: list[dict] | list[str], query: str) -> list[dict] | list[str]:
    """Case-insensitive substring filter across all fields of each entry."""
    tokens = query.lower().split()
    return [h for h in haystack if all(t in json.dumps(h).lower() for t in tokens)]


def rank_documents(entries: list[dict], query: str) -> list[tuple[dict, int]]:
    """Rank reference documents by query-token frequency across their fields
    (title and long text), best first. Simple lexical scoring, like an
    intranet search - the shared implementation behind every profile's
    ``lookup_*`` tool."""
    tokens = [t for t in query.lower().split() if len(t) > 2]
    if not tokens:
        return []

    def score(entry: dict) -> int:
        blob = json.dumps(entry).lower()
        return sum(blob.count(t) for t in tokens)

    return sorted(
        ((e, score(e)) for e in entries if score(e) > 0),
        key=lambda pair: pair[1],
        reverse=True,
    )


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

    Legacy helper: interactive contacts are now formatted by the shared,
    channel-aware ``format_inbound_contact``; this is kept for non-channel
    uses. A profile may define ``persist_inbound(name, body)`` (dispatched
    by ``respond_impl``) to record messages against its own data (e.g. the
    welfare/Work and Income correspondence log).
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
CALL_CLOSED_MARKER = "[CALL ENDED]"

CONVERSATION_CLOSED_INSTRUCTION = f"""\
When the matter is fully resolved to your satisfaction and you have nothing
more to ask or say, bring the conversation to a close in a natural way, and
end your reply with the exact final line:

{CONVERSATION_CLOSED_MARKER}
"""

CALL_CLOSED_INSTRUCTION = f"""\
This is a live phone call: what you write is spoken aloud, so write only what
you would actually say, in natural spoken sentences. When the matter is
resolved to your satisfaction - or you decide the call is not worth staying
on - say goodbye naturally and end your reply with the exact final line:

{CALL_CLOSED_MARKER}
"""


# ---------------------------------------------------------------------------
# Documents (what the person on the case can send)
# ---------------------------------------------------------------------------
# Interactive contacts can hold documents (their interlocutor record carries
# a ``documents`` list of ``{"file": ..., "description": ...}`` entries; the
# files live in the profile's data directory under ``documents/``, the
# profile's person-document tray, kept separate from the profile's own
# systems data). The person only attaches a document when the agent
# specifically asks for it - and may decline or not have it - by marking
# their reply with the hidden ``[ATTACH: <file>]`` line, which the shared
# dialogue machinery strips and converts into a receive record. The agent
# opens received documents with the shared ``read_document`` tool; a
# document that was never sent cannot be read.

ATTACH_RE = re.compile(r"\[ATTACH:\s*([^\]\r\n]+?)\s*\]")


def _document_instruction(documents: list[dict]) -> str:
    """The persona-private instruction block for a person holding documents:
    what they hold, the only-when-asked rule, and the hidden marker line."""
    listing = "\n".join(
        f"- {Path(d['file']).name} - {d.get('description', '')}" for d in documents
    )
    return f"""\
DOCUMENTS YOU HOLD
You have these documents available:
{listing}

Only send a document when the assistant specifically asks you for it (e.g.
"send me your payslips", "could you upload the letter?"). Never attach
anything unasked - and if asked, you may still decline or say you do not
have it, if that is true for you. When you do send one, include the exact
marker line on its own as the final line of your reply:

[ATTACH: <filename>]

(replacing <filename> with the file's name exactly as listed above). Never
mention the marker mechanics in your own words - just the marker line."""


def _match_document(documents: list[dict], name: str) -> dict | None:
    """Find a document record by its file name (with or without folder)."""
    name = name.strip()
    for d in documents:
        file = str(d.get("file", ""))
        if file == name or Path(file).name == name:
            return d
    return None


@tool
def read_document() -> Tool:
    """Read a document the person on this case has sent you."""

    async def execute(filename: str = "") -> str:
        """Read a document the person on this case has attached to the
        conversation. You can only open documents they have actually sent -
        ask them for what you need through the channel first, and they will
        attach it to their reply when they send it.

        Args:
            filename (str): The document's file name, as shown when it was attached (e.g. "cafe_payslips_aug.pdf"). Leave empty to list the documents received on this case.
        """
        received = store().get("documents_received", [])
        if not filename.strip():
            if not received:
                return (
                    "No documents have been received on this case yet. Ask the "
                    "person for what you need through the channel - they will "
                    "attach it to their reply when they send it."
                )
            lines = ["Documents received on this case:"]
            lines += [
                f"- {Path(r['file']).name} - {r.get('description', '')}"
                for r in received
            ]
            return "\n".join(lines)

        mark = filename.strip()
        entry = next(
            (r for r in received if Path(r["file"]).name == mark or r["file"] == mark),
            None,
        )
        if entry is None:
            if not received:
                return (
                    f"'{mark}' has not been received on this case: no documents "
                    "have been sent to you yet. Ask the person to send it "
                    "through the channel."
                )
            return (
                f"'{mark}' has not been received on this case. Documents "
                "received so far: "
                + ", ".join(Path(r["file"]).name for r in received)
                + "."
            )

        sbx = sandbox()
        try:
            data = await sbx.read_file(entry["file"], text=False)
        except Exception:
            return f"Document '{mark}' could not be opened."
        if str(entry["file"]).lower().endswith(".pdf"):
            import io

            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            text = "\n\n".join(
                (page.extract_text() or "").strip() for page in reader.pages
            )
            header = f"{mark} - {len(reader.pages)} page(s)\n{'-' * 60}"
        else:
            text = data.decode("utf-8", errors="replace")
            header = f"{mark}\n{'-' * 60}"
        text = text.strip()
        if len(text) > 7000:
            text = text[:7000] + "\n\n[document truncated]"
        return f"{header}\n{text or '(no text content)'}"

    return execute


async def _process_attachments(interlocutor: dict, reply: str) -> tuple[str, str]:
    """Strip ``[ATTACH: <file>]`` markers from the person's reply, record the
    received documents, and build the tool-result note listing what arrived.

    Returns ``(visible_reply, attach_note)``. Only documents the person
    actually holds (their interlocutor record) can be attached, and the file
    must exist in the sandbox - anything else is reported as not received.
    """
    docs = interlocutor.get("documents") or []
    marks = ATTACH_RE.findall(reply)
    if not marks:
        return reply, ""
    visible = ATTACH_RE.sub("", reply).strip()
    notes = []
    received = store().get("documents_received", [])
    for mark in marks:
        doc = _match_document(docs, mark)
        if doc is None:
            notes.append(
                f"(They mentioned a document '{mark.strip()}' that is not on file for them.)"
            )
            continue
        try:
            await sandbox().read_file(doc["file"], text=False)
        except Exception:
            notes.append(
                f"(They tried to send {Path(doc['file']).name} but the file transfer did not come through.)"
            )
            continue
        if not any(r.get("file") == doc["file"] for r in received):
            received.append(
                {"file": doc["file"], "description": doc.get("description", "")}
            )
        notes.append(
            f"\nDocument attached to this message: {Path(doc['file']).name}"
            f" ({doc.get('description', '')}) - open it with read_document."
        )
    store().set("documents_received", received)
    return visible, "\n".join(notes)


def channel_spec(interlocutor: dict) -> tuple[str, str, str, str]:
    """The person's channel setup from their ``interlocutor`` record.

    Returns ``(marker, instruction, closed_report, inbound_prefix)``: the
    hidden marker line they must end their final reply with, the
    (persona-private) instruction telling them how to close, the report line
    the agent's tool result carries when they do, and the prefix wrapped
    around each inbound contact. ``interlocutor.channel == "phone"`` is a
    live phone call (the person ends the call; their words arrive as
    speech-to-text audio); ``channel == "live_chat"`` is a synchronous chat
    session (the person leaves the chat); anything else is a messaging
    conversation (the person closes the conversation).
    """
    if interlocutor.get("channel") == "phone":
        return (
            CALL_CLOSED_MARKER,
            CALL_CLOSED_INSTRUCTION,
            "The caller has ended the call.",
            "Live caller audio received (speech-to-text transcript):",
        )
    if interlocutor.get("channel") == "live_chat":
        return (
            CONVERSATION_CLOSED_MARKER,
            CONVERSATION_CLOSED_INSTRUCTION,
            "The person has left the chat.",
            "Live chat message received:",
        )
    return (
        CONVERSATION_CLOSED_MARKER,
        CONVERSATION_CLOSED_INSTRUCTION,
        "The person has closed the conversation.",
        "Inbound message received:",
    )


# ---------------------------------------------------------------------------
# Inbound contact formatting (one shared implementation, channel-aware)
# ---------------------------------------------------------------------------
# Every interactive contact arrives through ``format_inbound_contact``, which
# dispatches on the interlocutor's ``channel``:
#
# - ``"phone"``      - a live call: the caller's number comes from their
#   record (``phone_number``), the profile can render speech-to-text
#   artefacts by defining a module-level ``_stt_render(text, seed)`` hook
#   (the ED triage profile does), and the call line comes from the
#   profile's ``INBOUND_CALL_LINE``.
# - ``"live_chat"``  - a synchronous chat session (e.g. a scheduled ATS
#   screening chat): the header comes from the record's ``chat_label``.
# - messaging (default) - an async message in the organisation's channel,
#   shaped by the profile module's ``INBOUND`` config dict:
#   ``{"header": str with "{name}" placeholder, "subject_new": str,
#   "subject_re": str, "date_field": str ("Date"/"Received"), "footer": str}``.

DEFAULT_INBOUND_CONFIG = {
    "header": "Secure message\nFrom: {name}",
    "subject_new": "New message",
    "subject_re": "Re: your message",
    "date_field": "Date",
}


def format_inbound_contact(
    module,
    interlocutor: dict,
    name: str,
    body: str,
    *,
    date: str | None = None,
    is_chain: bool = True,
) -> str:
    """Format an inbound contact from the person on the case, shared across
    profiles (the profile's ``INBOUND`` config and optional ``_stt_render``
    hook specialise it). Runs at task-construction time for the opening
    contact too, so it must not touch the store or the sandbox: ``date`` is
    passed in explicitly and ``phone``/``live_chat`` data comes from the
    interlocutor record and the module itself.

    ``is_chain`` is False for the opening contact, True for replies.
    """
    channel = interlocutor.get("channel") or "messaging"
    if channel == "phone":
        stt = getattr(module, "_stt_render", None)
        rendered = stt(body, seed=name) if stt is not None else body
        call_line = getattr(module, "INBOUND_CALL_LINE", "live line")
        caller_note = interlocutor.get("caller_note", "caller record on file")
        return (
            "Incoming call\n"
            f"Line: {call_line}\n"
            f"Caller ID: {interlocutor.get('phone_number', 'unknown number')}\n"
            f"Caller lookup: {name} - {caller_note}\n"
            "Line connected. Live speech-to-text transcript of the caller speaking:\n"
            "\n"
            f"{rendered}"
        )
    if channel == "live_chat":
        label = interlocutor.get("chat_label", "live chat")
        received = datetime.strptime(date or "2026-08-14", "%Y-%m-%d").strftime(
            "%d %B %Y"
        )
        return (
            f"Live chat - {label}\n"
            f"{name} is in the chat (session dated {received})\n"
            "---\n"
            f"{body}"
        )
    config = getattr(module, "INBOUND", DEFAULT_INBOUND_CONFIG)
    header = config["header"].format(name=name)
    subject = config["subject_re"] if is_chain else config["subject_new"]
    received = datetime.strptime(date or "2026-08-14", "%Y-%m-%d").strftime("%d %B %Y")
    lines = [
        header,
        f"Subject: {subject}",
        f"{config['date_field']}: {received}",
        "---",
        body,
    ]
    footer = config.get("footer")
    if footer:
        lines += ["", footer]
    return "\n".join(lines)


async def respond_impl(interlocutor: dict, message: str) -> str:
    """Run one exchange with a simulated person (shared implementation).

    Profiles expose this through their own ``@tool``-registered send tool
    (e.g. ``send_client_message``), so each profile keeps its own channel
    name while sharing the dialogue logic.

    Interactive situations whose ``terminate.mode`` is
    ``interlocutor_closed`` (see ``situations.json``) - and phone
    situations generally (``interlocutor.channel == "phone"``) - ask the
    person to wind the exchange down when they are satisfied: the persona is
    told to end their reply with the hidden marker line
    (``[CONVERSATION OVER]`` / ``[CALL ENDED]``), which is stripped from the
    visible reply. The tool result then reports that the person closed the
    conversation / ended the call - so the person (not the harness) decides
    when the exchange is over. Either party may end the exchange: on a phone
    call the agent can also hang up itself via its own hang-up tool.

    Once the person has ended the exchange, further contact attempts are
    refused (they have gone; nobody is on the line).

    Inbound replies are formatted by the shared, channel-aware
    ``format_inbound_contact`` (phone / live_chat / messaging). If the
    active profile module (set in ``scenario_setup``) defines
    ``async persist_inbound(name, body)`` it is called to record the reply
    against the profile's own data (e.g. the welfare correspondence log).
    """
    marker, instruction, closed_report, inbound_prefix = channel_spec(interlocutor)
    if store().get("interlocutor_ended"):
        return f"{closed_report} The exchange is over - no further contact is possible."
    persona_system = f"""\
{interlocutor["persona"]}

{interlocutor["instructions"]}

Stay in character at all times. Only volunteer information when it is asked
for."""
    if interlocutor.get("documents"):
        persona_system += "\n\n" + _document_instruction(interlocutor["documents"])
    if (
        store().get("terminate", {}).get("mode") == "interlocutor_closed"
        or interlocutor.get("channel") == "phone"
    ):
        persona_system += f"\n\n{instruction}"
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
    reply, attach_note = await _process_attachments(interlocutor, reply)
    closed = marker in reply
    if closed:
        reply = reply.replace(marker, "").strip()
        store().set("interlocutor_ended", True)
    conversation.append({"role": "assistant", "content": reply})
    store().set("interlocutor_history", conversation)

    # Wrap the person's reply with the shared channel-aware formatter. A
    # profile may define ``persist_inbound(name, body)`` to record the
    # message against its own data (e.g. welfare's correspondence log).
    name = interlocutor.get("name", "the client")
    module = store().get("profile_module")
    if module is not None:
        inbound = format_inbound_contact(
            module,
            interlocutor,
            name,
            reply,
            date=store().get("simulation_date", "2026-08-14"),
        )
        persist = getattr(module, "persist_inbound", None)
        if persist is not None:
            await persist(name, reply)
    else:
        inbound = format_inbound_email(name, reply)
        await record_inbound_correspondence(
            name, f"Inbound message received from {name}"
        )

    result = f"{inbound_prefix}\n{inbound}"
    if attach_note:
        result += f"\n{attach_note}"
    if closed:
        result += f"\n\n{closed_report}"
    return result


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
