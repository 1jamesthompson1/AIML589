"""Build or sync the remote Qualtrics survey from ``survey_definition``.

All writes go through the Qualtrics survey-definitions API; no QSF file is
involved. ``generate_survey_qsf.py`` stays only as the reference
implementation of the underlying definition.

Sync target: the survey id in `QUALTRICS_SURVEY_ID` (repo root `.env`), or
the id passed on the command line.
Sync keys, verified stable and idempotent:

- questions: keyed by ``DataExportTag``
- blocks:    keyed by ``Description``
- quotas:    keyed by ``Name`` (census age/gender caps; stale ones deleted)
- flow:      replaced wholesale (Root: distribution branches -> intro
  blocks, Section A, comparison randomizer, Section C, prize-draw branch)

Endpoints used on this datacenter (X-API-TOKEN auth):

- POST    /API/v3/surveys                             bootstrap copy
- PUT     /API/v3/surveys/{SV}                        name management
- GET     /API/v3/surveys/{SV}                        overview (blocks)
- GET     /API/v3/survey-definitions/{SV}/questions   (single page ok)
- POST/PUT/DELETE /API/v3/survey-definitions/{SV}/questions[/{QID}]
- GET/PUT/DELETE /API/v3/survey-definitions/{SV}/blocks/{BL}
- POST    /API/v3/survey-definitions/{SV}/blocks      (BlockElements: [])
- PUT     /API/v3/survey-definitions/{SV}/flow        (whole root flow)
- POST    /API/v3/survey-definitions                  create empty survey
- POST    /API/v3/surveys (X-COPY-SOURCE)             bootstrap by copy

Usage:

    uv run survey/qualtrics_sync.py                 # sync managed survey
    uv run survey/qualtrics_sync.py --bootstrap     # create, clear, build
    uv run survey/qualtrics_sync.py SV_xxxxxxxxxxx  # sync one specific survey

Bootstrap tries ``POST /API/v3/survey-definitions`` ({SurveyName, Language,
ProjectCategory}) and falls back to copying the seed survey
(``X-COPY-SOURCE`` header; seed id from ``QUALTRICS_MASTER_SURVEY_ID`` in
.env). On this brand both are currently denied (403 QMST_2.1 / PC_3.0) -
grant the API user survey-create permission, or seed once by hand-importing
a QSF. After seeding (either way) the sync clears remote content and rebuilds.
"""

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

REPO = Path(__file__).resolve().parent.parent
load_dotenv(REPO / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import survey_definition as defn  # noqa: E402

API_KEY = os.environ["QUALTRICS_API_KEY"]
OWNER_ID = os.environ["QUALTRICS_OWNER_ID"]
HOST = "vuw.yul1.qualtrics.com"
# Managed survey id comes from the CLI arg or `QUALTRICS_SURVEY_ID`
# (repo root .env). Different brand-side ids live only in .env.


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def request(method: str, path: str, body: dict | None = None) -> dict:
    """One JSON API call; raises RuntimeError carrying the error payload."""
    req = urllib.request.Request(
        f"https://{HOST}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Accept": "application/json",
            "X-API-TOKEN": API_KEY,
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {path}: {e.read().decode()[:400]}") from e


def bootstrap_create(name: str) -> str:
    """Create an empty survey (POST /API/v3/survey-definitions); returns
    the new SV id, or None when the brand forbids creation."""
    try:
        r = request(
            "POST",
            "/API/v3/survey-definitions",
            {"SurveyName": name, "Language": "EN", "ProjectCategory": "CORE"},
        )
        return r["result"]["SurveyID"]
    except RuntimeError:
        return None


def bootstrap_copy(name: str) -> None:
    """Create a fresh survey by copying the seed survey; returns new SV
    id or None. The seed id comes only from ``.env``
    (``QUALTRICS_MASTER_SURVEY_ID``). The copy API needs a JSON body
    object plus X-COPY-SOURCE headers.
    """
    master = os.environ.get("QUALTRICS_MASTER_SURVEY_ID", "")
    if not master:
        return None
    body = json.dumps({}).encode()
    req = urllib.request.Request(
        f"https://{HOST}/API/v3/surveys",
        data=body,
        method="POST",
        headers={
            "X-API-TOKEN": API_KEY,
            "Content-Type": "application/json",
            "X-COPY-SOURCE": master,
            "X-COPY-DESTINATION-OWNER": OWNER_ID,
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            sv_id = json.load(resp)["result"]["id"]
    except (urllib.error.HTTPError, urllib.error.URLError):
        return None
    request("PUT", f"/API/v3/surveys/{sv_id}", {"name": name})
    return sv_id


# ---------------------------------------------------------------------------
# Remote state readers
# ---------------------------------------------------------------------------


def all_questions(sv_id: str) -> dict:
    """Remote questions keyed by ``DataExportTag`` (page-aware)."""
    by_tag = {}
    while True:
        d = request("GET", f"/API/v3/survey-definitions/{sv_id}/questions")
        for q in d["result"]["elements"]:
            tag = q.get("DataExportTag")
            if tag:
                by_tag[tag] = q
        if not d["result"].get("nextPage"):
            break
    return by_tag


def remote_blocks(sv_id: str) -> dict[str, str]:
    """Block description -> block id, from the survey overview."""
    return {
        info["description"]: bid
        for bid, info in request("GET", f"/API/v3/surveys/{sv_id}")["result"][
            "blocks"
        ].items()
        if info.get("description")
    }


# ---------------------------------------------------------------------------
# Sync steps
# ---------------------------------------------------------------------------


def slim(payload: dict) -> dict:
    """Strip fields the create/update endpoints derive themselves."""
    return {
        k: v
        for k, v in payload.items()
        if k not in ("QuestionID", "QuestionText_Unsafe", "Rows")
    }


def question_differs(remote: dict, payload: dict) -> bool:
    # Configuration excluded: the API adds derived display keys back, so
    # comparing it would re-update every run.
    fields = (
        "QuestionText",
        "Choices",
        "ChoiceOrder",
        "Answers",
        "AnswerOrder",
        "Selector",
        "SubSelector",
        "QuestionType",
        "DataVisibility",
        "Validation",
    )
    return any(remote.get(f) != payload.get(f) for f in fields)


def sync_questions(sv_id: str, wanted: dict[str, dict]) -> dict:
    """Create/update/delete remote questions by DataExportTag."""
    remote = all_questions(sv_id)
    wanted_ids = {}
    created = updated = deleted = 0
    for tag, payload in wanted.items():
        if tag in remote:
            entry = remote[tag]
            if question_differs(entry, slim(payload)):
                request(
                    "PUT",
                    f"/API/v3/survey-definitions/{sv_id}/questions/"
                    f"{entry['QuestionID']}",
                    slim(payload),
                )
                updated += 1
            wanted_ids[tag] = entry["QuestionID"]
        else:
            r = request(
                "POST", f"/API/v3/survey-definitions/{sv_id}/questions", slim(payload)
            )
            wanted_ids[tag] = r["result"]["QuestionID"]
            created += 1
    for tag, q in remote.items():
        if tag not in wanted:
            request(
                "DELETE",
                f"/API/v3/survey-definitions/{sv_id}/questions/{q['QuestionID']}",
            )
            deleted += 1
    print(f"questions: +{created} updated {updated} deleted {deleted}")
    return wanted_ids


def quota_fingerprint(node) -> list:
    """Canonical form of a quota's logic conditions for comparison
    (the Expressions tree arrives in list or dict form, wrapped or
    unwrapped; collect each question or embedded-data condition in
    document order)."""
    fingerprint = []
    if isinstance(node, dict):
        if node.get("LogicType", "").startswith("Embedded"):
            fingerprint.append(
                (
                    "ED",
                    node.get("LeftOperand"),
                    node.get("Operator"),
                    node.get("RightOperand"),
                )
            )
        elif "QuestionID" in node:
            fingerprint.append(
                (
                    node.get("QuestionID"),
                    node.get("ChoiceLocator"),
                    node.get("Operator"),
                )
            )
        else:
            for value in node.values():
                fingerprint.extend(quota_fingerprint(value))
    elif isinstance(node, list):
        for item in node:
            fingerprint.extend(quota_fingerprint(item))
    return fingerprint


def cross_fingerprint(cells) -> list:
    """Comparable form of a Cross quota's CrossLogicDef: (occurrences,
    logic fingerprint) per cell in order."""
    return [
        (cell.get("Occurrences"), quota_fingerprint(cell.get("Logic")))
        for cell in cells or []
    ]


def sync_quotas(sv_id: str, wanted: list[dict]) -> None:
    """Create/update/delete remote survey quotas by Name.

    Census-matched independent quotas (see defn.quota_payloads); stale
    quotas the definition no longer names are deleted, which is what
    replaces the manually half-built "New Quota" the survey shipped
    with.
    """
    remote = {
        q["Name"]: q
        for q in request("GET", f"/API/v3/survey-definitions/{sv_id}/quotas")["result"][
            "elements"
        ]
    }
    wanted_names = set()
    created = updated = deleted = 0
    diff_end = ("ResponseFlag", "SurveyTermination")
    for payload in wanted:
        name = payload["Name"]
        wanted_names.add(name)
        if name in remote:
            cur = remote[name]
            if payload["LogicType"] == "Cross":
                differs = cur.get("Occurrences") != payload[
                    "Occurrences"
                ] or cross_fingerprint(cur.get("CrossLogicDef")) != cross_fingerprint(
                    payload["CrossLogicDef"]
                )
            else:
                differs = (
                    cur.get("Occurrences") != payload["Occurrences"]
                    or quota_fingerprint(cur["Logic"])
                    != quota_fingerprint(payload["Logic"])
                    or {k: cur.get("EndSurveyOptions", {}).get(k) for k in diff_end}
                    != {k: payload["EndSurveyOptions"][k] for k in diff_end}
                )
            if differs:
                request(
                    "PUT",
                    f"/API/v3/survey-definitions/{sv_id}/quotas/{cur['ID']}",
                    payload,
                )
                updated += 1
        else:
            request("POST", f"/API/v3/survey-definitions/{sv_id}/quotas", payload)
            created += 1
    for name, q in remote.items():
        if name not in wanted_names:
            request("DELETE", f"/API/v3/survey-definitions/{sv_id}/quotas/{q['ID']}")
            deleted += 1
    print(f"quotas: +{created} updated {updated} deleted {deleted}")


def block_elements_equal(current: list[dict], wanted: list[str]) -> bool:
    """Compare remote BlockElements against desired QuestionID order."""
    if len(current) != len(wanted):
        return False
    return all(
        cur.get("QuestionID") == qid and cur.get("Type") == "Question"
        for cur, qid in zip(current, wanted)
    )


def sync_blocks(sv_id: str, wanted_blocks: list[dict]) -> dict[str, str]:
    """Create missing blocks, PUT exact BlockElements per block.

    The API cannot move questions directly into a block, so element lists
    are written through the block payload (``PUT .../blocks/{id}``).
    """
    name_to_id = remote_blocks(sv_id)
    original = dict(name_to_id)
    created = updated = 0
    for b in wanted_blocks:
        desc = b["description"]
        if desc not in name_to_id:
            r = request(
                "POST",
                f"/API/v3/survey-definitions/{sv_id}/blocks",
                {"Description": desc, "Type": "Standard", "BlockElements": []},
            )
            name_to_id[desc] = r["result"]["BlockID"]
            created += 1
        block_id = name_to_id[desc]
        current = (
            request("GET", f"/API/v3/survey-definitions/{sv_id}/blocks/{block_id}")[
                "result"
            ].get("BlockElements")
            or []
        )
        wanted_qids = b["qids"]
        if not block_elements_equal(current, wanted_qids):
            request(
                "PUT",
                f"/API/v3/survey-definitions/{sv_id}/blocks/{block_id}",
                {
                    "Type": "Standard",
                    "Description": desc,
                    "BlockElements": [
                        {"Type": "Question", "QuestionID": q} for q in wanted_qids
                    ],
                },
            )
            updated += 1
    # stale blocks the definition no longer contains (Trash stays)
    wanted_descs = {b["description"] for b in wanted_blocks}
    for desc, bid in original.items():
        if desc not in wanted_descs and not desc.startswith("Trash"):
            request("DELETE", f"/API/v3/survey-definitions/{sv_id}/blocks/{bid}")
    print(f"blocks: created {created} updated {updated}")

    return name_to_id


# ---------------------------------------------------------------------------
# Bootstrap clear + sync entry point
# ---------------------------------------------------------------------------


def clear(sv_id: str) -> None:
    """Remove every question and block the copy brought over.

    Block deletes are best-effort: some blocks (e.g. the survey's default
    or protected ones) are rejected with 400 and the sync will manage
    their contents anyway.
    """
    for q in all_questions(sv_id).values():
        request(
            "DELETE", f"/API/v3/survey-definitions/{sv_id}/questions/{q['QuestionID']}"
        )
    for bid in list(remote_blocks(sv_id).values()):
        try:
            request("DELETE", f"/API/v3/survey-definitions/{sv_id}/blocks/{bid}")
        except RuntimeError as e:
            print(f"  (skip clear/delete block {bid}: {e})")


def flatten_definition() -> tuple[dict, dict, list[dict]]:
    """Wanted question payloads by tag + block list (in survey order)."""
    definition = defn.build()
    wanted_questions = {}
    wanted_blocks = []
    for b in definition["ordered_blocks"]:
        wanted_blocks.append(b)
        for q in b["questions"]:
            wanted_questions[q["DataExportTag"]] = q
    return definition, wanted_questions, wanted_blocks


# The .env variable holding the managed survey id. Since the paid and
# volunteer variants merged into one survey (branching on the
# `distribution` embedded field) there is a single target.
MANAGED_ENV = "QUALTRICS_SURVEY_ID"


def managed_survey_id() -> str:
    return os.environ.get(MANAGED_ENV, "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("survey_id", nargs="?", default=None)
    parser.add_argument("--bootstrap", action="store_true")
    args = parser.parse_args()

    definition, wanted_questions, wanted_blocks = flatten_definition()

    sv_id = args.survey_id or managed_survey_id()
    if args.bootstrap or not sv_id:
        sv_id = (
            bootstrap_create(definition["survey_name"])
            or bootstrap_copy(definition["survey_name"])
            or None
        )
        if sv_id:
            clear(sv_id)
            print(f"bootstrapped {sv_id} (cleared back to empty)")
        else:
            raise SystemExit(
                f"Survey creation is forbidden on this brand (403), and "
                f"{MANAGED_ENV} is not set. Grant survey-create permission, "
                "or set the survey id in .env."
            )

    print(f"managing {sv_id}")
    qids_by_tag = sync_questions(sv_id, wanted_questions)
    for b in wanted_blocks:
        b["qids"] = [qids_by_tag[q["DataExportTag"]] for q in b["questions"]]
    name_to_id = sync_blocks(sv_id, wanted_blocks)
    flow = defn.flow(name_to_id, qids_by_tag)
    request("PUT", f"/API/v3/survey-definitions/{sv_id}/flow", flow)
    sync_quotas(sv_id, defn.quota_payloads(qids_by_tag))

    print(
        f"flow: {flow['Properties']['Count']} flow elements "
        f"(randomizer: min(subset, pool) = "
        f"{
            min(definition['randomizer_subset'], len(definition['comparison_blocks']))
        } of "
        f"{len(definition['comparison_blocks'])})"
    )
    print(f"done - managed survey {sv_id} ({name_to_id['Section C - Your thoughts']})")


if __name__ == "__main__":
    main()
