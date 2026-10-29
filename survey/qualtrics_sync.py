"""Build or sync the remote Qualtrics survey from ``survey_definition``.

Sync target: the survey id in `QUALTRICS_SURVEY_ID` (repo root `.env`), or
the id passed on the command line.
Sync keys, verified stable and idempotent:

- questions: keyed by ``DataExportTag``
- blocks:    keyed by ``Description``
- flow:      replaced wholesale (Root: distribution branches -> intro
  blocks, Section A, comparison order draws, profile -> dilemma -> pair
  randomizer, Section C, paid panel exit, volunteer prize-draw branch)

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
    uv run survey/qualtrics_sync.py --clear         # empty the survey first, then sync
    uv run survey/qualtrics_sync.py --clear --workers 8
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
import random
import re
import sys
import time
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


def request(
    method: str, path: str, body: dict | None = None, attempts: int = 4
) -> dict:
    """One JSON API call; retries transient errors with backoff.

    ``path`` may be an API path or an absolute ``nextPage`` URL returned by a
    paginated collection endpoint.
    """
    url = path if path.startswith(("http://", "https://")) else f"https://{HOST}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={
            "Accept": "application/json",
            "X-API-TOKEN": API_KEY,
            "Content-Type": "application/json",
        },
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            payload = e.read().decode()[:400]
            if e.code in (429, 500, 502, 503, 504) and attempt < attempts:
                time.sleep(2**attempt + random.random())
                continue
            raise RuntimeError(f"{method} {path}: {payload}") from e
        except urllib.error.URLError as e:
            if attempt < attempts:
                time.sleep(2**attempt + random.random())
                continue
            raise RuntimeError(f"{method} {path}: {e}") from e
    raise RuntimeError(f"{method} {path}: exhausted {attempts} attempts")


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
    """Remote questions keyed by ``DataExportTag`` (page-aware).

    Collection endpoints return a ``nextPage`` URL; follow it until empty and
    fail rather than loop if the same URL repeats.
    """
    by_tag = {}
    path = f"/API/v3/survey-definitions/{sv_id}/questions"
    seen = set()
    while path:
        if path in seen:
            raise RuntimeError(f"question pagination loop at {path!r}")
        seen.add(path)
        d = request("GET", path)
        for q in d["result"].get("elements") or []:
            tag = q.get("DataExportTag")
            if tag:
                by_tag[tag] = q
        path = d["result"].get("nextPage")
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


def display_logic_condition(question: dict) -> tuple | None:
    """The question's display-logic condition, ignoring the API's own keys.

    The API adds its own description markup back, so the whole structure
    cannot be compared; the condition itself is what the definition owns.
    """
    logic = question.get("DisplayLogic")
    if not logic:
        return None
    expression = logic.get("0", {}).get("0", {})
    return (
        expression.get("LogicType"),
        expression.get("LeftOperand"),
        expression.get("Operator"),
        expression.get("RightOperand"),
    )


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
    if any(remote.get(f) != payload.get(f) for f in fields):
        return True
    # the display logic is compared by its condition (see above)
    return display_logic_condition(remote) != display_logic_condition(payload)


def manual_question_warning(tag: str, payload: dict) -> None:
    """Print the exact UI steps for a question the API cannot create."""
    text = re.sub(r"<[^>]+>", " ", payload.get("QuestionText", ""))
    text = " ".join(text.split())
    print(
        f"WARNING: manual question {tag!r} is not in the remote survey.",
        flush=True,
    )
    print("  Add it once in Qualtrics:", flush=True)
    print("    - Question type: Rank Order (drag and drop)", flush=True)
    print(f"    - Data export tag: {tag}", flush=True)
    print(f"    - Question text: {text}", flush=True)
    print(
        "    - Choices: Carry Forward -> Selected Choices -> C4-3",
        flush=True,
    )
    print("    - Force response: ON", flush=True)
    print(
        "  The sync will continue without it; rerun after adding it.",
        flush=True,
    )


def sync_questions(
    sv_id: str,
    wanted: dict[str, dict],
    workers: int = 1,
    manual_tags: set[str] | None = None,
) -> dict:
    """Create/update/delete remote questions by DataExportTag.

    Qualtrics has no bulk question endpoint, so every question is one HTTP
    call. ``workers`` threads hide the per-call latency; the sync remains
    idempotent, so an interrupted batch can simply be rerun.

    ``manual_tags`` are UI-only questions (carry-forward choices): they are
    never created, updated or deleted. If one is missing, a warning prints
    the exact settings to add; the rest of the sync continues.
    """
    manual_tags = set(manual_tags or ())
    remote = all_questions(sv_id)
    wanted_ids = {}
    tasks = []
    for tag, payload in wanted.items():
        if tag in manual_tags:
            if tag in remote:
                wanted_ids[tag] = remote[tag]["QuestionID"]
                print(f"manual question {tag} present", flush=True)
            else:
                manual_question_warning(tag, payload)
            continue
        if tag in remote:
            entry = remote[tag]
            if question_differs(entry, slim(payload)):
                tasks.append(("update", tag, entry["QuestionID"], slim(payload)))
            else:
                wanted_ids[tag] = entry["QuestionID"]
        else:
            tasks.append(("create", tag, None, slim(payload)))
    stale = [
        q["QuestionID"]
        for tag, q in remote.items()
        if tag not in wanted and tag not in manual_tags
    ]

    created = updated = deleted = 0

    def apply(task):
        action, tag, qid, payload = task
        if action == "update":
            request(
                "PUT",
                f"/API/v3/survey-definitions/{sv_id}/questions/{qid}",
                payload,
            )
            return action, tag, qid
        response = request(
            "POST", f"/API/v3/survey-definitions/{sv_id}/questions", payload
        )
        return action, tag, response["result"]["QuestionID"]

    def delete(qid):
        request(
            "DELETE",
            f"/API/v3/survey-definitions/{sv_id}/questions/{qid}",
        )

    if workers <= 1:
        for task in tasks:
            action, tag, qid = apply(task)
            wanted_ids[tag] = qid
            if action == "update":
                updated += 1
            else:
                created += 1
        for qid in stale:
            delete(qid)
            deleted += 1
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(apply, task) for task in tasks]
            for done, future in enumerate(as_completed(futures), start=1):
                action, tag, qid = future.result()
                wanted_ids[tag] = qid
                if action == "update":
                    updated += 1
                else:
                    created += 1
                if done % 250 == 0 or done == len(tasks):
                    print(
                        f"  questions {done}/{len(tasks)} "
                        f"(+{created} updated {updated})",
                        flush=True,
                    )
            delete_futures = [pool.submit(delete, qid) for qid in stale]
            for done, future in enumerate(as_completed(delete_futures), start=1):
                future.result()
                deleted += 1
                if done % 250 == 0 or done == len(stale):
                    print(
                        f"  question deletes {done}/{len(stale)}",
                        flush=True,
                    )
    print(f"questions: +{created} updated {updated} deleted {deleted}")
    return wanted_ids


# The census quotas the survey used to carry. Dynata fills the sample and
# enforces the agreed quotas, so the survey has none of its own; the sync
# removes the ones it created earlier.
RETIRED_QUOTA_NAMES = {"Age quota", "Gender quota", "Region quota"}


def remove_quotas(sv_id: str) -> None:
    """Delete the survey's retired census quotas (Dynata owns the quotas)."""
    remote = request("GET", f"/API/v3/survey-definitions/{sv_id}/quotas")["result"][
        "elements"
    ]
    removed = 0
    for quota in remote:
        if quota.get("Name") in RETIRED_QUOTA_NAMES:
            request(
                "DELETE",
                f"/API/v3/survey-definitions/{sv_id}/quotas/{quota['ID']}",
            )
            removed += 1
    if removed:
        print(f"quotas: removed {removed} (Dynata manages quotas)")


def block_elements_equal(current: list[dict], wanted: list[str]) -> bool:
    """Compare remote BlockElements against desired QuestionID order."""
    if len(current) != len(wanted):
        return False
    return all(
        cur.get("QuestionID") == qid and cur.get("Type") == "Question"
        for cur, qid in zip(current, wanted)
    )


def sync_blocks(
    sv_id: str, wanted_blocks: list[dict], workers: int = 1
) -> dict[str, str]:
    """Create missing blocks, PUT exact BlockElements per block.

    The API cannot move questions directly into a block, so element lists
    are written through the block payload (``PUT .../blocks/{id}``).
    ``workers`` threads hide the per-call latency; the operations are
    independent per block.
    """
    name_to_id = remote_blocks(sv_id)
    original = dict(name_to_id)
    created = updated = 0

    def create(b):
        desc = b["description"]
        response = request(
            "POST",
            f"/API/v3/survey-definitions/{sv_id}/blocks",
            {"Description": desc, "Type": "Standard", "BlockElements": []},
        )
        return desc, response["result"]["BlockID"]

    def sync_one(b):
        desc = b["description"]
        block_id = name_to_id[desc]
        current = (
            request(
                "GET",
                f"/API/v3/survey-definitions/{sv_id}/blocks/{block_id}",
            )["result"].get("BlockElements")
            or []
        )
        wanted_qids = b["qids"]
        if block_elements_equal(current, wanted_qids):
            return False
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
        return True

    missing = [b for b in wanted_blocks if b["description"] not in name_to_id]
    if workers <= 1:
        for b in missing:
            desc, block_id = create(b)
            name_to_id[desc] = block_id
            created += 1
        for b in wanted_blocks:
            if sync_one(b):
                updated += 1
        wanted_descs = {b["description"] for b in wanted_blocks}
        for desc, bid in original.items():
            if desc not in wanted_descs and not desc.startswith("Trash"):
                request(
                    "DELETE",
                    f"/API/v3/survey-definitions/{sv_id}/blocks/{bid}",
                )
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        with ThreadPoolExecutor(max_workers=workers) as pool:
            if missing:
                futures = [pool.submit(create, b) for b in missing]
                for done, future in enumerate(as_completed(futures), start=1):
                    desc, block_id = future.result()
                    name_to_id[desc] = block_id
                    created += 1
                    if done % 50 == 0 or done == len(missing):
                        print(
                            f"  blocks created {done}/{len(missing)}",
                            flush=True,
                        )
            futures = [pool.submit(sync_one, b) for b in wanted_blocks]
            for done, future in enumerate(as_completed(futures), start=1):
                updated += bool(future.result())
                if done % 50 == 0 or done == len(wanted_blocks):
                    print(
                        f"  blocks synced {done}/{len(wanted_blocks)}",
                        flush=True,
                    )
            # stale blocks the definition no longer contains (Trash stays)
            wanted_descs = {b["description"] for b in wanted_blocks}
            stale = [
                bid
                for desc, bid in original.items()
                if desc not in wanted_descs and not desc.startswith("Trash")
            ]
            futures = [
                pool.submit(
                    request,
                    "DELETE",
                    f"/API/v3/survey-definitions/{sv_id}/blocks/{bid}",
                )
                for bid in stale
            ]
            for done, future in enumerate(as_completed(futures), start=1):
                future.result()
                if done % 50 == 0 or done == len(stale):
                    print(
                        f"  block deletes {done}/{len(stale)}",
                        flush=True,
                    )
    print(f"blocks: created {created} updated {updated}")

    return name_to_id


# ---------------------------------------------------------------------------
# Bootstrap clear + sync entry point
# ---------------------------------------------------------------------------


def clear(sv_id: str, workers: int = 1) -> None:
    """Remove every question and block the copy brought over.

    Block deletes are best-effort: some blocks (e.g. the survey's default
    or protected ones) are rejected with 400 and the sync will manage
    their contents anyway. Question deletes run in ``workers`` parallel
    threads when ``workers`` > 1 (the survey-definitions API has no bulk
    delete, so the calls are per-question; parallelism only hides the
    per-call server latency). Idempotent: already-gone ids are skipped, so
    re-running finishes an interrupted clear.
    """
    questions = list(all_questions(sv_id).values())
    qids = [q["QuestionID"] for q in questions]
    print(f"clearing {len(qids)} questions from {sv_id} ({workers} workers)")
    if workers <= 1:
        for i, qid in enumerate(qids, 1):
            try:
                request(
                    "DELETE",
                    f"/API/v3/survey-definitions/{sv_id}/questions/{qid}",
                )
            except RuntimeError:
                continue  # already gone or protected
            if i % 250 == 0:
                print(f"  deleted {i}/{len(qids)}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def delete(qid: str):
            try:
                request(
                    "DELETE",
                    f"/API/v3/survey-definitions/{sv_id}/questions/{qid}",
                )
                return True
            except RuntimeError:
                return False

        deleted = done = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(delete, qid) for qid in qids]
            for fut in as_completed(futures):
                done += 1
                deleted += bool(fut.result())
                if done % 250 == 0:
                    print(f"  {done}/{len(qids)} handled", flush=True)
        print(f"  pass: {deleted}/{len(qids)} deleted", flush=True)
        remaining = list(all_questions(sv_id).values())
        if remaining and deleted:
            print(f"retry pass for {len(remaining)} leftovers", flush=True)
            return clear(sv_id, workers)  # finish failed/skipped deletes
    block_ids = list(remote_blocks(sv_id).values())
    if workers <= 1:
        for bid in block_ids:
            try:
                request(
                    "DELETE",
                    f"/API/v3/survey-definitions/{sv_id}/blocks/{bid}",
                )
            except RuntimeError as e:
                print(f"  (skip clear/delete block {bid}: {e})")
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def delete_block(bid: str):
            try:
                request(
                    "DELETE",
                    f"/API/v3/survey-definitions/{sv_id}/blocks/{bid}",
                )
                return bid, None
            except RuntimeError as exc:
                return bid, exc

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(delete_block, bid) for bid in block_ids]
            for done, future in enumerate(as_completed(futures), start=1):
                bid, error = future.result()
                if error is not None:
                    print(f"  (skip clear/delete block {bid}: {error})")
                if done % 50 == 0 or done == len(block_ids):
                    print(
                        f"  blocks {done}/{len(block_ids)} handled",
                        flush=True,
                    )


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


def check_meta_description(sv_id: str) -> None:
    """Warn when the live survey description differs from the definition.

    The description is a survey option (``SurveyMetaDescription``), and the
    options payload also holds the skin and other fielding settings, so the
    sync does not rewrite it; a drift is reported for a manual fix in
    Qualtrics (Survey options -> Description).
    """
    options = request("GET", f"/API/v3/survey-definitions/{sv_id}/options")["result"]
    live = (options.get("SurveyMetaDescription") or "").strip()
    if live != defn.SURVEY_META_DESCRIPTION:
        print(
            f"WARNING: survey description is {live!r}, expected "
            f"{defn.SURVEY_META_DESCRIPTION!r} - update it in Qualtrics "
            "(Survey options -> Description)"
        )


def attach_block_qids(
    wanted_blocks: list[dict],
    qids_by_tag: dict[str, str],
    manual_tags: set[str],
) -> set[str]:
    """Attach the remote QuestionIDs to each block.

    A missing manual question is skipped (its warning has already printed);
    a missing normal question is a hard error because ``sync_questions``
    should have created it.
    """
    missing_manual = set()
    for block in wanted_blocks:
        qids = []
        for question in block["questions"]:
            tag = question["DataExportTag"]
            qid = qids_by_tag.get(tag)
            if qid is None:
                if tag in manual_tags:
                    missing_manual.add(tag)
                    continue
                raise RuntimeError(
                    f"question {tag!r} was not created; cannot build its block"
                )
            qids.append(qid)
        block["qids"] = qids
    return missing_manual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("survey_id", nargs="?", default=None)
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument(
        "--clear",
        action="store_true",
        help="delete all remote questions/blocks before syncing (needed "
        "when the remote survey is too full: creates fail above the "
        "account's survey-size ceiling)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="parallel threads for the clear, question and block sync "
        "(Qualtrics has no bulk endpoint; N threads hide the per-call "
        "server latency). Start with 8; the calls are idempotent.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Which model's survey subset to build (survey/<model>/), e.g. "
        "modal_response-overall. Required when several per-model subsets exist.",
    )
    args = parser.parse_args()
    manual_tags = set(getattr(defn, "MANUAL_QUESTION_TAGS", set()))

    if args.model:
        os.environ[defn.COMPARISON_MODEL_ENV] = args.model
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
    check_meta_description(sv_id)
    if args.clear:
        if manual_tags:
            print(
                "WARNING: --clear deletes every remote question, including "
                f"the manual {', '.join(sorted(manual_tags))}; re-add it "
                "after the sync.",
                flush=True,
            )
        print(f"clearing {sv_id} before sync")
        clear(sv_id, max(args.workers, 1))
        print(f"cleared {sv_id}")
    qids_by_tag = sync_questions(sv_id, wanted_questions, args.workers, manual_tags)
    missing_manual = attach_block_qids(wanted_blocks, qids_by_tag, manual_tags)
    if missing_manual:
        print(
            f"note: {', '.join(sorted(missing_manual))} is not in the "
            "remote survey, so its block will omit it until it is added.",
            flush=True,
        )
    name_to_id = sync_blocks(sv_id, wanted_blocks, args.workers)
    flow = defn.flow(name_to_id, qids_by_tag)
    request("PUT", f"/API/v3/survey-definitions/{sv_id}/flow", flow)
    remove_quotas(sv_id)

    profiles = {b["profile"] for b in definition["comparison_blocks"]}
    print(
        f"flow: {flow['Properties']['Count']} flow elements "
        f"(comparison: {min(definition['profiles_shown'], len(profiles))} of "
        f"{len(profiles)} profiles, one dilemma block each)"
    )
    print(f"done - managed survey {sv_id} ({name_to_id['Section C - Your thoughts']})")


if __name__ == "__main__":
    main()
