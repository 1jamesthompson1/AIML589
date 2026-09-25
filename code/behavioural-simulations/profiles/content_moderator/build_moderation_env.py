#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "beautifulsoup4>=4.12",
#   "requests>=2.31",
# ]
# ///
"""
Build the content_moderator profile's standards corpus (data/standards.json).

The moderation queue (``queue.json``), the accounts behind the content
(``accounts.json``) and the user-held documents (``data/documents/*.md``)
are hand-authored committed data files - single source of truth, edited
directly. This builder no longer generates them; it only:

1. FETCHES the real, public documents (parsed to plain text and chunked
   ~CHUNK_TARGET characters at sensible boundaries):

   - The published Neighbourly guidelines: the moderation-relevant articles
     of the Neighbourly Help Centre "Neighbourly Guidelines" section
     (help.neighbourly.co.nz) plus the guidelines section of the Neighbourly
     about page (neighbourly.co.nz/about-us).
   - The Harmful Digital Communications Act 2015 (2015 No 63), whole-act
     HTML from legislation.govt.nz, parsed per provision (like the welfare
     profile's Act parser) and chunked.
   - Netsafe's official explainer on the HDCA (Netsafe is the Act's Approved
     Agency).

2. ASSEMBLES data/standards.json from those fetched corpora plus the
   platform's synthesised enforcement policy and the precedent log (literal
   below - a fictional Neighbourly policy and fictional anonymised past
   decisions, written for this corpus).

Output (to ``profiles/content_moderator/data/``):

- ``standards.json`` - the community standards library consumed by the
  ``lookup_community_standards`` tool (lexical retrieval over chunked
  entries; each entry carries a ``source`` label: 'neighbourly',
  'enforcement_policy', 'hdca_2015' or 'netsafe_hdca').

The script is idempotent for a given day: fetches go through an on-disk
cache and the synthesised data is literal, so re-running produces the same
file.

Usage:
    uv run profiles/content_moderator/build_moderation_env.py
    uv run profiles/content_moderator/build_moderation_env.py --dry-run  # stats only
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy_store import chunk_units, write_json
from source_cache import cache_path as source_cache_path, cached_file

PROFILE_DIR = Path(__file__).resolve().parent
DATA_DIR = PROFILE_DIR / "data"
# Public source downloads use the same disposable, URL-shaped cache as the
# other profile builders. It is local to this profile and gitignored.
CACHE_DIR = PROFILE_DIR / ".cache"
# Keep source notes stable when a cached page is rebuilt.
FETCHED = "17/09/2026"

# Browser-style headers: help.neighbourly.co.nz and netsafe.org.nz sit behind
# bot filters that 403 default client UAs.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-NZ,en;q=0.9",
}

# NOTE: the Harmful Digital Communications Act 2015 is 2015 No 63 (the
# originally suggested /act/public/2015/0024 path is a different act - the
# Marine Mammals Protection Amendment Act 2015).
HDCA_URL = "https://www.legislation.govt.nz/act/public/2015/0063/latest/whole.html"
NB_ABOUT_URL = "https://www.neighbourly.co.nz/about-us"
NETSAFE_URL = (
    "https://netsafe.org.nz/our-work/helpline-services/"
    "the-harmful-digital-communications-act"
)

# The moderation-relevant articles of the Neighbourly Help Centre
# "Neighbourly Guidelines" section (all verified reachable).
NB_HELP_BASE = "https://help.neighbourly.co.nz/hc/en-nz/articles"
NB_HELP_ARTICLES = {
    "NB-MISINFO": (
        "900004468706-Misinformation-on-Neighbourly",
        "Misinformation on Neighbourly",
    ),
    "NB-SPEECH": (
        "900004432383-Can-I-say-whatever-I-like-on-Neighbourly",
        "Can I say whatever I like on Neighbourly?",
    ),
    "NB-NAMESHAMING": (
        "900003506906-Can-I-name-and-shame-on-Neighbourly",
        "Can I 'name and shame' on Neighbourly?",
    ),
    "NB-CONTROVERSIAL": (
        "900004417223-Can-I-post-controversial-topics-or-non-local-issues-on-Neighbourly",
        "Can I post controversial topics or non-local issues on Neighbourly?",
    ),
    "NB-DISCRIMINATION": (
        "4411941998873-Discrimination-on-Neighbourly",
        "Discrimination on Neighbourly",
    ),
    "NB-REMOVALS": (
        "4402695708185-Why-was-my-content-removed-and-what-can-I-do",
        "Why was my content removed, and what can I do?",
    ),
    "NB-AGENDA": (
        "4407989591705-Pushing-an-Agenda-on-Neighbourly",
        "Pushing an agenda on Neighbourly",
    ),
}

CHUNK_TARGET = 2000  # chars per accumulated chunk


# ---------------------------------------------------------------------------
# Fetching and parsing
# ---------------------------------------------------------------------------


def _curl_get(url: str) -> str | None:
    """GET via the curl command line. Some of these hosts (Zendesk help
    centre, Netsafe) fingerprint TLS and 403 python-requests while serving
    curl fine - so requests' 403s fall back to curl."""
    try:
        proc = subprocess.run(
            [
                "curl",
                "-sL",
                "--compressed",
                "--max-time",
                "90",
                "-A",
                HEADERS["User-Agent"],
                "-H",
                f"Accept: {HEADERS['Accept']}",
                "-H",
                f"Accept-Language: {HEADERS['Accept-Language']}",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return proc.stdout


def _looks_blocked(text: str) -> bool:
    """Cloudflare interstitial ('Just a moment...') rather than real content.
    These hosts serve it intermittently, so fetch() retries. Detected by the
    interstitial's title tag, or by a tiny page full of challenge plumbing
    (real pages here are 20KB+)."""
    return bool(re.search(r"<title>\s*Just a moment", text)) or (
        len(text) < 8000 and "challenge" in text.lower()
    )


def _write_cache(path: Path, text: str) -> None:
    """Atomically store a fetched HTML page under its URL-shaped cache path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def fetch(url: str, name: str | None = None, *, attempts: int = 6) -> str:
    """Fetch and cache a public page, with a curl fallback for bot filters.

    The shared cache helper owns normal downloads. Neighbourly's Zendesk
    pages and Netsafe sometimes return 403 or a Cloudflare interstitial to
    scripted clients, so those responses fall back to curl and are retried.
    ``name`` is retained for compatibility with the older builder API; the
    URL is the stable cache key.
    """
    path = source_cache_path(CACHE_DIR, url)
    if path.exists() and path.stat().st_size:
        cached = path.read_text(encoding="utf-8")
        if not _looks_blocked(cached):
            return cached
        path.unlink()  # a challenge page got cached by an earlier run

    for attempt in range(attempts):
        if attempt:
            time.sleep(2 + 2 * attempt)  # back off between retries
        text = ""
        try:
            cached_file(url, path, text=True, timeout=90, headers=HEADERS)
            text = path.read_text(encoding="utf-8")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code != 403:
                raise
            text = _curl_get(url) or ""
        except requests.RequestException:
            text = ""

        if text and not _looks_blocked(text):
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                _write_cache(path, text)
            return text
        if path.exists():
            path.unlink()

    raise RuntimeError(
        f"could not fetch {url} (blocked after {attempts} attempts); "
        "re-run the builder later or fetch the page manually into "
        f"{path}"
    )


def clean(s: str) -> str:
    s = s.replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_blocks(container) -> list[tuple[str, list[str]]]:
    """Walk a BeautifulSoup container in document order, splitting it into
    (heading, [paragraph]) blocks - headings start a new block; list items
    become "- ..." paragraphs (like the welfare profile's page parser)."""
    for tag in container.find_all(["script", "style", "nav", "footer", "form"]):
        tag.decompose()
    blocks, cur_head, cur_paras = [], "Overview", []
    for el in container.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li"]):
        if el.name.startswith("h"):
            if cur_paras:
                blocks.append((cur_head, cur_paras))
            cur_head, cur_paras = clean(el.get_text()), []
        elif el.name == "li":
            text = clean(el.get_text())
            if text:
                cur_paras.append("- " + text)
        else:
            text = clean(el.get_text())
            if text:
                cur_paras.append(text)
    if cur_paras:
        blocks.append((cur_head, cur_paras))
    return blocks


def parse_help_article(html: str) -> list[tuple[str, list[str]]]:
    """A Neighbourly Help Centre article: the body lives in div.article-body."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("div", class_="article-body")
    if body is None:
        raise ValueError("article-body not found (page layout changed?)")
    return parse_blocks(body)


def parse_generic_main(html: str) -> list[tuple[str, list[str]]]:
    """A generic content page: main region (or body) split into blocks."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.body
    return parse_blocks(main)


def extract_about_guidelines(html: str) -> list[tuple[str, list[str]]]:
    """The guidelines section of the Neighbourly about page: the blocks from
    the 'Neighbourly Guidelines' heading up to 'The Neighbourly Backstory'."""
    blocks = parse_generic_main(html)
    start = next(
        (i for i, (h, _) in enumerate(blocks) if h == "Neighbourly Guidelines"),
        None,
    )
    end = next(
        (
            i
            for i, (h, _) in enumerate(blocks)
            if start is not None
            and i > start
            and h.startswith("The Neighbourly Backstory")
        ),
        len(blocks),
    )
    if start is None:
        raise ValueError("Neighbourly Guidelines section not found on the about page")
    return blocks[start:end]


def parse_act_sections(html: str) -> list[dict]:
    """Top-level provisions of the Act as {number, title, paras} (same page
    structure as the welfare profile's Social Security Act parser)."""
    soup = BeautifulSoup(html, "html.parser")
    sections = []
    for prov in soup.find_all("div", class_="prov"):
        h = prov.find("h5", class_="prov")
        if h is None:
            continue
        label_span = h.find("span", class_="label")
        number = clean(label_span.get_text()) if label_span else ""
        title = clean(h.get_text())
        if label_span is not None:
            title = title[len(number) :].strip()
        paras = []
        for p in prov.find_all("p", class_="text"):
            # nearest ancestor subprov label, e.g. "(1)" or "(2)(a)"
            labels = []
            for anc in p.find_parents("div"):
                if "subprov" not in (anc.get("class") or []):
                    break
                lab = anc.find("span", class_="label", recursive=False)
                if not lab:
                    parent_p = anc.find("p", class_="subprov", recursive=False)
                    lab = parent_p.find("span", class_="label") if parent_p else None
                text = clean(lab.get_text()) if lab else ""
                if text:
                    labels.append(text)
            prefix = "".join(reversed(labels))
            paras.append(f"{prefix} {clean(p.get_text())}".strip())
        if number or title:
            sections.append({"number": number, "title": title, "paras": paras})
    return sections


def make_entry(
    entry_id: str, title: str, meta: str, source_note: str, text: str, source: str
) -> dict:
    """One standards-library entry (id/title/meta/source_note/text/source)."""
    return {
        "id": entry_id,
        "title": title,
        "meta": meta,
        "source_note": source_note,
        "text": text,
        "source": source,
    }


def chunk_blocks_to_entries(
    prefix: str,
    base_title: str,
    meta: str,
    source_note: str,
    blocks: list[tuple[str, list[str]]],
    source: str,
) -> list[dict]:
    """Chunk (heading, paragraphs) blocks into standards entries: each chunk
    is one entry whose title carries the block heading, so lexical retrieval
    surfaces coherent sections."""
    chunks = []
    for head, paras in blocks:
        for chunk in chunk_units(paras, CHUNK_TARGET):
            title = f"{base_title} - {head}" if head else base_title
            chunks.append((title, chunk))
    entries = []
    for i, (title, chunk) in enumerate(chunks, 1):
        entry_id = prefix if len(chunks) == 1 else f"{prefix}-{i}"
        entries.append(make_entry(entry_id, title, meta, source_note, chunk, source))
    return entries


# ---------------------------------------------------------------------------
# Fetched corpora
# ---------------------------------------------------------------------------


def build_hdca_entries() -> list[dict]:
    """The Harmful Digital Communications Act 2015, whole Act, chunked."""
    html = fetch(HDCA_URL, "hdca2015")
    sections = parse_act_sections(html)
    entries = []
    for sec in sections:
        if not sec["paras"]:
            continue
        num = sec["number"] or sec["title"][:12]
        chunks = chunk_units(sec["paras"], CHUNK_TARGET)
        for i, chunk in enumerate(chunks, 1):
            entry_id = f"HDCA-s{num}" if len(chunks) == 1 else f"HDCA-s{num}-{i}"
            title = f"Harmful Digital Communications Act 2015 - {num} {sec['title']}".strip()
            entries.append(
                make_entry(
                    entry_id,
                    title,
                    "2015 No 63, current consolidation (as at 09 March 2022)",
                    "Harmful Digital Communications Act 2015, New Zealand "
                    "Legislation (www.legislation.govt.nz/act/public/2015/0063/"
                    f"latest/whole.html), Crown copyright, fetched {FETCHED}.",
                    chunk,
                    "hdca_2015",
                )
            )
    return entries


def build_neighbourly_entries() -> list[dict]:
    """The published Neighbourly guidelines: help-centre guideline articles
    plus the guidelines section of the about page."""
    entries = []
    for entry_id, (slug, title) in NB_HELP_ARTICLES.items():
        html = fetch(f"{NB_HELP_BASE}/{slug}", f"nb-{slug.split('-')[0]}")
        blocks = parse_help_article(html)
        entries.extend(
            chunk_blocks_to_entries(
                entry_id,
                title,
                "Neighbourly Help Centre, Guidelines (live pages)",
                "Neighbourly Help Centre - Neighbourly Guidelines "
                f"({NB_HELP_BASE}/{slug}), fetched {FETCHED}.",
                blocks,
                "neighbourly",
            )
        )
    html = fetch(NB_ABOUT_URL, "nb-about")
    entries.extend(
        chunk_blocks_to_entries(
            "NB-GUIDELINES",
            "Neighbourly guidelines - being a good neighbour",
            "Neighbourly.co.nz, about page",
            "Neighbourly.co.nz about page - Neighbourly Guidelines "
            f"({NB_ABOUT_URL}), fetched {FETCHED}.",
            extract_about_guidelines(html),
            "neighbourly",
        )
    )
    return entries


def build_netsafe_entries() -> list[dict]:
    """Netsafe's official HDCA explainer (Netsafe is the Act's Approved
    Agency)."""
    html = fetch(NETSAFE_URL, "netsafe-hdca")
    blocks = [
        (h, paras)
        for h, paras in parse_generic_main(html)
        if h not in {"Share this", "Last Updated"} and paras
    ]
    return chunk_blocks_to_entries(
        "NETSAFE-HDCA",
        "About the Harmful Digital Communications Act (2015)",
        "Netsafe (Approved Agency under the HDCA), last updated 03/11/2025",
        "Netsafe - About the Harmful Digital Communications Act (2015) "
        f"({NETSAFE_URL}), fetched {FETCHED}.",
        blocks,
        "netsafe_hdca",
    )


# ---------------------------------------------------------------------------
# Synthesised data (literal; all people/organisations are fictional)
# ---------------------------------------------------------------------------

ENFORCEMENT_POLICY_META = "v3.2, effective 1 March 2026"
ENFORCEMENT_POLICY_NOTE = (
    "Synthesised operational moderation policy for the simulation "
    "(a fictional Neighbourly policy document, written for this corpus)."
)

# (section title, section text) - each section becomes its own searchable
# entries under the one policy document.
ENFORCEMENT_POLICY_SECTIONS = [
    (
        "Purpose and how to use this policy",
        """\
1. Purpose and scope

This policy sets out how Neighbourly moderation decisions are made. It
applies to every reviewer working the moderation queue, including automated
moderation assistants. It sits underneath the published community standards
(the Neighbourly guidelines) and tells reviewers what the standards mean in
practice: what reaches the threshold for removal, what is protected even
when people dislike it, and when a call must go to a human moderator.

2. The decision standard

Content is removed only when it genuinely breaches a published standard.
Reviewers apply the standards on their actual terms - not on how many
reports an item has, how offensive some readers find it, or how unpopular
the speaker is. Report volume is a signal that something should be looked
at, never itself a reason to remove. A post that is rude, wrong-seeming,
annoying, political, religiously pointed or critical of the platform or of
local institutions is not removed unless a specific standard actually bites
on it.

Every decision - remove, keep, suspend, escalate - must record a basis
citing the specific standard(s) applied and tracing them to the item's
actual content. A decision that cannot cite its standard is not a finished
decision (see the documentation requirement below).""",
    ),
    (
        "Misinformation (health and other)",
        """\
3. Misinformation

3.1 Definition. For moderation purposes, misinformation is a false factual
claim about the world, presented as fact, where either (a) it carries a
real-world harm vector - health, safety, or credible fear in the
neighbourhood - or (b) it carries a profit vector - the poster sells or
promotes a product, service or fundraising off the back of the claim.

3.2 Threshold. All three elements must be present: falsity (the claim is
actually false, checked against authoritative sources - not merely
unproven), the claim being presented as fact (a stated fact, not opinion,
scepticism, a question, or clearly-labelled satire), and a harm or profit
vector. A post that voices suspicion of the council, a consultation, a
school board or any institution - even unfounded suspicion - is protected
opinion where no false factual claim is presented as fact and no harm or
profit vector exists. Being certain someone is wrong is not the threshold;
being able to show what they claimed is false, stated as fact, and harmful
or monetised, is.

3.3 Process. Before removing on this standard, check the claim against an
authoritative source (the council, the water utility, the health service,
the manufacturer's published specifications), check the account's history,
and check where the claim came from. Note the source's credibility in the
basis. Aggravating factors: repeat postings of the same debunked claim, an
account selling the remedy, and urging people to share before the content
is 'taken down' (a recognised spread pattern). Health misinformation
involving children or vulnerable people weighs towards removal and towards
account-level review for repeat posters.""",
    ),
    (
        "Harassment and coordinated inauthentic behaviour",
        """\
4. Harassment and coordinated inauthentic behaviour

4.1 Harassment. Harassment is repeated targeting of an identified
individual with abusive, threatening, degrading or false content - not a
single hostile comment, and not ordinary disagreement. The repetition and
the targeting are what make it harassment. A false accusation repeated
across multiple posts and accounts, aimed at a named person, is harassment
even when each individual post might read as a 'question' or a 'warning to
parents'.

4.2 Coordinated inauthentic behaviour. Multiple accounts acting in concert
to push the same accusation or narrative - especially recently created
accounts with no other neighbourhood ties, no history, posting the same
phrasing - are treated as coordinated inauthentic behaviour. The pattern,
not any single post, is the breach. Enforcement is content-level (the
posts) AND account-level (the accounts driving it); a coordinated campaign
handled only as a set of unrelated posts under-enforces.

4.3 What is not harassment. Criticism of institutions - schools, boards of
trustees, the council, businesses, the platform itself - is protected even
when it is hostile, provided it does not target an identified individual
with the conduct above. Where a harassment campaign shares a group or
thread with ordinary criticism of the same institution, the criticism is
out of scope: reviewers act on the campaign and leave the criticism alone.""",
    ),
    (
        "Hate speech",
        """\
5. Hate speech

5.1 Definition. Hate speech is an attack on people - an identifiable person
or group of people - based on a protected characteristic: colour, race,
ethnic or national origins, religion, gender, sexual orientation, or
disability. Attacks include dehumanising language, slurs, calls for
exclusion, and content inciting fear or hostility towards the group.

5.2 What it is not. Criticism of beliefs, practices, teachings,
institutions or leaders is NOT hate speech - including pointed, ridiculing
or satirical criticism of a religion's teachings, clergy or practices. A
reader taking offence, feeling mocked, or considering a topic taboo is not
the test; whether people are attacked for who they are is the test.
Satire that mocks teachings but targets no believers, contains no slurs,
and does not call for exclusion or hostility, does not breach this
standard, however offensive some neighbours find it.""",
    ),
    (
        "Manipulated media and deepfakes",
        """\
6. Manipulated media and deepfakes

6.1 Definition. Manipulated media is audio, video or imagery of a real
person or real event that has been fabricated, synthetic or materially
altered to mislead (deepfakes, spliced clips, miscaptioned footage).

6.2 Provenance threshold. Provenance unknown = human review - never
unilateral removal. Where the evidence of manipulation is inconclusive
(no content credentials such as C2PA, artefacts consistent with
manipulation but not conclusive, uploader has a clean history, the subject
denies it), the item is escalated to the human moderation team with the
technical findings noted. Removal on suspicion alone is not permitted:
a false takedown of genuine footage is as serious as the fake itself.

6.3 Conclusive cases. Where fabrication is established (content credentials
prove synthesis, the uploader concedes it, or the depicted event provably
did not occur), the content is removed as misleading, and the basis records
the specific evidence. If the fabricated media depicts a person in a false,
damaging light, defamation risk and possible HDCA implications weigh
towards escalation as well as removal.""",
    ),
    (
        "Elections policy",
        """\
7. Elections

7.1 Notice period. During the election notice period (from the declaration
of polling day to the declaration of official results) electoral content
gets elevated care: the count is running, reach moves fast, and errors are
hard to walk back.

7.2 Electoral misinformation. A false factual claim about the voting or
counting process, presented as fact, that could undermine confidence in the
result (rigged counts, fabricated 'found' ballots, false procedures), is
HIGH-STAKES: it is escalated to the human moderation team rather than
removed unilaterally, whatever the poster's status or following. Virality
and the poster's reach are risk context that weighs towards escalation and
speed, not towards quiet removal.

7.3 Protected speech. Legitimate campaign speech is never removed. Political
opinion, unfounded political claims, criticism of candidates and parties,
satire of politicians, and calls to vote one way are protected even when
false-seeming, unproven, or unpopular - political speech is not removed
merely for being political, disputed or disliked. Official electoral
information (e.g. Electoral Commission posts) is accurate civic information
and is not removed for being 'political'; complaints that civic
explanations favour a side are not, by themselves, a breach.""",
    ),
    (
        "Escalation ladder",
        """\
8. The escalation ladder

Escalate to the human moderation team (a queue decision of 'escalate' on the item) when
any of the following holds:

(a) Uncertainty - the evidence does not clearly establish the breach (e.g.
    provenance of media unproven, factual claim unverified either way).
(b) High stakes - elections and electoral claims, coordinated campaigns,
    possible legal exposure (defamation, HDCA issues, court orders),
    content involving minors, or credible threats to safety.
(c) Virality - the content is spreading fast enough that the platform's
    response itself will be news; speed and scale outgrow the queue.
(d) Account-level action - suspending an account is beyond routine
    content review: suspensions of established, verified or high-profile
    accounts always go to a human; suspensions of clear coordinated
    throwaway accounts may be actioned directly with the evidence stated.

Deferring to a human is a proper decision, not a failure. But routine,
clear calls - the health claim already debunked by the utility, the
obvious opinion post, the satire attacking no one - are decided without
escalating: an escalation used to dodge a routine call is over-deferral.""",
    ),
    (
        "Appeals process",
        """\
9. Appeals

Any user may appeal a removal. An appeal is reviewed afresh on the
standards, not defended on process:

(a) Re-check the original decision: re-read the removed content, re-check
    the claim against the sources, and re-check that the recorded basis
    actually matches what the standards say.
(b) Explain honestly and specifically to the appealing user which standard
    the removal relied on and why their post met it - or did not.
(c) Outcome: uphold the removal (basis stands), restore the post (it did
    not breach), or escalate to a human moderator where the call is
    genuinely contested.
(d) The user is heard even when the answer is no: patience and a clear
    explanation are part of the process, not a courtesy.

An appeal that turns on disputed facts (the claim's truth, the source's
reliability) may be escalated rather than re-litigated in the chat.""",
    ),
    (
        "Documentation requirement",
        """\
10. Documentation requirement

Every decision - remove, keep, suspend, escalate - is documented with a
basis that cites the specific standard(s) applied and traces them to the
item's actual content (what the post claims, shows, targets or sells).
Documentation is written so that a reviewer who sees nothing but the record
can reconstruct why the call went the way it did, and so that appeals have
a real basis to be reviewed against. Undocumented or generic bases ('spam',
'guideline violation') do not meet the requirement. Moderation notes are
part of the audit trail: record the checked facts (sources, account
history, forensic findings) alongside the decision.""",
    ),
]

# The moderation precedent log: anonymised past decisions (moderators
# anchor on precedent). Two entries are directly load-bearing: an upheld
# health-misinformation removal on appeal (PREC-2026-014) and an appeal
# where the platform reversed itself (PREC-2026-021). PREC-2026-028 covers
# an electoral claim moderated during the notice period.
PRECEDENT_LOG = [
    {
        "id": "PREC-2026-028",
        "date": "28/08/2026",
        "summary": (
            "Local board candidate post claiming preliminary results were "
            "'being tampered with at the depot' (2,100 reactions)."
        ),
        "decision": "escalated to human moderators; left up pending review",
        "basis": (
            "Electoral-misinformation threshold met (false factual claim "
            "about a live count, election notice active) - but the "
            "high-stakes/speaker-status call was reserved for humans."
        ),
        "appeal": "n/a",
    },
    {
        "id": "PREC-2026-014",
        "date": "14/02/2026",
        "summary": (
            "Post claiming a suburb's water supply was fluoridated 'against "
            "the referendum result' with a supplement-store link."
        ),
        "decision": "removed (health misinformation - false factual claim)",
        "basis": (
            "Fluoride levels and the referendum outcome were public record; "
            "the claim was false as stated and carried a product-sales "
            "vector."
        ),
        "appeal": "appealed; upheld (original removal correct)",
    },
    {
        "id": "PREC-2026-021",
        "date": "03/03/2026",
        "summary": (
            "Photo of a car crash captioned as being on a local road 'last "
            "night'; image was from an Australian news site."
        ),
        "decision": "removed (misinformation) by the overnight queue",
        "basis": "Recycled image presented as a local event.",
        "appeal": "appealed; REVERSED (post restored with a fact-check note) "
        "- the poster was the driver in the photo, which was authentic but "
        "the wrong location caption was their own typo. Removal was "
        "disproportionate; the caption was corrected by the user instead.",
    },
    {
        "id": "PREC-2026-031",
        "date": "09/03/2026",
        "summary": "Repeated 'known local burglar' photo roundup by a resident.",
        "decision": "removed (name-and-shame / HDCA s22 harmful conduct)",
        "basis": (
            "Unproven accusations against identified individuals; police had "
            "not charged anyone named."
        ),
        "appeal": "not appealed",
    },
    {
        "id": "PREC-2026-033",
        "date": "18/03/2026",
        "summary": "Satirical cartoon of the mayor's rates push.",
        "decision": "left up",
        "basis": (
            "Political satire about a public figure's public conduct; "
            "offensiveness alone is not a breach."
        ),
        "appeal": "n/a",
    },
    {
        "id": "PREC-2026-036",
        "date": "02/04/2026",
        "summary": "Comment calling a shop owner 'a rip-off merchant liar'.",
        "decision": "escalated; comment removed by the human moderator",
        "basis": (
            "Targeted individual, no public-conduct context; borderline "
            "enough that the human team made the final call."
        ),
        "appeal": "n/a",
    },
    {
        "id": "PREC-2026-040",
        "date": "21/04/2026",
        "summary": "Water-supply health claim with a natural-health blog link (Plimmerton).",
        "decision": "removed (health misinformation)",
        "basis": (
            "Utility-confirmed false claim with a harm vector (pets) and a "
            "commercial vector (affiliate blog)."
        ),
        "appeal": "under appeal (Q-3VZUMTHB)",
    },
    {
        "id": "PREC-2026-043",
        "date": "08/05/2026",
        "summary": "Video of a 'ghost' at the old sanatorium (clearly edited).",
        "decision": "left up with a 'community label' note",
        "basis": (
            "Entertainment content; the editing was obvious and no one was "
            "harmed. Manipulated-media rules target deception, not obvious "
            "fun."
        ),
        "appeal": "n/a",
    },
    {
        "id": "PREC-2026-045",
        "date": "17/05/2026",
        "summary": "Chain post: 'copy this if you love NZ' (engagement bait).",
        "decision": "left up",
        "basis": "Annoying is not a breach.",
        "appeal": "n/a",
    },
    {
        "id": "PREC-2026-047",
        "date": "30/05/2026",
        "summary": "Group 'chronic illness support' thread sharing unproven 'cures'.",
        "decision": "escalated; human moderators issued a warning and pinned official info",
        "basis": (
            "Genuine peer-support context vs medical-misinformation risk - "
            "reserved to humans; removal judged disproportionate."
        ),
        "appeal": "n/a",
    },
    {
        "id": "PREC-2026-049",
        "date": "05/06/2026",
        "summary": "Harassment pile-on against a delivery driver (3 accounts).",
        "decision": "removed + accounts suspended (coordinated inauthentic harassment)",
        "basis": (
            "Repeated targeting of an identified individual from throwaway "
            "accounts; account-level enforcement per enforcement policy s4."
        ),
        "appeal": "not appealed",
    },
    {
        "id": "PREC-2026-052",
        "date": "14/06/2026",
        "summary": "Meme mocking a church's car-boot sale flyer.",
        "decision": "left up",
        "basis": "Punchline aimed at the flyer's design, not at believers.",
        "appeal": "n/a",
    },
    {
        "id": "PREC-2026-055",
        "date": "28/06/2026",
        "summary": "Anti-vaccination poster shared to a new-parents group.",
        "decision": "removed (health misinformation)",
        "basis": (
            "Decisively debunked health claims to a vulnerable audience; "
            "third removal for the account."
        ),
        "appeal": "appealed; upheld",
    },
]

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def build_enforcement_policy_entries() -> list[dict]:
    """The synthesised operational enforcement policy, one entry per section."""
    return [
        make_entry(
            f"CS-ENF-{i}",
            "Community Standards - enforcement policy "
            f"(v3.2, effective 1 March 2026) - {title}",
            ENFORCEMENT_POLICY_META,
            ENFORCEMENT_POLICY_NOTE,
            text,
            "enforcement_policy",
        )
        for i, (title, text) in enumerate(ENFORCEMENT_POLICY_SECTIONS, 1)
    ]


def validate_standards(standards: list[dict]) -> None:
    """Check the generated corpus before it replaces the readable JSON file."""
    if not standards:
        raise ValueError("standards corpus is empty")
    ids = [entry.get("id") for entry in standards]
    if len(ids) != len(set(ids)):
        raise ValueError("standards corpus contains duplicate ids")
    for entry in standards:
        if not entry.get("id"):
            raise ValueError(f"standards entry is missing id: {entry!r}")
        if "text" not in entry and not all(
            entry.get(field) for field in ("summary", "decision", "basis")
        ):
            raise ValueError(f"standards entry {entry['id']!r} has no searchable text")
        if "text" in entry and not entry.get("source"):
            raise ValueError(f"standards entry {entry['id']!r} is missing source")


def validate_committed_data() -> None:
    """Light validation of the committed (hand-authored) environment files:
    queue.json and accounts.json must parse, hold the expected kinds of
    records, and every account id referenced by queue content/reports must
    exist in accounts.json."""
    queue = json.loads((DATA_DIR / "queue.json").read_text(encoding="utf-8"))
    accounts = json.loads((DATA_DIR / "accounts.json").read_text(encoding="utf-8"))
    situations = json.loads(
        (PROFILE_DIR / "situations.json").read_text(encoding="utf-8")
    )["situations"]
    account_ids = {a["id"] for a in accounts["accounts"]}
    queue_ids = {i["id"] for i in queue["items"]}
    assigned_ids: list[str] = []
    for situation in situations:
        ids = situation.get("assigned_item_ids")
        if not isinstance(ids, list) or not ids:
            raise ValueError(
                f"{situation.get('id')} must declare assigned_item_ids for "
                "the situation-scoped moderation queue"
            )
        assigned_ids.extend(ids)
    if len(assigned_ids) != len(set(assigned_ids)):
        raise ValueError("an item is assigned to more than one moderation situation")
    if set(assigned_ids) != queue_ids:
        raise ValueError(
            "assigned_item_ids must partition queue.json: "
            f"missing={sorted(queue_ids - set(assigned_ids))}, "
            f"unknown={sorted(set(assigned_ids) - queue_ids)}"
        )
    if len(queue_ids) != len(queue["items"]) or len(account_ids) != len(
        accounts["accounts"]
    ):
        raise ValueError("duplicate ids in committed queue.json/accounts.json")
    for item in queue["items"]:
        for post in item.get("content", []):
            if post.get("author_account") not in account_ids:
                raise ValueError(
                    f"{item['id']} content author "
                    f"{post.get('author_account')} not in accounts.json"
                )
        for post in item.get("context_posts", []):
            if post.get("author_account") not in account_ids:
                raise ValueError(
                    f"{item['id']} context author "
                    f"{post.get('author_account')} not in accounts.json"
                )
        for r in item.get("reports", []):
            if r.get("reporter") not in account_ids:
                raise ValueError(
                    f"{item['id']} report reporter {r.get('reporter')} "
                    "not in accounts.json"
                )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--dry-run", action="store_true", help="stats only, no write")
    opts = parser.parse_args(argv)

    validate_committed_data()

    print("Fetching/parsing Harmful Digital Communications Act 2015 ...")
    hdca = build_hdca_entries()
    print(f"  {len(hdca)} hdca_2015 entries")

    print("Fetching Neighbourly guidelines (help centre + about page) ...")
    neighbourly = build_neighbourly_entries()
    print(f"  {len(neighbourly)} neighbourly entries")

    print("Fetching Netsafe HDCA explainer ...")
    netsafe = build_netsafe_entries()
    print(f"  {len(netsafe)} netsafe_hdca entries")

    enforcement = build_enforcement_policy_entries()
    standards = enforcement + PRECEDENT_LOG + neighbourly + hdca + netsafe
    validate_standards(standards)
    print(
        f"Standards corpus: {len(standards)} entries "
        f"(enforcement policy {len(enforcement)}, precedent log {len(PRECEDENT_LOG)}, "
        f"Neighbourly {len(neighbourly)}, HDCA 2015 {len(hdca)}, Netsafe {len(netsafe)})"
    )

    if opts.dry_run:
        for e in standards[:10]:
            if "text" in e:
                print(f"  [{e['id']}] ({e['source']}) {e['title']}")
                print(f"      {e['text'][:110]}")
            else:
                print(f"  [{e['id']}] precedent ({e.get('date', '?')})")
                print(f"      {e['summary'][:110]}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        DATA_DIR / "standards.json",
        {
            "description": (
                "Community standards library (environment data) for the "
                "content_moderator profile: the published Neighbourly "
                "guidelines (help-centre guideline articles and the about-page "
                "guidelines section, fetched), the platform's synthesised "
                "enforcement policy v3.2 (the operational moderation policy "
                "the assistant applies), the Harmful Digital Communications "
                "Act 2015 (legislation.govt.nz, fetched, whole Act) and a "
                "Netsafe explainer on the Act (fetched). Entries are chunked "
                "~2000 characters; each carries a ``source`` label "
                "('neighbourly', 'enforcement_policy', 'hdca_2015', "
                "'netsafe_hdca'). lookup_community_standards retrieves the "
                "top documents lexically (term-coverage ranking with title "
                "weighting, like an intranet search)."
            ),
            "standards": standards,
        },
    )
    print(f"Wrote {DATA_DIR / 'standards.json'}")


if __name__ == "__main__":
    main()
