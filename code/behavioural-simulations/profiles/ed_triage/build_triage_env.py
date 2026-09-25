#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "beautifulsoup4>=4.12",
#   "requests>=2.31",
# ]
# ///
"""
Build the ed_triage profile's environment data.

Sources (all public; a research simulation corpus):

Fetched at build time (plain HTML, parsed to text, chunked ~2000 chars at
sensible boundaries):

1. ACEM - Triage (the Australasian Triage Scale: category table with target
   times and performance thresholds, plus scale description),
   acem.org.au.
2. KidsHealth NZ (Paediatric Society of NZ & Starship) - Fever in children
   (parent guidance: under-3-months red flag, when to call Healthline /
   be seen urgently / call 111), kidshealth.org.nz.
3. HealthEd (Health New Zealand) - Baby and Child Sickness: Danger Signs
   (HE4240), healthed.govt.nz.
4. HealthEd (Health New Zealand) - Healthcare: where should I go?
   (HE3514: GP / urgent care / ED / 111 / Healthline routing),
   healthed.govt.nz.
5. NZ Sepsis Trust - What is sepsis? (early recognition signs),
   sepsis.org.nz.
6. healthdirect (Australian government) - Chest pain (red flags: severe /
   worsening / >10 minutes, heart attack warning signs, what to do while
   waiting for an ambulance; page last reviewed September 2025),
   healthdirect.gov.au.
7. KidsHealth NZ - Meningococcal disease (rash + fever red flags, early
   symptoms, when to get urgent help; trimmed to the recognition/escalation
   sections), kidshealth.org.nz.
8. Patient.info - Head injuries (peer-reviewed leaflet: severity assessment
   with AVPU, when to call an ambulance - including for people over 65 and
   people on blood-thinning medication - and the symptoms of severe head
   injury; trimmed to the assessment/red-flag sections), patient.info.
9. Royal College of Physicians (UK) - National Early Warning Score (NEWS) 2
   (the six scored physiological parameters and how the aggregate score
   works), rcp.ac.uk.
10. BMJ Postgraduate Medical Journal (PMC) - "Using NEWS2: an essential
    component of reliable clinical assessment" (the NEWS2 trigger
    thresholds: clinical response to total scores 0 / 1-4 / red score 3 /
    5 or more / 7 or more; Table 1), pmc.ncbi.nlm.nih.gov.
11. Royal Children's Hospital Melbourne - Acceptable ranges for
    physiological variables (paediatric normal ranges for systolic BP,
    heart rate and respiratory rate by age; table plus key points),
    rch.org.au.
12. KidsHealth NZ - Bronchiolitis (urgent-help and 111 criteria for babies
    with breathing problems; trimmed to the recognition/escalation
    sections), kidshealth.org.nz.
13. KidsHealth NZ - Dehydration in babies & children (signs by severity and
    the see-a-doctor / urgently / 111 escalation ladder), kidshealth.org.nz.
14. healthdirect (Australian government) - Abdominal pain (when to see a
    doctor and when to seek urgent care; trimmed to the red-flag sections),
    healthdirect.gov.au.
15. healthdirect (Australian government) - Abdominal aortic aneurysm
    (ruptured-AAA warning signs: sudden severe abdominal/back pain, pale
    and sweaty, faint; emergency escalation),
    healthdirect.gov.au.

Captured via the research assistant (healthnz.govt.nz and some local
controlled-document URLs block automated fetchers; the text is committed under
``profiles/ed_triage/data/sources/`` and labelled in each entry's
source_note):

16. Health New Zealand | Te Whatu Ora - Emergency departments (ED) (when to
    go to ED, 111, triage in plain language, non-urgent alternatives,
    safety-netting on discharge; page last updated 5 August 2026).
17. Health New Zealand | Te Whatu Ora - Healthline (the 24/7 nurse advice
    line, GP/urgent-care/ED/ambulance routing, callback, Māori clinician,
    interpreter and NZ Relay access; page captured 25 September 2026).
18. Health New Zealand | Te Whatu Ora - Sprained ankle (causes, symptoms,
    x-ray referral criteria - difficulty weight-bearing, deformity - self
    care and treatment; page last updated 22 July 2026).
19. ACEM P06 V5 and G24 V6 (ATS policy, reassessment, staffing/performance
    thresholds, documentation and audit).
20. CCHV / Te Pae Tiaki ED triage document 1.989 (local staffing, ACNM/SMO
    escalation, reassessment and alternatives; issue 19 February 2024).
21. CCHV / Te Pae Tiaki responsibility document 1.1560 (bed block,
    overcrowding, VIS, Duty Nurse Manager and hospital-management escalation;
    issue 20 February 2025).
22. Health New Zealand acute-care programme guidance (six-hour SSED target,
    access block and whole-system flow).

Fetched pages are pinned by ``data/source_hashes.json``; a changed download
fails closed until the source is reviewed. Runtime lookup applies each
situation's simulated date as an as-of limit.

Output (to ``profiles/ed_triage/data/``, the profile's environment-data
directory):

- ``protocols.json`` - the source-labelled triage and operational-guidance
  corpus consumed by ``lookup_triage_protocol`` and
  ``lookup_operational_guidance``. Entries carry authority class,
  jurisdiction, scope, document/review dates and a human-readable source note.
- ``department_snapshots.json`` - hand-authored, point-in-time staffing,
  demand, capacity, flow, escalation, alternatives and communications
  fixtures. It is validated but never overwritten.
- ``patients.json`` - the hand-authored patient, caller and monitored-bed
  source of truth. It is validated but never overwritten.

Usage:
    uv run profiles/ed_triage/build_triage_env.py            # fetch + build
    uv run profiles/ed_triage/build_triage_env.py --dry-run  # stats only
"""

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy_store import chunk_units, write_json
from source_cache import cache_path as source_cache_path, cached_file

PROFILE_DIR = Path(__file__).resolve().parent
DATA_DIR = PROFILE_DIR / "data"
SOURCE_DIR = DATA_DIR / "sources"
PROTOCOLS_OUT = DATA_DIR / "protocols.json"
PATIENTS_OUT = DATA_DIR / "patients.json"
SNAPSHOTS_OUT = DATA_DIR / "department_snapshots.json"
SOURCE_HASHES_PATH = DATA_DIR / "source_hashes.json"
SOURCE_HASHES = json.loads(SOURCE_HASHES_PATH.read_text(encoding="utf-8"))
# Public source downloads use the same disposable, URL-shaped cache as the
# other profile builders. It is local to this profile and gitignored.
CACHE_DIR = PROFILE_DIR / ".cache"
# Keep source notes stable when a cached page is rebuilt. New/updated fetched
# sources should carry an explicit `retrieved` value; this is only a legacy
# fallback for older specs.
FETCHED = "17/09/2026"

CHUNK_TARGET = 2000  # chars per chunk (entry text; split at line boundaries)


def read_manual_source(fname: str) -> str:
    """Read a committed capture for a source that blocks automated fetching."""
    path = SOURCE_DIR / fname
    if not path.is_file():
        raise SystemExit(
            f"Missing manual source {path}. Restore the committed capture "
            "from git before rebuilding the protocol corpus."
        )
    return path.read_text(encoding="utf-8")


# Kept as a small compatibility alias for older local tooling.
read_env_src = read_manual_source


def clean(s: str) -> str:
    s = s.replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def fetch(url: str, name: str | None = None) -> str:
    """Fetch a public page through the profile-local URL-shaped cache.

    ``name`` is retained for compatibility with the older builder API; the
    URL, rather than a hand-written filename, is the stable cache key.
    """
    path = source_cache_path(CACHE_DIR, url)
    return cached_file(
        url,
        path,
        sha256=SOURCE_HASHES.get(url),
        text=True,
        timeout=120,
    ).read_text(encoding="utf-8")


def _is_translated_duplicate(text: str) -> bool:
    """True for lines/headings from a translated-language duplicate of the
    English page (te reo Maori / Samoan / Tongan text carries a high share
    of macron/okina words; English carries only the occasional loanword,
    e.g. 'Pepi', 'whanau'). Mirrors the welfare builder's exclusion of
    translated-language duplicates of English pages."""
    words = text.split()
    if not words:
        return False
    hits = sum(1 for w in words if re.search(r"[āēīōūĀĒĪŌŪãÃʻʼ]", w))
    return hits / len(words) > 0.15


def _line_filter(line: str) -> bool:
    """False for lines that are page furniture or translated duplicates."""
    if not line:
        return False
    if re.search(r"Share via email|Copy link|Print this page|added to your cart", line):
        return False
    if _is_translated_duplicate(line):
        return False
    return True


def _element_lines(container) -> list[str]:
    """Readable plain-text lines from a container's headings, paragraphs
    and list items, with page furniture and consecutive duplicates
    removed."""
    lines = []
    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        text = clean(el.get_text(" "))
        if el.name == "li":
            text = "- " + text
        if not text:
            continue
        # collapse consecutive duplicates (some CMS pages double each bullet)
        norm = text.lower().lstrip("- ")
        prev = lines[-1].lower().lstrip("- ") if lines else ""
        if norm and norm == prev:
            continue
        if _line_filter(text):
            lines.append(text)
    return lines


def _parse_main(html: str):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "form", "iframe", "noscript"]
    ):
        tag.decompose()
    return soup, soup.find("main") or soup.body or soup


def page_lines(html: str, *, min_para: int = 0) -> list[str]:
    """Parse a page's <main> (or body) into readable plain-text lines."""
    _soup, main = _parse_main(html)
    return [line for line in _element_lines(main) if len(line) >= min_para]


def kidshealth_lines(html: str) -> list[str]:
    """KidsHealth pages host the same content in several languages as
    tabbed panels (``div.media-language__media``, English first): keep the
    English panel and drop the whole translated-tab wrapper."""
    _soup, main = _parse_main(html)
    panels = main.select("div.media-language__media")
    lines = _element_lines(panels[0]) if panels else []
    wrapper = main.find("div", class_="media-language")
    if wrapper is not None:
        wrapper.decompose()
    return lines + _element_lines(main)


def healthed_lines(html: str) -> list[str]:
    """HealthEd resource pages put the English leaflet text and its te reo
    Maori mirror in one description block, the Maori half starting at a
    translated heading: drop that heading and every sibling after it, then
    extract lines."""
    _soup, main = _parse_main(html)
    content = main.find("div", class_="resource-content__description") or main
    dropping = False
    for child in list(content.find_all(recursive=False)):
        if dropping:
            child.decompose()
            continue
        if child.name in ("h2", "h3", "h4") and _is_translated_duplicate(
            clean(child.get_text(" "))
        ):
            child.decompose()
            dropping = True
    return _element_lines(content)


def acem_lines(html: str) -> list[str]:
    """The ACEM triage page: descriptive paragraphs plus the ATS category
    table rendered as lines."""
    soup = BeautifulSoup(html, "html.parser")
    paras = [
        clean(p.get_text(" "))
        for p in soup.find_all("p")
        if len(clean(p.get_text(" "))) > 60
    ]
    lines = list(paras)
    table = soup.find("table")
    if table is not None:
        lines.append("ATS category table:")
        for tr in table.find_all("tr"):
            cells = [clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                lines.append("- " + " | ".join(cells))
    return lines


def acem_policy_lines(html: str) -> list[str]:
    """Read ACEM's policy page as readable text, including the P06 headings."""
    return page_lines(html)


def chunk_text(text: str, target: int = CHUNK_TARGET) -> list[str]:
    """Split plain text into boundary-preserving chunks at line breaks."""
    return chunk_units(text.splitlines(), target, "\n")


def _slice_lines(lines: list[str], max_chars: int) -> list[str]:
    """Keep whole lines up to ~max_chars (a cheap trim of long pages to the
    red-flag/escalation content, which these sources front-load)."""
    out, size = [], 0
    for line in lines:
        out.append(line)
        size += len(line) + 1
        if size >= max_chars:
            break
    return out


def _keep_between(lines: list[str], start: str | None, end: str | None) -> list[str]:
    """Lines from the first line containing ``start`` (or the top if
    ``start`` is None or never matches) up to (excluding) the first later
    line containing ``end`` (or the end of the text if ``end`` is None or
    never matches)."""
    i = (
        0
        if start is None
        else next(
            (idx for idx, line in enumerate(lines) if start.lower() in line.lower()), 0
        )
    )
    j = (
        len(lines)
        if end is None
        else next(
            (
                idx
                for idx, line in enumerate(lines)
                if idx > i and end.lower() in line.lower()
            ),
            len(lines),
        )
    )
    return lines[i:j]


def _keep_ranges(lines: list[str], ranges: list[tuple[str | None, str]]) -> list[str]:
    """Concatenate several (start, end) windows, keeping the matching lines
    (``None`` starts from the top of the text). Used to skip page sections
    that are not triage-relevant (e.g. causes, diagnosis, vaccination)."""
    out = []
    for start, end in ranges:
        out.extend(_keep_between(lines, start, end))
    return out


def rch_lines(html: str) -> list[str]:
    """The RCH guideline pages are ASP.NET pages wrapped in a <form> (so the
    shared parser's form decompose strips everything) with the guideline
    body in ``div`` widgets whose ids contain ``plhContent``: take leaf
    headings/paragraphs/list items plus the ranges table from those
    containers only (skipping in-table cells, which the leaf scan would
    otherwise double up), then cut before the reference list."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "iframe", "noscript"]
    ):
        tag.decompose()
    lines, seen = [], set()

    def _add(text: str, bullet: bool = False) -> None:
        text = ("- " if bullet else "") + text
        if text and text not in seen:
            seen.add(text)
            lines.append(text)

    for container in soup.find_all("div", id=re.compile(r"plhContent")):
        for el in container.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
            if el.find(["p", "li"]):  # nested container, not leaf text
                continue
            if el.find_parent("table"):  # table rows are rendered below
                continue
            _add(clean(el.get_text(" ")), bullet=el.name == "li")
        for table in container.find_all("table"):
            for tr in table.find_all("tr"):
                cells = [clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
                cells = [c for c in cells if c]
                if cells:
                    _add("- " + " | ".join(cells))
    return _keep_between(lines, "Key points", "Reference List")


def pmc_lines(html: str) -> list[str]:
    """The PMC article pages: the abstract paragraphs (long <p>) plus the
    clinical-response table. Only the first table (NEWS2 trigger thresholds)
    is kept - the article's later tables are not triage material."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "form", "iframe", "noscript"]
    ):
        tag.decompose()
    lines = []
    for p in soup.find_all("p"):
        text = clean(p.get_text(" "))
        if len(text) < 150 or "Lock" in text or "Secure .gov" in text:
            continue  # page banner / short fragments
        lines.append(text)
        if len(lines) >= 2:
            break
    table = soup.find("table")
    if table is not None:
        lines.append("Clinical response to NEWS2 trigger thresholds (Table 1):")
        for tr in table.find_all("tr"):
            cells = [clean(td.get_text(" ")) for td in tr.find_all(["td", "th"])]
            cells = [c for c in cells if c]
            if cells:
                lines.append("- " + " | ".join(cells))
    return lines


def patient_info_lines(html: str) -> list[str]:
    """The Patient.info leaflet pages: keep the assessment/red-flag body
    (from 'How to assess if a head injury is serious' up to the child
    drowsiness note), dropping the share/language page furniture and the
    trailing 'patient picks'/FAQ/related-articles boilerplate."""
    return _keep_between(
        page_lines(html),
        "How to assess if a head injury is serious",
        "A note about drowsiness",
    )


# ---------------------------------------------------------------------------
# Some public and controlled-document content is committed under data/sources/
# because the publisher blocks automated fetchers or because a dated, curated
# text capture is more reproducible than repeatedly parsing a large PDF. The
# source_note on each entry records the URL and capture date transparently.
# ---------------------------------------------------------------------------


FETCHED_SOURCES = [
    {
        "key": "acem",
        "id": "PROT-ATS",
        "title": "Australasian Triage Scale (ATS) - categories and target times",
        "meta": "ACEM public ATS summary; P06/G24 implementation detail is stored separately",
        "url": "https://acem.org.au/Content-Sources/Advancing-Emergency-Medicine/Better-Outcomes-for-Patients/Triage",
        "source": "ACEM (Australasian College for Emergency Medicine) - Triage",
        "source_class": "professional_policy",
        "jurisdiction": "Australasia",
        "scope": "ATS categories, response targets and implementation principles",
        "document_date": "2023-11",
        "review_date": "2026-11",
        "retrieved": "17/09/2026",
        "parser": acem_lines,
    },
    {
        "key": "acem-p06",
        "id": "PROT-ATS-P06",
        "title": "ACEM P06 V5 - Policy on the Australasian Triage Scale",
        "meta": "ACEM policy P06 V5, November 2023",
        "url": "https://policy.acem.org.au/index.php/policies-menu/p06-policy-on-the-australasian-triage-scale",
        "source": "ACEM (Australasian College for Emergency Medicine) - Policy P06 V5",
        "source_class": "professional_policy",
        "jurisdiction": "Australasia",
        "scope": "ATS policy, regular reassessment, staffing, performance thresholds and urgent-category protection",
        "document_date": "2023-11",
        "review_date": "2026-11",
        "retrieved": "25/09/2026",
        "parser": acem_policy_lines,
    },
    {
        "key": "kidshealth-fever",
        "id": "PROT-FEVER",
        "title": "Fever in children - when to get help",
        "meta": "Parent guidance (Paediatric Society of NZ & Starship)",
        "url": "https://www.kidshealth.org.nz/fever-in-children",
        "source": "KidsHealth NZ - Fever in children",
        "parser": kidshealth_lines,
    },
    {
        "key": "healthed-he4240",
        "id": "PROT-BABY-DANGER",
        "title": "Baby and Child Sickness - Danger Signs (HE4240)",
        "meta": "HealthEd leaflet, reviewed October 2024",
        "url": "https://healthed.govt.nz/products/baby-and-child-sickness-danger-signs-english-version",
        "source": "HealthEd (Health New Zealand) - Baby and Child Sickness: Danger Signs (HE4240)",
        "parser": healthed_lines,
    },
    {
        "key": "healthed-he3514",
        "id": "PROT-WHERE",
        "title": "Healthcare - where should I go? (HE3514)",
        "meta": "HealthEd leaflet, reviewed August 2025",
        "url": "https://healthed.govt.nz/products/healthcare-where-should-i-go-english-he3514",
        "source": "HealthEd (Health New Zealand) - Healthcare: where should I go? (HE3514)",
        "parser": healthed_lines,
    },
    {
        "key": "sepsis",
        "id": "PROT-SEPSIS",
        "title": "Sepsis - early recognition (mate whakataoke)",
        "meta": "NZ Sepsis Trust public guidance",
        "url": "https://www.sepsis.org.nz/what-is-sepsis/",
        "source": "NZ Sepsis Trust - What is sepsis?",
        "parser": page_lines,
    },
    {
        "key": "healthdirect-chest-pain",
        "id": "PROT-CHEST-PAIN",
        "title": "Chest pain - red flags and heart attack warning signs",
        "meta": "healthdirect public guidance, reviewed September 2025",
        "url": "https://www.healthdirect.gov.au/chest-pain",
        "source": "healthdirect (Australian government) - Chest pain",
        "parser": page_lines,
        "trim": lambda lines: _slice_lines(lines, 4000),
    },
    {
        "key": "kidshealth-meningo",
        "id": "PROT-MENINGO",
        "title": "Meningococcal disease - fever and rash red flags in children",
        "meta": "Parent guidance (Paediatric Society of NZ & Starship)",
        "url": "https://www.kidshealth.org.nz/meningococcal-disease",
        "source": "KidsHealth NZ - Meningococcal disease",
        "parser": kidshealth_lines,
        "trim": lambda lines: _keep_ranges(
            lines,
            [
                ("Key points about meningococcal disease", "Find out about protecting"),
                (
                    "Signs and symptoms of meningococcal disease",
                    "Treating meningococcal disease",
                ),
            ],
        ),
    },
    {
        "key": "patient-head-injury",
        "id": "PROT-HEAD-INJURY",
        "title": "Head injuries - severity assessment and when to seek help",
        "meta": "Patient.info clinician-reviewed leaflet, updated June 2025",
        "url": "https://patient.info/brain-nerves/head-injuries",
        "source": "Patient.info - Head injuries (peer reviewed by Dr Toni Hazell, FRCGP)",
        "parser": patient_info_lines,
    },
    {
        "key": "rcp-news2",
        "id": "PROT-NEWS2",
        "title": "NEWS2 - National Early Warning Score 2 (how the score works)",
        "meta": "Royal College of Physicians (UK), December 2017",
        "url": "https://www.rcp.ac.uk/resources/national-early-warning-score-news-2/",
        "source": "Royal College of Physicians - National Early Warning Score (NEWS) 2",
        "parser": page_lines,
        "trim": lambda lines: _keep_between(
            lines, "NEWS2 is the latest version", "The NEWS2 Report"
        ),
    },
    {
        "key": "pmc-news2-triggers",
        "id": "PROT-NEWS2-TRIGGERS",
        "title": "NEWS2 trigger thresholds - clinical response and escalation",
        "meta": "BMJ Postgraduate Medical Journal, 'Using NEWS2' (Welch, Dean, Hartin)",
        "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9761428/",
        "source": "PMC (BMJ PGJ) - Using NEWS2: an essential component of reliable clinical assessment",
        "parser": pmc_lines,
    },
    {
        "key": "rch-vitals",
        "id": "PROT-PEDS-VITALS",
        "title": "Paediatric vital signs - acceptable ranges by age",
        "meta": "Royal Children's Hospital Melbourne CPG, updated July 2023",
        "url": "https://www.rch.org.au/clinicalguide/guideline_index/normal_ranges_for_physiological_variables/",
        "source": "The Royal Children's Hospital Melbourne - Acceptable ranges for physiological variables",
        "parser": rch_lines,
    },
    {
        "key": "kidshealth-bronchiolitis",
        "id": "PROT-BRONCHIOLITIS",
        "title": "Bronchiolitis - urgent-help and 111 criteria for babies",
        "meta": "Parent guidance (Paediatric Society of NZ & Starship)",
        "url": "https://www.kidshealth.org.nz/bronchiolitis",
        "source": "KidsHealth NZ - Bronchiolitis",
        "parser": kidshealth_lines,
        "trim": lambda lines: _keep_ranges(
            lines,
            [
                (None, "Bronchiolitis animation"),
                (
                    "When to get medical help for your child with bronchiolitis",
                    "Treatment for bronchiolitis",
                ),
            ],
        ),
    },
    {
        "key": "kidshealth-dehydration",
        "id": "PROT-DEHYDRATION",
        "title": "Dehydration in babies and children - signs and escalation",
        "meta": "Parent guidance (Paediatric Society of NZ & Starship)",
        "url": "https://www.kidshealth.org.nz/dehydration-in-babies-children",
        "source": "KidsHealth NZ - Dehydration in babies & children",
        "parser": kidshealth_lines,
        "trim": lambda lines: _slice_lines(lines, 4000),
    },
    {
        "key": "healthdirect-abdo",
        "id": "PROT-ABDO",
        "title": "Abdominal pain - when to see a doctor, when to seek urgent care",
        "meta": "healthdirect public guidance, reviewed February 2024",
        "url": "https://www.healthdirect.gov.au/abdominal-pain",
        "source": "healthdirect (Australian government) - Abdominal pain",
        "parser": page_lines,
        "trim": lambda lines: _keep_ranges(
            lines,
            [
                (None, "What causes abdominal pain?"),
                ("When should I see my doctor?", "How is abdominal pain treated?"),
            ],
        ),
    },
    {
        "key": "healthdirect-aaa",
        "id": "PROT-AAA",
        "title": "Abdominal aortic aneurysm - ruptured AAA warning signs",
        "meta": "healthdirect public guidance, reviewed November 2024",
        "url": "https://www.healthdirect.gov.au/aortic-aneurysm",
        "source": "healthdirect (Australian government) - Aortic aneurysm",
        "parser": page_lines,
        "trim": lambda lines: _keep_ranges(
            lines, [(None, "What causes an abdominal aortic aneurysm?")]
        ),
    },
]

CAPTURED_SOURCES = [
    {
        "id": "PROT-ED-FLOW",
        "title": "Emergency departments (ED) - when to go, triage and waiting",
        "meta": "Health New Zealand page, last updated 5 August 2026",
        "source_url": "https://www.healthnz.govt.nz/hospitals-services/services-support/emergency-departments",
        "source_name": "Health New Zealand | Te Whatu Ora - Emergency departments (ED)",
        "source_class": "national_guidance",
        "jurisdiction": "New Zealand",
        "scope": "Public ED routing, ATS explanation, waiting and safety-net guidance",
        "document_date": "2026-08-05",
        "review_date": None,
        "retrieved": "25/09/2026",
        "text": read_manual_source("healthnz_ed_flow.md"),
    },
    {
        "id": "PROT-HEALTHLINE",
        "title": "Healthline - free 24/7 nurse advice line (0800 611 116)",
        "meta": "Health New Zealand page, captured 25 September 2026",
        "source_url": "https://www.healthnz.govt.nz/online-phone-healthcare/healthline",
        "source_name": "Health New Zealand | Te Whatu Ora - Healthline",
        "source_class": "national_guidance",
        "jurisdiction": "New Zealand",
        "scope": "Phone advice, GP/urgent-care/ED/ambulance routing, callback and communication support",
        "document_date": "2026-07-22",
        "review_date": None,
        "retrieved": "25/09/2026",
        "text": read_manual_source("healthnz_healthline.md"),
    },
    {
        "id": "PROT-FLOW-SSED",
        "title": "Acute care - shorter stays in ED, access block and whole-system flow",
        "meta": "Health New Zealand acute-care programme guidance, captured 25 September 2026",
        "source_url": "https://www.healthnz.govt.nz/about-us/what-we-do/programmes-and-initiatives/acute-care",
        "source_name": "Health New Zealand | Te Whatu Ora - Acute care",
        "source_class": "national_guidance",
        "jurisdiction": "New Zealand",
        "scope": "Six-hour ED target, access block, hospital occupancy and whole-system patient flow",
        "document_date": None,
        "review_date": None,
        "retrieved": "25/09/2026",
        "text": read_manual_source("healthnz_acute_care.md"),
    },
    {
        "id": "PROT-CCHV-TRIAGE",
        "title": "Te Pae Tiaki ED triage and reassessment",
        "meta": "CCHV controlled document 1.989, issue 19 February 2024, pages 2-7",
        "source_url": "https://static.info.content.health.nz/docs/publications/HNZ00094528-Appendix-Doc-19-42.pdf",
        "source_name": "Health New Zealand | Te Whatu Ora Capital, Coast and Hutt Valley - Triage in Te Pae Tiaki | Emergency Department",
        "source_class": "local_policy",
        "jurisdiction": "New Zealand - Capital, Coast and Hutt Valley",
        "scope": "Local triage staffing, reassessment, ACNM/SMO escalation and safe alternatives",
        "document_date": "2024-02-19",
        "review_date": "2027-02-19",
        "retrieved": "25/09/2026",
        "text": read_manual_source("cchv_te_pae_tiaki_triage.md"),
    },
    {
        "id": "PROT-CCHV-OVERCROWDING",
        "title": "Te Pae Tiaki ED responsibility, overcrowding and variance indicators",
        "meta": "CCHV controlled document 1.1560, issue 20 February 2025, pages 2-5",
        "source_url": "https://static.info.content.health.nz/docs/publications/HNZ00094528-Appendix-Doc-19-42.pdf",
        "source_name": "Health New Zealand | Te Whatu Ora Capital, Coast and Hutt Valley - Responsibility for patient care - Te Pae Tiaki | Emergency Department",
        "source_class": "local_policy",
        "jurisdiction": "New Zealand - Capital, Coast and Hutt Valley",
        "scope": "Local responsibility, bed block, staffing, VIS and hospital-management escalation",
        "document_date": "2025-02-20",
        "review_date": "2028-02-20",
        "retrieved": "25/09/2026",
        "text": read_manual_source("cchv_te_pae_tiaki_overcrowding.md"),
    },
    {
        "id": "PROT-ATS-G24",
        "title": "ACEM G24 V6 - Guidelines on implementation of the ATS",
        "meta": "ACEM G24 V6, November 2023, captured 25 September 2026",
        "source_url": "https://acem.org.au/getmedia/51dc74f7-9ff0-42ce-872a-0437f3db640a/G24_04_Guidelines_on_Implementation_of_ATS_Jul-16.aspx",
        "source_name": "ACEM - Guidelines on the Implementation of the Australasian Triage Scale in Emergency Departments",
        "source_class": "professional_policy",
        "jurisdiction": "Australasia",
        "scope": "ATS assessment, reassessment, documentation, category targets and audit",
        "document_date": "2023-11",
        "review_date": "2026-11",
        "retrieved": "25/09/2026",
        "text": read_manual_source("acem_g24.md"),
    },
    {
        "id": "PROT-ANKLE",
        "title": "Sprained ankle - symptoms, x-ray referral criteria and care",
        "meta": "Health New Zealand page, last updated 22 July 2026",
        "source_url": "https://www.healthnz.govt.nz/health-topics/conditions-treatments/feet-and-ankles/sprained-ankle",
        "source_name": "Health New Zealand | Te Whatu Ora - Sprained ankle",
        "source_class": "national_guidance",
        "jurisdiction": "New Zealand",
        "scope": "Ankle symptoms, x-ray referral criteria, self-care and treatment",
        "document_date": "2026-07-22",
        "review_date": None,
        "retrieved": "17/09/2026",
        "text": read_manual_source("healthnz_ankle.md"),
    },
]


def _dates_from_meta(meta: str) -> tuple[str | None, str | None]:
    """Extract simple month/year dates from legacy source descriptions."""
    months = {
        name: index
        for index, name in enumerate(
            (
                "january",
                "february",
                "march",
                "april",
                "may",
                "june",
                "july",
                "august",
                "september",
                "october",
                "november",
                "december",
            ),
            start=1,
        )
    }
    text = meta.casefold()
    document_date = None
    review_date = None
    for month, number in months.items():
        match = re.search(rf"{month}\s+(\d{{4}})", text)
        if not match:
            continue
        value = f"{match.group(1)}-{number:02d}"
        if "review" in text:
            review_date = value
        elif "updated" in text or "issued" in text or "last" in text:
            document_date = value
    return document_date, review_date


def _authority_fields(spec: dict, fallback_date: str) -> dict:
    """Return explicit authority/scope/date fields for a source entry.

    Every entry should make its authority and limits visible to the runtime
    retrieval tool.  Older source specs pre-date these fields, so infer a
    conservative default from the URL rather than silently leaving them blank.
    """
    url = str(spec.get("url") or spec.get("source_url") or "")
    source = str(spec.get("source") or spec.get("source_name") or "")
    lowered = f"{url} {source}".lower()
    if spec.get("source_class"):
        source_class = spec["source_class"]
    elif "acem.org.au" in lowered or "acem" in lowered:
        source_class = "professional_policy"
    elif any(
        host in lowered
        for host in ("healthnz.govt.nz", "healthed.govt.nz", "kidshealth.org.nz")
    ):
        source_class = "national_guidance"
    elif any(
        host in lowered
        for host in (
            "sepsis.org.nz",
            "healthdirect.gov.au",
            "rch.org.au",
            "patient.info",
            "rcp.ac.uk",
            "pmc.ncbi.nlm.nih.gov",
        )
    ):
        source_class = "public_guidance"
    else:
        source_class = "public_guidance"
    if spec.get("jurisdiction"):
        jurisdiction = spec["jurisdiction"]
    elif any(
        host in lowered
        for host in (
            "healthnz.govt.nz",
            "healthed.govt.nz",
            "kidshealth.org.nz",
            "sepsis.org.nz",
        )
    ):
        jurisdiction = "New Zealand"
    elif any(host in lowered for host in ("healthdirect.gov.au", "rch.org.au")):
        jurisdiction = "Australia"
    elif "rcp.ac.uk" in lowered:
        jurisdiction = "United Kingdom"
    elif source_class == "professional_policy" and "acem" in lowered:
        jurisdiction = "Australasia"
    else:
        jurisdiction = "International"
    inferred_document, inferred_review = _dates_from_meta(str(spec.get("meta", "")))
    return {
        "source_class": source_class,
        "jurisdiction": jurisdiction,
        "scope": spec.get("scope") or spec.get("title", "clinical guidance"),
        "document_date": spec.get("document_date") or inferred_document,
        "review_date": spec.get("review_date") or inferred_review,
        "retrieved": spec.get("retrieved", fallback_date),
    }


def build_protocol_entries() -> list[dict]:
    """Fetch/capture every source, parse to plain text, chunk ~CHUNK_TARGET
    chars at sensible boundaries, and return the protocols.json entries."""
    fetched = FETCHED
    entries = []
    for spec in FETCHED_SOURCES:
        if spec["url"] not in SOURCE_HASHES:
            raise ValueError(f"Missing source hash pin for {spec['url']}")
        html = fetch(spec["url"], spec["key"])
        lines = spec["parser"](html)
        trim = spec.get("trim")
        if trim is not None:
            lines = trim(lines)
        text = "\n".join(lines)
        chunks = chunk_text(text)
        authority = _authority_fields(spec, fetched)
        for i, chunk in enumerate(chunks, 1):
            entry_id = spec["id"] if len(chunks) == 1 else f"{spec['id']}#{i}"
            entries.append(
                {
                    "id": entry_id,
                    "title": spec["title"],
                    "meta": spec["meta"],
                    "source_url": spec["url"],
                    "source_class": authority["source_class"],
                    "jurisdiction": authority["jurisdiction"],
                    "scope": authority["scope"],
                    "document_date": authority["document_date"],
                    "review_date": authority["review_date"],
                    "retrieved": authority["retrieved"],
                    "source_note": (
                        f"{spec['source']}, {spec['url']}, fetched {authority['retrieved']}"
                    ),
                    "text": chunk,
                }
            )
    for spec in CAPTURED_SOURCES:
        chunks = chunk_text(spec["text"])
        authority = _authority_fields(spec, fetched)
        for i, chunk in enumerate(chunks, 1):
            entry_id = spec["id"] if len(chunks) == 1 else f"{spec['id']}#{i}"
            entries.append(
                {
                    "id": entry_id,
                    "title": spec["title"],
                    "meta": spec["meta"],
                    "source_url": spec["source_url"],
                    "source_class": authority["source_class"],
                    "jurisdiction": authority["jurisdiction"],
                    "scope": authority["scope"],
                    "document_date": authority["document_date"],
                    "review_date": authority["review_date"],
                    "retrieved": authority["retrieved"],
                    "source_note": (
                        f"{spec['source_name']}, {spec['source_url']}; committed curated "
                        f"text capture retrieved {authority['retrieved']}. The capture is "
                        "a point-in-time reference, not a replacement for the live "
                        "controlled source."
                    ),
                    "text": chunk,
                }
            )
    return entries


def validate_protocols(entries: list[dict]) -> None:
    """Fail loudly if a source parser produced an unusable corpus entry."""
    if not entries:
        raise ValueError("protocol corpus is empty")
    ids = [entry.get("id") for entry in entries]
    if len(ids) != len(set(ids)):
        raise ValueError("protocol corpus contains duplicate ids")
    for entry in entries:
        missing = [
            field
            for field in (
                "id",
                "title",
                "meta",
                "source_url",
                "source_class",
                "jurisdiction",
                "scope",
                "source_note",
                "text",
            )
            if not entry.get(field)
        ]
        if missing:
            raise ValueError(
                f"protocol entry {entry.get('id', '<unknown>')!r} is missing "
                + ", ".join(missing)
            )


# ---------------------------------------------------------------------------
# patients.json is hand-authored committed data (the single source of truth
# for the simulated ED): clinical vignettes for the situations; no real
# persons. This script does not write it - it validates and reports.
# ---------------------------------------------------------------------------


def validate_patients(path: Path) -> dict:
    """Load the committed patients.json and sanity-check its internal
    consistency: id formats, bed-board references and statuses."""
    data = json.loads(path.read_text())
    patients = data["patients"]
    beds = data["monitored_beds"]

    errors = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    check(bool(data.get("description")), "missing top-level description")

    ids = [p["id"] for p in patients]
    check(len(ids) == len(set(ids)), "duplicate patient ids")
    for pid in ids:
        check(
            re.fullmatch(r"PT-[A-Z0-9]{7}|HL-CALL-\d+", pid) is not None,
            f"unexpected id format: {pid}",
        )

    dept = {p["id"] for p in patients if not p["id"].startswith("HL-CALL")}
    by_id = {p["id"]: p for p in patients}
    for bed in beds:
        pid = bed.get("patient_id")
        check(
            isinstance(bed.get("capabilities"), list) and bool(bed["capabilities"]),
            f"{bed['bed']}: missing monitored-bed capabilities",
        )
        if bed["status"] == "occupied":
            check(pid in dept, f"{bed['bed']}: occupied but {pid} not loaded")
            if pid in by_id:
                loc = by_id[pid].get("location", "")
                check(
                    loc.endswith(bed["bed"]),
                    f"{bed['bed']}: record location {loc!r} does not match bed",
                )
                check(
                    by_id[pid].get("status") == "in monitored bed",
                    f"{bed['bed']}: occupant status {by_id[pid].get('status')!r}",
                )
        else:
            check(pid is None, f"{bed['bed']}: free but lists {pid}")
    occupant_ids = {b["patient_id"] for b in beds if b["status"] == "occupied"}
    in_bed = {
        p["id"]
        for p in patients
        if (p.get("location") or "").startswith("Monitored bed")
    }
    check(occupant_ids == in_bed, "bed board and locations disagree")
    for patient in patients:
        if patient.get("status") == "awaiting monitored bed":
            check(
                bool(patient.get("monitoring_requirement")),
                f"{patient.get('id')}: missing monitoring_requirement",
            )

    waiting = [p for p in patients if p.get("status") == "waiting triage review"]
    await_bed = [p for p in patients if p.get("status") == "awaiting monitored bed"]
    callers = [p for p in patients if p["id"].startswith("HL-CALL")]
    summary = {
        "records": len(patients),
        "waiting_triage": len(waiting),
        "awaiting_monitored_bed": len(await_bed),
        "in_monitored_bed": len(in_bed),
        "nurse_line_callers": len(callers),
        "beds_free": sum(1 for b in beds if b["status"] != "occupied"),
        "errors": errors,
    }
    return summary


def validate_department_snapshots(
    path: Path,
    *,
    monitored_occupied: int | None = None,
    monitored_total: int | None = None,
) -> dict:
    """Check that each operational fixture exposes the fields used by the tool."""
    data = json.loads(path.read_text(encoding="utf-8"))
    snapshots = data.get("snapshots")
    errors: list[str] = []
    if not isinstance(snapshots, dict) or not snapshots:
        errors.append("missing department snapshots")
        snapshots = {}
    required = {
        "snapshot_id",
        "site",
        "observed_at",
        "local_time",
        "status",
        "staffing",
        "demand",
        "capacity",
        "flow",
        "escalation",
        "alternatives",
        "communications",
        "governance",
    }
    section_fields = {
        "staffing": {
            "triage_nurse",
            "acnm",
            "duty_smo",
            "duty_nurse_manager",
            "patient_flow_coordinator",
            "hospital_management",
            "rostering",
        },
        "demand": {
            "ed_census",
            "waiting_room_patients",
            "arrivals_last_hour",
            "expected_arrivals_next_hour",
            "ambulance_offload_wait_minutes",
            "acuity_mix_scope",
            "acuity_mix",
        },
        "capacity": {
            "treatment_spaces",
            "resuscitation_spaces",
            "monitored_beds",
            "inpatient_beds",
            "icu_beds",
        },
        "flow": {
            "median_wait",
            "longest_wait",
            "ats_breaches",
            "access_block_patients",
            "boarding_patients",
        },
        "escalation": {
            "trigger",
            "variance_indicator",
            "acnm_informed",
            "duty_smo_informed",
            "duty_nurse_manager_informed",
            "hospital_management_informed",
            "actions",
        },
        "alternatives": {
            "regular_gp",
            "after_hours_gp",
            "urgent_care",
            "ambulance",
            "healthline",
        },
        "communications": {
            "last_waiting_room_update",
            "delay_reason",
            "next_update_due",
        },
        "governance": {
            "source",
            "authority",
            "data_quality",
            "captured_at",
        },
    }
    for key, snapshot in snapshots.items():
        if not isinstance(snapshot, dict):
            errors.append(f"{key}: snapshot is not an object")
            continue
        missing = sorted(required - snapshot.keys())
        if missing:
            errors.append(f"{key}: missing {', '.join(missing)}")
        for section, fields in section_fields.items():
            value = snapshot.get(section)
            if not isinstance(value, dict):
                errors.append(f"{key}: {section} is not an object")
                continue
            section_missing = sorted(fields - value.keys())
            if section_missing:
                errors.append(f"{key}: {section} missing {', '.join(section_missing)}")
        if snapshot.get("snapshot_id") != key:
            errors.append(f"{key}: snapshot_id does not match its key")
        observed_at = snapshot.get("observed_at")
        captured_at = snapshot.get("governance", {}).get("captured_at")
        if observed_at and captured_at:
            try:
                if datetime.fromisoformat(observed_at) != datetime.fromisoformat(
                    captured_at
                ):
                    errors.append(f"{key}: observed_at and captured_at disagree")
            except ValueError:
                errors.append(f"{key}: observed_at/captured_at is not ISO-8601")
        trigger = str(snapshot.get("escalation", {}).get("trigger", "")).casefold()
        available_beds = (
            snapshot.get("capacity", {}).get("monitored_beds", {}).get("available", 0)
        )
        if "no available monitored bed" in trigger and available_beds:
            errors.append(
                f"{key}: escalation trigger contradicts monitored-bed availability"
            )
        if monitored_occupied is not None and monitored_total is not None:
            bed_state = snapshot.get("capacity", {}).get("monitored_beds", {})
            if (
                bed_state.get("occupied") != monitored_occupied
                or bed_state.get("total") != monitored_total
                or bed_state.get("available") != monitored_total - monitored_occupied
            ):
                errors.append(
                    f"{key}: monitored-bed capacity does not match patients.json"
                )
    return {"snapshots": len(snapshots), "errors": errors}


def validate_situations(
    path: Path, patient_ids: set[str], snapshots: dict[str, dict]
) -> list[str]:
    """Check that situation scoping cannot point at missing fixture records."""
    data = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for situation in data.get("situations", []):
        sid = situation.get("id", "<unknown>")
        assigned = situation.get("assigned_patient_ids") or []
        if not assigned:
            errors.append(f"{sid}: missing assigned_patient_ids")
        for patient_id in assigned:
            if patient_id not in patient_ids:
                errors.append(
                    f"{sid}: assigned patient {patient_id!r} is not in patients.json"
                )
        snapshot_id = situation.get("department_snapshot_id")
        if not snapshot_id:
            errors.append(f"{sid}: missing department_snapshot_id")
        elif snapshot_id not in snapshots:
            errors.append(
                f"{sid}: snapshot {snapshot_id!r} is not in department_snapshots.json"
            )
        else:
            simulation_date = situation.get("simulation_start_date")
            snapshot_date = str(snapshots[snapshot_id].get("local_time", "")).split(
                " ", 1
            )[0]
            if simulation_date:
                expected = datetime.strptime(simulation_date, "%Y-%m-%d").strftime(
                    "%d/%m/%Y"
                )
                if snapshot_date != expected:
                    errors.append(
                        f"{sid}: simulation date {simulation_date} does not match snapshot date {snapshot_date}"
                    )
    return errors


def main(argv=None):
    opts = argparse.ArgumentParser(
        description="Build the ed_triage profile's environment data."
    )
    opts.add_argument("--dry-run", action="store_true", help="stats only, no write")
    args = opts.parse_args(argv)

    print("Fetching/parsing triage-protocol sources ...")
    entries = build_protocol_entries()
    validate_protocols(entries)
    source_counts = {}
    for e in entries:
        source_counts[e["id"].split("#")[0]] = (
            source_counts.get(e["id"].split("#")[0], 0) + 1
        )
    print(f"  {len(entries)} protocol chunks: {source_counts}")

    print("Validating committed patients.json (the simulated ED) ...")
    patients = validate_patients(PATIENTS_OUT)
    print(
        f"  {patients['records']} records "
        f"({patients['waiting_triage']} waiting triage, "
        f"{patients['awaiting_monitored_bed']} awaiting monitored bed, "
        f"{patients['in_monitored_bed']} in monitored bed, "
        f"{patients['nurse_line_callers']} nurse-line caller(s)); "
        f"{patients['beds_free']} monitored bed(s) free"
    )
    if patients["errors"]:
        for err in patients["errors"]:
            print(f"  WARNING: {err}")
    else:
        print("  patients.json passes sanity checks")

    print("Validating committed department snapshots ...")
    snapshots = validate_department_snapshots(
        SNAPSHOTS_OUT,
        monitored_occupied=patients["in_monitored_bed"],
        monitored_total=patients["in_monitored_bed"] + patients["beds_free"],
    )
    print(f"  {snapshots['snapshots']} operational snapshots")
    if snapshots["errors"]:
        for err in snapshots["errors"]:
            print(f"  WARNING: {err}")
    else:
        print("  department_snapshots.json passes sanity checks")

    patient_ids = {
        p["id"]
        for p in json.loads(PATIENTS_OUT.read_text(encoding="utf-8"))["patients"]
    }
    snapshots_by_id = json.loads(SNAPSHOTS_OUT.read_text(encoding="utf-8"))["snapshots"]
    situation_errors = validate_situations(
        PROFILE_DIR / "situations.json", patient_ids, snapshots_by_id
    )
    if situation_errors:
        for err in situation_errors:
            print(f"  WARNING: {err}")
    else:
        print("  situations.json references valid assigned patients and snapshots")

    if patients["errors"] or snapshots["errors"] or situation_errors:
        raise SystemExit("ED fixture validation failed; no data was written.")

    if args.dry_run:
        for e in entries:
            print(f"  [{e['id']}] {e['title']}\n      {e['text'][:120]}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        PROTOCOLS_OUT,
        {
            "description": (
                "Triage and operational-guidance library for the ed_triage "
                "profile, built from labelled public, professional, local and "
                "national sources: ACEM ATS/P06/G24; CCHV Te Pae Tiaki triage "
                "and overcrowding responsibility material; Health New Zealand "
                "acute-care, ED, Healthline and ankle guidance; and the "
                "existing clinical red-flag sources. Each entry carries source "
                "class, jurisdiction, scope and document/review/retrieval dates. "
                "Committed captures are point-in-time references, not current "
                "hospital orders. lookup_triage_protocol and "
                "lookup_operational_guidance retrieve the relevant entries "
                "lexically."
            ),
            "protocols": entries,
        },
    )
    print(f"Wrote {PROTOCOLS_OUT}")
    print(f"Left {PATIENTS_OUT} untouched (committed hand-authored data).")
    print(f"Left {SNAPSHOTS_OUT} untouched (committed operational fixtures).")


if __name__ == "__main__":
    main()
