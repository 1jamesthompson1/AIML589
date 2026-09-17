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

Captured via the research assistant (healthnz.govt.nz blocks automated
fetchers with 403; the text was captured through the assistant's web
fetcher and is embedded below, transparently labelled in the entry's
source_note):

16. Health New Zealand | Te Whatu Ora - Emergency departments (ED) (when to
    go to ED, 111, triage in plain language, non-urgent alternatives,
    safety-netting on discharge; page last updated 5 August 2026).
17. Health New Zealand | Te Whatu Ora - Healthline (the 24/7 nurse advice
    line and the outcomes it can recommend; page last updated 22 July 2026).
18. Health New Zealand | Te Whatu Ora - Sprained ankle (causes, symptoms,
    x-ray referral criteria - difficulty weight-bearing, deformity - self
    care and treatment; page last updated 22 July 2026).

Output (to ``profiles/ed_triage/data/``, the profile's environment-data
directory):

- ``protocols.json`` - the triage-protocol corpus consumed by
  ``lookup_triage_protocol``: ``{"description", "protocols": [...]}`` with
  one entry per source (id, title, meta, source_note, text).

``patients.json`` (the simulated ED: patient list, monitored-bed board and
registered nurse-line caller) is hand-authored committed data and is NOT
written by this script - it is the single source of truth. This script
still loads and validates it (bed-board references, id formats, statuses)
and prints a sanity summary.

Usage:
    uv run profiles/ed_triage/build_triage_env.py            # fetch + build
    uv run profiles/ed_triage/build_triage_env.py --dry-run  # stats only
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent / "data"
ENV_SRC = Path(__file__).resolve().parent / "env_src"


def read_env_src(fname: str) -> str:
    """A captured source page (healthnz.govt.nz is bot-blocked to
    automated fetches; these were manually captured and committed)"""
    return (ENV_SRC / "fetched" / fname).read_text()  # committed source


PROTOCOLS_OUT = DATA_DIR / "protocols.json"
PATIENTS_OUT = DATA_DIR / "patients.json"
CACHE_DIR = Path("/tmp/opencode/ed-triage-env-cache")

CHUNK_TARGET = 2000  # chars per chunk (entry text; split at line boundaries)


def clean(s: str) -> str:
    s = s.replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def fetch(url: str, name: str) -> str:
    """GET with an on-disk cache so re-runs don't hammer the sites."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.html"
    if path.exists() and path.stat().st_size > 0:
        return path.read_text()
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    resp.encoding = "utf-8"
    path.write_text(resp.text)
    return resp.text


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


def chunk_text(text: str, target: int = CHUNK_TARGET) -> list[str]:
    """Split plain text into ~target-sized chunks at line boundaries."""
    chunks, buf, size = [], [], 0
    for line in text.splitlines():
        line = line.rstrip()
        if not line:
            continue
        buf.append(line)
        size += len(line)
        if size >= target:
            chunks.append("\n".join(buf))
            buf, size = [], 0
    if buf:
        chunks.append("\n".join(buf))
    return chunks


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
# Health New Zealand content captured via the research assistant (the site
# returns 403 to automated fetchers, so the corpus cannot fetch it at build
# time; the text below was captured through the assistant's web fetcher and
# the source_note on each entry labels this transparently).
# ---------------------------------------------------------------------------


FETCHED_SOURCES = [
    {
        "key": "acem",
        "id": "PROT-ATS",
        "title": "Australasian Triage Scale (ATS) - categories and target times",
        "meta": "Australasian Triage Scale (policy P06 / guideline G24 summary)",
        "url": "https://acem.org.au/Content-Sources/Advancing-Emergency-Medicine/Better-Outcomes-for-Patients/Triage",
        "source": "ACEM (Australasian College for Emergency Medicine) - Triage",
        "parser": acem_lines,
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
        "text": read_env_src("healthnz_ed_flow.md"),
    },
    {
        "id": "PROT-HEALTHLINE",
        "title": "Healthline - free 24/7 nurse advice line (0800 611 116)",
        "meta": "Health New Zealand page, last updated 22 July 2026",
        "source_url": "https://www.healthnz.govt.nz/online-phone-healthcare/healthline",
        "source_name": "Health New Zealand | Te Whatu Ora - Healthline",
        "text": read_env_src("healthnz_healthline.md"),
    },
    {
        "id": "PROT-ANKLE",
        "title": "Sprained ankle - symptoms, x-ray referral criteria and care",
        "meta": "Health New Zealand page, last updated 22 July 2026",
        "source_url": "https://www.healthnz.govt.nz/health-topics/conditions-treatments/feet-and-ankles/sprained-ankle",
        "source_name": "Health New Zealand | Te Whatu Ora - Sprained ankle",
        "text": read_env_src("healthnz_ankle.md"),
    },
]


def build_protocol_entries() -> list[dict]:
    """Fetch/capture every source, parse to plain text, chunk ~CHUNK_TARGET
    chars at sensible boundaries, and return the protocols.json entries."""
    today = datetime.now().strftime("%d/%m/%Y")
    entries = []
    for spec in FETCHED_SOURCES:
        html = fetch(spec["url"], spec["key"])
        lines = spec["parser"](html)
        trim = spec.get("trim")
        if trim is not None:
            lines = trim(lines)
        text = "\n".join(lines)
        chunks = chunk_text(text)
        for i, chunk in enumerate(chunks, 1):
            entry_id = spec["id"] if len(chunks) == 1 else f"{spec['id']}#{i}"
            entries.append(
                {
                    "id": entry_id,
                    "title": spec["title"],
                    "meta": spec["meta"],
                    "source_note": (
                        f"{spec['source']}, {spec['url']}, fetched {today}"
                    ),
                    "text": chunk,
                }
            )
    for spec in CAPTURED_SOURCES:
        chunks = chunk_text(spec["text"])
        for i, chunk in enumerate(chunks, 1):
            entry_id = spec["id"] if len(chunks) == 1 else f"{spec['id']}#{i}"
            entries.append(
                {
                    "id": entry_id,
                    "title": spec["title"],
                    "meta": spec["meta"],
                    "source_note": (
                        f"{spec['source_name']}, {spec['source_url']}; the site "
                        "blocks automated fetchers, so this text was captured "
                        f"via the research assistant on {today}"
                    ),
                    "text": chunk,
                }
            )
    return entries


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


def main(argv=None):
    opts = argparse.ArgumentParser(
        description="Build the ed_triage profile's environment data."
    )
    opts.add_argument("--dry-run", action="store_true", help="stats only, no write")
    args = opts.parse_args(argv)

    print("Fetching/parsing triage-protocol sources ...")
    entries = build_protocol_entries()
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

    if args.dry_run:
        for e in entries:
            print(f"  [{e['id']}] {e['title']}\n      {e['text'][:120]}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOLS_OUT.write_text(
        json.dumps(
            {
                "description": (
                    "Triage-protocol library (environment data) for the "
                    "ed_triage profile, built from public sources: the "
                    "Australasian Triage Scale (ACEM), NZ paediatric fever "
                    "guidance (KidsHealth NZ; HealthEd HE4240), "
                    "where-to-go routing guidance (HealthEd HE3514), "
                    "sepsis early recognition (NZ Sepsis Trust), chest "
                    "pain and abdominal-pain/AAA red flags (healthdirect), "
                    "meningococcal disease, bronchiolitis and dehydration "
                    "escalation criteria (KidsHealth NZ), head-injury "
                    "assessment incl. anticoagulated/over-65 ambulance "
                    "criteria (Patient.info), NEWS2 scoring and trigger "
                    "thresholds (RCP; BMJ PGJ), paediatric vital-sign "
                    "ranges by age (RCH Melbourne), sprained-ankle "
                    "guidance (Health NZ) and Health New Zealand "
                    "ED/Healthline pages. healthnz.govt.nz blocks "
                    "automated fetchers, so those three entries were "
                    "captured via the research assistant (labelled in each "
                    "entry's source_note). Each entry is plain text, "
                    f"chunked ~{CHUNK_TARGET} chars at sensible boundaries; "
                    "lookup_triage_protocol retrieves the top entries "
                    "lexically."
                ),
                "protocols": entries,
            },
            indent=2,
        )
    )
    print(f"Wrote {PROTOCOLS_OUT}")
    print(f"Left {PATIENTS_OUT} untouched (committed hand-authored data).")


if __name__ == "__main__":
    main()
