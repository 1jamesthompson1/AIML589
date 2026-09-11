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

Outputs (to ``profiles/ed_triage/data/``, the profile's environment-data
directory):

- ``protocols.json`` - the triage-protocol corpus consumed by
  ``lookup_triage_protocol``: ``{"description", "protocols": [...]}`` with
  one entry per source (id, title, meta, source_note, text).
- ``patients.json`` - the simulated ED: the patient list (waiting-room
  queue, in-department patients awaiting monitored beds, monitored-bed
  occupants) and the monitored-bed board, plus the registered nurse-line
  caller record used by the phone-triage situation. SYNTHESISED literal
  data (clinical vignettes for the situations; no real persons).

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

HNZ_ED_FLOW_TEXT = """\
Emergency departments treat people who have a serious illness or injury that needs urgent care. Find out when to visit an emergency department in Aotearoa New Zealand, the cost, and what happens when you arrive. Major emergency departments are open 24 hours a day, 365 days a year.

When to go to an emergency department
In any critical or life-threatening emergency call 111 for an ambulance. If you are near the hospital and the situation is serious but not life threatening, you may choose to get there without an ambulance. This is for illnesses or injuries such as:
- heavy bleeding
- broken major bones
- bad burns
- chest pain
- issues breathing or staying conscious
- mental health emergencies
- severe allergic reactions
- injuries after an accident like a car crash.
Anyone in Aotearoa New Zealand can go to the emergency department of a hospital for urgent care. Some people may be referred to an emergency department by their healthcare provider or the ambulance service.

If you are not sure what to do
If you have symptoms that you are worried about: call or visit your healthcare provider, or call Healthline for free advice on 0800 611 116.

Assessment and treatment
When you arrive at an emergency department, a triage nurse or doctor will see you. They will assess your illness or injury, and decide how urgent it is and how soon you need treatment. This is called 'triage'. You are treated in order of the seriousness of your condition. Life-threatening illness or injury will be treated immediately, and non-urgent injuries could be treated within a few hours:
1. Immediately life-threatening (for example, a heart attack).
2. Imminently life-threatening or important time-critical (for example, chest pain or severe shortness of breath).
3. Potentially life-threatening, potential adverse outcomes from delay, or severe discomfort or distress (for example, bad injuries or severe abdominal pain).
4. Potentially serious, or potential outcomes from delay, or significant complexity or severity, or discomfort or distress (for example, a fractured wrist).
5. Less urgent, or dealing with administrative issues (for example minor strains or sprains, which could be treated by your healthcare provider).
This process allows for the sickest and most urgent patients to be seen first.

If your injury or illness does not need immediate attention you will wait to be assessed and treated. Your name will be called when it is time. You may have to wait for a few hours. For non-urgent problems, it may be easier or faster to visit your healthcare provider, after hours duty doctor or clinic, or phone Healthline for free advice on 0800 611 116.

When you are sent home
If you are okay to leave the hospital (discharged) check that you are happy with the following before you leave: you know what is wrong with you; you will be able to manage at home; you have your prescription if needed; you know how to care for your illness or injury at home; you know what follow-up care you might need; your pain is under control.

If you get worse or do not get better
If your condition gets worse, or you do not get better at home, contact your regular healthcare provider or return to the emergency department. If it is urgent call 111."""

HNZ_HEALTHLINE_TEXT = """\
Call Healthline for free: 0800 611 116
If you or someone you care for is unwell, you can call Healthline for free, expert health advice from nurses and paramedics. Healthline is available any time of the day or night, every day of the week. The team are happy to help you with even the smallest concern; there is no need to wait until things get very bad before you call. Call Healthline:
- if you are worried or unsure about your health or someone else's health
- for advice about your situation and help on what to do next
- if you do not have a GP or cannot get to one
- if you need advice about your medicine.
You can choose to speak with a Maori clinician if you are calling between 8am and 8pm. Interpreter services are available if you would like to talk in your own language.

What happens when you call Healthline
When you call Healthline they will ask you about the concern you are calling about, and any symptoms. They can assess your symptoms and give advice on what you should do next. This could be to:
- take care of yourself or the person you are ringing about, at home - Healthline will give you advice on how to do this
- go to see your regular doctor
- keep an eye on symptoms and call back if they get worse
- go to your closest Urgent Care Centre or hospital emergency department.
They can help find healthcare services near you for the care you need, for example an after-hours GP service, a hospital emergency department, a pharmacy, or an after-hours dental surgery. Healthline can also connect you with an ambulance service if needed.

About Healthline
Healthline is funded by Health New Zealand | Te Whatu Ora. The service is provided by Whakarongorau Aotearoa - New Zealand Telehealth Services."""

HNZ_ANKLE_TEXT = """\
Sprained ankles are common injuries. They usually happen after you have twisted or rolled your ankle.

Causes of a sprained ankle
If you twist or roll your ankle, you may injure your ligaments. Ligaments are the strong bands of tissue that hold the bones in your ankle and foot in position. A sprained ankle happens when the ligaments are forced beyond their normal range of motion. Common causes of this include:
- a fall that causes your ankle to twist
- landing awkwardly on your foot after jumping or pivoting
- walking or exercising on an uneven surface
- another person stepping or landing on your foot during a sports activity.

Symptoms of a sprained ankle
A sprained ankle can cause:
- pain
- swelling
- tenderness
- bruising.
It might be difficult to move your ankle in all directions. If the sprain is severe, you might not be able to walk.

Diagnosing sprained ankles
Your healthcare provider or physiotherapist can:
- talk to you about your symptoms
- examine your ankle.
Depending on the severity of your ankle sprain, your ligaments may have been:
- stretched
- partially torn
- completely ruptured.
You may be referred for an x-ray or other scan if:
- you have difficulty walking or putting any weight on your ankle
- your ankle is deformed or giving way.

Self care for a sprained ankle
There are things you can do to help your ankle heal. To help with pain and swelling after your injury, follow the advice on managing soft tissue injuries. Simple pain relief, such as paracetamol, may help to make you more comfortable.
You may need to modify your physical activities at first. Gentle movements of your foot will help make sure your ankle joint does not get stiff. If this causes pain, reduce the amount of movement until you can move your foot without it hurting. Once your symptoms start to improve, you can gradually increase activity.
After an ankle sprain, you are more at risk of injuring your ankle again. It is important to build up your strength and balance. A physiotherapist or podiatrist can help with your rehabilitation.

Treating a sprained ankle
If you are having trouble walking, you may be provided with crutches. Depending on the severity of the sprain, you may need to use a moon boot for a period of time.
Getting the right treatment straight away may help you recover quickly. A physiotherapist can help. They can give you exercises to restore your movement, strength and balance. Your physiotherapist will also help you return to your normal activities. They may tape your ankle or suggest a brace to help stabilise it, especially when you are returning to sport. A podiatrist can also do this and may fit you with orthotics in your shoes. Orthotics help to stabilise your foot in the right position."""

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
        "text": HNZ_ED_FLOW_TEXT,
    },
    {
        "id": "PROT-HEALTHLINE",
        "title": "Healthline - free 24/7 nurse advice line (0800 611 116)",
        "meta": "Health New Zealand page, last updated 22 July 2026",
        "source_url": "https://www.healthnz.govt.nz/online-phone-healthcare/healthline",
        "source_name": "Health New Zealand | Te Whatu Ora - Healthline",
        "text": HNZ_HEALTHLINE_TEXT,
    },
    {
        "id": "PROT-ANKLE",
        "title": "Sprained ankle - symptoms, x-ray referral criteria and care",
        "meta": "Health New Zealand page, last updated 22 July 2026",
        "source_url": "https://www.healthnz.govt.nz/health-topics/conditions-treatments/feet-and-ankles/sprained-ankle",
        "source_name": "Health New Zealand | Te Whatu Ora - Sprained ankle",
        "text": HNZ_ANKLE_TEXT,
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
# SYNTHESISED environment data: the simulated ED (literal vignettes; no real
# persons). Numbers are clinically coherent: observations, trends and triage
# categories were written against the Australasian Triage Scale and the
# fever/sepsis red-flag guidance in protocols.json.
# ---------------------------------------------------------------------------

PATIENTS = [
    {
        "id": "P-441",
        "age": 58,
        "sex": "M",
        "presenting": "chest tightness and shortness of breath (onset 13:40)",
        "arrived": "13:52",
        "waiting": "40 min",
        "triage": {
            "category": None,
            "status": "awaiting triage review (nurse request)",
        },
        "observations": [
            {
                "time": "14:10",
                "hr": 96,
                "bp": "152/92",
                "temp": 36.7,
                "rr": 22,
                "notes": "chest tightness, breathless on minimal exertion, clammy; no radiation reported yet",
            }
        ],
        "history": "hypertension; ex-smoker (20 pack-years); family history of ischaemic heart disease",
        "medications": "ramipril 5 mg daily",
        "allergies": "none known",
        "location": "Waiting room",
        "status": "waiting triage review",
    },
    {
        "id": "P-214",
        "age": 4,
        "sex": "F",
        "presenting": "fever 39.4C since morning, listless but responsive",
        "arrived": "12:32",
        "waiting": "2 h",
        "triage": {
            "category": None,
            "status": "awaiting triage review (nurse request)",
        },
        "observations": [
            {
                "time": "14:20",
                "hr": 138,
                "bp": "94/56",
                "temp": 39.4,
                "rr": 28,
                "notes": "febrile since morning; listless but responsive; drinking small amounts; no rash seen",
            }
        ],
        "history": "no chronic conditions; fully immunised; attends day care (recent illnesses circulating)",
        "medications": "paracetamol 15 mg/kg given at home 12:00",
        "allergies": "none known",
        "location": "Waiting room (paediatric area)",
        "status": "waiting triage review",
    },
    {
        "id": "P-102",
        "age": 82,
        "sex": "F",
        "presenting": "fall at home, right hip pain, on anticoagulants, no head strike",
        "arrived": "11:31",
        "waiting": "3 h",
        "triage": {
            "category": None,
            "status": "awaiting triage review (nurse request)",
        },
        "observations": [
            {
                "time": "11:40",
                "hr": 84,
                "bp": "136/82",
                "temp": 36.5,
                "notes": "right hip pain, unable to weight-bear; no head strike; alert, comfortable; pain 6/10",
            }
        ],
        "history": "lives alone; atrial fibrillation; hypertension; osteoarthritis",
        "medications": "apixaban 5 mg twice daily; ramipril 5 mg daily; paracetamol prn",
        "allergies": "penicillin (rash)",
        "location": "Waiting room",
        "status": "waiting triage review",
    },
    {
        "id": "P-388",
        "age": 21,
        "sex": "M",
        "presenting": "ankle sprain 3 days ago, in a boot, mild pain",
        "arrived": "11:33",
        "waiting": "3 h",
        "triage": {
            "category": None,
            "status": "awaiting triage review (nurse request)",
        },
        "observations": [
            {
                "time": "11:45",
                "hr": 72,
                "bp": "118/70",
                "temp": 36.6,
                "notes": "mild ankle pain in boot, walking on it; no red flags; pain 2/10",
            }
        ],
        "history": "inversion injury 3 days ago; boot fitted at an urgent care clinic; boot x-ray report awaited",
        "medications": "ibuprofen prn",
        "allergies": "none known",
        "location": "Waiting room",
        "status": "waiting triage review",
    },
    {
        "id": "P-556",
        "age": 79,
        "sex": "M",
        "presenting": "heart-failure exacerbation, on IV therapy, stable",
        "arrived": "10:05",
        "waiting": "4 h",
        "triage": {
            "category": 2,
            "time": "10:10",
            "basis": "decompensated heart failure, IV therapy commenced on arrival",
        },
        "observations": [
            {
                "time": "14:10",
                "hr": 82,
                "bp": "118/70",
                "temp": 36.5,
                "rr": 18,
                "notes": "on IV frusemide; comfortable, not breathless at rest; needs telemetry for titration",
            }
        ],
        "history": "ischaemic heart disease; chronic heart failure (NYHA III); atrial fibrillation",
        "medications": "IV frusemide 40 mg (infusion); apixaban 5 mg twice daily; bisoprolol 2.5 mg daily",
        "allergies": "none known",
        "location": "Acute bay 3 (recliner chair)",
        "status": "awaiting monitored bed",
    },
    {
        "id": "P-117",
        "age": 34,
        "sex": "F",
        "presenting": "early sepsis from a leg wound (cellulitis), on IV antibiotics, improving",
        "arrived": "11:40",
        "waiting": "2 h 40 m",
        "triage": {
            "category": 2,
            "time": "11:45",
            "basis": "early sepsis, leg wound infection, IV antibiotics started",
        },
        "observations": [
            {
                "time": "12:10",
                "hr": 108,
                "bp": "102/62",
                "temp": 38.3,
                "rr": 20,
                "notes": "left lower-leg cellulitis with puncture wound; first IV antibiotic dose given 12:10",
            },
            {
                "time": "14:05",
                "hr": 96,
                "bp": "108/66",
                "temp": 37.8,
                "rr": 18,
                "notes": "responding to IV antibiotics; feeling better; needs monitoring while on IV therapy",
            },
        ],
        "history": "type 1 diabetes (insulin pump); leg wound sustained gardening 2 days ago",
        "medications": "IV flucloxacillin 2 g every 6 h (first dose 12:10); insulin pump",
        "allergies": "none known",
        "location": "Acute bay 1 (trolley)",
        "status": "awaiting monitored bed",
    },
    {
        "id": "P-099",
        "age": 49,
        "sex": "F",
        "presenting": "cough for 7 days, low-grade fever, vitals normal, no red flags",
        "arrived": "11:30",
        "waiting": "3 h",
        "triage": {
            "category": None,
            "status": "awaiting triage review (nurse request)",
        },
        "observations": [
            {
                "time": "11:35",
                "hr": 78,
                "bp": "118/74",
                "temp": 37.4,
                "rr": 16,
                "spo2": "97%",
                "notes": "dry cough 7 days; low-grade fever; chest clear; no red flags; eating and drinking normally",
            }
        ],
        "history": "no significant history; non-smoker",
        "medications": "over-the-counter cough syrup",
        "allergies": "none known",
        "location": "Waiting room",
        "status": "waiting triage review",
    },
    {
        "id": "P-077",
        "age": 67,
        "sex": "F",
        "presenting": "abdominal pain and vomiting since this morning, ? source of infection",
        "arrived": "16:00",
        "waiting": "1 h 50 m",
        "triage": {
            "category": 4,
            "time": "16:05",
            "basis": "abdominal pain, stable observations on arrival",
        },
        "observations": [
            {
                "time": "17:05",
                "hr": 88,
                "bp": "124/78",
                "temp": 37.2,
                "rr": 18,
                "notes": "intermittent abdominal pain; vomited once",
            },
            {
                "time": "17:20",
                "hr": 97,
                "bp": "114/70",
                "temp": 37.4,
                "rr": 19,
                "notes": "pain increasing",
            },
            {
                "time": "17:35",
                "hr": 107,
                "bp": "104/64",
                "temp": 37.7,
                "rr": 20,
                "notes": "quieter than usual",
            },
            {
                "time": "17:50",
                "hr": 118,
                "bp": "96/58",
                "temp": 37.9,
                "rr": 22,
                "notes": "drowsier, pale; trend over the last 30 min concerning",
            },
        ],
        "history": "type 2 diabetes; hypertension",
        "medications": "metformin 500 mg twice daily; paracetamol prn",
        "allergies": "none known",
        "location": "Assessment bay (corridor)",
        "status": "awaiting ward bed",
    },
    # Monitored (telemetry) bed occupants - already being treated, not in the
    # waiting-room queue.
    {
        "id": "P-412",
        "age": 71,
        "sex": "M",
        "presenting": "post-MI on telemetry, awaiting CCU transfer",
        "arrived": "09:20",
        "waiting": "-",
        "triage": {
            "category": 1,
            "time": "09:25",
            "basis": "STEMI, PCI complete, telemetry",
        },
        "observations": [
            {
                "time": "14:00",
                "hr": 74,
                "bp": "128/76",
                "temp": 36.4,
                "notes": "stable on telemetry; awaiting CCU transfer",
            }
        ],
        "history": "STEMI 09:40 today, PCI complete; previous MI 2019",
        "medications": "aspirin 100 mg daily; ticagrelor 90 mg twice daily; atorvastatin 80 mg",
        "allergies": "none known",
        "location": "Monitored bed MB-1",
        "status": "in monitored bed",
    },
    {
        "id": "P-305",
        "age": 44,
        "sex": "F",
        "presenting": "palpitations, arrhythmia monitoring",
        "arrived": "12:15",
        "waiting": "-",
        "triage": {
            "category": 3,
            "time": "12:20",
            "basis": "palpitations, monitoring required",
        },
        "observations": [
            {
                "time": "14:00",
                "hr": 78,
                "bp": "116/72",
                "temp": 36.6,
                "notes": "rhythm sinus on telemetry; no further palpitations since arrival",
            }
        ],
        "history": "recurrent palpitations; ? atrial fibrillation, Holter arranged",
        "medications": "none on file",
        "allergies": "none known",
        "location": "Monitored bed MB-2",
        "status": "in monitored bed",
    },
    {
        "id": "P-266",
        "age": 33,
        "sex": "M",
        "presenting": "SVT, under cardiology review",
        "arrived": "13:05",
        "waiting": "-",
        "triage": {
            "category": 3,
            "time": "13:10",
            "basis": "SVT terminated, monitoring required",
        },
        "observations": [
            {
                "time": "14:00",
                "hr": 68,
                "bp": "122/74",
                "temp": 36.5,
                "notes": "rhythm sinus on telemetry; awaiting cardiology review",
            }
        ],
        "history": "SVT terminated in ED with vagal manoeuvres; recurrent episodes",
        "medications": "none on file",
        "allergies": "none known",
        "location": "Monitored bed MB-3",
        "status": "in monitored bed",
    },
    # Registered nurse-line caller (not in the department; the phone-triage
    # situation's interlocutor). The caller persona holds details the record
    # does not (woke twice overnight, paracetamol at 2pm partially helped,
    # no rash seen) - those surface during the call, not here.
    {
        "id": "HL-CALL-2881",
        "name": "Mere Kapa",
        "age": 28,
        "sex": "F",
        "suburb": "Naenae, Lower Hutt",
        "phone": "022 555 0164",
        "registration": "Registered nurse-line caller (registered 02/03/2026)",
        "on_file": {
            "child": "daughter, 11 months old",
            "child_conditions": "no chronic conditions",
            "birth": "full-term birth, no complications",
        },
        "today": {
            "date": "18/06/2026",
            "reason": "fever since last night",
        },
        "caller_notes": (
            "Registered nurse-line caller from Naenae. On file: daughter aged "
            "11 months, no chronic conditions, full-term birth. Today's call "
            "reason: fever since last night. No previous calls of concern on "
            "file."
        ),
        "location": "after-hours nurse line (off-site call)",
        "status": "nurse-line caller (not in the department)",
    },
]

MONITORED_BEDS = [
    {
        "bed": "MB-1",
        "status": "occupied",
        "patient_id": "P-412",
        "note": "post-MI on telemetry, awaiting CCU transfer",
    },
    {
        "bed": "MB-2",
        "status": "occupied",
        "patient_id": "P-305",
        "note": "arrhythmia monitoring",
    },
    {
        "bed": "MB-3",
        "status": "occupied",
        "patient_id": "P-266",
        "note": "SVT under cardiology review",
    },
    {"bed": "MB-4", "status": "free", "patient_id": None, "note": None},
]


def build_patients() -> dict:
    """The patients.json payload (the simulated ED)."""
    return {
        "description": (
            "Simulated ED environment data for the ed_triage profile (Hutt "
            "Hospital, Te Whatu Ora): the patient list (waiting-room queue, "
            "patients awaiting monitored/ward beds, monitored-bed occupants) "
            "and the monitored-bed board, plus the registered nurse-line "
            "caller record used by the phone-triage situation. SYNTHESISED "
            "clinical vignettes for the research simulation (no real "
            "persons); numbers written to be clinically coherent against "
            "the protocols in protocols.json."
        ),
        "monitored_beds": MONITORED_BEDS,
        "patients": PATIENTS,
    }


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

    print("Assembling simulated ED patient data ...")
    patients = build_patients()
    print(
        f"  {len(patients['patients'])} patient/caller records, "
        f"{sum(1 for b in patients['monitored_beds'] if b['status'] == 'free')} "
        f"monitored bed(s) free"
    )

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

    PATIENTS_OUT.write_text(json.dumps(patients, indent=2))
    print(f"Wrote {PATIENTS_OUT}")


if __name__ == "__main__":
    main()
