#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "beautifulsoup4>=4.12",
#   "requests>=2.31",
#   "pypdf>=4.0",
# ]
# ///
"""
Build the lending_officer profile's policy corpus from public sources.

Sources (public, NZ):

1. Kiwibank's public Code of Banking Practice, financial hardship guidance,
   General Terms and Conditions, Credit Card Terms and Conditions and Latitude
   Personal Loan Contract Terms. These are labelled as public
   guidance/contract terms; they are not internal credit-approval policies.
2. Credit Contracts and Consumer Finance Act 2003 - the ENTIRE Act is parsed
   from the whole-act HTML on legislation.govt.nz and chunked per section
   (~CHUNK_TARGET characters per chunk at sub-clause boundaries). This is the
   core responsible-lending law: lender responsibility principles (s 9C),
   affordability/suitability inquiries, unforeseen hardship (ss 55-57A) and
   the oppression/collections-conduct provisions.
3. NZ Bankers Association Code of Banking Practice (nzba.org.nz PDF) - the
   banks' fair-treatment commitments (responsible credit, hardship, dealings
   with customers in difficulty).
4. Consumer Protection (MBIE) CCCFA guidance page - plain-language summary of
   responsible lending duties and the hardship-application process/timeframes.
5. Commerce Commission "new lender" tip sheet (comcom.govt.nz PDF) -
   regulator guidance covering disclosure, hardship and conduct on default.
6. Banking Ombudsman hardship quick guide (bankomb.org.nz) - a committed
   manual capture under ``data/sources/``: the site blocks automated requests
   (HTTP 403), so the builder cannot download it like the other sources.

Outputs (to ``profiles/lending_officer/data/``, the profile's
environment-data directory):

- ``policy.json`` - the chunked lending-policy corpus consumed by the
  ``lookup_lending_policy`` tool. Each entry carries ``id``, ``title``,
  ``meta``, ``source_note`` (source + URL + fetched date) and ``text``.
- ``documents/*.md`` - existing customer-held documents in the situations'
  document tray, validated but not regenerated.

The case data files ``customers.json``, ``credit_reports.json`` and
``applications.json`` are HAND-AUTHORED committed commits - the single
source of truth - and are NOT generated here. The builder reads
``customers.json`` for its build-time consistency check (the itemised
statement lines must sum to the expected monthly aggregates).

Usage:
    uv run profiles/lending_officer/build_lending_env.py            # build
    uv run profiles/lending_officer/build_lending_env.py --dry-run  # stats only
"""

import argparse
from datetime import date
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy_store import chunk_units, write_json as write_json_file
from source_cache import cache_path as source_cache_path, cached_file

PROFILE_DIR = Path(__file__).resolve().parent
DATA_DIR = PROFILE_DIR / "data"
# Public source downloads use the same disposable, URL-shaped cache as the
# other profile builders. The cache is local to this profile and gitignored;
# a fresh run downloads the sources, while later runs parse the saved copies.
CACHE_DIR = PROFILE_DIR / ".cache"

# Fetched date stamped into every source_note (kept constant so re-runs are
# byte-identical / idempotent).
FETCHED = "05/09/2026"

CCCFA_URL = "https://www.legislation.govt.nz/act/public/2003/0052/latest/whole.html"
NZBA_CODE_URL = (
    "https://nzba.org.nz/wp-content/uploads/2025/11/"
    "Code-of-Banking-Practice-A4-print.pdf"
)
CONSUMER_PROTECTION_URL = (
    "https://www.consumerprotection.govt.nz/general-help/consumer-laws/"
    "credit-contracts-and-consumer-finance-act"
)
COMCOM_URL = (
    "https://www.comcom.govt.nz/__data/assets/pdf_file/0029/109874/"
    "New-lender-tip-sheet-December-2018.pdf"
)
BANKOMB_URL = (
    "https://www.bankomb.org.nz/guides-and-cases/quick-guides/lending/"
    "hardship-and-financial-difficulty"
)
KIWIBANK_CODE_URL = (
    "https://media.kiwibank.co.nz/media/documents/Code-Of-Banking-Practice-Apr21.pdf"
)
KIWIBANK_HARDSHIP_URL = (
    "https://www.kiwibank.co.nz/help/accounts/financial-support/"
    "financial-hardship-assistance/"
)
KIWIBANK_GENERAL_TERMS_URL = (
    "https://media.kiwibank.co.nz/media/documents/"
    "General_Terms_and_Conditions_May26.pdf"
)
KIWIBANK_CREDIT_CARD_URL = (
    "https://media.kiwibank.co.nz/media/documents/"
    "Credit_Card_Terms_and_Conditions_Nov25.pdf"
)
KIWIBANK_PERSONAL_LOAN_URL = (
    "https://media.kiwibank.co.nz/media/documents/Personal_Loan_Contract_TCs_MAR21.pdf"
)
# This one source is deliberately committed rather than downloaded: the
# Banking Ombudsman site rejects scripted requests. Keeping the capture under
# data/sources makes the provenance visible alongside the rest of the profile
# fixtures.
BANKOMB_SOURCE = DATA_DIR / "sources" / "bankomb_hardship.md"

CHUNK_TARGET = 2000  # characters per accumulated chunk


# ---------------------------------------------------------------------------
# Fetch + parse helpers
# ---------------------------------------------------------------------------


def cache_path(url: str) -> Path:
    """Return the profile-local cache path for a public source URL."""
    return source_cache_path(CACHE_DIR, url)


def fetch(url: str) -> bytes:
    """Fetch a binary source through the shared URL-shaped cache."""
    return cached_file(url, cache_path(url), timeout=120).read_bytes()


def fetch_text(url: str) -> str:
    """Fetch a text source through the shared URL-shaped cache."""
    return cached_file(url, cache_path(url), text=True, timeout=120).read_text(
        encoding="utf-8"
    )


def read_manual_source(path: Path) -> str:
    """Read a committed capture for a source that blocks automated fetches."""
    if not path.is_file():
        raise SystemExit(
            f"Missing manual source {path}. Restore the committed capture from "
            "git before rebuilding the policy corpus."
        )
    return path.read_text(encoding="utf-8")


def clean(s: str) -> str:
    s = s.replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_act_sections(html: str) -> list[dict]:
    """Provisions of the Act as {number, title, paras} (legislation.govt.nz
    HTML renders provisions as <div class="prov"><h5 class="prov"> with the
    section number in a <span class="label">; sub-provision labels sit on
    ancestor <div class="subprov"> elements). Same parser shape as the
    welfare profile's Social Security Act builder."""
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


def pdf_text(data: bytes) -> list[str]:
    """A PDF's text as paragraph-ish lines (one per non-empty extracted
    line)."""
    import io

    reader = PdfReader(io.BytesIO(data))
    lines = []
    for page in reader.pages:
        for raw in (page.extract_text() or "").splitlines():
            line = clean(raw)
            if line:
                lines.append(line)
    return lines


def html_main_blocks(html: str) -> list[tuple[str, list[str]]]:
    """A page's <main> content as (heading, [paragraph]) blocks."""
    soup = BeautifulSoup(html, "html.parser")
    main = soup.find("main") or soup.body
    for tag in main.find_all(["script", "style", "nav", "footer", "form"]):
        tag.decompose()
    blocks, cur_head, cur_paras = [], "Overview", []
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        if el.name.startswith("h"):
            if cur_paras:
                blocks.append((cur_head, cur_paras))
            cur_head, cur_paras = clean(el.get_text()), []
        elif el.name == "li":
            cur_paras.append("- " + clean(el.get_text()))
        else:
            text = clean(el.get_text())
            if text:
                cur_paras.append(text)
    if cur_paras:
        blocks.append((cur_head, cur_paras))
    return blocks


# ---------------------------------------------------------------------------
# Policy corpus builders
# ---------------------------------------------------------------------------


def build_cccfa_entries() -> list[dict]:
    """The whole CCCFA 2003, chunked per section (source label cccfa_2003)."""
    html = fetch_text(CCCFA_URL)
    sections = parse_act_sections(html)
    source_note = (
        "Credit Contracts and Consumer Finance Act 2003 (whole act), "
        f"legislation.govt.nz: {CCCFA_URL} (fetched {FETCHED}; Crown copyright)."
    )
    entries = []
    for sec in sections:
        body = " ".join(sec["paras"])
        if not body:
            continue
        doc = f"cccfa_2003-s{sec['number']}" if sec["number"] else "cccfa_2003"
        title = f"{sec['number']} {sec['title']}".strip()
        for i, text in enumerate(chunk_units(sec["paras"], CHUNK_TARGET)):
            entries.append(
                {
                    "id": f"{doc}#{i}",
                    "title": title or "Credit Contracts and Consumer Finance Act 2003",
                    "meta": "Credit Contracts and Consumer Finance Act 2003 (NZ legislation)",
                    "source_note": source_note,
                    "text": text,
                }
            )
    return entries


def build_pdf_entries(
    data: bytes,
    *,
    prefix: str,
    title: str,
    meta: str,
    source_note: str,
) -> list[dict]:
    """A fetched PDF as ~CHUNK_TARGET-character entries."""
    entries = []
    for i, text in enumerate(chunk_units(pdf_text(data), CHUNK_TARGET)):
        entries.append(
            {
                "id": f"{prefix}#{i + 1}",
                "title": f"{title} (part {i + 1})",
                "meta": meta,
                "source_note": source_note,
                "text": text,
            }
        )
    return entries


def build_consumer_protection_entries() -> list[dict]:
    """Consumer Protection (MBIE) CCCFA guidance page, chunked by heading."""
    html = fetch_text(CONSUMER_PROTECTION_URL)
    source_note = (
        "Consumer Protection (MBIE), 'Credit Contracts and Consumer Finance "
        f"Act' guidance page: {CONSUMER_PROTECTION_URL} (fetched {FETCHED}; "
        "Crown copyright)."
    )
    entries = []
    for head, paras in html_main_blocks(html):
        for i, text in enumerate(chunk_units(paras, CHUNK_TARGET)):
            entries.append(
                {
                    "id": f"consumer_protection#{len(entries) + 1}",
                    "title": f"CCCFA overview - {head}",
                    "meta": "Consumer Protection (MBIE) CCCFA guidance",
                    "source_note": source_note,
                    "text": text,
                }
            )
    return entries


def build_html_entries(
    url: str,
    *,
    prefix: str,
    title: str,
    meta: str,
    source_note: str,
    heading_contains: str | None = None,
) -> list[dict]:
    """Fetch a Kiwibank public guidance page and chunk its readable blocks."""
    entries = []
    for head, paras in html_main_blocks(fetch_text(url)):
        if heading_contains and heading_contains.casefold() not in head.casefold():
            continue
        block_text = " ".join(paras).casefold()
        heading_text = head.casefold()
        navigation_headings = {
            "contact",
            "help & support",
        }
        if (
            heading_text in navigation_headings
            or "browser is a tad old" in heading_text
            or "browser is a tad old" in block_text
            or "unsupported browser" in block_text
        ):
            continue
        navigation_labels = {
            "products we no longer offer",
            "skip to main content",
        }
        content_paras = [
            para
            for para in paras
            if " ".join(para.split()).casefold() not in navigation_labels
        ]
        for text in chunk_units(content_paras, CHUNK_TARGET):
            entries.append(
                {
                    "id": f"{prefix}#{len(entries) + 1}",
                    "title": f"{title} - {head}",
                    "meta": meta,
                    "source_note": source_note,
                    "text": text,
                }
            )
    return entries


# The Banking Ombudsman site blocks automated requests (HTTP 403 to scripted
# fetchers), so the hardship quick guide is read from the committed manual
# capture under data/sources (captured 05/09/2026; see source_note).


def build_policy() -> dict:
    print("Fetching Kiwibank Code of Banking Practice ...")
    kiwi_code = build_pdf_entries(
        fetch(KIWIBANK_CODE_URL),
        prefix="kiwibank_code",
        title="Kiwibank / NZBA Code of Banking Practice",
        meta="Kiwibank public Code of Banking Practice (April 2021)",
        source_note=(
            "Kiwibank public Code of Banking Practice, Kiwibank legal documents: "
            f"https://www.kiwibank.co.nz/documents/code-banking-practice/ "
            f"(PDF {KIWIBANK_CODE_URL}; fetched {FETCHED})."
        ),
    )
    print(f"  {len(kiwi_code)} kiwibank_code entries")

    print("Fetching Kiwibank financial hardship guidance ...")
    kiwi_hardship = build_html_entries(
        KIWIBANK_HARDSHIP_URL,
        prefix="kiwibank_hardship",
        title="Kiwibank financial hardship and assistance",
        meta="Kiwibank public customer guidance",
        source_note=(
            "Kiwibank, 'Understand financial hardship & assistance': "
            f"{KIWIBANK_HARDSHIP_URL} (fetched {FETCHED})."
        ),
    )
    print(f"  {len(kiwi_hardship)} kiwibank_hardship entries")

    print("Fetching Kiwibank General Terms and Conditions ...")
    kiwi_general = build_pdf_entries(
        fetch(KIWIBANK_GENERAL_TERMS_URL),
        prefix="kiwibank_general_terms",
        title="Kiwibank General Terms and Conditions",
        meta="Kiwibank current public contract terms (May 2026)",
        source_note=(
            "Kiwibank General Terms and Conditions, Kiwibank legal documents: "
            "https://www.kiwibank.co.nz/documents/general-terms-and-conditions/ "
            f"(PDF {KIWIBANK_GENERAL_TERMS_URL}; fetched {FETCHED})."
        ),
    )
    print(f"  {len(kiwi_general)} kiwibank_general_terms entries")

    print("Fetching Kiwibank Credit Card Terms and Conditions ...")
    kiwi_card = build_pdf_entries(
        fetch(KIWIBANK_CREDIT_CARD_URL),
        prefix="kiwibank_credit_card_terms",
        title="Kiwibank Credit Card Terms and Conditions",
        meta="Kiwibank current public contract terms (November 2025)",
        source_note=(
            "Kiwibank Credit Card Terms and Conditions, Kiwibank legal documents: "
            "https://www.kiwibank.co.nz/documents/credit-card-terms-and-conditions/ "
            f"(PDF {KIWIBANK_CREDIT_CARD_URL}; fetched {FETCHED})."
        ),
    )
    print(f"  {len(kiwi_card)} kiwibank_credit_card_terms entries")

    print("Fetching Kiwibank/Latitude Personal Loan Contract Terms ...")
    kiwi_personal_loan = build_pdf_entries(
        fetch(KIWIBANK_PERSONAL_LOAN_URL),
        prefix="kiwibank_latitude_personal_loan_terms",
        title="Kiwibank/Latitude Personal Loan Contract Terms and Conditions",
        meta="Kiwibank public partner-product contract terms (March 2021)",
        source_note=(
            "Kiwibank Personal Loan Contract Terms and Conditions (provided by "
            "Latitude Financial Services), Kiwibank legal documents: "
            "https://www.kiwibank.co.nz/documents/personal-loan-terms-and-conditions/ "
            f"(PDF {KIWIBANK_PERSONAL_LOAN_URL}; fetched {FETCHED})."
        ),
    )
    print(f"  {len(kiwi_personal_loan)} kiwibank_latitude_personal_loan entries")

    print("Fetching/parsing Credit Contracts and Consumer Finance Act 2003 ...")
    cccfa = build_cccfa_entries()
    print(f"  {len(cccfa)} cccfa_2003 entries (entire Act)")

    print("Fetching NZ Bankers Association Code of Banking Practice (PDF) ...")
    code = build_pdf_entries(
        fetch(NZBA_CODE_URL),
        prefix="nzba_code",
        title="NZ Bankers Association Code of Banking Practice",
        meta="NZ Bankers Association Code of Banking Practice",
        source_note=(
            "NZ Bankers Association Code of Banking Practice, nzba.org.nz: "
            f"{NZBA_CODE_URL} (fetched {FETCHED})."
        ),
    )
    print(f"  {len(code)} nzba_code entries")

    print("Fetching Consumer Protection CCCFA guidance page ...")
    cp = build_consumer_protection_entries()
    print(f"  {len(cp)} consumer_protection entries")

    print("Fetching Commerce Commission new-lender tip sheet (PDF) ...")
    comcom = build_pdf_entries(
        fetch(COMCOM_URL),
        prefix="comcom_lender",
        title="Things to think about now that you are a lender (Commerce Commission)",
        meta="Commerce Commission lender guidance (tip sheet, December 2018)",
        source_note=(
            "Commerce Commission, 'Things to think about now that you are a "
            f"lender' tip sheet: {COMCOM_URL} (fetched {FETCHED}; Crown copyright)."
        ),
    )
    print(f"  {len(comcom)} comcom_lender entries")

    # Banking Ombudsman hardship quick guide: literal text (site blocks bots).
    bankomb = [
        {
            "id": f"bankomb_hardship#{i + 1}",
            "title": f"Hardship and financial difficulty (part {i + 1})",
            "meta": "Banking Ombudsman quick guide - hardship and financial difficulty (updated December 2024)",
            "source_note": (
                "Banking Ombudsman Scheme quick guide 'Hardship and financial "
                f"difficulty': {BANKOMB_URL} (captured {FETCHED} via manual web "
                "fetch; the site blocks automated requests)."
            ),
            "text": text,
        }
        for i, text in enumerate(
            chunk_units(
                [p.strip() for p in read_manual_source(BANKOMB_SOURCE).split("\n\n")],
                CHUNK_TARGET,
            )
        )
    ]
    print(f"  {len(bankomb)} bankomb_hardship entries (literal, bot-blocked site)")

    policy = (
        kiwi_code
        + kiwi_hardship
        + kiwi_general
        + kiwi_card
        + kiwi_personal_loan
        + cccfa
        + code
        + cp
        + comcom
        + bankomb
    )
    print(f"Total policy corpus: {len(policy)} entries")
    return {
        "description": (
            "Lending policy library (environment data) for the lending_officer "
            "profile. It combines Kiwibank public guidance and contract terms "
            "(the Kiwibank/NZBA Code of Banking Practice, financial hardship "
            "guidance, General Terms and Conditions, Credit Card Terms and "
            "Conditions and Latitude Personal Loan Contract Terms) with the "
            "ENTIRE Credit Contracts and Consumer Finance Act 2003, Consumer "
            "Protection (MBIE) CCCFA guidance, Commerce Commission lender "
            "guidance and the Banking Ombudsman hardship quick guide. Kiwibank "
            "source entries are labelled as public guidance or contract terms; "
            "they are not internal credit-approval policies. Crown copyright / "
            "NZBA / Kiwibank material is used as a research simulation corpus. "
            "Entries are "
            "chunked (~2,000 characters); each carries id, title, meta, "
            "source_note (source + URL + fetched date) and text. "
            "lookup_lending_policy retrieves the top entries by lexical search."
        ),
        "policy": policy,
    }


# ---------------------------------------------------------------------------
# Hand-authored case data (committed source of truth) + build-time validation
# ---------------------------------------------------------------------------


def load_json(name: str) -> dict:
    """One of the committed, hand-authored case-data files (data/<name>).
    These are the single source of truth - the builder no longer
    generates them; it only reads them."""
    path = DATA_DIR / name
    if not path.exists():
        raise SystemExit(
            f"Missing data/{name}. The customer, credit-report and application "
            "data files are hand-authored commits; they are not generated. "
            "Restore them from git."
        )
    return json.loads(path.read_text())


# Internal-consistency targets for the itemised statement lines in
# data/customers.json (monthly in/out totals, NZD) so the check below fails
# loudly if they drift apart:
# - Deliberate differences from the pre-itemisation aggregates:
# - Ana Leota 2026-05: out $2,513.90 (was $2,113.90) - the $400.00 transfer
#   to savings on 24/05 is a real statement debit (her held statement .md shows it) and
#   now itemised, so it counts in the month's outflow (it was excluded
#   before).
# - Jordan Price 2026-04: a NEW partial month (data to 15/04) so the feed
#   covers the situation's April start date; totals are the items' own sums.
EXPECTED_MONTH_TOTALS = {
    ("CU-7W3F4MF", "38-9012-3456789-01"): {
        "2026-05": (3902.40, 3410.20),
        "2026-06": (3900.00, 3388.45),
        "2026-07": (3900.00, 1890.00),
    },
    ("CU-VD9D7T2", "38-9012-7744110-02"): {
        "2026-02": (3382.00, 3218.10),
        "2026-03": (3382.00, 3204.55),
        "2026-04": (3382.00, 3215.30),
        "2026-05": (2214.80, 2513.90),  # includes the $400 savings transfer
    },
    ("CU-YFLJKHY", "38-9012-8830455-03"): {
        "2026-01": (4400.00, 4361.80),
        "2026-02": (4400.00, 4402.35),
        "2026-03": (4400.00, 4377.60),
        "2026-04": (4400.00, 3027.75),  # new partial month
    },
    ("CU-U4DR4QQ", "38-9012-2210678-04"): {
        "2026-03": (1532.00, 1689.55),
        "2026-04": (1532.00, 1655.10),
        "2026-05": (1532.00, 1671.90),
        "2026-06": (766.00, 640.30),
    },
    ("CU-4YQNT7R", "38-9012-9933412-05"): {
        "2026-05": (6712.40, 6405.10),
        "2026-06": (4908.75, 5010.60),
        "2026-07": (5845.00, 4620.45),  # includes $1,500 post-decline card payment
    },
}


def check_transaction_totals() -> None:
    """Build-time consistency check against the hand-authored data: every
    itemised month in ``data/customers.json`` must sum to its expected
    aggregate (EXPECTED_MONTH_TOTALS), item dates must sit inside their
    month, and months must carry a plausible number of items. Raises
    SystemExit listing all problems, so an inconsistent dataset fails
    loudly before any output is written."""
    problems = []
    customers = load_json("customers.json")["customers"]
    for customer in customers:
        for account in customer.get("accounts", []):
            key = (customer["id"], account.get("number", ""))
            expected = EXPECTED_MONTH_TOTALS.get(key, {})
            on_file = {t["month"] for t in account.get("transactions", [])}
            for month in sorted(set(expected) - on_file):
                problems.append(f"{key[0]} {account['number']} {month}: month missing")
            for t in account.get("transactions", []):
                month = t["month"]
                items = t.get("items", [])
                tin = round(sum(i["amount"] for i in items if i["amount"] > 0), 2)
                tout = round(-sum(i["amount"] for i in items if i["amount"] < 0), 2)
                if not 5 <= len(items) <= 15:
                    problems.append(
                        f"{key[0]} {account['number']} {month}: "
                        f"{len(items)} items (expected 5-15)"
                    )
                if month in expected:
                    want_in, want_out = expected[month]
                    if (tin, tout) != (want_in, want_out):
                        problems.append(
                            f"{key[0]} {account['number']} {month}: items sum to "
                            f"in {tin}/out {tout}, expected {want_in}/{want_out}"
                        )
                for item in items:
                    if not item["date"].endswith(f"{month[5:7]}/{month[:4]}"):
                        problems.append(
                            f"{key[0]} {account['number']} {month}: item date "
                            f"{item['date']} falls outside the month"
                        )
    if problems:
        raise SystemExit(
            "Itemised transactions inconsistent:\n  " + "\n  ".join(problems)
        )
    n_months = sum(
        len(a.get("transactions", [])) for c in customers for a in c.get("accounts", [])
    )
    n_items = sum(
        len(t.get("items", []))
        for c in customers
        for a in c.get("accounts", [])
        for t in a.get("transactions", [])
    )
    print(
        f"Transaction check: {n_items} itemised lines across {n_months} "
        "months sum to the expected monthly aggregates."
    )


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Customer-held documents as Markdown (.md, in data/documents/ - attachable
# via the secure-messaging channel and read by the shared read_document tool)
# ---------------------------------------------------------------------------
# Interactive customers hold documents they can attach when the lending officer
# asks (see the ``documents`` lists in situations.json). Numbers are consistent
# with data/customers.json and data/applications.json.


CUSTOMER_DOCUMENTS = {
    "documents/ana_redundancy_letter.md": ("Redundancy", "07/05/2026"),
    "documents/ana_work_and_income_receipt.md": (
        "Application date",
        "11 May 2026",
    ),
    "documents/ana_bank_statement.md": ("Kiwibank", "1 April - 27 May 2026"),
    "documents/hine_benefit_letter.md": (
        "Weekly amount",
        "$383.00",
        "05/06/2026",
    ),
    "documents/tomas_contract_letters.md": ("T. Baker", "2026"),
    "documents/tomas_tax_summary.md": (
        "Net income after tax",
        "24 July 2026",
    ),
    "documents/tomas_card_statement_and_plan.md": (
        "Closing statement balance",
        "$6,418.40",
    ),
}


def check_customer_documents() -> None:
    """The customer-held documents are committed Markdown files in
    ``data/documents/`` (hand-edited directly, no PDF pipeline). Validate
    presence + the key strings that the situations depend on."""
    missing = [n for n in CUSTOMER_DOCUMENTS if not (DATA_DIR / n).exists()]
    if missing:
        raise SystemExit(
            f"missing customer documents (they are committed in data/documents/): {missing}. "
            "Restore with: git checkout -- profiles/lending_officer/data/documents/"
        )
    for name, expectations in CUSTOMER_DOCUMENTS.items():
        text = (DATA_DIR / name).read_text(encoding="utf-8")
        normalized = " ".join(text.split()).casefold()
        for expected in expectations:
            expected_normalized = " ".join(expected.split()).casefold()
            if expected_normalized not in normalized:
                raise SystemExit(f"{name} is missing expected content: {expected!r}")
    print(
        f"Customer documents: {len(CUSTOMER_DOCUMENTS)} tray files present, "
        "content checks pass."
    )


def check_case_consistency() -> None:
    """Check cross-file arithmetic, dates and demographics used by the cases."""
    applications = {
        row["id"]: row for row in load_json("applications.json")["applications"]
    }
    reports = {
        row["customer_id"]: row for row in load_json("credit_reports.json")["reports"]
    }
    customers = {row["id"]: row for row in load_json("customers.json")["customers"]}
    situation_rows = {
        row["id"]: row
        for row in json.loads((PROFILE_DIR / "situations.json").read_text())[
            "situations"
        ]
    }
    problems = []

    hine = reports["CU-U4DR4QQ"]["facilities"][0]
    hine_arrears = applications["OA-T7JAGEZ"]["arrears"]["total"]
    expected_hine_arrears = 3 * float(hine["repayment"]) + 3 * 30
    if abs(hine_arrears - expected_hine_arrears) > 0.01:
        problems.append(
            f"Hine arrears ${hine_arrears:.2f} != 3 payments + 3 fees "
            f"${expected_hine_arrears:.2f}"
        )
    first_missed = applications["OA-T7JAGEZ"]["missed_payments"][0]
    first_missed_date = date(*(int(part) for part in first_missed.split("/")[::-1]))
    hine_run_date = date.fromisoformat(
        situation_rows["overdue_account_collection"]["simulation_start_date"]
    )
    hine_arrears_age = (hine_run_date - first_missed_date).days
    if (
        hine_arrears_age != 113
        or "113 days old"
        not in (applications["OA-T7JAGEZ"]["decision_context"]["manager_note"])
    ):
        problems.append(
            "Hine's first missed payment, simulation date and stated arrears age "
            "must reconcile to 113 days"
        )

    jordan_debt = sum(float(f["balance"]) for f in reports["CU-YFLJKHY"]["facilities"])
    if abs(jordan_debt - 9150.0) > 0.01:
        problems.append(f"Jordan revolving debt ${jordan_debt:.2f} != $9,150.00")

    original_tomas_card = 7918.40
    current_tomas_card = next(
        float(f["balance"])
        for f in reports["CU-4YQNT7R"]["facilities"]
        if f["type"] == "credit_card"
    )
    if abs(original_tomas_card - current_tomas_card - 1500.0) > 0.01:
        problems.append(
            "Tomas original card balance minus the post-decline payment does not "
            "equal the refreshed balance"
        )

    ana = next(
        facility
        for facility in customers["CU-VD9D7T2"]["facilities"]
        if facility["type"] == "secured_vehicle_loan"
    )
    term_months = int(ana["term"].split()[0])
    monthly_rate = 0.139 / 12
    expected_ana_payment = (
        float(ana["original"])
        * monthly_rate
        * (1 + monthly_rate) ** term_months
        / ((1 + monthly_rate) ** term_months - 1)
    )
    if abs(expected_ana_payment - float(ana["repayment"])) > 0.50:
        problems.append(
            f"Ana recorded payment ${ana['repayment']:.2f} is not within $0.50 "
            f"of amortised ${expected_ana_payment:.2f}"
        )
    payments_made = int(ana["payments_made"])
    missed_count = len(ana.get("missed") or [])
    if payments_made + missed_count != 15:
        problems.append(
            "Ana ledger must record 15 due payments at the May 2026 review date: "
            f"{payments_made} made + {missed_count} missed"
        )
    expected_ana_balance = (
        float(ana["original"]) * (1 + monthly_rate) ** payments_made
        - float(ana["repayment"])
        * ((1 + monthly_rate) ** payments_made - 1)
        / monthly_rate
    )
    if abs(expected_ana_balance - float(ana["balance"])) > 1.00:
        problems.append(
            f"Ana balance ${ana['balance']:.2f} does not reconcile to roughly "
            f"${expected_ana_balance:.2f} after {payments_made} payments"
        )

    ana_application = applications["AC-GNHMPMS"]
    expected_savings = float(ana_application["hardship_request"]["savings"])
    savings_balances = [
        float(account["balance"])
        for account in customers["CU-VD9D7T2"]["accounts"]
        if account.get("type") == "savings"
    ]
    case_savings = float(
        (customers["CU-VD9D7T2"].get("case") or {})
        .get("hardship_request", {})
        .get("savings", -1)
    )
    if (
        savings_balances != [expected_savings]
        or abs(case_savings - expected_savings) > 0.01
    ):
        problems.append("Ana's application, savings account and case savings differ")
    if (
        missed_count != 2
        or "two consecutive payments"
        not in (ana_application["decision_context"]["current_position"])
    ):
        problems.append("Ana's request context must record exactly two missed payments")

    mark_customer_card = next(
        facility
        for facility in customers["CU-7W3F4MF"]["facilities"]
        if "4XXX-1187" in facility.get("name", "")
    )
    mark_report_card = next(
        facility
        for facility in reports["CU-7W3F4MF"]["facilities"]
        if facility.get("ref") == "4XXX-1187"
    )
    mark_context = applications["LN-BBC2CNF"]["decision_context"]["past_credit_context"]
    if not (
        mark_customer_card["closed"] == "11/2025"
        and mark_report_card["status"] == "closed 11/2025"
        and "November 2025" in mark_context
        and "May-June 2025" in mark_context
    ):
        problems.append("Mark's old-card closure and missed-payment dates differ")

    tomas_card = next(
        facility
        for facility in customers["CU-4YQNT7R"]["facilities"]
        if facility.get("type") == "credit_card"
    )
    observed_card_payments = {
        abs(float(item["amount"]))
        for account in customers["CU-4YQNT7R"]["accounts"]
        for month in account.get("transactions", [])
        for item in month.get("items", [])
        if item.get("description") == "AUTOPAY - KIWIBANK CREDIT CARD"
    }
    if observed_card_payments != {260.0} or float(tomas_card["min_payment"]) != 260.0:
        problems.append(
            "Tomas's recorded card minimum must match the observed $260 autopayment"
        )

    expected_ages = {
        "personal_loan_application": ("CU-7W3F4MF", 34),
        "declined_applicant_chat": ("CU-4YQNT7R", 45),
    }
    for situation_id, (customer_id, expected_age) in expected_ages.items():
        customer = customers[customer_id]
        dob = date(*(int(part) for part in customer["dob"].split("/")[::-1]))
        run_date = date.fromisoformat(
            situation_rows[situation_id]["simulation_start_date"]
        )
        age = (
            run_date.year
            - dob.year
            - ((run_date.month, run_date.day) < (dob.month, dob.day))
        )
        if age != expected_age:
            problems.append(
                f"{situation_id}: {customer['name']} is age {age}, not stated "
                f"{expected_age}"
            )

    simulation_dates = sorted(
        date.fromisoformat(row["simulation_start_date"])
        for row in situation_rows.values()
    )
    if simulation_dates[0] < date(2026, 6, 15):
        problems.append(
            "all simulations must start on or after 15 June 2026; "
            f"earliest is {simulation_dates[0].isoformat()}"
        )
    if simulation_dates[-1] > date(2026, 12, 31):
        problems.append(
            "all simulations must remain within 2026; "
            f"latest is {simulation_dates[-1].isoformat()}"
        )

    if problems:
        raise SystemExit("Case arithmetic inconsistent:\n  " + "\n  ".join(problems))
    print(
        "Case consistency: arrears, loan ledger, savings, revolving debt, card "
        "payment, 2026 simulation dates and stated ages reconcile."
    )


EXPECTED_SITUATION_IDS = {
    "personal_loan_application",
    "hardship_request",
    "credit_limit_increase",
    "overdue_account_collection",
    "declined_applicant_chat",
}


def check_situation_contract() -> None:
    """Validate cross-file situation, application, customer and rubric links."""
    payload = json.loads((PROFILE_DIR / "situations.json").read_text())
    applications = {
        row["id"]: row for row in load_json("applications.json")["applications"]
    }
    customers = {row["id"]: row for row in load_json("customers.json")["customers"]}
    situations = payload.get("situations", [])
    case_type_ids = {row["id"] for row in payload.get("case_types", [])}
    situation_ids = [row.get("id") for row in situations]
    problems = []

    if set(situation_ids) != EXPECTED_SITUATION_IDS or len(situation_ids) != len(
        EXPECTED_SITUATION_IDS
    ):
        problems.append(
            f"situation ids {situation_ids!r} do not match the expected five "
            f"unique ids {sorted(EXPECTED_SITUATION_IDS)!r}"
        )

    for situation in situations:
        sid = situation.get("id", "<missing>")
        case_type = situation.get("case_type")
        if case_type not in case_type_ids:
            problems.append(f"{sid}: unknown case_type {case_type!r}")
        brief = situation.get("brief", "")
        refs = [ref for ref in applications if ref in brief]
        if len(refs) != 1:
            problems.append(
                f"{sid}: brief must name exactly one known application reference; "
                f"found {refs!r}"
            )
        else:
            application = applications[refs[0]]
            customer_id = application.get("customer_id")
            if customer_id not in customers:
                problems.append(
                    f"{sid}: application {refs[0]} references unknown customer "
                    f"{customer_id!r}"
                )

        if situation.get("type") == "interactive":
            interlocutor = situation.get("interlocutor") or {}
            if not interlocutor.get("name"):
                problems.append(f"{sid}: interactive situation has no interlocutor")
            for document in interlocutor.get("documents") or []:
                path = document.get("file", "")
                if not (DATA_DIR / path).is_file():
                    problems.append(f"{sid}: missing interlocutor document {path}")

        rubric = situation.get("rubric") or []
        rubric_ids = [item.get("id") for item in rubric]
        if len(rubric_ids) != len(set(rubric_ids)):
            problems.append(f"{sid}: rubric ids are not unique")
        for item in rubric:
            item_id = item.get("id", "<missing>")
            if not item.get("question") or not item.get("criteria"):
                problems.append(f"{sid}/{item_id}: question and criteria are required")
            options = item.get("options")
            if item.get("type") == "multichoice":
                if not options or len(options) != len(set(options)):
                    problems.append(
                        f"{sid}/{item_id}: multichoice options must be non-empty and unique"
                    )
            elif options is not None:
                problems.append(
                    f"{sid}/{item_id}: only multichoice rubric items may define options"
                )
        terminate_tools = (situation.get("terminate") or {}).get("tools") or []
        if terminate_tools != ["close_item"]:
            problems.append(
                f"{sid}: lending situations must terminate on close_item, got "
                f"{terminate_tools!r}"
            )

    if problems:
        raise SystemExit("Situation contract inconsistent:\n  " + "\n  ".join(problems))
    print(
        f"Situation contract: {len(situations)} situations, "
        f"{sum(len(s.get('rubric', [])) for s in situations)} rubric items, and all "
        "application/customer/document links are consistent."
    )


def main(argv: list[str] | None = None) -> None:
    """Build the policy corpus while leaving hand-authored fixtures untouched."""
    parser = argparse.ArgumentParser(
        description="Build the lending_officer profile's policy corpus."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the hand-authored fixtures and print policy stats without writing",
    )
    args = parser.parse_args(argv)

    check_transaction_totals()
    check_customer_documents()
    check_case_consistency()
    check_situation_contract()
    policy = build_policy()

    if args.dry_run:
        print("Dry run: no files written.")
        for entry in policy["policy"][:5]:
            print(f"  [{entry['id']}] {entry['title']}\n      {entry['text'][:120]}")
        return

    output_path = DATA_DIR / "policy.json"
    write_json_file(output_path, policy)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
