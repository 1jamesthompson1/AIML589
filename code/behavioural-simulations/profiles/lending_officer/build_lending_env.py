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

Sources (public, NZ):

1. Credit Contracts and Consumer Finance Act 2003 - the ENTIRE Act is parsed
   from the whole-act HTML on legislation.govt.nz and chunked per section
   (~CHUNK_TARGET characters per chunk at sub-clause boundaries). This is the
   core responsible-lending law: lender responsibility principles (s 9C),
   affordability/suitability inquiries, unforeseen hardship (ss 55-57A) and
   the oppression/collections-conduct provisions.
2. NZ Bankers Association Code of Banking Practice (nzba.org.nz PDF) - the
   banks' fair-treatment commitments (responsible credit, hardship, dealings
   with customers in difficulty).
3. Consumer Protection (MBIE) CCCFA guidance page - plain-language summary of
   responsible lending duties and the hardship-application process/timeframes.
4. Commerce Commission "new lender" tip sheet (comcom.govt.nz PDF) -
   regulator guidance covering disclosure, hardship and conduct on default.
5. Banking Ombudsman hardship quick guide (bankomb.org.nz) - captured as
   literal text: the site blocks automated requests (HTTP 403), so the
   content was obtained via a manual web fetch (see source_note).

Outputs (to ``profiles/lending_officer/data/``, the profile's
environment-data directory; all files overwritten on each run, so the build
is idempotent):

- ``policy.json`` - the chunked lending-policy corpus consumed by the
  ``lookup_lending_policy`` tool. Each entry carries ``id``, ``title``,
  ``meta``, ``source_note`` (source + URL + fetched date) and ``text``.
- ``documents/*.md`` - the customer-held documents in the situations'
  document tray, as Markdown (read natively by the shared read_document
  tool).

The case data files ``customers.json``, ``credit_reports.json`` and
``applications.json`` are HAND-AUTHORED committed commits - the single
source of truth - and are NOT generated here. The builder reads
``customers.json`` for its build-time consistency check (the itemised
statement lines must sum to the expected monthly aggregates).

Usage:
    uv run profiles/lending_officer/build_lending_env.py            # build
    uv run profiles/lending_officer/build_lending_env.py --dry-run  # stats only
"""

import json
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from pypdf import PdfReader

DATA_DIR = Path(__file__).resolve().parent / "data"
ENV_SRC = Path(__file__).resolve().parent / "env_src"


def read_env_src(fname: str) -> str:
    """A captured source page (Banking Ombudsman blocks automated requests;
    captured manually and committed)."""
    return (ENV_SRC / "fetched" / fname).read_text()


CACHE_DIR = Path("/tmp/opencode/lending-env-cache")

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

CHUNK_TARGET = 2000  # characters per accumulated chunk


# ---------------------------------------------------------------------------
# Fetch + parse helpers
# ---------------------------------------------------------------------------


def fetch(url: str, name: str) -> bytes | str:
    """GET with an on-disk cache so re-runs don't hammer the sites."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if path.exists() and path.stat().st_size > 0:
        return path.read_bytes()
    resp = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return resp.content


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


def chunk_units(units: list[str], target: int = CHUNK_TARGET) -> list[str]:
    """Accumulate small text units (paragraphs/lines) into ~target-sized
    chunks."""
    chunks, buf, size = [], [], 0
    for unit in units:
        if not unit:
            continue
        buf.append(unit)
        size += len(unit)
        if size >= target:
            chunks.append(" ".join(buf))
            buf, size = [], 0
    if buf:
        chunks.append(" ".join(buf))
    return chunks


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
    html = fetch(CCCFA_URL, "cccfa2003.html")
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
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
        for i, text in enumerate(chunk_units(sec["paras"])):
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
    for i, text in enumerate(chunk_units(pdf_text(data))):
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
    html = fetch(CONSUMER_PROTECTION_URL, "consumer-protection.html")
    if isinstance(html, bytes):
        html = html.decode("utf-8", errors="replace")
    source_note = (
        "Consumer Protection (MBIE), 'Credit Contracts and Consumer Finance "
        f"Act' guidance page: {CONSUMER_PROTECTION_URL} (fetched {FETCHED}; "
        "Crown copyright)."
    )
    entries = []
    for head, paras in html_main_blocks(html):
        for i, text in enumerate(chunk_units(paras)):
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


# The Banking Ombudsman site blocks automated requests (HTTP 403 to scripted
# fetchers), so the hardship quick guide is held here as literal text,
# captured via a manual web fetch on 05/09/2026 (see source_note).


def build_policy() -> dict:
    print("Fetching/parsing Credit Contracts and Consumer Finance Act 2003 ...")
    cccfa = build_cccfa_entries()
    print(f"  {len(cccfa)} cccfa_2003 entries (entire Act)")

    print("Fetching NZ Bankers Association Code of Banking Practice (PDF) ...")
    code = build_pdf_entries(
        fetch(NZBA_CODE_URL, "nzba-code.pdf"),
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
        fetch(COMCOM_URL, "comcom-tip.pdf"),
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
                [p.strip() for p in read_env_src("bankomb_hardship.md").split("\n\n")]
            )
        )
    ]
    print(f"  {len(bankomb)} bankomb_hardship entries (literal, bot-blocked site)")

    policy = cccfa + code + cp + comcom + bankomb
    print(f"Total policy corpus: {len(policy)} entries")
    return {
        "description": (
            "Lending policy library (environment data) for the lending_officer "
            "profile, built from public sources: the ENTIRE Credit Contracts "
            "and Consumer Finance Act 2003 (legislation.govt.nz; source label "
            "cccfa_2003), the NZ Bankers Association Code of Banking Practice "
            "(nzba.org.nz), Consumer Protection (MBIE) CCCFA guidance, a "
            "Commerce Commission lender tip sheet (comcom.govt.nz) and the "
            "Banking Ombudsman hardship quick guide (bankomb.org.nz, captured "
            "via manual web fetch as the site blocks automated requests). "
            "Crown copyright / NZBA material used as a research simulation "
            "corpus. Entries are chunked (~2,000 characters); each carries "
            "id, title, meta, source_note (source + URL + fetched date) and "
            "text. lookup_lending_policy retrieves the top entries by lexical "
            "search."
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
        "2026-01": (3902.40, 3410.20),
        "2026-02": (3900.00, 3388.45),
        "2026-03": (3900.00, 1890.00),
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
        "2026-01": (6712.40, 6405.10),
        "2026-02": (4908.75, 5010.60),
        "2026-03": (5845.00, 3120.45),
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


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {path}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Customer-held documents as Markdown (.md, in data/documents/ - attachable
# via the secure-messaging channel and read by the shared read_document tool)
# ---------------------------------------------------------------------------
# Ana Leota (hardship, AC-GNHMPMS) and Tomas Baker (declined-application chat,
# LN-8YNJ6HL) hold documents they can attach when the lending officer asks (see
# the ``documents`` lists on their interlocutor records in situations.json).
# Numbers are consistent with data/customers.json and data/applications.json.


CUSTOMER_DOCUMENTS = {
    "documents/ana_redundancy_letter.md": ("Redundancy", "07/05/2026"),
    "documents/ana_bank_statement.md": ("Kiwibank", "1 April - 27 May 2026"),
    "documents/tomas_contract_letters.md": ("T. Baker", "2026"),
    "documents/tomas_tax_summary.md": ("No outstanding tax debt", "24/03/2026"),
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
    for name, (needle, date) in CUSTOMER_DOCUMENTS.items():
        text = (DATA_DIR / name).read_text()
        for needle2 in (needle,):
            if needle2 not in text:
                raise SystemExit(f"{name} is missing expected content: {needle2!r}")
    print("Customer documents: 4 tray files present, content checks pass.")
