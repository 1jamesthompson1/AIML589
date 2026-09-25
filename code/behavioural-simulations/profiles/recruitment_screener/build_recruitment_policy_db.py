#!/usr/bin/env python3
"""Build the recruitment policy library from released Health NZ PDFs.

The OIA appendix contains three documents. PDF pages 1-22 are an older
Counties Manukau policy and are deliberately excluded. Pages 23-31 contain
the Health NZ National Office Recruitment and Selection Policy; pages 34-42
contain the substantive people-leader guide (pages 32-33 are front matter).
The National Office policy is split at its numbered clauses, and long
clauses and guide pages are split at sentence boundaries. Every chunk keeps
its document, scope, page and clause provenance. Existing public guidance
in policy.json is retained. Two
other National Office policies are extracted by numbered clause: the Interim
Code of Conduct and the Privacy Policy (document 1 only in its OIA bundle).

Usage from code/behavioural-simulations/:
    uv run python profiles/recruitment_screener/build_recruitment_policy_db.py
    uv run python profiles/recruitment_screener/build_recruitment_policy_db.py --pdf /path/to/Appendix-2.pdf
    uv run python profiles/recruitment_screener/build_recruitment_policy_db.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from bisect import bisect_right
from pathlib import Path

from pypdf import PdfReader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy_store import write_json
from source_cache import cached_file

PROFILE_DIR = Path(__file__).resolve().parent
POLICY_PATH = PROFILE_DIR / "data" / "policy.json"
CACHE_PATH = PROFILE_DIR / ".cache" / "healthnz_recruitment_appendix2.pdf"
SOURCE_URL = (
    "https://fyi.org.nz/request/29388/response/120008/attach/6/Appendix%202.pdf"
)
SOURCE_SHA256 = "244bb35effeac2a2d95be727a5a0bda081248e2a22b7c3c4dd5b8f7611e25c55"
CONDUCT_URL = "https://fyi.org.nz/request/24171/response/91587/attach/6/Te%20Whatu%20Ora%20Interim%20Code%20of%20Conduct.pdf"
CONDUCT_SHA256 = "f57fc9c8ef99a31a785f5551fa923565a8bc995cafccf2b614a8daa2b767b7b7"
PRIVACY_URL = "https://fyi.org.nz/request/20883/response/79423/attach/3/HNZ00005636%20documents.pdf"
PRIVACY_SHA256 = "1ad18c6935da87ad0d6c8c47c32190dc456f03e3af961c0d8e0ffd9e64a582c4"
CACHE_DIR = PROFILE_DIR / ".cache"
MAX_CHARS = 1300

POLICY_SECTIONS = {
    range(1, 8): "Purpose and application",
    range(8, 11): "Definitions and policy",
    range(11, 14): "Key principles and Māori workforce",
    range(14, 21): "Vacancy and position description",
    range(21, 23): "Advertising",
    range(23, 33): "Candidate care",
    range(33, 38): "Conflicts and vacancy notification",
    range(38, 42): "Selection panel",
    range(42, 45): "Shortlisting",
    range(45, 50): "Candidate assessment",
    range(50, 54): "Reference and pre-employment checks",
    range(54, 57): "Appointment recommendations",
    range(57, 60): "Roles and responsibilities",
}
GUIDE_SECTIONS = {
    34: "Before recruiting",
    35: "Position and vacancy types",
    36: "Advertising and agencies",
    37: "Shortlisting and phone screening",
    38: "Interview preparation",
    39: "Interview conduct and accommodation",
    40: "Talent pool and reference checks",
    41: "Reference checks and offers",
    42: "Offers and onboarding",
}


def source_pdf(path: Path | None) -> Path:
    if path is not None:
        return path
    return cached_file(SOURCE_URL, CACHE_PATH, sha256=SOURCE_SHA256)


def released_pdf(path: Path | None, url: str, filename: str, digest: str) -> Path:
    """Use a local override or a verified, disposable cached source."""
    return (
        path
        if path is not None
        else cached_file(url, CACHE_DIR / filename, sha256=digest)
    )


def checked_reader(path: Path, digest: str, pages: int) -> PdfReader:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != digest:
        raise ValueError(f"Unexpected PDF SHA256 for {path}: {actual}")
    reader = PdfReader(path)
    if len(reader.pages) != pages:
        raise ValueError(
            f"Expected {pages} PDF pages in {path}, found {len(reader.pages)}"
        )
    return reader


def clean_page(raw: str, page_number: int, *, policy: bool) -> str:
    text = raw.replace("Released under The Official Information Act 1982", "")
    if policy:
        # The numbered page header is a standalone line in the policy PDF.
        text = re.sub(r"^\s*" + str(page_number - 22) + r"\s*\n", "", text, count=1)
    return re.sub(r"\s+", " ", text).strip()


def split_long(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split at the last sentence or word boundary before the size limit."""
    pieces = []
    while len(text) > limit:
        boundary = max(text.rfind(". ", 0, limit), text.rfind("; ", 0, limit))
        if boundary >= limit // 2:
            boundary += 1
        else:
            boundary = text.rfind(" ", 0, limit)
        if boundary < 1:
            boundary = limit
        pieces.append(text[:boundary].strip())
        text = text[boundary:].strip()
    if text:
        pieces.append(text)
    return pieces


def policy_chunks(reader: PdfReader) -> list[dict]:
    pages = [
        clean_page(reader.pages[i].extract_text() or "", i + 1, policy=True)
        for i in range(22, 31)
    ]
    # The final page ends with related-document metadata rather than clauses.
    pages[-1] = pages[-1].split("Related Legislation", 1)[0].strip()
    offsets = []
    position = 0
    for page in pages:
        offsets.append(position)
        position += len(page) + 1
    whole = " ".join(pages)
    matches = list(re.finditer(r"(?<![A-Za-z0-9])(\d{1,2})\.\s+(?=[A-Z])", whole))
    numbers = [int(match.group(1)) for match in matches]
    if numbers != list(range(1, 60)):
        raise ValueError(f"Policy clause extraction changed: {numbers}")

    chunks = []
    for index, match in enumerate(matches):
        clause = numbers[index]
        end = matches[index + 1].start() if index + 1 < len(matches) else len(whole)
        body = whole[match.start() : end].strip()
        first_page = 23 + bisect_right(offsets, match.start()) - 1
        last_page = 23 + bisect_right(offsets, end - 1) - 1
        section = next(
            label for group, label in POLICY_SECTIONS.items() if clause in group
        )
        for part, piece in enumerate(split_long(body), start=1):
            chunks.append(
                {
                    "id": f"HNZ-NO-cl{clause:02d}-{part}",
                    "kind": "released_policy_chunk",
                    "title": f"Health NZ National Office recruitment: {section} (clause {clause})",
                    "text": piece,
                    "source_url": SOURCE_URL,
                    "source_page": f"PDF pp. {first_page}-{last_page}"
                    if first_page != last_page
                    else f"PDF p. {first_page}",
                    "source_clauses": str(clause),
                    "scope": "Health NZ National Office only; released July 2022 copy, current status unverified",
                    "source_note": "Health NZ OIA release; extracted and whitespace-normalised for retrieval.",
                }
            )
    return chunks


def guide_chunks(reader: PdfReader) -> list[dict]:
    chunks = []
    for page_number, section in GUIDE_SECTIONS.items():
        body = clean_page(
            reader.pages[page_number - 1].extract_text() or "",
            page_number,
            policy=False,
        )
        if not body:
            raise ValueError(f"Guide PDF page {page_number} has no text")
        for part, piece in enumerate(split_long(body), start=1):
            chunks.append(
                {
                    "id": f"HNZ-guide-p{page_number:02d}-{part}",
                    "kind": "released_guide_chunk",
                    "title": f"Health NZ people-leader recruitment guide: {section}",
                    "text": piece,
                    "source_url": SOURCE_URL,
                    "source_page": f"PDF p. {page_number}",
                    "scope": "Health NZ people-leader guide in March 2025 OIA release; current status unverified",
                    "source_note": "Health NZ OIA release; extracted and whitespace-normalised for retrieval. Guide suggestions are not binding policy.",
                }
            )
    return chunks


def numbered_policy_chunks(
    reader: PdfReader,
    *,
    title: str,
    prefix: str,
    url: str,
    page_count: int,
    clause_count: int,
) -> list[dict]:
    """Extract a released National Office policy without its OIA page furniture.

    Both policies start at PDF page 1. The privacy PDF includes other documents
    after page 9; ``page_count`` limits extraction to document 1.
    """
    pages = []
    for index in range(page_count):
        raw = reader.pages[index].extract_text() or ""
        raw = raw.replace("Released under the Official Information Act 1982", "")
        raw = re.sub(r"(?m)^\s*Document 1\s*$", "", raw)
        raw = re.sub(r"^\s*" + str(index + 1) + r"\s*\n", "", raw, count=1)
        pages.append(raw.strip())
    offsets = []
    position = 0
    for page in pages:
        offsets.append(position)
        position += len(page) + 1
    whole = "\n".join(pages).split("Related Policies and Procedures", 1)[0].strip()
    matches = list(re.finditer(r"(?m)^\s*(\d{1,2})\.\s+(?=[A-Z])", whole))
    numbers = [int(match.group(1)) for match in matches]
    if numbers != list(range(1, clause_count + 1)):
        raise ValueError(f"{title} clause extraction changed: {numbers}")

    chunks = []
    for index, match in enumerate(matches):
        clause = numbers[index]
        end = matches[index + 1].start() if index + 1 < len(matches) else len(whole)
        body = re.sub(r"\s+", " ", whole[match.start() : end]).strip()
        first_page = bisect_right(offsets, match.start())
        last_page = bisect_right(offsets, end - 1)
        for part, piece in enumerate(split_long(body), start=1):
            chunks.append(
                {
                    "id": f"{prefix}-cl{clause:02d}-{part}",
                    "kind": "released_policy_chunk",
                    "title": f"Health NZ National Office {title} (clause {clause})",
                    "text": piece,
                    "source_url": url,
                    "source_page": (
                        f"PDF pp. {first_page}-{last_page}"
                        if first_page != last_page
                        else f"PDF p. {first_page}"
                    ),
                    "source_clauses": str(clause),
                    "scope": "Health NZ National Office only; July 2022 policy, current status unverified",
                    "source_note": "Health NZ OIA release; extracted and whitespace-normalised for retrieval.",
                }
            )
    return chunks


def build(
    pdf_path: Path, conduct_path: Path, privacy_path: Path, *, check: bool
) -> None:
    reader = checked_reader(pdf_path, SOURCE_SHA256, 43)
    conduct = checked_reader(conduct_path, CONDUCT_SHA256, 7)
    privacy = checked_reader(privacy_path, PRIVACY_SHA256, 35)
    chunks = policy_chunks(reader) + guide_chunks(reader)
    chunks += numbered_policy_chunks(
        conduct,
        title="Interim Code of Conduct",
        prefix="HNZ-COC",
        url=CONDUCT_URL,
        page_count=7,
        clause_count=26,
    )
    chunks += numbered_policy_chunks(
        privacy,
        title="Privacy Policy",
        prefix="HNZ-PRIV",
        url=PRIVACY_URL,
        page_count=9,
        clause_count=40,
    )
    data = json.loads(POLICY_PATH.read_text())
    old_kinds = {
        "released_policy",
        "released_guide",
        "released_policy_chunk",
        "released_guide_chunk",
    }
    retained = [
        entry
        for entry in data["policy"]
        if entry["kind"] not in old_kinds and entry["kind"] != "simulation_procedure"
    ]
    data["policy"] = retained[:2] + chunks + retained[2:]
    data["source_pdf_sha256"] = SOURCE_SHA256
    data["source_pdf_sha256s"] = {
        "recruitment": SOURCE_SHA256,
        "code_of_conduct": CONDUCT_SHA256,
        "privacy_bundle": PRIVACY_SHA256,
    }
    data["description"] = (
        "Offline research snapshot for a simulated Health NZ National Office recruitment assistant. "
        "The March 2025 OIA appendix contributes full-text, scope-labelled chunks of a July 2022 "
        "National Office recruitment policy and a people-leader guide; older Counties Manukau pages are excluded. "
        "Released National Office Code of Conduct and Privacy Policy clauses are also included. "
        "Current policy status is unverified. Public guidance summaries are labelled separately. "
        "Health NZ source text was extracted and reformatted for retrieval."
    )
    output = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if check:
        if POLICY_PATH.read_text() != output:
            raise ValueError(
                "policy.json differs from the PDF-derived build; rerun without --check"
            )
        print(f"validated {len(chunks)} PDF chunks and policy.json (no write)")
    else:
        write_json(POLICY_PATH, data)
        print(
            f"wrote {len(chunks)} source chunks and {len(retained)} other entries to {POLICY_PATH}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdf", type=Path, help="use an existing copy of the released OIA PDF"
    )
    parser.add_argument(
        "--conduct-pdf", type=Path, help="use a saved Code of Conduct PDF"
    )
    parser.add_argument(
        "--privacy-pdf", type=Path, help="use a saved privacy-policy bundle PDF"
    )
    parser.add_argument(
        "--check", action="store_true", help="validate extraction without writing"
    )
    args = parser.parse_args()
    build(
        source_pdf(args.pdf),
        released_pdf(
            args.conduct_pdf,
            CONDUCT_URL,
            "healthnz_code_of_conduct.pdf",
            CONDUCT_SHA256,
        ),
        released_pdf(
            args.privacy_pdf, PRIVACY_URL, "healthnz_privacy_bundle.pdf", PRIVACY_SHA256
        ),
        check=args.check,
    )
