#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "beautifulsoup4>=4.12",
#   "requests>=2.31",
# ]
# ///
"""
Build the welfare profile's deskfile database from the public Map deskfile.

The deskfile is Work and Income's reference archive under
``workandincome.govt.nz/map/deskfile/`` - rate tables (current and
historical), area codes, calendars, income abatement thresholds and other
lookup tables. It is deliberately kept OUT of the semantic policy corpus
(``build_welfare_policy_db.py``): the tables are large and numeric, and
their text is a poor fit for embedding retrieval. Instead this script
builds a separate keyword-searchable database:

- every deskfile page is walked (from the deskfile index, through the
  profile's local HTTP cache under ``.cache/``) and parsed into one
  document: title, section path, URL, keyword tokens and a markdown
  rendering of its tables and text;
- the docs are written to ``data/deskfile.json``;
- at run time the agent uses ``search_deskfile`` (fuzzy keyword search over
  titles, keywords and text) to find candidate tables, then
  ``read_deskfile`` to retrieve the exact table.

No embeddings: the deskfile is looked up by name/keyword, the way a person
finds a table in a manual, and then read exactly.

Usage:
    uv run profiles/welfare/build_deskfile_db.py            # fetch + build
    uv run profiles/welfare/build_deskfile_db.py --dry-run  # stats only
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy_store import write_json
from source_cache import cache_path as source_cache_path, cached_file

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = DATA_DIR / "deskfile.json"

# Local HTTP page cache (gitignored, disposable), nested to mirror the site
# structure (see build_welfare_policy_db.py) and shared with the policy
# builder so a page fetched once is never downloaded twice.
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

SITE_ROOT = "https://www.workandincome.govt.nz"
DESKFILE_HOME = f"{SITE_ROOT}/map/deskfile/index.html"

# Paths below /map/deskfile/ that are site furniture rather than tables.
EXCLUDE_FRAGMENTS = ("/map/about/", "/map/sitemap", "/map/changes/")


def clean(s: str) -> str:
    s = s.replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def cache_path(url: str) -> Path:
    """Return the shared URL-shaped cache path for a source page."""
    return source_cache_path(CACHE_DIR, url)


def fetch(url: str) -> str:
    """GET through the shared profile-local source cache."""
    return cached_file(url, cache_path(url), text=True, timeout=60).read_text(
        encoding="utf-8"
    )


def slug_for(path: str) -> str:
    """Readable document slug from a path: the leading ``map`` segment and a
    trailing ``index.html`` (or any ``.html`` filename suffix) are dropped."""
    segs = [s for s in path.split("/") if s][1:]
    if segs and segs[-1] == "index.html":
        segs = segs[:-1]
    elif segs and segs[-1].endswith(".html"):
        segs[-1] = segs[-1][: -len(".html")]
    return "-".join(segs) or "index"


def fetch_deskfile_pages() -> dict[str, str]:
    """Walk the deskfile from its index and return ``{slug: url}``."""
    pages: dict[str, str] = {}
    seen: set[str] = set()
    queued: set[str] = set()
    queue = [DESKFILE_HOME]
    queued.add(DESKFILE_HOME)
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        path = urlparse(url).path
        if not path.startswith("/map/deskfile"):
            continue
        if any(f in path for f in EXCLUDE_FRAGMENTS):
            continue
        base = slug_for(path)
        slug, n = base, 1
        while slug in pages:
            n += 1
            slug = f"{base}-{n}"
        try:
            html = fetch(url)
        except Exception as exc:  # keep walking: one bad page is not fatal
            print(f"    skipped {url}: {exc}", flush=True)
            continue
        pages[slug] = url
        if len(pages) % 100 == 0:
            print(
                f"    ... {len(pages)} pages walked, {len(queue)} queued",
                flush=True,
            )
        soup = BeautifulSoup(html, "html.parser")
        for a in soup.find_all("a", href=True):
            link = urlparse(a["href"]).path
            if link.startswith("/map/deskfile") and (
                link.endswith(".html") or link.endswith("/")
            ):
                full = f"{SITE_ROOT}{link}"
                if full not in queued:
                    queued.add(full)
                    queue.append(full)
    return pages


def _tokens(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if 2 < len(t) <= 30]


def table_to_markdown(table) -> str:
    """Render a <table> as a markdown table (pipes escaped)."""
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [
            clean(c.get_text(" ")).replace("|", "\\|")
            for c in tr.find_all(["th", "td"])
        ]
        if cells:
            rows.append(cells)
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = [
        "| " + " | ".join(rows[0]) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    out.extend("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join(out)


def parse_deskfile_page(html: str, slug: str, url: str) -> dict:
    """One deskfile page as a searchable/retrievable document."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find(id="content") or soup.body
    for nav_id in ("above-pannel", "second-level-nav"):
        nav = content.find(id=nav_id)
        if nav is not None:
            nav.decompose()
    for tag in content.find_all(["script", "style", "nav", "footer", "form"]):
        tag.decompose()

    h1 = content.find("h1")
    title = clean(h1.get_text()) if h1 else "Deskfile"

    parts: list[str] = []
    headings: set[str] = set()
    keywords: set[str] = set(_tokens(title))
    n_tables = 0
    for el in content.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if el.find_parent("table") is not None:
            continue  # already rendered inside the table markdown
        if el.name.startswith("h"):
            text = clean(el.get_text())
            if text:
                headings.add(text)
                keywords.update(_tokens(text))
                parts.append(f"## {text}")
        elif el.name == "li":
            parts.append("- " + clean(el.get_text()))
        elif el.name == "table":
            n_tables += 1
            first_row = el.find("tr")
            if first_row is not None:
                for cell in first_row.find_all(["th", "td"]):
                    keywords.update(_tokens(clean(cell.get_text(" "))))
            rendered = table_to_markdown(el)
            if rendered:
                parts.append(rendered)
        else:
            text = clean(el.get_text())
            if text:
                parts.append(text)

    # Section path and URL come from the real URL (the slug's hyphens are
    # ambiguous): the directories under /map/deskfile/ form the section.
    path_segs = [s for s in urlparse(url).path.split("/") if s]
    dirs = path_segs[2:-1]  # drop "map", "deskfile" and the filename
    section = " > ".join(s.replace("-", " ") for s in dirs) or "deskfile"

    return {
        "id": f"DF-{slug}",
        "title": title,
        "section": section,
        "url": url,
        "tables": n_tables,
        "keywords": sorted(keywords)[:400],
        "text": "\n\n".join(parts),
    }


def main(argv=None):
    args = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    args.add_argument("--dry-run", action="store_true", help="stats only, no write")
    opts = args.parse_args(argv)

    print("Walking Map deskfile pages ...")
    pages = fetch_deskfile_pages()
    print(f"  {len(pages)} deskfile pages in scope")

    documents = []
    for slug, url in sorted(pages.items()):
        documents.append(parse_deskfile_page(fetch(url), slug, url))
    total_tables = sum(d["tables"] for d in documents)
    print(f"  -> {len(documents)} documents, {total_tables} tables")
    print(
        "  sample:",
        documents[0]["id"],
        "|",
        documents[0]["title"],
        "|",
        documents[0]["section"],
    )

    if opts.dry_run:
        for doc in documents[:5]:
            print(
                f"  [{doc['id']}] ({doc['tables']} tables) {doc['title']}\n"
                f"      {doc['text'][:140]}"
            )
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_json(
        OUT_PATH,
        {
            "description": (
                "Map deskfile database for the welfare profile: Work and "
                "Income's public reference tables (rates current and "
                "historical, area codes, thresholds, calendars) from "
                "workandincome.govt.nz/map/deskfile. Crown copyright, "
                "licensed for re-use under CC BY 4.0: attribute the "
                "Ministry of Social Development (and indicate changes - "
                "the tables have been extracted and reformatted for "
                "retrieval). Point-in-time research snapshot, not the "
                "official or current tables. Kept separate from the "
                "semantic policy corpus because the tables are exact "
                "lookups, not policy text. Each document has a title, "
                "section path, keyword tokens and a markdown rendering of "
                "its tables; the agent searches by keyword with "
                "search_deskfile and reads the exact table with "
                "read_deskfile."
            ),
            "source": f"{SITE_ROOT}/map/deskfile/index.html",
            "documents": documents,
        },
    )
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
