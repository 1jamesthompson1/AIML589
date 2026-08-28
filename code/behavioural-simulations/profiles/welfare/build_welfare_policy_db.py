#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "beautifulsoup4>=4.12",
#   "requests>=2.31",
#   "sentence-transformers>=3.0",
# ]
# ///
"""
Build the welfare profile's policy environment data from public documents.

Sources (all public, Crown copyright; a research simulation corpus):

1. Social Security Act 2018 (NZ legislation, whole-act HTML from
   legislation.govt.nz) - the ENTIRE Act is parsed and chunked (one entry
   per section, split into ~CHUNK_TARGET-character chunks at sub-clause
   boundaries).
2. Work and Income public guidance pages (workandincome.govt.nz). The whole
   site is discovered from its official XML sitemap and restricted to the
   header-bar policy sections - work, eligibility (benefits and payments),
   on-a-benefit, housing and products - which includes the full "A-Z
   benefits and payments" list under /products/a-z-benefits/*. Non-policy
   sub-pages (location/provider directories, translated-language duplicates
   of English pages) are excluded. Every page is fetched and chunked, with
   no per-situation tagging: this is a plain RAG retrieval corpus. Each
   chunk carries a ``source`` label ('act' or 'wi_page').

Outputs (to ``profiles/welfare/data/``, the profile's environment-data
directory):

- ``policy.json`` - the chunked corpus consumed by the profile's tools. Each
  chunk carries a ``source`` label ('act' or 'wi_page') and its ``doc``,
  ``title`` and ``text``.
- ``policy_vectors.npz`` - a simple vector index: one embedding vector per
  chunk (same order as ``chunks``), enabling real semantic (RAG-style)
  retrieval in the ``lookup_msd_policy`` tool.

Embeddings use the sentence-transformers model named in ``EMBEDDING_MODEL``.
At eval time, semantic retrieval needs ``sentence-transformers`` installed
in the run environment (plus the model cached from Hugging Face). There is
no lexical fallback: ``lookup_msd_policy`` fails loudly if the vector index
or embedding model is unavailable.

Usage:
    uv run profiles/welfare/build_welfare_policy_db.py            # fetch + build
    uv run profiles/welfare/build_welfare_policy_db.py --dry-run  # stats only
"""

import argparse
import json
import re
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = DATA_DIR / "policy.json"
VECTORS_PATH = DATA_DIR / "policy_vectors.npz"

# Embedding model for the vector index. Keep in sync with the runtime
# lookup in profiles/welfare/__init__.py (lookup_msd_policy).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CACHE_DIR = Path("/tmp/opencode/welfare-policy-cache")

ACT_URL = "https://www.legislation.govt.nz/act/public/2018/0032/latest/whole.html"

# The Work and Income site is fetched comprehensively from its official XML
# sitemap, restricted to the top-level sections that carry welfare policy for
# the public (see WI_SECTIONS). This includes the header-bar top-level pages
# (work, eligibility / "benefits and payments", on-a-benefit, housing) and the
# full "A-Z benefits and payments" list under /products/a-z-benefits/*.
WI_SITE_ROOT = "https://www.workandincome.govt.nz"
WI_SITEMAP = "https://www.workandincome.govt.nz/sitemap.xml"

# Top-level site sections that hold citizen-facing welfare policy. Anything in
# the header bar that lands here is included (plus every sub-page of these
# sections found in the sitemap, e.g. the A-Z benefit pages).
WI_SECTIONS = ("eligibility", "products", "on-a-benefit", "work", "housing")

# Sub-path fragments excluded as non-policy noise: location/provider
# directories and translated-language duplicates of English pages. The A-Z
# *index* page is listed but each individual benefit page is fetched, so the
# index itself adds nothing beyond a list of links.
WI_EXCLUDE_FRAGMENTS = (
    "/glasses-suppliers/",  # per-region supplier directories
    "/employment-services-provider-list",  # provider directory listing
    "/traffic-light-system/languages-and-alternate-formats",
    "/traffic-light-system/cantonese",
    "/traffic-light-system/hindi",
    "/traffic-light-system/khmer",
    "/traffic-light-system/korean",
    "/traffic-light-system/mandarin",
    "/traffic-light-system/maori",
    "/traffic-light-system/punjabi",
    "/traffic-light-system/samoan",
    "/traffic-light-system/tongan",
    "/special-portability-pacific-countries-cimaori",
    "/special-portability-pacific-countries-samoan",
    "/special-portability-pacific-countries-tongan",
)


def fetch_wi_pages() -> dict[str, str]:
    """Discover all public policy pages from the Work and Income sitemap.

    Returns ``{slug: url}`` for every page under the WI_SECTIONS top-level
    sections (minus WI_EXCLUDE_FRAGMENTS). Slugs are derived from the URL path
    and made unique by appending a counter when two paths collide.
    """
    idx = requests.get(WI_SITEMAP, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    idx.raise_for_status()
    sitemap_urls = re.findall(r"<loc>(.*?)</loc>", idx.text)
    pages: dict[str, str] = {}
    for sub in sitemap_urls:
        resp = requests.get(sub, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        for loc in re.findall(r"<loc>(.*?)</loc>", resp.text):
            loc = loc.strip()
            path = urlparse(loc).path
            if not path.startswith("/"):
                continue
            segs = [s for s in path.split("/") if s]
            if not segs or segs[0] not in WI_SECTIONS:
                continue
            if any(f in path for f in WI_EXCLUDE_FRAGMENTS):
                continue
            if path == "/products/a-z-benefits":
                continue  # index page only; each benefit is fetched individually
            base = segs[-1]
            slug, n = base, 1
            while slug in pages:
                n += 1
                slug = f"{base}-{n}"
            pages[slug] = loc
    return pages


CHUNK_TARGET = 700  # chars per accumulated chunk (tool retrieves raw text)


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
    text = resp.text
    path.write_text(text)
    return text


def parse_act_sections(html: str) -> list[dict]:
    """Top-level provisions of the Act as {number, title, paras}.

    Provisions render as <div class="prov"><h5 class="prov"><span class="label">
    N</span> Title</h5>...<p class="text">…</p>; sub-provisions carry their
    ((1), (a), …) label in an ancestor <span class="label">."""
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


# (no keyword filtering: the whole Act is kept, as a real policy library
# would hold the complete statute)


def make_chunks(
    doc: str, title: str, unit_texts: list[str], start_idx: int, *, source: str
):
    """Accumulate small units (paragraphs) into ~CHUNK_TARGET-sized chunks.

    ``source`` labels where the chunk came from ("act" or "wi_page", see
    source_groups in policy.json) so the lookup tool can tell the user whether
    retrieved text is from the statute or a public Work and Income page.
    """
    chunks = []
    buf, size = [], 0
    for unit in unit_texts:
        if not unit:
            continue
        buf.append(unit)
        size += len(unit)
        if size >= CHUNK_TARGET:
            chunks.append((doc, title, " ".join(buf)))
            buf, size = [], 0
    if buf:
        chunks.append((doc, title, " ".join(buf)))
    out = []
    for i, (_, t, text) in enumerate(chunks):
        out.append(
            {
                "id": f"{doc}#{start_idx + i}",
                "doc": doc,
                "title": t,
                "text": text,
                "source": source,
            }
        )
    return out


def build_act_chunks(sections: list[dict]):
    """Chunk every parsed section of the Act (no filtering)."""
    chunks, n = [], 0
    for sec in sections:
        body = " ".join(sec["paras"])
        if not body:
            continue
        doc = (
            f"SSA2018-s{sec['number']}"
            if sec["number"]
            else f"SSA2018-{sec['title'][:20]}"
        )
        title = f"{sec['number']} {sec['title']}".strip()
        chunks.extend(make_chunks(doc, title, sec["paras"], 0, source="act"))
        n += 1
    return chunks, n


def parse_wi_page(html: str):
    """Split a WI page's <main> content into (heading, [paragraph]) blocks."""
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


def build_wi_chunks(slug: str, html: str, *, source: str = "wi_page"):
    blocks = parse_wi_page(html)
    chunks = []
    for head, paras in blocks:
        units = make_chunks(
            f"WI-{slug}",
            head or "Work and Income guidance",
            paras,
            0,
            source=source,
        )
        chunks.extend(units)
    return chunks


def main(argv=None):
    args = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    args.add_argument("--dry-run", action="store_true", help="stats only, no write")
    opts = args.parse_args(argv)

    print("Fetching/parsing Social Security Act 2018 ...")
    sections = parse_act_sections(fetch(ACT_URL, "ssa2018"))
    print(f"  {len(sections)} provisions parsed")
    act_chunks, n_acts = build_act_chunks(sections)
    print(f"  {n_acts} sections -> {len(act_chunks)} chunks (entire Act)")

    print("Discovering Work and Income pages from sitemap ...")
    wi_pages = fetch_wi_pages()
    print(f"  {len(wi_pages)} public WI pages in scope")

    # Fetch and chunk every public page once to build the plain RAG corpus.
    wi_chunks = []
    for slug, url in sorted(wi_pages.items()):
        page_chunks = build_wi_chunks(slug, fetch(url, f"wi-{slug}"), source="wi_page")
        wi_chunks.extend(page_chunks)
    print(f"  -> {len(wi_chunks)} wi_page chunks")

    all_chunks = act_chunks + wi_chunks
    print(
        f"Total corpus: {len(all_chunks)} chunks "
        f"(Act {len(act_chunks)} + Work and Income {len(wi_chunks)})"
    )

    if opts.dry_run:
        for c in all_chunks[:10]:
            print(
                f"  [{c['id']}] ({c['source']}) {c['title']}\n      {c['text'][:120]}"
            )
        return

    # Embed the corpus for the vector index. Text passed to the model is the
    # same text the tool will embed queries against (title + body).
    print(f"Embedding with {EMBEDDING_MODEL} ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(
        [f"{c['title']} {c['text']}" for c in all_chunks],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    VECTORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(VECTORS_PATH, vectors=embeddings)
    print(f"Wrote {VECTORS_PATH} ({embeddings.shape[0]} x {embeddings.shape[1]})")

    from collections import Counter

    source_counts = Counter(c["source"] for c in all_chunks)

    OUT_PATH.write_text(
        json.dumps(
            {
                "description": (
                    "Policy library (environment data) for the welfare profile "
                    "built from public sources: the ENTIRE Social Security Act "
                    "2018 (legislation.govt.nz) and the broad Work and Income "
                    "public guidance corpus (workandincome.govt.nz) covering the "
                    "header-bar policy sections (work, eligibility, on-a-benefit, "
                    "housing, products) including the full A-Z benefits and "
                    "payments list. All Crown copyright. Each chunk carries a "
                    "``source`` label: 'act' (the statute) or 'wi_page' (a "
                    "public Work and Income page). Flat chunked RAG corpus; "
                    "lookup_msd_policy retrieves top chunks semantically (no "
                    "lexical fallback) using the "
                    f"companion vector index {VECTORS_PATH.name} (model: "
                    f"{EMBEDDING_MODEL})."
                ),
                "embedding_model": EMBEDDING_MODEL,
                "source_groups": {
                    "SSA2018": source_counts.get("act", 0),
                    "WI_PAGES": source_counts.get("wi_page", 0),
                },
                "chunks": all_chunks,
            },
            indent=2,
        )
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
