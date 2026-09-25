#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "beautifulsoup4>=4.12",
#   "requests>=2.31",
#   "sentence-transformers>=3.0",
#   "transformers>=4.40",
# ]
# ///
"""
Build the welfare profile's policy environment data from public documents.

Sources (all public, Crown copyright; a research simulation corpus):

1. Social Security Act 2018 (NZ legislation, whole-act HTML from
   legislation.govt.nz) - the ENTIRE Act is parsed and chunked (one entry
   per section, split at sub-clause boundaries to fit the embedding window).
   Source label: 'act'.
2. Map - the Guide to Social Development Policy (workandincome.govt.nz/map):
   Work and Income's public operational policy and procedures, the public
   copy of the Ministry's Manuals and Procedures (MAP). Every Map page is
   discovered by walking the site from its home page and chunked - core
   income-support policy, main benefits, extra help, social housing,
   students, overseas, legislation (Acts, regulations, ministerial
   directions, welfare programmes). Source label: 'map_page'. The Map
   deskfile (rate and reference tables) is deliberately NOT part of this
   semantic corpus: it is built separately as a keyword-searchable lookup
   database by ``build_deskfile_db.py`` (``data/deskfile.json``, read via
   the ``search_deskfile`` / ``read_deskfile`` tools).
3. Work and Income public guidance pages (workandincome.govt.nz). The whole
   site is discovered from its official XML sitemap and restricted to the
   header-bar policy sections - work, eligibility (benefits and payments),
   on-a-benefit, housing and products - which includes the full "A-Z
   benefits and payments" list under /products/a-z-benefits/*. Non-policy
   sub-pages (location/provider directories, translated-language duplicates
   of English pages) are excluded. Source label: 'wi_page'.

Chunking is token-aware: a page whose text fits the embedding window becomes
one whole-page chunk; bigger pages are split at heading/paragraph boundaries
into chunks that fit the window (measured with the embedding model's own
tokenizer). Repeated blocks across pages (Map reuses shared sections, e.g.
"Tenant under a tenancy agreement" under every housing product) are kept
once, at their first page, and "List of amendments" pages are skipped - they
are amendment histories, not policy.

Chunks are ordered Act -> Map -> Work and Income website, so the corpus
lists the statute first, then operational policy, then the public guidance
site.

Outputs (to ``profiles/welfare/data/``, the profile's environment-data
directory):

- ``policy.json`` - the chunked corpus consumed by the profile's tools. Each
  chunk carries a ``source`` label ('act', 'map_page' or 'wi_page') and its
  ``doc``, ``title`` and ``text``; the file also records the embedding model
  and the query prefix the runtime must use.
- ``policy_vectors.npz`` - a simple vector index: one embedding vector per
  chunk (same order as ``chunks``), enabling real semantic (RAG-style)
  retrieval in the ``lookup_msd_policy`` tool.

Embeddings use the sentence-transformers model named in ``EMBEDDING_MODEL``
(BGE models want their instruction prefix on queries only - the documents
are embedded plain, and the runtime applies ``query_prefix`` from
``policy.json``). At eval time, semantic retrieval needs
``sentence-transformers`` installed in the run environment (plus the model
cached from Hugging Face). There is no lexical fallback:
``lookup_msd_policy`` fails loudly if the vector index or embedding model is
unavailable.

Usage:
    uv run profiles/welfare/build_welfare_policy_db.py            # fetch + build
    uv run profiles/welfare/build_welfare_policy_db.py --dry-run  # stats only
"""

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from policy_store import write_json
from source_cache import cache_path as source_cache_path, cached_file

DATA_DIR = Path(__file__).resolve().parent / "data"
OUT_PATH = DATA_DIR / "policy.json"
VECTORS_PATH = DATA_DIR / "policy_vectors.npz"

# Embedding model for the vector index. Keep in sync with the runtime
# lookup in profiles/welfare/__init__.py (lookup_msd_policy reads the model
# name and query prefix from policy.json).
#
# bge-small-en-v1.5: 384 dims, 512-token window (2x MiniLM), same tokenizer
# family and near-MiniLM CPU cost. BGE retrieval wants the instruction
# prefix on queries only; documents are embedded plain.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Chunking fits this many tokens per chunk (headroom under the model's
# 512-token window for the title and special tokens).
TOKEN_LIMIT = 500
# Local HTTP page cache (gitignored, disposable), nested to mirror the site
# structure: a page at /map/income-support/... is cached at
# .cache/www.workandincome.govt.nz/map/income-support/... so the tree
# resembles the site's own map and pages are easy to find by hand.
CACHE_DIR = Path(__file__).resolve().parent / ".cache"

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

# Map - the Guide to Social Development Policy: Work and Income's public
# operational policy and procedures, and the public copy of the Ministry's
# internal Manuals and Procedures (MAP). Discovery walks the site from its
# home page (Map's site map index lists only section index pages).
MAP_HOME = "https://www.workandincome.govt.nz/map/index.html"

# Map pages that are site furniture rather than policy: the about page and
# change log, the site map itself, the deskfile - a large reference archive
# of rate tables (including historical tables back to the 1990s) that would
# swamp the corpus; current rates already live in the public Work and
# Income pages source - and "List of amendments" pages, which are amendment
# histories rather than policy.
MAP_EXCLUDE_FRAGMENTS = (
    "/map/about/",
    "/map/sitemap",
    "/map/changes/",
    "/map/deskfile",
    "-list-of-amendments",
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


def fetch_map_pages() -> dict[str, str]:
    """Discover every Map (Guide to Social Development Policy) page by
    walking the Map site from its home page.

    Map's site map index only lists section index pages; each index page's
    second-level menu lists its topic pages, so the walk follows internal
    /map/ links (fetched through the shared cache). Returns ``{slug: url}``;
    slugs drop the leading ``map`` and any trailing ``index.html``, so a
    page becomes one readable document id (e.g.
    ``income-support-core-policy-reviews-and-appeals``).
    """
    pages: dict[str, str] = {}
    seen: set[str] = set()
    queued: set[str] = set()
    queue = [MAP_HOME]
    queued.add(MAP_HOME)
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        path = urlparse(url).path
        if any(f in path for f in MAP_EXCLUDE_FRAGMENTS):
            continue
        segs = [s for s in path.split("/") if s][1:]  # drop the "map" segment
        if segs and segs[-1] == "index.html":
            segs = segs[:-1]
        base = "-".join(segs) or "index"
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
            if link.startswith("/map/") and (
                link.endswith(".html") or link.endswith("/")
            ):
                full = f"{WI_SITE_ROOT}{link}"
                if full not in queued:
                    queued.add(full)
                    queue.append(full)
    return pages


def clean(s: str) -> str:
    s = s.replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def count_tokens(text: str, tokenizer) -> int:
    """Token count under the embedding model's own tokenizer."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def split_oversize(text: str, tokenizer, limit: int) -> list[str]:
    """Split a single unit that alone exceeds the limit, at sentence
    boundaries (paragraphs normally fit, so this is a last resort)."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts, buf = [], ""
    for sentence in sentences:
        candidate = f"{buf} {sentence}".strip()
        if buf and count_tokens(candidate, tokenizer) > limit:
            parts.append(buf)
            buf = sentence
        else:
            buf = candidate
    if buf:
        parts.append(buf)
    return parts or [text]


def block_key(head: str, paras: list[str]) -> str:
    """Normalised key for duplicate detection across pages."""
    return re.sub(r"\s+", " ", f"{head} {' '.join(paras)}").strip().lower()


def unique_blocks(
    blocks: list[tuple[str, list[str]]], seen_blocks: set[str], min_chars: int = 80
):
    """Keep each block once across the corpus: Map repeats shared sections
    (e.g. "Tenant under a tenancy agreement") under every product page.
    Short blocks are always kept - they are too small to be usefully
    deduplicated."""
    kept = []
    for head, paras in blocks:
        key = block_key(head, paras)
        if len(key) >= min_chars:
            if key in seen_blocks:
                continue
            seen_blocks.add(key)
        kept.append((head, paras))
    return kept


def cache_path(url: str) -> Path:
    """The cache file for a URL, nested to mirror the site structure: the
    host and URL path become directories, and a directory URL is stored as
    index.html (so the cache tree resembles the site map)."""
    return source_cache_path(CACHE_DIR, url)


def fetch(url: str) -> str:
    """GET with an on-disk, URL-shaped cache so re-runs don't hammer the
    sites and the cache tree mirrors the site map."""
    return cached_file(url, cache_path(url), text=True).read_text()


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
    doc: str,
    title: str,
    unit_texts: list[str],
    start_idx: int,
    *,
    source: str,
    tokenizer,
    token_limit: int = TOKEN_LIMIT,
):
    """Accumulate small units (paragraphs) into chunks that fit the embedding
    model's token window; units are kept whole where they fit.

    ``source`` labels where the chunk came from ("act", "map_page" or
    "wi_page", see source_groups in policy.json) so the lookup tool can tell
    the user what they are reading.
    """
    chunks: list[tuple[str, str, str]] = []
    buf: list[str] = []
    size = 0
    for unit in unit_texts:
        if not unit:
            continue
        unit_tokens = count_tokens(unit, tokenizer)
        if unit_tokens > token_limit:
            if buf:
                chunks.append((doc, title, " ".join(buf)))
                buf, size = [], 0
            chunks.extend(
                (doc, title, part)
                for part in split_oversize(unit, tokenizer, token_limit)
            )
            continue
        if buf and size + unit_tokens > token_limit:
            chunks.append((doc, title, " ".join(buf)))
            buf, size = [], 0
        buf.append(unit)
        size += unit_tokens
    if buf:
        chunks.append((doc, title, " ".join(buf)))
    return [
        {
            "id": f"{doc}#{start_idx + i}",
            "doc": doc,
            "title": t,
            "text": text,
            "source": source,
        }
        for i, (_, t, text) in enumerate(chunks)
    ]


def build_act_chunks(sections: list[dict], tokenizer):
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
        chunks.extend(
            make_chunks(doc, title, sec["paras"], 0, source="act", tokenizer=tokenizer)
        )
        n += 1
    return chunks, n


def _blocks(root) -> list[tuple[str, list[str]]]:
    """Split a content element into (heading, [paragraph]) blocks."""
    for tag in root.find_all(["script", "style", "nav", "footer", "form"]):
        tag.decompose()
    blocks, cur_head, cur_paras = [], "Overview", []
    for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
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


def parse_wi_page(html: str):
    """Split a WI page's <main> content into (heading, [paragraph]) blocks."""
    soup = BeautifulSoup(html, "html.parser")
    return _blocks(soup.find("main") or soup.body)


def parse_map_page(html: str):
    """Split a Map page's #content into (heading, [paragraph]) blocks. The
    page-search box and the sibling-topic menu are dropped first: they are
    navigation, not policy."""
    soup = BeautifulSoup(html, "html.parser")
    content = soup.find(id="content") or soup.body
    for nav_id in ("above-pannel", "second-level-nav"):
        nav = content.find(id=nav_id)
        if nav is not None:
            nav.decompose()
    return _blocks(content)


# Headings whose blocks are navigation/cross-references rather than policy.
GENERIC_HEADS = {
    "for more information see",
    "related information",
    "related",
    "contents",
    "legislation",
}


def is_generic_head(head: str) -> bool:
    """Navigation/cross-reference headings (tolerating a trailing colon)."""
    return head.strip().rstrip(":").lower() in GENERIC_HEADS


def blocks_text(blocks: list[tuple[str, list[str]]]) -> str:
    """One page's blocks as text (headings + paragraphs)."""
    parts: list[str] = []
    for head, paras in blocks:
        if head:
            parts.append(f"## {head}")
        parts.extend(paras)
    return "\n".join(parts).strip()


def is_single_chunk_page(text: str, tokenizer) -> bool:
    """A non-trivial page that fits the embedding window becomes one chunk,
    so retrieval returns a complete, self-contained page."""
    return 15 <= len(text.split()) and count_tokens(text, tokenizer) <= TOKEN_LIMIT


def map_breadcrumb(url: str) -> str:
    """The directory path of a Map URL as a readable breadcrumb (e.g.
    "income support > main benefits > jobseeker support")."""
    segs = [s for s in urlparse(url).path.split("/") if s]
    dirs = segs[2:-1]  # drop "map" and the filename
    return " > ".join(s.replace("-", " ") for s in dirs)


def build_wi_chunks(
    slug: str,
    html: str,
    tokenizer,
    seen_blocks: set[str],
    *,
    source: str = "wi_page",
):
    blocks = unique_blocks(
        [
            (head, paras)
            for head, paras in parse_wi_page(html)
            if not is_generic_head(head)
        ],
        seen_blocks,
    )
    if not blocks:
        return []
    page_title = blocks[0][0] or "Work and Income guidance"
    whole = blocks_text(blocks)
    if is_single_chunk_page(whole, tokenizer):
        return [
            {
                "id": f"WI-{slug}#0",
                "doc": f"WI-{slug}",
                "title": page_title,
                "text": whole,
                "source": source,
            }
        ]
    chunks = []
    for head, paras in blocks:
        units = make_chunks(
            f"WI-{slug}",
            head or "Work and Income guidance",
            paras,
            0,
            source=source,
            tokenizer=tokenizer,
        )
        chunks.extend(u for u in units if len(u["text"].split()) >= 15)
    return chunks


def build_map_chunks(
    slug: str,
    url: str,
    html: str,
    tokenizer,
    seen_blocks: set[str],
):
    """Chunk one Map page (each page is one document, doc id MAP-<slug>).

    A page that fits the embedding window becomes a single chunk carrying
    the whole page (title + URL breadcrumb for context, e.g. "Income stops —
    income support > main benefits > jobseeker support"). Bigger pages are
    chunked by heading. Pure navigation / cross-reference blocks ("For more
    information see", "Related information", "Contents", "Legislation"),
    very short fragments, blocks already seen on an earlier page, and "List
    of amendments" pages are dropped: they are not policy."""
    breadcrumb = map_breadcrumb(url)
    blocks = [
        (head, paras)
        for head, paras in parse_map_page(html)
        if not is_generic_head(head)
    ]
    blocks = unique_blocks(blocks, seen_blocks)
    if not blocks:
        return []
    page_title = blocks[0][0] or "Map guidance"
    if page_title.strip().lower().startswith("list of amendments"):
        return []
    full_title = f"{page_title} — {breadcrumb}" if breadcrumb else page_title
    whole = blocks_text(blocks)
    if is_single_chunk_page(whole, tokenizer):
        return [
            {
                "id": f"MAP-{slug}#0",
                "doc": f"MAP-{slug}",
                "title": full_title,
                "text": whole,
                "source": "map_page",
            }
        ]
    chunks = []
    for head, paras in blocks:
        title = (
            f"{head or 'Overview'} — {breadcrumb}"
            if breadcrumb
            else (head or "Map guidance")
        )
        units = make_chunks(
            f"MAP-{slug}", title, paras, 0, source="map_page", tokenizer=tokenizer
        )
        chunks.extend(u for u in units if len(u["text"].split()) >= 15)
    return chunks


def main(argv=None):
    args = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    args.add_argument("--dry-run", action="store_true", help="stats only, no write")
    opts = args.parse_args(argv)

    # The embedding model's own tokenizer measures chunk sizes, so the
    # window is exactly the one the model will apply.
    print(f"Loading tokenizer for {EMBEDDING_MODEL} ...")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    # Measure true lengths: no truncation and no over-length warning.
    tokenizer.model_max_length = 100_000
    seen_blocks: set[str] = set()

    print("Fetching/parsing Social Security Act 2018 ...")
    sections = parse_act_sections(fetch(ACT_URL))
    print(f"  {len(sections)} provisions parsed")
    act_chunks, n_acts = build_act_chunks(sections, tokenizer)
    print(f"  {n_acts} sections -> {len(act_chunks)} chunks (entire Act)")

    print("Discovering Map (Manuals and Procedures) pages ...")
    map_pages = fetch_map_pages()
    print(f"  {len(map_pages)} Map pages in scope")

    # Fetch and chunk every Map page: the operational policy layer of the
    # corpus, between the statute and the public guidance site. Repeated
    # blocks across pages are kept once (``seen_blocks``).
    map_chunks = []
    for slug, url in sorted(map_pages.items()):
        map_chunks.extend(
            build_map_chunks(slug, url, fetch(url), tokenizer, seen_blocks)
        )
    print(f"  -> {len(map_chunks)} map_page chunks")

    print("Discovering Work and Income pages from sitemap ...")
    wi_pages = fetch_wi_pages()
    print(f"  {len(wi_pages)} public WI pages in scope")

    # Fetch and chunk every public page once to build the plain RAG corpus.
    wi_chunks = []
    for slug, url in sorted(wi_pages.items()):
        page_chunks = build_wi_chunks(
            slug, fetch(url), tokenizer, seen_blocks, source="wi_page"
        )
        wi_chunks.extend(page_chunks)
    print(f"  -> {len(wi_chunks)} wi_page chunks")

    all_chunks = act_chunks + map_chunks + wi_chunks
    print(
        f"Total corpus: {len(all_chunks)} chunks "
        f"(Act {len(act_chunks)} + Map {len(map_chunks)} + "
        f"Work and Income {len(wi_chunks)})"
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

    write_json(
        OUT_PATH,
        {
            "description": (
                "Policy library (environment data) for the welfare profile "
                "built from public sources, in this order: the ENTIRE "
                "Social Security Act 2018 (legislation.govt.nz), the Map - "
                "Guide to Social Development Policy corpus "
                "(workandincome.govt.nz/map - Work and Income's public "
                "operational policy and procedures, the public copy of the "
                "Ministry's Manuals and Procedures), and the public Work "
                "and Income guidance site (workandincome.govt.nz) covering "
                "the header-bar policy sections (work, eligibility, "
                "on-a-benefit, housing, products) including the full A-Z "
                "benefits and payments list. Content is Crown copyright, "
                "licensed for re-use under CC BY 4.0: attribute the "
                "Ministry of Social Development (and indicate changes - the "
                "text has been chunked and reformatted for retrieval). New "
                "Zealand legislation is not subject to copyright. This is a "
                "point-in-time research snapshot, not the official or "
                "current text. Each "
                "chunk carries a ``source`` label: 'act' (the statute), "
                "'map_page' (a Map operational policy page) or 'wi_page' "
                "(a public Work and Income page). Flat chunked RAG corpus; "
                "lookup_msd_policy retrieves top chunks semantically (no "
                "lexical fallback) using the "
                f"companion vector index {VECTORS_PATH.name} (model: "
                f"{EMBEDDING_MODEL})."
            ),
            "embedding_model": EMBEDDING_MODEL,
            "query_prefix": QUERY_PREFIX,
            "source_groups": {
                "SSA2018": source_counts.get("act", 0),
                "MAP": source_counts.get("map_page", 0),
                "WI_PAGES": source_counts.get("wi_page", 0),
            },
            "chunks": all_chunks,
        },
    )
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
