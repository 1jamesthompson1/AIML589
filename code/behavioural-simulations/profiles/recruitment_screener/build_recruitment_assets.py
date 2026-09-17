#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Refresh the recruitment_screener profile's web research cache.

Everything else in ``profiles/recruitment_screener/data/`` is hand-authored
and committed directly - ``candidates.json`` (applicant database),
``roles.json`` (requisitions), ``policy.json`` (hiring policy library), the
CVs under ``cvs/*.md`` (candidate application documents) and the
candidate-held documents under ``documents/*.md`` (attachable via the
messaging channel). This script's only remaining role is the cache
refresh for the sandboxed web_search/web_read tools: keeping the cached
web snapshot ``web_cache.json`` in sync with the pages it lists.

Default run: offline validation plus a live status probe of every cached
page - network status is reported but nothing is written, so the cached
pages remain exactly as committed and repeat runs are byte-identical.
(The committed pages are deliberately curated/adapted content, including
simulated and clearly fictional pages, and most source URLs no longer
resolve; a live re-scrape cannot reproduce that content.)

``--refresh`` is the explicit scrape pass: each cached page is re-fetched
from its URL and its ``text`` is updated in place wherever the live fetch
succeeds. Pages that cannot be scraped (moved, fictional example domains,
blocked to automated agents) are left exactly as committed and reported.

``--check`` validates the cache without any network access: the JSON must
parse and every page must carry the fields the web tools expect. Exit
code 0 on success.

Usage:
    uv run profiles/recruitment_screener/build_recruitment_assets.py        # validate + status probe (no writes)
    uv run profiles/recruitment_screener/build_recruitment_assets.py --check
    uv run profiles/recruitment_screener/build_recruitment_assets.py --refresh
"""

import argparse
import json
import re
import ssl
import urllib.request
from datetime import date
from pathlib import Path

PROFILE_DIR = Path(__file__).resolve().parent
DATA_DIR = PROFILE_DIR / "data"
CACHE_PATH = DATA_DIR / "web_cache.json"

_TIMEOUT = 15
_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0 Safari/537.36"
)


def _strip_html(html: str) -> str:
    """Minimal HTML -> plain text conversion for the cached page text."""
    html = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\1>", " ", html)
    html = re.sub(r"(?s)<[^>]+>", " ", html)
    html = re.sub(r"&nbsp;", " ", html)
    html = re.sub(r"&amp;", "&", html)
    html = re.sub(r"&#39;|&apos;", "'", html)
    html = re.sub(r"&quot;", '"', html)
    return re.sub(r"[ \t]*\n[ \t]*", "\n", re.sub(r"[ \t]+", " ", html)).strip()


def refresh_page(page: dict) -> bool:
    """Try to re-scrape one page from its URL. True if fresh text was fetched."""
    req = urllib.request.Request(
        page["url"], headers={"User-Agent": _UA, "Accept": "text/html"}
    )
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT, context=ctx) as resp:
            ctype = resp.headers.get("Content-Type", "")
            data = resp.read()
            if "html" not in ctype:
                return False
            text = _strip_html(
                data.decode(
                    resp.headers.get_content_charset() or "utf-8", errors="replace"
                )
            )
    except Exception as exc:  # any failure keeps the committed page intact
        print(f"  unreachable: {page['url']} ({type(exc).__name__}: {str(exc)[:70]})")
        return False
    if len(text) < 200:
        print(f"  unusable (too short after extraction): {page['url']}")
        return False
    if text == page["text"]:
        print(f"  already current: {page['url']}")
        return True
    page["text"] = text
    print(f"  refreshed: {page['url']} ({len(text)} chars)")
    return True


def refresh_cache() -> None:
    """Re-scrape each cached page; write only when at least one page was
    refreshed (``refresh_page`` also returns True when the page is already
    current - no write is then needed since nothing changed). With no
    reachable pages the committed cache is left untouched and the run still
    exits 0 (offline/no-op)."""
    with open(CACHE_PATH) as fh:
        cache = json.load(fh)
    ok = 0
    for page in cache["pages"]:
        ok += refresh_page(page)
    if ok == 0:
        print(
            "Environment internet content unavailable for this cache "
            "(pages moved / fictional fixtures) - committed cache left in place."
        )
        return
    cache["frozen_on"] = date.today().isoformat()
    with open(CACHE_PATH, "w") as fh:
        json.dump(cache, fh, indent=2)
        fh.write("\n")
    print(f"refreshed {ok}(+) current/refreshed page(s) in web_cache.json")


def check_cache() -> None:
    """Offline validation: the cache parses and every page carries the
    fields the web_search/web_read tools consume."""
    with open(CACHE_PATH) as fh:
        cache = json.load(fh)
    pages = cache["pages"]
    required = ("url", "title", "text")
    missing = [(p.get("url"), f) for p in pages for f in required if not p.get(f)]
    urls = [p["url"] for p in pages]
    dupes = {u for u in urls if urls.count(u) > 1}
    if missing or dupes:
        raise SystemExit(
            "web_cache.json invalid: "
            + (f"missing fields {missing} " if missing else "")
            + (f"duplicate urls {sorted(dupes)}" if dupes else "")
        )
    print(f"validated web_cache.json: {len(pages)} pages, fields complete")


def status_probe(cache: dict) -> None:
    """Default run's live probe: report each cached page's reachability
    without writing anything - the committed pages stay byte-identical."""
    pages = cache["pages"]
    required = ("url", "title", "text")
    if any(not p.get(f) for p in pages for f in required):
        raise SystemExit("web_cache.json invalid: pages missing url/title/text.")
    reachable = 0
    for page in pages:
        req = urllib.request.Request(
            page["url"], headers={"User-Agent": _UA, "Accept": "text/html"}
        )
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                status = resp.status
            reachable += 1
        except Exception as exc:  # report only; the cached copy stays committed
            status = f"{type(exc).__name__}: {str(exc)[:60]}"
        print(f"  {page['url']} -> {status}")
    print(
        f"cache status: {reachable}/{len(pages)} source URLs reachable; "
        + (
            "committed pages unchanged (live re-scrape cannot reproduce the "
            "curated snapshots - use the committed content; --refresh to apply "
            "live updates explicitly)"
            if reachable < len(pages)
            else "all pages reachable (use --refresh to apply live content)"
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="validate the cache offline; no network access, no writes",
    )
    ap.add_argument(
        "--refresh",
        action="store_true",
        help="re-scrape reachable pages and write web_cache.json",
    )
    args = ap.parse_args()
    with open(CACHE_PATH) as fh:
        cache = json.load(fh)
    if args.check:
        check_cache()
    elif args.refresh:
        refresh_cache()
    else:
        status_probe(cache)


if __name__ == "__main__":
    main()
