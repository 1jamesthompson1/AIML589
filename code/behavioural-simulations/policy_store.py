"""Small, dependency-free helpers for the profiles' JSON policy corpora.

The policy files intentionally remain ordinary JSON documents: a description
plus a list of source-labelled entries. Source fetching and parsing are
profile-specific; this module only provides the common storage and retrieval
operations so the runtime tools do not each grow their own mini-search engine.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

ENTRY_KEYS = ("policy", "chunks", "standards", "protocols")
DEFAULT_SEARCH_FIELDS = ("title", "text", "summary", "decision", "basis", "appeal")
DEFAULT_STOPWORDS = {
    "after",
    "and",
    "are",
    "for",
    "from",
    "how",
    "into",
    "must",
    "the",
    "this",
    "that",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
}
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def _entries_from_data(data: dict[str, Any] | list[dict[str, Any]], key: str | None):
    """Get a policy entry list from a JSON object or an already-loaded list."""
    if isinstance(data, list):
        return data
    if not isinstance(data, dict):
        raise TypeError("policy data must be a JSON object or list")
    if key is not None:
        entries = data.get(key)
    else:
        entries = next(
            (data[k] for k in ENTRY_KEYS if isinstance(data.get(k), list)), None
        )
    if not isinstance(entries, list):
        wanted = key or "/".join(ENTRY_KEYS)
        raise ValueError(f"policy JSON has no entry list for {wanted!r}")
    return entries


def load_policy(path: Path, key: str | None = None) -> list[dict[str, Any]]:
    """Load a policy/standards/protocol entry list from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return _entries_from_data(data, key)


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write readable, deterministic JSON with a final newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def chunk_units(
    units: Iterable[str], target: int = 2000, joiner: str = " "
) -> list[str]:
    """Combine small source units into approximately ``target``-sized chunks.

    A unit is never split, so a single long paragraph may make a chunk larger
    than the target. This is the simple boundary-preserving behaviour used by
    the non-semantic policy builders.
    """
    chunks: list[str] = []
    buffer: list[str] = []
    size = 0
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        buffer.append(unit)
        size += len(unit)
        if size >= target:
            chunks.append(joiner.join(buffer))
            buffer, size = [], 0
    if buffer:
        chunks.append(joiner.join(buffer))
    return chunks


def _words(value: Any) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {token.casefold() for token in _TOKEN_RE.findall(value)}


def _term_score(term: str, words: set[str], *, prefix: bool) -> int:
    if term in words:
        return 2
    if prefix and len(term) >= 4 and any(word.startswith(term) for word in words):
        return 1
    return 0


def search_entries(
    entries: Iterable[dict[str, Any]],
    query: str,
    *,
    fields: Iterable[str] = DEFAULT_SEARCH_FIELDS,
    limit: int | None = None,
    title_boost: int = 3,
    prefix: bool = True,
    stopwords: Iterable[str] = DEFAULT_STOPWORDS,
) -> list[tuple[dict[str, Any], int]]:
    """Rank JSON entries using only their searchable content fields.

    This is intentionally a small lexical search, not a second policy engine:
    it handles title/body keyword matches, gives title matches a small boost and
    supports moderation precedent fields. Source URLs and provenance notes are
    excluded by default so a query does not match an irrelevant citation.
    Short tokens containing digits are retained for section/clause lookups.
    """
    stopword_set = set(stopwords)
    terms = {
        token.casefold()
        for token in _TOKEN_RE.findall(query)
        if (len(token) > 2 or any(char.isdigit() for char in token))
        and token.casefold() not in stopword_set
    }
    if not terms:
        return []
    field_names = tuple(fields)
    title_names = {"title", "name"}
    ranked: list[tuple[dict[str, Any], int]] = []
    for entry in entries:
        title_words: set[str] = set()
        body_words: set[str] = set()
        for field in field_names:
            words = _words(entry.get(field))
            if field in title_names:
                title_words.update(words)
            else:
                body_words.update(words)
        coverage = sum(
            _term_score(term, title_words | body_words, prefix=prefix) for term in terms
        )
        if not coverage:
            continue
        title_matches = sum(
            _term_score(term, title_words, prefix=prefix) for term in terms
        )
        score = coverage * coverage + title_boost * title_matches
        ranked.append((entry, score))
    ranked.sort(key=lambda pair: (-pair[1], str(pair[0].get("id", ""))))
    return ranked[:limit] if limit is not None else ranked
