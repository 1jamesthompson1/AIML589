"""Disposable, on-disk cache for public policy source downloads."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

import requests


def cache_path(cache_dir: Path, url: str) -> Path:
    """Mirror the source host and path under a profile's cache directory."""
    parsed = urlparse(url)
    path = parsed.path or "/"
    if path.endswith("/") or "." not in Path(path).name:
        path = path.rstrip("/") + "/index.html"
    return cache_dir / parsed.netloc / path.lstrip("/")


def cached_file(
    url: str,
    path: Path,
    *,
    sha256: str | None = None,
    text: bool = False,
    timeout: int = 60,
    headers: Mapping[str, str] | None = None,
) -> Path:
    """Return a cached source, downloading once and verifying pinned files.

    Text sources are stored as UTF-8, matching the welfare builder's existing
    cache. Optional ``headers`` support sources that need a browser-like user
    agent. A failed download or checksum check never replaces a valid cache.
    """
    if path.exists() and path.stat().st_size:
        if sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError(f"Cached source has an unexpected SHA256: {path}")
        return path

    response = requests.get(
        url,
        timeout=timeout,
        headers=headers or {"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    if text:
        response.encoding = "utf-8"
        content = response.text.encode("utf-8")
    else:
        content = response.content
    if sha256 and hashlib.sha256(content).hexdigest() != sha256:
        raise ValueError(f"Downloaded source has an unexpected SHA256: {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(content)
    temporary.replace(path)
    return path
