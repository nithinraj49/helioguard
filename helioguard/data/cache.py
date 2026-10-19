"""Download-with-caching. The downloader is injectable so tests never touch the network."""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path
from typing import Callable

Downloader = Callable[[str], bytes]


def http_download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "helioguard/0.1"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def fetch_cached(url: str, dest: Path, *, max_age_days: float = 1.0,
                 downloader: Downloader = http_download) -> Path:
    """Return `dest`, downloading `url` if it is missing or older than `max_age_days`.

    If the refresh fails (offline) a stale cache is used; with no cache the error is raised.
    """
    dest = Path(dest)
    if dest.is_file() and time.time() - dest.stat().st_mtime < max_age_days * 86400:
        return dest
    try:
        data = downloader(url)
    except OSError:
        if dest.is_file():
            return dest
        raise
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return dest
