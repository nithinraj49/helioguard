"""Download-with-caching helper. No test touches the network: downloaders are injected."""
import os
import time

import pytest

from helioguard.data.cache import fetch_cached


def _age(path, days):
    old = time.time() - days * 86400
    os.utime(path, (old, old))


def _offline(url):
    raise OSError("no network")


def test_downloads_when_missing_and_then_uses_cache(tmp_path):
    calls = []
    dest = tmp_path / "x.csv"

    def dl(url):
        calls.append(url)
        return b"hello"

    assert fetch_cached("http://example/x", dest, downloader=dl).read_bytes() == b"hello"
    fetch_cached("http://example/x", dest, downloader=dl)
    assert calls == ["http://example/x"]          # second call served from cache


def test_refreshes_stale_cache(tmp_path):
    dest = tmp_path / "x.csv"
    dest.write_bytes(b"old")
    _age(dest, 10)
    fetch_cached("u", dest, max_age_days=1, downloader=lambda u: b"new")
    assert dest.read_bytes() == b"new"


def test_falls_back_to_stale_cache_when_offline(tmp_path):
    dest = tmp_path / "x.csv"
    dest.write_bytes(b"old")
    _age(dest, 10)
    assert fetch_cached("u", dest, max_age_days=1, downloader=_offline).read_bytes() == b"old"


def test_raises_when_offline_and_nothing_cached(tmp_path):
    with pytest.raises(OSError):
        fetch_cached("u", tmp_path / "missing.csv", downloader=_offline)
