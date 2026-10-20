"""Space-Track client for TLE histories (https://www.space-track.org).

Credentials come from the environment or a git-ignored `.env` file, never from code:
    SPACETRACK_USER=...   SPACETRACK_PASSWORD=...
Space-Track data is licensed for your own use; keep it in the git-ignored cache, not the repo.
Past element sets never change, so cached responses are kept indefinitely. Requests are
spaced out to stay inside the Space-Track rate limits (batch NORAD ids into one query).
"""
from __future__ import annotations

import hashlib
import http.cookiejar
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Protocol

from .celestrak import TLE, parse_tle_history

BASE = "https://www.space-track.org"
LOGIN_PATH = "/ajaxauth/login"


class SpaceTrackError(RuntimeError):
    pass


class MissingCredentials(SpaceTrackError):
    pass


@dataclass(frozen=True)
class Credentials:
    identity: str
    password: str = field(repr=False)

    def __str__(self) -> str:
        return f"Credentials(identity={self.identity!r})"


def load_dotenv(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines; ignores blanks and # comments; strips matching quotes."""
    values: dict[str, str] = {}
    p = Path(path)
    if not p.is_file():
        return values
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in "\"'":
            val = val[1:-1]
        values[key.strip()] = val
    return values


def credentials_from_env(dotenv_path: Path | None = None) -> Credentials:
    """Environment variables win over `.env`. Raises MissingCredentials with setup help."""
    file_vals = load_dotenv(dotenv_path or Path(".env"))
    user = os.environ.get("SPACETRACK_USER") or file_vals.get("SPACETRACK_USER")
    pw = os.environ.get("SPACETRACK_PASSWORD") or file_vals.get("SPACETRACK_PASSWORD")
    if not user or not pw:
        raise MissingCredentials(
            "Space-Track credentials not found. Create a free account at space-track.org, "
            "copy .env.example to .env and fill in SPACETRACK_USER and SPACETRACK_PASSWORD "
            "(.env is git-ignored).")
    return Credentials(user, pw)


class Transport(Protocol):
    def login(self, identity: str, password: str) -> None: ...
    def get(self, url: str) -> str: ...


class UrllibTransport:
    """Session-cookie login, then plain GETs."""

    def __init__(self) -> None:
        jar = http.cookiejar.CookieJar()
        self._opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))

    def login(self, identity: str, password: str) -> None:
        data = urllib.parse.urlencode({"identity": identity, "password": password}).encode()
        try:
            with self._opener.open(BASE + LOGIN_PATH, data, timeout=30) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.URLError as exc:
            raise SpaceTrackError(f"Space-Track login request failed: {exc.reason}") from None
        if "Failed" in body:
            raise SpaceTrackError("Space-Track login failed (check username and password)")

    def get(self, url: str) -> str:
        try:
            with self._opener.open(url, timeout=120) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.URLError as exc:
            raise SpaceTrackError(f"Space-Track request failed: {exc}") from None


class SpaceTrackClient:
    def __init__(self, credentials: Credentials, cache_dir: Path,
                 transport: Transport | None = None, min_interval_s: float = 3.0) -> None:
        self._creds = credentials
        self._cache = Path(cache_dir)
        self._transport = transport or UrllibTransport()
        self._min_interval = min_interval_s
        self._logged_in = False
        self._last = 0.0

    def _fetch(self, path: str, suffix: str) -> str:
        key = hashlib.sha256(path.encode()).hexdigest()[:16]
        cached = self._cache / f"{key}.{suffix}"
        if cached.is_file():
            return cached.read_text(encoding="utf-8")
        if not self._logged_in:
            self._transport.login(self._creds.identity, self._creds.password)
            self._logged_in = True
        wait = self._min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        text = self._transport.get(BASE + path)
        self._last = time.monotonic()
        self._cache.mkdir(parents=True, exist_ok=True)
        cached.write_text(text, encoding="utf-8")
        return text

    def query(self, path: str, suffix: str = "json") -> str:
        """Run a read-only query (path must start with /basicspacedata/query/); cached."""
        if not path.startswith("/basicspacedata/query/"):
            raise SpaceTrackError("only /basicspacedata/query/ paths are allowed")
        return self._fetch(path, suffix)

    def gp_history(self, norad_ids: list[int], start: date, end: date) -> list[TLE]:
        """All element sets for the objects with epoch in [start, end], sorted by epoch."""
        ids = ",".join(str(i) for i in norad_ids)
        path = (f"/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{ids}"
                f"/EPOCH/{start.isoformat()}--{end.isoformat()}"
                f"/orderby/EPOCH%20asc/format/3le")
        hist = parse_tle_history(self._fetch(path, "3le"))
        if not hist:
            raise SpaceTrackError(f"no element sets returned for {ids} between {start} and {end}")
        return hist

    def launch_objects(self, intl_designator: str) -> list[dict]:
        """Catalogue rows (SATCAT) for a launch, e.g. '2022-010'."""
        path = (f"/basicspacedata/query/class/satcat/INTLDES/~~{intl_designator}"
                f"/orderby/NORAD_CAT_ID%20asc/format/json")
        return json.loads(self._fetch(path, "json"))
