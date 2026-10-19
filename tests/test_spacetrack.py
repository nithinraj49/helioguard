"""Space-Track client: credentials from .env, offline transport, caching. No network, no secrets.

Fixture spacetrack_synthetic_3le.txt is SYNTHETIC (fake NORAD 99001/99002), see its generator.
"""
from datetime import date
from pathlib import Path

import pytest

from helioguard.data.celestrak import group_by_norad, parse_tle_history
from helioguard.data.spacetrack import (Credentials, MissingCredentials, SpaceTrackClient,
                                        SpaceTrackError, credentials_from_env, load_dotenv)

FIX = Path(__file__).parent / "fixtures"
ROOT = Path(__file__).parent.parent
SYNTH = (FIX / "spacetrack_synthetic_3le.txt").read_text()


class FakeTransport:
    def __init__(self, body=SYNTH, fail_login=False):
        self.body, self.fail_login = body, fail_login
        self.logins, self.gets = [], []

    def login(self, identity, password):
        self.logins.append((identity, password))
        if self.fail_login:
            raise SpaceTrackError("login failed")

    def get(self, url):
        self.gets.append(url)
        return self.body


def make(tmp_path, **kw):
    t = FakeTransport(**kw)
    return SpaceTrackClient(Credentials("me@example.com", "s3cret-pw"), tmp_path / "cache",
                            transport=t, min_interval_s=0), t


# --- credentials -------------------------------------------------------------
def test_load_dotenv_parses_and_ignores_comments(tmp_path):
    env = tmp_path / ".env"
    env.write_text('# comment\nSPACETRACK_USER=me@example.com\nSPACETRACK_PASSWORD="pw with=eq"\n\n')
    assert load_dotenv(env) == {"SPACETRACK_USER": "me@example.com",
                                "SPACETRACK_PASSWORD": "pw with=eq"}


def test_credentials_from_dotenv_and_environment_override(tmp_path, monkeypatch):
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    env = tmp_path / ".env"
    env.write_text("SPACETRACK_USER=a@b.c\nSPACETRACK_PASSWORD=pw1\n")
    assert credentials_from_env(env).identity == "a@b.c"
    monkeypatch.setenv("SPACETRACK_PASSWORD", "pw2")
    assert credentials_from_env(env).password == "pw2"


def test_missing_credentials_gives_a_helpful_error(tmp_path, monkeypatch):
    monkeypatch.delenv("SPACETRACK_USER", raising=False)
    monkeypatch.delenv("SPACETRACK_PASSWORD", raising=False)
    with pytest.raises(MissingCredentials, match=".env"):
        credentials_from_env(tmp_path / "absent.env")


def test_password_never_appears_in_repr_or_errors(tmp_path):
    creds = Credentials("me@example.com", "s3cret-pw")
    assert "s3cret-pw" not in repr(creds) and "s3cret-pw" not in str(creds)
    client, _ = make(tmp_path, fail_login=True)
    with pytest.raises(SpaceTrackError) as exc:
        client.gp_history([99001], date(2022, 2, 3), date(2022, 2, 5))
    assert "s3cret-pw" not in str(exc.value)


def test_dotenv_is_git_ignored_and_example_has_no_secrets():
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    assert ".env" in [ln.strip() for ln in ignored]
    example = (ROOT / ".env.example").read_text()
    assert "SPACETRACK_USER=" in example and "SPACETRACK_PASSWORD=" in example
    assert "@" not in example


# --- queries -----------------------------------------------------------------
def test_gp_history_builds_the_documented_query_and_parses(tmp_path):
    client, t = make(tmp_path)
    hist = client.gp_history([99001, 99002], date(2022, 2, 3), date(2022, 2, 5))
    assert t.logins == [("me@example.com", "s3cret-pw")]
    url = t.gets[0]
    assert "/basicspacedata/query/class/gp_history/" in url
    assert "NORAD_CAT_ID/99001,99002" in url
    assert "EPOCH/2022-02-03--2022-02-05" in url and url.endswith("format/3le")
    assert len(hist) == 12
    assert sorted(group_by_norad(hist)) == [99001, 99002]


def test_results_are_cached_so_repeat_calls_do_not_hit_the_service(tmp_path):
    client, t = make(tmp_path)
    a = client.gp_history([99001], date(2022, 2, 3), date(2022, 2, 5))
    b = client.gp_history([99001], date(2022, 2, 3), date(2022, 2, 5))
    assert len(t.gets) == 1 and [x.line1 for x in a] == [x.line1 for x in b]
    client.gp_history([99001], date(2022, 2, 3), date(2022, 2, 6))   # different query
    assert len(t.gets) == 2


def test_empty_response_is_an_error_not_silent_success(tmp_path):
    client, _ = make(tmp_path, body="")
    with pytest.raises(SpaceTrackError, match="no element sets"):
        client.gp_history([99001], date(2022, 2, 3), date(2022, 2, 5))


def test_launch_objects_query_uses_international_designator(tmp_path):
    client, t = make(tmp_path, body='[{"NORAD_CAT_ID":"51456","OBJECT_NAME":"STARLINK A"}]')
    rows = client.launch_objects("2022-010")
    assert "class/satcat/INTLDES/~~2022-010" in t.gets[0]
    assert rows[0]["NORAD_CAT_ID"] == "51456"


# --- tle helpers -------------------------------------------------------------
def test_tle_records_carry_the_norad_id():
    hist = parse_tle_history(SYNTH)
    assert {t.norad_id for t in hist} == {99001, 99002}
    groups = group_by_norad(hist)
    assert all(g == sorted(g, key=lambda t: t.epoch) for g in groups.values())
