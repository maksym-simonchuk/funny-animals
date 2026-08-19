"""Tests for the unattended browser run. No Chrome is launched and no HTTP is made:
Popen and urlopen are both stubbed, so what is asserted is the command line, the
session file the extension reads, and that the browser is always closed after.
"""
from __future__ import annotations

import io
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from src.collectors import browser
from src.config import BrowserModeCfg

TOKEN = "test-token"
SERVER = "http://127.0.0.1:8000"
URL = "https://www.instagram.com/explore/tags/funnyanimals/"


class _FakeChrome:
    """A browser that never exits on its own unless `runs_for` says otherwise."""

    def __init__(self, runs_for: float | None = None) -> None:
        self.runs_for = runs_for
        self.terminated = False
        self.killed = False
        self.alive = True

    def wait(self, timeout=None):
        if not self.alive:
            return 0
        if self.runs_for is not None and (timeout is None or self.runs_for <= timeout):
            self.alive = False
            return 0
        raise subprocess.TimeoutExpired("chrome", timeout)

    def poll(self):
        return None if self.alive else 0

    def terminate(self):
        self.terminated = True
        self.alive = False

    def kill(self):
        self.killed = True
        self.alive = False


@pytest.fixture
def launched(tmp_path: Path, monkeypatch):
    """Stub Chrome. Returns the recorder: `.args` and the session file as it was seen
    from inside the browser, which is the only moment it exists."""

    class Recorder:
        args: list[str] = []
        session: dict | None = None
        process = _FakeChrome()

    def fake_popen(args, **kwargs):
        Recorder.args = args
        Recorder.session = json.loads(browser.SESSION_FILE.read_text())
        return Recorder.process

    monkeypatch.setattr(browser.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(browser, "_CHROME_GLOBS", ())  # not whatever is installed here
    monkeypatch.setattr(browser.shutil, "which", lambda name: f"/fake/{Path(name).name}")
    monkeypatch.setattr(browser, "SESSION_FILE", tmp_path / "session.json")
    return Recorder


@pytest.fixture
def cfg(tmp_cfg, tmp_path: Path):
    return replace(
        tmp_cfg,
        browser_mode=BrowserModeCfg(
            enabled=True, ingest_token=TOKEN, profile_path=tmp_path / "profile"
        ),
    )


def _stub_queue(monkeypatch, *totals: int) -> None:
    """The /queue endpoint, answering each call with the next total."""
    remaining = list(totals)

    def fake_urlopen(url, timeout=None):
        body = json.dumps({"items": [], "page": 1, "per_page": 1, "total": remaining.pop(0)})
        return io.BytesIO(body.encode())

    monkeypatch.setattr(browser.urllib.request, "urlopen", fake_urlopen)


def test_the_session_file_sits_where_the_extension_can_read_it() -> None:
    # the extension fetches it by chrome.runtime.getURL, so it has to be packaged with it
    assert browser.SESSION_FILE.parent == browser.EXTENSION_DIR
    assert (browser.EXTENSION_DIR / "manifest.json").is_file()


def test_run_session_launches_headless_chrome_with_the_extension(cfg, launched, monkeypatch):
    _stub_queue(monkeypatch, 0, 0)
    launched.process.runs_for = 1.0

    browser.run_session(cfg, URL, server=SERVER, minutes=5)

    assert launched.args[1:] == [
        f"--user-data-dir={(cfg.browser_mode.profile_path).resolve()}",
        f"--disable-extensions-except={browser.EXTENSION_DIR}",
        f"--load-extension={browser.EXTENSION_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
        "--headless=new",
        URL,
    ]
    # the profile has to exist before Chrome is told to use it
    assert cfg.browser_mode.profile_path.is_dir()


def test_run_session_hands_the_extension_the_server_and_starts_it(cfg, launched, monkeypatch):
    _stub_queue(monkeypatch, 0, 0)
    launched.process.runs_for = 1.0

    browser.run_session(cfg, URL, server=SERVER, minutes=5)

    assert launched.session == {"serverUrl": SERVER, "token": TOKEN, "autostart": True}
    # and it is not left lying around with the token in it once the run is over
    assert not browser.SESSION_FILE.exists()


def test_run_session_counts_what_the_queue_took(cfg, launched, monkeypatch):
    _stub_queue(monkeypatch, 12, 30)
    launched.process.runs_for = 1.0

    assert browser.run_session(cfg, URL, server=SERVER, minutes=5) == 18


def test_run_session_closes_a_browser_that_outstays_its_minutes(cfg, launched, monkeypatch):
    _stub_queue(monkeypatch, 0, 0)
    launched.process.runs_for = None  # never exits by itself

    browser.run_session(cfg, URL, server=SERVER, minutes=0.01)

    assert launched.process.terminated and not launched.process.killed
    assert not browser.SESSION_FILE.exists()


def test_run_session_kills_a_browser_that_ignores_the_ask(cfg, launched, monkeypatch):
    _stub_queue(monkeypatch, 0, 0)
    launched.process.runs_for = None
    launched.process.terminate = lambda: None  # a Chrome that will not go quietly

    browser.run_session(cfg, URL, server=SERVER, minutes=0.01)

    assert launched.process.killed


def test_login_opens_a_visible_window_and_does_not_start_a_run(cfg, launched):
    launched.process.runs_for = 1.0

    browser.open_profile(cfg, URL, server=SERVER)

    assert "--headless=new" not in launched.args
    assert launched.session == {"serverUrl": SERVER, "token": TOKEN, "autostart": False}


def test_run_session_says_where_the_queue_server_should_be(cfg, launched, monkeypatch):
    def refuse(url, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(browser.urllib.request, "urlopen", refuse)

    with pytest.raises(browser.BrowserError, match="browser-mode"):
        browser.run_session(cfg, URL, server=SERVER, minutes=5)
    assert launched.args == []  # nothing was launched to collect into nothing


def test_chrome_binary_reports_a_browser_that_is_not_there(monkeypatch):
    monkeypatch.setattr(browser, "_CHROME_GLOBS", ())
    monkeypatch.setattr(browser.shutil, "which", lambda name: None)

    with pytest.raises(browser.BrowserError, match="--chrome"):
        browser.chrome_binary()
    with pytest.raises(browser.BrowserError, match="/nope/chrome"):
        browser.chrome_binary("/nope/chrome")


def test_chrome_binary_takes_the_newest_chrome_for_testing(tmp_path: Path, monkeypatch):
    for version in ("chromium-1108", "chromium-1223", "chromium-1217"):
        (tmp_path / version).mkdir()
        (tmp_path / version / "chrome").touch()
    monkeypatch.setattr(browser, "_CHROME_GLOBS", (f"{tmp_path}/chromium-*/chrome",))
    monkeypatch.setattr(browser.shutil, "which", lambda name: "/fake/chromium")

    assert browser.chrome_binary() == tmp_path / "chromium-1223" / "chrome"


def test_chrome_binary_refuses_google_chrome(monkeypatch):
    # it launches and browses fine, it just drops the extension without saying so
    branded = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
    monkeypatch.setattr(browser.shutil, "which", lambda name: branded)

    with pytest.raises(browser.BrowserError, match="v137"):
        browser.chrome_binary(branded)
