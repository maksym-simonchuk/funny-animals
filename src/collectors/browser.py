"""Run the URL collector unattended: a Chrome of our own, walking a feed on its own.

Nothing about what the extension does changes here -- same scroll pacing, same tag
filter, same local queue. What this adds is a window nobody has to sit in front of:
its own profile (log in once with `--login` and the cookies stay), the extension
loaded straight out of the repo, and a session file that tells it which server to
post to and to start without the popup. Headless by default, closed on the clock.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from src.config import Config

EXTENSION_DIR = Path(__file__).resolve().parents[2] / "browser_extension"
# Read by the extension's service worker at startup, and only there: an extension
# cannot be handed settings from outside, but it can read its own folder.
SESSION_FILE = EXTENSION_DIR / "session.json"

# Google Chrome stopped honouring --load-extension in v137: it starts, it browses, and
# it writes "--disable-extensions-except is not allowed in Google Chrome, ignoring" to
# its log while the extension is simply not there. So the run needs a Chromium that
# still honours it -- Chrome for Testing, which Playwright and Puppeteer each keep a
# copy of, or a plain Chromium.
_CHROME_GLOBS = (
    "~/Library/Caches/ms-playwright/chromium-*/chrome-mac*/Google Chrome for Testing.app"
    "/Contents/MacOS/Google Chrome for Testing",
    "~/.cache/puppeteer/chrome/*/chrome-mac*/Google Chrome for Testing.app"
    "/Contents/MacOS/Google Chrome for Testing",
    "~/.cache/puppeteer/chrome/*/chrome-linux*/chrome",
    "~/Library/Caches/ms-playwright/chromium-*/chrome-linux*/chrome",
)
_CHROME_NAMES = (
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "chromium",
    "chromium-browser",
)
_BRANDED = "Google Chrome.app"

_CLOSE_GRACE = 15.0  # seconds Chrome gets to shut down before it is killed


class BrowserError(RuntimeError):
    """No browser that can load the extension, or no queue server to post into."""


def chrome_binary(explicit: str = "") -> Path:
    """The browser to launch: the one you named, else the newest Chrome for Testing or
    Chromium on the machine."""
    if explicit:
        found = shutil.which(explicit)
        if not found:
            raise BrowserError(f"no browser at {explicit}")
        return _unbranded(Path(found))
    for pattern in _CHROME_GLOBS:
        matches = sorted(glob.glob(os.path.expanduser(pattern)))
        if matches:
            return Path(matches[-1])
    for name in _CHROME_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    raise BrowserError(
        "no Chrome for Testing or Chromium found, and Google Chrome ignores "
        "--load-extension since v137. Install one with `npx @puppeteer/browsers install "
        "chrome@stable`, or name yours with --chrome"
    )


def _unbranded(path: Path) -> Path:
    """Refuse Google Chrome by name: it launches and browses perfectly well, it just
    drops the extension on the floor, and a run that collects nothing that way looks
    exactly like a feed with nothing in it."""
    if _BRANDED in str(path):
        raise BrowserError(
            f"{path} is Google Chrome, which ignores --load-extension since v137 -- "
            "point --chrome at Chrome for Testing or Chromium instead"
        )
    return path


def run_session(
    cfg: "Config",
    url: str,
    *,
    server: str,
    minutes: float,
    headless: bool = True,
    chrome: str = "",
) -> int:
    """Walk `url` until the run ends or `minutes` are up. Returns links queued.

    The count is the queue's own before and after: the extension posts to the server,
    so what the server took is the only honest measure of what the run collected.
    """
    binary = chrome_binary(chrome)
    before = queue_total(server)
    _write_session(cfg, server, autostart=True)
    process = _launch(binary, cfg, url, headless=headless)
    logger.info(f"walking {url} for up to {minutes:g} min")
    _wait(process, minutes * 60)
    return queue_total(server) - before


def open_profile(cfg: "Config", url: str, *, server: str, chrome: str = "") -> None:
    """Open the same profile with nothing automated and wait for you to close it.

    Instagram will not sign in headless, so the first run is by hand; what it leaves
    in the profile is what every later `run_session` rides on.
    """
    binary = chrome_binary(chrome)
    # autostart off, but the server and token still land in the profile, so a manual
    # run from this window's popup works without pasting anything into it
    _write_session(cfg, server, autostart=False)
    _wait(_launch(binary, cfg, url, headless=False), None)


def queue_total(server: str) -> int:
    """How many links are waiting to be downloaded. Doubles as the server check:
    browsing is pointless with nothing listening to take what it finds."""
    try:
        with urllib.request.urlopen(f"{server.rstrip('/')}/queue?per_page=1", timeout=10) as reply:
            return int(json.load(reply)["total"])
    except (OSError, ValueError, KeyError) as exc:
        raise BrowserError(
            f"no queue server at {server} -- start one with `app.py browser-mode`"
        ) from exc


def _launch(binary: Path, cfg: "Config", url: str, *, headless: bool) -> subprocess.Popen:
    profile = cfg.browser_mode.profile_path.resolve()
    profile.mkdir(parents=True, exist_ok=True)
    args = [
        str(binary),
        f"--user-data-dir={profile}",
        f"--disable-extensions-except={EXTENSION_DIR}",
        f"--load-extension={EXTENSION_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
    ]
    if headless:
        args.append("--headless=new")
    args.append(url)
    # Chrome is noisy on stderr about everything and nothing; the run's own log is here
    return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait(process: subprocess.Popen, seconds: float | None) -> None:
    """Wait for Chrome and close it whatever happens -- the timer running out, the end
    of the feed, a Ctrl-C. The extension stops itself; nothing else stops a browser."""
    try:
        process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        logger.info("time is up, closing the browser")
    finally:
        _close(process)
        SESSION_FILE.unlink(missing_ok=True)


def _close(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=_CLOSE_GRACE)
    except subprocess.TimeoutExpired:
        process.kill()


def _write_session(cfg: "Config", server: str, *, autostart: bool) -> None:
    """Hand the extension the settings the popup would have been given by hand."""
    SESSION_FILE.write_text(
        json.dumps(
            {
                "serverUrl": server,
                "token": cfg.browser_mode.ingest_token,
                "autostart": autostart,
            },
            indent=2,
        )
        + "\n"
    )
