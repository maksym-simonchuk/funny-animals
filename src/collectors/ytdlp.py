"""Download videos from an explicit URL list via yt-dlp.

Single-threaded, fixed delay between requests, no login and no proxies — this is a
plain downloader for URLs the user collected themselves, not a crawler.
"""
from __future__ import annotations

import random
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from src.processors.dedupe import find_duplicate, sha256_file
from src.processors.video import probe
from src.storage.db import session_scope
from src.storage.files import video_path
from src.storage.models import BROWSER_SOURCE, Run, Video, VideoStatus

if TYPE_CHECKING:
    from src.config import Config

# A URL that yt-dlp will never resolve on a retry: no point burning the delay budget.
_PERMANENT_MARKERS = (
    "unsupported url",
    "is not available",
    "no longer available",
    "video unavailable",
    "private",
    "removed",
    "does not exist",
    "requested content is not available",
    "login required",
    "sign in",
    "404",
    # a photo post: the grid tile looks the same as a video's, so these do reach us
    "no video in this post",
)

_DEFAULTS = {"delay_min": 3.0, "delay_max": 5.0, "retries": 3}


@dataclass
class FetchStats:
    """Outcome of one `fetch` run. `reasons` counts why URLs were skipped."""

    found: int = 0
    downloaded: int = 0
    skipped: int = 0
    errors: int = 0
    reasons: Counter[str] = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.reasons[reason] += 1

    def merge(self, other: "FetchStats") -> None:
        self.found += other.found
        self.downloaded += other.downloaded
        self.skipped += other.skipped
        self.errors += other.errors
        self.reasons.update(other.reasons)


def read_urls(path: Path, limit: int) -> list[str]:
    """One URL per line. Blank lines and `#` comments are ignored, duplicates collapsed."""
    if not path.is_file():
        raise FileNotFoundError(f"URL list not found: {path}")

    seen: dict[str, None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        url = line.split("#", 1)[0].strip()
        if url:
            seen.setdefault(url, None)
    return list(seen)[:limit]


def _options(cfg: "Config") -> dict[str, Any]:
    collector = cfg.collectors.get("ytdlp")
    options = dict(_DEFAULTS)
    if collector is not None:
        options.update({k: v for k, v in collector.options.items() if k in _DEFAULTS})
    return options


def _ydl_opts(cfg: "Config", out_dir: Path) -> dict[str, Any]:
    return {
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "format": "bv*+ba/b",
        "merge_output_format": cfg.processing.video.target_format,
        "max_filesize": cfg.storage.max_file_size * 1024 * 1024,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        # Explicitly off: this tool downloads URLs the user supplies, nothing more.
        "cookiefile": None,
        "proxy": None,
    }


def _is_permanent(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _PERMANENT_MARKERS)


def _download(url: str, opts: dict[str, Any], retries: int) -> dict[str, Any]:
    """Download one URL, retrying transient network failures with a backing-off wait."""
    for attempt in range(1, retries + 1):
        try:
            with YoutubeDL(opts) as ydl:
                return ydl.extract_info(url, download=True)
        except DownloadError as exc:
            if _is_permanent(exc) or attempt == retries:
                raise
            wait = 2**attempt
            logger.warning(f"{url}: attempt {attempt}/{retries} failed ({exc}), retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def _downloaded_file(info: dict[str, Any], out_dir: Path) -> Path:
    """Locate what yt-dlp actually wrote — the extension is only known after the fact."""
    downloads = info.get("requested_downloads") or []
    for entry in downloads:
        candidate = entry.get("filepath")
        if candidate and Path(candidate).is_file():
            return Path(candidate)

    files = [p for p in out_dir.iterdir() if p.is_file() and not p.name.endswith(".part")]
    if not files:
        raise FileNotFoundError(f"yt-dlp reported success but wrote nothing to {out_dir}")
    return max(files, key=lambda p: p.stat().st_size)


def _already_fetched(session, url: str, exclude_id: int | None) -> bool:
    stmt = select(Video.id).where(Video.page_url == url)
    if exclude_id is not None:  # the queue row itself is not a prior fetch
        stmt = stmt.where(Video.id != exclude_id)
    return session.execute(stmt).first() is not None


def _resolve_queue(queue_id: int | None, reason: str) -> None:
    """Take a skipped queue row out of the queue so --watch does not retry it forever."""
    if queue_id is None:
        return
    with session_scope() as session:
        video = session.get(Video, queue_id)
        if video is not None:
            video.status = VideoStatus.REJECTED
            video.reject_reason = reason


def _fetch_one(
    cfg: "Config", url: str, retries: int, stats: FetchStats, queue_id: int | None = None
) -> None:
    """Download one URL and persist it, or record why it was skipped.

    With `queue_id` the row already exists (queued by the browser extension) and is
    updated in place; otherwise a new row is inserted.
    """
    with session_scope() as session:
        if _already_fetched(session, url, queue_id):
            logger.info(f"skip {url}: already in database")
            stats.skip("already_fetched")
            _resolve_queue(queue_id, "already_fetched")
            return

    with tempfile.TemporaryDirectory(prefix="fetch-") as tmp:
        tmp_dir = Path(tmp)
        info = _download(url, _ydl_opts(cfg, tmp_dir), retries)
        temp_file = _downloaded_file(info, tmp_dir)

        try:
            probed = probe(temp_file)
        except ValueError as exc:
            logger.info(f"skip {url}: unreadable download ({exc})")
            stats.skip("corrupt")
            _resolve_queue(queue_id, "corrupt")
            return

        video_cfg = cfg.processing.video
        if not video_cfg.min_duration <= probed.duration_s <= video_cfg.max_duration:
            logger.info(
                f"skip {url}: duration {probed.duration_s:.1f}s outside "
                f"{video_cfg.min_duration}-{video_cfg.max_duration}s"
            )
            stats.skip("duration")
            _resolve_queue(queue_id, "duration")
            return

        digest = sha256_file(temp_file)
        with session_scope() as session:
            existing = find_duplicate(session, digest, None)
            if existing is not None:
                logger.info(f"skip {url}: same file hash as video {existing.id}")
                stats.skip("duplicate")
                _resolve_queue(queue_id, "duplicate")
                return

        source = str(info.get("extractor") or "ytdlp").lower()
        source_id = str(info.get("id") or digest[:16])
        target = video_path(cfg, None, source, source_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(temp_file), target)

    fields = dict(
        download_url=info.get("url"),
        author=info.get("uploader") or info.get("channel"),
        author_url=info.get("uploader_url") or info.get("channel_url"),
        title=info.get("title"),
        tags=list(info.get("tags") or []),
        duration_s=probed.duration_s,
        width=probed.width,
        height=probed.height,
        fps=probed.fps,
        size_bytes=probed.size_bytes,
        codec=probed.codec,
        file_path=str(target),
        sha256=digest,
        status=VideoStatus.DOWNLOADED,
    )
    try:
        with session_scope() as session:
            if queue_id is None:
                session.add(
                    Video(
                        source=source,
                        source_id=source_id,
                        page_url=info.get("webpage_url") or url,
                        # Source terms are unknown for arbitrary URLs; export skips these
                        # by default so nothing unlicensed leaks into a published dataset.
                        license="unknown",
                        **fields,
                    )
                )
            else:
                # Keep source/source_id/page_url: they record how the URL was found.
                video = session.get(Video, queue_id)
                for key, value in fields.items():
                    setattr(video, key, value)
                video.reject_reason = None
    except IntegrityError:
        target.unlink(missing_ok=True)
        logger.info(f"skip {url}: {source}:{source_id} already in database")
        stats.skip("duplicate")
        _resolve_queue(queue_id, "duplicate")
        return

    stats.downloaded += 1
    logger.info(f"downloaded {url} -> {target} ({probed.duration_s:.1f}s, {probed.width}x{probed.height})")


def _run_loop(
    cfg: "Config", items: list[tuple[str, int | None]], source: str, query: str
) -> FetchStats:
    """Download `items` one at a time, pausing between requests. Records one Run row."""
    options = _options(cfg)
    retries = int(options["retries"])
    stats = FetchStats(found=len(items))

    with session_scope() as session:
        run = Run(command="fetch", source=source, query=query, found=len(items))
        session.add(run)
        session.flush()
        run_id = run.id

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("{task.fields[url]}"),
    ) as progress:
        task = progress.add_task("fetching", total=len(items), url="")
        for index, (url, queue_id) in enumerate(items):
            progress.update(task, url=url[:60])
            if index:  # fixed pause between requests, never before the first one
                time.sleep(random.uniform(options["delay_min"], options["delay_max"]))
            try:
                _fetch_one(cfg, url, retries, stats, queue_id)
            except (DownloadError, OSError, RuntimeError) as exc:
                stats.errors += 1
                stats.reasons["error"] += 1
                logger.error(f"failed {url}: {exc}")
                _resolve_queue(queue_id, "error")
            progress.advance(task)

    with session_scope() as session:
        run = session.get(Run, run_id)
        run.downloaded = stats.downloaded
        run.skipped = stats.skipped
        run.errors = stats.errors
        run.finished_at = datetime.now(timezone.utc)

    logger.info(
        f"fetch done: downloaded={stats.downloaded} skipped={stats.skipped} "
        f"errors={stats.errors} reasons={dict(stats.reasons)}"
    )
    return stats


def run_fetch(cfg: "Config", urls_path: Path, limit: int) -> FetchStats:
    """Download every URL in `urls_path`, one at a time, with a pause between requests."""
    urls = read_urls(Path(urls_path), limit)
    logger.info(f"fetch: {len(urls)} URL(s) from {urls_path}")
    return _run_loop(cfg, [(url, None) for url in urls], "ytdlp", str(urls_path))


def _queue_batch(limit: int) -> list[tuple[str, int | None]]:
    """Oldest `limit` URLs the browser extension queued and nothing has downloaded yet."""
    with session_scope() as session:
        rows = session.execute(
            select(Video.id, Video.page_url)
            .where(Video.source == BROWSER_SOURCE, Video.status == VideoStatus.DISCOVERED)
            .order_by(Video.id)
            .limit(limit)
        ).all()
    return [(page_url, video_id) for video_id, page_url in rows]


def run_fetch_from_queue(cfg: "Config", limit: int, watch: bool) -> FetchStats:
    """Download URLs the browser extension queued. With `watch`, poll until interrupted."""
    total = FetchStats()
    while True:
        batch = _queue_batch(limit)
        if batch:
            logger.info(f"fetch: {len(batch)} URL(s) from the browser queue")
            total.merge(_run_loop(cfg, batch, BROWSER_SOURCE, "queue"))
        elif not watch:
            logger.info("fetch: the browser queue is empty")
        if not watch:
            return total
        time.sleep(cfg.browser_mode.poll_interval)
