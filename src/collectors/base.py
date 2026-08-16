"""Shared collector machinery: candidate shape, rate limiting and the download loop.

A collector only implements `search()`; discovery, deduplication, rate-limited
downloading and status bookkeeping all live here.
"""
from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import aiohttp
from loguru import logger
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.storage.db import session_scope
from src.storage.files import atomic_write, video_path
from src.storage.models import Video, VideoStatus

if TYPE_CHECKING:
    from src.config import Config

_CHUNK = 64 * 1024


@dataclass(frozen=True)
class VideoCandidate:
    """One searchable result, before anything has been downloaded."""

    source: str
    source_id: str
    page_url: str
    download_url: str
    author: str | None = None
    author_url: str | None = None
    license: str = "unknown"
    license_url: str | None = None
    title: str | None = None
    tags: list[str] = field(default_factory=list)
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class RunStats:
    found: int = 0
    downloaded: int = 0
    skipped: int = 0
    errors: int = 0
    reasons: Counter[str] = field(default_factory=Counter)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.reasons[reason] += 1

    def merge(self, other: "RunStats") -> None:
        self.found += other.found
        self.downloaded += other.downloaded
        self.skipped += other.skipped
        self.errors += other.errors
        self.reasons.update(other.reasons)


class TokenBucket:
    """Async rate limiter expressed as requests per hour."""

    def __init__(self, rate_per_hour: int) -> None:
        self._interval = 3600.0 / rate_per_hour if rate_per_hour > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next_at = 0.0

    async def acquire(self) -> None:
        if not self._interval:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next_at - now
            self._next_at = max(now, self._next_at) + self._interval
        if wait > 0:
            await asyncio.sleep(wait)


COLLECTORS: dict[str, type["BaseCollector"]] = {}


def register(cls: type["BaseCollector"]) -> type["BaseCollector"]:
    """Class decorator: makes a collector discoverable by name from config.yaml."""
    COLLECTORS[cls.name] = cls
    return cls


class BaseCollector(ABC):
    name: ClassVar[str]

    def __init__(self, cfg: "Config", http: aiohttp.ClientSession) -> None:
        self.cfg = cfg
        self.http = http
        self.settings = cfg.collectors[self.name]
        self._bucket = TokenBucket(self.settings.rate_limit)
        self._semaphore = asyncio.Semaphore(cfg.storage.concurrency)

    @abstractmethod
    def search(self, query: str, limit: int) -> AsyncIterator[VideoCandidate]:
        """Yield up to `limit` candidates for `query`. Implementations are async generators."""
        raise NotImplementedError

    async def _request(self, url: str, **kwargs) -> aiohttp.ClientResponse:
        """Rate-limited GET that honours a 429 Retry-After once before giving up."""
        await self._bucket.acquire()
        response = await self.http.get(url, **kwargs)
        if response.status == 429:
            delay = float(response.headers.get("Retry-After", 60))
            response.release()
            logger.warning(f"{self.name}: rate limited, waiting {delay}s")
            await asyncio.sleep(delay)
            await self._bucket.acquire()
            response = await self.http.get(url, **kwargs)
        response.raise_for_status()
        return response

    def _existing_ids(self, source_ids: list[str]) -> set[str]:
        with session_scope() as session:
            rows = session.execute(
                select(Video.source_id).where(
                    Video.source == self.name, Video.source_id.in_(source_ids)
                )
            ).scalars().all()
        return set(rows)

    def _insert(self, candidate: VideoCandidate) -> int | None:
        """Insert a discovered candidate. Returns its id, or None if it already exists."""
        try:
            with session_scope() as session:
                video = Video(
                    source=candidate.source,
                    source_id=candidate.source_id,
                    page_url=candidate.page_url,
                    download_url=candidate.download_url,
                    author=candidate.author,
                    author_url=candidate.author_url,
                    license=candidate.license,
                    license_url=candidate.license_url,
                    title=candidate.title,
                    tags=list(candidate.tags),
                    duration_s=candidate.duration_s,
                    width=candidate.width,
                    height=candidate.height,
                    status=VideoStatus.DISCOVERED,
                )
                session.add(video)
                session.flush()
                return video.id
        except IntegrityError:
            # Another task inserted the same (source, source_id) between the pre-check
            # and here — the unique constraint is what actually settles the race.
            return None

    async def _download(self, video_id: int, candidate: VideoCandidate, stats: RunStats) -> None:
        target = video_path(self.cfg, None, candidate.source, candidate.source_id)
        max_bytes = self.cfg.storage.max_file_size * 1024 * 1024

        async with self._semaphore:
            await self._bucket.acquire()
            try:
                async with self.http.get(candidate.download_url) as response:
                    response.raise_for_status()
                    written = 0
                    with atomic_write(target) as partial:
                        with open(partial, "wb") as handle:
                            async for chunk in response.content.iter_chunked(_CHUNK):
                                written += len(chunk)
                                if written > max_bytes:
                                    raise ValueError(
                                        f"exceeds max_file_size ({self.cfg.storage.max_file_size} MB)"
                                    )
                                handle.write(chunk)
            except (aiohttp.ClientError, ValueError, OSError) as exc:
                stats.errors += 1
                stats.reasons["download_failed"] += 1
                logger.error(f"{self.name}: download failed for {candidate.page_url}: {exc}")
                self._mark_rejected(video_id, "download_failed")
                return

        with session_scope() as session:
            video = session.get(Video, video_id)
            video.file_path = str(target)
            video.size_bytes = target.stat().st_size
            video.status = VideoStatus.DOWNLOADED
        stats.downloaded += 1
        logger.info(f"{self.name}: downloaded {candidate.page_url} -> {target}")

    def _mark_rejected(self, video_id: int, reason: str) -> None:
        with session_scope() as session:
            video = session.get(Video, video_id)
            if video is not None:
                video.status = VideoStatus.REJECTED
                video.reject_reason = reason

    async def collect(self, query: str, limit: int) -> RunStats:
        """Search, insert new candidates and download them concurrently."""
        stats = RunStats()
        tasks: list[asyncio.Task] = []

        async for candidate in self.search(query, limit):
            stats.found += 1
            if self._existing_ids([candidate.source_id]):
                logger.debug(f"{self.name}: {candidate.source_id} already known")
                stats.skip("already_known")
                continue

            video_id = self._insert(candidate)
            if video_id is None:
                stats.skip("already_known")
                continue

            tasks.append(asyncio.create_task(self._download(video_id, candidate, stats)))

        if tasks:
            await asyncio.gather(*tasks)
        return stats
