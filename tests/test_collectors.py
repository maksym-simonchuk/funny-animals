"""Tests for the collector base machinery and the Pexels collector. No real network."""
from __future__ import annotations

import asyncio
import time
from dataclasses import replace
from pathlib import Path

import pytest

from src.collectors import COLLECTORS, get_enabled, run_collect
from src.collectors.base import BaseCollector, RunStats, TokenBucket, VideoCandidate
from src.collectors.pexels import PexelsCollector, _best_file
from src.storage.db import session_scope
from src.storage.models import Video, VideoStatus
from tests.fakes import FakeResponse, FakeSession

SEARCH_URL = "https://api.pexels.com/videos/search"
CDN_URL = "https://cdn.example/1000-hd.mp4"
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"x" * 512


def _pexels_payload(count: int = 1, next_page: str | None = None) -> dict:
    return {
        "videos": [
            {
                "id": 1000 + i,
                "url": f"https://www.pexels.com/video/cat-{1000 + i}/",
                "duration": 12,
                "user": {"name": "Jane Doe", "url": "https://www.pexels.com/@janedoe"},
                "alt": "a cat on a sofa",
                "video_files": [
                    {"file_type": "video/mp4", "link": f"https://cdn.example/{1000 + i}-sd.mp4",
                     "width": 640, "height": 360},
                    {"file_type": "video/mp4", "link": f"https://cdn.example/{1000 + i}-hd.mp4",
                     "width": 1920, "height": 1080},
                ],
            }
            for i in range(count)
        ],
        "next_page": next_page,
    }


def _enable_pexels(cfg, rate_limit: int = 3_600_000):
    # The default rate_limit is deliberately huge: TokenBucket is covered by its own
    # tests, and a realistic 200/hour would make every collect() test sleep 18 s.
    collectors = dict(cfg.collectors)
    collectors["pexels"] = replace(
        collectors["pexels"], enabled=True, api_key="k", rate_limit=rate_limit
    )
    return replace(cfg, collectors=collectors)


def _run(cfg, routes, coro_factory):
    """Run `coro_factory(collector)` against a FakeSession wired with `routes`."""
    http = FakeSession(routes)

    async def scenario():
        return await coro_factory(PexelsCollector(cfg, http))

    return asyncio.run(scenario()), http


# --- TokenBucket ------------------------------------------------------------


def test_token_bucket_spaces_out_acquires():
    async def scenario():
        bucket = TokenBucket(rate_per_hour=180_000)  # one every 20 ms
        started = time.monotonic()
        for _ in range(3):
            await bucket.acquire()
        return time.monotonic() - started

    elapsed = asyncio.run(scenario())
    assert elapsed >= 0.04  # the first acquire is free, the next two wait 20 ms each


def test_token_bucket_zero_rate_never_waits():
    async def scenario():
        bucket = TokenBucket(rate_per_hour=0)
        await bucket.acquire()
        await bucket.acquire()

    asyncio.run(scenario())  # must not hang


# --- registry ---------------------------------------------------------------


def test_pexels_is_registered():
    assert COLLECTORS["pexels"] is PexelsCollector


def test_get_enabled_returns_only_enabled_collectors(tmp_cfg):
    assert get_enabled(tmp_cfg) == []
    assert get_enabled(_enable_pexels(tmp_cfg)) == [PexelsCollector]


def test_run_collect_rejects_unknown_source(tmp_cfg):
    with pytest.raises(ValueError, match="unknown source"):
        asyncio.run(run_collect(tmp_cfg, "nope", "cat", 5))


def test_run_collect_rejects_disabled_source(tmp_cfg):
    with pytest.raises(ValueError, match="disabled"):
        asyncio.run(run_collect(tmp_cfg, "pexels", "cat", 5))


# --- pexels parsing ---------------------------------------------------------


def test_best_file_picks_highest_resolution_mp4():
    files = [
        {"file_type": "video/mp4", "link": "sd", "width": 640, "height": 360},
        {"file_type": "video/mp4", "link": "hd", "width": 1920, "height": 1080},
        {"file_type": "video/webm", "link": "webm", "width": 3840, "height": 2160},
    ]
    assert _best_file(files)["link"] == "hd"


def test_best_file_returns_none_without_mp4():
    assert _best_file([{"file_type": "video/webm", "link": "w"}]) is None


async def _search(collector, query="cat", limit=5):
    return [candidate async for candidate in collector.search(query, limit)]


def test_search_skips_items_without_mp4(tmp_cfg):
    payload = _pexels_payload(1)
    payload["videos"][0]["video_files"] = [{"file_type": "video/webm", "link": "w"}]
    candidates, _ = _run(
        _enable_pexels(tmp_cfg), {SEARCH_URL: FakeResponse(payload=payload)}, _search
    )
    assert candidates == []


def test_search_maps_license_and_attribution(tmp_cfg):
    candidates, _ = _run(
        _enable_pexels(tmp_cfg), {SEARCH_URL: FakeResponse(payload=_pexels_payload(1))}, _search
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source == "pexels"
    assert candidate.source_id == "1000"
    assert candidate.download_url == CDN_URL
    assert candidate.author == "Jane Doe"
    assert candidate.author_url == "https://www.pexels.com/@janedoe"
    assert candidate.license == "Pexels License"
    assert candidate.license_url == "https://www.pexels.com/license/"
    assert (candidate.width, candidate.height) == (1920, 1080)
    assert candidate.duration_s == 12.0


def test_search_sends_the_api_key_as_authorization(tmp_cfg):
    _, http = _run(
        _enable_pexels(tmp_cfg), {SEARCH_URL: FakeResponse(payload=_pexels_payload(1))}, _search
    )
    assert http.requests[0][1]["headers"]["Authorization"] == "k"


def test_search_stops_at_limit(tmp_cfg):
    candidates, http = _run(
        _enable_pexels(tmp_cfg),
        {SEARCH_URL: FakeResponse(payload=_pexels_payload(5))},
        lambda collector: _search(collector, limit=2),
    )
    assert len(candidates) == 2
    assert len(http.requests) == 1  # no needless second page


def test_search_stops_when_the_api_returns_nothing(tmp_cfg):
    candidates, _ = _run(
        _enable_pexels(tmp_cfg), {SEARCH_URL: FakeResponse(payload={"videos": []})}, _search
    )
    assert candidates == []


def test_search_retries_once_after_429(tmp_cfg, monkeypatch):
    import src.collectors.base as base

    slept: list[float] = []

    async def fake_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr(base.asyncio, "sleep", fake_sleep)
    routes = {
        SEARCH_URL: [
            FakeResponse(status=429, headers={"Retry-After": "7"}),
            FakeResponse(payload=_pexels_payload(1)),
        ]
    }
    candidates, http = _run(_enable_pexels(tmp_cfg), routes, _search)

    assert len(candidates) == 1
    assert len(http.requests) == 2
    assert 7.0 in slept  # Retry-After honoured rather than a hardcoded backoff


# --- collect() end to end ---------------------------------------------------


def test_collect_downloads_and_marks_downloaded(tmp_cfg, db):
    routes = {SEARCH_URL: FakeResponse(payload=_pexels_payload(1)), CDN_URL: FakeResponse(body=MP4)}
    stats, _ = _run(_enable_pexels(tmp_cfg), routes, lambda c: c.collect("cat", 1))

    assert (stats.found, stats.downloaded, stats.errors) == (1, 1, 0)
    with session_scope() as session:
        video = session.query(Video).one()
    assert video.status == VideoStatus.DOWNLOADED
    assert video.license == "Pexels License"
    assert video.size_bytes == len(MP4)
    assert Path(video.file_path).read_bytes() == MP4


def test_collect_skips_already_known_source_id(tmp_cfg, db):
    with session_scope() as session:
        session.add(Video(source="pexels", source_id="1000", license="Pexels License"))

    stats, http = _run(
        _enable_pexels(tmp_cfg),
        {SEARCH_URL: FakeResponse(payload=_pexels_payload(1))},
        lambda c: c.collect("cat", 1),
    )

    assert (stats.downloaded, stats.skipped) == (0, 1)
    assert stats.reasons["already_known"] == 1
    assert len(http.requests) == 1  # nothing downloaded


def test_collect_marks_rejected_when_download_fails(tmp_cfg, db):
    cfg = _enable_pexels(tmp_cfg)
    routes = {SEARCH_URL: FakeResponse(payload=_pexels_payload(1)), CDN_URL: FakeResponse(status=500)}
    stats, _ = _run(cfg, routes, lambda c: c.collect("cat", 1))

    assert stats.errors == 1
    with session_scope() as session:
        video = session.query(Video).one()
    assert video.status == VideoStatus.REJECTED
    assert video.reject_reason == "download_failed"
    assert video.file_path is None
    assert not list(cfg.storage.video_path.rglob("*.part"))  # no half-written leftovers


def test_collect_rejects_file_over_max_size(tmp_cfg, db):
    cfg = replace(_enable_pexels(tmp_cfg), storage=replace(tmp_cfg.storage, max_file_size=0))
    routes = {SEARCH_URL: FakeResponse(payload=_pexels_payload(1)), CDN_URL: FakeResponse(body=MP4)}
    stats, _ = _run(cfg, routes, lambda c: c.collect("cat", 1))

    assert stats.errors == 1
    with session_scope() as session:
        assert session.query(Video).one().status == VideoStatus.REJECTED
    assert not list(cfg.storage.video_path.rglob("*.part"))


# --- plumbing ---------------------------------------------------------------


def test_run_stats_merge_sums_counters():
    a = RunStats(found=1, downloaded=1)
    a.skip("x")
    b = RunStats(found=2, errors=1)
    b.skip("x")
    b.skip("y")
    a.merge(b)
    assert (a.found, a.downloaded, a.skipped, a.errors) == (3, 1, 3, 1)
    assert a.reasons == {"x": 2, "y": 1}


def test_video_candidate_defaults_to_unknown_license():
    candidate = VideoCandidate(source="s", source_id="1", page_url="p", download_url="d")
    assert candidate.license == "unknown"


def test_base_collector_search_is_abstract():
    assert BaseCollector.search.__isabstractmethod__
