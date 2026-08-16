"""Tests for the Pixabay collector. No real network.

Openverse is not implemented here: its public API only covers images and audio,
there is no video search endpoint (see docs.openverse.org) — flagged to lead.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path

from src.collectors import COLLECTORS
from src.collectors.pixabay import PixabayCollector, _best_rendition
from src.storage.db import session_scope
from src.storage.models import Video, VideoStatus
from tests.fakes import FakeResponse, FakeSession

SEARCH_URL = "https://pixabay.com/api/videos/"
CDN_URL = "https://cdn.pixabay.com/video/000/125_large.mp4"
MP4 = b"\x00\x00\x00\x18ftypmp42" + b"x" * 512


def _pixabay_payload(count: int = 1) -> dict:
    return {
        "total": count,
        "totalHits": count,
        "hits": [
            {
                "id": 125 + i,
                "pageURL": f"https://pixabay.com/videos/id-{125 + i}/",
                "tags": "flowers, yellow, blossom",
                "duration": 12,
                "user": "Coverr-Free-Footage",
                "user_id": 1281706,
                "videos": {
                    "large": {"url": f"https://cdn.pixabay.com/video/000/{125 + i}_large.mp4",
                              "width": 1920, "height": 1080},
                    "medium": {"url": f"https://cdn.pixabay.com/video/000/{125 + i}_medium.mp4",
                               "width": 1280, "height": 720},
                    "small": {"url": f"https://cdn.pixabay.com/video/000/{125 + i}_small.mp4",
                              "width": 640, "height": 360},
                    "tiny": {"url": f"https://cdn.pixabay.com/video/000/{125 + i}_tiny.mp4",
                             "width": 480, "height": 270},
                },
            }
            for i in range(count)
        ],
    }


def _enable_pixabay(cfg, rate_limit: int = 3_600_000):
    collectors = dict(cfg.collectors)
    collectors["pixabay"] = replace(
        collectors["pixabay"], enabled=True, api_key="k", rate_limit=rate_limit
    )
    return replace(cfg, collectors=collectors)


def _run(cfg, routes, coro_factory):
    http = FakeSession(routes)

    async def scenario():
        return await coro_factory(PixabayCollector(cfg, http))

    return asyncio.run(scenario()), http


async def _search(collector, query="cat", limit=5):
    return [candidate async for candidate in collector.search(query, limit)]


# --- registry -----------------------------------------------------------------


def test_pixabay_is_registered():
    assert COLLECTORS["pixabay"] is PixabayCollector


# --- parsing --------------------------------------------------------------


def test_best_rendition_picks_highest_resolution():
    videos = {
        "large": {"url": "l", "width": 1920, "height": 1080},
        "medium": {"url": "m", "width": 1280, "height": 720},
        "small": {"url": "", "width": 640, "height": 360},
    }
    assert _best_rendition(videos)["url"] == "l"


def test_best_rendition_skips_missing_urls():
    videos = {"large": {"url": "", "width": 1920, "height": 1080}, "medium": {"url": "m", "width": 1280, "height": 720}}
    assert _best_rendition(videos)["url"] == "m"


def test_best_rendition_returns_none_without_any_url():
    assert _best_rendition({"large": {"url": "", "width": 1920, "height": 1080}}) is None


def test_search_maps_license_and_attribution(tmp_cfg):
    candidates, _ = _run(
        _enable_pixabay(tmp_cfg), {SEARCH_URL: FakeResponse(payload=_pixabay_payload(1))}, _search
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.source == "pixabay"
    assert candidate.source_id == "125"
    assert candidate.download_url == CDN_URL
    assert candidate.author == "Coverr-Free-Footage"
    assert candidate.author_url == "https://pixabay.com/users/Coverr-Free-Footage-1281706/"
    assert candidate.license == "Pixabay License"
    assert candidate.license_url == "https://pixabay.com/service/license-summary/"
    assert (candidate.width, candidate.height) == (1920, 1080)
    assert candidate.duration_s == 12.0
    assert candidate.tags == ["flowers", "yellow", "blossom"]


def test_search_sends_the_api_key_as_query_param(tmp_cfg):
    _, http = _run(
        _enable_pixabay(tmp_cfg), {SEARCH_URL: FakeResponse(payload=_pixabay_payload(1))}, _search
    )
    assert http.requests[0][1]["params"]["key"] == "k"


def test_search_stops_at_limit(tmp_cfg):
    candidates, http = _run(
        _enable_pixabay(tmp_cfg),
        {SEARCH_URL: FakeResponse(payload=_pixabay_payload(5))},
        lambda collector: _search(collector, limit=2),
    )
    assert len(candidates) == 2
    assert len(http.requests) == 1  # no needless second page


def test_search_stops_when_the_api_returns_nothing(tmp_cfg):
    candidates, _ = _run(
        _enable_pixabay(tmp_cfg), {SEARCH_URL: FakeResponse(payload={"hits": []})}, _search
    )
    assert candidates == []


def test_search_skips_items_without_usable_rendition(tmp_cfg):
    payload = _pixabay_payload(1)
    payload["hits"][0]["videos"] = {"large": {"url": "", "width": 0, "height": 0}}
    candidates, _ = _run(
        _enable_pixabay(tmp_cfg), {SEARCH_URL: FakeResponse(payload=payload)}, _search
    )
    assert candidates == []


# --- collect() end to end ---------------------------------------------------


def test_collect_downloads_and_marks_downloaded(tmp_cfg, db):
    routes = {SEARCH_URL: FakeResponse(payload=_pixabay_payload(1)), CDN_URL: FakeResponse(body=MP4)}
    stats, _ = _run(_enable_pixabay(tmp_cfg), routes, lambda c: c.collect("cat", 1))

    assert (stats.found, stats.downloaded, stats.errors) == (1, 1, 0)
    with session_scope() as session:
        video = session.query(Video).one()
    assert video.status == VideoStatus.DOWNLOADED
    assert video.license == "Pixabay License"
    assert video.size_bytes == len(MP4)
    assert Path(video.file_path).read_bytes() == MP4


def test_collect_skips_already_known_source_id(tmp_cfg, db):
    with session_scope() as session:
        session.add(Video(source="pixabay", source_id="125", license="Pixabay License"))

    stats, http = _run(
        _enable_pixabay(tmp_cfg),
        {SEARCH_URL: FakeResponse(payload=_pixabay_payload(1))},
        lambda c: c.collect("cat", 1),
    )

    assert (stats.downloaded, stats.skipped) == (0, 1)
    assert stats.reasons["already_known"] == 1
    assert len(http.requests) == 1  # nothing downloaded
