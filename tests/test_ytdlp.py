"""Tests for src.collectors.ytdlp — yt-dlp is mocked, nothing touches the network."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from yt_dlp.utils import DownloadError

from src.collectors import ytdlp
from src.storage.db import session_scope
from src.storage.models import BROWSER_SOURCE, Run, Video, VideoStatus


class FakeYoutubeDL:
    """Stands in for yt_dlp.YoutubeDL: copies a fixture file into the configured outtmpl dir."""

    source_file: Path
    calls: list[str] = []
    fail_times: int = 0
    error_message: str = "network error"

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def extract_info(self, url, download=True):
        FakeYoutubeDL.calls.append(url)
        if FakeYoutubeDL.fail_times > 0:
            FakeYoutubeDL.fail_times -= 1
            raise DownloadError(FakeYoutubeDL.error_message)

        out_dir = Path(self.opts["outtmpl"]).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        video_id = f"vid{len(FakeYoutubeDL.calls)}"
        target = out_dir / f"{video_id}.mp4"
        shutil.copy(FakeYoutubeDL.source_file, target)
        return {
            "id": video_id,
            "extractor": "Instagram",
            "webpage_url": url,
            "url": f"{url}/media.mp4",
            "uploader": "someone",
            "uploader_url": "https://example.com/someone",
            "title": "a cat",
            "tags": ["cat"],
            "requested_downloads": [{"filepath": str(target)}],
        }


@pytest.fixture
def fake_ydl(monkeypatch, sample_video):
    FakeYoutubeDL.calls = []
    FakeYoutubeDL.fail_times = 0
    FakeYoutubeDL.error_message = "network error"
    FakeYoutubeDL.source_file = sample_video
    monkeypatch.setattr(ytdlp, "YoutubeDL", FakeYoutubeDL)
    monkeypatch.setattr(ytdlp.time, "sleep", lambda _: None)
    return FakeYoutubeDL


def _urls_file(tmp_path: Path, *urls: str) -> Path:
    path = tmp_path / "urls.txt"
    path.write_text("\n".join(urls) + "\n")
    return path


def test_read_urls_ignores_blanks_comments_and_duplicates(tmp_path):
    path = tmp_path / "urls.txt"
    path.write_text(
        "https://a.example/1\n"
        "\n"
        "# a comment\n"
        "https://a.example/2  # trailing comment\n"
        "https://a.example/1\n"
    )
    assert ytdlp.read_urls(path, 10) == ["https://a.example/1", "https://a.example/2"]


def test_read_urls_respects_limit(tmp_path):
    path = _urls_file(tmp_path, "https://a/1", "https://a/2", "https://a/3")
    assert ytdlp.read_urls(path, 2) == ["https://a/1", "https://a/2"]


def test_read_urls_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ytdlp.read_urls(tmp_path / "nope.txt", 10)


def test_fetch_downloads_and_persists_metadata(tmp_cfg, db, fake_ydl, tmp_path):
    stats = ytdlp.run_fetch(tmp_cfg, _urls_file(tmp_path, "https://insta.example/reel/1"), 10)

    assert (stats.found, stats.downloaded, stats.skipped, stats.errors) == (1, 1, 0, 0)
    with session_scope() as session:
        video = session.query(Video).one()
    assert video.source == "instagram"
    assert video.source_id == "vid1"
    assert video.page_url == "https://insta.example/reel/1"
    assert video.author == "someone"
    assert video.author_url == "https://example.com/someone"
    assert video.license == "unknown"
    assert video.status == VideoStatus.DOWNLOADED
    assert video.sha256 and len(video.sha256) == 64
    assert Path(video.file_path).is_file()
    assert Path(video.file_path).parent == tmp_cfg.storage.video_path / "unsorted" / "instagram"


def test_fetch_skips_video_shorter_than_min_duration(tmp_cfg, db, fake_ydl, tmp_path, short_video):
    fake_ydl.source_file = short_video
    stats = ytdlp.run_fetch(tmp_cfg, _urls_file(tmp_path, "https://insta.example/reel/short"), 10)

    assert (stats.downloaded, stats.skipped) == (0, 1)
    assert stats.reasons["duration"] == 1
    with session_scope() as session:
        assert session.query(Video).count() == 0


def test_fetch_skips_video_longer_than_max_duration(tmp_cfg, db, fake_ydl, tmp_path, long_video):
    fake_ydl.source_file = long_video
    stats = ytdlp.run_fetch(tmp_cfg, _urls_file(tmp_path, "https://insta.example/reel/long"), 10)

    assert stats.reasons["duration"] == 1
    with session_scope() as session:
        assert session.query(Video).count() == 0


def test_fetch_skips_identical_file_hash(tmp_cfg, db, fake_ydl, tmp_path):
    # Two different URLs resolving to byte-identical files: the second is a duplicate.
    urls = _urls_file(tmp_path, "https://insta.example/reel/1", "https://insta.example/reel/2")
    stats = ytdlp.run_fetch(tmp_cfg, urls, 10)

    assert (stats.downloaded, stats.skipped) == (1, 1)
    assert stats.reasons["duplicate"] == 1
    with session_scope() as session:
        assert session.query(Video).count() == 1


def test_fetch_skips_url_already_in_database(tmp_cfg, db, fake_ydl, tmp_path):
    urls = _urls_file(tmp_path, "https://insta.example/reel/1")
    ytdlp.run_fetch(tmp_cfg, urls, 10)
    fake_ydl.calls = []

    stats = ytdlp.run_fetch(tmp_cfg, urls, 10)

    assert stats.reasons["already_fetched"] == 1
    assert fake_ydl.calls == []  # the pre-check must fire before any download


def test_fetch_retries_transient_failure_then_succeeds(tmp_cfg, db, fake_ydl, tmp_path):
    fake_ydl.fail_times = 2  # retries defaults to 3

    stats = ytdlp.run_fetch(tmp_cfg, _urls_file(tmp_path, "https://insta.example/reel/1"), 10)

    assert stats.downloaded == 1
    assert len(fake_ydl.calls) == 3


def test_fetch_does_not_retry_permanent_failure(tmp_cfg, db, fake_ydl, tmp_path):
    fake_ydl.fail_times = 99
    fake_ydl.error_message = "Video unavailable: this post is private"

    stats = ytdlp.run_fetch(tmp_cfg, _urls_file(tmp_path, "https://insta.example/reel/x"), 10)

    assert stats.errors == 1
    assert len(fake_ydl.calls) == 1


def test_fetch_continues_after_broken_url(tmp_cfg, db, fake_ydl, tmp_path):
    fake_ydl.fail_times = 3  # exhausts the retries of the first URL only
    urls = _urls_file(tmp_path, "https://insta.example/broken", "https://insta.example/reel/2")

    stats = ytdlp.run_fetch(tmp_cfg, urls, 10)

    assert (stats.found, stats.downloaded, stats.errors) == (2, 1, 1)


def test_fetch_records_a_run_row(tmp_cfg, db, fake_ydl, tmp_path):
    ytdlp.run_fetch(tmp_cfg, _urls_file(tmp_path, "https://insta.example/reel/1"), 10)

    with session_scope() as session:
        run = session.query(Run).one()
    assert run.command == "fetch"
    assert (run.found, run.downloaded, run.skipped, run.errors) == (1, 1, 0, 0)
    assert run.finished_at is not None


def test_fetch_pauses_between_requests(tmp_cfg, db, fake_ydl, tmp_path, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr(ytdlp.time, "sleep", slept.append)
    urls = _urls_file(tmp_path, "https://insta.example/reel/1", "https://insta.example/reel/2")

    ytdlp.run_fetch(tmp_cfg, urls, 10)

    assert len(slept) == 1  # a pause between the two requests, none before the first
    assert 3.0 <= slept[0] <= 5.0


def test_ydl_opts_carry_no_login_or_proxy(tmp_cfg, tmp_path):
    opts = ytdlp._ydl_opts(tmp_cfg, tmp_path)
    assert opts["cookiefile"] is None
    assert opts["proxy"] is None
    assert opts["noplaylist"] is True
    assert opts["max_filesize"] == tmp_cfg.storage.max_file_size * 1024 * 1024


def _enqueue(url: str, source_id: str) -> int:
    """Insert the row the browser extension would have created."""
    with session_scope() as session:
        video = Video(
            source=BROWSER_SOURCE,
            source_id=source_id,
            page_url=url,
            license="unknown",
            status=VideoStatus.DISCOVERED,
        )
        session.add(video)
        session.flush()
        return video.id


def test_fetch_from_queue_updates_the_queued_row_in_place(tmp_cfg, db, fake_ydl):
    queue_id = _enqueue("https://www.instagram.com/reel/abcde/", "abcde")

    stats = ytdlp.run_fetch_from_queue(tmp_cfg, 10, watch=False)

    assert (stats.found, stats.downloaded, stats.errors) == (1, 1, 0)
    with session_scope() as session:
        video = session.query(Video).one()
    assert video.id == queue_id  # updated, not duplicated
    assert (video.source, video.source_id) == (BROWSER_SOURCE, "abcde")  # how it was found
    assert video.status == VideoStatus.DOWNLOADED
    assert Path(video.file_path).is_file()


def test_fetch_from_queue_rejects_a_permanently_broken_url(tmp_cfg, db, fake_ydl):
    # Otherwise --watch would pick the same dead URL up on every poll.
    fake_ydl.fail_times = 99
    fake_ydl.error_message = "Video unavailable: this post is private"
    _enqueue("https://www.instagram.com/reel/gone1/", "gone1")

    ytdlp.run_fetch_from_queue(tmp_cfg, 10, watch=False)

    with session_scope() as session:
        video = session.query(Video).one()
    assert video.status == VideoStatus.REJECTED
    assert video.reject_reason == "error"
    assert ytdlp._queue_batch(10) == []


def test_fetch_from_queue_on_empty_queue_does_nothing(tmp_cfg, db, fake_ydl):
    stats = ytdlp.run_fetch_from_queue(tmp_cfg, 10, watch=False)

    assert (stats.found, stats.downloaded) == (0, 0)
    assert fake_ydl.calls == []
    with session_scope() as session:
        assert session.query(Run).count() == 0
