"""Tests for src.storage: db init, model constraints, atomic file writes, path helpers."""
from __future__ import annotations

import pytest
from sqlalchemy import inspect
from sqlalchemy.exc import IntegrityError

from src.storage.db import init_db, session_scope
from src.storage.files import atomic_write, safe_id, video_path
from src.storage.models import Video, VideoStatus


def test_init_db_creates_tables(tmp_cfg):
    engine = init_db(tmp_cfg.storage.database)
    tables = set(inspect(engine).get_table_names())
    assert {"videos", "detections", "runs"} <= tables


def test_unique_source_source_id_blocks_duplicate(db):
    with session_scope() as session:
        session.add(Video(source="pexels", source_id="1", page_url="p", download_url="d", license="cc0"))

    with pytest.raises(IntegrityError):
        with session_scope() as session:
            session.add(Video(source="pexels", source_id="1", page_url="p2", download_url="d2", license="cc0"))


def test_video_status_stored_as_string(db):
    with session_scope() as session:
        session.add(
            Video(
                source="pexels",
                source_id="2",
                page_url="p",
                download_url="d",
                license="cc0",
                status=VideoStatus.DOWNLOADED,
            )
        )
    with session_scope() as session:
        video = session.query(Video).filter_by(source_id="2").one()
        assert video.status == "downloaded"


def test_atomic_write_cleans_up_on_exception(tmp_path):
    target = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError):
        with atomic_write(target) as tmp_target:
            tmp_target.write_text("partial")
            raise RuntimeError("boom")
    assert not target.exists()
    assert not tmp_target.exists()


def test_atomic_write_replaces_on_success(tmp_path):
    target = tmp_path / "out.mp4"
    with atomic_write(target) as tmp_target:
        tmp_target.write_text("data")
    assert target.exists()
    assert target.read_text() == "data"
    assert not target.with_suffix(target.suffix + ".part").exists()


def test_video_path_unsorted_when_no_category(tmp_cfg):
    path = video_path(tmp_cfg, None, "pexels", "abc123")
    assert path == tmp_cfg.storage.video_path / "unsorted" / "pexels" / "abc123.mp4"


def test_safe_id_strips_unsafe_chars():
    result = safe_id("../../etc/passwd")
    assert "/" not in result
    assert all(c.isalnum() or c in "._-" for c in result)


def test_safe_id_caps_length():
    assert len(safe_id("a" * 200)) == 120
