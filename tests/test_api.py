"""API tests. Uses tmp_cfg/db fixtures from tests/conftest.py. No network."""
from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import create_app
from src.storage.db import session_scope
from src.storage.files import frames_dir
from src.storage.models import Video, VideoStatus


def _insert_video(**overrides) -> int:
    defaults = dict(
        source="pexels",
        source_id="vid-1",
        page_url="https://example.com/p",
        download_url="https://example.com/d",
        license="cc0",
        status=VideoStatus.PROCESSED,
    )
    defaults.update(overrides)
    with session_scope() as session:
        video = Video(**defaults)
        session.add(video)
        session.flush()
        return video.id


def test_stats_empty_db_returns_zeros(tmp_cfg, db):
    client = TestClient(create_app(tmp_cfg))

    response = client.get("/stats")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 0


def test_videos_pagination_and_filters(tmp_cfg, db):
    _insert_video(source="pexels", source_id="a", category="cat", duration_s=10.0, has_animal=True)
    _insert_video(source="pexels", source_id="b", category="dog", duration_s=20.0, has_animal=True)
    _insert_video(source="pixabay", source_id="c", category="cat", duration_s=30.0, has_animal=False)
    client = TestClient(create_app(tmp_cfg))

    all_videos = client.get("/videos")
    assert all_videos.status_code == 200
    assert all_videos.json()["total"] == 3

    by_source = client.get("/videos", params={"source": "pexels"})
    assert by_source.json()["total"] == 2

    by_category = client.get("/videos", params={"category": "cat"})
    assert by_category.json()["total"] == 2

    by_duration = client.get("/videos", params={"min_duration": 15, "max_duration": 25})
    assert by_duration.json()["total"] == 1
    assert by_duration.json()["items"][0]["source_id"] == "b"

    by_animal = client.get("/videos", params={"has_animal": False})
    assert by_animal.json()["total"] == 1
    assert by_animal.json()["items"][0]["source_id"] == "c"

    page1 = client.get("/videos", params={"page": 1, "per_page": 2})
    page2 = client.get("/videos", params={"page": 2, "per_page": 2})
    assert len(page1.json()["items"]) == 2
    assert len(page2.json()["items"]) == 1
    assert page1.json()["page"] == 1
    assert page1.json()["per_page"] == 2


def test_get_video_not_found_404(tmp_cfg, db):
    client = TestClient(create_app(tmp_cfg))

    response = client.get("/videos/9999")

    assert response.status_code == 404


def test_get_video_deleted_is_404(tmp_cfg, db):
    video_id = _insert_video(source_id="deleted-1", status=VideoStatus.DELETED, file_path=None)
    client = TestClient(create_app(tmp_cfg))

    response = client.get(f"/videos/{video_id}")

    assert response.status_code == 404


def test_video_file_path_traversal_is_404(tmp_cfg, db):
    outside = tmp_cfg.storage.video_path.parent / "outside" / "evil.mp4"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_bytes(b"nope")
    video_id = _insert_video(source_id="traversal-1", file_path=str(outside))
    client = TestClient(create_app(tmp_cfg))

    response = client.get(f"/videos/{video_id}/file")

    assert response.status_code == 404


def test_video_file_missing_on_disk_is_404(tmp_cfg, db):
    missing = tmp_cfg.storage.video_path / "pexels" / "missing.mp4"
    video_id = _insert_video(source_id="missing-1", file_path=str(missing))
    client = TestClient(create_app(tmp_cfg))

    response = client.get(f"/videos/{video_id}/file")

    assert response.status_code == 404


def test_video_file_serves_when_valid(tmp_cfg, db):
    target = tmp_cfg.storage.video_path / "pexels" / "ok.mp4"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"video-bytes")
    video_id = _insert_video(source_id="ok-1", file_path=str(target))
    client = TestClient(create_app(tmp_cfg))

    response = client.get(f"/videos/{video_id}/file")

    assert response.status_code == 200
    assert response.content == b"video-bytes"


def test_delete_erases_files_and_marks_deleted(tmp_cfg, db):
    video_file = tmp_cfg.storage.video_path / "pexels" / "del.mp4"
    thumb_file = tmp_cfg.storage.thumbnail_path / "del.jpg"
    for f in (video_file, thumb_file):
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"x")
    video_id = _insert_video(
        source_id="del-1", file_path=str(video_file), thumb_path=str(thumb_file)
    )
    frame_dir = frames_dir(tmp_cfg, video_id)
    frame_dir.mkdir(parents=True, exist_ok=True)
    (frame_dir / "frame_0.jpg").write_bytes(b"x")
    client = TestClient(create_app(tmp_cfg))

    response = client.delete(f"/videos/{video_id}")

    assert response.status_code in (200, 204)
    assert not video_file.exists()
    assert not thumb_file.exists()
    assert not frame_dir.exists()
    with session_scope() as session:
        row = session.get(Video, video_id)
        assert row.status == VideoStatus.DELETED
        assert row.file_path is None


def test_delete_is_idempotent(tmp_cfg, db):
    video_id = _insert_video(source_id="del-2", file_path=None)
    client = TestClient(create_app(tmp_cfg))

    first = client.delete(f"/videos/{video_id}")
    second = client.delete(f"/videos/{video_id}")

    assert first.status_code in (200, 204)
    assert second.status_code in (200, 204)
