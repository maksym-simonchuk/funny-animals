"""Tests for src.stats: counts, and which rejected files pruning is allowed to delete."""
from __future__ import annotations

from pathlib import Path

from src.stats import collect_stats, prune_rejected
from src.storage.db import session_scope
from src.storage.files import frames_dir
from src.storage.models import Video, VideoStatus


def _add(cfg, session, name: str, status: str, reason: str | None, sharpness: float | None) -> Video:
    path = cfg.storage.video_path / f"{name}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")
    video = Video(
        source="browser",
        source_id=name,
        page_url="p",
        license="unknown",
        file_path=str(path),
        status=status,
        reject_reason=reason,
        sharpness=sharpness,
    )
    session.add(video)
    session.flush()
    frame = frames_dir(cfg, video.id) / "0.jpg"
    frame.parent.mkdir(parents=True, exist_ok=True)
    frame.write_bytes(b"frame")
    return video


def test_collect_stats_counts_by_status(tmp_cfg, db):
    with session_scope() as session:
        _add(tmp_cfg, session, "a", VideoStatus.PROCESSED, None, 20.0)
        _add(tmp_cfg, session, "b", VideoStatus.REJECTED, "no_animal", 20.0)

    stats = collect_stats()
    assert stats["total"] == 2
    assert stats["by_status"] == {"processed": 1, "rejected": 1}


def test_prune_drops_every_reject(tmp_cfg, db):
    with session_scope() as session:
        kept = _add(tmp_cfg, session, "keep", VideoStatus.PROCESSED, None, 20.0)
        gone = _add(tmp_cfg, session, "compilation", VideoStatus.REJECTED, "compilation", 20.0)
        blurry = _add(tmp_cfg, session, "blurry", VideoStatus.REJECTED, "low_sharpness", 3.0)
        borderline = _add(tmp_cfg, session, "borderline", VideoStatus.REJECTED, "low_sharpness", 14.0)
        ids = {n: v.id for n, v in
               (("keep", kept), ("gone", gone), ("blurry", blurry), ("borderline", borderline))}
        paths = {n: Path(v.file_path) for n, v in
                 (("keep", kept), ("gone", gone), ("blurry", blurry), ("borderline", borderline))}

    assert prune_rejected(tmp_cfg) == 3

    assert paths["keep"].exists()
    assert not paths["gone"].exists()
    assert not paths["blurry"].exists()
    assert not paths["borderline"].exists()
    assert not frames_dir(tmp_cfg, ids["gone"]).exists()
    assert not frames_dir(tmp_cfg, ids["borderline"]).exists()

    with session_scope() as session:
        cleared = {v.source_id: v.file_path for v in session.query(Video).all()}
    assert cleared["compilation"] is None and cleared["blurry"] is None
    assert cleared["borderline"] is None and cleared["keep"] is not None
