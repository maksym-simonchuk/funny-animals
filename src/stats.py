"""Dataset stats aggregation and pruning of rejected videos' files."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger
from sqlalchemy import func, select

from src.storage.db import session_scope
from src.storage.files import frames_dir, remove_quietly
from src.storage.models import Video, VideoStatus

if TYPE_CHECKING:
    from src.config import Config


def collect_stats(by_category: bool = False, by_source: bool = False) -> dict[str, Any]:
    """Aggregate video counts from the database. Requires init_db() to have run already."""
    with session_scope() as session:
        total = session.scalar(select(func.count()).select_from(Video)) or 0
        by_status = dict(session.execute(select(Video.status, func.count()).group_by(Video.status)).all())
        result: dict[str, Any] = {"total": total, "by_status": by_status}
        if by_category:
            result["by_category"] = dict(
                session.execute(select(Video.category, func.count()).group_by(Video.category)).all()
            )
        if by_source:
            result["by_source"] = dict(
                session.execute(select(Video.source, func.count()).group_by(Video.source)).all()
            )
        return result


def prune_rejected(cfg: "Config", sharpness_below: float = 0.0) -> int:
    """Delete video and frames for status=rejected videos, clear file_path. Returns count pruned.

    A low_sharpness reject is kept unless it scores under `sharpness_below`: min_sharpness is a
    config knob, and a clip sitting just under it comes back the moment that knob moves. Every
    other gate answers yes or no, so those files are dead weight whatever the config says.
    """
    with session_scope() as session:
        rows = (
            session.execute(
                select(Video).where(Video.status == VideoStatus.REJECTED, Video.file_path.is_not(None))
            )
            .scalars()
            .all()
        )
        count = 0
        for video in rows:
            if video.reject_reason == "low_sharpness" and (video.sharpness or 0.0) >= sharpness_below:
                continue
            path = Path(video.file_path)
            remove_quietly(path)
            remove_quietly(frames_dir(cfg, video.id))
            video.file_path = None
            count += 1
            logger.info(f"pruned video id={video.id} path={path}")
        return count
