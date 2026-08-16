"""Dataset export: COCO / WebDataset / HuggingFace `datasets`."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.storage.db import session_scope
from src.storage.models import Video, VideoStatus

if TYPE_CHECKING:
    from src.config import Config


def split_for(source: str, source_id: str, train_ratio: float) -> Literal["train", "val"]:
    """Deterministic train/val split: the same (source, source_id) always lands in the
    same split, so a single video never appears in both.
    """
    digest = hashlib.sha256(f"{source}:{source_id}".encode()).hexdigest()
    return "train" if int(digest, 16) % 100 < train_ratio * 100 else "val"


def select_export_videos(session: Session, include_unlicensed: bool) -> list[Video]:
    """Videos eligible for export: status=processed, license != "unknown" unless
    include_unlicensed is set.
    """
    stmt = select(Video).where(Video.status == VideoStatus.PROCESSED)
    if not include_unlicensed:
        stmt = stmt.where(Video.license != "unknown")
    return list(session.execute(stmt).scalars().all())


def run_export(
    cfg: "Config", fmt: str, output: Path, push: bool = False, include_unlicensed: bool = False
) -> Any:
    """CLI entry point: select DB-filtered videos and export them in `fmt`
    ("coco" | "webdataset" | "huggingface").
    """
    from src.export.coco import write_coco
    from src.export.hf import check_push_prerequisites, write_hf
    from src.export.webdataset import write_webdataset

    if fmt == "huggingface":
        check_push_prerequisites(push)

    with session_scope() as session:
        videos = select_export_videos(session, include_unlicensed)
        if not videos:
            scope = "status=processed" if include_unlicensed else "status=processed, license != unknown"
            raise ValueError(
                f"экспорт невозможен: 0 видео подходят под условие ({scope}); "
                "если ожидались видео с license=unknown — передайте --include-unlicensed"
            )

        if fmt == "coco":
            return write_coco(cfg, output, videos)
        if fmt == "webdataset":
            return write_webdataset(cfg, output, videos)
        if fmt == "huggingface":
            return write_hf(cfg, output, videos, push)
        raise ValueError(f"unknown export format: {fmt!r}")
