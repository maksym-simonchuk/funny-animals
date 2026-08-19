"""On-disk layout: data/videos/{category}/{source}/{source_id}.mp4 and friends."""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config
    from src.storage.models import Video

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe_id(raw: str) -> str:
    """Make an arbitrary source id usable as a filename component."""
    cleaned = _UNSAFE.sub("_", raw).strip("._-")
    return (cleaned or "unknown")[:120]


def ensure_dirs(cfg: "Config") -> None:
    """Create every directory the app writes to."""
    for path in (
        cfg.storage.video_path,
        cfg.storage.thumbnail_path,
        cfg.storage.frame_path,
        cfg.storage.log_path,
        cfg.storage.model_path,
    ):
        path.mkdir(parents=True, exist_ok=True)


def video_path(cfg: "Config", category: str | None, source: str, source_id: str) -> Path:
    return cfg.storage.video_path / (category or "unsorted") / safe_id(source) / (
        f"{safe_id(source_id)}.mp4"
    )


def frames_dir(cfg: "Config", video_id: int) -> Path:
    return cfg.storage.frame_path / str(video_id)


def thumb_path(cfg: "Config", video_id: int, suffix: str = ".gif") -> Path:
    return cfg.storage.thumbnail_path / f"{video_id}{suffix}"


@contextmanager
def atomic_write(target: Path) -> Iterator[Path]:
    """Yield a ``.part`` path, then move it into place; remove it on failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    try:
        yield partial
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    if not partial.exists():
        raise FileNotFoundError(f"nothing was written to {partial}")
    os.replace(partial, target)


def remove_quietly(path: str | Path | None) -> bool:
    """Delete a file (or a directory tree) if it exists. Returns True if removed."""
    if not path:
        return False
    target = Path(path)
    if target.is_dir():
        for child in sorted(target.rglob("*"), reverse=True):
            child.rmdir() if child.is_dir() else child.unlink(missing_ok=True)
        target.rmdir()
        return True
    if target.exists():
        target.unlink()
        return True
    return False


def drop(cfg: "Config", video: "Video") -> None:
    """Everything on disk for one video -- the file and its frames -- and the row stops
    pointing at what is not there any more."""
    remove_quietly(video.file_path)
    remove_quietly(frames_dir(cfg, video.id))
    video.file_path = None


def is_within(child: Path, parent: Path) -> bool:
    """Path-traversal guard for anything that serves files by database path."""
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True
