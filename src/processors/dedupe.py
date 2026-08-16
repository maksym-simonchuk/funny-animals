"""Deduplication: exact sha256 matches and near-duplicate detection via perceptual hashing."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import imagehash
from loguru import logger
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.storage.db import session_scope
from src.storage.models import Video, VideoStatus

if TYPE_CHECKING:
    from src.config import Config

_CHUNK_SIZE = 1024 * 1024  # 1 MB
_DEFAULT_MAX_DISTANCE = 5
_INACTIVE_STATUSES = (VideoStatus.REJECTED, VideoStatus.DELETED)


def sha256_file(path: Path) -> str:
    """Hash a file's contents in 1 MB chunks without loading it fully into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def phash_frames(frames: list[Path]) -> str:
    """Perceptual hash of up to 3 frames (first/middle/last), joined as "h1:h2:h3"."""
    if not frames:
        return ""
    if len(frames) <= 3:
        selected = frames
    else:
        selected = [frames[0], frames[len(frames) // 2], frames[-1]]

    hashes = [str(imagehash.phash(Image.open(frame))) for frame in selected]
    return ":".join(hashes)


def _component_distance(a: str, b: str) -> int:
    """Minimum Hamming distance across position-matched phash components (h1-h1, h2-h2, ...)."""
    ha = [imagehash.hex_to_hash(h) for h in a.split(":") if h]
    hb = [imagehash.hex_to_hash(h) for h in b.split(":") if h]
    if not ha or not hb:
        return _DEFAULT_MAX_DISTANCE + 1
    return min(x - y for x, y in zip(ha, hb))


def find_duplicate(
    session: Session,
    sha256: str,
    phash: str | None,
    max_distance: int = _DEFAULT_MAX_DISTANCE,
    exclude_id: int | None = None,
) -> Video | None:
    """Return an existing (non-rejected/deleted) Video that exactly or near-duplicates the given hashes.

    ``exclude_id`` skips one row: a video being re-hashed during processing already
    carries the sha256 that fetch stored for it, so without this it matches itself.
    """
    others = [Video.status.not_in(_INACTIVE_STATUSES)]
    if exclude_id is not None:
        others.append(Video.id != exclude_id)

    exact = session.execute(
        select(Video).where(Video.sha256 == sha256, *others)
    ).scalar_one_or_none()
    if exact is not None:
        return exact

    if not phash:
        return None

    candidates = session.execute(
        select(Video).where(Video.phash.is_not(None), *others)
    ).scalars().all()

    best: Video | None = None
    best_distance = max_distance + 1
    # ponytail: linear scan over phash candidates, switch to BK-tree above ~50k rows
    for candidate in candidates:
        distance = _component_distance(phash, candidate.phash)
        if distance <= max_distance and distance < best_distance:
            best = candidate
            best_distance = distance
    return best


def run_dedupe(cfg: "Config", method: str) -> dict[str, int]:
    """Compare all active (non-rejected/deleted) videos pairwise and reject losing duplicates.

    method="hash": exact sha256 match only. method="phash": near-duplicate via perceptual
    hash (component-wise Hamming distance <= 5). The winner is the record with the larger
    width*height; the loser is marked status=rejected, reject_reason="duplicate".
    """
    checked = 0
    duplicates = 0

    with session_scope() as session:
        rows = session.execute(
            select(Video).where(Video.status.not_in(_INACTIVE_STATUSES))
        ).scalars().all()

        kept: list[Video] = []
        for candidate in rows:
            checked += 1
            match: Video | None = None

            # ponytail: linear scan over kept candidates, switch to BK-tree above ~50k rows
            for existing in kept:
                if method == "hash":
                    is_dup = candidate.sha256 is not None and candidate.sha256 == existing.sha256
                else:
                    is_dup = bool(candidate.phash) and bool(existing.phash) and (
                        _component_distance(candidate.phash, existing.phash) <= _DEFAULT_MAX_DISTANCE
                    )
                if is_dup:
                    match = existing
                    break

            if match is None:
                kept.append(candidate)
                continue

            duplicates += 1
            cand_area = (candidate.width or 0) * (candidate.height or 0)
            match_area = (match.width or 0) * (match.height or 0)
            if cand_area > match_area:
                logger.info(f"duplicate: video {match.id} rejected in favor of {candidate.id}")
                match.status = VideoStatus.REJECTED
                match.reject_reason = "duplicate"
                kept.remove(match)
                kept.append(candidate)
            else:
                logger.info(f"duplicate: video {candidate.id} rejected in favor of {match.id}")
                candidate.status = VideoStatus.REJECTED
                candidate.reject_reason = "duplicate"

    return {"checked": checked, "duplicates": duplicates}
