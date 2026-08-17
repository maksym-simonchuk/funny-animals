"""Video processing pipeline: probe -> normalize -> keyframes -> quality -> dedupe -> detect."""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.compiler.plan import has_text
from src.processors import dedupe, quality, video
from src.processors.detector import AnimalDetector
from src.storage import files
from src.storage.db import session_scope
from src.storage.models import Detection, Video, VideoStatus

if TYPE_CHECKING:
    from src.config import Config


def run_processing(cfg: "Config", detect_animals: bool, check_quality: bool) -> dict[str, int]:
    """Advance status=downloaded videos through probe -> normalize -> keyframes -> quality ->
    dedupe -> detect -> processed|rejected(reason). Idempotent: only status=downloaded rows are
    touched, so a repeat run leaves already processed/rejected rows untouched. A failure on one
    video is logged and counted, without aborting the rest of the run.
    """
    processed = 0
    errors = 0
    reasons: Counter[str] = Counter()
    detector = AnimalDetector(cfg) if detect_animals else None

    with session_scope() as session:
        rows = session.execute(select(Video).where(Video.status == VideoStatus.DOWNLOADED)).scalars().all()

        for row in rows:
            try:
                _process_one(session, row, cfg, detector=detector, check_quality=check_quality)
            except Exception:
                logger.exception(f"processing failed for video id={row.id} path={row.file_path}")
                errors += 1
                continue

            if row.status == VideoStatus.PROCESSED:
                processed += 1
            elif row.status == VideoStatus.REJECTED:
                reasons[row.reject_reason or "unknown"] += 1

    result: dict[str, int] = {"processed": processed, "rejected": sum(reasons.values()), "errors": errors}
    result.update({f"rejected_{reason}": count for reason, count in reasons.items()})
    return result


def _process_one(
    session: Session, row: Video, cfg: "Config", *, detector: AnimalDetector | None, check_quality: bool
) -> None:
    path = Path(row.file_path)
    info = video.probe(path)
    row.duration_s = info.duration_s
    row.width = info.width
    row.height = info.height
    row.fps = info.fps
    row.codec = info.codec
    row.size_bytes = info.size_bytes

    video_cfg = cfg.processing.video
    if info.duration_s < video_cfg.min_duration or info.duration_s > video_cfg.max_duration:
        row.status = VideoStatus.REJECTED
        row.reject_reason = "duration"
        return

    # someone else's compilation is a dozen clips glued end to end; the dataset wants the
    # single take. Checked before normalize so a montage costs no transcode.
    if video.count_cuts(path) > video_cfg.max_cuts:
        row.status = VideoStatus.REJECTED
        row.reject_reason = "compilation"
        return

    tmp_out = path.with_suffix(".norm.mp4")
    normalized = video.normalize(path, tmp_out, cfg)
    if normalized != path:
        os.replace(normalized, path)

    frames = video.extract_keyframes(path, files.frames_dir(cfg, row.id), video_cfg.frame_interval)
    frame_paths = [fp for _, fp in frames]

    if check_quality:
        result = quality.assess(frame_paths, cfg)
        row.sharpness = result.sharpness
        row.brightness = result.brightness
        if not result.ok:
            row.status = VideoStatus.REJECTED
            row.reject_reason = result.reason
            return

    # a "TOP 10 FUNNIEST CATS" burned into the picture is someone else's ranking, and ours
    # would land on top of it. Last of the gates: it is the only one that costs a model call
    #
    # Every frame, not a sample of three: frames are two seconds apart, and reading only the
    # first, middle and last let a "All I said was good morning..." caption on a 14s clip into
    # a finished short -- it sat on the frames in between. `any` stops at the first yes, so a
    # clip that is going to be rejected still costs one call; only clean clips pay per frame.
    if any(has_text(frame, cfg.compiler) for frame in frame_paths):
        row.status = VideoStatus.REJECTED
        row.reject_reason = "burned_text"
        return

    sha256 = dedupe.sha256_file(path)
    phash = dedupe.phash_frames(frame_paths)
    # exclude_id: fetch already stored this row's sha256 at download time, so without it
    # a file normalize left untouched matches itself and rejects itself as a duplicate.
    dup = dedupe.find_duplicate(session, sha256, phash, exclude_id=row.id)

    if dup is not None:
        current_area = (row.width or 0) * (row.height or 0)
        dup_area = (dup.width or 0) * (dup.height or 0)
        if current_area > dup_area:
            # this video wins: the existing row is rejected and frees its sha256 (UNIQUE) for us
            dup.status = VideoStatus.REJECTED
            dup.reject_reason = "duplicate"
            dup.sha256 = None
            row.sha256 = sha256
            row.phash = phash
        else:
            row.status = VideoStatus.REJECTED
            row.reject_reason = "duplicate"
            row.phash = phash
            return
    else:
        row.sha256 = sha256
        row.phash = phash

    if detector is not None:
        detections = detector.detect(frames)
        if not detections:
            row.status = VideoStatus.REJECTED
            row.reject_reason = "no_animal"
            return

        category, conf = detector.summarize(detections)
        row.has_animal = True
        row.category = category
        row.detect_conf = conf
        for d in detections:
            session.add(
                Detection(
                    video_id=row.id, ts_s=d.ts_s, class_name=d.class_name, conf=d.conf,
                    x=d.x, y=d.y, w=d.w, h=d.h,
                )
            )

        new_path = files.video_path(cfg, category, row.source, row.source_id)
        new_path.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, new_path)
        row.file_path = str(new_path)

    row.status = VideoStatus.PROCESSED
