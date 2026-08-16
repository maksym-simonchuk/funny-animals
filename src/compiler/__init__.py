"""Assemble processed clips into a 1080x1920 short: pick clips, plan with a local
model, render segments, concatenate."""
from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import func, select

from src.compiler import render
from src.compiler.plan import Clip, Plan, PlanError, describe_frame, make_plan, pick_sound
from src.storage.db import session_scope
from src.storage.models import Detection, Video, VideoStatus

if TYPE_CHECKING:
    from src.config import Config

_LEAD_IN = 1.0  # start a segment slightly before the animal first shows up


class CompileError(RuntimeError):
    """Not enough usable material, or a step of the assembly failed."""


def _pick_clips(
    session, cfg: "Config", category: str | None, used: set[int] | None = None
) -> list[tuple[int, str, str, list[str], float, float]]:
    """Return (id, file_path, category, tags, duration_s, first_detection_s) for the
    newest processed clips, optionally narrowed to one category and skipping `used`."""
    first_seen = (
        select(Detection.video_id, func.min(Detection.ts_s).label("ts"))
        .group_by(Detection.video_id)
        .subquery()
    )
    query = (
        select(Video, first_seen.c.ts)
        .join(first_seen, first_seen.c.video_id == Video.id)
        .where(
            Video.status == VideoStatus.PROCESSED,
            Video.file_path.is_not(None),
            Video.duration_s >= cfg.compiler.segment_seconds,
        )
    )
    if category:
        query = query.where(Video.category == category)
    if used:
        query = query.where(Video.id.not_in(used))
    query = query.order_by(Video.id.desc()).limit(cfg.compiler.clips_per_short)

    rows = []
    for video, first_ts in session.execute(query).all():
        rows.append(
            (video.id, video.file_path, video.category or "animal",
             list(video.tags or []), video.duration_s or 0.0, float(first_ts))
        )
    return rows


def _segment_start(duration_s: float, first_detection_s: float, segment_s: float) -> float:
    """Where to cut from: just before the animal appears, clamped inside the clip."""
    return max(0.0, min(first_detection_s - _LEAD_IN, duration_s - segment_s))


def build_short(
    cfg: "Config",
    category: str | None = None,
    keep_work: bool = False,
    used: set[int] | None = None,
) -> Path:
    """Build one short from the most recent processed clips. Returns the output path.

    ``used`` is read and extended in place, so a caller building several shorts in a row
    gets a different set of clips in each of them.
    """
    compiler_cfg = cfg.compiler

    with session_scope() as session:
        picked = _pick_clips(session, cfg, category, used)

    if len(picked) < 2:
        raise CompileError(
            f"need at least 2 processed clips of >= {compiler_cfg.segment_seconds}s"
            f"{f' in category {category}' if category else ''}, found {len(picked)}"
        )

    missing = [path for _, path, *_ in picked if not Path(path).is_file()]
    if missing:
        raise CompileError(f"clip file missing on disk: {missing[0]}")

    work_dir = compiler_cfg.output_path / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        clips = _look_at_clips(picked, compiler_cfg, work_dir)
        plan = make_plan(clips, compiler_cfg)
        segments = _render_segments(picked, clips, plan, compiler_cfg, work_dir)
        music = _music_bed([Path(path) for _, path, *_ in picked], work_dir, compiler_cfg)
        out = _free_path(compiler_cfg.output_path / f"{_slug(plan.title)}.mp4")
        render.join(
            segments, out,
            transition_s=compiler_cfg.transition_seconds,
            music=music,
            music_volume=compiler_cfg.music_volume,
        )
    finally:
        if not keep_work:
            shutil.rmtree(work_dir, ignore_errors=True)

    if used is not None:
        used.update(video_id for video_id, *_ in picked)
    logger.info(f"short: {out} ({len(segments)} clips, {plan.category})")
    return out


def _look_at_clips(picked, compiler_cfg, work_dir: Path) -> list[Clip]:
    """Show the vision model a frame from the middle of every segment.

    The database knows only the animal class -- ``tags`` come back empty and ``title`` is
    just "Video by <author>" -- so without this pass the captions are pure invention.
    """
    clips: list[Clip] = []
    for index, (video_id, file_path, category, tags, duration_s, first_ts) in enumerate(picked):
        start = _segment_start(duration_s, first_ts, compiler_cfg.segment_seconds)
        frame = render.grab_frame(
            Path(file_path), work_dir / f"look_{index}.jpg",
            start + compiler_cfg.segment_seconds / 2,
        )
        seen = describe_frame(frame, compiler_cfg, hint=category)
        logger.info(f"vision {index + 1}/{len(picked)}: {seen or '(no description)'}")
        clips.append(Clip(video_id=video_id, category=category, tags=tags, seen=seen))
    return clips


def _render_segments(picked, clips: list[Clip], plan: Plan, compiler_cfg, work_dir: Path) -> list[Path]:
    # the rubric is the same on every clip, so it is drawn once
    rubric = render.caption_png(
        plan.category, work_dir / "rubric.png",
        compiler_cfg.font, compiler_cfg.title_size, top=True,
    )
    fills = _fill_audio(
        [Path(path) for _, path, *_ in picked], clips, work_dir, compiler_cfg
    )
    segments: list[Path] = []
    for index, (video_id, file_path, _, _, duration_s, first_ts) in enumerate(picked):
        overlays = [
            rubric,
            render.ranking_png(
                plan.captions, index, work_dir / f"rank_{index}.png",
                compiler_cfg.font, compiler_cfg.ranking_size,
            ),
        ]
        # the hook only belongs on the opening clip
        if index == 0:
            overlays.append(
                render.caption_png(
                    plan.hook or plan.title, work_dir / "hook.png",
                    compiler_cfg.font, compiler_cfg.caption_size, top=False,
                )
            )
        start = _segment_start(duration_s, first_ts, compiler_cfg.segment_seconds)
        segments.append(
            render.render_segment(
                Path(file_path), work_dir / f"seg_{index}.mp4",
                start, compiler_cfg.segment_seconds, overlays, fills.get(index),
            )
        )
        logger.info(f"segment {index + 1}/{len(picked)}: video {video_id} from {start:.1f}s")
    return segments


def _fill_audio(
    sources: list[Path], clips: list[Clip], work_dir: Path, compiler_cfg
) -> dict[int, Path]:
    """Segment index -> the track it borrows, for the clips that arrive without one.

    A dead-silent segment between two loud ones reads as a broken video. The fitting
    sound is already in the compilation -- the local model reads what the vision pass saw
    in each clip and matches the silent one to the soundtrack closest in mood, so nothing
    of unknown licensing has to be imported for it.
    """
    levels = [render.mean_volume(source) for source in sources]
    silent = [index for index, level in enumerate(levels) if level == float("-inf")]
    donors = [index for index, level in enumerate(levels) if level > float("-inf")]
    if not silent:
        return {}
    if not donors:
        logger.info("audio: every clip is silent, nothing to borrow")
        return {}

    fills: dict[int, Path] = {}
    for index in silent:
        choice = pick_sound(clips[index].seen, [clips[d].seen for d in donors], compiler_cfg)
        donor = donors[choice]
        out = work_dir / f"fill_{donor}.m4a"
        logger.info(f"audio: clip {index + 1} borrows the track of clip {donor + 1}")
        fills[index] = out if out.exists() else render.extract_audio(sources[donor], out)
    return fills


def _music_bed(sources: list[Path], work_dir: Path, compiler_cfg) -> Path | None:
    """Reuse the loudest clip's own audio as the soundtrack for the whole short.

    Reels carry their trending track in the video itself, so the compilation gets a
    coherent bed without importing music of unknown licensing. Silent clips score -inf.
    """
    if compiler_cfg.music_volume <= 0:
        return None

    level, loudest = max((render.mean_volume(source), source) for source in sources)
    if level == float("-inf"):
        logger.info("music: no clip has an audio track, skipping the bed")
        return None

    logger.info(f"music: bed taken from {loudest.name}")
    return render.extract_audio(loudest, work_dir / "bed.m4a")


def _free_path(path: Path) -> Path:
    """`name.mp4`, or `name-2.mp4` etc. -- two shorts in a batch can land on the same title."""
    candidate, index = path, 1
    while candidate.exists():
        index += 1
        candidate = path.with_name(f"{path.stem}-{index}{path.suffix}")
    return candidate


def _slug(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    return "-".join(part for part in cleaned.split("-") if part)[:60] or "short"


__all__ = ["CompileError", "PlanError", "build_short"]
