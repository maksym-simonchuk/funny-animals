"""Assemble processed clips into a 1080x1920 short: pick clips, plan with a local
model, render segments, concatenate."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from sqlalchemy import func, select

from src.compiler import render
from src.compiler.plan import (
    Clip, Plan, PlanError, describe_frame, group_clips, has_text, make_plan, pick_sound,
    tag_clip, tag_prop, unload,
)
from src.storage.db import session_scope
from src.storage.models import Detection, Video, VideoStatus

if TYPE_CHECKING:
    from src.config import Config

_LEAD_IN = 1.0  # start a segment slightly before the animal first shows up
# dBFS below which a track has nothing worth lending: the room tone of these reels
# measures around -39 dB once the rumble is filtered out, real sound sits above -30
_AUDIBLE = -35.0


class CompileError(RuntimeError):
    """Not enough usable material, or a step of the assembly failed."""


def _pick_clips(
    session, cfg: "Config", category: str | None
) -> list[tuple[int, str, str, list[str], float, float]]:
    """Return (id, file_path, category, tags, duration_s, first_detection_s) for every
    processed clip, optionally narrowed to one category. The whole pool comes back, not
    one short's worth: the five that end up in a compilation are chosen out of it by
    theme, not by how recent they are."""
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
    query = query.order_by(Video.id.desc())

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


def _memory(cfg: "Config") -> Path:
    return cfg.compiler.output_path.parent / "compiled.json"


def recall(cfg: "Config") -> tuple[set[int], set[str]]:
    """The clips and themes earlier runs already spent, as (used, themes).

    Both sets used to live for one batch only. Starting a second `compile` with them
    empty, the fullest bucket is the same bucket, so the run rebuilds the shorts the
    previous one made: "TOP 5 TOY DESTROYERS" over the same five clips, saved as
    `top-5-toy-destroyers-2.mp4`. Held on disk, each run has to look further down.
    """
    try:
        note = json.loads(_memory(cfg).read_text())
    except (OSError, json.JSONDecodeError):
        return set(), set()
    return set(note.get("clips", [])), set(note.get("themes", []))


def keep(cfg: "Config", used: set[int], themes: set[str]) -> None:
    """Write back what this batch spent. Neither set starves the pool: `build_short`
    falls back to the whole pool once the unused clips are under a third of it, and
    `group_clips` reruns the labels with an empty `done` when they are all spoken for."""
    _memory(cfg).write_text(json.dumps({"clips": sorted(used), "themes": sorted(themes)}))


def build_short(
    cfg: "Config",
    category: str | None = None,
    keep_work: bool = False,
    used: set[int] | None = None,
    themes: set[str] | None = None,
) -> Path:
    """Build one short from the most recent processed clips. Returns the output path.

    ``used`` and ``themes`` are read and extended in place, so a caller building several
    shorts in a row gets a different set of clips and a different theme in each of them.
    """
    compiler_cfg = cfg.compiler

    with session_scope() as session:
        pool = _pick_clips(session, cfg, category)

    fresh = [row for row in pool if row[0] not in (used or set())]
    # a batch of nineteen shorts wants 95 of the 96 clips in the pool, so the last few are
    # left choosing from a handful with nothing in common and get the generic heading.
    # While the unused clips are a third of the pool they are what a short is built from;
    # below that the whole pool comes back, and the same clip in two compilations under
    # two different themes beats a run of "TOP 5 FUNNY ANIMALS"
    if len(fresh) >= len(pool) * 0.3:
        pool = fresh
    else:
        logger.info(f"only {len(fresh)} of {len(pool)} clips unused; drawing from all of them")

    if len(pool) < 2:
        raise CompileError(
            f"need at least 2 processed clips of >= {compiler_cfg.segment_seconds}s"
            f"{f' in category {category}' if category else ''}, found {len(pool)}"
        )

    missing = [path for _, path, *_ in pool if not Path(path).is_file()]
    if missing:
        raise CompileError(f"clip file missing on disk: {missing[0]}")

    work_dir = compiler_cfg.output_path / ".work"
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        seen = _look_at_clips(pool, compiler_cfg, work_dir)
        # a reel that came with its own "Top 10" list burned in puts two rankings on the
        # screen at once; ours cannot be moved out of the way, so the clip goes
        keep = [index for index, clip in enumerate(seen) if not clip.text]
        if len(keep) < len(seen):
            logger.info(f"{len(seen) - len(keep)} clips carry their own captions, skipping them")
            pool = [pool[index] for index in keep]
            seen = [seen[index] for index in keep]
        chosen, theme = group_clips(seen, compiler_cfg.clips_per_short, compiler_cfg, themes)
        picked = [pool[index] for index in chosen]
        clips = [seen[index] for index in chosen]
        plan = make_plan(clips, compiler_cfg, theme)
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
    """Show the vision model a frame from the middle of every clip in the pool.

    The database knows only the animal class -- ``tags`` come back empty and ``title`` is
    just "Video by <author>" -- so without this pass there is nothing to group the clips
    on and the captions are pure invention.

    Descriptions are cached on disk by video id: a batch of nineteen shorts walks the same
    pool nineteen times, and each look costs about five seconds of local inference.
    """
    cache_path = compiler_cfg.output_path.parent / "vision.json"
    try:
        cache = json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        cache = {}

    clips: list[Clip] = []
    fresh = 0
    for index, (video_id, file_path, category, tags, duration_s, first_ts) in enumerate(picked):
        cached = cache.get(str(video_id))
        # the burned-text question reads the frame, so a cache written before it existed
        # costs another look -- once per clip, then it is in the file like the rest
        looked = cached is None or "text" not in cached
        if looked:
            start = _segment_start(duration_s, first_ts, compiler_cfg.segment_seconds)
            frame = render.grab_frame(
                Path(file_path), work_dir / f"look_{index}.jpg",
                start + compiler_cfg.segment_seconds / 2,
            )
        if cached is None:
            animal, scene = describe_frame(frame, compiler_cfg)
            cached = {"animal": animal, "scene": scene}
            cache[str(video_id)] = cached
        if "text" not in cached:
            cached["text"] = has_text(frame, compiler_cfg)
        # both labels read the description, not the frame: a cache written before the
        # second axis existed only needs the cheap text call, not another look
        missing = [key for key in ("tag", "prop") if key not in cached]
        for key, label in (("tag", tag_clip), ("prop", tag_prop)):
            if key in missing:
                cached[key] = label(cached.get("scene", ""), compiler_cfg)
        if missing or looked:
            fresh += 1
            logger.info(
                f"vision {index + 1}/{len(picked)} [{cached['tag']} / {cached['prop']}"
                f"{' / captioned' if cached['text'] else ''}]: "
                f"{cached.get('animal') or '?'} -- {cached.get('scene') or '(none)'}"
            )
        clips.append(Clip(
            video_id=video_id, category=category, tags=tags,
            seen=cached.get("scene", ""), animal=cached.get("animal", ""),
            tag=cached.get("tag", ""), prop=cached.get("prop", ""),
            text=bool(cached.get("text")),
        ))

    if fresh:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=1))
        # the two models are ~6 GB each: leaving this one resident while the text model
        # loads puts the machine into swap, and a one-second call takes fifteen minutes
        unload(compiler_cfg.vision_model, compiler_cfg)
    return clips


def _render_segments(picked, clips: list[Clip], plan: Plan, compiler_cfg, work_dir: Path) -> list[Path]:
    # the rubric is the same on every clip, so it is drawn once. It says how many rows the
    # ranking has, so a viewer landing mid-clip knows there is a countdown to stay for.
    rubric = render.rubric_png(
        f"TOP {len(picked)} {plan.category}",
        # replace, not format: a stray brace in a hand-edited template is a typo, not a crash
        compiler_cfg.cta.replace("{category}", plan.category),
        work_dir / "rubric.png",
        compiler_cfg.font, compiler_cfg.title_size, compiler_cfg.cta_size,
    )
    fills = _fill_audio(picked, clips, work_dir, compiler_cfg)
    segments: list[Path] = []
    for index, (video_id, file_path, _, _, duration_s, first_ts) in enumerate(picked):
        overlays = [
            rubric,
            render.ranking_png(
                plan.captions, index, work_dir / f"rank_{index}.png",
                compiler_cfg.font, compiler_cfg.ranking_size,
            ),
        ]
        # every clip gets its own meme caption along the bottom; the hook opens the short
        line = plan.lines[index] if index < len(plan.lines) else ""
        if index == 0:
            line = plan.hook or line or plan.title
        if line:
            overlays.append(
                render.caption_png(
                    line, work_dir / f"line_{index}.png",
                    compiler_cfg.font, compiler_cfg.caption_size,
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


def _fill_audio(picked, clips: list[Clip], work_dir: Path, compiler_cfg) -> dict[int, Path]:
    """Segment index -> the track it borrows, for the clips that arrive without one.

    A dead-silent segment between two loud ones reads as a broken video. The fitting
    sound is already in the compilation -- the local model reads what the vision pass saw
    in each clip and matches the silent one to the soundtrack closest in mood, so nothing
    of unknown licensing has to be imported for it. Only clips with something actually
    audible can donate, and each donates the same slice the compilation shows of it.
    """
    starts = [
        _segment_start(duration_s, first_ts, compiler_cfg.segment_seconds)
        for _, _, _, _, duration_s, first_ts in picked
    ]
    # measure the slice the compilation actually shows: a clip can be loud in its intro
    # and dead by the moment we cut into it
    levels = [
        render.mean_volume(Path(path), start, compiler_cfg.segment_seconds)
        for (_, path, *_), start in zip(picked, starts)
    ]
    silent = [index for index, level in enumerate(levels) if level == float("-inf")]
    if not silent:
        return {}

    donors = [index for index, level in enumerate(levels) if level > _AUDIBLE]
    if not donors:
        logger.info("audio: no clip here has an audible track, the silent ones stay silent")
        return {}

    fills: dict[int, Path] = {}
    for index in silent:
        choice = pick_sound(clips[index].seen, [clips[d].seen for d in donors], compiler_cfg)
        donor = donors[choice]
        out = work_dir / f"fill_{donor}.m4a"
        logger.info(f"audio: clip {index + 1} borrows the track of clip {donor + 1}")
        if not out.exists():
            source = Path(picked[donor][1])
            render.extract_audio(source, out, starts[donor], compiler_cfg.segment_seconds)
        fills[index] = out
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


__all__ = ["CompileError", "PlanError", "build_short", "keep", "recall"]
