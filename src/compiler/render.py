"""Render a 1080x1920 short: crop clips to a segment, letterbox them onto a blurred
fill, overlay caption images, concatenate.

Text is drawn with Pillow into an RGBA png rather than with ffmpeg's ``drawtext``:
the local ffmpeg is built without libfreetype, and Pillow additionally gives us word
wrapping and a stroke, neither of which ``drawtext`` can do.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageDraw, ImageFont

if TYPE_CHECKING:
    from src.config import CompilerCfg

WIDTH = 1080
HEIGHT = 1920
_FPS = 30
_MARGIN = 64
# hand-held phone clips carry a wall of sub-80 Hz rumble that no phone speaker can play
# back; left in, loudnorm spends the whole gain budget on it and the clip sounds like noise.
# Two stages: one 12 dB/oct pass leaves a third of a loud rumble standing.
_RUMBLE = "highpass=f=90,highpass=f=90"


def _run(cmd: list[str]) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg command failed: {' '.join(cmd)}\n{stderr}") from exc


def _load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError as exc:
        raise RuntimeError(f"cannot load font {path}: {exc}") from exc


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    words = text.split()
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textlength(candidate, font=font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _stroked(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont,
             y: int, size: int) -> None:
    """One centred line, white with a black outline -- legible over any clip."""
    x = (WIDTH - draw.textlength(text, font=font)) / 2
    draw.text(
        (x, y), text, font=font, fill=(255, 255, 255, 255),
        stroke_width=max(2, size // 12), stroke_fill=(0, 0, 0, 255),
    )


def caption_png(text: str, out: Path, font_path: str, size: int) -> Path:
    """Draw `text` along the bottom of a transparent 1080x1920 layer."""
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _load_font(font_path, size)

    lines = _wrap(draw, text.upper(), font, WIDTH - 2 * _MARGIN)
    line_height = int(size * 1.15)
    y = HEIGHT - _MARGIN * 3 - line_height * len(lines)

    for line in lines:
        _stroked(draw, line, font, y, size)
        y += line_height

    out.parent.mkdir(parents=True, exist_ok=True)
    layer.save(out)
    return out


def rubric_png(title: str, cta: str, out: Path, font_path: str, size: int,
               cta_size: int) -> Path:
    """The heading with the small like-prompt under it, on one layer.

    Drawn together because the prompt has to start below whatever the heading wrapped to:
    at a fixed y it lands on the second line of "TOP 5 PROFESSIONAL LOUNGERS".
    """
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = _load_font(font_path, size)

    y = _MARGIN * 3
    for line in _wrap(draw, title.upper(), font, WIDTH - 2 * _MARGIN):
        _stroked(draw, line, font, y, size)
        y += int(size * 1.15)

    if cta:
        small = _load_font(font_path, cta_size)
        y += cta_size // 2  # the prompt is a second thought, not the heading's third line
        for line in _wrap(draw, cta.upper(), small, WIDTH - 2 * _MARGIN):
            _stroked(draw, line, small, y, cta_size)
            y += int(cta_size * 1.15)

    out.parent.mkdir(parents=True, exist_ok=True)
    layer.save(out)
    return out


def _fit(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> str:
    """Trim `text` until it fits -- ranking rows must stay on one line.

    Whole words go first: "WATERMELON WIELDER" reads as a joke where the mid-word cut
    "WATERMELON WIELDER, UNSTOPP..." reads as a bug. The ellipsis is the last resort, for
    the single word that is too wide on its own.
    """
    words = text.split()
    while len(words) > 1 and draw.textlength(" ".join(words), font=font) > max_width:
        words.pop()
    # the comma the dropped word hung off would read as a row cut short
    text = " ".join(words).rstrip(",;:")
    if draw.textlength(text, font=font) <= max_width:
        return text
    while text and draw.textlength(f"{text}...", font=font) > max_width:
        text = text[:-1]
    return f"{text}..."


def _row_font(
    draw: ImageDraw.ImageDraw, rows: list[str], font_path: str, size: int, max_width: int
) -> ImageFont.FreeTypeFont:
    """Largest size at which every row fits on one line, down to 70% of `size`.

    Shrinking the whole list keeps it looking like one list and costs no words; below 70%
    the rows stop being readable on a phone, so the overlong one loses a word instead.
    """
    for candidate in range(size, int(size * 0.7), -2):
        font = _load_font(font_path, candidate)
        if all(draw.textlength(row, font=font) <= max_width for row in rows):
            return font
    return _load_font(font_path, int(size * 0.7))


def ranking_png(names: list[str], active: int, out: Path, font_path: str, size: int) -> Path:
    """Numbered list down the left edge, vertically centred, with row `active` lit up.

    Rows past `active` keep their number but hide their text -- the list fills in as the
    short plays, so the viewer stays for the ones not shown yet.
    """
    layer = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    # sized against every row, not just the visible ones: the list must not resize as it fills in
    limit = int(WIDTH * 0.62)
    labels = [f"{index + 1}. {name.upper()}" for index, name in enumerate(names)]
    font = _row_font(draw, labels, font_path, size, limit)

    line_height = int(size * 1.7)
    y = (HEIGHT - line_height * len(names)) // 2
    for index, label in enumerate(labels):
        current = index == active
        if index > active:
            label = f"{index + 1}."
        draw.text(
            (_MARGIN, y), _fit(draw, label, font, limit), font=font,
            fill=(255, 209, 26, 255) if current else (255, 255, 255, 150),
            stroke_width=max(2, size // 10), stroke_fill=(0, 0, 0, 255 if current else 150),
        )
        y += line_height

    out.parent.mkdir(parents=True, exist_ok=True)
    layer.save(out)
    return out


def _segment_filter(overlay_count: int) -> str:
    """Cover-scale a blurred copy of the clip as the background, fit the clip on top,
    then stack the caption layers (input 1..n) over the result."""
    chain = (
        f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,"
        f"crop={WIDTH}:{HEIGHT},gblur=sigma=30,setsar=1[bg];"
        f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,setsar=1[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,fps={_FPS}[v0]"
    )
    for index in range(overlay_count):
        chain += f";[v{index}][{index + 1}:v]overlay=0:0[v{index + 1}]"
    return chain


def render_segment(
    source: Path,
    out: Path,
    start_s: float,
    duration_s: float,
    overlays: list[Path],
    fill_audio: Path | None = None,
) -> Path:
    """Cut [start_s, start_s+duration_s] from `source` onto the 9:16 canvas with `overlays` burned in.

    Every segment leaves with identical parameters (1080x1920, 30fps, h264, aac 48k
    stereo) so the concat demuxer can stitch them without re-encoding. A clip with no
    audio track of its own plays `fill_audio` instead, looped to length, and falls back
    to silence without one -- concat would drop the stream mid-way otherwise.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}", "-i", str(source)]
    for overlay in overlays:
        cmd += ["-i", str(overlay)]

    has_audio = _has_audio(source)
    audio_map = "0:a:0"
    if not has_audio:
        if fill_audio is not None:
            # loop it: the borrowed track is usually shorter than the segment it covers
            cmd += ["-stream_loop", "-1", "-t", f"{duration_s:.3f}", "-i", str(fill_audio)]
        else:
            cmd += ["-f", "lavfi", "-t", f"{duration_s:.3f}", "-i", "anullsrc=r=48000:cl=stereo"]
        audio_map = f"{len(overlays) + 1}:a"

    cmd += [
        "-filter_complex", _segment_filter(len(overlays)),
        "-map", f"[v{len(overlays)}]", "-map", audio_map,
    ]
    if has_audio or fill_audio is not None:
        # reels arrive between -30 and 0 dBFS: even the loudness out so one clip does not
        # blast after another, and leave headroom so the join has room to mix on top
        cmd += ["-af", f"{_RUMBLE},loudnorm=I=-16:TP=-1.5:LRA=11"]
    cmd += [
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", str(out),
    ]
    _run(cmd)
    return out


def grab_frame(source: Path, out: Path, at_s: float) -> Path:
    """One downscaled still, for the vision model to look at."""
    out.parent.mkdir(parents=True, exist_ok=True)
    _run(["ffmpeg", "-y", "-ss", f"{at_s:.3f}", "-i", str(source),
          "-frames:v", "1", "-vf", "scale=640:-2", str(out)])
    return out


def _has_audio(source: Path) -> bool:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", str(source)],
        capture_output=True,
    )
    return bool(probe.stdout.strip())


def _duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True,
    )
    return float(result.stdout.strip() or 0.0)


def mean_volume(source: Path, start_s: float = 0.0, duration_s: float | None = None) -> float:
    """Mean dBFS of what is *audible* in a file, or -inf when it has no audio at all.

    Measured through the same high-pass the render applies, because phone-shot reels are
    full of sub-80 Hz handling rumble: one of these clips reads as the loudest in the set
    at -12 dB while 97% of that energy sits below 80 Hz, where nothing can hear it.
    """
    if not _has_audio(source):
        return float("-inf")
    cmd = ["ffmpeg", "-hide_banner", "-ss", f"{start_s:.3f}"]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-i", str(source), "-af", f"{_RUMBLE},volumedetect", "-f", "null", "-"]
    result = subprocess.run(cmd, capture_output=True)
    for line in reversed(result.stderr.decode("utf-8", errors="replace").splitlines()):
        if "mean_volume:" in line:
            try:
                return float(line.split("mean_volume:")[1].split("dB")[0].strip())
            except ValueError:
                break
    return float("-inf")


def extract_audio(
    source: Path, out: Path, start_s: float = 0.0, duration_s: float | None = None
) -> Path:
    """Pull a clip's audio out as an aac track, to be reused elsewhere in the short."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-ss", f"{start_s:.3f}"]
    if duration_s is not None:
        cmd += ["-t", f"{duration_s:.3f}"]
    cmd += ["-i", str(source), "-vn", "-c:a", "aac", "-ar", "48000", "-ac", "2", str(out)]
    _run(cmd)
    return out


def _filter_in(label: str) -> str:
    """A label as a filtergraph input: pad names are bracketed, raw stream specs are not."""
    return label if label.startswith("[") else f"[{label}]"


def _join_filter(durations: list[float], transition_s: float) -> tuple[str, str, str]:
    """Filter graph stitching N segments, plus the labels its video and audio end on.

    With transitions this is an xfade/acrossfade chain: each crossfade overlaps the pair
    by `transition_s`, so every following offset shifts back by the time already eaten.
    """
    count = len(durations)
    if count == 1:
        return "", "0:v", "0:a"
    if transition_s <= 0:
        streams = "".join(f"[{i}:v][{i}:a]" for i in range(count))
        return f"{streams}concat=n={count}:v=1:a=1[vout][aout]", "[vout]", "[aout]"

    steps: list[str] = []
    video, audio = "0:v", "0:a"
    elapsed = durations[0]
    for index in range(1, count):
        offset = max(0.0, elapsed - transition_s)
        steps.append(
            f"[{video}][{index}:v]xfade=transition=fade:"
            f"duration={transition_s:.3f}:offset={offset:.3f}[vx{index}]"
        )
        steps.append(f"[{audio}][{index}:a]acrossfade=d={transition_s:.3f}[ax{index}]")
        video, audio = f"vx{index}", f"ax{index}"
        elapsed = offset + durations[index]
    return ";".join(steps), f"[{video}]", f"[{audio}]"


def join(
    segments: list[Path],
    out: Path,
    transition_s: float = 0.0,
    music: Path | None = None,
    music_volume: float = 0.0,
) -> Path:
    """Stitch segments, crossfading between them and mixing `music` in underneath.

    Crossfading forces a re-encode -- there is no copy-through path once frames from two
    clips are blended -- so this is the one pass that touches every pixel of the result.
    """
    durations = [_duration(segment) for segment in segments]
    graph, video_label, audio_label = _join_filter(durations, transition_s)

    cmd = ["ffmpeg", "-y"]
    for segment in segments:
        cmd += ["-i", str(segment)]
    if music is not None and music_volume > 0:
        # loop the bed: one clip's audio is shorter than the compilation it plays under
        cmd += ["-stream_loop", "-1", "-i", str(music)]
        bed = f"[{len(segments)}:a]volume={music_volume},atrim=0:{sum(durations):.3f}[bed]"
        # normalize=0 keeps the clips' own audio at full level instead of halving it;
        # the limiter catches the sum, which would otherwise clip into audible grit
        mix = (
            f"{_filter_in(audio_label)}[bed]amix=inputs=2:duration=first:normalize=0,"
            f"alimiter=limit=0.9[amix]"
        )
        graph = ";".join(part for part in (graph, bed, mix) if part)
        audio_label = "[amix]"

    out.parent.mkdir(parents=True, exist_ok=True)
    if graph:
        cmd += ["-filter_complex", graph]
    cmd += [
        "-map", video_label, "-map", audio_label,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(out),
    ]
    _run(cmd)
    return out
