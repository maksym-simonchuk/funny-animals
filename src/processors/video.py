"""Video probing, normalization, keyframe extraction and GIF preview generation via ffmpeg/ffprobe."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config

_TARGET_AUDIO_CODEC = "aac"  # not separately configurable (VideoCfg only exposes target_codec for video)


@dataclass(frozen=True)
class VideoInfo:
    duration_s: float
    width: int
    height: int
    fps: float
    codec: str
    size_bytes: int


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run an ffmpeg/ffprobe subprocess, raising RuntimeError with stderr on non-zero exit."""
    try:
        return subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"ffmpeg command failed: {' '.join(cmd)}\n{stderr}") from exc


def _ffprobe_streams(path: Path) -> tuple[dict, list[dict]]:
    """Return (format, streams) parsed from ffprobe JSON. Raises ValueError on a corrupt/unreadable file."""
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise ValueError(f"corrupt or unreadable video file {path}: {stderr}") from exc

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"corrupt or unreadable video file {path}: invalid ffprobe output") from exc

    return data.get("format", {}), data.get("streams", [])


def _parse_frame_rate(raw: str) -> float:
    num, _, den = raw.partition("/")
    den = den or "1"
    try:
        denominator = float(den)
        return float(num) / denominator if denominator else 0.0
    except ValueError:
        return 0.0


def probe(path: Path) -> VideoInfo:
    """Probe a video file via ffprobe. Raises ValueError on a corrupt/unreadable input."""
    fmt, streams = _ffprobe_streams(path)
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"corrupt or unreadable video file {path}: no video stream found")

    duration_s = float(fmt.get("duration") or video_stream.get("duration") or 0.0)
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    fps = _parse_frame_rate(video_stream.get("r_frame_rate", "0/1"))
    codec = video_stream.get("codec_name", "")
    size_bytes = int(fmt.get("size") or path.stat().st_size)

    return VideoInfo(
        duration_s=duration_s, width=width, height=height,
        fps=fps, codec=codec, size_bytes=size_bytes,
    )


def normalize(path: Path, out: Path, cfg: "Config") -> Path:
    """Transcode to mp4/h264/aac only if the source doesn't already match; else return `path` unchanged."""
    fmt, streams = _ffprobe_streams(path)
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"corrupt or unreadable video file {path}: no video stream found")
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    video_codec = video_stream.get("codec_name", "")
    audio_codec = audio_stream.get("codec_name") if audio_stream else None
    height = int(video_stream.get("height") or 0)

    video_ok = video_codec == cfg.processing.video.target_codec
    audio_ok = audio_codec is None or audio_codec == _TARGET_AUDIO_CODEC
    container_ok = path.suffix.lower() == f".{cfg.processing.video.target_format}"

    if video_ok and audio_ok and container_ok:
        return path

    target_resolution = cfg.processing.video.target_resolution
    vf_filters = []
    if target_resolution and height > target_resolution:
        vf_filters.append(f"scale=-2:{target_resolution}")

    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(path)]
    if vf_filters:
        cmd += ["-vf", ",".join(vf_filters)]
    cmd += ["-c:v", "libx264", "-c:a", "aac", str(out)]
    _run(cmd)
    return out


def extract_keyframes(path: Path, out_dir: Path, interval_s: int) -> list[tuple[float, Path]]:
    """Extract one frame every `interval_s` seconds. Always returns at least one frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%04d.jpg"
    # format=yuvj420p avoids an mjpeg encoder-init failure ffmpeg raises on some
    # full-range-ambiguous yuv420p sources (seen with synthetic lavfi testsrc clips).
    cmd = ["ffmpeg", "-y", "-i", str(path), "-vf", f"fps=1/{interval_s},format=yuvj420p", "-q:v", "2", str(pattern)]
    try:
        _run(cmd)
    except RuntimeError:
        pass  # e.g. interval_s longer than the clip: fall through to the single-frame fallback below

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        single = out_dir / "frame_0001.jpg"
        _run(["ffmpeg", "-y", "-i", str(path), "-vf", "format=yuvj420p", "-vframes", "1", "-q:v", "2", str(single)])
        frames = [single]

    return [(idx * float(interval_s), frame) for idx, frame in enumerate(frames)]


def count_cuts(path: Path, threshold: float = 0.4) -> int:
    """How many hard cuts the video has.

    A reel shot in one take has none; a compilation someone stitched out of other people's
    clips has one per clip in it. ffmpeg's `scene` score is how much of the picture changed
    between two frames, and 0.4 is where the pool splits: 86 of its 96 clips score under it
    the whole way through, every known "Top 10" montage scores over it several times.
    """
    result = _run([
        "ffmpeg", "-v", "error", "-i", str(path), "-an",
        "-vf", f"select='gt(scene,{threshold})',metadata=print:file=-", "-f", "null", "-",
    ])
    return result.stdout.decode("utf-8", errors="replace").count("pts_time")


def make_gif(path: Path, out: Path, seconds: int = 3) -> Path:
    """Render a 320px-wide, 10fps GIF preview from the middle `seconds` of the video."""
    info = probe(path)
    start = max(0.0, (info.duration_s - seconds) / 2)
    out.parent.mkdir(parents=True, exist_ok=True)
    palette = out.with_suffix(".palette.png")

    try:
        _run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", str(seconds), "-i", str(path),
            "-vf", "fps=10,scale=320:-1:flags=lanczos,palettegen", str(palette),
        ])
        _run([
            "ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", str(seconds), "-i", str(path),
            "-i", str(palette),
            "-lavfi", "fps=10,scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse",
            str(out),
        ])
    finally:
        palette.unlink(missing_ok=True)

    return out
