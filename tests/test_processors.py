"""Tests for src.processors: video probing/transcoding, quality gating, dedupe.

Uses the shared fixtures from tests/conftest.py (tmp_cfg, db, sample_video, short_video,
long_video). No network access. Extra synthetic clips needed only by this module (odd
codec, near-dup pairs) are generated locally via ffmpeg lavfi, same convention as conftest.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import imagehash
import pytest

import src.processors as processors
from src.processors import dedupe, quality, video
from src.processors import run_processing
from src.processors.detector import DetectionResult
from src.storage import files
from src.storage.db import session_scope
from src.storage.models import Video, VideoStatus


def _make_video(
    path: Path,
    *,
    duration: float,
    codec: str = "libx264",
    pix_fmt: str = "yuv420p",
    width: int = 640,
    height: int = 360,
    bitrate: str | None = None,
    with_audio: bool = False,
    audio_codec: str = "aac",
) -> Path:
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate=25:duration={duration}"]
    if with_audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=1000:duration={duration}"]
    cmd += ["-c:v", codec, "-pix_fmt", pix_fmt]
    if bitrate:
        cmd += ["-b:v", bitrate]
    if with_audio:
        cmd += ["-c:a", audio_codec]
    cmd += [str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _stitch(path: Path, tmp_path: Path) -> Path:
    """Three unrelated 2s clips end to end -- what someone else's compilation looks like."""
    parts = []
    for index, source in enumerate(("testsrc", "smptebars", "rgbtestsrc")):
        part = tmp_path / f"part{index}.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", f"{source}=size=320x240:rate=25:duration=2",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(part)],
            check=True, capture_output=True,
        )
        parts.append(part)

    listing = tmp_path / "parts.txt"
    listing.write_text("".join(f"file '{part}'\n" for part in parts))
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(path)],
        check=True, capture_output=True,
    )
    return path


def _quality_cfg(
    min_sharpness: float = 0.0,
    min_brightness: float = 0.0,
    min_resolution: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        processing=SimpleNamespace(
            quality=SimpleNamespace(
                min_sharpness=min_sharpness,
                min_brightness=min_brightness,
                min_resolution=min_resolution,
            )
        )
    )


def _video_cfg(target_resolution: int = 720, target_codec: str = "h264", target_format: str = "mp4") -> SimpleNamespace:
    return SimpleNamespace(
        processing=SimpleNamespace(
            video=SimpleNamespace(
                target_resolution=target_resolution, target_codec=target_codec, target_format=target_format
            )
        )
    )


# --- video.probe -------------------------------------------------------------------


def test_probe_reads_video_info(sample_video: Path) -> None:
    info = video.probe(sample_video)
    assert info.width == 1280
    assert info.height == 720
    assert abs(info.fps - 25.0) < 0.1
    assert 5.5 < info.duration_s < 6.5
    assert info.codec
    assert info.size_bytes == sample_video.stat().st_size


def test_probe_corrupt_file_raises_value_error(tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a real video file")
    with pytest.raises(ValueError):
        video.probe(bad)


# --- video.normalize -----------------------------------------------------------------


def test_normalize_returns_original_when_already_compliant(sample_video: Path, tmp_cfg) -> None:
    info = video.probe(sample_video)
    if info.codec != "h264" or sample_video.suffix.lower() != ".mp4":
        pytest.skip("sample_video fixture is not h264/mp4; compliant-path assumption doesn't hold")
    out = video.normalize(sample_video, sample_video.with_name("out.mp4"), tmp_cfg)
    assert out == sample_video


def test_normalize_transcodes_noncompliant_codec(tmp_path: Path) -> None:
    src = _make_video(tmp_path / "odd.avi", duration=2, codec="mpeg4", width=320, height=240)
    out_path = tmp_path / "normalized.mp4"

    out = video.normalize(src, out_path, _video_cfg())

    assert out == out_path
    assert out.exists()
    info = video.probe(out)
    assert info.codec == "h264"


def test_normalize_downscales_only_when_taller_than_target(tmp_path: Path) -> None:
    src = _make_video(tmp_path / "tall.avi", duration=2, codec="mpeg4", width=1280, height=1280)
    out_path = tmp_path / "normalized.mp4"

    out = video.normalize(src, out_path, _video_cfg(target_resolution=720))

    info = video.probe(out)
    assert info.height == 720


# --- video.extract_keyframes ----------------------------------------------------------


def test_extract_keyframes_at_interval(sample_video: Path, tmp_path: Path) -> None:
    frames = video.extract_keyframes(sample_video, tmp_path / "frames", interval_s=2)
    assert len(frames) >= 1
    for ts, frame_path in frames:
        assert frame_path.exists()
        assert frame_path.stat().st_size > 0
    timestamps = [ts for ts, _ in frames]
    assert timestamps == sorted(timestamps)


def test_extract_keyframes_short_video_returns_at_least_one(short_video: Path, tmp_path: Path) -> None:
    frames = video.extract_keyframes(short_video, tmp_path / "frames", interval_s=30)
    assert len(frames) >= 1
    assert frames[0][1].exists()


# --- video.make_gif --------------------------------------------------------------------


def test_make_gif_creates_file_and_cleans_up_palette(sample_video: Path, tmp_path: Path) -> None:
    out = video.make_gif(sample_video, tmp_path / "preview.gif", seconds=3)
    assert out.exists()
    assert out.stat().st_size > 0
    assert not out.with_suffix(".palette.png").exists()


# --- quality.assess ----------------------------------------------------------------------


def test_quality_assess_ok(sample_video: Path, tmp_path: Path) -> None:
    frames = [fp for _, fp in video.extract_keyframes(sample_video, tmp_path / "frames", interval_s=2)]
    result = quality.assess(frames, _quality_cfg(min_sharpness=1.0, min_brightness=1.0, min_resolution=1))
    assert result.ok
    assert result.reason is None
    assert result.sharpness > 0
    assert result.brightness > 0


def test_quality_assess_low_sharpness(sample_video: Path, tmp_path: Path) -> None:
    frames = [fp for _, fp in video.extract_keyframes(sample_video, tmp_path / "frames", interval_s=2)]
    result = quality.assess(frames, _quality_cfg(min_sharpness=1e9, min_brightness=1.0, min_resolution=1))
    assert not result.ok
    assert result.reason == "low_sharpness"


def test_quality_assess_dark(sample_video: Path, tmp_path: Path) -> None:
    frames = [fp for _, fp in video.extract_keyframes(sample_video, tmp_path / "frames", interval_s=2)]
    result = quality.assess(frames, _quality_cfg(min_sharpness=1.0, min_brightness=1e9, min_resolution=1))
    assert not result.ok
    assert result.reason == "dark"


def test_quality_assess_low_resolution(sample_video: Path, tmp_path: Path) -> None:
    frames = [fp for _, fp in video.extract_keyframes(sample_video, tmp_path / "frames", interval_s=2)]
    result = quality.assess(frames, _quality_cfg(min_sharpness=1.0, min_brightness=1.0, min_resolution=999999))
    assert not result.ok
    assert result.reason == "low_resolution"


def test_quality_assess_reason_order_sharpness_wins_over_dark(sample_video: Path, tmp_path: Path) -> None:
    frames = [fp for _, fp in video.extract_keyframes(sample_video, tmp_path / "frames", interval_s=2)]
    result = quality.assess(frames, _quality_cfg(min_sharpness=1e9, min_brightness=1e9, min_resolution=1))
    assert result.reason == "low_sharpness"


# --- dedupe.sha256_file / phash_frames --------------------------------------------------


def test_sha256_file_deterministic(sample_video: Path) -> None:
    assert dedupe.sha256_file(sample_video) == dedupe.sha256_file(sample_video)


def test_sha256_file_differs_for_different_content(tmp_path: Path) -> None:
    a = _make_video(tmp_path / "a.mp4", duration=1, width=320, height=240)
    b = _make_video(tmp_path / "b.mp4", duration=1, width=320, height=240, bitrate="500k")
    assert dedupe.sha256_file(a) != dedupe.sha256_file(b)


def test_phash_frames_format(sample_video: Path, tmp_path: Path) -> None:
    frames = [fp for _, fp in video.extract_keyframes(sample_video, tmp_path / "frames", interval_s=2)]
    result = dedupe.phash_frames(frames)
    parts = result.split(":")
    assert 1 <= len(parts) <= 3
    for part in parts:
        imagehash.hex_to_hash(part)  # must parse without error


def test_phash_frames_empty_list() -> None:
    assert dedupe.phash_frames([]) == ""


def test_phash_near_duplicate_across_bitrates(tmp_path: Path) -> None:
    hi = _make_video(tmp_path / "hi.mp4", duration=4, width=640, height=360, bitrate="2000k")
    lo = _make_video(tmp_path / "lo.mp4", duration=4, width=640, height=360, bitrate="150k")

    frames_hi = [fp for _, fp in video.extract_keyframes(hi, tmp_path / "fhi", interval_s=1)]
    frames_lo = [fp for _, fp in video.extract_keyframes(lo, tmp_path / "flo", interval_s=1)]

    ph_hi = [imagehash.hex_to_hash(h) for h in dedupe.phash_frames(frames_hi).split(":")]
    ph_lo = [imagehash.hex_to_hash(h) for h in dedupe.phash_frames(frames_lo).split(":")]

    min_distance = min(a - b for a in ph_hi for b in ph_lo)
    assert min_distance <= 5

    assert dedupe.sha256_file(hi) != dedupe.sha256_file(lo)


# --- processors.run_processing (integration) --------------------------------------------


class _FakeDetector:
    """Stand-in for AnimalDetector: counts instantiations, returns scripted detections."""

    instances = 0

    def __init__(self, cfg) -> None:
        _FakeDetector.instances += 1
        self.result = _FakeDetector.next_result

    def detect(self, frames):
        return list(self.result)

    def summarize(self, results):
        if not results:
            return None, 0.0
        return results[0].class_name, sum(r.conf for r in results) / len(results)


_FakeDetector.next_result = []


def _insert_video(db, path: Path, *, source: str = "test", source_id: str = "1") -> Video:
    with session_scope() as session:
        row = Video(source=source, source_id=source_id, file_path=str(path), status=VideoStatus.DOWNLOADED)
        session.add(row)
        session.flush()
        video_id = row.id
    with session_scope() as session:
        return session.get(Video, video_id)


def _get_video(video_id: int) -> Video:
    with session_scope() as session:
        return session.get(Video, video_id)


def test_run_processing_rejects_short_video_on_duration(db, tmp_cfg, short_video: Path, tmp_path: Path) -> None:
    src = tmp_path / "short.mp4"
    shutil.copy(short_video, src)
    row = _insert_video(db, src)

    result = run_processing(tmp_cfg, detect_animals=False, check_quality=False)

    assert result == {"processed": 0, "rejected": 1, "errors": 0, "rejected_duration": 1}
    assert _get_video(row.id).status == VideoStatus.REJECTED
    assert _get_video(row.id).reject_reason == "duration"


def test_run_processing_processes_compliant_video(db, tmp_cfg, tmp_path: Path) -> None:
    src = tmp_path / "ok.mp4"
    _make_video(src, duration=6, width=640, height=360)
    row = _insert_video(db, src)

    result = run_processing(tmp_cfg, detect_animals=False, check_quality=False)

    assert result == {"processed": 1, "rejected": 0, "errors": 0}
    updated = _get_video(row.id)
    assert updated.status == VideoStatus.PROCESSED
    assert updated.sha256 is not None
    assert updated.phash


def test_count_cuts_tells_a_single_take_from_a_stitched_one(tmp_path: Path) -> None:
    single = _make_video(tmp_path / "single.mp4", duration=6, width=320, height=240)
    assert video.count_cuts(single) == 0
    assert video.count_cuts(_stitch(tmp_path / "stitched.mp4", tmp_path)) >= 2


def test_run_processing_rejects_someone_elses_compilation(db, tmp_cfg, tmp_path: Path) -> None:
    row = _insert_video(db, _stitch(tmp_path / "top10.mp4", tmp_path))

    result = run_processing(tmp_cfg, detect_animals=False, check_quality=False)

    assert result == {"processed": 0, "rejected": 1, "errors": 0, "rejected_compilation": 1}
    assert _get_video(row.id).reject_reason == "compilation"


def test_run_processing_rejects_a_clip_with_a_ranking_burned_into_it(
    db, tmp_cfg, tmp_path: Path, monkeypatch
) -> None:
    src = tmp_path / "captioned.mp4"
    _make_video(src, duration=6, width=640, height=360)
    row = _insert_video(db, src)
    monkeypatch.setattr(processors, "has_text", lambda frames, cfg: bool(frames))

    result = run_processing(tmp_cfg, detect_animals=False, check_quality=False)

    assert result == {"processed": 0, "rejected": 1, "errors": 0, "rejected_burned_text": 1}
    assert _get_video(row.id).reject_reason == "burned_text"


def test_run_processing_rejects_a_watermark_that_only_shows_up_late(
    db, tmp_cfg, tmp_path: Path, monkeypatch
) -> None:
    """One middle frame let a TikTok handle and a "Dola AI" corner mark through into a
    finished short: both fade in after the clip has started."""
    src = tmp_path / "late-mark.mp4"
    _make_video(src, duration=6, width=640, height=360)
    row = _insert_video(db, src)
    read: list[Path] = []

    def only_the_last(frames: list[Path], cfg) -> bool:
        # unpacking, not frames[0]: the gate passes one frame per call, and a signature
        # change on the real has_text has to fail here rather than in a night's processing run
        (frame,) = frames
        read.append(frame)
        return frame == sorted(frame.parent.glob("frame_*.jpg"))[-1]

    monkeypatch.setattr(processors, "has_text", only_the_last)

    result = run_processing(tmp_cfg, detect_animals=False, check_quality=False)

    assert result == {"processed": 0, "rejected": 1, "errors": 0, "rejected_burned_text": 1}
    assert _get_video(row.id).reject_reason == "burned_text"
    assert read  # the gate read the frame the mark is on, not only the middle one


def test_run_processing_keeps_the_videos_it_finished_when_the_run_is_killed(
    db, tmp_cfg, tmp_path: Path, monkeypatch
) -> None:
    """A killed run used to roll the whole loop back -- but the file moves it had already
    made are not in the transaction. 35 rows ended up saying "downloaded, in unsorted/"
    while the file sat in cat/, and every retry died in ffprobe."""
    first = _make_video(tmp_path / "first.mp4", duration=2)  # too short: rejected, no transcode
    second = _make_video(tmp_path / "second.mp4", duration=2)
    row = _insert_video(db, first, source_id="first")
    _insert_video(db, second, source_id="second")
    real = processors._process_one
    seen: list[int] = []

    def killed_on_the_second(session, video_row, cfg, **kwargs):
        seen.append(video_row.id)
        if len(seen) > 1:
            raise KeyboardInterrupt  # what a ^C or a job kill looks like from inside the loop
        real(session, video_row, cfg, **kwargs)

    monkeypatch.setattr(processors, "_process_one", killed_on_the_second)

    with pytest.raises(KeyboardInterrupt):
        run_processing(tmp_cfg, detect_animals=False, check_quality=False)

    assert _get_video(row.id).status == VideoStatus.REJECTED


def test_run_processing_quality_rejects_low_resolution(db, tmp_cfg, tmp_path: Path) -> None:
    src = tmp_path / "lowres.mp4"
    _make_video(src, duration=6, width=640, height=360)  # below default min_resolution=720
    row = _insert_video(db, src)

    result = run_processing(tmp_cfg, detect_animals=False, check_quality=True)

    assert result == {"processed": 0, "rejected": 1, "errors": 0, "rejected_low_resolution": 1}
    assert _get_video(row.id).reject_reason == "low_resolution"


def test_run_processing_dedupe_rejects_exact_duplicate(db, tmp_cfg, tmp_path: Path) -> None:
    original = tmp_path / "orig.mp4"
    _make_video(original, duration=6, width=640, height=360)
    copy_path = tmp_path / "copy.mp4"
    shutil.copy(original, copy_path)

    row_a = _insert_video(db, original, source_id="a")
    row_b = _insert_video(db, copy_path, source_id="b")

    result = run_processing(tmp_cfg, detect_animals=False, check_quality=False)

    assert result["processed"] == 1
    assert result["rejected"] == 1
    assert result.get("rejected_duplicate") == 1
    statuses = {row_a.id: _get_video(row_a.id).status, row_b.id: _get_video(row_b.id).status}
    assert VideoStatus.REJECTED in statuses.values()
    assert VideoStatus.PROCESSED in statuses.values()


def test_run_processing_does_not_reject_a_video_as_its_own_duplicate(db, tmp_cfg, tmp_path: Path) -> None:
    """fetch stores sha256 at download time; an already-h264 file keeps that hash through
    normalize, so dedupe must not match the row against itself."""
    src = tmp_path / "already_h264.mp4"
    _make_video(src, duration=6, width=640, height=360)
    row = _insert_video(db, src)
    with session_scope() as session:
        session.get(Video, row.id).sha256 = dedupe.sha256_file(src)

    result = run_processing(tmp_cfg, detect_animals=False, check_quality=False)

    assert result == {"processed": 1, "rejected": 0, "errors": 0}
    assert _get_video(row.id).status == VideoStatus.PROCESSED


def test_run_processing_does_not_reprocess_already_processed(
    db, tmp_cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    src = tmp_path / "ok.mp4"
    _make_video(src, duration=6, width=640, height=360)
    _insert_video(db, src)

    first = run_processing(tmp_cfg, detect_animals=False, check_quality=False)
    assert first["processed"] == 1

    calls = {"n": 0}
    original_probe = video.probe

    def _counting_probe(path):
        calls["n"] += 1
        return original_probe(path)

    monkeypatch.setattr(video, "probe", _counting_probe)

    second = run_processing(tmp_cfg, detect_animals=False, check_quality=False)

    assert second == {"processed": 0, "rejected": 0, "errors": 0}
    assert calls["n"] == 0


def test_run_processing_error_on_one_video_does_not_abort_run(db, tmp_cfg, tmp_path: Path) -> None:
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a real video")
    good = tmp_path / "good.mp4"
    _make_video(good, duration=6, width=640, height=360)

    _insert_video(db, bad, source_id="bad")
    _insert_video(db, good, source_id="good")

    result = run_processing(tmp_cfg, detect_animals=False, check_quality=False)

    assert result["errors"] == 1
    assert result["processed"] == 1


def test_run_processing_detects_animal_and_moves_file(
    db, tmp_cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.processors.AnimalDetector", _FakeDetector)
    _FakeDetector.next_result = [
        DetectionResult(ts_s=0.0, class_name="cat", conf=0.9, x=1.0, y=2.0, w=3.0, h=4.0)
    ]

    src = tmp_path / "animal.mp4"
    _make_video(src, duration=6, width=640, height=360)
    row = _insert_video(db, src, source="pexels", source_id="cat1")

    result = run_processing(tmp_cfg, detect_animals=True, check_quality=False)

    assert result == {"processed": 1, "rejected": 0, "errors": 0}
    updated = _get_video(row.id)
    assert updated.has_animal is True
    assert updated.category == "cat"
    assert updated.detect_conf == pytest.approx(0.9)
    assert updated.status == VideoStatus.PROCESSED

    expected_path = files.video_path(tmp_cfg, "cat", "pexels", "cat1")
    assert Path(updated.file_path) == expected_path
    assert expected_path.exists()
    assert not src.exists()

    with session_scope() as session:
        detections = session.get(Video, row.id).detections
        assert len(detections) == 1
        assert detections[0].class_name == "cat"


def test_run_processing_no_animal_rejects_and_takes_the_file_with_it(
    db, tmp_cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.processors.AnimalDetector", _FakeDetector)
    _FakeDetector.next_result = []

    src = tmp_path / "no_animal.mp4"
    _make_video(src, duration=6, width=640, height=360)
    row = _insert_video(db, src, source="pexels", source_id="noanimal1")

    result = run_processing(tmp_cfg, detect_animals=True, check_quality=False)

    assert result == {"processed": 0, "rejected": 1, "errors": 0, "rejected_no_animal": 1}
    updated = _get_video(row.id)
    assert updated.reject_reason == "no_animal"
    # nothing rejected is coming back, so the run deletes it instead of leaving it for `prune`
    assert not src.exists()
    assert updated.file_path is None


def test_run_processing_instantiates_detector_once_across_many_videos(
    db, tmp_cfg, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _FakeDetector.instances = 0
    monkeypatch.setattr("src.processors.AnimalDetector", _FakeDetector)
    # dedupe itself is covered by dedicated tests above; here we only care that all 20
    # near-identical synthetic clips reach the detector on a single shared instance.
    monkeypatch.setattr(dedupe, "find_duplicate", lambda *a, **k: None)
    _FakeDetector.next_result = [
        DetectionResult(ts_s=0.0, class_name="dog", conf=0.8, x=0.0, y=0.0, w=1.0, h=1.0)
    ]

    for i in range(20):
        # distinct bitrate per file -> distinct sha256, so the UNIQUE(sha256) column is happy
        path = tmp_path / f"v{i}.mp4"
        _make_video(path, duration=5, width=640, height=360, bitrate=f"{200 + i}k")
        _insert_video(db, path, source="pexels", source_id=f"v{i}")

    result = run_processing(tmp_cfg, detect_animals=True, check_quality=False)

    assert result["processed"] == 20
    assert _FakeDetector.instances == 1
