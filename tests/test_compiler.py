"""Tests for src.compiler: clip selection, the local-model plan, and the 9:16 render.

The model is never called for real -- ``urlopen`` is stubbed, so the suite stays offline.
ffmpeg/ffprobe are used as in tests/test_processors.py.
"""
from __future__ import annotations

import io
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from src.compiler import CompileError, _segment_start, build_short
from src.compiler import plan as plan_mod
from src.compiler import render
from src.storage.db import session_scope
from src.storage.models import Detection, Video, VideoStatus


def _make_video(
    path: Path, *, duration: float, width: int = 640, height: int = 360, tone_hz: int | None = None
) -> Path:
    cmd = ["ffmpeg", "-y",
           "-f", "lavfi", "-i", f"testsrc=size={width}x{height}:rate=25:duration={duration}"]
    if tone_hz is not None:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency={tone_hz}:duration={duration}",
                "-c:a", "aac", "-ac", "2"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-shortest", str(path)]
    subprocess.run(cmd, check=True, capture_output=True)
    return path


def _probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True,
    ).stdout
    return float(out)


def _stub_ollama(monkeypatch, payload: dict) -> list[dict]:
    """Replace urlopen with a canned Ollama reply; returns the list of sent requests."""
    sent: list[dict] = []

    class _Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            self.close()
            return False

    def fake_urlopen(request, timeout=None):
        sent.append(json.loads(request.data))
        body = {"message": {"content": json.dumps(payload)}}
        return _Response(json.dumps(body).encode())

    monkeypatch.setattr(plan_mod.urllib.request, "urlopen", fake_urlopen)
    return sent


def _insert_clip(path: Path, video_id_hint: str, *, category: str = "dog", first_ts: float = 0.0) -> int:
    with session_scope() as session:
        row = Video(
            source="test", source_id=video_id_hint, file_path=str(path), category=category,
            duration_s=6.0, width=640, height=360, has_animal=True, status=VideoStatus.PROCESSED,
        )
        session.add(row)
        session.flush()
        session.add(
            Detection(video_id=row.id, ts_s=first_ts, class_name=category, conf=0.9, x=0, y=0, w=10, h=10)
        )
        return row.id


# --- segment placement -------------------------------------------------------------


def test_segment_start_leads_in_before_the_first_detection() -> None:
    assert _segment_start(duration_s=30.0, first_detection_s=12.0, segment_s=8.0) == 11.0


def test_segment_start_never_goes_negative() -> None:
    assert _segment_start(duration_s=30.0, first_detection_s=0.0, segment_s=8.0) == 0.0


def test_segment_start_clamps_so_the_segment_fits_in_the_clip() -> None:
    assert _segment_start(duration_s=10.0, first_detection_s=9.5, segment_s=8.0) == 2.0


# --- plan --------------------------------------------------------------------------


def test_make_plan_sends_a_schema_pinned_to_the_clip_count(tmp_cfg, monkeypatch) -> None:
    sent = _stub_ollama(monkeypatch, {
        "category": "Paws & Chaos", "title": "Dogs Go Wild", "hook": "Watch this",
        "captions": ["one", "two"],
    })
    clips = [
        plan_mod.Clip(1, "dog", ["zoomies"], seen="a dog in a strawberry hat"),
        plan_mod.Clip(2, "cat", []),
    ]

    result = plan_mod.make_plan(clips, tmp_cfg.compiler)

    assert result.title == "Dogs Go Wild"
    # what the vision model saw reaches the prompt, otherwise the captions are invented
    assert "a dog in a strawberry hat" in sent[0]["messages"][1]["content"]
    schema = sent[0]["format"]["properties"]["captions"]
    assert (schema["minItems"], schema["maxItems"]) == (2, 2)
    assert sent[0]["think"] is False  # reasoning block off, we only want the JSON


def test_make_plan_rejects_a_reply_with_the_wrong_caption_count(tmp_cfg, monkeypatch) -> None:
    _stub_ollama(monkeypatch, {
        "category": "c", "title": "t", "hook": "h", "captions": ["only one"],
    })
    clips = [plan_mod.Clip(1, "dog", []), plan_mod.Clip(2, "cat", [])]

    with pytest.raises(plan_mod.PlanError, match="1 captions for 2 clips"):
        plan_mod.make_plan(clips, tmp_cfg.compiler)


def test_make_plan_reports_an_unreachable_ollama(tmp_cfg, monkeypatch) -> None:
    def refuse(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(plan_mod.urllib.request, "urlopen", refuse)

    with pytest.raises(plan_mod.PlanError, match="ollama serve"):
        plan_mod.make_plan([plan_mod.Clip(1, "dog", [])], tmp_cfg.compiler)


# --- vision ------------------------------------------------------------------------


def test_describe_frame_sends_the_image_and_returns_one_line(tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    frame = render.grab_frame(_make_video(tmp_path / "c.mp4", duration=1), tmp_path / "look.jpg", 0.5)
    cfg = replace(tmp_cfg.compiler, vision_model="qwen2.5vl:7b")
    sent: list[dict] = []

    def fake_urlopen(request, timeout=None):
        sent.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"message": {"content": " a dog\n in a hat "}}).encode())

    monkeypatch.setattr(plan_mod.urllib.request, "urlopen", fake_urlopen)

    assert plan_mod.describe_frame(frame, cfg, hint="dog") == "a dog in a hat"
    assert sent[0]["model"] == "qwen2.5vl:7b"
    assert sent[0]["messages"][0]["images"]  # base64 of the jpeg travelled with it
    # the YOLO class travels as a correctable hint, not as the answer
    assert 'labelled the animal "dog"' in sent[0]["messages"][0]["content"]


def test_describe_frame_falls_back_quietly_when_vision_is_off_or_down(tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    frame = tmp_path / "look.jpg"
    frame.write_bytes(b"not really a jpeg")

    assert plan_mod.describe_frame(frame, tmp_cfg.compiler) == ""  # vision_model="" in tests

    def refuse(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(plan_mod.urllib.request, "urlopen", refuse)
    # a missing vision model must not sink the whole compilation
    assert plan_mod.describe_frame(frame, replace(tmp_cfg.compiler, vision_model="x")) == ""


# --- caption layer -----------------------------------------------------------------


def test_caption_png_is_a_transparent_canvas_with_wrapped_text(tmp_cfg, tmp_path: Path) -> None:
    out = render.caption_png(
        "a very long caption that has to wrap across several lines to fit",
        tmp_path / "cap.png", tmp_cfg.compiler.font, 72, top=False,
    )

    image = Image.open(out)
    assert image.size == (render.WIDTH, render.HEIGHT)
    assert image.mode == "RGBA"
    # text sits in the lower half and leaves the upper-left corner untouched
    assert image.getpixel((0, 0))[3] == 0
    assert image.crop((0, render.HEIGHT // 2, render.WIDTH, render.HEIGHT)).getbbox() is not None


def test_pick_sound_asks_the_model_which_track_suits_the_silent_clip(monkeypatch, tmp_cfg) -> None:
    sent = _stub_ollama(monkeypatch, {"index": 1})

    choice = plan_mod.pick_sound("a cat asleep", ["dogs wrestling", "a cat yawning"],
                                 tmp_cfg.compiler)

    assert choice == 1
    prompt = sent[0]["messages"][1]["content"]
    assert "a cat asleep" in prompt and "1) a cat yawning" in prompt


def test_pick_sound_falls_back_to_the_first_track_on_an_answer_out_of_range(
    monkeypatch, tmp_cfg
) -> None:
    _stub_ollama(monkeypatch, {"index": 9})

    assert plan_mod.pick_sound("a cat asleep", ["one", "two"], tmp_cfg.compiler) == 0


def test_ranking_png_lights_up_the_running_row_on_the_left(tmp_cfg, tmp_path: Path) -> None:
    names = ["one", "two", "three", "four", "five"]
    first = render.ranking_png(names, 0, tmp_path / "r0.png", tmp_cfg.compiler.font, 46)
    third = render.ranking_png(names, 2, tmp_path / "r2.png", tmp_cfg.compiler.font, 46)

    box = Image.open(first).getbbox()
    assert box[0] < render.WIDTH // 2 and box[2] < render.WIDTH  # a column, not full width
    assert box[1] > 0 and box[3] < render.HEIGHT  # vertically centred block
    # only the highlighted row differs between the two layers
    assert Image.open(first).tobytes() != Image.open(third).tobytes()


def test_ranking_png_hides_the_names_of_the_rows_not_reached_yet(tmp_cfg, tmp_path: Path) -> None:
    names = ["a", "b", "cccccccccccccccccccc"]
    first = render.ranking_png(names, 0, tmp_path / "r0.png", tmp_cfg.compiler.font, 46)
    last = render.ranking_png(names, 2, tmp_path / "r2.png", tmp_cfg.compiler.font, 46)

    # rows 2 and 3 are bare numbers on the first clip, so the block is narrower there
    assert Image.open(first).getbbox()[2] < Image.open(last).getbbox()[2]
    # ... but the rows keep their places, so the list does not jump between segments
    assert Image.open(first).getbbox()[1] == Image.open(last).getbbox()[1]


def test_ranking_png_truncates_a_name_too_long_for_one_line(tmp_cfg, tmp_path: Path) -> None:
    out = render.ranking_png(
        ["word " * 40], 0, tmp_path / "r.png", tmp_cfg.compiler.font, 46
    )

    assert Image.open(out).getbbox()[2] <= int(render.WIDTH * 0.62) + render.WIDTH // 8


# --- render ------------------------------------------------------------------------


def test_render_segment_produces_the_shorts_canvas_with_audio(tmp_cfg, tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.mp4", duration=6, width=640, height=360)
    overlay = render.caption_png("hi", tmp_path / "o.png", tmp_cfg.compiler.font, 72, top=True)

    out = render.render_segment(source, tmp_path / "seg.mp4", 1.0, 2.0, [overlay])

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height",
         "-of", "json", str(out)],
        check=True, capture_output=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    video = next(s for s in streams if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (render.WIDTH, render.HEIGHT)
    # a silent track is added so concat does not lose the audio stream mid-way
    assert any(s["codec_type"] == "audio" for s in streams)


def test_render_segment_borrows_a_track_for_a_clip_with_no_audio(tmp_cfg, tmp_path: Path) -> None:
    silent = _make_video(tmp_path / "silent.mp4", duration=6)
    donor = _make_video(tmp_path / "donor.mp4", duration=2, tone_hz=440)
    fill = render.extract_audio(donor, tmp_path / "fill.m4a")

    out = render.render_segment(silent, tmp_path / "seg.mp4", 1.0, 4.0, [], fill)

    # the donor is looped to cover a segment twice its length, and it is audible
    assert render.mean_volume(out) > -40.0
    assert _probe_duration(out) == pytest.approx(4.0, abs=0.3)


# --- joining: transitions and the music bed ------------------------------------------


def test_join_filter_offsets_account_for_the_time_each_crossfade_eats() -> None:
    graph, video, audio = render._join_filter([4.0, 4.0, 4.0], transition_s=0.5)

    # first fade starts at 4-0.5=3.5, second at (3.5+4)-0.5=7.0 -- not at 8.0
    assert "offset=3.500" in graph and "offset=7.000" in graph
    assert (video, audio) == ("[vx2]", "[ax2]")


def test_join_filter_without_transitions_uses_plain_concat() -> None:
    graph, video, audio = render._join_filter([4.0, 4.0], transition_s=0.0)

    assert graph == "[0:v][0:a][1:v][1:a]concat=n=2:v=1:a=1[vout][aout]"
    assert (video, audio) == ("[vout]", "[aout]")


def test_mean_volume_ranks_a_loud_clip_above_a_silent_one(tmp_path: Path) -> None:
    loud = _make_video(tmp_path / "loud.mp4", duration=2, tone_hz=440)
    mute = _make_video(tmp_path / "mute.mp4", duration=2)

    assert render.mean_volume(mute) == float("-inf")
    assert render.mean_volume(loud) > -40.0


def test_join_crossfades_the_segments_and_mixes_the_music_bed(tmp_path: Path) -> None:
    segments = [
        _make_video(tmp_path / f"seg{index}.mp4", duration=2, width=320, height=240, tone_hz=440)
        for index in range(2)
    ]
    bed = render.extract_audio(segments[0], tmp_path / "bed.m4a")

    out = render.join(segments, tmp_path / "out.mp4", transition_s=0.5, music=bed, music_volume=0.2)

    # 2 + 2 minus the half second the two clips overlap
    assert _probe_duration(out) == pytest.approx(3.5, abs=0.2)
    codecs = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True,
    ).stdout.decode().split()
    assert "audio" in codecs and "video" in codecs


# --- end to end --------------------------------------------------------------------


def test_build_short_assembles_the_picked_clips(db, tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    _stub_ollama(monkeypatch, {
        "category": "Paws & Chaos", "title": "Dogs Go Wild", "hook": "Watch this",
        "captions": ["one", "two"],
    })
    for index in range(2):
        clip = _make_video(tmp_path / f"clip{index}.mp4", duration=6)
        _insert_clip(clip, f"v{index}", first_ts=1.0)

    out = build_short(tmp_cfg)

    assert out.name == "dogs-go-wild.mp4"
    assert out.is_file()
    duration = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(out)],
        check=True, capture_output=True,
    ).stdout
    assert float(duration) == pytest.approx(2 * tmp_cfg.compiler.segment_seconds, abs=0.5)
    assert not (tmp_cfg.compiler.output_path / ".work").exists()  # scratch cleaned up


def test_build_short_batch_does_not_reuse_the_same_clips(db, tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    _stub_ollama(monkeypatch, {
        "category": "Paws", "title": "Same Title", "hook": "Watch this",
        "captions": ["one", "two", "three"],
    })
    for index in range(6):
        _insert_clip(_make_video(tmp_path / f"clip{index}.mp4", duration=6), f"v{index}", first_ts=1.0)

    used: set[int] = set()
    first = build_short(tmp_cfg, used=used)
    second = build_short(tmp_cfg, used=used)

    assert len(used) == 6  # clips_per_short=3, no overlap between the two shorts
    # the model handed back the same title twice; neither file may overwrite the other
    assert {first.name, second.name} == {"same-title.mp4", "same-title-2.mp4"}
    assert first.is_file() and second.is_file()


def test_build_short_refuses_when_there_is_not_enough_material(db, tmp_cfg, tmp_path: Path) -> None:
    _insert_clip(_make_video(tmp_path / "only.mp4", duration=6), "v0")

    with pytest.raises(CompileError, match="at least 2 processed clips"):
        build_short(tmp_cfg)


def test_build_short_skips_clips_shorter_than_one_segment(db, tmp_cfg, tmp_path: Path) -> None:
    clip = _make_video(tmp_path / "tiny.mp4", duration=6)
    for index in range(2):
        video_id = _insert_clip(clip, f"v{index}")
        with session_scope() as session:
            session.get(Video, video_id).duration_s = 1.0  # below segment_seconds=2

    with pytest.raises(CompileError, match="at least 2 processed clips"):
        build_short(tmp_cfg)
