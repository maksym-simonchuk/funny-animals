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
from PIL import Image, ImageDraw

from src.compiler import CompileError, _segment_start, build_short, keep, recall
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
        plan_mod.Clip(1, "dog", ["zoomies"], seen="the animal is in a strawberry hat"),
        plan_mod.Clip(2, "cat", []),
    ]

    result = plan_mod.make_plan(clips, tmp_cfg.compiler, "Sneaky Thieves")

    assert result.title == "Dogs Go Wild"
    assert result.category == "Sneaky Thieves"  # the heading comes from the grouping pass
    assert "TOP 2 Sneaky Thieves" in sent[0]["messages"][1]["content"]
    # what the vision model saw reaches the prompt, otherwise the captions are invented
    assert "the animal is in a strawberry hat" in sent[0]["messages"][1]["content"]
    schema = sent[0]["format"]["properties"]["captions"]
    assert (schema["minItems"], schema["maxItems"]) == (2, 2)
    assert sent[0]["think"] is False  # reasoning block off, we only want the JSON


def test_make_plan_rejects_a_reply_with_the_wrong_caption_count(tmp_cfg, monkeypatch) -> None:
    _stub_ollama(monkeypatch, {
        "category": "c", "title": "t", "hook": "h", "captions": ["only one"],
    })
    clips = [plan_mod.Clip(1, "dog", []), plan_mod.Clip(2, "cat", [])]

    with pytest.raises(plan_mod.PlanError, match="1 captions for 2 clips"):
        plan_mod.make_plan(clips, tmp_cfg.compiler, "c")


def test_make_plan_reports_an_unreachable_ollama(tmp_cfg, monkeypatch) -> None:
    def refuse(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(plan_mod.urllib.request, "urlopen", refuse)

    with pytest.raises(plan_mod.PlanError, match="ollama serve"):
        plan_mod.make_plan([plan_mod.Clip(1, "dog", [])], tmp_cfg.compiler, "c")


# --- vision ------------------------------------------------------------------------


def test_describe_frames_sends_every_frame_and_returns_one_line(tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    clip = _make_video(tmp_path / "c.mp4", duration=1)
    frames = [render.grab_frame(clip, tmp_path / f"look{n}.jpg", at)
              for n, at in enumerate((0.1, 0.5, 0.9))]
    cfg = replace(tmp_cfg.compiler, vision_model="qwen2.5vl:7b")
    sent: list[dict] = []

    def fake_urlopen(request, timeout=None):
        sent.append(json.loads(request.data))
        reply = {"animal": "chihuahua", "scene": " the animal\n grabs a hat "}
        return io.BytesIO(json.dumps({"message": {"content": json.dumps(reply)}}).encode())

    monkeypatch.setattr(plan_mod.urllib.request, "urlopen", fake_urlopen)

    assert plan_mod.describe_frames(frames, cfg) == ("chihuahua", "the animal grabs a hat")
    assert sent[0]["model"] == "qwen2.5vl:7b"
    # all three travel in one question: one still cannot tell movement from furniture
    assert len(sent[0]["messages"][0]["images"]) == 3
    # the species stays out of it, or the text model opens every row with it
    assert "never naming it" in sent[0]["messages"][0]["content"]


def test_describe_frames_falls_back_quietly_when_vision_is_off_or_down(tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    frame = tmp_path / "look.jpg"
    frame.write_bytes(b"not really a jpeg")

    assert plan_mod.describe_frames([frame], tmp_cfg.compiler) == ("", "")  # vision_model=""

    def refuse(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(plan_mod.urllib.request, "urlopen", refuse)
    # a missing vision model must not sink the whole compilation
    assert plan_mod.describe_frames([frame], replace(tmp_cfg.compiler, vision_model="x")) == ("", "")


def test_rate_clip_weighs_an_unexpected_turn_double(tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    """Nothing used to rate the clips: a short was the first five of the fullest bucket in
    database order, so whether it was funny was down to what the scraper downloaded last."""
    frame = tmp_path / "look.jpg"
    frame.write_bytes(b"not really a jpeg")
    cfg = replace(tmp_cfg.compiler, vision_model="qwen2.5vl:7b")

    sent = _stub_ollama(monkeypatch, {"funny": "yes", "human": "no", "cute": "yes"})
    assert plan_mod.rate_clip([frame, frame], cfg) == 3
    # yes/no, not "rate this 1 to 5": asked for a number the model says 3 about everything
    assert sent[0]["format"]["properties"]["funny"]["enum"] == ["yes", "no"]

    _stub_ollama(monkeypatch, {"funny": "no", "human": "yes", "cute": "yes"})
    assert plan_mod.rate_clip([frame], cfg) == 2  # cute and human-like still lose to funny

    _stub_ollama(monkeypatch, {"funny": "no", "human": "no", "cute": "no"})
    assert plan_mod.rate_clip([frame], cfg) == 0


def test_rate_clip_puts_a_clip_it_could_not_see_mid_pack(tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    frame = tmp_path / "look.jpg"
    frame.write_bytes(b"not really a jpeg")

    assert plan_mod.rate_clip([frame], tmp_cfg.compiler) == plan_mod._UNRATED  # vision off

    def refuse(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(plan_mod.urllib.request, "urlopen", refuse)
    # buried at 0, one failed call would cost a clip the short it belonged in
    assert plan_mod.rate_clip([frame], replace(tmp_cfg.compiler, vision_model="x")) == 2


def test_heading_holds_asks_one_clip_at_a_time(tmp_cfg, monkeypatch) -> None:
    """Asked about all five at once the model waved "Sky Watchers" through on a set where
    one animal looked up and the rest looked at a table. One clip per question, and the
    first lie sinks the heading."""
    clips = [
        plan_mod.Clip(1, "bird", [], seen="the animal is watching the sky"),
        plan_mod.Clip(2, "bird", [], seen="the animal is walking on a table"),
        plan_mod.Clip(3, "cat", [], seen="the animal is watching the sky"),
    ]
    asked: list[str] = []

    def answer(prompt, question, cfg):
        asked.append(question)
        return "table" not in question

    monkeypatch.setattr(plan_mod, "_yes_no", answer)

    assert plan_mod.heading_holds("Sky Watchers", clips, tmp_cfg.compiler) is False
    assert len(asked) == 2  # stops at the clip the heading lies about
    assert plan_mod.heading_holds("Sky Watchers", clips[::2], tmp_cfg.compiler) is True


def test_recall_carries_spent_clips_and_themes_between_runs(tmp_cfg) -> None:
    """Empty at the start of every run, the second `compile` picked the same fullest
    bucket and remade the first one's shorts as `top-5-toy-destroyers-2.mp4`."""
    assert recall(tmp_cfg) == (set(), set())  # nothing built yet

    keep(tmp_cfg, {4, 7}, {"a toy", "toy destroyers"})
    assert recall(tmp_cfg) == ({4, 7}, {"a toy", "toy destroyers"})

    (tmp_cfg.compiler.output_path.parent / "compiled.json").write_text("{ truncated")
    assert recall(tmp_cfg) == (set(), set())  # a half-written note is not a crash


def test_has_text_asks_vision_and_stays_quiet_when_it_is_down(tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    clip = _make_video(tmp_path / "c.mp4", duration=1)
    frames = [render.grab_frame(clip, tmp_path / f"look{n}.jpg", at)
              for n, at in enumerate((0.1, 0.5, 0.9))]
    frame = frames[0]
    cfg = replace(tmp_cfg.compiler, vision_model="qwen2.5vl:7b")
    sent = _stub_ollama(monkeypatch, {"answer": "yes"})

    assert plan_mod.has_text(frames, cfg) is True
    # every frame of the look, not just the middle one: a caption can fade in halfway
    assert len(sent[0]["messages"][0]["images"]) == 3
    # the question used to wave small handles through, and a TikTok watermark reached a
    # finished short that way -- TikTok suppresses reach on another platform's mark
    assert "username" in sent[0]["messages"][0]["content"]
    assert "does not count" not in sent[0]["messages"][0]["content"]

    def refuse(request, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr(plan_mod.urllib.request, "urlopen", refuse)
    assert plan_mod.has_text([frame], cfg) is False  # vision down keeps the clip, not drops it


def test_group_clips_takes_the_fullest_bucket_and_strips_the_top_prefix(monkeypatch, tmp_cfg) -> None:
    # "answer" rides along: every candidate clip and the finished heading are checked
    sent = _stub_ollama(monkeypatch, {"theme": "TOP 3 Snack Bandits", "answer": "yes"})
    clips = [
        plan_mod.Clip(1, "dog", [], seen="one", tag="sleeping"),
        plan_mod.Clip(2, "dog", [], seen="two", tag="stealing food"),
        plan_mod.Clip(3, "dog", [], seen="three", tag="other"),
        plan_mod.Clip(4, "dog", [], seen="four", tag="stealing food"),
        plan_mod.Clip(5, "dog", [], seen="five", tag="stealing food"),
        plan_mod.Clip(6, "dog", [], seen="six", tag="stealing food"),
    ]

    chosen, theme = plan_mod.group_clips(clips, 3, tmp_cfg.compiler)

    # every clip in the short shares the tag, so the heading is true of all of them
    assert chosen == [1, 3, 4]
    assert theme == "Snack Bandits"  # the renderer prints "TOP 3" itself
    assert any("stealing food" in message["content"]
               for request in sent for message in request["messages"])


def test_group_clips_drops_a_clip_the_model_says_does_not_fit(monkeypatch, tmp_cfg) -> None:
    """The tag came out of one shot at temperature 0; the second opinion is what keeps a
    cockatoo facing a cat out of a compilation of animals playing with a toy."""
    def fake_fits(scene, label, cfg):
        return scene != "two"

    monkeypatch.setattr(plan_mod, "fits_tag", fake_fits)
    monkeypatch.setattr(plan_mod, "name_theme", lambda *args, **kwargs: "Toy Destroyers")
    clips = [
        plan_mod.Clip(1, "dog", [], seen="one", prop="a toy"),
        plan_mod.Clip(2, "dog", [], seen="two", prop="a toy"),
        plan_mod.Clip(3, "dog", [], seen="three", prop="a toy"),
        plan_mod.Clip(4, "dog", [], seen="four", prop="a toy"),
    ]

    assert plan_mod.group_clips(clips, 3, tmp_cfg.compiler) == ([0, 2, 3], "Toy Destroyers")


def test_group_clips_moves_on_to_a_theme_this_batch_has_not_used(monkeypatch, tmp_cfg) -> None:
    """Two shorts in a row off the pool's biggest bucket get the same heading over
    different clips; the second one has to look elsewhere."""
    monkeypatch.setattr(plan_mod, "fits_tag", lambda scene, label, cfg: True)
    monkeypatch.setattr(plan_mod, "name_theme", lambda tag, *args, **kwargs: tag.title())
    clips = [
        plan_mod.Clip(1, "dog", [], seen="one", tag="sleeping"),
        plan_mod.Clip(2, "dog", [], seen="two", tag="sleeping"),
        plan_mod.Clip(3, "dog", [], seen="three", tag="sleeping"),
        plan_mod.Clip(4, "dog", [], seen="four", tag="sleeping"),
        plan_mod.Clip(5, "dog", [], seen="five", tag="eating"),
        plan_mod.Clip(6, "dog", [], seen="six", tag="eating"),
    ]
    done: set[str] = set()

    assert plan_mod.group_clips(clips, 2, tmp_cfg.compiler, done) == ([0, 1], "Sleeping")
    # "sleeping" still has two clips going spare, and is skipped anyway
    assert plan_mod.group_clips(clips, 2, tmp_cfg.compiler, done) == ([4, 5], "Eating")
    # with every label spoken for, a used one comes back rather than a generic heading;
    # the name is what has to differ, and `done` is what name_theme reads for that
    assert plan_mod.group_clips(clips, 2, tmp_cfg.compiler, done)[0] == [0, 1]


def test_group_clips_takes_the_best_rated_clips_of_a_bucket_best_first(monkeypatch, tmp_cfg) -> None:
    """The bucket used to be spent on whichever five clips the scraper downloaded last."""
    monkeypatch.setattr(plan_mod, "fits_tag", lambda scene, label, cfg: True)
    monkeypatch.setattr(plan_mod, "name_theme", lambda *args, **kwargs: "Snack Bandits")
    clips = [
        plan_mod.Clip(1, "dog", [], seen="one", tag="stealing food", score=1),
        plan_mod.Clip(2, "dog", [], seen="two", tag="stealing food", score=4),
        plan_mod.Clip(3, "dog", [], seen="three", tag="stealing food", score=0),
        plan_mod.Clip(4, "dog", [], seen="four", tag="stealing food", score=3),
    ]

    # the two best, and the best of those opens the short
    assert plan_mod.group_clips(clips, 2, tmp_cfg.compiler) == ([1, 3], "Snack Bandits")


def test_group_clips_prefers_a_better_bucket_over_a_fuller_one(monkeypatch, tmp_cfg) -> None:
    """27 clips of an animal looking at something outnumbered every other bucket in the
    pool, and won every time on size alone -- five animals sitting still under a heading
    that promised a countdown."""
    monkeypatch.setattr(plan_mod, "fits_tag", lambda scene, label, cfg: True)
    monkeypatch.setattr(plan_mod, "name_theme", lambda tag, *args, **kwargs: tag.title())
    clips = [
        plan_mod.Clip(1, "dog", [], seen="a", tag="watching something", score=1),
        plan_mod.Clip(2, "dog", [], seen="b", tag="watching something", score=0),
        plan_mod.Clip(3, "dog", [], seen="c", tag="watching something", score=1),
        plan_mod.Clip(4, "dog", [], seen="d", tag="falling over", score=4),
        plan_mod.Clip(5, "dog", [], seen="e", tag="falling over", score=4),
    ]

    assert plan_mod.group_clips(clips, 2, tmp_cfg.compiler) == ([3, 4], "Falling Over")


def test_group_clips_stays_generic_when_no_tag_can_fill_a_short(tmp_cfg) -> None:
    clips = [
        plan_mod.Clip(1, "dog", [], tag="sleeping"),
        plan_mod.Clip(2, "cat", [], tag="eating"),
        plan_mod.Clip(3, "cat", [], tag="other"),
    ]

    # a themed heading over clips that do not share the theme is the bug being fixed
    assert plan_mod.group_clips(clips, 3, tmp_cfg.compiler) == ([0, 1, 2], "Funny Animals")


def test_group_clips_numbers_the_generic_heading_it_already_used(tmp_cfg) -> None:
    clips = [
        plan_mod.Clip(1, "dog", [], tag="sleeping"),
        plan_mod.Clip(2, "cat", [], tag="eating"),
        plan_mod.Clip(3, "cat", [], tag="other"),
    ]
    themes: set[str] = set()

    # three shorts off the same last resort are three editions, not one name three times
    headings = [plan_mod.group_clips(clips, 3, tmp_cfg.compiler, themes)[1] for _ in range(3)]

    assert headings == ["Funny Animals", "More Funny Animals", "Funny Animals 3"]


def test_tag_clip_answers_only_from_the_list(monkeypatch, tmp_cfg) -> None:
    sent = _stub_ollama(monkeypatch, {"tag": "stealing food"})

    assert plan_mod.tag_clip("the animal takes a strawberry", tmp_cfg.compiler) == "stealing food"
    # the enum is the decoding grammar, so an off-list answer is impossible by construction
    assert sent[0]["format"]["properties"]["tag"]["enum"][-1] == "other"

    _stub_ollama(monkeypatch, {"tag": "inventing something"})
    assert plan_mod.tag_clip("who knows", tmp_cfg.compiler) == "other"


# --- caption layer -----------------------------------------------------------------


def test_caption_png_is_a_transparent_canvas_with_wrapped_text(tmp_cfg, tmp_path: Path) -> None:
    out = render.caption_png(
        "a very long caption that has to wrap across several lines to fit",
        tmp_path / "cap.png", tmp_cfg.compiler.font, 72,
    )

    image = Image.open(out)
    assert image.size == (render.WIDTH, render.HEIGHT)
    assert image.mode == "RGBA"
    # text sits in the lower half and leaves the upper-left corner untouched
    assert image.getpixel((0, 0))[3] == 0
    assert image.crop((0, render.HEIGHT // 2, render.WIDTH, render.HEIGHT)).getbbox() is not None


def test_rubric_png_puts_the_cta_under_the_wrapped_heading(tmp_cfg, tmp_path: Path) -> None:
    """At a fixed y the prompt lands on the second line of a heading that wrapped, so it
    starts below whatever the heading actually used."""
    font, cta = tmp_cfg.compiler.font, "like if you love them too"
    one = render.rubric_png("TOP 5 DOGS", cta, tmp_path / "a.png", font, 92, 40)
    two = render.rubric_png("TOP 5 PROFESSIONAL LOUNGERS", cta, tmp_path / "b.png", font, 92, 40)

    assert Image.open(one).getbbox()[3] < Image.open(two).getbbox()[3]  # two lines push it down
    # and the ranking sits mid-frame, well clear of the whole top block
    assert Image.open(two).getbbox()[3] < render.HEIGHT // 3

    bare = render.rubric_png("TOP 5 DOGS", "", tmp_path / "c.png", font, 92, 40)
    assert Image.open(bare).getbbox()[3] < Image.open(one).getbbox()[3]  # "" drops the prompt


def test_make_plan_cuts_a_caption_the_model_ran_long(monkeypatch, tmp_cfg) -> None:
    _stub_ollama(monkeypatch, {
        "category": "c", "title": "t", "hook": "h",
        "captions": ["Dachshund with a toy? That's a hotdog!", "short one"],
    })
    clips = [plan_mod.Clip(1, "dog", []), plan_mod.Clip(2, "dog", [])]

    result = plan_mod.make_plan(clips, tmp_cfg.compiler, "c")

    # an overlong row stops being readable in the second it is on screen, and the cut
    # must not leave it hanging on an article
    assert result.captions == ["Dachshund with a toy", "short one"]


def test_make_plan_keeps_the_species_out_of_every_prompt_it_sends(monkeypatch, tmp_cfg) -> None:
    sent = _stub_ollama(monkeypatch, {
        "title": "t", "hook": "h", "captions": ["caught red pawed"],
        "line": "one two three four five six seven eight nine",
    })
    clips = [plan_mod.Clip(1, "dog", [], seen="the chihuahua wears a hat", animal="Chihuahua")]

    result = plan_mod.make_plan(clips, tmp_cfg.compiler, "c")

    for request in sent:
        # with a description in hand the YOLO class is dropped: it is the word the model
        # copies -- and vision names the species often enough that it is scrubbed as well
        assert "dog" not in request["messages"][1]["content"]
        assert "chihuahua" not in request["messages"][1]["content"].lower()
    # the bottom caption wraps, so it may be a sentence -- but not one that covers the clip
    assert result.lines == ["one two three four five six seven eight"]


def test_make_line_writes_one_caption_for_one_clip(monkeypatch, tmp_cfg) -> None:
    sent = _stub_ollama(monkeypatch, {"line": "caught eating the evidence"})

    line = plan_mod.make_line("the animal holds a strawberry", "STRAWBERRY BANDIT", tmp_cfg.compiler)

    assert line == "caught eating the evidence"
    # only this clip is in the prompt, so the caption cannot land on another one
    assert "the animal holds a strawberry" in sent[0]["messages"][1]["content"]
    assert "STRAWBERRY BANDIT" in sent[0]["messages"][1]["content"]


def test_make_line_refuses_the_description_read_back_at_it(monkeypatch, tmp_cfg) -> None:
    """A clip went out captioned "THE ANIMAL IS PARTIALLY HIDDEN BEHIND THE BABY," -- the
    model gave up on the joke and returned the prompt, cut mid-sentence by the word limit."""
    scene = "the animal is partially hidden behind the baby, watching the door"
    sent = _stub_ollama(monkeypatch, {"line": scene})

    assert plan_mod.make_line(scene, "SNEAKY GUEST", tmp_cfg.compiler) == ""
    assert len(sent) == 2  # asked again, got the description again, goes without a caption


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


def test_ranking_png_shrinks_the_list_instead_of_cutting_the_longest_row(
    tmp_cfg, tmp_path: Path
) -> None:
    # this row overruns the 669px limit at 46px, and used to come out as "...UNSTOPP..."
    names = ["SHORT ONE", "WATERMELON WIELDER, UNSTOPPABLE", "TINY"]
    out = render.ranking_png(names, 1, tmp_path / "r.png", tmp_cfg.compiler.font, 46)

    box = Image.open(out).getbbox()
    assert box[2] <= int(render.WIDTH * 0.62) + render.WIDTH // 8
    # the whole row survives: at the shrunk size it fits, so no word and no ellipsis
    font = render._load_font(tmp_cfg.compiler.font, 46)
    draw = ImageDraw.Draw(Image.new("RGBA", (render.WIDTH, render.HEIGHT)))
    shrunk = render._row_font(draw, [f"2. {names[1]}"], tmp_cfg.compiler.font, 46, int(render.WIDTH * 0.62))
    assert shrunk.size < font.size
    assert render._fit(draw, f"2. {names[1]}", shrunk, int(render.WIDTH * 0.62)).endswith("UNSTOPPABLE")


def test_fit_drops_whole_words_before_reaching_for_the_ellipsis(tmp_cfg) -> None:
    draw = ImageDraw.Draw(Image.new("RGBA", (render.WIDTH, render.HEIGHT)))
    font = render._load_font(tmp_cfg.compiler.font, 46)

    fitted = render._fit(draw, "1. WATERMELON WIELDER, UNSTOPPABLE", font, int(render.WIDTH * 0.62))

    assert fitted == "1. WATERMELON WIELDER"


# --- render ------------------------------------------------------------------------


def test_render_segment_produces_the_shorts_canvas_with_audio(tmp_cfg, tmp_path: Path) -> None:
    source = _make_video(tmp_path / "clip.mp4", duration=6, width=640, height=360)
    overlay = render.caption_png("hi", tmp_path / "o.png", tmp_cfg.compiler.font, 72)

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


def test_mean_volume_ignores_the_sub_bass_rumble_of_a_hand_held_clip(tmp_path: Path) -> None:
    rumble = _make_video(tmp_path / "rumble.mp4", duration=2, tone_hz=40)
    music = _make_video(tmp_path / "music.mp4", duration=2, tone_hz=1000)

    # same amplitude, but only one of the two can be heard on a phone speaker: ranked on
    # raw level the rumble wins and a clip of pure handling noise ends up the soundtrack
    assert render.mean_volume(rumble) < render.mean_volume(music) - 20


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


def test_look_at_clips_looks_again_only_at_what_an_older_cache_wrote(tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    """A first look over a pool this size is the better part of an hour, so an entry the
    current version wrote is never repeated -- and one written before the clips were rated
    always is."""
    import src.compiler as compiler_mod

    clip = _make_video(tmp_path / "c.mp4", duration=6)
    cache_path = tmp_cfg.compiler.output_path.parent / "vision.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "7": {"animal": "dog", "scene": "the animal lies on a bed",  # v1: one frame, no score
              "text": False, "tag": "sleeping", "prop": "a bed"},
        "8": {"v": 2, "animal": "cat", "scene": "the animal jumps off a shelf",
              "text": False, "score": 4, "tag": "jumping", "prop": "furniture"},
    }))
    looks: list[list[Path]] = []

    def fake_describe(frames, cfg):
        looks.append(frames)
        return "dog", "the animal drags a sock off the sofa"

    monkeypatch.setattr(compiler_mod, "describe_frames", fake_describe)
    monkeypatch.setattr(compiler_mod, "has_text", lambda frames, cfg: False)
    monkeypatch.setattr(compiler_mod, "rate_clip", lambda frames, cfg: 3)
    monkeypatch.setattr(compiler_mod, "tag_clip", lambda scene, cfg: "stealing food")
    monkeypatch.setattr(compiler_mod, "tag_prop", lambda scene, cfg: "a shoe")
    monkeypatch.setattr(compiler_mod, "unload", lambda model, cfg: None)

    picked = [(7, str(clip), "dog", [], 6.0, 0.0), (8, str(clip), "cat", [], 6.0, 0.0)]
    clips = compiler_mod._look_at_clips(
        picked, replace(tmp_cfg.compiler, vision_model="qwen2.5vl:7b"), tmp_path / "work"
    )

    assert len(looks) == 1  # the up-to-date entry is not looked at again
    assert len(looks[0]) == 3  # and the stale one is shown three frames, not one
    assert (clips[0].score, clips[0].seen) == (3, "the animal drags a sock off the sofa")
    assert (clips[1].score, clips[1].tag) == (4, "jumping")
    # the re-look is on disk before the labels run: an hour of inference is not redone
    assert json.loads(cache_path.read_text())["7"]["score"] == 3


# --- post copy ----------------------------------------------------------------------


def test_make_copy_lays_out_both_platforms(tmp_cfg, monkeypatch) -> None:
    _stub_ollama(monkeypatch, {
        "youtube_title": "Top 5 Tall Boys",
        "youtube_description": "Five animals on their hind legs.",
        "tiktok_caption": "The alpaca refuses to move 😭",
        "youtube_tags": ["funnyanimals", "cuteanimals", "pets"],
        "tiktok_tags": ["fyp", "funny"],
    })
    clips = [plan_mod.Clip(1, "dog", [], seen="the animal stands on its hind legs")]
    plan = plan_mod.Plan(category="Tall Boys", title="Tall Boys", hook="", captions=[], lines=[])

    text = plan_mod.make_copy(clips, plan, tmp_cfg.compiler)

    assert text.splitlines()[0] == "YOUTUBE SHORTS"
    assert "Top 5 Tall Boys #shorts" in text
    # the reach tags lead, what the model picked for this video follows
    assert "#funnyanimals #animals #animalvideos #cuteanimals #pets\n" in text
    # the TikTok half comes second, and the two are not the same copy twice
    assert text.index("TIKTOK") > text.index("Top 5 Tall Boys")
    assert "The alpaca refuses to move" in text
    assert text.rstrip().endswith("#fyp #viral #funnyanimals #animals #funny")


def test_make_copy_keeps_only_lowercase_tags_off_the_list(tmp_cfg, monkeypatch) -> None:
    _stub_ollama(monkeypatch, {
        "youtube_title": "Top 5 Tall Boys #Shorts",
        "youtube_description": "Five animals on their hind legs. #PetVids",
        "tiktok_caption": "The alpaca refuses to move #TallBoys",
        "youtube_tags": ["Funnyanimals", "#pets", "PetVids", "shorts", "pets"],
        "tiktok_tags": ["fyp"],
    })
    plan = plan_mod.Plan(category="Tall Boys", title="Tall Boys", hook="", captions=[], lines=[])

    text = plan_mod.make_copy([plan_mod.Clip(1, "dog", [])], plan, tmp_cfg.compiler)

    # invented tags are dropped, casing is flattened, the title's #shorts is not repeated,
    # and a tag written into the prose is not left there next to the tag line
    assert "#funnyanimals #animals #animalvideos #pets\n" in text
    assert "#PetVids" not in text and "#TallBoys" not in text
    assert text.count("#shorts") == 1
    assert text.splitlines()[1] == "Top 5 Tall Boys #shorts"


def test_make_copy_trims_a_youtube_title_over_the_limit(tmp_cfg, monkeypatch) -> None:
    long_title = "Absolutely " * 12 + "Tall Boys"
    sent = _stub_ollama(monkeypatch, {
        "youtube_title": long_title,
        "youtube_description": "d", "tiktok_caption": "t",
        "youtube_tags": ["pets"], "tiktok_tags": ["fyp"],
    })
    plan = plan_mod.Plan(category="Tall Boys", title="Tall Boys", hook="", captions=[], lines=[])

    text = plan_mod.make_copy([plan_mod.Clip(1, "dog", [])], plan, tmp_cfg.compiler)

    title = text.splitlines()[1]
    assert len(sent) == 2  # asked once more before cutting it
    assert len(title) <= 100 and not title.endswith("Absolutel")  # cut on a word boundary


def test_build_short_writes_the_copy_next_to_the_mp4(db, tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    _stub_ollama(monkeypatch, {
        "category": "Paws", "title": "Dogs Go Wild", "hook": "Watch this",
        "captions": ["one", "two"],
        "youtube_title": "Top 5 Dogs Going Wild",
        "youtube_description": "Five dogs, one couch.",
        "tiktok_caption": "Number 1 broke me 😭",
        "youtube_tags": ["dogs", "funnyanimals", "pets"], "tiktok_tags": ["fyp"],
    })
    for index in range(2):
        _insert_clip(_make_video(tmp_path / f"clip{index}.mp4", duration=6), f"v{index}", first_ts=1.0)

    out = build_short(tmp_cfg)

    copy = out.with_suffix(".txt")
    assert copy.is_file()
    assert "Top 5 Dogs Going Wild #shorts" in copy.read_text()
    assert "TIKTOK" in copy.read_text()


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


def test_build_short_with_named_clips_uses_exactly_those(db, tmp_cfg, tmp_path: Path, monkeypatch) -> None:
    _stub_ollama(monkeypatch, {
        "theme": "Paw Posers",
        "category": "Paw Posers", "title": "Paw Posers", "hook": "Watch this",
        "captions": ["one", "two", "three"],
    })
    ids = [
        _insert_clip(_make_video(tmp_path / f"clip{index}.mp4", duration=6), f"v{index}", first_ts=1.0)
        for index in range(5)
    ]

    used: set[int] = set()
    out = build_short(tmp_cfg, used=used, clip_ids=[ids[3], ids[0], ids[4]])

    assert out.is_file()
    assert used == {ids[0], ids[3], ids[4]}  # the other two stay out, theme picking skipped


def test_build_short_rejects_a_clip_id_that_is_not_in_the_pool(db, tmp_cfg, tmp_path: Path) -> None:
    video_id = _insert_clip(_make_video(tmp_path / "one.mp4", duration=6), "v0", first_ts=1.0)

    with pytest.raises(CompileError, match=r"not processed clips.*\[999\]"):
        build_short(tmp_cfg, clip_ids=[video_id, 999])


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
