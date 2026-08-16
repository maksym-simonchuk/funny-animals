"""Stage A tests: pure functions in src/export, no DB involved."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest
from pycocotools.coco import COCO

from src.export import run_export, select_export_videos, split_for
from src.export.coco import build_coco, export_coco
from src.export.hf import build_hf_dataset, check_push_prerequisites, export_hf
from src.export.webdataset import export_webdataset, write_shards
from src.storage.db import session_scope
from src.storage.files import frames_dir
from src.storage.models import Detection, Video, VideoStatus


def _insert_video(**overrides) -> int:
    defaults = dict(
        source="pexels",
        source_id="vid-1",
        page_url="https://example.com/p",
        download_url="https://example.com/d",
        license="cc0",
        status=VideoStatus.PROCESSED,
        width=640,
        height=480,
    )
    defaults.update(overrides)
    with session_scope() as session:
        video = Video(**defaults)
        session.add(video)
        session.flush()
        return video.id


def _insert_detection(video_id: int, **overrides) -> None:
    defaults = dict(ts_s=0.0, class_name="cat", conf=0.9, x=1.0, y=2.0, w=3.0, h=4.0)
    defaults.update(overrides)
    with session_scope() as session:
        session.add(Detection(video_id=video_id, **defaults))


# --- split_for -------------------------------------------------------------


def test_split_for_deterministic():
    first = split_for("pexels", "abc123", 0.8)
    second = split_for("pexels", "abc123", 0.8)
    assert first == second


def test_split_for_no_overlap_across_many_ids():
    pairs = [("pexels", str(i)) for i in range(500)]
    splits = {pair: split_for(pair[0], pair[1], 0.8) for pair in pairs}
    # Re-running never flips a pair between train and val.
    for pair, split in splits.items():
        assert split_for(pair[0], pair[1], 0.8) == split
        assert split in ("train", "val")


def test_split_for_respects_ratio_roughly():
    labels = [split_for("pexels", str(i), 0.8) for i in range(2000)]
    train_share = labels.count("train") / len(labels)
    assert 0.7 < train_share < 0.9


# --- coco.build_coco ---------------------------------------------------------


def test_build_coco_valid_via_pycocotools(tmp_path: Path):
    images = [
        {"file_name": "a.jpg", "width": 640, "height": 480, "split": "train"},
        {"file_name": "b.jpg", "width": 320, "height": 240, "split": "val"},
    ]
    annotations = [
        {"image_index": 0, "category_name": "cat", "bbox": [10, 20, 100, 50]},
        {"image_index": 1, "category_name": "dog", "bbox": [0, 0, 50, 50]},
    ]
    categories = [{"name": "cat"}, {"name": "dog"}]

    coco_dict = build_coco(images, annotations, categories)

    out_path = tmp_path / "annotations.json"
    out_path.write_text(json.dumps(coco_dict))

    coco = COCO(str(out_path))
    assert len(coco.getImgIds()) == 2
    assert len(coco.getAnnIds()) == 2
    assert len(coco.getCatIds()) == 2


def test_build_coco_assigns_sequential_ids_and_area():
    images = [{"file_name": "a.jpg", "width": 10, "height": 10, "split": "train"}]
    annotations = [{"image_index": 0, "category_name": "cat", "bbox": [1, 2, 3, 4]}]
    categories = [{"name": "cat"}]

    coco_dict = build_coco(images, annotations, categories)

    assert coco_dict["images"][0]["id"] == 1
    assert coco_dict["annotations"][0]["id"] == 1
    assert coco_dict["annotations"][0]["image_id"] == 1
    assert coco_dict["annotations"][0]["category_id"] == 1
    assert coco_dict["annotations"][0]["bbox"] == [1, 2, 3, 4]
    assert coco_dict["annotations"][0]["area"] == 12
    assert coco_dict["annotations"][0]["iscrowd"] == 0
    assert coco_dict["categories"][0]["id"] == 1


# --- webdataset.write_shards ------------------------------------------------


def _fake_video(tmp_path: Path, name: str, size_bytes: int) -> Path:
    path = tmp_path / name
    path.write_bytes(b"\0" * size_bytes)
    return path


def test_write_shards_single_shard(tmp_path: Path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    items = [
        (f"pexels_{i}", _fake_video(videos_dir, f"{i}.mp4", 1000), {"id": i})
        for i in range(3)
    ]
    out_dir = tmp_path / "shards"

    shards = write_shards(items, out_dir, max_shard_bytes=1_000_000)

    assert len(shards) == 1
    with tarfile.open(shards[0]) as tar:
        names = tar.getnames()
    assert sum(1 for n in names if n.endswith(".mp4")) == 3
    assert sum(1 for n in names if n.endswith(".json")) == 3


def test_write_shards_splits_when_exceeding_max_bytes(tmp_path: Path):
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    items = [
        (f"pexels_{i}", _fake_video(videos_dir, f"{i}.mp4", 800), {"id": i})
        for i in range(3)
    ]
    out_dir = tmp_path / "shards"

    shards = write_shards(items, out_dir, max_shard_bytes=1000)

    assert len(shards) >= 2
    total_mp4 = 0
    for shard in shards:
        with tarfile.open(shard) as tar:
            total_mp4 += sum(1 for n in tar.getnames() if n.endswith(".mp4"))
    assert total_mp4 == 3


# --- hf ----------------------------------------------------------------------


def test_check_push_prerequisites_raises_without_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        check_push_prerequisites(push=True)


def test_check_push_prerequisites_ok_with_token(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HF_TOKEN", "x")
    check_push_prerequisites(push=True)  # must not raise


def test_check_push_prerequisites_ok_without_push(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    check_push_prerequisites(push=False)  # must not raise


def test_build_hf_dataset_missing_package_raises():
    with pytest.raises(RuntimeError, match="pip install datasets"):
        build_hf_dataset([{"a": 1}])


# --- stage B: DB-backed selection + per-format export ------------------------


def test_select_export_videos_excludes_unlicensed_by_default(tmp_cfg, db):
    _insert_video(source_id="known", license="cc0")
    _insert_video(source_id="unknown", license="unknown")

    with session_scope() as session:
        licensed_only = select_export_videos(session, include_unlicensed=False)
        with_unlicensed = select_export_videos(session, include_unlicensed=True)

    assert [v.source_id for v in licensed_only] == ["known"]
    assert {v.source_id for v in with_unlicensed} == {"known", "unknown"}


def test_select_export_videos_only_processed_status(tmp_cfg, db):
    _insert_video(source_id="downloaded", status=VideoStatus.DOWNLOADED)
    _insert_video(source_id="processed", status=VideoStatus.PROCESSED)

    with session_scope() as session:
        videos = select_export_videos(session, include_unlicensed=True)

    assert [v.source_id for v in videos] == ["processed"]


def test_run_export_empty_selection_raises_clear_message(tmp_cfg, db):
    with pytest.raises(ValueError, match="0 видео"):
        run_export(tmp_cfg, "coco", tmp_cfg.storage.video_path / "out")


def test_export_coco_from_db_valid_via_pycocotools(tmp_cfg, db):
    video_id = _insert_video(source_id="coco-1")
    frame_dir = frames_dir(tmp_cfg, video_id)
    frame_dir.mkdir(parents=True, exist_ok=True)
    (frame_dir / "frame_0001.jpg").write_bytes(b"x")
    _insert_detection(video_id, ts_s=0.0, class_name="cat")

    out_path = export_coco(tmp_cfg, tmp_cfg.storage.video_path.parent / "dataset")

    assert out_path.name == "annotations.json"
    coco = COCO(str(out_path))
    assert len(coco.getImgIds()) == 1
    assert len(coco.getAnnIds()) == 1


def test_export_coco_raises_when_only_unlicensed(tmp_cfg, db):
    _insert_video(source_id="unlicensed-1", license="unknown")

    with pytest.raises(ValueError, match="0 видео"):
        export_coco(tmp_cfg, tmp_cfg.storage.video_path.parent / "dataset")


def test_export_webdataset_from_db(tmp_cfg, db):
    video_file = tmp_cfg.storage.video_path / "pexels" / "wd-1.mp4"
    video_file.parent.mkdir(parents=True, exist_ok=True)
    video_file.write_bytes(b"video-bytes")
    _insert_video(source_id="wd-1", file_path=str(video_file))

    shards = export_webdataset(tmp_cfg, tmp_cfg.storage.video_path.parent / "shards")

    assert len(shards) == 1
    with tarfile.open(shards[0]) as tar:
        names = tar.getnames()
    assert "pexels_wd-1.mp4" in names
    assert "pexels_wd-1.json" in names


def test_export_webdataset_raises_when_no_files_on_disk(tmp_cfg, db):
    _insert_video(source_id="wd-missing", file_path=str(tmp_cfg.storage.video_path / "gone.mp4"))

    with pytest.raises(ValueError, match="нет файла на диске"):
        export_webdataset(tmp_cfg, tmp_cfg.storage.video_path.parent / "shards")


def test_export_hf_push_without_token_fails_before_db_work(tmp_cfg, db, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        export_hf(tmp_cfg, tmp_cfg.storage.video_path.parent / "hf", push=True)


def test_export_hf_missing_datasets_package_raises(tmp_cfg, db):
    _insert_video(source_id="hf-1")

    with pytest.raises(RuntimeError, match="pip install datasets"):
        export_hf(tmp_cfg, tmp_cfg.storage.video_path.parent / "hf", push=False)


def test_run_export_unknown_format_raises(tmp_cfg, db):
    _insert_video(source_id="fmt-1")

    with pytest.raises(ValueError, match="unknown export format"):
        run_export(tmp_cfg, "bogus", tmp_cfg.storage.video_path.parent / "out")
