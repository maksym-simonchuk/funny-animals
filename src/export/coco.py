"""COCO-format export: images = keyframes, annotations = detections, categories = config."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from src.export import select_export_videos, split_for
from src.storage.db import session_scope
from src.storage.files import frames_dir
from src.storage.models import Video

if TYPE_CHECKING:
    from src.config import Config


def build_coco(
    images: Sequence[dict[str, Any]],
    annotations: Sequence[dict[str, Any]],
    categories: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble a COCO dict from prepared records. IDs are assigned sequentially from 1.

    images: [{"file_name": str, "width": int, "height": int, "split": "train"|"val"}, ...]
    annotations: [{"image_index": int, "category_name": str, "bbox": [x, y, w, h]}, ...]
        image_index is the 0-based position of the owning record in `images`.
    categories: [{"name": str, "supercategory": str}, ...] — list order defines category id.
    """
    image_ids = {index: index + 1 for index in range(len(images))}
    category_ids = {category["name"]: index + 1 for index, category in enumerate(categories)}

    coco_images = [
        {
            "id": image_ids[index],
            "file_name": image["file_name"],
            "width": image["width"],
            "height": image["height"],
            "split": image["split"],
        }
        for index, image in enumerate(images)
    ]

    coco_annotations = []
    for ann_id, annotation in enumerate(annotations, start=1):
        x, y, w, h = annotation["bbox"]
        coco_annotations.append(
            {
                "id": ann_id,
                "image_id": image_ids[annotation["image_index"]],
                "category_id": category_ids[annotation["category_name"]],
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
            }
        )

    coco_categories = [
        {
            "id": index + 1,
            "name": category["name"],
            "supercategory": category.get("supercategory", "animal"),
        }
        for index, category in enumerate(categories)
    ]

    return {"images": coco_images, "annotations": coco_annotations, "categories": coco_categories}


def _prepare_records(cfg: "Config", videos: Sequence[Video]) -> tuple[list[dict], list[dict]]:
    """Turn Video rows (with .detections loaded) into build_coco's images/annotations inputs.

    One image per distinct detection timestamp; keyframe file is matched by position against
    the on-disk frame_*.jpg files for that video (falls back to the ffmpeg frame_%04d.jpg
    naming convention from processors/video.py if fewer frame files exist than timestamps).
    """
    images: list[dict] = []
    annotations: list[dict] = []

    for video in videos:
        frame_files = sorted(frames_dir(cfg, video.id).glob("frame_*.jpg"))
        by_ts: dict[float, list] = defaultdict(list)
        for detection in video.detections:
            by_ts[detection.ts_s].append(detection)
        split = split_for(video.source, video.source_id, cfg.export.train_test_split)

        for local_index, ts in enumerate(sorted(by_ts)):
            frame_path = (
                frame_files[local_index]
                if local_index < len(frame_files)
                else frames_dir(cfg, video.id) / f"frame_{local_index + 1:04d}.jpg"
            )
            images.append(
                {
                    "file_name": f"{video.id}/{frame_path.name}",
                    "width": video.width or 0,
                    "height": video.height or 0,
                    "split": split,
                }
            )
            image_index = len(images) - 1
            for detection in by_ts[ts]:
                annotations.append(
                    {
                        "image_index": image_index,
                        "category_name": detection.class_name,
                        "bbox": [detection.x, detection.y, detection.w, detection.h],
                    }
                )

    return images, annotations


def write_coco(cfg: "Config", out_dir: Path, videos: Sequence[Video]) -> Path:
    """Write out_dir/annotations.json for the given (already-filtered) videos."""
    images, annotations = _prepare_records(cfg, videos)
    categories = [{"name": name} for name in cfg.processing.animal_detection.classes]
    coco_dict = build_coco(images, annotations, categories)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "annotations.json"
    out_path.write_text(json.dumps(coco_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def export_coco(cfg: "Config", out_dir: Path) -> Path:
    """Export processed videos with a known license to COCO json. Excludes license=="unknown"."""
    with session_scope() as session:
        videos = select_export_videos(session, include_unlicensed=False)
        if not videos:
            raise ValueError(
                "экспорт невозможен: 0 видео status=processed с известной лицензией "
                "(используйте run_export(..., include_unlicensed=True), если это ожидаемо)"
            )
        return write_coco(cfg, out_dir, videos)
