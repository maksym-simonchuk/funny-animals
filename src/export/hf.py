"""HuggingFace `datasets` export. `datasets` is an optional extra, imported lazily."""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from src.export import select_export_videos, split_for
from src.storage.db import session_scope
from src.storage.models import Video

if TYPE_CHECKING:
    from src.config import Config


def check_push_prerequisites(push: bool) -> None:
    """Fail fast, before any work starts, if --push was requested without HF_TOKEN set."""
    if push and not os.environ.get("HF_TOKEN"):
        raise RuntimeError(
            "export --format huggingface --push requires the HF_TOKEN environment variable"
        )


def build_hf_dataset(records: Iterable[dict[str, Any]]) -> Any:
    """Build a `datasets.Dataset` from prepared record dicts via `Dataset.from_generator`."""
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError(
            "huggingface export requires the optional `datasets` package: pip install datasets"
        ) from exc

    records = list(records)

    def _generator():
        yield from records

    return datasets.Dataset.from_generator(_generator)


def _record_for(cfg: "Config", video: Video) -> dict[str, Any]:
    return {
        "id": video.id,
        "source": video.source,
        "source_id": video.source_id,
        "file_path": video.file_path,
        "license": video.license,
        "license_url": video.license_url,
        "category": video.category,
        "duration_s": video.duration_s,
        "width": video.width,
        "height": video.height,
        "tags": video.tags,
        "split": split_for(video.source, video.source_id, cfg.export.train_test_split),
    }


def write_hf(cfg: "Config", out_dir: Path, videos: Sequence[Video], push: bool) -> str:
    """Build and save a `datasets.Dataset` from the given (already-filtered) videos.

    push=True pushes to the Hub under a repo named after out_dir; check_push_prerequisites
    must be called by the caller before any DB/dataset work starts.
    """
    records = [_record_for(cfg, video) for video in videos]
    dataset = build_hf_dataset(records)

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(out_dir))
    if push:
        dataset.push_to_hub(out_dir.name)
    return str(out_dir)


def export_hf(cfg: "Config", out_dir: Path, push: bool) -> str:
    """Export processed videos with a known license to a HuggingFace `datasets.Dataset`."""
    check_push_prerequisites(push)
    with session_scope() as session:
        videos = select_export_videos(session, include_unlicensed=False)
        if not videos:
            raise ValueError(
                "экспорт невозможен: 0 видео status=processed с известной лицензией "
                "(используйте run_export(..., include_unlicensed=True), если это ожидаемо)"
            )
        return write_hf(cfg, out_dir, videos, push)
