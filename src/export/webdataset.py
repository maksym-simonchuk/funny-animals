"""WebDataset export: pack (video, metadata) pairs into size-capped tar shards."""
from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from loguru import logger

from src.export import select_export_videos, split_for
from src.storage.db import session_scope
from src.storage.models import Video

if TYPE_CHECKING:
    from src.config import Config

DEFAULT_MAX_SHARD_BYTES = 1_000_000_000  # 1 GB


def write_shards(
    items: Iterable[tuple[str, Path, dict[str, Any]]],
    out_dir: Path,
    max_shard_bytes: int = DEFAULT_MAX_SHARD_BYTES,
) -> list[Path]:
    """Write (key, video_path, metadata) items as `{key}.mp4` + `{key}.json` pairs into tar
    shards under out_dir, starting a new shard whenever the current one would exceed
    max_shard_bytes. Returns the shard paths in write order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shards: list[Path] = []
    tar: tarfile.TarFile | None = None
    shard_path: Path | None = None

    for key, video_path, metadata in items:
        meta_bytes = json.dumps(metadata, ensure_ascii=False, indent=2).encode("utf-8")
        entry_size = video_path.stat().st_size + len(meta_bytes)

        if tar is not None and shard_path.stat().st_size + entry_size > max_shard_bytes:
            tar.close()
            tar = None

        if tar is None:
            shard_path = out_dir / f"shard-{len(shards):05d}.tar"
            tar = tarfile.open(shard_path, "w")
            shards.append(shard_path)

        tar.add(video_path, arcname=f"{key}.mp4")
        info = tarfile.TarInfo(name=f"{key}.json")
        info.size = len(meta_bytes)
        tar.addfile(info, io.BytesIO(meta_bytes))

    if tar is not None:
        tar.close()

    return shards


def _metadata_for(cfg: "Config", video: Video) -> dict[str, Any]:
    return {
        "id": video.id,
        "source": video.source,
        "source_id": video.source_id,
        "license": video.license,
        "license_url": video.license_url,
        "category": video.category,
        "duration_s": video.duration_s,
        "width": video.width,
        "height": video.height,
        "tags": video.tags,
        "split": split_for(video.source, video.source_id, cfg.export.train_test_split),
    }


def write_webdataset(cfg: "Config", out_dir: Path, videos: Sequence[Video]) -> list[Path]:
    """Pack the given (already-filtered) videos' files into tar shards."""
    items: list[tuple[str, Path, dict[str, Any]]] = []
    for video in videos:
        if not video.file_path or not Path(video.file_path).exists():
            logger.warning(f"webdataset export: skipping video id={video.id}, file missing on disk")
            continue
        key = f"{video.source}_{video.source_id}"
        items.append((key, Path(video.file_path), _metadata_for(cfg, video)))

    if not items:
        raise ValueError(
            f"экспорт невозможен: ни у одного из {len(videos)} видео нет файла на диске"
        )

    return write_shards(items, out_dir, max_shard_bytes=cfg.export.shard_bytes)


def export_webdataset(cfg: "Config", out_dir: Path) -> list[Path]:
    """Export processed videos with a known license to WebDataset tar shards."""
    with session_scope() as session:
        videos = select_export_videos(session, include_unlicensed=False)
        if not videos:
            raise ValueError(
                "экспорт невозможен: 0 видео status=processed с известной лицензией "
                "(используйте run_export(..., include_unlicensed=True), если это ожидаемо)"
            )
        return write_webdataset(cfg, out_dir, videos)
