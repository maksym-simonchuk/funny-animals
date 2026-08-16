"""Shared pytest fixtures: tmp config, empty DB, and ffmpeg-generated test videos."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
import pytest

from src.config import (
    AppCfg,
    Config,
    CollectorCfg,
    CompilerCfg,
    ExportCfg,
    ProcessingCfg,
    StorageCfg,
)
from src.storage.db import init_db


def _make_config(root: Path) -> Config:
    """Build a Config rooted entirely inside `root` (db file + video/thumbnail/frame dirs)."""
    return Config(
        app=AppCfg(name="test", log_level="INFO"),
        storage=StorageCfg(
            database=f"sqlite:///{root}/test.db",
            video_path=root / "videos",
            thumbnail_path=root / "thumbnails",
            frame_path=root / "frames",
            log_path=root / "logs",
            model_path=root / "models",
            concurrency=4,
        ),
        collectors={
            "pexels": CollectorCfg(name="pexels", enabled=False, api_key="test-key", rate_limit=200),
            "pixabay": CollectorCfg(name="pixabay", enabled=False, api_key="test-key", rate_limit=100),
            "ytdlp": CollectorCfg(
                name="ytdlp", enabled=False, options={"delay_min": 3, "delay_max": 5, "retries": 3}
            ),
        },
        processing=ProcessingCfg(),
        export=ExportCfg(),
        compiler=CompilerCfg(vision_model="", output_path=root / "shorts", segment_seconds=2.0, clips_per_short=3),
    )


@pytest.fixture
def tmp_cfg(tmp_path: Path) -> Config:
    """Config rooted entirely inside a fresh tmp_path."""
    return _make_config(tmp_path)


@pytest.fixture
def db(tmp_cfg: Config):
    """An initialized, empty database for tmp_cfg."""
    return init_db(tmp_cfg.storage.database)


def _generate_video(out: Path, duration: int) -> None:
    subprocess.run(
        [
            "ffmpeg", "-f", "lavfi", "-i", f"testsrc=duration={duration}:size=1280x720:rate=25",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-y", str(out),
        ],
        check=True,
        capture_output=True,
    )


def _video_fixture(tmp_path_factory: pytest.TempPathFactory, name: str, duration: int) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")
    out = tmp_path_factory.mktemp("videos") / f"{name}.mp4"
    _generate_video(out, duration)
    return out


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 6-second synthetic test video."""
    return _video_fixture(tmp_path_factory, "sample", 6)


@pytest.fixture(scope="session")
def short_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2-second synthetic test video (below min_duration)."""
    return _video_fixture(tmp_path_factory, "short", 2)


@pytest.fixture(scope="session")
def long_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 90-second synthetic test video (above max_duration)."""
    return _video_fixture(tmp_path_factory, "long", 90)

