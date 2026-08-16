"""Configuration loading: YAML + ${ENV} expansion into a frozen dataclass tree."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_ENV_REF = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class ConfigError(Exception):
    """Raised when the configuration is missing or inconsistent."""


@dataclass(frozen=True)
class AppCfg:
    name: str = "Animal Video Dataset Builder"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"


@dataclass(frozen=True)
class StorageCfg:
    database: str = "sqlite:///data/animal_videos.db"
    video_path: Path = Path("data/videos")
    thumbnail_path: Path = Path("data/thumbnails")
    frame_path: Path = Path("data/frames")
    log_path: Path = Path("data/logs")
    model_path: Path = Path("data/models")
    max_file_size: int = 100  # MB
    concurrency: int = 4


@dataclass(frozen=True)
class CollectorCfg:
    name: str
    enabled: bool = False
    api_key: str = ""
    api_key_env: str | None = None
    rate_limit: int = 100  # requests/hour
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VideoCfg:
    target_format: str = "mp4"
    target_codec: str = "h264"
    target_resolution: int = 720
    extract_frames: bool = True
    frame_interval: int = 5
    min_duration: float = 5.0
    max_duration: float = 60.0


@dataclass(frozen=True)
class DetectionCfg:
    model: str = "yolov8n.pt"
    confidence_threshold: float = 0.7
    classes: tuple[str, ...] = ("cat", "dog", "bird", "horse", "sheep", "cow", "bear")


@dataclass(frozen=True)
class QualityCfg:
    min_resolution: int = 720
    min_brightness: float = 50.0
    min_sharpness: float = 100.0


@dataclass(frozen=True)
class ProcessingCfg:
    video: VideoCfg = field(default_factory=VideoCfg)
    animal_detection: DetectionCfg = field(default_factory=DetectionCfg)
    quality: QualityCfg = field(default_factory=QualityCfg)


@dataclass(frozen=True)
class ExportCfg:
    formats: tuple[str, ...] = ("coco", "webdataset", "huggingface")
    include_metadata: bool = True
    train_test_split: float = 0.8
    shard_bytes: int = 1_000_000_000


@dataclass(frozen=True)
class BrowserModeCfg:
    """Server side of the browser URL collector. Scroll pacing lives in the extension."""

    enabled: bool = False
    ingest_token: str = ""
    ingest_token_env: str | None = None
    max_queue_size: int = 1000
    poll_interval: float = 10.0


@dataclass(frozen=True)
class CompilerCfg:
    """Shorts assembly. The model runs locally via Ollama -- nothing leaves the machine."""

    model: str = "qwen3:8b"
    # vision pass: "" turns it off and the captions fall back to metadata
    vision_model: str = "qwen2.5vl:7b"
    host: str = "http://127.0.0.1:11434"
    temperature: float = 0.8
    timeout: float = 120.0
    clips_per_short: int = 5
    segment_seconds: float = 8.0
    transition_seconds: float = 0.35
    # off by default: reels already carry their own track, so a bed doubles it into mush
    music_volume: float = 0.0
    output_path: Path = Path("data/shorts")
    font: str = "/System/Library/Fonts/Supplemental/Impact.ttf"
    title_size: int = 92
    caption_size: int = 72
    ranking_size: int = 46


@dataclass(frozen=True)
class Config:
    app: AppCfg
    storage: StorageCfg
    collectors: dict[str, CollectorCfg]
    processing: ProcessingCfg
    export: ExportCfg
    browser_mode: BrowserModeCfg = field(default_factory=BrowserModeCfg)
    compiler: CompilerCfg = field(default_factory=CompilerCfg)


def _resolution(value: Any, default: int) -> int:
    """Accept 720, "720" or "720p" and return an int height."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    return int(str(value).lower().rstrip("p"))


def _expand(value: Any) -> Any:
    """Replace a bare ``${VAR}`` scalar with its environment value ("" if unset)."""
    if isinstance(value, str):
        match = _ENV_REF.match(value.strip())
        if match:
            return os.environ.get(match.group(1), "")
        return value
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def _env_ref(value: Any) -> str | None:
    """Return the variable name if ``value`` is a bare ``${VAR}`` reference."""
    if isinstance(value, str):
        match = _ENV_REF.match(value.strip())
        if match:
            return match.group(1)
    return None


def _parse_collectors(raw: dict[str, Any]) -> dict[str, CollectorCfg]:
    collectors: dict[str, CollectorCfg] = {}
    for name, node in (raw or {}).items():
        node = node or {}
        known = {"enabled", "api_key", "rate_limit"}
        collectors[name] = CollectorCfg(
            name=name,
            enabled=bool(node.get("enabled", False)),
            # api_key_env is captured before expansion so errors can name the variable.
            api_key_env=_env_ref(node.get("api_key")),
            api_key=str(_expand(node.get("api_key", "")) or ""),
            rate_limit=int(node.get("rate_limit", 100)),
            options={k: _expand(v) for k, v in node.items() if k not in known},
        )
    return collectors


def _parse_browser_mode(raw: dict[str, Any]) -> BrowserModeCfg:
    raw = raw or {}
    defaults = BrowserModeCfg()
    return BrowserModeCfg(
        enabled=bool(raw.get("enabled", False)),
        # Captured before expansion so a missing token can name the variable to set.
        ingest_token_env=_env_ref(raw.get("ingest_token")),
        ingest_token=str(_expand(raw.get("ingest_token", "")) or ""),
        max_queue_size=int(raw.get("max_queue_size", defaults.max_queue_size)),
        poll_interval=float(raw.get("poll_interval", defaults.poll_interval)),
    )


def _parse_compiler(raw: dict[str, Any]) -> CompilerCfg:
    defaults = CompilerCfg()
    return CompilerCfg(
        model=str(raw.get("model", defaults.model)),
        vision_model=str(raw.get("vision_model", defaults.vision_model)),
        host=str(raw.get("host", defaults.host)),
        temperature=float(raw.get("temperature", defaults.temperature)),
        timeout=float(raw.get("timeout", defaults.timeout)),
        clips_per_short=int(raw.get("clips_per_short", defaults.clips_per_short)),
        segment_seconds=float(raw.get("segment_seconds", defaults.segment_seconds)),
        transition_seconds=float(raw.get("transition_seconds", defaults.transition_seconds)),
        music_volume=float(raw.get("music_volume", defaults.music_volume)),
        output_path=Path(raw.get("output_path", defaults.output_path)),
        font=str(raw.get("font", defaults.font)),
        title_size=int(raw.get("title_size", defaults.title_size)),
        caption_size=int(raw.get("caption_size", defaults.caption_size)),
        ranking_size=int(raw.get("ranking_size", defaults.ranking_size)),
    )


def _validate(cfg: Config) -> None:
    """Fail fast on enabled collectors whose API key is declared but unset."""
    for collector in cfg.collectors.values():
        if collector.enabled and collector.api_key_env and not collector.api_key:
            raise ConfigError(
                f"collector '{collector.name}' is enabled but environment variable "
                f"{collector.api_key_env} is not set. Add it to .env or export it, "
                f"or set collectors.{collector.name}.enabled: false in config.yaml."
            )
    if not 0.0 < cfg.export.train_test_split < 1.0:
        raise ConfigError(
            f"export.train_test_split must be between 0 and 1, got {cfg.export.train_test_split}"
        )
    if cfg.storage.concurrency < 1:
        raise ConfigError("storage.concurrency must be >= 1")


def load_config(path: str | Path = "config.yaml") -> Config:
    """Load ``config.yaml``, expand ``${ENV}`` references and validate the result."""
    load_dotenv(override=False)

    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path.resolve()}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} must contain a YAML mapping at the top level")

    app_raw = _expand(raw.get("app") or {})
    storage_raw = _expand(raw.get("storage") or {})
    processing_raw = _expand(raw.get("processing") or {})
    export_raw = _expand(raw.get("export") or {})
    video_raw = processing_raw.get("video") or {}
    detect_raw = processing_raw.get("animal_detection") or {}
    quality_raw = processing_raw.get("quality") or {}

    defaults = StorageCfg()
    storage = StorageCfg(
        # DATABASE_URL wins so docker-compose can point the same config at postgres.
        database=os.environ.get("DATABASE_URL") or storage_raw.get("database") or defaults.database,
        video_path=Path(storage_raw.get("video_path", defaults.video_path)),
        thumbnail_path=Path(storage_raw.get("thumbnail_path", defaults.thumbnail_path)),
        frame_path=Path(storage_raw.get("frame_path", defaults.frame_path)),
        log_path=Path(storage_raw.get("log_path", defaults.log_path)),
        model_path=Path(storage_raw.get("model_path", defaults.model_path)),
        max_file_size=int(storage_raw.get("max_file_size", defaults.max_file_size)),
        concurrency=int(storage_raw.get("concurrency", defaults.concurrency)),
    )

    video_defaults = VideoCfg()
    detect_defaults = DetectionCfg()
    quality_defaults = QualityCfg()
    export_defaults = ExportCfg()

    cfg = Config(
        app=AppCfg(
            name=app_raw.get("name", AppCfg().name),
            version=str(app_raw.get("version", AppCfg().version)),
            debug=bool(app_raw.get("debug", False)),
            log_level=str(app_raw.get("log_level", "INFO")).upper(),
        ),
        storage=storage,
        collectors=_parse_collectors(raw.get("collectors") or {}),
        processing=ProcessingCfg(
            video=VideoCfg(
                target_format=video_raw.get("target_format", video_defaults.target_format),
                target_codec=video_raw.get("target_codec", video_defaults.target_codec),
                target_resolution=_resolution(
                    video_raw.get("target_resolution"), video_defaults.target_resolution
                ),
                extract_frames=bool(video_raw.get("extract_frames", True)),
                frame_interval=int(video_raw.get("frame_interval", video_defaults.frame_interval)),
                min_duration=float(video_raw.get("min_duration", video_defaults.min_duration)),
                max_duration=float(video_raw.get("max_duration", video_defaults.max_duration)),
            ),
            animal_detection=DetectionCfg(
                model=detect_raw.get("model", detect_defaults.model),
                confidence_threshold=float(
                    detect_raw.get("confidence_threshold", detect_defaults.confidence_threshold)
                ),
                classes=tuple(detect_raw.get("classes") or detect_defaults.classes),
            ),
            quality=QualityCfg(
                min_resolution=_resolution(
                    quality_raw.get("min_resolution"), quality_defaults.min_resolution
                ),
                min_brightness=float(
                    quality_raw.get("min_brightness", quality_defaults.min_brightness)
                ),
                min_sharpness=float(
                    quality_raw.get("min_sharpness", quality_defaults.min_sharpness)
                ),
            ),
        ),
        export=ExportCfg(
            formats=tuple(export_raw.get("formats") or export_defaults.formats),
            include_metadata=bool(export_raw.get("include_metadata", True)),
            train_test_split=float(
                export_raw.get("train_test_split", export_defaults.train_test_split)
            ),
            shard_bytes=int(export_raw.get("shard_bytes", export_defaults.shard_bytes)),
        ),
        browser_mode=_parse_browser_mode(raw.get("browser_mode") or {}),
        compiler=_parse_compiler(_expand(raw.get("compiler") or {})),
    )
    _validate(cfg)
    return cfg
