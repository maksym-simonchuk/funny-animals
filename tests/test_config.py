"""Tests for src.config: env-var expansion and validation."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import Config, ConfigError, load_config

MINIMAL_YAML = """
app:
  name: test-app
  log_level: DEBUG

storage:
  database: sqlite:///data/test.db
  concurrency: 4

collectors:
  pexels:
    enabled: true
    api_key: "${PEXELS_API_KEY}"
    rate_limit: 200
  ytdlp:
    enabled: true
    delay_min: 3
    delay_max: 5
    retries: 3

processing:
  video:
    min_duration: 5
    max_duration: 60
  animal_detection:
    classes: [cat, dog]
  quality:
    min_resolution: 240

export:
  train_test_split: 0.8

unknown_section:
  foo: bar
"""


def _write_yaml(tmp_path: Path, text: str = MINIMAL_YAML) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text)
    return path


def test_env_var_expands(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "secret-123")
    cfg = load_config(_write_yaml(tmp_path))
    assert cfg.collectors["pexels"].api_key == "secret-123"


def test_missing_key_for_enabled_collector_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="PEXELS_API_KEY"):
        load_config(_write_yaml(tmp_path))


def test_ytdlp_needs_no_key(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "secret-123")
    cfg = load_config(_write_yaml(tmp_path))
    assert cfg.collectors["ytdlp"].enabled is True
    assert cfg.collectors["ytdlp"].options["delay_min"] == 3


def test_unknown_top_level_section_ignored(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "secret-123")
    cfg = load_config(_write_yaml(tmp_path))
    assert isinstance(cfg, Config)


def test_real_config_yaml_loads(monkeypatch):
    # Shipped defaults must load with no API keys at all, or a fresh clone can't run.
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("PIXABAY_API_KEY", raising=False)
    cfg = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    assert cfg.collectors["pexels"].enabled is False
    assert cfg.collectors["ytdlp"].enabled is True
    assert cfg.storage.concurrency == 4
    assert cfg.processing.video.min_duration == 5
    assert cfg.processing.video.max_duration == 60
