"""Logging setup (loguru): stderr plus a rotating file."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

_FORMAT = "<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}"


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    """Configure the global logger. Safe to call more than once."""
    logger.remove()
    logger.add(sys.stderr, level=level.upper(), format=_FORMAT, colorize=True)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_dir / "app.log",
            level=level.upper(),
            rotation="10 MB",
            retention=5,
            encoding="utf-8",
        )
