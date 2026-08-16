"""Collector registry. Dropping a `@register`-decorated module in here is enough."""
from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

import aiohttp
from loguru import logger

from src.collectors.base import COLLECTORS, BaseCollector, RunStats, TokenBucket, VideoCandidate, register

if TYPE_CHECKING:
    from src.config import Config

__all__ = [
    "COLLECTORS",
    "BaseCollector",
    "RunStats",
    "TokenBucket",
    "VideoCandidate",
    "get_enabled",
    "register",
    "run_collect",
]

_USER_AGENT = "funny-animals-enricher/1.0 (personal research dataset)"
_TIMEOUT = aiohttp.ClientTimeout(total=300, connect=30)
_discovered = False


def _autodiscover() -> None:
    """Import every sibling module once so `@register` decorators run."""
    global _discovered
    if _discovered:
        return
    for module in pkgutil.iter_modules(__path__):
        if not module.name.startswith("_") and module.name != "base":
            importlib.import_module(f"{__name__}.{module.name}")
    _discovered = True


def get_enabled(cfg: "Config") -> list[type[BaseCollector]]:
    """Registered collectors that are both enabled in config and actually importable."""
    _autodiscover()
    return [
        cls
        for name, cls in sorted(COLLECTORS.items())
        if name in cfg.collectors and cfg.collectors[name].enabled
    ]


async def run_collect(cfg: "Config", source: str, query: str, limit: int) -> RunStats:
    """Run one collector, or every enabled one when `source == "all"`."""
    _autodiscover()

    if source == "all":
        classes = get_enabled(cfg)
        if not classes:
            raise ValueError("no collectors are enabled in config.yaml")
    else:
        if source not in COLLECTORS:
            known = ", ".join(sorted(COLLECTORS)) or "none"
            raise ValueError(f"unknown source '{source}' (known: {known})")
        if source not in cfg.collectors or not cfg.collectors[source].enabled:
            raise ValueError(f"collector '{source}' is disabled in config.yaml")
        classes = [COLLECTORS[source]]

    total = RunStats()
    # One session shared by every collector: connection reuse, one place for the UA.
    async with aiohttp.ClientSession(
        timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}
    ) as http:
        for cls in classes:
            logger.info(f"collecting from {cls.name}: query={query!r} limit={limit}")
            total.merge(await cls(cfg, http).collect(query, limit))
    return total
