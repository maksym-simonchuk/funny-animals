"""Pexels video search (https://www.pexels.com/api/documentation/#videos-search)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

from loguru import logger

from src.collectors.base import BaseCollector, VideoCandidate, register

_SEARCH_URL = "https://api.pexels.com/videos/search"
_PER_PAGE = 80  # API maximum
_LICENSE = "Pexels License"
_LICENSE_URL = "https://www.pexels.com/license/"


def _best_file(video_files: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Highest-resolution mp4 rendition, or None when the item has no usable file."""
    mp4s = [f for f in video_files if f.get("file_type") == "video/mp4" and f.get("link")]
    if not mp4s:
        return None
    return max(mp4s, key=lambda f: (f.get("width") or 0) * (f.get("height") or 0))


@register
class PexelsCollector(BaseCollector):
    name: ClassVar[str] = "pexels"

    async def search(self, query: str, limit: int) -> AsyncIterator[VideoCandidate]:
        yielded = 0
        page = 1
        while yielded < limit:
            response = await self._request(
                _SEARCH_URL,
                headers={"Authorization": self.settings.api_key},
                params={"query": query, "per_page": min(_PER_PAGE, limit - yielded), "page": page},
            )
            async with response:
                payload = await response.json()

            videos = payload.get("videos") or []
            if not videos:
                return

            for item in videos:
                best = _best_file(item.get("video_files") or [])
                if best is None:
                    logger.debug(f"pexels: {item.get('id')} has no mp4 rendition, skipping")
                    continue

                user = item.get("user") or {}
                yield VideoCandidate(
                    source=self.name,
                    source_id=str(item["id"]),
                    page_url=item.get("url", ""),
                    download_url=best["link"],
                    author=user.get("name"),
                    author_url=user.get("url"),
                    license=_LICENSE,
                    license_url=_LICENSE_URL,
                    title=item.get("alt") or query,
                    tags=[query],
                    duration_s=float(item["duration"]) if item.get("duration") else None,
                    width=best.get("width") or item.get("width"),
                    height=best.get("height") or item.get("height"),
                )
                yielded += 1
                if yielded >= limit:
                    return

            if not payload.get("next_page"):
                return
            page += 1
