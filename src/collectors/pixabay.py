"""Pixabay video search (https://pixabay.com/api/docs/#api_search_videos)."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, ClassVar

from src.collectors.base import BaseCollector, VideoCandidate, register

_SEARCH_URL = "https://pixabay.com/api/videos/"
_PER_PAGE = 200  # API maximum
_LICENSE = "Pixabay License"
_LICENSE_URL = "https://pixabay.com/service/license-summary/"
_RENDITIONS = ("large", "medium", "small", "tiny")


def _best_rendition(videos: dict[str, Any]) -> dict[str, Any] | None:
    """Highest-resolution rendition with a usable URL, or None if the item has none."""
    candidates = [videos[key] for key in _RENDITIONS if videos.get(key, {}).get("url")]
    if not candidates:
        return None
    return max(candidates, key=lambda v: (v.get("width") or 0) * (v.get("height") or 0))


@register
class PixabayCollector(BaseCollector):
    name: ClassVar[str] = "pixabay"

    async def search(self, query: str, limit: int) -> AsyncIterator[VideoCandidate]:
        yielded = 0
        page = 1
        while yielded < limit:
            per_page = min(_PER_PAGE, limit - yielded)
            response = await self._request(
                _SEARCH_URL,
                params={
                    "key": self.settings.api_key,
                    "q": query,
                    "per_page": per_page,
                    "page": page,
                },
            )
            async with response:
                payload = await response.json()

            hits = payload.get("hits") or []
            if not hits:
                return

            for item in hits:
                best = _best_rendition(item.get("videos") or {})
                if best is None:
                    continue

                user = item.get("user")
                user_id = item.get("user_id")
                tags = [t.strip() for t in (item.get("tags") or "").split(",") if t.strip()]
                yield VideoCandidate(
                    source=self.name,
                    source_id=str(item["id"]),
                    page_url=item.get("pageURL", ""),
                    download_url=best["url"],
                    author=user,
                    author_url=f"https://pixabay.com/users/{user}-{user_id}/" if user else None,
                    license=_LICENSE,
                    license_url=_LICENSE_URL,
                    title=item.get("tags") or query,
                    tags=tags,
                    duration_s=float(item["duration"]) if item.get("duration") else None,
                    width=best.get("width"),
                    height=best.get("height"),
                )
                yielded += 1
                if yielded >= limit:
                    return

            if len(hits) < per_page:
                return
            page += 1
