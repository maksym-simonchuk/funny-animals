"""FastAPI app: read-only browsing of the video catalog + DMCA delete.

When `browser_mode.enabled` is set it also exposes the local ingest queue that the
browser extension posts collected reel URLs into.
"""
from __future__ import annotations

import re
import secrets
import shutil
from datetime import datetime
from pathlib import Path
from typing import Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Response
from fastapi.responses import FileResponse
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from src.config import Config
from src.stats import collect_stats
from src.storage.db import session_scope
from src.storage.files import frames_dir, is_within
from src.storage.models import BROWSER_SOURCE, Video, VideoStatus

_POST_RE = re.compile(
    r"^https?://(?:www\.)?instagram\.com/(reel|p)/([A-Za-z0-9_-]{5,})/?(?:\?.*)?$"
)


class VideoOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    source: str
    source_id: str
    # Both are nullable in the DB: a queued row has no download URL until yt-dlp runs.
    page_url: str | None
    download_url: str | None
    author: str | None
    author_url: str | None
    license: str
    license_url: str | None
    title: str | None
    tags: list
    duration_s: float | None
    width: int | None
    height: int | None
    fps: float | None
    size_bytes: int | None
    codec: str | None
    file_path: str | None
    thumb_path: str | None
    gif_path: str | None
    sha256: str | None
    phash: str | None
    sharpness: float | None
    brightness: float | None
    has_animal: bool | None
    category: str | None
    detect_conf: float | None
    status: str
    reject_reason: str | None
    dataset_version: str | None
    created_at: datetime
    updated_at: datetime


class VideoListOut(BaseModel):
    items: list[VideoOut]
    page: int
    per_page: int
    total: int


class UrlBatch(BaseModel):
    urls: list[str] = Field(min_length=1, max_length=100)


class QueueAddResult(BaseModel):
    accepted: int
    duplicates: int
    queue_size: int


def _canonical(url: str) -> tuple[str, str] | None:
    """Return (shortcode, canonical URL), or None if this is not an Instagram post.

    Reels (/reel/) and ordinary video posts (/p/) share one shortcode space, so the
    shortcode alone dedups them while the URL keeps the form yt-dlp was given.
    """
    match = _POST_RE.match(url.strip())
    if match is None:
        return None
    kind, code = match.group(1), match.group(2)
    return code, f"https://www.instagram.com/{kind}/{code}/"


def _queue_size(session: Session) -> int:
    return session.scalar(
        select(func.count())
        .select_from(Video)
        .where(Video.source == BROWSER_SOURCE, Video.status == VideoStatus.DISCOVERED)
    ) or 0


def create_app(cfg: Config) -> FastAPI:
    """Build the read-only catalog API. Assumes init_db() has already run (see app.py)."""
    app = FastAPI(title=cfg.app.name)

    def get_session() -> Iterator[Session]:
        with session_scope() as session:
            yield session

    @app.get("/stats")
    def get_stats() -> dict:
        return collect_stats(by_category=True, by_source=True)

    @app.get("/videos", response_model=VideoListOut)
    def list_videos(
        source: str | None = None,
        category: str | None = None,
        min_duration: float | None = None,
        max_duration: float | None = None,
        has_animal: bool | None = None,
        license: str | None = None,
        status: str | None = None,
        page: int = Query(1, ge=1),
        per_page: int = Query(50, ge=1, le=200),
        session: Session = Depends(get_session),
    ) -> VideoListOut:
        stmt = select(Video)
        if source is not None:
            stmt = stmt.where(Video.source == source)
        if category is not None:
            stmt = stmt.where(Video.category == category)
        if min_duration is not None:
            stmt = stmt.where(Video.duration_s >= min_duration)
        if max_duration is not None:
            stmt = stmt.where(Video.duration_s <= max_duration)
        if has_animal is not None:
            stmt = stmt.where(Video.has_animal == has_animal)
        if license is not None:
            stmt = stmt.where(Video.license == license)
        if status is not None:
            stmt = stmt.where(Video.status == status)

        total = session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = session.execute(
            stmt.order_by(Video.id).offset((page - 1) * per_page).limit(per_page)
        ).scalars().all()

        return VideoListOut(
            items=[VideoOut.model_validate(row) for row in rows],
            page=page,
            per_page=per_page,
            total=total,
        )

    @app.get("/videos/{video_id}", response_model=VideoOut)
    def get_video(video_id: int, session: Session = Depends(get_session)) -> VideoOut:
        video = session.get(Video, video_id)
        if video is None or video.status == VideoStatus.DELETED:
            raise HTTPException(status_code=404, detail="video not found")
        return VideoOut.model_validate(video)

    @app.get("/videos/{video_id}/file")
    def get_video_file(video_id: int, session: Session = Depends(get_session)) -> FileResponse:
        video = session.get(Video, video_id)
        if video is None or video.status == VideoStatus.DELETED or not video.file_path:
            raise HTTPException(status_code=404, detail="file not found")

        file_path = Path(video.file_path)
        if not is_within(file_path, cfg.storage.video_path):
            raise HTTPException(status_code=404, detail="file not found")
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="file not found")

        return FileResponse(file_path)

    @app.delete("/videos/{video_id}", status_code=204)
    def delete_video(video_id: int, session: Session = Depends(get_session)) -> Response:
        video = session.get(Video, video_id)
        if video is None:
            raise HTTPException(status_code=404, detail="video not found")

        for path_str in (video.file_path, video.thumb_path, video.gif_path):
            if path_str:
                Path(path_str).unlink(missing_ok=True)
        frame_dir = frames_dir(cfg, video.id)
        if frame_dir.exists():
            shutil.rmtree(frame_dir, ignore_errors=True)

        video.status = VideoStatus.DELETED
        video.file_path = None
        video.thumb_path = None
        video.gif_path = None

        return Response(status_code=204)

    if cfg.browser_mode.enabled:
        _mount_queue(app, cfg, get_session)

    return app


def _mount_queue(app: FastAPI, cfg: Config, get_session) -> None:
    """Ingest routes for the browser extension. Only mounted when browser_mode is on."""
    # No CORS headers on purpose: the extension reaches 127.0.0.1 through its own
    # host_permissions, so any origin a browser page could send from stays blocked.
    browser = cfg.browser_mode

    def require_ingest_token(x_ingest_token: str = Header(default="")) -> None:
        if not browser.ingest_token:
            variable = browser.ingest_token_env or "BROWSER_INGEST_TOKEN"
            logger.error(
                f"browser_mode is enabled but no ingest token is set. Generate one with "
                f"`python -c \"import secrets;print(secrets.token_urlsafe(32))\"` and put it "
                f"in .env as {variable}=<token>."
            )
            raise HTTPException(status_code=503, detail="ingest token is not configured")
        if not secrets.compare_digest(x_ingest_token, browser.ingest_token):
            raise HTTPException(status_code=401, detail="invalid ingest token")

    @app.post(
        "/queue/urls",
        response_model=QueueAddResult,
        dependencies=[Depends(require_ingest_token)],
    )
    def add_urls(batch: UrlBatch, session: Session = Depends(get_session)) -> QueueAddResult:
        shortcodes: dict[str, str] = {}
        for url in batch.urls:
            post = _canonical(url)
            if post is None:
                raise HTTPException(status_code=422, detail=f"not an Instagram post URL: {url}")
            code, canonical_url = post
            shortcodes[code] = canonical_url

        size = _queue_size(session)
        if size >= browser.max_queue_size:
            logger.warning(f"queue is full ({size}/{browser.max_queue_size}), rejecting batch")
            return QueueAddResult(accepted=0, duplicates=0, queue_size=size)

        known = set(
            session.execute(
                select(Video.source_id).where(
                    Video.source == BROWSER_SOURCE, Video.source_id.in_(list(shortcodes))
                )
            ).scalars().all()
        )

        accepted = 0
        duplicates = len(known)
        for code, url in shortcodes.items():
            if code in known:
                continue
            try:
                with session.begin_nested():
                    session.add(
                        Video(
                            source=BROWSER_SOURCE,
                            source_id=code,
                            page_url=url,
                            # yt-dlp fills author/title/duration at download time.
                            license="unknown",
                            status=VideoStatus.DISCOVERED,
                        )
                    )
                accepted += 1
            except IntegrityError:
                # Another batch inserted the same shortcode concurrently.
                duplicates += 1

        session.flush()
        return QueueAddResult(
            accepted=accepted, duplicates=duplicates, queue_size=_queue_size(session)
        )

    @app.get("/queue", response_model=VideoListOut)
    def list_queue(
        page: int = Query(1, ge=1),
        per_page: int = Query(50, ge=1, le=200),
        session: Session = Depends(get_session),
    ) -> VideoListOut:
        stmt = select(Video).where(
            Video.source == BROWSER_SOURCE, Video.status == VideoStatus.DISCOVERED
        )
        total = session.scalar(select(func.count()).select_from(stmt.subquery())) or 0
        rows = session.execute(
            stmt.order_by(Video.id).offset((page - 1) * per_page).limit(per_page)
        ).scalars().all()
        return VideoListOut(
            items=[VideoOut.model_validate(row) for row in rows],
            page=page,
            per_page=per_page,
            total=total,
        )
