"""SQLAlchemy models for the video dataset."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# Videos whose URL the browser extension queued; yt-dlp downloads them later.
BROWSER_SOURCE = "browser"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class VideoStatus(StrEnum):
    DISCOVERED = "discovered"
    DOWNLOADED = "downloaded"
    PROCESSED = "processed"
    REJECTED = "rejected"
    DELETED = "deleted"


class Video(Base):
    __tablename__ = "videos"
    __table_args__ = (
        UniqueConstraint("source", "source_id", name="uq_videos_source_source_id"),
        Index("ix_videos_phash", "phash"),
        Index("ix_videos_status", "status"),
        Index("ix_videos_category", "category"),
        Index("ix_videos_animal_duration", "has_animal", "duration_s"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # provenance
    source: Mapped[str] = mapped_column(String(32), nullable=False)
    source_id: Mapped[str] = mapped_column(String(128), nullable=False)
    page_url: Mapped[str | None] = mapped_column(Text)
    download_url: Mapped[str | None] = mapped_column(Text)

    # attribution / licensing
    author: Mapped[str | None] = mapped_column(String(256))
    author_url: Mapped[str | None] = mapped_column(Text)
    license: Mapped[str] = mapped_column(String(64), default="unknown", nullable=False)
    license_url: Mapped[str | None] = mapped_column(Text)

    # descriptive
    title: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)

    # technical
    duration_s: Mapped[float | None] = mapped_column(Float)
    width: Mapped[int | None] = mapped_column(Integer)
    height: Mapped[int | None] = mapped_column(Integer)
    fps: Mapped[float | None] = mapped_column(Float)
    size_bytes: Mapped[int | None] = mapped_column(Integer)
    codec: Mapped[str | None] = mapped_column(String(32))

    # files on disk
    file_path: Mapped[str | None] = mapped_column(Text)
    thumb_path: Mapped[str | None] = mapped_column(Text)
    gif_path: Mapped[str | None] = mapped_column(Text)

    # dedup
    sha256: Mapped[str | None] = mapped_column(String(64), unique=True)
    phash: Mapped[str | None] = mapped_column(String(128))

    # quality
    sharpness: Mapped[float | None] = mapped_column(Float)
    brightness: Mapped[float | None] = mapped_column(Float)

    # detection
    has_animal: Mapped[bool | None] = mapped_column(Boolean)
    category: Mapped[str | None] = mapped_column(String(64))
    detect_conf: Mapped[float | None] = mapped_column(Float)

    # lifecycle
    status: Mapped[str] = mapped_column(
        String(16), default=VideoStatus.DISCOVERED, nullable=False
    )
    reject_reason: Mapped[str | None] = mapped_column(String(64))
    dataset_version: Mapped[str | None] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    detections: Mapped[list["Detection"]] = relationship(
        back_populates="video", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Video {self.id} {self.source}:{self.source_id} {self.status}>"


class Detection(Base):
    """One animal detected on one keyframe."""

    __tablename__ = "detections"
    __table_args__ = (Index("ix_detections_video_class", "video_id", "class_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[int] = mapped_column(
        ForeignKey("videos.id", ondelete="CASCADE"), nullable=False
    )
    ts_s: Mapped[float] = mapped_column(Float, nullable=False)
    class_name: Mapped[str] = mapped_column(String(64), nullable=False)
    conf: Mapped[float] = mapped_column(Float, nullable=False)
    # bbox in pixels of the source frame
    x: Mapped[float] = mapped_column(Float, nullable=False)
    y: Mapped[float] = mapped_column(Float, nullable=False)
    w: Mapped[float] = mapped_column(Float, nullable=False)
    h: Mapped[float] = mapped_column(Float, nullable=False)

    video: Mapped[Video] = relationship(back_populates="detections")


class Run(Base):
    """One CLI invocation, kept for resume and for reporting."""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    command: Mapped[str] = mapped_column(String(32), nullable=False)
    source: Mapped[str | None] = mapped_column(String(32))
    query: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    found: Mapped[int] = mapped_column(Integer, default=0)
    downloaded: Mapped[int] = mapped_column(Integer, default=0)
    skipped: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[int] = mapped_column(Integer, default=0)
    cursor: Mapped[dict[str, Any] | None] = mapped_column(JSON)
