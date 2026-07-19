"""SQLAlchemy models. Works against SQLite (default) or Supabase Postgres."""
from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Base(DeclarativeBase):
    pass


class Channel(Base):
    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    call_sign: Mapped[str] = mapped_column(String(40))
    network: Mapped[str] = mapped_column(String(40), default="")
    market: Mapped[str] = mapped_column(String(80), default="")
    region: Mapped[str] = mapped_column(String(80), default="")
    url: Mapped[str] = mapped_column(String(300))
    channel_id: Mapped[str | None] = mapped_column(String(40), unique=True, nullable=True)
    uploads_playlist_id: Mapped[str | None] = mapped_column(String(40), nullable=True)
    channel_title: Mapped[str] = mapped_column(String(200), default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    # Monitor state: newest upload publish time we've already processed.
    last_seen_published_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_checked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    candidates: Mapped[list["Candidate"]] = relationship(back_populates="channel")


# Review statuses for a candidate video.
STATUS_NEW = "new"
STATUS_APPROVED = "approved"       # operator approved; scrape pending/running
STATUS_ARCHIVED = "archived"       # downloaded + transcribed + stored
STATUS_REJECTED = "rejected"
STATUS_FAILED = "failed"           # scrape failed; operator can retry


class Candidate(Base):
    __tablename__ = "candidates"
    __table_args__ = (UniqueConstraint("video_id", name="uq_candidate_video"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    video_id: Mapped[str] = mapped_column(String(20))
    channel_pk: Mapped[int] = mapped_column(ForeignKey("channels.id"))
    title: Mapped[str] = mapped_column(Text)
    description: Mapped[str] = mapped_column(Text, default="")
    url: Mapped[str] = mapped_column(String(200))
    thumbnail_url: Mapped[str] = mapped_column(String(300), default="")
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)

    matched_keywords: Mapped[str] = mapped_column(Text, default="")  # comma-separated
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevance_rationale: Mapped[str] = mapped_column(Text, default="")
    climate_topic: Mapped[str] = mapped_column(String(60), default="")  # LLM-tagged theme

    status: Mapped[str] = mapped_column(String(20), default=STATUS_NEW)
    approved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    archived_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scrape_error: Mapped[str] = mapped_column(Text, default="")

    local_video_path: Mapped[str] = mapped_column(Text, default="")
    transcript_path: Mapped[str] = mapped_column(Text, default="")
    transcript_text: Mapped[str] = mapped_column(Text, default="")
    transcription_method: Mapped[str] = mapped_column(String(20), default="")  # captions | "" (none)

    # Optional LLM assists (clearly drafts).
    suggested_highlight: Mapped[str] = mapped_column(Text, default="")  # e.g. "00:42-01:10: ..."
    draft_caption: Mapped[str] = mapped_column(Text, default="")

    # Trim step: operator-chosen segments (JSON [{start, end}, ...]) and the
    # exported supercut file produced from them.
    trim_segments: Mapped[str] = mapped_column(Text, default="")
    trimmed_clip_path: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    channel: Mapped[Channel] = relationship(back_populates="candidates")
    threads_posts: Mapped[list["ThreadsPost"]] = relationship(back_populates="candidate")


class ThreadsPost(Base):
    __tablename__ = "threads_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_pk: Mapped[int | None] = mapped_column(ForeignKey("candidates.id"), nullable=True)
    threads_media_id: Mapped[str] = mapped_column(String(60), default="")
    permalink: Mapped[str] = mapped_column(String(300), default="")
    caption: Mapped[str] = mapped_column(Text, default="")
    clip_object_path: Mapped[str] = mapped_column(Text, default="")  # Supabase Storage object key
    clip_local_path: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft | published | failed
    source: Mapped[str] = mapped_column(String(20), default="app")  # app | threads (imported history)
    error: Mapped[str] = mapped_column(Text, default="")
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Structured attributes for analytics slicing.
    caption_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    caption_has_question: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    caption_has_cta: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    caption_hashtag_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    caption_tone: Mapped[str] = mapped_column(String(40), default="")
    clip_length_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    post_day_of_week: Mapped[str] = mapped_column(String(10), default="")
    post_hour_local: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    candidate: Mapped[Candidate | None] = relationship(back_populates="threads_posts")
    comments: Mapped[list["ThreadsComment"]] = relationship(back_populates="post")
    metrics: Mapped[list["MetricSnapshot"]] = relationship(back_populates="post")


class ThreadsComment(Base):
    __tablename__ = "threads_comments"
    __table_args__ = (UniqueConstraint("comment_id", name="uq_comment_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    post_pk: Mapped[int] = mapped_column(ForeignKey("threads_posts.id"))
    comment_id: Mapped[str] = mapped_column(String(60))
    username: Mapped[str] = mapped_column(String(120), default="")
    text: Mapped[str] = mapped_column(Text, default="")
    commented_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    classification: Mapped[str] = mapped_column(String(40), default="")  # one of settings categories
    classification_rationale: Mapped[str] = mapped_column(Text, default="")
    risk_flags: Mapped[str] = mapped_column(Text, default="")  # comma-separated, e.g. duplicate_text
    eligible_for_reply: Mapped[bool] = mapped_column(Boolean, default=False)

    draft_reply: Mapped[str] = mapped_column(Text, default="")
    # queue statuses: pending (awaiting operator), posted, skipped, filtered
    reply_status: Mapped[str] = mapped_column(String(20), default="pending")
    reply_text_posted: Mapped[str] = mapped_column(Text, default="")
    reply_id: Mapped[str] = mapped_column(String(60), default="")
    replied_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    post: Mapped[ThreadsPost] = relationship(back_populates="comments")


class MetricSnapshot(Base):
    __tablename__ = "metric_snapshots"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    post_pk: Mapped[int] = mapped_column(ForeignKey("threads_posts.id"))
    captured_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    views: Mapped[int | None] = mapped_column(Integer, nullable=True)
    likes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    replies: Mapped[int | None] = mapped_column(Integer, nullable=True)
    reposts: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quotes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    shares: Mapped[int | None] = mapped_column(Integer, nullable=True)

    post: Mapped[ThreadsPost] = relationship(back_populates="metrics")
