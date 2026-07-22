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
    country: Mapped[str] = mapped_column(String(60), default="")
    # local (single-market station) | national | international
    scope: Mapped[str] = mapped_column(String(20), default="local")
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
    # LLM-generated, human-readable title for the exported clip (editable).
    clip_title: Mapped[str] = mapped_column(Text, default="")

    # Vision scoring: how engaging the FOOTAGE looks, judged from YouTube's
    # storyboard stills (0-1) plus which popularity traits were detected.
    visual_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    visual_traits: Mapped[str] = mapped_column(Text, default="")  # comma-separated
    visual_rationale: Mapped[str] = mapped_column(Text, default="")
    visual_scored_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Trim step: operator-chosen segments (JSON [{start, end}, ...]) and the
    # exported supercut file produced from them.
    trim_segments: Mapped[str] = mapped_column(Text, default="")
    trimmed_clip_path: Mapped[str] = mapped_column(Text, default="")
    # Optional stylized-caption variant of the exported clip (Funnel font,
    # word-by-word highlight). Cleared on re-export; posting uses it only when
    # ``use_subtitles`` is on.
    subtitled_clip_path: Mapped[str] = mapped_column(Text, default="")
    use_subtitles: Mapped[bool] = mapped_column(Boolean, default=False)

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
    # draft | queued | publishing | published | failed
    # (legacy "scheduled" is migrated to "queued" on startup)
    status: Mapped[str] = mapped_column(String(20), default="draft")
    source: Mapped[str] = mapped_column(String(20), default="app")  # app | threads (imported history)
    error: Mapped[str] = mapped_column(Text, default="")
    # Legacy exact-time field; unused by the adaptive window scheduler.
    scheduled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Adaptive queue: breaking posts bypass window/hot deferral (spacing floor still applies).
    is_breaking: Mapped[bool] = mapped_column(Boolean, default=False)
    # How many times this queued post has been deferred because the last post was hot.
    defer_count: Mapped[int] = mapped_column(Integer, default=0)
    last_deferred_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Optional pin to a specific upcoming window key (``YYYY-MM-DD#N`` in scheduler TZ).
    # Lets the operator drag a queued post onto an open calendar slot; unpinned posts
    # fill remaining windows FIFO. Cleared on publish.
    pinned_window_key: Mapped[str] = mapped_column(String(40), default="")

    # The LLM's caption draft as it stood when this post was created. The final
    # ``caption`` is what the operator actually posted, so the diff between the
    # two is a durable record of the operator's voice (feeds app/voice.py).
    suggested_caption: Mapped[str] = mapped_column(Text, default="")

    # Ground-truth footage traits, annotated from the POSTED clip's own frames
    # (not the pre-download storyboard). This is what the learning loop trains
    # on: it covers uploads and reflects the post-trim footage that actually ran.
    footage_traits: Mapped[str] = mapped_column(Text, default="")  # comma-separated
    footage_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    footage_rationale: Mapped[str] = mapped_column(Text, default="")
    footage_scored_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Structured attributes for analytics slicing.
    caption_length: Mapped[int | None] = mapped_column(Integer, nullable=True)
    caption_has_question: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    caption_has_cta: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    caption_hashtag_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    caption_tone: Mapped[str] = mapped_column(String(40), default="")
    clip_length_seconds: Mapped[int | None] = mapped_column(Integer, nullable=True)
    post_day_of_week: Mapped[str] = mapped_column(String(10), default="")
    post_hour_local: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Auto first-reply (text reply under the published post). Set after publish
    # when config/first_reply.yaml is enabled; failure does not fail the post.
    first_reply_id: Mapped[str] = mapped_column(String(60), default="")
    first_reply_text: Mapped[str] = mapped_column(Text, default="")
    first_reply_error: Mapped[str] = mapped_column(Text, default="")
    first_reply_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    candidate: Mapped[Candidate | None] = relationship(back_populates="threads_posts")
    comments: Mapped[list["ThreadsComment"]] = relationship(back_populates="post")
    metrics: Mapped[list["MetricSnapshot"]] = relationship(back_populates="post")


class SchedulerState(Base):
    """Singleton row tracking adaptive-scheduler progress across restarts.

    ``last_window_key`` is ``YYYY-MM-DD#N`` (ET date + 0-based window index) so
    each posting window is acted on at most once. ``last_publish_at`` enforces
    the spacing floor. Hot-check fields power the Posts status panel.
    """

    __tablename__ = "scheduler_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    last_window_key: Mapped[str] = mapped_column(String(40), default="")
    last_publish_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_metrics_poll_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_hot_check_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_hot_result: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    last_hot_likes_delta: Mapped[int | None] = mapped_column(Integer, nullable=True)
    last_action: Mapped[str] = mapped_column(String(80), default="")
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


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


class Trait(Base):
    """Flat vocabulary of footage traits the tagger can attach to clips.

    Traits are observations only — no desirable/undesirable polarity. Judgment
    comes later from published-clip performance (``TraitWeight``). The ``kind``
    column is retained for schema compatibility but ignored.
    """

    __tablename__ = "traits"
    __table_args__ = (UniqueConstraint("name", name="uq_trait_name"),)

    KIND_NEUTRAL = "neutral"
    # Legacy constants kept so older rows / call sites don't break.
    KIND_DESIRABLE = "desirable"
    KIND_UNDESIRABLE = "undesirable"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(60))
    kind: Mapped[str] = mapped_column(String(20), default=KIND_NEUTRAL)
    description: Mapped[str] = mapped_column(Text, default="")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TraitWeight(Base):
    """Learned performance of a footage trait, derived from the operator's own
    published posts (their post-level ``footage_traits`` annotations).

    Verdicts are threshold-gated: a trait's ``status`` only becomes ``active``
    (allowed to influence ranking/guidance) once the account has enough total
    posts AND the trait itself has enough observations — see
    ``analytics.learn_trait_weights`` and the ``learning.*`` settings.
    Correlational only, recomputed from scratch on every learn pass."""

    __tablename__ = "trait_weights"
    __table_args__ = (UniqueConstraint("trait", "metric", name="uq_trait_metric"),)

    STATUS_COLLECTING = "collecting"    # not enough data; influences nothing
    STATUS_PROVISIONAL = "provisional"  # halfway to the gate; display only
    STATUS_ACTIVE = "active"            # past both gates; nudges ranking

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    trait: Mapped[str] = mapped_column(String(60))
    metric: Mapped[str] = mapped_column(String(20), default="views")
    n_posts: Mapped[int] = mapped_column(Integer, default=0)
    # Recency-weighted sample size (sum of decay weights; <= n_posts).
    effective_n: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    overall_avg: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Weighted medians (robust to one viral outlier, unlike the means above).
    median_metric: Mapped[float | None] = mapped_column(Float, nullable=True)
    baseline: Mapped[float | None] = mapped_column(Float, nullable=True)
    # Fractional lift vs. the account baseline: (median - baseline) / baseline.
    lift: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default=STATUS_COLLECTING)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TriageDecision(Base):
    """Log of every operator triage decision (approve/reject) with the signals
    that were visible at decision time. This is the training record for an
    eventual AI-assisted triage: it captures what the operator chose given the
    scores and traits shown. ``undone`` marks decisions reverted via Undo."""

    __tablename__ = "triage_decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_pk: Mapped[int] = mapped_column(ForeignKey("candidates.id"))
    video_id: Mapped[str] = mapped_column(String(20), default="")
    action: Mapped[str] = mapped_column(String(10))  # approve | reject
    relevance_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    visual_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    visual_traits: Mapped[str] = mapped_column(Text, default="")  # comma-separated
    undone: Mapped[bool] = mapped_column(Boolean, default=False)
    decided_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AppToken(Base):
    """Service credentials (e.g. the Threads OAuth token) stored in the shared
    DB so a headless runner (GitHub Actions / cron) can publish without the
    operator's laptop. ``value`` is the token payload as JSON text."""

    __tablename__ = "app_tokens"

    name: Mapped[str] = mapped_column(String(40), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class MonitorRun(Base):
    """Durable record of a monitor (discovery) pass, so the dashboard can show
    an accurate running/last-run state that survives page refreshes and server
    restarts. A pass runs in an in-process background thread, so any row left
    ``running`` after a restart is reconciled to ``interrupted`` on startup.
    """

    __tablename__ = "monitor_runs"

    STATUS_RUNNING = "running"
    STATUS_DONE = "done"
    STATUS_FAILED = "failed"
    STATUS_INTERRUPTED = "interrupted"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    status: Mapped[str] = mapped_column(String(20), default=STATUS_RUNNING)
    scope: Mapped[str] = mapped_column(String(60), default="")  # e.g. "since last check"
    lookback_days: Mapped[int | None] = mapped_column(Integer, nullable=True)
    channels_checked: Mapped[int] = mapped_column(Integer, default=0)
    candidates_stored: Mapped[int] = mapped_column(Integer, default=0)
    vision_scored: Mapped[int] = mapped_column(Integer, default=0)
    result: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    started_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


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
