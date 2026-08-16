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

    # Programming category for the channel mix (one slug from settings
    # ``categories.options``; empty = untagged). The rationale is the LLM's
    # one-line reason when auto-tagged; cleared when the operator overrides.
    category: Mapped[str] = mapped_column(String(30), default="")
    category_rationale: Mapped[str] = mapped_column(Text, default="")

    # Operator marker: this video has material for more than one clip. Keeps
    # the video pinned in the dashboard's "Selected to trim" bucket (even after
    # exports/posts) until the operator toggles it off.
    multi_clip_potential: Mapped[bool] = mapped_column(Boolean, default=False)
    # Set by the clip suggester rather than the operator, so the marker can be
    # shown as a suggestion and audited later. Cleared the moment the operator
    # touches the toggle themselves.
    multi_clip_auto: Mapped[bool] = mapped_column(Boolean, default=False)

    status: Mapped[str] = mapped_column(String(20), default=STATUS_NEW)
    approved_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    archived_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scrape_error: Mapped[str] = mapped_column(Text, default="")

    local_video_path: Mapped[str] = mapped_column(Text, default="")
    transcript_path: Mapped[str] = mapped_column(Text, default="")
    transcript_text: Mapped[str] = mapped_column(Text, default="")
    transcription_method: Mapped[str] = mapped_column(String(20), default="")  # captions | whisper | captions+whisper | "" (none)
    # Full-video Whisper word stream ([{word, start, end}] JSON sidecar),
    # written at archive time. Clip suggestions cut against it and exports
    # slice it by trim windows instead of re-running Whisper per clip.
    word_transcript_path: Mapped[str] = mapped_column(Text, default="")

    # Video-level caption seed written by the clip-suggestion pass. Deliberately
    # NOT copied into ``Cut.draft_caption`` — captions are drafted per cut from
    # the trimmed clip's own transcript. It survives as the "still the untouched
    # seed" comparison the export poll makes before autocaptioning.
    draft_caption: Mapped[str] = mapped_column(Text, default="")

    # Vision scoring: how engaging the FOOTAGE looks, judged from YouTube's
    # storyboard stills (0-1) plus which popularity traits were detected.
    visual_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    visual_traits: Mapped[str] = mapped_column(Text, default="")  # comma-separated
    visual_rationale: Mapped[str] = mapped_column(Text, default="")
    visual_scored_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    channel: Mapped[Channel] = relationship(back_populates="candidates")
    # A video can be trimmed into several cuts (each a distinct topic/segment).
    cuts: Mapped[list["Cut"]] = relationship(
        back_populates="candidate", cascade="all, delete-orphan",
        order_by="Cut.created_at",
    )
    threads_posts: Mapped[list["ThreadsPost"]] = relationship(back_populates="candidate")


class Cut(Base):
    """A trimmed clip cut from a source video. One video can yield several cuts
    (e.g. two different topics covered in the same broadcast). A cut is the unit
    that gets captioned, titled, and posted; each post links back to its cut.
    """

    __tablename__ = "cuts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_pk: Mapped[int] = mapped_column(ForeignKey("candidates.id"))

    # LLM-generated, human-readable title for this clip (editable).
    clip_title: Mapped[str] = mapped_column(Text, default="")
    # LLM-condensed 1-4 word label of clip_title, sized to fit the calendar's
    # window slots. Generated alongside clip_title; falls back to clip_title
    # wherever it's blank (e.g. rows created before this field existed).
    calendar_name: Mapped[str] = mapped_column(Text, default="")
    # Per-cut caption draft. Written when the operator accepts a suggestion on
    # the Post step (or edits/queues/posts) — never auto-seeded, so it always
    # reflects text the operator has seen and approved.
    draft_caption: Mapped[str] = mapped_column(Text, default="")

    # Operator-chosen segments (JSON [{start, end}, ...]) and the exported
    # supercut file produced from them.
    trim_segments: Mapped[str] = mapped_column(Text, default="")
    trimmed_clip_path: Mapped[str] = mapped_column(Text, default="")
    # Background export job: "" | "exporting" | "failed". The Post step shows a
    # skeleton while exporting so Save can navigate immediately.
    export_status: Mapped[str] = mapped_column(String(20), default="")
    export_error: Mapped[str] = mapped_column(Text, default="")
    # Optional stylized-caption variant of the exported clip (Funnel font,
    # word-by-word highlight). Cleared on re-export; posting uses it only when
    # ``use_subtitles`` is on.
    subtitled_clip_path: Mapped[str] = mapped_column(Text, default="")
    use_subtitles: Mapped[bool] = mapped_column(Boolean, default=False)
    # Where burned-in captions sit on the frame: "bottom" (default) or "top".
    subs_position: Mapped[str] = mapped_column(String(10), default="bottom")
    # Whisper word stream of the trimmed clip (same pass that drives burned-in
    # captions). JSON sidecar path; Suggest caption / Copy transcript read it.
    # Cleared on re-export with the video files.
    clip_transcript_path: Mapped[str] = mapped_column(Text, default="")

    # Vertical (9:16) composite for Instagram Reels: hook text rendered at the
    # top of the frame, the 16:9 clip mid-frame on a branded background,
    # captions below. The hook is auto-drafted on the first compose and stays
    # operator-editable. Cleared on re-export with the video files.
    hook_text: Mapped[str] = mapped_column(Text, default="")
    # True once a hook has been drafted for this cut without being asked for.
    # The hook is optional, so an empty ``hook_text`` is a legitimate choice —
    # this flag is what stops the auto-draft from refilling a field the operator
    # deliberately cleared. The sparkle button ignores it.
    hook_autodrafted: Mapped[bool] = mapped_column(Boolean, default=False)
    vertical_clip_path: Mapped[str] = mapped_column(Text, default="")
    # Unused: reel caption is the Threads ``draft_caption``. Kept for existing DBs.
    ig_draft_caption: Mapped[str] = mapped_column(Text, default="")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    candidate: Mapped[Candidate] = relationship(back_populates="cuts")
    threads_posts: Mapped[list["ThreadsPost"]] = relationship(back_populates="cut")
    instagram_posts: Mapped[list["InstagramPost"]] = relationship(back_populates="cut")


class ThreadsPost(Base):
    __tablename__ = "threads_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_pk: Mapped[int | None] = mapped_column(ForeignKey("candidates.id"), nullable=True)
    # The specific cut this post was made from (null for imported Threads history).
    cut_pk: Mapped[int | None] = mapped_column(ForeignKey("cuts.id"), nullable=True)
    threads_media_id: Mapped[str] = mapped_column(String(60), default="")
    permalink: Mapped[str] = mapped_column(String(300), default="")
    caption: Mapped[str] = mapped_column(Text, default="")
    # Short 2-5 word calendar label, condensed from the caption. Only used when
    # there's no cut to hang a ``Cut.calendar_name`` off of (e.g. Threads
    # history imported from outside the app, which has no clip/title concept).
    calendar_name: Mapped[str] = mapped_column(Text, default="")
    clip_object_path: Mapped[str] = mapped_column(Text, default="")  # Supabase Storage object key
    clip_local_path: Mapped[str] = mapped_column(Text, default="")
    # draft | queued | publishing | published | failed
    # (legacy "scheduled" is migrated to "queued" on startup)
    status: Mapped[str] = mapped_column(String(20), default="draft")
    source: Mapped[str] = mapped_column(String(20), default="app")  # app | threads (imported history)
    error: Mapped[str] = mapped_column(Text, default="")
    # When set, a failed post has been acknowledged by the operator and no longer
    # surfaces in the Notifications "needs attention" list (kept for history).
    attention_dismissed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Legacy exact-time field; unused by the adaptive window scheduler.
    scheduled_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

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

    # Operator-set attribution for the first comment (a formal source citation,
    # e.g. 'Source: KXYZ (ABC), "Evening News", Springfield, IL, aired …').
    # Only ever filled by the operator — typed by hand or accepted from the
    # "Suggest a draft" LLM citation. Never auto-drafted: empty = this post
    # publishes without an attribution comment (the static
    # config/first_reply.yaml text, when enabled, is the fallback). What
    # actually got posted lands in first_reply_*.
    attribution_text: Mapped[str] = mapped_column(Text, default="")

    # Set when the operator clears an attribution that a post already carried —
    # kept as a record of that deliberate choice (publishing itself never drafts
    # attributions, so an empty ``attribution_text`` already means "no comment").
    attribution_skipped: Mapped[bool] = mapped_column(Boolean, default=False)

    # Auto first-reply (text reply under the published post). Set after publish
    # when config/first_reply.yaml is enabled; failure does not fail the post.
    first_reply_id: Mapped[str] = mapped_column(String(60), default="")
    first_reply_text: Mapped[str] = mapped_column(Text, default="")
    first_reply_error: Mapped[str] = mapped_column(Text, default="")
    first_reply_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    candidate: Mapped[Candidate | None] = relationship(back_populates="threads_posts")
    cut: Mapped["Cut | None"] = relationship(back_populates="threads_posts")
    comments: Mapped[list["ThreadsComment"]] = relationship(back_populates="post")
    metrics: Mapped[list["MetricSnapshot"]] = relationship(back_populates="post")
    instagram_post: Mapped["InstagramPost | None"] = relationship(
        back_populates="threads_post", uselist=False,
    )


class InstagramPost(Base):
    """An Instagram Reel queued/published alongside a ThreadsPost.

    The reel shares the paired Threads post's uploaded composite (same Supabase
    ``clip_object_path``) but carries its own caption and its own publish
    lifecycle: a reel failure never rolls back or blocks the Threads publish.
    """

    __tablename__ = "instagram_posts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    cut_pk: Mapped[int | None] = mapped_column(ForeignKey("cuts.id"), nullable=True)
    # The paired Threads post whose scheduler window fires this reel.
    threads_post_pk: Mapped[int | None] = mapped_column(
        ForeignKey("threads_posts.id"), nullable=True,
    )
    caption: Mapped[str] = mapped_column(Text, default="")
    clip_local_path: Mapped[str] = mapped_column(Text, default="")
    clip_object_path: Mapped[str] = mapped_column(Text, default="")  # Supabase object key
    ig_media_id: Mapped[str] = mapped_column(String(60), default="")
    permalink: Mapped[str] = mapped_column(String(300), default="")
    # draft | queued | publishing | published | failed
    status: Mapped[str] = mapped_column(String(20), default="draft")
    error: Mapped[str] = mapped_column(Text, default="")
    # Operator acknowledged a failure (drops out of needs-attention; kept for history).
    attention_dismissed_at: Mapped[dt.datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    published_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    cut: Mapped["Cut | None"] = relationship(back_populates="instagram_posts")
    threads_post: Mapped["ThreadsPost | None"] = relationship(back_populates="instagram_post")


class SchedulerState(Base):
    """Singleton row tracking scheduler progress across restarts.

    ``last_window_key`` is ``YYYY-MM-DD#N`` (ET date + 0-based window index) so
    each posting window is acted on at most once. ``last_publish_at`` enforces
    the spacing floor.
    """

    __tablename__ = "scheduler_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    last_window_key: Mapped[str] = mapped_column(String(40), default="")
    # Latest window the promo rotation has staged a post for. Cancelling a
    # staged promo deletes its row, so without this marker the next tick would
    # mint the promo straight back and the operator could never decline one.
    last_promo_window_key: Mapped[str] = mapped_column(String(40), default="")
    last_publish_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_metrics_poll_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
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

    classification: Mapped[str] = mapped_column(String(40), default="")  # legacy; unused
    classification_rationale: Mapped[str] = mapped_column(Text, default="")  # legacy; unused
    risk_flags: Mapped[str] = mapped_column(Text, default="")  # legacy; unused
    eligible_for_reply: Mapped[bool] = mapped_column(Boolean, default=False)  # legacy; unused

    draft_reply: Mapped[str] = mapped_column(Text, default="")  # legacy; unused
    # statuses: pending (unreplied), posted, skipped
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


class DraftProposal(Base):
    """Every AI-written draft and what the operator did with it.

    Covers both drafted texts: the Threads caption and the Instagram Reel hook
    (``kind``). They are the same shape — a draft, an edit, a final — and one
    table means "how often does the operator keep what we wrote" is a single
    query rather than a per-feature reimplementation.

    The loop only improves if a rejection is recorded as a rejection. Before
    this table neither draft was persisted: for captions, dismissing the
    proposal discarded it and ``record_post`` copied ``Cut.draft_caption`` —
    the operator's OWN text — into ``ThreadsPost.suggested_caption``, so every
    hand-written caption was filed as "the model got it right, no edits," the
    single least informative outcome. Hooks were worse: the draft overwrote
    ``Cut.hook_text`` directly, with no accept/dismiss step at all. Rows are
    written at draft time, before the operator can act, so the verdict is real.

    ``policy_version`` fingerprints what produced the draft (model, ceiling,
    length target, style guide, enabled rules). Without it a year of
    proposals is a blend of prompt regimes that can't be told apart, and
    "did that change help?" is unanswerable.
    """

    __tablename__ = "draft_proposals"

    KIND_CAPTION = "caption"
    KIND_HOOK = "hook"

    # Operator action on the draft, recorded when it happens. Hooks have no
    # accept/dismiss card, so they resolve straight from pending at post time.
    VERDICT_PENDING = "pending"          # drafted, not yet acted on
    VERDICT_ACCEPTED = "accepted"        # "Use this caption"
    VERDICT_DISMISSED = "dismissed"      # "Dismiss" — the rejection signal
    VERDICT_SUPERSEDED = "superseded"    # redrafted before acting on this one

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(20), default=KIND_CAPTION)
    cut_pk: Mapped[int] = mapped_column(ForeignKey("cuts.id"))
    # Which post this draft ended up behind, filled in when the post/reel is
    # recorded. Captions resolve against a Threads post, hooks against a reel.
    post_pk: Mapped[int | None] = mapped_column(ForeignKey("threads_posts.id"), nullable=True)
    ig_post_pk: Mapped[int | None] = mapped_column(ForeignKey("instagram_posts.id"), nullable=True)

    proposed: Mapped[str] = mapped_column(Text, default="")
    proposed_chars: Mapped[int] = mapped_column(Integer, default=0)
    proposed_words: Mapped[int] = mapped_column(Integer, default=0)

    verdict: Mapped[str] = mapped_column(String(20), default=VERDICT_PENDING)
    # What actually shipped, and how close it stayed to the draft (0..1).
    # Resolved at post time; the pair is the voice signal app/voice.py reads.
    final_text: Mapped[str] = mapped_column(Text, default="")
    final_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_words: Mapped[int | None] = mapped_column(Integer, nullable=True)
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)

    model: Mapped[str] = mapped_column(String(60), default="")
    policy_version: Mapped[str] = mapped_column(String(40), default="")
    # The ceiling and learned target in force for this draft, so a later
    # analysis can tell whether tightening them actually shortened the output.
    max_chars: Mapped[int | None] = mapped_column(Integer, nullable=True)
    target_words: Mapped[int | None] = mapped_column(Integer, nullable=True)
    voice_examples: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ClipProposal(Base):
    """Every clip the model proposed and what the operator actually cut.

    Separate from ``DraftProposal`` on purpose. That table resolves one text
    against another with a character-overlap ratio; this one resolves a set of
    time ranges against another set, and half its columns (chars, words, voice
    examples) are meaningless here. Its ``cut_pk`` is also non-nullable, while
    a proposed clip has no cut until the operator accepts it.

    A proposal is a PARTITION of one video: ``run_id`` groups the clips from a
    single suggestion pass, so the row set answers three different questions
    that a single blended score would hide:

    - Partition: did the model find the right number of stories? Compare
      ``clips_in_run`` against the cuts actually made on the candidate.
    - Compression: did it cut filler, or propose one continuous take? Compare
      ``proposed_segment_count`` / ``proposed_duration_s`` against the final.
      Most real clips join 2-4 segments, so a single-segment proposal is a
      miss even when its boundaries look reasonable.
    - Boundaries: ``iou`` over the union of intervals, plus SIGNED
      ``start_delta_s`` / ``end_delta_s``. The signed pair is the actionable
      half — "starts two seconds late, consistently" is a prompt fix, while an
      IoU of 0.7 says only that something was off.

    Rows are written the moment a pass runs, before the operator can act, so a
    rejection persists as a rejection instead of vanishing.
    """

    __tablename__ = "clip_proposals"

    VERDICT_PENDING = "pending"          # proposed, not yet acted on
    VERDICT_ACCEPTED = "accepted"        # loaded into a cut's segments
    VERDICT_DISMISSED = "dismissed"      # rejected outright — the real signal
    VERDICT_SUPERSEDED = "superseded"    # re-rolled before acting on this one

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_pk: Mapped[int] = mapped_column(ForeignKey("candidates.id"))
    # Groups the clips proposed by one pass, so partition accuracy is a query
    # rather than a reconstruction from timestamps.
    run_id: Mapped[str] = mapped_column(String(40), default="")
    clip_index: Mapped[int] = mapped_column(Integer, default=0)
    clips_in_run: Mapped[int] = mapped_column(Integer, default=0)

    # JSON [{start, end}, ...] in the same shape as ``Cut.trim_segments``.
    proposed_segments: Mapped[str] = mapped_column(Text, default="")
    proposed_segment_count: Mapped[int] = mapped_column(Integer, default=0)
    proposed_duration_s: Mapped[float] = mapped_column(Float, default=0.0)
    story: Mapped[str] = mapped_column(Text, default="")
    why: Mapped[str] = mapped_column(Text, default="")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)

    verdict: Mapped[str] = mapped_column(String(20), default=VERDICT_PENDING)
    # Filled when the proposal is accepted into a cut (existing or new).
    cut_pk: Mapped[int | None] = mapped_column(ForeignKey("cuts.id"), nullable=True)

    # What actually shipped, resolved at export time.
    final_segments: Mapped[str] = mapped_column(Text, default="")
    final_segment_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    final_duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    iou: Mapped[float | None] = mapped_column(Float, nullable=True)
    start_delta_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    end_delta_s: Mapped[float | None] = mapped_column(Float, nullable=True)

    model: Mapped[str] = mapped_column(String(60), default="")
    policy_version: Mapped[str] = mapped_column(String(40), default="")

    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    decided_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


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


class AnalyticsDigest(Base):
    """The written performance digest — stored because it costs an LLM call.

    Singleton row. The Analytics page used to write a fresh digest every time
    the report was rebuilt, which spent real money on a page view and quietly
    reworded the analysis between visits. It's now written only when the
    operator asks for one, and read from here in between.

    ``post_count`` records what it was written from, so the page can say when
    the numbers have moved on from what the text describes.
    """

    __tablename__ = "analytics_digests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    text: Mapped[str] = mapped_column(Text, default="")
    model: Mapped[str] = mapped_column(String(60), default="")
    post_count: Mapped[int] = mapped_column(Integer, default=0)
    generated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


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
