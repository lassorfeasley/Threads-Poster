"""Part 2 engagement: read comments on the OPERATOR'S OWN posts, classify each
with the LLM, and draft Renewables.org replies ONLY for supportive /
genuine-question comments. Every reply requires operator approval in the
dashboard before posting.

Hard scope limits enforced in code:
  * Comments are only ever read from posts this tool published (ThreadsPost
    rows), i.e. the operator's own posts. There is no search or outreach
    surface at all; `engagement.allow_other_users_posts` is checked and, since
    no other-post code path exists, enabling it still does nothing beyond
    logging a warning. It exists to document the posture.
  * Replies below are drafts; posting happens only via post_approved_reply(),
    called from the dashboard on operator click, and pacing caps are enforced
    there — not left to discretion.
"""
from __future__ import annotations

import datetime as dt
import logging

from sqlalchemy import func, select

from .config import load_settings
from .llm import classify_comment, draft_reply
from .models import ThreadsComment, ThreadsPost, utcnow
from .threads_api import fetch_replies, publish_text_reply

log = logging.getLogger("engagement")


class PacingLimitError(RuntimeError):
    """Raised when posting a reply would exceed the hourly/daily caps."""


def _parse_ts(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00").replace("+0000", "+00:00"))
    except ValueError:
        return None


def sync_comments(session) -> dict:
    """Pull new comments on the operator's own published posts and classify them."""
    settings = load_settings()
    if settings.get("engagement.allow_other_users_posts", False):
        log.warning(
            "engagement.allow_other_users_posts is enabled in config, but this build "
            "intentionally has no code path for other users' posts. Ignoring."
        )

    categories = settings.get("engagement.categories", [])
    eligible_categories = set(settings.get("engagement.reply_eligible_categories", []))
    classify_model = settings.get("engagement.classify_model", "claude-haiku-4-5")
    draft_model = settings.get("engagement.draft_model", "claude-sonnet-5")
    guidance = settings.get("engagement.reply_guidance", "")
    max_fraction = settings.get("engagement.max_reply_fraction_per_post", 0.5)

    posts = session.execute(
        select(ThreadsPost).where(ThreadsPost.status == "published")
    ).scalars().all()

    new_comments = 0
    drafted = 0

    for post in posts:
        try:
            replies = fetch_replies(post.threads_media_id)
        except Exception as exc:
            log.warning("Could not fetch replies for post %s: %s", post.threads_media_id, exc)
            continue

        # Duplicate-text detection across everything we've seen (coordinated spam signal).
        seen_texts = {
            t.lower().strip()
            for (t,) in session.execute(select(ThreadsComment.text)).all()
        }

        for r in replies:
            comment_id = r.get("id", "")
            if not comment_id:
                continue
            exists = session.execute(
                select(ThreadsComment.id).where(ThreadsComment.comment_id == comment_id)
            ).scalar_one_or_none()
            if exists is not None:
                continue

            text = r.get("text", "") or ""
            comment = ThreadsComment(
                post_pk=post.id,
                comment_id=comment_id,
                username=r.get("username", ""),
                text=text,
                commented_at=_parse_ts(r.get("timestamp")),
            )
            new_comments += 1

            risk_flags: list[str] = []
            if text.lower().strip() in seen_texts and len(text.strip()) > 12:
                risk_flags.append("duplicate_or_coordinated_text")

            try:
                result = classify_comment(classify_model, categories, post.caption, text, comment.username)
                comment.classification = result["category"]
                comment.classification_rationale = result["rationale"]
                risk_flags.extend(result["risk_flags"])
            except Exception as exc:
                # When uncertain, skip rather than draft.
                comment.classification = "off_topic"
                comment.classification_rationale = f"(classification failed: {exc})"
                risk_flags.append("classification_failed")

            comment.risk_flags = ",".join(sorted(set(risk_flags)))
            comment.eligible_for_reply = (
                comment.classification in eligible_categories and not risk_flags
            )
            if not comment.eligible_for_reply:
                comment.reply_status = "filtered"
            session.add(comment)
            session.flush()

            if comment.eligible_for_reply:
                # Per-post fraction cap: don't reply to every single comment.
                total_on_post = session.execute(
                    select(func.count(ThreadsComment.id)).where(ThreadsComment.post_pk == post.id)
                ).scalar_one()
                already_drafted_or_posted = session.execute(
                    select(func.count(ThreadsComment.id)).where(
                        ThreadsComment.post_pk == post.id,
                        ThreadsComment.reply_status.in_(["pending", "posted"]),
                        ThreadsComment.draft_reply != "",
                    )
                ).scalar_one()
                if total_on_post > 2 and already_drafted_or_posted >= max(1, int(total_on_post * max_fraction)):
                    comment.reply_status = "skipped"
                    comment.classification_rationale += " | skipped: per-post reply fraction cap"
                    continue

                recent = [
                    t for (t,) in session.execute(
                        select(ThreadsComment.reply_text_posted)
                        .where(ThreadsComment.reply_status == "posted")
                        .order_by(ThreadsComment.replied_at.desc())
                        .limit(8)
                    ).all()
                ]
                try:
                    comment.draft_reply = draft_reply(
                        draft_model, guidance, post.caption, text, comment.username, recent
                    )
                    drafted += 1
                except Exception as exc:
                    log.warning("Draft failed for comment %s: %s", comment_id, exc)

    log.info("Comment sync: %d new, %d drafts", new_comments, drafted)
    return {"new_comments": new_comments, "drafts": drafted}


def redraft_comment(session, comment: ThreadsComment) -> str:
    """Regenerate one comment's draft reply using the current reply guidance.
    Lets the operator refresh existing queue drafts after editing the guidance."""
    settings = load_settings()
    draft_model = settings.get("engagement.draft_model", "claude-sonnet-5")
    guidance = settings.get("engagement.reply_guidance", "")
    recent = [
        t for (t,) in session.execute(
            select(ThreadsComment.reply_text_posted)
            .where(ThreadsComment.reply_status == "posted")
            .order_by(ThreadsComment.replied_at.desc())
            .limit(8)
        ).all()
    ]
    post = comment.post
    comment.draft_reply = draft_reply(
        draft_model, guidance, post.caption if post else "",
        comment.text, comment.username, recent,
    )
    session.flush()
    return comment.draft_reply


def check_pacing(session) -> None:
    """Raise PacingLimitError if another reply would exceed hourly/daily caps."""
    settings = load_settings()
    now = utcnow()
    hour_ago = now - dt.timedelta(hours=1)
    day_ago = now - dt.timedelta(days=1)

    per_hour = session.execute(
        select(func.count(ThreadsComment.id)).where(
            ThreadsComment.reply_status == "posted", ThreadsComment.replied_at >= hour_ago
        )
    ).scalar_one()
    per_day = session.execute(
        select(func.count(ThreadsComment.id)).where(
            ThreadsComment.reply_status == "posted", ThreadsComment.replied_at >= day_ago
        )
    ).scalar_one()

    max_hour = settings.get("engagement.max_replies_per_hour", 4)
    max_day = settings.get("engagement.max_replies_per_day", 12)
    if per_hour >= max_hour:
        raise PacingLimitError(f"Hourly reply cap reached ({per_hour}/{max_hour}). Try again later.")
    if per_day >= max_day:
        raise PacingLimitError(f"Daily reply cap reached ({per_day}/{max_day}). Try again tomorrow.")


def post_approved_reply(session, comment: ThreadsComment, final_text: str) -> None:
    """Post an operator-approved (possibly edited) reply. Enforces pacing caps."""
    check_pacing(session)
    result = publish_text_reply(final_text, comment.comment_id)
    comment.reply_status = "posted"
    comment.reply_text_posted = final_text
    comment.reply_id = result["media_id"]
    comment.replied_at = utcnow()
    session.flush()
    log.info("Posted reply to comment %s", comment.comment_id)
