# Local Climate News Clip Monitor

A personal, locally-run tool that watches ~190 local TV news YouTube channels for
climate-related segments, presents matches in a review dashboard, and — only after
you manually approve a video — downloads it and stores the file plus its transcript
and metadata. It then assists with publishing your own trimmed/captioned clips to
Threads, drafting Renewables.org replies to supportive commenters (each approved
before posting), and analyzing what's performing and why.

**Hard rule throughout: you approve every outbound action.** Nothing downloads,
posts, or replies automatically.

## Operating context

- Personal, single-user, local use — monitoring, archival, and your own
  Renewables.org Threads presence. No multi-user features, no hosting for others.
- Downloading runs on a residential IP by design (YouTube blocks datacenter IPs);
  downloads are sequential with randomized 8–25 s delays and only happen for
  videos you explicitly approved. No bulk scraping.
- Only clips **you trimmed and captioned yourself** get published; the tool does
  not redistribute raw footage. Full segments stay on your local disk.
- Threads publishing and replies are all operator-approved; the tool posts nothing
  on its own. Engagement is limited to your own posts and to supportive,
  good-faith commenters. Pacing caps keep reply volume human. The tool respects
  Threads/Meta platform policies and errs toward skipping when a reply might look
  spammy or tone-deaf.

## Setup

Requires Python 3.11+ and `ffmpeg` (`brew install ffmpeg`) for yt-dlp merging and
clip-length probing.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in keys (below)
```

Optional extras:

- **Supabase Postgres instead of local SQLite**: `pip install psycopg2-binary`
  and set `DATABASE_URL` in `.env`. SQLite (default, zero-config) is the right
  choice for day-to-day local development — every page hit stays under ~100ms.
  Use Supabase Postgres when you need a shared DB (headless runners, a future
  multi-user deploy). Remote DBs cost ~100ms per query; the app batches those
  round trips, but local SQLite is still much snappier for UI work. Supabase
  **Storage** (clip hosting for Threads) is independent of which database you
  use — keep `SUPABASE_URL` / `SUPABASE_SERVICE_KEY` either way.

### API keys (`.env`)

| Variable | What it is |
|---|---|
| `YOUTUBE_API_KEY` | YouTube Data API v3 key. Create at [console.cloud.google.com](https://console.cloud.google.com) → enable "YouTube Data API v3" → Credentials → API key. Used for discovery only. |
| `ANTHROPIC_API_KEY` | Claude API key ([console.anthropic.com](https://console.anthropic.com)) for relevance scoring, comment classification, drafts, digest. |
| `DATABASE_URL` | Optional. Empty = local SQLite at `data/app.db` (fast for local UI work). For Supabase Postgres use the connection string from Project Settings → Database. Keep Storage (`SUPABASE_*`) even when using SQLite. |
| `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | Supabase project (Settings → API). Used for the trimmed-clip bucket only. Create a **private** Storage bucket named `trimmed-clips` (or change `storage.trimmed_clip_bucket` in settings). |
| `THREADS_APP_ID`, `THREADS_APP_SECRET`, `THREADS_REDIRECT_URI` | Meta app for the Threads API (below). |

### Threads / Meta app + OAuth

1. Create an app at [developers.facebook.com](https://developers.facebook.com) and
   add the **Threads** use case with scopes: `threads_basic`,
   `threads_content_publish`, `threads_manage_replies`, `threads_read_replies`,
   `threads_manage_insights`.
2. Register a redirect URI (Meta requires HTTPS; `https://localhost/threads/callback`
   works — the redirect will fail to load in the browser, which is fine, you just
   copy the `code=` parameter from the address bar).
3. Add your Threads account as a tester (App roles) and accept the invite in
   Threads (Settings → Website permissions → Invites).
4. Put app ID/secret/redirect URI in `.env`, then open the dashboard's **Publish**
   page → "Authorize with Threads" → paste the code. The long-lived token (~60
   days) is stored at `data/threads_token.json` (gitignored) and auto-refreshes.

## Running

```bash
python run.py dashboard        # web UI at http://127.0.0.1:8321
python run.py dashboard --no-reload   # stable run (skip auto-reload on edits)
python run.py monitor          # one discovery pass (or use the dashboard button)
python run.py monitor --loop   # keep polling at the configured interval
python run.py score-visuals    # backfill vision scores for unscored candidates
python run.py annotate-posts   # backfill footage traits for published posts (from posted clips)
python run.py comments         # pull + classify comments on your own posts
python run.py metrics          # snapshot Threads metrics (time series)
python run.py digest           # print the analytics digest
python run.py cleanup          # apply retention (only if you set it; default keeps everything)
```

Typical always-on setup on a Mac mini / Pi: run `python run.py monitor --loop` and
`python run.py dashboard` (e.g. under `launchd`/`systemd`/`tmux`), and cron
`comments` + `metrics` a few times a day if you want those refreshed automatically
(they still never post anything).

## Workflow

The dashboard (sidebar: Dashboard / Archive / Posts / Engagement / Analytics /
Channels / Keywords / Traits) guides each video through four breadcrumbed steps:
**Review → Scrape & Transcribe → Trim → Post.**

1. **Monitor** polls each channel's uploads playlist via the YouTube Data API,
   keyword-filters title+description, LLM-scores each hit for genuine climate
   relevance (cuts "political climate" false positives), and stores candidates.
   Candidates that clear the relevance threshold can also be **trait-tagged**
   from YouTube storyboard stills (metadata-only — nothing downloaded): a flat
   vocabulary of labels with no good/bad score. The Dashboard lists new matches
   ranked by climate relevance (later nudged by learned trait verdicts once
   unlocked), with detected traits shown as badges.
2. **Review** (step 1): embedded player, matched keywords, score + rationale.
   Approve or Reject. Approve is the hard gate — nothing downloads before it.
3. **Scrape & Transcribe** (step 2): on approval the tool fetches the transcript
   from YouTube's own captions and downloads the full segment via yt-dlp to
   `data/videos/<CHANNEL>/<DATE>/`. Timestamped transcripts
   land in `data/transcripts/...` and in the DB. Idempotent — nothing is ever
   re-downloaded. The screen advances automatically when done.
4. **Trim** (step 3): in-browser player with a clickable timestamped transcript.
   Mark start/end points, add multiple segments, reorder them, and export — one
   segment is a simple trim; several become a supercut joined into one mp4
   (ffmpeg, frame-accurate re-encode) saved to `data/clips/`. An LLM-suggested
   highlight window is shown as a hint, clearly marked as a draft.
5. **Post** (step 4): preview the exported clip, generate an LLM caption
   suggestion (a draft — edit freely), and confirm. Caption drafts are
   **voice-matched**: past published captions (hand-written and heavily-edited
   ones weighted highest) are fed to the model as few-shot examples plus a
   cached distilled style guide, so drafts read like you (`voice.*` in
   settings). The draft is also stored alongside your final caption, so every
   edit you make becomes future voice signal. The clip uploads to a private
   Supabase bucket (Threads fetches video by signed URL), posts on your
   confirmation, and is retained as the canonical record linked to the post ID.
   At publish, the posted clip's frames are tagged with **ground-truth footage
   traits** (works for uploads too — no YouTube storyboard needed).
6. **Engagement**: "Sync comments" reads replies on your own posts only. An LLM
   classifies each (supportive / genuine question / neutral / hostile / bait /
   spam / off-topic) plus risk flags (duplicate text, political bait, sus).
   Only supportive + genuine questions get a drafted Renewables.org reply; you
   edit and approve each one. Hourly/daily caps and a per-post reply-fraction cap
   are enforced in code. Filtered comments are visible in a low-priority view.
7. **Analytics**: metric snapshots over time, per-post attribute tagging (topic,
   region, clip length, caption traits, day/time, **footage traits**), slice
   tables, and an LLM digest with clearly-labeled correlational hypotheses and
   small-sample caveats.    This is also the **self-improvement loop**, built
   collect-first / judge-later: every published post is annotated with footage
   traits from its actual posted clip, but trait *verdicts* only activate once
   two gates are met — `learning.min_total_posts` (default 100) posts overall
   and `learning.min_trait_posts` (default 20) observations of that trait.
   Verdicts compare each trait's recency-weighted **median views at a fixed
   post age** (48h by default, from metric snapshots) against the account
   baseline, so old posts' accumulated views and one viral outlier can't skew
   the read, and conclusions decay as your audience drifts
   (`learning.halflife_days`). Only *active* verdicts nudge candidate ranking
   (influence still capped by `ranking.trait_influence`); everything below the
   gates is display-only on the Traits page. Triage decisions (approve/reject
   plus the signals on screen) are logged to `triage_decisions` — the training
   record for eventually letting AI assist with, then take over, screening.
   All correlational, never presented as proven cause.

### Cost control

Every Anthropic call's token usage is logged to `data/llm_spend.json` and
estimated against `llm.pricing`. Vision scoring is gated three ways so spend
stays bounded: it only runs on candidates above `vision.min_relevance`, it stops
for the day once `llm.daily_budget_usd` (default $3) is reached, and it's capped
at `vision.max_per_run` candidates per monitor pass. Today's spend and the
budget are shown on the Analytics page and after each monitor run. Relevance
scoring, caption/reply drafts, and the digest are cheap and always run.

## Config

Everything lives in `config/` — no code changes needed:

- `config/channels.yaml` — the ~190-station seed list (call sign, network, market,
  region, URL). Handles and legacy `/user/`, `/c/`, `/channel/` URLs all resolve
  automatically. You can also add/remove/disable channels on the dashboard's
  **Channels** page.
- `config/keywords.yaml` — climate keyword list for the first-pass filter.
- The **trait vocabulary** is seeded from `vision.traits` in `settings.yaml` on
  first run, then managed on the dashboard's **Traits** page (add/edit/disable).
  Traits are observations only; performance verdicts come from published clips.
- `config/settings.yaml` — poll interval, score threshold, storage paths,
  retention (defaults to keep-everything), politeness delays,
  engagement categories + reply-eligibility + pacing caps + reply guidance,
  analytics cadence, **vision scoring** (`vision.*`: enable, model, traits,
  per-run cap), **ranking** (`ranking.*`: relevance/visual blend weights and how
  strongly learned trait weights nudge results), and the **LLM budget +
  pricing** (`llm.daily_budget_usd`, `llm.pricing`).

The `engagement.allow_other_users_posts` flag defaults to `false` and is
**high-risk / not recommended**; this build intentionally contains no code path
for other users' posts, so enabling it only logs a warning.

## Storage layout

| Artifact | Where | Retention |
|---|---|---|
| Full segment (raw yt-dlp download) | Local disk, `data/videos/<channel>/<date>/` | Keep forever by default; optional `cleanup` command |
| Trimmed clip (your edit) | Supabase Storage (private bucket, signed URLs) | Kept after posting as the record of what was published |
| Transcripts + all metadata | SQLite/Postgres (`data/app.db` by default) | Permanent, queryable |

The DB maintains the full chain per item: source video ID → local segment path →
trimmed-clip object → Threads post ID → time-series metrics.
