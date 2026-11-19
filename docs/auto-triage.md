# Automatic triage

Written 2026-11-19. This is stage 1 of automating triage. It covers what automatic triage does, the numbers behind its default, how to switch it on, and how it keeps itself honest. The code is in `app/auto_triage.py` and `app/positioning.py`, and the page is **Settings → Automatic triage**.

## What it does

Each pass rejects the waiting videos least worth your time, before they reach your queue. It judges by one of two signals:

- **Approval history.** The approve predictor (`app/triage_model.py`) gives each video its chance of approval, learned from your own approve/reject decisions. It reads the title, the description, the relevance score and the length. It's strong once there are hundreds of decisions, and has nothing to go on in a new workspace.
- **Positioning fit.** One cheap model call per video scores 0–1 how well the video suits the account, from the **Brand & audience** profile: mission, audience, positioning, the on/off-topic notes and your rules. It also gives the deciding factor in a few words. It needs no history, so it's what a new customer's workspace triages by. Until an approval model exists, it's also what the queue is ordered by.

Rejecting is all it does:

- It never approves, downloads or posts anything.
- Every rejection can be undone, one video at a time or a whole pass at once.

It only judges the videos the dashboard shows: relevance at or above `matching.score_threshold`. The rest are off topic already. On 2026-11-19 that was 88 of 5,381 waiting videos; the other 5,293 had been hidden since July.

A pass never touches:

- a video you've decided, including one you decided and then took back with Undo;
- a video it has already rejected once, even if you restored it since;
- a video from a first-party channel;
- a video without a current score (the pass scores it first: with the newest model, or against the current profile).

## The approval-history numbers

About 19 videos a day reach the dashboard. On 1,066 past decisions, each one was scored by a model trained only on the decisions before it, and graded out of sample:

| Cut-off | Share of shown videos it rejects | Approvals it would have cost |
|---|---|---|
| 1% | 20% (≈4 a day) | 4 of 110 (3.6%) |
| 2% | 32% (≈6 a day) | 5 of 110 (4.5%) |
| 3% | 39% (≈7 a day) | 11 of 110 (10%) |
| 5% | 50% (≈9 a day) | 16 of 110 (15%) |

The model ranks well, with an AUC of about 0.76. It's good at spotting junk and not at spotting winners: even among the videos it rates highest, you approve fewer than half. That's why stage 1 only rejects.

How high to go depends on what you want. If every approval counts, stay at 2% or below. If only the best few a day matter, 5% and above trade approvals you'd make for a shorter queue. These counts are small, so read them as ranges rather than promises. The spot-checks measure the same thing going forward.

## Positioning fit

Fit is only as good as the profile. With mission, audience and positioning empty, it judges the topic and little else. On a first try it scored a COP31 live stream you'd rejected at 0.85. Fill in **Positioning** on the Brand & audience page with what the account's clips should be known for, and the angle it takes. For example: on-the-ground footage of real places and people, not summits or studio talk.

Fit never sees your decisions, so grading it against them is a fair test. `python run.py auto-triage grade-fit` scores your past decisions (about $0.0005 a video, 300 by default) and fills in the page's fit table. Editing the profile makes every fit score stale. Stale scores are redone 200 at a time and never judged in the meantime.

The page's **Choosing the cut-off** table recomputes these numbers from the current model every time it retrains. It also shows how many of the lost approvals went on to a published post. A model trained before this feature only has the old single-split grade; press **retrain to grade every cut-off** once to fill in the table.

## Switching it on

1. On **Settings → Automatic triage**, tick **Reject automatically**, choose a signal and a cut-off, keep a **10%** spot-check, and press **Save**.
2. Press **Run now** to apply it to what's waiting straight away; the button shows how many it will reject. Otherwise the next monitor pass does it.
3. Answer the **Spot-checks** as they come in (about three a day at 10%). For each one, **Agree — reject** or **Approve & download**.

It runs at the end of every monitor pass: the dashboard's, `run.py monitor`, and the scheduled one on GitHub Actions. GitHub Actions runs the code on `main`, so scheduled passes start once this is merged there.

The switch, cut-off and spot-check share live in the shared database, not in `settings.yaml`, so every runner obeys the same switch the moment it's flipped.

## Keeping it honest

**Spot-checks.** A random share of its rejections, drawn by video id, waits for your verdict.

- Because the sample is random, your answers measure what it throws away.
- A video you rescue while browsing the full rejected list doesn't count: you picked that one by eye.
- Each verdict is an ordinary triage decision, flagged as a spot-check, so the model learns from it too.

**It pauses itself** when any of these happens:

- **Spot-checks show too many approvals.** At least 20 answers since it was last switched on, resumed or re-tuned show that you'd approve clearly more than 5% of what it rejects. "Clearly" means the low end of a one-sided 95% interval is above 5%. A single early approval can't trip it; 3 approvals in 20 answers does.
- **A sudden swing.** One pass would reject more than 80% of the waiting videos. That's what a broken or badly shifted signal looks like.

**It waits**, when judging by approval history, while the model has learned from fewer than 500 decisions or grades below 0.65 AUC.

A pause stays until you press **Resume**. Resuming, switching on, or changing the signal or the cut-off in force starts a fresh spot-check count.

**It never learns from itself.** Its calls are stored in `auto_triage_actions`, apart from `triage_decisions`, so the model only ever trains on what you decided.

The guardrails are under `auto_triage:` in `settings.yaml`.

## Undoing

- **One video.** Click **Approve** to send it to download, or **Back to queue** to put it back in triage undecided. Both work from the page's rejected list, and Approve and Un-reject also work on the video page, where an automatic rejection is labelled with the chance it had.
- **A whole pass.** **Undo** in the Passes table puts back every rejection from that pass that still stands. Videos you've approved, or agreed with on a spot-check, stay as you left them.

A video it rejected and you put back is never rejected automatically again.

## Commands

```bash
python run.py auto-triage status            # the switch, the model, each cut-off's cost, recent passes
python run.py auto-triage on --cutoff 0.02 --audit-rate 0.1
python run.py auto-triage on --signal fit --fit-cutoff 0.4
python run.py auto-triage score-fit         # positioning fit for waiting videos now
python run.py auto-triage grade-fit         # score past decisions; grade fit against them
python run.py auto-triage run               # a pass now
python run.py auto-triage undo <pass>
python run.py auto-triage resume            # clear a safety pause
python run.py auto-triage off
python run.py triage-train                   # retrain; prints the single-split and rolling grades
```

## Data

- `auto_triage_runs`: one row per pass, including passes that paused or waited. It records the signal, trigger, model, cut-off, how many were judged, rejected and set aside, and the undo time.
- `auto_triage_actions`: one row per automatic rejection. It records the signal, the approve chance and model, the fit and its reason, and the cut-off at the time; whether it was drawn for a spot-check and your verdict; and whether it still stands, with the reason if not (`approved`, `restored`, `pass undone`).
- `candidates.fit_score`, `fit_reason`, `fit_basis` (a fingerprint of the profile, prompt and model behind the score), `fit_scored_at`.
- `triage_decisions.audit`: marks your spot-check verdicts.
- `AppToken` row `auto_triage:<workspace>`: the switch, signal, both cut-offs, spot-check share and pause state.

## Automatic approval

Built 2026-11-19 (`app/auto_select.py`), and off until you switch it on in the **Automatic approval** section of the same page. It approves the day's most promising videos for you. The home machine downloads each one, and the clip engine watches it and drafts clips. It never posts.

- **Pool.** Waiting videos the dashboard shows, published in the last 2 days (adjustable). It skips any video you've decided, any video automatic triage has touched, and first-party channels.
- **Ranking.** Your approval history's chance when the model is fit to act on, otherwise positioning fit. A video under the floors (approve chance 5%, fit 0.2 by default) is never picked, even to fill the quota.
- **Pace.** Up to N a day (10 by default), opened up evenly through the day in the posting schedule's time zone. Each run takes the best of what has arrived so far.
- **Exploring.** 10% of picks go, at random, to a video below the top, and are flagged. That's how it learns where its ranking is wrong now that you're not deciding.
- **Where it runs.** On the machine that downloads (`ARCHIVE_ON_THIS_MACHINE` on), the dashboard runs a round every 10 minutes. It approves what the pace allows, then downloads every approved video one at a time through scrape's machine-wide pacing. That loop also picks up videos approved on the hosted dashboard, which used to wait for a restart.
- **Cost.** The clip engine's cap for unattended runs (`clip_engine.daily_usd`) is now $9. That covers 10 a day at the $0.83 reserved per run, which is priced at 2027 Gemini rates; the real bill until then is about half. Past the cap, the cheaper transcript suggester takes over.
- **Records.** Picks are logged in `auto_triage_actions` (stage `select`), never as your decisions. The page lists recent picks with what came of each: download state, clips drafted, cut, posted. "Drafted 0" on many picks is the signal to revisit the cheap look.

## What's next

Stage 1 judges metadata only, and metadata tops out: the approval model grades 0.76 AUC, and positioning fit 0.66 even with the positioning drafted from your own decisions. You decide by watching, so the real judgment has to come from the footage.

Automatic approval (above) now sends the top ~10 videos a day to the clip engine. Next:

1. Let the engine's critic gate what reaches you, so a video whose drafts don't clear the critic produces nothing.
2. Automatic render, captions and subtitles for the drafts that pass.
3. The inbox: the best 5 a day, with one-click approve, reject and caption edits.

### Shelved: the cheap look (2026-11-19)

The idea: a stage between metadata and the engine that reads a video's YouTube captions and a few frames (about 1¢) to catch obvious duds — no English, all studio talk, long live streams — before paying 40–75¢ for the engine to watch.

It's shelved because it's an optimisation, not something the pipeline needs. The daily budget already caps engine spend, so the most it could save is roughly $1–2 a day. It would also add automated YouTube traffic from the home IP that downloads depend on.

The first test showed the risk: fetching captions and storyboards for 132 past videos, 4 at a time, made YouTube bot-check the home IP and was stopped. It produced no result, because storyboard sheets are WebP and were being sent labelled as JPEG. That bug is fixed (`llm.image_media_type`), and it had also been quietly breaking storyboard trait tagging.

Come back to it if, after a couple of weeks of the daily budget, many engine watches yield nothing usable. Then test it safely:

- frames from YouTube's image servers (`i.ytimg.com/vi/<id>/hq1.jpg` … `hq3.jpg`), not yt-dlp storyboards;
- captions fetched one at a time, spaced like downloads;
- graded against the approval model on past decisions before building anything.
