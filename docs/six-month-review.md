# Six-month review: are video models ready? (due 2027-05-09)

Written 2026-11-09. On the due date the Notifications bell shows this review until you mark it done (`reminders.model_review_on` in `settings.yaml`).

## Why we waited

The long-term goal is a closed loop: an agent that picks clips, schedules them, learns from how they perform, and runs the account mostly on its own.

In November 2026 clip suggestions were the weakest part. They opened on dull frames, let exposition drone on, lost the story between segments, and buried the topic. A first fix improved them, but only modestly:

- a stronger model (Sonnet instead of Haiku);
- an editorial rubric in the prompt;
- the account's own history fed in (best-performing clips, dismissals with reasons, measured boundary bias, recent posts);
- richer suggestion cards in the UI.

The root cause is that the selector reads the transcript. What we want is clips that are **visually gripping**, with a story told in as little exposition as possible. Gripping is not the same as pretty. It means something happening, human stakes, scale or contrast, the unexpected, raw footage, and how the picture pairs with the line being spoken. In late 2026, models were weak at judging that from video, especially anything that depends on motion over a few seconds.

We also concluded:

- **Performance data is noisy.** At a few posts a day, only big effects can be measured.
- **One effect is clearly big: a home-run post lifts the posts after it.** That matters for measurement (adjust for momentum) and for strategy (aim for hits, and use the window after one).
- **Scheduling may be lower-hanging fruit than clipping.** Examples: reruns of proven winners, responding to momentum, and an agent that plans on the calendar inside rule guardrails.
- **Rules should come from what performed,** not from hand-written guesses, using simple models, priors, and some deliberate exploration.

So the plan was: keep clip selection assisted by you, keep rich data, and come back when the models are better.

## What has been collecting since November

| Data | Where | Why it matters |
|---|---|---|
| Your "gripping?" labels on posted clips | `content_labels`, filled from Analytics → **Label clips** | The test set for judging video models |
| Follower count, about daily | `account_snapshots` | Separates account growth from post quality; tells a lasting hit effect from a temporary boost |
| Every scheduler decision: winner's score, top runners-up, recent posts' views, queue size, followers | `placement_logs` | Answers "would a different choice have done better?" — the basis for a smarter scheduler |
| Weekly metrics on posts 60–365 days old | `metric_snapshots` | The long-term value of evergreen clips and reruns |
| Clip proposals, verdicts, dismiss reasons, boundary deltas, prompted edits | `clip_proposals`, `clip_revisions` | How the suggester's picks compare with what you shipped (`policy_version` separates prompt versions; `v2` is the November prompt) |
| Source videos and finished clips | Supabase Storage | Lets future models re-score everything from the raw footage |

## The checklist

### 1. Check the data actually accumulated

- [ ] `account_snapshots` has about one row a day. If not, check that the Threads token has the `threads_manage_insights` scope.
- [ ] `placement_logs` has a row for most published windows, with non-empty `alternatives`.
- [ ] Posts older than 60 days have weekly snapshots.
- [ ] At least 30 clips labelled, including both hits and flops. Fewer than that and step 2 isn't meaningful; spend 15 minutes on the labels page first.

### 2. Test video models against your labels (the gate for everything else)

- [ ] Pick the strongest available video-understanding model(s).
- [ ] For each labelled clip, give the model the actual clip (video, not frames, if it supports that) and ask: "Would someone scrolling stop on this? Is it visually gripping?"
- [ ] Measure agreement with your labels. Look separately at agreement on your "yes" clips and on your "no" clips.
- [ ] **Decision:** if it agrees about 80% of the time or more, on both "yes" and "no", it's ready to use for scoring. If not, set another six-month reminder and keep labelling.

### 3. Measure what actually performs, before building anything

Run these as one-off analyses (cheap; no product changes):

- [ ] **Momentum:** how big is the lift after a hit, and how fast does it fade? Use `placement_logs.context` and `account_snapshots`. Is it new followers who stay (a step up that holds) or a temporary boost (a spike that fades)?
- [ ] **Gripping vs. performance:** compare your labels, and the model's scores if step 2 passed, against views relative to a rolling baseline. Does gripping actually predict performance on this account? If not, find out what does (speakers, lines, topics) before building around visuals.
- [ ] **Reruns:** how does a second airing compare with the first, and does the gap between them matter?
- [ ] **Timing:** do the posting windows differ, after adjusting for momentum? Expect noise; only act on big differences.
- [ ] **Scheduler counterfactuals:** in `placement_logs`, did runners-up with higher scores (or particular score parts) go on to do better when they aired later?

### 4. Decide what to build (in this order, each useful on its own)

1. **Smarter reruns of proven winners.** The least noisy signal you have.
2. **Momentum response.** When a post breaks out (early views well above the recent baseline), pull the strongest queued clip or a related follow-up forward. During quiet stretches, hold strong clips back.
3. **An agent planner for the scheduler.** It runs a few times a day, reads the queue, momentum, the rerun pool and the news, and writes its decisions onto the calendar as pins with a one-line reason. Rule guardrails stay (active hours, spacing, expiry, promos). Pins keep it compatible with the deterministic placement engine that three runners share. Log each agent decision next to what the rule engine would have done.
4. **Visual-first clip selection, only if step 2 passed:**
   - a shot map at archive time (shot boundaries plus gripping scores on short frame sequences, together with what's being said over each shot);
   - measured limits (share of gripping footage, longest talking-head stretch, exposition before the payoff, opening shot), with thresholds fit from performance rather than hard-coded;
   - assemble clips around the strongest visual stretches with as few spoken lines as possible;
   - a critic pass over the assembled draft;
   - each card shows its numbers so dismissals point at something concrete.

### 5. Keep it self-improving and honest

- [ ] Rules and thresholds are settings re-fit from data on a schedule, not constants.
- [ ] Keep exploration: about 1 in 10 posts deliberately off-pattern, marked in `placement_logs.experiment`, so wrong rules can be discovered.
- [ ] Compare against a fair baseline (views relative to a rolling baseline, adjusted for momentum and follower count), never raw views.
- [ ] Keep only big, robust findings. At this volume, subtle effects are noise.

## Cost estimates from November (verify against the spend ledger)

| Per video | Short (~3 min) | Long (~1 hr) |
|---|---|---|
| Suggestions as of November (Sonnet + history) | ~4¢ | ~8–10¢ |
| + shot map (cheap vision model, once per video) | +1¢ | +8–12¢ |
| + critic over 4–6 drafts | +7–10¢ | +7–10¢ |

At about 10 videos a day, the full visual-first pipeline is roughly $1.50–2 a day. Using a cheaper model for the critic, or critiquing only the top 2 drafts, roughly halves the critic cost.

## Things deliberately not done yet

- **Hook-first ordering** (opening on a strong line from mid-video, then giving context). The suggester only produces segments in time order; the trim editor already allows reordering.
- **A one-off re-suggest of all videos waiting to be trimmed** with the new prompt (`run.py resuggest`). Offered, not built.
- **Storing news or trending context** with posts: hard to capture well, and noisy.
