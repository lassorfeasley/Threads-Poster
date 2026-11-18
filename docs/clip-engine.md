# The clip engine

Written 2026-11-18. The clip engine suggests clips by watching the video. The original suggester only read the transcript. This page covers what the engine does, what it costs, how to switch it on and off, and how to test whether it is actually better.

## Why it exists

The original suggester (`clips.*` in `settings.yaml`) reads a video's transcript and guesses where the clips are. It can't see the picture or hear the delivery, and it never looks at what it proposed. Its suggestions failed in three ways:

1. **Weak openings.** Clips opened on an anchor, a reporter or generic b-roll. They should have opened on the person or the action that makes someone stop scrolling.
2. **Boring stretches.** Exposition and monologues dragged on, because a transcript can't tell a charged line from filler.
3. **Off-topic picks.** Some clips weren't really about climate. Others buried the climate angle under something else.

The engine does what an editor does:

1. **Log the video.** A model watches and listens to the source and writes a time-coded shot log: shot types, who is on screen, what is said, how energetic and visual each stretch is, how on-topic it is, and the standout moments. ffmpeg separately finds every shot cut and every pause.
2. **Draft about ten clips.** They come from different angles: a person, a picture, a line, a sound. They are also shaped by this account's archetypes (The Voice, The Scene, The Revelation, The Clash, The Pop Moment, The Witness). Each archetype has its own opening rule. Cold opens (a later moment first) are allowed.
3. **Snap every cut.** Each start and end lands on a word edge, a shot change or a pause, so no clip starts mid-word or flashes a frame of the next shot.
4. **Render every draft and watch it cold.** A critic sees each 360p render with no context, muted and with sound, and gets an extra close look at the first three seconds. It answers: does it stop the scroll, does it make sense on its own, where does it get slow, and is it on topic.
5. **Rank the best drafts head to head.** The ranking runs twice in opposite orders, against a few of this account's own hits and clips you labelled "not gripping".
6. **Revise the winner once.** The revision is kept only if a blind comparison prefers it in both presentation orders.

How this maps to the three complaints:

- **Openings:** exact cut times, a per-segment "opens on a picture or on a word" rule, the critic's close look at the first three seconds, and archetype rules such as "open on the person" for The Voice.
- **Boring stretches:** the critic marks where a clip gets slow and where the dead air is. The ranking compares drafts with your real hits.
- **Off-topic picks:** a draft is dropped only when the critic AND the shot log both say it's off topic. Allegory counts as on topic: the account's second-biggest post is a film scene that never says "climate". A doubtful topic stays on the card with a note.

If the engine fails (a provider outage, a render error, a run over its cost cap), or is over its daily budget, the video gets the original suggester's suggestions instead, as long as that one is within its own budget. If the engine legitimately finds nothing on topic, it writes nothing and your pending suggestions stay as they were.

## What you see in the dashboard

- **When it runs.** After a video is archived, the engine runs in the background. Pressing **Suggest** (or "Suggest a clip here" on a section) queues a run ahead of any unattended archive runs. A status line shows "Watching the video…" with the current stage, or "queued behind N". If the video's own archive run is still waiting its turn, Suggest stays pressable ("Suggest now" on the video page): pressing it moves that run to the front as yours, so it isn't held to the unattended daily budget.
- **Suggestion cards.** Each card has a playable preview of the actual cut, an archetype badge, and one critic line, e.g. "Stops the scroll: Bill Nye pulls off his safety glasses", plus "Gets slow at 0:21" when it sags. The old confidence badge is hidden; the critic's score is still stored as the confidence. "Not reviewed (critic unavailable)" means the critic call failed for that draft; it was kept rather than thrown away.
- **The preview is hidden** when a card was shortened around footage another clip already uses. The render no longer matches what "Use this clip" would load.
- **Restarts.** A dashboard restart, including the auto-reload when code changes, interrupts a run. The card then says "interrupted — press Suggest again". Unattended runs from the last 24 hours are resumed at startup, with two attempts at most per video.

## The four setups

| `clip_engine.setup` | Logs the video | Drafts the clips | Reviews and ranks | Needs |
|---|---|---|---|---|
| `transcript` | — | the original suggester, from the transcript | — | `ANTHROPIC_API_KEY` |
| `gemini` | Gemini, watching and listening | Gemini, watching the whole video | Gemini, watching each render | `GEMINI_API_KEY` |
| `hybrid` | Gemini | Claude, from the shot log plus stills of the strongest shots | Claude, from stills plus the words heard | both keys |
| `claude` | Claude, from stills every 2 s plus the Whisper words | Claude | Claude | `ANTHROPIC_API_KEY` |

`gemini` is the default for this workspace and the one the benchmark tests first. If a setup's key is missing, it quietly falls back to `transcript` and logs why.

### Switching setups

- Edit `clip_engine.setup` in `workspaces/climate/config/settings.yaml`. The next run uses it; no restart needed.
- To run a setup on one video without switching, use `python run.py clip-engine try <id> --setup hybrid`.
- For a Gemini key, create one at aistudio.google.com, put `GEMINI_API_KEY=...` in the root `.env`, then **restart the dashboard** (keys are read at startup). Use a paid-tier key: on the free tier, Google may use the data to improve its products.
- Gemini setups send 360p copies of the video and of each draft to Google's Gemini API. Small files go inline. Larger ones go through its Files API, which deletes them after 48 hours.
- Model ids, thinking levels, frame rates and the snapping windows are all under `clip_engine:` in `settings.yaml`. A blank model id means "use the default".

## What it costs

Rough cost per video for a typical 7–8 minute news segment. These are estimates, not measurements. Use `clip-engine try` (cost by stage) or a bench report (cost per video) for real numbers.

| Setup | Until 2026-12-31 | From 2027-01-01 | Reserved by the budget guard |
|---|---|---|---|
| `transcript` | about $0.05 | about $0.05 | — |
| `gemini` | about $0.40 | about $0.75 | $0.83 |
| `hybrid` | about $1.40 | about $1.50 | $1.55 |
| `claude` | about $1.50 | about $1.50 | $1.86 |

- **Gemini 3.8 Flash doubles in price on 2027-01-01.** The spend ledger already prices it at the 2027 rate, so until the end of 2026 the engine's reported spend is about twice your real Gemini bill. This is on purpose: the budget guards err high.
- **Longer sources cost more.** The guard reserves `estimate_usd_base + estimate_usd_per_minute × minutes` for each setup.

The budget settings, all under `clip_engine:`:

| Setting | Default | What it does |
|---|---|---|
| `daily_usd` | 5.00 | Daily cap for **unattended** runs (after archiving, and resumes). A run whose estimate would cross it uses the transcript suggester instead. Your own Suggest presses aren't capped. |
| `max_usd_per_video` | 1.50 | The optional revision round is skipped once a run has spent this much. |
| `hard_cap_usd` | 4.00 | A run that passes this stops and falls back to the transcript suggester. It is checked before every window of the shot log too, so a very long source (a four-hour hearing) stops near the cap rather than after logging the whole video. The windows already logged are kept: pressing Suggest again carries on from there, each press capped on its own. To let one long video through in a single run, raise this. |
| `max_source_seconds` | 3600 | Unattended whole-video runs on longer sources use the transcript suggester. Section passes still use the engine. |
| `estimate_usd_base`, `estimate_usd_per_minute` | per setup | The reservation above. |

Engine spend is booked to its own "engine" bucket in `data/llm_spend.json`, and the benchmark's to a "bench" bucket. Neither counts against `llm.daily_budget_usd`, so a busy engine day can't switch off caption checks, reruns or the digest.

If Gemini's daily quota runs out, or the project's prepaid credit does, the dashboard stops calling Gemini for 30 minutes. Runs in that time use the transcript suggester, and the suggestion page says why (for example "Gemini prepaid credits are depleted — add credit at https://ai.studio/projects"). After adding credit, press **Suggest**: a run you start yourself tries Gemini again straight away (if the credit still isn't there, it falls back and says so). Archive-time runs wait out the 30 minutes, or a dashboard restart. A daily quota can't be fixed that way, so it holds for every run until it clears.

## Trying it on one video

```bash
python run.py clip-engine try <id>                # candidate id or YouTube video id
python run.py clip-engine try <id> --setup hybrid
python run.py clip-engine try <id> --section 2:10 4:45   # like "Suggest a clip here"
python run.py clip-engine try <id> --log          # also save the picks as suggestions
```

`try` prints:

- every draft in the editor's order, with its archetype, length and score, the critic's verdict and fix, and why it was dropped;
- the top picks in play order, with their credits;
- cost by stage, tokens, timings, notes, and four orderings of the drafts (editor, cold critic, head-to-head ranking, final).

It writes nothing to the database unless you pass `--log`. `--log` replaces the video's pending suggestions, like Re-suggest. The full result is saved as JSON under `clip_engine/tries/`.

**Every video you `try` is added to the benchmark's `touched.json`**, so the test run never grades the engine on a video you tuned it against.

Two more commands:

- `python run.py clip-engine shotlog <id> [--perceiver gemini|claude] [--force]` builds a video's shot log, or shows the cached one, and prints it beat by beat. Logs are cached, so a second look is free; `--force` pays again.
- `python run.py clip-engine playbook` prints the archetypes and "what flops here" rules the engine runs with.
  - `--refresh` rebuilds them from this account's own first airings with one Claude call, and saves `config/clip_playbook.yaml`. The previous file is backed up under `clip_engine/playbooks/`.
  - The file is plain YAML; edit it by hand freely. Delete it to go back to the built-in default.

## Telling it what's wrong: advice and notes

The engine follows the playbook and your brand rules, but a single video can need something they don't say. Two ways to tell it, both from the dashboard:

- **"What's wrong?" on a suggestion card.** Pick one or more chips — Wrong language, No premise, Reporter monologue, Weak opening, Boring, Off topic, Bad cut points, Too long, Something else — and, if you like, a line of your own words. **Re-suggest with this** dismisses the card and runs the engine again on the whole video (or the same section, for a section suggestion) with that advice; **Just dismiss** only dismisses it.
- **Advice next to every Suggest button.** An optional line such as "open on the disaster footage; no interviews" for that one pass.

Advice outranks the playbook for that run: the editor, the reviser and the ranking all see it, and the critic checks each draft against it (a draft that ignores it loses 3 points). It's listed in the run's notes.

Every piece of advice is also kept as a **note** (`clip_engine/notes.jsonl`). When at least three new notes have piled up, one Claude call looks across your notes from the last 90 days for themes your brand rules don't already cover — a theme needs at least three notes on at least two videos — and proposes each as a rule. They appear under **Suggested rules** on the Brand & audience page with how many notes back them. **Add** makes one an ordinary brand rule (it then applies to every run); **Dismiss** drops it for good. Nothing changes until you add one.

## The clip handbook: house rules from your past edits

You've cut hundreds of clips, and they share habits no prompt spells out: what you open on, what you always cut, which languages you keep. The handbook reads those edits and proposes them as rules — in the same **Suggested rules** list as the ones drawn from your notes, and decided the same way.

**1. Build.**

```bash
python run.py clip-engine handbook build              # all your exported cuts
python run.py clip-engine handbook build --limit 40   # a cheap first look
```

- It studies every cut you exported (an exported file or a published post) whose source video and transcript are still available. Published clips count fully, exported-but-unposted ones half, and cuts that started from an accepted transcript suggestion 0.3 (their boundaries are partly the suggester's).
- **One video in five is held back** (fixed by the video id) and never studied, so the rules can be tested on edits they weren't learned from.
- For each cut, Gemini reads the video's shot log (made once per video and cached; a video already logged isn't even downloaded), the transcript, your segments in play order and the stretches you skipped. It describes what you chose, with times — what the clip opens on, what you kept and dropped, the structure, which languages are heard. It never guesses why.
- One Claude call then turns those descriptions into rules, each with its condition ("France 24 packages", "an interview in another language", or always), how many edits follow it and how many go the other way. A pattern seen in at least 5 edits that holds at least 70% of the time is a **rule**; weaker ones are **tendencies**, proposed with "usually".
- Candidates seen at least 3 times go into **Suggested rules** on the Brand & audience page (each one is tidied by the shared rule formatter, a small Claude call). A candidate you already dismissed, or that is already a brand rule, isn't asked again.
- Cost: roughly $0.02–0.05 a cut for the descriptions plus logging any video not yet logged (about $0.06–0.30 each), and about $1–2 for the rules. Descriptions are cached per cut, so a rebuild only pays for new cuts and the rules call. `--max-usd` (default `clip_engine.bench.max_usd`, 40) stops studying cuts past that spend; spend goes to the "bench" bucket. It restores offloaded sources only while free disk stays above `clip_engine.min_free_gb`, and deletes them again afterwards.

**2. Review.** Open **Analytics → Clip handbook** for the report: every candidate with its condition, support, exceptions, example cuts and where it stands (waiting for you, added, dismissed, not proposed). Then decide on the **Brand & audience** page. Nothing reaches the engine until you add it there; an added rule is a brand rule like any other, and you can edit or delete it on that page.

**3. Validate (optional, before adding).**

```bash
python run.py clip-engine handbook validate            # 15 held-out videos
```

It replays held-out videos through the engine twice — with your current rules, and with the current rules plus every candidate you haven't added or dismissed — and reports how often the top suggestion (and the top three) lands on a moment you actually cut ("moment hit"), for both. About $1.50 a video at the 2027 Gemini price; replays are cached, so a rerun continues. Fifteen videos is a small sample: a one- or two-video gap is noise. `handbook show` prints the report in the terminal.

**4. Refresh monthly.** Run `handbook build` again once a month or so: new edits are studied, the rest come from the cache, and only new patterns are proposed. The previous report is backed up under `clip_engine/handbook/backups/`.

## Metrics capture

The scheduler now takes the long-tail metrics snapshot by itself: every published post on its usual cadence (every 12 hours for 60 days, then weekly for a year), at most every 6 hours, and at most 200 Threads calls a run — the most overdue posts first; the rest are picked up next time. Before, this only happened when someone pressed **Pull latest metrics**, and views past a post's first six hours could go unrecorded for weeks. Those numbers are what taste examples, the playbook and the handbook are measured on.

- The last run's time is kept in the database (`app_tokens`, name `metrics_snapshot_at`), so a restart or a second scheduler doesn't run it again early.
- Settings under `analytics:` — `auto_snapshot_hours` (6; 0 turns it off) and `auto_snapshot_max_calls` (200).
- The button still takes every due snapshot at once.

## The benchmark: is the engine actually better?

The bench replays past videos through each setup ("arm") and compares the results. The decisive part is a **blind A/B test you judge yourself**: two clips cut from the same video, without knowing which suggester made which.

### 1. Iterate on a dev run

```bash
python run.py clipbench run --name dev-1 --seed dev --limit 10
python run.py clipbench pairs --name dev-1
```

Use dev runs, `clip-engine try` and `--candidate` hand-picked runs while you tune prompts and settings. All of them mark their videos as touched, and the report labels them "dev run — not the benchmark".

### 2. Run the test once, for the decision

```bash
python run.py clipbench run --name test-1          # --seed test is the default
python run.py clipbench pairs --name test-1
```

What a test run does:

- **Picks the videos.** It samples archived videos you exported at least one clip from, skipping every video in `touched.json`. It stratifies by source length (long sources oversampled) and by source kind.
- **Freezes the sample** in `videos.json`.
- **Replays each video as of its archive day.** The account context is the context of that day. Your later claims and dismissals are hidden, and the playbook's example titles are stripped. The critic's taste examples are frozen per video when the run is created, and a flop counts only if you labelled it "not gripping" before the video was archived.
- **Is resumable.** Rerun `clipbench run --name test-1` to continue: it keeps the run's own videos, arms and cap. `--arms` runs just those arms (adding any the run doesn't have yet), and `--max-usd` replaces the cap. Errored results are retried, finished ones are kept. One `clipbench run` of a run goes at a time; a second one refuses.
- **Stops after the video whose results pass `--max-usd`.** The default is 40, and spend is counted at ledger prices. A gemini+transcript run costs very roughly $0.80 per video at those prices, so the default cap may stop a 60-video run early; raise `--max-usd` if you want all of it. The report flags a run that was stopped.
- **Refuses to resume after prompts, settings or the playbook change.** Its fingerprint no longer matches. `--new-version` starts over under new arm labels instead of mixing versions, and it prints the spend so far and an estimate before it starts:
  - every arm runs again on all the run's videos;
  - what the run already spent still counts toward the cap, so raise `--max-usd` by about a full run's cost;
  - the pairs must be prepared and judged again: judgments of the old version don't carry over.

**Arms.** `--arms gemini,transcript` is the default (`clip_engine.bench.arms`).

- An arm is a setup, optionally with setting overrides, e.g. `gemini+editor_thinking=HIGH` or `gemini+editor_model=gemini-4-hydrogen`.
- A bare key with a role prefix (`editor_…`, `critic_…`, `perceiver_…`, `judge_…`) goes to the provider that plays that role in the setup. A dotted key is used as is (`clip_engine.snap.word_window=0.8`, or the short form `snap.word_window=0.8`). Engine arms take `clip_engine.*` keys; the transcript arm takes the `clips.*` keys its suggester reads.
- Overrides are checked, so an arm's label never claims a setting that didn't run. A typo, a setting the setup doesn't read, or one that can't vary per arm is refused.
- Taste examples and account context can only be reduced per arm:
  - `taste_hits` / `taste_flops` show fewer of the examples frozen per video;
  - `clips.learn_from_history=false` drops the account context, and `clips.context_examples` trims its shipped examples.

  `taste_lookback_days` can't vary per arm. `job_timeout_seconds` sets the arm's own time limit per video.
- Perception settings need a shot log of their own. They are `perception_fps`, `long_video_fps`, `media_resolution`, `perceiver_thinking` and `perception_window_seconds`, and for a Claude perceiver `frame_interval`, `frame_width`, `max_total_frames`, `max_frames_per_call` and `perceiver_effort`. Shot logs are cached per those settings, so an arm that changes one logs the video again with it (and pays for that) rather than reusing its baseline's log. The bench checks this when it reads the arm, and refuses the arm if a setting ever stopped being part of the cache key. A different `perceiver_model` always gets its own log.
- `transcript` means today's transcript suggester (labelled `transcript-v3`), not the one behind past suggestions.

**Other flags:**

- `--allow-no-words` also samples videos without a Whisper word stream. By default the bench skips them, and that skips several of the account's celebrity hits. Better: give them word streams first (see below).
- `--negatives K` adds K archived videos you never cut. You judge those on their own: "would you post this?" That measures false positives.

### 3. Judge

Open **Analytics → Clip bench** and pick the run. It shows you items one at a time, in an order that mixes arms and spaces out each video. There are three kinds; you aren't told which one you're looking at:

- **Arms pairs:** the engine's top pick against the transcript suggester's on the same video. Neither clip is yours, so this is the truly blind, **primary** test.
- **Ceiling pairs:** the engine's top pick against your own exported cut. You'll often recognise yours. Tick "I recognised one of these as mine" (key **M**) when you do, so the report can set those aside.
- **Solo clips** (with `--negatives`): would you post it?

All clips are rendered the same way (540p, no captions, no outro), and which side each one is on is random.

**Keys:**

- **1–4** choose A / B / Tie / Neither, or **1–2** for Would post / Wouldn't post.
- After a choice that has a loser, you can tag why it lost with **5–9**: weak opening, boring, off topic, bad boundaries, doesn't make sense. Those tags are how the report ties losses back to the three complaints.
- **Enter** saves without a reason; **Esc** changes your mind.

Pairs whose two clips are nearly identical are scored as ties automatically and never shown. Answer on what you'd rather post, not on which you think is the engine's.

### 4. Reveal and read the verdict

```bash
python run.py clipbench reveal --name test-1     # or the "Reveal results" button
python run.py clipbench report --name test-1
```

Results stay hidden until you reveal them, so they can't steer your judging. **Revealing closes judging for that run for good**, even while a `clipbench run` of it is still going. Judge everything first.

How to read the report:

- **The decision is the arms test.** It gives the engine's win rate against the transcript suggester among non-tied judgments, with a 95% confidence interval and a sign-test p-value. The engine is better when the whole interval sits above 50% (p < 0.05). An interval straddling 50% means no detectable difference yet.
- **The ceiling test is secondary.** It asks how often the engine's top pick wins or ties against your own cut. Videos where the engine proposed nothing count as losses, and "Neither" counts as not-a-win. It **passes** when the one-sided 95% lower bound (the "1-sided low" column) is at least 40%, **fails** when the whole printed 95% interval is below 40%, and is **inconclusive** in between. Read the line with recognised items excluded too.
- **Loss reasons per arm** show which complaint is still open.
- **Hit rates** compare each arm's picks with the clips you actually exported:
  - any-hit@1 and @3;
  - moment hits (the right moment at a different length);
  - opening match (does it start where you started);
  - **lift over chance** (on short videos, a random window often overlaps your cut).

  The hit rates are computed only on videos every arm finished; errors and empty results count as misses. The headline uses cuts you made from scratch, not cuts started from a suggestion. The report also shows these hits for the editor's own order and for the critic's, which tells you whether the critic and ranking add anything.
- **"Engine + fallback"** is what you'd actually get in the dashboard: the transcript suggester stands in wherever the engine failed.
- **"Anchored history"** is the suggestions you were actually shown back then. You saw those before cutting, so it is not comparable with the replay arms.

**How many judgments.** Judge every item; a 60-video run gives up to about 60 arms pairs.

- That is enough to detect a strong preference (about 70/30) but not a mild one. A 60/40 preference needs roughly 190 non-tied judgments.
- The ceiling verdict needs about 150 judged pairs to be decisive.
- If the result is inconclusive, start a **new** test run with a larger `--limit`. It draws fresh videos. Don't re-judge the same ones.

### Does the critic agree with you?

```bash
python run.py clipbench critic-check --roles gemini,claude --limit 40
```

This runs the cold critic over your posted clips that have a "gripping?" label (Analytics → Label clips). It reports how often the critic's verdict matches yours, separately for your "yes" and "no" clips, and how its score tracks views.

`docs/six-month-review.md` sets the bar: about 80% agreement on both before trusting a video model's scores. It costs about $0.01 a clip on Gemini.

`python run.py clipbench list` shows every run with its arms and judging progress.

## Videos without word streams

About 170 older videos have a transcript but no Whisper word stream. Their cut points are rougher, because rolling captions don't mark word edges. The bench also skips them by default, and that leaves out several of the celebrity hits.

```bash
python run.py retranscribe --missing-words --dry-run   # what it would do
python run.py retranscribe --missing-words --limit 20
```

- It adds word streams only. The existing transcript is never touched, and the method "captions" becomes "captions+whisper".
- Videos with an exported clip go first.
- Sources it has to bring back from storage are deleted again afterwards.
- It stops when free disk drops below `clip_engine.min_free_gb` (10 GB).
- Whisper runs on the CPU: allow tens of seconds to a few minutes per video.

## Where files live

Everything is files; the engine adds nothing to the database except the suggestions themselves.

| Path (under `workspaces/<workspace>/`) | What |
|---|---|
| `data/clip_engine/shotlogs/` | shot logs, one per video, perceiver, model and perception settings, plus each finished window of a long source (kept; they cost money to make) |
| `data/clip_engine/proposals/` | each suggestion's preview render and critic notes (kept 45 days) |
| `data/clip_engine/runs/` | one job record per video, plus the full result of each run |
| `data/clip_engine/proxies/`, `renders/`, `shotmaps/`, `taste/` | caches: 360p copies, draft renders, cut lists, this account's example clips (pruned after 14 days, at most daily) |
| `data/clip_engine/inflight/` | temporary hard links to sources while a run reads them |
| `data/clip_engine/tries/` | `clip-engine try` results |
| `data/clip_engine/playbooks/` | backups made by `clip-engine playbook --refresh` |
| `data/clip_engine/notes.jsonl` | your notes and advice, one per line |
| `data/clip_engine/rule_proposals.json` | suggested rules (from notes and the handbook) and what you decided |
| `data/clip_engine/handbook/edits/` | the handbook's description of each studied cut (cached) |
| `data/clip_engine/handbook/validation/`, `backups/` | handbook validation replays; previous reports |
| `data/clip_engine/gemini_files.json` | the Gemini upload cache |
| `data/clipbench/<run>/` | a bench run: `manifest.json`, `videos.json`, one result per arm and video, `pairs.json`, `judgments.jsonl`, `ledger.json`, `renders/` |
| `data/clipbench/touched.json` | videos the test seed must never draw |
| `data/clipbench/critic_check-<date>.json` | critic-check results |
| `config/clip_playbook.yaml` | the generated playbook (absent = built-in default) |
| `config/clip_handbook.yaml` | the handbook report (a report only: rules live in `brand.yaml` once you add them) |

`CLIP_ENGINE_DIR` and `CLIPBENCH_DIR` override the two data directories (the tests point them at temp dirs).

## Known limits

- **Gemini prices double on 2027-01-01.** Re-check the setup choice against the bench's cost line then.
- **Models won't name a face nobody names.** Credits ("0–12 s: Bill Nye, science educator") come only from names in the title, on screen or spoken. An unnamed celebrity is just "a man talks to camera" to every judge.
- **Sampling misses fast motion.** Gemini logs at 1 frame per second (0.5 past 20 minutes), and its critic sees 2 per second plus the first three seconds at 8. Claude setups see a still every 2 s when logging and every second when reviewing. A one-second event can slip between frames.
- **The bench flatters the engine somewhat.** The rubric and the default playbook were written from the same posts it replays. Stripping example titles and the dev/test split limit this; they can't remove it. Confirm on videos archived after 2026-11-17.
- **Claude setups see stills, not motion.** They hear only the Whisper words, not tone, music or natural sound.
- **Long sources.** The Gemini editor watches the whole source only up to 60 minutes. Past that it drafts from the shot log alone, and unattended runs use the transcript suggester.
- **Runs need this machine awake.** They happen inside the dashboard process, and a restart interrupts them.
