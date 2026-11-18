"""What the clip engine tells its models: the editorial judgment, in words.

Every rule about what makes a clip work for this account lives here, so the
rubric can be read, argued with and changed in one place. The functions only
assemble text; nothing here calls a model.

The rubric differs from the transcript suggester's (``llm.suggest_clips``) in
four ways that come straight from this account's own data:

- The opening depends on the kind of clip. A recognizable person delivering a
  charged line is the hook when they are on screen; opening on b-roll hides
  it. On-scene action is the hook when it is the footage. One global "open on
  the most striking image" rule gets the first kind wrong.
- Single takes are fine. About four in ten of the clips the operator shipped
  are one continuous take; telling the model "most good clips join 2-4
  segments" produced choppy, number-salad cuts.
- Cold opens are allowed (segments are in PLAY order): about one in six
  shipped clips opens on a later moment and then goes back for the setup.
- Drafts are judged by a critic that watches renders of them cold, so the
  editor is asked for real alternatives, not three versions of one guess.

Two blind spots are closed explicitly. Models won't name a face, so every
judge is handed "credits" (who is on screen, from names shown, said, or in
the title) — otherwise Bill Nye reads as "a man talks to camera". And the
topic test admits allegory: the account's second-biggest post is a scene from
a film about a comet that never says "climate".

Three tests came from the first live run (an India artisans video whose
picks were two non-English interviews and a reporter monologue, none of
which told a stranger what was going on): every word heard must be
understood (English, or carried by an English voiceover — burned-in
subtitles don't count, since this audience never uses those clips), the
first five seconds must give the premise (what is happening, and where),
and a reporter or anchor talking to camera is never the clip. The shot log
records each beat's speech language and whether an English voiceover is
heard, so the editor can see the first one coming.

The operator can also steer one pass in plain words (``advice``). It goes
into every prompt that chooses (draft, revise, rank, prefer) above the
playbook, and the cold critic checks the clip against it.
"""
from __future__ import annotations

# Shot vocabulary for the log. Small on purpose: the editor reads it, and a
# label nobody can apply consistently is noise.
SHOT_TYPES = (
    "aerial",           # drone / helicopter / satellite view
    "wide_scene",       # on-scene establishing or wide shot
    "action_closeup",   # people or things doing something, close
    "b_roll",           # illustrative footage under narration
    "interview",        # a person answering questions, on location or seated
    "talking_head",     # a person addressing the camera or an audience
    "anchor_desk",      # studio anchor or presenter
    "press_podium",     # press conference, hearing, lectern
    "crowd",            # protest, rally, audience
    "graphic",          # chart, map, full-screen text, animation
    "archival",         # old footage
    "text_card",        # slate, title card, logo
    "other",
)

LENSES = ("person", "picture", "line", "sound")

PROMINENCE = ("famous", "public_figure", "expert", "local", "unknown")

# What language the words in a beat are in. "none" = nobody speaks.
SPEECH_LANGUAGES = ("english", "other", "none")

ADVICE_HEADER = "THE OPERATOR'S ADVICE FOR THIS VIDEO — follow it; it outranks the playbook:"
_ADVICE_CHARS = 1500


def clean_advice(advice) -> str:
    """The operator's advice as one tidy line (whitespace collapsed, capped)."""
    text = " ".join(str(advice or "").split())
    return text[:_ADVICE_CHARS].rstrip()


def advice_block(advice) -> str:
    """The advice section for a system prompt, or "" when there is none."""
    text = clean_advice(advice)
    return f"{ADVICE_HEADER}\n{text}\n" if text else ""


# The three tests every judge applies (the critic as fields, the ranking and
# revision judges in their instructions).
THREE_TESTS = (
    "whether within about five seconds a stranger knows what is happening and "
    "where; whether every word heard is English or carried by an English "
    "voiceover (burned-in subtitles don't count); whether it is mostly one "
    "reporter or anchor talking to camera (those rank low)"
)


# ---- shared blocks ------------------------------------------------------------

def account_lines(brand: dict) -> str:
    """Who the account is, from brand.yaml. Blank fields are simply left out."""
    name = (brand.get("name") or "this account").strip()
    topic = (brand.get("topic") or "the channel's topic").strip()
    lines = [f"{name} is a social account about {topic}."]
    if brand.get("mission"):
        lines.append(f"Mission: {brand['mission']}")
    if brand.get("audience"):
        lines.append(f"Audience: {brand['audience']}")
    if brand.get("clip_guidance"):
        lines.append(f"The operator's editorial guidance: {brand['clip_guidance']}")
    return "\n".join(lines)


def _name(brand: dict) -> str:
    return (brand.get("name") or "this account").strip()


def _topic(brand: dict) -> str:
    return (brand.get("topic") or "the channel's topic").strip()


def _source_kind(brand: dict) -> str:
    return (brand.get("source_kind") or "news and documentary footage").strip()


def cutting_rules(*, min_s: float, max_s: float, max_segments: int,
                  has_words: bool = True) -> str:
    """The mechanical rules every cut follows — shared by drafting and revising."""
    if has_words:
        boundaries = (
            "- Times are seconds in the source video. Start a segment on the first "
            "word you want heard and end it right after the last, using the "
            "transcript entries' times (they sit on word boundaries).\n"
        )
    else:
        boundaries = (
            "- Times are seconds in the source video. The transcript here is "
            "rolling captions, so its times are approximate: start a segment just "
            "before the first words you want heard and end it just after the "
            "last.\n"
        )
    return (
        "CUTTING\n"
        "- Single takes are fine; many of this account's best clips are one "
        "continuous take. Join segments only when cutting makes the clip tighter.\n"
        "- List segments in PLAY order. A cold open (a later moment first, then "
        "the setup) is allowed when it makes the stronger opening. Never use the "
        "same footage twice in one clip.\n"
        f"- Each clip runs {min_s:g}-{max_s:g} seconds in total, in at most "
        f"{max_segments} segments, each at least 1.5 seconds.\n"
        + boundaries
        + "- Mark each segment's lead: \"word\" when it opens on speech, "
        "\"picture\" when it opens on a shot. A picture-led segment starts exactly "
        "on one of the shot changes listed in that beat's `cuts`.\n"
        "- Never use material inside used_ranges.\n"
    )


ACCOUNT_CONTEXT_GUIDE = (
    "LEARN FROM THIS ACCOUNT (account_context in the payload, when present)\n"
    "- shipped_examples: clips this operator posted, as the words heard, "
    "best-performing first. Match their density and pace; the words are only "
    "half of why they worked.\n"
    "- rejected_here: suggestions already dismissed on THIS video. Do not "
    "propose those moments again unless you fix the stated reason.\n"
    "- rejected_elsewhere: recent dismissals on other videos and why. Don't "
    "repeat those mistakes.\n"
    "- calibration: measured biases in past suggestions. Correct for them.\n"
    "- recently_posted: what the account posted lately. Don't retell a story "
    "already told unless this footage moves it forward.\n"
)


# ---- perception: the shot log ------------------------------------------------

def perceiver_system(brand: dict, *, mode: str, frame_interval: float = 2.0) -> str:
    """Logging a source video into beats. ``mode`` is "video" (the model
    watches and hears the file) or "frames" (stills + transcript only)."""
    topic = _topic(brand)
    if mode == "video":
        intro = (
            f"You are the assistant editor for {_name(brand)}, a social account "
            f"about {topic}. An editor who cannot watch this video will pick clips "
            "from your log, so log it precisely. Watch and listen to the whole "
            "video.\n\n"
        )
        clock = ("Give times as positions in this file, MM:SS, or MM:SS.s when a "
                 "change falls inside a second.")
        sound = ("- sound: what you hear besides speech: natural sound, crowd, "
                 "sirens, music, laughter, applause, shouting, silence; \"\" if "
                 "only speech.\n")
        language = (
            "- speech_language: the language of the words spoken in the beat: "
            "english, other (any other language), or none (nobody speaks). "
            "Narration and voiceover count as speech.\n"
            "- english_voiceover: true when an English voiceover, narration or "
            "interpreter is heard over this beat (for example a translator over a "
            "non-English interview, or a narrator over footage); false otherwise. "
            "Burned-in subtitles or on-screen translation are NOT a voiceover.\n"
        )
    else:
        intro = (
            f"You are the assistant editor for {_name(brand)}, a social account "
            f"about {topic}. An editor who cannot watch this video will pick clips "
            "from your log, so log it precisely.\n"
            "You can't play the video. You get still frames sampled every "
            f"{frame_interval:g} seconds, each labelled with its time in the video, "
            "and the transcript lines spoken in this stretch with their times. "
            "Infer motion and continuity from consecutive frames.\n\n"
        )
        clock = ("Give times in seconds from the start of the video (the frame "
                 "labels and the transcript use the same clock), e.g. \"75.5\".")
        sound = ("- sound: you can't hear the audio; fill this only from transcript "
                 "marks such as [music] or [applause], else \"\".\n")
        language = (
            "- speech_language: the language of the words spoken in the beat, "
            "judged from the transcript lines for its stretch: english, other (the "
            "words are in another language), or none (no speech in the beat).\n"
            "- english_voiceover: true when the transcript shows English narration "
            "or an interpreter speaking over someone whose own words are in another "
            "language, or a narrator over footage; false otherwise. Subtitles or "
            "translated captions visible in the frames are NOT a voiceover.\n"
        )
    return (
        intro
        + "Log it as BEATS, in order, covering the video from start to end. Start "
        "a new beat whenever the shot type, the speaker or the subject changes; "
        "most beats run 2-20 seconds. " + clock + "\n\n"
        "For each beat:\n"
        f"- shot: one of {', '.join(SHOT_TYPES)}.\n"
        "- on_screen: what is literally visible, in concrete nouns and verbs "
        "(\"firefighters drag a hose past a burning house\"), not adjectives. When "
        "the picture cuts inside the beat, list each shot with its start: "
        "\"00:12.5 aerial of the flooded town; 00:15.0 a man wading with a dog\".\n"
        "- face: is the speaker's face on camera? none, small (visible but "
        "distant or partial) or close.\n"
        "- speaker: who is talking. A name when one is shown on screen, said "
        "aloud, or given in the title or description; otherwise a role (anchor, "
        "reporter, narrator, official, scientist, witness, protester, crowd); "
        "\"none\" when nobody speaks.\n"
        "- line: the most important words spoken in the beat, close to verbatim, "
        "at most 25 words; \"\" if none.\n"
        + sound
        + language
        + "- energy 1-5: how much is happening and how charged the delivery is. "
        "1 = a static reading; 5 = shouting, tears, chaos, or a punchline "
        "landing.\n"
        "- visual 1-5: how hard the PICTURE alone would stop a stranger scrolling "
        "with the sound off. 1 = a routine desk read, slate or graphic; 5 = "
        "something astonishing in motion. A recognizable person mid-outburst can "
        "be a 5.\n"
        f"- topic 0-2: 2 = {topic} is central to this beat; 1 = mentioned; "
        "0 = not about it. If the title or description frames fiction, comedy or "
        f"satire as being about {topic}, score beats by their role in that "
        "allegory.\n"
        "- note: production furniture worth knowing: \"anchor toss\", "
        "\"sign-off\", \"ad\", \"slate\", \"text-heavy lower third\", "
        "\"dissolve\", \"repeat of earlier footage\"; \"\" otherwise.\n\n"
        "Also give:\n"
        "- people: everyone who appears or speaks, with prominence: famous (a "
        "household name), public_figure (politician, celebrity, executive), "
        "expert, local (an ordinary person or witness), unknown. Name people only "
        "from names shown on screen, said aloud, or given in the title or "
        "description, never from a face alone; evidence says which.\n"
        "- structure: one line mapping the video's parts with times, e.g. "
        "\"00:00-00:12 anchor intro; 00:12-02:41 field report; 02:41 sign-off\".\n"
        "- highlights: up to 12 of the strongest moments of any kind (a striking "
        "image, a quotable line, a telling sound, a recognizable person), at "
        "least 4 of them visual when the footage allows, each with its start, "
        "end and what it is.\n"
        "- format: what kind of video this is (local news package, cable panel, "
        "documentary, speech, hearing, talk show, interview, explainer...).\n"
        "- summary: two sentences on what the video is.\n\n"
        "Times will be used to cut the video, so be exact."
    )


def perceiver_user(*, title: str, description: str, duration: float,
                   part: tuple[int, int] | None = None,
                   span: tuple[float, float] | None = None) -> str:
    """The text that follows the video (or frames) in a logging request."""
    lines = [f"Video title: {title}"]
    if description:
        lines.append(f"Description: {description[:1500]}")
    lines.append(f"Full length: {_clock(duration)}.")
    if part and span:
        lines.append(
            f"This is part {part[0]} of {part[1]}: it covers {_clock(span[0])} to "
            f"{_clock(span[1])} of the full video. Give times as positions in THIS "
            "part, starting from 00:00.")
    return "\n".join(lines)


# ---- drafting ---------------------------------------------------------------

def editor_system(brand: dict, playbook_text: str, rules: str, *, n: int,
                  min_s: float, max_s: float, max_segments: int,
                  sees_video: bool, section: dict | None = None,
                  section_topic: str = "", has_words: bool = True,
                  has_keyframes: bool = False, advice: str = "") -> str:
    """The editor that drafts candidate clips from the shot log + transcript
    (and the video itself when ``sees_video``). ``advice`` is the operator's
    steer for this pass; it outranks the playbook."""
    topic = _topic(brand)
    if sees_video:
        eyes = (
            "The whole source video is attached: watch and listen to it. Its "
            "clock is the source clock the times below use. The shot log was "
            "written by an assistant who watched it; where your own eyes and ears "
            "disagree with it, trust them.\n"
        )
    else:
        eyes = (
            "You can't watch the video. The shot log was written by an assistant "
            "who watched it: it is your eyes and ears.\n"
            + ("Keyframes from the strongest shots follow the payload, labelled "
               "with their times.\n" if has_keyframes else "")
        )
    section_line = ""
    if section:
        section_line = (
            f"- You are cutting ONE SECTION of a longer video: "
            f"{section['start']:.0f}s to {section['end']:.0f}s"
            + (f" ({section_topic})" if section_topic else "")
            + ". Every segment must fall inside it.\n"
        )
    return (
        f"You are the editor of {_name(brand)}. {account_lines(brand)}\n"
        f"You cut short social clips from {_source_kind(brand)}.\n"
        + eyes + "\n"
        "WHAT WORKS FOR THIS ACCOUNT\n"
        + playbook_text.strip() + "\n"
        + (rules.strip() + "\n" if rules.strip() else "")
        + ("\n" + advice_block(advice) if clean_advice(advice) else "")
        + "\nYOUR JOB\n"
        f"Draft {n} candidate clips from this source. A critic will watch a render "
        "of every draft cold, the way a stranger scrolling past meets it, and the "
        "strongest three go to the operator. So give real choices:\n"
        "- Different moments, when the video has several worth posting.\n"
        "- When it has one great moment, cut it more than one way: a single take "
        "and a tightened version; a straight open and a cold open on the "
        "strongest beat.\n"
        "- Look through these lenses, and use each one that yields a clip you "
        "would actually post:\n"
        "  PERSON: a recognizable or compelling person delivering a charged line.\n"
        "  PICTURE: the most arresting footage, with the words that pay it off. A "
        "picture draft needs a person, stakes or a line too: generic spectacle "
        "under narration (fire, flood, crowds, aerials) does below average here.\n"
        "  LINE: the single most surprising or quotable statement.\n"
        "  SOUND: natural sound or a reaction that carries the moment.\n"
        "First list the distinct MOMENTS worth posting (up to 8), then draft from "
        "them: at most two drafts per moment, covering at least four moments when "
        "the video has that many. If the video truly holds less, draft fewer; "
        "never pad with clips you would not post.\n\n"
        "EVERY DRAFT\n"
        "1. Names its archetype from the playbook above and follows that "
        "archetype's opening rule. The opening depends on the archetype: when a "
        "recognizable person is the hook, open on them and their line, not on "
        "b-roll; when the hook is the footage, open on the action itself. Never "
        "open on routine presenter furniture: an anchor intro, a studio toss, a "
        "headline read, a slate, a graphic. When the presenter IS the hook (a "
        "famous host, a forecaster breaking script, a punchline), open on them.\n"
        "2. Hooks within two seconds, both with the sound off and with it on.\n"
        "3. Has no drone: at most one short context beat, directly before the "
        "payoff it sets up. When the viewer needs the premise (rule 8), the "
        "premise beat IS that one context beat. Cut preamble, restatement, a "
        "reporter re-explaining what a witness just said, sign-offs (\"back to "
        "you\", \"bye\"), and the next story's lead-in.\n"
        "4. Reads as one line of thought: no dangling \"he\", \"that\" or \"this\" "
        "whose referent was cut.\n"
        "5. Ends on the payoff: the consequence, the punchline, the quote.\n"
        f"6. Is about {topic} on its own terms: someone who sees only the clip "
        f"knows it is about {topic}, or it is a recognized {topic} allegory or "
        "culture moment whose link a one-line caption makes obvious. If "
        f"{topic} is incidental to a moment, it is not a clip for this account.\n"
        "7. LANGUAGE: the audience must understand every word heard. Speech in "
        "another language is usable only when an English voiceover carries it; "
        "burned-in subtitles are not enough. Check the shot log: a beat whose "
        "speech_language is \"other\" goes in only when its english_voiceover is "
        "true. Natural sound with no words is fine.\n"
        "8. PREMISE: within the first ~5 seconds a stranger knows what is "
        "happening and where: a picture plus a line that names the place or the "
        "stakes. A hook that leaves the viewer lost is not a hook.\n"
        "9. MONOLOGUE: a reporter or anchor speaking to camera is never the "
        "opening and never the clip; at most one short context beat (the "
        "premise beat, when one is needed). A famous presenter who IS the hook "
        "is the exception in rule 1.\n\n"
        + cutting_rules(min_s=min_s, max_s=max_s, max_segments=max_segments,
                        has_words=has_words)
        + section_line
        + "\n" + ACCOUNT_CONTEXT_GUIDE
        + "\nFOR EACH DRAFT ALSO WRITE\n"
        "- id: d1, d2, ... in the order you'd rank them. moment_id: the moment it "
        "draws on.\n"
        "- archetype: the playbook key it follows. lens: person, picture, line or "
        "sound.\n"
        "- story: one line on what the clip is.\n"
        "- story_key: a short slug for the underlying story; drafts telling the "
        "same story share it.\n"
        "- hook: what the viewer sees AND hears in the first two seconds.\n"
        "- payoff: the last line or beat.\n"
        "- why: one line on why it stops a scroll.\n"
        "- risk: the most likely reason it fails.\n"
        "- draft_caption: a plain, specific caption under 200 characters (the "
        "operator rewrites it).\n"
        "- challenges_rules: one line if the clip carries a claim the operator's "
        "rules reject, else \"\"."
    )


def revise_system(brand: dict, playbook_text: str, rules: str, *, min_s: float,
                  max_s: float, max_segments: int, sees_video: bool,
                  has_words: bool = True, advice: str = "") -> str:
    """One revision of one draft, from the critic's notes (and the
    operator's ``advice`` for this pass, which outranks the playbook)."""
    eyes = (
        "The whole source video is attached (source clock); watch the stretch "
        "around the draft.\n"
        if sees_video else
        "You can't watch the video; the shot log is your eyes and ears.\n"
    )
    return (
        f"You are the editor of {_name(brand)}. {account_lines(brand)}\n"
        "A critic watched a render of one of your drafts cold, as a stranger "
        "scrolling past would, and wrote notes. Make ONE revised cut that fixes "
        "what the notes identify. Keep what works.\n"
        + eyes
        + "The critic's times are seconds from the start of the CLIP. clip_map "
        "shows where each segment of the draft sits in clip time and in source "
        "time, so you can translate.\n\n"
        "WHAT WORKS FOR THIS ACCOUNT\n" + playbook_text.strip() + "\n"
        + (rules.strip() + "\n" if rules.strip() else "")
        + ("\n" + advice_block(advice) if clean_advice(advice) else "")
        + "\nTHE REVISED CUT MUST ALSO PASS THREE TESTS\n"
        "- Every word heard is understood: English, or carried by an English "
        "voiceover (burned-in subtitles are not enough; see each beat's "
        "speech_language and english_voiceover).\n"
        "- Within the first ~5 seconds a stranger knows what is happening and "
        "where.\n"
        "- It is not a reporter or anchor talking to camera; at most one short "
        "context beat of that.\n"
        + "\n" + cutting_rules(min_s=min_s, max_s=max_s, max_segments=max_segments,
                               has_words=has_words)
        + "\nIf the notes are wrong, or nothing you can cut from this footage "
        "would fix them, return the draft's segments unchanged and say so in "
        "`change`."
    )


# ---- critique ----------------------------------------------------------------

CREDITS_NOTE = (
    "The Credits line says who is on screen and when, the way a typical viewer "
    "of this account would recognize them: treat a person it names as "
    "recognized on sight, and judge the hook accordingly."
)

TOPIC_LABELS = ("central", "allegory", "incidental", "none")


def cold_system(brand: dict, *, mode: str, frame_interval: float = 1.0,
                advice: str = "") -> str:
    """A stranger meeting one rendered draft in a feed. Deliberately knows
    nothing about the account beyond who is on screen: this is the cold read.
    With ``advice`` (the operator's steer for this pass) it also checks the
    clip against it (``follows_advice`` / ``advice_note``)."""
    topic = _topic(brand)
    advice_text = clean_advice(advice)
    advice_test = ""
    if advice_text:
        advice_test = (
            "- follows_advice: the account's operator asked for this pass: "
            f"\"{advice_text}\". Does the clip follow that? advice_note: one line "
            "on how it does or doesn't.\n"
        )
    if mode == "video":
        how = ""
    else:
        how = (
            "You can't play it: you get frames every "
            f"{frame_interval:g} s, labelled with seconds from the clip's start, "
            "and the words heard with their times. Judge sound only from the words "
            "and marks like [music] or [applause].\n"
        )
    return (
        "You are scrolling a social feed and a short video starts playing, muted "
        "at first like most feeds; the post will carry burned-in captions of the "
        "spoken words. You know nothing about it beyond what you see and hear. "
        "React as a real viewer would, then explain precisely.\n"
        + how + CREDITS_NOTE + "\n\n"
        "- first_impression: literally what you see and hear in the first two "
        "seconds.\n"
        "- stops_scroll_muted: would those two seconds stop you with the sound "
        "OFF? muted_reason: why or why not.\n"
        "- stops_scroll: and with the sound on? stop_reason: why or why not.\n"
        "- person_at_start: is a recognizable or compelling person on screen in "
        "the first two seconds?\n"
        "- about: what the clip is about, in one line, from the clip alone.\n"
        "- makes_sense: could you follow it with no context? dangling: any "
        "reference you couldn't resolve (\"\" if none).\n"
        "- premise_clear: within about the first five seconds, do you know what "
        "is happening and where?\n"
        "- speech_understood: is every spoken word either in English or carried "
        "by an English voiceover? Speech in another language with only "
        "burned-in subtitles or captions does NOT count as understood. True "
        "when nobody speaks.\n"
        "- monologue: is the clip mostly one reporter or anchor talking to the "
        "camera?\n"
        + advice_test
        + "- bored_at: the second, counted from the start of the clip, at which "
        "you'd have swiped away; null if you'd watch to the end.\n"
        "- dead_air: stretches where nothing new happens, as start and end "
        "seconds from the clip's start.\n"
        "- ends_well: does it end on a payoff rather than mid-thought, on a "
        "sign-off, or on the next story's lead-in?\n"
        f"- topic: how the clip relates to {topic}: central (it is about it), "
        f"allegory (fiction, comedy or culture whose link to {topic} is "
        "obvious), incidental (mentioned in passing), or none. topic_note: one "
        "line.\n"
        "- person: who carries the clip, if anyone, and whether they're "
        "recognizable.\n"
        "- visual 1-5: how arresting the pictures are. audio 1-5: how compelling "
        "the sound and delivery are.\n"
        "- score 1-10: would this stop a stranger and hold them to the end? "
        "3 = scrolled past, 5 = forgettable, 7 = watched it all, 9 = would share.\n"
        "- fix: the single most valuable edit, concrete: \"start at 0:03 on the "
        "burning roof\", \"cut 0:12-0:18 where the reporter repeats the "
        "number\".\n\n"
        "Be harsh: most clips are forgettable."
    )


def rank_system(brand: dict, playbook_text: str, *, k: int, has_examples: bool,
                mode: str, frame_interval: float = 2.0, advice: str = "") -> str:
    """Head-to-head: which of these drafts would this account post first
    (the operator's ``advice`` for this pass outranks the playbook)."""
    topic = _topic(brand)
    if has_examples:
        examples = (
            "the examples show what this audience rewards and what it scrolls "
            "past. They are finished posts (burned-in captions, a "
            f"{_name(brand)} tab, sometimes an end card): ignore those production "
            "differences. Performance also depends on timing, so treat the "
            "examples as rough evidence of taste, not a template"
        )
    else:
        examples = "the playbook below describes what this audience rewards"
    how = "" if mode == "video" else (
        f"You can't play the clips: each comes as frames every {frame_interval:g} s "
        "(labelled with seconds from its start) and the words heard. Judge sound "
        "only from the words.\n"
    )
    return (
        f"You decide what {_name(brand)} posts. {account_lines(brand)}\n"
        f"You'll see {k} candidate clips cut from the same source video, "
        "each labelled Draft and a letter"
        + (", after examples of clips this account posted and how each did" if has_examples else "")
        + ".\n" + how + CREDITS_NOTE + "\n\n"
        "Rank every draft from the one you'd post first to the one you'd post "
        "last. Judge each the way a stranger scrolling past meets it: the first "
        "two seconds (muted, then with sound); whether it holds to the end; "
        "whether it makes sense with no context; whether it ends on a payoff; "
        f"whether {topic} is central to it; {THREE_TESTS}. Then weigh whether "
        f"it is the kind of clip that does well HERE: {examples}.\n\n"
        "WHAT WORKS FOR THIS ACCOUNT\n" + playbook_text.strip() + "\n\n"
        + (advice_block(advice) + "\n" if clean_advice(advice) else "")
        + "Write a one-line note for every draft first, then the ranking (every "
        "draft letter, best first), then best_reason: what makes your first "
        "choice beat your second."
    )


def prefer_system(brand: dict, *, mode: str, frame_interval: float = 1.0,
                  intent: str = "", advice: str = "") -> str:
    """Two cuts of the same clip: which one, or "same" (the operator's
    ``advice`` for this pass counts first)."""
    how = "" if mode == "video" else (
        f"You can't play them: each comes as frames every {frame_interval:g} s and "
        "the words heard.\n"
    )
    return (
        f"Two cuts of the same clip for {_name(brand)}, Version 1 and Version 2.\n"
        + how
        + (f"What the clip is going for: {intent}\n" if intent else "")
        + advice_block(advice)
        + CREDITS_NOTE + "\n"
        "Which would you post? Judge them the way a stranger scrolling past "
        "meets them: the first two seconds (muted, then with sound), whether it "
        "holds to the end, whether it makes sense with no context, whether it "
        f"ends on a payoff; {THREE_TESTS}. Answer choice \"1\" or \"2\", or "
        "\"same\" when neither is clearly better, with a one-line reason."
    )


# ---- playbook generation ------------------------------------------------------

def playbook_system(brand: dict) -> str:
    return (
        f"You are working out what has worked for {_name(brand)}. "
        f"{account_lines(brand)}\n"
        "Below are clips this account posted, with how each performed (its "
        "percentile among posts from the same weeks) and what it contains: title, "
        "caption, the words heard, and footage tags.\n\n"
        "Write a playbook of 4-7 ARCHETYPES that explain the hits, plus what "
        "flops. For each archetype give: key (one lowercase word), name, what it "
        "is, open (exactly what the first two seconds should show and say), avoid "
        "(what kills it), and 2-4 example post titles from the data that fit it, "
        "best first. Ground every claim in the data; where the data is thin, say "
        "so in evidence. flops lists the patterns that reliably do badly."
    )


def _clock(seconds: float) -> str:
    seconds = max(0.0, float(seconds or 0.0))
    m, s = divmod(int(round(seconds)), 60)
    return f"{m:02d}:{s:02d}"
