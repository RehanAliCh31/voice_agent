"""Speech over the bot: only the caller, meaning it, stops it.

The stand-in VAD scores by loudness, so the bot's own voice coming back
through the speaker is "speech" to it, as it is to Silero. What tells them
apart is the echo gate -- the page says how loud it is playing -- and the
words: none, a backchannel, or the bot's own.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.live_support import (
    RATE,
    Page,
    drained,
    open_session,
    quiet,
    services_for,
    settle,
    speak,
    tone,
)
from voicebot.agent.loop import HEARD_ONLY
from voicebot.llm.base import Reply

SENTENCE = (tone(900), quiet(700))
PLAYING = 0.9  # how loud the page says it plays the bot (RMS, full scale 1)
ECHO = 4500  # the bot coming back through the microphone: sure speech, and quiet
CALLER = 30000  # the caller speaking over it

LONG = (
    "Your checking account is open. Your savings account is open. Both are in good "
    "standing. Your card is active. Nothing is due this week. Anything else."
)


def _mix(*parts: np.ndarray) -> np.ndarray:
    size = max(len(p) for p in parts)
    total = np.zeros(size, dtype=np.int32)
    for part in parts:
        total[: len(part)] += part
    return np.clip(total, -32768, 32767).astype(np.int16)


def _cut_short(services) -> list[bool]:  # type: ignore[no-untyped-def]
    conversation = services.sessions.get("live-test").conversation
    return [bool(m.get(HEARD_ONLY)) for m in conversation.messages if m["role"] == "assistant"]


def _asked(page: Page, session, replies: int) -> str:  # type: ignore[no-untyped-def]
    """A question, answered: returns the reply's audio id once it is being played."""
    speak(session, *SENTENCE)
    assert page.wait(lambda p: len(p.of("reply")) == replies)
    assert settle(page, "agent_speaking")
    return str(page.of("reply")[-1]["audio_id"])


def test_its_own_voice_a_noisy_room_and_a_cough_never_stop_a_long_reply() -> None:
    runs = 50
    services = services_for(
        [(f"question {n}", "en") for n in range(runs)],
        [Reply(text=f"Answer {n}. {LONG}") for n in range(runs)],
        session={"max_turns": 1000, "max_minutes": 120},
    )
    page = Page(auto_finish=False, keep_audio=False)
    session = open_session(services, page)
    try:
        for n in range(runs):
            audio_id = _asked(page, session, n + 1)
            rng = np.random.default_rng(n)
            room = rng.normal(0, 400, 3 * RATE).astype(np.int16)
            echo = _mix(tone(3000, ECHO), room)
            # Its own voice coming back the whole time; a cough part way.
            speak(session, echo[:RATE], _mix(tone(250, CALLER), echo[RATE:]), far=PLAYING)
            assert page.wait(
                lambda p, a=audio_id: any(e["audio_id"] == a for e in p.of("audio_end"))
            ), n
            assert drained(session)  # the page says it finished after the audio was heard
            session.playback_done(audio_id)
            assert settle(page, "listening"), n
        assert not page.of("stop_audio"), "stopped without being interrupted"
        assert session.interruptions == 0 and session.false_barge_ins == 0
        assert session.echo_gated > 0  # the echo was there, and was taken for what it was
        assert not any(_cut_short(services))
    finally:
        session.close()
    assert session.threads_alive == 0


def test_its_own_words_coming_back_are_not_an_interruption() -> None:
    # A page that does not say how loud it plays: the gate cannot help, the words do.
    services = services_for(
        [("what is my balance", "en"), ("your savings account is open", "en")], [Reply(text=LONG)]
    )
    page = Page(auto_finish=False)
    page.stop_at = (1, 300.0)  # in "Your savings account is open."
    session = open_session(services, page)
    try:
        audio_id = _asked(page, session, 1)
        speak(session, tone(600, ECHO), quiet(700))
        assert page.wait(lambda p: bool(p.of("resume_audio")))
        assert settle(page, "agent_speaking")
        assert not page.of("discard_audio") and len(page.of("reply")) == 1
        # Not shown as something the caller said, either.
        assert [e["text"] for e in page.of("transcript")] == ["what is my balance"]
        assert session.interruptions == 0 and session.false_barge_ins == 1
        assert page.wait(lambda p: "audio_end" in p.kinds()) and drained(session)
        session.playback_done(audio_id)
        assert settle(page, "listening")
        assert not _cut_short(services)[0]
    finally:
        session.close()


def test_a_long_noise_with_no_words_stops_it_only_for_a_moment() -> None:
    services = services_for([("what is my balance", "en"), ("", "en")], [Reply(text=LONG)])
    page = Page(auto_finish=False)
    session = open_session(services, page)
    try:
        _asked(page, session, 1)
        # A door, the television: loud, long, and no words in it.
        speak(session, tone(2500, CALLER), quiet(700), far=PLAYING)
        assert page.wait(lambda p: bool(p.of("resume_audio")))
        assert not page.of("discard_audio") and session.interruptions == 0
        assert not _cut_short(services)[0]
    finally:
        session.close()


@pytest.mark.parametrize("where", ["start", "middle", "end"])
def test_the_caller_speaking_over_it_stops_it_at_once_wherever_it_is(where: str) -> None:
    services = services_for(
        [("what is my balance", "en"), ("stop, just my savings please", "en")],
        [Reply(text=LONG), Reply(text="Your savings are fine.")],
    )
    page = Page(auto_finish=False)
    page.stop_at = {"start": (0, 0.0), "middle": (2, 400.0), "end": (5, 10_000.0)}[where]
    session = open_session(services, page)
    try:
        _asked(page, session, 1)
        if where == "end":
            assert page.wait(lambda p: "audio_end" in p.kinds())
        echo = tone(1500, ECHO)
        speak(session, echo, far=PLAYING)  # its own voice, for a while
        speak(session, _mix(tone(1000, CALLER), echo), quiet(700), far=PLAYING)
        assert page.wait(lambda p: bool(p.of("stop_audio")))
        assert page.wait(lambda p: len(p.of("reply")) == 2)
        assert page.wait(lambda p: len(p.of("metrics")) == 2)
        assert page.of("metrics")[1]["barge_in_stop_ms"] <= 400
        assert page.of("discard_audio") and _cut_short(services)[0]
        assert session.interruptions == 1
    finally:
        session.close()


def test_a_pause_in_an_interruption_still_settles_the_reply_it_interrupted() -> None:
    services = services_for(
        [("what is my balance", "en"), ("stop", "en"), ("stop, my savings please", "en")],
        [Reply(text=LONG), Reply(text="Your savings are fine.")],
        asr_delay_s=0.5,
    )
    page = Page(auto_finish=False)
    session = open_session(services, page)
    try:
        _asked(page, session, 1)
        speak(session, tone(600, CALLER), quiet(700), far=PLAYING)
        assert settle(page, "thinking")
        # They go on before the first half is answered: one turn, the reply
        # it interrupted settled by all of it.
        speak(session, tone(800, CALLER), quiet(700))
        assert page.wait(lambda p: len(p.of("reply")) == 2, timeout=8)
        assert page.of("discard_audio") and _cut_short(services)[0]
        assert session.interruptions == 1
    finally:
        session.close()
